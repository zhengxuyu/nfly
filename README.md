# nfly — 果蝇连接组当神经网络，玩任意 Gymnasium 游戏

把 Janelia **MaleCNS v1.0**（雄性果蝇脑 + 腹神经索，166,700 个已注释神经元，1052 万条 ≥3 突触的连接）转换成一个稀疏、符号受 Dale 定律约束的循环网络，用复眼接收游戏画面，用下行神经元和运动神经元输出动作。

```
观测 ──ObservationEncoder──▶ 输入神经元 ──ConnectomeRNN──▶ 读出神经元 ──ActionDecoder──▶ 动作
 (gym space)   复眼采样 / 线性投影         整脑稀疏动力学        下行 + 运动神经元      Categorical / Gaussian
```

## 分层结构（每层只依赖上面的层）

| 层 | 模块 | 职责 |
|---|---|---|
| 数据 | `nfly.connectome` | MaleCNS feather → `Connectome`（神经元表 + 边张量），子集，合成测试数据 |
| 模型 | `nfly.brain` | `ConnectomeRNN`：固定连通性与符号，每条边可学增益，每神经元可学 α 和偏置；刺激传播实验 |
| 接口 | `nfly.interface` | `ObservationEncoder` / `ActionDecoder` 基类，按 gym space 自动选择：图像 → 复眼 `RetinaEncoder`，向量 → `VectorEncoder`；Discrete → `DiscreteDecoder`，Box → `BoxDecoder` |
| 智能体 | `nfly.agent` | `FlyAgent = encoder → brain → decoder (+ value head)`，显式循环状态 h |
| 套件 | `nfly.suite` | `GameSuite` 抽象基类 + 注册表：`atari`、`classic`、`gym`（任意 Gymnasium id）；`play_episode` |
| 训练 | `nfly.rl` | A2C + 截断 BPTT，只依赖 `FlyAgent` 和 `VectorEnv` |

模型层不知道游戏的存在；套件层不知道模型的存在；两者只通过 gym 的 `observation_space` / `action_space` 相遇。

## 生物量 → 网络量

| 生物量（MaleCNS 字段） | 网络量 | 规则 |
|---|---|---|
| `bodyId`（有 `superclass` 的 body） | 节点 i，标量状态 h_i(t) | 无 superclass 的碎片、胶质不进网络 |
| 连接权重表 (body_pre, body_post, weight) | 稀疏权重 W[post, pre] | 默认保留 ≥3 突触的对 |
| 突触前的 `consensus_nt` | W 该列的符号，固定 | ACh/DA/5-HT/OA 为 +，GABA/Glu/His 为 −，unclear 记 + |
| `weight` | 权重幅值 | weight / post 的总输入突触数 |
| `superclass` ∈ *_sensory 等 | 输入层 flow = afferent | |
| `superclass` ∈ *_motor, *_efferent, *_endocrine | 输出层 flow = efferent | |
| 其余 | 隐层 flow = intrinsic | |
| `assignedOlHex1/2`（视叶柱状神经元的六边形坐标） | 视网膜坐标 | 光感受器取其下游柱的突触加权坐标；左眼看画面左半，右眼看右半 |
| `descending_neuron` + `*_motor` | 动作读出神经元 | LayerNorm → 线性头 |

动力学：

```
h[t+1] = (1 − α) ⊙ h[t] + α ⊙ min( ReLU( W h[t] + b + u[t] ), h_max )
W[post, pre] = sign(pre) · (syn / Σ_in syn) · exp(g_edge)        g_edge 可学，初始 0
```

静息偏置 b 初始为 0.1：光感受器只释放组胺（抑制性），如果所有神经元从 0 出发，抑制性输入永远无法被 ReLU 单元读到；给每个神经元一点基线活动后，光照就表现为下游活动的下降，与真实的 L1/L2 反应方向一致。整脑在增益 1.0 下基线活动稳定（约 98% 神经元有活动，均值 0.1），增益 5 会发散。

稀疏乘法用 gather + index_add 实现并带自定义反向传播，BPTT 每步只保存 (batch × N) 的状态，不保存 (batch × 边数) 的消息。

## 数据

三个文件公开在 Google Storage（CC-BY 4.0），不需要登录，放进 `data/`：

```bash
B=https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome
curl -o data/body-annotations.feather        $B/body-annotations-male-cns-v1.0-minconf-0.5.feather   # 14 MB
curl -o data/body-neurotransmitters.feather  $B/body-neurotransmitters-male-cns-v1.0.feather         # 43 MB
curl -o data/connectome-weights.feather      $B/connectome-weights-male-cns-v1.0-minconf-0.5.feather # 1.1 GB
```

权重表有 1.52 亿行，加载时用 pyarrow 分批过滤到已注释神经元，首次约 25 秒，之后缓存到 `data/cache/`。

## 用法

```bash
pip install -e ".[dev]"
pytest                                                            # 用合成的小 MaleCNS 跑全部测试，不需要下载
python scripts/demo_stimulate.py --class gustatory                # 刺激味觉神经元，看谁被激活
python scripts/play.py  --suite atari   --game pong               # 整脑玩 Pong（未训练）
python scripts/play.py  --suite classic --game cartpole           # 同一个模型换向量观测的游戏
python scripts/train_rl.py --suite atari --game pong --subset visual --envs 8 --updates 2000
```

```python
from nfly import load_malecns, select_subset, FlyAgent
from nfly.suite import get_suite, play_episode

conn = select_subset(load_malecns("data"), "visual")            # all | brain | visual | visual_small
env = get_suite("atari").make("breakout", seed=0)
agent = FlyAgent.build(conn, env.observation_space, env.action_space)
print(play_episode(agent, env).ret)
```

### 接入你自己的游戏

任何 Gymnasium 环境都能直接用：`get_suite("gym").make("LunarLander-v3")`。要打包成套件，只需实现两个方法：

```python
from nfly.suite import GameSuite, register

@register("mygames")
class MySuite(GameSuite):
    def games(self):                                   # 游戏 id 列表
        return ["level1", "level2"]
    def make(self, game, seed=None, render_mode=None, **kw):
        return MyEnv(game, render_mode=render_mode)    # 任何遵守 gym.Env 接口的对象
```

`make_vector`、`spaces` 等方法由基类提供。智能体只看 `observation_space` 和 `action_space`，不需要改任何模型代码。

### 换感官或肌肉

继承 `ObservationEncoder`（实现 `idx` 和 `encode`）或 `ActionDecoder`（实现 `distribution`），传给 `FlyAgent.build(..., encoder=..., decoder=...)`。

## 子集

| 名称 | 内容 | 神经元数 |
|---|---|---|
| all | 整个中枢神经系统 | 166,700 |
| brain | 去掉腹神经索 | ~146,000 |
| visual | 视叶 + 视觉投射 + 中央脑 + 下行神经元 | ~138,000 |
| visual_small | 视叶 + 视觉投射 + 下行神经元 | ~106,000 |
