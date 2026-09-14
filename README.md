# nfly

A fly connectome as a recurrent neural network that plays any Gymnasium game.

nfly turns the Janelia **MaleCNS v1.0** release (male *Drosophila* brain + ventral nerve cord:
166,700 annotated neurons, 10.5M connections with >= 3 synapses) into a sparse, sign-constrained
rate RNN. Game frames land on the compound eye, activity flows through the real wiring, and
descending / motor neurons are read out as actions.

```text
observation --ObservationEncoder--> input neurons --ConnectomeRNN--> readout neurons --ActionDecoder--> action
 (gym space)   retina / projection      whole-CNS sparse dynamics    descending + motor      Categorical / Gaussian
```

## Layers

Each layer depends only on the layers to its left.

| Layer | Package | Responsibility |
| --- | --- | --- |
| data | `nfly.connectome` | MaleCNS feather files -> `Connectome` (neuron table + edge tensors); named subsets; synthetic fixtures |
| model | `nfly.brain` | `ConnectomeRNN`: fixed wiring and signs, learnable per-edge gain, per-neuron leak and bias; stimulation experiments |
| interface | `nfly.interface` | `ObservationEncoder` / `ActionDecoder` base classes, chosen from the env's spaces: images -> `RetinaEncoder`, vectors -> `VectorEncoder`; Discrete -> `DiscreteDecoder`, Box -> `BoxDecoder` |
| agent | `nfly.agent` | `FlyAgent = encoder -> brain -> decoder (+ value head)`, a recurrent policy with explicit state `h` |
| suite | `nfly.suite` | `GameSuite` abstract base class + registry (`atari`, `classic`, `gym`); `play_episode` |
| training | `nfly.rl` | `rl.simple`: readable pure-PyTorch A2C and PPO; `rl.rllib`: Ray RLlib `FlyRLModule` + config builders for PPO / APPO / IMPALA |
| viewer | `nfly.viz` | browser viewer for any (policy, gym env) session: live frame, action probabilities, action timeline, pause / step / reset |

The model layer does not know games exist; the suite layer does not know the model exists. They
meet only through `observation_space` and `action_space`.

## Biology -> network

| Biology (MaleCNS field) | Network | Rule |
| --- | --- | --- |
| `bodyId` with a `superclass` | node i, scalar state h_i(t) | fragments and glia (no superclass) are excluded |
| weights table (body_pre, body_post, weight) | sparse entry W[post, pre] | pairs with >= 3 synapses by default (`min_syn`) |
| presynaptic `consensus_nt` | sign of that column, fixed (Dale's law) | ACh / DA / 5-HT / OA are +, GABA / Glu / His are -, unclear is + |
| `weight` (synapse count) | magnitude | weight / total input synapses of the postsynaptic neuron |
| `superclass` in `*_sensory`, `sensory_ascending`, `sensory_descending` | input layer, flow = afferent | external drive u(t) enters here |
| `superclass` in `*_motor`, `*_efferent`, `*_endocrine` | output layer, flow = efferent | |
| everything else (intrinsic, visual projection, descending, ascending, ...) | hidden layer, flow = intrinsic | |
| `assignedOlHex1/2` (hex coordinates of optic-lobe columnar cells) | retina coordinates | photoreceptors are placed at the synapse-weighted coordinate of their columnar targets; left eye sees the left half of the frame |
| `descending_neuron` + `*_motor` | action readout neurons | LayerNorm -> linear heads |
| membrane time constant, threshold | per-neuron alpha_i, b_i | learnable |

Dynamics:

```text
h[t+1] = (1 - alpha) * h[t] + alpha * min( ReLU( W h[t] + b + u[t] ), h_max )
W[post, pre] = sign(pre) * (syn / sum_in syn) * exp(g_edge)        g_edge learnable, initialised to 0
```

The resting bias b starts at 0.1. Photoreceptors release only histamine (inhibitory), so if every
neuron started at 0 no inhibitory input could ever be read by a ReLU unit. With a little tonic
activity, light shows up as a *decrease* downstream, matching real L1/L2 responses. At global gain
1.0 the whole CNS has a stable baseline (about 98% of neurons active, mean 0.1); gain 5 diverges.

The sparse product is a gather + `index_add` with a custom backward, processed in edge chunks, so
backprop through time stores only the (batch x N) state per step and never a (batch x edges)
tensor. That keeps whole-CNS training inside ~1.5 GB of GPU memory.

## Data

Three public files on Google Storage (CC-BY 4.0, no login). Put them in `data/`:

```bash
B=https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome
curl -o data/body-annotations.feather        $B/body-annotations-male-cns-v1.0-minconf-0.5.feather   # 14 MB
curl -o data/body-neurotransmitters.feather  $B/body-neurotransmitters-male-cns-v1.0.feather         # 43 MB
curl -o data/connectome-weights.feather      $B/connectome-weights-male-cns-v1.0-minconf-0.5.feather # 1.1 GB
```

The weights table has 152M rows; it is filtered in pyarrow batches to annotated neurons (about
25 s the first time) and cached under `data/cache/`.

## Usage

```bash
pip install -e ".[dev]"            # add [rllib] for Ray RLlib
pytest                             # runs on a synthetic MaleCNS-format connectome, no download needed

python scripts/demo_stimulate.py --class gustatory                # who lights up after a sugar stimulus
python scripts/play.py  --suite atari   --game pong               # whole CNS plays Pong (untrained)
python scripts/play.py  --suite classic --game cartpole           # same model, vector observations

# simple trainers (read nfly/rl/simple to learn how it works)
python scripts/train_rl.py --algo ppo --suite atari --game pong --subset visual --envs 8 --updates 5000 --device cuda
python scripts/train_rl.py --algo a2c --suite classic --game cartpole --subset visual_small

# RLlib (scale out: env runners, GPUs, checkpoints, Tune)
python scripts/train_rllib.py --algo PPO --suite atari --game pong --subset visual --gpus 1 --iters 200

# watch it play in the browser (http://127.0.0.1:8000)
python scripts/serve.py --suite atari --game pong --checkpoint runs/ppo-atari-pong.pt
python scripts/serve.py --suite classic --game cartpole --policy random          # no data needed
```

```python
from nfly import load_malecns, select_subset, FlyAgent
from nfly.suite import get_suite, play_episode

conn = select_subset(load_malecns("data"), "visual")            # all | brain | visual | visual_small
env = get_suite("atari").make("breakout", seed=0)
agent = FlyAgent.build(conn, env.observation_space, env.action_space)
print(play_episode(agent, env).ret)
```

### Bring your own game

Any Gymnasium env works as is: `get_suite("gym").make("LunarLander-v3")`. To package a
collection, implement two methods:

```python
from nfly.suite import GameSuite, register

@register("mygames")
class MySuite(GameSuite):
    def games(self):
        return ["level1", "level2"]

    def make(self, game, seed=None, render_mode=None, **kw):
        return self.finish(MyEnv(game, render_mode=render_mode), seed)   # any gym.Env
```

`make_vector`, `spaces` and `finish` come from the base class. The agent only looks at
`observation_space` and `action_space`; no model code changes.

### Bring your own senses or muscles

Subclass `ObservationEncoder` (provide `idx` and `encode`) or `ActionDecoder` (provide
`dist_inputs` and `distribution`) and pass it to `FlyAgent.build(..., encoder=..., decoder=...)`.

### Subsets

| Name | Content | Neurons |
| --- | --- | --- |
| all | whole central nervous system | 166,700 |
| brain | without the ventral nerve cord | ~146,000 |
| visual | optic lobes + visual projection + central brain + descending neurons | ~139,000 |
| visual_small | optic lobes + visual projection + descending neurons | ~107,000 |

## Performance

| Scenario | CPU (M-series laptop) | RTX 5090 |
| --- | --- | --- |
| whole CNS playing Pong | 30 ms / step | 8 ms / step |
| `visual` subset, 8 envs, simple PPO/A2C | 41 steps / s | 236 steps / s |

### Watch any session in the browser

`nfly.viz` runs a (policy, env) session in a background thread and streams every step to a
single-page viewer: the rendered frame, the action taken with its probability distribution, a
colour-coded action timeline with reward ticks, and a step log. Pause, single-step, reset and
change the playback speed from the page. Programmatic use:

```python
from nfly.viz import SessionConfig, serve
server = serve(SessionConfig(suite="atari", game="breakout", subset="visual", checkpoint="runs/ppo.pt"), port=8000)
print(server.url)          # open in a browser; server.stop() when done
```

The launch flow is the same for every front end: `SessionConfig` -> `build_session` -> a
`Session` (env, policy, action names). Any object with `initial_state(batch)` and
`act(obs, h, greedy)` is a valid policy (`RandomPolicy` is the smallest example), and any
`gym.Env` created with `render_mode="rgb_array"` can be shown. The server is standard-library
only (HTTP + Server-Sent Events), so it needs no extra dependency.

## Two training tracks

- **`nfly.rl.simple`** is for reading and quick experiments: A2C and recurrent PPO in about 80
  lines each, sharing rollout collection, GAE and replay in `common.py`. No dependency beyond
  torch and gymnasium.
- **`nfly.rl.rllib`** is for producing models: `FlyRLModule` wraps the agent as a stateful RLlib
  module, `build_config("PPO" | "APPO" | "IMPALA", ...)` returns a ready `AlgorithmConfig`.
  Note that RLlib stores the recurrent state (one float per neuron) at every time step of every
  episode; with the `visual` subset that is 0.5 MB per step, so keep `rollout_fragment_length`
  and the number of env runners moderate.

## Code principles

The project follows the code-smell catalogue at
<https://refactoring.guru/refactoring/smells>. The checklist we apply to every change lives in
[CLAUDE.md](CLAUDE.md), together with the language rule (all committed text in English, ASCII
source) and the one-directional layer rule. In short:

- **Bloaters**: short functions, one reason to change per class, dataclasses instead of tuples and
  dicts, config objects instead of long parameter lists.
- **OO abusers**: type dispatch lives in one factory; no temporary fields; siblings share one
  interface.
- **Change preventers**: a new game, sense or algorithm touches only its own layer; every rule or
  constant exists once.
- **Dispensables**: no narrating comments, duplicate blocks, dead code or speculative options.
- **Couplers**: no reaching into other modules' internals, no message chains across layers, no
  pass-through classes.

## License and citation

Code: license to be chosen before the public release (MIT suggested). Data: MaleCNS v1.0, CC-BY 4.0, FlyEM (HHMI Janelia), University of Cambridge, MRC LMB,
Google Research. Please cite the MaleCNS release when you publish results built on this model.
