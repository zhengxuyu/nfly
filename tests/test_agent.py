import gymnasium as gym
import numpy as np
import pandas as pd
import pytest
import torch

from nfly import FlyAgent
from nfly.brain.rnn import _SparseRecurrent
from nfly.connectome import build_connectome
from nfly.interface import ImageProjectionEncoder, RetinaEncoder, VectorEncoder, BoxDecoder, DiscreteDecoder, build_retina
from nfly.suite import GameSuite, available_suites, get_suite, play_episode, register


@pytest.mark.parametrize("chunk", [7, 200, 10_000])
def test_sparse_recurrent_gradients_match_dense(chunk):
    torch.manual_seed(0)
    n, e, b = 30, 200, 4
    pre, post = torch.randint(0, n, (e,)), torch.randint(0, n, (e,))
    w = torch.randn(e, requires_grad=True); h = torch.randn(b, n, requires_grad=True)
    g = torch.randn(b, n)
    y = _SparseRecurrent.apply(h, w, pre, post, chunk)
    dense0 = torch.zeros(n, n).index_put((post, pre), w.detach(), accumulate=True)
    assert torch.allclose(y, h.detach() @ dense0.T, atol=1e-5)
    (y * g).sum().backward()
    w2 = w.detach().clone().requires_grad_(True); h2 = h.detach().clone().requires_grad_(True)
    dense = torch.zeros(n, n).index_put((post, pre), w2, accumulate=True)
    ((h2 @ dense.T) * g).sum().backward()
    assert torch.allclose(h.grad, h2.grad, atol=1e-5) and torch.allclose(w.grad, w2.grad, atol=1e-5)


def visual_connectome(n_col=20, hex_coords=True):
    """Tiny optic-lobe-like connectome: photoreceptors -> L1 columns (hex coords) -> descending neurons."""
    rows, pre, post, syn, k = [], [], [], [], 0
    for side in ("L", "R"):
        for c in range(n_col):
            rows.append(dict(root_id=k, cell_type="R1-R6", super_class="ol_sensory", flow="afferent", nt_type="HIS", side=side, hex1=-1, hex2=-1)); pr = k; k += 1
            rows.append(dict(root_id=k, cell_type="L1", super_class="ol_intrinsic", flow="intrinsic", nt_type="ACH", side=side,
                             hex1=c % 5 if hex_coords else -1, hex2=c // 5 if hex_coords else -1)); l1 = k; k += 1
            pre.append(pr); post.append(l1); syn.append(20)
    for _ in range(3):
        rows.append(dict(root_id=k, cell_type="DN", super_class="descending_neuron", flow="intrinsic", nt_type="ACH", side="R", hex1=-1, hex2=-1))
        for l1 in range(1, k, 2):
            pre.append(l1); post.append(k); syn.append(5)
        k += 1
    df = pd.DataFrame(rows); df["class"] = ""
    return build_connectome(df, pre, post, syn, np.where(df.nt_type.to_numpy()[pre] == "HIS", -1.0, 1.0))


def test_retina_places_photoreceptors_and_splits_eyes():
    c = visual_connectome()
    ret = build_retina(c)
    assert ret.n_inputs == 40 and (c.neurons.iloc[ret.idx.numpy()]["cell_type"] == "R1-R6").all()
    gx = ret.grid.view(-1, 2)[:, 0]
    assert (gx[torch.as_tensor(ret.side == "L")] <= 0).all() and (gx[torch.as_tensor(ret.side == "R")] >= 0).all()
    frame = torch.zeros(1, 84, 84); frame[:, :, 42:] = 1.0
    d = ret.encode(frame)[0]
    assert d[torch.as_tensor(ret.side == "R")].mean() > 0.5 and d[torch.as_tensor(ret.side == "L")].mean() < -0.5


def test_encoder_selection_from_spaces():
    c = visual_connectome()
    img = gym.spaces.Box(0, 1, (84, 84), np.float32)
    assert isinstance(FlyAgent.build(c, img, gym.spaces.Discrete(3)).encoder, RetinaEncoder)
    assert isinstance(FlyAgent.build(visual_connectome(hex_coords=False), img, gym.spaces.Discrete(3)).encoder, ImageProjectionEncoder)
    vec = gym.spaces.Box(-1, 1, (4,), np.float32)
    a = FlyAgent.build(c, vec, gym.spaces.Box(-2, 2, (2,), np.float32))
    assert isinstance(a.encoder, VectorEncoder) and isinstance(a.decoder, BoxDecoder)
    assert isinstance(FlyAgent.build(c, img, gym.spaces.Discrete(3)).decoder, DiscreteDecoder)


def test_agent_forward_backward_discrete_and_box():
    c = visual_connectome()
    a = FlyAgent.build(c, gym.spaces.Box(0, 1, (84, 84), np.float32), gym.spaces.Discrete(4))
    dist, v, h = a(torch.rand(3, 84, 84), a.initial_state(3))
    assert dist.sample().shape == (3,) and v.shape == (3,) and h.shape == (3, c.n_neurons)
    (dist.entropy().sum() + v.sum()).backward()
    assert a.brain.log_gain.grad is not None and a.input_gain.grad is not None
    b = FlyAgent.build(c, gym.spaces.Box(-1, 1, (6,), np.float32), gym.spaces.Box(-1, 1, (2,), np.float32))
    dist, v, h = b(torch.rand(2, 6), b.initial_state(2))
    assert b.decoder.to_env(dist.sample()).shape == (2, 2)


def test_suite_registry_and_custom_suite():
    assert {"gym", "classic"} <= set(available_suites())

    @register("_dummy")
    class Dummy(GameSuite):
        def games(self): return ["cp"]
        def make(self, game, seed=None, render_mode=None, **kw):
            env = gym.make("CartPole-v1")
            if seed is not None: env.reset(seed=seed)
            return env
    s = get_suite("_dummy")
    env = s.make("cp", seed=0)
    agent = FlyAgent.build(visual_connectome(), env.observation_space, env.action_space)
    res = play_episode(agent, env, seed=0, max_steps=20)
    assert res.steps >= 1 and isinstance(res.ret, float)
    venv = s.make_vector("cp", 2)
    assert venv.num_envs == 2; venv.close()


def test_agent_on_synthetic_release(tmp_path):
    from nfly import load_malecns
    from nfly.connectome import write_synthetic
    c = load_malecns(write_synthetic(tmp_path), cache=False)
    env = get_suite("atari").make("pong", seed=0)
    a = FlyAgent.build(c, env.observation_space, env.action_space)
    assert isinstance(a.encoder, RetinaEncoder) and a.encoder.n_inputs == 60 and a.decoder.n_readout == 30
    assert play_episode(a, env, seed=0, max_steps=3).steps == 3
    env.close()


def test_atari_suite():
    s = get_suite("atari")
    env = s.make("pong", seed=0)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (84, 84) and 0 <= obs.max() <= 1
    agent = FlyAgent.build(visual_connectome(), env.observation_space, env.action_space)
    assert play_episode(agent, env, seed=0, max_steps=5).steps == 5
    env.close()


def test_a2c_smoke():
    from nfly.rl import A2CConfig, train_a2c
    venv = get_suite("classic").make_vector("cartpole", 2)
    agent = FlyAgent.build(visual_connectome(), venv.single_observation_space, venv.single_action_space)
    train_a2c(agent, venv, A2CConfig(rollout=4, updates=3, log_every=100), log=lambda *_: None)
    venv.close()
