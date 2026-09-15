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
    assert obs.shape == (2, 84, 84) and 0 <= obs[0].max() <= 1        # [frame, change]
    agent = FlyAgent.build(visual_connectome(), env.observation_space, env.action_space)
    assert play_episode(agent, env, seed=0, max_steps=5).steps == 5
    env.close()


def test_a2c_smoke():
    from nfly.rl import A2CConfig, train_a2c
    venv = get_suite("classic").make_vector("cartpole", 2)
    agent = FlyAgent.build(visual_connectome(), venv.single_observation_space, venv.single_action_space)
    train_a2c(agent, venv, A2CConfig(rollout=4, updates=3, log_every=100), log=lambda *_: None)
    venv.close()


def test_readout_norm_calibration_exposes_small_signal():
    from nfly.interface import ReadoutNorm
    torch.manual_seed(0)
    pattern = torch.randn(50) * 5                      # large fixed per-neuron offset
    signal = torch.randn(200, 50) * 1e-3               # tiny informative part
    norm = ReadoutNorm(50)
    norm.calibrate(pattern + signal)
    out = norm(pattern + signal)
    assert out.abs().mean() < 3 and 0.5 < out.std(0).mean() < 2      # centred and rescaled to O(1)
    assert torch.equal(norm(pattern + signal[:1]), norm(pattern + signal[:1]))
    assert all(p.requires_grad for p in norm.parameters())


def test_agent_build_calibrates_readout():
    c = visual_connectome()
    a = FlyAgent.build(c, gym.spaces.Box(0, 1, (84, 84), np.float32), gym.spaces.Discrete(4))
    assert not torch.all(a.decoder.norm.mean == 0)      # calibrated, not the default zeros
    h = a.initial_state(6)
    for _ in range(40):                                  # features stay in range well after the reset transient
        feats, h = a.step(torch.rand(6, 84, 84), h)
    assert feats.abs().max() <= 10 and (feats.abs() >= 10).float().mean() < 0.05


def test_vector_suites_standardise_observations():
    import gymnasium as gym
    env = get_suite("classic").make("cartpole", seed=0)
    assert any(isinstance(w, gym.wrappers.NormalizeObservation) for w in _wrappers(env))
    atari = get_suite("atari").make("pong", seed=0)
    assert not any(isinstance(w, gym.wrappers.NormalizeObservation) for w in _wrappers(atari))
    env.close(); atari.close()


def _wrappers(env):
    while hasattr(env, "env"):
        yield env
        env = env.env


def test_mlp_reference_runs_through_simple_trainers():
    from nfly.rl import PPOConfig, train_ppo
    from nfly.rl.simple.reference import MLPReference
    venv = get_suite("classic").make_vector("cartpole", 2)
    agent = MLPReference(venv.single_observation_space, venv.single_action_space)
    returns = train_ppo(agent, venv, PPOConfig(rollout=8, updates=3, minibatch_envs=2, log_every=100), log=lambda *_: None)
    assert isinstance(returns, list)
    venv.close()


def test_param_groups_scale_brain_and_heads():
    c = visual_connectome()
    a = FlyAgent.build(c, gym.spaces.Box(0, 1, (84, 84), np.float32), gym.spaces.Discrete(4))
    rest, heads, brain = a.param_groups(1e-3, brain_scale=0.1, reference_fan_in=2)
    assert brain["lr"] == pytest.approx(1e-4) and rest["lr"] == 1e-3
    assert heads["lr"] == pytest.approx(1e-3 * 2 / a.decoder.n_features)
    names = {id(q): n for n, q in a.named_parameters()}
    assert all(names[id(q)].startswith("brain.") for q in brain["params"])
    assert all(names[id(q)].startswith(("value.", "decoder.head", "decoder.proj")) for q in heads["params"])
    assert sum(len(g["params"]) for g in (rest, heads, brain)) == sum(1 for q in a.parameters() if q.requires_grad)


def test_readout_bottleneck_shapes():
    c = visual_connectome()
    a = FlyAgent.build(c, gym.spaces.Box(0, 1, (84, 84), np.float32), gym.spaces.Discrete(4), readout_dim=8)
    k = min(8, a.decoder.n_readout)                                    # never wider than the readout
    feats, _ = a.step(torch.rand(3, 84, 84), a.initial_state(3))
    assert feats.shape == (3, k) and a.decoder.n_features == k and a.value[0].in_features == k
    full = FlyAgent.build(c, gym.spaces.Box(0, 1, (84, 84), np.float32), gym.spaces.Discrete(4), readout_dim=0)
    assert full.decoder.n_features == full.decoder.n_readout
    # head lr is unscaled with the bottleneck, scaled without it
    assert a.param_groups(1e-3)[1]["lr"] == pytest.approx(1e-3)
    assert full.param_groups(1e-3)[1]["lr"] == pytest.approx(1e-3 * min(1.0, 64 / full.decoder.n_readout))


def test_episodes_start_from_the_resting_state():
    c = visual_connectome()
    a = FlyAgent.build(c, gym.spaces.Box(0, 1, (84, 84), np.float32), gym.spaces.Discrete(4))
    h0 = a.initial_state(2)
    assert h0.shape == (2, c.n_neurons) and h0.abs().sum() > 0          # not silence
    with torch.no_grad():                                                # and (nearly) a fixed point with no input
        h1 = a.brain.step(h0, None, a.brain.weights())
    assert torch.allclose(h1, h0, atol=1e-3)


def test_calibrated_projection_reconstructs_vector_observations():
    from nfly import load_malecns
    from nfly.connectome import write_synthetic
    import tempfile, pathlib
    c = load_malecns(write_synthetic(pathlib.Path(tempfile.mkdtemp())), cache=False)
    space = gym.spaces.Box(-1, 1, (4,), np.float32)
    a = FlyAgent.build(c, space, gym.spaces.Discrete(2))
    assert a.decoder.n_features == 4 and a.value[0].in_features == 4    # k shrinks to the observation width
    r2 = a.calibrate(space)
    assert r2 is not None and r2 > 0.5
    # heads still produce the right shapes after the rebuild
    dist, v, _ = a(torch.rand(3, 4) * 2 - 1, a.initial_state(3))
    assert dist.sample().shape == (3,) and v.shape == (3,)


def test_calibrated_projection_uses_pca_for_images():
    c = visual_connectome()
    a = FlyAgent.build(c, gym.spaces.Box(0, 1, (84, 84), np.float32), gym.spaces.Discrete(4), readout_dim=8)
    assert a.decoder.n_features == min(8, a.decoder.n_readout) and a.decoder.proj.in_features == a.decoder.n_readout


def test_ppo_lr_schedule_anneals_and_adapts():
    from nfly.rl import PPOConfig, train_ppo
    from nfly.rl.simple.reference import MLPReference
    lines = []
    venv = get_suite("classic").make_vector("cartpole", 2)
    agent = MLPReference(venv.single_observation_space, venv.single_action_space)
    train_ppo(agent, venv, PPOConfig(rollout=8, updates=4, minibatch_envs=2, log_every=1, anneal_lr=True, adaptive_lr=True), log=lambda m: lines.append(m))
    lrs = [float(l.split("lr")[1].split()[0]) for l in lines]
    assert lrs[0] == 1.0 and lrs[-1] < lrs[0] and all(0.1 <= v <= 1.0 for v in lrs)
    venv.close()


def test_calibrate_on_env_uses_real_observations_and_motion():
    from nfly import load_malecns
    from nfly.connectome import write_synthetic
    import tempfile, pathlib
    c = load_malecns(write_synthetic(pathlib.Path(tempfile.mkdtemp())), cache=False)
    env = get_suite("atari").make("pong", seed=0)
    a = FlyAgent.build(c, env.observation_space, env.action_space, readout_dim=8)
    r2 = a.calibrate_on_env(env, steps=48)
    assert r2 is not None and 4 <= a.decoder.n_features <= 8 and a.value[0].in_features == a.decoder.n_features
    dist, v, _ = a(torch.rand(2, 84, 84), a.initial_state(2))
    assert dist.sample().shape == (2,) and v.shape == (2,)
    env.close()


def test_atari_temporal_contrast_and_retina_high_pass():
    env = get_suite("atari").make("pong", seed=0)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (2, 84, 84) and np.all(obs[1] == 0)          # first change is zero
    obs2, *_ = env.step(0)
    assert np.allclose(obs2[1], obs2[0] - obs[0])
    a = FlyAgent.build(visual_connectome(), env.observation_space, env.action_space)
    still = torch.zeros(1, 2, 84, 84); moving = still.clone(); moving[0, 1, :, 42:] = 1.0     # change over the right half
    d0, d1 = a.encoder.encode(still), a.encoder.encode(moving)
    assert (d1 - d0).abs().max() > 0                                  # change reaches the photoreceptor drive
    env.close()


def test_image_calibration_uses_readout_subspace():
    from nfly import load_malecns
    from nfly.connectome import write_synthetic
    import tempfile, pathlib
    c = load_malecns(write_synthetic(pathlib.Path(tempfile.mkdtemp())), cache=False)
    env = get_suite("atari").make("pong", seed=0)
    a = FlyAgent.build(c, env.observation_space, env.action_space)
    assert a.decoder.n_features == min(128, a.decoder.n_readout)      # image default
    kept = a.calibrate_on_env(env, steps=60)
    assert kept is not None and 0 < kept <= 1.0 + 1e-4                 # variance fraction kept by the subspace
    feats, _ = a.step(torch.as_tensor(env.reset()[0]).unsqueeze(0), a.initial_state(1))
    assert feats.shape[1] == a.decoder.n_features
    env.close()
