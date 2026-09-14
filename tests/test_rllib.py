import gymnasium as gym
import numpy as np
import pytest
import torch

pytest.importorskip("ray")
from ray.rllib.core.columns import Columns  # noqa: E402

from nfly.connectome import write_synthetic  # noqa: E402
from nfly.rl.rllib import FlyRLModule, build_config, env_id  # noqa: E402


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    return str(write_synthetic(tmp_path_factory.mktemp("malecns")))


@pytest.mark.parametrize("obs_space,act_space,dist", [
    (gym.spaces.Box(0, 1, (84, 84), np.float32), gym.spaces.Discrete(6), "TorchCategorical"),
    (gym.spaces.Box(-1, 1, (4,), np.float32), gym.spaces.Box(-2, 2, (2,), np.float32), "TorchDiagGaussian"),
])
def test_module_forward_passes(data_dir, obs_space, act_space, dist):
    m = FlyRLModule(observation_space=obs_space, action_space=act_space,
                    model_config={"data_dir": data_dir, "min_syn": 1, "subset": "all"})
    B, T = 3, 5
    state = {k: torch.as_tensor(v).unsqueeze(0).repeat(B, 1) for k, v in m.get_initial_state().items()}
    batch = {Columns.OBS: torch.rand(B, T, *obs_space.shape), Columns.STATE_IN: state}
    out = m.forward_train(batch)
    n_out = act_space.n if isinstance(act_space, gym.spaces.Discrete) else 2 * act_space.shape[0]
    assert out[Columns.ACTION_DIST_INPUTS].shape == (B, T, n_out)
    assert out[Columns.STATE_OUT]["h"].shape == (B, m.agent.n_neurons)
    assert m.compute_values(batch, out[Columns.EMBEDDINGS]).shape == (B, T)
    v_full = m.compute_values(batch)                       # GAE path: no grad, chunked over sequences
    assert v_full.shape == (B, T) and not v_full.requires_grad
    assert torch.allclose(v_full, m.compute_values(batch, out[Columns.EMBEDDINGS]).detach(), atol=1e-5)
    assert m.get_inference_action_dist_cls().__name__ == dist
    m.forward_inference(batch); m.forward_exploration(batch)


def test_ppo_trains_end_to_end(data_dir, tmp_path):
    import ray
    ray.init(num_cpus=2, include_dashboard=False, log_to_driver=False, ignore_reinit_error=True)
    try:
        cfg = build_config("PPO", suite="classic", game="cartpole", data_dir=data_dir, subset="all", min_syn=1,
                           num_env_runners=0, train_batch_size=128, minibatch_size=32, num_epochs=1, max_seq_len=8)
        assert cfg.env == env_id("classic", "cartpole")
        algo = cfg.build_algo()
        result = algo.train()
        assert result["num_env_steps_sampled_lifetime"] >= 128
        out = algo.save_to_path(str(tmp_path / "ckpt"))       # absolute path, as scripts/train_rllib.py passes it
        assert (tmp_path / "ckpt").exists() and out
    finally:
        ray.shutdown()
