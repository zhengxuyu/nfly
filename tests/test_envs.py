import numpy as np
import pytest

miniworld = pytest.importorskip("miniworld")


def test_miniworld_suite_is_a_fly_game():
    import nfly.envs.miniworld  # noqa: F401  (registers the suite)
    from nfly import FlyAgent
    from nfly.suite import get_suite
    from test_agent import visual_connectome

    env = get_suite("miniworld").make("hallway", seed=0)
    obs, _ = env.reset()
    assert obs.shape == (2, 84, 84) and obs.dtype == np.float32 and 0 <= obs[0].min() <= obs[0].max() <= 1
    agent = FlyAgent.build(visual_connectome(), env.observation_space, env.action_space)
    assert type(agent.encoder).__name__ == "RetinaEncoder"
    import torch
    a, _ = agent.act(torch.as_tensor(obs).unsqueeze(0), agent.initial_state(1))
    obs, r, term, trunc, _ = env.step(int(a[0]))
    assert obs.shape == (2, 84, 84)
    env.close()


mujoco = pytest.importorskip("mujoco")


def test_mujoco_suite_state_and_pixels():
    import torch
    from nfly import FlyAgent
    from nfly.suite import get_suite
    from test_agent import visual_connectome

    env = get_suite("mujoco").make("invertedpendulum", seed=0)
    obs, _ = env.reset()
    agent = FlyAgent.build(visual_connectome(), env.observation_space, env.action_space)
    assert type(agent.encoder).__name__ == "VectorEncoder" and type(agent.decoder).__name__ == "BoxDecoder"
    a, _ = agent.act(torch.as_tensor(obs).float().unsqueeze(0), agent.initial_state(1))
    obs, r, term, trunc, _ = env.step(a[0])
    assert env.action_space.contains(np.asarray(a[0], dtype=np.float32))
    env.close()

    env = get_suite("mujoco", observation="pixels").make("invertedpendulum", seed=0)
    obs, _ = env.reset()
    assert obs.shape == (2, 84, 84) and 0 <= obs[0].min() <= obs[0].max() <= 1
    agent = FlyAgent.build(visual_connectome(), env.observation_space, env.action_space)
    assert type(agent.encoder).__name__ == "RetinaEncoder"
    env.close()
