"""Exercise fixed-policy collection, complete targets and held-out critic fitting."""

import gymnasium as gym
import pytest
import torch

from nfly import FlyAgent
from scripts.probe_value import collect_episodes, discounted_returns, split_episodes, fit_critic, FitConfig
from test_agent import visual_connectome
from test_simple_controls import ShortEpisode


def test_complete_episode_probe_fits_without_changing_actor():
    torch.manual_seed(0)
    env = gym.vector.SyncVectorEnv([lambda: ShortEpisode(4) for _ in range(6)], autoreset_mode="SameStep")
    agent = FlyAgent.build(visual_connectome(), env.single_observation_space, env.single_action_space)
    before = {k: v.clone() for k, v in agent.state_dict().items()}
    episodes = collect_episodes(agent, env, list(range(6)))
    env.close()
    assert all(len(e["reward"]) == 4 for e in episodes)
    assert all(torch.equal(v, before[k]) for k, v in agent.state_dict().items())
    assert discounted_returns(torch.tensor([0., 1., 0., -1.]), 0.5).tolist() == [0.375, 0.75, -0.5, -1.0]
    splits = split_episodes(episodes, 0.99)
    assert [s["seeds"] for s in splits] == [[0, 1, 2, 3], [4], [5]]
    model = torch.nn.Linear(agent.decoder.n_features, 1)
    result = fit_critic(model, splits, "features", FitConfig(steps=100, lr=0.01, batch=16))
    assert result["selected_train"]["mse"] < result["history"][0]["train"]["mse"]
    assert torch.isfinite(torch.tensor(result["test"]["ev"]))


def test_probe_rejects_partial_episodes_and_repeated_seeds():
    env = gym.vector.SyncVectorEnv([lambda: ShortEpisode(4)], autoreset_mode="SameStep")
    agent = FlyAgent.build(visual_connectome(), env.single_observation_space, env.single_action_space)
    with pytest.raises(RuntimeError, match="incomplete"):
        collect_episodes(agent, env, [0], max_steps=1)
    env.close()
    with pytest.raises(ValueError, match="unique seeds"):
        split_episodes([{"seed": 0}] * 6, 0.99)
