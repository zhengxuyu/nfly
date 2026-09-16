"""Regression checks for recurrent resets and controlled PPO fine-tuning."""

import argparse

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn

from nfly.cli import apply_freezes
from nfly.rl.simple.common import collect, gae, replay
from nfly.rl.simple.ppo import PPOConfig, train_ppo


class ShortEpisode(gym.Env):
    observation_space = gym.spaces.Box(0, 20, (1,), np.float32)
    action_space = gym.spaces.Discrete(2)

    def __init__(self, length=1, timeout=False):
        self.length, self.timeout = length, timeout

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return np.array([0], np.float32), {}

    def step(self, action):
        self.t += 1
        done = self.t == self.length
        return np.array([self.t], np.float32), 1.0, done and not self.timeout, done and self.timeout, {}


class RestingAgent(nn.Module):
    def __init__(self):
        super().__init__()
        from types import SimpleNamespace
        self.policy = nn.Linear(1, 2)
        self.value = nn.Linear(1, 1)
        self.decoder = SimpleNamespace(to_env=lambda a: a.cpu().numpy())
        self.seen = []

    def initial_state(self, batch):
        return torch.full((batch, 1), 0.25)

    def weights(self):
        return None

    def forward(self, obs, h, weights=None):
        self.seen.append(h.detach().clone())
        return torch.distributions.Categorical(logits=self.policy(h)), self.value(obs.float()).squeeze(-1), h + 1


def vector_env():
    return gym.vector.SyncVectorEnv([lambda: ShortEpisode(1), lambda: ShortEpisode(3)], autoreset_mode="SameStep")


def test_collect_and_replay_reset_only_finished_states_to_rest():
    agent, env = RestingAgent(), vector_env()
    obs, _ = env.reset(seed=0)
    ro, _, _ = collect(agent, env, obs, agent.initial_state(2), 2, "cpu", False)
    assert torch.equal(agent.seen[1], torch.tensor([[0.25], [1.25]]))
    agent.seen.clear()
    logp, _, _ = replay(agent, ro, torch.arange(2), "cpu")
    assert torch.equal(agent.seen[1], torch.tensor([[0.25], [1.25]]))
    assert torch.allclose(logp, torch.stack(ro.logps))
    env.close()


def test_time_limit_bootstraps_final_observation_without_cross_episode_gae():
    agent = RestingAgent()
    with torch.no_grad():
        agent.value.weight.fill_(2)
        agent.value.bias.zero_()
    env = gym.vector.SyncVectorEnv([lambda: ShortEpisode(1, timeout=True)], autoreset_mode="SameStep")
    obs, _ = env.reset()
    ro, _, _ = collect(agent, env, obs, agent.initial_state(1), 2, "cpu", False)
    _, targets = gae(ro, gamma=0.9, lam=1)
    assert torch.allclose(targets, torch.full((2, 1), 2.8))
    env.close()


def test_critic_warmup_changes_only_value_parameters():
    torch.manual_seed(4)
    agent, env = RestingAgent(), vector_env()
    before = {n: p.detach().clone() for n, p in agent.named_parameters()}
    cfg = PPOConfig(rollout=4, updates=2, epochs=2, critic_warmup=2, minibatch_envs=2)
    train_ppo(agent, env, cfg, log=lambda _: None)
    assert all(torch.equal(p, before[n]) for n, p in agent.named_parameters() if not n.startswith("value."))
    assert any(not torch.equal(p, before[n]) for n, p in agent.named_parameters() if n.startswith("value."))
    env.close()


def test_heads_only_keeps_full_feature_path_fixed_during_ppo():
    from test_agent import visual_connectome
    from nfly import FlyAgent
    env = vector_env()
    agent = FlyAgent.build(visual_connectome(), env.single_observation_space, env.single_action_space, head_hidden=8)
    apply_freezes(agent, argparse.Namespace(heads_only=True))
    before = {n: p.detach().clone() for n, p in agent.named_parameters() if not p.requires_grad}
    train_ppo(agent, env, PPOConfig(rollout=4, updates=2, epochs=1), log=lambda _: None)
    assert before and all(torch.equal(dict(agent.named_parameters())[n], p) for n, p in before.items())
    assert not agent.input_gain.requires_grad and not agent.decoder.norm.shift.requires_grad
    env.close()


def test_exact_kl_detects_policy_change_after_optimizer_step():
    from nfly.rl.simple.common import policy_kl
    agent, env = RestingAgent(), vector_env()
    obs, _ = env.reset()
    ro, _, _ = collect(agent, env, obs, agent.initial_state(2), 3, "cpu", False)
    assert policy_kl(agent, ro, "cpu") == pytest.approx(0, abs=1e-7)
    with torch.no_grad():
        agent.policy.bias[0].add_(5)
    assert policy_kl(agent, ro, "cpu") > 0.1
    env.close()


def test_kl_guard_stops_after_first_excessive_minibatch(monkeypatch):
    torch.manual_seed(4)
    agent, env = RestingAgent(), vector_env()
    steps = []
    original = torch.optim.Adam.step
    def counted_step(self, *args, **kwargs):
        steps.append(1)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(torch.optim.Adam, "step", counted_step)
    cfg = PPOConfig(rollout=4, updates=1, epochs=5, lr=1, minibatch_envs=1, target_kl=1e-8)
    train_ppo(agent, env, cfg, log=lambda _: None)
    assert len(steps) == 1
    env.close()


def test_paired_evaluation_preserves_rng_weights_and_training_mode(tmp_path):
    from nfly.rl.simple.evaluation import EvaluationConfig, evaluate_policy
    agent = RestingAgent().train()
    before = {n: p.detach().clone() for n, p in agent.named_parameters()}
    rng = torch.random.get_rng_state().clone()
    cfg = EvaluationConfig(episodes=2, temperatures=(0.5, 1.0), output=str(tmp_path / "eval.jsonl"))
    result = evaluate_policy(agent, vector_env, cfg)
    assert result["greedy_mean"] == 2
    assert result["modes"]["sampled_t1"]["returns"] == [1, 3]
    assert result["modes"]["greedy"]["steps"] == [1, 3]
    assert torch.equal(rng, torch.random.get_rng_state()) and agent.training
    assert all(torch.equal(p, before[n]) for n, p in agent.named_parameters())
    assert len((tmp_path / "eval.jsonl").read_text().splitlines()) == 1


def test_evaluation_refuses_partial_returns():
    from nfly.rl.simple.evaluation import EvaluationConfig, evaluate_policy
    with pytest.raises(RuntimeError, match="partial episodes"):
        evaluate_policy(RestingAgent(), vector_env, EvaluationConfig(episodes=2, max_steps=1))


def test_best_checkpoint_retains_initial_policy_if_training_regresses(tmp_path):
    agent, env = RestingAgent(), vector_env()
    before = agent.policy.weight.detach().clone()
    evaluations = []
    def evaluate(model, update):
        evaluations.append(update)
        return {"greedy_mean": 10 - update}
    path = tmp_path / "train.pt"
    train_ppo(agent, env, PPOConfig(rollout=4, updates=2, epochs=1, eval_every=1, out=str(path)),
              log=lambda _: None, evaluate=evaluate)
    best = torch.load(tmp_path / "train-best.pt", weights_only=False)
    assert best["update"] == 0 and torch.equal(best["agent"]["policy.weight"], before)
    assert evaluations == [0, 1, 2]
    env.close()


def test_initial_temperature_preserves_argmax_and_changes_sampling():
    from nfly.cli import rescale_policy
    from nfly.rl.simple.reference import MLPReference
    torch.manual_seed(3)
    agent = MLPReference(gym.spaces.Box(-1, 1, (1,), np.float32), gym.spaces.Discrete(2))
    with torch.no_grad():
        agent.decoder.head.bias[0] = 0.5
    obs, h = torch.ones(4, 1), agent.initial_state(4)
    old, _, _ = agent(obs, h)
    rescale_policy(agent, 0.25)
    new, _, _ = agent(obs, h)
    assert torch.equal(old.mode, new.mode)
    assert (new.entropy() < old.entropy()).all()


def test_value_metrics_use_complete_returns_and_exclude_time_limits():
    from nfly.rl.simple.evaluation import value_metrics
    values = np.zeros((3, 2))
    rewards = np.ones((3, 2))
    masks = np.array([[True, True], [False, True], [False, True]])
    result = value_metrics(values, rewards, masks, np.array([True, True]), 0.9)
    assert result["value_mc_mse"] == pytest.approx(np.mean(np.square([1, 2.71, 1.9, 1])))
    assert result["value_mc_ev"] == pytest.approx(0)
    truncated = value_metrics(values, rewards, masks, np.array([False, False]), 0.9)
    assert truncated == {"value_mc_ev": None, "value_mc_mse": None}
