"""Paired evaluation on explicit seeds, isolated from training environments and RNG."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import torch

from .common import reset_state


@dataclasses.dataclass
class EvaluationConfig:
    episodes: int = 5
    seed: int = 10_000
    max_steps: int = 27_000
    temperatures: tuple[float, ...] = (1.0,)
    gamma: float = 0.99
    output: str | None = None

    def __post_init__(self):
        if self.episodes < 1 or self.max_steps < 1 or any(t <= 0 for t in self.temperatures):
            raise ValueError("Evaluation counts and temperatures must be positive")


@torch.no_grad()
def evaluate_mode(agent, env, cfg, temperature, device):
    obs, _ = env.reset(seed=[cfg.seed + i for i in range(cfg.episodes)])
    h = agent.initial_state(cfg.episodes)
    totals, lengths = np.zeros(cfg.episodes), np.zeros(cfg.episodes, dtype=int)
    active = np.ones(cfg.episodes, dtype=bool)
    terminated = np.zeros(cfg.episodes, dtype=bool)
    entropies, value_trace, reward_trace, masks = [], [], [], []
    weights = agent.weights()
    for _ in range(cfg.max_steps):
        dist, value, h = agent(torch.as_tensor(np.asarray(obs), device=device), h, weights)
        if temperature is not None and temperature != 1.0:
            if not isinstance(dist, torch.distributions.Categorical):
                raise ValueError("Temperature sweeps require discrete actions")
            dist = torch.distributions.Categorical(logits=dist.logits / temperature)
        entropies.append(float(dist.entropy()[torch.as_tensor(active, device=device)].mean()))
        action = dist.mode if temperature is None else dist.sample()
        obs, reward, term, trunc, _ = env.step(agent.decoder.to_env(action))
        value_trace.append(value.cpu().numpy())
        reward_trace.append(reward.copy())
        masks.append(active.copy())
        totals += reward * active
        lengths += active
        done = np.logical_or(term, trunc)
        terminated |= np.asarray(term) & active
        active &= ~done
        h = reset_state(agent, h, torch.as_tensor(done, device=device))
        if not active.any():
            break
    if active.any():
        raise RuntimeError("Evaluation step cap reached; refusing to report partial episodes as returns")
    return dict(**value_metrics(value_trace, reward_trace, masks, terminated, cfg.gamma),
                terminated=terminated.tolist(), mean=float(totals.mean()), returns=totals.tolist(),
                steps=lengths.tolist(), entropy=float(np.mean(entropies)), temperature=temperature)


def value_metrics(values, rewards, masks, terminated, gamma):
    """Monte Carlo prediction quality on complete, genuinely terminated episodes."""
    rewards, valid = np.asarray(rewards), np.asarray(masks)
    targets, tail = np.zeros_like(rewards), np.zeros(len(terminated))
    for t in reversed(range(len(rewards))):
        tail = (rewards[t] + gamma * tail) * valid[t]
        targets[t] = tail
    valid &= terminated[None, :]
    predicted, targets = np.asarray(values)[valid], targets[valid]
    ev = float(1 - np.var(targets - predicted) / max(np.var(targets), 1e-8)) if len(targets) else None
    mse = float(np.mean((predicted - targets) ** 2)) if len(targets) else None
    return dict(value_mc_ev=ev, value_mc_mse=mse)


def evaluate_policy(agent, make_env, cfg: EvaluationConfig, update: int = 0):
    device = next(agent.parameters()).device
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    was_training, numpy_state = agent.training, np.random.get_state()
    results = {}
    try:
        agent.eval()
        with torch.random.fork_rng(devices=devices):
            for temperature in (None, *cfg.temperatures):
                torch.manual_seed(cfg.seed)
                env = make_env()
                try:
                    key = "greedy" if temperature is None else f"sampled_t{temperature:g}"
                    results[key] = evaluate_mode(agent, env, cfg, temperature, device)
                finally:
                    env.close()
    finally:
        agent.train(was_training)
        np.random.set_state(numpy_state)
    result = dict(update=update, seed=cfg.seed, episodes=cfg.episodes,
                  greedy_mean=results["greedy"]["mean"], modes=results)
    if cfg.output:
        path = Path(cfg.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as stream:
            stream.write(json.dumps(result) + "\n")
    return result
