"""Shared pieces for the simple (pure-PyTorch) trainers: rollout storage, GAE, logging, checkpoints."""

from __future__ import annotations

import dataclasses
import os
import time
from pathlib import Path

import numpy as np
import torch


def finished_returns(info: dict) -> list[float]:
    """Episode returns recorded by RecordEpisodeStatistics, for both vector autoreset modes."""
    for src in (info.get("final_info"), info):
        if isinstance(src, dict) and "episode" in src:
            ep = src["episode"]
            mask = np.asarray(src.get("_episode", ep.get("_r", np.ones(len(ep["r"]), bool))))
            return [float(x) for x in np.asarray(ep["r"])[mask]]
    return []


def log_line(msg: str) -> None:
    print(msg, flush=True)   # flush so `tail -f` on a redirected log shows progress immediately


@dataclasses.dataclass
class Rollout:
    """One truncated-BPTT segment collected from a vector env (time-major lists of (n_envs, ...))."""

    obs: list          # T x (n_envs, ...) numpy
    actions: list      # T x (n_envs, ...) tensors
    rewards: list      # T x (n_envs,) tensors
    dones: list        # T x (n_envs,) float tensors, 1 where the episode ended at this step
    logps: list        # T x (n_envs,) tensors (behaviour policy)
    values: list       # T x (n_envs,) tensors
    h0: torch.Tensor   # (n_envs, N) hidden state at the start of the segment
    boot: torch.Tensor # (n_envs,) value estimate after the last step
    returns: list[float]


def collect(agent, venv, obs, h, steps: int, device, clip_reward: bool) -> tuple[Rollout, np.ndarray, torch.Tensor]:
    """Run the agent for `steps` env steps, returning the rollout, the last obs and the last h."""
    h0 = h.detach()
    obs_l, act_l, rew_l, done_l, logp_l, val_l, returns = [], [], [], [], [], [], []
    with torch.no_grad():
        for _ in range(steps):
            obs_l.append(np.asarray(obs))
            dist, value, h = agent(torch.as_tensor(np.asarray(obs), device=device), h)
            a = dist.sample()
            obs, r, term, trunc, info = venv.step(agent.decoder.to_env(a))
            d = torch.as_tensor(np.logical_or(term, trunc), dtype=torch.float32, device=device)
            act_l.append(a); logp_l.append(dist.log_prob(a)); val_l.append(value); done_l.append(d)
            rew_l.append(torch.as_tensor(np.sign(r) if clip_reward else r, dtype=torch.float32, device=device))
            h = h * (1 - d).unsqueeze(1)                           # reset finished episodes
            returns.extend(finished_returns(info))
        _, boot, _ = agent(torch.as_tensor(np.asarray(obs), device=device), h)
    return Rollout(obs_l, act_l, rew_l, done_l, logp_l, val_l, h0, boot, returns), obs, h


def gae(ro: Rollout, gamma: float, lam: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalised advantage estimation.  Returns (advantages, value targets), each (T, n_envs)."""
    T = len(ro.rewards)
    adv = torch.zeros(T, ro.boot.shape[0], device=ro.boot.device)
    last = torch.zeros_like(ro.boot)
    next_v = ro.boot
    for t in reversed(range(T)):
        nonterminal = 1 - ro.dones[t]
        delta = ro.rewards[t] + gamma * next_v * nonterminal - ro.values[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
        next_v = ro.values[t]
    return adv, adv + torch.stack(ro.values)


def replay(agent, ro: Rollout, env_idx: torch.Tensor, device):
    """Re-run the agent over a rollout segment for a subset of envs (with gradients).

    Returns log-probs (T, b), entropies (T, b), values (T, b)."""
    h = ro.h0[env_idx]
    weights = agent.weights()
    logps, ents, values = [], [], []
    for t in range(len(ro.obs)):
        dist, value, h = agent(torch.as_tensor(ro.obs[t][env_idx.cpu().numpy()], device=device), h, weights)
        logps.append(dist.log_prob(ro.actions[t][env_idx])); ents.append(dist.entropy()); values.append(value)
        h = h * (1 - ro.dones[t][env_idx]).unsqueeze(1)
    return torch.stack(logps), torch.stack(ents), torch.stack(values)


class Tracker:
    def __init__(self, log=log_line):
        self.returns, self.t0, self.log = [], time.time(), log

    def report(self, update: int, n_steps_total: int, **metrics) -> None:
        recent = np.mean(self.returns[-20:]) if self.returns else float("nan")
        m = " ".join(f"{k} {v:7.3f}" for k, v in metrics.items())
        self.log(f"upd {update:5d} {m} episodes {len(self.returns):4d} mean return(20) {recent:7.2f} "
                 f"{n_steps_total / (time.time() - self.t0):6.1f} steps/s")


def save_checkpoint(agent, path: str | Path, **extra) -> None:
    """Atomic save: readers (viewer, scp) never see a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"agent": agent.state_dict(), **extra}, tmp)
    os.replace(tmp, path)


def load_checkpoint(agent, path: str | Path) -> dict:
    """Restore the agent's state from `save_checkpoint` output; returns the extra fields."""
    payload = torch.load(path, map_location=next(agent.parameters()).device, weights_only=False)
    agent.load_state_dict(payload.pop("agent"))
    return payload
