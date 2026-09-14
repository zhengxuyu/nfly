"""Recurrent PPO for a FlyAgent, in ~80 lines: collect a truncated-BPTT segment from a vector env,
compute GAE, then for a few epochs replay minibatches of envs through the segment with gradients and
apply the clipped surrogate objective.  Written to be read, not to be fast.
"""

from __future__ import annotations

import dataclasses

import torch

from .common import Tracker, collect, gae, replay, save_checkpoint


@dataclasses.dataclass
class PPOConfig:
    """Defaults follow the SB3 rl-zoo PPO settings for CartPole (lr 1e-3, 10 epochs, gamma 0.98,
    lambda 0.8, no entropy bonus), which beat CleanRL-style and conservative settings 3x on an
    MLP through this very loop (223 vs 80 return at 100k steps). Two additions protect the fly:
    brain parameters (millions of edge gains) take a 10x smaller learning rate, and an epoch is
    cut short once the approximate KL to the behaviour policy exceeds `target_kl`."""

    rollout: int = 32          # env steps per segment (also the BPTT horizon)
    updates: int = 1000
    epochs: int = 10           # passes over each segment
    minibatch_envs: int = 8    # envs per minibatch (each minibatch replays the whole segment)
    lr: float = 1e-3
    brain_lr_scale: float = 0.1   # multiplier on lr for parameters under `agent.brain`
    gamma: float = 0.98
    lam: float = 0.8
    clip: float = 0.2
    entropy: float = 0.0
    value_coef: float = 0.5
    max_grad: float = 0.5
    target_kl: float | None = 0.02
    clip_reward: bool = True
    log_every: int = 10
    save_every: int = 100
    out: str | None = None


def param_groups(agent, lr: float, brain_lr_scale: float) -> list[dict]:
    """Ask the agent for its optimizer groups if it has an opinion (FlyAgent does), else one group."""
    if hasattr(agent, "param_groups"):
        return agent.param_groups(lr, brain_scale=brain_lr_scale)
    return [{"params": [q for q in agent.parameters() if q.requires_grad], "lr": lr}]


def train_ppo(agent, venv, cfg: PPOConfig, device="cpu", seed: int = 0, log=None) -> list[float]:
    dev = torch.device(device)
    n_envs = venv.num_envs
    opt = torch.optim.Adam(param_groups(agent, cfg.lr, cfg.brain_lr_scale), eps=1e-5)
    obs, _ = venv.reset(seed=seed)
    h = agent.initial_state(n_envs)
    track = Tracker(log) if log else Tracker()

    for update in range(1, cfg.updates + 1):
        ro, obs, h = collect(agent, venv, obs, h, cfg.rollout, dev, cfg.clip_reward)
        track.returns.extend(ro.returns)
        adv, v_target = gae(ro, cfg.gamma, cfg.lam)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        old_logp = torch.stack(ro.logps)

        stats = {"pg": 0.0, "v": 0.0, "ent": 0.0, "kl": 0.0, "clipfrac": 0.0}
        n_mb = 0
        for _ in range(cfg.epochs):
            if cfg.target_kl is not None and n_mb and stats["kl"] / n_mb > cfg.target_kl:
                break                                            # policy moved far enough this update
            for env_idx in torch.randperm(n_envs, device=dev).split(cfg.minibatch_envs):
                logp, ent, value = replay(agent, ro, env_idx, dev)
                ratio = torch.exp(logp - old_logp[:, env_idx])
                a = adv[:, env_idx]
                pg = -torch.min(ratio * a, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a).mean()
                vl = (value - v_target[:, env_idx]).pow(2).mean()
                loss = pg + cfg.value_coef * vl - cfg.entropy * ent.mean()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad)
                opt.step()
                with torch.no_grad():
                    stats["pg"] += pg.item(); stats["v"] += vl.item(); stats["ent"] += ent.mean().item()
                    stats["kl"] += (old_logp[:, env_idx] - logp).mean().item()
                    stats["clipfrac"] += ((ratio - 1).abs() > cfg.clip).float().mean().item()
                n_mb += 1

        if update % cfg.log_every == 0 or update == 1:
            track.report(update, update * cfg.rollout * n_envs, **{k: v / n_mb for k, v in stats.items()}, epochs=float(n_mb) / max(1, (n_envs + cfg.minibatch_envs - 1) // cfg.minibatch_envs))
        if cfg.out and (update % cfg.save_every == 0 or update == cfg.updates):
            save_checkpoint(agent, cfg.out, returns=track.returns, config=dataclasses.asdict(cfg))
    return track.returns
