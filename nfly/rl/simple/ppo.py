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
    rollout: int = 32          # env steps per segment (also the BPTT horizon)
    updates: int = 1000
    epochs: int = 3            # passes over each segment
    minibatch_envs: int = 4    # envs per minibatch (each minibatch replays the whole segment)
    lr: float = 2.5e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    entropy: float = 0.01
    value_coef: float = 0.5
    max_grad: float = 0.5
    clip_reward: bool = True
    log_every: int = 10
    save_every: int = 100
    out: str | None = None


def train_ppo(agent, venv, cfg: PPOConfig, device="cpu", seed: int = 0, log=None) -> list[float]:
    dev = torch.device(device)
    n_envs = venv.num_envs
    opt = torch.optim.Adam([q for q in agent.parameters() if q.requires_grad], lr=cfg.lr, eps=1e-5)
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
            track.report(update, update * cfg.rollout * n_envs, **{k: v / n_mb for k, v in stats.items()})
        if cfg.out and (update % cfg.save_every == 0 or update == cfg.updates):
            save_checkpoint(agent, cfg.out, returns=track.returns, config=dataclasses.asdict(cfg))
    return track.returns
