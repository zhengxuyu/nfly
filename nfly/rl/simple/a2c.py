"""A2C with truncated backprop through time: the smallest possible policy-gradient trainer for a
recurrent FlyAgent.  One rollout segment -> one gradient step, no replay, no clipping."""

from __future__ import annotations

import dataclasses

import torch

from .common import Tracker, collect, gae, replay, save_checkpoint


@dataclasses.dataclass
class A2CConfig:
    rollout: int = 16
    updates: int = 1000
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 1.0           # 1.0 = plain n-step returns
    entropy: float = 0.01
    value_coef: float = 0.5
    max_grad: float = 1.0
    clip_reward: bool = True
    log_every: int = 10
    save_every: int = 100
    out: str | None = None


def train_a2c(agent, venv, cfg: A2CConfig, device="cpu", seed: int = 0, log=None) -> list[float]:
    dev = torch.device(device)
    n_envs = venv.num_envs
    opt = torch.optim.Adam([q for q in agent.parameters() if q.requires_grad], lr=cfg.lr)
    obs, _ = venv.reset(seed=seed)
    h = agent.initial_state(n_envs)
    track = Tracker(log) if log else Tracker()
    all_envs = torch.arange(n_envs, device=dev)

    for update in range(1, cfg.updates + 1):
        ro, obs, h = collect(agent, venv, obs, h, cfg.rollout, dev, cfg.clip_reward)
        track.returns.extend(ro.returns)
        adv, v_target = gae(ro, cfg.gamma, cfg.lam)
        logp, ent, value = replay(agent, ro, all_envs, dev)        # one pass with gradients
        pg = -(logp * adv).mean()
        vl = (value - v_target).pow(2).mean()
        loss = pg + cfg.value_coef * vl - cfg.entropy * ent.mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad)
        opt.step()

        if update % cfg.log_every == 0 or update == 1:
            track.report(update, update * cfg.rollout * n_envs, loss=loss.item(), pg=pg.item(), v=vl.item(), ent=ent.mean().item())
        if cfg.out and (update % cfg.save_every == 0 or update == cfg.updates):
            save_checkpoint(agent, cfg.out, returns=track.returns, config=dataclasses.asdict(cfg))
    return track.returns
