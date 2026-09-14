"""A2C with truncated backprop through time for a recurrent FlyAgent on a vector env."""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path

import numpy as np
import torch


@dataclasses.dataclass
class A2CConfig:
    rollout: int = 16
    updates: int = 1000
    lr: float = 3e-4
    gamma: float = 0.99
    entropy: float = 0.01
    value_coef: float = 0.5
    max_grad: float = 1.0
    clip_reward: bool = True
    log_every: int = 10
    save_every: int = 100
    out: str | None = None


def _finished_returns(info: dict) -> list[float]:
    """Episode returns recorded by RecordEpisodeStatistics, for both vector autoreset modes."""
    for src in (info.get("final_info"), info):
        if isinstance(src, dict) and "episode" in src:
            ep = src["episode"]
            mask = np.asarray(src.get("_episode", ep.get("_r", np.ones(len(ep["r"]), bool))))
            return [float(x) for x in np.asarray(ep["r"])[mask]]
    return []


def _log(msg: str) -> None:
    print(msg, flush=True)   # flush so `tail -f` on a redirected log shows progress immediately


def train_a2c(agent, venv, cfg: A2CConfig, device="cpu", seed: int = 0, log=_log) -> list[float]:
    dev = torch.device(device)
    n_envs = venv.num_envs
    opt = torch.optim.Adam([q for q in agent.parameters() if q.requires_grad], lr=cfg.lr)
    obs, _ = venv.reset(seed=seed)
    h = agent.initial_state(n_envs)
    returns, t0 = [], time.time()

    for update in range(1, cfg.updates + 1):
        h = h.detach()
        logps, values, ents, rewards, dones = [], [], [], [], []
        for _ in range(cfg.rollout):
            dist, value, h = agent(torch.as_tensor(np.asarray(obs), device=dev), h)
            a = dist.sample()
            obs, r, term, trunc, info = venv.step(agent.decoder.to_env(a))
            done = np.logical_or(term, trunc)
            logps.append(dist.log_prob(a)); values.append(value); ents.append(dist.entropy())
            rewards.append(torch.as_tensor(np.sign(r) if cfg.clip_reward else r, dtype=torch.float32, device=dev))
            d = torch.as_tensor(done, dtype=torch.float32, device=dev)
            dones.append(d)
            h = h * (1 - d).unsqueeze(1)
            returns.extend(_finished_returns(info))

        with torch.no_grad():
            _, R, _ = agent(torch.as_tensor(np.asarray(obs), device=dev), h)
        rets = []
        for t in reversed(range(cfg.rollout)):
            R = rewards[t] + cfg.gamma * R * (1 - dones[t])
            rets.append(R)
        rets = torch.stack(rets[::-1]); vals = torch.stack(values)
        adv = rets - vals
        pg = -(torch.stack(logps) * adv.detach()).mean()
        vl = adv.pow(2).mean()
        ent = torch.stack(ents).mean()
        loss = pg + cfg.value_coef * vl - cfg.entropy * ent
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad)
        opt.step()

        if update % cfg.log_every == 0 or update == 1:
            recent = np.mean(returns[-20:]) if returns else float("nan")
            sps = update * cfg.rollout * n_envs / (time.time() - t0)
            log(f"upd {update:5d} loss {loss.item():7.3f} pg {pg.item():7.3f} v {vl.item():6.3f} ent {ent.item():5.3f} "
                f"episodes {len(returns):4d} mean return(20) {recent:7.2f} {sps:6.1f} steps/s")
        if cfg.out and (update % cfg.save_every == 0 or update == cfg.updates):
            Path(cfg.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"agent": agent.state_dict(), "returns": returns, "config": dataclasses.asdict(cfg)}, cfg.out)
    return returns
