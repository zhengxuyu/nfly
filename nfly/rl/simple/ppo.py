"""Recurrent PPO for a FlyAgent: collect a truncated-BPTT segment from a vector env,
compute GAE, then for a few epochs replay minibatches of envs through the segment with gradients and
apply the clipped surrogate objective.  Written to be read, not to be fast.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import torch

from .common import Tracker, collect, gae, policy_kl, replay, save_checkpoint


@dataclasses.dataclass
class PPOConfig:
    """Defaults follow the SB3 rl-zoo PPO settings for CartPole (lr 1e-3, 10 epochs, gamma 0.98,
    lambda 0.8, no entropy bonus), which beat CleanRL-style and conservative settings 3x on an
    MLP through this very loop (223 vs 80 return at 100k steps). Two additions protect the fly:
    brain parameters (millions of edge gains) take a 10x smaller learning rate, and an epoch is
    cut short once post-update distribution KL to the behaviour policy exceeds `target_kl`."""

    rollout: int = 32          # env steps per segment (also the BPTT horizon)
    updates: int = 1000
    epochs: int = 10           # passes over each segment
    minibatch_envs: int = 8    # envs per minibatch (each minibatch replays the whole segment)
    lr: float = 1e-3
    brain_lr_scale: float = 0.1   # multiplier on lr for parameters under `agent.brain`
    head_fan_in: int | None = 64  # head lr is scaled by head_fan_in / n_readout; None = no scaling
    gamma: float = 0.98
    lam: float = 0.8
    clip: float = 0.2
    entropy: float = 0.0
    value_coef: float = 0.5
    max_grad: float = 0.5
    target_kl: float | None = 0.02
    anneal_lr: bool = False       # linear decay of every learning rate to 0 over `updates` (off: neutral for the fly, hurt the MLP)
    adaptive_lr: bool = False     # after an update, halve the lr if post-update KL > 2 * target_kl, raise it x1.5 if < target_kl / 2
    adaptive_lr_floor: float = 0.1  # adaptive multiplier stays within [floor, 1]
    clip_reward: bool = True
    log_every: int = 10
    save_every: int = 100
    critic_warmup: int = 0       # initial updates train value.* only; policy and features stay fixed
    eval_every: int = 0          # callback at update 0, every N updates and the final update
    out: str | None = None


def param_groups(agent, lr: float, brain_lr_scale: float, head_fan_in: int | None) -> list[dict]:
    """Ask the agent for its optimizer groups if it has an opinion (FlyAgent does), else one group."""
    if hasattr(agent, "param_groups"):
        fan_in = head_fan_in if head_fan_in is not None else agent.decoder.n_readout
        return agent.param_groups(lr, brain_scale=brain_lr_scale, reference_fan_in=fan_in)
    return [{"params": [q for q in agent.parameters() if q.requires_grad], "lr": lr}]


def optimize_rollout(agent, opt, ro, cfg, device, warmup):
    adv, targets = gae(ro, cfg.gamma, cfg.lam)
    ev = 1 - (targets - torch.stack(ro.values)).var(unbiased=False) / targets.var(unbiased=False).clamp_min(1e-8)
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    old_logp = torch.stack(ro.logps)
    stats = {"pg": 0.0, "v": 0.0, "ent": 0.0, "clipfrac": 0.0}
    n_mb, kl, stop = 0, 0.0, False
    for _ in range(cfg.epochs):
        for idx in torch.randperm(ro.h0.shape[0], device=device).split(cfg.minibatch_envs):
            logp, ent, value = replay(agent, ro, idx, device)
            ratio = torch.exp(logp - old_logp[:, idx])
            pg = -torch.min(ratio * adv[:, idx], ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv[:, idx]).mean()
            vl = (value - targets[:, idx]).pow(2).mean()
            loss = cfg.value_coef * vl if warmup else pg + cfg.value_coef * vl - cfg.entropy * ent.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad)
            opt.step()
            kl = policy_kl(agent, ro, device)
            metrics = dict(pg=pg.item(), v=vl.item(), ent=ent.mean().item(),
                           clipfrac=((ratio - 1).abs() > cfg.clip).float().mean().item())
            for key, value in metrics.items():
                stats[key] += value
            n_mb += 1
            if not torch.isfinite(torch.tensor(kl)):
                raise FloatingPointError("Non-finite post-update policy KL")
            if cfg.target_kl is not None and kl > cfg.target_kl:
                stop = True
                break
        if stop:
            break
    batches = (ro.h0.shape[0] + cfg.minibatch_envs - 1) // cfg.minibatch_envs
    return dict(**{k: v / n_mb for k, v in stats.items()}, kl=kl, ev=float(ev),
                epochs=n_mb / batches, critic_only=float(warmup))


def evaluate_checkpoint(agent, evaluate, update, cfg, track, best):
    metrics = evaluate(agent, update)
    track.log(f"eval update {update} {metrics}")
    score = metrics["greedy_mean"]
    if cfg.out and score > best:
        path = Path(cfg.out)
        save_checkpoint(agent, path.with_name(path.stem + "-best" + path.suffix),
                        returns=track.returns, config=dataclasses.asdict(cfg), update=update, evaluation=metrics)
    return max(best, score)


def train_ppo(agent, venv, cfg: PPOConfig, device="cpu", seed: int = 0, log=None, evaluate=None) -> list[float]:
    """Train PPO; optional evaluation runs separately from sampled training returns."""
    if min(cfg.rollout, cfg.updates, cfg.epochs, cfg.minibatch_envs, cfg.log_every, cfg.save_every) < 1 or cfg.critic_warmup < 0:
        raise ValueError("PPO counts must be positive and critic_warmup nonnegative")
    flags = {n: p.requires_grad for n, p in agent.named_parameters()}
    if cfg.critic_warmup and not any(flags[n] for n in flags if n.startswith("value.")):
        raise ValueError("Critic warmup requires trainable value parameters")
    try:
        return run_updates(agent, venv, cfg, torch.device(device), seed, log, evaluate, flags)
    finally:
        for name, param in agent.named_parameters():
            param.requires_grad_(flags[name])


def run_updates(agent, venv, cfg, dev, seed, log, evaluate, flags):
    opt = torch.optim.Adam(param_groups(agent, cfg.lr, cfg.brain_lr_scale, cfg.head_fan_in), eps=1e-5)
    for group in opt.param_groups:
        group["base_lr"] = group["lr"]
    lr_scale, best = 1.0, float("-inf")
    obs, _ = venv.reset(seed=seed)
    h = agent.initial_state(venv.num_envs)
    track = Tracker(log) if log else Tracker()
    if evaluate:
        best = evaluate_checkpoint(agent, evaluate, 0, cfg, track, best)
    for update in range(1, cfg.updates + 1):
        warmup = update <= cfg.critic_warmup
        for name, param in agent.named_parameters():
            param.requires_grad_(flags[name] and (not warmup or name.startswith("value.")))
        frac = 1.0 - (update - 1) / cfg.updates if cfg.anneal_lr else 1.0
        for group in opt.param_groups:
            group["lr"] = group["base_lr"] * frac * lr_scale
        ro, obs, h = collect(agent, venv, obs, h, cfg.rollout, dev, cfg.clip_reward)
        track.returns.extend(ro.returns)
        stats = optimize_rollout(agent, opt, ro, cfg, dev, warmup)
        if cfg.adaptive_lr and cfg.target_kl and not warmup:
            if stats["kl"] > 2 * cfg.target_kl:
                lr_scale = max(cfg.adaptive_lr_floor, lr_scale * 0.5)
            elif stats["kl"] < cfg.target_kl / 2:
                lr_scale = min(1.0, lr_scale * 1.5)
        if update % cfg.log_every == 0 or update == 1:
            track.report(update, update * cfg.rollout * venv.num_envs, **stats, lr=frac * lr_scale)
        if evaluate and (update == cfg.updates or cfg.eval_every and update % cfg.eval_every == 0):
            best = evaluate_checkpoint(agent, evaluate, update, cfg, track, best)
        if cfg.out and (update % cfg.save_every == 0 or update == cfg.updates):
            save_checkpoint(agent, cfg.out, returns=track.returns, config=dataclasses.asdict(cfg), update=update)
    return track.returns
