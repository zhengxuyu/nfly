"""Fit critics to fixed-policy complete returns, splitting by whole episodes.

This is an offline diagnostic, not a trained policy or an RL benchmark.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from nfly import FlyAgent
from nfly.agent import PixelCritic, value_head
from nfly.cli import add_agent_args, add_connectome_args, agent_kwargs, connectome_from_args
from nfly.rl.simple.common import load_checkpoint, reset_state
from nfly.suite import get_suite


def discounted_returns(rewards, gamma):
    result = torch.empty_like(rewards, dtype=torch.float32)
    tail = 0.0
    for i in reversed(range(len(rewards))):
        tail = float(rewards[i]) + gamma * tail
        result[i] = tail
    return result


@torch.no_grad()
def record_step(agent, obs, h, weights, temperature):
    x = torch.as_tensor(np.asarray(obs), device=h.device)
    features, h = agent.step(x, h, weights)
    dist = torch.distributions.Categorical(logits=agent.decoder.dist_inputs(features) / temperature)
    action = dist.sample()
    return dict(features=features.cpu(), pixels=x.cpu().half(), action=action.cpu(),
                logits=dist.logits.cpu()), h, agent.decoder.to_env(action)


@torch.no_grad()
def collect_episodes(agent, env, seeds, temperature=0.25, max_steps=27000):
    """Record only the first complete episode of each vector slot."""
    torch.manual_seed(seeds[0])
    obs, _ = env.reset(seed=seeds)
    h, weights = agent.initial_state(len(seeds)), agent.weights()
    rows, active = [[] for _ in seeds], np.ones(len(seeds), dtype=bool)
    for step in range(max_steps):
        record, h, action = record_step(agent, obs, h, weights, temperature)
        obs, reward, term, trunc, _ = env.step(action)
        for i in np.flatnonzero(active):
            rows[i].append({**{k: v[i].clone() for k, v in record.items()},
                            "reward": torch.tensor(reward[i], dtype=torch.float32)})
        if (np.asarray(trunc) & active & ~np.asarray(term)).any():
            raise RuntimeError("Truncated episode has no complete Monte Carlo target")
        done = np.asarray(term) | np.asarray(trunc)
        active &= ~done
        h = reset_state(agent, h, torch.as_tensor(done, device=h.device))
        if step % 500 == 0:
            print(f"collect step {step}, active {active.sum()}", flush=True)
        if not active.any():
            return [{"seed": seed, **{k: torch.stack([r[k] for r in episode])
                                      for k in episode[0]}} for seed, episode in zip(seeds, rows)]
    raise RuntimeError("Step cap reached with incomplete episodes")


def split_episodes(episodes, gamma):
    """The final third is split into validation and test, without frame leakage."""
    if len(episodes) < 6 or len({e["seed"] for e in episodes}) != len(episodes):
        raise ValueError("Need at least six episodes with unique seeds")
    n = len(episodes)
    groups = (episodes[:n * 2 // 3], episodes[n * 2 // 3:n * 5 // 6], episodes[n * 5 // 6:])
    return [{**{k: torch.cat([e[k] for e in group]) for k in ("features", "pixels")},
             "target": torch.cat([discounted_returns(e["reward"], gamma) for e in group]),
             "seeds": [e["seed"] for e in group]} for group in groups]


def metrics(pred, target):
    residual = target - pred
    return dict(mse=float(residual.square().mean()),
                ev=float(1 - residual.var(unbiased=False) / target.var(unbiased=False).clamp_min(1e-8)))


@torch.no_grad()
def assess(model, data, key, device):
    pred = torch.cat([model(x.to(device).float()).flatten().cpu() for x in data[key].split(256)])
    return metrics(pred, data["target"])


@dataclass
class FitConfig:
    steps: int = 1000
    batch: int = 256
    lr: float = 0.001
    fan_in: int | None = None
    seed: int = 0
    device: str = "cpu"


def fit_critic(model, splits, key, cfg):
    train, valid, test = splits
    model = model.to(cfg.device)
    groups = [{"params": [p], "lr": cfg.lr * (min(1, cfg.fan_in / p.shape[1])
               if cfg.fan_in and p.ndim == 2 else 1)} for p in model.parameters()]
    optimizer = torch.optim.Adam(groups)
    rng = torch.Generator().manual_seed(cfg.seed)
    best, state, history = float("inf"), copy.deepcopy(model.state_dict()), []
    for step in range(cfg.steps + 1):
        if step % max(1, cfg.steps // 5) == 0 or step == cfg.steps:
            row = dict(step=step, train=assess(model, train, key, cfg.device),
                       validation=assess(model, valid, key, cfg.device))
            history.append(row)
            print(json.dumps(row), flush=True)
            if row["validation"]["mse"] < best:
                best, state = row["validation"]["mse"], copy.deepcopy(model.state_dict())
        if step == cfg.steps:
            break
        idx = torch.randint(len(train["target"]), (cfg.batch,), generator=rng)
        pred = model(train[key][idx].to(cfg.device).float()).flatten()
        loss = (pred - train["target"][idx].to(cfg.device)).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.load_state_dict(state)
    return dict(config=asdict(cfg), history=history, selected_train=assess(model, train, key, cfg.device),
                selected_validation=assess(model, valid, key, cfg.device),
                test=assess(model, test, key, cfg.device))


def collect_data(args):
    env = get_suite(args.suite).make_vector(args.game, args.episodes, seed=args.seed)
    try:
        agent = FlyAgent.build(connectome_from_args(args), env.single_observation_space,
                               env.single_action_space, **agent_kwargs(args)).to(args.device).eval()
        load_checkpoint(agent, args.init)
        episodes = collect_episodes(agent, env, list(range(args.seed, args.seed + args.episodes)),
                                    args.temperature)
        digest = hashlib.sha256()
        with open(args.init, "rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        data = dict(episodes=episodes, checkpoint_sha256=digest.hexdigest(), arguments=vars(args))
        torch.save(data, args.dataset)
        return data
    finally:
        env.close()


def run_fits(data, args):
    splits = split_episodes(data["episodes"], args.gamma)
    width = splits[0]["features"].shape[1]
    report = dict(checkpoint_sha256=data["checkpoint_sha256"], gamma=args.gamma,
                  collection_arguments=data["arguments"],
                  split_seeds=[s["seeds"] for s in splits], fits={},
                  episodes=[dict(seed=e["seed"], steps=len(e["reward"]),
                                 score=float(e["reward"].sum())) for e in data["episodes"]])
    report["constant_baseline"] = [metrics(torch.full_like(s["target"], splits[0]["target"].mean()),
                                           s["target"]) for s in splits]
    for name, key, hidden, lr, fan in (("readout_ppo_rate", "features", 64, 1e-4, 64),
                                       ("readout_fast", "features", 64, 1e-3, None),
                                       ("readout_wide", "features", 256, 1e-3, None),
                                       ("pixels", "pixels", 0, 1e-3, None)):
        torch.manual_seed(0)
        model = value_head(width, hidden) if hidden else PixelCritic(splits[0][key].shape[1:])
        print(name, flush=True)
        cfg = FitConfig(steps=args.fit_steps, lr=lr, fan_in=fan, device=args.device)
        report["fits"][name] = fit_critic(model, splits, key, cfg)
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_connectome_args(p, subset="visual")
    add_agent_args(p)
    p.add_argument("--suite", default="atari")
    p.add_argument("--game", default="pong")
    p.add_argument("--init")
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--temperature", type=float, default=0.25)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--dataset", required=True)
    p.add_argument("--fit-only", action="store_true")
    p.add_argument("--fit-steps", type=int, default=1000)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if args.episodes < 6 or args.temperature <= 0 or not 0 <= args.gamma <= 1:
        p.error("Need episodes >= 6, positive temperature, and gamma in [0, 1]")
    if not args.fit_only and not args.init:
        p.error("Collection requires --init")
    torch.manual_seed(args.seed)
    data = torch.load(args.dataset, weights_only=False, map_location="cpu") if args.fit_only else collect_data(args)
    run_fits(data, args)


if __name__ == "__main__":
    main()
