"""What can be linearly decoded from the fly's readout?  A diagnostic, not a trainer.

Runs a random policy in a game, records the normalised readout activity (all readout neurons)
and the calibrated readout features, and ridge-regresses named quantities of the game state from
each. For Pong the targets come from the ALE RAM: ball x/y, the player's paddle y, and the ball
velocity (frame-to-frame difference), i.e. the motion signal a linear policy needs.

    uv run scripts/probe_readout.py --suite atari --game pong --subset visual --steps 1500
    uv run scripts/probe_readout.py --suite classic --game cartpole --subset visual_small
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from nfly import FlyAgent
from nfly.cli import add_agent_args, add_connectome_args, agent_kwargs, calibrate_on, connectome_from_args
from nfly.suite import get_suite

PONG_RAM = {"ball x": 49, "ball y": 54, "player paddle y": 51, "cpu paddle y": 50}


def targets_of(env, prev: dict | None) -> tuple[dict, dict]:
    """Named state quantities for the current step, plus the previous step's values for velocities."""
    u = env.unwrapped
    if hasattr(u, "ale"):
        ram = u.ale.getRAM()
        cur = {k: float(ram[i]) for k, i in PONG_RAM.items()}
    else:
        cur = {f"state {i}": float(v) for i, v in enumerate(np.asarray(u.state).ravel())}
    vel = {f"d {k}": (cur[k] - prev[k]) if prev is not None else 0.0 for k in list(cur)[:2]}
    return {**cur, **vel}, cur


def ridge_r2(X: torch.Tensor, Y: torch.Tensor, lam: float) -> torch.Tensor:
    n = len(X) // 2
    X1 = torch.cat([X, torch.ones(len(X), 1)], 1)
    W = torch.linalg.solve(X1[:n].T @ X1[:n] + lam * torch.eye(X1.shape[1]), X1[:n].T @ Y[:n])
    pred = X1[n:] @ W
    return 1 - ((pred - Y[n:]) ** 2).sum(0) / ((Y[n:] - Y[n:].mean(0)) ** 2).sum(0)


def main() -> None:
    p = argparse.ArgumentParser()
    add_connectome_args(p, subset="visual")
    add_agent_args(p)
    p.add_argument("--suite", default="atari")
    p.add_argument("--game", default="pong")
    p.add_argument("--steps", type=int, default=1500)
    args = p.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    conn = connectome_from_args(args)
    env = get_suite(args.suite).make(args.game, seed=args.seed)
    agent = FlyAgent.build(conn, env.observation_space, env.action_space, **agent_kwargs(args)).to(args.device).eval()
    calibrate_on(agent, get_suite(args.suite).make(args.game, seed=args.seed + 1000))

    obs, _ = env.reset(); h = agent.initial_state(1); prev = None
    raw, feats, ys = [], [], []
    with torch.no_grad():
        for _ in range(args.steps):
            x = torch.as_tensor(np.asarray(obs), device=args.device).unsqueeze(0)
            f, h = agent.step(x, h)
            y, prev = targets_of(env, prev)
            raw.append(agent.decoder.norm(h[:, agent.decoder.idx])[0].cpu()); feats.append(f[0].cpu()); ys.append(list(y.values()))
            obs, _, term, trunc, _ = env.step(env.action_space.sample())
            if term or trunc:
                obs, _ = env.reset(); h = agent.initial_state(1); prev = None
    names = list(y)
    Y = torch.tensor(ys)
    for label, X, lam in ((f"all {agent.decoder.n_readout} readout neurons", torch.stack(raw), 1e-1),
                          (f"{agent.decoder.n_features} calibrated features", torch.stack(feats), 1e-2)):
        r2 = ridge_r2(X, Y, lam)
        print(f"held-out R^2 from {label}: " + ", ".join(f"{n} {float(v):.2f}" for n, v in zip(names, r2)), flush=True)


if __name__ == "__main__":
    main()
