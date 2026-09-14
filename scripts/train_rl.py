"""Train the fly agent with A2C on any suite/game.

    python scripts/train_rl.py --suite atari --game pong --subset visual --envs 8 --updates 2000
    python scripts/train_rl.py --suite classic --game cartpole --subset visual_small --envs 16
"""

from __future__ import annotations

import argparse
import logging

import torch

from nfly import FlyAgent, load_malecns, select_subset
from nfly.rl import A2CConfig, train_a2c
from nfly.suite import get_suite


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="atari")
    p.add_argument("--game", default="pong")
    p.add_argument("--data", default="data")
    p.add_argument("--subset", default="visual")
    p.add_argument("--min-syn", type=int, default=3)
    p.add_argument("--envs", type=int, default=8)
    p.add_argument("--rnn-steps", type=int, default=2)
    p.add_argument("--rollout", type=int, default=16)
    p.add_argument("--updates", type=int, default=1000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--entropy", type=float, default=0.01)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    torch.manual_seed(args.seed)

    conn = select_subset(load_malecns(args.data, min_syn=args.min_syn), args.subset)
    venv = get_suite(args.suite).make_vector(args.game, args.envs, seed=args.seed)
    agent = FlyAgent.build(conn, venv.single_observation_space, venv.single_action_space, rnn_steps=args.rnn_steps).to(args.device)
    print(agent.summary())
    cfg = A2CConfig(rollout=args.rollout, updates=args.updates, lr=args.lr, entropy=args.entropy,
                    out=args.out or f"runs/{args.suite}-{args.game}.pt")
    train_a2c(agent, venv, cfg, device=args.device, seed=args.seed)


if __name__ == "__main__":
    main()
