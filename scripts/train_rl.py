"""Train the fly agent with one of the simple trainers (A2C or PPO) on any suite/game.

    python scripts/train_rl.py --algo ppo --suite atari --game pong --subset visual --envs 8 --updates 2000
    python scripts/train_rl.py --algo a2c --suite classic --game cartpole --subset visual_small --envs 16
"""

from __future__ import annotations

import argparse

import torch

from nfly import FlyAgent
from nfly.cli import add_agent_args, add_connectome_args, connectome_from_args
from nfly.rl import A2CConfig, PPOConfig, train_a2c, train_ppo
from nfly.suite import get_suite

TRAINERS = {"a2c": (A2CConfig, train_a2c), "ppo": (PPOConfig, train_ppo)}


def main() -> None:
    p = argparse.ArgumentParser()
    add_connectome_args(p, subset="visual")
    add_agent_args(p)
    p.add_argument("--algo", default="ppo", choices=list(TRAINERS))
    p.add_argument("--suite", default="atari")
    p.add_argument("--game", default="pong")
    p.add_argument("--envs", type=int, default=8)
    p.add_argument("--rollout", type=int, help="env steps per segment (BPTT horizon); default per algo")
    p.add_argument("--updates", type=int, default=1000)
    p.add_argument("--lr", type=float)
    p.add_argument("--entropy", type=float)
    p.add_argument("--out", help="checkpoint path (default runs/<algo>-<suite>-<game>.pt)")
    args = p.parse_args()
    torch.manual_seed(args.seed)

    conn = connectome_from_args(args)
    venv = get_suite(args.suite).make_vector(args.game, args.envs, seed=args.seed)
    agent = FlyAgent.build(conn, venv.single_observation_space, venv.single_action_space, rnn_steps=args.rnn_steps).to(args.device)
    print(agent.summary())
    config_cls, train = TRAINERS[args.algo]
    overrides = {k: v for k, v in dict(rollout=args.rollout, lr=args.lr, entropy=args.entropy).items() if v is not None}
    cfg = config_cls(updates=args.updates, out=args.out or f"runs/{args.algo}-{args.suite}-{args.game}.pt", **overrides)
    train(agent, venv, cfg, device=args.device, seed=args.seed)


if __name__ == "__main__":
    main()
