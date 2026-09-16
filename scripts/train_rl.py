"""Train the fly agent with one of the simple trainers (A2C or PPO) on any suite/game.

    python scripts/train_rl.py --algo ppo --suite atari --game pong --subset visual --envs 8 --updates 2000
    python scripts/train_rl.py --algo a2c --suite classic --game cartpole --subset visual_small --envs 16
"""

from __future__ import annotations

import argparse
import json

import torch

from nfly import FlyAgent
from nfly.cli import add_agent_args, add_connectome_args, agent_kwargs, apply_freezes, calibrate_on, connectome_from_args
from nfly.rl import A2CConfig, PPOConfig, train_a2c, train_ppo
from nfly.rl.simple.common import load_checkpoint
from nfly.rl.simple.reference import MLPReference
from nfly.suite import get_suite

TRAINERS = {"a2c": (A2CConfig, train_a2c), "ppo": (PPOConfig, train_ppo)}


def main() -> None:
    p = argparse.ArgumentParser()
    add_connectome_args(p, subset="visual")
    add_agent_args(p)
    p.add_argument("--algo", default="ppo", choices=list(TRAINERS))
    p.add_argument("--model", default="fly", choices=["fly", "mlp"], help="mlp = MLPReference through the same trainer (no connectome)")
    p.add_argument("--mlp-hidden", type=int, nargs="+", default=[64], help="MLP layer widths (one value = two equal layers)")
    p.add_argument("--suite", default="atari")
    p.add_argument("--game", default="pong")
    p.add_argument("--envs", type=int, default=8)
    p.add_argument("--rollout", type=int, help="env steps per segment (BPTT horizon); default per algo")
    p.add_argument("--updates", type=int, default=1000)
    p.add_argument("--lr", type=float)
    p.add_argument("--entropy", type=float)
    p.add_argument("--head-fan-in", type=int, help="PPO: scale head lr by this / n_readout (0 = no scaling)")
    p.add_argument("--no-anneal", action="store_true", help="PPO: keep the learning rate constant")
    p.add_argument("--no-adaptive-lr", action="store_true", help="PPO: no KL-based learning-rate scaling")
    p.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE",
                   help="any trainer config field, e.g. --set gamma=0.99 --set lam=0.95 --set clip=0.1 --set epochs=4 (JSON values)")
    p.add_argument("--out", help="checkpoint path (default runs/<algo>-<suite>-<game>.pt)")
    p.add_argument("--init", help="start from the agent state in this checkpoint (e.g. a behaviour-cloned head from scripts/bc_pong.py)")
    args = p.parse_args()
    torch.manual_seed(args.seed)

    venv = get_suite(args.suite).make_vector(args.game, args.envs, seed=args.seed)
    if args.model == "mlp":
        hidden = args.mlp_hidden[0] if len(args.mlp_hidden) == 1 else args.mlp_hidden
        agent = MLPReference(venv.single_observation_space, venv.single_action_space, hidden).to(args.device)
    else:
        conn = connectome_from_args(args)
        agent = FlyAgent.build(conn, venv.single_observation_space, venv.single_action_space, **agent_kwargs(args)).to(args.device)
        calibrate_on(agent, get_suite(args.suite).make(args.game, seed=args.seed))
    if args.init:
        load_checkpoint(agent, args.init)
        print(f"agent state loaded from {args.init}")
    agent = apply_freezes(agent, args)
    print(agent.summary(), "| trainable parameters:", f"{sum(q.numel() for q in agent.parameters() if q.requires_grad):,}")
    config_cls, train = TRAINERS[args.algo]
    overrides = {k: v for k, v in dict(rollout=args.rollout, lr=args.lr, entropy=args.entropy).items() if v is not None}
    if args.algo == "ppo":
        if args.head_fan_in is not None:
            overrides["head_fan_in"] = args.head_fan_in or None
        if args.no_anneal:
            overrides["anneal_lr"] = False
        if args.no_adaptive_lr:
            overrides["adaptive_lr"] = False
    overrides.update({k: json.loads(v) for k, v in (kv.split("=", 1) for kv in args.set)})
    cfg = config_cls(updates=args.updates, out=args.out or f"runs/{args.algo}-{args.model}-{args.suite}-{args.game}.pt", **overrides)
    train(agent, venv, cfg, device=args.device, seed=args.seed)


if __name__ == "__main__":
    main()
