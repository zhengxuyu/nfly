"""Train the fly agent with an RLlib algorithm.

    python scripts/train_rllib.py --algo PPO --suite atari --game pong --subset visual --gpus 1 --iters 200
"""

from __future__ import annotations

import argparse
import json

import ray

from nfly.rl.rllib import build_config


def _num(x):
    """numpy scalars -> plain Python numbers for JSON."""
    return None if x is None else (float(x) if hasattr(x, "__float__") else x)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--algo", default="PPO", help="PPO | APPO | IMPALA")
    p.add_argument("--suite", default="atari")
    p.add_argument("--game", default="pong")
    p.add_argument("--data", default="data")
    p.add_argument("--subset", default="visual")
    p.add_argument("--min-syn", type=int, default=3)
    p.add_argument("--rnn-steps", type=int, default=2)
    p.add_argument("--max-seq-len", type=int, default=32)
    p.add_argument("--env-runners", type=int, default=4)
    p.add_argument("--envs-per-runner", type=int, default=2)
    p.add_argument("--gpus", type=float, default=0)
    p.add_argument("--train-batch", type=int, default=2048)
    p.add_argument("--minibatch", type=int, default=256)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--entropy-coeff", type=float, default=0.01)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--out", default="runs/rllib")
    args = p.parse_args()

    ray.init(ignore_reinit_error=True)
    config = build_config(args.algo, suite=args.suite, game=args.game, data_dir=args.data, subset=args.subset,
                          min_syn=args.min_syn, rnn_steps=args.rnn_steps, max_seq_len=args.max_seq_len,
                          num_env_runners=args.env_runners, num_envs_per_env_runner=args.envs_per_runner,
                          num_gpus=args.gpus, train_batch_size=args.train_batch, minibatch_size=args.minibatch,
                          num_epochs=args.epochs, lr=args.lr, entropy_coeff=args.entropy_coeff)
    algo = config.build_algo()
    for i in range(1, args.iters + 1):
        r = algo.train()
        er = r.get("env_runners", {})
        print(json.dumps({"iter": i, "return_mean": _num(er.get("episode_return_mean")), "episodes": _num(er.get("num_episodes")),
                          "steps": _num(r.get("num_env_steps_sampled_lifetime")), "seconds": _num(r.get("time_this_iter_s"))}), flush=True)
        if i % args.checkpoint_every == 0 or i == args.iters:
            print("checkpoint:", algo.save_to_path(args.out), flush=True)


if __name__ == "__main__":
    main()
