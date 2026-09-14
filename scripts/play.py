"""Let the fly play a game from any suite.

    python scripts/play.py --suite atari --game pong                       # untrained, whole CNS
    python scripts/play.py --suite classic --game cartpole --subset visual
    python scripts/play.py --suite gym --game LunarLander-v3 --checkpoint runs/lander.pt --video videos/
"""

from __future__ import annotations

import argparse
import logging

import gymnasium as gym
import torch

from nfly import FlyAgent, load_malecns, select_subset
from nfly.suite import get_suite, play_episode


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="atari")
    p.add_argument("--game", default="pong")
    p.add_argument("--data", default="data")
    p.add_argument("--subset", default="all")
    p.add_argument("--min-syn", type=int, default=3)
    p.add_argument("--checkpoint")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--rnn-steps", type=int, default=2)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--video", help="directory to write an mp4 of each episode")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    conn = select_subset(load_malecns(args.data, min_syn=args.min_syn), args.subset)
    suite = get_suite(args.suite)
    env = suite.make(args.game, seed=args.seed, render_mode="rgb_array" if args.video else None)
    if args.video:
        env = gym.wrappers.RecordVideo(env, args.video, episode_trigger=lambda e: True, name_prefix=f"fly-{args.game}")
    agent = FlyAgent.build(conn, env.observation_space, env.action_space, rnn_steps=args.rnn_steps).to(args.device)
    if args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint, map_location=args.device)["agent"])
    agent.eval()
    print(agent.summary())

    for ep in range(args.episodes):
        res = play_episode(agent, env, seed=args.seed + ep, max_steps=args.max_steps, greedy=args.greedy, device=args.device)
        print(f"episode {ep}: return {res.ret:.1f} in {res.steps} steps, {res.ms_per_step:.0f} ms/step")
    env.close()


if __name__ == "__main__":
    main()
