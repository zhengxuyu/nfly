"""Let the fly play a game from any suite.

    python scripts/play.py --suite atari --game pong                       # untrained, whole CNS
    python scripts/play.py --suite classic --game cartpole --subset visual
    python scripts/play.py --suite gym --game LunarLander-v3 --checkpoint runs/lander.pt --video videos/
"""

from __future__ import annotations

import argparse

import gymnasium as gym
import torch

from nfly import FlyAgent
from nfly.cli import add_agent_args, add_connectome_args, agent_kwargs, calibrate_on, connectome_from_args
from nfly.rl.simple.common import load_checkpoint
from nfly.suite import get_suite, play_episode


def main() -> None:
    p = argparse.ArgumentParser()
    add_connectome_args(p)
    add_agent_args(p)
    p.add_argument("--suite", default="atari")
    p.add_argument("--game", default="pong")
    p.add_argument("--checkpoint")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--video", help="directory to write an mp4 of each episode")
    args = p.parse_args()

    conn = connectome_from_args(args)
    env = get_suite(args.suite).make(args.game, seed=args.seed, render_mode="rgb_array" if args.video else None)
    if args.video:
        env = gym.wrappers.RecordVideo(env, args.video, episode_trigger=lambda e: True, name_prefix=f"fly-{args.game}")
    agent = FlyAgent.build(conn, env.observation_space, env.action_space, **agent_kwargs(args)).to(args.device)
    calibrate_on(agent, get_suite(args.suite).make(args.game, seed=args.seed + 1000))
    if args.checkpoint:
        load_checkpoint(agent, args.checkpoint)
    agent.eval()
    print(agent.summary())
    for ep in range(args.episodes):
        res = play_episode(agent, env, seed=args.seed + ep, max_steps=args.max_steps, greedy=args.greedy, device=args.device)
        print(f"episode {ep}: return {res.ret:.1f} in {res.steps} steps, {res.ms_per_step:.0f} ms/step")
    env.close()


if __name__ == "__main__":
    main()
