"""Compare a fixed CNN teacher's native and transferred observation protocols."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from nfly.rl.rllib.teacher import CNNTeacher, TeacherFrameBuffer, native_stack, teacher_frame
from nfly.suite.atari import TemporalContrast


@torch.no_grad()
def act_on_frame(teacher, frame, temperature=0.0):
    small = torch.as_tensor(frame, device=teacher.device).float()
    obs = native_stack(teacher.stack, small)
    logits = teacher.module.forward_inference({"obs": obs})["action_dist_inputs"][0]
    return int(logits.argmax() if temperature == 0 else torch.distributions.Categorical(logits=logits / temperature).sample())


def make_env(mode):
    import ale_py
    gym.register_envs(ale_py)
    raw = gym.make("ALE/Pong-v5", frameskip=1, repeat_action_probability=0.0,
                   full_action_space=False, max_episode_steps=108000)
    if mode == "native":
        from ray.rllib.env.wrappers.atari_wrappers import wrap_atari_for_new_api_stack
        return wrap_atari_for_new_api_stack(raw, framestack=None), None
    if mode == "pooled":
        raw = TeacherFrameBuffer(raw)
        frame_source = raw
    else:
        frame_source = None
    env = gym.wrappers.AtariPreprocessing(raw, noop_max=30, frame_skip=4, screen_size=84,
                                        grayscale_obs=True, scale_obs=True)
    return TemporalContrast(env), frame_source


def evaluate(teacher, mode, seeds, temperature=0.0):
    env, source = make_env(mode)
    results = []
    try:
        for seed in seeds:
            torch.manual_seed(seed)
            obs, _ = env.reset(seed=seed)
            teacher.reset()
            total = 0.0
            for step in range(27000):
                if mode == "legacy":
                    action = teacher(env)
                else:
                    frame = obs[..., 0] if mode == "native" else source.teacher_frame()
                    action = act_on_frame(teacher, frame, temperature)
                obs, reward, term, trunc, _ = env.step(action)
                total += reward
                if term or trunc:
                    break
            else:
                term, trunc = False, True
            row = dict(mode=mode, seed=seed, score=total, steps=step + 1,
                       terminated=bool(term), truncated=bool(trunc))
            results.append(row)
            print(json.dumps(row), flush=True)
    finally:
        env.close()
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=14000)
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--temperature", type=float, default=0.0, help="0 is greedy; positive values sample pooled/native policies")
    p.add_argument("--modes", nargs="+", choices=["legacy", "pooled", "native"],
                   default=["legacy", "pooled", "native"])
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if not math.isfinite(args.temperature) or args.temperature < 0 or args.episodes < 1:
        p.error("Need positive episodes and a finite nonnegative temperature")
    if args.temperature and "legacy" in args.modes:
        p.error("The legacy control is greedy; use --modes pooled native for sampled controls")
    args.teacher = str(Path(args.teacher).resolve())
    teacher = CNNTeacher(args.teacher, args.device, protocol="legacy")
    report = dict(arguments=vars(args), modes={})
    for mode in args.modes:
        rows = evaluate(teacher, mode, range(args.seed, args.seed + args.episodes), args.temperature)
        complete = [r["score"] for r in rows if r["terminated"]]
        report["modes"][mode] = dict(episodes=rows, observed_mean=float(np.mean([r["score"] for r in rows])),
                                    complete_count=len(complete),
                                    complete_mean=float(np.mean(complete)) if complete else None)
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
