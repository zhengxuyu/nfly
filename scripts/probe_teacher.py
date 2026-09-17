"""Compare a fixed CNN teacher's native and transferred observation protocols."""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from scripts.bc_pong import CNNTeacher
from nfly.suite.atari import TemporalContrast


def teacher_frame(rgb):
    from ray.rllib.env.wrappers.atari_wrappers import resize, rgb2gray
    return resize(rgb2gray(rgb), height=64, width=64).astype(np.float32) / 128.0 - 1.0


class TeacherFrameBuffer(gym.Wrapper):
    """Observe raw frames before the fly wrapper repeats actions and downsamples."""

    def __init__(self, env):
        super().__init__(env)
        self.frames = deque(maxlen=2)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames.clear()
        self.frames.append(teacher_frame(obs))
        return obs, info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        self.frames.append(teacher_frame(obs))
        return obs, reward, term, trunc, info

    def pooled(self):
        return np.maximum.reduce(self.frames)


def native_stack(history, frame):
    """RLlib's connector pads absent normalized observations with literal zeros."""
    history.append(frame)
    del history[:-4]
    return torch.stack([torch.zeros_like(frame)] * (4 - len(history)) + history, -1)[None]


@torch.no_grad()
def act_on_frame(teacher, frame):
    small = torch.as_tensor(frame, device=teacher.device).float()
    obs = native_stack(teacher.stack, small)
    return int(teacher.module.forward_inference({"obs": obs})["action_dist_inputs"][0].argmax())


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


def evaluate(teacher, mode, seeds):
    env, source = make_env(mode)
    results = []
    try:
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            teacher.reset()
            total = 0.0
            for step in range(27000):
                if mode == "legacy":
                    action = teacher(env)
                else:
                    frame = obs[..., 0] if mode == "native" else source.pooled()
                    action = act_on_frame(teacher, frame)
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
    p.add_argument("--modes", nargs="+", choices=["legacy", "pooled", "native"],
                   default=["legacy", "pooled", "native"])
    p.add_argument("--out", required=True)
    args = p.parse_args()
    args.teacher = str(Path(args.teacher).resolve())
    teacher = CNNTeacher(args.teacher, args.device)
    report = dict(arguments=vars(args), modes={})
    for mode in args.modes:
        rows = evaluate(teacher, mode, range(args.seed, args.seed + args.episodes))
        complete = [r["score"] for r in rows if r["terminated"]]
        report["modes"][mode] = dict(episodes=rows, observed_mean=float(np.mean([r["score"] for r in rows])),
                                    complete_count=len(complete),
                                    complete_mean=float(np.mean(complete)) if complete else None)
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
