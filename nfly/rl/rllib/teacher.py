"""RLlib image teachers with their training-time temporal observation protocol."""

from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from ...suite import get_suite


def teacher_frame(rgb):
    from ray.rllib.env.wrappers.atari_wrappers import resize, rgb2gray
    return resize(rgb2gray(rgb), height=64, width=64).astype(np.float32) / 128.0 - 1.0


class TeacherFrameBuffer(gym.Wrapper):
    """Observe raw frames without changing observations, actions, rewards or resets."""

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

    def teacher_frame(self):
        return np.maximum.reduce(self.frames)


def native_stack(history, frame, repeat_first=False):
    history.append(frame)
    del history[:-4]
    padding = history[0] if repeat_first else torch.zeros_like(frame)
    return torch.stack([padding] * (4 - len(history)) + history, -1)[None]


class CNNTeacher:
    """Use pooled frames and zero history padding; legacy is an explicit audit mode."""

    def __init__(self, checkpoint, device, protocol="pooled"):
        from ray.rllib.core.rl_module.rl_module import RLModule
        if protocol not in ("legacy", "pooled"):
            raise ValueError("Unknown teacher protocol")
        module_dir = Path(checkpoint).resolve() / "learner_group/learner/rl_module/default_policy"
        self.module = RLModule.from_checkpoint(str(module_dir)).to(device).eval()
        self.device, self.protocol, self.stack = device, protocol, []

    def reset(self):
        self.stack = []

    @torch.no_grad()
    def logits_on_frame(self, frame, repeat_first=False):
        small = torch.as_tensor(frame, device=self.device).float()
        obs = native_stack(self.stack, small, repeat_first)
        return self.module.forward_inference({"obs": obs})["action_dist_inputs"][0]

    @torch.no_grad()
    def __call__(self, env):
        legacy = self.protocol == "legacy"
        frame = teacher_frame(env.unwrapped.ale.getScreenRGB()) if legacy else env.get_wrapper_attr("teacher_frame")()
        return int(self.logits_on_frame(frame, repeat_first=legacy).argmax())


def make_teacher_env(game="pong", seed=None, protocol="pooled"):
    if protocol not in ("legacy", "pooled"):
        raise ValueError("Unknown teacher protocol")
    wrappers = (TeacherFrameBuffer,) if protocol == "pooled" else ()
    return get_suite("atari", raw_wrappers=wrappers).make(game, seed=seed)


class TeacherTrainingView(gym.ObservationWrapper):
    """Train the CNN on exactly the pooled input supplied by the teacher observer."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(-1.0, 1.0, (64, 64, 1), np.float32)

    def observation(self, observation):
        return self.env.get_wrapper_attr("teacher_frame")()[..., None]
