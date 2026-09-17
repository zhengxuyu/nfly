"""Miniworld suite: first-person 3-D navigation, the closest thing to what a fly does with its eyes.

    import nfly.envs.miniworld                                  # registers the "miniworld" suite
    from nfly.suite import get_suite
    env = get_suite("miniworld").make("hallway", seed=0)      # needs `uv sync --extra embodied`

Miniworld (Farama) renders simple 3-D rooms from the agent's viewpoint: 60 x 80 RGB, discrete
actions (turn left, turn right, move forward, ...), tasks such as reaching a box or following
a hallway. Everything the fly needs is what the Atari suite gives it: a grayscale frame in
[0, 1] plus its change from the previous frame, so the same RetinaEncoder, the same readout
and the same trainers apply without a line of change. This file is the whole integration:
the wrappers that turn Miniworld's RGB into the fly's observation and the suite that registers
the games.
"""

from __future__ import annotations

import os
import sys

import gymnasium as gym
import numpy as np

from ...suite.atari import TemporalContrast
from ...suite.base import GameSuite, register

GAMES = {
    "hallway": "MiniWorld-Hallway-v0",             # walk to the red box at the end of a corridor
    "oneroom": "MiniWorld-OneRoom-v0",             # find the box in a single room
    "tmaze": "MiniWorld-TMaze-v0",                 # choose the arm holding the box
    "fourrooms": "MiniWorld-FourRooms-v0",         # four connected rooms
    "maze": "MiniWorld-Maze-v0",                   # procedurally generated maze
    "collect": "MiniWorld-CollectHealth-v0",       # collect items before health runs out
    "sidewalk": "MiniWorld-Sidewalk-v0",           # follow the sidewalk to the goal
    "putnext": "MiniWorld-PutNext-v0",             # pick a box up and drop it next to another
}


class GrayFrame(gym.ObservationWrapper):
    """RGB (H, W, 3) uint8 -> grayscale (size, size) float32 in [0, 1], the Atari suite's format,
    so the retina samples a square frame like it does for Pong."""

    def __init__(self, env: gym.Env, size: int = 84):
        super().__init__(env)
        self.size = size
        self.observation_space = gym.spaces.Box(0.0, 1.0, (size, size), np.float32)

    def observation(self, obs):
        gray = obs.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
        return _resize(gray, self.size) / 255.0


def _resize(frame: np.ndarray, size: int) -> np.ndarray:
    """Area resampling by index mapping (no cv2 dependency): each output pixel is the mean of its
    source block."""
    h, w = frame.shape
    ys = np.linspace(0, h, size + 1).astype(int)
    xs = np.linspace(0, w, size + 1).astype(int)
    out = np.empty((size, size), np.float32)
    for i in range(size):
        rows = frame[ys[i]:max(ys[i + 1], ys[i] + 1)]
        for j in range(size):
            out[i, j] = rows[:, xs[j]:max(xs[j + 1], xs[j] + 1)].mean()
    return out


class ActionRepeat(gym.Wrapper):
    """Repeat each action `n` times, summing rewards: Miniworld's steps are small turns and
    steps, so one fly decision per 4 frames matches the Atari frame skip."""

    def __init__(self, env: gym.Env, n: int = 4):
        super().__init__(env)
        self.n = n

    def step(self, action):
        total = 0.0
        for _ in range(self.n):
            obs, r, term, trunc, info = self.env.step(action)
            total += float(r)
            if term or trunc:
                break
        return obs, total, term, trunc, info


def _headless_if_no_display() -> None:
    """On a Linux box without a display (a training server) pyglet must render through EGL;
    it reads this variable before its first import. A desktop keeps its window context."""
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        os.environ.setdefault("PYGLET_HEADLESS", "1")


@register("miniworld")
class MiniworldSuite(GameSuite):
    """First-person navigation tasks from Miniworld, observed as the fly observes Atari."""

    def __init__(self, frame_size: int = 84, action_repeat: int = 4, temporal_contrast: bool = True):
        self.frame_size, self.action_repeat, self.temporal_contrast = frame_size, action_repeat, temporal_contrast

    def games(self) -> list[str]:
        return list(GAMES)

    def make(self, game, seed=None, render_mode=None, **kw):
        _headless_if_no_display()
        import miniworld  # noqa: F401  (registers the MiniWorld-* ids)
        env = gym.make(GAMES.get(game, game), render_mode=render_mode, **kw)
        if self.action_repeat > 1:
            env = ActionRepeat(env, self.action_repeat)
        env = GrayFrame(env, self.frame_size)
        if self.temporal_contrast:
            env = TemporalContrast(env)
        return self.finish(env, seed)
