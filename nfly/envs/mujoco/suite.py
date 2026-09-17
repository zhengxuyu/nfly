"""MuJoCo suite: the fly drives a body (Gymnasium's MuJoCo locomotion and control tasks).

    from nfly.suite import get_suite
    env = get_suite("mujoco").make("ant", seed=0)                 # needs `uv sync --extra embodied`
    env = get_suite("mujoco", observation="pixels").make("ant")   # the fly sees a camera view instead

Observations are the simulator's joint positions and velocities (a vector, standardised by
`GameSuite.finish`), so the VectorEncoder drives sensory neurons and the BoxDecoder reads
continuous torques from the descending and motor neurons. `observation="pixels"` renders a
camera looking at the body every step and hands the fly the Atari-format frame (grayscale plus
its change), so the same retina that plays Pong watches the body move.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from ...suite.atari import TemporalContrast
from ...suite.base import GameSuite, register
from .. import offscreen_gl_if_no_display

GAMES = {
    "ant": "Ant-v5", "halfcheetah": "HalfCheetah-v5", "hopper": "Hopper-v5", "walker": "Walker2d-v5",
    "humanoid": "Humanoid-v5", "swimmer": "Swimmer-v5", "reacher": "Reacher-v5", "pusher": "Pusher-v5",
    "invertedpendulum": "InvertedPendulum-v5", "inverteddoublependulum": "InvertedDoublePendulum-v5",
}


class CameraFrame(gym.ObservationWrapper):
    """Replace the state vector by a rendered camera frame: grayscale (size, size) float32 in
    [0, 1], the Atari suite's format. The env must be made with render_mode="rgb_array"."""

    def __init__(self, env: gym.Env, size: int = 84):
        super().__init__(env)
        self.size = size
        self.observation_space = gym.spaces.Box(0.0, 1.0, (size, size), np.float32)

    def observation(self, _obs):
        rgb = self.env.render()
        gray = rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
        return _resize(gray, self.size) / 255.0


def _resize(frame: np.ndarray, size: int) -> np.ndarray:
    """Area resampling by block means (no image library needed)."""
    h, w = frame.shape
    ys = np.linspace(0, h, size + 1).astype(int)
    xs = np.linspace(0, w, size + 1).astype(int)
    out = np.empty((size, size), np.float32)
    for i in range(size):
        rows = frame[ys[i]:max(ys[i + 1], ys[i] + 1)]
        for j in range(size):
            out[i, j] = rows[:, xs[j]:max(xs[j + 1], xs[j] + 1)].mean()
    return out


@register("mujoco")
class MujocoSuite(GameSuite):
    """Gymnasium's MuJoCo tasks: state observations by default, or a camera frame for the retina."""

    def __init__(self, observation: str = "state", frame_size: int = 84, temporal_contrast: bool = True):
        if observation not in ("state", "pixels"):
            raise ValueError("observation must be 'state' or 'pixels'")
        self.observation, self.frame_size, self.temporal_contrast = observation, frame_size, temporal_contrast

    def games(self) -> list[str]:
        return list(GAMES)

    def make(self, game, seed=None, render_mode=None, **kw):
        pixels = self.observation == "pixels"
        if pixels:
            offscreen_gl_if_no_display()
            render_mode = "rgb_array"
        env = gym.make(GAMES.get(game, game), render_mode=render_mode, **kw)
        if pixels:
            env = CameraFrame(env, self.frame_size)
            if self.temporal_contrast:
                env = TemporalContrast(env)
        return self.finish(env, seed)
