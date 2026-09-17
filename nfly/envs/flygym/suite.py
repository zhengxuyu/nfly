"""flygym suite: the MaleCNS brain in NeuroMechFly's body.

    from nfly.suite import get_suite
    env = get_suite("flygym").make("approach", seed=0)          # needs `uv sync --extra embodied`
    env = get_suite("flygym", observation="state").make("walk")

Games are the tasks of `FlyWalkEnv`; observations default to the body's own eye cameras in the
Atari suite's format (grayscale frame plus its change), so the retina, readout and trainers
apply unchanged. Each env runs in its own process: MuJoCo's offscreen GL context, like
pyglet's, is bound to a thread.
"""

from __future__ import annotations

import os
import sys

import gymnasium as gym

from ...suite.atari import TemporalContrast
from ...suite.base import GameSuite, register
from ..miniworld.process_env import ProcessEnv
from .env import TASKS, FlyWalkEnv

GAMES = {t: t for t in TASKS}


def _offscreen_if_no_display() -> None:
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        os.environ.setdefault("MUJOCO_GL", "egl")


@register("flygym")
class FlygymSuite(GameSuite):
    def __init__(self, observation: str = "eyes", frame_size: int = 84, physics_per_step: int = 100,
                 max_steps: int = 400, temporal_contrast: bool = True, isolate: bool = True):
        self.observation, self.frame_size, self.physics_per_step = observation, frame_size, physics_per_step
        self.max_steps, self.temporal_contrast, self.isolate = max_steps, temporal_contrast, isolate

    def games(self) -> list[str]:
        return list(GAMES)

    def make(self, game, seed=None, render_mode=None, **kw):
        factory = _Factory(GAMES.get(game, game), render_mode, self.observation, self.frame_size,
                           self.physics_per_step, self.max_steps, self.temporal_contrast and self.observation == "eyes", kw)
        env = ProcessEnv(factory) if self.isolate else factory()
        return self.finish(env, seed)


class _Factory:
    """Builds the wrapped env; picklable so a child process can call it."""

    def __init__(self, task, render_mode, observation, frame_size, physics_per_step, max_steps, temporal_contrast, kw):
        self.task, self.render_mode, self.observation, self.frame_size = task, render_mode, observation, frame_size
        self.physics_per_step, self.max_steps, self.temporal_contrast, self.kw = physics_per_step, max_steps, temporal_contrast, kw

    def __call__(self) -> gym.Env:
        _offscreen_if_no_display()
        env = FlyWalkEnv(self.task, self.observation, self.frame_size, self.physics_per_step, self.max_steps,
                         render_mode=self.render_mode, **self.kw)
        if self.temporal_contrast:
            env = TemporalContrast(env)
        return env
