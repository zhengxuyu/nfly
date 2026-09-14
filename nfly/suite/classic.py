"""Classic-control suite (vector observations) - proves the agent is game-agnostic."""

from __future__ import annotations

import gymnasium as gym

from .base import GameSuite, register

GAMES = {"cartpole": "CartPole-v1", "acrobot": "Acrobot-v1", "mountaincar": "MountainCar-v0",
         "pendulum": "Pendulum-v1", "lunarlander": "LunarLander-v3"}


@register("classic")
class ClassicControlSuite(GameSuite):
    def games(self) -> list[str]:
        return list(GAMES)

    def make(self, game, seed=None, render_mode=None, **kw):
        env = gym.make(GAMES.get(game, game), render_mode=render_mode, **kw)
        return self.finish(env, seed)
