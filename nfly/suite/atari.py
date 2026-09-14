"""Atari 2600 suite (ALE) with DeepMind-style preprocessing."""

from __future__ import annotations

import gymnasium as gym

from .base import GameSuite, register

GAMES = {
    "pong": "ALE/Pong-v5", "breakout": "ALE/Breakout-v5", "freeway": "ALE/Freeway-v5",
    "boxing": "ALE/Boxing-v5", "enduro": "ALE/Enduro-v5", "spaceinvaders": "ALE/SpaceInvaders-v5",
    "seaquest": "ALE/Seaquest-v5", "beamrider": "ALE/BeamRider-v5", "asteroids": "ALE/Asteroids-v5",
    "mspacman": "ALE/MsPacman-v5", "qbert": "ALE/Qbert-v5",
}


@register("atari")
class AtariSuite(GameSuite):
    def __init__(self, frame_size: int = 84, frame_skip: int = 4, noop_max: int = 30,
                 max_episode_steps: int | None = 27_000, terminal_on_life_loss: bool = False):
        import ale_py
        gym.register_envs(ale_py)
        self.frame_size, self.frame_skip, self.noop_max = frame_size, frame_skip, noop_max
        self.max_episode_steps, self.terminal_on_life_loss = max_episode_steps, terminal_on_life_loss

    def games(self) -> list[str]:
        return list(GAMES)

    def make(self, game, seed=None, render_mode=None, **kw):
        env = gym.make(GAMES.get(game, game), frameskip=1, repeat_action_probability=0.0, full_action_space=False,
                       render_mode=render_mode, max_episode_steps=self.max_episode_steps, **kw)
        env = gym.wrappers.AtariPreprocessing(env, noop_max=self.noop_max, frame_skip=self.frame_skip,
                                              screen_size=self.frame_size, grayscale_obs=True, scale_obs=True,
                                              terminal_on_life_loss=self.terminal_on_life_loss)
        return self.finish(env, seed)

    @staticmethod
    def action_meanings(env: gym.Env) -> list[str]:
        return env.unwrapped.get_action_meanings()
