"""Atari 2600 suite (ALE) with DeepMind-style preprocessing."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from .base import GameSuite, register


class TemporalContrast(gym.ObservationWrapper):
    """Observation = (2, H, W): the frame and its difference from the previous frame.

    Fly photoreceptors and lamina cells respond transiently, i.e. to change, and adapt to
    steady light; a rate model driven by absolute contrast has no such adaptation, and a small
    moving object such as the Pong ball barely registers. Feeding the change as a second
    channel lets the retina encoder add it to the photoreceptor drive (a temporal high-pass)."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        h, w = env.observation_space.shape
        self.observation_space = gym.spaces.Box(-1.0, 1.0, (2, h, w), np.float32)
        self._prev = None

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self._prev = obs
        return np.stack([obs, np.zeros_like(obs)]).astype(np.float32), info

    def observation(self, obs):
        out = np.stack([obs, obs - self._prev]).astype(np.float32)
        self._prev = obs
        return out

GAMES = {
    "pong": "ALE/Pong-v5", "breakout": "ALE/Breakout-v5", "freeway": "ALE/Freeway-v5",
    "boxing": "ALE/Boxing-v5", "enduro": "ALE/Enduro-v5", "spaceinvaders": "ALE/SpaceInvaders-v5",
    "seaquest": "ALE/Seaquest-v5", "beamrider": "ALE/BeamRider-v5", "asteroids": "ALE/Asteroids-v5",
    "mspacman": "ALE/MsPacman-v5", "qbert": "ALE/Qbert-v5",
}


@register("atari")
class AtariSuite(GameSuite):
    def __init__(self, frame_size: int = 84, frame_skip: int = 4, noop_max: int = 30,
                 max_episode_steps: int | None = 27_000, terminal_on_life_loss: bool = False,
                 temporal_contrast: bool = True):
        import ale_py
        gym.register_envs(ale_py)
        self.frame_size, self.frame_skip, self.noop_max = frame_size, frame_skip, noop_max
        self.max_episode_steps, self.terminal_on_life_loss = max_episode_steps, terminal_on_life_loss
        self.temporal_contrast = temporal_contrast

    def games(self) -> list[str]:
        return list(GAMES)

    def make(self, game, seed=None, render_mode=None, **kw):
        env = gym.make(GAMES.get(game, game), frameskip=1, repeat_action_probability=0.0, full_action_space=False,
                       render_mode=render_mode, max_episode_steps=self.max_episode_steps, **kw)
        env = gym.wrappers.AtariPreprocessing(env, noop_max=self.noop_max, frame_skip=self.frame_skip,
                                              screen_size=self.frame_size, grayscale_obs=True, scale_obs=True,
                                              terminal_on_life_loss=self.terminal_on_life_loss)
        if self.temporal_contrast:
            env = TemporalContrast(env)
        return self.finish(env, seed)

    @staticmethod
    def action_meanings(env: gym.Env) -> list[str]:
        return env.unwrapped.get_action_meanings()
