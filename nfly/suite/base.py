"""GameSuite: the abstract base every game collection implements, plus a registry.

A suite is anything that can `make(game)` a Gymnasium env.  Swapping suites changes nothing
in the agent, because the agent is built from the env's observation/action spaces.

    from nfly.suite import get_suite
    suite = get_suite("atari")          # or "classic", "gym"
    env = suite.make("pong", seed=0)
    venv = suite.make_vector("pong", 8)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import gymnasium as gym

_REGISTRY: dict[str, type["GameSuite"]] = {}


def register(name: str) -> Callable[[type["GameSuite"]], type["GameSuite"]]:
    def deco(cls):
        cls.name = name
        _REGISTRY[name] = cls
        return cls
    return deco


def get_suite(name: str, **kw) -> "GameSuite":
    if name not in _REGISTRY:
        raise KeyError(f"unknown suite {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kw)


def available_suites() -> list[str]:
    return sorted(_REGISTRY)


def _is_vector_box(space: gym.Space) -> bool:
    return isinstance(space, gym.spaces.Box) and len(space.shape) == 1


class GameSuite(ABC):
    """Subclass and implement `games()` and `make()`; everything else has a default."""

    name: str = "base"

    @abstractmethod
    def games(self) -> list[str]:
        """Game ids this suite knows (short names)."""

    @abstractmethod
    def make(self, game: str, seed: int | None = None, render_mode: str | None = None, **kw) -> gym.Env:
        """Build a single, fully wrapped env whose reset/step follow the Gymnasium API."""

    def make_vector(self, game: str, n_envs: int, seed: int = 0, asynchronous: bool = False, **kw) -> gym.vector.VectorEnv:
        thunks = [self._thunk(game, seed + i, **kw) for i in range(n_envs)]
        cls = gym.vector.AsyncVectorEnv if asynchronous else gym.vector.SyncVectorEnv
        return cls(thunks, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)

    def spaces(self, game: str) -> tuple[gym.Space, gym.Space]:
        env = self.make(game)
        try:
            return env.observation_space, env.action_space
        finally:
            env.close()

    def _thunk(self, game, seed, **kw):
        return lambda: self.make(game, seed=seed, **kw)

    @staticmethod
    def finish(env: gym.Env, seed: int | None, normalize_obs: bool = True) -> gym.Env:
        """Wrap with episode statistics and seed; call at the end of every `make`.

        Vector (non-image) Box observations are standardised with running statistics, so
        dimensions of very different scale (cart position vs pole angle) reach the agent on
        equal footing."""
        if normalize_obs and _is_vector_box(env.observation_space):
            env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if seed is not None:
            env.reset(seed=seed)
            env.action_space.seed(seed)
        return env

    def __repr__(self) -> str:
        return f"{type(self).__name__}(games={self.games()})"


@register("gym")
class GymSuite(GameSuite):
    """Any registered Gymnasium id, unwrapped: `get_suite('gym').make('CartPole-v1')`."""

    def __init__(self, ids: list[str] | None = None, wrappers: list[Callable[[gym.Env], gym.Env]] | None = None):
        self.ids, self.wrappers = ids or [], wrappers or []

    def games(self) -> list[str]:
        return self.ids or sorted(gym.registry)

    def make(self, game, seed=None, render_mode=None, **kw):
        env = gym.make(game, render_mode=render_mode, **kw)
        for w in self.wrappers:
            env = w(env)
        return self.finish(env, seed)
