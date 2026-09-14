"""Suite layer: Gymnasium game collections behind one abstract base class."""

from .base import GameSuite, GymSuite, available_suites, get_suite, register
from .runner import EpisodeResult, play_episode
from . import classic  # noqa: F401  (registers "classic")

try:  # Atari needs ale-py; keep it optional
    from . import atari  # noqa: F401
except ImportError:  # pragma: no cover
    pass

__all__ = ["GameSuite", "GymSuite", "available_suites", "get_suite", "register", "EpisodeResult", "play_episode"]
