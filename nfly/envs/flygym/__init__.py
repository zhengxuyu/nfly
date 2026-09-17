"""NeuroMechFly (flygym 2) as a fly game suite: the MaleCNS brain in the fly's own body
(`uv sync --extra embodied`)."""

from .env import TASKS, FlyWalkEnv
from .suite import GAMES, FlygymSuite

__all__ = ["TASKS", "FlyWalkEnv", "GAMES", "FlygymSuite"]
