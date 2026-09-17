"""NeuroMechFly (flygym 2) as a fly game suite: the MaleCNS brain in the fly's own body
(`uv sync --extra embodied`)."""

from .. import offscreen_gl_if_no_display

offscreen_gl_if_no_display()                                   # before mujoco is imported below

from .env import TASKS, FlyWalkEnv  # noqa: E402
from .suite import GAMES, FlygymSuite  # noqa: E402

__all__ = ["TASKS", "FlyWalkEnv", "GAMES", "FlygymSuite"]
