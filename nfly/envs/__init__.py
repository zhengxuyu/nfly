"""Embodied environments: each subpackage wraps one simulator as a GameSuite and registers it
on import (`get_suite` imports `nfly.envs.<name>` on demand). Optional extras; nothing here is
imported by the core package. Available: `miniworld` (first-person 3-D rooms), `mujoco`
(Gymnasium's MuJoCo bodies, state or camera observations), `flygym` (the brain in
NeuroMechFly's body)."""

import os
import sys


def offscreen_gl_if_no_display() -> None:
    """On a Linux box without a display (a training server) MuJoCo and pyglet must render
    offscreen through EGL. Both read their variable at import time, so call this before
    importing them; a desktop keeps its window context."""
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYGLET_HEADLESS", "1")
