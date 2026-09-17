"""Run a Miniworld env in its own process.

pyglet's OpenGL context is bound to the thread that created it and, on macOS, must be the
main thread. The viewer steps its env from a background thread and a vector env may step
several from one process, both of which crash the renderer (SIGSEGV). A child process owns
one env and one context; the parent talks to it over a pipe with the plain Gymnasium API.
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Any, Callable

import gymnasium as gym


def _worker(conn, factory: Callable[[], gym.Env]) -> None:
    env = factory()
    conn.send((env.observation_space, env.action_space, env.metadata))
    while True:
        cmd, arg = conn.recv()
        if cmd == "reset":
            conn.send(env.reset(**arg))
        elif cmd == "step":
            conn.send(env.step(arg))
        elif cmd == "render":
            conn.send(env.render())
        elif cmd == "close":
            env.close(); conn.close(); break


class ProcessEnv(gym.Env):
    """A Gymnasium env whose implementation lives in a child process."""

    def __init__(self, factory: Callable[[], gym.Env]):
        ctx = mp.get_context("spawn")
        self._conn, child = ctx.Pipe()
        self._proc = ctx.Process(target=_worker, args=(child, factory), daemon=True)
        self._proc.start()
        self.observation_space, self.action_space, self.metadata = self._conn.recv()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._conn.send(("reset", {"seed": seed, "options": options}))
        return self._conn.recv()

    def step(self, action: Any):
        self._conn.send(("step", action))
        return self._conn.recv()

    def render(self):
        self._conn.send(("render", None))
        return self._conn.recv()

    def close(self):
        if self._proc.is_alive():
            try:
                self._conn.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
            self._proc.join(timeout=5)
