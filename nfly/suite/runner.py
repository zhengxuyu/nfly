"""Run a recurrent agent in a single env for one episode."""

from __future__ import annotations

import dataclasses
import time

import numpy as np
import torch


@dataclasses.dataclass
class EpisodeResult:
    ret: float
    steps: int
    seconds: float
    actions: list

    @property
    def ms_per_step(self) -> float:
        return 1000 * self.seconds / max(self.steps, 1)


def play_episode(agent, env, seed: int | None = None, max_steps: int = 10_000, greedy: bool = False,
                 device: str | torch.device = "cpu") -> EpisodeResult:
    obs, _ = env.reset(seed=seed)
    h = agent.initial_state(1)
    total, actions, t0 = 0.0, [], time.time()
    for t in range(max_steps):
        a, h = agent.act(torch.as_tensor(np.asarray(obs), device=device).unsqueeze(0), h, greedy=greedy)
        a = a[0]
        actions.append(a.tolist() if hasattr(a, "tolist") else a)
        obs, r, term, trunc, _ = env.step(a)
        total += float(r)
        if term or trunc:
            break
    return EpisodeResult(total, t + 1, time.time() - t0, actions)
