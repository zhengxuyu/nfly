"""A reference policy with the FlyAgent interface but an ordinary MLP inside.

Use it to check that a trainer works before blaming the fly: if the MLP learns a task through
the same code path and the fly does not, the problem is in the model, not the training loop.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from ...interface.decoders import ActionDecoder


class MLPReference(nn.Module):
    """Stateless MLP policy exposing initial_state / forward / act / decoder like FlyAgent."""

    def __init__(self, obs_space: gym.Space, act_space: gym.Space, hidden: int = 64):
        super().__init__()
        d = int(np.prod(obs_space.shape))
        self.body = nn.Sequential(nn.Flatten(), nn.Linear(d, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.decoder = ActionDecoder.for_space(_Stub(hidden), act_space, readout_idx=torch.arange(hidden), readout_dim=None)
        self.decoder.norm = nn.Identity()
        self.value = nn.Linear(hidden, 1)

    def initial_state(self, batch: int) -> torch.Tensor:
        return torch.zeros(batch, 0, device=self.value.weight.device)

    def weights(self):
        return None

    def forward(self, obs: torch.Tensor, h: torch.Tensor, weights=None):
        feats = self.body(obs.float())
        return self.decoder.distribution(feats), self.value(feats).squeeze(-1), h

    def act(self, obs, h, greedy: bool = False):
        with torch.no_grad():
            dist, _, h = self(obs, h)
            a = dist.mode if greedy else dist.sample()
        return self.decoder.to_env(a), h

    def summary(self) -> str:
        return f"MLPReference: {sum(p.numel() for p in self.parameters()):,} parameters"


class _Stub:
    """Minimal stand-in for a Connectome so ActionDecoder.for_space can be reused."""

    def __init__(self, n):
        self.n = n

    def output_nodes(self):
        return torch.arange(self.n)
