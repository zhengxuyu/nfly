"""Action decoders: turn the activity of readout neurons into a Gymnasium action distribution."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Distribution, Independent, Normal

from ..connectome.base import Connectome

READOUT_SUPERCLASSES = ("descending_neuron", "vnc_motor", "cb_motor", "efferent_descending", "descending_neuron_tbc")


def default_readout_nodes(conn: Connectome) -> torch.Tensor:
    """Descending + motor neurons if the table has them, else the connectome's efferent set."""
    idx = conn.where(super_class=list(READOUT_SUPERCLASSES))
    return idx if len(idx) else conn.output_nodes()


class ActionDecoder(nn.Module):
    idx: torch.Tensor    # (R,) readout node indices

    def __init__(self, readout_idx: torch.Tensor):
        super().__init__()
        self.register_buffer("idx", torch.as_tensor(readout_idx, dtype=torch.long))
        self.norm = nn.LayerNorm(len(readout_idx))

    @property
    def n_readout(self) -> int:
        return int(self.idx.numel())

    def features(self, h: torch.Tensor) -> torch.Tensor:
        return self.norm(h[:, self.idx])

    def distribution(self, feats: torch.Tensor) -> Distribution:
        raise NotImplementedError

    def to_env(self, action: torch.Tensor):
        """Tensor action -> what env.step expects (numpy)."""
        return action.detach().cpu().numpy()

    @staticmethod
    def for_space(conn: Connectome, space: gym.Space, readout_idx: torch.Tensor | None = None) -> "ActionDecoder":
        idx = readout_idx if readout_idx is not None else default_readout_nodes(conn)
        if isinstance(space, gym.spaces.Discrete):
            return DiscreteDecoder(idx, int(space.n))
        if isinstance(space, gym.spaces.Box):
            return BoxDecoder(idx, space)
        raise NotImplementedError(f"unsupported action space {space}")


class DiscreteDecoder(ActionDecoder):
    def __init__(self, readout_idx, n_actions: int):
        super().__init__(readout_idx)
        self.head = nn.Linear(self.n_readout, n_actions)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def distribution(self, feats):
        return Categorical(logits=self.head(feats))


class BoxDecoder(ActionDecoder):
    """Diagonal Gaussian, mean squashed by tanh into the action bounds."""

    def __init__(self, readout_idx, space: gym.spaces.Box):
        super().__init__(readout_idx)
        d = int(np.prod(space.shape))
        self.shape = space.shape
        self.mean = nn.Linear(self.n_readout, d)
        self.log_std = nn.Parameter(torch.full((d,), -0.5))
        self.register_buffer("lo", torch.as_tensor(space.low).flatten().float())
        self.register_buffer("hi", torch.as_tensor(space.high).flatten().float())

    def distribution(self, feats):
        mu = torch.tanh(self.mean(feats))
        mu = self.lo + (mu + 1) / 2 * (self.hi - self.lo)
        return Independent(Normal(mu, self.log_std.exp()), 1)

    def to_env(self, action):
        a = torch.max(torch.min(action, self.hi), self.lo)
        return a.detach().cpu().numpy().reshape(-1, *self.shape)
