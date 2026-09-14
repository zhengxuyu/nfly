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


class RunningNorm(nn.Module):
    """Per-neuron standardisation with running statistics, applied identically in train and eval.

    Readout neurons sit on a large, nearly constant resting pattern; the observation-dependent
    part of their activity is orders of magnitude smaller. Normalising each neuron by its own
    running mean and variance removes that pattern, so the heads (and their gradients) see the
    part that carries information. Statistics are a cumulative average for the first
    1/momentum updates (fast, unbiased warm-up), then an exponential moving average; they are
    updated in train mode only. Outputs are clipped to +-clip so an unseen extreme cannot blow
    up the heads."""

    def __init__(self, n: int, momentum: float = 0.01, eps: float = 1e-8, clip: float = 10.0):
        super().__init__()
        self.momentum, self.eps, self.clip = momentum, eps, clip
        self.register_buffer("mean", torch.zeros(n))
        self.register_buffer("var", torch.ones(n))
        self.register_buffer("count", torch.zeros((), dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self._update(x.detach().reshape(-1, x.shape[-1]))
        return ((x - self.mean) / torch.sqrt(self.var + self.eps)).clamp(-self.clip, self.clip)

    @torch.no_grad()
    def _update(self, flat: torch.Tensor) -> None:
        self.count += 1
        m = max(1.0 / float(self.count), self.momentum)
        self.mean.lerp_(flat.mean(0), m)
        self.var.lerp_(flat.var(0, unbiased=False), m)


class ActionDecoder(nn.Module):
    idx: torch.Tensor    # (R,) readout node indices

    def __init__(self, readout_idx: torch.Tensor):
        super().__init__()
        self.register_buffer("idx", torch.as_tensor(readout_idx, dtype=torch.long))
        self.norm = RunningNorm(len(readout_idx))

    @property
    def n_readout(self) -> int:
        return int(self.idx.numel())

    def features(self, h: torch.Tensor) -> torch.Tensor:
        return self.norm(h[:, self.idx])

    def dist_inputs(self, feats: torch.Tensor) -> torch.Tensor:
        """Raw distribution parameters (logits, or [mean, log_std]) - what RL libraries want."""
        raise NotImplementedError

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

    def dist_inputs(self, feats):
        return self.head(feats)

    def distribution(self, feats):
        return Categorical(logits=self.dist_inputs(feats))


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

    def dist_inputs(self, feats):
        mu = torch.tanh(self.mean(feats))
        mu = self.lo + (mu + 1) / 2 * (self.hi - self.lo)
        return torch.cat([mu, self.log_std.expand_as(mu)], dim=-1)

    def distribution(self, feats):
        mu, log_std = self.dist_inputs(feats).chunk(2, dim=-1)
        return Independent(Normal(mu, log_std.exp()), 1)

    def to_env(self, action):
        a = torch.max(torch.min(action, self.hi), self.lo)
        return a.detach().cpu().numpy().reshape(-1, *self.shape)
