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


class ReadoutNorm(nn.Module):
    """Per-neuron standardisation with learnable mean and scale, calibrated once at build time.

    Readout neurons sit on a large, nearly constant resting pattern; the observation-dependent
    part of their activity is orders of magnitude smaller. Subtracting each neuron's typical
    activity and dividing by its typical spread removes the pattern, so the heads (and their
    gradients) see the part that carries information. `calibrate` sets both from a probe of
    activities; afterwards they train like any parameter, which keeps every forward pass
    deterministic (no running statistics to drift between rollout, replay and target copies)."""

    def __init__(self, n: int, min_std: float = 1e-4, clip: float = 10.0):
        super().__init__()
        self.min_std, self.clip = min_std, clip
        self.mean = nn.Parameter(torch.zeros(n))
        self.log_scale = nn.Parameter(torch.zeros(n))

    @torch.no_grad()
    def calibrate(self, activity: torch.Tensor) -> None:
        """activity: (M, n) readout activities from a probe of observations."""
        flat = activity.reshape(-1, activity.shape[-1])
        self.mean.copy_(flat.mean(0))
        self.log_scale.copy_(torch.log(flat.std(0).clamp_min(self.min_std)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self.mean) * torch.exp(-self.log_scale)).clamp(-self.clip, self.clip)


class ActionDecoder(nn.Module):
    """Readout neurons -> normalised activity -> (optional) k-dimensional linear bottleneck -> heads.

    The bottleneck (`readout_dim`) is a plain linear map with no activation, so the whole decoder
    stays linear in the readout activity; it only limits the rank of what the heads can use.
    About 1,300 descending neurons carry a task signal that lives in a few dimensions, and
    policy-gradient noise on 1,300 weights per action swamps it; searching in k dimensions is
    far better conditioned. A behaviour-cloned linear head on the frozen network reaches
    500/500 on CartPole, so the information is there; the bottleneck is about finding it."""

    idx: torch.Tensor    # (R,) readout node indices

    def __init__(self, readout_idx: torch.Tensor, readout_dim: int | None = 32):
        super().__init__()
        self.register_buffer("idx", torch.as_tensor(readout_idx, dtype=torch.long))
        self.norm = ReadoutNorm(len(readout_idx))
        self.proj = nn.Linear(len(readout_idx), readout_dim, bias=False) if readout_dim else nn.Identity()
        if readout_dim:
            nn.init.orthogonal_(self.proj.weight)          # rows orthonormal: unit-variance inputs stay unit-variance

    @property
    def n_readout(self) -> int:
        return int(self.idx.numel())

    @property
    def n_features(self) -> int:
        """Width of what the heads see (readout_dim, or the number of readout neurons)."""
        return self.proj.out_features if isinstance(self.proj, nn.Linear) else self.n_readout

    def features(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(h[:, self.idx]))

    def calibrate(self, h: torch.Tensor) -> None:
        """Set the readout normalisation from a probe of states h (M, N)."""
        self.norm.calibrate(h[:, self.idx])

    def dist_inputs(self, feats: torch.Tensor) -> torch.Tensor:
        """Raw distribution parameters (logits, or [mean, log_std]) - what RL libraries want."""
        raise NotImplementedError

    def distribution(self, feats: torch.Tensor) -> Distribution:
        raise NotImplementedError

    def to_env(self, action: torch.Tensor):
        """Tensor action -> what env.step expects (numpy)."""
        return action.detach().cpu().numpy()

    @staticmethod
    def for_space(conn: Connectome, space: gym.Space, readout_idx: torch.Tensor | None = None,
                  readout_dim: int | None = 32) -> "ActionDecoder":
        idx = readout_idx if readout_idx is not None else default_readout_nodes(conn)
        if isinstance(space, gym.spaces.Discrete):
            return DiscreteDecoder(idx, int(space.n), readout_dim)
        if isinstance(space, gym.spaces.Box):
            return BoxDecoder(idx, space, readout_dim)
        raise NotImplementedError(f"unsupported action space {space}")


class DiscreteDecoder(ActionDecoder):
    def __init__(self, readout_idx, n_actions: int, readout_dim: int | None = 32):
        super().__init__(readout_idx, readout_dim)
        self.head = nn.Linear(self.n_features, n_actions)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def dist_inputs(self, feats):
        return self.head(feats)

    def distribution(self, feats):
        return Categorical(logits=self.dist_inputs(feats))


class BoxDecoder(ActionDecoder):
    """Diagonal Gaussian, mean squashed by tanh into the action bounds."""

    def __init__(self, readout_idx, space: gym.spaces.Box, readout_dim: int | None = 32):
        super().__init__(readout_idx, readout_dim)
        d = int(np.prod(space.shape))
        self.shape = space.shape
        self.mean = nn.Linear(self.n_features, d)
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
