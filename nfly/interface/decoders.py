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
    """Per-neuron standardisation: calibrated statistics, plus a learnable shift and gain in
    standardised units.

    Readout neurons sit on a large, nearly constant resting pattern; the observation-dependent
    part of their activity is orders of magnitude smaller. Subtracting each neuron's typical
    activity and dividing by its typical spread removes the pattern, so the heads (and their
    gradients) see the part that carries information. `calibrate` sets `mean` and `scale` from
    a probe of activities and they stay fixed; training moves `shift` and `log_gain`, which are
    measured in standard deviations, so an optimizer step of 1e-3 is 1e-3 of a spread. (When the
    raw mean itself was the parameter, one Adam step of 1e-3 was many spreads of a descending
    neuron's activity, saturated the features and turned the policy deterministic in one
    update.) No running statistics, so every forward pass is deterministic."""

    def __init__(self, n: int, min_std: float = 1e-4, clip: float = 10.0):
        super().__init__()
        self.min_std, self.clip = min_std, clip
        self.register_buffer("mean", torch.zeros(n))
        self.register_buffer("scale", torch.ones(n))
        self.shift = nn.Parameter(torch.zeros(n))
        self.log_gain = nn.Parameter(torch.zeros(n))

    @torch.no_grad()
    def calibrate(self, activity: torch.Tensor) -> None:
        """activity: (M, n) readout activities from a probe of observations."""
        flat = activity.reshape(-1, activity.shape[-1])
        self.mean.copy_(flat.mean(0))
        self.scale.copy_(flat.std(0).clamp_min(self.min_std))
        self.shift.zero_(); self.log_gain.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = (x - self.mean) / self.scale
        return ((z - self.shift) * torch.exp(self.log_gain)).clamp(-self.clip, self.clip)


class ActionDecoder(nn.Module):
    """Readout neurons -> normalised activity -> (optional) k-dimensional linear bottleneck -> heads.

    The bottleneck (`readout_dim`) is a plain linear map with no activation, so the whole decoder
    stays linear in the readout activity; it only limits the rank of what the heads can use.
    About 1,300 descending neurons carry a task signal that lives in a few dimensions, and
    policy-gradient noise on 1,300 weights per action swamps it; searching in k dimensions is
    far better conditioned. A behaviour-cloned linear head on the frozen network reaches
    500/500 on CartPole, so the information is there; the bottleneck is about finding it."""

    idx: torch.Tensor    # (R,) readout node indices

    feature_clip: float = 10.0       # features are clipped after the projection as well

    def __init__(self, readout_idx: torch.Tensor, readout_dim: int | None = 32):
        super().__init__()
        self.register_buffer("idx", torch.as_tensor(readout_idx, dtype=torch.long))
        self.norm = ReadoutNorm(len(readout_idx))
        readout_dim = min(readout_dim, len(readout_idx)) if readout_dim else None     # never wider than the readout
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
        return self.proj(self.norm(h[:, self.idx])).clamp(-self.feature_clip, self.feature_clip)

    def calibrate(self, h: torch.Tensor, targets: torch.Tensor | None = None, ridge: float = 0.1) -> float | None:
        """Set the readout normalisation from a probe of states h (M, N) and the bottleneck.

        With `targets` (M, T): ridge-regress the normalised readout onto them and install the
        solution (unit output variance) as the projection, so the heads see a reconstruction of
        what the agent observed (vectors as they are, images as coarse frame and motion maps).
        Returns the R^2 of the fit; None when there is no bottleneck."""
        self.norm.calibrate(h[:, self.idx])
        if targets is None or not isinstance(self.proj, nn.Linear):
            return None
        with torch.no_grad():
            x = self.norm(h[:, self.idx])
            k = min(self.proj.out_features, targets.shape[1])
            y = (targets[:, :k] - targets[:, :k].mean(0)) / (targets[:, :k].std(0) + 1e-6)
            xc = x - x.mean(0)
            gram = xc.T @ xc
            lam = ridge * gram.diagonal().mean()
            w = torch.linalg.solve(gram + lam * torch.eye(gram.shape[0], device=x.device), xc.T @ y)     # (R, k)
            pred = xc @ w
            r2 = float(1 - ((pred - y) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())
            w = w / (pred.std(0) + 1e-6)
            proj = nn.Linear(x.shape[1], k, bias=True).to(x.device)
            proj.weight.copy_(w.T)
            proj.bias.copy_(-(x.mean(0) @ w))
            self.proj = proj
            self._rebuild_heads(k)
        return r2

    def _rebuild_heads(self, n_features: int) -> None:
        """Heads are created for the initial feature width; recreate them when it changes."""
        raise NotImplementedError

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
                  readout_dim: int | None = 32, head_hidden: int = 0) -> "ActionDecoder":
        idx = readout_idx if readout_idx is not None else default_readout_nodes(conn)
        if isinstance(space, gym.spaces.Discrete):
            return DiscreteDecoder(idx, int(space.n), readout_dim, head_hidden)
        if isinstance(space, gym.spaces.Box):
            return BoxDecoder(idx, space, readout_dim)
        raise NotImplementedError(f"unsupported action space {space}")


def policy_head(n_features: int, n_out: int, hidden: int = 0) -> nn.Module:
    """Linear head (the default: the fly's readout neurons decide the action through one
    linear map), or, with `hidden` > 0, a one-hidden-layer tanh MLP. The MLP is a diagnostic
    control only: if it learns where the linear head does not, the readout carries the
    information and linearity is the limit; it is not the model's claim."""
    if hidden:
        head = nn.Sequential(nn.Linear(n_features, hidden), nn.Tanh(), nn.Linear(hidden, n_out))
        nn.init.zeros_(head[-1].weight); nn.init.zeros_(head[-1].bias)
        return head
    head = nn.Linear(n_features, n_out)
    nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)
    return head


class DiscreteDecoder(ActionDecoder):
    def __init__(self, readout_idx, n_actions: int, readout_dim: int | None = 32, head_hidden: int = 0):
        super().__init__(readout_idx, readout_dim)
        self.n_actions, self.head_hidden = n_actions, head_hidden
        self.head = policy_head(self.n_features, n_actions, head_hidden)

    def _rebuild_heads(self, n_features):
        self.head = policy_head(n_features, self.n_actions, self.head_hidden).to(self.idx.device)

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

    def _rebuild_heads(self, n_features):
        self.mean = nn.Linear(n_features, self.mean.out_features).to(self.mean.weight.device)

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
