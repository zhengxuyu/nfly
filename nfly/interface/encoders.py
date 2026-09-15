"""Observation encoders: map a Gymnasium observation onto drive for a set of input neurons.

An encoder owns `idx` (which neurons it drives) and `encode(obs) -> (B, K)`.  The agent
scatters that drive into the full state vector.  Pick one with `ObservationEncoder.for_space`.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from ..connectome.base import Connectome
from .retina import Retina, build_retina


class ObservationEncoder(nn.Module):
    idx: torch.Tensor    # (K,) node indices driven by this encoder

    @property
    def n_inputs(self) -> int:
        return int(self.idx.numel())

    def encode(self, obs: torch.Tensor) -> torch.Tensor:   # (B, ...) -> (B, K)
        raise NotImplementedError

    @staticmethod
    def for_space(conn: Connectome, space: gym.Space, prefer_retina: bool = True, **kw) -> "ObservationEncoder":
        """Choose an encoder for an observation space.

        Image-like Box (H,W) / (H,W,C) / (C,H,W)  -> RetinaEncoder if the connectome has eye
        coordinates, else ImageProjectionEncoder.  Other Box / Discrete -> VectorEncoder onto
        sensory neurons.
        """
        if isinstance(space, gym.spaces.Box) and len(space.shape) in (2, 3) and min(space.shape[-2:]) >= 8:
            if prefer_retina and (conn.neurons.get("hex1", None) is not None) and (conn.neurons["hex1"] >= 0).any():
                try:
                    return RetinaEncoder(conn, space, **kw)
                except ValueError:
                    pass
            return ImageProjectionEncoder(conn, space, **kw)
        return VectorEncoder(conn, space, **kw)


def _to_gray_2d(obs: torch.Tensor, space: gym.spaces.Box) -> torch.Tensor:
    """(B,H,W) | (B,H,W,C) | (B,C,H,W) uint8/float -> (B,H,W) float in [0,1].
    A (B,2,H,W) observation is [frame, change] (TemporalContrast): only the frame is returned."""
    x = obs.float()
    if space.dtype == np.uint8 or float(space.high.max()) > 1.0:
        x = x / 255.0
    if x.dim() == 4:
        if x.shape[1] == 2:
            x = x[:, 0]
        elif x.shape[-1] in (1, 3, 4):
            x = x.mean(-1)
        else:
            x = x.mean(1)
    return x


def _change_2d(obs: torch.Tensor) -> torch.Tensor | None:
    """The change channel of a (B,2,H,W) TemporalContrast observation, else None."""
    return obs.float()[:, 1] if obs.dim() == 4 and obs.shape[1] == 2 else None


def _finite_bounds(space: gym.Space) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """(low, high - low) of a bounded Box, else (None, None)."""
    if isinstance(space, gym.spaces.Box) and np.all(np.isfinite(space.low)) and np.all(np.isfinite(space.high)):
        lo, hi = torch.as_tensor(space.low).flatten().float(), torch.as_tensor(space.high).flatten().float()
        return lo, (hi - lo).clamp_min(1e-6)
    return None, None


def _sensory_nodes(conn: Connectome, n: int | None, seed: int) -> np.ndarray:
    idx = conn.input_nodes().numpy()
    if n is not None and n < len(idx):
        idx = np.random.default_rng(seed).choice(idx, n, replace=False)
    return np.sort(idx)


class RetinaEncoder(ObservationEncoder):
    """Frames sampled at the compound eye's photoreceptor positions (MaleCNS hex columns).

    With a [frame, change] observation the drive is contrast(frame) + surround * (frame minus
    its local mean) + temporal_gain * change, all sampled at the same positions: a spatial and a
    temporal high-pass, standing in for the antagonistic surround and the transient response of
    real photoreceptors and lamina cells. Defaults come from a probe sweep on Pong (ball y /
    vertical velocity at the descending neurons): plain contrast 0.45 / 0.41; temporal gain 4 ->
    0.64 / 0.56; temporal gain 8 + surround 4 -> 0.72 / 0.65; plus both eyes sampling the whole
    frame (Retina split=False) -> 0.86 / 0.68."""

    def __init__(self, conn: Connectome, space: gym.spaces.Box, mode: str = "photoreceptors",
                 temporal_gain: float = 8.0, surround: float = 4.0, surround_size: int = 9, **kw):
        """surround: weight of a centre-surround term, frame minus its local mean over a
        surround_size x surround_size window, added to the drive. Real lamina cells (L1/L2)
        have antagonistic surrounds; this makes small objects stand out against uniform
        backgrounds. 0 disables it."""
        super().__init__()
        self.space, self.temporal_gain, self.surround, self.surround_size = space, temporal_gain, surround, surround_size
        self.retina: Retina = build_retina(conn, mode=mode, **kw)
        self.register_buffer("idx", self.retina.idx.clone())

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        frame = _to_gray_2d(obs, self.space)
        drive = self.retina.encode(frame)
        if self.surround:
            local_mean = nn.functional.avg_pool2d(frame.unsqueeze(1), self.surround_size, stride=1,
                                                 padding=self.surround_size // 2, count_include_pad=False).squeeze(1)
            drive = drive + self.surround * self.retina.sample(frame - local_mean)
        change = _change_2d(obs)
        if change is not None:
            drive = drive + self.temporal_gain * self.retina.sample(change)
        return drive


class ImageProjectionEncoder(ObservationEncoder):
    """Fallback for connectomes without eye coordinates: pool the image to a coarse grid and
    project it linearly (learnable) onto sensory neurons."""

    def __init__(self, conn: Connectome, space: gym.spaces.Box, grid: int = 12, n_inputs: int | None = 2000, seed: int = 0):
        super().__init__()
        self.space, self.grid = space, grid
        self.register_buffer("idx", torch.as_tensor(_sensory_nodes(conn, n_inputs, seed), dtype=torch.long))
        self.proj = nn.Linear(grid * grid, self.n_inputs)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        x = _to_gray_2d(obs, self.space).unsqueeze(1)
        x = nn.functional.adaptive_avg_pool2d(x, self.grid).flatten(1) - 0.5
        return self.proj(x)


class VectorEncoder(ObservationEncoder):
    """Box (any shape, flattened) or Discrete (one-hot) observation -> learnable linear map onto sensory neurons."""

    def __init__(self, conn: Connectome, space: gym.Space, n_inputs: int | None = 512, seed: int = 0):
        super().__init__()
        self.space = space
        self.d = int(space.n) if isinstance(space, gym.spaces.Discrete) else int(np.prod(space.shape))
        self.register_buffer("idx", torch.as_tensor(_sensory_nodes(conn, n_inputs, seed), dtype=torch.long))
        self.proj = nn.Linear(self.d, self.n_inputs)
        lo, scale = _finite_bounds(space)
        self.register_buffer("lo", lo)
        self.register_buffer("scale", scale)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        if isinstance(self.space, gym.spaces.Discrete):
            x = nn.functional.one_hot(obs.long(), self.d).float()
        else:
            x = obs.float().flatten(1)
            if self.lo is not None:
                x = (x - self.lo) / self.scale * 2 - 1
        return self.proj(x)
