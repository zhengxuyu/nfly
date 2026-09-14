"""Map a 2-D image onto the fly's compound eye.

MaleCNS gives every optic-lobe columnar neuron (L1, L2, Mi1, Tm1, ...) a hexagonal column
coordinate (assignedOlHex1/2) per eye.  Photoreceptors themselves carry no coordinate, so each
photoreceptor is placed at the synapse-weighted mean coordinate of its columnar targets.
Hex axial coordinates are converted to 2-D, normalised per eye, and each eye is laid over one
half of the image (left eye = left half).  A frame is then sampled bilinearly at every
photoreceptor position to produce the input drive.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from ..connectome.base import Connectome

PHOTORECEPTOR_TYPES = ("R1-R6", "R7p", "R7y", "R7d", "R8p", "R8y", "R8d", "R7_unclear", "R8_unclear", "R7R8_unclear")
COLUMN_TYPES = ("L1", "L2", "L3", "Mi1", "Tm1", "Tm2", "Tm9", "Mi4", "Mi9", "C2", "C3", "T1", "Dm8", "Tm5a", "Tm5b", "Tm5c", "Tm20")


def hex_to_xy(h1: np.ndarray, h2: np.ndarray) -> np.ndarray:
    """Axial hex coordinates -> 2-D positions (unit column spacing)."""
    return np.stack([h1 + 0.5 * h2, h2 * math.sqrt(3) / 2], axis=1)


def photoreceptor_positions(conn: Connectome, mode: str = "photoreceptors"):
    """Return (node indices, side labels, hex coords) of the retina input neurons.

    mode="photoreceptors": R1-R8 placed via their columnar targets (needs edges).
    mode="columns":        use the L1/L2/L3 lamina cells directly (no edges needed).
    """
    nrn = conn.neurons
    has_hex = (nrn["hex1"] >= 0).to_numpy()
    if mode == "columns":
        m = has_hex & nrn["cell_type"].isin(["L1", "L2", "L3"]).to_numpy()
        idx = np.flatnonzero(m)
        return idx, nrn["side"].to_numpy()[idx], nrn[["hex1", "hex2"]].to_numpy()[idx]

    pr = np.flatnonzero(nrn["cell_type"].isin(PHOTORECEPTOR_TYPES).to_numpy())
    col = has_hex & nrn["cell_type"].isin(COLUMN_TYPES).to_numpy()
    pre, post, syn = conn.pre.numpy(), conn.post.numpy(), conn.syn_count.numpy()
    is_pr = np.zeros(conn.n_neurons, bool); is_pr[pr] = True
    e = is_pr[pre] & col[post]
    hexes = nrn[["hex1", "hex2"]].to_numpy()
    sides = nrn["side"].to_numpy()
    acc = np.zeros((conn.n_neurons, 2)); wsum = np.zeros(conn.n_neurons)
    np.add.at(acc, pre[e], hexes[post[e]] * syn[e, None]); np.add.at(wsum, pre[e], syn[e])
    # side of the photoreceptor: its own somaSide if given, else the side of its strongest target
    side_vote = {}
    for p_, q_, s_ in zip(pre[e], post[e], syn[e]):
        side_vote.setdefault(p_, {}); side_vote[p_][sides[q_]] = side_vote[p_].get(sides[q_], 0) + s_
    placed = pr[wsum[pr] > 0]
    xy = acc[placed] / wsum[placed, None]
    side = np.array([sides[i] if sides[i] in ("L", "R") else max(side_vote[i], key=side_vote[i].get) for i in placed])
    return placed, side, xy


class Retina(nn.Module):
    """Samples frames at the photoreceptor positions.  encode(frames (B,H,W)) -> drive (B,K)."""

    def __init__(self, idx: np.ndarray, side: np.ndarray, hex_xy: np.ndarray, layout: str = "split",
                 mirror_left: bool = True):
        super().__init__()
        xy = hex_to_xy(hex_xy[:, 0], hex_xy[:, 1])
        uv = np.zeros_like(xy)
        for s in ("L", "R"):
            m = side == s
            if not m.any():
                continue
            lo, hi = xy[m].min(0), xy[m].max(0)
            u = (xy[m] - lo) / np.maximum(hi - lo, 1e-6)          # per-eye normalised [0,1]^2
            if s == "L" and mirror_left:
                u[:, 0] = 1 - u[:, 0]
            if layout == "split":                                  # left eye sees the left half
                u[:, 0] = 0.5 * u[:, 0] + (0.0 if s == "L" else 0.5)
            uv[m] = u
        self.register_buffer("idx", torch.as_tensor(idx, dtype=torch.long))
        self.register_buffer("grid", torch.as_tensor(uv * 2 - 1, dtype=torch.float32).view(1, 1, -1, 2))
        self.side = side

    @property
    def n_inputs(self) -> int:
        return int(self.idx.numel())

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: (B, H, W) in [0, 1]  ->  (B, K) contrast signal in [-1, 1]."""
        x = frames.unsqueeze(1).float()                            # (B,1,H,W)
        grid = self.grid.expand(x.shape[0], -1, -1, -1)
        v = nn.functional.grid_sample(x, grid, mode="bilinear", align_corners=True)   # (B,1,1,K)
        return (v.view(x.shape[0], -1) - 0.5) * 2


def build_retina(conn: Connectome, mode: str = "photoreceptors", **kw) -> Retina:
    idx, side, hexes = photoreceptor_positions(conn, mode)
    if len(idx) == 0:
        raise ValueError("no retina neurons found; is this the MaleCNS connectome with hex coordinates?")
    return Retina(idx, side, hexes, **kw)
