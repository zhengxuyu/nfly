"""Map a 2-D image onto the fly's compound eye.

MaleCNS gives every optic-lobe columnar neuron (L1, L2, Mi1, Tm1, ...) a hexagonal column
coordinate per eye.  Photoreceptors carry no coordinate, so each one is placed at the
synapse-weighted mean coordinate of its columnar targets.  Hex axial coordinates are
converted to 2-D, normalised per eye, and each eye is laid over one half of the image
(left eye = left half).  A frame is then sampled bilinearly at every photoreceptor position.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..connectome.base import Connectome

PHOTORECEPTOR_TYPES = ("R1-R6", "R7p", "R7y", "R7d", "R8p", "R8y", "R8d", "R7_unclear", "R8_unclear", "R7R8_unclear")
COLUMN_TYPES = ("L1", "L2", "L3", "Mi1", "Tm1", "Tm2", "Tm9", "Mi4", "Mi9", "C2", "C3", "T1", "Dm8", "Tm5a", "Tm5b", "Tm5c", "Tm20")
LAMINA_TYPES = ("L1", "L2", "L3")
EYES = ("L", "R")


@dataclasses.dataclass
class RetinaLayout:
    idx: np.ndarray      # (K,) node indices of the input neurons
    side: np.ndarray     # (K,) "L" / "R"
    hex_xy: np.ndarray   # (K, 2) hex axial coordinates


def hex_to_xy(h1: np.ndarray, h2: np.ndarray) -> np.ndarray:
    """Axial hex coordinates -> 2-D positions (unit column spacing)."""
    return np.stack([h1 + 0.5 * h2, h2 * math.sqrt(3) / 2], axis=1)


def column_layout(conn: Connectome) -> RetinaLayout:
    """Use the L1/L2/L3 lamina cells themselves as the retina (no edges needed)."""
    nrn = conn.neurons
    idx = np.flatnonzero(((nrn["hex1"] >= 0) & nrn["cell_type"].isin(LAMINA_TYPES)).to_numpy())
    return RetinaLayout(idx, nrn["side"].to_numpy()[idx], nrn[["hex1", "hex2"]].to_numpy()[idx])


def photoreceptor_layout(conn: Connectome) -> RetinaLayout:
    """Place each photoreceptor at the synapse-weighted mean hex coordinate of its columnar targets."""
    nrn = conn.neurons
    is_pr = nrn["cell_type"].isin(PHOTORECEPTOR_TYPES).to_numpy()
    is_col = ((nrn["hex1"] >= 0) & nrn["cell_type"].isin(COLUMN_TYPES)).to_numpy()
    pre, post, syn = conn.pre.numpy(), conn.post.numpy(), conn.syn_count.numpy()
    e = is_pr[pre] & is_col[post]
    edges = pd.DataFrame({"pr": pre[e], "syn": syn[e], "side": nrn["side"].to_numpy()[post[e]],
                          "x": nrn["hex1"].to_numpy()[post[e]] * syn[e], "y": nrn["hex2"].to_numpy()[post[e]] * syn[e]})
    per_pr = edges.groupby("pr").agg(syn=("syn", "sum"), x=("x", "sum"), y=("y", "sum"))
    target_side = edges.groupby(["pr", "side"])["syn"].sum().unstack(fill_value=0).idxmax(axis=1)
    own_side = pd.Series(nrn["side"].to_numpy(), index=nrn.index).loc[per_pr.index]
    side = own_side.where(own_side.isin(EYES), target_side.loc[per_pr.index]).to_numpy()
    xy = (per_pr[["x", "y"]].to_numpy() / per_pr[["syn"]].to_numpy())
    return RetinaLayout(per_pr.index.to_numpy(), side, xy)


def eye_grid(layout: RetinaLayout, split: bool = True, mirror_left: bool = True) -> np.ndarray:
    """Per-eye normalised image coordinates in [-1, 1]^2 (grid_sample convention)."""
    xy = hex_to_xy(layout.hex_xy[:, 0], layout.hex_xy[:, 1])
    uv = np.zeros_like(xy)
    for eye in EYES:
        m = layout.side == eye
        if not m.any():
            continue
        lo, hi = xy[m].min(0), xy[m].max(0)
        u = (xy[m] - lo) / np.maximum(hi - lo, 1e-6)
        if eye == "L" and mirror_left:
            u[:, 0] = 1 - u[:, 0]
        if split:                                   # left eye sees the left half of the frame
            u[:, 0] = 0.5 * u[:, 0] + (0.0 if eye == "L" else 0.5)
        uv[m] = u
    return uv * 2 - 1


class Retina(nn.Module):
    """Samples frames at the retina positions.  encode(frames (B,H,W)) -> contrast drive (B,K)."""

    def __init__(self, layout: RetinaLayout, split: bool = True, mirror_left: bool = True):
        super().__init__()
        self.side = layout.side
        self.register_buffer("idx", torch.as_tensor(layout.idx, dtype=torch.long))
        grid = torch.as_tensor(eye_grid(layout, split, mirror_left), dtype=torch.float32)
        self.register_buffer("grid", grid.view(1, 1, -1, 2))

    @property
    def n_inputs(self) -> int:
        return int(self.idx.numel())

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """frames (B, H, W) in [0, 1] -> (B, K) contrast in [-1, 1]."""
        x = frames.unsqueeze(1).float()
        grid = self.grid.expand(x.shape[0], -1, -1, -1)
        v = nn.functional.grid_sample(x, grid, mode="bilinear", align_corners=True)
        return (v.view(x.shape[0], -1) - 0.5) * 2


def build_retina(conn: Connectome, mode: str = "photoreceptors", **kw) -> Retina:
    layout = photoreceptor_layout(conn) if mode == "photoreceptors" else column_layout(conn)
    if len(layout.idx) == 0:
        raise ValueError("no retina neurons found; is this a connectome with optic-lobe hex coordinates?")
    return Retina(layout, **kw)
