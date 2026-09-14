"""Connectome container: a neuron table plus edge tensors, independent of the data release.

Mapping (biology -> network):
    neuron id                      -> node index 0..N-1
    connection (pre, post, syn)    -> sparse entry W[post, pre]
    presynaptic transmitter        -> fixed sign of that column (Dale's law)
    syn_count                      -> |W| = syn_count / total input syn of post
    flow (afferent/intrinsic/efferent) -> input / hidden / output node sets
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

# Columns every loader must provide in `neurons`
REQUIRED_COLUMNS = ("root_id", "nt_type", "flow", "super_class", "cell_type", "side")


@dataclasses.dataclass
class Connectome:
    neurons: pd.DataFrame            # index 0..N-1; columns root_id, nt_type, flow, super_class, cell_type, side, ...
    pre: torch.Tensor                # (E,) int64  presynaptic node index
    post: torch.Tensor               # (E,) int64  postsynaptic node index
    syn_count: torch.Tensor          # (E,) float32 synapse count
    sign: torch.Tensor               # (E,) float32 +1 / -1 from presynaptic transmitter
    weight: torch.Tensor             # (E,) float32 syn_count normalised by post total input
    root_ids: np.ndarray             # (N,) int64

    @property
    def n_neurons(self) -> int:
        return len(self.neurons)

    @property
    def n_edges(self) -> int:
        return int(self.pre.numel())

    def index_of(self, root_ids: Iterable[int]) -> torch.Tensor:
        lut = pd.Series(np.arange(self.n_neurons), index=self.root_ids)
        return torch.as_tensor(lut.loc[list(root_ids)].to_numpy(), dtype=torch.long)

    def where(self, **conds) -> torch.Tensor:
        """Node indices matching all column==value (or column in list) conditions."""
        mask = np.ones(self.n_neurons, dtype=bool)
        for col, val in conds.items():
            s = self.neurons[col]
            mask &= s.isin(val).to_numpy() if isinstance(val, (list, tuple, set)) else (s == val).to_numpy()
        return torch.as_tensor(np.flatnonzero(mask), dtype=torch.long)

    def input_nodes(self) -> torch.Tensor:
        return self.where(flow="afferent")

    def output_nodes(self) -> torch.Tensor:
        return self.where(flow="efferent")

    def subset(self, keep: torch.Tensor) -> "Connectome":
        """Restrict to a node subset; weights are re-normalised."""
        keep = torch.as_tensor(keep, dtype=torch.long)
        remap = torch.full((self.n_neurons,), -1, dtype=torch.long)
        remap[keep] = torch.arange(len(keep))
        emask = (remap[self.pre] >= 0) & (remap[self.post] >= 0)
        neurons = self.neurons.iloc[keep.numpy()].reset_index(drop=True)
        return build_connectome(neurons, remap[self.pre[emask]], remap[self.post[emask]],
                                self.syn_count[emask], self.sign[emask])

    def summary(self) -> str:
        flows = self.neurons["flow"].value_counts().to_dict()
        exc = float((self.sign > 0).float().mean())
        return (f"Connectome: {self.n_neurons:,} neurons, {self.n_edges:,} edges, "
                f"{exc:.1%} excitatory edges, flow={flows}")

    def save(self, path: Path) -> None:
        torch.save({"neurons": self.neurons, "pre": self.pre, "post": self.post,
                    "syn_count": self.syn_count, "sign": self.sign, "weight": self.weight}, path)

    @staticmethod
    def load(path: Path) -> "Connectome":
        d = torch.load(path, weights_only=False)
        return Connectome(d["neurons"], d["pre"], d["post"], d["syn_count"], d["sign"], d["weight"],
                          d["neurons"]["root_id"].to_numpy())


def normalise_by_post_input(post: torch.Tensor, syn_count: torch.Tensor, n: int) -> torch.Tensor:
    """w_ij = syn_ij / sum_j syn_ij  (each postsynaptic neuron's inputs sum to 1)."""
    total = torch.zeros(n, dtype=torch.float32).index_add_(0, post, syn_count)
    return syn_count / total[post].clamp_min(1.0)


def build_connectome(neurons: pd.DataFrame, pre, post, syn_count, sign) -> Connectome:
    missing = [c for c in REQUIRED_COLUMNS if c not in neurons.columns]
    if missing:
        raise ValueError(f"neuron table lacks columns {missing}")
    # np.array copies, so read-only views handed out by pandas never back a tensor
    pre = torch.as_tensor(np.array(pre), dtype=torch.long)
    post = torch.as_tensor(np.array(post), dtype=torch.long)
    syn_count = torch.as_tensor(np.array(syn_count), dtype=torch.float32)
    sign = torch.as_tensor(np.array(sign), dtype=torch.float32)
    weight = normalise_by_post_input(post, syn_count, len(neurons))
    return Connectome(neurons, pre, post, syn_count, sign, weight, neurons["root_id"].to_numpy())
