"""Named neuron subsets of the MaleCNS connectome, to trade biology for speed."""

from __future__ import annotations

from .base import Connectome

SUBSETS = {
    "all": None,
    # brain only: drop the ventral nerve cord (keeps descending neurons as the readout)
    "brain": lambda sc: ~sc.str.startswith("vnc_"),
    # visual pathway: eyes -> optic lobe -> visual projection neurons -> descending neurons
    "visual": lambda sc: sc.isin(["ol_sensory", "ol_intrinsic", "visual_projection", "visual_centrifugal",
                                  "descending_neuron", "cb_intrinsic"]),
    "visual_small": lambda sc: sc.isin(["ol_sensory", "ol_intrinsic", "visual_projection", "visual_centrifugal",
                                        "descending_neuron"]),
}


def select_subset(conn: Connectome, name: str = "all") -> Connectome:
    rule = SUBSETS[name]
    if rule is None:
        return conn
    import numpy as np, torch
    keep = torch.as_tensor(np.flatnonzero(rule(conn.neurons["super_class"]).to_numpy()), dtype=torch.long)
    return conn.subset(keep)
