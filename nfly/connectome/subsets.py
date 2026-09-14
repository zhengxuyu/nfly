"""Named neuron subsets of the MaleCNS connectome, to trade biology for speed."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import torch

from .base import Connectome

VISUAL_CORE = ["ol_sensory", "ol_intrinsic", "visual_projection", "visual_centrifugal", "descending_neuron"]

SUBSETS: dict[str, Callable[[pd.Series], pd.Series] | None] = {
    "all": None,
    "brain": lambda sc: ~sc.str.startswith("vnc_"),                       # drop the ventral nerve cord
    "visual": lambda sc: sc.isin(VISUAL_CORE + ["cb_intrinsic"]),        # eyes -> optic lobe -> brain -> DNs
    "visual_small": lambda sc: sc.isin(VISUAL_CORE),                      # eyes -> optic lobe -> DNs
}


def select_subset(conn: Connectome, name: str = "all") -> Connectome:
    rule = SUBSETS[name]
    if rule is None:
        return conn
    keep = torch.as_tensor(np.flatnonzero(rule(conn.neurons["super_class"]).to_numpy()), dtype=torch.long)
    return conn.subset(keep)
