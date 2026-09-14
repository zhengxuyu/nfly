"""Stimulus-propagation experiment: drive a set of neurons and see who lights up."""

from __future__ import annotations

import dataclasses

import pandas as pd
import torch

from ..connectome.base import Connectome
from .rnn import ConnectomeRNN, InputDrive


@dataclasses.dataclass
class StimulationResult:
    final: torch.Tensor              # (N,) activity after the last step
    active_per_step: torch.Tensor    # (T+1,) number of active neurons at each step
    top_neurons: pd.DataFrame        # most active downstream neurons
    top_groups: pd.DataFrame         # most active downstream groups (mean / max / count)

    def report(self, group_by: str = "cell_type") -> str:
        a = self.active_per_step.tolist()
        return "\n".join([f"active neurons per step: {a[:10]} ... {a[-3:]}", "",
                           f"top downstream {group_by}s (mean activity):", self.top_groups.to_string(), "",
                           "top downstream neurons:",
                           self.top_neurons.to_string(index=False, float_format=lambda x: f"{x:.4f}")])


@torch.no_grad()
def stimulate(conn: Connectome, model: ConnectomeRNN, stim_idx: torch.Tensor, amplitude: float = 1.0,
              steps: int = 50, top: int = 20, group_by: str = "cell_type") -> StimulationResult:
    """Drive `stim_idx` with a constant input and summarise the resulting activity."""
    hist = model(InputDrive.constant(stim_idx, amplitude, steps))[0]          # (T+1, N)
    final = hist[-1].cpu()
    df = conn.neurons[["root_id", group_by, "super_class", "flow", "nt_type"]].copy()
    df["activity"] = final.numpy()
    df["stimulated"] = False
    df.loc[stim_idx.cpu().numpy(), "stimulated"] = True
    downstream = df[~df["stimulated"]]
    top_neurons = downstream.sort_values("activity", ascending=False).head(top)
    top_groups = (downstream[downstream[group_by] != ""].groupby(group_by)["activity"]
                  .agg(["mean", "max", "count"]).sort_values("mean", ascending=False).head(top))
    return StimulationResult(final, (hist > 1e-3).sum(dim=1).cpu(), top_neurons, top_groups)
