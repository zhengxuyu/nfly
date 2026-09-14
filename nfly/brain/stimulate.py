"""Stimulus-propagation experiment: drive a set of neurons and see who lights up."""

from __future__ import annotations

import pandas as pd
import torch

from ..connectome.base import Connectome
from .rnn import ConnectomeRNN, InputDrive


@torch.no_grad()
def stimulate(conn: Connectome, model: ConnectomeRNN, stim_idx: torch.Tensor, amplitude: float = 1.0,
              steps: int = 50, top: int = 20, group_by: str = "cell_type") -> dict:
    """Drive `stim_idx` with a constant input and summarise the resulting activity."""
    drive = InputDrive.constant(stim_idx, amplitude, steps)
    hist = model(drive)[0]                          # (T+1, N)
    final = hist[-1].cpu()
    active_per_step = (hist > 1e-3).sum(dim=1).cpu()

    df = conn.neurons[["root_id", group_by, "super_class", "flow", "nt_type"]].copy()
    df["activity"] = final.numpy()
    df["stimulated"] = False
    df.loc[stim_idx.cpu().numpy(), "stimulated"] = True
    downstream = df[~df["stimulated"]]

    top_neurons = downstream.sort_values("activity", ascending=False).head(top)
    by_group = (downstream[downstream[group_by] != ""]
                .groupby(group_by)["activity"].agg(["mean", "max", "count"])
                .sort_values("mean", ascending=False).head(top))
    return {"active_per_step": active_per_step, "top_neurons": top_neurons, "top_groups": by_group,
            "final": final}


def format_report(res: dict, group_by: str = "cell_type") -> str:
    a = res["active_per_step"].tolist()
    lines = [f"active neurons per step: {a[:10]} ... {a[-3:]}",
             "", f"top downstream {group_by}s (mean activity):", res["top_groups"].to_string(),
             "", "top downstream neurons:",
             res["top_neurons"].to_string(index=False, float_format=lambda x: f"{x:.4f}")]
    return "\n".join(lines)
