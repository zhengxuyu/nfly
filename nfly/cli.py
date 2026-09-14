"""Argument groups shared by the scripts, so each script only adds what is specific to it."""

from __future__ import annotations

import argparse
import logging

from .connectome import Connectome, load_malecns, select_subset, SUBSETS


def add_connectome_args(p: argparse.ArgumentParser, subset: str = "all") -> None:
    p.add_argument("--data", default="data", help="directory with the MaleCNS feather files")
    p.add_argument("--subset", default=subset, choices=list(SUBSETS))
    p.add_argument("--min-syn", type=int, default=3, help="drop neuron pairs with fewer synapses")


def connectome_from_args(args: argparse.Namespace) -> Connectome:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    return select_subset(load_malecns(args.data, min_syn=args.min_syn), args.subset)


def agent_kwargs(args: argparse.Namespace) -> dict:
    """FlyAgent.build keyword arguments derived from the agent argument group."""
    kw = {"rnn_steps": args.rnn_steps, "readout_dim": getattr(args, "readout_dim", 32) or None}
    if getattr(args, "freeze_brain", False) or getattr(args, "heads_only", False):
        kw.update(learn_gain=False, learn_alpha=False, learn_bias=False)
    return kw


def apply_freezes(agent, args: argparse.Namespace):
    """Post-build freezes that go beyond the brain: --heads-only leaves only the heads trainable."""
    if getattr(args, "heads_only", False):
        for name, q in agent.named_parameters():
            q.requires_grad_(name.startswith(("value.", "decoder.head", "decoder.mean", "decoder.log_std")))
    return agent


def add_agent_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--rnn-steps", type=int, default=4, help="network steps per env step")
    p.add_argument("--freeze-brain", action="store_true", help="train only encoder, readout and heads; keep the connectome parameters fixed")
    p.add_argument("--readout-dim", type=int, default=32, help="linear bottleneck width between readout neurons and heads (0 = none)")
    p.add_argument("--heads-only", action="store_true", help="train only the policy and value heads; freeze brain, encoder and readout calibration")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
