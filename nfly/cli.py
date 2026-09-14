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


def add_agent_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--rnn-steps", type=int, default=2, help="network steps per env step")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
