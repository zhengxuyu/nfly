"""Drive a set of neurons in the connectome RNN and print who gets activated.

    python scripts/demo_stimulate.py --synthetic                          # no data needed
    python scripts/demo_stimulate.py --class gustatory --steps 60         # all gustatory sensory neurons
    python scripts/demo_stimulate.py --cell-type DNp01 --steps 40         # giant fiber
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from nfly import ConnectomeRNN, load_malecns, stimulate
from nfly.cli import add_connectome_args, connectome_from_args
from nfly.connectome import write_synthetic


def stimulus_nodes(conn, args) -> torch.Tensor:
    picks = []
    if args.klass:
        picks.append(conn.where(**{"class": args.klass}))
    if args.cell_type:
        picks.append(conn.where(cell_type=args.cell_type))
    if args.root_id:
        picks.append(conn.index_of(args.root_id))
    if not picks:
        sensory_types = conn.neurons.loc[conn.neurons["flow"] == "afferent", "cell_type"]
        first = sensory_types[sensory_types != ""].iloc[0]
        print(f"no stimulus given, using cell_type={first}")
        picks.append(conn.where(cell_type=first))
    return torch.unique(torch.cat(picks))


def main() -> None:
    p = argparse.ArgumentParser()
    add_connectome_args(p)
    p.add_argument("--synthetic", action="store_true", help="use a small random connectome")
    p.add_argument("--class", dest="klass", action="append", default=[], help="stimulate all neurons of this class")
    p.add_argument("--cell-type", action="append", default=[])
    p.add_argument("--root-id", action="append", type=int, default=[])
    p.add_argument("--amplitude", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--scale", type=float, default=2.0, help="global multiplier on normalised weights")
    p.add_argument("--bias", type=float, default=0.0, help="resting drive of every neuron")
    p.add_argument("--group-by", default="cell_type")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    t0 = time.time()
    if args.synthetic:
        conn = load_malecns(write_synthetic(Path("data/synthetic")), min_syn=1, cache=False)
    else:
        conn = connectome_from_args(args)
    print(conn.summary(), f"(loaded in {time.time() - t0:.1f}s)")
    stim = stimulus_nodes(conn, args)
    print(f"stimulating {len(stim)} neurons")

    model = ConnectomeRNN(conn, alpha_init=args.alpha, global_scale=args.scale, bias_init=args.bias).to(args.device)
    t0 = time.time()
    res = stimulate(conn, model, stim.to(args.device), args.amplitude, args.steps, group_by=args.group_by)
    print(f"simulated {args.steps} steps in {time.time() - t0:.1f}s\n")
    print(res.report(args.group_by))


if __name__ == "__main__":
    main()
