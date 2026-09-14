"""Drive a set of neurons in the connectome RNN and print who gets activated.

    python scripts/demo_stimulate.py --synthetic                          # no data needed
    python scripts/demo_stimulate.py --class gustatory --steps 60         # all gustatory sensory neurons
    python scripts/demo_stimulate.py --cell-type DNp01 --steps 40         # giant fiber
    python scripts/demo_stimulate.py --root-id 10001 --root-id 10002
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch

from nfly import ConnectomeRNN, load_malecns, stimulate
from nfly.brain import format_report
from nfly.connectome import write_synthetic


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--synthetic", action="store_true", help="use a small random connectome")
    p.add_argument("--min-syn", type=int, default=3)
    p.add_argument("--class", dest="klass", action="append", default=[], help="stimulate all neurons of this class")
    p.add_argument("--cell-type", action="append", default=[])
    p.add_argument("--super-class", action="append", default=[])
    p.add_argument("--root-id", action="append", type=int, default=[])
    p.add_argument("--amplitude", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--scale", type=float, default=2.0, help="global multiplier on normalised weights")
    p.add_argument("--bias", type=float, default=0.0, help="resting drive of every neuron")
    p.add_argument("--group-by", default="cell_type")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    data = write_synthetic(Path("data/synthetic")) if args.synthetic else Path(args.data)
    t0 = time.time()
    conn = load_malecns(data, min_syn=1 if args.synthetic else args.min_syn, cache=not args.synthetic)
    print(conn.summary(), f"(loaded in {time.time() - t0:.1f}s)")

    idx = []
    if args.klass:
        idx.append(conn.where(**{"class": args.klass}))
    if args.cell_type:
        idx.append(conn.where(cell_type=args.cell_type))
    if args.super_class:
        idx.append(conn.where(super_class=args.super_class))
    if args.root_id:
        idx.append(conn.index_of(args.root_id))
    if not idx:
        ct = conn.neurons.loc[conn.neurons["flow"] == "afferent", "cell_type"]
        ct = ct[ct != ""].iloc[0]
        print(f"no stimulus given, using cell_type={ct}")
        idx.append(conn.where(cell_type=ct))
    stim = torch.unique(torch.cat(idx))
    print(f"stimulating {len(stim)} neurons")

    model = ConnectomeRNN(conn, alpha_init=args.alpha, global_scale=args.scale, bias_init=args.bias).to(args.device)
    t0 = time.time()
    res = stimulate(conn, model, stim.to(args.device), args.amplitude, args.steps, group_by=args.group_by)
    print(f"simulated {args.steps} steps in {time.time() - t0:.1f}s\n")
    print(format_report(res, args.group_by))


if __name__ == "__main__":
    main()
