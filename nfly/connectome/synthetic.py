"""Write a small random connectome in the MaleCNS v1.0 feather layout, for tests and dry runs.

Contains a toy optic lobe (photoreceptors -> L1/L2 columns with hex coordinates), a central
brain, descending neurons and a few motor neurons, so every layer of nfly can be exercised
without the 1 GB release.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

NT = ["acetylcholine", "gaba", "glutamate", "dopamine", "serotonin", "octopamine"]
NT_P = [0.55, 0.2, 0.15, 0.05, 0.03, 0.02]


def write_synthetic(data_dir: Path, n_columns: int = 30, n_central: int = 300, n_dn: int = 20, n_motor: int = 10,
                    edges: int = 6000, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    data_dir = Path(data_dir); data_dir.mkdir(parents=True, exist_ok=True)
    rows, w_pre, w_post, w_syn = [], [], [], []
    bid = 10000

    def add(soma=True, **kw):
        nonlocal bid
        bid += 1
        # toy soma positions in 8 nm voxels: left / right optic lobes at the sides, brain in the middle
        x = {"L": 20000, "R": 80000}.get(kw.get("somaSide"), 50000) + int(rng.integers(-8000, 8000))
        loc = [x, int(rng.integers(20000, 40000)), int(rng.integers(20000, 40000))] if soma else None
        rows.append(dict(bodyId=bid, somaLocation=loc, **kw)); return bid

    # optic lobes: one photoreceptor + L1 + L2 per hex column per eye
    columns = []
    for side in ("L", "R"):
        for c in range(n_columns):
            h1, h2 = float(c % 6 + 1), float(c // 6 + 1)
            pr = add(soma=False, superclass="ol_sensory", type="R1-R6", cls="visual", somaSide=side, nt="histamine")
            l1 = add(superclass="ol_intrinsic", type="L1", cls=None, somaSide=side, nt="glutamate", hex1=h1, hex2=h2)
            l2 = add(superclass="ol_intrinsic", type="L2", cls=None, somaSide=side, nt="acetylcholine", hex1=h1, hex2=h2)
            for tgt in (l1, l2):
                w_pre.append(pr); w_post.append(tgt); w_syn.append(int(rng.integers(20, 60)))
            columns += [l1, l2]
    central = [add(superclass="cb_intrinsic", type=f"CB{i % 40}", cls=None, somaSide=rng.choice(["L", "R"]),
                   nt=rng.choice(NT, p=NT_P)) for i in range(n_central)]
    dns = [add(superclass="descending_neuron", type=f"DN{i}", cls=None, somaSide=rng.choice(["L", "R"]),
               nt="acetylcholine") for i in range(n_dn)]
    motors = [add(superclass="vnc_motor", type=f"MN{i}", cls=None, somaSide=rng.choice(["L", "R"]),
                  nt="acetylcholine") for i in range(n_motor)]
    add(superclass=None, type=None, cls=None, somaSide=None, nt=None)      # an unannotated fragment
    # random wiring: columns -> central -> DN -> motor, plus recurrence in central
    src = np.array(columns + central + dns); dst = np.array(central + dns + motors)
    for _ in range(edges):
        w_pre.append(int(rng.choice(src))); w_post.append(int(rng.choice(dst))); w_syn.append(int(rng.integers(1, 40)))

    df = pd.DataFrame(rows)
    pd.DataFrame({"bodyId": df.bodyId, "superclass": df.superclass, "type": df.type, "class": df.cls,
                  "somaSide": df.somaSide, "instance": df.type, "statusLabel": "Roughly traced",
                  "assignedOlHex1": df.get("hex1"), "assignedOlHex2": df.get("hex2"), "somaLocation": df.somaLocation}
                 ).to_feather(data_dir / "body-annotations.feather")
    nt = df[df.nt.notna()]
    pd.DataFrame({"body": nt.bodyId, "consensus_nt": nt.nt, "predicted_nt_confidence": rng.random(len(nt))}
                 ).reset_index(drop=True).to_feather(data_dir / "body-neurotransmitters.feather")
    pd.DataFrame({"body_pre": w_pre, "body_post": w_post, "weight": w_syn}).to_feather(data_dir / "connectome-weights.feather")
    return data_dir
