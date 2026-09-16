"""Download the official MaleCNS v1.0 anatomy the viewer draws: neuropil meshes and neuron skeletons.

    python scripts/fetch_anatomy.py                       # neuropils + every descending neuron's skeleton
    python scripts/fetch_anatomy.py --per-stage 30        # ... plus 30 random neurons of every other stage
    python scripts/fetch_anatomy.py --bodies 10001 10010  # ... plus these bodies

Sources (public Google Storage bucket of the release):
    rois/fullbrain-roi-v4/mesh/            brain neuropil meshes (neuroglancer legacy format, nm)
    rois/malecns-vnc-neuropil-roi-v0/mesh/ ventral nerve cord neuropil meshes
    v1.0/segmentation/skeletons-malecns/skeletons-swc/<bodyId>.swc   skeletons, 8 nm voxels

Output, under <data>/anatomy/: neuropils.json (decimated meshes in micrometres) and
skeletons/<bodyId>.swc (as downloaded). Both are read by nfly.viz.anatomy.
"""

from __future__ import annotations

import argparse
import base64
import json
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from nfly.cli import add_connectome_args, connectome_from_args
from nfly.viz.anatomy import STAGES, stage_of

BUCKET = "https://storage.googleapis.com/flyem-male-cns"
ROI_VOLUMES = {"brain": "rois/fullbrain-roi-v4", "vnc": "rois/malecns-vnc-neuropil-roi-v0"}
SKELETONS = "v1.0/segmentation/skeletons-malecns/skeletons-swc"
NM_PER_UM = 1000.0
CLUSTER_UM = {"brain": 4.0, "vnc": 6.0}     # vertex-clustering cell size: a translucent shell needs no detail


def fetch(path: str, timeout: float = 120.0) -> bytes:
    with urllib.request.urlopen(f"{BUCKET}/{urllib.parse.quote(path)}", timeout=timeout) as r:
        return r.read()


def roi_names(volume: str) -> list[str]:
    info = json.loads(fetch(f"{volume}/segment_properties/info"))
    return info["inline"]["properties"][0]["values"]


def read_ngmesh(raw: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Neuroglancer legacy mesh: uint32 n, float32 (n, 3) vertices in nm, uint32 (m, 3) faces."""
    n = int(np.frombuffer(raw[:4], np.uint32)[0])
    vertices = np.frombuffer(raw[4:4 + 12 * n], np.float32).reshape(-1, 3) / NM_PER_UM
    faces = np.frombuffer(raw[4 + 12 * n:], np.uint32).reshape(-1, 3)
    return vertices, faces


def decimate(vertices: np.ndarray, faces: np.ndarray, cell: float) -> tuple[np.ndarray, np.ndarray]:
    """Vertex clustering: snap vertices to a grid of `cell` micrometres, merge, drop collapsed faces."""
    keys = np.floor(vertices / cell).astype(np.int64)
    _, new_index, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    new_index = new_index.reshape(-1)
    merged = np.zeros((len(counts), 3), np.float64)
    np.add.at(merged, new_index, vertices)
    merged /= counts[:, None]
    f = new_index[faces]
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    f = np.unique(np.sort(f, axis=1), axis=0)
    return merged.astype(np.float32), f.astype(np.uint32)


def fetch_neuropils(out: Path, regions: list[str]) -> None:
    rois = []
    for region in regions:
        volume = ROI_VOLUMES[region]
        names = roi_names(volume)
        print(f"{region}: {len(names)} neuropils", flush=True)

        def one(name: str) -> dict | None:
            try:
                v, f = read_ngmesh(fetch(f"{volume}/mesh/{name}.ngmesh"))
            except Exception as e:                      # a missing fragment loses one shell, not the run
                print(f"  {name}: skipped ({e})", flush=True); return None
            v, f = decimate(v, f, CLUSTER_UM[region])
            return {"name": name, "region": region, "n_vertices": int(len(v)), "n_faces": int(len(f)),
                    "vertices": b64(v), "faces": b64(f)}

        with ThreadPoolExecutor(8) as pool:
            rois += [r for r in pool.map(one, names) if r]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"units": "um", "source": "MaleCNS v1.0 neuropil ROI meshes", "rois": rois}))
    print(f"wrote {out}: {len(rois)} neuropils, {sum(r['n_faces'] for r in rois):,} faces", flush=True)


def fetch_skeletons(out_dir: Path, bodies: list[int]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [b for b in bodies if not (out_dir / f"{b}.swc").exists()]
    print(f"skeletons: {len(bodies)} requested, {len(todo)} to download", flush=True)

    def one(body: int) -> bool:
        try:
            (out_dir / f"{body}.swc").write_bytes(fetch(f"{SKELETONS}/{body}.swc"))
            return True
        except Exception as e:
            print(f"  {body}: skipped ({e})", flush=True); return False

    with ThreadPoolExecutor(16) as pool:
        got = sum(pool.map(one, todo))
    print(f"downloaded {got} skeletons into {out_dir}", flush=True)


def select_bodies(conn, superclass: list[str], per_stage: int, bodies: list[int], seed: int) -> list[int]:
    nrn = conn.neurons
    chosen = set(bodies) | set(nrn["root_id"][nrn["super_class"].isin(superclass)].tolist())
    if per_stage:
        rng = np.random.default_rng(seed)
        stage = stage_of(nrn)
        for s in range(len(STAGES)):
            ids = nrn["root_id"].to_numpy()[stage == s]
            chosen |= set(rng.choice(ids, min(per_stage, len(ids)), replace=False).tolist())
    return sorted(int(b) for b in chosen)


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def main() -> None:
    p = argparse.ArgumentParser()
    add_connectome_args(p, subset="all")
    p.add_argument("--regions", nargs="+", default=["brain", "vnc"], choices=list(ROI_VOLUMES))
    p.add_argument("--superclass", nargs="+", default=["descending_neuron"], help="download every skeleton of these superclasses")
    p.add_argument("--per-stage", type=int, default=0, help="also N random neurons of every processing stage")
    p.add_argument("--bodies", nargs="*", type=int, default=[], help="also these body ids")
    p.add_argument("--no-neuropils", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    out = Path(args.data) / "anatomy"
    if not args.no_neuropils:
        fetch_neuropils(out / "neuropils.json", args.regions)
    conn = connectome_from_args(args)
    fetch_skeletons(out / "skeletons", select_bodies(conn, args.superclass, args.per_stage, args.bodies, args.seed))


if __name__ == "__main__":
    main()
