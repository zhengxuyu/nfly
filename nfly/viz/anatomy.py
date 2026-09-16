"""The brain as a point cloud in MaleCNS coordinates, and activity in units the eye can read.

Every annotated neuron with a soma has an official position (micrometres in the EM space);
neurons without one (photoreceptors and other sensory neurons, whose somata lie outside the
volume) are placed at the synapse-weighted mean of their partners. Activity is shown as a
z-score against a random-policy rollout: (h - mean) / spread per neuron, so a descending
neuron whose whole dynamic range is 1e-6 lights up as brightly as a photoreceptor."""

from __future__ import annotations

import base64
import dataclasses
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from ..connectome import Connectome

STAGES = ["photoreceptors", "lamina", "medulla", "lobula / motion", "visual projection", "central brain",
          "descending", "VNC / motor", "other sensory"]
STAGE_COLORS = ["#ffd166", "#f4a261", "#e76f51", "#d62828", "#9b5de5", "#59c2ff", "#aad94c", "#95e6cb", "#8b93a7"]
LAMINA_PREFIXES = ("L1", "L2", "L3", "L4", "L5", "C2", "C3", "T1", "Lawf", "Lai")
LOBULA_PREFIXES = ("T4", "T5", "T2", "T3", "Li", "LPi", "Tlp", "LPT", "LoVP", "TmY", "Y")
Z_CLIP = 4.0


def stage_of(neurons) -> np.ndarray:
    """Coarse processing stage of every neuron (index into STAGES), from superclass and type."""
    sc, flow = neurons["super_class"].to_numpy(), neurons["flow"].to_numpy()
    ct = neurons["cell_type"].fillna("").astype(str).to_numpy()
    stage = np.full(len(neurons), STAGES.index("VNC / motor"), dtype=np.int8)
    stage[flow == "afferent"] = STAGES.index("other sensory")
    stage[sc == "ol_sensory"] = STAGES.index("photoreceptors")
    ol = sc == "ol_intrinsic"
    stage[ol] = STAGES.index("medulla")
    stage[ol & _starts_with(ct, LAMINA_PREFIXES)] = STAGES.index("lamina")
    stage[ol & _starts_with(ct, LOBULA_PREFIXES)] = STAGES.index("lobula / motion")
    stage[np.isin(sc, ["visual_projection", "visual_centrifugal"])] = STAGES.index("visual projection")
    stage[np.isin(sc, ["cb_intrinsic", "ascending_neuron"])] = STAGES.index("central brain")
    stage[sc == "descending_neuron"] = STAGES.index("descending")
    return stage


def _starts_with(types: np.ndarray, prefixes: tuple[str, ...]) -> np.ndarray:
    return np.array([str(t).startswith(prefixes) for t in types], dtype=bool)


def neuron_positions(conn: Connectome, passes: int = 3) -> np.ndarray:
    """(N, 3) positions in micrometres; neurons without a soma take the synapse-weighted mean
    position of their postsynaptic partners (then presynaptic ones), repeated `passes` times so
    chains of unplaced neurons resolve. Still-unplaced neurons stay NaN."""
    xyz = conn.neurons[["x", "y", "z"]].to_numpy(dtype=np.float32).copy()
    pre, post, syn = conn.pre.numpy(), conn.post.numpy(), conn.syn_count.numpy()
    for _ in range(passes):
        for src, dst in ((post, pre), (pre, post)):          # place src-side neurons from dst-side partners
            missing = np.isnan(xyz[:, 0])
            if not missing.any():
                return xyz
            use = missing[src] & ~np.isnan(xyz[dst, 0])
            acc = np.zeros_like(xyz); wsum = np.zeros(len(xyz), dtype=np.float32)
            np.add.at(acc, src[use], xyz[dst[use]] * syn[use, None]); np.add.at(wsum, src[use], syn[use])
            placed = missing & (wsum > 0)
            xyz[placed] = acc[placed] / wsum[placed, None]
    return xyz


@dataclasses.dataclass
class BrainAtlas:
    """What the page needs once: where the rendered neurons are and what stage they belong to."""
    index: np.ndarray        # (M,) node indices of the rendered sample
    xyz: np.ndarray          # (M, 3) float32 micrometres
    stage: np.ndarray        # (M,) int8 into STAGES
    retina_xy: np.ndarray    # (K, 2) float32 photoreceptor sampling positions in the frame, [-1, 1]
    retina_eye: np.ndarray   # (K,) uint8 0 = left, 1 = right
    body: np.ndarray         # (M,) int64 MaleCNS body ids
    type_index: np.ndarray   # (M,) int32 into `types`
    types: list[str]         # cell type names of the sample

    def to_dict(self) -> dict:
        return {"n": int(len(self.index)), "xyz": _b64(self.xyz), "stage": _b64(self.stage),
                "stages": STAGES, "colors": STAGE_COLORS, "counts": np.bincount(self.stage, minlength=len(STAGES)).tolist(),
                "retina_xy": _b64(self.retina_xy), "retina_eye": _b64(self.retina_eye),
                "body": self.body.tolist(), "type_index": _b64(self.type_index), "types": self.types}


def build_atlas(conn: Connectome, encoder, max_points: int = 30000, seed: int = 0,
                keep: np.ndarray | None = None) -> BrainAtlas:
    """Sample up to max_points placed neurons, keeping every photoreceptor and descending neuron
    and the node indices in `keep` (neurons with a skeleton)."""
    xyz, stage = neuron_positions(conn), stage_of(conn.neurons)
    placed = np.flatnonzero(~np.isnan(xyz[:, 0]))
    keep_all = np.isin(stage[placed], [STAGES.index("photoreceptors"), STAGES.index("descending")])
    if keep is not None:
        keep_all |= np.isin(placed, keep)
    rest = placed[~keep_all]
    rng = np.random.default_rng(seed)
    budget = max(0, max_points - int(keep_all.sum()))
    if len(rest) > budget:
        rest = rng.choice(rest, budget, replace=False)
    index = np.sort(np.concatenate([placed[keep_all], rest]))
    retina_xy, retina_eye = _retina_layout(encoder)
    cell_types = conn.neurons["cell_type"].fillna("").astype(str).to_numpy()[index]
    types, type_index = np.unique(np.where(cell_types == "", "(untyped)", cell_types), return_inverse=True)
    return BrainAtlas(index, xyz[index], stage[index], retina_xy, retina_eye,
                      conn.root_ids[index].astype(np.int64), type_index.astype(np.int32), types.tolist())


def _retina_layout(encoder) -> tuple[np.ndarray, np.ndarray]:
    retina = getattr(encoder, "retina", None)
    if retina is None:
        return np.zeros((0, 2), np.float32), np.zeros(0, np.uint8)
    xy = retina.grid.reshape(-1, 2).detach().cpu().numpy().astype(np.float32)
    eye = (np.asarray(retina.side) == "R").astype(np.uint8)
    return xy, eye


@dataclasses.dataclass
class ActivityScale:
    """Per-neuron mean and spread from a random-policy rollout, used to turn states into z-scores."""
    mean: torch.Tensor       # (N,)
    spread: torch.Tensor     # (N,)
    drive_scale: float       # typical |input current|, for the retina panel

    def z(self, h: torch.Tensor) -> torch.Tensor:
        return ((h - self.mean) / self.spread).clamp(-Z_CLIP, Z_CLIP)


def calibrate_activity(agent, env: gym.Env, steps: int = 96, seed: int = 0, min_spread: float = 1e-6) -> ActivityScale:
    dev = agent.input_gain.device
    obs, _ = env.reset(seed=seed)
    h = agent.initial_state(1)
    states, drives = [], []
    with torch.no_grad():
        weights = agent.weights()
        for _ in range(steps):
            x = torch.as_tensor(np.asarray(obs), device=dev).float().unsqueeze(0)
            drives.append(agent.drive(x).abs().max())
            _, h = agent.step(x, h, weights)
            states.append(h)
            obs, _, term, trunc, _ = env.step(env.action_space.sample())
            if term or trunc:
                obs, _ = env.reset(); h = agent.initial_state(1)
        states = torch.cat(states)
        drive_scale = float(torch.stack(drives).max().clamp_min(1e-6))
    return ActivityScale(states.mean(0), states.std(0).clamp_min(min_spread), drive_scale)


@dataclasses.dataclass
class BrainActivity:
    """One env step of activity, quantised for the wire: the sampled cloud at the last sub-step,
    mean |z| per stage at every sub-step, and the input current per photoreceptor."""
    cloud: np.ndarray        # (M,) uint8, 128 = resting
    stages: list[list[float]]  # [sub-step][stage] mean |z|
    retina: np.ndarray       # (K,) uint8, 128 = no current
    top_types: list[list]    # [[cell type, mean |z|, n drawn], ...] most responsive types this frame

    def to_dict(self) -> dict:
        return {"cloud": _b64(self.cloud), "stages": self.stages, "retina": _b64(self.retina), "top_types": self.top_types}


def summarise(states: list[torch.Tensor], drive: torch.Tensor, atlas: BrainAtlas, scale: ActivityScale) -> BrainActivity:
    """states: per-sub-step (1, N) tensors from FlyAgent.trace; drive: (K,) input current."""
    index = torch.as_tensor(atlas.index, device=states[-1].device)
    stage = torch.as_tensor(atlas.stage, device=states[-1].device, dtype=torch.long)
    n_stages = len(STAGES)
    per_step = []
    for h in states:
        z = scale.z(h[0])
        sums = torch.zeros(n_stages, device=z.device).index_add_(0, stage, z[index].abs())
        counts = torch.bincount(stage, minlength=n_stages).clamp_min(1)
        per_step.append((sums / counts).tolist())
    z_last = scale.z(states[-1][0])[index]
    cloud = _quantise(z_last, Z_CLIP)
    retina = _quantise(drive, scale.drive_scale)
    return BrainActivity(cloud, per_step, retina, top_cell_types(z_last.abs(), atlas))


def top_cell_types(abs_z: torch.Tensor, atlas: BrainAtlas, k: int = 12, min_count: int = 5) -> list[list]:
    """Cell types with the highest mean |z| among the drawn neurons (at least min_count drawn)."""
    ti = torch.as_tensor(atlas.type_index, device=abs_z.device, dtype=torch.long)
    n_types = len(atlas.types)
    sums = torch.zeros(n_types, device=abs_z.device).index_add_(0, ti, abs_z)
    counts = torch.bincount(ti, minlength=n_types)
    mean = torch.where(counts >= min_count, sums / counts.clamp_min(1), torch.zeros_like(sums))
    top = torch.topk(mean, min(k, n_types))
    return [[atlas.types[int(i)], round(float(v), 3), int(counts[int(i)])] for v, i in zip(top.values, top.indices) if v > 0]


def _quantise(x: torch.Tensor, full_scale: float) -> np.ndarray:
    return ((x / full_scale).clamp(-1, 1) * 127 + 128).round().to(torch.uint8).cpu().numpy()


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


# ---- official anatomy: neuropil meshes and neuron skeletons (scripts/fetch_anatomy.py) ---------

SKELETON_MAX_NODES = 400
VOXEL_UM = 0.008             # SWC coordinates are 8 nm voxels


@dataclasses.dataclass
class Skeleton:
    body: int
    node: int                # connectome node index
    xyz: np.ndarray          # (n, 3) float32 micrometres
    parent: np.ndarray       # (n,) int32, -1 at the root

    def to_dict(self, atlas_pos: int, stage: int) -> dict:
        return {"body": self.body, "atlas": atlas_pos, "stage": stage, "n": int(len(self.xyz)),
                "xyz": _b64(self.xyz), "parent": _b64(self.parent)}


def read_swc(path: Path, max_nodes: int = SKELETON_MAX_NODES) -> tuple[np.ndarray, np.ndarray]:
    """SWC -> (xyz in micrometres, parent index); pruned to about max_nodes keeping every root,
    tip and branch point plus every k-th node along the chains between them."""
    rows = np.loadtxt(path, comments="#", ndmin=2)
    ids = rows[:, 0].astype(np.int64)
    lut = {int(i): k for k, i in enumerate(ids)}
    parent = np.array([lut.get(int(pid), -1) for pid in rows[:, 6]], dtype=np.int32)
    xyz = (rows[:, 2:5] * VOXEL_UM).astype(np.float32)
    if len(xyz) <= max_nodes:
        return xyz, parent
    degree = np.bincount(parent[parent >= 0], minlength=len(xyz)) + (parent >= 0)
    stride = int(np.ceil(len(xyz) / max_nodes))
    keep = (degree != 2) | (np.arange(len(xyz)) % stride == 0)
    ancestor = parent.copy()                             # nearest kept ancestor of every node
    for _ in range(stride + 1):
        hop = (ancestor >= 0) & ~keep[np.maximum(ancestor, 0)]
        if not hop.any():
            break
        ancestor[hop] = parent[ancestor[hop]]
    new_index = np.cumsum(keep) - 1
    kept_parent = np.where(ancestor[keep] >= 0, new_index[np.maximum(ancestor[keep], 0)], -1).astype(np.int32)
    return xyz[keep], kept_parent


@dataclasses.dataclass
class AnatomyAssets:
    """Official meshes and skeletons found under <data>/anatomy, or empty when not fetched."""
    neuropils: list[dict]              # entries of neuropils.json (already base64)
    skeletons: list[Skeleton]

    @property
    def skeleton_nodes(self) -> np.ndarray:
        return np.array([s.node for s in self.skeletons], dtype=np.int64)

    def meshes_dict(self) -> dict:
        return {"units": "um", "rois": self.neuropils}

    def skeletons_dict(self, atlas: BrainAtlas) -> dict:
        """Skeletons with their position in the atlas sample (-1 if the neuron could not be placed)."""
        pos = {int(n): i for i, n in enumerate(atlas.index)}
        out = []
        for s in self.skeletons:
            i = pos.get(s.node, -1)
            out.append(s.to_dict(i, int(atlas.stage[i]) if i >= 0 else len(STAGES) - 1))
        return {"units": "um", "neurons": out}


def load_assets(anatomy_dir: str | Path, conn: Connectome) -> AnatomyAssets:
    anatomy_dir = Path(anatomy_dir)
    meshes = anatomy_dir / "neuropils.json"
    neuropils = json.loads(meshes.read_text())["rois"] if meshes.exists() else []
    lut = {int(b): i for i, b in enumerate(conn.root_ids)}
    skeletons = []
    for swc in sorted((anatomy_dir / "skeletons").glob("*.swc")) if (anatomy_dir / "skeletons").exists() else []:
        node = lut.get(int(swc.stem))
        if node is None:                                 # a neuron outside this sub-network
            continue
        xyz, parent = read_swc(swc)
        skeletons.append(Skeleton(int(swc.stem), node, xyz, parent))
    return AnatomyAssets(neuropils, skeletons)
