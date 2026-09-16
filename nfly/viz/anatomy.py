"""The brain as a point cloud in MaleCNS coordinates, and activity in units the eye can read.

Every annotated neuron with a soma has an official position (micrometres in the EM space);
neurons without one (photoreceptors and other sensory neurons, whose somata lie outside the
volume) are placed at the synapse-weighted mean of their partners. Activity is shown as a
z-score against a random-policy rollout: (h - mean) / spread per neuron, so a descending
neuron whose whole dynamic range is 1e-6 lights up as brightly as a photoreceptor."""

from __future__ import annotations

import base64
import dataclasses

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

    def to_dict(self) -> dict:
        return {"n": int(len(self.index)), "xyz": _b64(self.xyz), "stage": _b64(self.stage),
                "stages": STAGES, "colors": STAGE_COLORS, "counts": np.bincount(self.stage, minlength=len(STAGES)).tolist(),
                "retina_xy": _b64(self.retina_xy), "retina_eye": _b64(self.retina_eye)}


def build_atlas(conn: Connectome, encoder, max_points: int = 30000, seed: int = 0) -> BrainAtlas:
    """Sample up to max_points placed neurons, keeping every photoreceptor and descending neuron."""
    xyz, stage = neuron_positions(conn), stage_of(conn.neurons)
    placed = np.flatnonzero(~np.isnan(xyz[:, 0]))
    keep_all = np.isin(stage[placed], [STAGES.index("photoreceptors"), STAGES.index("descending")])
    rest = placed[~keep_all]
    rng = np.random.default_rng(seed)
    budget = max(0, max_points - int(keep_all.sum()))
    if len(rest) > budget:
        rest = rng.choice(rest, budget, replace=False)
    index = np.sort(np.concatenate([placed[keep_all], rest]))
    retina_xy, retina_eye = _retina_layout(encoder)
    return BrainAtlas(index, xyz[index], stage[index], retina_xy, retina_eye)


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

    def to_dict(self) -> dict:
        return {"cloud": _b64(self.cloud), "stages": self.stages, "retina": _b64(self.retina)}


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
    cloud = _quantise(scale.z(states[-1][0])[index], Z_CLIP)
    retina = _quantise(drive, scale.drive_scale)
    return BrainActivity(cloud, per_step, retina)


def _quantise(x: torch.Tensor, full_scale: float) -> np.ndarray:
    return ((x / full_scale).clamp(-1, 1) * 127 + 128).round().to(torch.uint8).cpu().numpy()


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")
