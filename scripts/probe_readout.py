"""What can be linearly decoded from the fly's readout?  A diagnostic, not a trainer.

Runs a random policy in a game, records the normalised readout activity (all readout neurons)
and the calibrated readout features, and ridge-regresses named quantities of the game state from
each. For Pong the targets come from the ALE RAM: ball x/y, the player's paddle y, and the ball
velocity (frame-to-frame difference), i.e. the motion signal a linear policy needs.

    uv run scripts/probe_readout.py --suite atari --game pong --subset visual --steps 1500
    uv run scripts/probe_readout.py --suite classic --game cartpole --subset visual_small
    uv run scripts/probe_readout.py --suite atari --game pong --layers      # also stage by stage

With --layers the same regression is run on the activity of successive stages of the visual
pathway (photoreceptor drive, lamina, medulla, lobula / T4-T5, visual projection neurons,
descending neurons), which shows where a signal such as the ball's position is lost.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from nfly import FlyAgent
from nfly.cli import add_agent_args, add_connectome_args, agent_kwargs, calibrate_on, connectome_from_args
from nfly.suite import get_suite

PONG_RAM = {"ball x": 49, "ball y": 54, "player paddle y": 51, "cpu paddle y": 50}

# Successive stages of the fly visual pathway, by MaleCNS cell type / superclass.
STAGES = [
    ("photoreceptors R1-R8", {"cell_type": ["R1-R6", "R7p", "R7y", "R8p", "R8y", "R7d", "R8d", "R7_unclear", "R8_unclear", "R7R8_unclear"]}),
    ("lamina L1-L5", {"cell_type": ["L1", "L2", "L3", "L4", "L5"]}),
    ("medulla Mi/Tm", {"cell_type": ["Mi1", "Mi4", "Mi9", "Tm1", "Tm2", "Tm3", "Tm4", "Tm9", "Tm20"]}),
    ("T4/T5 motion detectors", {"cell_type": ["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"]}),
    ("visual projection neurons", {"super_class": ["visual_projection"]}),
    ("descending neurons", {"super_class": ["descending_neuron"]}),
]


def targets_of(env, prev: dict | None) -> tuple[dict, dict]:
    """Named state quantities for the current step, plus the previous step's values for velocities."""
    u = env.unwrapped
    if hasattr(u, "ale"):
        ram = u.ale.getRAM()
        cur = {k: float(ram[i]) for k, i in PONG_RAM.items()}
    else:
        cur = {f"state {i}": float(v) for i, v in enumerate(np.asarray(u.state).ravel())}
    vel = {f"d {k}": (cur[k] - prev[k]) if prev is not None else 0.0 for k in list(cur)[:2]}
    return {**cur, **vel}, cur


def decodability(X: torch.Tensor, Y: torch.Tensor, n_components: int = 256, block: int = 25) -> torch.Tensor:
    """Held-out R^2 of a linear decoder of Y from X, robust to many more features than samples.

    Standardise (with a relative floor so a rarely active unit cannot explode), project to the
    top principal components, then ridge with the regularisation chosen by validation. Train,
    validation and test are interleaved blocks of `block` steps (not first half / second half):
    in Pong the score digits are the largest changing feature and drift monotonically through
    a game, so a contiguous split would test extrapolation to unseen scores instead of decoding."""
    std = X.std(0)
    X = (X - X.mean(0)) / (std + 0.1 * std.mean() + 1e-6)
    part = (torch.arange(len(X)) // block) % 3                 # 0 train, 1 validation, 2 test
    tr, va, te = (part == 0), (part == 1), (part == 2)
    if X.shape[1] > n_components:
        _, _, v = torch.pca_lowrank(X[tr], q=n_components, center=False)
        X = X @ v
    X1 = torch.cat([X, torch.ones(len(X), 1)], 1)
    def fit(A, B, lam):
        return torch.linalg.solve(A.T @ A + lam * torch.eye(A.shape[1]), A.T @ B)
    def r2(pred, target):
        return 1 - ((pred - target) ** 2).sum(0) / ((target - target.mean(0)) ** 2).sum(0)
    best = max((float(r2(X1[va] @ fit(X1[tr], Y[tr], lam), Y[va]).mean()), lam) for lam in (1e-2, 1e-1, 1, 10, 100, 1000))[1]
    fit_mask = tr | va
    return r2(X1[te] @ fit(X1[fit_mask], Y[fit_mask], best), Y[te])


def main() -> None:
    p = argparse.ArgumentParser()
    add_connectome_args(p, subset="visual")
    add_agent_args(p)
    p.add_argument("--suite", default="atari")
    p.add_argument("--game", default="pong")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--layers", action="store_true", help="also probe successive stages of the visual pathway")
    p.add_argument("--alpha", type=float, default=0.7, help="leak per network step")
    p.add_argument("--temporal-gain", type=float, default=4.0, help="retina: weight of the change channel")
    p.add_argument("--surround", type=float, default=0.0, help="retina: weight of the centre-surround term")
    p.add_argument("--max-neurons", type=int, default=3000, help="--layers: random subset size per stage")
    args = p.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    conn = connectome_from_args(args)
    env = get_suite(args.suite).make(args.game, seed=args.seed)
    agent = FlyAgent.build(conn, env.observation_space, env.action_space, alpha_init=args.alpha,
                           encoder_kw={"temporal_gain": args.temporal_gain, "surround": args.surround},
                           **agent_kwargs(args)).to(args.device).eval()
    print(f"config: rnn_steps {args.rnn_steps} alpha {args.alpha} temporal_gain {args.temporal_gain} surround {args.surround}", flush=True)
    calibrate_on(agent, get_suite(args.suite).make(args.game, seed=args.seed + 1000))

    rng = np.random.default_rng(args.seed)
    stages = []
    if args.layers:
        for name, cond in STAGES:
            idx = conn.where(**cond).numpy()
            if len(idx) > args.max_neurons:
                idx = np.sort(rng.choice(idx, args.max_neurons, replace=False))
            stages.append((f"{name} ({len(idx)} of {len(conn.where(**cond))})", torch.as_tensor(idx, device=args.device)))
    obs, _ = env.reset(); h = agent.initial_state(1); prev = None
    raw, feats, ys, stage_acts, drives = [], [], [], [[] for _ in stages], []
    with torch.no_grad():
        for _ in range(args.steps):
            x = torch.as_tensor(np.asarray(obs), device=args.device).unsqueeze(0)
            f, h = agent.step(x, h)
            y, prev = targets_of(env, prev)
            raw.append(agent.decoder.norm(h[:, agent.decoder.idx])[0].cpu()); feats.append(f[0].cpu()); ys.append(list(y.values()))
            if stages:
                drives.append(agent.encoder.encode(x)[0].cpu())
                for acts, (_, idx) in zip(stage_acts, stages):
                    acts.append(h[0, idx].cpu())
            obs, _, term, trunc, _ = env.step(env.action_space.sample())
            if term or trunc:
                obs, _ = env.reset(); h = agent.initial_state(1); prev = None
    names = list(y)
    Y = torch.tensor(ys)
    probes = [(f"all {agent.decoder.n_readout} readout neurons", torch.stack(raw)),
              (f"{agent.decoder.n_features} calibrated features", torch.stack(feats))]
    if stages:
        probes.insert(0, (f"photoreceptor drive ({agent.encoder.n_inputs} inputs, before the network)", torch.stack(drives)))
        probes[1:1] = [(label, torch.stack(acts)) for (label, _), acts in zip(stages, stage_acts)]
    for label, X in probes:
        r2 = decodability(X, Y)
        print(f"held-out R^2 from {label}: " + ", ".join(f"{n} {float(v):.2f}" for n, v in zip(names, r2)), flush=True)


if __name__ == "__main__":
    main()
