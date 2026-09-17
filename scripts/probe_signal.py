"""Compare sensory signals on identical random-action trajectories, without training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from nfly import FlyAgent
from nfly.cli import add_agent_args, add_connectome_args, agent_kwargs, connectome_from_args
from nfly.rl.simple.common import reset_state
from nfly.suite import get_suite


def random_bank(env, seeds, steps, burn_in=32):
    """Keep calibration and held-out environments separate, including reset boundaries."""
    rng = np.random.default_rng(seeds[0])
    obs, _ = env.reset(seed=seeds)
    age = np.zeros(len(seeds), dtype=int)
    rows, dones, valid = [], [], []
    for _ in range(steps):
        rows.append(torch.as_tensor(obs).clone())
        valid.append(torch.as_tensor(age >= burn_in))
        obs, _, term, trunc, _ = env.step(rng.integers(env.single_action_space.n, size=len(seeds)))
        done = np.asarray(term) | np.asarray(trunc)
        dones.append(torch.as_tensor(done.copy()))
        age = np.where(done, 0, age + 1)
    return dict(obs=torch.stack(rows), done=torch.stack(dones), valid=torch.stack(valid), seeds=seeds)


@torch.no_grad()
def trace_readout(agent, bank):
    device = agent.input_gain.device
    h = agent.initial_state(len(bank["seeds"]))
    weights = agent.weights()
    rows = []
    for obs, done in zip(bank["obs"], bank["done"]):
        _, h = agent.step(obs.to(device), h, weights)
        rows.append(h[:, agent.decoder.idx].cpu())
        h = reset_state(agent, h, done.to(device))
    return torch.stack(rows)


def variation(x):
    return float(x.std(0, unbiased=False).square().mean().sqrt())


@torch.no_grad()
def assess_readout(agent, raw):
    decoder = agent.decoder
    z = (raw - decoder.norm.mean) / decoder.norm.scale
    norm = decoder.norm(raw)
    features = decoder.proj(norm).clamp(-decoder.feature_clip, decoder.feature_clip)
    hidden = decoder.trunk(features)
    logits = decoder.dist_inputs(features)
    probs = logits.softmax(-1)
    # This diagnostic is intentionally restricted to independent readout MLP heads.
    critic_hidden = agent.value[:2](features)
    value = agent.value(features)
    return dict(raw_temporal_std=variation(raw), normalized_temporal_std=variation(norm),
                normalized_offset_rms=float(z.mean(0).square().mean().sqrt()),
                normalized_clip_fraction=float((norm.abs() >= decoder.norm.clip).float().mean()),
                actor_hidden_temporal_std=variation(hidden),
                actor_saturation_fraction=float((hidden.abs() > 0.99).float().mean()),
                critic_saturation_fraction=float((critic_hidden.abs() > 0.99).float().mean()),
                critic_hidden_temporal_std=variation(critic_hidden),
                probability_temporal_std=variation(probs), value_temporal_std=variation(value),
                mean_probabilities=probs.mean(0).tolist(),
                greedy_action_counts=torch.bincount(probs.argmax(-1), minlength=probs.shape[-1]).tolist())


@torch.no_grad()
def compare_heads(agent, states, traces, bank):
    device = agent.input_gain.device
    held = bank["valid"][:, 1]
    calibrated = bank["valid"][:, 0]
    if not held.any() or not calibrated.any():
        raise ValueError("No observations remain after burn-in")
    report = {}
    original = {k: v.detach().clone() for k, v in agent.state_dict().items()}
    try:
        for name in ("initial", "trained"):
            agent.load_state_dict(states[name])
            report[name] = assess_readout(agent, traces[name][held, 1].to(device))
        # Keep trained heads while restoring the exact initial feature representation.
        agent.decoder.norm.load_state_dict({k.removeprefix("decoder.norm."): v
                                           for k, v in states["initial"].items()
                                           if k.startswith("decoder.norm.")})
        agent.decoder.proj.load_state_dict({k.removeprefix("decoder.proj."): v
                                           for k, v in states["initial"].items()
                                           if k.startswith("decoder.proj.")})
        report["trained_heads_initial_features"] = assess_readout(agent, traces["initial"][held, 1].to(device))
        agent.load_state_dict(states["trained"])
        agent.decoder.norm.mean.copy_(traces["trained"][calibrated, 0].mean(0).to(device))
        report["trained_recenter_only"] = assess_readout(agent, traces["trained"][held, 1].to(device))
    finally:
        agent.load_state_dict(original)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_connectome_args(parser)
    add_agent_args(parser)
    parser.add_argument("--initial", required=True)
    parser.add_argument("--trained", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=512)
    args = parser.parse_args()
    if args.head_hidden <= 0 or args.critic != "readout" or args.share_trunk:
        parser.error("Use an MLP actor and independent readout critic")
    torch.manual_seed(args.seed)
    env = get_suite("atari").make_vector("pong", 2, seed=args.seed)
    try:
        bank = random_bank(env, [args.seed, args.seed + 1], args.steps)
        agent = FlyAgent.build(connectome_from_args(args), env.single_observation_space,
                               env.single_action_space, **agent_kwargs(args)).to(args.device).eval()
        states, traces, updates = {}, {}, {}
        for name in ("initial", "trained"):
            payload = torch.load(getattr(args, name), map_location="cpu", weights_only=False)
            states[name], updates[name] = payload["agent"], payload["update"]
            agent.load_state_dict(states[name])
            print(f"Tracing {name} update {updates[name]}", flush=True)
            traces[name] = trace_readout(agent, bank)
        report = dict(arguments=vars(args), updates=updates,
                      held_out_samples=int(bank["valid"][:, 1].sum()),
                      metrics=compare_heads(agent, states, traces, bank))
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2) + "\n")
        torch.save(dict(bank=bank, traces=traces, report=report), target.with_suffix(".pt"))
        print(json.dumps(report, indent=2), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
