"""Compute and energy cost of the fly against an MLP of the same parameter count.

    python scripts/benchmark_cost.py --subset visual --device cuda
    python scripts/benchmark_cost.py --subset visual --device cpu --no-power

For the Pong observation (2 x 84 x 84) both models get the same batch (16 envs, the trainer's
default) and the same unroll (32 steps, the PPO segment): forward only (rollout collection) and
forward + backward (one replay pass). GPU power is sampled with nvidia-smi while each loop runs
and the idle draw is subtracted, so the energy column is joules per env step attributable to the
model; on a shared GPU other jobs add noise to that number, which the idle sample only partly
removes. Prints a Markdown table for docs/benchmarks.md.
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import threading
import time

import torch

from nfly import FlyAgent
from nfly.cli import add_agent_args, add_connectome_args, agent_kwargs, calibrate_on, connectome_from_args
from nfly.rl.simple.reference import MLPReference
from nfly.suite import get_suite

MLP_HIDDEN = (600, 64)      # 14,112 x 600 + 600 x 64 + heads = 8.5M, the fly's 8.8M within 4%


class PowerMeter(threading.Thread):
    """Samples nvidia-smi power draw (W) every 0.2 s until stopped."""

    def __init__(self):
        super().__init__(daemon=True)
        self.samples: list[float] = []
        self._halt = threading.Event()          # not _stop: Thread uses that name internally

    def run(self) -> None:
        while not self._halt.is_set():
            out = subprocess.run(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True)
            try:
                self.samples.append(float(out.stdout.strip().splitlines()[0]))
            except (ValueError, IndexError):
                pass
            time.sleep(0.2)

    def stop(self) -> float:
        self._halt.set(); self.join()
        return statistics.mean(self.samples) if self.samples else float("nan")


def idle_power(seconds: float = 3.0) -> float:
    m = PowerMeter(); m.start(); time.sleep(seconds); return m.stop()


def unroll(agent, obs: torch.Tensor, steps: int, grad: bool) -> None:
    h = agent.initial_state(obs.shape[0])
    weights = agent.weights()
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        loss = 0.0
        for _ in range(steps):
            dist, value, h = agent(obs, h, weights)
            if grad:
                loss = loss + dist.entropy().mean() + value.mean()
        if grad:
            loss.backward()


def timed(agent, obs, steps: int, grad: bool, repeats: int, device: str, power: bool) -> tuple[float, float]:
    """Returns (ms per env step, W above idle) over `repeats` unrolls."""
    unroll(agent, obs, 2, grad)                                        # warm-up
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    meter = PowerMeter() if power else None
    if meter:
        meter.start()
    t0 = time.perf_counter()
    for _ in range(repeats):
        unroll(agent, obs, steps, grad)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    watts = meter.stop() if meter else float("nan")
    return 1000 * dt / (repeats * steps * obs.shape[0]), watts


def macs_per_frame(agent) -> int:
    if isinstance(agent, FlyAgent):
        edges = int(agent.brain.pre.numel())
        return edges * agent.rnn_steps + int(agent.encoder.n_inputs) + agent.decoder.n_readout * 8
    return sum(m.weight.numel() for m in agent.modules() if isinstance(m, torch.nn.Linear))


def main() -> None:
    p = argparse.ArgumentParser()
    add_connectome_args(p, subset="visual")
    add_agent_args(p)
    p.add_argument("--envs", type=int, default=16)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--no-power", action="store_true")
    args = p.parse_args()
    power = args.device.startswith("cuda") and not args.no_power

    env = get_suite("atari").make("pong", seed=0)
    conn = connectome_from_args(args)
    fly = FlyAgent.build(conn, env.observation_space, env.action_space, **agent_kwargs(args)).to(args.device)
    calibrate_on(fly, get_suite("atari").make("pong", seed=1000), log=lambda *_: None)
    mlp = MLPReference(env.observation_space, env.action_space, MLP_HIDDEN).to(args.device)
    obs = torch.as_tensor(env.reset(seed=0)[0], device=args.device).float().unsqueeze(0).repeat(args.envs, 1, 1, 1)

    p_idle = idle_power() if power else float("nan")
    rows = []
    for name, agent in (("fly (visual sub-network)", fly), (f"MLP {MLP_HIDDEN}", mlp)):
        agent.train()
        params = sum(q.numel() for q in agent.parameters() if q.requires_grad)
        fwd_ms, fwd_w = timed(agent, obs, args.steps, False, args.repeats, args.device, power)
        bwd_ms, bwd_w = timed(agent, obs, args.steps, True, args.repeats, args.device, power)
        rows.append((name, params, macs_per_frame(agent), fwd_ms, bwd_ms, fwd_w - p_idle, bwd_w - p_idle))

    dev = torch.cuda.get_device_name(0) if args.device.startswith("cuda") else "CPU"
    print(f"\n{dev}, batch {args.envs} envs, unroll {args.steps} steps; idle GPU draw {p_idle:.0f} W\n")
    print("| Model | Trainable parameters | Multiply-adds per frame | Rollout ms / env step | Replay (fwd+bwd) ms / env step | Rollout J / env step | Replay J / env step |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for name, params, macs, fwd_ms, bwd_ms, fwd_w, bwd_w in rows:
        e_fwd, e_bwd = fwd_w * fwd_ms / 1000, bwd_w * bwd_ms / 1000
        print(f"| {name} | {params:,} | {macs / 1e6:.1f}M | {fwd_ms:.3f} | {bwd_ms:.3f} | {e_fwd:.4f} | {e_bwd:.4f} |")


if __name__ == "__main__":
    main()
