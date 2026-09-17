"""DAgger on the frozen fly: the student plays, the CNN teacher labels every frame, the head is
refitted on all data so far.

    python scripts/dagger_pong.py --subset visual --device cuda --readout-dim 0 --head-hidden 64 \
        --teacher runs/baseline-cnn-pong --rounds 8 --steps-per-round 12000 --out runs/dagger-pong-mlp.pt

Behaviour cloning from 6,000 teacher steps (scripts/bc_pong.py) gave a head that wins some
episodes and loses others: once the student drifts into states the teacher never visited, it
has no label to lean on. DAgger (Ross et al. 2011) fixes exactly that: round 0 records the
teacher, every later round records the student's own states with the teacher's action as the
label, and the head is refitted on the aggregate. Only the head trains; the connectome, the
encoder and the readout calibration stay frozen, so the result is still "the wiring plus one
readout". Prints the greedy return after every round and saves the best head.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bc_pong import fit_head, mean_entropy, play, temper_head  # noqa: E402

from nfly import FlyAgent  # noqa: E402
from nfly.cli import add_agent_args, add_connectome_args, agent_kwargs, calibrate_on, connectome_from_args  # noqa: E402
from nfly.rl.simple.common import load_checkpoint, save_checkpoint  # noqa: E402
from nfly.rl.rllib.teacher import CNNTeacher, make_teacher_env  # noqa: E402
from nfly.suite import get_suite  # noqa: E402


def record(env, agent, teacher, steps: int, device: str, act_with_student: bool, noise: float, seed: int):
    """Roll out `steps` env steps; returns readout features and the teacher's label per step.
    Round 0 acts with the (noisy) teacher, later rounds with the student's greedy head."""
    obs, _ = env.reset(seed=seed); h = agent.initial_state(1); teacher.reset()
    F, A = [], []
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        for _ in range(steps):
            f, h = agent.step(torch.as_tensor(np.asarray(obs), device=device).unsqueeze(0), h)
            label = teacher(env)
            F.append(f[0]); A.append(label)
            if act_with_student:
                a = int(agent.decoder.head(f).argmax())
            else:
                a = int(env.action_space.sample()) if rng.random() < noise else label
            obs, _, term, trunc, _ = env.step(a)
            if term or trunc:
                obs, _ = env.reset(); h = agent.initial_state(1); teacher.reset()
    return torch.stack(F), torch.tensor(A, device=device)


def main() -> None:
    p = argparse.ArgumentParser()
    add_connectome_args(p, subset="visual")
    add_agent_args(p)
    p.add_argument("--teacher", required=True, help="path to a baseline_cnn_pong checkpoint")
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--steps-per-round", type=int, default=12000)
    p.add_argument("--noise", type=float, default=0.3, help="round 0 only: fraction of random actions")
    p.add_argument("--fit-steps", type=int, default=3000)
    p.add_argument("--episodes", type=int, default=5, help="greedy evaluation episodes per round")
    p.add_argument("--entropy-target", type=float, default=1.0, help="temper the saved head's logits to this entropy for RL")
    p.add_argument("--init", help="start from the head in this checkpoint (e.g. an earlier DAgger run)")
    p.add_argument("--teacher-protocol", choices=["pooled", "legacy"], default="pooled")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    conn = connectome_from_args(args)
    env = make_teacher_env(seed=args.seed, protocol=args.teacher_protocol)
    agent = FlyAgent.build(conn, env.observation_space, env.action_space, **agent_kwargs(args)).to(args.device).eval()
    calibrate_on(agent, get_suite("atari").make("pong", seed=args.seed + 1000))
    if args.init:
        load_checkpoint(agent, args.init)
    teacher = CNNTeacher(args.teacher, args.device, args.teacher_protocol)
    head = agent.decoder.head
    n_actions = env.action_space.n

    F_all, A_all, best = None, None, -float("inf")
    for rnd in range(args.rounds):
        F, A = record(env, agent, teacher, args.steps_per_round, args.device, act_with_student=rnd > 0 or bool(args.init),
                      noise=args.noise, seed=args.seed + 10 * rnd)
        F_all = F if F_all is None else torch.cat([F_all, F]); A_all = A if A_all is None else torch.cat([A_all, A])
        fit_head(head, F_all, A_all, args.fit_steps)
        acc = float((head(F_all).argmax(1) == A_all).float().mean())
        shares = ", ".join(f"{a} {float((A == a).float().mean()):.0%}" for a in range(n_actions))
        scores = play(env, agent, lambda f, e, o: int(head(f).argmax()), args.episodes, args.device)
        mean = float(np.mean(scores))
        print(f"round {rnd}: {len(F_all):,} labelled steps (this round's action shares {shares}); fit accuracy {acc:.1%}; "
              f"greedy return {mean:.1f} (episodes {scores})", flush=True)
        if mean > best:
            best = mean
            saved = {k: v.clone() for k, v in head.state_dict().items()}
    head.load_state_dict(saved)
    ent = mean_entropy(head, F_all)
    scale = temper_head(head, F_all, args.entropy_target)
    print(f"best greedy return {best:.1f}; logits x {scale:.3g} ({ent:.2f} -> {args.entropy_target:.2f} nats) for RL", flush=True)
    save_checkpoint(agent, args.out, teacher=args.teacher, teacher_protocol=args.teacher_protocol,
                    dagger_return=best, labelled_steps=int(len(F_all)))
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
