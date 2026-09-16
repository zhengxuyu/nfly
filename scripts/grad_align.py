"""Does the policy gradient point where supervision points? A diagnostic for RL on the fly readout.

    python scripts/grad_align.py --subset visual --device cuda --readout-dim 0 --head-hidden 64 \
        --init runs/dagger-pong-mlp.pt --teacher runs/baseline-cnn-pong

At a head that plays (a DAgger head), roll out the sampled policy in n envs for T steps, keep
the readout features, the actions, the log-probs, the critic's values, the rewards and the CNN
teacher's label for every step. Then, on random minibatches, compare three gradients on the
head's parameters:

    g_pg     the PPO / policy-gradient direction at ratio 1: -A * grad log pi(a), A from GAE
             with the fly's own critic (what training actually follows)
    g_teacher the same with teacher agreement: +1 if a is the teacher's action else -1
             (a supervision control, not the true Pong advantage)
    g_ce     the cross-entropy gradient towards the teacher's label (what DAgger follows)

and report cosine similarities and the signal-to-noise ratio of each across minibatches
(norm of the mean gradient over the mean norm of the deviations). A random-advantage control
gives the noise floor. Also reports the critic's explained variance on this data. Brain,
encoder and readout stay frozen, as in the RL runs.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bc_pong import CNNTeacher  # noqa: E402

from nfly import FlyAgent  # noqa: E402
from nfly.cli import add_agent_args, add_connectome_args, agent_kwargs, calibrate_on, connectome_from_args  # noqa: E402
from nfly.rl.simple.common import load_checkpoint, reset_state  # noqa: E402
from nfly.suite import get_suite  # noqa: E402


def rollout(agent, envs, teacher, steps: int, device: str):
    """Lockstep rollout of the sampled policy in `envs`; returns per-step tensors (T, n, ...)."""
    n = len(envs)
    obs = [e.reset(seed=100 + i)[0] for i, e in enumerate(envs)]
    stacks = [[] for _ in envs]
    h = agent.initial_state(n)
    F, A, LP, V, R, D, LAB = [], [], [], [], [], [], []
    with torch.no_grad():
        for _ in range(steps):
            x = torch.as_tensor(np.stack([np.asarray(o) for o in obs]), device=device)
            f, h = agent.step(x, h)
            dist = agent.decoder.distribution(f); a = dist.sample()
            labels = []
            for i, e in enumerate(envs):
                teacher.stack = stacks[i]; labels.append(teacher(e)); stacks[i] = teacher.stack
            F.append(f); A.append(a); LP.append(dist.log_prob(a)); V.append(agent.value(f).squeeze(-1)); LAB.append(torch.tensor(labels, device=device))
            r, d = [], []
            for i, e in enumerate(envs):
                o, rew, term, trunc, _ = e.step(int(a[i])); r.append(np.sign(rew)); d.append(float(term or trunc))
                if term or trunc:
                    o, _ = e.reset(); stacks[i] = []
                obs[i] = o
            R.append(torch.tensor(r, dtype=torch.float32, device=device)); D.append(torch.tensor(d, device=device))
            h = reset_state(agent, h, D[-1])
        x = torch.as_tensor(np.stack([np.asarray(o) for o in obs]), device=device)
        f, _ = agent.step(x, h); boot = agent.value(f).squeeze(-1)
    return [torch.stack(t) for t in (F, A, LP, V, R, D, LAB)] + [boot]


def gae(V, R, D, boot, gamma=0.99, lam=0.95):
    T = len(R); adv = torch.zeros_like(R); last = torch.zeros_like(boot); next_v = boot
    for t in reversed(range(T)):
        nonterminal = 1 - D[t]
        delta = R[t] + gamma * next_v * nonterminal - V[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last; next_v = V[t]
    return adv, adv + V


def flat_grad(loss, params):
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(grads, params)])


def snr(gs: list[torch.Tensor]) -> float:
    G = torch.stack(gs); mean = G.mean(0)
    return float(mean.norm() / (G - mean).norm(dim=1).mean().clamp_min(1e-12))


def cos(a, b) -> float:
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


def main() -> None:
    p = argparse.ArgumentParser()
    add_connectome_args(p, subset="visual")
    add_agent_args(p)
    p.add_argument("--init", required=True); p.add_argument("--teacher", required=True)
    p.add_argument("--envs", type=int, default=8); p.add_argument("--steps", type=int, default=256)
    p.add_argument("--minibatch", type=int, default=256); p.add_argument("--batches", type=int, default=32)
    args = p.parse_args()
    torch.manual_seed(args.seed)

    conn = connectome_from_args(args)
    envs = [get_suite("atari").make("pong", seed=args.seed + i) for i in range(args.envs)]
    agent = FlyAgent.build(conn, envs[0].observation_space, envs[0].action_space, **agent_kwargs(args)).to(args.device).eval()
    calibrate_on(agent, get_suite("atari").make("pong", seed=args.seed + 1000))
    load_checkpoint(agent, args.init)
    teacher = CNNTeacher(args.teacher, args.device)

    F, A, LP, V, R, D, LAB, boot = rollout(agent, envs, teacher, args.steps, args.device)
    adv, v_target = gae(V, R, D, boot)
    ev = 1 - float((v_target - V).var() / v_target.var().clamp_min(1e-12))
    adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
    teacher_agreement = torch.where(A == LAB, 1.0, -1.0)
    rand = torch.randn_like(adv_n)
    N = args.steps * args.envs
    F2, A2, LAB2 = F.reshape(N, -1), A.reshape(N), LAB.reshape(N)
    advs = {"policy gradient (fly critic)": adv_n.reshape(N), "teacher-agreement control": teacher_agreement.reshape(N),
            "random advantage (noise floor)": rand.reshape(N)}
    params = [q for q in agent.decoder.head.parameters()]
    print(f"{N} steps, rewards {int((R != 0).sum())} nonzero, teacher agreement {float((A == LAB).float().mean()):.1%}, "
          f"critic explained variance {ev:.3f}, mean return per env {float(R.sum(0).mean()):.2f}", flush=True)

    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    grads = {k: [] for k in advs}; ce_grads = []
    for _ in range(args.batches):
        idx = torch.randperm(N, generator=gen)[: args.minibatch].to(args.device)
        logits = agent.decoder.head(F2[idx]); logp = torch.distributions.Categorical(logits=logits).log_prob(A2[idx])
        for k, a in advs.items():
            grads[k].append(flat_grad(-(a[idx] * logp).mean(), params))
        ce_grads.append(flat_grad(torch.nn.functional.cross_entropy(agent.decoder.head(F2[idx]), LAB2[idx]), params))
    ce_mean = torch.stack(ce_grads).mean(0)
    print("\n| Gradient on the head | cos with the supervised gradient (mean over batches) | cos of the batch-mean gradients | SNR across batches |")
    print("| --- | --- | --- | --- |")
    print(f"| supervised (cross-entropy to the teacher) | 1.000 | 1.000 | {snr(ce_grads):.3f} |")
    for k, gs in grads.items():
        per = np.mean([cos(g, c) for g, c in zip(gs, ce_grads)])
        print(f"| {k} | {per:.3f} | {cos(torch.stack(gs).mean(0), ce_mean):.3f} | {snr(gs):.3f} |")


if __name__ == "__main__":
    main()
