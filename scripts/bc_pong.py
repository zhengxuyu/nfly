"""Is the fly's readout sufficient to play Pong?  Behaviour cloning on the frozen network.

A heuristic teacher (move the paddle towards the ball's y, read from ALE RAM) plays with some
exploration noise while the untrained, frozen fly watches; a head is then fitted by supervised
learning on the readout features to predict the teacher's action, and the cloned policy plays
Pong on its own. On CartPole the same test scored 500/500 and showed the representation was
sufficient; here it separates "the readout cannot support control" from "reinforcement learning
cannot find the head".

    uv run scripts/bc_pong.py --subset visual --steps 6000 --device cuda
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from nfly import FlyAgent
from nfly.cli import add_agent_args, add_connectome_args, agent_kwargs, calibrate_on, connectome_from_args
from nfly.suite import get_suite

UP, DOWN, NOOP = 2, 3, 0      # ALE Pong: RIGHT = up, LEFT = down


class CNNTeacher:
    """The trained CNN baseline (scripts/baseline_cnn_pong.py checkpoint) as the teacher.

    It reads the raw 210x160 screen and applies RLlib's own grayscale, 64x64 resize and
    uint8 / 128 - 1 normalisation, stacked over the last four env steps. Its own score is
    printed so a preprocessing mismatch shows up as a bad teacher, not a bad clone."""

    def __init__(self, checkpoint: str, device: str):
        import os
        from ray.rllib.core.rl_module.rl_module import RLModule
        module_dir = os.path.join(checkpoint, "learner_group", "learner", "rl_module", "default_policy")
        self.module = RLModule.from_checkpoint(module_dir).to(device).eval()
        self.device, self.stack = device, []

    def reset(self):
        self.stack = []

    def __call__(self, env) -> int:
        from ray.rllib.env.wrappers.atari_wrappers import resize, rgb2gray
        screen = resize(rgb2gray(env.unwrapped.ale.getScreenRGB()), height=64, width=64)
        small = torch.as_tensor(screen, device=self.device).float() / 128.0 - 1.0
        self.stack = (self.stack + [small])[-4:]
        while len(self.stack) < 4:
            self.stack.insert(0, self.stack[0])
        obs = torch.stack(self.stack, -1)[None]                      # (1, 64, 64, 4)
        with torch.no_grad():
            out = self.module.forward_inference({"obs": obs})
        return int(out["action_dist_inputs"][0].argmax())


def teacher_action(env, dead_zone: int = 3) -> int:
    ram = env.unwrapped.ale.getRAM()
    ball_y, paddle_y = int(ram[54]), int(ram[51])
    if ball_y == 0:
        return NOOP
    if ball_y < paddle_y - dead_zone:
        return UP
    if ball_y > paddle_y + dead_zone:
        return DOWN
    return NOOP


def play(env, agent, policy, episodes: int, device: str, max_steps: int = 6000, teacher=None) -> list[float]:
    rets = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=500 + ep); h = agent.initial_state(1); total = 0.0
        if teacher is not None:
            teacher.reset()
        for _ in range(max_steps):
            with torch.no_grad():
                f, h = agent.step(torch.as_tensor(np.asarray(obs), device=device).unsqueeze(0), h)
            obs, r, term, trunc, _ = env.step(policy(f, env, obs)); total += r
            if term or trunc:
                break
        rets.append(total)
    return rets


def main() -> None:
    p = argparse.ArgumentParser()
    add_connectome_args(p, subset="visual")
    add_agent_args(p)
    p.add_argument("--steps", type=int, default=6000, help="teacher steps to record")
    p.add_argument("--noise", type=float, default=0.3, help="teacher exploration: fraction of random actions")
    p.add_argument("--hidden", type=int, default=0, help="0 = linear head; else tanh MLP width")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--teacher", default="heuristic", help="'heuristic' (RAM rule) or path to a baseline_cnn_pong checkpoint")
    args = p.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    conn = connectome_from_args(args)
    env = get_suite("atari").make("pong", seed=args.seed)
    agent = FlyAgent.build(conn, env.observation_space, env.action_space, **agent_kwargs(args)).to(args.device).eval()
    calibrate_on(agent, get_suite("atari").make("pong", seed=args.seed + 1000))
    n_actions = env.action_space.n
    cnn = None if args.teacher == "heuristic" else CNNTeacher(args.teacher, args.device)
    def teach(env, obs):
        return teacher_action(env) if cnn is None else cnn(env)

    obs, _ = env.reset(); h = agent.initial_state(1); F, A = [], []
    if cnn: cnn.reset()
    with torch.no_grad():
        for _ in range(args.steps):
            f, h = agent.step(torch.as_tensor(np.asarray(obs), device=args.device).unsqueeze(0), h)
            a_star = teach(env, obs)
            F.append(f[0]); A.append(a_star)
            a = env.action_space.sample() if np.random.rand() < args.noise else a_star
            obs, _, term, trunc, _ = env.step(a)
            if term or trunc:
                obs, _ = env.reset(); h = agent.initial_state(1)
                if cnn: cnn.reset()
    F, A = torch.stack(F), torch.tensor(A, device=args.device)
    part = (torch.arange(len(F), device=args.device) // 25) % 2; tr, te = part == 0, part == 1
    print(f"recorded {len(F)} steps, {F.shape[1]} features; teacher {args.teacher}; action shares: "
          + ", ".join(f"{a} {float((A == a).float().mean()):.0%}" for a in range(n_actions)), flush=True)

    head = (torch.nn.Sequential(torch.nn.Linear(F.shape[1], args.hidden), torch.nn.Tanh(), torch.nn.Linear(args.hidden, n_actions))
            if args.hidden else torch.nn.Linear(F.shape[1], n_actions)).to(args.device)
    opt = torch.optim.Adam(head.parameters(), 3e-3, weight_decay=1e-3)
    for _ in range(1500):
        loss = torch.nn.functional.cross_entropy(head(F[tr]), A[tr]); opt.zero_grad(); loss.backward(); opt.step()
    acc_tr = float((head(F[tr]).argmax(1) == A[tr]).float().mean()); acc_te = float((head(F[te]).argmax(1) == A[te]).float().mean())
    print(f"behaviour cloning accuracy: train {acc_tr:.1%}, held-out {acc_te:.1%}", flush=True)

    teacher = play(env, agent, lambda f, e, o: teach(e, o), args.episodes, args.device, teacher=cnn)
    cloned = play(env, agent, lambda f, e, o: int(head(f).argmax()), args.episodes, args.device)
    print(f"teacher playing: {np.mean(teacher):.1f} (episodes {teacher})", flush=True)
    print(f"cloned head on frozen fly readout: {np.mean(cloned):.1f} (episodes {cloned})", flush=True)


if __name__ == "__main__":
    main()
