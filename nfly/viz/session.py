"""A session is one (policy, env) pair described by a config, so every front end (web page,
video recorder, notebook) starts a game the same way."""

from __future__ import annotations

import dataclasses
from typing import Protocol, runtime_checkable

import gymnasium as gym
import numpy as np
import torch

from ..agent import FlyAgent
from ..connectome import load_malecns, select_subset
from ..rl.simple.common import load_checkpoint
from ..suite import get_suite
from .anatomy import ActivityScale, AnatomyAssets, BrainAtlas, build_atlas, calibrate_activity, load_assets


@runtime_checkable
class Policy(Protocol):
    """What the visualiser needs from an agent: a recurrent state and an act() step."""

    def initial_state(self, batch: int) -> torch.Tensor: ...

    def act(self, obs: torch.Tensor, h: torch.Tensor, greedy: bool = False): ...


class RandomPolicy:
    """Uniform random actions; useful to check an env renders before loading a brain."""

    def __init__(self, action_space: gym.Space):
        self.action_space = action_space

    def initial_state(self, batch: int) -> torch.Tensor:
        return torch.zeros(batch, 0)

    def act(self, obs, h, greedy: bool = False):
        return np.asarray([self.action_space.sample()]), h


@dataclasses.dataclass
class SessionConfig:
    suite: str = "atari"
    game: str = "pong"
    data_dir: str = "data"
    subset: str = "visual"
    min_syn: int = 3
    rnn_steps: int = 4
    readout_dim: int | None = None # must match the checkpoint's agent (None = default, 0 = no bottleneck)
    head_hidden: int = 0           # 0 = linear policy head, else tanh MLP width (must match the checkpoint)
    checkpoint: str | None = None
    compare_checkpoint: str | None = None   # a second agent (same build) played side by side; "untrained" = no checkpoint
    policy: str = "fly"            # fly | random
    greedy: bool = False
    device: str = "cpu"
    seed: int = 0
    fps: float = 15.0              # playback speed of the stream
    anatomy: bool = True           # stream brain activity for the 3-D view (fly policy only)
    max_points: int = 30000        # neurons rendered in the 3-D view
    anatomy_dir: str | None = None # official meshes and skeletons (default <data_dir>/anatomy, see scripts/fetch_anatomy.py)


@dataclasses.dataclass
class Session:
    config: SessionConfig
    env: gym.Env
    policy: Policy
    action_names: list[str]
    atlas: BrainAtlas | None = None          # set for fly policies with anatomy on
    activity: ActivityScale | None = None
    assets: AnatomyAssets | None = None      # official meshes and skeletons, when fetched
    compare: "Session | None" = None         # the second (policy, env) pair for side-by-side play


def action_names_of(env: gym.Env) -> list[str]:
    unwrapped = env.unwrapped
    if hasattr(unwrapped, "get_action_meanings"):
        return list(unwrapped.get_action_meanings())
    if isinstance(env.action_space, gym.spaces.Discrete):
        return [f"a{i}" for i in range(env.action_space.n)]
    return [f"dim{i}" for i in range(int(np.prod(env.action_space.shape)))]


def build_session(cfg: SessionConfig) -> Session:
    env = get_suite(cfg.suite).make(cfg.game, seed=cfg.seed, render_mode="rgb_array")
    if cfg.policy == "random":
        return Session(cfg, env, RandomPolicy(env.action_space), action_names_of(env))
    conn = select_subset(load_malecns(cfg.data_dir, min_syn=cfg.min_syn), cfg.subset)
    policy = _fly_policy(cfg, conn, env, cfg.checkpoint)
    session = Session(cfg, env, policy, action_names_of(env))
    if cfg.anatomy:
        session.assets = load_assets(cfg.anatomy_dir or f"{cfg.data_dir}/anatomy", conn)
        session.atlas = build_atlas(conn, policy.encoder, cfg.max_points, keep=session.assets.skeleton_nodes)
        session.activity = calibrate_activity(policy, get_suite(cfg.suite).make(cfg.game, seed=cfg.seed + 2000))
    if cfg.compare_checkpoint:
        env_b = get_suite(cfg.suite).make(cfg.game, seed=cfg.seed, render_mode="rgb_array")
        cfg_b = dataclasses.replace(cfg, checkpoint=cfg.compare_checkpoint, compare_checkpoint=None, anatomy=False)
        session.compare = Session(cfg_b, env_b, _fly_policy(cfg, conn, env_b, cfg.compare_checkpoint), action_names_of(env_b))
    return session


def _fly_policy(cfg: SessionConfig, conn, env: gym.Env, checkpoint: str | None) -> FlyAgent:
    agent = FlyAgent.build(conn, env.observation_space, env.action_space, rnn_steps=cfg.rnn_steps,
                           readout_dim=cfg.readout_dim, head_hidden=cfg.head_hidden).to(cfg.device)
    agent.calibrate_on_env(get_suite(cfg.suite).make(cfg.game, seed=cfg.seed + 1000))
    if checkpoint and checkpoint != "untrained":
        load_checkpoint(agent, checkpoint)
    return agent.eval()
