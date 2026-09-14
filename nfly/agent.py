"""FlyAgent = ObservationEncoder -> ConnectomeRNN -> ActionDecoder (+ value head).

The agent is a recurrent policy with an explicit state `h` (one scalar per neuron).  It is
built from a connectome and a pair of Gymnasium spaces, so any env works:

    agent = FlyAgent.build(conn, env.observation_space, env.action_space)
    h = agent.initial_state(batch)
    dist, value, h = agent(obs_tensor, h)
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from .brain.rnn import ConnectomeRNN, Weights
from .connectome.base import Connectome
from .interface.decoders import ActionDecoder
from .interface.encoders import ObservationEncoder


def value_head(n_features: int, hidden: int = 64) -> nn.Module:
    """A small tanh MLP critic. The policy stays linear on the readout, so this changes nothing
    about how the fly acts; it only gives training a value function that can fit the task
    (a linear critic could not, and a linear policy with a linear critic collapsed on CartPole
    while the same policy with this critic learned as fast as an MLP policy)."""
    return nn.Sequential(nn.Linear(n_features, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))


def _projection_targets(observed: torch.Tensor, k: int, previous: torch.Tensor | None = None) -> torch.Tensor:
    """What the readout projection should reconstruct: vector observations as they are; images
    (or any observation with more than k values) through the top-k principal components of
    [frame, frame - previous frame], so that the readout has to carry motion as well as layout.
    A single-frame target would calibrate the readout to a still picture, and a linear policy
    on a still picture cannot see where the ball is going."""
    flat = observed.reshape(observed.shape[0], -1)
    if flat.shape[1] <= k:
        return flat
    if previous is not None:
        flat = torch.cat([flat, flat - previous.reshape(previous.shape[0], -1)], 1)
    centred = flat - flat.mean(0)
    _, _, v = torch.pca_lowrank(centred, q=k, center=False)
    return centred @ v


class FlyAgent(nn.Module):
    """Defaults (rnn_steps 4, alpha 0.7, input_gain 5) come from a probe on CartPole: with one
    network step per env step and alpha 0.3, the 3-4 synaptic hops from sensory to descending
    neurons act as a low-pass filter that erases fast observation components (angular velocity
    was not linearly decodable at the readout, imitation of a competent policy was at chance);
    with four steps per env step and alpha 0.7 every observation dimension is decodable
    (R^2 0.8-0.93) and a linear probe imitates the policy at 88%."""

    def __init__(self, brain: ConnectomeRNN, encoder: ObservationEncoder, decoder: ActionDecoder,
                 rnn_steps: int = 4, input_gain: float = 5.0):
        super().__init__()
        self.brain, self.encoder, self.decoder = brain, encoder, decoder
        self.rnn_steps = rnn_steps
        self.input_gain = nn.Parameter(torch.tensor(float(input_gain)))
        self.value = value_head(decoder.n_features)
        self.register_buffer("h_rest", torch.zeros(brain.n))    # resting state; episodes start here

    @classmethod
    def build(cls, conn: Connectome, obs_space: gym.Space, act_space: gym.Space, rnn_steps: int = 4,
              input_gain: float = 5.0, alpha_init: float = 0.7, global_scale: float = 1.0, bias_init: float = 0.1,
              encoder: ObservationEncoder | None = None, decoder: ActionDecoder | None = None,
              readout_idx: torch.Tensor | None = None, readout_dim: int | None = 32,
              encoder_kw: dict | None = None, **rnn_kw) -> "FlyAgent":
        brain = ConnectomeRNN(conn, alpha_init=alpha_init, global_scale=global_scale, bias_init=bias_init, **rnn_kw)
        enc = encoder or ObservationEncoder.for_space(conn, obs_space, **(encoder_kw or {}))
        dec = decoder or ActionDecoder.for_space(conn, act_space, readout_idx, readout_dim)
        agent = cls(brain, enc, dec, rnn_steps=rnn_steps, input_gain=input_gain)
        agent.calibrate(obs_space)
        return agent

    @torch.no_grad()
    def calibrate(self, obs_space: gym.Space, n_probe: int = 16, steps: int = 64, rest_steps: int = 64,
                  seed: int = 0) -> float | None:
        """Set the resting state, the readout normalisation and the readout projection.

        The resting state is the activity after `rest_steps` steps with no input; episodes start
        there. The probe then runs `n_probe` parallel sequences of `steps` env steps from the
        resting state, each a smooth random walk through observation space (unbounded Box
        values clipped to +-3), keeping the readout state and the observation of every step.
        The normalisation comes from those states; the bottleneck projection is regressed from
        them onto the observations (vectors as they are, images through a PCA to the bottleneck
        width). Returns the projection fit's R^2, or None when the decoder has no bottleneck."""
        rng = np.random.default_rng(seed)
        obs_space.seed(int(rng.integers(2**31)))
        unbounded = isinstance(obs_space, gym.spaces.Box) and not (np.all(np.isfinite(obs_space.low)) and np.all(np.isfinite(obs_space.high)))
        dev = self.input_gain.device

        def sample():
            probe = np.stack([obs_space.sample() for _ in range(n_probe)])
            return torch.as_tensor(np.clip(probe, -3, 3) if unbounded else probe, device=dev).float()

        weights = self.brain.weights()
        h = torch.zeros(1, self.brain.n, device=dev)
        for _ in range(rest_steps):                              # settle with no input: the resting state
            h = self.brain.step(h, None, weights)
        self.h_rest.copy_(h[0])

        h = self.initial_state(n_probe)
        obs = sample()
        states, observed = [], []
        for _ in range(steps):
            obs = 0.8 * obs + 0.2 * sample()                     # smooth walk, as real observations are
            _, h = self.step(obs, h, weights)
            states.append(h)
            observed.append(obs)
        states, observed = torch.cat(states), torch.cat(observed)
        r2 = self.decoder.calibrate(states, _projection_targets(observed, self.decoder.n_features))
        self.value = value_head(self.decoder.n_features).to(dev)
        return r2

    @torch.no_grad()
    def calibrate_on_env(self, env: gym.Env, steps: int = 512, seed: int = 0) -> float | None:
        """Calibrate the readout on real observations: a random-policy rollout of `steps` env
        steps (episodes reset as they end), instead of the synthetic probe of `calibrate`.
        Needed for images, where random samples of the observation space are noise and say
        nothing about what frames look like. Uses the resting state set by `calibrate`."""
        dev = self.input_gain.device
        weights = self.brain.weights()
        obs, _ = env.reset(seed=seed)
        h = self.initial_state(1)
        states, observed, previous = [], [], []
        prev = torch.as_tensor(np.asarray(obs), device=dev).float()
        for _ in range(steps):
            x = torch.as_tensor(np.asarray(obs), device=dev).float()
            _, h = self.step(x.unsqueeze(0), h, weights)
            states.append(h); observed.append(x); previous.append(prev)
            prev = x
            obs, _, term, trunc, _ = env.step(env.action_space.sample())
            if term or trunc:
                obs, _ = env.reset(); h = self.initial_state(1); prev = torch.as_tensor(np.asarray(obs), device=dev).float()
        states, observed, previous = torch.cat(states), torch.stack(observed), torch.stack(previous)
        r2 = self.decoder.calibrate(states, _projection_targets(observed, self.decoder.n_features, previous))
        self.value = value_head(self.decoder.n_features).to(dev)
        return r2

    @property
    def n_neurons(self) -> int:
        return self.brain.n

    def initial_state(self, batch: int) -> torch.Tensor:
        """Episodes start from the network's resting state, not from silence: starting at zero
        makes the first ~10 steps of every episode a large transient (readout features 10x their
        steady-state scale) that dominates policy-gradient updates."""
        return self.h_rest.unsqueeze(0).expand(batch, -1).clone()

    def weights(self) -> Weights:
        """Derived parameters for an unroll; trainers compute this once per replayed segment."""
        return self.brain.weights()

    def param_groups(self, lr: float, brain_scale: float = 0.1, reference_fan_in: int = 64) -> list[dict]:
        """Optimizer groups with learning rates matched to each part of the agent.

        brain      lr * brain_scale: millions of edge gains under Adam each move by about lr per
                   update, which shifts the whole network; a smaller step keeps updates in the
                   trust region.
        heads      lr * min(1, reference_fan_in / n_features): with Adam the logit shift per
                   update grows with the number of head inputs, so heads reading more than
                   reference_fan_in features get a proportionally smaller rate (no scaling
                   with the default 32-dimensional readout bottleneck).
        the rest   lr (encoder projection, input gain, readout normalisation)."""
        brain, heads, rest = [], [], []
        for name, q in self.named_parameters():
            if not q.requires_grad:
                continue
            if name.startswith("brain."):
                brain.append(q)
            elif name.startswith("value.") or name.startswith("decoder.") and not name.startswith("decoder.norm."):
                heads.append(q)
            else:
                rest.append(q)
        head_lr = lr * min(1.0, reference_fan_in / self.decoder.n_features)
        return [{"params": rest, "lr": lr}, {"params": heads, "lr": head_lr}, {"params": brain, "lr": lr * brain_scale}]

    def step(self, obs: torch.Tensor, h: torch.Tensor, weights: Weights | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """One env step: obs (B, ...) and state h (B, N) -> readout features (B, R) and new h.

        Callers that unroll several steps with gradients should pass `weights=self.brain.weights()`
        computed once, so autograd does not keep E-sized intermediates per step."""
        weights = weights or self.brain.weights()
        drive = self.encoder.encode(obs) * self.input_gain
        u = torch.zeros_like(h).index_copy(1, self.encoder.idx, drive)
        for _ in range(self.rnn_steps):
            h = self.brain.step(h, u, weights)
        return self.decoder.features(h), h

    def forward(self, obs: torch.Tensor, h: torch.Tensor, weights: Weights | None = None):
        """Returns (action distribution, value (B,), new h)."""
        feats, h = self.step(obs, h, weights)
        return self.decoder.distribution(feats), self.value(feats).squeeze(-1), h

    def act(self, obs, h, greedy: bool = False):
        """Convenience for evaluation: returns (env action, new h)."""
        with torch.no_grad():
            dist, _, h = self(obs, h)
            a = dist.mode if greedy else dist.sample()
        return self.decoder.to_env(a), h

    def summary(self) -> str:
        return (f"FlyAgent: {self.n_neurons:,} neurons, {self.brain.pre.numel():,} edges | "
                f"{type(self.encoder).__name__} -> {self.encoder.n_inputs:,} input neurons | "
                f"{type(self.decoder).__name__} <- {self.decoder.n_readout:,} readout neurons | "
                f"{self.rnn_steps} rnn steps / env step")
