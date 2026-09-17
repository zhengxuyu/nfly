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


def _is_image(space: gym.Space) -> bool:
    return isinstance(space, gym.spaces.Box) and len(space.shape) in (2, 3) and min(space.shape[-2:]) >= 8


class PixelCritic(nn.Module):
    """A small CNN over the observation for the value function only (asymmetric actor-critic).

    The actor stays the wiring plus one readout; at test time the critic is not used.
    This is an optional comparison to value prediction from the brain's readout."""

    def __init__(self, obs_shape: tuple[int, ...]):
        super().__init__()
        c, h, w = (obs_shape if len(obs_shape) == 3 else (1, *obs_shape))
        conv = nn.Sequential(nn.Conv2d(c, 16, 8, stride=4), nn.ReLU(), nn.Conv2d(16, 32, 4, stride=2), nn.ReLU(), nn.Flatten())
        with torch.no_grad():
            n = conv(torch.zeros(1, c, h, w)).shape[1]
        self.net = nn.Sequential(conv, nn.Linear(n, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs.float()
        return self.net(x.unsqueeze(1) if x.dim() == 3 else x)


def value_head(n_features: int, hidden: int = 64) -> nn.Module:
    """A small tanh MLP critic. The policy stays linear on the readout, so this changes nothing
    about how the fly acts; it only gives training a value function that can fit the task
    (a linear critic could not, and a linear policy with a linear critic collapsed on CartPole
    while the same policy with this critic learned as fast as an MLP policy)."""
    return nn.Sequential(nn.Linear(n_features, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))


class StandardizedCritic(nn.Module):
    """Readout critic with fixed statistics fitted on training trajectories only."""

    def __init__(self, n_features: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(n_features))
        self.register_buffer("scale", torch.ones(n_features))
        self.net = value_head(n_features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net((features - self.mean) / self.scale)


COARSE_MAP = 8   # images are reconstructed as coarse frame and motion maps of this size


def _projection_targets(observed: torch.Tensor, k: int, previous: torch.Tensor | None = None) -> torch.Tensor:
    """What the readout projection should reconstruct.

    Vector observations: as they are. Images: coarse maps (COARSE_MAP x COARSE_MAP, average
    pooled) of the frame and of the absolute change since the previous frame, i.e. where things
    are and where things move. Principal components of frames were tried first and lose small
    moving objects: on Pong the descending neurons carry ball y / vertical velocity at R^2
    0.66 / 0.50, frame-PCA features kept 0.35 / 0.16 and the readout's own PCA 0.31 / 0.15, the
    coarse maps 0.47 / 0.28, the best of the targets tried."""
    flat = observed.reshape(observed.shape[0], -1)
    if observed.dim() < 3 or flat.shape[1] <= k:                 # vectors are reconstructed as they are
        return flat
    frames = observed if observed.dim() == 3 else observed[:, 0]
    change = (observed[:, 1] if observed.dim() == 4 and observed.shape[1] == 2
              else frames - previous.reshape(frames.shape) if previous is not None else torch.zeros_like(frames))
    pool = lambda img: nn.functional.adaptive_avg_pool2d(img.unsqueeze(1), COARSE_MAP).flatten(1)
    return torch.cat([pool(frames), pool(change.abs())], 1)


class FlyAgent(nn.Module):
    """Defaults (rnn_steps 4, alpha 0.7, input_gain 5) come from a probe on CartPole: with one
    network step per env step and alpha 0.3, the 3-4 synaptic hops from sensory to descending
    neurons act as a low-pass filter that erases fast observation components (angular velocity
    was not linearly decodable at the readout, imitation of a competent policy was at chance);
    with four steps per env step and alpha 0.7 every observation dimension is decodable
    (R^2 0.8-0.93) and a linear probe imitates the policy at 88%."""

    def __init__(self, brain: ConnectomeRNN, encoder: ObservationEncoder, decoder: ActionDecoder,
                 rnn_steps: int = 4, input_gain: float = 5.0, share_trunk: bool = False, critic: str = "readout"):
        super().__init__()
        self.brain, self.encoder, self.decoder = brain, encoder, decoder
        self.rnn_steps = rnn_steps
        self.input_gain = nn.Parameter(torch.tensor(float(input_gain)))
        self.share_trunk = share_trunk          # critic on the policy head's hidden layer, so the value loss trains it too
        self.critic = critic
        self.obs_shape = tuple(encoder.space.shape) if hasattr(encoder, "space") else None
        self.value = self._new_value_head()
        self.register_buffer("h_rest", torch.zeros(brain.n))    # resting state; episodes start here

    def _new_value_head(self) -> nn.Module:
        if self.critic == "pixels":
            if self.obs_shape is None:
                raise ValueError("critic='pixels' needs an image observation")
            return PixelCritic(self.obs_shape)
        if self.critic == "standardized":
            if self.share_trunk:
                raise ValueError("The standardized critic requires the independent readout")
            return StandardizedCritic(self.decoder.n_features)
        return value_head(self.decoder.trunk_dim if self.share_trunk else self.decoder.n_features)

    def _value(self, feats: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        if self.critic == "pixels":
            return self.value(obs)
        return self.value(self.decoder.trunk(feats) if self.share_trunk else feats)

    @classmethod
    def build(cls, conn: Connectome, obs_space: gym.Space, act_space: gym.Space, rnn_steps: int = 4,
              input_gain: float = 5.0, alpha_init: float = 0.7, global_scale: float = 1.0, bias_init: float = 0.1,
              encoder: ObservationEncoder | None = None, decoder: ActionDecoder | None = None,
              readout_idx: torch.Tensor | None = None, readout_dim: int | None = None, head_hidden: int = 0,
              share_trunk: bool = False, critic: str = "readout",
              encoder_kw: dict | None = None, **rnn_kw) -> "FlyAgent":
        """readout_dim: width of the readout bottleneck; None picks 32 for vector observations
        (calibrated to reconstruct the observation) and 128 for images (calibrated to reconstruct
        coarse frame and motion maps); 0 means no bottleneck."""
        if readout_dim is None:
            readout_dim = 128 if _is_image(obs_space) else 32
        elif readout_dim == 0:
            readout_dim = None                                        # explicit: no bottleneck
        brain = ConnectomeRNN(conn, alpha_init=alpha_init, global_scale=global_scale, bias_init=bias_init, **rnn_kw)
        enc = encoder or ObservationEncoder.for_space(conn, obs_space, **(encoder_kw or {}))
        dec = decoder or ActionDecoder.for_space(conn, act_space, readout_idx, readout_dim, head_hidden)
        agent = cls(brain, enc, dec, rnn_steps=rnn_steps, input_gain=input_gain, share_trunk=share_trunk, critic=critic)
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
        self.value = self._new_value_head().to(dev)
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
        self.value = self._new_value_head().to(dev)
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
        heads      per weight matrix, lr * min(1, reference_fan_in / fan_in): with Adam the shift
                   of a layer's output per update grows with its fan-in, so a layer reading 1,314
                   readout neurons steps 20x more gently than one reading 64 features. Biases and
                   the layers of a bottlenecked head (fan-in <= reference) keep the full lr.
        the rest   lr (encoder projection, input gain, readout normalisation)."""
        rest, brain, by_lr = [], [], {}
        for name, q in self.named_parameters():
            if not q.requires_grad:
                continue
            if name.startswith("brain."):
                brain.append(q)
            elif name.startswith("value.") and self.critic == "pixels":
                rest.append(q)                        # a conventional CNN critic trains at the full rate
            elif name.startswith("value.") or name.startswith("decoder.") and not name.startswith("decoder.norm."):
                fan_in = q.shape[1] if q.dim() == 2 else 1
                by_lr.setdefault(lr * min(1.0, reference_fan_in / fan_in), []).append(q)
            else:
                rest.append(q)
        groups = [{"params": rest, "lr": lr}]
        groups += [{"params": ps, "lr": g_lr} for g_lr, ps in sorted(by_lr.items(), reverse=True)]
        groups.append({"params": brain, "lr": lr * brain_scale})
        return groups

    def step(self, obs: torch.Tensor, h: torch.Tensor, weights: Weights | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """One env step: obs (B, ...) and state h (B, N) -> readout features (B, R) and new h.

        Callers that unroll several steps with gradients should pass `weights=self.brain.weights()`
        computed once, so autograd does not keep E-sized intermediates per step."""
        h = self.trace(obs, h, weights)[-1]
        return self.decoder.features(h), h

    def drive(self, obs: torch.Tensor) -> torch.Tensor:
        """Input current delivered to the input neurons for obs: (B, K) in encoder order."""
        return self.encoder.encode(obs) * self.input_gain

    def trace(self, obs: torch.Tensor, h: torch.Tensor, weights: Weights | None = None) -> list[torch.Tensor]:
        """The state after each of the rnn_steps network steps of one env step; the last entry
        is the new h. Lets a viewer show activity propagating within a frame."""
        weights = weights or self.brain.weights()
        u = torch.zeros_like(h).index_copy(1, self.encoder.idx, self.drive(obs))
        states = []
        for _ in range(self.rnn_steps):
            h = self.brain.step(h, u, weights); states.append(h)
        return states

    def forward(self, obs: torch.Tensor, h: torch.Tensor, weights: Weights | None = None):
        """Returns (action distribution, value (B,), new h)."""
        feats, h = self.step(obs, h, weights)
        return self.decoder.distribution(feats), self._value(feats, obs).squeeze(-1), h

    def act(self, obs, h, greedy: bool = False):
        """Convenience for evaluation: returns (env action, new h)."""
        action, h, _, _ = self.act_traced(obs, h, greedy)
        return action, h

    def act_traced(self, obs, h, greedy: bool = False):
        """act() that also returns the action distribution and the per-sub-step states, so a
        viewer gets action, probabilities and brain activity from one pass."""
        with torch.no_grad():
            states = self.trace(obs, h)
            dist = self.decoder.distribution(self.decoder.features(states[-1]))
            a = dist.mode if greedy else dist.sample()
        return self.decoder.to_env(a), states[-1], dist, states

    def summary(self) -> str:
        return (f"FlyAgent: {self.n_neurons:,} neurons, {self.brain.pre.numel():,} edges | "
                f"{type(self.encoder).__name__} -> {self.encoder.n_inputs:,} input neurons | "
                f"{type(self.decoder).__name__} <- {self.decoder.n_readout:,} readout neurons | "
                f"{self.rnn_steps} rnn steps / env step")
