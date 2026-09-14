"""FlyAgent = ObservationEncoder -> ConnectomeRNN -> ActionDecoder (+ value head).

The agent is a recurrent policy with an explicit state `h` (one scalar per neuron).  It is
built from a connectome and a pair of Gymnasium spaces, so any env works:

    agent = FlyAgent.build(conn, env.observation_space, env.action_space)
    h = agent.initial_state(batch)
    dist, value, h = agent(obs_tensor, h)
"""

from __future__ import annotations

import gymnasium as gym
import torch
from torch import nn

from .brain.rnn import ConnectomeRNN, Weights
from .connectome.base import Connectome
from .interface.decoders import ActionDecoder
from .interface.encoders import ObservationEncoder


class FlyAgent(nn.Module):
    def __init__(self, brain: ConnectomeRNN, encoder: ObservationEncoder, decoder: ActionDecoder,
                 rnn_steps: int = 2, input_gain: float = 1.0):
        super().__init__()
        self.brain, self.encoder, self.decoder = brain, encoder, decoder
        self.rnn_steps = rnn_steps
        self.input_gain = nn.Parameter(torch.tensor(float(input_gain)))
        self.value = nn.Linear(decoder.n_readout, 1)

    @classmethod
    def build(cls, conn: Connectome, obs_space: gym.Space, act_space: gym.Space, rnn_steps: int = 2,
              input_gain: float = 1.0, alpha_init: float = 0.3, global_scale: float = 1.0, bias_init: float = 0.1,
              encoder: ObservationEncoder | None = None, decoder: ActionDecoder | None = None,
              readout_idx: torch.Tensor | None = None, encoder_kw: dict | None = None, **rnn_kw) -> "FlyAgent":
        brain = ConnectomeRNN(conn, alpha_init=alpha_init, global_scale=global_scale, bias_init=bias_init, **rnn_kw)
        enc = encoder or ObservationEncoder.for_space(conn, obs_space, **(encoder_kw or {}))
        dec = decoder or ActionDecoder.for_space(conn, act_space, readout_idx)
        return cls(brain, enc, dec, rnn_steps=rnn_steps, input_gain=input_gain)

    @property
    def n_neurons(self) -> int:
        return self.brain.n

    def initial_state(self, batch: int) -> torch.Tensor:
        return torch.zeros(batch, self.brain.n, device=self.input_gain.device)

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
