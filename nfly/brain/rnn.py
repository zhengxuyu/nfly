"""Connectome-constrained rate RNN.

    h[t+1] = (1 - alpha) * h[t] + alpha * phi( W h[t] + b + u[t] )

W is sparse with fixed connectivity and fixed signs (Dale's law); every edge has a
learnable positive gain on top of the connectome-derived magnitude.  alpha (per-neuron
leak, i.e. 1/time-constant) and b (bias / negative threshold) are also per-neuron parameters.
"""

from __future__ import annotations

import dataclasses
from typing import NamedTuple

import torch
from torch import nn

from ..connectome.base import Connectome


@dataclasses.dataclass
class InputDrive:
    """External input restricted to a subset of neurons: drive[b, t, k] goes to node idx[k]."""

    idx: torch.Tensor      # (K,) long
    drive: torch.Tensor    # (B, T, K) float

    @property
    def steps(self) -> int:
        return self.drive.shape[1]

    @property
    def batch(self) -> int:
        return self.drive.shape[0]

    @staticmethod
    def constant(idx: torch.Tensor, amplitude: float, steps: int, batch: int = 1,
                 on: tuple[int, int] | None = None) -> "InputDrive":
        """Constant drive of `amplitude` on nodes idx, optionally only during steps [on[0], on[1])."""
        d = torch.zeros(batch, steps, len(idx))
        a, b = on if on else (0, steps)
        d[:, a:b, :] = amplitude
        return InputDrive(torch.as_tensor(idx, dtype=torch.long), d)


class Weights(NamedTuple):
    """Per-forward derived parameters: effective edge weights (E,) and per-neuron leak (N,).
    Compute once per unroll (`ConnectomeRNN.weights()`), not once per step: the E-sized
    intermediates would otherwise be kept by autograd for every step."""

    w: torch.Tensor
    alpha: torch.Tensor


EDGE_CHUNK = 2_000_000   # edges processed per chunk; transient memory is batch x chunk floats


class _SparseRecurrent(torch.autograd.Function):
    """y[b, post] += h[b, pre] * w, processed in edge chunks so that neither the forward nor
    the backward pass ever materialises a full (B, E) tensor, and only h (B, N) and w (E,) are
    saved for backward."""

    @staticmethod
    def forward(ctx, h, w, pre, post, chunk):
        ctx.save_for_backward(h, w, pre, post)
        ctx.chunk = chunk
        y = torch.zeros_like(h)
        for a in range(0, pre.numel(), chunk):
            b = a + chunk
            y.index_add_(1, post[a:b], h[:, pre[a:b]] * w[a:b])
        return y

    @staticmethod
    def backward(ctx, grad_y):
        h, w, pre, post = ctx.saved_tensors
        chunk = ctx.chunk
        need_h, need_w = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        grad_h = torch.zeros_like(h) if need_h else None
        grad_w = torch.empty_like(w) if need_w else None
        for a in range(0, pre.numel(), chunk):
            b = a + chunk
            g = grad_y[:, post[a:b]]                                   # (B, chunk)
            if need_h:                                                 # dL/dh[b, pre] += dL/dy[b, post] * w
                grad_h.index_add_(1, pre[a:b], g * w[a:b])
            if need_w:                                                 # dL/dw = sum_b h[b, pre] * dL/dy[b, post]
                grad_w[a:b] = (h[:, pre[a:b]] * g).sum(0)
        return grad_h, grad_w, None, None, None


class ConnectomeRNN(nn.Module):
    def __init__(self, conn: Connectome, alpha_init: float = 0.1, activation: str = "relu",
                 learn_gain: bool = True, learn_alpha: bool = True, learn_bias: bool = True,
                 global_scale: float = 1.0, bias_init: float = 0.0, h_max: float | None = 10.0,
                 edge_chunk: int = EDGE_CHUNK):
        """
        alpha_init   leak per step (1/time-constant), per neuron, learnable
        global_scale multiplier on the normalised connectome weights
        bias_init    resting drive.  >0 gives every neuron tonic activity so that inhibitory
                     inputs (e.g. histaminergic photoreceptors) can be *read* by ReLU units.
        h_max        saturation: activity is clamped to [0, h_max] to keep long rollouts finite
        edge_chunk   edges per chunk in the sparse product (bounds transient memory)
        """
        super().__init__()
        self.n = conn.n_neurons
        self.register_buffer("pre", conn.pre)
        self.register_buffer("post", conn.post)
        self.register_buffer("sign", conn.sign)
        self.register_buffer("w0", conn.weight * global_scale)   # connectome magnitude, fixed

        e, n = conn.n_edges, self.n
        self.log_gain = nn.Parameter(torch.zeros(e), requires_grad=learn_gain)
        self.bias = nn.Parameter(torch.full((n,), float(bias_init)), requires_grad=learn_bias)
        self.h_max = h_max
        self.edge_chunk = edge_chunk
        self._cached: tuple[int, Weights] | None = None
        a = torch.full((n,), float(alpha_init))
        self.alpha_logit = nn.Parameter(torch.log(a / (1 - a)), requires_grad=learn_alpha)
        self.act = {"relu": torch.relu, "tanh": torch.tanh, "softplus": nn.functional.softplus}[activation]

    # ---- derived quantities -------------------------------------------------
    def edge_weights(self) -> torch.Tensor:
        """Signed effective weight of every edge, shape (E,)."""
        return self.sign * self.w0 * torch.exp(self.log_gain)

    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha_logit)

    def weights(self) -> Weights:
        """Derived parameters for one unroll.  Under no_grad (inference) the result is cached
        and only recomputed after the parameters were modified in place (optimizer step,
        load_state_dict), which is what `_version` tracks."""
        if torch.is_grad_enabled():
            return Weights(self.edge_weights(), self.alpha())
        version = self.log_gain._version + self.alpha_logit._version
        if self._cached is None or self._cached[0] != version:
            self._cached = (version, Weights(self.edge_weights(), self.alpha()))
        return self._cached[1]

    def recurrent_input(self, h: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """(W h) for a batch h of shape (B, N) using gather + index_add (works on CPU/CUDA/MPS).

        Uses a custom autograd function so that backprop through time stores only h (B, N)
        per step instead of the (B, E) message tensor."""
        return _SparseRecurrent.apply(h, w, self.pre, self.post, self.edge_chunk)

    # ---- dynamics --------------------------------------------------------------
    def step(self, h: torch.Tensor, u: torch.Tensor | None, weights: Weights) -> torch.Tensor:
        x = self.recurrent_input(h, weights.w) + self.bias
        if u is not None:
            x = x + u
        x = self.act(x)
        if self.h_max is not None:
            x = x.clamp(max=self.h_max)
        return (1 - weights.alpha) * h + weights.alpha * x

    def forward(self, drive: InputDrive | None = None, steps: int | None = None, batch: int = 1,
                h0: torch.Tensor | None = None, record: torch.Tensor | None = None) -> torch.Tensor:
        """Run the dynamics.

        Returns the state history (B, T+1, N) or, if `record` (node indices) is given,
        only those nodes: (B, T+1, len(record)).  Recording a subset keeps memory small
        for the 139k-neuron whole brain.
        """
        if drive is not None:
            steps, batch = drive.steps, drive.batch
        assert steps is not None, "give either `drive` or `steps`"
        dev = self.w0.device
        h = torch.zeros(batch, self.n, device=dev) if h0 is None else h0.to(dev)
        weights = self.weights()
        hist = [h if record is None else h[:, record]]
        for t in range(steps):
            u = None
            if drive is not None:
                u = torch.zeros(batch, self.n, device=dev)
                u[:, drive.idx.to(dev)] = drive.drive[:, t].to(dev)
            h = self.step(h, u, weights)
            hist.append(h if record is None else h[:, record])
        return torch.stack(hist, dim=1)

    def extra_repr(self) -> str:
        return f"neurons={self.n:,}, edges={self.pre.numel():,}"
