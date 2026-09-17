"""Argument groups shared by the scripts, so each script only adds what is specific to it."""

from __future__ import annotations

import argparse
import logging

from .connectome import Connectome, load_malecns, select_subset, SUBSETS


def add_connectome_args(p: argparse.ArgumentParser, subset: str = "all") -> None:
    p.add_argument("--data", default="data", help="directory with the MaleCNS feather files")
    p.add_argument("--subset", default=subset, choices=list(SUBSETS))
    p.add_argument("--min-syn", type=int, default=3, help="drop neuron pairs with fewer synapses")


def connectome_from_args(args: argparse.Namespace) -> Connectome:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    return select_subset(load_malecns(args.data, min_syn=args.min_syn), args.subset)


def agent_kwargs(args: argparse.Namespace) -> dict:
    """FlyAgent.build keyword arguments derived from the agent argument group."""
    kw = {"rnn_steps": args.rnn_steps}
    rd = getattr(args, "readout_dim", None)
    if rd is not None:
        kw["readout_dim"] = rd                                         # 0 = no bottleneck
    if getattr(args, "head_hidden", 0):
        kw["head_hidden"] = args.head_hidden
    if getattr(args, "split_eyes", False):
        kw["encoder_kw"] = {"split": True}
    if getattr(args, "share_trunk", False):
        kw["share_trunk"] = True
    if getattr(args, "critic", "readout") != "readout":
        kw["critic"] = args.critic
    if getattr(args, "freeze_brain", False) or getattr(args, "heads_only", False):
        kw.update(learn_gain=False, learn_alpha=False, learn_bias=False)
    return kw


def calibrate_on(agent, env, log=print):
    """Calibrate the readout on real observations from `env` (a random-policy rollout)."""
    r2 = agent.calibrate_on_env(env)
    if r2 is not None:
        log(f"readout calibrated on {type(env.unwrapped).__name__}: projection R^2 {r2:.3f}, {agent.decoder.n_features} features")
    return agent


def apply_freezes(agent, args: argparse.Namespace):
    """Post-build freezes that go beyond the brain: --heads-only leaves only the heads trainable."""
    if getattr(args, "heads_only", False):
        for name, q in agent.named_parameters():
            q.requires_grad_(name.startswith(("value.", "decoder.head", "decoder.mean", "decoder.log_std")))
    return agent


def add_agent_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--rnn-steps", type=int, default=4, help="network steps per env step")
    p.add_argument("--freeze-brain", action="store_true", help="train only encoder, readout and heads; keep the connectome parameters fixed")
    p.add_argument("--readout-dim", type=int, help="linear bottleneck width between readout neurons and heads (default: 32 for vectors, 128 for images; 0 = none)")
    p.add_argument("--heads-only", action="store_true", help="train only the policy and value heads; freeze brain, encoder and readout calibration")
    p.add_argument("--head-hidden", type=int, default=0, help="diagnostic: tanh MLP policy head of this width (0 = linear head, the default)")
    p.add_argument("--critic", default="readout", choices=["readout", "pixels", "standardized"], help="value input: brain readout, pixels, or readout with fixed training-set statistics")
    p.add_argument("--share-trunk", action="store_true", help="critic reads the policy head's hidden layer (needs --head-hidden), so the value loss also trains it")
    p.add_argument("--split-eyes", action="store_true", help="retina: each eye sees its half of the frame at double density (default: both eyes see the whole frame)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)


def rescale_policy(agent, temperature: float) -> None:
    """Set an explicit starting temperature without changing the greedy discrete policy."""
    import torch
    from .interface import DiscreteDecoder
    if temperature <= 0:
        raise ValueError("Initial policy temperature must be positive")
    if temperature == 1.0:
        return
    if not isinstance(agent.decoder, DiscreteDecoder):
        raise ValueError("Initial policy temperature requires discrete actions")
    head = agent.decoder.head
    last = head[-1] if isinstance(head, torch.nn.Sequential) else head
    with torch.no_grad():
        last.weight.div_(temperature)
        last.bias.div_(temperature)


def load_probe_critic(agent, path, gamma: float) -> None:
    """Initialize only the value function from the fixed-policy diagnostic."""
    import torch
    from .agent import StandardizedCritic
    payload = torch.load(path, map_location=agent.input_gain.device, weights_only=True)
    if abs(payload["gamma"] - gamma) > 1e-8:
        raise ValueError("Probe gamma differs from the training gamma")
    key = payload["input_key"]
    expected = {"readout": "features", "pixels": "pixels", "standardized": "standardized"}[agent.critic]
    if key != expected or agent.share_trunk:
        raise ValueError("Probe input does not match the independent critic architecture")
    if isinstance(agent.value, StandardizedCritic):
        mean, scale = payload["feature_mean"], payload["feature_scale"]
        if not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("Probe normalization must be finite with positive scales")
        agent.value.net.load_state_dict(payload["state_dict"])
        agent.value.mean.copy_(mean)
        agent.value.scale.copy_(scale)
    else:
        agent.value.load_state_dict(payload["state_dict"])
