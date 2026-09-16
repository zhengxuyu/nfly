"""FlyRLModule: a FlyAgent wrapped as a stateful RLlib RLModule (new API stack).

RLlib hands stateful modules batches shaped (B, T, ...) plus `state_in`; we unroll the
FlyAgent over T and return action-distribution inputs, the final state and the readout
features (as EMBEDDINGS, so PPO/IMPALA can compute values from them).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch
from ray.rllib.core.columns import Columns
from ray.rllib.core.learner.utils import make_target_network
from ray.rllib.core.rl_module.apis.target_network_api import TARGET_NETWORK_ACTION_DIST_INPUTS, TargetNetworkAPI
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.models.torch.torch_distributions import TorchCategorical, TorchDiagGaussian

from ...agent import FlyAgent
from ...connectome import load_malecns, select_subset
from ...suite import get_suite

STATE_KEY = "h"
VALUE_CHUNK = 8    # sequences per chunk when the GAE connector asks for values of a whole train batch


class FlyRLModule(TorchRLModule, ValueFunctionAPI, TargetNetworkAPI):
    """model_config keys: data_dir, subset, min_syn, rnn_steps, max_seq_len (RLlib BPTT horizon).

    ValueFunctionAPI serves PPO / IMPALA / APPO; TargetNetworkAPI (a lagged copy of the whole
    agent, synced by the learner) is what APPO's v-trace correction needs."""

    def setup(self) -> None:
        cfg = self.model_config
        conn = select_subset(load_malecns(cfg.get("data_dir", "data"), min_syn=cfg.get("min_syn", 3)),
                             cfg.get("subset", "visual"))
        self.agent = FlyAgent.build(conn, self.observation_space, self.action_space,
                                    rnn_steps=cfg.get("rnn_steps", 4))
        if "suite" in cfg and "game" in cfg:                    # calibrate the readout on real observations
            self.agent.calibrate_on_env(get_suite(cfg["suite"]).make(cfg["game"], seed=cfg.get("seed", 0) + 1000),
                                        steps=cfg.get("calibration_steps", 512))

    def get_initial_state(self) -> dict[str, np.ndarray]:
        return {STATE_KEY: self.agent.h_rest.detach().cpu().numpy().astype(np.float32)}

    def _dist_cls(self):
        return TorchCategorical if isinstance(self.action_space, gym.spaces.Discrete) else TorchDiagGaussian

    def get_inference_action_dist_cls(self):
        return self._dist_cls()

    def get_exploration_action_dist_cls(self):
        return self._dist_cls()

    def get_train_action_dist_cls(self):
        return self._dist_cls()

    def _unroll(self, batch: dict[str, Any], agent: FlyAgent | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """(B, T, obs...) + state_in -> readout features (B, T, R) and state_out."""
        agent = agent or self.agent
        obs = batch[Columns.OBS]
        h = batch[Columns.STATE_IN][STATE_KEY]
        weights = agent.brain.weights()
        feats = []
        for t in range(obs.shape[1]):
            f, h = agent.step(obs[:, t], h, weights)
            feats.append(f)
        return torch.stack(feats, dim=1), {STATE_KEY: h}

    # ---- TargetNetworkAPI (APPO) -----------------------------------------------------------------
    def make_target_networks(self) -> None:
        self.target_agent = make_target_network(self.agent)

    def get_target_network_pairs(self):
        return [(self.agent, self.target_agent)]

    def forward_target(self, batch: dict[str, Any]) -> dict[str, Any]:
        with torch.no_grad():
            feats, _ = self._unroll(batch, self.target_agent)
            return {TARGET_NETWORK_ACTION_DIST_INPUTS: self.target_agent.decoder.dist_inputs(feats)}

    def get_non_inference_attributes(self) -> list[str]:
        """The target copy lives on the learner only; env runners never need it."""
        return ["target_agent"]

    def _forward(self, batch: dict[str, Any], **kwargs) -> dict[str, Any]:
        feats, state_out = self._unroll(batch)
        return {Columns.ACTION_DIST_INPUTS: self.agent.decoder.dist_inputs(feats), Columns.STATE_OUT: state_out}

    def _forward_train(self, batch: dict[str, Any], **kwargs) -> dict[str, Any]:
        feats, state_out = self._unroll(batch)
        return {Columns.ACTION_DIST_INPUTS: self.agent.decoder.dist_inputs(feats),
                Columns.STATE_OUT: state_out, Columns.EMBEDDINGS: feats}

    def compute_values(self, batch: dict[str, Any], embeddings: Any = None) -> torch.Tensor:
        """With `embeddings` (from forward_train) this is the differentiable value head.  Without
        them RLlib's GAE connector is asking for bootstrap values of an entire train batch: that
        needs no gradient, so unroll chunk by chunk to keep the (sequences x neurons) state small."""
        if embeddings is not None:
            return self.agent.value(embeddings).squeeze(-1)
        with torch.no_grad():
            values = [self.agent.value(self._unroll(chunk)[0]).squeeze(-1) for chunk in _split_batch(batch, VALUE_CHUNK)]
        return torch.cat(values, dim=0)


def _split_batch(batch: dict[str, Any], size: int):
    """Yield sub-batches of `size` sequences: (obs (B,T,...), state_in {h: (B,N)}) slices."""
    n = batch[Columns.OBS].shape[0]
    for a in range(0, n, size):
        yield {Columns.OBS: batch[Columns.OBS][a:a + size],
               Columns.STATE_IN: {k: v[a:a + size] for k, v in batch[Columns.STATE_IN].items()}}
