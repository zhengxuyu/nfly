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
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.models.torch.torch_distributions import TorchCategorical, TorchDiagGaussian

from ...agent import FlyAgent
from ...connectome import load_malecns, select_subset

STATE_KEY = "h"


class FlyRLModule(TorchRLModule, ValueFunctionAPI):
    """model_config keys: data_dir, subset, min_syn, rnn_steps, max_seq_len (RLlib BPTT horizon)."""

    def setup(self) -> None:
        cfg = self.model_config
        conn = select_subset(load_malecns(cfg.get("data_dir", "data"), min_syn=cfg.get("min_syn", 3)),
                             cfg.get("subset", "visual"))
        self.agent = FlyAgent.build(conn, self.observation_space, self.action_space,
                                    rnn_steps=cfg.get("rnn_steps", 2))

    def get_initial_state(self) -> dict[str, np.ndarray]:
        return {STATE_KEY: np.zeros(self.agent.n_neurons, dtype=np.float32)}

    def _dist_cls(self):
        return TorchCategorical if isinstance(self.action_space, gym.spaces.Discrete) else TorchDiagGaussian

    def get_inference_action_dist_cls(self):
        return self._dist_cls()

    def get_exploration_action_dist_cls(self):
        return self._dist_cls()

    def get_train_action_dist_cls(self):
        return self._dist_cls()

    def _unroll(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """(B, T, obs...) + state_in -> readout features (B, T, R) and state_out."""
        obs = batch[Columns.OBS]
        h = batch[Columns.STATE_IN][STATE_KEY]
        feats = []
        for t in range(obs.shape[1]):
            f, h = self.agent.step(obs[:, t], h)
            feats.append(f)
        return torch.stack(feats, dim=1), {STATE_KEY: h}

    def _forward(self, batch: dict[str, Any], **kwargs) -> dict[str, Any]:
        feats, state_out = self._unroll(batch)
        return {Columns.ACTION_DIST_INPUTS: self.agent.decoder.dist_inputs(feats), Columns.STATE_OUT: state_out}

    def _forward_train(self, batch: dict[str, Any], **kwargs) -> dict[str, Any]:
        feats, state_out = self._unroll(batch)
        return {Columns.ACTION_DIST_INPUTS: self.agent.decoder.dist_inputs(feats),
                Columns.STATE_OUT: state_out, Columns.EMBEDDINGS: feats}

    def compute_values(self, batch: dict[str, Any], embeddings: Any = None) -> torch.Tensor:
        if embeddings is None:
            embeddings, _ = self._unroll(batch)
        return self.agent.value(embeddings).squeeze(-1)
