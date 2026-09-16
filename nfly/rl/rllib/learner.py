"""RLlib learners that give the fly's parameter groups their own learning rates.

RLlib's default learner uses one Adam optimizer at one learning rate for the whole module.
For the fly that means the millions of edge gains move as fast as the policy head, which
collapsed the policy within 25k Pong steps. These learners register one optimizer per group
from `FlyAgent.param_groups` (rest, heads, brain), each with the base lr or schedule scaled by
the group's multiplier; the learner's own scheduling and grad clipping apply unchanged.
"""

from __future__ import annotations

import torch
from ray.rllib.algorithms.appo.torch.appo_torch_learner import APPOTorchLearner
from ray.rllib.algorithms.impala.torch.impala_torch_learner import IMPALATorchLearner
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner

BRAIN_LR_SCALE = 0.1
REFERENCE_FAN_IN = 64


def _scaled(lr_or_schedule, factor: float):
    if isinstance(lr_or_schedule, (int, float)):
        return float(lr_or_schedule) * factor
    return [[t, v * factor] for t, v in lr_or_schedule]


class FlyOptimizerMixin:
    """Registers one optimizer per FlyAgent parameter group. Mix in before an RLlib torch learner."""

    def configure_optimizers_for_module(self, module_id, config=None):
        module = self._module[module_id]
        agent = getattr(module, "agent", None)
        if agent is None or not hasattr(agent, "param_groups"):
            return super().configure_optimizers_for_module(module_id, config)
        for name, group in zip(("rest", "heads", "brain"), agent.param_groups(1.0, BRAIN_LR_SCALE, REFERENCE_FAN_IN)):
            params = [p for p in group["params"] if p.requires_grad]
            if not params:
                continue
            self.register_optimizer(module_id=module_id, optimizer_name=name, optimizer=torch.optim.Adam(params),
                                    params=params, lr_or_lr_schedule=_scaled(config.lr, group["lr"]))


class FlyPPOTorchLearner(FlyOptimizerMixin, PPOTorchLearner):
    pass


class FlyAPPOTorchLearner(FlyOptimizerMixin, APPOTorchLearner):
    pass


class FlyIMPALATorchLearner(FlyOptimizerMixin, IMPALATorchLearner):
    pass


LEARNERS = {"PPO": FlyPPOTorchLearner, "APPO": FlyAPPOTorchLearner, "IMPALA": FlyIMPALATorchLearner}
