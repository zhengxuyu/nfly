"""Build RLlib AlgorithmConfigs (PPO / APPO / IMPALA) for the fly agent.

    config = build_config("PPO", suite="atari", game="pong", subset="visual", num_gpus=1)
    algo = config.build_algo()
    for _ in range(100):
        print(algo.train()["env_runners"]["episode_return_mean"])
"""

from __future__ import annotations

from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.tune.registry import get_trainable_cls

from .env import register_nfly_env
from .learner import LEARNERS
from .module import FlyRLModule


def build_config(algo: str = "PPO", suite: str = "atari", game: str = "pong", data_dir: str = "data",
                 subset: str = "visual", min_syn: int = 3, rnn_steps: int = 4, max_seq_len: int = 32,
                 num_env_runners: int = 4, num_envs_per_env_runner: int = 2, num_gpus: float = 0,
                 train_batch_size: int = 2048, minibatch_size: int = 256, num_epochs: int = 3,
                 lr: float = 2.5e-4, **training_kw) -> AlgorithmConfig:
    """A ready-to-run config; extra keyword args go to `config.training(...)` (e.g. entropy_coeff)."""
    env = register_nfly_env(suite, game)
    module = RLModuleSpec(module_class=FlyRLModule,
                          model_config={"data_dir": data_dir, "subset": subset, "min_syn": min_syn,
                                        "rnn_steps": rnn_steps, "max_seq_len": max_seq_len,
                                        "suite": suite, "game": game})
    config = (get_trainable_cls(algo).get_default_config()
              .environment(env)
              .env_runners(num_env_runners=num_env_runners, num_envs_per_env_runner=num_envs_per_env_runner,
                           rollout_fragment_length=max_seq_len)
              .learners(num_learners=0, num_gpus_per_learner=num_gpus)
              .rl_module(rl_module_spec=module)
              # minibatch_size bounds the learner's unroll batch for every algorithm; without it
              # APPO / IMPALA unroll the whole train batch at once and run out of GPU memory
              # per-group learning rates (brain x0.1, heads by fan-in), as in the simple trainers
              .training(learner_class=LEARNERS.get(algo), train_batch_size_per_learner=train_batch_size,
                        minibatch_size=minibatch_size, lr=lr, **training_kw))
    if algo == "PPO":
        config = config.training(num_epochs=num_epochs)
    return config
