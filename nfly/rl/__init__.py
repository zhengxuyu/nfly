"""Training algorithms for the FlyAgent.

    nfly.rl.simple   pure-PyTorch A2C and PPO, short and readable (for learning and quick experiments)
    nfly.rl.rllib    Ray RLlib integration: FlyRLModule + config builders for PPO / APPO / IMPALA
                     (for scaling out: many env runners, GPUs, checkpoints, Tune)
"""

from .simple import A2CConfig, PPOConfig, train_a2c, train_ppo

__all__ = ["A2CConfig", "PPOConfig", "train_a2c", "train_ppo"]
