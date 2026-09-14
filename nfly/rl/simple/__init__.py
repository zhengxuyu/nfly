"""Simple, dependency-free trainers.  Read these to learn how the fly agent is trained;
use `nfly.rl.rllib` when you need a production RL stack."""

from .a2c import A2CConfig, train_a2c
from .ppo import PPOConfig, train_ppo

__all__ = ["A2CConfig", "train_a2c", "PPOConfig", "train_ppo"]
