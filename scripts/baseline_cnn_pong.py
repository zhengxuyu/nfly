"""CNN baseline for Pong, same machine and budget as the fly runs.

The configuration is RLlib's own tuned Atari PPO example
(rllib/examples/algorithms/ppo/atari_ppo.py: WarpFrame 64x64 grayscale, max-and-skip 4,
4-frame stacking via connectors, a 4-layer CNN with a 256-unit head, PPO with batch 4000,
minibatch 128, 10 epochs, lr 1.5e-4, clip 0.1, entropy 0.01, kl 0.5, grad clip 100), which
the RLlib team reports solving Pong in about 3M steps. It is here so the benchmark table can
carry a conventional policy trained on the same hardware.

    uv run scripts/baseline_cnn_pong.py --env-runners 8 --gpus 1 --stop-steps 3000000
"""

from __future__ import annotations

import argparse
import json
import os

import gymnasium as gym
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.connectors.env_to_module.frame_stacking import FrameStackingEnvToModule
from ray.rllib.connectors.learner.frame_stacking import FrameStackingLearner
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.env.wrappers.atari_wrappers import wrap_atari_for_new_api_stack
from ray.tune.registry import register_env


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="ale_py:ALE/Pong-v5")
    p.add_argument("--env-runners", type=int, default=8)
    p.add_argument("--envs-per-runner", type=int, default=2)
    p.add_argument("--gpus", type=float, default=1)
    p.add_argument("--stop-steps", type=int, default=3_000_000)
    p.add_argument("--stop-return", type=float, default=18.0)
    p.add_argument("--out", default="runs/baseline-cnn-pong")
    args = p.parse_args()

    register_env("atari_cnn_baseline", lambda cfg: wrap_atari_for_new_api_stack(
        gym.make(args.env, **cfg, render_mode="rgb_array"), framestack=None))
    config = (
        PPOConfig()
        .environment("atari_cnn_baseline",
                     env_config={"frameskip": 1, "full_action_space": False, "repeat_action_probability": 0.0},
                     clip_rewards=True)
        .env_runners(num_env_runners=args.env_runners, num_envs_per_env_runner=args.envs_per_runner,
                     env_to_module_connector=lambda env, spaces, device: FrameStackingEnvToModule(num_frames=4))
        .learners(num_learners=0, num_gpus_per_learner=args.gpus)
        .training(learner_connector=lambda obs_space, act_space: FrameStackingLearner(num_frames=4),
                  train_batch_size_per_learner=4000, minibatch_size=128, lambda_=0.95, kl_coeff=0.5,
                  clip_param=0.1, vf_clip_param=10.0, entropy_coeff=0.01, num_epochs=10, lr=0.00015,
                  grad_clip=100.0, grad_clip_by="global_norm")
        .rl_module(model_config=DefaultModelConfig(conv_filters=[[16, 4, 2], [32, 4, 2], [64, 4, 2], [128, 4, 2]],
                                                   conv_activation="relu", head_fcnet_hiddens=[256], vf_share_layers=True))
    )
    ray.init(ignore_reinit_error=True)
    algo = config.build_algo()
    i = 0
    while True:
        i += 1
        r = algo.train()
        er = r.get("env_runners", {})
        steps = int(r.get("num_env_steps_sampled_lifetime", 0))
        ret = er.get("episode_return_mean")
        print(json.dumps({"iter": i, "steps": steps, "return_mean": None if ret is None else float(ret),
                          "episodes": float(er.get("num_episodes", 0) or 0), "seconds": round(float(r.get("time_this_iter_s", 0)), 1)}), flush=True)
        if i % 10 == 0:
            algo.save_to_path(os.path.abspath(args.out))
        if steps >= args.stop_steps or (ret is not None and ret >= args.stop_return):
            algo.save_to_path(os.path.abspath(args.out))
            break


if __name__ == "__main__":
    main()
