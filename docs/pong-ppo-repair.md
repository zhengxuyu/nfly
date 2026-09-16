# Controlled Pong PPO fine-tuning

The 2026-09-16 audit found inconsistent recurrent resets, incomplete feature freezing in
experiments labelled frozen-brain, and incomparable greedy imitation versus sampled PPO
scores. The gradient-alignment diagnostic used an untrained critic and teacher agreement,
not an oracle Pong advantage. These findings do not establish a single cause of failure.

## Fixes and controls

- Collection, replay and the gradient diagnostic reset completed episodes with
  `agent.initial_state()`. Unfinished recurrent states are preserved.
- GAE bootstraps time limits from the final observation and pre-reset state. True terminals
  do not bootstrap, and neither boundary carries advantages into the next episode.
- PPO checks exact distribution KL over the collected rollout after every optimizer step
  and stops further minibatches when the threshold is exceeded. This detects overshoot;
  it does not undo a step or guarantee a hard KL bound. The extra replay costs throughput.
- `--heads-only` freezes the entire feature path. `--freeze-brain` retains its existing,
  documented meaning: encoder and readout normalization remain trainable.
- `--set critic_warmup=N` trains only `value.*` for the first N updates, then restores the
  original trainable parameter set. The feature path and actor stay fixed during warmup.
- Explicit paired evaluation reports greedy and sampled scores on identical seed lists,
  separately from training returns. It preserves training RNG and uses fresh environments.
  Incomplete evaluations fail rather than reporting partial scores as complete episodes.
- Evaluation includes Monte Carlo return prediction MSE and explained variance, excluding
  time-truncated episodes from value metrics. These are independent of the training GAE
  bootstrap targets. Use gamma 0.99 when comparing these diagnostics to this Pong recipe.
- `*-best.pt` retains the best greedy validation score, including the initial checkpoint.
  This selection set is not an unbiased final test set; use new seeds for final reporting.
- `--init-temperature` is an explicit policy intervention. Its default 1.0 leaves the
  checkpoint unchanged. Positive rescaling preserves argmax, not sampled trajectories.

## Zero-learning baseline

Use an immutable copy of the DAgger checkpoint and record its SHA256. Do not mix the old-eye
checkpoint with the current retina. Run in a dedicated worktree with the GPU environment:

```bash
PYTHONPATH=. ../nfly/.venv/bin/python scripts/train_rl.py \
  --data ../nfly/data --suite atari --game pong --subset visual \
  --device cuda --envs 4 --readout-dim 0 --head-hidden 64 --heads-only \
  --init runs/dagger-start.pt --eval-only --eval-episodes 4 --eval-seed 10000 \
  --eval-temperatures 1 0.25 0.1 --eval-json runs/baseline.jsonl
```

A temperature sweep is validation, not proof of a robust +20 policy. Keep these seeds separate
from a final evaluation of at least 20 fresh starts. Training always uses sampled actions.

## Conservative first control

Start with the existing separate readout critic and original checkpoint temperature, so the
implementation repairs are not confounded with changing the critic or exploration policy:

```bash
PYTHONPATH=. ../nfly/.venv/bin/python scripts/train_rl.py \
  --data ../nfly/data --suite atari --game pong --subset visual \
  --device cuda --envs 4 --readout-dim 0 --head-hidden 64 --heads-only \
  --init runs/dagger-start.pt --rollout 128 --updates 200 --lr 0.0001 \
  --set gamma=0.99 --set lam=0.95 --set clip=0.1 --set entropy=0.0 \
  --set epochs=2 --set minibatch_envs=4 --set critic_warmup=40 \
  --set target_kl=0.01 --set save_every=20 --set adaptive_lr=true \
  --eval-every 40 --eval-episodes 4 --eval-seed 10000 \
  --eval-json runs/control-eval.jsonl --out runs/control.pt
```

This is a 102,400-transition diagnostic, not a claim that this budget solves Pong. Inspect
warmup held-out value metrics before interpreting actor learning. Compare a pixel critic on
the same data/protocol if readout value prediction remains weak. Compare rollout 32 versus
128 with matched environment steps, then test a measured temperature intervention separately.
Do not unfreeze the brain or normalization until this control is understood.
