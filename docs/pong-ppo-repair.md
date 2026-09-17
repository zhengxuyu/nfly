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

## Completed temperature controls (2026-09-17)

All three runs completed 102,400 transitions. The same four validation seeds were used
throughout; these are neither independent temperature selection nor a final test set.

| Initial temperature | Initial sampled mean | Final sampled mean | Final greedy mean | Final sampled MC value EV |
| --- | --- | --- | --- | --- |
| 1.0 | -12.75 | -15.00 | 10.25 | 0.0050 |
| 0.25 | 8.25 | 8.50 | 11.50 | 0.0661 |
| 0.1 | 10.00 | 12.00 | 10.00 | 0.0537 |

No run improved on the initial greedy mean of 11.50. The losing opening persists. Lower
temperature repairs much of the initial sampling gap without learning, but does not establish
stable improvement from PPO. Low fitted value EV does not prove insufficient features.

## Fixed-policy full-return diagnosis

`scripts/probe_value.py` records complete episodes from an immutable checkpoint without actor
updates. It fits readout and pixel critics to the same Monte Carlo targets. Whole episodes
are split into training, validation and test sets; only validation selects the fitted model.
Different seeds may still produce similar trajectories in deterministic Atari, so this small
probe is a diagnostic rather than evidence of robust generalization.

```bash
PYTHONPATH=. ../nfly/.venv/bin/python scripts/probe_value.py \
  --data ../nfly/data --suite atari --game pong --subset visual \
  --device cuda --readout-dim 0 --head-hidden 64 --heads-only \
  --init runs/dagger-start.pt --episodes 12 --seed 11000 --temperature 0.25 \
  --dataset runs/value-probe-t025.pt --fit-steps 1000 \
  --out runs/value-probe-t025.json
```

The readout MLP is fitted at the PPO optimizer rate and an unscaled higher rate. A wider
readout MLP and the existing PixelCritic test capacity on the same targets. In the first
condition, the first readout layer receives 1e-4 * 64/1314, approximately 4.87e-6. This is an
optimization comparison, not a matched-parameter architecture benchmark. Use `--fit-only`
with the saved dataset to repeat fits without changing the policy or collecting new data.

### Results and next training control

The first 12 new starts (11000-11011) scored -2.08 on average at T=0.25; the independent
24-start repeat (12000-12023) scored -2.42. The earlier four validation starts were favorable.
These are sampled-policy measurements, not a new measurement of the greedy policy.

The 24-episode dataset splits into 16 training, 4 validation and 4 test episodes. Fixed-target
fits used 1000 Adam steps with batch size 256. The checkpoint with lowest validation MSE was
tested. Readout and actor parameters remained unchanged throughout collection and fitting.

| Critic input and optimizer | Validation MC EV | Test MC EV |
| --- | --- | --- |
| Readout, original PPO rate and fan-in scaling | 0.331 | 0.224 |
| Readout, unscaled learning rate 1e-3 | 0.650 | 0.475 |
| Wider readout MLP, unscaled 1e-3 | 0.658 | 0.489 |
| Four causal readout frames, unscaled 1e-3 | 0.718 | 0.616 |
| Readout standardized on training episodes only, unscaled 1e-3 | 0.756 | 0.621 |
| PixelCritic, unscaled 1e-3 | 0.780 | 0.647 |

The readout has usable value information. Optimization rate, conditioning and temporal input
matter more here than simply widening the MLP. However, pooled EV partly reflects differences
between winning and losing episodes: the standardized critic's per-test-episode EV is
0.128, 0.034, 0.282 and 0.536, with pooled within-episode-centered EV 0.276. This does not
establish an accurate advantage estimator or improved gameplay. The 12-episode pilot also
failed to generalize when its validation and test splits contained only losing episodes.

`--critic standardized` adds fixed training-set mean/scale to the value path only.
`--init-critic` loads the compatible fitted value network and statistics without changing
the actor. The new `--set critic_lr=0.001` overrides only value-parameter rates; other
optimizer groups retain their existing rates. Statistics stay fixed during PPO and are
included in the agent checkpoint. Training and evaluation use the same discount factor.

Use the saved probe artifact with the existing recipe, adding:

```bash
--critic standardized \
--init-critic runs/value-probe-24-expanded.readout_standardized.pt \
--set critic_lr=0.001 --set gamma=0.99
```

This warm-start spends additional transitions on critic pretraining and is not a
matched-total-budget comparison with the earlier PPO controls. Subsequent gameplay must
be evaluated on fresh starts, separately from critic fitting and checkpoint selection.
