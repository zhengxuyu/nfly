# Why the fly did not learn: an ablation log

A chronological record of the investigation into why the connectome network could not learn
CartPole or Pong, what each experiment tested, what it showed, and what changed as a result.
Everything here was run on 2026-09-14 with the `visual` (Pong) or `visual_small` (CartPole)
sub-network of MaleCNS v1.0, simple PPO unless stated, on one shared RTX 5090. "Return" is the
mean of the last 20 episodes. Reference points: CartPole random about 22, max 500; an MLP
through the same trainer reaches 174 at 100k steps with the current defaults.

## 0. Starting point

Model as first built: one network step per env step, leak alpha 0.3, resting bias 0.1,
LayerNorm across the 1,314 descending / motor readout neurons, linear heads, input gain 1.

| Run | Task | Steps | Return | Entropy | Observation |
| --- | --- | --- | --- | --- | --- |
| simple A2C | Pong | 321k | -20.5 | 0.64 | collapsed to two actions within 100k steps |
| simple PPO | Pong | 289k | -19.8 | 0.55 | slower collapse, no gain |
| RLlib PPO | Pong | 41k | -20.6 | 1.66 | stopped by a checkpoint-path bug |
| RLlib APPO | Pong | 755k | -20.3 | 1.71 | return and entropy flat throughout |
| simple PPO (v1) | CartPole | 51k | 23 | 0.53 | never above random |

Checkpoint inspection after 280k APPO steps: policy-head weights had moved by 0.001 (max 0.02),
edge gains by 0.005 std. The parameters were barely moving.

## 1. Is the training loop sound?

**Experiment.** `MLPReference`, a 4.7k-parameter tanh MLP with the FlyAgent interface, through
the identical simple PPO loop and config, CartPole.

**Result.** 18 -> 51 in 110 updates (old defaults); 174 at 100k steps (current defaults).

**Conclusion.** The loop learns. The fault is in the model or its interface.

## 2. Does the observation reach the readout at all?

**Experiment.** Frozen, untrained network. Drive with 64 random CartPole observations, run 8
steps, measure how much the 1,314 readout activities vary with the observation versus how much
they differ between neurons.

**Result.** Std across observations 0.00002; std across neurons 0.0435; ratio 0.0004. The
observation contributed one part in 2,500 of what the readout showed. Sweeping global gain
(1-5) and input gain (1-50) moved the ratio at most to 0.003.

**Conclusion.** The readout sat on a large fixed resting pattern; LayerNorm across neurons kept
that pattern, so the policy head's gradient pointed along a constant direction.
**Change:** `ReadoutNorm`, a per-neuron mean/scale standardisation, calibrated at build from a
probe, learnable afterwards. First version used running statistics; replaced by calibrated
parameters because running stats drift between rollout, replay and target-network passes.

A first calibration on 8 steps from the zero state clipped 93.6% of features at +-10 in real
play (the network drifts for tens of steps after a reset). Calibrating on a 64-step probe with
every step kept brought clipping to 0%.

## 3. Which observation components survive the path to the readout?

**Experiment.** Frozen network, real CartPole play with a heuristic controller, ridge-regress the
observation from the calibrated readout features; a logistic probe imitating the heuristic.

**Result.** R^2 [x 0.95, xdot 0.28, theta -0.66, thetadot -0.03]; imitation accuracy 49.5%
(chance). Standardising observations per dimension lifted theta to 0.61; thetadot stayed
undecodable. Sweeping network steps per env step and leak:

| steps / frame | alpha | R^2 [x, xdot, theta, thetadot] | imitation |
| --- | --- | --- | --- |
| 1 | 0.3 | [0.91, 0.61, 0.29, -0.49] | 51.8% |
| 4 | 0.3 | [0.42, 0.50, 0.68, -1.92] | 56.8% |
| 8 | 0.3 | [0.84, 0.89, 0.90, 0.91] | 80.6% |
| 1 | 0.7 | [0.98, -0.59, 0.74, -0.70] | 51.4% |
| **4** | **0.7** | **[0.93, 0.87, 0.93, 0.79]** | **87.8%** |
| 2 | 0.9 | [0.27, 0.16, 0.94, -3.2] | 51.8% |

**Conclusion.** With one network step per frame the 3-4 synaptic hops act as a low-pass filter
that erases fast components (angular velocity). **Change:** defaults rnn_steps 4, alpha 0.7,
input gain 5; `NormalizeObservation` on vector-observation suites.

**Run v2** (these defaults, old PPO settings): first run above random, 39 at update 30, 48 peak,
then a KL spike (0.165, clipfrac 0.68) and collapse to 9 by update 200.

## 4. Training configuration

**Experiment.** Sweep PPO settings on the MLP with a 100k-step budget.

| Config | Return at 100k |
| --- | --- |
| old defaults (lr 2.5e-4, 3 epochs, minibatch 4 envs, gamma 0.99, lambda 0.95, entropy 0.01) | 80 |
| CleanRL-style (rollout 128, 4 epochs) | 42 |
| SB3 rl-zoo (lr 1e-3, 10 epochs, minibatch 8 envs, gamma 0.98, lambda 0.8, entropy 0) | 223 |
| rl-zoo + target_kl 0.02 (adopted) | 174 |

**Change:** rl-zoo defaults; brain parameters at lr x0.1; epochs stop when approx KL > 0.02.

**Run v3** (fly, new defaults): collapsed within 10 updates (entropy 0.02). With Adam the logit
shift per update grows with the number of head inputs; 1,314 inputs at lr 1e-3 saturate at once.
**Change:** `FlyAgent.param_groups` scales the head lr by 64 / n_features.

**Run v4** (head lr 5e-5): stable for the first time, 40 at update 60, 63 at update 110 and
rising slowly, oscillating 20-40 in between; stopped in favour of the ablations below.

## 5. Is the representation sufficient?

**Experiment.** Frozen, untrained network; behaviour-clone a linear head from the calibrated
1,314-d features to the heuristic action on 4,000 steps; then play with that head.

**Result.** Fit accuracy 92.8%; the behaviour-cloned head scores **500/500 in all 10 episodes**,
the same as the heuristic on the raw state. A head fit to the action one frame earlier reached
94.7%, so the readout lags the state by about one frame.

**Conclusion.** The information is fully present and linearly readable. What remains is
optimisation: RL has to find that head through noisy policy gradients.

## 6. Where does the collapse come from?

**Experiment.** Freeze the brain (only encoder, readout and heads train); compare head lr.

| Head lr (1,314-d head) | Result |
| --- | --- |
| 1e-3 | entropy 0 by update 30, return 14 |
| 5e-5 | 29 at update 30, 48 at update 70, stable |

**Conclusion.** The collapse comes from the high-dimensional head, not the brain parameters.

## 7. Reduce the head's search space

**Experiment.** A 32-d linear bottleneck (no activation, orthogonal init) between readout and
heads; full and frozen-brain runs at lr 1e-3.

**Result.** Full: KL 0.10, clipfrac 0.58, return 12-17. Frozen: entropy 0.006 by update 50.

**Diagnosis of the features in real play.** The first 10 steps after every reset had feature
std 2.1 (max 16.5) versus 0.2 later: starting the state at zero produced a transient 10x the
steady-state scale. **Change:** episodes start from the network's resting state (`h_rest`,
settled with no input); the calibration probe starts there too. Transient gone (std 0.57 vs
0.65, max 3.7).

**Experiment.** Frozen brain, 32-d bottleneck, resting start; head lr 1e-4 and 3e-4.

**Result.** Both stable (entropy 0.66-0.69) and both at random level (20-27) at update 70.

**Diagnosis.** Only 40-50% of the 32-d feature variance was explained by the 4 observation
dimensions; the rest was the network's own drift. Regressing the 1,314-d normalised readout
onto the observations (no task labels) reconstructed them with held-out R^2 0.96-0.99, and a
head cloned on those 4 dims reached 97% with half the weight norm.

**Change:** the bottleneck projection is calibrated by ridge regression from the readout onto
the probe observations (vectors directly, images through a PCA to the bottleneck width), on a
smooth random walk through observation space from the resting state. Validation on the real
network: probe R^2 0.98; in real play R^2 [0.91, 0.95, 0.92, 0.93]; behaviour cloning 88% with
head norm 1.3.

**Run v5** (regression readout; full model, frozen brain, heads only): all three ended at
update 200 (100k steps) at random level: 21, 19 and 19. The representation was fine (section
5), so the remaining suspect was the head itself.

## 8. Is a linear policy the limiting factor?

**Experiment.** Does a one-frame delay plus exponential smoothing of the observation (the fly
readout's temporal profile) stop the MLP? Does a purely linear policy on the plain observation
learn through this PPO?

**Result.** Delayed and smoothed observation: MLP reaches 150 at update 120 (140 without), so
the temporal profile is not the blocker. A purely linear policy and value function on the raw
observation: 72 at update 60, then collapse to 19 by update 120.

**Conclusion.** The fly's linear head on a 4-d reconstruction of the observation inherits the
weakness of a linear policy under this trainer. The culprit is the linear value function
(CartPole's value is nonlinear in the state), which corrupts the advantages.

**Experiment.** Linear policy + 64-unit tanh MLP critic on the plain observation, two seeds.

**Result.** 127 and 164 at update 120, the same range as the full MLP policy (140-150).

**Conclusion.** A nonlinear critic is enough. It is training-only and does not change what the
fly computes when acting, so it is admissible. **Change:** the value head is a small tanh MLP
on the readout features; the policy stays linear.

**Run v6** (regression readout + MLP critic), 200 updates, 102k steps:

| Update | 70 | 90 | 110 | 130 | 150 | 170 | 190 | 200 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full model | 131 | 151 | 180 | 223 | 168 | 147 | 228 | 36 |
| brain frozen | 55 | 98 | 133 | 24 | 109 | 121 | 21 | 101 |
| MLP (same trainer) | 99 | ~130 | 168 | ~175 | 179 | ~175 | 174 | |

**Conclusion.** The fly learns CartPole, reaching and exceeding the MLP's level with the brain
trainable, and learns more slowly and less steadily with the brain frozen. Both oscillate:
updates with KL 0.02-0.03 cut the epochs to 1-2 by target_kl and still move the policy far
enough to lose 100-200 points in ten updates. Stabilising that (smaller head steps once the
policy is good, or lr annealing) is the next item.

## 9. Seeds: does it take off every time?

**Experiment.** Three seeds each of the v6 configuration (constant lr) and of v7 (v6 plus
linear lr annealing and KL-adaptive lr scaling), 200 updates, one run at a time.

| Config | seed 0 | seed 1 | seed 2 | took off |
| --- | --- | --- | --- | --- |
| v6, constant lr | 228 (peak) | 20 | 13 | 1 of 3 |
| v7, lr schedule | 24 | 30 | 285 (peak) | 1 of 3 |

**Conclusion.** The lr schedule is neutral for the fly and hurt the MLP (174 -> 109); it stays
available but off by default. The real
instability is not late oscillation but whether a run takes off at all: two thirds of the runs
never leave random level, while the MLP learns in every seed. Seed 2 stayed flat under v6 and
reached 285 under v7 with the same initialisation, so take-off is decided by the training
trajectory, not by the initial network.

## 10. RLlib on Pong, and what the readout carries

With the interface fixes and the official RLlib APPO recipe (batch 500, target-network update
every 2, entropy 0.01 -> 0, vf 1.0, grad clip 40, circular buffer 4 x 2), Pong v4 collapsed to
zero entropy within 25k steps; v5, with per-group learning rates ported into the RLlib learner
(brain x0.1, heads by fan-in), collapsed within 5k steps. APPO has no target-KL guard and
replays every batch twice; this track is parked until the readout question below is settled.

**Probe (scripts/probe_readout.py).** Random-policy Pong, ridge regression of ALE RAM quantities
from the readout. Held-out R^2 from all 1,314 readout neurons: ball x -0.03, ball y 0.24, ball
velocity -0.11 / -0.02, player paddle y 0.82, cpu paddle y 0.68. From the 32 calibrated
features: worse. The descending neurons carry the paddles and not the ball: a 2x4-pixel object
lands on one or two of 5,494 photoreceptors and is diluted away through the optic lobe. No
trainer can learn Pong from this readout.

**Stage-by-stage probe** (after making the probe robust: PCA to 256 components, ridge with
validated strength, interleaved block splits because the score digits drift through a game).
Held-out R^2, random-policy Pong:

| Stage | ball y | ball dy | player paddle y |
| --- | --- | --- | --- |
| photoreceptor drive (pixels sampled by the retina) | 0.33 | 0.42 | -0.09 |
| lamina L1-L5 | 0.63 | 0.41 | 0.86 |
| T4/T5 motion detectors | 0.61 | 0.46 | 0.86 |
| visual projection neurons | 0.66 | 0.49 | 0.87 |
| descending neurons (1,314) | 0.64 | 0.42 | 0.86 |
| 32 calibrated features (regressed onto [frame, diff] PCA) | 0.35 | 0.16 | 0.76 |

Ball x is not linearly decodable at any stage (it is not needed for Pong). Ball y and its
vertical velocity travel from the lamina to the descending neurons and are lost in the last
step, the calibrated projection: PCA targets of frames are dominated by paddles and score
digits. No PCA-based target (frame, [frame, diff], diff only, k up to 128) or the readout's own
PCA kept ball velocity.

**Temporal contrast.** Real photoreceptors and lamina cells respond to change and adapt to
steady light; the rate model has no adaptation. Adding the change since the previous frame to
the photoreceptor drive (gain 4) raised, at the descending neurons, ball y to 0.65 and ball dy
to 0.47 in the layered probe (0.66 / 0.64 in the offline test), with T4/T5 at 0.68 / 0.60.
**Change:** the Atari suite emits [frame, change] observations and the retina adds
temporal_gain x change to the drive.

**Readout bottleneck for images.** No frame-based target keeps the ball, so for images the
bottleneck is now the readout's own principal subspace, keeping the components that explain
99% of the probe variance (33 on Pong). Whitening all 128 components had turned near-null probe
directions into noise amplifiers (features of 500 in play; the first PPO update had KL 0.6 and
the policy was deterministic within ten updates). Features are also clipped after the
projection.

**Pong v7** (simple PPO, 16 envs on the GPU, temporal contrast, 33-d readout subspace, MLP
critic, gamma 0.99, lambda 0.95, clip 0.1, entropy 0.01, 4 epochs): 94 env steps / s, 2.3x the
RLlib CPU env runners; entropy 1.66-1.78 through the first 100 updates, no collapse; return
-20.4 at 51k steps (expected this early). Running to 1.5M steps.

## What is settled and what is open

Settled:
- The training loop is sound (MLP learns).
- The connectome transmits the full CartPole state to the descending neurons; a linear head
  fitted by supervision on the frozen network plays perfectly.
- Three interface defects were real and are fixed: readout hidden by the resting pattern,
  fast components filtered by one step per frame, a 10x transient at every reset.
- Collapse comes from high-dimensional or high-rate heads, not from the brain parameters.

Settled by v6:
- With the MLP critic, RL finds the head: 228 at update 190 versus the MLP's 174.
- Training the brain parameters helps: the frozen-brain run peaks lower (133) and collapses
  to random twice; the trainable run is steadier and higher.

Open:
- Take-off: only about one seed in three learns CartPole at all (section 9).
- Pong: ball y and vertical velocity do reach the descending neurons (R^2 0.65 / 0.47 with
  temporal contrast); the calibrated projection loses them. The readout bottleneck needs a
  target that keeps small moving objects, or no bottleneck for images.
- Take-off on CartPole is not fixed by entropy 0.01 (seeds 1, 2 stayed at 20-44) or by 64 envs
  (seed 1 stayed at 20).
- RLlib APPO collapses even with per-group learning rates; a KL guard is needed there.
- Everything above is CartPole; Pong will re-open the question of what the optic lobe
  contributes beyond transmission.
