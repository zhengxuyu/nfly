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

**Readout bottleneck for images.** Every linear bottleneck tried loses part of the ball.
Held-out R^2 of ball y / ball vertical velocity from the calibrated features, Pong, random
play (the raw 1,314-d normalised readout has 0.66 / 0.50):

| Projection target | ball y | ball dy |
| --- | --- | --- |
| frame PCA, k=32 / 128 | 0.32 / 0.50 | -0.33 / -0.61 |
| [frame, diff] PCA, k=32 (first image target) | 0.05 | -0.08 |
| readout's own PCA, k=26 (99% variance) / 64 / 128 | 0.31 / 0.41 / 0.45 | 0.15 / 0.25 / 0.22 |
| coarse 8x8 maps of frame and |change|, 128-d (adopted) | 0.47 | 0.28 |

Whitening all 128 readout components had also turned near-null probe directions into noise
amplifiers (features of 500 in play; the first PPO update had KL 0.6 and the policy was
deterministic within ten updates); features are now clipped after the projection as well.

**Run v7** (26-d readout subspace): 435k steps, return -21, entropy 1.5, no learning; stopped.
**Run v8a** (no bottleneck: linear head on all 1,314 descending neurons, MLP critic): 717k
steps, return -20.65, entropy 0.9-1.5, no learning. Twice the budget in which the CNN solved
Pong on the same machine. Removing the bottleneck did not help, so the loss at the bottleneck
was not what stood between the fly and Pong.

**Run v8c** (as v8a with a 64-unit tanh MLP policy head, a diagnostic control that is not the
model's claim): 660k steps, return -20.35, entropy 1.68, no learning. (A first launch had the
whole head scaled to lr x 64/1314 by the fan-in rule and never moved; the rule now scales each
layer by its own fan-in.) So the linear readout is not the limit either: the ball signal at the
descending neurons (R^2 0.66 for position, 0.50 for vertical velocity) is too weak for control,
and the work moves upstream, to the retina encoding and the network dynamics.

## 11. Upstream: retina and dynamics (probe sweep)

Probe sweeps on Pong, ball position / ball vertical velocity (R^2) at the descending neurons:

| Setting (rnn 4, alpha 0.7 unless noted) | ball y | ball dy |
| --- | --- | --- |
| temporal gain 4 (previous default) | 0.64 | 0.56 |
| temporal gain 8 | 0.69 | 0.60 |
| surround 2 / 4, temporal gain 4 | 0.72 / 0.73 | 0.61 / 0.62 |
| surround 4, temporal gain 8 | 0.72 | 0.65 |
| alpha 0.5 / 0.9 | 0.75 / 0.63 | 0.48 / 0.60 |
| rnn 8, alpha 0.5 / 0.7 / 0.9 | 0.64 / 0.60 / 0.58 | 0.58 / 0.65 / 0.61 |
| input gain 2 / 10 (surround 4, gain 8) | 0.70 / 0.73 | 0.61 / 0.63 |
| **both eyes sampling the whole frame** (surround 4, gain 8) | **0.86** | **0.68** (ball x velocity 0.19 -> 0.65) |

Dynamics settings and input gain move little; the spatial and temporal high-pass help by
about 0.1 each; sampling density is the lever: letting both eyes see the whole frame instead of
one hemifield each doubles the photoreceptors a small object hits. **Change:** defaults are now
full-field sampling, surround 4, temporal gain 8.

**Run v9** (these defaults, otherwise as v8a): 916k steps, return -20.55, entropy 1.2, no
learning. With ball y at R^2 0.86 and paddle y at 0.84 in the readout, a linear rule (move
towards the ball) is representable, yet RL did not find it in 2.5x the CNN's budget. Next test:
behaviour cloning of a heuristic Pong teacher on the frozen readout (`scripts/bc_pong.py`), the
same sufficiency test that settled CartPole, to separate "the readout cannot support control"
from "RL cannot find the head under Pong's sparse, delayed reward".

**Pong v7** (simple PPO, 16 envs on the GPU, temporal contrast, 33-d readout subspace, MLP
critic, gamma 0.99, lambda 0.95, clip 0.1, entropy 0.01, 4 epochs): 94 env steps / s, 2.3x the
RLlib CPU env runners; entropy 1.66-1.78 through the first 100 updates, no collapse; return
-20.4 at 51k steps; -21.0 at 310k steps with entropy 1.37 (running to 1.5M steps).

**CNN baseline on the same machine** (`scripts/baseline_cnn_pong.py`, RLlib's tuned Atari
PPO with 4-frame stacking): -19 at 280k steps, +6 at 320k, +14.5 at 352k, **+19 at 356k**,
25 minutes wall clock. That is the bar: a conventional policy learns Pong here in a third of
a million steps; the fly at the same step count shows no movement.

## 12. Is the Pong readout sufficient? (behaviour cloning from the CNN)

**Experiment.** `scripts/bc_pong.py --readout-dim 0 --teacher runs/baseline-cnn-pong`: frozen,
untrained `visual` network with the v9 retina, no readout bottleneck (all 1,314 descending
neurons); the trained CNN baseline (+19 in its own env) is the teacher. It reads the raw
210x160 screen through RLlib's own grayscale / 64x64 / uint8 / 128 - 1 preprocessing and a
4-frame stack, so it plays inside the fly's suite env; its score there is printed as a control.
Record 6,000 steps (30% random actions for coverage), fit a head by cross-entropy, play 3
episodes with the head alone.

Two pitfalls on the way, both visible as a bad teacher rather than a bad clone: feeding the
suite's [0, 1] frames to a CNN trained on [-1, 1] gave a one-action teacher (-21); resizing the
suite's 84x84 frame instead of the raw screen left it at -3.

**Result.**

| Head on the frozen 1,314-d readout | Fit accuracy (train / held-out) | Cloned head plays | Teacher in the same env |
| --- | --- | --- | --- |
| linear | 61.3% / 55.6% | **-9.7** (episodes 12, -20, -21) | 6.0 (19, 16, -17) |
| 64-unit tanh MLP | 85.5% / 62.3% | **+8.0** (episodes 19, 20, -15) | 6.0 (19, 16, -17) |

Both heads reproduce the teacher only about 60% of the time on held-out steps, yet the MLP head
wins two of three episodes 19-20 to the CNN's 19 and 16, and the linear head wins one (+12).
The third episode is a start state that also beats the teacher (-17). Nothing trained by RL on
this readout ever left -20.

**Conclusion.** The frozen, untrained connectome with the calibrated readout carries enough
information to play Pong at the CNN's level: a 64-unit head fitted from 6,000 supervised
steps does it. The failure of v7-v9 (up to 916k RL steps at -20.5) is therefore not the retina,
the dynamics, or the readout; it is credit assignment. Pong's reward arrives 20-60 frames after
the decisive paddle move, and the 1,314-d (linear) or 85k-parameter (MLP) head has to be found
from that signal alone, where the CNN's convolutional inductive bias makes the same search easy.
The natural next steps are on the RL side: initialise the head from behaviour cloning and let
PPO continue (DAgger-style), or shape the reward with the ball-paddle distance, before touching
the brain again.

## 13. PPO from the cloned head: the readout normalisation was never trainable at that rate

**Experiment.** `train_rl.py --init` starts PPO (v9 settings) from the behaviour-cloned heads of
section 12, logits tempered to 1.0 nats so the sampled policy can still move (sampled play:
linear -18.7, MLP -5.3).

**Result.** Update 1 had KL 722 (linear) and 7.3 (MLP); entropy went to 0.000 and 0.005 and the
gradient to zero. Both runs were dead after one update. The tempering changed nothing (fitted
logits were already at 1.2 / 0.8 nats), so the cause was not over-confident logits.

Four-update diagnostics with the same linear checkpoint:

| Trainable | KL at update 1 | Entropy after |
| --- | --- | --- |
| everything (default) | 722 | 0.000 |
| brain frozen (`--freeze-brain`) | 719 | 0.000 |
| brain lr x 0.001 | 720 | 0.000 |
| heads only (`--heads-only`) | 0.000 | 0.99 (unchanged) |

Neither the brain nor the heads: what remained trainable in the second row was the encoder's
input gain and the readout normalisation's per-neuron `mean` and `log_scale`, in the "rest"
group at the full lr 1e-3. A descending neuron's activity spread is far below 1e-3 in raw
units, so one Adam step on `mean` moved every feature by many standard deviations, the
features saturated at the clip, and the linear head's logits went with them.

**Fix.** `ReadoutNorm` keeps the calibrated `mean` and `scale` as fixed buffers and trains a
per-neuron `shift` and `log_gain` in standardised units, both zero after calibration; a step
of 1e-3 is now 1e-3 of a spread. Old checkpoints do not load (different state keys).

**After the fix**, the same four-update diagnostic (linear cloned head, v9 settings):

| Trainable | KL at update 1 | Entropy after 4 updates |
| --- | --- | --- |
| brain frozen | 0.001 | 1.09 |
| brain lr x 0.01 (1e-5) | 10.6 | 0.000 |
| brain lr x 0.001 (1e-6) | 0.34 | 0.75 (epochs early-stopped) |
| brain lr x 0.0001 (1e-7) | 0.003 | 1.11 |

The normalisation no longer blows up, but the brain does at any learning rate above about
1e-7, four orders of magnitude below the head's. The reason is in the calibrated statistics:
99.6% of the descending neurons have a spread at the `min_std` floor of 1e-4 against a resting
level of about 0.1 (median mean / spread 1,069). The observation-driven part of a descending
neuron's activity is below one thousandth of its resting activity, so a brain step that moves
that activity by 0.1% moves the feature by a full spread, and 1% saturates it. Input gain 5
to 200 leaves the statistics unchanged to four digits: photoreceptors are inhibitory onto
lamina neurons resting at 0.1, so beyond a little light they are silenced and the signal
amplitude is capped by the resting level, then decays through the layers.

Without the floor (`min_std` 1e-12), the true spread of a descending neuron over a random
rollout is 1.6e-6 (median; 1st to 99th percentile 8e-8 to 4e-5), 1.7e-5 of its resting level.
The floor therefore compresses the features about 60-fold below unit variance, and the head
undoes that with large weights, which is the same sensitivity seen from the other side. The
global synaptic scale is not a clean lever: at 1.5 some neurons already sit at the activity
ceiling `h_max` = 10, and at 4.0 half the descending neurons are still at the floor while their
median resting level has fallen from 0.107 to 0.020. Runaway subcircuits saturate before the
visual signal grows.

**Consequence for the earlier rows.** Every RL run since v2 had these parameters drifting at
lr 1e-3 under a zero-initialised head, where the drift is invisible in KL but still moves the
features the head is trying to read. That is a candidate cause for the CartPole take-off
lottery (section 9) and for part of the Pong failure; v9 and the CartPole seeds should be
rerun with the fix before any further model changes.

## 14. The viewer finds a distorted eye

**Observation.** The new viewer panel that draws every photoreceptor at the position where it
samples the frame showed each eye covering a triangle, not an oval: the left eye the lower-left
half of the frame, the right eye the lower-right, the top strip sampled by almost nothing.

**Cause.** `hex_to_xy` turned the MaleCNS hex column coordinates into plane positions with the
shear x = h1 + h2 / 2. For the release's axes the sign is the other way: with + the columns of
one eye form a diagonal band (x-y correlation 0.80, filling 46% of the bounding box), which the
per-eye bounding-box normalisation stretches into two triangles; with x = h1 - h2 / 2 the
correlation is 0.03 and the fill 80%, a round eye.

**Fix.** The sign. Photoreceptor sampling positions change for every image task, so the retina
sweep (section 11), the probes, v9 and the behaviour-cloning results (section 12) were all
obtained with the distorted eye and are due for a rerun. The other findings (readout floor,
normalisation drift, brain sensitivity) do not depend on where the eye samples.

**Second look.** With the round eye the panel showed the remaining gap: an oval on a rectangle.
Measured on an 84x84 frame (a pixel counts as seen if a photoreceptor samples within 2 px):

| Eye mapping | whole frame | paddle columns (outer 12%) | top / bottom 12% |
| --- | --- | --- | --- |
| round eye, bounding box to frame | 85% | 59% | 54% |
| round eye warped onto the square (`fill_frame`, elliptical-grid mapping) | 98% | 92% | 94% |

Pong's paddles live in the outer columns and the ball turns at the top and bottom walls, so
the first row means the eye barely saw the events that decide a point. The readout probe
agrees (frozen network, held-out R^2 at the descending neurons, same retina settings for both
rows: temporal gain 4, no surround):

| Eye mapping | ball x | ball y | player paddle y | cpu paddle y |
| --- | --- | --- | --- | --- |
| round eye, oval on the frame | 0.20 | 0.65 | 0.74 | 0.49 |
| round eye warped onto the square | 0.23 | 0.72 | 0.75 | 0.81 |

The CPU paddle, in the far column, is the target the oval lost; ball x, never decodable with
the distorted eye, is now weakly present at both settings. `fill_frame=True` is
now the default; it is a deliberate distortion of the eye's field of view onto the game frame,
documented in docs/design.md. The 5,494 photoreceptors occupy 1,633 distinct positions (the
R1-R8 of one ommatidium share a column), unchanged by the warp.

## 15. DAgger on the frozen fly: +13.4

**Experiment.** `scripts/dagger_pong.py` with the frame-filling eye and the 64-unit MLP head:
round 0 records 12,000 steps of the CNN teacher (30% random actions), every later round
records 12,000 steps of the student's own greedy play with the teacher's action as the label,
and the head is refitted on the aggregate (3,000 Adam steps). Greedy evaluation on 5 fixed
starts after each round. Brain, encoder and readout calibration frozen throughout.

| Round | Labelled steps | Fit accuracy | Greedy return (5 starts) |
| --- | --- | --- | --- |
| 0 (teacher only) | 12k | 79.4% | -3.4 (-20, 20, -17, -20, 20) |
| 1 | 24k | 81.2% | -1.6 |
| 2 | 36k | 81.7% | 9.2 (7, 20, -8, 7, 20) |
| 3 | 48k | 80.3% | -3.8 |
| 4 | 60k | 81.1% | 13.0 (17, 20, -9, 17, 20) |
| 5 | 72k | 80.6% | 12.4 |
| 6 | 84k | 81.0% | 13.2 |
| 7 | 96k | 81.0% | **13.4** (18, 20, -9, 18, 20) |

**Result.** From round 4 on, four of the five starts end at +17 to +20 and one start is lost
every time; the mean settles at +13. Pong is deterministic given the start, so a greedy head
either falls into a winning rally or does not, and the losing start needs its own labelled
states. One-pass cloning on 6,000 steps (section 12, +8.0 with the old eye, -15 with the new
one) was limited by exactly this: no labels on the student's own mistakes.

**Meaning.** The frozen wiring plus one readout plays Pong near the human reference (14.6).
Every RL attempt from the same or a weaker start stayed flat or eroded (sections 13 and 14),
so the gap between +13 and -16 is what the policy-gradient signal fails to deliver on this
representation, not what the representation lacks. Next: more DAgger rounds seeded from this
head for the losing start, the linear head through the same procedure (the model's claim), and
then PPO from the DAgger head with a KL guard and a dense shaping reward.

## 16. Why RL fails where supervision works: the advantage is noise

**Experiment.** `scripts/grad_align.py` at the DAgger head (section 15): 8 envs x 256 steps of
the sampled policy (entropy 1.0), keeping readout features, actions, the critic's values,
rewards and the CNN teacher's label per step. On 32 random minibatches of 256 the gradient on
the head's parameters is computed three ways and compared with the supervised cross-entropy
gradient towards the teacher: cosine similarity per batch, cosine of the batch means, and the
signal-to-noise ratio across batches (norm of the mean gradient over the mean norm of the
deviations).

Data: 26 nonzero rewards in 2,048 steps, teacher agreement 51%, **critic explained variance
0.001**.

| Gradient on the head | cos with supervised (per batch) | cos of batch means | SNR |
| --- | --- | --- | --- |
| supervised (cross-entropy to the teacher) | 1.000 | 1.000 | 0.78 |
| policy gradient, GAE with the fly's critic (what PPO follows) | 0.052 | -0.002 | 0.41 |
| oracle advantage: +1 if the action is the teacher's, else -1 | 0.667 | 0.778 | 0.83 |
| random advantage (noise floor) | 0.027 | 0.087 | 0.45 |

**Result.** What PPO follows is indistinguishable from the random-advantage control. With a
perfect advantage the same head, features and estimator point where supervision points (cos
0.67 to 0.78, SNR above the supervised gradient's). The representation and the policy head are
not the problem; the advantage estimate is, and it is broken because the critic, a separate
MLP on the frozen readout, explains nothing of the return under 1 reward per 80 steps.

**Why the CNN learns.** In the CNN actor-critic the policy and value heads share the
convolutional trunk, and the value regression is a dense signal at every step. The trunk is
shaped by that signal, the critic becomes informative, and the actor's gradient acquires a
direction. On the fly, the trunk is frozen and the two heads are separate, so the only dense
signal in RL never reaches what the policy reads. The equal-parameter MLP baseline has the
same separation and the same failure.

**Consequences.** Two fixes follow directly and are cheap: (a) `--share-trunk`, the critic on
the policy head's hidden layer so the value loss trains it; (b) a privileged critic during
training (the ALE RAM state, or a small CNN on pixels), which makes the advantage informative
without touching the fly. Both keep the agent's policy "wiring plus one readout" at test time.

**Test of (a).** PPO from the DAgger head (greedy +13.4; sampled at 1.0 nats about -14), brain
frozen, v9 settings, 16 envs:

| Critic | Updates | Return (last 20) | Value loss |
| --- | --- | --- | --- |
| separate MLP on the readout (the usual) | 340 | -17.6, flat between -14 and -18 | 0.03 to 0.13, no trend |
| shared trunk (`--share-trunk`) | 250 | -20.6, eroding | 0.06 to 0.11, no trend |

Sharing the trunk does not make the critic learn on the frozen readout; the value gradient only
disturbs the policy's hidden layer. The 64-unit layer on top of the readout is not enough of a
representation for the return, which needs where the ball is heading over the next 20 to 60
frames, not just where it is.

**Test of (b), first attempt.** `--critic pixels` (a two-layer CNN over the observation, value
only) from the same head: -17.6 at update 140 with the value loss again flat at 0.03 to 0.11.
This run had the CNN critic's dense layer under the fan-in learning-rate scaling meant for the
readout heads (lr 2.5e-5), which is fixed; the rerun at the full rate is pending on the shared
GPU.

**Supervised controls run alongside** (DAgger, 8 rounds, 10 greedy starts):

| Readout | Head | Best greedy return | Fit accuracy |
| --- | --- | --- | --- |
| descending neurons (`visual`) | 64-unit MLP | +13.4 (5 starts) / +10.8 (10 starts) | 85% |
| descending neurons (`visual`) | 256-unit MLP | +8.6 | 82% |
| descending neurons (`visual`) | linear | -18.6 | 69% |
| descending + motor neurons (`all`, VNC included) | 64-unit MLP | +10.3 | 82% |
| descending + motor neurons (`all`, VNC included) | linear | -19.2 | 71% |

A wider head does not fix the lost starts (capacity is not the limit; 3 of 10 starts are lost
by every head), and reading the motor neurons through the real ventral nerve cord does not make
the policy linearly readable either: on the untrained wiring the VNC adds no useful
nonlinearity on top of the descending neurons. The linear-head claim therefore does not hold
on Pong with the untrained connectome; the 64-unit head stays the working configuration.
Reward shaping (ball to paddle distance) attacks the sparsity itself. The linear head is a
separate limit: DAgger with a linear readout stays at -19 (fit accuracy 69%), so Pong's policy
is not linearly readable from the descending neurons; the MLP head's 64 units are needed.

## What is settled and what is open

Settled:
- The training loop is sound (MLP learns).
- The connectome transmits the full CartPole state to the descending neurons; a linear head
  fitted by supervision on the frozen network plays perfectly.
- Three interface defects were real and are fixed: readout hidden by the resting pattern,
  fast components filtered by one step per frame, a 10x transient at every reset.
- Collapse comes from high-dimensional or high-rate heads, not from the brain parameters.
- Pong: the frozen, untrained network with the full-field retina and the 1,314-d readout
  supports Pong at the CNN's level; a 64-unit head behaviour-cloned from the CNN in 6,000
  steps scores +19 / +20 in two of three episodes (section 12).

Settled by v6:
- With the MLP critic, RL finds the head: 228 at update 190 versus the MLP's 174.
- Training the brain parameters helps: the frozen-brain run peaks lower (133) and collapses
  to random twice; the trainable run is steadier and higher.

Open:
- Take-off: only about one seed in three learns CartPole at all (section 9).
- Pong, RL: no run has left -20.5 (v7-v9, up to 916k steps), although the readout is
  sufficient (section 12). Those runs, and the CartPole seeds, trained the readout
  normalisation at a rate that saturates the features (section 13); rerun with the fix, then
  behaviour-cloned initialisation or reward shaping if they still fail.
- Take-off on CartPole is not fixed by entropy 0.01 (seeds 1, 2 stayed at 20-44) or by 64 envs
  (seed 1 stayed at 20).
- RLlib APPO collapses even with per-group learning rates; a KL guard is needed there.
- Rerun the retina sweep, v9 and the Pong behaviour cloning with the round eye (section 14).
- The wiring's potential advantages are unmeasured: sample efficiency, transfer, robustness to
  input perturbations, and cost on event-driven hardware. Each needs a matched comparison
  against the equal-parameter MLP and the CNN.
- The encoder is fixed apart from one global gain; a learnable version (input, surround and
  temporal gains, possibly a gain per photoreceptor, geometry still fixed) is untried.
- What the optic lobe contributes beyond transmission is still unmeasured; section 12 shows
  transmission alone is enough for Pong once a head is found.
