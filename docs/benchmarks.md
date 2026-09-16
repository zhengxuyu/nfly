# Benchmarks and performance

Scores obtained so far, recorded as they are, including negative results. "Return" is the mean
episode return of the last 20 episodes; "MLP" is `nfly.rl.simple.reference.MLPReference`, an
ordinary 4.7k-parameter network run through the same trainer and config, so the fly can be
compared against a conventional policy under identical conditions. The experiment-by-experiment
record is [ablation-cartpole.md](ablation-cartpole.md).

## CartPole-v1 (max 500, random about 22)

Simple PPO, 16 envs, rollout 32. Fly runs use the `visual_small` sub-network (106,579
neurons, 5.2M edges) on one shared RTX 5090; the MLP runs on a laptop CPU.

| Model | Trainer config | Env steps | Return | Notes |
| --- | --- | --- | --- | --- |
| MLP | old defaults (lr 2.5e-4, 3 epochs, minibatch 4 envs, gamma 0.99, lambda 0.95, entropy 0.01) | 100k | 80 | loop works, but conservative settings learn slowly |
| MLP | CleanRL-style (rollout 128, 4 epochs) | 100k | 42 | |
| MLP | SB3 rl-zoo settings (lr 1e-3, 10 epochs, minibatch 8 envs, gamma 0.98, lambda 0.8, no entropy) | 100k | 223 | |
| MLP | **current defaults** (zoo settings + target_kl 0.02) | 100k | 174 | |
| Fly v1 | old defaults; rnn_steps 1, alpha 0.3, LayerNorm readout | 51k | 23 | never above random: observation signal lost at the readout |
| Fly v2 | old defaults; rnn_steps 4, alpha 0.7, calibrated readout, obs normalisation | 102k | 9 (peak 48) | first run above random, then KL spike (0.16) and collapse under uniform lr 2.5e-4 |
| Fly v3 | zoo defaults, uniform lr 1e-3 | 5k | 11 | 1,314-input linear head saturated within 10 updates |
| Fly v4 | zoo defaults, head lr 5e-5 | 56k | 63 | stable, slow rise |
| Fly v5 | + resting-state start, readout regressed onto observations (4-d), linear critic | 102k | 21 | random level; a linear policy with a linear critic fails on the raw observation too (72 then 19) |
| **Fly v6** | **v5 + small MLP critic (policy still linear on the readout)** | **102k** | **36 (peak 228 at 97k)** | learns to MLP level and beyond but oscillates: 131, 151, 180, 223, 168, 147, 228, 36 over updates 70-200 |
| Fly v6, brain frozen | as v6, connectome parameters fixed | 102k | 101 (peak 133) | learns, oscillates more (drops to 21 twice) |

What the CartPole rows established: the training loop is sound (MLP learns); the connectome
transmits the full state to the descending neurons (a behaviour-cloned linear head on the
frozen, untrained network scores 500/500); and five interface and training defects, found with
linear probes on the frozen network, stood between that and reinforcement learning: a readout
dominated by the resting pattern, fast observation components filtered by one step per frame,
a 10x transient at every reset, a random readout projection half made of drift, and a linear
critic. With those fixed the fly learns CartPole to the MLP's level and beyond, though not yet
stably: only about one seed in three takes off.

## Pong (ALE, max +21, random about -20.7, human about 14.6)

Same machine (one shared RTX 5090, 8 env runners or 16 in-process envs) for every row. Fly runs
use the `visual` sub-network (138,743 neurons, 8.4M edges).

| Model | Trainer | Config | Env steps | Return | Entropy at end | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| **CNN (RLlib tuned Atari PPO)** | RLlib PPO | `scripts/baseline_cnn_pong.py`: 4-frame stack, 4-conv CNN, batch 4000, 10 epochs, lr 1.5e-4 | **356k** | **+19.0** | | reached the stop criterion (>= 18) in 25 minutes; -19 at 280k, +6 at 320k, +14.5 at 352k |
| Fly | simple A2C | rnn_steps 2, 8 envs, rollout 16, lr 3e-4, entropy 0.01 | 321k | -20.5 | 0.64 | policy collapsed to two actions within 100k steps |
| Fly | simple PPO | rnn_steps 2, 8 envs, rollout 32, 3 epochs, entropy 0.01 | 289k | -19.8 | 0.55 | slower collapse, no score gain |
| Fly | RLlib PPO | rnn_steps 2, 8 runners x 2 envs, batch 4096, minibatch 256 | 41k | -20.6 | 1.66 | stopped by a checkpoint-path bug (fixed) |
| Fly | RLlib APPO | rnn_steps 1, 8 runners x 2 envs, batch 4096, minibatch 256, entropy 0.01 | 755k | -20.3 | 1.71 | pre-fix model; entropy and return flat throughout; 160 env steps / s |
| Fly v4 / v5 | RLlib APPO | fixed model, official APPO recipe; v5 with per-group lr | 25k / 5k | -21 | 0.0 | collapsed to a deterministic policy |
| Fly v7 | simple PPO | temporal-contrast retina, 26-d readout subspace, MLP critic, 16 envs, gamma 0.99, lambda 0.95, clip 0.1, entropy 0.01, 4 epochs | 435k | -21.0 | 1.50 | no collapse, no learning |
| Fly v8a | simple PPO | as v7 but no readout bottleneck: linear head on all 1,314 descending neurons | 717k | -20.65 | 1.53 | no collapse, no learning at twice the CNN's solving budget |
| Fly v8c (control) | simple PPO | as v8a with a 64-unit tanh MLP policy head (diagnostic, not the model's claim) | 660k | -20.35 | 1.68 | no learning either |
| Fly v9 | simple PPO | as v8a with the swept retina: full-field sampling, surround 4, temporal gain 8 (ball y R^2 0.86 at the descending neurons) | 916k | -20.55 | 1.16 | no learning at 2.5x the CNN's solving budget |
| Fly, frozen + behaviour cloning | supervised (`scripts/bc_pong.py`) | untrained v9 network, linear head on the 1,314 descending neurons cloned from the CNN teacher on 6,000 steps | 6k | -9.7 | | episodes 12, -20, -21; teacher scored 6.0 (19, 16, -17) in the same env |
| Fly, frozen + behaviour cloning | supervised (`scripts/bc_pong.py`) | untrained v9 network, 64-unit tanh MLP head cloned the same way | 6k | **+8.0** | | episodes 19, 20, -15: the frozen connectome's readout supports Pong at the CNN's level; RL has not found the head |
| Fly, frozen + DAgger | supervised (`scripts/dagger_pong.py`) | frame-filling eye, 64-unit MLP head; 8 rounds x 12,000 steps, CNN teacher labels the student's own states | 96k labelled | **+13.4** | | greedy, 5 fixed starts: 18, 20, -9, 18, 20 in rounds 4-7; the same start is lost every round |
| Fly, from the cloned head + PPO (frame-filling eye) | simple PPO | v9 settings, brain lr 1e-7, MLP head cloned from the CNN (sampled start -13.7) | 307k | -16.5 | 0.97 | flat around the start for 600 updates |
| Fly, from the cloned head + PPO (frame-filling eye) | simple PPO | as above with the linear head (sampled start -18.7) | 307k | -21.0 | 0.48 | erodes below its own start |
| MLP (600, 64), 8.5M parameters | simple PPO | flattened 2 x 84 x 84 frame, lr 2.5e-4, otherwise the v9 settings | 330k | -20.4 | 1.64 | no learning: on raw pixels the MLP has no better prior than the fly |

On Pong the same sufficiency test as on CartPole passes (a head cloned from the CNN plays +19
on the frozen network, and DAgger on the frozen network reaches +13.4, close to the human
reference of 14.6) while every RL run stays at or below its starting point, so the open items
are CartPole take-off stability and credit assignment on Pong, not the model's representation.

To add a row, run one of the training commands in the README, read the last log line (simple
trainers) or the last `{"iter": ...}` line (RLlib), and record the config, env steps, return
and entropy.

## Compute and energy against an equal-parameter MLP

`scripts/benchmark_cost.py`, Pong observation (2 x 84 x 84), batch 16 envs, unroll 32 steps
(the trainer's segment), server CPU while three GPU jobs were also stepping envs:

| Model | Trainable parameters | Multiply-adds per frame | Rollout ms / env step | Replay (fwd + bwd) ms / env step |
| --- | --- | --- | --- | --- |
| fly, `visual` sub-network | 8,847,598 | 33.6M | 147.6 | 526.7 |
| MLP (600, 64) on the flattened frame | 8,506,719 | 8.5M | 4.2 | 9.2 |

Same parameter count, 4x the arithmetic, 35 to 57x the time: the fly's cost is a sparse
gather and scatter over 8.4M edges four times per frame, which is bound by memory bandwidth,
while the MLP is two dense matrix products. In training on the GPU the same gap shows as 44
versus 148 env steps per second. GPU joules per env step are measured by the same script
when the card has room (it needs about 2 GB free).

The equal-parameter MLP trained through the same simple PPO (lr 2.5e-4, otherwise the v9
settings) stays at -20.4 after 330k steps with entropy 1.5: on raw pixels an MLP has no
better inductive bias than the fly, and the CNN's convolutions are what solve Pong quickly.

## Throughput

| Scenario | CPU (M-series laptop) | RTX 5090 |
| --- | --- | --- |
| whole CNS playing Pong (inference) | 30 ms / step | 8 ms / step |
| `visual` subset, simple PPO, 8 envs | 41 steps / s | 220 steps / s |
| `visual` subset, 32-step unroll fwd+bwd, batch 8 | | 0.9 s (about 280 steps / s) |

Training throughput is bound by the sparse product over millions of edges, not by the RL
framework. `--rnn-steps 1`, the `visual_small` subset and asynchronous APPO are the three knobs
that buy the most speed. RLlib stores the recurrent state (one float per neuron) at every time
step of every episode, so keep `--max-seq-len` moderate.
