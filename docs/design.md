# Design notes

How the connectome becomes a network, and what the parts of the agent are. The one-screen
version is the "Biology -> network" and "Anatomy of the agent" sections of the README; this
file holds the details.

## Dynamics

```text
h[t+1] = (1 - alpha) * h[t] + alpha * min( ReLU( W h[t] + b + u[t] ), h_max )
W[post, pre] = sign(pre) * (syn / sum_in syn) * exp(g_edge)        g_edge learnable, initialised to 0
```

- `alpha_i` (leak, from the membrane time constant) and `b_i` (threshold / resting bias) are one
  learnable scalar per neuron; `g_edge` is one learnable gain per edge. Wiring, signs and synapse
  counts are fixed.
- The resting bias b starts at 0.1: photoreceptors release only histamine (inhibitory), so with
  all neurons at 0 no inhibitory input could ever be read by a ReLU unit. With a little tonic
  activity, light shows up as a decrease downstream, matching real L1/L2 responses. At global
  gain 1.0 the whole CNS keeps a stable baseline; gain 5 diverges.
- The network runs `rnn_steps` (default 4) update steps per environment frame with alpha 0.7;
  one step per frame filtered out the fast observation components (ablation, section 2).
- Episodes start from the calibrated resting state `h_rest`, not from zeros; starting from zeros
  gave a 10x transient at every reset.

The sparse product is a gather + `index_add` with a custom backward, processed in edge chunks
(`EDGE_CHUNK`), so backprop through time stores only the (batch x N) state per step and never a
(batch x edges) tensor. Derived edge weights are computed once per unroll and cached during
inference.

## Sub-networks

| Name | Content | Neurons | Edges (>= 3 syn) |
| --- | --- | --- | --- |
| all | whole central nervous system | 166,700 | 10.5M |
| brain | without the ventral nerve cord | ~146,000 | |
| visual | optic lobes + visual projection + central brain + descending neurons | 138,743 | 8.4M |
| visual_small | optic lobes + visual projection + descending neurons | 106,579 | 5.2M |

`visual` is the default for image tasks; `visual_small` is the fastest one that still contains
the descending neurons.

## The five parts of the agent

| Part | What it does | Parameters | Origin |
| --- | --- | --- | --- |
| 1. Encoder (retina) | frame -> input current of 5,494 photoreceptors: hex photoreceptor layout, centre-surround, temporal contrast | none learnable except one global input gain | our design; geometry from the MaleCNS column coordinates |
| 2. Brain (`ConnectomeRNN`) | 138,743 neurons, 8.4M edges (visual sub-network), 4 network steps per frame | wiring, signs and synapse counts fixed; one learnable gain per edge (8.4M), one bias and one time constant per neuron | MaleCNS v1.0 |
| 3. Readout normalisation | activity of the 1,314 descending neurons, mean-centred, scaled and clipped per neuron | calibrated mean and scale per neuron (fixed) plus a learnable shift and gain in standardised units | our design |
| 4. Policy head | normalised descending activity -> action logits | linear: 1,314 x 6 (about 8k); `--head-hidden 64` tanh MLP: about 85k | our design |
| 5. Value head (critic) | descending activity -> state value, used by PPO during training only | 64-unit tanh MLP, about 85k | our design |

### Encoder

`RetinaEncoder` places one photoreceptor per optic-lobe column at the synapse-weighted hex
coordinate of its targets (`assignedOlHex1/2`), samples the frame there, and drives those
neurons with

```text
u = gain * ( sample(frame) + surround * sample(frame - local_mean) + temporal_gain * sample(frame - previous) )
```

with surround 4 over a 9x9 neighbourhood and temporal gain 8 by default. Both eyes sample the
whole frame (`split=True` gives each eye its half). Centre-surround and temporal contrast are
computations the fly's lamina performs; they live in the encoder because the probe sweep
(ablation, section 11) showed each adds about 0.1 R^2 of ball position at the descending
neurons. Nothing in the encoder is learnable except the global input gain, so it cannot turn
into a convolutional front end; making the three gains and a per-photoreceptor gain learnable
(photoreceptor adaptation) is on the roadmap.

`VectorEncoder` handles Box observations: each component is scaled to [0, 1] with the space's
bounds (or a running estimate for unbounded components) and drives a fixed random subset of
sensory neurons.

### Readout and heads

`ReadoutNorm` subtracts a per-neuron mean and divides by a per-neuron scale, both set by
`FlyAgent.calibrate_on_env` from a random-policy rollout and then fixed; training moves a
per-neuron shift and gain expressed in standard deviations, then the result is clipped at 10.
Without the normalisation the observation signal is hidden under the resting pattern
(ablation, section 2); with the raw mean as a trainable parameter, one optimizer step
saturated the features (section 13). The optional
linear bottleneck (`readout_dim`) is calibrated by regressing readout activity onto the
observation (vectors) or onto coarse 8x8 frame and change maps (images); for images the
default is now no bottleneck.

The linear policy head is the model's claim: the descending neurons decide the action through
one weighted vote, so a trained agent's competence is the connectome's, and each weight says
which descending neuron drives which action. The MLP head (`--head-hidden`) is a diagnostic
control that stands in for the ventral nerve cord circuits between descending and motor
neurons; when it learns where the linear head does not, the information is in the descending
neurons but not in linear form. The critic is an MLP as well, but it exists only for training
and is not part of the playing agent (a linear critic could not learn CartPole, section 5).

### Baselines

`MLPReference` (CartPole) and `scripts/baseline_cnn_pong.py` (Pong) have no brain at all. They
check that the training pipeline learns on a conventional model under the same conditions, and
set the bar the fly is measured against.

## What this is and is not

nfly reproduces the fly's **wiring**: which neurons exist, who connects to whom with how many
synapses, and the predicted sign of each connection. Everything else is a deliberate
simplification: one scalar rate per neuron, no spikes, ion channels, gap junctions, glia,
neuromodulation or plasticity rules; game pixels reach the photoreceptors through an assumed
hex-to-image mapping; the descending-neuron-to-action readout is learned; and training changes
edge gains, biases and time constants, so a trained model is connectome-*constrained*, not a
copy of the animal.
