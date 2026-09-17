# Extending nfly

Everything is layered one way (`connectome -> brain -> interface -> agent -> suite -> rl / viz`):
the brain never sees a game, the suite never sees the brain, and a new task, sense, or
algorithm is a new subclass in its own layer.

## Your own game collection

```python
from nfly.suite import GameSuite, register

@register("mygames")
class MySuite(GameSuite):
    def games(self):
        return ["level1", "level2"]

    def make(self, game, seed=None, render_mode=None, **kw):
        return self.finish(MyEnv(game, render_mode=render_mode), seed)   # any gym.Env
```

`make_vector`, `spaces` and `finish` come from the base class; every script accepts
`--suite mygames --game level1`. `finish` seeds the env and, for vector Box observations, adds
observation normalisation.

## Embodied simulators (`nfly/envs/`)

Each subpackage of `nfly.envs` wraps one simulator as a `GameSuite` and registers it on import;
they are optional extras so the core package never imports them. `nfly.envs.miniworld`
(`uv sync --extra embodied`) exposes [Miniworld](https://github.com/Farama-Foundation/Miniworld)'s
first-person 3-D navigation tasks (the package still installs and runs, version 2.1.0, but
Farama deprecated the project on 2025-08-11 and its documentation site is offline) (hallway,
oneroom, tmaze, fourrooms, maze, collect, sidewalk, putnext) as 84 x 84 grayscale frames with
a change channel, the Atari suite's format, so the retina, readout and trainers apply
unchanged. `scripts/play.py`, `train_rl.py` and `serve.py` import it when `--suite miniworld`
is given.

`nfly.envs.mujoco` exposes Gymnasium's MuJoCo bodies (ant, halfcheetah, hopper, walker,
humanoid, swimmer, reacher, pusher, the inverted pendulums): state observations by default
(VectorEncoder and BoxDecoder), or `observation="pixels"` for a camera frame in the Atari
format.

`nfly.envs.flygym` puts the MaleCNS brain in NeuroMechFly's body (flygym 2, EPFL): the fly
walks with flygym's hybrid CPG controller and the agent supplies the two descending drives
(left, right in [-1, 1]) that steer it, the interface flygym's own turning examples use.
Observations are the body's two eye cameras rendered by MuJoCo, grayscale, side by side, plus
the change from the previous step, so the retina that plays Pong looks through the body's eyes
(`observation="state"` gives joint angles, velocities, heading and the target vector instead).
Tasks: `walk` (reward = forward progress of the thorax) and `approach` (a red ball at a random
bearing in front of the fly; reward = progress towards it, +10 on arrival). One env step is
100 physics steps of 0.1 ms (about 60 ms of compute on a laptop core); each env runs in its
own process because MuJoCo's offscreen GL context is thread-bound.

On a headless Linux server the suites switch pyglet and MuJoCo to EGL by themselves (no
`DISPLAY`); MuJoCo's EGL works with the system libraries alone (the error it prints when a
renderer is garbage-collected is harmless). pyglet, for Miniworld only, also needs `libGLU`
and the GLVND libraries; without root, download them into a user prefix and set the
library path only for Miniworld runs:

```bash
mkdir -p ~/.local/lib/glu && cd ~/.local/lib/glu
apt-get download libglu1-mesa libopengl0 libglvnd0 libglx0 libgl1 && for d in *.deb; do dpkg-deb -x "$d" .; done
ln -sf $PWD/usr/lib/x86_64-linux-gnu/libGLU.so.1 $PWD/usr/lib/x86_64-linux-gnu/libGLU.so
export LD_LIBRARY_PATH=$HOME/.local/lib/glu/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```

## Your own senses or muscles

Subclass `ObservationEncoder` (provide `idx`, the neurons you drive, and `encode`) or
`ActionDecoder` (provide `dist_inputs` and `distribution`) and pass them to
`FlyAgent.build(..., encoder=..., decoder=...)`. The defaults are chosen from the env's spaces
by `ObservationEncoder.for_space` and `ActionDecoder.for_space`: images go to the retina,
vectors to `VectorEncoder`; Discrete actions to a Categorical head, Box actions to a Gaussian
head.

## Your own algorithm

`nfly.rl.simple.common` has rollout collection, GAE and replay; a new trainer is a config
dataclass plus a loop (see `ppo.py`, about 80 lines). For RLlib, `FlyRLModule` already
implements the stateful `TorchRLModule` + `ValueFunctionAPI` + `TargetNetworkAPI` contract, so
any value-based RLlib algorithm works through `build_config(algo, ...)`; the learner classes in
`nfly.rl.rllib.learner` apply the per-group learning rates.

## Starting from a supervised head

`scripts/bc_pong.py --out runs/head.pt` behaviour-clones a policy head from a teacher (the RAM
heuristic or a trained CNN checkpoint) onto the frozen readout and saves the agent;
`scripts/train_rl.py --init runs/head.pt` loads that state after calibration so RL continues
from a head that already plays.

## Programmatic use

```python
from nfly import load_malecns, select_subset, FlyAgent
from nfly.suite import get_suite, play_episode
from nfly.viz import SessionConfig, serve

conn = select_subset(load_malecns("data"), "visual")            # all | brain | visual | visual_small
env = get_suite("atari").make("breakout", seed=0)
agent = FlyAgent.build(conn, env.observation_space, env.action_space)
print(play_episode(agent, env).ret)

server = serve(SessionConfig(suite="atari", game="breakout", checkpoint="runs/ppo.pt"), port=8000)
```

## Shared argument groups

Scripts share their argument groups through `nfly/cli.py` (`add_connectome_args`,
`add_agent_args`, `agent_kwargs`, `calibrate_on`, `apply_freezes`), so a new script gets the
same `--subset`, `--rnn-steps`, `--readout-dim`, `--head-hidden`, `--freeze-brain` and
`--device` flags by calling those.

## Viewer

`scripts/serve.py` (or `nfly.viz.serve(SessionConfig(...))`) starts a standard-library HTTP
server: `GET /` is the page, `GET /api/state` the session and recent history, `GET /api/stream`
a text/event-stream of step events, `POST /api/control` pause / resume / step / reset / fps.
With a fly policy the session also builds a `BrainAtlas` (`GET /api/anatomy`): up to
`--max-points` neurons at their MaleCNS soma positions (micrometres; neurons without a soma,
such as photoreceptors, are placed at the synapse-weighted mean of their partners), their
processing stage, and the photoreceptor sampling positions. Each step event then carries
`brain.cloud` (one byte per drawn neuron: activity as a z-score against a random-policy
rollout, 128 = resting), `brain.stages` (mean |z| per stage after every network sub-step,
which the page animates as the signal travels from the eye to the descending neurons) and
`brain.retina` (input current per photoreceptor). `--no-anatomy` turns this off.

`scripts/fetch_anatomy.py` downloads the release's own 3-D anatomy into `<data>/anatomy`: the
neuropil meshes of the brain and ventral nerve cord (neuroglancer legacy meshes, decimated by
vertex clustering to a few thousand faces each, stored in micrometres in `neuropils.json`) and
the SWC skeletons of every descending neuron (`--superclass`, `--per-stage N` and `--bodies`
choose more). When present, the session serves them at `/api/anatomy/meshes` and
`/api/anatomy/skeletons` (skeletons pruned to about 400 nodes, keeping root, tips and branch
points), and the page draws the shells translucent and the skeletons as lines coloured by the
neuron's activity, hiding the soma points of neurons that have a skeleton.

On the page, every stage in the legend is a toggle for its somata and skeletons; clicking a
neuron shows its MaleCNS body id, cell type and stage with a link to neuPrint and a trace of
its z-score over the last 256 steps (the atlas carries `body`, `type_index` and `types`);
the "most responsive cell types" table lists the cell types with the highest mean |z| among
the drawn neurons for the current frame (`brain.top_types`). `--compare <checkpoint>` (or
`--compare untrained`) plays a second agent with the same build in its own copy of the env,
one step per step, and streams its frame, action, probabilities and return as `compare` on
every event, shown under the eyes.
