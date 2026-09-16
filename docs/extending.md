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
