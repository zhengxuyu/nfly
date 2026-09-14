# Working on nfly

Rules for any agent or person editing this repository. They apply to every file: library code,
scripts, tests, docs, commit messages.

## Language

- Everything committed is in English: code, identifiers, comments, docstrings, README, commit
  messages, log messages. ASCII only in source files (no typographic dashes or quotes).
- Talk to the user in whatever language they use; write the artefacts in English.

## Architecture (keep the layers one-directional)

```
connectome  ->  brain  ->  interface  ->  agent  ->  suite  ->  rl
   data        model     senses/muscles   policy    gym games   training
```

- A layer imports only from layers to its left. `brain` never imports gym; `suite` never imports
  the agent; `rl.simple` has no dependency beyond torch and gymnasium; `rl.rllib` is the only
  place that imports ray.
- The agent meets a game only through `observation_space` and `action_space`. Never add
  game-specific branches to the brain, the agent, or the encoders/decoders. New senses, muscles,
  games and algorithms are new subclasses registered in their layer, not edits to existing ones.
- Data releases are loaders that return a `Connectome`; nothing downstream may know which release
  it came from. The MaleCNS v1.0 release is the only supported source; do not reintroduce others.

## Code smells to avoid (from refactoring.guru/refactoring/smells)

Check every change against this list before committing; fix the smell rather than documenting it.

**Bloaters**
- Long method: a function does one thing and fits on a screen (~40 lines). Extract helpers with
  names that say what, not how.
- Large class: a class with more than one reason to change gets split.
- Primitive obsession: related values travel together as a dataclass (`Rollout`, `RetinaLayout`,
  `StimulationResult`), not as tuples, dicts or parallel lists.
- Long parameter list: more than ~5 parameters means a config dataclass (`PPOConfig`) or the
  parameters belong to an object.
- Data clumps: the same group of arguments appearing in several signatures is one object.

**Object-orientation abusers**
- Switch statements on type: dispatch on `isinstance` or string tags lives in exactly one factory
  (`ObservationEncoder.for_space`, `ActionDecoder.for_space`, the suite registry), never spread
  through callers.
- Temporary field: no attribute that is `None` until some later method fills it. Build objects
  complete (`collect` builds `Rollout` once at the end).
- Refused bequest: subclass only when every inherited method makes sense for the subclass.
- Alternative classes with different interfaces: siblings (encoders, decoders, suites, trainers)
  share one interface and one signature shape.

**Change preventers**
- Divergent change: a new game, sense, or algorithm must not require touching more than its own
  layer.
- Shotgun surgery: a constant or rule exists once (`DEFAULT_SIGN`, `AFFERENT`, `EDGE_CHUNK`);
  scripts share argument groups through `nfly/cli.py`.
- Parallel inheritance hierarchies: adding a suite must not require adding a matching class
  elsewhere.

**Dispensables**
- Comments that narrate code: delete them and improve the name. Keep comments that say *why*
  (a biological convention, a numerical trick, a memory bound).
- Duplicate code: two similar blocks become one helper (`GameSuite.finish`, `rl.simple.common`).
- Lazy class, dead code, speculative generality: no parameters, flags, branches or classes that
  nothing uses today. Delete on sight; git remembers.
- Data class without behaviour: a dataclass may carry the methods that belong to its data
  (`Connectome.where`, `StimulationResult.report`).

**Couplers**
- Feature envy: a method that mostly reads another object's fields moves to that object.
- Inappropriate intimacy: use public methods; do not reach into another module's private
  attributes.
- Message chains: no `a.b().c().d()` across layers; ask the nearest object for what you need.
- Middle man: a class that only forwards calls is removed.

## Environment

- Environments and packages are managed with uv only: `uv sync --extra dev` creates `.venv`
  from `pyproject.toml` and `uv.lock`; run everything through `uv run ...` or the `.venv`
  interpreter. Do not `pip install` into system or conda Pythons, and do not hand-edit `.venv`.
- Adding a dependency means editing `pyproject.toml` (core or the right extra) and committing the
  regenerated `uv.lock`.

## Working practice

- Tests run without the 1 GB data release: use `write_synthetic` (MaleCNS format) for fixtures.
  Every new module gets a test that exercises it end to end at small scale.
- `uv run pytest -q` must pass before every commit. Run the relevant script once on real data when the
  change touches loading, the retina, or training.
- Keep the biology -> network mapping table in README.md in sync with the loader.
- Commit messages: imperative subject line, body says why. No scp of code to remote machines;
  always go through git so history is kept.
