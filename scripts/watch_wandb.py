"""Backfill and follow a running simple PPO experiment without changing its process.

Run with an isolated SDK environment, for example:
uv run --no-project --with wandb==0.24.2 scripts/watch_wandb.py \
    --log runs/ppo.log --eval runs/ppo-eval.jsonl --pid 1234 \
    --project nfly --id pong-scratch-s0 --training-revision abc123
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys
import time


PROGRESS = re.compile(r"upd\s+(\d+)\s+(.*?)\s+episodes\s+(\d+)\s+mean return\(20\)\s+(\S+)\s+(\S+) steps/s$")
METRICS = {"pg": "policy_loss", "v": "value_loss", "ent": "entropy", "clipfrac": "clip_fraction",
           "kl": "kl", "ev": "explained_variance", "epochs": "epochs",
           "critic_only": "critic_only", "lr": "lr_multiplier"}


def complete_lines(path):
    if not path.exists():
        return []
    return [line.rstrip("\n") for line in path.read_text().splitlines(keepends=True) if line.endswith("\n")]


def run_config(path, revision):
    for line in complete_lines(path):
        if line.startswith("run arguments: "):
            config = json.loads(line.removeprefix("run arguments: "))
            config.update({k: json.loads(v) for k, v in (item.split("=", 1) for item in config.get("set", []))})
            return {**config, "training_revision": revision, "logging_source": "existing_stdout_and_eval_jsonl",
                    "training_metric_precision": "stdout rounded to 3 decimals; evaluations retain JSON precision"}
    raise ValueError("Training log has no complete run-arguments line")


def train_event(line, stride):
    match = PROGRESS.fullmatch(line.strip())
    if not match:
        return None
    update, metrics, episodes, reward, speed = match.groups()
    parts = metrics.split()
    values = {METRICS[k]: float(v) for k, v in zip(parts[::2], parts[1::2]) if k in METRICS}
    values.update(episodes=int(episodes), mean_return_20=float(reward), steps_per_second=float(speed),
                  update=int(update), env_steps=int(update) * stride)
    return int(update), "train", {f"train/{k}": v for k, v in values.items() if math.isfinite(v)}


def eval_event(row, stride):
    values = {"eval/update": row["update"], "eval/env_steps": row["update"] * stride}
    for mode, result in row["modes"].items():
        scores, ended = result["returns"], result["terminated"]
        metrics = {"mean_return": result["mean"], "min_return": min(scores), "max_return": max(scores),
                   "completed_episodes": sum(ended), "episodes": len(scores),
                   "win_rate": sum(score > 0 and done for score, done in zip(scores, ended)) / len(scores)}
        metrics.update({k: result[k] for k in ("value_mc_ev", "value_mc_mse", "entropy") if k in result})
        values.update({f"eval/{mode}/{k}": v for k, v in metrics.items() if v is not None and math.isfinite(v)})
    return row["update"], "eval", values


def read_events(log, evaluation, config):
    stride = config["envs"] * config["rollout"]
    events = [event for line in complete_lines(log) if (event := train_event(line, stride)) is not None]
    events.extend(eval_event(json.loads(line), stride) for line in complete_lines(evaluation) if line.strip())
    return sorted(events, key=lambda event: (event[0], event[1]))


def sync_events(run, events):
    for update, kind, values in events:
        cursor = f"logger/{kind}_update"
        if update <= run.summary.get(cursor, -1):
            continue
        run.log(values)
        run.summary[cursor] = update
        print(f"synced {kind} update {update}", flush=True)


def process_identity(pid):
    """The Linux start time prevents a reused PID from keeping the logger alive."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[19] if fields[0] != "Z" else None
    except FileNotFoundError:
        return None


def follow(run, args, config):
    identity = process_identity(args.pid)
    while True:
        sync_events(run, read_events(args.log, args.eval, config))
        if identity is None or process_identity(args.pid) != identity:
            # Drain after observing exit so the final flush cannot be missed.
            sync_events(run, read_events(args.log, args.eval, config))
            final_update = max(run.summary.get(f"logger/{kind}_update", -1) for kind in ("train", "eval"))
            complete = final_update >= config["updates"]
            run.summary["trainer_state"] = "completed" if complete else "stopped_early"
            return 0 if complete else 1
        run.summary["trainer_state"] = "running"
        time.sleep(args.interval)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--project", default="nfly")
    parser.add_argument("--entity")
    parser.add_argument("--id", required=True)
    parser.add_argument("--name", default="Pong / pure PPO / trainable brain / seed 0")
    parser.add_argument("--training-revision", required=True)
    parser.add_argument("--interval", type=float, default=30)
    parser.add_argument("--api-key-stdin", action="store_true", help="read authentication from stdin, never a command argument")
    args = parser.parse_args()
    if args.interval <= 0 or args.pid <= 0:
        parser.error("Need a positive interval and trainer PID")
    return args


def main():
    args = parse_args()
    settings = {"disable_git": True, "disable_code": True, "x_stats_pid": args.pid}
    if args.api_key_stdin:
        key = sys.stdin.readline().strip()
        if not key:
            raise ValueError("No W&B credential supplied on stdin")
        settings["api_key"] = key
    import wandb
    config = run_config(args.log, args.training_revision)
    run = wandb.init(project=args.project, entity=args.entity, id=args.id, name=args.name,
                     resume="allow", config=config, settings=wandb.Settings(**settings),
                     notes="Live mirror of the trainer logs. The logging process does not modify training. "
                           "Training metrics retain stdout precision; evaluation metrics retain JSON precision.")
    print(f"W&B run: {run.url}", flush=True)
    for kind in ("train", "eval"):
        run.define_metric(f"{kind}/env_steps")
        run.define_metric(f"{kind}/*", step_metric=f"{kind}/env_steps")
    run.summary["trainer_pid"] = args.pid
    code = 1
    try:
        code = follow(run, args, config)
    finally:
        run.finish(exit_code=code)


if __name__ == "__main__":
    main()
