"""Check backfill and live ingestion without a W&B account or a running trainer."""

import json

from scripts.watch_wandb import read_events, sync_events, run_config


ARGS = {"envs": 4, "rollout": 128, "updates": 2000, "init": None,
        "heads_only": False, "freeze_brain": False, "set": ["gamma=0.99"]}
TRAIN = "upd    10 pg -0.002 v 0.091 ent 1.752 clipfrac 0.066 kl 0.004 ev 0.005 epochs 4.000 critic_only 0.000 lr 1.000 episodes 4 mean return(20) -20.75 14.2 steps/s\n"
EVAL = {"update": 0, "seed": 19000, "modes": {"greedy": {
    "mean": -21., "returns": [-21., -20.], "steps": [761, 800],
    "terminated": [True, False], "value_mc_ev": None, "entropy": 1.79}}}


class Run:
    def __init__(self):
        self.summary = {}
        self.rows = []

    def log(self, metrics):
        self.rows.append(metrics)


def test_backfill_and_resume_with_late_evaluation(tmp_path):
    log, evaluation = tmp_path / "run.log", tmp_path / "eval.jsonl"
    log.write_text("run arguments: " + json.dumps(ARGS) + "\n" + TRAIN)
    evaluation.write_text(json.dumps(EVAL) + "\n")
    config = run_config(log, "e39d5ec")
    assert config["init"] is None and config["gamma"] == .99
    assert config["training_revision"] == "e39d5ec"
    run = Run()
    sync_events(run, read_events(log, evaluation, config))
    assert len(run.rows) == 2
    assert run.rows[0]["eval/env_steps"] == 0
    assert run.rows[0]["eval/greedy/completed_episodes"] == 1
    assert "eval/greedy/value_mc_ev" not in run.rows[0]
    assert run.rows[1]["train/env_steps"] == 5120
    assert run.rows[1]["train/mean_return_20"] == -20.75
    sync_events(run, read_events(log, evaluation, config))
    assert len(run.rows) == 2
    evaluation.write_text(evaluation.read_text() + json.dumps({**EVAL, "update": 5}) + "\n")
    sync_events(run, read_events(log, evaluation, config))
    assert len(run.rows) == 3 and run.rows[-1]["eval/env_steps"] == 2560


def test_incomplete_lines_are_retried_and_nonfinite_metrics_omitted(tmp_path):
    log, evaluation = tmp_path / "run.log", tmp_path / "eval.jsonl"
    log.write_text("run arguments: " + json.dumps(ARGS) + "\n" + TRAIN.replace("-20.75", "nan").rstrip())
    evaluation.write_text(json.dumps(EVAL))
    assert read_events(log, evaluation, ARGS) == []
    log.write_text(log.read_text() + "\n")
    evaluation.write_text(evaluation.read_text() + "\n")
    run = Run()
    sync_events(run, read_events(log, evaluation, ARGS))
    assert len(run.rows) == 2 and "train/mean_return_20" not in run.rows[-1]


def test_malformed_complete_evaluation_is_not_silently_dropped(tmp_path):
    import pytest
    log, evaluation = tmp_path / "run.log", tmp_path / "eval.jsonl"
    log.write_text("")
    evaluation.write_text("broken json\n")
    with pytest.raises(json.JSONDecodeError):
        read_events(log, evaluation, ARGS)


def test_trainer_exit_does_not_mark_partial_training_complete(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from scripts import watch_wandb
    log, evaluation = tmp_path / "run.log", tmp_path / "eval.jsonl"
    log.write_text(TRAIN)
    identity = iter(["original", "reused-pid"])
    monkeypatch.setattr(watch_wandb, "process_identity", lambda pid: next(identity))
    run = Run()
    args = SimpleNamespace(log=log, eval=evaluation, pid=1234, interval=30)
    assert watch_wandb.follow(run, args, ARGS) == 1
    assert run.summary["trainer_state"] == "stopped_early"
    assert len(run.rows) == 1
