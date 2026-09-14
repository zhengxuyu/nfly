"""Run a session step by step in a background thread and broadcast one event per step."""

from __future__ import annotations

import base64
import dataclasses
import logging
import queue
import threading
import time
import traceback
from typing import Any

import cv2
import numpy as np
import torch

from .session import Session

log = logging.getLogger(__name__)


@dataclasses.dataclass
class StepEvent:
    step: int
    episode: int
    action: int | list[float]
    action_name: str
    reward: float
    episode_return: float
    done: bool
    frame_jpeg_b64: str
    probs: list[float] | None       # action probabilities for discrete policies

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def encode_frame(rgb: np.ndarray, quality: int = 80) -> str:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""


class Broadcast:
    """Fan-out of events to any number of subscriber queues."""

    def __init__(self, maxsize: int = 64):
        self._subs: list[queue.Queue] = []
        self._lock = threading.Lock()
        self.maxsize = maxsize

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=self.maxsize)
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def publish(self, event: dict) -> None:
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            if q.full():                       # drop the oldest frame for slow clients
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
            q.put_nowait(event)


class EpisodeStreamer(threading.Thread):
    """Plays episodes forever (until stop()), honouring pause / single-step / reset commands."""

    def __init__(self, session: Session, broadcast: Broadcast, history: int = 512):
        super().__init__(daemon=True, name="nfly-streamer")
        self.session, self.broadcast = session, broadcast
        self.fps = session.config.fps
        self.history: list[dict] = []
        self.history_size = history
        self.step_no = self.episode = 0
        self.episode_return = 0.0
        self.error: str | None = None          # set if the loop died; shown by the page and tests
        self._paused = threading.Event()
        self._single_step = threading.Event()
        self._reset = threading.Event()
        self._stop = threading.Event()
        self._lock = threading.Lock()

    # ---- controls (called from the HTTP thread) -------------------------------------------------
    def pause(self) -> None: self._paused.set()
    def resume(self) -> None: self._paused.clear()
    def single_step(self) -> None: self._single_step.set()
    def reset(self) -> None: self._reset.set()
    def stop(self) -> None: self._stop.set(); self._paused.clear()

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    def state(self) -> dict[str, Any]:
        cfg = self.session.config
        with self._lock:
            return {"suite": cfg.suite, "game": cfg.game, "policy": cfg.policy, "subset": cfg.subset,
                    "checkpoint": cfg.checkpoint, "action_names": self.session.action_names, "fps": self.fps,
                    "step": self.step_no, "episode": self.episode, "episode_return": self.episode_return,
                    "paused": self.paused, "error": self.error, "history": list(self.history)}

    # ---- main loop -------------------------------------------------------------------------------
    def run(self) -> None:
        try:
            self._loop()
        except Exception:                      # a dead streamer must be visible, not silent
            self.error = traceback.format_exc()
            log.error("streamer stopped: %s", self.error)
            self.broadcast.publish({"error": self.error})

    def _loop(self) -> None:
        env, policy = self.session.env, self.session.policy
        obs, _ = env.reset(seed=self.session.config.seed)
        h = policy.initial_state(1)
        while not self._stop.is_set():
            if self._reset.is_set():
                self._reset.clear()
                obs, _ = env.reset(); h = policy.initial_state(1)
                self.episode += 1; self.episode_return = 0.0
            if self._paused.is_set() and not self._single_step.is_set():
                time.sleep(0.02); continue
            self._single_step.clear()
            t0 = time.time()
            obs, h, done = self._one_step(obs, h)
            if done:
                obs, _ = env.reset(); h = policy.initial_state(1)
                self.episode += 1; self.episode_return = 0.0
            time.sleep(max(0.0, 1.0 / self.fps - (time.time() - t0)))

    def _one_step(self, obs, h):
        env, policy = self.session.env, self.session.policy
        obs_t = torch.as_tensor(np.asarray(obs)).unsqueeze(0)
        probs = self._action_probs(obs_t, h)
        action, h = policy.act(obs_t, h, greedy=self.session.config.greedy)
        a = action[0]
        obs, r, term, trunc, _ = env.step(a)
        a_val = int(a) if np.ndim(a) == 0 else [float(x) for x in np.asarray(a).ravel()]
        name = self.session.action_names[a_val] if isinstance(a_val, int) else "continuous"
        frame = encode_frame(env.render())
        with self._lock:                       # counters and history change together
            self.step_no += 1; self.episode_return += float(r)
            ev = StepEvent(self.step_no, self.episode, a_val, name, float(r), self.episode_return,
                           bool(term or trunc), frame, probs).to_dict()
            self.history.append({k: ev[k] for k in ("step", "episode", "action", "action_name", "reward")})
            del self.history[:-self.history_size]
        self.broadcast.publish(ev)
        return obs, h, bool(term or trunc)

    def _action_probs(self, obs_t, h):
        policy = self.session.policy
        if not hasattr(policy, "forward"):
            return None
        with torch.no_grad():
            dist, _, _ = policy(obs_t, h)
        return dist.probs[0].tolist() if hasattr(dist, "probs") else None
