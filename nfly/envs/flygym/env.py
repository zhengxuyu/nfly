"""A Gymnasium env around NeuroMechFly (flygym 2): the MaleCNS brain steers the fly's own body.

The body walks with flygym's hybrid CPG controller; the agent supplies the two descending
drives (left, right, each in [-1, 1]) that the controller turns into leg joint angles, the
same interface flygym's own turning examples use. Observations are the fly's two eye cameras
(the body's compound eyes, rendered by MuJoCo), grayscale, side by side, plus the change from
the previous step: the Atari suite's format, so the same retina that plays Pong sees through
the body's eyes. `observation="state"` gives joint angles, heading and the target vector
instead.

Tasks:
    walk      reward = forward progress of the thorax along its initial heading (mm per step)
    approach  a red ball is placed in front of the fly at a random bearing; reward = progress
              towards it, +10 on arrival (within 2 mm), episode ends
"""

from __future__ import annotations

import gymnasium as gym
import mujoco as mj
import numpy as np

from .world import FlyWorld, build_world

TASKS = ("walk", "approach")
ARRIVAL_MM = 2.0
ARRIVAL_BONUS = 10.0


class FlyWalkEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(self, task: str = "approach", observation: str = "eyes", frame_size: int = 84,
                 physics_per_step: int = 100, max_steps: int = 400, render_mode: str | None = None,
                 timestep: float = 1e-4):
        if task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}")
        if observation not in ("eyes", "state"):
            raise ValueError("observation must be 'eyes' or 'state'")
        self.task, self.observation_kind, self.frame_size = task, observation, frame_size
        self.physics_per_step, self.max_steps, self.render_mode = physics_per_step, max_steps, render_mode
        self.world: FlyWorld = build_world(timestep=timestep, with_target=task == "approach")
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        if observation == "eyes":
            self.observation_space = gym.spaces.Box(0.0, 1.0, (frame_size, frame_size), np.float32)
            self._eye_renderer = mj.Renderer(self.world.sim.mj_model, frame_size, frame_size // 2)
            self._eye_option = mj.MjvOption()
            self._eye_option.geomgroup[1] = 0; self._eye_option.geomgroup[2] = 0   # flygym hides the fly's own head parts from its eyes
        else:
            n = len(self.world.dof_order)
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, (2 * n + 3 + 3,), np.float32)
        self._track_renderer = None
        self._rng = np.random.default_rng()
        self._t = 0
        self._last_progress_ref = 0.0

    # ---- Gymnasium API ------------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        w = self.world
        w.sim.reset(); w.controller.reset(seed=seed); w.sim.warmup(0.05)
        self._t = 0
        self._heading0 = w.heading()
        if self.task == "approach":
            bearing = self._rng.uniform(-np.pi / 3, np.pi / 3)
            dist = self._rng.uniform(6.0, 12.0)
            pos = w.thorax_position()
            h = self._heading0
            side = np.array([-h[1], h[0], 0.0])
            w.set_target(pos + dist * (np.cos(bearing) * h + np.sin(bearing) * side))
        self._last_progress_ref = self._progress_metric()
        return self._observe(), {}

    def step(self, action):
        w = self.world
        drive = np.clip(np.asarray(action, dtype=float).reshape(2), -1.0, 1.0)
        for _ in range(self.physics_per_step):
            w.control_step(drive)
        self._t += 1
        metric = self._progress_metric()
        reward = float(metric - self._last_progress_ref)
        self._last_progress_ref = metric
        terminated = False
        if self.task == "approach" and w.distance_to_target() < ARRIVAL_MM:
            reward += ARRIVAL_BONUS; terminated = True
        if w.fell_over():
            terminated = True
        truncated = self._t >= self.max_steps
        return self._observe(), reward, terminated, truncated, {"distance": w.distance_to_target() if self.task == "approach" else 0.0}

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._track_renderer is None:
            self._track_renderer = mj.Renderer(self.world.sim.mj_model, 240, 320)
        self._track_renderer.update_scene(self.world.sim.mj_data, camera=self.world.track_camera_id)
        return self._track_renderer.render()

    def close(self):
        for r in (getattr(self, "_eye_renderer", None), self._track_renderer):
            if r is not None:
                r.close()

    # ---- helpers ------------------------------------------------------------------------------
    def _progress_metric(self) -> float:
        w = self.world
        if self.task == "approach":
            return -w.distance_to_target()
        return float(np.dot(w.thorax_position()[:2], self._heading0[:2]))

    def _observe(self):
        w = self.world
        if self.observation_kind == "state":
            angles, vel = w.sim.get_joint_angles(w.fly.name), w.sim.get_joint_velocities(w.fly.name)
            target = w.target_vector_in_body_frame() if self.task == "approach" else np.zeros(3)
            return np.concatenate([angles, vel, w.heading(), target]).astype(np.float32)
        halves = []
        for cam_id in w.eye_camera_ids:
            self._eye_renderer.update_scene(w.sim.mj_data, camera=cam_id, scene_option=self._eye_option)
            rgb = self._eye_renderer.render()
            halves.append(rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32))
        frame = np.concatenate(halves, axis=1) / 255.0                # (size, size): left eye | right eye
        return frame.astype(np.float32)
