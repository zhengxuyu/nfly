"""Build a NeuroMechFly with legs, adhesion, eye cameras and a tracking camera on flat ground,
with flygym's hybrid turning controller driving the legs from a two-sided descending signal."""

from __future__ import annotations

import dataclasses

import mujoco as mj
import numpy as np

TARGET_GEOM = "nfly_target"


@dataclasses.dataclass
class FlyWorld:
    sim: object                    # flygym.Simulation
    fly: object                    # flygym.compose.NeuroMechFly
    controller: object             # flygym_demo HybridTurningController
    dof_order: list
    thorax_index: int
    track_camera_id: int
    eye_camera_ids: list[int]
    target_geom_id: int            # -1 when the world has no target

    def control_step(self, drive: np.ndarray) -> None:
        """One physics step under the CPG controller modulated by the descending drive."""
        from flygym_demo.complex_terrain import HybridControllerObservation, apply_locomotion_action
        obs = HybridControllerObservation.from_sim(self.sim, self.fly.name)
        apply_locomotion_action(self.sim, self.fly.name, self.controller.step(drive, obs))
        self.sim.step()

    def thorax_position(self) -> np.ndarray:
        return self.sim.get_body_positions(self.fly.name)[self.thorax_index].copy()

    def heading(self) -> np.ndarray:
        body_id = self.sim._internal_bodyids_by_fly[self.fly.name][self.thorax_index]
        return self.sim.mj_data.xmat[body_id].reshape(3, 3)[:, 0].copy()

    def fell_over(self) -> bool:
        body_id = self.sim._internal_bodyids_by_fly[self.fly.name][self.thorax_index]
        up = self.sim.mj_data.xmat[body_id].reshape(3, 3)[:, 2]
        return bool(up[2] < 0.2 or self.thorax_position()[2] < 0.3)

    def set_target(self, pos: np.ndarray) -> None:
        self.sim.mj_model.geom_pos[self.target_geom_id] = pos

    def target_position(self) -> np.ndarray:
        return self.sim.mj_model.geom_pos[self.target_geom_id].copy()

    def distance_to_target(self) -> float:
        d = self.target_position()[:2] - self.thorax_position()[:2]
        return float(np.linalg.norm(d))

    def target_vector_in_body_frame(self) -> np.ndarray:
        body_id = self.sim._internal_bodyids_by_fly[self.fly.name][self.thorax_index]
        rot = self.sim.mj_data.xmat[body_id].reshape(3, 3)
        return rot.T @ (self.target_position() - self.thorax_position())


def build_world(timestep: float = 1e-4, with_target: bool = True) -> FlyWorld:
    from flygym import Simulation
    from flygym.compose import FlatGroundWorld
    from flygym.utils.math import Rotation3D
    from flygym_demo.complex_terrain import HybridTurningController, get_default_locomotion_dof_order, make_locomotion_fly

    fly = make_locomotion_fly(colorize=True)
    fly.add_vision()
    fly.add_tracking_camera()
    world = FlatGroundWorld()
    if with_target:
        world.mjcf_root.worldbody.add_geom(type=mj.mjtGeom.mjGEOM_SPHERE, name=TARGET_GEOM, pos=(8.0, 0.0, 0.6),
                                           size=(0.6, 0.0, 0.0), rgba=(1.0, 0.0, 0.0, 1.0), contype=0, conaffinity=0)
    world.add_fly(fly, (0.0, 0.0, 0.5), Rotation3D("quat", (1, 0, 0, 0)))
    sim = Simulation(world, timestep=timestep)
    dof_order = get_default_locomotion_dof_order()
    controller = HybridTurningController(timestep=timestep, output_dof_order=dof_order)
    order = fly.get_bodysegs_order()
    thorax = next(i for i, b in enumerate(order) if b.link == "thorax")
    names = [mj.mj_id2name(sim.mj_model, mj.mjtObj.mjOBJ_CAMERA, i) for i in range(sim.mj_model.ncam)]
    track = next(i for i, n in enumerate(names) if n.endswith("trackcam"))
    eyes = [i for i, n in enumerate(names) if n.endswith("l_eye_cam_camera")] + [i for i, n in enumerate(names) if n.endswith("r_eye_cam_camera")]
    target = mj.mj_name2id(sim.mj_model, mj.mjtObj.mjOBJ_GEOM, TARGET_GEOM) if with_target else -1
    return FlyWorld(sim, fly, controller, dof_order, thorax, track, eyes, target)
