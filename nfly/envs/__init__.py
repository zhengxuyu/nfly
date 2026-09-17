"""Embodied environments: each subpackage wraps one simulator as a GameSuite and registers it
on import (`get_suite` imports `nfly.envs.<name>` on demand). Optional extras; nothing here is
imported by the core package. Available: `miniworld` (first-person 3-D rooms), `mujoco`
(Gymnasium's MuJoCo bodies, state or camera observations)."""
