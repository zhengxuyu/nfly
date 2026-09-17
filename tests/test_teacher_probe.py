"""Check teacher transfer preprocessing against the installed RLlib protocol."""

import gymnasium as gym
import numpy as np
import torch

from scripts.probe_teacher import TeacherFrameBuffer, native_stack, teacher_frame, act_on_frame


class RawFrames(gym.Env):
    observation_space = gym.spaces.Box(0, 255, (210, 160, 3), np.uint8)
    action_space = gym.spaces.Discrete(2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return np.zeros(self.observation_space.shape, np.uint8), {}

    def step(self, action):
        self.t += 1
        image = np.zeros(self.observation_space.shape, np.uint8)
        image[:, 20 * self.t:20 * self.t + 10] = 255
        return image, 0, self.t == 3, False, {}


def test_teacher_pooling_and_zero_padding_end_to_end():
    from types import SimpleNamespace
    from ray.rllib.env.wrappers.atari_wrappers import GrayScaleAndResize, NormalizedImageEnv
    raw, buffered = RawFrames(), TeacherFrameBuffer(RawFrames())
    native = NormalizedImageEnv(GrayScaleAndResize(raw, dim=64))
    a, _ = buffered.reset()
    b, _ = native.reset()
    np.testing.assert_array_equal(teacher_frame(a), b[..., 0])
    history = []
    frame = torch.as_tensor(buffered.teacher_frame())
    stack = native_stack(history, frame)
    assert torch.count_nonzero(stack[..., :3]) == 0
    torch.testing.assert_close(stack[..., 3], frame[None])
    previous = []
    for _ in range(2):
        image, *_ = buffered.step(0)
        previous.append(teacher_frame(image))
    np.testing.assert_array_equal(buffered.teacher_frame(), np.maximum(*previous))
    class Module:
        def forward_inference(self, data):
            assert data["obs"].shape == (1, 64, 64, 4)
            return {"action_dist_inputs": torch.tensor([[0., 1.]])}
    teacher = SimpleNamespace(device="cpu", stack=[], module=Module())
    assert act_on_frame(teacher, buffered.teacher_frame()) == 1
    torch.manual_seed(7)
    actions = [act_on_frame(teacher, buffered.teacher_frame(), 1.0) for _ in range(30)]
    assert set(actions) == {0, 1}
    torch.manual_seed(7)
    assert actions == [act_on_frame(teacher, buffered.teacher_frame(), 1.0) for _ in range(30)]
    buffered.reset()
    assert len(buffered.frames) == 1
    buffered.close()
    native.close()


def test_teacher_audit_marks_timeouts_as_incomplete(monkeypatch):
    from scripts import probe_teacher
    class Teacher:
        def reset(self):
            pass
        def __call__(self, env):
            return 0
    monkeypatch.setattr(probe_teacher, "make_env", lambda mode: (gym.wrappers.TimeLimit(RawFrames(), 1), None))
    rows = probe_teacher.evaluate(Teacher(), "legacy", [0, 1])
    assert len(rows) == 2
    assert all(r["truncated"] and not r["terminated"] and r["steps"] == 1 for r in rows)


def test_production_teacher_matches_native_history_and_requires_frame_buffer(monkeypatch):
    from nfly.rl.rllib.teacher import CNNTeacher, TeacherFrameBuffer
    from ray.rllib.core.rl_module.rl_module import RLModule
    seen = []
    class Module(torch.nn.Module):
        def forward_inference(self, data):
            seen.append(data["obs"].clone())
            return {"action_dist_inputs": torch.tensor([[0., 1.]])}
    monkeypatch.setattr(RLModule, "from_checkpoint", lambda path: Module())
    teacher = CNNTeacher("unused-checkpoint", "cpu")
    env = TeacherFrameBuffer(RawFrames())
    env.reset()
    assert teacher(env) == 1
    assert torch.count_nonzero(seen[-1][..., :3]) == 0
    env.step(0)
    teacher(env)
    assert torch.equal(seen[-1][..., 2], seen[-2][..., 3])
    import pytest
    raw = RawFrames()
    raw.reset()
    with pytest.raises(AttributeError):
        teacher(raw)
    teacher.reset()
    env.reset()
    teacher(env)
    assert torch.count_nonzero(seen[-1][..., :3]) == 0
    env.close()


def test_teacher_observer_preserves_fly_environment():
    from nfly.rl.rllib.teacher import TeacherFrameBuffer
    from nfly.suite import get_suite
    plain = get_suite("atari").make("pong")
    observed = get_suite("atari", raw_wrappers=(TeacherFrameBuffer,)).make("pong")
    try:
        a, _ = plain.reset(seed=123)
        b, _ = observed.reset(seed=123)
        np.testing.assert_array_equal(a, b)
        for action in [0, 1, 2, 3, 4, 5] * 3:
            a, ra, ta, ca, _ = plain.step(action)
            b, rb, tb, cb, _ = observed.step(action)
            np.testing.assert_array_equal(a, b)
            assert (ra, ta, ca) == (rb, tb, cb)
        assert observed.get_wrapper_attr("teacher_frame")().shape == (64, 64)
    finally:
        plain.close()
        observed.close()


def test_teacher_training_view_matches_inference_input():
    from nfly.rl.rllib.teacher import TeacherTrainingView, make_teacher_env
    env = TeacherTrainingView(make_teacher_env(seed=19))
    try:
        obs, _ = env.reset(seed=19)
        assert obs.shape == (64, 64, 1) and env.observation_space.contains(obs)
        np.testing.assert_array_equal(obs[..., 0], env.get_wrapper_attr("teacher_frame")())
        for action in [0, 2, 3, 0]:
            obs, *_ = env.step(action)
            np.testing.assert_array_equal(obs[..., 0], env.get_wrapper_attr("teacher_frame")())
    finally:
        env.close()
