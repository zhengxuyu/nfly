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
    frame = torch.as_tensor(buffered.pooled())
    stack = native_stack(history, frame)
    assert torch.count_nonzero(stack[..., :3]) == 0
    torch.testing.assert_close(stack[..., 3], frame[None])
    previous = []
    for _ in range(2):
        image, *_ = buffered.step(0)
        previous.append(teacher_frame(image))
    np.testing.assert_array_equal(buffered.pooled(), np.maximum(*previous))
    class Module:
        def forward_inference(self, data):
            assert data["obs"].shape == (1, 64, 64, 4)
            return {"action_dist_inputs": torch.tensor([[0., 1.]])}
    teacher = SimpleNamespace(device="cpu", stack=[], module=Module())
    assert act_on_frame(teacher, buffered.pooled()) == 1
    buffered.reset()
    assert len(buffered.frames) == 1
    buffered.close()
    native.close()
