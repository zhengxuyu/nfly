"""The signal probe uses identical frames and held-out calibration without fitting heads."""

import gymnasium as gym
import torch

from nfly import FlyAgent
from scripts.probe_signal import random_bank, trace_readout, compare_heads
from test_agent import visual_connectome
from test_simple_controls import ShortEpisode


def test_signal_probe_resets_and_preserves_weights():
    torch.manual_seed(0)
    env = gym.vector.SyncVectorEnv([lambda: ShortEpisode(4) for _ in range(2)], autoreset_mode="SameStep")
    agent = FlyAgent.build(visual_connectome(), env.single_observation_space, env.single_action_space,
                           readout_dim=0, head_hidden=8)
    bank = random_bank(env, [31, 32], 12, burn_in=1)
    env.close()
    assert bank["valid"][:, 0].tolist() == [False, True, True, True] * 3
    initial = {k: v.clone() for k, v in agent.state_dict().items()}
    trace = trace_readout(agent, bank)
    # Each first frame starts from the saved resting state, even after autoreset.
    with torch.no_grad():
        for t in (0, 4, 8):
            _, h = agent.step(bank["obs"][t], agent.initial_state(2), agent.weights())
            torch.testing.assert_close(trace[t], h[:, agent.decoder.idx])
        agent.brain.bias.add_(0.001)
    trained = {k: v.clone() for k, v in agent.state_dict().items()}
    traces = dict(initial=trace, trained=trace_readout(agent, bank))
    states = dict(initial=initial, trained=trained)
    report = compare_heads(agent, states, traces, bank)
    assert set(report) == {"initial", "trained", "trained_heads_initial_features", "trained_recenter_only"}
    assert all(torch.equal(v, trained[k]) for k, v in agent.state_dict().items())
    changed = {k: v.clone() for k, v in traces.items()}
    changed["trained"][:, 1] += 1
    other = compare_heads(agent, states, changed, bank)
    assert other["trained_recenter_only"]["normalized_offset_rms"] > report["trained_recenter_only"]["normalized_offset_rms"]
    assert all(torch.equal(v, trained[k]) for k, v in agent.state_dict().items())
