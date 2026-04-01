#!/usr/bin/env python3
"""Quick sanity test for the sample() lock-release fix."""
import numpy as np
from replay_buffer import PrioritizedReplayBuffer

buf = PrioritizedReplayBuffer(capacity=1000, state_size=16)
for i in range(500):
    state = np.full(16, float(i), dtype=np.float32)
    next_state = np.full(16, float(i + 1000), dtype=np.float32)
    buf.add(state, i % 8, float(i) * 0.1, next_state, i == 499,
            horizon=1, expert=1 if i < 50 else 0, wave_number=max(1, i % 20))

batch = buf.sample(32, beta=0.6)
assert batch is not None
states, actions, rewards, next_states, dones, horizons, is_expert, is_self, waves, starts, indices, weights = batch
for j in range(32):
    idx = indices[j]
    assert np.allclose(states[j], buf.states[idx]), f"State mismatch at {j}"
    assert np.allclose(next_states[j], buf.next_states[idx]), f"Next state mismatch at {j}"
    assert actions[j] == buf.actions[idx], f"Action mismatch at {j}"
    assert np.isclose(rewards[j], buf.rewards[idx]), f"Reward mismatch at {j}"

print(f"OK: sample() returned correct data for all 32 items (buf={len(buf)}, shape={states.shape})")
