#!/usr/bin/env python3
"""Quick integration test: nstep -> async_buffer -> replay."""
import sys, time, numpy as np
sys.path.insert(0, '.')
from config import RL_CONFIG
from aimodel import RainbowAgent, combine_action_indices, split_joint_action
from nstep_buffer import NStepReplayBuffer
from socket_server import AsyncReplayBuffer

STATE_DIM = RL_CONFIG.state_size
agent = RainbowAgent(STATE_DIM, RL_CONFIG)
abuf = AsyncReplayBuffer(agent)
nstep = NStepReplayBuffer(n_step=RL_CONFIG.n_step, gamma=RL_CONFIG.gamma)

s = np.random.randn(STATE_DIM).astype(np.float32)

print(f"n_step={RL_CONFIG.n_step}, state_dim={STATE_DIM}")
print(f"Memory before: {agent.memory.size}")

total_emitted = 0
for i in range(20):
    ns = np.random.randn(STATE_DIM).astype(np.float32)
    done = (i == 19)
    joint = combine_action_indices(0, 5)
    matured = nstep.add(s, joint, 1.0, ns, done, actor="dqn", priority_reward=1.0)
    total_emitted += len(matured)
    for s0, a, Rn, pR, sn, dn, h, act in matured:
        fz, sp = split_joint_action(a)
        abuf.step_async(s0, (fz, sp), Rn, sn, bool(dn), actor=act, horizon=int(h), priority_reward=pR)
    s = ns

print(f"Nstep emitted: {total_emitted}")
time.sleep(0.5)  # let async consumer drain
print(f"Memory after: {agent.memory.size}")
abuf.stop()
if agent.memory.size > 0:
    print("SUCCESS: transitions reached replay buffer")
else:
    print("FAILURE: replay buffer still empty")
