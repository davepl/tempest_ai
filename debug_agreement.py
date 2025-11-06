#!/usr/bin/env python3
"""
Debug low agreement issue - check if network is learning backwards
"""

import torch
import numpy as np
import sys
sys.path.append('Scripts') 

from config import metrics, RL_CONFIG
from aimodel import HybridDQNAgent

# Load the current model
agent = HybridDQNAgent(
    state_size=RL_CONFIG.state_size,
    discrete_actions=4
)

print("Loading model...")
agent.load('models/tempest_model_latest.pt')

print(f"Buffer size: {len(agent.memory)}")

if len(agent.memory) < 100:
    print("Not enough data in buffer")
    sys.exit(0)

# Sample a batch and check agreement
print("\nSampling batch to check agreement...")
indices = np.random.choice(len(agent.memory), size=min(2048, len(agent.memory)), replace=False)

states = []
actions = []
actors = []

for idx in indices:
    s, a_d, a_c, r, ns, d, h, actor = agent.memory.get_single(idx)
    states.append(s)
    actions.append(a_d)
    actors.append(actor)

states = torch.FloatTensor(np.array(states)).to(agent.device)
actions = np.array(actions)
actors = np.array(actors)

# Get greedy actions from current network
with torch.no_grad():
    agent.qnetwork_local.eval()
    q_values, _ = agent.qnetwork_local(states)
    greedy_actions = q_values.argmax(dim=1).cpu().numpy()
    agent.qnetwork_local.train()

# Filter to DQN only
dqn_mask = (actors == 0)
n_dqn = dqn_mask.sum()

if n_dqn == 0:
    print("No DQN samples in batch!")
    sys.exit(0)

greedy_dqn = greedy_actions[dqn_mask]
actions_dqn = actions[dqn_mask]

# Calculate agreement
matches = (greedy_dqn == actions_dqn)
agree_pct = matches.mean() * 100.0

print(f"\n{'='*60}")
print(f"DQN samples: {n_dqn}")
print(f"Agreement: {agree_pct:.1f}%")
print(f"\nAction distribution in replay buffer (DQN only):")
for a in range(4):
    count = (actions_dqn == a).sum()
    pct = count / len(actions_dqn) * 100
    print(f"  Action {a}: {count:4d} ({pct:5.1f}%)")

print(f"\nGreedy action distribution (what network wants):")
for a in range(4):
    count = (greedy_dqn == a).sum()
    pct = count / len(greedy_dqn) * 100
    print(f"  Action {a}: {count:4d} ({pct:5.1f}%)")

print(f"\nQ-value statistics:")
q_np = q_values.cpu().numpy()[dqn_mask]
for a in range(4):
    q_a = q_np[:, a]
    print(f"  Action {a}: mean={q_a.mean():7.2f}, std={q_a.std():6.2f}, min={q_a.min():7.2f}, max={q_a.max():7.2f}")

print(f"\nFirst 20 comparisons:")
print(f"{'Replay':>8} {'Greedy':>8} {'Match':>8}")
for i in range(min(20, len(actions_dqn))):
    match_str = "✓" if actions_dqn[i] == greedy_dqn[i] else "✗"
    print(f"{actions_dqn[i]:>8} {greedy_dqn[i]:>8} {match_str:>8}")

# Check if network is just stuck on one action
unique_greedy = np.unique(greedy_dqn)
if len(unique_greedy) == 1:
    print(f"\n⚠️  WARNING: Network is stuck choosing only action {unique_greedy[0]}!")

print(f"{'='*60}")
