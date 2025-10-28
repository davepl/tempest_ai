#!/usr/bin/env python3
"""Test script to verify agreement calculation works correctly."""

import sys
import os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Scripts'))

import torch
import numpy as np
from Scripts.aimodel import HybridDQNAgent, fire_zap_to_discrete
from Scripts.training import compute_action_agreement
from Scripts.config import metrics, RL_CONFIG

def test_agreement_calculation():
    """Test that agreement calculation works with the current network."""
    print("Testing agreement calculation...")

    # Create a small agent for testing
    agent = HybridDQNAgent(
        state_size=10,  # Small state for testing
        discrete_actions=4
    )

    # Create some test data
    batch_size = 32
    state_size = 10

    # Create states
    states = torch.randn(batch_size, state_size).to(agent.device)

    # Create discrete actions (0-3)
    discrete_actions = torch.randint(0, 4, (batch_size, 1)).to(agent.device)

    # Create actors list with mix of 'dqn' and 'expert'
    actors = ['dqn' if i % 2 == 0 else 'expert' for i in range(batch_size)]

    print(f"Created batch with {batch_size} samples")
    print(f"DQN samples: {sum(1 for a in actors if a == 'dqn')}")
    print(f"Expert samples: {sum(1 for a in actors if a == 'expert')}")

    # Get policy Q-values from network
    with torch.no_grad():
        agent.qnetwork_local.eval()
        policy_q = agent.qnetwork_local(states)
        agent.qnetwork_local.train()

    print(f"Policy Q shape: {policy_q.shape}")
    print(f"Q value range: {policy_q.min().item():.3f} to {policy_q.max().item():.3f}")

    # Get greedy actions
    greedy_actions = policy_q.argmax(dim=1, keepdim=True)
    print(f"Greedy actions shape: {greedy_actions.shape}")

    # Calculate action matches
    action_matches = (greedy_actions == discrete_actions).float().squeeze(1)
    print(f"Action matches shape: {action_matches.shape}")

    # Filter to DQN only
    actors_np = np.array(actors)
    dqn_mask_np = actors_np == "dqn"
    n_dqn = int(dqn_mask_np.sum())

    print(f"DQN mask sum: {n_dqn}")

    if n_dqn > 0:
        dqn_indices = torch.tensor(np.nonzero(dqn_mask_np)[0], dtype=torch.long, device=states.device)
        agree_mean = float(action_matches[dqn_indices].mean().item())
        print(f"Agreement on DQN samples: {agree_mean:.3f} ({agree_mean*100:.1f}%)")

        # Show some examples
        print("\nFirst 10 DQN samples:")
        print("Replay Action | Greedy Action | Match")
        for i in range(min(10, len(dqn_indices))):
            idx = dqn_indices[i].item()
            replay_action = discrete_actions[idx].item()
            greedy_action = greedy_actions[idx].item()
            match = "✓" if replay_action == greedy_action else "✗"
            print("8")
    else:
        print("No DQN samples found!")

    print("✓ Agreement calculation test completed!")

if __name__ == "__main__":
    test_agreement_calculation()


def test_spinner_sign_agreement():
    """Sign-aware spinner comparison should treat identical fire/zap with matching signs as agreement."""
    spinner_levels = tuple(getattr(RL_CONFIG, "spinner_command_levels", (0,)))
    assert spinner_levels, "spinner_command_levels must not be empty"
    spinner_actions = len(spinner_levels)
    device = torch.device("cpu")

    zero_idx = next((i for i, v in enumerate(spinner_levels) if v == 0), None)
    pos_indices = [i for i, v in enumerate(spinner_levels) if v > 0]
    neg_indices = [i for i, v in enumerate(spinner_levels) if v < 0]

    assert pos_indices, "Expected at least one positive spinner bucket"
    assert neg_indices, "Expected at least one negative spinner bucket"

    pos_idx = pos_indices[0]
    alt_pos_idx = pos_indices[1] if len(pos_indices) > 1 else pos_indices[0]
    neg_idx = neg_indices[0]
    zero_idx = zero_idx if zero_idx is not None else pos_idx

    def make_action(fire: int, zap: int, spinner_idx: int) -> int:
        fire_zap_idx = fire_zap_to_discrete(bool(fire), bool(zap))
        return fire_zap_idx * spinner_actions + spinner_idx

    greedy_actions = torch.tensor(
        [
            make_action(1, 0, pos_idx),       # exact match
            make_action(1, 0, pos_idx),       # same sign different magnitude
            make_action(1, 0, pos_idx),       # opposite sign
            make_action(1, 0, pos_idx),       # fire/zap mismatch
            make_action(1, 0, zero_idx),      # zero spinner reference
            make_action(1, 0, zero_idx),      # zero vs positive sign
        ],
        dtype=torch.long,
    )

    taken_actions = torch.tensor(
        [
            make_action(1, 0, pos_idx),       # identical
            make_action(1, 0, alt_pos_idx),   # same sign bucket
            make_action(1, 0, neg_idx),       # opposite sign
            make_action(0, 1, alt_pos_idx),   # fire/zap mismatch
            make_action(1, 0, zero_idx),      # zero matched
            make_action(1, 0, pos_idx),       # zero vs positive
        ],
        dtype=torch.long,
    )

    agreement = compute_action_agreement(
        greedy_actions.view(-1, 1),
        taken_actions.view(-1, 1),
        spinner_actions,
        device,
    )

    expected = torch.tensor([True, True, False, False, True, False], dtype=torch.bool)
    assert torch.equal(agreement.cpu(), expected)
