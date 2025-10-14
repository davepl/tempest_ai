#!/usr/bin/env python3
"""
Simple test to check what values are being stored in discrete_actions
"""

import sys
sys.path.insert(0, 'Scripts')

import numpy as np
from aimodel import HybridReplayBuffer

def test_discrete_action_storage():
    """Test what gets stored in discrete_actions array"""
    print("Testing discrete action storage...")
    
    buffer = HybridReplayBuffer(1000, state_size=10)
    
    # Push some experiences with different action values
    test_actions = [0, 1, 2, 3, 0, 1, 2, 3]
    
    for i, action in enumerate(test_actions):
        state = np.random.randn(10).astype(np.float32)
        next_state = np.random.randn(10).astype(np.float32)
        continuous_action = 0.0
        reward = float(i)
        done = False
        
        print(f"Pushing action {action} (type: {type(action)})")
        buffer.push(state, action, continuous_action, reward, next_state, done, 'dqn', 1)
    
    # Check what was stored
    print("\nStored discrete_actions:")
    print(buffer.discrete_actions[:len(test_actions)])
    print(f"Data type: {buffer.discrete_actions.dtype}")
    print(f"Min value: {buffer.discrete_actions[:len(test_actions)].min()}")
    print(f"Max value: {buffer.discrete_actions[:len(test_actions)].max()}")
    
    # Check for any invalid values
    invalid = (buffer.discrete_actions[:len(test_actions)] < 0) | (buffer.discrete_actions[:len(test_actions)] >= 4)
    if np.any(invalid):
        print(f"\n❌ FOUND INVALID ACTIONS: {buffer.discrete_actions[:len(test_actions)][invalid]}")
    else:
        print("\n✓ All actions are valid (0-3)")

if __name__ == "__main__":
    test_discrete_action_storage()
