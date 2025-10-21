#!/usr/bin/env python3
"""Test script to verify the bucket stats hotkey 'b' works correctly."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Scripts'))

import numpy as np
from aimodel import HybridReplayBuffer, HybridDQNAgent
from config import RL_CONFIG

# Mock keyboard handler for testing
class MockKeyboardHandler:
    def set_raw_mode(self):
        pass

def test_bucket_stats_display():
    """Test the bucket stats display function."""
    print("Testing N-bucket replay buffer statistics display...\n")
    
    # Create agent with replay buffer
    agent = HybridDQNAgent(
        state_size=128,
        discrete_actions=4,
        learning_rate=0.0001,
        gamma=0.99,
        epsilon=0.1,
        epsilon_min=0.01,
        memory_size=100000,
        batch_size=64
    )
    
    # Fill buffer with some test data
    print("Filling buffer with test experiences...")
    state_size = 128
    
    # Add varied TD-error experiences to see bucket distribution
    for i in range(5000):
        state = np.random.randn(state_size).astype(np.float32)
        next_state = np.random.randn(state_size).astype(np.float32)
        discrete_action = np.random.randint(0, 4)
        continuous_action = np.random.uniform(-0.9, 0.9)
        reward = np.random.randn()
        done = (i % 100 == 0)  # Terminal every 100 steps
        
        # Vary TD-errors: some low, some medium, some high
        if i < 2000:
            td_error = np.random.exponential(0.5)  # Low TD-errors
            actor = 'expert'
        elif i < 4000:
            td_error = np.random.uniform(0, 3)  # Medium TD-errors
            actor = 'dqn'
        else:
            td_error = np.random.gamma(2, 2)  # Higher TD-errors
            actor = 'dqn'
        
        agent.memory.push(
            state=state,
            discrete_action=discrete_action,
            continuous_action=continuous_action,
            reward=reward,
            next_state=next_state,
            done=done,
            actor=actor,
            horizon=1,
            td_error=td_error
        )
    
    print(f"Added {len(agent.memory)} experiences to buffer\n")
    
    # Import and call the print_bucket_stats function
    from main import print_bucket_stats
    
    kb_handler = MockKeyboardHandler()
    print_bucket_stats(agent, kb_handler)
    
    print("\n✓ Bucket stats display test completed successfully!")

if __name__ == '__main__':
    test_bucket_stats_display()
