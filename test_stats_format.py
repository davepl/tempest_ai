#!/usr/bin/env python3
"""Test script to verify the updated Stats column format."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Scripts'))

import numpy as np
from aimodel import HybridDQNAgent
from config import RL_CONFIG, metrics

# Mock keyboard handler for testing
class MockKeyboardHandler:
    def set_raw_mode(self):
        pass
    def restore_terminal(self):
        pass

def test_stats_column_format():
    """Smoke-test the Stats column in metrics display for the simplified buffer."""
    print("Testing uniform-buffer Stats column format...\n")
    
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
    
    # Set some test metrics
    metrics.memory_buffer_size = 105000
    metrics.total_training_steps = 225120
    
    # Fill buffer with test data to create varied bucket fills
    state_size = 128
    
    print("Filling buffer with test experiences to populate replay memory...")
    for i in range(5000):
        state = np.random.randn(state_size).astype(np.float32)
        next_state = np.random.randn(state_size).astype(np.float32)
        discrete_action = np.random.randint(0, agent.discrete_actions)
        reward = np.random.randn()
        done = (i % 100 == 0)

        actor = 'dqn' if i % 2 == 0 else 'expert'
        
        agent.memory.push(
            state=state,
            discrete_action=discrete_action,
            reward=reward,
            next_state=next_state,
            done=done,
            actor=actor,
            horizon=1,
        )
    
    print(f"Added {len(agent.memory)} experiences to buffer\n")
    
    # Import and test the metrics display
    from metrics_display import display_metrics_row, display_metrics_header
    
    kb_handler = MockKeyboardHandler()
    
    print("Displaying header and sample row with new Stats format:\n")
    display_metrics_header()
    display_metrics_row(agent, kb_handler)
    
    # Also show buffer stats for reference
    print("\n" + "="*80)
    print("Replay buffer stats snapshot:")
    print("="*80)

    pstats = agent.memory.get_partition_stats()
    print(f"  priority_buckets_enabled: {pstats.get('priority_buckets_enabled')}")
    print(f"  main_fill_pct:            {pstats.get('main_fill_pct', 0):.1f}%")
    print(f"  total_size:               {pstats.get('total_size', 0)}")
    
    print("\n✓ Stats column format smoke test completed successfully!")

if __name__ == '__main__':
    test_stats_column_format()
