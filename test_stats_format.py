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
    """Test the updated Stats column format in metrics display."""
    print("Testing updated Stats column format...\n")
    
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
    
    print("Filling buffer with test experiences to create varied bucket distributions...")
    for i in range(10000):
        state = np.random.randn(state_size).astype(np.float32)
        next_state = np.random.randn(state_size).astype(np.float32)
        discrete_action = np.random.randint(0, 4)
        continuous_action = np.random.uniform(-0.9, 0.9)
        reward = np.random.randn()
        done = (i % 100 == 0)
        
        # Create varied TD-error distribution
        if i < 1000:
            td_error = np.random.gamma(5, 2)  # Very high errors -> p90-100
            actor = 'dqn'
        elif i < 2000:
            td_error = np.random.gamma(3, 1.5)  # High errors -> p80-90
            actor = 'dqn'
        elif i < 4000:
            td_error = np.random.uniform(1, 3)  # Medium errors -> p70-80, p60-70
            actor = 'dqn'
        elif i < 6000:
            td_error = np.random.uniform(0.5, 2)  # Low-medium -> p50-60
            actor = 'expert'
        else:
            td_error = np.random.exponential(0.5)  # Low errors -> main
            actor = 'expert'
        
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
    
    # Import and test the metrics display
    from metrics_display import display_metrics_row, display_metrics_header
    
    kb_handler = MockKeyboardHandler()
    
    print("Displaying header and sample row with new Stats format:\n")
    display_metrics_header()
    display_metrics_row(agent, kb_handler)
    
    # Also show bucket stats for reference
    print("\n" + "="*80)
    print("For reference, here are the detailed bucket stats:")
    print("="*80)
    
    pstats = agent.memory.get_partition_stats()
    print(f"\nBucket Fill Percentages:")
    
    # Dynamically get bucket names
    bucket_names = []
    for key in pstats.keys():
        if key.startswith('p') and key.endswith('_fill_pct') and key != 'main_fill_pct':
            bucket_names.append(key.replace('_fill_pct', ''))
    
    bucket_names.sort(key=lambda x: int(x.split('_')[0][1:]), reverse=True)
    
    for name in bucket_names:
        fill_pct = pstats.get(f'{name}_fill_pct', 0)
        # Format name nicely (e.g., p98_100 -> p98-100)
        display_name = name.replace('_', '-')
        print(f"  {display_name}: {fill_pct:.1f}%")
    
    print(f"  main:    {pstats.get('main_fill_pct', 0):.1f}%")
    
    # Build expected format string dynamically
    expected_parts = [f"105k/225120"]
    for name in bucket_names:
        expected_parts.append(f"{pstats.get(f'{name}_fill_pct', 0):.0f}%")
    expected_parts.append(f"{pstats.get('main_fill_pct', 0):.0f}%")
    
    print(f"\nExpected Stats format: {'/'.join(expected_parts)}")
    
    print("\n✓ Stats column format test completed successfully!")

if __name__ == '__main__':
    test_stats_column_format()
