#!/usr/bin/env python3
"""Test pre-death partition functionality in HybridReplayBuffer."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Scripts'))

import numpy as np
from aimodel import HybridReplayBuffer
from config import SERVER_CONFIG

def test_pre_death_partition():
    """Test that pre-death frames are tracked and sampled correctly."""
    print("Testing Pre-Death Partition Implementation")
    print("=" * 60)
    
    # Create buffer
    capacity = 10000
    state_size = SERVER_CONFIG.params_count
    buffer = HybridReplayBuffer(capacity, state_size)
    
    print(f"\n1. Buffer Configuration:")
    print(f"   Total capacity: {capacity}")
    print(f"   High-reward capacity: {buffer.high_reward_capacity} ({buffer.high_reward_capacity/capacity*100:.0f}%)")
    print(f"   Regular capacity: {buffer.regular_capacity} ({buffer.regular_capacity/capacity*100:.0f}%)")
    print(f"   Expected split: 25% high-reward, 25% pre-death, 50% regular")
    
    # Add normal experiences
    print(f"\n2. Adding normal experiences...")
    state = np.random.randn(state_size)
    for i in range(1000):
        reward = np.random.randn() * 2.0  # Random rewards
        done = False
        buffer.push(state, 0, reward, state, done, 'dqn', 1)
    
    stats = buffer.get_partition_stats()
    print(f"   Buffer size: {buffer.size}")
    print(f"   Terminal indices tracked: {stats['terminal_count']}")
    
    # Add some deaths
    print(f"\n3. Adding episodes with deaths...")
    death_count = 0
    for episode in range(50):
        # Normal frames leading to death
        for frame in range(20):
            reward = np.random.randn()
            done = (frame == 19)  # Death on last frame
            buffer.push(state, 0, reward, state, done, 'dqn', 1)
            if done:
                death_count += 1
    
    stats = buffer.get_partition_stats()
    print(f"   Deaths added: {death_count}")
    print(f"   Terminal indices tracked: {stats['terminal_count']}")
    print(f"   Buffer size: {buffer.size}")
    
    # Add more normal frames to ensure we have a good mix
    print(f"\n4. Adding more normal experiences...")
    for i in range(2000):
        reward = np.random.randn() * 2.0
        done = False
        buffer.push(state, 0, reward, state, done, 'dqn', 1)
    
    stats = buffer.get_partition_stats()
    print(f"   Buffer size: {buffer.size}")
    print(f"   High-reward partition: {stats['high_reward']}")
    print(f"   Regular partition: {stats['regular']}")
    print(f"   Terminal indices: {stats['terminal_count']}")
    print(f"   Reward threshold: {stats['high_reward_threshold']:.3f}")
    
    # Test sampling
    print(f"\n5. Testing batch sampling...")
    batch_size = 2048
    batch = buffer.sample(batch_size)
    
    if batch is None:
        print("   ERROR: Failed to sample batch!")
        return False
    
    states, discrete_actions, rewards, next_states, dones, actors, horizons = batch
    
    print(f"   Batch size: {len(states)}")
    print(f"   Terminal states in batch: {dones.sum().item()}")
    print(f"   Terminal state percentage: {dones.sum().item() / len(states) * 100:.1f}%")
    
    # Expected: With 50 deaths and 5-10 frame lookback, we should have ~375 pre-death frames
    # These are sampled at 25%, so we'd expect around 25% terminal frames in batch
    expected_terminal_pct = 25.0
    actual_terminal_pct = dones.sum().item() / len(states) * 100
    
    print(f"   Expected terminal %: ~{expected_terminal_pct:.1f}%")
    print(f"   Actual terminal %: {actual_terminal_pct:.1f}%")
    
    # Test multiple batches to verify consistency
    print(f"\n6. Testing sampling consistency (10 batches)...")
    terminal_counts = []
    for i in range(10):
        batch = buffer.sample(batch_size)
        if batch:
            _, _, _, _, _, dones, _, _ = batch
            terminal_counts.append(dones.sum().item())
    
    avg_terminals = np.mean(terminal_counts)
    std_terminals = np.std(terminal_counts)
    avg_pct = avg_terminals / batch_size * 100
    
    print(f"   Average terminals per batch: {avg_terminals:.1f} ± {std_terminals:.1f}")
    print(f"   Average terminal %: {avg_pct:.1f}%")
    
    # Check if we're getting pre-death sampling
    if stats['terminal_count'] > 0 and avg_pct > 10:
        print(f"\n✓ Pre-death partition appears to be working!")
        print(f"  - {stats['terminal_count']} terminal indices tracked")
        print(f"  - ~{avg_pct:.1f}% of batch samples are near-death frames")
        return True
    else:
        print(f"\n✗ Pre-death partition may not be working correctly")
        print(f"  - Terminal count: {stats['terminal_count']}")
        print(f"  - Terminal %: {avg_pct:.1f}%")
        return False

if __name__ == "__main__":
    success = test_pre_death_partition()
    sys.exit(0 if success else 1)
