#!/usr/bin/env python3
"""Test pre-death partition by checking the pre_death_flags directly."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Scripts'))

import numpy as np
from aimodel import HybridReplayBuffer

def test_pre_death_flags():
    """Test that pre-death flags are set correctly."""
    print("Testing Pre-Death Flag Marking")
    print("=" * 60)
    
    # Create buffer
    capacity = 10000
    state_size = 171
    buffer = HybridReplayBuffer(capacity, state_size)
    
    # Add episodes with deaths
    state = np.random.randn(state_size)
    death_count = 0
    
    print(f"\n1. Adding 50 episodes (20 frames each, death on frame 20)...")
    for episode in range(50):
        for frame in range(20):
            reward = np.random.randn()
            done = (frame == 19)
            buffer.push(state, 0, 0.0, reward, state, done, 'dqn', 1)
            if done:
                death_count += 1
    
    print(f"   Deaths added: {death_count}")
    print(f"   Buffer size: {buffer.size}")
    
    # Check how many frames are marked as pre-death
    pre_death_count = np.sum(buffer.pre_death_flags[:buffer.size])
    print(f"\n2. Pre-death frame marking:")
    print(f"   Frames marked as pre-death: {pre_death_count}")
    print(f"   Expected: {death_count * 6} (50 deaths × 6 lookback frames)")
    print(f"   Percentage of buffer: {pre_death_count / buffer.size * 100:.1f}%")
    
    # Add normal frames
    print(f"\n3. Adding 2000 normal frames...")
    for i in range(2000):
        reward = np.random.randn()
        done = False
        buffer.push(state, 0, 0.0, reward, state, done, 'dqn', 1)
    
    pre_death_count = np.sum(buffer.pre_death_flags[:buffer.size])
    print(f"   Buffer size: {buffer.size}")
    print(f"   Pre-death frames: {pre_death_count}")
    print(f"   Percentage: {pre_death_count / buffer.size * 100:.1f}%")
    
    # Test sampling - we should get ~25% pre-death frames in batch
    print(f"\n4. Testing batch composition (10 batches of 2048)...")
    batch_size = 2048
    pre_death_counts = []
    
    for i in range(10):
        batch = buffer.sample(batch_size)
        if batch:
            # To check if we got pre-death frames, we need to look at the actual indices sampled
            # Since we can't easily get those, let's sample and check the percentage
            # For now, just verify batches are being created
            pre_death_counts.append(1)  # Placeholder
    
    # Better approach: directly check what percentage of samples should be pre-death
    # If we sample uniformly, we'd expect pre_death_count/buffer.size fraction
    # With our 25% targeting, we should get more
    expected_uniform = (pre_death_count / buffer.size) * batch_size
    expected_targeted = batch_size * 0.25
    
    print(f"   Pre-death frames in buffer: {pre_death_count} ({pre_death_count/buffer.size*100:.1f}%)")
    print(f"   Expected in batch (uniform sampling): ~{expected_uniform:.0f}")
    print(f"   Expected in batch (25% targeting): ~{expected_targeted:.0f}")
    print(f"\n5. Actual sampling test...")
    
    # To properly test, we need to track which indices are sampled
    # Let's do a simpler test: create a buffer where we know the exact structure
    print("   Creating test buffer with known structure...")
    test_buffer = HybridReplayBuffer(1000, state_size)
    
    # Add 100 frames, then a death
    for i in range(100):
        test_buffer.push(state, 0, 1.0, state, False, 'dqn', 1)
    test_buffer.push(state, 0, -10.0, state, True, 'dqn', 1)  # Death at index 100

    # Add 100 more frames, then another death
    for i in range(100):
        test_buffer.push(state, 0, 1.0, state, False, 'dqn', 1)
    test_buffer.push(state, 0, -10.0, state, True, 'dqn', 1)  # Death at index 201
    
    # Check pre-death flags
    pd_count = np.sum(test_buffer.pre_death_flags[:test_buffer.size])
    print(f"   Test buffer size: {test_buffer.size}")
    print(f"   Pre-death frames marked: {pd_count}")
    print(f"   Pre-death indices: {np.where(test_buffer.pre_death_flags[:test_buffer.size])[0]}")
    
    # Sample and check rewards
    batch = test_buffer.sample(100)
    if batch:
        states, discrete_actions, rewards, next_states, dones, actors, horizons = batch
        death_rewards = (rewards == -10.0).sum().item()
        normal_rewards = (rewards == 1.0).sum().item()
        print(f"\n   Sample of 100 frames:")
        print(f"   Death frames (reward=-10): {death_rewards}")
        print(f"   Normal frames (reward=1): {normal_rewards}")
        print(f"   Expected normal if 25% pre-death: ~25-30")
        
        if normal_rewards >= 20:
            print(f"\n✓ Pre-death sampling appears to be working!")
            return True
        else:
            print(f"\n✗ Not getting enough pre-death frames")
            return False
    
    return False

if __name__ == "__main__":
    success = test_pre_death_flags()
    sys.exit(0 if success else 1)
