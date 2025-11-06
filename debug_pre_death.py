#!/usr/bin/env python3
"""Debug pre-death partition sampling logic."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Scripts'))

import numpy as np
from config import RL_CONFIG
from aimodel import HybridReplayBuffer

def debug_pre_death_sampling():
    """Debug the pre-death sampling to see what's happening."""
    print("Debugging Pre-Death Sampling")
    print("=" * 60)
    
    # Create small buffer for easier debugging
    capacity = 1000
    state_size = RL_CONFIG.state_size
    buffer = HybridReplayBuffer(capacity, state_size)
    
    # Add exactly 20 frames per episode, with death on frame 20
    state = np.random.randn(state_size)
    
    print(f"\n1. Adding 10 episodes (20 frames each, death on frame 20)...")
    for episode in range(10):
        for frame in range(20):
            reward = 0.5 if frame < 19 else -10.0  # Negative reward for death
            done = (frame == 19)
            buffer.push(state, 0, 0.0, reward, state, done, 'dqn', 1)
            if done:
                actual_idx = buffer.size - 1  # The index where this death was stored
                print(f"   Episode {episode}: Death at buffer index {actual_idx}")
    
    print(f"\n2. Buffer state:")
    print(f"   Total size: {buffer.size}")
    print(f"   Terminal indices tracked: {len(buffer.terminal_indices)}")
    print(f"   Terminal indices: {list(buffer.terminal_indices)[:20]}")  # Show first 20
    
    print(f"\n3. Manually checking what pre-death sampling should give:")
    from config import RL_CONFIG
    print(f"   Lookback range: {RL_CONFIG.replay_terminal_lookback_min} to {RL_CONFIG.replay_terminal_lookback_max}")
    
    # Check a few terminal indices
    for i, term_idx in enumerate(list(buffer.terminal_indices)[:5]):
        print(f"   Terminal {i} at index {term_idx}:")
        for lookback in range(5, 11):
            pre_death_idx = term_idx - lookback
            if pre_death_idx >= 0:
                print(f"      Lookback {lookback}: index {pre_death_idx}, done={buffer.dones[pre_death_idx]}, reward={buffer.rewards[pre_death_idx]:.2f}")
    
    print(f"\n4. Testing sample() method:")
    batch_size = 100
    
    # Capture indices by modifying sample to return them
    # We'll do this by sampling and checking
    batch = buffer.sample(batch_size)
    if batch:
        states, discrete_actions, rewards, next_states, dones, actors, horizons = batch
        print(f"   Batch size: {len(states)}")
        print(f"   Rewards in batch: min={rewards.min():.2f}, max={rewards.max():.2f}, mean={rewards.mean():.2f}")
        print(f"   Death rewards (=-10): {(rewards == -10.0).sum().item()}")
        print(f"   Normal rewards (=0.5): {(rewards == 0.5).sum().item()}")
        
        # The key insight: pre-death frames should have reward=0.5, not -10
        # So if we're successfully sampling pre-death, we should see mostly 0.5 rewards
        # Given 25% pre-death target = 25 samples, we should see at least 20+ with reward=0.5
        
        expected_predeath = batch_size // 4  # 25%
        actual_normal = (rewards == 0.5).sum().item()
        actual_death = (rewards == -10.0).sum().item()
        
        print(f"\n5. Analysis:")
        print(f"   Expected pre-death samples: ~{expected_predeath}")
        print(f"   Normal reward frames (potential pre-death): {actual_normal}")
        print(f"   Death reward frames (terminal): {actual_death}")
        
        if actual_normal > expected_predeath * 0.8:  # Allow some tolerance
            print(f"   ✓ Likely getting pre-death frames!")
        else:
            print(f"   ✗ Not getting enough pre-death frames")

if __name__ == "__main__":
    debug_pre_death_sampling()
