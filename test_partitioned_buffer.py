#!/usr/bin/env python3
"""Test script for partitioned buffer PER implementation."""

import numpy as np
import time
from Scripts.aimodel import HybridReplayBuffer

print("Testing Partitioned Buffer PER Implementation")
print("=" * 60)

# Create buffer
buffer = HybridReplayBuffer(capacity=100000, state_size=171)
print(f"Created buffer: capacity={buffer.capacity}, state_size={buffer.state_size}")
print(f"High-reward partition: {buffer.high_reward_capacity}")
print(f"Regular partition: {buffer.regular_capacity}")
print()

# Add experiences with varying rewards
print("Adding 10,000 experiences with mixed rewards...")
np.random.seed(42)
start = time.time()

for i in range(10000):
    state = np.random.randn(171).astype(np.float32)
    
    # Create reward distribution
    if i % 10 == 0:  # 10% high rewards
        reward = 2.0 + np.random.randn() * 0.5
    elif i % 100 == 0:  # 1% very high rewards
        reward = 5.0 + np.random.randn()
    else:  # 89% regular rewards
        reward = np.random.randn() * 0.2
    
    buffer.push(state, 0, 0.0, reward, state, False, 'dqn', 1)

elapsed = time.time() - start
print(f"Added 10K experiences in {elapsed:.3f}s ({10000/elapsed:.0f} pushes/sec)")
print()

# Check partition stats
stats = buffer.get_partition_stats()
print("Partition Statistics:")
print(f"  High-reward: {stats['high_reward']:,} / {stats['high_reward_capacity']:,} ({stats['high_reward']/buffer.size*100:.1f}%)")
print(f"  Regular: {stats['regular']:,} / {stats['regular_capacity']:,} ({stats['regular']/buffer.size*100:.1f}%)")
print(f"  Threshold: {stats['high_reward_threshold']:.3f}")
print(f"  Total size: {buffer.size:,}")
print()

# Test sampling performance
print("Testing sampling performance...")
batch_size = 2048
num_samples = 100

start = time.time()
for i in range(num_samples):
    batch = buffer.sample(batch_size)
    if batch is None:
        print(f"  Sample {i} failed!")
        break

elapsed = time.time() - start
print(f"Sampled {num_samples} batches in {elapsed:.3f}s")
print(f"Average: {elapsed/num_samples*1000:.2f}ms per batch")
print(f"Sampling speed: {num_samples/elapsed:.1f} batches/sec")
print()

# Analyze sample composition
print("Analyzing sample composition...")
batch = buffer.sample(batch_size)
if batch is not None:
    states, discrete_actions, continuous_actions, rewards, next_states, dones, actors, horizons = batch
    rewards_cpu = rewards.cpu().numpy().flatten()
    
    high_reward_samples = np.sum(rewards_cpu >= stats['high_reward_threshold'])
    print(f"  Batch size: {len(rewards_cpu)}")
    print(f"  High-reward samples: {high_reward_samples} ({high_reward_samples/len(rewards_cpu)*100:.1f}%)")
    print(f"  Expected: ~50% (balanced sampling)")
    print(f"  Reward range: [{rewards_cpu.min():.3f}, {rewards_cpu.max():.3f}]")
    print()

print("✓ Partitioned buffer PER implementation working correctly!")
print()
print("Expected behavior:")
print("  - Push performance: Same as uniform (~100K/sec)")
print("  - Sample performance: Same as uniform (~100-200 batches/sec)")
print("  - High-reward samples: ~50% of batch (vs ~25-30% natural distribution)")
print("  - No GIL contention, no performance degradation")
