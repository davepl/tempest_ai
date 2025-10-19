#!/usr/bin/env python3
"""Test if fresh model initialization has action bias"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Scripts'))

import torch
import numpy as np
from aimodel import HybridDQNAgent
from config import RL_CONFIG

def test_fresh_model():
    """Test action distribution of a freshly initialized model"""
    print("=" * 80)
    print("TESTING FRESH MODEL INITIALIZATION")
    print("=" * 80)
    
    # Create fresh agent (no loading)
    print("\nCreating fresh model...")
    agent = HybridDQNAgent(
        state_size=RL_CONFIG.state_size,
        discrete_actions=4,
        learning_rate=RL_CONFIG.lr,
    )
    print("✓ Fresh model created")
    
    # Generate random states
    num_samples = 1000
    random_states = np.random.randn(num_samples, RL_CONFIG.state_size).astype(np.float32)
    
    # Get Q-values for random states
    agent.qnetwork_local.eval()
    with torch.no_grad():
        states_tensor = torch.FloatTensor(random_states).to(agent.device)
        discrete_q, _ = agent.qnetwork_local(states_tensor)
        discrete_actions = discrete_q.argmax(dim=1).cpu()
    
    # Analyze action distribution
    action_counts = np.bincount(discrete_actions.numpy(), minlength=4)
    action_percentages = 100.0 * action_counts / num_samples
    
    print(f"\nAction Distribution (n={num_samples} random states):")
    print("-" * 60)
    for i in range(4):
        fire = "F" if (i & 0b10) else "-"
        zap = "Z" if (i & 0b01) else "-"
        bar_length = int(action_percentages[i] / 2)
        bar = "█" * bar_length
        print(f"  Action {i} ({fire}{zap}): {action_counts[i]:4d} ({action_percentages[i]:5.1f}%) {bar}")
    
    # Calculate expected random baseline
    random_baseline = 25.0
    print(f"\nRandom Baseline (uniform): {random_baseline:.1f}%")
    
    # Check if there's a strong bias
    max_action_pct = action_percentages.max()
    min_action_pct = action_percentages.min()
    bias_ratio = max_action_pct / min_action_pct if min_action_pct > 0 else float('inf')
    
    print(f"\nBias Analysis:")
    print("-" * 60)
    print(f"  Max action %:     {max_action_pct:5.1f}%")
    print(f"  Min action %:     {min_action_pct:5.1f}%")
    print(f"  Bias ratio:       {bias_ratio:5.2f}x")
    
    if bias_ratio > 3.0:
        print(f"  ✗ FAIL: Strong action bias detected (>{bias_ratio:.1f}x)")
        print(f"      Fresh model should have balanced action distribution")
        return False
    elif max_action_pct > 35.0:
        print(f"  ⚠️  WARNING: Moderate action bias ({max_action_pct:.1f}%)")
        return False
    else:
        print(f"  ✓ PASS: Action distribution is reasonably balanced")
    
    # Analyze Q-value statistics
    q_values_np = discrete_q.cpu().numpy()
    print(f"\nQ-Value Statistics:")
    print("-" * 60)
    print(f"  Mean:   {q_values_np.mean():8.4f}")
    print(f"  Std:    {q_values_np.std():8.4f}")
    print(f"  Range:  [{q_values_np.min():.4f}, {q_values_np.max():.4f}]")
    
    # Per-action Q-value statistics
    print(f"\nPer-Action Q-Value Means:")
    print("-" * 60)
    q_means = []
    for i in range(4):
        fire = "F" if (i & 0b10) else "-"
        zap = "Z" if (i & 0b01) else "-"
        mean_q = q_values_np[:, i].mean()
        std_q = q_values_np[:, i].std()
        q_means.append(mean_q)
        print(f"  Action {i} ({fire}{zap}): {mean_q:8.4f} ± {std_q:6.4f}")
    
    # Check if Q-values are too similar (could indicate dead network)
    q_range = max(q_means) - min(q_means)
    if q_range < 0.01:
        print(f"\n  ⚠️  WARNING: Q-values are very similar (range={q_range:.4f})")
        print(f"      This might cause random action selection")
    
    print("\n" + "=" * 80)
    if bias_ratio <= 3.0 and max_action_pct <= 35.0:
        print("✓ TEST PASSED: Fresh model has balanced action distribution")
        print("=" * 80)
        return True
    else:
        print("✗ TEST FAILED: Fresh model still has action bias")
        print("=" * 80)
        return False

if __name__ == "__main__":
    success = test_fresh_model()
    sys.exit(0 if success else 1)
