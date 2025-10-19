#!/usr/bin/env python3
"""Diagnose why agreement starts worse than random (13.4% < 25%)"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Scripts'))

import torch
import numpy as np
from aimodel import HybridDQNAgent
from config import RL_CONFIG

def diagnose_agreement():
    """Check Q-value distribution and action bias"""
    print("=" * 80)
    print("AGREEMENT DIAGNOSTIC")
    print("=" * 80)
    
    # Create agent
    agent = HybridDQNAgent(
        state_size=RL_CONFIG.state_size,
        discrete_actions=4,
        learning_rate=RL_CONFIG.lr,
    )
    
    # Load model
    try:
        checkpoint = torch.load('models/tempest_model_latest.pt', map_location=agent.device)
        agent.qnetwork_local.load_state_dict(checkpoint['local_state_dict'])
        agent.qnetwork_inference.load_state_dict(checkpoint['local_state_dict'])
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return
    
    # Generate random states
    num_samples = 1000
    random_states = np.random.randn(num_samples, RL_CONFIG.state_size).astype(np.float32)
    
    # Get Q-values for random states
    agent.qnetwork_inference.eval()
    with torch.no_grad():
        states_tensor = torch.FloatTensor(random_states).to(agent.device)
        discrete_q, _ = agent.qnetwork_inference(states_tensor)
        discrete_actions = discrete_q.argmax(dim=1)
    
    # Analyze action distribution
    action_counts = np.bincount(discrete_actions.cpu().numpy(), minlength=4)
    action_percentages = 100.0 * action_counts / num_samples
    
    print(f"\nAction Distribution (n={num_samples} random states):")
    print("-" * 60)
    for i in range(4):
        fire = "F" if (i & 0b10) else "-"
        zap = "Z" if (i & 0b01) else "-"
        print(f"  Action {i} ({fire}{zap}): {action_counts[i]:4d} ({action_percentages[i]:5.1f}%)")
    
    # Calculate expected random baseline
    random_baseline = 25.0
    print(f"\nRandom Baseline (uniform): {random_baseline:.1f}%")
    
    # Analyze Q-value statistics
    q_values_np = discrete_q.cpu().numpy()
    print(f"\nQ-Value Statistics:")
    print("-" * 60)
    print(f"  Mean:   {q_values_np.mean():8.3f}")
    print(f"  Std:    {q_values_np.std():8.3f}")
    print(f"  Min:    {q_values_np.min():8.3f}")
    print(f"  Max:    {q_values_np.max():8.3f}")
    print(f"  Range:  [{q_values_np.min():.3f}, {q_values_np.max():.3f}]")
    
    # Per-action Q-value statistics
    print(f"\nPer-Action Q-Value Means:")
    print("-" * 60)
    for i in range(4):
        fire = "F" if (i & 0b10) else "-"
        zap = "Z" if (i & 0b01) else "-"
        mean_q = q_values_np[:, i].mean()
        std_q = q_values_np[:, i].std()
        print(f"  Action {i} ({fire}{zap}): {mean_q:8.3f} ± {std_q:6.3f}")
    
    # Check if there's a strong bias
    max_action_pct = action_percentages.max()
    min_action_pct = action_percentages.min()
    bias_ratio = max_action_pct / min_action_pct if min_action_pct > 0 else float('inf')
    
    print(f"\nBias Analysis:")
    print("-" * 60)
    print(f"  Max action %:     {max_action_pct:5.1f}%")
    print(f"  Min action %:     {min_action_pct:5.1f}%")
    print(f"  Bias ratio:       {bias_ratio:5.2f}x")
    
    if bias_ratio > 2.0:
        print(f"  ⚠️  WARNING: Strong action bias detected (>{bias_ratio:.1f}x)")
        print(f"      This explains why agreement < 25% random baseline")
    elif max_action_pct < 30.0:
        print(f"  ✓ Action distribution is reasonably balanced")
    
    # Check superzap gate effect
    zap_actions = action_counts[1] + action_counts[3]  # Actions with zap bit set
    no_zap_actions = action_counts[0] + action_counts[2]  # Actions without zap
    zap_percentage = 100.0 * zap_actions / num_samples
    
    print(f"\nSuperzap Gate Analysis:")
    print("-" * 60)
    print(f"  Zap actions (1,3):    {zap_actions:4d} ({zap_percentage:5.1f}%)")
    print(f"  No-zap actions (0,2): {no_zap_actions:4d} ({100-zap_percentage:5.1f}%)")
    print(f"  Superzap enabled:     {RL_CONFIG.enable_superzap_gate}")
    print(f"  Superzap prob:        {RL_CONFIG.superzap_prob:.3f}")
    
    if RL_CONFIG.enable_superzap_gate and zap_percentage > 10.0:
        print(f"  ⚠️  Model predicts {zap_percentage:.1f}% zap actions")
        print(f"      but superzap gate only allows {RL_CONFIG.superzap_prob*100:.1f}% success rate")
        print(f"      This creates action mismatch in replay buffer!")
    
    print("\n" + "=" * 80)
    print("DIAGNOSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    diagnose_agreement()
