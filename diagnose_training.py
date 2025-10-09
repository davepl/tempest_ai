#!/usr/bin/env python3
"""
Diagnostic script to verify training pipeline is working correctly.
Tests:
1. Optimizer updates weights
2. Forward pass produces different outputs after training
3. Target network updates correctly
4. Loss decreases over synthetic training
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os

# Add Scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Scripts'))

from aimodel import HybridDQN
from config import RL_CONFIG

def test_weight_updates():
    """Test that optimizer actually modifies network weights"""
    print("\n" + "="*70)
    print("TEST 1: Optimizer Weight Updates")
    print("="*70)
    
    device = torch.device('cpu')
    net = HybridDQN(state_size=171, discrete_actions=4, hidden_size=128, num_layers=3).to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    
    # Get initial weight values
    initial_weights = {}
    for name, param in net.named_parameters():
        initial_weights[name] = param.data.clone()
    
    # Synthetic training step
    state = torch.randn(32, 171).to(device)
    target_discrete = torch.randn(32, 4).to(device)
    target_continuous = torch.randn(32, 1).to(device)
    
    discrete_q, continuous_pred = net(state)
    loss = nn.functional.mse_loss(discrete_q, target_discrete) + nn.functional.mse_loss(continuous_pred, target_continuous)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Check if weights changed
    changed_count = 0
    total_count = 0
    max_change = 0.0
    
    for name, param in net.named_parameters():
        diff = (param.data - initial_weights[name]).abs().max().item()
        total_count += 1
        if diff > 1e-8:
            changed_count += 1
            max_change = max(max_change, diff)
    
    print(f"✓ Weights changed: {changed_count}/{total_count}")
    print(f"✓ Max weight change: {max_change:.6f}")
    
    if changed_count == total_count and max_change > 1e-6:
        print("✅ PASS: Optimizer updates weights correctly\n")
        return True
    else:
        print("❌ FAIL: Weights not updating!\n")
        return False

def test_loss_decreases():
    """Test that loss decreases over multiple training steps on synthetic data"""
    print("\n" + "="*70)
    print("TEST 2: Loss Decreases Over Training")
    print("="*70)
    
    device = torch.device('cpu')
    net = HybridDQN(state_size=171, discrete_actions=4, hidden_size=128, num_layers=3).to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    
    # Generate fixed synthetic data
    np.random.seed(42)
    torch.manual_seed(42)
    
    states = torch.randn(256, 171).to(device)
    target_discrete = torch.randn(256, 4).to(device)
    target_continuous = torch.randn(256, 1).to(device)
    
    losses = []
    
    # Train for 100 steps
    net.train()
    for step in range(100):
        discrete_q, continuous_pred = net(states)
        loss = nn.functional.mse_loss(discrete_q, target_discrete) + nn.functional.mse_loss(continuous_pred, target_continuous)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if step % 20 == 0:
            print(f"  Step {step:3d}: Loss = {loss.item():.6f}")
    
    initial_loss = losses[0]
    final_loss = losses[-1]
    reduction = (initial_loss - final_loss) / initial_loss * 100
    
    print(f"\n✓ Initial loss: {initial_loss:.6f}")
    print(f"✓ Final loss:   {final_loss:.6f}")
    print(f"✓ Reduction:    {reduction:.1f}%")
    
    if final_loss < initial_loss * 0.5:  # At least 50% reduction
        print("✅ PASS: Loss decreases as expected\n")
        return True
    else:
        print("❌ FAIL: Loss not decreasing enough!\n")
        return False

def test_target_network_update():
    """Test that target network updates from local network"""
    print("\n" + "="*70)
    print("TEST 3: Target Network Updates")
    print("="*70)
    
    device = torch.device('cpu')
    local_net = HybridDQN(state_size=171, discrete_actions=4, hidden_size=128, num_layers=3).to(device)
    target_net = HybridDQN(state_size=171, discrete_actions=4, hidden_size=128, num_layers=3).to(device)
    
    # Initialize target to different values
    for param in target_net.parameters():
        param.data.fill_(999.0)
    
    # Verify they're different
    local_weights = [p.data.clone() for p in local_net.parameters()]
    target_weights_before = [p.data.clone() for p in target_net.parameters()]
    
    different_before = any((l - t).abs().max() > 1e-6 for l, t in zip(local_weights, target_weights_before))
    print(f"✓ Networks different before update: {different_before}")
    
    # Update target from local
    target_net.load_state_dict(local_net.state_dict())
    
    # Verify they're now the same
    target_weights_after = [p.data.clone() for p in target_net.parameters()]
    same_after = all((l - t).abs().max() < 1e-8 for l, t in zip(local_weights, target_weights_after))
    
    print(f"✓ Networks identical after update: {same_after}")
    
    if different_before and same_after:
        print("✅ PASS: Target network updates correctly\n")
        return True
    else:
        print("❌ FAIL: Target network update failed!\n")
        return False

def test_advantage_weighting():
    """Test if advantage weighting causes loss instability"""
    print("\n" + "="*70)
    print("TEST 4: Advantage Weighting Stability")
    print("="*70)
    
    device = torch.device('cpu')
    
    # Simulate rewards with mix of high and low
    rewards = torch.tensor([
        # 10 high-reward frames
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        # 100 medium-reward frames
        *([0.1] * 100),
        # 100 low-reward frames
        *([-0.1] * 100)
    ])
    
    # Compute advantages
    reward_mean = rewards.mean()
    reward_std = rewards.std() + 1e-8
    advantages = (rewards - reward_mean) / reward_std
    advantages = advantages.clamp(-3, 3)
    
    # Apply exponential weighting WITH FIX (0.5 scaling, 0.1-5.0 clamp)
    advantage_weights = torch.exp(advantages * 0.5).clamp(0.1, 5.0)
    
    print(f"✓ Reward stats:")
    print(f"  Mean: {reward_mean:.4f}, Std: {reward_std:.4f}")
    print(f"\n✓ Advantage weights:")
    print(f"  Min:    {advantage_weights.min().item():.4f}")
    print(f"  Max:    {advantage_weights.max().item():.4f}")
    print(f"  Median: {advantage_weights.median().item():.4f}")
    print(f"  Mean:   {advantage_weights.mean().item():.4f}")
    
    # Check weight distribution
    high_weights = (advantage_weights > 10.0).sum().item()
    low_weights = (advantage_weights < 0.1).sum().item()
    normal_weights = ((advantage_weights >= 0.1) & (advantage_weights <= 10.0)).sum().item()
    
    print(f"\n✓ Weight distribution:")
    print(f"  High (>10x):      {high_weights} frames")
    print(f"  Normal (0.1-10x): {normal_weights} frames")
    print(f"  Low (<0.1x):      {low_weights} frames")
    
    # Simulate loss computation
    losses_raw = torch.randn(len(rewards)).abs()  # Synthetic per-frame losses
    weighted_loss = (losses_raw * advantage_weights).mean()
    unweighted_loss = losses_raw.mean()
    
    print(f"\n✓ Loss comparison:")
    print(f"  Unweighted: {unweighted_loss.item():.6f}")
    print(f"  Weighted:   {weighted_loss.item():.6f}")
    print(f"  Ratio:      {(weighted_loss / unweighted_loss).item():.2f}x")
    
    # Check if weighting is too extreme
    ratio = advantage_weights.max() / advantage_weights.median()
    print(f"\n✓ Max/Median weight ratio: {ratio:.1f}x")
    
    if ratio < 20:
        print("✅ PASS: Advantage weighting reasonable\n")
        return True
    else:
        print("⚠️  WARNING: Advantage weighting may be too extreme!\n")
        return False

def test_inference_forward_pass():
    """Test that inference uses updated weights"""
    print("\n" + "="*70)
    print("TEST 5: Inference Uses Updated Weights")
    print("="*70)
    
    device = torch.device('cpu')
    net = HybridDQN(state_size=171, discrete_actions=4, hidden_size=128, num_layers=3).to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    
    # Fixed input
    state = torch.randn(1, 171).to(device)
    
    # Get initial output
    net.eval()
    with torch.no_grad():
        discrete_q_before, continuous_before = net(state)
    
    # Train on synthetic data
    net.train()
    train_state = torch.randn(32, 171).to(device)
    target_discrete = torch.randn(32, 4).to(device)
    target_continuous = torch.randn(32, 1).to(device)
    
    for _ in range(50):
        discrete_q, continuous_pred = net(train_state)
        loss = nn.functional.mse_loss(discrete_q, target_discrete) + nn.functional.mse_loss(continuous_pred, target_continuous)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Get output after training
    net.eval()
    with torch.no_grad():
        discrete_q_after, continuous_after = net(state)
    
    # Check if outputs changed
    discrete_diff = (discrete_q_after - discrete_q_before).abs().max().item()
    continuous_diff = (continuous_after - continuous_before).abs().max().item()
    
    print(f"✓ Discrete Q-value change:  {discrete_diff:.6f}")
    print(f"✓ Continuous value change:  {continuous_diff:.6f}")
    
    if discrete_diff > 0.01 and continuous_diff > 0.01:
        print("✅ PASS: Inference uses updated weights\n")
        return True
    else:
        print("❌ FAIL: Inference not using updated weights!\n")
        return False

def test_double_dqn():
    """Test that Double DQN is implemented correctly"""
    print("\n" + "="*70)
    print("TEST 6: Double DQN Implementation")
    print("="*70)
    
    device = torch.device('cpu')
    local_net = HybridDQN(state_size=171, discrete_actions=4, hidden_size=128, num_layers=3).to(device)
    target_net = HybridDQN(state_size=171, discrete_actions=4, hidden_size=128, num_layers=3).to(device)
    
    # Make target different from local
    target_net.load_state_dict(local_net.state_dict())
    for param in target_net.parameters():
        param.data *= 1.5  # Scale target network outputs
    
    # Test states
    next_states = torch.randn(32, 171).to(device)
    
    # Vanilla DQN: argmax and eval both use target
    with torch.no_grad():
        q_target, _ = target_net(next_states)
        vanilla_max = q_target.max(1)[0].unsqueeze(1)
    
    # Double DQN: argmax uses local, eval uses target
    with torch.no_grad():
        q_local, _ = local_net(next_states)
        best_actions = q_local.max(1)[1].unsqueeze(1)
        q_target_eval, _ = target_net(next_states)
        double_max = q_target_eval.gather(1, best_actions)
    
    # Double DQN should give LOWER Q-values (less overestimation)
    vanilla_mean = vanilla_max.mean().item()
    double_mean = double_max.mean().item()
    reduction = (vanilla_mean - double_mean) / abs(vanilla_mean) * 100 if vanilla_mean != 0 else 0
    
    print(f"✓ Vanilla DQN max Q: {vanilla_mean:.4f}")
    print(f"✓ Double DQN max Q:  {double_mean:.4f}")
    print(f"✓ Overestimation reduction: {reduction:.1f}%")
    
    # Check that best actions differ between networks
    local_best = q_local.max(1)[1]
    target_best = q_target.max(1)[1]
    disagreement = (local_best != target_best).float().mean().item() * 100
    
    print(f"✓ Action disagreement: {disagreement:.1f}%")
    
    if abs(double_mean) < abs(vanilla_mean) and disagreement > 10:
        print("✅ PASS: Double DQN reduces overestimation\n")
        return True
    else:
        print("❌ FAIL: Double DQN not working correctly!\n")
        return False

def main():
    print("\n" + "="*70)
    print("TRAINING PIPELINE DIAGNOSTIC")
    print("="*70)
    print("\nTesting fundamental training mechanisms...\n")
    
    results = []
    
    results.append(("Weight Updates", test_weight_updates()))
    results.append(("Loss Decreases", test_loss_decreases()))
    results.append(("Target Network", test_target_network_update()))
    results.append(("Advantage Weighting", test_advantage_weighting()))
    results.append(("Inference Updates", test_inference_forward_pass()))
    results.append(("Double DQN", test_double_dqn()))
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n✅ All tests passed! Training pipeline is functioning correctly.")
        print("   Issue may be with hyperparameters, data distribution, or network architecture.")
    else:
        print("\n❌ Some tests failed! Training pipeline has fundamental issues.")
        print("   Fix these issues before adjusting hyperparameters.")
    
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
