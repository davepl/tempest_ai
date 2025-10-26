#!/usr/bin/env python3
"""
Diagnose why discrete DQN head isn't learning properly.
Tests for fundamental bugs in the discrete action path.
"""

import sys
import os
sys.path.insert(0, 'Scripts')

import torch
import torch.nn.functional as F
import numpy as np

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

from aimodel import HybridDQN, HybridDQNAgent
from config import RL_CONFIG, metrics

def test_network_modes():
    """Test if network modes are set correctly"""
    print("=" * 80)
    print("TEST 1: Network Mode Configuration")
    print("=" * 80)
    
    # Create a simple agent
    agent = HybridDQNAgent(
        state_size=10,
        discrete_actions=4,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=0.1,
        epsilon_min=0.05,
        memory_size=1000,
        batch_size=32
    )
    
    print(f"qnetwork_local training mode: {agent.qnetwork_local.training}")
    print(f"qnetwork_target training mode: {agent.qnetwork_target.training}")
    print(f"qnetwork_inference training mode: {agent.qnetwork_inference.training}")
    print(f"qnetwork_inference is qnetwork_local: {agent.qnetwork_inference is agent.qnetwork_local}")
    
    if agent.qnetwork_inference.training:
        print("❌ CRITICAL BUG: Inference network is in TRAINING mode!")
        print("   This means inference has non-deterministic behavior.")
    else:
        print("✓ Inference network is in eval mode (correct)")
    
    return agent

def test_discrete_gradients(agent):
    """Test if gradients flow to discrete head"""
    print("\n" + "=" * 80)
    print("TEST 2: Discrete Head Gradient Flow")
    print("=" * 80)
    
    # Create dummy data
    batch_size = 4
    state_size = agent.state_size
    
    states = torch.randn(batch_size, state_size).to(agent.device)
    discrete_actions = torch.tensor([[0], [1], [2], [3]]).to(agent.device)
    rewards = torch.tensor([[1.0], [2.0], [3.0], [4.0]]).to(agent.device)
    next_states = torch.randn(batch_size, state_size).to(agent.device)
    dones = torch.zeros(batch_size, 1).to(agent.device)
    
    # Forward pass
    discrete_q_pred = agent.qnetwork_local(states)
    discrete_q_selected = discrete_q_pred.gather(1, discrete_actions)
    
    # Compute simple target
    with torch.no_grad():
        next_q = agent.qnetwork_target(next_states)
        discrete_targets = rewards + 0.99 * next_q.max(1, keepdim=True)[0] * (1 - dones)
    
    # Compute loss for discrete head only
    d_loss = F.huber_loss(discrete_q_selected, discrete_targets, reduction='mean')
    
    # Zero gradients and backprop
    agent.optimizer.zero_grad()
    d_loss.backward()
    
    # Check if discrete head has gradients
    discrete_head_params = [
        ('discrete_fc.weight', agent.qnetwork_local.discrete_fc.weight),
        ('discrete_fc.bias', agent.qnetwork_local.discrete_fc.bias),
        ('discrete_out.weight', agent.qnetwork_local.discrete_out.weight),
        ('discrete_out.bias', agent.qnetwork_local.discrete_out.bias),
    ]
    
    has_grads = True
    for name, param in discrete_head_params:
        if param.grad is None:
            print(f"❌ {name}: NO GRADIENT")
            has_grads = False
        else:
            grad_norm = param.grad.norm().item()
            print(f"✓ {name}: grad_norm = {grad_norm:.6f}")
    
    if has_grads:
        print("\n✓ All discrete head parameters have gradients (correct)")
    else:
        print("\n❌ CRITICAL BUG: Some discrete head parameters have no gradients!")
    
    return has_grads

def test_action_consistency(agent):
    """Test if actions are consistent between training and inference"""
    print("\n" + "=" * 80)
    print("TEST 3: Action Selection Consistency")
    print("=" * 80)
    
    # Create a fixed state
    state = np.random.randn(agent.state_size).astype(np.float32)
    
    # Get action via act() method (inference path)
    discrete_action_inference = agent.act(state, epsilon=0.0, add_noise=False)
    
    # Get action via direct network forward (training path)
    state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(agent.device)
    with torch.no_grad():
        discrete_q = agent.qnetwork_local(state_tensor)
        discrete_action_training = discrete_q.argmax(dim=1).item()
    
    print(f"Inference path action: {discrete_action_inference}")
    print(f"Training path action: {discrete_action_training}")
    print(f"Q-values: {discrete_q.cpu().numpy()}")
    
    if discrete_action_inference == discrete_action_training:
        print("✓ Actions match between inference and training paths")
    else:
        print("❌ CRITICAL BUG: Action mismatch between inference and training!")
    
    return discrete_action_inference == discrete_action_training

def test_network_updates(agent):
    """Test if network actually updates after training"""
    print("\n" + "=" * 80)
    print("TEST 4: Network Parameter Updates")
    print("=" * 80)
    
    # Save initial discrete head weights
    initial_weights = agent.qnetwork_local.discrete_out.weight.data.clone()
    
    # Add some experiences to replay buffer
    for i in range(100):
        state = np.random.randn(agent.state_size).astype(np.float32)
        next_state = np.random.randn(agent.state_size).astype(np.float32)
        discrete_action = i % agent.discrete_actions
        reward = float(np.random.randn())
        done = False
        
        agent.memory.push(state, discrete_action, reward, next_state, done, 'dqn', 1)
    
    # Run multiple training steps
    print(f"Initial discrete_out.weight mean: {initial_weights.mean().item():.6f}")
    
    losses = []
    for step in range(10):
        result = agent.train_step()
        if result is not None:
            if isinstance(result, dict):
                loss_val = result.get('total_loss', 0)
            else:
                loss_val = float(result) if result else 0
            losses.append(loss_val)
            print(f"  Step {step+1}: Loss = {loss_val:.6f}")
    
    # Check if weights changed
    final_weights = agent.qnetwork_local.discrete_out.weight.data
    weight_diff = (final_weights - initial_weights).abs().mean().item()
    
    print(f"Final discrete_out.weight mean: {final_weights.mean().item():.6f}")
    print(f"Mean absolute weight change: {weight_diff:.6f}")
    
    if weight_diff > 1e-6:
        print(f"✓ Discrete head weights updated (Δ = {weight_diff:.6f})")
    else:
        print(f"❌ CRITICAL BUG: Discrete head weights did NOT update!")
    
    return weight_diff > 1e-6

def test_q_value_prediction():
    """Test if Q-values make sense"""
    print("\n" + "=" * 80)
    print("TEST 5: Q-Value Prediction Sanity Check")
    print("=" * 80)
    
    agent = HybridDQNAgent(
        state_size=10,
        discrete_actions=4,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=0.1,
        epsilon_min=0.05,
        memory_size=1000,
        batch_size=32
    )
    
    # Generate experiences with clear pattern: action 2 always gets reward +5
    for i in range(200):
        state = np.random.randn(agent.state_size).astype(np.float32)
        next_state = np.random.randn(agent.state_size).astype(np.float32)
        
        discrete_action = i % agent.discrete_actions
        # Action 2 always gets high reward
        reward = 5.0 if discrete_action == 2 else 0.0
        done = False
        
        agent.memory.push(state, discrete_action, reward, next_state, done, 'dqn', 1)
    
    # Train for several steps
    print("Training network to learn that action 2 gives reward +5...")
    for step in range(100):
        agent.train_step()
    
    # Test prediction on random states
    test_states = [np.random.randn(agent.state_size).astype(np.float32) for _ in range(10)]
    
    action_2_preferred = 0
    for state in test_states:
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(agent.device)
        with torch.no_grad():
            discrete_q = agent.qnetwork_local(state_tensor)
            best_action = discrete_q.argmax().item()
            if best_action == 2:
                action_2_preferred += 1
    
    print(f"Action 2 selected: {action_2_preferred}/10 times")
    
    if action_2_preferred >= 7:
        print("✓ Network learned to prefer high-reward action (correct)")
    else:
        print("❌ CRITICAL BUG: Network did NOT learn to prefer high-reward action!")
        print("   This suggests the discrete head is not learning from experience.")
    
    return action_2_preferred >= 7

def main():
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "DISCRETE DQN HEAD DIAGNOSTIC" + " " * 30 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    results = {}
    
    # Run tests
    agent = test_network_modes()
    results['modes'] = not agent.qnetwork_inference.training or agent.qnetwork_inference is agent.qnetwork_local
    
    results['gradients'] = test_discrete_gradients(agent)
    results['consistency'] = test_action_consistency(agent)
    results['updates'] = test_network_updates(agent)
    results['learning'] = test_q_value_prediction()
    
    # Summary
    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)
    
    all_pass = all(results.values())
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    if all_pass:
        print("\n✓ All tests passed - discrete head appears functional")
        print("  Issue may be in replay buffer sampling or data quality")
    else:
        print("\n❌ CRITICAL BUGS FOUND - discrete head has fundamental issues!")
        print("  Focus on failed tests above to identify root cause")
    
    print()

if __name__ == "__main__":
    main()
