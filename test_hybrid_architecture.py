#!/usr/bin/env python3
"""
Comprehensive unit tests for hybrid discrete-continuous action architecture.
Tests network forward pass, action selection, experience storage, training loop, and helper functions.
"""

import numpy as np
import torch
import sys
import os
from unittest.mock import patch, MagicMock

from Scripts.config import RL_CONFIG, FIRE_ZAP_MAPPING, SPINNER_MIN, SPINNER_MAX
from Scripts.aimodel import (
    HybridDQN, HybridReplayBuffer, HybridDQNAgent,
    fire_zap_to_discrete, discrete_to_fire_zap,
    get_expert_hybrid_action, hybrid_to_game_action
)

def test_fire_zap_conversion():
    """Test discrete fire/zap conversion functions"""
    print("=== Testing Fire/Zap Conversion ===")
    
    # Test all combinations
    test_cases = [
        (False, False, 0),  # No fire, no zap
        (True, False, 2),   # Fire, no zap  
        (False, True, 1),   # No fire, zap
        (True, True, 3),    # Fire, zap
    ]
    
    for fire, zap, expected_discrete in test_cases:
        discrete = fire_zap_to_discrete(fire, zap)
        assert discrete == expected_discrete, f"fire_zap_to_discrete({fire}, {zap}) = {discrete}, expected {expected_discrete}"
        
        # Test reverse conversion
        fire_back, zap_back = discrete_to_fire_zap(discrete)
        assert fire_back == fire, f"discrete_to_fire_zap({discrete}) fire = {fire_back}, expected {fire}"
        assert zap_back == zap, f"discrete_to_fire_zap({discrete}) zap = {zap_back}, expected {zap}"
    
    print("✓ Fire/zap conversion functions work correctly")

def test_hybrid_dqn_network():
    """Test HybridDQN network architecture"""
    print("\n=== Testing HybridDQN Network ===")
    
    state_size = 175
    discrete_actions = 4
    batch_size = 32
    
    # Test network creation
    network = HybridDQN(state_size=state_size, discrete_actions=discrete_actions, 
                       hidden_size=512, use_dueling=False)
    
    # Test forward pass
    states = torch.randn(batch_size, state_size)
    discrete_q, continuous_spinner = network(states)
    
    # Verify output shapes
    assert discrete_q.shape == (batch_size, discrete_actions), f"Discrete Q shape: {discrete_q.shape}"
    assert continuous_spinner.shape == (batch_size, 1), f"Continuous spinner shape: {continuous_spinner.shape}"
    
    # Verify continuous output range
    spinner_values = continuous_spinner.detach().numpy().flatten()
    assert np.all(spinner_values >= -0.9), f"Spinner values below -0.9: {spinner_values[spinner_values < -0.9]}"
    assert np.all(spinner_values <= 0.9), f"Spinner values above 0.9: {spinner_values[spinner_values > 0.9]}"
    
    print("✓ HybridDQN network forward pass works correctly")
    
    # Test dueling architecture
    dueling_network = HybridDQN(state_size=state_size, discrete_actions=discrete_actions,
                               hidden_size=512, use_dueling=True)
    
    discrete_q_dual, continuous_spinner_dual = dueling_network(states)
    assert discrete_q_dual.shape == (batch_size, discrete_actions), f"Dueling discrete Q shape: {discrete_q_dual.shape}"
    assert continuous_spinner_dual.shape == (batch_size, 1), f"Dueling continuous shape: {continuous_spinner_dual.shape}"
    
    print("✓ HybridDQN dueling architecture works correctly")

def test_hybrid_replay_buffer():
    """Test HybridReplayBuffer functionality"""
    print("\n=== Testing HybridReplayBuffer ===")
    
    capacity = 1000
    buffer = HybridReplayBuffer(capacity)
    
    # Test empty buffer
    assert len(buffer) == 0
    assert buffer.sample(32) is None
    
    # Add experiences
    state_size = 175
    num_experiences = 100
    
    for i in range(num_experiences):
        state = np.random.randn(state_size).astype(np.float32)
        discrete_action = np.random.randint(0, 4)
        continuous_action = np.random.uniform(-0.9, 0.9)
        reward = np.random.randn()
        next_state = np.random.randn(state_size).astype(np.float32)
        done = np.random.random() < 0.1
        
        buffer.push(state, discrete_action, continuous_action, reward, next_state, done)
    
    assert len(buffer) == num_experiences
    print(f"✓ Added {num_experiences} experiences to buffer")
    
    # Test sampling
    batch_size = 32
    batch = buffer.sample(batch_size)
    assert batch is not None
    
    states, discrete_actions, continuous_actions, rewards, next_states, dones = batch
    
    # Verify batch shapes
    assert states.shape == (batch_size, state_size)
    assert discrete_actions.shape == (batch_size, 1)
    assert continuous_actions.shape == (batch_size, 1)
    assert rewards.shape == (batch_size, 1)
    assert next_states.shape == (batch_size, state_size)
    assert dones.shape == (batch_size, 1)
    
    # Verify discrete action range
    discrete_vals = discrete_actions.cpu().numpy().flatten()
    assert np.all(discrete_vals >= 0) and np.all(discrete_vals < 4), f"Invalid discrete actions: {discrete_vals}"
    
    # Verify continuous action range
    continuous_vals = continuous_actions.cpu().numpy().flatten()
    assert np.all(continuous_vals >= -0.9) and np.all(continuous_vals <= 0.9), f"Invalid continuous actions: {continuous_vals}"
    
    print("✓ HybridReplayBuffer sampling works correctly")
    
    # Test buffer overflow
    for i in range(capacity + 100):
        state = np.random.randn(state_size).astype(np.float32)
        buffer.push(state, 0, 0.0, 0.0, state, False)
    
    assert len(buffer) == capacity, f"Buffer size after overflow: {len(buffer)}"
    print("✓ Buffer correctly handles overflow")

def test_hybrid_agent_action_selection():
    """Test HybridDQNAgent action selection"""
    print("\n=== Testing HybridDQNAgent Action Selection ===")
    
    state_size = 175
    agent = HybridDQNAgent(state_size=state_size, discrete_actions=4)
    
    # Test deterministic action selection (epsilon=0)
    state = np.random.randn(state_size).astype(np.float32)
    discrete_action, continuous_action, log_prob = agent.act(state, epsilon=0.0, add_noise=False)
    
    # Verify output types and ranges
    assert isinstance(discrete_action, int), f"Discrete action type: {type(discrete_action)}"
    assert isinstance(continuous_action, float), f"Continuous action type: {type(continuous_action)}"
    assert 0 <= discrete_action < 4, f"Discrete action out of range: {discrete_action}"
    assert -0.9 <= continuous_action <= 0.9, f"Continuous action out of range: {continuous_action}"
    
    print("✓ Deterministic action selection works correctly")
    
    # Test epsilon-greedy action selection
    discrete_actions = []
    continuous_actions = []
    
    for _ in range(100):
        discrete, continuous, log_prob = agent.act(state, epsilon=0.5, add_noise=True)
        discrete_actions.append(discrete)
        continuous_actions.append(continuous)
    
    # Should have some variety in actions due to exploration
    unique_discrete = len(set(discrete_actions))
    assert unique_discrete > 1, f"Not enough discrete action variety: {unique_discrete}"
    
    continuous_range = max(continuous_actions) - min(continuous_actions)
    assert continuous_range > 0.01, f"Not enough continuous action variety: {continuous_range}"
    
    print("✓ Epsilon-greedy action selection works correctly")

def test_expert_hybrid_integration():
    """Test expert system integration with hybrid actions"""
    print("\n=== Testing Expert Hybrid Integration ===")
    
    # Test various game scenarios
    test_scenarios = [
        (8, 0, False, False, False),   # Enemy at segment 8, player at 0, closed level
        (0, 8, True, True, False),     # Enemy at 0, player at 8, open level, fire
        (-1, 5, False, False, True),   # No enemy (-1), player at 5, zap
        (3, 3, False, True, True),     # Same position, fire + zap
    ]
    
    for enemy_seg, player_seg, is_open, expert_fire, expert_zap in test_scenarios:
        discrete_action, continuous_spinner = get_expert_hybrid_action(
            enemy_seg, player_seg, is_open, expert_fire, expert_zap
        )
        
        # Verify output types and ranges
        assert isinstance(discrete_action, int), f"Discrete action type: {type(discrete_action)}"
        assert isinstance(continuous_spinner, float), f"Continuous spinner type: {type(continuous_spinner)}"
        assert 0 <= discrete_action < 4, f"Discrete action out of range: {discrete_action}"
        assert -0.9 <= continuous_spinner <= 0.9, f"Continuous spinner out of range: {continuous_spinner}"
        
        # Verify fire/zap encoding
        fire, zap = discrete_to_fire_zap(discrete_action)
        assert fire == expert_fire, f"Fire mismatch: {fire} != {expert_fire}"
        assert zap == expert_zap, f"Zap mismatch: {zap} != {expert_zap}"
    
    print("✓ Expert hybrid action integration works correctly")

def test_game_action_conversion():
    """Test conversion to game action format"""
    print("\n=== Testing Game Action Conversion ===")
    
    test_cases = [
        (0, 0.0),      # No fire, no zap, center
        (2, -0.9),     # Fire, no zap, hard left
        (1, 0.5),      # No fire, zap, right
        (3, -0.4),     # Fire, zap, soft left
    ]
    
    for discrete_action, continuous_spinner in test_cases:
        fire_cmd, zap_cmd, spinner_cmd = hybrid_to_game_action(discrete_action, continuous_spinner)
        
        # Verify output types
        assert isinstance(fire_cmd, int), f"Fire cmd type: {type(fire_cmd)}"
        assert isinstance(zap_cmd, int), f"Zap cmd type: {type(zap_cmd)}"
        assert isinstance(spinner_cmd, int), f"Spinner cmd type: {type(spinner_cmd)}"
        
        # Verify ranges
        assert fire_cmd in [0, 1], f"Fire cmd out of range: {fire_cmd}"
        assert zap_cmd in [0, 1], f"Zap cmd out of range: {zap_cmd}"
        assert -32 <= spinner_cmd <= 31, f"Spinner cmd out of range: {spinner_cmd}"
        
        # Verify fire/zap encoding
        expected_fire, expected_zap = discrete_to_fire_zap(discrete_action)
        assert fire_cmd == expected_fire, f"Fire cmd mismatch: {fire_cmd} != {expected_fire}"
        assert zap_cmd == expected_zap, f"Zap cmd mismatch: {zap_cmd} != {expected_zap}"
        
        # Verify spinner scaling (exact with rounding and clamp)
        expected_spinner_cmd = int(round(continuous_spinner * 32))
        expected_spinner_cmd = max(-32, min(31, expected_spinner_cmd))
        assert spinner_cmd == expected_spinner_cmd, f"Spinner cmd mismatch: {spinner_cmd} != {expected_spinner_cmd}"
    
    print("✓ Game action conversion works correctly")

@patch('aimodel.training_device', torch.device('cpu'))
@patch('aimodel.inference_device', torch.device('cpu'))
def test_hybrid_training_loop():
    """Test hybrid agent training step"""
    print("\n=== Testing Hybrid Training Loop ===")
    
    state_size = 175
    agent = HybridDQNAgent(state_size=state_size, discrete_actions=4, batch_size=16)
    
    # Add training experiences
    for i in range(50):  # Need enough for a batch
        state = np.random.randn(state_size).astype(np.float32)
        discrete_action = np.random.randint(0, 4)
        continuous_action = np.random.uniform(-0.9, 0.9)
        log_prob = np.random.uniform(-2.0, 0.0)  # Reasonable log probability range
        reward = np.random.randn()
        next_state = np.random.randn(state_size).astype(np.float32)
        done = np.random.random() < 0.1
        
        agent.step(state, discrete_action, continuous_action, log_prob, reward, next_state, done)
    
    # Perform training step
    loss = agent.train_step()
    
    # Verify training occurred
    assert isinstance(loss, float), f"Loss type: {type(loss)}"
    assert loss >= 0, f"Loss should be non-negative: {loss}"
    assert agent.training_steps > 0, f"Training steps not incremented: {agent.training_steps}"
    
    print(f"✓ Training step completed with loss: {loss:.4f}")

def test_model_save_load():
    """Test hybrid model save/load functionality"""
    print("\n=== Testing Model Save/Load ===")
    
    state_size = 175
    agent1 = HybridDQNAgent(state_size=state_size, discrete_actions=4)
    
    # Create temporary save path
    save_path = "/tmp/test_hybrid_model.pt"
    
    try:
        # Save model
        agent1.save(save_path)
        assert os.path.exists(save_path), "Model file was not created"
        
        # Create new agent and load model
        agent2 = HybridDQNAgent(state_size=state_size, discrete_actions=4)
        success = agent2.load(save_path)
        
        assert success, "Model loading failed"
        
        # Test that models produce similar outputs
        state = np.random.randn(state_size).astype(np.float32)
        action1_discrete, action1_continuous = agent1.act(state, epsilon=0.0, add_noise=False)
        action2_discrete, action2_continuous = agent2.act(state, epsilon=0.0, add_noise=False)
        
        assert action1_discrete == action2_discrete, f"Discrete actions differ: {action1_discrete} != {action2_discrete}"
        assert abs(action1_continuous - action2_continuous) < 1e-5, f"Continuous actions differ: {action1_continuous} != {action2_continuous}"
        
        print("✓ Model save/load works correctly")
        
    finally:
        # Clean up
        if os.path.exists(save_path):
            os.remove(save_path)

def run_all_tests():
    """Run all hybrid architecture tests"""
    print("🚀 Starting Hybrid Architecture Test Suite\n")
    
    try:
        test_fire_zap_conversion()
        test_hybrid_dqn_network()
        test_hybrid_replay_buffer()
        test_hybrid_agent_action_selection()
        test_expert_hybrid_integration()
        test_game_action_conversion()
        test_hybrid_training_loop()
        test_model_save_load()
        
        print("\n🎉 All tests passed! Hybrid architecture is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)