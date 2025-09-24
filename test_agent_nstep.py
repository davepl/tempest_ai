#!/usr/bin/env python3
"""
Test to verify DQNAgent is actually using n-step learning correctly.
"""
import os, sys
sys.path.append(os.path.dirname(__file__))

# Mock the config before importing aimodel
class MockRLConfig:
    n_step = 7
    gamma = 0.99
    state_size = 175
    lr = 0.0001
    epsilon = 0.1
    epsilon_min = 0.05
    memory_size = 10000
    batch_size = 32
    use_per = False
    use_mixed_precision = False
    use_torch_compile = False
    hidden_size = 512

# Replace config before import
import Scripts.config
Scripts.config.RL_CONFIG = MockRLConfig()

# Now we can import aimodel
from Scripts.aimodel import DQNAgent
import numpy as np
import torch

def test_agent_nstep_usage():
    """Test that DQNAgent uses n-step correctly."""
    print("Testing DQNAgent n-step integration...")
    
    # Create a minimal agent (will fail on GPU init but we just need to check n_step)
    try:
        # Mock the devices to CPU to avoid GPU requirements
        import Scripts.aimodel
        Scripts.aimodel.training_device = torch.device("cpu")
        Scripts.aimodel.inference_device = torch.device("cpu")
        
        agent = DQNAgent(state_size=175, action_size=18)
        
        # Check if n_step buffer was created
        print(f"Agent n_step setting: {agent.n_step}")
        print(f"N-step buffer created: {agent.n_step_buffer is not None}")
        
        if agent.n_step_buffer:
            print(f"Buffer n_step: {agent.n_step_buffer.n_step}")
            print(f"Buffer gamma: {agent.n_step_buffer.gamma}")
        
        # Test a few steps to see if experiences are generated correctly
        state = np.random.randn(175).astype(np.float32)
        next_state = np.random.randn(175).astype(np.float32)
        
        initial_memory_size = len(agent.memory)
        print(f"Initial memory size: {initial_memory_size}")
        
        # Add some steps - first 6 should add nothing to memory (buffer fills)
        for i in range(6):
            agent.step(state, i % 18, 0.01, next_state, False)
            print(f"After step {i+1}: memory size = {len(agent.memory)}")
        
        # 7th step should add 1 experience to memory
        agent.step(state, 7, 0.01, next_state, False)
        print(f"After step 7: memory size = {len(agent.memory)}")
        
        # 8th step should add 1 more
        agent.step(state, 8, 0.01, next_state, False)
        print(f"After step 8: memory size = {len(agent.memory)}")
        
        expected_memory_size = initial_memory_size + 2  # Should have 2 experiences after 8 steps
        if len(agent.memory) == expected_memory_size:
            print("✓ N-step buffer is working correctly in DQNAgent")
            return True
        else:
            print(f"❌ Expected {expected_memory_size} experiences, got {len(agent.memory)}")
            return False
            
    except Exception as e:
        print(f"Error testing agent: {e}")
        # Try to extract info about whether n_step was configured
        try:
            print(f"Config n_step: {Scripts.config.RL_CONFIG.n_step}")
        except:
            pass
        return False

if __name__ == '__main__':
    success = test_agent_nstep_usage()
    print(f"\nResult: {'PASS' if success else 'FAIL'}")