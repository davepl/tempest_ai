#!/usr/bin/env python3
"""
Simple profiling script to test train_step performance directly.
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

import torch
import torch.nn as nn
import numpy as np
from Scripts.aimodel import HybridDQNAgent
from Scripts.config import RL_CONFIG, metrics

# Initialize metrics
metrics.frame_count = 0
metrics.training_enabled = True

def create_test_agent():
    """Create a test agent with minimal setup."""
    agent = HybridDQNAgent(
        state_size=100,  # Small state for testing
        discrete_actions=4
    )
    return agent

def create_test_batch(batch_size=64, state_size=100):
    """Create a test batch of training data."""
    # Create single experience, not a batch
    state = np.random.randn(state_size).astype(np.float32)
    discrete_action = np.random.randint(0, 4)
    reward = np.random.randn()
    next_state = np.random.randn(state_size).astype(np.float32)
    done = np.random.randint(0, 2)
    actor = 'dqn' if np.random.random() < 0.5 else 'expert'
    horizon = 1

    return (state, discrete_action, reward, next_state, done, actor, horizon)

def profile_train_step(agent, num_steps=100):
    """Profile the train_step method."""
    print(f"Profiling {num_steps} train_step calls...")

    # Fill memory with test data
    for _ in range(1000):
        batch = create_test_batch()
        agent.memory.push(*batch)

    # Stop background threads for direct profiling
    agent.stop(join=True, timeout=1.0)

    # Run profiling
    import time
    start_time = time.time()

    for i in range(num_steps):
        loss = agent.train_step()
        if (i + 1) % 10 == 0:
            print(f"Completed {i+1}/{num_steps} steps, training_steps={agent.training_steps}")
            import sys
            sys.stdout.flush()

    total_time = time.time() - start_time
    steps_per_sec = num_steps / total_time

    print(".2f")
    print(".4f")

if __name__ == "__main__":
    agent = create_test_agent()
    profile_train_step(agent, num_steps=100)
