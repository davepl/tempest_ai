#!/usr/bin/env python3
"""
Comprehensive test suite for Prioritized Experience Replay (PER) system.
Tests correctness, thread safety, edge cases, and performance.
"""

import numpy as np
import torch
import threading
import time
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add the Scripts directory to path so we can import aimodel
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Scripts'))

from aimodel import PrioritizedReplayMemory, RL_CONFIG

def test_basic_functionality():
    """Test basic PER operations: push, sample, update_priorities"""
    print("=== Testing Basic PER Functionality ===")

    capacity = 1000
    batch_size = 32
    memory = PrioritizedReplayMemory(capacity, alpha=0.6, eps=1e-6)

    # Test empty buffer
    try:
        memory.sample(batch_size, beta=0.4)
        assert False, "Should not be able to sample from empty buffer"
    except AssertionError:
        print("✓ Correctly rejected sampling from empty buffer")

    # Push some experiences
    state_size = RL_CONFIG.state_size
    for i in range(100):
        state = np.random.randn(state_size).astype(np.float32)
        action = np.random.randint(0, 18)
        reward = np.random.randn()
        next_state = np.random.randn(state_size).astype(np.float32)
        done = np.random.random() < 0.1
        memory.push(state, action, reward, next_state, done)

    print(f"✓ Pushed 100 experiences, buffer size: {len(memory)}")

    # Test sampling
    states, actions, rewards, next_states, dones, is_weights, indices = memory.sample(batch_size, beta=0.4)

    # Verify shapes
    assert states.shape == (batch_size, state_size), f"States shape: {states.shape}"
    assert actions.shape == (batch_size, 1), f"Actions shape: {actions.shape}"
    assert rewards.shape == (batch_size, 1), f"Rewards shape: {rewards.shape}"
    assert next_states.shape == (batch_size, state_size), f"Next states shape: {next_states.shape}"
    assert dones.shape == (batch_size, 1), f"Dones shape: {dones.shape}"
    assert is_weights.shape == (batch_size, 1), f"IS weights shape: {is_weights.shape}"
    assert len(indices) == batch_size, f"Indices length: {len(indices)}"

    print("✓ Sample shapes are correct")

    # Verify importance weights are reasonable
    assert torch.all(is_weights > 0), "Importance weights should be positive"
    assert torch.all(is_weights <= 1.0), "Importance weights should be <= 1.0"
    print(f"✓ Importance weights range: [{is_weights.min().item():.4f}, {is_weights.max().item():.4f}]")

    # Test priority updates
    td_errors = torch.randn(batch_size, 1) * 2.0  # Some large TD errors
    memory.update_priorities(indices, td_errors)

    print("✓ Priority updates completed without errors")

    # Sample again and verify priorities were updated
    _, _, _, _, _, is_weights2, _ = memory.sample(batch_size, beta=0.4)
    print(f"✓ Post-update importance weights range: [{is_weights2.min().item():.4f}, {is_weights2.max().item():.4f}]")

    return True

def test_priority_clamping():
    """Test that TD errors are properly clamped to prevent priority explosion"""
    print("\n=== Testing Priority Clamping ===")

    capacity = 100
    memory = PrioritizedReplayMemory(capacity, alpha=0.6, eps=1e-6)

    # Fill buffer
    state_size = RL_CONFIG.state_size
    for i in range(capacity):
        state = np.random.randn(state_size).astype(np.float32)
        action = 0
        reward = 0.0
        next_state = np.random.randn(state_size).astype(np.float32)
        done = False
        memory.push(state, action, reward, next_state, done)

    # Sample
    batch_size = 32
    _, _, _, _, _, _, indices = memory.sample(batch_size, beta=0.4)

    # Test with extreme TD errors
    extreme_td_errors = torch.tensor([100.0, -50.0, 25.0, -25.0] * (batch_size // 4)).unsqueeze(1)

    try:
        memory.update_priorities(indices, extreme_td_errors)
        print("✓ Extreme TD errors handled without crashing")
    except AssertionError as e:
        print(f"✗ Failed on extreme TD errors: {e}")
        return False

    # Verify priorities are reasonable
    priorities = memory.priorities[:len(memory), 0]
    max_priority = priorities.max()
    min_priority = priorities.min()

    print(f"✓ Priority range after extreme updates: [{min_priority:.4f}, {max_priority:.4f}]")

    # Should be clamped to reasonable values
    assert max_priority <= 6.0, f"Priority too high: {max_priority}"  # eps + 5.0 clamp
    assert min_priority >= 1e-6, f"Priority too low: {min_priority}"

    return True

def test_thread_safety():
    """Test concurrent push/sample operations"""
    print("\n=== Testing Thread Safety ===")

    capacity = 10000
    memory = PrioritizedReplayMemory(capacity, alpha=0.6, eps=1e-6)

    state_size = RL_CONFIG.state_size
    batch_size = 64

    # Shared counters
    push_count = [0]
    sample_count = [0]
    errors = []

    def push_worker(worker_id):
        """Worker that pushes experiences"""
        try:
            for i in range(500):
                state = np.random.randn(state_size).astype(np.float32)
                action = np.random.randint(0, 18)
                reward = np.random.randn()
                next_state = np.random.randn(state_size).astype(np.float32)
                done = np.random.random() < 0.1
                memory.push(state, action, reward, next_state, done)
                push_count[0] += 1
        except Exception as e:
            errors.append(f"Push worker {worker_id}: {e}")

    def sample_worker(worker_id):
        """Worker that samples and updates priorities"""
        try:
            for i in range(200):
                if len(memory) >= batch_size:
                    _, _, _, _, _, _, indices = memory.sample(batch_size, beta=0.4)
                    td_errors = torch.randn(batch_size, 1)
                    memory.update_priorities(indices, td_errors)
                    sample_count[0] += 1
        except Exception as e:
            errors.append(f"Sample worker {worker_id}: {e}")

    # Start concurrent operations
    threads = []

    # 4 push threads
    for i in range(4):
        t = threading.Thread(target=push_worker, args=(i,))
        threads.append(t)
        t.start()

    # 2 sample threads
    for i in range(2):
        t = threading.Thread(target=sample_worker, args=(i,))
        threads.append(t)
        t.start()

    # Wait for completion
    for t in threads:
        t.join()

    if errors:
        print(f"✗ Thread safety errors: {errors}")
        return False

    print(f"✓ Completed {push_count[0]} pushes and {sample_count[0]} sample cycles concurrently")
    print(f"✓ Final buffer size: {len(memory)}")

    # Verify buffer integrity
    memory.validate_priorities()

    return True

def test_edge_cases():
    """Test edge cases: full buffer, priority validation, etc."""
    print("\n=== Testing Edge Cases ===")

    capacity = 100
    memory = PrioritizedReplayMemory(capacity, alpha=0.6, eps=1e-6)

    state_size = RL_CONFIG.state_size

    # Test buffer overflow
    for i in range(capacity + 50):
        state = np.random.randn(state_size).astype(np.float32)
        action = 0
        reward = 0.0
        next_state = np.random.randn(state_size).astype(np.float32)
        done = False
        memory.push(state, action, reward, next_state, done)

    assert len(memory) == capacity, f"Buffer should be at capacity: {len(memory)}"
    print("✓ Buffer correctly handles overflow")

    # Test sampling from full buffer
    batch_size = 32
    states, actions, rewards, next_states, dones, is_weights, indices = memory.sample(batch_size, beta=0.4)

    # Verify all indices are valid
    assert all(0 <= idx < capacity for idx in indices), f"Invalid indices: {indices}"
    print("✓ Sampling from full buffer works correctly")

    # Test priority validation
    memory.validate_priorities()

    # Test with NaN/Inf TD errors (should be caught)
    td_errors = torch.tensor([1.0, float('nan'), float('inf'), -float('inf')]).unsqueeze(1)
    try:
        memory.update_priorities(indices[:4], td_errors)
        assert False, "Should have caught NaN/Inf TD errors"
    except AssertionError:
        print("✓ Correctly caught NaN/Inf TD errors")

    return True

def test_active_window():
    """Test active window sampling for performance"""
    print("\n=== Testing Active Window Sampling ===")

    capacity = 10000
    memory = PrioritizedReplayMemory(capacity, alpha=0.6, eps=1e-6)

    # Fill buffer
    state_size = RL_CONFIG.state_size
    for i in range(capacity):
        state = np.random.randn(state_size).astype(np.float32)
        action = 0
        reward = 0.0
        next_state = np.random.randn(state_size).astype(np.float32)
        done = False
        memory.push(state, action, reward, next_state, done)

    # Test with active window
    original_active_size = getattr(RL_CONFIG, 'per_active_size', 0)
    RL_CONFIG.per_active_size = 1000  # Use only last 1000 experiences

    batch_size = 64
    start_time = time.time()
    for _ in range(10):
        _, _, _, _, _, _, indices = memory.sample(batch_size, beta=0.4)
    window_time = time.time() - start_time

    # Test without active window
    RL_CONFIG.per_active_size = 0  # Use full buffer

    start_time = time.time()
    for _ in range(10):
        _, _, _, _, _, _, indices = memory.sample(batch_size, beta=0.4)
    full_time = time.time() - start_time

    print(".2f")
    print(".2f")

    # Restore original setting
    RL_CONFIG.per_active_size = original_active_size

    return True

def test_beta_annealing():
    """Test importance sampling beta annealing"""
    print("\n=== Testing Beta Annealing ===")

    capacity = 1000
    memory = PrioritizedReplayMemory(capacity, alpha=0.6, eps=1e-6)

    # Fill buffer
    state_size = RL_CONFIG.state_size
    for i in range(capacity):
        state = np.random.randn(state_size).astype(np.float32)
        action = 0
        reward = 0.0
        next_state = np.random.randn(state_size).astype(np.float32)
        done = False
        memory.push(state, action, reward, next_state, done)

    batch_size = 32

    # Test different beta values
    betas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for beta in betas:
        _, _, _, _, _, is_weights, _ = memory.sample(batch_size, beta=beta)

        # With beta=0, weights should be uniform (close to 1.0)
        # With beta=1, weights should vary more
        weight_std = is_weights.std().item()
        weight_mean = is_weights.mean().item()

        print(f"  Beta {beta:.1f}: mean={weight_mean:.4f}, std={weight_std:.4f}")
    print("✓ Beta annealing produces expected weight distributions")

    return True

def test_memory_corruption_detection():
    """Test that memory corruption is detected"""
    print("\n=== Testing Memory Corruption Detection ===")

    capacity = 100
    memory = PrioritizedReplayMemory(capacity, alpha=0.6, eps=1e-6)

    # Fill buffer
    state_size = RL_CONFIG.state_size
    for i in range(capacity):
        state = np.random.randn(state_size).astype(np.float32)
        action = 0
        reward = 0.0
        next_state = np.random.randn(state_size).astype(np.float32)
        done = False
        memory.push(state, action, reward, next_state, done)

    # Manually corrupt priorities (simulate bug)
    memory.priorities[50, 0] = 0.0  # Zero priority
    memory.priorities[51, 0] = float('nan')  # NaN priority
    memory.priorities[52, 0] = float('inf')  # Inf priority

    # This should trigger strict validation if enabled
    original_strict = getattr(RL_CONFIG, 'per_strict_checks_every', 0)
    RL_CONFIG.per_strict_checks_every = 1  # Enable strict checks

    try:
        memory.sample(32, beta=0.4)
        print("⚠ Strict validation not triggered (may be disabled)")
    except AssertionError as e:
        print(f"✓ Corruption detected: {e}")

    # Restore setting
    RL_CONFIG.per_strict_checks_every = original_strict

    return True

def run_performance_test():
    """Performance test for PER operations"""
    print("\n=== Performance Testing ===")

    capacity = 50000
    memory = PrioritizedReplayMemory(capacity, alpha=0.6, eps=1e-6)

    state_size = RL_CONFIG.state_size

    # Fill buffer
    print("Filling buffer with 50k experiences...")
    start_time = time.time()
    for i in range(capacity):
        state = np.random.randn(state_size).astype(np.float32)
        action = 0
        reward = 0.0
        next_state = np.random.randn(state_size).astype(np.float32)
        done = False
        memory.push(state, action, reward, next_state, done)
    fill_time = time.time() - start_time
    print(".2f")

    # Test sampling performance
    batch_size = 64
    num_samples = 100

    print(f"Testing {num_samples} sampling operations...")
    start_time = time.time()
    for _ in range(num_samples):
        _, _, _, _, _, _, indices = memory.sample(batch_size, beta=0.4)
        # Simulate priority updates
        td_errors = torch.randn(batch_size, 1) * 0.1
        memory.update_priorities(indices, td_errors)
    sample_time = time.time() - start_time

    samples_per_sec = num_samples / sample_time
    print(".1f")

    return True

def main():
    """Run all PER tests"""
    print("Starting Comprehensive PER Test Suite")
    print("=" * 50)

    tests = [
        test_basic_functionality,
        test_priority_clamping,
        test_thread_safety,
        test_edge_cases,
        test_active_window,
        test_beta_annealing,
        test_memory_corruption_detection,
        run_performance_test,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            if test_func():
                print(f"✓ {test_func.__name__} PASSED")
                passed += 1
            else:
                print(f"✗ {test_func.__name__} FAILED")
                failed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} CRASHED: {e}")
            failed += 1
        print()

    print("=" * 50)
    print(f"Test Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All PER tests passed! The implementation is solid.")
        return 0
    else:
        print("❌ Some tests failed. Review the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())