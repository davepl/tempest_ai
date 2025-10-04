#!/usr/bin/env python3
"""
Quick diagnostic test for N-Step buffer with realistic scenarios.
Focus on potential issues that could cause Q-value inflation.
"""
import os, sys
sys.path.append(os.path.dirname(__file__))

import numpy as np
from Scripts.nstep_buffer import NStepReplayBuffer


def test_realistic_tempest_scenario():
    """Test with realistic Tempest-like rewards and n_step=7."""
    print("Testing realistic Tempest scenario (n_step=7, gamma=0.99)...")
    
    buffer = NStepReplayBuffer(n_step=7, gamma=0.99)
    
    # Simulate typical Tempest rewards (small values)
    states = [np.random.randn(175).astype(np.float32) for _ in range(20)]
    rewards = [0.01, -0.005, 0.0, 0.02, -0.01, 0.015, 0.0, 0.05, -0.02, 0.0]
    
    experiences = []
    
    # Build up buffer
    for i in range(10):
        out = buffer.add(states[i], i % 18, rewards[i], states[i+1], False)
        experiences.extend(out)
        if len(out) > 0:
            print(f"Step {i}: emitted {len(out)} experiences")
            for j, (s, a, r, r_subj, r_obj, ns, d) in enumerate(out):
                print(f"  Exp {j}: action={a}, n_step_return={r:.6f}, subj={r_subj:.6f}, obj={r_obj:.6f}")
    
    print(f"Total experiences generated: {len(experiences)}")
    
    # Check for reasonable return magnification
    max_single_reward = max(abs(r) for r in rewards)
    max_nstep_return = max(abs(exp[2]) for exp in experiences) if experiences else 0
    
    print(f"Max single reward: {max_single_reward:.6f}")
    print(f"Max n-step return: {max_nstep_return:.6f}")
    print(f"Magnification factor: {max_nstep_return/max_single_reward:.2f}")
    
    # With gamma=0.99 and n=7, max theoretical magnification is sum(0.99^i for i in range(7)) ≈ 6.8
    expected_max_magnification = sum(0.99**i for i in range(7))
    print(f"Expected max magnification: {expected_max_magnification:.2f}")
    
    if max_nstep_return > max_single_reward * expected_max_magnification * 1.1:
        print("❌ WARNING: N-step returns seem inflated!")
        return False
    else:
        print("✓ N-step return magnification looks reasonable")
        return True


def test_episode_end_behavior():
    """Test behavior at episode boundaries that could cause issues."""
    print("\nTesting episode boundary behavior...")
    
    buffer = NStepReplayBuffer(n_step=7, gamma=0.99)
    
    # Simulate episode that ends after just a few steps
    states = [np.random.randn(175).astype(np.float32) for _ in range(5)]
    
    # Add 3 steps then terminal
    buffer.add(states[0], 5, 1.0, states[1], False)
    buffer.add(states[1], 10, 2.0, states[2], False) 
    experiences = buffer.add(states[2], 15, 100.0, states[3], True)  # Large terminal reward
    
    print(f"Episode end produced {len(experiences)} experiences")
    
    for i, (s, a, r, r_subj, r_obj, ns, d) in enumerate(experiences):
        print(f"  Exp {i}: action={a}, return={r:.3f}, subj={r_subj:.3f}, obj={r_obj:.3f}, done={d}")
    
    # Check that all are marked as done
    all_done = all(exp[6] for exp in experiences)
    if not all_done:
        print("❌ WARNING: Some experiences not marked as done!")
        return False
    
    # Check return calculations
    expected_returns = [
        1.0 + 0.99*2.0 + 0.99**2*100.0,  # First step gets 3-step return
        2.0 + 0.99*100.0,                # Second step gets 2-step return  
        100.0                            # Terminal step gets 1-step return
    ]
    
    for i, (expected, (_, _, actual, _, _, _, _)) in enumerate(zip(expected_returns, experiences)):
        if abs(actual - expected) > 1e-6:
            print(f"❌ WARNING: Experience {i} return mismatch! Expected {expected:.3f}, got {actual:.3f}")
            return False
    
    print("✓ Episode boundary behavior correct")
    return True


def test_training_volume():
    """Test training data volume - are we getting the right number of experiences?"""
    print("\nTesting training data volume...")
    
    buffer = NStepReplayBuffer(n_step=7, gamma=0.99)
    
    total_experiences = 0
    steps_taken = 0
    
    # Simulate 100 steps of continuous play
    states = [np.random.randn(175).astype(np.float32) for _ in range(101)]
    
    for i in range(100):
        reward = np.random.normal(0, 0.01)  # Small random rewards
        out = buffer.add(states[i], i % 18, reward, states[i+1], False)
        total_experiences += len(out)
        steps_taken += 1
        
        # After the first n_step steps, we should get exactly 1 experience per step
        if i >= 7:  # n_step = 7
            if len(out) != 1:
                print(f"❌ WARNING: At step {i}, expected 1 experience, got {len(out)}")
                return False
    
    print(f"After {steps_taken} steps: {total_experiences} experiences generated")
    expected_experiences = steps_taken - 7 + 1  # Should be steps - n_step + 1
    print(f"Expected: {expected_experiences} experiences")
    
    if total_experiences != expected_experiences:
        print(f"❌ WARNING: Experience count mismatch!")
        return False
    
    print("✓ Training volume looks correct")
    return True


def main():
    """Run all diagnostic tests."""
    print("=== N-Step Buffer Diagnostic ===\n")
    
    tests = [
        test_realistic_tempest_scenario,
        test_episode_end_behavior, 
        test_training_volume
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"❌ {test.__name__} failed with error: {e}")
            results.append(False)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\n=== Results: {passed}/{total} tests passed ===")
    
    if passed == total:
        print("🎉 N-Step buffer appears to be working correctly!")
        print("The Q-value growth you're seeing might be due to other factors.")
    else:
        print("❌ Found issues with N-Step buffer that could explain the problems.")
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)