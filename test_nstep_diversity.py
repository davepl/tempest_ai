#!/usr/bin/env python3
"""
Test cases for diversity bonus features.
Tests for n-step returns removed since add_trajectory method was removed
(n-step is now handled by NStepReplayBuffer in socket_server.py).
"""

import sys
import os
import numpy as np

# Add Scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Scripts'))

from aimodel import HybridDQNAgent


def test_diversity_bonus_novel_action():
    """Test that diversity bonus rewards novel actions"""
    print("\n=== Testing Diversity Bonus for Novel Actions ===")
    
    # Create agent with diversity bonus enabled
    agent = HybridDQNAgent(
        state_size=10,
        discrete_actions=4,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=0.1,
        memory_size=1000,
        batch_size=32
    )
    
    # Ensure diversity bonus is enabled
    agent.diversity_bonus_enabled = True
    agent.diversity_bonus_weight = 1.0  # Use 1.0 for easier verification
    
    # Create a state
    state = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    
    # First action in this state should get bonus
    bonus1 = agent.calculate_diversity_bonus(state, discrete_action=0, continuous_action=0.5)
    print(f"First action bonus: {bonus1:.4f}")
    
    # Same action again should get no bonus
    bonus2 = agent.calculate_diversity_bonus(state, discrete_action=0, continuous_action=0.5)
    print(f"Repeated action bonus: {bonus2:.4f}")
    
    # Different action should get bonus (decayed by sqrt(2))
    bonus3 = agent.calculate_diversity_bonus(state, discrete_action=1, continuous_action=0.5)
    print(f"Second novel action bonus: {bonus3:.4f}")
    expected_bonus3 = 1.0 / np.sqrt(2)
    print(f"Expected second bonus: {expected_bonus3:.4f}")
    
    # Verify bonuses
    success = True
    if bonus1 <= 0:
        print("✗ First action should get positive bonus")
        success = False
    if bonus2 != 0:
        print("✗ Repeated action should get zero bonus")
        success = False
    if abs(bonus3 - expected_bonus3) > 0.01:
        print(f"✗ Second bonus incorrect (got {bonus3:.4f}, expected {expected_bonus3:.4f})")
        success = False
    
    if success:
        print("✓ Diversity bonus calculation PASSED")
    else:
        print("✗ Diversity bonus calculation FAILED")
    
    return success


def test_diversity_bonus_state_clustering():
    """Test that diversity bonus clusters similar states"""
    print("\n=== Testing Diversity Bonus State Clustering ===")
    
    agent = HybridDQNAgent(
        state_size=25,
        discrete_actions=4,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=0.1,
        memory_size=1000,
        batch_size=32
    )
    
    agent.diversity_bonus_enabled = True
    agent.diversity_bonus_weight = 1.0
    
    # Create two similar states (same when rounded to 0.1)
    state1 = np.array([1.02, 2.03, 3.01] + [0.0] * 22)
    state2 = np.array([1.04, 1.98, 2.99] + [0.0] * 22)  # Rounds to same values
    
    # Try action in state1
    bonus1 = agent.calculate_diversity_bonus(state1, discrete_action=0, continuous_action=0.5)
    print(f"Action in state1 bonus: {bonus1:.4f}")
    
    # Try same action in state2 (should be treated as repeated due to clustering)
    bonus2 = agent.calculate_diversity_bonus(state2, discrete_action=0, continuous_action=0.5)
    print(f"Same action in state2 (similar) bonus: {bonus2:.4f}")
    
    # Create a different state
    state3 = np.array([5.0, 6.0, 7.0] + [0.0] * 22)
    bonus3 = agent.calculate_diversity_bonus(state3, discrete_action=0, continuous_action=0.5)
    print(f"Action in different state3 bonus: {bonus3:.4f}")
    
    success = True
    if bonus1 <= 0:
        print("✗ First state should get bonus")
        success = False
    if bonus2 != 0:
        print("✗ Similar state should get zero bonus (clustering failed)")
        success = False
    if bonus3 <= 0:
        print("✗ Different state should get bonus")
        success = False
    
    if success:
        print("✓ State clustering PASSED")
    else:
        print("✗ State clustering FAILED")
    
    return success


def test_diversity_bonus_toggle():
    """Test that diversity bonus can be toggled on/off"""
    print("\n=== Testing Diversity Bonus Toggle ===")
    
    agent = HybridDQNAgent(
        state_size=10,
        discrete_actions=4,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=0.1,
        memory_size=1000,
        batch_size=32
    )
    
    state = np.array([1.0] * 10)
    
    # Enable and test
    agent.set_diversity_bonus_enabled(True)
    bonus_on = agent.calculate_diversity_bonus(state, discrete_action=0, continuous_action=0.5)
    print(f"Bonus when enabled: {bonus_on:.4f}")
    
    # Clear history to test fresh action
    agent.action_history.clear()
    
    # Disable and test
    agent.set_diversity_bonus_enabled(False)
    bonus_off = agent.calculate_diversity_bonus(state, discrete_action=0, continuous_action=0.5)
    print(f"Bonus when disabled: {bonus_off:.4f}")
    
    success = True
    if bonus_on <= 0:
        print("✗ Bonus should be positive when enabled")
        success = False
    if bonus_off != 0:
        print("✗ Bonus should be zero when disabled")
        success = False
    
    if success:
        print("✓ Toggle functionality PASSED")
    else:
        print("✗ Toggle functionality FAILED")
    
    return success


def test_n_step_toggle():
    """Test that n-step learning can be toggled"""
    print("\n=== Testing N-Step Toggle ===")
    
    agent = HybridDQNAgent(
        state_size=10,
        discrete_actions=4,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=0.1,
        memory_size=1000,
        batch_size=32
    )
    
    # Test initial state
    print(f"Initial n-step enabled: {agent.n_step_enabled}")
    
    # Toggle off
    agent.set_n_step_enabled(False)
    print(f"After toggle off: {agent.n_step_enabled}")
    
    # Toggle on
    agent.set_n_step_enabled(True)
    print(f"After toggle on: {agent.n_step_enabled}")
    
    if agent.n_step_enabled == True:
        print("✓ N-step toggle PASSED")
        return True
    else:
        print("✗ N-step toggle FAILED")
        return False


def run_all_tests():
    """Run all test cases"""
    print("=" * 60)
    print("Running Diversity Bonus Test Suite")
    print("(N-step tests removed - handled by NStepReplayBuffer)")
    print("=" * 60)
    
    tests = [
        ("Diversity Bonus Novel Actions", test_diversity_bonus_novel_action),
        ("Diversity Bonus State Clustering", test_diversity_bonus_state_clustering),
        ("Diversity Bonus Toggle", test_diversity_bonus_toggle),
        ("N-Step Toggle", test_n_step_toggle),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Test '{name}' raised exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
