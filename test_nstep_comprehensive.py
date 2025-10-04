#!/usr/bin/env python3
"""
Comprehensive correctness tests for NStepReplayBuffer.
Tests edge cases, mathematical correctness, and integration scenarios.
"""
import os, sys
sys.path.append(os.path.dirname(__file__))

import numpy as np
from Scripts.nstep_buffer import NStepReplayBuffer


def assert_close(actual, expected, eps=1e-6, msg=""):
    assert abs(actual - expected) <= eps, f"{msg}: expected {expected}, got {actual}"


def test_n_step_1_equivalent_to_standard():
    """n_step=1 should behave exactly like standard 1-step returns."""
    print("Testing n_step=1 equivalence...")
    
    buffer = NStepReplayBuffer(n_step=1, gamma=0.9)
    
    s0, s1 = np.array([0.0]), np.array([1.0])
    
    # Add one step
    out = buffer.add(s0, 5, 10.0, s1, False)
    assert len(out) == 1
    s, a, r, r_subj, r_obj, ns, d = out[0]
    assert a == 5 and r == 10.0 and d == False
    assert_close(r_subj, r)
    assert_close(r_obj, r)
    np.testing.assert_array_equal(s, s0)
    np.testing.assert_array_equal(ns, s1)
    
    # Add terminal step
    s2 = np.array([2.0])
    out = buffer.add(s1, 3, 5.0, s2, True)
    assert len(out) == 1
    s, a, r, r_subj, r_obj, ns, d = out[0]
    assert a == 3 and r == 5.0 and d == True
    assert_close(r_subj, r)
    assert_close(r_obj, r)
    np.testing.assert_array_equal(s, s1)
    np.testing.assert_array_equal(ns, s2)


def test_mathematical_correctness():
    """Test exact n-step return calculations."""
    print("Testing mathematical correctness...")
    
    buffer = NStepReplayBuffer(n_step=3, gamma=0.9)
    
    states = [np.array([float(i)]) for i in range(10)]
    
    # Build up to 3 steps
    buffer.add(states[0], 1, 1.0, states[1], False)  # No output yet
    buffer.add(states[1], 2, 2.0, states[2], False)  # No output yet
    
    # Third step should emit first n-step return
    out = buffer.add(states[2], 3, 3.0, states[3], False)
    assert len(out) == 1
    
    s, a, R, R_subj, R_obj, ns, d = out[0]
    expected_R = 1.0 + 0.9*2.0 + 0.9*0.9*3.0  # 1 + 1.8 + 2.43 = 5.23
    assert_close(R, expected_R, msg="3-step return calculation")
    assert a == 1  # Action from first step
    np.testing.assert_array_equal(s, states[0])  # State from first step
    np.testing.assert_array_equal(ns, states[3])  # Next state from third step
    assert d == False
    assert_close(R_subj, R)
    assert_close(R_obj, R)
    
    # Fourth step should emit second n-step return
    out = buffer.add(states[3], 4, 4.0, states[4], False)
    assert len(out) == 1
    
    s, a, R, R_subj, R_obj, ns, d = out[0]
    expected_R = 2.0 + 0.9*3.0 + 0.9*0.9*4.0  # 2 + 2.7 + 3.24 = 7.94
    assert_close(R, expected_R, msg="Second 3-step return")
    assert a == 2
    assert_close(R_subj, R)
    assert_close(R_obj, R)


def test_terminal_within_window():
    """Test proper handling when episode ends before n steps."""
    print("Testing terminal within window...")
    
    buffer = NStepReplayBuffer(n_step=5, gamma=0.9)
    
    states = [np.array([float(i)]) for i in range(5)]
    
    # Add 3 steps, then terminal
    buffer.add(states[0], 1, 1.0, states[1], False)
    buffer.add(states[1], 2, 2.0, states[2], False)
    out = buffer.add(states[2], 3, 3.0, states[3], True)  # Terminal!
    
    # Should flush all 3 transitions
    assert len(out) == 3
    
    # First transition: gets 3-step return (but truncated at terminal)
    s, a, R, R_subj, R_obj, ns, d = out[0]
    expected_R = 1.0 + 0.9*2.0 + 0.9*0.9*3.0  # Full 3-step
    assert_close(R, expected_R, msg="First transition n-step return")
    assert a == 1 and d == True  # Should be marked done
    np.testing.assert_array_equal(ns, states[3])  # Terminal next state
    assert_close(R_subj, R)
    assert_close(R_obj, R)
    
    # Second transition: gets 2-step return
    s, a, R, R_subj, R_obj, ns, d = out[1]
    expected_R = 2.0 + 0.9*3.0  # 2-step return
    assert_close(R, expected_R, msg="Second transition 2-step return")
    assert a == 2 and d == True
    assert_close(R_subj, R)
    assert_close(R_obj, R)
    
    # Third transition: gets 1-step return
    s, a, R, R_subj, R_obj, ns, d = out[2]
    expected_R = 3.0  # Just immediate reward
    assert_close(R, expected_R, msg="Third transition 1-step return")
    assert a == 3 and d == True
    assert_close(R_subj, R)
    assert_close(R_obj, R)


def test_episode_boundary_isolation():
    """Test that episodes don't leak into each other."""
    print("Testing episode boundary isolation...")
    
    buffer = NStepReplayBuffer(n_step=3, gamma=0.9)
    
    # Episode 1: Add 2 steps then terminal
    s1_states = [np.array([1.0, float(i)]) for i in range(3)]
    buffer.add(s1_states[0], 1, 10.0, s1_states[1], False)
    out = buffer.add(s1_states[1], 2, 20.0, s1_states[2], True)
    
    # Should get 2 transitions from episode 1
    assert len(out) == 2
    
    # Start episode 2 immediately
    s2_states = [np.array([2.0, float(i)]) for i in range(4)]
    out = buffer.add(s2_states[0], 10, 100.0, s2_states[1], False)
    assert len(out) == 0  # No output yet
    
    out = buffer.add(s2_states[1], 11, 200.0, s2_states[2], False)
    assert len(out) == 0  # Still no output
    
    # Third step of episode 2 should emit first n-step return
    out = buffer.add(s2_states[2], 12, 300.0, s2_states[3], False)
    assert len(out) == 1
    
    s, a, R, R_subj, R_obj, ns, d = out[0]
    # Should be pure episode 2 data, no contamination from episode 1
    expected_R = 100.0 + 0.9*200.0 + 0.9*0.9*300.0
    assert_close(R, expected_R, msg="Episode 2 n-step return")
    assert a == 10  # First action of episode 2
    np.testing.assert_array_equal(s, s2_states[0])
    assert d == False
    assert_close(R_subj, R)
    assert_close(R_obj, R)


def test_varying_gamma_values():
    """Test different gamma values for correctness."""
    print("Testing various gamma values...")
    
    # Test gamma = 0 (no discounting beyond immediate)
    buffer = NStepReplayBuffer(n_step=3, gamma=0.0)
    states = [np.array([float(i)]) for i in range(4)]
    
    buffer.add(states[0], 1, 5.0, states[1], False)
    buffer.add(states[1], 2, 10.0, states[2], False)
    out = buffer.add(states[2], 3, 15.0, states[3], False)
    
    s, a, R, R_subj, R_obj, ns, d = out[0]
    assert_close(R, 5.0, msg="gamma=0 should only use immediate reward")
    assert_close(R_subj, R)
    assert_close(R_obj, R)
    
    # Test gamma = 1 (no discounting)
    buffer = NStepReplayBuffer(n_step=3, gamma=1.0)
    buffer.add(states[0], 1, 5.0, states[1], False)
    buffer.add(states[1], 2, 10.0, states[2], False)
    out = buffer.add(states[2], 3, 15.0, states[3], False)
    
    s, a, R, R_subj, R_obj, ns, d = out[0]
    assert_close(R, 30.0, msg="gamma=1 should sum all rewards")  # 5 + 10 + 15
    assert_close(R_subj, R)
    assert_close(R_obj, R)


def test_large_n_step():
    """Test larger n-step values."""
    print("Testing large n-step values...")
    
    buffer = NStepReplayBuffer(n_step=10, gamma=0.95)
    states = [np.array([float(i)]) for i in range(20)]
    
    # Add 9 steps (no output yet)
    for i in range(9):
        out = buffer.add(states[i], i, float(i+1), states[i+1], False)
        assert len(out) == 0, f"Should have no output at step {i}"
    
    # 10th step should produce output
    out = buffer.add(states[9], 9, 10.0, states[10], False)
    assert len(out) == 1
    
    s, a, R, R_subj, R_obj, ns, d = out[0]
    # Manual calculation for 10-step return
    expected_R = sum((0.95 ** i) * (i + 1) for i in range(10))
    assert_close(R, expected_R, msg="10-step return calculation")
    assert a == 0  # First action
    np.testing.assert_array_equal(ns, states[10])  # 10th next state
    assert_close(R_subj, R)
    assert_close(R_obj, R)


def test_reset_functionality():
    """Test buffer reset clears state properly."""
    print("Testing reset functionality...")
    
    buffer = NStepReplayBuffer(n_step=3, gamma=0.9)
    states = [np.array([float(i)]) for i in range(5)]
    
    # Add some transitions
    buffer.add(states[0], 1, 1.0, states[1], False)
    buffer.add(states[1], 2, 2.0, states[2], False)
    
    # Reset should clear everything
    buffer.reset()
    
    # Now add new transitions - should start fresh
    out = buffer.add(states[2], 10, 10.0, states[3], False)
    assert len(out) == 0, "After reset, should need n steps before output"
    
    out = buffer.add(states[3], 11, 11.0, states[4], False)
    assert len(out) == 0, "Still should need one more step"
    
    # Third step after reset
    out = buffer.add(states[4], 12, 12.0, states[0], False)
    assert len(out) == 1
    
    s, a, R, R_subj, R_obj, ns, d = out[0]
    # Should be based on post-reset data only
    expected_R = 10.0 + 0.9*11.0 + 0.9*0.9*12.0
    assert_close(R, expected_R, msg="Post-reset n-step calculation")
    assert a == 10  # First action after reset
    assert_close(R_subj, R)
    assert_close(R_obj, R)


def test_state_shape_preservation():
    """Test that state arrays maintain their shapes and dtypes."""
    print("Testing state shape preservation...")
    
    buffer = NStepReplayBuffer(n_step=2, gamma=0.9)
    
    # Use different shaped states
    state_2d = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    state_1d = np.array([5.0, 6.0], dtype=np.float32)
    state_scalar = np.array([7.0], dtype=np.float32)
    
    buffer.add(state_2d, 1, 1.0, state_1d, False)
    out = buffer.add(state_1d, 2, 2.0, state_scalar, False)
    
    assert len(out) == 1
    s, a, R, R_subj, R_obj, ns, d = out[0]
    
    # Check shapes and dtypes preserved
    np.testing.assert_array_equal(s, state_2d)
    np.testing.assert_array_equal(ns, state_scalar)
    assert s.shape == (2, 2)
    assert ns.shape == (1,)
    assert s.dtype == np.float32
    assert ns.dtype == np.float32
    assert_close(R_subj, R)
    assert_close(R_obj, R)


def run_all_tests():
    """Run all tests and report results."""
    test_functions = [
        test_n_step_1_equivalent_to_standard,
        test_mathematical_correctness,
        test_terminal_within_window,
        test_episode_boundary_isolation,
        test_varying_gamma_values,
        test_large_n_step,
        test_reset_functionality,
        test_state_shape_preservation,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            test_func()
            print(f"✓ {test_func.__name__}")
            passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__}: {e}")
            failed += 1
    
    print(f"\nTest Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("🎉 All tests passed! NStepReplayBuffer appears mathematically correct.")
    else:
        print("❌ Some tests failed. Review the implementation.")
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)