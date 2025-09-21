#!/usr/bin/env python3
"""
Unit tests for NStepReplayBuffer preprocessor.
Validates:
- Normal n-step emission after at least n transitions
- Terminal within window shortens return and sets done
- Terminal flush emits remaining tail transitions and clears buffer
"""
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(__file__))

from Scripts.nstep_buffer import NStepReplayBuffer


def close(a, b, eps=1e-6):
    return abs(a - b) <= eps


def test_normal_flow():
    n, gamma = 3, 0.9
    nbuf = NStepReplayBuffer(n_step=n, gamma=gamma)

    s0 = np.array([0.0], dtype=np.float32)
    s1 = np.array([1.0], dtype=np.float32)
    s2 = np.array([2.0], dtype=np.float32)
    s3 = np.array([3.0], dtype=np.float32)

    # step 0
    out = nbuf.add(s0, 1, 1.0, s1, False)
    assert out == []
    # step 1
    out = nbuf.add(s1, 2, 2.0, s2, False)
    assert out == []
    # step 2 (now we have 3)
    out = nbuf.add(s2, 3, 3.0, s3, False)
    assert len(out) == 1
    s, a, R, ns, d = out[0]
    assert a == 1 and ns is s3 and d is False
    # R = 1 + 0.9*2 + 0.9^2*3 = 1 + 1.8 + 2.43 = 5.23
    assert close(R, 5.23)

    # Next add adds one more and pops left
    s4 = np.array([4.0], dtype=np.float32)
    out = nbuf.add(s3, 4, 4.0, s4, False)
    assert len(out) == 1
    s, a, R, ns, d = out[0]
    assert a == 2 and ns is s4 and d is False
    # R = 2 + 0.9*3 + 0.9^2*4 = 2 + 2.7 + 3.24 = 7.94
    assert close(R, 7.94)


def test_done_inside_window():
    n, gamma = 3, 0.9
    nbuf = NStepReplayBuffer(n_step=n, gamma=gamma)

    s0 = np.array([0.0], dtype=np.float32)
    s1 = np.array([1.0], dtype=np.float32)
    s2 = np.array([2.0], dtype=np.float32)

    out = nbuf.add(s0, 1, 1.0, s1, False)
    assert out == []
    out = nbuf.add(s1, 2, 2.0, s2, True)
    # Terminal arrived before we had n items; should flush 2 outputs (from s0 and s1)
    assert len(out) == 2

    (s, a, R, ns, d) = out[0]
    assert a == 1 and d is True and ns is s2
    # R = 1 + 0.9*2 (episode ended at second transition)
    assert close(R, 2.8)

    (s, a, R, ns, d) = out[1]
    assert a == 2 and d is True and ns is s2
    # R = 2 (only immediate reward due to terminal)
    assert close(R, 2.0)


def test_flush_on_done_with_full_window():
    n, gamma = 3, 1.0
    nbuf = NStepReplayBuffer(n_step=n, gamma=gamma)

    s0 = np.array([0.0], dtype=np.float32)
    s1 = np.array([1.0], dtype=np.float32)
    s2 = np.array([2.0], dtype=np.float32)
    s3 = np.array([3.0], dtype=np.float32)

    nbuf.add(s0, 1, 1.0, s1, False)
    nbuf.add(s1, 2, 2.0, s2, False)
    # third add with done=True should emit: from s0 (1+2+3), from s1 (2+3), from s2 (3)
    out = nbuf.add(s2, 3, 3.0, s3, True)
    assert len(out) == 3

    a_vals = [a for (_, a, _, _, _) in out]
    assert a_vals == [1, 2, 3]

    R_vals = [R for (_, _, R, _, _) in out]
    assert R_vals == [6.0, 5.0, 3.0]


if __name__ == "__main__":
    test_normal_flow()
    test_done_inside_window()
    test_flush_on_done_with_full_window()
    print("NStepReplayBuffer tests passed")
