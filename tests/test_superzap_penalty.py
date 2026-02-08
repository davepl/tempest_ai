#!/usr/bin/env python3
"""Regression tests for n-step rollout behavior across actor source changes."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Scripts"))

from nstep_buffer import NStepReplayBuffer  # type: ignore  # pylint: disable=import-error


def test_nstep_does_not_truncate_on_actor_switch():
    gamma = 0.99
    n = 5
    buf = NStepReplayBuffer(n_step=n, gamma=gamma)

    outs = []
    outs.extend(buf.add("s0", 0, 1.0, "s1", False, actor="dqn"))
    outs.extend(buf.add("s1", 1, 1.0, "s2", False, actor="expert"))
    outs.extend(buf.add("s2", 2, 1.0, "s3", False, actor="dqn"))
    outs.extend(buf.add("s3", 3, 1.0, "s4", False, actor="expert"))
    outs.extend(buf.add("s4", 4, 1.0, "s5", False, actor="dqn"))

    assert len(outs) >= 1
    first = outs[0]
    # Tuple layout: (s0, a0, Rn, pRn, sn, done, horizon, actor)
    assert first[6] == 5
    assert first[0] == "s0"
    assert first[4] == "s5"
