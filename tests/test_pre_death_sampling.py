#!/usr/bin/env python3
"""Tests for pre-death priority boosting in the replay buffer."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Scripts'))

from replay_buffer import PrioritizedReplayBuffer  # type: ignore
from config import RL_CONFIG, SERVER_CONFIG  # type: ignore


def test_boost_priorities_increases_sampling_probability():
    """Transitions whose priorities are boosted should be sampled more often."""
    capacity = 2048
    state_size = SERVER_CONFIG.params_count
    buf = PrioritizedReplayBuffer(capacity=capacity, state_size=state_size, alpha=0.6)

    sample_state = np.zeros(state_size, dtype=np.float32)
    n_transitions = 500

    # Fill buffer with uniform transitions
    for i in range(n_transitions):
        buf.add(sample_state, 0, 0.0, sample_state, False)

    # Boost the last 50 transitions (simulating pre-death window)
    boost_indices = list(range(n_transitions - 50, n_transitions))
    buf.boost_priorities(boost_indices, multiplier=5.0)

    # Sample many batches and count how often boosted indices appear
    boosted_set = set(boost_indices)
    boosted_hits = 0
    total_samples = 0
    for _ in range(100):
        batch = buf.sample(64, beta=0.4)
        if batch is None:
            continue
        indices = batch[7]  # indices are at position 7
        for idx in indices:
            total_samples += 1
            if int(idx) in boosted_set:
                boosted_hits += 1

    # Boosted transitions are 10% of the buffer (50/500).
    # With 5× priority boost, they should appear much more than 10% of the time.
    boosted_fraction = boosted_hits / max(1, total_samples)
    assert boosted_fraction > 0.15, (
        f"Boosted transitions only appeared {boosted_fraction:.1%} of the time "
        f"(expected >15% with 5× boost on 10% of buffer)"
    )


def test_boost_priorities_noop_when_multiplier_is_one():
    """Boost with multiplier <= 1.0 should be a no-op."""
    capacity = 256
    state_size = SERVER_CONFIG.params_count
    buf = PrioritizedReplayBuffer(capacity=capacity, state_size=state_size, alpha=0.6)

    sample_state = np.zeros(state_size, dtype=np.float32)
    for i in range(100):
        buf.add(sample_state, 0, 0.0, sample_state, False)

    # Get priority before
    before = buf.tree.tree[buf.tree.capacity + 50]
    buf.boost_priorities([50], multiplier=1.0)
    after = buf.tree.tree[buf.tree.capacity + 50]
    assert before == after, "multiplier=1.0 should not change priorities"
