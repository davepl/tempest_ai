#!/usr/bin/env python3
"""Replay sampling tests for the current PER implementation."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Scripts"))

from replay_buffer import PrioritizedReplayBuffer  # type: ignore  # pylint: disable=import-error


def test_priority_hint_biases_sampling():
    np.random.seed(7)
    capacity = 512
    state_size = 16
    buffer = PrioritizedReplayBuffer(capacity, state_size, alpha=0.6)

    state = np.zeros((state_size,), dtype=np.float32)
    high_priority_indices = set()
    high_count = 24

    # Add a small high-priority cohort and a large low-priority cohort.
    for i in range(capacity):
        hint = 25.0 if i < high_count else 0.05
        buffer.add(state, i % 4, 0.0, state, False, horizon=1, expert=0, priority_hint=hint)
        if i < high_count:
            high_priority_indices.add(i)

    draws = 300
    batch_size = 64
    sampled_high = 0
    sampled_total = 0

    for _ in range(draws):
        batch = buffer.sample(batch_size, beta=0.4)
        assert batch is not None
        indices = batch[7]
        sampled_total += len(indices)
        sampled_high += int(np.isin(indices, list(high_priority_indices)).sum())

    observed_fraction = sampled_high / max(1, sampled_total)
    baseline_fraction = high_count / capacity

    # Expect material over-sampling of hinted transitions vs uniform chance.
    assert observed_fraction > baseline_fraction * 3.0
