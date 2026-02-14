#!/usr/bin/env python3
"""Tests for pre-death sampling emphasis in the replay buffer."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Scripts'))

from aimodel import HybridReplayBuffer  # type: ignore  # pylint: disable=import-error
from config import RL_CONFIG, SERVER_CONFIG  # type: ignore  # pylint: disable=import-error


def test_pre_death_samples_are_biased_into_batches():
    capacity = 512
    buffer = HybridReplayBuffer(capacity, SERVER_CONFIG.params_count)
    original_frac = getattr(RL_CONFIG, 'pre_death_sample_fraction', 0.25)
    try:
        RL_CONFIG.pre_death_sample_fraction = 0.5
        sample_state = np.zeros((SERVER_CONFIG.params_count,), dtype=np.float32)
        episodes = 40
        steps_per_episode = 6
        for _ in range(episodes):
            for step in range(steps_per_episode):
                done = step == steps_per_episode - 1
                buffer.push(sample_state, 0, -1.0 if done else 0.0, sample_state, done, 'dqn', 1)
        flagged = int(buffer.pre_death_flags[: buffer._main.size].sum())
        assert flagged > 0
        batch = buffer.sample(64, return_indices=True)
        if batch is None:
            pytest.skip("Buffer did not return a batch")
        indices = batch[-2] if len(batch) == 9 else batch[-1]
        selected_flags = [buffer.pre_death_flags[idx] for idx in indices if idx >= 0]
        flagged_count = sum(1 for flag in selected_flags if flag)
        assert flagged_count >= max(1, int(0.2 * len(selected_flags)))
    finally:
        RL_CONFIG.pre_death_sample_fraction = original_frac
