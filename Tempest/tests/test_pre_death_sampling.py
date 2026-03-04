#!/usr/bin/env python3
"""Tests for pre-death priority boosting in the replay buffer."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Scripts'))

from replay_buffer import PrioritizedReplayBuffer  # type: ignore
from config import RL_CONFIG  # type: ignore


def _fill_buffer(buf, n_episodes=10, steps_per_ep=20):
    """Push several episodes into *buf*, returning the last data_ptr per episode."""
    state_size = buf.state_size
    s = np.zeros(state_size, dtype=np.float32)
    episode_end_indices = []
    for _ in range(n_episodes):
        for step in range(steps_per_ep):
            done = (step == steps_per_ep - 1)
            buf.add(s, 0, -1.0 if done else 0.01, s, done, horizon=1)
        # index of the terminal frame just written
        episode_end_indices.append((buf.tree.data_ptr - 1) % buf.capacity)
    return episode_end_indices


def test_boost_priorities_multiplies_existing():
    """boost_priorities should multiply each target slot's priority by *factor*."""
    buf = PrioritizedReplayBuffer(capacity=256, state_size=4)
    s = np.zeros(4, dtype=np.float32)
    # Add a few transitions so they get default max-priority
    indices = []
    for _ in range(5):
        buf.add(s, 0, 0.1, s, False)
        indices.append((buf.tree.data_ptr - 1) % buf.capacity)

    before = [buf.tree.priority(i) for i in indices]
    factor = 3.0
    buf.boost_priorities(indices, factor)
    after = [buf.tree.priority(i) for i in indices]

    for b, a in zip(before, after):
        assert abs(a - b * factor) < 1e-6, f"Expected {b * factor}, got {a}"


def test_boost_priorities_noop_when_factor_le_1():
    """Factor <= 1.0 should leave priorities unchanged."""
    buf = PrioritizedReplayBuffer(capacity=64, state_size=4)
    s = np.zeros(4, dtype=np.float32)
    buf.add(s, 0, 0.1, s, False)
    idx = (buf.tree.data_ptr - 1) % buf.capacity
    before = buf.tree.priority(idx)
    buf.boost_priorities([idx], 0.5)
    assert buf.tree.priority(idx) == before
    buf.boost_priorities([idx], 1.0)
    assert buf.tree.priority(idx) == before


def test_boost_priorities_empty_indices():
    """Empty index list should not raise."""
    buf = PrioritizedReplayBuffer(capacity=64, state_size=4)
    buf.boost_priorities([], 5.0)  # no-op, no crash


def test_boosted_frames_sampled_more_often():
    """After boosting a subset of frames, they should appear more frequently in samples."""
    buf = PrioritizedReplayBuffer(capacity=512, state_size=4)
    s = np.zeros(4, dtype=np.float32)

    # Fill buffer with 200 transitions
    for _ in range(200):
        buf.add(s, 0, 0.01, s, False)

    # Boost the last 10 frames
    boosted = [(buf.tree.data_ptr - 1 - i) % buf.capacity for i in range(10)]
    buf.boost_priorities(boosted, 10.0)

    boosted_set = set(boosted)
    hit = 0
    N = 5000
    for _ in range(N):
        batch = buf.sample(32)
        if batch is None:
            continue
        indices = batch[7]  # indices position in the return tuple
        hit += sum(1 for idx in indices if idx in boosted_set)

    # 10/200 = 5% base rate; with 10x boost we expect ≫ 5%
    hit_rate = hit / (N * 32)
    assert hit_rate > 0.15, f"Boosted frames sampled at only {hit_rate:.1%}, expected > 15%"

