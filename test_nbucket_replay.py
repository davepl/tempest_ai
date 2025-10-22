#!/usr/bin/env python3
"""Unit tests for the simplified uniform replay buffer."""

import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Scripts"))

from aimodel import HybridReplayBuffer


class TestUniformReplayBuffer(unittest.TestCase):
    def setUp(self):
        self.state_size = 32
        self.capacity = 256
        self.buffer = HybridReplayBuffer(self.capacity, self.state_size)

    def _make_transition(self, actor="dqn"):
        state = np.random.randn(self.state_size).astype(np.float32)
        next_state = np.random.randn(self.state_size).astype(np.float32)
        discrete_action = np.random.randint(0, 4)
        continuous_action = float(np.random.uniform(-0.9, 0.9))
        reward = float(np.random.randn())
        done = bool(np.random.rand() < 0.05)
        horizon = np.random.randint(1, 4)
        return state, discrete_action, continuous_action, reward, next_state, done, actor, horizon

    def test_initial_state(self):
        stats = self.buffer.get_partition_stats()
        self.assertEqual(stats["total_size"], 0)
        self.assertEqual(stats["total_capacity"], self.capacity)
        self.assertFalse(stats["priority_buckets_enabled"])
        self.assertEqual(self.buffer.get_actor_composition()["total"], 0)

    def test_push_and_len(self):
        for _ in range(10):
            self.buffer.push(*self._make_transition())
        self.assertEqual(len(self.buffer), 10)

        # Capacity clamp
        for _ in range(self.capacity * 2):
            self.buffer.push(*self._make_transition())
        self.assertEqual(len(self.buffer), self.capacity)

    def test_sample_shapes(self):
        for _ in range(self.capacity // 2):
            self.buffer.push(*self._make_transition())

        batch = self.buffer.sample(64)
        self.assertIsNotNone(batch)
        states, discrete, continuous, rewards, next_states, dones, actors, horizons = batch
        self.assertEqual(states.shape, (64, self.state_size))
        self.assertEqual(discrete.shape, (64, 1))
        self.assertEqual(continuous.shape, (64, 1))
        self.assertEqual(rewards.shape, (64, 1))
        self.assertEqual(next_states.shape, (64, self.state_size))
        self.assertEqual(dones.shape, (64, 1))
        self.assertEqual(len(actors), 64)
        self.assertEqual(horizons.shape, (64, 1))

    def test_actor_composition(self):
        for _ in range(50):
            self.buffer.push(*self._make_transition(actor="dqn"))
        for _ in range(30):
            self.buffer.push(*self._make_transition(actor="expert"))

        comp = self.buffer.get_actor_composition()
        self.assertEqual(comp["total"], 80)
        self.assertEqual(comp["dqn"], 50)
        self.assertEqual(comp["expert"], 30)
        self.assertAlmostEqual(comp["frac_dqn"], 50 / 80, places=3)

    def test_partition_stats(self):
        for _ in range(40):
            self.buffer.push(*self._make_transition())
        stats = self.buffer.get_partition_stats()
        self.assertEqual(stats["total_size"], 40)
        self.assertIn("main_fill_pct", stats)
        self.assertFalse(stats["priority_buckets_enabled"])

    def test_sample_requires_enough_data(self):
        self.assertIsNone(self.buffer.sample(32))
        for _ in range(20):
            self.buffer.push(*self._make_transition())
        self.assertIsNone(self.buffer.sample(32))
        for _ in range(20):
            self.buffer.push(*self._make_transition())
        self.assertIsNotNone(self.buffer.sample(32))


if __name__ == "__main__":
    unittest.main()
