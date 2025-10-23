#!/usr/bin/env python3
"""Unit tests for the segmented replay buffer with priority buckets."""

import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Scripts"))

from aimodel import HybridReplayBuffer
from config import RL_CONFIG


class TestSegmentedReplayBuffer(unittest.TestCase):
    def setUp(self):
        self.state_size = 32
        self.capacity = 256

        # Preserve existing configuration and switch to a small, deterministic layout
        self._old_n = RL_CONFIG.replay_n_buckets
        self._old_bucket = RL_CONFIG.replay_bucket_size
        self._old_main = RL_CONFIG.replay_main_bucket_size

        RL_CONFIG.replay_n_buckets = 2
        RL_CONFIG.replay_bucket_size = 32
        RL_CONFIG.replay_main_bucket_size = 192

        self.buffer = HybridReplayBuffer(self.capacity, self.state_size)

    def tearDown(self):
        RL_CONFIG.replay_n_buckets = self._old_n
        RL_CONFIG.replay_bucket_size = self._old_bucket
        RL_CONFIG.replay_main_bucket_size = self._old_main

    def _make_transition(self, actor="dqn", reward=None):
        state = np.random.randn(self.state_size).astype(np.float32)
        next_state = np.random.randn(self.state_size).astype(np.float32)
        discrete_action = np.random.randint(0, 4)
        continuous_action = float(np.random.uniform(-0.9, 0.9))
        reward_val = float(np.random.randn()) if reward is None else float(reward)
        done = bool(np.random.rand() < 0.05)
        horizon = np.random.randint(1, 4)
        return state, discrete_action, continuous_action, reward_val, next_state, done, actor, horizon

    def _push_transition(self, actor="dqn", reward=None):
        state, discrete_action, continuous_action, reward_val, next_state, done, act_tag, horizon = self._make_transition(actor=actor, reward=reward)
        priority_val = abs(reward_val)
        self.buffer.push(
            state,
            discrete_action,
            continuous_action,
            reward_val,
            next_state,
            done,
            actor=act_tag,
            horizon=horizon,
            priority_reward=priority_val,
        )

    def test_initial_state(self):
        stats = self.buffer.get_partition_stats()
        self.assertEqual(stats["total_size"], 0)
        self.assertEqual(stats["total_capacity"], self.buffer.capacity)
        self.assertTrue(stats["priority_buckets_enabled"])
        self.assertEqual(stats["priority_bucket_count"], self.buffer.n_buckets)
        self.assertEqual(len(stats.get("bucket_labels", [])), self.buffer.n_buckets)
        self.assertEqual(self.buffer.get_actor_composition()["total"], 0)

    def test_push_and_len(self):
        for _ in range(10):
            self._push_transition()
        self.assertEqual(len(self.buffer), 10)

        for _ in range(self.capacity * 2):
            self.buffer.push(*self._make_transition())
        self.assertEqual(len(self.buffer), self.buffer.capacity)

    def test_sample_shapes(self):
        for _ in range(self.capacity // 2):
            self._push_transition()

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
        self.assertIsInstance(actors[0], str)

    def test_actor_composition(self):
        for _ in range(50):
            self._push_transition(actor="dqn")
        for _ in range(30):
            self._push_transition(actor="expert")

        comp = self.buffer.get_actor_composition()
        total = comp["total"]
        self.assertEqual(total, len(self.buffer))
        self.assertEqual(comp["dqn"] + comp["expert"], total)
        if total > 0:
            self.assertAlmostEqual(comp["frac_dqn"], comp["dqn"] / total, places=6)

    def test_partition_stats(self):
        for _ in range(40):
            self._push_transition()
        stats = self.buffer.get_partition_stats()
        self.assertEqual(stats["total_size"], 40)
        self.assertIn("main_fill_pct", stats)
        self.assertTrue(stats["priority_buckets_enabled"])
        priority_keys = [k for k in stats if k.startswith("p") and k.endswith("_fill_pct")]
        self.assertGreater(len(priority_keys), 0)

    def test_sample_requires_enough_data(self):
        self.assertIsNone(self.buffer.sample(32))
        for _ in range(20):
            self._push_transition()
        self.assertIsNone(self.buffer.sample(32))
        for _ in range(20):
            self._push_transition()
        self.assertIsNotNone(self.buffer.sample(32))

    def test_priority_bucket_receives_expert(self):
        for _ in range(10):
            self._push_transition(actor="expert", reward=5.0)
        stats = self.buffer.get_partition_stats()
        bucket_labels = stats.get("bucket_labels", [])
        priority_sizes = [stats.get(f"{label}_size", 0) for label in bucket_labels]
        self.assertTrue(any(size > 0 for size in priority_sizes))


if __name__ == "__main__":
    unittest.main()
