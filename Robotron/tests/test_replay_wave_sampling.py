#!/usr/bin/env python3
"""Regression tests for wave-aware replay sampling."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Scripts"))

from config import RL_CONFIG, metrics  # type: ignore
from replay_buffer import PrioritizedReplayBuffer  # type: ignore


class ReplayWaveSamplingTests(unittest.TestCase):
    def test_wave_aware_sampling_preserves_frontier_and_high_wave_dqn(self):
        np.random.seed(0)

        keys = [
            "replay_wave_sampling_enabled",
            "replay_wave_frontier_frac",
            "replay_wave_high_frac",
            "replay_wave_frontier_margin",
            "replay_wave_high_offset",
            "replay_wave_candidate_multiplier",
            "replay_wave_min_frontier",
            "replay_expert_max_frac",
        ]
        saved = {k: getattr(RL_CONFIG, k) for k in keys}
        old_avg = metrics.average_level
        old_peak = metrics.peak_level

        try:
            RL_CONFIG.replay_wave_sampling_enabled = True
            RL_CONFIG.replay_wave_frontier_frac = 0.35
            RL_CONFIG.replay_wave_high_frac = 0.20
            RL_CONFIG.replay_wave_frontier_margin = 1
            RL_CONFIG.replay_wave_high_offset = 2
            RL_CONFIG.replay_wave_candidate_multiplier = 16
            RL_CONFIG.replay_wave_min_frontier = 4
            RL_CONFIG.replay_expert_max_frac = 0.25
            metrics.average_level = 6.4
            metrics.peak_level = 8

            buf = PrioritizedReplayBuffer(capacity=256, state_size=4, alpha=RL_CONFIG.priority_alpha)
            state = np.zeros(4, dtype=np.float32)

            for _ in range(64):
                buf.add(state, 0, 0.5, state, False, expert=1, priority_hint=1.0, wave_number=1, start_wave=1)
            for _ in range(64):
                buf.add(state, 0, 0.5, state, False, expert=0, priority_hint=1.0, wave_number=1, start_wave=1)
            for _ in range(64):
                buf.add(state, 0, 0.5, state, False, expert=0, priority_hint=1.0, wave_number=7, start_wave=7)
            for _ in range(64):
                buf.add(state, 0, 0.5, state, False, expert=0, priority_hint=1.0, wave_number=10, start_wave=10)

            batch = buf.sample(32, beta=0.4)
            self.assertIsNotNone(batch)
            (
                _states,
                _actions,
                _rewards,
                _next_states,
                _dones,
                _horizons,
                is_expert,
                wave_numbers,
                start_waves,
                _indices,
                _weights,
            ) = batch

            effective_wave = np.maximum(wave_numbers.astype(np.int16), start_waves.astype(np.int16))
            frontier_dqn = ((is_expert == 0) & (effective_wave >= 6) & (effective_wave <= 8)).sum()
            high_dqn = ((is_expert == 0) & (effective_wave >= 9)).sum()

            self.assertGreaterEqual(int(frontier_dqn), 8)
            self.assertGreaterEqual(int(high_dqn), 4)
            self.assertLessEqual(int(is_expert.sum()), 8)
        finally:
            for k, v in saved.items():
                setattr(RL_CONFIG, k, v)
            metrics.average_level = old_avg
            metrics.peak_level = old_peak


if __name__ == "__main__":
    unittest.main()
