#!/usr/bin/env python3
"""Unit tests for the discrete spinner action mapping."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Scripts"))

from aimodel import (  # noqa: E402
    SPINNER_BUCKET_VALUES,
    NUM_SPINNER_BUCKETS,
    action_index_to_components,
    compose_action_index,
    decompose_action_index,
    encode_action_from_components,
    quantize_spinner_value,
    fire_zap_to_discrete,
    discrete_to_fire_zap,
    HybridDQNAgent,
)
from config import RL_CONFIG  # noqa: E402


class SpinnerMappingTests(unittest.TestCase):
    def test_round_trip_indices(self):
        """Every fire/zap + spinner bucket pair should round-trip through helpers."""
        for fire_zap in range(4):
            for spinner_idx in range(NUM_SPINNER_BUCKETS):
                action_index = compose_action_index(fire_zap, spinner_idx)
                self.assertEqual(decompose_action_index(action_index), (fire_zap, spinner_idx))

                fire, zap, recovered_idx, spinner_value = action_index_to_components(action_index)
                self.assertEqual(recovered_idx, spinner_idx)
                self.assertEqual(fire_zap_to_discrete(fire, zap), fire_zap)
                self.assertAlmostEqual(spinner_value, SPINNER_BUCKET_VALUES[spinner_idx], places=6)

                rebuilt_index, rebuilt_spinner_idx, rebuilt_value = encode_action_from_components(
                    fire, zap, spinner_value
                )
                self.assertEqual(rebuilt_index, action_index)
                self.assertEqual(rebuilt_spinner_idx, spinner_idx)
                self.assertAlmostEqual(rebuilt_value, spinner_value, places=6)

    def test_quantization_prefers_nearest_bucket(self):
        """Quantization should snap to the nearest configured spinner bucket."""
        for idx, bucket_value in enumerate(SPINNER_BUCKET_VALUES):
            noisy_value = bucket_value + 0.01
            self.assertEqual(quantize_spinner_value(noisy_value), idx)
            noisy_value = bucket_value - 0.01
            self.assertEqual(quantize_spinner_value(noisy_value), idx)

    def test_agent_act_returns_valid_index(self):
        """HybridDQNAgent.act should always emit an in-range action index."""
        agent = HybridDQNAgent(state_size=RL_CONFIG.state_size)
        dummy_state = np.zeros(agent.state_size, dtype=np.float32)
        for eps in (0.0, 0.5, 1.0):
            action_index = agent.act(dummy_state, epsilon=eps, add_noise=True)
            self.assertIsInstance(action_index, int)
            self.assertGreaterEqual(action_index, 0)
            self.assertLess(action_index, agent.discrete_actions)


if __name__ == "__main__":
    unittest.main()
