#!/usr/bin/env python3
"""Regression tests for BC schedule/metric reporting."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import training  # noqa: E402


class TrainingMetricTests(unittest.TestCase):
    def test_bc_schedule_decays_to_zero_after_handoff(self):
        self.assertAlmostEqual(training._bc_weight_schedule(0), 0.30)
        self.assertAlmostEqual(training._bc_weight_schedule(6_000_000), 0.30)
        self.assertAlmostEqual(training._bc_weight_schedule(14_000_000), 0.0)
        self.assertAlmostEqual(training._bc_weight_schedule(112_000_000), 0.0)

    def test_bc_metric_contribution_zero_when_bc_is_inactive(self):
        self.assertEqual(training._bc_metric_contribution(0.42, 0.0, 0.25), 0.0)
        self.assertEqual(training._bc_metric_contribution(0.42, 0.3, 0.0), 0.0)
        self.assertEqual(training._bc_metric_contribution(0.0, 0.3, 0.25), 0.0)

    def test_bc_metric_contribution_matches_weighted_term(self):
        self.assertAlmostEqual(training._bc_metric_contribution(0.5, 0.3, 0.25), 0.0375)


if __name__ == "__main__":
    unittest.main()
