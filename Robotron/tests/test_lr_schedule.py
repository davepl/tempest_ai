#!/usr/bin/env python3
"""Regression tests for Robotron optimizer LR scheduling."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Scripts"))

from aimodel import compute_scheduled_lr  # type: ignore
from config import RL_CONFIG  # type: ignore


class LearningRateScheduleTests(unittest.TestCase):
    def test_warmup_starts_from_floor_and_reaches_peak(self):
        first = compute_scheduled_lr(0)
        end_warmup = compute_scheduled_lr(RL_CONFIG.lr_warmup_steps - 1)
        self.assertGreater(first, RL_CONFIG.lr_min)
        self.assertLess(first, RL_CONFIG.lr)
        self.assertAlmostEqual(end_warmup, RL_CONFIG.lr, places=12)

    def test_cosine_schedule_matches_restart_setting(self):
        warm = RL_CONFIG.lr_warmup_steps
        early = compute_scheduled_lr(warm + 10_000)
        mid = compute_scheduled_lr(warm + 80_000)
        late = compute_scheduled_lr(warm + 140_000)
        self.assertGreater(early, mid)
        self.assertGreater(mid, late)
        restart = compute_scheduled_lr(warm + RL_CONFIG.lr_cosine_period + 10_000)
        if RL_CONFIG.lr_use_restarts:
            self.assertAlmostEqual(restart, early, places=12)
            self.assertGreater(restart, RL_CONFIG.lr_min)
        else:
            self.assertAlmostEqual(restart, RL_CONFIG.lr_min, places=12)


if __name__ == "__main__":
    unittest.main()
