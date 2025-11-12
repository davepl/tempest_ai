#!/usr/bin/env python3
"""Tests for the supervision warm-up schedule logic."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Scripts"))

from training import _is_supervision_active  # noqa: E402
from config import RL_CONFIG  # noqa: E402


class SupervisionScheduleTests(unittest.TestCase):
    def test_supervision_window(self):
        old_value = RL_CONFIG.supervision_warmup_frames
        try:
            RL_CONFIG.supervision_warmup_frames = 100
            self.assertTrue(_is_supervision_active(0))
            self.assertTrue(_is_supervision_active(99))
            self.assertFalse(_is_supervision_active(100))
            self.assertFalse(_is_supervision_active(500))
        finally:
            RL_CONFIG.supervision_warmup_frames = old_value


if __name__ == "__main__":
    unittest.main()
