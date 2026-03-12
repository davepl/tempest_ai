#!/usr/bin/env python3
"""Stdlib unit tests for Robotron curriculum game settings."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import GameSettings, compute_robotron_auto_curriculum_level  # noqa: E402


class GameSettingsTests(unittest.TestCase):
    def test_start_level_is_clamped_to_robotron_wave_range(self):
        gs = GameSettings()
        gs.start_level_min = -50
        self.assertEqual(gs.start_level_min, 1)

        gs.start_level_min = 999
        self.assertEqual(gs.start_level_min, 81)

    def test_snapshot_round_trips_persisted_values(self):
        gs = GameSettings()
        gs.start_advanced = True
        gs.start_level_min = 27
        gs.auto_curriculum = True
        gs.epsilon_pct = 12
        gs.expert_pct = 34

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "game_settings.json")
            gs.save(path)

            loaded = GameSettings()
            loaded.load(path)

        self.assertEqual(
            loaded.snapshot(),
            {
                "start_advanced": True,
                "start_level_min": 27,
                "epsilon_pct": 12,
                "expert_pct": 34,
                "auto_curriculum": True,
            },
        )

    def test_robotron_auto_curriculum_level_uses_avg_minus_three(self):
        self.assertEqual(compute_robotron_auto_curriculum_level(1.0), 1)
        self.assertEqual(compute_robotron_auto_curriculum_level(3.9), 1)
        self.assertEqual(compute_robotron_auto_curriculum_level(8.4), 5)
        self.assertEqual(compute_robotron_auto_curriculum_level(81.0), 78)
        self.assertEqual(compute_robotron_auto_curriculum_level(999.0), 81)


if __name__ == "__main__":
    unittest.main()
