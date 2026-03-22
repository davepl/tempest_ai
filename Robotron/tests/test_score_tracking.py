#!/usr/bin/env python3
"""Regression tests for Robotron score tracking guards."""

import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Scripts"))

from score_tracking import advance_score_tracking  # type: ignore


def _frame(score: int, *, alive: bool = True, wave: int = 1, lasers: int = 3):
    return SimpleNamespace(
        player_alive=alive,
        game_score=score,
        level_number=wave,
        num_lasers=lasers,
    )


class ScoreTrackingTests(unittest.TestCase):
    def test_rejects_off_grid_score_even_when_alive(self):
        cs = {
            "last_alive_game_score": 10_000,
            "score_tracking_ready": True,
            "score_tracking_candidate_score": None,
            "score_tracking_candidate_frames": 0,
        }

        accepted, completed, reason = advance_score_tracking(cs, _frame(50_535_055))
        self.assertIsNone(accepted)
        self.assertIsNone(completed)
        self.assertEqual(reason, "score_not_multiple_of_25")
        self.assertTrue(cs["score_tracking_ready"])

        accepted, completed, reason = advance_score_tracking(cs, _frame(10_100))
        self.assertEqual(accepted, 10_100)
        self.assertIsNone(completed)
        self.assertIsNone(reason)

    def test_rejects_implausible_positive_score_jump(self):
        cs = {
            "last_alive_game_score": 20_000,
            "score_tracking_ready": True,
            "score_tracking_candidate_score": None,
            "score_tracking_candidate_frames": 0,
        }

        accepted, completed, reason = advance_score_tracking(cs, _frame(220_025))
        self.assertIsNone(accepted)
        self.assertIsNone(completed)
        self.assertEqual(reason, "score_jump_too_large")

        accepted, completed, reason = advance_score_tracking(cs, _frame(20_025))
        self.assertEqual(accepted, 20_025)
        self.assertIsNone(completed)
        self.assertIsNone(reason)

    def test_records_prior_game_once_after_confirmed_restart(self):
        cs = {
            "last_alive_game_score": 30_000,
            "score_tracking_ready": False,
            "score_tracking_candidate_score": None,
            "score_tracking_candidate_frames": 0,
        }

        accepted, completed, reason = advance_score_tracking(cs, _frame(0))
        self.assertIsNone(accepted)
        self.assertIsNone(completed)
        self.assertIsNone(reason)

        accepted, completed, reason = advance_score_tracking(cs, _frame(25))
        self.assertEqual(accepted, 25)
        self.assertEqual(completed, 30_000)
        self.assertIsNone(reason)
        cs["last_alive_game_score"] = accepted

        accepted, completed, reason = advance_score_tracking(cs, _frame(50))
        self.assertEqual(accepted, 50)
        self.assertIsNone(completed)
        self.assertIsNone(reason)

    def test_dead_frames_reset_confirmation_state(self):
        cs = {
            "last_alive_game_score": 12_500,
            "score_tracking_ready": True,
            "score_tracking_candidate_score": None,
            "score_tracking_candidate_frames": 0,
        }

        accepted, completed, reason = advance_score_tracking(cs, _frame(12_500, alive=False))
        self.assertIsNone(accepted)
        self.assertIsNone(completed)
        self.assertEqual(reason, "player_dead")
        self.assertFalse(cs["score_tracking_ready"])


if __name__ == "__main__":
    unittest.main()
