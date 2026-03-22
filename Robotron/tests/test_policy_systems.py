#!/usr/bin/env python3
"""Regression tests for plateau response and expert heuristics."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Scripts"))

from aimodel import get_expert_action  # type: ignore
from config import (  # type: ignore
    LEGACY_CORE_FEATURES,
    LEGACY_ELIST_FEATURES,
    LEGACY_SLOT_STATE_FEATURES,
    RL_CONFIG,
    PlateauPulser,
    UNIFIED_NUM_TYPES,
    UNIFIED_TYPE_NAMES,
)


def _blank_state() -> np.ndarray:
    state = np.zeros(RL_CONFIG.state_size, dtype=np.float32)
    latest = state[-RL_CONFIG.base_state_size:]
    latest[0] = 1.0
    latest[5] = 0.5
    latest[6] = 0.5
    return state


def _slot_offset(idx: int) -> int:
    entity_base = LEGACY_CORE_FEATURES + LEGACY_ELIST_FEATURES
    slot_base = entity_base + 1
    return (RL_CONFIG.state_size - RL_CONFIG.base_state_size) + slot_base + (idx * LEGACY_SLOT_STATE_FEATURES)


def _set_token(state: np.ndarray, idx: int, category: str, *, dx: float, dy: float,
               dist: float, threat: float):
    entity_base = (RL_CONFIG.state_size - RL_CONFIG.base_state_size) + LEGACY_CORE_FEATURES + LEGACY_ELIST_FEATURES
    state[entity_base] = 1.0
    off = _slot_offset(idx)
    type_norm = UNIFIED_TYPE_NAMES.index(category) / max(1, UNIFIED_NUM_TYPES - 1)
    state[off:off + LEGACY_SLOT_STATE_FEATURES] = np.array([
        1.0, dx, dy, dist, 0.0, 0.0, threat, 0.60, 0.25, 0.50, type_norm,
    ], dtype=np.float32)


class PolicySystemTests(unittest.TestCase):
    def test_plateau_pulser_triggers_frontier_pulse(self):
        keys = [
            "plateau_pulse_enabled",
            "plateau_pulse_frames",
            "plateau_confirm_frames",
            "plateau_cooldown_frames",
            "plateau_min_frame",
            "plateau_reward_delta",
            "plateau_level_delta",
            "plateau_curriculum_wave_offset",
            "replay_wave_min_frontier",
        ]
        saved = {k: getattr(RL_CONFIG, k) for k in keys}
        try:
            RL_CONFIG.plateau_pulse_enabled = True
            RL_CONFIG.plateau_pulse_frames = 6
            RL_CONFIG.plateau_confirm_frames = 5
            RL_CONFIG.plateau_cooldown_frames = 8
            RL_CONFIG.plateau_min_frame = 10
            RL_CONFIG.plateau_reward_delta = 0.5
            RL_CONFIG.plateau_level_delta = 0.25
            RL_CONFIG.plateau_curriculum_wave_offset = 2
            RL_CONFIG.replay_wave_min_frontier = 4

            pulser = PlateauPulser()
            pulser.update(10, 6.2, 0.0, 10.0, 10.0)
            self.assertEqual(pulser.state, PlateauPulser.WATCHING)

            pulser.update(16, 6.2, 0.0, 10.1, 10.0)
            self.assertEqual(pulser.state, PlateauPulser.PULSING)
            self.assertGreaterEqual(pulser.total_pulses, 1)

            gs = pulser.overlay_game_settings({"start_advanced": False, "start_level_min": 1}, 6.2)
            self.assertTrue(gs["start_advanced"])
            self.assertGreaterEqual(int(gs["start_level_min"]), 8)

            pulser.update(23, 6.2, 0.0, 10.1, 10.0)
            self.assertEqual(pulser.state, PlateauPulser.RECOVERING)

            pulser.update(40, 7.0, 0.0, 11.0, 10.5)
            self.assertEqual(pulser.state, PlateauPulser.WATCHING)
        finally:
            for k, v in saved.items():
                setattr(RL_CONFIG, k, v)

    def test_high_wave_expert_uses_center_bias_and_priority_fire(self):
        state = _blank_state()
        latest = state[-RL_CONFIG.base_state_size:]
        latest[4] = 8.0 / 40.0   # wave number
        latest[5] = 0.95         # player near right wall
        latest[6] = 0.50

        _set_token(state, 0, "grunt", dx=0.20, dy=0.0, dist=0.20, threat=0.55)
        _set_token(state, 1, "projectile", dx=-0.08, dy=0.0, dist=0.06, threat=0.95)

        move_dir, fire_dir = get_expert_action(state)

        self.assertIn(fire_dir, {5, 6, 7})  # left-biased toward the close projectile / blocking hazard
        self.assertIn(move_dir, range(RL_CONFIG.num_move_actions))


if __name__ == "__main__":
    unittest.main()
