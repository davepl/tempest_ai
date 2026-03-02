#!/usr/bin/env python3
"""Regression tests for enemy slot depth masking behavior."""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Scripts"))

from aimodel import RainbowNet  # type: ignore  # pylint: disable=import-error
from config import SERVER_CONFIG, RL_CONFIG  # type: ignore  # pylint: disable=import-error


def test_top_of_tube_depth_is_not_masked_as_empty():
    fs = max(1, RL_CONFIG.frame_stack)
    stacked_size = SERVER_CONFIG.params_count * fs
    net = RainbowNet(stacked_size)
    state = torch.zeros((1, stacked_size), dtype=torch.float32)

    # Slot 0: top-of-tube enemy depth (0x10) encoded as raw normalized depth.
    # These indices are within the first frame (indices 0..194).
    state[0, 133] = 0.0  # segment rel 0
    state[0, 140] = 0x10 / 255.0
    state[0, 147] = 0.0  # top-segment rel 0

    # Slot 1: explicitly empty.
    state[0, 134] = -1.0
    state[0, 141] = 0.0
    state[0, 148] = -1.0

    tokens, empty_mask = net._build_enemy_tokens(state)

    # With temporal tokens: first 7 slots are current frame, sorted by depth.
    # Sorted slot 0 should be the active top-of-tube enemy and must not be masked.
    assert tokens[0, 0, 7].item() > 0.0  # depth feature (index 7 in the 15-feature token)
    assert not bool(empty_mask[0, 0].item())

    # At least one slot should still be considered empty in this mixed case.
    assert bool(empty_mask[0].any().item())
