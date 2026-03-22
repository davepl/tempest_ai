#!/usr/bin/env python3
"""Regression tests for the current Robotron unified-slot model path."""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Scripts"))

from aimodel import RainbowNet  # type: ignore
from config import (  # type: ignore
    LEGACY_CORE_FEATURES,
    LEGACY_ELIST_FEATURES,
    LEGACY_SLOT_STATE_FEATURES,
    RL_CONFIG,
    SERVER_CONFIG,
    UNIFIED_HUMAN_TYPE_ID,
    UNIFIED_NUM_TYPES,
)


class ModelShapeTests(unittest.TestCase):
    def test_config_uses_stacked_unified_slot_state(self):
        self.assertEqual(RL_CONFIG.frame_stack, 2)
        self.assertEqual(RL_CONFIG.base_state_size, SERVER_CONFIG.params_count)
        self.assertEqual(RL_CONFIG.state_size, SERVER_CONFIG.params_count * RL_CONFIG.frame_stack)
        self.assertTrue(RL_CONFIG.use_directional_lanes)
        self.assertTrue(RL_CONFIG.use_pointer_action_heads)
        self.assertTrue(RL_CONFIG.temporal_memory_enabled)

    def test_unified_slot_tokens_use_latest_frame_block(self):
        net = RainbowNet(RL_CONFIG.state_size)
        state = torch.zeros((1, RL_CONFIG.state_size), dtype=torch.float32)

        frame_off = net._frame_offsets(state)[-1]
        entity_base = frame_off + LEGACY_CORE_FEATURES + LEGACY_ELIST_FEATURES
        slot_base = entity_base + 1
        type_norm = UNIFIED_HUMAN_TYPE_ID / max(1, UNIFIED_NUM_TYPES - 1)

        state[0, entity_base] = 1.0  # occupancy
        state[0, slot_base + 0] = 1.0
        state[0, slot_base + 1] = 0.25
        state[0, slot_base + 2] = -0.50
        state[0, slot_base + 3] = 0.30
        state[0, slot_base + 4] = 0.10
        state[0, slot_base + 5] = -0.05
        state[0, slot_base + 6] = 0.85
        state[0, slot_base + 7] = 0.65
        state[0, slot_base + 8] = 0.50
        state[0, slot_base + 9] = 0.75
        state[0, slot_base + 10] = type_norm

        tokens, mask = net._build_frame_object_tokens(state, frame_off)

        self.assertEqual(tuple(tokens.shape), (1, RL_CONFIG.object_slots, 27))
        self.assertEqual(tuple(mask.shape), (1, RL_CONFIG.object_slots))
        self.assertFalse(bool(mask[0, 0]))
        self.assertTrue(bool(mask[0, 1]))
        torch.testing.assert_close(
            tokens[0, 0, :9],
            torch.tensor([0.25, -0.50, 0.10, -0.05, 0.30, 0.85, 0.65, 0.50, 0.75]),
            atol=1e-6,
            rtol=0.0,
        )
        self.assertEqual(float(tokens[0, 0, -2].item()), 1.0)  # is_human
        self.assertEqual(float(tokens[0, 0, -1].item()), 0.0)  # is_dangerous

    def test_pointer_policy_and_forward_shapes(self):
        net = RainbowNet(RL_CONFIG.state_size)
        net.eval()
        state = torch.zeros((2, RL_CONFIG.state_size), dtype=torch.float32)
        with torch.no_grad():
            debug = net.debug_pointer_policy(state)
            out = net(state)

        self.assertEqual(tuple(out.shape), (2, RL_CONFIG.num_joint_actions, RL_CONFIG.num_atoms))
        self.assertEqual(tuple(debug["move_pointer_probs"].shape), (2, RL_CONFIG.object_slots + 1))
        self.assertEqual(tuple(debug["fire_pointer_probs"].shape), (2, RL_CONFIG.object_slots + 1))
        self.assertEqual(tuple(debug["move_mode_logits"].shape), (2, RL_CONFIG.move_mode_count))
        self.assertEqual(tuple(debug["move_dir_logits"].shape), (2, RL_CONFIG.num_move_actions))
        self.assertEqual(tuple(debug["fire_dir_logits"].shape), (2, RL_CONFIG.num_fire_actions))
        self.assertEqual(tuple(debug["move_target_slot"].shape), (2,))
        self.assertEqual(tuple(debug["fire_target_slot"].shape), (2,))
        self.assertEqual(tuple(debug["memory_ctx"].shape), (2, RL_CONFIG.temporal_memory_hidden))
        self.assertTrue(torch.isfinite(out).all().item())
        self.assertTrue(torch.isfinite(debug["move_dir_logits"]).all().item())
        self.assertTrue(torch.isfinite(debug["fire_dir_logits"]).all().item())


if __name__ == "__main__":
    unittest.main()
