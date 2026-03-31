#!/usr/bin/env python3
"""Smoke test for 4-frame directional temporal attention."""
import sys

import torch

sys.path.insert(0, ".")

from config import RL_CONFIG
from aimodel import NUM_JOINT, RainbowNet

EXPECTED_FRAME_STACK = 4
EXPECTED_STATE_DIM = 5832
EXPECTED_NUM_JOINT = 81


def main():
    assert int(RL_CONFIG.frame_stack) == EXPECTED_FRAME_STACK
    assert bool(RL_CONFIG.attn_all_frames) is True
    assert int(RL_CONFIG.state_size) == EXPECTED_STATE_DIM
    assert int(RL_CONFIG.num_fire_actions) == 9
    assert NUM_JOINT == EXPECTED_NUM_JOINT

    net = RainbowNet(RL_CONFIG.state_size)
    assert net.use_directional_lanes
    assert net.attn_frame_count == EXPECTED_FRAME_STACK
    assert net.directional_temporal_fusion
    assert net.use_directional_action_priors
    assert net.lane_temporal_in_dim == net.lane_summary_dim * EXPECTED_FRAME_STACK
    assert net.entity_temporal_in_dim == net.entity_pool_summary_dim * EXPECTED_FRAME_STACK
    assert net.grid_temporal_in_dim == net.grid_summary_dim * EXPECTED_FRAME_STACK

    batch_size = 2
    state = torch.zeros(batch_size, RL_CONFIG.state_size, dtype=torch.float32)

    projectile_base = None
    for pool_name, base, _slots, _slot_features, _pool_id in net._pool_info:
        if pool_name == "projectile":
            projectile_base = int(base)
            break
    assert projectile_base is not None

    for frame_idx, frame_off in enumerate(net._frame_offsets(state)):
        lane_base = frame_off + net.core_feature_count
        state[:, lane_base + 0] = 0.20 * (frame_idx + 1)
        state[:, lane_base + 1] = 0.10
        state[:, lane_base + 2] = -0.08
        state[:, lane_base + 5] = 0.60
        state[:, lane_base + 7] = 1.00
        state[:, lane_base + 16] = 1.00
        state[:, lane_base + 17] = 0.35
        state[:, lane_base + 18] = 0.12
        state[:, lane_base + 19] = 1.00

        proj_slot = frame_off + projectile_base + 1
        state[:, proj_slot + 0] = 1.00
        state[:, proj_slot + 1] = 0.05 * (frame_idx + 1)
        state[:, proj_slot + 2] = -0.03 * (frame_idx + 1)
        state[:, proj_slot + 3] = 0.20
        state[:, proj_slot + 4] = -0.01
        state[:, proj_slot + 5] = 0.02
        state[:, proj_slot + 6] = 0.80
        state[:, proj_slot + 7] = 0.25
        state[:, proj_slot + 8] = 0.10
        state[:, proj_slot + 9] = 0.60

        grid_start = frame_off + net.local_grid_offset
        center_idx = grid_start + (net.local_grid_feature_width // 2)
        state[:, center_idx] = 0.10 * (frame_idx + 1)
        state[:, grid_start + 3] = 0.40

    with torch.no_grad():
        lane_summary, entity_summary, grid_summary, frame_offsets, lane_frames, entity_frames, grid_frames = (
            net._build_directional_temporal_summaries(state, return_per_frame=True)
        )
        latest_lane_tokens, _lane_active = net._build_directional_lane_tokens(state)
        move_prior, fire_prior = net._build_directional_action_priors(latest_lane_tokens)
        dist = net(state)
        q_values = net.q_values(state)

    assert len(frame_offsets) == EXPECTED_FRAME_STACK
    assert len(lane_frames) == EXPECTED_FRAME_STACK
    assert len(entity_frames) == EXPECTED_FRAME_STACK
    assert len(grid_frames) == EXPECTED_FRAME_STACK
    assert lane_summary.shape == (batch_size, net.lane_summary_dim)
    assert entity_summary.shape == (batch_size, net.entity_pool_summary_dim)
    assert grid_summary.shape == (batch_size, net.grid_summary_dim)
    assert move_prior is not None and move_prior.shape == (batch_size, RL_CONFIG.num_move_actions)
    assert fire_prior is not None and fire_prior.shape == (batch_size, RL_CONFIG.num_fire_actions)
    assert dist.shape == (batch_size, NUM_JOINT, RL_CONFIG.num_atoms)
    assert q_values.shape == (batch_size, NUM_JOINT)

    print(
        "temporal_directional_ok",
        f"frames={len(frame_offsets)}",
        f"state_dim={RL_CONFIG.state_size}",
        f"dist_shape={tuple(dist.shape)}",
        f"q_shape={tuple(q_values.shape)}",
    )


if __name__ == "__main__":
    main()
