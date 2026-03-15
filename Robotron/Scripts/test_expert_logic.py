#!/usr/bin/env python3
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aimodel import (  # noqa: E402
    RL_CONFIG,
    _POS_MAX_DIAG,
    _REL_POS_X_RANGE,
    _REL_POS_Y_RANGE,
    get_cleanup_fire_override,
    get_expert_action,
)


def _category_offsets():
    return {name: idx for idx, (name, _slots) in enumerate(RL_CONFIG.entity_categories)}


_OFFSETS = _category_offsets()
_GLOBAL = int(getattr(RL_CONFIG, "global_feature_count", 98))
_GRID = int(getattr(RL_CONFIG, "grid_width", 12)) * int(getattr(RL_CONFIG, "grid_height", 12)) * int(getattr(RL_CONFIG, "grid_channels", 8))
_TOKEN_COUNT = int(getattr(RL_CONFIG, "object_token_count", 64))
_TOKEN_FEATURES = int(getattr(RL_CONFIG, "object_token_features", 15))
_HYBRID_BASE = _GLOBAL + _GRID + (_TOKEN_COUNT * _TOKEN_FEATURES)
_LEGACY_SLOT_FEATURES = int(getattr(RL_CONFIG, "slot_state_features", 4))


def _legacy_category_bases() -> dict[str, int]:
    bases = {}
    offset = 59
    for name, slots in getattr(RL_CONFIG, "entity_categories", ()):
        bases[name] = offset
        offset += 1 + (int(slots) * _LEGACY_SLOT_FEATURES)
    return bases


_LEGACY_BASES = _legacy_category_bases()
_LEGACY_SLOTS = {name: int(slots) for name, slots in getattr(RL_CONFIG, "entity_categories", ())}


def _blank_state() -> np.ndarray:
    state = np.zeros(int(RL_CONFIG.base_state_size), dtype=np.float32)
    state[0] = 1.0
    state[5] = 0.5
    state[6] = 0.5
    return state


def _add_entity(
    state: np.ndarray,
    category: str,
    slot_index: int,
    dx_px: float,
    dy_px: float,
) -> None:
    dx_world = float(dx_px) * 256.0
    dy_world = float(dy_px) * 256.0
    dist_world = math.hypot(dx_world, dy_world)
    if int(RL_CONFIG.base_state_size) >= _HYBRID_BASE:
        assert 0 <= slot_index < _TOKEN_COUNT
        token_base = _GLOBAL + _GRID + slot_index * _TOKEN_FEATURES
        cat_norm = _OFFSETS[category] / max(1, len(_OFFSETS) - 1)

        state[token_base + 0] = 1.0
        state[token_base + 1] = dx_world / _REL_POS_X_RANGE
        state[token_base + 2] = dy_world / _REL_POS_Y_RANGE
        state[token_base + 5] = dist_world / _POS_MAX_DIAG
        if dist_world > 1.0:
            state[token_base + 6] = dx_world / dist_world
            state[token_base + 7] = dy_world / dist_world
        state[token_base + 8] = max(0.0, min(1.0, 1.0 - state[token_base + 5]))
        state[token_base + 9] = 0.5
        state[token_base + 10] = 0.5
        state[token_base + 11] = cat_norm
        state[token_base + 12] = 1.0 if category == "human" else 0.0
        state[token_base + 13] = 0.0 if category == "human" else 1.0
        return

    assert category in _LEGACY_BASES
    assert 0 <= slot_index < _LEGACY_SLOTS[category]
    block_base = _LEGACY_BASES[category]
    state[block_base] = 1.0
    slot_base = block_base + 1 + slot_index * _LEGACY_SLOT_FEATURES
    state[slot_base + 0] = 1.0
    state[slot_base + 1] = dx_world / _REL_POS_X_RANGE
    state[slot_base + 2] = dy_world / _REL_POS_Y_RANGE
    state[slot_base + 3] = dist_world / _POS_MAX_DIAG


def test_aligned_fire_keeps_human_rescue_movement():
    state = _blank_state()
    _add_entity(state, "human", 0, 0, -20)
    _add_entity(state, "grunt", 1, 40, 0)

    move_dir, fire_dir = get_expert_action(state)

    assert move_dir == 0
    assert fire_dir == 2


def test_aligned_fire_picks_closest_aligned_target_across_directions():
    state = _blank_state()
    _add_entity(state, "grunt", 0, 40, 0)
    _add_entity(state, "projectile", 0, 0, -30)

    _, fire_dir = get_expert_action(state)

    assert fire_dir == 0


def test_axis_align_shortens_shorter_axis_once_no_humans_remain():
    state = _blank_state()
    _add_entity(state, "grunt", 0, 10, 40)

    move_dir, _ = get_expert_action(state)

    assert move_dir == 2


def test_axis_align_keeps_vertical_alignment_when_enemy_directly_below():
    state = _blank_state()
    _add_entity(state, "grunt", 0, 0, 40)

    move_dir, fire_dir = get_expert_action(state)

    assert move_dir == 4
    assert fire_dir == 4


def test_aligned_fire_includes_hulk_targets():
    state = _blank_state()
    _add_entity(state, "hulk", 0, 0, 30)

    move_dir, fire_dir = get_expert_action(state)

    assert move_dir == 4
    assert fire_dir == 4


def test_endgame_cleanup_aligns_to_last_hulk_once_humans_are_gone():
    state = _blank_state()
    _add_entity(state, "hulk", 0, 12, 36)

    move_dir, _ = get_expert_action(state)

    assert move_dir == 2


def test_cleanup_fire_override_only_applies_with_no_humans_and_few_targets():
    state = _blank_state()
    _add_entity(state, "hulk", 0, 0, 20)

    assert get_cleanup_fire_override(state) == 4

    _add_entity(state, "human", 0, -10, 0)
    assert get_cleanup_fire_override(state) is None


def test_final_hazard_repulsion_turns_move_away_from_close_hulk():
    state = _blank_state()
    _add_entity(state, "human", 0, 8, 0)
    _add_entity(state, "hulk", 0, 6, 0)

    move_dir, _ = get_expert_action(state)

    assert move_dir == 6


def test_final_hazard_repulsion_turns_move_away_from_close_electrode():
    state = _blank_state()
    _add_entity(state, "human", 0, 8, 0)
    _add_entity(state, "electrode", 0, 6, 0)

    move_dir, _ = get_expert_action(state)

    assert move_dir == 6


def test_final_hazard_check_avoids_obstacle_above():
    state = _blank_state()
    _add_entity(state, "human", 0, 0, -20)
    _add_entity(state, "electrode", 0, 0, -8)

    move_dir, _ = get_expert_action(state)

    assert move_dir == 4


def test_final_hazard_check_avoids_obstacle_on_right():
    state = _blank_state()
    _add_entity(state, "human", 0, 20, 0)
    _add_entity(state, "electrode", 0, 8, 0)

    move_dir, _ = get_expert_action(state)

    assert move_dir == 6


def test_final_hazard_check_avoids_obstacle_on_left():
    state = _blank_state()
    _add_entity(state, "human", 0, -20, 0)
    _add_entity(state, "electrode", 0, -8, 0)

    move_dir, _ = get_expert_action(state)

    assert move_dir == 2


def test_final_hazard_check_avoids_tank_above():
    state = _blank_state()
    _add_entity(state, "human", 0, 0, -20)
    _add_entity(state, "tank", 0, 2, -8)

    move_dir, _ = get_expert_action(state)

    assert move_dir == 4


def test_final_hazard_check_flees_instead_of_idling():
    state = _blank_state()
    _add_entity(state, "electrode", 0, 8, 0)

    move_dir, _ = get_expert_action(state)

    assert move_dir == 6
