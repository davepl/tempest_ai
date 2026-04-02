#!/usr/bin/env python3
"""Robotron AI v3 — Symbolic state processor.

Converts the raw 1454-float Lua wire state into structured entity sets
and global context tensors suitable for the Set Transformer.

Wire layout (from Lua):
  [0..17]     Core player features (18)
  [18..39]    ELIST mirror (22)
  [40..279]   Tactical lanes: 8 × 30 features (240)
  [280..765]  Tactical grid: 9×9×6 (486)
  [766..]     Entity pools:
                projectile: 1 occupancy + 24 × 10 = 241
                danger:     1 occupancy + 32 × 10 = 321
                human:      1 occupancy + 12 × 7  = 85
                electrode:  1 occupancy + 8 × 5   = 41
              Total pools: 688
  Total: 18 + 22 + 240 + 486 + 688 = 1454

Entity feature vector (per entity, 18 dims):
  [x, y, w, h, vx, vy, type_one_hot(12)]
  type_one_hot maps to: grunt, hulk, brain, tank, spawner, enforcer,
                        projectile, human, electrode, missile, spark, prog
"""

import numpy as np
import torch
from typing import Optional
from .config import (
    LEGACY_CORE_FEATURES, LEGACY_ELIST_FEATURES,
    TACTICAL_LANE_COUNT, TACTICAL_LANE_FEATURES,
    TACTICAL_LOCAL_GRID_FEATURES,
    ENTITY_POOL_DEFS,
    AUGMENTED_PARAMS_COUNT,
    PY_CONTROL_CONTEXT_FEATURES,
    CONFIG,
)

# Offsets into the wire state vector
_CORE_START = 0
_CORE_END = LEGACY_CORE_FEATURES                          # 18
_ELIST_START = _CORE_END
_ELIST_END = _ELIST_START + LEGACY_ELIST_FEATURES          # 40
_LANES_START = _ELIST_END
_LANES_END = _LANES_START + TACTICAL_LANE_COUNT * TACTICAL_LANE_FEATURES  # 280
_GRID_START = _LANES_END
_GRID_END = _GRID_START + TACTICAL_LOCAL_GRID_FEATURES     # 766
_POOLS_START = _GRID_END                                    # 766

# Pool offsets within the pools section
_POOL_OFFSETS = []
offset = 0
for name, slots, feats in ENTITY_POOL_DEFS:
    _POOL_OFFSETS.append((name, offset, slots, feats))
    offset += 1 + slots * feats  # 1 for occupancy counter

# Number of entity type classes for one-hot encoding
NUM_ENTITY_CLASSES = 12  # grunt, hulk, brain, tank, spawner, enforcer,
                         # projectile, human, electrode, missile, spark, prog

# Map pool name → entity type index for one-hot
_POOL_TYPE_MAP = {
    "danger": {
        # Danger pool can be grunt, hulk, brain, tank, enforcer, prog
        # We use per-slot heuristic classification based on features.
        # Default to "grunt" (0); recategorize by threat feature.
        "default": 0,
    },
    "projectile": {
        # Can be spark, missile, bounce bomb, electrode
        "default": 6,  # projectile
    },
    "human": {
        "default": 7,  # human
    },
    "electrode": {
        "default": 8,  # electrode
    },
}


def _decode_unified_type_id(type_norm: float) -> int:
    """Decode Lua's normalized UNIFIED_TYPE_ID back to an integer type id."""
    try:
        val = float(type_norm)
    except Exception:
        return 0
    val = max(0.0, min(1.0, val))
    # Lua emits type_id / (UNIFIED_NUM_TYPES - 1), currently /8.0.
    return int(round(val * 8.0))


def _classify_danger_entity(features: np.ndarray) -> int:
    """Heuristic type classification for entities in the danger pool.

    Danger pool slot layout (10 floats from Lua):
      [0] occupied (1.0)
      [1] dx  [2] dy  [3] dist_norm
      [4] vx  [5] vy  [6] threat
      [7] approach  [8] ttc_norm  [9] type_id/type_denom
    """
    if len(features) < 10:
        return 0  # grunt
    explicit_type = _decode_unified_type_id(features[9] if len(features) > 9 else 0.0)
    if 0 <= explicit_type <= 8:
        return explicit_type
    threat = features[6]
    speed = np.sqrt(features[4]**2 + features[5]**2)
    if threat > 0.8:
        return 2  # brain (highest threat)
    if threat > 0.5:
        return 3  # tank
    if speed < 0.01 and threat > 0.1:
        return 1  # hulk (slow but threatening, indestructible)
    if threat > 0.3:
        return 5  # enforcer
    return 0  # grunt


def _classify_danger_batch(pool_data: np.ndarray, feat_per_slot: int) -> np.ndarray:
    """Vectorized type classification for all slots in the danger pool.

    Args:
        pool_data: (max_slots, feat_per_slot) raw slot features
    Returns:
        type_ids: (max_slots,) int32 array of type IDs
    """
    n = pool_data.shape[0]
    type_ids = np.zeros(n, dtype=np.int32)
    if feat_per_slot < 10:
        return type_ids

    # Try explicit type first (index 9)
    raw_type = np.clip(pool_data[:, 9], 0.0, 1.0)
    explicit = np.rint(raw_type * 8.0).astype(np.int32)
    valid_explicit = (explicit >= 0) & (explicit <= 8)
    type_ids[valid_explicit] = explicit[valid_explicit]

    # Fallback heuristic for slots without valid explicit type
    need_heuristic = ~valid_explicit
    if need_heuristic.any():
        threat = pool_data[need_heuristic, 6]
        vx = pool_data[need_heuristic, 4]
        vy = pool_data[need_heuristic, 5]
        speed = np.sqrt(vx ** 2 + vy ** 2)
        h_ids = np.zeros(need_heuristic.sum(), dtype=np.int32)
        h_ids[threat > 0.8] = 2  # brain
        mask_tank = (threat > 0.5) & (threat <= 0.8)
        h_ids[mask_tank] = 3
        mask_hulk = (speed < 0.01) & (threat > 0.1) & (threat <= 0.5)
        h_ids[mask_hulk] = 1
        mask_enforcer = (threat > 0.3) & (threat <= 0.5) & ~mask_hulk
        h_ids[mask_enforcer] = 5
        type_ids[need_heuristic] = h_ids

    return type_ids


def extract_entities(
    wire_state: np.ndarray,
    max_entities: int = 128,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Extract entity set from wire state.

    Pre-allocates output arrays and writes directly — no per-entity
    allocations or Python list appends.

    Returns:
        entity_features: (max_entities, 18) float32 — padded entity features
        entity_mask: (max_entities,) bool — True for padding positions
        num_entities: int — actual number of entities found
    """
    pools_data = wire_state[_POOLS_START:]
    entity_dim = 6 + NUM_ENTITY_CLASSES  # 18

    # Pre-allocate output buffers
    features = np.zeros((max_entities, entity_dim), dtype=np.float32)
    mask = np.ones(max_entities, dtype=bool)  # True = padding
    write_idx = 0

    pool_offset = 0
    pools_len = len(pools_data)

    for pool_name, max_slots, feat_per_slot in ENTITY_POOL_DEFS:
        slot_start = pool_offset + 1
        slot_end_abs = slot_start + max_slots * feat_per_slot

        if slot_end_abs > pools_len:
            pool_offset += 1 + max_slots * feat_per_slot
            continue

        # Reshape all slots for this pool into (max_slots, feat_per_slot)
        raw = pools_data[slot_start:slot_end_abs].reshape(max_slots, feat_per_slot)

        # Occupied flag is index 0 of each slot
        occupied = raw[:, 0] > 0.5
        # Position check: skip slots at origin
        has_pos = (np.abs(raw[:, 1]) > 1e-6) | (np.abs(raw[:, 2]) > 1e-6)
        valid = occupied & has_pos
        valid_indices = np.where(valid)[0]

        if len(valid_indices) == 0:
            pool_offset += 1 + max_slots * feat_per_slot
            continue

        n_valid = len(valid_indices)
        slots = raw[valid_indices]  # (n_valid, feat_per_slot)

        # Determine how many we can write
        space = max_entities - write_idx
        if space <= 0:
            break
        n_write = min(n_valid, space)
        out = features[write_idx:write_idx + n_write]

        # Position: dx, dy at indices 1, 2
        out[:, 0] = slots[:n_write, 1]  # x
        out[:, 1] = slots[:n_write, 2]  # y
        out[:, 2] = 0.03                # w
        out[:, 3] = 0.06                # h

        # Velocity
        has_vel = pool_name in {"projectile", "danger", "human"} and feat_per_slot > 5
        if has_vel:
            out[:, 4] = slots[:n_write, 4]  # vx
            out[:, 5] = slots[:n_write, 5]  # vy
        # else: already zeros from pre-allocation

        # Type classification
        if pool_name == "danger":
            type_ids = _classify_danger_batch(slots[:n_write], feat_per_slot)
        else:
            default_type = _POOL_TYPE_MAP.get(pool_name, {}).get("default", 0)
            type_ids = np.full(n_write, default_type, dtype=np.int32)

        # One-hot encoding: set the appropriate column
        np.clip(type_ids, 0, NUM_ENTITY_CLASSES - 1, out=type_ids)
        # out[:, 6:18] already zeros; set one-hot
        out[np.arange(n_write), 6 + type_ids] = 1.0

        mask[write_idx:write_idx + n_write] = False
        write_idx += n_write

        pool_offset += 1 + max_slots * feat_per_slot

    return features, mask, write_idx


def extract_global_context(wire_state: np.ndarray) -> np.ndarray:
    """Extract the global context vector (core + ELIST features).

    Returns: (40,) float32 array
    """
    return wire_state[_CORE_START:_ELIST_END].astype(np.float32).copy()


class StateProcessor:
    """Processes raw wire states into tensors for the Set Transformer.

    Manages per-client frame stacking and converts each frame's wire
    state into (entity_features, entity_mask, global_context).
    """

    def __init__(
        self,
        max_entities: int = None,
        frame_stack: int = None,
    ):
        cfg = CONFIG.model
        self.max_entities = max_entities or cfg.max_entities
        self.frame_stack = frame_stack or cfg.frame_stack
        self.entity_dim = 6 + NUM_ENTITY_CLASSES
        self.global_dim = LEGACY_CORE_FEATURES + LEGACY_ELIST_FEATURES

    def process_frame(
        self,
        wire_state: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Process a single frame's wire state.

        Returns dict with:
          - entity_features: (max_entities, 18)
          - entity_mask: (max_entities,)
          - global_context: (40,)
          - num_entities: int
        """
        features, mask, num_ents = extract_entities(wire_state, self.max_entities)
        global_ctx = extract_global_context(wire_state)

        return {
            "entity_features": features,
            "entity_mask": mask,
            "global_context": global_ctx,
            "num_entities": num_ents,
        }

    def stack_frames(
        self,
        frame_list: list[dict[str, np.ndarray]],
    ) -> dict[str, np.ndarray]:
        """Stack T processed frames into temporal tensors.

        Args:
            frame_list: list of T dicts from process_frame()

        Returns dict with:
          - entity_features: (T, max_entities, 18)
          - entity_mask: (T, max_entities)
          - global_context: (T, 40)
        """
        T = len(frame_list)
        assert T == self.frame_stack, f"Expected {self.frame_stack} frames, got {T}"

        ent_feats = np.stack([f["entity_features"] for f in frame_list], axis=0)
        ent_masks = np.stack([f["entity_mask"] for f in frame_list], axis=0)
        global_ctx = np.stack([f["global_context"] for f in frame_list], axis=0)

        return {
            "entity_features": ent_feats,
            "entity_mask": ent_masks,
            "global_context": global_ctx,
        }

    def to_tensors(
        self,
        stacked: dict[str, np.ndarray],
        device: torch.device = None,
    ) -> dict[str, torch.Tensor]:
        """Convert stacked numpy arrays to PyTorch tensors."""
        if device is None:
            device = torch.device("cpu")

        return {
            "entity_features": torch.from_numpy(stacked["entity_features"]).float().to(device),
            "entity_mask": torch.from_numpy(stacked["entity_mask"]).bool().to(device),
            "global_context": torch.from_numpy(stacked["global_context"]).float().to(device),
        }
