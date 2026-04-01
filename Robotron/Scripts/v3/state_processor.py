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


def _classify_danger_entity(features: np.ndarray) -> int:
    """Heuristic type classification for entities in the danger pool.

    Danger pool features (10): [dx, dy, vx, vy, dist, threat, approach, ttc, closest_pass, size]
    We use the threat and size hints to distinguish enemy types.
    """
    # Simple heuristic: most entities in the danger pool are grunts unless
    # they exhibit high threat (brain, tank) or zero velocity (hulk).
    if len(features) < 10:
        return 0  # grunt
    threat = features[5]
    speed = np.sqrt(features[2]**2 + features[3]**2)
    size = features[9] if len(features) > 9 else 0

    if threat > 0.8:
        return 2  # brain (highest threat)
    if threat > 0.5:
        return 3  # tank
    if speed < 0.01 and threat > 0.1:
        return 1  # hulk (slow but threatening, indestructible)
    if threat > 0.3:
        return 5  # enforcer
    return 0  # grunt


def extract_entities(
    wire_state: np.ndarray,
    max_entities: int = 128,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Extract entity set from wire state.

    Returns:
        entity_features: (max_entities, 18) float32 — padded entity features
        entity_mask: (max_entities,) bool — True for padding positions
        num_entities: int — actual number of entities found
    """
    pools_data = wire_state[_POOLS_START:]

    entities = []
    pool_offset = 0

    for pool_name, max_slots, feat_per_slot in ENTITY_POOL_DEFS:
        occupancy = int(pools_data[pool_offset]) if pool_offset < len(pools_data) else 0
        slot_start = pool_offset + 1

        for slot_idx in range(max_slots):
            feat_offset = slot_start + slot_idx * feat_per_slot
            feat_end = feat_offset + feat_per_slot

            if feat_end > len(pools_data):
                break

            slot_data = pools_data[feat_offset:feat_end]

            # Check if slot is occupied: dx,dy should be non-zero, or dist > 0
            # Slot data format: [dx, dy, vx, vy, dist, threat, approach, ...]
            if slot_idx >= occupancy:
                break

            # All pools have at least: dx, dy as first two features
            dx, dy = slot_data[0], slot_data[1]
            if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                continue  # skip empty slots

            # Reconstruct entity features for the Set Transformer
            # We normalize position as offset from player → absolute-ish coords
            x = dx  # These are already relative to player
            y = dy
            vx = slot_data[2] if feat_per_slot > 2 else 0.0
            vy = slot_data[3] if feat_per_slot > 3 else 0.0

            # Width/height: use pool-specific defaults (not in wire protocol)
            # Approximate from entity type
            w = 0.03  # ~8px / 256px
            h = 0.06  # ~16px / 256px

            # Entity type via pool classification
            if pool_name == "danger":
                type_id = _classify_danger_entity(slot_data)
            else:
                type_id = _POOL_TYPE_MAP.get(pool_name, {}).get("default", 0)

            # Build one-hot type vector
            type_onehot = np.zeros(NUM_ENTITY_CLASSES, dtype=np.float32)
            type_onehot[min(type_id, NUM_ENTITY_CLASSES - 1)] = 1.0

            entity = np.concatenate([
                np.array([x, y, w, h, vx, vy], dtype=np.float32),
                type_onehot,
            ])
            entities.append(entity)

        pool_offset += 1 + max_slots * feat_per_slot

    num_entities = len(entities)
    entity_dim = 6 + NUM_ENTITY_CLASSES  # 18

    # Pad to max_entities
    features = np.zeros((max_entities, entity_dim), dtype=np.float32)
    mask = np.ones(max_entities, dtype=bool)  # True = padding

    for i, ent in enumerate(entities[:max_entities]):
        features[i] = ent
        mask[i] = False

    return features, mask, min(num_entities, max_entities)


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
