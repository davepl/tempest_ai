#!/usr/bin/env python3
"""Robotron AI v3 — Potential field expert system.

Generates heuristic movement and firing commands using vector field
summation over the entity set. Used for:
  1. Behavioral cloning demonstrations during early training
  2. Safety-override actions mixed with policy output
  3. Standalone expert play for baseline measurement

Movement: force vector from repulsive (enemy) and attractive (human) fields.
Firing: priority queue — missiles > spawners > brains near humans > density.

Performance: all operations are fully vectorized with NumPy — no Python loops
over entities. Accepts pre-extracted entity arrays to avoid redundant extraction.
"""

import numpy as np
from typing import Optional
from .config import CONFIG, ExpertConfig
from .state_processor import (
    extract_entities,
    extract_global_context,
    NUM_ENTITY_CLASSES,
    _CORE_START, _ELIST_END,
)

# Type indices in our one-hot encoding (matching state_processor.py)
TYPE_GRUNT = 0
TYPE_HULK = 1
TYPE_BRAIN = 2
TYPE_TANK = 3
TYPE_SPAWNER = 4
TYPE_ENFORCER = 5
TYPE_PROJECTILE = 6
TYPE_HUMAN = 7
TYPE_ELECTRODE = 8
TYPE_MISSILE = 9
TYPE_SPARK = 10
TYPE_PROG = 11

# 8-way direction vectors (matching game's direction encoding)
_DIR_VECTORS = np.array([
    [ 0.0, -1.0],  # 0: N
    [ 0.7071, -0.7071],  # 1: NE
    [ 1.0,  0.0],  # 2: E
    [ 0.7071,  0.7071],  # 3: SE
    [ 0.0,  1.0],  # 4: S
    [-0.7071,  0.7071],  # 5: SW
    [-1.0,  0.0],  # 6: W
    [-0.7071, -0.7071],  # 7: NW
], dtype=np.float32)

# Fire-priority lookup: type_id → priority (lower = higher priority).
# Types not in this table are ignored for firing.
_FIRE_PRIORITY = np.full(NUM_ENTITY_CLASSES, 99, dtype=np.int32)
_FIRE_PRIORITY[TYPE_MISSILE] = 0
_FIRE_PRIORITY[TYPE_SPARK] = 0
_FIRE_PRIORITY[TYPE_PROJECTILE] = 0
_FIRE_PRIORITY[TYPE_SPAWNER] = 1
_FIRE_PRIORITY[TYPE_BRAIN] = 2
_FIRE_PRIORITY[TYPE_TANK] = 3
_FIRE_PRIORITY[TYPE_ENFORCER] = 4
_FIRE_PRIORITY[TYPE_GRUNT] = 5
_FIRE_PRIORITY[TYPE_PROG] = 5

# Missile-class types (for critical-radius approach check)
_MISSILE_TYPES = np.zeros(NUM_ENTITY_CLASSES, dtype=bool)
_MISSILE_TYPES[TYPE_MISSILE] = True
_MISSILE_TYPES[TYPE_SPARK] = True
_MISSILE_TYPES[TYPE_PROJECTILE] = True


def _vec_to_dir(vec: np.ndarray) -> int:
    """Convert a 2D force vector to nearest 8-way direction index (0-7) or 8 (idle)."""
    mag = np.linalg.norm(vec)
    if mag < 1e-6:
        return 8  # idle
    unit = vec / mag
    return int(np.argmax(_DIR_VECTORS @ unit))


class PotentialFieldExpert:
    """Potential field expert for Robotron.

    Computes movement via force vector summation and firing via
    priority-based target selection. All inner loops are vectorized.
    """

    def __init__(self, cfg: Optional[ExpertConfig] = None):
        self.cfg = cfg or CONFIG.expert

        # Weight lookup by type index
        self._weights = np.zeros(NUM_ENTITY_CLASSES, dtype=np.float64)
        self._weights[TYPE_GRUNT] = self.cfg.weight_grunt
        self._weights[TYPE_HULK] = self.cfg.weight_hulk
        self._weights[TYPE_BRAIN] = self.cfg.weight_brain
        self._weights[TYPE_TANK] = self.cfg.weight_tank
        self._weights[TYPE_SPAWNER] = self.cfg.weight_spawner
        self._weights[TYPE_ENFORCER] = self.cfg.weight_enforcer
        self._weights[TYPE_PROJECTILE] = self.cfg.weight_projectile
        self._weights[TYPE_HUMAN] = self.cfg.weight_human
        self._weights[TYPE_ELECTRODE] = self.cfg.weight_electrode
        self._weights[TYPE_MISSILE] = self.cfg.weight_cruise_missile
        self._weights[TYPE_SPARK] = self.cfg.weight_projectile  # same as projectile
        self._weights[TYPE_PROG] = self.cfg.weight_grunt * 1.5  # slightly more dangerous

    def get_action(
        self,
        wire_state: np.ndarray,
        max_entities: int = 128,
    ) -> tuple[int, int]:
        """Compute expert move and fire directions from raw wire state.

        Returns: (move_dir, fire_dir) — each in [0..8]
        """
        entity_features, entity_mask, num_entities = extract_entities(
            wire_state, max_entities
        )
        return self.get_action_from_entities(entity_features, entity_mask, num_entities)

    def get_action_from_entities(
        self,
        entity_features: np.ndarray,
        entity_mask: np.ndarray,
        num_entities: int,
    ) -> tuple[int, int]:
        """Compute expert move and fire from pre-extracted entities (no re-extraction)."""
        move_dir = self._compute_move(entity_features, entity_mask, num_entities)
        fire_dir = self._compute_fire(entity_features, entity_mask, num_entities)
        return move_dir, fire_dir

    def _compute_move(
        self,
        entity_features: np.ndarray,
        entity_mask: np.ndarray,
        num_entities: int,
    ) -> int:
        """Compute movement direction via vectorized force summation."""
        if num_entities == 0:
            return 8

        # Slice to active region and filter out masked (padding) entries
        active = ~entity_mask[:num_entities]
        if not active.any():
            return 8

        ents = entity_features[:num_entities][active]
        pos = ents[:, 0:2].astype(np.float64)                    # (K, 2)
        type_ids = np.argmax(ents[:, 6:6 + NUM_ENTITY_CLASSES], axis=1)  # (K,)
        weights = self._weights[type_ids]                         # (K,)
        dist = np.linalg.norm(pos, axis=1) + 1e-5                # (K,)

        # Attractive (weight > 0, e.g. humans): w * pos / dist²
        attractive = weights > 0
        # Repulsive (weight < 0, e.g. enemies): |w| * (-pos) / dist³
        repulsive = weights < 0

        force = np.zeros(2, dtype=np.float64)
        if attractive.any():
            w_a = weights[attractive, np.newaxis]           # (Ka, 1)
            p_a = pos[attractive]                           # (Ka, 2)
            d_a = dist[attractive, np.newaxis]              # (Ka, 1)
            force += (w_a * p_a / (d_a ** 2)).sum(axis=0)
        if repulsive.any():
            w_r = np.abs(weights[repulsive, np.newaxis])    # (Kr, 1)
            p_r = pos[repulsive]                            # (Kr, 2)
            d_r = dist[repulsive, np.newaxis]               # (Kr, 1)
            force += (w_r * (-p_r) / (d_r ** 3)).sum(axis=0)

        return _vec_to_dir(force.astype(np.float32))

    def _compute_fire(
        self,
        entity_features: np.ndarray,
        entity_mask: np.ndarray,
        num_entities: int,
    ) -> int:
        """Compute firing direction via vectorized priority target selection."""
        if num_entities == 0:
            return 8

        active = ~entity_mask[:num_entities]
        if not active.any():
            return 8

        ents = entity_features[:num_entities][active]
        pos = ents[:, 0:2]                                        # (K, 2)
        vel = ents[:, 4:6]                                        # (K, 2)
        type_ids = np.argmax(ents[:, 6:6 + NUM_ENTITY_CLASSES], axis=1)  # (K,)
        dist = np.linalg.norm(pos, axis=1) + 1e-5                # (K,)
        priorities = _FIRE_PRIORITY[type_ids].copy()              # (K,)

        # Priority 0 missiles: only if close AND approaching
        is_missile = _MISSILE_TYPES[type_ids]
        if is_missile.any():
            in_range = dist < self.cfg.missile_critical_radius
            # approach = -dot(pos, vel) / dist — positive means approaching
            approach = -(pos[:, 0] * vel[:, 0] + pos[:, 1] * vel[:, 1]) / (dist + 1e-5)
            # Demote missiles that are out of range or not approaching
            demote = is_missile & (~in_range | (approach <= 0))
            priorities[demote] = 99

        # Filter out entities with no firing relevance
        shootable = priorities < 99
        if not shootable.any():
            return 8

        # Select best target: min priority, then min distance
        s_pri = priorities[shootable]
        s_dist = dist[shootable]
        s_pos = pos[shootable]

        # Lexicographic argmin over (priority, distance)
        best_pri = s_pri.min()
        top_mask = s_pri == best_pri
        top_dist = s_dist.copy()
        top_dist[~top_mask] = np.inf
        best_idx = np.argmin(top_dist)

        return _vec_to_dir(s_pos[best_idx])

    def get_action_with_context(
        self,
        wire_state: np.ndarray,
        wave_number: int = 1,
        max_entities: int = 128,
    ) -> tuple[int, int]:
        """Wave-aware expert action. Adjusts behavior for specific wave types."""
        # Temporarily adjust weights based on wave
        original_weights = self._weights.copy()

        if wave_number % 5 == 0:
            self._weights[TYPE_BRAIN] *= 2.0
        if wave_number % 10 == 9:
            self._weights[TYPE_GRUNT] *= 1.5

        move_dir, fire_dir = self.get_action(wire_state, max_entities)

        self._weights = original_weights
        return move_dir, fire_dir

    def get_action_with_context_from_entities(
        self,
        entity_features: np.ndarray,
        entity_mask: np.ndarray,
        num_entities: int,
        wave_number: int = 1,
    ) -> tuple[int, int]:
        """Wave-aware expert from pre-extracted entities (no re-extraction)."""
        original_weights = self._weights.copy()

        if wave_number % 5 == 0:
            self._weights[TYPE_BRAIN] *= 2.0
        if wave_number % 10 == 9:
            self._weights[TYPE_GRUNT] *= 1.5

        move_dir, fire_dir = self.get_action_from_entities(
            entity_features, entity_mask, num_entities
        )

        self._weights = original_weights
        return move_dir, fire_dir


# Module-level singleton for convenience
_expert: Optional[PotentialFieldExpert] = None

def get_expert_action(
    wire_state: np.ndarray,
    wave_number: int = 1,
    max_entities: int = 128,
) -> tuple[int, int]:
    """Get expert action using module-level singleton."""
    global _expert
    if _expert is None:
        _expert = PotentialFieldExpert()
    return _expert.get_action_with_context(wire_state, wave_number, max_entities)


def get_expert_action_from_entities(
    entity_features: np.ndarray,
    entity_mask: np.ndarray,
    num_entities: int,
    wave_number: int = 1,
) -> tuple[int, int]:
    """Get expert action from pre-extracted entities (no re-extraction)."""
    global _expert
    if _expert is None:
        _expert = PotentialFieldExpert()
    return _expert.get_action_with_context_from_entities(
        entity_features, entity_mask, num_entities, wave_number
    )
