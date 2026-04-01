#!/usr/bin/env python3
"""Robotron AI v3 — Potential field expert system.

Generates heuristic movement and firing commands using vector field
summation over the entity set. Used for:
  1. Behavioral cloning demonstrations during early training
  2. Safety-override actions mixed with policy output
  3. Standalone expert play for baseline measurement

Movement: force vector from repulsive (enemy) and attractive (human) fields.
Firing: priority queue — missiles > spawners > brains near humans > density.
"""

import math
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


def _vec_to_dir(vec: np.ndarray) -> int:
    """Convert a 2D force vector to nearest 8-way direction index (0-7) or 8 (idle)."""
    if np.linalg.norm(vec) < 1e-6:
        return 8  # idle
    # Compute angle and snap to nearest direction
    angle = math.atan2(vec[1], vec[0])
    # Normalize to [0, 2π]
    if angle < 0:
        angle += 2 * math.pi
    # Map to direction index (0=N at -π/2, rotated to game coords)
    # Game uses: 0=N(up), 1=NE, 2=E, ..., 7=NW
    # atan2 uses: 0=E, positive=S
    # Convert: game_angle = atan2(y, x), but N=-y in screen coords
    # So negate y for the angle calculation
    game_angle = math.atan2(-vec[1], vec[0])  # negate y because screen y is down
    if game_angle < 0:
        game_angle += 2 * math.pi
    # 0=E in atan2, game 0=N → rotate by π/2
    game_angle = (game_angle + math.pi / 2) % (2 * math.pi)
    # Quantize to 8 directions
    idx = int(round(game_angle / (math.pi / 4))) % 8
    return idx


class PotentialFieldExpert:
    """Potential field expert for Robotron.

    Computes movement via force vector summation and firing via
    priority-based target selection.
    """

    def __init__(self, cfg: Optional[ExpertConfig] = None):
        self.cfg = cfg or CONFIG.expert

        # Weight lookup by type index
        self._weights = np.zeros(NUM_ENTITY_CLASSES, dtype=np.float32)
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

        move_dir = self._compute_move(entity_features, entity_mask, num_entities)
        fire_dir = self._compute_fire(entity_features, entity_mask, num_entities)

        return move_dir, fire_dir

    def _compute_move(
        self,
        entity_features: np.ndarray,
        entity_mask: np.ndarray,
        num_entities: int,
    ) -> int:
        """Compute movement direction via force vector summation."""
        force = np.zeros(2, dtype=np.float64)

        for i in range(num_entities):
            if entity_mask[i]:
                continue

            ent = entity_features[i]
            pos = ent[0:2]  # dx, dy relative to player
            type_onehot = ent[6:6 + NUM_ENTITY_CLASSES]
            type_id = int(np.argmax(type_onehot))

            dist = np.linalg.norm(pos) + 1e-5
            weight = self._weights[type_id]

            if weight > 0:
                # Attractive force (toward humans)
                # Use 1/dist² for nearby humans (urgent rescue)
                force += weight * pos / (dist ** 2)
            else:
                # Repulsive force (away from enemies)
                # Use -weight * direction / dist³ (cubic falloff for strong nearby repulsion)
                direction = -pos  # away from entity
                force += abs(weight) * direction / (dist ** 3)

        # Wall avoidance: soft repulsion from boundaries
        # Player position is in core features (indices 5,6 of the global context)
        # But since entity positions are relative, we approximate wall distance
        # by checking if we're near edges (absolute pos not directly available
        # in entity features, but core features have it)
        # For simplicity, add mild center-seeking force when near walls
        # This will be refined when we integrate with the global context

        return _vec_to_dir(force.astype(np.float32))

    def _compute_fire(
        self,
        entity_features: np.ndarray,
        entity_mask: np.ndarray,
        num_entities: int,
    ) -> int:
        """Compute firing direction via priority target selection.

        Priority:
          1. Incoming missiles/sparks within critical radius
          2. Spawners (spheroids/quarks)
          3. Brains near humans
          4. Highest density of grunts
        """
        cfg = self.cfg

        # Collect entities by type with distances
        threats = []       # (priority, distance, position, type_id)

        for i in range(num_entities):
            if entity_mask[i]:
                continue
            ent = entity_features[i]
            pos = ent[0:2]
            vel = ent[4:6]
            type_onehot = ent[6:6 + NUM_ENTITY_CLASSES]
            type_id = int(np.argmax(type_onehot))
            dist = np.linalg.norm(pos) + 1e-5

            # Priority 1: incoming missiles/sparks
            if type_id in (TYPE_MISSILE, TYPE_SPARK, TYPE_PROJECTILE):
                if dist < cfg.missile_critical_radius:
                    # Check if approaching (velocity toward player)
                    approach = -np.dot(pos, vel) / (dist + 1e-5)
                    if approach > 0:
                        threats.append((0, dist, pos, type_id))
                        continue

            # Priority 2: spawners
            if type_id == TYPE_SPAWNER:
                threats.append((1, dist, pos, type_id))
                continue

            # Priority 3: brains (always high priority)
            if type_id == TYPE_BRAIN:
                threats.append((2, dist, pos, type_id))
                continue

            # Priority 4: tanks
            if type_id == TYPE_TANK:
                threats.append((3, dist, pos, type_id))
                continue

            # Priority 5: enforcers
            if type_id == TYPE_ENFORCER:
                threats.append((4, dist, pos, type_id))
                continue

            # Priority 6: grunts/progs
            if type_id in (TYPE_GRUNT, TYPE_PROG):
                threats.append((5, dist, pos, type_id))
                continue

        if not threats:
            return 8  # idle: nothing to shoot at

        # Sort by (priority, distance) — lowest priority number first, then nearest
        threats.sort(key=lambda t: (t[0], t[1]))

        # Fire at highest priority target
        target_pos = threats[0][2]
        return _vec_to_dir(target_pos)

    def get_action_with_context(
        self,
        wire_state: np.ndarray,
        wave_number: int = 1,
        max_entities: int = 128,
    ) -> tuple[int, int]:
        """Wave-aware expert action. Adjusts behavior for specific wave types.

        Brain waves (5, 10, 15, ...): extra weight on brain avoidance/targeting.
        Grunt mob waves (9, 19, 29, ...): favor perimeter movement.
        """
        # Temporarily adjust weights based on wave
        original_weights = self._weights.copy()

        if wave_number % 5 == 0:
            # Brain wave: boost brain priority
            self._weights[TYPE_BRAIN] *= 2.0
        if wave_number % 10 == 9:
            # Grunt mob: boost grunt avoidance
            self._weights[TYPE_GRUNT] *= 1.5

        move_dir, fire_dir = self.get_action(wire_state, max_entities)

        # Restore weights
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
