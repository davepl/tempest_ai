# Hybrid Unified-Pool State Layout

## Motivation

The current Robotron state vector is **7204 floats** (1801 per frame × 4 stacked frames).
The entity-slot portion alone accounts for **1737 of the 1801** per-frame floats —
96% of the signal. Much of it is redundant, empty, or derivable:

| Waste source | Floats per frame | × 4 frames |
|---|---|---|
| `hit_w`, `hit_h` — constant per category, already implicit from bucket | 288 | 1152 |
| `dir_x`, `dir_y` — unit vector of `(dx, dy)`, derivable from `dx/dy/dist` | 288 | 1152 |
| ELIST bytes 23–50 — undocumented padding, almost always zero | 28 | 112 |
| Empty slots (e.g. 40 grunt slots when only 15 grunts exist) | variable | variable |

This document specifies a **hybrid unified-pool** redesign that cuts the per-frame
state from 1801 to ~760 floats (**2× reduction**), yielding a 4-frame stacked input of
~3040 vs the current 7204 — without losing any information the model actually uses.

---

## Current Layout (1801 floats/frame)

```
[14 core] [50 ELIST] [9 typed buckets × (1 occupancy + slots × 12 features)]
```

### Core Features (14 floats)

| Index | Feature | Range | Notes |
|---|---|---|---|
| 0 | alive | 0 / 1 | player alive flag |
| 1 | score | score/1M | cumulative — limited decision value |
| 2 | replay_norm | replay/99999999 | BCD field |
| 3 | lasers_norm | lasers/9 | active laser count |
| 4 | wave_norm | wave/40 | current wave number |
| 5 | player_pos_x | [0, 1] | absolute position |
| 6 | player_pos_y | [0, 1] | absolute position |
| 7 | player_vel_x | [-1, 1] | one-frame delta |
| 8 | player_vel_y | [-1, 1] | one-frame delta |
| 9 | nearest_enemy_dist | [0, 1] | global nearest |
| 10 | nearest_human_dist | [0, 1] | global nearest |
| 11 | nearest_enemy_dx | [-1, 1] | relative to player |
| 12 | nearest_enemy_dy | [-1, 1] | relative to player |
| 13 | num_humans/255 | [0, 1] | human count |

### ELIST Block (50 floats)

Bytes 1–22 are meaningful Robotron game-state counters (speeds, spawn timers,
entity counts). Bytes 23–50 are reserved/stale memory — almost always zero.

| Byte | Name | Purpose |
|---|---|---|
| 1 | robspd | Grunt speed |
| 2 | rmxspd | Grunt max speed |
| 3 | enfnum | Enforcer count (spawned by brains) |
| 4 | enstim | Enforcer spawn timer |
| 5 | cdptim | Cruise missile deploy timer |
| 6 | hlkspd | Hulk speed |
| 7 | bshtim | Brain shot timer |
| 8 | brnspd | Brain speed |
| 9 | tnksht | Tank shot timer |
| 10 | shlspd | Shell (projectile) speed |
| 11 | tdptim | Tank deploy timer |
| 12 | sqspd | Quark speed |
| 13 | robcnt | Grunt count |
| 14 | pstcnt | Prog/post count |
| 15 | momcnt | Mom count |
| 16 | dadcnt | Dad count |
| 17 | kidcnt | Kid count |
| 18 | hlkcnt | Hulk count |
| 19 | brncnt | Brain count |
| 20 | circnt | Quark count |
| 21 | sqcnt | Spawner/square count |
| 22 | tnkcnt | Tank count |
| 23–50 | (reserved) | Padding — drop |

### Typed Entity Buckets (1737 floats)

9 separate buckets, each with a 1-float occupancy header and N slots × 12 features.
Slots are stable-assigned (previous-frame pointer matching, then nearest-first fill).

| Category | Slots | Per-slot features (12) | Floats |
|---|---|---|---|
| grunt | 40 | present, dx, dy, dist, hit_w, hit_h, vx, vy, dir_x, dir_y, threat, approach | 481 |
| hulk | 16 | (same) | 193 |
| brain | 16 | (same) | 193 |
| tank | 8 | (same) | 97 |
| spawner | 8 | (same) | 97 |
| enforcer | 12 | (same) | 145 |
| projectile | 12 | (same) | 145 |
| human | 16 | (same) | 193 |
| electrode | 16 | (same) | 193 |
| **Total** | **144** | | **1737** |

---

## Proposed Layout (~760 floats/frame)

### Design Principles

1. **Unified enemy pool** — all dangerous entities (grunts, hulks, brains, tanks,
   spawners, enforcers, projectiles) share a single distance-sorted pool,
   with a `type_norm` feature to distinguish categories. Follows the Tempest
   precedent where 5 enemy types share 7 unified slots.

2. **Separate human and electrode pools** — humans (rescue targets) and
   electrodes (static obstacles) serve fundamentally different purposes than
   enemies. Mixing them into the enemy pool risks crowding them out on
   enemy-heavy waves. Small dedicated pools preserve their signal.

3. **Trimmed features** — drop `hit_w`, `hit_h` (constant per type, now encoded
   in `type_norm`), and `dir_x`, `dir_y` (derivable from `dx/dy/dist`).
   Keep the 8 features the lane encoder and cross-attention actually consume.

4. **Trimmed ELIST** — only transmit the 22 meaningful bytes, not the 28 bytes
   of padding.

5. **Stable slot assignment preserved** — the Lua stable-assignment algorithm
   (pointer-match previous frame, then nearest-fill) carries over to the
   unified pool identically.

### State Vector Structure

```
[14 core] [22 ELIST] [1 enemy_occ + 50 enemy slots × 9] [1 human_occ + 16 human slots × 7] [1 electrode_occ + 12 electrode slots × 5]
```

#### Core Features (14 floats) — unchanged

Same as current. Could optionally drop `score` and `replay_norm` (minimal
decision value), but keeping them costs only 2 floats and avoids a
protocol-breaking change for those fields.

#### ELIST (22 floats) — trimmed from 50

Only bytes 1–22 (the named game-state counters). Each normalized by `/255.0`
as before. Bytes 23–50 are dropped.

#### Enemy Pool (451 floats)

**1 occupancy** float (total enemy count / 50, clamped to [0, 1]).

**50 unified slots**, sorted by distance (nearest first), with **9 features each**:

| Index | Feature | Range | Notes |
|---|---|---|---|
| 0 | present | 0 / 1 | slot occupied |
| 1 | dx | [-1, 1] | relative x, normalized by POS_X_RANGE |
| 2 | dy | [-1, 1] | relative y, normalized by POS_Y_RANGE |
| 3 | dist | [0, 1] | Euclidean distance / POS_MAX_DIAG |
| 4 | vx | [-1, 1] | one-frame x velocity / POS_X_RANGE |
| 5 | vy | [-1, 1] | one-frame y velocity / POS_Y_RANGE |
| 6 | threat | [0, 1] | composite threat score (category weight × proximity × approach) |
| 7 | approach | [-1, 1] | radial approach velocity (positive = closing) |
| 8 | type_norm | [0, 1] | category ID / 6: grunt=0, hulk=1/6, brain=2/6, tank=3/6, spawner=4/6, enforcer=5/6, projectile=1.0 |

**Category encoding for `type_norm`**:

| Type | ID | type_norm |
|---|---|---|
| grunt | 0 | 0.000 |
| hulk | 1 | 0.167 |
| brain | 2 | 0.333 |
| tank | 3 | 0.500 |
| spawner | 4 | 0.667 |
| enforcer | 5 | 0.833 |
| projectile | 6 | 1.000 |

**Why 50 slots**: Robotron enemy peaks per category: grunt≤80, hulk≤25,
brain≤25, etc. In the hardest waves, total on-screen enemies max out around
40–50. The current 112 enemy slots (144 - 16 human - 16 electrode) include
massive headroom that creates empty-slot noise. 50 captures the nearest threats
in all realistic scenarios.

#### Human Pool (113 floats)

**1 occupancy** float (human count / 16).

**16 slots**, sorted by distance, with **7 features each**:

| Index | Feature | Range | Notes |
|---|---|---|---|
| 0 | present | 0 / 1 | slot occupied |
| 1 | dx | [-1, 1] | relative x |
| 2 | dy | [-1, 1] | relative y |
| 3 | dist | [0, 1] | distance |
| 4 | vx | [-1, 1] | velocity x (humans wander) |
| 5 | vy | [-1, 1] | velocity y |
| 6 | threat | [0, 1] | rescue urgency score (proximity-weighted) |

No `approach` — humans don't chase the player. No `type_norm` — they're all
the same type. Threat here encodes "rescue urgency" (closer = higher priority),
already computed by `_object_threat_score` with the human-specific formula:
`base × (0.35 + proximity)`.

#### Electrode Pool (61 floats)

**1 occupancy** float (electrode count / 12).

**12 slots**, sorted by distance, with **5 features each**:

| Index | Feature | Range | Notes |
|---|---|---|---|
| 0 | present | 0 / 1 | slot occupied |
| 1 | dx | [-1, 1] | relative x |
| 2 | dy | [-1, 1] | relative y |
| 3 | dist | [0, 1] | distance |
| 4 | threat | [0, 1] | proximity-weighted obstruction score |

No velocity — electrodes are static. No `type_norm` — one type. No `approach`
— they don't move. Only position and proximity matter.

### Size Summary

| Section | Current | Proposed | Saved |
|---|---|---|---|
| Core features | 14 | 14 | 0 |
| ELIST | 50 | 22 | 28 |
| Entity slots | 1737 | 625 (451 + 113 + 61) | 1112 |
| **Per frame** | **1801** | **661** | **1140** |
| **× 4 frames** | **7204** | **2644** | **4560** |

**63% reduction in total state size** (7204 → 2644).

---

## Impact on Model Components

### DirectionalLaneEncoder

The lane encoder currently extracts `dx, dy, dist, vx, vy, threat, approach,
present, is_human, is_electrode` from the slot data. The new layout provides
all of these directly:

- **Enemy pool**: `present, dx, dy, dist, vx, vy, threat, approach` — all present.
  `is_human = False, is_electrode = False` implicit from pool identity.
- **Human pool**: `present, dx, dy, dist, vx, vy, threat`.
  `is_human = True` implicit. `approach` not needed for lane binning (set to 0).
- **Electrode pool**: `present, dx, dy, dist, threat`.
  `is_electrode = True` implicit. `vx = vy = approach = 0`.

The `_build_directional_lane_tokens` method simplifies: instead of iterating
9 categories and checking names, it reads 3 pool slices directly.

### EntitySetEncoder (legacy fallback)

Currently builds 14-feature tokens: `dx, dy, vx, vy, dist, hit_w, hit_h,
dir_x, dir_y, threat, approach, category_norm, is_human, is_dangerous`.

Adapts to new layout by:
- Dropping `hit_w, hit_h, dir_x, dir_y` (no longer in state)
- Using `type_norm` from enemy pool as `category_norm`
- Adding `is_human` and `is_dangerous` from pool identity
- New token features = 10: `dx, dy, vx, vy, dist, threat, approach, type_norm, is_human, is_dangerous`

This changes `legacy_slot_token_features` from 14 → 10.

### MLP Trunk

Input shrinks from `2644 + 256 (lane summary) = 2900` to about 40% of current.
The `state_norm` LayerNorm adjusts its dimension automatically from
`__init__` parameters. Hidden layer sizes (1024 → 512 → 256) may be slightly
oversized for the smaller input but this is harmless and can be tuned later.

### Cross-Attention

Entity tokens fed to the lane encoder's cross-attention go from
`(B, 144, 14)` to `(B, 78, 10)`. This is a meaningful compute reduction
per forward pass: ~45% fewer key/value tokens.

---

## Lua Wire Protocol Changes

### Serialization

The `_stable_assign_bucket_slots` algorithm remains, but is applied to **3 pools**
instead of 9 typed buckets:

1. **Enemy pool**: Collect all classified objects where
   `CATEGORY_IS_DANGEROUS[category] == true`. Sort by `dist_norm`. Stable-assign
   into 50 slots. Emit 9 features per slot.

2. **Human pool**: Collect `category == "human"`. Sort by `dist_norm`. Stable-assign
   into 16 slots. Emit 7 features per slot.

3. **Electrode pool**: Collect `category == "electrode"`. Sort by `dist_norm`.
   Stable-assign into 12 slots. Emit 5 features per slot.

### ELIST Trimming

Change the ELIST emission loop from `for i = 1, ZP1ENM_SIZE` (50) to
`for i = 1, 22`.

### type_norm Computation

In Lua, assign each dangerous entity a category ID:

```lua
ENEMY_TYPE_ID = {
    grunt = 0, hulk = 1, brain = 2, tank = 3,
    spawner = 4, enforcer = 5, projectile = 6,
}
ENEMY_TYPE_COUNT = 7  -- for normalization denominator
```

Emit `type_norm = ENEMY_TYPE_ID[category] / (ENEMY_TYPE_COUNT - 1)` as the 9th
feature per enemy slot.

### EXPECTED_STATE_VALUES

```
14 (core) + 22 (ELIST) + 1 + 50×9 (enemies) + 1 + 16×7 (humans) + 1 + 12×5 (electrodes)
= 14 + 22 + 451 + 113 + 61
= 661
```

---

## Python Config Changes

```python
# New pool definitions (replace LEGACY_ENTITY_CATEGORIES)
HYBRID_ENEMY_SLOTS = 50
HYBRID_ENEMY_FEATURES = 9   # present, dx, dy, dist, vx, vy, threat, approach, type_norm
HYBRID_HUMAN_SLOTS = 16
HYBRID_HUMAN_FEATURES = 7   # present, dx, dy, dist, vx, vy, threat
HYBRID_ELECTRODE_SLOTS = 12
HYBRID_ELECTRODE_FEATURES = 5  # present, dx, dy, dist, threat
HYBRID_ELIST_FEATURES = 22
HYBRID_CORE_FEATURES = 14

HYBRID_PARAMS_COUNT = (
    HYBRID_CORE_FEATURES
    + HYBRID_ELIST_FEATURES
    + 1 + HYBRID_ENEMY_SLOTS * HYBRID_ENEMY_FEATURES      # 451
    + 1 + HYBRID_HUMAN_SLOTS * HYBRID_HUMAN_FEATURES       # 113
    + 1 + HYBRID_ELECTRODE_SLOTS * HYBRID_ELECTRODE_FEATURES  # 61
)  # = 661

# state_size = 661 × 4 = 2644
```

---

## Python Model Changes

### `_cat_info` Replacement

Replace the 9-entry `_cat_info` list with a 3-pool descriptor:

```python
self._pool_info = [
    ("enemy",     offset_enemy,     HYBRID_ENEMY_SLOTS,     HYBRID_ENEMY_FEATURES),
    ("human",     offset_human,     HYBRID_HUMAN_SLOTS,     HYBRID_HUMAN_FEATURES),
    ("electrode", offset_electrode, HYBRID_ELECTRODE_SLOTS, HYBRID_ELECTRODE_FEATURES),
]
```

### Token Construction

`_build_frame_object_tokens`: Read each pool, extract features, pad shorter
pools to a common feature width (10) with zeros for missing features, concatenate.

`_build_directional_lane_tokens`: Read enemy pool (dx, dy, dist, vx, vy,
threat, approach, present, is_human=False, is_electrode=False), human pool
(same with is_human=True), electrode pool (same with is_electrode=True, vx=vy=0).

### object_token_features

Changes from 14 → 10: `dx, dy, vx, vy, dist, threat, approach, type_norm,
is_human, is_dangerous`.

---

## Stable Slot Assignment

The current Lua stable-assignment algorithm works identically on a unified pool.
The key property is pointer-based identity tracking: each slot remembers which
game-object pointer was assigned to it last frame. If that pointer is still in
this frame's nearest-K set, it keeps its slot. This prevents the slot index from
thrashing when entities maintain similar proximity ranks.

For the unified enemy pool, all 7 dangerous categories contribute objects to one
sorted list. The stable-assignment machinery runs once on this combined list
with `max_slots = 50`.

For humans and electrodes, the existing per-category stable assignment runs
unchanged (just on their own smaller pools).

---

## Migration Path

This is a **full protocol-breaking change** requiring:

1. **Lua**: New `build_hybrid_state` function replacing the legacy entity emission
2. **Python config**: New constants, `params_count` → 661
3. **Python aimodel.py**: New `_cat_info` / `_pool_info`, updated token builders
4. **MODEL_ARCH_VERSION**: Increment (11 → 12)
5. **Fresh checkpoint**: No backward compatibility with old models

The old `LEGACY_*` constants should be retained (commented or gated) so the
codebase can reference them if needed for replay-buffer migration tooling.

---

## Risk Assessment

| Risk | Severity | Mitigation |
|---|---|---|
| 50 enemy slots miss entities on extreme waves | Low | Peak concurrent dangerous ≈ 40–50; 50 captures the nearest-and-most-relevant. Entities beyond slot 50 are far away and not immediate threats. |
| Loss of per-type bucket structure | Low | `type_norm` provides explicit type identity. Cross-attention can learn type-specific behaviors from this feature — proven in Tempest. |
| Dropping `hit_w, hit_h` loses collision awareness | Negligible | Hitbox dimensions are constant per category. The type_norm feature encodes category, from which hitbox is fully determined. |
| Dropping `dir_x, dir_y` loses directional signal | Negligible | `dir = (dx, dy) / dist` — 100% derivable from retained features. Network already computes this implicitly. |
| Reduced ELIST loses game state | Negligible | Bytes 23–50 are documented as reserved. Spot-checks show them as zero. |
| Human pool gets enemy-type features it doesn't need | N/A | Humans have a shorter feature set (7 vs 9). No wasted features. |
