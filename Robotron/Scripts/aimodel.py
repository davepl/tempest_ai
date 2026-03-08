#!/usr/bin/env python3
# ==================================================================================================================
# ||  ROBOTRON AI v2 • MODEL, AGENT, AND UTILITIES                                                              ||
# ||                                                                                                              ||
# ||  Rainbow-lite with:                                                                                          ||
# ||    • Distributional C51 value estimation                                                                     ||
# ||    • Factored action heads (move dir 8 + fire dir 8 = 64 total)                                             ||
# ||    • Multi-head self-attention over 7 enemy slots                                                            ||
# ||    • Dueling architecture                                                                                     ||
# ||    • Prioritised experience replay (in replay_buffer.py)                                                     ||
# ||    • N-step returns                                                                                           ||
# ||    • Cosine-annealing LR with warm-up                                                                        ||
# ||    • Expert behavioural-cloning regulariser                                                                   ||
# ==================================================================================================================

if __name__ == "__main__":
    print("This is not the main application, run 'main.py' instead")
    exit(1)

# ── patch print to always flush ─────────────────────────────────────────────
import builtins
_original_print = builtins.print
def _flushing_print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    kwargs["end"] = kwargs.get("end", "\r\n")
    return _original_print(*args, **kwargs)
builtins.print = _flushing_print

import os, sys, time, struct, random, math, warnings, threading, queue, traceback, shutil
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

try:
    from config import SERVER_CONFIG, RL_CONFIG, MODEL_DIR, LATEST_MODEL_PATH, \
                        metrics as config_metrics, RESET_METRICS, IS_INTERACTIVE
    from training import train_step
    from replay_buffer import PrioritizedReplayBuffer
except ImportError:
    from Scripts.config import SERVER_CONFIG, RL_CONFIG, MODEL_DIR, LATEST_MODEL_PATH, \
                               metrics as config_metrics, RESET_METRICS, IS_INTERACTIVE
    from Scripts.training import train_step
    from Scripts.replay_buffer import PrioritizedReplayBuffer

sys.modules.setdefault("aimodel", sys.modules[__name__])
warnings.filterwarnings("default")

metrics = config_metrics

# ── Device selection ────────────────────────────────────────────────────────
def _cuda_device(index_hint: int) -> torch.device:
    n = torch.cuda.device_count()
    if n <= 0:
        return torch.device("cpu")
    idx = int(index_hint)
    if idx < 0 or idx >= n:
        idx = 0
    return torch.device(f"cuda:{idx}")


if torch.cuda.is_available():
    device = _cuda_device(getattr(RL_CONFIG, "train_cuda_device_index", 0))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# ── Action helpers ──────────────────────────────────────────────────────────
# Direction encoding:
#   Move: 0..7 are directions, 8 is idle/no-move (when enabled via num_move_actions=9)
#   Fire: 0..7 are directions (no idle action in policy space)
NUM_MOVE = RL_CONFIG.num_move_actions
NUM_FIRE = RL_CONFIG.num_fire_actions
NUM_JOINT = RL_CONFIG.num_joint_actions


def _clamp_game_dir(idx: int) -> int:
    i = int(idx)
    if i < 0:
        # Preserve -1 as an explicit neutral sentinel for Lua fallback paths.
        return -1
    return max(0, min(7, i))


def _move_idle_action_index() -> int:
    # Convention: if an explicit move-idle action exists, it is the final move bin.
    if NUM_MOVE >= 9:
        return NUM_MOVE - 1
    return 0


def _encode_move_to_game(move_dir: int) -> int:
    i = int(move_dir)
    if NUM_MOVE >= 9 and i == (NUM_MOVE - 1):
        return -1
    return _clamp_game_dir(i)


def _random_move_fire() -> Tuple[int, int]:
    return random.randrange(NUM_MOVE), random.randrange(NUM_FIRE)


def combine_action_indices(move_dir: int, fire_dir: int) -> int:
    move_dir = max(0, min(NUM_MOVE - 1, int(move_dir)))
    fire_dir = max(0, min(NUM_FIRE - 1, int(fire_dir)))
    return move_dir * NUM_FIRE + fire_dir


def split_joint_action(idx: int) -> Tuple[int, int]:
    idx = max(0, min(NUM_JOINT - 1, int(idx)))
    return idx // NUM_FIRE, idx % NUM_FIRE


def encode_action_to_game(move_dir: int, fire_dir: int) -> Tuple[int, int]:
    return _encode_move_to_game(move_dir), _clamp_game_dir(fire_dir)

# ── Frame data ──────────────────────────────────────────────────────────────
@dataclass
class FrameData:
    state: np.ndarray
    subjreward: float
    objreward: float
    done: bool
    save_signal: bool
    player_alive: bool
    level_number: int = 0
    game_score: int = 0
    next_replay_level: int = 0
    num_lasers: int = 0

def parse_frame_data(data: bytes) -> Optional[FrameData]:
    try:
        fmt = ">HddBIBBIBB"
        hdr = struct.calcsize(fmt)
        if not data or len(data) < hdr:
            return None
        vals = struct.unpack(fmt, data[:hdr])
        (n, subj, obj, done, score, player_alive, save, replay_level, num_lasers, wave_number) = vals
        expected_len = hdr + (int(n) * 4)
        if len(data) != expected_len:
            return None
        state = np.frombuffer(data[hdr:], dtype=">f4", count=n).astype(np.float32)
        if state.shape[0] != int(n):
            return None
        return FrameData(
            state=state, subjreward=float(subj), objreward=float(obj),
            done=bool(done), save_signal=bool(save),
            player_alive=bool(player_alive),
            level_number=int(wave_number),
            game_score=int(score),
            next_replay_level=int(replay_level),
            num_lasers=int(num_lasers),
        )
    except Exception as e:
        print(f"Parse error: {e}")
        return None


def _latest_frame_state(state: np.ndarray) -> np.ndarray:
    """Return the most recent base-frame slice from a possibly stacked state vector."""
    s = np.asarray(state, dtype=np.float32).reshape(-1)
    base_state_size = int(getattr(RL_CONFIG, "base_state_size", SERVER_CONFIG.params_count))
    if base_state_size > 0 and s.size > base_state_size:
        s = s[-base_state_size:]
    return s


# ── Expert system (heuristic) ───────────────────────────────────────────────
_DIR8_VECTORS: tuple[tuple[float, float], ...] = (
    (0.0, -1.0),                     # up
    (0.70710678, -0.70710678),       # up-right
    (1.0, 0.0),                      # right
    (0.70710678, 0.70710678),        # down-right
    (0.0, 1.0),                      # down
    (-0.70710678, 0.70710678),       # down-left
    (-1.0, 0.0),                     # left
    (-0.70710678, -0.70710678),      # up-left
)

# Lua encodes relative dx/dy with different per-axis divisors:
#   rel_x = delta_x / 34816, rel_y = delta_y / 53760
# Re-scale before direction quantisation so 8-way aiming uses true geometry.
_REL_POS_X_RANGE = 34816.0
_REL_POS_Y_RANGE = 53760.0
_POS_MAX_DIAG = 64022.0  # sqrt(34816² + 53760²)

# "Safe" distance: 1/8 screen height = 26.25 px = 6720 x16-units.
# Enemies beyond this are not an immediate threat.
_SAFE_DIST = 6720.0 / _POS_MAX_DIAG  # ~0.105

# 16 screen-pixels = 4096 x16-units, normalised per axis.
# The outer 16 px of the playfield is "lava" — never move into it.
_LAVA_X = 4096.0 / _REL_POS_X_RANGE  # ~0.118
_LAVA_Y = 4096.0 / _REL_POS_Y_RANGE  # ~0.076

# Direction index → (x_sign, y_sign) components.
_DIR8_COMPONENTS: tuple[tuple[int, int], ...] = (
    ( 0, -1),  # 0 up
    ( 1, -1),  # 1 up-right
    ( 1,  0),  # 2 right
    ( 1,  1),  # 3 down-right
    ( 0,  1),  # 4 down
    (-1,  1),  # 5 down-left
    (-1,  0),  # 6 left
    (-1, -1),  # 7 up-left
)


def _forbid_lava(move_dir: int, px: float, py: float) -> int:
    """Prevent the player from moving into the outer-16-px lava zone.

    If the chosen direction has a component pointing into a lava wall,
    that component is stripped.  If both axes are blocked (corner) the
    direction is returned unchanged — the player is already in lava and
    any movement is acceptable.
    """
    if move_dir < 0 or move_dir >= 8:
        return move_dir
    cx, cy = _DIR8_COMPONENTS[move_dir]
    block_x = (cx < 0 and px <= _LAVA_X) or (cx > 0 and px >= 1.0 - _LAVA_X)
    block_y = (cy < 0 and py <= _LAVA_Y) or (cy > 0 and py >= 1.0 - _LAVA_Y)
    if not block_x and not block_y:
        return move_dir
    if block_x and block_y:
        return move_dir  # cornered — can't make it worse
    nx = 0 if block_x else cx
    ny = 0 if block_y else cy
    if nx == 0 and ny == 0:
        return move_dir
    # Convert remaining component to a cardinal/diagonal direction.
    return _closest_dir8(float(nx), float(ny))


def _closest_dir8(vx: float, vy: float, default_dir: int = 0) -> int:
    if not np.isfinite(vx) or not np.isfinite(vy):
        return int(default_dir)
    mag2 = (vx * vx) + (vy * vy)
    if mag2 <= 1e-10:
        return int(default_dir)
    best_idx = int(default_dir)
    best_dot = -1e30
    for i, (dx, dy) in enumerate(_DIR8_VECTORS):
        dot = vx * dx + vy * dy
        if dot > best_dot:
            best_dot = dot
            best_idx = i
    return best_idx


def _nearest_enemy_vector_from_state(state: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """Return (dx, dy, dist) for nearest non-human object in typed slot state."""
    s = _latest_frame_state(state)
    if s.size < 60:
        return None

    cat_defs = getattr(RL_CONFIG, "entity_categories", [])
    if not cat_defs:
        return None

    base = 59  # 9 core + 50 ELIST
    best: Optional[Tuple[float, float, float]] = None
    for name, slots in cat_defs:
        n_slots = int(slots)
        if n_slots <= 0:
            continue
        slot_base = base + 1  # skip occupancy
        if name == "human":
            base += 1 + n_slots * 4
            continue

        # Slots are nearest-first within each category; scan until first present.
        for j in range(n_slots):
            i = slot_base + j * 4
            if i + 3 >= s.size:
                break
            present = float(s[i])
            if present < 0.5:
                continue
            dx = float(s[i + 1])
            dy = float(s[i + 2])
            dist = float(s[i + 3])
            if not np.isfinite(dist):
                # Keep scanning this category for a valid nearest entry.
                continue
            if best is None or dist < best[2]:
                best = (dx, dy, dist)
            break

        base += 1 + n_slots * 4

    return best


def _nearest_human_vector_from_state(state: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """Return (dx, dy, dist) for nearest human in typed slot state."""
    s = _latest_frame_state(state)
    if s.size < 60:
        return None

    cat_defs = getattr(RL_CONFIG, "entity_categories", [])
    if not cat_defs:
        return None

    base = 59  # 9 core + 50 ELIST
    for name, slots in cat_defs:
        n_slots = int(slots)
        if n_slots <= 0:
            continue
        slot_base = base + 1  # skip occupancy
        if name == "human":
            for j in range(n_slots):
                i = slot_base + j * 4
                if i + 3 >= s.size:
                    break
                present = float(s[i])
                if present < 0.5:
                    continue
                dx = float(s[i + 1])
                dy = float(s[i + 2])
                dist = float(s[i + 3])
                if not np.isfinite(dist):
                    continue
                return (dx, dy, dist)
            return None
        base += 1 + n_slots * 4

    return None


def get_expert_action(state: np.ndarray, locked_fire: Optional[int] = None) -> Tuple[int, int]:
    """Heuristic Robotron expert.

    Fire:  Always toward nearest enemy.
    Move:  1. Human closer than nearest enemy → move toward human.
           2. All enemies beyond safe distance → move toward human.
           3. Otherwise → move away from nearest enemy.
    """
    nearest_enemy = _nearest_enemy_vector_from_state(state)
    nearest_human = _nearest_human_vector_from_state(state)

    # ── Fire direction (always at nearest enemy) ────────────────────
    if nearest_enemy is not None:
        fx, fy, _ = nearest_enemy
        fire_dir = _closest_dir8(fx * _REL_POS_X_RANGE, fy * _REL_POS_Y_RANGE, default_dir=0)
    else:
        fire_dir = 0
    if locked_fire is not None and int(locked_fire) >= 0:
        fire_dir = max(0, min(NUM_FIRE - 1, int(locked_fire)))

    # ── Movement ────────────────────────────────────────────────────
    if nearest_enemy is not None:
        ex, ey, enemy_dist = nearest_enemy
        ex_world = ex * _REL_POS_X_RANGE
        ey_world = ey * _REL_POS_Y_RANGE

        # Human closer than nearest enemy → seek human
        if nearest_human is not None and nearest_human[2] < enemy_dist:
            hx, hy, _ = nearest_human
            move_dir = _closest_dir8(hx * _REL_POS_X_RANGE, hy * _REL_POS_Y_RANGE)
        # All enemies at safe distance → seek human
        elif enemy_dist > _SAFE_DIST and nearest_human is not None:
            hx, hy, _ = nearest_human
            move_dir = _closest_dir8(hx * _REL_POS_X_RANGE, hy * _REL_POS_Y_RANGE)
        else:
            # Flee from nearest enemy
            move_dir = _closest_dir8(-ex_world, -ey_world, default_dir=0)
    elif nearest_human is not None:
        hx, hy, _ = nearest_human
        move_dir = _closest_dir8(hx * _REL_POS_X_RANGE, hy * _REL_POS_Y_RANGE)
    else:
        move_dir = _move_idle_action_index()

    # ── Lava guard: never move into the outer 16 px ─────────────────
    s = _latest_frame_state(state)
    px = float(s[5]) if s.size > 6 else 0.5
    py = float(s[6]) if s.size > 6 else 0.5
    move_dir = _forbid_lava(move_dir, px, py)

    return move_dir, fire_dir

# ── Enemy-Slot Self-Attention ───────────────────────────────────────────────
class EnemyAttention(nn.Module):
    """Multi-head self-attention over entity slots, producing a fixed-size summary."""

    def __init__(self, slot_features: int, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed = nn.Linear(slot_features, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        # Learned query token for stronger "critical-threat" readout than mean-pooling.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.out_dim = embed_dim

    def forward(self, slots: torch.Tensor, mask: torch.Tensor = None,
                return_weights: bool = False):
        """slots: (B, num_slots, slot_features) → (B, embed_dim)
        mask: (B, num_slots) bool tensor — True = slot is EMPTY (ignored in attention).
        If return_weights=True, also returns (B, num_heads, S+1, S+1) attention weights."""
        x = self.embed(slots)                   # (B, S, D)
        x = self.norm(x)

        # Prepend a learned CLS token and read it back after self-attention.
        bsz = x.shape[0]
        cls = self.cls_token.expand(bsz, -1, -1)   # (B, 1, D)
        x = torch.cat([cls, x], dim=1)             # (B, S+1, D)

        key_padding_mask = None
        if mask is not None:
            cls_mask = torch.zeros((bsz, 1), dtype=torch.bool, device=mask.device)
            key_padding_mask = torch.cat([cls_mask, mask], dim=1)  # (B, S+1)

        attn_out, attn_weights = self.attn(
            x, x, x,
            key_padding_mask=key_padding_mask,
            average_attn_weights=False,
        )  # (B, S+1, D), (B, H, S+1, S+1)

        pooled = attn_out[:, 0, :]  # CLS output
        if return_weights:
            return pooled, attn_weights
        return pooled

# ── Lane-Cross-Attention Encoder ───────────────────────────────────────────
class LaneCrossAttentionEncoder(nn.Module):
    """
    Lane-centric spatial encoder with cross-attention from 16 tube lanes to enemy slots.

    Architecture:
      1. Lane tokens:  16 × [spike, angle, player_here, sin_pos, cos_pos] → Linear → embed
      2. Enemy tokens:  7 × [decoded(6), seg, depth, top, toprail, Δseg, Δdepth, sin, cos] → Linear → embed
      3. Cross-attention: lanes (Q) attend to enemies (K/V) with empty-slot masking
      4. Residual connection + LayerNorm on enriched lanes
      5. Mean-pool enriched lanes → fixed-size summary vector
    """

    def __init__(self, lane_features: int, enemy_features: int, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim

        # Lane embedding
        self.lane_embed = nn.Linear(lane_features, embed_dim)
        self.lane_norm = nn.LayerNorm(embed_dim)

        # Enemy embedding
        self.enemy_embed = nn.Linear(enemy_features, embed_dim)
        self.enemy_norm = nn.LayerNorm(embed_dim)

        # Cross-attention: lanes (Q) attend to enemies (K, V)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(embed_dim)

        self.out_dim = embed_dim

    def forward(self, lane_tokens: torch.Tensor, enemy_tokens: torch.Tensor,
                enemy_mask: torch.Tensor = None, return_weights: bool = False):
        """
        Args:
            lane_tokens:  (B, 16, lane_features)
            enemy_tokens: (B, 7, enemy_features)
            enemy_mask:   (B, 7) bool — True = EMPTY slot (excluded from attention)
            return_weights: if True, also return (B, num_heads, 16, 7) attention weights

        Returns:
            pooled: (B, embed_dim) — mean-pooled enriched lane representation
        """
        lane_emb = self.lane_norm(self.lane_embed(lane_tokens))      # (B, 16, D)
        enemy_emb = self.enemy_norm(self.enemy_embed(enemy_tokens))  # (B, 7, D)

        enriched, weights = self.cross_attn(
            lane_emb, enemy_emb, enemy_emb,
            key_padding_mask=enemy_mask,
            average_attn_weights=False,
        )  # (B, 16, D), (B, H, 16, 7)

        enriched = self.cross_norm(lane_emb + enriched)  # residual + norm
        pooled = enriched.mean(dim=1)                    # (B, D)

        if return_weights:
            return pooled, weights
        return pooled

# ── Distributional Dueling Network ─────────────────────────────────────────
class RainbowNet(nn.Module):
    """
    C51 distributional network with:
      - Lane-cross-attention encoder (16-lane spatial + 7-enemy cross-attention)
      - Shared trunk
      - Dueling value + advantage streams
      - Factored action heads (move direction × fire direction)
    """

    def __init__(self, state_size: int):
        super().__init__()
        cfg = RL_CONFIG
        self.state_size = state_size
        self.use_dist = cfg.use_distributional
        self.num_atoms = cfg.num_atoms if self.use_dist else 1
        self.v_min = cfg.v_min
        self.v_max = cfg.v_max
        self.use_dueling = cfg.use_dueling
        self.num_actions = NUM_JOINT

        # ── Object-Slot Self-Attention ──────────────────────────────────
        # 9 per-type entity categories with variable slots, 144 tokens total.
        # Each token gets 7 features:
        #   dx, dy, dist, ddx, ddy, category_id_norm, present.
        # dx/dy are player-relative (already in state vector from Lua).
        # Type is implicit in category position — no noisy type pointer needed.
        self.use_attn = cfg.use_enemy_attention
        attn_out_dim = 0
        if self.use_attn:
            self.base_state_size = int(getattr(cfg, "base_state_size", SERVER_CONFIG.params_count))
            self.num_object_slots = getattr(cfg, 'object_slots', 144)
            self.object_token_features = getattr(cfg, 'object_token_features', 7)
            self.slot_state_features = getattr(cfg, 'slot_state_features', 4)  # present, dx, dy, dist

            # Category layout: (slots, category_id) — must match Lua ENTITY_CATEGORIES order.
            cat_defs = getattr(cfg, 'entity_categories', [
                ("grunt", 16), ("hulk", 8), ("brain", 4), ("tank", 4),
                ("spawner", 4), ("enforcer", 8), ("projectile", 8),
                ("human", 8), ("electrode", 8),
            ])
            # Pre-compute base index in state vector for each category.
            # State: 9 core + 50 ELIST = 59, then per-category blocks.
            self._cat_info = []   # [(base, slots, cat_id), ...]
            offset = 59
            for cat_id, (name, slots) in enumerate(cat_defs):
                self._cat_info.append((offset, slots, cat_id))
                offset += 1 + slots * self.slot_state_features
            self._num_categories = len(cat_defs)

            self.object_attn = EnemyAttention(
                slot_features=self.object_token_features,
                embed_dim=cfg.attn_dim,
                num_heads=cfg.attn_heads,
            )
            attn_out_dim = cfg.attn_dim

        # ── Trunk ──────────────────────────────────────────────────────
        # Input: full state concatenated with attention output
        trunk_in = state_size + attn_out_dim
        layers = []
        for i in range(cfg.trunk_layers):
            out_dim = cfg.trunk_hidden
            layers.append(nn.Linear(trunk_in if i == 0 else cfg.trunk_hidden, out_dim))
            if cfg.use_layer_norm:
                layers.append(nn.LayerNorm(out_dim))
            layers.append(nn.ReLU())
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
        self.trunk = nn.Sequential(*layers)

        # ── Heads ──────────────────────────────────────────────────────
        head_in = cfg.trunk_hidden
        head_mid = head_in // 2

        if self.use_dueling:
            # Value stream → (num_atoms,)
            self.val_fc = nn.Linear(head_in, head_mid)
            self.val_out = nn.Linear(head_mid, self.num_atoms)
            # Advantage stream → (num_actions × num_atoms)
            self.adv_fc = nn.Linear(head_in, head_mid)
            self.adv_out = nn.Linear(head_mid, self.num_actions * self.num_atoms)
        else:
            self.q_fc = nn.Linear(head_in, head_mid)
            self.q_out = nn.Linear(head_mid, self.num_actions * self.num_atoms)

        self._init_weights()

        # Register support as buffer (not a parameter)
        if self.use_dist:
            support = torch.linspace(self.v_min, self.v_max, self.num_atoms)
            self.register_buffer("support", support)
            self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)

    def _build_object_tokens(self, state: torch.Tensor):
        """Build 144 object tokens from 9 per-type entity categories.

        Each token gets 7 features:
          [dx, dy, dist, ddx, ddy, category_id_norm, present]
        where ddx/ddy are frame-to-frame deltas from the previous stacked frame.

        Slots are sorted by distance (nearest first) per category.
        Type is implicit in category position — no noisy type pointer.

        Returns:
          tokens: (B, 144, 7)
          mask:   (B, 144) bool — True where slot is EMPTY
        """
        B = state.shape[0]
        device = state.device
        base_state_size = max(1, int(getattr(self, "base_state_size", SERVER_CONFIG.params_count)))
        total_dim = int(state.shape[1])
        stack_depth = max(1, total_dim // base_state_size)
        latest_off = (stack_depth - 1) * base_state_size
        prev_off = (stack_depth - 2) * base_state_size if stack_depth >= 2 else latest_off

        all_tokens = []
        all_masks = []

        for base, slots, cat_id in self._cat_info:
            # Each category: [occupancy, slot0_present, slot0_dx, slot0_dy, slot0_dist, ...]
            # Skip occupancy (+1), extract slots × 4 state features
            latest = state[:, latest_off + base + 1 : latest_off + base + 1 + slots * self.slot_state_features]
            latest = latest.reshape(B, slots, self.slot_state_features)  # (B, slots, 4)
            prev = state[:, prev_off + base + 1 : prev_off + base + 1 + slots * self.slot_state_features]
            prev = prev.reshape(B, slots, self.slot_state_features)      # (B, slots, 4)

            present = latest[:, :, 0]            # (B, slots) — used as mask
            dx      = latest[:, :, 1:2]          # (B, slots, 1) — relative to player
            dy      = latest[:, :, 2:3]          # (B, slots, 1) — relative to player
            dist    = latest[:, :, 3:4]          # (B, slots, 1)

            prev_present = prev[:, :, 0]
            prev_dx = prev[:, :, 1:2]
            prev_dy = prev[:, :, 2:3]
            vel_valid = ((present > 0.5) & (prev_present > 0.5)).unsqueeze(2).to(dtype=state.dtype)
            ddx = (dx - prev_dx) * vel_valid
            ddy = (dy - prev_dy) * vel_valid

            # Category identity as normalised scalar
            cat_norm = torch.full((B, slots, 1),
                                  cat_id / max(1, self._num_categories - 1),
                                  device=device, dtype=state.dtype)

            # Include present flag so attention can see slot occupancy
            tokens = torch.cat([dx, dy, dist, ddx, ddy, cat_norm, present.unsqueeze(2)], dim=2)  # (B, slots, 7)
            all_tokens.append(tokens)
            all_masks.append(present < 0.5)  # True = empty slot

        tokens = torch.cat(all_tokens, dim=1)   # (B, 144, 7)
        mask = torch.cat(all_masks, dim=1)       # (B, 144)

        # If ALL slots empty (e.g. between waves), unmask all to avoid NaN
        all_empty = mask.all(dim=1, keepdim=True)
        mask = mask & ~all_empty

        return tokens, mask

    def forward(self, state: torch.Tensor, log: bool = False):
        """
        Returns:
          - If distributional: (B, num_actions, num_atoms) log-probabilities or probabilities
          - If scalar: (B, num_actions) Q-values
        """
        B = state.shape[0]

        # Object-slot self-attention
        if self.use_attn:
            obj_tokens, obj_mask = self._build_object_tokens(state)  # (B, 144, 7), (B, 144)
            attn_out = self.object_attn(obj_tokens, obj_mask)       # (B, attn_dim)
            trunk_in = torch.cat([state, attn_out], dim=1)
        else:
            trunk_in = state

        h = self.trunk(trunk_in)

        if self.use_dueling:
            val = F.relu(self.val_fc(h))
            val = self.val_out(val).view(B, 1, self.num_atoms)
            adv = F.relu(self.adv_fc(h))
            adv = self.adv_out(adv).view(B, self.num_actions, self.num_atoms)
            q_atoms = val + adv - adv.mean(dim=1, keepdim=True)
        else:
            q = F.relu(self.q_fc(h))
            q_atoms = self.q_out(q).view(B, self.num_actions, self.num_atoms)

        if self.use_dist:
            # Keep distribution logits math in fp32 even under autocast to
            # avoid fp16 underflow/overflow in softmax/log_softmax.
            q_atoms = q_atoms.float()
            if log:
                return F.log_softmax(q_atoms, dim=2)
            else:
                return F.softmax(q_atoms, dim=2)
        else:
            return q_atoms.squeeze(2)  # (B, num_actions)

    def q_values(self, state: torch.Tensor) -> torch.Tensor:
        """Compute expected Q-values: (B, num_actions)."""
        if self.use_dist:
            probs = self.forward(state, log=False)         # (B, A, N)
            return (probs * self.support.unsqueeze(0).unsqueeze(0)).sum(dim=2)
        else:
            return self.forward(state, log=False)

# ── Keyboard handler ────────────────────────────────────────────────────────
msvcrt = termios = tty = fcntl = None
if sys.platform == "win32":
    try:
        import msvcrt
    except ImportError:
        pass
elif sys.platform in ("linux", "darwin"):
    try:
        import termios, tty, fcntl
    except ImportError:
        pass

import select as _select

class KeyboardHandler:
    def __init__(self):
        self.platform = sys.platform
        self.fd = None
        self.old_settings = None
        if not IS_INTERACTIVE:
            return
        if self.platform in ("linux", "darwin") and termios:
            try:
                self.fd = sys.stdin.fileno()
                self.old_settings = termios.tcgetattr(self.fd)
            except Exception:
                self.fd = None

    def setup_terminal(self):
        if self.platform in ("linux", "darwin") and self.fd is not None and tty and fcntl:
            try:
                tty.setraw(self.fd)
                flags = fcntl.fcntl(self.fd, fcntl.F_GETFL)
                fcntl.fcntl(self.fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
            except Exception:
                pass

    def __enter__(self):
        self.setup_terminal()
        return self

    def __exit__(self, *a):
        self.restore_terminal()

    def check_key(self):
        if not IS_INTERACTIVE:
            return None
        try:
            if self.platform == "win32" and msvcrt:
                if msvcrt.kbhit():
                    return msvcrt.getch().decode("utf-8")
            elif self.platform in ("linux", "darwin") and self.fd is not None:
                if _select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                    return sys.stdin.read(1)
        except Exception:
            pass
        return None

    def restore_terminal(self):
        if self.platform in ("linux", "darwin") and self.fd is not None and termios:
            try:
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)
            except Exception:
                pass

    def set_raw_mode(self):
        if self.platform in ("linux", "darwin") and self.fd is not None and tty:
            try:
                tty.setraw(self.fd)
            except Exception:
                pass

def print_with_terminal_restore(kb, *args, **kwargs):
    if IS_INTERACTIVE and kb and kb.platform in ("linux", "darwin"):
        kb.restore_terminal()
    try:
        # Large outputs can overflow the non-blocking stdout buffer.
        # Print line-by-line with short sleeps to let the buffer drain.
        text = " ".join(str(a) for a in args)
        import time as _time
        for line in text.split("\n"):
            for attempt in range(5):
                try:
                    print(line, **kwargs, flush=True)
                    break
                except BlockingIOError:
                    _time.sleep(0.05)
    except Exception:
        pass
    if IS_INTERACTIVE and kb and kb.platform in ("linux", "darwin"):
        kb.set_raw_mode()

# ── SafeMetrics wrapper (used by socket_server) ────────────────────────────
class SafeMetrics:
    def __init__(self, m):
        self.metrics = m
        self.lock = threading.Lock()

    def update_frame_count(self, delta=1):
        self.metrics.update_frame_count(delta)

    def add_episode_reward(self, total, dqn, expert, subj=None, obj=None, length=0):
        self.metrics.add_episode_reward(total, dqn, expert, subj, obj, length)

    def update_epsilon(self):
        return self.metrics.update_epsilon()

    def update_expert_ratio(self):
        return self.metrics.update_expert_ratio()

    def get_effective_epsilon(self):
        return self.metrics.get_effective_epsilon()

    def get_expert_ratio(self):
        return self.metrics.get_expert_ratio()

    def increment_total_controls(self):
        self.metrics.increment_total_controls()

    def add_inference_time(self, t):
        self.metrics.add_inference_time(t)

    def update_game_state(self, e, o):
        pass

    @property
    def peak_game_score(self):
        return self.metrics.peak_game_score

    @peak_game_score.setter
    def peak_game_score(self, v):
        self.metrics.peak_game_score = v

    def add_game_score(self, score):
        self.metrics.add_game_score(score)

    @property
    def episodes_this_run(self):
        return self.metrics.episodes_this_run

# ── Agent ───────────────────────────────────────────────────────────────────
class RainbowAgent:
    """Rainbow-lite agent with factored actions, C51, PER, n-step, attention."""

    def __init__(self, state_size: int):
        self.state_size = state_size
        self.device = device
        cfg = RL_CONFIG
        self.factored_greedy_action = bool(getattr(cfg, "factored_greedy_action", False))

        # Counters and locks (must be created before _sync_inference)
        self.training_steps = 0
        self.loaded_training_steps = 0
        self.last_inference_sync = 0
        self._sync_lock = threading.Lock()
        self.training_enabled = True
        self.running = True

        # Networks
        self.online_net = RainbowNet(state_size).to(self.device)
        self.target_net = RainbowNet(state_size).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        self.online_net.train()

        # Inference model (optionally on CPU for non-blocking frame serving)
        self.use_separate_inference = cfg.use_separate_inference_model
        if cfg.inference_on_cpu:
            infer_dev = torch.device("cpu")
        elif torch.cuda.is_available():
            infer_dev = _cuda_device(getattr(cfg, "inference_cuda_device_index", 0))
        else:
            infer_dev = self.device
        self.inference_device = infer_dev

        # ── CUDA streams for overlapping training & inference ───────
        # Training uses the default stream; inference gets a dedicated
        # stream so forward passes on infer_net can overlap with
        # backprop on online_net.  A CUDA event gates weight sync so
        # inference never reads a partially-copied state dict.
        self._inference_stream: torch.cuda.Stream | None = None
        self._sync_event: torch.cuda.Event | None = None
        if (
            self.use_separate_inference
            and infer_dev.type == "cuda"
            and self.device.type == "cuda"
            and infer_dev.index == self.device.index
        ):
            self._inference_stream = torch.cuda.Stream(device=infer_dev)
            self._sync_event = torch.cuda.Event()

        if self.use_separate_inference:
            self.infer_net = RainbowNet(state_size).to(infer_dev)
            self.infer_net.eval()
            self._sync_inference(force=True)
        else:
            self.infer_net = self.online_net

        _stream_info = f", inference_stream={'yes' if self._inference_stream else 'no'}"
        print(
            f"Agent devices: train={self.device}, infer={self.inference_device}, "
            f"separate_infer={self.use_separate_inference}{_stream_info}"
        )

        # Optimizer
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=cfg.lr, eps=1.5e-4)

        # Replay
        use_memmap_replay = bool(
            getattr(cfg, "replay_use_memmap_storage",
                    getattr(cfg, "replay_use_mmap_persistence", True))
        )
        replay_memmap_dir = None
        if use_memmap_replay:
            configured_dir = str(getattr(cfg, "replay_memmap_dir", "") or "").strip()
            replay_memmap_dir = configured_dir or (LATEST_MODEL_PATH.rsplit(".", 1)[0] + "_replay")
        self.memory = PrioritizedReplayBuffer(
            capacity=cfg.memory_size,
            state_size=state_size,
            alpha=cfg.priority_alpha,
            memmap_dir=replay_memmap_dir,
        )
        if replay_memmap_dir:
            print(f"Replay storage: memmap ({replay_memmap_dir})")

        # AMP
        self.use_amp = cfg.enable_amp and (self.device.type == "cuda")
        self.amp_dtype = torch.float16
        if self.use_amp and self.device.type == "cuda":
            try:
                if torch.cuda.is_bf16_supported():
                    self.amp_dtype = torch.bfloat16
            except Exception:
                self.amp_dtype = torch.float16
        # GradScaler is needed only for fp16; bf16 does not require scaling.
        self.grad_scaler = None
        if self.use_amp and self.amp_dtype == torch.float16:
            try:
                self.grad_scaler = torch.amp.GradScaler("cuda", enabled=True)
            except Exception:
                self.grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
        if self.use_amp:
            print(f"AMP enabled (dtype={self.amp_dtype})")
        else:
            print("AMP disabled")

        # Background training thread
        self._train_queue = queue.Queue(maxsize=8)
        self._train_thread = threading.Thread(target=self._background_train, daemon=True, name="TrainWorker")
        self._train_thread.start()

    # ── LR schedule ─────────────────────────────────────────────────────
    def get_lr(self) -> float:
        cfg = RL_CONFIG
        step = self.training_steps
        if step < cfg.lr_warmup_steps:
            return cfg.lr * (step + 1) / max(1, cfg.lr_warmup_steps)
        decay_horizon = max(1, cfg.lr_cosine_period)
        if bool(getattr(cfg, "lr_use_restarts", False)):
            t = (step - cfg.lr_warmup_steps) % decay_horizon
        else:
            # Monotonic cosine decay: reach lr_min, then stay there.
            t = min(step - cfg.lr_warmup_steps, decay_horizon)
        cosine = 0.5 * (1.0 + math.cos(math.pi * t / decay_horizon))
        return cfg.lr_min + (cfg.lr - cfg.lr_min) * cosine

    def _update_lr(self):
        lr = self.get_lr()
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    # ── Inference ───────────────────────────────────────────────────────
    def _sync_inference(self, force=False):
        if not self.use_separate_inference:
            return
        if not force and (self.training_steps - self.last_inference_sync < RL_CONFIG.inference_sync_steps):
            return
        with self._sync_lock:
            same_cuda_device = (
                self.device.type == "cuda"
                and self.inference_device.type == "cuda"
                and self.device.index == self.inference_device.index
            )
            if self.inference_device.type == "cpu":
                sd = {k: v.detach().cpu() for k, v in self.online_net.state_dict().items()}
            elif same_cuda_device:
                # Copy weights on the default (training) stream, then record
                # an event so the inference stream knows the copy is done.
                sd = self.online_net.state_dict()
            else:
                sd = {k: v.detach().to(self.inference_device) for k, v in self.online_net.state_dict().items()}
            self.infer_net.load_state_dict(sd, strict=False)
            self.infer_net.eval()
            self.last_inference_sync = self.training_steps
            # Signal inference stream that new weights are ready
            if self._sync_event is not None:
                self._sync_event.record()  # recorded on default stream

    def act(self, state: np.ndarray, epsilon: float, locked_fire: Optional[int] = None) -> Tuple[int, int, bool]:
        """Return (move_dir_idx, fire_dir_idx, is_epsilon).

        Move and fire independently undergo epsilon-random exploration,
        so the player can randomly explore firing while making a greedy
        move, or vice versa.

        If locked_fire >= 0, only the move axis is decided; fire is fixed
        to locked_fire (used by fire-hold cadence in the socket server).
        """
        lock_fire = None
        if locked_fire is not None:
            lf = int(locked_fire)
            if lf >= 0:
                lock_fire = max(0, min(NUM_FIRE - 1, lf))

        rand_move = random.random() < epsilon
        if lock_fire is not None:
            if rand_move:
                return random.randrange(NUM_MOVE), lock_fire, True
            st = torch.from_numpy(state).float().unsqueeze(0).to(self.inference_device)
            q = self._infer_q_values(st)
            # For move-only greedy selection, maximize over fire axis.
            q_joint = q.view(-1, NUM_MOVE, NUM_FIRE)
            move_scores = q_joint.max(dim=2).values
            greedy_move = int(move_scores.argmax(dim=1)[0].item())
            return greedy_move, lock_fire, False

        rand_fire = random.random() < epsilon

        if rand_move and rand_fire:
            return random.randrange(NUM_MOVE), random.randrange(NUM_FIRE), True

        # Need greedy action for at least one axis
        st = torch.from_numpy(state).float().unsqueeze(0).to(self.inference_device)
        q = self._infer_q_values(st)
        if self.factored_greedy_action:
            greedy_move_t, greedy_fire_t = self._greedy_axes_from_q(q)
            greedy_move = int(greedy_move_t[0].item())
            greedy_fire = int(greedy_fire_t[0].item())
        else:
            joint = int(q.argmax(dim=1).item())
            greedy_move, greedy_fire = split_joint_action(joint)

        move_idx = random.randrange(NUM_MOVE) if rand_move else greedy_move
        fire_idx = random.randrange(NUM_FIRE) if rand_fire else greedy_fire
        return move_idx, fire_idx, rand_move or rand_fire

    @staticmethod
    def _greedy_axes_from_q(q_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode per-axis greedy actions from a joint Q-table.

        Args:
            q_values: (B, NUM_MOVE*NUM_FIRE)
        Returns:
            move_idx: (B,) long
            fire_idx: (B,) long
        """
        q_joint = q_values.view(-1, NUM_MOVE, NUM_FIRE)
        move_scores = q_joint.max(dim=2).values  # (B, NUM_MOVE)
        fire_scores = q_joint.max(dim=1).values  # (B, NUM_FIRE)
        return move_scores.argmax(dim=1), fire_scores.argmax(dim=1)

    def _infer_q_values(self, states_t: torch.Tensor) -> torch.Tensor:
        net = self.infer_net if self.use_separate_inference else self.online_net
        net.eval()
        with torch.no_grad():
            if self._inference_stream is not None:
                # Wait for any in-flight weight sync to finish, then run
                # the forward pass on the dedicated inference stream.
                self._inference_stream.wait_event(self._sync_event)
                with torch.cuda.stream(self._inference_stream):
                    return net.q_values(states_t)
            elif self.use_separate_inference:
                with self._sync_lock:
                    return net.q_values(states_t)
            return net.q_values(states_t)

    def act_batch(
        self,
        states: list[np.ndarray],
        epsilons: list[float],
        locked_fires: Optional[list[Optional[int]]] = None,
    ) -> list[Tuple[int, int, bool]]:
        """Return batched actions for aligned state/epsilon lists.
        Each element is (move_dir_idx, fire_dir_idx, is_epsilon).

        Move and fire independently undergo epsilon-random exploration unless
        a per-element locked_fires entry is provided (>=0), in which case
        fire is held fixed and only move is selected.
        """
        n = min(len(states), len(epsilons))
        if n <= 0:
            return []
        if locked_fires is None:
            lock_list: list[Optional[int]] = [None] * n
        else:
            lock_list = list(locked_fires[:n])
            if len(lock_list) < n:
                lock_list.extend([None] * (n - len(lock_list)))

        # Independent epsilon flips for move and fire per element
        rand_moves = [False] * n
        rand_fires = [False] * n
        rnd_move_vals = [0] * n
        rnd_fire_vals = [0] * n
        fire_fixed = [False] * n
        fire_fixed_vals = [0] * n
        greedy_idx: list[int] = []
        greedy_states: list[np.ndarray] = []

        for i in range(n):
            eps = float(epsilons[i])
            lf = lock_list[i]
            if lf is not None:
                lf_i = int(lf)
                if lf_i >= 0:
                    fire_fixed[i] = True
                    fire_fixed_vals[i] = max(0, min(NUM_FIRE - 1, lf_i))
            rand_moves[i] = random.random() < eps
            if rand_moves[i]:
                rnd_move_vals[i] = random.randrange(NUM_MOVE)
            if fire_fixed[i]:
                rand_fires[i] = False
                rnd_fire_vals[i] = fire_fixed_vals[i]
                needs_greedy = not rand_moves[i]  # need greedy move only
            else:
                rand_fires[i] = random.random() < eps
                if rand_fires[i]:
                    rnd_fire_vals[i] = random.randrange(NUM_FIRE)
                # Need inference whenever at least one axis is greedy
                needs_greedy = not (rand_moves[i] and rand_fires[i])
            if needs_greedy:
                greedy_idx.append(i)
                greedy_states.append(states[i])

        greedy_actions: dict[int, Tuple[int, int]] = {}
        if greedy_idx:
            batch_np = np.asarray(greedy_states, dtype=np.float32)
            st = torch.from_numpy(batch_np).to(self.inference_device)
            q = self._infer_q_values(st)
            gm_t, gf_t = self._greedy_axes_from_q(q)
            gm = gm_t.detach().cpu().tolist()
            gf = gf_t.detach().cpu().tolist()
            if self.factored_greedy_action:
                for pos, m, f in zip(greedy_idx, gm, gf):
                    greedy_actions[pos] = (int(m), int(f))
            else:
                joints = q.argmax(dim=1).detach().cpu().tolist()
                for k, (pos, joint) in enumerate(zip(greedy_idx, joints)):
                    if fire_fixed[pos]:
                        # When fire is locked, move should still be greedy over
                        # fire axis rather than tied to a joint argmax fire.
                        greedy_actions[pos] = (int(gm[k]), int(gf[k]))
                    else:
                        greedy_actions[pos] = split_joint_action(int(joint))

        actions: list[Tuple[int, int, bool]] = []
        for i in range(n):
            g_move, g_fire = greedy_actions.get(i, (0, 0))
            m = rnd_move_vals[i] if rand_moves[i] else g_move
            if fire_fixed[i]:
                f = fire_fixed_vals[i]
            else:
                f = rnd_fire_vals[i] if rand_fires[i] else g_fire
            actions.append((m, f, rand_moves[i] or rand_fires[i]))

        return actions

    # ── Step (add experience) ───────────────────────────────────────────
    def step(self, state, action, reward, next_state, done, actor="dqn", horizon=1, priority_reward=None):
        if isinstance(action, (tuple, list)) and len(action) >= 2:
            action_idx = combine_action_indices(action[0], action[1])
        else:
            action_idx = int(max(0, min(NUM_JOINT - 1, int(action))))
        is_expert = 1 if actor == "expert" else 0
        pri = float(priority_reward) if priority_reward is not None else 0.0
        # Ensure terminal transitions get a minimum priority floor
        if done:
            boost = float(getattr(RL_CONFIG, 'death_priority_boost', 0.0))
            if boost > 0:
                pri = max(abs(pri), boost) * (-1.0 if pri < 0 else 1.0)
        self.memory.add(state, action_idx, float(reward), next_state, bool(done), int(horizon), is_expert, priority_hint=pri)
        # Return the index of the just-written transition for pre-death tracking
        try:
            return int(self.memory.tree.data_ptr - 1) % self.memory.capacity
        except AttributeError:
            return -1

    # ── Background training ─────────────────────────────────────────────
    def _background_train(self):
        pending_batch = None                  # prefetched batch for next step
        while self.running:
            try:
                # Check for stop signal
                try:
                    tok = self._train_queue.get_nowait()
                    if tok is None:
                        break
                except queue.Empty:
                    pass

                if not self.training_enabled or not getattr(metrics, "training_enabled", True):
                    pending_batch = None
                    time.sleep(0.01)
                    continue

                did = False
                for _ in range(RL_CONFIG.training_steps_per_cycle):
                    loss = train_step(self, prefetched_batch=pending_batch)
                    pending_batch = None      # consumed
                    if loss is None:
                        break
                    did = True
                    # Prefetch next batch while GPU may still be finishing
                    pending_batch = self._prefetch_batch()
                if not did:
                    pending_batch = None
                    time.sleep(0.002)
            except Exception as e:
                pending_batch = None
                print(f"Training error: {e}")
                traceback.print_exc()
                time.sleep(0.1)

    def _prefetch_batch(self):
        """Pre-sample a batch from replay so it's ready for the next step."""
        try:
            if len(self.memory) < max(RL_CONFIG.min_replay_to_train, RL_CONFIG.batch_size):
                return None
            from training import _beta_schedule
            beta = _beta_schedule(metrics.frame_count)
            return self.memory.sample(RL_CONFIG.batch_size, beta=beta)
        except Exception:
            return None

    # ── Target update ───────────────────────────────────────────────────
    def update_target(self, tau: float = None):
        if tau is None:
            tau = RL_CONFIG.target_tau
        if tau >= 1.0:
            # Hard copy
            self.target_net.load_state_dict(self.online_net.state_dict())
        else:
            # Polyak (soft) averaging: target = (1-tau)*target + tau*online
            for tp, op in zip(self.target_net.parameters(), self.online_net.parameters()):
                tp.data.mul_(1.0 - tau).add_(op.data, alpha=tau)
        self.target_net.eval()
        try:
            metrics.last_target_update_step = metrics.total_training_steps
            metrics.last_target_update_time = time.time()
        except Exception:
            pass

    # ── Save / Load ─────────────────────────────────────────────────────
    @staticmethod
    def _load_compatible(model, ckpt_sd):
        """Load state dict, silently skipping keys with shape mismatches."""
        model_sd = model.state_dict()
        compatible = {}
        skipped = []
        for k, v in ckpt_sd.items():
            if k in model_sd:
                if model_sd[k].shape == v.shape:
                    compatible[k] = v
                else:
                    skipped.append(f"{k}: {tuple(v.shape)} → {tuple(model_sd[k].shape)}")
        if skipped:
            print(f"  Skipped {len(skipped)} shape-mismatched keys:")
            for s in skipped[:5]:
                print(f"    {s}")
            if len(skipped) > 5:
                print(f"    ... and {len(skipped) - 5} more")
        return model.load_state_dict(compatible, strict=False)

    @staticmethod
    def _text_progress(label: str, frac: float, width: int = 24):
        frac_clamped = max(0.0, min(1.0, float(frac)))
        filled = int(round(frac_clamped * width))
        bar = "#" * filled + "-" * (width - filled)
        sys.stdout.write(f"\r{label} [{bar}] {frac_clamped * 100.0:5.1f}%")
        sys.stdout.flush()
        if frac_clamped >= 1.0:
            sys.stdout.write("\n")
            sys.stdout.flush()

    def save(self, filepath, is_forced_save=False, show_status=True):
        try:
            with metrics.lock:
                fc = int(metrics.frame_count)
                ts = int(metrics.total_training_steps)
                er = float(metrics.expert_ratio)
                ep = float(metrics.epsilon)
        except Exception:
            fc, ts, er, ep = 0, self.training_steps, RL_CONFIG.expert_ratio_start, RL_CONFIG.epsilon_start

        ckpt = {
            "online_state_dict": self.online_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_steps": self.training_steps,
            "frame_count": fc,
            "total_training_steps": ts,
            "expert_ratio": er,
            "epsilon": ep,
            "engine_version": 2,
        }
        if hasattr(self, "grad_scaler") and self.grad_scaler is not None:
            ckpt["grad_scaler_state_dict"] = self.grad_scaler.state_dict()
        if show_status:
            self._text_progress("  Model save", 0.0)

        # Backup existing checkpoint before overwriting
        if os.path.exists(filepath):
            try:
                shutil.copy2(filepath, filepath + ".bak")
            except Exception as e:
                print(f"  [WARN] Backup copy failed: {e}")

        # Atomic save: write to .tmp then rename
        tmp_path = filepath + ".tmp"
        torch.save(ckpt, tmp_path)
        os.replace(tmp_path, filepath)

        if show_status:
            self._text_progress("  Model save", 1.0)
        if is_forced_save and show_status:
            print(f"Model saved to {filepath}")

        # Save replay buffer alongside the model (directory format)
        buf_path = filepath.rsplit(".", 1)[0] + "_replay"
        try:
            self.memory.save(buf_path, verbose=bool(show_status))
        except Exception as e:
            print(f"  Replay buffer save failed: {e}")

    def load(self, filepath, show_status=True) -> bool:
        if not os.path.exists(filepath):
            return False
        try:
            if show_status:
                self._text_progress("  Model load", 0.0)
            ckpt = torch.load(filepath, map_location=self.device, weights_only=False)
            if show_status:
                self._text_progress("  Model load", 1.0)

            # Detect old engine (v1) checkpoints
            if "engine_version" not in ckpt:
                print("⚠  Old engine checkpoint detected — starting fresh with new architecture.")
                return False

            m1, u1 = self._load_compatible(self.online_net, ckpt.get("online_state_dict", {}))
            m2, u2 = self._load_compatible(self.target_net,
                ckpt.get("target_state_dict", ckpt.get("online_state_dict", {})))

            opt_sd = ckpt.get("optimizer_state_dict")
            if opt_sd:
                try:
                    self.optimizer.load_state_dict(opt_sd)
                except Exception as e:
                    print(f"Optimizer state skipped: {e}")

            gs_sd = ckpt.get("grad_scaler_state_dict")
            if gs_sd and hasattr(self, "grad_scaler") and self.grad_scaler is not None:
                try:
                    self.grad_scaler.load_state_dict(gs_sd)
                except Exception as e:
                    print(f"GradScaler state skipped: {e}")

            self.training_steps = ckpt.get("training_steps", 0)
            self.loaded_training_steps = self.training_steps
            self._sync_inference(force=True)

            if m1 or u1 or m2 or u2:
                print(f"Partial load (missing={len(m1)}, unexpected={len(u1)})")

            try:
                with metrics.lock:
                    ckpt_expert_ratio = float(ckpt.get("expert_ratio", RL_CONFIG.expert_ratio_start))
                    if not math.isfinite(ckpt_expert_ratio):
                        ckpt_expert_ratio = RL_CONFIG.expert_ratio_start
                    ckpt_expert_ratio = max(0.0, min(1.0, ckpt_expert_ratio))
                    if not RESET_METRICS:
                        metrics.expert_ratio = ckpt_expert_ratio
                        metrics.epsilon = ckpt.get("epsilon", RL_CONFIG.epsilon_start)
                        metrics.frame_count = int(ckpt.get("frame_count", 0))
                        metrics.loaded_frame_count = metrics.frame_count
                        metrics.total_training_steps = int(ckpt.get("total_training_steps", self.training_steps))
                    else:
                        metrics.expert_ratio = RL_CONFIG.expert_ratio_start
                        metrics.epsilon = RL_CONFIG.epsilon_start
                        metrics.frame_count = 0
                        metrics.loaded_frame_count = 0
                        metrics.total_training_steps = self.training_steps
            except Exception:
                pass

            print(f"Loaded v2 model from {filepath}")

            # Load replay buffer if present alongside the model
            buf_path = filepath.rsplit(".", 1)[0] + "_replay"
            try:
                if not self.memory.load(buf_path, verbose=bool(show_status)):
                    print("  No replay buffer found — starting with empty buffer.")
            except Exception as e:
                print(f"  Replay buffer load failed: {e}")

            return True
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            traceback.print_exc()
            return False

    def flush_replay_buffer(self):
        """Clear the entire replay buffer."""
        self.memory.flush()

    def reset_attention_weights(self):
        """Reinitialize only the object self-attention weights, keeping trunk and heads intact."""
        if not self.online_net.use_attn:
            print("No attention layer to reset.")
            return
        for net in (self.online_net, self.target_net):
            for m in net.object_attn.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=1.0)
                    nn.init.constant_(m.bias, 0.0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0.0)
        self._sync_inference(force=True)
        # Reset optimizer state for attention parameters so momentum doesn't carry old bias
        attn_param_ids = {id(p) for p in self.online_net.object_attn.parameters()}
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if id(p) in attn_param_ids and p in self.optimizer.state:
                    del self.optimizer.state[p]
        print("Object self-attention weights and optimizer state reset (trunk + heads preserved)")

    def diagnose_attention(self, num_samples: int = 256) -> str:
        """Analyze object self-attention patterns to determine if they're meaningful."""
        if not self.online_net.use_attn:
            return "Attention is disabled in this model."
        if not hasattr(self.online_net, 'object_attn'):
            return "No object self-attention found."
        if len(self.memory) < num_samples:
            return f"Need {num_samples} samples in buffer, have {len(self.memory)}."

        batch = self.memory.sample(num_samples, beta=0.4)
        if batch is None:
            return "Could not sample from buffer."

        states = torch.from_numpy(batch[0]).float().to(self.device)
        self.online_net.eval()
        with torch.no_grad():
            obj_tokens, obj_mask = self.online_net._build_object_tokens(states)
            _, attn_w = self.online_net.object_attn(
                obj_tokens, obj_mask, return_weights=True
            )
        self.online_net.train()

        import numpy as np
        eps = 1e-8

        # attn_w: (B, H, T, T) self-attention over T=144 object tokens
        B, H, T, _ = attn_w.shape
        aw = attn_w.cpu().numpy()
        em = obj_mask.cpu().numpy()  # (B, T) bool — True = empty

        # Category layout for labelling
        cat_defs = getattr(RL_CONFIG, 'entity_categories', [])
        cat_ranges = []
        offset = 0
        for name, slots in cat_defs:
            cat_ranges.append((name, offset, offset + slots))
            offset += slots

        lines = []
        lines.append("\n" + "=" * 70)
        lines.append("  OBJECT SELF-ATTENTION DIAGNOSTICS".center(70))
        lines.append("=" * 70)
        lines.append(f"  Shape: {B} samples x {H} heads x {T} tokens (self-attn)")

        # Per-category occupancy
        lines.append(f"\n  Per-category slot occupancy:")
        for name, lo, hi in cat_ranges:
            occ = 1.0 - em[:, lo:hi].mean()
            bar = "#" * int(occ * 20) + "." * (20 - int(occ * 20))
            lines.append(f"    {name:12s} [{lo:3d}..{hi:3d}): {occ:.1%}  {bar}")

        # Overall active tokens
        avg_active = (~em).sum(axis=1).mean()
        lines.append(f"    Avg active tokens: {avg_active:.1f} / {T}")

        # 1. Entropy per head
        max_entropy = np.log(T)
        entropy = -(aw * np.log(aw + eps)).sum(axis=-1)  # (B, H, T)
        mean_entropy_per_head = entropy.mean(axis=(0, 2))  # (H,)
        overall_entropy = entropy.mean()
        ratio = overall_entropy / max_entropy

        lines.append(f"\n  Entropy per head (uniform = {max_entropy:.3f}):")
        for h in range(H):
            e = mean_entropy_per_head[h]
            pct = e / max_entropy * 100
            bar = "#" * int(pct / 5) + "." * (20 - int(pct / 5))
            lines.append(f"    Head {h}: {e:.3f} ({pct:.0f}% uniform)  {bar}")
        lines.append(f"    Overall: {overall_entropy:.3f} ({ratio*100:.0f}% uniform)")

        if ratio > 0.95:
            lines.append("    -> Near-uniform: not yet selective")
        elif ratio > 0.80:
            lines.append("    -> Mildly selective: some structure emerging")
        elif ratio > 0.60:
            lines.append("    -> Moderately selective: meaningful patterns forming")
        else:
            lines.append("    -> Highly selective: strong learned patterns")

        # 2. Cross-category attention: do entities attend to other types?
        lines.append(f"\n  Cross-category attention (avg weight given to other categories):")
        for name_q, lo_q, hi_q in cat_ranges:
            same_attn = aw[:, :, lo_q:hi_q, lo_q:hi_q].mean()
            total_attn = aw[:, :, lo_q:hi_q, :].mean()
            cross_attn = total_attn - same_attn if total_attn > 0 else 0
            lines.append(f"    {name_q:12s}: same={same_attn:.4f}  cross={cross_attn:.4f}")

        # 3. Empty-slot masking
        if em.any():
            # Average attention weight received by empty vs active tokens
            recv_attn = aw.mean(axis=(1, 2))  # (B, T)
            empty_recv = recv_attn[em].mean() if em.any() else 0
            active_recv = recv_attn[~em].mean() if (~em).any() else 0
            lines.append(f"\n  Empty-slot masking:")
            lines.append(f"    Avg attention to active tokens: {active_recv:.4f}")
            lines.append(f"    Avg attention to empty tokens:  {empty_recv:.4f}")
            if empty_recv < 0.01:
                lines.append("    -> Empty slots effectively masked")
            elif empty_recv < active_recv * 0.1:
                lines.append("    -> Minimal attention leakage")
            else:
                lines.append("    -> Significant attention to empty slots")

        # 4. Head specialization
        head_avg = aw.mean(axis=(0, 2))  # (H, T)
        head_kls = []
        for i in range(H):
            for j in range(i + 1, H):
                p, q = head_avg[i] + eps, head_avg[j] + eps
                p, q = p / p.sum(), q / q.sum()
                kl = (p * np.log(p / q)).sum()
                head_kls.append(kl)
        avg_kl = np.mean(head_kls) if head_kls else 0
        lines.append(f"\n  Head specialization (avg KL between heads): {avg_kl:.4f}")
        if avg_kl > 0.1:
            lines.append("    -> Heads are specialized")
        elif avg_kl > 0.01:
            lines.append("    -> Mild specialization")
        else:
            lines.append("    -> Heads are redundant")

        lines.append("\n" + "=" * 70)
        return "\n".join(lines)

    def get_q_value_range(self):
        if len(self.memory) < 32:
            return float("nan"), float("nan")
        batch = self.memory.sample(32, beta=0.4)
        if batch is None:
            return float("nan"), float("nan")
        states = batch[0]
        st = torch.from_numpy(states).float().to(self.device)
        self.online_net.eval()
        with torch.no_grad():
            q = self.online_net.q_values(st)
            mn, mx = q.min().item(), q.max().item()
        self.online_net.train()
        return mn, mx

    def stop(self):
        self.running = False
        try:
            self._train_queue.put(None, block=False)
        except queue.Full:
            pass
        self._train_thread.join(timeout=3.0)

# Legacy alias
DiscreteDQNAgent = RainbowAgent

def setup_environment():
    os.makedirs(MODEL_DIR, exist_ok=True)
