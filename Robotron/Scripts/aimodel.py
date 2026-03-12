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


def compute_scheduled_lr(step: int, cfg=RL_CONFIG) -> float:
    """Compute the optimizer LR for a given training step."""
    step_i = max(0, int(step))
    lr_max = max(0.0, float(getattr(cfg, "lr", 0.0)))
    lr_min = max(0.0, float(getattr(cfg, "lr_min", 0.0)))
    if lr_min > lr_max:
        lr_min = lr_max

    warmup_steps = max(0, int(getattr(cfg, "lr_warmup_steps", 0)))
    if warmup_steps > 0 and step_i < warmup_steps:
        frac = float(step_i + 1) / float(warmup_steps)
        return lr_min + (lr_max - lr_min) * frac

    decay_horizon = max(1, int(getattr(cfg, "lr_cosine_period", 1)))
    after_warmup = max(0, step_i - warmup_steps)
    if bool(getattr(cfg, "lr_use_restarts", False)):
        t = after_warmup % decay_horizon
    else:
        t = min(after_warmup, decay_horizon)
    cosine = 0.5 * (1.0 + math.cos(math.pi * t / decay_horizon))
    return lr_min + (lr_max - lr_min) * cosine


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
    preview_width: int = 0
    preview_height: int = 0
    preview_format: int = 0
    preview_pixels: Optional[bytes] = None
    preview_encoded_format: int = 0
    preview_encoded_bytes: int = 0
    preview_raw_bytes: int = 0

def parse_frame_data(data: bytes, parse_preview: bool = True) -> Optional[FrameData]:
    try:
        fmt = ">HddBIBBIBB"
        hdr = struct.calcsize(fmt)
        if not data or len(data) < hdr:
            return None
        vals = struct.unpack(fmt, data[:hdr])
        (n, subj, obj, done, score, player_alive, save, replay_level, num_lasers, wave_number) = vals
        base_len = hdr + (int(n) * 4)
        if len(data) < base_len:
            return None
        state = np.frombuffer(data[hdr:base_len], dtype=">f4", count=n).astype(np.float32)
        if state.shape[0] != int(n):
            return None

        preview_width = 0
        preview_height = 0
        preview_format = 0
        preview_pixels = None
        preview_encoded_format = 0
        preview_encoded_bytes = 0
        preview_raw_bytes = 0

        if len(data) > base_len:
            if len(data) < (base_len + 4):
                return None
            preview_len = struct.unpack(">I", data[base_len:base_len + 4])[0]
            tail_start = base_len + 4
            tail_end = tail_start + int(preview_len)
            if tail_end != len(data):
                return None
            if (not parse_preview) and preview_len > 0:
                # Fast path for non-preview clients: validate framing only.
                preview_len = 0
            elif preview_len > 0 and preview_len < 5:
                return None
            if preview_len >= 5:
                preview_width, preview_height, preview_format = struct.unpack(">HHB", data[tail_start:tail_start + 5])
                pixels = data[tail_start + 5:tail_end]
                if preview_width <= 0 or preview_height <= 0 or len(pixels) <= 0:
                    return None
                expected_px_bytes = int(preview_width) * int(preview_height) * 2
                pf = int(preview_format)
                preview_encoded_format = pf
                preview_encoded_bytes = int(len(pixels))
                preview_raw_bytes = int(expected_px_bytes)
                if pf == 1:
                    if len(pixels) != expected_px_bytes:
                        return None
                    preview_pixels = bytes(pixels)
                elif pf == 2:
                    # LZSS stream: flag byte + 8 tokens (literal or 2-byte match)
                    # Match token: [len_minus_3:4 | dist_hi:4], [dist_lo:8].
                    out = bytearray(expected_px_bytes)
                    oi = 0
                    si = 0
                    plen = len(pixels)
                    ok = True
                    while oi < expected_px_bytes and si < plen:
                        flags = pixels[si]
                        si += 1
                        for bit in range(8):
                            if oi >= expected_px_bytes:
                                break
                            if (flags >> bit) & 1:
                                if (si + 1) >= plen:
                                    ok = False
                                    break
                                b1 = pixels[si]
                                b2 = pixels[si + 1]
                                si += 2
                                mlen = ((b1 >> 4) & 0x0F) + 3
                                dist = ((b1 & 0x0F) << 8) | b2
                                if dist <= 0 or dist > oi:
                                    ok = False
                                    break
                                src_idx = oi - dist
                                for _ in range(mlen):
                                    if oi >= expected_px_bytes:
                                        break
                                    out[oi] = out[src_idx]
                                    oi += 1
                                    src_idx += 1
                            else:
                                if si >= plen:
                                    ok = False
                                    break
                                out[oi] = pixels[si]
                                oi += 1
                                si += 1
                        if not ok:
                            break
                    if (not ok) or (oi != expected_px_bytes):
                        return None
                    preview_pixels = bytes(out)
                    preview_format = 1
                elif pf == 3:
                    # Simple word-RLE on RGB565BE:
                    #  ctrl (1B): high bit=run/literal, low 7 bits = count-1 (1..128 words)
                    #  run:   [ctrl][word_hi][word_lo]
                    #  lit:   [ctrl][count*2 bytes literal words]
                    out = bytearray(expected_px_bytes)
                    oi = 0
                    si = 0
                    plen = len(pixels)
                    ok = True
                    while si < plen and oi < expected_px_bytes:
                        ctrl = pixels[si]
                        si += 1
                        words = (ctrl & 0x7F) + 1
                        if (ctrl & 0x80) != 0:
                            if (si + 1) >= plen:
                                ok = False
                                break
                            b0 = pixels[si]
                            b1 = pixels[si + 1]
                            si += 2
                            need = words * 2
                            if (oi + need) > expected_px_bytes:
                                ok = False
                                break
                            for _ in range(words):
                                out[oi] = b0
                                out[oi + 1] = b1
                                oi += 2
                        else:
                            need = words * 2
                            if (si + need) > plen or (oi + need) > expected_px_bytes:
                                ok = False
                                break
                            out[oi:oi + need] = pixels[si:si + need]
                            oi += need
                            si += need
                    if (not ok) or (oi != expected_px_bytes) or (si != plen):
                        return None
                    preview_pixels = bytes(out)
                    preview_format = 1
                else:
                    return None

        return FrameData(
            state=state, subjreward=float(subj), objreward=float(obj),
            done=bool(done), save_signal=bool(save),
            player_alive=bool(player_alive),
            level_number=int(wave_number),
            game_score=int(score),
            next_replay_level=int(replay_level),
            num_lasers=int(num_lasers),
            preview_width=int(preview_width),
            preview_height=int(preview_height),
            preview_format=int(preview_format),
            preview_pixels=preview_pixels,
            preview_encoded_format=int(preview_encoded_format),
            preview_encoded_bytes=int(preview_encoded_bytes),
            preview_raw_bytes=int(preview_raw_bytes),
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
_WORLD_UNITS_PER_PIXEL = 256.0  # Robotron positions are stored as 8.8 fixed-point.

# "Safe" distance: 1/8 screen height = 26.25 px = 6720 x16-units.
# Enemies beyond this are not an immediate threat.
_SAFE_DIST = 6720.0 / _POS_MAX_DIAG  # ~0.105
# Alignment safe distance: ~12 px = 3072 x16-units.
# When nearest enemy is beyond this, expert aligns on one axis before engaging.
_ALIGN_SAFE_DIST = 3072.0 / _POS_MAX_DIAG  # ~0.048
_PROJECTILE_DANGER_DIST = 6144.0 / _POS_MAX_DIAG  # ~24 px
_ALIGN_HALF_WINDOW_PX = 8.0
_ALIGN_HALF_WINDOW_WORLD = _ALIGN_HALF_WINDOW_PX * _WORLD_UNITS_PER_PIXEL
_ALIGN_ROBOT_CATEGORIES = {"grunt", "brain", "tank", "spawner", "enforcer"}
_ALIGNED_FIRE_CATEGORIES = _ALIGN_ROBOT_CATEGORIES | {"projectile"}
_PLAYER_BOX_W_PX = 4.0
_PLAYER_BOX_H_PX = 12.0
_PLAYER_BOX_W_WORLD = _PLAYER_BOX_W_PX * _WORLD_UNITS_PER_PIXEL
_PLAYER_BOX_H_WORLD = _PLAYER_BOX_H_PX * _WORLD_UNITS_PER_PIXEL
_AVOIDANCE_BASE_PADDING_PX = 1.0
_AVOIDANCE_PADDING_PX_BY_CATEGORY = {
    "grunt": 0.5,
    "hulk": 1.0,
    "brain": 1.0,
    "tank": 1.0,
    "spawner": 1.0,
    "enforcer": 1.0,
    "projectile": 0.5,
    "electrode": 0.5,
}
_MOVE_SAFETY_LOOKAHEAD_PX = 10.0
_MOVE_SAFETY_PATH_RADIUS_PX = 2.0
_MOVE_SAFETY_LOOKAHEAD_WORLD = _MOVE_SAFETY_LOOKAHEAD_PX * _WORLD_UNITS_PER_PIXEL
_MOVE_SAFETY_PATH_RADIUS_WORLD = _MOVE_SAFETY_PATH_RADIUS_PX * _WORLD_UNITS_PER_PIXEL
_HIGH_WAVE_THRESHOLD = 6
_BRAIN_GUARD_WAVE = 4
_LOCAL_PRESSURE_DIST = 0.13
_LOCAL_PRESSURE_HEAVY = 1.55
_CENTER_PULL_WALL_X = 0.16
_CENTER_PULL_WALL_Y = 0.12
_PRIORITY_TARGET_BONUS = {
    "grunt": 0.20,
    "hulk": -0.20,
    "brain": 1.25,
    "tank": 0.55,
    "spawner": 0.65,
    "enforcer": 0.80,
    "projectile": 1.40,
    "electrode": -0.65,
}

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


def _axis_align_toward_enemy(ex_world: float, ey_world: float) -> int:
    """Move toward enemy on the smaller-offset axis (x or y)."""
    # If we are already aligned on one axis, move along the other axis
    # instead of introducing unnecessary lateral drift.
    if abs(ex_world) <= _ALIGN_HALF_WINDOW_WORLD:
        return _closest_dir8(0.0, 1.0 if ey_world >= 0.0 else -1.0, default_dir=0)
    if abs(ey_world) <= _ALIGN_HALF_WINDOW_WORLD:
        return _closest_dir8(1.0 if ex_world >= 0.0 else -1.0, 0.0, default_dir=0)

    if abs(ex_world) <= abs(ey_world):
        vx = 1.0 if ex_world >= 0.0 else -1.0
        vy = 0.0
    else:
        vx = 0.0
        vy = 1.0 if ey_world >= 0.0 else -1.0
    return _closest_dir8(vx, vy, default_dir=0)


def _move_dir_vector(move_dir: int) -> tuple[float, float]:
    if 0 <= move_dir < len(_DIR8_VECTORS):
        return _DIR8_VECTORS[move_dir]
    return 0.0, 0.0


def _move_dir_endpoint_world(move_dir: int) -> tuple[float, float]:
    vx, vy = _move_dir_vector(move_dir)
    return vx * _MOVE_SAFETY_LOOKAHEAD_WORLD, vy * _MOVE_SAFETY_LOOKAHEAD_WORLD


def _point_to_segment_distance(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    abx = bx - ax
    aby = by - ay
    ab_len2 = (abx * abx) + (aby * aby)
    if ab_len2 <= 1e-10:
        return math.hypot(px - ax, py - ay)
    apx = px - ax
    apy = py - ay
    t = max(0.0, min(1.0, ((apx * abx) + (apy * aby)) / ab_len2))
    closest_x = ax + (abx * t)
    closest_y = ay + (aby * t)
    return math.hypot(px - closest_x, py - closest_y)


def _aabb_clearance_top_left(
    ax: float,
    ay: float,
    aw: float,
    ah: float,
    bx: float,
    by: float,
    bw: float,
    bh: float,
) -> float:
    """Signed clearance between two top-left anchored AABBs.

    Positive when separated, negative when overlapping.
    """
    dx1 = bx - (ax + aw)
    dx2 = ax - (bx + bw)
    dy1 = by - (ay + ah)
    dy2 = ay - (by + bh)
    sep_x = max(dx1, dx2)
    sep_y = max(dy1, dy2)
    if sep_x > 0.0 or sep_y > 0.0:
        return math.hypot(max(sep_x, 0.0), max(sep_y, 0.0))
    overlap_x = min((ax + aw) - bx, (bx + bw) - ax)
    overlap_y = min((ay + ah) - by, (by + bh) - ay)
    return -min(overlap_x, overlap_y)


def _player_box_center(ax: float = 0.0, ay: float = 0.0) -> tuple[float, float]:
    return ax + (0.5 * _PLAYER_BOX_W_WORLD), ay + (0.5 * _PLAYER_BOX_H_WORLD)


def _closest_point_on_aabb(px: float, py: float, bx: float, by: float, bw: float, bh: float) -> tuple[float, float]:
    return (
        min(max(px, bx), bx + bw),
        min(max(py, by), by + bh),
    )


def _hazard_repulsion_vector(
    bx: float,
    by: float,
    bw: float,
    bh: float,
    pad_world: float,
) -> tuple[float, float, float]:
    hazard_x = bx - pad_world
    hazard_y = by - pad_world
    hazard_w = bw + (2.0 * pad_world)
    hazard_h = bh + (2.0 * pad_world)
    clearance = _aabb_clearance_top_left(
        0.0,
        0.0,
        _PLAYER_BOX_W_WORLD,
        _PLAYER_BOX_H_WORLD,
        hazard_x,
        hazard_y,
        hazard_w,
        hazard_h,
    )
    player_cx, player_cy = _player_box_center()
    nearest_x, nearest_y = _closest_point_on_aabb(player_cx, player_cy, hazard_x, hazard_y, hazard_w, hazard_h)
    repulse_x = player_cx - nearest_x
    repulse_y = player_cy - nearest_y
    if abs(repulse_x) <= 1e-6 and abs(repulse_y) <= 1e-6:
        hazard_cx = hazard_x + (0.5 * hazard_w)
        hazard_cy = hazard_y + (0.5 * hazard_h)
        repulse_x = player_cx - hazard_cx
        repulse_y = player_cy - hazard_cy
    return repulse_x, repulse_y, clearance


_CATEGORY_NAMES = tuple(name for name, _ in getattr(RL_CONFIG, "entity_categories", ()))


def _state_layout() -> tuple[int, int, int, int, int]:
    cfg = RL_CONFIG
    g = int(getattr(cfg, "global_feature_count", 98))
    gw = int(getattr(cfg, "grid_width", 12))
    gh = int(getattr(cfg, "grid_height", 12))
    gc = int(getattr(cfg, "grid_channels", 8))
    tc = int(getattr(cfg, "object_token_count", 64))
    tf = int(getattr(cfg, "object_token_features", 15))
    return g, gw * gh * gc, tc, tf, gc


def _split_latest_sections(state: np.ndarray):
    s = _latest_frame_state(state)
    g, grid_n, tc, tf, gc = _state_layout()
    need = g + grid_n + (tc * tf)
    if s.size < need:
        return None, None, None
    globals_ = s[:g]
    grid = s[g:g + grid_n].reshape(gc, int(getattr(RL_CONFIG, "grid_height", 12)), int(getattr(RL_CONFIG, "grid_width", 12)))
    tokens = s[g + grid_n:g + grid_n + (tc * tf)].reshape(tc, tf)
    return globals_, grid, tokens


def _active_tokens_from_state(state: np.ndarray) -> np.ndarray:
    _, _, tokens = _split_latest_sections(state)
    if tokens is None or tokens.size == 0:
        return np.zeros((0, int(getattr(RL_CONFIG, "object_token_features", 15))), dtype=np.float32)
    active = tokens[tokens[:, 0] > 0.5]
    if active.size == 0:
        return np.zeros((0, tokens.shape[1]), dtype=np.float32)
    return active


def _token_category_name(tok: np.ndarray) -> str:
    if tok.shape[0] < 12 or not _CATEGORY_NAMES:
        return ""
    idx = int(round(float(tok[11]) * max(1, len(_CATEGORY_NAMES) - 1)))
    idx = max(0, min(len(_CATEGORY_NAMES) - 1, idx))
    return _CATEGORY_NAMES[idx]


def _wave_number_from_state(state: np.ndarray) -> int:
    globals_, _, _ = _split_latest_sections(state)
    if globals_ is None or globals_.shape[0] < 5:
        return 1
    try:
        wave_norm = float(globals_[4])
    except Exception:
        wave_norm = 0.0
    if not np.isfinite(wave_norm):
        wave_norm = 0.0
    return max(1, min(40, int(round(max(0.0, min(1.0, wave_norm)) * 40.0))))


def _player_position_from_state(state: np.ndarray) -> tuple[float, float]:
    globals_, _, _ = _split_latest_sections(state)
    if globals_ is None or globals_.shape[0] < 7:
        return 0.5, 0.5
    try:
        px = float(globals_[5])
        py = float(globals_[6])
    except Exception:
        return 0.5, 0.5
    if not np.isfinite(px):
        px = 0.5
    if not np.isfinite(py):
        py = 0.5
    return max(0.0, min(1.0, px)), max(0.0, min(1.0, py))


def _center_vector_from_state(state: np.ndarray) -> tuple[float, float]:
    px, py = _player_position_from_state(state)
    return (0.5 - px) * _REL_POS_X_RANGE, (0.5 - py) * _REL_POS_Y_RANGE


def _category_count_from_state(state: np.ndarray, category: str) -> int:
    toks = _active_tokens_from_state(state)
    if toks.shape[0] == 0:
        return 0
    total = 0
    for tok in toks:
        if _token_category_name(tok) == category:
            total += 1
    return total


def _nearest_category_vector_from_state(
    state: np.ndarray,
    categories: set[str] | frozenset[str],
) -> Optional[Tuple[float, float, float]]:
    toks = _active_tokens_from_state(state)
    if toks.shape[0] == 0:
        return None
    best = None
    for tok in toks:
        if _token_category_name(tok) not in categories:
            continue
        dist = float(tok[5])
        if not np.isfinite(dist):
            continue
        if best is None or dist < best[2]:
            best = (float(tok[1]), float(tok[2]), dist)
    return best


def _local_pressure_from_state(state: np.ndarray) -> tuple[float, int]:
    toks = _active_tokens_from_state(state)
    if toks.shape[0] == 0:
        return 0.0, 0
    pressure = 0.0
    danger_count = 0
    for tok in toks:
        if tok[12] > 0.5 or tok[13] < 0.5:
            continue
        dist = float(tok[5])
        if not np.isfinite(dist) or dist > _LOCAL_PRESSURE_DIST:
            continue
        name = _token_category_name(tok)
        threat = float(tok[8]) if np.isfinite(float(tok[8])) else 0.0
        weight = threat
        if name == "projectile":
            weight += 0.55
        elif name == "brain":
            weight += 0.65
        elif name == "enforcer":
            weight += 0.35
        elif name == "tank":
            weight += 0.20
        pressure += max(0.0, weight) * (1.0 - (dist / _LOCAL_PRESSURE_DIST))
        danger_count += 1
    return pressure, danger_count


def _priority_target_vector_from_state(
    state: np.ndarray,
    wave_number: int,
    num_humans: int,
) -> Optional[Tuple[float, float, float, str]]:
    toks = _active_tokens_from_state(state)
    if toks.shape[0] == 0:
        return None
    best = None
    best_score = -1e30
    for tok in toks:
        if tok[12] > 0.5:
            continue
        name = _token_category_name(tok)
        dist = float(tok[5])
        if not np.isfinite(dist):
            continue
        dx = float(tok[1])
        dy = float(tok[2])
        threat = float(tok[8]) if np.isfinite(float(tok[8])) else 0.0
        approach = max(0.0, min(1.0, float(tok[14]) if np.isfinite(float(tok[14])) else 0.0))
        score = (1.15 - min(1.0, dist)) + threat + _PRIORITY_TARGET_BONUS.get(name, 0.0)
        score += 0.45 * approach
        if name == "brain" and num_humans > 0:
            score += 0.85
        if name == "projectile" and dist <= (_PROJECTILE_DANGER_DIST * 1.6):
            score += 0.90
        if name == "enforcer" and wave_number >= _HIGH_WAVE_THRESHOLD:
            score += 0.35
        if name == "spawner" and wave_number >= (_HIGH_WAVE_THRESHOLD + 2):
            score += 0.30
        if best is None or score > best_score:
            best = (dx, dy, dist, name)
            best_score = score
    return best


def _priority_aligned_fire_direction_from_state(
    state: np.ndarray,
    wave_number: int,
    num_humans: int,
) -> Optional[int]:
    toks = _active_tokens_from_state(state)
    if toks.shape[0] == 0:
        return None

    per_direction: list[Optional[tuple[float, int]]] = [None] * len(_DIR8_VECTORS)
    for tok in toks:
        name = _token_category_name(tok)
        if name == "human":
            continue
        dx = float(tok[1])
        dy = float(tok[2])
        dist = float(tok[5])
        if not (np.isfinite(dx) and np.isfinite(dy) and np.isfinite(dist)):
            continue

        world_dx = dx * _REL_POS_X_RANGE
        world_dy = dy * _REL_POS_Y_RANGE
        world_dist = math.hypot(world_dx, world_dy)
        threat = float(tok[8]) if np.isfinite(float(tok[8])) else 0.0
        base_score = (1.10 - min(1.0, dist)) + threat + _PRIORITY_TARGET_BONUS.get(name, 0.0)
        if name == "brain" and num_humans > 0:
            base_score += 0.85
        if name == "projectile" and dist <= (_PROJECTILE_DANGER_DIST * 1.6):
            base_score += 0.90
        if name == "enforcer" and wave_number >= _HIGH_WAVE_THRESHOLD:
            base_score += 0.35

        for fire_dir, (dir_x, dir_y) in enumerate(_DIR8_VECTORS):
            forward = (world_dx * dir_x) + (world_dy * dir_y)
            if forward <= 0.0:
                continue
            perp = abs((world_dx * dir_y) - (world_dy * dir_x))
            if perp > _ALIGN_HALF_WINDOW_WORLD:
                continue
            score = base_score + (0.35 * (1.0 - min(1.0, world_dist / 32768.0)))
            best_dir = per_direction[fire_dir]
            if best_dir is None or score > best_dir[0]:
                per_direction[fire_dir] = (score, fire_dir)

    best_aligned: Optional[tuple[float, int]] = None
    for candidate in per_direction:
        if candidate is None:
            continue
        if best_aligned is None or candidate[0] > best_aligned[0]:
            best_aligned = candidate
    if best_aligned is None:
        return None
    return int(best_aligned[1])


def _nearest_enemy_vector_from_state(state: np.ndarray) -> Optional[Tuple[float, float, float]]:
    toks = _active_tokens_from_state(state)
    if toks.shape[0] == 0:
        return None
    best = None
    best_cat = None
    for tok in toks:
        if tok[12] > 0.5 or tok[13] < 0.5:
            continue
        dist = float(tok[5])
        if not np.isfinite(dist):
            continue
        if best is None or dist < best[2]:
            best = (float(tok[1]), float(tok[2]), dist)
            best_cat = _token_category_name(tok)
    if best_cat == "electrode" and best is not None:
        import os
        if os.getenv("ROBOTRON_LOG_OBSTACLES", "").strip().lower() not in {"", "0", "false", "off", "no"}:
            print(f"[OBSTACLE] Nearest enemy is electrode at dist={best[2]:.4f} ({best[2]*64.0:.1f}px), dx={best[0]:.3f}, dy={best[1]:.3f}")
    return best


def _nearest_align_robot_vector_from_state(state: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """Return nearest robot candidate for axis alignment (non-grunt, non-obstacle)."""
    toks = _active_tokens_from_state(state)
    if toks.shape[0] == 0:
        return None
    best: Optional[Tuple[float, float, float]] = None
    for tok in toks:
        name = _token_category_name(tok)
        if name not in _ALIGN_ROBOT_CATEGORIES:
            continue
        dist = float(tok[5])
        if not np.isfinite(dist):
            continue
        if best is None or dist < best[2]:
            best = (float(tok[1]), float(tok[2]), dist)
    return best


def _nearest_human_vector_from_state(state: np.ndarray) -> Optional[Tuple[float, float, float]]:
    toks = _active_tokens_from_state(state)
    if toks.shape[0] == 0:
        return None
    best = None
    for tok in toks:
        if tok[12] < 0.5:
            continue
        dist = float(tok[5])
        if not np.isfinite(dist):
            continue
        if best is None or dist < best[2]:
            best = (float(tok[1]), float(tok[2]), dist)
    return best


def _nearest_projectile_vector_from_state(state: np.ndarray) -> Optional[Tuple[float, float, float]]:
    toks = _active_tokens_from_state(state)
    if toks.shape[0] == 0:
        return None
    best = None
    for tok in toks:
        if _token_category_name(tok) != "projectile":
            continue
        dist = float(tok[5])
        if not np.isfinite(dist):
            continue
        if best is None or dist < best[2]:
            best = (float(tok[1]), float(tok[2]), dist)
    return best


def _nearby_avoidance_vectors_from_state(
    state: np.ndarray,
) -> list[tuple[float, float, float, float, float, float, float]]:
    """Return hazards close enough to intersect the player's short swept path."""
    toks = _active_tokens_from_state(state)
    if toks.shape[0] == 0:
        return []
    nearby: list[tuple[float, float, float, float, float, float, float]] = []
    for tok in toks:
        name = _token_category_name(tok)
        if tok[12] > 0.5:
            continue
        dx = float(tok[1])
        dy = float(tok[2])
        dist = float(tok[5])
        if not (np.isfinite(dx) and np.isfinite(dy) and np.isfinite(dist)):
            continue
        box_x = dx * _REL_POS_X_RANGE
        box_y = dy * _REL_POS_Y_RANGE
        width_px = max(1.0, float(tok[9]) * 16.0)
        height_px = max(1.0, float(tok[10]) * 16.0)
        box_w = width_px * _WORLD_UNITS_PER_PIXEL
        box_h = height_px * _WORLD_UNITS_PER_PIXEL
        pad_px = _AVOIDANCE_BASE_PADDING_PX + _AVOIDANCE_PADDING_PX_BY_CATEGORY.get(name, 0.5)
        pad_world = pad_px * _WORLD_UNITS_PER_PIXEL
        center_x = box_x + (0.5 * box_w)
        center_y = box_y + (0.5 * box_h)
        clearance_now = _aabb_clearance_top_left(
            0.0,
            0.0,
            _PLAYER_BOX_W_WORLD,
            _PLAYER_BOX_H_WORLD,
            box_x - pad_world,
            box_y - pad_world,
            box_w + (2.0 * pad_world),
            box_h + (2.0 * pad_world),
        )
        if clearance_now <= (_MOVE_SAFETY_LOOKAHEAD_WORLD + _MOVE_SAFETY_PATH_RADIUS_WORLD):
            nearby.append((box_x, box_y, box_w, box_h, pad_world, center_x, center_y))
    return nearby


def _move_candidate_hazard_score(
    move_dir: int,
    hazards: list[tuple[float, float, float, float, float, float, float]],
) -> tuple[float, float]:
    end_x, end_y = _move_dir_endpoint_world(move_dir)
    is_idle = move_dir == _move_idle_action_index()

    total_penalty = 0.0
    min_clearance = float("inf")
    for bx, by, bw, bh, pad_world, _cx, _cy in hazards:
        hazard_x = bx - pad_world
        hazard_y = by - pad_world
        hazard_w = bw + (2.0 * pad_world)
        hazard_h = bh + (2.0 * pad_world)
        if is_idle:
            samples = ((0.0, 0.0),)
        else:
            samples = ((0.0, 0.0), (end_x * 0.5, end_y * 0.5), (end_x, end_y))
        clearances = [
            _aabb_clearance_top_left(
                px,
                py,
                _PLAYER_BOX_W_WORLD,
                _PLAYER_BOX_H_WORLD,
                hazard_x,
                hazard_y,
                hazard_w,
                hazard_h,
            )
            for px, py in samples
        ]
        clearance = min(clearances)
        if clearance < min_clearance:
            min_clearance = clearance

        if clearance < 0.0:
            total_penalty += 1.0 + ((-clearance) / max(1.0, min(hazard_w, hazard_h)))
            end_clearance = clearances[-1]
            if end_clearance < clearances[0]:
                total_penalty += (clearances[0] - end_clearance) / max(1.0, min(hazard_w, hazard_h))

    return total_penalty, min_clearance


def _hazard_escape_vector(
    hazards: list[tuple[float, float, float, float, float, float, float]],
) -> tuple[float, float]:
    escape_x = 0.0
    escape_y = 0.0
    nearest: Optional[tuple[float, float, float]] = None

    for bx, by, bw, bh, pad_world, cx, cy in hazards:
        repulse_x, repulse_y, clearance = _hazard_repulsion_vector(bx, by, bw, bh, pad_world)
        repulse_len = math.hypot(repulse_x, repulse_y)
        if nearest is None or clearance < nearest[2]:
            nearest = (repulse_x, repulse_y, clearance)

        weight = max(0.0, (_MOVE_SAFETY_LOOKAHEAD_WORLD + _MOVE_SAFETY_PATH_RADIUS_WORLD) - clearance)
        if weight <= 0.0:
            continue
        scale = weight / max(1.0, repulse_len)
        escape_x += repulse_x * scale
        escape_y += repulse_y * scale

    if ((escape_x * escape_x) + (escape_y * escape_y)) > 1e-10:
        return escape_x, escape_y
    if nearest is not None:
        return nearest[0], nearest[1]
    return 0.0, 0.0


def _blocking_hazard_fire_direction(
    intended_move_dir: int,
    hazards: list[tuple[float, float, float, float, float, float, float]],
) -> Optional[int]:
    """Return fire dir toward the hazard most responsible for blocking intended movement."""
    if not hazards:
        return None
    end_x, end_y = _move_dir_endpoint_world(intended_move_dir)
    is_idle = intended_move_dir == _move_idle_action_index()
    best_key: Optional[tuple[float, float]] = None
    best_center: Optional[tuple[float, float]] = None
    for bx, by, bw, bh, pad_world, cx, cy in hazards:
        hazard_x = bx - pad_world
        hazard_y = by - pad_world
        hazard_w = bw + (2.0 * pad_world)
        hazard_h = bh + (2.0 * pad_world)
        samples = ((0.0, 0.0),) if is_idle else ((0.0, 0.0), (end_x * 0.5, end_y * 0.5), (end_x, end_y))
        clearance = min(
            _aabb_clearance_top_left(
                px, py,
                _PLAYER_BOX_W_WORLD, _PLAYER_BOX_H_WORLD,
                hazard_x, hazard_y, hazard_w, hazard_h,
            )
            for px, py in samples
        )
        if clearance >= 0.0:
            continue
        center_dist = math.hypot(cx, cy)
        key = (clearance, center_dist)
        if best_key is None or key < best_key:
            best_key = key
            best_center = (cx, cy)
    if best_center is None:
        return None
    return _closest_dir8(best_center[0], best_center[1], default_dir=0)


def _primary_blocking_hazard(
    intended_move_dir: int,
    hazards: list[tuple[float, float, float, float, float, float, float]],
) -> Optional[tuple[float, float, float, float, float, float, float]]:
    if not hazards:
        return None
    end_x, end_y = _move_dir_endpoint_world(intended_move_dir)
    is_idle = intended_move_dir == _move_idle_action_index()
    best_hazard: Optional[tuple[float, float, float, float, float, float, float]] = None
    best_key: Optional[tuple[float, float]] = None
    for hazard in hazards:
        bx, by, bw, bh, pad_world, cx, cy = hazard
        hazard_x = bx - pad_world
        hazard_y = by - pad_world
        hazard_w = bw + (2.0 * pad_world)
        hazard_h = bh + (2.0 * pad_world)
        samples = ((0.0, 0.0),) if is_idle else ((0.0, 0.0), (end_x * 0.5, end_y * 0.5), (end_x, end_y))
        clearance = min(
            _aabb_clearance_top_left(
                px,
                py,
                _PLAYER_BOX_W_WORLD,
                _PLAYER_BOX_H_WORLD,
                hazard_x,
                hazard_y,
                hazard_w,
                hazard_h,
            )
            for px, py in samples
        )
        if clearance >= 0.0:
            continue
        center_dist = math.hypot(cx, cy)
        key = (clearance, center_dist)
        if best_key is None or key < best_key:
            best_key = key
            best_hazard = hazard
    return best_hazard


def _slide_candidate_dirs(
    intended_move_dir: int,
    hazard: tuple[float, float, float, float, float, float, float],
) -> list[int]:
    bx, by, bw, bh, pad_world, _cx, _cy = hazard
    hazard_x = bx - pad_world
    hazard_y = by - pad_world
    hazard_w = bw + (2.0 * pad_world)
    hazard_h = bh + (2.0 * pad_world)
    desired_vx, desired_vy = _move_dir_vector(intended_move_dir)
    if abs(desired_vx) <= 1e-6 and abs(desired_vy) <= 1e-6:
        return []

    player_cx, player_cy = _player_box_center()
    hazard_cx = hazard_x + (0.5 * hazard_w)
    hazard_cy = hazard_y + (0.5 * hazard_h)
    rel_x = player_cx - hazard_cx
    rel_y = player_cy - hazard_cy

    candidates: list[int] = []
    if abs(desired_vx) >= abs(desired_vy):
        go_above = rel_y <= 0.0
        primary_vy = -1.0 if go_above else 1.0
        secondary_vy = -primary_vy
        if hazard_h <= hazard_w:
            candidates.append(_closest_dir8(desired_vx, primary_vy, default_dir=intended_move_dir))
            candidates.append(_closest_dir8(0.0, primary_vy))
            candidates.append(_closest_dir8(desired_vx, secondary_vy, default_dir=intended_move_dir))
            candidates.append(_closest_dir8(0.0, secondary_vy))
        else:
            candidates.append(_closest_dir8(0.0, primary_vy))
            candidates.append(_closest_dir8(desired_vx, primary_vy, default_dir=intended_move_dir))
            candidates.append(_closest_dir8(0.0, secondary_vy))
            candidates.append(_closest_dir8(desired_vx, secondary_vy, default_dir=intended_move_dir))
    else:
        go_left = rel_x <= 0.0
        primary_vx = -1.0 if go_left else 1.0
        secondary_vx = -primary_vx
        if hazard_w <= hazard_h:
            candidates.append(_closest_dir8(primary_vx, desired_vy, default_dir=intended_move_dir))
            candidates.append(_closest_dir8(primary_vx, 0.0))
            candidates.append(_closest_dir8(secondary_vx, desired_vy, default_dir=intended_move_dir))
            candidates.append(_closest_dir8(secondary_vx, 0.0))
        else:
            candidates.append(_closest_dir8(primary_vx, 0.0))
            candidates.append(_closest_dir8(primary_vx, desired_vy, default_dir=intended_move_dir))
            candidates.append(_closest_dir8(secondary_vx, 0.0))
            candidates.append(_closest_dir8(secondary_vx, desired_vy, default_dir=intended_move_dir))

    seen: set[int] = set()
    ordered: list[int] = []
    for cand in candidates:
        cand = int(max(0, min(NUM_MOVE - 1, cand)))
        if cand not in seen:
            seen.add(cand)
            ordered.append(cand)
    return ordered


def _apply_final_hazard_move_check(
    intended_move_dir: int,
    hazards: list[tuple[float, float, float, float, float, float, float]],
) -> int:
    if not hazards:
        return intended_move_dir

    intended_penalty, intended_clearance = _move_candidate_hazard_score(intended_move_dir, hazards)
    if intended_penalty <= 1e-6 and intended_clearance >= 0.0:
        return intended_move_dir

    desired_vx, desired_vy = _move_dir_vector(intended_move_dir)
    escape_x, escape_y = _hazard_escape_vector(hazards)
    primary_hazard = _primary_blocking_hazard(intended_move_dir, hazards)
    slide_dirs = _slide_candidate_dirs(intended_move_dir, primary_hazard) if primary_hazard is not None else []
    best_dir = intended_move_dir
    best_key: Optional[tuple[float, float, float, float]] = None

    for slide_rank, cand_dir in enumerate(slide_dirs):
        cand_penalty, cand_clearance = _move_candidate_hazard_score(cand_dir, hazards)
        if cand_penalty > 1e-6 or cand_clearance < 0.0:
            continue
        cand_vx, cand_vy = _move_dir_vector(cand_dir)
        slide_alignment = (desired_vx * cand_vx) + (desired_vy * cand_vy)
        return cand_dir if slide_alignment >= -0.25 else cand_dir

    for cand_dir in range(min(8, NUM_MOVE)):
        cand_penalty, cand_clearance = _move_candidate_hazard_score(cand_dir, hazards)
        cand_vx, cand_vy = _move_dir_vector(cand_dir)
        escape_alignment = (escape_x * cand_vx) + (escape_y * cand_vy)
        alignment = (desired_vx * cand_vx) + (desired_vy * cand_vy)
        key = (cand_penalty, -escape_alignment, -cand_clearance, -alignment)
        if best_key is None or key < best_key:
            best_key = key
            best_dir = cand_dir

    return best_dir


def _nearest_aligned_fire_direction_from_state(state: np.ndarray) -> Optional[int]:
    """Return fire direction for the closest non-human target aligned to any 8-way shot."""
    toks = _active_tokens_from_state(state)
    if toks.shape[0] == 0:
        return None

    per_direction: list[Optional[tuple[float, int]]] = [None] * len(_DIR8_VECTORS)
    for tok in toks:
        name = _token_category_name(tok)
        if name not in _ALIGNED_FIRE_CATEGORIES:
            continue
        dx = float(tok[1])
        dy = float(tok[2])
        dist = float(tok[5])
        if not (np.isfinite(dx) and np.isfinite(dy) and np.isfinite(dist)):
            continue

        world_dx = dx * _REL_POS_X_RANGE
        world_dy = dy * _REL_POS_Y_RANGE
        world_dist = math.hypot(world_dx, world_dy)
        for fire_dir, (dir_x, dir_y) in enumerate(_DIR8_VECTORS):
            forward = (world_dx * dir_x) + (world_dy * dir_y)
            if forward <= 0.0:
                continue
            perp = abs((world_dx * dir_y) - (world_dy * dir_x))
            if perp > _ALIGN_HALF_WINDOW_WORLD:
                continue
            best_dir = per_direction[fire_dir]
            if best_dir is None or world_dist < best_dir[0]:
                per_direction[fire_dir] = (world_dist, fire_dir)

    best_aligned: Optional[tuple[float, int]] = None
    for candidate in per_direction:
        if candidate is None:
            continue
        if best_aligned is None or candidate[0] < best_aligned[0]:
            best_aligned = candidate
    if best_aligned is None:
        return None
    return int(best_aligned[1])


def get_expert_action(state: np.ndarray, locked_fire: Optional[int] = None) -> Tuple[int, int]:
    """Heuristic Robotron expert.

    Fire:
        1) Prefer aligned high-priority threats (projectiles, brains near humans,
           enforcers/spawners on later waves).
        2) Otherwise, fallback to the highest-priority visible threat.
     Move priority:
        1) Flee close threats, mixing in a center pull on hard/high-pressure waves.
        2) Guard humans from brains before chasing rescues on later waves.
        3) When safe, rescue humans or align to a firing lane on priority targets.
        4) Bias back toward center when pressure rises near the walls.
    """
    wave_number = _wave_number_from_state(state)
    high_wave = wave_number >= _HIGH_WAVE_THRESHOLD
    num_humans = _category_count_from_state(state, "human")
    nearest_enemy = _nearest_enemy_vector_from_state(state)
    nearest_projectile = _nearest_projectile_vector_from_state(state)
    nearest_align_robot = _nearest_align_robot_vector_from_state(state)
    nearest_human = _nearest_human_vector_from_state(state)
    nearest_brain = _nearest_category_vector_from_state(state, {"brain"})
    nearby_avoidance = _nearby_avoidance_vectors_from_state(state)
    aligned_fire_dir = _priority_aligned_fire_direction_from_state(state, wave_number, num_humans)
    priority_target = _priority_target_vector_from_state(state, wave_number, num_humans)
    pressure, _danger_count = _local_pressure_from_state(state)
    center_x, center_y = _center_vector_from_state(state)
    px, py = _player_position_from_state(state)
    near_wall = (
        px <= _CENTER_PULL_WALL_X
        or px >= (1.0 - _CENTER_PULL_WALL_X)
        or py <= _CENTER_PULL_WALL_Y
        or py >= (1.0 - _CENTER_PULL_WALL_Y)
    )
    brain_guard_active = (
        nearest_brain is not None
        and num_humans > 0
        and wave_number >= _BRAIN_GUARD_WAVE
    )

    # ── Fire direction ───────────────────────────────────────────────
    if aligned_fire_dir is not None:
        fire_dir = int(aligned_fire_dir)
    elif priority_target is not None:
        fx, fy, _fd, _cat = priority_target
        fire_dir = _closest_dir8(fx * _REL_POS_X_RANGE, fy * _REL_POS_Y_RANGE, default_dir=0)
    elif nearest_projectile is not None and (
        nearest_enemy is None or nearest_projectile[2] <= max(nearest_enemy[2], _PROJECTILE_DANGER_DIST)
    ):
        fx, fy, _ = nearest_projectile
        fire_dir = _closest_dir8(fx * _REL_POS_X_RANGE, fy * _REL_POS_Y_RANGE, default_dir=0)
    elif nearest_enemy is not None:
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

        is_close_threat = enemy_dist < _ALIGN_SAFE_DIST
        if nearest_projectile is not None and nearest_projectile[2] < _PROJECTILE_DANGER_DIST:
            ex, ey, enemy_dist = nearest_projectile
            ex_world = ex * _REL_POS_X_RANGE
            ey_world = ey * _REL_POS_Y_RANGE
            is_close_threat = True

        # 1) Flee close threats first (including hulks).
        if is_close_threat:
            flee_x = -ex_world
            flee_y = -ey_world
            if high_wave or pressure >= _LOCAL_PRESSURE_HEAVY or near_wall:
                flee_x += center_x * 0.75
                flee_y += center_y * 0.75
            move_dir = _closest_dir8(flee_x, flee_y, default_dir=0)
        # 2) On harder waves, cut off brains before committing to rescues.
        elif brain_guard_active and nearest_brain is not None and pressure < (_LOCAL_PRESSURE_HEAVY * 1.15):
            bx, by, _ = nearest_brain
            brain_x = bx * _REL_POS_X_RANGE
            brain_y = by * _REL_POS_Y_RANGE
            if near_wall:
                brain_x += center_x * 0.35
                brain_y += center_y * 0.35
            move_dir = _axis_align_toward_enemy(brain_x, brain_y)
        # 3) Safe to rescue: prioritize humans when available.
        elif nearest_human is not None:
            hx, hy, _ = nearest_human
            rescue_x = hx * _REL_POS_X_RANGE
            rescue_y = hy * _REL_POS_Y_RANGE
            if high_wave and near_wall:
                rescue_x += center_x * 0.45
                rescue_y += center_y * 0.45
            move_dir = _closest_dir8(rescue_x, rescue_y)
        # 4) Safe standoff: align on one axis with nearest non-hulk robot.
        elif nearest_align_robot is not None:
            ax, ay, _ = nearest_align_robot
            align_x = ax * _REL_POS_X_RANGE
            align_y = ay * _REL_POS_Y_RANGE
            if high_wave and near_wall:
                align_x += center_x * 0.35
                align_y += center_y * 0.35
            move_dir = _axis_align_toward_enemy(align_x, align_y)
        else:
            # 5) Fallback: flee nearest enemy, but step back toward center when pressured.
            flee_x = -ex_world
            flee_y = -ey_world
            if high_wave or pressure >= _LOCAL_PRESSURE_HEAVY or near_wall:
                flee_x += center_x * 0.85
                flee_y += center_y * 0.85
            move_dir = _closest_dir8(flee_x, flee_y, default_dir=0)
    elif nearest_human is not None:
        hx, hy, _ = nearest_human
        if brain_guard_active and nearest_brain is not None:
            bx, by, _ = nearest_brain
            move_dir = _axis_align_toward_enemy(bx * _REL_POS_X_RANGE, by * _REL_POS_Y_RANGE)
        else:
            move_dir = _closest_dir8(hx * _REL_POS_X_RANGE, hy * _REL_POS_Y_RANGE)
    elif (high_wave or pressure > 0.0) and near_wall:
        move_dir = _closest_dir8(center_x, center_y, default_dir=_move_idle_action_index())
    else:
        move_dir = _move_idle_action_index()

    # ── Final nearby-object collision check ─────────────────────────
    intended_move_dir = move_dir
    move_dir = _apply_final_hazard_move_check(move_dir, nearby_avoidance)
    if (
        locked_fire is None or int(locked_fire) < 0
    ) and move_dir != intended_move_dir:
        blocked_fire_dir = _blocking_hazard_fire_direction(intended_move_dir, nearby_avoidance)
        if blocked_fire_dir is not None:
            fire_dir = int(blocked_fire_dir)

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

class GlobalEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpatialGridEncoder(nn.Module):
    def __init__(self, channels: int, hidden_channels: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 3, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        flat_dim = (hidden_channels * 3) * 2 * 2
        self.proj = nn.Sequential(
            nn.Linear(flat_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )
        self.out_dim = out_dim

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        h = self.net(grid)
        return self.proj(h.flatten(start_dim=1))


class EntitySetEncoder(nn.Module):
    def __init__(self, token_features: int, embed_dim: int, num_heads: int, num_layers: int):
        super().__init__()
        self.embed = nn.Linear(token_features, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(embed_dim)
        self.out_dim = embed_dim

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.norm(self.embed(tokens))
        bsz = x.shape[0]
        cls = self.cls_token.expand(bsz, -1, -1)
        x = torch.cat([cls, x], dim=1)
        key_padding_mask = None
        if mask is not None:
            cls_mask = torch.zeros((bsz, 1), dtype=torch.bool, device=mask.device)
            key_padding_mask = torch.cat([cls_mask, mask], dim=1)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return self.out_norm(x[:, 0, :])


# ── Distributional Dueling Network ─────────────────────────────────────────
class RainbowNet(nn.Module):
    """Hybrid Robotron network with global, spatial, and token encoders."""

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
        self.use_attn = bool(getattr(cfg, "use_enemy_attention", True))
        self.base_state_size = int(getattr(cfg, "base_state_size", SERVER_CONFIG.params_count))
        self.global_feature_count = int(getattr(cfg, "global_feature_count", 98))
        self.grid_width = int(getattr(cfg, "grid_width", 12))
        self.grid_height = int(getattr(cfg, "grid_height", 12))
        self.grid_channels = int(getattr(cfg, "grid_channels", 8))
        self.object_token_count = int(getattr(cfg, "object_token_count", 64))
        self.object_token_features = int(getattr(cfg, "object_token_features", 15))
        self.grid_feature_count = self.grid_width * self.grid_height * self.grid_channels
        self.token_feature_count = self.object_token_count * self.object_token_features

        self.global_encoder = GlobalEncoder(self.global_feature_count, int(getattr(cfg, "global_hidden", 192)))
        self.grid_encoder = SpatialGridEncoder(
            self.grid_channels,
            int(getattr(cfg, "grid_hidden_channels", 32)),
            int(getattr(cfg, "attn_dim", 192)),
        )
        self.entity_encoder = EntitySetEncoder(
            self.object_token_features,
            int(getattr(cfg, "attn_dim", 192)),
            int(getattr(cfg, "attn_heads", 8)),
            int(getattr(cfg, "attn_layers", 3)),
        )

        trunk_in = self.global_encoder.out_dim + self.grid_encoder.out_dim + self.entity_encoder.out_dim
        layers = []
        for i in range(int(cfg.trunk_layers)):
            out_dim = int(cfg.trunk_hidden)
            layers.append(nn.Linear(trunk_in if i == 0 else out_dim, out_dim))
            if cfg.use_layer_norm:
                layers.append(nn.LayerNorm(out_dim))
            layers.append(nn.ReLU())
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
        self.trunk = nn.Sequential(*layers)

        head_in = int(cfg.trunk_hidden)
        head_mid = max(64, head_in // 2)
        if self.use_dueling:
            self.val_fc = nn.Linear(head_in, head_mid)
            self.val_out = nn.Linear(head_mid, self.num_atoms)
            self.adv_fc = nn.Linear(head_in, head_mid)
            self.adv_out = nn.Linear(head_mid, self.num_actions * self.num_atoms)
        else:
            self.q_fc = nn.Linear(head_in, head_mid)
            self.q_out = nn.Linear(head_mid, self.num_actions * self.num_atoms)

        self._init_weights()
        if self.use_dist:
            support = torch.linspace(self.v_min, self.v_max, self.num_atoms)
            self.register_buffer("support", support)
            self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def _split_state_sections(self, state: torch.Tensor):
        B = state.shape[0]
        total_dim = int(state.shape[1])
        stack_depth = max(1, total_dim // max(1, self.base_state_size))
        latest_off = (stack_depth - 1) * self.base_state_size
        latest = state[:, latest_off: latest_off + self.base_state_size]
        globals_ = latest[:, :self.global_feature_count]
        grid_start = self.global_feature_count
        grid_end = grid_start + self.grid_feature_count
        grid = latest[:, grid_start:grid_end].reshape(B, self.grid_channels, self.grid_height, self.grid_width)
        tokens = latest[:, grid_end:grid_end + self.token_feature_count].reshape(
            B, self.object_token_count, self.object_token_features
        )
        return globals_, grid, tokens

    def _build_object_tokens(self, state: torch.Tensor):
        _, _, tokens = self._split_state_sections(state)
        mask = tokens[:, :, 0] < 0.5
        all_empty = mask.all(dim=1, keepdim=True)
        mask = mask & ~all_empty
        return tokens, mask

    def forward(self, state: torch.Tensor, log: bool = False):
        B = state.shape[0]
        globals_, grid, tokens = self._split_state_sections(state)
        token_mask = tokens[:, :, 0] < 0.5
        all_empty = token_mask.all(dim=1, keepdim=True)
        token_mask = token_mask & ~all_empty

        global_h = self.global_encoder(globals_)
        grid_h = self.grid_encoder(grid)
        entity_h = self.entity_encoder(tokens, token_mask) if self.use_attn else tokens.mean(dim=1)
        h = self.trunk(torch.cat([global_h, grid_h, entity_h], dim=1))

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
            q_atoms = q_atoms.float()
            if log:
                return F.log_softmax(q_atoms, dim=2)
            return F.softmax(q_atoms, dim=2)
        return q_atoms.squeeze(2)

    def q_values(self, state: torch.Tensor) -> torch.Tensor:
        if self.use_dist:
            probs = self.forward(state, log=False)
            return (probs * self.support.unsqueeze(0).unsqueeze(0)).sum(dim=2)
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
        return compute_scheduled_lr(self.training_steps, RL_CONFIG)

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
    def step(self, state, action, reward, next_state, done, actor="dqn", horizon=1,
             priority_reward=None, wave_number=1, start_wave=1):
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
        self.memory.add(
            state,
            action_idx,
            float(reward),
            next_state,
            bool(done),
            int(horizon),
            is_expert,
            priority_hint=pri,
            wave_number=int(wave_number),
            start_wave=int(start_wave),
        )
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
        """Reinitialize only the entity transformer, keeping other encoders and heads intact."""
        if not self.online_net.use_attn:
            print("No attention layer to reset.")
            return
        for net in (self.online_net, self.target_net):
            for m in net.entity_encoder.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=1.0)
                    nn.init.constant_(m.bias, 0.0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0.0)
        self._sync_inference(force=True)
        attn_param_ids = {id(p) for p in self.online_net.entity_encoder.parameters()}
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if id(p) in attn_param_ids and p in self.optimizer.state:
                    del self.optimizer.state[p]
        print("Entity encoder weights and optimizer state reset (other modules preserved)")

    def diagnose_attention(self, num_samples: int = 256) -> str:
        """Report token occupancy and salience stats for the hybrid observation."""
        if not self.online_net.use_attn:
            return "Attention is disabled in this model."
        if len(self.memory) < num_samples:
            return f"Need {num_samples} samples in buffer, have {len(self.memory)}."

        batch = self.memory.sample(num_samples, beta=0.4)
        if batch is None:
            return "Could not sample from buffer."

        states = torch.from_numpy(batch[0]).float().to(self.device)
        with torch.no_grad():
            obj_tokens, obj_mask = self.online_net._build_object_tokens(states)
        toks = obj_tokens.detach().cpu().numpy()
        mask = obj_mask.detach().cpu().numpy()
        active = ~mask
        B, T, Fdim = toks.shape
        active_counts = active.sum(axis=1)
        threat = toks[:, :, 8]
        humans = toks[:, :, 12]
        dangerous = toks[:, :, 13]

        lines = []
        lines.append("\n" + "=" * 70)
        lines.append("  HYBRID OBSERVATION DIAGNOSTICS".center(70))
        lines.append("=" * 70)
        lines.append(f"  Shape: {B} samples x {T} tokens x {Fdim} features")
        lines.append(f"  Avg active tokens: {active_counts.mean():.1f} / {T}")
        lines.append(f"  Avg dangerous tokens: {(dangerous * active).sum() / max(1, active.sum()):.3f}")
        lines.append(f"  Avg human tokens: {(humans * active).sum() / max(1, active.sum()):.3f}")
        lines.append(f"  Avg token threat: {(threat * active).sum() / max(1, active.sum()):.3f}")
        lines.append(f"  Max token threat: {(threat * active).max():.3f}")
        lines.append("\n  Attention weights are not surfaced from the transformer encoder.")
        lines.append("  Use this report to verify token occupancy and salience instead.")

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
