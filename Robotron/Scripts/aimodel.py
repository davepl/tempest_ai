#!/usr/bin/env python3
# ==================================================================================================================
# ||  ROBOTRON AI v2 • MODEL, AGENT, AND UTILITIES                                                              ||
# ||                                                                                                              ||
# ||  Rainbow-lite with:                                                                                          ||
# ||    • Distributional C51 value estimation                                                                     ||
# ||    • Factored action heads (move dir 8 + fire dir 8 = 64 total)                                             ||
# ||    • Unified 64-slot entity pool with learned type embeddings                                                ||
# ||    • Entity self-attention (2 layers) + 8-direction lane cross-attention                                     ||
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
                        LEGACY_CORE_FEATURES, LEGACY_ELIST_FEATURES, LEGACY_SLOT_STATE_FEATURES, \
                        UNIFIED_TYPE_NAMES, UNIFIED_NUM_TYPES, \
                        UNIFIED_HUMAN_TYPE_ID, UNIFIED_ELECTRODE_TYPE_ID, \
                        metrics as config_metrics, RESET_METRICS, IS_INTERACTIVE
    from training import train_step
    from replay_buffer import PrioritizedReplayBuffer
except ImportError:
    from Scripts.config import SERVER_CONFIG, RL_CONFIG, MODEL_DIR, LATEST_MODEL_PATH, \
                               LEGACY_CORE_FEATURES, LEGACY_ELIST_FEATURES, LEGACY_SLOT_STATE_FEATURES, \
                               UNIFIED_TYPE_NAMES, UNIFIED_NUM_TYPES, \
                               UNIFIED_HUMAN_TYPE_ID, UNIFIED_ELECTRODE_TYPE_ID, \
                               metrics as config_metrics, RESET_METRICS, IS_INTERACTIVE
    from Scripts.training import train_step
    from Scripts.replay_buffer import PrioritizedReplayBuffer

sys.modules.setdefault("aimodel", sys.modules[__name__])
warnings.filterwarnings("default")

metrics = config_metrics


def _export_metrics_snapshot() -> dict:
    snap = {}
    try:
        with metrics.lock:
            snap = {
                "episode_rewards": list(metrics.episode_rewards),
                "dqn_rewards": list(metrics.dqn_rewards),
                "expert_rewards": list(metrics.expert_rewards),
                "subj_rewards": list(metrics.subj_rewards),
                "obj_rewards": list(metrics.obj_rewards),
                "game_scores": list(metrics.game_scores),
                "avg_game_score": float(metrics.avg_game_score),
                "total_games_played": int(metrics.total_games_played),
                "peak_level": int(metrics.peak_level),
                "peak_level_verified": bool(getattr(metrics, "peak_level_verified", False)),
                "peak_episode_reward": float(metrics.peak_episode_reward),
                "peak_game_score": int(metrics.peak_game_score),
                "records_reset_seq": int(getattr(metrics, "records_reset_seq", 0)),
            }
    except Exception:
        snap = {}
    try:
        from metrics_display import export_window_state
    except ImportError:
        try:
            from Scripts.metrics_display import export_window_state
        except ImportError:
            export_window_state = None
    if export_window_state is not None:
        try:
            snap["display_windows"] = export_window_state()
        except Exception:
            pass
    return snap


def _restore_metrics_snapshot(snap: dict | None) -> None:
    if not isinstance(snap, dict):
        return
    try:
        with metrics.lock:
            if "episode_rewards" in snap:
                metrics.episode_rewards.clear()
                metrics.episode_rewards.extend(float(x) for x in snap.get("episode_rewards", []))
            if "dqn_rewards" in snap:
                metrics.dqn_rewards.clear()
                metrics.dqn_rewards.extend(float(x) for x in snap.get("dqn_rewards", []))
            if "expert_rewards" in snap:
                metrics.expert_rewards.clear()
                metrics.expert_rewards.extend(float(x) for x in snap.get("expert_rewards", []))
            if "subj_rewards" in snap:
                metrics.subj_rewards.clear()
                metrics.subj_rewards.extend(float(x) for x in snap.get("subj_rewards", []))
            if "obj_rewards" in snap:
                metrics.obj_rewards.clear()
                metrics.obj_rewards.extend(float(x) for x in snap.get("obj_rewards", []))
            if "game_scores" in snap:
                metrics.game_scores.clear()
                metrics.game_scores.extend(int(x) for x in snap.get("game_scores", []))
            metrics.avg_game_score = float(snap.get("avg_game_score", metrics.avg_game_score))
            metrics.total_games_played = int(snap.get("total_games_played", metrics.total_games_played))
            metrics.peak_level_verified = bool(snap.get("peak_level_verified", False))
            metrics.peak_level = int(snap.get("peak_level", 0)) if metrics.peak_level_verified else 0
            metrics.peak_episode_reward = float(snap.get("peak_episode_reward", metrics.peak_episode_reward))
            metrics.peak_game_score = int(snap.get("peak_game_score", metrics.peak_game_score))
            metrics.records_reset_seq = int(snap.get("records_reset_seq", getattr(metrics, "records_reset_seq", 0)))
    except Exception:
        pass
    try:
        from metrics_display import import_window_state
    except ImportError:
        try:
            from Scripts.metrics_display import import_window_state
        except ImportError:
            import_window_state = None
    if import_window_state is not None:
        try:
            import_window_state(snap.get("display_windows"))
        except Exception:
            pass

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
MODEL_ARCH_VERSION = 16


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
    start_pressed: bool = False
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
        fmt = ">HddBIBBBIBB"
        hdr = struct.calcsize(fmt)
        if not data or len(data) < hdr:
            return None
        vals = struct.unpack(fmt, data[:hdr])
        (n, subj, obj, done, score, player_alive, save, start_pressed, replay_level, num_lasers, wave_number) = vals
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
            start_pressed=bool(start_pressed),
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
    latest, _prev, _base = _latest_prev_frame_state(state)
    return latest


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
_ENDGAME_CLEANUP_CATEGORIES = _ALIGN_ROBOT_CATEGORIES | {"hulk"}
_ALIGNED_FIRE_CATEGORIES = _ENDGAME_CLEANUP_CATEGORIES | {"projectile"}
_PERIMETER_ORBIT_RING_MIN = 0.16
_PERIMETER_ORBIT_RING_TARGET = 0.30
_PERIMETER_ORBIT_PRESSURE_THRESHOLD = 6.0
_PERIMETER_ORBIT_PROJECTILE_BONUS = 2.5
_RESCUE_NEAR_DIST = 4608.0 / _POS_MAX_DIAG  # ~18 px
_RESCUE_ABORT_PROJECTILES = 2
_RESCUE_ABORT_PRESSURE = _PERIMETER_ORBIT_PRESSURE_THRESHOLD + 1.5
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
_LEGACY_CATEGORY_BOX_PX: dict[str, tuple[float, float]] = {
    "grunt": (5.0, 13.0),
    "hulk": (7.0, 16.0),
    "brain": (7.0, 16.0),
    "tank": (7.0, 16.0),
    "spawner": (8.0, 15.0),
    "enforcer": (8.0, 15.0),
    "projectile": (4.0, 7.0),
    "human": (5.0, 13.0),
    "electrode": (6.0, 6.0),
}
_LEGACY_THREAT_WEIGHT: dict[str, float] = {
    "grunt": 0.55,
    "hulk": 1.00,
    "brain": 0.95,
    "tank": 0.95,
    "spawner": 0.85,
    "enforcer": 0.85,
    "projectile": 0.90,
    "human": 0.35,
    "electrode": 0.75,
}
_LEGACY_DANGEROUS_CATEGORIES = {
    "grunt",
    "hulk",
    "brain",
    "tank",
    "spawner",
    "enforcer",
    "projectile",
    "electrode",
}


def _state_layout() -> tuple[int, int, int, int, int]:
    cfg = RL_CONFIG
    g = int(getattr(cfg, "global_feature_count", 98))
    gw = int(getattr(cfg, "grid_width", 12))
    gh = int(getattr(cfg, "grid_height", 12))
    gc = int(getattr(cfg, "grid_channels", 8))
    tc = int(getattr(cfg, "object_token_count", 64))
    tf = int(getattr(cfg, "object_token_features", 15))
    return g, gw * gh * gc, tc, tf, gc


def _hybrid_base_state_size() -> int:
    g, grid_n, tc, tf, _gc = _state_layout()
    return g + grid_n + (tc * tf)


def _legacy_slot_layout() -> tuple[list[tuple[str, int, int, int]], int]:
    slot_features = int(getattr(RL_CONFIG, "slot_state_features", LEGACY_SLOT_STATE_FEATURES))
    offset = int(LEGACY_CORE_FEATURES + LEGACY_ELIST_FEATURES)
    info: list[tuple[str, int, int, int]] = []
    for cat_idx, (name, slots) in enumerate(getattr(RL_CONFIG, "entity_categories", ())):
        slots_i = int(slots)
        info.append((str(name), offset, slots_i, cat_idx))
        offset += 1 + (slots_i * slot_features)
    return info, offset


def _detect_base_state_size(total_size: int) -> int:
    total = max(0, int(total_size))
    if total <= 0:
        return max(1, int(getattr(RL_CONFIG, "base_state_size", SERVER_CONFIG.params_count)))
    candidates: list[int] = []
    for cand in {
        int(getattr(RL_CONFIG, "base_state_size", SERVER_CONFIG.params_count)),
        _hybrid_base_state_size(),
        _legacy_slot_layout()[1],
    }:
        if cand > 0 and total >= cand and (total % cand) == 0:
            candidates.append(int(cand))
    if candidates:
        return max(candidates)
    return max(1, int(getattr(RL_CONFIG, "base_state_size", SERVER_CONFIG.params_count)))


def _latest_prev_frame_state(state: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    s = np.asarray(state, dtype=np.float32).reshape(-1)
    base_state_size = _detect_base_state_size(s.size)
    if base_state_size <= 0:
        return s, s, max(1, s.size)
    stack_depth = max(1, s.size // base_state_size)
    latest_off = (stack_depth - 1) * base_state_size
    latest = s[latest_off: latest_off + base_state_size]
    if stack_depth >= 2:
        prev_off = (stack_depth - 2) * base_state_size
        prev = s[prev_off: prev_off + base_state_size]
    else:
        prev = latest
    return latest, prev, base_state_size


def _split_latest_sections(state: np.ndarray):
    s, _prev, _base = _latest_prev_frame_state(state)
    g, grid_n, tc, tf, gc = _state_layout()
    need = g + grid_n + (tc * tf)
    if s.size < need:
        return None, None, None
    globals_ = s[:g]
    grid = s[g:g + grid_n].reshape(gc, int(getattr(RL_CONFIG, "grid_height", 12)), int(getattr(RL_CONFIG, "grid_width", 12)))
    tokens = s[g + grid_n:g + grid_n + (tc * tf)].reshape(tc, tf)
    return globals_, grid, tokens


def _legacy_slot_tokens_from_state(state: np.ndarray) -> np.ndarray:
    latest, prev, base_state_size = _latest_prev_frame_state(state)
    cat_info, legacy_size = _legacy_slot_layout()
    if base_state_size != legacy_size or latest.size < legacy_size:
        return np.zeros((0, 15), dtype=np.float32)

    slot_features = int(getattr(RL_CONFIG, "slot_state_features", LEGACY_SLOT_STATE_FEATURES))
    rows: list[list[float]] = []
    type_den = max(1, UNIFIED_NUM_TYPES - 1)

    for _name, base, slots, _cat_idx in cat_info:
        latest_block = latest[base + 1: base + 1 + slots * slot_features].reshape(slots, slot_features)
        prev_block = prev[base + 1: base + 1 + slots * slot_features].reshape(slots, slot_features)

        for slot_idx in range(slots):
            if float(latest_block[slot_idx, 0]) < 0.5:
                continue
            dx = float(latest_block[slot_idx, 1])
            dy = float(latest_block[slot_idx, 2])
            dist = float(latest_block[slot_idx, 3])
            if not np.isfinite(dist):
                continue

            vx = float(latest_block[slot_idx, 4])
            vy = float(latest_block[slot_idx, 5])
            threat = float(latest_block[slot_idx, 6])
            approach = float(latest_block[slot_idx, 7])

            hit_w_norm = None
            hit_h_norm = None
            if slot_features >= 11:
                hit_w_norm = float(latest_block[slot_idx, 8])
                hit_h_norm = float(latest_block[slot_idx, 9])
                type_id_norm = float(latest_block[slot_idx, 10])
            else:
                type_id_norm = float(latest_block[slot_idx, 8])
            type_id = int(round(type_id_norm * type_den))
            type_id = max(0, min(UNIFIED_NUM_TYPES - 1, type_id))
            type_name = UNIFIED_TYPE_NAMES[type_id]

            is_human = 1.0 if type_id == UNIFIED_HUMAN_TYPE_ID else 0.0
            is_dangerous = 1.0 if type_name in _LEGACY_DANGEROUS_CATEGORIES else 0.0
            cat_norm = type_id_norm  # Preserve normalised type for backward compat

            fallback_w_px, fallback_h_px = _LEGACY_CATEGORY_BOX_PX.get(type_name, (8.0, 8.0))
            if hit_w_norm is None or not np.isfinite(hit_w_norm) or hit_w_norm <= 0.0:
                hit_w_norm = max(0.0, min(1.0, fallback_w_px / 16.0))
            else:
                hit_w_norm = max(0.0, min(1.0, hit_w_norm))
            if hit_h_norm is None or not np.isfinite(hit_h_norm) or hit_h_norm <= 0.0:
                hit_h_norm = max(0.0, min(1.0, fallback_h_px / 16.0))
            else:
                hit_h_norm = max(0.0, min(1.0, hit_h_norm))

            world_dx = dx * _REL_POS_X_RANGE
            world_dy = dy * _REL_POS_Y_RANGE
            world_dist = math.hypot(world_dx, world_dy)
            if world_dist > 1e-6:
                dir_x = world_dx / world_dist
                dir_y = world_dy / world_dist
            else:
                dir_x = 0.0
                dir_y = 0.0

            rows.append([
                1.0,
                dx,
                dy,
                vx,
                vy,
                dist,
                dir_x,
                dir_y,
                threat,
                hit_w_norm,
                hit_h_norm,
                cat_norm,
                is_human,
                is_dangerous,
                approach,
            ])

    if not rows:
        return np.zeros((0, 15), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def _active_tokens_from_state(state: np.ndarray) -> np.ndarray:
    latest = _latest_frame_state(state)
    if latest.size == _legacy_slot_layout()[1]:
        return _legacy_slot_tokens_from_state(state)
    _, _, tokens = _split_latest_sections(state)
    if tokens is None or tokens.size == 0:
        return np.zeros((0, int(getattr(RL_CONFIG, "object_token_features", 15))), dtype=np.float32)
    active = tokens[tokens[:, 0] > 0.5]
    if active.size == 0:
        return np.zeros((0, tokens.shape[1]), dtype=np.float32)
    return active


def _token_category_name(tok: np.ndarray) -> str:
    """Decode the entity type name from a legacy slot token's cat_norm field.

    With unified pool, cat_norm (index 11) stores the normalised type_id:
        type_id = round(cat_norm * (UNIFIED_NUM_TYPES - 1))
    """
    if tok.shape[0] < 12:
        return ""
    type_den = max(1, UNIFIED_NUM_TYPES - 1)
    idx = int(round(float(tok[11]) * type_den))
    idx = max(0, min(UNIFIED_NUM_TYPES - 1, idx))
    return UNIFIED_TYPE_NAMES[idx]


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
    """Return nearest robot candidate for safe standoff alignment (non-hulk, non-obstacle)."""
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


def _preferred_align_robot_vector_from_state(state: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """Prefer visible spheroids for alignment; otherwise fall back to nearest alignable robot."""
    nearest_spheroid = _nearest_category_vector_from_state(state, {"spawner"})
    if nearest_spheroid is not None:
        return nearest_spheroid
    return _nearest_align_robot_vector_from_state(state)


def _nearest_endgame_cleanup_vector_from_state(state: np.ndarray) -> Optional[Tuple[float, float, float]]:
    toks = _active_tokens_from_state(state)
    if toks.shape[0] == 0:
        return None
    best: Optional[Tuple[float, float, float]] = None
    for tok in toks:
        name = _token_category_name(tok)
        if name not in _ENDGAME_CLEANUP_CATEGORIES:
            continue
        dist = float(tok[5])
        if not np.isfinite(dist):
            continue
        if best is None or dist < best[2]:
            best = (float(tok[1]), float(tok[2]), dist)
    return best


def _count_humans_and_cleanup_targets(state: np.ndarray) -> tuple[int, int]:
    toks = _active_tokens_from_state(state)
    if toks.shape[0] == 0:
        return 0, 0
    humans = 0
    cleanup_targets = 0
    for tok in toks:
        if tok[12] > 0.5:
            humans += 1
            continue
        if _token_category_name(tok) in _ENDGAME_CLEANUP_CATEGORIES:
            cleanup_targets += 1
    return humans, cleanup_targets


def _count_humans_and_destructible_enemies(state: np.ndarray) -> tuple[int, int]:
    toks = _active_tokens_from_state(state)
    if toks.shape[0] == 0:
        return 0, 0
    humans = 0
    destructible = 0
    for tok in toks:
        if tok[12] > 0.5:
            humans += 1
            continue
        if _token_category_name(tok) in _ALIGN_ROBOT_CATEGORIES:
            destructible += 1
    return humans, destructible


def _defensive_fire_direction_from_state(state: np.ndarray) -> Optional[int]:
    """Only return a shot for nearby threats that justify interrupting a rescue."""
    close_projectile = _nearest_projectile_vector_from_state(state)
    if close_projectile is not None and close_projectile[2] <= _PROJECTILE_DANGER_DIST:
        px, py, _ = close_projectile
        return _closest_dir8(px * _REL_POS_X_RANGE, py * _REL_POS_Y_RANGE, default_dir=0)

    close_robot = _nearest_category_vector_from_state(state, _ENDGAME_CLEANUP_CATEGORIES)
    if close_robot is not None and close_robot[2] <= _ALIGN_SAFE_DIST:
        ex, ey, _ = close_robot
        return _closest_dir8(ex * _REL_POS_X_RANGE, ey * _REL_POS_Y_RANGE, default_dir=0)
    return None


def _strategic_threat_pressure(state: np.ndarray) -> tuple[float, int]:
    toks = _active_tokens_from_state(state)
    if toks.shape[0] == 0:
        return 0.0, 0
    pressure = 0.0
    projectile_count = 0
    for tok in toks:
        if tok[12] > 0.5:
            continue
        name = _token_category_name(tok)
        if name in {"human", "electrode", ""}:
            continue
        dist = float(tok[5])
        if not np.isfinite(dist):
            continue
        prox = max(0.0, 1.0 - min(1.0, dist / max(_SAFE_DIST, 1e-6)))
        weight = 1.0
        if name == "projectile":
            projectile_count += 1
            weight = 1.75
        elif name in {"brain", "tank", "enforcer", "spawner"}:
            weight = 1.25
        elif name == "hulk":
            weight = 1.5
        pressure += weight * (0.35 + prox)
    return pressure, projectile_count


def _perimeter_orbit_move(state: np.ndarray, fire_dir: int) -> int:
    """Bias movement outward first, then tangentially around the arena edge."""
    s = _latest_frame_state(state)
    px = float(s[5]) if s.size > 6 else 0.5
    py = float(s[6]) if s.size > 6 else 0.5

    rel_x = px - 0.5
    rel_y = py - 0.5
    radial_len = math.hypot(rel_x, rel_y)
    if radial_len <= 1e-6:
        rel_x, rel_y = 0.0, 1.0
        radial_len = 1.0
    radial_x = rel_x / radial_len
    radial_y = rel_y / radial_len

    fire_dx, fire_dy = _move_dir_vector(int(fire_dir) % max(1, len(_DIR8_VECTORS)))
    clockwise = ((radial_x * fire_dy) - (radial_y * fire_dx)) >= 0.0
    tangent_x = radial_y if clockwise else -radial_y
    tangent_y = -radial_x if clockwise else radial_x

    ring_push = max(0.0, (_PERIMETER_ORBIT_RING_TARGET - radial_len) / max(_PERIMETER_ORBIT_RING_TARGET, 1e-6))
    outward_w = 0.35 + (0.95 * ring_push)
    tangent_w = 1.05 if radial_len >= _PERIMETER_ORBIT_RING_MIN else 0.45
    move_vx = (radial_x * outward_w) + (tangent_x * tangent_w)
    move_vy = (radial_y * outward_w) + (tangent_y * tangent_w)
    return _closest_dir8(move_vx, move_vy, default_dir=0)


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


def _nearest_category_vector_from_state(
    state: np.ndarray,
    categories: set[str],
) -> Optional[Tuple[float, float, float]]:
    toks = _active_tokens_from_state(state)
    if toks.shape[0] == 0:
        return None
    best = None
    allowed = set(categories)
    for tok in toks:
        if _token_category_name(tok) not in allowed:
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
        center_x = (0.5 * _PLAYER_BOX_W_WORLD) + (dx * _REL_POS_X_RANGE)
        center_y = (0.5 * _PLAYER_BOX_H_WORLD) + (dy * _REL_POS_Y_RANGE)
        width_px = max(1.0, float(tok[9]) * 16.0)
        height_px = max(1.0, float(tok[10]) * 16.0)
        box_w = width_px * _WORLD_UNITS_PER_PIXEL
        box_h = height_px * _WORLD_UNITS_PER_PIXEL
        box_x = center_x - (0.5 * box_w)
        box_y = center_y - (0.5 * box_h)
        pad_px = _AVOIDANCE_BASE_PADDING_PX + _AVOIDANCE_PADDING_PX_BY_CATEGORY.get(name, 0.5)
        pad_world = pad_px * _WORLD_UNITS_PER_PIXEL
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


def _nearest_aligned_fire_direction_from_state(
    state: np.ndarray,
    categories: Optional[set[str]] = None,
) -> Optional[int]:
    """Return fire direction for the closest non-human target aligned to any 8-way shot."""
    toks = _active_tokens_from_state(state)
    if toks.shape[0] == 0:
        return None
    allowed_categories = _ALIGNED_FIRE_CATEGORIES if categories is None else set(categories)

    per_direction: list[Optional[tuple[float, int]]] = [None] * len(_DIR8_VECTORS)
    for tok in toks:
        name = _token_category_name(tok)
        if name not in allowed_categories:
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


def get_cleanup_fire_override(state: np.ndarray, locked_fire: Optional[int] = None) -> Optional[int]:
    """Return an obvious endgame cleanup shot, or None if no override is warranted."""
    if locked_fire is not None and int(locked_fire) >= 0:
        return None
    humans, cleanup_targets = _count_humans_and_cleanup_targets(state)
    if humans > 0 or cleanup_targets <= 0 or cleanup_targets > 2:
        return None
    return _nearest_aligned_fire_direction_from_state(state, categories=_ENDGAME_CLEANUP_CATEGORIES)


def _priority_spawn_fire_direction_from_state(
    state: np.ndarray,
    nearest_enemy: Optional[Tuple[float, float, float]] = None,
    nearest_projectile: Optional[Tuple[float, float, float]] = None,
) -> Optional[int]:
    """Prefer spawn-class targets (spheroids/tanks), but never over an immediate projectile shot."""
    if nearest_projectile is not None and nearest_projectile[2] <= _PROJECTILE_DANGER_DIST:
        projectile_fire_dir = _nearest_aligned_fire_direction_from_state(state, categories={"projectile"})
        if projectile_fire_dir is not None:
            return int(projectile_fire_dir)

    aligned_spawn_fire_dir = _nearest_aligned_fire_direction_from_state(state, categories={"spawner", "tank"})
    if aligned_spawn_fire_dir is not None:
        return int(aligned_spawn_fire_dir)

    nearest_spawn_target = _nearest_category_vector_from_state(state, {"spawner", "tank"})
    if nearest_spawn_target is None:
        return None

    close_threat = False
    if nearest_enemy is not None and nearest_enemy[2] < _ALIGN_SAFE_DIST:
        close_threat = True
    if nearest_projectile is not None and nearest_projectile[2] < _PROJECTILE_DANGER_DIST:
        close_threat = True
    if close_threat:
        return None

    sx, sy, _ = nearest_spawn_target
    return _closest_dir8(sx * _REL_POS_X_RANGE, sy * _REL_POS_Y_RANGE, default_dir=0)


def _get_simple_expert_action(
    state: np.ndarray,
    locked_fire: Optional[int] = None,
    wave_number: Optional[int] = None,
) -> Tuple[int, int]:
    """Simpler, more stable heuristic Robotron expert.

    Fire:
        1) If a projectile is dangerously close and aligned, clear it.
        2) Otherwise prefer spawn-class targets (spheroids/tanks) before they can spawn more danger.
        3) If any enemy/projectile is aligned with one of the 8 fire directions
           within an 8 px half-window, shoot the closest aligned target.
        4) Otherwise, fallback to shooting the nearest enemy.
     Move priority:
        1) Rescue humans when safe.
        2) Flee close threats (including hulks).
        3) If configured, stop shooting the last distant enemy while rescuing
           remaining humans, only firing defensively at close threats.
        4) Otherwise, align to the short axis of the nearest non-hulk robot.
        5) Fallback to fleeing the nearest threat.
    """
    nearest_enemy = _nearest_enemy_vector_from_state(state)
    nearest_projectile = _nearest_projectile_vector_from_state(state)
    nearest_align_robot = _preferred_align_robot_vector_from_state(state)
    nearest_endgame_cleanup = _nearest_endgame_cleanup_vector_from_state(state)
    nearest_human = _nearest_human_vector_from_state(state)
    nearby_avoidance = _nearby_avoidance_vectors_from_state(state)
    aligned_fire_dir = _nearest_aligned_fire_direction_from_state(state)
    priority_spawn_fire_dir = _priority_spawn_fire_direction_from_state(
        state,
        nearest_enemy=nearest_enemy,
        nearest_projectile=nearest_projectile,
    )
    humans_remaining, _cleanup_targets = _count_humans_and_cleanup_targets(state)
    humans_for_rescue_mode, destructible_enemies = _count_humans_and_destructible_enemies(state)
    hold_fire_for_last_enemy = bool(getattr(RL_CONFIG, "expert_hold_fire_for_last_enemy_rescue", True))
    last_enemy_rescue_mode = (
        hold_fire_for_last_enemy
        and humans_for_rescue_mode > 0
        and destructible_enemies <= 1
    )
    s = _latest_frame_state(state)
    px = float(s[5]) if s.size > 6 else 0.5
    py = float(s[6]) if s.size > 6 else 0.5

    # ── Fire direction ───────────────────────────────────────────────
    if last_enemy_rescue_mode:
        defensive_fire_dir = _defensive_fire_direction_from_state(state)
        fire_dir = int(defensive_fire_dir) if defensive_fire_dir is not None else 0
    else:
        if priority_spawn_fire_dir is not None:
            fire_dir = int(priority_spawn_fire_dir)
        elif aligned_fire_dir is not None:
            fire_dir = int(aligned_fire_dir)
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
            move_dir = _closest_dir8(-ex_world, -ey_world, default_dir=0)
        # 2) Safe to rescue: prioritize humans when available.
        elif nearest_human is not None:
            hx, hy, _ = nearest_human
            move_dir = _closest_dir8(hx * _REL_POS_X_RANGE, hy * _REL_POS_Y_RANGE)
        # 3) Safe standoff: align on one axis with nearest non-hulk robot.
        elif nearest_align_robot is not None:
            ax, ay, _ = nearest_align_robot
            move_dir = _axis_align_toward_enemy(ax * _REL_POS_X_RANGE, ay * _REL_POS_Y_RANGE)
        elif humans_remaining <= 0 and nearest_endgame_cleanup is not None:
            ax, ay, _ = nearest_endgame_cleanup
            move_dir = _axis_align_toward_enemy(ax * _REL_POS_X_RANGE, ay * _REL_POS_Y_RANGE)
        else:
            # 4) Fallback: flee nearest enemy.
            move_dir = _closest_dir8(-ex_world, -ey_world, default_dir=0)
    elif nearest_human is not None:
        hx, hy, _ = nearest_human
        move_dir = _closest_dir8(hx * _REL_POS_X_RANGE, hy * _REL_POS_Y_RANGE)
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

    return move_dir, fire_dir


def _get_strategic_expert_action(
    state: np.ndarray,
    locked_fire: Optional[int] = None,
    wave_number: Optional[int] = None,
) -> Tuple[int, int]:
    """Heuristic Robotron expert with strategic late-wave modes."""
    nearest_enemy = _nearest_enemy_vector_from_state(state)
    nearest_projectile = _nearest_projectile_vector_from_state(state)
    nearest_priority_hunt = _nearest_category_vector_from_state(state, {"brain", "tank", "spawner"})
    nearest_align_robot = _preferred_align_robot_vector_from_state(state)
    nearest_endgame_cleanup = _nearest_endgame_cleanup_vector_from_state(state)
    nearest_human = _nearest_human_vector_from_state(state)
    nearby_avoidance = _nearby_avoidance_vectors_from_state(state)
    aligned_fire_dir = _nearest_aligned_fire_direction_from_state(state)
    priority_spawn_fire_dir = _priority_spawn_fire_direction_from_state(
        state,
        nearest_enemy=nearest_enemy,
        nearest_projectile=nearest_projectile,
    )
    humans_remaining, _cleanup_targets = _count_humans_and_cleanup_targets(state)
    humans_for_rescue_mode, destructible_enemies = _count_humans_and_destructible_enemies(state)
    threat_pressure, projectile_count = _strategic_threat_pressure(state)
    wave = max(0, int(wave_number or 0))
    rescue_wave = (wave > 0) and ((wave % 5) == 0)
    tank_wave = (wave >= 7) and (((wave - 7) % 5) == 0)
    protect_humans_wave = rescue_wave or tank_wave
    hold_fire_for_last_enemy = bool(getattr(RL_CONFIG, "expert_hold_fire_for_last_enemy_rescue", True))
    last_enemy_rescue_mode = (
        hold_fire_for_last_enemy
        and humans_for_rescue_mode > 0
        and destructible_enemies <= 1
    )
    s = _latest_frame_state(state)
    px = float(s[5]) if s.size > 6 else 0.5
    py = float(s[6]) if s.size > 6 else 0.5
    radial_dist = math.hypot(px - 0.5, py - 0.5)
    near_human_rescue_ok = False
    if nearest_human is not None:
        _hx, _hy, human_dist = nearest_human
        near_human_rescue_ok = (
            human_dist <= _RESCUE_NEAR_DIST
            and projectile_count <= _RESCUE_ABORT_PROJECTILES
            and threat_pressure < _RESCUE_ABORT_PRESSURE
        )

    rescue_projectile_fire_dir = None
    rescue_hunt_fire_dir = None
    if last_enemy_rescue_mode:
        defensive_fire_dir = _defensive_fire_direction_from_state(state)
        fire_dir = int(defensive_fire_dir) if defensive_fire_dir is not None else 0
    else:
        if protect_humans_wave:
            if nearest_projectile is not None and nearest_projectile[2] <= _PROJECTILE_DANGER_DIST:
                rescue_projectile_fire_dir = _nearest_aligned_fire_direction_from_state(state, categories={"projectile"})
            rescue_hunt_fire_dir = _nearest_aligned_fire_direction_from_state(state, categories={"brain", "tank", "spawner"})
        if rescue_projectile_fire_dir is not None:
            fire_dir = int(rescue_projectile_fire_dir)
        elif rescue_hunt_fire_dir is not None:
            fire_dir = int(rescue_hunt_fire_dir)
        elif protect_humans_wave and nearest_priority_hunt is not None:
            fx, fy, _ = nearest_priority_hunt
            fire_dir = _closest_dir8(fx * _REL_POS_X_RANGE, fy * _REL_POS_Y_RANGE, default_dir=0)
        elif priority_spawn_fire_dir is not None:
            fire_dir = int(priority_spawn_fire_dir)
        elif aligned_fire_dir is not None:
            fire_dir = int(aligned_fire_dir)
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

        if is_close_threat:
            move_dir = _closest_dir8(-ex_world, -ey_world, default_dir=0)
        elif nearest_human is not None and threat_pressure < (_PERIMETER_ORBIT_PRESSURE_THRESHOLD - 1.0):
            hx, hy, _ = nearest_human
            move_dir = _closest_dir8(hx * _REL_POS_X_RANGE, hy * _REL_POS_Y_RANGE)
        elif last_enemy_rescue_mode and nearest_human is not None:
            hx, hy, _ = nearest_human
            move_dir = _closest_dir8(hx * _REL_POS_X_RANGE, hy * _REL_POS_Y_RANGE)
        elif protect_humans_wave and nearest_human is not None:
            hx, hy, _ = nearest_human
            move_dir = _closest_dir8(hx * _REL_POS_X_RANGE, hy * _REL_POS_Y_RANGE)
        elif near_human_rescue_ok:
            hx, hy, _ = nearest_human
            move_dir = _closest_dir8(hx * _REL_POS_X_RANGE, hy * _REL_POS_Y_RANGE)
        elif (not protect_humans_wave) and (not last_enemy_rescue_mode) and (
            (threat_pressure + (projectile_count * _PERIMETER_ORBIT_PROJECTILE_BONUS)) >= _PERIMETER_ORBIT_PRESSURE_THRESHOLD or (
                threat_pressure >= _PERIMETER_ORBIT_PRESSURE_THRESHOLD and radial_dist < _PERIMETER_ORBIT_RING_MIN
            )
        ):
            move_dir = _perimeter_orbit_move(state, fire_dir)
        elif nearest_align_robot is not None:
            ax, ay, _ = nearest_align_robot
            move_dir = _axis_align_toward_enemy(ax * _REL_POS_X_RANGE, ay * _REL_POS_Y_RANGE)
        elif humans_remaining <= 0 and nearest_endgame_cleanup is not None:
            ax, ay, _ = nearest_endgame_cleanup
            move_dir = _axis_align_toward_enemy(ax * _REL_POS_X_RANGE, ay * _REL_POS_Y_RANGE)
        else:
            move_dir = _closest_dir8(-ex_world, -ey_world, default_dir=0)
    elif nearest_human is not None:
        hx, hy, _ = nearest_human
        move_dir = _closest_dir8(hx * _REL_POS_X_RANGE, hy * _REL_POS_Y_RANGE)
    else:
        move_dir = _move_idle_action_index()

    intended_move_dir = move_dir
    move_dir = _apply_final_hazard_move_check(move_dir, nearby_avoidance)
    if (
        locked_fire is None or int(locked_fire) < 0
    ) and move_dir != intended_move_dir:
        blocked_fire_dir = _blocking_hazard_fire_direction(intended_move_dir, nearby_avoidance)
        if blocked_fire_dir is not None:
            fire_dir = int(blocked_fire_dir)

    return move_dir, fire_dir


def get_expert_action(
    state: np.ndarray,
    locked_fire: Optional[int] = None,
    wave_number: Optional[int] = None,
) -> Tuple[int, int]:
    profile = str(getattr(RL_CONFIG, "expert_profile", "simple") or "simple").strip().lower()
    if profile == "strategic":
        return _get_strategic_expert_action(state, locked_fire=locked_fire, wave_number=wave_number)
    return _get_simple_expert_action(state, locked_fire=locked_fire, wave_number=wave_number)

# ── Enemy-Slot Attention Pooling ────────────────────────────────────────────
class EnemyAttention(nn.Module):
    """Encode object slots, then pool them with a learned attention query."""

    def __init__(self, slot_features: int, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(slot_features, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )
        self.query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.out_dim = embed_dim
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.embed.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
        nn.init.normal_(self.query, mean=0.0, std=0.02)
        try:
            self.attn._reset_parameters()
        except Exception:
            pass

    def encode(self, slots: torch.Tensor) -> torch.Tensor:
        return self.embed(slots)

    def forward(
        self,
        slots: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_weights: bool = False,
        encoded: bool = False,
    ):
        """slots: (B, num_slots, features or embed_dim) → (B, embed_dim)."""
        x = slots if encoded else self.encode(slots)
        bsz = x.shape[0]
        query = self.query.expand(bsz, -1, -1)
        attn_out, attn_weights = self.attn(
            query,
            x,
            x,
            key_padding_mask=mask,
            average_attn_weights=False,
        )  # (B, 1, D), (B, H, 1, S)
        pooled = attn_out[:, 0, :]
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


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0, use_layer_norm: bool = True):
        super().__init__()
        hidden = dim * 2
        self.norm1 = nn.LayerNorm(dim) if use_layer_norm else nn.Identity()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        h = self.fc2(h)
        return x + h


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
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(embed_dim)
        self.out_dim = embed_dim
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

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


# ── Directional Lane Encoder ───────────────────────────────────────────────
# 8-direction lane system inspired by the Tempest lane-cross-attention encoder.
# Entities are binned into 8 angular wedges (matching fire/move directions)
# relative to the player.  Each lane token summarises the nearest threats and
# humans in that direction.  Cross-attention enriches lane tokens with the
# full entity set, producing an 8-lane spatial summary vector.

_NUM_LANES = 8
# Unit vectors for the 8 canonical directions (right=0, then CCW in game coords).
# Game Y axis is inverted (down is positive), so Up = (0, -1).
_LANE_DIR_X = torch.tensor([1.0, 0.7071, 0.0, -0.7071, -1.0, -0.7071,  0.0,  0.7071])
_LANE_DIR_Y = torch.tensor([0.0, -0.7071, -1.0, -0.7071,  0.0,  0.7071, 1.0,  0.7071])

# Features per lane token (built from aggregated entity data):
#   nearest_enemy_dist, nearest_enemy_dx, nearest_enemy_dy,
#   nearest_enemy_vx, nearest_enemy_vy, nearest_enemy_threat, nearest_enemy_approach,
#   enemy_count_in_lane,
#   nearest_human_dist, nearest_human_dx, nearest_human_dy,
#   human_count_in_lane,
#   nearest_electrode_dist,
#   sin_dir, cos_dir
_LANE_TOKEN_FEATURES = 15


class DirectionalLaneEncoder(nn.Module):
    """Bin entities into 8 directional lanes and cross-attend.

    Lane tokens (8 × 15 features) serve as queries; all entity tokens serve
    as keys/values.  The resulting enriched lane representation is gated-pooled
    (learned per-lane importance weights) into a fixed-size summary vector for
    the MLP trunk.  Empty lanes are masked out of the pooling.
    """

    def __init__(self, entity_token_features: int, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim

        # Lane embedding
        self.lane_embed = nn.Linear(_LANE_TOKEN_FEATURES, embed_dim)
        self.lane_norm = nn.LayerNorm(embed_dim)

        # Entity embedding (reuses the same token schema as EntitySetEncoder)
        self.entity_embed = nn.Linear(entity_token_features, embed_dim)
        self.entity_norm = nn.LayerNorm(embed_dim)

        # Cross-attention: lanes (Q) attend to entities (K, V)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(embed_dim)

        # Attention-weighted pooling: learnable gate produces per-lane
        # importance so empty lanes are down-weighted instead of diluting
        # the signal via uniform mean-pooling.
        self.pool_gate = nn.Linear(embed_dim, 1)

        self.out_dim = embed_dim

        # Pre-compute lane direction encodings
        angles = torch.arange(_NUM_LANES, dtype=torch.float32) * (2.0 * math.pi / _NUM_LANES)
        self.register_buffer('_lane_sin', torch.sin(angles))
        self.register_buffer('_lane_cos', torch.cos(angles))
        self.register_buffer('_lane_dir_x', _LANE_DIR_X.clone())
        self.register_buffer('_lane_dir_y', _LANE_DIR_Y.clone())

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _build_lane_tokens(
        self,
        entity_dx: torch.Tensor,     # (B, S)
        entity_dy: torch.Tensor,     # (B, S)
        entity_dist: torch.Tensor,   # (B, S)
        entity_vx: torch.Tensor,     # (B, S)
        entity_vy: torch.Tensor,     # (B, S)
        entity_threat: torch.Tensor, # (B, S)
        entity_approach: torch.Tensor,  # (B, S)
        entity_present: torch.Tensor,   # (B, S) bool
        entity_is_human: torch.Tensor,  # (B, S) bool
        entity_is_electrode: torch.Tensor,  # (B, S) bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Aggregate per-entity data into 8 directional lane tokens.

        Returns:
            lane_tokens: (B, 8, 15) directional aggregate features.
            lane_has_entity: (B, 8) bool mask indicating non-empty lanes.
        """
        B, S = entity_dx.shape
        device = entity_dx.device

        # Compute angle from player to each entity.
        # Lua normalises dx by POS_X_RANGE and dy by POS_Y_RANGE (different
        # values), so we must unscale to world units before atan2 to get
        # true geometric angles.  We also negate dy because Robotron's
        # screen Y axis is inverted (down = +).
        world_dx = entity_dx * _REL_POS_X_RANGE
        world_dy = entity_dy * _REL_POS_Y_RANGE
        angles = torch.atan2(-world_dy, world_dx)  # (B, S) in [-pi, pi]
        # Shift to [0, 2pi) then quantise to 0..7 using floor(x + 0.5)
        # instead of round() to avoid banker's rounding at wedge boundaries.
        angles_pos = (angles + 2.0 * math.pi) % (2.0 * math.pi)
        wedge = 2.0 * math.pi / _NUM_LANES
        lane_idx = torch.floor(angles_pos / wedge + 0.5).long() % _NUM_LANES

        # Masks for entity types
        is_enemy = entity_present & ~entity_is_human & ~entity_is_electrode
        is_human = entity_present & entity_is_human
        is_electrode = entity_present & entity_is_electrode

        # Use very large distance for absent/wrong-type entities so they lose min()
        BIG = 999.0
        enemy_dist = torch.where(is_enemy, entity_dist, torch.full_like(entity_dist, BIG))
        human_dist = torch.where(is_human, entity_dist, torch.full_like(entity_dist, BIG))
        electrode_dist = torch.where(is_electrode, entity_dist, torch.full_like(entity_dist, BIG))

        # Build per-lane aggregates using scatter
        lane_tokens = torch.zeros(B, _NUM_LANES, _LANE_TOKEN_FEATURES, device=device, dtype=entity_dx.dtype)

        for lane_i in range(_NUM_LANES):
            in_lane = (lane_idx == lane_i)  # (B, S)

            # ── Nearest enemy in this lane ──
            e_mask = in_lane & is_enemy
            e_dist = torch.where(e_mask, enemy_dist, torch.full_like(enemy_dist, BIG))
            nearest_e_idx = e_dist.argmin(dim=1)  # (B,)
            nearest_e_dist = e_dist.gather(1, nearest_e_idx.unsqueeze(1)).squeeze(1)
            has_enemy = nearest_e_dist < BIG

            b_idx = torch.arange(B, device=device)
            lane_tokens[:, lane_i, 0] = torch.where(has_enemy, nearest_e_dist, torch.zeros_like(nearest_e_dist))
            lane_tokens[:, lane_i, 1] = torch.where(has_enemy, entity_dx[b_idx, nearest_e_idx], torch.zeros_like(nearest_e_dist))
            lane_tokens[:, lane_i, 2] = torch.where(has_enemy, entity_dy[b_idx, nearest_e_idx], torch.zeros_like(nearest_e_dist))
            lane_tokens[:, lane_i, 3] = torch.where(has_enemy, entity_vx[b_idx, nearest_e_idx], torch.zeros_like(nearest_e_dist))
            lane_tokens[:, lane_i, 4] = torch.where(has_enemy, entity_vy[b_idx, nearest_e_idx], torch.zeros_like(nearest_e_dist))
            lane_tokens[:, lane_i, 5] = torch.where(has_enemy, entity_threat[b_idx, nearest_e_idx], torch.zeros_like(nearest_e_dist))
            lane_tokens[:, lane_i, 6] = torch.where(has_enemy, entity_approach[b_idx, nearest_e_idx], torch.zeros_like(nearest_e_dist))
            # Enemy count in lane (normalise by 50 — enough to distinguish
            # "busy" from "overwhelming" even on crowded late waves)
            lane_tokens[:, lane_i, 7] = e_mask.float().sum(dim=1).clamp(max=50.0) / 50.0

            # ── Nearest human in this lane ──
            h_mask = in_lane & is_human
            h_dist = torch.where(h_mask, human_dist, torch.full_like(human_dist, BIG))
            nearest_h_idx = h_dist.argmin(dim=1)
            nearest_h_dist = h_dist.gather(1, nearest_h_idx.unsqueeze(1)).squeeze(1)
            has_human = nearest_h_dist < BIG
            lane_tokens[:, lane_i, 8] = torch.where(has_human, nearest_h_dist, torch.zeros_like(nearest_h_dist))
            lane_tokens[:, lane_i, 9] = torch.where(has_human, entity_dx[b_idx, nearest_h_idx], torch.zeros_like(nearest_h_dist))
            lane_tokens[:, lane_i, 10] = torch.where(has_human, entity_dy[b_idx, nearest_h_idx], torch.zeros_like(nearest_h_dist))
            lane_tokens[:, lane_i, 11] = h_mask.float().sum(dim=1).clamp(max=16.0) / 16.0

            # ── Nearest electrode in this lane ──
            el_mask = in_lane & is_electrode
            el_dist = torch.where(el_mask, electrode_dist, torch.full_like(electrode_dist, BIG))
            nearest_el_dist = el_dist.min(dim=1).values
            lane_tokens[:, lane_i, 12] = torch.where(nearest_el_dist < BIG, nearest_el_dist, torch.zeros_like(nearest_el_dist))

            # ── Positional encoding for this lane direction ──
            lane_tokens[:, lane_i, 13] = self._lane_sin[lane_i]
            lane_tokens[:, lane_i, 14] = self._lane_cos[lane_i]

        # Per-lane occupancy: True if at least one present entity is in this lane.
        lane_has_entity = torch.zeros(B, _NUM_LANES, dtype=torch.bool, device=device)
        for lane_i in range(_NUM_LANES):
            lane_has_entity[:, lane_i] = ((lane_idx == lane_i) & entity_present).any(dim=1)

        return lane_tokens, lane_has_entity

    def forward(
        self,
        lane_tokens: torch.Tensor,      # (B, 8, 15)
        entity_tokens: torch.Tensor,    # (B, S, entity_features)
        entity_mask: torch.Tensor,      # (B, S) bool — True = empty/absent
        lane_mask: torch.Tensor,        # (B, 8) bool — True = lane has entities
    ) -> torch.Tensor:
        """Returns (B, embed_dim) gated-pooled lane summary."""
        lane_emb = self.lane_norm(self.lane_embed(lane_tokens))        # (B, 8, D)
        entity_emb = self.entity_norm(self.entity_embed(entity_tokens))  # (B, S, D)
        summary = lane_emb.new_zeros((lane_emb.shape[0], self.embed_dim))

        # Avoid all-masked attention rows: PyTorch MHA emits NaNs when every key
        # is masked, and masked softmax over all-empty lanes would otherwise
        # assign a uniform distribution instead of zero weight.
        active_rows = (~entity_mask).any(dim=1) & lane_mask.any(dim=1)
        if not torch.any(active_rows):
            return summary

        active_lane_emb = lane_emb[active_rows]
        active_entity_emb = entity_emb[active_rows]
        active_entity_mask = entity_mask[active_rows]
        active_lane_mask = lane_mask[active_rows]

        enriched, _ = self.cross_attn(
            active_lane_emb, active_entity_emb, active_entity_emb,
            key_padding_mask=active_entity_mask,
        )  # (A, 8, D)

        enriched = self.cross_norm(active_lane_emb + enriched)   # residual + norm
        # Gated pool: learned per-lane importance, empty lanes hard-masked.
        gate_logits = self.pool_gate(enriched).squeeze(-1)       # (A, 8)
        gate_logits = gate_logits.masked_fill(~active_lane_mask, -1e9)
        gate_weights = torch.softmax(gate_logits, dim=1)         # (A, 8)
        summary[active_rows] = (enriched * gate_weights.unsqueeze(-1)).sum(dim=1)
        return summary


# ── Distributional Dueling Network ─────────────────────────────────────────
class RainbowNet(nn.Module):
    """Robotron network with configurable flat-MLP and object-set branches."""

    def __init__(self, state_size: int):
        super().__init__()
        cfg = RL_CONFIG
        self.state_size = state_size
        self.use_dist = cfg.use_distributional
        self.num_atoms = cfg.num_atoms if self.use_dist else 1
        self.v_min = cfg.v_min
        self.v_max = cfg.v_max
        self.use_dueling = cfg.use_dueling
        self.use_factorized_action_heads = bool(getattr(cfg, "factorized_action_heads", False))
        self.num_actions = NUM_JOINT
        self.use_pure_mlp = bool(getattr(cfg, "pure_mlp", False))
        self.use_mlp_with_attention = (
            self.use_pure_mlp
            and bool(getattr(cfg, "mlp_with_attention", False))
            and bool(getattr(cfg, "use_enemy_attention", True))
        )
        self.use_attn = (
            ((not self.use_pure_mlp) or self.use_mlp_with_attention)
            and bool(getattr(cfg, "use_enemy_attention", True))
        )
        self.base_state_size = int(getattr(cfg, "base_state_size", SERVER_CONFIG.params_count))
        self.stack_depth = max(1, int(self.state_size // max(1, self.base_state_size)))
        self.attn_all_frames = bool(getattr(cfg, "attn_all_frames", False))
        self.attn_frame_count = self.stack_depth if self.attn_all_frames else 1
        self.num_object_slots = int(getattr(cfg, "object_slots", 64))
        self.slot_state_features = int(getattr(cfg, "slot_state_features", LEGACY_SLOT_STATE_FEATURES))
        self.core_feature_count = int(LEGACY_CORE_FEATURES + LEGACY_ELIST_FEATURES)
        self._cat_info = []
        self._category_ranges = []
        self._num_categories = 0
        self._occupancy_offsets = []

        # Unified pool: single ("entity", 64) category with per-slot type_id
        cat_defs = getattr(cfg, "entity_categories", [("entity", 64)])
        if self.use_attn:
            offset = self.core_feature_count
            slot_cursor = 0
            for cat_id, (name, slots) in enumerate(cat_defs):
                slots_i = int(slots)
                self._cat_info.append((name, offset, slots_i, cat_id))
                self._category_ranges.append((name, slot_cursor, slot_cursor + slots_i))
                offset += 1 + (slots_i * self.slot_state_features)
                slot_cursor += slots_i
            self._num_categories = len(cat_defs)
            self._occupancy_offsets = [base for _name, base, _slots, _cat_id in self._cat_info]

        # Learned type embedding replaces the old scalar cat_norm / is_human / is_dangerous meta
        self.type_embedding_dim = int(getattr(cfg, "type_embedding_dim", 16))
        self.type_embedding = nn.Embedding(UNIFIED_NUM_TYPES, self.type_embedding_dim)
        # Entity token: 9 raw features + type_emb(16) + is_human(1) + is_dangerous(1) = 27
        entity_raw_features = 9  # dx, dy, vx, vy, dist, threat, approach, hit_w, hit_h
        self._entity_token_dim = entity_raw_features + self.type_embedding_dim + 2
        self.object_token_features = self._entity_token_dim

        if self.use_pure_mlp:
            mlp_hidden = [int(v) for v in getattr(cfg, "mlp_hidden_layers", [1024, 512]) or [1024, 512]]
            mlp_hidden = [max(1, int(v)) for v in mlp_hidden]
            mlp_out = max(1, int(getattr(cfg, "mlp_output_dim", 256) or 256))
            self.mlp_hidden_layers = list(mlp_hidden)
            self.mlp_output_dim = mlp_out

            # ── Directional lane or legacy EntitySetEncoder ────────────
            self.use_directional_lanes = (
                self.use_mlp_with_attention
                and bool(getattr(cfg, "use_directional_lanes", False))
            )
            lane_out_dim = 0
            entity_pool_dim = 0
            attn_dim = int(getattr(cfg, "attn_dim", 256))
            if self.use_directional_lanes:
                # ── Entity input projection: raw token → attn_dim ──────
                self.entity_input_proj = nn.Sequential(
                    nn.Linear(self._entity_token_dim, attn_dim),
                    nn.LayerNorm(attn_dim),
                    nn.ReLU(),
                )
                # ── Entity self-attention: entities see each other ─────
                num_sa_layers = int(getattr(cfg, "entity_self_attn_layers", 2))
                sa_layer = nn.TransformerEncoderLayer(
                    d_model=attn_dim,
                    nhead=int(getattr(cfg, "attn_heads", 4)),
                    dim_feedforward=attn_dim * 4,
                    dropout=float(cfg.dropout),
                    batch_first=True,
                    norm_first=True,
                )
                self.entity_self_attn = nn.TransformerEncoder(sa_layer, num_layers=num_sa_layers)

                # ── Lane cross-attention over self-attended entity tokens
                self.lane_encoder = DirectionalLaneEncoder(
                    entity_token_features=attn_dim,
                    embed_dim=attn_dim,
                    num_heads=int(getattr(cfg, "attn_heads", 4)),
                )
                lane_out_dim = self.lane_encoder.out_dim

                # ── Entity pool: mean-pool self-attended tokens → summary
                entity_pool_dim = attn_dim
                self.entity_pool_proj = nn.Linear(attn_dim, entity_pool_dim)
                self.entity_pool_norm = nn.LayerNorm(entity_pool_dim)
            elif self.use_mlp_with_attention:
                self.entity_input_proj = nn.Sequential(
                    nn.Linear(self._entity_token_dim, attn_dim),
                    nn.LayerNorm(attn_dim),
                    nn.ReLU(),
                )
                entity_hidden = int(getattr(cfg, "entity_hidden", 192))
                self.object_attn = EntitySetEncoder(
                    token_features=attn_dim,
                    embed_dim=attn_dim,
                    num_heads=int(getattr(cfg, "attn_heads", 4)),
                    num_layers=int(getattr(cfg, "attn_layers", 1)),
                )
                self.entity_proj = nn.Sequential(
                    nn.Linear(attn_dim * self.attn_frame_count, entity_hidden),
                    nn.LayerNorm(entity_hidden),
                    nn.ReLU(),
                )

            # MLP trunk: stacked flat state + lane summary + entity pool summary
            # Separate LayerNorm per branch before concatenation so each
            # branch's features start on equal footing at init.
            if self.use_directional_lanes:
                self.state_norm = nn.LayerNorm(int(state_size))
                self.lane_norm_out = nn.LayerNorm(lane_out_dim)
            trunk_in = int(state_size) + lane_out_dim + entity_pool_dim
            layers = []
            in_dim = trunk_in
            for width in mlp_hidden:
                layers.append(nn.Linear(in_dim, width))
                if cfg.use_layer_norm:
                    layers.append(nn.LayerNorm(width))
                layers.append(nn.ReLU())
                if float(cfg.dropout) > 0.0:
                    layers.append(nn.Dropout(float(cfg.dropout)))
                in_dim = width
            layers.append(nn.Linear(in_dim, mlp_out))
            if cfg.use_layer_norm:
                layers.append(nn.LayerNorm(mlp_out))
            layers.append(nn.ReLU())
            if float(cfg.dropout) > 0.0:
                layers.append(nn.Dropout(float(cfg.dropout)))
            self.trunk = nn.Sequential(*layers)
            head_in = mlp_out

            # Legacy EntitySetEncoder fusion (only when NOT using directional lanes)
            if self.use_mlp_with_attention and not self.use_directional_lanes:
                entity_hidden = int(getattr(cfg, "entity_hidden", 192))
                self.mlp_attn_fusion = nn.Sequential(
                    nn.Linear(mlp_out + entity_hidden, mlp_out),
                    nn.LayerNorm(mlp_out) if cfg.use_layer_norm else nn.Identity(),
                    nn.ReLU(),
                )

            head_mid = max(64, head_in // 2)
            self._build_action_heads(head_in, head_mid)
            self._init_weights()
            if self.use_dist:
                support = torch.linspace(self.v_min, self.v_max, self.num_atoms)
                self.register_buffer("support", support)
                self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
            return

        attn_dim = int(getattr(cfg, "attn_dim", 96))
        entity_hidden = int(getattr(cfg, "entity_hidden", 192))
        self.global_encoder = GlobalEncoder(
            in_dim=self.stack_depth * (self.core_feature_count + self._num_categories),
            hidden_dim=int(getattr(cfg, "global_hidden", 128)),
        )
        # Encode each frame's visible object set independently, then fuse all
        # stacked frame summaries so all temporal slices contribute to policy.
        self.object_attn = EntitySetEncoder(
            token_features=self.object_token_features,
            embed_dim=attn_dim,
            num_heads=int(getattr(cfg, "attn_heads", 4)),
            num_layers=int(getattr(cfg, "attn_layers", 1)),
        )
        self.entity_proj = nn.Sequential(
            nn.Linear(attn_dim * self.attn_frame_count, entity_hidden),
            nn.LayerNorm(entity_hidden),
            nn.ReLU(),
        )
        fusion_in = self.global_encoder.out_dim + entity_hidden
        self.input_proj = nn.Sequential(
            nn.Linear(fusion_in, int(cfg.trunk_hidden)),
            nn.LayerNorm(int(cfg.trunk_hidden)) if cfg.use_layer_norm else nn.Identity(),
            nn.ReLU(),
        )
        self.trunk = nn.Sequential(*[
            ResidualMLPBlock(
                int(cfg.trunk_hidden),
                dropout=float(cfg.dropout),
                use_layer_norm=bool(cfg.use_layer_norm),
            )
            for _ in range(int(cfg.trunk_layers))
        ])

        head_in = int(cfg.trunk_hidden)
        head_mid = max(64, head_in // 2)
        self._build_action_heads(head_in, head_mid)

        self._init_weights()
        if self.use_dist:
            support = torch.linspace(self.v_min, self.v_max, self.num_atoms)
            self.register_buffer("support", support)
            self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def _build_action_heads(self, head_in: int, head_mid: int):
        self.head_mid = int(head_mid)
        if self.use_factorized_action_heads:
            if self.use_dueling:
                self.val_fc = nn.Linear(head_in, head_mid)
                self.val_out = nn.Linear(head_mid, self.num_atoms)
                self.move_adv_fc = nn.Linear(head_in, head_mid)
                self.move_adv_out = nn.Linear(head_mid, NUM_MOVE * self.num_atoms)
                self.fire_adv_fc = nn.Linear(head_in, head_mid)
                self.fire_adv_out = nn.Linear(head_mid, NUM_FIRE * self.num_atoms)
            else:
                self.move_q_fc = nn.Linear(head_in, head_mid)
                self.move_q_out = nn.Linear(head_mid, NUM_MOVE * self.num_atoms)
                self.fire_q_fc = nn.Linear(head_in, head_mid)
                self.fire_q_out = nn.Linear(head_mid, NUM_FIRE * self.num_atoms)
            return

        if self.use_dueling:
            self.val_fc = nn.Linear(head_in, head_mid)
            self.val_out = nn.Linear(head_mid, self.num_atoms)
            self.adv_fc = nn.Linear(head_in, head_mid)
            self.adv_out = nn.Linear(head_mid, self.num_actions * self.num_atoms)
        else:
            self.q_fc = nn.Linear(head_in, head_mid)
            self.q_out = nn.Linear(head_mid, self.num_actions * self.num_atoms)

    def _action_head_q_atoms(self, h: torch.Tensor, B: int) -> torch.Tensor:
        if self.use_factorized_action_heads:
            if self.use_dueling:
                val = F.relu(self.val_fc(h))
                val = self.val_out(val).view(B, 1, 1, self.num_atoms)

                move_adv = F.relu(self.move_adv_fc(h))
                move_adv = self.move_adv_out(move_adv).view(B, NUM_MOVE, 1, self.num_atoms)
                fire_adv = F.relu(self.fire_adv_fc(h))
                fire_adv = self.fire_adv_out(fire_adv).view(B, 1, NUM_FIRE, self.num_atoms)

                move_adv = move_adv - move_adv.mean(dim=1, keepdim=True)
                fire_adv = fire_adv - fire_adv.mean(dim=2, keepdim=True)
                q_atoms = val + move_adv + fire_adv
            else:
                move_q = F.relu(self.move_q_fc(h))
                move_q = self.move_q_out(move_q).view(B, NUM_MOVE, 1, self.num_atoms)
                fire_q = F.relu(self.fire_q_fc(h))
                fire_q = self.fire_q_out(fire_q).view(B, 1, NUM_FIRE, self.num_atoms)
                q_atoms = move_q + fire_q
            return q_atoms.view(B, self.num_actions, self.num_atoms)

        if self.use_dueling:
            val = F.relu(self.val_fc(h))
            val = self.val_out(val).view(B, 1, self.num_atoms)
            adv = F.relu(self.adv_fc(h))
            adv = self.adv_out(adv).view(B, self.num_actions, self.num_atoms)
            return val + adv - adv.mean(dim=1, keepdim=True)

        q = F.relu(self.q_fc(h))
        return self.q_out(q).view(B, self.num_actions, self.num_atoms)

    def _frame_offsets(self, state: torch.Tensor) -> list[int]:
        total_dim = int(state.shape[1])
        stack_depth = max(1, total_dim // self.base_state_size)
        return [i * self.base_state_size for i in range(stack_depth)]

    def _build_global_features(self, state: torch.Tensor) -> torch.Tensor:
        parts = []
        for frame_off in self._frame_offsets(state):
            parts.append(state[:, frame_off: frame_off + self.core_feature_count])
            for base in self._occupancy_offsets:
                parts.append(state[:, frame_off + base: frame_off + base + 1])
        return torch.cat(parts, dim=1)

    def _build_frame_object_tokens(self, state: torch.Tensor, frame_off: int):
        """Build unified pool entity tokens with learned type embeddings.

        Wire format per slot (11 features):
            present(0), dx(1), dy(2), dist(3), vx(4), vy(5),
            threat(6), approach(7), hit_w_norm(8), hit_h_norm(9), type_id_norm(10)

        Output token (27 features):
            dx, dy, vx, vy, dist, threat, approach, hit_w, hit_h,
            type_emb(16), is_human, is_dangerous
        """
        B = state.shape[0]
        device = state.device

        # Single unified category — extract the 64×11 block
        _name, base, slots, _cat_id = self._cat_info[0]
        block = state[:, frame_off + base + 1: frame_off + base + 1 + slots * self.slot_state_features]
        block = block.reshape(B, slots, self.slot_state_features)  # (B, 64, 11)

        present = block[:, :, 0]                    # (B, 64)
        raw_feats = block[:, :, 1:10]               # (B, 64, 9) — dx,dy,dist,vx,vy,threat,approach,hit_w,hit_h
        type_id_norm = block[:, :, 10]              # (B, 64) in [0, 1]

        # Decode type_id from normalised float back to integer index
        type_id_int = (type_id_norm * (UNIFIED_NUM_TYPES - 1)).round().long().clamp(0, UNIFIED_NUM_TYPES - 1)
        type_emb = self.type_embedding(type_id_int)  # (B, 64, 16)

        # Derive boolean flags from type_id
        is_human = (type_id_int == UNIFIED_HUMAN_TYPE_ID).float().unsqueeze(-1)       # (B, 64, 1)
        is_dangerous = (type_id_int != UNIFIED_HUMAN_TYPE_ID).float().unsqueeze(-1)   # (B, 64, 1)

        # Reorder raw features to: dx, dy, vx, vy, dist, threat, approach, hit_w, hit_h
        dx = raw_feats[:, :, 0:1]           # (B, 64, 1)
        dy = raw_feats[:, :, 1:2]
        dist = raw_feats[:, :, 2:3]
        vx = raw_feats[:, :, 3:4]
        vy = raw_feats[:, :, 4:5]
        threat = raw_feats[:, :, 5:6]
        approach = raw_feats[:, :, 6:7]
        hit_w = raw_feats[:, :, 7:8]
        hit_h = raw_feats[:, :, 8:9]

        tokens = torch.cat(
            [dx, dy, vx, vy, dist, threat, approach, hit_w, hit_h, type_emb, is_human, is_dangerous],
            dim=2,
        )  # (B, 64, 27)

        mask = present < 0.5
        return tokens, mask

    def _build_object_tokens(self, state: torch.Tensor):
        frame_offsets = self._frame_offsets(state)
        return self._build_frame_object_tokens(state, frame_offsets[-1])

    def _build_directional_lane_tokens(self, state: torch.Tensor) -> torch.Tensor:
        """Extract raw slot fields from the unified pool and build 8-lane tokens.

        Returns: (lane_tokens (B, 8, 15), lane_has_entity (B, 8) bool)
        """
        B = state.shape[0]
        frame_off = self._frame_offsets(state)[-1]

        # Single unified category
        _name, base, slots, _cat_id = self._cat_info[0]
        block = state[:, frame_off + base + 1: frame_off + base + 1 + slots * self.slot_state_features]
        block = block.reshape(B, slots, self.slot_state_features)  # (B, 64, 11)

        present = block[:, :, 0] > 0.5       # (B, 64) bool
        dx = block[:, :, 1]
        dy = block[:, :, 2]
        dist = block[:, :, 3]
        vx = block[:, :, 4]
        vy = block[:, :, 5]
        threat = block[:, :, 6]
        approach = block[:, :, 7]
        type_id_norm = block[:, :, 10]

        # Decode type_id from normalised float
        type_id_int = (type_id_norm * (UNIFIED_NUM_TYPES - 1)).round().long().clamp(0, UNIFIED_NUM_TYPES - 1)
        is_human = (type_id_int == UNIFIED_HUMAN_TYPE_ID)
        is_electrode = (type_id_int == UNIFIED_ELECTRODE_TYPE_ID)

        return self.lane_encoder._build_lane_tokens(
            entity_dx=dx,
            entity_dy=dy,
            entity_dist=dist,
            entity_vx=vx,
            entity_vy=vy,
            entity_threat=threat,
            entity_approach=approach,
            entity_present=present,
            entity_is_human=is_human,
            entity_is_electrode=is_electrode,
        )

    def _build_all_frame_object_tokens(self, state: torch.Tensor):
        frame_offsets = self._frame_offsets(state)
        if not self.attn_all_frames and frame_offsets:
            frame_offsets = frame_offsets[-1:]
        token_batches = []
        mask_batches = []
        for frame_off in frame_offsets:
            tokens, mask = self._build_frame_object_tokens(state, frame_off)
            token_batches.append(tokens)
            mask_batches.append(mask)
        return torch.cat(token_batches, dim=0), torch.cat(mask_batches, dim=0)

    def forward(self, state: torch.Tensor, log: bool = False):
        B = state.shape[0]
        if self.use_pure_mlp:
            if self.use_directional_lanes:
                # Build raw entity tokens for the latest frame → (B, 64, 27)
                raw_tokens, entity_mask = self._build_frame_object_tokens(
                    state, self._frame_offsets(state)[-1]
                )
                # Project to attn_dim → (B, 64, attn_dim)
                entity_tokens = self.entity_input_proj(raw_tokens)
                # Entity self-attention: entities reason about each other
                # src_key_padding_mask=True means "ignore this position"
                # Guard: all-masked keys produce NaN in attention softmax
                any_present = (~entity_mask).any(dim=1)
                if any_present.all():
                    entity_tokens = self.entity_self_attn(
                        entity_tokens, src_key_padding_mask=entity_mask
                    )
                elif any_present.any():
                    attended = self.entity_self_attn(
                        entity_tokens[any_present],
                        src_key_padding_mask=entity_mask[any_present],
                    )
                    entity_tokens = entity_tokens.clone()
                    entity_tokens[any_present] = attended
                # else: all batches empty — skip self-attention entirely
                # Build 8-directional lane tokens from raw slot data
                lane_tokens, lane_active = self._build_directional_lane_tokens(state)
                # Cross-attend lanes to self-attended entity tokens → (B, lane_dim)
                lane_summary = self.lane_encoder(lane_tokens, entity_tokens, entity_mask, lane_active)
                # Entity pool: masked mean-pool of self-attended tokens → summary
                present_mask = ~entity_mask  # True = present
                present_count = present_mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
                entity_pooled = (entity_tokens * present_mask.unsqueeze(-1).float()).sum(dim=1) / present_count
                entity_pool_summary = self.entity_pool_norm(F.relu(self.entity_pool_proj(entity_pooled)))
                # Normalise each branch independently, then concatenate → trunk
                h = self.trunk(torch.cat([self.state_norm(state),
                                          self.lane_norm_out(lane_summary),
                                          entity_pool_summary], dim=1))
            elif self.use_mlp_with_attention:
                h = self.trunk(state)
                all_tokens, all_masks = self._build_all_frame_object_tokens(state)
                frame_summaries = self.object_attn(all_tokens, all_masks).view(B, self.attn_frame_count, -1)
                entity_out = self.entity_proj(frame_summaries.reshape(B, -1))
                h = self.mlp_attn_fusion(torch.cat([h, entity_out], dim=1))
            else:
                h = self.trunk(state)
            q_atoms = self._action_head_q_atoms(h, B)
            if self.use_dist:
                q_atoms = q_atoms.float()
                if log:
                    return F.log_softmax(q_atoms, dim=2)
                return F.softmax(q_atoms, dim=2)
            return q_atoms.squeeze(2)

        global_in = self._build_global_features(state)
        global_out = self.global_encoder(global_in)

        all_tokens, all_masks = self._build_all_frame_object_tokens(state)
        frame_summaries = self.object_attn(all_tokens, all_masks).view(B, self.attn_frame_count, -1)
        entity_out = self.entity_proj(frame_summaries.reshape(B, -1))
        h = self.input_proj(torch.cat([global_out, entity_out], dim=1))
        h = self.trunk(h)
        q_atoms = self._action_head_q_atoms(h, B)

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
        with torch.inference_mode():
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
        # Log10 level-priority: later waves get higher initial sampling weight.
        # mult = max(1.0, log10(wave) * scale)  →  wave 100 at scale=1 gives 2×,
        # at scale=5 gives 10×.  Capped again inside memory.add().
        level_scale = float(getattr(RL_CONFIG, "level_priority_log_scale", 0.0))
        level_mult = 1.0
        if level_scale > 0.0:
            wave = max(1, int(wave_number or 1))
            level_mult = max(1.0, math.log10(wave) * level_scale)
        self.memory.add(
            state,
            action_idx,
            float(reward),
            next_state,
            bool(done),
            int(horizon),
            is_expert,
            priority_hint=pri,
            level_mult=level_mult,
            wave_number=max(1, int(wave_number or 1)),
            start_wave=max(1, int(start_wave or 1)),
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

    def persist_metrics_state_only(self, filepath=LATEST_MODEL_PATH) -> bool:
        """Update only the checkpoint's metrics snapshot without resaving replay."""
        if not os.path.exists(filepath):
            return False
        try:
            ckpt = torch.load(filepath, map_location="cpu", weights_only=False)
            try:
                with metrics.lock:
                    ckpt["frame_count"] = int(metrics.frame_count)
                    ckpt["total_training_steps"] = int(metrics.total_training_steps)
                    ckpt["expert_ratio"] = float(metrics.expert_ratio)
                    ckpt["epsilon"] = float(metrics.epsilon)
            except Exception:
                pass
            ckpt["training_steps"] = int(getattr(self, "training_steps", ckpt.get("training_steps", 0)))
            ckpt["metrics_state"] = _export_metrics_snapshot()

            tmp_path = filepath + ".metrics.tmp"
            torch.save(ckpt, tmp_path)
            os.replace(tmp_path, filepath)
            return True
        except Exception:
            try:
                if os.path.exists(filepath + ".metrics.tmp"):
                    os.remove(filepath + ".metrics.tmp")
            except Exception:
                pass
            return False

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
            "metrics_state": _export_metrics_snapshot(),
            "engine_version": 2,
            "model_arch_version": MODEL_ARCH_VERSION,
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
            if int(ckpt.get("model_arch_version", 0)) != MODEL_ARCH_VERSION:
                print("⚠  Checkpoint architecture mismatch — starting fresh with structured slot encoder.")
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

            _restore_metrics_snapshot(ckpt.get("metrics_state"))

            print(f"Loaded v2 model from {filepath}")

            # Load replay buffer if present alongside the model
            buf_path = filepath.rsplit(".", 1)[0] + "_replay"
            try:
                if not self.memory.load(buf_path, verbose=bool(show_status)):
                    print("  No replay buffer found — starting with empty buffer.")
            except Exception as e:
                print(f"  Replay buffer load failed: {e}")

            try:
                with metrics.lock:
                    metrics.memory_buffer_size = len(self.memory)
            except Exception:
                pass

            return True
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            traceback.print_exc()
            return False

    def flush_replay_buffer(self):
        """Clear the entire replay buffer."""
        self.memory.flush()

    def reset_attention_weights(self):
        """Reinitialize only the object-slot attention encoder, keeping trunk and heads intact."""
        if not self.online_net.use_attn:
            print("No attention layer to reset.")
            return
        for net in (self.online_net, self.target_net):
            if hasattr(net.object_attn, "reset_parameters"):
                net.object_attn.reset_parameters()
        self._sync_inference(force=True)
        attn_param_ids = {id(p) for p in self.online_net.object_attn.parameters()}
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if id(p) in attn_param_ids and p in self.optimizer.state:
                    del self.optimizer.state[p]
        print("Object-slot attention weights and optimizer state reset (trunk + heads preserved)")

    def diagnose_attention(self, num_samples: int = 256) -> str:
        """Analyze object self-attention patterns to determine if they're meaningful."""
        if not self.online_net.use_attn:
            return "Attention is disabled in this model."
        if not hasattr(self.online_net, "object_attn"):
            return "No object self-attention found."
        if isinstance(self.online_net.object_attn, EntitySetEncoder):
            if len(self.memory) < num_samples:
                return f"Need {num_samples} samples in buffer, have {len(self.memory)}."
            batch = self.memory.sample(num_samples, beta=0.4)
            if batch is None:
                return "Could not sample from buffer."
            states = torch.from_numpy(batch[0]).float().to(self.device)
            self.online_net.eval()
            with torch.no_grad():
                obj_tokens, obj_mask = self.online_net._build_object_tokens(states)
            self.online_net.train()
            active = (~obj_mask).sum(dim=1).float()
            avg_active = float(active.mean().item()) if active.numel() else 0.0
            return (
                "Object encoder is now a per-frame transformer set encoder.\n"
                f"Latest-frame active tokens: {avg_active:.1f} / {obj_mask.shape[1]}.\n"
                "Direct pooled-attention weight diagnostics are not available for this architecture."
            )
        if len(self.memory) < num_samples:
            return f"Need {num_samples} samples in buffer, have {len(self.memory)}."

        batch = self.memory.sample(num_samples, beta=0.4)
        if batch is None:
            return "Could not sample from buffer."

        states = torch.from_numpy(batch[0]).float().to(self.device)
        self.online_net.eval()
        with torch.no_grad():
            obj_tokens, obj_mask = self.online_net._build_object_tokens(states)
            obj_encoded = self.online_net.object_attn.encode(obj_tokens)
            _, attn_w = self.online_net.object_attn(
                obj_encoded, obj_mask, return_weights=True, encoded=True
            )
        self.online_net.train()

        import numpy as np
        eps = 1e-8

        B, H, Q, T = attn_w.shape
        aw = attn_w.cpu().numpy()
        em = obj_mask.cpu().numpy()

        cat_ranges = list(getattr(self.online_net, "_category_ranges", []))
        slot_w = aw[:, :, 0, :]

        lines = []
        lines.append("\n" + "=" * 70)
        lines.append("  OBJECT SLOT-ATTENTION DIAGNOSTICS".center(70))
        lines.append("=" * 70)
        lines.append(f"  Shape: {B} samples x {H} heads x {T} source slots")

        lines.append(f"\n  Per-category slot occupancy:")
        for name, lo, hi in cat_ranges:
            occ = 1.0 - em[:, lo:hi].mean()
            bar = "#" * int(occ * 20) + "." * (20 - int(occ * 20))
            lines.append(f"    {name:12s} [{lo:3d}..{hi:3d}): {occ:.1%}  {bar}")

        avg_active = (~em).sum(axis=1).mean()
        lines.append(f"    Avg active tokens: {avg_active:.1f} / {T}")

        max_entropy = np.log(T)
        entropy = -(slot_w * np.log(slot_w + eps)).sum(axis=-1)
        mean_entropy_per_head = entropy.mean(axis=0)
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

        lines.append(f"\n  Attention mass received by each category:")
        for name_q, lo_q, hi_q in cat_ranges:
            recv = slot_w[:, :, lo_q:hi_q].sum(axis=-1).mean()
            lines.append(f"    {name_q:12s}: {recv:.4f}")

        if em.any():
            recv_attn = slot_w.mean(axis=1)
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

        head_avg = slot_w.mean(axis=0)
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
        st = torch.from_numpy(states).float().to(self.inference_device)
        q = self._infer_q_values(st)
        mn, mx = q.min().item(), q.max().item()
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
