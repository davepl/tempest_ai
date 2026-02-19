#!/usr/bin/env python3
# ==================================================================================================================
# ||  TEMPEST AI v2 • MODEL, AGENT, AND UTILITIES                                                                ||
# ||                                                                                                              ||
# ||  Rainbow-lite with:                                                                                          ||
# ||    • Distributional C51 value estimation                                                                     ||
# ||    • Factored action heads (fire/zap 4 + spinner 11 = 44 total)                                             ||
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
NUM_FIREZAP = RL_CONFIG.num_firezap_actions          # 4
NUM_SPINNER = RL_CONFIG.num_spinner_actions           # 11
NUM_JOINT   = RL_CONFIG.num_joint_actions             # 44
SPINNER_LEVELS = RL_CONFIG.spinner_command_levels      # (0, 12, 9, …, -12)

def fire_zap_to_discrete(fire: bool, zap: bool) -> int:
    return int(fire) * 2 + int(zap)

def discrete_to_fire_zap(idx: int) -> Tuple[bool, bool]:
    return bool((idx >> 1) & 1), bool(idx & 1)

def spinner_index_to_value(idx: int) -> float:
    """Map spinner action index → normalised spinner command [-1, +1]."""
    idx = max(0, min(NUM_SPINNER - 1, idx))
    return SPINNER_LEVELS[idx] / 32.0

def quantize_spinner_value(value: float) -> int:
    """Find the closest spinner command index for a continuous value."""
    target = float(value) * 32.0
    best, best_d = 0, float("inf")
    for i, lv in enumerate(SPINNER_LEVELS):
        d = abs(lv - target)
        if d < best_d:
            best, best_d = i, d
    return best

def _random_firezap() -> int:
    """Sample a random fire/zap index with reduced superzap probability."""
    zap_p = float(getattr(RL_CONFIG, 'epsilon_zap_prob', 0.5))
    fire = random.random() < 0.5
    zap = random.random() < zap_p
    return fire_zap_to_discrete(fire, zap)

def combine_action_indices(fz: int, sp: int) -> int:
    return max(0, min(NUM_FIREZAP - 1, fz)) * NUM_SPINNER + max(0, min(NUM_SPINNER - 1, sp))

def split_joint_action(idx: int) -> Tuple[int, int]:
    idx = max(0, min(NUM_JOINT - 1, idx))
    return idx // NUM_SPINNER, idx % NUM_SPINNER

def encode_action_to_game(fire, zap, spinner_val):
    sv = int(round(float(spinner_val) * 32.0))
    sv = max(-32, min(31, sv))
    return int(fire), int(zap), sv

# ── Frame data ──────────────────────────────────────────────────────────────
@dataclass
class FrameData:
    state: np.ndarray
    subjreward: float
    objreward: float
    action: Tuple[bool, bool, float]
    gamestate: int
    done: bool
    save_signal: bool
    enemy_seg: int
    player_seg: int
    open_level: bool
    expert_fire: bool
    expert_zap: bool
    level_number: int

def parse_frame_data(data: bytes) -> Optional[FrameData]:
    try:
        fmt = ">HddBBBHIBBBhhBBBBB"
        hdr = struct.calcsize(fmt)
        if not data or len(data) < hdr:
            return None
        vals = struct.unpack(fmt, data[:hdr])
        (n, subj, obj, gs, mode, done, frame, score,
         save, fire, zap, spinner, enemy, player, open_lvl,
         exp_fire, exp_zap, level) = vals
        state = np.frombuffer(data[hdr:], dtype=">f4", count=n).astype(np.float32)
        return FrameData(
            state=state, subjreward=float(subj), objreward=float(obj),
            action=(bool(fire), bool(zap), spinner), gamestate=int(gs),
            done=bool(done), save_signal=bool(save),
            enemy_seg=int(enemy), player_seg=int(player),
            open_level=bool(open_lvl), expert_fire=bool(exp_fire),
            expert_zap=bool(exp_zap), level_number=int(level),
        )
    except Exception as e:
        print(f"Parse error: {e}")
        return None

# ── Expert system ───────────────────────────────────────────────────────────
def get_expert_action(enemy_seg, player_seg, is_open, expert_fire=False, expert_zap=False):
    """Returns (fire, zap, spinner_value)."""
    if enemy_seg == -32768 or enemy_seg == -1:
        return expert_fire, expert_zap, 0.0
    enemy_seg = int(enemy_seg) % 16
    player_seg = int(player_seg) % 16
    if is_open:
        rel = enemy_seg - player_seg
        if abs(rel) == 8:
            rel = 8 if random.random() < 0.5 else -8
    else:
        cw = (enemy_seg - player_seg) % 16
        ccw = (player_seg - enemy_seg) % 16
        if cw < 8:
            rel = cw
        elif ccw < 8:
            rel = -ccw
        else:
            rel = 8 if random.random() < 0.5 else -8
    if rel == 0:
        return expert_fire, expert_zap, 0.0
    intensity = min(0.9, 0.3 + abs(rel) * 0.05)
    spinner = -intensity if rel > 0 else intensity
    return expert_fire, expert_zap, spinner

# ── Enemy-Slot Self-Attention ───────────────────────────────────────────────
class EnemyAttention(nn.Module):
    """Multi-head self-attention over 7 enemy slots, producing a fixed-size summary."""

    def __init__(self, slot_features: int, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed = nn.Linear(slot_features, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.out_dim = embed_dim

    def forward(self, slots: torch.Tensor, mask: torch.Tensor = None,
                return_weights: bool = False):
        """slots: (B, num_slots, slot_features) → (B, embed_dim)
        mask: (B, num_slots) bool tensor — True = slot is EMPTY (will be ignored).
        If return_weights=True, also returns (B, num_heads, S, S) attention weights."""
        x = self.embed(slots)                   # (B, S, D)
        x = self.norm(x)
        # key_padding_mask: True positions are excluded from attention
        attn_out, attn_weights = self.attn(
            x, x, x,
            key_padding_mask=mask,
            average_attn_weights=False,
        )  # (B, S, D), (B, H, S, S)
        # Mean-pool over ACTIVE slots only
        if mask is not None:
            active = (~mask).unsqueeze(2).float()          # (B, S, 1)
            n_active = active.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1, 1)
            pooled = (attn_out * active).sum(dim=1) / n_active.squeeze(2)  # (B, D)
        else:
            pooled = attn_out.mean(dim=1)                  # (B, D)
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
      - Factored action heads (fire/zap × spinner)
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

        # ── Lane-Cross-Attention Encoder ───────────────────────────────
        self.use_attn = cfg.use_enemy_attention
        attn_out_dim = 0
        if self.use_attn:
            self.lane_features = 5    # spike, angle, player_here, sin_pos, cos_pos
            self.enemy_slot_features = 14  # 6 decoded + seg+depth+top+toprail + Δseg+Δdepth + sin+cos pos
            self.num_lanes = 16
            self.num_enemy_slots = cfg.enemy_slots  # 7

            self.lane_cross_attn = LaneCrossAttentionEncoder(
                lane_features=self.lane_features,
                enemy_features=self.enemy_slot_features,
                embed_dim=cfg.attn_dim,
                num_heads=cfg.attn_heads,
            )
            attn_out_dim = cfg.attn_dim

            # Pre-compute circular positional encoding for lanes
            lane_idx = torch.arange(16, dtype=torch.float32)
            self.register_buffer('_lane_sin_pos', torch.sin(2 * math.pi * lane_idx / 16))
            self.register_buffer('_lane_cos_pos', torch.cos(2 * math.pi * lane_idx / 16))

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

    def _build_lane_tokens(self, state: torch.Tensor):
        """Build 16 lane tokens from flat state vector.

        Each lane gets: [spike_height, tube_angle, player_here, sin_pos, cos_pos]
        Returns: (B, 16, 5)
        """
        B = state.shape[0]

        # Static per-lane features
        spike_heights = state[:, 31:47]   # (B, 16)
        tube_angles   = state[:, 47:63]   # (B, 16)

        # Player lane indicator (one-hot)
        player_pos_norm = state[:, 5]                                    # (B,) normalized pos/15
        player_lane = (player_pos_norm * 15).round().long().clamp(0, 15) # (B,)
        player_here = F.one_hot(player_lane, 16).float()                 # (B, 16)

        # Circular positional encoding (pre-computed buffers)
        lane_sin = self._lane_sin_pos.unsqueeze(0).expand(B, -1)  # (B, 16)
        lane_cos = self._lane_cos_pos.unsqueeze(0).expand(B, -1)  # (B, 16)

        # Stack: (B, 16, 5)
        tokens = torch.stack([spike_heights, tube_angles, player_here, lane_sin, lane_cos], dim=2)
        return tokens

    def _build_enemy_tokens(self, state: torch.Tensor):
        """Build 7 enemy tokens from flat state vector.

        Each enemy gets 14 features:
          [core_type, direction, between, moving_away, can_shoot, split,   (6 decoded)
           seg, depth, top_seg, toprail,                                   (4 spatial)
           Δseg, Δdepth,                                                   (2 velocity)
           sin_pos, cos_pos]                                                (2 circular pos)

        Returns:
          tokens: (B, 7, 14) — sorted by depth (nearest first, empty last)
          mask:   (B, 7) bool — True where slot is EMPTY
        """
        B = state.shape[0]
        device = state.device

        # Decoded info: 7 slots × 6 features at indices 86..127
        decoded = state[:, 86:128].reshape(B, 7, 6)          # (B, 7, 6)
        # Spatial per-slot
        segs    = state[:, 128:135].unsqueeze(2)              # (B, 7, 1)
        depths  = state[:, 135:142].unsqueeze(2)              # (B, 7, 1)
        tops    = state[:, 142:149].unsqueeze(2)              # (B, 7, 1)
        toprail = state[:, 171:178].unsqueeze(2)              # (B, 7, 1)
        # Velocity per-slot (new state indices 181-194)
        delta_seg   = state[:, 181:188].unsqueeze(2)          # (B, 7, 1)
        delta_depth = state[:, 188:195].unsqueeze(2)          # (B, 7, 1)

        # Circular positional encoding for enemy absolute position
        player_pos_norm = state[:, 5]                          # (B,)
        enemy_rel_seg   = state[:, 128:135]                    # (B, 7) normalised [-1, 1]
        # enemy_abs ≈ (player * 15 + enemy_rel * 15) mod 16 → fraction around circle
        enemy_abs_raw  = player_pos_norm.unsqueeze(1) * 15 + enemy_rel_seg * 15  # (B, 7)
        enemy_abs_frac = torch.remainder(enemy_abs_raw, 16.0) / 16.0             # (B, 7) ∈ [0, 1)
        enemy_sin = torch.sin(2 * math.pi * enemy_abs_frac).unsqueeze(2)         # (B, 7, 1)
        enemy_cos = torch.cos(2 * math.pi * enemy_abs_frac).unsqueeze(2)         # (B, 7, 1)

        # Concatenate all 14 features
        tokens = torch.cat([decoded, segs, depths, tops, toprail,
                            delta_seg, delta_depth, enemy_sin, enemy_cos], dim=2)  # (B, 7, 14)

        # Empty mask: depth ≈ 0 → inactive slot
        depth_vals = state[:, 135:142]                         # (B, 7)
        empty = (depth_vals < 1e-6)                            # (B, 7) True = empty

        # Sort by depth: nearest first, empty slots last
        sort_key = torch.where(empty, torch.tensor(2.0, device=device), depth_vals)
        order = sort_key.argsort(dim=1)
        order_exp = order.unsqueeze(2).expand_as(tokens)
        tokens = torch.gather(tokens, 1, order_exp)
        empty  = torch.gather(empty,  1, order)

        # If ALL empty (e.g. between rounds), unmask all to avoid NaN
        all_empty = empty.all(dim=1, keepdim=True)
        empty = empty & ~all_empty
        return tokens, empty

    def forward(self, state: torch.Tensor, log: bool = False):
        """
        Returns:
          - If distributional: (B, num_actions, num_atoms) log-probabilities or probabilities
          - If scalar: (B, num_actions) Q-values
        """
        B = state.shape[0]

        # Lane-Cross-Attention
        if self.use_attn:
            lane_tokens = self._build_lane_tokens(state)                    # (B, 16, 5)
            enemy_tokens, enemy_mask = self._build_enemy_tokens(state)      # (B, 7, 14), (B, 7)
            attn_out = self.lane_cross_attn(lane_tokens, enemy_tokens, enemy_mask)  # (B, D)
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

# ── Agent ───────────────────────────────────────────────────────────────────
class RainbowAgent:
    """Rainbow-lite agent with factored actions, C51, PER, n-step, attention."""

    def __init__(self, state_size: int):
        self.state_size = state_size
        self.device = device
        cfg = RL_CONFIG

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
        if self.use_separate_inference:
            self.infer_net = RainbowNet(state_size).to(infer_dev)
            self.infer_net.eval()
            self._sync_inference(force=True)
        else:
            self.infer_net = self.online_net
        print(
            f"Agent devices: train={self.device}, infer={self.inference_device}, "
            f"separate_infer={self.use_separate_inference}"
        )

        # Optimizer
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=cfg.lr, eps=1.5e-4)

        # Replay
        self.memory = PrioritizedReplayBuffer(
            capacity=cfg.memory_size,
            state_size=state_size,
            alpha=cfg.priority_alpha,
        )

        # AMP
        self.use_amp = cfg.enable_amp and (self.device.type == "cuda")
        try:
            self.grad_scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        except Exception:
            self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

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
                sd = self.online_net.state_dict()
            else:
                sd = {k: v.detach().to(self.inference_device) for k, v in self.online_net.state_dict().items()}
            self.infer_net.load_state_dict(sd, strict=False)
            self.infer_net.eval()
            self.last_inference_sync = self.training_steps

    def act(self, state: np.ndarray, epsilon: float) -> Tuple[int, int]:
        """Return (firezap_idx, spinner_idx)."""
        if random.random() < epsilon:
            return _random_firezap(), random.randrange(NUM_SPINNER)

        st = torch.from_numpy(state).float().unsqueeze(0).to(self.inference_device)
        q = self._infer_q_values(st)
        joint = int(q.argmax(dim=1).item())
        return split_joint_action(joint)

    def _infer_q_values(self, states_t: torch.Tensor) -> torch.Tensor:
        net = self.infer_net if self.use_separate_inference else self.online_net
        net.eval()
        with torch.no_grad():
            if self.use_separate_inference:
                with self._sync_lock:
                    return net.q_values(states_t)
            return net.q_values(states_t)

    def act_batch(self, states: list[np.ndarray], epsilons: list[float]) -> list[Tuple[int, int]]:
        """Return batched actions for aligned state/epsilon lists."""
        n = min(len(states), len(epsilons))
        if n <= 0:
            return []

        actions: list[Tuple[int, int] | None] = [None] * n
        greedy_idx: list[int] = []
        greedy_states: list[np.ndarray] = []

        for i in range(n):
            eps = float(epsilons[i])
            if random.random() < eps:
                actions[i] = (_random_firezap(), random.randrange(NUM_SPINNER))
            else:
                greedy_idx.append(i)
                greedy_states.append(states[i])

        if greedy_idx:
            batch_np = np.asarray(greedy_states, dtype=np.float32)
            st = torch.from_numpy(batch_np).to(self.inference_device)
            q = self._infer_q_values(st)
            joints = q.argmax(dim=1).detach().cpu().tolist()
            for pos, joint in zip(greedy_idx, joints):
                actions[pos] = split_joint_action(int(joint))

        return [a if a is not None else (0, 0) for a in actions]

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

        # Save replay buffer alongside the model
        buf_path = filepath.rsplit(".", 1)[0] + "_replay.npz"
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
                    if not RESET_METRICS:
                        metrics.expert_ratio = ckpt.get("expert_ratio", RL_CONFIG.expert_ratio_start)
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
            buf_path = filepath.rsplit(".", 1)[0] + "_replay.npz"
            try:
                if os.path.exists(buf_path):
                    self.memory.load(buf_path, verbose=bool(show_status))
                else:
                    print("  No replay buffer file found — starting with empty buffer.")
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
        """Reinitialize only the lane-cross-attention weights, keeping trunk and heads intact."""
        if not self.online_net.use_attn:
            print("No attention layer to reset.")
            return
        for net in (self.online_net, self.target_net):
            for m in net.lane_cross_attn.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=1.0)
                    nn.init.constant_(m.bias, 0.0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0.0)
        self._sync_inference(force=True)
        # Reset optimizer state for attention parameters so momentum doesn't carry old bias
        attn_param_ids = {id(p) for p in self.online_net.lane_cross_attn.parameters()}
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if id(p) in attn_param_ids and p in self.optimizer.state:
                    del self.optimizer.state[p]
        print("✓ Lane-cross-attention weights and optimizer state reset (trunk + heads preserved)")

    def diagnose_attention(self, num_samples: int = 256) -> str:
        """Analyze lane-cross-attention patterns to determine if they're meaningful."""
        if not self.online_net.use_attn:
            return "Attention is disabled in this model."
        if not hasattr(self.online_net, 'lane_cross_attn'):
            return "No lane-cross-attention found (old architecture?)."
        if len(self.memory) < num_samples:
            return f"Need {num_samples} samples in buffer, have {len(self.memory)}."

        batch = self.memory.sample(num_samples, beta=0.4)
        if batch is None:
            return "Could not sample from buffer."

        states = torch.from_numpy(batch[0]).float().to(self.device)
        self.online_net.eval()
        with torch.no_grad():
            lane_tokens = self.online_net._build_lane_tokens(states)
            enemy_tokens, enemy_mask = self.online_net._build_enemy_tokens(states)
            _, attn_w = self.online_net.lane_cross_attn(
                lane_tokens, enemy_tokens, enemy_mask, return_weights=True
            )  # (B, H, 16, 7)
        self.online_net.train()

        # attn_w: (B, num_heads, 16_lanes, 7_enemies)
        B, H, L, S = attn_w.shape
        aw = attn_w.cpu().numpy()
        em = enemy_mask.cpu().numpy()  # (B, 7) bool — True = empty

        import numpy as np
        eps = 1e-8
        lines = []
        lines.append("\n" + "=" * 70)
        lines.append("  LANE-CROSS-ATTENTION DIAGNOSTICS".center(70))
        lines.append("=" * 70)
        lines.append(f"  Shape: {B} samples × {H} heads × {L} lanes → {S} enemy slots")

        # Enemy slot occupancy
        occ_rate = 1.0 - em.mean(axis=0)  # (7,)
        lines.append(f"\n  Enemy slot occupancy:")
        for s in range(S):
            bar = "█" * int(occ_rate[s] * 20) + "░" * (20 - int(occ_rate[s] * 20))
            lines.append(f"    Slot {s}: {occ_rate[s]:.1%}  {bar}")
        lines.append(f"    Avg active: {occ_rate.sum():.1f} / {S}")

        # 1. Entropy per head (max = ln(7) ≈ 1.946 for 7 enemy keys)
        max_entropy = np.log(S)
        entropy = -(aw * np.log(aw + eps)).sum(axis=-1)  # (B, H, L)
        mean_entropy_per_head = entropy.mean(axis=(0, 2))  # (H,)
        overall_entropy = entropy.mean()
        ratio = overall_entropy / max_entropy

        lines.append(f"\n  Entropy per head (uniform = {max_entropy:.3f}):")
        for h in range(H):
            e = mean_entropy_per_head[h]
            pct = e / max_entropy * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            lines.append(f"    Head {h}: {e:.3f} ({pct:.0f}% uniform)  {bar}")
        lines.append(f"    Overall: {overall_entropy:.3f} ({ratio*100:.0f}% uniform)")

        if ratio > 0.95:
            lines.append("    → ⚠️  Near-uniform: not yet selective")
        elif ratio > 0.80:
            lines.append("    → 🟡 Mildly selective: some structure emerging")
        elif ratio > 0.60:
            lines.append("    → 🟢 Moderately selective: meaningful patterns forming")
        else:
            lines.append("    → 🟢 Highly selective: strong learned patterns")

        # 2. Player-lane focus: does the player's lane attend more strongly?
        player_pos = states[:, 5].cpu().numpy()  # (B,) normalized pos/15
        player_lanes = np.round(player_pos * 15).astype(int).clip(0, 15)
        # Avg total attention weight from player's lane vs other lanes
        player_attn_vals = []
        other_attn_vals = []
        for b in range(B):
            pl = player_lanes[b]
            player_attn_vals.append(aw[b, :, pl, :].sum(axis=-1).mean())
            mask_ = np.ones(L, dtype=bool)
            mask_[pl] = False
            other_attn_vals.append(aw[b, :, mask_, :].sum(axis=-1).mean())
        player_mean = np.mean(player_attn_vals)
        other_mean = np.mean(other_attn_vals)
        lines.append(f"\n  Player-lane focus:")
        lines.append(f"    Player's lane avg attention sum: {player_mean:.4f}")
        lines.append(f"    Other lanes avg attention sum:   {other_mean:.4f}")
        if player_mean > other_mean * 1.3:
            lines.append("    → 🟢 Player's lane attends more strongly to enemies")
        else:
            lines.append("    → ⚪ Uniform across lanes (spatial differentiation not yet learned)")

        # 3. Spatial coherence: do lanes attend more to nearby enemies?
        #    For each (lane, enemy) pair, compute angular distance on the tube circle.
        #    Check if attention weight correlates with proximity.
        enemy_segs = states[:, 128:135].cpu().numpy()  # (B, 7) normalised rel segs
        spatial_close_attn = []
        spatial_far_attn = []
        for b in range(B):
            for l in range(L):
                for s_idx in range(S):
                    if em[b, s_idx]:
                        continue
                    # Enemy lane on circle
                    e_abs = (player_lanes[b] + enemy_segs[b, s_idx] * 15) % 16
                    dist = min(abs(l - e_abs), 16 - abs(l - e_abs))
                    w = aw[b, :, l, s_idx].mean()
                    if dist <= 2:
                        spatial_close_attn.append(w)
                    else:
                        spatial_far_attn.append(w)
        if spatial_close_attn and spatial_far_attn:
            close_mean = np.mean(spatial_close_attn)
            far_mean = np.mean(spatial_far_attn)
            lines.append(f"\n  Spatial coherence (lane↔enemy proximity):")
            lines.append(f"    Nearby (≤2 lanes) avg attn: {close_mean:.4f}")
            lines.append(f"    Distant (>2 lanes) avg attn: {far_mean:.4f}")
            if close_mean > far_mean * 1.5:
                lines.append("    → 🟢 Strong spatial coherence: lanes attend to nearby enemies")
            elif close_mean > far_mean * 1.1:
                lines.append("    → 🟡 Mild spatial coherence")
            else:
                lines.append("    → ⚪ No spatial preference yet")

        # 4. Empty-slot masking
        lane_avg_attn = aw.mean(axis=(1, 2))  # (B, 7) avg over heads and lanes
        active_mask_all = ~em
        if em.any():
            empty_recv = lane_avg_attn[em].mean()
            active_recv = lane_avg_attn[active_mask_all].mean() if active_mask_all.any() else 0
            lines.append(f"\n  Empty-slot masking:")
            lines.append(f"    Avg attention to active enemies: {active_recv:.4f}")
            lines.append(f"    Avg attention to empty enemies:  {empty_recv:.4f}")
            if empty_recv < 0.01:
                lines.append("    → 🟢 Empty slots effectively masked")
            elif empty_recv < active_recv * 0.1:
                lines.append("    → 🟢 Minimal attention leakage")
            else:
                lines.append("    → ⚠️  Significant attention to empty slots")

        # 5. Head specialization
        head_avg = aw.mean(axis=(0, 2))  # (H, S)
        head_kls = []
        for i in range(H):
            for j in range(i + 1, H):
                p, q = head_avg[i] + eps, head_avg[j] + eps
                kl = (p * np.log(p / q)).sum()
                head_kls.append(kl)
        avg_kl = np.mean(head_kls) if head_kls else 0
        lines.append(f"\n  Head specialization (avg KL between heads): {avg_kl:.4f}")
        if avg_kl > 0.1:
            lines.append("    → 🟢 Heads are specialized")
        elif avg_kl > 0.01:
            lines.append("    → 🟡 Mild specialization")
        else:
            lines.append("    → ⚠️  Heads are redundant")

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
