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

import os, sys, time, struct, random, math, warnings, threading, queue, traceback
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
if torch.cuda.is_available():
    device = torch.device("cuda:0")
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

# ── Distributional Dueling Network ─────────────────────────────────────────
class RainbowNet(nn.Module):
    """
    C51 distributional network with:
      - Optional enemy-slot attention
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

        # ── Enemy attention ────────────────────────────────────────────
        self.use_attn = cfg.use_enemy_attention
        attn_out_dim = 0
        if self.use_attn:
            # Enemy decoded info: indices 91–132 = 42 features (7 slots × 6)
            # Enemy segs/depths: 133–153 = 21 features (7 × 3)
            # Total per-slot features we'll feed: 6 (decoded) + 3 (seg/depth/top) = 9
            self.slot_features = 9
            self.num_slots = cfg.enemy_slots
            self.attn = EnemyAttention(self.slot_features, cfg.attn_dim, cfg.attn_heads)
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

    def _extract_enemy_slots(self, state: torch.Tensor):
        """Pull per-enemy features from the flat state vector.
        
        Enemies are **sorted by depth** (nearest first, empty slots last)
        so that slot position carries semantic meaning rather than reflecting
        Tempest's arbitrary hardware register assignment.
        
        Returns:
          slots: (B, 7, 9) tensor — sorted by depth (nearest first)
          mask:  (B, 7) bool tensor — True where slot is EMPTY
        """
        B = state.shape[0]
        # Decoded info: 7 slots × 6 features at indices 91..132
        decoded = state[:, 91:133].reshape(B, 7, 6)       # (B, 7, 6)
        # Segments: 133..139, Depths: 140..146, Top-segs: 147..153
        segs   = state[:, 133:140].unsqueeze(2)            # (B, 7, 1)
        depths = state[:, 140:147].unsqueeze(2)            # (B, 7, 1)
        tops   = state[:, 147:154].unsqueeze(2)            # (B, 7, 1)
        slots = torch.cat([decoded, segs, depths, tops], dim=2)  # (B, 7, 9)

        # Depth feature is at index 7 in slot features (segs=6, depths=7, tops=8)
        depth_vals = slots[:, :, 7]                        # (B, 7) in [0, 1]

        # Sort by depth: active enemies (depth > 0) nearest-first,
        # empty slots (depth == 0) pushed to the end.
        # We use a sort key: empty slots get depth=2.0 (above max 1.0) so they sort last.
        empty = (depth_vals < 1e-6)                        # (B, 7) True = empty
        sort_key = torch.where(empty, torch.tensor(2.0, device=state.device), depth_vals)  # (B, 7)
        order = sort_key.argsort(dim=1)                    # (B, 7) indices

        # Gather-sort slots and empty mask
        order_expanded = order.unsqueeze(2).expand_as(slots)  # (B, 7, 9)
        slots = torch.gather(slots, 1, order_expanded)       # (B, 7, 9) sorted
        empty_mask = torch.gather(empty, 1, order)            # (B, 7) sorted

        # If ALL slots are empty (e.g. between rounds), unmask all to avoid NaN
        all_empty = empty_mask.all(dim=1, keepdim=True)    # (B, 1)
        empty_mask = empty_mask & ~all_empty               # unmask everything when all empty
        return slots, empty_mask

    def forward(self, state: torch.Tensor, log: bool = False):
        """
        Returns:
          - If distributional: (B, num_actions, num_atoms) log-probabilities or probabilities
          - If scalar: (B, num_actions) Q-values
        """
        B = state.shape[0]

        # Attention
        if self.use_attn:
            slots, empty_mask = self._extract_enemy_slots(state)
            attn_out = self.attn(slots, mask=empty_mask)   # (B, attn_dim)
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
        infer_dev = torch.device("cpu") if cfg.inference_on_cpu else self.device
        self.inference_device = infer_dev
        if self.use_separate_inference:
            self.infer_net = RainbowNet(state_size).to(infer_dev)
            self.infer_net.eval()
            self._sync_inference(force=True)
        else:
            self.infer_net = self.online_net

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
        t = (step - cfg.lr_warmup_steps) % max(1, cfg.lr_cosine_period)   # warm-restart
        cosine = 0.5 * (1.0 + math.cos(math.pi * t / max(1, cfg.lr_cosine_period)))
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
            sd = {k: v.detach().cpu() for k, v in self.online_net.state_dict().items()} \
                 if self.inference_device.type == "cpu" else self.online_net.state_dict()
            self.infer_net.load_state_dict(sd, strict=False)
            self.infer_net.eval()
            self.last_inference_sync = self.training_steps

    def act(self, state: np.ndarray, epsilon: float) -> Tuple[int, int]:
        """Return (firezap_idx, spinner_idx)."""
        if random.random() < epsilon:
            return random.randrange(NUM_FIREZAP), random.randrange(NUM_SPINNER)

        net = self.infer_net if self.use_separate_inference else self.online_net
        st = torch.from_numpy(state).float().unsqueeze(0).to(self.inference_device)
        net.eval()
        with torch.no_grad():
            q = net.q_values(st)
        joint = int(q.argmax(dim=1).item())
        return split_joint_action(joint)

    # ── Step (add experience) ───────────────────────────────────────────
    def step(self, state, action, reward, next_state, done, actor="dqn", horizon=1, priority_reward=None):
        if isinstance(action, (tuple, list)) and len(action) >= 2:
            action_idx = combine_action_indices(action[0], action[1])
        else:
            action_idx = int(max(0, min(NUM_JOINT - 1, int(action))))
        is_expert = 1 if actor == "expert" else 0
        self.memory.add(state, action_idx, float(reward), next_state, bool(done), int(horizon), is_expert)

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
    def update_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        try:
            metrics.last_target_update_step = metrics.total_training_steps
            metrics.last_target_update_time = time.time()
        except Exception:
            pass

    # ── Save / Load ─────────────────────────────────────────────────────
    def save(self, filepath, is_forced_save=False):
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
        torch.save(ckpt, filepath)
        if is_forced_save:
            print(f"Model saved to {filepath}")

    def load(self, filepath) -> bool:
        if not os.path.exists(filepath):
            return False
        try:
            ckpt = torch.load(filepath, map_location=self.device, weights_only=False)

            # Detect old engine (v1) checkpoints
            if "engine_version" not in ckpt:
                print("⚠  Old engine checkpoint detected — starting fresh with new architecture.")
                return False

            m1, u1 = self.online_net.load_state_dict(ckpt.get("online_state_dict", {}), strict=False)
            m2, u2 = self.target_net.load_state_dict(
                ckpt.get("target_state_dict", ckpt.get("online_state_dict", {})), strict=False)

            opt_sd = ckpt.get("optimizer_state_dict")
            if opt_sd:
                try:
                    self.optimizer.load_state_dict(opt_sd)
                except Exception as e:
                    print(f"Optimizer state skipped: {e}")

            self.training_steps = ckpt.get("training_steps", 0)
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
            return True
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            traceback.print_exc()
            return False

    def reset_attention_weights(self):
        """Reinitialize only the attention layer weights, keeping trunk and heads intact."""
        if not self.online_net.use_attn:
            print("No attention layer to reset.")
            return
        for net in (self.online_net, self.target_net):
            for m in net.attn.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=1.0)
                    nn.init.constant_(m.bias, 0.0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0.0)
        self._sync_inference(force=True)
        # Reset optimizer state for attention parameters so momentum doesn't carry old bias
        attn_param_ids = {id(p) for p in self.online_net.attn.parameters()}
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if id(p) in attn_param_ids and p in self.optimizer.state:
                    del self.optimizer.state[p]
        print("✓ Attention weights and optimizer state reset (trunk + heads preserved)")

    def diagnose_attention(self, num_samples: int = 256) -> str:
        """Analyze attention patterns to determine if they're meaningful."""
        if not self.online_net.use_attn:
            return "Attention is disabled in this model."
        if len(self.memory) < num_samples:
            return f"Need {num_samples} samples in buffer, have {len(self.memory)}."

        batch = self.memory.sample(num_samples, beta=0.4)
        if batch is None:
            return "Could not sample from buffer."

        states = torch.from_numpy(batch[0]).float().to(self.device)
        self.online_net.eval()
        with torch.no_grad():
            slots, empty_mask = self.online_net._extract_enemy_slots(states)  # (B, 7, 9), (B, 7)
            _, attn_w = self.online_net.attn(slots, mask=empty_mask, return_weights=True)  # (B, H, 7, 7)
        self.online_net.train()

        # attn_w: (B, num_heads, 7, 7) — each row sums to 1
        B, H, S, _ = attn_w.shape
        aw = attn_w.cpu().numpy()
        em = empty_mask.cpu().numpy()  # (B, 7) bool — True = empty

        lines = []
        lines.append("\n" + "=" * 70)
        lines.append("  ATTENTION DIAGNOSTICS".center(70))
        lines.append("=" * 70)

        # Slot occupancy stats
        occ_rate = 1.0 - em.mean(axis=0)  # (7,) fraction of time each slot is occupied
        lines.append(f"\n  Slot occupancy (fraction of samples with active enemy):")
        for s in range(S):
            bar = "█" * int(occ_rate[s] * 20) + "░" * (20 - int(occ_rate[s] * 20))
            lines.append(f"    Slot {s}: {occ_rate[s]:.1%}  {bar}")
        lines.append(f"    Avg active slots: {occ_rate.sum():.1f} / {S}")

        # 1. Entropy per head (uniform over 7 slots = ln(7) ≈ 1.946)
        import numpy as np
        max_entropy = np.log(S)  # 1.946
        eps = 1e-8
        # aw shape: (B, H, S, S) — for each query slot, distribution over keys
        log_aw = np.log(aw + eps)
        entropy = -(aw * log_aw).sum(axis=-1)  # (B, H, S)
        mean_entropy_per_head = entropy.mean(axis=(0, 2))  # (H,)
        overall_entropy = entropy.mean()

        lines.append(f"\n  Entropy (uniform = {max_entropy:.3f}, focused = 0.0):")
        for h in range(H):
            e = mean_entropy_per_head[h]
            pct = e / max_entropy * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            lines.append(f"    Head {h}: {e:.3f} ({pct:.0f}% of uniform)  {bar}")
        lines.append(f"    Overall: {overall_entropy:.3f} ({overall_entropy/max_entropy*100:.0f}% of uniform)")

        # Interpret
        ratio = overall_entropy / max_entropy
        if ratio > 0.95:
            lines.append("    → ⚠️  Near-uniform: attention is NOT selective (not yet useful)")
        elif ratio > 0.80:
            lines.append("    → 🟡 Mildly selective: some structure emerging")
        elif ratio > 0.60:
            lines.append("    → 🟢 Moderately selective: meaningful patterns forming")
        else:
            lines.append("    → 🟢 Highly selective: strong learned attention patterns")

        # 2. Which slots receive the most attention (averaged across query positions)
        # Show attention CONDITIONAL on the slot being active (occupied).
        # Raw averages conflate attention preference with occupancy rate.
        recv = aw.mean(axis=(0, 1, 2))  # (S,) — raw avg (includes zeros from empty masking)
        active_mask_all = ~em              # (B, 7) True = occupied
        occupancy = active_mask_all.mean(axis=0)  # (S,) fraction of samples where slot is active

        lines.append(f"\n  Attention per slot | given active (uniform = {1/S:.3f}):")
        for s in range(S):
            if occupancy[s] > 1e-6:
                cond_attn = recv[s] / occupancy[s]
            else:
                cond_attn = 0.0
            bar_len = int(cond_attn * S * 20)  # normalize so uniform = 20
            bar = "█" * min(bar_len, 30)
            lines.append(f"    Slot {s}: {cond_attn:.4f} ({occupancy[s]*100:4.0f}% occ)  {bar}")

        # 3. Self-attention vs cross-attention (diagonal dominance)
        diag_mean = np.mean([aw[:, :, i, i].mean() for i in range(S)])
        off_diag_mean = (aw.sum(axis=-1).sum(axis=-1).mean() - diag_mean * S) / (S * (S - 1))
        lines.append(f"\n  Self-attention (diagonal): {diag_mean:.4f}")
        lines.append(f"  Cross-attention (off-diag): {off_diag_mean:.4f}")
        if diag_mean > 1.5 / S:
            lines.append("    → Slots attend to themselves more than others (feature extraction mode)")
        else:
            lines.append("    → Slots attend to other slots (relational reasoning mode)")

        # 4. Proximity focus: does the network attend more to nearby enemies?
        #    Slots are depth-sorted (0 = nearest).  Compare MEAN per-slot
        #    attention in the nearer half vs the farther half (using means
        #    avoids bias from unequal half sizes with odd active counts).
        recv_per_sample = aw.mean(axis=(1, 2))  # (B, 7) — attention received per slot per sample
        active_mask = ~em  # (B, 7) True = occupied

        near_mean_sum = 0.0
        far_mean_sum = 0.0
        measured_samples = 0
        for b in range(B):
            active_indices = np.where(active_mask[b])[0]
            n_active = len(active_indices)
            if n_active >= 2:
                half = (n_active + 1) // 2          # ceiling division — near gets the middle slot
                near_half = active_indices[:half]    # lower-index = nearer
                far_half  = active_indices[half:]    # higher-index = farther
                near_mean_sum += recv_per_sample[b, near_half].mean()
                far_mean_sum  += recv_per_sample[b, far_half].mean()
                measured_samples += 1

        if measured_samples > 0 and far_mean_sum > 1e-8:
            near_avg = near_mean_sum / measured_samples
            far_avg  = far_mean_sum / measured_samples
            proximity_ratio = near_avg / far_avg
            lines.append(f"\n  Proximity focus (avg per-slot attention: near-half vs far-half):")
            lines.append(f"    Near-half avg: {near_avg:.4f}   Far-half avg: {far_avg:.4f}")
            lines.append(f"    Ratio: {proximity_ratio:.2f}x  (>1 = focuses on nearest)")
            if proximity_ratio > 2.0:
                lines.append("    → 🟢 Strong proximity focus: nearest enemies get most attention")
            elif proximity_ratio > 1.3:
                lines.append("    → 🟢 Moderate proximity focus")
            elif proximity_ratio > 0.8:
                lines.append("    → ⚪ Depth-independent attention (looks at all enemies equally)")
            else:
                lines.append("    → ⚠️  Attends more to DISTANT enemies (unusual)")
        else:
            lines.append(f"\n  Proximity focus: insufficient active slots to measure")

        # Empty-slot masking effectiveness
        if em.any():
            empty_recv = recv_per_sample[em].mean() if em.any() else 0
            active_recv = recv_per_sample[active_mask].mean() if active_mask.any() else 0
            lines.append(f"\n  Empty-slot masking:")
            lines.append(f"    Avg attention to active slots: {active_recv:.4f}")
            lines.append(f"    Avg attention to empty slots:  {empty_recv:.4f}")
            if empty_recv < 0.01:
                lines.append("    → 🟢 Empty slots effectively masked out")
            elif empty_recv < active_recv * 0.1:
                lines.append("    → 🟢 Minimal attention leakage to empty slots")
            else:
                lines.append("    → ⚠️  Significant attention to empty slots")

        # 5. Head specialization (do heads look at different things?)
        # Compare attention distributions between heads using KL divergence
        head_avg = aw.mean(axis=(0, 2))  # (H, S) — each head's avg attention over keys
        head_kls = []
        for i in range(H):
            for j in range(i + 1, H):
                p, q = head_avg[i] + eps, head_avg[j] + eps
                kl = (p * np.log(p / q)).sum()
                head_kls.append(kl)
        avg_kl = np.mean(head_kls) if head_kls else 0
        lines.append(f"\n  Head specialization (avg KL between heads): {avg_kl:.4f}")
        if avg_kl > 0.1:
            lines.append("    → 🟢 Heads are specialized (looking at different things)")
        elif avg_kl > 0.01:
            lines.append("    → 🟡 Mild specialization")
        else:
            lines.append("    → ⚠️  Heads are redundant (looking at same things)")

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
