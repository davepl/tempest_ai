#!/usr/bin/env python3
# ==================================================================================================================
# ||                                                                                                              ||
# ||                              TEMPEST AI • MODEL, AGENT, AND UTILITIES                                       ||
# ||                                                                                                              ||
# ||  FILE: Scripts/aimodel.py                                                                                    ||
# ||  ROLE: Neural model (HybridDQN), training agent, parsing, expert helpers, keyboard, and utilities.           ||
# ||                                                                                                              ||
# ||  NEED TO KNOW:                                                                                               ||
# ||   - HybridDQN: shared trunk + discrete head over fire/zap/spinner combinations.                              ||
# ||   - HybridDQNAgent: replay, background training, epsilon/actor logic, loss computation, target updates.      ||
# ||   - parse_frame_data: unpacks OOB header and float32 state from Lua.                                         ||
# ||   - KeyboardHandler & metrics-safe print helpers.                                                             ||
# ||                                                                                                              ||
# ||  CONSUMES: RL_CONFIG, SERVER_CONFIG, metrics                                                                 ||
# ||  PRODUCES: actions, trained weights, metrics updates                                                          ||
# ||                                                                                                              ||
# ==================================================================================================================
"""
Tempest AI Model: Hybrid expert-guided and DQN-based gameplay system.
- Makes intelligent decisions based on enemy positions and level types
- Uses a Deep Q-Network (DQN) for reinforcement learning
- Expert system provides guidance and training examples
- Communicates with Tempest via socket connection
"""

# Prevent direct execution
if __name__ == "__main__":
    print("This is not the main application, run 'main.py' instead")
    exit(1)

# Global debug flag - set to False to disable debug output
DEBUG_MODE = False

# Override the built-in print function to always flush output
# This ensures proper line breaks in output when running in background
import builtins
_original_print = builtins.print

def _flushing_print(*args, **kwargs):
    # Use CR+LF line endings consistently
    new_args = []
    for arg in args:
        if isinstance(arg, str):
            # Strip trailing whitespace and line endings
            arg = arg.rstrip()
            new_args.append(arg)
        else:
            new_args.append(arg)
    
    # Set end parameter to use CR+LF
    kwargs["end"] = "\r\n"
    kwargs['flush'] = True
    return _original_print(*new_args, **kwargs)

builtins.print = _flushing_print

import os
import time
import struct
import random
import sys
import warnings
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Deque
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import select
import threading
import queue
from collections import deque
from datetime import datetime
import socket
import traceback
# Robust import for running as script (Scripts on sys.path) and as package (repo root on sys.path)

# Platform-specific imports for KeyboardHandler
import sys
# Initialize all potential module variables to None first
msvcrt = termios = tty = fcntl = None

if sys.platform == 'win32':
    try:
        import msvcrt
    except ImportError:
        # msvcrt remains None
        print("Warning: msvcrt module not found on Windows. Keyboard input will be disabled.")
elif sys.platform in ('linux', 'darwin'):
    try:
        import termios
        import tty
        import fcntl
        import select
    except ImportError:
        # termios, tty, fcntl remain None
        print("Warning: termios, tty, or fcntl module not found. Keyboard input will be disabled.")
else:
    # All remain None
    print(f"Warning: Unsupported platform '{sys.platform}' for keyboard input.")

# Import from config.py
try:
    from config import (
        SERVER_CONFIG,
        RL_CONFIG,
        MODEL_DIR,
        LATEST_MODEL_PATH,
        metrics as config_metrics,
        ServerConfigData,
        RLConfigData,
        RESET_METRICS,
    )
    from training import train_step
except ImportError:
    from Scripts.config import (
        SERVER_CONFIG,
        RL_CONFIG,
        MODEL_DIR,
        LATEST_MODEL_PATH,
        metrics as config_metrics,
        ServerConfigData,
        RLConfigData,
        RESET_METRICS,
    )
    from Scripts.training import train_step

# Expose module under short name for compatibility with legacy imports/tests
sys.modules.setdefault('aimodel', sys.modules[__name__])

# Suppress warnings
warnings.filterwarnings('default')

# Global flag to track if running interactively
# Check this early before any potential tty interaction
IS_INTERACTIVE = sys.stdin.isatty()

# Initialize configuration
server_config = ServerConfigData()
rl_config = RLConfigData()

# Use values from config
params_count = server_config.params_count
state_size = rl_config.state_size

SPINNER_SCALE = 32.0
_RAW_SPINNER_LEVELS = tuple(int(v) for v in getattr(RL_CONFIG, "spinner_command_levels", (0,)))
if not _RAW_SPINNER_LEVELS:
    _RAW_SPINNER_LEVELS = (0,)
SPINNER_BUCKET_VALUES = tuple(level / SPINNER_SCALE for level in _RAW_SPINNER_LEVELS)
NUM_SPINNER_BUCKETS = len(SPINNER_BUCKET_VALUES)
FIRE_ZAP_ACTIONS = 4
TOTAL_DISCRETE_ACTIONS = FIRE_ZAP_ACTIONS * NUM_SPINNER_BUCKETS


def _clamp_spinner_index(index: int) -> int:
    if NUM_SPINNER_BUCKETS <= 0:
        return 0
    return int(max(0, min(NUM_SPINNER_BUCKETS - 1, index)))


def spinner_index_to_value(index: int) -> float:
    if not SPINNER_BUCKET_VALUES:
        return 0.0
    return SPINNER_BUCKET_VALUES[_clamp_spinner_index(index)]


def quantize_spinner_value(spinner_value: float) -> int:
    if not SPINNER_BUCKET_VALUES:
        return 0
    target = float(spinner_value)
    best_idx = 0
    best_dist = float("inf")
    for idx, bucket_value in enumerate(SPINNER_BUCKET_VALUES):
        dist = abs(bucket_value - target)
        if dist < best_dist:
            best_dist = dist
            best_idx = idx
    return best_idx


def compose_action_index(fire_zap_index: int, spinner_index: int) -> int:
    return int(max(0, fire_zap_index)) * NUM_SPINNER_BUCKETS + _clamp_spinner_index(spinner_index)


def decompose_action_index(action_index: int) -> tuple[int, int]:
    if NUM_SPINNER_BUCKETS <= 0:
        return int(action_index), 0
    spinner_index = action_index % NUM_SPINNER_BUCKETS
    fire_zap_index = action_index // NUM_SPINNER_BUCKETS
    fire_zap_index = int(max(0, min(FIRE_ZAP_ACTIONS - 1, fire_zap_index)))
    return fire_zap_index, _clamp_spinner_index(spinner_index)


def action_index_to_components(action_index: int) -> tuple[bool, bool, int, float]:
    fire_zap_index, spinner_index = decompose_action_index(int(action_index))
    fire, zap = discrete_to_fire_zap(fire_zap_index)
    spinner_value = spinner_index_to_value(spinner_index)
    return fire, zap, spinner_index, spinner_value


def encode_action_from_components(fire: bool, zap: bool, spinner_value: float) -> tuple[int, int, float]:
    fire_zap_index = fire_zap_to_discrete(fire, zap)
    spinner_index = quantize_spinner_value(spinner_value)
    action_index = compose_action_index(fire_zap_index, spinner_index)
    quantized_spinner = spinner_index_to_value(spinner_index)
    return action_index, spinner_index, quantized_spinner


@dataclass
class FrameData:
    """Game state data for a single frame"""
    state: np.ndarray
    subjreward: float  # Subjective reward (movement/aiming)
    objreward: float   # Objective reward (scoring)
    action: Tuple[bool, bool, float]  # fire, zap, spinner
    gamestate: int    # Added: Game state value from Lua
    done: bool
    save_signal: bool
    enemy_seg: int
    player_seg: int
    open_level: bool
    expert_fire: bool  # Added: Expert system fire recommendation
    expert_zap: bool   # Added: Expert system zap recommendation
    level_number: int  # Added: Current level number from Lua
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FrameData':
        """Create FrameData from dictionary"""
        return cls(
            state=data["state"],
            subjreward=data["subjreward"],
            objreward=data["objreward"],
            action=data["action"],
            gamestate=data["gamestate"],
            done=data["done"],
            save_signal=data["save_signal"],
            enemy_seg=data["enemy_seg"],
            player_seg=data["player_seg"],
            open_level=data["open_level"],
            expert_fire=data["expert_fire"],
            expert_zap=data["expert_zap"],
            level_number=data["level_number"],
        )

# Configuration constants
SERVER_CONFIG = server_config
RL_CONFIG = rl_config

# Initialize device (single GPU setup)
if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Low-risk math speedups (CUDA only): allow TF32 and tune matmul/cudnn
try:
    if device.type == 'cuda':
        # Enable TF32 where available for faster matmuls
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        # Prefer faster algorithms for float32 matmul in PyTorch 2+
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass
except Exception:
    pass

# ----------------------------------------------------------------------------------
# Defensive de-compile guard:
# If a previous run left torch.compile OptimizedModules resident in this process
# (e.g., via hot reload or checkpoint load) we forcibly unwrap them so that the
# current session runs in plain eager mode. This prevents lingering Inductor
# cudagraph / TLS assertion failures after source reversion.
# ----------------------------------------------------------------------------------
try:
    import types as _types
    _DYNAMO_SEEN = False
    def _unwrap_if_compiled(mod):
        """Return the underlying eager module if this is a torch.compile wrapper."""
        try:
            # PyTorch's compiled wrapper class name can vary; check common patterns
            if hasattr(mod, '_orig_mod'):
                base = getattr(mod, '_orig_mod')
                return base if isinstance(base, torch.nn.Module) else mod
            # Fallback heuristic: class name contains 'Optimized' or resides in torch._dynamo
            cname = mod.__class__.__name__
            if 'Optimized' in cname or mod.__class__.__module__.startswith('torch._dynamo'):
                return getattr(mod, '_orig_mod', mod)
        except Exception:
            return mod
        return mod
    # Expose for external use (tests / debugging)
    unwrap_compiled_module = _unwrap_if_compiled
except Exception:
    def unwrap_compiled_module(mod):  # type: ignore
        return mod

def _force_dynamo_reset_once():
    global _DYNAMO_SEEN
    if _DYNAMO_SEEN:
        return
    try:
        import torch._dynamo as _dynamo
        _dynamo.reset()
        # Disable further frame evaluation for absolute safety this run
        os.environ['TORCHDYNAMO_DISABLE'] = '1'
    except Exception:
        pass
    _DYNAMO_SEEN = True


# Display key configuration parameters
# Configuration display removed for cleaner output

# For compatibility with single-device code

# Initialize metrics
metrics = config_metrics

# Global reference to server for metrics display
metrics.global_server = None

class HybridDQN(nn.Module):
    """DQN with shared trunk feeding a single discrete head over joint fire/zap/spinner actions."""

    def __init__(self, state_size: int, discrete_actions: int = TOTAL_DISCRETE_ACTIONS,
                 hidden_size: int = 512, num_layers: int = 3):
        super(HybridDQN, self).__init__()
        
        self.state_size = state_size
        self.discrete_actions = discrete_actions
        self.num_layers = num_layers
        
        # Shared trunk for feature extraction
        LinearOrNoisy = nn.Linear  # Noisy networks removed for simplification
        
        # Dynamic layer sizing with pairs: A,A -> A/2,A/2 -> A/4,A/4 -> ...
        # Pattern: 512,512 -> 256,256 -> 128,128 -> ...
        layer_sizes = []
        current_size = hidden_size
        for i in range(num_layers):
            # Determine size for this layer pair
            pair_index = i // 2  # 0,0 -> 1,1 -> 2,2 -> ...
            layer_size = max(32, hidden_size // (2 ** pair_index))
            layer_sizes.append(layer_size)
        
        # Create shared layers dynamically
        self.shared_layers = nn.ModuleList()
        
        # First layer: state_size -> hidden_size
        self.shared_layers.append(LinearOrNoisy(state_size, layer_sizes[0]))
        
        # Subsequent layers: hidden_size -> hidden_size/2 -> hidden_size/4 -> ...
        for i in range(1, num_layers):
            self.shared_layers.append(LinearOrNoisy(layer_sizes[i-1], layer_sizes[i]))
        
        # Initialize shared trunk with xavier to ensure good gradient flow
        for layer in self.shared_layers:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight, gain=1.0)
                torch.nn.init.constant_(layer.bias, 0.0)
        
        # Final layer size for heads
        shared_output_size = layer_sizes[-1]
        head_size = max(64, shared_output_size // 2)  # Head layer size
        
        # Standard Q-network for discrete Q-values
        self.discrete_fc = nn.Linear(shared_output_size, head_size)
        self.discrete_out = nn.Linear(head_size, discrete_actions)
        
        # Initialize discrete head with balanced initialization to prevent action bias
        # CRITICAL: Without this, default initialization creates strong bias (e.g., 93% action 3)
        # Strategy: Use small random weights to allow gradient flow while maintaining initial balance
        torch.nn.init.xavier_uniform_(self.discrete_fc.weight, gain=1.0)
        torch.nn.init.constant_(self.discrete_fc.bias, 0.0)
        # Output layer: Small random initialization for gradient flow with minimal bias
        torch.nn.init.uniform_(self.discrete_out.weight, -0.003, 0.003)  # Small random weights for gradient flow
        torch.nn.init.constant_(self.discrete_out.bias, 0.0)              # Zero bias for balance
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning Q-values for every joint action combination."""
        # Shared feature extraction through dynamic layers
        shared = x
        for layer in self.shared_layers:
            shared = F.relu(layer(shared))
        
        # Discrete Q-values head
        discrete = F.relu(self.discrete_fc(shared))
        discrete_q = self.discrete_out(discrete)  # (B, total_discrete_actions)
        
        # Clamp Q-values to [-50, +50] to prevent divergence
        discrete_q = torch.clamp(discrete_q, -50.0, 50.0)
        return discrete_q


class HybridReplayBuffer:
    """Segmented replay buffer with optional priority buckets (PER-lite)."""

    _DEFAULT_BUCKET_LABELS = [
        ("p99_100", 0.995),
        ("p75_99", 0.75),
        ("p90_95", 0.90),
        ("p85_90", 0.85),
    ]
    _PRIORITY_SAMPLE_FRACTION = 0.20
    _THRESHOLD_WINDOW = 5000
    _THRESHOLD_MIN_SAMPLES = 64
    _THRESHOLD_UPDATE_INTERVAL = 128

    class _Segment:
        __slots__ = (
            "capacity",
            "state_size",
            "states",
            "next_states",
            "discrete_actions",
            "rewards",
            "dones",
            "horizons",
            "actors",
            "position",
            "size",
        )

        def __init__(self, capacity: int, state_size: int):
            self.capacity = int(max(0, capacity))
            self.state_size = int(max(1, state_size))
            self.position = 0
            self.size = 0

            if self.capacity > 0:
                self.states = np.zeros((self.capacity, self.state_size), dtype=np.float32)
                self.next_states = np.zeros((self.capacity, self.state_size), dtype=np.float32)
                self.discrete_actions = np.zeros((self.capacity,), dtype=np.int64)
                self.rewards = np.zeros((self.capacity,), dtype=np.float32)
                self.dones = np.zeros((self.capacity,), dtype=np.bool_)
                self.horizons = np.ones((self.capacity,), dtype=np.int32)
                self.actors = np.full((self.capacity,), 'dqn', dtype='U10')
            else:
                self.states = np.zeros((0, self.state_size), dtype=np.float32)
                self.next_states = np.zeros((0, self.state_size), dtype=np.float32)
                self.discrete_actions = np.zeros((0,), dtype=np.int64)
                self.rewards = np.zeros((0,), dtype=np.float32)
                self.dones = np.zeros((0,), dtype=np.bool_)
                self.horizons = np.zeros((0,), dtype=np.int32)
                self.actors = np.zeros((0,), dtype='U10')

        def add(
            self,
            state: np.ndarray,
            discrete_action: int,
            reward: float,
            next_state: np.ndarray,
            done: bool,
            actor: str,
            horizon: int,
        ):
            if self.capacity <= 0:
                return

            idx = self.position
            self.states[idx] = state
            self.next_states[idx] = next_state
            self.discrete_actions[idx] = discrete_action
            self.rewards[idx] = reward
            self.dones[idx] = done
            self.actors[idx] = actor
            self.horizons[idx] = horizon

            self.position = (idx + 1) % self.capacity
            if self.size < self.capacity:
                self.size += 1

        def gather(self, indices: np.ndarray):
            if indices.size == 0:
                zero_states = np.zeros((0, self.state_size), dtype=np.float32)
                zero_scalar = np.zeros((0, 1), dtype=np.float32)
                zero_int = np.zeros((0, 1), dtype=np.int64)
                return (
                    zero_states,
                    zero_int,
                    zero_scalar,
                    zero_states,
                    zero_scalar,
                    [],
                    zero_int.astype(np.int32),
                )

            states_np = self.states[indices].copy()
            next_states_np = self.next_states[indices].copy()
            discrete_np = self.discrete_actions[indices].copy().reshape(-1, 1)
            rewards_np = self.rewards[indices].copy().reshape(-1, 1)
            dones_np = self.dones[indices].astype(np.float32).reshape(-1, 1)
            horizons_np = self.horizons[indices].copy().reshape(-1, 1)
            actors_list = [str(a) for a in self.actors[indices]]
            return (
                states_np,
                discrete_np,
                rewards_np,
                next_states_np,
                dones_np,
                actors_list,
                horizons_np,
            )

        def clear(self):
            self.position = 0
            self.size = 0

    def __init__(self, capacity: int, state_size: int):
        self.total_capacity = int(max(1, capacity))
        self.state_size = int(max(1, state_size))

        self._lock = threading.Lock()
        self._rng = np.random.default_rng()

        configured_n = int(getattr(RL_CONFIG, 'replay_n_buckets', 0) or 0)
        configured_bucket_size = int(getattr(RL_CONFIG, 'replay_bucket_size', 0) or 0)
        configured_fraction = float(getattr(RL_CONFIG, 'priority_sample_fraction', self._PRIORITY_SAMPLE_FRACTION) or 0.0)
        configured_fraction = float(np.clip(configured_fraction, 0.0, 0.9))

        self.priority_buckets_enabled = False
        self.priority_segments: List[HybridReplayBuffer._Segment] = []
        self.bucket_labels: List[str] = []
        self.bucket_percentiles: List[float] = []
        self._priority_thresholds: List[float] = []
        self._priority_sample_fraction = configured_fraction if configured_fraction > 0 else self._PRIORITY_SAMPLE_FRACTION

        bucket_capacity = 0
        main_capacity = self.total_capacity
        active_bucket_count = 0

        if configured_n > 0 and configured_bucket_size > 0:
            tentative_bucket_cap = min(configured_bucket_size, max(1, self.total_capacity // (configured_n + 1)))
            tentative_main = self.total_capacity - (tentative_bucket_cap * configured_n)
            if tentative_bucket_cap > 0 and tentative_main > 0:
                bucket_capacity = tentative_bucket_cap
                main_capacity = tentative_main
                active_bucket_count = configured_n

        if active_bucket_count > 0:
            label_pairs = list(self._DEFAULT_BUCKET_LABELS)
            while len(label_pairs) < active_bucket_count:
                start = max(0, 100 - (len(label_pairs) + 1) * 5)
                end = min(100, start + 5)
                percentile = max(0.5, end / 100.0)
                label_pairs.append((f"p{start}_{end}", percentile))

            selected = label_pairs[:active_bucket_count]
            self.bucket_labels = [lbl for lbl, _ in selected]
            self.bucket_percentiles = [perc for _, perc in selected]
            self._priority_thresholds = [float('inf')] * active_bucket_count
            self.priority_segments = [
                self._Segment(bucket_capacity, self.state_size) for _ in range(active_bucket_count)
            ]
            self.priority_buckets_enabled = True
        else:
            self.bucket_labels = []
            self.bucket_percentiles = []
            self._priority_thresholds = []
            self.priority_segments = []
            self.priority_buckets_enabled = False

        self._main = self._Segment(main_capacity, self.state_size)
        self.capacity = self._main.capacity
        self.size = self._main.size
        self.position = self._main.position
        self.n_buckets = len(self.priority_segments)

        # Maintain direct references for compatibility with existing code paths
        self.states = self._main.states
        self.next_states = self._main.next_states
        self.discrete_actions = self._main.discrete_actions
        self.rewards = self._main.rewards
        self.dones = self._main.dones
        self.horizons = self._main.horizons
        self.actors = self._main.actors

        self._recent_scores: Deque[float] = deque(maxlen=self._THRESHOLD_WINDOW)
        self._total_additions = 0

    def _to_state_array(self, value):
        try:
            arr = np.asarray(value, dtype=np.float32)
            if arr.ndim > 1:
                arr = arr.reshape(-1)
        except Exception:
            arr = np.zeros((self.state_size,), dtype=np.float32)
        if arr.size < self.state_size:
            padded = np.zeros((self.state_size,), dtype=np.float32)
            padded[:arr.size] = arr
            return padded
        if arr.size > self.state_size:
            return arr[:self.state_size]
        return arr

    def _normalize_actor(self, actor: Optional[str]) -> str:
        if not actor:
            return 'dqn'
        actor_str = str(actor).strip().lower()
        if actor_str in ('dqn', 'expert'):
            return actor_str
        if actor_str in ('player', 'human'):
            return 'expert'
        return 'dqn'

    def push(
        self,
        state,
        discrete_action,
        reward,
        next_state,
        done,
        actor: str = 'dqn',
        horizon: int = 1,
        td_error: Optional[float] = None,
        priority_reward: Optional[float] = None,
    ):
        """Store a transition; td_error is accepted for backward compatibility."""
        _ = td_error  # Compatibility shim for previous PER variants

        state_arr = self._to_state_array(state)
        next_state_arr = self._to_state_array(next_state)

        try:
            action_idx = int(discrete_action)
        except Exception:
            action_idx = 0

        try:
            reward_val = float(reward)
        except Exception:
            reward_val = 0.0

        if priority_reward is None:
            priority_val = reward_val
        else:
            try:
                priority_val = float(priority_reward)
            except Exception:
                priority_val = reward_val

        done_flag = bool(done)
        actor_tag = self._normalize_actor(actor)

        try:
            horizon_val = int(horizon)
        except Exception:
            horizon_val = 1
        if horizon_val < 1:
            horizon_val = 1

        with self._lock:
            self._main.add(
                state_arr,
                action_idx,
                reward_val,
                next_state_arr,
                done_flag,
                actor_tag,
                horizon_val,
            )
            self.size = self._main.size
            self.position = self._main.position

            if self.priority_buckets_enabled and self.priority_segments:
                score = max(priority_val, 0.0)
                self._recent_scores.append(score)
                self._total_additions += 1

                if (
                    self._total_additions % self._THRESHOLD_UPDATE_INTERVAL == 0
                    and len(self._recent_scores) >= self._THRESHOLD_MIN_SAMPLES
                ):
                    self._update_priority_thresholds()
                elif any(math.isinf(t) for t in self._priority_thresholds) and len(self._recent_scores) >= self._THRESHOLD_MIN_SAMPLES:
                    self._update_priority_thresholds()

                bucket_idx = self._select_priority_bucket(score, actor_tag)
                if bucket_idx is not None and 0 <= bucket_idx < len(self.priority_segments):
                    self.priority_segments[bucket_idx].add(
                        state_arr,
                        action_idx,
                        reward_val,
                        next_state_arr,
                        done_flag,
                        actor_tag,
                        horizon_val,
                    )

    def sample(self, batch_size: int, return_indices: bool = False):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        with self._lock:
            if self._main.size < batch_size:
                return None

            priority_chunks = []
            remaining = batch_size

            for segment in self.priority_segments:
                if remaining <= 0:
                    priority_chunks.append(None)
                    continue

                desired = int(batch_size * self._priority_sample_fraction)
                desired = min(desired, remaining)
                desired = min(desired, segment.size)

                if desired <= 0:
                    priority_chunks.append(None)
                    continue

                idx = self._rng.choice(segment.size, size=desired, replace=False)
                priority_chunks.append(segment.gather(idx))
                remaining -= desired

            main_count = min(self._main.size, remaining)
            main_indices = self._rng.choice(self._main.size, size=main_count, replace=False)
            main_chunk = self._main.gather(main_indices)

        sample_chunks = [chunk for chunk in priority_chunks if chunk is not None]
        sample_chunks.append(main_chunk)

        states_list, discrete_list = [], []
        rewards_list, next_states_list, dones_list = [], [], []
        actors_list: List[str] = []
        horizons_list = []

        for chunk in sample_chunks:
            states_list.append(chunk[0])
            discrete_list.append(chunk[1])
            rewards_list.append(chunk[2])
            next_states_list.append(chunk[3])
            dones_list.append(chunk[4])
            actors_list.extend(chunk[5])
            horizons_list.append(chunk[6])

        states_np = np.concatenate(states_list, axis=0) if states_list else np.zeros((0, self.state_size), dtype=np.float32)
        discrete_np = np.concatenate(discrete_list, axis=0) if discrete_list else np.zeros((0, 1), dtype=np.int64)
        rewards_np = np.concatenate(rewards_list, axis=0) if rewards_list else np.zeros((0, 1), dtype=np.float32)
        next_states_np = np.concatenate(next_states_list, axis=0) if next_states_list else np.zeros((0, self.state_size), dtype=np.float32)
        dones_np = np.concatenate(dones_list, axis=0) if dones_list else np.zeros((0, 1), dtype=np.float32)
        horizons_np = np.concatenate(horizons_list, axis=0) if horizons_list else np.zeros((0, 1), dtype=np.int32)

        states = torch.from_numpy(states_np).float().to(device)
        next_states = torch.from_numpy(next_states_np).float().to(device)
        discrete_actions = torch.from_numpy(discrete_np).long().to(device)
        rewards = torch.from_numpy(rewards_np).float().to(device)
        dones = torch.from_numpy(dones_np).float().to(device)
        horizons = torch.from_numpy(horizons_np.astype(np.float32)).float().to(device)

        if return_indices:
            dummy_indices = np.full(states_np.shape[0], -1, dtype=np.int64)
            return (
                states,
                discrete_actions,
                rewards,
                next_states,
                dones,
                actors_list,
                horizons,
                dummy_indices,
            )

        return (
            states,
            discrete_actions,
            rewards,
            next_states,
            dones,
            actors_list,
            horizons,
        )

    def __len__(self):
        with self._lock:
            return self._main.size

    def clear(self):
        with self._lock:
            self._main.clear()
            for segment in self.priority_segments:
                segment.clear()
            self.size = self._main.size
            self.position = self._main.position
            self._recent_scores.clear()
            self._priority_thresholds = [float('inf')] * len(self.priority_segments)
            self._total_additions = 0

    def get_partition_stats(self):
        with self._lock:
            main_fill_pct = (self._main.size / self._main.capacity * 100.0) if self._main.capacity > 0 else 0.0
            stats = {
                'total_size': self._main.size,
                'total_capacity': self._main.capacity,
                'priority_buckets_enabled': self.priority_buckets_enabled,
                'priority_bucket_count': self.n_buckets,
                'main_size': self._main.size,
                'main_actual_size': self._main.size,
                'main_capacity': self._main.capacity,
                'main_fill_pct': main_fill_pct,
            }

            if self.priority_buckets_enabled:
                total_priority = 0
                for idx, (label, segment) in enumerate(zip(self.bucket_labels, self.priority_segments)):
                    fill_pct = (segment.size / segment.capacity * 100.0) if segment.capacity > 0 else 0.0
                    stats[f'{label}_size'] = segment.size
                    stats[f'{label}_actual_size'] = segment.size
                    stats[f'{label}_capacity'] = segment.capacity
                    stats[f'{label}_fill_pct'] = fill_pct
                    threshold_val = self._priority_thresholds[idx] if idx < len(self._priority_thresholds) else float('inf')
                    stats[f'{label}_threshold'] = threshold_val
                    total_priority += segment.size
                stats['total_priority_size'] = total_priority
                stats['bucket_labels'] = list(self.bucket_labels)
            else:
                stats['total_priority_size'] = 0
                stats['bucket_labels'] = []

            return stats

    def get_actor_composition(self):
        with self._lock:
            if self._main.size == 0:
                return {'total': 0, 'dqn': 0, 'expert': 0, 'frac_dqn': 0.0, 'frac_expert': 0.0}
            actors = self._main.actors[: self._main.size]
            n_dqn = int(np.count_nonzero(actors == 'dqn'))
            n_expert = int(np.count_nonzero(actors == 'expert'))
            total = self._main.size
        frac_dqn = float(n_dqn) / float(total) if total else 0.0
        frac_expert = float(n_expert) / float(total) if total else 0.0
        return {
            'total': total,
            'dqn': n_dqn,
            'expert': n_expert,
            'frac_dqn': frac_dqn,
            'frac_expert': frac_expert,
        }

    def _update_priority_thresholds(self):
        if not self.bucket_percentiles or not self._recent_scores:
            return

        scores = np.asarray(self._recent_scores, dtype=np.float32)
        percentiles = np.asarray(self.bucket_percentiles, dtype=np.float32)
        sorted_idx = np.argsort(percentiles)
        sorted_percentiles = np.clip(percentiles[sorted_idx], 0.0, 0.999)

        if scores.size == 1:
            quantiles = np.full_like(sorted_percentiles, scores[0], dtype=np.float32)
        else:
            quantiles = np.quantile(scores, sorted_percentiles, method='nearest')

        thresholds = np.zeros_like(percentiles, dtype=np.float32)
        thresholds[sorted_idx] = quantiles
        self._priority_thresholds = thresholds.tolist()

    def _select_priority_bucket(self, score: float, actor_tag: str) -> Optional[int]:
        if not self.priority_buckets_enabled or not self.priority_segments:
            return None

        if actor_tag == 'expert':
            return 0

        if score <= 0.0:
            return None

        fallback_idx = None
        for idx, threshold in enumerate(self._priority_thresholds):
            if math.isinf(threshold):
                fallback_idx = idx
                continue
            if score >= threshold:
                return idx

        if fallback_idx is not None:
            return fallback_idx

        if self._priority_thresholds:
            return len(self._priority_thresholds) - 1
        return None


class KeyboardHandler:
    """Cross-platform non-blocking keyboard input handler."""
    def __init__(self):
        self.platform = sys.platform
        # Store module references or None if import failed
        self.msvcrt = msvcrt 
        self.termios = termios
        self.tty = tty
        self.fcntl = fcntl
        
        self.fd = None
        self.old_settings = None

        if not IS_INTERACTIVE:
            return 

        if self.platform == 'win32':
            if self.msvcrt:
                print("KeyboardHandler: Using msvcrt for Windows.")
        elif self.platform in ('linux', 'darwin'):
            if self.termios: 
                try:
                    self.fd = sys.stdin.fileno()
                    self.old_settings = self.termios.tcgetattr(self.fd)
                    print(f"KeyboardHandler: Using termios/tty for {self.platform}.")
                except self.termios.error as e:
                    print(f"Warning: Failed to get terminal attributes: {e}. Keyboard input might be impaired.")
                    self.fd = None 
                    self.old_settings = None
            else:
                 print("Warning: Unix terminal modules failed to import. Keyboard input disabled.")
        else:
            print(f"Warning: Unsupported platform '{self.platform}' for KeyboardHandler. Keyboard input disabled.")

    def setup_terminal(self):
        """Set the terminal to raw/non-blocking (Unix-only)."""
        if self.platform in ('linux', 'darwin') and self.fd is not None and self.old_settings is not None:
            # Use instance attributes to check if modules were imported successfully
            if self.tty and self.fcntl and self.termios:
                try:
                    self.tty.setraw(self.fd)
                    flags = self.fcntl.fcntl(self.fd, self.fcntl.F_GETFL)
                    self.fcntl.fcntl(self.fd, self.fcntl.F_SETFL, flags | os.O_NONBLOCK)
                except self.termios.error as e:
                    print(f"Warning: Could not set terminal to raw/non-blocking mode: {e}")
            else:
                 print("Warning: Required Unix terminal modules not available, cannot set raw mode.")
        # No setup needed for Windows msvcrt

    def __enter__(self):
        self.setup_terminal()
        return self
        
    def __exit__(self, *args):
        self.restore_terminal()
        
    def check_key(self):
        """Check for keyboard input non-blockingly (cross-platform)."""
        if not IS_INTERACTIVE:
            return None

        try:
            if self.platform == 'win32' and self.msvcrt:
                if self.msvcrt.kbhit():
                    # Read the byte and decode assuming simple ASCII/UTF-8 key
                    key_byte = self.msvcrt.getch()
                    try:
                        return key_byte.decode('utf-8')
                    except UnicodeDecodeError:
                        return None # Or handle special keys differently
                else:
                    return None # No key waiting on Windows
            elif self.platform in ('linux', 'darwin') and self.fd is not None:
                 # Check if required modules are available via instance attributes
                 if self.termios and select: # Need termios for setup, select for check
                     try:
                         if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                             return sys.stdin.read(1)
                         else:
                             return None 
                     except Exception as e:
                        return None
                 else:
                     print("Warning: Unix termios/select modules not available for key check.")
                     return None
            else:
                return None
        except (IOError, TypeError) as e:
            return None

    def restore_terminal(self):
        """Restore original terminal settings (Unix-only)."""
        if self.platform in ('linux', 'darwin') and self.fd is not None and self.old_settings is not None:
             # Check if termios module is available via instance attribute
             if self.termios:
                try:
                    self.termios.tcsetattr(self.fd, self.termios.TCSADRAIN, self.old_settings)
                except self.termios.error as e:
                    print(f"Warning: Could not restore terminal settings: {e}")
             else:
                 print("Warning: Unix termios module not available for restore.")
        # No restore needed for Windows msvcrt

    def set_raw_mode(self):
        """Set terminal back to raw mode (Unix-only)."""
        if self.platform in ('linux', 'darwin') and self.fd is not None:
             # Check if tty/termios modules are available via instance attributes
             if self.tty and self.termios:
                try:
                    self.tty.setraw(self.fd)
                except self.termios.error as e: 
                    print(f"Warning: Could not set terminal to raw mode: {e}")
             else:
                  print("Warning: Unix tty/termios modules not available for set_raw_mode.")
        # No equivalent needed for Windows msvcrt

def print_with_terminal_restore(kb_handler, *args, **kwargs):
    """Print with proper terminal settings (cross-platform safe)."""
    # Only attempt restore/set_raw if on Unix and handler is valid
    is_unix_like = kb_handler and kb_handler.platform in ('linux', 'darwin')
    
    if IS_INTERACTIVE and is_unix_like:
        kb_handler.restore_terminal()
        
    # Standard print call - works on all platformse
    print(*args, **kwargs, flush=True)
    
    if IS_INTERACTIVE and is_unix_like:
        kb_handler.set_raw_mode()

def setup_environment():
    """Set up environment for socket server"""
    os.makedirs(MODEL_DIR, exist_ok=True)


class HybridDQNAgent:
    """Simplified hybrid DQN agent focused on fast uniform replay."""

    def __init__(
        self,
        state_size,
        discrete_actions: int = 4,
        learning_rate: float = RL_CONFIG.lr,
        gamma: float = RL_CONFIG.gamma,
        epsilon: float = RL_CONFIG.epsilon,
        epsilon_min: float = RL_CONFIG.epsilon_min,
        memory_size: int = RL_CONFIG.memory_size,
        batch_size: int = RL_CONFIG.batch_size,
    ):
        self.state_size = int(state_size)
        self.fire_zap_actions = int(discrete_actions)
        self.spinner_actions = max(1, NUM_SPINNER_BUCKETS)
        self.discrete_actions = max(1, self.fire_zap_actions * self.spinner_actions)
        self.learning_rate = float(learning_rate)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.batch_size = int(batch_size)
        self.device = device

        self.qnetwork_local = HybridDQN(
            state_size=self.state_size,
            discrete_actions=self.discrete_actions,
            hidden_size=RL_CONFIG.hidden_size,
            num_layers=RL_CONFIG.num_layers,
        ).to(self.device)

        self.qnetwork_target = HybridDQN(
            state_size=self.state_size,
            discrete_actions=self.discrete_actions,
            hidden_size=RL_CONFIG.hidden_size,
            num_layers=RL_CONFIG.num_layers,
        ).to(self.device)
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        self.qnetwork_target.eval()
        self.qnetwork_local.train()

        # Inference uses the same weights but we toggle eval/train inside act()
        self.qnetwork_inference = self.qnetwork_local

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.learning_rate)
        self.memory = HybridReplayBuffer(memory_size, state_size=self.state_size)

        self.training_enabled = True
        self.training_steps = 0
        self.last_target_update_step = 0
        self.last_save_time = 0.0

        self.train_queue = queue.Queue(maxsize=10000)
        self.running = True
        self.training_threads: list[threading.Thread] = []
        self.num_training_workers = int(getattr(RL_CONFIG, 'training_workers', 1) or 1)
        self.training_lock = threading.Lock()

        for idx in range(self.num_training_workers):
            worker = threading.Thread(
                target=self.background_train,
                daemon=True,
                name=f"TrainWorker-{idx}",
            )
            worker.start()
            self.training_threads.append(worker)

    def _sample_fire_zap_action(self) -> int:
        """Sample a fire/zap combination with optional zap discount."""
        base_actions = max(1, self.fire_zap_actions)
        zap_discount = float(getattr(RL_CONFIG, 'epsilon_random_zap_discount', 0.0) or 0.0)
        if zap_discount <= 0.0 or base_actions <= 1:
            return random.randrange(base_actions)

        zap_indices = [idx for idx in (1, 3) if idx < base_actions]
        non_zap_indices = [idx for idx in range(base_actions) if idx not in zap_indices]

        if not zap_indices or not non_zap_indices:
            return random.randrange(base_actions)

        base_prob = 1.0 / float(base_actions)
        max_discount = max(0.0, (base_prob * len(zap_indices)) - 1e-6)
        total_discount = min(zap_discount, max_discount)

        per_zap = total_discount / len(zap_indices)
        per_non = total_discount / len(non_zap_indices)

        probs = [base_prob for _ in range(base_actions)]
        for idx in zap_indices:
            probs[idx] = max(0.0, probs[idx] - per_zap)
        for idx in non_zap_indices:
            probs[idx] = max(0.0, min(1.0, probs[idx] + per_non))

        r = random.random()
        cumulative = 0.0
        for idx, prob in enumerate(probs):
            cumulative += prob
            if r < cumulative:
                return idx
        return base_actions - 1

    def _sample_random_action_index(self) -> int:
        fire_zap_idx = self._sample_fire_zap_action()
        spinner_idx = random.randrange(self.spinner_actions)
        return compose_action_index(fire_zap_idx, spinner_idx)

    def _queue_training_steps(self, n_steps: int):
        for _ in range(max(0, n_steps)):
            try:
                self.train_queue.put_nowait(1)
                metrics.training_steps_requested_interval += 1
            except queue.Full:
                try:
                    metrics.training_steps_missed_interval += 1
                    metrics.total_training_steps_missed += 1
                except Exception:
                    pass
                break

    def act(self, state, epsilon: float, add_noise: bool):
        state_arr = np.asarray(state, dtype=np.float32).reshape(-1)
        if state_arr.size < self.state_size:
            padded = np.zeros((self.state_size,), dtype=np.float32)
            padded[:state_arr.size] = state_arr
            state_arr = padded
        elif state_arr.size > self.state_size:
            state_arr = state_arr[:self.state_size]

        state_tensor = torch.from_numpy(state_arr).float().unsqueeze(0).to(self.device)

        self.qnetwork_local.eval()
        with torch.no_grad():
            q_values = self.qnetwork_local(state_tensor)
        self.qnetwork_local.train()

        if random.random() < epsilon:
            return self._sample_random_action_index()

        return int(q_values.argmax(dim=1).item())

    def step(
        self,
        state,
        discrete_action,
        reward,
        next_state,
        done,
        actor,
        horizon,
        priority_reward: Optional[float] = None,
    ):
        self.memory.push(
            state,
            discrete_action,
            reward,
            next_state,
            done,
            actor=actor,
            horizon=horizon,
            priority_reward=priority_reward,
        )

        try:
            metrics.memory_buffer_size = len(self.memory)
        except Exception:
            pass

        if not self.training_enabled:
            return

        global_training_enabled = bool(getattr(metrics, 'training_enabled', True))
        if not global_training_enabled:
            return

        steps_per_sample = int(getattr(RL_CONFIG, 'training_steps_per_sample', 1) or 1)
        self._queue_training_steps(steps_per_sample)

    def background_train(self):
        while self.running:
            try:
                token = self.train_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if token is None:
                self.train_queue.task_done()
                break

            if not self.training_enabled or not getattr(metrics, 'training_enabled', True):
                self.train_queue.task_done()
                continue

            try:
                with self.training_lock:
                    train_step(self)
            except Exception as exc:
                print(f"Training error: {exc}")
                traceback.print_exc()
            finally:
                self.train_queue.task_done()

    def train_step(self):
        return train_step(self)

    def _apply_target_update(self):
        if getattr(RL_CONFIG, 'use_soft_target_update', False):
            tau = float(getattr(RL_CONFIG, 'soft_target_tau', 0.005) or 0.005)
            with torch.no_grad():
                for tgt_param, src_param in zip(
                    self.qnetwork_target.parameters(),
                    self.qnetwork_local.parameters(),
                ):
                    tgt_param.data.mul_(1.0 - tau).add_(src_param.data, alpha=tau)
        else:
            update_freq = int(getattr(RL_CONFIG, 'target_update_freq', 500) or 500)
            if update_freq > 0 and self.training_steps % update_freq == 0:
                self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
                try:
                    metrics.last_target_update_step = self.training_steps
                    metrics.last_target_update_time = time.time()
                    metrics.last_target_update_frame = metrics.frame_count
                except Exception:
                    pass

    def update_target_network(self):
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        try:
            metrics.last_target_update_time = time.time()
            metrics.last_target_update_frame = metrics.frame_count
            metrics.last_target_update_step = self.training_steps
            metrics.last_hard_target_update_frame = metrics.frame_count
            metrics.last_hard_target_update_time = metrics.last_target_update_time
        except Exception:
            pass

    def force_hard_target_update(self):
        self.update_target_network()

    def set_training_enabled(self, enabled: bool):
        self.training_enabled = bool(enabled)
        try:
            metrics.training_enabled = bool(enabled)
        except Exception:
            pass
        if not self.training_enabled:
            try:
                while not self.train_queue.empty():
                    self.train_queue.get_nowait()
                    self.train_queue.task_done()
            except Exception:
                pass

    def stop(self, join: bool = True, timeout: float = 2.0):
        self.running = False
        try:
            self.train_queue.put_nowait(None)
        except Exception:
            pass
        if join:
            for worker in self.training_threads:
                try:
                    worker.join(timeout=timeout)
                except Exception:
                    pass

    def get_learning_rate(self) -> float:
        try:
            return float(self.optimizer.param_groups[0]['lr'])
        except Exception:
            return self.learning_rate

    def adjust_learning_rate(self, delta: float, kb_handler=None) -> float:
        current_lr = self.get_learning_rate()
        new_lr = max(1e-6, current_lr + float(delta))
        for group in self.optimizer.param_groups:
            group['lr'] = new_lr
        self.learning_rate = new_lr
        try:
            metrics.manual_lr_override = True
            metrics.manual_learning_rate = new_lr
        except Exception:
            pass
        message = f"\nLearning rate adjusted to {new_lr:.6f}\r"
        if kb_handler and IS_INTERACTIVE:
            try:
                print_with_terminal_restore(kb_handler, message)
            except Exception:
                print(message.strip())
        else:
            print(message.strip())
        return new_lr

    def save(self, filepath, now=None, is_forced_save: bool = False):
        if now is None:
            now = time.time()

        self.update_target_network()

        try:
            if metrics.expert_mode or metrics.override_expert:
                ratio_to_save = metrics.saved_expert_ratio
            else:
                ratio_to_save = metrics.expert_ratio
        except Exception:
            ratio_to_save = float(getattr(metrics, 'expert_ratio', RL_CONFIG.expert_ratio_start))

        checkpoint = {
            'local_state_dict': self.qnetwork_local.state_dict(),
            'target_state_dict': self.qnetwork_target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_steps': self.training_steps,
            'state_size': self.state_size,
            'discrete_actions': self.discrete_actions,
            'memory_size': len(self.memory),
            'architecture': 'simple_hybrid',
            'frame_count': int(getattr(metrics, 'frame_count', 0)),
            'epsilon': float(getattr(metrics, 'epsilon', RL_CONFIG.epsilon_start)),
            'expert_ratio': float(ratio_to_save),
            'last_decay_step': int(getattr(metrics, 'last_decay_step', 0)),
            'last_epsilon_decay_step': int(getattr(metrics, 'last_epsilon_decay_step', 0)),
            'episode_rewards': list(getattr(metrics, 'episode_rewards', []) or []),
            'dqn_rewards': list(getattr(metrics, 'dqn_rewards', []) or []),
            'expert_rewards': list(getattr(metrics, 'expert_rewards', []) or []),
        }
        torch.save(checkpoint, filepath)
        self.last_save_time = now
        if is_forced_save:
            print(f"Model saved to {filepath} (frame {getattr(metrics, 'frame_count', 0)})")

    def load(self, filepath):
        if not os.path.exists(filepath):
            print(f"No checkpoint found at {filepath}. Starting fresh model.")
            return False

        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.qnetwork_local.load_state_dict(checkpoint['local_state_dict'])
            self.qnetwork_target.load_state_dict(checkpoint['target_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.training_steps = int(checkpoint.get('training_steps', 0))

            try:
                metrics.frame_count = int(checkpoint.get('frame_count', 0))
                metrics.loaded_frame_count = metrics.frame_count
                metrics.epsilon = float(checkpoint.get('epsilon', RL_CONFIG.epsilon_start))
                metrics.expert_ratio = float(checkpoint.get('expert_ratio', RL_CONFIG.expert_ratio_start))
                metrics.last_decay_step = int(checkpoint.get('last_decay_step', 0))
                metrics.last_epsilon_decay_step = int(checkpoint.get('last_epsilon_decay_step', 0))
            except Exception:
                pass

            print(f"Loaded simplified hybrid model from {filepath}")
            return True
        except Exception as exc:
            print(f"Error loading checkpoint {filepath}: {exc}")
            traceback.print_exc()
            return False

    def get_q_value_range(self):
        try:
            with torch.no_grad():
                if len(self.memory) >= max(32, self.batch_size // 2):
                    sample = self.memory.sample(min(len(self.memory), 256))
                    if sample is not None:
                        states = sample[0]
                        q_values = self.qnetwork_local(states)
                        return float(q_values.min().item()), float(q_values.max().item())
                dummy = torch.zeros(1, self.state_size, device=self.device)
                q_values = self.qnetwork_local(dummy)
                return float(q_values.min().item()), float(q_values.max().item())
        except Exception:
            return float('nan'), float('nan')


Agent = HybridDQNAgent

def parse_frame_data(data: bytes) -> Optional[FrameData]:
    """Parse binary frame data from Lua into game state - SIMPLIFIED with float32 payload"""
    try:
        if not data or len(data) < 10:  # Minimal size check
            print("ERROR: Received empty or too small data packet", flush=True)
            sys.exit(1)

        # Fixed OOB header format (must match Lua exactly). Precompute once.
        # Format: ">HddBBBHIBBBhhBBBBB"  (reward and attract removed)
        global _FMT_OOB, _HDR_OOB
        try:
            _FMT_OOB
        except NameError:
            _FMT_OOB = ">HddBBBHIBBBhhBBBBB"
            _HDR_OOB = struct.calcsize(_FMT_OOB)

        if len(data) < _HDR_OOB:
            print(f"ERROR: Received data too small: {len(data)} bytes, need {_HDR_OOB}", flush=True)
            sys.exit(1)

        values = struct.unpack(_FMT_OOB, data[:_HDR_OOB])
        (num_values, subjreward, objreward, gamestate, game_mode, done, frame_counter, score,
         save_signal, fire, zap, spinner, nearest_enemy, player_seg, is_open,
         expert_fire, expert_zap, level_number) = values
        header_size = _HDR_OOB

        # Apply subjective/objective reward scaling and compute total
        subjreward = float(subjreward) * float(getattr(RL_CONFIG, 'subj_reward_scale', 1.0) or 1.0)
        objreward = float(objreward) * float(getattr(RL_CONFIG, 'obj_reward_scale', 1.0) or 1.0)
        
        if (ignore_subj_reward := getattr(RL_CONFIG, 'ignore_subjective_rewards', True)):
            reward = objreward
        else:
            reward = float(subjreward + objreward)

        state_data = memoryview(data)[header_size:]

        # SIMPLIFIED: Parse as float32 array directly (no complex normalization needed!)
        try:
            # Each float32 is 4 bytes, big-endian format
            expected_bytes = num_values * 4
            if len(state_data) < expected_bytes:
                print(f"ERROR: Expected {expected_bytes} bytes for {num_values} floats, got {len(state_data)}", flush=True)
                sys.exit(1)

            state = np.frombuffer(state_data, dtype='>f4', count=num_values)  # big-endian float32
            state = state.astype(np.float32)  # Ensure correct dtype

        except ValueError as e:
            print(f"ERROR: frombuffer failed: {e}", flush=True)
            sys.exit(1)

        if state.size != num_values:
            print(f"ERROR: Expected {num_values} state values but got {state.size}", flush=True)
            sys.exit(1)

        # CRITICAL: Check for NaN/Inf values that would cause training instability
        nan_count = np.sum(np.isnan(state))
        inf_count = np.sum(np.isinf(state))
        if nan_count > 0 or inf_count > 0:
            print(
                f"[CRITICAL] Frame {frame_counter}: {nan_count} NaN values, {inf_count} Inf values detected! "
                f"This will cause training instability!",
                flush=True,
            )

        # Ensure all values are in expected range [-1, 1] with warnings for issues
        # Allow small floating point precision errors (±0.01)
        out_of_range_count = np.sum((state < -1.01) | (state > 1.01))
        if out_of_range_count > 0:
            # Debug: identify which values are out of range
            out_of_range_indices = np.where((state < -1.01) | (state > 1.01))[0]
            out_of_range_values = state[out_of_range_indices]
            print(f"[WARNING] Frame {frame_counter}: {out_of_range_count} values outside [-1,1] range", flush=True)
            print(f"[DEBUG] Out-of-range indices: {out_of_range_indices[:10]}...")  # Show first 10
            print(f"[DEBUG] Out-of-range values: {out_of_range_values[:10]}...")  # Show first 10 values
            state = np.clip(state, -1.0, 1.0)

        frame_data = FrameData(
            state=state,
            subjreward=float(subjreward),
            objreward=float(objreward),
            action=(bool(fire), bool(zap), spinner),
            gamestate=int(gamestate),
            done=bool(done),
            save_signal=bool(save_signal),
            enemy_seg=int(nearest_enemy),
            player_seg=int(player_seg),
            open_level=bool(is_open),
            expert_fire=bool(expert_fire),
            expert_zap=bool(expert_zap),
            level_number=int(level_number),
        )
        return frame_data

    except Exception as e:
        print(f"ERROR parsing frame data: {e}", flush=True)
        sys.exit(1)

def get_expert_action(enemy_seg, player_seg, is_open_level, expert_fire=False, expert_zap=False):
    """Expert policy to move toward nearest enemy with neutral tie-breaker.
    Returns (fire, zap, spinner)
    """
    # Check for INVALID_SEGMENT (-32768) or no target (-1) which indicates no valid target (like during tube transitions)
    if enemy_seg == -32768 or enemy_seg == -1:  # INVALID_SEGMENT or no target
        return expert_fire, expert_zap, 0  # Use Lua's recommendations with no movement

    # Normalize to ring indices
    enemy_seg = int(enemy_seg) % 16
    player_seg = int(player_seg) % 16

    # Compute signed shortest distance with neutral tie-break at exactly 8
    if is_open_level:
        # Open level: direct arithmetic distance; treat exact 8 as neutral tie
        relative_dist = enemy_seg - player_seg
        if abs(relative_dist) == 8:
            relative_dist = 8 if random.random() < 0.5 else -8
    else:
        clockwise = (enemy_seg - player_seg) % 16
        counter = (player_seg - enemy_seg) % 16
        if clockwise < 8:
            relative_dist = clockwise
        elif counter < 8:
            relative_dist = -counter
        else:
            # Tie (8 vs 8)
            relative_dist = 8 if random.random() < 0.5 else -8

    if relative_dist == 0:
        return expert_fire, expert_zap, 0  # No movement needed

    # Calculate intensity based on distance
    distance = abs(relative_dist)
    intensity = min(0.9, 0.3 + (distance * 0.05))  # Match Lua intensity calculation

    # For positive relative_dist (need to move clockwise), use negative spinner
    spinner = -intensity if relative_dist > 0 else intensity

    return expert_fire, expert_zap, spinner

def encode_action_to_game(fire, zap, spinner):
    """Convert action values to game-compatible format.

    Spinner mapping:
    - Game expects an integer spinner command in [-32, +31]
    - We interpret 'spinner' here as a normalized float roughly in [-1.0, +1.0]
    - Encode by scaling with 32 and rounding, then clamp to the valid range
    """
    # Scale and round symmetrically with 32 to match original semantics
    try:
        sval = float(spinner)
    except Exception:
        sval = 0.0
    spinner_val = int(round(sval * 32.0))
    # Clamp to hardware range [-32, +31]
    if spinner_val > 31:
        spinner_val = 31
    elif spinner_val < -32:
        spinner_val = -32
    return int(fire), int(zap), int(spinner_val)

def fire_zap_to_discrete(fire: bool, zap: bool) -> int:
    """Convert fire/zap booleans to discrete action index (0-3).

    Historical encoding warning:
    - Bit 0 represents ZAP.
    - Bit 1 represents FIRE.
    This ordering is preserved to remain compatible with existing models and replay buffers.
    """
    return int(fire) * 2 + int(zap)

def discrete_to_fire_zap(discrete_action: int) -> tuple[bool, bool]:
    """Convert discrete action index (0-3) back to (fire, zap) booleans using the historical bit layout."""
    discrete_action = int(discrete_action)
    fire = (discrete_action >> 1) & 1  # fire stored in bit 1
    zap = discrete_action & 1          # zap stored in bit 0
    return bool(fire), bool(zap)

def get_expert_hybrid_action(enemy_seg, player_seg, is_open_level, expert_fire=False, expert_zap=False):
    """Return (action_index, quantized_spinner) for the expert policy."""
    fire, zap, spinner = get_expert_action(enemy_seg, player_seg, is_open_level, expert_fire, expert_zap)
    action_index, _, quantized_spinner = encode_action_from_components(fire, zap, spinner)
    return action_index, quantized_spinner

def hybrid_to_game_action(action_index: int):
    """Convert a joint action index to game-compatible fire/zap/spinner commands."""
    fire, zap, _, spinner_value = action_index_to_components(action_index)
    return encode_action_to_game(fire, zap, spinner_value)

def decay_epsilon(frame_count):
    """Calculate decayed exploration rate using step-based decay with adaptive floor.

    Goal: when stuck near the ~2.5 band, temporarily raise the epsilon floor to
    encourage exploration without destabilizing training.
    """
    # Skip decay if override or manual override is active
    if metrics.override_epsilon or getattr(metrics, 'manual_epsilon_override', False):
        return metrics.epsilon
    
    step_interval = frame_count // RL_CONFIG.epsilon_decay_steps

    # Only decay if a new step interval is reached
    if step_interval > metrics.last_epsilon_decay_step:
        # Apply decay multiplicatively for the number of steps missed
        num_steps_to_apply = step_interval - metrics.last_epsilon_decay_step
        decay_multiplier = RL_CONFIG.epsilon_decay_factor ** num_steps_to_apply
        metrics.epsilon *= decay_multiplier

        # Update the last step tracker
        metrics.last_epsilon_decay_step = step_interval


    # Enforce floor so epsilon doesn't decay below target minimum
    try:
        floor = float(getattr(RL_CONFIG, 'epsilon_end', getattr(RL_CONFIG, 'epsilon_min', 0.0)))
    except Exception:
        floor = 0.0
    if metrics.epsilon < floor:
        metrics.epsilon = floor

    # Always return the current epsilon value (which might have just been decayed)
    return metrics.epsilon

def decay_expert_ratio(current_step):
    """Update expert ratio periodically with a performance- and slope-aware floor.

    Stronger hold when near the 2.5 plateau:
    - Hold >=50% expert until BOTH (DQN5M > 2.55) AND (slope >= +0.03)
    - Then allow >=45% expert while 2.55 < DQN5M <= 2.70 and slope >= 0.00
    - Above that, revert to configured min (e.g., 0.40)
    """
    # Skip decay if expert mode, override, or manual override is active
    if metrics.expert_mode or metrics.override_expert or getattr(metrics, 'manual_expert_override', False):
        return metrics.expert_ratio
    
    # DON'T auto-initialize to start value at frame 0 - respect loaded checkpoint values
    # Only initialize if expert_ratio is somehow invalid (negative or > 1)
    if current_step == 0 and (metrics.expert_ratio < 0 or metrics.expert_ratio > 1):
        metrics.expert_ratio = RL_CONFIG.expert_ratio_start
        metrics.last_decay_step = 0
        return metrics.expert_ratio

    step_interval = current_step // RL_CONFIG.expert_ratio_decay_steps

    # Apply scheduled decay when we cross an interval boundary
    if step_interval > metrics.last_decay_step:
        steps_to_apply = step_interval - metrics.last_decay_step
        for _ in range(steps_to_apply):
            metrics.expert_ratio *= RL_CONFIG.expert_ratio_decay
        metrics.last_decay_step = step_interval

    return metrics.expert_ratio

# Discrete-only SimpleReplayBuffer removed (hybrid-only)

# Thread-safe metrics storage
class SafeMetrics:
    def __init__(self, metrics):
        self.metrics = metrics
        self.lock = threading.Lock()
    
    def update_frame_count(self, delta: int = 1):
        with self.lock:
            # Update total frame count
            if delta < 1:
                delta = 1
            self.metrics.frame_count += delta
            # Track interval frames for display rate
            try:
                self.metrics.frames_count_interval += delta
            except Exception:
                pass
            
            # Update FPS tracking
            current_time = time.time()
            
            # Initialize last_fps_time if this is the first frame
            if self.metrics.last_fps_time == 0:
                self.metrics.last_fps_time = current_time
                
            # Count frames for this second
            self.metrics.frames_last_second += delta
            
            # Calculate FPS every second
            elapsed = current_time - self.metrics.last_fps_time
            if elapsed >= 1.0:
                # Calculate frames per second with more accuracy
                new_fps = self.metrics.frames_last_second / elapsed
                
                # Store the new FPS value
                self.metrics.fps = new_fps
                
                # Reset counters
                self.metrics.frames_last_second = 0
                self.metrics.last_fps_time = current_time
                
            return self.metrics.frame_count
            
    def get_epsilon(self):
        with self.lock:
            return self.metrics.epsilon

    def get_effective_epsilon(self):
        """Thread-safe access to effective epsilon (0.0 when override_epsilon is ON)."""
        with self.lock:
            try:
                return 0.0 if getattr(self.metrics, 'override_epsilon', False) else float(self.metrics.epsilon)
            except Exception:
                return float(self.metrics.epsilon)

    def get_effective_epsilon_with_state(self, gamestate: int):
        """Return effective epsilon possibly adjusted by game state.

        Currently: if GS_ZoomingDown (0x20), return epsilon * RL_CONFIG.zoom_epsilon_scale (default 0.25).
        Honors override_epsilon by returning 0.0 regardless of state.
        """
        with self.lock:
            try:
                if getattr(self.metrics, 'override_epsilon', False):
                    return 0.0
                eps = float(self.metrics.epsilon)
                if gamestate == 0x20:
                    try:
                        scale = float(getattr(RL_CONFIG, 'zoom_epsilon_scale', 0.25) or 0.25)
                    except Exception:
                        scale = 0.25
                    eps = eps * scale
                # Bound within [0,1] as an effective runtime parameter
                if eps < 0.0:
                    eps = 0.0
                elif eps > 1.0:
                    eps = 1.0
                return eps
            except Exception:
                return float(self.metrics.epsilon)
            
    def update_epsilon(self):
        with self.lock:
            self.metrics.epsilon = decay_epsilon(self.metrics.frame_count)
            return self.metrics.epsilon
            
    def update_expert_ratio(self):
        with self.lock:
            # Respect expert_mode, override_expert, and manual_expert_override: freeze expert_ratio while any are ON
            if self.metrics.expert_mode or self.metrics.override_expert or getattr(self.metrics, 'manual_expert_override', False):
                return self.metrics.expert_ratio
            decay_expert_ratio(self.metrics.frame_count)
            return self.metrics.expert_ratio
    
    def add_episode_reward(self, total_reward, dqn_reward, expert_reward, subj_reward=None, obj_reward=None, episode_length=0):
        """Record per-episode rewards in a thread-safe way.

        Forward to the underlying MetricsData.add_episode_reward when available so that:
        - All episodes (including zero/negative totals) are recorded to keep deques aligned
        - Interval accumulators (for per-row means) are updated consistently for Rwrd/DQN/Exp
        Fallback to direct appends if the underlying metrics object lacks the method.
        """
        with self.lock:
            try:
                add_fn = getattr(self.metrics, 'add_episode_reward', None)
                if callable(add_fn):
                    add_fn(float(total_reward), float(dqn_reward), float(expert_reward), subj_reward, obj_reward, episode_length)
                    return
            except Exception:
                pass
            # Fallback: append directly without filtering to preserve alignment
            try:
                self.metrics.episode_rewards.append(float(total_reward))
            except Exception:
                pass
            try:
                self.metrics.dqn_rewards.append(float(dqn_reward))
            except Exception:
                pass
            try:
                self.metrics.expert_rewards.append(float(expert_reward))
            except Exception:
                pass
            try:
                if subj_reward is not None:
                    self.metrics.subj_rewards.append(float(subj_reward))
            except Exception:
                pass
            try:
                if obj_reward is not None:
                    self.metrics.obj_rewards.append(float(obj_reward))
            except Exception:
                pass
    
    def increment_guided_count(self):
        with self.lock:
            self.metrics.guided_count += 1
    
    def increment_total_controls(self):
        with self.lock:
            self.metrics.total_controls += 1
            
    def update_action_source(self, source):
        with self.lock:
            self.metrics.last_action_source = source
            
    def update_game_state(self, enemy_seg, open_level):
        with self.lock:
            self.metrics.enemy_seg = enemy_seg
            self.open_level = open_level
    
    def get_expert_ratio(self):
        with self.lock:
            return self.metrics.expert_ratio
            
    def is_override_active(self):
        with self.lock:
            return self.metrics.override_expert

    def get_fps(self):
        with self.lock:
            return self.metrics.fps

    # Thread-safe helpers for common aggregated metrics
    def add_inference_time(self, t: float):
        """Accumulate inference time and count in a thread-safe way."""
        try:
            dt = float(t)
        except Exception:
            dt = 0.0
        with self.lock:
            # Underlying MetricsData holds these fields
            try:
                self.metrics.total_inference_time += dt
                self.metrics.total_inference_requests += 1
            except Exception:
                # If fields are missing for any reason, initialize defensively
                try:
                    if not hasattr(self.metrics, 'total_inference_time'):
                        self.metrics.total_inference_time = 0.0
                    if not hasattr(self.metrics, 'total_inference_requests'):
                        self.metrics.total_inference_requests = 0
                    self.metrics.total_inference_time += dt
                    self.metrics.total_inference_requests += 1
                except Exception:
                    pass
