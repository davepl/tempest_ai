#!/usr/bin/env python3
# ==================================================================================================================
# ||                                                                                                              ||
# ||                              TEMPEST AI • MODEL, AGENT, AND UTILITIES                                       ||
# ||                                                                                                              ||
# ||  FILE: Scripts/aimodel.py                                                                                    ||
# ||  ROLE: Neural model (DiscreteDQN), training agent, parsing, expert helpers, keyboard, and utilities.         ||
# ||                                                                                                              ||
# ||  NEED TO KNOW:                                                                                               ||
# ||   - DiscreteDQN: shared trunk + two discrete heads (fire/zap and spinner).                                   ||
# ||   - DiscreteDQNAgent: replay, background training, epsilon/actor logic, loss computation, target updates.    ||
# ||   - StratifiedReplayBuffer: Separate buffers for Agent and Expert to enforce sampling ratios.                ||
# ||   - parse_frame_data: unpacks OOB header and float32 state from Lua.                                         ||
# ||   - KeyboardHandler & metrics-safe print helpers.                                                             ||
# ||                                                                                                              ||
# ||  CONSUMES: RL_CONFIG, SERVER_CONFIG, metrics                                                                 ||
# ||  PRODUCES: actions, trained weights, metrics updates                                                          ||
# ||                                                                                                              ||
# ==================================================================================================================
"""
Tempest AI Model: Discrete expert-guided and DQN-based gameplay system.
- Makes intelligent decisions based on enemy positions and level types
- Uses a Deep Q-Network (DQN) with two discrete heads (FireZap + Spinner)
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
import builtins
_original_print = builtins.print

def _flushing_print(*args, **kwargs):
    new_args = []
    for arg in args:
        if isinstance(arg, str):
            arg = arg.rstrip()
            new_args.append(arg)
        else:
            new_args.append(arg)
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

class NoisyLinear(nn.Module):
    """Factorized NoisyNet layer for exploration without epsilon."""
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.std_init = float(std_init)

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        self.register_buffer("weight_epsilon", torch.zeros(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.zeros(out_features))

        self.reset_parameters()
        self.reset_noise()

    @staticmethod
    def _scaled_noise(size: int, device: torch.device) -> torch.Tensor:
        noise = torch.randn(size, device=device)
        return noise.sign().mul_(noise.abs().sqrt_())

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        sigma_init = self.std_init / math.sqrt(self.in_features)
        self.weight_sigma.data.fill_(sigma_init)
        self.bias_sigma.data.fill_(sigma_init)

    def reset_noise(self):
        device = self.weight_mu.device
        eps_in = self._scaled_noise(self.in_features, device)
        eps_out = self._scaled_noise(self.out_features, device)
        self.weight_epsilon.copy_(eps_out.ger(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

# Platform-specific imports for KeyboardHandler
import sys
msvcrt = termios = tty = fcntl = None

if sys.platform == 'win32':
    try:
        import msvcrt
    except ImportError:
        print("Warning: msvcrt module not found on Windows. Keyboard input will be disabled.")
elif sys.platform in ('linux', 'darwin'):
    try:
        import termios
        import tty
        import fcntl
        import select
    except ImportError:
        print("Warning: termios, tty, or fcntl module not found. Keyboard input will be disabled.")
else:
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

# Expose module under short name
sys.modules.setdefault('aimodel', sys.modules[__name__])

warnings.filterwarnings('default')

IS_INTERACTIVE = sys.stdin.isatty()

server_config = ServerConfigData()
rl_config = RLConfigData()

params_count = server_config.params_count
state_size = rl_config.state_size

SPINNER_SCALE = 32.0
# Force 64 buckets as per new architecture spec (-32 to +31)
NUM_SPINNER_BUCKETS = 64
SPINNER_BUCKET_VALUES = tuple((i - 32) / SPINNER_SCALE for i in range(64))
FIRE_ZAP_ACTIONS = 4
NUM_JOINT_ACTIONS = FIRE_ZAP_ACTIONS * NUM_SPINNER_BUCKETS

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

def fire_zap_to_discrete(fire: bool, zap: bool) -> int:
    """Convert fire/zap booleans to discrete action index (0-3)."""
    return int(fire) * 2 + int(zap)

def discrete_to_fire_zap(discrete_action: int) -> tuple[bool, bool]:
    """Convert discrete action index (0-3) back to (fire, zap) booleans."""
    discrete_action = int(discrete_action)
    fire = (discrete_action >> 1) & 1
    zap = discrete_action & 1
    return bool(fire), bool(zap)

def encode_action_to_game(fire, zap, spinner):
    """Convert action values to game-compatible format."""
    try:
        sval = float(spinner)
    except Exception:
        sval = 0.0
    spinner_val = int(round(sval * 32.0))
    if spinner_val > 31:
        spinner_val = 31
    elif spinner_val < -32:
        spinner_val = -32
    return int(fire), int(zap), int(spinner_val)

def combine_action_indices(firezap_idx: int, spinner_idx: int) -> int:
    """Pack (firezap, spinner) into a single joint action index."""
    fz = int(max(0, min(FIRE_ZAP_ACTIONS - 1, int(firezap_idx))))
    sp = _clamp_spinner_index(int(spinner_idx))
    return fz * NUM_SPINNER_BUCKETS + sp

def split_joint_action(action_idx: int) -> tuple[int, int]:
    """Unpack a joint action index into (firezap_idx, spinner_idx)."""
    idx = int(max(0, min(NUM_JOINT_ACTIONS - 1, int(action_idx))))
    fz = idx // NUM_SPINNER_BUCKETS
    sp = idx % NUM_SPINNER_BUCKETS
    return int(fz), int(sp)

# Backward-compatible aliases used by older diagnostics/tests.
def compose_action_index(firezap_idx: int, spinner_idx: int) -> int:
    return combine_action_indices(firezap_idx, spinner_idx)

def decompose_action_index(action_idx: int) -> tuple[int, int]:
    return split_joint_action(action_idx)

def action_index_to_components(action_idx: int) -> tuple[bool, bool, int, float]:
    fz_idx, sp_idx = split_joint_action(action_idx)
    fire, zap = discrete_to_fire_zap(fz_idx)
    sp_val = spinner_index_to_value(sp_idx)
    return fire, zap, sp_idx, sp_val

def encode_action_from_components(fire: bool, zap: bool, spinner_value: float) -> tuple[int, int, float]:
    fz_idx = fire_zap_to_discrete(fire, zap)
    sp_idx = quantize_spinner_value(float(spinner_value))
    action_idx = combine_action_indices(fz_idx, sp_idx)
    return action_idx, sp_idx, spinner_index_to_value(sp_idx)

@dataclass
class FrameData:
    """Game state data for a single frame"""
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
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FrameData':
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

SERVER_CONFIG = server_config
RL_CONFIG = rl_config

if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

try:
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass
except Exception:
    pass

metrics = config_metrics
metrics.global_server = None

class DiscreteDQN(nn.Module):
    """Joint-action Double DQN: one head over 4x64=256 actions."""

    def __init__(self, state_size: int, hidden_size: int | None = None, num_layers: int | None = None):
        super(DiscreteDQN, self).__init__()

        self.state_size = int(state_size)
        self.hidden_size = int(hidden_size or getattr(RL_CONFIG, "hidden_size", 512) or 512)
        self.num_layers = int(max(1, num_layers or getattr(RL_CONFIG, "num_layers", 3) or 3))
        self.use_noisy = bool(getattr(RL_CONFIG, "use_noisy_nets", False))
        self.noisy_std_init = float(getattr(RL_CONFIG, "noisy_std_init", 0.5) or 0.5)
        self.use_dueling = bool(getattr(RL_CONFIG, "use_dueling", True))
        self.use_layer_norm = bool(getattr(RL_CONFIG, "use_layer_norm", False))
        max_q_cfg = getattr(RL_CONFIG, "max_q_value", None)
        try:
            max_q_cfg = float(max_q_cfg) if max_q_cfg is not None else None
        except Exception:
            max_q_cfg = None
        self.max_q_value = max_q_cfg if (max_q_cfg is not None and max_q_cfg > 0.0) else None

        LinearOrNoisy = nn.Linear
        HeadLinear = NoisyLinear if self.use_noisy else nn.Linear
        head_kwargs = {"std_init": self.noisy_std_init} if self.use_noisy else {}

        # Shared trunk
        layer_sizes = []
        for i in range(self.num_layers):
            pair_index = i // 2
            layer_size = max(32, self.hidden_size // (2 ** pair_index))
            layer_sizes.append(layer_size)

        self.shared_layers = nn.ModuleList()
        self.shared_layers.append(LinearOrNoisy(self.state_size, layer_sizes[0]))
        for i in range(1, self.num_layers):
            self.shared_layers.append(LinearOrNoisy(layer_sizes[i - 1], layer_sizes[i]))

        self.shared_norms = nn.ModuleList()
        if self.use_layer_norm:
            for sz in layer_sizes:
                self.shared_norms.append(nn.LayerNorm(sz))

        for layer in self.shared_layers:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight, gain=1.0)
                torch.nn.init.constant_(layer.bias, 0.0)

        shared_output_size = layer_sizes[-1]
        head_size = max(64, shared_output_size // 2)

        if self.use_dueling:
            # Value stream
            self.value_fc = HeadLinear(shared_output_size, head_size, **head_kwargs)
            self.value_out = HeadLinear(head_size, 1, **head_kwargs)
            # Advantage stream
            self.adv_fc = HeadLinear(shared_output_size, head_size, **head_kwargs)
            self.adv_out = HeadLinear(head_size, NUM_JOINT_ACTIONS, **head_kwargs)
            init_pairs = [(self.value_fc, self.value_out), (self.adv_fc, self.adv_out)]
        else:
            self.q_fc = HeadLinear(shared_output_size, head_size, **head_kwargs)
            self.q_out = HeadLinear(head_size, NUM_JOINT_ACTIONS, **head_kwargs)
            init_pairs = [(self.q_fc, self.q_out)]

        for fc, out in init_pairs:
            if isinstance(fc, nn.Linear):
                torch.nn.init.xavier_uniform_(fc.weight, gain=1.0)
                torch.nn.init.constant_(fc.bias, 0.0)
            if isinstance(out, nn.Linear):
                torch.nn.init.uniform_(out.weight, -0.003, 0.003)
                torch.nn.init.constant_(out.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared = x
        for i, layer in enumerate(self.shared_layers):
            shared = layer(shared)
            if self.use_layer_norm and i < len(self.shared_norms):
                shared = self.shared_norms[i](shared)
            shared = F.relu(shared)

        if self.use_dueling:
            value = self.value_out(F.relu(self.value_fc(shared)))
            advantage = self.adv_out(F.relu(self.adv_fc(shared)))
            q = value + (advantage - advantage.mean(dim=1, keepdim=True))
            if self.max_q_value is not None:
                q = torch.clamp(q, -self.max_q_value, self.max_q_value)
            return q

        q = self.q_out(F.relu(self.q_fc(shared)))
        if self.max_q_value is not None:
            q = torch.clamp(q, -self.max_q_value, self.max_q_value)
        return q

    def reset_noise(self):
        if not self.use_noisy:
            return
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

class _ReplayPartition:
    """Fixed-size ring buffer for one actor partition."""

    def __init__(self, capacity: int, state_size: int):
        self.capacity = int(max(1, capacity))
        self.state_size = int(max(1, state_size))
        self.states = np.empty((self.capacity, self.state_size), dtype=np.float32)
        self.next_states = np.empty((self.capacity, self.state_size), dtype=np.float32)
        self.actions = np.empty((self.capacity,), dtype=np.int64)
        self.rewards = np.empty((self.capacity,), dtype=np.float32)
        self.dones = np.empty((self.capacity,), dtype=np.float32)
        self.horizons = np.empty((self.capacity,), dtype=np.int64)
        self.size = 0
        self.pos = 0

    def add(self, state, action_idx: int, reward: float, next_state, done: bool, horizon: int):
        i = self.pos
        self.states[i] = np.asarray(state, dtype=np.float32)
        self.actions[i] = int(action_idx)
        self.rewards[i] = float(reward)
        self.next_states[i] = np.asarray(next_state, dtype=np.float32)
        self.dones[i] = 1.0 if bool(done) else 0.0
        self.horizons[i] = int(max(1, horizon))

        self.pos = (i + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def sample(self, n: int):
        if self.size <= 0 or n <= 0:
            return None
        # With-replacement sampling is much faster and acceptable for SGD.
        idx = np.random.randint(0, self.size, size=int(n))
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
            self.horizons[idx],
        )

class StratifiedReplayBuffer:
    """Replay with separate DQN/Expert partitions and vectorized sampling."""

    def __init__(self, capacity: int, state_size: int):
        self.capacity = int(max(1, capacity))
        self.state_size = int(max(1, state_size))
        self.lock = threading.Lock()

        expert_hint = float(getattr(RL_CONFIG, "expert_ratio_start", 0.25) or 0.25)
        expert_hint = max(0.05, min(0.95, expert_hint))
        self.capacity_expert = max(1, int(round(self.capacity * expert_hint)))
        self.capacity_agent = max(1, self.capacity - self.capacity_expert)
        if self.capacity_agent + self.capacity_expert < self.capacity:
            self.capacity_agent += self.capacity - (self.capacity_agent + self.capacity_expert)
        elif self.capacity_agent + self.capacity_expert > self.capacity:
            self.capacity_agent = max(1, self.capacity - self.capacity_expert)

        self.buffer_agent = _ReplayPartition(self.capacity_agent, self.state_size)
        self.buffer_expert = _ReplayPartition(self.capacity_expert, self.state_size)
        self.total_added = 0

    def add(self, state, action_idx, reward, next_state, done, actor='dqn', horizon: int = 1):
        action = int(max(0, min(NUM_JOINT_ACTIONS - 1, int(action_idx))))
        h = int(max(1, int(horizon)))
        with self.lock:
            if actor == 'expert':
                self.buffer_expert.add(state, action, reward, next_state, done, h)
            else:
                self.buffer_agent.add(state, action, reward, next_state, done, h)
            self.total_added += 1

    def sample(self, batch_size: int, expert_ratio: float):
        with self.lock:
            bsz = int(max(1, batch_size))
            ratio = float(max(0.0, min(1.0, expert_ratio)))
            avail_expert = int(self.buffer_expert.size)
            avail_agent = int(self.buffer_agent.size)

            target_expert = int(round(bsz * ratio))
            n_expert = min(target_expert, avail_expert)
            n_agent = min(bsz - n_expert, avail_agent)

            remaining = bsz - (n_expert + n_agent)
            if remaining > 0:
                extra_agent = min(remaining, max(0, avail_agent - n_agent))
                n_agent += extra_agent
                remaining -= extra_agent
            if remaining > 0:
                extra_expert = min(remaining, max(0, avail_expert - n_expert))
                n_expert += extra_expert
                remaining -= extra_expert

            total = n_expert + n_agent
            if total <= 0:
                return None

            parts = []
            actor_flags = []
            if n_expert > 0:
                parts.append(self.buffer_expert.sample(n_expert))
                actor_flags.append(np.ones((n_expert,), dtype=np.uint8))
            if n_agent > 0:
                parts.append(self.buffer_agent.sample(n_agent))
                actor_flags.append(np.zeros((n_agent,), dtype=np.uint8))

            if len(parts) == 1:
                states, actions, rewards, next_states, dones, horizons = parts[0]
                actors = actor_flags[0]
            else:
                states = np.concatenate([p[0] for p in parts], axis=0)
                actions = np.concatenate([p[1] for p in parts], axis=0)
                rewards = np.concatenate([p[2] for p in parts], axis=0)
                next_states = np.concatenate([p[3] for p in parts], axis=0)
                dones = np.concatenate([p[4] for p in parts], axis=0)
                horizons = np.concatenate([p[5] for p in parts], axis=0)
                actors = np.concatenate(actor_flags, axis=0)

            if total > 1:
                perm = np.random.permutation(total)
                states = states[perm]
                actions = actions[perm]
                rewards = rewards[perm]
                next_states = next_states[perm]
                dones = dones[perm]
                horizons = horizons[perm]
                actors = actors[perm]

            return (
                states,
                actions,
                rewards,
                next_states,
                dones,
                horizons,
                actors,  # uint8: 1=expert, 0=dqn
            )

    def __len__(self):
        with self.lock:
            return int(self.buffer_agent.size + self.buffer_expert.size)

    def get_partition_stats(self):
        with self.lock:
            n_agent = int(self.buffer_agent.size)
            n_expert = int(self.buffer_expert.size)
            total = n_agent + n_expert
            return {
                'total_size': total,
                'total_capacity': self.capacity,
                'dqn': n_agent,
                'expert': n_expert,
                'frac_dqn': n_agent / total if total else 0,
                'frac_expert': n_expert / total if total else 0
            }

    def get_actor_composition(self):
        return self.get_partition_stats()

class KeyboardHandler:
    """Cross-platform non-blocking keyboard input handler."""
    def __init__(self):
        self.platform = sys.platform
        self.msvcrt = msvcrt 
        self.termios = termios
        self.tty = tty
        self.fcntl = fcntl
        self.fd = None
        self.old_settings = None

        if not IS_INTERACTIVE: return 

        if self.platform == 'win32' and self.msvcrt:
            pass
        elif self.platform in ('linux', 'darwin') and self.termios:
            try:
                self.fd = sys.stdin.fileno()
                self.old_settings = self.termios.tcgetattr(self.fd)
            except Exception:
                self.fd = None
        else:
            pass

    def setup_terminal(self):
        if self.platform in ('linux', 'darwin') and self.fd is not None and self.tty and self.fcntl:
            try:
                self.tty.setraw(self.fd)
                flags = self.fcntl.fcntl(self.fd, self.fcntl.F_GETFL)
                self.fcntl.fcntl(self.fd, self.fcntl.F_SETFL, flags | os.O_NONBLOCK)
            except Exception: pass

    def __enter__(self):
        self.setup_terminal()
        return self
        
    def __exit__(self, *args):
        self.restore_terminal()
        
    def check_key(self):
        if not IS_INTERACTIVE: return None
        try:
            if self.platform == 'win32' and self.msvcrt:
                if self.msvcrt.kbhit():
                    return self.msvcrt.getch().decode('utf-8')
            elif self.platform in ('linux', 'darwin') and self.fd is not None and select:
                 if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                     return sys.stdin.read(1)
        except Exception: pass
        return None

    def restore_terminal(self):
        if self.platform in ('linux', 'darwin') and self.fd is not None and self.termios:
            try:
                self.termios.tcsetattr(self.fd, self.termios.TCSADRAIN, self.old_settings)
            except Exception: pass

    def set_raw_mode(self):
        if self.platform in ('linux', 'darwin') and self.fd is not None and self.tty:
            try: self.tty.setraw(self.fd)
            except Exception: pass

def print_with_terminal_restore(kb_handler, *args, **kwargs):
    is_unix_like = kb_handler and kb_handler.platform in ('linux', 'darwin')
    if IS_INTERACTIVE and is_unix_like: kb_handler.restore_terminal()
    print(*args, **kwargs, flush=True)
    if IS_INTERACTIVE and is_unix_like: kb_handler.set_raw_mode()

def setup_environment():
    os.makedirs(MODEL_DIR, exist_ok=True)

class DiscreteDQNAgent:
    """Agent using joint-action DiscreteDQN and stratified replay."""

    def __init__(self, state_size, discrete_actions=None, learning_rate=RL_CONFIG.lr, 
                 gamma=RL_CONFIG.gamma, epsilon=RL_CONFIG.epsilon, memory_size=RL_CONFIG.memory_size, 
                 batch_size=RL_CONFIG.batch_size):
        self.state_size = int(state_size)
        self.learning_rate = float(learning_rate)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.batch_size = int(batch_size)
        self.discrete_actions = NUM_JOINT_ACTIONS
        self.device = device

        self.qnetwork_local = DiscreteDQN(
            state_size=self.state_size,
            hidden_size=getattr(RL_CONFIG, "hidden_size", 512),
            num_layers=getattr(RL_CONFIG, "num_layers", 3),
        ).to(self.device)
        self.qnetwork_target = DiscreteDQN(
            state_size=self.state_size,
            hidden_size=getattr(RL_CONFIG, "hidden_size", 512),
            num_layers=getattr(RL_CONFIG, "num_layers", 3),
        ).to(self.device)
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        self.qnetwork_target.eval()
        self.qnetwork_local.train()

        self.use_separate_inference_model = bool(getattr(RL_CONFIG, "use_separate_inference_model", True))
        infer_on_cpu = bool(getattr(RL_CONFIG, "inference_on_cpu", True))
        self.inference_device = torch.device("cpu") if infer_on_cpu else self.device
        self.inference_sync_steps = int(max(1, int(getattr(RL_CONFIG, "inference_sync_steps", 250) or 250)))
        self.last_inference_sync_step = 0
        self._sync_lock = threading.Lock()
        self.training_steps = 0
        self.train_metrics_interval_steps = int(
            max(1, int(getattr(RL_CONFIG, "metrics_update_interval_steps", 1) or 1))
        )
        self.use_amp = bool(getattr(RL_CONFIG, "enable_amp", False)) and (self.device.type == "cuda")
        try:
            self.grad_scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        except Exception:
            self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        if self.use_separate_inference_model:
            self.qnetwork_infer = DiscreteDQN(
                state_size=self.state_size,
                hidden_size=getattr(RL_CONFIG, "hidden_size", 512),
                num_layers=getattr(RL_CONFIG, "num_layers", 3),
            ).to(self.inference_device)
            self.qnetwork_infer.eval()
            self.sync_inference_model(force=True)
        else:
            self.qnetwork_infer = self.qnetwork_local

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.learning_rate)
        self.memory = StratifiedReplayBuffer(memory_size, state_size=self.state_size)

        self.training_enabled = True
        self.train_queue = queue.Queue(maxsize=32)  # Kept for shutdown signaling compatibility.
        self.running = True
        self.training_threads = []
        
        worker = threading.Thread(target=self.background_train, daemon=True, name="TrainWorker")
        worker.start()
        self.training_threads.append(worker)

    def sync_inference_model(self, force: bool = False):
        if not self.use_separate_inference_model:
            return

        if (not force) and (self.training_steps - self.last_inference_sync_step < self.inference_sync_steps):
            return

        with self._sync_lock:
            if self.inference_device.type == "cpu":
                state_dict = {k: v.detach().cpu() for k, v in self.qnetwork_local.state_dict().items()}
            else:
                state_dict = self.qnetwork_local.state_dict()
            self.qnetwork_infer.load_state_dict(state_dict, strict=False)
            self.qnetwork_infer.eval()
            self.last_inference_sync_step = int(self.training_steps)
            try:
                with metrics.lock:
                    metrics.last_inference_sync_time = time.time()
                    metrics.last_inference_sync_frame = int(getattr(metrics, "frame_count", 0))
            except Exception:
                pass

    def act(self, state, epsilon: float, add_noise: bool = False):
        """Return (fire_zap_idx, spinner_idx)."""
        if random.random() < epsilon:
            # Epsilon exploration: bias away from random zaps (they're often catastrophic noise)
            # Actions 0..3 encode (fire,zap) as (bit1,bit0). Zap actions are 1 and 3.
            try:
                zap_discount = float(getattr(RL_CONFIG, "epsilon_random_zap_discount", 1.0) or 1.0)
                if not math.isfinite(zap_discount):
                    zap_discount = 1.0
                zap_discount = max(0.0, min(1.0, zap_discount))
            except Exception:
                zap_discount = 1.0

            if zap_discount >= 1.0:
                fz = random.randrange(FIRE_ZAP_ACTIONS)
            else:
                weights = (1.0, zap_discount, 1.0, zap_discount)
                total = sum(weights)
                pick = random.random() * total
                running = 0.0
                fz = 0
                for idx, w in enumerate(weights):
                    running += w
                    if pick <= running:
                        fz = idx
                        break
            sp = random.randrange(NUM_SPINNER_BUCKETS)
            return fz, sp

        infer_model = self.qnetwork_infer if self.use_separate_inference_model else self.qnetwork_local
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.inference_device)
        infer_model.eval()
        with torch.no_grad():
            q_joint = infer_model(state_t)
        if not self.use_separate_inference_model:
            self.qnetwork_local.train()

        action_idx = int(q_joint.argmax(dim=1).item())
        return split_joint_action(action_idx)

    def step(self, state, action, reward, next_state, done, actor='dqn', horizon=1, priority_reward=None):
        # Action can be (fire_zap_idx, spinner_idx) or packed joint action.
        if isinstance(action, (tuple, list)) and len(action) >= 2:
            action_idx = combine_action_indices(action[0], action[1])
        else:
            action_idx = int(max(0, min(NUM_JOINT_ACTIONS - 1, int(action))))
        h = int(max(1, int(horizon)))
        self.memory.add(state, action_idx, reward, next_state, done, actor, horizon=h)
        
        # Learner now runs continuously in background; no per-frame queue token required.

    def background_train(self):
        while self.running:
            try:
                # Non-blocking shutdown signal check
                try:
                    token = self.train_queue.get_nowait()
                    if token is None:
                        break
                    self.train_queue.task_done()
                except queue.Empty:
                    pass

                if (not self.training_enabled) or (not getattr(metrics, "training_enabled", True)):
                    time.sleep(0.01)
                    continue

                steps_per_cycle = int(max(1, int(getattr(RL_CONFIG, "training_steps_per_sample", 1) or 1)))
                did_update = False
                for _ in range(steps_per_cycle):
                    out = train_step(self)
                    if out is None:
                        break
                    did_update = True

                if not did_update:
                    time.sleep(0.002)
                continue
            except Exception as e:
                print(f"Training error: {e}")
                traceback.print_exc()

    def save(self, filepath, now=None, is_forced_save=False):
        # Persist lightweight training progress so restarts keep long-run counters (Frame/Steps).
        try:
            with metrics.lock:
                frame_count = int(getattr(metrics, 'frame_count', 0))
                total_training_steps = int(getattr(metrics, 'total_training_steps', self.training_steps))
                total_training_steps_missed = int(getattr(metrics, 'total_training_steps_missed', 0))
                expert_ratio = float(getattr(metrics, 'expert_ratio', RL_CONFIG.expert_ratio_start))
                epsilon = float(getattr(metrics, 'epsilon', RL_CONFIG.epsilon_start))
        except Exception:
            frame_count = 0
            total_training_steps = int(self.training_steps)
            total_training_steps_missed = 0
            expert_ratio = float(getattr(RL_CONFIG, 'expert_ratio_start', 0.0))
            epsilon = float(getattr(RL_CONFIG, 'epsilon_start', 0.0))

        checkpoint = {
            'local_state_dict': self.qnetwork_local.state_dict(),
            'target_state_dict': self.qnetwork_target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_steps': self.training_steps,
            'frame_count': frame_count,
            'total_training_steps': total_training_steps,
            'total_training_steps_missed': total_training_steps_missed,
            'expert_ratio': expert_ratio,
            'epsilon': epsilon,
        }
        torch.save(checkpoint, filepath)
        if is_forced_save:
            print(f"Model saved to {filepath}")

    def load(self, filepath):
        if not os.path.exists(filepath): return False
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            local_missing, local_unexpected = self.qnetwork_local.load_state_dict(
                checkpoint.get('local_state_dict', {}),
                strict=False,
            )
            target_missing, target_unexpected = self.qnetwork_target.load_state_dict(
                checkpoint.get('target_state_dict', checkpoint.get('local_state_dict', {})),
                strict=False,
            )

            optimizer_state = checkpoint.get('optimizer_state_dict')
            if optimizer_state:
                try:
                    self.optimizer.load_state_dict(optimizer_state)
                except Exception as opt_err:
                    print(f"Optimizer state load skipped: {opt_err}")
            self.training_steps = checkpoint.get('training_steps', 0)

            if local_missing or local_unexpected or target_missing or target_unexpected:
                print(
                    "Checkpoint partially loaded "
                    f"(local missing={len(local_missing)}, local unexpected={len(local_unexpected)}, "
                    f"target missing={len(target_missing)}, target unexpected={len(target_unexpected)})"
                )

            try:
                self.sync_inference_model(force=True)
            except Exception:
                pass

            # Restore long-run counters for display/schedules (optional fields in older checkpoints)
            try:
                with metrics.lock:
                    if not RESET_METRICS:
                        metrics.expert_ratio = checkpoint.get('expert_ratio', RL_CONFIG.expert_ratio_start)
                        metrics.epsilon = checkpoint.get('epsilon', RL_CONFIG.epsilon_start)
                        metrics.frame_count = int(checkpoint.get('frame_count', 0))
                        metrics.loaded_frame_count = int(getattr(metrics, 'frame_count', 0))
                        metrics.total_training_steps = int(checkpoint.get('total_training_steps', self.training_steps))
                        metrics.total_training_steps_missed = int(checkpoint.get('total_training_steps_missed', 0))
                    else:
                        # Fresh-run counters/UI state while still loading weights.
                        metrics.expert_ratio = RL_CONFIG.expert_ratio_start
                        metrics.epsilon = RL_CONFIG.epsilon_start
                        metrics.frame_count = 0
                        metrics.loaded_frame_count = 0
                        metrics.total_training_steps = int(self.training_steps)
                        metrics.total_training_steps_missed = 0
            except Exception:
                pass

            print(f"Loaded model from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return False

    def get_q_value_range(self):
        """Return (min_q, max_q) from the current Q-network on a sample batch."""
        if len(self.memory) < 32:
            return float('nan'), float('nan')
        
        # Sample a small batch to estimate Q-range
        # Use 0.5 ratio to get a mix if possible, or just whatever is available
        batch = self.memory.sample(32, 0.5) 
        if not batch: return float('nan'), float('nan')
        
        states = batch[0]
        states_t = torch.from_numpy(states).float().to(self.device)
        
        self.qnetwork_local.eval()
        with torch.no_grad():
            q_joint = self.qnetwork_local(states_t)
            min_q = q_joint.min().item()
            max_q = q_joint.max().item()
        self.qnetwork_local.train()
        
        return min_q, max_q

    def stop(self):
        self.running = False
            
        try:
            self.train_queue.put(None, block=False)
        except queue.Full:
            pass
            
        for t in self.training_threads: 
            t.join(timeout=2.0)

# Alias for compatibility
HybridDQNAgent = DiscreteDQNAgent

def parse_frame_data(data: bytes) -> Optional[FrameData]:
    try:
        if not data or len(data) < 10: return None
        
        # Format: ">HddBBBHIBBBhhBBBBB"
        fmt = ">HddBBBHIBBBhhBBBBB"
        hdr_size = struct.calcsize(fmt)
        
        if len(data) < hdr_size: return None
        
        values = struct.unpack(fmt, data[:hdr_size])
        (num_values, subj, obj, gamestate, mode, done, frame, score,
         save, fire, zap, spinner, enemy, player, open_lvl,
         exp_fire, exp_zap, level) = values
         
        state_data = data[hdr_size:]
        state = np.frombuffer(state_data, dtype='>f4', count=num_values).astype(np.float32)
        
        return FrameData(
            state=state,
            subjreward=float(subj),
            objreward=float(obj),
            action=(bool(fire), bool(zap), spinner),
            gamestate=int(gamestate),
            done=bool(done),
            save_signal=bool(save),
            enemy_seg=int(enemy),
            player_seg=int(player),
            open_level=bool(open_lvl),
            expert_fire=bool(exp_fire),
            expert_zap=bool(exp_zap),
            level_number=int(level)
        )
    except Exception as e:
        print(f"Parse error: {e}")
        return None

def get_expert_action(enemy_seg, player_seg, is_open_level, expert_fire=False, expert_zap=False):
    """Returns (fire, zap, spinner_value)"""
    if enemy_seg == -32768 or enemy_seg == -1:
        return expert_fire, expert_zap, 0

    enemy_seg = int(enemy_seg) % 16
    player_seg = int(player_seg) % 16

    if is_open_level:
        relative_dist = enemy_seg - player_seg
        if abs(relative_dist) == 8:
            relative_dist = 8 if random.random() < 0.5 else -8
    else:
        clockwise = (enemy_seg - player_seg) % 16
        counter = (player_seg - enemy_seg) % 16
        if clockwise < 8: relative_dist = clockwise
        elif counter < 8: relative_dist = -counter
        else: relative_dist = 8 if random.random() < 0.5 else -8

    if relative_dist == 0:
        return expert_fire, expert_zap, 0

    intensity = min(0.9, 0.3 + (abs(relative_dist) * 0.05))
    spinner = -intensity if relative_dist > 0 else intensity
    return expert_fire, expert_zap, spinner

def decay_epsilon(frame_count: int) -> float:
    """Frame-driven epsilon schedule used by metrics controls."""
    try:
        fc = max(0, int(frame_count))
        start = float(getattr(RL_CONFIG, "epsilon_start", getattr(metrics, "epsilon", 0.0)))
        end = float(getattr(RL_CONFIG, "epsilon_end", start))
        step = max(1, int(getattr(RL_CONFIG, "epsilon_decay_steps", 1) or 1))
        factor = float(getattr(RL_CONFIG, "epsilon_decay_factor", 1.0) or 1.0)
        if factor >= 0.999999:
            value = start
        else:
            n = fc // step
            value = start * (factor ** n)
        value = max(end, value)
    except Exception:
        value = float(getattr(metrics, "epsilon", 0.0))
    try:
        metrics.epsilon = float(value)
    except Exception:
        pass
    return float(value)

def decay_expert_ratio(frame_count: int) -> float:
    """Frame-driven expert-ratio schedule used by metrics controls."""
    try:
        fc = max(0, int(frame_count))
        start = float(getattr(RL_CONFIG, "expert_ratio_start", getattr(metrics, "expert_ratio", 0.0)))
        minimum = float(getattr(RL_CONFIG, "expert_ratio_min", 0.0))
        step = max(1, int(getattr(RL_CONFIG, "expert_ratio_decay_steps", 1) or 1))
        factor = float(getattr(RL_CONFIG, "expert_ratio_decay", 1.0) or 1.0)
        if factor >= 0.999999:
            value = start
        else:
            n = fc // step
            value = start * (factor ** n)
        value = max(minimum, value)
    except Exception:
        value = float(getattr(metrics, "expert_ratio", 0.0))
    try:
        metrics.expert_ratio = float(value)
    except Exception:
        pass
    return float(value)

# SafeMetrics class (simplified for brevity but functional)
class SafeMetrics:
    def __init__(self, metrics):
        self.metrics = metrics
        self.lock = threading.Lock()
    
    def update_frame_count(self, delta=1):
        # Delegate to the metrics object which handles FPS calculation
        if hasattr(self.metrics, 'update_frame_count'):
            self.metrics.update_frame_count(delta)
        else:
            with self.lock: self.metrics.frame_count += delta
    
    def add_episode_reward(self, total, dqn, expert, subj=None, obj=None, length=0):
        with self.lock:
            self.metrics.episode_rewards.append(total)
            self.metrics.dqn_rewards.append(dqn)
            self.metrics.expert_rewards.append(expert)
            if subj is not None: self.metrics.subj_rewards.append(subj)
            if obj is not None: self.metrics.obj_rewards.append(obj)
            
            # Update interval accumulators for display
            self.metrics.reward_sum_interval_total += total
            self.metrics.reward_count_interval_total += 1
            self.metrics.reward_sum_interval_dqn += dqn
            self.metrics.reward_count_interval_dqn += 1
            self.metrics.reward_sum_interval_expert += expert
            self.metrics.reward_count_interval_expert += 1
            if subj is not None:
                self.metrics.reward_sum_interval_subj += subj
                self.metrics.reward_count_interval_subj += 1
            if obj is not None:
                self.metrics.reward_sum_interval_obj += obj
                self.metrics.reward_count_interval_obj += 1
                
            if length > 0:
                self.metrics.episode_length_sum_interval += length
                self.metrics.episode_length_count_interval += 1
        
        # Update rolling windows in metrics_display
        try:
            import metrics_display
            # Pass the raw DQN reward (already scaled in socket_server)
            metrics_display.add_episode_to_dqn1m_window(dqn, length)
            metrics_display.add_episode_to_dqn5m_window(dqn, length)
        except ImportError:
            print("Warning: Could not import metrics_display to update DQN windows")
        except Exception as e:
            print(f"Error updating DQN windows: {e}")

    def update_epsilon(self):
        # Simple decay logic
        with self.lock:
            if self.metrics.frame_count % RL_CONFIG.epsilon_decay_steps == 0:
                self.metrics.epsilon = max(RL_CONFIG.epsilon_end, self.metrics.epsilon * RL_CONFIG.epsilon_decay_factor)
            return self.metrics.epsilon

    def update_expert_ratio(self):
        with self.lock:
            # Simple decay logic
            if self.metrics.frame_count % RL_CONFIG.expert_ratio_decay_steps == 0:
                self.metrics.expert_ratio = max(0.0, self.metrics.expert_ratio * RL_CONFIG.expert_ratio_decay)
            return self.metrics.expert_ratio
            
    def get_effective_epsilon(self):
        with self.lock:
            getter = getattr(self.metrics, "get_effective_epsilon", None)
            if callable(getter):
                try:
                    return float(getter())
                except Exception:
                    pass
            return float(getattr(self.metrics, "epsilon", 0.0))
        
    def get_expert_ratio(self):
        with self.lock:
            getter = getattr(self.metrics, "get_expert_ratio", None)
            if callable(getter):
                try:
                    return float(getter())
                except Exception:
                    pass
            return float(getattr(self.metrics, "expert_ratio", 0.0))

    def increment_guided_count(self):
        with self.lock: self.metrics.guided_count += 1
        
    def increment_total_controls(self):
        with self.lock: self.metrics.total_controls += 1
        
    def update_action_source(self, source):
        with self.lock: self.metrics.last_action_source = source

    def get_fps(self):
        with self.lock: return getattr(self.metrics, 'fps', 0.0)

    def add_inference_time(self, t):
        with self.lock:
            if not hasattr(self.metrics, 'total_inference_time'): self.metrics.total_inference_time = 0
            self.metrics.total_inference_time += t
            if not hasattr(self.metrics, 'total_inference_requests'): self.metrics.total_inference_requests = 0
            self.metrics.total_inference_requests += 1

    def update_game_state(self, enemy_seg, open_level):
        with self.lock:
            self.metrics.enemy_seg = enemy_seg
            self.metrics.open_level = open_level
