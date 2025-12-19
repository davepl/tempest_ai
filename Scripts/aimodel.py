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
    """DQN with shared trunk and two discrete heads: FireZap (4) and Spinner (64)."""

    def __init__(self, state_size: int, hidden_size: int = 512, num_layers: int = 3):
        super(DiscreteDQN, self).__init__()
        
        self.state_size = state_size
        self.num_layers = num_layers
        self.use_noisy = bool(getattr(RL_CONFIG, "use_noisy_nets", False))
        self.noisy_std_init = float(getattr(RL_CONFIG, "noisy_std_init", 0.5) or 0.5)
        
        LinearOrNoisy = nn.Linear
        HeadLinear = NoisyLinear if self.use_noisy else nn.Linear
        head_kwargs = {"std_init": self.noisy_std_init} if self.use_noisy else {}
        
        # Shared trunk
        layer_sizes = []
        for i in range(num_layers):
            pair_index = i // 2
            layer_size = max(32, hidden_size // (2 ** pair_index))
            layer_sizes.append(layer_size)
        
        self.shared_layers = nn.ModuleList()
        self.shared_layers.append(LinearOrNoisy(state_size, layer_sizes[0]))
        for i in range(1, num_layers):
            self.shared_layers.append(LinearOrNoisy(layer_sizes[i-1], layer_sizes[i]))
        
        for layer in self.shared_layers:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight, gain=1.0)
                torch.nn.init.constant_(layer.bias, 0.0)
        
        shared_output_size = layer_sizes[-1]
        head_size = max(64, shared_output_size // 2)

        # FireZap Head (4 actions)
        self.firezap_fc = HeadLinear(shared_output_size, head_size, **head_kwargs)
        self.firezap_out = HeadLinear(head_size, FIRE_ZAP_ACTIONS, **head_kwargs)

        # Spinner Head (64 actions)
        self.spinner_fc = HeadLinear(shared_output_size, head_size, **head_kwargs)
        self.spinner_out = HeadLinear(head_size, NUM_SPINNER_BUCKETS, **head_kwargs)

        # Init heads
        for fc, out in [(self.firezap_fc, self.firezap_out), (self.spinner_fc, self.spinner_out)]:
            if isinstance(fc, nn.Linear):
                torch.nn.init.xavier_uniform_(fc.weight, gain=1.0)
                torch.nn.init.constant_(fc.bias, 0.0)
            if isinstance(out, nn.Linear):
                torch.nn.init.uniform_(out.weight, -0.003, 0.003)
                torch.nn.init.constant_(out.bias, 0.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shared = x
        for layer in self.shared_layers:
            shared = F.relu(layer(shared))
        
        # FireZap
        fz = F.relu(self.firezap_fc(shared))
        q_fz = self.firezap_out(fz)

        # Spinner
        sp = F.relu(self.spinner_fc(shared))
        q_sp = self.spinner_out(sp)

        return q_fz, q_sp

    def reset_noise(self):
        if not self.use_noisy:
            return
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

class StratifiedReplayBuffer:
    """Replay buffer with separate deques for Agent and Expert to enforce sampling ratios."""

    def __init__(self, capacity: int, state_size: int):
        self.capacity = int(max(1, capacity))
        self.state_size = int(max(1, state_size))
        self.lock = threading.Lock()
        
        # Separate buffers
        self.buffer_agent = deque(maxlen=self.capacity)
        self.buffer_expert = deque(maxlen=self.capacity)
        
        # Stats
        self.total_added = 0

    def add(self, state, firezap_idx, spinner_idx, reward, next_state, done, actor='dqn'):
        """Add experience to the appropriate buffer."""
        experience = (state, firezap_idx, spinner_idx, reward, next_state, done)
        
        with self.lock:
            if actor == 'expert':
                self.buffer_expert.append(experience)
            else:
                self.buffer_agent.append(experience)
            self.total_added += 1

    def sample(self, batch_size: int, expert_ratio: float):
        """Sample a batch with guaranteed expert ratio."""
        with self.lock:
            n_expert = int(batch_size * expert_ratio)
            n_agent = batch_size - n_expert
            
            # Adjust if not enough samples
            if len(self.buffer_expert) < n_expert:
                n_expert = len(self.buffer_expert)
                n_agent = batch_size - n_expert # Try to fill with agent
            
            if len(self.buffer_agent) < n_agent:
                n_agent = len(self.buffer_agent)
                # If we still need more and have expert, take more expert? 
                # For now, just return what we have, or smaller batch.
            
            if n_expert + n_agent == 0:
                return None

            batch_expert = random.sample(self.buffer_expert, n_expert) if n_expert > 0 else []
            batch_agent = random.sample(self.buffer_agent, n_agent) if n_agent > 0 else []

            # Attach actor tags so training can apply expert-only supervision and report per-actor stats
            batch = []
            if batch_expert:
                batch.extend([(state, fz, sp, r, ns, d, "expert") for (state, fz, sp, r, ns, d) in batch_expert])
            if batch_agent:
                batch.extend([(state, fz, sp, r, ns, d, "dqn") for (state, fz, sp, r, ns, d) in batch_agent])
            random.shuffle(batch)  # Shuffle to mix them

            # Unzip
            states, fz_idxs, sp_idxs, rewards, next_states, dones, actors = zip(*batch)

            return (
                np.array(states, dtype=np.float32),
                np.array(fz_idxs, dtype=np.int64),
                np.array(sp_idxs, dtype=np.int64),
                np.array(rewards, dtype=np.float32),
                np.array(next_states, dtype=np.float32),
                np.array(dones, dtype=np.float32),
                list(actors),
            )

    def __len__(self):
        with self.lock:
            return len(self.buffer_agent) + len(self.buffer_expert)

    def get_partition_stats(self):
        with self.lock:
            n_agent = len(self.buffer_agent)
            n_expert = len(self.buffer_expert)
            total = n_agent + n_expert
            return {
                'total_size': total,
                'total_capacity': self.capacity * 2, # Roughly
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
    """Agent using DiscreteDQN with two heads and StratifiedReplayBuffer."""

    def __init__(self, state_size, discrete_actions=None, learning_rate=RL_CONFIG.lr, 
                 gamma=RL_CONFIG.gamma, epsilon=RL_CONFIG.epsilon, memory_size=RL_CONFIG.memory_size, 
                 batch_size=RL_CONFIG.batch_size):
        self.state_size = int(state_size)
        self.learning_rate = float(learning_rate)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.batch_size = int(batch_size)
        self.device = device

        self.qnetwork_local = DiscreteDQN(state_size=self.state_size).to(self.device)
        self.qnetwork_target = DiscreteDQN(state_size=self.state_size).to(self.device)
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        self.qnetwork_target.eval()
        self.qnetwork_local.train()

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.learning_rate)
        self.memory = StratifiedReplayBuffer(memory_size, state_size=self.state_size)

        self.training_enabled = True
        self.training_steps = 0
        self.train_queue = queue.Queue(maxsize=10000)
        self.running = True
        self.training_threads = []
        
        worker = threading.Thread(target=self.background_train, daemon=True, name="TrainWorker")
        worker.start()
        self.training_threads.append(worker)

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

        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            q_fz, q_sp = self.qnetwork_local(state_t)
        self.qnetwork_local.train()
        
        fz = int(q_fz.argmax(dim=1).item())
        sp = int(q_sp.argmax(dim=1).item())
        return fz, sp

    def step(self, state, action, reward, next_state, done, actor='dqn', horizon=1, priority_reward=None):
        # action is (fire_zap_idx, spinner_idx)
        fz_idx, sp_idx = action
        self.memory.add(state, fz_idx, sp_idx, reward, next_state, done, actor)
        
        if self.training_enabled:
            try:
                if hasattr(metrics, 'training_steps_requested_interval'):
                    metrics.training_steps_requested_interval += 1
                self.train_queue.put(1, block=False)
            except queue.Full:
                if hasattr(metrics, 'training_steps_missed_interval'):
                    metrics.training_steps_missed_interval += 1
                    metrics.total_training_steps_missed += 1
                pass

    def background_train(self):
        while self.running:
            try:
                token = self.train_queue.get(timeout=0.1)
                if token is None: break
                train_step(self)
                self.train_queue.task_done()
            except queue.Empty:
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
            self.qnetwork_local.load_state_dict(checkpoint['local_state_dict'], strict=False)
            self.qnetwork_target.load_state_dict(checkpoint['target_state_dict'], strict=False)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.training_steps = checkpoint.get('training_steps', 0)

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
            q_fz, q_sp = self.qnetwork_local(states_t)
            min_q = min(q_fz.min().item(), q_sp.min().item())
            max_q = max(q_fz.max().item(), q_sp.max().item())
        self.qnetwork_local.train()
        
        return min_q, max_q

    def stop(self):
        self.running = False
        # Clear queue to ensure shutdown signal is received
        try:
            while not self.train_queue.empty():
                self.train_queue.get_nowait()
                self.train_queue.task_done()
        except Exception:
            pass
            
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
        with self.lock: return self.metrics.epsilon
        
    def get_expert_ratio(self):
        with self.lock: return self.metrics.expert_ratio

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
