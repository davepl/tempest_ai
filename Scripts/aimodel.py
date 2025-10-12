#!/usr/bin/env python3
# ==================================================================================================================
# ||                                                                                                              ||
# ||                              TEMPEST AI • MODEL, AGENT, AND UTILITIES                                       ||
# ||                                                                                                              ||
# ||  FILE: Scripts/aimodel.py                                                                                    ||
# ||  ROLE: Neural model (HybridDQN), training agent, parsing, expert helpers, keyboard, and utilities.           ||
# ||                                                                                                              ||
# ||  NEED TO KNOW:                                                                                               ||
# ||   - HybridDQN: shared trunk + discrete head (4 Q-values) + continuous head (spinner in [-0.9,0.9]).          ||
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
import subprocess
import warnings
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import select
import threading
import queue
from collections import deque, namedtuple
from datetime import datetime
import socket
import traceback
from torch.nn import SmoothL1Loss # Import SmoothL1Loss
# Robust import for running as script (Scripts on sys.path) and as package (repo root on sys.path)
try:
    from nstep_buffer import NStepReplayBuffer  # when running `python Scripts/main.py`
except Exception:
    try:
        from Scripts.nstep_buffer import NStepReplayBuffer  # when repo root is on sys.path
    except Exception:
        # If executed as package (python -m Scripts.main) and Scripts is a package
        from .nstep_buffer import NStepReplayBuffer

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

# Expose module under short name for compatibility with legacy imports/tests
sys.modules.setdefault('aimodel', sys.modules[__name__])

# Suppress warnings
warnings.filterwarnings('default')

# Global flag to track if running interactively
# Check this early before any potential tty interaction
IS_INTERACTIVE = sys.stdin.isatty()
print(f"Script Start: sys.stdin.isatty() = {IS_INTERACTIVE}") # DEBUG

# Initialize configuration
server_config = ServerConfigData()
rl_config = RLConfigData()

# Use values from config
params_count = server_config.params_count
state_size = rl_config.state_size

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
    print(f"Using CUDA device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"Using MPS device: {device}")
else:
    device = torch.device("cpu")
    print(f"Using CPU device: {device}")

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
        print("[DecompileGuard] torch._dynamo.reset() executed; forcing eager mode.")
    except Exception:
        pass
    _DYNAMO_SEEN = True


# Display key configuration parameters
print(f"Learning rate: {RL_CONFIG.lr}")
print(f"Batch size: {RL_CONFIG.batch_size}")
print(f"Memory size: {RL_CONFIG.memory_size:,}")
print(f"Hidden size: {RL_CONFIG.hidden_size}")
print(f"Number of layers: {RL_CONFIG.num_layers}")

# Calculate and display layer architecture
layer_sizes = []
for i in range(RL_CONFIG.num_layers):
    pair_index = i // 2  # 0,0 -> 1,1 -> 2,2 -> ...
    layer_size = max(32, RL_CONFIG.hidden_size // (2 ** pair_index))
    layer_sizes.append(layer_size)

print(f"Layer architecture:")
print(f"  Layer 1: {RL_CONFIG.state_size} → {layer_sizes[0]}")
for i in range(1, len(layer_sizes)):
    print(f"  Layer {i+1}: {layer_sizes[i-1]} → {layer_sizes[i]}")
print(f"  Head layers: {layer_sizes[-1]} → {max(64, layer_sizes[-1] // 2)}")

print("Replay: hybrid experience buffer (discrete + continuous)")
print(f"Mixed precision: {'enabled' if getattr(RL_CONFIG, 'use_mixed_precision', False) else 'disabled'}")
print(f"State size: {RL_CONFIG.state_size}")

# For compatibility with single-device code

# Initialize metrics
metrics = config_metrics

# Global reference to server for metrics display
metrics.global_server = None

class HybridDQN(nn.Module):
    """Hybrid DQN with discrete fire/zap actions + continuous spinner.
    
    Architecture:
    - Shared trunk: processes state features
    - Discrete head: outputs Q-values for 4 fire/zap combinations
    - Continuous head: outputs single continuous spinner value in [-0.9, +0.9]
    
    Forward pass returns: (discrete_q_values, continuous_spinner)
    - discrete_q_values: (batch_size, 4) Q-values for fire/zap combinations
    - continuous_spinner: (batch_size, 1) spinner values in [-0.9, +0.9]
    """
    def __init__(self, state_size: int, discrete_actions: int = 4, 
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
        
        # Final layer size for heads
        shared_output_size = layer_sizes[-1]
        head_size = max(64, shared_output_size // 2)  # Head layer size
        
        # Standard Q-network for discrete Q-values
        self.discrete_fc = nn.Linear(shared_output_size, head_size)
        self.discrete_out = nn.Linear(head_size, discrete_actions)
        
        #BUGBUG why fc1 and fc2?  What's the difference and why are there both?
        # Continuous head for spinner (always separate from dueling)
        self.continuous_fc1 = nn.Linear(shared_output_size, head_size)
        continuous_head_size = max(32, head_size // 2)
        self.continuous_fc2 = nn.Linear(head_size, continuous_head_size)
        self.continuous_out = nn.Linear(continuous_head_size, 1)
        
        # Initialize continuous head with smaller weights for stable training
        # BUGBUG what does this do?  Do we still need or want it?
        torch.nn.init.xavier_normal_(self.continuous_out.weight, gain=0.1)
        torch.nn.init.constant_(self.continuous_out.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (discrete_q_values, continuous_spinner)
        
        Args:
            x: Input state tensor (batch_size, state_size)
            
        Returns:
            discrete_q_values: Q-values for fire/zap combinations (batch_size, 4)
            continuous_spinner: Spinner values in [-0.9, +0.9] (batch_size, 1)
        """
        # Shared feature extraction through dynamic layers
        shared = x
        for layer in self.shared_layers:
            shared = F.relu(layer(shared))
        
        # Discrete Q-values head
        discrete = F.relu(self.discrete_fc(shared))
        discrete_q = self.discrete_out(discrete)  # (B, discrete_actions)
        
        # Continuous spinner head
        continuous = F.relu(self.continuous_fc1(shared))
        continuous = F.relu(self.continuous_fc2(continuous))
        continuous_raw = self.continuous_out(continuous)  # (B, 1)
        
        # Apply tanh to bound spinner to [-1, +1] then scale to [-0.9, +0.9]
        continuous_spinner = torch.tanh(continuous_raw) * 0.9
        
        return discrete_q, continuous_spinner

class HybridReplayBuffer:
    """Experience replay buffer with efficient PER for hybrid discrete-continuous actions.
    
    Stores experiences as: (state, discrete_action, continuous_action, reward, next_state, done, actor, horizon)
    - discrete_action: integer index for fire/zap combination (0-3)
    - continuous_action: float spinner value in [-0.9, +0.9]
    - actor: string tag identifying source of experience ('expert' or 'dqn')
    
    Uses dynamic priority sampling for efficient experience selection:
    - high_reward: Experiences with rewards above 75th percentile (computed on-the-fly)
    - recent: Most recent experiences (last 10% of buffer or 50K, whichever is larger)
    - regular: All other experiences (early samples with low rewards)
    
    Sampling uses numpy boolean indexing and array slicing for O(1) performance.
    No expensive list conversions or deque operations during sampling.
    """
    def __init__(self, capacity: int, state_size: int):
        self.capacity = capacity
        self.position = 0
        self.size = 0
        self.state_size = int(state_size)
        
        # Pre-allocated arrays for maximum speed
        self.states = np.empty((capacity, self.state_size), dtype=np.float32)
        self.discrete_actions = np.empty((capacity,), dtype=np.int32)
        self.continuous_actions = np.empty((capacity,), dtype=np.float32)
        self.rewards = np.empty((capacity,), dtype=np.float32)
        self.next_states = np.empty((capacity, self.state_size), dtype=np.float32)
        self.dones = np.empty((capacity,), dtype=np.bool_)
        self.actors = np.empty((capacity,), dtype='U10')  # Store actor tags (up to 10 chars)
        self.horizons = np.ones((capacity,), dtype=np.int32)  # n-step horizon per transition (default 1)
        
        # Rolling reward window for percentile calculation
        self.reward_window = deque(maxlen=50000)  # Track last 50K rewards
        self.high_reward_threshold = 0.0
        
        # Recent window configuration
        self.recent_window_size = 50000  # Will be updated dynamically
        
        # Random number generator
        self._rand = np.random.default_rng()


    def _update_reward_percentile(self):
        """Update high-reward threshold using rolling percentile."""
        if len(self.reward_window) < 100:  # Need minimum samples
            self.high_reward_threshold = 0.0
            return
        
        # Use 75th percentile for high-reward classification
        rewards = np.array(list(self.reward_window))
        self.high_reward_threshold = float(np.percentile(rewards, 75.0))

    def push(self, state, discrete_action, continuous_action, reward, next_state, done, actor, horizon: int):
        """Add experience to buffer with actor tag"""
        # Coerce inputs to proper types
        discrete_idx = int(discrete_action) if not isinstance(discrete_action, int) else discrete_action
        continuous_val = float(continuous_action) if not isinstance(continuous_action, float) else continuous_action
        # Require explicit, non-empty actor tag
        if not actor:
            raise ValueError("HybridReplayBuffer.push requires a non-empty actor tag")
        actor_tag = str(actor).lower().strip()
        if actor_tag in ('unknown', 'none', 'random', ''):
            raise ValueError(f"HybridReplayBuffer.push received invalid actor tag '{actor_tag}'")

        # Clamp continuous action to valid range
        continuous_val = max(-0.9, min(0.9, continuous_val))

        # Store experience
        try:
            s = np.asarray(state, dtype=np.float32)
            if s.ndim > 1:
                s = s.reshape(-1)
            if s.size < self.state_size:
                tmp = np.zeros((self.state_size,), dtype=np.float32)
                tmp[:s.size] = s
                s = tmp
            elif s.size > self.state_size:
                s = s[:self.state_size]
            self.states[self.position, :] = s
        except Exception:
            # Fallback to zeros if something went wrong
            self.states[self.position, :] = 0.0
        
        self.discrete_actions[self.position] = discrete_idx
        self.continuous_actions[self.position] = continuous_val
        self.rewards[self.position] = reward
        
        try:
            ns = np.asarray(next_state, dtype=np.float32)
            if ns.ndim > 1:
                ns = ns.reshape(-1)
            if ns.size < self.state_size:
                tmp = np.zeros((self.state_size,), dtype=np.float32)
                tmp[:ns.size] = ns
                ns = tmp
            elif ns.size > self.state_size:
                ns = ns[:self.state_size]
            self.next_states[self.position, :] = ns
        except Exception:
            self.next_states[self.position, :] = 0.0
        
        self.dones[self.position] = done
        self.actors[self.position] = actor_tag
        
        # Require explicit positive horizon
        h = int(horizon)
        if h < 1:
            raise ValueError("HybridReplayBuffer.push requires horizon >= 1")
        self.horizons[self.position] = h

        # Update position and size
        self.position = (self.position + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1
        
        # Update recent window size (10% of buffer or 50K, whichever is larger)
        self.recent_window_size = max(50000, int(self.size * 0.1))
        
        # Update rolling reward window and percentile
        self.reward_window.append(reward)
        if len(self.reward_window) % 1000 == 0:  # Update percentile every 1000 samples
            self._update_reward_percentile()
    
    def sample(self, batch_size):
        """Sample batch using bucket-based PER for quality sampling.
        
        Samples from buffer using dynamic prioritization without expensive list conversions.
        Uses numpy array slicing for O(1) sampling performance.
        """
        if self.size < batch_size:
            return None

        # Calculate recent window boundary (recent_window_size most recent samples)
        recent_start = max(0, self.size - self.recent_window_size)
        
        # Build index pools efficiently
        # High-reward pool: Find indices where reward > threshold (fast numpy boolean indexing)
        high_reward_mask = self.rewards[:self.size] >= self.high_reward_threshold
        high_reward_indices = np.where(high_reward_mask)[0]
        
        # Recent pool: Last N samples
        recent_indices = np.arange(recent_start, self.size)
        
        # Regular pool: Everything else (early samples with low rewards)
        regular_mask = (~high_reward_mask) & (np.arange(self.size) < recent_start)
        regular_indices = np.where(regular_mask)[0]
        
        # Sample from each pool proportionally
        all_indices = []
        
        # Try to get batch_size from high-reward if available
        if len(high_reward_indices) >= batch_size:
            sampled = self._rand.choice(high_reward_indices, size=batch_size, replace=False)
            all_indices.extend(sampled)
        
        # Try to get batch_size from recent if available
        if len(recent_indices) >= batch_size:
            sampled = self._rand.choice(recent_indices, size=batch_size, replace=False)
            all_indices.extend(sampled)
        
        # Try to get batch_size from regular if available
        if len(regular_indices) >= batch_size:
            sampled = self._rand.choice(regular_indices, size=batch_size, replace=False)
            all_indices.extend(sampled)
        
        # Select final batch from combined pools
        if len(all_indices) < batch_size:
            # Not enough samples in pools, fall back to uniform sampling
            indices = self._rand.choice(self.size, size=batch_size, replace=False)
        else:
            # Randomly select from the combined pool
            indices = self._rand.choice(all_indices, size=batch_size, replace=False)

        # Vectorized gather for batch data
        states_np = self.states[indices]
        next_states_np = self.next_states[indices]
        batch_discrete_actions = self.discrete_actions[indices]
        batch_continuous_actions = self.continuous_actions[indices]
        batch_rewards = self.rewards[indices]
        batch_dones = self.dones[indices]
        batch_actors = self.actors[indices]  # Get actor tags for batch
        batch_horizons = self.horizons[indices]

        # Convert to tensors (pin memory for fast H2D when on CUDA)
        use_pinned = (device.type == 'cuda')
        states = torch.from_numpy(states_np).float()
        next_states = torch.from_numpy(next_states_np).float()
        discrete_actions = torch.from_numpy(batch_discrete_actions.reshape(-1, 1)).long()
        continuous_actions = torch.from_numpy(batch_continuous_actions.reshape(-1, 1)).float()
        rewards = torch.from_numpy(batch_rewards.reshape(-1, 1)).float()
        dones = torch.from_numpy(batch_dones.reshape(-1, 1).astype(np.uint8)).float()
        horizons = torch.from_numpy(batch_horizons.reshape(-1, 1)).float()

        if use_pinned:
            states = states.pin_memory()
            next_states = next_states.pin_memory()
            discrete_actions = discrete_actions.pin_memory()
            continuous_actions = continuous_actions.pin_memory()
            rewards = rewards.pin_memory()
            dones = dones.pin_memory()
            horizons = horizons.pin_memory()

        non_block = True if device.type == 'cuda' else False
        states = states.to(device, non_blocking=non_block)
        next_states = next_states.to(device, non_blocking=non_block)
        discrete_actions = discrete_actions.to(device, non_blocking=non_block)
        continuous_actions = continuous_actions.to(device, non_blocking=non_block)
        rewards = rewards.to(device, non_blocking=non_block)
        dones = dones.to(device, non_blocking=non_block)
        horizons = horizons.to(device, non_blocking=non_block)
        
        return states, discrete_actions, continuous_actions, rewards, next_states, dones, batch_actors, horizons
    
    def __len__(self):
        return self.size
    
    def get_actor_composition(self):
        """Return statistics on actor composition of buffer"""
        if self.size == 0:
            return {'total': 0, 'dqn': 0, 'expert': 0, 'frac_dqn': 0.0, 'frac_expert': 0.0}
        
        # Count actors in the filled portion of the buffer
        actors_slice = self.actors[:self.size]
        n_dqn = np.sum(actors_slice == 'dqn')
        n_expert = np.sum(actors_slice == 'expert')
        
        return {
            'total': self.size,
            'dqn': int(n_dqn),
            'expert': int(n_expert),
            'frac_dqn': float(n_dqn) / self.size,
            'frac_expert': float(n_expert) / self.size,
        }
    
    def get_bucket_stats(self):
        """Return statistics about priority bucket sizes for monitoring."""
        # Calculate bucket sizes on-the-fly using the same logic as sample()
        recent_start = max(0, self.size - self.recent_window_size)
        high_reward_mask = self.rewards[:self.size] >= self.high_reward_threshold
        high_reward_count = np.sum(high_reward_mask)
        recent_count = self.size - recent_start
        regular_mask = (~high_reward_mask) & (np.arange(self.size) < recent_start)
        regular_count = np.sum(regular_mask)
        
        return {
            'high_reward': int(high_reward_count),
            'recent': int(recent_count),
            'regular': int(regular_count),
            'high_reward_threshold': self.high_reward_threshold,
            'recent_window_size': self.recent_window_size,
        }

## Hybrid-only path: legacy discrete-only DQNAgent removed
## N-step learning handled upstream (server) or via dedicated buffer; hybrid agent expects 1-step or n-step rewards provided.
    
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
    print("\nSetting up environment...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"Model directory: {MODEL_DIR}")
    print(f"Server will listen on {SERVER_CONFIG.host}:{SERVER_CONFIG.port}")
    print(f"Ready to handle up to {SERVER_CONFIG.max_clients} clients")

class HybridDQNAgent:
    """Hybrid DQN Agent with discrete fire/zap actions + continuous spinner"""
    
    def __init__(self, state_size, discrete_actions=4, learning_rate=RL_CONFIG.lr, 
                 gamma=RL_CONFIG.gamma, epsilon=RL_CONFIG.epsilon, 
                 epsilon_min=RL_CONFIG.epsilon_min, memory_size=RL_CONFIG.memory_size, 
                 batch_size=RL_CONFIG.batch_size):
        
        self.state_size = state_size
        self.discrete_actions = discrete_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.last_save_time = 0.0
        
        # Hybrid neural networks (single device)
        self.qnetwork_local = HybridDQN(
            state_size=state_size,
            discrete_actions=discrete_actions,
            hidden_size=RL_CONFIG.hidden_size,
            num_layers=RL_CONFIG.num_layers
        ).to(device)
        
        self.qnetwork_target = HybridDQN(
            state_size=state_size,
            discrete_actions=discrete_actions,
            hidden_size=RL_CONFIG.hidden_size,
            num_layers=RL_CONFIG.num_layers
        ).to(device)
        
        # Use same network for inference (no separate inference network)
        self.qnetwork_inference = self.qnetwork_local
        
        # Initialize target network
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        # Establish persistent modes:
        # - training (local) stays in train mode
        # - target stays in eval mode
        # - inference stays in eval mode if it's a dedicated copy
        self.qnetwork_local.train()
        self.qnetwork_target.eval()
        if self.qnetwork_inference is not self.qnetwork_local:
            self.qnetwork_inference.eval()

        # Defensive: unwrap any lingering compiled wrappers & reset dynamo once.
        try:
            before_types = (self.qnetwork_local.__class__.__name__, self.qnetwork_target.__class__.__name__)
            self.qnetwork_local = unwrap_compiled_module(self.qnetwork_local)
            self.qnetwork_target = unwrap_compiled_module(self.qnetwork_target)
            self.qnetwork_inference = unwrap_compiled_module(self.qnetwork_inference)
            after_types = (self.qnetwork_local.__class__.__name__, self.qnetwork_target.__class__.__name__)
            if before_types != after_types:
                print(f"[DecompileGuard] Unwrapped compiled modules: {before_types} -> {after_types}")
            _force_dynamo_reset_once()
        except Exception:
            pass
        
        # Store device reference
        self.device = device
        
        # Optimizer with separate parameter groups for discrete and continuous components
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)
        
        # Experience replay (simplified to uniform sampling only)
        self.memory = HybridReplayBuffer(memory_size, state_size=self.state_size)
        print("Using standard HybridReplayBuffer (uniform sampling, eager mode)")

        # Training queue and metrics
        self.train_queue = queue.Queue(maxsize=10000)
        self.training_steps = 0
        self.last_target_update = 0
        self.last_inference_sync = 0
        # Honor global training enable toggle
        self.training_enabled = True

        # Thread synchronization for training
        self.training_lock = threading.Lock()

        # Background training worker(s)
        self.running = True
        self.num_training_workers = int(getattr(RL_CONFIG, 'training_workers', 1) or 1)
        self.training_threads = []
        for i in range(self.num_training_workers):
            t = threading.Thread(target=self.background_train, daemon=True, name=f"HybridTrainWorker-{i}")
            t.start()
            self.training_threads.append(t)
        
    def act(self, state, epsilon: float, add_noise: bool):
        """Select hybrid action using epsilon-greedy for discrete + Gaussian noise for continuous
        
        Returns:
            discrete_action: int (0-3) for fire/zap combination
            continuous_action: float in [-0.9, +0.9] for spinner
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        # Do not flip modes every call; rely on persistent .eval() for dedicated inference model
        with torch.no_grad():
            discrete_q, continuous_pred = self.qnetwork_inference(state)
        
        # Discrete action selection (epsilon-greedy) or fixed FIRE when spinner_only
        if getattr(RL_CONFIG, 'spinner_only', False):
            # FIRE=1, ZAP=0 -> discrete index 2 (binary 10)
            discrete_action = 2
        else:
            if random.random() < epsilon:
                discrete_action = random.randint(0, self.discrete_actions - 1)
            else:
                discrete_action = discrete_q.cpu().data.numpy().argmax()
        
        # Continuous action selection (predicted value + optional exploration noise)
        continuous_action = continuous_pred.cpu().data.numpy()[0, 0]
        if add_noise and epsilon > 0:
            # Add Gaussian noise scaled by epsilon for exploration
            # Use HIGHER noise for continuous (2x epsilon factor) to break local minima
            noise_scale = epsilon * 1.8  # 180% of action range at full epsilon (was 90%)
            noise = np.random.normal(0, noise_scale)
            continuous_action = np.clip(continuous_action + noise, -0.9, 0.9)
        
        return int(discrete_action), float(continuous_action)
    
    def step(self, state, discrete_action, continuous_action, reward, next_state, done, actor, horizon):
        """Add experience to memory and queue training"""
        if actor is None:
            actor = 'dqn'  # Default to DQN if not specified
        
        self.memory.push(
            state,
            discrete_action,
            continuous_action,
            reward,
            next_state,
            done,
            actor=actor,
            horizon=horizon,
        )
        
        # Queue multiple training steps per experience
        if not getattr(metrics, 'training_enabled', True) or not self.training_enabled:
            return
        # Apply training_steps_per_sample ONLY in the background trainer to avoid double-counting here.
        # Enqueue a single token per environment sample to minimize queue overhead.
        try:
            # Record a requested training step token
            try:
                metrics.training_steps_requested_interval += 1
            except Exception:
                pass
            self.train_queue.put_nowait(True)
        except queue.Full:
            #BUGBUG we should at least track how many we discard vs process to keep that as a metric
            try:
                metrics.training_steps_missed_interval += 1
                metrics.total_training_steps_missed += 1
            except Exception:
                pass
        # Optional telemetry
        try:
            metrics.training_queue_size = int(self.train_queue.qsize())
        except Exception:
            pass

    def background_train(self):
        """Background worker that drains the train_queue and performs train steps."""
        worker_id = threading.current_thread().name
        print(f"Training thread {worker_id} started on {self.device}")
        while self.running:
            try:
                _ = self.train_queue.get()  # block until work arrives
                # Skip training work if disabled; still mark task done
                if not getattr(metrics, 'training_enabled', True) or not self.training_enabled:
                    self.train_queue.task_done()
                    continue
                steps_per_req = int(getattr(RL_CONFIG, 'training_steps_per_sample', 1) or 1)
                for _ in range(steps_per_req):
                    # Acquire training lock to prevent concurrent model/optimizer access
                    with self.training_lock:
                        loss_val = self.train_step()
                        did_train = (loss_val is not None)
                        # Count only real optimizer updates, not skipped/no-op passes
                        if did_train:
                            try:
                                metrics.training_steps_interval += 1
                                metrics.total_training_steps += 1
                            except Exception:
                                pass
                # Optional telemetry after consuming a token
                try:
                    with self.training_lock:  # Protect metrics access
                        metrics.training_queue_size = int(self.train_queue.qsize())
                except Exception:
                    pass
                self.train_queue.task_done()
            except AssertionError:
                raise
            except Exception as e:
                print(f"Hybrid training error in {worker_id}: {e}")
                import traceback; traceback.print_exc()
                time.sleep(0.01)
    
    def train_step(self):
        """Perform one optimizer update."""
        import time
        start_time = time.time()
        
        # Global gate
        if not getattr(metrics, 'training_enabled', True) or not self.training_enabled:
            return None
        # Post-load burn-in: require some fresh frames after loading before training
        try:
            loaded_fc = int(getattr(metrics, 'loaded_frame_count', 0) or 0)
            require_new = int(getattr(RL_CONFIG, 'min_new_frames_after_load_to_train', 0) or 0)
            if loaded_fc > 0 and (metrics.frame_count - loaded_fc) < require_new:
                return None
        except Exception:
            pass

        # Require at least one batch worth of data
        if len(self.memory) < self.batch_size:
            return None

        # Sample batch
        batch_start = time.time()
        batch = self.memory.sample(self.batch_size)
        if batch is None:
            return None

        states, discrete_actions, continuous_actions, rewards, next_states, dones, actors, horizons = batch
        batch_time = time.time() - batch_start

        # Compute batch actor composition for diagnostics
        try:
            actor_dqn_mask = np.array([a == 'dqn' for a in actors], dtype=bool)
            actor_expert_mask = np.array([a == 'expert' for a in actors], dtype=bool)
            n_dqn = actor_dqn_mask.sum()
            n_expert = actor_expert_mask.sum()
            frac_dqn = n_dqn / len(actors) if len(actors) > 0 else 0.0
            
            # Store batch composition metrics
            metrics.batch_frac_dqn = float(frac_dqn)
            metrics.batch_n_dqn = int(n_dqn)
            metrics.batch_n_expert = int(n_expert)
            
            # Log batch composition every 10000 training steps (disabled for normal operation)
            # if self.training_steps > 0 and self.training_steps % 10000 == 0:
            #     print(f"[BATCH] Step {self.training_steps}: {n_dqn} DQN ({frac_dqn*100:.1f}%) / {n_expert} expert ({(1-frac_dqn)*100:.1f}%)")
        except Exception:
            actor_dqn_mask = None
            actor_expert_mask = None
            pass

        # Forward pass
        forward_start = time.time()
        discrete_q_pred, continuous_pred = self.qnetwork_local(states)
        discrete_q_selected = discrete_q_pred.gather(1, discrete_actions)
        forward_time = time.time() - forward_start

        # Target computation using DOUBLE DQN to prevent Q-value overestimation
        # Vanilla DQN: target = r + γ * max_a' Q_target(s',a')  ← Maximization bias!
        # Double DQN: target = r + γ * Q_target(s', argmax_a' Q_local(s',a'))  ← Debiased!
        target_start = time.time()
        with torch.no_grad():
            # Use LOCAL network to SELECT best action (argmax)
            next_q_local, _ = self.qnetwork_local(next_states)
            best_actions = next_q_local.max(1)[1].unsqueeze(1)  # argmax over actions
            
            # Use TARGET network to EVALUATE that action
            next_q_target, _ = self.qnetwork_target(next_states)
            discrete_q_next_max = next_q_target.gather(1, best_actions)  # Q_target(s', a*) where a* = argmax Q_local
            # If horizons>1 (n-step return), apply gamma^h to the bootstrap term
            gamma_h = torch.pow(torch.tensor(self.gamma, device=rewards.device), horizons)
            discrete_targets = rewards + (gamma_h * discrete_q_next_max * (1 - dones))
            # Optional safety: clip TD targets to avoid runaway value scales
            td_clip = getattr(RL_CONFIG, 'td_target_clip', None)
            if td_clip is not None:
                try:
                    lo, hi = (-float(td_clip), float(td_clip)) if isinstance(td_clip, (int, float)) else td_clip
                    discrete_targets = discrete_targets.clamp(min=float(lo), max=float(hi))
                except Exception:
                    pass
            
            # ADVANTAGE-WEIGHTED POLICY GRADIENT for continuous actions
            # KEY FIX: Compute advantages SEPARATELY per actor type to avoid cross-contamination
            # If we compute advantages across mixed expert+DQN batch, expert's high rewards
            # make DQN's lower rewards look bad, suppressing DQN learning!
            
            # CRITICAL FIX: Reduced advantage scaling from 1.5 → 0.5 to prevent extreme weights
            # Was: exp(adv * 1.5) with 90x max weight causing loss instability
            # Now: exp(adv * 0.5) with ~4.5x max weight for stable learning
            
            advantage_weights = torch.ones_like(rewards)
            
            try:
                if actor_dqn_mask is not None and actor_expert_mask is not None:
                    # Convert numpy masks to torch boolean masks on the same device for indexing
                    torch_mask_dqn = torch.from_numpy(actor_dqn_mask).to(device=rewards.device, dtype=torch.bool).view(-1)
                    torch_mask_exp = torch.from_numpy(actor_expert_mask).to(device=rewards.device, dtype=torch.bool).view(-1)

                    # DQN advantages: compare DQN frames to OTHER DQN frames
                    if n_dqn > 1:
                        dqn_rewards = rewards[torch_mask_dqn]
                        dqn_mean = dqn_rewards.mean()
                        dqn_std = dqn_rewards.std() + 1e-8
                        dqn_advantages = (dqn_rewards - dqn_mean) / dqn_std
                        dqn_advantages = dqn_advantages.clamp(-3, 3)
                        # Reduced scaling: 0.5 instead of 1.5 (was causing 90x weights!)
                        dqn_weights = torch.exp(dqn_advantages * 0.5).clamp(0.1, 5.0)
                        advantage_weights[torch_mask_dqn] = dqn_weights
                    
                    # Expert advantages: compare expert frames to OTHER expert frames
                    if n_expert > 1:
                        exp_rewards = rewards[torch_mask_exp]
                        exp_mean = exp_rewards.mean()
                        exp_std = exp_rewards.std() + 1e-8
                        exp_advantages = (exp_rewards - exp_mean) / exp_std
                        exp_advantages = exp_advantages.clamp(-3, 3)
                        # Reduced scaling: 0.5 instead of 1.5
                        exp_weights = torch.exp(exp_advantages * 0.5).clamp(0.1, 5.0)
                        advantage_weights[torch_mask_exp] = exp_weights
                else:
                    # Fallback: compute advantages across full batch (old behavior)
                    reward_mean = rewards.mean()
                    reward_std = rewards.std() + 1e-8
                    advantages = (rewards - reward_mean) / reward_std
                    advantages = advantages.clamp(-3, 3)
                    advantage_weights = torch.exp(advantages * 0.5).clamp(0.1, 5.0)
            except Exception:
                # Fallback: compute advantages across full batch
                reward_mean = rewards.mean()
                reward_std = rewards.std() + 1e-8
                advantages = (rewards - reward_mean) / reward_std
                advantages = advantages.clamp(-3, 3)
                advantage_weights = torch.exp(advantages * 0.5).clamp(0.1, 5.0)
            
            continuous_targets = continuous_actions.clone()  # Start with taken actions
            
            # For DQN samples, use predicted continuous as target to reinforce current policy
            # For expert samples, use taken actions to learn optimal behavior
            if 'torch_mask_dqn' in locals() and torch_mask_dqn.any():
                continuous_targets[torch_mask_dqn] = continuous_pred[torch_mask_dqn]
        target_time = time.time() - target_start

        # Losses
        loss_start = time.time()
        w_cont = float(getattr(RL_CONFIG, 'continuous_loss_weight', 1.0) or 1.0)
        w_disc = float(getattr(RL_CONFIG, 'discrete_loss_weight', 1.0) or 1.0)

        # Optionally restrict discrete loss to expert frames only
        if bool(getattr(RL_CONFIG, 'discrete_expert_only', False)) and 'torch_mask_exp' in locals() and torch_mask_exp.any():
            d_loss_raw = F.huber_loss(discrete_q_selected, discrete_targets, reduction='none')
            d_mask = torch_mask_exp.view(-1, 1).float()
            denom = d_mask.mean().clamp(min=1e-6)
            d_loss = (d_loss_raw * d_mask).sum() / (d_loss_raw.numel() * denom.item())
        else:
            d_loss = F.huber_loss(discrete_q_selected, discrete_targets, reduction='mean')
        
        # Continuous loss: ADVANTAGE-WEIGHTED to amplify learning from good experiences
        # REDUCED SCALING: exp(adv * 0.5) with 5x max weight (was 1.5 * 100x causing instability!)
        # High reward (+1.5σ) → 2.1x gradient strength → moderate boost
        # Average reward (0σ) → 1.0x gradient → normal learning
        # Low reward (-1.5σ) → 0.47x gradient → slight reduction
        # This prevents overfitting to rare high-reward frames while still biasing toward quality

        c_loss_raw = F.mse_loss(continuous_pred, continuous_targets, reduction='none')
        c_loss = (c_loss_raw * advantage_weights).mean()
        
        # In spinner-only mode, ignore discrete loss entirely
        if getattr(RL_CONFIG, 'spinner_only', False):
            total_loss = w_cont * c_loss
        else:
            total_loss = (w_disc * d_loss) + (w_cont * c_loss)
        loss_time = time.time() - loss_start

        # Compute per-actor metrics for diagnostics
        try:
            if actor_dqn_mask is not None and actor_expert_mask is not None:
                # TD errors per actor
                td_errors = (discrete_q_selected - discrete_targets).detach().cpu().numpy().flatten()
                if n_dqn > 0:
                    metrics.td_err_mean_dqn = float(np.abs(td_errors[actor_dqn_mask]).mean())
                    metrics.reward_mean_dqn = float(rewards.cpu().numpy().flatten()[actor_dqn_mask].mean())
                    metrics.q_mean_dqn = float(discrete_q_selected.detach().cpu().numpy().flatten()[actor_dqn_mask].mean())
                if n_expert > 0:
                    metrics.td_err_mean_expert = float(np.abs(td_errors[actor_expert_mask]).mean())
                    metrics.reward_mean_expert = float(rewards.cpu().numpy().flatten()[actor_expert_mask].mean())
                    metrics.q_mean_expert = float(discrete_q_selected.detach().cpu().numpy().flatten()[actor_expert_mask].mean())
                # Per-actor discrete loss means (per-sample)
                try:
                    d_loss_per = F.huber_loss(discrete_q_selected, discrete_targets, reduction='none').detach().cpu().numpy().flatten()
                    if n_dqn > 0:
                        metrics.d_loss_mean_dqn = float(d_loss_per[actor_dqn_mask].mean())
                    if n_expert > 0:
                        metrics.d_loss_mean_expert = float(d_loss_per[actor_expert_mask].mean())
                except Exception:
                    pass
                # Selected/target Q means per actor
                try:
                    q_sel_np = discrete_q_selected.detach().cpu().numpy().flatten()
                    q_tgt_np = discrete_targets.detach().cpu().numpy().flatten()
                    if n_dqn > 0:
                        metrics.q_sel_mean_dqn = float(q_sel_np[actor_dqn_mask].mean())
                        metrics.q_tgt_mean_dqn = float(q_tgt_np[actor_dqn_mask].mean())
                    if n_expert > 0:
                        metrics.q_sel_mean_expert = float(q_sel_np[actor_expert_mask].mean())
                        metrics.q_tgt_mean_expert = float(q_tgt_np[actor_expert_mask].mean())
                except Exception:
                    pass
        except Exception:
            pass

        # Backward pass
        backward_start = time.time()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        # If spinner-only, clear any stray grads in discrete head to avoid updates
        if getattr(RL_CONFIG, 'spinner_only', False):
            try:
                for name, p in self.qnetwork_local.named_parameters():
                    if name.startswith('discrete_') and p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()
            except Exception:
                pass
        backward_time = time.time() - backward_start

        # Optional gradient diagnostics: measure each head's contribution
        grad_diag_start = time.time()
        try:
            interval = int(getattr(RL_CONFIG, 'grad_diag_interval', 0) or 0)
        except Exception:
            interval = 0
        if interval and (self.training_steps % interval == 0):
            try:
                # Compute separate grads by re-backpropagating each component on a fresh graph snapshot
                # Snapshot current parameter grads to restore later
                saved_grads = [p.grad.detach().clone() if p.grad is not None else None for p in self.qnetwork_local.parameters()]

                def zero_grads():
                    for p in self.qnetwork_local.parameters():
                        if p.grad is not None:
                            p.grad.detach_()
                            p.grad.zero_()

                # Recompute forward needed tensors under no grad accumulation
                # Note: reuse already computed discrete_q_selected/continuous_pred where possible

                # 1) Discrete-only backward
                zero_grads()
                d_loss.backward(retain_graph=True)
                trunk_norm_d, disc_head_norm_d, cont_head_norm_d = 0.0, 0.0, 0.0
                try:
                    for name, p in self.qnetwork_local.named_parameters():
                        if p.grad is None:
                            continue
                        g = p.grad.data
                        gn = g.norm(2).item()
                        if name.startswith('shared_layers'):
                            trunk_norm_d += gn * gn
                        elif name.startswith('discrete_'):
                            disc_head_norm_d += gn * gn
                        elif name.startswith('continuous_'):
                            cont_head_norm_d += gn * gn
                    trunk_norm_d = trunk_norm_d ** 0.5
                    disc_head_norm_d = disc_head_norm_d ** 0.5
                    cont_head_norm_d = cont_head_norm_d ** 0.5
                except Exception:
                    pass

                # 2) Continuous-only backward
                zero_grads()
                (w_cont * c_loss).backward(retain_graph=False)
                trunk_norm_c, disc_head_norm_c, cont_head_norm_c = 0.0, 0.0, 0.0
                try:
                    for name, p in self.qnetwork_local.named_parameters():
                        if p.grad is None:
                            continue
                        g = p.grad.data
                        gn = g.norm(2).item()
                        if name.startswith('shared_layers'):
                            trunk_norm_c += gn * gn
                        elif name.startswith('discrete_'):
                            disc_head_norm_c += gn * gn
                        elif name.startswith('continuous_'):
                            cont_head_norm_c += gn * gn
                    trunk_norm_c = trunk_norm_c ** 0.5
                    disc_head_norm_c = disc_head_norm_c ** 0.5
                    cont_head_norm_c = cont_head_norm_c ** 0.5
                except Exception:
                    pass

                # Restore original grads
                for p, g in zip(self.qnetwork_local.parameters(), saved_grads):
                    if g is None:
                        p.grad = None
                    else:
                        if p.grad is None:
                            p.grad = g.clone()
                        else:
                            p.grad.detach_()
                            p.grad.copy_(g)

                # Publish metrics
                try:
                    metrics.grad_trunk_d = float(trunk_norm_d)
                    metrics.grad_trunk_c = float(trunk_norm_c)
                    metrics.grad_head_disc_d = float(disc_head_norm_d)
                    metrics.grad_head_disc_c = float(disc_head_norm_c)
                    metrics.grad_head_cont_d = float(cont_head_norm_d)
                    metrics.grad_head_cont_c = float(cont_head_norm_c)
                except Exception:
                    pass
            except Exception:
                pass
        grad_diag_time = time.time() - grad_diag_start
        
        # Compute gradient norm BEFORE clipping
        grad_norm_start = time.time()
        total_grad_norm = 0.0
        for p in self.qnetwork_local.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_grad_norm += param_norm.item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        
        # Gradient clipping for stability (critical with 100x advantage weights)
        max_norm = 10.0
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), max_norm=max_norm)
        
        # Compute gradient norm AFTER clipping to measure clip effect
        clipped_grad_norm = 0.0
        for p in self.qnetwork_local.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                clipped_grad_norm += param_norm.item() ** 2
        clipped_grad_norm = clipped_grad_norm ** 0.5
        
        # Clip delta: ratio of clipped to original (1.0 = no clipping, <1.0 = clipped)
        clip_delta = clipped_grad_norm / max(total_grad_norm, 1e-8)
        grad_norm_time = time.time() - grad_norm_start
        
        # Optimizer step
        optimizer_start = time.time()
        self.optimizer.step()
        optimizer_time = time.time() - optimizer_start

        # Update training counters
        counter_start = time.time()
        self.training_steps += 1
        try:
            metrics.total_training_steps += 1
            metrics.training_steps_interval += 1
            metrics.memory_buffer_size = len(self.memory)
            
            # Report buffer composition every 10000 training steps (disabled for normal operation)
            # if self.training_steps % 10000 == 0:
            #     comp = self.memory.get_actor_composition()
            #     print(f"[BUFFER] Step {self.training_steps}: {comp['total']} total, "
            #           f"{comp['dqn']} DQN ({comp['frac_dqn']*100:.1f}%), "
            #           f"{comp['expert']} expert ({comp['frac_expert']*100:.1f}%)")
        except Exception:
            pass
        # Track loss values
        try:
            loss_val = float(total_loss.item())
            metrics.losses.append(loss_val)
            metrics.loss_sum_interval += loss_val
            metrics.loss_count_interval += 1
            # Track gradient norms for monitoring
            metrics.last_grad_norm = float(total_grad_norm)
            metrics.last_clip_delta = float(clip_delta)
            # Track component losses
            last_d = float((w_disc * d_loss).item())
            last_c = float((w_cont * c_loss).item())
            metrics.last_d_loss = last_d
            metrics.last_c_loss = last_c
            # Accumulate interval-averaged component losses
            try:
                metrics.d_loss_sum_interval += last_d
                metrics.d_loss_count_interval += 1
                metrics.c_loss_sum_interval += last_c
                metrics.c_loss_count_interval += 1
            except Exception:
                pass
            # Advantage weight summaries
            try:
                metrics.adv_w_mean = float(advantage_weights.mean().item())
                metrics.adv_w_max = float(advantage_weights.max().item())
                # If masks available, compute per-actor means
                if 'torch_mask_dqn' in locals() and torch_mask_dqn.any():
                    metrics.adv_w_mean_dqn = float(advantage_weights[torch_mask_dqn].mean().item())
                if 'torch_mask_exp' in locals() and torch_mask_exp.any():
                    metrics.adv_w_mean_expert = float(advantage_weights[torch_mask_exp].mean().item())
            except Exception:
                pass
            # Discrete-head global diagnostics for this batch
            try:
                # Action distribution (0..3)
                acts_np = discrete_actions.detach().cpu().numpy().flatten()
                for a in range(4):
                    frac = float((acts_np == a).mean())
                    if a == 0: metrics.action_frac_0 = frac
                    elif a == 1: metrics.action_frac_1 = frac
                    elif a == 2: metrics.action_frac_2 = frac
                    elif a == 3: metrics.action_frac_3 = frac
                # Agreement rate: taken action vs current policy argmax
                with torch.no_grad():
                    dq_now, _ = self.qnetwork_local(states)
                    a_star = dq_now.max(1)[1].unsqueeze(1)
                    agree = (a_star == discrete_actions).float().mean().item()
                metrics.action_agree_pct = float(agree * 100.0)
                # Batch done fraction and horizon mean
                try:
                    metrics.batch_done_frac = float(dones.detach().cpu().numpy().mean())
                except Exception:
                    metrics.batch_done_frac = 0.0
                try:
                    metrics.batch_h_mean = float(horizons.detach().cpu().numpy().mean())
                except Exception:
                    metrics.batch_h_mean = 1.0
            except Exception:
                pass
        except Exception:
            pass
        counter_time = time.time() - counter_start

        # Target network update: support soft updates (Polyak) or periodic hard copy
        target_update_start = time.time()
        if getattr(RL_CONFIG, 'use_soft_target_update', False):
            try:
                tau = float(getattr(RL_CONFIG, 'soft_target_tau', 0.005) or 0.005)
            except Exception:
                tau = 0.005
            with torch.no_grad():
                for tgt_p, src_p in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
                    tgt_p.data.mul_(1.0 - tau).add_(src_p.data, alpha=tau)
            # Telemetry (treat as a target update for age tracking)
            try:
                metrics.last_target_update_frame = metrics.frame_count
                metrics.last_target_update_time = time.time()
            except Exception:
                pass
        else:
            # Hard target update
            if self.training_steps % RL_CONFIG.target_update_freq == 0:
                self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
                # Telemetry
                try:
                    metrics.last_target_update_frame = metrics.frame_count
                    metrics.last_target_update_time = time.time()
                    metrics.last_hard_target_update_frame = metrics.frame_count
                    metrics.last_hard_target_update_time = time.time()
                except Exception:
                    pass
        target_update_time = time.time() - target_update_start

        # Print profiling info every 100 steps
        total_time = time.time() - start_time
        #if self.training_steps % 100 == 0:
            #print(f"[PROFILING] Step {self.training_steps}: Total={total_time:.4f}s | "
            #      f"Batch={batch_time:.4f}s | Forward={forward_time:.4f}s | Target={target_time:.4f}s | "
            #      f"Loss={loss_time:.4f}s | Backward={backward_time:.4f}s | GradDiag={grad_diag_time:.4f}s | "
            #      f"GradNorm={grad_norm_time:.4f}s | Optimizer={optimizer_time:.4f}s | "
            #      f"Counters={counter_time:.4f}s | TargetUpdate={target_update_time:.4f}s")

        return float(total_loss.item())

    def set_training_enabled(self, enabled: bool):
        """Enable/disable training at runtime. When disabling, drain pending work."""
        self.training_enabled = bool(enabled)
        # Reflect into global metrics for consistency
        try:
            metrics.training_enabled = bool(enabled)
        except Exception:
            pass
        # If disabling, drain queue quickly to avoid stale tasks
        if not self.training_enabled:
            try:
                while not self.train_queue.empty():
                    self.train_queue.get_nowait()
                    self.train_queue.task_done()
            except Exception:
                pass

    def get_learning_rate(self) -> float:
        """Return the current optimizer learning rate."""
        try:
            if self.optimizer.param_groups:
                return float(self.optimizer.param_groups[0].get('lr', self.learning_rate))
        except Exception:
            pass
        return float(self.learning_rate)

    def adjust_learning_rate(self, delta: float, kb_handler=None) -> float:
        """Adjust learning rate by delta (positive or negative) with safety clamping."""
        current_lr = self.get_learning_rate()
        try:
            delta = float(delta)
        except Exception:
            delta = 0.0
        new_lr = max(1e-6, current_lr + delta)
        for group in self.optimizer.param_groups:
            group['lr'] = float(new_lr)
        self.learning_rate = float(new_lr)
        try:
            metrics.manual_lr_override = True
            metrics.manual_learning_rate = float(new_lr)
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
        return float(new_lr)

    def update_target_network(self):
        """Hard update target network from local and record telemetry."""
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        try:
            metrics.last_target_update_frame = metrics.frame_count
            metrics.last_target_update_time = time.time()
            metrics.last_hard_target_update_frame = metrics.frame_count
            metrics.last_hard_target_update_time = time.time()
        except Exception:
            pass

    def force_hard_target_update(self):
        """Alias for explicit hard target refresh (server compatibility)."""
        self.update_target_network()
    
    def save(self, filepath):
        """Save hybrid model checkpoint with basic rate limiting"""
        is_forced_save = ("exit" in str(filepath)) or ("shutdown" in str(filepath))
        now = time.time()
        min_interval = 30.0
        if not is_forced_save and (now - self.last_save_time) < min_interval:
            return
        # Persist key training/metrics to avoid losing progress on restart
        # Respect saved_expert_ratio when override/expert_mode is active
        try:
            if metrics.expert_mode or metrics.override_expert:
                ratio_to_save = metrics.saved_expert_ratio
            else:
                ratio_to_save = metrics.expert_ratio
        except Exception:
            ratio_to_save = float(getattr(metrics, 'expert_ratio', 0.0))
        checkpoint = {
            'local_state_dict': self.qnetwork_local.state_dict(),
            'target_state_dict': self.qnetwork_target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_steps': self.training_steps,
            'state_size': self.state_size,
            'discrete_actions': self.discrete_actions,
            'memory_size': len(self.memory),
            'architecture': 'hybrid',  # Mark as hybrid architecture
            # Metrics/state for persistence
            'frame_count': int(getattr(metrics, 'frame_count', 0)),
            'epsilon': float(getattr(metrics, 'epsilon', getattr(RL_CONFIG, 'epsilon_start', 0.1))),
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
            print(f"Hybrid model saved to {filepath} (frame {getattr(metrics, 'frame_count', 0)})")
    
    def load(self, filepath):
        """Load hybrid model checkpoint"""
        try:
            from config import RESET_METRICS, SERVER_CONFIG, FORCE_FRESH_MODEL
        except ImportError:
            from Scripts.config import RESET_METRICS, SERVER_CONFIG, FORCE_FRESH_MODEL

        if FORCE_FRESH_MODEL:
            print("FORCE_FRESH_MODEL is True - skipping model load, starting fresh weights")
            return False

        if os.path.exists(filepath):
            try:
                checkpoint = torch.load(filepath, map_location=device)

                # Verify architecture compatibility
                if checkpoint.get('architecture') != 'hybrid':
                    print("Warning: Loading non-hybrid checkpoint into hybrid agent")
                    return False

                # Load weights (tolerate minor shape mismatches by strict=True here as it's same arch)
                self.qnetwork_local.load_state_dict(checkpoint['local_state_dict'])
                self.qnetwork_target.load_state_dict(checkpoint['target_state_dict'])
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    # Ensure LR honors current config
                    for g in self.optimizer.param_groups:
                        g['lr'] = RL_CONFIG.lr
                    print(f"Optimizer state loaded. LR set to config value: {RL_CONFIG.lr}")
                except Exception:
                    print("Optimizer state not loaded (mismatch). Using fresh optimizer.")

                self.training_steps = int(checkpoint.get('training_steps', 0))

                # No separate inference network to sync in single device setup

                # Sanity-check Q-values on load to catch corruption/explosion or collapse
                try:
                    with torch.no_grad():
                        dummy = torch.zeros(1, self.state_size, device=self.device)
                        dq, _ = self.qnetwork_local(dummy)
                        qmax = float(dq.max().item())
                        qmin = float(dq.min().item())
                        q_range = max(abs(qmax), abs(qmin))
                        
                        # Check for extreme Q-values (too large or too small)
                        # Normal range should be roughly [-5, +5] for typical reward scales
                        needs_rescale = False
                        target_scale = 2.0  # Target magnitude for rescaling
                        
                        if q_range > 10.0:
                            # Q-values too large - risk of overflow
                            print(f"WARNING: Loaded model has large Q-values [{qmin:.3f}, {qmax:.3f}]. Rescaling down...")
                            scale = target_scale / q_range
                            needs_rescale = True
                        elif q_range < 0.01 and q_range > 0:
                            # Q-values collapsed near zero - may indicate vanishing gradients or bad init
                            print(f"WARNING: Loaded model has collapsed Q-values [{qmin:.6f}, {qmax:.6f}]. Rescaling up...")
                            scale = target_scale / q_range
                            needs_rescale = True
                        
                        if needs_rescale:
                            for p in self.qnetwork_local.parameters():
                                p.data.mul_(scale)
                            for p in self.qnetwork_target.parameters():
                                p.data.mul_(scale)
                            dq2, _ = self.qnetwork_local(dummy)
                            print(f"Rescaled Q-value range: [{float(dq2.min().item()):.3f}, {float(dq2.max().item()):.3f}]")
                except Exception as e:
                    print(f"Warning: Q-value sanity check failed: {e}")
                    pass

                # Restore metrics unless explicitly reset
                if RESET_METRICS:
                    print("RESET_METRICS=True: resetting epsilon/expert_ratio; leaving frame_count at 0")
                    metrics.epsilon = RL_CONFIG.epsilon_start
                    metrics.expert_ratio = RL_CONFIG.expert_ratio_start
                    metrics.last_decay_step = 0
                    metrics.last_epsilon_decay_step = 0
                    metrics.frame_count = 0
                    metrics.loaded_frame_count = 0
                    try:
                        metrics.episode_rewards.clear(); metrics.dqn_rewards.clear(); metrics.expert_rewards.clear()
                    except Exception:
                        pass
                else:
                    # Frame count and training schedule
                    metrics.frame_count = int(checkpoint.get('frame_count', 0))
                    metrics.loaded_frame_count = int(metrics.frame_count)
                    # Epsilon/Expert
                    metrics.epsilon = float(checkpoint.get('epsilon', RL_CONFIG.epsilon_start))
                    metrics.expert_ratio = float(checkpoint.get('expert_ratio', RL_CONFIG.expert_ratio_start))
                    metrics.last_decay_step = int(checkpoint.get('last_decay_step', 0))
                    metrics.last_epsilon_decay_step = int(checkpoint.get('last_epsilon_decay_step', 0))
                    # Reward histories (bounded by deque maxlen)
                    try:
                        if hasattr(metrics, 'episode_rewards'):
                            metrics.episode_rewards.clear()
                            for v in (checkpoint.get('episode_rewards', []) or [])[-metrics.episode_rewards.maxlen:]:
                                metrics.episode_rewards.append(float(v))
                        if hasattr(metrics, 'dqn_rewards'):
                            metrics.dqn_rewards.clear()
                            for v in (checkpoint.get('dqn_rewards', []) or [])[-metrics.dqn_rewards.maxlen:]:
                                metrics.dqn_rewards.append(float(v))
                        if hasattr(metrics, 'expert_rewards'):
                            metrics.expert_rewards.clear()
                            for v in (checkpoint.get('expert_rewards', []) or [])[-metrics.expert_rewards.maxlen:]:
                                metrics.expert_rewards.append(float(v))
                    except Exception:
                        pass

                    # Respect server toggles that may force resets regardless of checkpoint
                    if getattr(SERVER_CONFIG, 'reset_expert_ratio', False):
                        print(f"Resetting expert_ratio to start per RL_CONFIG: {RL_CONFIG.expert_ratio_start}")
                        metrics.expert_ratio = RL_CONFIG.expert_ratio_start
                    if getattr(SERVER_CONFIG, 'reset_epsilon', False):
                        print(f"Resetting epsilon to start per SERVER_CONFIG: {RL_CONFIG.epsilon_start}")
                        metrics.epsilon = RL_CONFIG.epsilon_start
                        # Continue decays from the current frame interval so we don't retroactively overshoot
                        try:
                            metrics.last_epsilon_decay_step = int(metrics.frame_count // RL_CONFIG.epsilon_decay_steps)
                        except Exception:
                            metrics.last_epsilon_decay_step = 0
                    if getattr(SERVER_CONFIG, 'force_expert_ratio_recalc', False):
                        print(f"Force recalculating expert_ratio based on {metrics.frame_count:,} frames...")
                        decay_expert_ratio(metrics.frame_count)
                        print(f"Expert ratio recalculated to: {metrics.expert_ratio:.4f}")
                    if getattr(SERVER_CONFIG, 'reset_frame_count', False):
                        print("Resetting frame count per SERVER_CONFIG")
                        metrics.frame_count = 0
                        metrics.loaded_frame_count = 0
                        metrics.last_decay_step = 0
                        metrics.last_epsilon_decay_step = 0

                print(f"Loaded hybrid model from {filepath}")
                print(f"  - Resuming from frame: {metrics.frame_count}")
                print(f"  - Resuming epsilon: {metrics.epsilon:.4f}")
                print(f"  - Resuming expert_ratio: {metrics.expert_ratio:.4f}")
                print(f"  - Resuming last_decay_step: {metrics.last_decay_step}")
                return True
            except Exception as e:
                print(f"Error loading hybrid checkpoint: {e}")
                return False
        else:
            print(f"No checkpoint found at {filepath}. Starting new hybrid model.")
            return False

    def stop(self, join: bool, timeout: float):
        """Gracefully stop background threads."""
        self.running = False
        # Unblock workers by enqueueing no-op tokens
        try:
            for _ in range(max(1, self.num_training_workers)):
                self.train_queue.put_nowait(True)
        except Exception:
            pass
        if join:
            for t in self.training_threads:
                try:
                    t.join(timeout=timeout)
                except Exception:
                    pass

    def get_q_value_range(self):
        """Return min/max Q-values from the discrete head over a representative batch.

        We prefer sampling the replay buffer; if insufficient data, we probe a zero state.
        """
        try:
            with torch.no_grad():
                min_q = max_q = None

                buf_len = 0
                try:
                    buf_len = len(self.memory)
                except Exception:
                    buf_len = 0

                probe_bs = 256
                min_required = max(64, min(probe_bs, getattr(RL_CONFIG, 'batch_size', 64)))

                if buf_len >= min_required:
                    sample = self.memory.sample(min(probe_bs, buf_len))
                    if sample is not None:
                        states = sample[0]
                        if isinstance(states, torch.Tensor):
                            states = states.to(self.device, non_blocking=True)
                            discrete_q, _ = self.qnetwork_inference(states)
                            min_q = discrete_q.min().item()
                            max_q = discrete_q.max().item()

                if min_q is None or max_q is None:
                    dummy = torch.zeros(1, self.state_size, device=self.device)
                    discrete_q, _ = self.qnetwork_inference(dummy)
                    min_q = discrete_q.min().item()
                    max_q = discrete_q.max().item()

                return float(min_q), float(max_q)
        except Exception:
            return float('nan'), float('nan')




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

        # DIAGNOSTIC: Check for negative subjective rewards before scaling
        if subjreward < 0:
            print(f"[DIAGNOSTIC] Frame {frame_counter}: Negative subjective reward received from Lua: {subjreward:.6f} (objective={objreward:.6f})", flush=True)

        # Apply subjective/objective reward scaling and compute total
        subjreward = float(subjreward) * float(getattr(RL_CONFIG, 'subj_reward_scale', 1.0) or 1.0)
        objreward = float(objreward) * float(getattr(RL_CONFIG, 'reward_scale', 1.0) or 1.0)
        reward = float(subjreward + objreward)
        
        # DIAGNOSTIC: Also check after scaling
        if subjreward < 0:
            print(f"[DIAGNOSTIC] Frame {frame_counter}: Negative subjective reward AFTER scaling: {subjreward:.6f} (scale={getattr(RL_CONFIG, 'subj_reward_scale', 1.0)}, total reward={reward:.6f})", flush=True)

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

# Rolling window for DQN reward over last 1M frames
DQN1M_WINDOW_FRAMES = 1_000_000
_dqn1m_window = deque()  # entries: (frames_in_interval: int, dqn_reward_mean: float, frame_end: int)
_dqn1m_window_frames = 0
_last_frame_count_seen_1m = None

def _update_dqn1m_window(mean_dqn_reward: float):
    """Update the 1M-frames rolling window with the latest interval.

    Uses the number of frames progressed since the last row as the weight.
    """
    global _dqn1m_window_frames, _last_frame_count_seen_1m
    current_frame = metrics.frame_count
    # Determine frames elapsed since last sample
    if _last_frame_count_seen_1m is None:
        delta_frames = 0
    else:
        delta_frames = max(0, current_frame - _last_frame_count_seen_1m)
    _last_frame_count_seen_1m = current_frame

    # If no frame progress (e.g., first row), just return without adding
    if delta_frames <= 0:
        return

    # Append new interval
    _dqn1m_window.append((delta_frames, float(mean_dqn_reward), int(current_frame)))
    _dqn1m_window_frames += delta_frames

    # Trim window to last 1M frames (may need partial trim of the oldest bucket)
    while _dqn1m_window and _dqn1m_window_frames > DQN1M_WINDOW_FRAMES:
        overflow = _dqn1m_window_frames - DQN1M_WINDOW_FRAMES
        oldest_frames, oldest_val, oldest_end = _dqn1m_window[0]
        if oldest_frames <= overflow:
            _dqn1m_window.popleft()
            _dqn1m_window_frames -= oldest_frames
        else:
            # Partially trim the oldest bucket
            kept_frames = oldest_frames - overflow
            _dqn1m_window[0] = (kept_frames, oldest_val, oldest_end)
            _dqn1m_window_frames = DQN1M_WINDOW_FRAMES
            break

def _compute_dqn1m_window_stats():
    """Compute weighted average for the 1M-frame window.

    Returns avg.
    """
    if not _dqn1m_window or _dqn1m_window_frames <= 0:
        return 0.0

    # Weighted average
    w_sum = float(_dqn1m_window_frames)
    wy_sum = sum(fr * val for fr, val, _ in _dqn1m_window)
    avg = wy_sum / w_sum if w_sum > 0 else 0.0
    return avg

def display_metrics_header(kb=None):
    """Display header for metrics table"""
    if not IS_INTERACTIVE: return
    header = (
        f"{'Frame':>8} | {'FPS':>5} | {'Clients':>7} | {'Rwrd':>6} | {'Subj':>6} | {'Obj':>6} | {'DQN':>6} | {'DQN1M':>6} | {'Loss':>8} | "
        f"{'Epsilon':>7} | {'Xprt':>6} | {'Mem Size':>8} | {'Avg Level':>9} | {'Level Type':>10} | {'OVR':>3} | {'Expert':>6} | "
        f"{'Q-Value Range':>14}"
    )
    print_with_terminal_restore(kb, f"\n{'-' * len(header)}")
    print_with_terminal_restore(kb, header)
    print_with_terminal_restore(kb, f"{'-' * len(header)}")

def display_metrics_row(agent, kb=None):
    """Display current metrics in tabular format"""
    if not IS_INTERACTIVE: return
    mean_reward = np.mean(list(metrics.episode_rewards)) if metrics.episode_rewards else float('nan')
    mean_subj_reward = np.mean(list(metrics.subj_rewards)) if metrics.subj_rewards else float('nan')
    mean_obj_reward = np.mean(list(metrics.obj_rewards)) if metrics.obj_rewards else float('nan')
    dqn_reward = np.mean(list(metrics.dqn_rewards)) if metrics.dqn_rewards else float('nan')
    mean_loss = np.mean(list(metrics.losses)) if metrics.losses else float('nan')
    guided_ratio = metrics.expert_ratio
    mem_size = len(agent.memory) if agent else 0
    
    # Update DQN1M window and compute stats
    _update_dqn1m_window(dqn_reward)
    dqn1m_avg = _compute_dqn1m_window_stats()
    
    # Get client count from server if available
    client_count = 0
    if hasattr(metrics, 'global_server') and metrics.global_server:
        with metrics.global_server.client_lock:
            client_count = len(metrics.global_server.clients)
            # Update the metrics.client_count value
            metrics.client_count = client_count
    
    # Determine override status
    # Display simple ON/OFF for override in its own column; Expert has its own ON/OFF column
    override_status = "ON" if metrics.override_expert else "OFF"
    
    # Display average level as 1-based instead of 0-based
    display_level = metrics.average_level + 1.0
    
    # Get Q-value range from the agent
    q_range = "N/A"
    if agent:
        try:
            min_q, max_q = agent.get_q_value_range()
            if not (np.isnan(min_q) or np.isnan(max_q)):
                q_range = f"[{min_q:.2f}, {max_q:.2f}]"
        except Exception:
            q_range = "Error"
    
    row = (
        f"{metrics.frame_count:8d} | {metrics.fps:5.1f} | {client_count:>7} | {mean_reward:6.2f} | {mean_subj_reward:6.2f} | {mean_obj_reward:6.2f} | {dqn_reward:6.2f} | {dqn1m_avg:6.2f} | {mean_loss:8.2f} | "
        f"{metrics.epsilon:7.3f} | {guided_ratio*100:6.1f} | {mem_size:8d} | {display_level:9.2f} | {'Open' if metrics.open_level else 'Closed':10} | {override_status:>3} | {'ON' if metrics.expert_mode else 'OFF':>6} | "
        f"{q_range:>14}"
    )
    print_with_terminal_restore(kb, row)

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

def fire_zap_to_discrete(fire, zap):
    """Convert fire/zap booleans to discrete action index (0-3)"""
    return int(fire) * 2 + int(zap)

def discrete_to_fire_zap(discrete_action):
    """Convert discrete action index (0-3) to fire/zap booleans"""
    discrete_action = int(discrete_action)
    fire = (discrete_action >> 1) & 1  # Extract fire bit
    zap = discrete_action & 1         # Extract zap bit
    return bool(fire), bool(zap)

def get_expert_hybrid_action(enemy_seg, player_seg, is_open_level, expert_fire=False, expert_zap=False):
    """Get expert action in hybrid format (discrete_action, continuous_spinner)
    
    Returns:
        discrete_action: int (0-3) for fire/zap combination
        continuous_spinner: float in [-0.9, +0.9] for movement (full expert range)
    """
    # Get continuous expert action
    fire, zap, spinner = get_expert_action(enemy_seg, player_seg, is_open_level, expert_fire, expert_zap)
    
    # Convert to hybrid format - preserve full expert system range!
    discrete_action = fire_zap_to_discrete(fire, zap)
    continuous_spinner = float(spinner)  # No clamping - use expert's full range
    
    return discrete_action, continuous_spinner

def hybrid_to_game_action(discrete_action, continuous_spinner):
    """Convert hybrid action to game format
    
    Args:
        discrete_action: int (0-3) for fire/zap combination
        continuous_spinner: float in [-0.3, +0.3] for movement
        
    Returns:
        fire_cmd: int (0 or 1)
        zap_cmd: int (0 or 1) 
        spinner_cmd: int (-9 to +9, scaled from spinner * 31)
    """
    fire, zap = discrete_to_fire_zap(discrete_action)
    return encode_action_to_game(fire, zap, continuous_spinner)

def legacy_action_to_hybrid(action_idx):
    """Legacy path removed. Placeholder returns no-op hybrid action."""
    return 0, 0.0

def hybrid_to_legacy_action(discrete_action, continuous_spinner):
    """Legacy path removed. Placeholder returns center legacy index (unused)."""
    return 3

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
    
    def add_episode_reward(self, total_reward, dqn_reward, expert_reward, subj_reward=None, obj_reward=None):
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
                    add_fn(float(total_reward), float(dqn_reward), float(expert_reward), subj_reward, obj_reward)
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
