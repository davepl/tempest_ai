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
from collections import deque
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
        
        #BUGBUG why fc1 and fc2?  What's the difference and why are there both?
        # Continuous head for spinner (always separate from dueling)
        self.continuous_fc1 = nn.Linear(shared_output_size, head_size)
        continuous_head_size = max(32, head_size // 2)
        self.continuous_fc2 = nn.Linear(head_size, continuous_head_size)
        self.continuous_out = nn.Linear(continuous_head_size, 1)
        
        # Initialize discrete head with balanced initialization to prevent action bias
        # CRITICAL: Without this, default initialization creates strong bias (e.g., 93% action 3)
        # Strategy: Use small random weights to allow gradient flow while maintaining initial balance
        torch.nn.init.xavier_uniform_(self.discrete_fc.weight, gain=1.0)
        torch.nn.init.constant_(self.discrete_fc.bias, 0.0)
        # Output layer: Small random initialization for gradient flow with minimal bias
        torch.nn.init.uniform_(self.discrete_out.weight, -0.003, 0.003)  # Small random weights for gradient flow
        torch.nn.init.constant_(self.discrete_out.bias, 0.0)              # Zero bias for balance
        
        # Initialize continuous head with smaller weights for stable training
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
    """Experience replay buffer with N-bucket stratified storage for hybrid discrete-continuous actions.
    
    Stores experiences as: (state, discrete_action, continuous_action, reward, next_state, done, actor, horizon)
    - discrete_action: integer index for fire/zap combination (0-3)
    - continuous_action: float spinner value in [-0.9, +0.9]
    - actor: string tag identifying source of experience ('expert' or 'dqn')
    
    Uses N-bucket stratified sampling based on TD-error percentiles:
    - N priority buckets (default 5): Split top 50% of TD errors into deciles
      (e.g., 90-99%, 80-89%, 70-79%, 60-69%, 50-59%)
    - Each priority bucket: 250K capacity (configurable)
    - Main bucket: 1M capacity for bottom 50% of TD errors
    
    This provides PER-like benefits (oversampling high TD-error experiences)
    with O(1) performance (no tree structures, just percentile thresholds and ring buffers).
    """
    def __init__(self, capacity: int, state_size: int):
        from config import RL_CONFIG
        
        self.state_size = int(state_size)
        
        # N-bucket configuration
        self.n_buckets = getattr(RL_CONFIG, 'replay_n_buckets', 3)
        bucket_size = getattr(RL_CONFIG, 'replay_bucket_size', 250000)
        main_bucket_size = getattr(RL_CONFIG, 'replay_main_bucket_size', 1500000)
        
        # Initialize buckets: N priority buckets + 1 main bucket
        # Ultra-focused strategy: Priority buckets cover 90-100th percentile (top 10%)
        # Using power-law spacing: 98-100%, 95-98%, 90-95%
        # Main bucket holds everything below 90th percentile
        self.buckets = []
        
        # Define percentile ranges for priority buckets
        # Top 35% of experiences distributed across priority buckets
        # This dedicates 33% of capacity (3x250k) to top 35% of TD-errors
        if self.n_buckets == 3:
            # Top 35% split into 3 buckets: [95-100%, 85-95%, 75-85%], remaining 65% in main
            percentile_ranges = [(95, 100), (85, 95), (75, 85)]
            main_percentile_high = 75
        elif self.n_buckets == 4:
            # Top 35% split into 4 buckets
            percentile_ranges = [(95, 100), (85, 95), (75, 85), (65, 75)]
            main_percentile_high = 65
        elif self.n_buckets == 5:
            # Top 35% split into 5 buckets
            percentile_ranges = [(95, 100), (88, 95), (81, 88), (74, 81), (67, 74)]
            main_percentile_high = 67
        else:
            # Fallback to old linear spacing for other values
            percentile_ranges = [(100 - (i+1)*10, 100 - i*10) for i in range(self.n_buckets)]
            main_percentile_high = 100 - self.n_buckets * 10
        
        # Create priority buckets with defined ranges
        for percentile_low, percentile_high in percentile_ranges:
            self.buckets.append({
                'name': f'p{percentile_low}-{percentile_high}',
                'percentile_low': percentile_low,
                'percentile_high': percentile_high,
                'capacity': bucket_size,
                'position': 0,
                'size': 0
            })
        
        # Add main bucket for experiences below main_percentile_high
        self.buckets.append({
            'name': 'main',
            'percentile_low': 0,
            'percentile_high': main_percentile_high,
            'capacity': main_bucket_size,
            'position': 0,
            'size': 0
        })
        
        # Calculate total capacity
        self.capacity = sum(b['capacity'] for b in self.buckets)
        self.size = 0
        
        # Allocate storage arrays across all buckets (contiguous for uniform sampling)
        self.states = np.zeros((self.capacity, self.state_size), dtype=np.float32)
        self.discrete_actions = np.zeros((self.capacity,), dtype=np.int32)
        self.continuous_actions = np.zeros((self.capacity,), dtype=np.float32)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, self.state_size), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.bool_)
        self.actors = np.full((self.capacity,), '', dtype='U10')
        self.horizons = np.ones((self.capacity,), dtype=np.int32)
        
        # Assign storage offset for each bucket (contiguous layout)
        offset = 0
        for bucket in self.buckets:
            bucket['offset'] = offset
            offset += bucket['capacity']
        
        # Rolling priority metric window for percentile threshold calculation
        # Currently using reward magnitude as a fast proxy for experience importance
        self.priority_metric_window = deque(maxlen=50000)
        
        # Initialize thresholds with reasonable defaults based on expected reward range
        # Use conservative initial values that will be updated as data arrives
        # Assume rewards typically range from 0 to ~10 for Tempest gameplay
        if self.n_buckets == 3:
            # Target percentiles: 95th, 85th, 75th for top 35% distribution
            # Initial guesses: higher values for more selective priority buckets
            self.percentile_thresholds = [7.0, 5.0, 3.0]
        elif self.n_buckets == 4:
            # Target percentiles: 95th, 85th, 75th, 65th
            self.percentile_thresholds = [7.0, 5.0, 3.5, 2.0]
        elif self.n_buckets == 5:
            # Target percentiles: 95th, 88th, 81st, 74th, 67th
            self.percentile_thresholds = [7.0, 5.5, 4.0, 3.0, 2.0]
        else:
            # Fallback: evenly spaced from 1.0 to n_buckets
            self.percentile_thresholds = [float(self.n_buckets - i) for i in range(self.n_buckets)]
        
        self.threshold_update_counter = 0
        
        # Pre-death frame tracking (terminal states)
        self.terminal_indices = deque(maxlen=10000)
        self.pre_death_flags = np.zeros((self.capacity,), dtype=np.bool_)
        
        # Random number generator
        self._rand = np.random.default_rng()

    def push(self, state, discrete_action, continuous_action, reward, next_state, done, actor, horizon: int, td_error: float = 0.0):
        """Add experience to buffer with N-bucket stratified storage based on priority metric.
        
        Args:
            td_error: Priority metric for this experience (currently reward magnitude, used for bucket classification)
        """
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
        
        # Update priority metric window for threshold calculation
        abs_priority_metric = abs(float(td_error))
        self.priority_metric_window.append(abs_priority_metric)
        self.threshold_update_counter += 1
        
        # Update percentile thresholds periodically
        # Start with frequent updates (every 100 samples) to quickly adapt from initial defaults
        # After 10k samples, reduce frequency to every 1000 samples for efficiency
        update_interval = 100 if len(self.priority_metric_window) < 10000 else 1000
        if self.threshold_update_counter >= update_interval and len(self.priority_metric_window) >= update_interval:
            self._update_percentile_thresholds()
            self.threshold_update_counter = 0
        
        # Determine which bucket to write to based on priority metric
        bucket_idx = self._get_bucket_index(abs_priority_metric)
        bucket = self.buckets[bucket_idx]
        
        # Calculate absolute index in storage arrays
        idx = bucket['offset'] + bucket['position']
        
        # Update bucket ring buffer position
        bucket['position'] = (bucket['position'] + 1) % bucket['capacity']
        if bucket['size'] < bucket['capacity']:
            bucket['size'] += 1
        
        # Update total size
        self.size = sum(b['size'] for b in self.buckets)

        # Store experience at determined index
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
            self.states[idx, :] = s
        except Exception:
            self.states[idx, :] = 0.0
        
        self.discrete_actions[idx] = discrete_idx
        self.continuous_actions[idx] = continuous_val
        self.rewards[idx] = reward
        
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
            self.next_states[idx, :] = ns
        except Exception:
            self.next_states[idx, :] = 0.0
        
        self.dones[idx] = done
        self.actors[idx] = actor_tag
        
        # Require explicit positive horizon
        h = int(horizon)
        if h < 1:
            raise ValueError("HybridReplayBuffer.push requires horizon >= 1")
        self.horizons[idx] = h

        # Track terminal indices for pre-death sampling
        if done:
            self.terminal_indices.append(idx)
            # Mark frames 5-10 steps back as pre-death
            from config import RL_CONFIG
            for lookback in range(RL_CONFIG.replay_terminal_lookback_min, RL_CONFIG.replay_terminal_lookback_max + 1):
                pre_death_idx = idx - lookback
                if pre_death_idx >= 0:
                    self.pre_death_flags[pre_death_idx] = True
                elif self.size >= self.capacity:
                    pre_death_idx = pre_death_idx % self.capacity
                    self.pre_death_flags[pre_death_idx] = True
    
    def _get_bucket_index(self, priority_metric: float) -> int:
        """Classify priority metric (reward magnitude) into appropriate bucket.
        
        Priority buckets cover high percentiles (e.g., 95-100th, 85-95th, 75-85th for N=3).
        Main bucket holds everything below the main threshold (e.g., <75th percentile).
        """
        # Check against thresholds from highest to lowest
        for i, threshold in enumerate(self.percentile_thresholds):
            if priority_metric >= threshold:
                return i
        
        # If below all thresholds, goes to main bucket (last bucket)
        return len(self.buckets) - 1
    
    def _update_percentile_thresholds(self):
        """Update percentile thresholds from rolling priority metric window.
        
        Calculates thresholds based on configured bucket ranges.
        Currently using reward magnitude as priority metric proxy.
        For N=3: 95th, 85th, 75th percentiles (top 35% split into 3 buckets, bottom 65% in main)
        """
        if len(self.priority_metric_window) < 100:
            return
        
        errors = np.array(self.priority_metric_window)
        
        # Calculate percentiles based on bucket configuration
        # Goal: Distribute top 35% of priority metrics (rewards) across N priority buckets
        # Main bucket gets bottom 65% (for N=3) or 35% (for N=4)
        if self.n_buckets == 3:
            # Top 35% split into 3 buckets: 95-100%, 85-95%, 75-85%
            # Thresholds at 95th, 85th, 75th percentiles
            percentiles = [95, 85, 75]
        elif self.n_buckets == 4:
            # Top 35% split into 4 buckets: 95-100%, 85-95%, 75-85%, 65-75%
            percentiles = [95, 85, 75, 65]
        elif self.n_buckets == 5:
            # Top 35% split into 5 buckets
            percentiles = [95, 88, 81, 74, 67]
        else:
            # Fallback to evenly-spaced logic
            percentiles = [100 - ((i + 1) * 10) for i in range(self.n_buckets)]
        
        self.percentile_thresholds = [np.percentile(errors, p) for p in percentiles]
    
    def sample(self, batch_size):
        """Sample batch uniformly from all buckets using stratified sampling.
        
        Instead of building a list of all valid indices (slow for large buffers),
        we sample proportionally from each bucket based on its fill ratio.
        This is O(n_buckets) instead of O(buffer_size).
        """
        if self.size < batch_size:
            return None
        
        # Calculate how many samples to draw from each bucket (proportional to bucket size)
        samples_per_bucket = []
        total_allocated = 0
        
        for i, bucket in enumerate(self.buckets):
            if bucket['size'] == 0:
                samples_per_bucket.append(0)
            else:
                # Allocate samples proportionally to bucket's contribution to total size
                proportion = bucket['size'] / self.size
                n_samples = int(batch_size * proportion)
                
                # Last bucket gets remainder to ensure we hit batch_size exactly
                if i == len(self.buckets) - 1:
                    n_samples = batch_size - total_allocated
                
                samples_per_bucket.append(n_samples)
                total_allocated += n_samples
        
        # Sample from each bucket
        all_indices = []
        for bucket, n_samples in zip(self.buckets, samples_per_bucket):
            if n_samples > 0 and bucket['size'] > 0:
                # Sample uniformly from this bucket's valid range
                # Valid indices are [offset, offset + size)
                bucket_indices = self._rand.integers(
                    bucket['offset'], 
                    bucket['offset'] + bucket['size'],
                    size=n_samples,
                    dtype=np.int64
                )
                all_indices.append(bucket_indices)
        
        # Concatenate all sampled indices
        if not all_indices:
            return None
        
        indices = np.concatenate(all_indices)

        # Vectorized gather for batch data
        states_np = self.states[indices]
        next_states_np = self.next_states[indices]
        batch_discrete_actions = self.discrete_actions[indices]
        batch_continuous_actions = self.continuous_actions[indices]
        batch_rewards = self.rewards[indices]
        batch_dones = self.dones[indices]
        batch_actors = self.actors[indices]
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
    
    def get_partition_stats(self):
        """Return statistics about N-bucket stratified buffer for monitoring."""
        stats = {
            'total_size': self.size,
            'total_capacity': self.capacity,
        }
        
        # Add stats for each bucket
        for bucket in self.buckets:
            prefix = bucket['name'].replace('-', '_')
            stats[f'{prefix}_size'] = bucket['size']
            stats[f'{prefix}_capacity'] = bucket['capacity']
            stats[f'{prefix}_fill_pct'] = (bucket['size'] / bucket['capacity'] * 100) if bucket['capacity'] > 0 else 0.0
        
        # Add TD-error threshold information
        # Map threshold indices to actual percentiles based on configuration
        if self.n_buckets == 3:
            percentile_labels = [95, 85, 75]
        elif self.n_buckets == 4:
            percentile_labels = [95, 85, 75, 65]
        elif self.n_buckets == 5:
            percentile_labels = [95, 88, 81, 74, 67]
        else:
            # Fallback to old evenly-spaced logic
            percentile_labels = [100 - ((i + 1) * 10) for i in range(self.n_buckets)]
        
        for i, threshold in enumerate(self.percentile_thresholds):
            if i < len(percentile_labels):
                stats[f'threshold_p{percentile_labels[i]}'] = threshold
        
        return stats
    
    def get_actor_composition(self):
        """Return statistics on actor composition of buffer"""
        if self.size == 0:
            return {'total': 0, 'dqn': 0, 'expert': 0, 'frac_dqn': 0.0, 'frac_expert': 0.0}

        total = 0
        n_dqn = 0
        n_expert = 0

        for bucket in self.buckets:
            bucket_size = bucket['size']
            if bucket_size <= 0:
                continue

            start = bucket['offset']
            if bucket_size < bucket['capacity']:
                end = start + bucket_size
            else:
                end = start + bucket['capacity']

            actors_slice = self.actors[start:end]
            n_dqn += int(np.count_nonzero(actors_slice == 'dqn'))
            n_expert += int(np.count_nonzero(actors_slice == 'expert'))
            total += bucket_size

        if total <= 0:
            return {'total': 0, 'dqn': 0, 'expert': 0, 'frac_dqn': 0.0, 'frac_expert': 0.0}

        frac_dqn = float(n_dqn) / float(total)
        frac_expert = float(n_expert) / float(total)

        return {
            'total': int(total),
            'dqn': int(n_dqn),
            'expert': int(n_expert),
            'frac_dqn': frac_dqn,
            'frac_expert': frac_expert,
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
    os.makedirs(MODEL_DIR, exist_ok=True)

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
        # - inference: MUST be in eval mode for deterministic behavior
        self.qnetwork_local.train()
        self.qnetwork_target.eval()
        # CRITICAL FIX: Even though qnetwork_inference IS qnetwork_local,
        # we need to track that it should be in eval mode during inference.
        # The act() method will handle mode switching.

        # Defensive: unwrap any lingering compiled wrappers & reset dynamo once.
        try:
            before_types = (self.qnetwork_local.__class__.__name__, self.qnetwork_target.__class__.__name__)
            self.qnetwork_local = unwrap_compiled_module(self.qnetwork_local)
            self.qnetwork_target = unwrap_compiled_module(self.qnetwork_target)
            self.qnetwork_inference = unwrap_compiled_module(self.qnetwork_inference)
            after_types = (self.qnetwork_local.__class__.__name__, self.qnetwork_target.__class__.__name__)
            if before_types != after_types:
                pass
            _force_dynamo_reset_once()
        except Exception:
            pass
        
        # Store device reference
        self.device = device
        
        # Optimizer with separate parameter groups for discrete and continuous components
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)
        
        # Experience replay (simplified to uniform sampling only)
        self.memory = HybridReplayBuffer(memory_size, state_size=self.state_size)

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
        """Select hybrid action using epsilon-greedy for discrete + Gaussian noise for cont7inuous
        
        Returns:
            discrete_action: int (0-3) for fire/zap combination
            continuous_action: float in [-0.9, +0.9] for spinner
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        # CRITICAL FIX: Put network in eval mode for deterministic inference
        self.qnetwork_inference.eval()
        with torch.no_grad():
            discrete_q, continuous_pred = self.qnetwork_inference(state)
        # Restore train mode for training
        self.qnetwork_inference.train()
        
        # Discrete action selection via epsilon-greedy
        if random.random() < epsilon:
            discrete_action = random.randint(0, self.discrete_actions - 1)
        else:
            discrete_action = discrete_q.argmax(dim=1).item()  # Fixed: specify dimension
        
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
        
        # Use reward magnitude as a fast priority metric proxy instead of expensive forward passes
        # This is orders of magnitude faster than computing actual TD-error
        # High rewards indicate important transitions (enemy kills, level completion)
        # Actual TD-errors can be computed during sampling/training if needed
        td_error = abs(float(reward))
        
        self.memory.push(
            state,
            discrete_action,
            continuous_action,
            reward,
            next_state,
            done,
            actor=actor,
            horizon=horizon,
            td_error=td_error,
        )
        
        # Queue training steps only if training is enabled
        # NOTE: Experiences are ALWAYS collected in the replay buffer regardless of training state
        if not getattr(metrics, 'training_enabled', True) or not self.training_enabled:
            return  # Skip queueing training, but buffer is already updated above
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
                        loss_val = train_step(self)
                        did_train = (loss_val is not None)
                        # Count only real optimizer updates, not skipped/no-op passes
                        if did_train:
                            try:
                                metrics.training_steps_interval += 1
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

    def save(self, filepath, now=None, is_forced_save=False):
        """Save hybrid model checkpoint with training state."""
        if now is None:
            import time
            now = time.time()
        
        # Update target network before saving
        self.update_target_network()
        
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

    def update_target_network(self):
        """Hard update target network from local and record telemetry."""
        try:
            self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
            metrics.last_target_update_time = time.time()
            metrics.last_hard_target_update_frame = metrics.frame_count
            metrics.last_hard_target_update_time = time.time()
            metrics.last_target_update_step = self.training_steps
        except Exception:
            pass
    
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

                # Sync metrics counters with loaded training_steps
                try:
                    metrics.total_training_steps = self.training_steps
                    metrics.last_target_update_step = self.training_steps
                    metrics.last_target_update_frame = metrics.frame_count
                except Exception:
                    pass

                # No separate inference network to sync in single device setup

                # Sanity-check Q-values on load to catch corruption/explosion or collapse
                # DISABLED: This rescaling was causing catastrophic Q-value resets and DLoss spikes
                # The logic would trigger on perfectly healthy Q-values (e.g., [-10, -0.4])
                # and scale them down to [0.03, 0.08], creating massive TD errors when
                # fresh replay buffer data arrives with rewards of 5-7.
                # Better to let Q-values naturally stabilize through training.
                try:
                    with torch.no_grad():
                        dummy = torch.zeros(1, self.state_size, device=self.device)
                        dq, _ = self.qnetwork_local(dummy)
                        qmax = float(dq.max().item())
                        qmin = float(dq.min().item())
                        q_range = max(abs(qmax), abs(qmin))
                        
                        # Just log Q-values for monitoring, but don't rescale
                        print(f"Loaded model Q-value range: [{qmin:.3f}, {qmax:.3f}]")
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
                
                # CRITICAL FIX: Handle replay buffer reset
                try:
                    from config import RESET_REPLAY_BUFFER
                except ImportError:
                    from Scripts.config import RESET_REPLAY_BUFFER
                
                if RESET_REPLAY_BUFFER:
                    # Clear the replay buffer (silent operation)
                    self.memory.size = 0
                    self.memory.high_reward_size = 0
                    self.memory.regular_size = 0
                    self.memory.high_reward_position = 0
                    self.memory.regular_position = 0
                    # Clear any cached indices
                    try:
                        self.memory.terminal_indices.clear()
                    except Exception:
                        pass
                    # Reset target network to match local network (prevent Q-explosion from stale targets)
                    self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
                
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
