#!/usr/bin/env python3
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
# Removed StepLR and AMP imports (hybrid-only path doesn't use them)
import select
import threading
import queue
from collections import deque, namedtuple
from datetime import datetime
# Removed stable_baselines3 imports (no longer used after refactor)
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
from config import (
    SERVER_CONFIG,
    RL_CONFIG,
    MODEL_DIR,
    LATEST_MODEL_PATH,
    metrics as config_metrics,
    ServerConfigData,
    RLConfigData,
    RESET_METRICS
)

# Suppress warnings
warnings.filterwarnings('ignore')

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
    reward: float
    action: Tuple[bool, bool, float]  # fire, zap, spinner
    mode: int
    gamestate: int    # Added: Game state value from Lua
    done: bool
    attract: bool
    save_signal: bool
    enemy_seg: int
    player_seg: int
    open_level: bool
    expert_fire: bool  # Added: Expert system fire recommendation
    expert_zap: bool   # Added: Expert system zap recommendation
    level_number: int  # Added: Current level number from Lua
    player_alive: bool  # Added: Player alive flag (from $0201 high bit inverted)
    death_reason: int   # Added: Death reason signed byte from $013B (-1, 7, 9, etc.)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FrameData':
        """Create FrameData from dictionary"""
        return cls(
            state=data["state"],
            reward=data["reward"],
            action=data["action"],
            mode=data["mode"],
            gamestate=data.get("gamestate", 0),  # Default to 0 if not provided
            done=data["done"],
            attract=data["attract"],
            save_signal=data["save_signal"],
            enemy_seg=data["enemy_seg"],
            player_seg=data["player_seg"],
            open_level=data["open_level"],
            expert_fire=data["expert_fire"],
            expert_zap=data["expert_zap"],
            level_number=data.get("level_number", 0),  # Default to 0 if not provided
            player_alive=bool(data.get("player_alive", True)),
            death_reason=int(data.get("death_reason", 0))
        )

# Configuration constants
SERVER_CONFIG = server_config
RL_CONFIG = rl_config

# Initialize devices (dual GPU setup: inference on GPU1, training on GPU0)
if torch.cuda.is_available():
    # Check if both GPUs are available for dual GPU setup
    if torch.cuda.device_count() >= 2:
        training_device = torch.device("cuda:0")
        inference_device = torch.device("cuda:1")
        print(f"Using dual GPU setup: Training on {training_device}, Inference on {inference_device}")
        print(f"Available GPUs: {torch.cuda.device_count()}")
        print(f"GPU 0: {torch.cuda.get_device_name(0)}")
        print(f"GPU 1: {torch.cuda.get_device_name(1)}")
    else:
        # Fall back to single GPU if only one is available
        training_device = torch.device("cuda:0")
        inference_device = torch.device("cuda:0")
        print(f"Only {torch.cuda.device_count()} GPU(s) available, using single GPU setup: {training_device}")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    training_device = torch.device("mps")
    inference_device = torch.device("mps")
    print(f"Using MPS for both inference and training: {training_device}")
else:
    training_device = torch.device("cpu")
    inference_device = torch.device("cpu")
    print(f"Using CPU for both inference and training: {training_device}")

# Low-risk math speedups (CUDA only): allow TF32 and tune matmul/cudnn
try:
    if training_device.type == 'cuda':
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

print(f"Dueling: {'enabled' if getattr(RL_CONFIG, 'use_dueling', False) else 'disabled' } ")
print("Replay: hybrid experience buffer (discrete + continuous)")
print(f"Mixed precision: {'enabled' if getattr(RL_CONFIG, 'use_mixed_precision', False) else 'disabled'}")
print(f"State size: {RL_CONFIG.state_size}")

# For compatibility with dual-device code
device = training_device  # Legacy compatibility

# Initialize metrics
metrics = config_metrics

# Global reference to server for metrics display
metrics.global_server = None

# Legacy discrete-only replay types removed; hybrid-only

# Discrete-only QNetwork removed (hybrid-only)

class NoisyLinear(nn.Module):
    """Factorized Gaussian NoisyNet layer (adds noise only in training mode)."""
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Buffers for noise
        self.register_buffer('weight_eps', torch.empty(out_features, in_features))
        self.register_buffer('bias_eps', torch.empty(out_features))

        self.reset_parameters(sigma_init)
        self.reset_noise()

    def reset_parameters(self, sigma_init: float):
        mu_range = 1 / np.sqrt(self.in_features)
        with torch.no_grad():
            self.weight_mu.uniform_(-mu_range, mu_range)
            self.bias_mu.uniform_(-mu_range, mu_range)
            self.weight_sigma.fill_(sigma_init / np.sqrt(self.in_features))
            self.bias_sigma.fill_(sigma_init / np.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_eps.copy_(eps_out.ger(eps_in))
        self.bias_eps.copy_(eps_out)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Use current sampled noise; do not mutate buffers during forward
            weight = self.weight_mu + self.weight_sigma * self.weight_eps
            bias = self.bias_mu + self.bias_sigma * self.bias_eps
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(input, weight, bias)

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
                 hidden_size: int = 512, num_layers: int = 3, 
                 use_dueling: bool = False, use_noisy: bool = False, noisy_std: float = 0.1):
        super(HybridDQN, self).__init__()
        
        self.state_size = state_size
        self.discrete_actions = discrete_actions
        self.use_dueling = use_dueling
        self.use_noisy = use_noisy
        self.num_layers = num_layers
        
        # Shared trunk for feature extraction
        LinearOrNoisy = NoisyLinear if use_noisy else nn.Linear
        
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
        if use_noisy:
            self.shared_layers.append(LinearOrNoisy(state_size, layer_sizes[0], noisy_std))
        else:
            self.shared_layers.append(LinearOrNoisy(state_size, layer_sizes[0]))
        
        # Subsequent layers: hidden_size -> hidden_size/2 -> hidden_size/4 -> ...
        for i in range(1, num_layers):
            if use_noisy:
                self.shared_layers.append(LinearOrNoisy(layer_sizes[i-1], layer_sizes[i], noisy_std))
            else:
                self.shared_layers.append(LinearOrNoisy(layer_sizes[i-1], layer_sizes[i]))
        
        # Final layer size for heads
        shared_output_size = layer_sizes[-1]
        head_size = max(64, shared_output_size // 2)  # Head layer size
        
        if use_dueling:
            # Dueling architecture for discrete Q-values
            self.discrete_val_fc = nn.Linear(shared_output_size, head_size)
            self.discrete_adv_fc = nn.Linear(shared_output_size, head_size)
            self.discrete_val_out = nn.Linear(head_size, 1)  # State value
            self.discrete_adv_out = nn.Linear(head_size, discrete_actions)  # Advantages
        else:
            # Standard architecture for discrete Q-values
            self.discrete_fc = nn.Linear(shared_output_size, head_size)
            self.discrete_out = nn.Linear(head_size, discrete_actions)
        
        # Continuous head for spinner (always separate from dueling)
        self.continuous_fc1 = nn.Linear(shared_output_size, head_size)
        continuous_head_size = max(32, head_size // 2)
        self.continuous_fc2 = nn.Linear(head_size, continuous_head_size)
        self.continuous_out = nn.Linear(continuous_head_size, 1)
        
        # Initialize continuous head with smaller weights for stable training
        torch.nn.init.xavier_normal_(self.continuous_out.weight, gain=0.1)
        torch.nn.init.constant_(self.continuous_out.bias, 0.0)
    
    def reset_noise(self):
        """Reset noise layers if using noisy networks"""
        if self.use_noisy:
            for m in self.modules():
                if isinstance(m, NoisyLinear):
                    m.reset_noise()
    
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
        if self.use_dueling:
            # Dueling network: V(s) + A(s,a) - mean(A(s,·))
            discrete_val = F.relu(self.discrete_val_fc(shared))
            discrete_val = self.discrete_val_out(discrete_val)  # (B, 1)
            
            discrete_adv = F.relu(self.discrete_adv_fc(shared))
            discrete_adv = self.discrete_adv_out(discrete_adv)  # (B, discrete_actions)
            
            # Center advantages around their mean
            discrete_q = discrete_val + (discrete_adv - discrete_adv.mean(dim=1, keepdim=True))
        else:
            # Standard Q-network
            discrete = F.relu(self.discrete_fc(shared))
            discrete_q = self.discrete_out(discrete)  # (B, discrete_actions)
        
        # Continuous spinner head
        continuous = F.relu(self.continuous_fc1(shared))
        continuous = F.relu(self.continuous_fc2(continuous))
        continuous_raw = self.continuous_out(continuous)  # (B, 1)
        
        # Apply tanh to bound spinner to [-1, +1] then scale to [-0.9, +0.9]
        continuous_spinner = torch.tanh(continuous_raw) * 0.9
        
        return discrete_q, continuous_spinner

class SegmentTree:
    """Segment tree for efficient prioritized sampling with O(log n) updates and sampling."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, dtype=np.float32)
        self.data = np.zeros(capacity, dtype=np.int32)  # Store indices
        self.size = 0
        
    def _propagate(self, idx: int, change: float):
        """Propagate change up the tree iteratively."""
        while idx > 0:
            parent = idx // 2
            if parent >= 0:
                self.tree[parent] += change
            idx = parent
    
    def update(self, idx: int, priority: float):
        """Update priority for an index."""
        idx = int(idx)
        if idx < 0 or idx >= self.capacity:
            return
            
        # Calculate change
        old_priority = self.tree[self.capacity + idx]
        change = priority - old_priority
        
        # Update leaf
        self.tree[self.capacity + idx] = priority
        
        # Propagate up
        self._propagate(self.capacity + idx, change)
        
        # Update size
        if priority > 0 and idx >= self.size:
            self.size = idx + 1
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample indices based on priorities."""
        if self.size == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float32)
            
        indices = np.zeros(batch_size, dtype=np.int32)
        priorities = np.zeros(batch_size, dtype=np.float32)
        
        for i in range(batch_size):
            # Sample a value between 0 and total priority
            total_priority = self.tree[1]  # Root node
            if total_priority <= 0:
                # Fallback to uniform sampling
                idx = np.random.randint(0, self.size)
                indices[i] = idx
                priorities[i] = 1.0
                continue
                
            sample_val = np.random.uniform(0, total_priority)
            
            # Find the leaf node
            node = 1
            while node < self.capacity:
                left_child = 2 * node
                right_child = 2 * node + 1
                
                if sample_val <= self.tree[left_child]:
                    node = left_child
                else:
                    sample_val -= self.tree[left_child]
                    node = right_child
            
            # Convert back to data index
            data_idx = node - self.capacity
            indices[i] = data_idx
            priorities[i] = self.tree[node]
        
        return indices, priorities
    
    def get_total_priority(self) -> float:
        """Get total priority (root node value)."""
        return self.tree[1] if len(self.tree) > 1 else 0.0


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer using segment tree for efficient sampling."""
    
    def __init__(self, capacity: int, state_size: int, alpha: float = 0.6, beta: float = 0.4, 
                 beta_increment: float = 1e-6, max_priority: float = 1.0):
        self.capacity = capacity
        self.state_size = int(state_size)
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling exponent
        self.beta_increment = beta_increment
        self.max_priority = max_priority
        
        # Pre-allocated arrays for maximum speed
        self.states = np.empty((capacity, self.state_size), dtype=np.float32)
        self.discrete_actions = np.empty((capacity,), dtype=np.int32)
        self.continuous_actions = np.empty((capacity,), dtype=np.float32)
        self.rewards = np.empty((capacity,), dtype=np.float32)
        self.next_states = np.empty((capacity, self.state_size), dtype=np.float32)
        self.dones = np.empty((capacity,), dtype=np.bool_)
        self.priorities = np.full(capacity, max_priority, dtype=np.float32)
        self.frame_indices = np.zeros(capacity, dtype=np.int64)  # For age tracking
        
        # Segment tree for efficient sampling
        self.tree = SegmentTree(capacity)
        
        self.position = 0
        self.size = 0
        self.frame_count = 0  # Track current frame count for age calculation
        
        # Initialize all priorities
        for i in range(capacity):
            self.tree.update(i, max_priority)
    
    def push(self, state, discrete_action, continuous_action, reward, next_state, done, frame_idx: int = 0):
        """Add experience to buffer with high initial priority."""
        # Coerce inputs to proper types
        discrete_idx = int(discrete_action) if not isinstance(discrete_action, int) else discrete_action
        continuous_val = float(continuous_action) if not isinstance(continuous_action, float) else continuous_action
        
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
        self.frame_indices[self.position] = frame_idx
        
        # Set high priority for new experiences
        self.priorities[self.position] = self.max_priority
        self.tree.update(self.position, self.max_priority ** self.alpha)
        
        # Update position and size
        self.position = (self.position + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1
    
    def sample(self, batch_size, frame_count: int = 0):
        """Sample batch using prioritized sampling with importance sampling weights."""
        if self.size < batch_size:
            return None
            
        # Update beta for importance sampling
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Sample indices and priorities from segment tree
        indices, sampled_priorities = self.tree.sample(batch_size)
        
        # Calculate importance sampling weights
        total_priority = self.tree.get_total_priority()
        if total_priority > 0:
            probs = sampled_priorities / total_priority
            weights = (self.size * probs) ** (-self.beta)
            weights /= weights.max()  # Normalize
        else:
            weights = np.ones(batch_size, dtype=np.float32)
        
        # Vectorized gather for batch data
        states_np = self.states[indices]
        next_states_np = self.next_states[indices]
        batch_discrete_actions = self.discrete_actions[indices]
        batch_continuous_actions = self.continuous_actions[indices]
        batch_rewards = self.rewards[indices]
        batch_dones = self.dones[indices]
        batch_weights = weights
        batch_indices = indices  # For priority updates
        
        # Calculate buffer age statistics
        current_frame_idx = frame_count
        ages = current_frame_idx - self.frame_indices[indices]
        avg_age = np.mean(ages) if len(ages) > 0 else 0.0
        
        # Convert to tensors
        use_pinned = (training_device.type == 'cuda')
        states = torch.from_numpy(states_np).float()
        next_states = torch.from_numpy(next_states_np).float()
        discrete_actions = torch.from_numpy(batch_discrete_actions.reshape(-1, 1)).long()
        continuous_actions = torch.from_numpy(batch_continuous_actions.reshape(-1, 1)).float()
        rewards = torch.from_numpy(batch_rewards.reshape(-1, 1)).float()
        dones = torch.from_numpy(batch_dones.reshape(-1, 1).astype(np.uint8)).float()
        importance_weights = torch.from_numpy(batch_weights.reshape(-1, 1)).float()

        if use_pinned:
            states = states.pin_memory()
            next_states = next_states.pin_memory()
            discrete_actions = discrete_actions.pin_memory()
            continuous_actions = continuous_actions.pin_memory()
            rewards = rewards.pin_memory()
            dones = dones.pin_memory()
            importance_weights = importance_weights.pin_memory()

        non_block = True if training_device.type == 'cuda' else False
        states = states.to(training_device, non_blocking=non_block)
        next_states = next_states.to(training_device, non_blocking=non_block)
        discrete_actions = discrete_actions.to(training_device, non_blocking=non_block)
        continuous_actions = continuous_actions.to(training_device, non_blocking=non_block)
        rewards = rewards.to(training_device, non_blocking=non_block)
        dones = dones.to(training_device, non_blocking=non_block)
        importance_weights = importance_weights.to(training_device, non_blocking=non_block)
        
        return (states, discrete_actions, continuous_actions, rewards, next_states, dones, 
                importance_weights, batch_indices, avg_age)
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled experiences."""
        for idx, priority in zip(indices, priorities):
            idx = int(idx)
            priority = float(priority)
            if idx < 0 or idx >= self.capacity:
                continue
            self.priorities[idx] = priority
            self.tree.update(idx, priority ** self.alpha)
    
    def get_buffer_age_stats(self, frame_count: int) -> Tuple[float, float, float]:
        """Get buffer age statistics: mean, min, max age in frames."""
        if self.size == 0:
            return 0.0, 0.0, 0.0
            
        valid_indices = np.arange(self.size)
        ages = frame_count - self.frame_indices[valid_indices]
        return float(np.mean(ages)), float(np.min(ages)), float(np.max(ages))
    
    def __len__(self):
        return self.size


class HybridReplayBuffer:
    """Experience replay buffer for hybrid discrete-continuous actions.
    
    Supports both uniform and prioritized sampling modes.
    """
    def __init__(self, capacity: int, state_size: int, use_prioritized: bool = True):
        self.capacity = capacity
        self.state_size = int(state_size)
        self.use_prioritized = use_prioritized
        
        if use_prioritized:
            self.buffer = PrioritizedReplayBuffer(capacity, state_size)
        else:
            # Fallback to uniform sampling buffer
            self.buffer = self._create_uniform_buffer(capacity, state_size)
    
    def _create_uniform_buffer(self, capacity: int, state_size: int):
        """Create a simple uniform sampling buffer."""
        class UniformBuffer:
            def __init__(self, cap, state_sz):
                self.capacity = cap
                self.state_size = state_sz
                self.states = np.empty((cap, state_sz), dtype=np.float32)
                self.discrete_actions = np.empty((cap,), dtype=np.int32)
                self.continuous_actions = np.empty((cap,), dtype=np.float32)
                self.rewards = np.empty((cap,), dtype=np.float32)
                self.next_states = np.empty((cap, state_sz), dtype=np.float32)
                self.dones = np.empty((cap,), dtype=np.bool_)
                self.position = 0
                self.size = 0
                
            def push(self, state, discrete_action, continuous_action, reward, next_state, done, frame_idx=0):
                # Same push logic as original
                discrete_idx = int(discrete_action)
                continuous_val = float(continuous_action)
                continuous_val = max(-0.9, min(0.9, continuous_val))
                
                try:
                    s = np.asarray(state, dtype=np.float32).reshape(-1)
                    if s.size < self.state_size:
                        tmp = np.zeros((self.state_size,), dtype=np.float32)
                        tmp[:s.size] = s
                        s = tmp
                    elif s.size > self.state_size:
                        s = s[:self.state_size]
                    self.states[self.position, :] = s
                except:
                    self.states[self.position, :] = 0.0
                    
                self.discrete_actions[self.position] = discrete_idx
                self.continuous_actions[self.position] = continuous_val
                self.rewards[self.position] = reward
                
                try:
                    ns = np.asarray(next_state, dtype=np.float32).reshape(-1)
                    if ns.size < self.state_size:
                        tmp = np.zeros((self.state_size,), dtype=np.float32)
                        tmp[:ns.size] = ns
                        ns = tmp
                    elif ns.size > self.state_size:
                        ns = ns[:self.state_size]
                    self.next_states[self.position, :] = ns
                except:
                    self.next_states[self.position, :] = 0.0
                    
                self.dones[self.position] = done
                self.position = (self.position + 1) % self.capacity
                if self.size < self.capacity:
                    self.size += 1
            
            def sample(self, batch_size, frame_count=0):
                if self.size < batch_size:
                    return None
                indices = np.random.randint(0, self.size, size=batch_size)
                
                # Same tensor conversion logic
                use_pinned = (training_device.type == 'cuda')
                states = torch.from_numpy(self.states[indices]).float()
                next_states = torch.from_numpy(self.next_states[indices]).float()
                discrete_actions = torch.from_numpy(self.discrete_actions[indices].reshape(-1, 1)).long()
                continuous_actions = torch.from_numpy(self.continuous_actions[indices].reshape(-1, 1)).float()
                rewards = torch.from_numpy(self.rewards[indices].reshape(-1, 1)).float()
                dones = torch.from_numpy(self.dones[indices].reshape(-1, 1).astype(np.uint8)).float()
                
                if use_pinned:
                    states = states.pin_memory()
                    next_states = next_states.pin_memory()
                    discrete_actions = discrete_actions.pin_memory()
                    continuous_actions = continuous_actions.pin_memory()
                    rewards = rewards.pin_memory()
                    dones = dones.pin_memory()
                
                non_block = True if training_device.type == 'cuda' else False
                states = states.to(training_device, non_blocking=non_block)
                next_states = next_states.to(training_device, non_blocking=non_block)
                discrete_actions = discrete_actions.to(training_device, non_blocking=non_block)
                continuous_actions = continuous_actions.to(training_device, non_blocking=non_block)
                rewards = rewards.to(training_device, non_blocking=non_block)
                dones = dones.to(training_device, non_blocking=non_block)
                
                # Return uniform weights and dummy indices for compatibility
                importance_weights = torch.ones_like(rewards)
                batch_indices = torch.from_numpy(indices.reshape(-1, 1)).long()
                avg_age = 0.0  # Not tracked in uniform buffer
                
                return (states, discrete_actions, continuous_actions, rewards, next_states, dones,
                       importance_weights, batch_indices, avg_age)
            
            def update_priorities(self, indices, priorities):
                pass  # No-op for uniform buffer
                
            def get_buffer_age_stats(self, frame_count):
                return 0.0, 0.0, 0.0
                
            def __len__(self):
                return self.size
                
        return UniformBuffer(capacity, state_size)
    
    def push(self, state, discrete_action, continuous_action, reward, next_state, done, frame_idx: int = 0):
        """Add experience to buffer."""
        self.buffer.push(state, discrete_action, continuous_action, reward, next_state, done, frame_idx)
    
    def sample(self, batch_size, frame_count: int = 0):
        """Sample batch from buffer."""
        return self.buffer.sample(batch_size, frame_count)
    
    def update_priorities(self, indices, priorities):
        """Update priorities (no-op for uniform buffer)."""
        if hasattr(self.buffer, 'update_priorities'):
            self.buffer.update_priorities(indices, priorities)
    
    def get_buffer_age_stats(self, frame_count: int):
        """Get buffer age statistics."""
        return self.buffer.get_buffer_age_stats(frame_count)
    
    def __len__(self):
        return len(self.buffer)

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
        
    # Standard print call - works on all platforms
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
        
        # Hybrid neural networks
        self.qnetwork_local = HybridDQN(
            state_size=state_size,
            discrete_actions=discrete_actions,
            hidden_size=RL_CONFIG.hidden_size,
            num_layers=RL_CONFIG.num_layers,
            use_dueling=RL_CONFIG.use_dueling,
            use_noisy=RL_CONFIG.use_noisy_nets
        ).to(training_device)
        
        self.qnetwork_target = HybridDQN(
            state_size=state_size,
            discrete_actions=discrete_actions,
            hidden_size=RL_CONFIG.hidden_size,
            num_layers=RL_CONFIG.num_layers,
            use_dueling=RL_CONFIG.use_dueling,
            use_noisy=RL_CONFIG.use_noisy_nets
        ).to(training_device)
        
        # Create inference copy on inference device
        if inference_device != training_device:
            print(f"Creating dedicated hybrid inference network on {inference_device}")
            self.qnetwork_inference = HybridDQN(
                state_size=state_size,
                discrete_actions=discrete_actions,
                hidden_size=RL_CONFIG.hidden_size,
                num_layers=RL_CONFIG.num_layers,
                use_dueling=RL_CONFIG.use_dueling,
                use_noisy=RL_CONFIG.use_noisy_nets
            ).to(inference_device)
        else:
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
        
        # Store device references
        self.inference_device = inference_device
        self.training_device = training_device
        
        # Optimizer with separate parameter groups for discrete and continuous components
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)
        
        # Experience replay (vectorized 2D storage with PER support)
        use_per = bool(getattr(RL_CONFIG, 'use_prioritized_replay', True))
        self.memory = HybridReplayBuffer(memory_size, state_size=self.state_size, use_prioritized=use_per)

        # AMP persistent settings
        self.use_amp = bool(getattr(RL_CONFIG, 'use_mixed_precision', False) and self.training_device.type == 'cuda')
        if self.use_amp:
            try:
                from torch.cuda.amp import autocast as _autocast, GradScaler
                self.autocast = _autocast
                self.scaler = GradScaler(enabled=True)
            except Exception:
                self.use_amp = False
                import contextlib
                self.autocast = contextlib.nullcontext
                self.scaler = None
        else:
            import contextlib
            self.autocast = contextlib.nullcontext
            self.scaler = None
        
        # Training queue and metrics
        self.train_queue = queue.Queue(maxsize=100)
        self.training_steps = 0
        self.last_target_update = 0
        self.last_inference_sync = 0
        # Honor global training enable toggle
        self.training_enabled = True

        # Background training worker(s)
        self.running = True
        self.num_training_workers = int(getattr(RL_CONFIG, 'training_workers', 1) or 1)
        # Serialize optimizer updates across threads to avoid race conditions
        self.training_lock = threading.Lock()
        self.training_threads = []
        for i in range(self.num_training_workers):
            t = threading.Thread(target=self.background_train, daemon=True, name=f"HybridTrainWorker-{i}")
            t.start()
            self.training_threads.append(t)

        # Inference sync heartbeat to ensure periodic sync even if training is sparse
        self._sync_interval_seconds = float(getattr(RL_CONFIG, 'inference_sync_interval_sec', 10.0) or 10.0)
        self._sync_thread = threading.Thread(target=self._inference_sync_heartbeat, daemon=True, name="HybridInferenceSyncHeartbeat")
        self._sync_thread.start()
        
    def act(self, state, epsilon=0.0, add_noise=True):
        """Select hybrid action using epsilon-greedy for discrete + Gaussian noise for continuous
        
        Returns:
            discrete_action: int (0-3) for fire/zap combination
            continuous_action: float in [-0.9, +0.9] for spinner
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.inference_device)
        
        # Do not flip modes every call; rely on persistent .eval() for dedicated inference model
        with torch.no_grad():
            discrete_q, continuous_pred = self.qnetwork_inference(state)
        
        # Discrete action selection (epsilon-greedy)
        if random.random() < epsilon:
            discrete_action = random.randint(0, self.discrete_actions - 1)
        else:
            discrete_action = discrete_q.cpu().data.numpy().argmax()
        
        # Continuous action selection (predicted value + optional exploration noise)
        continuous_action = continuous_pred.cpu().data.numpy()[0, 0]
        if add_noise and epsilon > 0:
            # Add Gaussian noise scaled by epsilon for exploration
            noise_scale = epsilon * 0.3  # 30% of action range at full epsilon
            noise = np.random.normal(0, noise_scale)
            continuous_action = np.clip(continuous_action + noise, -0.9, 0.9)
        
        return int(discrete_action), float(continuous_action)
    
    def step(self, state, discrete_action, continuous_action, reward, next_state, done):
        """Add experience to memory and queue training"""
        self.memory.push(state, discrete_action, continuous_action, reward, next_state, done, metrics.frame_count)
        
        # Queue multiple training steps per experience
        if not getattr(metrics, 'training_enabled', True) or not self.training_enabled:
            return
        # Apply training_steps_per_sample ONLY in the background trainer to avoid double-counting here.
        # Enqueue a single token per environment sample to minimize queue overhead.
        try:
            self.train_queue.put_nowait(True)
        except queue.Full:
            pass
        # Optional telemetry
        try:
            metrics.training_queue_size = int(self.train_queue.qsize())
        except Exception:
            pass

    def background_train(self):
        """Background worker that drains the train_queue and performs train steps."""
        worker_id = threading.current_thread().name
        print(f"Training thread {worker_id} started on {self.training_device}")
        while self.running:
            try:
                _ = self.train_queue.get()  # block until work arrives
                # Skip training work if disabled; still mark task done
                if not getattr(metrics, 'training_enabled', True) or not self.training_enabled:
                    self.train_queue.task_done()
                    continue
                steps_per_req = int(getattr(RL_CONFIG, 'training_steps_per_sample', 2) or 1)
                for _ in range(steps_per_req):
                    loss_val = self.train_step()
                    if loss_val is None:
                        loss_val = 0.0
                # Optional telemetry after consuming a token
                try:
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
        """Perform one optimizer update with gradient accumulation over micro-batches.

        Accumulates gradients over RL_CONFIG.gradient_accumulation_steps micro-batches,
        each of size RL_CONFIG.batch_size, then applies a single optimizer.step().
        The entire accumulation window is performed under a lock to keep weights
        consistent across micro-batches when multiple training workers are enabled.
        """
        # Global gate
        if not getattr(metrics, 'training_enabled', True) or not self.training_enabled:
            return 0.0
        # Post-load burn-in: require some fresh frames after loading before training
        try:
            loaded_fc = int(getattr(metrics, 'loaded_frame_count', 0) or 0)
            require_new = int(getattr(RL_CONFIG, 'min_new_frames_after_load_to_train', 0) or 0)
            if loaded_fc > 0 and (metrics.frame_count - loaded_fc) < require_new:
                return 0.0
        except Exception:
            pass

        # Require at least one micro-batch worth of data
        if len(self.memory) < self.batch_size:
            return 0.0

        # Accumulation setup
        grad_accum_steps = max(1, int(getattr(RL_CONFIG, 'gradient_accumulation_steps', 1) or 1))
        use_amp = self.use_amp
        scaler = self.scaler
        w_cont = float(getattr(RL_CONFIG, 'continuous_loss_weight', 0.5) or 0.5)

        # Keep track of loss for telemetry
        total_loss_value = 0.0

        # Perform accumulation and optimizer step atomically for thread safety
        with self.training_lock:
            # Zero gradients before accumulation
            self.optimizer.zero_grad(set_to_none=True)

            preclip_grad_norm = 0.0
            postclip_grad_norm = 0.0

            for acc_idx in range(grad_accum_steps):
                # Sample micro-batch
                batch = self.memory.sample(self.batch_size, metrics.frame_count)
                if batch is None:
                    if acc_idx == 0:
                        return 0.0
                    else:
                        break

                # Handle both PER (9 elements) and uniform (6 elements) sampling
                if len(batch) == 9:
                    # PER enabled: (states, discrete_actions, continuous_actions, rewards, next_states, dones, 
                    #               importance_weights, batch_indices, avg_age)
                    states, discrete_actions, continuous_actions, rewards, next_states, dones, importance_weights, batch_indices, avg_age = batch
                elif len(batch) == 6:
                    # Uniform sampling: (states, discrete_actions, continuous_actions, rewards, next_states, dones)
                    states, discrete_actions, continuous_actions, rewards, next_states, dones = batch
                    importance_weights, batch_indices, avg_age = None, None, 0.0
                else:
                    # Unexpected format
                    print(f"Unexpected batch format with {len(batch)} elements")
                    return 0.0

                # Forward pass (local) under autocast as configured
                if use_amp:
                    with self.autocast():
                        discrete_q_pred, continuous_pred = self.qnetwork_local(states)
                        discrete_q_selected = discrete_q_pred.gather(1, discrete_actions)
                else:
                    discrete_q_pred, continuous_pred = self.qnetwork_local(states)
                    discrete_q_selected = discrete_q_pred.gather(1, discrete_actions)

                # Target computation detached
                with torch.no_grad():
                    if bool(getattr(RL_CONFIG, 'use_double_dqn', True)):
                        next_q_local, _ = self.qnetwork_local(next_states)
                        next_actions = next_q_local.argmax(dim=1, keepdim=True)
                        next_q_target, _ = self.qnetwork_target(next_states)
                        discrete_q_next_max = next_q_target.gather(1, next_actions)
                    else:
                        next_q_target, _ = self.qnetwork_target(next_states)
                        discrete_q_next_max = next_q_target.max(1)[0].unsqueeze(1)

                    # n-step gamma and reward transforms
                    n_step = int(getattr(RL_CONFIG, 'n_step', 1) or 1)
                    gamma_boot = (self.gamma ** n_step) if n_step > 1 else self.gamma
                    r = rewards
                    try:
                        rs = float(getattr(RL_CONFIG, 'reward_scale', 1.0) or 1.0)
                        if rs != 1.0:
                            r = r * rs
                        rc = float(getattr(RL_CONFIG, 'reward_clamp_abs', 0.0) or 0.0)
                        if rc > 0.0:
                            r = torch.clamp(r, -rc, rc)
                        if bool(getattr(RL_CONFIG, 'reward_tanh', False)):
                            r = torch.tanh(r)
                    except Exception:
                        pass
                    discrete_targets = r + (gamma_boot * discrete_q_next_max * (1 - dones))
                    continuous_targets = continuous_actions

                # Match dtypes under AMP
                if use_amp:
                    try:
                        discrete_targets = discrete_targets.to(discrete_q_selected.dtype)
                        continuous_targets = continuous_targets.to(continuous_pred.dtype)
                    except Exception:
                        pass

                # Losses per micro-batch (autocast already applied for forward)
                if use_amp:
                    with self.autocast():
                        d_loss = F.huber_loss(discrete_q_selected, discrete_targets)
                        c_loss = F.mse_loss(continuous_pred, continuous_targets)
                        micro_total = d_loss + w_cont * c_loss
                        # Scale loss for accumulation so effective LR remains constant
                        micro_total = micro_total / float(grad_accum_steps)
                else:
                    d_loss = F.huber_loss(discrete_q_selected, discrete_targets)
                    c_loss = F.mse_loss(continuous_pred, continuous_targets)
                    micro_total = (d_loss + w_cont * c_loss) / float(grad_accum_steps)

                total_loss_value += float((d_loss + w_cont * c_loss).item()) / float(grad_accum_steps)

                # Backward accumulate
                if use_amp and scaler is not None:
                    scaler.scale(micro_total).backward()
                else:
                    micro_total.backward()

            # Unscale once before measuring/clipping
            if use_amp and scaler is not None:
                try:
                    scaler.unscale_(self.optimizer)
                except Exception:
                    pass

            # Replace non-finite grads and compute pre-clip norm
            preclip_sq_sum = 0.0
            try:
                for p in self.qnetwork_local.parameters():
                    if p.grad is not None:
                        if not torch.isfinite(p.grad).all():
                            p.grad.data = torch.nan_to_num(p.grad.data, nan=0.0, posinf=0.0, neginf=0.0)
                        g = p.grad.data.norm(2)
                        if torch.isfinite(g):
                            preclip_sq_sum += float(g.item() ** 2)
                preclip_grad_norm = (preclip_sq_sum ** 0.5)
            except Exception:
                preclip_grad_norm = 0.0

            # Optional gradient value clamp and norm clip
            postclip_grad_norm = preclip_grad_norm
            if hasattr(RL_CONFIG, 'max_grad_norm') and RL_CONFIG.max_grad_norm > 0:
                try:
                    max_gv = float(getattr(RL_CONFIG, 'max_grad_value', 0.0) or 0.0)
                    if max_gv > 0.0:
                        for p in self.qnetwork_local.parameters():
                            if p.grad is not None:
                                p.grad.data.clamp_(-max_gv, max_gv)
                    torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), RL_CONFIG.max_grad_norm)
                    post_sq_sum = 0.0
                    for p in self.qnetwork_local.parameters():
                        if p.grad is not None:
                            g = p.grad.data.norm(2)
                            if torch.isfinite(g):
                                x = float(g.item())
                                post_sq_sum += x * x
                    postclip_grad_norm = (post_sq_sum ** 0.5)
                except Exception:
                    postclip_grad_norm = preclip_grad_norm

            # Optimizer step
            if use_amp and scaler is not None:
                scaler.step(self.optimizer)
                scaler.update()
            else:
                self.optimizer.step()

        # LR scheduling (frame-based)
        try:
            self._apply_lr_schedule()
        except Exception:
            pass
        
        # Update training counters and metrics
        self.training_steps += 1
        try:
            metrics.total_training_steps += 1
            metrics.training_steps_interval += 1
            metrics.memory_buffer_size = len(self.memory)
        except Exception:
            pass
        # Track loss values for display
        try:
            metrics.losses.append(float(total_loss_value))
            metrics.loss_sum_interval += float(total_loss_value)
            metrics.loss_count_interval += 1
        except Exception:
            pass
        # Publish grad diagnostics
        try:
            # Only publish finite diagnostics
            if preclip_grad_norm > 0 and np.isfinite(preclip_grad_norm) and np.isfinite(postclip_grad_norm):
                metrics.grad_clip_delta = float(postclip_grad_norm / max(preclip_grad_norm, 1e-12))
                metrics.grad_norm = float(preclip_grad_norm)
            else:
                metrics.grad_clip_delta = 1.0
                metrics.grad_norm = 0.0
        except Exception:
            pass

        # Periodic hard refresh during warmup or watchdog interval
        try:
            # Hard refresh during warmup steps
            warm_until = int(getattr(RL_CONFIG, 'warmup_hard_refresh_until_steps', 0) or 0)
            warm_every = int(getattr(RL_CONFIG, 'warmup_hard_refresh_every_steps', 0) or 0)
            if warm_until > 0 and self.training_steps <= warm_until and warm_every > 0:
                if (self.training_steps % warm_every) == 0:
                    self.update_target_network()
            # Time-based watchdog
            watchdog_secs = float(getattr(RL_CONFIG, 'hard_update_watchdog_seconds', 0.0) or 0.0)
            if watchdog_secs > 0.0:
                last_t = float(getattr(metrics, 'last_hard_target_update_time', 0.0) or 0.0)
                if last_t <= 0.0 or (time.time() - last_t) >= watchdog_secs:
                    self.update_target_network()
            # Step-based periodic refresh even when using soft target
            step_every = int(getattr(RL_CONFIG, 'hard_target_refresh_every_steps', 0) or 0)
            if step_every > 0 and (self.training_steps % step_every) == 0:
                self.update_target_network()
        except Exception:
            pass

        # Soft target update
        if getattr(RL_CONFIG, 'use_soft_target', True):
            tau = getattr(RL_CONFIG, 'tau', 0.005)
            for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
                target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
            # Telemetry for target updates (soft)
            try:
                metrics.last_target_update_frame = metrics.frame_count
                metrics.last_target_update_time = time.time()
            except Exception:
                pass
        
        # Hard target update
        elif self.training_steps % RL_CONFIG.target_update_freq == 0:
            self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
            # Telemetry for hard target updates
            try:
                metrics.last_target_update_frame = metrics.frame_count
                metrics.last_target_update_time = time.time()
                metrics.last_hard_target_update_frame = metrics.frame_count
                metrics.last_hard_target_update_time = time.time()
            except Exception:
                pass
        
        # Sync inference network
        if inference_device != training_device and self.training_steps % 100 == 0:
            self.sync_inference_network()
        elif self.training_steps % 100 == 0:
            # Even in single-device mode, update sync telemetry periodically
            try:
                metrics.last_inference_sync_frame = metrics.frame_count
                metrics.last_inference_sync_time = time.time()
            except Exception:
                pass
        
        # Update PER priorities if enabled
        try:
            if hasattr(self.memory, 'update_priorities') and len(batch) >= 8:
                _, _, _, _, _, _, importance_weights, batch_indices, _ = batch
                if batch_indices is not None:  # Only update if PER is enabled
                    # Calculate TD errors for priority updates (simplified - using loss magnitude)
                    # In a full implementation, you'd compute the full TD error here
                    # For now, use a simple heuristic based on loss magnitude
                    if total_loss_value > 0:
                        # Scale priorities by loss magnitude, with minimum priority
                        new_priorities = np.maximum(total_loss_value, 0.01) * np.ones(len(batch_indices))
                        self.memory.update_priorities(batch_indices.cpu().numpy(), new_priorities)
        except Exception:
            pass  # Silently skip priority updates if anything goes wrong
        
        return float(total_loss_value)

    def _apply_lr_schedule(self):
        """Compute and apply per-frame LR based on RL_CONFIG schedule."""
        sched = str(getattr(RL_CONFIG, 'lr_schedule', 'none') or 'none').lower()
        if sched == 'none':
            return
        # Determine current global frame
        try:
            fc = int(getattr(metrics, 'frame_count', 0) or 0)
        except Exception:
            fc = 0
        lr_base = float(getattr(RL_CONFIG, 'lr_base', RL_CONFIG.lr) or RL_CONFIG.lr)
        lr_min = float(getattr(RL_CONFIG, 'lr_min', lr_base * 0.1) or (lr_base * 0.1))
        warm = int(getattr(RL_CONFIG, 'lr_warmup_frames', 0) or 0)
        hold = int(getattr(RL_CONFIG, 'lr_hold_until_frames', 0) or 0)
        decay_end = int(getattr(RL_CONFIG, 'lr_decay_until_frames', 0) or 0)

        # Piecewise: warmup -> hold -> cosine decay -> floor
        if fc <= warm and warm > 0:
            # Linear from lr_min to lr_base
            t = fc / max(warm, 1)
            lr_now = lr_min + (lr_base - lr_min) * t
        elif fc <= hold or decay_end <= hold:
            lr_now = lr_base
        elif fc <= decay_end:
            # Cosine from lr_base to lr_min over [hold, decay_end]
            import math
            t = (fc - hold) / max(decay_end - hold, 1)
            lr_now = lr_min + 0.5 * (lr_base - lr_min) * (1 + math.cos(math.pi * t))
        else:
            lr_now = lr_min

        # Apply to optimizer groups
        for g in self.optimizer.param_groups:
            g['lr'] = float(lr_now)

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

    def sync_inference_network(self):
        """Synchronize inference network weights and record telemetry."""
        did_copy = False
        try:
            if hasattr(self, 'qnetwork_inference') and self.qnetwork_inference is not self.qnetwork_local:
                with torch.no_grad():
                    state = {}
                    for k, p in self.qnetwork_local.state_dict().items():
                        state[k] = p.to(self.inference_device)
                    self.qnetwork_inference.load_state_dict(state)
                did_copy = True
        finally:
            try:
                metrics.last_inference_sync_frame = metrics.frame_count
                metrics.last_inference_sync_time = time.time()
            except Exception:
                pass
        return did_copy

    def _inference_sync_heartbeat(self):
        """Periodic sync to keep inference network fresh even with low training activity."""
        while self.running:
            try:
                now = time.time()
                last = getattr(metrics, 'last_inference_sync_time', 0.0)
                if last <= 0.0 or (now - last) >= self._sync_interval_seconds:
                    self.sync_inference_network()
                time.sleep(1.0)
            except Exception:
                time.sleep(1.0)

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
        from config import RESET_METRICS, SERVER_CONFIG, FORCE_FRESH_MODEL

        if FORCE_FRESH_MODEL:
            print("FORCE_FRESH_MODEL is True - skipping model load, starting fresh weights")
            return False

        if os.path.exists(filepath):
            try:
                checkpoint = torch.load(filepath, map_location=training_device)

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

                # Sync inference network
                if inference_device != training_device:
                    self.qnetwork_inference.load_state_dict(checkpoint['local_state_dict'])

                # Sanity-check Q-values on load to catch corruption/explosion
                try:
                    with torch.no_grad():
                        dummy = torch.zeros(1, self.state_size, device=self.training_device)
                        dq, _ = self.qnetwork_local(dummy)
                        qmax = float(dq.max().item())
                        qmin = float(dq.min().item())
                        if max(abs(qmax), abs(qmin)) > 10.0:
                            print("WARNING: Loaded hybrid model has extreme Q-values. Rescaling weights...")
                            scale = 5.0 / max(abs(qmax), abs(qmin))
                            for p in self.qnetwork_local.parameters():
                                p.data.mul_(scale)
                            for p in self.qnetwork_target.parameters():
                                p.data.mul_(scale)
                            dq2, _ = self.qnetwork_local(dummy)
                            print(f"Rescaled Q-value range: [{float(dq2.min().item()):.3f}, {float(dq2.max().item()):.3f}]")
                except Exception:
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

    def stop(self, join: bool = True, timeout: float = 2.0):
        """Gracefully stop background threads and heartbeat."""
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
            try:
                self._sync_thread.join(timeout=timeout)
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
                            states = states.to(self.inference_device, non_blocking=True)
                            discrete_q, _ = self.qnetwork_inference(states)
                            min_q = discrete_q.min().item()
                            max_q = discrete_q.max().item()

                if min_q is None or max_q is None:
                    dummy = torch.zeros(1, self.state_size, device=self.inference_device)
                    discrete_q, _ = self.qnetwork_inference(dummy)
                    min_q = discrete_q.min().item()
                    max_q = discrete_q.max().item()

                return float(min_q), float(max_q)
        except Exception:
            return float('nan'), float('nan')

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

        Currently: if GS_ZoomingDown (0x20), return min(current_epsilon, epsilon_when_zooming).
        Honors override_epsilon by returning 0.0 regardless of state.
        """
        with self.lock:
            try:
                if getattr(self.metrics, 'override_epsilon', False):
                    return 0.0
                eps = float(self.metrics.epsilon)
                if gamestate == 0x20:  # GS_ZoomingDown
                    try:
                        zoom_eps = float(getattr(RL_CONFIG, 'epsilon_when_zooming', 0.05) or 0.05)
                    except Exception:
                        zoom_eps = 0.05
                    eps = min(eps, zoom_eps)  # Use the lower of current epsilon and zoom epsilon
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
            # Respect override_expert: freeze expert_ratio at 0 while override is ON
            if self.metrics.override_expert:
                return self.metrics.expert_ratio
            decay_expert_ratio(self.metrics.frame_count)
            return self.metrics.expert_ratio
    
    def add_episode_reward(self, total_reward, dqn_reward, expert_reward):
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
                    add_fn(float(total_reward), float(dqn_reward), float(expert_reward))
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
            self.metrics.open_level = open_level
    
    def get_expert_ratio(self):
        with self.lock:
            return self.metrics.expert_ratio
            
    def is_override_active(self):
        with self.lock:
            return self.metrics.override_expert

    def get_fps(self):
        with self.lock:
            return self.metrics.fps

    def record_death(self, death_reason: int):
        """Thread-safe passthrough to MetricsData.record_death if available."""
        with self.lock:
            try:
                if hasattr(self.metrics, 'record_death') and callable(self.metrics.record_death):
                    self.metrics.record_death(int(death_reason))
            except Exception:
                pass

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


def parse_frame_data(data: bytes) -> Optional[FrameData]:
    """Parse binary frame data from Lua into game state - SIMPLIFIED with float32 payload"""
    try:
        if not data or len(data) < 10:  # Minimal size check
            print("ERROR: Received empty or too small data packet", flush=True)
            sys.exit(1)
        
        # Fixed OOB header format (must match Lua exactly). Precompute once.
        # Format: ">HdBBBHHHBBBhBhBBBBBBb"
        global _FMT_OOB, _HDR_OOB
        try:
            _FMT_OOB
        except NameError:
            _FMT_OOB = ">HdBBBHHHBBBhBhBBBBBBb"
            _HDR_OOB = struct.calcsize(_FMT_OOB)

        if len(data) < _HDR_OOB:
            print(f"ERROR: Received data too small: {len(data)} bytes, need {_HDR_OOB}", flush=True)
            sys.exit(1)

        values = struct.unpack(_FMT_OOB, data[:_HDR_OOB])
        (num_values, reward, gamestate, game_mode, done, frame_counter, score_high, score_low,
         save_signal, fire, zap, spinner, is_attract, nearest_enemy, player_seg, is_open,
         expert_fire, expert_zap, level_number, player_alive, death_reason) = values
        header_size = _HDR_OOB
        
        # Combine score components
        score = (score_high * 65536) + score_low
        
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
            print(f"[CRITICAL] Frame {frame_counter}: {nan_count} NaN values, {inf_count} Inf values detected! "
                  f"This will cause training instability!", flush=True)

        # Ensure all values are in expected range [-1, 1] with warnings for issues
        out_of_range_count = np.sum((state < -1.001) | (state > 1.001))
        if out_of_range_count > 0:
            print(f"[WARNING] Frame {frame_counter}: {out_of_range_count} values outside [-1,1] range", flush=True)
            state = np.clip(state, -1.0, 1.0)
        
        frame_data = FrameData(
            state=state,
            reward=reward,
            action=(bool(fire), bool(zap), spinner),
            mode=game_mode,
            gamestate=gamestate,
            done=bool(done),
            attract=bool(is_attract),
            save_signal=bool(save_signal),
            enemy_seg=nearest_enemy,
            player_seg=player_seg,
            open_level=bool(is_open),
            expert_fire=bool(expert_fire),
            expert_zap=bool(expert_zap),
            level_number=level_number,
            player_alive=bool(player_alive),
            death_reason=int(death_reason)
        )
        
        return frame_data
    except Exception as e:
        print(f"ERROR parsing frame data: {e}", flush=True)
        sys.exit(1)

def display_metrics_header(kb=None):
    """Display header for metrics table"""
    if not IS_INTERACTIVE: return
    header = (
        f"{'Frame':>8} | {'FPS':>5} | {'Clients':>7} | {'Mean Reward':>12} | {'DQN Reward':>10} | {'Loss':>8} | "
        f"{'Epsilon':>7} | {'Guided %':>8} | {'Mem Size':>8} | {'Avg Level':>9} | {'Level Type':>10} | {'OVR':>3} | {'Expert':>6} | "
        f"{'InferSync ΔF/Δt':>16} | {'HardUpd ΔF/Δt':>17} | {'Q-Value Range':>14}"
    )
    print_with_terminal_restore(kb, f"\n{'-' * len(header)}")
    print_with_terminal_restore(kb, header)
    print_with_terminal_restore(kb, f"{'-' * len(header)}")

def display_metrics_row(agent, kb=None):
    """Display current metrics in tabular format"""
    if not IS_INTERACTIVE: return
    mean_reward = np.mean(list(metrics.episode_rewards)) if metrics.episode_rewards else float('nan')
    dqn_reward = np.mean(list(metrics.dqn_rewards)) if metrics.dqn_rewards else float('nan')
    mean_loss = np.mean(list(metrics.losses)) if metrics.losses else float('nan')
    guided_ratio = metrics.expert_ratio
    mem_size = len(agent.memory) if agent else 0
    
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

    # Compute frames/time since last inference sync and last HARD target update
    now = time.time()
    sync_df = metrics.frame_count - getattr(metrics, 'last_inference_sync_frame', 0)
    targ_df = metrics.frame_count - getattr(metrics, 'last_hard_target_update_frame', 0)
    last_sync_time = getattr(metrics, 'last_inference_sync_time', 0.0)
    last_targ_time = getattr(metrics, 'last_hard_target_update_time', 0.0)
    sync_dt = (now - last_sync_time) if last_sync_time > 0.0 else None
    targ_dt = (now - last_targ_time) if last_targ_time > 0.0 else None
    sync_col = f"{sync_df:6d}/{(f'{sync_dt:6.1f}s' if sync_dt is not None else '   n/a')}"
    targ_col = f"{targ_df:6d}/{(f'{targ_dt:6.1f}s' if targ_dt is not None else '   n/a')}"
    
    row = (
        f"{metrics.frame_count:8d} | {metrics.fps:5.1f} | {client_count:>7} | {mean_reward:12.2f} | {dqn_reward:10.2f} | "
        f"{mean_loss:8.2f} | {metrics.epsilon:7.3f} | {guided_ratio*100:7.2f}% | "
    f"{mem_size:8d} | {display_level:9.2f} | {'Open' if metrics.open_level else 'Closed':10} | {override_status:>3} | {'ON' if metrics.expert_mode else 'OFF':>6} | "
        f"{sync_col:>16} | {targ_col:>17} | {q_range:>14}"
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

### Legacy conversion helpers removed (pure hybrid model)

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
    # Skip decay if expert mode or override is active
    if metrics.expert_mode or metrics.override_expert:
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

    # Enforce minimum to prevent expert ratio from decaying below configured floor
    metrics.expert_ratio = max(metrics.expert_ratio, RL_CONFIG.expert_ratio_min)

    return metrics.expert_ratio

# Discrete-only SimpleReplayBuffer removed (hybrid-only)
