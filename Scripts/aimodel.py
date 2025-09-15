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
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import GradScaler, autocast  # For mixed precision training
import select
import threading
import queue
from collections import deque, namedtuple
from datetime import datetime
from stable_baselines3 import DQN
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.logger import configure
import socket
import traceback
from torch.nn import SmoothL1Loss # Import SmoothL1Loss

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
    ACTION_MAPPING,
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
    # Lua-provided reward component breakdown (OOB header). Defaults keep parsing robust.
    rc_safety: float = 0.0
    rc_proximity: float = 0.0
    rc_shots: float = 0.0
    rc_threats: float = 0.0
    rc_pulsar: float = 0.0
    rc_score: float = 0.0
    
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
            level_number=data.get("level_number", 0)  # Default to 0 if not provided
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

# Display key configuration parameters
print(f"Learning rate: {RL_CONFIG.lr}")
print(f"Batch size: {RL_CONFIG.batch_size}")
print(f"Memory size: {RL_CONFIG.memory_size:,}")
print(f"Hidden size: {RL_CONFIG.hidden_size}")
if getattr(RL_CONFIG, 'use_per', False):
    print(f"Using PER with alpha: {getattr(RL_CONFIG, 'per_alpha', 0.6)}")
else:
    print("Using standard experience replay")
print(f"Mixed precision: {'enabled' if getattr(RL_CONFIG, 'use_mixed_precision', False) else 'disabled'}")
print(f"State size: {RL_CONFIG.state_size}")

# For compatibility with dual-device code
device = training_device  # Legacy compatibility

# Initialize metrics
metrics = config_metrics

# Global reference to server for metrics display
metrics.global_server = None

# Experience replay memory
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory:
    """Replay buffer with O(1) sampling using numpy circular buffer"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.size = 0
        self.index = 0
        
        # Pre-allocate numpy arrays for O(1) access - will be initialized on first push
        self.states = None
        self.actions = None 
        self.rewards = None
        self.next_states = None
        self.dones = None
        
    def push(self, state, action, reward, next_state, done):
        """Add experience to memory - O(1) operation with thread safety"""
        # Normalize inputs first
        try:
            a = int(action)
        except Exception:
            a = int(np.array(action).reshape(-1)[0])
        action_val = np.array([a], dtype=np.int64)
        reward_val = np.array([float(reward)], dtype=np.float32)
        done_val = np.array([float(done)], dtype=np.float32)
        
        # Initialize arrays on first push - thread safe
        if self.states is None:
            # Initialize all arrays atomically
            states = np.zeros((self.capacity,) + state.shape, dtype=np.float32)
            actions = np.zeros((self.capacity, 1), dtype=np.int64)
            rewards = np.zeros((self.capacity, 1), dtype=np.float32)
            next_states = np.zeros((self.capacity,) + next_state.shape, dtype=np.float32)
            dones = np.zeros((self.capacity, 1), dtype=np.float32)
            
            # Assign all at once to avoid race conditions
            self.states = states
            self.actions = actions
            self.rewards = rewards
            self.next_states = next_states
            self.dones = dones
            
        # Store data in circular buffer
        self.states[self.index] = state
        self.actions[self.index] = action_val
        self.rewards[self.index] = reward_val
        self.next_states[self.index] = next_state  
        self.dones[self.index] = done_val
        
        # Update pointers
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size):
        """Sample random batch of experiences - TRUE O(1) operation"""
        if self.size == 0:
            return None
            
        batch_size = min(batch_size, self.size)
        
        # O(1) random sampling using numpy indexing - no loops!
        indices = np.random.randint(0, self.size, size=batch_size)
        
        # O(1) batch extraction using numpy advanced indexing
        batch_states = self.states[indices]
        batch_actions = self.actions[indices]
        batch_rewards = self.rewards[indices]
        batch_next_states = self.next_states[indices]
        batch_dones = self.dones[indices]
        
        # Direct GPU transfer with pinned memory
        return (torch.from_numpy(batch_states).pin_memory().to(training_device, non_blocking=True),
                torch.from_numpy(batch_actions).pin_memory().to(training_device, non_blocking=True),
                torch.from_numpy(batch_rewards).pin_memory().to(training_device, non_blocking=True),
                torch.from_numpy(batch_next_states).pin_memory().to(training_device, non_blocking=True),
                torch.from_numpy(batch_dones).pin_memory().to(training_device, non_blocking=True))
        
    def __len__(self):
        return self.size

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

class DQN(nn.Module):
    """Deep Q-Network model - Balanced architecture for dual GPU setup with reasonable training load."""
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        use_noisy = RL_CONFIG.use_noisy_nets
        use_dueling = getattr(RL_CONFIG, 'use_dueling', False)
        noisy_std = getattr(RL_CONFIG, 'noisy_std_init', 0.5)
        hidden_size = RL_CONFIG.hidden_size  # 4096 - Balanced for dual GPU setup
        LinearOrNoisy = NoisyLinear if use_noisy else nn.Linear

        # Balanced feature extractor for dual GPU setup
        self.fc1 = nn.Linear(state_size, hidden_size)         # 175 -> 4096  
        self.fc2 = nn.Linear(hidden_size, hidden_size)        # 4096 -> 4096
        self.fc3 = nn.Linear(hidden_size, hidden_size//2)     # 4096 -> 2048
        self.fc4 = nn.Linear(hidden_size//2, hidden_size//4)  # 2048 -> 1024
        self.fc5 = nn.Linear(hidden_size//4, hidden_size//8)  # 1024 -> 512
        
        # Apply noisy layers if enabled
        if use_noisy:
            self.fc2 = NoisyLinear(hidden_size, hidden_size, noisy_std)
            self.fc3 = NoisyLinear(hidden_size, hidden_size//2, noisy_std)
            self.fc4 = NoisyLinear(hidden_size//2, hidden_size//4, noisy_std)
            self.fc5 = NoisyLinear(hidden_size//4, hidden_size//8, noisy_std)

        self.use_dueling = use_dueling
        if use_dueling:
            # Dueling streams with balanced sizes
            self.val_fc = nn.Linear(hidden_size//8, hidden_size//16)  # 512 -> 256
            self.adv_fc = nn.Linear(hidden_size//8, hidden_size//16)  # 512 -> 256
            self.val_out = nn.Linear(hidden_size//16, 1)              # 256 -> 1
            self.adv_out = nn.Linear(hidden_size//16, action_size)    # 256 -> 18
        else:
            # Single stream with balanced sizes
            self.fc6 = nn.Linear(hidden_size//8, hidden_size//16)     # 512 -> 256
            self.out = nn.Linear(hidden_size//16, action_size)        # 256 -> 18

    def reset_noise(self):
        # Reset noise for all NoisyLinear layers
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def forward(self, x):
        # Balanced feature extraction for dual GPU setup
        x = F.relu(self.fc1(x))    # 175 -> 4096
        x = F.relu(self.fc2(x))    # 4096 -> 4096
        x = F.relu(self.fc3(x))    # 4096 -> 2048
        x = F.relu(self.fc4(x))    # 2048 -> 1024
        x = F.relu(self.fc5(x))    # 1024 -> 512
        
        if self.use_dueling:
            v = F.relu(self.val_fc(x))  # 512 -> 256
            v = self.val_out(v)         # 256 -> 1
            a = F.relu(self.adv_fc(x))  # 512 -> 256
            a = self.adv_out(a)         # 256 -> 18
            # Subtract mean advantage to keep Q identifiable
            q = v + (a - a.mean(dim=1, keepdim=True))
            return q
        else:
            x = F.relu(self.fc6(x))     # 512 -> 256
            return self.out(x)          # 256 -> 18

class QRDQN(nn.Module):
    """Quantile Regression DQN (distributional), optional dueling streams.
    Outputs shape: (batch, action_size, num_quantiles)
    """
    def __init__(self, state_size: int, action_size: int, num_quantiles: int):
        super().__init__()
        use_noisy = RL_CONFIG.use_noisy_nets
        use_dueling = getattr(RL_CONFIG, 'use_dueling', False)
        noisy_std = getattr(RL_CONFIG, 'noisy_std_init', 0.5)
        LinearOrNoisy = NoisyLinear if use_noisy else nn.Linear

        self.num_quantiles = num_quantiles
        self.action_size = action_size
        self.use_dueling = use_dueling

        # Shared trunk
        self.fc1 = nn.Linear(state_size, 512)
        self.fc2 = LinearOrNoisy(512, 384, noisy_std) if use_noisy else LinearOrNoisy(512, 384)
        self.fc3 = LinearOrNoisy(384, 192, noisy_std) if use_noisy else LinearOrNoisy(384, 192)

        if use_dueling:
            self.val_fc = nn.Linear(192, 128)
            self.adv_fc = nn.Linear(192, 128)
            self.val_out = nn.Linear(128, num_quantiles)             # (B, N)
            self.adv_out = nn.Linear(128, action_size * num_quantiles) # (B, A*N)
        else:
            self.fc4 = nn.Linear(192, 128)
            self.out = nn.Linear(128, action_size * num_quantiles)

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        if self.use_dueling:
            v = F.relu(self.val_fc(x))
            v = self.val_out(v)  # (B, N)
            a = F.relu(self.adv_fc(x))
            a = self.adv_out(a)  # (B, A*N)
            a = a.view(-1, self.action_size, self.num_quantiles)
            v = v.unsqueeze(1)  # (B, 1, N)
            # Center advantages over actions
            q = v + (a - a.mean(dim=1, keepdim=True))
            return q  # (B, A, N)
        else:
            x = F.relu(self.fc4(x))
            out = self.out(x)  # (B, A*N)
            return out.view(-1, self.action_size, self.num_quantiles)

class PrioritizedReplayMemory:
    """Prioritized experience replay using proportional priorities."""
    def __init__(self, capacity: int, alpha: float = 0.6, eps: float = 1e-6):
        self.capacity = capacity
        self.alpha = alpha
        self.eps = eps
        self.pos = 0
        self.full = False
        self.states = np.zeros((capacity, RL_CONFIG.state_size), dtype=np.float32)
        self.actions = np.zeros((capacity, 1), dtype=np.int64)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, RL_CONFIG.state_size), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.priorities = np.full((capacity, 1), self.eps, dtype=np.float32)  # CRITICAL FIX: Initialize with eps, not zero!

        # Running average for monitoring
        self.avg_priority = 0.0

    def __len__(self):
        return self.capacity if self.full else self.pos

    def push(self, state, action, reward, next_state, done):
        idx = self.pos
        self.states[idx] = state
        # Coerce action to scalar index
        try:
            if isinstance(action, np.ndarray):
                a_idx = int(action.reshape(-1)[0])
            elif isinstance(action, (list, tuple)):
                a_idx = int(action[0])
            else:
                a_idx = int(action)
        except Exception:
            a_idx = int(action)
      
        self.actions[idx, 0] = a_idx
        self.rewards[idx, 0] = reward
        self.next_states[idx] = next_state
        self.dones[idx, 0] = float(done)
        
        # Set priority for new experience
        current_priorities = self.priorities[:self.pos, 0] if not self.full else self.priorities[:, 0]
        if len(current_priorities) > 0:
            # STRICT ASSERTION: All existing priorities should be valid
            assert not np.any(current_priorities <= 0), f"CRITICAL BUG: Found {np.sum(current_priorities <= 0)} zero/negative priorities in buffer during push!"
            assert np.all(np.isfinite(current_priorities)), f"CRITICAL BUG: Found NaN/Inf priorities in buffer during push!"
            max_prio = current_priorities.max()
        else:
            max_prio = 1.0
            
        self.priorities[idx, 0] = max_prio
        
        # STRICT ASSERTION: Verify the priority we just set is valid
        assert max_prio > 0 and np.isfinite(max_prio), f"CRITICAL BUG: Setting invalid priority {max_prio}"

        # Update position
        self.pos = (self.pos + 1) % self.capacity
        if self.pos == 0:
            self.full = True

    def sample(self, batch_size: int, beta: float = 0.4):
        """Sample batch with importance sampling weights - STRICT MODE with early assertions"""
        size = len(self)
        assert size > 0, "Cannot sample from empty buffer"
        assert batch_size <= size, f"Cannot sample {batch_size} items from buffer of size {size}"
            
        # Get all current priorities
        prios = self.priorities[:size, 0]
        
        # STRICT ASSERTION: No zero/negative priorities should ever exist
        zero_count = np.sum(prios <= 0)
        assert zero_count == 0, f"CRITICAL BUG: Found {zero_count} zero/negative priorities in buffer! This indicates persisted bad state."
        
        # STRICT ASSERTION: No NaN/Inf priorities should ever exist  
        nan_count = np.sum(~np.isfinite(prios))
        assert nan_count == 0, f"CRITICAL BUG: Found {nan_count} NaN/Inf priorities in buffer! This indicates memory corruption."
        
        # Calculate probabilities with alpha weighting
        prob_alpha = prios ** self.alpha
        probs = prob_alpha / prob_alpha.sum()
        
        # STRICT ASSERTION: Probabilities must be valid
        assert np.all(np.isfinite(probs)), "CRITICAL BUG: Invalid probability distribution contains NaN/Inf!"
        assert np.allclose(probs.sum(), 1.0, atol=1e-6), f"CRITICAL BUG: Probabilities don't sum to 1.0! Sum={probs.sum()}"
        
        # Sample indices
        indices = np.random.choice(size, batch_size, p=probs, replace=False)
        
        # Calculate importance sampling weights
        # weight = (N * P(i))^(-β) 
        sampled_probs = probs[indices]
        weights = (size * sampled_probs) ** (-beta)
        weights = weights / weights.max()  # Normalize by max weight
        
        # STRICT ASSERTION: Weights must be valid
        assert np.all(np.isfinite(weights)), "CRITICAL BUG: Invalid importance sampling weights contain NaN/Inf!"
        assert np.all(weights > 0), f"CRITICAL BUG: Found zero/negative importance sampling weights! Range=[{weights.min()}, {weights.max()}]"
        
        # Return sampled batch as PyTorch tensors on training device
        states = torch.from_numpy(self.states[indices]).float().to(training_device)
        actions = torch.from_numpy(self.actions[indices]).long().to(training_device)
        rewards = torch.from_numpy(self.rewards[indices]).float().to(training_device)
        next_states = torch.from_numpy(self.next_states[indices]).float().to(training_device)
        dones = torch.from_numpy(self.dones[indices]).float().to(training_device)
        is_weights = torch.from_numpy(weights.reshape(-1, 1)).float().to(training_device)
        
        return states, actions, rewards, next_states, dones, is_weights, indices


    def validate_priorities(self):
        """Debug function to validate priority buffer integrity"""
        size = len(self)
        if size == 0:
            return
            
        prios = self.priorities[:size, 0]
        print(f"Priority Validation (size {size}):")
        print(f"  Min: {prios.min():.6f}, Max: {prios.max():.6f}")
        print(f"  Mean: {prios.mean():.6f}, Std: {prios.std():.6f}")
        print(f"  NaN count: {np.isnan(prios).sum()}")
        print(f"  Inf count: {np.isinf(prios).sum()}")
        print(f"  Zero count: {(prios == 0).sum()}")
        print(f"  Negative count: {(prios < 0).sum()}")

    def update_priorities(self, indices, td_errors):
        """Update priorities with strict assertions - fails fast on any issues"""
        assert td_errors is not None and len(td_errors) > 0, "update_priorities called with empty/None td_errors"
            
        # Convert to numpy and validate
        td = td_errors.detach().abs().cpu().numpy().reshape(-1)
        
        # STRICT ASSERTION: TD errors must be finite and positive
        assert not np.any(np.isnan(td)) and not np.any(np.isinf(td)), f"CRITICAL BUG: TD errors contain NaN/Inf! This indicates training instability."
        assert not np.any(td < 0), f"CRITICAL BUG: TD errors are negative after abs()! Min value: {td.min()}"
        
        # DIAGNOSTIC: Log TD error distribution before clamping
        # if len(self) > 1000 and np.random.random() < 0.01:  # 1% chance to log
        #     print(f"TD errors before clamping: min={td.min():.4f}, max={td.max():.4f}, mean={td.mean():.4f}")
        
        # CRITICAL FIX: Clamp TD errors to prevent priority explosion feedback loop
        # This breaks the cycle where large TD errors -> high priorities -> more sampling -> larger gradients -> larger TD errors
        td = np.clip(td, 0.0, 5.0)  # Clamp TD errors to max 5.0 to prevent priority explosion
        
        # STRICT ASSERTION: TD errors should be reasonable (not exploded)
        max_td = td.max()
        assert max_td <= 5.0, f"CRITICAL BUG: TD error still too large after clamping! Max TD error = {max_td:.6f}."
        
        # Validate indices
        indices = np.asarray(indices)
        assert len(indices) == len(td), f"Mismatched lengths: {len(indices)} indices vs {len(td)} TD errors"
        assert not np.any(indices < 0) and not np.any(indices >= len(self)), f"Invalid indices: range [{indices.min()}, {indices.max()}] for buffer size {len(self)}"
        
        # Update priorities (add epsilon for stability)
        new_priorities = td + self.eps
        old_priorities = self.priorities[indices, 0].copy()
        self.priorities[indices, 0] = new_priorities
        
        # DIAGNOSTIC: Track priority updates
        # if len(self) > 1000 and np.random.random() < 0.01:  # 1% chance to log
        #     print(f"Priority update: old_max={old_priorities.max():.4f}, new_max={new_priorities.max():.4f}")
        
        # Update running average
        self.avg_priority = float(new_priorities.mean())

class DQNAgent:
    """DQN Agent with experience replay and target network"""
    def __init__(self, state_size, action_size, learning_rate=RL_CONFIG.lr, gamma=RL_CONFIG.gamma, 
                 epsilon=RL_CONFIG.epsilon, epsilon_min=RL_CONFIG.epsilon_min, 
                 memory_size=RL_CONFIG.memory_size, batch_size=RL_CONFIG.batch_size):
        self.state_size = state_size
        self.action_size = action_size
        self.last_save_time = 0.0 # Initialize last save time
        
        # Q-Networks on separate devices for optimal performance
        if getattr(RL_CONFIG, 'use_distributional', False):
            n_quant = getattr(RL_CONFIG, 'num_atoms', 51)
            self.qnetwork_local = QRDQN(state_size, action_size, n_quant).to(training_device)
            self.qnetwork_target = QRDQN(state_size, action_size, n_quant).to(training_device)
        else:
            self.qnetwork_local = DQN(state_size, action_size).to(training_device)
            self.qnetwork_target = DQN(state_size, action_size).to(training_device)
        
        # Create inference copy on inference device for low-latency serving
        if inference_device != training_device:
            print(f"Creating dedicated inference network on {inference_device} for low-latency serving")
            if getattr(RL_CONFIG, 'use_distributional', False):
                self.qnetwork_inference = QRDQN(state_size, action_size, n_quant).to(inference_device)
            else:
                self.qnetwork_inference = DQN(state_size, action_size).to(inference_device)
        else:
            # Same device - use training network for inference
            self.qnetwork_inference = self.qnetwork_local
        
        # torch.compile removed from forward pass - causes CUDA graph issues with multi-threaded clients
        print("Using standard (uncompiled) neural networks for stability.")
        
        # Store device references
        self.inference_device = inference_device
        self.training_device = training_device
        
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)
        # Optional LR scheduler
        self.scheduler = None
        if getattr(RL_CONFIG, 'use_lr_scheduler', False):
            step_size = getattr(RL_CONFIG, 'scheduler_step_size', 100000)
            gamma = getattr(RL_CONFIG, 'scheduler_gamma', 0.5)
            self.scheduler = StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        
        # Replay memory (optionally prioritized)
        if getattr(RL_CONFIG, 'use_per', False):
            print("Using Prioritized Experience Replay (PER)")
            self.memory = PrioritizedReplayMemory(memory_size, alpha=getattr(RL_CONFIG, 'per_alpha', 0.6), eps=getattr(RL_CONFIG, 'per_eps', 1e-6))
        else:
            print("Using standard Replay Memory")
            self.memory = ReplayMemory(memory_size)

        # Initialize target network with same weights as local network
        self.update_target_network()

        # Training queue for background thread - much larger to prevent throttling
        self.train_queue = queue.Queue(maxsize=50000)  # Increased from 20000 to 50000
        self.training_thread = None
        self.running = True
        # Multi-threaded training with gradient synchronization
        self.num_training_workers = getattr(RL_CONFIG, 'training_workers', 1)  # Configurable worker count
        self.training_threads = []
        self._train_step_counter = 0
        self._gradient_accumulation_counter = 0  # Track gradient accumulation steps
        self.training_lock = threading.Lock()  # Lock for optimizer updates only
        self.gradient_lock = threading.Lock()  # Separate lock for gradient accumulation
        
        # Mixed precision training setup
        self.use_mixed_precision = getattr(RL_CONFIG, 'use_mixed_precision', False)
        if self.use_mixed_precision and torch.cuda.is_available():
            self.scaler = GradScaler()
            print("Mixed precision training enabled")
        else:
            self.scaler = None
        
        # torch.compile for training computation - safe in single-threaded training context
        self._compiled_loss_computation = None
        if (torch.cuda.is_available() and hasattr(torch, 'compile') and 
            getattr(RL_CONFIG, 'use_torch_compile', False)):
            try:
                print("Attempting to compile loss computation for training acceleration...")
                self._compiled_loss_computation = torch.compile(
                    self._compute_dqn_loss,
                    mode='max-autotune',  # Aggressive optimization for training
                    fullgraph=False  # Allow graph breaks for compatibility
                )
                print("Loss computation compilation successful")
            except Exception as e:
                print(f"Loss computation compilation failed: {e}")
                self._compiled_loss_computation = None
        
        # CRITICAL FIX: Initialize training enabled flag to ensure training happens
        self.training_enabled = True

        # Memory optimizations for dual GPU training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear any existing cache
            torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
            torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
            # Additional GPU optimizations for dual GPU setup
            torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for faster matmul on Ampere+
            torch.backends.cudnn.allow_tf32 = True        # Enable TF32 for cuDNN
            # Moderate GPU memory allocation for dual GPU stability
            torch.cuda.set_per_process_memory_fraction(0.7)  # Use up to 70% of GPU memory per GPU

        # Start multiple background threads for parallel training
        for i in range(self.num_training_workers):
            worker_thread = threading.Thread(target=self.background_train, daemon=True, name=f"TrainingWorker-{i}")
            worker_thread.start()
            self.training_threads.append(worker_thread)
        print(f"Started {self.num_training_workers} training worker threads")
        
    def background_train(self):
        """High-throughput training loop on dedicated GPU for maximum utilization"""
        worker_id = threading.current_thread().name
        print(f"Training thread {worker_id} started on {self.training_device}")
        
        while self.running: 
            try:
                # Get training request (blocking)
                _ = self.train_queue.get()
                
                # Process multiple training steps per request for maximum GPU utilization
                steps_per_batch = getattr(RL_CONFIG, 'training_steps_per_sample', 3)
                for _ in range(steps_per_batch):
                    if len(self.memory) >= RL_CONFIG.batch_size:
                        self._train_step_counter += 1
                        self.train_step()
                        
                        # Sync inference network every 100 training steps for responsiveness
                        if self._train_step_counter % 100 == 0:
                            self.sync_inference_network()
                    else:
                        break
                        
                self.train_queue.task_done()

            except AssertionError:
                # CRITICAL: Re-raise AssertionErrors to stop execution immediately
                raise
            except Exception as e:
                print(f"Training error in {worker_id}: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.01)
    
    def _compute_dqn_loss(self, states, actions, rewards, next_states, dones, is_weights=None):
        """
        Compiled core DQN loss computation - separate method for torch.compile optimization.
        This method handles the most computationally intensive parts of training.
        """
        # Standard DQN loss with Double DQN to reduce overestimation bias
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        with torch.no_grad():
            # Double DQN: Use local network to select actions, target network to evaluate them
            next_state_q_all = self.qnetwork_local(next_states)  # Local net selects actions
            best_actions = next_state_q_all.argmax(1, keepdim=True)
            Q_targets_next = self.qnetwork_target(next_states).gather(1, best_actions)  # Target net evaluates
            Q_targets = rewards + (RL_CONFIG.gamma * Q_targets_next * (1 - dones))
        
        # Compute TD errors
        td_errors = (Q_expected - Q_targets).abs()
        
        # Compute loss (with or without importance sampling weights)
        if is_weights is not None:
            weighted_td_loss = is_weights * (Q_expected - Q_targets).pow(2)
            loss = weighted_td_loss.mean()
        else:
            # Use MSE loss instead of SmoothL1Loss for stronger gradients
            # MSE is more sensitive to outliers and will produce larger loss values
            mse_loss = (Q_expected - Q_targets).pow(2).mean()
            loss = mse_loss
        
        return loss, td_errors, Q_expected, Q_targets

    def train_step(self):
        """Perform a single training step - optimized for single thread maximum throughput"""
        buffer_size = len(self.memory)
        
        # CRITICAL FIX #6: Enhanced cold start protection for PER
        min_buffer_size = RL_CONFIG.batch_size
        if hasattr(RL_CONFIG, 'use_per') and RL_CONFIG.use_per:
            # PER needs extra buffer to handle priority sampling effectively
            # Reasonable requirement for dual GPU setup
            min_buffer_size = max(RL_CONFIG.batch_size * 2, 12288)  # Reduced requirements for faster training start
            
        if buffer_size < min_buffer_size:
            return
            
        try:
            import time
            step_start = time.time()
            
            # Thread-safe gradient accumulation and counter management
            with self.gradient_lock:
                # Update training metrics (thread-safe)
                metrics.total_training_steps += 1
                metrics.memory_buffer_size = buffer_size
                
                # Update gradient accumulation counter
                self._gradient_accumulation_counter += 1
                current_accumulation_step = self._gradient_accumulation_counter
                
                # Check if we need to zero gradients (start of new accumulation cycle)
                accumulation_steps = getattr(RL_CONFIG, 'gradient_accumulation_steps', 1)
                should_zero_gradients = (current_accumulation_step - 1) % accumulation_steps == 0
                should_step_optimizer = current_accumulation_step % accumulation_steps == 0
            
            # Zero gradients outside lock (but coordinate via flag)
            if should_zero_gradients:
                self.optimizer.zero_grad()
                
            # If using NoisyNets, sample noise once per train step for consistency
            if getattr(RL_CONFIG, 'use_noisy_nets', False):
                if hasattr(self.qnetwork_local, 'reset_noise'):
                    self.qnetwork_local.reset_noise()
                if hasattr(self.qnetwork_target, 'reset_noise'):
                    self.qnetwork_target.reset_noise()
            
            sampling_start = time.time()
            use_per = getattr(RL_CONFIG, 'use_per', False)
            if use_per:
                # Anneal beta towards 1.0
                frame = metrics.frame_count
                beta_start = getattr(RL_CONFIG, 'per_beta_start', 0.4)
                beta_frames = max(1, getattr(RL_CONFIG, 'per_beta_frames', 200000))
                beta = min(1.0, beta_start + (1.0 - beta_start) * (frame / beta_frames))
                
                # DEBUG: Log beta calculation
                # if metrics.total_training_steps % 1000 == 0:
                #     print(f"PER Beta: {beta:.4f} (frame {frame}, beta_frames {beta_frames})")
                
                states, actions, rewards, next_states, dones, is_weights, indices = self.memory.sample(RL_CONFIG.batch_size, beta=beta)
            else:
                states, actions, rewards, next_states, dones = self.memory.sample(RL_CONFIG.batch_size)
            sampling_time = time.time() - sampling_start
            
            forward_start = time.time()
            if getattr(RL_CONFIG, 'use_distributional', False):
                # Distributional QR loss
                num_quant = getattr(RL_CONFIG, 'num_atoms', 51)
                # Current quantiles for taken actions: (B, N)
                z_pred_all = self.qnetwork_local(states)  # (B, A, N)
                batch_indices = torch.arange(z_pred_all.size(0), device=states.device)
                z_pred = z_pred_all[batch_indices, actions.squeeze(1)]  # (B, N)

                with torch.no_grad():
                    # Next action by mean over quantiles of local net
                    z_next_local = self.qnetwork_local(next_states)  # (B, A, N)
                    q_next_local = z_next_local.mean(dim=2)  # (B, A)
                    next_actions = q_next_local.argmax(dim=1)  # (B,)
                    # Target quantiles for those actions from target net
                    z_next_target_all = self.qnetwork_target(next_states)  # (B, A, N)
                    z_next_target = z_next_target_all[batch_indices, next_actions]  # (B, N)
                    # Bellman update for quantiles
                    z_target = rewards + (RL_CONFIG.gamma * (1 - dones)) * z_next_target  # (B, N)

                # Pairwise TD errors: (B, N_pred, N_tgt) => (B, N, N)
                z_pred_exp = z_pred.unsqueeze(2)
                z_target_exp = z_target.unsqueeze(1)
                td = z_target_exp - z_pred_exp  # (B, N, N)

                # Huber loss
                huber_k = 1.0
                abs_td = td.abs()
                huber_loss = torch.where(abs_td <= huber_k, 0.5 * td.pow(2), huber_k * (abs_td - 0.5 * huber_k))

                # Quantile weights
                with torch.no_grad():
                    taus = (torch.arange(num_quant, device=states.device, dtype=states.dtype) + 0.5) / num_quant  # (N,)
                taus = taus.view(1, -1, 1)  # (1, N, 1)
                quantile_weight = (taus - (td.detach() < 0).float()).abs()
                qr_loss = (quantile_weight * huber_loss).mean(dim=2).sum(dim=1).mean()

                loss = qr_loss if not use_per else (is_weights.view(-1) * (quantile_weight * huber_loss).mean(dim=2).sum(dim=1)).mean()

                # For PER priority updates, define td_errors as mean absolute TD per sample
                td_errors = td.abs().mean(dim=(1, 2))  # CRITICAL FIX: Use abs() for priority updates
            else:
                # Standard DQN loss computation - use compiled version if available
                if self._compiled_loss_computation is not None:
                    # Use compiled loss computation for performance
                    is_weights_tensor = is_weights if use_per else None
                    loss, td_errors, Q_expected, Q_targets = self._compiled_loss_computation(
                        states, actions, rewards, next_states, dones, is_weights_tensor)
                else:
                    # Fallback to original computation if compilation failed
                    loss, td_errors, Q_expected, Q_targets = self._compute_dqn_loss(
                        states, actions, rewards, next_states, dones, 
                        is_weights if use_per else None)
                
                # DIAGNOSTIC: Track Q-value statistics (extracted from compiled computation)
                q_exp_max = Q_expected.max().item()
                q_exp_min = Q_expected.min().item()
                q_exp_mean = Q_expected.mean().item()
                q_exp_std = Q_expected.std().item()
                
                # STRICT ASSERTION: Q_expected should be reasonable (not exploded)
                assert abs(q_exp_max) < 30.0 and abs(q_exp_min) < 30.0, f"CRITICAL BUG: Q_expected explosion! Q_expected range: [{q_exp_min:.3f}, {q_exp_max:.3f}]. Neural network is broken - restart with fresh model."
                
                # Additional diagnostic checks for targets and rewards
                q_tgt_max = Q_targets.max().item()
                q_tgt_min = Q_targets.min().item()
                q_tgt_mean = Q_targets.mean().item()
                
                reward_max = rewards.max().item()
                reward_min = rewards.min().item()
                assert abs(reward_max) < 50.0 and abs(reward_min) < 50.0, f"CRITICAL BUG: Reward corruption! Reward range: [{reward_min:.3f}, {reward_max:.3f}]"

                if use_per:
                    # DIAGNOSTIC: Track TD error distribution before explosion check
                    td_max = td_errors.max().item()
                    td_min = td_errors.min().item()
                    td_mean = td_errors.mean().item()
                    td_std = td_errors.std().item()
                    
                    # DIAGNOSTIC: Track priority distribution health
                    if hasattr(self.memory, 'priorities'):
                        current_priorities = self.memory.priorities[:len(self.memory), 0]
                        prio_max = current_priorities.max()
                        prio_mean = current_priorities.mean()
                        high_prio_count = np.sum(current_priorities > td_mean * 3)  # Count priorities > 3x mean TD error
                    
                    # STRICT ASSERTION: TD errors should be reasonable (catch Q-value explosion)
                    max_td_error = td_errors.max().item()
                    if max_td_error >= 25.0:
                        print(f"\n!!! CRITICAL BUG DETECTED !!!")
                        print(f"TD error explosion! Max TD error = {max_td_error:.6f}")
                        print(f"Q_expected: [{q_exp_min:.3f}, {q_exp_max:.3f}], mean={q_exp_mean:.3f}")
                        print(f"Q_targets: [{q_tgt_min:.3f}, {q_tgt_max:.3f}], mean={q_tgt_mean:.3f}")
                        print(f"Rewards: [{rewards.min().item():.3f}, {rewards.max().item():.3f}]")
                        print(f"This indicates Q-value explosion - neural network is broken.")
                        print(f"FORCING APPLICATION SHUTDOWN...")
                        import os
                        os._exit(1)  # Force immediate process termination
                    
                    # DEBUG: Check for zero loss issues
                    if loss.item() < 1e-8:
                        print(f"PER ZERO LOSS DEBUG:")
                        print(f"  Q_expected range: [{Q_expected.min():.4f}, {Q_expected.max():.4f}]")
                        print(f"  Q_targets range: [{Q_targets.min():.4f}, {Q_targets.max():.4f}]")  
                        print(f"  TD errors range: [{td_errors.min():.4f}, {td_errors.max():.4f}]")
                        print(f"  IS weights range: [{is_weights.min():.4f}, {is_weights.max():.4f}]")
                        print(f"  Final loss: {loss.item():.8f}")
            forward_time = time.time() - forward_start

            # Calculate loss scaling for gradient accumulation
            accumulation_steps = getattr(RL_CONFIG, 'gradient_accumulation_steps', 1)
            loss = loss / accumulation_steps  # Scale loss for accumulation
            
            # Backward pass with mixed precision support (can be done in parallel)
            backward_start = time.time()
            if self.use_mixed_precision and self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            backward_time = time.time() - backward_start
            
            # Thread-safe optimizer stepping - CRITICAL SECTION
            if should_step_optimizer:
                with self.training_lock:  # Serialize optimizer updates across all threads
                    optimizer_start = time.time()
                    
                    # DIAGNOSTIC: Track gradient norms before clipping
                    total_grad_norm = 0.0
                    param_count = 0
                    for param in self.qnetwork_local.parameters():
                        if param.grad is not None:
                            param_norm = param.grad.data.norm(2)
                            total_grad_norm += param_norm.item() ** 2
                            param_count += 1
                    total_grad_norm = total_grad_norm ** (1. / 2)
                    
                    # Enable gradient norm logging to diagnose vanishing gradients
                    if metrics.total_training_steps % 1000 == 0:
                        print(f"Step {metrics.total_training_steps}: Gradient norm: {total_grad_norm:.6f}, Loss: {(loss * accumulation_steps).item():.6f}")
                    
                    if self.use_mixed_precision and self.scaler is not None:
                        # Gradient clipping with mixed precision
                        self.scaler.unscale_(self.optimizer)
                        clipped_grad_norm = torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), max_norm=10.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        # Standard gradient clipping and optimization
                        clipped_grad_norm = torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), max_norm=10.0)
                        self.optimizer.step()
                    optimizer_time = time.time() - optimizer_start
            else:
                optimizer_time = 0
        
            # Always update loss metrics (use unscaled loss for reporting)
            metrics.losses.append((loss * accumulation_steps).item())

            # PER: update priorities and track average
            if use_per:
                with torch.no_grad():
                    self.memory.update_priorities(indices, td_errors)
                metrics.average_priority = self.memory.avg_priority
                
                # DEBUG: Validate priorities periodically
                # if metrics.total_training_steps % 1000 == 0:
                #     self.memory.validate_priorities()            # Only update target network and scheduler on actual optimizer steps
            if should_step_optimizer:
                # Target network and scheduler updates are already inside training_lock above
                # Soft target update if enabled (Polyak averaging every step)
                if getattr(RL_CONFIG, 'use_soft_target', False):
                    tau = getattr(RL_CONFIG, 'tau', 0.005)
                    self.soft_update(tau)
                else:
                    # Hard target network update every target_update_freq steps
                    target_freq = getattr(RL_CONFIG, 'target_update_freq', 200)
                    if metrics.total_training_steps % target_freq == 0:
                        self.update_target_network()

                # Step LR scheduler if enabled
                if self.scheduler is not None:
                    self.scheduler.step()
            
            total_time = time.time() - step_start
            
            # Print timing every 100 steps for debugging
            # if metrics.total_training_steps % 100 == 0:
            #    print(f"Training step timing - Total: {total_time*1000:.1f}ms, "
            #          f"Sampling: {sampling_time*1000:.1f}ms, Forward: {forward_time*1000:.1f}ms, "
            #          f"Backward: {backward_time*1000:.1f}ms, Optimizer: {optimizer_time*1000:.1f}ms")
                    
        except AssertionError:
            # CRITICAL: Re-raise AssertionErrors to stop execution immediately
            raise
        except Exception as e:
            print(f"Training error: {e}")
            import traceback
            traceback.print_exc()

    def update_target_network(self):
        """Update target network with weights from local network"""
        # Track when target network was last updated
        metrics.last_target_update_frame = metrics.frame_count
        
        if getattr(RL_CONFIG, 'use_soft_target', False):
            tau = getattr(RL_CONFIG, 'tau', 0.005)
            self.soft_update(tau)
        else:
            # Handle cross-device state dict copying
            local_state_dict = self.qnetwork_local.state_dict()
            # Get device from first parameter
            local_device = next(self.qnetwork_local.parameters()).device
            target_device = next(self.qnetwork_target.parameters()).device
            
            if target_device != local_device:
                # Move state dict to target device
                target_state_dict = {}
                for key, param in local_state_dict.items():
                    target_state_dict[key] = param.to(target_device)
                self.qnetwork_target.load_state_dict(target_state_dict)
            else:
                # Same device - direct load
                self.qnetwork_target.load_state_dict(local_state_dict)

    def soft_update(self, tau: float = 0.005):
        """Soft-update target network parameters: θ_target = τ*θ_local + (1-τ)*θ_target"""
        with torch.no_grad():
            for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
                # Handle cross-device parameter updates
                if target_param.device != local_param.device:
                    # Move local parameter to target device for computation
                    local_data = local_param.data.to(target_param.device)
                    target_param.data.copy_(tau * local_data + (1.0 - tau) * target_param.data)
                else:
                    # Same device - direct update
                    target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def sync_inference_network(self):
        """Synchronize inference network with training updates"""
        if hasattr(self, 'qnetwork_inference') and self.qnetwork_inference != self.qnetwork_local:
            # Copy weights from training network to inference network across devices
            with torch.no_grad():
                inference_state_dict = {}
                for key, param in self.qnetwork_local.state_dict().items():
                    inference_state_dict[key] = param.to(self.inference_device)
                self.qnetwork_inference.load_state_dict(inference_state_dict)
    
    def get_q_value_range(self):
        """Get current Q-value range from the network for monitoring"""
        try:
            with torch.no_grad():
                # Test Q-values on a dummy state using inference network
                dummy_state = torch.zeros(1, self.state_size).to(self.inference_device)
                self.qnetwork_inference.eval()
                
                if getattr(RL_CONFIG, 'use_distributional', False):
                    q_dist = self.qnetwork_inference(dummy_state)  # (1, A, N)
                    test_q_values = q_dist.mean(dim=2)   # (1, A)
                else:
                    test_q_values = self.qnetwork_inference(dummy_state)
                
                self.qnetwork_inference.train()
                
                max_q = test_q_values.max().item()
                min_q = test_q_values.min().item()
                return min_q, max_q
        except Exception:
            return float('nan'), float('nan')

    def act(self, state, epsilon=0.0):
        """Select action using epsilon-greedy policy - optimized for low-latency inference"""
        # Convert state to tensor on inference device
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.inference_device)
        
        # Set evaluation mode
        self.qnetwork_inference.eval()
        
        with torch.no_grad():
            if getattr(RL_CONFIG, 'use_distributional', False):
                q_dist = self.qnetwork_inference(state)  # (1, A, N)
                action_values = q_dist.mean(dim=2)   # (1, A)
            else:
                action_values = self.qnetwork_inference(state)
            
        # Set training mode back
        self.qnetwork_inference.train()
        
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            return random.randrange(self.action_size)
        else:
            return action_values.cpu().data.numpy().argmax()
            
    def step(self, state, action, reward, next_state, done):
        """Add experience to memory and queue for training"""
        # Save experience in replay memory
        self.memory.push(state, action, reward, next_state, done)
        
        # BALANCED TRAINING: Moderate training intensity to maintain client responsiveness
        # This balances learning speed with inference performance
        training_multiplier = getattr(RL_CONFIG, 'training_steps_per_sample', 3)  # Use config value (3)
        for _ in range(training_multiplier):  # Reasonable training frequency
            try:
                self.train_queue.put_nowait(True)
            except queue.Full:
                break  # Queue full - stop trying
            
    def save(self, filename):
        """Save model weights, rate-limited unless forced."""
        is_forced_save = "exit" in filename or "shutdown" in filename
        current_time = time.time()
        save_interval = 30.0  # Minimum seconds between saves
        # Rate limit non-forced saves
        if not is_forced_save:
            if current_time - self.last_save_time < save_interval:
                return  # Skip save
        
        # Proceed with save if forced or interval elapsed
        try:
            # Determine the actual expert ratio to save (not the override value)
            if metrics.expert_mode or metrics.override_expert:
                ratio_to_save = metrics.saved_expert_ratio
            else:
                ratio_to_save = metrics.expert_ratio

            payload = {
                'policy_state_dict': self.qnetwork_local.state_dict(),
                'target_state_dict': self.qnetwork_target.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'memory_size': len(self.memory),
                'epsilon': metrics.epsilon,
                'frame_count': metrics.frame_count,
                'expert_ratio': ratio_to_save, # Save the determined ratio
                'last_decay_step': metrics.last_decay_step,
                'last_epsilon_decay_step': metrics.last_epsilon_decay_step
            }
            # Save scheduler state if present
            if self.scheduler is not None:
                payload['scheduler_state_dict'] = self.scheduler.state_dict()
            torch.save(payload, filename)
            
            # Update last save time ONLY on successful save
            self.last_save_time = current_time
            
            # Only print on forced exit/shutdown saves
            if is_forced_save:
                print(f"Model saved to {filename} (frame {metrics.frame_count}, expert ratio {ratio_to_save:.2f})")
        except Exception as e:
            print(f"ERROR saving model to {filename}: {e}")

    def load(self, filename):
        """Load model weights"""
        from config import FORCE_FRESH_MODEL
        
        if FORCE_FRESH_MODEL:
            print(f"FORCE_FRESH_MODEL is True - skipping model loading and starting with fresh weights")
            return
            
        if os.path.exists(filename):
            try:
                print(f"Loading model checkpoint from {filename}")
                checkpoint = torch.load(filename, map_location=training_device)
                policy_sd = checkpoint['policy_state_dict']
                target_sd = checkpoint['target_state_dict']

                def _try_load(model: nn.Module, sd: dict) -> bool:
                    try:
                        model.load_state_dict(sd)
                        return True
                    except Exception:
                        return False

                ok_local = _try_load(self.qnetwork_local, policy_sd)
                ok_target = _try_load(self.qnetwork_target, target_sd)

                used_compat = False
                if not (ok_local and ok_target):
                    # Compatibility path: map Linear <-> NoisyLinear differences
                    def _compat_remap(model: nn.Module, sd: dict) -> dict:
                        model_sd = model.state_dict()
                        new_sd = {}
                        for k in model_sd.keys():
                            if k in sd:
                                new_sd[k] = sd[k]
                                continue
                            # Noisy expecting mu from Linear
                            if k.endswith('weight_mu'):
                                base = k[:-3]  # strip '_mu' -> 'weight'
                                if base in sd:
                                    new_sd[k] = sd[base]
                                    continue
                            if k.endswith('bias_mu'):
                                base = k[:-3]  # strip '_mu' -> 'bias'
                                if base in sd:
                                    new_sd[k] = sd[base]
                                    continue
                            # Linear expecting weight/bias from Noisy mu
                            if k.endswith('weight') and (k + '_mu') in sd:
                                new_sd[k] = sd[k + '_mu']
                                continue
                            if k.endswith('bias') and (k + '_mu') in sd:
                                new_sd[k] = sd[k + '_mu']
                                continue
                            # Otherwise keep initialized tensor (for sigma/eps etc.)
                            new_sd[k] = model_sd[k]
                        return new_sd

                    remapped_policy = _compat_remap(self.qnetwork_local, policy_sd)
                    remapped_target = _compat_remap(self.qnetwork_target, target_sd)
                    self.qnetwork_local.load_state_dict(remapped_policy, strict=False)
                    self.qnetwork_target.load_state_dict(remapped_target, strict=False)
                    used_compat = True

                if used_compat:
                    print("Checkpoint loaded with compatibility remap (Linear<->Noisy).")
                else:
                    print("Checkpoint loaded with strict key match.")

                # Optimizer/Scheduler may fail to load if param shapes changed; guard it
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print("Optimizer state loaded.")
                    
                    # CRITICAL FIX: Override loaded learning rate with current config value
                    # This ensures config changes actually take effect!
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = RL_CONFIG.lr
                    print(f"Learning rate overridden to config value: {RL_CONFIG.lr}")
                    
                except Exception:
                    print("Optimizer state not loaded (shape mismatch). Using freshly initialized optimizer.")
                # Load scheduler state if present and configured
                if 'scheduler_state_dict' in checkpoint and getattr(RL_CONFIG, 'use_lr_scheduler', False):
                    if self.scheduler is None:
                        step_size = getattr(RL_CONFIG, 'scheduler_step_size', 100000)
                        gamma = getattr(RL_CONFIG, 'scheduler_gamma', 0.5)
                        self.scheduler = StepLR(self.optimizer, step_size=step_size, gamma=gamma)
                    try:
                        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                        print("LR scheduler state loaded.")
                    except Exception:
                        print("LR scheduler state not loaded (mismatch). Using freshly initialized scheduler.")
                
                # Load training state (frame count, epsilon, expert ratio, decay step)
                metrics.frame_count = checkpoint.get('frame_count', 0)
                
                # CRITICAL FIX: Detect and fix Q-value explosion in loaded model
                print("Checking loaded model for Q-value corruption...")
                with torch.no_grad():
                    # Test Q-values on a dummy state
                    dummy_state = torch.zeros(1, self.state_size).to(device)
                    test_q_values = self.qnetwork_local(dummy_state)
                    max_q = test_q_values.max().item()
                    min_q = test_q_values.min().item()
                    
                    print(f"Loaded model Q-value range: [{min_q:.3f}, {max_q:.3f}]")
                    
                    # If Q-values are corrupted (too large), rescale the network weights
                    if abs(max_q) > 10.0 or abs(min_q) > 10.0:
                        print(f"WARNING: Loaded model has corrupted Q-values! Rescaling network weights...")
                        scale_factor = 5.0 / max(abs(max_q), abs(min_q))  # Scale to max ±5.0
                        print(f"Applying scale factor: {scale_factor:.4f}")
                        
                        # Rescale all weights in both networks
                        for param in self.qnetwork_local.parameters():
                            param.data *= scale_factor
                        for param in self.qnetwork_target.parameters():
                            param.data *= scale_factor
                            
                        # Verify the fix
                        test_q_values_fixed = self.qnetwork_local(dummy_state)
                        max_q_fixed = test_q_values_fixed.max().item()
                        min_q_fixed = test_q_values_fixed.min().item()
                        print(f"Rescaled Q-value range: [{min_q_fixed:.3f}, {max_q_fixed:.3f}]")
                    else:
                        print("Model Q-values are in acceptable range.")
                
                # Load or reset metrics based on RESET_METRICS flag from config
                if RESET_METRICS:
                    print("RESET_METRICS flag is True. Resetting epsilon/expert_ratio.")
                    metrics.epsilon = RL_CONFIG.epsilon_start
                    metrics.expert_ratio = SERVER_CONFIG.expert_ratio_start
                    metrics.last_decay_step = 0
                    metrics.last_epsilon_decay_step = 0 # Reset epsilon step tracker
                else:
                    # Load metrics from checkpoint
                    metrics.epsilon = checkpoint.get('epsilon', RL_CONFIG.epsilon_start)
                    metrics.expert_ratio = checkpoint.get('expert_ratio', SERVER_CONFIG.expert_ratio_start)
                    metrics.last_decay_step = checkpoint.get('last_decay_step', 0)
                    metrics.last_epsilon_decay_step = checkpoint.get('last_epsilon_decay_step', 0) # Load epsilon step tracker

                # Respect server flags to reset at startup regardless of checkpoint values
                if getattr(SERVER_CONFIG, 'reset_expert_ratio', False):
                    print(f"Resetting expert_ratio to start value per SERVER_CONFIG: {SERVER_CONFIG.expert_ratio_start}")
                    metrics.expert_ratio = SERVER_CONFIG.expert_ratio_start
                if getattr(SERVER_CONFIG, 'reset_epsilon', False):
                    print(f"Resetting epsilon to start value per SERVER_CONFIG: {RL_CONFIG.epsilon_start}")
                    metrics.epsilon = RL_CONFIG.epsilon_start
                    metrics.last_epsilon_decay_step = 0
                # One-time flag to force expert ratio recalculation based on current training progress
                if getattr(SERVER_CONFIG, 'force_expert_ratio_recalc', False):
                    print(f"Force recalculating expert_ratio based on {metrics.frame_count:,} frames...")
                    decay_expert_ratio(metrics.frame_count)
                    print(f"Expert ratio recalculated to: {metrics.expert_ratio:.4f}")
                if getattr(SERVER_CONFIG, 'reset_frame_count', False):
                    print("Resetting frame count/epsilon per SERVER_CONFIG.")
                    metrics.frame_count = 0
                    metrics.last_decay_step = 0
                    metrics.last_epsilon_decay_step = 0
                    metrics.epsilon = RL_CONFIG.epsilon_start
                                
                print(f"Loaded model from {filename}")
                print(f"  - Resuming from frame: {metrics.frame_count}")
                print(f"  - Resuming epsilon: {metrics.epsilon:.4f}")
                print(f"  - Resuming expert_ratio: {metrics.expert_ratio:.4f}")
                print(f"  - Resuming last_decay_step: {metrics.last_decay_step}")

                return True
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Starting with fresh model parameters.")
                return False
        else:
            print(f"No checkpoint found at {filename}. Starting new model.")
        return False

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
            
    def update_epsilon(self):
        with self.lock:
            self.metrics.epsilon = decay_epsilon(self.metrics.frame_count)
            return self.metrics.epsilon
            
    def update_expert_ratio(self):
        with self.lock:
            decay_expert_ratio(self.metrics.frame_count)
            return self.metrics.expert_ratio
    
    def add_episode_reward(self, total_reward, dqn_reward, expert_reward):
        with self.lock:
            if total_reward > 0:
                self.metrics.episode_rewards.append(total_reward)
            if dqn_reward > 0:
                self.metrics.dqn_rewards.append(dqn_reward)
            if expert_reward > 0:
                self.metrics.expert_rewards.append(expert_reward)
    
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

def parse_frame_data(data: bytes) -> Optional[FrameData]:
    """Parse binary frame data from Lua into game state - SIMPLIFIED with float32 payload"""
    try:
        if not data or len(data) < 10:  # Minimal size check
            print("ERROR: Received empty or too small data packet", flush=True)
            sys.exit(1)
        
        # Fixed OOB header format (must match Lua exactly). Precompute once.
        # Format: ">HdBBBHHHBBBhBhBBBBBffffff"
        global _FMT_OOB, _HDR_OOB
        try:
            _FMT_OOB
        except NameError:
            _FMT_OOB = ">HdBBBHHHBBBhBhBBBBBffffff"
            _HDR_OOB = struct.calcsize(_FMT_OOB)

        if len(data) < _HDR_OOB:
            print(f"ERROR: Received data too small: {len(data)} bytes, need {_HDR_OOB}", flush=True)
            sys.exit(1)

        values = struct.unpack(_FMT_OOB, data[:_HDR_OOB])
        (num_values, reward, gamestate, game_mode, done, frame_counter, score_high, score_low,
         save_signal, fire, zap, spinner, is_attract, nearest_enemy, player_seg, is_open,
         expert_fire, expert_zap, level_number,
         rc_safety, rc_proximity, rc_shots, rc_threats, rc_pulsar, rc_score) = values
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
            rc_safety=float(rc_safety),
            rc_proximity=float(rc_proximity),
            rc_shots=float(rc_shots),
            rc_threats=float(rc_threats),
            rc_pulsar=float(rc_pulsar),
            rc_score=float(rc_score)
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
        f"{'Epsilon':>7} | {'Guided %':>8} | {'Mem Size':>8} | {'Avg Level':>9} | {'Level Type':>10} | {'Override':>10} | {'Q-Value Range':>14}"
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
    override_status = "OFF"
    if metrics.override_expert:
        override_status = "SELF"
    elif metrics.expert_mode:
        override_status = "BOT"
    
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
        f"{metrics.frame_count:8d} | {metrics.fps:5.1f} | {client_count:7d} | {mean_reward:12.2f} | {dqn_reward:10.2f} | "
        f"{mean_loss:8.2f} | {metrics.epsilon:7.3f} | {guided_ratio*100:7.2f}% | "
        f"{mem_size:8d} | {display_level:9.2f} | {'Open' if metrics.open_level else 'Closed':10} | {override_status:10} | {q_range:>14}"
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

def expert_action_to_index(fire, zap, spinner):
    """Convert continuous expert actions to discrete action index"""
    if zap:
        return 14  # Special case for zap action
        
    # Clamp spinner value between -0.3 and 0.3 to match ACTION_MAPPING
    spinner_value = max(-0.3, min(0.3, spinner))
    
    # Map spinner to 0-6 range based on ACTION_MAPPING values
    # -0.3 -> 0, -0.2 -> 1, -0.1 -> 2, 0.0 -> 3, 0.1 -> 4, 0.2 -> 5, 0.3 -> 6
    spinner_idx = int((spinner_value + 0.3) / 0.1)
    spinner_idx = min(6, max(0, spinner_idx))  # Ensure we don't exceed valid range
    
    # If firing, offset by 7 to get into the firing action range (7-13)
    base_idx = 7 if fire else 0
    
    return base_idx + spinner_idx

def encode_action_to_game(fire, zap, spinner):
    """Convert action values to game-compatible format"""
    spinner_val = spinner * 31
    return int(fire), int(zap), int(spinner_val)

def decay_epsilon(frame_count):
    """Calculate decayed exploration rate using step-based decay."""
    step_interval = frame_count // RL_CONFIG.epsilon_decay_steps
    
    # Only decay if a new step interval is reached
    if step_interval > metrics.last_epsilon_decay_step:
        # Apply decay multiplicatively for the number of steps missed
        num_steps_to_apply = step_interval - metrics.last_epsilon_decay_step
        decay_multiplier = RL_CONFIG.epsilon_decay_factor ** num_steps_to_apply
        metrics.epsilon *= decay_multiplier
        
        # Update the last step tracker
        metrics.last_epsilon_decay_step = step_interval

        # Ensure epsilon doesn't go below the minimum effective exploration rate
        metrics.epsilon = max(RL_CONFIG.epsilon_end, metrics.epsilon)

    # Always return the current epsilon value (which might have just been decayed)
    return metrics.epsilon

def decay_expert_ratio(current_step):
    """Update expert ratio based on 10,000 frame intervals"""
    # Skip decay if expert mode is active
    if metrics.expert_mode:
        return metrics.expert_ratio
    
    # DON'T auto-initialize to start value at frame 0 - respect loaded checkpoint values
    # Only initialize if expert_ratio is somehow invalid (negative or > 1)
    if current_step == 0 and (metrics.expert_ratio < 0 or metrics.expert_ratio > 1):
        metrics.expert_ratio = SERVER_CONFIG.expert_ratio_start
        metrics.last_decay_step = 0
        return metrics.expert_ratio

    step_interval = current_step // SERVER_CONFIG.expert_ratio_decay_steps

    # Only update if we've moved to a new interval
    if step_interval > metrics.last_decay_step:
        # Apply decay for each step we've advanced
        steps_to_apply = step_interval - metrics.last_decay_step
        for _ in range(steps_to_apply):
            metrics.expert_ratio *= SERVER_CONFIG.expert_ratio_decay
        
        metrics.last_decay_step = step_interval

        # Ensure we don't go below the minimum
        metrics.expert_ratio = max(SERVER_CONFIG.expert_ratio_min, metrics.expert_ratio)

    return metrics.expert_ratio