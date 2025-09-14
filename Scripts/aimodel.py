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

# Initialize devices (use single device for both inference and training for stability)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU (CUDA) for both inference and training: {device}")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"Using MPS for both inference and training: {device}")
else:
    device = torch.device("cpu")
    print(f"Using CPU for both inference and training: {device}")

# For compatibility with dual-device code
training_device = device
inference_device = device

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
    """Deep Q-Network model - Balanced architecture for good GPU utilization without starving inference."""
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        use_noisy = RL_CONFIG.use_noisy_nets
        use_dueling = getattr(RL_CONFIG, 'use_dueling', False)
        noisy_std = getattr(RL_CONFIG, 'noisy_std_init', 0.5)
        hidden_size = RL_CONFIG.hidden_size  # 2048 - reasonable size
        LinearOrNoisy = NoisyLinear if use_noisy else nn.Linear

        # Balanced feature extractor - slightly larger for better GPU utilization
        self.fc1 = nn.Linear(state_size, hidden_size)         # 176 -> 2560  
        self.fc2 = nn.Linear(hidden_size, hidden_size)        # 2560 -> 2560
        self.fc3 = nn.Linear(hidden_size, hidden_size//2)     # 2560 -> 1280
        self.fc4 = nn.Linear(hidden_size//2, hidden_size//4)  # 1280 -> 640
        
        # Apply noisy layers if enabled
        if use_noisy:
            self.fc2 = NoisyLinear(hidden_size, hidden_size, noisy_std)
            self.fc3 = NoisyLinear(hidden_size, hidden_size//2, noisy_std)

        self.use_dueling = use_dueling
        if use_dueling:
            # Dueling streams with reasonable sizes
            self.val_fc = nn.Linear(hidden_size//4, hidden_size//8)  # 640 -> 320
            self.adv_fc = nn.Linear(hidden_size//4, hidden_size//8)  # 640 -> 320
            self.val_out = nn.Linear(hidden_size//8, 1)              # 320 -> 1
            self.adv_out = nn.Linear(hidden_size//8, action_size)    # 320 -> 18
        else:
            # Single stream with reasonable sizes
            self.fc5 = nn.Linear(hidden_size//4, hidden_size//8)     # 640 -> 320
            self.out = nn.Linear(hidden_size//8, action_size)        # 320 -> 18

    def reset_noise(self):
        # Reset noise for all NoisyLinear layers
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def forward(self, x):
        # Balanced feature extraction - slightly more compute for better GPU utilization
        x = F.relu(self.fc1(x))    # 176 -> 2560
        x = F.relu(self.fc2(x))    # 2560 -> 2560  
        x = F.relu(self.fc3(x))    # 2560 -> 1280
        x = F.relu(self.fc4(x))    # 1280 -> 640
        
        if self.use_dueling:
            v = F.relu(self.val_fc(x))  # 640 -> 320
            v = self.val_out(v)         # 320 -> 1
            a = F.relu(self.adv_fc(x))  # 640 -> 320
            a = self.adv_out(a)         # 320 -> 18
            # Subtract mean advantage to keep Q identifiable
            q = v + (a - a.mean(dim=1, keepdim=True))
            return q
        else:
            x = F.relu(self.fc5(x))     # 640 -> 320
            return self.out(x)          # 320 -> 18

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
        self.priorities = np.zeros((capacity, 1), dtype=np.float32)

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
        # Max priority so new samples are seen at least once
        max_prio = self.priorities.max() if self.pos > 0 or self.full else 1.0
        if max_prio == 0:
            max_prio = 1.0
        self.priorities[idx, 0] = max_prio

        # Update position
        self.pos = (self.pos + 1) % self.capacity
        if self.pos == 0:
            self.full = True

    def sample(self, batch_size: int, beta: float = 0.4):
        size = len(self)
        if size == 0:
            raise ValueError("Cannot sample from empty memory buffer")
            
        prios = self.priorities[:size, 0]
        # Add epsilon to avoid zero-probability entries
        probs = (prios + self.eps) ** self.alpha
        probs_sum = probs.sum()
        if probs_sum <= 0:
            # Fallback to uniform if all priorities are zero
            probs = np.ones_like(probs) / max(1, len(probs))
        else:
            probs = probs / probs_sum
            
        # CRITICAL FIX: Ensure we don't try to sample more than available
        k = min(batch_size, size)
        if k <= 0:
            raise ValueError(f"Invalid sample size: k={k}, size={size}")
            
        # CRITICAL FIX: Handle case where we need replacement due to small buffer
        replace_needed = k > size
        indices = np.random.choice(size, k, p=probs, replace=replace_needed)
        
        # IS weights
        weights = (size * probs[indices]) ** (-beta)
        weights = weights / weights.max()
        
        # Tensors
        states = torch.from_numpy(self.states[indices]).float().to(device)
        actions = torch.from_numpy(self.actions[indices]).long().to(device)
        rewards = torch.from_numpy(self.rewards[indices]).float().to(device)
        next_states = torch.from_numpy(self.next_states[indices]).float().to(device)
        dones = torch.from_numpy(self.dones[indices]).float().to(device)
        is_weights = torch.from_numpy(weights.reshape(-1, 1)).float().to(device)
        return states, actions, rewards, next_states, dones, is_weights, indices

    def update_priorities(self, indices, td_errors):
        td = td_errors.detach().abs().cpu().numpy().reshape(-1) + self.eps
        self.priorities[indices, 0] = td
        # Track average
        self.avg_priority = float(td.mean())

class DQNAgent:
    """DQN Agent with experience replay and target network"""
    def __init__(self, state_size, action_size, learning_rate=RL_CONFIG.lr, gamma=RL_CONFIG.gamma, 
                 epsilon=RL_CONFIG.epsilon, epsilon_min=RL_CONFIG.epsilon_min, 
                 memory_size=RL_CONFIG.memory_size, batch_size=RL_CONFIG.batch_size):
        self.state_size = state_size
        self.action_size = action_size
        self.last_save_time = 0.0 # Initialize last save time
        
        # Q-Networks on single device for stability
        if getattr(RL_CONFIG, 'use_distributional', False):
            n_quant = getattr(RL_CONFIG, 'num_atoms', 51)
            self.qnetwork_local = QRDQN(state_size, action_size, n_quant).to(device)
            self.qnetwork_target = QRDQN(state_size, action_size, n_quant).to(device)
        else:
            self.qnetwork_local = DQN(state_size, action_size).to(device)
            self.qnetwork_target = DQN(state_size, action_size).to(device)
        
        # DISABLED torch.compile() due to CUDA graph issues
        # Will focus on other optimizations instead
        print("Using standard (uncompiled) neural networks for stability.")
        
        # Store device references (now both are the same)
        self.inference_device = device
        self.training_device = device
        
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
        # Simple approach: single training thread for maximum throughput
        self.num_training_workers = 1  # Single worker to avoid gradient race conditions
        self.training_threads = []
        self._train_step_counter = 0
        self._gradient_accumulation_counter = 0  # Track gradient accumulation steps
        self.training_lock = threading.Lock()  # Add lock for thread safety
        
        # Mixed precision training setup
        self.use_mixed_precision = getattr(RL_CONFIG, 'use_mixed_precision', False)
        if self.use_mixed_precision and torch.cuda.is_available():
            self.scaler = GradScaler()
            print("Mixed precision training enabled")
        else:
            self.scaler = None
        
        # CRITICAL FIX: Initialize training enabled flag to ensure training happens
        self.training_enabled = True

        # Memory optimizations for aggressive training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear any existing cache
            torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
            torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
            # Additional GPU optimizations for maximum utilization
            torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for faster matmul on Ampere+
            torch.backends.cudnn.allow_tf32 = True        # Enable TF32 for cuDNN
            # Increase GPU memory allocation efficiency
            torch.cuda.set_per_process_memory_fraction(0.9)  # Use up to 90% of GPU memory

        # Start multiple background threads for parallel training
        for i in range(self.num_training_workers):
            worker_thread = threading.Thread(target=self.background_train, daemon=True, name=f"TrainingWorker-{i}")
            worker_thread.start()
            self.training_threads.append(worker_thread)
        print(f"Started {self.num_training_workers} training worker threads")
        
    def background_train(self):
        """High-throughput single-threaded training loop for maximum GPU utilization"""
        worker_id = threading.current_thread().name
        print(f"Training thread {worker_id} started on {device}")
        
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
                    else:
                        break
                        
                self.train_queue.task_done()

            except Exception as e:
                print(f"Training error in {worker_id}: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.01)

    def train_step(self):
        """Perform a single training step - optimized for single thread maximum throughput"""
        if len(self.memory) < RL_CONFIG.batch_size:
            return
            
        try:
            import time
            step_start = time.time()
            
            # Update training metrics
            metrics.total_training_steps += 1
            metrics.memory_buffer_size = len(self.memory)
                
            # Zero gradients at the start of accumulation cycle
            accumulation_steps = getattr(RL_CONFIG, 'gradient_accumulation_steps', 1)
            if self._gradient_accumulation_counter % accumulation_steps == 0:
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
                # Standard DQN loss
                Q_expected = self.qnetwork_local(states).gather(1, actions)
                with torch.no_grad():
                    best_actions = self.qnetwork_local(next_states).argmax(1, keepdim=True)
                    Q_targets_next = self.qnetwork_target(next_states).gather(1, best_actions)
                    Q_targets = rewards + (RL_CONFIG.gamma * Q_targets_next * (1 - dones))

                if use_per:
                    td_errors = (Q_expected - Q_targets).abs()  # CRITICAL FIX: Use abs() for priority updates
                    loss = (is_weights * (Q_expected - Q_targets).pow(2)).mean()
                else:
                    criterion = SmoothL1Loss()
                    loss = criterion(Q_expected, Q_targets)
                    td_errors = (Q_expected - Q_targets).abs()  # For consistency
            forward_time = time.time() - forward_start

            # Calculate loss scaling for gradient accumulation
            accumulation_steps = getattr(RL_CONFIG, 'gradient_accumulation_steps', 1)
            loss = loss / accumulation_steps  # Scale loss for accumulation
            
            # Backward pass with mixed precision support
            backward_start = time.time()
            if self.use_mixed_precision and self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            backward_time = time.time() - backward_start
            
            # Update gradient accumulation counter
            self._gradient_accumulation_counter += 1
            
            # Only step optimizer every N accumulation steps
            if self._gradient_accumulation_counter % accumulation_steps == 0:
                optimizer_start = time.time()
                if self.use_mixed_precision and self.scaler is not None:
                    # Gradient clipping with mixed precision
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), max_norm=10.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Standard gradient clipping and optimization
                    torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), max_norm=10.0)
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

            # Only update target network and scheduler on actual optimizer steps
            if self._gradient_accumulation_counter % accumulation_steps == 0:
                # Soft target update if enabled (Polyak averaging every step)
                if getattr(RL_CONFIG, 'use_soft_target', False):
                    tau = getattr(RL_CONFIG, 'tau', 0.005)
                    self.soft_update(tau)

                # Step LR scheduler if enabled
                if self.scheduler is not None:
                    self.scheduler.step()
            
            total_time = time.time() - step_start
            
            # Print timing every 100 steps for debugging
            # if metrics.total_training_steps % 100 == 0:
            #    print(f"Training step timing - Total: {total_time*1000:.1f}ms, "
            #          f"Sampling: {sampling_time*1000:.1f}ms, Forward: {forward_time*1000:.1f}ms, "
            #          f"Backward: {backward_time*1000:.1f}ms, Optimizer: {optimizer_time*1000:.1f}ms")
                    
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
        """Synchronize inference network (CPU) with training updates"""
        # This is called after training to ensure inference network has latest weights
        # Since qnetwork_local is used for both inference and training, no sync needed
        # Just ensure it's on the correct device for next inference
        current_device = next(self.qnetwork_local.parameters()).device
        if current_device != self.inference_device:
            self.qnetwork_local = self.qnetwork_local.to(self.inference_device)
        
    def act(self, state, epsilon=0.0):
        """Select action using epsilon-greedy policy - optimized for parallel execution"""
        # Convert state to tensor on inference device
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.inference_device)
        
        # Set evaluation mode
        self.qnetwork_local.eval()
        
        with torch.no_grad():
            if getattr(RL_CONFIG, 'use_distributional', False):
                q_dist = self.qnetwork_local(state)  # (1, A, N)
                action_values = q_dist.mean(dim=2)   # (1, A)
            else:
                action_values = self.qnetwork_local(state)
            
        # Set training mode back
        self.qnetwork_local.train()
        
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
        if os.path.exists(filename):
            try:
                print(f"Loading model checkpoint from {filename}")
                checkpoint = torch.load(filename, map_location=device)
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
    """Parse binary frame data from Lua into game state"""
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
        # Parse as uint8 payload then normalize based on encoding type
        try:
            vals_u8 = np.frombuffer(state_data, dtype=np.uint8, count=num_values)
        except ValueError as e:
            print(f"ERROR: frombuffer failed: {e}", flush=True)
            sys.exit(1)
        if vals_u8.size != num_values:
            print(f"ERROR: Expected {num_values} state values but got {vals_u8.size}", flush=True)
            sys.exit(1)
        
        # Mixed normalization based on Lua packing:
        # - Natural 8-bit values (0-255): normalize to [0,1] via /255
        # - Relative segments: INVALID=0→-1, valid [1,255]→[-1,+1] via (v-128)/127  
        # - Booleans: 0→0, 255→1 via /255
        # - Unit floats: [0,255]→[0,1] via /255
        
        # Define ranges for different encoding types (matching Lua packing order):
        # Game state (5): natural u8
        # Targeting (5): nearest_seg(rel), nearest_seg_dup(rel), depth(u8), aligned(bool), error_mag(unit)
        # Player state (23): pos(u8), alive(u8), state(u8), depth(u8), zap_uses(u8), zap_active(u8), shot_count(u8), 
        #                   shot_pos[8](u8), shot_seg[8](rel)  
        # Level state (35): level_num(u8), level_type(u8), level_shape(u8), spike_heights[16](u8), level_angles[16](u8)
        # Enemies state (16): all counts are natural u8
        # Enemy info (42): all natural u8 flags/types
        # Enemy segments (7): rel segments  
        # Enemy depths (7): natural u8
        # Top enemy segs (7): rel segments
        # Enemy shot pos (4): natural u8
        # Enemy shot segs (4): rel segments
        # Charging fuseball segs (7): rel segments
        # Active pulsar segs (7): rel segments  
        # Top rail enemy segs (7): rel segments
        
        # Build mask for relative vs natural encoding
        rel_indices = set()
        idx = 0
        
        # Game state (5) - all natural
        idx += 5
        
        # Targeting (5) - nearest_seg(rel), nearest_seg_dup(rel), depth(nat), aligned(bool→nat), error_mag(unit→nat)
        rel_indices.update([idx, idx+1])  # nearest_seg, nearest_seg_dup
        idx += 5
        
        # Player state (23) - shot_seg[8] are relative at indices [15:23]
        shot_seg_start = idx + 15  # pos(1) + alive(1) + state(1) + depth(1) + zap_uses(1) + zap_active(1) + shot_count(1) + shot_pos[8](8) = 15
        rel_indices.update(range(shot_seg_start, shot_seg_start + 8))
        idx += 23
        
        # Level state (35) - all natural
        idx += 35
        
        # Enemies state (16) - all natural  
        idx += 16
        
        # Enemy info (42) - all natural
        idx += 42
        
        # Enemy segments (7) - all relative
        rel_indices.update(range(idx, idx + 7))
        idx += 7
        
        # Enemy depths (7) - all natural
        idx += 7
        
        # Top enemy segs (7) - all relative
        rel_indices.update(range(idx, idx + 7))
        idx += 7
        
        # Enemy shot pos (4) - all natural
        idx += 4
        
        # Enemy shot segs (4) - all relative  
        rel_indices.update(range(idx, idx + 4))
        idx += 4
        
        # Charging fuseball segs (7) - all relative
        rel_indices.update(range(idx, idx + 7))
        idx += 7
        
        # Active pulsar segs (7) - all relative
        rel_indices.update(range(idx, idx + 7))
        idx += 7
        
        # Top rail enemy segs (7) - all relative
        rel_indices.update(range(idx, idx + 7))
        idx += 7
        
        # Apply mixed normalization
        state = np.zeros(vals_u8.shape, dtype=np.float32)
        for i in range(len(vals_u8)):
            if i in rel_indices:
                # Relative encoding: 0=INVALID→-1, [1,255]→[-1,+1] via (v-128)/127
                if vals_u8[i] == 0:
                    state[i] = -1.0  # INVALID
                else:
                    state[i] = (vals_u8[i] - 128.0) / 127.0
            else:
                # Natural encoding: [0,255]→[0,1] via /255
                state[i] = vals_u8[i] / 255.0
        
        state = np.clip(state, -1.0, 1.0)
        
        frame_data = FrameData(
            state=state,
            reward=reward,
            action=(bool(fire), bool(zap), spinner),
            mode=game_mode,
            gamestate=gamestate,        # Added: Include gamestate from OOB header
            done=bool(done),
            attract=bool(is_attract),
            save_signal=bool(save_signal),
            enemy_seg=nearest_enemy,
            player_seg=player_seg,
            open_level=bool(is_open),
            expert_fire=bool(expert_fire),
            expert_zap=bool(expert_zap),
            level_number=level_number,  # Use the received level_number from OOB data
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
        f"{'Epsilon':>7} | {'Guided %':>8} | {'Mem Size':>8} | {'Avg Level':>9} | {'Level Type':>10} | {'Override':>10}"
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
    
    row = (
        f"{metrics.frame_count:8d} | {metrics.fps:5.1f} | {client_count:7d} | {mean_reward:12.2f} | {dqn_reward:10.2f} | "
        f"{mean_loss:8.2f} | {metrics.epsilon:7.3f} | {guided_ratio*100:7.2f}% | "
        f"{mem_size:8d} | {display_level:9.2f} | {'Open' if metrics.open_level else 'Closed':10} | {override_status:10}"
    )
    print_with_terminal_restore(kb, row)

def get_expert_action(enemy_seg, player_seg, is_open_level, expert_fire=False, expert_zap=False):
    """Expert policy to move toward nearest enemy with neutral tie-breaker.
    Returns (fire, zap, spinner)
    """
    # Check for INVALID_SEGMENT (-32768) which indicates no valid target (like during tube transitions)
    if enemy_seg == -32768:  # INVALID_SEGMENT
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