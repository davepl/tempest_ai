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
    done: bool
    attract: bool
    save_signal: bool
    enemy_seg: int
    player_seg: int
    open_level: bool
    expert_fire: bool  # Added: Expert system fire recommendation
    expert_zap: bool   # Added: Expert system zap recommendation
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FrameData':
        """Create FrameData from dictionary"""
        return cls(
            state=data["state"],
            reward=data["reward"],
            action=data["action"],
            mode=data["mode"],
            done=data["done"],
            attract=data["attract"],
            save_signal=data["save_signal"],
            enemy_seg=data["enemy_seg"],
            player_seg=data["player_seg"],
            open_level=data["open_level"],
            expert_fire=data["expert_fire"],
            expert_zap=data["expert_zap"]
        )

# Configuration constants
SERVER_CONFIG = server_config
RL_CONFIG = rl_config

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else 
                      "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize metrics
metrics = config_metrics

# Global reference to server for metrics display
metrics.global_server = None

# Experience replay memory
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

# PER hyperparameters
PER_ALPHA = 0.6  # Priority exponent
PER_BETA_START = 0.4  # Initial importance sampling weight
PER_BETA_FRAMES = 100000  # Frames over which to anneal beta to 1.0
PER_EPSILON = 1e-5  # Small constant to ensure non-zero priorities
TARGET_UPDATE_FREQUENCY = 1000  # Update target network every N training steps

class ReplayMemory:
    """Prioritized Experience Replay buffer to store and sample experiences for training"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)  # Store priorities for each experience
        self.alpha = PER_ALPHA
        self.epsilon = PER_EPSILON
        self.max_priority = 1.0  # Initial max priority for new experiences
        self.lock = threading.Lock()  # Lock for thread safety
        
    def push(self, state, action, reward, next_state, done):
        """Add experience to memory with maximum priority"""
        # Ensure action is an integer (index in the action space)
        if not isinstance(action, int) and not isinstance(action, np.int64) and not isinstance(action, np.int32):
            try:
                action = int(action)  # Try to convert to int if possible
            except (TypeError, ValueError):
                print(f"Warning: Expected action to be an integer, got {type(action)}. Converting to int.")
                action = 0  # Default action if conversion fails
        
        experience = Experience(state, action, reward, next_state, done)
        
        # Acquire lock when modifying shared data structures
        with self.lock:
            self.memory.append(experience)
            self.priorities.append(self.max_priority)  # New experiences get max priority
        
    def sample(self, batch_size, beta=PER_BETA_START):
        """Sample batch of experiences based on priorities with importance sampling weights"""
        with self.lock:
            if len(self.memory) < batch_size:
                return None, None, None, None, None, None, None
                
            # Calculate sampling probabilities - ensure memory and priorities have same length
            memory_len = len(self.memory)
            
            # Check if lengths match - they should, but just in case
            if len(self.priorities) != memory_len:
                print(f"Warning: Memory length ({memory_len}) doesn't match priorities length ({len(self.priorities)}). Fixing.")
                # Adjust priorities to match memory length - take the most recent ones or pad with max_priority
                if len(self.priorities) > memory_len:
                    # If priorities is larger, trim it to match memory
                    self.priorities = deque(list(self.priorities)[-memory_len:], maxlen=self.capacity)
                else:
                    # If priorities is smaller, pad with max_priority
                    while len(self.priorities) < memory_len:
                        self.priorities.append(self.max_priority)
            
            # Make a copy of priorities and memory to work with outside the lock
            priorities_array = np.array(self.priorities)
            memory_snapshot = list(self.memory)
            indices_range = memory_len
        
        # Release the lock before doing the expensive sampling operations
        # Calculate sampling probabilities
        probabilities = priorities_array ** self.alpha
        
        # Normalize probabilities to sum to 1
        sum_probs = np.sum(probabilities)
        if sum_probs == 0:  # Avoid division by zero
            print("Warning: All priorities are zero. Using uniform sampling.")
            probabilities = np.ones_like(priorities_array) / indices_range
        else:
            probabilities = probabilities / sum_probs
            
        # Double-check sizes before sampling to prevent numpy error
        if len(probabilities) != indices_range:
            print(f"Error: Probability length ({len(probabilities)}) doesn't match memory length ({indices_range}). Using uniform sampling.")
            # Fall back to uniform sampling
            indices = np.random.choice(indices_range, batch_size, replace=False)
        else:
            # Sample indices based on priorities
            try:
                indices = np.random.choice(indices_range, batch_size, p=probabilities, replace=False)
            except ValueError as e:
                print(f"Sampling error: {e}. Falling back to uniform sampling.")
                # Fall back to uniform sampling if there's an issue with weighted sampling
                indices = np.random.choice(indices_range, batch_size, replace=False)
        
        experiences = [memory_snapshot[i] for i in indices]
        
        try:
            # Calculate importance sampling weights
            weights = (indices_range * probabilities[indices]) ** -beta
            weights /= weights.max()  # Normalize weights
            
            # Prepare batch data for training - with more explicit conversion and error handling
            # Convert state batch
            states = np.vstack([e.state for e in experiences])
            
            # Convert action batch - ensure each action is a scalar
            actions = np.array([int(e.action) for e in experiences]).reshape(-1, 1)
            
            # Convert reward batch
            rewards = np.array([e.reward for e in experiences]).reshape(-1, 1)
            
            # Convert next_state batch
            next_states = np.vstack([e.next_state for e in experiences])
            
            # Convert done flags
            dones = np.array([e.done for e in experiences], dtype=np.uint8).reshape(-1, 1)
            
            # Convert to tensors
            states_tensor = torch.from_numpy(states).float().to(device)
            actions_tensor = torch.from_numpy(actions).long().to(device)
            rewards_tensor = torch.from_numpy(rewards).float().to(device)
            next_states_tensor = torch.from_numpy(next_states).float().to(device)
            dones_tensor = torch.from_numpy(dones).float().to(device)
            weights_tensor = torch.from_numpy(weights).float().unsqueeze(1).to(device)
            
            return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor, weights_tensor, indices
            
        except (ValueError, TypeError) as e:
            print(f"Error creating batch: {e}")
            # Debug information to identify which part of the batch is causing issues
            shapes = {
                "states": [e.state.shape if hasattr(e.state, 'shape') else None for e in experiences],
                "actions": [type(e.action) for e in experiences],
                "rewards": [type(e.reward) for e in experiences],
                "next_states": [e.next_state.shape if hasattr(e.next_state, 'shape') else None for e in experiences],
                "dones": [type(e.done) for e in experiences]
            }
            print(f"Batch shapes: {shapes}")
            return None, None, None, None, None, None, None
    
    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD errors"""
        with self.lock:
            for idx, error in zip(indices, td_errors):
                if idx >= len(self.priorities):
                    print(f"Warning: Index {idx} out of range for priorities list with length {len(self.priorities)}. Skipping.")
                    continue
                    
                # Convert to float if not already and get absolute value
                error_value = float(abs(error)) if not isinstance(error, float) else abs(error)
                
                # Add epsilon to prevent zero priority
                raw_priority = error_value + self.epsilon
                
                # Clip priority to prevent extreme values from dominating sampling
                # A reasonable upper bound based on typical TD error magnitudes
                max_allowed_priority = 10.0  # Maximum allowed priority
                raw_priority = min(raw_priority, max_allowed_priority)
                
                # Store priority (without alpha exponent - alpha is applied during sampling)
                self.priorities[idx] = raw_priority
                
                # Update max_priority for new experiences
                self.max_priority = max(self.max_priority, raw_priority)
        
    def __len__(self):
        with self.lock:
            return len(self.memory)

class DQN(nn.Module):
    """Deep Q-Network model."""
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 768) 
        self.fc2 = nn.Linear(768, 512)  
        self.fc3 = nn.Linear(512, 256)        
        self.fc4 = nn.Linear(256, 128)        
        self.fc5 = nn.Linear(128, 64)         
        self.out = nn.Linear(64, action_size) 

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))      
        x = F.relu(self.fc5(x))
        return self.out(x)

class DQNAgent:
    """DQN Agent with experience replay and target network"""
    def __init__(self, state_size, action_size, learning_rate=RL_CONFIG.learning_rate, gamma=RL_CONFIG.gamma, 
                 epsilon=RL_CONFIG.epsilon, epsilon_min=RL_CONFIG.epsilon_min, 
                 memory_size=RL_CONFIG.memory_size, batch_size=RL_CONFIG.batch_size):
        self.state_size = state_size
        self.action_size = action_size
        self.last_save_time = 0.0 # Initialize last save time
        
        # PER parameters
        self.beta = PER_BETA_START
        # Use a slower annealing rate for beta - adjust based on expected total training time
        # This ensures beta increases more gradually throughout the entire training process
        total_expected_frames = 20000000  # Adjust based on your expected total training duration
        self.beta_increment = (1.0 - PER_BETA_START) / min(total_expected_frames, PER_BETA_FRAMES * 5)
        
        self.train_steps_count = 0  # Counter for target network updates
        self.batch_size = batch_size  # Store batch size locally
        
        # Q-Networks (online and target)
        self.qnetwork_local = DQN(state_size, action_size).to(device)
        self.qnetwork_target = DQN(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)
        
        # Replay memory
        self.memory = ReplayMemory(memory_size)
        
        # Initialize target network with same weights as local network
        self.update_target_network()
        
        # Training queue for background thread
        self.train_queue = queue.Queue(maxsize=1000)
        self.training_thread = None
        self.running = True

        """Start background thread for training"""
        self.training_thread = threading.Thread(target=self.background_train, daemon=True, name="TrainingWorker")
        self.training_thread.start()
        
    def background_train(self):
        """Run training in background thread"""
        print(f"Training thread started on {device}")
        while self.running: 
            try:
                # Get an item from the queue. This blocks if the queue is empty.
                # The item itself is just a signal (True), we don't need its value.
                _ = self.train_queue.get() 
                
                # If we get an item, process one training step
                self.train_step()
                
                # Signal that the task is done
                self.train_queue.task_done()

            except queue.Empty: 
                # This shouldn't typically happen with a blocking get unless maybe
                # a timeout was added, but handle just in case.
                continue 
            except Exception as e:
                print(f"Training error: {e}")
                traceback.print_exc() # Print full traceback for errors
                time.sleep(1) # Pause after error

    def train_step(self):
        """Perform a single training step on one batch"""
        if len(self.memory) < self.batch_size:
            return
            
        # Get a batch using PER sampling with current beta value
        states, actions, rewards, next_states, dones, weights, indices = self.memory.sample(self.batch_size, self.beta)
        
        # If sample returned None (shouldn't happen due to our length check above, but just in case)
        if states is None:
            return
            
        # Calculate Q values from local network for the actions taken
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        # Calculate target Q values using Double DQN approach
        with torch.no_grad():
            best_actions = self.qnetwork_local(next_states).argmax(1, keepdim=True)
            Q_targets_next = self.qnetwork_target(next_states).gather(1, best_actions)
            Q_targets = rewards + (RL_CONFIG.gamma * Q_targets_next * (1 - dones))
        
        # Calculate TD errors for updating priorities
        td_errors = (Q_targets - Q_expected).detach().cpu().numpy()
        
        # Update priorities in replay memory (using absolute TD errors)
        self.memory.update_priorities(indices, np.abs(td_errors))
        
        # Compute loss using importance sampling weights
        criterion = SmoothL1Loss(reduction='none')
        elementwise_loss = criterion(Q_expected, Q_targets)
        
        # Ensure shapes match for weights and elementwise_loss before multiplication
        # weights should be [batch_size, 1] and elementwise_loss should be [batch_size, 1]
        weighted_loss = (weights * elementwise_loss).mean()
        
        # Perform optimization
        self.optimizer.zero_grad()
        weighted_loss.backward() 
        self.optimizer.step() 
        metrics.losses.append(weighted_loss.item())
        
        # Anneal beta for importance sampling - slower increment for more stable training
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Increment training step counter and update target network if needed
        self.train_steps_count += 1
        if self.train_steps_count % TARGET_UPDATE_FREQUENCY == 0:
            self.update_target_network()
            if DEBUG_MODE:
                print(f"Target network updated at step {self.train_steps_count}, beta: {self.beta:.4f}")

    def update_target_network(self):
        """Update target network with weights from local network"""
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        
    def act(self, state, epsilon=0.0):
        """Select action using epsilon-greedy policy"""
        # Convert state to tensor
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        # Set evaluation mode
        self.qnetwork_local.eval()
        
        with torch.no_grad():
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
        
        # Add training task to queue if not full
        if not self.train_queue.full():
            self.train_queue.put(True)
        
        # Note: Target network updates are now handled exclusively in train_step
        # based on training step count, not here based on frame count
            
    def save(self, filename):
        """Save model weights, rate-limited unless forced."""
        is_forced_save = "exit" in filename or "shutdown" in filename
        current_time = time.time()
        save_interval = 30.0 # Minimum seconds between saves
        
        # Rate limit non-forced saves
        if not is_forced_save:
            if current_time - self.last_save_time < save_interval:
                # Optional: Add a debug print if needed
                # print(f"Skipping save to {filename}, too soon since last save.")
                return # Skip save
        
        # Proceed with save if forced or interval elapsed
        try:
            # Determine the actual expert ratio to save (not the override value)
            if metrics.expert_mode or metrics.override_expert:
                ratio_to_save = metrics.saved_expert_ratio
            else:
                ratio_to_save = metrics.expert_ratio

            torch.save({
                'policy_state_dict': self.qnetwork_local.state_dict(),
                'target_state_dict': self.qnetwork_target.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'memory_size': len(self.memory),
                'epsilon': metrics.epsilon,
                'frame_count': metrics.frame_count,
                'expert_ratio': ratio_to_save, # Save the determined ratio
                'last_decay_step': metrics.last_decay_step,
                'last_epsilon_decay_step': metrics.last_epsilon_decay_step
            }, filename)
            
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
                checkpoint = torch.load(filename, map_location=device)
                self.qnetwork_local.load_state_dict(checkpoint['policy_state_dict'])
                self.qnetwork_target.load_state_dict(checkpoint['target_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
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
                                
                print(f"Loaded model from {filename}")
                print(f"  - Resuming from frame: {metrics.frame_count}")
                print(f"  - Resuming epsilon: {metrics.epsilon:.4f}")
                print(f"  - Resuming expert_ratio: {metrics.expert_ratio:.4f}")
                print(f"  - Resuming last_decay_step: {metrics.last_decay_step}")

                return True
            except Exception as e:
                print(f"Error loading model: {e}")
                return False
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
    
    def update_frame_count(self):
        with self.lock:
            # Update total frame count
            self.metrics.frame_count += 1
            
            # Update FPS tracking
            current_time = time.time()
            
            # Initialize last_fps_time if this is the first frame
            if self.metrics.last_fps_time == 0:
                self.metrics.last_fps_time = current_time
                
            # Count frames for this second
            self.metrics.frames_last_second += 1
            
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
        
        format_str = ">HdBBBHHHBBBhBhBBBB"  # Updated format string
        header_size = struct.calcsize(format_str)
        
        if len(data) < header_size:
            print(f"ERROR: Received data too small: {len(data)} bytes, need {header_size}", flush=True)
            sys.exit(1)
            
        values = struct.unpack(format_str, data[:header_size])
        num_values, reward, game_action, game_mode, done, frame_counter, score_high, score_low, \
        save_signal, fire, zap, spinner, is_attract, nearest_enemy, player_seg, is_open, \
        expert_fire, expert_zap = values  # Added expert recommendations
        
        # Combine score components
        score = (score_high * 65536) + score_low
        
        state_data = data[header_size:]
        
        # Safely process state values with error handling
        state_values = []
        for i in range(0, len(state_data), 2):  # Using 2 bytes per value
            if i + 1 < len(state_data):
                try:
                    value = struct.unpack(">H", state_data[i:i+2])[0]
                    normalized = (value / 255.0) * 2.0 - 1.0
                    state_values.append(normalized)
                except struct.error as e:
                    print(f"ERROR: Failed to unpack state value at position {i}: {e}", flush=True)
                    sys.exit(1)
        
        state = np.array(state_values, dtype=np.float32)  # Already normalized
        
        # Verify we got the expected number of values
        if len(state_values) != num_values:
            print(f"ERROR: Expected {num_values} state values but got {len(state_values)}", flush=True)
            sys.exit(1)
        
        frame_data = FrameData(
            state=state,
            reward=reward,
            action=(bool(fire), bool(zap), spinner),
            mode=game_mode,
            done=bool(done),
            attract=bool(is_attract),
            save_signal=bool(save_signal),
            enemy_seg=nearest_enemy,
            player_seg=player_seg,
            open_level=bool(is_open),
            expert_fire=bool(expert_fire),
            expert_zap=bool(expert_zap)
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
        f"{'Epsilon':>7} | {'Guided %':>8} | {'Mem Size':>8} | {'Level Type':>10} | {'Override':>10}"
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
    
    row = (
        f"{metrics.frame_count:8d} | {metrics.fps:5.1f} | {client_count:7d} | {mean_reward:12.2f} | {dqn_reward:10.2f} | "
        f"{mean_loss:8.2f} | {metrics.epsilon:7.3f} | {guided_ratio*100:7.2f}% | "
        f"{mem_size:8d} | {'Open' if metrics.open_level else 'Closed':10} | {override_status:10}"
    )
    print_with_terminal_restore(kb, row)

def get_expert_action(enemy_seg, player_seg, is_open_level, expert_fire=False, expert_zap=False):
    """Calculate expert-guided action based on game state and Lua recommendations"""
    # Check for INVALID_SEGMENT (-32768) which indicates no valid target (like during tube transitions)
    if enemy_seg == -32768:  # INVALID_SEGMENT
        return expert_fire, expert_zap, 0  # Use Lua's recommendations with no movement
        
    # Convert absolute segments to relative distance
    if is_open_level:
        # Open level: direct distance calculation (-15 to +15)
        relative_dist = enemy_seg - player_seg
    else:
        # Closed level: find shortest path around the circle (-7 to +8)
        clockwise = (enemy_seg - player_seg) % 16
        counter = (player_seg - enemy_seg) % 16
        if clockwise <= 8:
            relative_dist = clockwise  # Move clockwise
        else:
            relative_dist = -counter  # Move counter-clockwise

    if relative_dist == 0:
        return expert_fire, expert_zap, 0  # Use Lua's recommendations when aligned
        
    # Calculate intensity based on distance
    distance = abs(relative_dist)
    intensity = min(0.9, 0.3 + (distance * 0.05))  # Match Lua intensity calculation
    
    # For positive relative_dist (need to move clockwise), use negative spinner
    spinner = -intensity if relative_dist > 0 else intensity
    
    # Always use Lua's recommendations for fire/zap
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
        
    step_interval = current_step // SERVER_CONFIG.expert_ratio_decay_steps
    
    # Only update if we've moved to a new interval
    if step_interval > metrics.last_decay_step:
        metrics.last_decay_step = step_interval
        if step_interval == 0:
            # First interval - use starting value
            metrics.expert_ratio = SERVER_CONFIG.expert_ratio_start
        else:
            # Apply decay
            metrics.expert_ratio *= SERVER_CONFIG.expert_ratio_decay
        
        # Ensure we don't go below the minimum
        metrics.expert_ratio = max(SERVER_CONFIG.expert_ratio_min, metrics.expert_ratio)
        
    return metrics.expert_ratio