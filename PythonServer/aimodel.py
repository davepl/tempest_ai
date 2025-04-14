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
from pathlib import Path

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
    RLConfigData
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
            open_level=data["open_level"]
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

class ReplayMemory:
    """Replay buffer to store and sample experiences for training"""
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        """Add experience to memory"""
        self.memory.append(Experience(state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        """Sample random batch of experiences"""
        experiences = random.sample(self.memory, min(batch_size, len(self.memory)))
        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(device)
        return states, actions, rewards, next_states, dones
        
    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    """Deep Q-Network model."""
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 256) # Input -> Hidden 1 (256)
        self.fc2 = nn.Linear(256, 128)        # Hidden 1 -> Hidden 2 (128)
        self.fc3 = nn.Linear(128, 64)         # Hidden 2 -> Hidden 3 (64)
        self.out = nn.Linear(64, action_size) # Hidden 3 -> Output

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x)) # Added ReLU for fc3 output
        return self.out(x)

class DQNAgent:
    """Deep Q-Network agent"""
    def __init__(self, state_size, action_size, learning_rate=RL_CONFIG.learning_rate, gamma=RL_CONFIG.gamma, 
                 memory_size=RL_CONFIG.memory_size, batch_size=RL_CONFIG.batch_size):
        """Initialize agent components"""
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayMemory(memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.is_ready = False # <-- Add readiness flag, initially False
        
        # Use defined DEVICE from config
        self.device = device 
        print(f"[DQNAgent] Initializing models on device: {self.device}")
        
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Target network is only for inference

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.SmoothL1Loss() # Huber loss

        # Removed train_queue, training_thread, and related events/locks
        # Training is now external
        print("[DQNAgent] Initialized. Training will be handled externally.")
        self.is_ready = True # <-- Set to True after successful initialization

    def train_step(self, batch):
        """Perform a single training step on a provided batch."""
        if not batch:
            return None

        # Unpack the batch (expecting tuples of NumPy arrays from the queue)
        try:
            states_np, actions_np, rewards_np, next_states_np, dones_np = batch
        except ValueError as e:
             print(f"[DQNAgent Error] Failed to unpack batch: {e}. Batch type: {type(batch)}") 
             return None # Cannot proceed if batch is malformed

        # Convert NumPy arrays to tensors and move to the configured device
        try:
            states = torch.tensor(states_np, dtype=torch.float32).to(self.device)
            # Actions might need squeeze/unsqueeze depending on shape from numpy
            actions = torch.tensor(actions_np, dtype=torch.int64).to(self.device)
            if actions.ndim == 1: actions = actions.unsqueeze(-1) # Ensure shape [batch_size, 1]
                
            rewards = torch.tensor(rewards_np, dtype=torch.float32).to(self.device)
            if rewards.ndim == 1: rewards = rewards.unsqueeze(-1)
                
            next_states = torch.tensor(next_states_np, dtype=torch.float32).to(self.device)
            
            # Ensure dones are boolean and correct shape
            dones = torch.tensor(dones_np.astype(np.bool_), dtype=torch.bool).to(self.device)
            if dones.ndim == 1: dones = dones.unsqueeze(-1)

        except Exception as tensor_err:
             print(f"[DQNAgent Error] Failed converting batch numpy arrays to tensors: {tensor_err}")
             return None

        # --- Q-value prediction and target calculation ---
        # Get current Q values from policy network
        current_q_values = self.policy_net(states).gather(1, actions)

        # Get next Q values from target network
        with torch.no_grad(): # No gradients needed for target calculation
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(-1)
            # Zero out Q values for terminal states
            next_q_values[dones] = 0.0

        # Compute the expected Q values (Bellman equation)
        target_q_values = rewards + (self.gamma * next_q_values)

        # --- Loss Calculation and Optimization ---
        # Compute loss (e.g., Huber loss)
        loss = self.loss_fn(current_q_values, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Optional: Gradient clipping
        # torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        return loss # Return the calculated loss tensor

    def update_target_network(self):
        """Copy weights from policy network to target network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # print("[DQNAgent] Updated target network.") # Optional log

    def act(self, state, epsilon=0.0):
        """Choose action based on epsilon-greedy policy"""
        if random.random() < epsilon:
            return random.randrange(self.action_size) # Explore
        else:
            # Exploit
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state)
            return q_values.max(1)[1].item() # Return action index

    def step(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        # Reward scaling/clipping can be done here before pushing
        # reward = np.sign(reward) # Example: clipping reward
        # reward = reward / 1000.0 # Example: scaling reward
        
        self.memory.push(state, action, reward, next_state, done)
        # Note: Removed the trigger to put data in train_queue

    def save(self, filename, metrics_state: Optional[Dict] = None):
        """Save model weights and optionally metrics state."""
        try:
            # Ensure the directory exists
            filename.parent.mkdir(parents=True, exist_ok=True)
            
            save_data = {
                'policy_net_state_dict': self.policy_net.state_dict(),
                'target_net_state_dict': self.target_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }
            # Include metrics state if provided
            if metrics_state:
                 save_data['metrics_state'] = metrics_state
                 
            torch.save(save_data, str(filename))
            # print(f"[DQNAgent] Model saved to {filename}")
        except Exception as e:
             print(f"[DQNAgent Error] Failed to save model to {filename}: {e}")
             traceback.print_exc()

    def load(self, filename):
        """Load model weights and return success status and metrics state if available."""
        # Reset readiness before attempting load
        self.is_ready = False 
        if not Path(filename).is_file():
            print(f"[DQNAgent] Model file not found at {filename}. Cannot load.")
            return False, None # Return success=False, metrics=None
            
        try:
            # Explicitly set weights_only=False as we are saving more than just weights
            checkpoint = torch.load(str(filename), map_location=self.device, weights_only=False)
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            
            # Load target_net state if present
            if 'target_net_state_dict' in checkpoint:
                 self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            else:
                 self.target_net.load_state_dict(self.policy_net.state_dict())
                 print("[DQNAgent Warning] Target network state not found in checkpoint, copied from policy network.")
            
            # Load optimizer state if present
            if 'optimizer_state_dict' in checkpoint:
                 self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            else:
                 print("[DQNAgent Warning] Optimizer state not found in checkpoint, optimizer not loaded.")
            
            self.target_net.eval() # Ensure target net is in eval mode
            print(f"[DQNAgent] Model weights loaded from {filename}")
            
            # Load metrics state if present
            loaded_metrics_state = checkpoint.get('metrics_state', None)
            if loaded_metrics_state:
                 print(f"[DQNAgent] Found metrics state in checkpoint.")
                 
            self.is_ready = True # <-- Set ready flag only on SUCCESSFUL load
            return True, loaded_metrics_state # Return success=True, metrics_state (or None)
            
        except KeyError as e:
            print(f"[DQNAgent Error] Failed to load model from {filename}. Missing key: {e}")
            traceback.print_exc()
            return False, None
        except Exception as e:
             print(f"[DQNAgent Error] Failed to load model from {filename}: {e}")
             traceback.print_exc()
             return False, None

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
        if not data:
            print("ERROR: Received empty data packet", flush=True)
            return None # Return None instead of sys.exit
        
        format_str = ">HdBBBHHHBBBhBhBB"  # Updated to include both score components
        header_size = struct.calcsize(format_str)
        
        if len(data) < header_size:
            print(f"ERROR: Received data too small: {len(data)} bytes, need {header_size}", flush=True)
            return None
            
        # Unpack header values
        header_values = struct.unpack(format_str, data[:header_size])
        num_values, reward, game_action, game_mode, done, frame_counter, score_high, score_low, \
        save_signal, fire, zap, spinner, is_attract, nearest_enemy, player_seg, is_open = header_values
        
        # Combine score components
        score = (score_high * 65536) + score_low
        
        # --- State Data Parsing (Optimized) ---
        state_data_bytes = data[header_size:]
        expected_state_bytes = num_values * 2 # Each state value is >H (2 bytes)
        
        # Validate state data length
        if len(state_data_bytes) != expected_state_bytes:
            print(f"ERROR: Expected {expected_state_bytes} state bytes ({num_values} values) but got {len(state_data_bytes)}", flush=True)
            return None

        # Efficiently create NumPy array from buffer
        # '>u2' matches the struct format '>H' (big-endian unsigned short)
        state_int_array = np.frombuffer(state_data_bytes, dtype=np.dtype('>u2'))
        
        # Convert to float32 and normalize using vectorized operations
        state = (state_int_array.astype(np.float32) / 255.0) * 2.0 - 1.0
        
        # --- Create FrameData Object (Unchanged) ---
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
            open_level=bool(is_open)
        )
        
        return frame_data
        
    except struct.error as e:
        print(f"ERROR unpacking header data: {e}", flush=True)
        return None
    except ValueError as e:
        print(f"ERROR during state data processing: {e}", flush=True)
        return None
    except Exception as e:
        print(f"ERROR parsing frame data: {e}", flush=True)
        traceback.print_exc() # Print full traceback for unexpected errors
        return None

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

def get_expert_action(enemy_seg, player_seg, is_open_level):
    """Calculate expert-guided action based on game state"""
 
    if enemy_seg == -1:
        return 1, 0, 0  # No enemies, might as well fire

    if enemy_seg == player_seg:
        return 1, 0, 0  # Fire when aligned
        
    # Calculate movement based on level type
    if is_open_level:
        distance = abs(enemy_seg - player_seg)
        intensity = min(0.9, 0.1 + (distance * 0.25))  # Lower base intensity
        spinner = -intensity if enemy_seg > player_seg else intensity
    else:
        # Calculate shortest path with wraparound
        clockwise = (enemy_seg - player_seg) % 16
        counter = (player_seg - enemy_seg) % 16
        min_dist = min(clockwise, counter)
        intensity = min(0.9, 0.1 + (min_dist * 0.25))  # Lower base intensity
        spinner = -intensity if clockwise < counter else intensity
    
    return 1, 0, spinner  # Fire while moving

def expert_action_to_index(fire, zap, spinner):
    """Convert continuous expert actions to discrete action index"""
    if zap:
        return 14  # Special case for zap action
        
    # Clamp spinner value between -0.9 and 0.9
    spinner_value = max(-0.9, min(0.9, spinner))
    
    # Map spinner to 0-6 range
    spinner_idx = int((spinner_value + 0.9) / 0.3)
    spinner_idx = min(6, spinner_idx)  # Ensure we don't exceed valid range
    
    # If firing, offset by 7 to get into the firing action range (7-13)
    base_idx = 0 if not fire else 7
    
    return base_idx + spinner_idx

def encode_action_to_game(fire, zap, spinner):
    """Convert action values to game-compatible format"""
    spinner_val = spinner * 31
    return int(fire), int(zap), int(spinner_val)

def decay_epsilon(frame_count):
    """Calculate decayed exploration rate"""
    return max(RL_CONFIG.epsilon_end, 
               RL_CONFIG.epsilon_start * 
               np.exp(-frame_count / RL_CONFIG.epsilon_decay))

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