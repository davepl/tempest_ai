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
    """DQN Agent with experience replay and target network"""
    def __init__(self, state_size, action_size, learning_rate=RL_CONFIG.learning_rate, gamma=RL_CONFIG.gamma, 
                 epsilon=RL_CONFIG.epsilon, epsilon_min=RL_CONFIG.epsilon_min, 
                 memory_size=RL_CONFIG.memory_size, batch_size=RL_CONFIG.batch_size):
        self.state_size = state_size
        self.action_size = action_size
        self.last_save_time = 0.0 # Initialize last save time
        
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
        if len(self.memory) < RL_CONFIG.batch_size:
            return
            
        states, actions, rewards, next_states, dones = self.memory.sample(RL_CONFIG.batch_size)
        
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        with torch.no_grad():
            best_actions = self.qnetwork_local(next_states).argmax(1, keepdim=True)
            Q_targets_next = self.qnetwork_target(next_states).gather(1, best_actions)
            Q_targets = rewards + (RL_CONFIG.gamma * Q_targets_next * (1 - dones))
        
        # Compute loss and perform optimization
        criterion = SmoothL1Loss()
        loss = criterion(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward() 
        self.optimizer.step() 
        metrics.losses.append(loss.item())

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
    
    # Always use Lua's recommendations for fire/zap while moving
    return expert_fire, expert_zap, spinner

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