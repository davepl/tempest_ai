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
import termios
import tty
import fcntl
import socket
import traceback

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
    """Deep Q-Network model (Improved)"""
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 32)
        self.out = nn.Linear(32, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.out(x)

class DQNAgent:
    """DQN Agent with experience replay and target network"""
    def __init__(self, state_size, action_size, learning_rate=1e-4, gamma=0.99, 
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=10000, 
                 memory_size=500000, batch_size=512):
        self.state_size = state_size
        self.action_size = action_size
        
        # Q-Networks (online and target)
        self.qnetwork_local = DQN(state_size, action_size).to(device)
        self.qnetwork_target = DQN(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)
        
        # Replay memory
        self.memory = ReplayMemory(memory_size)
        
        # Initialize target network with same weights as local network
        self.update_target_network()
        
        # Training queue for background thread
        self.train_queue = queue.Queue(maxsize=100)
        self.training_thread = None
        self.running = True

        """Start background thread for training"""
        self.training_thread = threading.Thread(target=self.background_train, daemon=True)
        self.training_thread.start()
        
    def background_train(self):
        """Run training in background thread"""
        print(f"Training thread started on {device}")
        while self.running:
            try:
                if self.train_queue.qsize() > 0:
                    # Process batch
                    self.train_step()
                else:
                    # Sleep to reduce CPU usage
                    time.sleep(0.01)
            except Exception as e:
                print(f"Training error: {e}")
                time.sleep(1)  # Prevent tight loop on error
                
    def train_step(self):
        """Perform a single training step on one batch"""
        if len(self.memory) < RL_CONFIG.batch_size:
            return
            
        # Sample batch from memory
        states, actions, rewards, next_states, dones = self.memory.sample(RL_CONFIG.batch_size)
        
        # Get Q values for current states
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        # Compute target Q values (Double DQN approach)
        with torch.no_grad():
            best_actions = self.qnetwork_local(next_states).argmax(1, keepdim=True)
            Q_targets_next = self.qnetwork_target(next_states).gather(1, best_actions)
            Q_targets = rewards + (RL_CONFIG.gamma * Q_targets_next * (1 - dones))
        
        # Compute loss and perform optimization
        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to stabilize training
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 1)
        self.optimizer.step()
        
        # Update metrics
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
        """Save model weights"""
        torch.save({
            'policy_state_dict': self.qnetwork_local.state_dict(),
            'target_state_dict': self.qnetwork_target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'memory_size': len(self.memory),
            'epsilon': metrics.epsilon,
            'frame_count': metrics.frame_count,
            'expert_ratio': metrics.expert_ratio
        }, filename)
        
        # Only print on exit save (modified externally)
        if "exit" in filename or "shutdown" in filename:
            print(f"Model saved to {filename} (frame {metrics.frame_count}, expert ratio {metrics.expert_ratio:.2f})")

    def load(self, filename):
        """Load model weights"""
        if os.path.exists(filename):
            try:
                checkpoint = torch.load(filename, map_location=device)
                self.qnetwork_local.load_state_dict(checkpoint['policy_state_dict'])
                self.qnetwork_target.load_state_dict(checkpoint['target_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                # Load frame count and epsilon (for exploration)
                metrics.epsilon = checkpoint.get('epsilon', RL_CONFIG.epsilon_start)
                metrics.frame_count = checkpoint.get('frame_count', 0)
                
                # Always set the expert ratio to the start value
                metrics.expert_ratio = SERVER_CONFIG.expert_ratio_start
                metrics.last_decay_step = 0
                
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
                return False
        return False

class KeyboardHandler:
    """Non-blocking keyboard input handler"""
    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        
    def __enter__(self):
        tty.setraw(sys.stdin.fileno())
        # Set stdin non-blocking
        flags = fcntl.fcntl(self.fd, fcntl.F_GETFL)
        fcntl.fcntl(self.fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        return self
        
    def __exit__(self, *args):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)
        
    def check_key(self):
        """Check for keyboard input non-blockingly"""
        try:
            return sys.stdin.read(1)
        except (IOError, TypeError):
            return None
            
    def restore_terminal(self):
        """Temporarily restore terminal settings for printing"""
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)
        
    def set_raw_mode(self):
        """Set terminal back to raw mode"""
        tty.setraw(sys.stdin.fileno())

def print_with_terminal_restore(kb_handler, *args, **kwargs):
    """Print with proper terminal settings"""
    if IS_INTERACTIVE and kb_handler:
        kb_handler.restore_terminal()
    print(*args, **kwargs, flush=True)
    if IS_INTERACTIVE and kb_handler:
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
        
        format_str = ">HdBBBHHHBBBhBhBB"  # Updated to include both score components
        header_size = struct.calcsize(format_str)
        
        if len(data) < header_size:
            print(f"ERROR: Received data too small: {len(data)} bytes, need {header_size}", flush=True)
            sys.exit(1)
            
        values = struct.unpack(format_str, data[:header_size])
        num_values, reward, game_action, game_mode, done, frame_counter, score_high, score_low, \
        save_signal, fire, zap, spinner, is_attract, nearest_enemy, player_seg, is_open = values
        
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
            open_level=bool(is_open)
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

def get_expert_action(enemy_seg, player_seg, is_open_level):
    """Calculate expert-guided action based on game state"""
 
    if enemy_seg == -1:
        return 1, 0, 0  # No enemies, might as well fire at spikes

    if enemy_seg == player_seg:
        return 1, 0, 0  # Fire when aligned
        
    # Calculate movement based on level type
    if is_open_level:
        distance = abs(enemy_seg - player_seg)
        intensity = min(0.9, 0.3 + (distance * 0.05))
        spinner = -intensity if enemy_seg > player_seg else intensity
    else:
        # Calculate shortest path with wraparound
        clockwise = (enemy_seg - player_seg) % 16
        counter = (player_seg - enemy_seg) % 16
        min_dist = min(clockwise, counter)
        intensity = min(0.9, 0.3 + (min_dist * 0.05))
        spinner = -intensity if clockwise < counter else intensity
    
    return 1, 0, spinner  # Fire while moving

def expert_action_to_index(fire, zap, spinner):
    """Convert continuous expert actions to discrete action index"""
    # Map fire, zap, spinner to closest discrete action
    # First, determine fire component (0 or 1)
    fire_idx = 7 if fire else 0  # 0 for no fire (actions 0-6), 7 for fire (actions 7-13)
    
    # Then determine spinner component (0-6)
    spinner_value = max(-0.9, min(0.9, spinner))  # Clamp between -0.9 and 0.9
    spinner_idx = int((spinner_value + 0.9) / 0.3)  # Map to 0-6
    spinner_idx = min(6, spinner_idx)  # Just to be safe
    
    return fire_idx + spinner_idx

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