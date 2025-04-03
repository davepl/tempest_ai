#!/usr/bin/env python3
"""
Tempest AI Model: Hybrid expert-guided and DQN-based gameplay system.
- Makes intelligent decisions based on enemy positions and level types
- Uses a Deep Q-Network (DQN) for reinforcement learning
- Expert system provides guidance and training examples
- Communicates with Tempest via socket connection
"""

# Override the built-in print function to always flush output
# This ensures proper line breaks in output when running in background
import builtins
_original_print = builtins.print

def _flushing_print(*args, **kwargs):
    kwargs['flush'] = True
    return _original_print(*args, **kwargs)

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

# Suppress warnings
warnings.filterwarnings('ignore')

# Global flag to track if running interactively
# Check this early before any potential tty interaction
IS_INTERACTIVE = sys.stdin.isatty()
print(f"Script Start: sys.stdin.isatty() = {IS_INTERACTIVE}") # DEBUG

@dataclass
class ServerConfigData:
    """Configuration for socket server"""
    host: str = "0.0.0.0"  # Listen on all interfaces
    port: int = 9999
    max_clients: int = 16
    params_count: int = 128
    expert_ratio_start: float = 0.5
    expert_ratio_min: float = 0.01
    expert_ratio_decay: float = 0.98
    expert_ratio_decay_steps: int = 10000
    reset_frame_count: bool = False
    reset_expert_ratio: bool = True

@dataclass
class RLConfigData:
    """Configuration for reinforcement learning"""
    batch_size: int = 512
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: int = 10000
    update_target_every: int = 1000
    learning_rate: float = 1e-4
    memory_size: int = 500000
    save_interval: int = 50000
    train_freq: int = 4

@dataclass
class MetricsData:
    """Metrics tracking for training progress"""
    frame_count: int = 0
    guided_count: int = 0
    total_controls: int = 0
    episode_rewards: Deque[float] = field(default_factory=lambda: deque(maxlen=20))
    dqn_rewards: Deque[float] = field(default_factory=lambda: deque(maxlen=20))
    expert_rewards: Deque[float] = field(default_factory=lambda: deque(maxlen=20))
    losses: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    epsilon: float = 1.0
    expert_ratio: float = 0.75
    last_decay_step: int = 0
    enemy_seg: int = -1
    open_level: bool = False
    override_expert: bool = False  # New field for expert override
    saved_expert_ratio: float = 0.75  # New field to save ratio during override
    last_action_source: str = ""
    
    @property
    def guided_ratio(self) -> float:
        """Calculate the ratio of guided actions"""
        return self.guided_count / max(1, self.total_controls)
    
    @property
    def mean_reward(self) -> float:
        """Calculate mean reward over recent episodes"""
        if not self.episode_rewards:
            return float('nan')
        return np.mean(list(self.episode_rewards))

    def toggle_override(self, kb):
        """Toggle expert guidance override"""
        self.override_expert = not self.override_expert
        if self.override_expert:
            self.saved_expert_ratio = self.expert_ratio
            self.expert_ratio = 0.0
            print_with_terminal_restore(kb, "\nExpert guidance disabled (override ON)")
        else:
            self.expert_ratio = self.saved_expert_ratio
            print_with_terminal_restore(kb, "\nExpert guidance restored (override OFF)")
        display_metrics_header(kb)

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
SERVER_CONFIG = ServerConfigData()
RL_CONFIG = RLConfigData()

MODEL_DIR = "models"
LATEST_MODEL_PATH = f"{MODEL_DIR}/tempest_model_latest.pt"

# Define action space
# We'll discretize the spinner movement into 7 values:
# -3: hard left, -2: medium left, -1: soft left, 0: center, 1: soft right, 2: medium right, 3: hard right
# And 2 options for fire button (0: no fire, 1: fire)
# Total action space: 14 actions (7 spinner positions × 2 fire options)
ACTION_MAPPING = {
    0: (0, 0, -0.3),   # Hard left, no fire, no zap
    1: (0, 0, -0.2),   # Medium left, no fire, no zap
    2: (0, 0, -0.1),   # Soft left, no fire, no zap
    3: (0, 0, 0.0),    # Center, no fire, no zap
    4: (0, 0, 0.1),    # Soft right, no fire, no zap
    5: (0, 0, 0.2),    # Medium right, no fire, no zap
    6: (0, 0, 0.3),    # Hard right, no fire, no zap
    7: (1, 0, -0.3),   # Hard left, fire, no zap
    8: (1, 0, -0.2),   # Medium left, fire, no zap
    9: (1, 0, -0.1),   # Soft left, fire, no zap
    10: (1, 0, 0.0),   # Center, fire, no zap
    11: (1, 0, 0.1),   # Soft right, fire, no zap
    12: (1, 0, 0.2),   # Medium right, fire, no zap
    13: (1, 0, 0.3),   # Hard right, fire, no zap
    14: (1, 1, 0.0),   # Zap+Fire+Sit
}

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else 
                      "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

metrics = MetricsData()

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
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.out = nn.Linear(64, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.out(x)

class DQNAgent:
    """DQN Agent with experience replay and target network"""
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Q-Networks (online and target)
        self.qnetwork_local = DQN(state_size, action_size).to(device)
        self.qnetwork_target = DQN(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=RL_CONFIG.learning_rate)
        
        # Replay memory
        self.memory = ReplayMemory(RL_CONFIG.memory_size)
        
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
                
                print(f"Loaded model from frame {metrics.frame_count}, expert ratio reset to {metrics.expert_ratio:.2f}")
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
            self.metrics.frame_count += 1
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

class SocketServer:
    """Socket-based server to handle multiple clients"""
    def __init__(self, host, port, agent, safe_metrics):
        self.host = host
        self.port = port
        self.agent = agent
        self.metrics = safe_metrics
        self.running = True
        self.clients = {}  # Dictionary to track active clients
        self.client_states = {}  # Dictionary to store per-client state
        self.client_lock = threading.Lock()  # Lock for client dictionaries
        
    def start(self):
        """Start the socket server"""
        try:
            # Create server socket
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(SERVER_CONFIG.max_clients)
            
            print(f"Server started on {self.host}:{self.port}", flush=True)
            print("Waiting for client connections...", flush=True)
            
            # Accept client connections in a loop
            while self.running:
                try:
                    # Accept new connection (timeout after 1 second to check running flag)
                    self.server_socket.settimeout(1.0)
                    client_socket, client_address = self.server_socket.accept()
                    
                    # Generate a unique client ID
                    client_id = self.generate_client_id()
                    
                    print(f"New connection from {client_address}, assigned ID: {client_id}", flush=True)
                    
                    # Initialize client state
                    client_state = {
                        'address': client_address,
                        'last_state': None,
                        'last_action_idx': None,
                        'total_reward': 0,
                        'was_done': False,
                        'episode_dqn_reward': 0,
                        'episode_expert_reward': 0,
                        'connected_time': datetime.now(),
                        'frames_processed': 0
                    }
                    
                    # Store client information
                    with self.client_lock:
                        self.client_states[client_id] = client_state
                    
                    # Start a thread to handle this client
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, client_id),
                        daemon=True
                    )
                    
                    # Store thread and start it
                    with self.client_lock:
                        self.clients[client_id] = client_thread
                    
                    client_thread.start()
                    
                except socket.timeout:
                    # This is expected - just a way to periodically check self.running
                    continue
                except Exception as e:
                    print(f"Error accepting client connection: {e}")
                    traceback.print_exc()
                    time.sleep(1)  # Brief pause on error
            
        except Exception as e:
            print(f"Server error: {e}")
            traceback.print_exc()
        finally:
            # Close the server socket
            try:
                self.server_socket.close()
            except:
                pass
            print("Server stopped")
    
    def generate_client_id(self):
        """Generate a unique client ID"""
        with self.client_lock:
            # Find the first available ID
            for i in range(SERVER_CONFIG.max_clients):
                if i not in self.clients:
                    return i
            # If all IDs are used, use timestamp as fallback
            return f"overflow_{int(time.time())}"
    
    def handle_client(self, client_socket, client_id):
        """Handle communication with a client"""
        print(f"Starting handler for client {client_id}", flush=True)
        
        try:
            # Set socket to non-blocking mode
            client_socket.setblocking(False)
            
            # Make buffer size for receiving
            buffer_size = 4096 # Buffer for frame data
            
            # --- Initial Ping Handshake ---
            ping_ok = False # Flag to track successful handshake
            try:
                # Set blocking briefly to wait for the initial ping header
                client_socket.setblocking(True)
                client_socket.settimeout(2.0) # 2 second timeout for ping
                ping_header = client_socket.recv(4)
                if not ping_header or len(ping_header) < 4:
                    print(f"Client {client_id} disconnected (no initial ping header)", flush=True)
                else:
                    print(f"Client {client_id}: Initial ping received successfully.", flush=True)
                    ping_ok = True # Signal success
            except socket.timeout:
                print(f"Client {client_id} timed out waiting for initial ping.", flush=True)
            except Exception as e:
                print(f"Client {client_id} error during initial ping: {e}", flush=True)
            finally:
                # Restore non-blocking mode for main loop
                client_socket.setblocking(False)
                client_socket.settimeout(None)
            # --- End Initial Ping Handshake ---

            # Exit loop if ping handshake failed
            if not ping_ok:
                return
            
            # Main communication loop
            while self.running:
                # Wait for data with timeout using select
                ready = select.select([client_socket], [], [], 0.1)
                if not ready[0]:
                    # No data available, continue
                    continue
                
                try:
                    # Receive data length first (4-byte integer)
                    length_data = client_socket.recv(4)
                    if not length_data or len(length_data) < 4:
                        print(f"Client {client_id} disconnected (no length header)", flush=True)
                        break
                    
                    # Unpack data length
                    data_length = struct.unpack(">I", length_data)[0]
                    
                    # Now receive the actual data
                    data = b""
                    remaining = data_length
                    
                    while remaining > 0:
                        chunk = client_socket.recv(min(buffer_size, remaining))
                        if not chunk:
                            break
                        data += chunk
                        remaining -= len(chunk)
                    
                    if len(data) < data_length:
                        print(f"Client {client_id} sent incomplete data, expected {data_length}, got {len(data)}", flush=True)
                        continue
                    
                    # Parse the frame data
                    frame = parse_frame_data(data)
                    if not frame:
                        # Send empty response on parsing failure
                        client_socket.sendall(struct.pack("bbb", 0, 0, 0))
                        continue
                    
                    # Get client state
                    with self.client_lock:
                        state = self.client_states[client_id]
                        state['frames_processed'] += 1
                    
                    # Handle save signal from game
                    if frame.save_signal:
                        print(f"Client {client_id}: SAVE SIGNAL RECEIVED", flush=True)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        save_path = f"{MODEL_DIR}/tempest_model_{timestamp}.pt"
                        
                        try:
                            self.agent.save(save_path)
                            self.agent.save(LATEST_MODEL_PATH)
                            print(f"Model saved to {save_path}", flush=True)
                        except Exception as e:
                            print(f"ERROR saving model: {e}", flush=True)
                    
                    # Update global metrics
                    current_frame = self.metrics.update_frame_count()
                    self.metrics.update_epsilon()
                    self.metrics.update_expert_ratio()
                    self.metrics.update_game_state(frame.enemy_seg, frame.open_level)
                    
                    # Process previous step's results if available
                    if state['last_state'] is not None and state['last_action_idx'] is not None:
                        # Add experience to replay memory
                        self.agent.step(
                            state['last_state'],
                            np.array([[state['last_action_idx']]]),
                            frame.reward,
                            frame.state,
                            frame.done
                        )
                        
                        # Track rewards
                        state['total_reward'] += frame.reward
                        
                        # Track which system's rewards
                        if hasattr(metrics, 'last_action_source'):
                            if metrics.last_action_source == "expert":
                                state['episode_expert_reward'] += frame.reward
                            else:
                                state['episode_dqn_reward'] += frame.reward
                    
                    # Handle episode completion
                    if frame.done:
                        if not state['was_done']:
                            self.metrics.add_episode_reward(
                                state['total_reward'],
                                state['episode_dqn_reward'],
                                state['episode_expert_reward']
                            )
                            print(f"Client {client_id}: Episode complete, reward={state['total_reward']:.2f}", flush=True)
                        
                        state['was_done'] = True
                        client_socket.sendall(struct.pack("bbb", 0, 0, 0))
                        state['last_state'] = None
                        state['last_action_idx'] = None
                        continue
                    elif state['was_done']:
                        state['was_done'] = False
                        state['total_reward'] = 0
                        state['episode_dqn_reward'] = 0
                        state['episode_expert_reward'] = 0
                    
                    # Generate action
                    self.metrics.increment_total_controls()
                    
                    # Decide between expert system and DQN
                    if random.random() < self.metrics.get_expert_ratio() and not self.metrics.is_override_active():
                        # Use expert system
                        fire, zap, spinner = get_expert_action(
                            frame.enemy_seg, frame.player_seg, frame.open_level
                        )
                        self.metrics.increment_guided_count()
                        self.metrics.update_action_source("expert")
                        action_idx = expert_action_to_index(fire, zap, spinner)
                    else:
                        # Use DQN with current epsilon
                        action_idx = self.agent.act(frame.state, self.metrics.get_epsilon())
                        fire, zap, spinner = ACTION_MAPPING[action_idx]
                        self.metrics.update_action_source("dqn")
                    
                    # Store state and action for next iteration
                    state['last_state'] = frame.state
                    state['last_action_idx'] = action_idx
                    
                    # Send action to game
                    game_fire, game_zap, game_spinner = encode_action_to_game(fire, zap, spinner)
                    client_socket.sendall(struct.pack("bbb", game_fire, game_zap, game_spinner))
                    
                    # Periodic target network update (only from client 0)
                    if client_id == 0 and current_frame % RL_CONFIG.update_target_every == 0:
                        self.agent.update_target_network()
                        print(f"Updated target network at frame {current_frame}", flush=True)
                    
                    # Periodic model saving (only from client 0)
                    if client_id == 0 and current_frame % RL_CONFIG.save_interval == 0:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        save_path = f"{MODEL_DIR}/tempest_model_{timestamp}.pt"
                        self.agent.save(save_path)
                        self.agent.save(LATEST_MODEL_PATH)
                        print(f"Periodic save at frame {current_frame}", flush=True)
                    
                    # Periodic client status update
                    if state['frames_processed'] % 1000 == 0:
                        duration = datetime.now() - state['connected_time']
                        print(f"Client {client_id}: Processed {state['frames_processed']} frames, "
                              f"connected for {duration}, latest reward: {state['total_reward']:.2f}", flush=True)
                
                except BlockingIOError:
                    # Expected with non-blocking socket
                    time.sleep(0.001)
                except ConnectionResetError:
                    print(f"Client {client_id} connection reset", flush=True)
                    break
                except BrokenPipeError:
                    print(f"Client {client_id} connection broken", flush=True)
                    break
                except Exception as e:
                    print(f"Error handling client {client_id}: {e}", flush=True)
                    traceback.print_exc()
                    break
        
        finally:
            # Clean up client resources
            try:
                client_socket.close()
            except:
                pass
            
            # Remove client from tracking
            with self.client_lock:
                if client_id in self.clients:
                    del self.clients[client_id]
                print(f"Client {client_id} disconnected, {len(self.clients)} clients remaining", flush=True)

def parse_frame_data(data: bytes) -> Optional[FrameData]:
    """Parse binary frame data from Lua into game state"""
    try:
        if not data or len(data) < 10:  # Minimal size check
            return None
        
        format_str = ">IdBBBIIBBBhBhBB"
        header_size = struct.calcsize(format_str)
        
        if len(data) < header_size:
            print(f"Received data too small: {len(data)} bytes, need {header_size}", flush=True)
            return None
            
        values = struct.unpack(format_str, data[:header_size])
        num_values, reward, game_action, game_mode, done, frame_counter, score, \
        save_signal, fire, zap, spinner, is_attract, nearest_enemy, player_seg, is_open = values
        
        # Debug: Print when save signal is received in raw data
        if save_signal:
            print(f"\nDEBUG: Raw save signal received in header: {save_signal}", flush=True)
        
        state_data = data[header_size:]
        
        # Safely process state values with error handling
        state_values = []
        for i in range(0, len(state_data), 2):
            if i + 1 < len(state_data):
                try:
                    value = struct.unpack(">H", state_data[i:i+2])[0] - 32768
                    state_values.append(value)
                except struct.error:
                    continue  # Skip this value on error
        
        state = np.array(state_values, dtype=np.float32) / 32768.0
        
        # Ensure state has correct dimensions
        if len(state) > SERVER_CONFIG.params_count:
            state = state[:SERVER_CONFIG.params_count]
        elif len(state) < SERVER_CONFIG.params_count:
            state = np.pad(state, (0, SERVER_CONFIG.params_count - len(state)))
                
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
        
        # Debug: Print when save signal is set in FrameData
        if frame_data.save_signal:
            print(f"DEBUG: Created FrameData with save_signal=True", flush=True)
            
        return frame_data
    except Exception as e:
        print(f"Error parsing frame data: {e}", flush=True)
        return None

def display_metrics_header(kb=None):
    """Display header for metrics table"""
    if not IS_INTERACTIVE: return
    header = (
        f"{'Frame':>8} | {'Mean Reward':>12} | {'DQN Reward':>10} | {'Loss':>8} | "
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
    
    row = (
        f"{metrics.frame_count:8d} | {mean_reward:12.2f} | {dqn_reward:10.2f} | "
        f"{mean_loss:8.4f} | {metrics.epsilon:7.3f} | {guided_ratio*100:7.2f}% | "
        f"{mem_size:8d} | {'Open' if metrics.open_level else 'Closed':10} | {'ON' if metrics.override_expert else 'OFF':10}"
    )
    print_with_terminal_restore(kb, row)

def get_expert_action(enemy_seg, player_seg, is_open_level):
    """Calculate expert-guided action based on game state"""
    if enemy_seg == player_seg:
        return 1, 0, 0  # Fire only when no enemies or aligned
        
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

def main():
    """Main function to launch server with multi-client support"""
    setup_environment()
    
    print("\nStarting Tempest AI Server with socket communication", flush=True)
    
    # Initialize DQN agent (shared across all clients)
    agent = DQNAgent(SERVER_CONFIG.params_count, len(ACTION_MAPPING))
    
    # Try to load existing model
    if not agent.load(LATEST_MODEL_PATH):
        print("No existing model found. Starting with new model.", flush=True)
    
    # Setup keyboard handler ONLY if interactive
    kb_handler = None
    if IS_INTERACTIVE:
        kb_handler = KeyboardHandler()
        print("Running in interactive mode. Press 'o' to toggle expert override, 'q' to quit.", flush=True)
    else:
        print("Running in non-interactive mode (background/redirected). Keyboard input disabled.", flush=True)
    
    # Initialize metrics and wrap in thread-safe container
    metrics.expert_ratio = SERVER_CONFIG.expert_ratio_start
    metrics.last_decay_step = 0
    safe_metrics = SafeMetrics(metrics)
    
    print("\n" + "="*80, flush=True)
    print("Tempest AI Socket Server Started", flush=True)
    print("="*80 + "\n", flush=True)
    
    # Create and start the socket server
    server = SocketServer(SERVER_CONFIG.host, SERVER_CONFIG.port, agent, safe_metrics)
    
    # Start a stats thread or keyboard handler loop based on interactivity
    def stats_reporter():
        last_stats_time = time.time()
        while True:
            time.sleep(1)
            
            # Every 10 seconds, display summary stats
            current_time = time.time()
            if current_time - last_stats_time >= 10:
                with safe_metrics.lock:
                    frame_count = metrics.frame_count
                    mean_reward = np.mean(list(metrics.episode_rewards)) if metrics.episode_rewards else float('nan')
                    epsilon = metrics.epsilon
                    expert_ratio = metrics.expert_ratio
                    memory_size = len(agent.memory)
                    guided_pct = (metrics.guided_count / max(1, metrics.total_controls)) * 100
                    client_count = len(server.clients)
                
                # Use print_with_terminal_restore if interactive, regular print otherwise
                status_func = print_with_terminal_restore if IS_INTERACTIVE else print
                status_func(kb_handler if IS_INTERACTIVE else None,
                               f"Status: Clients={client_count}, Frame={frame_count}, "
                               f"Reward={mean_reward:.2f}, ε={epsilon:.3f}, "
                               f"Expert={expert_ratio*100:.1f}%, Memory={memory_size}")
                last_stats_time = current_time
    
    # Start stats reporter thread (runs regardless of interactivity)
    stats_thread = threading.Thread(target=stats_reporter, daemon=True)
    stats_thread.start()
    
    # Start server thread
    server_thread = threading.Thread(target=server.start, daemon=True)
    server_thread.start()
    
    try:
        # If interactive, handle keyboard input
        if IS_INTERACTIVE:
            with kb_handler:
                display_metrics_header(kb_handler)
                display_interval = 5.0 # Update display every 5 seconds
                last_display_time = time.time()

                while server.running:
                    char = kb_handler.check_key()
                    if char:
                        if char == 'q':
                            print_with_terminal_restore(kb_handler, "\n'q' pressed, shutting down...")
                            server.running = False
                            break
                        elif char == 'o':
                            metrics.toggle_override(kb_handler) # Pass handler for printing
                    
                    # Update metrics display periodically
                    current_time = time.time()
                    if current_time - last_display_time >= display_interval:
                        display_metrics_row(agent, kb_handler)
                        last_display_time = current_time
                    
                    time.sleep(0.01) # Prevent busy-waiting for keys
        else:
            # If not interactive, just wait for server thread to finish (e.g., on error or external signal)
            server_thread.join()

    except KeyboardInterrupt:
        print("\nShutting down server (KeyboardInterrupt)...")

if __name__ == "__main__":
    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting gracefully...")