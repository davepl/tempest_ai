#!/usr/bin/env python3
"""
Tempest AI Model: Hybrid expert-guided and DQN-based gameplay system.
- Makes intelligent decisions based on enemy positions and level types
- Uses a Deep Q-Network (DQN) for reinforcement learning
- Expert system provides guidance and training examples
- Communicates with Tempest via Lua pipes
"""

import os
import time
import struct
import random
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

# Configuration constants
PIPE_CONFIG = {
    "lua_to_py": "/tmp/lua_to_py",
    "py_to_lua": "/tmp/py_to_lua",
    "params_count": 128,
    "expert_ratio_start": 0.4,    # Start with these level of expert guidance
    "expert_ratio_min": 0.01,     # Minimum expert guidance (10%)
    "expert_ratio_decay": 0.98,  # Multiply by 0.98 each decay step
    "expert_ratio_decay_steps": 10000,  # Decay every 10,000 frames
    "expert_ratio_cycle": False,    # If true, reset expert ratio every 10,000 frames
    "reset_frame_count": False,  # Set to True to reset frame count when loading a model
    "reset_expert_ratio": True   # Set to True to reset expert ratio to start value
}
MODEL_DIR = "models"
LATEST_MODEL_PATH = f"{MODEL_DIR}/tempest_model_latest.pt"

# RL hyperparameters
RL_CONFIG = {
    "batch_size": 512,
    "gamma": 0.99,         # Discount factor
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay": 10000,
    "update_target_every": 1000,
    "learning_rate": 1e-4,
    "memory_size": 500000,
    "save_interval": 50000,
    "train_freq": 4
}

# Define action space
# We'll discretize the spinner movement into 7 values:
# -3: hard left, -2: medium left, -1: soft left, 0: center, 1: soft right, 2: medium right, 3: hard right
# And 2 options for fire button (0: no fire, 1: fire)
# Total action space: 14 actions (7 spinner positions × 2 fire options)
ACTION_MAPPING = {
    0: (0, 0, -0.3),   # Hard left, no fire
    1: (0, 0, -0.2),   # Medium left, no fire
    2: (0, 0, -0.1),   # Soft left, no fire
    3: (0, 0, 0.0),    # Center, no fire
    4: (0, 0, 0.1),    # Soft right, no fire
    5: (0, 0, 0.2),    # Medium right, no fire
    6: (0, 0, 0.3),    # Hard right, no fire
    7: (1, 0, -0.3),   # Hard left, fire
    8: (1, 0, -0.2),   # Medium left, fire
    9: (1, 0, -0.1),   # Soft left, fire
    10: (1, 0, 0.0),   # Center, fire
    11: (1, 0, 0.1),   # Soft right, fire
    12: (1, 0, 0.2),   # Medium right, fire
    13: (1, 0, 0.3),   # Hard right, fire
}

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else 
                      "mps" if torch.backends.mps.is_available() else "cpu")

# Metrics tracking
metrics = {
    "frame_count": 0,
    "guided_count": 0,
    "total_controls": 0,
    "episode_rewards": deque(maxlen=20),
    "dqn_rewards": deque(maxlen=20),
    "expert_rewards": deque(maxlen=20),
    "losses": deque(maxlen=100),
    "epsilon": RL_CONFIG["epsilon_start"],
    "expert_ratio": PIPE_CONFIG["expert_ratio_start"],
    "last_decay_step": 0
}

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
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=RL_CONFIG["learning_rate"])
        
        # Replay memory
        self.memory = ReplayMemory(RL_CONFIG["memory_size"])
        
        # Initialize target network with same weights as local network
        self.update_target_network()
        
        # Training queue for background thread
        self.train_queue = queue.Queue(maxsize=100)
        self.training_thread = None
        self.running = True
        self.start_training_thread()
        
    def start_training_thread(self):
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
        if len(self.memory) < RL_CONFIG["batch_size"]:
            return
            
        # Sample batch from memory
        states, actions, rewards, next_states, dones = self.memory.sample(RL_CONFIG["batch_size"])
        
        # Get Q values for current states
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        # Compute target Q values (Double DQN approach)
        with torch.no_grad():
            best_actions = self.qnetwork_local(next_states).argmax(1, keepdim=True)
            Q_targets_next = self.qnetwork_target(next_states).gather(1, best_actions)
            Q_targets = rewards + (RL_CONFIG["gamma"] * Q_targets_next * (1 - dones))
        
        # Compute loss and perform optimization
        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to stabilize training
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 1)
        self.optimizer.step()
        
        # Update metrics
        metrics["losses"].append(loss.item())
    
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
            'epsilon': metrics["epsilon"],
            'frame_count': metrics["frame_count"],
            'expert_ratio': metrics["expert_ratio"]
        }, filename)
        print(f"Model saved to {filename} (frame {metrics['frame_count']}, expert ratio {metrics['expert_ratio']:.2f})")
        
    def load(self, filename):
        """Load model weights"""
        if os.path.exists(filename):
            try:
                checkpoint = torch.load(filename, map_location=device)
                self.qnetwork_local.load_state_dict(checkpoint['policy_state_dict'])
                self.qnetwork_target.load_state_dict(checkpoint['target_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                # Load frame count and epsilon (for exploration)
                metrics["epsilon"] = checkpoint.get('epsilon', RL_CONFIG["epsilon_start"])
                metrics["frame_count"] = checkpoint.get('frame_count', 0)
                
                # Always set the expert ratio to the start value
                metrics["expert_ratio"] = PIPE_CONFIG["expert_ratio_start"]
                metrics["last_decay_step"] = 0
                
                print(f"Loaded model from frame {metrics['frame_count']}, expert ratio reset to {metrics['expert_ratio']:.2f}")
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
                return False
        return False

def setup_environment():
    """Set up pipes and model directory"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    for pipe in [PIPE_CONFIG["lua_to_py"], PIPE_CONFIG["py_to_lua"]]:
        if os.path.exists(pipe):
            os.unlink(pipe)
        os.mkfifo(pipe)
        os.chmod(pipe, 0o666)

def parse_frame_data(data):
    """Parse binary frame data from Lua into game state"""
    if not data:
        return None
    
    # Expected format: timestamp, reward, actions, mode, done, counters, etc.
    format_str = ">IdBBBIIBBBhBhBB"
    header_size = struct.calcsize(format_str)
    
    if len(data) < header_size:
        return None
        
    # Extract header values
    values = struct.unpack(format_str, data[:header_size])
    num_values, reward, game_action, game_mode, done, frame_counter, score, \
    save_signal, fire, zap, spinner, is_attract, nearest_enemy, player_seg, is_open = values
    
    # Log save signal if received
    if save_signal:
        print(f"Save signal detected: {save_signal}")
    
    # Process remaining game state data
    state_data = data[header_size:]
    state_values = [
        struct.unpack(">H", state_data[i:i+2])[0] - 32768
        for i in range(0, len(state_data), 2)
        if i + 1 < len(state_data)
    ]
    
    # Normalize and pad/truncate state array
    state = np.array(state_values, dtype=np.float32) / 32768.0
    state = (state[:PIPE_CONFIG["params_count"]] if len(state) > PIPE_CONFIG["params_count"]
            else np.pad(state, (0, PIPE_CONFIG["params_count"] - len(state))))
            
    return {
        "state": state,
        "reward": reward,
        "action": (bool(fire), bool(zap), spinner),
        "mode": game_mode,
        "done": bool(done),
        "attract": bool(is_attract),
        "save_signal": bool(save_signal),
        "enemy_seg": nearest_enemy,
        "player_seg": player_seg,
        "open_level": bool(is_open)
    }

def display_metrics_header():
    """Display header for metrics table"""
    header = (
        f"{'Frame':>8} | {'Mean Reward':>12} | {'DQN Reward':>10} | {'Loss':>8} | "
        f"{'Epsilon':>7} | {'Guided %':>8} | {'Mem Size':>8} | {'Level Type':>10}"
    )
    print(f"\n{'-' * len(header)}\n{header}\n{'-' * len(header)}")

def display_metrics_row(agent):
    """Display current metrics in tabular format"""
    mean_reward = np.mean(list(metrics["episode_rewards"])) if metrics["episode_rewards"] else float('nan')
    dqn_reward = np.mean(list(metrics["dqn_rewards"])) if metrics["dqn_rewards"] else float('nan')
    mean_loss = np.mean(list(metrics["losses"])) if metrics["losses"] else float('nan')
    # Calculate guided ratio from the current expert_ratio setting, not from count history
    guided_ratio = metrics["expert_ratio"]
    expert_counts = f"{metrics['guided_count']}/{metrics['total_controls']}"
    mem_size = len(agent.memory) if agent else 0
    
    row = (
        f"{metrics['frame_count']:8d} | {mean_reward:12.2f} | {dqn_reward:10.2f} | "
        f"{mean_loss:8.4f} | {metrics['epsilon']:7.3f} | {guided_ratio*100:7.2f}% ({expert_counts}) | "
        f"{mem_size:8d} | {'Open' if metrics.get('open_level', False) else 'Closed':10}"
    )
    print(row)

def get_expert_action(enemy_seg, player_seg, is_open_level):
    """Calculate expert-guided action based on game state"""
    if enemy_seg < 0 or enemy_seg == player_seg:
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
    return max(RL_CONFIG["epsilon_end"], 
               RL_CONFIG["epsilon_start"] * 
               np.exp(-frame_count / RL_CONFIG["epsilon_decay"]))

def decay_expert_ratio(current_step):
    """Update expert ratio based on 10,000 frame intervals"""
    step_interval = current_step // PIPE_CONFIG["expert_ratio_decay_steps"]
    
    # Only update if we've moved to a new interval
    if step_interval > metrics["last_decay_step"]:
        metrics["last_decay_step"] = step_interval
        if step_interval == 0:
            # First interval - use starting value
            metrics["expert_ratio"] = PIPE_CONFIG["expert_ratio_start"]
        else:
            # Apply decay
            metrics["expert_ratio"] *= PIPE_CONFIG["expert_ratio_decay"]
        
        # Ensure we don't go below the minimum
        metrics["expert_ratio"] = max(PIPE_CONFIG["expert_ratio_min"], metrics["expert_ratio"])
        
    return metrics["expert_ratio"]

def main():
    """Main game loop handling Lua communication and decisions"""
    setup_environment()
    
    # Initialize DQN agent
    agent = DQNAgent(PIPE_CONFIG["params_count"], len(ACTION_MAPPING))
    
    # Try to load existing model
    if not agent.load(LATEST_MODEL_PATH):
        print("No existing model found. Starting with new model.")
    
    # Always initialize expert ratio to the start value
    metrics["expert_ratio"] = PIPE_CONFIG["expert_ratio_start"]
    metrics["last_decay_step"] = 0
        
    # Display initial metrics header and values
    display_metrics_header()
    display_metrics_row(agent)
    
    total_reward = 0
    was_done = False
    last_state = None
    last_action_idx = None
    episode_dqn_reward = 0
    episode_expert_reward = 0
    
    with os.fdopen(os.open(PIPE_CONFIG["lua_to_py"], os.O_RDONLY | os.O_NONBLOCK), "rb") as lua_in, \
         open(PIPE_CONFIG["py_to_lua"], "wb") as lua_out:
         
        while True:
            # Check for incoming data
            if not select.select([lua_in], [], [], 0.01)[0]:
                time.sleep(0.001)
                continue
                
            data = lua_in.read()
            if not data:
                time.sleep(0.001)
                continue
                
            frame = parse_frame_data(data)
            if not frame:
                lua_out.write(struct.pack("bbb", 0, 0, 0))
                lua_out.flush()
                continue
                
            # Update metrics counter
            metrics["frame_count"] += 1
            
            # Update exploration rate and expert ratio
            metrics["epsilon"] = decay_epsilon(metrics["frame_count"])
            decay_expert_ratio(metrics["frame_count"])
            
            # Process previous step's results if we have them
            if last_state is not None and last_action_idx is not None:
                # Add experience to replay memory
                agent.step(
                    last_state, 
                    np.array([[last_action_idx]]), 
                    frame["reward"], 
                    frame["state"], 
                    frame["done"]
                )
                
                # Track total reward
                total_reward += frame["reward"]
                
                # Track which system's rewards (expert or DQN)
                if metrics.get("last_action_source") == "expert":
                    episode_expert_reward += frame["reward"]
                else:
                    episode_dqn_reward += frame["reward"]
            
            # Update episode tracking
            if frame["done"]:
                if not was_done:
                    metrics["episode_rewards"].append(total_reward)
                    if episode_dqn_reward > 0:
                        metrics["dqn_rewards"].append(episode_dqn_reward)
                    if episode_expert_reward > 0:
                        metrics["expert_rewards"].append(episode_expert_reward)
                    
                was_done = True
                lua_out.write(struct.pack("bbb", 0, 0, 0))
                lua_out.flush()
                last_state = None
                last_action_idx = None
                continue
            elif was_done:
                was_done = False
                total_reward = 0
                episode_dqn_reward = 0
                episode_expert_reward = 0
            
            # Store frame data for metrics
            metrics.update({
                "enemy_seg": frame["enemy_seg"],
                "open_level": frame["open_level"]
            })
            
            # Generate action (expert or DQN based on expert_ratio)
            metrics["total_controls"] += 1
            
            # Decide between expert system and DQN
            if random.random() < metrics["expert_ratio"]:
                # Use expert system
                fire, zap, spinner = get_expert_action(
                    frame["enemy_seg"], frame["player_seg"], frame["open_level"]
                )
                metrics["guided_count"] += 1
                metrics["last_action_source"] = "expert"
                
                # Convert expert action to index for training
                action_idx = expert_action_to_index(fire, zap, spinner)
            else:
                # Use DQN
                action_idx = agent.act(frame["state"], metrics["epsilon"])
                fire, zap, spinner = ACTION_MAPPING[action_idx]
                metrics["last_action_source"] = "dqn"
            
            # Store state and action for next frame's training
            last_state = frame["state"]
            last_action_idx = action_idx
            
            # Send action to game
            game_fire, game_zap, game_spinner = encode_action_to_game(fire, zap, spinner)
            lua_out.write(struct.pack("bbb", game_fire, game_zap, game_spinner))
            lua_out.flush()
            
            # Update target network periodically
            if metrics["frame_count"] % RL_CONFIG["update_target_every"] == 0:
                agent.update_target_network()
            
            # Update metrics display
            if metrics["frame_count"] % 1000 == 0:
                display_metrics_row(agent)
                
            # Save model periodically
            if metrics["frame_count"] % RL_CONFIG["save_interval"] == 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                agent.save(f"{MODEL_DIR}/tempest_model_{timestamp}.pt")
                agent.save(LATEST_MODEL_PATH)
                
            # Handle save signal from game
            if frame["save_signal"]:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"{MODEL_DIR}/tempest_model_{timestamp}.pt"
                agent.save(save_path)
                agent.save(LATEST_MODEL_PATH)
                print(f"\nSave signal received. Model saved to {save_path}")
                display_metrics_header()

if __name__ == "__main__":
    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting gracefully...")