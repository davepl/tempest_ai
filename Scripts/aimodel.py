#!/usr/bin/env python3
"""
Tempest AI Model with BC-to-RL Transition
Author: Dave Plummer (davepl) and various AI assists
Date: 2023-03-06 (Updated)

This script implements a hybrid AI model for the Tempest arcade game that:
1. Uses Behavioral Cloning (BC) during attract mode to learn from the game's AI
2. Uses Reinforcement Learning (RL) during actual gameplay
3. Transfers knowledge from BC to RL for efficient learning
"""

import os
import sys
import time
import struct
import random
import stat
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Define the paths to the named pipes
LUA_TO_PY_PIPE = "/tmp/lua_to_py"
PY_TO_LUA_PIPE = "/tmp/py_to_lua"

# Define action mapping
ACTION_MAP = {
    0: "fire",
    1: "zap",
    2: "left",
    3: "right",
    4: "none"
}

class TempestEnv(gym.Env):
    """
    Custom Gymnasium environment for Tempest arcade game.
    """
    metadata = {"render_modes": ["human"], "render_fps": 60}
    
    def __init__(self):
        super().__init__()
        
        # Define action space: fire, zap, left, right, none
        self.action_space = spaces.Discrete(5)
        
        # Define observation space - 117 features based on game state
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(117,), dtype=np.float32)
        
        # Initialize state
        self.state = np.zeros(117, dtype=np.float32)
        self.reward = 0
        self.done = False
        self.info = {}
        self.episode_step = 0
        self.total_reward = 0
        self.is_attract_mode = False
        self.prev_state = None
        
        print("Tempest Gymnasium environment initialized")
    
    def reset(self, seed=None, options=None):
        """Reset the environment to start a new episode."""
        super().reset(seed=seed)
        
        self.state = np.zeros(117, dtype=np.float32)
        self.reward = 0
        self.done = False
        self.episode_step = 0
        self.total_reward = 0
        self.prev_state = None
        
        return self.state, self.info
    
    def step(self, action):
        """
        Take a step in the environment using the given action.
        """
        self.episode_step += 1
        self.total_reward += self.reward
        
        terminated = self.done  # Use Lua-provided done flag
        truncated = self.episode_step >= 10000  # Episode too long
        
        self.info = {
            "action_taken": ACTION_MAP[action],
            "episode_step": self.episode_step,
            "total_reward": self.total_reward,
            "attract_mode": self.is_attract_mode
        }
        
        return self.state, self.reward, terminated, truncated, self.info
    
    def update_state(self, new_state, reward, game_action=None, done=False):
        """
        Update the environment state with new data from the game.
        """
        self.prev_state = self.state.copy() if self.state is not None else None
        self.state = new_state
        self.reward = reward
        self.done = done
        if game_action is not None:
            self.info["game_action"] = game_action
        return self.state

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(next_states), np.array(dones, dtype=np.float32))
    
    def __len__(self):
        return len(self.buffer)

class UnifiedModel(nn.Module):
    """
    A neural network model that can be used for both Behavioral Cloning and RL
    """
    def __init__(self, input_size, output_size):
        super().__init__()
        
        # Define a shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Define policy head
        self.policy_head = nn.Linear(128, output_size)
        
        # Define value head for RL
        self.value_head = nn.Linear(128, output_size)
    
    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)
        
        # Get policy and value outputs
        policy_output = self.policy_head(features)
        value_output = self.value_head(features)
        
        return policy_output, value_output
    
    def act(self, state, epsilon=0.1):
        """
        Select an action using epsilon-greedy policy
        """
        if random.random() < epsilon:
            return random.randint(0, 4)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                policy_output, _ = self.forward(state_tensor)
                return torch.argmax(policy_output, dim=1).item()

def process_frame_data(data):
    """
    Process the binary frame data received from Lua.
    
    Args:
        data (bytes): Binary data containing OOB header and game state information
        
    Returns:
        tuple: (processed_data, frame_counter, reward, game_action, is_attract, done)
    """
    if len(data) < 15:  # Header (4+8) + action (1) + mode (1) + done (1)
        print(f"Warning: Data too small ({len(data)} bytes)")
        return None, 0, 0.0, None, False, False
    
    try:
        # Extract out-of-band information
        num_oob_values = struct.unpack(">I", data[0:4])[0]
        reward = struct.unpack(">d", data[4:12])[0]
        game_action = struct.unpack(">B", data[12:13])[0]
        game_mode = struct.unpack(">B", data[13:14])[0]
        done = struct.unpack(">B", data[14:15])[0] != 0
        
        # Debug output for game mode occasionally
        if random.random() < 0.01:  # Show debug info about 1% of the time
            print(f"Game Mode: 0x{game_mode:02X}, Is Attract Mode: {(game_mode & 0x80) == 0}")
            print(f"OOB Data: values={num_oob_values}, reward={reward:.2f}, action={game_action}, done={done}")
        
        # Calculate header size: 4 bytes for count + (num_oob_values * 8) bytes for values + 3 bytes for extra data
        header_size = 4 + (num_oob_values * 8) + 3
        
        # Extract game state data (everything after the header)
        game_data = data[header_size:]
        
        # Calculate how many 16-bit integers we have in the game data
        num_ints = len(game_data) // 2
        
        # Unpack the binary data into integers
        unpacked_data = []
        for i in range(num_ints):
            value = struct.unpack(">H", game_data[i*2:i*2+2])[0]
            # Convert from offset encoding (values were sent with +32768)
            value = value - 32768
            unpacked_data.append(value)
        
        # Extract frame counter from the game state data
        frame_counter = unpacked_data[6] if len(unpacked_data) > 6 else 0
        
        # Normalize the data to -1 to 1 range for the neural network
        normalized_data = np.array([float(x) / 32767.0 if x > 0 else float(x) / 32768.0 for x in unpacked_data], dtype=np.float32)
        
        # Check if we're in attract mode (bit 0x80 of game_mode is clear)
        is_attract = (game_mode & 0x80) == 0
        
        # Debug output for attract mode transitions
        if hasattr(process_frame_data, 'last_attract_mode') and process_frame_data.last_attract_mode != is_attract:
            print(f"ATTRACT MODE TRANSITION: {'Attract → Play' if not is_attract else 'Play → Attract'}")
            print(f"Game Mode: 0x{game_mode:02X}, Is Attract: {is_attract}")
        
        # Store for next comparison
        process_frame_data.last_attract_mode = is_attract
        
        # Pad or truncate to match the expected observation space size
        expected_size = 117
        if len(normalized_data) < expected_size:
            padded_data = np.zeros(expected_size, dtype=np.float32)
            padded_data[:len(normalized_data)] = normalized_data
            normalized_data = padded_data
        elif len(normalized_data) > expected_size:
            normalized_data = normalized_data[:expected_size]
        
        return normalized_data, frame_counter, reward, game_action, is_attract, done
    
    except Exception as e:
        print(f"Error processing frame data: {e}")
        return None, 0, 0.0, None, False, False

# Initialize static variable for process_frame_data
process_frame_data.last_attract_mode = True

# Create a global environment instance
env = TempestEnv()

# Create a unified model instance
model = UnifiedModel(117, 5)
optimizer = optim.Adam(model.parameters(), lr=0.001)
replay_buffer = ReplayBuffer(10000)
batch_size = 32
gamma = 0.99  # Discount factor
epsilon = 0.5  # Starting exploration rate
epsilon_decay = 0.995  # Exploration decay rate
min_epsilon = 0.1  # Minimum exploration rate

# Loss functions
bc_loss_fn = nn.CrossEntropyLoss()
rl_loss_fn = nn.MSELoss()

# Training stats
bc_episodes = 0
rl_episodes = 0
bc_losses = []
rl_losses = []
rewards_history = []

# Track mode transitions for logging
last_mode_was_attract = True
mode_transitions = 0

def train_bc(state, action):
    """Train the model using Behavioral Cloning"""
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    action_tensor = torch.LongTensor([action])
    
    # Forward pass
    policy_output, _ = model(state_tensor)
    
    # Calculate loss
    loss = bc_loss_fn(policy_output, action_tensor)
    
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

def train_rl(batch_size):
    """Train the model using Reinforcement Learning (DQN)"""
    if len(replay_buffer) < batch_size:
        return 0.0
    
    # Sample from replay buffer
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
    
    # Convert to tensors
    states_tensor = torch.FloatTensor(states)
    actions_tensor = torch.LongTensor(actions)
    rewards_tensor = torch.FloatTensor(rewards)
    next_states_tensor = torch.FloatTensor(next_states)
    dones_tensor = torch.FloatTensor(dones)
    
    # Get current Q values
    current_q_values = model(states_tensor)[1].gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
    
    # Get next Q values (target network)
    with torch.no_grad():
        next_q_values = model(next_states_tensor)[1].max(1)[0]
        target_q_values = rewards_tensor + gamma * next_q_values * (1 - dones_tensor)
    
    # Calculate loss
    loss = rl_loss_fn(current_q_values, target_q_values)
    
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

def ai_model(game_state, frame_counter, reward, game_action, is_attract, done):
    """
    AI model that determines the action based on the game state.
    Uses Behavioral Cloning in attract mode, RL in normal play.
    Implements bidirectional knowledge transfer between modes.
    
    Args:
        game_state (numpy.ndarray): Processed game state data
        frame_counter (int): Current frame counter
        reward (float): Current reward value
        game_action (int): Action from the game
        is_attract (bool): Whether we're in attract mode
        done (bool): Whether the episode is done
        
    Returns:
        str: Action to take (fire, zap, left, right, none)
    """
    global epsilon, bc_episodes, rl_episodes, last_mode_was_attract, mode_transitions
    
    # Detect mode transition
    if is_attract != last_mode_was_attract:
        mode_transitions += 1
        print(f"\n*** MODE TRANSITION #{mode_transitions}: {'Attract → Play' if not is_attract else 'Play → Attract'} ***")
        print(f"Frame: {frame_counter}, Is Attract: {is_attract}, Game Action: {game_action}")
        last_mode_was_attract = is_attract
    
    # Update environment
    env.update_state(game_state, reward, game_action, done)
    env.is_attract_mode = is_attract
    
    # Store transition in replay buffer (if we have a previous state)
    if env.prev_state is not None:
        # For replay buffer, use game_action in attract mode, model action in play mode
        if is_attract:
            # In attract mode, we use the game's action for learning (BC)
            if game_action is not None and game_action < 5:  # Ensure it's a valid action
                replay_buffer.add(env.prev_state, game_action, reward, game_state, done)
                if frame_counter % 100 == 0:
                    print(f"Added attract mode transition to buffer: action={ACTION_MAP[game_action]}")
        else:
            # In play mode, we use the model's action (RL)
            model_action = model.act(env.prev_state)
            replay_buffer.add(env.prev_state, model_action, reward, game_state, done)
            if frame_counter % 100 == 0:
                print(f"Added play mode transition to buffer: action={ACTION_MAP[model_action]}")
    
    # Different behavior based on mode
    if is_attract:
        # Behavioral Cloning mode
        # Log the attract mode state occasionally
        if frame_counter % 300 == 0:
            print(f"In ATTRACT mode - Frame {frame_counter}, Buffer size: {len(replay_buffer)}")
        
        # Ensure we have a valid game action
        if game_action is not None and game_action < 5:
            # Learn from the game's action using BC
            loss = train_bc(game_state, game_action)
            bc_losses.append(loss)
            
            # Log progress occasionally
            if frame_counter % 100 == 0:
                print(f"BC training - Frame {frame_counter}, Action: {ACTION_MAP[game_action]}, Loss: {loss:.6f}")
            
            # In attract mode, just return the game's action
            action = game_action
        else:
            # If we don't have a valid game_action, choose a random one
            action = random.randint(0, 4)
            print(f"Warning: Invalid game_action {game_action} in attract mode, using random action")
        
        # Track BC episodes
        if done:
            bc_episodes += 1
            avg_loss = np.mean(bc_losses[-100:]) if bc_losses else 0
            print(f"BC Episode {bc_episodes} completed, avg loss: {avg_loss:.6f}")
    else:
        # Reinforcement Learning mode
        # Log the play mode state occasionally
        if frame_counter % 300 == 0:
            print(f"In PLAY mode - Frame {frame_counter}, Buffer size: {len(replay_buffer)}")
        
        # Decay exploration rate - we decay more slowly if we've had more mode transitions
        # This ensures that after returning from BC, we still explore a bit
        decay_factor = epsilon_decay * (0.95 + 0.05 * min(mode_transitions, 10))
        epsilon = max(min_epsilon, epsilon * decay_factor)
        
        # Choose an action using the model with epsilon-greedy
        action = model.act(game_state, epsilon)
        
        # Train the model with RL if we have enough samples
        if len(replay_buffer) > batch_size:
            rl_loss = train_rl(batch_size)
            rl_losses.append(rl_loss)
            
            # Log progress occasionally
            if frame_counter % 100 == 0:
                print(f"RL training - Frame {frame_counter}, Epsilon: {epsilon:.4f}, Loss: {rl_loss:.6f}")
        
        # Track RL episodes and save model periodically
        if done:
            rl_episodes += 1
            rewards_history.append(env.total_reward)
            avg_reward = np.mean(rewards_history[-10:]) if rewards_history else 0
            print(f"RL Episode {rl_episodes} completed, reward: {env.total_reward:.2f}, avg reward: {avg_reward:.2f}")
            
            # Save model periodically
            if rl_episodes % 10 == 0:
                torch.save(model.state_dict(), f"tempest_model_ep{rl_episodes}.pt")
    
    # Map the action to a string command
    action_str = ACTION_MAP[action]
    
    # Log the action (for debugging) - simplified to reduce console spam
    if frame_counter % 30 == 0:
        mode_str = "Attract (BC)" if is_attract else "Play (RL)"
        print(f"Frame {frame_counter}: [{mode_str}] Action: {action_str}, Reward: {reward:.2f}")
    
    return action_str

def main():
    """
    Main function that handles the communication with Lua and processes game frames.
    """
    print("Python AI model starting with bidirectional BC/RL knowledge transfer...")
    
    # Initialize the environment
    env.reset()
    
    # Load a previously saved model if it exists
    model_path = "tempest_model_final.pt"
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path))
            print(f"Loaded existing model from {model_path}")
        except Exception as e:
            print(f"Failed to load model: {e}")
    
    # Create the pipes (same as before)
    for pipe_path in [LUA_TO_PY_PIPE, PY_TO_LUA_PIPE]:
        try:
            if os.path.exists(pipe_path):
                os.unlink(pipe_path)
            os.mkfifo(pipe_path)
            os.chmod(pipe_path, 0o666)
            print(f"Created pipe: {pipe_path}")
        except OSError as e:
            print(f"Error with pipe {pipe_path}: {e}")
            sys.exit(1)
    
    print("Pipes created successfully. Waiting for Lua connection...")
    
    # Connection retry loop
    while True:
        try:
            # Open pipes in non-blocking mode to avoid deadlock
            fd = os.open(LUA_TO_PY_PIPE, os.O_RDONLY | os.O_NONBLOCK)
            lua_to_py = os.fdopen(fd, "rb")
            print("Input pipe opened successfully")
            
            py_to_lua = open(PY_TO_LUA_PIPE, "w")
            print("Output pipe opened successfully")
            
            try:
                frame_count = 0
                last_frame_time = time.time()
                fps = 0
                
                while True:
                    try:
                        # Read from pipe
                        data = lua_to_py.read()
                        
                        if not data:
                            time.sleep(0.01)
                            continue
                        
                        # Process the frame data
                        result = process_frame_data(data)
                        if result is None or result[0] is None:
                            print("Error processing frame data, skipping frame")
                            continue
                        
                        processed_data, frame_counter, reward, game_action, is_attract, done = result
                        
                        # Get the action from the AI model
                        action = ai_model(processed_data, frame_counter, reward, game_action, is_attract, done)
                        
                        # Write the action back to Lua
                        py_to_lua.write(action + "\n")
                        py_to_lua.flush()
                        
                        # Calculate and display FPS occasionally
                        frame_count += 1
                        if frame_count % 100 == 0:
                            current_time = time.time()
                            fps = 100 / (current_time - last_frame_time)
                            last_frame_time = current_time
                            
                            # Log overall statistics
                            bc_loss = np.mean(bc_losses[-100:]) if bc_losses else 0
                            rl_loss = np.mean(rl_losses[-100:]) if rl_losses else 0
                            print(f"Stats: {frame_count} frames, {fps:.1f} FPS, Mode transitions: {mode_transitions}")
                            print(f"       BC episodes: {bc_episodes}, BC loss: {bc_loss:.6f}")
                            print(f"       RL episodes: {rl_episodes}, RL loss: {rl_loss:.6f}")
                    
                    except BlockingIOError:
                        # Expected in non-blocking mode
                        time.sleep(0.01)
                        continue
            
            finally:
                lua_to_py.close()
                py_to_lua.close()
                print("Pipes closed, reconnecting...")
        
        except KeyboardInterrupt:
            print("Interrupted by user")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)
    
    print("Python AI model shutting down")
    # Save final model
    torch.save(model.state_dict(), "tempest_model_final.pt")

if __name__ == "__main__":
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    main()
        

