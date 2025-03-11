#!/usr/bin/env python3
"""
Tempest AI Model
Author: Dave Plummer (davepl) and various AI assists
Date: 2023-03-06

This script implements a simple AI model for the Tempest arcade game.
It receives game state data from a Lua script running in MAME via a named pipe,
processes the data, and returns actions to control the game.

The script uses a named pipe for communication with the Lua script.

Installation:
To install the required dependencies, run:
    pip install numpy gymnasium

For GPU acceleration (optional but recommended for training):
    pip install torch torchvision torchaudio

For visualization tools (optional):
    pip install matplotlib pygame

Usage:
    python aimodel.py
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
        
        # Define observation space
        # This is a placeholder - adjust the shape based on your actual state representation
        self.observation_space = spaces.Box(low=-1, high=1, shape=(200,), dtype=np.float32)
        
        # Initialize state
        self.state = np.zeros(200, dtype=np.float32)
        self.reward = 0
        self.done = False
        self.info = {}
        self.episode_step = 0
        self.total_reward = 0
        self.level_shape = 0    # Level shape (level_number % 16)
        
        print("Tempest Gymnasium environment initialized")
    
    def reset(self, seed=None, options=None):
        """Reset the environment to start a new episode."""
        super().reset(seed=seed)
        
        self.state = np.zeros(200, dtype=np.float32)
        self.reward = 0
        self.done = False
        self.info = {}
        self.episode_step = 0
        self.total_reward = 0
        
        return self.state, self.info
    
    def step(self, action):
        """
        Take a step in the environment using the given action.
        
        Args:
            action (int): The action to take (0-4)
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # This is a placeholder - in a real implementation, the game state would be updated
        # based on the action, but here we're just returning the current state since the
        # actual game state is managed by MAME
        
        self.episode_step += 1
        
        # Simple reward function - this should be customized based on game events
        # For now, we'll just give a small positive reward for staying alive
        self.reward = 0.1
        
        # Check if episode is done (placeholder logic)
        terminated = False  # Game over
        truncated = self.episode_step >= 10000  # Episode too long
        
        self.total_reward += self.reward
        
        self.info = {
            "action_taken": ACTION_MAP[action],
            "episode_step": self.episode_step,
            "total_reward": self.total_reward
        }
        
        return self.state, self.reward, terminated, truncated, self.info
    
    def update_state(self, new_state):
        """
        Update the environment state with new data from the game.
        
        Args:
            new_state (numpy.ndarray): The new state data
            
        Returns:
            numpy.ndarray: The updated state
        """
        self.state = new_state
        return self.state

def process_frame_data(data):
    """
    Process the binary frame data received from Lua.
    
    Args:
        data (bytes): Binary data containing game state information
        
    Returns:
        numpy.ndarray: Processed game state as a normalized array
    """
    # Calculate how many 16-bit integers we have
    num_ints = len(data) // 2
    
    # Unpack the binary data into 16-bit signed integers (big-endian)
    unpacked_data = []
    for i in range(num_ints):
        value = struct.unpack(">h", data[i*2:i*2+2])[0]
        unpacked_data.append(value)
    
    # Normalize the data to -1 to 1 range for the neural network
    normalized_data = np.array([float(x) / 32767.0 if x > 0 else float(x) / 32768.0 for x in unpacked_data], dtype=np.float32)
    
    # Pad or truncate to match the expected observation space size
    expected_size = 200  # Should match the observation space shape
    if len(normalized_data) < expected_size:
        # Pad with zeros if the data is smaller than expected
        padded_data = np.zeros(expected_size, dtype=np.float32)
        padded_data[:len(normalized_data)] = normalized_data
        normalized_data = padded_data
    elif len(normalized_data) > expected_size:
        # Truncate if the data is larger than expected
        normalized_data = normalized_data[:expected_size]
    
    return normalized_data

# Create a global environment instance
env = TempestEnv()

def ai_model(game_state):
    """
    AI model that determines the action based on the game state using Gymnasium.
    
    Args:
        game_state (numpy.ndarray): Processed game state data
        
    Returns:
        str: Action to take (fire, zap, left, right, none)
    """
    # Update the environment state
    env.update_state(game_state)
    
    # For now, use a random policy
    # In a real RL implementation, you would use a trained policy here
    action = env.action_space.sample()
    
    # Take a step in the environment
    _, reward, terminated, truncated, info = env.step(action)
    
    # If the episode is done, reset the environment
    if terminated or truncated:
        print(f"Episode finished after {env.episode_step} steps with total reward {env.total_reward}")
        env.reset()
    
    # Map the action to a string command
    action_str = ACTION_MAP[action]
    
    # Log the action (for debugging)
    print(f"AI choosing action: {action_str} (action_id: {action}, reward: {reward:.2f})")
    
    return action_str

def main():
    """
    Main function that handles the communication with Lua and processes game frames.
    """
    print("Python AI model starting with Gymnasium integration...")
    
    # Initialize the environment
    env.reset()
    
    # Remove existing pipes to ensure clean state
    for pipe_path in [LUA_TO_PY_PIPE, PY_TO_LUA_PIPE]:
        try:
            os.unlink(pipe_path)
            print(f"Removed existing pipe: {pipe_path}")
        except FileNotFoundError:
            print(f"Pipe {pipe_path} did not exist, no need to remove")
    
    # Create the pipes
    for pipe_path in [LUA_TO_PY_PIPE, PY_TO_LUA_PIPE]:
        try:
            os.mkfifo(pipe_path)
            # Set permissions to ensure they're readable/writable
            os.chmod(pipe_path, 0o666)
            print(f"Created pipe: {pipe_path}")
        except OSError as e:
            print(f"Error creating pipe {pipe_path}: {e}")
            sys.exit(1)
    
    # Verify pipes exist
    for pipe_path in [LUA_TO_PY_PIPE, PY_TO_LUA_PIPE]:
        if os.path.exists(pipe_path):
            mode = os.stat(pipe_path).st_mode
            if stat.S_ISFIFO(mode):
                print(f"Verified {pipe_path} exists and is a named pipe")
            else:
                print(f"Warning: {pipe_path} exists but is not a named pipe!")
        else:
            print(f"Error: {pipe_path} does not exist after creation!")
            sys.exit(1)
    
    print("Pipes created successfully. Waiting for Lua connection...")
    
    # Connection retry loop
    while True:
        try:
            # IMPORTANT: Open pipes in the correct order to avoid deadlock
            # First, open the reading pipe (lua_to_py) in non-blocking mode
            # This is critical because opening a pipe for reading normally blocks until someone opens it for writing
            print("Opening input pipe (lua_to_py) in non-blocking mode...")
            fd = os.open(LUA_TO_PY_PIPE, os.O_RDONLY | os.O_NONBLOCK)
            lua_to_py = os.fdopen(fd, "rb")
            print("Input pipe opened successfully in non-blocking mode")
            
            # Then open the writing pipe (py_to_lua)
            print("Opening output pipe (py_to_lua)...")
            py_to_lua = open(PY_TO_LUA_PIPE, "w")
            print("Output pipe opened successfully")
            
            print("Connected to Lua pipes! Ready to process game frames.")
            
            try:
                frame_count = 0
                while True:
                    try:
                        # Try to read from the pipe (may return empty if no data available)
                        data = lua_to_py.read()
                        
                        if not data:
                            # In non-blocking mode, this could mean no data yet
                            # Small delay and continue
                            time.sleep(0.01)
                            continue
                        
                        # Process the frame data
                        processed_data = process_frame_data(data)
                        
                        # Get the action from the AI model
                        action = ai_model(processed_data)
                        
                        # Write the action back to Lua
                        py_to_lua.write(action + "\n")
                        py_to_lua.flush()  # Make sure the data is sent immediately
                        
                        # Log every 100 frames to avoid excessive output
                        frame_count += 1
                        if frame_count % 100 == 0:
                            print(f"Processed {frame_count} frames, last action: {action}")
                    
                    except BlockingIOError:
                        # This is expected in non-blocking mode when no data is available
                        time.sleep(0.01)
                        continue
            
            finally:
                # Clean up resources
                lua_to_py.close()
                py_to_lua.close()
                print("Pipes closed, attempting to reconnect...")
        
        except KeyboardInterrupt:
            print("Interrupted by user")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Waiting 5 seconds before retry...")
            time.sleep(5)
    
    print("Python AI model shutting down")

if __name__ == "__main__":
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    main()
        

