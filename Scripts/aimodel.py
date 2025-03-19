#!/usr/bin/env python3
"""
Tempest AI Model with BC-to-RL Transition
Author: Dave Plummer (davepl) and various AI assists
Date: 2025-03-06 (Updated)

This script implements a hybrid AI model for Tempest:
1. Uses Behavioral Cloning (BC) during attract mode to mimic the game's AI
2. Transitions to Reinforcement Learning (RL) during gameplay
"""

### Imports and Environment Setup
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Enable fallback for PyTorch on Apple Silicon
import sys
import time
import struct
import random
import argparse
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn.functional as F
import threading

# Constants
NumberOfParams = 247  # Number of parameters in the observation space

# File paths for pipes and model storage
LUA_TO_PY_PIPE = "/tmp/lua_to_py"
PY_TO_LUA_PIPE = "/tmp/py_to_lua"
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODEL_DIR, exist_ok=True)
LATEST_MODEL_PATH = os.path.join(MODEL_DIR, "tempest_model_latest.zip")
BC_MODEL_PATH = os.path.join(MODEL_DIR, "tempest_bc_model.pt")

# Command-line argument parsing
parser = argparse.ArgumentParser(description='Tempest AI Model')
parser.add_argument('--replay', type=str, help='Path to a log file to replay for BC training')
args = parser.parse_args()

# Device configuration: Prefer MPS (Apple Silicon) if available, else CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

### Environment Definition
class TempestEnv(gym.Env):
    """
    Custom Gym environment for Tempest.
    - Action space: 3D box (fire, zap, spinner_delta) in [0,1], [-1,1] ranges
    - Observation space: 247D box normalized to [-1, 1]
    """
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(low=np.array([0.0, 0.0, -1.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(NumberOfParams,), dtype=np.float32)
        self.state = np.zeros(NumberOfParams, dtype=np.float32)
        self.reward = 0.0
        self.done = False
        self.is_attract_mode = False
        self.prev_state = None

    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        self.state = np.zeros(NumberOfParams, dtype=np.float32)
        self.reward = 0.0
        self.done = False
        self.prev_state = None
        return self.state, {}

    def step(self, action):
        """Execute an action and return the next state, reward, and done flag."""
        fire = 1 if action[0] > 0.5 else 0
        zap = 1 if action[1] > 0.5 else 0
        spinner_delta = int(round(action[2] * 127.0))
        spinner_delta = max(-128, min(128, spinner_delta))
        return self.state, self.reward, self.done, False, {"action_taken": (fire, zap, spinner_delta)}

    def update_state(self, game_state, reward, game_action=None, done=False):
        """Update environment with new game state and reward."""
        self.prev_state = self.state.copy()
        self.state = game_state
        self.reward = reward
        self.done = done
        return self.state

### Neural Network Models
class TempestFeaturesExtractor(BaseFeaturesExtractor):
    """Feature extractor for RL model, transforming observations into a lower-dimensional space."""
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        self.feature_extractor = nn.Sequential(
            nn.Linear(observation_space.shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU()
        ).to(device)

    def forward(self, observations):
        return self.feature_extractor(observations)

class BCModel(nn.Module):
    """Behavioral Cloning model for learning from attract mode demonstrations."""
    def __init__(self, input_size=NumberOfParams):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fire_output = nn.Linear(64, 1)
        self.zap_output = nn.Linear(64, 1)
        self.spinner_output = nn.Linear(64, 1)
        # Xavier initialization for better convergence
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fire_output.weight)
        nn.init.xavier_uniform_(self.zap_output.weight)
        nn.init.xavier_uniform_(self.spinner_output.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        fire = torch.sigmoid(self.fire_output(x))
        zap = torch.sigmoid(self.zap_output(x))
        spinner = self.spinner_output(x)
        return fire, zap, spinner

### Utility Functions
def process_frame_data(data):
    """
    Process incoming frame data from the game.
    - Unpacks header and game state
    - Normalizes data to [-1, 1]
    """
    try:
        header_fmt = ">IdBBBIIBBBh"
        header_size = struct.calcsize(header_fmt)
        num_values, reward, game_action, game_mode, is_done, frame_counter, score, save_signal, fire, zap, spinner = struct.unpack(header_fmt, data[:header_size])
        game_data = data[header_size:]
        unpacked_data = np.frombuffer(game_data, dtype=np.uint16).astype(np.float32) - 32768.0
        normalized_data = unpacked_data / np.where(unpacked_data > 0, 32767.0, 32768.0)
        if len(normalized_data) != NumberOfParams:
            normalized_data = np.pad(normalized_data, (0, NumberOfParams - len(normalized_data)), 'constant')[:NumberOfParams]
        is_attract = (game_mode & 0x80) == 0
        return normalized_data, frame_counter, reward, (fire, zap, spinner), is_attract, is_done, save_signal
    except Exception as e:
        print(f"Error processing frame data: {e}")
        return None, 0, 0.0, None, False, False, 0

def train_bc(model, state, fire_target, zap_target, spinner_target):
    """Train the BC model on a single frame of demonstration data."""
    model = model.to(device)
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    fire_target = torch.tensor([[float(fire_target)]], dtype=torch.float32).to(device)
    zap_target = torch.tensor([[float(zap_target)]], dtype=torch.float32).to(device)
    spinner_target = torch.tensor([[float(spinner_target) / 128.0]], dtype=torch.float32).to(device)
    
    fire_pred, zap_pred, spinner_pred = model(state_tensor)
    loss = (nn.BCELoss()(fire_pred, fire_target) + 
            nn.BCELoss()(zap_pred, zap_target) + 
            nn.MSELoss()(spinner_pred, spinner_target))
    
    model.optimizer.zero_grad()
    loss.backward()
    model.optimizer.step()
    
    fire_action = 1 if fire_pred.item() > 0.5 else 0
    zap_action = 1 if zap_pred.item() > 0.5 else 0
    spinner_action = int(round(spinner_pred.item() * 128.0))
    return loss.item(), (fire_action, zap_action, spinner_action)

def normalize_reward(reward):
    """
    Normalize reward to [0, 1] range.
    - Clamps to [-32768, 32768]
    """
    clamped_reward = max(-32768.0, min(32767.0, reward))
    normalized_reward = (clamped_reward + 32767.0) / 65536.0
    return float(normalized_reward)

def safe_add_to_buffer(buffer, obs, next_obs, action, reward, done):
    """Safely add experience to the RL replay buffer with shape validation."""
    if buffer is None:
        return False
    try:
        obs = np.asarray(obs, dtype=np.float32).flatten()
        next_obs = np.asarray(next_obs, dtype=np.float32).flatten()
        action = np.asarray(action, dtype=np.float32).flatten()
        if obs.shape != (NumberOfParams,) or next_obs.shape != (NumberOfParams,) or action.shape != (3,):
            raise ValueError(f"Invalid shapes - obs: {obs.shape}, next_obs: {next_obs.shape}, action: {action.shape}")
        normalized_reward = normalize_reward(float(reward))
        done = bool(done)
        buffer.add(obs, next_obs, action, normalized_reward, done, [{}] if hasattr(buffer, 'handle_timeout_termination') else None)
        return True
    except Exception as e:
        print(f"Error adding to replay buffer: {e}")
        return False

def apply_minimal_compatibility_patches(model):
    """
    Apply compatibility patches for Stable Baselines3.
    - BUGBUG: Review necessity with current SB3 version; may be removable
    """
    try:
        if not hasattr(model, '_logger'):
            from stable_baselines3.common.logger import Logger
            model._logger = Logger(folder=None, output_formats=[])
        model.action_space.low = model.action_space.low.astype(np.float32)
        model.action_space.high = model.action_space.high.astype(np.float32)
        model.observation_space.low = model.observation_space.low.astype(np.float32)
        model.observation_space.high = model.observation_space.high.astype(np.float32)
    except Exception as e:
        print(f"Warning: Failed to apply compatibility patches: {e}")

### Replay Functionality
def replay_log_file(log_file_path, bc_model):
    """Replay a log file to train the BC model."""
    if not os.path.exists(log_file_path):
        print(f"Error: Log file {log_file_path} not found")
        return False
    try:
        print(f"Replaying log file: {log_file_path}")
        frame_count = 0
        with open(log_file_path, 'rb') as log_file:
            while True:
                header_size_bytes = log_file.read(4)
                if len(header_size_bytes) < 4:
                    break
                header_size = struct.unpack(">I", header_size_bytes)[0]
                payload_size_bytes = log_file.read(4)
                if len(payload_size_bytes) < 4:
                    break
                payload_size = struct.unpack(">I", payload_size_bytes)[0]
                payload_data = log_file.read(payload_size)
                if len(payload_data) < payload_size:
                    break
                processed_data, _, reward, game_action, is_attract, done, _ = process_frame_data(payload_data)
                if processed_data is not None and game_action:
                    train_bc(bc_model, processed_data, *game_action)
                    frame_count += 1
                    if frame_count % 1000 == 0:
                        print(f"Processed {frame_count} frames")
        print(f"Replay complete. Processed {frame_count} frames.")
        torch.save(bc_model.state_dict(), BC_MODEL_PATH)
        return True
    except Exception as e:
        print(f"Error replaying log file: {e}")
        return False

### Core Functions
def initialize_models():
    """Initialize environment and models, loading saved states if available."""
    env = TempestEnv()
    bc_model = BCModel().to(device)
    bc_model.optimizer = optim.Adam(bc_model.parameters(), lr=0.005)
    if os.path.exists(BC_MODEL_PATH):
        bc_model.load_state_dict(torch.load(BC_MODEL_PATH, map_location=device))
        print(f"Loaded BC model from {BC_MODEL_PATH}")

    policy_kwargs = {
        "features_extractor_class": TempestFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": 128},
        "net_arch": [128, 64]
    }
    rl_model = SAC(
        "MlpPolicy", env, policy_kwargs=policy_kwargs, learning_rate=0.0001,
        buffer_size=100000, learning_starts=1000, batch_size=64,
        train_freq=1, gradient_steps=10, device=device  # Increased gradient_steps
    )
    if os.path.exists(LATEST_MODEL_PATH):
        rl_model = SAC.load(LATEST_MODEL_PATH, env=env)
        print(f"Loaded RL model from {LATEST_MODEL_PATH}")
    apply_minimal_compatibility_patches(rl_model)
    return env, bc_model, rl_model

def save_models(rl_model, bc_model):
    """Save RL and BC models to disk."""
    try:
        rl_model.save(LATEST_MODEL_PATH)
        torch.save(bc_model.state_dict(), BC_MODEL_PATH)
        print(f"Models saved to {MODEL_DIR}")
    except Exception as e:
        print(f"Error saving models: {e}")

def train_model_background(model):
    """Background RL training with increased gradient steps for CPU efficiency."""
    if model and model.replay_buffer.pos > model.learning_starts:
        try:
            model.train(gradient_steps=10, batch_size=64)  # More work per call
        except Exception as e:
            print(f"Training error: {e}")

def main():
    """Main loop to handle game interaction and model training."""
    env, bc_model, rl_model = initialize_models()

    if args.replay:
        replay_log_file(args.replay, bc_model)
        if not os.path.exists(LUA_TO_PY_PIPE):
            return

    # Setup pipes
    for pipe in [LUA_TO_PY_PIPE, PY_TO_LUA_PIPE]:
        if os.path.exists(pipe):
            os.unlink(pipe)
        os.mkfifo(pipe)
        os.chmod(pipe, 0o666)

    print("Pipes created. Waiting for Lua...")
    while True:
        try:
            with os.fdopen(os.open(LUA_TO_PY_PIPE, os.O_RDONLY | os.O_NONBLOCK), "rb") as lua_to_py, \
                 open(PY_TO_LUA_PIPE, "wb") as py_to_lua:
                print("✔️ Lua connected.")
                frame_count = 0
                while True:
                    data = lua_to_py.read()
                    if not data:
                        time.sleep(0.01)
                        continue
                    processed_data, _, reward, game_action, is_attract, done, save_signal = process_frame_data(data)
                    if processed_data is None:
                        continue
                    env.update_state(processed_data, reward, game_action, done)
                    env.is_attract_mode = is_attract

                    if save_signal:
                        save_models(rl_model, bc_model)

                    if is_attract:
                        if game_action:
                            _, bc_action = train_bc(bc_model, processed_data, *game_action)
                            action = encode_action(*bc_action)
                        else:
                            action = encode_action(0, 0, 0)
                    else:
                        action, _ = rl_model.predict(processed_data, deterministic=False)
                        if env.prev_state is not None:
                            safe_add_to_buffer(rl_model.replay_buffer, env.prev_state, env.state, action, env.reward, env.done)
                        if frame_count % 50 == 0 and rl_model.replay_buffer.pos > rl_model.learning_starts:
                            threading.Thread(target=train_model_background, args=(rl_model,), daemon=True).start()

                    fire, zap, spinner = decode_action(action)
                    py_to_lua.write(struct.pack("bbb", fire, zap, spinner))
                    py_to_lua.flush()
                    frame_count += 1

        except KeyboardInterrupt:
            save_models(rl_model, bc_model)
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)

def encode_action(fire, zap, spinner_delta):
    """Convert discrete actions to continuous RL action space."""
    return np.array([float(fire), float(zap), max(-1.0, min(1.0, spinner_delta / 128.0))], dtype=np.float32)

def decode_action(action):
    """Convert RL actions to discrete game inputs."""
    return 1 if action[0] > 0.5 else 0, 1 if action[1] > 0.5 else 0, int(round(action[2] * 127.0))

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    main()