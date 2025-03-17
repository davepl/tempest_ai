#!/usr/bin/env python3
"""
Tempest AI Model with BC-to-RL Transition
Author: Dave Plummer (davepl) and various AI assists
Date: 2025-03-06 (Updated)

This script implements a hybrid AI model for Tempest:
1. Uses Behavioral Cloning (BC) during attract mode to learn from the game's AI
2. Switches to Reinforcement Learning (RL) during gameplay
"""

### Imports and Environment Setup
# Imports libraries for math, file I/O, RL, and neural networks; sets up PyTorch device and file paths.
# Essential for enabling the hybrid BC-RL approach and communication with the game via pipes.
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
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

# Pipe and model paths
LUA_TO_PY_PIPE = "/tmp/lua_to_py"
PY_TO_LUA_PIPE = "/tmp/py_to_lua"
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODEL_DIR, exist_ok=True)
LATEST_MODEL_PATH = os.path.join(MODEL_DIR, "tempest_model_latest.zip")
BC_MODEL_PATH = os.path.join(MODEL_DIR, "tempest_bc_model.pt")

# Parse command line arguments
parser = argparse.ArgumentParser(description='Tempest AI Model')
parser.add_argument('--replay', type=str, help='Path to a log file to replay for BC training')
args = parser.parse_args()

# Device setup
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

### Environment Definition
# Defines a custom Gym environment for Tempest with action/observation spaces.
# Necessary to interface the game with RL algorithms and handle state transitions.
class TempestEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            shape=(3,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=np.float32(-1.0),
            high=np.float32(1.0),
            shape=(247,),
            dtype=np.float32
        )
        self.state = np.zeros(247, dtype=np.float32)
        self.reward = 0.0
        self.done = False
        self.is_attract_mode = False
        self.prev_state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.zeros(247, dtype=np.float32)
        self.reward = 0.0
        self.done = False
        self.prev_state = None
        return self.state, {}

    def step(self, action):
        fire = 1 if action[0] > 0.5 else 0
        zap = 1 if action[1] > 0.5 else 0
        spinner_delta = int(round(action[2] * 64.0))
        spinner_delta = max(-64, min(64, spinner_delta))
        return self.state, self.reward, self.done, False, {"action_taken": (fire, zap, spinner_delta)}

    def update_state(self, game_state, reward, game_action=None, done=False):
        self.prev_state = self.state.copy()
        self.state = game_state
        self.reward = reward
        self.done = done
        return self.state

### Feature Extractor and Models
# Custom feature extractor and BC model for processing game states and cloning AI behavior.
# Supports RL feature extraction and BC learning from attract mode demonstrations.
class TempestFeaturesExtractor(BaseFeaturesExtractor):
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
    def __init__(self, input_size=247):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fire_output = nn.Linear(64, 1)
        self.zap_output = nn.Linear(64, 1)
        self.spinner_output = nn.Linear(64, 1)
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
# Helper functions for data processing, BC training, buffer management, and model compatibility.
# Ensures smooth data flow between Lua and Python, and robust model training.
def process_frame_data(data):
    if len(data) < 24:
        return None, 0, 0.0, None, False, False
    try:
        header_fmt = ">IdBBBIIBBBh"
        header_size = struct.calcsize(header_fmt)
        num_values, reward, game_action, game_mode, is_done, frame_counter, score, save_signal, fire, zap, spinner = struct.unpack(header_fmt, data[:header_size])
        game_data = data[header_size:]
        num_ints = len(game_data) // 2
        unpacked_data = [struct.unpack(">H", game_data[i*2:i*2+2])[0] - 32768 for i in range(num_ints)]
        normalized_data = np.array([float(x) / (32767.0 if x > 0 else 32768.0) for x in unpacked_data], dtype=np.float32)
        if len(normalized_data) != 247:
            normalized_data = np.pad(normalized_data, (0, 247 - len(normalized_data)), 'constant')[:247]
        is_attract = (game_mode & 0x80) == 0
        return normalized_data, frame_counter, reward, (fire, zap, spinner), is_attract, is_done, save_signal
    except Exception as e:
        print(f"Error processing frame data: {e}")
        return None, 0, 0.0, None, False, False

def train_bc(model, state, fire_target, zap_target, spinner_target):
    model = model.to(device)
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    fire_target = torch.tensor([[float(fire_target)]], dtype=torch.float32).to(device)
    zap_target = torch.tensor([[float(zap_target)]], dtype=torch.float32).to(device)
    spinner_target = torch.tensor([[float(spinner_target) / 128.0]], dtype=torch.float32).to(device)
    fire_pred, zap_pred, spinner_pred = model(state_tensor)
    loss = nn.BCELoss()(fire_pred, fire_target) + nn.BCELoss()(zap_pred, zap_target) + nn.MSELoss()(spinner_pred, spinner_target)
    model.optimizer.zero_grad()
    loss.backward()
    model.optimizer.step()
    fire_action = 1 if fire_pred.item() > 0.5 else 0
    zap_action = 1 if zap_pred.item() > 0.5 else 0
    spinner_action = int(round(spinner_pred.item() * 64.0))
    return loss.item(), (fire_action, zap_action, spinner_action)

def safe_add_to_buffer(buffer, obs, next_obs, action, reward, done):
    if buffer is None:
        return False
    try:
        obs = np.asarray(obs).flatten()
        next_obs = np.asarray(next_obs).flatten()
        action = np.asarray(action).flatten()
        if obs.shape != (247,) or next_obs.shape != (247,) or action.shape != (3,):
            raise ValueError(f"Invalid shapes - obs: {obs.shape}, next_obs: {next_obs.shape}, action: {action.shape}")
        reward = float(reward)
        done = float(done)
        buffer.add(obs, next_obs, action, reward, done, [{}]) if hasattr(buffer, 'handle_timeout_termination') else buffer.add(obs, next_obs, action, reward, done)
        return True
    except Exception as e:
        print(f"Error adding to replay buffer: {e}")
        return False

def apply_minimal_compatibility_patches(model):
    try:
        if not hasattr(model, '_logger'):
            from stable_baselines3.common.logger import Logger
            model._logger = Logger(folder=None, output_formats=[])
        model.action_space.low = model.action_space.low.astype(np.float32)
        model.action_space.high = model.action_space.high.astype(np.float32)
        model.observation_space.low = model.observation_space.low.astype(np.float32)
        model.observation_space.high = model.observation_space.high.astype(np.float32)
        if hasattr(model, 'critic') and hasattr(model.critic, 'forward'):
            original_forward = model.critic.forward
            def safe_forward(obs, actions):
                try:
                    return original_forward(obs, actions)
                except IndexError as e:
                    if "too many indices" in str(e):
                        return original_forward(obs.reshape(obs.shape[0], -1), actions.reshape(actions.shape[0], -1))
                    raise
            model.critic.forward = safe_forward
    except Exception as e:
        print(f"Warning: Failed to apply compatibility patches: {e}")

### Log File Replay Function
def replay_log_file(log_file_path, bc_model):
    """
    Replay a log file to train the BC model
    Format: each record has:
    - 4-byte header size (uint32)
    - 4-byte payload size (uint32)
    - Raw payload data
    """
    if not os.path.exists(log_file_path):
        print(f"Error: Log file {log_file_path} not found")
        return False
    
    try:
        print(f"Replaying log file: {log_file_path}")
        frame_count = 0
        bc_loss_sum = 0.0
        
        with open(log_file_path, 'rb') as log_file:
            while True:
                # Read header size (uint32)
                header_size_bytes = log_file.read(4)
                if not header_size_bytes or len(header_size_bytes) < 4:
                    break  # End of file
                
                header_size = struct.unpack(">I", header_size_bytes)[0]
                
                # Read payload size (uint32)
                payload_size_bytes = log_file.read(4)
                if not payload_size_bytes or len(payload_size_bytes) < 4:
                    break  # Incomplete record
                
                payload_size = struct.unpack(">I", payload_size_bytes)[0]
                
                # Read payload data
                payload_data = log_file.read(payload_size)
                if not payload_data or len(payload_data) < payload_size:
                    break  # Incomplete record
                
                # Process the frame data
                processed_data, _, reward, game_action, is_attract, done, save_signal = process_frame_data(payload_data)
                
                if processed_data is not None and game_action:
                    # Train the BC model
                    loss, _ = train_bc(bc_model, processed_data, *game_action)
                    bc_loss_sum += loss
                    frame_count += 1
                    
                    # Print progress every 1,000 frames
                    if frame_count % 1000 == 0:
                        print(f"Processed {frame_count} frames, average BC loss: {bc_loss_sum / 1000:.6f}")
                        bc_loss_sum = 0.0
        
        print(f"Replay complete. Processed {frame_count} frames from log file.")
        if frame_count > 0 and bc_loss_sum > 0:
            print(f"Final average BC loss: {bc_loss_sum / (frame_count % 10000 or 1):.6f}")
        
        # Save the trained BC model
        torch.save(bc_model.state_dict(), BC_MODEL_PATH)
        print(f"BC model saved to {BC_MODEL_PATH}")
        return True
    
    except Exception as e:
        print(f"Error replaying log file: {e}")
        return False

### Initialization and Main Loop
# Sets up models, runs the main training/inference loop, and handles I/O with the game.
# Orchestrates the hybrid BC-RL logic and ensures persistent model state.
def initialize_models():
    env = TempestEnv()
    bc_model = BCModel().to(device)
    bc_model.optimizer = optim.Adam(bc_model.parameters(), lr=0.001)
    if os.path.exists(BC_MODEL_PATH):
        try:
            bc_model.load_state_dict(torch.load(BC_MODEL_PATH, map_location=device))
            print(f"Loaded BC model from {BC_MODEL_PATH}")
        except Exception as e:
            print(f"Error loading BC model: {e}")
    policy_kwargs = {"features_extractor_class": TempestFeaturesExtractor, "features_extractor_kwargs": {"features_dim": 128}, "net_arch": [128, 64]}
    rl_model = SAC("MlpPolicy", env, policy_kwargs=policy_kwargs, learning_rate=0.0001, buffer_size=100000, learning_starts=1000, batch_size=64, device=device)
    if os.path.exists(LATEST_MODEL_PATH):
        try:
            rl_model = SAC.load(LATEST_MODEL_PATH, env=env)
            print(f"Loaded RL model from {LATEST_MODEL_PATH}")
        except Exception as e:
            print(f"Error loading RL model: {e}")
    apply_minimal_compatibility_patches(rl_model)
    return env, bc_model, rl_model

def save_models(rl_model, bc_model):
    try:
        rl_model.save(LATEST_MODEL_PATH)
        torch.save(bc_model.state_dict(), BC_MODEL_PATH)
        print(f"Models saved to {MODEL_DIR}")
    except Exception as e:
        print(f"Error saving models: {e}")

def train_model_background(model):
    if model and model.replay_buffer.pos > model.learning_starts:
        try:
            model.train(gradient_steps=5, batch_size=64)
        except Exception as e:
            print(f"Training error: {e}")

def main():
    print("Starting Tempest AI...")
    env, bc_model, rl_model = initialize_models()
    
    # args.replay = "/Users/dave/mame/tempest.log"
    # If a replay file is specified, replay it first
    if args.replay:
        replay_log_file(args.replay, bc_model)
        # Exit if we're only replaying
        if not os.path.exists(LUA_TO_PY_PIPE):
            print("Replay complete. Exiting.")
            return
    
    for pipe in [LUA_TO_PY_PIPE, PY_TO_LUA_PIPE]:
        if os.path.exists(pipe):
            os.unlink(pipe)
        os.mkfifo(pipe)
        os.chmod(pipe, 0o666)
    print("Pipes created. Waiting for Lua...")

    while True:
        try:
            with os.fdopen(os.open(LUA_TO_PY_PIPE, os.O_RDONLY | os.O_NONBLOCK), "rb") as lua_to_py, open(PY_TO_LUA_PIPE, "wb") as py_to_lua:
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
                            if random.random() < 0.2:
                                spinner = max(-64, min(63, bc_action[2] + random.randint(-10, 10)))
                                bc_action = (bc_action[0], bc_action[1], spinner)
                            action = encode_action(*bc_action)
                        else:
                            action = encode_action(0, 0, 0)
                    else:
                        try:
                            action, _ = rl_model.predict(processed_data, deterministic=False)
                            action = action.flatten()
                        except Exception as e:
                            print(f"Prediction error: {e}")
                            fire, zap, spinner = bc_model(torch.FloatTensor(processed_data).unsqueeze(0).to(device))
                            action = encode_action(1 if fire.item() > 0.5 else 0, 1 if zap.item() > 0.5 else 0, int(round(spinner.item() * 64.0)))

                        if random.random() < 0.2:
                            fire, zap, spinner = bc_model(torch.FloatTensor(processed_data).unsqueeze(0).to(device))
                            action = encode_action(1 if fire.item() > 0.5 else 0, 1 if zap.item() > 0.5 else 0, int(round(spinner.item() * 64.0)))

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

    print("Shutting down...")

def encode_action(fire, zap, spinner_delta):
    return np.array([float(fire), float(zap), max(-1.0, min(1.0, spinner_delta / 64.0))], dtype=np.float32)

def decode_action(action):
    return 1 if action[0] > 0.5 else 0, 1 if action[1] > 0.5 else 0, int(round(action[2] * 64.0))

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    main()