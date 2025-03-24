#!/usr/bin/env python3
"""
Tempest AI Model: Combines Behavioral Cloning (BC) in attract mode with Reinforcement Learning (RL) in gameplay.
- BC learns from game AI demonstrations; RL (SAC) optimizes during player control with robust training architecture.
- Communicates with Tempest via Lua pipes; saves/loads models for persistence.
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import time
import struct
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3 import SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.logger import Logger
import gymnasium as gym
from gymnasium import spaces
import threading
import queue
from collections import deque
import concurrent.futures
from threading import Lock
import traceback
import select

# Constants
ShouldReplayLog = True
LogFile = "/Users/dave/mame/500k.log"
MaxLogFrames = 10000000

NumberOfParams = 247
LUA_TO_PY_PIPE = "/tmp/lua_to_py"
PY_TO_LUA_PIPE = "/tmp/py_to_lua"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
LATEST_MODEL_PATH = f"{MODEL_DIR}/tempest_model_latest.zip"
BC_MODEL_PATH = f"{MODEL_DIR}/tempest_bc_model.pt"

# Parse command line arguments
parser = argparse.ArgumentParser(description='Tempest AI Model')
parser.add_argument('--replay', type=str, help='Path to a log file to replay for BC training')
args = parser.parse_args()

# Device selection: CUDA > MPS > CPU
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device.type.upper()}")

# Training metrics buffers and synchronization
actor_losses = deque(maxlen=100)
critic_losses = deque(maxlen=100)
ent_coefs = deque(maxlen=100)
mean_rewards = deque(maxlen=100)
training_queue = queue.Queue()  # For RL training
bc_training_queue = queue.Queue(maxsize=1000)  # For BC training
bc_model_lock = threading.Lock()  # Lock for BC model thread safety

class TempestEnv(gym.Env):
    """Custom Gym environment interfacing with Tempest game via Lua pipes."""
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(low=np.array([0.0, 0.0, -1.0], dtype=np.float32), 
                                       high=np.array([1.0, 1.0, 1.0], dtype=np.float32), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(NumberOfParams,), dtype=np.float32)
        self.state = np.zeros(NumberOfParams, dtype=np.float32)
        self.reward = np.float32(0.0)
        self.done = False
        self.is_attract_mode = False
        self.prev_state = None
        self.prev_action = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.zeros(NumberOfParams, dtype=np.float32)
        self.reward = np.float32(0.0)
        self.done = False
        self.is_attract_mode = False
        self.prev_state = None
        self.prev_action = None
        return self.state, {}

    def step(self, action):
        self.prev_action = np.array(action, dtype=np.float32)
        fire, zap, spinner = self.decode_action(action)
        return self.state, self.reward, self.done, False, {"action_taken": (fire, zap, spinner)}

    def update_state(self, game_state, reward, game_action=None, done=False):
        self.prev_state = self.state.copy()
        self.state = game_state.astype(np.float32)
        self.reward = np.float32(reward)
        self.done = bool(done)
        return self.state

    def decode_action(self, action):
        fire = 1 if action[0] > 0.5 else 0
        zap = 1 if action[1] > 0.5 else 0
        spinner = int(round(action[2] * 127.0))
        return fire, zap, max(-127, min(127, spinner))

class TempestFeaturesExtractor(BaseFeaturesExtractor):
    """Feature extractor for SAC, reducing observation dimensionality."""
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        self.feature_extractor = nn.Sequential(
            nn.Linear(observation_space.shape[0], 256), nn.ReLU(),
            nn.Linear(256, features_dim), nn.ReLU()
        ).to(device)

    def forward(self, observations):
        if observations.device != device:
            observations = observations.to(device)
        return self.feature_extractor(observations)

class BCModel(nn.Module):
    """Simplified BC model optimized for Tempest gameplay."""
    def __init__(self, input_size=NumberOfParams):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU()
        )
        self.fire_output = nn.Linear(128, 1)
        self.zap_output = nn.Linear(128, 1)
        self.spinner_output = nn.Linear(128, 1)
        
        # Simplified initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        self.optimizer = optim.Adam(self.parameters(), lr=0.003)
        self.to(device)

    def forward(self, x):
        if x.device != device:
            x = x.to(device)
        features = self.feature_extractor(x)
        fire_out = torch.sigmoid(self.fire_output(features))
        zap_out = torch.sigmoid(self.zap_output(features))
        spinner_out = torch.tanh(self.spinner_output(features))
        return torch.cat([fire_out, zap_out, spinner_out], dim=1)

def process_frame_data(data, header_data=None):
    """Process frame data from Lua using exact same format string as Lua."""
    if not data:
        return None
    
    # Fixed format string - MUST MATCH EXACTLY with Lua's string.pack
    format_string = ">IdBBBIIBBBhB"
    
    try:
        # Calculate the size of the header
        header_size = struct.calcsize(format_string)
        
        # Debug: Print the length of the data received
        # print(f"Received data length: {len(data)}")
        
        # Make sure we have enough data
        if len(data) < header_size:
            print(f"Data too short: {len(data)} < {header_size}")
            return None
        
        # Unpack using the same format string as Lua packs with
        # Slice the data to ensure only the header is unpacked
        values = struct.unpack(format_string, data[:header_size])
        
        # Extract the values - same order as in Lua
        num_values, reward, game_action, game_mode, done, frame_counter, score, save_signal, fire, zap, spinner, is_attract = values
        
        # print(f"UNPACKED: values={num_values}, reward={reward}, action={game_action}, mode={game_mode}, done={done}, frame={frame_counter}")
        # print(f"  score={score}, save={save_signal}, fire={fire}, zap={zap}, spinner={spinner}, attract={is_attract}")
        
        # Get the game state data (all remaining data)
        game_data_bytes = data[header_size:]
        
        # Process game state as 16-bit values
        state_values = []
        for i in range(0, len(game_data_bytes), 2):
            if i + 1 < len(game_data_bytes):
                # Unpack 16-bit big-endian values
                value = struct.unpack(">H", game_data_bytes[i:i+2])[0]
                # Normalize: 0-65535 -> -32768 to 32767
                normalized = value - 32768
                state_values.append(normalized)
        
        # Create numpy array
        state = np.array(state_values, dtype=np.float32)
        
        # Normalize to -1.0 to 1.0 range
        state = state / 32768.0
        
        # Create game action tuple for convenience
        game_action_tuple = (bool(fire), bool(zap), spinner)
        
        # Return all values
        return state, reward, game_action_tuple, game_mode, bool(done), bool(is_attract), save_signal
        
    except Exception as e:
        print(f"ERROR unpacking data: {e}")
        traceback.print_exc()
        return None

def train_bc(model, state, fire_target, zap_target, spinner_target):
    """Train the BC model with explicit device verification"""
    global bc_optimizer
    
    # Initialize optimizer if it doesn't exist
    if bc_optimizer is None:
        bc_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Move input tensor to MPS and verify
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Move target tensor to MPS and verify
    targets = torch.tensor([[fire_target, zap_target, spinner_target]], dtype=torch.float32).to(device)
    
    # Forward pass
    predictions = model(state_tensor)
    
    # Loss calculation
    loss = nn.MSELoss()(predictions, targets)
    
    # Backward pass and optimization
    if bc_optimizer:
        bc_optimizer.zero_grad()
        loss.backward()
        bc_optimizer.step()
    
    # Move result back to CPU for numpy conversion
    return loss.item(), predictions.detach().cpu().numpy()[0]  # Added .cpu() here

def safe_add_to_buffer(buffer, obs, next_obs, action, reward, done):
    """Add experience to RL replay buffer with validation."""
    try:
        obs = np.asarray(obs, dtype=np.float32).flatten()
        next_obs = np.asarray(next_obs, dtype=np.float32).flatten()
        action = np.asarray(action, dtype=np.float32).flatten()
        
        if obs.shape != (NumberOfParams,) or next_obs.shape != (NumberOfParams,) or action.shape != (3,):
            obs = np.resize(obs, NumberOfParams)
            next_obs = np.resize(next_obs, NumberOfParams)
            action = np.resize(action, 3)
        
        reward_scalar = float(reward)
        done_scalar = bool(done)
        
        if hasattr(buffer, 'device') and hasattr(buffer, '_store_transition'):
            obs_tensor = torch.as_tensor(obs, device=buffer.device)
            next_obs_tensor = torch.as_tensor(next_obs, device=buffer.device)
            action_tensor = torch.as_tensor(action, device=buffer.device)
            reward_tensor = torch.as_tensor(reward_scalar, device=buffer.device)
            done_tensor = torch.as_tensor(done_scalar, device=buffer.device)
            buffer._store_transition(obs_tensor, next_obs_tensor, action_tensor, 
                                    reward_tensor, done_tensor, [{}])
        else:
            buffer.add(obs, next_obs, action, reward_scalar, done_scalar,
                       [{}] if hasattr(buffer, 'handle_timeout_termination') else None)
        
        return True
    except Exception as e:
        print(f"Error adding to buffer: {e}")
        return False

def apply_minimal_compatibility_patches(model):
    """Ensure SB3 model compatibility and device consistency."""
    if not hasattr(model, '_logger'):
        model._logger = Logger(folder=None, output_formats=[])
    
    model.policy.to(device)
    model.actor.to(device)
    model.critic.to(device)
    if hasattr(model, 'critic_target'):
        model.critic_target.to(device)
    
    if hasattr(model.replay_buffer, 'device'):
        model.replay_buffer.device = device
    
    if hasattr(model.replay_buffer, '_tensor_names'):
        for tensor_name in model.replay_buffer._tensor_names:
            tensor = getattr(model.replay_buffer, tensor_name)
            if isinstance(tensor, torch.Tensor) and tensor.device != device:
                setattr(model.replay_buffer, tensor_name, tensor.to(device))
    
    if hasattr(model, 'log_ent_coef') and isinstance(model.log_ent_coef, torch.nn.Parameter):
        model.log_ent_coef.data = model.log_ent_coef.data.to(device)

def replay_log_file(log_file_path, bc_model):
    """Train BC model by replaying demonstration log file."""
    if not os.path.exists(log_file_path):
        print(f"Error: Log file {log_file_path} not found")
        return False
    
    print(f"Replaying {log_file_path}")
    frame_count = 0
    batch_size = 64
    training_data = []
    batch_loss = 0
    
    total_fire = 0
    total_zap = 0
    total_reward = 0
    total_spinner = 0
    
    with open(log_file_path, 'rb') as f:
        frames_processed = 0
        while frame_count < MaxLogFrames:
            header_size_bytes = f.read(4)
            if not header_size_bytes or len(header_size_bytes) < 4:
                print(f"End of file reached after {frames_processed} frames")
                break
                
            header_size = struct.unpack(">I", header_size_bytes)[0]
            if header_size != 16:
                print(f"Warning: Expected header size of 15 bytes, got {header_size}")
                break

            payload_size_bytes = f.read(4)
            if not payload_size_bytes or len(payload_size_bytes) < 4:
                break
                
            payload_size = struct.unpack(">I", payload_size_bytes)[0]
            header_data = f.read(8)
            if len(header_data) < 8:
                break
                
            reward_int = struct.unpack(">i", header_data[0:4])[0]
            reward = reward_int / 1000.0
            done = struct.unpack(">B", bytes([header_data[4]]))[0]
            zap = struct.unpack(">B", bytes([header_data[5]]))[0]
            fire = struct.unpack(">B", bytes([header_data[6]]))[0]
            spinner = struct.unpack(">b", bytes([header_data[7]]))[0]
            
            game_data_bytes = f.read(payload_size)
            if len(game_data_bytes) < payload_size:
                break
                
            game_data = np.frombuffer(game_data_bytes, dtype=np.uint16)
            game_data = game_data.astype(np.float32) - 32768.0
            state = game_data / np.where(game_data > 0, 32767.0, 32768.0).astype(np.float32)
            
            if len(state) > NumberOfParams:
                state = state[:NumberOfParams]
            elif len(state) < NumberOfParams:
                state = np.pad(state, (0, NumberOfParams - len(state)), 'constant')
                
            frames_processed += 1
            total_fire += fire
            total_zap += zap
            total_reward += reward
            total_spinner += spinner
            
            action = (fire, zap, spinner)
            training_data.append((state, action, reward))
            
            if len(training_data) >= batch_size:
                batch_loss = train_model_with_batch(bc_model, training_data[:batch_size])
                if frame_count % 100 == 0:
                    print(f"Frames: {frame_count} - Trained batch - loss: {batch_loss:.6f}")
                training_data = training_data[batch_size:]
                frame_count += batch_size
        
        if training_data:
            batch_loss = train_model_with_batch(bc_model, training_data)
            frame_count += len(training_data)
        
        print(f"Replay complete: {frame_count} frames processed")
        if frames_processed > 0:
            print(f"FINAL STATS - Avg Fire: {total_fire/frames_processed:.4f}, "
                  f"Avg Zap: {total_zap/frames_processed:.4f}, Avg Reward: {total_reward/frames_processed:.4f}")
        
        torch.save(bc_model.state_dict(), BC_MODEL_PATH)
        return True

def train_model_with_batch(model, batch):
    """Train model with a batch of data and reward information."""
    with bc_model_lock:
        model.train()
        states = []
        fire_targets = []
        zap_targets = []
        spinner_targets = []
        rewards = []
        
        for state, game_action, reward in batch:
            fire, zap, spinner = game_action
            normalized_spinner = max(-127, min(127, spinner)) / 127.0
            states.append(state)
            fire_targets.append(float(fire))
            zap_targets.append(float(zap))
            spinner_targets.append(normalized_spinner)
            rewards.append(float(reward))
        
        state_tensor = torch.FloatTensor(np.array(states)).to(device)
        targets = torch.FloatTensor([[f, z, s] for f, z, s in zip(fire_targets, zap_targets, spinner_targets)]).to(device)
        reward_tensor = torch.FloatTensor(rewards).to(device)
        
        model.optimizer.zero_grad()
        preds = model(state_tensor)
        
        reward_weights = torch.log1p(reward_tensor - reward_tensor.min() + 1e-6)
        reward_weights = reward_weights / reward_weights.mean()
        reward_weights = reward_weights.unsqueeze(1)
        
        fire_zap_loss = F.binary_cross_entropy(preds[:, :2], targets[:, :2], reduction='none') * reward_weights
        fire_zap_loss = fire_zap_loss.mean()
        spinner_loss = F.mse_loss(preds[:, 2:], targets[:, 2:], reduction='none') * reward_weights * 1.0
        spinner_loss = spinner_loss.mean()
        
        loss = fire_zap_loss + spinner_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        model.optimizer.step()
  
    return loss.item()

def background_bc_train(bc_model):
    """Background BC training thread."""
    while True:
        try:
            batch = []
            while len(batch) < 256 and not bc_training_queue.empty():
                batch.append(bc_training_queue.get())
            if batch:
                train_model_with_batch(bc_model, batch)
            time.sleep(0.01)
        except Exception as e:
            print(f"BC training error: {e}")

def background_rl_train(rl_model):
    """Background RL training thread."""
    while True:
        try:
            training_queue.get(timeout=1.0)
            if rl_model.replay_buffer.pos > rl_model.learning_starts:
                apply_minimal_compatibility_patches(rl_model)
                buffer_size = rl_model.replay_buffer.pos
                batch_size = min(64, max(8, buffer_size // 2))
                rl_model.train(gradient_steps=5, batch_size=batch_size)
                if hasattr(rl_model, 'logger'):
                    log_vals = rl_model.logger.name_to_value
                    actor_losses.append(log_vals.get('train/actor_loss', float('nan')))
                    critic_losses.append(log_vals.get('train/critic_loss', float('nan')))
                    ent_coefs.append(log_vals.get('train/ent_coef', float('nan')))
            training_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"RL training error: {e}")
            training_queue.task_done()

def main():
    """Main execution loop for Tempest AI."""
    env = TempestEnv()
    bc_model, rl_model = initialize_models(env)
    
    # Start background training threads
    threading.Thread(target=background_rl_train, args=(rl_model,), daemon=True).start()
    threading.Thread(target=background_bc_train, args=(bc_model,), daemon=True).start()

    if ShouldReplayLog:
        args.replay = LogFile

    if args.replay:
        replay_log_file(args.replay, bc_model)
        if not os.path.exists(LUA_TO_PY_PIPE):
            return

    for pipe in [LUA_TO_PY_PIPE, PY_TO_LUA_PIPE]:
        if os.path.exists(pipe): os.unlink(pipe)
        os.mkfifo(pipe)
        os.chmod(pipe, 0o666)

    print("Pipes created. Waiting for Lua...")
    frame_count = 0
    episode_rewards = deque(maxlen=5)
    total_episode_reward = 0
    last_training_request = time.time()
    training_interval = 0.1
    done_latch = False
    game_action = (0, 0, 0)
    game_mode = 0
    
    initial_exploration_ratio = 0.80
    min_exploration_ratio = 0.05
    exploration_decay = 0.9999
    current_exploration_ratio = initial_exploration_ratio
    
    with os.fdopen(os.open(LUA_TO_PY_PIPE, os.O_RDONLY | os.O_NONBLOCK), "rb") as lua_to_py, \
         open(PY_TO_LUA_PIPE, "wb") as py_to_lua:
        print("✔️ Lua connected.")
        
        while True:
            try:
                data = lua_to_py.read()
                if not data:
                    time.sleep(0.001)
                    continue
                state, reward, game_action, game_mode, done, is_attract, save_signal = process_frame_data(data)
                if state is None:
                    continue

                if done and not done_latch:
                    done_latch = True
                    env.update_state(state, reward, game_action, True)
                    total_episode_reward += reward
                    episode_rewards.append(total_episode_reward)
                    if not is_attract and env.prev_state is not None and env.prev_action is not None:
                        threading.Thread(target=safe_add_to_buffer,
                                        args=(rl_model.replay_buffer, env.prev_state, env.state, 
                                              env.prev_action, reward, True), daemon=True).start()
                    env.reset()
                    total_episode_reward = 0
                    frame_count = 0
                    
                elif done and done_latch:
                    py_to_lua.write(struct.pack("bbb", 0, 0, 0))
                    py_to_lua.flush()
                    continue
                    
                elif not done and done_latch:
                    done_latch = False
                    env.update_state(state, reward, game_action, False)
                    total_episode_reward += reward
                    
                else:
                    env.update_state(state, reward, game_action, False)
                    total_episode_reward += reward
                    if not is_attract and env.prev_state is not None and env.prev_action is not None:
                        threading.Thread(target=safe_add_to_buffer,
                                        args=(rl_model.replay_buffer, env.prev_state, env.state, 
                                              env.prev_action, reward, False), daemon=True).start()

                if is_attract and game_action:
                    try:
                        bc_training_queue.put((state, game_action, reward), block=False)
                        action = encode_action(1, 1, -1)  # Default action during attract mode
                    except queue.Full:
                        print("BC training queue is full, skipping attract mode action")
                        action = encode_action(0, 0, 0)
                else:
                    if frame_count % 1000 == 0 and current_exploration_ratio > min_exploration_ratio:
                        current_exploration_ratio *= exploration_decay
                        print(f"New Exploration ratio: {current_exploration_ratio}")
                    
                    if random.random() < current_exploration_ratio:
                        with bc_model_lock:
                            bc_model.eval()
                            with torch.no_grad():
                                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                                bc_action = bc_model(state_tensor)
                                bc_action = bc_action.cpu().numpy()[0]
                            action = bc_action
                    else:
                        with torch.no_grad():
                            # Create state tensor on CPU, not on MPS
                            state_tensor = torch.tensor(state.reshape(1, -1), dtype=torch.float32)  # No .to(device)
                            # Or alternatively, use numpy directly:
                            # state_tensor = state.reshape(1, -1)  # Keep as numpy array
                            
                            action, _ = rl_model.predict(state_tensor, deterministic=True)
                            action = action.flatten()

                env.step(action)
                fire, zap, spinner = env.decode_action(action)
                py_to_lua.write(struct.pack("bbb", fire, zap, spinner))
                py_to_lua.flush()

                current_time = time.time()
                if (current_time - last_training_request > training_interval and 
                    rl_model.replay_buffer.pos > rl_model.learning_starts and
                    not training_queue.full()):
                    training_queue.put(True, block=False)
                    mean_rewards.append(reward)
                    last_training_request = current_time

                if frame_count % 100 == 0:
                    actor_loss_mean = np.nanmean(actor_losses) if actor_losses else np.nan
                    critic_loss_mean = np.nanmean(critic_losses) if critic_losses else np.nan
                    ent_coef_mean = np.nanmean(ent_coefs) if ent_coefs else np.nan
                    reward_mean = np.nanmean(mean_rewards) if mean_rewards else np.nan
                    if not is_attract:
                        print(f"Frame {frame_count}, Reward: {reward:.2f}, Done: {done}, "
                              f"Buffer Size: {rl_model.replay_buffer.size()}")
                        print(f"Metrics - Actor Loss: {actor_loss_mean:.3f}, Critic Loss: {critic_loss_mean:.3f}, "
                              f"Entropy Coef: {ent_coef_mean:.3f}, Mean Reward: {reward_mean:.3f}, "
                              f"Expl: {current_exploration_ratio:.3f}")

                if save_signal:
                    threading.Thread(target=save_models, args=(rl_model, bc_model), daemon=True).start()
                    current_exploration_ratio = min(0.1, current_exploration_ratio * 2.0)
                
                frame_count += 1

            except KeyboardInterrupt:
                save_models(rl_model, bc_model)
                break
            except Exception as e:
                print(f"Error: {e}")
                traceback.print_exc()
                time.sleep(5)

def initialize_models(env):
    """Initialize models with improved SAC parameters."""
    bc_model = BCModel()
    if os.path.exists(BC_MODEL_PATH):
        bc_model.load_state_dict(torch.load(BC_MODEL_PATH, map_location=device))
        print(f"Loaded BC model from {BC_MODEL_PATH}")

    rl_model = SAC("MlpPolicy", env, policy_kwargs={
        "features_extractor_class": TempestFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": 128},
        "net_arch": dict(pi=[256, 128], qf=[256, 128])
    }, learning_rate=0.0005, buffer_size=100000, learning_starts=1000, batch_size=32,
       train_freq=(10, "step"), gradient_steps=5, ent_coef="auto", target_entropy=-1.5,
       tau=0.005, gamma=0.99, device=device, verbose=1)
    
    if os.path.exists(LATEST_MODEL_PATH):
        rl_model = SAC.load(LATEST_MODEL_PATH, env=env, device=device)
        print(f"Loaded RL model from {LATEST_MODEL_PATH}")
    
    apply_minimal_compatibility_patches(rl_model)
    return bc_model, rl_model

def save_models(rl_model, bc_model):
    """Save RL and BC models to disk."""
    try:
        with bc_model_lock:
            torch.save(bc_model.state_dict(), BC_MODEL_PATH)
        rl_model.save(LATEST_MODEL_PATH)
        print(f"Models saved to {MODEL_DIR}")
    except Exception as e:
        print(f"Error saving models: {e}")

def encode_action(fire, zap, spinner_delta):
    """Encode actions for RL model consistency."""
    fire_val = float(fire)
    zap_val = float(zap)
    if abs(spinner_delta) < 10:
        normalized_spinner = spinner_delta / 127.0
    else:
        sign = np.sign(spinner_delta)
        magnitude = abs(spinner_delta)
        squashed = 10.0 + (magnitude - 10.0) * (1.0 / (1.0 + (magnitude - 10.0) / 20.0))
        normalized_spinner = sign * min(squashed / 127.0, 1.0)
    result = torch.tensor([fire_val, zap_val, normalized_spinner], device=device)
    return result.cpu().numpy()

def tensor_to_numpy(tensor):
    """Safely convert a tensor to numpy, handling device placement."""
    if isinstance(tensor, torch.Tensor):
        if tensor.device.type != 'cpu':
            tensor = tensor.cpu()
        return tensor.numpy()
    return tensor  # Already a numpy array or other type

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    main()