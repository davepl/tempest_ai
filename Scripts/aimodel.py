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
import traceback
import select

# Constants
ShouldReplayLog = False
LogFile = "/Users/dave/mame/1m.log"
MaxLogFrames = 100000

NumberOfParams = 112
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
training_queue = queue.Queue()
bc_training_queue = queue.Queue(maxsize=1000)
bc_model_lock = threading.Lock()
replay_buffer_batch = []  # Batch for replay buffer additions
replay_buffer_lock = threading.Lock()

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
        spinner = int(round(np.clip(action[2], -1.0, 1.0) * 127.0))  # Ensure [-127, 127]
        return fire, zap, spinner

class TempestFeaturesExtractor(BaseFeaturesExtractor):
    """Feature extractor for SAC, reducing observation dimensionality."""
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        self.feature_extractor = nn.Sequential(
            nn.Linear(observation_space.shape[0], 256), nn.ReLU(),
            nn.Linear(256, features_dim), nn.ReLU()
        ).to(device)

    def forward(self, observations):
        if observations.dim() == 1:
            observations = observations.unsqueeze(0)
        if observations.device != device:
            observations = observations.to(device)
        return self.feature_extractor(observations)

class BCModel(nn.Module):
    """Enhanced BC model with clipped linear spinner output."""
    def __init__(self, input_size=NumberOfParams):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )
        self.fire_output = nn.Linear(128, 1)
        self.zap_output = nn.Linear(128, 1)
        self.spinner_output = nn.Linear(128, 1)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)
        self.to(device)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.device != device:
            x = x.to(device)
        features = self.feature_extractor(x)
        fire_out = torch.sigmoid(self.fire_output(features))
        zap_out = torch.sigmoid(self.zap_output(features))
        spinner_out_raw = self.spinner_output(features)
        spinner_out = torch.clamp(spinner_out_raw, -1.0, 1.0)
        return torch.cat([fire_out, zap_out, spinner_out], dim=1), spinner_out_raw

# Custom Actor-Critic RL implementation
class Actor(nn.Module):
    def __init__(self, state_dim=NumberOfParams):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fire_head = nn.Linear(256, 1)
        self.zap_head = nn.Linear(256, 1)
        self.spinner_head = nn.Linear(256, 1)
        self.to(device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        fire = torch.sigmoid(self.fire_head(x))
        zap = torch.sigmoid(self.zap_head(x))
        spinner = torch.tanh(self.spinner_head(x))
        return torch.cat([fire, zap, spinner], dim=-1)

actor = Actor()
actor_optimizer = optim.Adam(actor.parameters(), lr=1e-4)

# Replay Buffer
replay_buffer = deque(maxlen=100000)        

def process_frame_data(data, header_data=None):
    """Process frame data from Lua using exact same format string as Lua."""
    if not data:
        return None
    
    format_string = ">IdBBBIIBBBhB"
    
    try:
        header_size = struct.calcsize(format_string)
        
        if len(data) < header_size:
            print(f"Data too short: {len(data)} < {header_size}")
            return None
        
        values = struct.unpack(format_string, data[:header_size])
        
        num_values, reward, game_action, game_mode, done, frame_counter, score, save_signal, fire, zap, spinner, is_attract = values
        
        game_data_bytes = data[header_size:]
        
        state_values = []
        for i in range(0, len(game_data_bytes), 2):
            if i + 1 < len(game_data_bytes):
                value = struct.unpack(">H", game_data_bytes[i:i+2])[0]
                normalized = value - 32768
                state_values.append(normalized)
        
        state = np.array(state_values, dtype=np.float32)
        state = state / 32768.0
        
        if len(state) > NumberOfParams:
            state = state[:NumberOfParams]
        elif len(state) < NumberOfParams:
            state = np.pad(state, (0, NumberOfParams - len(state)), 'constant')
        
        game_action_tuple = (bool(fire), bool(zap), spinner)
        
        return state, reward, game_action_tuple, game_mode, bool(done), bool(is_attract), save_signal
        
    except Exception as e:
        print(f"ERROR unpacking data: {e}")
        traceback.print_exc()
        return None

def train_bc(model, state, fire_target, zap_target, spinner_target):
    """Train the BC model with explicit device verification."""
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    normalized_spinner = max(-127, min(127, spinner_target)) / 127.0
    targets = torch.tensor([[float(fire_target), float(zap_target), normalized_spinner]], 
                           dtype=torch.float32).to(device)
    
    model.optimizer.zero_grad()
    preds, _ = model(state_tensor)
    loss = nn.MSELoss()(preds, targets)
    
    loss.backward()
    model.optimizer.step()
    
    actions = preds.detach().cpu().numpy()[0]
    return loss.item(), actions

def safe_add_to_buffer(buffer, obs, next_obs, action, reward, done):
    """Add experience to RL replay buffer with validation."""
    try:
        obs = tensor_to_numpy(obs).flatten()
        next_obs = tensor_to_numpy(next_obs).flatten()
        action = tensor_to_numpy(action).flatten()
        
        if obs.shape != (NumberOfParams,) or next_obs.shape != (NumberOfParams,) or action.shape != (3,):
            obs = np.resize(obs, NumberOfParams)
            next_obs = np.resize(next_obs, NumberOfParams)
            action = np.resize(action, 3)
        
        reward_scalar = float(tensor_to_numpy(reward))
        done_scalar = int(bool(tensor_to_numpy(done)))
        
        with replay_buffer_lock:
            replay_buffer_batch.append((obs, next_obs, action, reward_scalar, done_scalar))
            if len(replay_buffer_batch) >= 32:  # Batch size for efficiency
                for item in replay_buffer_batch:
                    buffer.add(*item, [{}] if hasattr(buffer, 'handle_timeout_termination') else None)
                replay_buffer_batch.clear()
        
        return True
    except Exception as e:
        print(f"Error adding to buffer: {e}")
        return False

def flush_replay_buffer_batch(buffer):
    """Flush remaining replay buffer batch."""
    with replay_buffer_lock:
        for item in replay_buffer_batch:
            buffer.add(*item, [{}] if hasattr(buffer, 'handle_timeout_termination') else None)
        replay_buffer_batch.clear()

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
    """Train BC model by replaying demonstration log file using process_frame_data."""
    if not os.path.exists(log_file_path):
        print(f"Error: Log file {log_file_path} not found")
        return False
    
    print(f"Replaying {log_file_path}")
    frame_count = 0
    batch_size = 128
    training_data = []
    batch_loss = 0
    total_loss = 0
    num_batches = 0
    
    total_fire = 0
    total_zap = 0
    total_reward = 0
    total_spinner = 0
    
    format_string = ">IdBBBIIBBBhB"
    header_size = struct.calcsize(format_string)
    
    with open(log_file_path, 'rb') as f:
        frames_processed = 0
        while frame_count < MaxLogFrames:
            header_bytes = f.read(header_size)
            if not header_bytes or len(header_bytes) < header_size:
                print(f"End of file reached after {frames_processed} frames")
                break
            
            header_values = struct.unpack(format_string, header_bytes)
            num_values = header_values[0]
            
            if (num_values != NumberOfParams):
                print(f"Warning: Invalid number of values: {num_values} != {NumberOfParams}")
                continue
            
            payload_size = num_values * 2
            payload_bytes = f.read(payload_size)
            if len(payload_bytes) < payload_size:
                print(f"Payload data incomplete after {frames_processed} frames")
                break
            
            frame_data = header_bytes + payload_bytes
            
            result = process_frame_data(frame_data)
            if result is None:
                print(f"Warning: Invalid frame data at frame {frames_processed}, skipping")
                continue
            
            state, reward, game_action, game_mode, done, is_attract, save_signal = result
            fire, zap, spinner = game_action
            
            frames_processed += 1
            total_fire += fire
            total_zap += zap
            total_reward += reward
            total_spinner += spinner
            
            action = (fire, zap, spinner)
            training_data.append((state, action, reward))
            
            if len(training_data) >= batch_size:
                batch_loss = train_model_with_batch(bc_model, training_data[:batch_size])
                total_loss += batch_loss
                num_batches += 1
                if frame_count % 100 == 0:
                    print(f"Frames: {frame_count} - Trained batch - loss: {batch_loss:.6f}")
                training_data = training_data[batch_size:]
                frame_count += batch_size
        
        if training_data:
            batch_loss = train_model_with_batch(bc_model, training_data)
            total_loss += batch_loss
            num_batches += 1  # Corrected from 'num STACKs'
            frame_count += len(training_data)
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('nan')
        
        print(f"Replay complete: {frame_count} frames processed")
        if frames_processed > 0:
            print(f"FINAL STATS - Avg Fire: {total_fire/frames_processed:.4f}, "
                  f"Avg Zap: {total_zap/frames_processed:.4f}, Avg Reward: {total_reward/frames_processed:.4f}, "
                  f"Avg Loss: {avg_loss:.6f}")
        
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
            normalized_spinner = max(-31, min(31, spinner)) / 31.0
            states.append(state)
            fire_targets.append(float(fire))
            zap_targets.append(float(zap))
            spinner_targets.append(normalized_spinner)
            rewards.append(float(reward))
        
        state_tensor = torch.FloatTensor(np.array(states)).to(device)
        targets = torch.FloatTensor([[f, z, s] for f, z, s in zip(fire_targets, zap_targets, spinner_targets)]).to(device)
        reward_tensor = torch.FloatTensor(rewards).to(device)
        
        model.optimizer.zero_grad()
        preds, spinner_out_raw = model(state_tensor)
        
        reward_weights = torch.log1p(reward_tensor - reward_tensor.min() + 1e-6)
        reward_weights = reward_weights / reward_weights.mean()
        reward_weights = reward_weights.unsqueeze(1)
        
        fire_zap_loss = F.binary_cross_entropy(preds[:, :2], targets[:, :2], reduction='none') * reward_weights
        fire_zap_loss = fire_zap_loss.mean()
        spinner_loss = F.mse_loss(preds[:, 2:], targets[:, 2:], reduction='none') * reward_weights * 0.5
        spinner_loss = spinner_loss.mean()
        
        loss = fire_zap_loss + spinner_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
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
            traceback.print_exc()

def background_rl_train():
    while True:
        if len(replay_buffer) < 64:
            time.sleep(0.01)
            continue
        batch = random.sample(replay_buffer, 64)
        states, actions, rewards, next_states, dones = map(torch.tensor, zip(*batch))
        states, actions, rewards, next_states, dones = [x.float().to(device) for x in [states, actions, rewards, next_states, dones]]

        predicted_actions = actor(states)
        loss = F.mse_loss(predicted_actions, actions)

        actor_optimizer.zero_grad()
        loss.backward()
        actor_optimizer.step()

        actor_losses.append(loss.item())  # Track loss here

frame_count = 0

def main():
    """Main execution loop for Tempest AI."""
    global frame_count
    env = TempestEnv()
    bc_model, rl_model = initialize_models(env)
    
    threading.Thread(target=background_rl_train, daemon=True).start()
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
    episode_rewards = deque(maxlen=5)
    total_episode_reward = 0
    last_training_request = time.time()
    training_interval = 0.1
    done_latch = False
    
    initial_exploration_ratio = 0.80
    min_exploration_ratio = 0.05
    exploration_decay = 0.9999
    current_exploration_ratio = initial_exploration_ratio
    
    with os.fdopen(os.open(LUA_TO_PY_PIPE, os.O_RDONLY | os.O_NONBLOCK), "rb") as lua_to_py, \
         open(PY_TO_LUA_PIPE, "wb") as py_to_lua:
        print("✔️ Lua connected.")
        
        while True:
            try:
                ready_to_read, _, _ = select.select([lua_to_py], [], [], 0.01)
                if not ready_to_read:
                    time.sleep(0.001)
                    continue
                
                data = lua_to_py.read()
                if not data:
                    time.sleep(0.001)
                    continue

                result = process_frame_data(data)
                if result is None:
                    print(f"Warning: Invalid data received, skipping frame.")
                    py_to_lua.write(struct.pack("bbb", 0, 0, 0))
                    py_to_lua.flush()
                    continue
                
                state, reward, game_action, game_mode, done, is_attract, save_signal = result
                
                if (state is None or reward is None or game_action is None or 
                    game_mode is None or done is None or is_attract is None):
                    print(f"Warning: Invalid data in processed result, skipping frame.")
                    py_to_lua.write(struct.pack("bbb", 0, 0, 0))
                    py_to_lua.flush()
                    continue
                
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

                if done and not done_latch:
                    done_latch = True
                    env.update_state(state, reward, game_action, True)
                    total_episode_reward += reward
                    episode_rewards.append(total_episode_reward)
                    if not is_attract and env.prev_state is not None and env.prev_action is not None:
                        safe_add_to_buffer(rl_model.replay_buffer, env.prev_state, env.state, 
                                          env.prev_action, reward, True)
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
                        replay_buffer.append((env.prev_state, env.prev_action, reward, env.state, done))

                action = encode_action(0, 0, 0)
                
                if is_attract and game_action:
                    try:
                        bc_training_queue.put((state, game_action, reward), block=False)
                        action = encode_action(1, 1, -1)
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
                                action_tensor = actor(state_tensor).squeeze(0)
                            action = action_tensor.cpu().numpy()
                    else:
                        with torch.no_grad():
                            action_tensor = actor(state_tensor).squeeze(0)
                        action = action_tensor.cpu().numpy()

                env.step(action)
                fire, zap, spinner = env.decode_action(action)
                py_to_lua.write(struct.pack("bbb", fire, zap, spinner))
                py_to_lua.flush()

                if frame_count % 100 == 0:
                    actor_loss_mean = np.mean(actor_losses) if actor_losses else float('nan')
                    mean_reward = np.mean(episode_rewards) if episode_rewards else float('nan')
                    print(f"Frame {frame_count}, Reward: {reward:.2f}, Done: {done}, Buffer Size: {len(replay_buffer)}")
                    print(f"Metrics - Actor Loss: {actor_loss_mean:.4f}, "
                        f"Mean Episode Reward: {mean_reward:.3f}, "
                        f"Exploration Ratio: {current_exploration_ratio:.3f}")

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

# Update initialize_models to load the actor:
def initialize_models(env):
    bc_model = BCModel()
    if os.path.exists(BC_MODEL_PATH):
        bc_model.train()  # Set training mode before loading
        bc_model.load_state_dict(torch.load(BC_MODEL_PATH, map_location=device))
        print(f"Loaded BC model from {BC_MODEL_PATH}")

    if os.path.exists(LATEST_MODEL_PATH):
        actor.train()  # Set training mode before loading
        actor.load_state_dict(torch.load(LATEST_MODEL_PATH, map_location=device))
        print(f"Loaded RL model from {LATEST_MODEL_PATH}")

    return bc_model, actor

def save_models(rl_model=None, bc_model=None):
    try:
        with bc_model_lock:
            torch.save(bc_model.state_dict(), BC_MODEL_PATH)
        torch.save(actor.state_dict(), LATEST_MODEL_PATH)
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
    return np.array([fire_val, zap_val, normalized_spinner], dtype=np.float32)

def tensor_to_numpy(tensor):
    """Safely convert a tensor to numpy, handling device placement."""
    if isinstance(tensor, torch.Tensor):
        if tensor.device.type != 'cpu':
            tensor = tensor.cpu()
        return tensor.numpy()
    return tensor

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    main()