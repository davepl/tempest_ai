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

# Constants
ShouldReplayLog = False
LogFile = "/Users/dave/mame/big.log"
MaxLogFrames = 100000

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

# Training metrics buffers
actor_losses = deque(maxlen=100)
critic_losses = deque(maxlen=100)
ent_coefs = deque(maxlen=100)
mean_rewards = deque(maxlen=100)
training_queue = queue.Queue()

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
        """Reset environment state, reward, and flags."""
        super().reset(seed=seed)
        self.state = np.zeros(NumberOfParams, dtype=np.float32)
        self.reward = np.float32(0.0)
        self.done = False
        self.is_attract_mode = False
        self.prev_state = None
        self.prev_action = None
        return self.state, {}

    def step(self, action):
        """Process action, update prev_action, and return current state/reward."""
        self.prev_action = np.array(action, dtype=np.float32)
        fire, zap, spinner = self.decode_action(action)
        return self.state, self.reward, self.done, False, {"action_taken": (fire, zap, spinner)}

    def update_state(self, game_state, reward, game_action=None, done=False):
        """Update state with Lua data, preserving previous state."""
        self.prev_state = self.state.copy()
        self.state = game_state.astype(np.float32)
        self.reward = np.float32(reward)
        self.done = bool(done)
        return self.state

    def decode_action(self, action):
        """Decode RL actions to game inputs."""
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
        # Ensure observations are on the correct device
        if observations.device != device:
            observations = observations.to(device)
        return self.feature_extractor(observations)

class ResidualBlock(nn.Module):
    """Residual block with layer normalization for better gradient flow."""
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.relu(x + self.block(x))

class BCModel(nn.Module):
    """Enhanced BC model with deeper architecture and residual connections."""
    def __init__(self, input_size=NumberOfParams):
        super().__init__()
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Deeper processing with residual connections
        self.deep_processing = nn.Sequential(
            ResidualBlock(512),
            nn.Dropout(0.2),
            ResidualBlock(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            ResidualBlock(256),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        
        # Separate action heads with their own processing
        self.fire_path = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.zap_path = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.spinner_path = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Better initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Learning rate scheduler and optimizer with HIGHER learning rate
        self.optimizer = optim.AdamW(self.parameters(), lr=0.003, weight_decay=1e-5)
        
        # More conservative scheduler - much higher patience, smaller factor
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.7, patience=50, 
            verbose=True, min_lr=1e-5
        )
        
        # Track loss history for further diagnostics
        self.loss_history = []
        
        self.to(device)
        self.train()

    def forward(self, x):
        if x.device != device:
            x = x.to(device)
        
        # Extract features
        features = self.feature_extractor(x)
        
        # Deep processing
        processed = self.deep_processing(features)
        
        # Get action outputs
        fire_out = torch.sigmoid(self.fire_path(processed))
        zap_out = torch.sigmoid(self.zap_path(processed))
        spinner_out = torch.tanh(self.spinner_path(processed))
        
        return torch.cat([fire_out, zap_out, spinner_out], dim=1)

def process_frame_data(data):
    """Unpack and normalize game frame data from Lua with same format for pipe/log."""
    try:
        # Default values
        save_signal = 0
        is_done = False
        frame_counter = 0
        game_mode = 0
        
        # Get start position in the binary data
        pos = 0
        
        # Read header size (should be 15)
        if len(data) < pos + 4:
            raise ValueError(f"Data too short for header size: {len(data)} bytes")
        header_size = struct.unpack(">I", data[pos:pos+4])[0]
        pos += 4
        
        if header_size != 15:
            print(f"Warning: Unexpected header size {header_size}, expected 15")
        
        # Read payload size
        if len(data) < pos + 4:
            raise ValueError(f"Data too short for payload size: {len(data)} bytes")
        payload_size = struct.unpack(">I", data[pos:pos+4])[0]
        pos += 4
        
        # Read header content (reward, zap, fire, spinner) - 7 bytes total
        if len(data) < pos + 7:
            raise ValueError(f"Data too short for header content: {len(data)} bytes")
        
        # Parse header components exactly like the log file
        reward_int = struct.unpack(">i", data[pos:pos+4])[0]  # Signed 32-bit integer
        reward = reward_int / 1000.0  # Divide by 1000 to get float
        pos += 4
        
        zap = struct.unpack(">B", bytes([data[pos]]))[0]  # Unsigned byte
        pos += 1
        
        fire = struct.unpack(">B", bytes([data[pos]]))[0]  # Unsigned byte
        pos += 1
        
        spinner = struct.unpack(">b", bytes([data[pos]]))[0]  # Signed byte
        pos += 1
        
        # print(f"PIPE DEBUG - Decoded header: Reward={reward:.3f}, Zap={zap}, Fire={fire}, Spinner={spinner}")
        
        # Read game state data
        if len(data) < pos + payload_size:
            raise ValueError(f"Data too short for payload: expected {payload_size}, got {len(data) - pos}")
        
        game_data = np.frombuffer(data[pos:pos+payload_size], dtype=np.uint16)
                   
        # Convert from uint16 to normalized float32 in range [-1, 1]
        game_data = game_data.astype(np.float32) - 32768.0
        state = game_data / np.where(game_data > 0, 32767.0, 32768.0).astype(np.float32)
        
        # Important: Always maintain consistent size with NumberOfParams
        if len(state) > NumberOfParams:
            state = state[:NumberOfParams]  # Truncate if too long
        elif len(state) < NumberOfParams:
            state = np.pad(state, (0, NumberOfParams - len(state)), 'constant')  # Pad if too short
            
        # Return with all the necessary values
        return state, frame_counter, float(reward), (fire, zap, spinner), True, is_done, save_signal
    except Exception as e:
        print(f"Error processing frame data: {e}")
        traceback.print_exc()
        return None, 0, 0.0, None, False, False, 0

def train_bc(model, state, fire_target, zap_target, spinner_target):
    """Train BC model on a single demonstration frame with enhanced spinner value handling."""
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Apply a bias towards smaller spinner values in training data
    normalized_spinner = max(-127, min(127, spinner_target)) / 127.0
    
    # Higher weight for spinner component to emphasize its importance
    spinner_weight = 1.5
    
    targets = torch.tensor([
        [float(fire_target), float(zap_target), normalized_spinner]], 
        dtype=torch.float32
    ).to(device)
    
    preds = model(state_tensor)
    
    # Weighted loss: emphasize spinner accuracy
    fire_zap_loss = F.binary_cross_entropy(preds[:, :2], targets[:, :2])
    spinner_loss = F.mse_loss(preds[:, 2:], targets[:, 2:]) * spinner_weight
    loss = fire_zap_loss + spinner_loss
    
    model.optimizer.zero_grad()
    loss.backward()
    model.optimizer.step()
    
    actions = preds.detach().cpu().numpy()[0].astype(np.float32)
    
    # Apply the easing function to prefer smaller spinner values during inference
    spinner_action = int(round(actions[2] * 127.0))
    if abs(spinner_action) > 20:
        spinner_action = int(np.sign(spinner_action) * (20 + (abs(spinner_action) - 20) * 0.5))
    
    return loss.item(), encode_action(int(actions[0] > 0.5), int(actions[1] > 0.5), spinner_action)

def safe_add_to_buffer(buffer, obs, next_obs, action, reward, done):
    """Add experience to RL replay buffer with validation."""
    try:
        # Normalize input arrays
        obs = np.asarray(obs, dtype=np.float32).flatten()
        next_obs = np.asarray(next_obs, dtype=np.float32).flatten()
        action = np.asarray(action, dtype=np.float32).flatten()
        
        # Ensure correct shapes
        if obs.shape != (NumberOfParams,) or next_obs.shape != (NumberOfParams,) or action.shape != (3,):
            obs = np.resize(obs, NumberOfParams)
            next_obs = np.resize(next_obs, NumberOfParams)
            action = np.resize(action, 3)
        
        # Convert to scalar values for reward and done
        reward_scalar = float(reward)
        done_scalar = bool(done)
        
        # For newer SB3 versions with tensor-based replay buffer
        if hasattr(buffer, 'device') and hasattr(buffer, '_store_transition'):
            # Make sure the device for the replay buffer data is consistent
            obs_tensor = torch.as_tensor(obs, device=buffer.device)
            next_obs_tensor = torch.as_tensor(next_obs, device=buffer.device)
            action_tensor = torch.as_tensor(action, device=buffer.device)
            reward_tensor = torch.as_tensor(reward_scalar, device=buffer.device)
            done_tensor = torch.as_tensor(done_scalar, device=buffer.device)
            
            # Use scalar tensors for reward and done (no extra dimensions)
            buffer._store_transition(
                obs_tensor, next_obs_tensor, action_tensor, 
                reward_tensor, done_tensor, [{}]
            )
        else:
            # For older SB3 versions with numpy-based replay buffer
            buffer.add(
                obs, next_obs, action, reward_scalar, done_scalar,
                [{}] if hasattr(buffer, 'handle_timeout_termination') else None
            )
        
        return True
    except Exception as e:
        print(f"Error adding to buffer: {e}")
        return False

def apply_minimal_compatibility_patches(model):
    """Ensure SB3 model has a logger and moves all tensors to the correct device."""
    if not hasattr(model, '_logger'):
        model._logger = Logger(folder=None, output_formats=[])
    
    # Move all model components to the specified device
    model.policy.to(device)
    model.actor.to(device)
    model.critic.to(device)
    
    # Fix potential shape issues in critic target networks
    if hasattr(model, 'critic_target'):
        model.critic_target.to(device)
    
    # Set replay buffer device
    if hasattr(model.replay_buffer, 'device'):
        model.replay_buffer.device = device
    
    # For SB3 v1.8.0+, ensure all tensors in the replay buffer are on the correct device
    if hasattr(model.replay_buffer, '_tensor_names'):
        for tensor_name in model.replay_buffer._tensor_names:
            if hasattr(model.replay_buffer, tensor_name):
                tensor = getattr(model.replay_buffer, tensor_name)
                if isinstance(tensor, torch.Tensor) and tensor.device != device:
                    setattr(model.replay_buffer, tensor_name, tensor.to(device))
    
    # Handle SAC-specific parameters
    if hasattr(model, 'ent_coef_optimizer'):
        # Ensure log_ent_coef is properly initialized and on correct device
        if hasattr(model, 'log_ent_coef') and isinstance(model.log_ent_coef, torch.nn.Parameter):
            model.log_ent_coef.data = model.log_ent_coef.data.to(device)
    
    # Check if there are any issues with target entropy shape
    if hasattr(model, 'target_entropy'):
        # Make sure target_entropy is a scalar and not an array
        if isinstance(model.target_entropy, (torch.Tensor, np.ndarray)) and hasattr(model.target_entropy, 'shape'):
            if len(model.target_entropy.shape) > 0 and model.target_entropy.shape[0] > 1:
                # Convert to scalar if it's not already
                if isinstance(model.target_entropy, torch.Tensor):
                    model.target_entropy = model.target_entropy[0].item()
                else:
                    model.target_entropy = float(model.target_entropy[0])

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
    
    # Stats tracking
    total_fire = 0
    total_zap = 0
    total_reward = 0
    total_spinner = 0
    
    try:
        with open(log_file_path, 'rb') as f:
            file_size = os.path.getsize(log_file_path)
            print(f"Log file size: {file_size} bytes")
            
            frames_processed = 0
            while frame_count < MaxLogFrames:
                # Read header size (should be 15)
                header_size_bytes = f.read(4)
                if not header_size_bytes or len(header_size_bytes) < 4:
                    print(f"End of file reached after {frames_processed} frames")
                    break
                
                header_size = struct.unpack(">I", header_size_bytes)[0]
                if header_size != 15:
                    print(f"Warning: Expected header size of 15 bytes, got {header_size}")
                    exit

                # Read payload size
                payload_size_bytes = f.read(4)
                if not payload_size_bytes or len(payload_size_bytes) < 4:
                    print(f"Failed to read payload size after {frames_processed} frames")
                    break
                
                payload_size = struct.unpack(">I", payload_size_bytes)[0]
                
                # Read header content (reward, zap, fire, spinner)
                header_data = f.read(7)  # 4+1+1+1 bytes
                if len(header_data) < 7:
                    print(f"Short read on header data: got {len(header_data)}, expected 7")
                    break
                
                # Parse header components EXACTLY as they're encoded in Lua
                reward_int = struct.unpack(">i", header_data[0:4])[0]  # Signed int (lowercase i)
                reward = reward_int / 1000.0  # Divide by 1000 to get float
                
                # Use struct.unpack for all values to match Lua encoding
                zap = struct.unpack(">B", bytes([header_data[4]]))[0]  # Unsigned byte
                fire = struct.unpack(">B", bytes([header_data[5]]))[0]  # Unsigned byte
                spinner = struct.unpack(">b", bytes([header_data[6]]))[0]  # Signed byte
                
                print(f"LOG DEBUG - Decoded: Reward={reward:.3f}, Zap={zap}, Fire={fire}, Spinner={spinner}")

                # Read game state data
                game_data_bytes = f.read(payload_size)
                if len(game_data_bytes) < payload_size:
                    print(f"Short read on game data: got {len(game_data_bytes)}, expected {payload_size}")
                    break
                
                # Convert to state array - match format in Lua params
                game_data = np.frombuffer(game_data_bytes, dtype=np.uint16)
                
                # Convert from uint16 to normalized float32 in range [-1, 1]
                game_data = game_data.astype(np.float32) - 32768.0
                state = game_data / np.where(game_data > 0, 32767.0, 32768.0).astype(np.float32)
                
                # Ensure consistent size
                if len(state) > NumberOfParams:
                    state = state[:NumberOfParams]  # Truncate if too long
                elif len(state) < NumberOfParams:
                    state = np.pad(state, (0, NumberOfParams - len(state)), 'constant')  # Pad if too short
                
                frames_processed += 1
                
                # Update stats
                total_fire += fire
                total_zap += zap
                total_reward += reward
                total_spinner += spinner
                
                # Print occasional debug info
                if frames_processed <= 10 or frames_processed % 1000 == 0:
                    print(f"Frame {frames_processed}: Reward={reward:.3f}, Fire={fire}, Zap={zap}, Spinner={spinner}")
                    print(f"Game data shape: {state.shape}, Payload size: {payload_size}")
                
                # Add to training data
                action = (fire, zap, spinner)
                training_data.append((state, action, reward))
                
                # Train in batches
                if len(training_data) >= batch_size:
                    batch_loss = train_model_with_batch(bc_model, training_data[:batch_size])
                    if frame_count % 100 == 0:
                        print(f"Trained batch - loss: {batch_loss:.6f}")
                    training_data = training_data[batch_size:]
                    frame_count += batch_size
            
            # Train any remaining data
            if training_data:
                batch_loss = train_model_with_batch(bc_model, training_data)
                print(f"Trained final batch - loss: {batch_loss:.6f}")
                frame_count += len(training_data)
            
            print(f"Replay complete: {frame_count} frames added from {frames_processed} processed")
            if frames_processed > 0:
                print(f"FINAL STATS - Avg Fire: {total_fire/frames_processed:.4f}, " +
                      f"Avg Zap: {total_zap/frames_processed:.4f}, Avg Reward: {total_reward/frames_processed:.4f}")
            
            torch.save(bc_model.state_dict(), BC_MODEL_PATH)
            return True
            
    except Exception as e:
        print(f"Error during replay: {e}")
        traceback.print_exc()
        if frame_count > 0:
            print(f"Saving partial model after {frame_count} frames")
            torch.save(bc_model.state_dict(), BC_MODEL_PATH)
        return False

def train_model_with_batch(model, batch):
    """Train model with a batch of data and reward information."""
    model.train()
    
    # Create tensors for batch training
    states = []
    fire_targets = []
    zap_targets = []
    spinner_targets = []
    rewards = []  # Now tracking rewards
    
    for state, game_action, reward in batch:
        fire, zap, spinner = game_action
        # Normalize spinner to [-1, 1]
        normalized_spinner = max(-127, min(127, spinner)) / 127.0
        
        states.append(state)
        fire_targets.append(float(fire))
        zap_targets.append(float(zap))
        spinner_targets.append(normalized_spinner)
        rewards.append(float(reward))
    
    # Convert to tensors
    state_tensor = torch.FloatTensor(np.array(states)).to(device)
    targets = torch.FloatTensor([
        [fire, zap, spinner] for fire, zap, spinner in 
        zip(fire_targets, zap_targets, spinner_targets)
    ]).to(device)
    reward_tensor = torch.FloatTensor(rewards).to(device)
    
    # Training step
    model.optimizer.zero_grad()
    preds = model(state_tensor)
    
    # Weight the loss by reward - actions with higher rewards matter more
    # Add small constant to avoid zero rewards
    reward_weights = torch.clamp(reward_tensor, min=0.1) / torch.clamp(reward_tensor, min=0.1).mean()
    reward_weights = reward_weights.unsqueeze(1)
    
    # Balanced loss components with reward weighting
    fire_zap_loss = F.binary_cross_entropy(preds[:, :2], targets[:, :2], reduction='none') * reward_weights
    fire_zap_loss = fire_zap_loss.mean()
    
    spinner_loss = F.mse_loss(preds[:, 2:], targets[:, 2:], reduction='none') * reward_weights * 1.5
    spinner_loss = spinner_loss.mean()
    
    loss = fire_zap_loss + spinner_loss
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    model.optimizer.step()
    
    # Update learning rate based on loss
    if hasattr(model, 'scheduler') and hasattr(model, 'loss_history'):
        model.loss_history.append(loss.item())
        if len(model.loss_history) >= 10:
            avg_loss = sum(model.loss_history) / len(model.loss_history)
            model.scheduler.step(avg_loss)
            model.loss_history = []
    
    return loss.item()

def background_train(rl_model):
    """Background RL training thread with queue-based triggering."""
    while True:
        try:
            training_queue.get(timeout=1.0)
            if rl_model.replay_buffer.pos > rl_model.learning_starts:
                # Ensure all components are on the specified device before training
                apply_minimal_compatibility_patches(rl_model)
                
                # Start with smaller batch size and gradually increase to avoid broadcasting issues
                buffer_size = rl_model.replay_buffer.pos
                # Use a smaller batch size to start, ensuring we don't have dimension mismatch
                if buffer_size < 100:
                    batch_size = 8
                elif buffer_size < 500:
                    batch_size = 16
                elif buffer_size < 1000:
                    batch_size = 32
                else:
                    batch_size = 64
                
                try:
                    # Wrap training in another try-except to catch tensor shape issues
                    rl_model.train(gradient_steps=10, batch_size=batch_size)
                    
                    if hasattr(rl_model, 'logger'):
                        log_vals = rl_model.logger.name_to_value
                        actor_losses.append(log_vals.get('train/actor_loss', float('nan')))
                        critic_losses.append(log_vals.get('train/critic_loss', float('nan')))
                        ent_coefs.append(log_vals.get('train/ent_coef', float('nan')))
                except RuntimeError as rt_err:
                    if "doesn't match the broadcast shape" in str(rt_err):
                        print(f"Broadcasting error with batch size {batch_size}, retrying with smaller batch...")
                        # Retry with smaller batch size
                        if batch_size > 8:
                            reduced_batch = max(8, batch_size // 2)
                            try:
                                rl_model.train(gradient_steps=5, batch_size=reduced_batch)
                            except Exception as retry_err:
                                print(f"Still failed with reduced batch size: {retry_err}")
                    else:
                        print(f"Training runtime error: {rt_err}")
            
            training_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Training error: {e}")
            # Try to fix device issues on error
            try:
                apply_minimal_compatibility_patches(rl_model)
            except Exception as fix_error:
                print(f"Error fixing device issues: {fix_error}")
            training_queue.task_done()

def main():
    """Main execution loop for Tempest AI."""
    env = TempestEnv()
    bc_model, rl_model = initialize_models(env)
    threading.Thread(target=background_train, args=(rl_model,), daemon=True).start()

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
    
    # Track game and model performance over time
    episode_rewards = deque(maxlen=5)
    total_episode_reward = 0
    
    with os.fdopen(os.open(LUA_TO_PY_PIPE, os.O_RDONLY | os.O_NONBLOCK), "rb") as lua_to_py, \
         open(PY_TO_LUA_PIPE, "wb") as py_to_lua:
        print("✔️ Lua connected.")
        
        # Track spinner movements for analytics
        spinner_history = deque(maxlen=100)
        positive_spinners = 0
        negative_spinners = 0
        
        # Exploration annealing
        initial_exploration_ratio = 1.0
        min_exploration_ratio = 0.05
        exploration_decay = 0.9999  # Slow decay
        current_exploration_ratio = initial_exploration_ratio
        
        # For debugging directional bias
        directional_balance_history = deque(maxlen=20)
        
        while True:
            try:
                data = lua_to_py.read()
                if not data:
                    time.sleep(0.01)
                    continue
                state, frame_counter, reward, game_action, is_attract, done, save_signal = process_frame_data(data)
                if state is None:
                    continue

                env.update_state(state, reward, game_action, done)
                total_episode_reward += reward
                
                if done:
                    # End of episode tracking
                    episode_rewards.append(total_episode_reward)
                    print(f"Episode finished! Reward: {total_episode_reward:.2f}, "
                          f"Avg Episode Reward: {np.mean(episode_rewards):.2f}")
                    env.reset()
                    frame_count = 0
                    total_episode_reward = 0
                    # Reset bias tracking
                    positive_spinners = 0
                    negative_spinners = 0

                # Add to replay buffer with careful device handling
                if not is_attract and env.prev_state is not None and env.prev_action is not None:
                    safe_add_to_buffer(rl_model.replay_buffer, env.prev_state, env.state, env.prev_action, reward, done)

                if is_attract:
                    action = train_bc(bc_model, state, *game_action)[1] if game_action else encode_action(0, 0, 0)
                else:
                    # Anneal exploration over time
                    if frame_count % 1000 == 0 and current_exploration_ratio > min_exploration_ratio:
                        current_exploration_ratio *= exploration_decay
                    
                    # Get BC or RL prediction with balanced exploration
                    if random.random() < current_exploration_ratio:
                        # BC action for exploration
                        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                        bc_action = bc_model(state_tensor).detach().cpu().numpy()[0]
                        
                        # Use BC prediction but add small noise to avoid getting stuck
                        fire_noise = 0 if random.random() < 0.9 else (1 if bc_action[0] < 0.5 else 0) 
                        zap_noise = 0 if random.random() < 0.9 else (1 if bc_action[1] < 0.5 else 0)
                        
                        # Small directional noise for spinner with bias correction
                        # Check recent directional balance and add noise in opposite direction
                        spinner_bias_correction = 0
                        if len(directional_balance_history) > 10:
                            positive_ratio = sum(1 for d in directional_balance_history if d > 0) / len(directional_balance_history)
                            if positive_ratio > 0.7:  # Too many positive values
                                spinner_bias_correction = -0.05  # Encourage negative
                            elif positive_ratio < 0.3:  # Too many negative values
                                spinner_bias_correction = 0.05   # Encourage positive
                        
                        # Add spinner balance correction to the action
                        spinner_noise = np.random.normal(0, 0.05) + spinner_bias_correction
                        bc_action[2] = np.clip(bc_action[2] + spinner_noise, -1.0, 1.0)
                        
                        action = bc_action
                    else:
                        # RL action with exploration noise biased towards smaller spinner deltas
                        state_tensor = torch.tensor(state.reshape(1, -1), dtype=torch.float32)
                        deterministic = frame_count > 10000 or random.random() < 0.7  # Increase deterministic actions
                        raw_action = rl_model.predict(state_tensor, deterministic=deterministic)[0].flatten()
                        
                        # Apply directional bias correction if needed
                        if not deterministic and len(directional_balance_history) > 10:
                            positive_ratio = sum(1 for d in directional_balance_history if d > 0) / len(directional_balance_history)
                            if positive_ratio > 0.7:  # Too many positive values
                                # Encourage negative spinner values
                                if raw_action[2] > 0:
                                    raw_action[2] *= 0.5  # Reduce positive values
                            elif positive_ratio < 0.3:  # Too many negative values
                                # Encourage positive spinner values
                                if raw_action[2] < 0:
                                    raw_action[2] *= 0.5  # Reduce negative values
                                    
                        action = raw_action

                env.step(action)
                fire, zap, spinner = env.decode_action(action)
                
                # Track spinner actions for analytics and bias correction
                spinner_history.append(spinner)
                if spinner > 0:
                    positive_spinners += 1
                elif spinner < 0:
                    negative_spinners += 1
                
                # Track directional balance for later correction
                directional_balance_history.append(spinner)
                
                py_to_lua.write(struct.pack("bbb", fire, zap, spinner))
                py_to_lua.flush()

                if frame_count % 10 == 0 and rl_model.replay_buffer.pos > rl_model.learning_starts:
                    training_queue.put(True)
                    mean_rewards.append(reward)

                if frame_count % 100 == 0:
                    actor_loss_mean = np.nanmean(actor_losses) if actor_losses else np.nan
                    critic_loss_mean = np.nanmean(critic_losses) if critic_losses else np.nan
                    ent_coef_mean = np.nanmean(ent_coefs) if ent_coefs else np.nan
                    reward_mean = np.nanmean(mean_rewards) if mean_rewards else np.nan
                    print(f"Frame {frame_count}, Reward: {reward:.2f}, Done: {done}, Buffer Size: {rl_model.replay_buffer.pos} Losses: Actor: ", actor_loss_mean, "Critic: ", critic_loss_mean, "Entropy: ", ent_coef_mean)

                    # Report spinner statistics with directional bias info
                    if spinner_history:
                        spinner_abs = [abs(s) for s in spinner_history]
                        spinner_avg = np.mean(spinner_abs)
                        spinner_max = np.max(spinner_abs)
                        total_spinners = positive_spinners + negative_spinners
                        dir_ratio = 0.5
                        if total_spinners > 0:
                            dir_ratio = positive_spinners / total_spinners
                        
                        print(f"Spinner stats - Avg: {spinner_avg:.2f}, Max: {spinner_max}, "
                              f"% small (<10): {sum(1 for s in spinner_abs if s < 10)/len(spinner_abs)*100:.1f}%, "
                              f"Dir bias: {dir_ratio:.2f} (+/-)")
                    
                    if not is_attract:
                        print(f"Metrics - Actor Loss: {actor_loss_mean:.6f}, Critic Loss: {critic_loss_mean:.6f}, "
                              f"Entropy Coef: {ent_coef_mean:.6f}, Mean Reward: {reward_mean:.2f}, "
                          f"Expl: {current_exploration_ratio:.3f}")

                if save_signal:
                    save_models(rl_model, bc_model)
                    # Also reset exploration temporarily to try new strategies after save
                    current_exploration_ratio = min(0.1, current_exploration_ratio * 2.0)
                frame_count += 1

            except KeyboardInterrupt:
                save_models(rl_model, bc_model)
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(5) 
                initialize_models(env)
    threading.Thread(target=background_train, args=(rl_model,), daemon=True).start()

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
    with os.fdopen(os.open(LUA_TO_PY_PIPE, os.O_RDONLY | os.O_NONBLOCK), "rb") as lua_to_py, \
         open(PY_TO_LUA_PIPE, "wb") as py_to_lua:
        print("✔️ Lua connected.")
        
        # Track spinner movements for analytics
        spinner_history = deque(maxlen=100)
        
        while True:
            try:
                data = lua_to_py.read()
                if not data:
                    time.sleep(0.01)
                    continue
                state, frame_counter, reward, game_action, is_attract, done, save_signal = process_frame_data(data)
                if state is None:
                    continue

                env.update_state(state, reward, game_action, done)
                if done:
                    env.reset()
                    frame_count = 0

                # Add to replay buffer with careful device handling
                if not is_attract and env.prev_state is not None and env.prev_action is not None:
                    safe_add_to_buffer(rl_model.replay_buffer, env.prev_state, env.state, env.prev_action, reward, done)

                if is_attract:
                    action = train_bc(bc_model, state, *game_action)[1] if game_action else encode_action(0, 0, 0)
                else:
                    # Get BC or RL prediction with device consistency and bias towards smaller spinner movements
                    if frame_count < 5000 and random.random() < 0.2:
                        # BC action for exploration
                        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                        action = bc_model(state_tensor).detach().cpu().numpy()[0]
                    else:
                        # RL action with exploration noise biased towards smaller spinner deltas
                        state_tensor = torch.tensor(state.reshape(1, -1), dtype=torch.float32)
                        raw_action = rl_model.predict(state_tensor, deterministic=frame_count > 10000)[0].flatten()
                        
                        # Apply additional spinner bias based on observed game state patterns
                        # This creates a "guided exploration" that focuses on smaller movements
                        if not rl_model.predict(state_tensor, deterministic=True)[0].flatten()[2] == raw_action[2]:
                            # We're in exploration mode for spinner
                            # Bias exploration toward smaller deltas (less than 20) using a beta distribution
                            spinner_scale = 0.15  # Controls the scale of typical random movements
                            spinner_delta = np.random.beta(1.5, 3.0) * 2.0 - 1.0  # Beta dist centered and scaled
                            raw_action[2] = spinner_delta * spinner_scale
                            
                        action = raw_action

                env.step(action)
                fire, zap, spinner = env.decode_action(action)
                
                # Track spinner actions for analytics
                spinner_history.append(spinner)
                
                py_to_lua.write(struct.pack("bbb", fire, zap, spinner))
                py_to_lua.flush()

                if frame_count % 10 == 0 and rl_model.replay_buffer.pos > rl_model.learning_starts:
                    training_queue.put(True)
                    mean_rewards.append(reward)

                if frame_count % 100 == 0:
                    print(f"Frame {frame_count}, Reward: {reward}, Done: {done}, Buffer Size: {rl_model.replay_buffer.pos}")
                    actor_loss_mean = np.nanmean(actor_losses) if actor_losses else np.nan
                    critic_loss_mean = np.nanmean(critic_losses) if critic_losses else np.nan
                    ent_coef_mean = np.nanmean(ent_coefs) if ent_coefs else np.nan
                    reward_mean = np.nanmean(mean_rewards) if mean_rewards else np.nan
                    
                    # Report spinner statistics
                    if spinner_history:
                        spinner_abs = [abs(s) for s in spinner_history]
                        spinner_avg = np.mean(spinner_abs)
                        spinner_max = np.max(spinner_abs)
                        print(f"Spinner stats - Avg: {spinner_avg:.2f}, Max: {spinner_max}, "
                              f"% small moves (<10): {sum(1 for s in spinner_abs if s < 10)/len(spinner_abs)*100:.1f}%")
                    
                    print(f"Metrics - Actor Loss: {actor_loss_mean:.6f}, Critic Loss: {critic_loss_mean:.6f}, "
                          f"Entropy Coef: {ent_coef_mean:.6f}, Mean Reward: {reward_mean:.2f}")

                if save_signal:
                    save_models(rl_model, bc_model)
                frame_count += 1

            except KeyboardInterrupt:
                save_models(rl_model, bc_model)
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(5)

def initialize_models(env):
    """Initialize models with improved SAC parameters."""
    bc_model = BCModel()
    if os.path.exists(BC_MODEL_PATH):
        bc_model.load_state_dict(torch.load(BC_MODEL_PATH, map_location=device))
        print(f"Loaded BC model from {BC_MODEL_PATH}")

    # Use a lower learning rate and smaller batch size to improve stability
    # Set entropy target to a smaller value to reduce exploration
    rl_model = SAC("MlpPolicy", env, policy_kwargs={
        "features_extractor_class": TempestFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": 128},
        "net_arch": dict(pi=[256, 128], qf=[256, 128])
    }, learning_rate=0.0005,  # Reduced from 0.001
       buffer_size=100000, 
       learning_starts=1000,
       batch_size=32,       # Reduced from 64 
       train_freq=(10, "step"),
       gradient_steps=5,    # Reduced from 10
       ent_coef="auto",     # Auto entropy adjustment
       target_entropy=-1.5, # Set a specific target entropy value
       tau=0.005,          # Soft update coefficient
       gamma=0.99,         # Discount factor
       device=device,
       verbose=1)
    
    if os.path.exists(LATEST_MODEL_PATH):
        rl_model = SAC.load(LATEST_MODEL_PATH, env=env, device=device)
        print(f"Loaded RL model from {LATEST_MODEL_PATH}")
    
    # Fix target entropy to be a scalar
    if hasattr(rl_model, 'target_entropy'):
        # Check if it's a tensor and convert to scalar if needed
        if isinstance(rl_model.target_entropy, torch.Tensor):
            rl_model.target_entropy = float(rl_model.target_entropy.item())
        # Set a reasonable target entropy
        action_dim = env.action_space.shape[0]
        rl_model.target_entropy = -0.5 * action_dim
        print(f"Set target entropy to {rl_model.target_entropy}")
    
    # Make sure the model is properly set up with consistent device placement
    apply_minimal_compatibility_patches(rl_model)
    return bc_model, rl_model

def save_models(rl_model, bc_model):
    """Save RL and BC models to disk."""
    try:
        # Ensure consistent device state before saving
        apply_minimal_compatibility_patches(rl_model)
        rl_model.save(LATEST_MODEL_PATH)
        torch.save(bc_model.state_dict(), BC_MODEL_PATH)
        print(f"Models saved to {MODEL_DIR}")
    except Exception as e:
        print(f"Error saving models: {e}")

def encode_action(fire, zap, spinner_delta):
    """Encode actions for RL model consistency with easing function for spinner values.
    
    Args:
        fire (int): 0 or 1 for fire action
        zap (int): 0 or 1 for zap action
        spinner_delta (int): Value between -127 and 127 for spinner movement
        
    Returns:
        numpy.ndarray: Normalized action vector [fire, zap, spinner] in range [0,1,[-1,1]]
    """
    # Convert to float values first
    fire_val = float(fire)
    zap_val = float(zap)
    
    # Apply stronger squashing for large spinner values
    # Use custom sigmoid-based function to ensure balanced positive/negative distribution
    if abs(spinner_delta) < 10:
        # Small deltas pass through with minimal transformation
        normalized_spinner = spinner_delta / 127.0
    else:
        # Larger deltas get increasingly squashed - sign preserved
        sign = np.sign(spinner_delta)
        magnitude = abs(spinner_delta)
        # Apply stronger squashing to large values
        squashed = 10.0 + (magnitude - 10.0) * (1.0 / (1.0 + (magnitude - 10.0) / 20.0))
        normalized_spinner = sign * min(squashed / 127.0, 1.0)
    
    # Verify the value is in range before returning
    spinner_val = max(-1.0, min(1.0, normalized_spinner))
    
    return np.array([fire_val, zap_val, spinner_val], dtype=np.float32)

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    main()