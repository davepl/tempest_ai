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
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

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
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer as SB3ReplayBuffer
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
import torch.nn.functional as F
import threading

# Define global shutdown tracking variable
shutdown_requested = False

# Define the paths to the named pipes
LUA_TO_PY_PIPE = "/tmp/lua_to_py"
PY_TO_LUA_PIPE = "/tmp/py_to_lua"

# Create a directory for model checkpoints
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Define paths for latest models and checkpoints
LATEST_MODEL_PATH = os.path.join(MODEL_DIR, "tempest_model_latest.zip")
BC_MODEL_PATH = os.path.join(MODEL_DIR, "tempest_bc_model.pt")

# Define the device globally
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

class TempestEnv(gym.Env):
    """
    Custom Gymnasium environment for Tempest arcade game.
    """
    metadata = {"render_modes": ["human"], "render_fps": 60}
    
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            shape=(3,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(243,),
            dtype=np.float32
        )
        self.state = np.zeros(243, dtype=np.float32)
        self.player_inputs = np.zeros(3, dtype=np.float32)
        self.reward = 0
        self.done = False
        self.info = {}
        self.episode_step = 0
        self.total_reward = 0
        self.is_attract_mode = False
        self.prev_state = None
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.zeros(243, dtype=np.float32)
        self.player_inputs = np.zeros(3, dtype=np.float32)
        self.reward = 0
        self.done = False
        self.episode_step = 0
        self.total_reward = 0
        self.prev_state = None
        return self.state, self.info
    
    def step(self, action):
        fire = 1 if action[0] > 0.5 else 0
        zap = 1 if action[1] > 0.5 else 0
        spinner_delta = int(round(action[2] * 128.0))
        spinner_delta = max(-128, min(127, spinner_delta))
        self.player_inputs = np.array([fire, zap, spinner_delta], dtype=np.float32)
        self.episode_step += 1
        self.total_reward += self.reward
        terminated = self.done
        truncated = self.episode_step >= 10000
        self.info = {
            "action_taken": (fire, zap, spinner_delta),
            "episode_step": self.episode_step,
            "total_reward": self.total_reward,
            "attract_mode": self.is_attract_mode,
            "player_inputs": self.player_inputs
        }
        return self.state, self.reward, terminated, truncated, self.info
    
    def update_state(self, game_state, reward, game_action=None, done=False):
        self.prev_state = self.state.copy() if self.state is not None else None
        self.state = game_state
        if game_action is not None:
            self.player_inputs = np.array([game_action[0], game_action[1], game_action[2]], dtype=np.float32)
        self.reward = reward
        self.done = done
        if game_action is not None:
            self.info["game_action"] = game_action
            self.info["player_inputs"] = self.player_inputs
        return self.state

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
    def __init__(self, input_size=243):
        super(BCModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fire_output = nn.Linear(64, 1)
        self.zap_output = nn.Linear(64, 1)
        self.spinner_output = nn.Linear(64, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        fire = torch.sigmoid(self.fire_output(x))
        zap = torch.sigmoid(self.zap_output(x))
        spinner = self.spinner_output(x)
        return fire, zap, spinner

class SaveOnSignalCallback(BaseCallback):
    def __init__(self, save_path, bc_model=None, verbose=1):
        try:
            super().__init__(verbose)
        except AttributeError:
            self._verbose = verbose
        self.save_path = save_path
        self.bc_model = bc_model
        self.force_save = False
        self.model = None

    def _on_step(self):
        if self.force_save:
            self._save_models()
            self.force_save = False
        return True

    def _save_models(self):
        try:
            if self.model is not None:
                self.model.save(self.save_path)
            if self.bc_model is not None:
                torch.save(self.bc_model.state_dict(), BC_MODEL_PATH)
            print(f"Models saved to {MODEL_DIR}")
        except Exception as e:
            print(f"Error saving models: {e}")

    def signal_save(self):
        self._save_models()

def process_frame_data(data):
    global shutdown_requested
    if len(data) < 24:
        return None, 0, 0.0, None, False, False, False
    try:
        header_fmt = ">IdBBBIIBBBh"
        header_size = struct.calcsize(header_fmt)
        (num_values, reward, game_action, game_mode, is_done, frame_counter, score, save_signal, fire_commanded, zap_commanded, spinner_delta) = struct.unpack(header_fmt, data[:header_size])
        if save_signal:
            shutdown_requested = True
        header_size = 4 + (num_values * 8) + 3 + 8 + 1
        game_data = data[header_size:]
        num_ints = len(game_data) // 2
        unpacked_data = [struct.unpack(">H", game_data[i*2:i*2+2])[0] - 32768 for i in range(num_ints)]
        normalized_data = np.array([float(x) / 32767.0 if x > 0 else float(x) / 32768.0 for x in unpacked_data], dtype=np.float32)
        is_attract = (game_mode & 0x80) == 0
        expected_size = 243
        if len(normalized_data) < expected_size:
            padded_data = np.zeros(expected_size, dtype=np.float32)
            padded_data[:len(normalized_data)] = normalized_data
            normalized_data = padded_data
        elif len(normalized_data) > expected_size:
            normalized_data = normalized_data[:expected_size]
        # Ensure observation has correct shape (243,)
        return normalized_data.reshape(243), frame_counter, reward, (fire_commanded, zap_commanded, spinner_delta), is_attract, is_done, save_signal
    except Exception as e:
        print(f"Error processing frame data: {e}")
        return None, 0, 0.0, None, False, False, False

def train_bc(model, state, fire_target, zap_target, spinner_target):
    model = model.to(device)
    # Ensure state has correct shape (243,)
    state_tensor = torch.FloatTensor(state.reshape(243)).unsqueeze(0).to(device)
    fire_target = torch.FloatTensor([fire_target]).unsqueeze(1).to(device)
    zap_target = torch.FloatTensor([zap_target]).unsqueeze(1).to(device)
    spinner_target_normalized = spinner_target / 64.0
    spinner_target_tensor = torch.FloatTensor([spinner_target_normalized]).unsqueeze(1).to(device)
    fire_pred, zap_pred, spinner_pred = model(state_tensor)
    bce_loss = nn.BCELoss()
    mse_loss = nn.MSELoss()
    fire_loss = bce_loss(fire_pred, fire_target)
    zap_loss = bce_loss(zap_pred, zap_target)
    spinner_loss = mse_loss(spinner_pred, spinner_target_tensor)
    total_loss = fire_loss + zap_loss + spinner_loss
    model.optimizer.zero_grad()
    total_loss.backward()
    model.optimizer.step()
    fire_action = 1 if fire_pred.item() > 0.5 else 0
    zap_action = 1 if zap_pred.item() > 0.5 else 0
    spinner_action = int(round(spinner_pred.item() * 64.0))
    spinner_action = max(-64, min(63, spinner_action))
    return total_loss.item(), (fire_action, zap_action, spinner_action)

def get_buffer_size(buffer):
    """Safely get the size of a replay buffer with different SB3 versions"""
    if buffer is None:
        return 0
    try:
        if hasattr(buffer, 'size'):
            return buffer.size()
        elif hasattr(buffer, '__len__'):
            return len(buffer)
        elif hasattr(buffer, 'buffer_size'):
            return buffer.buffer_size
        elif hasattr(buffer, 'pos'):
            return buffer.pos
        else:
            if isinstance(buffer, dict) and 'observations' in buffer:
                return len(buffer['observations'])
            return 0
    except Exception as e:
        print(f"Error getting buffer size: {e}")
        return 0

def normalize_action(action, game_action=None):
    """Normalize action to ensure it has shape (3,) regardless of input format"""
    # If action is already a numpy array with the right shape, return it
    if isinstance(action, np.ndarray) and action.shape == (3,):
        return action
    
    # If action is a tuple or list of 3 elements, convert to numpy array
    if isinstance(action, (tuple, list)) and len(action) == 3:
        return np.array(action, dtype=np.float32)
    
    # If action is a numpy array with wrong shape but right size
    if isinstance(action, np.ndarray) and action.size == 3:
        return action.reshape(3)
    
    # If we have a game action tuple, use it
    if game_action is not None and len(game_action) == 3:
        return np.array(game_action, dtype=np.float32)
    
    # Default: return zeros and log warning
    print(f"Warning: Could not normalize action {action}, returning zeros")
    return np.zeros(3, dtype=np.float32)

def ensure_buffer_consistency(buffer):
    """
    Ensure replay buffer arrays have consistent dimensions.
    Only called once during initialization, not repeatedly.
    """
    if buffer is None or not hasattr(buffer, 'observations'):
        return
    
    try:
        buffer_size = len(buffer.observations)
        
        # Only log and fix if actually needed
        needs_fix = False
        
        # Check observations shape
        if hasattr(buffer, 'observations') and (len(buffer.observations.shape) != 2 or buffer.observations.shape[1] != 243):
            needs_fix = True
        
        # Check actions shape
        if hasattr(buffer, 'actions') and (len(buffer.actions.shape) != 2 or buffer.actions.shape[1] != 3):
            needs_fix = True
        
        if needs_fix:
            print(f"Fixing replay buffer dimensions. Size: {buffer_size}")
            
            # Fix observations
            if hasattr(buffer, 'observations') and (len(buffer.observations.shape) != 2 or buffer.observations.shape[1] != 243):
                fixed_obs = np.zeros((buffer_size, 243), dtype=np.float32)
                for i in range(buffer_size):
                    if i < len(buffer.observations):
                        obs = buffer.observations[i]
                        fixed_obs[i] = obs.flatten()[:243] if obs.size >= 243 else np.pad(obs.flatten(), (0, 243 - min(obs.size, 243)))
                buffer.observations = fixed_obs
            
            # Fix next_observations
            if hasattr(buffer, 'next_observations') and (len(buffer.next_observations.shape) != 2 or buffer.next_observations.shape[1] != 243):
                fixed_next_obs = np.zeros((buffer_size, 243), dtype=np.float32)
                for i in range(buffer_size):
                    if i < len(buffer.next_observations):
                        next_obs = buffer.next_observations[i]
                        fixed_next_obs[i] = next_obs.flatten()[:243] if next_obs.size >= 243 else np.pad(next_obs.flatten(), (0, 243 - min(next_obs.size, 243)))
                buffer.next_observations = fixed_next_obs
            
            # Fix actions
            if hasattr(buffer, 'actions') and (len(buffer.actions.shape) != 2 or buffer.actions.shape[1] != 3):
                fixed_actions = np.zeros((buffer_size, 3), dtype=np.float32)
                for i in range(buffer_size):
                    if i < len(buffer.actions):
                        action = buffer.actions[i]
                        # Normalize action to shape (3,)
                        fixed_actions[i] = normalize_action(action).flatten()[:3]
                buffer.actions = fixed_actions
            
            print("Replay buffer dimensions fixed")
        
    except Exception as e:
        print(f"Error fixing buffer dimensions: {e}")

def initialize_models():
    env = TempestEnv()
    bc_model = BCModel(input_size=243).to(device)
    bc_model.optimizer = optim.Adam(bc_model.parameters(), lr=0.001)
    if os.path.exists(BC_MODEL_PATH):
        try:
            state_dict = torch.load(BC_MODEL_PATH, map_location=device)
            bc_model.load_state_dict(state_dict)
            print(f"Successfully loaded BC model from {BC_MODEL_PATH}")
        except Exception as e:
            print(f"ERROR loading BC model: {e}")
    
    # Policy network setup with explicit output dimension
    policy_kwargs = dict(
        features_extractor_class=TempestFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=[128, 64, dict(pi=[32], vf=[32])]  # Ensure policy outputs 3D vector
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=os.path.join(MODEL_DIR, "checkpoints"),
        name_prefix="tempest_sac"
    )
    
    rl_model = SAC(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=0.0001,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=64,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        verbose=0,
        device=device
    )
    
    if os.path.exists(LATEST_MODEL_PATH):
        try:
            rl_model = SAC.load(LATEST_MODEL_PATH, env=env)
            print(f"Successfully loaded RL model from {LATEST_MODEL_PATH}")
        except Exception as e:
            print(f"ERROR loading RL model: {e}")
    
    # Apply minimal compatibility patches
    apply_minimal_compatibility_patches(rl_model)
    
    # Ensure buffer consistency once during initialization
    if hasattr(rl_model, 'replay_buffer') and rl_model.replay_buffer is not None:
        ensure_buffer_consistency(rl_model.replay_buffer)
    
    save_signal_callback = SaveOnSignalCallback(save_path=LATEST_MODEL_PATH, bc_model=bc_model)
    save_signal_callback.model = rl_model
    return env, bc_model, rl_model, save_signal_callback

def patch_critic_network(model):
    """
    Minimal patch for the critic network to handle array indexing issues.
    Only addresses the specific 'too many indices' error.
    """
    if model is None or not hasattr(model, 'critic') or not hasattr(model.critic, 'forward'):
        return False
    
    try:
        # Get the original forward method
        original_forward = model.critic.forward
        
        # Define a safe forward method that only reshapes on error
        def safe_forward(obs, actions):
            try:
                # Call the original forward method
                return original_forward(obs, actions)
            except IndexError as e:
                if "too many indices for array" in str(e):
                    print("Handling array indexing issue in critic network...")
                    # Only reshape if absolutely necessary and preserve batch dimension
                    if len(obs.shape) > 2:
                        obs = obs.reshape(obs.shape[0], -1)
                    if len(actions.shape) > 2:
                        actions = actions.reshape(actions.shape[0], -1)
                    # Try again with correctly shaped inputs
                    return original_forward(obs, actions)
                else:
                    raise
        
        # Replace the forward method
        model.critic.forward = safe_forward
        
        print("Applied minimal critic network patch")
        return True
    except Exception as e:
        print(f"Error patching critic network: {e}")
        return False

def apply_minimal_compatibility_patches(model):
    """
    Apply only the essential compatibility patches required for functionality.
    Focuses on ensuring correct shapes and handling the specific error.
    """
    try:
        # Fix missing logger (required by SB3)
        if not ('_logger' in model.__dict__ or hasattr(model, 'logger')):
            from stable_baselines3.common.logger import Logger
            model.__dict__['_logger'] = Logger(folder=None, output_formats=[])
        
        # Ensure action and observation spaces have correct dtype
        if hasattr(model, 'action_space') and isinstance(model.action_space, spaces.Box):
            model.action_space.low = model.action_space.low.astype(np.float32)
            model.action_space.high = model.action_space.high.astype(np.float32)
            
        if hasattr(model, 'observation_space') and isinstance(model.observation_space, spaces.Box):
            model.observation_space.low = model.observation_space.low.astype(np.float32)
            model.observation_space.high = model.observation_space.high.astype(np.float32)
        
        # Apply minimal critic network patch
        patch_critic_network(model)
        
        # Patch replay buffer's sample method to handle indexing errors
        if hasattr(model, 'replay_buffer') and hasattr(model.replay_buffer, 'sample'):
            original_sample = model.replay_buffer.sample
            
            def safe_sample(*args, **kwargs):
                try:
                    # Try original sample
                    return original_sample(*args, **kwargs)
                except IndexError as e:
                    if "too many indices for array" in str(e):
                        print("Fixing buffer before sampling...")
                        ensure_buffer_consistency(model.replay_buffer)
                        # Try again after fixing
                        return original_sample(*args, **kwargs)
                    else:
                        raise
            
            # Replace the sample method with the safe version
            model.replay_buffer.sample = safe_sample
        
        print("Applied minimal compatibility patches")
    except Exception as e:
        print(f"Warning: Failed to apply compatibility patches: {e}")

def encode_action(fire, zap, spinner_delta):
    """Convert discrete actions to Box action space format."""
    fire_float = float(fire)
    zap_float = float(zap)
    reduced_range = 64
    spinner_float = spinner_delta / reduced_range
    spinner_float = max(-1.0, min(1.0, spinner_float))
    return np.array([fire_float, zap_float, spinner_float], dtype=np.float32)

def decode_action(action):
    """Decode Box action space to discrete actions."""
    # Ensure action has shape (3,)
    action = normalize_action(action)
    fire = 1 if action[0] > 0.5 else 0
    zap = 1 if action[1] > 0.5 else 0
    reduced_range = 64
    spinner_delta = int(round(action[2] * reduced_range))
    spinner_delta = max(-reduced_range, min(reduced_range, spinner_delta))
    return fire, zap, spinner_delta

def safe_add_to_buffer(buffer, obs, next_obs, action, reward, done):
    """
    Safely add an experience to the replay buffer with consistent shapes.
    """
    if buffer is None:
        return False
    
    try:
        # Ensure consistent shapes
        obs = obs.reshape(243)
        next_obs = next_obs.reshape(243)
        action = normalize_action(action)
        
        # Ensure reward and done are scalars
        reward = float(reward) if hasattr(reward, 'item') else reward
        done = bool(done) if hasattr(done, 'item') else done
        
        # Add to buffer based on SB3 version
        if hasattr(buffer, 'handle_timeout_termination'):
            buffer.add(
                obs=obs,
                next_obs=next_obs,
                action=action,
                reward=reward,
                done=done,
                infos=[{}]
            )
        else:
            buffer.add(
                obs=obs,
                next_obs=next_obs,
                action=action,
                reward=reward,
                done=done
            )
        return True
    except Exception as e:
        print(f"Error adding to replay buffer: {e}")
        return False

def train_model_safely(model, gradient_steps, batch_size):
    """Train the model with error handling and minimal reshaping"""
    if model is None:
        return
    
    try:
        # Try training with the specified batch size
        model.train(gradient_steps=gradient_steps, batch_size=batch_size)
        
        # If training successful, log completion and buffer size
        buffer_size = get_buffer_size(model.replay_buffer)
        print(f"Model trained successfully. Buffer size: {buffer_size}")
        
    except Exception as e:
        error_msg = str(e)
        print(f"Training error: {error_msg}")
        
        # Handle dimension mismatch errors
        if ("doesn't match the broadcast shape" in error_msg or 
            "too many indices for array" in error_msg):
            
            # Try with reduced batch size as last resort
            try:
                reduced_batch_size = max(1, batch_size // 4)
                print(f"Retrying with batch_size={reduced_batch_size}")
                model.train(gradient_steps=1, batch_size=reduced_batch_size)
            except Exception as inner_e:
                print(f"Failed to train even with reduced parameters: {inner_e}")
        else:
            # Log other errors
            print(f"Unhandled training error: {e}")

# In main(), replace the train_model_background function with this simplified version
def train_model_background(model, gradient_steps, batch_size):
    """Simplified training function for background thread"""
    train_model_safely(model, gradient_steps, batch_size)

def main():
    global shutdown_requested
    print("Python AI model starting...")
    env, bc_model, rl_model, save_signal_callback = initialize_models()
    bc_episodes = 0
    rl_episodes = 0
    bc_losses = []
    last_mode_was_attract = True
    mode_transitions = 0
    
    # Create the pipes
    for pipe_path in [LUA_TO_PY_PIPE, PY_TO_LUA_PIPE]:
        if os.path.exists(pipe_path):
            os.unlink(pipe_path)
        os.mkfifo(pipe_path)
        os.chmod(pipe_path, 0o666)
    print("Pipes created successfully. Waiting for Lua connection...")

    
    # Connection retry loop
    while True:
        try:
            fd = os.open(LUA_TO_PY_PIPE, os.O_RDONLY | os.O_NONBLOCK)
            lua_to_py = os.fdopen(fd, "rb")
            py_to_lua = open(PY_TO_LUA_PIPE, "wb")

            # Add a debug message with a checkmark when Lua connects
            print("✔️ Lua has connected successfully.")

            try:
                frame_count = 0
                last_frame_time = time.time()
                
                # Training lock for background training
                training_lock = threading.Lock()
                is_training = False
                
                # Define a training function for background training
                def train_model_background(model, gradient_steps, batch_size):
                    """Simplified training function for background thread"""
                    train_model_safely(model, gradient_steps, batch_size)
                
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
                            continue
                        
                        processed_data, frame_counter, reward, game_action, is_attract, done, save_signal = result
                        
                        # Update the environment state
                        env.update_state(processed_data, reward, game_action, done)
                        env.is_attract_mode = is_attract
                        
                        # Detect mode transition
                        if is_attract != last_mode_was_attract:
                            mode_transitions += 1
                            last_mode_was_attract = is_attract
                        
                        # Process save signal
                        if save_signal:
                            save_signal_callback.signal_save()
                        
                        # Different behavior based on mode
                        if is_attract:
                            # Behavioral Cloning mode
                            if game_action is not None:
                                game_state = processed_data
                                fire_target, zap_target, spinner_delta = game_action
                                loss, bc_predicted_action = train_bc(bc_model, game_state, fire_target, zap_target, spinner_delta)
                                bc_losses.append(loss)
                                
                                # Add exploration occasionally
                                if random.random() < 0.2:
                                    spinner_noise = random.randint(-10, 10)
                                    new_spinner = max(-64, min(63, bc_predicted_action[2] + spinner_noise))
                                    bc_predicted_action = (bc_predicted_action[0], bc_predicted_action[1], new_spinner)
                                
                                game_action = bc_predicted_action
                                action = encode_action(*game_action)
                            else:
                                game_action = (0, 0, 0)
                                action = encode_action(*game_action)
                            
                            # Track BC episodes
                            if done:
                                bc_episodes += 1
                        else:
                            # Reinforcement Learning mode
                            try:
                                # Add batch dimension to processed_data
                                batched_data = np.expand_dims(processed_data, axis=0)
                                action_tensor, _ = rl_model.predict(batched_data, deterministic=False)
                                # Remove batch dimension from output
                                action = action_tensor.flatten() if hasattr(action_tensor, 'flatten') else action_tensor
                            except Exception as predict_error:
                                print(f"Error during prediction: {predict_error}")
                                # Fallback to BC model
                                game_state = torch.FloatTensor(processed_data).unsqueeze(0).to(device)
                                with torch.no_grad():
                                    bc_model = bc_model.to(device)
                                    fire_pred, zap_pred, spinner_pred = bc_model(game_state)
                                
                                fire = 1 if fire_pred.item() > 0.5 else 0
                                zap = 1 if zap_pred.item() > 0.5 else 0
                                spinner_delta = int(round(spinner_pred.item() * 64.0))
                                spinner_delta = max(-64, min(63, spinner_delta))
                                
                                game_action = (fire, zap, spinner_delta)
                                action = encode_action(*game_action)
                            
                            # Decode for game
                            fire, zap, spinner_delta = decode_action(action)
                            game_action = (fire, zap, spinner_delta)
                            
                            # Optionally use BC-guided exploration
                            epsilon = 0.1
                            if random.random() < epsilon * 6:
                                game_state = torch.FloatTensor(processed_data).unsqueeze(0).to(device)
                                with torch.no_grad():
                                    fire_pred, zap_pred, spinner_pred = bc_model(game_state)
                                fire = 1 if fire_pred.item() > 0.5 else 0
                                zap = 1 if zap_pred.item() > 0.5 else 0
                                spinner_delta = int(round(spinner_pred.item() * 64.0))
                                spinner_delta = max(-64, min(63, spinner_delta))
                                game_action = (fire, zap, spinner_delta)
                                action = encode_action(*game_action)
                            
                            # Add experience to replay buffer
                            if rl_model is not None and rl_model.replay_buffer is not None:
                                try:
                                    # Process the action to ensure correct shape
                                    processed_action = normalize_action(action, game_action)
                                    
                                    # Ensure states have the right shape
                                    prev_state = env.prev_state.reshape(env.observation_space.shape)
                                    curr_state = processed_data.reshape(env.observation_space.shape)
                                    
                                    # Ensure action has the right shape for the buffer (should be 1D)
                                    if len(processed_action.shape) != 1:
                                        processed_action = processed_action.flatten()
                                    
                                    # Add experience to replay buffer
                                    safe_add_to_buffer(rl_model.replay_buffer, prev_state, curr_state, processed_action, env.reward, done)
                                except Exception as e:
                                    print(f"Error adding to replay buffer: {e}")
                            
                            # Train SAC if we have enough experiences and not currently training
                            buffer_size = get_buffer_size(rl_model.replay_buffer)
                            if (buffer_size > rl_model.learning_starts and 
                                frame_count % 50 == 0 and not is_training):  # Reduced training frequency to every 50 frames
                                # Start training in a background thread
                                training_thread = threading.Thread(
                                    target=train_model_background, 
                                    args=(rl_model, 5, 64)  # Reduced gradient steps to 5
                                )
                                training_thread.daemon = True
                                training_thread.start()
                        
                        # Pack the action values into binary format
                        binary_action = struct.pack("bbb", 
                                                   int(game_action[0]),
                                                   int(game_action[1]),
                                                   int(game_action[2]))
                        
                        # Write the binary data to the pipe
                        py_to_lua.write(binary_action)
                        py_to_lua.flush()
                        
                        # Increment frame counter
                        frame_count += 1
                        
                    except BlockingIOError:
                        # Expected in non-blocking mode
                        time.sleep(0.01)
                    
                    except Exception as e:
                        print(f"Error during frame processing: {e}")
            
            finally:
                print("Pipe connection ended - Performing emergency save before exit")
                try:
                    rl_model.save(LATEST_MODEL_PATH)
                    torch.save(bc_model.state_dict(), BC_MODEL_PATH)
                except Exception as e:
                    print(f"Failed emergency save: {e}")
                
                lua_to_py.close()
                py_to_lua.close()
                print("Pipes closed, reconnecting...")
        
        except KeyboardInterrupt:
            print("Interrupted by user")
            try:
                save_signal_callback.signal_save()
            except Exception as e:
                print(f"Error during save: {e}")
            break
        
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)
    
    print("Python AI model shutting down")

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    main()