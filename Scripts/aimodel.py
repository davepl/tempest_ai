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
        spinner_delta = int(round(action[2] * 64.0))
        spinner_delta = max(-64, min(64, spinner_delta))
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
        return normalized_data, frame_counter, reward, (fire_commanded, zap_commanded, spinner_delta), is_attract, is_done, save_signal
    except Exception as e:
        print(f"Error processing frame data: {e}")
        return None, 0, 0.0, None, False, False, False

def train_bc(model, state, fire_target, zap_target, spinner_target):
    model = model.to(device)
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
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

def process_action_for_buffer(action, game_action=None):
    """Process action to ensure it has the correct shape for the replay buffer
    
    Args:
        action: The action to process
        game_action: Optional game action tuple (fire, zap, spinner) to use as reference
        
    Returns:
        Processed action with shape (3,)
    """
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
    
    # If action is a numpy array with wrong size
    if isinstance(action, np.ndarray):
        # Create a new array with the right shape
        result = np.zeros(3, dtype=np.float32)
        # Copy as many elements as possible
        size = min(action.size, 3)
        result[:size] = action.flatten()[:size]
        return result
    
    # Default: return zeros
    print(f"Warning: Could not process action {action}, returning zeros")
    return np.zeros(3, dtype=np.float32)

def ensure_correct_array_dimensions(buffer):
    """
    Ensure all arrays in the replay buffer have the correct dimensions.
    This helps prevent "too many indices for array" errors during training.
    """
    if buffer is None:
        return
    
    try:
        # Check if buffer has the necessary attributes
        required_attrs = ['observations', 'actions', 'next_observations', 'rewards', 'dones']
        for attr in required_attrs:
            if not hasattr(buffer, attr):
                print(f"Buffer missing attribute: {attr}")
                return
        
        buffer_size = len(buffer.observations)
        print(f"Checking replay buffer dimensions. Size: {buffer_size}")
        
        # Fix observations
        if len(buffer.observations.shape) != 2 or buffer.observations.shape[1] != 243:
            print(f"Fixing observations shape: {buffer.observations.shape}")
            fixed_obs = np.zeros((buffer_size, 243), dtype=np.float32)
            for i in range(buffer_size):
                if i < len(buffer.observations):
                    obs = buffer.observations[i]
                    if obs.size == 243:
                        fixed_obs[i] = obs.flatten()
                    else:
                        # Pad or truncate
                        fixed_obs[i, :min(obs.size, 243)] = obs.flatten()[:min(obs.size, 243)]
            buffer.observations = fixed_obs
        
        # Fix next_observations
        if len(buffer.next_observations.shape) != 2 or buffer.next_observations.shape[1] != 243:
            print(f"Fixing next_observations shape: {buffer.next_observations.shape}")
            fixed_next_obs = np.zeros((buffer_size, 243), dtype=np.float32)
            for i in range(buffer_size):
                if i < len(buffer.next_observations):
                    next_obs = buffer.next_observations[i]
                    if next_obs.size == 243:
                        fixed_next_obs[i] = next_obs.flatten()
                    else:
                        # Pad or truncate
                        fixed_next_obs[i, :min(next_obs.size, 243)] = next_obs.flatten()[:min(next_obs.size, 243)]
            buffer.next_observations = fixed_next_obs
        
        # Fix actions
        if len(buffer.actions.shape) != 2 or buffer.actions.shape[1] != 3:
            print(f"Fixing actions shape: {buffer.actions.shape}")
            fixed_actions = np.zeros((buffer_size, 3), dtype=np.float32)
            for i in range(buffer_size):
                action = buffer.actions[i]  # Define action variable here
                if action.size == 3:
                    fixed_actions[i] = action.flatten()
                else:
                    # Pad or truncate
                    fixed_actions[i, :min(action.size, 3)] = action.flatten()[:min(action.size, 3)]
            buffer.actions = fixed_actions
        
        # Fix rewards
        if len(buffer.rewards.shape) != 1 or buffer.rewards.shape[0] != buffer_size:
            print(f"Fixing rewards shape: {buffer.rewards.shape}")
            fixed_rewards = np.zeros(buffer_size, dtype=np.float32)
            for i in range(min(buffer_size, len(buffer.rewards))):
                # Ensure we're getting a scalar value
                reward_value = buffer.rewards[i]
                if hasattr(reward_value, 'item'):
                    reward_value = reward_value.item()
                elif hasattr(reward_value, 'size') and reward_value.size > 1:
                    reward_value = reward_value.flatten()[0]
                fixed_rewards[i] = reward_value
            buffer.rewards = fixed_rewards
        
        # Fix dones
        if len(buffer.dones.shape) != 1 or buffer.dones.shape[0] != buffer_size:
            print(f"Fixing dones shape: {buffer.dones.shape}")
            fixed_dones = np.zeros(buffer_size, dtype=np.float32)
            for i in range(min(buffer_size, len(buffer.dones))):
                # Ensure we're getting a scalar value
                done_value = buffer.dones[i]
                if hasattr(done_value, 'item'):
                    done_value = done_value.item()
                elif hasattr(done_value, 'size') and done_value.size > 1:
                    done_value = done_value.flatten()[0]
                fixed_dones[i] = done_value
            buffer.dones = fixed_dones
        
        print("Replay buffer dimensions fixed")
    except Exception as e:
        print(f"Error fixing buffer dimensions: {e}")

def fix_replay_buffer_shapes(buffer):
    """Fix inconsistent action shapes in the replay buffer"""
    if buffer is None or not hasattr(buffer, 'actions') or len(buffer.actions) == 0:
        return
    
    # First ensure all arrays have correct dimensions
    ensure_correct_array_dimensions(buffer)
    
    # Check if we need to fix shapes
    need_fixing = False
    expected_shape = (3,)  # Expected shape for our action space
    
    # Sample some actions to check their shapes
    sample_indices = np.random.choice(len(buffer.actions), min(100, len(buffer.actions)), replace=False)
    for idx in sample_indices:
        action = buffer.actions[idx]
        if len(action.shape) != 1 or action.shape[0] != 3:
            need_fixing = True
            break
    
    if need_fixing:
        print(f"Fixing replay buffer action shapes. Buffer size: {len(buffer.actions)}")
        fixed_actions = []
        
        for i in range(len(buffer.actions)):
            action = buffer.actions[i]
            
            # Process the action to ensure correct shape
            if len(action.shape) != 1 or action.shape[0] != 3:
                if action.size == 3:
                    # Reshape to (3,)
                    fixed_action = action.flatten()
                elif action.size > 3:
                    # Take first 3 elements
                    fixed_action = action.flatten()[:3]
                else:
                    # Pad with zeros
                    fixed_action = np.zeros(3, dtype=np.float32)
                    fixed_action[:action.size] = action.flatten()
            else:
                fixed_action = action
            
            fixed_actions.append(fixed_action)
        
        # Replace the actions in the buffer
        buffer.actions = np.array(fixed_actions)
        print(f"Fixed {len(fixed_actions)} actions in replay buffer")

def convert_buffer_to_compatible_format(model):
    """
    Convert the replay buffer to a compatible format if needed.
    This is useful when loading a model trained with a different version of SB3.
    """
    if model is None or not hasattr(model, 'replay_buffer'):
        return
    
    buffer = model.replay_buffer
    
    try:
        # Check if we need to convert the buffer
        if not hasattr(buffer, 'observations') or not hasattr(buffer, 'actions'):
            print("Replay buffer format not recognized, skipping conversion")
            return
        
        # Get buffer size
        buffer_size = get_buffer_size(buffer)
        if buffer_size == 0:
            print("Empty replay buffer, nothing to convert")
            return
        
        print(f"Converting replay buffer with {buffer_size} experiences to compatible format")
        
        # Create a new buffer with the same parameters
        from stable_baselines3.common.buffers import ReplayBuffer
        
        # Get the observation and action space from the model
        observation_space = model.observation_space
        action_space = model.action_space
        
        # Create a new buffer
        new_buffer = ReplayBuffer(
            buffer_size=buffer.buffer_size if hasattr(buffer, 'buffer_size') else buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=model.device if hasattr(model, 'device') else 'cpu',
            n_envs=1,
            optimize_memory_usage=False
        )
        
        # Copy data from old buffer to new buffer
        for i in range(buffer_size):
            # Get data from old buffer
            obs = buffer.observations[i].reshape(observation_space.shape)
            next_obs = buffer.next_observations[i].reshape(observation_space.shape)
            
            # Process action to ensure correct shape
            action = buffer.actions[i]
            if len(action.shape) != 1 or action.shape[0] != action_space.shape[0]:
                action = action.flatten()[:action_space.shape[0]]
            
            # Get reward and done
            reward = buffer.rewards[i]
            if hasattr(reward, 'item'):
                reward = reward.item()
            
            done = buffer.dones[i]
            if hasattr(done, 'item'):
                done = done.item()
            
            # Add to new buffer
            safe_add_to_buffer(new_buffer, obs, next_obs, action, reward, done)
        
        # Replace old buffer with new buffer
        model.replay_buffer = new_buffer
        print("Replay buffer successfully converted")
        
    except Exception as e:
        print(f"Error converting replay buffer: {e}")

def create_clean_replay_buffer(model):
    """
    Create a new clean replay buffer and safely transfer data from the old one.
    This is a more aggressive approach to fixing replay buffer issues.
    """
    if model is None or not hasattr(model, 'replay_buffer'):
        return
    
    try:
        old_buffer = model.replay_buffer
        buffer_size = get_buffer_size(old_buffer)
        
        if buffer_size == 0:
            print("Empty replay buffer, no need to create a new one")
            return
        
        print(f"Creating a new clean replay buffer and transferring {buffer_size} experiences")
        
        # Create a new buffer with the same parameters
        from stable_baselines3.common.buffers import ReplayBuffer
        
        # Get the observation and action space from the model
        observation_space = model.observation_space
        action_space = model.action_space
        
        # Create a new buffer with a smaller size to avoid memory issues
        new_buffer_size = min(buffer_size, 10000)  # Limit to 10,000 experiences
        
        new_buffer = ReplayBuffer(
            buffer_size=new_buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=model.device if hasattr(model, 'device') else 'cpu',
            n_envs=1,
            optimize_memory_usage=False
        )
        
        # Sample a subset of experiences to transfer
        transfer_size = min(buffer_size, new_buffer_size)
        indices = np.random.choice(buffer_size, transfer_size, replace=False)
        
        # Transfer experiences one by one
        transfer_count = 0
        for i in indices:
            try:
                # Get data from old buffer
                if not hasattr(old_buffer, 'observations') or i >= len(old_buffer.observations):
                    continue
                
                obs = old_buffer.observations[i]
                if obs.shape != observation_space.shape:
                    obs = obs.reshape(observation_space.shape)
                
                if not hasattr(old_buffer, 'next_observations') or i >= len(old_buffer.next_observations):
                    continue
                
                next_obs = old_buffer.next_observations[i]
                if next_obs.shape != observation_space.shape:
                    next_obs = next_obs.reshape(observation_space.shape)
                
                if not hasattr(old_buffer, 'actions') or i >= len(old_buffer.actions):
                    continue
                
                action = old_buffer.actions[i]
                if len(action.shape) != 1 or action.shape[0] != action_space.shape[0]:
                    action = action.flatten()[:action_space.shape[0]]
                
                if not hasattr(old_buffer, 'rewards') or i >= len(old_buffer.rewards):
                    continue
                
                reward = old_buffer.rewards[i]
                if hasattr(reward, 'item'):
                    reward = reward.item()
                elif hasattr(reward, 'size') and reward.size > 1:
                    reward = float(reward.flatten()[0])
                else:
                    reward = float(reward)
                
                if not hasattr(old_buffer, 'dones') or i >= len(old_buffer.dones):
                    continue
                
                done = old_buffer.dones[i]
                if hasattr(done, 'item'):
                    done = done.item()
                elif hasattr(done, 'size') and done.size > 1:
                    done = bool(done.flatten()[0])
                else:
                    done = bool(done)
                
                # Add to new buffer
                if safe_add_to_buffer(new_buffer, obs, next_obs, action, reward, done):
                    transfer_count += 1
            except Exception as e:
                print(f"Error transferring experience {i}: {e}")
        
        print(f"Successfully transferred {transfer_count} experiences to new buffer")
        
        # Replace old buffer with new buffer
        model.replay_buffer = new_buffer
        
        return True
    except Exception as e:
        print(f"Error creating clean replay buffer: {e}")
        return False

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
                        fixed_actions[i] = process_action_for_buffer(action).flatten()[:3]
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
    
    # Simplify policy_kwargs to avoid empty() tensor error
    # Use a simpler network architecture that's more compatible
    policy_kwargs = dict(
        features_extractor_class=TempestFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=[128, 64]  # Simplified architecture
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=os.path.join(MODEL_DIR, "checkpoints"),
        name_prefix="tempest_sac"
    )
    
    # Try-except to catch and diagnose any initialization errors
    try:
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
            verbose=1,  # Increased verbosity for debugging
            device=device
        )
        print("Successfully created SAC model")
    except Exception as e:
        print(f"ERROR initializing SAC model: {e}")
        # Fallback to even simpler initialization if the above fails
        try:
            print("Trying fallback initialization...")
            policy_kwargs = dict(
                features_extractor_class=TempestFeaturesExtractor,
                features_extractor_kwargs=dict(features_dim=128)
            )
            rl_model = SAC(
                "MlpPolicy",
                env,
                policy_kwargs=policy_kwargs,
                verbose=1,
                device=device
            )
            print("Successfully created SAC model with fallback options")
        except Exception as e2:
            print(f"ERROR even with fallback initialization: {e2}")
            # Create a minimal SAC model as a last resort
            rl_model = SAC("MlpPolicy", env)
            print("Created minimal SAC model")
    
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
        ensure_correct_array_dimensions(rl_model.replay_buffer)
    
    save_signal_callback = SaveOnSignalCallback(save_path=LATEST_MODEL_PATH, bc_model=bc_model)
    save_signal_callback.model = rl_model
    return env, bc_model, rl_model, save_signal_callback

def patch_critic_network(model):
    """
    Patch the critic network to handle array indexing issues.
    This is a direct fix for the "too many indices for array" error.
    """
    if model is None or not hasattr(model, 'critic') or not hasattr(model.critic, 'forward'):
        return False
    
    try:
        # Get the original forward method
        original_forward = model.critic.forward
        
        # Define a safe forward method
        def safe_forward(obs, actions):
            try:
                # Ensure inputs have the right shape
                if len(obs.shape) > 2:
                    obs = obs.reshape(obs.shape[0], -1)
                if len(actions.shape) > 2:
                    actions = actions.reshape(actions.shape[0], -1)
                
                # Call the original forward method
                return original_forward(obs, actions)
            except IndexError as e:
                if "too many indices for array" in str(e):
                    print("Handling array indexing issue in critic network...")
                    # Try to reshape the inputs more aggressively
                    if len(obs.shape) > 2:
                        obs = obs.reshape(-1, model.observation_space.shape[0])
                    if len(actions.shape) > 2:
                        actions = actions.reshape(-1, model.action_space.shape[0])
                    
                    # Try again with reshaped inputs
                    return original_forward(obs, actions)
                else:
                    raise
        
        # Replace the forward method
        model.critic.forward = safe_forward
        
        # Also patch the qf1 and qf2 networks if they exist
        for qf_name in ['qf1', 'qf2']:
            if hasattr(model.critic, qf_name) and hasattr(getattr(model.critic, qf_name), 'forward'):
                qf = getattr(model.critic, qf_name)
                original_qf_forward = qf.forward
                
                def safe_qf_forward(x):
                    try:
                        # Ensure input has the right shape
                        if len(x.shape) > 2:
                            x = x.reshape(x.shape[0], -1)
                        
                        # Call the original forward method
                        return original_qf_forward(x)
                    except IndexError as e:
                        if "too many indices for array" in str(e):
                            print(f"Handling array indexing issue in {qf_name}...")
                            # Try to reshape the input more aggressively
                            x = x.reshape(-1, x.shape[-1])
                            
                            # Try again with reshaped input
                            return original_qf_forward(x)
                        else:
                            raise
                
                # Replace the forward method
                qf.forward = safe_qf_forward
        
        print("Successfully patched critic network")
        return True
    except Exception as e:
        print(f"Error patching critic network: {e}")
        return False

def apply_sb3_compatibility_patches(model):
    """Apply compatibility patches for different versions of Stable Baselines3"""
    try:
        # Check if the model has a logger
        has_logger = '_logger' in model.__dict__ or hasattr(model, 'logger')
        
        if not has_logger:
            # Create a minimal logger implementation
            from stable_baselines3.common.logger import Logger
            dummy_logger = Logger(folder=None, output_formats=[])
            
            # Add logger attribute to model
            model.__dict__['_logger'] = dummy_logger
            print("Applied logger compatibility patch")
        
        # Ensure action space has correct dtype
        if hasattr(model, 'action_space'):
            if isinstance(model.action_space, spaces.Box):
                # Ensure low and high have correct dtype
                if model.action_space.low.dtype != np.float32:
                    model.action_space.low = model.action_space.low.astype(np.float32)
                if model.action_space.high.dtype != np.float32:
                    model.action_space.high = model.action_space.high.astype(np.float32)
                print("Fixed action space dtype")
        
        # Ensure observation space has correct dtype
        if hasattr(model, 'observation_space'):
            if isinstance(model.observation_space, spaces.Box):
                # Ensure low and high have correct dtype
                if model.observation_space.low.dtype != np.float32:
                    model.observation_space.low = model.observation_space.low.astype(np.float32)
                if model.observation_space.high.dtype != np.float32:
                    model.observation_space.high = model.observation_space.high.astype(np.float32)
                print("Fixed observation space dtype")
        
        # Patch the critic network to handle array indexing issues
        patch_critic_network(model)
        
        # Monkey patch the train method to handle missing logger
        original_train = model.train
        
        def safe_train(*args, **kwargs):
            try:
                return original_train(*args, **kwargs)
            except AttributeError as e:
                if "'SAC' object has no attribute '_logger'" in str(e):
                    # Add logger if it's missing
                    from stable_baselines3.common.logger import Logger
                    model.__dict__['_logger'] = Logger(folder=None, output_formats=[])
                    # Try again with the logger added
                    return original_train(*args, **kwargs)
                else:
                    raise
        
        # Replace the train method with our safe version
        model.train = safe_train
        
        # Fix for _update_learning_rate method
        if hasattr(model, '_update_learning_rate'):
            original_update_lr = model._update_learning_rate
            
            def safe_update_lr(optimizer):
                try:
                    return original_update_lr(optimizer)
                except AttributeError as e:
                    if "'SAC' object has no attribute '_logger'" in str(e):
                        # Add logger if it's missing
                        from stable_baselines3.common.logger import Logger
                        model.__dict__['_logger'] = Logger(folder=None, output_formats=[])
                        return original_update_lr(optimizer)
                    else:
                        # Fallback implementation if original fails
                        if hasattr(model, 'lr_schedule') and callable(model.lr_schedule):
                            new_lr = model.lr_schedule(model._current_progress_remaining 
                                                     if hasattr(model, '_current_progress_remaining') else 1.0)
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = new_lr
                        return
            
            # Replace the method with our safe version
            model._update_learning_rate = safe_update_lr
        
        # Patch the replay buffer's sample method to handle array indexing issues
        if hasattr(model, 'replay_buffer') and hasattr(model.replay_buffer, 'sample'):
            original_sample = model.replay_buffer.sample
            
            def safe_sample(*args, **kwargs):
                try:
                    return original_sample(*args, **kwargs)
                except IndexError as e:
                    if "too many indices for array" in str(e):
                        print("Fixing replay buffer before sampling...")
                        fix_replay_buffer_shapes(model.replay_buffer)
                        # Try again after fixing
                        return original_sample(*args, **kwargs)
                    else:
                        raise
            
            # Replace the sample method with our safe version
            model.replay_buffer.sample = safe_sample
        
        # Patch the _sample_action method to handle array indexing issues
        if hasattr(model, 'policy') and hasattr(model.policy, '_sample_action'):
            original_sample_action = model.policy._sample_action
            
            def safe_sample_action(*args, **kwargs):
                try:
                    return original_sample_action(*args, **kwargs)
                except IndexError as e:
                    if "too many indices for array" in str(e):
                        print("Handling array indexing issue in _sample_action...")
                        # Get the arguments
                        mean_actions = args[0]
                        log_std = args[1]
                        
                        # Ensure they have the right shape
                        if len(mean_actions.shape) > 2:
                            mean_actions = mean_actions.reshape(mean_actions.shape[0], -1)
                        if len(log_std.shape) > 2:
                            log_std = log_std.reshape(log_std.shape[0], -1)
                        
                        # Call the original function with fixed arguments
                        return original_sample_action(mean_actions, log_std, *args[2:], **kwargs)
                    else:
                        raise
            
            # Replace the _sample_action method with our safe version
            model.policy._sample_action = safe_sample_action
        
        print("Applied SB3 compatibility patches")
    except Exception as e:
        print(f"Warning: Failed to apply compatibility patches: {e}")

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
            print("Applied logger compatibility patch")
        
        # Ensure action and observation spaces have correct dtype
        if hasattr(model, 'action_space') and isinstance(model.action_space, spaces.Box):
            model.action_space.low = model.action_space.low.astype(np.float32)
            model.action_space.high = model.action_space.high.astype(np.float32)
        
        if hasattr(model, 'observation_space') and isinstance(model.observation_space, spaces.Box):
            model.observation_space.low = model.observation_space.low.astype(np.float32)
            model.observation_space.high = model.observation_space.high.astype(np.float32)
        
        # Apply minimal critic network patch
        if hasattr(model, 'critic') and hasattr(model.critic, 'forward'):
            original_forward = model.critic.forward
            
            def safe_forward(obs, actions):
                try:
                    return original_forward(obs, actions)
                except IndexError as e:
                    if "too many indices for array" in str(e):
                        print("Handling array indexing issue in critic network...")
                        if len(obs.shape) > 2:
                            obs = obs.reshape(obs.shape[0], -1)
                        if len(actions.shape) > 2:
                            actions = actions.reshape(actions.shape[0], -1)
                        return original_forward(obs, actions)
                    else:
                        raise
            
            model.critic.forward = safe_forward
        
        # Patch replay buffer's sample method for "too many indices" error
        if hasattr(model, 'replay_buffer') and hasattr(model.replay_buffer, 'sample'):
            original_sample = model.replay_buffer.sample
            
            def safe_sample(*args, **kwargs):
                try:
                    return original_sample(*args, **kwargs)
                except IndexError as e:
                    if "too many indices for array" in str(e):
                        print("Fixing buffer before sampling...")
                        if hasattr(model, 'replay_buffer') and model.replay_buffer is not None:
                            ensure_correct_array_dimensions(model.replay_buffer)
                        return original_sample(*args, **kwargs)
                    else:
                        raise
            
            model.replay_buffer.sample = safe_sample
        
        print("Applied minimal compatibility patches")
    except Exception as e:
        print(f"Warning: Failed to apply minimal compatibility patches: {e}")

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
                    """Train the model in the background with error handling"""
                    if model is None:
                        return
                    
                    try:
                        # Fix replay buffer shapes before training
                        if hasattr(model, 'replay_buffer'):
                            fix_replay_buffer_shapes(model.replay_buffer)
                        
                        # Check if model has _logger attribute
                        if not hasattr(model, '_logger'):
                            try:
                                # Try to add a logger
                                from stable_baselines3.common.logger import Logger
                                model.__dict__['_logger'] = Logger(folder=None, output_formats=[])
                                print("Added missing logger to model")
                            except Exception as e:
                                print(f"Error adding logger: {e}")
                        
                        # Try training with the specified batch size
                        try:
                            with np.errstate(all='raise'):  # This will raise exceptions for numpy warnings
                                model.train(gradient_steps=gradient_steps, batch_size=batch_size)
                        except (RuntimeError, IndexError, FloatingPointError, DeprecationWarning, FutureWarning) as e:
                            error_msg = str(e)
                            print(f"Training error: {error_msg}")
                            
                            # Handle various error types
                            if ("doesn't match the broadcast shape" in error_msg or 
                                "too many indices for array" in error_msg or
                                "Conversion of an array with ndim > 0 to a scalar is deprecated" in error_msg):
                                
                                # Try creating a clean replay buffer as a more aggressive fix
                                print("Attempting to create a clean replay buffer to fix training issues...")
                                if create_clean_replay_buffer(model):
                                    print("Successfully created a clean replay buffer, retrying training...")
                                    try:
                                        # Try training with reduced parameters
                                        model.train(gradient_steps=1, batch_size=16)
                                        return
                                    except Exception as clean_buffer_error:
                                        print(f"Still having issues after creating clean buffer: {clean_buffer_error}")
                                
                                # If clean buffer approach fails, try with a smaller batch size
                                reduced_batch_size = max(1, batch_size // 4)  # More aggressive reduction
                                print(f"Shape/indexing/deprecation error. Retrying with batch_size={reduced_batch_size}")
                                
                                # Fix replay buffer again to be sure
                                if hasattr(model, 'replay_buffer'):
                                    ensure_correct_array_dimensions(model.replay_buffer)
                                    fix_replay_buffer_shapes(model.replay_buffer)
                                
                                try:
                                    # Try training with reduced batch size and gradient steps
                                    model.train(gradient_steps=1, batch_size=reduced_batch_size)
                                except Exception as inner_e:
                                    print(f"Still having issues with reduced batch size: {inner_e}")
                                    # Try with minimal parameters as a last resort
                                    try:
                                        model.train(gradient_steps=1, batch_size=1)
                                    except Exception as final_e:
                                        print(f"Failed even with minimal training parameters: {final_e}")
                                        print("Skipping training for this frame to avoid further errors")
                            else:
                                # Re-raise other runtime errors
                                raise
                    except Exception as e:
                        print(f"Error during training: {e}")
                        # Reset training state if needed
                        if hasattr(model, 'policy') and hasattr(model.policy, 'set_training_mode'):
                            model.policy.set_training_mode(False)
                
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
                                    processed_action = process_action_for_buffer(action, game_action)
                                    
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
    fire = 1 if action[0] > 0.5 else 0
    zap = 1 if action[1] > 0.5 else 0
    reduced_range = 64
    spinner_delta = int(round(action[2] * reduced_range))
    spinner_delta = max(-reduced_range, min(reduced_range, spinner_delta))
    return fire, zap, spinner_delta

def safe_add_to_buffer(buffer, obs, next_obs, action, reward, done):
    """
    Safely add an experience to the replay buffer, handling any potential errors.
    
    Args:
        buffer: The replay buffer to add the experience to
        obs: The observation
        next_obs: The next observation
        action: The action
        reward: The reward
        done: Whether the episode is done
    """
    if buffer is None:
        return False
    
    try:
        # Ensure action has the right shape (should be 1D)
        if len(action.shape) != 1:
            action = action.flatten()
        
        # Ensure reward is a scalar
        if hasattr(reward, 'item'):
            reward = reward.item()
        elif hasattr(reward, 'size') and reward.size > 1:
            reward = reward.flatten()[0]
        
        # Ensure done is a scalar
        if hasattr(done, 'item'):
            done = done.item()
        elif hasattr(done, 'size') and done.size > 1:
            done = done.flatten()[0]
        
        # Check if the buffer has the handle_timeout_termination attribute
        # This is present in newer versions of SB3
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
            # Older versions of SB3 don't have the infos parameter
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

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    main()