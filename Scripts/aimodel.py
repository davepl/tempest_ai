#!/usr/bin/env python3
"""
Tempest AI Model: Combines Behavioral Cloning (BC) in attract mode with Reinforcement Learning (RL) in gameplay.
- BC learns from game AI demonstrations; RL optimizes during player control.
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
import threading
import queue
from collections import deque
import traceback
import select
import sys
import termios
import fcntl

# Constants

ShouldReplayLog = False
DEBUG_MODE = False  # Set to False in production for better performance
FORCE_CPU = False  # Force CPU usage if having persistent issues with MPS
SPINNER_POWER = 1.0  # Linear spinner movement (1.0 = linear, higher = more small movements)
initial_exploration_ratio = 0.60
min_exploration_ratio = 0.05  # Lowest exploration rate
exploration_decay = 0.99  # Decay rate
BC_GUIDED_EXPLORATION_RATIO = 1.0

LogFile = "/Users/dave/mame/big.log"
MaxLogFrames = 1000000

# Target optimal spinner value from reward function - must match Lua
FIRE_EXPLORATION_RATIO = 0.05

# Ensure this matches exactly what is sent from Lua
NumberOfParams = 128  # Confirm this is correct based on Lua data serialization
LUA_TO_PY_PIPE = "/tmp/lua_to_py"
PY_TO_LUA_PIPE = "/tmp/py_to_lua"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
LATEST_MODEL_PATH = f"{MODEL_DIR}/tempest_model_latest.zip"
BC_MODEL_PATH = f"{MODEL_DIR}/tempest_bc_model.pt"

# Training metrics tracking
TRACK_BATCH_METRICS = True  # Set to False to disable detailed batch metrics for performance

# Parse command line arguments
parser = argparse.ArgumentParser(description='Tempest AI Model')
parser.add_argument('--replay', type=str, help='Path to a log file to replay for BC training')
args = parser.parse_args()

# Device selection: CUDA > MPS > CPU
if FORCE_CPU:
    device = torch.device("cpu")
    print("Using device: CPU (forced)")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device.type.upper()}")

# Training metrics buffers and synchronization
actor_losses = deque(maxlen=100)
bc_training_queue = queue.Queue(maxsize=1000)
bc_model_lock = threading.Lock()
replay_buffer = deque(maxlen=100000)
episode_rewards = deque(maxlen=5)

# Tracking metrics for debugging training progress
fire_probs_before = deque(maxlen=20)
fire_probs_after = deque(maxlen=20)
zap_probs_before = deque(maxlen=20)
zap_probs_after = deque(maxlen=20)
spinner_means_before = deque(maxlen=20)
spinner_means_after = deque(maxlen=20)
spinner_vars_before = deque(maxlen=20)
spinner_vars_after = deque(maxlen=20)
value_errors_before = deque(maxlen=20)
value_errors_after = deque(maxlen=20)

# Control source tracking
random_control_count = 0
bc_control_count = 0
rl_control_count = 0
total_control_count = 0

# BC training metrics
bc_fire_accuracy_before = deque(maxlen=20)
bc_fire_accuracy_after = deque(maxlen=20)
bc_zap_accuracy_before = deque(maxlen=20)
bc_zap_accuracy_after = deque(maxlen=20)
bc_spinner_error_before = deque(maxlen=20)
bc_spinner_error_after = deque(maxlen=20)
bc_train_count = 0

class BCModel(nn.Module):
    """Enhanced BC model with clipped linear spinner output."""
    def __init__(self, input_size=NumberOfParams):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 1024),  # Increased from 512
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024),  # Added BatchNorm for better training stability
            nn.Linear(1024, 512),  # Increased from 256
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),  # Added BatchNorm
            nn.Linear(512, 256),   # Increased from 128
            nn.LeakyReLU(),
            nn.Linear(256, 128),   # Added an extra layer
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )
        self.fire_output = nn.Linear(128, 1)
        self.zap_output = nn.Linear(128, 1)
        self.spinner_output = nn.Linear(128, 1)
        self.spinner_var_output = nn.Linear(128, 1)  # New output for spinner variance
        
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
            
        # Handle batch normalization manually with special case for batch size of 1
        # First linear + ReLU
        x = self.feature_extractor[0](x)  # Linear
        x = self.feature_extractor[1](x)  # LeakyReLU
        
        # First BatchNorm
        if x.size(0) == 1:
            # Skip BatchNorm for single sample
            pass
        else:
            x = self.feature_extractor[2](x)  # BatchNorm
        
        # Second linear + ReLU
        x = self.feature_extractor[3](x)  # Linear
        x = self.feature_extractor[4](x)  # LeakyReLU
        
        # Second BatchNorm
        if x.size(0) == 1:
            # Skip BatchNorm for single sample
            pass
        else:
            x = self.feature_extractor[5](x)  # BatchNorm
        
        # Third linear + ReLU
        x = self.feature_extractor[6](x)  # Linear
        x = self.feature_extractor[7](x)  # LeakyReLU
        
        # Fourth linear + ReLU + Dropout
        x = self.feature_extractor[8](x)  # Linear
        x = self.feature_extractor[9](x)  # LeakyReLU
        x = self.feature_extractor[10](x)  # Dropout
        
        # Output heads
        fire_out = torch.sigmoid(self.fire_output(x))
        zap_out = torch.sigmoid(self.zap_output(x))
        spinner_out = torch.tanh(self.spinner_output(x))
        spinner_var = torch.clamp(F.softplus(self.spinner_var_output(x)), 0.01, 0.5)
        
        return torch.cat([fire_out, zap_out, spinner_out, spinner_var], dim=1), spinner_out

class Actor(nn.Module):
    def __init__(self, state_dim=NumberOfParams):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 512)  # Increased from 256
        self.bn1 = nn.BatchNorm1d(512)  # Added batch normalization
        self.fc2 = nn.Linear(512, 256)  # Increased from 256 and added an extra layer
        self.bn2 = nn.BatchNorm1d(256)  # Added batch normalization
        self.fc3 = nn.Linear(256, 256)  # New layer 
        self.fire_head = nn.Linear(256, 1)
        self.zap_head = nn.Linear(256, 1)
        self.spinner_head = nn.Linear(256, 1)
        self.spinner_var_head = nn.Linear(256, 1)  # New head for spinner variance
        
        # Initialize parameters with smaller values to reduce extreme outputs
        torch.nn.init.xavier_uniform_(self.spinner_head.weight, gain=0.1)
        torch.nn.init.xavier_uniform_(self.spinner_var_head.weight, gain=0.1)
        if self.spinner_head.bias is not None:
            torch.nn.init.zeros_(self.spinner_head.bias)
        if self.spinner_var_head.bias is not None:
            torch.nn.init.constant_(self.spinner_var_head.bias, -1.0)  # Start with low variance
            
        self.to(device)

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        # Handle batch size of 1 for BatchNorm1d
        if x.size(0) == 1:
            x1 = x  # Skip batch norm for single sample
        else:
            x1 = self.bn1(x)
        
        x = self.fc2(x1)
        x = F.relu(x)
        # Handle batch size of 1 for BatchNorm1d
        if x.size(0) == 1:
            x2 = x  # Skip batch norm for single sample
        else:
            x2 = self.bn2(x)
        
        x = F.relu(self.fc3(x2))
        
        fire_logits = torch.clamp(self.fire_head(x), -10.0, 10.0)  # Prevent extreme logits
        zap_logits = torch.clamp(self.zap_head(x), -10.0, 10.0)
        
        # Get raw spinner value
        raw_spinner = self.spinner_head(x)
        
        # Create a bell curve preference centered at the normalized optimal value
        # This aligns the network's preference with the reward function's bell curve
        
        normalized_optimal = 0
        
        # Apply a transformation that creates a bell curve with peak at normalized_optimal
        # We'll use a Gaussian-like penalty: exp(-(x-target)²/variance)
        distance_from_optimal = raw_spinner - normalized_optimal
        bell_curve_factor = torch.exp(-4.0 * distance_from_optimal * distance_from_optimal)
        
        # Add penalty for extreme values and boost for values near optimal
        spinner_mean = torch.tanh(raw_spinner * bell_curve_factor * 2.0)
        
        # Keep variance in reasonable range with sigmoid activation
        # Lower variance near optimal value, higher at extremes
        raw_var = self.spinner_var_head(x)
        spinner_var = torch.sigmoid(raw_var) * 0.15 
        
        return torch.cat([fire_logits, zap_logits, spinner_mean, spinner_var], dim=-1)

    def get_action(self, state, deterministic=False):
        logits = self.forward(state)
        fire_logits, zap_logits, spinner_mean, spinner_var = torch.split(logits, 1, dim=-1)
        
        if deterministic:
            fire = torch.sigmoid(fire_logits) > 0.5
            zap = torch.sigmoid(zap_logits) > 0.5
            # Less noise in deterministic mode to better follow the learned policy
            spinner_noise = torch.randn_like(spinner_mean) * spinner_var * 0.3
        else:
            fire = torch.bernoulli(torch.sigmoid(fire_logits))
            zap = torch.bernoulli(torch.sigmoid(zap_logits))
            # Full noise in exploration mode
            spinner_noise = torch.randn_like(spinner_mean) * spinner_var
        
        # Combine mean and noise, then clamp
        spinner = torch.clamp(spinner_mean + spinner_noise, -0.95, 0.95)
        
        # Ensure spinner is finite
        spinner = torch.nan_to_num(spinner, nan=0.0, posinf=0.5, neginf=-0.5)
        return torch.cat([fire, zap, spinner], dim=-1)

class Critic(nn.Module):
    def __init__(self, state_dim=NumberOfParams):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 512)  # Increased from 256
        self.bn1 = nn.BatchNorm1d(512)  # Added batch normalization
        self.fc2 = nn.Linear(512, 256)  # Increased from 256
        self.bn2 = nn.BatchNorm1d(256)  # Added batch normalization
        self.fc3 = nn.Linear(256, 256)  # New layer
        self.value_head = nn.Linear(256, 1)
        self.to(device)

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        # Handle batch size of 1 for BatchNorm1d
        if x.size(0) == 1:
            x1 = x  # Skip batch norm for single sample
        else:
            x1 = self.bn1(x)
        
        x = self.fc2(x1)
        x = F.relu(x)
        # Handle batch size of 1 for BatchNorm1d
        if x.size(0) == 1:
            x2 = x  # Skip batch norm for single sample
        else:
            x2 = self.bn2(x)
        
        x = F.relu(self.fc3(x2))
        
        return self.value_head(x)

actor = Actor()
critic = Critic()
actor_optimizer = optim.Adam(actor.parameters(), lr=1e-4)
critic_optimizer = optim.Adam(critic.parameters(), lr=1e-4)

# Add critic losses to track
critic_losses = deque(maxlen=100)

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
    normalized_spinner = max(-31, min(31, spinner_target)) / 31.0
    
    # Add default value for spinner variance - low for demo data
    targets = torch.tensor([[float(fire_target), float(zap_target), normalized_spinner, 0.1]], 
                           dtype=torch.float32).to(device)
    
    model.optimizer.zero_grad()
    preds, _ = model(state_tensor)
    
    # Only train on fire, zap and spinner mean (not variance)
    loss = nn.MSELoss()(preds[:, :3], targets[:, :3])
    
    loss.backward()
    model.optimizer.step()
    
    actions = preds.detach().cpu().numpy()[0]
    return loss.item(), actions

def train_model_with_batch(model, batch):
    """Train model with a batch of data and reward information."""
    global bc_fire_accuracy_before, bc_fire_accuracy_after, bc_zap_accuracy_before, bc_zap_accuracy_after
    global bc_spinner_error_before, bc_spinner_error_after, bc_train_count, frame_count
    
    with bc_model_lock:
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
        targets = torch.FloatTensor([[f, z, s, 0.1] for f, z, s in zip(fire_targets, zap_targets, spinner_targets)]).to(device)
        reward_tensor = torch.FloatTensor(rewards).to(device)
        
        # Only set to train mode if we have more than 1 sample (BatchNorm requirement)
        batch_size = state_tensor.size(0)
        should_train = batch_size > 1
        
        if should_train:
            model.train()
        else:
            model.eval()  # Use eval mode for single samples to avoid BatchNorm issues
        
        # Before training metrics if enabled
        if TRACK_BATCH_METRICS:
            with torch.no_grad():
                preds_before, _ = model(state_tensor)
                
                # Calculate fire accuracy
                fire_pred_before = (preds_before[:, 0] > 0.5).float()
                fire_acc_before = (fire_pred_before == targets[:, 0]).float().mean().item()
                bc_fire_accuracy_before.append(fire_acc_before)
                
                # Calculate zap accuracy
                zap_pred_before = (preds_before[:, 1] > 0.5).float()
                zap_acc_before = (zap_pred_before == targets[:, 1]).float().mean().item()
                bc_zap_accuracy_before.append(zap_acc_before)
                
                # Calculate spinner error (MSE)
                spinner_error_before = F.mse_loss(preds_before[:, 2], targets[:, 2]).item()
                bc_spinner_error_before.append(spinner_error_before)
        
        model.optimizer.zero_grad()
        preds, spinner_out_raw = model(state_tensor)
        
        reward_weights = torch.log1p(reward_tensor - reward_tensor.min() + 1e-6)
        reward_weights = reward_weights / reward_weights.mean()
        reward_weights = reward_weights.unsqueeze(1)
        
        # Only use first 3 outputs for the loss (fire, zap, spinner mean)
        fire_zap_loss = F.binary_cross_entropy(preds[:, :2], targets[:, :2], reduction='none') * reward_weights
        fire_zap_loss = fire_zap_loss.mean()
        spinner_loss = F.mse_loss(preds[:, 2:3], targets[:, 2:3], reduction='none') * reward_weights * 0.5
        spinner_loss = spinner_loss.mean()
        
        loss = fire_zap_loss + spinner_loss
        
        if should_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            model.optimizer.step()
        
        # After training metrics if enabled
        if TRACK_BATCH_METRICS:
            bc_train_count += 1
            with torch.no_grad():
                preds_after, _ = model(state_tensor)
                
                # Calculate fire accuracy
                fire_pred_after = (preds_after[:, 0] > 0.5).float()
                fire_acc_after = (fire_pred_after == targets[:, 0]).float().mean().item()
                bc_fire_accuracy_after.append(fire_acc_after)
                
                # Calculate zap accuracy
                zap_pred_after = (preds_after[:, 1] > 0.5).float()
                zap_acc_after = (zap_pred_after == targets[:, 1]).float().mean().item()
                bc_zap_accuracy_after.append(zap_acc_after)
                
                # Calculate spinner error (MSE)
                spinner_error_after = F.mse_loss(preds_after[:, 2], targets[:, 2]).item()
                bc_spinner_error_after.append(spinner_error_after)
    
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

def background_rl_train(rl_model_lock, actor_model, critic_model):
    """Background RL training thread with explicit model parameters."""
    gamma = 0.99
    batch_size = 64
    
    global actor_losses, critic_losses, frame_count
    global fire_probs_before, fire_probs_after, zap_probs_before, zap_probs_after
    global spinner_means_before, spinner_means_after, spinner_vars_before, spinner_vars_after
    global value_errors_before, value_errors_after

    while True:
        if len(replay_buffer) < batch_size:
            time.sleep(0.01)
            continue

        try:
            with rl_model_lock:
                batch = random.sample(replay_buffer, batch_size)
                
                # Convert batch data to numpy arrays first for efficiency
                states_list, actions_list, rewards_list, next_states_list, dones_list = zip(*batch)
                
                # Convert lists to numpy arrays before creating tensors
                states = torch.tensor(np.array(states_list), dtype=torch.float32, device=device)
                actions = torch.tensor(np.array(actions_list), dtype=torch.float32, device=device)
                rewards = torch.tensor(np.array(rewards_list), dtype=torch.float32, device=device).unsqueeze(1)
                next_states = torch.tensor(np.array(next_states_list), dtype=torch.float32, device=device)
                dones = torch.tensor(np.array(dones_list), dtype=torch.float32, device=device).unsqueeze(1)

                # Collect metrics before training if enabled
                if TRACK_BATCH_METRICS:
                    with torch.no_grad():
                        # Actor metrics before training
                        action_logits_before = actor_model(states)
                        fire_logits_before, zap_logits_before, spinner_mean_before, spinner_var_before = torch.split(
                            action_logits_before, 1, dim=-1)
                        
                        fire_prob_before = torch.sigmoid(fire_logits_before).mean().item()
                        zap_prob_before = torch.sigmoid(zap_logits_before).mean().item()
                        spinner_mean_avg_before = spinner_mean_before.mean().item()
                        spinner_var_avg_before = spinner_var_before.mean().item()
                        
                        fire_probs_before.append(fire_prob_before)
                        zap_probs_before.append(zap_prob_before)
                        spinner_means_before.append(spinner_mean_avg_before)
                        spinner_vars_before.append(spinner_var_avg_before)
                        
                        # Critic metrics before training
                        current_values_before = critic_model(states)
                        next_values_before = critic_model(next_states)
                        target_values_before = rewards + (1.0 - dones) * gamma * next_values_before
                        value_error_before = F.mse_loss(current_values_before, target_values_before).item()
                        value_errors_before.append(value_error_before)

                # Critic training
                critic_optimizer.zero_grad(set_to_none=True)
                current_values = critic_model(states)
                with torch.no_grad():
                    next_values = critic_model(next_states)
                    target_values = rewards + (1.0 - dones) * gamma * next_values
                critic_loss = F.mse_loss(current_values, target_values)
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic_model.parameters(), max_norm=0.5)  # Stricter clipping
                critic_optimizer.step()
                critic_losses.append(critic_loss.item())

                # Actor training
                actor_optimizer.zero_grad(set_to_none=True)
                action_logits = actor_model(states)
                with torch.no_grad():
                    current_values = critic_model(states)
                    advantages = torch.clamp(target_values - current_values, -10.0, 10.0)  # Clip advantages
                fire_logits, zap_logits, spinner_mean, spinner_var = torch.split(action_logits, 1, dim=-1)
                
                fire_probs = torch.clamp(torch.sigmoid(fire_logits), 1e-8, 1 - 1e-8)
                zap_probs = torch.clamp(torch.sigmoid(zap_logits), 1e-8, 1 - 1e-8)
                
                fire_log_probs = (torch.log(fire_probs) * actions[:, 0:1]) + (torch.log(1 - fire_probs) * (1 - actions[:, 0:1]))
                zap_log_probs = (torch.log(zap_probs) * actions[:, 1:2]) + (torch.log(1 - zap_probs) * (1 - actions[:, 1:2]))
                
                # Calculate log probabilities for spinner using Gaussian distribution
                # Reduce variance scaling to make extreme log probs less likely
                spinner_scaled_var = spinner_var * 0.5 + 0.01  # Add a minimum variance to avoid division by near-zero
                spinner_diff = spinner_mean - actions[:, 2:3]
                spinner_log_probs = -0.5 * (spinner_diff**2) / (spinner_scaled_var + 1e-8) - 0.5 * torch.log(2 * np.pi * (spinner_scaled_var + 1e-8))
                
                # Clip log probs to prevent extremely large values
                fire_log_probs = torch.clamp(fire_log_probs, -10.0, 0.0)
                zap_log_probs = torch.clamp(zap_log_probs, -10.0, 0.0)
                spinner_log_probs = torch.clamp(spinner_log_probs, -10.0, 0.0)
                
                log_probs = fire_log_probs + zap_log_probs + spinner_log_probs
                if advantages.dim() != log_probs.dim():
                    log_probs = log_probs.sum(dim=1, keepdim=True)
                
                # Multiply by normalized advantages instead of raw advantages to prevent huge gradient steps
                normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                normalized_advantages = torch.clamp(normalized_advantages, -3.0, 3.0)
                actor_loss = -(log_probs * normalized_advantages.detach()).mean()
                
                # Add L2 regularization for spinner head to prevent extreme values
                l2_reg = 0.001 * (actor_model.spinner_head.weight.pow(2).sum() + actor_model.spinner_var_head.weight.pow(2).sum())
                total_loss = actor_loss + l2_reg
                
                total_loss.backward()
                # Increase gradient clipping to prevent too small updates
                torch.nn.utils.clip_grad_norm_(actor_model.parameters(), max_norm=1.0)
                actor_optimizer.step()
                actor_losses.append(actor_loss.item())
                
                # Collect metrics after training if enabled
                if TRACK_BATCH_METRICS:
                    with torch.no_grad():
                        # Actor metrics after training
                        action_logits_after = actor_model(states)
                        fire_logits_after, zap_logits_after, spinner_mean_after, spinner_var_after = torch.split(
                            action_logits_after, 1, dim=-1)
                        
                        fire_prob_after = torch.sigmoid(fire_logits_after).mean().item()
                        zap_prob_after = torch.sigmoid(zap_logits_after).mean().item()
                        spinner_mean_avg_after = spinner_mean_after.mean().item()
                        spinner_var_avg_after = spinner_var_after.mean().item()
                        
                        fire_probs_after.append(fire_prob_after)
                        zap_probs_after.append(zap_prob_after)
                        spinner_means_after.append(spinner_mean_avg_after)
                        spinner_vars_after.append(spinner_var_avg_after)
                        
                        # Critic metrics after training
                        current_values_after = critic_model(states)
                        next_values_after = critic_model(next_states)
                        target_values_after = rewards + (1.0 - dones) * gamma * next_values_after
                        value_error_after = F.mse_loss(current_values_after, target_values_after).item()
                        value_errors_after.append(value_error_after)
                        
                        # Just update metrics without printing
                        fire_trend = np.mean(fire_probs_after) - np.mean(fire_probs_before)
                        zap_trend = np.mean(zap_probs_after) - np.mean(zap_probs_before)
                        spinner_mean_trend = np.mean(spinner_means_after) - np.mean(spinner_means_before)
                        spinner_var_trend = np.mean(spinner_vars_after) - np.mean(spinner_vars_before)
                        value_error_trend = np.mean(value_errors_after) - np.mean(value_errors_before)

        except Exception as e:
            print(f"RL training error: {e}")
            traceback.print_exc()
            time.sleep(1)

def replay_log_file(log_file_path, bc_model):
    """Train BC model by replaying demonstration log file using process_frame_data."""
    if not os.path.exists(log_file_path):
        print(f"Error: Log file {log_file_path} not found")
        return False
    
    print(f"Replaying log file: {log_file_path}")
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
    
    # Setup progress tracking
    print("\n" + "-" * 50)
    print(f"{'Frames':>8} | {'Loss':>8} | {'Fire%':>6} | {'Zap%':>6} | {'Reward':>8}")
    print("-" * 50)
    
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
                
                # Print progress every 1000 frames
                if frame_count % 1000 == 0:
                    current_fire_rate = total_fire / max(1, frames_processed) * 100
                    current_zap_rate = total_zap / max(1, frames_processed) * 100
                    avg_reward = total_reward / max(1, frames_processed)
                    
                    print(f"{frame_count:8d} | {batch_loss:8.4f} | {current_fire_rate:6.2f} | "
                          f"{current_zap_rate:6.2f} | {avg_reward:8.2f}")
                
                training_data = training_data[batch_size:]
                frame_count += batch_size
        
        if training_data:
            batch_loss = train_model_with_batch(bc_model, training_data)
            total_loss += batch_loss
            num_batches += 1
            frame_count += len(training_data)
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('nan')
        
        print("-" * 50)
        print(f"Replay complete: {frame_count} frames processed")
        print(f"Final Statistics:")
        print(f"  - Fire Rate: {total_fire/frames_processed*100:.2f}%")
        print(f"  - Zap Rate: {total_zap/frames_processed*100:.2f}%") 
        print(f"  - Avg Reward: {total_reward/frames_processed:.2f}")
        print(f"  - Avg Loss: {avg_loss:.6f}")
        
        print(f"Saving BC model to {BC_MODEL_PATH}")
        torch.save(bc_model.state_dict(), BC_MODEL_PATH)
        return True

frame_count = 0

def setup_keyboard_input():
    """Set up non-blocking keyboard input."""
    fd = sys.stdin.fileno()
    # Save the old terminal settings
    old_settings = termios.tcgetattr(fd)
    # Set the terminal to raw mode
    new_settings = termios.tcgetattr(fd)
    new_settings[3] = new_settings[3] & ~termios.ICANON & ~termios.ECHO
    termios.tcsetattr(fd, termios.TCSANOW, new_settings)
    # Set non-blocking mode
    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
    return old_settings

def restore_keyboard_input(old_settings):
    """Restore original keyboard input settings."""
    termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, old_settings)

def print_metrics_table_header():
    """Print the header row for the metrics table."""
    header = (
        f"{'Frame':>8} | {'Actor Loss':>10} | {'Critic Loss':>11} | {'Mean Reward':>12} | {'Explore %':>9} | "
        f"{'Random %':>8} | {'BC %':>8} | {'RL %':>8} | {'Spin Err':>8} | {'Extreme %':>9} | {'Buffer':>7}"
    )
    separator = "-" * len(header)
    print("\n" + separator)
    print(header)
    print(separator)

def print_metrics_table_row(frame_count, metrics_dict):
    """Print a single row of the metrics table with the provided metrics."""
    row = (
        f"{frame_count:8d} | {metrics_dict.get('actor_loss', 'N/A'):10.4f} | "
        f"{metrics_dict.get('critic_loss', 'N/A'):11.2f} | {metrics_dict.get('mean_reward', 'N/A'):12.2f} | "
        f"{metrics_dict.get('exploration', 0.0)*100:8.2f}% | "
        f"{metrics_dict.get('random_ratio', 0.0)*100:7.2f}% | {metrics_dict.get('bc_ratio', 0.0)*100:7.2f}% | "
        f"{metrics_dict.get('rl_ratio', 0.0)*100:7.2f}% | {metrics_dict.get('spinner_error', 'N/A'):8.4f} | "
        f"{metrics_dict.get('extreme_ratio', 0.0)*100:8.2f}% | {metrics_dict.get('buffer_size', 0):7d}"
    )
    print(row)

def main():
    """Main execution loop for Tempest AI."""
    global frame_count, actor, critic
    global random_control_count, bc_control_count, rl_control_count, total_control_count
    
    # Reset control source counters
    random_control_count = 0
    bc_control_count = 0
    rl_control_count = 0
    total_control_count = 0
    
    bc_model, actor = initialize_models(actor, critic)
    
    rl_model_lock = threading.Lock()
    
    # Set up non-blocking keyboard input
    try:
        old_terminal_settings = setup_keyboard_input()
        print("Keyboard commands enabled: Press 'm' for metrics display")
    except:
        print("Warning: Could not set up keyboard input. Metrics display via keyboard disabled.")
        old_terminal_settings = None
    
    threading.Thread(target=background_rl_train, args=(rl_model_lock, actor, critic), daemon=True).start()
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
    global episode_rewards
    total_episode_reward = 0
    done_latch = False
    
    # Lower initial exploration and make it decay faster

    current_exploration_ratio = initial_exploration_ratio
    
    # Define exploration strategies
    
    # Add tracking for spinner values
    last_spinner_values = deque(maxlen=100)
    extreme_threshold = 0.8  # Consider values above this magnitude as extreme
    
    # Print metrics table header
    print_metrics_table_header()
    
    # Store previous state for proper next_state in replay buffer
    previous_state = None
    
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
                    total_episode_reward += reward
                    episode_rewards.append(total_episode_reward)
                    
                    # print(f"Episode completed with reward: {total_episode_reward:.2f}")
                    
                    # Reset tracking on new episode
                    last_spinner_values.clear()
                    previous_state = None
                    
                    if len(replay_buffer) >= 256 and not is_attract:
                        #print("Performing end-of-episode model update...")
                        with rl_model_lock:
                            for update_step in range(5):
                                batch = random.sample(replay_buffer, min(256, len(replay_buffer)))
                                
                                # Convert batch data to numpy arrays first for efficiency
                                states_list, actions_list, rewards_list, next_states_list, dones_list = zip(*batch)
                                
                                # Convert lists to numpy arrays before creating tensors
                                states = torch.tensor(np.array(states_list), dtype=torch.float32, device=device)
                                actions = torch.tensor(np.array(actions_list), dtype=torch.float32, device=device)
                                rewards = torch.tensor(np.array(rewards_list), dtype=torch.float32, device=device).unsqueeze(1)
                                next_states = torch.tensor(np.array(next_states_list), dtype=torch.float32, device=device)
                                dones = torch.tensor(np.array(dones_list), dtype=torch.float32, device=device).unsqueeze(1)

                                # Rest of the training code remains the same
                                # ... [unchanged code]
                
                elif done and done_latch:
                    py_to_lua.write(struct.pack("bbb", 0, 0, 0))
                    py_to_lua.flush()
                    continue
                    
                elif not done and done_latch:
                    done_latch = False
                    total_episode_reward = 0  # Reset reward at beginning of new episode
                    previous_state = None  # Reset previous state when episode starts
                    
                else:
                    total_episode_reward += reward
                    if not is_attract and previous_state is not None:
                        # Store (state, action, reward, next_state, done) - now with proper next_state
                        replay_buffer.append((previous_state, game_action, reward, state, done))

                # Update previous state for next iteration
                previous_state = state

                if is_attract and game_action:
                    try:
                        bc_training_queue.put((state, game_action, reward), block=False)
                        action = encode_action(1, 1, -1)
                    except queue.Full:
                        print("BC training queue is full, skipping attract mode action")
                        action = encode_action(0, 0, 0)
                else:
                    # Detect if we're stuck in a pattern of extreme spinner values
                    if len(last_spinner_values) >= 50:  # Wait until we have enough data
                        extreme_count = sum(1 for v in last_spinner_values if abs(v) > extreme_threshold)
                        extreme_ratio = extreme_count / len(last_spinner_values)
                        
                        # Calculate how close spinner values are to optimal
                        normalized_optimal = 0
                        avg_distance = sum(abs(abs(v) - abs(normalized_optimal)) for v in last_spinner_values) / len(last_spinner_values)
                        
                        # If over 70% of recent actions are extreme values, we might be stuck
                        if extreme_ratio > 0.7:
                            # Increase exploration to break out of the pattern
                            current_exploration_ratio = max(current_exploration_ratio, 0.5)
                    
                    # Update exploration ratio
                    if frame_count % 1000 == 0 and current_exploration_ratio > min_exploration_ratio:
                        current_exploration_ratio *= exploration_decay
                    
                    # Choose between exploitation and exploration
                    if random.random() < current_exploration_ratio:
                        # In exploration mode, choose between BC-guided and random exploration
                        if random.random() < BC_GUIDED_EXPLORATION_RATIO:
                            # BC-guided exploration: use the BC model's prediction with added noise
                            with torch.no_grad():
                                # Temporarily set to eval mode to prevent BatchNorm issues with single samples
                                bc_model.eval()
                                bc_preds, _ = bc_model(state_tensor)
                                
                                # Extract predictions
                                bc_fire = bc_preds[0, 0].item() > 0.5
                                bc_zap = bc_preds[0, 1].item() > 0.5
                                
                                # For spinner, add noise to BC's prediction to encourage exploration
                                # while still leveraging the BC model's knowledge
                                bc_spinner = bc_preds[0, 2].item()  # Raw spinner value from BC model
                                
                                # Add Gaussian noise to spinner (more noise during early exploration)
                                noise_scale = 0.3  # Base noise scale
                                spinner_value = bc_spinner + random.gauss(0, noise_scale)
                                spinner_value = max(-0.95, min(0.95, spinner_value))
                                
                                if (random.random() < FIRE_EXPLORATION_RATIO):
                                    bc_fire = 0

                                # Create action tensor with BC guidance plus noise
                                action = torch.tensor([[float(bc_fire), float(bc_zap), spinner_value]], 
                                                     dtype=torch.float32, device=device)
                                
                                # Track BC-guided control
                                bc_control_count += 1
                                total_control_count += 1
                        else:
                            # Pure random exploration (20% of exploration actions)
                            # For truly random exploration, sometimes use values near the optimal range
                            if random.random() < 0.4:  # 40% of pure random exploration targets optimal range
                                # Generate values with a bias toward the optimal spinner range
                                optimal_normalized = 0
                                # Add Gaussian noise centered at optimal value
                                spinner_val = optimal_normalized + random.gauss(0, 0.2)
                                spinner_val = max(-0.95, min(0.95, spinner_val))
                                
                                random_actions = torch.tensor([
                                    float(random.random() > 0.5),
                                    float(random.random() > 0.95),
                                    spinner_val
                                ], dtype=torch.float32, device=device)
                            else:
                                random_actions = torch.tensor([
                                    float(random.random() > 0.5),
                                    float(random.random() > 0.95),
                                    random.uniform(-1.0, 1.0)
                                ], dtype=torch.float32, device=device)
                            action = random_actions.unsqueeze(0)
                            
                            # Track random control
                            random_control_count += 1
                            total_control_count += 1
                    else:
                        # In exploitation mode, use the RL policy
                        with torch.no_grad():
                            action = actor.get_action(state_tensor, deterministic=True)
                        
                        # Track RL control
                        rl_control_count += 1
                        total_control_count += 1
                
                action_cpu = action.cpu()
                fire, zap, spinner = decode_action(action_cpu)
                
                # Track the model's raw spinner output for analysis
                last_spinner_values.append(action_cpu[0, 2].item())

                # Check for keyboard input (non-blocking)
                try:
                    key = sys.stdin.read(1)
                    if key == 'm':
                        print("\nDisplaying metrics summary (triggered by keyboard)...")
                        display_combined_metrics()
                        print_metrics_table_header()  # Reprint header after detailed metrics
                    # Add more keyboard commands here if needed
                except:
                    pass
                
                py_to_lua.write(struct.pack("bbb", fire, zap, spinner))
                py_to_lua.flush()

                if frame_count % 1000 == 0:
                    # Calculate metrics for table row
                    actor_loss_mean = np.mean(actor_losses) if actor_losses else float('nan')
                    critic_loss_mean = np.mean(critic_losses) if critic_losses else float('nan')
                    
                    # Only use the last 5 episode rewards for the mean to prevent endless increase
                    mean_reward = np.mean(list(episode_rewards)) if episode_rewards else float('nan')
                    
                    # Calculate spinner value distribution
                    extreme_ratio = 0.0
                    optimal_ratio = 0.0
                    if last_spinner_values:
                        extreme_count = sum(1 for v in last_spinner_values if abs(v) > extreme_threshold)
                        extreme_ratio = extreme_count / len(last_spinner_values)
                        
                        # Calculate how often we're near the optimal spinner value
                        normalized_optimal = 0
                        close_to_optimal_count = sum(1 for v in last_spinner_values if abs(abs(v) - abs(normalized_optimal)) < 0.1)
                        optimal_ratio = close_to_optimal_count / len(last_spinner_values)
                    
                    # Calculate control source ratios
                    random_ratio = random_control_count / max(1, total_control_count)
                    bc_ratio = bc_control_count / max(1, total_control_count)
                    rl_ratio = rl_control_count / max(1, total_control_count)
                    
                    # Get spinner error for BC model
                    spinner_error = np.mean(bc_spinner_error_after) if len(bc_spinner_error_after) > 0 else float('nan')
                    
                    # Prepare metrics dictionary for table display
                    metrics = {
                        'actor_loss': actor_loss_mean,
                        'critic_loss': critic_loss_mean,
                        'mean_reward': mean_reward,
                        'exploration': current_exploration_ratio,
                        'random_ratio': random_ratio,
                        'bc_ratio': bc_ratio,
                        'rl_ratio': rl_ratio,
                        'spinner_error': spinner_error,
                        'extreme_ratio': extreme_ratio,
                        'optimal_ratio': optimal_ratio,
                        'buffer_size': len(replay_buffer)
                    }
                    
                    # Print metrics in table format
                    print_metrics_table_row(frame_count, metrics)
                    
                    # Show full metrics summary every 50k frames
                    if frame_count % 50000 == 0:
                        display_combined_metrics()
                        print_metrics_table_header()  # Reprint header after detailed metrics

                if save_signal:
                    print("\nSaving models...")
                    threading.Thread(target=save_models, args=(actor, bc_model), daemon=True).start()
                    #current_exploration_ratio = min(0.5, current_exploration_ratio * 1.5)  # Increase exploration after save
                    #print(f"Increased exploration to {current_exploration_ratio:.2f} after save")
                    print_metrics_table_header()  # Reprint header after save message
                
                frame_count += 1

            except KeyboardInterrupt:
                save_models(actor, bc_model)
                break
            except Exception as e:
                print(f"Error: {e}")
                traceback.print_exc()
                time.sleep(5)
        
        # Restore terminal settings before exiting
        if old_terminal_settings:
            restore_keyboard_input(old_terminal_settings)

def transfer_knowledge_from_bc_to_rl(bc_model, actor_model):
    """Transfer knowledge from Behavioral Cloning model to Reinforcement Learning model.
    
    This makes the RL model start with priors learned from demonstrations, significantly
    improving initial performance and exploration.
    """
    print("Transferring knowledge from BC model to RL model...")
    
    # First, get the feature extractor weights from BC model
    with torch.no_grad():
        # Transfer first layer - BC index 0 to Actor fc1
        actor_model.fc1.weight.data.copy_(bc_model.feature_extractor[0].weight.data)
        actor_model.fc1.bias.data.copy_(bc_model.feature_extractor[0].bias.data)
        
        # Transfer second layer - BC index 4 to Actor fc2
        actor_model.fc2.weight.data.copy_(bc_model.feature_extractor[4].weight.data)
        actor_model.fc2.bias.data.copy_(bc_model.feature_extractor[4].bias.data)
        
        # Transfer third layer - BC index 6 to Actor fc3
        actor_model.fc3.weight.data.copy_(bc_model.feature_extractor[6].weight.data)
        actor_model.fc3.bias.data.copy_(bc_model.feature_extractor[6].bias.data)
        
        # Transfer output heads
        actor_model.fire_head.weight.data.copy_(bc_model.fire_output.weight.data)
        actor_model.fire_head.bias.data.copy_(bc_model.fire_output.bias.data)
        
        actor_model.zap_head.weight.data.copy_(bc_model.zap_output.weight.data)
        actor_model.zap_head.bias.data.copy_(bc_model.zap_output.bias.data)
        
        actor_model.spinner_head.weight.data.copy_(bc_model.spinner_output.weight.data)
        actor_model.spinner_head.bias.data.copy_(bc_model.spinner_output.bias.data)
        
        # Initialize spinner variance head with reasonable values
        actor_model.spinner_var_head.bias.data.fill_(-1.0)  # Start with low variance
    
    print("Knowledge transfer complete!")
    return actor_model

def initialize_models(actor_model=None, critic_model=None):
    """Initialize BC and RL models, with knowledge transfer between them."""
    if actor_model is None:
        actor_model = Actor()
    
    if critic_model is None:
        critic_model = Critic()
    
    bc_model = BCModel()
    transfer_happened = False
    
    # First, try to load the BC model if it exists
    if os.path.exists(BC_MODEL_PATH):
        # Always load in eval mode to avoid BatchNorm issues
        bc_model.eval()
        bc_model.load_state_dict(torch.load(BC_MODEL_PATH, map_location=device))
        print(f"Loaded BC model from {BC_MODEL_PATH}")
        bc_loaded = True
    else:
        bc_loaded = False
        print("No existing BC model found - will train from scratch")
    
    # Check if RL model exists
    if os.path.exists(LATEST_MODEL_PATH):
        # Always load in eval mode to avoid BatchNorm issues
        actor_model.eval()
        critic_model.eval()
        try:
            actor_model.load_state_dict(torch.load(LATEST_MODEL_PATH, map_location=device))
            print(f"Loaded RL model from {LATEST_MODEL_PATH}")
        except Exception as e:
            print(f"Error loading RL model: {e}. Will initialize from BC model.")
            if bc_loaded:
                actor_model = transfer_knowledge_from_bc_to_rl(bc_model, actor_model)
                transfer_happened = True
    elif bc_loaded:
        # If RL model doesn't exist but BC model does, transfer knowledge
        actor_model = transfer_knowledge_from_bc_to_rl(bc_model, actor_model)
        transfer_happened = True
    
    # Even if we loaded the RL model, we might still benefit from BC knowledge
    if not transfer_happened and bc_loaded and random.random() < 0.3:  # 30% chance to refresh with BC knowledge
        print("Refreshing RL model with BC knowledge (periodic knowledge update)")
        actor_model = transfer_knowledge_from_bc_to_rl(bc_model, actor_model)
    
    return bc_model, actor_model

def save_models(actor_model=None, bc_model=None):
    try:
        if bc_model is not None:
            with bc_model_lock:
                # Set to eval mode before saving to avoid BatchNorm issues on reload
                bc_model.eval()
                torch.save(bc_model.state_dict(), BC_MODEL_PATH)
                print(f"BC model saved to {BC_MODEL_PATH}")
                
        if actor_model is not None:
            # Set to eval mode before saving
            actor_model.eval()
            torch.save(actor_model.state_dict(), LATEST_MODEL_PATH)
            print(f"Actor model saved to {LATEST_MODEL_PATH}")
            
        print(f"All models saved to {MODEL_DIR}")
    except Exception as e:
        print(f"Error saving models: {e}")
        traceback.print_exc()

def encode_action(fire, zap, spinner_delta):
    """Encode actions for RL model consistency."""
    fire_val = float(fire)
    zap_val = float(zap)
    if abs(spinner_delta) < 10:
        normalized_spinner = spinner_delta / 31.0  # Scale to -1 to +1 from -31 to +31
    else:
        sign = np.sign(spinner_delta)
        magnitude = abs(spinner_delta)
        squashed = 10.0 + (magnitude - 10.0) * (1.0 / (1.0 + (magnitude - 10.0) / 20.0))
        normalized_spinner = sign * min(squashed / 31.0, 1.0)  # Scale to -1 to +1 from -31 to +31
    # Return a PyTorch tensor with a batch dimension instead of numpy array
    return torch.tensor([[fire_val, zap_val, normalized_spinner]], dtype=torch.float32, device=device)

def decode_action(action, spinner_power=1.0):
    """Decode actions from PyTorch tensor with a preference for optimal speed."""
    fire = 1 if action[0, 0].item() > 0.5 else 0
    zap = 1 if action[0, 1].item() > 0.5 else 0
    
    # Get raw spinner value between -1 and 1
    spinner_val = action[0, 2].item()

    threshold = 0.1
    if (spinner_val > threshold):
        spinner = 9
    elif (spinner_val < -threshold):
        spinner = -9
    else:
        spinner = 0
    
    return fire, zap, spinner

def display_combined_metrics():
    """Display a compact summary of training metrics."""
    global episode_rewards, replay_buffer
    
    try:
        print("\n========== METRICS SUMMARY ==========")
        
        # Define table headers for each metric category
        print("\nBC METRICS:")
        print(f"{'Metric':<25} {'Before':<10} {'After':<10} {'Change':<10}")
        print("-" * 55)
        
        # BC metrics
        if len(bc_fire_accuracy_before) > 0 and len(bc_fire_accuracy_after) > 0:
            fire_before = np.mean(bc_fire_accuracy_before)
            fire_after = np.mean(bc_fire_accuracy_after)
            fire_change = fire_after - fire_before
            
            zap_before = np.mean(bc_zap_accuracy_before)
            zap_after = np.mean(bc_zap_accuracy_after)
            zap_change = zap_after - zap_before
            
            spinner_err_before = np.mean(bc_spinner_error_before)
            spinner_err_after = np.mean(bc_spinner_error_after)
            spinner_change = spinner_err_before - spinner_err_after
            
            print(f"{'Fire Accuracy':<25} {fire_before:<10.4f} {fire_after:<10.4f} {fire_change:+<10.4f}")
            print(f"{'Zap Accuracy':<25} {zap_before:<10.4f} {zap_after:<10.4f} {zap_change:+<10.4f}")
            print(f"{'Spinner Error':<25} {spinner_err_before:<10.4f} {spinner_err_after:<10.4f} {-spinner_change:+<10.4f}")
        else:
            print("No BC training data available")
        
        # RL metrics
        print("\nRL METRICS:")
        print(f"{'Metric':<25} {'Value':<10} {'Trend':<10}")
        print("-" * 45)
        
        if len(actor_losses) > 0:
            actor_loss_avg = np.mean(actor_losses)
            critic_loss_avg = np.mean(critic_losses) if critic_losses else float('nan')
            
            print(f"{'Actor Loss':<25} {actor_loss_avg:<10.4f}")
            print(f"{'Critic Loss':<25} {critic_loss_avg:<10.2f}")
            
            if len(fire_probs_before) > 0 and len(fire_probs_after) > 0:
                fire_trend = np.mean(fire_probs_after) - np.mean(fire_probs_before)
                zap_trend = np.mean(zap_probs_after) - np.mean(zap_probs_before)
                spinner_mean_trend = np.mean(spinner_means_after) - np.mean(spinner_means_before)
                spinner_var_trend = np.mean(spinner_vars_after) - np.mean(spinner_vars_before)
                
                print(f"{'Fire Probability':<25} {np.mean(fire_probs_after):<10.4f} {fire_trend:+<10.4f}")
                print(f"{'Zap Probability':<25} {np.mean(zap_probs_after):<10.4f} {zap_trend:+<10.4f}")
                print(f"{'Spinner Mean':<25} {np.mean(spinner_means_after):<10.4f} {spinner_mean_trend:+<10.4f}")
                print(f"{'Spinner Variance':<25} {np.mean(spinner_vars_after):<10.4f} {spinner_var_trend:+<10.4f}")
                
                if len(value_errors_before) > 0 and len(value_errors_after) > 0:
                    value_error_trend = np.mean(value_errors_before) - np.mean(value_errors_after)
                    print(f"{'Value Error Reduction':<25} {np.mean(value_errors_after):<10.4f} {value_error_trend:+<10.4f}")
        else:
            print("No RL training data available")
        
        # Game performance
        print("\nGAME PERFORMANCE:")
        print(f"{'Metric':<25} {'Value':<10}")
        print("-" * 35)
        
        if episode_rewards and len(episode_rewards) > 0:
            mean_reward = np.mean(episode_rewards)
            latest_reward = episode_rewards[-1] if episode_rewards else 0
            buffer_size = len(replay_buffer) if 'replay_buffer' in globals() else 0
            
            print(f"{'Latest Episode Reward':<25} {latest_reward:<10.1f}")
            print(f"{'Mean Episode Reward':<25} {mean_reward:<10.1f}")
            print(f"{'Buffer Size':<25} {buffer_size:<10d}")
            print(f"{'Episodes Completed':<25} {len(episode_rewards):<10d}")
        else:
            print("No completed episodes yet")
        
        print("\n========== END SUMMARY ==========\n")
    except Exception as e:
        print(f"\nError displaying metrics: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    main()