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
FORCE_CPU = False  # Force CPU usage if having persistent issues with MPS
SPINNER_POWER = 1.0  # Linear spinner movement (1.0 = linear, higher = more small movements)
initial_exploration_ratio = 0.25
min_exploration_ratio = 0.05  # Lowest exploration rate
exploration_decay = 0.99  # Decay rate
BC_GUIDED_EXPLORATION_RATIO = 1.0
LogFile = "/Users/dave/mame/big.log"
MaxLogFrames = 1000000
FIRE_EXPLORATION_RATIO = 0.05
NumberOfParams = 128  # Confirm this matches Lua data serialization
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
if FORCE_CPU:
    device = torch.device("cpu")
    print("Using device: CPU (forced)")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device.type.upper()}")

# Training metrics buffers and synchronization
actor_losses = deque(maxlen=100)
critic_losses = deque(maxlen=100)
bc_training_queue = queue.Queue(maxsize=1000)
bc_model_lock = threading.Lock()
replay_buffer = deque(maxlen=100000)
episode_rewards = deque(maxlen=5)
bc_spinner_error_after = deque(maxlen=20)

# Control source tracking
random_control_count = 0
bc_control_count = 0
rl_control_count = 0
total_control_count = 0

class BCModel(nn.Module):
    """Enhanced BC model with clipped linear spinner output."""
    def __init__(self, input_size=NumberOfParams):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )
        self.fire_output = nn.Linear(128, 1)
        self.zap_output = nn.Linear(128, 1)
        self.spinner_output = nn.Linear(128, 1)
        self.spinner_var_output = nn.Linear(128, 1)
        
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
            
        x = self.feature_extractor[0](x)
        x = self.feature_extractor[1](x)
        if x.size(0) > 1:
            x = self.feature_extractor[2](x)
        x = self.feature_extractor[3](x)
        x = self.feature_extractor[4](x)
        if x.size(0) > 1:
            x = self.feature_extractor[5](x)
        x = self.feature_extractor[6](x)
        x = self.feature_extractor[7](x)
        x = self.feature_extractor[8](x)
        x = self.feature_extractor[9](x)
        x = self.feature_extractor[10](x)
        
        fire_out = torch.sigmoid(self.fire_output(x))
        zap_out = torch.sigmoid(self.zap_output(x))
        spinner_out = torch.tanh(self.spinner_output(x))
        spinner_var = torch.clamp(F.softplus(self.spinner_var_output(x)), 0.01, 0.5)
        
        return torch.cat([fire_out, zap_out, spinner_out, spinner_var], dim=1), spinner_out

class Actor(nn.Module):
    def __init__(self, state_dim=NumberOfParams):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 256)
        self.fire_head = nn.Linear(256, 1)
        self.zap_head = nn.Linear(256, 1)
        self.spinner_head = nn.Linear(256, 1)
        self.spinner_var_head = nn.Linear(256, 1)
        
        torch.nn.init.xavier_uniform_(self.spinner_head.weight, gain=0.1)
        torch.nn.init.xavier_uniform_(self.spinner_var_head.weight, gain=0.1)
        if self.spinner_head.bias is not None:
            torch.nn.init.zeros_(self.spinner_head.bias)
        if self.spinner_var_head.bias is not None:
            torch.nn.init.constant_(self.spinner_var_head.bias, -1.0)
            
        self.to(device)

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        if x.size(0) == 1:
            x1 = x
        else:
            x1 = self.bn1(x)
        x = self.fc2(x1)
        x = F.relu(x)
        if x.size(0) == 1:
            x2 = x
        else:
            x2 = self.bn2(x)
        x = F.relu(self.fc3(x2))
        
        fire_logits = torch.clamp(self.fire_head(x), -10.0, 10.0)
        zap_logits = torch.clamp(self.zap_head(x), -10.0, 10.0)
        raw_spinner = self.spinner_head(x)
        normalized_optimal = 0
        distance_from_optimal = raw_spinner - normalized_optimal
        bell_curve_factor = torch.exp(-4.0 * distance_from_optimal * distance_from_optimal)
        spinner_mean = torch.tanh(raw_spinner * bell_curve_factor * 2.0)
        raw_var = self.spinner_var_head(x)
        spinner_var = torch.sigmoid(raw_var) * 0.15
        
        return torch.cat([fire_logits, zap_logits, spinner_mean, spinner_var], dim=-1)

    def get_action(self, state, deterministic=False):
        logits = self.forward(state)
        fire_logits, zap_logits, spinner_mean, spinner_var = torch.split(logits, 1, dim=-1)
        
        if deterministic:
            fire = torch.sigmoid(fire_logits) > 0.5
            zap = torch.sigmoid(zap_logits) > 0.5
            spinner_noise = torch.randn_like(spinner_mean) * spinner_var * 0.3
        else:
            fire = torch.bernoulli(torch.sigmoid(fire_logits))
            zap = torch.bernoulli(torch.sigmoid(zap_logits))
            spinner_noise = torch.randn_like(spinner_mean) * spinner_var
        
        spinner = torch.clamp(spinner_mean + spinner_noise, -0.95, 0.95)
        spinner = torch.nan_to_num(spinner, nan=0.0, posinf=0.5, neginf=-0.5)
        return torch.cat([fire, zap, spinner], dim=-1)

class Critic(nn.Module):
    def __init__(self, state_dim=NumberOfParams):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 256)
        self.value_head = nn.Linear(256, 1)
        self.to(device)

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        if x.size(0) == 1:
            x1 = x
        else:
            x1 = self.bn1(x)
        x = self.fc2(x1)
        x = F.relu(x)
        if x.size(0) == 1:
            x2 = x
        else:
            x2 = self.bn2(x)
        x = F.relu(self.fc3(x2))
        return self.value_head(x)

actor = Actor()
critic = Critic()
actor_optimizer = optim.Adam(actor.parameters(), lr=1e-4)
critic_optimizer = optim.Adam(critic.parameters(), lr=1e-4)

def process_frame_data(data):
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
        
        state = np.array(state_values, dtype=np.float32) / 32768.0
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

def train_model_with_batch(model, batch):
    """Train model with a batch of data and reward information."""
    global bc_spinner_error_after
    
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
        
        batch_size = state_tensor.size(0)
        should_train = batch_size > 1
        
        if should_train:
            model.train()
        else:
            model.eval()
        
        model.optimizer.zero_grad()
        preds, _ = model(state_tensor)
        
        reward_weights = torch.log1p(reward_tensor - reward_tensor.min() + 1e-6)
        reward_weights = reward_weights / reward_weights.mean()
        reward_weights = reward_weights.unsqueeze(1)
        
        fire_zap_loss = F.binary_cross_entropy(preds[:, :2], targets[:, :2], reduction='none') * reward_weights
        fire_zap_loss = fire_zap_loss.mean()
        spinner_loss = F.mse_loss(preds[:, 2:3], targets[:, 2:3], reduction='none') * reward_weights * 0.5
        spinner_loss = spinner_loss.mean()
        
        loss = fire_zap_loss + spinner_loss
        
        if should_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            model.optimizer.step()
        
        with torch.no_grad():
            preds_after, _ = model(state_tensor)
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
    
    while True:
        if len(replay_buffer) < batch_size:
            time.sleep(0.01)
            continue

        try:
            with rl_model_lock:
                batch = random.sample(replay_buffer, batch_size)
                states_list, actions_list, rewards_list, next_states_list, dones_list = zip(*batch)
                
                states = torch.tensor(np.array(states_list), dtype=torch.float32, device=device)
                actions = torch.tensor(np.array(actions_list), dtype=torch.float32, device=device)
                rewards = torch.tensor(np.array(rewards_list), dtype=torch.float32, device=device).unsqueeze(1)
                next_states = torch.tensor(np.array(next_states_list), dtype=torch.float32, device=device)
                dones = torch.tensor(np.array(dones_list), dtype=torch.float32, device=device).unsqueeze(1)

                critic_optimizer.zero_grad(set_to_none=True)
                current_values = critic_model(states)
                with torch.no_grad():
                    next_values = critic_model(next_states)
                    target_values = rewards + (1.0 - dones) * gamma * next_values
                critic_loss = F.mse_loss(current_values, target_values)
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic_model.parameters(), max_norm=0.5)
                critic_optimizer.step()
                critic_losses.append(critic_loss.item())

                actor_optimizer.zero_grad(set_to_none=True)
                action_logits = actor_model(states)
                with torch.no_grad():
                    current_values = critic_model(states)
                    advantages = torch.clamp(target_values - current_values, -10.0, 10.0)
                fire_logits, zap_logits, spinner_mean, spinner_var = torch.split(action_logits, 1, dim=-1)
                
                fire_probs = torch.clamp(torch.sigmoid(fire_logits), 1e-8, 1 - 1e-8)
                zap_probs = torch.clamp(torch.sigmoid(zap_logits), 1e-8, 1 - 1e-8)
                
                fire_log_probs = (torch.log(fire_probs) * actions[:, 0:1]) + (torch.log(1 - fire_probs) * (1 - actions[:, 0:1]))
                zap_log_probs = (torch.log(zap_probs) * actions[:, 1:2]) + (torch.log(1 - zap_probs) * (1 - actions[:, 1:2]))
                
                spinner_scaled_var = spinner_var * 0.5 + 0.01
                spinner_diff = spinner_mean - actions[:, 2:3]
                spinner_log_probs = -0.5 * (spinner_diff**2) / (spinner_scaled_var + 1e-8) - 0.5 * torch.log(2 * np.pi * (spinner_scaled_var + 1e-8))
                
                fire_log_probs = torch.clamp(fire_log_probs, -10.0, 0.0)
                zap_log_probs = torch.clamp(zap_log_probs, -10.0, 0.0)
                spinner_log_probs = torch.clamp(spinner_log_probs, -10.0, 0.0)
                
                log_probs = fire_log_probs + zap_log_probs + spinner_log_probs
                if advantages.dim() != log_probs.dim():
                    log_probs = log_probs.sum(dim=1, keepdim=True)
                
                normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                normalized_advantages = torch.clamp(normalized_advantages, -3.0, 3.0)
                actor_loss = -(log_probs * normalized_advantages.detach()).mean()
                
                l2_reg = 0.001 * (actor_model.spinner_head.weight.pow(2).sum() + actor_model.spinner_var_head.weight.pow(2).sum())
                total_loss = actor_loss + l2_reg
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor_model.parameters(), max_norm=1.0)
                actor_optimizer.step()
                actor_losses.append(actor_loss.item())
                
        except Exception as e:
            print(f"RL training error: {e}")
            traceback.print_exc()
            time.sleep(1)

def replay_log_file(log_file_path, bc_model):
    """Train BC model by replaying demonstration log file."""
    if not os.path.exists(log_file_path):
        print(f"Error: Log file {log_file_path} not found")
        return False
    
    print(f"Replaying log file: {log_file_path}")
    frame_count = 0
    batch_size = 128
    training_data = []
    total_loss = 0
    num_batches = 0
    
    total_fire = 0
    total_zap = 0
    total_reward = 0
    
    format_string = ">IdBBBIIBBBhB"
    header_size = struct.calcsize(format_string)
    
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
            
            if num_values != NumberOfParams:
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
            
            action = (fire, zap, spinner)
            training_data.append((state, action, reward))
            
            if len(training_data) >= batch_size:
                batch_loss = train_model_with_batch(bc_model, training_data[:batch_size])
                total_loss += batch_loss
                num_batches += 1
                
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
    old_settings = termios.tcgetattr(fd)
    new_settings = termios.tcgetattr(fd)
    new_settings[3] = new_settings[3] & ~termios.ICANON & ~termios.ECHO
    termios.tcsetattr(fd, termios.TCSANOW, new_settings)
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
    global frame_count, actor, critic, random_control_count, bc_control_count, rl_control_count, total_control_count
    
    bc_model, actor = initialize_models(actor, critic)
    rl_model_lock = threading.Lock()
    
    try:
        old_terminal_settings = setup_keyboard_input()
        print("Keyboard commands enabled: Press 'm' for metrics display (detailed metrics disabled in this version)")
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
    total_episode_reward = 0
    done_latch = False
    current_exploration_ratio = initial_exploration_ratio
    last_spinner_values = deque(maxlen=100)
    extreme_threshold = 0.8
    print_metrics_table_header()
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
                
                if state is None or reward is None or game_action is None or game_mode is None or done is None or is_attract is None:
                    print(f"Warning: Invalid data in processed result, skipping frame.")
                    py_to_lua.write(struct.pack("bbb", 0, 0, 0))
                    py_to_lua.flush()
                    continue
                
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

                if done and not done_latch:
                    done_latch = True
                    total_episode_reward += reward
                    episode_rewards.append(total_episode_reward)
                    last_spinner_values.clear()
                    previous_state = None
                    
                elif done and done_latch:
                    py_to_lua.write(struct.pack("bbb", 0, 0, 0))
                    py_to_lua.flush()
                    continue
                    
                elif not done and done_latch:
                    done_latch = False
                    total_episode_reward = 0
                    previous_state = None
                    
                else:
                    total_episode_reward += reward
                    if not is_attract and previous_state is not None:
                        replay_buffer.append((previous_state, game_action, reward, state, done))

                previous_state = state

                if is_attract and game_action:
                    try:
                        bc_training_queue.put((state, game_action, reward), block=False)
                        action = encode_action(1, 1, -1)
                    except queue.Full:
                        print("BC training queue is full, skipping attract mode action")
                        action = encode_action(0, 0, 0)
                else:
                    if len(last_spinner_values) >= 50:
                        extreme_count = sum(1 for v in last_spinner_values if abs(v) > extreme_threshold)
                        extreme_ratio = extreme_count / len(last_spinner_values)
                        if extreme_ratio > 0.7:
                            current_exploration_ratio = max(current_exploration_ratio, 0.5)
                    
                    if frame_count % 1000 == 0 and current_exploration_ratio > min_exploration_ratio:
                        current_exploration_ratio *= exploration_decay
                    
                    if random.random() < current_exploration_ratio:
                        if random.random() < BC_GUIDED_EXPLORATION_RATIO:
                            with torch.no_grad():
                                bc_model.eval()
                                bc_preds, _ = bc_model(state_tensor)
                                bc_fire = bc_preds[0, 0].item() > 0.5
                                bc_zap = bc_preds[0, 1].item() > 0.5
                                bc_spinner = bc_preds[0, 2].item()
                                noise_scale = 0.3
                                spinner_value = bc_spinner + random.gauss(0, noise_scale)
                                spinner_value = max(-0.95, min(0.95, spinner_value))
                                if random.random() < FIRE_EXPLORATION_RATIO:
                                    bc_fire = 0
                                action = torch.tensor([[float(bc_fire), float(bc_zap), spinner_value]], 
                                                     dtype=torch.float32, device=device)
                                bc_control_count += 1
                                total_control_count += 1
                        else:
                            if random.random() < 0.4:
                                optimal_normalized = 0
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
                            random_control_count += 1
                            total_control_count += 1
                    else:
                        with torch.no_grad():
                            action = actor.get_action(state_tensor, deterministic=True)
                        rl_control_count += 1
                        total_control_count += 1
                
                action_cpu = action.cpu()
                fire, zap, spinner = decode_action(action_cpu)
                last_spinner_values.append(action_cpu[0, 2].item())

                try:
                    sys.stdin.read(1)
                except:
                    pass
                
                py_to_lua.write(struct.pack("bbb", fire, zap, spinner))
                py_to_lua.flush()

                if frame_count % 1000 == 0:
                    actor_loss_mean = np.mean(actor_losses) if actor_losses else float('nan')
                    critic_loss_mean = np.mean(critic_losses) if critic_losses else float('nan')
                    mean_reward = np.mean(list(episode_rewards)) if episode_rewards else float('nan')
                    extreme_ratio = 0.0
                    if last_spinner_values:
                        extreme_count = sum(1 for v in last_spinner_values if abs(v) > extreme_threshold)
                        extreme_ratio = extreme_count / len(last_spinner_values)
                    random_ratio = random_control_count / max(1, total_control_count)
                    bc_ratio = bc_control_count / max(1, total_control_count)
                    rl_ratio = rl_control_count / max(1, total_control_count)
                    spinner_error = np.mean(bc_spinner_error_after) if bc_spinner_error_after else float('nan')
                    
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
                        'buffer_size': len(replay_buffer)
                    }
                    print_metrics_table_row(frame_count, metrics)

                if save_signal:
                    print("\nSaving models...")
                    threading.Thread(target=save_models, args=(actor, bc_model), daemon=True).start()
                    print_metrics_table_header()
                
                frame_count += 1

            except KeyboardInterrupt:
                save_models(actor, bc_model)
                break
            except Exception as e:
                print(f"Error: {e}")
                traceback.print_exc()
                time.sleep(5)
        
        if old_terminal_settings:
            restore_keyboard_input(old_terminal_settings)

def transfer_knowledge_from_bc_to_rl(bc_model, actor_model):
    """Transfer knowledge from BC model to RL model."""
    print("Transferring knowledge from BC model to RL model...")
    with torch.no_grad():
        actor_model.fc1.weight.data.copy_(bc_model.feature_extractor[0].weight.data)
        actor_model.fc1.bias.data.copy_(bc_model.feature_extractor[0].bias.data)
        actor_model.fc2.weight.data.copy_(bc_model.feature_extractor[4].weight.data)
        actor_model.fc2.bias.data.copy_(bc_model.feature_extractor[4].bias.data)
        actor_model.fc3.weight.data.copy_(bc_model.feature_extractor[6].weight.data)
        actor_model.fc3.bias.data.copy_(bc_model.feature_extractor[6].bias.data)
        actor_model.fire_head.weight.data.copy_(bc_model.fire_output.weight.data)
        actor_model.fire_head.bias.data.copy_(bc_model.fire_output.bias.data)
        actor_model.zap_head.weight.data.copy_(bc_model.zap_output.weight.data)
        actor_model.zap_head.bias.data.copy_(bc_model.zap_output.bias.data)
        actor_model.spinner_head.weight.data.copy_(bc_model.spinner_output.weight.data)
        actor_model.spinner_head.bias.data.copy_(bc_model.spinner_output.bias.data)
        actor_model.spinner_var_head.bias.data.fill_(-1.0)
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
    
    if os.path.exists(BC_MODEL_PATH):
        bc_model.eval()
        bc_model.load_state_dict(torch.load(BC_MODEL_PATH, map_location=device))
        print(f"Loaded BC model from {BC_MODEL_PATH}")
        bc_loaded = True
    else:
        bc_loaded = False
        print("No existing BC model found - will train from scratch")
    
    if os.path.exists(LATEST_MODEL_PATH):
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
        actor_model = transfer_knowledge_from_bc_to_rl(bc_model, actor_model)
        transfer_happened = True
    
    if not transfer_happened and bc_loaded and random.random() < 0.3:
        print("Refreshing RL model with BC knowledge (periodic knowledge update)")
        actor_model = transfer_knowledge_from_bc_to_rl(bc_model, actor_model)
    
    return bc_model, actor_model

def save_models(actor_model=None, bc_model=None):
    """Save BC and RL models."""
    try:
        if bc_model is not None:
            with bc_model_lock:
                bc_model.eval()
                torch.save(bc_model.state_dict(), BC_MODEL_PATH)
                print(f"BC model saved to {BC_MODEL_PATH}")
                
        if actor_model is not None:
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
        normalized_spinner = spinner_delta / 31.0
    else:
        sign = np.sign(spinner_delta)
        magnitude = abs(spinner_delta)
        squashed = 10.0 + (magnitude - 10.0) * (1.0 / (1.0 + (magnitude - 10.0) / 20.0))
        normalized_spinner = sign * min(squashed / 31.0, 1.0)
    return torch.tensor([[fire_val, zap_val, normalized_spinner]], dtype=torch.float32, device=device)

def decode_action(action, spinner_power=SPINNER_POWER):
    """Decode actions from PyTorch tensor."""
    fire = 1 if action[0, 0].item() > 0.5 else 0
    zap = 1 if action[0, 1].item() > 0.5 else 0
    spinner_val = action[0, 2].item()
    spinner = int(spinner_val * 31)
    return fire, zap, spinner

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    main()