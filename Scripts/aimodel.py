#!/usr/bin/env python3
"""
Tempest AI Model: Uses expert-guided exploration with Reinforcement Learning (RL) in gameplay.
- Expert guidance provides useful moves; RL optimizes during player control.
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
FORCE_CPU = False
initial_exploration_ratio = 0.5
min_exploration_ratio = 0.05
exploration_decay = 0.99
AIMING_GUIDED_EXPLORATION_RATIO = 0.9
FIRE_EXPLORATION_RATIO = 0.05
NumberOfParams = 128

LUA_TO_PY_PIPE = "/tmp/lua_to_py"
PY_TO_LUA_PIPE = "/tmp/py_to_lua"

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
LATEST_MODEL_PATH = f"{MODEL_DIR}/tempest_model_latest.zip"

# Parse command line arguments
parser = argparse.ArgumentParser(description='Tempest AI Model')
args = parser.parse_args()

# Device selection
if FORCE_CPU:
    device = torch.device("cpu")
    print("Using device: CPU (forced)")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device.type.upper()}")

# Training metrics
actor_losses = deque(maxlen=100)
critic_losses = deque(maxlen=100)
replay_buffer = deque(maxlen=100000)
episode_rewards = deque(maxlen=5)

# Control source tracking
random_control_count = 0
guided_control_count = 0
rl_control_count = 0
total_control_count = 0

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
        x = F.relu(x)  # Fixed from 'najczęściej(x)'
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
    if not data:
        return None
    
    format_string = ">IdBBBIIBBBhBhB"
    try:
        header_size = struct.calcsize(format_string)
        if len(data) < header_size:
            print(f"Data too short: {len(data)} < {header_size}")
            return None
        
        values = struct.unpack(format_string, data[:header_size])
        num_values, reward, game_action, game_mode, done, frame_counter, score, save_signal, fire, zap, spinner, is_attract, nearest_enemy_segment, player_segment = values
        
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
        return state, reward, game_action_tuple, game_mode, bool(done), bool(is_attract), save_signal, nearest_enemy_segment, player_segment
        
    except Exception as e:
        print(f"ERROR unpacking data: {e}")
        traceback.print_exc()
        return None

def background_rl_train(rl_model_lock, actor_model, critic_model):
    gamma = 0.99
    batch_size = 1024
    
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

                # Critic training
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

                # Actor training
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

frame_count = 0

def setup_keyboard_input():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    new_settings = termios.tcgetattr(fd)
    new_settings[3] = new_settings[3] & ~termios.ICANON & ~termios.ECHO
    termios.tcsetattr(fd, termios.TCSANOW, new_settings)
    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
    return old_settings

def restore_keyboard_input(old_settings):
    termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, old_settings)

def print_metrics_table_header():
    header = (
        f"{'Frame':>8} | {'Actor Loss':>10} | {'Critic Loss':>11} | {'Mean Reward':>12} | {'Explore %':>9} | "
        f"{'Random %':>8} | {'Guided %':>8} | {'RL %':>8} | {'Extreme %':>9} | {'Buffer':>7}"
    )
    separator = "-" * len(header)
    print("\n" + separator)
    print(header)
    print(separator)

def print_metrics_table_row(frame_count, metrics_dict):
    row = (
        f"{frame_count:8d} | {metrics_dict.get('actor_loss', 'N/A'):10.4f} | "
        f"{metrics_dict.get('critic_loss', 'N/A'):11.2f} | {metrics_dict.get('mean_reward', 'N/A'):12.2f} | "
        f"{metrics_dict.get('exploration', 0.0)*100:8.2f}% | "
        f"{metrics_dict.get('random_ratio', 0.0)*100:7.2f}% | {metrics_dict.get('guided_ratio', 0.0)*100:7.2f}% | "
        f"{metrics_dict.get('rl_ratio', 0.0)*100:7.2f}% | "
        f"{metrics_dict.get('extreme_ratio', 0.0)*100:8.2f}% | {metrics_dict.get('buffer_size', 0):7d}"
    )
    print(row)

def main():
    global frame_count, actor, critic, random_control_count, guided_control_count, rl_control_count, total_control_count
    
    rl_model_lock = threading.Lock()
    threading.Thread(target=background_rl_train, args=(rl_model_lock, actor, critic), daemon=True).start()

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
                
                state, reward, game_action, game_mode, done, is_attract, save_signal, nearest_enemy_segment, player_segment = result
                
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
                        replay_buffer.append((previous_state, action.cpu().numpy()[0], reward, state, done))

                previous_state = state

                if len(last_spinner_values) >= 50:
                    extreme_count = sum(1 for v in last_spinner_values if abs(v) > extreme_threshold)
                    extreme_ratio = extreme_count / len(last_spinner_values)
                    if extreme_ratio > 0.7:
                        current_exploration_ratio = max(current_exploration_ratio, 0.5)
                
                if frame_count % 1000 == 0 and current_exploration_ratio > min_exploration_ratio:
                    current_exploration_ratio *= exploration_decay
                
                if random.random() < current_exploration_ratio:
                    if random.random() < AIMING_GUIDED_EXPLORATION_RATIO:
                        if nearest_enemy_segment == player_segment or nearest_enemy_segment < 0:
                            spinner_value = 0
                        elif nearest_enemy_segment < player_segment:
                            spinner_value = 0.35
                        else:
                            spinner_value = -0.35
                        action = torch.tensor([[1.0, 0.0, spinner_value]], dtype=torch.float32, device=device)
                        guided_control_count += 1
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
                    guided_ratio = guided_control_count / max(1, total_control_count)
                    rl_ratio = rl_control_count / max(1, total_control_count)
                    
                    metrics = {
                        'actor_loss': actor_loss_mean,
                        'critic_loss': critic_loss_mean,
                        'mean_reward': mean_reward,
                        'exploration': current_exploration_ratio,
                        'random_ratio': random_ratio,
                        'guided_ratio': guided_ratio,
                        'rl_ratio': rl_ratio,
                        'extreme_ratio': extreme_ratio,
                        'buffer_size': len(replay_buffer)
                    }
                    print_metrics_table_row(frame_count, metrics)

                if save_signal:
                    print("\nSaving model...")
                    threading.Thread(target=save_model, args=(actor,), daemon=True).start()
                    print_metrics_table_header()
                
                frame_count += 1

            except KeyboardInterrupt:
                save_model(actor)
                break
            except Exception as e:
                print(f"Error: {e}")
                traceback.print_exc()
                time.sleep(5)

def save_model(actor_model):
    try:
        actor_model.eval()
        torch.save(actor_model.state_dict(), LATEST_MODEL_PATH)
        print(f"Actor model saved to {LATEST_MODEL_PATH}")
    except Exception as e:
        print(f"Error saving model: {e}")
        traceback.print_exc()

def encode_action(fire, zap, spinner_delta):
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

def decode_action(action):
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