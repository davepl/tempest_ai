#!/usr/bin/env python3
"""
Tempest AI Model: Expert-guided gameplay system with infrastructure for future RL integration.
- Makes intelligent decisions based on enemy positions and level types
- Communicates with Tempest via Lua pipes
- Maintains hooks for future RL model integration
"""

import os
import time
import struct
import random
import numpy as np
import torch
import select
import sys
import traceback
from collections import deque

# Constants
NumberOfParams = 128
EXPERT_GUIDANCE_RATIO = 1.0  # Always use expert guidance for now
LUA_TO_PY_PIPE = "/tmp/lua_to_py"
PY_TO_LUA_PIPE = "/tmp/py_to_lua"

# Setup
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
LATEST_MODEL_PATH = f"{MODEL_DIR}/tempest_model_latest.zip"
device = torch.device("cuda" if torch.cuda.is_available() else 
                      "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device.type.upper()}")

# Tracking metrics
frame_count = 0
guided_control_count = 0
total_control_count = 0
episode_rewards = deque(maxlen=5)

def process_frame_data(data):
    """Process binary frame data from Lua script"""
    if not data:
        return None
    
    format_string = ">IdBBBIIBBBhBhBB"  # Includes OOB data with is_open_level flag
    try:
        header_size = struct.calcsize(format_string)
        if len(data) < header_size:
            print(f"Data too short: {len(data)} < {header_size}")
            return None
        
        # Unpack OOB data
        values = struct.unpack(format_string, data[:header_size])
        num_values, reward, game_action, game_mode, done, frame_counter, score, save_signal, fire, zap, spinner, is_attract, nearest_enemy_segment, player_segment, is_open_level = values
        
        # Process game state data
        game_data_bytes = data[header_size:]
        state_values = []
        for i in range(0, len(game_data_bytes), 2):
            if i + 1 < len(game_data_bytes):
                value = struct.unpack(">H", game_data_bytes[i:i+2])[0]
                normalized = value - 32768
                state_values.append(normalized)
        
        # Create and normalize state array
        state = np.array(state_values, dtype=np.float32) / 32768.0
        if len(state) > NumberOfParams:
            state = state[:NumberOfParams]
        elif len(state) < NumberOfParams:
            state = np.pad(state, (0, NumberOfParams - len(state)), 'constant')
        
        # Convert numeric values to appropriate types
        is_open_level = bool(is_open_level)
        game_action_tuple = (bool(fire), bool(zap), spinner)
        
        return state, reward, game_action_tuple, game_mode, bool(done), bool(is_attract), save_signal, nearest_enemy_segment, player_segment, is_open_level
        
    except Exception as e:
        print(f"ERROR unpacking data: {e}")
        traceback.print_exc()
        return None

def print_metrics_table_header():
    """Print header for metrics table"""
    header = f"{'Frame':>8} | {'Mean Reward':>12} | {'Guided %':>8} | {'Nearest Enemy':>15} | {'Level Type':>10}"
    separator = "-" * len(header)
    print("\n" + separator)
    print(header)
    print(separator)

def print_metrics_table_row(frame_count, metrics_dict):
    """Print row for metrics table"""
    row = (
        f"{frame_count:8d} | {metrics_dict.get('mean_reward', 'N/A'):12.2f} | "
        f"{metrics_dict.get('guided_ratio', 0.0)*100:7.2f}% | "
        f"{metrics_dict.get('nearest_enemy', -1):15d} | "
        f"{'Open' if metrics_dict.get('is_open_level', False) else 'Closed':10}"
    )
    print(row)

def expert_decision(nearest_enemy_segment, player_segment, is_open_level=False):
    """Make an expert-guided decision based on game state"""
    # Handle special cases first
    if nearest_enemy_segment < 0:
        # No enemies - hold position and fire
        return 1, 0, 0  # fire, zap, spinner
        
    elif nearest_enemy_segment == player_segment:
        # Aligned with enemy - fire but don't move
        return 1, 0, 0  # fire, zap, spinner
        
    # Need to move toward enemy
    if is_open_level:
        # Open level logic (no wraparound)
        distance = abs(nearest_enemy_segment - player_segment)
        intensity = min(0.9, 0.3 + (distance * 0.05))
        
        # Direction depends on relative positions
        if nearest_enemy_segment > player_segment:
            # Enemy is to the right, move clockwise (negative spinner)
            spinner_value = -intensity
        else:
            # Enemy is to the left, move counterclockwise (positive spinner)
            spinner_value = intensity
            
    else:
        # Closed level logic (wraparound)
        # Calculate distances in both directions
        if nearest_enemy_segment > player_segment:
            clockwise_distance = nearest_enemy_segment - player_segment
            counterclockwise_distance = player_segment + (16 - nearest_enemy_segment)
        else:
            clockwise_distance = nearest_enemy_segment + (16 - player_segment)
            counterclockwise_distance = player_segment - nearest_enemy_segment
        
        # Choose shortest path
        min_distance = min(clockwise_distance, counterclockwise_distance)
        intensity = min(0.9, 0.3 + (min_distance * 0.05))
        
        if clockwise_distance < counterclockwise_distance:
            # Move clockwise (negative spinner)
            spinner_value = -intensity
        else:
            # Move counterclockwise (positive spinner)
            spinner_value = intensity
            
    # Always fire when tracking enemies
    return 1, 0, spinner_value  # fire, zap, spinner

def save_model(model_data=None):
    """Placeholder for future model saving functionality"""
    print(f"Model saving functionality will be implemented with future RL models")

def load_model():
    """Placeholder for future model loading functionality"""
    print(f"Model loading functionality will be implemented with future RL models")
    return None

def encode_action(fire, zap, spinner_delta):
    """Encode actions for future RL model"""
    fire_val = float(fire)
    zap_val = float(zap)
    
    # Normalize spinner delta to [-1, 1] range with non-linear scaling for precision
    if abs(spinner_delta) < 10:
        normalized_spinner = spinner_delta / 31.0
    else:
        sign = np.sign(spinner_delta)
        magnitude = abs(spinner_delta)
        squashed = 10.0 + (magnitude - 10.0) * (1.0 / (1.0 + (magnitude - 10.0) / 20.0))
        normalized_spinner = sign * min(squashed / 31.0, 1.0)
        
    return torch.tensor([[fire_val, zap_val, normalized_spinner]], dtype=torch.float32, device=device)

def decode_action(action):
    """Decode actions from model output"""
    if isinstance(action, torch.Tensor):
        fire = 1 if action[0, 0].item() > 0.5 else 0
        zap = 1 if action[0, 1].item() > 0.5 else 0
        spinner_val = action[0, 2].item()
    else:
        # Handle non-tensor actions (like from expert system)
        fire, zap, spinner_val = action
        
    # Convert normalized spinner to actual game values with improved precision
    if abs(spinner_val) < 0.3:
        # More precise control for small adjustments
        spinner = int(spinner_val * 20)
    else:
        # Full range for larger movements
        spinner = int(spinner_val * 31)
    
    return fire, zap, spinner

def main():
    """Main function that handles communication with Lua and decision making"""
    global frame_count, guided_control_count, total_control_count
    
    # Create pipes
    for pipe in [LUA_TO_PY_PIPE, PY_TO_LUA_PIPE]:
        if os.path.exists(pipe): 
            os.unlink(pipe)
        os.mkfifo(pipe)
        os.chmod(pipe, 0o666)

    print("Pipes created. Waiting for Lua...")
    total_episode_reward = 0
    done_latch = False
    print_metrics_table_header()
    
    # Main loop connecting to Lua
    with os.fdopen(os.open(LUA_TO_PY_PIPE, os.O_RDONLY | os.O_NONBLOCK), "rb") as lua_to_py, \
         open(PY_TO_LUA_PIPE, "wb") as py_to_lua:
        print("✔️ Lua connected.")
        
        while True:
            try:
                # Wait for data from Lua
                ready_to_read, _, _ = select.select([lua_to_py], [], [], 0.01)
                if not ready_to_read:
                    time.sleep(0.001)
                    continue
                
                data = lua_to_py.read()
                if not data:
                    time.sleep(0.001)
                    continue

                # Process frame data from Lua
                result = process_frame_data(data)
                if result is None:
                    # Send null action on error
                    py_to_lua.write(struct.pack("bbb", 0, 0, 0))
                    py_to_lua.flush()
                    continue
                
                # Unpack frame data
                state, reward, game_action, game_mode, done, is_attract, save_signal, nearest_enemy_segment, player_segment, is_open_level = result
                
                if state is None or reward is None or game_action is None or game_mode is None or done is None or is_attract is None:
                    # Send null action on invalid data
                    py_to_lua.write(struct.pack("bbb", 0, 0, 0))
                    py_to_lua.flush()
                    continue
                
                # Episode management
                if done and not done_latch:
                    # Episode just ended
                    done_latch = True
                    total_episode_reward += reward
                    episode_rewards.append(total_episode_reward)
                    
                elif done and done_latch:
                    # Still in done state, send null action
                    py_to_lua.write(struct.pack("bbb", 0, 0, 0))
                    py_to_lua.flush()
                    continue
                    
                elif not done and done_latch:
                    # New episode starting
                    done_latch = False
                    total_episode_reward = 0
                    
                else:
                    # Normal gameplay
                    total_episode_reward += reward

                # Get expert guidance
                fire_value, zap_value, spinner_value = expert_decision(nearest_enemy_segment, player_segment, is_open_level)
                guided_control_count += 1
                total_control_count += 1

                # Convert decision to game controls
                fire, zap, spinner = decode_action((fire_value, zap_value, spinner_value))

                # Send controls to Lua
                py_to_lua.write(struct.pack("bbb", fire, zap, spinner))
                py_to_lua.flush()

                # Print metrics periodically
                if frame_count % 1000 == 0:
                    mean_reward = np.mean(list(episode_rewards)) if episode_rewards else float('nan')
                    guided_ratio = guided_control_count / max(1, total_control_count)
                    
                    metrics = {
                        'mean_reward': mean_reward,
                        'guided_ratio': guided_ratio,
                        'nearest_enemy': nearest_enemy_segment,
                        'is_open_level': is_open_level
                    }
                    print_metrics_table_row(frame_count, metrics)

                if save_signal:
                    print("\nSave signal received (placeholder for future model saving)")
                    print_metrics_table_header()
                
                frame_count += 1

            except KeyboardInterrupt:
                print("\nKeyboard interrupt detected. Exiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
                traceback.print_exc()
                time.sleep(1)

if __name__ == "__main__":
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    main()