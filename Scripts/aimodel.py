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
from collections import deque

# Configuration constants
PIPE_CONFIG = {
    "lua_to_py": "/tmp/lua_to_py",
    "py_to_lua": "/tmp/py_to_lua",
    "params_count": 128,
    "expert_ratio": 1.0  # 100% expert guidance currently
}
MODEL_DIR = "models"
LATEST_MODEL_PATH = f"{MODEL_DIR}/tempest_model_latest.zip"

# Initialize device and metrics
device = torch.device("cuda" if torch.cuda.is_available() else 
                      "mps" if torch.backends.mps.is_available() else "cpu")
metrics = {
    "frame_count": 0,
    "guided_count": 0,
    "total_controls": 0,
    "episode_rewards": deque(maxlen=5)
}

def setup_environment():
    """Set up pipes and model directory"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    for pipe in [PIPE_CONFIG["lua_to_py"], PIPE_CONFIG["py_to_lua"]]:
        if os.path.exists(pipe):
            os.unlink(pipe)
        os.mkfifo(pipe)
        os.chmod(pipe, 0o666)

def parse_frame_data(data):
    """Parse binary frame data from Lua into game state"""
    if not data:
        return None
    
    # Expected format: timestamp, reward, actions, mode, done, counters, etc.
    format_str = ">IdBBBIIBBBhBhBB"
    header_size = struct.calcsize(format_str)
    
    if len(data) < header_size:
        return None
        
    # Extract header values
    values = struct.unpack(format_str, data[:header_size])
    num_values, reward, game_action, game_mode, done, frame_counter, score, \
    save_signal, fire, zap, spinner, is_attract, nearest_enemy, player_seg, is_open = values
    
    # Process remaining game state data
    state_data = data[header_size:]
    state_values = [
        struct.unpack(">H", state_data[i:i+2])[0] - 32768
        for i in range(0, len(state_data), 2)
        if i + 1 < len(state_data)
    ]
    
    # Normalize and pad/truncate state array
    state = np.array(state_values, dtype=np.float32) / 32768.0
    state = (state[:PIPE_CONFIG["params_count"]] if len(state) > PIPE_CONFIG["params_count"]
            else np.pad(state, (0, PIPE_CONFIG["params_count"] - len(state))))
            
    return {
        "state": state,
        "reward": reward,
        "action": (bool(fire), bool(zap), spinner),
        "mode": game_mode,
        "done": bool(done),
        "attract": bool(is_attract),
        "save_signal": save_signal,
        "enemy_seg": nearest_enemy,
        "player_seg": player_seg,
        "open_level": bool(is_open)
    }

def display_metrics_header():
    """Display header for metrics table"""
    header = f"{'Frame':>8} | {'Mean Reward':>12} | {'Guided %':>8} | {'Nearest Enemy':>15} | {'Level Type':>10}"
    print(f"\n{'-' * len(header)}\n{header}\n{'-' * len(header)}")

def display_metrics_row():
    """Display current metrics in tabular format"""
    mean_reward = np.mean(list(metrics["episode_rewards"])) if metrics["episode_rewards"] else float('nan')
    guided_ratio = metrics["guided_count"] / max(1, metrics["total_controls"])
    
    row = (
        f"{metrics['frame_count']:8d} | {mean_reward:12.2f} | "
        f"{guided_ratio*100:7.2f}% | {metrics.get('enemy_seg', -1):15d} | "
        f"{'Open' if metrics.get('open_level', False) else 'Closed':10}"
    )
    print(row)

def get_expert_action(enemy_seg, player_seg, is_open_level):
    """Calculate expert-guided action based on game state"""
    if enemy_seg < 0 or enemy_seg == player_seg:
        return 1, 0, 0  # Fire only when no enemies or aligned
        
    # Calculate movement based on level type
    if is_open_level:
        distance = abs(enemy_seg - player_seg)
        intensity = min(0.9, 0.3 + (distance * 0.05))
        spinner = -intensity if enemy_seg > player_seg else intensity
    else:
        # Calculate shortest path with wraparound
        clockwise = (enemy_seg - player_seg) % 16
        counter = (player_seg - enemy_seg) % 16
        min_dist = min(clockwise, counter)
        intensity = min(0.9, 0.3 + (min_dist * 0.05))
        spinner = -intensity if clockwise < counter else intensity
    
    return 1, 0, spinner  # Fire while moving

def encode_action_to_game(fire, zap, spinner):
    """Convert action values to game-compatible format"""
    spinner_val = spinner * (20 if abs(spinner) < 0.3 else 31)
    return int(fire), int(zap), int(spinner_val)

def main():
    """Main game loop handling Lua communication and decisions"""
    setup_environment()
    display_metrics_header()
    total_reward = 0
    was_done = False
    
    with os.fdopen(os.open(PIPE_CONFIG["lua_to_py"], os.O_RDONLY | os.O_NONBLOCK), "rb") as lua_in, \
         open(PIPE_CONFIG["py_to_lua"], "wb") as lua_out:
         
        while True:
            # Check for incoming data
            if not select.select([lua_in], [], [], 0.01)[0]:
                time.sleep(0.001)
                continue
                
            data = lua_in.read()
            if not data:
                time.sleep(0.001)
                continue
                
            frame = parse_frame_data(data)
            if not frame:
                lua_out.write(struct.pack("bbb", 0, 0, 0))
                lua_out.flush()
                continue
                
            # Update episode tracking
            if frame["done"]:
                if not was_done:
                    total_reward += frame["reward"]
                    metrics["episode_rewards"].append(total_reward)
                was_done = True
                lua_out.write(struct.pack("bbb", 0, 0, 0))
                lua_out.flush()
                continue
            elif was_done:
                was_done = False
                total_reward = 0
            total_reward += frame["reward"]
            
            # Store frame data for metrics
            metrics.update({
                "enemy_seg": frame["enemy_seg"],
                "open_level": frame["open_level"]
            })
            
            # Generate and send action
            action = get_expert_action(frame["enemy_seg"], frame["player_seg"], frame["open_level"])
            metrics["guided_count"] += 1
            metrics["total_controls"] += 1
            fire, zap, spinner = encode_action_to_game(*action)
            lua_out.write(struct.pack("bbb", fire, zap, spinner))
            lua_out.flush()
            
            # Update metrics display
            metrics["frame_count"] += 1
            if metrics["frame_count"] % 1000 == 0:
                display_metrics_row()
            if frame["save_signal"]:
                print("\nSave signal received (future model saving placeholder)")
                display_metrics_header()

if __name__ == "__main__":
    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting gracefully...")