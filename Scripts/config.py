#!/usr/bin/env python3
"""
Configuration and shared types for Tempest AI.
"""

# Prevent direct execution
if __name__ == "__main__":
    print("This is not the main application, run 'main.py' instead")
    exit(1)

import os
import sys
import time
import threading
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple
from collections import deque

# Global flags
IS_INTERACTIVE = sys.stdin.isatty()

# Flag to control metric reset on load
RESET_METRICS = False # Set to True to ignore saved epsilon/expert ratio

# Directory paths
MODEL_DIR = "models"
LATEST_MODEL_PATH = f"{MODEL_DIR}/tempest_model_latest.pt"

@dataclass
class ServerConfigData:
    """Configuration for socket server"""
    host: str = "0.0.0.0"  # Listen on all interfaces
    port: int = 9999
    max_clients: int = 36
    params_count: int = 359
    expert_ratio_start: float = 1.0
    expert_ratio_min: float = 0.0
    expert_ratio_decay: float = 0.999
    expert_ratio_decay_steps: int = 10000
    reset_frame_count: bool = False
    reset_expert_ratio: bool = True

# Create instance of ServerConfigData first
SERVER_CONFIG = ServerConfigData()

@dataclass
class RLConfigData:
    """Configuration for reinforcement learning"""
    state_size: int = SERVER_CONFIG.params_count  # Use value from ServerConfigData
    action_size: int = 15  # Number of possible actions (from ACTION_MAPPING)
    batch_size: int = 1024
    gamma: float = 0.99
    epsilon: float = 1.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_min: float = 0.001
    epsilon_decay_factor: float = 0.9969 # Multiplicative factor per step (Adjusted for ~15M frames)
    epsilon_decay_steps: int = 10000   # Frames per decay step
    update_target_every: int = 500
    learning_rate: float = 0.001
    memory_size: int = 200000
    save_interval: int = 50000
    train_freq: int = 4
    target_update: int = 10000

# Create instance of RLConfigData after its definition
RL_CONFIG = RLConfigData()

@dataclass
class MetricsData:
    """Metrics tracking for training progress"""
    frame_count: int = 0
    guided_count: int = 0
    total_controls: int = 0
    episode_rewards: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    dqn_rewards: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    expert_rewards: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    losses: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    epsilon: float = 1.0
    expert_ratio: float = 1.0
    last_decay_step: int = 0
    last_epsilon_decay_step: int = 0 # Added tracker for epsilon decay
    enemy_seg: int = -1
    open_level: bool = False
    override_expert: bool = False
    saved_expert_ratio: float = 0.75
    expert_mode: bool = False
    last_action_source: str = ""
    frames_last_second: int = 0
    last_fps_time: float = 0
    fps: float = 0.0
    client_count: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)
    total_inference_time: float = 0.0
    total_inference_requests: int = 0
    
    def update_frame_count(self):
        """Update frame count and FPS tracking"""
        with self.lock:
            # Update total frame count
            self.frame_count += 1
            
            # Update FPS tracking
            current_time = time.time()
            
            # Initialize last_fps_time if this is the first frame
            if self.last_fps_time == 0:
                self.last_fps_time = current_time
                
            # Count frames for this second
            self.frames_last_second += 1
            
            # Calculate FPS every second
            elapsed = current_time - self.last_fps_time
            if elapsed >= 1.0:
                # Calculate frames per second with more accuracy
                new_fps = self.frames_last_second / elapsed
                
                # Store the new FPS value
                self.fps = new_fps
                
                # Reset counters
                self.frames_last_second = 0
                self.last_fps_time = current_time
                
            return self.frame_count
    
    def get_epsilon(self):
        """Get current epsilon value"""
        with self.lock:
            return self.epsilon
    
    def update_epsilon(self):
        """Update epsilon based on frame count"""
        with self.lock:
            # Import here to avoid circular imports
            from aimodel import decay_epsilon
            self.epsilon = decay_epsilon(self.frame_count)
            return self.epsilon
    
    def update_expert_ratio(self):
        """Update expert ratio based on frame count"""
        with self.lock:
            # Import here to avoid circular imports
            from aimodel import decay_expert_ratio
            # Skip decay if expert mode is active
            if self.expert_mode:
                return self.expert_ratio
            decay_expert_ratio(self.frame_count)
            return self.expert_ratio
    
    def add_episode_reward(self, total_reward, dqn_reward, expert_reward):
        """Add episode rewards to tracking"""
        with self.lock:
            if total_reward > 0:
                self.episode_rewards.append(total_reward)
            if dqn_reward > 0:
                self.dqn_rewards.append(dqn_reward)
            if expert_reward > 0:
                self.expert_rewards.append(expert_reward)
    
    def increment_guided_count(self):
        """Increment guided count"""
        with self.lock:
            self.guided_count += 1
    
    def increment_total_controls(self):
        """Increment total controls"""
        with self.lock:
            self.total_controls += 1
    
    def update_action_source(self, source):
        """Update last action source"""
        with self.lock:
            self.last_action_source = source
    
    def update_game_state(self, enemy_seg, open_level):
        """Update game state"""
        with self.lock:
            self.enemy_seg = enemy_seg
            self.open_level = open_level
    
    def get_expert_ratio(self):
        """Get current expert ratio"""
        with self.lock:
            return self.expert_ratio
    
    def is_override_active(self):
        """Check if override is active"""
        with self.lock:
            return self.override_expert
    
    def get_fps(self):
        """Get current FPS"""
        with self.lock:
            return self.fps
    
    def toggle_override(self, kb_handler=None):
        """Toggle override mode"""
        with self.lock:
            self.override_expert = not self.override_expert
            if self.override_expert:
                self.saved_expert_ratio = self.expert_ratio
                self.expert_ratio = 0.0
            else:
                self.expert_ratio = self.saved_expert_ratio
            if kb_handler and IS_INTERACTIVE:
                # Import here to avoid circular import at top level
                from aimodel import print_with_terminal_restore
                print_with_terminal_restore(kb_handler, f"\nOverride mode: {'ON' if self.override_expert else 'OFF'}\r")
    
    def toggle_expert_mode(self, kb_handler=None):
        """Toggle expert mode"""
        with self.lock:
            self.expert_mode = not self.expert_mode
            if self.expert_mode:
                # Save current expert ratio and set to 1.0 (100%) when expert mode is ON
                self.saved_expert_ratio = self.expert_ratio
                self.expert_ratio = 1.0
            else:
                # Restore the saved expert ratio when expert mode is OFF
                self.expert_ratio = self.saved_expert_ratio
            if kb_handler and IS_INTERACTIVE:
                # Import here to avoid circular import at top level
                from aimodel import print_with_terminal_restore
                print_with_terminal_restore(kb_handler, f"\nExpert mode: {'ON' if self.expert_mode else 'OFF'}\r")

# Define action space
ACTION_MAPPING = {
    0: (0, 0, -0.3),   # Hard left, no fire, no zap
    1: (0, 0, -0.2),   # Medium left, no fire, no zap
    2: (0, 0, -0.1),   # Soft left, no fire, no zap
    3: (0, 0, 0.0),    # Center, no fire, no zap
    4: (0, 0, 0.1),    # Soft right, no fire, no zap
    5: (0, 0, 0.2),    # Medium right, no fire, no zap
    6: (0, 0, 0.3),    # Hard right, no fire, no zap
    7: (1, 0, -0.3),   # Hard left, fire, no zap
    8: (1, 0, -0.2),   # Medium left, fire, no zap
    9: (1, 0, -0.1),   # Soft left, fire, no zap
    10: (1, 0, 0.0),   # Center, fire, no zap
    11: (1, 0, 0.1),   # Soft right, fire, no zap
    12: (1, 0, 0.2),   # Medium right, fire, no zap
    13: (1, 0, 0.3),   # Hard right, fire, no zap
    14: (1, 1, 0.0),   # Zap+Fire+Sit
}

# Create instances of config classes
metrics = MetricsData()

# # Import print_with_terminal_restore from metrics_display to avoid circular imports
# # # DEF print_with_terminal_restore(kb_handler, *args, **kwargs):
# # #     \"\"\"Print with terminal restore if in interactive mode\"\"\"
# # #     if IS_INTERACTIVE and kb_handler:
# # #         # Import here to avoid circular imports
# # #         from metrics_display import print_with_terminal_restore as _print
# # #         _print(*args, **kwargs)
# # #     else:
# # #         print(*args, **kwargs) 
