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
from pathlib import Path
import torch
# Import the actual Metrics class
from metrics import Metrics

# Global flags
IS_INTERACTIVE = sys.stdin.isatty()

# Directory paths
MODEL_DIR = Path("models") # Define as Path object
# Define as Path object using / operator for path joining
LATEST_MODEL_PATH = MODEL_DIR / "tempest_model_latest.pt"

# -- Determine PyTorch Device --
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else # Check for Apple Silicon GPU
    "cpu"
)
print(f"[Config] Using PyTorch device: {DEVICE}")

@dataclass
class ServerConfigData:
    host: str = "0.0.0.0"
    port: int = 9999
    max_clients: int = 36
    params_count: int = 311
    expert_ratio_start: float = 1.0
    expert_ratio_min: float = 0.0
    expert_ratio_decay: float = 0.995  
    expert_ratio_decay_steps: int = 100000
    reset_frame_count: bool = False
    reset_expert_ratio: bool = True
    cpu_inference: bool = False


# Create instance of ServerConfigData first
SERVER_CONFIG = ServerConfigData()

@dataclass
class RLConfigData:
    """Configuration for reinforcement learning"""
    state_size: int = SERVER_CONFIG.params_count  # Use value from ServerConfigData
    action_size: int = 15  # Number of possible actions (from ACTION_MAPPING)
    batch_size: int = 512
    gamma: float = 0.99
    epsilon: float = 1.0
    epsilon_start: float = 0.75
    epsilon_end: float = 0.001
    epsilon_min: float = 0.001
    epsilon_decay_rate: float = 0.995
    decay_epsilon_frames: int = 100000
    learning_rate: float = 1e-4
    memory_size: int = 200000
    save_interval: int = 500000
    save_interval_seconds: int = 300
    train_freq: int = 4
    target_update: int = 10000
    min_buffer_size: int = 10000
    train_queue_size: int = 20
    loss_queue_size: int = 100
    device_preference: str = "mps"

# Create instance of RLConfigData after its definition
RL_CONFIG = RLConfigData()

# Create the global metrics instance using the imported class
metrics = Metrics()

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

# Determine the primary device
def get_primary_device():
    if RL_CONFIG.device_preference == "cuda" and torch.cuda.is_available():
        return "cuda"
    elif RL_CONFIG.device_preference == "mps" and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

DEVICE = get_primary_device() # Global device for training/main agent
