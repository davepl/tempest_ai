#!/usr/bin/env python3
"""
Metrics display for Tempest AI.
"""

# Prevent direct execution
if __name__ == "__main__":
    print("This is not the main application, run 'main.py' instead")
    exit(1)

import os
import sys
import time
import threading
import numpy as np
from typing import Optional, List, Dict, Any

# Import from config.py
from config import metrics, IS_INTERACTIVE

# Add a counter to track the number of rows printed
row_counter = 0

def clear_screen():
    """Clear the screen and move cursor to home position"""
    if IS_INTERACTIVE:
        # Clear screen and move to home
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()

def print_metrics_line(message, is_header=False):
    """Print a metrics line with proper formatting"""
    global row_counter
    
    if IS_INTERACTIVE:
        if is_header:
            # For header: print at current position
            print(message)
            print("-" * len(message))  # Add separator line
            # Reset row counter when header is printed
            row_counter = 0
        else:
            # For rows: just print at current position
            print(message)
            # Increment row counter
            row_counter += 1
        sys.stdout.flush()
    else:
        print(message)

def display_metrics_header():
    """Display the header for metrics output"""
    # Clear screen first
    # clear_screen()
    
    # Print header (Increase Frame width, Add TrainQ)
    header = (
        f"{'Frame':>11} {'FPS':>6} {'Epsilon':>8} {'Expert%':>8} "
        f"{'Mean Reward':>12} {'DQN Reward':>12} {'Loss':>10} "
        f"{'Clients':>8} {'Override':>9} {'Expert Mode':>11} "
        f"{'TrainQ':>7} {'InfTime(ms)':>10}" # Added InfTime column
    )
    print_metrics_line(header, is_header=True)
    # Print an empty line after header
    print()

def display_metrics_row(agent, kb_handler):
    """Display a row of metrics data"""
    global row_counter
    
    # Check if we need to print the header (every 30th row)
    if row_counter > 0 and row_counter % 30 == 0:
        display_metrics_header()
    
    # Calculate mean reward from the last 100 episodes
    mean_reward = 0
    if metrics.episode_rewards:
        rewards_list = list(metrics.episode_rewards)
        mean_reward = sum(rewards_list[-100:]) / min(len(rewards_list), 100)
    
    # Calculate mean DQN reward
    mean_dqn_reward = 0
    if metrics.dqn_rewards:
        dqn_rewards_list = list(metrics.dqn_rewards)
        mean_dqn_reward = sum(dqn_rewards_list[-100:]) / min(len(dqn_rewards_list), 100)
    
    # Get the latest loss value
    latest_loss = metrics.losses[-1] if metrics.losses else 0
    
    # Get Training Queue Size
    train_q_size = 0
    if agent and hasattr(agent, 'train_queue'):
        train_q_size = agent.train_queue.qsize()
    
    # Get Average DQN Inference Time (calculated in stats_reporter)
    avg_inf_time_ms = metrics.avg_dqn_inf_time

    # Format the row (Add InfTime)
    row = (
        f"{metrics.frame_count:>11,} {metrics.fps:>6.1f} {metrics.epsilon:>8.4f} " # Add comma for thousands
        f"{metrics.expert_ratio*100:>7.1f}% {mean_reward:>12.2f} {mean_dqn_reward:>12.2f} "
        f"{latest_loss:>10.2f} {metrics.client_count:>8} "
        f"{'ON' if metrics.override_expert else 'OFF':>9} "
        f"{'ON' if metrics.expert_mode else 'OFF':>11} "
        f"{train_q_size:>7} {avg_inf_time_ms:>10.2f}" # Added Avg Inf Time value
    )
    
    print_metrics_line(row)

def run_stats_reporter(metrics):
    """Run the stats reporter in a loop"""
    display_metrics_header()
    
    while True:
        try:
            time.sleep(1)  # Update every second
            display_metrics_row(None, None)
        except Exception as e:
            print(f"Error in stats reporter: {e}")
            time.sleep(1)  # Wait a bit before retrying 