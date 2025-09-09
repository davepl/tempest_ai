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
    
    # Print header (Remove Open/Closed level column)
    header = (
        f"{'Frame':>11} {'FPS':>6} {'Epsilon':>8} {'Expert%':>8} "
        f"{'Mean Reward':>12} {'DQN Reward':>12} {'Loss':>10} "
        f"{'Clients':>8} {'Avg Level':>10} {'Override':>9} {'Expert Mode':>11} "
        f"{'AvgInf(ms)':>11} {'Training Stats':>15}"
    )
    print_metrics_line(header, is_header=True)
    
    # Add reward components header
    reward_header = (
        f"{'':>11} {'Reward Components: Safety/Proximity/Shots/Threats/Pulsar':>65} "
    )
    print_metrics_line(reward_header)
    print()  # Print an empty line after headers

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
    latest_loss = metrics.losses[-1] if metrics.losses else 0.0
    
    # Calculate Average Inference Time (ms) and reset counters
    avg_inference_time_ms = 0.0
    with metrics.lock:
        if metrics.total_inference_requests > 0:
            avg_inference_time_ms = (metrics.total_inference_time / metrics.total_inference_requests) * 1000
        # Reset counters for the next interval
        metrics.total_inference_time = 0.0
        metrics.total_inference_requests = 0
    
    # Display average level as 1-based instead of 0-based
    display_level = metrics.average_level + 1.0

    # Calculate training rate (training steps per 1000 frames)
    train_rate = 0.0
    if metrics.frame_count > 0:
        train_rate = (metrics.total_training_steps * 1000.0) / metrics.frame_count
    
    # Calculate frames since last target update  
    frames_since_target_update = metrics.frame_count - metrics.last_target_update_frame
    
    # Format training stats: Memory/TrainSteps/Rate/TargetAge
    training_stats = f"{metrics.memory_buffer_size//1000}k/{metrics.total_training_steps}/{train_rate:.1f}/{frames_since_target_update//1000}k"

    # Format the row (Remove TrainQ)
    row = (
        f"{metrics.frame_count:>11,} {metrics.fps:>6.1f} {metrics.epsilon:>8.4f} " # Add comma for thousands
        f"{metrics.expert_ratio*100:>7.1f}% {mean_reward:>12.2f} {mean_dqn_reward:>12.2f} " # Show decimals to avoid rounding tiny rewards
        f"{latest_loss:>10.4f} {metrics.client_count:>8} "  
        f"{display_level:>10.1f} " # Added average level column, displayed as 1-based
        f"{'ON' if metrics.override_expert else 'OFF':>9} "
        f"{'ON' if metrics.expert_mode else 'OFF':>11} "
        f"{avg_inference_time_ms:>11.2f} "
        f"{training_stats:>15}"
    )
    
    print_metrics_line(row)
    
    # Display reward components on the next line
    reward_averages = metrics.get_reward_component_averages()
    reward_components_line = (
        f"{'':>11} "
        f"S:{reward_averages['safety']:>6.3f} "
        f"P:{reward_averages['proximity']:>6.3f} "
        f"Sh:{reward_averages['shots']:>6.3f} "
        f"T:{reward_averages['threats']:>6.3f} "
        f"Pu:{reward_averages['pulsar']:>6.3f} "
        f"Sc:{reward_averages['score']:>6.3f} "
        f"Tot:{reward_averages['total']:>6.3f}"
    )
    print_metrics_line(reward_components_line)

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