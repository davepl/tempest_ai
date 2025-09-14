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

def _should_show_reward_columns() -> Dict[str, bool]:
    """Determine which reward component columns to show based on recent non-zero averages."""
    avgs = metrics.get_reward_component_averages()
    # Threshold to treat as effectively zero
    tol = 1e-6
    return {
        'safety': abs(avgs.get('safety', 0.0)) > tol,
        'proximity': abs(avgs.get('proximity', 0.0)) > tol,
        'shots': abs(avgs.get('shots', 0.0)) > tol,
        'threats': abs(avgs.get('threats', 0.0)) > tol,
        # Pulsar intentionally excluded from display
    }

def display_metrics_header():
    """Display the header for metrics output"""
    # Clear screen first
    # clear_screen()
    
    # Base header
    base = (
        f"{'Frame':>11} {'FPS':>6} {'Epsilon':>8} {'Expert%':>8} "
        f"{'Mean Reward':>12} {'DQN Reward':>12} {'Loss':>10} "
        f"{'Clients':>8} {'Avg Level':>10} {'Override':>9} {'Expert Mode':>11} "
        f"{'AvgInf(ms)':>11} {'Training Stats':>15}"
    )

    # Conditionally append reward component headers (without Pulsar)
    show = _should_show_reward_columns()
    reward_parts = []
    if show['safety']:
        reward_parts.append(f"{'Safety':>7}")
    if show['proximity']:
        reward_parts.append(f"{'Prox':>6}")
    if show['shots']:
        reward_parts.append(f"{'Shots':>6}")
    if show['threats']:
        reward_parts.append(f"{'Threats':>7}")

    header = base if not reward_parts else base + ' ' + ' '.join(reward_parts)
    print_metrics_line(header, is_header=True)

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

    # Base row text
    row = (
        f"{metrics.frame_count:>11,} {metrics.fps:>6.1f} {metrics.epsilon:>8.4f} "
        f"{metrics.expert_ratio*100:>7.1f}% {mean_reward:>12.2f} {mean_dqn_reward:>12.2f} "
        f"{latest_loss:>10.4f} {metrics.client_count:>8} "
        f"{display_level:>10.1f} "
        f"{'ON' if metrics.override_expert else 'OFF':>9} "
        f"{'ON' if metrics.expert_mode else 'OFF':>11} "
        f"{avg_inference_time_ms:>11.2f} "
        f"{training_stats:>15}"
    )

    # Conditionally append reward component values (without Pulsar)
    show = _should_show_reward_columns()
    if any(show.values()):
        avgs = metrics.get_reward_component_averages()
        parts = []
        if show['safety']:
            parts.append(f"{avgs['safety']:>7.3f}")
        if show['proximity']:
            parts.append(f"{avgs['proximity']:>6.3f}")
        if show['shots']:
            parts.append(f"{avgs['shots']:>6.3f}")
        if show['threats']:
            parts.append(f"{avgs['threats']:>7.3f}")
        row = row + ' ' + ' '.join(parts)
    
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