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
import traceback

# Import from config.py
from config import metrics, IS_INTERACTIVE
# Import the helper function
from aimodel import print_with_terminal_restore, KeyboardHandler # Need KeyboardHandler for type hint

# Add a counter to track the number of rows printed
row_counter = 0

def clear_screen():
    """Clear the screen and move cursor to home position"""
    if IS_INTERACTIVE:
        # Clear screen and move to home
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()

def print_metrics_line(message, is_header=False, kb_handler: Optional[KeyboardHandler] = None):
    """Print a metrics line using terminal restore if interactive."""
    global row_counter
    
    # Use helper function for printing to handle terminal state
    print_func = print if not IS_INTERACTIVE else lambda msg: print_with_terminal_restore(kb_handler, msg)

    if IS_INTERACTIVE:
        if is_header:
            print_func(message)
            print_func("-" * len(message)) # Add separator line
            row_counter = 0
        else:
            print_func(message)
            row_counter += 1
        # Flush is handled by print_with_terminal_restore or print
    else:
        # Non-interactive just prints
        print(message)

def display_metrics_header(kb_handler: Optional[KeyboardHandler] = None):
    """Display the header for metrics output"""
    # Print header (Remove TrainQ, Adjust Time column for Xd HH:MM)
    header = (
        f"{'Frame':>11} {'Time':>9} {'FPS':>6} {'Epsilon':>8} {'Expert%':>8} " # Adjusted Time width/label
        f"{'Mean Reward':>12} {'DQN Reward':>12} {'Loss':>11} "
        f"{'Clients':>8} {'Override':>9} {'Expert Mode':>11} "
        f"{'InfTime(ms)':>10}"
    )
    print_metrics_line(header, is_header=True, kb_handler=kb_handler)
    # Print an empty line after header
    # print_metrics_line("", kb_handler=kb_handler)

def display_metrics_row(metrics_obj, agent, kb_handler: Optional[KeyboardHandler] = None):
    """Display a row of metrics data using the provided metrics object."""
    global row_counter
    
    # Check if we need to print the header (every 30th row)
    if IS_INTERACTIVE and row_counter > 0 and row_counter % 30 == 0:
        # Header requires kb_handler if interactive
        display_metrics_header(kb_handler=kb_handler)
    
    # Calculate mean reward using passed metrics_obj
    mean_reward = 0
    if metrics_obj.episode_rewards:
        rewards_list = list(metrics_obj.episode_rewards)
        mean_reward = sum(rewards_list[-100:]) / min(len(rewards_list), 100)
    
    # Calculate mean DQN reward using passed metrics_obj
    mean_dqn_reward = 0
    if metrics_obj.dqn_rewards:
        dqn_rewards_list = list(metrics_obj.dqn_rewards)
        mean_dqn_reward = sum(dqn_rewards_list[-100:]) / min(len(dqn_rewards_list), 100)
    
    # Get the latest loss value using passed metrics_obj
    latest_loss = metrics_obj.losses[-1] if metrics_obj.losses else 0
    
    # Get Average DQN Inference Time using passed metrics_obj
    avg_inf_time_ms = metrics_obj.avg_dqn_inf_time

    # Calculate Time (Xd HH:MM @ 30 FPS)
    total_seconds = metrics_obj.frame_count / 30
    days = int(total_seconds // 86400)
    hours = int((total_seconds % 86400) // 3600)
    minutes = int((total_seconds % 3600) // 60)
    # seconds = int(total_seconds % 60) # Ignore seconds
    time_str = f"{days}d {hours:02d}:{minutes:02d}"

    # Format the row using passed metrics_obj
    # Use get_expert_ratio() for display
    effective_expert_ratio = metrics_obj.get_expert_ratio()
    row = (
        f"{metrics_obj.frame_count:>11,} {time_str:>9} {metrics_obj.fps:>6.1f} {metrics_obj.epsilon:>8.4f} " # Adjusted time width
        f"{effective_expert_ratio*100:>7.1f}% {mean_reward:>12.2e} {mean_dqn_reward:>12.2e} " # Display effective ratio
        f"{latest_loss:>11.2e} {metrics_obj.client_count:>8} "
        f"{'ON' if metrics_obj.override_expert else 'OFF':>9} "
        f"{'ON' if metrics_obj.expert_mode else 'OFF':>11} "
        f"{avg_inf_time_ms:>10.2f}"
    )
    
    print_metrics_line(row, kb_handler=kb_handler)

def run_stats_reporter(metrics_obj, kb_handler: Optional[KeyboardHandler] = None):
    """Run the stats reporter in a loop, using kb_handler for safe printing."""
    # Pass kb_handler to header display
    if IS_INTERACTIVE:
         display_metrics_header(kb_handler=kb_handler)
    else:
         display_metrics_header() # Non-interactive doesn't need it
    
    # Use a shutdown flag checked more often than sleep
    while True:
        try:
            # Check for shutdown more frequently
            for _ in range(100): # Check every 0.1s for 10 seconds total
                 if threading.main_thread().is_alive() == False: # Exit if main thread dies
                      print("[StatsReporter] Main thread seems to have exited. Stopping.")
                      return
                 time.sleep(0.1)

            # Calculate FPS for the completed interval
            metrics_obj.calculate_interval_fps()
            # Calculate average inference time for the completed interval
            metrics_obj.calculate_interval_avg_inf_time()
            
            # Display row, passing metrics_obj and kb_handler
            display_metrics_row(metrics_obj, None, kb_handler=kb_handler)
            
        except Exception as e:
            err_msg = f"[StatsReporter] Error: {e}"
            if IS_INTERACTIVE:
                print_with_terminal_restore(kb_handler, err_msg)
            else:
                print(err_msg)
            traceback.print_exc() # Print full trace too
            time.sleep(1) # Wait a bit after error 