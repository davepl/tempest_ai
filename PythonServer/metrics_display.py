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
import multiprocessing as mp
import numpy as np
import math
from collections import deque
from typing import Optional, List, Dict, Any, Union, TextIO
import traceback
from pathlib import Path

# Import from config.py
from config import metrics, IS_INTERACTIVE, RL_CONFIG
# Import the helper function
from aimodel import print_with_terminal_restore, KeyboardHandler # Need KeyboardHandler for type hint

# Add a counter to track the number of rows printed
row_counter = 0
HEADER_REPRINT_INTERVAL = 25 # Define how often to reprint the header

# --- Log file setup ---
LOG_FILE_PATH = Path("teapot.log") # Log file in the same directory as the script

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

def get_metrics_header_string() -> str:
    """Returns the header string for metrics."""
    # Change header to Inf(ms) and reduce width (e.g., from 11 to 7)
    return (
        f"{'Frame':>8} | {'FPS':>5} | {'Clnts':>5} | {'Mean Reward':>11} | {'DQN Reward':>10} | {'Loss':>9} | "
        f"{'Epsilon':>7} | {'Inf(ms)':>7} | {'Guided %':>8} | {'Mem Size':>8} | {'Level Type':>10} | {'Override':>10}" # Changed header text and width
    )

def display_metrics_header(kb_handler=None, log_file_handle: Optional[TextIO] = None):
    """Display header for metrics table and optionally log it."""
    global row_counter
    header = get_metrics_header_string() # Use the updated helper
    if IS_INTERACTIVE:
        print_with_terminal_restore(kb_handler, f"\n{'-' * len(header)}")
        print_with_terminal_restore(kb_handler, header)
        print_with_terminal_restore(kb_handler, f"{'-' * len(header)}")
    row_counter = 0

    if log_file_handle:
        try:
            log_file_handle.write(header + "\n")
            log_file_handle.flush()
        except Exception as e:
            print(f"[StatsReporter Warning] Failed to write header to log file {LOG_FILE_PATH}: {e}")

def display_metrics_row(metrics_obj, agent_ref=None, kb_handler=None, log_file_handle: Optional[TextIO] = None):
    """Display current metrics, log the row, and increment row counter."""
    global row_counter
    if not IS_INTERACTIVE and not log_file_handle:
        return

    # --- Safely read metrics ---
    with metrics_obj.lock:
        # --- Calculate Metrics ---
        mean_reward_val = np.mean(list(metrics_obj.episode_rewards)) if metrics_obj.episode_rewards else float('nan')
        dqn_reward_val = np.mean(list(metrics_obj.dqn_rewards)) if metrics_obj.dqn_rewards else float('nan')
        mean_loss_val = np.mean(list(metrics_obj.losses)) if metrics_obj.losses else float('nan')
        # Read avg_dqn_inf_time (calculated in run_stats_reporter)
        avg_inf_time_sec = getattr(metrics_obj, 'avg_dqn_inf_time', float('nan')) # In seconds
        guided_ratio = metrics_obj.expert_ratio
        mem_size = len(agent_ref.memory) if agent_ref and hasattr(agent_ref, 'memory') else 0
        client_count = metrics_obj.client_count
        frame_count = metrics_obj.frame_count
        fps_float = metrics_obj.fps
        epsilon = metrics_obj.epsilon
        open_level = metrics_obj.open_level
        override_expert = metrics_obj.override_expert
        expert_mode = metrics_obj.expert_mode

        override_status = "OFF"
        if override_expert:
            override_status = "SELF"
        elif expert_mode:
            override_status = "BOT"

    # --- Conditional Formatting ---
    large_threshold = 1_000_000

    def format_large_number(value, width, precision=2, force_int=False): # Add force_int flag
        """Formats a number, using scientific notation if abs(value) >= threshold."""
        if math.isnan(value) or math.isinf(value):
            return f"{'NaN':>{width}}" if math.isnan(value) else f"{'Inf':>{width}}"

        if abs(value) >= large_threshold:
            # Scientific notation uses 'e' format
            return f"{value:{width}.{precision}e}"
        elif force_int:
             # --- Force Integer Output ---
             # Use floor and 'd' format for integer
             int_value = math.floor(value)
             return f"{int_value:{width}d}"
        else:
            # Default to fixed-point float
            return f"{value:{width}.{precision}f}"

    # Format the specific columns
    # --- Pass force_int=True for rewards ---
    mean_reward_str = format_large_number(mean_reward_val, width=11, force_int=True) # Adjusted width
    dqn_reward_str = format_large_number(dqn_reward_val, width=10, force_int=True) # Adjusted width
    mean_loss_str = format_large_number(mean_loss_val, width=9) # Keep loss potentially float

    # --- Format FPS as integer ---
    # Handle potential NaN before converting to int
    fps_int = int(fps_float) if not math.isnan(fps_float) else 0

    # --- Format InfTime ---
    inf_time_ms = avg_inf_time_sec * 1000 # Convert seconds to milliseconds
    # Adjust width in format specifier (e.g., from 11 to 7)
    inf_time_str = f"{inf_time_ms:7.2f}" if not math.isnan(inf_time_ms) else f"{'NaN':>7}"

    # --- Build Row String ---
    row = (
        f"{frame_count:8d} | {fps_int:5d} | {client_count:5d} | {mean_reward_str} | {dqn_reward_str} | " # Adjusted client width
        f"{mean_loss_str} | {epsilon:7.3f} | {inf_time_str} | {guided_ratio*100:7.2f}% | " # Updated width reflected here
        f"{mem_size:8d} | {'Open' if open_level else 'Closed':10} | {override_status:10}"
    )

    # --- Print to Console (if interactive) ---
    if IS_INTERACTIVE:
        print_with_terminal_restore(kb_handler, row)
        row_counter += 1 # Increment console row counter

    # --- Write to Log File (if handle provided) ---
    if log_file_handle:
        try:
            log_file_handle.write(row + "\n")
            log_file_handle.flush() # Optional: flush frequently for real-time view
        except Exception as e:
            # Avoid spamming logs with log write errors
            print(f"[StatsReporter Warning] Failed to write row to log file {LOG_FILE_PATH}: {e}", file=sys.stderr)

def run_stats_reporter(metrics_obj, shutdown_event: Union[threading.Event, mp.Event], agent_ref=None, kb_handler=None):
    """Thread function to report stats periodically and log to file."""
    global row_counter
    print(f"[StatsReporter] Starting stats reporter thread. Logging to: {LOG_FILE_PATH.resolve()}")
    last_report_time = time.time()
    report_interval = 10.0

    # --- Variables for FPS calculation (ideally should be in Metrics class) ---
    last_fps_update_time = time.time()
    last_frame_count = 0
    # Initialize metrics attributes if they don't exist (temporary fix)
    with metrics_obj.lock:
        if not hasattr(metrics_obj, 'fps'): metrics_obj.fps = 0.0
        # Read initial frame count safely
        last_frame_count = metrics_obj.frame_count

    log_file = None # Initialize file handle
    try:
        # --- Open Log File ---
        # Open in append mode ('a'). Creates the file if it doesn't exist.
        log_file = LOG_FILE_PATH.open('a', encoding='utf-8')
        write_header = log_file.tell() == 0 # Check if file is empty/new

        # --- Write Header (if needed) ---
        if write_header:
            print("[StatsReporter] Log file is new or empty. Writing header.")
            display_metrics_header(kb_handler=None, log_file_handle=log_file) # Write header to file only
        else:
            # Optionally add a separator on restart if file exists
            log_file.write(f"\n--- Reporter Restarted: {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            log_file.flush()

        # --- Display Initial Console Header ---
        if IS_INTERACTIVE:
            display_metrics_header(kb_handler=kb_handler) # Console only

        # --- Main Loop ---
        while not shutdown_event.is_set():
            try: # Inner try for loop operations
                current_time = time.time()

                # --- Calculate FPS periodically ---
                # (This block ideally belongs in the Metrics class update method)
                time_since_last_fps_update = current_time - last_fps_update_time
                if time_since_last_fps_update >= 1.0: # Update FPS roughly every second
                    with metrics_obj.lock:
                        current_frame_count = metrics_obj.frame_count
                        frames_processed = current_frame_count - last_frame_count
                        # Calculate FPS, handle division by zero
                        calculated_fps = frames_processed / time_since_last_fps_update if time_since_last_fps_update > 0 else 0.0
                        metrics_obj.fps = calculated_fps # Store calculated FPS
                        # Reset for next calculation interval
                        last_frame_count = current_frame_count
                    last_fps_update_time = current_time # Reset time *after* accessing metrics count

                # --- Check if it's time to report ---
                if current_time - last_report_time >= report_interval:

                    # --- Update Avg Inference Time ---
                    with metrics_obj.lock:
                        if hasattr(metrics_obj, 'dqn_inference_times') and metrics_obj.dqn_inference_times:
                            try:
                                 metrics_obj.avg_dqn_inf_time = np.mean(list(metrics_obj.dqn_inference_times))
                            except ValueError:
                                 metrics_obj.avg_dqn_inf_time = float('nan')
                        else:
                            metrics_obj.avg_dqn_inf_time = float('nan')

                    # --- Display Row (Console and Log) ---
                    display_metrics_row(metrics_obj, agent_ref, kb_handler, log_file_handle=log_file)
                    last_report_time = current_time

                    # --- Reprint Console Header periodically ---
                    if IS_INTERACTIVE and row_counter >= HEADER_REPRINT_INTERVAL:
                        display_metrics_header(kb_handler=kb_handler) # Console only

                time.sleep(0.1)

            except Exception as loop_err: # Catch errors within the loop
                print(f"[StatsReporter] Error in stats reporter inner loop: {loop_err}")
                traceback.print_exc()
                time.sleep(5) # Avoid busy-looping on error

    except Exception as setup_err: # Catch errors during setup (e.g., file open)
        print(f"[StatsReporter] Error setting up stats reporter: {setup_err}")
        traceback.print_exc()
    finally:
        # --- Ensure Log File is Closed ---
        if log_file:
            try:
                print("[StatsReporter] Closing log file.")
                log_file.close()
            except Exception as close_err:
                print(f"[StatsReporter Warning] Error closing log file {LOG_FILE_PATH}: {close_err}")

        print("[StatsReporter] Shutdown event detected, exiting stats reporter thread.")

# Example of how you might start it in main.py (ensure correct args are passed)
# stats_thread = threading.Thread(
#     target=run_stats_reporter,
#     args=(metrics, main_agent, kb_handler), # Pass metrics, agent ref, kb_handler
#     name="StatsReporterThread",
#     daemon=True
# ) 