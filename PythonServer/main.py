#!/usr/bin/env python3
"""
Tempest AI Main Entry Point (Single Process Architecture)
Coordinates the socket server, metrics display, keyboard handling, and integrated training/inference.
"""

import os
import sys
import time
import signal
import traceback
import threading
from threading import Thread, Event as ThreadEvent, Lock as ThreadLock
from queue import Queue, Empty, Full
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import setproctitle
from pathlib import Path

from aimodel import (
    DQNAgent, KeyboardHandler,
    setup_environment,
    print_with_terminal_restore
)
from config import (
    RL_CONFIG, MODEL_DIR, LATEST_MODEL_PATH, IS_INTERACTIVE, metrics, SERVER_CONFIG,
    ACTION_MAPPING, DEVICE
)
from socket_server import SocketServer
from metrics_display import display_metrics_header, display_metrics_row, run_stats_reporter

# Setup path if necessary (assuming script is run from project root)
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def training_loop(training_job_queue: Queue, agent: DQNAgent, metrics_obj, shutdown_event: ThreadEvent):
    """
    Thread function dedicated to performing training steps.
    Triggered by messages placed in the training_job_queue.
    """
    print("[TrainingThread] Starting...")
    try:
        # Limit PyTorch CPU threads for *this specific thread*
        # torch.set_num_threads(8)
        print("[TrainingThread] Set torch num_threads=any")
    except Exception as thread_err:
        print(f"[TrainingThread Warning] Failed to set torch threads: {thread_err}")
        
    batches_processed = 0
    target_update_frequency = RL_CONFIG.target_update // RL_CONFIG.train_freq # Calculate frequency once

    while not shutdown_event.is_set():
        try:
            # ---> Check shutdown event AGAIN before blocking <--- 
            if shutdown_event.is_set():
                break
                
            # Wait for a signal to train (blocking with timeout)
            train_signal = training_job_queue.get(timeout=1.0)

            # ---> Check for shutdown sentinel <--- 
            if train_signal is None:
                print("[TrainingThread] Received None sentinel, shutting down.")
                break # Exit loop immediately

            if train_signal: # Received a real signal (should always be True now)
                # Check buffer size directly on the shared agent's memory
                if len(agent.memory) >= RL_CONFIG.min_buffer_size:
                    # Sample directly from the shared agent's memory
                    batch = agent.memory.sample(RL_CONFIG.batch_size)

                    if batch:
                        # Perform training step on the shared agent
                        loss = agent.train_step(batch)
                        batches_processed += 1

                        if loss is not None:
                            metrics_obj.add_loss(loss.item()) # Update shared metrics

                        # --- Update Target Network ---
                        # Check should be based on total frames processed, not just batches here
                        # Let the main loop handle target updates based on frame count
                        # if batches_processed > 0 and batches_processed % target_update_frequency == 0:
                        #     agent.update_target_network()
                        #     # print(f"[TrainingThread] Updated target network.") # Optional log

                else:
                     # Not enough memory, wait a bit
                     time.sleep(0.2)

        except Empty:
            # Queue was empty, loop and check shutdown event
            continue
        except Exception as e:
            print(f"[TrainingThread] Error: {e}")
            traceback.print_exc()
            time.sleep(1) # Wait after error

    print("[TrainingThread] Shutdown signal received, exiting.")

def decay_epsilon(metrics_obj, frame_count):
    """Calculate and update epsilon based on a multiplicative decay schedule."""
    # Calculate the number of decay intervals that have passed
    # Use RL_CONFIG.decay_epsilon_frames for the interval length
    if RL_CONFIG.decay_epsilon_frames <= 0: # Avoid division by zero or infinite loop
        # Check if metrics_obj has an epsilon attribute first for robustness
        return getattr(metrics_obj, 'epsilon', RL_CONFIG.epsilon_start) # Return current or start epsilon
        
    decay_intervals = frame_count // RL_CONFIG.decay_epsilon_frames 

    # Calculate the new epsilon using multiplicative decay
    # Use the correct attribute 'epsilon_decay_rate' from RL_CONFIG
    decay_factor = RL_CONFIG.epsilon_decay_rate 
    # Apply the decay factor exponentially based on the number of intervals
    new_epsilon = RL_CONFIG.epsilon_start * (decay_factor ** decay_intervals)
    
    # Clamp the epsilon value to the minimum defined in the config
    new_epsilon = max(RL_CONFIG.epsilon_end, new_epsilon)
    
    # Update the epsilon value in the shared metrics object only if it changed
    # Use a small tolerance for floating point comparison to avoid unnecessary updates
    current_epsilon = getattr(metrics_obj, 'epsilon', RL_CONFIG.epsilon_start) 
    if not np.isclose(current_epsilon, new_epsilon):
         metrics_obj.update_epsilon(new_epsilon)
    
    return new_epsilon

def decay_expert_ratio(metrics_obj, frame_count):
    """Update expert ratio based on configured intervals."""
    # This function assumes it's called periodically
    with metrics_obj.lock:
        # Skip decay if expert mode or override is active
        if metrics_obj.expert_mode or metrics_obj.override_expert:
            # Keep base ratio unchanged, effective ratio is handled by get_expert_ratio
            return metrics_obj.expert_ratio 

        step_interval = frame_count // SERVER_CONFIG.expert_ratio_decay_steps
        
        # Only update if we've moved to a new interval
        if step_interval > metrics_obj.last_decay_step:
            metrics_obj.last_decay_step = step_interval
            if step_interval == 0:
                # First interval - use starting value
                metrics_obj.expert_ratio = SERVER_CONFIG.expert_ratio_start
            else:
                # Apply decay
                metrics_obj.expert_ratio *= SERVER_CONFIG.expert_ratio_decay
            
            # Ensure we don't go below the minimum
            metrics_obj.expert_ratio = max(SERVER_CONFIG.expert_ratio_min, metrics_obj.expert_ratio)
            # print(f"[Decay] Frame {frame_count}: Decayed expert_ratio to {metrics_obj.expert_ratio:.4f}") # Debug
            
        return metrics_obj.expert_ratio

def main():
    setup_environment()
    setproctitle.setproctitle("python_main_single") # Set title for main process

    # --- Use standard threading primitives ---
    # Queue for signaling training thread
    training_job_queue = Queue(maxsize=RL_CONFIG.train_queue_size * 2) # Signal queue can be larger
    # Shutdown Signal
    shutdown_event = ThreadEvent()
    # Lock for saving model (still useful if multiple threads might save)
    save_lock = ThreadLock() # Use threading lock

    # --- Initialize Main Agent (Single Instance) ---
    num_flattened_values = SERVER_CONFIG.params_count # Use config value
    state_size = num_flattened_values
    action_size = len(ACTION_MAPPING)
    print(f"[Main] State Size: {state_size}, Action Size: {action_size}")
    print(f"[Main] Using PyTorch device: '{DEVICE}'")

    # Initialize the single DQNAgent instance
    print("[Main] Initializing DQNAgent...")
    main_agent = DQNAgent(state_size, action_size, device=DEVICE)
    # DQNAgent init prints its device

    # Load weights/metrics into the agent
    if LATEST_MODEL_PATH.exists():
        try:
            success, loaded_metrics_state = main_agent.load(LATEST_MODEL_PATH)
            # Print status AFTER attempting load
            if success:
                print(f"[Main] Initial agent state loaded successfully from {LATEST_MODEL_PATH}")
                if loaded_metrics_state:
                    print("[Main] Restoring metrics from checkpoint...")
                    with metrics.lock:
                         metrics.frame_count = loaded_metrics_state.get('frame_count', metrics.frame_count)
                         metrics.epsilon = loaded_metrics_state.get('epsilon', metrics.epsilon)
                         metrics.expert_ratio = loaded_metrics_state.get('expert_ratio', metrics.expert_ratio)
                         metrics.last_decay_step = loaded_metrics_state.get('last_decay_step', metrics.last_decay_step)
                    print(f"[Main] Metrics restored: Frame={metrics.frame_count}")
            else:
                 print(f"[Main Warning] Loading agent model from {LATEST_MODEL_PATH} failed (see agent log). Starting fresh.")
        except Exception as e:
             print(f"[Main Error] Error processing model load: {e}. Starting fresh.")
             traceback.print_exc()
    else:
        print(f"[Main] Agent model file not found ({LATEST_MODEL_PATH}). Starting agent with fresh weights.")

    # --- Start Training Thread ---
    print("[Main] Starting training thread...")
    training_thread = Thread(
        target=training_loop,
        args=(training_job_queue, main_agent, metrics, shutdown_event),
        name="TrainingThread",
        daemon=True
    )
    training_thread.start()
    print("[Main] Training thread started.")

    # --- Initialize Keyboard Handler (if interactive) ---
    kb_handler = None
    if IS_INTERACTIVE:
         kb_handler = KeyboardHandler()
         print("[Main] Keyboard handler active. Keys: (o)verride, (e)xpert mode, (q)uit, (s)ave") # Added save key info

    # --- Start Socket Server ---
    print("[Main] Starting socket server...")
    # Pass training_job_queue and the single main_agent reference
    socket_server = SocketServer(
        metrics=metrics,
        main_agent_ref=main_agent,
        shutdown_event=shutdown_event,
        training_job_queue=training_job_queue # Pass the queue
    )
    server_thread = threading.Thread(target=socket_server.start, name="SocketServerThread", daemon=True)
    server_thread.start()
    print("[Main] Socket server thread started.")

    # --- Start Stats Reporter Thread ---
    print("[Main] Starting stats reporter thread...")
    stats_thread = threading.Thread(
        target=run_stats_reporter,
        args=(metrics, shutdown_event, main_agent, kb_handler), # Pass threading event
        name="StatsReporterThread",
        daemon=True
    )
    stats_thread.start()
    # print("[Main] Stats reporter thread started.") # Redundant

    # --- Graceful Shutdown Handling ---
    def signal_handler(sig, frame):
        print(f"\n[Main] Signal {sig} received. Initiating graceful shutdown...")
        shutdown_event.set() # Signal all threads
        try:
            training_job_queue.put(None, block=False) # Signal training thread to exit
        except Full:
            pass # Ignore if queue is full
        socket_server.shutdown() # Tell server to stop accepting

    signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler) # Handle termination signals

    # --- Main Loop (Keep alive, handle keyboard, target updates) ---
    try:
        frame_counter_cache = 0
        last_save_frame = -1 # Track frame for periodic save (optional now)
        last_target_update_frame = -1 # Track frame for target update

        while not shutdown_event.is_set():
            current_frame_count = metrics.frame_count # Read thread-safe value

            # --- Periodic Updates (Epsilon/Expert Decay) ---
            if current_frame_count > frame_counter_cache:
                 decay_epsilon(metrics, current_frame_count)
                 decay_expert_ratio(metrics, current_frame_count)
                 frame_counter_cache = current_frame_count

            # --- Periodic Target Network Update ---
            # Check if we crossed a target update interval boundary
            if RL_CONFIG.target_update > 0 and current_frame_count // RL_CONFIG.target_update > last_target_update_frame // RL_CONFIG.target_update:
                if main_agent.is_ready:
                     # print(f"[Main] Frame {current_frame_count}: Triggering target network update.")
                     main_agent.update_target_network()
                     last_target_update_frame = current_frame_count
                else:
                    print(f"[Main] Frame {current_frame_count}: Skipped target update (Agent not ready).")


            # --- Optional Periodic Save (less critical now trainer saves) --- REMOVING BLOCK
            # Keep manual save via 's' key
            # if RL_CONFIG.save_interval > 0 and current_frame_count // RL_CONFIG.save_interval > last_save_frame // RL_CONFIG.save_interval:
            #      if main_agent.is_ready:
            #          print(f"[Main] Frame {current_frame_count}: Triggering periodic save...")
            #          # ... save logic using save_lock ...
            #          last_save_frame = current_frame_count

            # --- Keyboard Input ---
            if kb_handler:
                key = kb_handler.check_key()
                if key == 'q':
                    print("[Main] 'q' pressed. Quitting...")
                    shutdown_event.set()
                    socket_server.shutdown()
                    break
                elif key == 's':
                    print("[Main] 's' pressed. Saving model...")
                    try:
                        # Acquire lock for saving
                        with save_lock:
                            # Get metrics state *under the metrics lock*
                            with metrics.lock:
                                metrics_state_to_save = metrics.get_state_for_save()
                            print("[Main DEBUG] Calling main_agent.save() with metrics state...")
                            main_agent.save(LATEST_MODEL_PATH, metrics_state=metrics_state_to_save)
                            print("[Main DEBUG] main_agent.save() completed.")
                    except Exception as save_err:
                        print(f"[Main Error] Failed manual save: {save_err}")
                        traceback.print_exc()

                elif key == 'o':
                    with metrics.lock:
                        metrics.override_expert = not metrics.override_expert
                    print_with_terminal_restore(kb_handler, f"Inference Override Toggled: {'ON' if metrics.override_expert else 'OFF'}")
                elif key == 'e':
                    with metrics.lock:
                         metrics.expert_mode = not metrics.expert_mode
                    print_with_terminal_restore(kb_handler, f"Expert Mode Toggled: {'ON' if metrics.expert_mode else 'OFF'}")
                elif key == ' ': # Spacebar for stats display
                    display_metrics_row(metrics, main_agent, kb_handler)

            # --- Health Checks (Simplified) ---
            if not server_thread.is_alive():
                 print("[Main Error] Socket server thread died unexpectedly. Shutting down.")
                 shutdown_event.set()
                 break
            if not training_thread.is_alive():
                 print("[Main Error] Training thread died unexpectedly. Shutting down.")
                 shutdown_event.set()
                 break
            if not stats_thread.is_alive():
                 print("[Main Error] Stats reporter thread died unexpectedly. Shutting down.")
                 shutdown_event.set()
                 break

            time.sleep(0.1) # Main loop sleep

    except Exception as main_err:
         print(f"[Main] Unexpected error in main loop: {main_err}")
         traceback.print_exc()
         shutdown_event.set()
         try:
             training_job_queue.put(None, block=False) # Signal training thread to exit
         except Full:
             pass # Ignore if queue is full
         socket_server.shutdown()

    finally:
        print("[Main] Entering shutdown sequence...")
        if kb_handler:
            kb_handler.restore_terminal()
            # print("[Main] Terminal restored.") # Less critical message

        # Ensure shutdown event is set
        shutdown_event.set()
        # print("[Main] Shutdown event confirmed set.") # Less critical message

        # ---> Perform Final Save BEFORE joining threads <--- 
        print("[Main] Performing final save...") # Shortened message
        if 'main_agent' in locals() and main_agent.is_ready:
            try:
                 with save_lock:
                     # Get metrics state *under the metrics lock*
                     with metrics.lock:
                         metrics_state_to_save = metrics.get_state_for_save()
                     print("[Main DEBUG] Calling main_agent.save() with metrics state...")
                     main_agent.save(LATEST_MODEL_PATH, metrics_state=metrics_state_to_save)
                     print("[Main DEBUG] main_agent.save() completed.")
            except Exception as final_save_err:
                 print(f"[Main Error] Failed final save: {final_save_err}")
                 traceback.print_exc()
        else:
            print("[Main] Final save skipped (Agent not found or not ready).")
        # ---> End Final Save <--- 

        # Wait for essential threads (longer timeout)...
        print("[Main] Signaling essential threads to exit (if not already). Relying on daemon status for cleanup.")

        # Server socket closure is handled by the SocketServer thread's finally block.

        print("[Main] Shutdown complete.")
        sys.stdout.flush() # Explicitly flush output
        sys.stderr.flush()
        sys.exit(0) # Explicit exit

if __name__ == "__main__":
    main() 