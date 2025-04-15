#!/usr/bin/env python3
"""
Tempest AI Main Entry Point
Coordinates the socket server, metrics display, and keyboard handling.
"""

import os
import sys
import time
import signal
import traceback
import threading
import multiprocessing as mp
from multiprocessing import Process, Queue
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import setproctitle
from queue import Empty, Full
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

# Import necessary modules from the project
from training import training_worker

# +++ Define the DQN model structure (must match the one in aimodel.py) +++
class DQN(nn.Module):
    """Deep Q-Network model (copied for inference worker)."""
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 256) # Input -> Hidden 1 (256)
        self.fc2 = nn.Linear(256, 128)        # Hidden 1 -> Hidden 2 (128)
        self.fc3 = nn.Linear(128, 64)         # Hidden 2 -> Hidden 3 (64)
        self.out = nn.Linear(64, action_size) # Hidden 3 -> Output

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)
# +++ End DQN model definition +++

def inference_worker(state_queue: mp.Queue, action_queue: mp.Queue, 
                     model_path: Path, shutdown_event: mp.Event, 
                     inference_ready_event: mp.Event,
                     state_size: int, action_size: int):
    """Worker process dedicated to DQN inference."""
    setproctitle.setproctitle("python_inference")
    print("[InferWorker] Starting...")
    
    # Limit threads for inference efficiency
    try:
        torch.set_num_threads(1)
        print("[InferWorker] Set torch num_threads=1")
    except Exception as thread_err:
        print(f"[InferWorker Warning] Failed to set torch threads: {thread_err}")

    agent = None
    last_model_check_time = 0
    model_check_interval = 300 # seconds

    try:
        # --- Initialize Agent --- 
        agent = DQNAgent(state_size, action_size)
        print(f"[InferWorker] DQNAgent initialized on device: {agent.device}")

        # --- Load Initial Model --- 
        if model_path.exists():
            success, _ = agent.load(model_path)
            if success:
                print(f"[InferWorker] Initial model loaded from {model_path}")
            else:
                print(f"[InferWorker Warning] Failed to load initial model from {model_path}. Using fresh weights.")
        else:
             print(f"[InferWorker Warning] Model file {model_path} not found. Using fresh weights.")
             
        # --- Signal Main Process that Worker is Ready --- 
        inference_ready_event.set()
        print("[InferWorker] Signaled ready to main process.")

        # --- Inference Loop --- 
        while not shutdown_event.is_set():
            try:
                # Periodically check for updated model weights
                current_time = time.time()
                if current_time - last_model_check_time > model_check_interval:
                     if model_path.exists():
                          # print("[InferWorker] Checking for model update...") # Debug
                          # TODO: Implement more robust check? Stat file mtime?
                          mtime = model_path.stat().st_mtime
                          if mtime > last_model_check_time: # Check if file is newer
                               print(f"[InferWorker] Model file updated (mtime: {mtime:.0f} > last load: {last_model_check_time:.0f}). Reloading...")
                               success, _ = agent.load(model_path) # Reload weights
                               if success:
                                    agent.policy_net.eval()
                                    last_model_check_time = mtime # Use mtime as new load time
                               else:
                                    print("[InferWorker] Model reload FAILED.")
                                    last_model_check_time = current_time # Update time anyway to avoid rapid checks
                          # else: # Optional: print("[InferWorker] Model file unchanged, skipping reload.")
                     # else: Keep using existing weights if file disappeared
                     last_model_check_time = current_time # Update check time even if file missing

                # Get state from the queue (blocking with timeout)
                state = state_queue.get(timeout=0.5) # Timeout to allow checking shutdown/reload
                
                if state is None: # Check for shutdown sentinel
                    print("[InferWorker] Received None state, shutting down.")
                    break

                # Perform inference
                inf_start_time = time.time()
                action_idx = agent.act(state, epsilon=0.0) # Use epsilon=0 for deterministic exploitation
                inf_end_time = time.time()
                inference_time_sec = inf_end_time - inf_start_time
                
                # Put action and inference time back
                try:
                    action_queue.put((action_idx, inference_time_sec), block=False)
                except Full:
                    # This shouldn't happen often if socket server consumes actions
                    print("[InferWorker Warning] Action queue full. Discarding action.")

            except Empty:
                # state_queue was empty, just loop again to check shutdown/reload
                continue
            except (KeyboardInterrupt, SystemExit):
                print("[InferWorker] Interrupted, shutting down...")
                break
            except Exception as e:
                print(f"[InferWorker] Error in inference loop: {e}")
                traceback.print_exc()
                time.sleep(0.5) # Avoid tight loop on error

        print("[InferWorker] Exiting.")

    except Exception as init_err:
         print(f"[InferWorker] Fatal error during initialization: {init_err}")
         traceback.print_exc()

def stats_reporter(agent, kb_handler):
    """Thread function to report stats periodically"""
    print("Starting stats reporter thread...")
    last_report = time.time()
    report_interval = 10.0  # Report every 10 seconds
    
    # Display the header once at the beginning
    display_metrics_header()
    
    while True:
        try:
            current_time = time.time()
            if current_time - last_report >= report_interval:
                # +++ Calculate average DQN inference time +++
                with metrics.lock: # Access deque under lock
                    if metrics.dqn_inference_times: # Avoid division by zero
                        metrics.avg_dqn_inf_time = np.mean(list(metrics.dqn_inference_times))
                    else:
                        metrics.avg_dqn_inf_time = 0.0
                # +++ End calculation +++
                display_metrics_row(agent, kb_handler)
                last_report = current_time
            
            # Check if server is still running
            if metrics.global_server is None or not metrics.global_server.running:
                print("Server stopped running, exiting stats reporter")
                break
                
            time.sleep(0.1)
        except Exception as e:
            print(f"Error in stats reporter: {e}")
            traceback.print_exc()
            break

def keyboard_input_handler(agent, keyboard_handler):
    """Thread function to handle keyboard input"""
    print("Starting keyboard input handler thread...")
    
    while True:
        try:
            # Check for keyboard input
            key = keyboard_handler.check_key()
            
            if key:
                # Handle different keys
                if key == 'q':
                    print("Quit command received, shutting down...")
                    metrics.global_server.running = False
                    break
                elif key == 's':
                    print("[Main] 's' pressed. Saving model...")
                    # Fetch current metrics state under lock
                    try:
                        with metrics.lock:
                            metrics_state_to_save = {
                                'frame_count': metrics.frame_count,
                                'epsilon': metrics.epsilon,
                                'expert_ratio': metrics.expert_ratio,
                                'last_decay_step': metrics.last_decay_step
                            }
                        # Save with the fetched metrics state
                        agent.save(LATEST_MODEL_PATH, metrics_state=metrics_state_to_save)
                        print(f"[Main] Model saved to {LATEST_MODEL_PATH} (Frame: {metrics_state_to_save['frame_count']})")
                    except Exception as save_err:
                        print(f"[Main Error] Failed manual save: {save_err}")
                elif key == 'o':
                    # Toggle override flag directly
                    with metrics.lock:
                        metrics.override_expert = not metrics.override_expert
                    # Use clearer terminology
                    print_with_terminal_restore(kb_handler, f"Inference Override Toggled: {'ON' if metrics.override_expert else 'OFF'}")
                elif key == 'e':
                    # Toggle expert mode flag directly
                    with metrics.lock:
                         metrics.expert_mode = not metrics.expert_mode
                    print_with_terminal_restore(kb_handler, f"Expert Mode Toggled: {'ON' if metrics.expert_mode else 'OFF'}")
            
            time.sleep(0.1)
        except Exception as e:
            print(f"Error in keyboard input handler: {e}")
            break

def batch_sampler_thread(replay_memory, batch_queue, shutdown_event, batch_size, min_buffer_size):
    """Samples batches from memory and puts them in the queue for the trainer."""
    print("[BatchSampler] Starting...")
    while not shutdown_event.is_set():
        try:
            if len(replay_memory) >= min_buffer_size:
                # ReplayMemory.sample now returns NumPy arrays directly
                batch_numpy = replay_memory.sample(batch_size)
                
                if batch_numpy:
                    # No conversion needed, data is already numpy
                    try:
                        batch_queue.put(batch_numpy, timeout=0.1) # Put numpy batch in queue
                    except Full:
                        time.sleep(0.05) # Give trainer time to catch up
                        continue

                else:
                     # Sample returned None (e.g., memory empty)
                     time.sleep(0.1)
            else:
                # Not enough memory yet, wait longer
                time.sleep(0.5)
        except Exception as e:
            print(f"[BatchSampler] Error: {e}")
            traceback.print_exc()
            time.sleep(1) # Wait after error
    print("[BatchSampler] Shutdown signal received, exiting.")

def loss_reporter_thread(loss_queue, metrics_obj, shutdown_event):
    """Receives loss values from the trainer process and updates metrics."""
    print("[LossReporter] Starting...")
    while not shutdown_event.is_set():
        try:
            loss = loss_queue.get(timeout=1.0) # Wait for loss
            if loss is not None:
                metrics_obj.add_loss(loss) # Add loss using the dedicated method
        except Empty:
            continue # No loss received, check shutdown event
        except Exception as e:
            print(f"[LossReporter] Error: {e}")
            traceback.print_exc()
            time.sleep(1) # Wait after error
    print("[LossReporter] Shutdown signal received, exiting.")

# --- Decay Functions ---
def decay_epsilon(metrics_obj, frame_count):
    """Calculate and update decayed exploration rate."""
    # This function assumes it's called periodically
    new_epsilon = max(RL_CONFIG.epsilon_end, 
                   RL_CONFIG.epsilon_start * 
                   np.exp(-frame_count / RL_CONFIG.epsilon_decay))
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
    setproctitle.setproctitle("python_main") # Set title for main process

    # Use multiprocessing context for better compatibility
    ctx = mp.get_context('spawn') # 'spawn' is often safer than 'fork'

    # --- Queues and Events ---
    # For Inference Process
    state_queue = ctx.Queue(maxsize=100) # States TO inference worker
    action_queue = ctx.Queue(maxsize=100) # Actions FROM inference worker
    # For Training Process
    train_batch_queue = ctx.Queue(maxsize=RL_CONFIG.train_queue_size) 
    loss_queue = ctx.Queue(maxsize=RL_CONFIG.loss_queue_size)
    # Shutdown Signal
    shutdown_event = ctx.Event()
    # Inference Ready Signal
    inference_ready_event = ctx.Event()

    # --- Initialize Main Agent (for Memory, Step, Target Updates) ---
    num_flattened_values = 315 # UPDATE THIS IF LUA CHANGES
    state_size = num_flattened_values
    action_size = len(ACTION_MAPPING)
    print(f"[Main] State Size: {state_size}, Action Size: {action_size}")

    main_agent = DQNAgent(state_size, action_size)
    # Load weights/metrics into main agent
    if LATEST_MODEL_PATH.exists():
        try:
            success, loaded_metrics_state = main_agent.load(LATEST_MODEL_PATH)
            if success and loaded_metrics_state:
                print("[Main] Restoring metrics from checkpoint...")
                with metrics.lock:
                     metrics.frame_count = loaded_metrics_state.get('frame_count', metrics.frame_count)
                     metrics.epsilon = loaded_metrics_state.get('epsilon', metrics.epsilon)
                     metrics.expert_ratio = loaded_metrics_state.get('expert_ratio', metrics.expert_ratio)
                     metrics.last_decay_step = loaded_metrics_state.get('last_decay_step', metrics.last_decay_step)
                print(f"[Main] Metrics restored: Frame={metrics.frame_count}")
        except Exception as e:
             print(f"[Main] Error processing model load for main agent: {e}. Starting fresh.")
             traceback.print_exc()
    else:
        print(f"[Main] Agent model file not found at {LATEST_MODEL_PATH}. Main agent will start fresh.")

    # --- Start Inference Process --- 
    inference_process = ctx.Process(
        target=inference_worker,
        args=(state_queue, action_queue, LATEST_MODEL_PATH, 
              shutdown_event, inference_ready_event, # Pass ready event
              state_size, action_size),
        name="InferenceProcess"
    )
    inference_process.daemon = True
    inference_process.start()
    print("[Main] Inference process started.")

    # --- Start Dedicated Training Process --- 
    training_process = ctx.Process(
        target=training_worker,
        args=(train_batch_queue, loss_queue, shutdown_event, state_size, action_size),
        name="TrainingProcess"
    )
    training_process.daemon = True
    training_process.start()
    print("[Main] Training process started.")

    # --- Start Helper Threads in Main Process ---
    # Batch sampler uses main_agent.memory
    batch_sampler = threading.Thread(
        target=batch_sampler_thread,
        args=(main_agent.memory, train_batch_queue, shutdown_event, RL_CONFIG.batch_size, RL_CONFIG.min_buffer_size),
        name="BatchSamplerThread",
        daemon=True
    )
    batch_sampler.start()
    print("[Main] Batch sampler thread started.")
    # Thread to get loss values from loss_queue and update metrics
    loss_reporter = threading.Thread(
        target=loss_reporter_thread,
        args=(loss_queue, metrics, shutdown_event),
        name="LossReporterThread",
        daemon=True
    )
    loss_reporter.start()
    print("[Main] Loss reporter thread started.")

    # --- Initialize Keyboard Handler (if interactive) ---
    kb_handler = None
    if IS_INTERACTIVE:
         kb_handler = KeyboardHandler()
         print("[Main] Keyboard handler active. Keys: (o)verride, (e)xpert mode, (q)uit")

    # --- Wait for Inference Worker to be Ready --- 
    print("[Main] Waiting for inference worker to signal readiness...")
    inference_ready_event.wait() # Block until event is set
    print("[Main] Inference worker ready. Starting server...")

    # --- Start Socket Server ---
    # Pass queues and main_agent (for step/target updates)
    socket_server = SocketServer(
        state_queue=state_queue, 
        action_queue=action_queue, 
        metrics=metrics, 
        main_agent_ref=main_agent # Pass agent reference
    )
    server_thread = threading.Thread(target=socket_server.start, name="SocketServerThread", daemon=True)
    server_thread.start()
    print("[Main] Socket server thread started.")

    # --- Start Stats Reporter Thread ---
    # This runs in the main process, reading shared metrics
    # Pass kb_handler if interactive
    stats_thread_args = (metrics,) if not IS_INTERACTIVE else (metrics, kb_handler)
    stats_thread = threading.Thread(
        target=run_stats_reporter, 
        args=stats_thread_args, 
        name="StatsReporterThread", 
        daemon=True
    )
    stats_thread.start()
    print("[Main] Stats reporter thread started.")

    # --- Keyboard Handling Loop Setup (Moved after kb_handler init) ---
    # Setup only needed if kb_handler was created
    # if kb_handler:
    #     print("[Main] Setting up keyboard loop logic...") # Logic moved to main loop

    # --- Graceful Shutdown Handling ---
    def signal_handler(sig, frame):
        print(f"\n[Main] Signal {sig} received. Initiating graceful shutdown...")
        shutdown_event.set() # Signal all processes/threads
        # Optionally tell server to stop accepting new clients
        socket_server.shutdown()

    signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler) # Handle termination signals

    # --- Main Loop (Keep alive, handle keyboard) ---
    try:
        frame_counter_cache = 0 # Cache frame count locally for decay checks
        while not shutdown_event.is_set():
            # --- Periodic Updates --- 
            # Update decay values based on frame count (use cached value)
            # Note: metrics.frame_count is updated by socket server threads
            current_frame_count = metrics.frame_count # Read thread-safe value
            if current_frame_count > frame_counter_cache:
                 decay_epsilon(metrics, current_frame_count)
                 decay_expert_ratio(metrics, current_frame_count)
                 frame_counter_cache = current_frame_count

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
                    # Fetch current metrics state under lock
                    try:
                        with metrics.lock:
                            metrics_state_to_save = {
                                'frame_count': metrics.frame_count,
                                'epsilon': metrics.epsilon,
                                'expert_ratio': metrics.expert_ratio,
                                'last_decay_step': metrics.last_decay_step
                            }
                        # Save with the fetched metrics state
                        main_agent.save(LATEST_MODEL_PATH, metrics_state=metrics_state_to_save)
                        print(f"[Main] Model saved to {LATEST_MODEL_PATH} (Frame: {metrics_state_to_save['frame_count']})")
                    except Exception as save_err:
                        print(f"[Main Error] Failed manual save: {save_err}")
                elif key == 'o':
                    # Toggle override flag directly
                    with metrics.lock:
                        metrics.override_expert = not metrics.override_expert
                    # Use clearer terminology
                    print_with_terminal_restore(kb_handler, f"Inference Override Toggled: {'ON' if metrics.override_expert else 'OFF'}")
                elif key == 'e':
                    # Toggle expert mode flag directly
                    with metrics.lock:
                         metrics.expert_mode = not metrics.expert_mode
                    print_with_terminal_restore(kb_handler, f"Expert Mode Toggled: {'ON' if metrics.expert_mode else 'OFF'}")
            
            # --- Health Checks ---
            # Check if essential threads/processes are alive
            if not server_thread.is_alive():
                 print("[Main Error] Socket server thread died unexpectedly. Shutting down.")
                 shutdown_event.set()
                 break
            if not inference_process.is_alive():
                 print("[Main Error] Inference process died unexpectedly. Shutting down.")
                 shutdown_event.set()
                 break
            if not training_process.is_alive():
                 print("[Main Error] Training process died unexpectedly. Shutting down.")
                 shutdown_event.set()
                 break
            if not batch_sampler.is_alive():
                 print("[Main Error] Batch sampler thread died unexpectedly. Shutting down.")
                 shutdown_event.set()
                 break
            if not loss_reporter.is_alive():
                 print("[Main Error] Loss reporter thread died unexpectedly. Shutting down.")
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
         shutdown_event.set() # Trigger shutdown on error
         socket_server.shutdown()

    finally:
        print("[Main] Entering shutdown sequence...")
        if kb_handler:
            kb_handler.restore_terminal()
            print("[Main] Terminal restored.")

        # Ensure shutdown event is set
        shutdown_event.set()
        print("[Main] Shutdown event confirmed set.")
        
        # Signal inference worker to stop (via Queue)
        try:
             state_queue.put(None, timeout=0.5) # Send sentinel
        except Full:
             print("[Main Warning] State queue full during shutdown signal for inference worker.")
        except Exception as sq_err:
             print(f"[Main Error] Error sending shutdown signal to inference worker: {sq_err}")
             
        # Wait for inference process to finish (with timeout)
        print("[Main] Waiting for inference process to terminate...")
        inference_process.join(timeout=5.0) 
        if inference_process.is_alive():
            print("[Main Warning] Inference process did not terminate gracefully. Terminating forcibly.")
            inference_process.terminate()
        else:
             print("[Main] Inference process terminated.")

        # Wait for training process to finish (with timeout)
        print("[Main] Waiting for training process to terminate...")
        training_process.join(timeout=10.0) 
        if training_process.is_alive():
            print("[Main Warning] Training process did not terminate gracefully. Terminating forcibly.")
            training_process.terminate() # Force terminate if needed
        else:
             print("[Main] Training process terminated.")

        # Wait for helper threads (should exit quickly after shutdown event)
        print("[Main] Waiting for helper threads...")
        # No need to explicitly join daemon threads usually, but can be done
        # server_thread.join(timeout=2.0)
        # batch_sampler.join(timeout=2.0)
        # loss_reporter.join(timeout=2.0)
        # stats_thread.join(timeout=2.0)
        print("[Main] Helper threads likely exited (daemon).")

        # Ensure server socket is closed (if not already)
        if hasattr(socket_server, 'server_socket') and socket_server.server_socket:
            try:
                socket_server.server_socket.close()
                print("[Main] Server socket closed.")
            except Exception as sock_err:
                 print(f"[Main Error] Error closing server socket: {sock_err}")

        # Final save of main agent with metrics
        # --> Add Diagnostic Print <--
        agent_ready_status = main_agent.is_ready if 'main_agent' in locals() else 'AgentNotFound'
        print(f"[Main DEBUG] Reached final save logic. Agent ready status: {agent_ready_status}")
        if 'main_agent' in locals() and main_agent.is_ready:
            print("[Main] Performing final save of agent and metrics...")
            try:
                 with metrics.lock:
                     metrics_state_to_save = {
                         'frame_count': metrics.frame_count,
                         'epsilon': metrics.epsilon,
                         'expert_ratio': metrics.expert_ratio,
                         'last_decay_step': metrics.last_decay_step
                     }
                 main_agent.save(LATEST_MODEL_PATH, metrics_state=metrics_state_to_save)
                 print(f"[Main] Final save complete (Frame: {metrics_state_to_save['frame_count']}).")
            except Exception as final_save_err:
                 print(f"[Main Error] Failed final save: {final_save_err}")

        print("[Main] Shutdown complete.")
        sys.exit(0) # Explicit exit

if __name__ == "__main__":
    main() 