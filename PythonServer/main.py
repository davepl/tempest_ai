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
from multiprocessing import Queue as MpQueue, Lock as MpLock, Process, Event as MpEvent
from threading import Thread, Event as ThreadEvent
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

# Import necessary modules from the project
from training import training_worker

# +++ Define the DQN model structure (must match the one in aimodel.py) +++
class DQN(nn.Module):
    """Deep Q-Network model (copied for inference worker)."""
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.out = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.out(x)
# +++ End DQN model definition +++

def inference_worker(state_queue: Queue, action_queue: Queue,
                     model_path: Path, shutdown_event: ThreadEvent,
                     inference_ready_event: ThreadEvent,
                     state_size: int, action_size: int):
    """Worker THREAD dedicated to DQN inference."""
    print("[InferThread] Starting...")

    inference_device_str = None
    agent = None

    try:
        # Determine Inference Device
        print("[InferThread] Determining device...")
        inference_device_str = 'cpu' if SERVER_CONFIG.cpu_inference else DEVICE
        print(f"[InferThread] Determined inference device string: '{inference_device_str}'...")

        # Limit threads (might be less critical in a thread, but keep for consistency)
        try:
            torch.set_num_threads(1)
            print("[InferThread] Set torch num_threads=1")
        except Exception as thread_err:
            print(f"[InferThread Warning] Failed to set torch threads: {thread_err}")

        # Initialize Agent
        print("[InferThread] Initializing DQNAgent...")
        # Agent created within this thread
        agent = DQNAgent(state_size, action_size, device=inference_device_str)
        print(f"[InferThread DEBUG] DQNAgent object created (Device: {agent.device}).")

        # Load Initial Model
        print(f"[InferThread DEBUG] Checking model path: {model_path} (Exists: {model_path.exists()})")
        if model_path.exists():
            print("[InferThread DEBUG] Attempting agent.load()...")
            success, _ = agent.load(model_path)
            print(f"[InferThread DEBUG] agent.load() returned: {success}")
            if success: print(f"[InferThread] Initial model loaded from {model_path}")
            else: print(f"[InferThread Warning] Failed to load initial model.")
        else: print(f"[InferThread Warning] Model file {model_path} not found.")

        # Set to eval mode
        print("[InferThread DEBUG] Setting policy_net to eval mode...")
        agent.policy_net.eval()
        print("[InferThread DEBUG] Eval mode set.")

        # Signal Main Thread that Worker is Ready
        print("[InferThread DEBUG] Setting inference_ready_event...")
        inference_ready_event.set()
        print("[InferThread] Signaled ready to main thread.")

        # --- Inference Loop ---
        print("[InferThread DEBUG] Entering inference loop...")
        last_model_check_time = 0
        model_check_interval = 300

        while not shutdown_event.is_set():
            try:
                # Check for model updates (logic remains the same)
                # ... (model update check logic) ...

                # Get state from the queue
                state = state_queue.get(timeout=0.5)
                if state is None:
                    print("[InferThread] Received None state, shutting down.")
                    break

                # Perform inference
                inf_start_time = time.time()
                action_idx = agent.act(state, epsilon=0.0) # state is np.int16 array
                inf_end_time = time.time()
                inference_time_sec = inf_end_time - inf_start_time

                # Put action and inference time back
                try:
                    action_queue.put((action_idx, inference_time_sec), block=False)
                except Full:
                    print("[InferThread Warning] Action queue full.")

            except Empty:
                continue # Queue was empty
            except (KeyboardInterrupt, SystemExit):
                print("[InferThread] Interrupted, shutting down...")
                break
            except Exception as e:
                print(f"[InferThread] Error in inference loop: {e}")
                traceback.print_exc()
                time.sleep(0.5)

        print("[InferThread] Exiting.")

    except Exception as worker_err:
        print(f"[InferThread FATAL ERROR] Unhandled exception: {worker_err}")
        traceback.print_exc()
        # Ensure ready event is set even on error so main doesn't hang forever
        if not inference_ready_event.is_set():
             inference_ready_event.set()

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
                        time.sleep(0.05) # Give trainer time to catch up
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
# --- Replaced with custom piecewise linear decay ---
# def decay_epsilon(metrics_obj, frame_count):
#     """Calculate and update decayed exploration rate."""
#     # This function assumes it's called periodically
#     new_epsilon = max(RL_CONFIG.epsilon_end, 
#                    RL_CONFIG.epsilon_start * 
#                    np.exp(-frame_count / RL_CONFIG.epsilon_decay))
#     metrics_obj.update_epsilon(new_epsilon)
#     return new_epsilon

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
    setproctitle.setproctitle("python_main") # Set title for main process

    # Use multiprocessing context for better compatibility
    ctx = mp.get_context('spawn') # 'spawn' is often safer than 'fork'

    # --- Queues and Events ---
    # For Inference Process
    state_queue = Queue(maxsize=100) # States TO inference worker (Use queue.Queue)
    action_queue = Queue(maxsize=100) # Actions FROM inference worker (Use queue.Queue)
    # For Training Process
    train_batch_queue = MpQueue(maxsize=RL_CONFIG.train_queue_size) 
    loss_queue = MpQueue(maxsize=RL_CONFIG.loss_queue_size)
    # Shutdown Signal
    shutdown_event = MpEvent()
    # Inference Ready Signal
    inference_ready_event = ThreadEvent()
    save_lock = MpLock() # Multiprocessing lock for saving

    # --- Initialize Main Agent (for Memory, Step, Target Updates) ---
    num_flattened_values = SERVER_CONFIG.params_count # Use config value
    state_size = num_flattened_values
    action_size = len(ACTION_MAPPING)
    print(f"[Main] State Size: {state_size}, Action Size: {action_size}")
    print(f"[Main] Global DEVICE determined by config: '{DEVICE}'")

    # --- Explicitly initialize main_agent on the global DEVICE ---
    print("[Main] Initializing main DQNAgent...")
    main_agent = DQNAgent(state_size, action_size, device=DEVICE) # <<< PASS GLOBAL DEVICE EXPLICITLY
    # The DQNAgent init will print its own device confirmation log

    # Load weights/metrics into main agent
    if LATEST_MODEL_PATH.exists():
        try:
            # DQNAgent.load now returns success status and metrics_state
            success, loaded_metrics_state = main_agent.load(LATEST_MODEL_PATH)
            if success:
                print(f"[Main] Loaded main agent model from {LATEST_MODEL_PATH}")
                if loaded_metrics_state:
                    print("[Main] Restoring metrics from checkpoint...")
                    with metrics.lock:
                         metrics.frame_count = loaded_metrics_state.get('frame_count', metrics.frame_count)
                         metrics.epsilon = loaded_metrics_state.get('epsilon', metrics.epsilon)
                         metrics.expert_ratio = loaded_metrics_state.get('expert_ratio', metrics.expert_ratio)
                         metrics.last_decay_step = loaded_metrics_state.get('last_decay_step', metrics.last_decay_step)
                    print(f"[Main] Metrics restored: Frame={metrics.frame_count}")
            else:
                 print(f"[Main Warning] Loading main agent model from {LATEST_MODEL_PATH} failed (see agent log). Starting fresh.")
        except Exception as e:
             print(f"[Main Error] Error processing model load for main agent: {e}. Starting fresh.")
             traceback.print_exc()
    else:
        print(f"[Main] Agent model file not found at {LATEST_MODEL_PATH}. Main agent will start fresh.")

    # --- Start Inference Thread --- 
    print("[Main] Starting inference thread...")
    inference_thread = Thread(
        target=inference_worker,
        args=(state_queue, action_queue, LATEST_MODEL_PATH,
              shutdown_event, inference_ready_event,
              state_size, action_size),
        name="InferenceThread",
        daemon=True
    )
    inference_thread.start()
    print("[Main] Inference thread started.")

    # --- Start Dedicated Training Process --- 
    print("[Main] Starting training process...")
    training_process = Process(
        target=training_worker,
        args=(train_batch_queue, loss_queue, shutdown_event, save_lock, state_size, action_size),
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

    # --- Wait for Inference Thread to be Ready --- 
    print("[Main] Waiting for inference thread to signal readiness...")
    ready = inference_ready_event.wait(timeout=30.0) # Keep timeout
    if ready:
        print("[Main] Inference thread ready. Starting server...")
    else:
        print("[Main Error] Timed out waiting for inference thread!")
        shutdown_event.set()
        # No need to terminate inference_thread explicitly if it's a daemon
        if 'training_process' in locals() and training_process.is_alive(): training_process.terminate()
        sys.exit(1)

    # --- Start Socket Server ---
    # Pass queues and main_agent (for step/target updates)
    socket_server = SocketServer(
        state_queue=state_queue, 
        action_queue=action_queue, 
        metrics=metrics, 
        main_agent_ref=main_agent,
        shutdown_event=shutdown_event # Pass the mp.Event
    )
    server_thread = threading.Thread(target=socket_server.start, name="SocketServerThread", daemon=True)
    server_thread.start()
    print("[Main] Socket server thread started.")

    # --- Start Stats Reporter Thread ---
    # This runs in the main process, reading shared metrics
    print("[Main] Starting stats reporter thread...")
    stats_thread = threading.Thread(
        target=run_stats_reporter,
        # Pass metrics, shutdown_event, main_agent reference, and kb_handler
        args=(metrics, shutdown_event, main_agent, kb_handler), # <<< ADD shutdown_event
        name="StatsReporterThread",
        daemon=True
    )
    stats_thread.start()
    # print("[Main] Stats reporter thread started.") # Redundant

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
        last_save_frame = -1 # Track the frame number of the last save
        
        while not shutdown_event.is_set():
            # --- Periodic Updates --- 
            # Update decay values based on frame count (use cached value)
            # Note: metrics.frame_count is updated by socket server threads
            current_frame_count = metrics.frame_count # Read thread-safe value
            if current_frame_count > frame_counter_cache:
                 decay_epsilon(metrics, current_frame_count)
                 decay_expert_ratio(metrics, current_frame_count)
                 frame_counter_cache = current_frame_count

            # --- Periodic Save based on frame count --- 
            # Check if we crossed a save interval boundary
            if RL_CONFIG.save_interval > 0 and current_frame_count // RL_CONFIG.save_interval > last_save_frame // RL_CONFIG.save_interval:
                 # Check if agent is ready before saving
                 if 'main_agent' in locals() and main_agent.is_ready:
                     print(f"[Main] Frame {current_frame_count}: Triggering periodic save (Interval: {RL_CONFIG.save_interval})...")
                     try:
                          with metrics.lock:
                              metrics_state_to_save = {
                                  'frame_count': metrics.frame_count,
                                  'epsilon': metrics.epsilon,
                                  'expert_ratio': metrics.expert_ratio,
                                  'last_decay_step': metrics.last_decay_step
                              }
                          # ---> Add context to save call <---                          
                          # print("[Main Periodic Save] Calling agent.save()...") # Add context
                          main_agent.save(LATEST_MODEL_PATH, metrics_state=metrics_state_to_save)
                          # Log success after the save method prints its own message
                          last_save_frame = current_frame_count # Update last save frame
                     except Exception as save_err:
                          print(f"[Main Error] Failed periodic save: {save_err}")
                 else:
                     print(f"[Main] Frame {current_frame_count}: Skipped periodic save (Agent not ready)...")
            
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
                        # ---> Add context to save call <---   
                        # print("[Main Manual Save] Calling agent.save()...") # Add context                     
                        main_agent.save(LATEST_MODEL_PATH, metrics_state=metrics_state_to_save)
                        # Log success after the save method prints its own message
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
                elif key == ' ': # Spacebar
                    # print("[Main] SPACE pressed. Displaying current stats...")
                    # Manually trigger calculations for immediate display
                    # Note: This reads the *current* interval data, reporter thread resets it.
                    current_avg_inf = metrics.calculate_interval_avg_inf_time()
                    current_avg_dqn_f_reward = metrics.calculate_interval_avg_dqn_frame_reward()
                    # FPS is calculated frequently, just read the latest
                    # current_fps = metrics.calculate_interval_fps() # Or just metrics.get_fps()
                    
                    # Display the row (pass necessary args, kb_handler for restore)
                    display_metrics_row(metrics, main_agent, kb_handler)
                    # No need to pass avg_dqn_reward_per_frame as display_metrics_row reads it
            
            # --- Health Checks ---
            # Check if essential threads/processes are alive
            if not server_thread.is_alive():
                 print("[Main Error] Socket server thread died unexpectedly. Shutting down.")
                 shutdown_event.set()
                 break
            if not inference_thread.is_alive():
                 print("[Main Error] Inference thread died unexpectedly. Shutting down.")
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
        
        # Wait for helper threads (should exit quickly after shutdown event)
        print("[Main] Waiting for helper threads...")
        if 'batch_sampler' in locals() and batch_sampler.is_alive(): batch_sampler.join(timeout=2.0)
        if 'loss_reporter' in locals() and loss_reporter.is_alive(): loss_reporter.join(timeout=2.0)
        if 'stats_thread' in locals() and stats_thread is not None and stats_thread.is_alive():
            stats_thread.join(timeout=2.0)
        print("[Main] Helper threads likely exited or timed out.")

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
                 # ---> Add context to save call <---                    
                 # print("[Main Final Save] Calling agent.save()...") # Add context
                 main_agent.save(LATEST_MODEL_PATH, metrics_state=metrics_state_to_save)
                 # Log success after the save method prints its own message
            except Exception as final_save_err:
                 print(f"[Main Error] Failed final save: {final_save_err}")

        print("[Main] Shutdown complete.")
        sys.exit(0) # Explicit exit

if __name__ == "__main__":
    main() 