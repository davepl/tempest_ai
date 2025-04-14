#!/usr/bin/env python3
"""
Tempest AI Main Entry Point
Coordinates the socket server, metrics display, and keyboard handling.
"""

import os
import time
import threading
from datetime import datetime
import traceback
import multiprocessing as mp
from multiprocessing import Process, Queue
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import setproctitle
import queue

from aimodel import (
    DQNAgent, KeyboardHandler
)
from config import (
    RL_CONFIG, MODEL_DIR, LATEST_MODEL_PATH, IS_INTERACTIVE, metrics, SERVER_CONFIG
)
from socket_server import SocketServer
from metrics_display import display_metrics_header, display_metrics_row

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

def inference_worker(state_queue: Queue, action_queue: Queue, model_path: str):
    setproctitle.setproctitle("python_inference")
    # Set process priority (Unix only)
    try:
        os.nice(-10)  # Higher priority
    except:
        pass  # Ignore if not supported

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[InferWorker] Starting. Using device: {device}. Loading model from: {model_path}")

    # Optimize PyTorch settings
    torch.set_num_threads(1)  # Use single thread for inference
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner

    model = None
    state_size = RL_CONFIG.state_size
    action_size = RL_CONFIG.action_size

    # Pre-allocate tensors
    state_tensor = torch.zeros((1, state_size), dtype=torch.float32, device=device)
    
    try:
        if os.path.exists(model_path):
            # Load and compile model
            model = DQN(state_size, action_size).to(device)
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['policy_state_dict'])
            model.eval()
            
            # Compile model for better performance
            try:
                model = torch.jit.script(model)
                print("[InferWorker] Model compiled with torch.jit.script")
            except Exception as compile_e:
                print(f"[InferWorker] Model compilation failed: {compile_e}. Using uncompiled model.")
            
            print("[InferWorker] Model loaded and optimized successfully.")
            
            # Warm up model
            try:
                with torch.no_grad():
                    _ = model(state_tensor)
                print("[InferWorker] Model warm-up prediction done.")
            except Exception as warmup_e:
                print(f"[InferWorker] Model warm-up failed: {warmup_e}")
        else:
            print(f"[InferWorker] Model file not found at {model_path}. Worker will not predict.")

        while True:
            try:
                state_data = state_queue.get(timeout=1.0)
                if state_data is None:
                    print("[InferWorker] Received shutdown signal. Exiting.")
                    break

                start_time = time.time()
                action_idx = 0
                
                if model is not None:
                    try:
                        # Copy state data directly into pre-allocated tensor
                        state_tensor.copy_(torch.from_numpy(state_data).float())
                        
                        # Run inference
                        with torch.no_grad():
                            q_values = model(state_tensor)
                        
                        inference_time = time.time() - start_time
                        action_idx = int(q_values.argmax().item())
                        
                        action_queue.put((action_idx, inference_time))
                        
                    except Exception as pred_e:
                        print(f"[InferWorker] Error during prediction: {pred_e}")
                        traceback.print_exc()
                        action_queue.put((0, 0.0))
                else:
                    action_queue.put((0, 0.0))
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[InferWorker] Error in main loop: {e}")
                traceback.print_exc()
                break

    except Exception as e:
        print(f"[InferWorker] An unexpected error occurred: {e}")
        traceback.print_exc()
    finally:
        print("[InferWorker] Finished.")

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
                    print("Save command received, saving model...")
                    agent.save(LATEST_MODEL_PATH)
                    print(f"Model saved to {LATEST_MODEL_PATH}")
                elif key == 'o':
                    metrics.toggle_override(keyboard_handler)
                    display_metrics_row(agent, keyboard_handler)
                elif key == 'e':
                    metrics.toggle_expert_mode(keyboard_handler)
                    display_metrics_row(agent, keyboard_handler)
            
            time.sleep(0.1)
        except Exception as e:
            print(f"Error in keyboard input handler: {e}")
            break

def main():
    setproctitle.setproctitle("python_socket")
    """Main function to run the Tempest AI application"""
    print("[Main] Starting main function...")
    # Ensure clean start for processes, especially on macOS/Windows
    # mp.set_start_method('spawn', force=True) # Uncomment if needed, place at very top if so

    print("[Main] Setting up multiprocessing queues...")
    state_queue = Queue()
    action_queue = Queue()
    print("[Main] Queues created.")

    # Start the inference worker process
    print("[Main] Starting inference worker process...")
    inference_p = Process(
        target=inference_worker,
        args=(state_queue, action_queue, LATEST_MODEL_PATH),
        daemon=True # Daemonize so it exits if main process exits unexpectedly
    )
    inference_p.start()
    # Check if process started immediately (optional)
    time.sleep(0.5) # Give a moment for process to potentially error out on startup
    if not inference_p.is_alive():
        print("[Main] CRITICAL: Inference worker process failed to start or exited immediately.")
        return # Exit if worker failed
    print(f"[Main] Inference worker process appears started (PID: {inference_p.pid}).")

    # Create model directory if it doesn't exist
    if not os.path.exists(MODEL_DIR):
        print(f"[Main] Creating model directory: {MODEL_DIR}")
        os.makedirs(MODEL_DIR)
    
    print("[Main] Initializing DQNAgent...")
    # Initialize the DQN agent
    agent = DQNAgent(
        state_size=RL_CONFIG.state_size,
        action_size=RL_CONFIG.action_size,
        learning_rate=RL_CONFIG.learning_rate,
        gamma=RL_CONFIG.gamma,
        epsilon=RL_CONFIG.epsilon,
        epsilon_min=RL_CONFIG.epsilon_min,
        epsilon_decay=RL_CONFIG.epsilon_decay,
        memory_size=RL_CONFIG.memory_size,
        batch_size=RL_CONFIG.batch_size
    )
    print("[Main] DQNAgent initialized.")
    
    # Load the model if it exists
    if os.path.exists(LATEST_MODEL_PATH):
        print(f"[Main] Attempting to load agent model from: {LATEST_MODEL_PATH}")
        agent.load(LATEST_MODEL_PATH)
        print(f"[Main] Agent model loaded.")
    else:
        print(f"[Main] Agent model file not found at {LATEST_MODEL_PATH}. Agent will start fresh.")
    
    print("[Main] Initializing SocketServer...")
    # Initialize the socket server
    server = SocketServer(
        SERVER_CONFIG.host,
        SERVER_CONFIG.port,
        agent, # Pass agent for remember/replay
        metrics,
        state_queue,  # Pass state queue
        action_queue # Pass action queue
    )
    print(f"[Main] SocketServer initialized for {SERVER_CONFIG.host}:{SERVER_CONFIG.port}.")
    
    # Set the global server reference in metrics
    metrics.global_server = server
    
    # Initialize client_count in metrics
    metrics.client_count = 0
    
    print("[Main] Starting server thread...")
    # Start the server in a separate thread
    server_thread = threading.Thread(target=server.start)
    server_thread.daemon = True
    server_thread.start()
    print("[Main] Server thread started.")
    
    # Set up keyboard handler for interactive mode
    keyboard_handler = None
    if IS_INTERACTIVE:
        print("[Main] Setting up keyboard handler...")
        keyboard_handler = KeyboardHandler()
        keyboard_handler.setup_terminal()
        keyboard_thread = threading.Thread(target=keyboard_input_handler, args=(agent, keyboard_handler))
        keyboard_thread.daemon = True
        keyboard_thread.start()
        print("[Main] Keyboard handler thread started.")
    else:
        print("[Main] Interactive mode disabled, skipping keyboard handler.")
    
    print("[Main] Starting stats reporter thread...")
    # Start the stats reporter in a separate thread
    stats_thread = threading.Thread(target=stats_reporter, args=(agent, keyboard_handler))
    stats_thread.daemon = True
    stats_thread.start()
    print("[Main] Stats reporter thread started.")
    
    # Track last save time
    last_save_time = time.time()
    save_interval = 300  # 5 minutes in seconds
    
    print("[Main] Entering main loop...")
    try:
        # Keep the main thread alive
        while server.running and inference_p.is_alive():
            # Main loop keeps running while server is up and worker is alive
            current_time = time.time()
            # Save model every 5 minutes
            if current_time - last_save_time >= save_interval:
                print("[Main] Periodic save triggered.")
                agent.save(LATEST_MODEL_PATH)
                last_save_time = current_time
            time.sleep(1) # Sleep to prevent busy-waiting

        # Check the reason for loop termination
        if not inference_p.is_alive():
             print("[Main] Loop terminated: Inference worker process died.")
             # Attempt to gracefully stop the server if it's still marked as running
             if server.running:
                  print("[Main] Signaling server thread to stop due to worker death...")
                  server.running = False # Signal server thread to stop
        elif not server.running:
             print("[Main] Loop terminated: Server stopped (likely via user command or internal error).")
        else:
             # Should not happen if loop condition is correct
             print("[Main] Loop terminated for unknown reason.")

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received, saving and shutting down...")
    finally:
        # Initiating shutdown sequence...
        print("Initiating shutdown sequence...")
        # 1. Signal Inference Worker to Stop
        if inference_p and inference_p.is_alive():
            print("Sending shutdown signal to inference worker...")
            try:
                state_queue.put(None) # Use None as the sentinel value
            except Exception as q_err:
                 print(f"Error putting shutdown signal in queue: {q_err}")

        # 2. Stop the Socket Server (if not already stopped)
        if hasattr(metrics, 'global_server') and metrics.global_server and metrics.global_server.running:
            print("Stopping socket server...")
            metrics.global_server.running = False # Signal server threads to stop

        # 3. Wait for Inference Worker to finish
        if inference_p and inference_p.is_alive():
             print(f"Waiting for inference worker (PID: {inference_p.pid}) to finish...")
             inference_p.join(timeout=10.0) # Wait for 10 seconds
             if inference_p.is_alive():
                 print("Inference worker did not exit gracefully after 10s, terminating.")
                 inference_p.terminate() # Force terminate if it doesn't exit
                 inference_p.join() # Wait for termination to complete
             else:
                 print("Inference worker finished gracefully.")
        else:
             print("Inference process not running or already finished at shutdown.")

        # 4. Wait for Server Thread to finish (add timeout)
        if 'server_thread' in locals() and server_thread.is_alive():
             print("Waiting for server thread to finish...")
             server_thread.join(timeout=5.0)
             if server_thread.is_alive():
                  print("Server thread did not finish gracefully.") # Cannot terminate threads easily

        # 5. Wait for Stats Thread (add timeout)
        if 'stats_thread' in locals() and stats_thread.is_alive():
            print("Waiting for stats thread...")
            stats_thread.join(timeout=2.0)

        # 6. Wait for Keyboard Thread (add timeout)
        if IS_INTERACTIVE and 'keyboard_thread' in locals() and keyboard_thread.is_alive():
            print("Waiting for keyboard thread...")
            # Keyboard thread checks server.running, should exit quickly after server stops
            keyboard_thread.join(timeout=2.0)
            # Restore terminal settings happens after join attempt
            if keyboard_handler:
                print("Restoring terminal settings...")
                keyboard_handler.restore_terminal()

        # 7. Save the model before exiting (if agent exists)
        if 'agent' in locals():
            print("Saving final model state...")
            try:
                agent.save(LATEST_MODEL_PATH)
                print(f"Final model state saved to {LATEST_MODEL_PATH}")
            except Exception as save_err:
                print(f"Error saving final model state: {save_err}")

        print("\nApplication shutdown complete.")

if __name__ == "__main__":
    main() 