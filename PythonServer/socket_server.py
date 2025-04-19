#!/usr/bin/env python3
"""
Socket server for Tempest AI.
"""

# Prevent direct execution
if __name__ == "__main__":
    print("This is not the main application, run 'main.py' instead")
    exit(1)

import os
import sys
import time
import socket
import select
import struct
import threading
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import random
import traceback
from datetime import datetime
import multiprocessing as mp # Added for queue Empty exception
from queue import Full, Empty

# Import from config.py
from config import (
    SERVER_CONFIG,
    MODEL_DIR,
    LATEST_MODEL_PATH,
    ACTION_MAPPING,
    metrics,
    RL_CONFIG
)

# Import from aimodel.py
from aimodel import (
    parse_frame_data, 
    get_expert_action, 
    expert_action_to_index, 
    encode_action_to_game
)

class SocketServer:
    """Socket-based server to handle multiple clients (Single Process Version)"""
    def __init__(self, metrics=None, main_agent_ref=None,
                 host=SERVER_CONFIG.host, port=SERVER_CONFIG.port,
                 shutdown_event=None, training_job_queue=None): # Add training_job_queue
        print(f"Initializing SocketServer on {host}:{port}")
        self.host = host
        self.port = port
        # Removed state_queue and action_queue
        self.main_agent_ref = main_agent_ref # Reference to the single agent
        self.metrics = metrics # Shared metrics object
        self.training_job_queue = training_job_queue # Queue to signal training thread
        self.running = True
        self.clients = {}  # Dictionary to track active clients (thread objects or None)
        self.client_states = {}  # Dictionary to store per-client state
        self.client_lock = threading.Lock()  # Lock for client dictionaries
        self.shutdown_event = shutdown_event  # Event to signal shutdown to client threads
        
        # Use passed metrics object
        if self.metrics is None:
            from config import metrics as global_metrics # Fallback
            self.metrics = global_metrics
            print("[SocketServer Warning] Metrics object not passed, using global fallback.")
        
    def start(self):
        """Start the socket server"""
        try:
            # Create server socket
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(SERVER_CONFIG.max_clients)
            
            # print(f"Server started on {self.host}:{self.port}, waiting for connections...")
            
            # Accept client connections in a loop
            while self.running:
                try:
                    # Accept new connection (timeout after 1 second to check running flag)
                    self.server_socket.settimeout(1.0)
                    client_socket, client_address = self.server_socket.accept()
                    
                    # Generate a unique client ID
                    client_id = self.generate_client_id()
                    
                    # print(f"New connection from {client_address[0]}:{client_address[1]}, ID: {client_id}")
                    
                    # Initialize client state
                    client_state = {
                        'socket': client_socket,
                        'address': client_address,
                        'last_state': None,
                        'last_action_idx': None,
                        'total_reward': 0,
                        'was_done': False,
                        'episode_dqn_reward': 0,
                        'episode_expert_reward': 0,
                        'connected_time': datetime.now(),
                        'frames_processed': 0,
                        'fps': 0.0,  # Track per-client FPS
                        'frames_last_second': 0,  # Frames in current second
                        'last_fps_update': time.time()  # Last FPS calculation time
                    }
                    
                    # Store client information
                    with self.client_lock:
                        self.client_states[client_id] = client_state
                        self.clients[client_id] = client_socket
                        # Update metrics.client_count
                        metrics.client_count = len(self.clients)
                    
                    # Start a thread to handle this client
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, client_id),
                        daemon=True,
                        name=f"T-{client_address[0]}:{client_address[1]}"
                    )
                    
                    # Store thread and start it
                    with self.client_lock:
                        self.clients[client_id] = client_thread
                    
                    client_thread.start()
                    
                except socket.timeout:
                    # This is expected - just a way to periodically check self.running
                    continue
                except Exception as e:
                    print(f"Error accepting client connection: {e}")
                    traceback.print_exc()
                    time.sleep(1)  # Brief pause on error
            
        except Exception as e:
            print(f"Server error: {e}")
            traceback.print_exc()
        finally:
            print("Server shutting down...")
            # Signal all client threads to shut down
            self.shutdown_event.set()
            
            # Close the server socket
            try:
                self.server_socket.close()
                print("Server socket closed")
            except Exception as e:
                print(f"Error closing server socket: {e}")
                
            # Wait for all client threads to finish
            self.cleanup_all_clients()
            print("Server stopped")
            
    def shutdown(self):
        """Initiates the server shutdown sequence."""
        print("[SocketServer] Shutdown requested.")
        self.running = False # Stop accepting new connections
        self.shutdown_event.set() # Signal client threads to stop
        
        # --- Forcefully close active client sockets --- 
        with self.client_lock:
            client_sockets_to_close = []
            # Iterate over a copy of items in case the dictionary changes
            for client_id, thread_or_socket in self.clients.items():
                # In the single-process model, self.clients holds threads
                # We need to find the actual socket. Let's assume client_states holds it if thread is alive?
                # This is a bit messy. Let's refine the client tracking first.
                # For now, let's assume we need a way to get the socket associated with the client_id.
                # We'll need to modify handle_client to store the socket reference perhaps.
                # --- TEMPORARY APPROACH: Requires client_socket stored in client_states ---
                if client_id in self.client_states:
                    client_state = self.client_states[client_id]
                    if 'socket' in client_state:
                         client_sockets_to_close.append(client_state['socket'])

            print(f"[SocketServer] Attempting to close {len(client_sockets_to_close)} active client sockets...")
            for sock in client_sockets_to_close:
                try:
                    # Shutdown both read/write ends
                    sock.shutdown(socket.SHUT_RDWR)
                except (OSError, socket.error) as shut_err:
                    # Ignore errors if socket is already closed/invalid
                    # print(f"[SocketServer] Info: Error shutting down socket: {shut_err}") 
                    pass 
                finally:
                    try:
                        # Ensure close is called
                        sock.close()
                    except (OSError, socket.error) as close_err:
                        # print(f"[SocketServer] Info: Error closing socket: {close_err}")
                        pass
        print("[SocketServer] Client socket close attempts complete.")
        # --- End forceful close --- 

        # Closing the server socket is handled in the finally block of start()
        # after threads have been potentially joined/signaled.

    def cleanup_all_clients(self):
        """Clean up all client threads and resources"""
        with self.client_lock:
            # Get all client IDs
            client_ids = list(self.clients.keys())
            
            # Remove all clients from tracking
            for client_id in client_ids:
                if client_id in self.clients:
                    del self.clients[client_id]
                if client_id in self.client_states:
                    del self.client_states[client_id]
            
            print(f"Cleaned up all {len(client_ids)} clients during shutdown")
    
    def generate_client_id(self):
        """Generate a unique client ID"""
        with self.client_lock:
            # --- Method 1: Reuse available IDs within max_clients ---
            all_possible_ids = set(range(SERVER_CONFIG.max_clients))
            # Get keys *currently* in the clients dictionary
            current_ids = set(self.clients.keys())
            available_ids = list(all_possible_ids - current_ids)

            if available_ids:
                available_ids.sort()
                # print(f"DEBUG: Reusing available ID {available_ids[0]}") # Debug print
                return available_ids[0]

            # --- Method 2: Clean up dead threads and reuse their ID ---
            disconnected_ids_found = []
            # Iterate over a copy of items for safety during potential modifications
            items_copy = list(self.clients.items())
            for client_id, thread_or_none in items_copy:
                # Check if it's an integer ID (ignore potential non-int keys if any)
                if not isinstance(client_id, int):
                    continue

                # Check if thread is None OR if it's a Thread object but not alive
                is_disconnected = False
                if thread_or_none is None:
                    is_disconnected = True
                elif isinstance(thread_or_none, threading.Thread) and not thread_or_none.is_alive():
                     is_disconnected = True
                # elif not isinstance(thread_or_none, threading.Thread): # Handle cases where value isn't None or Thread (shouldn't happen ideally)
                #     print(f"DEBUG: Found unexpected type for client {client_id}: {type(thread_or_none)}")
                #     is_disconnected = True # Treat unexpected types as disconnected

                if is_disconnected:
                    disconnected_ids_found.append(client_id)
                    # Clean up this specific ID immediately since we found it dead
                    if client_id in self.clients:
                        # print(f"DEBUG: Cleaning up dead client {client_id} during ID generation") # Debug print
                        del self.clients[client_id]
                    # ADDED BACK: Clean up state immediately before reusing ID
                    if client_id in self.client_states:
                        del self.client_states[client_id]

            # If we cleaned up any dead clients, update count and return the first ID found
            if disconnected_ids_found:
                metrics.client_count = len(self.clients) # Update count after removal
                reused_id = disconnected_ids_found[0]
                # print(f"DEBUG: Reusing ID {reused_id} after immediate cleanup. Active: {metrics.client_count}") # Debug print
                return reused_id

            # --- Method 3: Assign an ID above max_clients ---
            overflow_id = SERVER_CONFIG.max_clients + len(self.clients)
            # print(f"DEBUG: Assigning overflow ID {overflow_id}. Active: {len(self.clients)}") # Debug print
            return overflow_id
    
    def handle_client(self, client_socket, client_id):
        """Handle communication with a client"""
        try:
            # Set socket to non-blocking mode
            client_socket.setblocking(False)
            
            # Disable Nagle's algorithm to reduce latency
            client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            
            # Increase socket buffer sizes for better performance
            client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 131072)  # 128KB
            client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 131072)  # 128KB
            
            buffer_size = 65536  # 64KB
            
            # --- Initial Ping Handshake ---
            ping_ok = False
            try:
                client_socket.setblocking(True)
                client_socket.settimeout(1.0)  # Reduced from 2.0
                ping_header = client_socket.recv(2)
                if not ping_header or len(ping_header) < 2:
                    print(f"Client {client_id} disconnected (no initial ping header)")
                else:
                    ping_ok = True
            except (socket.timeout, ConnectionResetError, BrokenPipeError) as e:
                print(f"Client {client_id} error during handshake: {e}")
            finally:
                client_socket.setblocking(False)
                client_socket.settimeout(None)
                
            if not ping_ok:
                raise ConnectionError("Handshake failed")

            # Main communication loop
            while self.running and not self.shutdown_event.is_set():
                try:
                    ready = select.select([client_socket], [], [], 0.001)  # Reduced from 0.01
                    if not ready[0]:
                        continue

                    # Receive data length
                    length_data = client_socket.recv(2)
                    if not length_data or len(length_data) < 2:
                        raise ConnectionError("Client disconnected (failed to read length)")

                    data_length = struct.unpack(">H", length_data)[0]
                    
                    data = b""
                    remaining = data_length

                    # Ensure we read the entire message based on length header
                    while remaining > 0:
                        chunk = client_socket.recv(min(buffer_size, remaining))
                        if not chunk:
                            raise ConnectionError("Connection broken during data receive")
                        data += chunk
                        remaining -= len(chunk)

                    if len(data) < data_length:
                        print(f"Client {client_id}: Received incomplete data packet ({len(data)}/{data_length} bytes)")
                        continue

                    # --- Parameter Count Validation ---
                    # Peek at the number of values from the OOB header *before* full parsing
                    try:
                        # Parameter count (H) starts after the 4-byte frame count (I)
                        header_format_peek = ">H" # Format of the count field itself
                        peek_size = struct.calcsize(header_format_peek)
                        oob_count_offset = 4 # Offset of the count field (after >I)
                        if len(data) >= oob_count_offset + peek_size:
                            # Unpack starting from the offset
                            num_values_received = struct.unpack(header_format_peek, data[oob_count_offset:oob_count_offset+peek_size])[0]
                            # expected_count = SERVER_CONFIG.params_count # Redundant check now?
                            # print(f"Client {client_id}: Parameter Count Check - Received={num_values_received}, Expected={expected_count}") # DEBUG
                            if num_values_received != SERVER_CONFIG.params_count:
                                # Raise a specific error if count mismatches config
                                raise ValueError(f"Parameter count mismatch! Expected {SERVER_CONFIG.params_count}, received {num_values_received}")
                        else:
                             # This should ideally not happen if length check above is correct
                             raise ConnectionError(f"Data too short ({len(data)} bytes) to read parameter count at offset {oob_count_offset}")
                    except ValueError as ve:
                         # Catch the specific ValueError for parameter count mismatch
                         print(f"Client {client_id}: {ve}. Closing connection.")
                         break # Exit the loop to close the connection
                    except Exception as e:
                         # Catch other potential unpacking errors during peek
                         print(f"Client {client_id}: Error checking parameter count: {e}. Closing connection.")
                         traceback.print_exc()
                         break # Exit the loop

                    # --- End Parameter Count Validation ---

                    # Parse the full frame data (only if count check passed)
                    frame = parse_frame_data(data)
                    if not frame:
                        print(f"Client {client_id}: Failed to parse frame data after count check.")
                        # Send empty response on parsing failure
                        client_socket.sendall(struct.pack("bbb", 0, 0, 0))
                        continue

                    # Get client state
                    with self.client_lock:
                        # Check if client_id still exists before accessing state
                        if client_id not in self.client_states:
                            print(f"Client {client_id}: State not found, likely disconnected during processing. Aborting frame.")
                            break # Exit loop if state is gone

                        state = self.client_states[client_id]
                        state['frames_processed'] += 1
                        
                        # Update client-specific FPS tracking
                        current_time = time.time()
                        state['frames_last_second'] += 1
                        elapsed = current_time - state['last_fps_update']
                        
                        # Calculate client FPS every second
                        if elapsed >= 1.0:
                            state['fps'] = state['frames_last_second'] / elapsed
                            state['frames_last_second'] = 0
                            state['last_fps_update'] = current_time
                    
                    # Update global metrics
                    current_frame = self.metrics.update_frame_count()
                    self.metrics.update_epsilon()
                    self.metrics.update_expert_ratio()
                    self.metrics.update_game_state(frame.enemy_seg, frame.open_level)
                    
                    # Process previous step's results if available
                    last_state_for_step = None # Keep track of state/action for step call after send
                    last_action_idx_for_step = None
                    # ---> MOVE step call to after sendall <---
                    # if state.get('last_state') is not None and state.get('last_action_idx') is not None:
                    #      # Ensure agent exists before stepping
                    #     if hasattr(self, 'main_agent_ref') and self.main_agent_ref:
                    #         self.main_agent_ref.step(
                    #              state['last_state'],
                    #              np.array([[state['last_action_idx']]]),
                    #              frame.reward,
                    #              frame.state,
                    #              frame.done
                    #          )
                    #     else:
                    #          print(f"Client {client_id}: Agent not available for step.")

                    # ---> Store state/action for step call later <---
                    if state.get('last_state') is not None and state.get('last_action_idx') is not None:
                        last_state_for_step = state['last_state']
                        last_action_idx_for_step = state['last_action_idx']


                        # Track rewards
                        state['total_reward'] = state.get('total_reward', 0) + frame.reward

                        # Track which system's rewards
                        if hasattr(metrics, 'last_action_source'):
                            if metrics.last_action_source == "expert":
                                state['episode_expert_reward'] = state.get('episode_expert_reward', 0) + frame.reward
                            else:
                                state['episode_dqn_reward'] = state.get('episode_dqn_reward', 0) + frame.reward
                    
                    # Handle episode completion
                    if frame.done:
                        if not state.get('was_done', False):
                             # Add rewards only if the episode wasn't already marked done
                             self.metrics.add_episode_reward(
                                 state.get('total_reward', 0),
                                 state.get('episode_dqn_reward', 0),
                                 state.get('episode_expert_reward', 0)
                             )

                        state['was_done'] = True
                        # Send empty action on 'done' frame to prevent issues
                        try:
                            client_socket.sendall(struct.pack("bbb", 0, 0, 0))
                        except (BrokenPipeError, ConnectionResetError):
                             print(f"Client {client_id}: Connection lost when sending done confirmation.")
                             break # Exit loop if connection is lost


                        # Reset state for next episode
                        state['last_state'] = None
                        state['last_action_idx'] = None
                        state['total_reward'] = 0
                        state['episode_dqn_reward'] = 0
                        state['episode_expert_reward'] = 0
                        continue # Skip action generation for this frame

                    elif state.get('was_done', False):
                         # Reset episode state if previous frame was done
                         state['was_done'] = False
                         state['total_reward'] = 0
                         state['episode_dqn_reward'] = 0
                         state['episode_expert_reward'] = 0
                    
                    # Generate action (only if not 'done')
                    self.metrics.increment_total_controls()
                    
                    # Decide between expert system and DQN
                    action_idx = None
                    fire, zap, spinner = 0, 0, 0.0 # Default actions
                    inference_time_ms = 0.0 # Initialize inference time

                    # Ensure main_agent_ref exists before deciding action
                    if hasattr(self, 'main_agent_ref') and self.main_agent_ref:
                         # --- Action Decision Logic ---
                         # Get decision parameters once to reduce locking
                         effective_ratio, override_active, expert_mode_active, current_epsilon = self.metrics.get_decision_params()
                         
                         # Decide between expert system and DQN based on effective ratio
                         # Also check epsilon for exploration during DQN choice
                         use_expert = random.random() < effective_ratio and not override_active
                         explore = random.random() < current_epsilon # Check for epsilon-greedy exploration

                         if use_expert:
                              # --- Use expert system (No change here) ---
                              start_expert_time = time.time()
                              fire, zap, spinner = get_expert_action(
                                  frame.enemy_seg, frame.player_seg, frame.open_level
                              )
                              action_idx = expert_action_to_index(fire, zap, spinner)
                              expert_time_ms = (time.time() - start_expert_time) * 1000
                              self.metrics.increment_guided_count()
                              self.metrics.update_action_source("expert")
                              self.metrics.update_expert_inference_time(expert_time_ms)
                              # --- End Expert System ---
                         elif explore:
                              # --- Epsilon-Greedy Exploration --- 
                              self.metrics.update_action_source("explore")
                              action_idx = random.randrange(self.main_agent_ref.action_size)
                              fire, zap, spinner = ACTION_MAPPING[action_idx]
                              # No specific inference time for random action
                              # print(f"[Explore Action] Client {client_id}: Chose Index={action_idx}") # Optional log
                              # --- End Exploration ---
                         else:
                              # --- Use DQN Agent Directly ---
                              start_dqn_time = time.time()
                              self.metrics.update_action_source("dqn")
                              
                              # Ensure agent is ready and in eval mode for inference
                              if self.main_agent_ref.is_ready:
                                  self.main_agent_ref.policy_net.eval() # Set to eval mode
                                  # Direct call to the agent's act method
                                  action_idx = self.main_agent_ref.act(frame.state, epsilon=0.0) # Use epsilon=0 for exploitation
                              else:
                                  print(f"Client {client_id}: Agent not ready, using default action.")
                                  action_idx = 0 # Default action

                              # Get action details
                              fire, zap, spinner = ACTION_MAPPING[action_idx]

                              # Calculate inference time
                              inference_time_sec = time.time() - start_dqn_time
                              inference_time_ms = inference_time_sec * 1000

                              self.metrics.add_inference_time(inference_time_sec)

                              # --- End DQN Agent --- 
                    else:
                         # Handle case where main_agent_ref might be None (though unlikely now)
                         print(f"Client {client_id}: Main agent reference not available for action generation.")
                         action_idx = 0
                         fire, zap, spinner = ACTION_MAPPING[action_idx]


                    # Store state and action for next iteration (only if action was generated)
                    if action_idx is not None:
                        # Check if client state still exists before writing
                        with self.client_lock:
                            if client_id in self.client_states:
                                state = self.client_states[client_id]
                                state['last_state'] = frame.state
                                state['last_action_idx'] = action_idx
                                # ---> STORE Full Last Action Tuple for Debugging <---
                                state['last_action'] = (fire, zap, spinner) # Store commanded action
                            else:
                                 print(f"Client {client_id}: State disappeared before storing last_state/action. Disconnecting.")
                                 break # Exit loop

                    # Send action to game
                    game_fire, game_zap, game_spinner = encode_action_to_game(fire, zap, spinner)
                    try:
                        client_socket.sendall(struct.pack("bbb", game_fire, game_zap, game_spinner))
                    except (BrokenPipeError, ConnectionResetError):
                        print(f"Client {client_id}: Connection lost when sending action.")
                        break # Exit loop if connection is lost

                    # --- Perform step and metrics updates AFTER sending response ---
                    # Call agent step if we have previous state/action
                    if last_state_for_step is not None and last_action_idx_for_step is not None:
                        # ---> Add reward to DQN frame totals if applicable <--- 
                        # Check the action source from the *previous* step (which led to this reward)
                        if self.metrics and self.metrics.last_action_source == "dqn":
                             self.metrics.add_dqn_frame_reward(frame.reward)
                        # ---> End Add <--- 
                        
                        if hasattr(self, 'main_agent_ref') and self.main_agent_ref:
                            # Retrieve the actual fire/zap/spinner values from client state
                            last_action_tuple = None
                            with self.client_lock:
                                if client_id in self.client_states:
                                    last_action_tuple = self.client_states[client_id].get('last_action')
                            
                            # ---> Print Replay Buffer Add Info <--- 
                            # if last_action_idx_for_step is not None and frame is not None:
                            #   print(f"[ReplayBuffer] Frame(s'): {frame.lua_frame_count}, Action(a): {last_action_idx_for_step}, Reward(r): {frame.reward:.2f}")
                            # ---> End Print <--- 
                            
                            self.main_agent_ref.step(
                                last_state_for_step,
                                np.array([[last_action_idx_for_step]]),
                                frame.reward,
                                frame.state,
                                frame.done
                            )
                        else:
                             print(f"Client {client_id}: Agent not available for step (post-send).")

                    # Log inference time if applicable (DQN path)
                    if self.metrics and hasattr(self, 'metrics') and self.metrics.last_action_source == "dqn" and 'inference_time_sec' in locals():
                         self.metrics.add_inference_time(inference_time_sec)
                    # --- End Post-Send Updates ---

                    # ---> Trigger training thread periodically <--- 
                    if current_frame % RL_CONFIG.train_freq == 0:
                        try:
                            # Send a signal (e.g., True) to the training queue
                            self.training_job_queue.put(True, block=False)
                        except Full:
                            # print("[SocketServer] Training job queue full, skipping trigger.") # Optional log
                            pass

                except (BlockingIOError, InterruptedError):
                    # Expected with non-blocking socket, short sleep prevents busy-waiting
                    time.sleep(0.0001)  # Reduced from 0.001
                except (ConnectionResetError, BrokenPipeError, ConnectionError) as e:
                    print(f"Client {client_id} connection error: {e}")
                    break # Exit the loop on connection errors
                except Exception as e:
                    print(f"Error handling client {client_id}: {e}")
                    traceback.print_exc()
                    break # Exit the loop on other errors

        except Exception as e:
            print(f"Fatal error handling client {client_id}: {e}")
            traceback.print_exc()
        finally:
            # Ensure proper cleanup in all cases
            try:
                client_socket.shutdown(socket.SHUT_RDWR)
            except:
                pass # Ignore errors during shutdown
            try:
                client_socket.close()
            except:
                pass # Ignore errors during close

            # Clean up client state atomically
            with self.client_lock:
                client_exists = client_id in self.client_states
                if client_exists:
                    del self.client_states[client_id]
                if client_id in self.clients:
                     # Replace thread with None to mark for cleanup
                     self.clients[client_id] = None

                # Update metrics after cleanup
                metrics.client_count = len([c for c in self.clients.values() if c is not None])
                if client_exists:
                    print(f"Client {client_id} cleanup complete. Active clients: {metrics.client_count}")

            # Schedule cleanup of disconnected clients (thread safe)
            threading.Timer(1.0, self.cleanup_disconnected_clients).start()

    def cleanup_disconnected_clients(self):
        """Clean up any disconnected clients marked as None to free up their IDs"""
        cleaned_count = 0
        with self.client_lock:
            # Find clients marked as None
            disconnected_ids = [
                client_id for client_id, thread_or_none in self.clients.items()
                if thread_or_none is None
            ]

            # Remove disconnected clients from the main dictionary
            for client_id in disconnected_ids:
                if client_id in self.clients:
                    del self.clients[client_id]
                    cleaned_count += 1

            # Update metrics if needed
            if cleaned_count > 0:
                metrics.client_count = len(self.clients)
                print(f"Background cleanup removed {cleaned_count} disconnected clients. Active: {metrics.client_count}")

    def is_override_active(self):
        with self.client_lock:
            return self.metrics.override_expert
            
    def get_fps(self):
        with self.client_lock:
            return self.metrics.fps 