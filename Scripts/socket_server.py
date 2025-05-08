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
    """Socket-based server to handle multiple clients"""
    def __init__(self, host, port, agent, safe_metrics):
        print(f"Initializing SocketServer on {host}:{port}")
        self.host = host
        self.port = port
        self.agent = agent
        self.metrics = safe_metrics
        self.running = True
        self.clients = {}  # Dictionary to track active clients
        self.client_states = {}  # Dictionary to store per-client state
        self.client_lock = threading.Lock()  # Lock for client dictionaries
        self.shutdown_event = threading.Event()  # Event to signal shutdown to client threads
        
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
                        'address': client_address,
                        'last_state': None,
                        'last_action_idx': None,
                        'last_action_source': None,
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
            client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
            client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)
            
            buffer_size = 32768
            
            # --- Initial Ping Handshake ---
            ping_ok = False
            try:
                client_socket.setblocking(True)
                client_socket.settimeout(2.0)
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
                    ready = select.select([client_socket], [], [], 0.01)
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
                        # Expected header format includes num_values as the first H (unsigned short)
                        header_format_peek = ">H" # Only need the first value
                        peek_size = struct.calcsize(header_format_peek)
                        if len(data) >= peek_size:
                            num_values_received = struct.unpack(header_format_peek, data[:peek_size])[0]
                            if num_values_received != SERVER_CONFIG.params_count:
                                # Raise a specific error if count mismatches config
                                raise ValueError(f"Parameter count mismatch! Expected {SERVER_CONFIG.params_count}, received {num_values_received}")
                        else:
                             # This should ideally not happen if length check above is correct
                             raise ConnectionError("Data too short to read parameter count")
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
                    
                    # Handle save signal from game
                    if frame.save_signal:
                        try:
                            # Ensure agent exists before saving
                            if hasattr(self, 'agent') and self.agent:
                                self.agent.save(LATEST_MODEL_PATH)
                            else:
                                print(f"Client {client_id}: Agent not available for saving.")
                        except Exception as e:
                            print(f"Client {client_id}: ERROR saving model: {e}")
                    
                    # Update global metrics
                    current_frame = self.metrics.update_frame_count()
                    self.metrics.update_epsilon()
                    self.metrics.update_expert_ratio()
                    self.metrics.update_game_state(frame.enemy_seg, frame.open_level)
                    
                    # Process previous step's results if available
                    if state.get('last_state') is not None and state.get('last_action_idx') is not None:
                         # Ensure agent exists before stepping
                        if hasattr(self, 'agent') and self.agent:
                            self.agent.step(
                                state['last_state'],
                                np.array([[state['last_action_idx']]]),
                                frame.reward,
                                frame.state,
                                frame.done
                            )
                        else:
                             print(f"Client {client_id}: Agent not available for step.")


                        # Track rewards
                        state['total_reward'] = state.get('total_reward', 0) + frame.reward

                        # Track which system's rewards based on the *previous* action source
                        prev_action_source = state.get('last_action_source')
                        if prev_action_source == "dqn":
                            state['episode_dqn_reward'] = state.get('episode_dqn_reward', 0) + frame.reward
                        elif prev_action_source == "expert":
                            state['episode_expert_reward'] = state.get('episode_expert_reward', 0) + frame.reward
                        # If prev_action_source is None or unknown, we don't add to either specific reward accumulator
                    
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
                    action_idx = None # Initialize action_idx
                    fire, zap, spinner = 0, 0, 0.0 # Default actions
                    action_source = "unknown" # Track source for this frame

                    # Ensure agent exists before deciding action
                    if hasattr(self, 'agent') and self.agent:
                        if random.random() < self.metrics.get_expert_ratio() and not self.metrics.is_override_active():
                            # Use expert system
                            fire, zap, spinner = get_expert_action(
                                frame.enemy_seg,
                                frame.player_seg,
                                frame.open_level,
                                frame.expert_fire,
                                frame.expert_zap
                            )
                            self.metrics.increment_guided_count()
                            action_source = "expert"
                            action_idx = expert_action_to_index(fire, zap, spinner)
                        else:
                            # Use DQN with current epsilon
                            start_time = time.perf_counter()
                            action_idx = self.agent.act(frame.state, self.metrics.get_epsilon())
                            end_time = time.perf_counter()
                            inference_time = end_time - start_time
                            
                            # Update inference metrics
                            with self.metrics.lock:
                                self.metrics.total_inference_time += inference_time
                                self.metrics.total_inference_requests += 1
                                
                            fire, zap, spinner = ACTION_MAPPING[action_idx]
                            action_source = "dqn"
                    else:
                         print(f"Client {client_id}: Agent not available for action generation.")
                         # Keep default action (0,0,0)
                         action_source = "none" # No action generated

                    # Store state and action for next iteration (only if action was generated)
                    if action_idx is not None:
                        state['last_state'] = frame.state
                        state['last_action_idx'] = action_idx
                        state['last_action_source'] = action_source # Store the source with the state

                    # Send action to game
                    game_fire, game_zap, game_spinner = encode_action_to_game(fire, zap, spinner)
                    try:
                        client_socket.sendall(struct.pack("bbb", game_fire, game_zap, game_spinner))
                    except (BrokenPipeError, ConnectionResetError):
                        print(f"Client {client_id}: Connection lost when sending action.")
                        break # Exit loop if connection is lost


                    # Periodic target network update (only from client 0)
                    if client_id == 0 and hasattr(self, 'agent') and self.agent and current_frame % RL_CONFIG.update_target_every == 0:
                        self.agent.update_target_network()

                    # Periodic model saving (only from client 0)
                    if client_id == 0 and hasattr(self, 'agent') and self.agent and current_frame % RL_CONFIG.save_interval == 0:
                        self.agent.save(LATEST_MODEL_PATH)


                except (BlockingIOError, InterruptedError):
                    # Expected with non-blocking socket, short sleep prevents busy-waiting
                    time.sleep(0.001)
                    continue
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

    def is_override_active(self):
        with self.client_lock:
            return self.metrics.override_expert
            
    def get_fps(self):
        with self.client_lock:
            return self.metrics.fps