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
            
            print(f"Cleaned up all {len(client_ids)} clients during shutdown")
    
    def generate_client_id(self):
        """Generate a unique client ID"""
        with self.client_lock:
            # Check for reusable IDs first (clients that have disconnected)
            for i in range(SERVER_CONFIG.max_clients):
                if i not in self.clients:
                    return i
                    
            # Before falling back to overflow, try to clean up disconnected clients
            # that might still be in the dictionary but are no longer active
            disconnected_ids = []
            for client_id, thread in self.clients.items():
                if isinstance(client_id, int) and not thread.is_alive():
                    disconnected_ids.append(client_id)
                    
            # If we found disconnected clients, reuse the first ID
            if disconnected_ids:
                reused_id = disconnected_ids[0]
                return reused_id
                
            # If we still can't find an ID, use a numeric ID above the max_clients
            # but keep it as a number instead of a string with "overflow_"
            return SERVER_CONFIG.max_clients + len(self.clients) - SERVER_CONFIG.max_clients
    
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
                    ready = select.select([client_socket], [], [], 0.1)
                    if not ready[0]:
                        continue

                    # Receive data length
                    length_data = client_socket.recv(2)
                    if not length_data or len(length_data) < 2:
                        raise ConnectionError("Client disconnected")

                    data_length = struct.unpack(">H", length_data)[0]
                    data = b""
                    remaining = data_length

                    while remaining > 0:
                        chunk = client_socket.recv(min(buffer_size, remaining))
                        if not chunk:
                            raise ConnectionError("Connection broken during data receive")
                        data += chunk
                        remaining -= len(chunk)

                    if len(data) < data_length:
                        print(f"Client {client_id} sent incomplete data")
                        continue

                    # Parse the frame data
                    frame = parse_frame_data(data)
                    if not frame:
                        # Send empty response on parsing failure
                        client_socket.sendall(struct.pack("bbb", 0, 0, 0))
                        continue
                    
                    # Get client state
                    with self.client_lock:
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
                            self.agent.save(LATEST_MODEL_PATH)
                        except Exception as e:
                            print(f"ERROR saving model: {e}")
                    
                    # Update global metrics
                    current_frame = self.metrics.update_frame_count()
                    self.metrics.update_epsilon()
                    self.metrics.update_expert_ratio()
                    self.metrics.update_game_state(frame.enemy_seg, frame.open_level)
                    
                    # Process previous step's results if available
                    if state['last_state'] is not None and state['last_action_idx'] is not None:
                        # Add experience to replay memory
                        self.agent.step(
                            state['last_state'],
                            np.array([[state['last_action_idx']]]),
                            frame.reward,
                            frame.state,
                            frame.done
                        )
                        
                        # Track rewards
                        state['total_reward'] += frame.reward
                        
                        # Track which system's rewards
                        if hasattr(metrics, 'last_action_source'):
                            if metrics.last_action_source == "expert":
                                state['episode_expert_reward'] += frame.reward
                            else:
                                state['episode_dqn_reward'] += frame.reward
                    
                    # Handle episode completion
                    if frame.done:
                        if not state['was_done']:
                            self.metrics.add_episode_reward(
                                state['total_reward'],
                                state['episode_dqn_reward'],
                                state['episode_expert_reward']
                            )
                        
                        state['was_done'] = True
                        client_socket.sendall(struct.pack("bbb", 0, 0, 0))
                        state['last_state'] = None
                        state['last_action_idx'] = None
                        continue
                    elif state['was_done']:
                        state['was_done'] = False
                        state['total_reward'] = 0
                        state['episode_dqn_reward'] = 0
                        state['episode_expert_reward'] = 0
                    
                    # Generate action
                    self.metrics.increment_total_controls()
                    
                    # Decide between expert system and DQN
                    if random.random() < self.metrics.get_expert_ratio() and not self.metrics.is_override_active():
                        # Use expert system
                        fire, zap, spinner = get_expert_action(
                            frame.enemy_seg, frame.player_seg, frame.open_level
                        )
                        self.metrics.increment_guided_count()
                        self.metrics.update_action_source("expert")
                        action_idx = expert_action_to_index(fire, zap, spinner)
                    else:
                        # Use DQN with current epsilon
                        action_idx = self.agent.act(frame.state, self.metrics.get_epsilon())
                        fire, zap, spinner = ACTION_MAPPING[action_idx]
                        self.metrics.update_action_source("dqn")
                    
                    # Store state and action for next iteration
                    state['last_state'] = frame.state
                    state['last_action_idx'] = action_idx
                    
                    # Send action to game
                    game_fire, game_zap, game_spinner = encode_action_to_game(fire, zap, spinner)
                    client_socket.sendall(struct.pack("bbb", game_fire, game_zap, game_spinner))
                    
                    # Periodic target network update (only from client 0)
                    if client_id == 0 and current_frame % RL_CONFIG.update_target_every == 0:
                        self.agent.update_target_network()
                    
                    # Periodic model saving (only from client 0)
                    if client_id == 0 and current_frame % RL_CONFIG.save_interval == 0:
                        self.agent.save(LATEST_MODEL_PATH)
                
                except (BlockingIOError, InterruptedError):
                    # Expected with non-blocking socket
                    time.sleep(0.001)
                    continue
                except (ConnectionResetError, BrokenPipeError, ConnectionError) as e:
                    print(f"Client {client_id} connection error: {e}")
                    break
                except Exception as e:
                    print(f"Error handling client {client_id}: {e}")
                    traceback.print_exc()
                    break

        except Exception as e:
            print(f"Fatal error handling client {client_id}: {e}")
            traceback.print_exc()
        finally:
            # Ensure proper cleanup in all cases
            try:
                client_socket.shutdown(socket.SHUT_RDWR)
            except:
                pass
            try:
                client_socket.close()
            except:
                pass

            # Clean up client state atomically
            with self.client_lock:
                if client_id in self.client_states:
                    del self.client_states[client_id]
                if client_id in self.clients:
                    del self.clients[client_id]
                # Update metrics after cleanup
                metrics.client_count = len(self.clients)
                print(f"Client {client_id} cleanup complete, {len(self.clients)} clients remaining")

    def cleanup_disconnected_clients(self):
        """Clean up any disconnected clients to free up their IDs"""
        with self.client_lock:
            # Find clients with dead threads or None entries
            disconnected_clients = [
                client_id for client_id, thread in self.clients.items()
                if thread is None or (isinstance(thread, threading.Thread) and not thread.is_alive())
            ]
            
            # Remove disconnected clients and their states
            for client_id in disconnected_clients:
                if client_id in self.clients:
                    del self.clients[client_id]
                if client_id in self.client_states:
                    del self.client_states[client_id]
            
            # Update metrics
            metrics.client_count = len(self.clients)
            if disconnected_clients:
                print(f"Cleaned up {len(disconnected_clients)} disconnected clients. Available IDs: {SERVER_CONFIG.max_clients - len(self.clients)}")

    def is_override_active(self):
        with self.client_lock:
            return self.metrics.override_expert
            
    def get_fps(self):
        with self.client_lock:
            return self.metrics.fps 