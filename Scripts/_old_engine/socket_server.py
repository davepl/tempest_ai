#!/usr/bin/env python3
# ==================================================================================================================
# ||                                                                                                              ||
# ||                                    TEMPEST AI • SOCKET BRIDGE SERVER                                        ||
# ||                                                                                                              ||
# ||  FILE: Scripts/socket_server.py                                                                              ||
# ||  ROLE: TCP server bridging Lua (MAME) and Python: receives frames, returns actions, manages clients.          ||
# ||                                                                                                              ||
# ||  NEED TO KNOW:                                                                                               ||
# ||   - Accepts Lua client(s), handshakes, reads OOB+state, decodes, queries agent, replies action bytes.         ||
# ||   - Supports optional server-side n-step (if agent doesn’t own one); updates metrics thread-safely.          ||
# ||   - Robust shutdown handling; per-client worker threads.                                                      ||
# ||                                                                                                              ||
# ||  CONSUMES: RL_CONFIG, SERVER_CONFIG, metrics, NStepReplayBuffer (optional)                                   ||
# ||  PRODUCES: agent experiences, actions to Lua, metrics updates                                                ||
# ||                                                                                                              ||
# ==================================================================================================================
"""
Socket server for Tempest AI (joint-action DQN).
Bridges Lua frames to a DiscreteDQNAgent while preserving the 3-byte action socket protocol.
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
import traceback
import random
import errno
import queue
import math
from collections import deque

import numpy as np

from aimodel import (
    parse_frame_data,
    get_expert_action,
    encode_action_to_game,
    fire_zap_to_discrete,
    discrete_to_fire_zap,
    quantize_spinner_value,
    spinner_index_to_value,
    combine_action_indices,
    split_joint_action,
    SafeMetrics,
)
from config import RL_CONFIG, SERVER_CONFIG, metrics, LATEST_MODEL_PATH

try:
    from nstep_buffer import NStepReplayBuffer
except ImportError:
    from Scripts.nstep_buffer import NStepReplayBuffer


class AsyncReplayBuffer:
    """
    Non-blocking async wrapper for agent.step() calls.
    Queues experiences and inserts them in batches on a background thread.
    """
    def __init__(self, agent, batch_size=1000, max_queue_size=10000):
        self.agent = agent
        self.batch_size = batch_size
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.running = True
        self.put_timeout = 0.05
        self.worker_thread = threading.Thread(target=self._consume_queue, daemon=True)
        self.worker_thread.start()
        self.items_queued = 0
        self.items_processed = 0
        self.items_dropped = 0
        
    def step_async(self, *args, **kwargs):
        """Non-blocking step - queues experience for later insertion."""
        try:
            self.queue.put((args, kwargs), timeout=self.put_timeout)
            self.items_queued += 1
            return True
        except queue.Full:
            self.items_dropped += 1
            return False
    
    def _consume_queue(self):
        batch = []
        while self.running:
            try:
                if not batch:
                    item = self.queue.get(timeout=0.01)
                    batch.append(item)
                while len(batch) < self.batch_size:
                    try:
                        batch.append(self.queue.get_nowait())
                    except queue.Empty:
                        break

                for args, kwargs in batch:
                    try:
                        self.agent.step(*args, **kwargs)
                        self.items_processed += 1
                    except Exception as e:
                        print(f"AsyncReplayBuffer: Error in agent.step(): {e}")
                batch.clear()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"AsyncReplayBuffer worker error: {e}")
                time.sleep(0.01)
                
    def stop(self):
        self.running = False
        remaining = []
        try:
            while True:
                remaining.append(self.queue.get_nowait())
        except queue.Empty:
            pass
        
        for args, kwargs in remaining:
            try:
                self.agent.step(*args, **kwargs)
                self.items_processed += 1
            except Exception:
                pass
        self.worker_thread.join(timeout=5.0)
        
    def get_stats(self):
        return {
            'queued': self.items_queued,
            'processed': self.items_processed,
            'dropped': self.items_dropped,
            'pending': self.queue.qsize(),
            'queue_full': self.queue.full()
        }


class SocketServer:
    def __init__(self, host, port, agent, metrics_wrapper):
        self.host = host
        self.port = port
        self.agent = agent
        self.async_buffer = AsyncReplayBuffer(agent, batch_size=100, max_queue_size=10000) if agent else None
        self.metrics = SafeMetrics(metrics_wrapper)

        self.server_socket = None
        self.running = False
        self.shutdown_event = threading.Event()

        self.clients = {}
        self.client_states = {}
        self.client_lock = threading.Lock()

    def _allocate_client_id(self):
        with self.client_lock:
            existing = set(self.clients.keys())
            cid = 0
            while cid in existing:
                cid += 1
            return cid

    def _init_client_state(self, client_id):
        n_step = int(max(1, int(getattr(RL_CONFIG, "n_step", 1) or 1)))
        gamma = float(getattr(RL_CONFIG, "gamma", 0.99) or 0.99)
        nstep_buffer = NStepReplayBuffer(n_step=n_step, gamma=gamma) if n_step > 1 else None

        with self.client_lock:
            self.client_states[client_id] = {
                'frames_processed': 0,
                'last_frame_time': time.time(),
                'fps': 0.0,
                'level_number': 0,
                'prev_frame': None,
                'current_frame': None,
                'last_state': None,
                'last_action': None, # (fz_idx, sp_idx)
                'last_action_source': None,
                'prev_action_source': None,
                'total_reward': 0.0,
                'episode_dqn_reward': 0.0,
                'episode_expert_reward': 0.0,
                'was_done': False,
                'nstep_buffer': nstep_buffer,
                'pending_action_penalty': 0.0,
            }
            metrics.client_count = len(self.client_states)

    def handle_client(self, client_socket, client_id):
        try:
            client_socket.setblocking(False)
            client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
            client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)

            buffer_size = 32768

            # handshake
            try:
                client_socket.setblocking(True)
                client_socket.settimeout(5.0)
                ping = client_socket.recv(2)
                if not ping or len(ping) < 2:
                    raise ConnectionError("No initial ping header")
            finally:
                client_socket.setblocking(False)
                client_socket.settimeout(None)

            METRICS_BATCH = 8
            local_frame_accum = 0

            while self.running and not self.shutdown_event.is_set():
                ready = select.select([client_socket], [], [], 0.0)
                if not ready[0]:
                    time.sleep(0.0005)
                    continue

                length_data = client_socket.recv(2)
                if not length_data or len(length_data) < 2:
                    raise ConnectionError("Failed to read length")
                data_length = struct.unpack('>H', length_data)[0]

                data = b''
                remaining = data_length
                while remaining > 0:
                    chunk = client_socket.recv(min(buffer_size, remaining))
                    if not chunk:
                        raise ConnectionError("Connection broken during receive")
                    data += chunk
                    remaining -= len(chunk)

                if len(data) >= 2:
                    num_values_received = struct.unpack('>H', data[:2])[0]
                    if num_values_received != SERVER_CONFIG.params_count:
                        print(f"Client {client_id}: Param mismatch {num_values_received} != {SERVER_CONFIG.params_count}")
                        break
                else:
                    break

                frame = parse_frame_data(data)
                if not frame:
                    client_socket.sendall(struct.pack('bbb', 0, 0, 0))
                    continue

                with self.client_lock:
                    if client_id not in self.client_states:
                        break
                    state = self.client_states[client_id]
                    state['frames_processed'] += 1
                    state['level_number'] = frame.level_number
                    state['prev_frame'] = state.get('current_frame')
                    state['current_frame'] = frame
                    now = time.time()
                    elapsed = now - state['last_frame_time']
                    if elapsed >= 1.0:
                        state['fps'] = 1.0 / elapsed
                        state['last_frame_time'] = now

                local_frame_accum += 1
                if local_frame_accum >= METRICS_BATCH:
                    self.metrics.update_frame_count(delta=local_frame_accum)
                    local_frame_accum = 0
                    self.metrics.update_epsilon()
                    self.metrics.update_expert_ratio()
                    try:
                        self.calculate_average_level()
                    except Exception:
                        pass
                self.metrics.update_game_state(frame.enemy_seg, frame.open_level)

                # Process previous frame reward
                if state.get('last_state') is not None and state.get('last_action') is not None:
                    last_action = state['last_action'] # (fz_idx, sp_idx)
                    
                    # Apply scaling to rewards received from the game
                    subj_scale = getattr(RL_CONFIG, 'subj_reward_scale', 1.0)
                    obj_scale = getattr(RL_CONFIG, 'obj_reward_scale', 1.0)
                    
                    subj_reward = float(frame.subjreward) * subj_scale
                    obj_reward = float(frame.objreward) * obj_scale
                    
                    training_reward = obj_reward + subj_reward
                    pending_action_penalty = float(state.get('pending_action_penalty', 0.0) or 0.0)
                    if pending_action_penalty != 0.0:
                        training_reward += pending_action_penalty
                        state['pending_action_penalty'] = 0.0

                    # Push to agent
                    if self.agent:
                        actor_tag = state.get('prev_action_source', 'dqn')
                        nstep = state.get('nstep_buffer')

                        if nstep is not None:
                            joint_action = combine_action_indices(last_action[0], last_action[1])
                            matured = nstep.add(
                                state['last_state'],
                                joint_action,
                                training_reward,
                                frame.state,
                                bool(frame.done),
                                actor=actor_tag,
                                priority_reward=training_reward,
                            )
                            for item in matured:
                                s0, a_joint, Rn, priority_R, sn, done_n, horizon_n, actor_n = item
                                fz_n, sp_n = split_joint_action(a_joint)
                                self.async_buffer.step_async(
                                    s0,
                                    (fz_n, sp_n),
                                    Rn,
                                    sn,
                                    bool(done_n),
                                    actor=actor_n,
                                    horizon=int(horizon_n),
                                    priority_reward=priority_R,
                                )
                        else:
                            self.async_buffer.step_async(
                                state['last_state'],
                                last_action,
                                training_reward,
                                frame.state,
                                bool(frame.done),
                                actor=actor_tag,
                                horizon=1,
                                priority_reward=training_reward,
                            )

                    # Reward accounting
                    state['total_reward'] = state.get('total_reward', 0.0) + training_reward
                    state['episode_subj_reward'] = state.get('episode_subj_reward', 0.0) + subj_reward
                    state['episode_obj_reward'] = state.get('episode_obj_reward', 0.0) + obj_reward
                    state['episode_frames'] = state.get('episode_frames', 0) + 1
                    
                    prev_src = state.get('prev_action_source')
                    if prev_src == 'dqn':
                        state['episode_dqn_reward'] = state.get('episode_dqn_reward', 0.0) + training_reward
                    elif prev_src == 'expert':
                        state['episode_expert_reward'] = state.get('episode_expert_reward', 0.0) + training_reward

                # Terminal handling
                if frame.done:
                    if not state.get('was_done', False):
                        self.metrics.add_episode_reward(
                            state.get('total_reward', 0.0),
                            state.get('episode_dqn_reward', 0.0),
                            state.get('episode_expert_reward', 0.0),
                            state.get('episode_subj_reward', 0.0),
                            state.get('episode_obj_reward', 0.0),
                            length=state.get('episode_frames', 0)
                        )
                    state['was_done'] = True
                    try:
                        client_socket.sendall(struct.pack('bbb', 0, 0, 0))
                    except Exception:
                        break

                    state['last_state'] = None
                    state['last_action'] = None
                    state['last_action_source'] = None
                    state['prev_action_source'] = None
                    state['total_reward'] = 0.0
                    state['episode_dqn_reward'] = 0.0
                    state['episode_expert_reward'] = 0.0
                    state['episode_subj_reward'] = 0.0
                    state['episode_obj_reward'] = 0.0
                    state['episode_frames'] = 0
                    state['pending_action_penalty'] = 0.0
                    continue

                elif state.get('was_done', False):
                    state['was_done'] = False
                    state['total_reward'] = 0.0
                    state['episode_dqn_reward'] = 0.0
                    state['episode_expert_reward'] = 0.0
                    state['episode_subj_reward'] = 0.0
                    state['episode_obj_reward'] = 0.0
                    state['episode_frames'] = 0
                    state['pending_action_penalty'] = 0.0

                # Choose Action
                self.metrics.increment_total_controls()
                
                action_source = None
                fz_idx = 0
                sp_idx = 0
                fire = False
                zap = False
                spinner_val = 0.0

                if self.agent:
                    expert_ratio = self.metrics.get_expert_ratio()
                    if frame.gamestate == 0x20: # Zooming
                        expert_ratio = min(1.0, expert_ratio * 4.0)
                    
                    use_expert = (random.random() < expert_ratio) and (not self.metrics.metrics.override_expert)

                    if use_expert:
                        fire, zap, spinner_val = get_expert_action(
                            frame.enemy_seg, frame.player_seg, frame.open_level,
                            frame.expert_fire, frame.expert_zap
                        )
                        fz_idx = fire_zap_to_discrete(fire, zap)
                        sp_idx = quantize_spinner_value(spinner_val)
                        action_source = 'expert'
                    else:
                        epsilon = self.metrics.get_effective_epsilon()
                        if frame.gamestate == 0x20:
                            epsilon *= float(getattr(RL_CONFIG, "zoom_epsilon_scale", 0.25) or 0.25)
                        
                        start_t = time.perf_counter()
                        fz_idx, sp_idx = self.agent.act(frame.state, epsilon)
                        infer_t = time.perf_counter() - start_t
                        self.metrics.add_inference_time(infer_t)
                        
                        fire, zap = discrete_to_fire_zap(fz_idx)
                        spinner_val = spinner_index_to_value(sp_idx)
                        action_source = 'dqn'
                else:
                    action_source = 'none'

                # Store for next step
                if zap and bool(getattr(RL_CONFIG, "enable_superzap_gate", False)):
                    zap_prob = float(getattr(RL_CONFIG, "superzap_prob", 1.0) or 1.0)
                    zap_prob = max(0.0, min(1.0, zap_prob))
                    if random.random() > zap_prob:
                        zap = False
                        fz_idx = fire_zap_to_discrete(fire, zap)
                        penalty = float(getattr(RL_CONFIG, "superzap_block_penalty", 0.0) or 0.0)
                        if penalty != 0.0:
                            state['pending_action_penalty'] = state.get('pending_action_penalty', 0.0) + penalty

                state['last_state'] = frame.state
                state['last_action'] = (fz_idx, sp_idx)
                state['prev_action_source'] = action_source
                state['last_action_source'] = action_source

                # Send to game
                game_fire, game_zap, game_spinner = encode_action_to_game(fire, zap, spinner_val)
                try:
                    client_socket.sendall(struct.pack('bbb', game_fire, game_zap, game_spinner))
                except Exception:
                    break

        except Exception as e:
            print(f"Error handling client {client_id}: {e}")
            traceback.print_exc()
        finally:
            try:
                client_socket.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                client_socket.close()
            except Exception:
                pass

            with self.client_lock:
                if client_id in self.client_states:
                    del self.client_states[client_id]
                if client_id in self.clients:
                    self.clients[client_id] = None
                metrics.client_count = len([c for c in self.clients.values() if c is not None])

            threading.Timer(1.0, self.cleanup_disconnected_clients).start()

    def cleanup_disconnected_clients(self):
        cleaned = 0
        with self.client_lock:
            to_delete = [cid for cid, t in self.clients.items() if t is None]
            for cid in to_delete:
                del self.clients[cid]
                cleaned += 1
            if cleaned:
                metrics.client_count = len(self.clients)

    def calculate_average_level(self):
        with self.client_lock:
            valid = [s.get('level_number', 0) for s in self.client_states.values() if s.get('level_number', 0) >= 0]
            if valid:
                avg = sum(valid) / len(valid)
                metrics.average_level = avg
                return avg
            else:
                metrics.average_level = 0
                return 0

    def start(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Retry binding if address is in use (e.g. previous instance shutting down)
        for i in range(10):
            try:
                if self.shutdown_event.is_set() or self.server_socket is None:
                    return
                self.server_socket.bind((self.host, self.port))
                break
            except OSError as e:
                if e.errno == 98:  # Address already in use
                    print(f"Port {self.port} in use, retrying in 1s... ({i+1}/10)")
                    time.sleep(1.0)
                else:
                    raise e
        else:
            raise OSError(f"Could not bind to {self.host}:{self.port} after 10 attempts")

        if self.server_socket is None:
            return

        self.server_socket.listen(SERVER_CONFIG.max_clients)
        self.server_socket.setblocking(False)

        self.running = True
        print(f"SocketServer listening on {self.host}:{self.port}")

        try:
            while self.running and not self.shutdown_event.is_set():
                try:
                    readable, _, _ = select.select([self.server_socket], [], [], 0.05)
                except (OSError, ValueError):
                    if self.shutdown_event.is_set() or not self.running:
                        break
                    raise

                if not self.server_socket:
                    break

                if self.server_socket in readable:
                    try:
                        client_socket, addr = self.server_socket.accept()
                    except OSError:
                        continue
                    client_id = self._allocate_client_id()
                    self._init_client_state(client_id)
                    t = threading.Thread(target=self.handle_client, args=(client_socket, client_id), daemon=True)
                    with self.client_lock:
                        self.clients[client_id] = t
                    t.start()

        except Exception as e:
            if not (self.shutdown_event.is_set() or not self.running):
                print(f"Server loop error: {e}")
                traceback.print_exc()
        finally:
            self.stop()

    def stop(self):
        # Idempotent stop: avoid duplicate shutdown/flush when called from multiple places.
        if self.shutdown_event.is_set() and not self.running:
            return

        self.running = False
        self.shutdown_event.set()
        if self.async_buffer:
            print("Flushing async replay buffer...")
            self.async_buffer.stop()
            self.async_buffer = None
        
        try:
            if self.server_socket:
                try:
                    self.server_socket.shutdown(socket.SHUT_RDWR)
                except Exception:
                    pass
                self.server_socket.close()
                self.server_socket = None
        except Exception:
            pass
