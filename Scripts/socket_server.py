#!/usr/bin/env python3
"""
Socket server for Tempest AI (hybrid-only).
Bridges Lua frames to a HybridDQNAgent (4 discrete fire/zap + 1 continuous spinner).
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
from collections import deque

import numpy as np

from aimodel import (
    parse_frame_data,
    get_expert_action,
    hybrid_to_game_action,
    fire_zap_to_discrete,
    SafeMetrics,
)
from config import RL_CONFIG, SERVER_CONFIG, metrics, LATEST_MODEL_PATH


class SocketServer:
    def __init__(self, host, port, agent, metrics_wrapper):
        self.host = host
        self.port = port
        self.agent = agent
        # Always wrap with SafeMetrics for thread-safe updates
        self.metrics = SafeMetrics(metrics_wrapper)

        self.server_socket = None
        self.running = False
        self.shutdown_event = threading.Event()

        # client_id -> thread
        self.clients = {}
        # client_id -> state dict
        self.client_states = {}
        self.client_lock = threading.Lock()

        # Action distribution stats
        self.expert_action_counts = np.zeros(16, dtype=np.int64)  # 4 discrete * 4 spinner buckets (diagnostic only)
        self.dqn_action_counts = np.zeros(16, dtype=np.int64)

    def start(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(SERVER_CONFIG.max_clients)
        self.server_socket.setblocking(False)

        self.running = True
        print(f"SocketServer listening on {self.host}:{self.port}")

        try:
            while self.running and not self.shutdown_event.is_set():
                readable, _, _ = select.select([self.server_socket], [], [], 0.05)
                if self.server_socket in readable:
                    client_socket, addr = self.server_socket.accept()
                    client_id = self._allocate_client_id()
                    self._init_client_state(client_id)
                    t = threading.Thread(target=self.handle_client, args=(client_socket, client_id), daemon=True)
                    with self.client_lock:
                        self.clients[client_id] = t
                    t.start()
        except Exception as e:
            print(f"Server loop error: {e}")
            traceback.print_exc()
        finally:
            self.stop()

    def stop(self):
        self.running = False
        self.shutdown_event.set()
        try:
            if self.server_socket:
                self.server_socket.close()
        except Exception:
            pass

    def _allocate_client_id(self):
        with self.client_lock:
            existing = set(self.clients.keys())
            cid = 0
            while cid in existing:
                cid += 1
            return cid

    def _init_client_state(self, client_id):
        with self.client_lock:
            self.client_states[client_id] = {
                'frames_processed': 0,
                'frames_processed_since_stats': 0,
                'last_frame_time': time.time(),
                'fps': 0.0,
                'level_number': 0,
                'prev_frame': None,
                'current_frame': None,
                'last_state': None,
                'last_action_hybrid': None,  # (discrete, continuous)
                'last_action_source': None,
                'total_reward': 0.0,
                'episode_dqn_reward': 0.0,
                'episode_expert_reward': 0.0,
                'was_done': False,
                'nstep_buf': deque(maxlen=max(1, int(getattr(RL_CONFIG, 'n_step', 1))))
            }
            metrics.client_count = len(self.client_states)

    def handle_client(self, client_socket, client_id):
        try:
            client_socket.setblocking(False)
            client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
            client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)

            buffer_size = 32768

            # handshake: expect 2 bytes quickly
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

                # read length
                length_data = client_socket.recv(2)
                if not length_data or len(length_data) < 2:
                    raise ConnectionError("Failed to read length")
                data_length = struct.unpack('>H', length_data)[0]

                # read payload
                data = b''
                remaining = data_length
                while remaining > 0:
                    chunk = client_socket.recv(min(buffer_size, remaining))
                    if not chunk:
                        raise ConnectionError("Connection broken during receive")
                    data += chunk
                    remaining -= len(chunk)

                # quick param-count sanity
                if len(data) >= 2:
                    num_values_received = struct.unpack('>H', data[:2])[0]
                    if num_values_received != SERVER_CONFIG.params_count:
                        print(f"Client {client_id}: Param mismatch {num_values_received} != {SERVER_CONFIG.params_count}")
                        break
                else:
                    print(f"Client {client_id}: Data too short for param count")
                    break

                frame = parse_frame_data(data)
                if not frame:
                    client_socket.sendall(struct.pack('bbb', 0, 0, 0))
                    continue

                # update per-client state
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

                # global metrics batching
                local_frame_accum += 1
                if local_frame_accum >= METRICS_BATCH:
                    current_frame = self.metrics.update_frame_count(delta=local_frame_accum)
                    local_frame_accum = 0
                    self.metrics.update_epsilon()
                    self.metrics.update_expert_ratio()
                    # Update average level across clients periodically
                    try:
                        self.calculate_average_level()
                    except Exception:
                        pass
                self.metrics.update_game_state(frame.enemy_seg, frame.open_level)

                # previous step (n-step)
                if state.get('last_state') is not None and state.get('last_action_hybrid') is not None:
                    n = int(getattr(RL_CONFIG, 'n_step', 1))
                    if n <= 1:
                        if self.agent:
                            da, ca = state['last_action_hybrid']
                            try:
                                self.agent.step(state['last_state'], int(da), float(ca), frame.reward, frame.state, frame.done)
                            except TypeError:
                                # If agent does not support hybrid step, skip (pure hybrid expected)
                                pass
                    else:
                        state['nstep_buf'].append((state['last_state'], tuple(state['last_action_hybrid']), frame.reward))
                        if len(state['nstep_buf']) >= n and self.agent:
                            gamma = RL_CONFIG.gamma
                            R = 0.0
                            for i in range(n):
                                R += (gamma ** i) * state['nstep_buf'][i][2]
                            s0, (da0, ca0), _ = state['nstep_buf'][0]
                            try:
                                self.agent.step(s0, int(da0), float(ca0), R, frame.state, bool(frame.done))
                            except TypeError:
                                pass
                            try:
                                state['nstep_buf'].popleft()
                            except Exception:
                                pass

                    # reward accounting
                    state['total_reward'] = state.get('total_reward', 0.0) + frame.reward
                    self._update_reward_components(float(frame.reward))
                    src = state.get('last_action_source')
                    if src == 'dqn':
                        state['episode_dqn_reward'] = state.get('episode_dqn_reward', 0.0) + frame.reward
                    elif src == 'expert':
                        state['episode_expert_reward'] = state.get('episode_expert_reward', 0.0) + frame.reward

                # terminal handling
                if frame.done:
                    if not state.get('was_done', False):
                        self.metrics.add_episode_reward(
                            state.get('total_reward', 0.0),
                            state.get('episode_dqn_reward', 0.0),
                            state.get('episode_expert_reward', 0.0)
                        )
                    state['was_done'] = True

                    # flush remaining n-step
                    n = int(getattr(RL_CONFIG, 'n_step', 1))
                    if n > 1 and self.agent and state.get('nstep_buf') is not None:
                        gamma = RL_CONFIG.gamma
                        buf = state['nstep_buf']
                        while len(buf) > 0:
                            L = len(buf)
                            R = 0.0
                            for i in range(L):
                                R += (gamma ** i) * buf[i][2]
                            s0, (da0, ca0), _ = buf[0]
                            try:
                                self.agent.step(s0, int(da0), float(ca0), R, frame.state, True)
                            except TypeError:
                                pass
                            try:
                                buf.popleft()
                            except Exception:
                                break

                    # confirm done
                    try:
                        client_socket.sendall(struct.pack('bbb', 0, 0, 0))
                    except Exception:
                        break

                    # reset for next episode
                    state['last_state'] = None
                    state['last_action_hybrid'] = None
                    try:
                        state['nstep_buf'].clear()
                    except Exception:
                        pass
                    state['total_reward'] = 0.0
                    state['episode_dqn_reward'] = 0.0
                    state['episode_expert_reward'] = 0.0
                    continue

                elif state.get('was_done', False):
                    state['was_done'] = False
                    state['total_reward'] = 0.0
                    state['episode_dqn_reward'] = 0.0
                    state['episode_expert_reward'] = 0.0

                # choose action (hybrid-only)
                self.metrics.increment_total_controls()

                discrete_action, continuous_spinner = 0, 0.0
                action_source = 'unknown'

                if self.agent:
                    # expert vs dqn mixture with override forcing pure dqn
                    expert_ratio = self.metrics.get_expert_ratio()
                    if frame.gamestate == 0x20:  # GS_ZoomingDown
                        expert_ratio *= 2
                    use_expert = (random.random() < expert_ratio) and (not self.metrics.is_override_active())

                    if use_expert:
                        fire, zap, spin = get_expert_action(
                            frame.enemy_seg,
                            frame.player_seg,
                            frame.open_level,
                            frame.expert_fire,
                            frame.expert_zap,
                        )
                        discrete_action = fire_zap_to_discrete(fire, zap)
                        continuous_spinner = float(spin)
                        action_source = 'expert'
                    else:
                        epsilon = 0.0 if self.metrics.is_override_active() else self.metrics.get_epsilon()
                        start_t = time.perf_counter()
                        da, ca = self.agent.act(frame.state, epsilon)
                        infer_t = time.perf_counter() - start_t
                        # Record inference timing via SafeMetrics API
                        if hasattr(self.metrics, 'add_inference_time'):
                            self.metrics.add_inference_time(infer_t)
                        else:
                            # Fallback: best-effort direct update with lock if exposed
                            try:
                                with self.metrics.lock:
                                    self.metrics.total_inference_time += infer_t
                                    self.metrics.total_inference_requests += 1
                            except Exception:
                                pass
                        discrete_action, continuous_spinner = int(da), float(ca)
                        action_source = 'dqn'
                else:
                    action_source = 'none'

                # store for next step
                state['last_state'] = frame.state
                state['last_action_hybrid'] = (discrete_action, continuous_spinner)
                state['last_action_source'] = action_source

                # send to game
                game_fire, game_zap, game_spinner = hybrid_to_game_action(discrete_action, continuous_spinner)
                try:
                    client_socket.sendall(struct.pack('bbb', game_fire, game_zap, game_spinner))
                except Exception:
                    break

                # maintenance
                if (client_id == 0 and 'current_frame' in locals() and self.agent
                        and current_frame % RL_CONFIG.update_target_every == 0):
                    if not getattr(RL_CONFIG, 'use_soft_target', True):
                        if hasattr(self.agent, 'force_hard_target_update'):
                            self.agent.force_hard_target_update()
                        elif hasattr(self.agent, 'update_target_network'):
                            self.agent.update_target_network()

                if client_id == 0 and 'current_frame' in locals() and self.agent and current_frame % RL_CONFIG.save_interval == 0:
                    try:
                        self.agent.save(LATEST_MODEL_PATH)
                    except Exception:
                        pass

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
            # Include level 0 (first level) as a valid value; exclude only negatives/uninitialized
            valid = [s.get('level_number', 0) for s in self.client_states.values() if s.get('level_number', 0) >= 0]
            if valid:
                avg = sum(valid) / len(valid)
                # Update the global metrics object so display picks it up
                metrics.average_level = avg
                try:
                    metrics.level_sum_interval += float(avg)
                    metrics.level_count_interval += 1
                except Exception:
                    pass
                return avg
            else:
                metrics.average_level = 0
                return 0

    def is_override_active(self):
        with self.client_lock:
            return self.metrics.override_expert

    def get_fps(self):
        with self.client_lock:
            return self.metrics.fps

    # Internal helpers
    def _update_reward_components(self, total_reward: float):
        """Safely update reward components in metrics if available."""
        try:
            # Preferred: SafeMetrics exposes update_reward_components
            if hasattr(self.metrics, 'update_reward_components'):
                self.metrics.update_reward_components({'total': float(total_reward)})
                return
            # Fallback: underlying metrics object might have it
            if hasattr(self.metrics, 'metrics') and hasattr(self.metrics.metrics, 'update_reward_components'):
                with self.metrics.lock:
                    self.metrics.metrics.update_reward_components({'total': float(total_reward)})
        except Exception:
            pass