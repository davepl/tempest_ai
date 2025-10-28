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
Socket server for Tempest AI (hybrid-only).
Bridges Lua frames to a HybridDQNAgent with a joint fire/zap/spinner action space.
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
from collections import deque

import numpy as np

from aimodel import (
    parse_frame_data,
    get_expert_action,
    get_expert_hybrid_action,
    compose_action_index,
    action_index_to_components,
    hybrid_to_game_action,
    fire_zap_to_discrete,
    SafeMetrics,
)
from nstep_buffer import NStepReplayBuffer
from config import RL_CONFIG, SERVER_CONFIG, metrics, LATEST_MODEL_PATH


def _spinner_sign(value: float) -> int:
    """Classify spinner magnitude by sign for agreement comparison."""
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


class AsyncReplayBuffer:
    """
    Non-blocking async wrapper for agent.step() calls.
    Queues experiences and inserts them in batches on a background thread.
    This prevents client threads from blocking on buffer insertion.
    """
    def __init__(self, agent, batch_size=1000, max_queue_size=10000):
        self.agent = agent
        self.batch_size = batch_size
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.running = True
        self.worker_thread = threading.Thread(target=self._consume_queue, daemon=True)
        self.worker_thread.start()
        self.items_queued = 0
        self.items_processed = 0
        self.items_dropped = 0
        
    def step_async(self, *args, **kwargs):
        """Non-blocking step - queues experience for later insertion."""
        try:
            self.queue.put_nowait((args, kwargs))
            self.items_queued += 1
            return True
        except queue.Full:
            # Queue full - drop frame (better than blocking)
            self.items_dropped += 1
            return False
    
    def _consume_queue(self):
        """Background thread that processes queued experiences in batches."""
        batch = []
        while self.running:
            try:
                # Try to get item without blocking (aggressive draining)
                try:
                    # Drain multiple items quickly
                    while len(batch) < self.batch_size:
                        item = self.queue.get_nowait()
                        batch.append(item)
                except queue.Empty:
                    pass
                
                # If we have a batch or any items and queue is empty, process immediately
                if batch:
                    for args, kwargs in batch:
                        try:
                            self.agent.step(*args, **kwargs)
                            self.items_processed += 1
                        except Exception as e:
                            print(f"AsyncReplayBuffer: Error in agent.step(): {e}")
                    batch.clear()
                else:
                    # Only sleep briefly if queue is truly empty
                    time.sleep(0.001)  # 1ms sleep to avoid busy-wait
                    
            except Exception as e:
                print(f"AsyncReplayBuffer worker error: {e}")
                
    def stop(self):
        """Stop the background worker and flush remaining queue."""
        self.running = False
        # Process any remaining items
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
        """Get async buffer statistics."""
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
        # Create async buffer wrapper for non-blocking experience insertion
        self.async_buffer = AsyncReplayBuffer(agent, batch_size=100, max_queue_size=10000) if agent else None
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
        #BUGBUG Is this still accurate with the hybrid model?
        self.expert_action_counts = np.zeros(16, dtype=np.int64)  # 4 discrete * 4 spinner buckets (diagnostic only)
        self.dqn_action_counts = np.zeros(16, dtype=np.int64)

    def _verbose_enabled(self) -> bool:
        """Return True when verbose debug logging is enabled."""
        try:
            if hasattr(self.metrics, 'metrics') and self.metrics.metrics is not None:
                return bool(getattr(self.metrics.metrics, 'verbose_mode', False))
            return bool(getattr(self.metrics, 'verbose_mode', False))
        except Exception:
            return False

    def _server_nstep_enabled(self) -> bool:
        """Decide whether this server should perform n-step preprocessing.

        Rules:
        - If RL_CONFIG.n_step <= 1: no server-side n-step.
        - If the agent already exposes its own n_step_buffer (non-None), let the agent handle n-step.
        - Otherwise, perform server-side n-step (typical for HybridDQNAgent).
        """
        try:
            n = int(getattr(RL_CONFIG, 'n_step', 1) or 1)
        except Exception:
            n = 1
        if n <= 1:
            return False
        # If the agent has its own n-step buffer, skip server-side n-step to avoid double application
        if hasattr(self.agent, 'n_step_buffer') and getattr(self.agent, 'n_step_buffer') is not None:
            return False
        return True

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
                # select may raise or the socket may be closed concurrently during shutdown; handle gracefully
                try:
                    readable, _, _ = select.select([self.server_socket], [], [], 0.05)
                except (OSError, ValueError) as e:
                    # If we're shutting down, break quietly
                    if self.shutdown_event.is_set() or not self.running:
                        break
                    # If the server socket was closed, EBADF/EINVAL are expected; exit loop silently
                    if isinstance(e, OSError) and getattr(e, 'errno', None) in (errno.EBADF, errno.EINVAL):
                        break
                    # Otherwise, re-raise to outer handler
                    raise

                if not self.server_socket:
                    break

                if self.server_socket in readable:
                    try:
                        client_socket, addr = self.server_socket.accept()
                    except OSError as e:
                        # During shutdown accept may see EBADF/EINVAL; exit loop quietly
                        if (self.shutdown_event.is_set() or not self.running) and getattr(e, 'errno', None) in (errno.EBADF, errno.EINVAL):
                            break
                        # EAGAIN/EWOULDBLOCK can happen with non-blocking sockets; continue loop
                        if getattr(e, 'errno', None) in (errno.EAGAIN, errno.EWOULDBLOCK):
                            continue
                        raise
                    client_id = self._allocate_client_id()
                    self._init_client_state(client_id)
                    t = threading.Thread(target=self.handle_client, args=(client_socket, client_id), daemon=True)
                    with self.client_lock:
                        self.clients[client_id] = t
                    t.start()

        except Exception as e:
            # Suppress noisy tracebacks if we're shutting down intentionally
            if not (self.shutdown_event.is_set() or not self.running):
                print(f"Server loop error: {e}")
                traceback.print_exc()

        finally:
            self.stop()

    def stop(self):
        self.running = False
        self.shutdown_event.set()
        
        # Stop async buffer and flush remaining experiences
        if self.async_buffer:
            print("Flushing async replay buffer...")
            self.async_buffer.stop()
            stats = self.async_buffer.get_stats()
            print(f"Async buffer stats: {stats['processed']:,} processed, {stats['pending']} remaining, {stats['dropped']} dropped")
        
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
                'last_action_index': None,
                'last_action_source': None,
                'total_reward': 0.0,
                'episode_dqn_reward': 0.0,
                'episode_expert_reward': 0.0,
                'was_done': False,
                # Only create an n-step buffer if the server is responsible for n-step preprocessing
                'nstep_buffer': (
                    NStepReplayBuffer(RL_CONFIG.n_step, RL_CONFIG.gamma)
                    if self._server_nstep_enabled() else None
                )
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
                    state['episode_frame_count'] = state.get('episode_frame_count', 0) + 1  # Track episode length
                    state['level_number'] = frame.level_number
                    state['prev_frame'] = state.get('current_frame')
                    state['current_frame'] = frame
                    # No per-level frame tracking needed for probabilistic zap gate
                    now = time.time()
                    elapsed = now - state['last_frame_time']
                    if elapsed >= 1.0:
                        state['fps'] = 1.0 / elapsed
                        state['last_frame_time'] = now

                def require_actor_tag(context: str) -> str:
                    actor_value = state.get('last_action_source')
                    if actor_value is None:
                        raise RuntimeError(f"{context}: missing actor tag for experience emission")
                    actor_norm = str(actor_value).strip().lower()
                    if not actor_norm:
                        raise RuntimeError(f"{context}: blank actor tag for experience emission")
                    if actor_norm in ('unknown', 'none', 'random'):
                        raise RuntimeError(f"{context}: invalid actor tag '{actor_norm}'")
                    return actor_norm

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

                # N-step experience processing (server-side only when enabled) or direct 1-step fallback
                if state.get('last_state') is not None and state.get('last_action_index') is not None:
                    action_index = state['last_action_index']
                    
                    # Apply superzap penalty to subjective reward if configured
                    subj_reward = float(frame.subjreward)
                    superzap_penalty = float(getattr(RL_CONFIG, 'superzap_penalty', 0.0) or 0.0)
                    if superzap_penalty > 0.0:
                        # Check if last action was superzap (fire + zap)
                        last_fire, last_zap, _, _ = action_index_to_components(int(action_index))
                        if last_fire and last_zap:
                            subj_reward -= superzap_penalty
                    
                    if self._server_nstep_enabled() and state.get('nstep_buffer') is not None:
                        # Add experience to n-step buffer and get matured experiences
                        # Compute rewards separately for training and for bucket priority
                        obj_reward = float(frame.objreward)
                        priority_reward_step = obj_reward + subj_reward
                        terminal_bonus = float(getattr(RL_CONFIG, 'priority_terminal_bonus', 0.0) or 0.0)
                        if frame.done and terminal_bonus != 0.0:
                            priority_reward_step += terminal_bonus
                        if (ignore_subjective_rewards := getattr(RL_CONFIG, 'ignore_subjective_rewards', True)):
                            total_reward = obj_reward
                        else:
                            total_reward = priority_reward_step

                        experiences = state['nstep_buffer'].add(
                            state['last_state'],
                            int(action_index),
                            total_reward,
                            frame.state,
                            frame.done,
                            actor=require_actor_tag("n-step buffer push"),
                            priority_reward=priority_reward_step,
                        )

                        # Push all matured experiences to agent
                        if self.agent and experiences:
                            for item in experiences:
                                if len(item) != 8:
                                    continue
                                (
                                    exp_state,
                                    exp_action,
                                    exp_reward,
                                    exp_priority_reward,
                                    exp_next_state,
                                    exp_done,
                                    exp_steps,
                                    exp_actor,
                                ) = item

                                self.async_buffer.step_async(
                                    exp_state,
                                    exp_action,
                                    exp_reward,
                                    exp_next_state,
                                    exp_done,
                                    actor=exp_actor,
                                    horizon=exp_steps,
                                    priority_reward=exp_priority_reward,
                                )
                    else:
                        # Server is not handling n-step: push single-step transition directly to the agent
                        obj_reward = float(frame.objreward)
                        priority_reward_step = obj_reward + subj_reward
                        terminal_bonus = float(getattr(RL_CONFIG, 'priority_terminal_bonus', 0.0) or 0.0)
                        if frame.done and terminal_bonus != 0.0:
                            priority_reward_step += terminal_bonus
                        if (ignore_subjective_rewards := getattr(RL_CONFIG, 'ignore_subjective_rewards', True)):
                            direct_reward = obj_reward
                        else:
                            direct_reward = priority_reward_step

                        if self.agent:
                            self.async_buffer.step_async(
                                state['last_state'],
                                int(action_index),
                                direct_reward,
                                frame.state,
                                bool(frame.done),
                                actor=require_actor_tag("direct push"),
                                horizon=1,
                                priority_reward=priority_reward_step,
                            )

                    # reward accounting
                    # Update reward accounting using subj+obj derived total, respecting ignore_subjective_rewards
                    
                    if (ignore_subjective_rewards := getattr(RL_CONFIG, 'ignore_subjective_rewards', True)):
                        total_reward = float(frame.objreward)
                    else:
                        total_reward = subj_reward + float(frame.objreward)

                    state['total_reward'] = state.get('total_reward', 0.0) + total_reward
                    state['episode_subj_reward'] = state.get('episode_subj_reward', 0.0) + subj_reward
                    state['episode_obj_reward'] = state.get('episode_obj_reward', 0.0) + frame.objreward
                    src = state.get('last_action_source')
                    # Persist last-step attribution details for debugging/terminal reporting
                    try:
                        state['last_step_total_reward'] = float(total_reward)
                        state['last_step_obj_reward'] = float(frame.objreward)
                        state['last_step_subj_reward'] = float(subj_reward)
                        state['last_step_actor'] = str(src) if src is not None else 'unknown'
                    except Exception:
                        pass
                    if src == 'dqn':
                        state['episode_dqn_reward'] = state.get('episode_dqn_reward', 0.0) + total_reward
                    elif src == 'expert':
                        state['episode_expert_reward'] = state.get('episode_expert_reward', 0.0) + total_reward
                    else:
                        # In verbose mode, surface unexpected/missing actor tags
                        try:
                            verbose = False
                            if hasattr(self.metrics, 'get'):
                                # SafeMetrics may not expose direct fields; attempt safe get
                                verbose = bool(self.metrics.get('verbose_mode', False))
                            elif hasattr(self.metrics, 'metrics') and hasattr(self.metrics, 'lock'):
                                with self.metrics.lock:
                                    verbose = bool(getattr(self.metrics.metrics, 'verbose_mode', False))
                            if verbose:
                                print(f"[ATTR] Client {client_id}: unexpected actor tag '{src}' on reward attribution; reward={total_reward:.4f}")
                        except Exception:
                            pass

                # terminal handling
                if frame.done:
                    if not state.get('was_done', False):
                        # Calculate episode length (frames in this episode)
                        episode_length = state.get('episode_frame_count', 0)
                        # Optional verbose attribution snapshot at episode end to validate who received terminal penalty
                        try:
                            verbose = False
                            if hasattr(self.metrics, 'get'):
                                verbose = bool(self.metrics.get('verbose_mode', False))
                            elif hasattr(self.metrics, 'metrics') and hasattr(self.metrics, 'lock'):
                                with self.metrics.lock:
                                    verbose = bool(getattr(self.metrics.metrics, 'verbose_mode', False))
                            if verbose:
                                last_actor = state.get('last_step_actor', state.get('last_action_source'))
                                last_tot = state.get('last_step_total_reward', 0.0)
                                last_obj = state.get('last_step_obj_reward', 0.0)
                                last_subj = state.get('last_step_subj_reward', 0.0)
                                ep_dqn = state.get('episode_dqn_reward', 0.0)
                                ep_exp = state.get('episode_expert_reward', 0.0)
                                ep_total = state.get('total_reward', 0.0)
                                print(
                                    f"[ATTR] Episode end (client {client_id}, frames={episode_length}): "
                                    f"last_step_actor={last_actor}, last_step_obj={last_obj:.4f}, last_step_subj={last_subj:.4f}, "
                                    f"attrib_last_step_total={last_tot:.4f} | ep_dqn={ep_dqn:.4f}, ep_expert={ep_exp:.4f}, ep_total={ep_total:.4f}"
                                )
                        except Exception:
                            pass
                        self.metrics.add_episode_reward(
                            state.get('total_reward', 0.0),
                            state.get('episode_dqn_reward', 0.0),
                            state.get('episode_expert_reward', 0.0),
                            state.get('episode_subj_reward', 0.0),
                            state.get('episode_obj_reward', 0.0),
                            episode_length=episode_length
                        )
                    state['was_done'] = True

                    # Reset n-step buffer only if server-side n-step is enabled
                    try:
                        if self._server_nstep_enabled() and state.get('nstep_buffer') is not None:
                            state['nstep_buffer'].reset()
                    except Exception:
                        pass

                    # confirm done
                    try:
                        client_socket.sendall(struct.pack('bbb', 0, 0, 0))
                    except Exception:
                        break

                    # reset for next episode
                    state['last_state'] = None
                    state['last_action_index'] = None
                    state['total_reward'] = 0.0
                    state['episode_dqn_reward'] = 0.0
                    state['episode_expert_reward'] = 0.0
                    state['episode_subj_reward'] = 0.0
                    state['episode_obj_reward'] = 0.0
                    state['episode_frame_count'] = 0  # Reset episode length counter
                    continue

                elif state.get('was_done', False):
                    state['was_done'] = False
                    state['total_reward'] = 0.0
                    state['episode_dqn_reward'] = 0.0
                    state['episode_expert_reward'] = 0.0
                    state['episode_subj_reward'] = 0.0
                    state['episode_obj_reward'] = 0.0
                    state['episode_frame_count'] = 0  # Reset episode length counter
                    # No per-level frame tracking

                # choose action (hybrid-only)
                self.metrics.increment_total_controls()

                action_index = 0
                spinner_value = 0.0
                action_source = None

                if self.agent:
                    # expert vs dqn mixture with override forcing pure dqn
                    expert_ratio = self.metrics.get_expert_ratio()
                    if frame.gamestate == 0x20:  # GS_ZoomingDown
                        expert_ratio *= 2
                    use_expert = (random.random() < expert_ratio) and (not self.metrics.is_override_active())

                    if use_expert:
                        action_index, spinner_value = get_expert_hybrid_action(
                            frame.enemy_seg,
                            frame.player_seg,
                            frame.open_level,
                            frame.expert_fire,
                            frame.expert_zap,
                        )
                        action_source = 'expert'
                    else:
                        # Epsilon policy: when override_epsilon is ON, force 0.0 (pure greedy).
                        # Otherwise, always use the current decayed epsilon even during expert/inference overrides.
                        try:
                            if hasattr(self.metrics, 'get_effective_epsilon'):
                                epsilon = float(self.metrics.get_effective_epsilon())
                            elif hasattr(self.metrics, 'metrics') and hasattr(self.metrics.metrics, 'get_effective_epsilon'):
                                with self.metrics.lock:
                                    epsilon = float(self.metrics.metrics.get_effective_epsilon())
                            else:
                                epsilon = float(self.metrics.get_epsilon())
                        except Exception:
                            epsilon = float(self.metrics.get_epsilon())

                        # Reduce random exploration while zooming: use a fraction of the current epsilon
                        try:
                            if frame.gamestate == 0x20:  # GS_ZoomingDown
                                scale = float(getattr(RL_CONFIG, 'zoom_epsilon_scale', 0.25) or 0.25)
                                epsilon = epsilon * scale
                                if epsilon < 0.0:
                                    epsilon = 0.0
                                elif epsilon > 1.0:
                                    epsilon = 1.0
                        except Exception:
                            pass

                        start_t = time.perf_counter()
                        action_index = int(self.agent.act(frame.state, epsilon, True))
                        infer_t = time.perf_counter() - start_t

                        if hasattr(self.metrics, 'add_inference_time'):
                            self.metrics.add_inference_time(infer_t)
                        else:
                            try:
                                with self.metrics.lock:
                                    self.metrics.total_inference_time += infer_t
                                    self.metrics.total_inference_requests += 1
                            except Exception:
                                pass

                        fire, zap, spinner_bucket, spinner_value = action_index_to_components(action_index)
                        try:
                            enable_zap_gate = bool(getattr(RL_CONFIG, 'enable_superzap_gate', False))
                        except Exception:
                            enable_zap_gate = False

                        # Never run the zap gate during training. If the config accidentally
                        # leaves it enabled, force it off whenever the agent is updating so
                        # we don't overwrite the DQN's chosen actions (which torpedoes Agree%).
                        if enable_zap_gate:
                            try:
                                training_active = bool(getattr(self.agent, 'training_enabled', True))
                            except Exception:
                                training_active = True
                            if training_active:
                                enable_zap_gate = False

                        if enable_zap_gate and zap:
                            try:
                                pzap = float(getattr(RL_CONFIG, 'superzap_prob', 0.01))
                            except Exception:
                                pzap = 0.01
                            if random.random() >= max(0.0, min(1.0, pzap)):
                                zap = False
                                fire_zap_idx = fire_zap_to_discrete(fire, zap)
                                action_index = compose_action_index(fire_zap_idx, spinner_bucket)
                        action_source = 'dqn'
                        if self._verbose_enabled():
                            try:
                                expert_idx, exp_spinner_val = get_expert_hybrid_action(
                                    frame.enemy_seg,
                                    frame.player_seg,
                                    frame.open_level,
                                    frame.expert_fire,
                                    frame.expert_zap,
                                )
                                exp_fire, exp_zap, exp_spin_bucket, exp_spin_quant = action_index_to_components(int(expert_idx))
                                dqn_agree = (int(action_index) == int(expert_idx))
                                if not dqn_agree:
                                    same_fire = bool(fire) == bool(exp_fire)
                                    same_zap = bool(zap) == bool(exp_zap)
                                    if same_fire and same_zap:
                                        dqn_sign = _spinner_sign(float(spinner_value))
                                        exp_sign = _spinner_sign(float(exp_spin_quant))
                                        dqn_agree = dqn_sign == exp_sign
                                # Fetch latest global frame count safely
                                try:
                                    with self.metrics.lock:
                                        frame_counter_dbg = getattr(self.metrics.metrics, 'frame_count', 0)
                                except Exception:
                                    frame_counter_dbg = getattr(metrics, 'frame_count', 0)
                                verbose_line = (
                                    f"[VERBOSE DQN] frame={frame_counter_dbg:,} client={client_id} lvl={frame.level_number} open={frame.open_level} "
                                    f"eps={epsilon:.4f} exp_ratio={expert_ratio:.4f} "
                                    f"dqn(fire={int(fire)},zap={int(zap)},bucket={spinner_bucket},spin={spinner_value:.3f}) "
                                    f"expert(fire={int(exp_fire)},zap={int(exp_zap)},bucket={exp_spin_bucket},spin={exp_spin_quant:.3f}) "
                                    f"agree={'Y' if dqn_agree else 'N'} "
                                    f"obj={frame.objreward:.5f} subj={frame.subjreward:.5f}"
                                )
                                print(verbose_line)
                            except Exception as dbg_exc:
                                print(f"[VERBOSE DQN] error while dumping frame info: {dbg_exc}")
                else:
                    action_source = 'none'

                # store for next step
                state['last_state'] = frame.state
                state['last_action_index'] = action_index
                state['last_action_source'] = action_source

                # send to game
                game_fire, game_zap, game_spinner = hybrid_to_game_action(action_index)
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
