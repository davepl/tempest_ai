#!/usr/bin/env python3
# ==================================================================================================================
# ||  ROBOTRON AI v2 • SOCKET BRIDGE SERVER                                                                      ||
# ||  TCP server bridging Lua (MAME) ↔ Python.  Protocol unchanged from v1.                                     ||
# ==================================================================================================================
"""Socket server bridging Lua (MAME) and Python for Robotron training."""

if __name__ == "__main__":
    print("This is not the main application, run 'main.py' instead")
    exit(1)

import time, socket, select, struct, threading, traceback, queue
import numpy as np
from collections import deque

from aimodel import (
    parse_frame_data,
    encode_action_to_game,
    combine_action_indices,
    split_joint_action,
    SafeMetrics,
)
from config import RL_CONFIG, SERVER_CONFIG, metrics

try:
    from nstep_buffer import NStepReplayBuffer
except ImportError:
    from Scripts.nstep_buffer import NStepReplayBuffer

try:
    from metrics_display import (
        add_episode_to_dqn100k_window,
        add_episode_to_dqn1m_window,
        add_episode_to_dqn5m_window,
        add_episode_to_total_windows,
        add_episode_to_eplen_window,
    )
except ImportError:
    add_episode_to_dqn100k_window = add_episode_to_dqn1m_window = add_episode_to_dqn5m_window = lambda *a: None
    add_episode_to_total_windows = lambda *a: None
    add_episode_to_eplen_window = lambda *a: None


# ── Async buffer (queues step() calls to avoid blocking frame loop) ─────────
class AsyncReplayBuffer:
    def __init__(self, agent, batch_size=100, max_queue_size=10000):
        self.agent = agent
        self.batch_size = batch_size
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.running = True
        # Per-client rolling window of buffer indices for pre-death boosting
        self._lookback = int(getattr(RL_CONFIG, 'pre_death_lookback', 120))
        self._client_indices = {}          # client_id -> deque(maxlen=lookback)
        self._thread = threading.Thread(target=self._consume, daemon=True)
        self._thread.start()

    def step_async(self, *args, client_id=None, **kwargs):
        try:
            self.queue.put(("step", client_id, args, kwargs), timeout=0.05)
        except queue.Full:
            pass

    def boost_pre_death(self, client_id):
        """Queue a pre-death priority boost for recent indices of *client_id*."""
        try:
            self.queue.put(("boost", client_id, None, None), timeout=0.05)
        except queue.Full:
            pass

    def _consume(self):
        while self.running:
            try:
                item = self.queue.get(timeout=0.01)
            except queue.Empty:
                continue
            batch = [item]
            while len(batch) < self.batch_size:
                try:
                    batch.append(self.queue.get_nowait())
                except queue.Empty:
                    break
            for cmd, cid, a, kw in batch:
                try:
                    if cmd == "step":
                        idx = self.agent.step(*a, **kw)
                        if cid is not None and idx is not None and idx >= 0:
                            if cid not in self._client_indices:
                                self._client_indices[cid] = deque(maxlen=self._lookback)
                            self._client_indices[cid].append(idx)
                    elif cmd == "boost":
                        self._do_boost(cid)
                except Exception as e:
                    print(f"AsyncReplayBuffer error: {e}")

    def _do_boost(self, client_id):
        """Boost priorities of recent buffer indices for *client_id*."""
        indices = self._client_indices.get(client_id)
        if not indices:
            return
        boost = float(getattr(RL_CONFIG, 'pre_death_priority_boost', 2.0))
        if boost <= 1.0:
            indices.clear()
            return
        try:
            self.agent.memory.boost_priorities(list(indices), boost)
        except Exception as e:
            print(f"  Pre-death boost error: {e}")
        indices.clear()

    def remove_client(self, client_id):
        """Clean up index tracking when a client disconnects."""
        self._client_indices.pop(client_id, None)

    def stop(self):
        self.running = False
        # Drain remaining
        while True:
            try:
                cmd, cid, a, kw = self.queue.get_nowait()
                if cmd == "step" and a is not None:
                    self.agent.step(*a, **kw)
                elif cmd == "boost":
                    self._do_boost(cid)
            except queue.Empty:
                break
            except Exception:
                pass
        self._thread.join(timeout=5.0)


class _InferenceRequest:
    __slots__ = ("state", "epsilon", "event", "action")

    def __init__(self, state, epsilon: float):
        self.state = state
        self.epsilon = float(epsilon)
        self.event = threading.Event()
        self.action = None


class AsyncInferenceBatcher:
    """Micro-batch inference requests across clients for better GPU utilization."""

    def __init__(self, agent, max_batch_size=32, max_wait_ms=1.0, request_timeout_ms=50.0):
        self.agent = agent
        self.max_batch_size = max(1, int(max_batch_size))
        self.max_wait_s = max(0.0, float(max_wait_ms) / 1000.0)
        self.request_timeout_s = max(0.001, float(request_timeout_ms) / 1000.0)
        self.queue = queue.Queue(maxsize=20000)
        self.running = True
        self._thread = threading.Thread(target=self._consume, daemon=True, name="InferBatchWorker")
        self._thread.start()

    def infer(self, state, epsilon: float):
        if not self.running:
            return self.agent.act(state, epsilon)
        req = _InferenceRequest(state, epsilon)
        try:
            self.queue.put(req, timeout=self.request_timeout_s)
        except queue.Full:
            return self.agent.act(state, epsilon)
        if not req.event.wait(timeout=self.request_timeout_s):
            return self.agent.act(state, epsilon)
        return req.action if req.action is not None else (0, 0, False)

    def _consume(self):
        while self.running or not self.queue.empty():
            try:
                first = self.queue.get(timeout=0.01)
            except queue.Empty:
                continue

            batch = [first]
            deadline = time.perf_counter() + self.max_wait_s
            while len(batch) < self.max_batch_size:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    batch.append(self.queue.get(timeout=remaining))
                except queue.Empty:
                    break

            try:
                states = [r.state for r in batch]
                epsilons = [r.epsilon for r in batch]
                actions = self.agent.act_batch(states, epsilons)
            except Exception as e:
                print(f"AsyncInferenceBatcher error: {e}")
                actions = []

            for idx, req in enumerate(batch):
                act = actions[idx] if idx < len(actions) else None
                if act is None:
                    try:
                        act = self.agent.act(req.state, req.epsilon)
                    except Exception:
                        act = (0, 0, False)
                req.action = act
                req.event.set()

    def stop(self):
        self.running = False
        self._thread.join(timeout=5.0)
        while True:
            try:
                req = self.queue.get_nowait()
            except queue.Empty:
                break
            req.action = (0, 0, False)
            req.event.set()


# ── Socket Server ───────────────────────────────────────────────────────────
class SocketServer:
    def __init__(self, host, port, agent, metrics_wrapper):
        self.host = host
        self.port = port
        self.agent = agent
        self.async_buffer = AsyncReplayBuffer(agent) if agent else None
        self.inference_batcher = None
        if agent and bool(getattr(RL_CONFIG, "inference_batching_enabled", True)):
            self.inference_batcher = AsyncInferenceBatcher(
                agent,
                max_batch_size=int(getattr(RL_CONFIG, "inference_batch_max_size", 32)),
                max_wait_ms=float(getattr(RL_CONFIG, "inference_batch_wait_ms", 1.0)),
                request_timeout_ms=float(getattr(RL_CONFIG, "inference_request_timeout_ms", 50.0)),
            )
            print(
                "Async inference batching enabled: "
                f"max_batch={self.inference_batcher.max_batch_size}, "
                f"wait_ms={self.inference_batcher.max_wait_s * 1000.0:.2f}"
            )
        self.metrics = SafeMetrics(metrics_wrapper)

        self.server_socket = None
        self.running = False
        self.shutdown_event = threading.Event()

        self.clients = {}
        self.client_states = {}
        self.client_lock = threading.Lock()

    def _alloc_id(self):
        with self.client_lock:
            cid = 0
            while cid in self.clients:
                cid += 1
            return cid

    def _init_client(self, cid):
        n = max(1, int(getattr(RL_CONFIG, "n_step", 1)))
        gamma = float(getattr(RL_CONFIG, "gamma", 0.99))
        nstep = NStepReplayBuffer(n_step=n, gamma=gamma) if n > 1 else None
        with self.client_lock:
            self.client_states[cid] = {
                "frames": 0, "last_time": time.time(), "fps": 0.0,
                "level_number": 0, "last_state": None, "last_action": None,
                "last_action_source": None, "prev_action_source": None,
                "total_reward": 0.0, "ep_dqn_reward": 0.0, "ep_expert_reward": 0.0,
                "ep_subj_reward": 0.0, "ep_obj_reward": 0.0, "ep_frames": 0,
                "was_done": False, "nstep": nstep,
            }
            metrics.client_count = len(self.client_states)

    def handle_client(self, sock, cid):
        try:
            sock.setblocking(False)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)

            # Handshake
            try:
                sock.setblocking(True)
                sock.settimeout(5.0)
                ping = sock.recv(2)
                if not ping or len(ping) < 2:
                    raise ConnectionError("No handshake")
            finally:
                sock.setblocking(False)
                sock.settimeout(None)

            while self.running and not self.shutdown_event.is_set():
                ready = select.select([sock], [], [], 0.002)
                if not ready[0]:
                    continue

                # Read length header
                hdr = sock.recv(2)
                if not hdr or len(hdr) < 2:
                    raise ConnectionError("EOF")
                dlen = struct.unpack(">H", hdr)[0]

                # Read payload
                data = b""
                rem = dlen
                while rem > 0:
                    chunk = sock.recv(min(32768, rem))
                    if not chunk:
                        raise ConnectionError("Broken")
                    data += chunk
                    rem -= len(chunk)

                if len(data) >= 2:
                    n = struct.unpack(">H", data[:2])[0]
                    if n != SERVER_CONFIG.params_count:
                        print(f"Client {cid}: param mismatch {n} != {SERVER_CONFIG.params_count}")
                        break
                else:
                    break

                frame = parse_frame_data(data)
                if not frame:
                    sock.sendall(struct.pack("bb", 0, 0))
                    continue

                with self.client_lock:
                    if cid not in self.client_states:
                        break
                    cs = self.client_states[cid]
                    cs["frames"] += 1
                    cs["level_number"] = frame.level_number
                    # Ignore attract/menu/transient frames for high-score tracking.
                    if frame.player_alive and frame.game_score > self.metrics.peak_game_score:
                        self.metrics.peak_game_score = frame.game_score
                    now = time.time()
                    el = now - cs["last_time"]
                    if el >= 1.0:
                        cs["fps"] = 1.0 / el
                        cs["last_time"] = now

                # Keep metrics synchronized to actual processed frames.
                # Do not batch this by connection-local counters, because
                # reconnects can reset local state and hide frame progress.
                self.metrics.update_frame_count(delta=1)
                self.metrics.update_epsilon()
                self.metrics.update_expert_ratio()
                self._calc_avg_level()
                self.metrics.update_game_state(0, False)

                # ── Process previous step ───────────────────────────────
                if cs.get("last_state") is not None and cs.get("last_action") is not None:
                    move_i, fire_i = cs["last_action"]
                    subj_r = float(frame.subjreward) * RL_CONFIG.subj_reward_scale
                    obj_r = float(frame.objreward) * RL_CONFIG.obj_reward_scale
                    total_r = obj_r + subj_r
                    # Use wider clip on terminal frames so death penalty passes through
                    clip = RL_CONFIG.death_reward_clip if frame.done else RL_CONFIG.reward_clip
                    total_r = max(-clip, min(clip, total_r))

                    if self.agent:
                        tag = cs.get("prev_action_source", "dqn")
                        nstep = cs.get("nstep")
                        if nstep is not None:
                            joint = combine_action_indices(move_i, fire_i)
                            matured = nstep.add(cs["last_state"], joint, total_r,
                                                frame.state, bool(frame.done),
                                                actor=tag, priority_reward=total_r)
                            for s0, a, Rn, pR, sn, dn, h, act in matured:
                                move_n, fire_n = split_joint_action(a)
                                self.async_buffer.step_async(
                                    s0, (move_n, fire_n), Rn, sn, bool(dn),
                                    client_id=cid, actor=act, horizon=int(h), priority_reward=pR)
                        else:
                            self.async_buffer.step_async(
                                cs["last_state"], (move_i, fire_i), total_r,
                                frame.state, bool(frame.done), client_id=cid, actor=tag, horizon=1,
                                priority_reward=total_r)

                    cs["total_reward"] += total_r
                    cs["ep_subj_reward"] = cs.get("ep_subj_reward", 0) + subj_r
                    cs["ep_obj_reward"] = cs.get("ep_obj_reward", 0) + obj_r
                    cs["ep_frames"] = cs.get("ep_frames", 0) + 1
                    src = cs.get("prev_action_source")
                    if src == "dqn":
                        cs["ep_dqn_reward"] += total_r
                    elif src == "expert":
                        cs["ep_expert_reward"] += total_r

                # ── Terminal ────────────────────────────────────────────
                if frame.done:
                    # Boost priorities of the last N frames for this client
                    if self.async_buffer is not None:
                        self.async_buffer.boost_pre_death(cid)
                    if not cs.get("was_done", False):
                        self.metrics.add_episode_reward(
                            cs["total_reward"], cs["ep_dqn_reward"], cs["ep_expert_reward"],
                            cs.get("ep_subj_reward", 0), cs.get("ep_obj_reward", 0),
                            length=cs.get("ep_frames", 0))
                        try:
                            add_episode_to_dqn100k_window(cs["ep_dqn_reward"], cs.get("ep_frames", 0))
                            add_episode_to_dqn1m_window(cs["ep_dqn_reward"], cs.get("ep_frames", 0))
                            add_episode_to_dqn5m_window(cs["ep_dqn_reward"], cs.get("ep_frames", 0))
                            add_episode_to_total_windows(cs["total_reward"], cs.get("ep_frames", 0))
                            add_episode_to_eplen_window(cs.get("ep_frames", 0))
                        except Exception:
                            pass
                    cs["was_done"] = True
                    try:
                        sock.sendall(struct.pack("bb", 0, 0))
                    except Exception:
                        break
                    cs["last_state"] = cs["last_action"] = None
                    cs["last_action_source"] = cs["prev_action_source"] = None
                    cs["total_reward"] = cs["ep_dqn_reward"] = cs["ep_expert_reward"] = 0.0
                    cs["ep_subj_reward"] = cs["ep_obj_reward"] = 0.0
                    cs["ep_frames"] = 0
                    continue

                if cs.get("was_done"):
                    cs["was_done"] = False
                    cs["total_reward"] = cs["ep_dqn_reward"] = cs["ep_expert_reward"] = 0.0
                    cs["ep_subj_reward"] = cs["ep_obj_reward"] = 0.0
                    cs["ep_frames"] = 0

                # ── Choose action ───────────────────────────────────────
                self.metrics.increment_total_controls()
                move_idx = fire_idx = 0
                action_source = "none"

                if self.agent:
                    epsilon = self.metrics.get_effective_epsilon()
                    t0 = time.perf_counter()
                    if self.inference_batcher is not None:
                        move_idx, fire_idx, _ = self.inference_batcher.infer(frame.state, epsilon)
                    else:
                        move_idx, fire_idx, _ = self.agent.act(frame.state, epsilon)
                    self.metrics.add_inference_time(time.perf_counter() - t0)
                    action_source = "dqn"

                cs["last_state"] = frame.state
                cs["last_action"] = (move_idx, fire_idx)
                cs["prev_action_source"] = action_source
                cs["last_action_source"] = action_source

                move_cmd, fire_cmd = encode_action_to_game(move_idx, fire_idx)
                try:
                    sock.sendall(struct.pack("bb", move_cmd, fire_cmd))
                except Exception:
                    break

        except Exception as e:
            print(f"Client {cid} error: {e}")
            traceback.print_exc()
        finally:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                sock.close()
            except Exception:
                pass
            with self.client_lock:
                self.client_states.pop(cid, None)
                self.clients[cid] = None
                metrics.client_count = sum(1 for v in self.clients.values() if v is not None)
            if self.async_buffer is not None:
                self.async_buffer.remove_client(cid)
            threading.Timer(1.0, self._cleanup).start()

    def _cleanup(self):
        with self.client_lock:
            dead = [k for k, v in self.clients.items() if v is None]
            for k in dead:
                del self.clients[k]
            metrics.client_count = len(self.clients)

    def _calc_avg_level(self):
        try:
            with self.client_lock:
                lvls = [s.get("level_number", 0) for s in self.client_states.values() if s.get("level_number", 0) >= 0]
                metrics.average_level = sum(lvls) / len(lvls) if lvls else 0
                for lv in lvls:
                    if lv > metrics.peak_level:
                        metrics.peak_level = lv
        except Exception:
            pass

    def start(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        for i in range(10):
            try:
                if self.shutdown_event.is_set():
                    return
                self.server_socket.bind((self.host, self.port))
                break
            except OSError as e:
                if e.errno in (98, 48):
                    print(f"Port {self.port} busy, retry {i+1}/10")
                    time.sleep(1.0)
                else:
                    raise
        else:
            raise OSError(f"Cannot bind {self.host}:{self.port}")

        self.server_socket.listen(SERVER_CONFIG.max_clients)
        self.server_socket.setblocking(False)
        self.running = True
        print(f"SocketServer listening on {self.host}:{self.port}")

        try:
            while self.running and not self.shutdown_event.is_set():
                try:
                    rd, _, _ = select.select([self.server_socket], [], [], 0.05)
                except (OSError, ValueError):
                    if self.shutdown_event.is_set():
                        break
                    raise
                if not self.server_socket:
                    break
                if self.server_socket in rd:
                    try:
                        cs, addr = self.server_socket.accept()
                    except OSError:
                        continue
                    cid = self._alloc_id()
                    self._init_client(cid)
                    t = threading.Thread(target=self.handle_client, args=(cs, cid), daemon=True)
                    with self.client_lock:
                        self.clients[cid] = t
                    t.start()
        except Exception as e:
            if not self.shutdown_event.is_set():
                print(f"Server error: {e}")
                traceback.print_exc()
        finally:
            self.stop()

    def stop(self):
        if self.shutdown_event.is_set() and not self.running:
            return
        self.running = False
        self.shutdown_event.set()
        if self.inference_batcher:
            print("Stopping async inference batcher...")
            self.inference_batcher.stop()
            self.inference_batcher = None
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
