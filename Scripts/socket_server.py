#!/usr/bin/env python3
# ==================================================================================================================
# ||  TEMPEST AI v2 • SOCKET BRIDGE SERVER                                                                       ||
# ||  TCP server bridging Lua (MAME) ↔ Python.  Protocol unchanged from v1.                                     ||
# ==================================================================================================================
"""Socket server — receives frames from Lua, queries agent, returns 3-byte actions."""

if __name__ == "__main__":
    print("This is not the main application, run 'main.py' instead")
    exit(1)

import os, sys, time, socket, select, struct, threading, traceback, random, queue
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
        self._thread = threading.Thread(target=self._consume, daemon=True)
        self._thread.start()

    def step_async(self, *args, **kwargs):
        try:
            self.queue.put((args, kwargs), timeout=0.05)
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
            for a, kw in batch:
                try:
                    self.agent.step(*a, **kw)
                except Exception as e:
                    print(f"AsyncReplayBuffer error: {e}")

    def stop(self):
        self.running = False
        # Drain remaining
        while True:
            try:
                a, kw = self.queue.get_nowait()
                self.agent.step(*a, **kw)
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
        return req.action if req.action is not None else (0, 0)

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
                        act = (0, 0)
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
            req.action = (0, 0)
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

            BATCH = 8
            local_accum = 0

            while self.running and not self.shutdown_event.is_set():
                ready = select.select([sock], [], [], 0.0)
                if not ready[0]:
                    time.sleep(0.0005)
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
                    sock.sendall(struct.pack("bbb", 0, 0, 0))
                    continue

                with self.client_lock:
                    if cid not in self.client_states:
                        break
                    cs = self.client_states[cid]
                    cs["frames"] += 1
                    cs["level_number"] = frame.level_number
                    now = time.time()
                    el = now - cs["last_time"]
                    if el >= 1.0:
                        cs["fps"] = 1.0 / el
                        cs["last_time"] = now

                local_accum += 1
                if local_accum >= BATCH:
                    self.metrics.update_frame_count(delta=local_accum)
                    local_accum = 0
                    self.metrics.update_epsilon()
                    self.metrics.update_expert_ratio()
                    self._calc_avg_level()
                self.metrics.update_game_state(frame.enemy_seg, frame.open_level)

                # ── Process previous step ───────────────────────────────
                if cs.get("last_state") is not None and cs.get("last_action") is not None:
                    fz_i, sp_i = cs["last_action"]
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
                            joint = combine_action_indices(fz_i, sp_i)
                            matured = nstep.add(cs["last_state"], joint, total_r,
                                                frame.state, bool(frame.done),
                                                actor=tag, priority_reward=total_r)
                            for s0, a, Rn, pR, sn, dn, h, act in matured:
                                fz_n, sp_n = split_joint_action(a)
                                self.async_buffer.step_async(
                                    s0, (fz_n, sp_n), Rn, sn, bool(dn),
                                    actor=act, horizon=int(h), priority_reward=pR)
                        else:
                            self.async_buffer.step_async(
                                cs["last_state"], (fz_i, sp_i), total_r,
                                frame.state, bool(frame.done), actor=tag, horizon=1,
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
                        sock.sendall(struct.pack("bbb", 0, 0, 0))
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
                fz_idx = sp_idx = 0
                fire = zap = False
                spinner_val = 0.0
                action_source = "none"

                if self.agent:
                    expert_ratio = self.metrics.get_expert_ratio()
                    # Boost expert sampling during tube zoom (e.g., 0.5 -> 1.0 with 2x multiplier).
                    if frame.gamestate == int(getattr(RL_CONFIG, "expert_ratio_zoom_gamestate", 0x20)):
                        zoom_mult = float(getattr(RL_CONFIG, "expert_ratio_zoom_multiplier", 1.0))
                        expert_ratio = max(0.0, min(1.0, expert_ratio * zoom_mult))
                    use_expert = (random.random() < expert_ratio) and not metrics.override_expert

                    if use_expert:
                        fire, zap, spinner_val = get_expert_action(
                            frame.enemy_seg, frame.player_seg, frame.open_level,
                            frame.expert_fire, frame.expert_zap)
                        fz_idx = fire_zap_to_discrete(fire, zap)
                        sp_idx = quantize_spinner_value(spinner_val)
                        action_source = "expert"
                    else:
                        epsilon = self.metrics.get_effective_epsilon()
                        t0 = time.perf_counter()
                        if self.inference_batcher is not None:
                            fz_idx, sp_idx = self.inference_batcher.infer(frame.state, epsilon)
                        else:
                            fz_idx, sp_idx = self.agent.act(frame.state, epsilon)
                        self.metrics.add_inference_time(time.perf_counter() - t0)
                        fire, zap = discrete_to_fire_zap(fz_idx)
                        spinner_val = spinner_index_to_value(sp_idx)
                        action_source = "dqn"

                cs["last_state"] = frame.state
                cs["last_action"] = (fz_idx, sp_idx)
                cs["prev_action_source"] = action_source
                cs["last_action_source"] = action_source

                gf, gz, gs = encode_action_to_game(fire, zap, spinner_val)
                try:
                    sock.sendall(struct.pack("bbb", gf, gz, gs))
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
