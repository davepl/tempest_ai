#!/usr/bin/env python3
# ==================================================================================================================
# ||  ROBOTRON AI v2 • SOCKET BRIDGE SERVER                                                                      ||
# ||  TCP server bridging Lua (MAME) ↔ Python.                                                                    ||
# ==================================================================================================================
"""Socket server bridging Lua (MAME) and Python for Robotron training."""

if __name__ == "__main__":
    print("This is not the main application, run 'main.py' instead")
    exit(1)

import os, time, socket, select, struct, threading, traceback, queue, random
import numpy as np
from collections import deque

from aimodel import (
    parse_frame_data,
    get_expert_action,
    get_cleanup_fire_override,
    encode_action_to_game,
    combine_action_indices,
    split_joint_action,
    SafeMetrics,
)
from config import RL_CONFIG, SERVER_CONFIG, metrics, game_settings
try:
    from inference_pool import ProcessInferencePool
except ImportError:
    try:
        from Scripts.inference_pool import ProcessInferencePool
    except ImportError:
        ProcessInferencePool = None

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

_ACTION_DIAGNOSTICS = os.getenv("ROBOTRON_ACTION_DIAGNOSTICS", "").strip().lower() not in {"", "0", "false", "off", "no"}
try:
    _ACTION_DIAG_INTERVAL = max(500, int(os.getenv("ROBOTRON_ACTION_DIAG_INTERVAL", "5000")))
except Exception:
    _ACTION_DIAG_INTERVAL = 5000
_FORCE_RANDOM_ACTIONS = os.getenv("ROBOTRON_FORCE_RANDOM_ACTIONS", "").strip().lower() not in {"", "0", "false", "off", "no"}
_MAX_FRAME_PAYLOAD_BYTES = 4 * 1024 * 1024

# ── Fire-hold (keeps fire direction stable for N frames so game's LSPROC registers a shot)
FIRE_HOLD_FRAMES = 4     # must be ≥3 (game needs 3 stable frames to fire)
FRAME_STACK = max(1, int(getattr(RL_CONFIG, "frame_stack", 1)))
_START_PULSE_VALID_FRAMES = 240
_GAMEPLAY_RESET_DEAD_FRAMES = 180
_GAMEPLAY_PLAUSIBLE_START_STREAK = 8


def _reset_state_stack(cs: dict):
    st = cs.get("state_stack")
    if st is not None:
        st.clear()


def _push_stacked_state(cs: dict, frame_state: np.ndarray) -> np.ndarray:
    """Push one base-frame state and return stacked state (oldest→latest)."""
    st = cs.get("state_stack")
    if st is None or getattr(st, "maxlen", None) != FRAME_STACK:
        st = deque(maxlen=FRAME_STACK)
        cs["state_stack"] = st

    cur = np.asarray(frame_state, dtype=np.float32).copy()
    if len(st) == 0:
        for _ in range(FRAME_STACK):
            st.append(cur.copy())
    else:
        st.append(cur)

    if FRAME_STACK == 1:
        return st[-1]
    return np.concatenate(list(st), axis=0).astype(np.float32, copy=False)


def _apply_fire_hold(cs: dict, raw_fire: int) -> int:
    """Fixed-cadence fire hold: direction changes at most every FIRE_HOLD_FRAMES.
    Between cadence ticks the held direction is emitted unchanged.
    On each tick the most-recently-commanded direction is adopted.
    This guarantees LSPROC's 3-stable-frame requirement while letting
    the model's latest intent always win on the next boundary."""
    # Always track the newest requested fire direction so commands issued
    # during a hold window are not dropped.
    cs["fire_pending_dir"] = int(raw_fire)
    count = cs.get("fire_hold_count", 0)
    if count > 0:
        cs["fire_hold_count"] = count - 1
        return cs.get("fire_hold_dir", raw_fire)
    # Cadence tick: adopt the latest queued request.
    next_fire = int(cs.get("fire_pending_dir", raw_fire))
    cs["fire_hold_dir"] = next_fire
    cs["fire_hold_count"] = FIRE_HOLD_FRAMES - 1
    return next_fire


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
        self._episode_indices = {}         # (client_id, episode_id) -> [replay_idx, ...]
        self._episode_stats = {}           # (client_id, episode_id) -> episode summary
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

    @staticmethod
    def _episode_key(client_id, episode_id) -> tuple[int, int]:
        return int(client_id if client_id is not None else -1), int(episode_id or 0)

    def _track_episode_step(
        self,
        client_id,
        episode_id,
        replay_idx,
        reward: float,
        done: bool,
        actor: str,
        wave_number: int,
        start_wave: int,
    ) -> None:
        key = self._episode_key(client_id, episode_id)
        stats = self._episode_stats.get(key)
        if stats is None:
            stats = {
                "dqn_reward": 0.0,
                "total_reward": 0.0,
                "length": 0,
                "max_wave": max(1, int(start_wave or wave_number or 1)),
                "start_wave": max(1, int(start_wave or 1)),
            }
            self._episode_stats[key] = stats

        reward_f = float(reward)
        stats["total_reward"] += reward_f
        stats["length"] += 1
        stats["max_wave"] = max(int(stats["max_wave"]), max(1, int(wave_number or 1)))
        if actor == "dqn" and replay_idx is not None and int(replay_idx) >= 0:
            self._episode_indices.setdefault(key, []).append(int(replay_idx))
            stats["dqn_reward"] += reward_f

        if done:
            idx_list = self._episode_indices.pop(key, [])
            final_stats = self._episode_stats.pop(key, None)
            if final_stats is not None and idx_list:
                try:
                    self.agent.consider_self_imitation(
                        idx_list,
                        dqn_reward=float(final_stats["dqn_reward"]),
                        total_reward=float(final_stats["total_reward"]),
                        length=int(final_stats["length"]),
                        max_wave=int(final_stats["max_wave"]),
                        start_wave=int(final_stats["start_wave"]),
                    )
                except Exception as e:
                    print(f"AsyncReplayBuffer self-imitation error: {e}")

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
            step_items = [(cid, a, kw) for cmd, cid, a, kw in batch if cmd == "step"]
            if step_items:
                try:
                    if hasattr(self.agent, "step_batch"):
                        transitions = []
                        episode_meta = []
                        for cid, a, kw in step_items:
                            state, action, reward, next_state, done = a[:5]
                            actor = kw.get("actor", "dqn")
                            horizon = kw.get("horizon", 1)
                            priority_reward = kw.get("priority_reward", None)
                            wave_number = kw.get("wave_number", 1)
                            start_wave = kw.get("start_wave", 1)
                            transitions.append((
                                state,
                                action,
                                reward,
                                next_state,
                                done,
                                actor,
                                horizon,
                                priority_reward,
                                wave_number,
                                start_wave,
                            ))
                            episode_meta.append((
                                cid,
                                kw.get("episode_id", 0),
                                float(reward),
                                bool(done),
                                str(actor),
                                int(wave_number),
                                int(start_wave),
                            ))
                        indices = self.agent.step_batch(transitions)
                        if indices is None:
                            indices = [-1] * len(episode_meta)
                        for (cid, ep_id, reward, done, actor, wave_number, start_wave), idx in zip(episode_meta, indices):
                            if cid is not None and idx is not None and idx >= 0:
                                if cid not in self._client_indices:
                                    self._client_indices[cid] = deque(maxlen=self._lookback)
                                self._client_indices[cid].append(int(idx))
                            self._track_episode_step(
                                cid,
                                ep_id,
                                idx,
                                reward,
                                done,
                                actor,
                                wave_number,
                                start_wave,
                            )
                    else:
                        for cid, a, kw in step_items:
                            idx = self.agent.step(*a, **kw)
                            if cid is not None and idx is not None and idx >= 0:
                                if cid not in self._client_indices:
                                    self._client_indices[cid] = deque(maxlen=self._lookback)
                                self._client_indices[cid].append(idx)
                            state, action, reward, next_state, done = a[:5]
                            self._track_episode_step(
                                cid,
                                kw.get("episode_id", 0),
                                idx,
                                float(reward),
                                bool(done),
                                str(kw.get("actor", "dqn")),
                                int(kw.get("wave_number", 1)),
                                int(kw.get("start_wave", 1)),
                            )
                except Exception as e:
                    print(f"AsyncReplayBuffer error: {e}")

            for cmd, cid, a, kw in batch:
                if cmd != "boost":
                    continue
                try:
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
        prefix = int(client_id if client_id is not None else -1)
        for key in [k for k in self._episode_indices.keys() if k[0] == prefix]:
            self._episode_indices.pop(key, None)
        for key in [k for k in self._episode_stats.keys() if k[0] == prefix]:
            self._episode_stats.pop(key, None)

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
    __slots__ = ("state", "epsilon", "locked_fire", "event", "action")

    def __init__(self, state, epsilon: float, locked_fire: int | None = None):
        self.state = state
        self.epsilon = float(epsilon)
        self.locked_fire = locked_fire
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

    def infer(self, state, epsilon: float, locked_fire: int | None = None, route_key: int | None = None):
        if not self.running:
            return self.agent.act(state, epsilon, locked_fire=locked_fire)
        req = _InferenceRequest(state, epsilon, locked_fire=locked_fire)
        try:
            self.queue.put(req, timeout=self.request_timeout_s)
        except queue.Full:
            return self.agent.act(state, epsilon, locked_fire=locked_fire)
        if not req.event.wait(timeout=self.request_timeout_s):
            return self.agent.act(state, epsilon, locked_fire=locked_fire)
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
                locked_fires = [r.locked_fire for r in batch]
                actions = self.agent.act_batch(states, epsilons, locked_fires=locked_fires)
            except Exception as e:
                print(f"AsyncInferenceBatcher error: {e}")
                actions = []

            for idx, req in enumerate(batch):
                act = actions[idx] if idx < len(actions) else None
                if act is None:
                    try:
                        act = self.agent.act(req.state, req.epsilon, locked_fire=req.locked_fire)
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
        allow_async_replay = bool(getattr(agent, "allow_async_replay_buffer", True)) if agent else False
        self.async_buffer = AsyncReplayBuffer(agent) if agent and allow_async_replay else None
        self.inference_batcher = None
        if agent:
            allow_proc_pool = bool(getattr(agent, "allow_process_inference_pool", True))
            allow_async_batcher = bool(getattr(agent, "allow_async_inference_batcher", True))
            shard_workers = max(0, int(getattr(SERVER_CONFIG, "shard_workers", 0) or 0))
            is_master_shard_server = shard_workers > 0 and bool(getattr(agent, "publish_global_client_metrics", True))
            if is_master_shard_server:
                # In sharded mode, worker servers should carry inference load.
                # Avoid spawning an extra master-side process pool that competes
                # for the same GPU while only serving the preview/master path.
                allow_proc_pool = False
            proc_workers = max(1, int(getattr(RL_CONFIG, "inference_process_workers", 1) or 1))
            if proc_workers > 1 and ProcessInferencePool is not None and allow_proc_pool:
                self.inference_batcher = ProcessInferencePool(
                    agent,
                    worker_count=proc_workers,
                    max_batch_size=int(getattr(RL_CONFIG, "inference_batch_max_size", 32)),
                    max_wait_ms=float(getattr(RL_CONFIG, "inference_batch_wait_ms", 1.0)),
                    request_timeout_ms=float(getattr(RL_CONFIG, "inference_request_timeout_ms", 50.0)),
                )
                print(
                    "Multiprocess inference pool enabled: "
                    f"workers={proc_workers}, "
                    f"max_batch={int(getattr(RL_CONFIG, 'inference_batch_max_size', 32))}, "
                    f"wait_ms={float(getattr(RL_CONFIG, 'inference_batch_wait_ms', 1.0)):.2f}"
                )
            elif bool(getattr(RL_CONFIG, "inference_batching_enabled", True)) and allow_async_batcher:
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
        self.publish_global_client_metrics = bool(getattr(agent, "publish_global_client_metrics", True)) if agent else True
        if _FORCE_RANDOM_ACTIONS:
            print("FORCE RANDOM ACTIONS enabled (ROBOTRON_FORCE_RANDOM_ACTIONS=1)")

        self.server_socket = None
        self.running = False
        self.shutdown_event = threading.Event()

        self.clients = {}
        self.client_states = {}
        self.client_lock = threading.Lock()
        self.preview_cid = None

    def _submit_experience(self, *args, client_id=None, **kwargs):
        if self.async_buffer is not None:
            self.async_buffer.step_async(*args, client_id=client_id, **kwargs)
            return
        if self.agent is None:
            return
        try:
            self.agent.step(*args, client_id=client_id, **kwargs)
        except Exception as e:
            print(f"SocketServer experience submit error: {e}")

    def _alloc_id(self):
        with self.client_lock:
            cid = 0
            while cid in self.clients:
                cid += 1
            return cid

    @staticmethod
    def _source_preview_flag(enabled: bool) -> int:
        return 0x40 if enabled else 0x00

    @staticmethod
    def _source_hud_flag(enabled: bool) -> int:
        return 0x80 if enabled else 0x00

    @classmethod
    def _pack_action(
        cls,
        move_cmd: int,
        fire_cmd: int,
        source_code: int,
        preview_enabled: bool,
        hud_enabled: bool,
    ) -> bytes:
        gs = game_settings.snapshot()
        start_advanced = 1 if bool(gs.get("start_advanced", False)) else 0
        start_level_min = max(1, min(81, int(gs.get("start_level_min", 1) or 1)))
        source_u8 = (
            (int(source_code) & 0x0F)
            | cls._source_preview_flag(bool(preview_enabled))
            | cls._source_hud_flag(bool(hud_enabled))
        )
        return struct.pack("bbBBB", int(move_cmd), int(fire_cmd), int(source_u8), int(start_advanced), int(start_level_min))

    @staticmethod
    def _clear_preview_cache() -> None:
        with metrics.lock:
            metrics.game_preview_client_id = -1
            metrics.game_preview_seq = 0
            metrics.game_preview_width = 0
            metrics.game_preview_height = 0
            metrics.game_preview_format = ""
            metrics.game_preview_data = b""
            metrics.game_preview_updated_ts = 0.0
            metrics.game_preview_source_format = ""
            metrics.game_preview_encoded_bytes = 0
            metrics.game_preview_raw_bytes = 0
            metrics.game_preview_compression_ratio = 1.0
            metrics.game_preview_fps = 0.0

    @staticmethod
    def _parse_client_handshake(handshake_value: int) -> tuple[bool, int]:
        raw = max(0, int(handshake_value or 0))
        preview_capable = (raw & 0x01) != 0
        client_slot = max(0, raw >> 1)
        return preview_capable, client_slot

    @staticmethod
    def _expected_start_wave() -> int:
        try:
            gs = game_settings.snapshot()
            if bool(gs.get("start_advanced", False)):
                return max(1, min(81, int(gs.get("start_level_min", 1) or 1)))
        except Exception:
            pass
        return 1

    @classmethod
    def _looks_like_real_game_start(cls, frame) -> bool:
        wave = max(0, int(getattr(frame, "level_number", 0) or 0))
        if wave <= 0 or wave > 81:
            return False
        lives = max(0, int(getattr(frame, "num_lasers", 0) or 0))
        if lives > 9:
            return False
        score = max(0, int(getattr(frame, "game_score", 0) or 0))
        if score > 0:
            return True
        expected = cls._expected_start_wave()
        return abs(wave - expected) <= 1

    def _pick_default_preview_client_locked(self) -> int | None:
        candidates = []
        for cid, cs in self.client_states.items():
            if not bool(cs.get("preview_capable", False)):
                continue
            try:
                slot = max(0, int(cs.get("client_slot", cid)))
            except Exception:
                slot = int(cid)
            candidates.append((slot, int(cid)))
        if not candidates:
            return None
        candidates.sort()
        return int(candidates[0][1])

    def _ensure_preview_client_selected_locked(self) -> tuple[int | None, bool]:
        selected = self.preview_cid
        if selected is not None:
            cs = self.client_states.get(int(selected))
            if isinstance(cs, dict) and bool(cs.get("preview_capable", False)):
                return int(selected), False
        fallback = self._pick_default_preview_client_locked()
        changed = self.preview_cid != fallback
        self.preview_cid = fallback
        return fallback, changed

    def _is_preview_client(self, cid: int) -> bool:
        changed = False
        with self.client_lock:
            preview_cid, changed = self._ensure_preview_client_selected_locked()
        if changed:
            self._clear_preview_cache()
        return preview_cid is not None and int(cid) == int(preview_cid)

    def _maybe_claim_preview_client(self, cid: int, handshake_value: int) -> bool:
        preview_capable, client_slot = self._parse_client_handshake(handshake_value)
        changed = False
        selected = None
        with self.client_lock:
            cs = self.client_states.get(cid)
            if isinstance(cs, dict):
                cs["preview_capable"] = bool(preview_capable)
                cs["client_slot"] = int(client_slot)
            selected, changed = self._ensure_preview_client_selected_locked()
        if changed:
            self._clear_preview_cache()
        return bool(preview_capable) and selected is not None and int(selected) == int(cid)

    def _preview_enabled_for_client(self, cid: int) -> bool:
        if not self._is_preview_client(cid):
            return False
        with self.client_lock:
            cs = self.client_states.get(int(cid))
            if not isinstance(cs, dict) or not bool(cs.get("preview_capable", False)):
                return False
        with metrics.lock:
            # Check if preview is enabled via dashboard checkbox
            if not getattr(metrics, "preview_capture_enabled", True):
                return False
            # Only enable if web clients are connected
            return int(getattr(metrics, "web_client_count", 0) or 0) > 0

    def _hud_enabled_for_client(self, cid: int) -> bool:
        if not self._is_preview_client(cid):
            return False
        with metrics.lock:
            return bool(getattr(metrics, "hud_enabled", True))

    def get_selected_preview_client_id(self) -> int | None:
        changed = False
        with self.client_lock:
            selected, changed = self._ensure_preview_client_selected_locked()
        if changed:
            self._clear_preview_cache()
        return None if selected is None else int(selected)

    def get_selected_preview_client_slot(self) -> int | None:
        selected = self.get_selected_preview_client_id()
        if selected is None:
            return None
        with self.client_lock:
            cs = self.client_states.get(int(selected))
            if not isinstance(cs, dict):
                return None
            try:
                return max(0, int(cs.get("client_slot", selected)))
            except Exception:
                return int(selected)

    def set_preview_client(self, cid: int | None) -> tuple[bool, int | None]:
        changed = False
        selected = None
        with self.client_lock:
            if cid is None:
                selected = self._pick_default_preview_client_locked()
            else:
                cs = self.client_states.get(int(cid))
                if not isinstance(cs, dict) or not bool(cs.get("preview_capable", False)):
                    return False, self.preview_cid
                selected = int(cid)
            changed = self.preview_cid != selected
            self.preview_cid = selected
        if changed:
            self._clear_preview_cache()
        return True, selected

    def get_client_rows(self) -> list[dict]:
        changed = False
        with self.client_lock:
            selected, changed = self._ensure_preview_client_selected_locked()
            rows = []
            for cid, cs in self.client_states.items():
                if bool(cs.get("gameplay_seen", False)):
                    try:
                        lives = max(0, int(cs.get("num_lasers", 0) or 0))
                    except Exception:
                        lives = 0
                    try:
                        level = max(0, int(cs.get("level_number", 0) or 0))
                    except Exception:
                        level = 0
                    try:
                        score = max(0, int(cs.get("game_score", 0) or 0))
                    except Exception:
                        score = 0
                else:
                    lives = 0
                    level = 0
                    score = 0
                rows.append({
                    "client_id": int(cid),
                    "client_slot": int(cs.get("client_slot", cid)),
                    "duration_seconds": float(max(0, int(cs.get("game_frames", 0) or 0))) / 60.0,
                    "lives": lives,
                    "level": level,
                    "score": score,
                    "selected_preview": (selected is not None and int(selected) == int(cid)),
                    "preview_capable": bool(cs.get("preview_capable", False)),
                })
        if changed:
            self._clear_preview_cache()
        rows.sort(key=lambda row: int(row.get("client_id", 0)))
        return rows

    def _cache_client_preview(self, cid: int, frame) -> None:
        if not self._is_preview_client(cid):
            return
        pixels = getattr(frame, "preview_pixels", None)
        width = int(getattr(frame, "preview_width", 0) or 0)
        height = int(getattr(frame, "preview_height", 0) or 0)
        fmt = int(getattr(frame, "preview_format", 0) or 0)
        enc_fmt = int(getattr(frame, "preview_encoded_format", 0) or 0)
        enc_bytes = int(getattr(frame, "preview_encoded_bytes", 0) or 0)
        raw_bytes = int(getattr(frame, "preview_raw_bytes", 0) or 0)
        if not pixels or width <= 0 or height <= 0:
            return
        if fmt != 1:  # 1 = RGB565BE packed 16-bit pixels
            return
        expected_len = width * height * 2
        if expected_len <= 0 or len(pixels) != expected_len:
            return
        with metrics.lock:
            prev_seq = int(getattr(metrics, "game_preview_seq", 0) or 0)
            prev_ts = float(getattr(metrics, "game_preview_updated_ts", 0.0) or 0.0)
            now_ts = time.time()
            next_seq = prev_seq + 1
            if prev_ts > 0.0 and next_seq > prev_seq:
                dt = max(1e-6, now_ts - prev_ts)
                dseq = max(1, next_seq - prev_seq)
                metrics.game_preview_fps = float(dseq) / dt
            metrics.game_preview_client_id = int(cid)
            metrics.game_preview_seq = next_seq
            metrics.game_preview_width = width
            metrics.game_preview_height = height
            metrics.game_preview_format = "rgb565be"
            metrics.game_preview_data = bytes(pixels)
            metrics.game_preview_updated_ts = now_ts
            metrics.game_preview_source_format = {1: "raw", 2: "lzss", 3: "rle"}.get(enc_fmt, "unknown")
            metrics.game_preview_encoded_bytes = int(enc_bytes)
            metrics.game_preview_raw_bytes = int(raw_bytes if raw_bytes > 0 else expected_len)
            if enc_bytes > 0 and raw_bytes > 0:
                metrics.game_preview_compression_ratio = float(raw_bytes) / float(enc_bytes)
            else:
                metrics.game_preview_compression_ratio = 1.0

    @staticmethod
    def _peek_preview_len(payload: bytes) -> int:
        """Read preview_len tail field without decoding payload preview blob."""
        try:
            if not payload or len(payload) < 2:
                return 0
            n = struct.unpack(">H", payload[:2])[0]
            base_len = struct.calcsize(">HddBIBBIBB") + (int(n) * 4)
            if len(payload) < (base_len + 4):
                return 0
            return int(struct.unpack(">I", payload[base_len:base_len + 4])[0] or 0)
        except Exception:
            return 0

    def _init_client(self, cid):
        n = max(1, int(getattr(RL_CONFIG, "n_step", 1)))
        gamma = float(getattr(RL_CONFIG, "gamma", 0.99))
        nstep = NStepReplayBuffer(n_step=n, gamma=gamma) if n > 1 else None
        move_bins = max(1, int(getattr(RL_CONFIG, "num_move_actions", 8)))
        fire_bins = max(1, int(getattr(RL_CONFIG, "num_fire_actions", 8)))
        pair_bins = move_bins * fire_bins
        with self.client_lock:
            self.client_states[cid] = {
                "frames": 0, "last_time": time.time(), "fps": 0.0,
                "level_number": 0, "last_state": None, "last_action": None,
                "last_player_alive": False,
                "player_alive": False,
                "game_score": 0,
                "num_lasers": 0,
                "alive_streak": 0,
                "dead_streak": 0,
                "plausible_start_streak": 0,
                "gameplay_seen": False,
                "start_pulse_window": 0,
                "last_action_source": None, "prev_action_source": None,
                "act_total": 0, "act_eps": 0, "act_xprt": 0, "act_last_diag_total": 0,
                "act_same": 0, "act_diff": 0,
                "act_move_hist": [0] * move_bins, "act_fire_hist": [0] * fire_bins,
                "act_pair_hist": [0] * pair_bins,
                "act_move_hist_eps": [0] * move_bins, "act_fire_hist_eps": [0] * fire_bins,
                "act_pair_hist_eps": [0] * pair_bins,
                "total_reward": 0.0, "ep_dqn_reward": 0.0, "ep_expert_reward": 0.0,
                "ep_subj_reward": 0.0, "ep_obj_reward": 0.0, "ep_frames": 0,
                "was_done": False, "nstep": nstep,
                "state_stack": deque(maxlen=FRAME_STACK),
                "fire_hold_dir": -1, "fire_hold_count": 0, "fire_pending_dir": -1,
                "last_alive_game_score": 0, "prev_game_final_score": 0,
                "game_frames": 0,
                "start_wave": 1,
                "episode_id": 1,
                "preview_capable": False,
                "client_slot": int(cid),
            }
            if self.publish_global_client_metrics:
                metrics.client_count = len(self.client_states)

    def _recv_exact(self, sock, nbytes: int, timeout_s: float = 0.5):
        """Read exactly nbytes from a non-blocking socket, or return None on timeout/EOF."""
        remaining = int(nbytes)
        if remaining <= 0:
            return b""
        chunks = []
        deadline = time.time() + max(0.001, float(timeout_s))

        while remaining > 0 and self.running and not self.shutdown_event.is_set():
            try:
                chunk = sock.recv(remaining)
            except (BlockingIOError, InterruptedError):
                if time.time() >= deadline:
                    return None
                ready = select.select([sock], [], [], 0.002)
                if not ready[0]:
                    continue
                continue
            if not chunk:
                return None
            chunks.append(chunk)
            remaining -= len(chunk)
        if remaining > 0:
            return None
        return b"".join(chunks)

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
                handshake_value = int(struct.unpack(">H", ping[:2])[0])
            finally:
                sock.setblocking(False)
                sock.settimeout(None)

            self._maybe_claim_preview_client(cid, handshake_value)

            while self.running and not self.shutdown_event.is_set():
                preview_enabled = self._preview_enabled_for_client(cid)
                hud_enabled = self._hud_enabled_for_client(cid)
                ready = select.select([sock], [], [], 0.002)
                if not ready[0]:
                    continue

                # Read length header
                hdr = self._recv_exact(sock, 4, timeout_s=0.25)
                if hdr is None or len(hdr) < 4:
                    raise ConnectionError("EOF")
                dlen = struct.unpack(">I", hdr)[0]
                if dlen <= 0 or dlen > _MAX_FRAME_PAYLOAD_BYTES:
                    raise ConnectionError(f"Invalid payload length: {dlen}")

                # Read payload
                data = self._recv_exact(sock, dlen, timeout_s=0.5)
                if data is None:
                    raise ConnectionError("Broken")

                if len(data) >= 2:
                    n = struct.unpack(">H", data[:2])[0]
                    if n != SERVER_CONFIG.params_count:
                        print(f"Client {cid}: param mismatch {n} != {SERVER_CONFIG.params_count}")
                        break
                else:
                    break

                preview_parse_allowed = bool(self._is_preview_client(cid) and preview_enabled)
                if not preview_parse_allowed:
                    pl = self._peek_preview_len(data)
                    if pl > 0:
                        now_t = time.time()
                        with self.client_lock:
                            cs = self.client_states.get(cid, {})
                            last = float(cs.get("last_nonzero_preview_warn_ts", 0.0) or 0.0)
                            if (now_t - last) >= 5.0:
                                cs["last_nonzero_preview_warn_ts"] = now_t

                # Hard gate preview parsing to the designated preview client only.
                # Even if another client sends preview bytes, skip decode cost.
                frame = parse_frame_data(data, parse_preview=preview_parse_allowed)
                if not frame:
                    sock.sendall(self._pack_action(-1, -1, 0, preview_enabled, hud_enabled))
                    continue
                if self._is_preview_client(cid) and preview_parse_allowed:
                    enc_fmt = int(getattr(frame, "preview_encoded_format", 0) or 0)
                    enc_bytes = int(getattr(frame, "preview_encoded_bytes", 0) or 0)
                    raw_bytes = int(getattr(frame, "preview_raw_bytes", 0) or 0)
                    if enc_fmt > 0 and enc_bytes > 0 and raw_bytes > 0:
                        now_t = time.time()
                        with self.client_lock:
                            cs = self.client_states.get(cid, {})
                            last_t = float(cs.get("last_preview_ingest_log_ts", 0.0) or 0.0)
                            last_fmt = int(cs.get("last_preview_ingest_fmt", -1))
                            if (enc_fmt != last_fmt) or ((now_t - last_t) >= 5.0):
                                cs["last_preview_ingest_log_ts"] = now_t
                                cs["last_preview_ingest_fmt"] = enc_fmt
                self._cache_client_preview(cid, frame)

                with self.client_lock:
                    if cid not in self.client_states:
                        break
                    cs = self.client_states[cid]
                    cs["frames"] += 1
                    if bool(getattr(frame, "start_pressed", False)):
                        cs["start_pulse_window"] = _START_PULSE_VALID_FRAMES
                    elif int(cs.get("start_pulse_window", 0) or 0) > 0:
                        cs["start_pulse_window"] = max(0, int(cs.get("start_pulse_window", 0) or 0) - 1)
                    cs["player_alive"] = bool(frame.player_alive)
                    if frame.player_alive:
                        cs["alive_streak"] = int(cs.get("alive_streak", 0) or 0) + 1
                        cs["dead_streak"] = 0
                    else:
                        cs["alive_streak"] = 0
                        cs["dead_streak"] = int(cs.get("dead_streak", 0) or 0) + 1
                    if int(cs.get("dead_streak", 0) or 0) >= _GAMEPLAY_RESET_DEAD_FRAMES:
                        cs["gameplay_seen"] = False
                        cs["start_pulse_window"] = 0
                        cs["plausible_start_streak"] = 0
                        cs["level_number"] = 0
                        cs["start_wave"] = 1
                        cs["game_score"] = 0
                        cs["num_lasers"] = 0
                    plausible_start = self._looks_like_real_game_start(frame)
                    if int(cs.get("start_pulse_window", 0) or 0) > 0 and frame.player_alive and plausible_start:
                        cs["plausible_start_streak"] = int(cs.get("plausible_start_streak", 0) or 0) + 1
                    elif not frame.player_alive or not plausible_start:
                        cs["plausible_start_streak"] = 0
                    if (
                        int(cs.get("start_pulse_window", 0) or 0) > 0
                        and int(cs.get("alive_streak", 0) or 0) >= 15
                        and int(cs.get("plausible_start_streak", 0) or 0) >= _GAMEPLAY_PLAUSIBLE_START_STREAK
                    ):
                        if not bool(cs.get("gameplay_seen", False)):
                            cs["gameplay_seen"] = True
                            cs["game_frames"] = 0
                            cs["start_wave"] = max(1, int(getattr(frame, "level_number", 1) or 1))
                        cs["start_pulse_window"] = 0
                    elif (
                        not bool(cs.get("gameplay_seen", False))
                        and frame.player_alive
                        and plausible_start
                        and int(cs.get("alive_streak", 0) or 0) >= 30
                        and max(0, int(getattr(frame, "game_score", 0) or 0)) > 0
                    ):
                        cs["gameplay_seen"] = True
                        cs["game_frames"] = 0
                        cs["start_wave"] = max(1, int(getattr(frame, "level_number", 1) or 1))
                        cs["start_pulse_window"] = 0
                    if bool(cs.get("gameplay_seen", False)):
                        cs["num_lasers"] = max(0, int(getattr(frame, "num_lasers", 0) or 0))
                        if frame.player_alive or frame.game_score > 0 or int(cs.get("game_score", 0) or 0) > 0:
                            cs["game_frames"] = max(0, int(cs.get("game_frames", 0) or 0)) + 1
                    # Only trust RAM-read level/score while player is alive;
                    # during attract/death the byte can hold garbage.
                    if frame.player_alive and bool(cs.get("gameplay_seen", False)):
                        cs["level_number"] = frame.level_number
                        cs["game_score"] = max(0, int(getattr(frame, "game_score", 0) or 0))
                    # Only accept a new peak while actively playing the last life.
                    if (
                        frame.player_alive
                        and bool(cs.get("gameplay_seen", False))
                        and max(0, int(getattr(frame, "num_lasers", 0) or 0)) == 0
                        and frame.game_score > self.metrics.peak_game_score
                    ):
                        self.metrics.peak_game_score = frame.game_score
                    # Track per-game score: detect new game when score drops.
                    if frame.player_alive:
                        if frame.game_score < cs.get("last_alive_game_score", 0):
                            # Score dropped → new game started; record previous game total.
                            prev_final = cs.get("last_alive_game_score", 0)
                            if prev_final > 0:
                                self.metrics.add_game_score(prev_final)
                            cs["game_frames"] = 0
                            cs["start_pulse_window"] = 0
                            cs["start_wave"] = max(1, int(getattr(frame, "level_number", 1) or 1))
                        cs["last_alive_game_score"] = frame.game_score
                    now = time.time()
                    el = now - cs["last_time"]
                    if el >= 1.0:
                        cs["fps"] = 1.0 / el
                        cs["last_time"] = now

                # Keep metrics synchronized to actual processed frames.
                # Do not batch this by connection-local counters, because
                # reconnects can reset local state and hide frame progress.
                # Count only active gameplay frames for RL schedules so
                # attract/death downtime does not decay epsilon.
                active_delta = 1 if frame.player_alive else 0
                self.metrics.update_frame_count(delta=active_delta)
                if active_delta:
                    self.metrics.update_epsilon()
                    self.metrics.update_expert_ratio()
                self._calc_avg_level()
                self.metrics.update_game_state(0, False)
                stacked_state = _push_stacked_state(cs, frame.state)

                # ── Process previous step ───────────────────────────────
                if (
                    cs.get("last_state") is not None
                    and cs.get("last_action") is not None
                    and cs.get("last_player_alive", False)
                ):
                    move_i, fire_i = cs["last_action"]
                    subj_r = float(frame.subjreward) * RL_CONFIG.subj_reward_scale
                    obj_r = float(frame.objreward) * RL_CONFIG.obj_reward_scale
                    total_unclipped_r = obj_r + subj_r
                    # Use wider clip on terminal frames so death penalty passes through
                    clip = RL_CONFIG.death_reward_clip if frame.done else RL_CONFIG.reward_clip
                    total_r = max(-clip, min(clip, total_unclipped_r))
                    eff_obj_r = obj_r
                    eff_subj_r = subj_r
                    if abs(total_unclipped_r) > 1e-9:
                        # Display reward components after the same clip that the trainer sees
                        # so Rwrd == Obj + Subj in the metrics table.
                        scale = total_r / total_unclipped_r
                        eff_obj_r = obj_r * scale
                        eff_subj_r = subj_r * scale

                    if self.agent:
                        tag = cs.get("prev_action_source", "dqn")
                        nstep = cs.get("nstep")
                        if nstep is not None:
                            joint = combine_action_indices(move_i, fire_i)
                            matured = nstep.add(cs["last_state"], joint, total_r,
                                                stacked_state, bool(frame.done),
                                                actor=tag, priority_reward=total_r,
                                                wave_number=max(1, int(cs.get("level_number", 1) or 1)),
                                                start_wave=max(1, int(cs.get("start_wave", 1) or 1)))
                            wave = max(1, cs.get("level_number", 1) or 1)
                            for s0, a, Rn, pR, sn, dn, h, act, wave_n, start_wave_n in matured:
                                move_n, fire_n = split_joint_action(a)
                                self._submit_experience(
                                    s0, (move_n, fire_n), Rn, sn, bool(dn),
                                    client_id=cid, actor=act, horizon=int(h),
                                    priority_reward=pR,
                                    wave_number=max(1, int(wave_n or wave)),
                                    start_wave=max(1, int(start_wave_n or cs.get("start_wave", 1) or 1)),
                                    episode_id=max(1, int(cs.get("episode_id", 1) or 1)),
                                )
                        else:
                            wave = max(1, cs.get("level_number", 1) or 1)
                            self._submit_experience(
                                cs["last_state"], (move_i, fire_i), total_r,
                                stacked_state, bool(frame.done), client_id=cid, actor=tag, horizon=1,
                                priority_reward=total_r,
                                wave_number=wave,
                                start_wave=max(1, int(cs.get("start_wave", 1) or 1)),
                                episode_id=max(1, int(cs.get("episode_id", 1) or 1)),
                            )

                    cs["total_reward"] += total_r
                    cs["ep_subj_reward"] = cs.get("ep_subj_reward", 0) + eff_subj_r
                    cs["ep_obj_reward"] = cs.get("ep_obj_reward", 0) + eff_obj_r
                    cs["ep_frames"] = cs.get("ep_frames", 0) + 1
                    src = cs.get("prev_action_source")
                    if src == "dqn":
                        cs["ep_dqn_reward"] += total_r
                    elif src == "expert":
                        cs["ep_expert_reward"] += total_r

                # ── Terminal ────────────────────────────────────────────
                if frame.done:
                    _reset_state_stack(cs)
                    # Reset fire hold on death
                    cs["fire_hold_dir"] = -1
                    cs["fire_hold_count"] = 0
                    cs["fire_pending_dir"] = -1
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
                        sock.sendall(self._pack_action(-1, -1, 0, preview_enabled, hud_enabled))
                    except Exception:
                        break
                    cs["last_state"] = cs["last_action"] = None
                    cs["last_player_alive"] = False
                    cs["last_action_source"] = cs["prev_action_source"] = None
                    cs["episode_id"] = max(1, int(cs.get("episode_id", 1) or 1) + 1)
                    cs["total_reward"] = cs["ep_dqn_reward"] = cs["ep_expert_reward"] = 0.0
                    cs["ep_subj_reward"] = cs["ep_obj_reward"] = 0.0
                    cs["ep_frames"] = 0
                    continue

                if cs.get("was_done"):
                    cs["was_done"] = False
                    cs["total_reward"] = cs["ep_dqn_reward"] = cs["ep_expert_reward"] = 0.0
                    cs["ep_subj_reward"] = cs["ep_obj_reward"] = 0.0
                    cs["ep_frames"] = 0

                # Do not let dead/attract frames pollute replay; actions are ignored there.
                if not frame.player_alive:
                    _reset_state_stack(cs)
                    nstep = cs.get("nstep")
                    if nstep is not None:
                        nstep.reset()
                    cs["last_state"] = cs["last_action"] = None
                    cs["last_player_alive"] = False
                    cs["last_action_source"] = cs["prev_action_source"] = None
                    # Reset fire hold when player is dead
                    cs["fire_hold_dir"] = -1
                    cs["fire_hold_count"] = 0
                    cs["fire_pending_dir"] = -1
                    try:
                        sock.sendall(self._pack_action(-1, -1, 0, preview_enabled, hud_enabled))
                    except Exception:
                        break
                    continue

                # ── Choose action ───────────────────────────────────────
                self.metrics.increment_total_controls()
                move_idx = fire_idx = 0
                action_source = "none"
                is_epsilon = False
                epsilon = float(self.metrics.get_effective_epsilon())
                if not np.isfinite(epsilon):
                    epsilon = 0.0
                epsilon = max(0.0, min(1.0, epsilon))
                # Fire updates only on cadence boundaries; between boundaries
                # we keep fire fixed so decisions are not silently overwritten.
                fire_update_open = int(cs.get("fire_hold_count", 0)) <= 0
                locked_fire = None
                if not fire_update_open:
                    held = int(cs.get("fire_hold_dir", -1))
                    if held >= 0:
                        max_fire = max(1, int(getattr(RL_CONFIG, "num_fire_actions", 8)))
                        locked_fire = max(0, min(max_fire - 1, held))
                    else:
                        locked_fire = 0

                if _FORCE_RANDOM_ACTIONS:
                    move_idx = random.randrange(max(1, int(getattr(RL_CONFIG, "num_move_actions", 8))))
                    if fire_update_open:
                        fire_idx = random.randrange(max(1, int(getattr(RL_CONFIG, "num_fire_actions", 8))))
                    else:
                        fire_idx = int(locked_fire)
                    is_epsilon = True
                    action_source = "forced_random"
                elif self.agent:
                    expert_ratio = float(self.metrics.get_expert_ratio())
                    if not np.isfinite(expert_ratio):
                        expert_ratio = 0.0
                    expert_ratio = max(0.0, min(1.0, expert_ratio))
                    use_expert = random.random() < expert_ratio
                    if use_expert:
                        move_idx, fire_idx = get_expert_action(
                            stacked_state,
                            locked_fire=locked_fire,
                            wave_number=int(cs.get("level_number", 0) or 0),
                        )
                        is_epsilon = False
                        action_source = "expert"
                    else:
                        t0 = time.perf_counter()
                        if self.inference_batcher is not None:
                            move_idx, fire_idx, is_epsilon = self.inference_batcher.infer(
                                stacked_state, epsilon, locked_fire=locked_fire, route_key=cid
                            )
                        else:
                            move_idx, fire_idx, is_epsilon = self.agent.act(
                                stacked_state, epsilon, locked_fire=locked_fire
                            )
                        self.metrics.add_inference_time(time.perf_counter() - t0)
                        action_source = "dqn"

                # When the wave is effectively over (no humans, <=2 cleanup targets),
                # correct only the fire axis toward an obvious aligned kill shot.
                if action_source == "dqn" and not is_epsilon:
                    cleanup_fire = get_cleanup_fire_override(stacked_state, locked_fire=locked_fire)
                    if cleanup_fire is not None:
                        fire_idx = int(cleanup_fire)

                move_hist = cs.get("act_move_hist", [])
                fire_hist = cs.get("act_fire_hist", [])
                pair_hist = cs.get("act_pair_hist", [])
                move_hist_eps = cs.get("act_move_hist_eps", [])
                fire_hist_eps = cs.get("act_fire_hist_eps", [])
                pair_hist_eps = cs.get("act_pair_hist_eps", [])
                move_bins = len(move_hist)
                fire_bins = len(fire_hist)

                cs["act_total"] = int(cs.get("act_total", 0)) + 1
                if 0 <= int(move_idx) < move_bins:
                    move_hist[int(move_idx)] += 1
                if 0 <= int(fire_idx) < fire_bins:
                    fire_hist[int(fire_idx)] += 1
                if 0 <= int(move_idx) < move_bins and 0 <= int(fire_idx) < fire_bins:
                    pair_idx = int(move_idx) * fire_bins + int(fire_idx)
                    if 0 <= pair_idx < len(pair_hist):
                        pair_hist[pair_idx] += 1
                if int(move_idx) == int(fire_idx):
                    cs["act_same"] = int(cs.get("act_same", 0)) + 1
                else:
                    cs["act_diff"] = int(cs.get("act_diff", 0)) + 1
                if is_epsilon:
                    cs["act_eps"] = int(cs.get("act_eps", 0)) + 1
                    if 0 <= int(move_idx) < move_bins:
                        move_hist_eps[int(move_idx)] += 1
                    if 0 <= int(fire_idx) < fire_bins:
                        fire_hist_eps[int(fire_idx)] += 1
                    if 0 <= int(move_idx) < move_bins and 0 <= int(fire_idx) < fire_bins:
                        pair_idx = int(move_idx) * fire_bins + int(fire_idx)
                        if 0 <= pair_idx < len(pair_hist_eps):
                            pair_hist_eps[pair_idx] += 1
                if action_source == "expert":
                    cs["act_xprt"] = int(cs.get("act_xprt", 0)) + 1
                if _ACTION_DIAGNOSTICS:
                    last_diag = int(cs.get("act_last_diag_total", 0))
                    total = int(cs.get("act_total", 0))
                    if (total - last_diag) >= _ACTION_DIAG_INTERVAL:
                        def _top_pairs(hist, k=4):
                            tops = sorted(enumerate(hist), key=lambda x: x[1], reverse=True)
                            out = []
                            for idx, cnt in tops:
                                if cnt <= 0:
                                    break
                                if fire_bins > 0:
                                    out.append(f"{idx // fire_bins}->{idx % fire_bins}:{cnt}")
                                else:
                                    out.append(f"{idx}:{cnt}")
                                if len(out) >= k:
                                    break
                            return "[" + ", ".join(out) + "]"

                        eps_used = int(cs.get("act_eps", 0))
                        eps_rate = eps_used / max(1, total)
                        xprt_used = int(cs.get("act_xprt", 0))
                        xprt_rate = xprt_used / max(1, total)
                        same = int(cs.get("act_same", 0))
                        diff = int(cs.get("act_diff", 0))
                        diff_rate = diff / max(1, (same + diff))
                        print(
                            f"[actdiag] client={cid} eps_eff={float(epsilon):.3f} xprt_eff={float(expert_ratio):.3f} "
                            f"eps_used={eps_used}/{total} ({eps_rate:.1%}) "
                            f"xprt_used={xprt_used}/{total} ({xprt_rate:.1%}) "
                            f"same={same} diff={diff} ({diff_rate:.1%} diff) "
                            f"move_all={cs.get('act_move_hist')} "
                            f"fire_all={cs.get('act_fire_hist')} "
                            f"top_pairs={_top_pairs(cs.get('act_pair_hist', []))} "
                            f"top_pairs_eps={_top_pairs(cs.get('act_pair_hist_eps', []))}"
                        )
                        cs["act_last_diag_total"] = total

                # Apply fire hold and store the EFFECTIVE fire direction.
                # Since fire selection is gated to hold boundaries above, we
                # avoid dropping 3/4 fire decisions while still satisfying
                # LSPROC's 3-stable-frame requirement.
                effective_fire = _apply_fire_hold(cs, fire_idx)

                cs["last_state"] = stacked_state
                cs["last_action"] = (move_idx, effective_fire)
                cs["last_player_alive"] = bool(frame.player_alive)
                cs["prev_action_source"] = action_source
                cs["last_action_source"] = action_source

                move_cmd, fire_cmd = encode_action_to_game(move_idx, effective_fire)
                # Action source byte (low nibble): 0=none, 1=dqn, 2=epsilon, 3=expert, 4=forced_random.
                # Bit 0x40 flags the Lua client to send game-preview snapshots.
                if action_source == "expert":
                    source_byte = 3
                elif is_epsilon:
                    source_byte = 2
                elif action_source == "dqn":
                    source_byte = 1
                elif action_source == "forced_random":
                    source_byte = 4
                else:
                    source_byte = 0
                try:
                    sock.sendall(self._pack_action(move_cmd, fire_cmd, source_byte, preview_enabled, hud_enabled))
                except Exception:
                    break

        except Exception as e:
            msg = str(e or "").strip()
            is_expected_disconnect = (
                isinstance(e, ConnectionError) and msg in {"EOF", "Broken", "No handshake"}
            ) or isinstance(e, (BrokenPipeError, ConnectionResetError, TimeoutError))
            if not is_expected_disconnect:
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
            changed = False
            with self.client_lock:
                was_selected = (self.preview_cid is not None and int(self.preview_cid) == int(cid))
                self.client_states.pop(cid, None)
                self.clients[cid] = None
                if self.publish_global_client_metrics:
                    metrics.client_count = sum(1 for v in self.clients.values() if v is not None)
                if was_selected:
                    self.preview_cid = None
                    changed = True
                if self.preview_cid is None:
                    _selected, selected_changed = self._ensure_preview_client_selected_locked()
                    changed = changed or selected_changed
            if self.async_buffer is not None:
                self.async_buffer.remove_client(cid)
            if changed:
                self._clear_preview_cache()
            threading.Timer(1.0, self._cleanup).start()

    def _cleanup(self):
        with self.client_lock:
            dead = [k for k, v in self.clients.items() if v is None]
            for k in dead:
                del self.clients[k]
            if self.publish_global_client_metrics:
                metrics.client_count = len(self.clients)

    def _calc_avg_level(self):
        if not self.publish_global_client_metrics:
            return
        try:
            with self.client_lock:
                lvls = [
                    s.get("level_number", 0)
                    for s in self.client_states.values()
                    if bool(s.get("gameplay_seen", False)) and 0 < s.get("level_number", 0) <= 81
                ]
                metrics.average_level = sum(lvls) / len(lvls) if lvls else 0
                if lvls:
                    metrics.peak_level_verified = True
                    best_level = max(int(lv) for lv in lvls)
                    if best_level > metrics.peak_level:
                        metrics.peak_level = best_level
                elif not bool(getattr(metrics, "peak_level_verified", False)):
                    metrics.peak_level = 0
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
            print("Stopping inference dispatcher...")
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
