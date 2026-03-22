#!/usr/bin/env python3
# ==================================================================================================================
# ||  ROBOTRON AI v2 • SHARDED SOCKET SERVERS                                                                    ||
# ||  Master process keeps training, metrics, preview, and dashboard; worker processes serve additional MAME      ||
# ||  clients on their own ports and stream experience / metrics back to the master.                              ||
# ==================================================================================================================
"""True multi-process socket-server sharding for Robotron."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
import traceback
from typing import Any, Optional

import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.append(_SCRIPT_DIR)

try:
    from aimodel import RainbowNet, split_joint_action, NUM_MOVE, NUM_FIRE, _cuda_device
    from config import RL_CONFIG, SERVER_CONFIG, metrics, game_settings
    from metrics_display import (
        add_episode_to_dqn100k_window,
        add_episode_to_dqn1m_window,
        add_episode_to_dqn5m_window,
        add_episode_to_total_windows,
        add_episode_to_eplen_window,
    )
    from socket_server import SocketServer
except ImportError:
    try:
        from .aimodel import RainbowNet, split_joint_action, NUM_MOVE, NUM_FIRE, _cuda_device
        from .config import RL_CONFIG, SERVER_CONFIG, metrics, game_settings
        from .metrics_display import (
            add_episode_to_dqn100k_window,
            add_episode_to_dqn1m_window,
            add_episode_to_dqn5m_window,
            add_episode_to_total_windows,
            add_episode_to_eplen_window,
        )
        from .socket_server import SocketServer
    except ImportError:
        from Scripts.aimodel import RainbowNet, split_joint_action, NUM_MOVE, NUM_FIRE, _cuda_device
        from Scripts.config import RL_CONFIG, SERVER_CONFIG, metrics, game_settings
        from Scripts.metrics_display import (
            add_episode_to_dqn100k_window,
            add_episode_to_dqn1m_window,
            add_episode_to_dqn5m_window,
            add_episode_to_total_windows,
            add_episode_to_eplen_window,
        )
        from Scripts.socket_server import SocketServer


def _infer_device() -> torch.device:
    if bool(getattr(RL_CONFIG, "inference_on_cpu", False)):
        return torch.device("cpu")
    if torch.cuda.is_available():
        return _cuda_device(getattr(RL_CONFIG, "inference_cuda_device_index", 0))
    return torch.device("cpu")


class _NoopReplayMemory:
    def boost_priorities(self, indices, boost):
        return None


class WorkerPolicyAgent:
    """Inference-only agent used inside shard worker processes."""

    allow_process_inference_pool = False
    allow_async_inference_batcher = True
    allow_async_replay_buffer = False
    publish_global_client_metrics = False

    def __init__(self, state_size: int, initial_state_dict: dict[str, torch.Tensor], experience_q, worker_id: int = 0):
        self.state_size = int(state_size)
        self.worker_id = int(worker_id)
        self.device = _infer_device()
        if self.device.type == "cuda" and self.device.index is not None:
            torch.cuda.set_device(self.device.index)
        self.net = RainbowNet(self.state_size).to(self.device)
        self.net.load_state_dict(initial_state_dict, strict=False)
        self.net.eval()
        self.memory = _NoopReplayMemory()
        self.experience_q = experience_q
        self.factored_greedy_action = bool(getattr(RL_CONFIG, "factored_greedy_action", False))
        self._local_step_idx = 0
        self._experience_batch: list[tuple[Any, ...]] = []
        self._experience_batch_max = 64
        self._experience_flush_s = 0.010
        self._last_experience_flush = time.perf_counter()

    def update_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.net.load_state_dict(state_dict, strict=False)
        self.net.eval()

    def _flush_experience_batch(self, force: bool = False) -> None:
        if not self._experience_batch:
            return
        if not force:
            now = time.perf_counter()
            if len(self._experience_batch) < self._experience_batch_max and (now - self._last_experience_flush) < self._experience_flush_s:
                return
        payload = list(self._experience_batch)
        self._experience_batch.clear()
        self._last_experience_flush = time.perf_counter()
        try:
            self.experience_q.put(("exp_batch", payload), timeout=0.01)
        except Exception:
            pass

    @staticmethod
    def _greedy_axes_from_q(q_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q_joint = q_values.view(-1, NUM_MOVE, NUM_FIRE)
        move_scores = q_joint.max(dim=2).values
        fire_scores = q_joint.max(dim=1).values
        return move_scores.argmax(dim=1), fire_scores.argmax(dim=1)

    def act_batch(
        self,
        states: list[np.ndarray],
        epsilons: list[float],
        locked_fires: Optional[list[Optional[int]]] = None,
    ) -> list[tuple[int, int, bool]]:
        """Batched inference path for shard-local AsyncInferenceBatcher."""
        n = min(len(states), len(epsilons))
        if n <= 0:
            return []
        if locked_fires is None:
            lock_list: list[Optional[int]] = [None] * n
        else:
            lock_list = list(locked_fires[:n])
            if len(lock_list) < n:
                lock_list.extend([None] * (n - len(lock_list)))

        rand_moves = [False] * n
        rand_fires = [False] * n
        rnd_move_vals = [0] * n
        rnd_fire_vals = [0] * n
        fire_fixed = [False] * n
        fire_fixed_vals = [0] * n
        greedy_idx: list[int] = []
        greedy_states: list[np.ndarray] = []

        for i in range(n):
            eps = max(0.0, min(1.0, float(epsilons[i])))
            lf = lock_list[i]
            if lf is not None:
                lf_i = int(lf)
                if lf_i >= 0:
                    fire_fixed[i] = True
                    fire_fixed_vals[i] = max(0, min(NUM_FIRE - 1, lf_i))

            rand_moves[i] = np.random.random() < eps
            if rand_moves[i]:
                rnd_move_vals[i] = int(np.random.randint(NUM_MOVE))

            if fire_fixed[i]:
                rand_fires[i] = False
                rnd_fire_vals[i] = fire_fixed_vals[i]
                needs_greedy = not rand_moves[i]
            else:
                rand_fires[i] = np.random.random() < eps
                if rand_fires[i]:
                    rnd_fire_vals[i] = int(np.random.randint(NUM_FIRE))
                needs_greedy = not (rand_moves[i] and rand_fires[i])

            if needs_greedy:
                greedy_idx.append(i)
                greedy_states.append(states[i])

        greedy_actions: dict[int, tuple[int, int]] = {}
        if greedy_idx:
            batch_np = np.asarray(greedy_states, dtype=np.float32)
            st = torch.from_numpy(batch_np).to(self.device)
            with torch.inference_mode():
                q = self.net.q_values(st)
            gm_t, gf_t = self._greedy_axes_from_q(q)
            gm = gm_t.detach().cpu().tolist()
            gf = gf_t.detach().cpu().tolist()
            if self.factored_greedy_action:
                for pos, m, f in zip(greedy_idx, gm, gf):
                    greedy_actions[pos] = (int(m), int(f))
            else:
                joints = q.argmax(dim=1).detach().cpu().tolist()
                for k, (pos, joint) in enumerate(zip(greedy_idx, joints)):
                    if fire_fixed[pos]:
                        greedy_actions[pos] = (int(gm[k]), int(gf[k]))
                    else:
                        greedy_actions[pos] = split_joint_action(int(joint))

        actions: list[tuple[int, int, bool]] = []
        for i in range(n):
            g_move, g_fire = greedy_actions.get(i, (0, 0))
            move_idx = rnd_move_vals[i] if rand_moves[i] else g_move
            if fire_fixed[i]:
                fire_idx = fire_fixed_vals[i]
            else:
                fire_idx = rnd_fire_vals[i] if rand_fires[i] else g_fire
            actions.append((int(move_idx), int(fire_idx), bool(rand_moves[i] or rand_fires[i])))
        return actions

    def act(self, state: np.ndarray, epsilon: float, locked_fire: Optional[int] = None):
        lock_fire = None
        if locked_fire is not None:
            lf = int(locked_fire)
            if lf >= 0:
                lock_fire = max(0, min(NUM_FIRE - 1, lf))

        epsilon = max(0.0, min(1.0, float(epsilon)))
        rand_move = np.random.random() < epsilon
        if lock_fire is not None:
            if rand_move:
                return int(np.random.randint(NUM_MOVE)), lock_fire, True
            st = torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                q = self.net.q_values(st)
            q_joint = q.view(-1, NUM_MOVE, NUM_FIRE)
            move_scores = q_joint.max(dim=2).values
            greedy_move = int(move_scores.argmax(dim=1)[0].item())
            return greedy_move, lock_fire, False

        rand_fire = np.random.random() < epsilon
        if rand_move and rand_fire:
            return int(np.random.randint(NUM_MOVE)), int(np.random.randint(NUM_FIRE)), True

        st = torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            q = self.net.q_values(st)
        if self.factored_greedy_action:
            greedy_move_t, greedy_fire_t = self._greedy_axes_from_q(q)
            greedy_move = int(greedy_move_t[0].item())
            greedy_fire = int(greedy_fire_t[0].item())
        else:
            joint = int(q.argmax(dim=1).item())
            greedy_move, greedy_fire = split_joint_action(joint)

        move_idx = int(np.random.randint(NUM_MOVE)) if rand_move else greedy_move
        fire_idx = int(np.random.randint(NUM_FIRE)) if rand_fire else greedy_fire
        return move_idx, fire_idx, bool(rand_move or rand_fire)

    def step(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        actor="dqn",
        horizon=1,
        priority_reward=None,
        wave_number=1,
        start_wave=1,
        client_id=None,
        episode_id=0,
    ):
        payload = (
            np.asarray(state, dtype=np.float32),
            tuple(action) if isinstance(action, (tuple, list)) else int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            bool(done),
            str(actor),
            int(horizon),
            None if priority_reward is None else float(priority_reward),
            int(wave_number),
            int(start_wave),
            int(self.worker_id),
            int(client_id if client_id is not None else -1),
            int(episode_id or 0),
        )
        self._experience_batch.append(payload)
        self._flush_experience_batch(force=bool(done) or len(self._experience_batch) >= self._experience_batch_max)
        self._local_step_idx += 1
        return self._local_step_idx

    def stop(self):
        self._flush_experience_batch(force=True)
        return None


class WorkerMetricsProxy:
    """Metrics facade used by shard workers to report to the master process."""

    def __init__(self, worker_id: int, event_q, control_state: dict[str, Any]):
        self.worker_id = int(worker_id)
        self.event_q = event_q
        self.control_state = control_state
        self._peak_game_score = 0
        self.episodes_this_run = 0
        self._lock = threading.Lock()
        self._frame_delta = 0
        self._total_controls = 0
        self._inference_time = 0.0
        self._inference_requests = 0

    def _emit(self, kind: str, *payload) -> None:
        try:
            self.event_q.put((kind, self.worker_id, *payload), timeout=0.01)
        except Exception:
            pass

    def update_frame_count(self, delta: int = 1):
        d = max(0, int(delta))
        if d > 0:
            with self._lock:
                self._frame_delta += d

    def add_episode_reward(self, total, dqn, expert, subj=None, obj=None, length=0):
        self.episodes_this_run += 1
        self._emit(
            "episode_reward",
            float(total),
            float(dqn),
            float(expert),
            None if subj is None else float(subj),
            None if obj is None else float(obj),
            int(length),
        )

    def update_epsilon(self):
        return float(self.control_state.get("effective_epsilon", 0.0) or 0.0)

    def update_expert_ratio(self):
        return float(self.control_state.get("expert_ratio", 0.0) or 0.0)

    def get_effective_epsilon(self):
        return float(self.control_state.get("effective_epsilon", 0.0) or 0.0)

    def get_expert_ratio(self):
        return float(self.control_state.get("expert_ratio", 0.0) or 0.0)

    def increment_total_controls(self):
        with self._lock:
            self._total_controls += 1

    def add_inference_time(self, t):
        with self._lock:
            self._inference_time += float(t)
            self._inference_requests += 1

    def update_game_state(self, e, o):
        return None

    @property
    def peak_game_score(self):
        return int(self._peak_game_score)

    @peak_game_score.setter
    def peak_game_score(self, v):
        score = max(0, int(v))
        if score > self._peak_game_score:
            self._peak_game_score = score
            self._emit("peak_game_score", score)

    def add_game_score(self, score):
        self._emit("game_score", max(0, int(score)))

    def flush_snapshot(self) -> dict[str, Any]:
        with self._lock:
            snap = {
                "frame_delta": int(self._frame_delta),
                "total_controls": int(self._total_controls),
                "inference_time": float(self._inference_time),
                "inference_requests": int(self._inference_requests),
            }
            self._frame_delta = 0
            self._total_controls = 0
            self._inference_time = 0.0
            self._inference_requests = 0
            return snap


def _worker_control_loop(server: SocketServer, agent: WorkerPolicyAgent, control_q, control_state: dict[str, Any]):
    while not server.shutdown_event.is_set():
        if not server.running:
            time.sleep(0.05)
            continue
        try:
            item = control_q.get(timeout=0.1)
        except queue.Empty:
            continue
        kind = item[0]
        if kind == "shutdown":
            try:
                server.stop()
            except Exception:
                pass
            break
        if kind == "weights":
            _, state_dict = item
            try:
                agent.update_weights(state_dict)
            except Exception:
                traceback.print_exc()
        elif kind == "control":
            _, snapshot = item
            control_state.clear()
            control_state.update(dict(snapshot))
            try:
                game_settings.start_advanced = bool(control_state.get("start_advanced", False))
                game_settings.start_level_min = int(control_state.get("start_level_min", 1) or 1)
            except Exception:
                pass


def _worker_report_loop(server: SocketServer, worker_id: int, event_q, metrics_proxy: WorkerMetricsProxy):
    while not server.shutdown_event.is_set():
        if not server.running:
            time.sleep(0.05)
            continue
        try:
            rows = list(server.get_client_rows() or [])
            snapshot = metrics_proxy.flush_snapshot()
            event_q.put(("worker_snapshot", int(worker_id), snapshot, rows), timeout=0.05)
        except Exception:
            pass
        time.sleep(0.25)
    try:
        snapshot = metrics_proxy.flush_snapshot()
        event_q.put(("worker_snapshot", int(worker_id), snapshot, []), timeout=0.05)
    except Exception:
        pass


def _worker_process_main(
    worker_id: int,
    host: str,
    port: int,
    state_size: int,
    initial_state_dict: dict[str, torch.Tensor],
    event_q,
    experience_q,
    control_q,
):
    try:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        torch.set_num_threads(1)

        control_state = {
            "effective_epsilon": float(RL_CONFIG.epsilon_end),
            "expert_ratio": float(RL_CONFIG.expert_ratio_end),
            "start_advanced": False,
            "start_level_min": 1,
        }
        try:
            game_settings.start_advanced = False
            game_settings.start_level_min = 1
        except Exception:
            pass
        agent = WorkerPolicyAgent(
            state_size=state_size,
            initial_state_dict=initial_state_dict,
            experience_q=experience_q,
            worker_id=worker_id,
        )
        metrics_proxy = WorkerMetricsProxy(worker_id=worker_id, event_q=event_q, control_state=control_state)
        server = SocketServer(host, int(port), agent, metrics_proxy)

        threading.Thread(
            target=_worker_control_loop,
            args=(server, agent, control_q, control_state),
            daemon=True,
            name=f"ShardCtl{worker_id}",
        ).start()
        threading.Thread(
            target=_worker_report_loop,
            args=(server, worker_id, event_q, metrics_proxy),
            daemon=True,
            name=f"ShardRpt{worker_id}",
        ).start()
        server.start()
    except Exception:
        traceback.print_exc()
    finally:
        try:
            agent.stop()
        except Exception:
            pass


class ShardedServerCoordinator:
    """Master-side orchestrator for preview/master server + worker shard servers."""

    def __init__(self, master_server: SocketServer, master_agent, metrics_obj):
        self.master_server = master_server
        self.master_server.publish_global_client_metrics = False
        self.agent = master_agent
        self.metrics = metrics_obj
        self.ctx = mp.get_context("spawn")
        self.event_q = self.ctx.Queue(maxsize=20000)
        self.experience_q = self.ctx.Queue(maxsize=20000)
        self.control_queues = []
        self.worker_processes = []
        self.worker_rows: dict[int, list[dict[str, Any]]] = {}
        self._episode_indices: dict[tuple[int, int, int], list[int]] = {}
        self._episode_stats: dict[tuple[int, int, int], dict[str, Any]] = {}
        self.running = False
        self.master_thread: Optional[threading.Thread] = None
        self._event_thread: Optional[threading.Thread] = None
        self._experience_thread: Optional[threading.Thread] = None
        self._sync_thread: Optional[threading.Thread] = None
        self._sync_marker = -1

        self.preview_slot = int(getattr(SERVER_CONFIG, "shard_preview_slot", 0) or 0)
        self.worker_count = max(0, int(getattr(SERVER_CONFIG, "shard_workers", 0) or 0))
        base_port = int(getattr(SERVER_CONFIG, "shard_worker_port_base", 0) or 0)
        if base_port <= 0:
            base_port = int(getattr(SERVER_CONFIG, "port", 9998) or 9998) + 1
        self.worker_ports = [base_port + idx for idx in range(self.worker_count)]

        robotron_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.shard_env_path = os.path.join(robotron_root, "logs", "server_shards.env")

    def _normalize_public_host(self) -> str:
        explicit = os.getenv("ROBOTRON_SOCKET_PUBLIC_HOST", "").strip()
        if explicit:
            return explicit
        bind_host = str(getattr(SERVER_CONFIG, "host", "127.0.0.1") or "127.0.0.1")
        if bind_host in {"0.0.0.0", "::", "[::]"}:
            return "127.0.0.1"
        return bind_host

    def _write_shard_env(self) -> None:
        os.makedirs(os.path.dirname(self.shard_env_path), exist_ok=True)
        lines = [
            f"ROBOTRON_SHARD_ENABLED={1 if self.worker_ports else 0}",
            f"ROBOTRON_SOCKET_HOST={self._normalize_public_host()}",
            f"ROBOTRON_MASTER_PORT={int(getattr(SERVER_CONFIG, 'port', 9998) or 9998)}",
            f"ROBOTRON_WORKER_PORTS={','.join(str(p) for p in self.worker_ports)}",
            f"ROBOTRON_PREVIEW_SLOT={self.preview_slot}",
        ]
        tmp_path = self.shard_env_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        os.replace(tmp_path, self.shard_env_path)

    def _remove_shard_env(self) -> None:
        try:
            if os.path.exists(self.shard_env_path):
                os.remove(self.shard_env_path)
        except Exception:
            pass

    def _control_snapshot(self) -> dict[str, Any]:
        return {
            "effective_epsilon": float(self.metrics.get_effective_epsilon()),
            "expert_ratio": float(self.metrics.get_expert_ratio()),
            "start_advanced": bool(game_settings.start_advanced),
            "start_level_min": int(game_settings.start_level_min),
        }

    def _export_inference_state_dict(self) -> dict[str, torch.Tensor]:
        src_net = self.agent.infer_net if bool(getattr(self.agent, "use_separate_inference", False)) else self.agent.online_net
        return {k: v.detach().cpu() for k, v in src_net.state_dict().items()}

    def _sync_source_marker(self) -> int:
        if bool(getattr(self.agent, "use_separate_inference", False)):
            return int(getattr(self.agent, "last_inference_sync", 0) or 0)
        return int(getattr(self.agent, "training_steps", 0) or 0)

    def _remap_row(self, row: dict[str, Any]) -> dict[str, Any]:
        out = dict(row)
        slot = int(out.get("client_slot", out.get("client_id", -1)))
        out["client_id"] = slot
        out["client_slot"] = slot
        return out

    def _track_episode_step(
        self,
        worker_id: int,
        client_id: int,
        episode_id: int,
        replay_idx,
        reward: float,
        done: bool,
        actor: str,
        wave_number: int,
        start_wave: int,
    ) -> None:
        key = (int(worker_id), int(client_id), int(episode_id or 0))
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
                self.agent.consider_self_imitation(
                    idx_list,
                    dqn_reward=float(final_stats["dqn_reward"]),
                    total_reward=float(final_stats["total_reward"]),
                    length=int(final_stats["length"]),
                    max_wave=int(final_stats["max_wave"]),
                    start_wave=int(final_stats["start_wave"]),
                )

    def _combined_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        try:
            rows.extend(self._remap_row(r) for r in (self.master_server.get_client_rows() or []))
        except Exception:
            pass
        for shard_rows in list(self.worker_rows.values()):
            rows.extend(dict(r) for r in shard_rows)
        rows.sort(key=lambda row: int(row.get("client_id", 0)))
        return rows

    def _refresh_aggregate_client_metrics(self) -> None:
        rows = self._combined_rows()
        levels = [int(r.get("level", 0) or 0) for r in rows if 0 < int(r.get("level", 0) or 0) <= 81]
        with self.metrics.lock:
            self.metrics.client_count = len(rows)
            self.metrics.average_level = (float(sum(levels)) / len(levels)) if levels else 0.0
            if levels:
                self.metrics.peak_level_verified = True
                best_level = max(levels)
                if best_level > self.metrics.peak_level:
                    self.metrics.peak_level = best_level
            elif not bool(getattr(self.metrics, "peak_level_verified", False)):
                self.metrics.peak_level = 0

    def _event_loop(self) -> None:
        while self.running:
            try:
                item = self.event_q.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                kind = item[0]
                worker_id = int(item[1])
                if kind == "worker_snapshot":
                    snapshot = dict(item[2] or {})
                    rows = [self._remap_row(r) for r in list(item[3] or [])]
                    for row in rows:
                        row["selected_preview"] = False
                    self.worker_rows[worker_id] = rows
                    delta = max(0, int(snapshot.get("frame_delta", 0) or 0))
                    if delta > 0:
                        self.metrics.update_frame_count(delta=delta)
                        self.metrics.update_epsilon()
                        self.metrics.update_expert_ratio()
                    controls = max(0, int(snapshot.get("total_controls", 0) or 0))
                    inf_time = max(0.0, float(snapshot.get("inference_time", 0.0) or 0.0))
                    inf_reqs = max(0, int(snapshot.get("inference_requests", 0) or 0))
                    if controls or inf_time or inf_reqs:
                        with self.metrics.lock:
                            self.metrics.total_controls += controls
                            self.metrics.total_inference_time += inf_time
                            self.metrics.total_inference_requests += inf_reqs
                    self._refresh_aggregate_client_metrics()
                elif kind == "episode_reward":
                    total, dqn, expert, subj, obj, length = item[2:]
                    self.metrics.add_episode_reward(total, dqn, expert, subj, obj, length=length)
                    try:
                        add_episode_to_dqn100k_window(float(dqn), int(length))
                        add_episode_to_dqn1m_window(float(dqn), int(length))
                        add_episode_to_dqn5m_window(float(dqn), int(length))
                        add_episode_to_total_windows(float(total), int(length))
                        add_episode_to_eplen_window(int(length))
                    except Exception:
                        pass
                elif kind == "peak_game_score":
                    score = max(0, int(item[2]))
                    if score > self.metrics.peak_game_score:
                        self.metrics.peak_game_score = score
                elif kind == "game_score":
                    self.metrics.add_game_score(max(0, int(item[2])))
            except Exception:
                traceback.print_exc()

    def _experience_loop(self) -> None:
        while self.running:
            try:
                item = self.experience_q.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                kind, payload = item
                if kind == "exp_batch":
                    batch = list(payload or [])
                elif kind == "exp":
                    batch = [payload]
                else:
                    continue
                if kind == "exp_batch" and hasattr(self.agent, "step_batch"):
                    base_batch = [tuple(x[:10]) for x in batch]
                    indices = self.agent.step_batch(base_batch)
                    if indices is None:
                        indices = [-1] * len(base_batch)
                    for item_row, idx in zip(batch, indices):
                        worker_id = int(item_row[10]) if len(item_row) > 10 else -1
                        client_id = int(item_row[11]) if len(item_row) > 11 else -1
                        episode_id = int(item_row[12]) if len(item_row) > 12 else 0
                        reward = float(item_row[2])
                        done = bool(item_row[4])
                        actor = str(item_row[5])
                        wave_number = int(item_row[8])
                        start_wave = int(item_row[9])
                        self._track_episode_step(
                            worker_id,
                            client_id,
                            episode_id,
                            idx,
                            reward,
                            done,
                            actor,
                            wave_number,
                            start_wave,
                        )
                else:
                    for item_row in batch:
                        (
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
                        ) = item_row[:10]
                        idx = self.agent.step(
                            state,
                            action,
                            reward,
                            next_state,
                            done,
                            actor=actor,
                            horizon=horizon,
                            priority_reward=priority_reward,
                            wave_number=wave_number,
                            start_wave=start_wave,
                        )
                        worker_id = int(item_row[10]) if len(item_row) > 10 else -1
                        client_id = int(item_row[11]) if len(item_row) > 11 else -1
                        episode_id = int(item_row[12]) if len(item_row) > 12 else 0
                        self._track_episode_step(
                            worker_id,
                            client_id,
                            episode_id,
                            idx,
                            float(reward),
                            bool(done),
                            str(actor),
                            int(wave_number),
                            int(start_wave),
                        )
            except Exception:
                traceback.print_exc()

    def _sync_loop(self) -> None:
        last_control_json = ""
        while self.running:
            try:
                control = self._control_snapshot()
                control_json = json.dumps(control, sort_keys=True)
                if control_json != last_control_json:
                    for ctl_q in self.control_queues:
                        try:
                            ctl_q.put(("control", control), timeout=0.1)
                        except Exception:
                            pass
                    last_control_json = control_json

                marker = self._sync_source_marker()
                if marker > self._sync_marker:
                    state_dict = self._export_inference_state_dict()
                    for ctl_q in self.control_queues:
                        try:
                            ctl_q.put(("weights", state_dict), timeout=0.2)
                        except Exception:
                            pass
                    self._sync_marker = marker
            except Exception:
                traceback.print_exc()
            time.sleep(0.1)

    def start(self) -> None:
        self.running = True
        self.master_thread = threading.Thread(target=self.master_server.start, daemon=True, name="MasterSocketServer")
        self.master_thread.start()

        initial_state = self._export_inference_state_dict()
        for worker_id, port in enumerate(self.worker_ports):
            ctl_q = self.ctx.Queue(maxsize=16)
            proc = self.ctx.Process(
                target=_worker_process_main,
                args=(
                    worker_id,
                    str(getattr(SERVER_CONFIG, "host", "0.0.0.0")),
                    int(port),
                    int(self.agent.state_size),
                    initial_state,
                    self.event_q,
                    self.experience_q,
                    ctl_q,
                ),
                daemon=True,
                name=f"RobotronShard{worker_id}",
            )
            proc.start()
            self.control_queues.append(ctl_q)
            self.worker_processes.append(proc)

        self._event_thread = threading.Thread(target=self._event_loop, daemon=True, name="ShardEventCollector")
        self._event_thread.start()
        self._experience_thread = threading.Thread(target=self._experience_loop, daemon=True, name="ShardExperienceIngest")
        self._experience_thread.start()
        self._sync_thread = threading.Thread(target=self._sync_loop, daemon=True, name="ShardSync")
        self._sync_thread.start()
        self._write_shard_env()

    def stop(self) -> None:
        if not self.running:
            return
        self.running = False
        for ctl_q in self.control_queues:
            try:
                ctl_q.put(("shutdown",), timeout=0.1)
            except Exception:
                pass
        try:
            self.master_server.stop()
        except Exception:
            pass
        for proc in self.worker_processes:
            try:
                proc.join(timeout=2.0)
            except Exception:
                pass
        try:
            if self.master_thread is not None:
                self.master_thread.join(timeout=2.0)
        except Exception:
            pass
        self._remove_shard_env()

    def is_alive(self) -> bool:
        master_alive = bool(self.master_thread is not None and self.master_thread.is_alive())
        workers_alive = any(proc.is_alive() for proc in self.worker_processes)
        return master_alive or workers_alive

    def get_client_rows(self) -> list[dict]:
        return self._combined_rows()

    def get_selected_preview_client_id(self) -> int | None:
        try:
            return self.master_server.get_selected_preview_client_slot()
        except Exception:
            return None

    def get_selected_preview_client_slot(self) -> int | None:
        return self.get_selected_preview_client_id()

    def set_preview_client(self, cid: int | None) -> tuple[bool, int | None]:
        if cid is None:
            ok, _selected_local = self.master_server.set_preview_client(None)
            return ok, self.get_selected_preview_client_slot()
        try:
            target_slot = int(cid)
        except Exception:
            return False, self.get_selected_preview_client_slot()
        target_local = None
        with self.master_server.client_lock:
            for local_cid, cs in self.master_server.client_states.items():
                if int(cs.get("client_slot", -1)) == target_slot and bool(cs.get("preview_capable", False)):
                    target_local = int(local_cid)
                    break
        if target_local is None:
            return False, self.get_selected_preview_client_slot()
        ok, _selected_local = self.master_server.set_preview_client(target_local)
        return ok, self.get_selected_preview_client_slot()


def clear_shard_env_file() -> None:
    robotron_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    shard_env_path = os.path.join(robotron_root, "logs", "server_shards.env")
    try:
        if os.path.exists(shard_env_path):
            os.remove(shard_env_path)
    except Exception:
        pass
