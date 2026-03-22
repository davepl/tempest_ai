#!/usr/bin/env python3
# ==================================================================================================================
# ||  ROBOTRON AI v2 • MULTIPROCESS INFERENCE POOL                                                               ||
# ||  Fans out batched policy inference across multiple worker processes while the main process keeps training,   ||
# ||  metrics aggregation, preview routing, and the dashboard.                                                   ||
# ==================================================================================================================
"""Multiprocess inference pool for Robotron."""

from __future__ import annotations

import itertools
import multiprocessing as mp
import os
import queue
import random
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.append(_SCRIPT_DIR)

try:
    from aimodel import RainbowNet, split_joint_action, NUM_MOVE, NUM_FIRE, _cuda_device
    from config import RL_CONFIG
except ImportError:
    try:
        from .aimodel import RainbowNet, split_joint_action, NUM_MOVE, NUM_FIRE, _cuda_device
        from .config import RL_CONFIG
    except ImportError:
        from Scripts.aimodel import RainbowNet, split_joint_action, NUM_MOVE, NUM_FIRE, _cuda_device
        from Scripts.config import RL_CONFIG


def _worker_inference_device() -> torch.device:
    if bool(getattr(RL_CONFIG, "inference_on_cpu", False)):
        return torch.device("cpu")
    if torch.cuda.is_available():
        return _cuda_device(getattr(RL_CONFIG, "inference_cuda_device_index", 0))
    return torch.device("cpu")


def _greedy_axes_from_q(q_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    q_joint = q_values.view(-1, NUM_MOVE, NUM_FIRE)
    move_scores = q_joint.max(dim=2).values
    fire_scores = q_joint.max(dim=1).values
    return move_scores.argmax(dim=1), fire_scores.argmax(dim=1)


def _compute_actions(
    net: RainbowNet,
    device: torch.device,
    states: list[np.ndarray],
    epsilons: list[float],
    locked_fires: list[Optional[int]],
) -> list[tuple[int, int, bool]]:
    n = min(len(states), len(epsilons), len(locked_fires))
    if n <= 0:
        return []

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
        lf = locked_fires[i]
        if lf is not None:
            lf_i = int(lf)
            if lf_i >= 0:
                fire_fixed[i] = True
                fire_fixed_vals[i] = max(0, min(NUM_FIRE - 1, lf_i))

        rand_moves[i] = random.random() < eps
        if rand_moves[i]:
            rnd_move_vals[i] = random.randrange(NUM_MOVE)

        if fire_fixed[i]:
            rand_fires[i] = False
            rnd_fire_vals[i] = fire_fixed_vals[i]
            needs_greedy = not rand_moves[i]
        else:
            rand_fires[i] = random.random() < eps
            if rand_fires[i]:
                rnd_fire_vals[i] = random.randrange(NUM_FIRE)
            needs_greedy = not (rand_moves[i] and rand_fires[i])

        if needs_greedy:
            greedy_idx.append(i)
            greedy_states.append(states[i])

    greedy_actions: dict[int, tuple[int, int]] = {}
    if greedy_idx:
        batch_np = np.asarray(greedy_states, dtype=np.float32)
        st = torch.from_numpy(batch_np).to(device)
        with torch.inference_mode():
            q = net.q_values(st)
        if bool(getattr(RL_CONFIG, "factored_greedy_action", False)):
            gm_t, gf_t = _greedy_axes_from_q(q)
            gm = gm_t.detach().cpu().tolist()
            gf = gf_t.detach().cpu().tolist()
            for pos, m, f in zip(greedy_idx, gm, gf):
                greedy_actions[pos] = (int(m), int(f))
        else:
            gm_t, gf_t = _greedy_axes_from_q(q)
            gm = gm_t.detach().cpu().tolist()
            gf = gf_t.detach().cpu().tolist()
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


def _load_state_dict(net: RainbowNet, state_dict: dict[str, torch.Tensor]) -> None:
    net.load_state_dict(state_dict, strict=False)
    net.eval()


def _drain_control_queue(
    control_q,
    net: RainbowNet,
    device: torch.device,
) -> bool:
    """Handle queued control messages. Returns False when worker should exit."""
    while True:
        try:
            item = control_q.get_nowait()
        except queue.Empty:
            return True
        kind = item[0]
        if kind == "shutdown":
            return False
        if kind == "weights":
            _, state_dict = item
            _load_state_dict(net, state_dict)


def _inference_worker_main(
    worker_id: int,
    state_size: int,
    request_q,
    control_q,
    result_q,
    initial_state_dict: dict[str, torch.Tensor],
    max_batch_size: int,
    max_wait_ms: float,
) -> None:
    try:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        torch.set_num_threads(1)
        device = _worker_inference_device()
        if device.type == "cuda" and device.index is not None:
            torch.cuda.set_device(device.index)

        net = RainbowNet(state_size).to(device)
        _load_state_dict(net, initial_state_dict)

        max_batch = max(1, int(max_batch_size))
        max_wait_s = max(0.0, float(max_wait_ms) / 1000.0)
        running = True

        while running:
            running = _drain_control_queue(control_q, net, device)
            if not running:
                break
            try:
                first = request_q.get(timeout=0.02)
            except queue.Empty:
                continue

            batch = [first]
            deadline = time.perf_counter() + max_wait_s
            while len(batch) < max_batch:
                running = _drain_control_queue(control_q, net, device)
                if not running:
                    break
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    batch.append(request_q.get(timeout=remaining))
                except queue.Empty:
                    break
            if not running:
                break

            req_ids = [int(item[0]) for item in batch]
            states = [item[1] for item in batch]
            epsilons = [float(item[2]) for item in batch]
            locked_fires = [item[3] for item in batch]
            try:
                actions = _compute_actions(net, device, states, epsilons, locked_fires)
            except Exception:
                traceback.print_exc()
                actions = [(0, 0, False)] * len(batch)

            for req_id, action in zip(req_ids, actions):
                result_q.put((req_id, action))
    except Exception:
        traceback.print_exc()


@dataclass
class _PendingResult:
    event: threading.Event
    action: Optional[tuple[int, int, bool]] = None


class ProcessInferencePool:
    """Multiprocess inference fan-out with local fallback to the master agent."""

    def __init__(
        self,
        agent,
        worker_count: int,
        max_batch_size: int,
        max_wait_ms: float,
        request_timeout_ms: float,
    ):
        self.agent = agent
        self.worker_count = max(1, int(worker_count))
        self.max_batch_size = max(1, int(max_batch_size))
        self.max_wait_ms = max(0.0, float(max_wait_ms))
        self.request_timeout_s = max(0.001, float(request_timeout_ms) / 1000.0)
        self.ctx = mp.get_context("spawn")
        self.result_q = self.ctx.Queue(maxsize=max(2048, self.worker_count * 2048))
        self.request_queues = []
        self.control_queues = []
        self.processes = []
        self.running = True

        self._req_counter = itertools.count(1)
        self._rr_counter = itertools.count(0)
        self._pending: dict[int, _PendingResult] = {}
        self._pending_lock = threading.Lock()
        self._sync_marker = -1

        initial_state = self._export_inference_state_dict()
        for worker_id in range(self.worker_count):
            req_q = self.ctx.Queue(maxsize=max(1024, self.max_batch_size * 64))
            ctl_q = self.ctx.Queue(maxsize=4)
            proc = self.ctx.Process(
                target=_inference_worker_main,
                args=(
                    worker_id,
                    int(self.agent.state_size),
                    req_q,
                    ctl_q,
                    self.result_q,
                    initial_state,
                    self.max_batch_size,
                    self.max_wait_ms,
                ),
                daemon=True,
                name=f"InferProc{worker_id}",
            )
            proc.start()
            self.request_queues.append(req_q)
            self.control_queues.append(ctl_q)
            self.processes.append(proc)

        self._result_thread = threading.Thread(
            target=self._consume_results, daemon=True, name="InferPoolResults"
        )
        self._result_thread.start()
        self._sync_thread = threading.Thread(
            target=self._sync_loop, daemon=True, name="InferPoolSync"
        )
        self._sync_thread.start()

    def _export_inference_state_dict(self) -> dict[str, torch.Tensor]:
        src_net = self.agent.infer_net if bool(getattr(self.agent, "use_separate_inference", False)) else self.agent.online_net
        return {k: v.detach().cpu() for k, v in src_net.state_dict().items()}

    def _sync_source_marker(self) -> int:
        if bool(getattr(self.agent, "use_separate_inference", False)):
            return int(getattr(self.agent, "last_inference_sync", 0) or 0)
        return int(getattr(self.agent, "training_steps", 0) or 0)

    def _pick_worker(self, route_key: Optional[int]) -> int:
        if self.worker_count <= 1:
            return 0
        if route_key is not None:
            try:
                return abs(int(route_key)) % self.worker_count
            except Exception:
                pass
        return next(self._rr_counter) % self.worker_count

    def _consume_results(self) -> None:
        while self.running:
            try:
                req_id, action = self.result_q.get(timeout=0.05)
            except queue.Empty:
                continue
            pending = None
            with self._pending_lock:
                pending = self._pending.pop(int(req_id), None)
            if pending is None:
                continue
            pending.action = tuple(action) if action is not None else (0, 0, False)
            pending.event.set()

    def _sync_loop(self) -> None:
        while self.running:
            try:
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

    def infer(
        self,
        state,
        epsilon: float,
        locked_fire: Optional[int] = None,
        route_key: Optional[int] = None,
    ):
        if not self.running:
            return self.agent.act(state, epsilon, locked_fire=locked_fire)

        req_id = int(next(self._req_counter))
        pending = _PendingResult(event=threading.Event())
        with self._pending_lock:
            self._pending[req_id] = pending

        worker_idx = self._pick_worker(route_key)
        payload = (
            req_id,
            np.asarray(state, dtype=np.float32),
            float(epsilon),
            None if locked_fire is None else int(locked_fire),
        )
        try:
            self.request_queues[worker_idx].put(payload, timeout=self.request_timeout_s)
        except Exception:
            with self._pending_lock:
                self._pending.pop(req_id, None)
            return self.agent.act(state, epsilon, locked_fire=locked_fire)

        if not pending.event.wait(timeout=self.request_timeout_s):
            with self._pending_lock:
                self._pending.pop(req_id, None)
            return self.agent.act(state, epsilon, locked_fire=locked_fire)
        return pending.action if pending.action is not None else self.agent.act(state, epsilon, locked_fire=locked_fire)

    def stop(self):
        self.running = False
        for ctl_q in self.control_queues:
            try:
                ctl_q.put(("shutdown",), timeout=0.1)
            except Exception:
                pass
        for proc in self.processes:
            try:
                proc.join(timeout=2.0)
            except Exception:
                pass
        try:
            self._result_thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            self._sync_thread.join(timeout=1.0)
        except Exception:
            pass
        with self._pending_lock:
            pending_items = list(self._pending.values())
            self._pending.clear()
        for pending in pending_items:
            pending.action = (0, 0, False)
            pending.event.set()
