#!/usr/bin/env python3
"""Robotron AI v3 — Socket server bridge.

TCP server bridging Lua (MAME) ↔ Python for the v3 PPO architecture.
Preserves the exact binary wire protocol from v2 so the Lua side
is completely unchanged.

Wire protocol:
  Inbound (Lua → Python):
    4-byte big-endian length
    Header: >HddBIBBBIBB (n_params, subj_reward, obj_reward, done,
            score, player_alive, save, start_pressed, replay_level,
            num_lasers, wave_number)
    State: n × float32 big-endian
    Optional: preview data

  Outbound (Python → Lua):
    5 bytes: move_cmd(i8), fire_cmd(i8), source_byte(u8),
             start_advanced(u8), start_level_min(u8)
"""

import os
import sys
import time
import struct
import socket
import select
import threading
import traceback
import random
import pathlib
import numpy as np
import torch
from collections import deque
from typing import Optional
from dataclasses import dataclass

from .config import CONFIG, GAME_SETTINGS, WIRE_PARAMS_COUNT
from .agent import PPOAgent
from .expert import get_expert_action, get_expert_action_from_entities
from .state_processor import extract_entities
from .reward import shape_reward
from .metrics_display import add_episode_to_reward_windows, add_episode_to_eplen_windows
from .rollout_buffer import RolloutBuffer

# ── Constants ───────────────────────────────────────────────────────────────

FIRE_HOLD_FRAMES = 4
_MAX_FRAME_PAYLOAD_BYTES = 4 * 1024 * 1024
_DIAG = 0.70710678
_FIRE_DIR_VECTORS = (
    (0.0, -1.0), (_DIAG, -_DIAG), (1.0, 0.0), (_DIAG, _DIAG),
    (0.0, 1.0), (-_DIAG, _DIAG), (-1.0, 0.0), (-_DIAG, -_DIAG),
)
_START_PULSE_VALID_FRAMES = 240
_GAMEPLAY_RESET_DEAD_FRAMES = 180
_GAMEPLAY_PLAUSIBLE_START_STREAK = 8


def _env_flag(name: str, default: bool = False) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return bool(default)
    return raw not in {"0", "false", "off", "no"}


# ── Frame data (parsed from wire) ──────────────────────────────────────────

@dataclass
class FrameData:
    state: np.ndarray
    subjreward: float
    objreward: float
    done: bool
    player_alive: bool
    save_signal: bool
    start_pressed: bool = False
    level_number: int = 0
    game_score: int = 0
    next_replay_level: int = 0
    num_lasers: int = 0
    preview_width: int = 0
    preview_height: int = 0
    preview_format: int = 0
    preview_pixels: Optional[bytes] = None
    preview_encoded_format: int = 0
    preview_encoded_bytes: int = 0
    preview_raw_bytes: int = 0


def parse_frame_data(data: bytes, parse_preview: bool = False) -> Optional[FrameData]:
    """Parse the binary wire protocol from Lua."""
    fmt = ">HddBIBBBIBB"
    hdr_size = struct.calcsize(fmt)
    if not data or len(data) < hdr_size:
        return None

    vals = struct.unpack(fmt, data[:hdr_size])
    n, subj, obj, done, score, alive, save, start, replay, lasers, wave = vals

    base_len = hdr_size + n * 4
    if len(data) < base_len:
        return None

    state = np.frombuffer(data[hdr_size:base_len], dtype=">f4", count=n).astype(np.float32)
    if state.shape[0] != n:
        return None

    preview_width = preview_height = preview_format = 0
    preview_pixels = None
    preview_encoded_format = preview_encoded_bytes = preview_raw_bytes = 0

    if len(data) > base_len:
        if len(data) < (base_len + 4):
            return None
        preview_len = struct.unpack(">I", data[base_len:base_len + 4])[0]
        tail_start = base_len + 4
        tail_end = tail_start + int(preview_len)
        if tail_end != len(data):
            return None
        if (not parse_preview) and preview_len > 0:
            preview_len = 0
        elif preview_len > 0 and preview_len < 5:
            return None
        if preview_len >= 5:
            preview_width, preview_height, preview_format = struct.unpack(
                ">HHB", data[tail_start:tail_start + 5]
            )
            pixels = data[tail_start + 5:tail_end]
            if preview_width <= 0 or preview_height <= 0 or len(pixels) <= 0:
                return None
            expected_px = preview_width * preview_height * 2
            pf = int(preview_format)
            preview_encoded_format = pf
            preview_encoded_bytes = len(pixels)
            preview_raw_bytes = expected_px
            if pf == 1:
                if len(pixels) != expected_px:
                    return None
                preview_pixels = bytes(pixels)
            elif pf == 2:
                # LZSS decompression
                out = bytearray(expected_px)
                oi = si = 0
                plen = len(pixels)
                ok = True
                while oi < expected_px and si < plen:
                    flags = pixels[si]; si += 1
                    for bit in range(8):
                        if oi >= expected_px:
                            break
                        if (flags >> bit) & 1:
                            if (si + 1) >= plen:
                                ok = False; break
                            b1, b2 = pixels[si], pixels[si + 1]; si += 2
                            mlen = ((b1 >> 4) & 0x0F) + 3
                            dist = ((b1 & 0x0F) << 8) | b2
                            if dist <= 0 or dist > oi:
                                ok = False; break
                            src_idx = oi - dist
                            for _ in range(mlen):
                                if oi >= expected_px:
                                    break
                                out[oi] = out[src_idx]; oi += 1; src_idx += 1
                        else:
                            if si >= plen:
                                ok = False; break
                            out[oi] = pixels[si]; oi += 1; si += 1
                    if not ok:
                        break
                if (not ok) or (oi != expected_px):
                    return None
                preview_pixels = bytes(out)
                preview_format = 1
            elif pf == 3:
                # Word-RLE decompression
                out = bytearray(expected_px)
                oi = si = 0
                plen = len(pixels)
                ok = True
                while si < plen and oi < expected_px:
                    ctrl = pixels[si]; si += 1
                    words = (ctrl & 0x7F) + 1
                    if (ctrl & 0x80) != 0:
                        if (si + 1) >= plen:
                            ok = False; break
                        b0, b1 = pixels[si], pixels[si + 1]; si += 2
                        need = words * 2
                        if (oi + need) > expected_px:
                            ok = False; break
                        for _ in range(words):
                            out[oi] = b0; out[oi + 1] = b1; oi += 2
                    else:
                        need = words * 2
                        if (si + need) > plen or (oi + need) > expected_px:
                            ok = False; break
                        out[oi:oi + need] = pixels[si:si + need]
                        oi += need; si += need
                if (not ok) or (oi != expected_px) or (si != plen):
                    return None
                preview_pixels = bytes(out)
                preview_format = 1
            else:
                return None

    return FrameData(
        state=state,
        subjreward=subj,
        objreward=obj,
        done=bool(done),
        player_alive=bool(alive),
        save_signal=bool(save),
        start_pressed=bool(start),
        level_number=int(wave),
        game_score=int(score),
        next_replay_level=int(replay),
        num_lasers=int(lasers),
        preview_width=int(preview_width),
        preview_height=int(preview_height),
        preview_format=int(preview_format),
        preview_pixels=preview_pixels,
        preview_encoded_format=int(preview_encoded_format),
        preview_encoded_bytes=int(preview_encoded_bytes),
        preview_raw_bytes=int(preview_raw_bytes),
    )


# ── Action encoding (matching game's joystick directions) ──────────────────

def encode_action_to_game(move_dir: int, fire_dir: int) -> tuple[int, int]:
    """Convert model action indices to game joystick commands.

    move/fire 0-7 map to game directions 0-7.
    move/fire 8 (idle) maps to game -1 (no input).
    """
    move_cmd = int(move_dir) if 0 <= move_dir <= 7 else -1
    fire_cmd = int(fire_dir) if 0 <= fire_dir <= 7 else -1
    return move_cmd, fire_cmd


# ── Fire hold logic ─────────────────────────────────────────────────────────

def _apply_fire_hold(cs: dict, raw_fire: int) -> int:
    """Fixed-cadence fire hold for LSPROC's 3-stable-frame requirement."""
    cs["fire_pending_dir"] = int(raw_fire)
    count = cs.get("fire_hold_count", 0)
    if count > 0:
        cs["fire_hold_count"] = count - 1
        return cs.get("fire_hold_dir", raw_fire)
    next_fire = int(cs.get("fire_pending_dir", raw_fire))
    cs["fire_hold_dir"] = next_fire
    cs["fire_hold_count"] = FIRE_HOLD_FRAMES - 1
    return next_fire


# ── Metrics (lightweight rolling stats) ─────────────────────────────────────

class Metrics:
    """Thread-safe rolling metrics for the v3 system."""

    def __init__(self):
        self.lock = threading.Lock()
        self.total_frames = 0
        self.episode_rewards = deque(maxlen=200)
        self.episode_lengths = deque(maxlen=200)
        self.fps_window = deque(maxlen=60)
        self.peak_game_score = 0
        self.avg_game_score = 0.0
        self.total_games_played = 0
        self.episodes_this_run = 0
        self.client_count = 0
        self.web_client_count = 0
        self.avg_level = 0.0
        self.peak_level = 0.0
        self._level_window = deque(maxlen=200)
        self._game_scores = deque(maxlen=200)
        self._last_fps_time = time.time()
        self._fps_frames = 0
        # Preview frame data
        self.preview_capture_enabled = True
        self.hud_enabled = True
        self.game_preview_seq = 0
        self.game_preview_client_id = -1
        self.game_preview_width = 0
        self.game_preview_height = 0
        self.game_preview_format = ""
        self.game_preview_data = b""
        self.game_preview_updated_ts = 0.0
        self.game_preview_source_format = ""
        self.game_preview_encoded_bytes = 0
        self.game_preview_raw_bytes = 0
        self.game_preview_compression_ratio = 1.0
        self.game_preview_fps = 0.0
        # Reference to the server for client row queries
        self.global_server = None

    def update_frame(self):
        with self.lock:
            self.total_frames += 1
            self._fps_frames += 1
            now = time.time()
            elapsed = now - self._last_fps_time
            if elapsed >= 1.0:
                self.fps_window.append(self._fps_frames / elapsed)
                self._fps_frames = 0
                self._last_fps_time = now

    def add_episode(self, reward: float, length: int, level: float = 0.0, game_score: int = 0):
        with self.lock:
            self.episode_rewards.append(reward)
            self.episode_lengths.append(length)
            self.episodes_this_run += 1
            if level > 0:
                self._level_window.append(level)
                self.avg_level = sum(self._level_window) / len(self._level_window)
                if level > self.peak_level:
                    self.peak_level = level
            self._game_scores.append(int(game_score))
            self.avg_game_score = sum(self._game_scores) / len(self._game_scores)
            self.total_games_played += 1

    def update_client_count(self, count: int):
        with self.lock:
            self.client_count = count

    @property
    def avg_reward(self) -> float:
        with self.lock:
            if not self.episode_rewards:
                return 0.0
            return sum(self.episode_rewards) / len(self.episode_rewards)

    @property
    def avg_ep_len(self) -> float:
        with self.lock:
            if not self.episode_lengths:
                return 0.0
            return sum(self.episode_lengths) / len(self.episode_lengths)

    @property
    def fps(self) -> float:
        with self.lock:
            if not self.fps_window:
                return 0.0
            return sum(self.fps_window) / len(self.fps_window)


# ── Batched inference ───────────────────────────────────────────────────────

class _InferenceRequest:
    __slots__ = ("tensors", "need_actions", "result", "event")

    def __init__(self, tensors: dict, need_actions: bool):
        self.tensors = tensors          # dict of (1, ...) tensors on CPU
        self.need_actions = need_actions # True → sample actions; False → value only
        self.result: Optional[dict] = None
        self.event = threading.Event()


class InferenceBatcher:
    """Dedicated GPU inference thread that batches requests from client threads.

    Client threads call submit_action() or submit_value() which block until
    the batch is processed.  The GPU thread collects requests, concatenates
    tensors along dim 0, runs one forward pass for the whole batch, then
    distributes the per-sample results back via threading events.

    GPU isolation strategy:
      - Multi-GPU: inference runs on infer_device with infer_net — completely
        independent of training on train_device.  No lock needed.
      - Single-GPU with CUDA streams: inference runs on infer_stream,
        training on train_stream.  Streams overlap on the hardware scheduler.
        A lightweight gpu_lock is kept only for the brief net.eval()/train()
        mode toggle when both happen on the same net object.
      - CPU/MPS: falls back to a simple gpu_lock (same as before).
    """

    def __init__(self, agent, max_batch: int = 64, max_wait_ms: float = 1.5):
        self.agent = agent
        self.net = agent.get_inference_net()
        self.device = agent.infer_device
        self.stream = agent.infer_stream        # may be None (CPU/MPS)
        self._multi_gpu = agent._multi_gpu
        self.max_batch = max_batch
        self.max_wait_s = max_wait_ms / 1000.0
        self._queue: deque[_InferenceRequest] = deque()
        self._lock = threading.Lock()
        self._has_work = threading.Event()
        self._stopped = False
        # gpu_lock is only needed when inference and training share the
        # same net object AND there are no CUDA streams (i.e. CPU/MPS).
        self.gpu_lock = threading.Lock()
        self._use_gpu_lock = (self.stream is None and not self._multi_gpu)
        self._thread = threading.Thread(target=self._run, daemon=True, name="infer-batch")
        self._thread.start()

    def stop(self):
        self._stopped = True
        self._has_work.set()

    def submit_action(self, tensors: dict) -> dict:
        """Submit tensors for action sampling.  Blocks until result ready.

        Returns dict: move_action, fire_action, log_prob, entropy, value
        """
        req = _InferenceRequest(tensors, need_actions=True)
        with self._lock:
            self._queue.append(req)
        self._has_work.set()
        req.event.wait()
        return req.result

    def submit_value(self, tensors: dict) -> float:
        """Submit tensors for value-only estimation.  Blocks until ready.

        Returns scalar value estimate.
        """
        req = _InferenceRequest(tensors, need_actions=False)
        with self._lock:
            self._queue.append(req)
        self._has_work.set()
        req.event.wait()
        return req.result["value"]

    def _run(self):
        """GPU thread main loop: collect → batch → forward → distribute."""
        while not self._stopped:
            self._has_work.wait(timeout=0.05)
            self._has_work.clear()
            if self._stopped:
                break

            # Collect batch — drain queue, brief wait for stragglers
            batch: list[_InferenceRequest] = []
            deadline = time.monotonic() + self.max_wait_s

            while len(batch) < self.max_batch:
                with self._lock:
                    while self._queue and len(batch) < self.max_batch:
                        batch.append(self._queue.popleft())
                if len(batch) >= self.max_batch:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0 or len(batch) > 0:
                    break
                time.sleep(min(0.0005, remaining))

            if not batch:
                continue

            try:
                self._process_batch(batch)
            except Exception:
                # On error return fallbacks so client threads don't hang
                for req in batch:
                    if req.result is None:
                        req.result = {
                            "move_action": 0, "fire_action": 0,
                            "move_log_prob": 0.0, "fire_log_prob": 0.0,
                            "log_prob": 0.0, "entropy": 0.0, "value": 0.0,
                        }
                    req.event.set()

    @torch.no_grad()
    def _process_batch(self, batch: list[_InferenceRequest]):
        """Cat tensors, single forward pass, split results, signal events."""
        # Concatenate along batch dim
        efs = torch.cat([r.tensors["entity_features"] for r in batch], dim=0).to(self.device, non_blocking=True)
        ems = torch.cat([r.tensors["entity_mask"] for r in batch], dim=0).to(self.device, non_blocking=True)
        gcs = torch.cat([r.tensors["global_context"] for r in batch], dim=0).to(self.device, non_blocking=True)

        if self.stream is not None:
            # CUDA stream path — no lock needed (multi-GPU: separate device;
            # single-GPU: stream isolation handles scheduling)
            with torch.cuda.stream(self.stream):
                self.net.eval()
                out = self.net.forward(efs, ems, gcs)
            # Synchronize so results are ready before CPU reads them
            self.stream.synchronize()
        elif self._use_gpu_lock:
            # CPU/MPS fallback — serialise with training via lock
            with self.gpu_lock:
                self.net.eval()
                out = self.net.forward(efs, ems, gcs)
        else:
            self.net.eval()
            out = self.net.forward(efs, ems, gcs)

        # NaN-safe logit clamping
        move_logits = out["move_logits"].clamp(-50.0, 50.0)
        fire_logits = out["fire_logits"].clamp(-50.0, 50.0)
        values = out["value"]
        if torch.isnan(move_logits).any() or torch.isinf(move_logits).any():
            move_logits = torch.zeros_like(move_logits)
        if torch.isnan(fire_logits).any() or torch.isinf(fire_logits).any():
            fire_logits = torch.zeros_like(fire_logits)
        if torch.isnan(values).any():
            values = torch.where(torch.isnan(values), torch.zeros_like(values), values)

        # Sample actions (cheap — kept on GPU for the batch)
        move_dist = torch.distributions.Categorical(logits=move_logits)
        fire_dist = torch.distributions.Categorical(logits=fire_logits)
        move_actions = move_dist.sample()
        fire_actions = fire_dist.sample()
        move_log_probs = move_dist.log_prob(move_actions)
        fire_log_probs = fire_dist.log_prob(fire_actions)
        log_probs = move_log_probs + fire_log_probs
        entropies = move_dist.entropy() + fire_dist.entropy()

        # Move to CPU once for all items
        move_actions = move_actions.cpu()
        fire_actions = fire_actions.cpu()
        move_log_probs = move_log_probs.cpu()
        fire_log_probs = fire_log_probs.cpu()
        log_probs = log_probs.cpu()
        entropies = entropies.cpu()
        values = values.cpu()

        # Distribute results back to client threads
        for i, req in enumerate(batch):
            if req.need_actions:
                req.result = {
                    "move_action": int(move_actions[i].item()),
                    "fire_action": int(fire_actions[i].item()),
                    "move_log_prob": float(move_log_probs[i].item()),
                    "fire_log_prob": float(fire_log_probs[i].item()),
                    "log_prob": float(log_probs[i].item()),
                    "entropy": float(entropies[i].item()),
                    "value": float(values[i].item()),
                }
            else:
                req.result = {"value": float(values[i].item())}
            req.event.set()


# ── Socket Server ───────────────────────────────────────────────────────────

class SocketServer:
    """TCP server bridging MAME/Lua clients to the PPO agent.

    Each MAME instance connects on its own TCP socket. The server
    manages per-client state, frame processing, action selection,
    and rollout collection.
    """

    def __init__(
        self,
        agent: PPOAgent,
        host: str = None,
        port: int = None,
        max_clients: int = None,
    ):
        cfg = CONFIG.server
        self.agent = agent
        self.host = host or cfg.host
        self.port = port or cfg.port
        self.max_clients = max_clients or cfg.max_clients

        self.metrics = Metrics()
        self.metrics.global_server = self
        self.running = False
        self.shutdown_event = threading.Event()
        self.client_states: dict[int, dict] = {}
        self.client_lock = threading.Lock()
        self._next_cid = 0
        self.preview_cid: Optional[int] = None

        # ── Training: async transition collection ────────────────────
        # Transitions queued from client threads, drained by training thread.
        self._transition_queue: deque = deque()
        self._transition_lock = threading.Lock()
        self._train_start_lock = threading.Lock()
        self._train_thread: Optional[threading.Thread] = None
        self._training_active = threading.Event()  # set while train_step is running
        self._train_batch_size = CONFIG.train.rollout_length  # collect this many before training

        # ── Batched inference ────────────────────────────────────────
        self.batcher = InferenceBatcher(
            agent=agent,
            max_batch=max(64, self.max_clients + 8),
            max_wait_ms=1.5,
        )
        default_skip_expert_value = True
        if self.agent.infer_device.type == "cpu":
            default_skip_expert_value = _env_flag("ROBOTRON_SKIP_EXPERT_VALUE_ON_CPU", True)
        self._skip_expert_value = _env_flag("ROBOTRON_SKIP_EXPERT_VALUE", default_skip_expert_value)

    def start(self):
        """Start the server (blocking)."""
        self.running = True
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.settimeout(1.0)

        server_sock.bind((self.host, self.port))
        server_sock.listen(self.max_clients)
        print(f"v3 Socket server listening on {self.host}:{self.port}")

        # Write server_shards.env so startmame.sh sends ALL clients to this single port
        self._write_shard_env()

        try:
            while self.running and not self.shutdown_event.is_set():
                try:
                    sock, addr = server_sock.accept()
                except socket.timeout:
                    continue

                with self.client_lock:
                    cid = self._next_cid
                    self._next_cid += 1
                    self.client_states[cid] = self._new_client_state()

                thread = threading.Thread(
                    target=self._handle_client,
                    args=(sock, cid),
                    daemon=True,
                )
                thread.start()
                print(f"Client {cid} connected from {addr}")
                self.metrics.update_client_count(len(self.client_states))
        finally:
            self.running = False
            server_sock.close()

    def stop(self):
        """Signal shutdown."""
        self.running = False
        self.shutdown_event.set()
        self.batcher.stop()

    def _write_shard_env(self):
        """Write server_shards.env so startmame.sh routes all clients here."""
        host = os.getenv("ROBOTRON_SOCKET_PUBLIC_HOST", "").strip()
        if not host:
            bind_host = str(self.host or "127.0.0.1")
            if bind_host in {"0.0.0.0", "::", "[::]"}:
                host = "127.0.0.1"
            else:
                host = bind_host
        log_dir = pathlib.Path(__file__).resolve().parent.parent.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        env_path = log_dir / "server_shards.env"
        tmp_path = env_path.with_suffix(".tmp")
        lines = [
            f"ROBOTRON_SHARD_ENABLED=0",
            f"ROBOTRON_SOCKET_HOST={host}",
            f"ROBOTRON_MASTER_PORT={self.port}",
            f"ROBOTRON_WORKER_PORTS=",
            f"ROBOTRON_PREVIEW_SLOT=0",
        ]
        tmp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        tmp_path.replace(env_path)
        print(f"  Wrote {env_path} (all clients → port {self.port})")

    # ── Training coordinator ────────────────────────────────────────────

    def _push_transition(
        self,
        entity_features: torch.Tensor,
        entity_mask: torch.Tensor,
        global_context: torch.Tensor,
        move_action: int,
        fire_action: int,
        log_prob: float,
        value: float,
        has_value: bool,
        reward: float,
        done: bool,
        next_value: float,
        expert_move: int = 8,
        expert_fire: int = 8,
        is_expert: bool = False,
        policy_sampled: bool = False,
        fire_locked: bool = False,
    ):
        """Thread-safe push of one transition. Triggers training when batch is full."""
        txn = (
            entity_features, entity_mask, global_context,
            move_action, fire_action, log_prob, value, has_value, reward, done, next_value,
            expert_move, expert_fire, is_expert, policy_sampled, fire_locked,
        )
        with self._transition_lock:
            self._transition_queue.append(txn)
            queue_len = len(self._transition_queue)

        if queue_len >= self._train_batch_size and self.agent.training_enabled:
            self._start_training()

    def _start_training(self):
        """Launch training on a background thread if not already running."""
        thread = None
        with self._train_start_lock:
            if self._training_active.is_set():
                return
            if self._train_thread is not None and self._train_thread.is_alive():
                return
            # Mark the slot active before publishing the thread object so
            # concurrent client threads cannot race a second start() call.
            self._training_active.set()
            thread = threading.Thread(target=self._train_worker, daemon=True, name="v3-train")
            self._train_thread = thread
        try:
            thread.start()
        except Exception:
            with self._train_start_lock:
                if self._train_thread is thread:
                    self._train_thread = None
                self._training_active.clear()
            raise

    def _train_worker(self):
        """Drain transition queue into a RolloutBuffer and run PPO update."""
        try:
            with self._transition_lock:
                batch_size = min(len(self._transition_queue), self._train_batch_size)
                if batch_size < 64:
                    return  # not enough data
                batch = [self._transition_queue.popleft() for _ in range(batch_size)]

            # Build a single-actor rollout buffer
            rollout = RolloutBuffer(
                rollout_length=batch_size,
                num_actors=1,
                device=torch.device("cpu"),
            )

            for t, txn in enumerate(batch):
                (ef, em, gc, ma, fa, lp, val, has_val, rew, done, next_val,
                 ex_m, ex_f, is_exp, policy_sampled, fire_locked) = txn
                rollout.entity_features[t, 0] = ef
                rollout.entity_masks[t, 0] = em
                rollout.global_contexts[t, 0] = gc
                rollout.move_actions[t, 0] = ma
                rollout.fire_actions[t, 0] = fa
                rollout.log_probs[t, 0] = lp
                rollout.values[t, 0] = val
                rollout.has_value[t, 0] = has_val
                rollout.rewards[t, 0] = rew
                rollout.dones[t, 0] = done
                rollout.expert_move[t, 0] = ex_m
                rollout.expert_fire[t, 0] = ex_f
                rollout.is_expert[t, 0] = is_exp
                rollout.policy_sampled[t, 0] = policy_sampled
                rollout.fire_locked[t, 0] = fire_locked
                if has_val:
                    bootstrap = 0.0 if done else float(next_val)
                    rollout.advantages[t, 0] = float(rew) + (float(CONFIG.train.gamma) * bootstrap) - float(val)
                    rollout.returns[t, 0] = rollout.advantages[t, 0] + float(val)
                else:
                    rollout.advantages[t, 0] = 0.0
                    rollout.returns[t, 0] = 0.0

            rollout.step = batch_size
            rollout.ready = True

            # Run training on the appropriate device/stream
            if self.agent.train_stream is not None:
                with torch.cuda.stream(self.agent.train_stream):
                    self.agent.train_step(rollout)
                self.agent.train_stream.synchronize()
            elif self.batcher._use_gpu_lock:
                with self.batcher.gpu_lock:
                    self.agent.train_step(rollout)
            else:
                self.agent.train_step(rollout)

            # Sync weights to inference network after training update
            self.agent.sync_inference_weights()

        except Exception as e:
            print(f"[v3] Training error: {e}")
            traceback.print_exc()
        finally:
            with self._train_start_lock:
                self._training_active.clear()
                self._train_thread = None


    def _new_client_state(self) -> dict:
        return {
            "frames": 0,
            "game_frames": 0,
            "player_alive": False,
            "alive_streak": 0,
            "dead_streak": 0,
            "gameplay_seen": False,
            "start_pulse_window": 0,
            "plausible_start_streak": 0,
            "level_number": 0,
            "start_wave": 1,
            "game_score": 0,
            "num_lasers": 0,
            "last_time": time.time(),
            "fps": 0.0,
            "was_done": False,
            "total_reward": 0.0,
            "ep_frames": 0,
            "episode_id": 1,
            "fire_hold_dir": -1,
            "fire_hold_count": 0,
            "fire_pending_dir": -1,
            "last_state": None,
            "last_action": None,
            "last_player_alive": False,
            "prev_action_source": None,
            "last_alive_game_score": 0,
            "preview_capable": False,
            "client_slot": 0,
            # Training state (tensors from act_with_value)
            "last_tensors": None,
            "last_log_prob": 0.0,
            "last_value": 0.0,
            "last_has_value": False,
            "last_is_expert": False,
            "last_expert_move": 8,
            "last_expert_fire": 8,
            "last_policy_sampled": False,
            "last_fire_locked": False,
        }

    # ── Preview client selection ────────────────────────────────────────

    @staticmethod
    def _parse_client_handshake(handshake_value: int) -> tuple[bool, int]:
        raw = max(0, int(handshake_value or 0))
        preview_capable = (raw & 0x01) != 0
        client_slot = max(0, raw >> 1)
        return preview_capable, client_slot

    def _pick_default_preview_client_locked(self) -> Optional[int]:
        candidates = []
        for cid, cs in self.client_states.items():
            if not bool(cs.get("preview_capable", False)):
                continue
            slot = int(cs.get("client_slot", cid))
            candidates.append((slot, int(cid)))
        if not candidates:
            return None
        candidates.sort()
        return int(candidates[0][1])

    def _ensure_preview_client_selected_locked(self) -> tuple[Optional[int], bool]:
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

    def _preview_enabled_for_client(self, cid: int) -> bool:
        if not self._is_preview_client(cid):
            return False
        with self.client_lock:
            cs = self.client_states.get(int(cid))
            if not isinstance(cs, dict) or not bool(cs.get("preview_capable", False)):
                return False
        with self.metrics.lock:
            if not self.metrics.preview_capture_enabled:
                return False
            return int(self.metrics.web_client_count or 0) > 0

    def _hud_enabled_for_client(self, cid: int) -> bool:
        if not self._is_preview_client(cid):
            return False
        with self.metrics.lock:
            return bool(self.metrics.hud_enabled)

    def _clear_preview_cache(self):
        with self.metrics.lock:
            self.metrics.game_preview_client_id = -1
            self.metrics.game_preview_seq = 0
            self.metrics.game_preview_width = 0
            self.metrics.game_preview_height = 0
            self.metrics.game_preview_format = ""
            self.metrics.game_preview_data = b""
            self.metrics.game_preview_updated_ts = 0.0
            self.metrics.game_preview_source_format = ""
            self.metrics.game_preview_encoded_bytes = 0
            self.metrics.game_preview_raw_bytes = 0
            self.metrics.game_preview_compression_ratio = 1.0
            self.metrics.game_preview_fps = 0.0

    def _cache_client_preview(self, cid: int, frame: FrameData):
        if not self._is_preview_client(cid):
            return
        pixels = frame.preview_pixels
        width = frame.preview_width
        height = frame.preview_height
        if not pixels or width <= 0 or height <= 0:
            return
        if frame.preview_format != 1:
            return
        expected_len = width * height * 2
        if len(pixels) != expected_len:
            return
        enc_fmt = frame.preview_encoded_format
        enc_bytes = frame.preview_encoded_bytes
        raw_bytes = frame.preview_raw_bytes
        with self.metrics.lock:
            prev_seq = self.metrics.game_preview_seq
            prev_ts = self.metrics.game_preview_updated_ts
            now_ts = time.time()
            next_seq = prev_seq + 1
            if prev_ts > 0.0:
                dt = max(1e-6, now_ts - prev_ts)
                self.metrics.game_preview_fps = 1.0 / dt
            self.metrics.game_preview_client_id = int(cid)
            self.metrics.game_preview_seq = next_seq
            self.metrics.game_preview_width = width
            self.metrics.game_preview_height = height
            self.metrics.game_preview_format = "rgb565be"
            self.metrics.game_preview_data = bytes(pixels)
            self.metrics.game_preview_updated_ts = now_ts
            self.metrics.game_preview_source_format = {1: "raw", 2: "lzss", 3: "rle"}.get(enc_fmt, "unknown")
            self.metrics.game_preview_encoded_bytes = enc_bytes
            self.metrics.game_preview_raw_bytes = raw_bytes if raw_bytes > 0 else expected_len
            if enc_bytes > 0 and raw_bytes > 0:
                self.metrics.game_preview_compression_ratio = float(raw_bytes) / float(enc_bytes)
            else:
                self.metrics.game_preview_compression_ratio = 1.0

    def get_client_rows(self) -> list[dict]:
        changed = False
        with self.client_lock:
            selected, changed = self._ensure_preview_client_selected_locked()
            rows = []
            for cid, cs in self.client_states.items():
                if bool(cs.get("gameplay_seen", False)):
                    lives = max(0, int(cs.get("num_lasers", 0) or 0))
                    level = max(0, int(cs.get("level_number", 0) or 0))
                    score = max(0, int(cs.get("game_score", 0) or 0))
                else:
                    lives = level = score = 0
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
        rows.sort(key=lambda r: int(r.get("client_id", 0)))
        return rows

    def get_selected_preview_client_id(self) -> Optional[int]:
        changed = False
        with self.client_lock:
            selected, changed = self._ensure_preview_client_selected_locked()
        if changed:
            self._clear_preview_cache()
        return None if selected is None else int(selected)

    def get_selected_preview_client_slot(self) -> Optional[int]:
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

    def set_preview_client(self, cid: Optional[int]) -> tuple[bool, Optional[int]]:
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

    def _recv_exact(self, sock, n: int, timeout_s: float = 0.5) -> Optional[bytes]:
        """Read exactly n bytes from socket."""
        chunks = []
        remaining = n
        deadline = time.time() + timeout_s
        while remaining > 0:
            if time.time() > deadline:
                return None
            ready = select.select([sock], [], [], max(0.001, deadline - time.time()))
            if not ready[0]:
                continue
            chunk = sock.recv(min(remaining, 65536))
            if not chunk:
                return None
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _pack_action(
        self,
        move_cmd: int,
        fire_cmd: int,
        source_code: int,
        preview_enabled: bool = False,
        hud_enabled: bool = False,
    ) -> bytes:
        """Pack 5-byte action response for Lua.

        source byte layout:
          bits 0-3: action source (0=none, 1=policy, 2=epsilon, 3=expert)
          bit 6:    preview enable flag (tells Lua to send preview frame)
          bit 7:    HUD enable flag
        """
        start_adv = 1 if GAME_SETTINGS.start_advanced else 0
        start_level = max(1, GAME_SETTINGS.start_level_min)
        source_u8 = (int(source_code) & 0x0F)
        if preview_enabled:
            source_u8 |= 0x40
        if hud_enabled:
            source_u8 |= 0x80
        return struct.pack(
            ">bbBBB",
            int(move_cmd),
            int(fire_cmd),
            int(source_u8),
            int(start_adv),
            int(start_level),
        )

    def _handle_client(self, sock: socket.socket, cid: int):
        """Per-client frame loop."""
        try:
            sock.setblocking(False)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

            # Handshake — parse preview capability from the 2-byte value
            sock.setblocking(True)
            sock.settimeout(5.0)
            ping = self._recv_exact(sock, 2, timeout_s=5.0)
            if not ping or len(ping) < 2:
                raise ConnectionError("No handshake")
            handshake_val = struct.unpack(">H", ping)[0]
            preview_capable, client_slot = self._parse_client_handshake(handshake_val)
            is_preview_client = False
            with self.client_lock:
                cs = self.client_states.get(cid)
                if isinstance(cs, dict):
                    cs["preview_capable"] = bool(preview_capable)
                    cs["client_slot"] = int(client_slot)
                selected, changed = self._ensure_preview_client_selected_locked()
            if changed:
                self._clear_preview_cache()
            is_preview_client = bool(preview_capable) and selected is not None and int(selected) == int(cid)
            sock.setblocking(False)
            sock.settimeout(None)

            while self.running and not self.shutdown_event.is_set():
                ready = select.select([sock], [], [], 0.002)
                if not ready[0]:
                    continue

                # Read length header
                hdr = self._recv_exact(sock, 4, timeout_s=0.25)
                if hdr is None or len(hdr) < 4:
                    raise ConnectionError("EOF")
                dlen = struct.unpack(">I", hdr)[0]
                if dlen <= 0 or dlen > _MAX_FRAME_PAYLOAD_BYTES:
                    raise ConnectionError(f"Invalid payload: {dlen}")

                data = self._recv_exact(sock, dlen, timeout_s=0.5)
                if data is None:
                    raise ConnectionError("Broken")

                # Validate param count
                if len(data) >= 2:
                    n = struct.unpack(">H", data[:2])[0]
                    if n != WIRE_PARAMS_COUNT:
                        print(f"Client {cid}: param mismatch {n} != {WIRE_PARAMS_COUNT}")
                        break

                preview_enabled = self._preview_enabled_for_client(cid)
                hud_enabled = self._hud_enabled_for_client(cid)
                should_parse_preview = bool(self._is_preview_client(cid) and preview_enabled)
                frame = parse_frame_data(data, parse_preview=should_parse_preview)
                if not frame:
                    sock.sendall(self._pack_action(-1, -1, 0, preview_enabled=preview_enabled, hud_enabled=hud_enabled))
                    continue

                with self.client_lock:
                    if cid not in self.client_states:
                        break
                    cs = self.client_states[cid]
                    cs["frames"] += 1
                    cs["game_frames"] = cs.get("game_frames", 0) + 1
                    cs["player_alive"] = frame.player_alive
                    cs["num_lasers"] = frame.num_lasers

                    # Track alive/dead streaks
                    if frame.player_alive:
                        cs["alive_streak"] = cs.get("alive_streak", 0) + 1
                        cs["dead_streak"] = 0
                    else:
                        cs["alive_streak"] = 0
                        cs["dead_streak"] = cs.get("dead_streak", 0) + 1

                    if cs["dead_streak"] >= _GAMEPLAY_RESET_DEAD_FRAMES:
                        cs["gameplay_seen"] = False
                        cs["start_pulse_window"] = 0
                        cs["level_number"] = 0
                        cs["start_wave"] = 1
                        cs["game_score"] = 0

                    # Detect game start
                    if frame.start_pressed:
                        cs["start_pulse_window"] = _START_PULSE_VALID_FRAMES
                    elif cs.get("start_pulse_window", 0) > 0:
                        cs["start_pulse_window"] -= 1

                    plausible = frame.player_alive and frame.game_score >= 0
                    if cs.get("start_pulse_window", 0) > 0 and plausible:
                        cs["plausible_start_streak"] = cs.get("plausible_start_streak", 0) + 1
                    else:
                        cs["plausible_start_streak"] = 0

                    if (
                        cs.get("start_pulse_window", 0) > 0
                        and cs.get("alive_streak", 0) >= 15
                        and cs.get("plausible_start_streak", 0) >= _GAMEPLAY_PLAUSIBLE_START_STREAK
                    ):
                        if not cs.get("gameplay_seen"):
                            cs["gameplay_seen"] = True
                            cs["ep_frames"] = 0
                            cs["start_wave"] = max(1, frame.level_number)
                        cs["start_pulse_window"] = 0

                    if frame.player_alive and cs.get("gameplay_seen"):
                        cs["level_number"] = frame.level_number
                        cs["game_score"] = max(0, frame.game_score)

                    if (
                        frame.player_alive
                        and cs.get("gameplay_seen")
                        and frame.num_lasers == 0
                        and frame.game_score > self.metrics.peak_game_score
                    ):
                        self.metrics.peak_game_score = frame.game_score

                    # Track per-game score
                    if frame.player_alive:
                        if frame.game_score < cs.get("last_alive_game_score", 0):
                            cs["ep_frames"] = 0
                            cs["start_wave"] = max(1, frame.level_number)
                        cs["last_alive_game_score"] = frame.game_score

                # Cache preview frame if this is the selected preview client
                if should_parse_preview and frame.preview_pixels:
                    self._cache_client_preview(cid, frame)

                self.metrics.update_frame()
                self.agent.total_frames = self.metrics.total_frames

                pending_reward = None
                pending_prev_tensors = None
                pending_prev_action = None
                pending_prev_log_prob = 0.0
                pending_prev_value = 0.0
                pending_prev_has_value = False
                pending_prev_expert_move = 8
                pending_prev_expert_fire = 8
                pending_prev_is_expert = False
                pending_prev_policy_sampled = False
                pending_prev_fire_locked = False

                # ── Process previous step reward → prepare transition ──
                if cs.get("last_state") is not None and cs.get("last_action") is not None:
                    pending_reward = shape_reward(frame.objreward, frame.subjreward, frame.done)
                    cs["total_reward"] += pending_reward
                    cs["ep_frames"] = cs.get("ep_frames", 0) + 1

                    pending_prev_tensors = cs.get("last_tensors")
                    pending_prev_action = cs.get("last_action")
                    pending_prev_log_prob = float(cs.get("last_log_prob", 0.0) or 0.0)
                    pending_prev_value = float(cs.get("last_value", 0.0) or 0.0)
                    pending_prev_has_value = bool(cs.get("last_has_value", False))
                    pending_prev_expert_move = int(cs.get("last_expert_move", 8) or 8)
                    pending_prev_expert_fire = int(cs.get("last_expert_fire", 8) or 8)
                    pending_prev_is_expert = bool(cs.get("last_is_expert", False))
                    pending_prev_policy_sampled = bool(cs.get("last_policy_sampled", False))
                    pending_prev_fire_locked = bool(cs.get("last_fire_locked", False))

                # ── Terminal ────────────────────────────────────────────
                if frame.done:
                    if pending_prev_tensors is not None and pending_prev_action is not None and pending_reward is not None:
                        self._push_transition(
                            entity_features=pending_prev_tensors["entity_features"],
                            entity_mask=pending_prev_tensors["entity_mask"],
                            global_context=pending_prev_tensors["global_context"],
                            move_action=pending_prev_action[0],
                            fire_action=pending_prev_action[1],
                            log_prob=pending_prev_log_prob,
                            value=pending_prev_value,
                            has_value=pending_prev_has_value,
                            reward=pending_reward,
                            done=True,
                            next_value=0.0,
                            expert_move=pending_prev_expert_move,
                            expert_fire=pending_prev_expert_fire,
                            is_expert=pending_prev_is_expert,
                            policy_sampled=pending_prev_policy_sampled,
                            fire_locked=pending_prev_fire_locked,
                        )
                    self.agent._reset_frame_buffer(cid)
                    cs["fire_hold_dir"] = -1
                    cs["fire_hold_count"] = 0
                    cs["fire_pending_dir"] = -1

                    if not cs.get("was_done"):
                        ep_reward = cs["total_reward"]
                        ep_len = cs.get("ep_frames", 0)
                        ep_level = float(cs.get("level_number", 0))
                        ep_score = int(cs.get("game_score", 0))
                        self.metrics.add_episode(
                            ep_reward, ep_len,
                            level=ep_level, game_score=ep_score,
                        )
                        add_episode_to_reward_windows(ep_reward, ep_len)
                        add_episode_to_eplen_windows(ep_len)
                    cs["was_done"] = True
                    _pv = self._preview_enabled_for_client(cid)
                    _hd = self._hud_enabled_for_client(cid)
                    sock.sendall(self._pack_action(-1, -1, 0, preview_enabled=_pv, hud_enabled=_hd))
                    cs["last_state"] = cs["last_action"] = None
                    cs["last_tensors"] = None
                    cs["last_log_prob"] = 0.0
                    cs["last_value"] = 0.0
                    cs["last_has_value"] = False
                    cs["last_is_expert"] = False
                    cs["last_expert_move"] = 8
                    cs["last_expert_fire"] = 8
                    cs["last_policy_sampled"] = False
                    cs["last_fire_locked"] = False
                    cs["last_player_alive"] = False
                    cs["prev_action_source"] = None
                    cs["episode_id"] = cs.get("episode_id", 1) + 1
                    cs["total_reward"] = 0.0
                    cs["ep_frames"] = 0
                    continue

                if cs.get("was_done"):
                    cs["was_done"] = False
                    cs["total_reward"] = 0.0
                    cs["ep_frames"] = 0

                # Skip dead/attract frames
                if not frame.player_alive:
                    if pending_prev_tensors is not None and pending_prev_action is not None and pending_reward is not None:
                        self._push_transition(
                            entity_features=pending_prev_tensors["entity_features"],
                            entity_mask=pending_prev_tensors["entity_mask"],
                            global_context=pending_prev_tensors["global_context"],
                            move_action=pending_prev_action[0],
                            fire_action=pending_prev_action[1],
                            log_prob=pending_prev_log_prob,
                            value=pending_prev_value,
                            has_value=pending_prev_has_value,
                            reward=pending_reward,
                            done=True,
                            next_value=0.0,
                            expert_move=pending_prev_expert_move,
                            expert_fire=pending_prev_expert_fire,
                            is_expert=pending_prev_is_expert,
                            policy_sampled=pending_prev_policy_sampled,
                            fire_locked=pending_prev_fire_locked,
                        )
                    self.agent._reset_frame_buffer(cid)
                    cs["last_state"] = cs["last_action"] = None
                    cs["last_tensors"] = None
                    cs["last_log_prob"] = 0.0
                    cs["last_value"] = 0.0
                    cs["last_has_value"] = False
                    cs["last_is_expert"] = False
                    cs["last_expert_move"] = 8
                    cs["last_expert_fire"] = 8
                    cs["last_policy_sampled"] = False
                    cs["last_fire_locked"] = False
                    cs["last_player_alive"] = False
                    cs["prev_action_source"] = None
                    cs["fire_hold_dir"] = -1
                    cs["fire_hold_count"] = 0
                    cs["fire_pending_dir"] = -1
                    _pv = self._preview_enabled_for_client(cid)
                    _hd = self._hud_enabled_for_client(cid)
                    sock.sendall(self._pack_action(-1, -1, 0, preview_enabled=_pv, hud_enabled=_hd))
                    continue

                # ── Choose action ───────────────────────────────────────
                wire_state = frame.state
                epsilon = self.agent.get_epsilon()
                expert_ratio = self.agent.get_expert_ratio()

                fire_update_open = cs.get("fire_hold_count", 0) <= 0
                locked_fire = None
                if not fire_update_open:
                    held = cs.get("fire_hold_dir", -1)
                    locked_fire = max(0, min(8, held)) if held >= 0 else 8

                # Decide: expert vs policy
                use_expert = random.random() < expert_ratio
                action_source = "none"
                is_epsilon = False
                log_prob = 0.0
                value = 0.0
                has_value = False
                tensors_dict = None
                expert_move_out = 8
                expert_fire_out = 8
                policy_sampled = False
                fire_locked_now = locked_fire is not None

                # Process state once — shared by expert + model (eliminates
                # double entity extraction that was the #1 bottleneck).
                tensors = self.agent._process_and_stack(wire_state, cid)

                if use_expert:
                    wave = max(1, cs.get("level_number", 1))
                    # Use pre-extracted entities from the already-processed
                    # frame inside the agent's frame buffer — avoids the old
                    # double-extraction path.
                    buf = self.agent._get_frame_buffer(cid)
                    latest = buf[-1] if buf else None
                    if latest is not None:
                        move_idx, fire_idx = get_expert_action_from_entities(
                            latest["entity_features"],
                            latest["entity_mask"],
                            latest["num_entities"],
                            wave_number=wave,
                        )
                    else:
                        move_idx, fire_idx = get_expert_action(wire_state, wave_number=wave)
                    expert_move_out = move_idx
                    if locked_fire is not None:
                        fire_idx = locked_fire
                    expert_fire_out = fire_idx
                    action_source = "expert"
                    if pending_prev_has_value or not self._skip_expert_value:
                        value = self.batcher.submit_value(tensors)
                        has_value = True
                    tensors_dict = self.agent._detach_tensors(tensors)
                else:
                    is_epsilon = random.random() < epsilon
                    if is_epsilon:
                        move_idx = random.randrange(CONFIG.model.num_move_actions)
                        fire_idx = random.randrange(CONFIG.model.num_fire_actions)
                        if locked_fire is not None:
                            fire_idx = locked_fire
                        value = self.batcher.submit_value(tensors)
                        has_value = True
                        log_prob = 0.0
                    else:
                        res = self.batcher.submit_action(tensors)
                        move_idx = res["move_action"]
                        fire_idx = res["fire_action"]
                        log_prob = res["move_log_prob"] if locked_fire is not None else res["log_prob"]
                        value = res["value"]
                        has_value = True
                        if locked_fire is not None:
                            fire_idx = locked_fire
                        policy_sampled = True
                    tensors_dict = self.agent._detach_tensors(tensors)
                    action_source = "policy"

                if pending_prev_tensors is not None and pending_prev_action is not None and pending_reward is not None:
                    self._push_transition(
                        entity_features=pending_prev_tensors["entity_features"],
                        entity_mask=pending_prev_tensors["entity_mask"],
                        global_context=pending_prev_tensors["global_context"],
                        move_action=pending_prev_action[0],
                        fire_action=pending_prev_action[1],
                        log_prob=pending_prev_log_prob,
                        value=pending_prev_value,
                        has_value=pending_prev_has_value,
                        reward=pending_reward,
                        done=False,
                        next_value=value,
                        expert_move=pending_prev_expert_move,
                        expert_fire=pending_prev_expert_fire,
                        is_expert=pending_prev_is_expert,
                        policy_sampled=pending_prev_policy_sampled,
                        fire_locked=pending_prev_fire_locked,
                    )

                # Apply fire hold
                effective_fire = _apply_fire_hold(cs, fire_idx)

                cs["last_state"] = wire_state
                cs["last_action"] = (move_idx, effective_fire)
                cs["last_player_alive"] = frame.player_alive
                cs["prev_action_source"] = action_source
                cs["last_tensors"] = tensors_dict
                cs["last_log_prob"] = log_prob
                cs["last_value"] = value
                cs["last_has_value"] = has_value
                cs["last_is_expert"] = use_expert
                cs["last_expert_move"] = expert_move_out
                cs["last_expert_fire"] = expert_fire_out
                cs["last_policy_sampled"] = policy_sampled
                cs["last_fire_locked"] = fire_locked_now

                # Save signal
                if frame.save_signal:
                    self.agent.save()

                # Send action
                move_cmd, fire_cmd = encode_action_to_game(move_idx, effective_fire)
                if action_source == "expert":
                    source_byte = 3
                elif is_epsilon:
                    source_byte = 2
                elif action_source == "policy":
                    source_byte = 1
                else:
                    source_byte = 0

                _pv = self._preview_enabled_for_client(cid)
                _hd = self._hud_enabled_for_client(cid)
                sock.sendall(self._pack_action(move_cmd, fire_cmd, source_byte, preview_enabled=_pv, hud_enabled=_hd))

        except Exception as e:
            is_expected = isinstance(e, (ConnectionError, BrokenPipeError, ConnectionResetError, TimeoutError))
            if not is_expected:
                print(f"Client {cid} error: {e}")
                traceback.print_exc()
        finally:
            with self.client_lock:
                self.client_states.pop(cid, None)
                self.metrics.update_client_count(len(self.client_states))
            try:
                sock.close()
            except Exception:
                pass
            print(f"Client {cid} disconnected")
