#!/usr/bin/env python3
"""Robotron AI v3 — Live metrics dashboard.

Grafana-style web dashboard for PPO training telemetry.
Ported from the v2 dashboard, adapted for Set Transformer + PPO architecture.
"""

import atexit
import asyncio
import base64
import json
import math
import os
import queue
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import webbrowser
from collections import deque
from fractions import Fraction
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

from .config import CONFIG, GAME_SETTINGS
from .metrics_display import (
    get_reward_window_averages,
    get_eplen_100k_average,
    get_eplen_1m_average,
)

try:
    from aiortc import (
        RTCPeerConnection,
        RTCSessionDescription,
        RTCConfiguration,
        RTCIceServer,
        AudioStreamTrack,
    )
    from av import AudioFrame
    _AUDIO_WEBRTC_AVAILABLE = True
    _AUDIO_WEBRTC_IMPORT_ERROR = ""
except Exception as e:
    RTCPeerConnection = None
    RTCSessionDescription = None
    RTCConfiguration = None
    RTCIceServer = None
    AudioStreamTrack = object
    AudioFrame = None
    _AUDIO_WEBRTC_AVAILABLE = False
    _AUDIO_WEBRTC_IMPORT_ERROR = f"{type(e).__name__}: {e}"

# ── Constants ───────────────────────────────────────────────────────────────

LEVEL_100K_FRAMES = 100_000
LEVEL_1M_FRAMES = 1_000_000
WEB_CLIENT_TIMEOUT_S = 5.0
DASH_HISTORY_LIMIT = 40_000
PREVIEW_HTTP_DEFAULT_MAX_W = 320
PREVIEW_HTTP_DEFAULT_MAX_H = 240
_FALLBACK_ICE_SERVERS = [{"urls": ["stun:stun.l.google.com:19302"]}]


def _env_flag(name: str, default: bool = False) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return bool(default)
    return raw not in {"0", "false", "off", "no"}


_PREVIEW_GAME_AUDIO_TRANSPORT_ENABLED = _env_flag("ROBOTRON_GAME_AUDIO_ENABLED", True) and _AUDIO_WEBRTC_AVAILABLE
_AUDIO_WEBRTC_ICE_SERVERS = _FALLBACK_ICE_SERVERS


def _tail_mean(values, count: int = 20) -> float:
    if not values:
        return 0.0
    tail = list(values)[-count:]
    return float(sum(tail) / max(1, len(tail))) if tail else 0.0


def _downscale_rgb565_nearest(
    raw: bytes, width: int, height: int, max_width: int, max_height: int
) -> tuple[bytes, int, int]:
    w, h = max(0, width), max(0, height)
    mw, mh = max(0, max_width), max(0, max_height)
    if not raw or w <= 0 or h <= 0:
        return b"", 0, 0
    if (mw <= 0 or w <= mw) and (mh <= 0 or h <= mh):
        return raw, w, h
    sx = (float(w) / float(mw)) if mw > 0 else 1.0
    sy = (float(h) / float(mh)) if mh > 0 else 1.0
    scale = max(1.0, sx, sy)
    if scale <= 1.0:
        return raw, w, h
    tw = max(1, int(math.floor(w / scale)))
    th = max(1, int(math.floor(h / scale)))
    src_stride = w * 2
    out = bytearray(tw * th * 2)
    oi = 0
    for ty in range(th):
        sy_idx = min(h - 1, int(((ty + 0.5) * h) / th))
        row_base = sy_idx * src_stride
        for tx in range(tw):
            sx_idx = min(w - 1, int(((tx + 0.5) * w) / tw))
            si = row_base + (sx_idx * 2)
            out[oi] = raw[si]
            out[oi + 1] = raw[si + 1]
            oi += 2
    return bytes(out), tw, th


# ── Dashboard state ─────────────────────────────────────────────────────────

class _DashboardState:
    def __init__(self, metrics_obj, agent_obj=None, history_limit: int = DASH_HISTORY_LIMIT):
        self.metrics = metrics_obj
        self.agent = agent_obj
        self.history: deque = deque(maxlen=max(120, history_limit))
        self.latest: dict[str, Any] = {}
        self.lock = threading.Lock()
        self.last_steps: Optional[int] = None
        self.last_steps_time: Optional[float] = None
        self._web_clients: dict[str, float] = {}
        self._cached_now_body = b"{}"
        self._model_desc: Optional[str] = None
        self._level_windows = {
            "100k": {"limit": LEVEL_100K_FRAMES, "samples": deque(), "frames": 0, "weighted": 0.0},
            "1m": {"limit": LEVEL_1M_FRAMES, "samples": deque(), "frames": 0, "weighted": 0.0},
        }
        self._last_level_frame_count: Optional[int] = None
        self._sample_count: int = 0
        self._first_sample_time: Optional[float] = None
        # GPU / CPU polling
        self._gpu_snapshot: list[dict[str, Any]] = []
        self._gpu_last_poll_ts: float = 0.0
        self._gpu_poll_interval_s: float = 1.0
        self._cpu_last_totals: Optional[tuple[int, int]] = None
        self._cpu_snapshot: dict[str, float] = {"cpu_pct": 0.0, "mem_free_gb": 0.0, "disk_free_gb": 0.0}
        self._cpu_last_poll_ts: float = 0.0
        self._cpu_poll_interval_s: float = 1.0

    def _update_web_client_count_locked(self, now_ts: Optional[float] = None) -> int:
        now = float(now_ts if now_ts is not None else time.time())
        stale = [cid for cid, ts in self._web_clients.items() if ts < now - WEB_CLIENT_TIMEOUT_S]
        for cid in stale:
            self._web_clients.pop(cid, None)
        active = len(self._web_clients)
        with self.metrics.lock:
            self.metrics.web_client_count = active
        return active

    def touch_web_client(self, client_id: Optional[str]):
        if not client_id:
            return
        now = time.time()
        with self.lock:
            self._web_clients[client_id] = now
            self._update_web_client_count_locked(now)

    def _update_level_windows(self, frame_count: int, average_level: float) -> tuple[float, float]:
        level = round(float(average_level), 4) if math.isfinite(average_level) else 0.0
        if self._last_level_frame_count is None:
            self._last_level_frame_count = frame_count
            return level, level
        if frame_count < self._last_level_frame_count:
            for win in self._level_windows.values():
                win["samples"].clear()
                win["frames"] = 0
                win["weighted"] = 0.0
            self._last_level_frame_count = frame_count
            return level, level
        delta = max(0, frame_count - self._last_level_frame_count)
        self._last_level_frame_count = frame_count
        if delta > 0:
            for win in self._level_windows.values():
                samples = win["samples"]
                if samples and abs(samples[-1][0] - level) < 1e-9:
                    old_l, old_f = samples[-1]
                    samples[-1] = (old_l, old_f + delta)
                else:
                    samples.append((level, delta))
                win["frames"] += delta
                win["weighted"] += level * delta
                while samples and win["frames"] > win["limit"]:
                    overflow = win["frames"] - win["limit"]
                    ol, of = samples[0]
                    if of <= overflow:
                        samples.popleft()
                        win["frames"] -= of
                        win["weighted"] -= ol * of
                    else:
                        samples[0] = (ol, of - overflow)
                        win["frames"] -= overflow
                        win["weighted"] -= ol * overflow
                        break

        def _mean(w):
            return w["weighted"] / max(1, w["frames"]) if w["frames"] > 0 else level

        return _mean(self._level_windows["100k"]), _mean(self._level_windows["1m"])

    def _get_model_desc(self) -> str:
        if self._model_desc:
            return self._model_desc
        cfg = CONFIG.model
        agent = self.agent
        param_count = 0
        if agent and hasattr(agent, "net"):
            param_count = sum(p.numel() for p in agent.net.parameters())
        layers = [
            f"ent{cfg.entity_feature_dim}",
            f"isab{cfg.num_isab_layers}x{cfg.embed_dim}",
            f"ind{cfg.num_inducing_points}",
            f"g{cfg.global_context_dim}",
            f"fs{cfg.frame_stack}",
            f"fus{cfg.fusion_hidden}x{cfg.fusion_layers}",
            f"act({cfg.num_move_actions}+{cfg.num_fire_actions})",
        ]
        arch = " » ".join(layers)
        if param_count >= 1_000_000:
            p_str = f"{param_count / 1_000_000:.1f}M"
        elif param_count >= 1_000:
            p_str = f"{param_count / 1_000:.0f}K"
        else:
            p_str = str(param_count)
        self._model_desc = f"Model: SetTransformer+PPO · {arch} · {p_str} params"
        return self._model_desc

    def _get_model_summary_text(self) -> str:
        cfg = CONFIG.model
        tcfg = CONFIG.train
        lines = [
            self._get_model_desc(),
            "",
            "Encoder",
            f"  Entity feature dim: {cfg.entity_feature_dim}",
            f"  Max entities: {cfg.max_entities}",
            f"  ISAB layers / heads: {cfg.num_isab_layers} / {cfg.num_heads}",
            f"  Inducing points: {cfg.num_inducing_points}",
            "",
            "Policy",
            f"  Frame stack: {cfg.frame_stack}",
            f"  Global context dim: {cfg.global_context_dim}",
            f"  Fusion hidden/layers: {cfg.fusion_hidden} / {cfg.fusion_layers}",
            f"  Move / fire actions: {cfg.num_move_actions} / {cfg.num_fire_actions}",
            "",
            "Training",
            f"  Rollout length: {tcfg.rollout_length}",
            f"  Mini-batch size: {tcfg.mini_batch_size}",
            f"  PPO epochs: {tcfg.num_epochs}",
            f"  LR: {tcfg.lr:.1e}  min {tcfg.lr_min:.1e}",
            f"  Gamma / GAE: {tcfg.gamma:.3f} / {tcfg.gae_lambda:.3f}",
        ]
        return "\n".join(lines)

    def _sample_gpu_status(self, now_ts: Optional[float] = None) -> list[dict[str, Any]]:
        now = float(now_ts if now_ts is not None else time.time())
        if self._gpu_snapshot and (now - self._gpu_last_poll_ts) < self._gpu_poll_interval_s:
            return list(self._gpu_snapshot)
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=0.75, check=False,
            )
            rows: list[dict[str, Any]] = []
            if res.returncode == 0:
                for line in (res.stdout or "").splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) < 5:
                        continue
                    try:
                        idx = int(parts[0])
                        util = max(0.0, min(100.0, float(parts[2])))
                        mem_used = max(0.0, float(parts[3]))
                        mem_total = max(1.0, float(parts[4]))
                    except Exception:
                        continue
                    rows.append({
                        "index": idx, "name": parts[1], "util": util,
                        "mem_used_mb": mem_used, "mem_total_mb": mem_total,
                        "mem_pct": max(0.0, min(100.0, (mem_used / mem_total) * 100.0)),
                    })
            self._gpu_snapshot = rows
            self._gpu_last_poll_ts = now
        except Exception:
            self._gpu_last_poll_ts = now
        return list(self._gpu_snapshot)

    def _sample_system_status(self, now_ts: Optional[float] = None) -> dict[str, float]:
        now = float(now_ts if now_ts is not None else time.time())
        if (now - self._cpu_last_poll_ts) < self._cpu_poll_interval_s:
            return dict(self._cpu_snapshot)
        cpu_pct = self._cpu_snapshot.get("cpu_pct", 0.0)
        mem_free_gb = self._cpu_snapshot.get("mem_free_gb", 0.0)
        disk_free_gb = self._cpu_snapshot.get("disk_free_gb", 0.0)
        try:
            with open("/proc/stat", "r") as fh:
                first = fh.readline().strip().split()
            if len(first) >= 5 and first[0] == "cpu":
                vals = [int(v) for v in first[1:] if v.isdigit()]
                if vals:
                    total = sum(vals)
                    idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
                    prev = self._cpu_last_totals
                    if prev is not None:
                        dt = max(1, total - prev[0])
                        didle = max(0, idle - prev[1])
                        cpu_pct = max(0.0, min(100.0, (1.0 - didle / dt) * 100.0))
                    self._cpu_last_totals = (total, idle)
            with open("/proc/meminfo", "r") as fh:
                for line in fh:
                    if line.startswith("MemAvailable:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            mem_free_gb = max(0.0, int(parts[1]) / (1024.0 * 1024.0))
                        break
            du = shutil.disk_usage("/")
            disk_free_gb = max(0.0, du.free / (1024.0 ** 3))
        except Exception:
            pass
        self._cpu_snapshot = {"cpu_pct": cpu_pct, "mem_free_gb": mem_free_gb, "disk_free_gb": disk_free_gb}
        self._cpu_last_poll_ts = now
        return dict(self._cpu_snapshot)

    def _build_snapshot(self) -> dict[str, Any]:
        now = time.time()
        m = self.metrics
        agent = self.agent

        with m.lock:
            frame_count = m.total_frames
            fps = sum(m.fps_window) / len(m.fps_window) if m.fps_window else 0.0
            episode_rewards = list(m.episode_rewards)
            episode_lengths = list(m.episode_lengths)
            client_count = m.client_count
            web_client_count = m.web_client_count
            average_level = m.avg_level
            peak_level = m.peak_level
            peak_game_score = m.peak_game_score
            avg_game_score = m.avg_game_score
            game_count = m.total_games_played
            episodes_this_run = m.episodes_this_run
            game_preview_seq = m.game_preview_seq
            game_preview_client_id = m.game_preview_client_id
            game_preview_width = m.game_preview_width
            game_preview_height = m.game_preview_height
            game_preview_format = m.game_preview_format
            game_preview_fps = m.game_preview_fps
            game_preview_source_format = m.game_preview_source_format
            game_preview_encoded_bytes = m.game_preview_encoded_bytes
            game_preview_raw_bytes = m.game_preview_raw_bytes
            game_preview_compression_ratio = m.game_preview_compression_ratio

        reward_avg = _tail_mean(episode_rewards)
        eplen_avg = _tail_mean(episode_lengths)

        rwd100k, rwd1m, rwd5m = get_reward_window_averages()
        level_100k, level_1m = self._update_level_windows(frame_count, average_level)

        epsilon = agent.get_epsilon() if agent else 0.0
        expert_ratio = agent.get_expert_ratio() if agent else 0.0
        bc_weight = agent._get_bc_weight() if agent else 0.0
        lr = agent.optimizer.param_groups[0]["lr"] if agent else 0.0
        policy_loss = agent.last_policy_loss if agent else 0.0
        value_loss = agent.last_value_loss if agent else 0.0
        entropy = agent.last_entropy if agent else 0.0
        bc_loss = agent.last_bc_loss if agent else 0.0
        grad_norm = agent.last_grad_norm if agent else 0.0
        total_loss = agent.last_loss if agent else 0.0
        training_steps = agent._training_steps if agent else 0

        steps_per_sec = 0.0
        if self.last_steps is not None and self.last_steps_time is not None:
            dt = max(1e-6, now - self.last_steps_time)
            ds = max(0, training_steps - self.last_steps)
            steps_per_sec = ds / dt
        self.last_steps = training_steps
        self.last_steps_time = now

        # Client rows from server
        client_rows: list[dict[str, Any]] = []
        preview_selected_client_id = -1
        try:
            srv = getattr(m, "global_server", None)
            if srv is not None:
                client_rows = list(srv.get_client_rows() or [])
                sel = srv.get_selected_preview_client_id()
                if sel is not None:
                    preview_selected_client_id = int(sel)
        except Exception:
            pass

        gpu_rows = self._sample_gpu_status(now)
        gpu0 = gpu_rows[0] if len(gpu_rows) >= 1 else {}
        gpu1 = gpu_rows[1] if len(gpu_rows) >= 2 else {}
        sys_status = self._sample_system_status(now)

        return {
            "ts": now,
            "frame_count": frame_count,
            "fps": fps,
            "training_steps": training_steps,
            "steps_per_sec": steps_per_sec,
            "epsilon": epsilon,
            "expert_ratio": expert_ratio,
            "bc_weight": bc_weight,
            "client_count": client_count,
            "web_client_count": web_client_count,
            "average_level": average_level,
            "level_100k": level_100k,
            "level_1m": level_1m,
            "peak_level": peak_level,
            "peak_game_score": peak_game_score,
            "avg_game_score": avg_game_score,
            "game_count": game_count,
            "episodes_this_run": episodes_this_run,
            "reward_avg": reward_avg,
            "rwd_100k": rwd100k,
            "rwd_1m": rwd1m,
            "rwd_5m": rwd5m,
            "eplen_100k": get_eplen_100k_average(),
            "eplen_1m": get_eplen_1m_average(),
            "loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "bc_loss": bc_loss,
            "grad_norm": grad_norm,
            "lr": lr,
            "lr_max": CONFIG.train.lr,
            "lr_min": CONFIG.train.lr_min,
            "model_desc": self._get_model_desc(),
            "model_summary_text": self._get_model_summary_text(),
            "client_rows": client_rows,
            "preview_selected_client_id": preview_selected_client_id,
            "game_preview_seq": game_preview_seq,
            "game_preview_client_id": game_preview_client_id,
            "game_preview_width": game_preview_width,
            "game_preview_height": game_preview_height,
            "game_preview_format": game_preview_format,
            "game_preview_fps": game_preview_fps,
            "game_preview_source_format": game_preview_source_format,
            "game_preview_encoded_bytes": game_preview_encoded_bytes,
            "game_preview_raw_bytes": game_preview_raw_bytes,
            "game_preview_compression_ratio": game_preview_compression_ratio,
            "preview_capture_enabled": bool(getattr(m, "preview_capture_enabled", True)),
            "hud_enabled": bool(getattr(m, "hud_enabled", True)),
            "gpu_rows": gpu_rows,
            "gpu0_util": float(gpu0.get("util", 0.0) or 0.0),
            "gpu0_mem_pct": float(gpu0.get("mem_pct", 0.0) or 0.0),
            "gpu0_name": str(gpu0.get("name", "") or ""),
            "gpu1_util": float(gpu1.get("util", 0.0) or 0.0),
            "gpu1_mem_pct": float(gpu1.get("mem_pct", 0.0) or 0.0),
            "gpu1_name": str(gpu1.get("name", "") or ""),
            "cpu_pct": sys_status.get("cpu_pct", 0.0),
            "mem_free_gb": sys_status.get("mem_free_gb", 0.0),
            "disk_free_gb": sys_status.get("disk_free_gb", 0.0),
            "game_settings": {"start_advanced": GAME_SETTINGS.start_advanced, "start_level_min": GAME_SETTINGS.start_level_min},
        }

    def sample(self):
        with self.lock:
            self._update_web_client_count_locked()
        snap = self._build_snapshot()
        with self.lock:
            now = time.time()
            if self._first_sample_time is None:
                self._first_sample_time = now
            self._sample_count += 1
            if self._sample_count <= 10 and (now - self._first_sample_time) < 2.0:
                self.latest = snap
                self._cached_now_body = json.dumps(snap).encode("utf-8")
                return
            self.latest = snap
            self.history.append(snap)
            self._cached_now_body = json.dumps(snap).encode("utf-8")

    def payload(self) -> dict[str, Any]:
        with self.lock:
            return {"now": self.latest, "history": list(self.history)}

    def now_body(self) -> bytes:
        with self.lock:
            return self._cached_now_body

    def game_preview_body(self, since_seq: Optional[int] = None) -> bytes:
        m = self.metrics
        with m.lock:
            seq = int(m.game_preview_seq)
            if since_seq is not None and seq <= since_seq:
                return json.dumps({"seq": seq, "changed": False}).encode("utf-8")
            w = int(m.game_preview_width)
            h = int(m.game_preview_height)
            raw = m.game_preview_data
        if not raw or w <= 0 or h <= 0:
            return json.dumps({"seq": seq, "changed": False}).encode("utf-8")
        scaled, sw, sh = _downscale_rgb565_nearest(raw, w, h, PREVIEW_HTTP_DEFAULT_MAX_W, PREVIEW_HTTP_DEFAULT_MAX_H)
        b64 = base64.b64encode(scaled).decode("ascii")
        return json.dumps({
            "seq": seq, "changed": True, "width": sw, "height": sh,
            "format": "rgb565be", "data": b64,
        }).encode("utf-8")


class _PreviewAudioSource:
    """Tail the selected preview client's bounded relay WAV and fan-out PCM frames."""

    _FRAME_PTIME_S = 0.02
    _MAX_SOURCE_LATENCY_S = 0.50
    _IDLE_KEEP_FRAMES = 1
    _SUBSCRIBER_QUEUE_FRAMES = 16

    def __init__(self, metrics_obj, audio_dir: str = "/tmp", file_template: str = "robotron_audio_client{slot}.wav"):
        self.metrics = metrics_obj
        self.audio_dir = str(audio_dir)
        self.file_template = str(file_template)
        self._sample_rate = 48000
        self._channels = 2
        self._buffer = bytearray()
        self._lock = threading.Lock()
        self._audio_ready = False
        self._stop_event = threading.Event()
        self._reader_thread_obj: Optional[threading.Thread] = None
        self._subscribers: dict[int, queue.Queue] = {}
        self._next_subscriber_id = 1
        self._current_slot: Optional[int] = None
        self._current_path: Optional[str] = None
        self._current_file = None
        self._current_inode: Optional[int] = None
        self._source_generation = 0
        self._read_offset = 0
        self._header_parsed = False
        self._start_reader()

    def _selected_slot(self) -> Optional[int]:
        try:
            srv = getattr(self.metrics, "global_server", None)
            if srv is not None:
                slot = srv.get_selected_preview_client_slot()
                if slot is not None:
                    return max(0, int(slot))
        except Exception:
            pass
        return None

    def _audio_path_for_slot(self, slot: int) -> str:
        return os.path.join(self.audio_dir, self.file_template.format(slot=max(0, int(slot))))

    def _parse_wav_header(self, header: bytes) -> Optional[tuple[int, int]]:
        if len(header) < 44:
            return None
        try:
            if header[0:4] != b"RIFF" or header[8:12] != b"WAVE":
                return None
            channels = int.from_bytes(header[22:24], "little", signed=False)
            sample_rate = int.from_bytes(header[24:28], "little", signed=False)
            bits_per_sample = int.from_bytes(header[34:36], "little", signed=False)
            if channels not in (1, 2) or sample_rate <= 0 or bits_per_sample != 16:
                return None
            return sample_rate, channels
        except Exception:
            return None

    def _frame_params_locked(self) -> tuple[int, int]:
        samples_per_frame = max(1, int(round(self._sample_rate * self._FRAME_PTIME_S)))
        bytes_per_frame = samples_per_frame * self._channels * 2
        return samples_per_frame, bytes_per_frame

    def _drop_queue_head(self, q: queue.Queue) -> None:
        try:
            q.get_nowait()
        except queue.Empty:
            return

    def _dispatch_ready_frames_locked(self) -> None:
        samples_per_frame, bytes_per_frame = self._frame_params_locked()
        if bytes_per_frame <= 0:
            return
        source_slot = self._current_slot
        source_generation = int(self._source_generation)
        if not self._subscribers:
            keep_bytes = max(bytes_per_frame * self._IDLE_KEEP_FRAMES, self._channels * 2)
            if len(self._buffer) > keep_bytes:
                del self._buffer[:-keep_bytes]
            return
        while len(self._buffer) >= bytes_per_frame:
            payload = bytes(self._buffer[:bytes_per_frame])
            del self._buffer[:bytes_per_frame]
            packet = (payload, self._sample_rate, self._channels, samples_per_frame, source_slot, source_generation)
            for q in list(self._subscribers.values()):
                try:
                    q.put_nowait(packet)
                except queue.Full:
                    self._drop_queue_head(q)
                    try:
                        q.put_nowait(packet)
                    except queue.Full:
                        pass

    def _reset_subscribers_locked(self) -> None:
        for q in self._subscribers.values():
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

    def _reset_audio_state_locked(self) -> None:
        self._audio_ready = False
        self._source_generation += 1
        self._buffer.clear()
        self._reset_subscribers_locked()

    def _close_current_source(self) -> None:
        f = self._current_file
        self._current_file = None
        self._current_inode = None
        self._current_path = None
        self._current_slot = None
        self._read_offset = 0
        self._header_parsed = False
        if f is not None:
            try:
                f.close()
            except Exception:
                pass
        with self._lock:
            self._reset_audio_state_locked()

    def _open_current_source(self, slot: int, path: str) -> bool:
        try:
            f = open(path, "rb", buffering=0)
        except Exception:
            return False
        try:
            st = os.fstat(f.fileno())
            inode = int(getattr(st, "st_ino", 0) or 0)
        except Exception:
            inode = None
        self._current_file = f
        self._current_inode = inode
        self._current_path = path
        self._current_slot = int(slot)
        self._read_offset = 0
        self._header_parsed = False
        with self._lock:
            self._reset_audio_state_locked()
        return True

    def _source_rotated(self) -> bool:
        path = self._current_path
        if not path or self._current_file is None:
            return False
        try:
            st = os.stat(path)
        except Exception:
            return True
        inode = int(getattr(st, "st_ino", 0) or 0)
        size = int(getattr(st, "st_size", 0) or 0)
        if self._current_inode is not None and inode != self._current_inode:
            return True
        return size < self._read_offset

    def _reader_thread(self) -> None:
        while not self._stop_event.is_set():
            try:
                desired_slot = self._selected_slot()
                if desired_slot is None:
                    if self._current_file is not None:
                        self._close_current_source()
                    time.sleep(0.05)
                    continue

                desired_path = self._audio_path_for_slot(desired_slot)
                if desired_slot != self._current_slot or desired_path != self._current_path:
                    if self._current_file is not None:
                        self._close_current_source()

                if self._current_file is None:
                    if not os.path.exists(desired_path):
                        time.sleep(0.05)
                        continue
                    if not self._open_current_source(desired_slot, desired_path):
                        time.sleep(0.10)
                        continue

                if self._current_file is None:
                    time.sleep(0.05)
                    continue

                if not self._header_parsed:
                    try:
                        size = os.path.getsize(desired_path)
                    except Exception:
                        size = 0
                    if size < 44:
                        time.sleep(0.02)
                        continue
                    self._current_file.seek(0)
                    header = self._current_file.read(44)
                    parsed = self._parse_wav_header(header)
                    if parsed is None:
                        time.sleep(0.05)
                        continue
                    sample_rate, channels = parsed
                    self._header_parsed = True
                    self._read_offset = 44
                    self._current_file.seek(self._read_offset)
                    with self._lock:
                        self._sample_rate = sample_rate
                        self._channels = channels
                        self._buffer.clear()
                        self._audio_ready = True
                        self._reset_subscribers_locked()

                chunk = self._current_file.read(4096)
                if chunk:
                    self._read_offset += len(chunk)
                    with self._lock:
                        self._buffer.extend(chunk)
                        max_buffer_bytes = int(self._sample_rate * self._channels * 2 * self._MAX_SOURCE_LATENCY_S)
                        if len(self._buffer) > max_buffer_bytes:
                            drop = len(self._buffer) - max_buffer_bytes
                            sample_bytes = self._channels * 2
                            drop -= drop % max(1, sample_bytes)
                            if drop > 0:
                                del self._buffer[:drop]
                        self._dispatch_ready_frames_locked()
                    continue

                if self._source_rotated():
                    self._close_current_source()
                    time.sleep(0.02)
                    continue
                time.sleep(0.01)
            except Exception:
                self._close_current_source()
                time.sleep(0.10)

    def _start_reader(self) -> None:
        if self._reader_thread_obj is not None and self._reader_thread_obj.is_alive():
            return
        self._reader_thread_obj = threading.Thread(target=self._reader_thread, daemon=True, name="v3-audio-reader")
        self._reader_thread_obj.start()

    def subscribe(self) -> tuple[int, queue.Queue]:
        self._start_reader()
        q: queue.Queue = queue.Queue(maxsize=self._SUBSCRIBER_QUEUE_FRAMES)
        with self._lock:
            subscriber_id = self._next_subscriber_id
            self._next_subscriber_id += 1
            self._subscribers[subscriber_id] = q
            _, bytes_per_frame = self._frame_params_locked()
            keep_bytes = max(bytes_per_frame, self._channels * 2)
            if len(self._buffer) > keep_bytes:
                del self._buffer[:-keep_bytes]
        return subscriber_id, q

    def unsubscribe(self, subscriber_id: Optional[int]) -> None:
        if subscriber_id is None:
            return
        with self._lock:
            self._subscribers.pop(int(subscriber_id), None)

    def get_format(self) -> tuple[int, int]:
        with self._lock:
            return int(self._sample_rate), int(self._channels)

    def get_stream_identity(self) -> tuple[Optional[int], int]:
        with self._lock:
            return self._current_slot, int(self._source_generation)

    def close(self) -> None:
        self._stop_event.set()
        if self._reader_thread_obj is not None:
            self._reader_thread_obj.join(timeout=1.0)
        self._close_current_source()
        with self._lock:
            self._subscribers.clear()


class _PreviewAudioTrack(AudioStreamTrack):
    """WebRTC audio track backed by the selected preview WAV reader."""

    _FRAME_PTIME_S = 0.02
    _QUEUE_WAIT_S = 0.03
    _FRAME_HOLD_S = 0.18

    def __init__(self, source: _PreviewAudioSource):
        super().__init__()
        self._source = source
        self._subscriber_id, self._queue = self._source.subscribe()
        self._active_generation: Optional[int] = None
        self._last_frame_data: Optional[bytes] = None
        self._last_frame_wallclock = 0.0

    def stop(self) -> None:
        self._source.unsubscribe(self._subscriber_id)
        self._subscriber_id = None
        super().stop()

    async def recv(self):
        sample_rate, channels = self._source.get_format()
        samples_per_frame = max(1, int(round(sample_rate * self._FRAME_PTIME_S)))
        if hasattr(self, "_timestamp"):
            self._timestamp += samples_per_frame
            wait = self._start + (self._timestamp / sample_rate) - time.time()
            if wait > 0:
                await asyncio.sleep(wait)
        else:
            self._start = time.time()
            self._timestamp = 0

        latest = None
        try:
            latest = await asyncio.to_thread(self._queue.get, True, self._QUEUE_WAIT_S)
        except queue.Empty:
            latest = None

        while latest is not None:
            try:
                latest = self._queue.get_nowait()
            except queue.Empty:
                break

        if latest is not None:
            frame_data, sample_rate, channels, samples_per_frame, source_slot, source_generation = latest
            current_slot, current_generation = self._source.get_stream_identity()
            if (
                current_slot is None
                or source_slot is None
                or int(source_generation) != int(current_generation)
                or int(source_slot) != int(current_slot)
            ):
                frame_data = b"\x00" * (samples_per_frame * channels * 2)
            elif self._active_generation != int(source_generation):
                self._start = time.time()
                self._timestamp = 0
                self._active_generation = int(source_generation)
            self._last_frame_data = frame_data
            self._last_frame_wallclock = time.time()
        else:
            hold_age = time.time() - float(self._last_frame_wallclock or 0.0)
            if self._last_frame_data is not None and hold_age <= self._FRAME_HOLD_S:
                frame_data = self._last_frame_data
            else:
                frame_data = b"\x00" * (samples_per_frame * channels * 2)

        expected_bytes = samples_per_frame * channels * 2
        if len(frame_data) != expected_bytes:
            if len(frame_data) > expected_bytes:
                frame_data = frame_data[:expected_bytes]
            else:
                frame_data = frame_data + (b"\x00" * (expected_bytes - len(frame_data)))

        layout = "mono" if int(channels) == 1 else "stereo"
        frame = AudioFrame(format="s16", layout=layout, samples=samples_per_frame)
        frame.planes[0].update(frame_data)
        frame.pts = self._timestamp
        frame.sample_rate = sample_rate
        frame.time_base = Fraction(1, sample_rate)
        return frame


class _PreviewAudioWebRTCBridge:
    def __init__(self, metrics_obj, ice_servers: Optional[list[dict[str, Any]]] = None):
        self.metrics = metrics_obj
        self.enabled = bool(_PREVIEW_GAME_AUDIO_TRANSPORT_ENABLED)
        self.ice_servers = list(ice_servers) if ice_servers else list(_AUDIO_WEBRTC_ICE_SERVERS)
        self.error_reason = "" if self.enabled else (_AUDIO_WEBRTC_IMPORT_ERROR or "webrtc_unavailable")
        self._thread: Optional[threading.Thread] = None
        self._loop = None
        self._pcs = set()
        self._lock = threading.Lock()
        self._audio_source = _PreviewAudioSource(self.metrics) if self.enabled else None

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return

        def _runner():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            loop.run_forever()
            try:
                loop.run_until_complete(self._shutdown_async())
            finally:
                loop.close()

        self._thread = threading.Thread(target=_runner, daemon=True, name="v3-preview-audio-rtc")
        self._thread.start()
        for _ in range(200):
            if self._loop is not None:
                return
            time.sleep(0.01)
        self.enabled = False
        self.error_reason = "webrtc_loop_unavailable"

    async def _shutdown_async(self):
        pcs = list(self._pcs)
        self._pcs.clear()
        for pc in pcs:
            try:
                await pc.close()
            except Exception:
                pass

    def stop(self) -> None:
        loop = self._loop
        if loop is not None:
            try:
                fut = asyncio.run_coroutine_threadsafe(self._shutdown_async(), loop)
                fut.result(timeout=3.0)
            except Exception:
                pass
            try:
                loop.call_soon_threadsafe(loop.stop)
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self._thread = None
        self._loop = None
        if self._audio_source is not None:
            self._audio_source.close()
            self._audio_source = None

    def _pc_configuration(self):
        if RTCConfiguration is None or RTCIceServer is None:
            return None
        ice_servers = []
        for item in self.ice_servers:
            if not isinstance(item, dict):
                continue
            urls = item.get("urls")
            if isinstance(urls, str):
                urls = [urls]
            if not isinstance(urls, list) or not urls:
                continue
            kwargs = {"urls": urls}
            username = item.get("username")
            credential = item.get("credential")
            if isinstance(username, str):
                kwargs["username"] = username
            if isinstance(credential, str):
                kwargs["credential"] = credential
            try:
                ice_servers.append(RTCIceServer(**kwargs))
            except Exception:
                continue
        return RTCConfiguration(iceServers=ice_servers)

    async def _create_answer_async(self, offer_sdp: str, offer_type: str):
        if not self.enabled:
            return {"ok": False, "error": self.error_reason or "webrtc_unavailable"}
        pc = RTCPeerConnection(configuration=self._pc_configuration())
        self._pcs.add(pc)
        audio_track = None

        @pc.on("connectionstatechange")
        async def _on_state_change():
            if pc.connectionState in {"closed", "failed", "disconnected"}:
                self._pcs.discard(pc)
                if audio_track is not None:
                    try:
                        audio_track.stop()
                    except Exception:
                        pass
                try:
                    await pc.close()
                except Exception:
                    pass

        if self._audio_source is None:
            self._audio_source = _PreviewAudioSource(self.metrics)
        try:
            audio_track = _PreviewAudioTrack(self._audio_source)
            pc.addTrack(audio_track)
        except Exception as e:
            return {"ok": False, "error": f"audio_track_failed: {e}"}

        await pc.setRemoteDescription(RTCSessionDescription(sdp=offer_sdp, type=offer_type))
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        return {"ok": True, "type": pc.localDescription.type, "sdp": pc.localDescription.sdp}

    def create_answer(self, offer_sdp: str, offer_type: str, timeout_s: float = 10.0) -> dict[str, Any]:
        if not self.enabled or self._loop is None:
            return {"ok": False, "error": self.error_reason or "webrtc_unavailable"}
        try:
            fut = asyncio.run_coroutine_threadsafe(
                self._create_answer_async(offer_sdp, offer_type),
                self._loop,
            )
            return fut.result(timeout=max(1.0, float(timeout_s)))
        except Exception as e:
            return {"ok": False, "error": f"webrtc_offer_failed: {e}"}


# ── Dashboard HTML ──────────────────────────────────────────────────────────


def _render_dashboard_html() -> str:
    return r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Robotron AI v3 — PPO Dashboard</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=DotGothic16&display=swap');
:root {
  --bg0:#040510; --bg1:#0b1433; --bg2:#1a0a33;
  --panel:rgba(6,10,28,0.78);
  --line:rgba(0,229,255,0.26);
  --ink:#e8f6ff; --muted:#9cb6d4;
  --accentA:#00e5ff; --accentB:#ffe600; --accentC:#39ff14; --accentD:#ff2bd6;
  --neonRed:#ff2a55; --neonEdge:rgba(0,229,255,0.65);
  --panelGlowA:rgba(0,229,255,0.22); --panelGlowB:rgba(255,43,214,0.18);
  --vfdCyan:#70f7ff;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg0);color:var(--ink);font:13px/1.5 'Segoe UI',system-ui,sans-serif;overflow-x:hidden;padding-bottom:24px}
a{color:var(--accentA);text-decoration:none}

/* ── Shared base: top bar, metric cards, chart panels ── */
.top,.card,.panel{
  position:relative;overflow:hidden;
  background:
    radial-gradient(120% 160% at 0% 0%,rgba(0,229,255,0.10),transparent 58%),
    radial-gradient(140% 150% at 100% 0%,rgba(255,43,214,0.10),transparent 58%),
    linear-gradient(155deg,rgba(7,12,30,0.93) 0%,rgba(7,10,27,0.87) 100%);
  border:1px solid var(--line);
  box-shadow:inset 0 0 0 1px rgba(0,229,255,0.06),0 0 26px var(--panelGlowA),0 0 36px var(--panelGlowB);
}
.card{
  --card-border:rgba(100,180,255,0.45);
  --card-glow:rgba(100,180,255,0.18);
  border:2px solid var(--card-border);
  box-shadow:inset 0 0 0 1px color-mix(in srgb,var(--card-border) 15%,transparent),0 0 18px var(--card-glow),0 0 32px var(--card-glow);
}

/* ── Pseudo-elements ── */
.top::before,.panel::before{
  content:"";position:absolute;inset:0;border-radius:inherit;padding:1px;
  background:linear-gradient(110deg,rgba(0,229,255,0.95),rgba(57,255,20,0.9),rgba(255,43,214,0.95),rgba(255,230,0,0.9));
  background-size:250% 250%;opacity:0.35;pointer-events:none;
  -webkit-mask:linear-gradient(#000 0 0) content-box,linear-gradient(#000 0 0);
  -webkit-mask-composite:xor;mask-composite:exclude;
}
.card::before{content:"";position:absolute;inset:0;border-radius:inherit;pointer-events:none;opacity:0}
.top::after,.panel::after{
  content:"";position:absolute;inset:0;border-radius:inherit;pointer-events:none;
  background:linear-gradient(180deg,rgba(255,255,255,0.07),transparent 36%);opacity:0.35;
}

/* ── Header ── */
.top{
  padding:14px 18px;display:flex;align-items:center;justify-content:space-between;gap:12px;
  margin:0 10px;border-radius:0 0 14px 14px
}
.title{display:flex;flex-direction:column;gap:3px;position:relative;z-index:2}
.title h1{margin:0;font-size:24px;letter-spacing:0.5px;font-weight:700;color:#f5fbff;text-shadow:0 0 14px rgba(0,229,255,0.45),0 0 28px rgba(57,255,20,0.24)}
.subtitle{color:var(--muted);font-size:13px;text-shadow:0 0 9px rgba(0,229,255,0.14)}
.top-right{display:inline-flex;align-items:center;gap:10px;position:relative;z-index:2}
.status{
  display:inline-flex;align-items:center;gap:9px;font-size:13px;
  border:1px solid rgba(0,229,255,0.33);padding:8px 12px;border-radius:999px;
  background:rgba(2,6,23,0.80);
  box-shadow:inset 0 0 14px rgba(0,229,255,0.18),0 0 16px rgba(0,229,255,0.16)
}
.status.offline{
  border-color:rgba(255,80,90,0.35);
  box-shadow:inset 0 0 14px rgba(255,70,90,0.16),0 0 16px rgba(255,70,90,0.12)
}
.display-fps-box{
  display:inline-flex;flex-direction:column;align-items:center;gap:2px;
  border:1px solid rgba(0,229,255,0.33);padding:4px 12px;border-radius:999px;
  background:rgba(2,6,23,0.80);
  box-shadow:inset 0 0 14px rgba(0,229,255,0.12),0 0 14px rgba(0,229,255,0.10)
}
.display-fps-label{font-size:8px;text-transform:uppercase;letter-spacing:0.8px;color:rgba(0,229,255,0.55)}
.display-fps-value{
  font-family:"LED Dot-Matrix","Dot Matrix","DotGothic16","Courier New",monospace;
  font-size:16px;color:#c8e8ff;line-height:1;
  text-shadow:0 0 5px rgba(100,160,255,0.7),0 0 14px rgba(60,120,255,0.55),0 0 28px rgba(40,80,255,0.4)
}
.dot{
  width:10px;height:10px;border-radius:50%;
  background:var(--accentC);
  box-shadow:0 0 0 7px rgba(57,255,20,0.2),0 0 14px rgba(57,255,20,0.5)
}
.status.offline .dot{
  background:#ff5a6c;
  box-shadow:0 0 0 7px rgba(255,90,108,0.18),0 0 14px rgba(255,90,108,0.46)
}

/* ── Cards grid ── */
.cards{display:grid;grid-template-columns:repeat(12,minmax(0,1fr));gap:10px;padding:12px 10px}
.card{
  grid-column:span 2;border-radius:14px;padding:6px 9px;min-height:44px;
  display:flex;flex-direction:column;justify-content:flex-start;gap:3px;overflow:hidden;
}

/* ── Label / Value (VFD text) ── */
.label{color:#a5bfde;font-size:12px;text-transform:uppercase;letter-spacing:0.8px;text-shadow:0 0 8px rgba(0,229,255,0.18);position:relative;z-index:2}
.value{
  font-size:28px;line-height:1;font-weight:700;letter-spacing:0.35px;color:#f0fbff;
  text-shadow:0 0 10px rgba(0,229,255,0.28),0 0 22px rgba(57,255,20,0.16);
  position:relative;z-index:2;
}
.card:not(.gauge-card) .value{
  font-family:"LED Dot-Matrix","Dot Matrix","DotGothic16","Courier New",monospace;
  color:#c8e8ff;font-weight:400;letter-spacing:normal;font-variant-numeric:normal;
  text-shadow:0 0 5px rgba(100,160,255,0.7),0 0 14px rgba(60,120,255,0.55),0 0 28px rgba(40,80,255,0.45),0 0 48px rgba(30,60,220,0.3);
  filter:drop-shadow(0 0 8px rgba(60,130,255,0.5)) drop-shadow(0 0 18px rgba(40,80,255,0.35));
}
.sub-row{display:flex;gap:8px;flex-wrap:wrap;margin-top:2px}
.sub-row .kv{font-size:11px;color:var(--muted)}
.sub-row .kv b{color:var(--ink);font-weight:600}

/* ── Sparklines ── */
.sparkline{width:100%;height:36px;margin-top:auto}
.sparkline canvas{width:100%;height:100%}

/* ── Gauge cards ── */
.gauge-card{grid-column:span 2;grid-row:span 2;min-height:200px;order:-30}
.gauge-card canvas{border:none;background:transparent;box-shadow:none}
.gauge-head{display:flex;justify-content:space-between;align-items:baseline;gap:8px;padding:0 4px}

/* ── Preview card ── */
.preview-card{grid-column:span 4;grid-row:span 2;min-height:300px}
#preview-canvas{width:100%;max-height:260px;image-rendering:pixelated;background:#000;border-radius:4px;margin-top:4px}
.preview-controls{display:flex;gap:12px;align-items:center;margin-top:2px}
.preview-controls label{font-size:10px;cursor:pointer;color:var(--muted)}
.preview-controls input[disabled]{opacity:.6}

/* ── Game settings / model summary ── */
.game-settings-card{grid-column:span 2}
.game-settings-row{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:6px}
.game-settings-row label{font-size:11px;color:var(--muted);display:flex;gap:6px;align-items:center}
.game-settings-row select{background:rgba(5,12,28,.88);color:var(--ink);border:1px solid rgba(0,229,255,.24);border-radius:6px;padding:4px 8px}
.game-settings-row input[type="checkbox"]{accent-color:var(--accentA)}
.model-summary-wrap{padding:0 10px}
.model-summary-panel{min-height:unset}
.model-summary-box{
  margin-top:8px;padding:12px 14px;border-radius:10px;
  background:rgba(2,10,28,.72);border:1px solid rgba(0,229,255,.18);
  color:#bfe6ff;font:12px/1.45 "SFMono-Regular",Consolas,"Liberation Mono",Menlo,monospace;
  white-space:pre-wrap;
}

/* ── Client table ── */
.client-table-card{grid-column:span 4}
table.clients{width:100%;border-collapse:collapse;font-size:12px}
table.clients th{text-align:left;color:var(--muted);border-bottom:1px solid var(--line);padding:3px 6px;font-weight:500;text-transform:uppercase;font-size:10px;letter-spacing:0.8px}
table.clients td{padding:3px 6px;border-bottom:1px solid rgba(255,255,255,0.04)}
table.clients tr{cursor:pointer;transition:background 0.15s}
table.clients tr:hover{background:rgba(0,229,255,0.06)}
table.clients tr.selected{background:rgba(0,229,255,0.12);border-left:2px solid var(--accentA)}

/* ── Chart panels ── */
.charts{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;padding:0 10px}
.panel{border-radius:14px;padding:12px;min-height:180px;display:grid;grid-template-rows:auto 1fr;gap:4px;overflow:hidden}
.panel h2{margin:0;font-size:14px;font-weight:640;letter-spacing:0.35px;color:#effbff;text-shadow:0 0 10px rgba(0,229,255,0.28),0 0 22px rgba(255,43,214,0.18);position:relative;z-index:2}
.panel canvas{width:100%;height:100%}
.legend{display:flex;gap:10px;flex-wrap:wrap;font-size:11px;color:#9db7d7;position:relative;z-index:2}
.legend span{display:inline-flex;align-items:center;gap:5px}
.sw{width:10px;height:10px;border-radius:999px;display:inline-block}

/* ── Badges ── */
.badge{display:inline-block;font-size:10px;padding:2px 6px;border-radius:3px;margin-left:4px;vertical-align:middle}
.badge-green{background:rgba(57,255,20,0.15);color:var(--accentC)}
.badge-amber{background:rgba(255,230,0,0.15);color:var(--accentB)}

/* ── Responsive ── */
@media(max-width:1300px){
  .cards{grid-template-columns:repeat(8,minmax(0,1fr))}
  .gauge-card{grid-column:span 2;grid-row:span 2;min-height:180px}
  .preview-card{grid-column:span 4;grid-row:span 1;min-height:124px}
  .client-table-card{grid-column:span 4;grid-row:span 1;min-height:124px}
  .charts{grid-template-columns:1fr 1fr}
}
@media(max-width:950px){
  .cards{grid-template-columns:repeat(4,minmax(0,1fr))}
  .charts{grid-template-columns:1fr}
  .top{flex-direction:column;align-items:flex-start}
  .gauge-card{grid-column:span 2;order:-40}
  .preview-card{grid-column:span 4;grid-row:span 1;min-height:150px}
  .client-table-card{grid-column:span 4;grid-row:span 1;min-height:150px}
}
</style>
</head>
<body>

<!-- ── Header ── -->
<header class="top">
  <div class="title">
    <h1>Robotron AI Dashboard</h1>
    <div class="subtitle" id="model-desc">Loading model info…</div>
  </div>
  <div class="top-right">
    <div class="status" id="status-pill"><span class="dot"></span><span id="status-text">Connecting</span></div>
    <div class="display-fps-box"><span class="display-fps-label">Display FPS</span><span class="display-fps-value" id="display-fps">0</span></div>
  </div>
</header>

<!-- ── Cards grid ── -->
<section class="cards">

  <!-- Gauge: FPS -->
  <article class="card gauge-card" style="--card-border:rgba(255,60,60,0.66);--card-glow:rgba(255,40,40,0.26)">
    <div class="gauge-head"><div class="label">FRAMES PER SECOND</div></div>
    <canvas id="cFpsGauge"></canvas>
  </article>

  <!-- Gauge: Steps/sec -->
  <article class="card gauge-card" style="--card-border:rgba(255,220,40,0.66);--card-glow:rgba(255,200,20,0.26)">
    <div class="gauge-head"><div class="label">SAMPLES PER SECOND</div></div>
    <canvas id="cStepGauge"></canvas>
  </article>

  <!-- Reward 1M -->
  <article class="card">
    <div class="label">AVG REWARD 1M</div>
    <div class="value" id="v-rwd1m">&mdash;</div>
    <div class="sub-row">
      <span class="kv">100K: <b id="v-rwd100k">&mdash;</b></span>
      <span class="kv">5M: <b id="v-rwd5m">&mdash;</b></span>
      <span class="kv">Recent: <b id="v-rwd-avg">&mdash;</b></span>
    </div>
    <div class="sparkline"><canvas id="sp-reward"></canvas></div>
  </article>

  <!-- Avg Level -->
  <article class="card">
    <div class="label">AVG LEVEL</div>
    <div class="value" id="v-level">&mdash;</div>
    <div class="sub-row">
      <span class="kv">100K: <b id="v-level100k">&mdash;</b></span>
      <span class="kv">1M: <b id="v-level1m">&mdash;</b></span>
      <span class="kv">Peak: <b id="v-peak-level">&mdash;</b></span>
    </div>
    <div class="sparkline"><canvas id="sp-level"></canvas></div>
  </article>

  <!-- Episode Length -->
  <article class="card">
    <div class="label">EPISODE LENGTH</div>
    <div class="value" id="v-eplen1m">&mdash;</div>
    <div class="sub-row">
      <span class="kv">100K: <b id="v-eplen100k">&mdash;</b></span>
      <span class="kv">Episodes: <b id="v-episodes">&mdash;</b></span>
    </div>
    <div class="sparkline"><canvas id="sp-eplen"></canvas></div>
  </article>

  <!-- Loss -->
  <article class="card">
    <div class="label">LOSS</div>
    <div class="value" id="v-loss">&mdash;</div>
    <div class="sub-row">
      <span class="kv">Policy: <b id="v-pi-loss">&mdash;</b></span>
      <span class="kv">Value: <b id="v-v-loss">&mdash;</b></span>
    </div>
    <div class="sparkline"><canvas id="sp-loss"></canvas></div>
  </article>

  <!-- Entropy -->
  <article class="card">
    <div class="label">ENTROPY</div>
    <div class="value" id="v-entropy">&mdash;</div>
    <div class="sparkline"><canvas id="sp-entropy"></canvas></div>
  </article>

  <!-- Grad Norm -->
  <article class="card">
    <div class="label">GRAD NORM</div>
    <div class="value" id="v-gradnorm">&mdash;</div>
    <div class="sparkline"><canvas id="sp-gradnorm"></canvas></div>
  </article>

  <!-- Exploration -->
  <article class="card">
    <div class="label">EXPLORATION</div>
    <div class="sub-row">
      <span class="kv">Epsilon: <b id="v-epsilon" style="color:var(--accentB)">&mdash;</b></span>
      <span class="kv">Expert: <b id="v-expert" style="color:var(--accentD)">&mdash;</b></span>
      <span class="kv">BC Wt: <b id="v-bc-weight" style="color:var(--accentA)">&mdash;</b></span>
    </div>
    <div class="sub-row">
      <span class="kv">BC Loss: <b id="v-bc-loss">&mdash;</b></span>
    </div>
  </article>

  <!-- Score / Games -->
  <article class="card">
    <div class="label">SCORE / GAMES</div>
    <div class="value" id="v-avg-score">&mdash;</div>
    <div class="sub-row">
      <span class="kv">Peak: <b id="v-peak-score">&mdash;</b></span>
      <span class="kv">Games: <b id="v-game-count">&mdash;</b></span>
    </div>
  </article>

  <!-- Learning Rate -->
  <article class="card">
    <div class="label">LEARNING RATE</div>
    <div class="value" id="v-lr">&mdash;</div>
    <div class="sparkline"><canvas id="sp-lr"></canvas></div>
  </article>

  <!-- GPU 0 -->
  <article class="card">
    <div class="label">GPU 0</div>
    <div class="value" id="v-gpu0">&mdash;</div>
    <div class="sub-row"><span class="kv" id="v-gpu0-name">&mdash;</span></div>
    <div class="sparkline"><canvas id="sp-gpu0"></canvas></div>
  </article>

  <!-- System -->
  <article class="card">
    <div class="label">SYSTEM</div>
    <div class="sub-row">
      <span class="kv">CPU: <b id="v-cpu">&mdash;</b></span>
      <span class="kv">Free Mem: <b id="v-mem">&mdash;</b></span>
      <span class="kv">Free Disk: <b id="v-disk">&mdash;</b></span>
    </div>
  </article>

  <!-- Clients -->
  <article class="card">
    <div class="label">CLIENTS <span class="badge badge-green" id="v-clnt-count">0</span>
      <span class="badge badge-amber" id="v-web-count">0 web</span></div>
    <div class="sub-row">
      <span class="kv">Episodes this run: <b id="v-ep-run">&mdash;</b></span>
    </div>
  </article>

  <!-- Game Settings -->
  <article class="card game-settings-card">
    <div class="label">GAME SETTINGS</div>
    <div class="game-settings-row">
      <label>Start level
        <select id="gs-level"></select>
      </label>
      <label>
        <input type="checkbox" id="gs-advanced" checked/>
        Advanced
      </label>
    </div>
  </article>

  <!-- Preview -->
  <article class="card preview-card" style="--card-border:rgba(0,229,255,0.55);--card-glow:rgba(0,229,255,0.22)">
    <div class="label">GAME PREVIEW
      <span style="float:right;font-size:10px;color:var(--muted)" id="v-preview-info">&mdash;</span>
    </div>
    <div class="preview-controls">
      <label><input type="checkbox" id="preview-enable" checked style="vertical-align:middle"/> Capture</label>
      <label><input type="checkbox" id="hud-enable" checked style="vertical-align:middle"/> HUD</label>
      <label><input type="checkbox" id="preview-audio-enable" style="vertical-align:middle"/> Game Audio</label>
    </div>
    <canvas id="preview-canvas" width="320" height="240"></canvas>
  </article>

  <!-- Clients Table -->
  <article class="card client-table-card" style="--card-border:rgba(57,255,20,0.45);--card-glow:rgba(57,255,20,0.18)">
    <div class="label">CONNECTED CLIENTS</div>
    <table class="clients" id="clients-table">
      <thead><tr>
        <th>ID</th><th>Slot</th><th>Time</th><th>Lives</th>
        <th>Level</th><th>Score</th><th>Preview</th>
      </tr></thead>
      <tbody id="clients-body"></tbody>
    </table>
  </article>

</section>

<!-- ── Chart panels ── -->
<section class="charts">
  <div class="panel">
    <h2>Throughput</h2>
    <div class="legend">
      <span><span class="sw" style="background:#39ff14"></span>FPS</span>
      <span><span class="sw" style="background:#ffe600"></span>Samples/Sec</span>
      <span><span class="sw" style="background:#22d3ee"></span>Avg Lvl (100K)</span>
      <span><span class="sw" style="background:#e879f9"></span>Ep Len (100K)</span>
    </div>
    <canvas id="chart-throughput"></canvas>
  </div>
  <div class="panel">
    <h2>Rewards</h2>
    <div class="legend">
      <span><span class="sw" style="background:#ff073a"></span>100K</span>
      <span><span class="sw" style="background:#39ff14"></span>1M</span>
      <span><span class="sw" style="background:#00e5ff"></span>5M</span>
    </div>
    <canvas id="chart-rewards"></canvas>
  </div>
  <div class="panel">
    <h2>Learning</h2>
    <div class="legend">
      <span><span class="sw" style="background:#39ff14"></span>Loss</span>
      <span><span class="sw" style="background:#ffe600"></span>Grad Norm</span>
      <span><span class="sw" style="background:#00e5ff"></span>BC Loss</span>
    </div>
    <canvas id="chart-learning"></canvas>
  </div>
</section>

<section class="model-summary-wrap">
  <div class="panel model-summary-panel">
    <h2>Model Summary</h2>
    <div class="model-summary-box" id="model-summary-box">Loading model summary…</div>
  </div>
</section>

<audio id="preview-audio" autoplay playsinline></audio>

<script>
"use strict";

// ── State ──
let history = [], sparkData = {};
const SP_LEN = 200;
const spKeys = ["fps","steps_per_sec","rwd_1m","average_level","eplen_1m","loss","entropy","grad_norm","lr","gpu0_util"];
spKeys.forEach(k => sparkData[k] = []);
const PREVIEW_GAME_AUDIO_TRANSPORT_ENABLED = __PREVIEW_GAME_AUDIO_TRANSPORT_ENABLED__;
const WEBRTC_ICE_SERVERS = __WEBRTC_ICE_SERVERS_JSON__;
const previewAudioEl = document.getElementById("preview-audio");
const previewAudioCheckbox = document.getElementById("preview-audio-enable");
const gsAdvancedEl = document.getElementById("gs-advanced");
const gsLevelEl = document.getElementById("gs-level");
const modelSummaryBox = document.getElementById("model-summary-box");
const statusPillEl = document.getElementById("status-pill");
const statusTextEl = document.getElementById("status-text");
const displayFpsEl = document.getElementById("display-fps");
const PREVIEW_AUDIO_PREF_KEY = "robotron_v3_preview_audio_enabled";
let previewAudioPc = null;
let previewAudioStream = null;
let previewAudioConnecting = false;
let previewAudioRetryTimer = null;
let previewAudioEnabled = PREVIEW_GAME_AUDIO_TRANSPORT_ENABLED && (localStorage.getItem(PREVIEW_AUDIO_PREF_KEY) !== "0");
let failedPolls = 0;
let displayFpsFrames = 0;
let displayFpsLastTs = performance.now();

function cid() { return localStorage.getItem("v3_cid") || (localStorage.setItem("v3_cid", Math.random().toString(36).slice(2,10)), localStorage.getItem("v3_cid")); }

function populateLevelOptions() {
  if (!gsLevelEl) return;
  const current = parseInt(gsLevelEl.value || "1", 10);
  gsLevelEl.innerHTML = "";
  for (let lv = 1; lv <= 81; lv += 1) {
    const opt = document.createElement("option");
    opt.value = String(lv);
    opt.textContent = String(lv);
    if (lv === current) opt.selected = true;
    gsLevelEl.appendChild(opt);
  }
}

function postGameSettings(body) {
  return fetch("/api/game_settings", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify(body),
  }).catch(() => {});
}

function updateConnectionStatus() {
  const connected = failedPolls < 2;
  if (statusPillEl) statusPillEl.classList.toggle("offline", !connected);
  if (statusTextEl) statusTextEl.textContent = connected ? "Connected" : "Waiting for data";
}

function tickDisplayFps() {
  displayFpsFrames += 1;
  const now = performance.now();
  const elapsed = now - displayFpsLastTs;
  if (elapsed >= 1000) {
    if (displayFpsEl) displayFpsEl.textContent = String(Math.round((displayFpsFrames * 1000) / elapsed));
    displayFpsFrames = 0;
    displayFpsLastTs = now;
  }
  window.requestAnimationFrame(tickDisplayFps);
}

// ── Gauge constants ──
const GAUGE_MIN_FPS = 0, GAUGE_MAX_FPS = 15000, GAUGE_FPS_RED_MAX = 3000, GAUGE_FPS_YELLOW_MAX = 6000;
const GAUGE_MIN_STEPS = 0, GAUGE_MAX_STEPS = 80000;

// ── Gauge helpers ──
function roundRectPath(ctx, x, y, w, h, r) {
  const rr = Math.max(0, Math.min(r, Math.min(w, h) * 0.5));
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr);
  ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr);
  ctx.arcTo(x, y, x + w, y, rr);
  ctx.closePath();
}

function drawBloomText(ctx, text, x, y, layers, crisp) {
  for (const l of layers) {
    ctx.fillStyle = l.fill; ctx.shadowColor = l.shadow; ctx.shadowBlur = l.blur;
    ctx.fillText(text, x, y); ctx.fillText(text, x, y);
  }
  ctx.fillStyle = crisp.fill; ctx.shadowColor = crisp.shadow; ctx.shadowBlur = crisp.blur;
  ctx.fillText(text, x, y); ctx.shadowBlur = 0;
}

const VFD_BLOOM_ORANGE = {
  layers: [
    { fill:"rgba(180,80,0,0.50)", shadow:"rgba(255,100,0,0.60)", blur:48 },
    { fill:"rgba(220,120,10,0.70)", shadow:"rgba(255,130,10,0.70)", blur:28 },
    { fill:"rgba(240,160,30,0.80)", shadow:"rgba(255,160,20,0.80)", blur:14 },
    { fill:"rgba(255,190,60,0.90)", shadow:"rgba(255,180,40,0.90)", blur:5 },
  ],
  crisp: { fill:"#ffaa33", shadow:"rgba(255,140,20,0.60)", blur:10 },
};
const VFD_BLOOM_BLUE = {
  layers: [
    { fill:"rgba(60,120,255,0.50)", shadow:"rgba(60,130,255,0.80)", blur:52 },
    { fill:"rgba(80,140,255,0.65)", shadow:"rgba(80,150,255,0.85)", blur:30 },
    { fill:"rgba(100,160,255,0.75)", shadow:"rgba(100,170,255,0.90)", blur:16 },
    { fill:"rgba(140,190,255,0.85)", shadow:"rgba(120,180,255,0.95)", blur:6 },
  ],
  crisp: { fill:"#c8e8ff", shadow:"rgba(80,160,255,0.70)", blur:12 },
};

// ── drawStyledGauge ──
function drawStyledGauge(canvas, valueRaw, cfg) {
  if (!canvas) return;
  const width = canvas.clientWidth || 360, height = canvas.clientHeight || 210;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.floor(width * dpr); canvas.height = Math.floor(height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const minV = Number(cfg.min), maxV = Number(cfg.max);
  const spanV = Math.max(1e-9, maxV - minV);
  const clampVal = v => Math.max(minV, Math.min(maxV, Number(v) || 0));
  const value = clampVal(valueRaw);
  const displayVal = Math.max(minV, Math.min(199999, Number(valueRaw) || 0));

  const pad = 2, outerExtent = 1.08, downExtent = 1.00;
  const maxRByW = (width - 2*pad) / (2.0*outerExtent);
  const maxRByH = (height - 2*pad) / (outerExtent + downExtent);
  const radius = Math.max(20, Math.min(maxRByW, maxRByH) * 1.08);
  const cx = width * 0.5, cy = (height * 0.5) + ((outerExtent - downExtent) * radius * 0.5);

  const degToRad = d => (d * Math.PI) / 180.0;
  const startDeg = 135, spanDeg = 270;
  const startRad = degToRad(startDeg), endRad = degToRad(startDeg + spanDeg);
  const valToAngle = v => { const t = (clampVal(v) - minV) / spanV; return startRad + t * (endRad - startRad); };

  // Dial face
  const face = ctx.createRadialGradient(cx, cy - radius*0.5, radius*0.2, cx, cy, radius*1.02);
  face.addColorStop(0.0, "rgba(36,40,46,0.0)"); face.addColorStop(0.55, "rgba(22,25,30,0.0)"); face.addColorStop(1.0, "rgba(12,14,18,0.0)");
  ctx.fillStyle = face; ctx.beginPath(); ctx.arc(cx, cy, radius*1.02, 0, Math.PI*2); ctx.fill();

  // Colored arc
  const arcLW = Math.max(3.5, radius*0.040), arcR = radius*0.92, arcSegs = 120;
  ctx.lineWidth = arcLW; ctx.lineCap = "butt";
  for (let s = 0; s < arcSegs; s++) {
    const t0 = s/arcSegs, t1 = (s+1)/arcSegs, tMid = (t0+t1)*0.5;
    const redEnd = (cfg.red_max - minV)/spanV, yelEnd = (cfg.yellow_max - minV)/spanV;
    let r,g,b;
    if (tMid <= redEnd) { r=255; g=58; b=58; }
    else if (tMid <= yelEnd) { const f=(tMid-redEnd)/Math.max(1e-9,yelEnd-redEnd); r=255; g=Math.round(58+f*(176-58)); b=Math.round(58-f*(58-44)); }
    else { const f=(tMid-yelEnd)/Math.max(1e-9,1.0-yelEnd); r=Math.round(252-f*(252-51)); g=Math.round(176+f*(219-176)); b=Math.round(44+f*(107-44)); }
    const a0 = startRad + t0*(endRad-startRad), a1 = startRad + t1*(endRad-startRad);
    ctx.strokeStyle = `rgba(${r},${g},${b},0.85)`;
    ctx.beginPath(); ctx.arc(cx, cy, arcR, a0, a1+0.005, false); ctx.stroke();
  }

  // Scale ticks
  const minorStep = Math.max(1e-9, Number(cfg.minor_step));
  const majorStep = Math.max(minorStep, Number(cfg.major_step));
  const majorEvery = Math.max(1, Math.round(majorStep/minorStep));
  const tickOuter = radius*0.89, tickMinorInner = radius*0.825, tickMajorInner = radius*0.775;
  const labelRadius = radius*0.66 - (cfg.label_inset||0);
  const tickCount = Math.max(1, Math.round((maxV-minV)/minorStep));

  for (let i = 0; i <= tickCount; i++) {
    const vv = (i >= tickCount) ? maxV : (minV + i*minorStep);
    const a = valToAngle(vv);
    const cosA = Math.cos(a), sinA = Math.sin(a);
    const isMajor = ((i%majorEvery)===0) || i===tickCount;
    const isLastUnaligned = (i===tickCount) && (Math.abs(vv%majorStep)>1e-6) && (Math.abs((vv%majorStep)-majorStep)>1e-6);
    const drawAsMajor = isMajor && !isLastUnaligned;
    const inner = drawAsMajor ? tickMajorInner : tickMinorInner;
    let c = "rgba(236,241,247,0.90)";
    if (drawAsMajor) {
      if (vv <= cfg.red_max) c = "rgba(255,58,58,0.95)";
      else if (vv <= cfg.yellow_max) c = "rgba(252,176,44,0.95)";
      else c = "rgba(51,219,107,0.95)";
    }
    ctx.strokeStyle = c;
    ctx.lineWidth = drawAsMajor ? Math.max(3.0, radius*0.03) : Math.max(0.9, radius*0.008);
    ctx.beginPath();
    ctx.moveTo(cx + tickOuter*cosA, cy + tickOuter*sinA);
    ctx.lineTo(cx + inner*cosA, cy + inner*sinA);
    ctx.stroke();

    // Value labels at major ticks
    const labelEvery = cfg.label_every || 1;
    const majorIdx = Math.round(i/majorEvery);
    const showLabel = drawAsMajor && (i >= tickCount || (majorIdx%labelEvery)===0);
    if (showLabel) {
      const bottomFactor = Math.max(0, sinA);
      const inwardOffset = 8 - bottomFactor*16;
      const radialExtra = (cfg.label_radial_offset && i > 0 && i < tickCount) ? cfg.label_radial_offset : 0;
      let lx = cx + (labelRadius - inwardOffset - radialExtra)*cosA;
      let ly = cy + (labelRadius - inwardOffset - radialExtra)*sinA;
      const lateralPx = cfg.label_lateral || 0;
      if (lateralPx && i > 0 && i < tickCount && Math.abs(cosA) > 0.25) {
        lx += (lx < cx) ? lateralPx : -lateralPx;
      }
      ctx.fillStyle = "rgba(230,235,242,0.82)"; ctx.textBaseline = "middle";
      const normA = ((a%(2*Math.PI))+2*Math.PI)%(2*Math.PI);
      if (Math.abs(cosA) < 0.15) ctx.textAlign = "center";
      else if (normA > Math.PI*0.5 && normA < Math.PI*1.5) ctx.textAlign = "right";
      else ctx.textAlign = "left";
      if (i >= tickCount) {
        ctx.font = `${Math.max(12,Math.round(radius*0.196))}px 'Avenir Next','Segoe UI',sans-serif`;
        ctx.fillText("\u221E", lx, ly);
      } else {
        const lfs = cfg.label_font_scale || 0.11;
        ctx.font = `${Math.max(8,Math.round(radius*lfs))}px 'Avenir Next','Segoe UI',sans-serif`;
        ctx.fillText(vv >= 1000 ? `${Math.round(vv/1000)}K` : `${Math.round(vv)}`, lx, ly);
      }
    }
  }

  // Needle
  const needleAngle = valToAngle(value);
  const nCos = Math.cos(needleAngle), nSin = Math.sin(needleAngle);
  const needleLen = radius*0.84, tailLen = radius*0.10;
  const baseHalfW = Math.max(4.0, radius*0.030);
  const pTipX = cx + needleLen*nCos, pTipY = cy + needleLen*nSin;
  const pTailX = cx - tailLen*nCos, pTailY = cy - tailLen*nSin;
  const perpX = -nSin, perpY = nCos;

  const needlePath = () => { ctx.beginPath(); ctx.moveTo(pTipX,pTipY); ctx.lineTo(pTailX+perpX*baseHalfW,pTailY+perpY*baseHalfW); ctx.lineTo(pTailX-perpX*baseHalfW,pTailY-perpY*baseHalfW); ctx.closePath(); };

  const drawNeedle = () => {
    // Shadow
    ctx.fillStyle = "rgba(0,0,0,0.42)";
    ctx.beginPath(); ctx.moveTo(pTipX+2.5,pTipY+2.5); ctx.lineTo(pTailX+perpX*baseHalfW+2.5,pTailY+perpY*baseHalfW+2.5); ctx.lineTo(pTailX-perpX*baseHalfW+2.5,pTailY-perpY*baseHalfW+2.5); ctx.closePath(); ctx.fill();
    // Glow bloom
    const glowLayers = [
      {blur:64, color:"rgba(255,140,10,0.70)", fill:"rgba(255,120,10,0.06)"},
      {blur:38, color:"rgba(255,160,20,0.80)", fill:"rgba(255,130,10,0.10)"},
      {blur:18, color:"rgba(255,170,40,0.90)", fill:"rgba(255,140,20,0.14)"},
      {blur:7,  color:"rgba(255,190,60,1.00)", fill:"rgba(255,160,30,0.18)"},
    ];
    ctx.save();
    for (const gl of glowLayers) { ctx.shadowColor=gl.color; ctx.shadowBlur=gl.blur; ctx.shadowOffsetX=0; ctx.shadowOffsetY=0; ctx.fillStyle=gl.fill; needlePath(); ctx.fill(); }
    ctx.restore();
    // Crisp fill
    const grad = ctx.createLinearGradient(pTailX,pTailY,pTipX,pTipY);
    grad.addColorStop(0.0,"#c85a00"); grad.addColorStop(0.6,"#ff8a00"); grad.addColorStop(1.0,"#ffc04d");
    ctx.fillStyle = grad; needlePath(); ctx.fill();
  };

  const drawHub = () => {
    const hubOuter = ctx.createRadialGradient(cx-2,cy-2,2,cx,cy,radius*0.16);
    hubOuter.addColorStop(0.0,"rgba(167,174,180,0.98)"); hubOuter.addColorStop(1.0,"rgba(38,44,51,0.98)");
    ctx.fillStyle=hubOuter; ctx.beginPath(); ctx.arc(cx,cy,radius*0.16,0,Math.PI*2); ctx.fill();
    ctx.fillStyle="rgba(8,12,17,0.96)"; ctx.beginPath(); ctx.arc(cx,cy,radius*0.09,0,Math.PI*2); ctx.fill();
  };

  // Center title
  ctx.fillStyle = "rgba(232,236,241,0.85)";
  ctx.font = `700 ${Math.max(13,Math.round(radius*0.16))}px 'Avenir Next','Segoe UI',sans-serif`;
  ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.fillText(cfg.title || "", cx, cy - radius*0.36);

  // Sub-text (odometer)
  const badgeW = radius*1.0, badgeH = radius*0.48;
  const badgeX = cx - badgeW*0.5, badgeY = cy + radius*0.44;

  if (cfg.sub_text) {
    const subFont = `400 ${Math.max(10,Math.round(radius*0.125))}px 'LED Dot-Matrix','Dot Matrix','DotGothic16','Courier New',monospace`;
    ctx.font = subFont; ctx.textAlign = "right"; ctx.textBaseline = "middle";
    const tmMax = ctx.measureText("8,888,888,888");
    const odoPad = radius*0.05;
    const odoW = tmMax.width + odoPad*2, odoH = Math.max(14, Math.round(radius*0.16));
    const odoX = cx - odoW*0.5, odoY = badgeY - odoH - 3;
    const odoR = Math.max(3, radius*0.035);
    const subX = odoX + odoW - odoPad;
    ctx.save(); ctx.shadowBlur = 0;
    const odoColor = cfg.sub_text_color || "amber";
    roundRectPath(ctx, odoX, odoY, odoW, odoH, odoR);
    ctx.fillStyle = "#130A2C"; ctx.fill();
    ctx.strokeStyle = odoColor === "orange" ? "rgba(180,100,20,0.35)" : "rgba(60,130,255,0.35)";
    ctx.lineWidth = 1; ctx.stroke();
    ctx.restore();
    const bloom = (odoColor === "orange") ? VFD_BLOOM_ORANGE : VFD_BLOOM_BLUE;
    drawBloomText(ctx, cfg.sub_text, subX, odoY + odoH*0.5, bloom.layers, bloom.crisp);
  }

  // Badge background
  const badgeFill = ctx.createLinearGradient(0, badgeY, 0, badgeY + badgeH);
  badgeFill.addColorStop(0.0, "rgba(44,10,12,0.95)"); badgeFill.addColorStop(1.0, "rgba(30,6,8,0.98)");
  roundRectPath(ctx, badgeX, badgeY, badgeW, badgeH, Math.max(8, radius*0.08));
  ctx.fillStyle = badgeFill; ctx.fill();

  // LED value
  const valueText = Number(displayVal).toFixed(cfg.decimals ?? 1);
  const ledX = cx, ledY = badgeY + badgeH*0.5;
  const ledScale = valueText.length > 5 ? 0.282 : 0.338;
  const ledFont = `400 ${Math.max(13,Math.round(radius*ledScale))}px 'DS-Digital','LED Dot-Matrix','Dot Matrix','DotGothic16','Courier New',monospace`;
  ctx.font = ledFont; ctx.textAlign = "center"; ctx.textBaseline = "middle";
  // Multi-layer LED bloom
  ctx.fillStyle="rgba(255,20,20,0.45)"; ctx.shadowColor="rgba(255,30,30,0.80)"; ctx.shadowBlur=Math.max(48,radius*0.55); ctx.fillText(valueText,ledX,ledY);
  ctx.fillStyle="rgba(255,30,30,0.55)"; ctx.shadowColor="rgba(255,40,40,0.85)"; ctx.shadowBlur=Math.max(28,radius*0.35); ctx.fillText(valueText,ledX,ledY);
  ctx.fillStyle="rgba(255,40,40,0.65)"; ctx.shadowColor="rgba(255,50,50,0.90)"; ctx.shadowBlur=Math.max(14,radius*0.18); ctx.fillText(valueText,ledX,ledY);
  ctx.fillStyle="rgba(255,50,50,0.80)"; ctx.shadowColor="rgba(255,60,60,0.95)"; ctx.shadowBlur=Math.max(5,radius*0.08); ctx.fillText(valueText,ledX,ledY);
  ctx.fillStyle="rgba(255,52,52,0.98)"; ctx.shadowBlur=0; ctx.fillText(valueText,ledX,ledY);

  drawNeedle();
  drawHub();
}

function drawFpsGauge(canvas, fps, totalFrames) {
  const framesText = totalFrames != null ? Number(totalFrames).toLocaleString() : null;
  drawStyledGauge(canvas, fps, {
    min:GAUGE_MIN_FPS, max:GAUGE_MAX_FPS, red_max:GAUGE_FPS_RED_MAX, yellow_max:GAUGE_FPS_YELLOW_MAX,
    minor_step:1000, major_step:3000, title:"FPS", unit:"FPS", decimals:0,
    label_inset:4, label_lateral:8, label_font_scale:0.088, label_every:1, label_radial_offset:-16,
    sub_text:framesText, sub_text_color:"blue",
  });
}

function drawStepGauge(canvas, samplesPerSec, totalSamples) {
  const samplesText = totalSamples != null ? Number(totalSamples).toLocaleString() : null;
  drawStyledGauge(canvas, samplesPerSec, {
    min:GAUGE_MIN_STEPS, max:GAUGE_MAX_STEPS, red_max:10000, yellow_max:30000,
    minor_step:5000, major_step:10000, title:"SAMP/s", unit:"S/S", decimals:0,
    label_font_scale:0.088, label_radial_offset:-4,
    sub_text:samplesText, sub_text_color:"orange",
  });
}

// ── Sparklines ──
function drawSpark(canvasId, data, color) {
  const el = document.getElementById(canvasId);
  if (!el) return;
  const ctx = el.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const w = el.clientWidth, h = el.clientHeight;
  el.width = w*dpr; el.height = h*dpr;
  ctx.scale(dpr, dpr); ctx.clearRect(0, 0, w, h);
  if (data.length < 2) return;
  let mn = Infinity, mx = -Infinity;
  data.forEach(v => { if (v < mn) mn = v; if (v > mx) mx = v; });
  if (mx - mn < 1e-9) { mn -= 1; mx += 1; }
  const range = mx - mn;
  ctx.strokeStyle = color || "#00e5ff"; ctx.lineWidth = 1.2; ctx.globalAlpha = 0.8;
  ctx.beginPath();
  for (let i = 0; i < data.length; i++) {
    const x = (i/(data.length-1)) * w;
    const y = h - ((data[i]-mn)/range) * (h-4) - 2;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.stroke();
}
function pushSpark(key, val) {
  if (!sparkData[key]) sparkData[key] = [];
  sparkData[key].push(val);
  if (sparkData[key].length > SP_LEN) sparkData[key].shift();
}

// ── Chart panels ──
function drawChart(canvasId, series) {
  const el = document.getElementById(canvasId);
  if (!el) return;
  const ctx = el.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const w = el.clientWidth, h = el.clientHeight;
  el.width = w*dpr; el.height = h*dpr;
  ctx.scale(dpr, dpr); ctx.clearRect(0, 0, w, h);
  series.forEach(s => {
    const data = s.data;
    if (!data || data.length < 2) return;
    let mn = Infinity, mx = -Infinity;
    data.forEach(v => { if (v < mn) mn = v; if (v > mx) mx = v; });
    if (mx - mn < 1e-9) { mn -= 1; mx += 1; }
    const range = mx - mn;
    ctx.strokeStyle = s.color || "#00e5ff"; ctx.lineWidth = 1.2; ctx.globalAlpha = 0.7;
    ctx.beginPath();
    for (let i = 0; i < data.length; i++) {
      const x = (i/(data.length-1)) * w;
      const y = h - ((data[i]-mn)/range) * (h-8) - 4;
      i === 0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
    }
    ctx.stroke();
  });
  ctx.globalAlpha = 1.0;
}

// ── Preview ──
let lastPreviewSeq = -1;
const previewCanvas = document.getElementById("preview-canvas");
const previewCtx = previewCanvas ? previewCanvas.getContext("2d") : null;

function pollPreview() {
  const since = lastPreviewSeq >= 0 ? `&since=${lastPreviewSeq}` : "";
  fetch(`/api/game_preview?cid=${cid()}${since}`)
    .then(r => r.json())
    .then(d => {
      if (d.changed && d.data && d.width && d.height) {
        lastPreviewSeq = d.seq;
        renderRgb565(d.data, d.width, d.height);
      }
    })
    .catch(() => {})
    .finally(() => setTimeout(pollPreview, 200));
}

function renderRgb565(b64, w, h) {
  if (!previewCtx) return;
  const raw = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  previewCanvas.width = w; previewCanvas.height = h;
  const img = previewCtx.createImageData(w, h);
  const px = img.data;
  for (let i = 0, j = 0; i < raw.length - 1; i += 2, j += 4) {
    const v = (raw[i] << 8) | raw[i+1];
    px[j]   = ((v >> 11) & 0x1F) * 255 / 31;
    px[j+1] = ((v >> 5)  & 0x3F) * 255 / 63;
    px[j+2] = (v & 0x1F) * 255 / 31;
    px[j+3] = 255;
  }
  previewCtx.putImageData(img, 0, 0);
}

function closePreviewAudioConnection() {
  if (previewAudioRetryTimer) {
    clearTimeout(previewAudioRetryTimer);
    previewAudioRetryTimer = null;
  }
  if (previewAudioPc) {
    try { previewAudioPc.close(); } catch (_) {}
    previewAudioPc = null;
  }
  previewAudioStream = null;
  if (previewAudioEl) {
    try { previewAudioEl.pause(); } catch (_) {}
    previewAudioEl.srcObject = null;
    previewAudioEl.muted = true;
  }
}

function schedulePreviewAudioRetry(delayMs) {
  if (!previewAudioEnabled || !PREVIEW_GAME_AUDIO_TRANSPORT_ENABLED) return;
  if (previewAudioRetryTimer) return;
  previewAudioRetryTimer = setTimeout(() => {
    previewAudioRetryTimer = null;
    ensurePreviewAudioConnection();
  }, Math.max(500, delayMs || 1500));
}

async function startPreviewAudioConnection() {
  if (!previewAudioEnabled || !PREVIEW_GAME_AUDIO_TRANSPORT_ENABLED || !window.RTCPeerConnection || !previewAudioEl) return false;
  if (previewAudioConnecting || previewAudioPc) return true;
  previewAudioConnecting = true;
  const pc = new RTCPeerConnection({
    iceServers: Array.isArray(WEBRTC_ICE_SERVERS) && WEBRTC_ICE_SERVERS.length
      ? WEBRTC_ICE_SERVERS
      : [{ urls: ["stun:stun.l.google.com:19302"] }],
  });
  previewAudioPc = pc;

  pc.ontrack = (ev) => {
    const stream = ev.streams && ev.streams[0]
      ? ev.streams[0]
      : new MediaStream(ev.track ? [ev.track] : []);
    previewAudioStream = stream;
    previewAudioEl.srcObject = stream;
    previewAudioEl.muted = !previewAudioEnabled;
    previewAudioEl.volume = previewAudioEnabled ? 1.0 : 0.0;
    const playPromise = previewAudioEl.play();
    if (playPromise && typeof playPromise.catch === "function") playPromise.catch(() => {});
  };

  pc.onconnectionstatechange = () => {
    const st = pc.connectionState || "";
    if (st === "failed" || st === "closed" || st === "disconnected") {
      if (previewAudioPc === pc) previewAudioPc = null;
      if (previewAudioEnabled) schedulePreviewAudioRetry(1500);
    }
  };

  try {
    pc.addTransceiver("audio", { direction: "recvonly" });
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    const res = await fetch("/api/game_preview_offer", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({
        sdp: pc.localDescription ? pc.localDescription.sdp : offer.sdp,
        type: pc.localDescription ? pc.localDescription.type : offer.type,
      }),
    });
    const ans = await res.json();
    if (!res.ok || !ans || !ans.ok || !ans.sdp || !ans.type) {
      throw new Error(ans && ans.error ? ans.error : "audio offer rejected");
    }
    await pc.setRemoteDescription({ type: ans.type, sdp: ans.sdp });
    return true;
  } catch (_) {
    if (previewAudioPc === pc) previewAudioPc = null;
    try { pc.close(); } catch (_) {}
    schedulePreviewAudioRetry(2000);
    return false;
  } finally {
    previewAudioConnecting = false;
  }
}

function ensurePreviewAudioConnection() {
  if (!previewAudioEnabled || !PREVIEW_GAME_AUDIO_TRANSPORT_ENABLED) {
    closePreviewAudioConnection();
    return;
  }
  if (!previewAudioPc && !previewAudioConnecting) {
    startPreviewAudioConnection();
  } else if (previewAudioEl) {
    previewAudioEl.muted = false;
    previewAudioEl.volume = 1.0;
  }
}

// ── Client table ──
document.getElementById("clients-body").addEventListener("click", function(e) {
  const tr = e.target.closest("tr");
  if (!tr || !tr.dataset.cid) return;
  fetch("/api/preview_settings", {
    method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify({client_id: parseInt(tr.dataset.cid)})
  }).catch(() => {});
});

// ── Preview / HUD checkboxes ──
document.getElementById("preview-enable").addEventListener("change", function() {
  fetch("/api/preview_settings", {
    method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify({enabled: this.checked})
  }).catch(() => {});
});
document.getElementById("hud-enable").addEventListener("change", function() {
  fetch("/api/preview_settings", {
    method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify({hud_enabled: this.checked})
  }).catch(() => {});
});
if (previewAudioCheckbox) {
  previewAudioCheckbox.checked = !!previewAudioEnabled;
  previewAudioCheckbox.disabled = !PREVIEW_GAME_AUDIO_TRANSPORT_ENABLED || !window.RTCPeerConnection;
  previewAudioCheckbox.addEventListener("change", function() {
    previewAudioEnabled = !!this.checked;
    localStorage.setItem(PREVIEW_AUDIO_PREF_KEY, previewAudioEnabled ? "1" : "0");
    if (previewAudioEnabled) {
      ensurePreviewAudioConnection();
    } else {
      closePreviewAudioConnection();
    }
  });
}
populateLevelOptions();
if (gsAdvancedEl) {
  gsAdvancedEl.addEventListener("change", function() {
    postGameSettings({start_advanced: !!this.checked});
  });
}
if (gsLevelEl) {
  gsLevelEl.addEventListener("change", function() {
    postGameSettings({start_level_min: parseInt(this.value, 10) || 1});
  });
}

// ── Formatting ──
function fmt(v, d) { return v != null ? Number(v).toFixed(d || 0) : "\u2014"; }
function fmtK(v) { return v >= 1e6 ? (v/1e6).toFixed(1)+"M" : v >= 1e3 ? (v/1e3).toFixed(1)+"K" : String(v); }
function fmtPct(v) { return v != null ? (v*100).toFixed(1)+"%" : "\u2014"; }
function fmtSci(v) { return v != null ? Number(v).toExponential(1) : "\u2014"; }
function hField(field) { return history.map(h => h[field] || 0); }

// ── Main update ──
function update(d) {
  const mdEl = document.getElementById("model-desc");
  if (mdEl && d.model_desc) mdEl.textContent = d.model_desc;
  if (modelSummaryBox && d.model_summary_text) modelSummaryBox.textContent = d.model_summary_text;
  if (gsAdvancedEl && d.game_settings) gsAdvancedEl.checked = !!d.game_settings.start_advanced;
  if (gsLevelEl && d.game_settings && d.game_settings.start_level_min != null) {
    const nextLv = String(parseInt(d.game_settings.start_level_min, 10) || 1);
    if (gsLevelEl.value !== nextLv) gsLevelEl.value = nextLv;
  }

  // Gauges
  drawFpsGauge(document.getElementById("cFpsGauge"), d.fps, d.frame_count);
  drawStepGauge(document.getElementById("cStepGauge"), d.steps_per_sec, d.training_steps);

  // Values
  document.getElementById("v-rwd1m").textContent = fmt(d.rwd_1m, 2);
  document.getElementById("v-rwd100k").textContent = fmt(d.rwd_100k, 2);
  document.getElementById("v-rwd5m").textContent = fmt(d.rwd_5m, 2);
  document.getElementById("v-rwd-avg").textContent = fmt(d.reward_avg, 2);
  document.getElementById("v-level").textContent = fmt(d.average_level, 1);
  document.getElementById("v-level100k").textContent = fmt(d.level_100k, 1);
  document.getElementById("v-level1m").textContent = fmt(d.level_1m, 1);
  document.getElementById("v-peak-level").textContent = fmt(d.peak_level, 1);
  document.getElementById("v-eplen1m").textContent = fmt(d.eplen_1m, 0);
  document.getElementById("v-eplen100k").textContent = fmt(d.eplen_100k, 0);
  document.getElementById("v-episodes").textContent = fmtK(d.episodes_this_run);
  document.getElementById("v-loss").textContent = fmt(d.loss, 6);
  document.getElementById("v-pi-loss").textContent = fmt(d.policy_loss, 5);
  document.getElementById("v-v-loss").textContent = fmt(d.value_loss, 5);
  document.getElementById("v-entropy").textContent = fmt(d.entropy, 4);
  document.getElementById("v-gradnorm").textContent = fmt(d.grad_norm, 3);
  document.getElementById("v-epsilon").textContent = fmtPct(d.epsilon);
  document.getElementById("v-expert").textContent = fmtPct(d.expert_ratio);
  document.getElementById("v-bc-weight").textContent = fmt(d.bc_weight, 3);
  document.getElementById("v-bc-loss").textContent = fmt(d.bc_loss, 5);
  document.getElementById("v-avg-score").textContent = fmt(d.avg_game_score, 0);
  document.getElementById("v-peak-score").textContent = d.peak_game_score != null ? Number(d.peak_game_score).toLocaleString() : "\u2014";
  document.getElementById("v-game-count").textContent = fmtK(d.game_count);
  document.getElementById("v-lr").textContent = fmtSci(d.lr);
  document.getElementById("v-gpu0").textContent = fmt(d.gpu0_util, 0) + "%";
  document.getElementById("v-gpu0-name").textContent = d.gpu0_name || "\u2014";
  document.getElementById("v-cpu").textContent = fmt(d.cpu_pct, 0) + "%";
  document.getElementById("v-mem").textContent = fmt(d.mem_free_gb, 1) + " GB";
  document.getElementById("v-disk").textContent = fmt(d.disk_free_gb, 0) + " GB";
  document.getElementById("v-clnt-count").textContent = d.client_count || 0;
  document.getElementById("v-web-count").textContent = (d.web_client_count || 0) + " web";
  document.getElementById("v-ep-run").textContent = fmtK(d.episodes_this_run);

  // Preview info
  const pvInfo = document.getElementById("v-preview-info");
  if (pvInfo) {
    pvInfo.textContent = d.game_preview_width > 0
      ? d.game_preview_width + "\u00d7" + d.game_preview_height + " " + (d.game_preview_source_format||"") + " " + fmt(d.game_preview_fps,0) + "fps"
      : "No preview";
  }
  document.getElementById("preview-enable").checked = d.preview_capture_enabled !== false;
  document.getElementById("hud-enable").checked = d.hud_enabled !== false;
  if (previewAudioCheckbox) previewAudioCheckbox.checked = !!previewAudioEnabled;
  if (previewAudioEnabled) ensurePreviewAudioConnection();

  // Sparklines
  pushSpark("fps", d.fps);
  pushSpark("steps_per_sec", d.steps_per_sec);
  pushSpark("rwd_1m", d.rwd_1m);
  pushSpark("average_level", d.average_level);
  pushSpark("eplen_1m", d.eplen_1m);
  pushSpark("loss", d.loss);
  pushSpark("entropy", d.entropy);
  pushSpark("grad_norm", d.grad_norm);
  pushSpark("lr", d.lr);
  pushSpark("gpu0_util", d.gpu0_util);

  drawSpark("sp-reward", sparkData.rwd_1m, "#00e5ff");
  drawSpark("sp-level", sparkData.average_level, "#ff073a");
  drawSpark("sp-eplen", sparkData.eplen_1m, "#bf40ff");
  drawSpark("sp-loss", sparkData.loss, "#39ff14");
  drawSpark("sp-entropy", sparkData.entropy, "#ffe600");
  drawSpark("sp-gradnorm", sparkData.grad_norm, "#ff073a");
  drawSpark("sp-lr", sparkData.lr, "#00e5ff");
  drawSpark("sp-gpu0", sparkData.gpu0_util, "#39ff14");

  // Charts
  if (history.length > 5) {
    drawChart("chart-throughput", [
      {data:hField("fps"),color:"#39ff14"},
      {data:hField("steps_per_sec"),color:"#ffe600"},
      {data:hField("level_100k"),color:"#22d3ee"},
      {data:hField("eplen_100k"),color:"#e879f9"},
    ]);
    drawChart("chart-rewards", [{data:hField("rwd_100k"),color:"#ff073a"},{data:hField("rwd_1m"),color:"#39ff14"},{data:hField("rwd_5m"),color:"#00e5ff"}]);
    drawChart("chart-learning", [{data:hField("loss"),color:"#39ff14"},{data:hField("grad_norm"),color:"#ffe600"},{data:hField("bc_loss"),color:"#00e5ff"}]);
  }

  // Client table
  const tbody = document.getElementById("clients-body");
  const rows = d.client_rows || [];
  tbody.innerHTML = "";
  rows.forEach(c => {
    const tr = document.createElement("tr");
    tr.dataset.cid = c.client_id;
    if (c.selected_preview) tr.classList.add("selected");
    const dur = c.duration_seconds > 60 ? fmt(c.duration_seconds/60,1)+"m" : fmt(c.duration_seconds,0)+"s";
    tr.innerHTML = `<td>${c.client_id}</td><td>${c.client_slot}</td><td>${dur}</td>` +
      `<td>${c.lives}</td><td>${c.level}</td><td>${c.score.toLocaleString()}</td>` +
      `<td>${c.selected_preview ? "\u25cf" : (c.preview_capable ? "\u25cb" : "\u2014")}</td>`;
    tbody.appendChild(tr);
  });
}

// ── Polling ──
function poll() {
  fetch("/api/now?cid=" + cid())
    .then(r => r.json())
    .then(d => {
      failedPolls = 0;
      updateConnectionStatus();
      history.push(d);
      if (history.length > 2000) history = history.slice(-1500);
      update(d);
    })
    .catch(() => {
      failedPolls += 1;
      updateConnectionStatus();
    })
    .finally(() => setTimeout(poll, 1000));
}

fetch("/api/history?cid=" + cid())
  .then(r => r.json())
  .then(d => {
    failedPolls = 0;
    updateConnectionStatus();
    if (d.history) history = d.history;
    if (d.now) update(d.now);
    setTimeout(poll, 1000);
  })
  .catch(() => {
    failedPolls += 1;
    updateConnectionStatus();
    setTimeout(poll, 1000);
  });

updateConnectionStatus();
window.requestAnimationFrame(tickDisplayFps);
setTimeout(pollPreview, 500);
</script>
</body>
</html>""".replace(
        "__PREVIEW_GAME_AUDIO_TRANSPORT_ENABLED__",
        "true" if _PREVIEW_GAME_AUDIO_TRANSPORT_ENABLED else "false",
    ).replace(
        "__WEBRTC_ICE_SERVERS_JSON__",
        json.dumps(_AUDIO_WEBRTC_ICE_SERVERS),
    )


def _make_handler(state: _DashboardState, rtc_bridge: Optional[_PreviewAudioWebRTCBridge] = None):
    page = _render_dashboard_html().encode("utf-8")

    class DashboardHandler(BaseHTTPRequestHandler):
        def _send(self, payload: bytes, content_type: str = "text/plain",
                  status: int = 200, cache_control: str = "no-store"):
            try:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", cache_control)
                self.end_headers()
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path
            query = parse_qs(parsed.query)
            client_id = (query.get("cid") or [None])[0]
            if path in ("/api/ping", "/api/now", "/api/history", "/api/game_preview"):
                state.touch_web_client(client_id)
            if path == "/":
                self._send(page, "text/html; charset=utf-8")
                return
            if path == "/api/ping":
                body = json.dumps({"ok": True, "ts": time.time()}).encode("utf-8")
                self._send(body, "application/json")
                return
            if path == "/api/now":
                self._send(state.now_body(), "application/json")
                return
            if path == "/api/history":
                body = json.dumps(state.payload()).encode("utf-8")
                self._send(body, "application/json")
                return
            if path == "/api/game_preview":
                since_seq = None
                try:
                    if "since" in query:
                        since_seq = int((query.get("since") or [0])[0])
                except Exception:
                    pass
                self._send(state.game_preview_body(since_seq=since_seq), "application/json")
                return
            if path == "/api/game_settings":
                agent = state.agent
                expert_pct = -1
                epsilon_pct = -1
                if agent:
                    if agent.is_expert_overridden():
                        expert_pct = int(round(agent.get_expert_ratio() * 100))
                    if agent.is_epsilon_overridden():
                        epsilon_pct = int(round(agent.get_epsilon() * 100))
                body = json.dumps({
                    "start_advanced": GAME_SETTINGS.start_advanced,
                    "start_level_min": GAME_SETTINGS.start_level_min,
                    "expert_pct": expert_pct,
                    "epsilon_pct": epsilon_pct,
                }).encode("utf-8")
                self._send(body, "application/json")
                return
            self._send(b"Not Found", "text/plain; charset=utf-8", status=404)

        def do_POST(self):
            parsed = urlparse(self.path)
            path = parsed.path
            if path == "/api/game_preview_offer":
                try:
                    length = int(self.headers.get("Content-Length", 0))
                    raw = self.rfile.read(length) if length > 0 else b"{}"
                    data = json.loads(raw)
                    sdp = str(data.get("sdp", "") or "")
                    typ = str(data.get("type", "offer") or "offer")
                    if not sdp:
                        raise ValueError("missing_sdp")
                    if rtc_bridge is None:
                        raise RuntimeError("webrtc_bridge_unavailable")
                    answer = rtc_bridge.create_answer(sdp, typ, timeout_s=10.0)
                    body = json.dumps(answer).encode("utf-8")
                    status = 200 if bool(answer.get("ok")) else 503
                    self._send(body, "application/json", status=status)
                except Exception as e:
                    body = json.dumps({"ok": False, "error": str(e)}).encode("utf-8")
                    self._send(body, "application/json", status=400)
                return
            if path == "/api/game_settings":
                try:
                    length = int(self.headers.get("Content-Length", 0))
                    raw = self.rfile.read(length) if length > 0 else b"{}"
                    data = json.loads(raw)
                    if "start_advanced" in data:
                        GAME_SETTINGS.start_advanced = bool(data["start_advanced"])
                    if "start_level_min" in data:
                        GAME_SETTINGS.start_level_min = int(data["start_level_min"])
                    agent = state.agent
                    if agent:
                        if "expert_pct" in data:
                            pct = int(data["expert_pct"])
                            if pct < 0:
                                agent.restore_natural_expert_ratio()
                            else:
                                pct = max(0, min(100, pct))
                                with agent._override_lock:
                                    agent._manual_expert_ratio = pct / 100.0
                                    agent.manual_expert_override = True
                                    agent.override_expert = False
                                    agent.expert_mode = False
                        if "epsilon_pct" in data:
                            pct = int(data["epsilon_pct"])
                            if pct < 0:
                                agent.restore_natural_epsilon()
                            else:
                                pct = max(0, min(100, pct))
                                with agent._override_lock:
                                    agent._manual_epsilon = pct / 100.0
                                    agent.manual_epsilon_override = True
                    GAME_SETTINGS.save()
                    resp = {
                        "start_advanced": GAME_SETTINGS.start_advanced,
                        "start_level_min": GAME_SETTINGS.start_level_min,
                    }
                    if agent:
                        resp["expert_pct"] = int(round(agent.get_expert_ratio() * 100)) if agent.is_expert_overridden() else -1
                        resp["epsilon_pct"] = int(round(agent.get_epsilon() * 100)) if agent.is_epsilon_overridden() else -1
                    body = json.dumps(resp).encode("utf-8")
                    self._send(body, "application/json")
                except Exception:
                    self._send(b'{"error":"bad request"}', "application/json", status=400)
                return
            if path == "/api/preview_settings":
                try:
                    length = int(self.headers.get("Content-Length", 0))
                    raw = self.rfile.read(length) if length > 0 else b"{}"
                    data = json.loads(raw)
                    updates = {}
                    if "enabled" in data:
                        with state.metrics.lock:
                            state.metrics.preview_capture_enabled = bool(data["enabled"])
                            updates["enabled"] = bool(data["enabled"])
                    if "hud_enabled" in data:
                        with state.metrics.lock:
                            state.metrics.hud_enabled = bool(data["hud_enabled"])
                            updates["hud_enabled"] = bool(data["hud_enabled"])
                    if "client_id" in data:
                        target_cid = data.get("client_id")
                        if target_cid in ("", None):
                            target_cid = None
                        else:
                            target_cid = int(target_cid)
                            if target_cid < 0:
                                target_cid = None
                        srv = getattr(state.metrics, "global_server", None)
                        if srv is None:
                            raise RuntimeError("server unavailable")
                        ok, selected = srv.set_preview_client(target_cid)
                        if not ok:
                            raise ValueError("invalid client")
                        updates["client_id"] = -1 if selected is None else int(selected)
                    if updates:
                        body = json.dumps({"ok": True, **updates}).encode("utf-8")
                        self._send(body, "application/json")
                    else:
                        self._send(b'{"error":"missing field"}', "application/json", status=400)
                except Exception:
                    self._send(b'{"error":"bad request"}', "application/json", status=400)
                return
            self._send(b"Not Found", "text/plain; charset=utf-8", status=404)

        def log_message(self, fmt, *args):
            return

        def handle_one_request(self):
            try:
                super().handle_one_request()
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True

    return DashboardHandler


# ── Dashboard server ────────────────────────────────────────────────────────

class MetricsDashboard:
    """Managed dashboard server + optional browser window."""

    def __init__(
        self,
        metrics_obj,
        agent_obj=None,
        host: str = "127.0.0.1",
        port: int = 8796,
        sample_interval: float = 0.10,
        history_limit: int = DASH_HISTORY_LIMIT,
        open_browser: bool = True,
    ):
        self.metrics = metrics_obj
        self.agent = agent_obj
        self.host = host
        self.port = port
        self.sample_interval = max(0.033, sample_interval)
        self.open_browser = open_browser

        self.state = _DashboardState(metrics_obj, agent_obj, history_limit=history_limit)
        self.rtc_bridge = _PreviewAudioWebRTCBridge(metrics_obj, _AUDIO_WEBRTC_ICE_SERVERS)
        self.stop_event = threading.Event()
        self.httpd: Optional[ThreadingHTTPServer] = None
        self.server_thread: Optional[threading.Thread] = None
        self.sampler_thread: Optional[threading.Thread] = None
        self.browser_proc: Optional[subprocess.Popen] = None
        self.browser_profile_dir: Optional[str] = None
        self.url: Optional[str] = None
        self._closed = False
        self._lock = threading.Lock()
        atexit.register(self.stop)

    def _sampling_loop(self):
        while not self.stop_event.is_set():
            try:
                self.state.sample()
            except Exception:
                pass
            self.stop_event.wait(self.sample_interval)

    def _bind_server(self):
        handler_cls = _make_handler(self.state, self.rtc_bridge)
        last_err = None
        for p in range(self.port, self.port + 40):
            try:
                self.httpd = ThreadingHTTPServer((self.host, p), handler_cls)
                self.port = p
                return
            except OSError as e:
                last_err = e
        if last_err:
            raise last_err
        raise OSError("Could not bind dashboard server")

    @staticmethod
    def _resolve_browser_binary(candidate: str) -> Optional[str]:
        if os.path.isabs(candidate):
            return candidate if os.path.exists(candidate) else None
        return shutil.which(candidate)

    def _launch_browser(self, url: str):
        candidates = [
            ("google-chrome", True), ("google-chrome-stable", True),
            ("chromium", True), ("chromium-browser", True),
            ("brave-browser", True), ("msedge", True),
        ]
        for candidate, chromium_like in candidates:
            binary = self._resolve_browser_binary(candidate)
            if not binary:
                continue
            cmd = [binary]
            profile_dir = None
            if chromium_like:
                profile_dir = tempfile.mkdtemp(prefix="robotron_v3_dash_")
                cmd.extend([
                    "--new-window", f"--app={url}",
                    f"--user-data-dir={profile_dir}",
                    "--no-first-run", "--disable-features=TranslateUI",
                    "--no-default-browser-check",
                ])
            else:
                cmd.extend(["--new-window", url])
            try:
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    start_new_session=(os.name != "nt"),
                )
                self.browser_proc = proc
                self.browser_profile_dir = profile_dir
                return
            except Exception:
                if profile_dir:
                    shutil.rmtree(profile_dir, ignore_errors=True)
        try:
            webbrowser.open_new(url)
        except Exception:
            pass

    def start(self) -> str:
        self.state.sample()
        self.rtc_bridge.start()
        self._bind_server()
        self.url = f"http://{self.host}:{self.port}"

        self.server_thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.server_thread.start()

        self.sampler_thread = threading.Thread(target=self._sampling_loop, daemon=True)
        self.sampler_thread.start()

        if self.open_browser:
            self._launch_browser(self.url)

        print(f"Dashboard: {self.url}")
        return self.url

    def stop(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self.stop_event.set()
        if self.httpd:
            try:
                self.httpd.shutdown()
            except Exception:
                pass
            self.httpd = None
        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(timeout=2.0)
        self.server_thread = None
        if self.sampler_thread and self.sampler_thread.is_alive():
            self.sampler_thread.join(timeout=1.0)
        self.sampler_thread = None
        try:
            if self.rtc_bridge is not None:
                self.rtc_bridge.stop()
        except Exception:
            pass
        if self.browser_proc and self.browser_proc.poll() is None:
            try:
                os.killpg(self.browser_proc.pid, signal.SIGTERM)
                self.browser_proc.wait(timeout=2.0)
            except Exception:
                try:
                    os.killpg(self.browser_proc.pid, signal.SIGKILL)
                except Exception:
                    pass
        self.browser_proc = None
        if self.browser_profile_dir:
            shutil.rmtree(self.browser_profile_dir, ignore_errors=True)
            self.browser_profile_dir = None
