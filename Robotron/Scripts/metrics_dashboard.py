#!/usr/bin/env python3
# ==================================================================================================================
# ||  ROBOTRON AI v2 • LIVE DASHBOARD                                                                            ||
# ||  Lightweight Grafana-style metrics dashboard served locally and managed by the Python app lifecycle.         ||
# ==================================================================================================================
"""Live dashboard for Robotron AI metrics."""

if __name__ == "__main__":
    print("This module is launched from main.py")
    raise SystemExit(1)

import atexit
import asyncio
import base64
import json
import math
import mimetypes
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import webbrowser
import sys
import queue
from fractions import Fraction
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse


def _env_flag(name: str, default: bool = False) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return bool(default)
    return raw not in {"0", "false", "off", "no"}


_PREVIEW_GAME_AUDIO_TRANSPORT_ENABLED = _env_flag("ROBOTRON_GAME_AUDIO_ENABLED", True)

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except Exception:
    np = None
    _NUMPY_AVAILABLE = False

try:
    from aiortc import (
        RTCPeerConnection,
        RTCSessionDescription,
        RTCConfiguration,
        RTCIceServer,
        VideoStreamTrack,
        AudioStreamTrack,
    )
    from av import VideoFrame, AudioFrame
    _WEBRTC_AVAILABLE = True
    _WEBRTC_IMPORT_ERROR = ""
except Exception as e:
    RTCPeerConnection = None
    RTCSessionDescription = None
    RTCConfiguration = None
    RTCIceServer = None
    VideoStreamTrack = object
    AudioStreamTrack = object
    VideoFrame = None
    AudioFrame = None
    _WEBRTC_AVAILABLE = False
    _WEBRTC_IMPORT_ERROR = f"{type(e).__name__}: {e}"

try:
    from config import RL_CONFIG, plateau_pulser, PlateauPulser, game_settings, WEBRTC_ICE_SERVERS
except ImportError:
    from Scripts.config import RL_CONFIG, plateau_pulser, PlateauPulser, game_settings, WEBRTC_ICE_SERVERS

try:
    from metrics_display import get_dqn_window_averages, get_total_window_averages, get_eplen_1m_average, get_eplen_100k_average
except ImportError:
    try:
        from Scripts.metrics_display import get_dqn_window_averages, get_total_window_averages, get_eplen_1m_average, get_eplen_100k_average
    except ImportError:
        def get_dqn_window_averages():
            return 0.0, 0.0, 0.0
        def get_total_window_averages():
            return 0.0, 0.0, 0.0
        def get_eplen_1m_average():
            return 0.0
        def get_eplen_100k_average():
            return 0.0

try:
    from chat_store import ChatStore
except ImportError:
    from Scripts.chat_store import ChatStore


def _tail_mean(values, count: int = 20) -> float:
    if not values:
        return 0.0
    tail = list(values)[-count:]
    if not tail:
        return 0.0
    return float(sum(tail) / max(1, len(tail)))


LEVEL_25K_FRAMES = 25_000
LEVEL_100K_FRAMES = 100_000
LEVEL_1M_FRAMES = 1_000_000
LEVEL_5M_FRAMES = 5_000_000
WEB_CLIENT_TIMEOUT_S = 5.0
DASH_HISTORY_LIMIT = 40_000
AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".m4a", ".aac", ".flac"}
FONT_EXTENSIONS = {".ttf", ".otf", ".woff", ".woff2"}
VIDEO_EXTENSIONS = {".mov", ".mp4", ".webm", ".ogv"}
PREVIEW_HTTP_DEFAULT_MAX_W = 320
PREVIEW_HTTP_DEFAULT_MAX_H = 240
PREVIEW_HTTP_MOBILE_MAX_W = 192
PREVIEW_HTTP_MOBILE_MAX_H = 144


_FALLBACK_ICE_SERVERS = [{"urls": ["stun:stun.l.google.com:19302"]}]
_WEBRTC_ICE_SERVERS = WEBRTC_ICE_SERVERS if isinstance(WEBRTC_ICE_SERVERS, list) and WEBRTC_ICE_SERVERS else _FALLBACK_ICE_SERVERS


def _normalize_preview_profile(profile: str | None) -> str:
    raw = str(profile or "default").strip().lower()
    return "mobile" if raw == "mobile" else "default"


def _preview_http_limits(profile: str | None) -> tuple[int, int]:
    profile_norm = _normalize_preview_profile(profile)
    if profile_norm == "mobile":
        return PREVIEW_HTTP_MOBILE_MAX_W, PREVIEW_HTTP_MOBILE_MAX_H
    return PREVIEW_HTTP_DEFAULT_MAX_W, PREVIEW_HTTP_DEFAULT_MAX_H


def _downscale_rgb565_nearest(raw: bytes, width: int, height: int, max_width: int, max_height: int) -> tuple[bytes, int, int]:
    w = max(0, int(width))
    h = max(0, int(height))
    mw = max(0, int(max_width))
    mh = max(0, int(max_height))
    if not raw or w <= 0 or h <= 0:
        return b"", 0, 0
    if (mw <= 0 and mh <= 0) or ((mw <= 0 or w <= mw) and (mh <= 0 or h <= mh)):
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

def _audio_dir() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(os.path.dirname(script_dir)), "audio")


def _fonts_dir() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(script_dir), "fonts")


def _html_dir() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(script_dir), "html")


def _list_audio_files() -> list[str]:
    root = _audio_dir()
    try:
        names = os.listdir(root)
    except Exception:
        return []
    files = []
    for name in names:
        ext = os.path.splitext(name)[1].lower()
        if ext not in AUDIO_EXTENSIONS:
            continue
        path = os.path.join(root, name)
        if os.path.isfile(path):
            files.append(name)
    files.sort(key=lambda s: s.lower())
    return files


class _DashboardState:
    def __init__(self, metrics_obj, agent_obj=None, history_limit: int = DASH_HISTORY_LIMIT):
        self.metrics = metrics_obj
        self.agent = agent_obj
        self.history = deque(maxlen=max(120, history_limit))
        self.latest: dict[str, Any] = {}
        self.lock = threading.Lock()
        self.last_steps: int | None = None
        self.last_steps_time: float | None = None
        self._level_windows = {
            "25k": {"limit": LEVEL_25K_FRAMES, "samples": deque(), "frames": 0, "weighted": 0.0},
            "100k": {"limit": LEVEL_100K_FRAMES, "samples": deque(), "frames": 0, "weighted": 0.0},
            "1m": {"limit": LEVEL_1M_FRAMES, "samples": deque(), "frames": 0, "weighted": 0.0},
            "5m": {"limit": LEVEL_5M_FRAMES, "samples": deque(), "frames": 0, "weighted": 0.0},
        }
        self._last_level_frame_count: int | None = None
        self._last_avg_inf_ms = 0.0
        self._web_clients: dict[str, float] = {}
        self._cached_now_body = b"{}"
        self._model_desc: str | None = None
        self._model_desc_key: tuple[Any, ...] | None = None
        # Agreement 1M window (frame-weighted EMA)
        self._agree_window: deque = deque()  # [(agreement, frame_delta), ...]
        self._agree_window_frames: int = 0
        self._agree_window_weighted: float = 0.0
        self._last_agree_frame_count: int | None = None
        # Skip initial samples to let values stabilize
        self._sample_count: int = 0
        self._first_sample_time: float | None = None
        self._gpu_snapshot: list[dict[str, Any]] = []
        self._gpu_last_poll_ts: float = 0.0
        self._gpu_poll_interval_s: float = 1.0
        self._cpu_last_totals: tuple[int, int] | None = None
        self._cpu_snapshot: dict[str, float] = {"cpu_pct": 0.0, "mem_free_gb": 0.0, "disk_free_gb": 0.0}
        self._cpu_last_poll_ts: float = 0.0
        self._cpu_poll_interval_s: float = 1.0

    def _clear_level_windows(self):
        for win in self._level_windows.values():
            win["samples"].clear()
            win["frames"] = 0
            win["weighted"] = 0.0

    def _update_web_client_count_locked(self, now_ts: float | None = None) -> int:
        now = float(now_ts if now_ts is not None else time.time())
        stale_before = now - WEB_CLIENT_TIMEOUT_S
        stale = [cid for cid, ts in self._web_clients.items() if ts < stale_before]
        for cid in stale:
            self._web_clients.pop(cid, None)
        active = len(self._web_clients)
        with self.metrics.lock:
            self.metrics.web_client_count = active
        return active

    def touch_web_client(self, client_id: str | None):
        if not client_id:
            return
        now = time.time()
        with self.lock:
            self._web_clients[client_id] = now
            self._update_web_client_count_locked(now)

    def _update_level_windows(self, frame_count: int, average_level: float) -> tuple[float, float, float, float]:
        raw_level = float(average_level)
        level = round(raw_level, 4) if math.isfinite(raw_level) else 0.0
        if self._last_level_frame_count is None:
            self._last_level_frame_count = frame_count
            return level, level, level, level

        if frame_count < self._last_level_frame_count:
            self._clear_level_windows()
            self._last_level_frame_count = frame_count
            return level, level, level, level

        frame_delta = max(0, int(frame_count - self._last_level_frame_count))
        self._last_level_frame_count = frame_count

        if frame_delta > 0:
            for win in self._level_windows.values():
                samples = win["samples"]
                if samples and abs(samples[-1][0] - level) < 1e-9:
                    last_level, last_frames = samples[-1]
                    samples[-1] = (last_level, last_frames + frame_delta)
                else:
                    samples.append((level, frame_delta))

                win["frames"] += frame_delta
                win["weighted"] += (level * frame_delta)

                while samples and win["frames"] > win["limit"]:
                    overflow = win["frames"] - win["limit"]
                    oldest_level, oldest_frames = samples[0]
                    if oldest_frames <= overflow:
                        samples.popleft()
                        win["frames"] -= oldest_frames
                        win["weighted"] -= (oldest_level * oldest_frames)
                    else:
                        samples[0] = (oldest_level, oldest_frames - overflow)
                        win["frames"] -= overflow
                        win["weighted"] -= (oldest_level * overflow)
                        break

        def _mean_or_level(win):
            if win["frames"] <= 0:
                return level
            return win["weighted"] / max(1, win["frames"])

        return (
            _mean_or_level(self._level_windows["25k"]),
            _mean_or_level(self._level_windows["100k"]),
            _mean_or_level(self._level_windows["1m"]),
            _mean_or_level(self._level_windows["5m"]),
        )

    def _update_agreement_window(self, frame_count: int, agreement: float) -> float:
        """Track agreement over a 1M-frame sliding window, frame-weighted."""
        agree = round(float(agreement), 6) if math.isfinite(float(agreement)) else 0.0
        if self._last_agree_frame_count is None:
            self._last_agree_frame_count = frame_count
            return agree
        if frame_count < self._last_agree_frame_count:
            self._agree_window.clear()
            self._agree_window_frames = 0
            self._agree_window_weighted = 0.0
            self._last_agree_frame_count = frame_count
            return agree
        frame_delta = max(0, int(frame_count - self._last_agree_frame_count))
        self._last_agree_frame_count = frame_count
        if frame_delta > 0:
            samples = self._agree_window
            if samples and abs(samples[-1][0] - agree) < 1e-9:
                old_a, old_f = samples[-1]
                samples[-1] = (old_a, old_f + frame_delta)
            else:
                samples.append((agree, frame_delta))
            self._agree_window_frames += frame_delta
            self._agree_window_weighted += (agree * frame_delta)
            limit = LEVEL_1M_FRAMES
            while samples and self._agree_window_frames > limit:
                overflow = self._agree_window_frames - limit
                oldest_a, oldest_f = samples[0]
                if oldest_f <= overflow:
                    samples.popleft()
                    self._agree_window_frames -= oldest_f
                    self._agree_window_weighted -= (oldest_a * oldest_f)
                else:
                    samples[0] = (oldest_a, oldest_f - overflow)
                    self._agree_window_frames -= overflow
                    self._agree_window_weighted -= (oldest_a * overflow)
                    break
        if self._agree_window_frames <= 0:
            return agree
        return self._agree_window_weighted / max(1, self._agree_window_frames)

    def _get_model_desc(self) -> str:
      cfg = RL_CONFIG
      net = getattr(self.agent, "online_net", None) if self.agent is not None else None

      # Prefer describing the active network structure rather than the older
      # hybrid config fields that may still exist only for compatibility.
      if net is not None and hasattr(net, "base_state_size"):
        if bool(getattr(net, "use_pure_mlp", False)):
          base_state = int(getattr(net, "base_state_size", int(getattr(cfg, "base_state_size", 0) or 0)))
          stack_depth = int(getattr(net, "stack_depth", max(1, int(getattr(cfg, "frame_stack", 1) or 1))))
          hidden_layers = list(getattr(net, "mlp_hidden_layers", list(getattr(cfg, "mlp_hidden_layers", [1024, 512]) or [1024, 512])))
          output_dim = int(getattr(net, "mlp_output_dim", int(getattr(cfg, "mlp_output_dim", 256) or 256)))
          uses_attn = bool(getattr(net, "use_mlp_with_attention", False))
          uses_dir_lanes = bool(getattr(net, "use_directional_lanes", False))
          slot_count = int(getattr(net, "num_object_slots", int(getattr(cfg, "object_slots", 0) or 0)))
          token_features = int(getattr(net, "object_token_features", int(getattr(cfg, "legacy_slot_token_features", 0) or 0)))
          lane_enc = getattr(net, "lane_encoder", None)
          attn_dim = int(getattr(lane_enc, "embed_dim", 0)) if lane_enc is not None else int(getattr(getattr(net, "object_attn", None), "out_dim", int(getattr(cfg, "attn_dim", 0) or 0)))
          attn_layers = int(getattr(cfg, "attn_layers", 1) or 1)
          attn_frame_count = int(getattr(net, "attn_frame_count", 1) or 1)
          attn_scope = "all" if bool(getattr(net, "attn_all_frames", False)) else "latest"
          head_fc = getattr(net, "val_fc", None) or getattr(net, "q_fc", None) or getattr(net, "move_adv_fc", None)
          head_mid = int(head_fc.out_features) if (head_fc is not None and hasattr(head_fc, "out_features")) else max(64, output_dim // 2)
          param_count = sum(p.numel() for p in net.parameters())

          model_desc_key = (
            id(net),
            "mlp_dir_lanes" if uses_dir_lanes else ("mlp_with_attn" if uses_attn else "pure_mlp"),
            base_state,
            stack_depth,
            tuple(hidden_layers),
            output_dim,
            slot_count,
            token_features,
            attn_dim,
            attn_layers,
            attn_frame_count,
            attn_scope,
            head_mid,
            bool(getattr(net, "use_factorized_action_heads", False)),
            int(getattr(cfg, "num_move_actions", 0) or 0),
            int(getattr(cfg, "num_fire_actions", 0) or 0),
            param_count,
          )
          if self._model_desc is not None and self._model_desc_key == model_desc_key:
            return self._model_desc

          layers = [f"{base_state * stack_depth}"]
          if uses_dir_lanes:
            layers.append(f"lanes8×{attn_dim}")
          elif uses_attn:
            layers.extend([f"slot{slot_count}x{token_features}", f"set{attn_dim}x{attn_layers}@{attn_scope}{attn_frame_count}"])
          layers.extend([*[str(v) for v in hidden_layers], str(output_dim), str(head_mid)])
          head_style = (
            f"mf({int(getattr(cfg, 'num_move_actions', 0) or 0)}+{int(getattr(cfg, 'num_fire_actions', 0) or 0)})"
            if bool(getattr(net, "use_factorized_action_heads", False))
            else f"joint{int(getattr(cfg, 'num_joint_actions', 0) or 0)}"
          )
          layers.append(head_style)
          arch_str = " » ".join(layers)
          if param_count >= 1_000_000:
            p_str = f"{param_count / 1_000_000:.1f}M"
          elif param_count >= 1_000:
            p_str = f"{param_count / 1_000:.0f}K"
          else:
            p_str = str(param_count)
          mode_label = "MLP+Lanes" if uses_dir_lanes else ("MLP+Attn" if uses_attn else "MLP")
          desc = f"Model: {mode_label} · {arch_str} · {p_str} params"
          self._model_desc = desc
          self._model_desc_key = model_desc_key
          return desc

        base_state = int(getattr(net, "base_state_size", int(getattr(cfg, "base_state_size", 0) or 0)))
        stack_depth = int(getattr(net, "stack_depth", max(1, int(getattr(cfg, "frame_stack", 1) or 1))))
        slot_count = int(getattr(net, "num_object_slots", int(getattr(cfg, "object_slots", 0) or 0)))
        token_features = int(getattr(net, "object_token_features", int(getattr(cfg, "legacy_slot_token_features", 0) or 0)))
        attn_dim = int(getattr(getattr(net, "object_attn", None), "out_dim", int(getattr(cfg, "attn_dim", 0) or 0)))
        attn_layers = int(getattr(cfg, "attn_layers", 1) or 1)
        attn_frame_count = int(getattr(net, "attn_frame_count", 1) or 1)
        attn_scope = "all" if bool(getattr(net, "attn_all_frames", False)) else "latest"
        trunk_widths = [int(getattr(cfg, "trunk_hidden", 256) or 256)] * int(getattr(cfg, "trunk_layers", 1) or 1)
        head_fc = getattr(net, "val_fc", None) or getattr(net, "q_fc", None)
        head_mid = int(head_fc.out_features) if (head_fc is not None and hasattr(head_fc, "out_features")) else max(64, trunk_widths[-1] // 2)
        param_count = sum(p.numel() for p in net.parameters())

        model_desc_key = (
          id(net),
          base_state,
          stack_depth,
          slot_count,
          token_features,
          attn_dim,
          attn_layers,
          attn_frame_count,
          attn_scope,
          tuple(trunk_widths),
          head_mid,
          bool(getattr(net, "use_factorized_action_heads", False)),
          int(getattr(cfg, "num_move_actions", 0) or 0),
          int(getattr(cfg, "num_fire_actions", 0) or 0),
          param_count,
        )
        if self._model_desc is not None and self._model_desc_key == model_desc_key:
          return self._model_desc

        layers = [
          f"base{base_state}x{stack_depth}",
          f"slot{slot_count}x{token_features}",
          f"set{attn_dim}x{attn_layers}@{attn_scope}{attn_frame_count}",
        ]
        layers.extend(str(width) for width in trunk_widths)
        layers.append(str(head_mid))
        head_style = (
          f"mf({int(getattr(cfg, 'num_move_actions', 0) or 0)}+{int(getattr(cfg, 'num_fire_actions', 0) or 0)})"
          if bool(getattr(net, "use_factorized_action_heads", False))
          else f"joint{int(getattr(cfg, 'num_joint_actions', 0) or 0)}"
        )
        layers.append(head_style)
        arch_str = " » ".join(layers)
        if param_count >= 1_000_000:
          p_str = f"{param_count / 1_000_000:.1f}M"
        elif param_count >= 1_000:
          p_str = f"{param_count / 1_000:.0f}K"
        else:
          p_str = str(param_count)
        desc = f"Model: SlotSet · {arch_str} · {p_str} params"
        self._model_desc = desc
        self._model_desc_key = model_desc_key
        return desc

      gg = int(getattr(cfg, "global_feature_count", 98))
      gc = int(getattr(cfg, "grid_channels", 8))
      gh = int(getattr(cfg, "grid_height", 12))
      gw = int(getattr(cfg, "grid_width", 12))
      tk = int(getattr(cfg, "object_token_count", 64))
      tf = int(getattr(cfg, "object_token_features", 15))
      ad = int(getattr(cfg, "attn_dim", 192))
      ghid = int(getattr(cfg, "global_hidden", 192))
      trunk_in = ghid + ad + ad
      trunk_widths: list[int] = [int(getattr(cfg, "trunk_hidden", 512))] * int(getattr(cfg, "trunk_layers", 1))
      head_mid = max(64, int(getattr(cfg, "trunk_hidden", 512)) // 2)

      if net is not None:
        gg = int(getattr(net, "global_feature_count", gg))
        gc = int(getattr(net, "grid_channels", gc))
        gh = int(getattr(net, "grid_height", gh))
        gw = int(getattr(net, "grid_width", gw))
        tk = int(getattr(net, "object_token_count", tk))
        tf = int(getattr(net, "object_token_features", tf))
        trunk_in = int(
          getattr(getattr(net, "global_encoder", None), "out_dim", ghid)
          + getattr(getattr(net, "grid_encoder", None), "out_dim", ad)
          + getattr(getattr(net, "entity_encoder", None), "out_dim", ad)
        )
        trunk_widths = [
          int(layer.out_features)
          for layer in getattr(net, "trunk", [])
          if hasattr(layer, "out_features")
        ]
        head_fc = getattr(net, "val_fc", None) or getattr(net, "q_fc", None)
        if head_fc is not None and hasattr(head_fc, "out_features"):
          head_mid = int(head_fc.out_features)

      model_desc_key = (id(net), gg, gw, gh, gc, tk, tf, trunk_in, tuple(trunk_widths), head_mid)
      if self._model_desc is not None and self._model_desc_key == model_desc_key:
        return self._model_desc

      try:
        if net is not None:
          param_count = sum(p.numel() for p in net.parameters())
        else:
          th = int(getattr(cfg, "trunk_hidden", 512))
          tl = int(getattr(cfg, "trunk_layers", 1))
          na = cfg.num_move_actions * cfg.num_fire_actions
          n_atoms = cfg.num_atoms if cfg.use_distributional else 1
          hm = th // 2
          global_p = (gg * cfg.global_hidden + cfg.global_hidden) + (cfg.global_hidden * cfg.global_hidden + cfg.global_hidden)
          grid_p = (gc * cfg.grid_hidden_channels * 9 + cfg.grid_hidden_channels)
          grid_p += (cfg.grid_hidden_channels * (cfg.grid_hidden_channels * 2) * 9 + (cfg.grid_hidden_channels * 2))
          grid_p += ((cfg.grid_hidden_channels * 2) * (cfg.grid_hidden_channels * 3) * 9 + (cfg.grid_hidden_channels * 3))
          grid_p += ((cfg.grid_hidden_channels * 3) * 4 * ad + ad)
          token_p = (tf * ad + ad) + (4 * ad * ad * cfg.attn_layers)
          trunk_p = (trunk_in * th + th)
          for _ in range(1, tl):
            trunk_p += th * th + th
          heads_p = 2 * (th * hm + hm) + hm * n_atoms + n_atoms + hm * (na * n_atoms) + na * n_atoms
          param_count = global_p + grid_p + token_p + trunk_p + heads_p
      except Exception:
        param_count = 0

      layers = [f"g{gg}", f"grid{gw}x{gh}x{gc}", f"tok{tk}x{tf}", str(trunk_in)]
      layers.extend(str(width) for width in trunk_widths)
      layers.append(str(head_mid))
      arch_str = " \u00bb ".join(layers)
      if param_count >= 1_000_000:
        p_str = f"{param_count / 1_000_000:.1f}M"
      elif param_count >= 1_000:
        p_str = f"{param_count / 1_000:.0f}K"
      else:
        p_str = str(param_count)
      desc = f"Model: Legacy/Compat \u00b7 {arch_str} \u00b7 {p_str} params"
      self._model_desc = desc
      self._model_desc_key = model_desc_key
      return desc

    def _sample_gpu_status(self, now_ts: float | None = None) -> list[dict[str, Any]]:
        now = float(now_ts if now_ts is not None else time.time())
        if self._gpu_snapshot and (now - self._gpu_last_poll_ts) < self._gpu_poll_interval_s:
            return list(self._gpu_snapshot)
        try:
            res = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=0.75,
                check=False,
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
                        "index": idx,
                        "name": parts[1],
                        "util": util,
                        "mem_used_mb": mem_used,
                        "mem_total_mb": mem_total,
                        "mem_pct": max(0.0, min(100.0, (mem_used / mem_total) * 100.0)),
                    })
            self._gpu_snapshot = rows
            self._gpu_last_poll_ts = now
        except Exception:
            self._gpu_last_poll_ts = now
        return list(self._gpu_snapshot)

    def _sample_system_status(self, now_ts: float | None = None) -> dict[str, float]:
        now = float(now_ts if now_ts is not None else time.time())
        if (now - self._cpu_last_poll_ts) < self._cpu_poll_interval_s:
            return dict(self._cpu_snapshot)
        cpu_pct = float(self._cpu_snapshot.get("cpu_pct", 0.0) or 0.0)
        mem_free_gb = float(self._cpu_snapshot.get("mem_free_gb", 0.0) or 0.0)
        disk_free_gb = float(self._cpu_snapshot.get("disk_free_gb", 0.0) or 0.0)
        try:
            with open("/proc/stat", "r", encoding="utf-8") as fh:
                first = fh.readline().strip().split()
            if len(first) >= 5 and first[0] == "cpu":
                vals = [int(v) for v in first[1:] if v.isdigit() or (v and v[0] == "-" and v[1:].isdigit())]
                if vals:
                    total = int(sum(vals))
                    idle = int(vals[3] + (vals[4] if len(vals) > 4 else 0))
                    prev = self._cpu_last_totals
                    if prev is not None:
                        dt = max(1, total - prev[0])
                        didle = max(0, idle - prev[1])
                        cpu_pct = max(0.0, min(100.0, (1.0 - (didle / dt)) * 100.0))
                    self._cpu_last_totals = (total, idle)
            mem_avail_kb = None
            with open("/proc/meminfo", "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("MemAvailable:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            mem_avail_kb = int(parts[1])
                        break
            if mem_avail_kb is not None:
                mem_free_gb = max(0.0, float(mem_avail_kb) / (1024.0 * 1024.0))
            du = shutil.disk_usage("/")
            disk_free_gb = max(0.0, float(du.free) / (1024.0 ** 3))
        except Exception:
            pass
        self._cpu_snapshot = {"cpu_pct": cpu_pct, "mem_free_gb": mem_free_gb, "disk_free_gb": disk_free_gb}
        self._cpu_last_poll_ts = now
        return dict(self._cpu_snapshot)

    def _build_snapshot(self) -> dict[str, Any]:
        now = time.time()
        _prs = float(RL_CONFIG.point_reward_scale)  # display-only multiplier

        fps = self.metrics.get_fps()

        with self.metrics.lock:
            frame_count = int(self.metrics.frame_count)
            epsilon_raw = float(self.metrics.epsilon)
            _eps_ov = game_settings.epsilon_pct
            if _eps_ov >= 0:
                epsilon_effective = _eps_ov / 100.0
            else:
                epsilon_effective = 0.0 if bool(self.metrics.override_epsilon) else epsilon_raw
            _xprt_ov = game_settings.expert_pct
            expert_ratio = (_xprt_ov / 100.0) if _xprt_ov >= 0 else float(self.metrics.expert_ratio)
            client_count = int(self.metrics.client_count)
            web_client_count = int(self.metrics.web_client_count)
            game_preview_seq = int(getattr(self.metrics, "game_preview_seq", 0))
            game_preview_width = int(getattr(self.metrics, "game_preview_width", 0))
            game_preview_height = int(getattr(self.metrics, "game_preview_height", 0))
            game_preview_format = str(getattr(self.metrics, "game_preview_format", "") or "")
            game_preview_source_format = str(getattr(self.metrics, "game_preview_source_format", "") or "")
            game_preview_encoded_bytes = int(getattr(self.metrics, "game_preview_encoded_bytes", 0) or 0)
            game_preview_raw_bytes = int(getattr(self.metrics, "game_preview_raw_bytes", 0) or 0)
            game_preview_compression_ratio = float(getattr(self.metrics, "game_preview_compression_ratio", 1.0) or 1.0)
            game_preview_fps = float(getattr(self.metrics, "game_preview_fps", 0.0) or 0.0)
            average_level = float(self.metrics.average_level)
            memory_buffer_size = int(self.metrics.memory_buffer_size)
            memory_buffer_k = int(memory_buffer_size // 1000)
            buffer_capacity = int(max(1, getattr(RL_CONFIG, "memory_size", 1)))
            memory_buffer_pct = max(0.0, min(100.0, (memory_buffer_size / buffer_capacity) * 100.0))
            total_training_steps = int(self.metrics.total_training_steps)
            last_loss = float(self.metrics.last_loss)
            last_grad_norm = float(self.metrics.last_grad_norm)
            last_bc_loss = float(self.metrics.last_bc_loss)
            last_q_mean = float(self.metrics.last_q_mean)
            training_enabled = bool(self.metrics.training_enabled)
            override_expert = bool(self.metrics.override_expert)
            override_epsilon = bool(self.metrics.override_epsilon)
            inference_requests = int(self.metrics.total_inference_requests)
            inference_time = float(self.metrics.total_inference_time)
            last_agreement = float(self.metrics.last_agreement)

            reward_total = _tail_mean(self.metrics.episode_rewards) * _prs
            reward_dqn = _tail_mean(self.metrics.dqn_rewards) * _prs
            reward_subj = _tail_mean(self.metrics.subj_rewards) * _prs
            reward_obj = _tail_mean(self.metrics.obj_rewards) * _prs

        try:
            dqn100k_raw, dqn1m_raw, dqn5m_raw = get_dqn_window_averages()
        except Exception:
            dqn100k_raw = dqn1m_raw = dqn5m_raw = 0.0
        try:
            total100k_raw, total1m_raw, total5m_raw = get_total_window_averages()
        except Exception:
            total100k_raw = total1m_raw = total5m_raw = 0.0
        level_25k, level_100k, level_1m, level_5m = self._update_level_windows(frame_count, average_level)
        agreement_1m = self._update_agreement_window(frame_count, last_agreement)
        if inference_requests > 0:
            self._last_avg_inf_ms = (inference_time / max(1, inference_requests)) * 1000.0
        avg_inf_ms = self._last_avg_inf_ms

        steps_per_sec = 0.0
        if self.last_steps is not None and self.last_steps_time is not None:
            dt = max(1e-6, now - self.last_steps_time)
            ds = max(0, total_training_steps - self.last_steps)
            steps_per_sec = ds / dt
        self.last_steps = total_training_steps
        self.last_steps_time = now
        replay_per_frame = (steps_per_sec * float(getattr(RL_CONFIG, "batch_size", 1))) / max(1e-6, float(fps))

        lr = None
        q_min = None
        q_max = None
        if self.agent is not None:
            try:
                lr_val = self.agent.get_lr()
                lr = float(lr_val)
            except Exception:
                lr = None
            try:
                mn, mx = self.agent.get_q_value_range()
                if math.isfinite(float(mn)) and math.isfinite(float(mx)):
                    q_min = float(mn)
                    q_max = float(mx)
            except Exception:
                q_min = q_max = None

        client_rows: list[dict[str, Any]] = []
        preview_selected_client_id = -1
        try:
            srv = getattr(self.metrics, "global_server", None)
            if srv is not None:
                client_rows = list(srv.get_client_rows() or [])
                selected_cid = srv.get_selected_preview_client_id()
                if selected_cid is not None:
                    preview_selected_client_id = int(selected_cid)
        except Exception:
            client_rows = []
            preview_selected_client_id = -1

        gpu_rows = self._sample_gpu_status(now)
        gpu0 = gpu_rows[0] if len(gpu_rows) >= 1 else {}
        gpu1 = gpu_rows[1] if len(gpu_rows) >= 2 else {}
        sys_status = self._sample_system_status(now)

        return {
            "ts": now,
            "frame_count": frame_count,
            "fps": fps,
            "training_steps": total_training_steps,
            "steps_per_sec": steps_per_sec,
            "batch_size": int(getattr(RL_CONFIG, "batch_size", 1)),
            "rpl_per_frame": replay_per_frame,
            "epsilon": epsilon_effective,
            "epsilon_raw": epsilon_raw,
            "expert_ratio": expert_ratio,
            "client_count": client_count,
            "web_client_count": web_client_count,
            "game_preview_seq": game_preview_seq,
            "game_preview_width": game_preview_width,
            "game_preview_height": game_preview_height,
            "game_preview_format": game_preview_format,
            "game_preview_source_format": game_preview_source_format,
            "game_preview_encoded_bytes": game_preview_encoded_bytes,
            "game_preview_raw_bytes": game_preview_raw_bytes,
            "game_preview_compression_ratio": game_preview_compression_ratio,
            "game_preview_fps": game_preview_fps,
            "average_level": average_level,
            "memory_buffer_size": memory_buffer_size,
            "memory_buffer_k": memory_buffer_k,
            "memory_buffer_pct": memory_buffer_pct,
            "avg_inf_ms": avg_inf_ms,
            "loss": last_loss,
            "grad_norm": last_grad_norm,
            "bc_loss": last_bc_loss,
            "q_mean": last_q_mean,
            "reward_total": reward_total,
            "reward_dqn": reward_dqn,
            "reward_subj": reward_subj,
            "reward_obj": reward_obj,
            "dqn_100k": float(dqn100k_raw) * _prs,
            "dqn_1m": float(dqn1m_raw) * _prs,
            "dqn_5m": float(dqn5m_raw) * _prs,
            "total_100k": float(total100k_raw) * _prs,
            "total_1m": float(total1m_raw) * _prs,
            "total_5m": float(total5m_raw) * _prs,
            "level_25k": float(level_25k),
            "level_100k": float(level_100k),
            "level_1m": float(level_1m),
            "level_5m": float(level_5m),
            "training_enabled": training_enabled,
            "override_expert": override_expert,
            "override_epsilon": override_epsilon,
            "lr": lr,
            "training_steps": total_training_steps,
            "lr_max": float(RL_CONFIG.lr),
            "lr_min": float(RL_CONFIG.lr_min),
            "lr_warmup_steps": int(RL_CONFIG.lr_warmup_steps),
            "lr_cosine_period": int(RL_CONFIG.lr_cosine_period),
            "lr_use_restarts": bool(getattr(RL_CONFIG, "lr_use_restarts", False)),
            "q_min": q_min,
            "q_max": q_max,
            "eplen_1m": get_eplen_1m_average(),
            "eplen_100k": get_eplen_100k_average(),
            "peak_level": float(self.metrics.peak_level),
            "peak_episode_reward": float(self.metrics.peak_episode_reward) * _prs,
            "peak_game_score": int(self.metrics.peak_game_score),
            "records_reset_seq": int(getattr(self.metrics, "records_reset_seq", 0) or 0),
            "avg_game_score": float(self.metrics.avg_game_score),
            "game_count": int(self.metrics.total_games_played),
            "episodes_this_run": int(self.metrics.episodes_this_run),
            "agreement": last_agreement,
            "agreement_1m": agreement_1m,
            "model_desc": self._get_model_desc(),
            "pulse_state": plateau_pulser.state,
            "pulse_remaining": int(self.metrics.manual_pulse_frames_remaining),
            "pulse_count": plateau_pulser.total_pulses,
            "pulse_enabled": True,  # manual pulse is always available
            "client_rows": client_rows,
            "preview_selected_client_id": preview_selected_client_id,
            "gpu_rows": gpu_rows,
            "gpu0_util": float(gpu0.get("util", 0.0) or 0.0),
            "gpu0_mem_pct": float(gpu0.get("mem_pct", 0.0) or 0.0),
            "gpu0_name": str(gpu0.get("name", "") or ""),
            "gpu1_util": float(gpu1.get("util", 0.0) or 0.0),
            "gpu1_mem_pct": float(gpu1.get("mem_pct", 0.0) or 0.0),
            "gpu1_name": str(gpu1.get("name", "") or ""),
            "cpu_pct": float(sys_status.get("cpu_pct", 0.0) or 0.0),
            "mem_free_gb": float(sys_status.get("mem_free_gb", 0.0) or 0.0),
            "disk_free_gb": float(sys_status.get("disk_free_gb", 0.0) or 0.0),
            "game_settings": game_settings.snapshot(),
            "preview_capture_enabled": bool(getattr(self.metrics, "preview_capture_enabled", True)),
            "hud_enabled": bool(getattr(self.metrics, "hud_enabled", True)),
        }

    @staticmethod
    def _pulse_remaining(frame_count: int) -> int:
        """Frames remaining in current manual pulse, 0 if idle."""
        from config import metrics as _m
        return int(_m.manual_pulse_frames_remaining)

    def sample(self):
        with self.lock:
            self._update_web_client_count_locked()
        snap = self._build_snapshot()
        with self.lock:
            now = time.time()
            if self._first_sample_time is None:
                self._first_sample_time = now
            self._sample_count += 1
            # Skip the first 10 samples or 2 seconds, whichever comes first
            if self._sample_count <= 10 and (now - self._first_sample_time) < 2.0:
                self.latest = snap
                self._cached_now_body = json.dumps(snap).encode("utf-8")
                return
            self.latest = snap
            self.history.append(snap)
            self._cached_now_body = json.dumps(snap).encode("utf-8")

    def payload(self) -> dict[str, Any]:
        with self.lock:
            return {
                "now": self.latest,
                "history": list(self.history),
            }

    def now_body(self) -> bytes:
        with self.lock:
            return self._cached_now_body

    def game_preview_body(self, since_seq: int | None = None, wait_timeout_s: float = 0.0, profile: str = "default") -> bytes:
        since = None if since_seq is None else int(max(0, since_seq))
        timeout = max(0.0, float(wait_timeout_s))
        profile_norm = _normalize_preview_profile(profile)
        deadline = time.time() + timeout
        seq = 0
        client_id = -1
        width = 0
        height = 0
        fmt = ""
        ts = 0.0
        raw = b""
        while True:
            with self.metrics.lock:
                seq = int(getattr(self.metrics, "game_preview_seq", 0))
                client_id = int(getattr(self.metrics, "game_preview_client_id", -1))
                width = int(getattr(self.metrics, "game_preview_width", 0))
                height = int(getattr(self.metrics, "game_preview_height", 0))
                fmt = str(getattr(self.metrics, "game_preview_format", "") or "")
                ts = float(getattr(self.metrics, "game_preview_updated_ts", 0.0))
                raw = bytes(getattr(self.metrics, "game_preview_data", b"") or b"")
            if since is None or seq != since or timeout <= 0.0 or time.time() >= deadline:
                break
            time.sleep(0.01)
        changed = since is None or seq != since
        out_w = width
        out_h = height
        out_data_b64 = ""
        if changed and seq > 0 and width > 0 and height > 0 and fmt == "rgb565be" and raw:
            try:
                max_w, max_h = _preview_http_limits(profile_norm)
                raw, out_w, out_h = _downscale_rgb565_nearest(raw, width, height, max_w, max_h)
                out_data_b64 = base64.b64encode(raw).decode("ascii") if raw else ""
            except Exception:
                out_data_b64 = ""
                out_w = 0
                out_h = 0

        payload = {
            "seq": seq,
            "client_id": client_id,
            "width": out_w if changed else width,
            "height": out_h if changed else height,
            "format": fmt,
            "ts": ts,
            "profile": profile_norm,
            "unchanged": not changed,
            "data": out_data_b64,
        }
        return json.dumps(payload, separators=(",", ":")).encode("utf-8")


class _PreviewVideoTrack(VideoStreamTrack):
    """Pull latest cached preview frame and expose as a WebRTC video track."""

    def __init__(self, metrics_obj, fps: float = 30.0, max_width: int = 0, max_height: int = 0):
        super().__init__()
        self.metrics = metrics_obj
        self.fps = max(1.0, float(fps))
        self.max_width = max(0, int(max_width))
        self.max_height = max(0, int(max_height))
        self._tb = Fraction(1, 90000)
        self._last_seq = -1
        self._last_rgb = None
        self._last_w = 320
        self._last_h = 224
        self._frames_sent = 0
        self._last_log_ts = 0.0
        self._frame_interval_s = 1.0 / self.fps
        self._next_emit_ts = 0.0

    @staticmethod
    def _rgb565_to_rgb24(raw: bytes, width: int, height: int):
        if np is None:
            return None
        px = int(width) * int(height)
        arr = np.frombuffer(raw, dtype=np.dtype(">u2"), count=px)
        if arr.size != px:
            return None
        arr = arr.reshape((height, width))
        r = ((arr >> 11) & 0x1F).astype(np.uint8)
        g = ((arr >> 5) & 0x3F).astype(np.uint8)
        b = (arr & 0x1F).astype(np.uint8)
        rgb = np.empty((height, width, 3), dtype=np.uint8)
        # Widen before scaling to avoid uint8 overflow (which produces near-black output).
        rgb[..., 0] = ((r.astype(np.uint16) * 255) // 31).astype(np.uint8)
        rgb[..., 1] = ((g.astype(np.uint16) * 255) // 63).astype(np.uint8)
        rgb[..., 2] = ((b.astype(np.uint16) * 255) // 31).astype(np.uint8)
        return rgb

    def _snapshot_rgb_frame(self):
        with self.metrics.lock:
            seq = int(getattr(self.metrics, "game_preview_seq", 0))
            w = int(getattr(self.metrics, "game_preview_width", 0))
            h = int(getattr(self.metrics, "game_preview_height", 0))
            fmt = str(getattr(self.metrics, "game_preview_format", "") or "")
            raw = bytes(getattr(self.metrics, "game_preview_data", b"") or b"")

        if seq <= 0 or w <= 0 or h <= 0 or fmt != "rgb565be" or not raw:
            return None, None, None, f"missing seq={seq} w={w} h={h} fmt={fmt} raw={len(raw)}"

        if seq == self._last_seq and self._last_rgb is not None:
            return self._last_rgb, self._last_w, self._last_h, ""

        expected = int(w) * int(h) * 2
        if len(raw) != expected:
            return None, None, None, f"len_mismatch got={len(raw)} expected={expected}"

        rgb = self._rgb565_to_rgb24(raw, w, h)
        if rgb is None:
            return None, None, None, "rgb_convert_failed"

        self._last_seq = seq
        self._last_rgb = rgb
        self._last_w = int(w)
        self._last_h = int(h)
        return rgb, w, h, ""

    @staticmethod
    def _debug_pattern(width: int, height: int, tick: int):
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        t = int(tick) & 0xFF
        rgb[..., 0] = (32 + t) & 0xFF
        rgb[..., 1] = (80 + (t * 3)) & 0xFF
        rgb[..., 2] = (160 + (t * 5)) & 0xFF
        # Bright border so "frame exists" is obvious.
        rgb[0:4, :, :] = 255
        rgb[-4:, :, :] = 255
        rgb[:, 0:4, :] = 255
        rgb[:, -4:, :] = 255
        return rgb

    def _downscale_if_needed(self, rgb):
        if rgb is None:
            return rgb
        h, w = rgb.shape[0], rgb.shape[1]
        if w <= 0 or h <= 0:
            return rgb
        if self.max_width <= 0 and self.max_height <= 0:
            return rgb
        sx = (float(w) / float(self.max_width)) if self.max_width > 0 else 1.0
        sy = (float(h) / float(self.max_height)) if self.max_height > 0 else 1.0
        scale = max(1.0, sx, sy)
        if scale <= 1.0:
            return rgb
        step = max(2, int(math.ceil(scale)))
        return rgb[::step, ::step, :].copy()

    async def recv(self):
        now_emit = time.perf_counter()
        if self._next_emit_ts <= 0.0:
            self._next_emit_ts = now_emit
        sleep_s = self._next_emit_ts - now_emit
        if sleep_s > 0.0:
            await asyncio.sleep(sleep_s)
        self._next_emit_ts = max(self._next_emit_ts + self._frame_interval_s, time.perf_counter())

        pts, _ = await self.next_timestamp()
        rgb, w, h, err = self._snapshot_rgb_frame()
        self._frames_sent += 1
        now = time.time()
        if rgb is None:
            if np is None:
                raise RuntimeError("numpy is required for WebRTC preview streaming")
            w = int(self._last_w or 320)
            h = int(self._last_h or 224)
            rgb = self._debug_pattern(w, h, self._frames_sent)
        rgb = self._downscale_if_needed(rgb)
        out_h = int(rgb.shape[0]) if rgb is not None else 0
        out_w = int(rgb.shape[1]) if rgb is not None else 0
        frame = VideoFrame.from_ndarray(rgb, format="rgb24")
        frame.pts = int(pts)
        frame.time_base = self._tb
        return frame


class _PreviewAudioSource:
    """Tail the selected preview client's bounded relay WAV and fan-out 20 ms PCM frames."""

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
        self._reader_thread_obj = None
        self._subscribers: dict[int, queue.Queue] = {}
        self._next_subscriber_id = 1
        self._current_slot: int | None = None
        self._current_path: str | None = None
        self._current_file = None
        self._current_inode: int | None = None
        self._source_generation = 0
        self._read_offset = 0
        self._header_parsed = False
        self._start_reader()

    def _selected_slot(self) -> int | None:
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

    def _parse_wav_header(self, header: bytes) -> tuple[int, int] | None:
        if len(header) < 44:
            return None
        try:
            import struct
            if header[0:4] != b"RIFF" or header[8:12] != b"WAVE":
                return None
            channels = int(struct.unpack("<H", header[22:24])[0])
            sample_rate = int(struct.unpack("<I", header[24:28])[0])
            bits_per_sample = int(struct.unpack("<H", header[34:36])[0])
            if channels not in (1, 2):
                return None
            if sample_rate <= 0:
                return None
            if bits_per_sample != 16:
                return None
            return sample_rate, channels
        except Exception:
            return None

    def _frame_params_locked(self) -> tuple[int, int]:
        samples_per_frame = max(1, int(round(self._sample_rate * self._FRAME_PTIME_S)))
        bytes_per_frame = samples_per_frame * self._channels * 2
        return samples_per_frame, bytes_per_frame

    def _drop_queue_head(self, q: queue.Queue):
        try:
            q.get_nowait()
        except queue.Empty:
            return

    def _dispatch_ready_frames_locked(self):
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

    def _reset_subscribers_locked(self):
        for q in self._subscribers.values():
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

    def _reset_audio_state_locked(self):
        self._audio_ready = False
        self._source_generation += 1
        self._buffer.clear()
        self._reset_subscribers_locked()

    def _close_current_source(self):
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

    def _reader_thread(self):
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
                        max_buffer_bytes = int(
                            self._sample_rate * self._channels * 2 * self._MAX_SOURCE_LATENCY_S
                        )
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

    def _start_reader(self):
        if self._reader_thread_obj is not None and self._reader_thread_obj.is_alive():
            return
        self._reader_thread_obj = threading.Thread(target=self._reader_thread, daemon=True, name="AudioReader")
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

    def unsubscribe(self, subscriber_id: int | None):
        if subscriber_id is None:
            return
        with self._lock:
            self._subscribers.pop(int(subscriber_id), None)

    def get_format(self) -> tuple[int, int]:
        with self._lock:
            return int(self._sample_rate), int(self._channels)

    def get_stream_identity(self) -> tuple[int | None, int]:
        with self._lock:
            return self._current_slot, int(self._source_generation)

    def close(self):
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
        self._active_generation = None
        self._last_frame_data = None
        self._last_frame_wallclock = 0.0

    def stop(self):
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
                # Restart RTP audio timing cleanly when preview ownership moves
                # to a different client/source file.
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


class _PreviewWebRTCBridge:
    def __init__(self, metrics_obj, ice_servers: list[dict[str, Any]] | None = None):
        self.metrics = metrics_obj
        self.enabled = bool(_WEBRTC_AVAILABLE and np is not None)
        self.ice_servers = list(ice_servers) if ice_servers else list(_WEBRTC_ICE_SERVERS)
        if self.enabled:
            self.error_reason = ""
        else:
            missing = []
            if not _NUMPY_AVAILABLE:
                missing.append("numpy")
            if not _WEBRTC_AVAILABLE:
                missing.append("aiortc/av")
            if missing and _WEBRTC_IMPORT_ERROR:
                self.error_reason = f"missing_deps:{','.join(missing)} ({_WEBRTC_IMPORT_ERROR})"
            else:
                self.error_reason = f"missing_deps:{','.join(missing)}" if missing else "webrtc_unavailable"
        self._thread = None
        self._loop = None
        self._pcs = set()
        self._lock = threading.Lock()
        self._audio_source = _PreviewAudioSource(self.metrics) if (self.enabled and _PREVIEW_GAME_AUDIO_TRANSPORT_ENABLED) else None

    def start(self):
        if not self.enabled:
            return
        if self._thread is not None:
            return
        if self._audio_source is None and _PREVIEW_GAME_AUDIO_TRANSPORT_ENABLED:
            self._audio_source = _PreviewAudioSource(self.metrics)

        def _runner():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            loop.run_forever()
            try:
                loop.run_until_complete(self._shutdown_async())
            finally:
                loop.close()

        self._thread = threading.Thread(target=_runner, daemon=True, name="PreviewWebRTC")
        self._thread.start()
        for _ in range(200):
            if self._loop is not None:
                return
            time.sleep(0.01)
        self.enabled = False

    async def _shutdown_async(self):
        pcs = list(self._pcs)
        self._pcs.clear()
        for pc in pcs:
            try:
                await pc.close()
            except Exception:
                pass

    def stop(self):
        if not self.enabled:
            return
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

    async def _create_answer_async(self, offer_sdp: str, offer_type: str, profile: str = "default"):
        if not self.enabled:
            return {"ok": False, "error": "webrtc_unavailable"}
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

        profile_norm = str(profile or "default").strip().lower()
        if profile_norm == "mobile":
            track = _PreviewVideoTrack(self.metrics, fps=12.0, max_width=160, max_height=132)
        else:
            profile_norm = "default"
            track = _PreviewVideoTrack(self.metrics, fps=30.0)
        pc.addTrack(track)

        if self._audio_source is None and _PREVIEW_GAME_AUDIO_TRANSPORT_ENABLED:
            try:
                self._audio_source = _PreviewAudioSource(self.metrics)
            except Exception:
                self._audio_source = None

        if self._audio_source is not None and _PREVIEW_GAME_AUDIO_TRANSPORT_ENABLED:
            try:
                audio_track = _PreviewAudioTrack(self._audio_source)
                pc.addTrack(audio_track)
            except Exception:
                audio_track = None

        await pc.setRemoteDescription(RTCSessionDescription(sdp=offer_sdp, type=offer_type))
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        return {
            "ok": True,
            "type": pc.localDescription.type,
            "sdp": pc.localDescription.sdp,
        }

    def create_answer(self, offer_sdp: str, offer_type: str, timeout_s: float = 10.0, profile: str = "default") -> dict[str, Any]:
        if not self.enabled or self._loop is None:
            return {"ok": False, "error": self.error_reason or "webrtc_unavailable"}
        try:
            fut = asyncio.run_coroutine_threadsafe(
                self._create_answer_async(offer_sdp, offer_type, profile=profile),
                self._loop,
            )
            answer = fut.result(timeout=max(1.0, float(timeout_s)))
            return answer
        except Exception as e:
            return {"ok": False, "error": f"webrtc_offer_failed: {e}"}


def _render_dashboard_html(webrtc_ice_servers: list[dict[str, Any]] | None = None) -> str:
    _ice_json = json.dumps(webrtc_ice_servers if webrtc_ice_servers else _WEBRTC_ICE_SERVERS)
    _preview_game_audio_json = "true" if _PREVIEW_GAME_AUDIO_TRANSPORT_ENABLED else "false"
    _preview_game_audio_checked = " checked" if _PREVIEW_GAME_AUDIO_TRANSPORT_ENABLED else ""
    _preview_game_audio_disabled = "" if _PREVIEW_GAME_AUDIO_TRANSPORT_ENABLED else " disabled"
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Robotron AI Metrics</title>
  <style>
    /* ══════════════════════════════════════════════════════════════
     * FONTS — Custom typefaces for VFD / LED readouts
     * ══════════════════════════════════════════════════════════════ */
    @font-face {
      font-family: "LED Dot-Matrix";
      src: url("/api/font/LED%20Dot-Matrix.ttf") format("truetype");
      font-display: swap;
    }
    @font-face {
      font-family: "DS-Digital";
      src: url("/api/font/DS-DIGI.TTF") format("truetype");
      font-display: swap;
    }
    /* ══════════════════════════════════════════════════════════════
     * CSS VARIABLES — Theme colors and neon accents
     * ══════════════════════════════════════════════════════════════ */
    :root {
      --bg0: #040510;
      --bg1: #0b1433;
      --bg2: #1a0a33;
      --panel: rgba(6, 10, 28, 0.78);
      --line: rgba(0, 229, 255, 0.26);
      --ink: #e8f6ff;
      --muted: #9cb6d4;
      --accentA: #00e5ff;
      --accentB: #ffe600;
      --accentC: #39ff14;
      --accentD: #ff2bd6;
      --neonRed: #ff2a55;
      --neonEdge: rgba(0, 229, 255, 0.65);
      --panelGlowA: rgba(0, 229, 255, 0.22);
      --panelGlowB: rgba(255, 43, 214, 0.18);
      --vfdCyan: #70f7ff;
    }
    * { box-sizing: border-box; }
    *::before, *::after { box-sizing: border-box; }
    /* ══════════════════════════════════════════════════════════════
     * BASE LAYOUT — Body background, pseudo-element overlays
     * Animations intentionally removed for GPU performance.
     * ══════════════════════════════════════════════════════════════ */
    html, body { margin: 0; padding: 0; color: var(--ink); background: var(--bg0); }
    body {
      font-family: "Avenir Next", "Segoe UI", "Helvetica Neue", sans-serif;
      min-height: 100vh;
      position: relative;
      isolation: isolate;
      overflow-x: hidden;
      background: var(--bg0);
    }
    /* ── Starfield background video ─────────────────────────────── */
    #bgVideo {
      position: fixed;
      top: 0; left: 0;
      width: 100vw; height: 100vh;
      object-fit: cover;
      z-index: -2;
      pointer-events: none;
      opacity: 0.55;
    }
    /* Gradient overlay on top of video for readability */
    #bgOverlay {
      position: fixed;
      top: 0; left: 0;
      width: 100vw; height: 100vh;
      z-index: -1;
      pointer-events: none;
      background:
        radial-gradient(1300px 650px at 6% -8%, rgba(0, 229, 255, 0.18), transparent 58%),
        radial-gradient(950px 540px at 102% -4%, rgba(255, 43, 214, 0.16), transparent 56%),
        radial-gradient(900px 500px at 52% 112%, rgba(57, 255, 20, 0.10), transparent 62%);
    }
    main {
      max-width: 1500px;
      margin: 0 auto;
      padding: 20px;
      display: grid;
      gap: 16px;
      position: relative;
      z-index: 2;
    }
    /* ══════════════════════════════════════════════════════════════
     * CARD & PANEL COMPONENTS — Metric cards, gauge cards, borders
     * Each card gets --card-border and --card-glow via inline style.
     * ══════════════════════════════════════════════════════════════ */
    /* Shared base for header bar, metric cards, and chart panels */
    .top, .card, .panel {
      position: relative;
      overflow: hidden;
      background:
        radial-gradient(120% 160% at 0% 0%, rgba(0, 229, 255, 0.10), transparent 58%),
        radial-gradient(140% 150% at 100% 0%, rgba(255, 43, 214, 0.10), transparent 58%),
        linear-gradient(155deg, rgba(7, 12, 30, 0.93) 0%, rgba(7, 10, 27, 0.87) 100%);
      border: 1px solid var(--line);
      box-shadow:
        inset 0 0 0 1px rgba(0, 229, 255, 0.06),
        0 0 26px var(--panelGlowA),
        0 0 36px var(--panelGlowB);
    }
    .card {
      --card-border: rgba(100, 180, 255, 0.45);
      --card-glow: rgba(100, 180, 255, 0.18);
      border: 2px solid var(--card-border);
      box-shadow:
        inset 0 0 0 1px color-mix(in srgb, var(--card-border) 15%, transparent),
        0 0 18px var(--card-glow),
        0 0 32px var(--card-glow);
    }
    .top::before, .panel::before {
      content: "";
      position: absolute;
      inset: 0;
      border-radius: inherit;
      padding: 1px;
      background: linear-gradient(110deg, rgba(0, 229, 255, 0.95), rgba(57, 255, 20, 0.9), rgba(255, 43, 214, 0.95), rgba(255, 230, 0, 0.9));
      background-size: 250% 250%;
      opacity: 0.35;
      pointer-events: none;
      -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
      -webkit-mask-composite: xor;
      mask-composite: exclude;
    }
    .card::before {
      content: "";
      position: absolute;
      inset: 0;
      border-radius: inherit;
      pointer-events: none;
      opacity: 0;
    }
    .top::after, .panel::after {
      content: "";
      position: absolute;
      inset: 0;
      border-radius: inherit;
      pointer-events: none;
      background: linear-gradient(180deg, rgba(255, 255, 255, 0.07), transparent 36%);
      opacity: 0.35;
    }
    /* ══════════════════════════════════════════════════════════════
     * HEADER BAR — Title, subtitle, status dot, display FPS, audio
     * ══════════════════════════════════════════════════════════════ */
    .top {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      border-radius: 16px;
      padding: 14px 18px;
    }
    .title {
      display: flex;
      flex-direction: column;
      gap: 3px;
      position: relative;
      z-index: 2;
    }
    .title h1 {
      margin: 0;
      font-size: 24px;
      letter-spacing: 0.5px;
      font-weight: 700;
      color: #f5fbff;
      text-shadow: 0 0 14px rgba(0, 229, 255, 0.45), 0 0 28px rgba(57, 255, 20, 0.24);
    }
    .subtitle {
      color: var(--muted);
      font-size: 13px;
      text-shadow: 0 0 9px rgba(0, 229, 255, 0.14);
    }
    .status {
      display: inline-flex;
      align-items: center;
      gap: 9px;
      font-size: 13px;
      border: 1px solid rgba(0, 229, 255, 0.33);
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(2, 6, 23, 0.80);
      box-shadow: inset 0 0 14px rgba(0, 229, 255, 0.18), 0 0 16px rgba(0, 229, 255, 0.16);
      position: relative;
      z-index: 2;
    }
    .top-right {
      display: inline-flex;
      align-items: center;
      gap: 10px;
      position: relative;
      z-index: 2;
    }
    .display-fps-box {
      display: inline-flex;
      flex-direction: column;
      align-items: center;
      gap: 2px;
      border: 1px solid rgba(0, 229, 255, 0.33);
      padding: 4px 12px;
      border-radius: 999px;
      background: rgba(2, 6, 23, 0.80);
      box-shadow: inset 0 0 14px rgba(0, 229, 255, 0.12), 0 0 14px rgba(0, 229, 255, 0.10);
    }
    .display-fps-label {
      font-size: 8px;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      color: rgba(0, 229, 255, 0.55);
    }
    .display-fps-value {
      font-family: "LED Dot-Matrix", "Dot Matrix", "DotGothic16", "Courier New", monospace;
      font-size: 16px;
      color: #c8e8ff;
      text-shadow: 0 0 5px rgba(100, 160, 255, 0.7), 0 0 14px rgba(60, 120, 255, 0.55), 0 0 28px rgba(40, 80, 255, 0.4);
      line-height: 1;
    }
    .audio-toggle {
      border: 1px solid rgba(0, 229, 255, 0.33);
      background: rgba(2, 6, 23, 0.80);
      color: var(--ink);
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 12px;
      line-height: 1;
      letter-spacing: 0.4px;
      text-transform: uppercase;
      cursor: pointer;
      box-shadow: inset 0 0 14px rgba(0, 229, 255, 0.12), 0 0 14px rgba(0, 229, 255, 0.10);
    }
    .audio-toggle.on {
      border-color: rgba(57, 255, 20, 0.50);
      box-shadow: inset 0 0 14px rgba(57, 255, 20, 0.18), 0 0 14px rgba(57, 255, 20, 0.14);
      color: #d9ffd0;
    }
    .audio-toggle:disabled {
      opacity: 0.55;
      cursor: default;
    }
    /* Status LED dot (connection indicator) */
    .dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: var(--accentC);
      box-shadow: 0 0 0 7px rgba(57, 255, 20, 0.2), 0 0 14px rgba(57, 255, 20, 0.5);
    }
    .cards {
      display: grid;
      grid-template-columns: repeat(12, minmax(0, 1fr));
      gap: 10px;
    }
    .card {
      grid-column: span 2;
      border-radius: 14px;
      padding: 6px 9px;
      min-height: 44px;
      display: flex;
      flex-direction: column;
      justify-content: flex-start;
      gap: 3px;
      overflow: hidden;
    }
    .label {
      color: #a5bfde;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      text-shadow: 0 0 8px rgba(0, 229, 255, 0.18);
      position: relative;
      z-index: 2;
    }
    /* ══════════════════════════════════════════════════════════════
     * METRIC VALUES — VFD-style text, fixed-width char cells,
     * record-high labels, mini-chart inline layout
     * ══════════════════════════════════════════════════════════════ */
    .value {
      font-size: 28px;
      line-height: 1;
      font-weight: 700;
      letter-spacing: 0.35px;
      color: #f0fbff;
      text-shadow: 0 0 10px rgba(0, 229, 255, 0.28), 0 0 22px rgba(57, 255, 20, 0.16);
      position: relative;
      z-index: 2;
    }
    .card:not(.gauge-card) .value {
      font-family: "LED Dot-Matrix", "Dot Matrix", "DotGothic16", "Courier New", monospace;
      color: #c8e8ff;
      font-weight: 400;
      letter-spacing: normal;
      font-variant-numeric: normal;
      text-shadow:
        0 0 5px rgba(100, 160, 255, 0.7),
        0 0 14px rgba(60, 120, 255, 0.55),
        0 0 28px rgba(40, 80, 255, 0.45),
        0 0 48px rgba(30, 60, 220, 0.3);
      filter:
        drop-shadow(0 0 8px rgba(60, 130, 255, 0.5))
        drop-shadow(0 0 18px rgba(40, 80, 255, 0.35));
    }
    .panel-vfd {
      font-family: "LED Dot-Matrix", "Dot Matrix", "DotGothic16", "Courier New", monospace;
      font-size: 21px;
      font-weight: 400;
      color: #c8e8ff;
      letter-spacing: normal;
      font-variant-numeric: normal;
      text-shadow:
        0 0 5px rgba(100, 160, 255, 0.7),
        0 0 14px rgba(60, 120, 255, 0.55),
        0 0 28px rgba(40, 80, 255, 0.45),
        0 0 48px rgba(30, 60, 220, 0.3);
      filter:
        drop-shadow(0 0 8px rgba(60, 130, 255, 0.5))
        drop-shadow(0 0 18px rgba(40, 80, 255, 0.35));
      margin: -4px 0 2px;
      position: relative;
      z-index: 2;
    }
    .value-inline {
      display: inline-flex;
      align-items: center;
      gap: 10px;
    }
    #mQ, #mLoss, #mGrad, #mRplF {
      display: inline-grid;
      grid-auto-flow: column;
      grid-auto-columns: 0.62em;
      align-items: center;
      justify-content: start;
      white-space: nowrap;
      font-variant-ligatures: none;
      font-kerning: none;
      letter-spacing: 0 !important;
      line-height: 1;
      font-size: 144%;
    }
    #mQ .qch, #mLoss .qch, #mGrad .qch, #mRplF .qch {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 0.72em;
      min-width: 0.72em;
    }
    #mRplF {
      font-size: 100%;
    }
    .mini-metric-card .mini-inline {
      display: grid;
      grid-template-columns: minmax(0, 23fr) minmax(0, 37fr);
      grid-template-rows: auto 1fr;
      align-items: stretch;
      column-gap: 8px;
      min-height: 0;
      flex: 1 1 auto;
      overflow: hidden;
    }
    .mini-metric-card .mini-inline .mini-canvas {
      grid-column: 2;
      grid-row: 1 / 3;
    }
    .mini-metric-card .mini-legend {
      display: flex;
      flex-direction: column;
      align-self: stretch;
      justify-content: center;
      gap: 0px;
      font-size: 9px;
      color: #adc4df;
      z-index: 2;
      margin-top: 0;
      padding-left: 8px;
    }
    .mini-metric-card .mini-legend .sw {
      width: 8px;
      height: 8px;
      border-radius: 2px;
      display: inline-block;
      margin-right: 3px;
      position: relative;
      top: 1px;
      box-shadow: 0 0 6px currentColor;
    }
    .mini-metric-card .mini-inline .value {
      justify-self: start;
      min-width: 0;
      white-space: nowrap;
      font-size: 21px;
    }
    .record-row {
      display: flex;
      align-items: baseline;
      gap: 6px;
      margin-top: 2px;
      position: relative;
      z-index: 2;
    }
    .record-label {
      font-size: 10px;
      color: var(--muted);
      letter-spacing: 0.5px;
      text-transform: uppercase;
    }
    .record-value {
      font-family: "LED Dot-Matrix", "Dot Matrix", "DotGothic16", "Courier New", monospace;
      font-size: 11px;
      color: #c8e8ff;
      letter-spacing: 0.3px;
      text-shadow:
        0 0 5px rgba(100, 160, 255, 0.7),
        0 0 14px rgba(60, 120, 255, 0.55),
        0 0 28px rgba(40, 80, 255, 0.45);
      filter:
        drop-shadow(0 0 6px rgba(60, 130, 255, 0.5));
    }
    .mini-metric-card .mini-canvas {
      width: 100%;
      max-width: 100%;
      height: auto;
      min-height: 0;
      max-height: 100%;
      display: block;
      border-radius: 8px;
      border: none;
      background:
        linear-gradient(180deg, rgba(2, 6, 23, 0.40), rgba(2, 6, 23, 0.55)),
        repeating-linear-gradient(0deg, rgba(120, 150, 210, 0.035) 0px, rgba(120, 150, 210, 0.035) 1px, transparent 1px, transparent 4px);
      box-shadow: inset 0 0 14px rgba(0, 229, 255, 0.10), 0 0 12px rgba(0, 229, 255, 0.09);
      position: relative;
      z-index: 2;
      flex: 1 1 auto;
      align-self: stretch;
      justify-self: end;
    }
    .mini-metric-card > .mini-canvas {
      flex: 1 1 auto;
      min-height: 0;
      height: auto;
      align-self: stretch;
    }
    /* Per-metric status LED (colored via JS) */
    .metric-led {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: var(--neonRed);
      box-shadow: 0 0 0 4px rgba(255, 42, 85, 0.24), 0 0 12px rgba(255, 42, 85, 0.58);
      flex: 0 0 auto;
    }
    .gauge-card {
      grid-column: span 2;
      grid-row: span 2;
      min-height: 200px;
      order: -30;
    }
    .preview-card {
      grid-column: span 4;
      grid-row: span 2;
      min-height: 200px;
      gap: 6px;
      order: -20;
    }
    .client-table-card {
      grid-column: span 4;
      grid-row: span 2;
      min-height: 200px;
      gap: 8px;
      order: -19;
    }
    .chat-card {
      grid-column: span 4;
      grid-row: span 2;
      min-height: 200px;
      gap: 8px;
      order: -18;
    }
    .chat-window {
      flex: 1 1 auto;
      min-height: 0;
      max-height: 420px;
      overflow-y: auto;
      border: 1px solid rgba(80, 160, 255, 0.35);
      border-radius: 10px;
      padding: 0;
      background: rgba(2, 6, 23, 0.78);
      font-size: 12px;
      line-height: 1.35;
      color: #c8e8ff;
      text-shadow: 0 0 6px rgba(60, 130, 255, 0.35);
      scrollbar-width: thin;
      scrollbar-color: rgba(85, 120, 170, 0.95) rgba(4, 10, 20, 0.92);
      scrollbar-gutter: stable;
    }
    .chat-window::-webkit-scrollbar {
      width: 10px;
      height: 10px;
    }
    .chat-window::-webkit-scrollbar-track {
      background: rgba(4, 10, 20, 0.92);
      border-left: 1px solid rgba(100, 180, 255, 0.10);
      border-radius: 999px;
    }
    .chat-window::-webkit-scrollbar-thumb {
      background: linear-gradient(180deg, rgba(85, 120, 170, 0.98), rgba(55, 80, 120, 0.98));
      border: 1px solid rgba(120, 180, 255, 0.18);
      border-radius: 999px;
      box-shadow: inset 0 0 6px rgba(200, 230, 255, 0.08);
    }
    .chat-window::-webkit-scrollbar-thumb:hover {
      background: linear-gradient(180deg, rgba(105, 145, 200, 0.98), rgba(65, 95, 145, 0.98));
    }
    .chat-row {
      display: grid;
      grid-template-columns: 11ch 10ch minmax(0, 1fr);
      gap: 6px;
      align-items: baseline;
      margin-bottom: 6px;
      overflow-wrap: anywhere;
      white-space: pre-wrap;
    }
    .chat-messages {
      padding: 8px;
      padding-top: 6px;
    }
    .chat-row:last-child {
      margin-bottom: 0;
    }
    .chat-columns {
      display: grid;
      grid-template-columns: 11ch 10ch minmax(0, 1fr);
      gap: 6px;
      padding: 8px;
      padding-bottom: 0;
      color: #7dd3fc;
      font-size: 8.4px;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      align-items: baseline;
      position: sticky;
      top: 0;
      z-index: 1;
      background: rgba(2, 6, 23, 0.96);
    }
    .chat-columns > div:nth-child(2) {
      text-align: left;
    }
    .chat-time {
      color: #93c5fd;
      white-space: nowrap;
      font-size: 8.4px;
      line-height: 1.15;
      margin-top: 4px;
    }
    .chat-sender {
      color: #7dd3fc;
      font-weight: 600;
      font-size: 8.4px;
      line-height: 1.15;
      margin-top: 4px;
      white-space: nowrap;
    }
    .chat-text {
      min-width: 0;
    }
    .chat-controls {
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .chat-input {
      width: 100%;
      border: 1px solid rgba(80, 160, 255, 0.45);
      border-radius: 8px;
      padding: 6px 8px;
      background: rgba(2, 6, 23, 0.88);
      color: #e5f4ff;
      font-size: 12px;
      outline: none;
    }
    .chat-input:focus {
      border-color: rgba(57, 255, 20, 0.55);
      box-shadow: inset 0 0 12px rgba(57, 255, 20, 0.12), 0 0 12px rgba(57, 255, 20, 0.10);
    }
    .chat-name-input {
      flex: 0 0 16ch;
      width: 16ch;
      max-width: 16ch;
      min-width: 16ch;
      justify-self: start;
    }
    #chatInput {
      flex: 1 1 auto;
      min-width: 0;
    }
    .chat-submit {
      flex: 0 0 auto;
      border: 1px solid rgba(57, 255, 20, 0.42);
      border-radius: 8px;
      padding: 6px 10px;
      background: rgba(15, 23, 42, 0.84);
      color: #d9ffd0;
      font-size: 12px;
      cursor: pointer;
      white-space: nowrap;
    }
    .chat-submit:disabled,
    .chat-input:disabled {
      opacity: 0.6;
      cursor: default;
    }
    .chat-status {
      min-height: 1.1em;
      font-size: 11px;
      color: #93c5fd;
    }
    @media (max-width: 720px) {
      .chat-columns,
      .chat-row {
        grid-template-columns: 11ch 10ch minmax(0, 1fr);
      }
      .chat-controls {
        flex-wrap: wrap;
      }
      .chat-name-input {
        flex-basis: 16ch;
      }
      #chatInput {
        flex-basis: min(100%, 20rem);
      }
    }
    .dell-card {
      display: grid;
      grid-template-rows: auto 1fr;
      gap: 6px;
    }
    .dell-wrap {
      display: flex;
      align-items: center;
      gap: 10px;
      min-height: 0;
      border: 1px solid rgba(120, 190, 255, 0.35);
      border-radius: 10px;
      padding: 8px;
      background:
        radial-gradient(120% 130% at 0% 0%, rgba(0, 229, 255, 0.14), transparent 62%),
        rgba(2, 6, 23, 0.80);
      box-shadow: inset 0 0 14px rgba(0, 229, 255, 0.10), 0 0 12px rgba(0, 229, 255, 0.08);
    }
    .dell-badge {
      width: 44px;
      height: 44px;
      flex: 0 0 auto;
      display: grid;
      place-items: center;
      border-radius: 50%;
      border: 2px solid rgba(125, 211, 252, 0.75);
      color: #bfe9ff;
      font-family: "DotGothic16", "Courier New", monospace;
      font-size: 11px;
      letter-spacing: 0.4px;
      text-shadow: 0 0 10px rgba(56, 189, 248, 0.65);
      box-shadow: inset 0 0 12px rgba(56, 189, 248, 0.18), 0 0 14px rgba(56, 189, 248, 0.22);
      background: rgba(8, 18, 38, 0.85);
    }
    .dell-meta {
      min-width: 0;
      display: grid;
      gap: 2px;
      color: #c8e8ff;
      font-family: "LED Dot-Matrix", "Dot Matrix", "DotGothic16", "Courier New", monospace;
      text-shadow: 0 0 8px rgba(96, 165, 250, 0.45);
      font-size: 11px;
      line-height: 1.25;
    }
    .preview-wrap {
      position: relative;
      flex: 1 1 auto;
      min-height: 0;
      border-radius: 10px;
      overflow: hidden;
      border: 1px solid rgba(80, 160, 255, 0.35);
      background:
        radial-gradient(120% 120% at 10% 0%, rgba(0, 229, 255, 0.16), transparent 60%),
        radial-gradient(120% 120% at 95% 100%, rgba(57, 255, 20, 0.13), transparent 60%),
        rgba(2, 6, 23, 0.95);
      box-shadow: inset 0 0 18px rgba(0, 229, 255, 0.16), 0 0 16px rgba(0, 229, 255, 0.14);
    }
    .preview-canvas {
      display: block;
      width: 100%;
      height: 100%;
      border: none;
      background: #020617;
      box-shadow: none;
      image-rendering: pixelated;
    }
    .preview-video {
      display: none;
      width: 100%;
      height: 100%;
      object-fit: contain;
      background: #020617;
    }
    .preview-msg {
      position: absolute;
      inset: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 8px;
      font-family: "LED Dot-Matrix", "Dot Matrix", "DotGothic16", "Courier New", monospace;
      font-size: 16px;
      color: #c8e8ff;
      text-transform: uppercase;
      letter-spacing: 0.4px;
      text-shadow:
        0 0 5px rgba(100, 160, 255, 0.7),
        0 0 14px rgba(60, 120, 255, 0.55),
        0 0 28px rgba(40, 80, 255, 0.45);
      background: linear-gradient(180deg, rgba(2, 6, 23, 0.28), rgba(2, 6, 23, 0.50));
      pointer-events: none;
    }
    .preview-head-controls {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      min-width: 0;
      flex-wrap: wrap;
      justify-content: flex-end;
    }
    .preview-quality-select {
      background: rgba(10, 20, 40, 0.85);
      color: #c8e8ff;
      border: 1px solid rgba(100, 180, 255, 0.35);
      border-radius: 4px;
      padding: 2px 6px;
      font-size: 11px;
      font-family: inherit;
      cursor: pointer;
      outline: none;
    }
    .preview-quality-select:focus {
      border-color: rgba(100, 180, 255, 0.7);
      box-shadow: 0 0 6px rgba(100, 180, 255, 0.3);
    }
    .client-table-count {
      font-family: "LED Dot-Matrix", "Dot Matrix", "DotGothic16", "Courier New", monospace;
      font-size: 16px;
      color: #c8e8ff;
      text-shadow:
        0 0 5px rgba(100, 160, 255, 0.7),
        0 0 14px rgba(60, 120, 255, 0.55),
        0 0 28px rgba(40, 80, 255, 0.45);
    }
    .client-table-wrap {
      flex: 1 1 auto;
      min-height: 0;
      max-height: 420px;
      overflow: auto;
      border-radius: 10px;
      border: 1px solid rgba(80, 160, 255, 0.35);
      background:
        radial-gradient(120% 120% at 10% 0%, rgba(0, 229, 255, 0.12), transparent 60%),
        radial-gradient(120% 120% at 95% 100%, rgba(57, 255, 20, 0.10), transparent 60%),
        rgba(2, 6, 23, 0.90);
      box-shadow: inset 0 0 18px rgba(0, 229, 255, 0.12), 0 0 16px rgba(0, 229, 255, 0.10);
      scrollbar-width: thin;
      scrollbar-color: rgba(85, 120, 170, 0.95) rgba(4, 10, 20, 0.92);
    }
    .client-table-wrap::-webkit-scrollbar {
      width: 10px;
      height: 10px;
    }
    .client-table-wrap::-webkit-scrollbar-track {
      background: rgba(4, 10, 20, 0.92);
      border-left: 1px solid rgba(100, 180, 255, 0.10);
      border-radius: 999px;
    }
    .client-table-wrap::-webkit-scrollbar-thumb {
      background: linear-gradient(180deg, rgba(85, 120, 170, 0.98), rgba(55, 80, 120, 0.98));
      border: 1px solid rgba(120, 180, 255, 0.18);
      border-radius: 999px;
      box-shadow: inset 0 0 6px rgba(200, 230, 255, 0.08);
    }
    .client-table-wrap::-webkit-scrollbar-thumb:hover {
      background: linear-gradient(180deg, rgba(105, 145, 200, 0.98), rgba(65, 95, 145, 0.98));
    }
    .client-table {
      width: 100%;
      border-collapse: collapse;
      font-family: "LED Dot-Matrix", "Dot Matrix", "DotGothic16", "Courier New", monospace;
      color: #c8e8ff;
      font-size: 13px;
      letter-spacing: 0.1px;
    }
    .client-table col.client-col-id {
      width: 5.2ch;
    }
    .client-table col.client-col-duration {
      width: 6.2ch;
    }
    .client-table col.client-col-lives {
      width: 3.6ch;
    }
    .client-table col.client-col-level {
      width: 3.6ch;
    }
    .client-table thead th {
      position: sticky;
      top: 0;
      z-index: 1;
      padding: 0;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.6px;
      color: #a5bfde;
      background: rgba(7, 15, 32, 0.96);
      text-align: left;
      border-bottom: 1px solid rgba(100, 180, 255, 0.26);
    }
    .client-table-sort-btn {
      width: 100%;
      display: inline-flex;
      align-items: center;
      justify-content: flex-start;
      gap: 6px;
      padding: 8px 10px;
      border: none;
      background: transparent;
      color: inherit;
      font: inherit;
      letter-spacing: inherit;
      text-transform: inherit;
      cursor: pointer;
    }
    .client-table-sort-btn:hover {
      color: #d8eeff;
    }
    .client-table th.num .client-table-sort-btn {
      justify-content: flex-end;
      text-align: right;
    }
    .client-table-sort-btn.active {
      color: #e9fbff;
      text-shadow: 0 0 8px rgba(0, 200, 255, 0.25);
    }
    .client-table-sort-indicator {
      min-width: 0.8em;
      text-align: center;
      color: #00c8ff;
    }
    .client-table th.num,
    .client-table td.num {
      text-align: right;
    }
    .client-table tbody tr {
      border-top: 1px solid rgba(100, 180, 255, 0.14);
      transition: background 0.15s ease;
    }
    .client-table tbody tr:hover {
      background: rgba(100, 180, 255, 0.08);
    }
    .client-table tbody tr.selected {
      background: linear-gradient(90deg, rgba(0, 200, 255, 0.16), rgba(57, 255, 20, 0.08));
      box-shadow: inset 3px 0 0 rgba(0, 200, 255, 0.65);
    }
    .client-table tbody tr.preview-capable {
      cursor: pointer;
    }
    .client-table tbody tr.inactive {
      opacity: 0.72;
      cursor: default;
    }
    .client-table td {
      padding: 7px 10px;
      white-space: nowrap;
    }
    .client-table thead th:nth-child(1) .client-table-sort-btn,
    .client-table tbody td:nth-child(1) {
      padding-left: 7px;
      padding-right: 6px;
    }
    .client-table thead th:nth-child(2) .client-table-sort-btn,
    .client-table tbody td:nth-child(2) {
      padding-left: 6px;
      padding-right: 6px;
    }
    .client-table thead th:nth-child(4) .client-table-sort-btn,
    .client-table tbody td:nth-child(4),
    .client-table thead th:nth-child(5) .client-table-sort-btn,
    .client-table tbody td:nth-child(5) {
      padding-left: 5px;
      padding-right: 5px;
    }
    .client-table-empty {
      padding: 18px 10px;
      color: #8fa9c9;
      text-align: center;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    .preview-toggle {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      cursor: pointer;
      font-size: 11px;
      color: #b0c8e8;
    }
    .preview-toggle input {
      display: none;
    }
    .mobile-summary-meter-card {
      order: 10;
    }
    .card-narrow {
      grid-column: span 1;
      min-height: 0;
      padding: 6px 9px;
      gap: 2px;
    }
    .card-half {
      min-height: 0;
      padding: 6px 9px;
      gap: 2px;
    }
    .card-half.mini-metric-card .mini-inline {
      min-height: 0;
      flex: 1 1 auto;
    }
    .card-half.mini-metric-card .mini-canvas {
      height: auto;
      min-height: 0;
      flex: 1 1 auto;
      border-radius: 4px;
    }
    .system-mini-card .mini-inline {
      min-height: 0;
      align-items: stretch;
    }
    .system-mini-card .mini-canvas {
      height: auto;
      min-height: 0;
    }
    .system-mini-stack {
      display: flex;
      flex-direction: column;
      justify-content: center;
      min-width: 0;
    }
    .system-mini-stack .value {
      line-height: 0.95;
    }
    .system-mini-sub {
      margin-top: 4px;
      font-size: 9px;
      color: #adc4df;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      text-shadow: 0 0 6px rgba(0, 229, 255, 0.10);
      position: relative;
      z-index: 2;
    }
    .system-mini-subval {
      margin-top: 1px;
      font-family: "LED Dot-Matrix", "Dot Matrix", "DotGothic16", "Courier New", monospace;
      font-size: 14px;
      color: #c8e8ff;
      text-shadow:
        0 0 4px rgba(100, 160, 255, 0.6),
        0 0 10px rgba(60, 120, 255, 0.45),
        0 0 20px rgba(40, 80, 255, 0.3);
      line-height: 1;
      position: relative;
      z-index: 2;
    }
    /* Game-settings card controls */
    .game-settings-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      margin-top: 4px;
    }
    .game-settings-row label {
      font-size: 11px;
      color: #b0c8e8;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 4px;
    }
    .game-settings-row select {
      background: rgba(10, 20, 40, 0.85);
      color: #c8e8ff;
      border: 1px solid rgba(100, 180, 255, 0.35);
      border-radius: 4px;
      padding: 2px 4px;
      font-size: 11px;
      font-family: inherit;
      cursor: pointer;
      outline: none;
    }
    .game-settings-row select:focus {
      border-color: rgba(100, 180, 255, 0.7);
      box-shadow: 0 0 6px rgba(100, 180, 255, 0.3);
    }
    /* Dark toggle switch */
    .toggle-switch { position: relative; display: inline-block; width: 28px; height: 14px; flex-shrink: 0; }
    .toggle-switch input { opacity: 0; width: 0; height: 0; position: absolute; }
    .toggle-switch .slider {
      position: absolute; inset: 0; border-radius: 7px; cursor: pointer;
      background: rgba(60, 80, 110, 0.7); transition: background 0.2s;
    }
    .toggle-switch .slider::before {
      content: ""; position: absolute; left: 2px; top: 2px;
      width: 10px; height: 10px; border-radius: 50%;
      background: #8899aa; transition: transform 0.2s, background 0.2s;
    }
    .toggle-switch input:checked + .slider { background: rgba(0, 200, 255, 0.45); }
    .toggle-switch input:checked + .slider::before { transform: translateX(14px); background: #00c8ff; }
    .toggle-switch input:disabled + .slider { opacity: 0.4; cursor: default; }
    /* Preview checkbox indicator */
    .preview-checkbox-indicator::after {
      content: "✓";
      position: absolute;
      color: #00c8ff;
      font-size: 10px;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%) scale(0);
      transition: transform 0.2s;
      text-shadow: 0 0 4px rgba(0, 200, 255, 0.8);
    }
    .preview-toggle input:checked + .preview-checkbox-indicator {
      background: rgba(0, 200, 255, 0.35);
      border-color: rgba(0, 200, 255, 0.7);
      box-shadow: 0 0 6px rgba(0, 200, 255, 0.4);
    }
    .preview-toggle input:checked + .preview-checkbox-indicator::after {
      transform: translate(-50%, -50%) scale(1);
    }
    /* Up/down override controls for Epsilon / Expert */
    .ud-col { display:flex; flex-direction:column; gap:0; line-height:1; }
    .ud-btn {
      background:none; border:none; color:#556; cursor:pointer;
      font-size:9px; padding:0 2px; line-height:1; user-select:none;
      transition: color 0.15s;
    }
    .ud-btn:hover { color:#0ff; }
    .ud-btn:disabled { opacity:0.25; cursor:default; color:#556; }
    .gauge-card canvas {
      border: none;
      background: transparent;
      box-shadow: none;
    }
    .gauge-head {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 8px;
      padding: 0 4px;
    }
    /* ── Chart panels ────────────────────────────────────────────── */
    .charts {
      display: grid;
      grid-template-columns: repeat(2, minmax(320px, 1fr));
      gap: 14px;
    }
    .panel {
      border-radius: 14px;
      padding: 12px;
      min-height: 224px;
      display: grid;
      grid-template-rows: auto auto auto 1fr;
      gap: 4px;
      overflow: hidden;
    }
    .panel h2 {
      margin: 0;
      font-size: 16px;
      font-weight: 640;
      letter-spacing: 0.35px;
      color: #effbff;
      text-shadow: 0 0 10px rgba(0, 229, 255, 0.28), 0 0 22px rgba(255, 43, 214, 0.18);
      position: relative;
      z-index: 2;
    }
    .legend {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      color: #adc4df;
      font-size: 12px;
      text-shadow: 0 0 8px rgba(0, 229, 255, 0.14);
      position: relative;
      z-index: 2;
    }
    .legend .sw {
      width: 10px;
      height: 10px;
      border-radius: 2px;
      display: inline-block;
      margin-right: 6px;
      position: relative;
      top: 1px;
      box-shadow: 0 0 10px currentColor;
    }
    canvas {
      display: block;
      width: 100%;
      height: 210px;
      border-radius: 10px;
      border: 1px solid rgba(0, 229, 255, 0.26);
      background:
        linear-gradient(180deg, rgba(2, 6, 23, 0.55), rgba(2, 6, 23, 0.65)),
        repeating-linear-gradient(0deg, rgba(120, 150, 210, 0.04) 0px, rgba(120, 150, 210, 0.04) 1px, transparent 1px, transparent 5px);
      box-shadow: inset 0 0 28px rgba(0, 229, 255, 0.10), 0 0 20px rgba(0, 229, 255, 0.12);
      position: relative;
      z-index: 2;
    }
    /* ── Responsive breakpoints ───────────────────────────────────── */
    @media (max-width: 1300px) {
      .cards { grid-template-columns: repeat(8, minmax(0, 1fr)); }
      .gauge-card { grid-column: span 2; grid-row: span 2; min-height: 180px; }
      .preview-card { grid-column: span 4; grid-row: span 1; min-height: 124px; }
      .client-table-card { grid-column: span 4; grid-row: span 1; min-height: 124px; }
      .chat-card { grid-column: span 4; grid-row: span 1; min-height: 124px; }
    }
    @media (max-width: 950px) {
      .cards { grid-template-columns: repeat(4, minmax(0, 1fr)); }
      .charts { grid-template-columns: 1fr; }
      .top { flex-direction: column; align-items: flex-start; }
      .gauge-card { grid-column: span 2; order: -40; }
      .mobile-summary-meter-card { grid-column: span 2; order: -40; }
      .preview-card { grid-column: span 4; grid-row: span 1; min-height: 150px; }
      .client-table-card { grid-column: span 4; grid-row: span 1; min-height: 150px; }
      .chat-card { grid-column: span 4; grid-row: span 1; min-height: 150px; }
      .mini-metric-card .mini-canvas { height: auto; }
    }
  </style>
</head>
<!-- ══════════════════════════════════════════════════════════════════
     HTML MARKUP — Dashboard layout: cards grid, gauge canvases,
     mini-metric cards with sparklines, time-series chart panels,
     and the header bar (rendered last, positioned via CSS grid).
     ══════════════════════════════════════════════════════════════════ -->
<body>
  <video id="bgVideo" autoplay loop muted playsinline src="/api/html/Starfield.mov"></video>
  <div id="bgOverlay"></div>
  <main>
    <section class="cards">
      <article class="card gauge-card" style="--card-border:rgba(255,60,60,0.66);--card-glow:rgba(255,40,40,0.26)">
        <div class="gauge-head">
          <div class="label">FRAMES PER SECOND</div>
        </div>
        <canvas id="cFpsGauge"></canvas>
      </article>
      <article class="card gauge-card" style="--card-border:rgba(255,220,40,0.66);--card-glow:rgba(255,200,20,0.26)">
        <div class="gauge-head">
          <div class="label">SAMPLES PER SECOND</div>
        </div>
        <canvas id="cStepGauge"></canvas>
      </article>
      <article class="card preview-card" style="--card-border:rgba(80,170,255,0.72);--card-glow:rgba(60,150,255,0.30)">
        <div class="gauge-head">
          <div style="display:flex;align-items:center;gap:10px;">
            <label id="previewToggleWrap" class="preview-toggle">
              <input type="checkbox" id="chkPreviewEnabled" checked>
              <span class="preview-checkbox-indicator" style="display:inline-block;width:12px;height:12px;border:1px solid rgba(100,180,255,0.5);border-radius:3px;position:relative;background:rgba(0,200,255,0.15);transition:all 0.2s;"></span>
            </label>
            <div class="label" id="mPreviewLabel">PREVIEW</div>
            <label class="preview-toggle">
              <input type="checkbox" id="chkHudEnabled">
              <span class="preview-checkbox-indicator" style="display:inline-block;width:12px;height:12px;border:1px solid rgba(100,180,255,0.5);border-radius:3px;position:relative;background:rgba(0,200,255,0.15);transition:all 0.2s;"></span>
              <span>HUD</span>
            </label>
            <label class="preview-toggle">
              <input type="checkbox" id="chkPreviewGameAudio"__PREVIEW_GAME_AUDIO_CHECKED____PREVIEW_GAME_AUDIO_DISABLED__>
              <span class="preview-checkbox-indicator" style="display:inline-block;width:12px;height:12px;border:1px solid rgba(100,180,255,0.5);border-radius:3px;position:relative;background:rgba(0,200,255,0.15);transition:all 0.2s;"></span>
              <span>GAME AUDIO</span>
            </label>
          </div>
          <div class="preview-head-controls">
            <select id="selPreviewQuality" class="preview-quality-select" aria-label="Preview quality">
              <option value="auto" selected>Auto</option>
              <option value="default">High</option>
              <option value="mobile">Low</option>
            </select>
          </div>
        </div>
        <div class="preview-wrap">
          <video id="vGamePreview" class="preview-video" autoplay playsinline></video>
          <canvas id="cGamePreview" class="preview-canvas"></canvas>
          <div id="mGamePreviewMsg" class="preview-msg">No Clients</div>
        </div>
      </article>
      <article class="card client-table-card" style="--card-border:rgba(70,210,255,0.68);--card-glow:rgba(40,150,255,0.24)">
        <div class="gauge-head">
          <div class="label">CONNECTED CLIENTS</div>
          <div class="client-table-count" id="mClientTableCount">0</div>
        </div>
        <div class="client-table-wrap">
          <table class="client-table" aria-label="Connected clients">
            <colgroup>
              <col class="client-col-id">
              <col class="client-col-duration">
              <col>
              <col class="client-col-lives">
              <col class="client-col-level">
              <col>
            </colgroup>
            <thead id="tblClientsHead">
              <tr>
                <th aria-sort="ascending"><button type="button" class="client-table-sort-btn active" data-sort-key="client_id">CLNT<span class="client-table-sort-indicator">▲</span></button></th>
                <th class="num" aria-sort="none"><button type="button" class="client-table-sort-btn" data-sort-key="duration_seconds">DUR<span class="client-table-sort-indicator"></span></button></th>
                <th class="num" aria-sort="none"><button type="button" class="client-table-sort-btn" data-sort-key="efficiency">Efficiency<span class="client-table-sort-indicator"></span></button></th>
                <th class="num" aria-sort="none"><button type="button" class="client-table-sort-btn" data-sort-key="lives">LIV<span class="client-table-sort-indicator"></span></button></th>
                <th class="num" aria-sort="none"><button type="button" class="client-table-sort-btn" data-sort-key="level">LVL<span class="client-table-sort-indicator"></span></button></th>
                <th class="num" aria-sort="none"><button type="button" class="client-table-sort-btn" data-sort-key="score">Score<span class="client-table-sort-indicator"></span></button></th>
              </tr>
            </thead>
            <tbody id="tblClientsBody">
              <tr><td colspan="6" class="client-table-empty">No Clients</td></tr>
            </tbody>
          </table>
        </div>
      </article>
      <article class="card chat-card" style="--card-border:rgba(140,220,255,0.68);--card-glow:rgba(90,170,255,0.24)">
        <div class="gauge-head">
          <div class="label">CHAT</div>
          <div class="client-table-count" id="mChatCount">0</div>
        </div>
        <div class="chat-window" id="chatWindow">
          <div class="chat-columns" aria-hidden="true">
            <div>Date Time</div>
            <div>Name</div>
            <div>Message</div>
          </div>
          <div class="chat-messages" id="chatMessages" aria-live="polite">
            <div class="chat-row">Chat ready.</div>
          </div>
        </div>
        <div class="chat-controls">
          <input id="chatNameInput" class="chat-input chat-name-input" maxlength="16" placeholder="Name" />
          <input id="chatInput" class="chat-input" maxlength="240" placeholder="Say something (max 240 chars)" />
          <button id="chatSubmit" class="chat-submit" type="button">Submit</button>
        </div>
        <div class="chat-status" id="chatStatus"></div>
      </article>
      <article class="card mini-metric-card" style="--card-border:rgba(50,220,80,0.66);--card-glow:rgba(40,200,60,0.26)">
        <div class="label">AVG REWARD 1M</div>
        <div class="mini-inline">
          <div class="value" id="mRwrd">0</div>
          <canvas id="cRewardMini" class="mini-canvas"></canvas>
          <div class="mini-legend"><span><span class="sw" style="background:#ef4444;"></span>100K</span><span><span class="sw" style="background:#22c55e;"></span>1M</span><span><span class="sw" style="background:#38bdf8;"></span>5M</span></div>
        </div>
        <div class="record-row"><span class="record-label">HIGH SCORE:</span><span class="record-value" id="recRwrd">—</span></div>
      </article>
      <article class="card mini-metric-card" style="--card-border:rgba(255,140,30,0.66);--card-glow:rgba(255,120,20,0.26)">
        <div class="label">Q-VALUE RANGE</div>
        <canvas id="cQRange" class="mini-canvas" style="flex:0 1 auto;"></canvas>
      </article>
      <article class="card mini-metric-card" style="--card-border:rgba(60,130,255,0.66);--card-glow:rgba(40,100,255,0.26)">
        <div class="label">Avg Level</div>
        <div class="mini-inline">
          <div class="value" id="mLevel">0.0</div>
          <canvas id="cLevelMini" class="mini-canvas"></canvas>
        </div>
        <div class="record-row"><span class="record-label">Record:</span><span class="record-value" id="recLevel">—</span></div>
      </article>
      <article class="card mini-metric-card" style="--card-border:rgba(255,160,80,0.66);--card-glow:rgba(255,140,60,0.26)">
        <div class="label">EPISODE LENGTH</div>
        <div class="mini-inline">
          <div class="value" id="mEpLen">0</div>
          <canvas id="cEpLenMini" class="mini-canvas"></canvas>
          <div style="grid-column:1;"><div class="record-label" style="margin-bottom:1px;">Duration:</div><div class="record-value" id="mDuration">0:00</div><div class="record-label" style="margin-bottom:1px;margin-top:2px;">Count:</div><div class="record-value" id="mEpisodes">0</div><div class="record-label" style="margin-bottom:1px;margin-top:2px;">Rate:</div><div class="record-value" id="mEpRate">—</div></div>
        </div>
        <div class="record-row"><span class="record-label">Record:</span><span class="record-value" id="recEpLen">—</span></div>
      </article>
      <article class="card mini-metric-card card-half" style="--card-border:rgba(180,80,255,0.66);--card-glow:rgba(160,50,255,0.26)">
        <div class="label">Loss</div>
        <div class="mini-inline">
          <div class="value" id="mLoss">0</div>
          <canvas id="cLossMini" class="mini-canvas"></canvas>
        </div>
      </article>
      <article class="card mini-metric-card card-half" style="--card-border:rgba(255,60,180,0.66);--card-glow:rgba(255,40,160,0.26)">
        <div class="label">Grad Norm</div>
        <div class="mini-inline">
          <div class="value" id="mGrad">0</div>
          <canvas id="cGradMini" class="mini-canvas"></canvas>
        </div>
      </article>
      <article class="card card-half" style="--card-border:rgba(255,100,100,0.66);--card-glow:rgba(255,80,80,0.26)">
        <div style="display:flex;justify-content:space-between;align-items:baseline;">
          <div>
            <div class="label">Epsilon</div>
            <div style="display:flex;align-items:center;gap:2px;">
              <div class="ud-col" id="epsUD">
                <button class="ud-btn" data-field="epsilon_pct" data-dir="1" title="Increase epsilon">&#9650;</button>
                <button class="ud-btn" data-field="epsilon_pct" data-dir="-1" title="Decrease epsilon">&#9660;</button>
              </div>
              <div class="value" id="mEps">0%</div>
            </div>
          </div>
          <div style="text-align:right;">
            <div class="label">Expert</div>
            <div style="display:flex;align-items:center;gap:2px;justify-content:flex-end;">
              <div class="ud-col" id="xprtUD">
                <button class="ud-btn" data-field="expert_pct" data-dir="1" title="Increase expert">&#9650;</button>
                <button class="ud-btn" data-field="expert_pct" data-dir="-1" title="Decrease expert">&#9660;</button>
              </div>
              <div class="value" id="mXprt">0%</div>
            </div>
          </div>
        </div>
        <div id="mPulseStatus" style="font-size:0.55em;color:#888;margin-top:-2px;min-height:1.1em;"></div>
      </article>
      <article class="card card-half" style="--card-border:rgba(255,160,60,0.66);--card-glow:rgba(255,140,40,0.26)">
        <div style="display:flex;justify-content:space-between;align-items:baseline;">
          <div>
            <div class="label">AVG SCORE</div>
            <div class="value" id="mAvgScore">0</div>
          </div>
          <div style="text-align:right;">
            <div class="label">GAMES</div>
            <div class="value" id="mGameCount">0</div>
          </div>
        </div>
      </article>
      <article class="card mini-metric-card card-half" style="--card-border:rgba(100,160,255,0.66);--card-glow:rgba(80,140,255,0.26)">
        <div class="label">LEARNING RATE</div>
        <div class="mini-inline" style="overflow:hidden;min-height:0;flex:1;">
          <div class="value" id="mLr">-</div>
          <canvas id="lrMiniChart" class="mini-canvas"></canvas>
        </div>
      </article>
      <article class="card dell-card" style="--card-border:rgba(110,180,255,0.70);--card-glow:rgba(90,160,255,0.24)">
        <div class="label" style="text-transform:none;">DELL 7875 Precision</div>
        <div class="dell-wrap" aria-label="Dell system badge">
          <div class="dell-badge">DELL</div>
          <div class="dell-meta">
            <div>Threadripper 9995WX</div>
            <div>Dual RTX6000</div>
          </div>
        </div>
      </article>
      <article class="card mini-metric-card card-half system-mini-card" style="--card-border:rgba(80,230,255,0.66);--card-glow:rgba(60,210,255,0.24)">
        <div class="label">GPU 0</div>
        <div class="mini-inline">
          <div class="system-mini-stack"><div class="value" id="mGpu0">0%</div></div>
          <canvas id="cGpu0Mini" class="mini-canvas"></canvas>
        </div>
      </article>
      <article class="card mini-metric-card card-half system-mini-card" style="--card-border:rgba(255,120,220,0.66);--card-glow:rgba(255,90,200,0.24)">
        <div class="label">GPU 1</div>
        <div class="mini-inline">
          <div class="system-mini-stack"><div class="value" id="mGpu1">0%</div></div>
          <canvas id="cGpu1Mini" class="mini-canvas"></canvas>
        </div>
      </article>
      <article class="card mini-metric-card card-half system-mini-card" style="--card-border:rgba(120,255,160,0.66);--card-glow:rgba(90,240,130,0.24)">
        <div class="label">CPU</div>
        <div class="mini-inline">
          <div class="system-mini-stack"><div class="value" id="mCpu">0%</div></div>
          <canvas id="cCpuMini" class="mini-canvas"></canvas>
        </div>
      </article>
      <article class="card mini-metric-card card-half system-mini-card" style="--card-border:rgba(255,215,90,0.66);--card-glow:rgba(255,195,60,0.24)">
        <div class="label">Free Memory</div>
        <div class="mini-inline">
          <div class="system-mini-stack">
            <div class="value" id="mMemFree">0.0G</div>
            <div class="system-mini-sub">Disk Space</div>
            <div class="system-mini-subval" id="mDiskFree">0.0G</div>
          </div>
          <canvas id="cMemFreeMini" class="mini-canvas"></canvas>
        </div>
      </article>
      <article class="card card-half card-narrow mobile-summary-meter-card" style="--card-border:rgba(120,220,60,0.66);--card-glow:rgba(100,200,40,0.26)"><div class="label">Clnt</div><div class="value" id="mClients">0</div></article>
      <article class="card card-half card-narrow mobile-summary-meter-card" style="--card-border:rgba(255,180,60,0.66);--card-glow:rgba(255,160,40,0.26)"><div class="label">Web</div><div class="value" id="mWeb">0</div></article>
      <article class="card" style="--card-border:rgba(100,200,255,0.66);--card-glow:rgba(80,180,255,0.26)">
        <div class="label">AVG INFERENCE</div>
        <div class="value value-inline"><span class="metric-led" id="mInfLed"></span><span id="mInf">0.00ms</span></div>
      </article>
      <article class="card" style="--card-border:rgba(220,180,255,0.66);--card-glow:rgba(200,150,255,0.26)">
        <div class="label">REPLAYS PER FRAME</div>
        <div class="value value-inline"><span class="metric-led" id="mRplLed"></span><span id="mRplF">0.00</span></div>
      </article>
      <article class="card" style="--card-border:rgba(255,220,100,0.66);--card-glow:rgba(255,200,80,0.26)"><div class="label">BUFFER SIZE</div><div class="value" id="mBuf">0k (0%)</div></article>
      <article class="card" style="--card-border:rgba(200,100,255,0.66);--card-glow:rgba(180,80,255,0.26)"><div class="label">Q Range</div><div class="value" id="mQ">-</div></article>
      <article class="card" style="--card-border:rgba(0,200,255,0.66);--card-glow:rgba(0,180,255,0.26)">
        <div class="label" style="display:flex;justify-content:space-between;align-items:center;">GAME SETTINGS<label style="font-size:10px;color:#b0c8e8;display:flex;align-items:center;gap:5px;font-weight:normal;cursor:pointer;">Automatic <span class="toggle-switch"><input type="checkbox" id="gsAutoCurriculum"><span class="slider"></span></span></label></div>
        <div class="game-settings-row">
          <label>Level:
            <select id="gsLevel"></select>
          </label>
          <label style="gap:6px;">Advanced <span class="toggle-switch"><input type="checkbox" id="gsAdvanced" checked><span class="slider"></span></span></label>
        </div>
      </article>
    </section>

    <section class="charts">
      <article class="panel">
        <h2>Throughput</h2>
        <div class="legend">
          <span><span class="sw" style="background:#22c55e;"></span>FPS</span>
          <span><span class="sw" style="background:#f59e0b;"></span>Samples/Sec</span>
          <span><span class="sw" style="background:#22d3ee;"></span>Avg Lvl (100K)</span>
          <span><span class="sw" style="background:#e879f9;"></span>Ep Len (100K)</span>
        </div>
        <canvas id="cThroughput"></canvas>
      </article>

      <article class="panel">
        <h2>Rewards</h2>
        <div class="legend">
          <span><span class="sw" style="background:#ef4444;"></span>100K</span>
          <span><span class="sw" style="background:#22c55e;"></span>1M</span>
          <span><span class="sw" style="background:#38bdf8;"></span>5M</span>
          <span><span class="sw" style="background:#facc15;"></span>Subj</span>
        </div>
        <canvas id="cRewards"></canvas>
      </article>

      <article class="panel">
        <h2>Learning</h2>
        <div class="legend">
          <span><span class="sw" style="background:#22c55e;"></span>Loss</span>
          <span><span class="sw" style="background:#f59e0b;"></span>Grad Norm</span>
          <span><span class="sw" style="background:#22d3ee;"></span>BC Loss</span>
        </div>
        <canvas id="cLearning"></canvas>
      </article>

      <article class="panel" style="position:relative;">
        <div style="display:flex;align-items:baseline;justify-content:space-between;">
          <h2 style="margin:0;">Performance</h2>
          <div style="display:flex;align-items:baseline;gap:6px;"><span style="font-size:11px;color:#a5bfde;letter-spacing:0.5px;text-transform:uppercase;">Agreement:</span><div class="value panel-vfd" id="mAgreePanel">0.0%</div></div>
        </div>
        <div class="legend">
          <span><span class="sw" style="background:#00c8ff;"></span>Agreement 1M</span>
          <span><span class="sw" style="background:#0090cc55;"></span>Agreement Raw</span>
          <span><span class="sw" style="background:#ef4444;"></span>Avg Lvl</span>
          <span><span class="sw" style="background:#ff9f43;"></span>Avg Score</span>
        </div>
        <canvas id="cAgreement"></canvas>
      </article>

    </section>
    <section class="top">
      <div class="title">
        <h1>Robotron AI Dashboard</h1>
        <div class="subtitle" id="modelDesc">Loading model info…</div>
      </div>
      <div class="top-right">
        <div class="status"><span class="dot" id="statusDot"></span><span id="statusText">Connected</span></div>
        <div class="display-fps-box"><span class="display-fps-label">Display FPS</span><span class="display-fps-value" id="displayFps">0</span></div>
        <button id="audioToggle" class="audio-toggle" type="button">Audio Off</button>
      </div>
    </section>
  </main>
  <audio id="bgAudio" preload="auto" autoplay></audio>

  <script>
    /* ══════════════════════════════════════════════════════════════
     * CONSTANTS — Refresh rates, history depth, gauge ranges
     * ══════════════════════════════════════════════════════════════ */
    const num = new Intl.NumberFormat("en-US");
    const DASH_MAX_FPS = 30;
    const DASH_DEFAULT_FPS = 2;
    const GAME_FPS = 60;
    const DASH_REFRESH_FPS = (() => {
      try {
        const raw = new URLSearchParams(window.location.search).get("fps");
        const parsed = Number(raw);
        if (!Number.isFinite(parsed) || parsed <= 0) return DASH_DEFAULT_FPS;
        return Math.min(DASH_MAX_FPS, parsed);
      } catch (_) {
        return DASH_DEFAULT_FPS;
      }
    })();
    const DASH_REFRESH_MS = Math.max(1, Math.round(1000 / DASH_REFRESH_FPS));
    const HISTORY_WINDOW_MINUTES = 65;
    const MAX_HISTORY_POINTS = Math.max(
      900,
      Math.round((HISTORY_WINDOW_MINUTES * 60 * 1000) / DASH_REFRESH_MS)
    );
    const MAX_CHART_POINTS = 1400;
    const STEP_GAUGE_AVG_WINDOW = 10;
    const GAUGE_MIN_FPS = 0;
    const GAUGE_MAX_FPS = 15000;
    const GAUGE_FPS_RED_MAX = 3000;
    const GAUGE_FPS_YELLOW_MAX = 6000;
    const GAUGE_MIN_STEPS = 0;
    const GAUGE_MAX_STEPS = 80000;
    const AUDIO_PREF_COOKIE = "robotron_dashboard_audio_enabled";
    const PREVIEW_GAME_AUDIO_PREF_COOKIE = "robotron_preview_game_audio_enabled";
    const PREVIEW_GAME_AUDIO_TRANSPORT_ENABLED = __PREVIEW_GAME_AUDIO_TRANSPORT_ENABLED__;
    const AUDIO_START_RETRY_MS = 800;
    const CHAT_POLL_MS = 5000;

    /* ── Display FPS counter ──────────────────────────────────────────── */
    let _dispFpsFrames = 0;
    let _dispFpsLast = performance.now();
    const _dispFpsEl = document.getElementById("displayFps");
    function _tickDisplayFps() {
      _dispFpsFrames++;
      const now = performance.now();
      const elapsed = now - _dispFpsLast;
      if (elapsed >= 1000) {
        const fps = Math.round((_dispFpsFrames * 1000) / elapsed);
        _dispFpsEl.textContent = fps;
        _dispFpsFrames = 0;
        _dispFpsLast = now;
      }
    }
    const CHART_VALUE_SMOOTH_ALPHA = 0.22;
    const MINI_CHART_VALUE_SMOOTH_ALPHA = 0.28;
    let failedPings = 0;
    const CLIENT_ID = (() => {
      try {
        if (window.crypto && window.crypto.randomUUID) return window.crypto.randomUUID();
      } catch (_) {}
      return `c_${Date.now()}_${Math.random().toString(36).slice(2)}`;
    })();
    /* ══════════════════════════════════════════════════════════════
     * AUDIO — Background music playlist with cookie-based preference
     * ══════════════════════════════════════════════════════════════ */
    const bgAudio = document.getElementById("bgAudio");
    const audioToggle = document.getElementById("audioToggle");
    let audioPlaylist = [];
    let audioIndex = 0;
    let audioEnabled = false;
    let audioRetryTimer = null;

    /* ══════════════════════════════════════════════════════════════
     * DOM REFERENCES — Metric card elements, record labels, gauges
     * ══════════════════════════════════════════════════════════════ */
    const cards = {
      clients: document.getElementById("mClients"),
      web: document.getElementById("mWeb"),
      level: document.getElementById("mLevel"),
      inf: document.getElementById("mInf"),
      infLed: document.getElementById("mInfLed"),
      rplf: document.getElementById("mRplF"),
      rplLed: document.getElementById("mRplLed"),
      eps: document.getElementById("mEps"),
      pulseStatus: document.getElementById("mPulseStatus"),
      xprt: document.getElementById("mXprt"),
      avgScore: document.getElementById("mAvgScore"),
      gameCount: document.getElementById("mGameCount"),
      rwrd: document.getElementById("mRwrd"),
      dqnRwrd: null,
      loss: document.getElementById("mLoss"),
      grad: document.getElementById("mGrad"),
      buf: document.getElementById("mBuf"),
      lr: document.getElementById("mLr"),
      q: document.getElementById("mQ"),
      epLen: document.getElementById("mEpLen"),
      duration: document.getElementById("mDuration"),
      episodes: document.getElementById("mEpisodes"),
      epRate: document.getElementById("mEpRate"),
      agreePanel: document.getElementById("mAgreePanel"),
      gpu0: document.getElementById("mGpu0"),
      gpu1: document.getElementById("mGpu1"),
      cpu: document.getElementById("mCpu"),
      memFree: document.getElementById("mMemFree"),
      diskFree: document.getElementById("mDiskFree"),
      previewMsg: document.getElementById("mGamePreviewMsg"),
      previewLabel: document.getElementById("mPreviewLabel"),
      clientTableCount: document.getElementById("mClientTableCount"),
      chatWindow: document.getElementById("chatWindow"),
      chatMessages: document.getElementById("chatMessages"),
      chatNameInput: document.getElementById("chatNameInput"),
      chatInput: document.getElementById("chatInput"),
      chatSubmit: document.getElementById("chatSubmit"),
      chatStatus: document.getElementById("chatStatus"),
      chatCount: document.getElementById("mChatCount"),
    };
    /* Game-settings controls */
    const gsAdvancedEl = document.getElementById("gsAdvanced");
    const gsLevelEl = document.getElementById("gsLevel");
    const gsAutoCurrEl = document.getElementById("gsAutoCurriculum");
    const chkPreviewEl = document.getElementById("chkPreviewEnabled");
    const chkHudEl = document.getElementById("chkHudEnabled");
    const chkPreviewGameAudioEl = document.getElementById("chkPreviewGameAudio");
    const previewToggleWrap = document.getElementById("previewToggleWrap");
    const _gsAdmin = new URLSearchParams(window.location.search).get("admin") === "yes";
    const _chatEnabled = true;
    const _chatNameStorageKey = "robotronChatDisplayName";
    let _chatRows = [];
    let _chatReqInFlight = false;

    function _setChatStatus(text, isError = false) {
      if (!cards.chatStatus) return;
      cards.chatStatus.textContent = text || "";
      cards.chatStatus.style.color = isError ? "#fca5a5" : "#93c5fd";
    }

    function _formatChatTimestamp(rawTs) {
      const ts = Number(rawTs);
      if (!Number.isFinite(ts) || ts <= 0) return "--/-- --:--";
      const d = new Date(ts * 1000);
      const mm = String(d.getMonth() + 1).padStart(2, "0");
      const dd = String(d.getDate()).padStart(2, "0");
      const mins = String(d.getMinutes()).padStart(2, "0");
      const hrs24 = d.getHours();
      const period = hrs24 >= 12 ? "PM" : "AM";
      const hrs12 = (hrs24 % 12) || 12;
      return `${mm}/${dd} ${hrs12}:${mins}${period}`;
    }

    function _renderChatRows(rows) {
      if (!cards.chatWindow || !cards.chatMessages) return;
      cards.chatMessages.innerHTML = "";
      if (!rows || rows.length <= 0) {
        const empty = document.createElement("div");
        empty.className = "chat-row";
        empty.textContent = "No messages yet.";
        cards.chatMessages.appendChild(empty);
      } else {
        for (const row of rows) {
          const line = document.createElement("div");
          line.className = "chat-row";
          const time = document.createElement("span");
          time.className = "chat-time";
          time.textContent = _formatChatTimestamp(row.ts);
          const sender = document.createElement("span");
          sender.className = "chat-sender";
          sender.textContent = String(row.sender || "unknown");
          sender.title = String(row.display_ip || row.sender || "unknown");
          const txt = document.createElement("span");
          txt.className = "chat-text";
          txt.textContent = String(row.text || "");
          line.appendChild(time);
          line.appendChild(sender);
          line.appendChild(txt);
          cards.chatMessages.appendChild(line);
        }
      }
      if (cards.chatCount) cards.chatCount.textContent = String(rows ? rows.length : 0);
      cards.chatWindow.scrollTop = cards.chatWindow.scrollHeight;
    }

    function _sanitizeChatName(rawName) {
      return String(rawName || "").replace(/[\\r\\n]+/g, " ").trim().slice(0, 16);
    }

    async function _fetchChatRows() {
      if (!_chatEnabled) return;
      if (_chatReqInFlight) return;
      _chatReqInFlight = true;
      try {
        const res = await fetch(`/api/chat?cid=${encodeURIComponent(CLIENT_ID)}&limit=120&t=${Date.now()}`, { cache: "no-store" });
        if (!res.ok) throw new Error("chat fetch failed");
        const payload = await res.json();
        const nextRows = Array.isArray(payload && payload.messages) ? payload.messages : [];
        _chatRows = nextRows;
        _renderChatRows(_chatRows);
      } catch (_) {
        _setChatStatus("Chat unavailable", true);
      } finally {
        _chatReqInFlight = false;
      }
    }

    async function _submitChat() {
      if (!_chatEnabled || !cards.chatInput) return;
      const displayName = _sanitizeChatName((cards.chatNameInput && cards.chatNameInput.value) || "");
      if (cards.chatNameInput) cards.chatNameInput.value = displayName;
      const msg = String(cards.chatInput.value || "").replace(/[\\r\\n]+/g, " ").trim();
      if (!msg) {
        _setChatStatus("Type a message first.", true);
        return;
      }
      if (msg.length > 240) {
        _setChatStatus("Message too long (240 max).", true);
        return;
      }
      try {
        if (cards.chatSubmit) cards.chatSubmit.disabled = true;
        const res = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: msg, display_name: displayName }),
        });
        if (!res.ok) throw new Error("chat submit failed");
        cards.chatInput.value = "";
        _setChatStatus("Posted.");
        await _fetchChatRows();
      } catch (_) {
        _setChatStatus("Could not post message.", true);
      } finally {
        if (cards.chatSubmit) cards.chatSubmit.disabled = !_chatEnabled;
      }
    }

    if (cards.chatInput && cards.chatSubmit) {
      if (!_chatEnabled) {
        if (cards.chatNameInput) cards.chatNameInput.disabled = true;
        cards.chatInput.disabled = true;
        cards.chatSubmit.disabled = true;
        cards.chatInput.placeholder = "Chat disabled";
        _setChatStatus("Read-only mode.");
        const chatCard = cards.chatInput.closest(".chat-card");
        if (chatCard) chatCard.style.display = "none";
      } else {
        if (cards.chatNameInput) {
          try {
            cards.chatNameInput.value = _sanitizeChatName(window.localStorage.getItem(_chatNameStorageKey) || "");
          } catch (_) {}
          cards.chatNameInput.addEventListener("input", () => {
            const nextName = _sanitizeChatName(cards.chatNameInput.value || "");
            if (cards.chatNameInput.value !== nextName) cards.chatNameInput.value = nextName;
            try {
              window.localStorage.setItem(_chatNameStorageKey, nextName);
            } catch (_) {}
          });
          cards.chatNameInput.addEventListener("keydown", (ev) => {
            if (ev.key === "Enter") {
              ev.preventDefault();
              _submitChat();
            }
          });
        }
        cards.chatSubmit.disabled = false;
        cards.chatSubmit.addEventListener("click", _submitChat);
        cards.chatInput.addEventListener("keydown", (ev) => {
          if (ev.key === "Enter") {
            ev.preventDefault();
            _submitChat();
          }
        });
      }
    }
    function _ensureRobotronLevelOptions() {
      if (!gsLevelEl) return;
      const currentValue = parseInt(gsLevelEl.value, 10);
      gsLevelEl.innerHTML = "";
      for (let lv = 1; lv <= 81; lv += 1) {
        const opt = document.createElement("option");
        opt.value = String(lv);
        opt.textContent = String(lv);
        if ((Number.isFinite(currentValue) && currentValue === lv) || (!Number.isFinite(currentValue) && lv === 1)) {
          opt.selected = true;
        }
        gsLevelEl.appendChild(opt);
      }
      if (!Number.isFinite(currentValue)) gsLevelEl.value = "1";
    }
    _ensureRobotronLevelOptions();
    if (!_gsAdmin) { 
      gsAdvancedEl.disabled = true; 
      gsLevelEl.disabled = true; 
      gsAutoCurrEl.disabled = true;
      if (chkPreviewEl) {
        chkPreviewEl.disabled = true;
      }
      if (previewToggleWrap) {
        previewToggleWrap.style.cursor = "default";
        previewToggleWrap.style.opacity = "0.65";
      }
    }
    /* Epsilon / Expert up-down buttons — admin gate */
    document.querySelectorAll('#epsUD .ud-btn, #xprtUD .ud-btn').forEach(btn => {
      if (!_gsAdmin) { btn.disabled = true; return; }
      btn.addEventListener('click', () => {
        const field = btn.dataset.field; /* epsilon_pct or expert_pct */
        const dir   = parseInt(btn.dataset.dir, 10); /* +1 or -1 */
        const cur   = field === 'epsilon_pct'
          ? Math.round((_lastNow?.epsilon ?? 0) * 100)
          : Math.round((_lastNow?.expert_ratio ?? 0) * 100);
        const next  = Math.max(0, Math.min(100, cur + dir));
        _postGameSettings({ [field]: next });
        /* Optimistic UI update */
        if (field === 'epsilon_pct') cards.eps.textContent = fmtPct(next / 100);
        else cards.xprt.textContent = fmtPct(next / 100);
      });
    });
    let _lastNow = null;  /* stash latest snapshot for up/down reference */
    let _gsIgnoreSync = false;  /* suppress sync while user is changing */
    async function _postGameSettings(obj) {
      try {
        _gsIgnoreSync = true;
        await fetch("/api/game_settings", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(obj),
        });
      } catch (e) { /* ignore */ }
      finally { setTimeout(() => { _gsIgnoreSync = false; }, 1500); }
    }
    if (_gsAdmin) {
      gsAdvancedEl.addEventListener("change", () => {
        _postGameSettings({ start_advanced: gsAdvancedEl.checked });
      });
      gsLevelEl.addEventListener("change", () => {
        _postGameSettings({ start_level_min: parseInt(gsLevelEl.value, 10) });
      });
      gsAutoCurrEl.addEventListener("change", () => {
        _postGameSettings({ auto_curriculum: gsAutoCurrEl.checked });
        _applyAutoCurriculum(gsAutoCurrEl.checked);
      });
    }
    const _selectableLevels = Array.from({ length: 81 }, (_, i) => i + 1);
    function _computeAutoLevel(avgLevel) {
      const target = Math.floor(avgLevel) - 1;
      let best = _selectableLevels[0];
      for (const lv of _selectableLevels) { if (lv <= target) best = lv; else break; }
      return best;
    }
    function _applyAutoCurriculum(on) {
      gsAdvancedEl.disabled = on || !_gsAdmin;
      gsLevelEl.disabled    = on || !_gsAdmin;
      if (on) {
        if (!gsAdvancedEl.checked) {
          gsAdvancedEl.checked = true;
          _postGameSettings({ start_advanced: true });
        }
        if (_lastNow) {
          const lv = _computeAutoLevel(_lastNow.average_level || 1);
          gsLevelEl.value = String(lv);
          _postGameSettings({ start_level_min: lv });
        }
      }
    }

    /* Preview enable/disable checkbox — admin only */
    if (_gsAdmin && chkPreviewEl) {
      chkPreviewEl.addEventListener("change", async () => {
        try {
          await fetch("/api/preview_settings", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ enabled: chkPreviewEl.checked }),
          });
        } catch (e) {
          console.error("Failed to update preview settings:", e);
        }
      });
    }
    if (chkHudEl) {
      chkHudEl.addEventListener("change", async () => {
        try {
          await fetch("/api/preview_settings", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ hud_enabled: chkHudEl.checked }),
          });
        } catch (e) {
          console.error("Failed to update HUD settings:", e);
        }
      });
    }

    const recEls = {
      rwrd: document.getElementById("recRwrd"),
      level: document.getElementById("recLevel"),
      epLen: document.getElementById("recEpLen"),
    };
    const modelDescEl = document.getElementById("modelDesc");
    const recordHighs = { rwrd: -Infinity, level: -Infinity, epLen: -Infinity };
    let _recordsResetSeqSeen = -1;

    function _clearRecordDisplays() {
      recordHighs.rwrd = -Infinity;
      recordHighs.level = -Infinity;
      recordHighs.epLen = -Infinity;
      if (recEls.rwrd) recEls.rwrd.textContent = "—";
      if (recEls.level) recEls.level.textContent = "—";
      if (recEls.epLen) recEls.epLen.textContent = "—";
    }
    /* Episode rate: 30-second rolling window */
    const epRateHistory = [];  /* {ts, episodes} */
    const EP_RATE_WINDOW = 30; /* seconds */
    const fpsGaugeCanvas = document.getElementById("cFpsGauge");
    const stepGaugeCanvas = document.getElementById("cStepGauge");
    const gamePreviewCanvas = document.getElementById("cGamePreview");
    const gamePreviewVideo = document.getElementById("vGamePreview");
    const previewQualitySelect = document.getElementById("selPreviewQuality");
    const clientTableHead = document.getElementById("tblClientsHead");
    const clientTableBody = document.getElementById("tblClientsBody");
    const _previewSrcCanvas = document.createElement("canvas");
    const _previewSrcCtx = _previewSrcCanvas.getContext("2d");
    let _previewSeqLoaded = -1;
    let _previewFetchInFlight = false;
    let _previewHasFrame = false;
    let _previewPumpRunning = false;
    let _previewRtcPc = null;
    let _previewRtcConnected = false;
    let _previewRtcEnabled = false;
    let _previewRtcError = "";
    let _previewRtcRetryTimer = null;
    let _previewRtcConnecting = false;
    let _previewVideoHasFrame = false;
    let _previewVideoLastFrameTs = 0;
    let _previewRtcStream = null;
    let _previewGameAudioEnabled = PREVIEW_GAME_AUDIO_TRANSPORT_ENABLED;
    let _previewClientRequestInFlight = false;
    let _previewPendingClientId = null;
    let _clientTableSortKey = "client_id";
    let _clientTableSortDir = "asc";
    // Preview is controlled via checkbox; start enabled by default.
    const ENABLE_CLIENT0_PREVIEW = true;

    // ── Gauge needle damping ────────────────────────────────────────
    // Time-constant in seconds: the needle closes ~63% of the gap
    // in this many seconds.  Lower = snappier, higher = smoother.
    const GAUGE_DAMPING_TAU = 0.35;
    const gaugeState = {
      fps:  { current: 0, target: 0 },
      step: { current: 0, target: 0 },
    };
    let latestRow = null;
    let lastGaugeFrameTs = 0;

    function gaugeAnimationLoop(ts) {
      if (!lastGaugeFrameTs) lastGaugeFrameTs = ts;
      const dtSec = Math.min((ts - lastGaugeFrameTs) / 1000, 0.25); // cap to avoid jump after tab-hide
      lastGaugeFrameTs = ts;

      // Exponential ease: factor = 1 - e^(-dt/tau)
      const alpha = 1.0 - Math.exp(-dtSec / GAUGE_DAMPING_TAU);

      let needsRedraw = false;
      for (const g of Object.values(gaugeState)) {
        const diff = g.target - g.current;
        if (Math.abs(diff) > 0.05) {
          g.current += diff * alpha;
          needsRedraw = true;
        } else if (g.current !== g.target) {
          g.current = g.target;
          needsRedraw = true;
        }
      }

      if (needsRedraw) {
        drawFpsGauge(fpsGaugeCanvas, gaugeState.fps.current, latestRow ? latestRow.frame_count : null);
        const bs = latestRow ? (Number(latestRow.batch_size) || 1) : 1;
        drawStepGauge(stepGaugeCanvas, gaugeState.step.current * bs, latestRow ? latestRow.training_steps * bs : null);
      }

      requestAnimationFrame(gaugeAnimationLoop);
    }
    // Draw gauges once at zero so they appear before any data arrives
    drawFpsGauge(fpsGaugeCanvas, 0, null);
    drawStepGauge(stepGaugeCanvas, 0, null);
    requestAnimationFrame(gaugeAnimationLoop);

    function setPreviewMessage(text) {
      if (!cards.previewMsg) return;
      if (text) {
        cards.previewMsg.textContent = text;
        cards.previewMsg.style.display = "flex";
      } else {
        cards.previewMsg.style.display = "none";
      }
    }

    function setPreviewRenderMode(mode) {
      const useVideo = (mode === "video");
      if (gamePreviewVideo) gamePreviewVideo.style.display = useVideo ? "block" : "none";
      if (gamePreviewCanvas) gamePreviewCanvas.style.display = useVideo ? "none" : "block";
    }

    function _videoElementHasTrack() {
      if (!gamePreviewVideo || !gamePreviewVideo.srcObject) return false;
      try {
        const tracks = gamePreviewVideo.srcObject.getVideoTracks();
        return Array.isArray(tracks) ? tracks.length > 0 : !!(tracks && tracks.length > 0);
      } catch (_) {
        return false;
      }
    }

    function _markPreviewVideoFrame() {
      _previewVideoHasFrame = true;
      _previewVideoLastFrameTs = (window.performance && typeof window.performance.now === "function")
        ? window.performance.now()
        : Date.now();
      setPreviewRenderMode("video");
      setPreviewMessage("");
    }

    function _previewVideoIsFresh(maxAgeMs = 1800) {
      if (!_previewRtcEnabled || !_previewRtcConnected || !_previewVideoHasFrame) return false;
      const now = (window.performance && typeof window.performance.now === "function")
        ? window.performance.now()
        : Date.now();
      return (now - _previewVideoLastFrameTs) <= Math.max(250, Number(maxAgeMs) || 1800);
    }

    function _ensurePreviewVideoAudio() {
      if (!gamePreviewVideo) return;
      const enableAudio = !!(PREVIEW_GAME_AUDIO_TRANSPORT_ENABLED && _previewGameAudioEnabled);
      gamePreviewVideo.muted = !enableAudio;
      gamePreviewVideo.volume = enableAudio ? 1.0 : 0.0;
      try {
        const audioTracks = gamePreviewVideo.srcObject ? gamePreviewVideo.srcObject.getAudioTracks() : [];
        if (audioTracks && audioTracks.length) {
          for (let i = 0; i < audioTracks.length; i++) {
            audioTracks[i].enabled = enableAudio;
          }
        }
      } catch (_) {}
      try {
        const p = gamePreviewVideo.play();
        if (p && typeof p.catch === "function") p.catch(() => {});
      } catch (_) {}
    }

    function clearPreviewCanvas() {
      if (!gamePreviewCanvas) return;
      setPreviewRenderMode("canvas");
      const width = gamePreviewCanvas.clientWidth || 320;
      const height = gamePreviewCanvas.clientHeight || 180;
      const dpr = window.devicePixelRatio || 1;
      gamePreviewCanvas.width = Math.floor(width * dpr);
      gamePreviewCanvas.height = Math.floor(height * dpr);
      const ctx = gamePreviewCanvas.getContext("2d");
      if (!ctx) return;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.fillStyle = "#020617";
      ctx.fillRect(0, 0, width, height);
      ctx.strokeStyle = "rgba(100, 160, 255, 0.20)";
      ctx.lineWidth = 1;
      ctx.strokeRect(0.5, 0.5, Math.max(0, width - 1), Math.max(0, height - 1));
    }

    function _normalizeClientRows(rows) {
      if (!Array.isArray(rows)) return [];
      return rows
        .map((row) => {
          const clientId = Number(row && row.client_id);
          if (!Number.isFinite(clientId) || clientId < 0) return null;
          const level = Math.max(0, Math.trunc(Number(row && row.level) || 0));
          const score = Math.max(0, Math.trunc(Number(row && row.score) || 0));
          const efficiency = level > 0 ? (score / level) : 0;
          return {
            client_id: Math.trunc(clientId),
            duration_seconds: Math.max(0, Number(row && row.duration_seconds) || 0),
            lives: Math.max(0, Math.trunc(Number(row && row.lives) || 0)),
            level,
            score,
            efficiency,
            selected_preview: !!(row && row.selected_preview),
            preview_capable: !!(row && row.preview_capable),
          };
        })
        .filter((row) => row !== null)
        .sort((a, b) => a.client_id - b.client_id);
    }

    function _effectivePreviewClientId(rows, selectedId) {
      const requestedId = (_previewPendingClientId !== null && Number.isFinite(_previewPendingClientId))
        ? Math.trunc(_previewPendingClientId)
        : null;
      const snapshotId = Number.isFinite(selectedId) ? Math.trunc(selectedId) : -1;
      if (requestedId !== null && rows.some((row) => row.preview_capable && row.client_id === requestedId)) {
        return requestedId;
      }
      if (rows.some((row) => row.preview_capable && row.client_id === snapshotId)) {
        return snapshotId;
      }
      const flagged = rows.find((row) => row.preview_capable && row.selected_preview);
      if (flagged) return flagged.client_id;
      const firstCapable = rows.find((row) => row.preview_capable);
      return firstCapable ? firstCapable.client_id : -1;
    }

    function _sortClientRows(rows) {
      const key = String(_clientTableSortKey || "client_id");
      const dir = (_clientTableSortDir === "desc") ? -1 : 1;
      return rows.slice().sort((a, b) => {
        const av = Number(a && a[key]) || 0;
        const bv = Number(b && b[key]) || 0;
        if (av !== bv) return (av - bv) * dir;
        return a.client_id - b.client_id;
      });
    }

    function updateClientTableSortIndicators() {
      if (!clientTableHead) return;
      const buttons = clientTableHead.querySelectorAll("button[data-sort-key]");
      for (const button of buttons) {
        const sortKey = String(button.dataset.sortKey || "");
        const active = sortKey === _clientTableSortKey;
        button.classList.toggle("active", active);
        const indicator = button.querySelector(".client-table-sort-indicator");
        if (indicator) {
          indicator.textContent = active ? (_clientTableSortDir === "desc" ? "▼" : "▲") : "";
        }
        const th = button.closest("th");
        if (th) {
          th.setAttribute("aria-sort", active ? (_clientTableSortDir === "desc" ? "descending" : "ascending") : "none");
        }
      }
    }

    function renderClientTable(rows, selectedId) {
      const normalizedRows = _normalizeClientRows(rows);
      const sortedRows = _sortClientRows(normalizedRows);
      const effectiveSelectedId = _effectivePreviewClientId(normalizedRows, selectedId);
      updateClientTableSortIndicators();
      if (cards.clientTableCount) cards.clientTableCount.textContent = fmtInt(normalizedRows.length);
      if (!clientTableBody) return;
        if (!sortedRows.length) {
        const existing = Array.from(clientTableBody.children);
        let emptyRow = existing.find((child) =>
          child instanceof HTMLTableRowElement && (
            String(child.dataset.empty || "") === "1" ||
            !!child.querySelector(".client-table-empty")
          )
        );
        if (!(emptyRow instanceof HTMLTableRowElement)) {
          emptyRow = document.createElement("tr");
          emptyRow.dataset.empty = "1";
          const td = document.createElement("td");
          td.colSpan = 6;
          td.className = "client-table-empty";
          td.textContent = "No Clients";
          emptyRow.appendChild(td);
        }
        for (const child of existing) {
          if (child !== emptyRow && child.parentNode === clientTableBody) {
            clientTableBody.removeChild(child);
          }
        }
        if (clientTableBody.firstChild !== emptyRow) {
          clientTableBody.insertBefore(emptyRow, clientTableBody.firstChild);
        }
        return;
      }
      const existingRows = new Map();
      for (const child of Array.from(clientTableBody.children)) {
        if (
          child instanceof HTMLTableRowElement && (
            String(child.dataset.empty || "") === "1" ||
            !!child.querySelector(".client-table-empty")
          )
        ) {
          if (child.parentNode === clientTableBody) clientTableBody.removeChild(child);
          continue;
        }
        const key = child instanceof HTMLTableRowElement ? String(child.dataset.clientId || "") : "";
        if (key) existingRows.set(key, child);
      }
      const orderedRows = [];
      for (const row of sortedRows) {
        const rowKey = String(row.client_id);
        let tr = existingRows.get(rowKey);
        if (!(tr instanceof HTMLTableRowElement)) {
          tr = document.createElement("tr");
          for (let i = 0; i < 6; i += 1) {
            tr.appendChild(document.createElement("td"));
          }
        }
        tr.dataset.clientId = String(row.client_id);
        tr.dataset.previewCapable = row.preview_capable ? "1" : "0";
        tr.classList.toggle("selected", row.client_id === effectiveSelectedId);
        tr.classList.toggle("preview-capable", !!row.preview_capable);
        tr.classList.toggle("inactive", !row.preview_capable);
        const cellDefs = [
          { value: fmtInt(row.client_id), className: "" },
          { value: fmtGameDuration(row.duration_seconds), className: "num" },
          { value: fmtInt(row.efficiency), className: "num" },
          { value: fmtInt(row.lives), className: "num" },
          { value: fmtInt(row.level), className: "num" },
          { value: fmtInt(row.score), className: "num" },
        ];
        const cells = tr.children;
        for (let i = 0; i < cellDefs.length; i += 1) {
          let td = cells[i];
          if (!(td instanceof HTMLTableCellElement)) {
            td = document.createElement("td");
            tr.appendChild(td);
          }
          td.textContent = cellDefs[i].value;
          td.className = cellDefs[i].className;
        }
        existingRows.delete(rowKey);
        orderedRows.push(tr);
      }
      for (const staleRow of existingRows.values()) {
        if (staleRow.parentNode === clientTableBody) {
          clientTableBody.removeChild(staleRow);
        }
      }
      for (let i = 0; i < orderedRows.length; i += 1) {
        const tr = orderedRows[i];
        const currentAtIndex = clientTableBody.children[i] || null;
        if (currentAtIndex !== tr) {
          clientTableBody.insertBefore(tr, currentAtIndex);
        }
      }
    }

    function syncPreviewClientSelection(rows, selectedId) {
      const normalizedRows = _normalizeClientRows(rows);
      const snapshotId = Number.isFinite(selectedId) ? Math.trunc(selectedId) : -1;
      if (_previewPendingClientId !== null && snapshotId === Math.trunc(_previewPendingClientId)) {
        _previewPendingClientId = null;
      }
      const firstCapable = normalizedRows.find((row) => row.preview_capable);
      if (
        snapshotId < 0 &&
        _previewPendingClientId === null &&
        !_previewClientRequestInFlight &&
        firstCapable
      ) {
        _requestPreviewClientSelection(firstCapable.client_id, "Loading Preview");
      }
    }

    async function _postPreviewClientSelection(clientId) {
      const targetId = Number(clientId);
      if (!Number.isFinite(targetId) || targetId < 0) return;
      _previewClientRequestInFlight = true;
      _previewPendingClientId = Math.trunc(targetId);
      try {
        const res = await fetch("/api/preview_settings", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ client_id: Math.trunc(targetId) }),
        });
        if (!res.ok) throw new Error("bad preview client response");
        const payload = await res.json();
        if (payload && Number.isFinite(Number(payload.client_id))) {
          _previewPendingClientId = Math.trunc(Number(payload.client_id));
        }
      } catch (e) {
        _previewPendingClientId = null;
        console.error("Failed to update preview client:", e);
      } finally {
        _previewClientRequestInFlight = false;
      }
    }

    function _requestPreviewClientSelection(clientId, messageText = "Switching Preview") {
      const targetId = Number(clientId);
      if (!Number.isFinite(targetId) || targetId < 0) return;
      _previewSeqLoaded = -1;
      _previewHasFrame = false;
      _previewVideoHasFrame = false;
      _previewVideoLastFrameTs = 0;
      clearPreviewCanvas();
      setPreviewMessage(messageText);
      _postPreviewClientSelection(targetId);
    }

    function drawPreviewToCard() {
      if (!gamePreviewCanvas || !_previewHasFrame) return;
      const sw = _previewSrcCanvas.width || 0;
      const sh = _previewSrcCanvas.height || 0;
      if (sw <= 0 || sh <= 0) return;
      const width = gamePreviewCanvas.clientWidth || 320;
      const height = gamePreviewCanvas.clientHeight || 180;
      const dpr = window.devicePixelRatio || 1;
      gamePreviewCanvas.width = Math.floor(width * dpr);
      gamePreviewCanvas.height = Math.floor(height * dpr);
      const ctx = gamePreviewCanvas.getContext("2d");
      if (!ctx) return;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.fillStyle = "#020617";
      ctx.fillRect(0, 0, width, height);
      const scale = Math.min(width / sw, height / sh);
      const dw = Math.max(1, Math.floor(sw * scale));
      const dh = Math.max(1, Math.floor(sh * scale));
      const dx = Math.floor((width - dw) * 0.5);
      const dy = Math.floor((height - dh) * 0.5);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(_previewSrcCanvas, 0, 0, sw, sh, dx, dy, dw, dh);
    }

    function _base64ToBytes(b64) {
      const bin = atob(String(b64 || ""));
      const len = bin.length;
      const out = new Uint8Array(len);
      for (let i = 0; i < len; i++) out[i] = bin.charCodeAt(i);
      return out;
    }

    function _decodeRgb565ToSource(width, height, b64data) {
      if (!_previewSrcCtx) return false;
      const w = Math.max(0, Number(width) || 0);
      const h = Math.max(0, Number(height) || 0);
      if (w > 512 || h > 512) return false;
      if (w <= 0 || h <= 0 || !b64data) return false;
      const bytes = _base64ToBytes(b64data);
      const pxCount = w * h;
      if (bytes.length !== pxCount * 2) return false;
      _previewSrcCanvas.width = w;
      _previewSrcCanvas.height = h;
      const img = _previewSrcCtx.createImageData(w, h);
      const dst = img.data;
      let si = 0;
      for (let i = 0, di = 0; i < pxCount; i++, di += 4) {
        const v = (bytes[si] << 8) | bytes[si + 1];
        si += 2;
        dst[di] = ((v >> 11) & 0x1f) * 255 / 31;
        dst[di + 1] = ((v >> 5) & 0x3f) * 255 / 63;
        dst[di + 2] = (v & 0x1f) * 255 / 31;
        dst[di + 3] = 255;
      }
      _previewSrcCtx.putImageData(img, 0, 0);
      _previewHasFrame = true;
      return true;
    }

    async function fetchGamePreview() {
      if (_previewFetchInFlight) return;
      _previewFetchInFlight = true;
      try {
        const profile = encodeURIComponent(_currentPreviewProfile());
        const res = await fetch(`/api/game_preview?cid=${encodeURIComponent(CLIENT_ID)}&profile=${profile}&t=${Date.now()}`, { cache: "no-store" });
        if (!res.ok) throw new Error("bad preview response");
        const payload = await res.json();
        const seq = Number(payload && payload.seq);
        if (Number.isFinite(seq) && seq > 0) _previewSeqLoaded = seq;
        const fmt = String(payload && payload.format || "");
        const ok = (fmt === "rgb565be") && _decodeRgb565ToSource(payload.width, payload.height, payload.data);
        if (ok) {
          drawPreviewToCard();
          setPreviewMessage("");
        }
      } catch (_) {
        /* ignore transient preview failures */
      } finally {
        _previewFetchInFlight = false;
      }
    }

    const WEBRTC_ICE_SERVERS = __WEBRTC_ICE_SERVERS_JSON__;
    const PREVIEW_PROFILE_STORAGE_KEY = "robotron.preview_quality";
    const PREVIEW_PROFILE_AUTO = /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent || "") ? "mobile" : "default";
    function _loadStoredPreviewQuality() {
      try {
        const raw = String(window.localStorage.getItem(PREVIEW_PROFILE_STORAGE_KEY) || "").trim().toLowerCase();
        if (raw === "auto" || raw === "default" || raw === "mobile") return raw;
      } catch (_) {}
      return _gsAdmin ? "default" : "mobile";
    }
    function _saveStoredPreviewQuality(value) {
      try {
        window.localStorage.setItem(PREVIEW_PROFILE_STORAGE_KEY, String(value || "auto"));
      } catch (_) {}
    }
    let _previewProfileMode = _loadStoredPreviewQuality();
    function _currentPreviewProfile() {
      return _previewProfileMode === "auto" ? PREVIEW_PROFILE_AUTO : _previewProfileMode;
    }
    function _previewLog(...args) {
      return;
    }
    function _previewWarn(...args) {
      return;
    }

    function _parseFmtpParams(fmtpLine) {
      const out = {};
      const src = String(fmtpLine || "");
      if (!src) return out;
      const parts = src.split(";");
      for (let i = 0; i < parts.length; i++) {
        const kv = String(parts[i] || "").trim();
        if (!kv) continue;
        const eq = kv.indexOf("=");
        if (eq <= 0) {
          out[kv.toLowerCase()] = "";
          continue;
        }
        const k = kv.slice(0, eq).trim().toLowerCase();
        const v = kv.slice(eq + 1).trim();
        out[k] = v;
      }
      return out;
    }

    function _fmtpHasBaselinePm1(codec) {
      const p = _parseFmtpParams(codec && codec.sdpFmtpLine ? codec.sdpFmtpLine : "");
      const pm = String(p["packetization-mode"] || "");
      const prof = String(p["profile-level-id"] || "").toLowerCase();
      return (pm === "1") && (prof === "42e01f" || prof.startsWith("42e0"));
    }

    function _preferH264BaselineCodecs(codecs) {
      const arr = Array.isArray(codecs) ? codecs.slice() : [];
      const rank = (c) => {
        const m = String(c && c.mimeType ? c.mimeType : "").toLowerCase();
        if (!m.includes("h264")) {
          if (m.includes("vp8")) return 20;
          if (m.includes("vp9")) return 30;
          return 40;
        }
        if (_fmtpHasBaselinePm1(c)) return 0;
        const p = _parseFmtpParams(c && c.sdpFmtpLine ? c.sdpFmtpLine : "");
        if (String(p["packetization-mode"] || "") === "1") return 1;
        return 2;
      };
      arr.sort((a, b) => rank(a) - rank(b));
      return arr;
    }

    function _mungeOfferSdpForH264BaselinePm1(sdp) {
      const text = String(sdp || "");
      if (!text) return text;
      const lines = text.split(/\\r\\n/);
      let mVideoIdx = -1;
      for (let i = 0; i < lines.length; i++) {
        if (String(lines[i] || "").startsWith("m=video ")) {
          mVideoIdx = i;
          break;
        }
      }
      if (mVideoIdx < 0) return text;

      const h264Pts = new Set();
      const fmtpIdxByPt = {};
      const fmtpByPt = {};

      for (let i = 0; i < lines.length; i++) {
        const ln = String(lines[i] || "");
        let m = ln.match(/^a=rtpmap:(\\d+)\\s+H264\\/\\d+/i);
        if (m) {
          h264Pts.add(m[1]);
          continue;
        }
        m = ln.match(/^a=fmtp:(\\d+)\\s+(.+)$/i);
        if (m) {
          const pt = m[1];
          fmtpIdxByPt[pt] = i;
          fmtpByPt[pt] = m[2];
        }
      }

      const h264PtList = Array.from(h264Pts);
      if (!h264PtList.length) return text;

      const scorePt = (pt) => {
        const p = _parseFmtpParams(fmtpByPt[pt] || "");
        const pm1 = String(p["packetization-mode"] || "") === "1";
        const prof = String(p["profile-level-id"] || "").toLowerCase();
        if (pm1 && prof === "42e01f") return 0;
        if (pm1 && prof.startsWith("42e0")) return 1;
        if (pm1) return 2;
        return 3;
      };
      h264PtList.sort((a, b) => scorePt(a) - scorePt(b));
      const chosenPt = h264PtList[0];

      const p = _parseFmtpParams(fmtpByPt[chosenPt] || "");
      p["packetization-mode"] = "1";
      p["profile-level-id"] = "42e01f";
      if (!p["level-asymmetry-allowed"]) p["level-asymmetry-allowed"] = "1";
      const keys = Object.keys(p).sort();
      const fmtpNew = keys.map((k) => `${k}=${p[k]}`).join(";");
      if (Object.prototype.hasOwnProperty.call(fmtpIdxByPt, chosenPt)) {
        lines[fmtpIdxByPt[chosenPt]] = `a=fmtp:${chosenPt} ${fmtpNew}`;
      } else {
        lines.push(`a=fmtp:${chosenPt} ${fmtpNew}`);
      }

      const mParts = String(lines[mVideoIdx] || "").trim().split(/\\s+/);
      if (mParts.length > 3) {
        const hdr = mParts.slice(0, 3);
        const pts = mParts.slice(3);
        const reordered = [chosenPt];
        for (let i = 0; i < pts.length; i++) {
          if (pts[i] !== chosenPt) reordered.push(pts[i]);
        }
        lines[mVideoIdx] = hdr.concat(reordered).join(" ");
      }

      return lines.join("\\r\\n");
    }
    if (gamePreviewVideo) {
      const _onVideoFrame = () => { _markPreviewVideoFrame(); };
      gamePreviewVideo.addEventListener("loadeddata", () => {
        _previewLog("video loadeddata", { w: gamePreviewVideo.videoWidth || 0, h: gamePreviewVideo.videoHeight || 0 });
      });
      gamePreviewVideo.addEventListener("playing", () => {
        _previewLog("video playing");
        _onVideoFrame();
      });
      gamePreviewVideo.addEventListener("stalled", () => {
        _previewWarn("video stalled");
      });
      gamePreviewVideo.addEventListener("waiting", () => {
        _previewWarn("video waiting");
      });
      if (typeof gamePreviewVideo.requestVideoFrameCallback === "function") {
        const _pumpVideoFrame = () => {
          gamePreviewVideo.requestVideoFrameCallback(() => {
            _onVideoFrame();
            _pumpVideoFrame();
          });
        };
        _pumpVideoFrame();
      }
    }

    function _schedulePreviewWebRTCRetry(delayMs = 1500) {
      if (_previewRtcRetryTimer) return;
      _previewRtcRetryTimer = setTimeout(() => {
        _previewRtcRetryTimer = null;
        _ensurePreviewWebRTC();
      }, Math.max(250, Number(delayMs) || 1500));
    }

    function _cancelPreviewWebRTCRetry() {
      if (_previewRtcRetryTimer) {
        clearTimeout(_previewRtcRetryTimer);
        _previewRtcRetryTimer = null;
      }
    }

    function _resetPreviewRtcState() {
      _previewRtcConnected = false;
      _previewRtcEnabled = false;
      _previewRtcConnecting = false;
      _previewRtcStream = null;
      _previewVideoHasFrame = false;
      _previewVideoLastFrameTs = 0;
      if (gamePreviewVideo) {
        try { gamePreviewVideo.pause(); } catch (_) {}
        gamePreviewVideo.srcObject = null;
      }
    }

    async function _restartPreviewWebRTC() {
      _cancelPreviewWebRTCRetry();
      const pc = _previewRtcPc;
      _previewRtcPc = null;
      _resetPreviewRtcState();
      if (pc) {
        try { await pc.close(); } catch (_) {}
      }
      if (ENABLE_CLIENT0_PREVIEW) {
        setPreviewRenderMode("canvas");
        if (_previewHasFrame) drawPreviewToCard();
        else clearPreviewCanvas();
        await _ensurePreviewWebRTC();
      }
    }

    function _waitForIceGatheringComplete(pc, timeoutMs = 2500) {
      return new Promise((resolve) => {
        if (!pc || pc.iceGatheringState === "complete") {
          resolve();
          return;
        }
        let done = false;
        const finish = () => {
          if (done) return;
          done = true;
          try { pc.removeEventListener("icegatheringstatechange", onState); } catch (_) {}
          clearTimeout(tid);
          resolve();
        };
        const onState = () => {
          if (pc.iceGatheringState === "complete") finish();
        };
        const tid = setTimeout(finish, Math.max(300, Number(timeoutMs) || 1200));
        try { pc.addEventListener("icegatheringstatechange", onState); } catch (_) { finish(); }
      });
    }

    async function _startPreviewWebRTC() {
      if (!window.RTCPeerConnection || !gamePreviewVideo) return false;
      if (_previewRtcConnecting) return false;
      if (_previewRtcPc) return _previewRtcConnected;
      _previewRtcConnecting = true;
      const previewProfile = _currentPreviewProfile();
      _previewLog("starting WebRTC offer", { iceServers: WEBRTC_ICE_SERVERS, profile: previewProfile });
      const pc = new RTCPeerConnection({
        iceServers: Array.isArray(WEBRTC_ICE_SERVERS) && WEBRTC_ICE_SERVERS.length
          ? WEBRTC_ICE_SERVERS
          : [{ urls: ["stun:stun.l.google.com:19302"] }],
      });
      _previewRtcPc = pc;
      _previewRtcConnected = false;
      _previewRtcStream = null;

      pc.ontrack = (ev) => {
        _previewLog("ontrack received", { trackKind: ev && ev.track ? ev.track.kind : "unknown" });
        if (!_previewRtcStream) {
          _previewRtcStream = new MediaStream();
        }
        const incomingStream = ev.streams && ev.streams[0] ? ev.streams[0] : null;
        if (incomingStream) {
          const incomingTracks = incomingStream.getTracks();
          for (let i = 0; i < incomingTracks.length; i++) {
            const t = incomingTracks[i];
            if (!_previewRtcStream.getTracks().some((x) => x && t && x.id === t.id)) {
              _previewRtcStream.addTrack(t);
            }
          }
        } else if (ev.track) {
          if (!_previewRtcStream.getTracks().some((x) => x && ev.track && x.id === ev.track.id)) {
            _previewRtcStream.addTrack(ev.track);
          }
        }
        gamePreviewVideo.srcObject = _previewRtcStream;
        // Show video element immediately; some mobile browsers will not
        // reliably start decode/render while the element is display:none.
        setPreviewRenderMode("video");
        _ensurePreviewVideoAudio();
        const hasVideo = !!(_previewRtcStream && _previewRtcStream.getVideoTracks().length > 0);
        _previewRtcConnected = hasVideo;
        _previewRtcEnabled = true;
        _previewRtcError = "";
        setPreviewMessage(hasVideo ? "Loading Preview" : "Waiting For Video Track");
      };

      pc.onconnectionstatechange = () => {
        const st = pc.connectionState || "";
        _previewLog("connectionState", st);
        if (st === "failed" || st === "closed") {
          _previewRtcConnected = false;
          _previewRtcEnabled = false;
          _previewRtcStream = null;
          setPreviewRenderMode("canvas");
          try { pc.close(); } catch (_) {}
          if (_previewRtcPc === pc) _previewRtcPc = null;
          _schedulePreviewWebRTCRetry(1000);
        } else if (st === "disconnected") {
          _previewRtcConnected = false;
          _previewRtcEnabled = false;
          _previewRtcStream = null;
          if (!_videoElementHasTrack()) {
            setPreviewRenderMode("canvas");
          }
          _schedulePreviewWebRTCRetry(1000);
        }
      };

      pc.oniceconnectionstatechange = () => {
        _previewLog("iceConnectionState", pc.iceConnectionState || "");
      };
      pc.onicegatheringstatechange = () => {
        _previewLog("iceGatheringState", pc.iceGatheringState || "");
      };

      try {
        const vtx = pc.addTransceiver("video", { direction: "recvonly" });
        if (PREVIEW_GAME_AUDIO_TRANSPORT_ENABLED) {
          pc.addTransceiver("audio", { direction: "recvonly" });
        }
        try {
          if (vtx && typeof vtx.setCodecPreferences === "function" && window.RTCRtpReceiver && typeof RTCRtpReceiver.getCapabilities === "function") {
            const caps = RTCRtpReceiver.getCapabilities("video");
            const codecs = (caps && Array.isArray(caps.codecs)) ? caps.codecs.filter((c) => c && c.mimeType && !/rtx|red|ulpfec/i.test(c.mimeType)) : [];
            if (codecs.length) {
              const preferred = _preferH264BaselineCodecs(codecs);
              vtx.setCodecPreferences(preferred);
              _previewLog("codec prefs set", preferred.map((c) => `${c.mimeType}${c.sdpFmtpLine ? `;${c.sdpFmtpLine}` : ""}`));
            }
          }
        } catch (e) {
          _previewWarn("setCodecPreferences failed", (e && e.message) ? e.message : e);
        }
        const offer = await pc.createOffer();
        const mungedOfferSdp = _mungeOfferSdpForH264BaselinePm1(offer && offer.sdp ? offer.sdp : "");
        await pc.setLocalDescription({
          type: (offer && offer.type) ? offer.type : "offer",
          sdp: mungedOfferSdp || ((offer && offer.sdp) ? offer.sdp : ""),
        });
        await _waitForIceGatheringComplete(pc, 2500);
        const res = await fetch("/api/game_preview_offer", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            sdp: (pc.localDescription && pc.localDescription.sdp) ? pc.localDescription.sdp : offer.sdp,
            type: (pc.localDescription && pc.localDescription.type) ? pc.localDescription.type : offer.type,
            profile: previewProfile,
          }),
        });
        _previewLog("offer POST status", res.status);
        let ans = null;
        try { ans = await res.json(); } catch (_) { ans = null; }
        if (!res.ok) throw new Error(ans && ans.error ? ans.error : "offer rejected");
        if (!ans || !ans.ok || !ans.sdp || !ans.type) throw new Error(ans && ans.error ? ans.error : "invalid answer");
        _previewLog("answer accepted", { type: ans.type, sdpLen: ans.sdp ? ans.sdp.length : 0 });
        await pc.setRemoteDescription({ type: ans.type, sdp: ans.sdp });
        _previewRtcEnabled = true;
        _previewRtcError = "";
        return true;
      } catch (err) {
        _previewRtcError = (err && err.message) ? String(err.message) : "webrtc init failed";
        _previewWarn("start failed", _previewRtcError);
        _previewRtcConnected = false;
        _previewRtcEnabled = false;
        setPreviewRenderMode("canvas");
        try { pc.close(); } catch (_) {}
        if (_previewRtcPc === pc) _previewRtcPc = null;
        _schedulePreviewWebRTCRetry(1500);
        return false;
      } finally {
        _previewRtcConnecting = false;
      }
    }

    async function _ensurePreviewWebRTC() {
      if (_previewRtcConnected || _previewRtcConnecting) return;
      _previewLog("ensure WebRTC requested");
      await _startPreviewWebRTC();
    }

    async function previewPumpLoop() {
      if (_previewPumpRunning) return;
      _previewPumpRunning = true;
      while (true) {
        if (_previewVideoIsFresh()) {
          await new Promise((resolve) => setTimeout(resolve, 200));
          continue;
        }
        try {
          const since = Math.max(0, Number(_previewSeqLoaded) || 0);
          const profile = encodeURIComponent(_currentPreviewProfile());
          const res = await fetch(
            `/api/game_preview?cid=${encodeURIComponent(CLIENT_ID)}&since=${since}&wait=1&timeout=4.0&profile=${profile}&t=${Date.now()}`,
            { cache: "no-store" }
          );
          if (!res.ok) throw new Error("bad preview stream response");
          const payload = await res.json();
          const seq = Number(payload && payload.seq);
          if (Number.isFinite(seq) && seq > _previewSeqLoaded) {
            _previewSeqLoaded = seq;
            const fmt = String(payload && payload.format || "");
            const ok = (fmt === "rgb565be") && _decodeRgb565ToSource(payload.width, payload.height, payload.data);
            if (ok) {
              if (!_previewVideoIsFresh()) setPreviewRenderMode("canvas");
              drawPreviewToCard();
              setPreviewMessage("");
            }
          }
        } catch (_) {
          await new Promise((resolve) => setTimeout(resolve, 250));
        }
      }
    }
    clearPreviewCanvas();

    const charts = {
      throughput: {
        canvas: document.getElementById("cThroughput"),
        series: [
          {
            key: "fps",
            color: "#22c55e",
            axis: { side: "left", min: 0, max: 3000, group_keys: ["fps", "eplen_100k"] },
            smooth_alpha: 0.14,
          },
          {
            key: "samples_per_sec_chart",
            color: "#f59e0b",
            axis: { side: "right", min: 0, max_floor: 25000, max_snap: 10000, group_keys: ["samples_per_sec_chart"] },
            smooth_alpha: 0.10,
          },
          {
            key: "eplen_100k",
            color: "#e879f9",
            axis_ref: "fps",
            smooth_alpha: 0.20,
          }
        ]
      },
      rewards: {
        canvas: document.getElementById("cRewards"),
        series: [
          { key: "total_5m", color: "#38bdf8", median_window: 3,
            axis: {
              side: "left",
              label_pad: 52,
              group_keys: ["total_100k", "total_1m", "total_5m"],
            },
          },
          { key: "total_1m", color: "#22c55e", median_window: 3, axis_ref: "total_5m" },
          { key: "total_100k", color: "#ef4444", median_window: 3, axis_ref: "total_5m" },
          { key: "reward_subj", color: "#facc15", median_window: 5, smooth_alpha: 0.15,
            axis: {
              side: "right",
              label_pad: 40,
              group_keys: ["reward_subj"],
            },
          }
        ]
      },
      learning: {
        canvas: document.getElementById("cLearning"),
        series: [
          {
            key: "loss",
            color: "#22c55e",
            axis: { side: "left", min: 0, group_keys: ["loss", "grad_norm"], max_floor: 1.5, tick_decimals: 1 },
            smooth_alpha: 0.55,
          },
          { key: "grad_norm", color: "#f59e0b", axis_ref: "loss", smooth_alpha: 0.55 },
          {
            key: "bc_loss",
            color: "#22d3ee",
            axis: { side: "right", min: 0, group_keys: ["bc_loss"], max_floor: 1.5, tick_decimals: 1 },
            smooth_alpha: 0.55,
          }
        ]
      },
      qRange: {
        canvas: document.getElementById("cQRange"),
        fill_between: ["q_max", "q_min", "#22c55e", "#38bdf8"],
        series: [
          { key: "q_max", color: "#22c55e", axis: { target_ticks: 4, group_keys: ["q_max", "q_min"] } },
          { key: "q_min", color: "#38bdf8", axis_ref: "q_max" }
        ]
      },
      level1m: {
        canvas: document.getElementById("cLevelMini"),
        series: [
          {
            key: "level_25k",
            color: "#22c55e",
            axis: { side: "left", min: 0, group_keys: ["level_25k", "level_1m", "level_5m"] },
          },
          { key: "level_1m", color: "#f59e0b", axis_ref: "level_25k" },
          { key: "level_5m", color: "#22d3ee", axis_ref: "level_25k" }
        ]
      },
      rewardMini: {
        canvas: document.getElementById("cRewardMini"),
        series: [
          { key: "total_5m", color: "#38bdf8", median_window: 3, axis: { target_ticks: 3 } },
          { key: "total_1m", color: "#22c55e", median_window: 3 },
          { key: "total_100k", color: "#ef4444", median_window: 3, linearTime: true }
        ]
      },
      lossMini: {
        canvas: document.getElementById("cLossMini"),
        series: [
          { key: "loss", color: "#22c55e", axis: { min: 0 } }
        ]
      },
      gradMini: {
        canvas: document.getElementById("cGradMini"),
        series: [
          { key: "grad_norm", color: "#f59e0b", axis: { min: 0 } }
        ]
      },
      epLenMini: {
        canvas: document.getElementById("cEpLenMini"),
        series: [
          { key: "eplen_1m", color: "#ff9f43", axis: { min: 0 } }
        ]
      },
      gpu0Mini: {
        canvas: document.getElementById("cGpu0Mini"),
        series: [
          { key: "gpu0_util", color: "#22d3ee", axis: { min: 0, max: 100 } },
          { key: "gpu0_mem_pct", color: "#38bdf8", axis_ref: "gpu0_util" }
        ]
      },
      gpu1Mini: {
        canvas: document.getElementById("cGpu1Mini"),
        series: [
          { key: "gpu1_util", color: "#f472b6", axis: { min: 0, max: 100 } },
          { key: "gpu1_mem_pct", color: "#fb7185", axis_ref: "gpu1_util" }
        ]
      },
      cpuMini: {
        canvas: document.getElementById("cCpuMini"),
        series: [
          { key: "cpu_pct", color: "#4ade80", axis: { min: 0, max: 100 } }
        ]
      },
      memFreeMini: {
        canvas: document.getElementById("cMemFreeMini"),
        series: [
          { key: "mem_free_gb", color: "#facc15", axis: { min: 0, min_range: 1.0 } }
        ]
      },
      agreement: {
        canvas: document.getElementById("cAgreement"),
        series: [
          { key: "agreement_1m", color: "#00c8ff", smooth_alpha: 0.35,
            axis: { side: "left", min: 0, min_range: 0.10, label_pad: 52, group_keys: ["agreement_1m", "agreement"], tick_decimals: 2 }
          },
          { key: "agreement", color: "#0090cc55", axis_ref: "agreement_1m", smooth_alpha: 0.10 },
          { key: "level_100k", color: "#ef4444", smooth_alpha: 0.20,
            axis: { side: "right", min_range: 1.0, label_pad: 40, group_keys: ["level_100k"], tick_decimals: 1 }
          },
          { key: "avg_game_score", color: "#ff9f43", smooth_alpha: 0.25,
            axis: { side: "right", min: 0, min_range: 500, label_pad: 52, group_keys: ["avg_game_score"], tick_decimals: 0 }
          }
        ]
      }
    };

    function fmtInt(v) {
      if (v === null || v === undefined || Number.isNaN(v)) return "0";
      return num.format(Math.round(v));
    }

    function fmtFloat(v, d = 2) {
      if (v === null || v === undefined || Number.isNaN(v)) return "0";
      return Number(v).toFixed(d);
    }

    function fmtSignedFloat(v, d = 2) {
      if (v === null || v === undefined || Number.isNaN(v)) return "+0";
      const n = Number(v);
      const mag = Math.abs(n).toFixed(d);
      // Pad integer part to 2 digits so display stays fixed-width
      const [ip, fp] = mag.split(".");
      const padded = ip.padStart(2, "0") + (fp !== undefined ? "." + fp : "");
      return (n < 0 ? "-" : "+") + padded;
    }

    function fmtPaddedFloat(v, intDigits = 2, decDigits = 2, padChar = "0") {
      if (v === null || v === undefined || Number.isNaN(v)) return "0";
      const n = Math.abs(Number(v));
      const [ip, fp] = n.toFixed(decDigits).split(".");
      return ip.padStart(intDigits, padChar) + (fp !== undefined ? "." + fp : "");
    }

    function qColor(v) {
      // Color by absolute magnitude relative to C51 support [-100,100]
      const m = Math.abs(v);
      if (m < 20) return "#39ff14";         // green – healthy
      if (m < 40) return "#ffaa33";         // amber – caution
      if (m < 60) return "#ff9900";         // dark amber – concerning
      if (m < 80) return "#ff8c14";         // orange – warning
      return "#ff2020";                      // red – exploding
    }

    function toColoredQRange(qMin, qMax) {
      const minStr = fmtSignedFloat(qMin, 1);
      const maxStr = fmtSignedFloat(qMax, 1);
      const minColor = qColor(qMin);
      const maxColor = qColor(qMax);
      function coloredCells(text, color) {
        return Array.from(String(text)).map((ch) => {
          const html = (ch === " ") ? "&nbsp;" : ch
            .replaceAll("&", "&amp;").replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;").replaceAll('"', "&quot;");
          return `<span class="qch" style="color:${color};text-shadow:0 0 6px ${color}44">${html}</span>`;
        }).join("");
      }
      return coloredCells(minStr, minColor) +
             `<span class="qch">,</span>` +
             coloredCells(maxStr, maxColor);
    }

    function toFixedCharCells(text) {
      const s = String(text ?? "");
      return Array.from(s).map((ch) => {
        const html = (ch === " ") ? "&nbsp;" : ch
          .replaceAll("&", "&amp;")
          .replaceAll("<", "&lt;")
          .replaceAll(">", "&gt;")
          .replaceAll('"', "&quot;")
          .replaceAll("'", "&#39;");
        return `<span class="qch">${html}</span>`;
      }).join("");
    }

    function downsampleHistory(rows, targetPoints) {
      if (!Array.isArray(rows)) return [];
      const n = rows.length;
      const limit = Math.max(2, Number(targetPoints) || 0);
      if (n <= limit) return rows;
      // End-anchored sampling keeps newest (right-side) points stable as new
      // samples arrive, reducing visible squirm near "now".
      const outRev = [];
      const step = (n - 1) / (limit - 1);
      for (let i = 0; i < limit; i++) {
        const fromEnd = Math.floor(i * step);
        outRev.push(rows[n - 1 - fromEnd]);
      }
      outRev[0] = rows[n - 1];
      outRev[limit - 1] = rows[0];
      return outRev.reverse();
    }

    function sliceHistoryLookback(rows, lookbackSec) {
      if (!Array.isArray(rows) || !rows.length) return [];
      const lb = Number(lookbackSec);
      if (!Number.isFinite(lb) || lb <= 0) return rows.slice();
      const newestTs = Number(rows[rows.length - 1] && rows[rows.length - 1].ts);
      if (!Number.isFinite(newestTs)) return rows.slice();
      const cutoff = newestTs - lb;
      let start = 0;
      while (start < rows.length) {
        const ts = Number(rows[start] && rows[start].ts);
        if (!Number.isFinite(ts) || ts >= cutoff) break;
        start += 1;
      }
      return rows.slice(start);
    }

    function fmtPct(v) {
      if (v === null || v === undefined || Number.isNaN(v)) return "0%";
      return `${(Number(v) * 100.0).toFixed(1)}%`;
    }

    function fmtGameDuration(v) {
      const secs = Math.max(0, Math.round(Number(v) || 0));
      const hrs = Math.floor(secs / 3600);
      const mins = Math.floor((secs % 3600) / 60);
      const rem = secs % 60;
      if (hrs > 0) return `${hrs}:${String(mins).padStart(2, "0")}:${String(rem).padStart(2, "0")}`;
      return `${mins}:${String(rem).padStart(2, "0")}`;
    }

    function setAudioToggle(enabled, hasTracks = true) {
      if (!audioToggle) return;
      audioToggle.textContent = hasTracks ? (enabled ? "Audio On" : "Audio Off") : "No Audio";
      audioToggle.classList.toggle("on", !!enabled && !!hasTracks);
      audioToggle.disabled = !hasTracks;
    }

    function clearAudioRetryTimer() {
      if (audioRetryTimer) {
        clearTimeout(audioRetryTimer);
        audioRetryTimer = null;
      }
    }

    function scheduleAudioRetry() {
      clearAudioRetryTimer();
      if (!audioEnabled || !audioPlaylist.length) return;
      audioRetryTimer = setTimeout(() => {
        audioRetryTimer = null;
        ensureAudioPlaying();
      }, AUDIO_START_RETRY_MS);
    }

    function getCookieValue(name) {
      const key = `${name}=`;
      const parts = String(document.cookie || "").split(";");
      for (const raw of parts) {
        const part = raw.trim();
        if (part.startsWith(key)) {
          return decodeURIComponent(part.slice(key.length));
        }
      }
      return null;
    }

    function setCookieValue(name, value) {
      // Keep preference for ~1 year and scope to dashboard path.
      document.cookie = `${name}=${encodeURIComponent(value)}; Max-Age=31536000; Path=/; SameSite=Lax`;
    }

    function stopAudio() {
      clearAudioRetryTimer();
      if (!bgAudio) return;
      bgAudio.pause();
      bgAudio.removeAttribute("src");
      try { bgAudio.load(); } catch (_) {}
    }

    function playAudioAt(index) {
      if (!bgAudio || !audioPlaylist.length) return;
      const n = audioPlaylist.length;
      audioIndex = ((index % n) + n) % n;
      bgAudio.src = audioPlaylist[audioIndex].url;
      try { bgAudio.load(); } catch (_) {}
      const p = bgAudio.play();
      if (p && typeof p.then === "function") {
        p.then(() => {
          clearAudioRetryTimer();
        }).catch(() => {
          scheduleAudioRetry();
        });
      }
    }

    function ensureAudioPlaying() {
      if (!audioEnabled || !audioPlaylist.length) return;
      if (!bgAudio || !bgAudio.src) {
        playAudioAt(audioIndex);
        return;
      }
      const p = bgAudio.play();
      if (p && typeof p.then === "function") {
        p.then(() => {
          clearAudioRetryTimer();
        }).catch(() => {
          scheduleAudioRetry();
        });
      }
    }

    function setAudioEnabled(next) {
      audioEnabled = !!next;
      setCookieValue(AUDIO_PREF_COOKIE, audioEnabled ? "1" : "0");
      setAudioToggle(audioEnabled, audioPlaylist.length > 0);
      if (audioEnabled) ensureAudioPlaying();
      else stopAudio();
    }

    async function loadAudioPlaylist() {
      try {
        const res = await fetch(`/api/audio_playlist?t=${Date.now()}`, { cache: "no-store" });
        if (!res.ok) throw new Error("playlist");
        const payload = await res.json();
        const tracks = Array.isArray(payload && payload.tracks) ? payload.tracks : [];
        audioPlaylist = tracks
          .map((t) => ({ name: String(t.name || ""), url: String(t.url || "") }))
          .filter((t) => t.url.length > 0);
      } catch (_) {
        audioPlaylist = [];
      }
      if (audioIndex >= audioPlaylist.length) audioIndex = 0;
      setAudioToggle(audioEnabled, audioPlaylist.length > 0);
      if (audioEnabled) ensureAudioPlaying();
    }

    /* ══════════════════════════════════════════════════════════════
     * STATUS & LED INDICATORS — Connection dot, inference/replay LEDs
     * Colors change based on metric thresholds (green/yellow/red).
     * ══════════════════════════════════════════════════════════════ */
    function setConnected(connected) {
      const dot = document.getElementById("statusDot");
      const text = document.getElementById("statusText");
      if (connected) {
        dot.style.background = "#39ff14";
        dot.style.boxShadow = "0 0 0 7px rgba(57,255,20,0.22), 0 0 16px rgba(57,255,20,0.55)";
        text.textContent = "Connected";
      } else {
        dot.style.background = "#ff2a55";
        dot.style.boxShadow = "0 0 0 7px rgba(255,42,85,0.24), 0 0 16px rgba(255,42,85,0.55)";
        text.textContent = "Disconnected";
      }
    }

    function setInfLed(avgInfMs) {
      if (!cards.infLed) return;
      const ms = Number(avgInfMs);
      if (!Number.isFinite(ms) || ms < 5.0) {
        cards.infLed.style.background = "#39ff14";
        cards.infLed.style.boxShadow = "0 0 0 4px rgba(57,255,20,0.22), 0 0 12px rgba(57,255,20,0.6)";
        return;
      }
      if (ms < 10.0) {
        cards.infLed.style.background = "#ffe600";
        cards.infLed.style.boxShadow = "0 0 0 4px rgba(255,230,0,0.22), 0 0 12px rgba(255,230,0,0.58)";
        return;
      }
      cards.infLed.style.background = "#ff2a55";
      cards.infLed.style.boxShadow = "0 0 0 4px rgba(255,42,85,0.24), 0 0 12px rgba(255,42,85,0.58)";
    }

    function setRplLed(rplPerFrame) {
      if (!cards.rplLed) return;
      const v = Number(rplPerFrame);
      if (!Number.isFinite(v)) {
        cards.rplLed.style.background = "#94a3b8";
        cards.rplLed.style.boxShadow = "0 0 0 4px rgba(148,163,184,0.18), 0 0 12px rgba(148,163,184,0.35)";
        return;
      }
      if (v > 8.0 || v < 0.25) {
        cards.rplLed.style.background = "#ff2a55";
        cards.rplLed.style.boxShadow = "0 0 0 4px rgba(255,42,85,0.24), 0 0 12px rgba(255,42,85,0.58)";
        return;
      }
      if (v >= 4.0) {
        cards.rplLed.style.background = "#f59e0b";
        cards.rplLed.style.boxShadow = "0 0 0 4px rgba(245,158,11,0.24), 0 0 12px rgba(245,158,11,0.58)";
        return;
      }
      if (v >= 1.0) {
        cards.rplLed.style.background = "#39ff14";
        cards.rplLed.style.boxShadow = "0 0 0 4px rgba(57,255,20,0.22), 0 0 12px rgba(57,255,20,0.6)";
        return;
      }
      cards.rplLed.style.background = "#ffe600";
      cards.rplLed.style.boxShadow = "0 0 0 4px rgba(255,230,0,0.22), 0 0 12px rgba(255,230,0,0.58)";
    }

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

    /* ── VFD bloom text helper ───────────────────────────────────────
     * Draws multi-layer glowing text simulating a vacuum fluorescent
     * display (VFD).  Each layer is progressively brighter and tighter
     * to create a realistic bloom/glow effect.
     *
     * @param {CanvasRenderingContext2D} ctx   Canvas context
     * @param {string}  text    The text to render
     * @param {number}  x, y   Position to draw at
     * @param {Array}   layers  Array of {fill, shadow, blur} objects,
     *                          ordered widest/faintest → tightest/brightest.
     *                          Each layer is drawn twice for extra intensity.
     * @param {object}  crisp   Final crisp layer: {fill, shadow, blur}
     */
    function drawBloomText(ctx, text, x, y, layers, crisp) {
      for (const l of layers) {
        ctx.fillStyle = l.fill;
        ctx.shadowColor = l.shadow;
        ctx.shadowBlur = l.blur;
        ctx.fillText(text, x, y);
        ctx.fillText(text, x, y);
      }
      ctx.fillStyle = crisp.fill;
      ctx.shadowColor = crisp.shadow;
      ctx.shadowBlur = crisp.blur;
      ctx.fillText(text, x, y);
      ctx.shadowBlur = 0;
    }

    /* VFD bloom color presets for gauge sub-text */
    const VFD_BLOOM_ORANGE = {
      layers: [
        { fill: "rgba(180, 80, 0, 0.50)",  shadow: "rgba(255, 100, 0, 0.60)",  blur: 48 },
        { fill: "rgba(220, 120, 10, 0.70)", shadow: "rgba(255, 130, 10, 0.70)", blur: 28 },
        { fill: "rgba(240, 160, 30, 0.80)", shadow: "rgba(255, 160, 20, 0.80)", blur: 14 },
        { fill: "rgba(255, 190, 60, 0.90)", shadow: "rgba(255, 180, 40, 0.90)", blur: 5 },
      ],
      crisp: { fill: "#ffaa33", shadow: "rgba(255, 140, 20, 0.60)", blur: 10 },
    };
    const VFD_BLOOM_BLUE = {
      layers: [
        { fill: "rgba(60, 120, 255, 0.50)",  shadow: "rgba(60, 130, 255, 0.80)",  blur: 52 },
        { fill: "rgba(80, 140, 255, 0.65)",  shadow: "rgba(80, 150, 255, 0.85)",  blur: 30 },
        { fill: "rgba(100, 160, 255, 0.75)", shadow: "rgba(100, 170, 255, 0.90)", blur: 16 },
        { fill: "rgba(140, 190, 255, 0.85)", shadow: "rgba(120, 180, 255, 0.95)", blur: 6 },
      ],
      crisp: { fill: "#c8e8ff", shadow: "rgba(80, 160, 255, 0.70)", blur: 12 },
    };

    /* ══════════════════════════════════════════════════════════════
     * GAUGE RENDERING — Analogue-style gauge with needle, color arc,
     * VFD sub-text, and LED odometer badge.
     * ══════════════════════════════════════════════════════════════ */
    function drawStyledGauge(canvas, valueRaw, cfg) {
      if (!canvas) return;

      const width  = canvas.clientWidth  || 360;
      const height = canvas.clientHeight || 210;
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);

      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);

      const minV = Number(cfg.min);
      const maxV = Number(cfg.max);
      const spanV = Math.max(1e-9, maxV - minV);
      const clampVal = (v) => Math.max(minV, Math.min(maxV, Number(v) || 0));
      const value = clampVal(valueRaw);
      const displayVal = Math.max(minV, Math.min(199999, Number(valueRaw) || 0));

      const pad = 2;
      const outerExtent = 1.08;   // tight fit — faint glow may clip, bezel stays
      const downExtent  = 1.00;   // badge hangs below center — room for LED box
      // Size the radius so the dial fits snugly inside the canvas
      const maxRByWidth  = (width  - 2 * pad) / (2.0 * outerExtent);
      const maxRByHeight = (height - 2 * pad) / (outerExtent + downExtent);
      const radius = Math.max(20, Math.min(maxRByWidth, maxRByHeight) * 1.08);
      // Center the full envelope (bezel-top to badge-bottom) in the canvas
      const cx = width  * 0.5;
      const cy = (height * 0.5) + ((outerExtent - downExtent) * radius * 0.5);

      const degToRad = (d) => (d * Math.PI) / 180.0;
      const startDeg = 135;
      const spanDeg = 270;
      const startRad = degToRad(startDeg);
      const endRad = degToRad(startDeg + spanDeg);
      const valToAngle = (v) => {
        const t = (clampVal(v) - minV) / spanV;
        return startRad + (t * (endRad - startRad));
      };

      // Dial face.
      const face = ctx.createRadialGradient(cx, cy - radius * 0.5, radius * 0.2, cx, cy, radius * 1.02);
      face.addColorStop(0.0, "rgba(36, 40, 46, 0.0)");
      face.addColorStop(0.55, "rgba(22, 25, 30, 0.0)");
      face.addColorStop(1.0, "rgba(12, 14, 18, 0.0)");
      ctx.fillStyle = face;
      ctx.beginPath();
      ctx.arc(cx, cy, radius * 1.02, 0, Math.PI * 2);
      ctx.fill();

      // Dial ring — colored by threshold zones (red → yellow → green).
      const arcLW = Math.max(3.5, radius * 0.040);
      const arcR  = radius * 0.92;
      // Draw a smooth continuous gradient along the arc
      const arcSegs = 120;
      ctx.lineWidth = arcLW;
      ctx.lineCap = "butt";
      for (let s = 0; s < arcSegs; s++) {
        const t0 = s / arcSegs;
        const t1 = (s + 1) / arcSegs;
        const tMid = (t0 + t1) * 0.5;
        const vMid = minV + tMid * spanV;
        // Interpolate color: red at red_max, yellow at yellow_max, green above
        const redEnd = (cfg.red_max - minV) / spanV;
        const yelEnd = (cfg.yellow_max - minV) / spanV;
        let r, g, b;
        if (tMid <= redEnd) {
          // Pure red zone
          r = 255; g = 58; b = 58;
        } else if (tMid <= yelEnd) {
          // Red → Yellow transition
          const f = (tMid - redEnd) / Math.max(1e-9, yelEnd - redEnd);
          r = 255; g = Math.round(58 + f * (176 - 58)); b = Math.round(58 - f * (58 - 44));
        } else {
          // Yellow → Green transition
          const f = (tMid - yelEnd) / Math.max(1e-9, 1.0 - yelEnd);
          r = Math.round(252 - f * (252 - 51)); g = Math.round(176 + f * (219 - 176)); b = Math.round(44 + f * (107 - 44));
        }
        const a0 = startRad + t0 * (endRad - startRad);
        const a1 = startRad + t1 * (endRad - startRad);
        ctx.strokeStyle = `rgba(${r},${g},${b},0.85)`;
        ctx.beginPath();
        ctx.arc(cx, cy, arcR, a0, a1 + 0.005, false);
        ctx.stroke();
      }

      // Scale ticks with threshold coloring.
      const minorStep = Math.max(1e-9, Number(cfg.minor_step));
      const majorStep = Math.max(minorStep, Number(cfg.major_step));
      const majorEvery = Math.max(1, Math.round(majorStep / minorStep));
      const tickOuter = radius * 0.89;
      const tickMinorInner = radius * 0.825;
      const tickMajorInner = radius * 0.775;
      const labelRadius = radius * 0.66 - (cfg.label_inset || 0);
      const tickCount = Math.max(1, Math.round((maxV - minV) / minorStep));
      for (let i = 0; i <= tickCount; i++) {
        const vv = (i >= tickCount) ? maxV : (minV + (i * minorStep));
        const a = valToAngle(vv);
        const cosA = Math.cos(a);
        const sinA = Math.sin(a);
        const isMajor = ((i % majorEvery) === 0) || i === tickCount;
        // Skip major tick at end if it doesn't align with a clean major step
        const isLastAndUnaligned = (i === tickCount) && (Math.abs(vv % majorStep) > 1e-6) && (Math.abs((vv % majorStep) - majorStep) > 1e-6);
        const drawAsMajor = isMajor && !isLastAndUnaligned;
        const inner = drawAsMajor ? tickMajorInner : tickMinorInner;
        let c = "rgba(236, 241, 247, 0.90)";
        if (drawAsMajor) {
          if (vv <= cfg.red_max) c = "rgba(255, 58, 58, 0.95)";
          else if (vv <= cfg.yellow_max) c = "rgba(252, 176, 44, 0.95)";
          else c = "rgba(51, 219, 107, 0.95)";
        }
        ctx.strokeStyle = c;
        ctx.lineWidth = drawAsMajor ? Math.max(3.0, radius * 0.03) : Math.max(0.9, radius * 0.008);
        ctx.beginPath();
        ctx.moveTo(cx + (tickOuter * cosA), cy + (tickOuter * sinA));
        ctx.lineTo(cx + (inner * cosA), cy + (inner * sinA));
        ctx.stroke();

        // Value labels at major ticks (infinity symbol on last tick)
        const labelEvery = cfg.label_every || 1;
        const majorIdx = Math.round(i / majorEvery);  // which major tick is this (0, 1, 2...)
        const showLabel = drawAsMajor && (i >= tickCount || (majorIdx % labelEvery) === 0);
        if (showLabel) {
          // Push bottom labels outward (closer to ticks) so they don't overlap the LED badge
          const bottomFactor = Math.max(0, sinA);  // 0 at top, 1 at bottom
          const inwardOffset = 8 - (bottomFactor * 16);  // 8px inward at top, -8px outward at bottom
          // Additional radial offset for non-endpoint labels (push toward tick marks)
          const radialExtra = (cfg.label_radial_offset && i > 0 && i < tickCount) ? cfg.label_radial_offset : 0;
          let lx = cx + ((labelRadius - inwardOffset - radialExtra) * cosA);
          let ly = cy + ((labelRadius - inwardOffset - radialExtra) * sinA);
          // For non-endpoint labels, apply lateral inset toward center (FPS uses label_lateral)
          const lateralPx = cfg.label_lateral || 0;
          if (lateralPx && i > 0 && i < tickCount && Math.abs(cosA) > 0.25) {
            // Shift label horizontally toward cx (skip near-top-center labels)
            lx += (lx < cx) ? lateralPx : -lateralPx;
          }
          ctx.fillStyle = "rgba(230, 235, 242, 0.82)";
          ctx.textBaseline = "middle";
          // Outward-justify: left-align on left side, right-align on right side
          const normA = ((a % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
          if (Math.abs(cosA) < 0.15) ctx.textAlign = "center";
          else if (normA > Math.PI * 0.5 && normA < Math.PI * 1.5) ctx.textAlign = "right";
          else ctx.textAlign = "left";
          if (i >= tickCount) {
            ctx.font = `${Math.max(12, Math.round(radius * 0.196))}px 'Avenir Next', 'Segoe UI', sans-serif`;
            ctx.fillText("\u221E", lx, ly);
          } else {
            const labelFontScale = cfg.label_font_scale || 0.11;
            ctx.font = `${Math.max(8, Math.round(radius * labelFontScale))}px 'Avenir Next', 'Segoe UI', sans-serif`;
            let labelStr;
            if (vv >= 1000) {
              labelStr = `${Math.round(vv / 1000)}K`;
            } else {
              labelStr = `${Math.round(vv)}`;
            }
            ctx.fillText(labelStr, lx, ly);
          }
        }
      }

      // Needle geometry for classic pointy orange pointer.
      const needleAngle = valToAngle(value);
      const nCos = Math.cos(needleAngle);
      const nSin = Math.sin(needleAngle);
      const needleLen = radius * 0.84;
      const tailLen = radius * 0.10;
      const baseHalfW = Math.max(4.0, radius * 0.030);
      const pTipX = cx + (needleLen * nCos);
      const pTipY = cy + (needleLen * nSin);
      const pTailX = cx - (tailLen * nCos);
      const pTailY = cy - (tailLen * nSin);
      const perpX = -nSin;
      const perpY = nCos;

      const drawNeedle = () => {
        // Drop shadow
        ctx.fillStyle = "rgba(0, 0, 0, 0.42)";
        ctx.beginPath();
        ctx.moveTo(pTipX + 2.5, pTipY + 2.5);
        ctx.lineTo(pTailX + (perpX * baseHalfW) + 2.5, pTailY + (perpY * baseHalfW) + 2.5);
        ctx.lineTo(pTailX - (perpX * baseHalfW) + 2.5, pTailY - (perpY * baseHalfW) + 2.5);
        ctx.closePath();
        ctx.fill();

        // Multi-layer yellow-orange bloom glow (like VFD but warm)
        const glowLayers = [
          { blur: 64, color: "rgba(255, 140, 10, 0.70)", fill: "rgba(255, 120, 10, 0.06)" },
          { blur: 38, color: "rgba(255, 160, 20, 0.80)", fill: "rgba(255, 130, 10, 0.10)" },
          { blur: 18, color: "rgba(255, 170, 40, 0.90)", fill: "rgba(255, 140, 20, 0.14)" },
          { blur:  7, color: "rgba(255, 190, 60, 1.00)", fill: "rgba(255, 160, 30, 0.18)" },
        ];
        const needlePath = () => {
          ctx.beginPath();
          ctx.moveTo(pTipX, pTipY);
          ctx.lineTo(pTailX + (perpX * baseHalfW), pTailY + (perpY * baseHalfW));
          ctx.lineTo(pTailX - (perpX * baseHalfW), pTailY - (perpY * baseHalfW));
          ctx.closePath();
        };
        ctx.save();
        for (const gl of glowLayers) {
          ctx.shadowColor = gl.color;
          ctx.shadowBlur = gl.blur;
          ctx.shadowOffsetX = 0;
          ctx.shadowOffsetY = 0;
          ctx.fillStyle = gl.fill;
          needlePath();
          ctx.fill();
        }
        ctx.restore();

        // Crisp needle fill
        const needleGrad = ctx.createLinearGradient(pTailX, pTailY, pTipX, pTipY);
        needleGrad.addColorStop(0.0, "#c85a00");
        needleGrad.addColorStop(0.6, "#ff8a00");
        needleGrad.addColorStop(1.0, "#ffc04d");
        ctx.fillStyle = needleGrad;
        needlePath();
        ctx.fill();
      };
      const drawHub = () => {
        const hubOuter = ctx.createRadialGradient(cx - 2, cy - 2, 2, cx, cy, radius * 0.16);
        hubOuter.addColorStop(0.0, "rgba(167, 174, 180, 0.98)");
        hubOuter.addColorStop(1.0, "rgba(38, 44, 51, 0.98)");
        ctx.fillStyle = hubOuter;
        ctx.beginPath();
        ctx.arc(cx, cy, radius * 0.16, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = "rgba(8, 12, 17, 0.96)";
        ctx.beginPath();
        ctx.arc(cx, cy, radius * 0.09, 0, Math.PI * 2);
        ctx.fill();
      };

      // Center title (matching reference style).
      ctx.fillStyle = "rgba(232, 236, 241, 0.85)";
      ctx.font = `700 ${Math.max(13, Math.round(radius * 0.16))}px 'Avenir Next', 'Segoe UI', sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(cfg.title || "", cx, cy - radius * 0.36);

      // Bottom value badge.
      const badgeW = radius * 1.0;
      const badgeH = radius * 0.48;
      const badgeX = cx - (badgeW * 0.5);
      const badgeY = cy + radius * 0.44;

      // Sub-text above the badge (e.g. total training steps) — VFD odometer display
      if (cfg.sub_text) {
        const subFont = `400 ${Math.max(10, Math.round(radius * 0.125))}px 'LED Dot-Matrix', 'Dot Matrix', 'DotGothic16', 'Courier New', monospace`;
        ctx.font = subFont;
        ctx.textAlign = "right";
        ctx.textBaseline = "middle";
        // Fixed width: measure max-width template so box never resizes
        const maxTemplate = "8,888,888,888";
        const tmMax = ctx.measureText(maxTemplate);
        const odoPad = radius * 0.05;
        const odoW = tmMax.width + odoPad * 2;
        const odoH = Math.max(14, Math.round(radius * 0.16));
        const odoX = cx - odoW * 0.5;
        const odoY = badgeY - odoH - 3;
        const odoR = Math.max(3, radius * 0.035);
        const subX = odoX + odoW - odoPad;  // right-justified with padding
        // VFD housing background — color keyed to gauge type
        ctx.save();
        ctx.shadowBlur = 0;
        const odoColor = cfg.sub_text_color || "amber";
        roundRectPath(ctx, odoX, odoY, odoW, odoH, odoR);
        ctx.fillStyle = "#130A2C";  // uniform dark VFD housing
        ctx.fill();
        // Subtle inset border matching text color
        ctx.strokeStyle = odoColor === "orange"
          ? "rgba(180, 100, 20, 0.35)"
          : "rgba(60, 130, 255, 0.35)";
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.restore();
        // Text centered in the VFD box
        const subTextY = odoY + odoH * 0.5;
        const bloom = (odoColor === "orange") ? VFD_BLOOM_ORANGE : VFD_BLOOM_BLUE;
        drawBloomText(ctx, cfg.sub_text, subX, subTextY, bloom.layers, bloom.crisp);
      }

      const badgeFill = ctx.createLinearGradient(0, badgeY, 0, badgeY + badgeH);
      badgeFill.addColorStop(0.0, "rgba(44, 10, 12, 0.95)");
      badgeFill.addColorStop(1.0, "rgba(30, 6, 8, 0.98)");
      roundRectPath(ctx, badgeX, badgeY, badgeW, badgeH, Math.max(8, radius * 0.08));
      ctx.fillStyle = badgeFill;
      ctx.fill();

      const valueText = Number(displayVal).toFixed(cfg.decimals ?? 1);
      const ledX = cx;
      const ledY = badgeY + (badgeH * 0.5);
      const ledScale = valueText.length > 5 ? 0.282 : 0.338;
      const ledFont = `400 ${Math.max(13, Math.round(radius * ledScale))}px 'DS-Digital', 'LED Dot-Matrix', 'Dot Matrix', 'DotGothic16', 'Courier New', monospace`;
      ctx.font = ledFont;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      // Multi-layer LED bloom (widest/faintest first)
      ctx.fillStyle = "rgba(255, 20, 20, 0.45)";
      ctx.shadowColor = "rgba(255, 30, 30, 0.80)";
      ctx.shadowBlur = Math.max(48, radius * 0.55);
      ctx.fillText(valueText, ledX, ledY);
      ctx.fillStyle = "rgba(255, 30, 30, 0.55)";
      ctx.shadowColor = "rgba(255, 40, 40, 0.85)";
      ctx.shadowBlur = Math.max(28, radius * 0.35);
      ctx.fillText(valueText, ledX, ledY);
      ctx.fillStyle = "rgba(255, 40, 40, 0.65)";
      ctx.shadowColor = "rgba(255, 50, 50, 0.90)";
      ctx.shadowBlur = Math.max(14, radius * 0.18);
      ctx.fillText(valueText, ledX, ledY);
      ctx.fillStyle = "rgba(255, 50, 50, 0.80)";
      ctx.shadowColor = "rgba(255, 60, 60, 0.95)";
      ctx.shadowBlur = Math.max(5, radius * 0.08);
      ctx.fillText(valueText, ledX, ledY);
      // Bright LED text on top
      ctx.fillStyle = "rgba(255, 52, 52, 0.98)";
      ctx.shadowBlur = 0;
      ctx.fillText(valueText, ledX, ledY);
      drawNeedle();
      drawHub();
    }

    function drawFpsGauge(canvas, fps, totalFrames) {
      const framesText = totalFrames != null ? Number(totalFrames).toLocaleString() : null;
      drawStyledGauge(canvas, fps, {
        min: GAUGE_MIN_FPS,
        max: GAUGE_MAX_FPS,
        red_max: GAUGE_FPS_RED_MAX,
        yellow_max: GAUGE_FPS_YELLOW_MAX,
        minor_step: 1000,
        major_step: 3000,
        title: "FPS",
        unit: "FPS",
        decimals: 0,
        label_inset: 4,
        label_lateral: 8,
        label_font_scale: 0.088,
        label_every: 1,
        label_radial_offset: -16,
        sub_text: framesText,
        sub_text_color: "blue",
      });
    }

    function drawStepGauge(canvas, samplesPerSec, totalSamples) {
      const samplesText = totalSamples != null ? Number(totalSamples).toLocaleString() : null;
      drawStyledGauge(canvas, samplesPerSec, {
        min: GAUGE_MIN_STEPS,
        max: GAUGE_MAX_STEPS,
        red_max: 10000,
        yellow_max: 30000,
        minor_step: 5000,
        major_step: 10000,
        title: "SAMP/s",
        unit: "S/S",
        decimals: 0,
        label_font_scale: 0.088,
        label_radial_offset: -4,
        sub_text: samplesText,
        sub_text_color: "orange",
      });
    }

    /* ═══════════════ Median-of-N filter (spike rejection) ════════════ */
    // Returns the median of up to `w` values centered on index `i`.
    // `valueFn(j)` should return Number or NaN for index j.
    function medianNeighbors(i, n, w, valueFn) {
      const half = (w - 1) >> 1;
      const lo = Math.max(0, i - half);
      const hi = Math.min(n - 1, i + half);
      const vals = [];
      for (let j = lo; j <= hi; j++) {
        const v = valueFn(j);
        if (Number.isFinite(v)) vals.push(v);
      }
      if (!vals.length) return NaN;
      vals.sort((a, b) => a - b);
      const mid = vals.length >> 1;
      return (vals.length & 1) ? vals[mid] : (vals[mid - 1] + vals[mid]) * 0.5;
    }

    /* ═══════════════ Nice-tick interval (1-2-5 sequence) ═══════════════ */
    function niceInterval(range, targetTicks) {
      if (!Number.isFinite(range) || range <= 0) return 1;
      const rawStep = range / Math.max(1, targetTicks);
      const mag = Math.pow(10, Math.floor(Math.log10(rawStep)));
      const residual = rawStep / mag;
      let nice;
      if (residual < 1.5) nice = 1;
      else if (residual < 3) nice = 2;
      else if (residual < 7) nice = 5;
      else nice = 10;
      return nice * mag;
    }

    /* ══════════════════════════════════════════════════════════════
     * CHART RENDERING — Time-series line charts with gradient fill,
     * smoothing, grid, axes, and auto-scaled Y range.
     * ══════════════════════════════════════════════════════════════ */
    function drawChart(canvas, history, seriesDefs, maxLookbackSec = (5 * 60), useLinearTime = false) {
      const points = Array.isArray(history) ? history : [];
      const width = canvas.clientWidth || 320;
      const height = canvas.clientHeight || 210;
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);

      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);

      if (!points.length) return;

      const axisDefs = seriesDefs.filter((s) => !!s.axis);
      const axisSourceSeries = axisDefs.length ? axisDefs : seriesDefs;
      const axisSlotDefault = 34;
      const axisSlotFor = (s) => {
        const v = Number(s?.axis?.label_pad);
        return Number.isFinite(v) ? Math.max(20, v) : axisSlotDefault;
      };
      const leftAxisSeries = axisSourceSeries.filter((s) => (s.axis?.side || "left") === "left");
      const rightAxisSeries = axisSourceSeries.filter((s) => (s.axis?.side || "right") === "right");
      const leftAxisPad = leftAxisSeries.length
        ? leftAxisSeries.reduce((sum, s) => sum + axisSlotFor(s), 0)
        : axisSlotDefault;
      const rightAxisPad = rightAxisSeries.length
        ? rightAxisSeries.reduce((sum, s) => sum + axisSlotFor(s), 0)
        : 0;
      const padL = 26 + leftAxisPad;
      const padR = rightAxisSeries.length ? (26 + rightAxisPad) : 12;
      const padT = 14, padB = 30;
      const plotW = width - padL - padR;
      const plotH = height - padT - padB;
      if (plotW <= 20 || plotH <= 20) return;

      // Time-compressed x-axis with quarter anchors.
      // Window spans min(20m, actual buffered time), so during first 20m
      // nothing scrolls off the left edge.
      const tsVals = points
        .map((p) => Number(p.ts))
        .filter((v) => Number.isFinite(v));
      const hasTimeAxis = tsVals.length >= 2;
      const newestTs = hasTimeAxis ? Math.max(...tsVals) : 0.0;
      const oldestTs = hasTimeAxis ? Math.min(...tsVals) : 0.0;
      const maxAge = hasTimeAxis ? Math.max(1e-6, newestTs - oldestTs) : 0.0;
      const AXIS_HARD_MAX_LOOKBACK_S = Math.max(1, Number(maxLookbackSec) || (5 * 60));
      const axisMaxLookbackSec = hasTimeAxis
        ? Math.min(AXIS_HARD_MAX_LOOKBACK_S, maxAge)
        : AXIS_HARD_MAX_LOOKBACK_S;
      const anchorScale = axisMaxLookbackSec / AXIS_HARD_MAX_LOOKBACK_S;
      const LOOKBACK_FRAC_ANCHORS = [0.0, 0.25, 0.50, 0.75, 1.0];
      const LOOKBACK_AGE_ANCHORS = [
        0.0,
        10.0 * anchorScale,
        60.0 * anchorScale,
        600.0 * anchorScale,
        axisMaxLookbackSec,
      ];
      // Shape-preserving monotone cubic interpolation (continuous slope)
      // so compression changes smoothly instead of with visible segment kinks.
      const makeMonotoneSpline = (xsRaw, ysRaw) => {
        const n = Math.min(xsRaw.length, ysRaw.length);
        if (n < 2) {
          const y0 = Number(ysRaw?.[0]) || 0.0;
          return () => y0;
        }
        const xs = xsRaw.slice(0, n).map((v) => Number(v));
        const ys = ysRaw.slice(0, n).map((v) => Number(v));
        const h = new Array(n - 1);
        const d = new Array(n - 1);
        for (let i = 0; i < n - 1; i++) {
          const dx = Math.max(1e-9, xs[i + 1] - xs[i]);
          h[i] = dx;
          d[i] = (ys[i + 1] - ys[i]) / dx;
        }
        const m = new Array(n);
        m[0] = d[0];
        m[n - 1] = d[n - 2];
        for (let i = 1; i < n - 1; i++) {
          m[i] = 0.5 * (d[i - 1] + d[i]);
        }
        for (let i = 0; i < n - 1; i++) {
          if (Math.abs(d[i]) <= 1e-12) {
            m[i] = 0.0;
            m[i + 1] = 0.0;
            continue;
          }
          const a = m[i] / d[i];
          const b = m[i + 1] / d[i];
          const s = (a * a) + (b * b);
          if (s > 9.0) {
            const t = 3.0 / Math.sqrt(s);
            m[i] = t * a * d[i];
            m[i + 1] = t * b * d[i];
          }
        }
        return (xRaw) => {
          const x = Math.max(xs[0], Math.min(xs[n - 1], Number(xRaw) || 0.0));
          let k = n - 2;
          for (let i = 0; i < n - 1; i++) {
            if (x <= xs[i + 1]) {
              k = i;
              break;
            }
          }
          const hk = h[k];
          const t = (x - xs[k]) / hk;
          const t2 = t * t;
          const t3 = t2 * t;
          const h00 = (2.0 * t3) - (3.0 * t2) + 1.0;
          const h10 = t3 - (2.0 * t2) + t;
          const h01 = (-2.0 * t3) + (3.0 * t2);
          const h11 = t3 - t2;
          return (h00 * ys[k]) + (h10 * hk * m[k]) + (h01 * ys[k + 1]) + (h11 * hk * m[k + 1]);
        };
      };
      const ageFromLookbackFrac = makeMonotoneSpline(LOOKBACK_FRAC_ANCHORS, LOOKBACK_AGE_ANCHORS);
      const lookbackFracFromAge = makeMonotoneSpline(LOOKBACK_AGE_ANCHORS, LOOKBACK_FRAC_ANCHORS);

      const xNormFromAge = (ageRaw) => {
        if (!hasTimeAxis || maxAge <= 0) return 1.0;
        const age = Math.max(0.0, Math.min(axisMaxLookbackSec, ageRaw));
        const lookbackFrac = useLinearTime
          ? (age / Math.max(1e-9, axisMaxLookbackSec))
          : lookbackFracFromAge(age);
        return 1.0 - lookbackFrac;
      };

      const ageFromXNorm = (xNormRaw) => {
        const xn = Math.max(0.0, Math.min(1.0, xNormRaw));
        const lookbackFrac = 1.0 - xn;
        return useLinearTime
          ? (lookbackFrac * axisMaxLookbackSec)
          : ageFromLookbackFrac(lookbackFrac);
      };

      const formatLookback = (ageSecRaw) => {
        const ageSec = Math.max(0.0, Number(ageSecRaw) || 0.0);
        if (ageSec < 90.0) {
          return `${Math.round(ageSec)}s`;
        }
        const mins = ageSec / 60.0;
        if (mins < 90.0) {
          return `${Math.round(mins)}m`;
        }
        const hours = mins / 60.0;
        if (hours < 48.0) {
          return `${hours < 10.0 ? hours.toFixed(1) : Math.round(hours)}h`;
        }
        const days = hours / 24.0;
        return `${days < 10.0 ? days.toFixed(1) : Math.round(days)}d`;
      };

      const xAt = (i) => {
        if (!hasTimeAxis) {
          const t = points.length <= 1 ? 1.0 : (i / (points.length - 1));
          return padL + (t * plotW);
        }
        const ts = Number(points[i].ts);
        const age = Number.isFinite(ts) ? (newestTs - ts) : maxAge;
        const xn = xNormFromAge(age);
        return padL + (xn * plotW);
      };

      const seriesByKey = new Map(seriesDefs.map((s) => [s.key, s]));
      const seriesValue = (row, key) => {
        const spec = seriesByKey.get(key);
        const raw = row[key];
        const val = spec && spec.map ? spec.map(raw) : raw;
        return Number(val);
      };

      const axes = [];
      let leftAxisOffset = 0;
      let rightAxisOffset = 0;
      for (const s of axisSourceSeries) {
        const side = s.axis?.side === "right" ? "right" : "left";
        const axisSlot = axisSlotFor(s);
        const sourceKeys = Array.isArray(s.axis?.group_keys) && s.axis.group_keys.length
          ? s.axis.group_keys
          : [s.key];
        const values = [];
        for (const row of points) {
          for (const key of sourceKeys) {
            const val = seriesValue(row, key);
            if (Number.isFinite(val)) {
              values.push(val);
            }
          }
        }

        const hasFixedMin = Number.isFinite(s.axis?.min);
        const hasFixedMax = Number.isFinite(s.axis?.max);
        let minV = hasFixedMin ? Number(s.axis.min) : (values.length ? Math.min(...values) : 0.0);
        let maxV = hasFixedMax ? Number(s.axis.max) : (values.length ? Math.max(...values) : 1.0);
        const minFloor = Number(s.axis?.min_floor);
        const maxFloor = Number(s.axis?.max_floor);
        if (!hasFixedMin && Number.isFinite(minFloor)) {
          minV = Math.min(minV, minFloor);
        }
        if (!hasFixedMax && Number.isFinite(maxFloor)) {
          maxV = Math.max(maxV, maxFloor);
        }
        // Snap max to a fixed multiple (e.g. multiples of 25)
        const maxSnap = Number(s.axis?.max_snap);
        if (!hasFixedMax && Number.isFinite(maxSnap) && maxSnap > 0) {
          maxV = Math.ceil(maxV / maxSnap) * maxSnap;
        }
        if (maxV < minV) maxV = minV + 1.0;
        if (minV === maxV) {
          if (hasFixedMin && !hasFixedMax) maxV = minV + 1.0;
          else if (!hasFixedMin && hasFixedMax) minV = maxV - 1.0;
          else { minV -= 1.0; maxV += 1.0; }
        }
        // Nice-tick autoscaling: snap axis bounds to clean 1-2-5 intervals
        const targetTicks = Number.isFinite(s.axis?.target_ticks) ? Math.max(2, s.axis.target_ticks) : 4;
        const niceStep = niceInterval(maxV - minV, targetTicks);
        if (!hasFixedMin) minV = Math.floor(minV / niceStep) * niceStep;
        if (!hasFixedMax) maxV = Math.ceil(maxV / niceStep) * niceStep;
        if (maxV <= minV) maxV = minV + niceStep;

        const axisX = side === "left"
          ? (padL - 20 - leftAxisOffset)
          : (width - padR + 20 + rightAxisOffset);
        if (side === "left") {
          leftAxisOffset += axisSlot;
        } else {
          rightAxisOffset += axisSlot;
        }
        let ticks;
        if (Array.isArray(s.axis?.ticks) && s.axis.ticks.length) {
          ticks = s.axis.ticks;
        } else {
          // If the nice step produces too many ticks, coarsen until it fits
          let step = niceStep;
          for (let attempt = 0; attempt < 3; attempt++) {
            ticks = [];
            for (let t = minV; t <= maxV + step * 0.001; t += step) {
              ticks.push(t);
            }
            if (ticks.length <= targetTicks + 1) break;
            step = niceInterval(maxV - minV, Math.max(2, Math.floor(targetTicks * 0.6)));
            if (!hasFixedMin) minV = Math.floor(minV / step) * step;
            if (!hasFixedMax) maxV = Math.ceil(maxV / step) * step;
            if (maxV <= minV) maxV = minV + step;
          }
          if (!ticks.length) ticks = [minV, maxV];
        }

        axes.push({
          key: s.key,
          side,
          x: axisX,
          color: s.color,
          min: minV,
          max: maxV,
          ticks,
          tickDecimals: Number.isFinite(s.axis?.tick_decimals)
            ? Math.max(0, Number(s.axis.tick_decimals))
            : Math.max(0, -Math.floor(Math.log10(ticks.length > 1 ? (ticks[1] - ticks[0]) : niceStep))),
        });
      }
      if (!axes.length) return;

      const axisByKey = new Map(axes.map((a) => [a.key, a]));

      // Build map of extra series colors sharing each axis (via axis_ref)
      const axisExtraColors = new Map();
      for (const s of seriesDefs) {
        if (s.axis_ref) {
          const colors = axisExtraColors.get(s.axis_ref) || [];
          colors.push(s.color);
          axisExtraColors.set(s.axis_ref, colors);
        }
      }

      const yAt = (axis, value) => {
        const t = (value - axis.min) / (axis.max - axis.min);
        return padT + (1.0 - t) * plotH;
      };

      ctx.lineWidth = 1.0;
      ctx.strokeStyle = "rgba(148,163,184,0.18)";
      // Denser grid: 4x the prior vertical density, then mirror that pixel step
      // on X so the plot reads as a true grid.
      const gridRows = 12; // was effectively 3 intervals
      const gridStep = plotH / gridRows;
      for (let i = 0; i <= gridRows; i++) {
        const y = padT + (gridStep * i);
        ctx.beginPath();
        ctx.moveTo(padL, y);
        ctx.lineTo(width - padR, y);
        ctx.stroke();
      }
      for (let x = padL + gridStep; x < (width - padR - 0.5); x += gridStep) {
        ctx.beginPath();
        ctx.moveTo(x, padT);
        ctx.lineTo(x, height - padB);
        ctx.stroke();
      }

      // Colored vertical axes with matching tick marks.
      ctx.font = "11px 'Avenir Next', 'Segoe UI', sans-serif";
      ctx.textBaseline = "middle";
      for (const axis of axes) {
        const isLeft = axis.side === "left";
        const tickDir = isLeft ? 1 : -1;

        // Draw extra color indicator lines for series sharing this axis
        const extraColors = axisExtraColors.get(axis.key) || [];
        for (let ec = extraColors.length - 1; ec >= 0; ec--) {
          const offset = ((ec + 1) * 2 + 2) * (isLeft ? -1 : 1);
          ctx.strokeStyle = extraColors[ec];
          ctx.globalAlpha = 0.65;
          ctx.lineWidth = 2.0;
          ctx.beginPath();
          ctx.moveTo(axis.x + offset, padT);
          ctx.lineTo(axis.x + offset, height - padB);
          ctx.stroke();
          ctx.globalAlpha = 1.0;
        }

        // Main axis line with tick marks
        ctx.strokeStyle = axis.color;
        ctx.globalAlpha = 0.65;
        ctx.lineWidth = 2.0;
        ctx.beginPath();
        ctx.moveTo(axis.x, padT);
        ctx.lineTo(axis.x, height - padB);
        ctx.stroke();
        ctx.globalAlpha = 1.0;

        for (const tv of axis.ticks) {
          if (!Number.isFinite(tv)) continue;
          const y = yAt(axis, Number(tv));
          if (!Number.isFinite(y)) continue;
          if (y < (padT - 3) || y > (height - padB + 3)) continue;

          // "Cool" tick: bright short tick plus a faint glow extension.
          ctx.strokeStyle = axis.color;
          ctx.lineWidth = 2.2;
          ctx.beginPath();
          ctx.moveTo(axis.x, y);
          ctx.lineTo(axis.x + tickDir * 8, y);
          ctx.stroke();

          ctx.globalAlpha = 0.35;
          ctx.lineWidth = 1.2;
          ctx.beginPath();
          ctx.moveTo(axis.x + tickDir * 8, y);
          ctx.lineTo(axis.x + tickDir * 14, y);
          ctx.stroke();
          ctx.globalAlpha = 1.0;

          const labelText = Number.isFinite(axis.tickDecimals)
            ? Number(tv).toFixed(axis.tickDecimals)
            : (Math.abs(tv) >= 100 ? `${Math.round(tv)}` : `${Number(tv).toFixed(0)}`);
          ctx.fillStyle = axis.color;
          ctx.textAlign = isLeft ? "right" : "left";
          ctx.fillText(labelText, axis.x - tickDir * 12, y);
        }

      }

      // Horizontal lookback axis (0 = now at right, 1 = oldest at left).
      const xAxisColor = axes[0]?.color || "#22c55e";
      const xAxisY = height - 18;
      const xTickDefs = [
        { frac: 0.0 },
        { frac: 0.25 },
        { frac: 0.5 },
        { frac: 0.75 },
        { frac: 1.0 },
      ];
      ctx.strokeStyle = xAxisColor;
      ctx.globalAlpha = 0.65;
      ctx.lineWidth = 2.0;
      ctx.beginPath();
      ctx.moveTo(padL, xAxisY);
      ctx.lineTo(width - padR, xAxisY);
      ctx.stroke();
      ctx.globalAlpha = 1.0;

      ctx.font = "11px 'Avenir Next', 'Segoe UI', sans-serif";
      ctx.textBaseline = "top";
      for (const tk of xTickDefs) {
        const frac = Math.max(0.0, Math.min(1.0, Number(tk.frac)));
        const xNorm = 1.0 - frac;
        const x = padL + (xNorm * plotW);
        const labelText = hasTimeAxis
          ? formatLookback(ageFromXNorm(xNorm))
          : (frac === 0.0 ? "0s" : "n/a");

        // Bright tick plus faint extension, matching vertical style.
        ctx.strokeStyle = xAxisColor;
        ctx.lineWidth = 2.2;
        ctx.beginPath();
        ctx.moveTo(x, xAxisY);
        ctx.lineTo(x, xAxisY - 8);
        ctx.stroke();

        ctx.globalAlpha = 0.35;
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(x, xAxisY - 8);
        ctx.lineTo(x, xAxisY - 14);
        ctx.stroke();
        ctx.globalAlpha = 1.0;

        ctx.fillStyle = xAxisColor;
        if (frac <= 1e-9) {
          ctx.textAlign = "right";
        } else if (frac >= (1.0 - 1e-9)) {
          ctx.textAlign = "left";
        } else {
          ctx.textAlign = "center";
        }
        ctx.fillText(labelText, x, xAxisY + 2);
      }

      const n = points.length;
      for (let si = 0; si < seriesDefs.length; si++) {
        const s = seriesDefs[si];
        const yOff = si;  // 1px vertical offset per series so overlapping lines stay visible
        const axis = s.axis_ref
          ? axisByKey.get(s.axis_ref)
          : (axisByKey.get(s.key) || axes[0]);
        if (!axis) continue;
        const smoothAlpha = Number.isFinite(s.smooth_alpha) ? Number(s.smooth_alpha) : CHART_VALUE_SMOOTH_ALPHA;
        const mw = Number.isFinite(s.median_window) ? s.median_window : 0;
        const readVal = mw >= 3
          ? (i) => medianNeighbors(i, n, mw, (j) => seriesValue(points[j], s.key))
          : (i) => seriesValue(points[i], s.key);
        ctx.strokeStyle = s.color;
        ctx.lineWidth = 2.0;
        ctx.beginPath();
        let started = false;
        let smoothVal = null;
        if (s.pixel_bin_avg) {
          const bins = new Map();
          for (let i = n - 1; i >= 0; i--) {
            const val = readVal(i);
            if (!Number.isFinite(val)) continue;
            const x = xAt(i);
            const xPx = Math.max(padL, Math.min(width - padR, Math.round(x)));
            const b = bins.get(xPx);
            if (b) {
              b.sum += Number(val);
              b.count += 1;
            } else {
              bins.set(xPx, { sum: Number(val), count: 1 });
            }
          }
          const xKeys = Array.from(bins.keys()).sort((a, b) => b - a);
          for (const xPx of xKeys) {
            const b = bins.get(xPx);
            if (!b || b.count <= 0) continue;
            const vAvg = b.sum / b.count;
            smoothVal = (smoothVal === null)
              ? vAvg
              : (smoothVal + ((vAvg - smoothVal) * smoothAlpha));
            const y = yAt(axis, smoothVal) + yOff;
            if (!started) {
              ctx.moveTo(xPx, y);
              started = true;
            } else {
              ctx.lineTo(xPx, y);
            }
          }
        } else {
          for (let i = n - 1; i >= 0; i--) {
            const val = readVal(i);
            if (!Number.isFinite(val)) continue;
            smoothVal = (smoothVal === null)
              ? Number(val)
              : (smoothVal + ((Number(val) - smoothVal) * smoothAlpha));
            const x = xAt(i);
            const y = yAt(axis, smoothVal) + yOff;
            if (!started) {
              ctx.moveTo(x, y);
              started = true;
            } else {
              ctx.lineTo(x, y);
            }
          }
        }
        ctx.stroke();
      }
    }

    /* ── Mini chart (sparkline) for inline card display ─────────── */
    function _outerHeight(el) {
      if (!(el instanceof HTMLElement)) return 0;
      const rect = el.getBoundingClientRect();
      const cs = getComputedStyle(el);
      return rect.height + (parseFloat(cs.marginTop) || 0) + (parseFloat(cs.marginBottom) || 0);
    }

    function layoutMiniCharts() {
      const cards = document.querySelectorAll(".mini-metric-card");
      const MIN_MINI_CHART_HEIGHT = 32;
      const MAX_MINI_CHART_HEIGHT = 180;
      for (const card of cards) {
        if (!(card instanceof HTMLElement)) continue;
        const cardChildren = Array.from(card.children).filter((child) => child instanceof HTMLElement);
        const cardStyle = getComputedStyle(card);
        const cardGap = parseFloat(cardStyle.rowGap || cardStyle.gap) || 0;
        const cardPadTop = parseFloat(cardStyle.paddingTop) || 0;
        const cardPadBottom = parseFloat(cardStyle.paddingBottom) || 0;
        // clientHeight includes card padding; subtract it so we compute the
        // available content area and avoid feedback growth on each relayout.
        const cardContentHeight = Math.max(0, card.clientHeight - cardPadTop - cardPadBottom);
        const inline = card.querySelector(":scope > .mini-inline");
        const directCanvas = card.querySelector(":scope > .mini-canvas");

        if (inline instanceof HTMLElement) {
          const nonInline = cardChildren.filter((child) => child !== inline);
          const used = nonInline.reduce((sum, child) => sum + _outerHeight(child), 0);
          const gaps = Math.max(0, cardChildren.length - 1) * cardGap;
          const available = Math.max(
            MIN_MINI_CHART_HEIGHT,
            Math.min(MAX_MINI_CHART_HEIGHT, Math.floor(cardContentHeight - used - gaps))
          );
          inline.style.height = `${available}px`;
          const inlineCanvases = inline.querySelectorAll(".mini-canvas");
          for (const canvas of inlineCanvases) {
            if (canvas instanceof HTMLElement) {
              canvas.style.height = `${available}px`;
              canvas.style.maxHeight = `${available}px`;
            }
          }
          continue;
        }

        if (directCanvas instanceof HTMLElement) {
          const nonCanvas = cardChildren.filter((child) => child !== directCanvas);
          const used = nonCanvas.reduce((sum, child) => sum + _outerHeight(child), 0);
          const gaps = Math.max(0, cardChildren.length - 1) * cardGap;
          const available = Math.max(
            MIN_MINI_CHART_HEIGHT,
            Math.min(MAX_MINI_CHART_HEIGHT, Math.floor(cardContentHeight - used - gaps))
          );
          directCanvas.style.height = `${available}px`;
          directCanvas.style.maxHeight = `${available}px`;
        }
      }
    }

    function drawMiniChart(canvas, history, seriesDefs, fillBetween) {
      if (!canvas) return;
      const points = history.slice(-240);
      const width = canvas.clientWidth || 120;
      const height = canvas.clientHeight || 104;
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);

      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);
      if (!points.length) return;

      const axisPad = 30;
      const padL = axisPad;
      const padR = 4;
      const isHalf = canvas.closest && canvas.closest('.card-half');
      const padTop = isHalf ? 1 : 6;
      const padBot = isHalf ? 3 : 6;
      const plotW = width - padL - padR;
      const plotH = height - padTop - padBot;
      if (plotW <= 4 || plotH <= 4) return;

      const values = [];
      for (const row of points) {
        for (const s of seriesDefs) {
          const v = Number(row[s.key]);
          if (Number.isFinite(v)) {
            values.push(v);
          }
        }
      }
      if (!values.length) return;

      const hasFixedMin = Number.isFinite(seriesDefs?.[0]?.axis?.min);
      const hasFixedMax = Number.isFinite(seriesDefs?.[0]?.axis?.max);
      const minRange = Number.isFinite(seriesDefs?.[0]?.axis?.min_range)
        ? Number(seriesDefs[0].axis.min_range) : 0;
      let minV = hasFixedMin ? Number(seriesDefs[0].axis.min) : Math.min(...values);
      let maxV = hasFixedMax ? Number(seriesDefs[0].axis.max) : Math.max(...values);
      if (maxV <= minV) maxV = minV + 1.0;
      // Nice-tick autoscaling for mini charts
      const miniTargetTicks = Number.isFinite(seriesDefs?.[0]?.axis?.target_ticks)
        ? Math.max(2, seriesDefs[0].axis.target_ticks) : 2;
      const miniNiceStep = niceInterval(maxV - minV, miniTargetTicks);
      if (!hasFixedMin) minV = Math.floor(minV / miniNiceStep) * miniNiceStep;
      if (!hasFixedMax) maxV = Math.ceil(maxV / miniNiceStep) * miniNiceStep;
      if (maxV <= minV) maxV = minV + miniNiceStep;
      // Enforce minimum visible range (expand around midpoint, respecting fixed bounds)
      if (minRange > 0 && (maxV - minV) < minRange) {
        const mid = (minV + maxV) * 0.5;
        minV = mid - minRange * 0.5;
        maxV = mid + minRange * 0.5;
        if (hasFixedMin && minV < Number(seriesDefs[0].axis.min)) {
          minV = Number(seriesDefs[0].axis.min);
          maxV = minV + minRange;
        }
        if (hasFixedMax && maxV > Number(seriesDefs[0].axis.max)) {
          maxV = Number(seriesDefs[0].axis.max);
          minV = maxV - minRange;
        }
      }

      // Logarithmic time-compressed x-axis (same anchors as big charts)
      const tsVals = points.map(p => Number(p.ts)).filter(v => Number.isFinite(v));
      const hasTs = tsVals.length >= 2;
      const newestTs = hasTs ? Math.max(...tsVals) : 0;
      const oldestTs = hasTs ? Math.min(...tsVals) : 0;
      const maxAge = hasTs ? Math.max(1e-6, newestTs - oldestTs) : 0;
      const HARD_MAX = 60 * 60;
      const axisMax = hasTs ? Math.min(HARD_MAX, maxAge) : HARD_MAX;
      const sc = axisMax / HARD_MAX;
      const FRAC_A = [0, 0.25, 0.50, 0.75, 1.0];
      const AGE_A = [0, 10 * sc, 60 * sc, 600 * sc, axisMax];
      const buildSpline = (xs, ys) => {
        const n2 = xs.length;
        if (n2 < 2) return () => ys[0] || 0;
        const h2 = [], d2 = [];
        for (let j = 0; j < n2 - 1; j++) {
          h2[j] = Math.max(1e-9, xs[j+1] - xs[j]);
          d2[j] = (ys[j+1] - ys[j]) / h2[j];
        }
        const m2 = [d2[0]];
        for (let j = 1; j < n2 - 1; j++) m2[j] = 0.5 * (d2[j-1] + d2[j]);
        m2[n2-1] = d2[n2-2];
        for (let j = 0; j < n2 - 1; j++) {
          if (Math.abs(d2[j]) <= 1e-12) { m2[j] = 0; m2[j+1] = 0; continue; }
          const aa = m2[j]/d2[j], bb = m2[j+1]/d2[j];
          const ss = aa*aa + bb*bb;
          if (ss > 9) { const tt = 3/Math.sqrt(ss); m2[j]=tt*aa*d2[j]; m2[j+1]=tt*bb*d2[j]; }
        }
        return (xr) => {
          const x2 = Math.max(xs[0], Math.min(xs[n2-1], xr));
          let k2 = n2 - 2;
          for (let j = 0; j < n2 - 1; j++) { if (x2 <= xs[j+1]) { k2 = j; break; } }
          const hk = h2[k2], t2 = (x2 - xs[k2]) / hk;
          const t22 = t2*t2, t23 = t22*t2;
          return ((2*t23 - 3*t22 + 1)*ys[k2]) + ((t23 - 2*t22 + t2)*hk*m2[k2])
               + ((-2*t23 + 3*t22)*ys[k2+1]) + ((t23 - t22)*hk*m2[k2+1]);
        };
      };
      const fracFromAge = buildSpline(AGE_A, FRAC_A);
      const xAtLog = (i) => {
        if (!hasTs) {
          const t = points.length <= 1 ? 1.0 : (i / (points.length - 1));
          return padL + (t * plotW);
        }
        const ts = Number(points[i].ts);
        if (!Number.isFinite(ts)) return padL;
        const age = Math.max(0, Math.min(axisMax, newestTs - ts));
        const frac = fracFromAge(age);
        return padL + ((1.0 - frac) * plotW);
      };
      const xAtLinear = (i) => {
        const t = points.length <= 1 ? 1.0 : (i / (points.length - 1));
        return padL + (t * plotW);
      };
      const xAt = xAtLog;
      const yAt = (v) => {
        const t = (v - minV) / (maxV - minV);
        return padTop + ((1.0 - t) * plotH);
      };

      const fmtTick = (v) => {
        const n = Number(v);
        if (!Number.isFinite(n)) return "";
        const a = Math.abs(n);
        if (a >= 100) return `${Math.round(n)}`;
        if (a >= 10) return `${n.toFixed(1)}`;
        if (a >= 1) return `${n.toFixed(2)}`;
        return `${n.toFixed(3)}`;
      };

      const axisColor = seriesDefs?.[seriesDefs.length - 1]?.color || "#22c55e";
      const axisX = padL - 1;
      // Generate ticks from the nice step so target_ticks controls density
      const yTicks = [];
      for (let tv = minV; tv <= maxV + miniNiceStep * 0.001; tv += miniNiceStep) {
        yTicks.push(tv);
      }
      if (!yTicks.length) yTicks.push(minV, maxV);
      ctx.strokeStyle = axisColor;
      ctx.globalAlpha = 0.52;
      ctx.lineWidth = 1.0;
      ctx.beginPath();
      ctx.moveTo(axisX, padTop);
      ctx.lineTo(axisX, height - padBot);
      ctx.stroke();
      ctx.globalAlpha = 1.0;

      ctx.font = "7px 'DotGothic16', 'Courier New', monospace";
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      for (const tv of yTicks) {
        const y = yAt(tv);
        ctx.strokeStyle = axisColor;
        ctx.globalAlpha = 0.70;
        ctx.lineWidth = 1.0;
        ctx.beginPath();
        ctx.moveTo(axisX - 2, y);
        ctx.lineTo(axisX + 2, y);
        ctx.stroke();
        ctx.globalAlpha = 1.0;
        ctx.fillStyle = "rgba(165, 191, 222, 0.95)";
        ctx.fillText(fmtTick(tv), axisX - 3, y);
      }

      // Soft center guide.
      const yMid = padTop + (plotH * 0.5);
      ctx.strokeStyle = "rgba(148, 163, 184, 0.18)";
      ctx.lineWidth = 1.0;
      ctx.beginPath();
      ctx.moveTo(padL, yMid);
      ctx.lineTo(width - padR, yMid);
      ctx.stroke();

      const n = points.length;

      // ── Fill-between gradient (e.g. q_max → q_min) ────────────
      if (fillBetween && fillBetween.length >= 4) {
        const [topKey, botKey, topColor, botColor] = fillBetween;
        const topSDef = seriesDefs.find(s => s.key === topKey);
        const botSDef = seriesDefs.find(s => s.key === botKey);
        if (topSDef && botSDef) {
          const smA_top = Number.isFinite(topSDef.smooth_alpha) ? topSDef.smooth_alpha : MINI_CHART_VALUE_SMOOTH_ALPHA;
          const smA_bot = Number.isFinite(botSDef.smooth_alpha) ? botSDef.smooth_alpha : MINI_CHART_VALUE_SMOOTH_ALPHA;
          const mwT = Number.isFinite(topSDef.median_window) ? topSDef.median_window : 0;
          const mwB = Number.isFinite(botSDef.median_window) ? botSDef.median_window : 0;
          const readT = mwT >= 3
            ? (i) => medianNeighbors(i, n, mwT, (j) => Number(points[j][topKey]))
            : (i) => Number(points[i][topKey]);
          const readB = mwB >= 3
            ? (i) => medianNeighbors(i, n, mwB, (j) => Number(points[j][botKey]))
            : (i) => Number(points[i][botKey]);
          const topPts = [], botPts = [];
          let svT = null, svB = null;
          for (let i = n - 1; i >= 0; i--) {
            const vT = readT(i), vB = readB(i);
            if (!Number.isFinite(vT) || !Number.isFinite(vB)) continue;
            svT = svT === null ? vT : svT + (vT - svT) * smA_top;
            svB = svB === null ? vB : svB + (vB - svB) * smA_bot;
            const x = xAtLog(i);
            topPts.push({ x, y: yAt(svT) });
            botPts.push({ x, y: yAt(svB) });
          }
          if (topPts.length >= 2) {
            const grad = ctx.createLinearGradient(0, padTop, 0, padTop + plotH);
            grad.addColorStop(0, topColor + "60");
            grad.addColorStop(0.5, botColor + "40");
            grad.addColorStop(1, botColor + "60");
            ctx.beginPath();
            ctx.moveTo(topPts[0].x, topPts[0].y);
            for (let i = 1; i < topPts.length; i++) ctx.lineTo(topPts[i].x, topPts[i].y);
            for (let i = botPts.length - 1; i >= 0; i--) ctx.lineTo(botPts[i].x, botPts[i].y);
            ctx.closePath();
            ctx.fillStyle = grad;
            ctx.globalAlpha = 0.5;
            ctx.fill();
            ctx.globalAlpha = 1.0;
          }
        }
      }

      for (let si = 0; si < seriesDefs.length; si++) {
        const s = seriesDefs[si];
        const yOff = si;  // 1px vertical offset per series so overlapping lines stay visible
        const xFn = s.linearTime ? xAtLinear : xAtLog;
        const smoothAlpha = Number.isFinite(s.smooth_alpha) ? Number(s.smooth_alpha) : MINI_CHART_VALUE_SMOOTH_ALPHA;
        const mw = Number.isFinite(s.median_window) ? s.median_window : 0;
        const readVal = mw >= 3
          ? (i) => medianNeighbors(i, n, mw, (j) => Number(points[j][s.key]))
          : (i) => Number(points[i][s.key]);
        ctx.strokeStyle = s.color;
        ctx.globalAlpha = (s.key === "level_1m") ? 0.95 : 0.82;
        ctx.lineWidth = (s.key === "level_1m") ? 2.0 : 1.6;
        ctx.beginPath();
        let started = false;
        let smoothVal = null;
        if (s.pixel_bin_avg) {
          const bins = new Map();
          for (let i = n - 1; i >= 0; i--) {
            const val = readVal(i);
            if (!Number.isFinite(val)) continue;
            const x = xFn(i);
            const xPx = Math.max(padL, Math.min(width - padR, Math.round(x)));
            const b = bins.get(xPx);
            if (b) {
              b.sum += val;
              b.count += 1;
            } else {
              bins.set(xPx, { sum: val, count: 1 });
            }
          }
          const xKeys = Array.from(bins.keys()).sort((a, b) => b - a);
          for (const xPx of xKeys) {
            const b = bins.get(xPx);
            if (!b || b.count <= 0) continue;
            const vAvg = b.sum / b.count;
            smoothVal = (smoothVal === null)
              ? vAvg
              : (smoothVal + ((vAvg - smoothVal) * smoothAlpha));
            const y = yAt(smoothVal) + yOff;
            if (!started) {
              ctx.moveTo(xPx, y);
              started = true;
            } else {
              ctx.lineTo(xPx, y);
            }
          }
        } else {
          for (let i = n - 1; i >= 0; i--) {
            const val = readVal(i);
            if (!Number.isFinite(val)) continue;
            smoothVal = (smoothVal === null)
              ? val
              : (smoothVal + ((val - smoothVal) * smoothAlpha));
            const x = xFn(i);
            const y = yAt(smoothVal) + yOff;
            if (!started) {
              ctx.moveTo(x, y);
              started = true;
            } else {
              ctx.lineTo(x, y);
            }
          }
        }
        ctx.stroke();
      }
      ctx.globalAlpha = 1.0;
    }

    function computeSmoothedStepSpd(now, history) {
      const rows = Array.isArray(history) ? history.slice(-STEP_GAUGE_AVG_WINDOW) : [];
      const vals = [];
      for (const row of rows) {
        const v = Number(row && row.steps_per_sec);
        if (Number.isFinite(v)) {
          vals.push(v);
        }
      }
      if (!vals.length) {
        const fallback = Number(now && now.steps_per_sec);
        return Number.isFinite(fallback) ? fallback : 0.0;
      }
      return vals.reduce((a, b) => a + b, 0.0) / vals.length;
    }

    function buildThroughputHistory(rows, stepWindowSec = 2.0, emaAlpha = 0.12) {
      const src = Array.isArray(rows) ? rows : [];
      if (!src.length) return [];
      const out = new Array(src.length);
      let j = 0;
      let ema = null;
      for (let i = 0; i < src.length; i++) {
        const row = src[i] || {};
        const tsI = Number(row.ts);
        const stI = Number(row.training_steps);
        const bs = Number(row.batch_size) || 1;
        while (j < i) {
          const tsJ = Number(src[j] && src[j].ts);
          if (!Number.isFinite(tsI) || !Number.isFinite(tsJ) || (tsI - tsJ) <= stepWindowSec) break;
          j += 1;
        }

        let rate = Number(row.steps_per_sec);
        if (i > j) {
          const tsJ = Number(src[j] && src[j].ts);
          const stJ = Number(src[j] && src[j].training_steps);
          const dt = tsI - tsJ;
          const ds = stI - stJ;
          if (Number.isFinite(dt) && dt > 1e-6 && Number.isFinite(ds) && ds >= 0) {
            rate = ds / dt;
          }
        }
        if (!Number.isFinite(rate)) rate = 0.0;
        const samplesRate = rate * bs;
        ema = (ema === null) ? samplesRate : (ema + ((samplesRate - ema) * emaAlpha));
        out[i] = { ...row, samples_per_sec_chart: ema };
      }
      return out;
    }

    function buildWindowSmoothedHistory(rows, specs, windowSec = 2.0) {
      const src = Array.isArray(rows) ? rows : [];
      if (!src.length) return [];
      const defs = Array.isArray(specs) ? specs : [];
      const out = new Array(src.length);
      const starts = new Array(defs.length).fill(0);
      const sums = new Array(defs.length).fill(0.0);
      const counts = new Array(defs.length).fill(0);
      const emas = new Array(defs.length).fill(null);

      for (let i = 0; i < src.length; i++) {
        const row = src[i] || {};
        const tsI = Number(row.ts);
        const nextRow = { ...row };

        for (let k = 0; k < defs.length; k++) {
          const def = defs[k] || {};
          const key = String(def.key || "");
          if (!key) continue;
          const alpha = Number.isFinite(def.alpha) ? Number(def.alpha) : 0.12;

          const vNow = Number(row[key]);
          if (Number.isFinite(vNow)) {
            sums[k] += vNow;
            counts[k] += 1;
          }

          while (starts[k] < i) {
            const rowStart = src[starts[k]] || {};
            const tsS = Number(rowStart.ts);
            if (!Number.isFinite(tsI) || !Number.isFinite(tsS) || (tsI - tsS) <= windowSec) break;
            const vS = Number(rowStart[key]);
            if (Number.isFinite(vS)) {
              sums[k] -= vS;
              counts[k] = Math.max(0, counts[k] - 1);
            }
            starts[k] += 1;
          }

          const avg = counts[k] > 0 ? (sums[k] / counts[k]) : (Number.isFinite(vNow) ? vNow : 0.0);
          emas[k] = (emas[k] === null) ? avg : (emas[k] + ((avg - emas[k]) * alpha));
          nextRow[key] = emas[k];
        }
        out[i] = nextRow;
      }
      return out;
    }

    /* ══════════════════════════════════════════════════════════════
     * LR COSINE MINI CHART — Draws the cosine annealing curve with
     * a dot showing current position in the cycle.
     * ══════════════════════════════════════════════════════════════ */
    const _lrCanvas = document.getElementById("lrMiniChart");
    const _lrCtx = _lrCanvas ? _lrCanvas.getContext("2d") : null;

    function drawLrMiniChart(now) {
      if (!_lrCtx) return;
      const width  = _lrCanvas.clientWidth  || 120;
      const height = _lrCanvas.clientHeight || 104;
      const dpr = window.devicePixelRatio || 1;
      _lrCanvas.width  = Math.floor(width * dpr);
      _lrCanvas.height = Math.floor(height * dpr);
      const ctx = _lrCtx;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);
      const W = width, H = height;

      const lrMax     = now.lr_max       || 5e-5;
      const lrMin     = now.lr_min       || 2e-5;
      const warmup    = now.lr_warmup_steps    || 5000;
      const period    = now.lr_cosine_period   || 3000000;
      const restarts  = !!now.lr_use_restarts;
      const step      = now.training_steps     || 0;

      // Draw area with left axis padding
      const axisPad = 38;
      const padL = axisPad, padR = 2, padTop = 2, padBot = 2;
      const gw = W - padL - padR, gh = H - padTop - padBot;
      if (gw <= 4 || gh <= 4) return;

      // Total steps we show: 2 full periods (or up to step + some if larger)
      const totalShow = restarts ? Math.max(period * 2, step + period * 0.3) : Math.max(period + warmup, step * 1.2);

      // Map step → x
      const sx = (s) => padL + (s / totalShow) * gw;
      // Map lr → y (top = lrMax, bottom = lrMin)
      const lrRange = Math.max(1e-12, lrMax - lrMin);
      const sy = (lr) => padTop + (1.0 - (lr - lrMin) / lrRange) * gh;

      // ── Vertical axis and ticks ──────────────────────────────────
      const fmtLr = (v) => {
        if (v === 0) return "0";
        const exp = Math.floor(Math.log10(Math.abs(v)));
        const mantissa = v / Math.pow(10, exp);
        if (Math.abs(mantissa - Math.round(mantissa)) < 0.05)
          return Math.round(mantissa) + "e" + exp;
        return mantissa.toFixed(1) + "e" + exp;
      };
      const nTicks = 3;
      const tickStep = niceInterval(lrRange, nTicks);
      const axisX = padL - 1;
      ctx.strokeStyle = "rgba(100,160,255,0.52)";
      ctx.lineWidth = 1.0;
      ctx.beginPath();
      ctx.moveTo(axisX, padTop);
      ctx.lineTo(axisX, H - padBot);
      ctx.stroke();

      ctx.font = "7px 'DotGothic16', 'Courier New', monospace";
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      for (let tv = Math.ceil(lrMin / tickStep) * tickStep; tv <= lrMax + tickStep * 0.001; tv += tickStep) {
        const y = sy(tv);
        ctx.strokeStyle = "rgba(100,160,255,0.70)";
        ctx.lineWidth = 1.0;
        ctx.beginPath();
        ctx.moveTo(axisX - 2, y);
        ctx.lineTo(axisX + 2, y);
        ctx.stroke();
        ctx.fillStyle = "rgba(165, 191, 222, 0.95)";
        ctx.fillText(fmtLr(tv), axisX - 3, y);
      }

      // Compute lr at a given step (mirrors get_lr)
      function lrAt(s) {
        if (s < warmup) return lrMin + (lrMax - lrMin) * ((s + 1) / Math.max(1, warmup));
        const dh = Math.max(1, period);
        const t = restarts ? ((s - warmup) % dh) : Math.min(s - warmup, dh);
        const cosine = 0.5 * (1.0 + Math.cos(Math.PI * t / dh));
        return lrMin + (lrMax - lrMin) * cosine;
      }

      // Draw curve
      const nPts = Math.min(280, gw);
      ctx.beginPath();
      for (let i = 0; i <= nPts; i++) {
        const s = (i / nPts) * totalShow;
        const x = sx(s), y = sy(lrAt(s));
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = "rgba(100,160,255,0.5)";
      ctx.lineWidth = 1.2;
      ctx.stroke();

      // Draw current position dot
      if (step > 0) {
        const cx = sx(step), cy = sy(lrAt(step));
        ctx.beginPath();
        ctx.arc(cx, cy, 3, 0, Math.PI * 2);
        ctx.fillStyle = "#0f0";
        ctx.fill();
        ctx.strokeStyle = "rgba(255,255,255,0.7)";
        ctx.lineWidth = 0.8;
        ctx.stroke();
      }
    }

    /* ══════════════════════════════════════════════════════════════
     * CARD UPDATER — Pushes latest snapshot values into DOM elements,
     * updates record highs, and sets model description.
     * ══════════════════════════════════════════════════════════════ */
    function updateCards(now, smoothedSteps) {
      _lastNow = now;
      const resetSeq = Number(now && now.records_reset_seq);
      if (Number.isFinite(resetSeq) && resetSeq !== _recordsResetSeqSeen) {
        _recordsResetSeqSeen = resetSeq;
        _clearRecordDisplays();
      }
      const clientRows = _normalizeClientRows(now.client_rows);
      const selectedPreviewClientId = Number(now.preview_selected_client_id);
      renderClientTable(clientRows, selectedPreviewClientId);
      syncPreviewClientSelection(clientRows, selectedPreviewClientId);
      cards.clients.textContent = fmtInt(now.client_count);
      cards.web.textContent = fmtInt(now.web_client_count);
      cards.level.textContent = fmtFloat(now.average_level, 2);
      cards.inf.textContent = `${fmtFloat(now.avg_inf_ms, 2)} ms`;
      setInfLed(now.avg_inf_ms);
      cards.rplf.innerHTML = toFixedCharCells(fmtPaddedFloat(now.rpl_per_frame, 2, 2, " "));
      setRplLed(now.rpl_per_frame);
      if (!_gsIgnoreSync) cards.eps.textContent = fmtPct(now.epsilon);

      // Pulse status indicator (SVG bolt prefix)
      const bolt = (fill) => `<svg width="10" height="14" viewBox="0 0 16 22" style="vertical-align:-2px;margin-right:2px"><path d="M10 0L3 12h5l-2 10 7-14H8z" fill="${fill}"/></svg>`;
      if (now.pulse_state === "pulsing") {
        cards.pulseStatus.innerHTML = bolt("#facc15") + 'PULSE ' + fmtInt(now.pulse_remaining) + " left";
        cards.pulseStatus.style.color = "#ef4444";
      } else if (now.pulse_state === "recovering") {
        cards.pulseStatus.innerHTML = bolt("#facc15") + 'RECOVER ' + fmtInt(now.pulse_remaining) + " left";
        cards.pulseStatus.style.color = "#f59e0b";
      } else if (!now.pulse_enabled) {
        cards.pulseStatus.innerHTML = bolt("#777") + 'DISABLED';
        cards.pulseStatus.style.color = "#555";
      } else {
        cards.pulseStatus.innerHTML = bolt("#7dd3fc") + 'PULSE ARMED';
        cards.pulseStatus.style.color = "#888";
      }

      if (!_gsIgnoreSync) cards.xprt.textContent = fmtPct(now.expert_ratio);
      cards.avgScore.textContent = fmtInt(now.avg_game_score || 0);
      cards.gameCount.textContent = fmtInt(now.game_count || 0);
      if (cards.gpu0) cards.gpu0.textContent = fmtInt(now.gpu0_util || 0) + "%";
      if (cards.gpu1) cards.gpu1.textContent = fmtInt(now.gpu1_util || 0) + "%";
      if (cards.cpu) cards.cpu.textContent = fmtInt(now.cpu_pct || 0) + "%";
      if (cards.memFree) cards.memFree.textContent = fmtFloat(now.mem_free_gb || 0, 1) + "G";
      if (cards.diskFree) cards.diskFree.textContent = fmtFloat(now.disk_free_gb || 0, 1) + "G";
      cards.rwrd.textContent = fmtInt(now.total_1m || 0);
      cards.loss.innerHTML = toFixedCharCells(fmtPaddedFloat(now.loss, 2, 2));
      cards.grad.innerHTML = toFixedCharCells(fmtPaddedFloat(now.grad_norm, 1, 3));
      cards.buf.textContent = fmtInt(now.memory_buffer_size);
      cards.lr.textContent = (now.lr === null || now.lr === undefined) ? "-" : Number(now.lr).toExponential(1);
      drawLrMiniChart(now);
      cards.q.innerHTML = (now.q_min === null || now.q_max === null)
        ? toFixedCharCells("-")
        : toColoredQRange(now.q_min, now.q_max);
      cards.epLen.textContent = fmtInt(now.eplen_1m);
      { const secs = Math.round((now.eplen_1m || 0) / GAME_FPS); const m = Math.floor(secs / 60); const s = secs % 60; cards.duration.textContent = m + ":" + String(s).padStart(2, "0"); }

      /* Episodes count + 30-second rolling rate */
      cards.episodes.textContent = fmtInt(now.episodes_this_run);
      {
        const t = now.ts || (Date.now() / 1000);
        epRateHistory.push({ts: t, episodes: now.episodes_this_run});
        while (epRateHistory.length > 1 && epRateHistory[0].ts < t - EP_RATE_WINDOW) epRateHistory.shift();
        if (epRateHistory.length >= 2) {
          const first = epRateHistory[0], last = epRateHistory[epRateHistory.length - 1];
          const dt = last.ts - first.ts;
          if (dt > 0.5) {
            const rate = (last.episodes - first.episodes) / dt;
            cards.epRate.textContent = fmtFloat(rate, 1) + "/s";
          }
        }
      }

      cards.agreePanel.textContent = (now.agreement_1m != null && isFinite(now.agreement_1m))
        ? fmtFloat(now.agreement_1m * 100, 1) + "%"
        : "0.0%";

      if (!ENABLE_CLIENT0_PREVIEW) {
        if (cards.previewLabel) cards.previewLabel.textContent = "PREVIEW (DISABLED)";
        setPreviewRenderMode("canvas");
        setPreviewMessage("Preview Disabled");
      } else {
        const previewSeq = Number(now.game_preview_seq || 0);
        const previewFps = Number(now.game_preview_fps || 0);
        const previewFmt = String(now.game_preview_source_format || "").toUpperCase();
        const previewRatio = Number(now.game_preview_compression_ratio || 1);
        const previewClientId = Number(now.game_preview_client_id);
        const previewLabelClientId = Number.isFinite(selectedPreviewClientId) && selectedPreviewClientId >= 0
          ? Math.trunc(selectedPreviewClientId)
          : previewClientId;
        const previewLabelBase = Number.isFinite(previewLabelClientId) && previewLabelClientId >= 0
          ? `CLIENT ${previewLabelClientId} PREVIEW`
          : "PREVIEW";
        if (cards.previewLabel) {
          let statText = "";
          if (previewFmt === "RLE" && Number.isFinite(previewRatio) && previewRatio > 1.0) {
            statText = `RLE ${previewRatio.toFixed(2)}x`;
          } else if (previewFmt === "LZSS" && Number.isFinite(previewRatio) && previewRatio > 1.0) {
            statText = `LZSS ${previewRatio.toFixed(2)}x`;
          } else if (previewFmt === "RAW") {
            statText = "RAW";
          }
          if (Number.isFinite(previewFps) && previewFps > 0.1) {
            statText = statText ? `${statText} @ ${previewFps.toFixed(1)}fps` : `${previewFps.toFixed(1)}fps`;
          }
          cards.previewLabel.textContent = statText ? `${previewLabelBase} ${statText}` : previewLabelBase;
        }
        const hasVideoTrack = _videoElementHasTrack();
        const freshVideo = _previewVideoIsFresh();
        if (freshVideo) {
          setPreviewRenderMode("video");
          setPreviewMessage("");
        } else if (_previewHasFrame) {
          setPreviewRenderMode("canvas");
          setPreviewMessage("");
        } else if ((_previewRtcEnabled && _previewRtcConnected) || hasVideoTrack) {
          setPreviewRenderMode("video");
          setPreviewMessage("Loading Preview");
        } else if ((now.client_count || 0) <= 0) {
          setPreviewMessage("No Clients");
          _previewHasFrame = false;
          clearPreviewCanvas();
        } else if (!Number.isFinite(previewSeq) || previewSeq <= 0) {
          setPreviewMessage(_previewRtcError ? `WebRTC off: ${_previewRtcError}` : "Waiting For Preview Client");
          if (!_previewHasFrame) clearPreviewCanvas();
        } else {
          setPreviewMessage(_previewHasFrame ? "" : (_previewRtcError ? `WebRTC off: ${_previewRtcError}` : "Loading Preview"));
        }
      }

      // ── Record highs ──────────────────────────────────────────────
      const recPairs = [
        ["rwrd", now.peak_game_score, fmtInt],
        ["level", now.peak_level, v => "LEVEL " + Math.round(v)],
        ["epLen", now.eplen_1m, fmtInt],
      ];
      for (const [k, v, fmt] of recPairs) {
        if (v != null && isFinite(v) && v > recordHighs[k]) {
          recordHighs[k] = v;
          if (recEls[k]) recEls[k].textContent = fmt(v);
        }
      }

      // ── Model description ─────────────────────────────────────────
      if (now.model_desc && modelDescEl) modelDescEl.textContent = now.model_desc;

      // ── Game settings sync ────────────────────────────────────────
      if (!_gsIgnoreSync && now.game_settings) {
        const gs = now.game_settings;
        if (gsAutoCurrEl.checked !== gs.auto_curriculum) {
          gsAutoCurrEl.checked = gs.auto_curriculum;
          _applyAutoCurriculum(gs.auto_curriculum);
        }
        if (gsAdvancedEl.checked !== gs.start_advanced) gsAdvancedEl.checked = gs.start_advanced;
        if (parseInt(gsLevelEl.value, 10) !== gs.start_level_min) gsLevelEl.value = String(gs.start_level_min);
      }
      // ── Preview checkbox sync from server ────────────────────────
      if (chkPreviewEl && typeof now.preview_capture_enabled !== "undefined") {
        chkPreviewEl.checked = !!now.preview_capture_enabled;
      }
      if (chkHudEl && typeof now.hud_enabled !== "undefined") {
        chkHudEl.checked = !!now.hud_enabled;
      }
      // ── Auto-curriculum: continuously recompute level each tick ──
      if (gsAutoCurrEl.checked && now.average_level != null) {
        const lv = _computeAutoLevel(now.average_level);
        if (parseInt(gsLevelEl.value, 10) !== lv) {
          gsLevelEl.value = String(lv);
          _postGameSettings({ start_level_min: lv });
        }
      }
    }

    function render(payload) {
      _tickDisplayFps();
      if (!payload || !payload.now) return;
      const history = Array.isArray(payload.history) ? payload.history.slice(-MAX_HISTORY_POINTS) : [];
      const history60m = sliceHistoryLookback(history, 60 * 60);
      const history2m = sliceHistoryLookback(history, 2 * 60);
      const history1m = sliceHistoryLookback(history, 60);
      const chartHistory60m = downsampleHistory(history60m, MAX_CHART_POINTS);
      const chartHistory2m = downsampleHistory(history2m, MAX_CHART_POINTS);
      const chartHistory1m = downsampleHistory(history1m, MAX_CHART_POINTS);
      const throughputHistory = buildThroughputHistory(chartHistory60m);
      const smoothedStepSpd = computeSmoothedStepSpd(payload.now, history60m);
      updateCards(payload.now, smoothedStepSpd);
      layoutMiniCharts();
      gaugeState.fps.target  = payload.now.fps;
      gaugeState.step.target = smoothedStepSpd;
      latestRow = payload.now;
      drawChart(charts.throughput.canvas, throughputHistory, charts.throughput.series, 60 * 60);
      drawChart(charts.rewards.canvas, chartHistory60m, charts.rewards.series, 60 * 60);
      drawChart(charts.learning.canvas, chartHistory1m, charts.learning.series, 60, true);
      drawChart(charts.agreement.canvas, chartHistory60m, charts.agreement.series, 60 * 60);
      drawMiniChart(charts.qRange.canvas, history60m, charts.qRange.series, charts.qRange.fill_between);
      drawMiniChart(charts.rewardMini.canvas, history60m, charts.rewardMini.series);
      drawMiniChart(charts.level1m.canvas, history60m, charts.level1m.series);
      drawMiniChart(charts.lossMini.canvas, history2m, charts.lossMini.series);
      drawMiniChart(charts.gradMini.canvas, history2m, charts.gradMini.series);
      drawMiniChart(charts.epLenMini.canvas, history60m, charts.epLenMini.series);
      drawMiniChart(charts.gpu0Mini.canvas, history2m, charts.gpu0Mini.series);
      drawMiniChart(charts.gpu1Mini.canvas, history2m, charts.gpu1Mini.series);
      drawMiniChart(charts.cpuMini.canvas, history2m, charts.cpuMini.series);
      drawMiniChart(charts.memFreeMini.canvas, history2m, charts.memFreeMini.series);
    }

    let historyCache = [];
    let latestNow = null;
    let lastTs = -1;

    function renderCurrent() {
      if (!latestNow) return;
      render({ now: latestNow, history: historyCache });
    }

    async function fetchHistory() {
      try {
        const res = await fetch(`/api/history?cid=${encodeURIComponent(CLIENT_ID)}&t=${Date.now()}`, { cache: "no-store" });
        if (!res.ok) throw new Error("bad response");
        const payload = await res.json();
        const now = payload && payload.now ? payload.now : null;
        const history = Array.isArray(payload && payload.history) ? payload.history : [];
        historyCache = history.slice(-MAX_HISTORY_POINTS);
        latestNow = now || historyCache[historyCache.length - 1] || latestNow;
        const ts = Number(latestNow && latestNow.ts);
        if (Number.isFinite(ts)) {
          lastTs = ts;
        }
        renderCurrent();
        setConnected(true);
      } catch (err) {
        setConnected(false);
      }
    }

    /* ══════════════════════════════════════════════════════════════
     * DATA FETCH LOOP — Polls /api/now and /api/ping at intervals,
     * maintains history cache, and triggers re-renders.
     * ══════════════════════════════════════════════════════════════ */
    async function fetchNow() {
      try {
        const res = await fetch(`/api/now?cid=${encodeURIComponent(CLIENT_ID)}&t=${Date.now()}`, { cache: "no-store" });
        if (!res.ok) throw new Error("bad response");
        const now = await res.json();
        latestNow = now;
        const ts = Number(now && now.ts);
        if (Number.isFinite(ts) && ts > lastTs + 1e-9) {
          historyCache.push(now);
          if (historyCache.length > MAX_HISTORY_POINTS) {
            historyCache.shift();
          }
          lastTs = ts;
        }
        renderCurrent();
        setConnected(true);
      } catch (err) {
        setConnected(false);
      }
    }

    async function heartbeat() {
      try {
        const res = await fetch(`/api/ping?cid=${encodeURIComponent(CLIENT_ID)}&t=${Date.now()}`, { cache: "no-store" });
        if (!res.ok) throw new Error("no ping");
        failedPings = 0;
        setConnected(true);
      } catch (err) {
        failedPings += 1;
        setConnected(false);
        if (failedPings >= 3) {
          try { window.close(); } catch (_) {}
        }
      }
    }

    if (previewQualitySelect) {
      previewQualitySelect.value = _previewProfileMode;
      previewQualitySelect.addEventListener("change", () => {
        const nextMode = String(previewQualitySelect.value || "auto").trim().toLowerCase();
        _previewProfileMode = (nextMode === "default" || nextMode === "mobile") ? nextMode : "auto";
        previewQualitySelect.value = _previewProfileMode;
        _saveStoredPreviewQuality(_previewProfileMode);
        _restartPreviewWebRTC().catch(() => {});
      });
    }
    if (clientTableHead) {
      clientTableHead.addEventListener("click", (ev) => {
        const target = ev.target instanceof Element ? ev.target : null;
        const button = target ? target.closest("button[data-sort-key]") : null;
        if (!button) return;
        const nextKey = String(button.dataset.sortKey || "client_id");
        if (_clientTableSortKey === nextKey) {
          _clientTableSortDir = (_clientTableSortDir === "asc") ? "desc" : "asc";
        } else {
          _clientTableSortKey = nextKey;
          _clientTableSortDir = "asc";
        }
        updateClientTableSortIndicators();
        if (_lastNow) {
          renderClientTable(_lastNow.client_rows, _lastNow.preview_selected_client_id);
        }
      });
    }
    if (clientTableBody) {
      clientTableBody.addEventListener("click", (ev) => {
        const target = ev.target instanceof Element ? ev.target : null;
        const row = target ? target.closest("tr[data-client-id]") : null;
        if (!row || row.dataset.previewCapable !== "1" || _previewClientRequestInFlight) return;
        if (row.classList.contains("selected") && _previewPendingClientId === null) return;
        const nextClientId = Number(row.dataset.clientId);
        if (!Number.isFinite(nextClientId) || nextClientId < 0) return;
        _requestPreviewClientSelection(nextClientId, "Switching Preview");
      });
    }

    if (_chatEnabled) {
      _fetchChatRows();
      setInterval(() => {
        if (document.visibilityState !== "visible") return;
        _fetchChatRows();
      }, CHAT_POLL_MS);
    }

    const cookiePref = getCookieValue(AUDIO_PREF_COOKIE);
    audioEnabled = (cookiePref === null) ? true : (cookiePref === "1");
    setAudioToggle(audioEnabled, false);
    const previewAudioPref = getCookieValue(PREVIEW_GAME_AUDIO_PREF_COOKIE);
    _previewGameAudioEnabled = PREVIEW_GAME_AUDIO_TRANSPORT_ENABLED && ((previewAudioPref === null) ? true : (previewAudioPref === "1"));
    if (chkPreviewGameAudioEl) {
      chkPreviewGameAudioEl.checked = _previewGameAudioEnabled;
      chkPreviewGameAudioEl.disabled = !PREVIEW_GAME_AUDIO_TRANSPORT_ENABLED;
      if (PREVIEW_GAME_AUDIO_TRANSPORT_ENABLED) {
        chkPreviewGameAudioEl.addEventListener("change", () => {
          _previewGameAudioEnabled = !!chkPreviewGameAudioEl.checked;
          setCookieValue(PREVIEW_GAME_AUDIO_PREF_COOKIE, _previewGameAudioEnabled ? "1" : "0");
          _ensurePreviewVideoAudio();
        });
      }
    }
    const kickAudioStart = () => {
      if (audioEnabled) ensureAudioPlaying();
      _ensurePreviewVideoAudio();
    };
    document.addEventListener("pointerdown", kickAudioStart, { passive: true });
    document.addEventListener("keydown", kickAudioStart);
    window.addEventListener("focus", kickAudioStart);
    document.addEventListener("visibilitychange", () => {
      if (!document.hidden) kickAudioStart();
    });
    if (audioToggle) {
      audioToggle.addEventListener("click", () => setAudioEnabled(!audioEnabled));
    }
    if (bgAudio) {
      bgAudio.addEventListener("playing", () => clearAudioRetryTimer());
      bgAudio.addEventListener("ended", () => {
        if (!audioEnabled || !audioPlaylist.length) return;
        playAudioAt(audioIndex + 1);
      });
      bgAudio.addEventListener("error", () => {
        if (!audioEnabled || !audioPlaylist.length) return;
        playAudioAt(audioIndex + 1);
      });
    }
    loadAudioPlaylist().catch(() => {});

    if (ENABLE_CLIENT0_PREVIEW) {
      _ensurePreviewWebRTC().finally(() => { previewPumpLoop(); });
    } else {
      _previewRtcEnabled = false;
      _previewRtcConnected = false;
      _previewHasFrame = false;
      setPreviewRenderMode("canvas");
      clearPreviewCanvas();
      setPreviewMessage("Preview Disabled");
    }
    fetchHistory().then(() => fetchNow()).catch(() => {});
    setInterval(fetchNow, DASH_REFRESH_MS);
    setInterval(heartbeat, 1000);
    window.addEventListener("resize", () => {
      renderCurrent();
      // Force gauge repaint since canvas dimensions changed
      drawFpsGauge(fpsGaugeCanvas, gaugeState.fps.current, latestRow ? latestRow.frame_count : null);
      const bsR = latestRow ? (Number(latestRow.batch_size) || 1) : 1;
      drawStepGauge(stepGaugeCanvas, gaugeState.step.current * bsR, latestRow ? latestRow.training_steps * bsR : null);
      if (!ENABLE_CLIENT0_PREVIEW) {
        clearPreviewCanvas();
        setPreviewMessage("Preview Disabled");
      } else if (_previewRtcEnabled && _previewRtcConnected) {
        setPreviewRenderMode("video");
      } else if (_previewHasFrame) drawPreviewToCard();
      else clearPreviewCanvas();
    });
  </script>
</body>
</html>
""".replace("__WEBRTC_ICE_SERVERS_JSON__", _ice_json) \
    .replace("__PREVIEW_GAME_AUDIO_TRANSPORT_ENABLED__", _preview_game_audio_json) \
    .replace("__PREVIEW_GAME_AUDIO_CHECKED__", _preview_game_audio_checked) \
    .replace("__PREVIEW_GAME_AUDIO_DISABLED__", _preview_game_audio_disabled)


def _make_handler(state: _DashboardState, rtc_bridge: _PreviewWebRTCBridge | None = None):
    _ice = rtc_bridge.ice_servers if rtc_bridge is not None else _WEBRTC_ICE_SERVERS
    page = _render_dashboard_html(_ice).encode("utf-8")
    chat_store = ChatStore(max_messages=240)
    audio_root = os.path.abspath(_audio_dir())
    fonts_root = os.path.abspath(_fonts_dir())
    html_root = os.path.abspath(_html_dir())

    class DashboardHandler(BaseHTTPRequestHandler):
        def _send(
            self,
            payload: bytes,
            content_type: str = "text/plain",
            status: int = 200,
            cache_control: str = "no-store",
        ):
            try:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", cache_control)
                self.end_headers()
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def _send_file(self, filepath: str, content_type: str):
            try:
                size = os.path.getsize(filepath)
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(size))
                self.send_header("Cache-Control", "public, max-age=3600")
                self.end_headers()
                with open(filepath, "rb") as f:
                    shutil.copyfileobj(f, self.wfile, length=64 * 1024)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
            except Exception:
                try:
                    self._send(b"Not Found", "text/plain; charset=utf-8", status=404)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass

        @staticmethod
        def _safe_audio_file(name: str) -> str | None:
            if not name or name.startswith("."):
                return None
            if "/" in name or "\\" in name:
                return None
            candidate = os.path.abspath(os.path.join(audio_root, name))
            try:
                if os.path.commonpath([audio_root, candidate]) != audio_root:
                    return None
            except Exception:
                return None
            if not os.path.isfile(candidate):
                return None
            ext = os.path.splitext(name)[1].lower()
            if ext not in AUDIO_EXTENSIONS:
                return None
            return candidate

        @staticmethod
        def _safe_font_file(name: str) -> str | None:
            if not name or name.startswith("."):
                return None
            if "/" in name or "\\" in name:
                return None
            candidate = os.path.abspath(os.path.join(fonts_root, name))
            try:
                if os.path.commonpath([fonts_root, candidate]) != fonts_root:
                    return None
            except Exception:
                return None
            if not os.path.isfile(candidate):
                return None
            ext = os.path.splitext(name)[1].lower()
            if ext not in FONT_EXTENSIONS:
                return None
            return candidate

        @staticmethod
        def _safe_html_file(name: str) -> str | None:
            if not name or name.startswith("."):
                return None
            if "/" in name or "\\" in name:
                return None
            candidate = os.path.abspath(os.path.join(html_root, name))
            try:
                if os.path.commonpath([html_root, candidate]) != html_root:
                    return None
            except Exception:
                return None
            if not os.path.isfile(candidate):
                return None
            return candidate

        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path
            query = parse_qs(parsed.query)
            client_id = (query.get("cid") or [None])[0]
            if path in ("/api/ping", "/api/now", "/api/history", "/api/game_preview", "/api/chat"):
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
                wait_timeout_s = 0.0
                profile = "default"
                try:
                    if "since" in query:
                        since_seq = int((query.get("since") or [0])[0])
                except Exception:
                    since_seq = None
                try:
                    wait_raw = (query.get("wait") or ["0"])[0]
                    if str(wait_raw).strip().lower() in {"1", "true", "yes", "on"}:
                        t_raw = (query.get("timeout") or ["1.0"])[0]
                        wait_timeout_s = max(0.0, min(5.0, float(t_raw)))
                except Exception:
                    wait_timeout_s = 0.0
                try:
                    profile = _normalize_preview_profile((query.get("profile") or ["default"])[0])
                except Exception:
                    profile = "default"
                self._send(state.game_preview_body(since_seq=since_seq, wait_timeout_s=wait_timeout_s, profile=profile), "application/json")
                return
            if path == "/api/audio_playlist":
                tracks = _list_audio_files()
                body = json.dumps({
                    "tracks": [{"name": name, "url": f"/api/audio/{quote(name)}"} for name in tracks]
                }).encode("utf-8")
                self._send(body, "application/json")
                return
            if path.startswith("/api/audio/"):
                raw_name = unquote(path[len("/api/audio/"):])
                safe_path = self._safe_audio_file(raw_name)
                if not safe_path:
                    self._send(b"Not Found", "text/plain; charset=utf-8", status=404)
                    return
                ctype = mimetypes.guess_type(safe_path)[0] or "application/octet-stream"
                self._send_file(safe_path, ctype)
                return
            if path.startswith("/api/font/"):
                raw_name = unquote(path[len("/api/font/"):])
                safe_path = self._safe_font_file(raw_name)
                if not safe_path:
                    self._send(b"Not Found", "text/plain; charset=utf-8", status=404)
                    return
                ctype = mimetypes.guess_type(safe_path)[0] or "font/ttf"
                self._send_file(safe_path, ctype)
                return
            if path.startswith("/api/html/"):
                raw_name = unquote(path[len("/api/html/"):])
                safe_path = self._safe_html_file(raw_name)
                if not safe_path:
                    self._send(b"Not Found", "text/plain; charset=utf-8", status=404)
                    return
                ctype = mimetypes.guess_type(safe_path)[0] or "application/octet-stream"
                self._send_file(safe_path, ctype)
                return
            if path == "/api/game_settings":
                body = json.dumps(game_settings.snapshot()).encode("utf-8")
                self._send(body, "application/json")
                return
            if path == "/api/chat":
                limit_raw = (query.get("limit") or ["120"])[0]
                try:
                    limit = max(1, min(240, int(limit_raw)))
                except Exception:
                    limit = 120
                body = json.dumps({"messages": chat_store.snapshot(limit=limit)}).encode("utf-8")
                self._send(body, "application/json")
                return
            self._send(b"Not Found", "text/plain; charset=utf-8", status=404)

        def do_POST(self):
            parsed = urlparse(self.path)
            path = parsed.path
            query = parse_qs(parsed.query)
            if path == "/api/game_preview_offer":
                try:
                    length = int(self.headers.get("Content-Length", 0))
                    raw = self.rfile.read(length) if length > 0 else b"{}"
                    data = json.loads(raw)
                    sdp = str(data.get("sdp", "") or "")
                    typ = str(data.get("type", "offer") or "offer")
                    profile = str(data.get("profile", "default") or "default").strip().lower()
                    if profile not in {"default", "mobile"}:
                        profile = "default"
                    if not sdp:
                        raise ValueError("missing_sdp")
                    if rtc_bridge is None:
                        raise RuntimeError("webrtc_bridge_unavailable")
                    answer = rtc_bridge.create_answer(sdp, typ, timeout_s=10.0, profile=profile)
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
                        game_settings.start_advanced = bool(data["start_advanced"])
                    if "start_level_min" in data:
                        game_settings.start_level_min = int(data["start_level_min"])
                    if "epsilon_pct" in data:
                        game_settings.epsilon_pct = int(data["epsilon_pct"])
                        # Dashboard wins → clear keyboard epsilon overrides
                        with state.metrics.lock:
                            state.metrics.manual_epsilon_override = False
                            state.metrics.override_epsilon = False
                    if "expert_pct" in data:
                        game_settings.expert_pct = int(data["expert_pct"])
                        # Dashboard wins → clear keyboard expert overrides
                        with state.metrics.lock:
                            state.metrics.manual_expert_override = False
                            state.metrics.override_expert = False
                            state.metrics.expert_mode = False
                    if "auto_curriculum" in data:
                        game_settings.auto_curriculum = bool(data["auto_curriculum"])
                    game_settings.save()
                    body = json.dumps(game_settings.snapshot()).encode("utf-8")
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
                        enabled = bool(data["enabled"])
                        with state.metrics.lock:
                            state.metrics.preview_capture_enabled = enabled
                            updates["enabled"] = enabled
                    if "hud_enabled" in data:
                        hud_enabled = bool(data["hud_enabled"])
                        with state.metrics.lock:
                            state.metrics.hud_enabled = hud_enabled
                            updates["hud_enabled"] = hud_enabled
                    if "client_id" in data:
                        preview_client_id = data.get("client_id")
                        if preview_client_id in ("", None):
                            target_cid = None
                        else:
                            target_cid = int(preview_client_id)
                            if target_cid < 0:
                                target_cid = None
                        srv = getattr(state.metrics, "global_server", None)
                        if srv is None:
                            raise RuntimeError("preview routing server unavailable")
                        ok, selected_cid = srv.set_preview_client(target_cid)
                        if not ok:
                            raise ValueError("invalid preview client")
                        updates["client_id"] = -1 if selected_cid is None else int(selected_cid)
                    if updates:
                        body = json.dumps({"ok": True, **updates}).encode("utf-8")
                        self._send(body, "application/json")
                    else:
                        self._send(b'{"error":"missing settings field"}', "application/json", status=400)
                except Exception:
                    self._send(b'{"error":"bad request"}', "application/json", status=400)
                return
            if path == "/api/chat":
                try:
                    ctype = str(self.headers.get("Content-Type", "") or "").lower()
                    if "application/json" not in ctype:
                        self._send(b'{"error":"content_type_must_be_json"}', "application/json", status=415)
                        return
                    length = int(self.headers.get("Content-Length", 0))
                    if length <= 0 or length > 4096:
                        self._send(b'{"error":"invalid_content_length"}', "application/json", status=400)
                        return
                    raw = self.rfile.read(length) if length > 0 else b"{}"
                    data = json.loads(raw)
                    if not isinstance(data, dict):
                        self._send(b'{"error":"invalid_json_payload"}', "application/json", status=400)
                        return
                    message_raw = data.get("message", "")
                    message = message_raw if isinstance(message_raw, str) else str(message_raw)
                    display_name_raw = data.get("display_name", "")
                    display_name = display_name_raw if isinstance(display_name_raw, str) else str(display_name_raw)

                    peer_ip = str(self.client_address[0] if self.client_address else "unknown")
                    # Only trust proxy-forwarded IP when request came from local loopback.
                    if peer_ip in {"127.0.0.1", "::1", "localhost"}:
                        forwarded = str(self.headers.get("X-Forwarded-For", "") or "").split(",")[0].strip()
                        ip = forwarded or peer_ip
                    else:
                        ip = peer_ip

                    row = chat_store.add_message(ip, message, display_name=display_name)
                    body = json.dumps({"ok": True, "message": row}).encode("utf-8")
                    self._send(body, "application/json", status=200)
                except ValueError as e:
                    code = str(e)
                    if code == "message_too_long":
                        self._send(b'{"error":"message_too_long","max":240}', "application/json", status=400)
                    else:
                        self._send(b'{"error":"message_empty"}', "application/json", status=400)
                except Exception:
                    self._send(b'{"error":"bad request"}', "application/json", status=400)
                return
            self._send(b"Not Found", "text/plain; charset=utf-8", status=404)

        def log_message(self, fmt, *args):
            return

        def handle_one_request(self):
            try:
                super().handle_one_request()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                self.close_connection = True

    return DashboardHandler


class MetricsDashboard:
    """Managed dashboard server + browser window."""

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
        # Cap sampler refresh at 30 Hz max.
        self.sample_interval = max(0.033, sample_interval)
        self.open_browser = open_browser

        self.state = _DashboardState(metrics_obj, agent_obj, history_limit=history_limit)
        self.rtc_bridge = _PreviewWebRTCBridge(metrics_obj, _WEBRTC_ICE_SERVERS)
        self.stop_event = threading.Event()
        self.httpd: ThreadingHTTPServer | None = None
        self.server_thread: threading.Thread | None = None
        self.sampler_thread: threading.Thread | None = None
        self.browser_proc: subprocess.Popen | None = None
        self.browser_profile_dir: str | None = None
        self.url: str | None = None
        self._closed = False
        self._lock = threading.Lock()
        atexit.register(self.stop)

    def _sampling_loop(self):
        while not self.stop_event.is_set():
            try:
                self.state.sample()
            except Exception:
                pass
            if self.open_browser and self.url and self.browser_proc and self.browser_proc.poll() is not None:
                try:
                    self.browser_proc = None
                    if self.browser_profile_dir:
                        shutil.rmtree(self.browser_profile_dir, ignore_errors=True)
                        self.browser_profile_dir = None
                    self._launch_browser(self.url)
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
    def _resolve_browser_binary(candidate: str) -> str | None:
        if os.path.isabs(candidate):
            return candidate if os.path.exists(candidate) else None
        return shutil.which(candidate)

    def _launch_browser(self, url: str):
        candidates = [
            ("google-chrome", True),
            ("google-chrome-stable", True),
            ("chromium", True),
            ("chromium-browser", True),
            ("brave-browser", True),
            ("msedge", True),
            ("microsoft-edge", True),
            ("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", True),
            ("/Applications/Chromium.app/Contents/MacOS/Chromium", True),
            ("/Applications/Brave Browser.app/Contents/MacOS/Brave Browser", True),
            ("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge", True),
        ]

        for candidate, chromium_like in candidates:
            binary = self._resolve_browser_binary(candidate)
            if not binary:
                continue
            cmd = [binary]
            profile_dir = None
            if chromium_like:
                profile_dir = tempfile.mkdtemp(prefix="robotron_dashboard_")
                cmd.extend(
                    [
                        "--new-window",
                        f"--app={url}",
                        f"--user-data-dir={profile_dir}",
                        "--no-first-run",
                        "--disable-features=TranslateUI",
                        "--no-default-browser-check",
                    ]
                )
            else:
                cmd.extend(["--new-window", url])

            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=(os.name != "nt"),
                )
                self.browser_proc = proc
                self.browser_profile_dir = profile_dir
                return
            except Exception:
                if profile_dir:
                    shutil.rmtree(profile_dir, ignore_errors=True)

        # Fallback if no managed browser was available.
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
            try:
                self.httpd.server_close()
            except Exception:
                pass
            self.httpd = None

        if self.server_thread and self.server_thread.is_alive():
            try:
                self.server_thread.join(timeout=2.0)
            except Exception:
                pass
        self.server_thread = None

        if self.sampler_thread and self.sampler_thread.is_alive():
            try:
                self.sampler_thread.join(timeout=1.0)
            except Exception:
                pass
        self.sampler_thread = None

        try:
            if self.rtc_bridge is not None:
                self.rtc_bridge.stop()
        except Exception:
            pass

        if self.browser_proc and self.browser_proc.poll() is None:
            try:
                if os.name == "nt":
                    self.browser_proc.terminate()
                    self.browser_proc.wait(timeout=2.0)
                else:
                    os.killpg(self.browser_proc.pid, signal.SIGTERM)
                    self.browser_proc.wait(timeout=2.0)
            except Exception:
                try:
                    if os.name == "nt":
                        self.browser_proc.kill()
                    else:
                        os.killpg(self.browser_proc.pid, signal.SIGKILL)
                except Exception:
                    pass
        self.browser_proc = None

        if self.browser_profile_dir:
            shutil.rmtree(self.browser_profile_dir, ignore_errors=True)
            self.browser_profile_dir = None
