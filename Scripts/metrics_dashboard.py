#!/usr/bin/env python3
# ==================================================================================================================
# ||  TEMPEST AI v2 • LIVE DASHBOARD                                                                             ||
# ||  Lightweight Grafana-style metrics dashboard served locally and managed by the Python app lifecycle.         ||
# ==================================================================================================================
"""Live dashboard for Tempest AI metrics."""

if __name__ == "__main__":
    print("This module is launched from main.py")
    raise SystemExit(1)

import atexit
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
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse

try:
    from config import RL_CONFIG
except ImportError:
    from Scripts.config import RL_CONFIG

try:
    from metrics_display import get_dqn_window_averages, get_total_window_averages, get_eplen_1m_average
except ImportError:
    try:
        from Scripts.metrics_display import get_dqn_window_averages, get_total_window_averages, get_eplen_1m_average
    except ImportError:
        def get_dqn_window_averages():
            return 0.0, 0.0, 0.0
        def get_total_window_averages():
            return 0.0, 0.0, 0.0
        def get_eplen_1m_average():
            return 0.0


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


def _audio_dir() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(script_dir), "audio")


def _fonts_dir() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(script_dir), "fonts")


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

    def _build_snapshot(self) -> dict[str, Any]:
        now = time.time()
        inv_obj = 1.0 / max(1e-9, float(getattr(RL_CONFIG, "obj_reward_scale", 1.0)))
        inv_subj = 1.0 / max(1e-9, float(getattr(RL_CONFIG, "subj_reward_scale", 1.0)))

        with self.metrics.lock:
            frame_count = int(self.metrics.frame_count)
            fps = float(self.metrics.fps)
            epsilon_raw = float(self.metrics.epsilon)
            epsilon_effective = 0.0 if bool(self.metrics.override_epsilon) else epsilon_raw
            expert_ratio = float(self.metrics.expert_ratio)
            client_count = int(self.metrics.client_count)
            web_client_count = int(self.metrics.web_client_count)
            average_level = float(self.metrics.average_level + 1.0)
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

            reward_total = _tail_mean(self.metrics.episode_rewards) * inv_obj
            reward_dqn = _tail_mean(self.metrics.dqn_rewards) * inv_obj
            reward_subj = _tail_mean(self.metrics.subj_rewards) * inv_subj
            reward_obj = _tail_mean(self.metrics.obj_rewards) * inv_obj

        try:
            dqn100k_raw, dqn1m_raw, dqn5m_raw = get_dqn_window_averages()
        except Exception:
            dqn100k_raw = dqn1m_raw = dqn5m_raw = 0.0
        try:
            total100k_raw, total1m_raw, total5m_raw = get_total_window_averages()
        except Exception:
            total100k_raw = total1m_raw = total5m_raw = 0.0
        level_25k, level_100k, level_1m, level_5m = self._update_level_windows(frame_count, average_level)
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

        return {
            "ts": now,
            "frame_count": frame_count,
            "fps": fps,
            "training_steps": total_training_steps,
            "steps_per_sec": steps_per_sec,
            "rpl_per_frame": replay_per_frame,
            "epsilon": epsilon_effective,
            "epsilon_raw": epsilon_raw,
            "expert_ratio": expert_ratio,
            "client_count": client_count,
            "web_client_count": web_client_count,
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
            "dqn_100k": float(dqn100k_raw) * inv_obj,
            "dqn_1m": float(dqn1m_raw) * inv_obj,
            "dqn_5m": float(dqn5m_raw) * inv_obj,
            "total_1m": float(total1m_raw) * inv_obj,
            "total_5m": float(total5m_raw) * inv_obj,
            "level_25k": float(level_25k),
            "level_100k": float(level_100k),
            "level_1m": float(level_1m),
            "level_5m": float(level_5m),
            "training_enabled": training_enabled,
            "override_expert": override_expert,
            "override_epsilon": override_epsilon,
            "lr": lr,
            "q_min": q_min,
            "q_max": q_max,
            "eplen_1m": get_eplen_1m_average(),
        }

    def sample(self):
        with self.lock:
            self._update_web_client_count_locked()
        snap = self._build_snapshot()
        with self.lock:
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


def _render_dashboard_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Tempest AI Metrics</title>
  <style>
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
    html, body { margin: 0; padding: 0; color: var(--ink); background: var(--bg0); }
    body {
      font-family: "Avenir Next", "Segoe UI", "Helvetica Neue", sans-serif;
      min-height: 100vh;
      position: relative;
      isolation: isolate;
      overflow-x: hidden;
      background:
        radial-gradient(1300px 650px at 6% -8%, rgba(0, 229, 255, 0.24), transparent 58%),
        radial-gradient(950px 540px at 102% -4%, rgba(255, 43, 214, 0.22), transparent 56%),
        radial-gradient(900px 500px at 52% 112%, rgba(57, 255, 20, 0.12), transparent 62%),
        repeating-linear-gradient(0deg, rgba(130, 168, 224, 0.03) 0px, rgba(130, 168, 224, 0.03) 1px, transparent 1px, transparent 4px),
        linear-gradient(158deg, var(--bg0) 0%, var(--bg1) 58%, var(--bg2) 100%);
      background-attachment: fixed;
    }
    body::before {
      content: "";
      position: fixed;
      inset: -35vh -20vw;
      pointer-events: none;
      z-index: 0;
      background:
        radial-gradient(circle at 18% 24%, rgba(0, 229, 255, 0.24), transparent 30%),
        radial-gradient(circle at 78% 18%, rgba(255, 43, 214, 0.22), transparent 34%),
        radial-gradient(circle at 68% 78%, rgba(57, 255, 20, 0.18), transparent 36%);
      animation: orbDrift 20s ease-in-out infinite alternate;
    }
    body::after {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      z-index: 1;
      background: linear-gradient(90deg, rgba(0, 229, 255, 0.045), transparent 36%, transparent 64%, rgba(255, 43, 214, 0.045));
      mix-blend-mode: screen;
      animation: rgbSweep 14s linear infinite;
    }
    @keyframes orbDrift {
      0% { transform: translate3d(-2.5%, -2%, 0) scale(1.0); }
      50% { transform: translate3d(2%, 1.5%, 0) scale(1.08); }
      100% { transform: translate3d(3%, -1.5%, 0) scale(1.03); }
    }
    @keyframes rgbSweep {
      0% { opacity: 0.15; transform: translateX(-5%); }
      50% { opacity: 0.42; transform: translateX(5%); }
      100% { opacity: 0.15; transform: translateX(-5%); }
    }
    @keyframes borderShift {
      0% { background-position: 0% 50%; }
      50% { background-position: 100% 50%; }
      100% { background-position: 0% 50%; }
    }
    @keyframes ledPulse {
      0% { transform: scale(0.95); filter: saturate(1.0); }
      50% { transform: scale(1.10); filter: saturate(1.5); }
      100% { transform: scale(0.95); filter: saturate(1.0); }
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
    .top, .card, .panel {
      position: relative;
      overflow: hidden;
      background:
        radial-gradient(120% 160% at 0% 0%, rgba(0, 229, 255, 0.10), transparent 58%),
        radial-gradient(140% 150% at 100% 0%, rgba(255, 43, 214, 0.10), transparent 58%),
        linear-gradient(155deg, rgba(7, 12, 30, 0.86) 0%, rgba(7, 10, 27, 0.74) 100%);
      border: 1px solid var(--line);
      box-shadow:
        inset 0 0 0 1px rgba(0, 229, 255, 0.06),
        0 0 26px var(--panelGlowA),
        0 0 36px var(--panelGlowB);
      backdrop-filter: blur(10px) saturate(118%);
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
      animation: borderShift 7s linear infinite;
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
      background: rgba(2, 6, 23, 0.50);
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
      background: rgba(2, 6, 23, 0.50);
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
      background: rgba(2, 6, 23, 0.50);
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
    .dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: var(--accentC);
      box-shadow: 0 0 0 7px rgba(57, 255, 20, 0.2), 0 0 14px rgba(57, 255, 20, 0.5);
      animation: ledPulse 1.8s ease-in-out infinite;
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
    .value-inline {
      display: inline-flex;
      align-items: center;
      gap: 10px;
    }
    #mQ {
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
    }
    #mQ .qch {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 0.70em;
      min-width: 0.70em;
    }
    .mini-metric-card .mini-inline {
      display: grid;
      grid-template-columns: minmax(0, 23fr) minmax(0, 37fr);
      align-items: stretch;
      column-gap: 8px;
      min-height: 18px;
    }
    .mini-metric-card .mini-inline .value {
      justify-self: start;
      min-width: 0;
      white-space: nowrap;
    }
    .mini-metric-card .mini-canvas {
      width: 100%;
      max-width: 100%;
      height: 104px;
      border-radius: 8px;
      border: none;
      background:
        linear-gradient(180deg, rgba(2, 6, 23, 0.18), rgba(2, 6, 23, 0.30)),
        repeating-linear-gradient(0deg, rgba(120, 150, 210, 0.035) 0px, rgba(120, 150, 210, 0.035) 1px, transparent 1px, transparent 4px);
      box-shadow: inset 0 0 14px rgba(0, 229, 255, 0.10), 0 0 12px rgba(0, 229, 255, 0.09);
      position: relative;
      z-index: 2;
      flex: 0 0 auto;
      justify-self: end;
    }
    .metric-led {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: var(--neonRed);
      box-shadow: 0 0 0 4px rgba(255, 42, 85, 0.24), 0 0 12px rgba(255, 42, 85, 0.58);
      flex: 0 0 auto;
      animation: ledPulse 1.4s ease-in-out infinite;
    }
    .gauge-card {
      grid-column: span 2;
      grid-row: span 2;
      min-height: 200px;
      max-height: 320px;
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
      flex: 1;
    }
    .card-half.mini-metric-card .mini-canvas {
      height: 36px;
      min-height: 36px;
      flex: 1;
      border-radius: 4px;
    }
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
    .gauge-readout {
      display: none;
    }
    .gauge-canvas-wrap,
    .gauge-foot {
      display: none;
    }
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
      grid-template-rows: auto auto 1fr;
      gap: 8px;
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
        linear-gradient(180deg, rgba(2, 6, 23, 0.28), rgba(2, 6, 23, 0.36)),
        repeating-linear-gradient(0deg, rgba(120, 150, 210, 0.04) 0px, rgba(120, 150, 210, 0.04) 1px, transparent 1px, transparent 5px);
      box-shadow: inset 0 0 28px rgba(0, 229, 255, 0.10), 0 0 20px rgba(0, 229, 255, 0.12);
      position: relative;
      z-index: 2;
    }
    /* Low-GPU mode: disable continuous compositing-heavy effects. */
    body::before,
    body::after,
    .top::before,
    .card::before,
    .panel::before,
    .dot,
    .metric-led {
      animation: none !important;
    }
    .top, .card, .panel {
      backdrop-filter: none;
    }
    body::after {
      mix-blend-mode: normal;
      opacity: 0.10;
    }
    @media (max-width: 1300px) {
      .cards { grid-template-columns: repeat(8, minmax(0, 1fr)); }
      .gauge-card { grid-column: span 2; grid-row: span 2; min-height: 180px; }
    }
    @media (max-width: 950px) {
      .cards { grid-template-columns: repeat(4, minmax(0, 1fr)); }
      .charts { grid-template-columns: 1fr; }
      .top { flex-direction: column; align-items: flex-start; }
      .gauge-card { grid-column: span 2; }
      .mini-metric-card .mini-canvas { height: 96px; }
    }
  </style>
</head>
<body>
  <main>
    <section class="cards">
      <article class="card gauge-card" style="--card-border:rgba(255,60,60,0.55);--card-glow:rgba(255,40,40,0.22)">
        <div class="gauge-head">
          <div class="label">FRAMES PER SECOND</div>
        </div>
        <canvas id="cFpsGauge"></canvas>
      </article>
      <article class="card gauge-card" style="--card-border:rgba(255,220,40,0.55);--card-glow:rgba(255,200,20,0.22)">
        <div class="gauge-head">
          <div class="label">STEPS PER SECOND</div>
        </div>
        <canvas id="cStepGauge"></canvas>
      </article>
      <article class="card mini-metric-card" style="--card-border:rgba(255,140,30,0.55);--card-glow:rgba(255,120,20,0.22)">
        <div class="label">DQN REWARD 1M</div>
        <div class="mini-inline">
          <div class="value" id="mDqnRwrd">0</div>
          <canvas id="cDqnRewardMini" class="mini-canvas"></canvas>
        </div>
      </article>
      <article class="card mini-metric-card" style="--card-border:rgba(50,220,80,0.55);--card-glow:rgba(40,200,60,0.22)">
        <div class="label">AVG REWARD 1M</div>
        <div class="mini-inline">
          <div class="value" id="mRwrd">0</div>
          <canvas id="cRewardMini" class="mini-canvas"></canvas>
        </div>
      </article>
      <article class="card mini-metric-card" style="--card-border:rgba(60,130,255,0.55);--card-glow:rgba(40,100,255,0.22)">
        <div class="label">Avg Level</div>
        <div class="mini-inline">
          <div class="value" id="mLevel">0.0</div>
          <canvas id="cLevelMini" class="mini-canvas"></canvas>
        </div>
      </article>
      <article class="card mini-metric-card" style="--card-border:rgba(180,80,255,0.55);--card-glow:rgba(160,50,255,0.22)">
        <div class="label">Loss</div>
        <div class="mini-inline">
          <div class="value" id="mLoss">0</div>
          <canvas id="cLossMini" class="mini-canvas"></canvas>
        </div>
      </article>
      <article class="card mini-metric-card card-half" style="--card-border:rgba(255,60,180,0.55);--card-glow:rgba(255,40,160,0.22)">
        <div class="label">Grad Norm</div>
        <div class="mini-inline">
          <div class="value" id="mGrad">0</div>
          <canvas id="cGradMini" class="mini-canvas"></canvas>
        </div>
      </article>
      <article class="card mini-metric-card card-half" style="--card-border:rgba(255,160,80,0.55);--card-glow:rgba(255,140,60,0.22)">
        <div class="label">EPISODE LENGTH</div>
        <div class="mini-inline">
          <div class="value" id="mEpLen">0</div>
          <canvas id="cEpLenMini" class="mini-canvas"></canvas>
        </div>
      </article>
      <article class="card card-half card-narrow" style="--card-border:rgba(120,220,60,0.55);--card-glow:rgba(100,200,40,0.22)"><div class="label">Clnt</div><div class="value" id="mClients">0</div></article>
      <article class="card card-half card-narrow" style="--card-border:rgba(255,180,60,0.55);--card-glow:rgba(255,160,40,0.22)"><div class="label">Web</div><div class="value" id="mWeb">0</div></article>
      <article class="card card-half card-narrow" style="--card-border:rgba(255,100,100,0.55);--card-glow:rgba(255,80,80,0.22)"><div class="label">Epsilon</div><div class="value" id="mEps">0%</div></article>
      <article class="card card-half card-narrow" style="--card-border:rgba(80,255,180,0.55);--card-glow:rgba(60,235,160,0.22)"><div class="label">Expert</div><div class="value" id="mXprt">0%</div></article>
      <article class="card" style="--card-border:rgba(100,200,255,0.55);--card-glow:rgba(80,180,255,0.22)">
        <div class="label">AVG INFERENCE</div>
        <div class="value value-inline"><span class="metric-led" id="mInfLed"></span><span id="mInf">0.00ms</span></div>
      </article>
      <article class="card" style="--card-border:rgba(220,180,255,0.55);--card-glow:rgba(200,150,255,0.22)">
        <div class="label">REPLAYS PER FRAME</div>
        <div class="value value-inline"><span class="metric-led" id="mRplLed"></span><span id="mRplF">0.00</span></div>
      </article>
      <article class="card" style="--card-border:rgba(255,220,100,0.55);--card-glow:rgba(255,200,80,0.22)"><div class="label">BUFFER SIZE</div><div class="value" id="mBuf">0k (0%)</div></article>
      <article class="card" style="--card-border:rgba(100,160,255,0.55);--card-glow:rgba(80,140,255,0.22)"><div class="label">LEARNING RATE</div><div class="value" id="mLr">-</div></article>
      <article class="card" style="--card-border:rgba(200,100,255,0.55);--card-glow:rgba(180,80,255,0.22)"><div class="label">Q Range</div><div class="value" id="mQ">-</div></article>
      <article class="card card-half" style="--card-border:rgba(0,220,200,0.55);--card-glow:rgba(0,200,180,0.22)"><div class="label">Frame</div><div class="value" id="mFrame">0</div></article>
    </section>

    <section class="charts">
      <article class="panel">
        <h2>Throughput</h2>
        <div class="legend">
          <span><span class="sw" style="background:#22c55e;"></span>FPS</span>
          <span><span class="sw" style="background:#f59e0b;"></span>Steps/Sec</span>
          <span><span class="sw" style="background:#22d3ee;"></span>Avg Lvl (100K)</span>
        </div>
        <canvas id="cThroughput"></canvas>
      </article>

      <article class="panel">
        <h2>Rewards</h2>
        <div class="legend">
          <span><span class="sw" style="background:#22c55e;"></span>Total</span>
          <span><span class="sw" style="background:#f59e0b;"></span>DQN</span>
          <span><span class="sw" style="background:#22d3ee;"></span>Objective</span>
          <span><span class="sw" style="background:#f43f5e;"></span>Subjective</span>
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

      <article class="panel">
        <h2>DQN Rolling</h2>
        <div class="legend">
          <span><span class="sw" style="background:#22c55e;"></span>DQN Inst</span>
          <span><span class="sw" style="background:#f59e0b;"></span>DQN100K</span>
          <span><span class="sw" style="background:#22d3ee;"></span>DQN1M</span>
          <span><span class="sw" style="background:#f43f5e;"></span>DQN5M</span>
        </div>
        <canvas id="cDqn"></canvas>
      </article>

    </section>
    <section class="top">
      <div class="title">
        <h1>Tempest AI Dashboard</h1>
        <div class="subtitle">Primary training and runtime telemetry (live)</div>
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
    const num = new Intl.NumberFormat("en-US");
    const DASH_MAX_FPS = 30;
    const DASH_DEFAULT_FPS = 2;
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
    const GAUGE_MAX_FPS = 5000;
    const GAUGE_FPS_RED_MAX = 1000;
    const GAUGE_FPS_YELLOW_MAX = 1500;
    const GAUGE_MIN_STEPS = 0;
    const GAUGE_MAX_STEPS = 50;
    const AUDIO_PREF_COOKIE = "tempest_dashboard_audio_enabled";
    const AUDIO_START_RETRY_MS = 800;

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
    const bgAudio = document.getElementById("bgAudio");
    const audioToggle = document.getElementById("audioToggle");
    let audioPlaylist = [];
    let audioIndex = 0;
    let audioEnabled = false;
    let audioRetryTimer = null;

    const cards = {
      frame: document.getElementById("mFrame"),
      fps: document.getElementById("mFps"),
      steps: document.getElementById("mSteps"),
      clients: document.getElementById("mClients"),
      web: document.getElementById("mWeb"),
      level: document.getElementById("mLevel"),
      inf: document.getElementById("mInf"),
      infLed: document.getElementById("mInfLed"),
      rplf: document.getElementById("mRplF"),
      rplLed: document.getElementById("mRplLed"),
      eps: document.getElementById("mEps"),
      xprt: document.getElementById("mXprt"),
      rwrd: document.getElementById("mRwrd"),
      dqnRwrd: document.getElementById("mDqnRwrd"),
      loss: document.getElementById("mLoss"),
      grad: document.getElementById("mGrad"),
      buf: document.getElementById("mBuf"),
      lr: document.getElementById("mLr"),
      q: document.getElementById("mQ"),
      epLen: document.getElementById("mEpLen"),
    };
    const fpsGaugeCanvas = document.getElementById("cFpsGauge");
    const stepGaugeCanvas = document.getElementById("cStepGauge");

    // ── Gauge needle damping ────────────────────────────────────────
    // Time-constant in seconds: the needle closes ~63% of the gap
    // in this many seconds.  Lower = snappier, higher = smoother.
    const GAUGE_DAMPING_TAU = 0.35;
    const gaugeState = {
      fps:  { current: 0, target: 0 },
      step: { current: 0, target: 0 },
    };
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
        drawFpsGauge(fpsGaugeCanvas, gaugeState.fps.current);
        drawStepGauge(stepGaugeCanvas, gaugeState.step.current);
      }

      requestAnimationFrame(gaugeAnimationLoop);
    }
    requestAnimationFrame(gaugeAnimationLoop);

    const charts = {
      throughput: {
        canvas: document.getElementById("cThroughput"),
        series: [
          {
            key: "fps",
            color: "#22c55e",
            axis: { side: "left", min: 0 },
            smooth_alpha: 0.14,
          },
          {
            key: "steps_per_sec_chart",
            color: "#f59e0b",
            axis: { side: "right", min: 0, max: 50, ticks: [0, 10, 20, 30, 40, 50] },
            smooth_alpha: 0.10,
          },
          {
            key: "level_100k",
            color: "#22d3ee",
            axis_ref: "steps_per_sec_chart",
            smooth_alpha: 0.20,
          }
        ]
      },
      rewards: {
        canvas: document.getElementById("cRewards"),
        series: [
          {
            key: "reward_total",
            color: "#22c55e",
            axis: {
              side: "left",
              min: 0,
              label_pad: 52,
              round_max: 1000,
              group_keys: ["reward_total", "reward_dqn", "reward_obj", "reward_subj"],
            }
          },
          { key: "reward_dqn", color: "#f59e0b", axis_ref: "reward_total" },
          { key: "reward_obj", color: "#22d3ee", axis_ref: "reward_total" },
          { key: "reward_subj", color: "#f43f5e", axis_ref: "reward_total" }
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
      dqn: {
        canvas: document.getElementById("cDqn"),
        series: [
          { key: "dqn_100k", color: "#f59e0b", axis_ref: "reward_dqn" },
          { key: "dqn_1m", color: "#22d3ee", axis_ref: "reward_dqn" },
          { key: "dqn_5m", color: "#f43f5e", axis_ref: "reward_dqn" },
          {
            key: "reward_dqn",
            color: "#22c55e",
            axis: { side: "left", min: 0, label_pad: 52, group_keys: ["dqn_100k", "dqn_1m", "dqn_5m", "reward_dqn"] },
          }
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
          { key: "total_5m", color: "#3b82f6" },
          { key: "total_1m", color: "#22c55e" }
        ]
      },
      dqnRewardMini: {
        canvas: document.getElementById("cDqnRewardMini"),
        series: [
          { key: "dqn_5m", color: "#3b82f6" },
          { key: "dqn_1m", color: "#f59e0b" }
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
      return (n < 0 ? "-" : "+") + mag;
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

      const pad = 2;
      const outerExtent = 1.08;   // tight fit — faint glow may clip, bezel stays
      const downExtent  = 0.78;   // badge hangs below center
      // Size the radius so the dial fits snugly inside the canvas
      const maxRByWidth  = (width  - 2 * pad) / (2.0 * outerExtent);
      const maxRByHeight = (height - 2 * pad) / (outerExtent + downExtent);
      const radius = Math.max(20, Math.min(maxRByWidth, maxRByHeight));
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

      // Outer glow + bezel (metallic ring style). [TEMPORARILY HIDDEN]
      /*
      const glow = ctx.createRadialGradient(cx, cy, radius * 0.9, cx, cy, radius * 1.22);
      glow.addColorStop(0.0, "rgba(0, 0, 0, 0.00)");
      glow.addColorStop(1.0, "rgba(0, 0, 0, 0.55)");
      ctx.fillStyle = glow;
      ctx.beginPath();
      ctx.arc(cx, cy, radius * 1.22, 0, Math.PI * 2);
      ctx.fill();

      ctx.lineWidth = Math.max(10, radius * 0.14);
      const bezel = ctx.createLinearGradient(cx - radius, cy - radius, cx + radius, cy + radius);
      bezel.addColorStop(0.0, "#8f979f");
      bezel.addColorStop(0.20, "#31373d");
      bezel.addColorStop(0.55, "#14181d");
      bezel.addColorStop(0.80, "#6f7881");
      bezel.addColorStop(1.0, "#d8dde1");
      ctx.strokeStyle = bezel;
      ctx.beginPath();
      ctx.arc(cx, cy, radius * 1.15, 0, Math.PI * 2);
      ctx.stroke();
      */

      // Dial face.
      const face = ctx.createRadialGradient(cx, cy - radius * 0.5, radius * 0.2, cx, cy, radius * 1.02);
      face.addColorStop(0.0, "rgba(62, 67, 73, 0.96)");
      face.addColorStop(0.55, "rgba(39, 42, 47, 0.98)");
      face.addColorStop(1.0, "rgba(22, 25, 30, 0.99)");
      ctx.fillStyle = face;
      ctx.beginPath();
      ctx.arc(cx, cy, radius * 1.02, 0, Math.PI * 2);
      ctx.fill();

      // Dial ring.
      ctx.strokeStyle = "rgba(210, 215, 221, 0.72)";
      ctx.lineWidth = Math.max(3.5, radius * 0.040);
      ctx.beginPath();
      ctx.arc(cx, cy, radius * 0.92, startRad, endRad, false);
      ctx.stroke();

      // Scale ticks with threshold coloring.
      const minorStep = Math.max(1e-9, Number(cfg.minor_step));
      const majorStep = Math.max(minorStep, Number(cfg.major_step));
      const majorEvery = Math.max(1, Math.round(majorStep / minorStep));
      const tickOuter = radius * 0.89;
      const tickMinorInner = radius * 0.76;
      const tickMajorInner = radius * 0.66;
      const tickCount = Math.max(1, Math.round((maxV - minV) / minorStep));
      for (let i = 0; i <= tickCount; i++) {
        const vv = (i >= tickCount) ? maxV : (minV + (i * minorStep));
        const a = valToAngle(vv);
        const cosA = Math.cos(a);
        const sinA = Math.sin(a);
        const isMajor = ((i % majorEvery) === 0) || i === tickCount;
        const inner = isMajor ? tickMajorInner : tickMinorInner;
        let c = "rgba(236, 241, 247, 0.90)";
        if (isMajor) {
          if (vv <= cfg.red_max) c = "rgba(255, 58, 58, 0.95)";
          else if (vv <= cfg.yellow_max) c = "rgba(252, 176, 44, 0.95)";
          else c = "rgba(51, 219, 107, 0.95)";
        }
        ctx.strokeStyle = c;
        ctx.lineWidth = isMajor ? Math.max(3.0, radius * 0.03) : Math.max(0.9, radius * 0.008);
        ctx.beginPath();
        ctx.moveTo(cx + (tickOuter * cosA), cy + (tickOuter * sinA));
        ctx.lineTo(cx + (inner * cosA), cy + (inner * sinA));
        ctx.stroke();
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
        ctx.fillStyle = "rgba(0, 0, 0, 0.42)";
        ctx.beginPath();
        ctx.moveTo(pTipX + 2.5, pTipY + 2.5);
        ctx.lineTo(pTailX + (perpX * baseHalfW) + 2.5, pTailY + (perpY * baseHalfW) + 2.5);
        ctx.lineTo(pTailX - (perpX * baseHalfW) + 2.5, pTailY - (perpY * baseHalfW) + 2.5);
        ctx.closePath();
        ctx.fill();

        const needleGrad = ctx.createLinearGradient(pTailX, pTailY, pTipX, pTipY);
        needleGrad.addColorStop(0.0, "#c85a00");
        needleGrad.addColorStop(0.6, "#ff8a00");
        needleGrad.addColorStop(1.0, "#ffc04d");
        ctx.fillStyle = needleGrad;
        ctx.beginPath();
        ctx.moveTo(pTipX, pTipY);
        ctx.lineTo(pTailX + (perpX * baseHalfW), pTailY + (perpY * baseHalfW));
        ctx.lineTo(pTailX - (perpX * baseHalfW), pTailY - (perpY * baseHalfW));
        ctx.closePath();
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
      const badgeW = radius * 0.86;
      const badgeH = radius * 0.48;
      const badgeX = cx - (badgeW * 0.5);
      const badgeY = cy + radius * 0.30;
      const badgeFill = ctx.createLinearGradient(0, badgeY, 0, badgeY + badgeH);
      badgeFill.addColorStop(0.0, "rgba(16, 18, 22, 0.92)");
      badgeFill.addColorStop(1.0, "rgba(8, 10, 13, 0.96)");
      roundRectPath(ctx, badgeX, badgeY, badgeW, badgeH, Math.max(8, radius * 0.08));
      ctx.fillStyle = badgeFill;
      ctx.fill();

      const valueText = Number(value).toFixed(cfg.decimals ?? 1);
      ctx.fillStyle = "rgba(255, 52, 52, 0.98)";
      ctx.shadowColor = "rgba(255, 52, 52, 0.45)";
      ctx.shadowBlur = Math.max(4, radius * 0.08);
      ctx.font = `400 ${Math.max(20, Math.round(radius * 0.44))}px 'DS-Digital', 'LED Dot-Matrix', 'Dot Matrix', 'DotGothic16', 'Courier New', monospace`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(valueText, cx, badgeY + (badgeH * 0.58));
      ctx.shadowBlur = 0;
      drawNeedle();
      drawHub();
    }

    function drawFpsGauge(canvas, fps) {
      drawStyledGauge(canvas, fps, {
        min: GAUGE_MIN_FPS,
        max: GAUGE_MAX_FPS,
        red_max: GAUGE_FPS_RED_MAX,
        yellow_max: GAUGE_FPS_YELLOW_MAX,
        minor_step: 50,
        major_step: 250,
        title: "FPS",
        unit: "FPS",
        decimals: 0,
      });
    }

    function drawStepGauge(canvas, stepsPerSec) {
      drawStyledGauge(canvas, stepsPerSec, {
        min: GAUGE_MIN_STEPS,
        max: GAUGE_MAX_STEPS,
        red_max: 10,
        yellow_max: 20,
        minor_step: 1,
        major_step: 10,
        title: "STEP",
        unit: "S/S",
        decimals: 0,
      });
    }

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
        : axisSlotDefault;
      const padL = 26 + leftAxisPad;
      const padR = 26 + rightAxisPad;
      const padT = 10, padB = 30;
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
        const roundMax = Number(s.axis?.round_max);
        if (!hasFixedMax && roundMax > 0 && Number.isFinite(roundMax)) {
          maxV = Math.ceil(maxV / roundMax) * roundMax;
        }
        if (maxV < minV) {
          maxV = minV + 1.0;
        }
        if (minV === maxV) {
          if (hasFixedMin && !hasFixedMax) {
            maxV = minV + 1.0;
          } else if (!hasFixedMin && hasFixedMax) {
            minV = maxV - 1.0;
          } else {
            minV -= 1.0;
            maxV += 1.0;
          }
        } else {
          const p = (maxV - minV) * 0.12;
          if (!hasFixedMin) {
            minV -= p;
          }
          if (!hasFixedMax) {
            maxV += p;
          }
        }

        const axisX = side === "left"
          ? (padL - 20 - leftAxisOffset)
          : (width - padR + 20 + rightAxisOffset);
        if (side === "left") {
          leftAxisOffset += axisSlot;
        } else {
          rightAxisOffset += axisSlot;
        }
        const ticks = Array.isArray(s.axis?.ticks) && s.axis.ticks.length
          ? s.axis.ticks
          : [minV, minV + (maxV - minV) * 0.25, minV + (maxV - minV) * 0.5, minV + (maxV - minV) * 0.75, maxV];

        axes.push({
          key: s.key,
          side,
          x: axisX,
          color: s.color,
          min: minV,
          max: maxV,
          ticks,
          tickDecimals: Number.isFinite(s.axis?.tick_decimals) ? Math.max(0, Number(s.axis.tick_decimals)) : null,
        });
      }
      if (!axes.length) return;

      const axisByKey = new Map(axes.map((a) => [a.key, a]));
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
      for (const s of seriesDefs) {
        const axis = s.axis_ref
          ? axisByKey.get(s.axis_ref)
          : (axisByKey.get(s.key) || axes[0]);
        if (!axis) continue;
        const smoothAlpha = Number.isFinite(s.smooth_alpha) ? Number(s.smooth_alpha) : CHART_VALUE_SMOOTH_ALPHA;
        ctx.strokeStyle = s.color;
        ctx.lineWidth = 2.0;
        ctx.beginPath();
        let started = false;
        let smoothVal = null;
        if (s.pixel_bin_avg) {
          const bins = new Map();
          for (let i = n - 1; i >= 0; i--) {
            const val = seriesValue(points[i], s.key);
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
            const y = yAt(axis, smoothVal);
            if (!started) {
              ctx.moveTo(xPx, y);
              started = true;
            } else {
              ctx.lineTo(xPx, y);
            }
          }
        } else {
          for (let i = n - 1; i >= 0; i--) {
            const val = seriesValue(points[i], s.key);
            if (!Number.isFinite(val)) continue;
            smoothVal = (smoothVal === null)
              ? Number(val)
              : (smoothVal + ((Number(val) - smoothVal) * smoothAlpha));
            const x = xAt(i);
            const y = yAt(axis, smoothVal);
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

    function drawMiniChart(canvas, history, seriesDefs) {
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

      const axisPad = 20;
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
      let minV = hasFixedMin ? Number(seriesDefs[0].axis.min) : Math.min(...values);
      let maxV = hasFixedMax ? Number(seriesDefs[0].axis.max) : Math.max(...values);
      if (maxV <= minV) {
        maxV = minV + 1.0;
      } else {
        const p = (maxV - minV) * 0.08;
        if (!hasFixedMin) minV -= p;
        if (!hasFixedMax) maxV += p;
      }

      const xAt = (i) => {
        const t = points.length <= 1 ? 1.0 : (i / (points.length - 1));
        return padL + (t * plotW);
      };
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
      const yTicks = [minV, minV + ((maxV - minV) * 0.5), maxV];
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
      for (const s of seriesDefs) {
        const smoothAlpha = Number.isFinite(s.smooth_alpha) ? Number(s.smooth_alpha) : MINI_CHART_VALUE_SMOOTH_ALPHA;
        ctx.strokeStyle = s.color;
        ctx.globalAlpha = (s.key === "level_1m") ? 0.95 : 0.82;
        ctx.lineWidth = (s.key === "level_1m") ? 2.0 : 1.6;
        ctx.beginPath();
        let started = false;
        let smoothVal = null;
        if (s.pixel_bin_avg) {
          const bins = new Map();
          for (let i = n - 1; i >= 0; i--) {
            const val = Number(points[i][s.key]);
            if (!Number.isFinite(val)) continue;
            const x = xAt(i);
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
            const y = yAt(smoothVal);
            if (!started) {
              ctx.moveTo(xPx, y);
              started = true;
            } else {
              ctx.lineTo(xPx, y);
            }
          }
        } else {
          for (let i = n - 1; i >= 0; i--) {
            const val = Number(points[i][s.key]);
            if (!Number.isFinite(val)) continue;
            smoothVal = (smoothVal === null)
              ? val
              : (smoothVal + ((val - smoothVal) * smoothAlpha));
            const x = xAt(i);
            const y = yAt(smoothVal);
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
        ema = (ema === null) ? rate : (ema + ((rate - ema) * emaAlpha));
        out[i] = { ...row, steps_per_sec_chart: ema };
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

    function updateCards(now, smoothedSteps) {
      cards.frame.textContent = fmtInt(now.frame_count);
      if (cards.fps) cards.fps.textContent = fmtInt(now.fps);
      if (cards.steps) cards.steps.textContent = fmtInt(smoothedSteps);
      cards.clients.textContent = fmtInt(now.client_count);
      cards.web.textContent = fmtInt(now.web_client_count);
      cards.level.textContent = fmtFloat(now.average_level, 2);
      cards.inf.textContent = `${fmtFloat(now.avg_inf_ms, 2)} ms`;
      setInfLed(now.avg_inf_ms);
      cards.rplf.textContent = fmtFloat(now.rpl_per_frame, 2);
      setRplLed(now.rpl_per_frame);
      cards.eps.textContent = fmtPct(now.epsilon);
      cards.xprt.textContent = fmtPct(now.expert_ratio);
      cards.rwrd.textContent = fmtInt(now.total_1m);
      cards.dqnRwrd.textContent = fmtInt(now.dqn_1m);
      cards.loss.textContent = fmtFloat(now.loss, 2);
      cards.grad.textContent = fmtFloat(now.grad_norm, 3);
      cards.buf.textContent = fmtInt(now.memory_buffer_size);
      cards.lr.textContent = (now.lr === null || now.lr === undefined) ? "-" : Number(now.lr).toExponential(1);
      cards.q.innerHTML = (now.q_min === null || now.q_max === null)
        ? toFixedCharCells("-")
        : toFixedCharCells(`${fmtSignedFloat(now.q_min, 1)},${fmtSignedFloat(now.q_max, 1)}`);
      cards.epLen.textContent = fmtInt(now.eplen_1m);
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
      gaugeState.fps.target  = payload.now.fps;
      gaugeState.step.target = smoothedStepSpd;
      drawChart(charts.throughput.canvas, throughputHistory, charts.throughput.series, 60 * 60);
      drawChart(charts.rewards.canvas, chartHistory60m, charts.rewards.series, 60 * 60);
      drawChart(charts.learning.canvas, chartHistory1m, charts.learning.series, 60, true);
      drawChart(charts.dqn.canvas, chartHistory60m, charts.dqn.series, 60 * 60);
      drawMiniChart(charts.dqnRewardMini.canvas, history60m, charts.dqnRewardMini.series);
      drawMiniChart(charts.rewardMini.canvas, history60m, charts.rewardMini.series);
      drawMiniChart(charts.level1m.canvas, history60m, charts.level1m.series);
      drawMiniChart(charts.lossMini.canvas, history2m, charts.lossMini.series);
      drawMiniChart(charts.gradMini.canvas, history2m, charts.gradMini.series);
      drawMiniChart(charts.epLenMini.canvas, history60m, charts.epLenMini.series);
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

    async function fetchNow() {
      try {
        const res = await fetch(`/api/now?cid=${encodeURIComponent(CLIENT_ID)}&t=${Date.now()}`, { cache: "no-store" });
        if (!res.ok) throw new Error("bad response");
        const now = await res.json();
        const hadNow = !!latestNow;
        latestNow = now;
        const ts = Number(now && now.ts);
        let hasNewSample = false;
        if (Number.isFinite(ts) && ts > lastTs + 1e-9) {
          historyCache.push(now);
          if (historyCache.length > MAX_HISTORY_POINTS) {
            historyCache.shift();
          }
          lastTs = ts;
          hasNewSample = true;
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

    const cookiePref = getCookieValue(AUDIO_PREF_COOKIE);
    audioEnabled = (cookiePref === null) ? true : (cookiePref === "1");
    setAudioToggle(audioEnabled, false);
    const kickAudioStart = () => {
      if (audioEnabled) ensureAudioPlaying();
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

    fetchHistory().then(() => fetchNow()).catch(() => {});
    setInterval(fetchNow, DASH_REFRESH_MS);
    setInterval(heartbeat, 1000);
    window.addEventListener("resize", () => renderCurrent());
  </script>
</body>
</html>
"""


def _make_handler(state: _DashboardState):
    page = _render_dashboard_html().encode("utf-8")
    audio_root = os.path.abspath(_audio_dir())
    fonts_root = os.path.abspath(_fonts_dir())

    class DashboardHandler(BaseHTTPRequestHandler):
        def _send(
            self,
            payload: bytes,
            content_type: str = "text/plain",
            status: int = 200,
            cache_control: str = "no-store",
        ):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", cache_control)
            self.end_headers()
            self.wfile.write(payload)

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
            except Exception:
                self._send(b"Not Found", "text/plain; charset=utf-8", status=404)

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

        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path
            query = parse_qs(parsed.query)
            client_id = (query.get("cid") or [None])[0]
            if path in ("/api/ping", "/api/now", "/api/history"):
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
            self._send(b"Not Found", "text/plain; charset=utf-8", status=404)

        def log_message(self, fmt, *args):
            return

    return DashboardHandler


class MetricsDashboard:
    """Managed dashboard server + browser window."""

    def __init__(
        self,
        metrics_obj,
        agent_obj=None,
        host: str = "127.0.0.1",
        port: int = 8765,
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
        handler_cls = _make_handler(self.state)
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
                profile_dir = tempfile.mkdtemp(prefix="tempest_dashboard_")
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
