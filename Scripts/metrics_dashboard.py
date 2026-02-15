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
from urllib.parse import parse_qs, urlparse

try:
    from config import RL_CONFIG
except ImportError:
    from Scripts.config import RL_CONFIG

try:
    from metrics_display import get_dqn_window_averages
except ImportError:
    try:
        from Scripts.metrics_display import get_dqn_window_averages
    except ImportError:
        def get_dqn_window_averages():
            return 0.0, 0.0, 0.0


def _tail_mean(values, count: int = 20) -> float:
    if not values:
        return 0.0
    tail = list(values)[-count:]
    if not tail:
        return 0.0
    return float(sum(tail) / max(1, len(tail)))


LEVEL_25K_FRAMES = 25_000
LEVEL_1M_FRAMES = 1_000_000
LEVEL_5M_FRAMES = 5_000_000
WEB_CLIENT_TIMEOUT_S = 5.0


class _DashboardState:
    def __init__(self, metrics_obj, agent_obj=None, history_limit: int = 900):
        self.metrics = metrics_obj
        self.agent = agent_obj
        self.history = deque(maxlen=max(120, history_limit))
        self.latest: dict[str, Any] = {}
        self.lock = threading.Lock()
        self.last_steps: int | None = None
        self.last_steps_time: float | None = None
        self._level_windows = {
            "25k": {"limit": LEVEL_25K_FRAMES, "samples": deque(), "frames": 0, "weighted": 0.0},
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

    def _update_level_windows(self, frame_count: int, average_level: float) -> tuple[float, float, float]:
        raw_level = float(average_level)
        level = round(raw_level, 4) if math.isfinite(raw_level) else 0.0
        if self._last_level_frame_count is None:
            self._last_level_frame_count = frame_count
            return level, level, level

        if frame_count < self._last_level_frame_count:
            self._clear_level_windows()
            self._last_level_frame_count = frame_count
            return level, level, level

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
        level_25k, level_1m, level_5m = self._update_level_windows(frame_count, average_level)
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
            "epsilon": epsilon_effective,
            "epsilon_raw": epsilon_raw,
            "expert_ratio": expert_ratio,
            "client_count": client_count,
            "web_client_count": web_client_count,
            "average_level": average_level,
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
            "level_25k": float(level_25k),
            "level_1m": float(level_1m),
            "level_5m": float(level_5m),
            "training_enabled": training_enabled,
            "override_expert": override_expert,
            "override_epsilon": override_epsilon,
            "lr": lr,
            "q_min": q_min,
            "q_max": q_max,
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
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=DotGothic16&display=swap" rel="stylesheet">
  <style>
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
    .top::before, .card::before, .panel::before {
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
    .top::after, .card::after, .panel::after {
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
      grid-template-columns: repeat(6, minmax(140px, 1fr));
      gap: 12px;
    }
    .card {
      border-radius: 14px;
      padding: 10px 12px;
      min-height: 86px;
      display: flex;
      flex-direction: column;
      justify-content: center;
      gap: 6px;
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
      font-family: "DotGothic16", "Courier New", monospace;
      color: var(--vfdCyan);
      letter-spacing: 0.95px;
      font-variant-numeric: tabular-nums;
      text-shadow: none;
    }
    .value-inline {
      display: inline-flex;
      align-items: center;
      gap: 10px;
    }
    .avg-level-card .level-inline {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      min-height: 58px;
    }
    .avg-level-card .level-mini-canvas {
      width: 132px;
      height: 58px;
      border-radius: 8px;
      border: 1px solid rgba(0, 229, 255, 0.30);
      background:
        linear-gradient(180deg, rgba(2, 6, 23, 0.18), rgba(2, 6, 23, 0.30)),
        repeating-linear-gradient(0deg, rgba(120, 150, 210, 0.035) 0px, rgba(120, 150, 210, 0.035) 1px, transparent 1px, transparent 4px);
      box-shadow: inset 0 0 14px rgba(0, 229, 255, 0.10), 0 0 12px rgba(0, 229, 255, 0.09);
      position: relative;
      z-index: 2;
      flex: 0 0 auto;
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
      grid-column: span 1;
      grid-row: span 2;
      min-height: 188px;
      padding: 12px;
      justify-content: flex-start;
      gap: 8px;
    }
    .gauge-head {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 8px;
    }
    .gauge-readout {
      display: inline-flex;
      align-items: baseline;
      gap: 6px;
      font-size: 30px;
      line-height: 1;
      font-weight: 700;
      letter-spacing: 0.3px;
      color: #d8f5ff;
      text-shadow: 0 0 14px rgba(0, 229, 255, 0.34), 0 0 30px rgba(57, 255, 20, 0.22);
      position: relative;
      z-index: 2;
    }
    .gauge-readout small {
      color: var(--muted);
      font-size: 14px;
      font-weight: 560;
      letter-spacing: 0.3px;
      text-transform: uppercase;
    }
    .gauge-canvas {
      width: 100%;
      height: 158px;
      border-radius: 12px;
      border: 1px solid rgba(0, 229, 255, 0.30);
      background:
        radial-gradient(circle at 50% 56%, rgba(2, 6, 23, 0.16) 0%, rgba(2, 6, 23, 0.56) 68%, rgba(2, 6, 23, 0.84) 100%),
        repeating-linear-gradient(0deg, rgba(130, 168, 224, 0.045) 0px, rgba(130, 168, 224, 0.045) 1px, transparent 1px, transparent 4px);
      box-shadow: inset 0 0 22px rgba(0, 229, 255, 0.10), 0 0 18px rgba(0, 229, 255, 0.12);
      position: relative;
      z-index: 2;
    }
    .gauge-foot {
      display: flex;
      justify-content: space-between;
      color: var(--muted);
      font-size: 11px;
      letter-spacing: 0.3px;
      text-transform: uppercase;
      position: relative;
      z-index: 2;
    }
    .charts {
      display: grid;
      grid-template-columns: repeat(2, minmax(320px, 1fr));
      gap: 14px;
    }
    .panel {
      border-radius: 14px;
      padding: 12px;
      min-height: 280px;
      display: grid;
      grid-template-rows: auto auto 1fr;
      gap: 8px;
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
      .cards { grid-template-columns: repeat(4, minmax(130px, 1fr)); }
      .gauge-card { grid-column: span 1; grid-row: span 2; min-height: 188px; }
      .avg-level-card .level-mini-canvas { width: 110px; }
    }
    @media (max-width: 950px) {
      .cards { grid-template-columns: repeat(2, minmax(130px, 1fr)); }
      .charts { grid-template-columns: 1fr; }
      .top { flex-direction: column; align-items: flex-start; }
      .gauge-card { grid-column: span 2; }
      .avg-level-card .level-mini-canvas { width: 96px; height: 52px; }
    }
  </style>
</head>
<body>
  <main>
    <section class="top">
      <div class="title">
        <h1>Tempest AI Dashboard</h1>
        <div class="subtitle">Primary training and runtime telemetry (live)</div>
      </div>
      <div class="status"><span class="dot" id="statusDot"></span><span id="statusText">Connected</span></div>
    </section>

    <section class="cards">
      <article class="card gauge-card">
        <div class="gauge-head">
          <div class="label">FPS SPD</div>
          <div class="gauge-readout"><span id="mFps">0.0</span><small>fps</small></div>
        </div>
        <canvas id="cFpsGauge" class="gauge-canvas"></canvas>
        <div class="gauge-foot"><span>RL 1K</span><span>MX 1.2K</span></div>
      </article>
      <article class="card gauge-card">
        <div class="gauge-head">
          <div class="label">STEP SPD</div>
          <div class="gauge-readout"><span id="mSteps">0.0</span><small>s/s</small></div>
        </div>
        <canvas id="cStepGauge" class="gauge-canvas"></canvas>
        <div class="gauge-foot"><span>R10 Y20</span><span>G30</span></div>
      </article>
      <article class="card"><div class="label">Frame</div><div class="value" id="mFrame">0</div></article>
      <article class="card"><div class="label">Clnt</div><div class="value" id="mClients">0</div></article>
      <article class="card"><div class="label">Web</div><div class="value" id="mWeb">0</div></article>
      <article class="card avg-level-card">
        <div class="label">Avg Level</div>
        <div class="level-inline">
          <div class="value" id="mLevel">0.0</div>
          <canvas id="cLevelMini" class="level-mini-canvas"></canvas>
        </div>
      </article>
      <article class="card">
        <div class="label">Avg Inf</div>
        <div class="value value-inline"><span class="metric-led" id="mInfLed"></span><span id="mInf">0.00ms</span></div>
      </article>
      <article class="card"><div class="label">Epsilon</div><div class="value" id="mEps">0%</div></article>
      <article class="card"><div class="label">Expert Ratio</div><div class="value" id="mXprt">0%</div></article>
      <article class="card"><div class="label">Avg Reward</div><div class="value" id="mRwrd">0</div></article>
      <article class="card"><div class="label">Loss</div><div class="value" id="mLoss">0</div></article>
      <article class="card"><div class="label">Grad Norm</div><div class="value" id="mGrad">0</div></article>
      <article class="card"><div class="label">Buffer</div><div class="value" id="mBuf">0k (0%)</div></article>
      <article class="card"><div class="label">LR</div><div class="value" id="mLr">-</div></article>
      <article class="card"><div class="label">Q Range</div><div class="value" id="mQ">-</div></article>
    </section>

    <section class="charts">
      <article class="panel">
        <h2>Throughput</h2>
        <div class="legend">
          <span><span class="sw" style="background:#22c55e;"></span>FPS</span>
          <span><span class="sw" style="background:#f59e0b;"></span>Steps/Sec</span>
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
  </main>

  <script>
    const num = new Intl.NumberFormat("en-US");
    const maxPoints = 900;
    const DASH_REFRESH_MS = 100;
    const STEP_GAUGE_AVG_WINDOW = 10;
    const GAUGE_MIN_FPS = 0;
    const GAUGE_MAX_FPS = 1200;
    const GAUGE_REDLINE_FPS = 1000;
    const GAUGE_MIN_STEPS = 0;
    const GAUGE_MAX_STEPS = 30;
    let failedPings = 0;
    const CLIENT_ID = (() => {
      try {
        if (window.crypto && window.crypto.randomUUID) return window.crypto.randomUUID();
      } catch (_) {}
      return `c_${Date.now()}_${Math.random().toString(36).slice(2)}`;
    })();

    const cards = {
      frame: document.getElementById("mFrame"),
      fps: document.getElementById("mFps"),
      steps: document.getElementById("mSteps"),
      clients: document.getElementById("mClients"),
      web: document.getElementById("mWeb"),
      level: document.getElementById("mLevel"),
      inf: document.getElementById("mInf"),
      infLed: document.getElementById("mInfLed"),
      eps: document.getElementById("mEps"),
      xprt: document.getElementById("mXprt"),
      rwrd: document.getElementById("mRwrd"),
      loss: document.getElementById("mLoss"),
      grad: document.getElementById("mGrad"),
      buf: document.getElementById("mBuf"),
      lr: document.getElementById("mLr"),
      q: document.getElementById("mQ"),
    };
    const fpsGaugeCanvas = document.getElementById("cFpsGauge");
    const stepGaugeCanvas = document.getElementById("cStepGauge");

    const charts = {
      throughput: {
        canvas: document.getElementById("cThroughput"),
        series: [
          {
            key: "fps",
            color: "#22c55e",
            axis: { side: "left", min: 0, max: 1200, ticks: [0, 300, 600, 900, 1200] }
          },
          {
            key: "steps_per_sec",
            color: "#f59e0b",
            axis: { side: "right", min: 0, max: 50, ticks: [0, 10, 20, 30, 40, 50] }
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
              label_pad: 52,
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
            axis: { side: "left", group_keys: ["loss", "grad_norm", "bc_loss"] },
          },
          { key: "grad_norm", color: "#f59e0b", axis_ref: "loss" },
          { key: "bc_loss", color: "#22d3ee", axis_ref: "loss" }
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

    function fmtPct(v) {
      if (v === null || v === undefined || Number.isNaN(v)) return "0%";
      return `${(Number(v) * 100.0).toFixed(1)}%`;
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

    function drawFpsGauge(canvas, fps) {
      if (!canvas) return;

      const width = canvas.clientWidth || 360;
      const height = canvas.clientHeight || 160;
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);

      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);

      const cx = width * 0.5;
      const cy = height * 0.68;
      const radius = Math.max(44, Math.min(width * 0.30, height * 0.56));

      // Sweep 240° from lower-left to lower-right across the top.
      const startDeg = 150;
      const spanDeg = 240;
      const degToRad = (d) => (d * Math.PI) / 180.0;
      const clampFps = (v) => Math.max(GAUGE_MIN_FPS, Math.min(GAUGE_MAX_FPS, Number(v) || 0));
      const valToAngle = (v) => {
        const t = (clampFps(v) - GAUGE_MIN_FPS) / (GAUGE_MAX_FPS - GAUGE_MIN_FPS);
        return degToRad(startDeg + spanDeg * t);
      };

      const trackW = Math.max(9, radius * 0.11);
      ctx.lineCap = "round";

      // Main arc track
      ctx.strokeStyle = "rgba(148, 163, 184, 0.58)";
      ctx.lineWidth = trackW;
      ctx.beginPath();
      ctx.arc(cx, cy, radius, degToRad(startDeg), degToRad(startDeg + spanDeg), false);
      ctx.stroke();

      // Redline arc
      ctx.strokeStyle = "rgba(255, 59, 78, 0.92)";
      ctx.lineWidth = trackW + 1.0;
      ctx.beginPath();
      ctx.arc(cx, cy, radius, valToAngle(GAUGE_REDLINE_FPS), valToAngle(GAUGE_MAX_FPS), false);
      ctx.stroke();

      // Tick marks (major every 200, minor every 100)
      for (let v = GAUGE_MIN_FPS; v <= GAUGE_MAX_FPS; v += 100) {
        const isMajor = (v % 200) === 0;
        const a = valToAngle(v);
        const cosA = Math.cos(a);
        const sinA = Math.sin(a);

        const outer = radius + trackW * 0.38;
        const inner = outer - (isMajor ? trackW * 1.6 : trackW * 0.95);

        ctx.strokeStyle = v >= GAUGE_REDLINE_FPS
          ? "rgba(255, 76, 98, 0.96)"
          : isMajor
            ? "rgba(226, 232, 240, 0.95)"
            : "rgba(148, 163, 184, 0.78)";
        ctx.lineWidth = isMajor ? 3.0 : 1.8;
        ctx.beginPath();
        ctx.moveTo(cx + outer * cosA, cy + outer * sinA);
        ctx.lineTo(cx + inner * cosA, cy + inner * sinA);
        ctx.stroke();
      }

      // Needle
      const needleAngle = valToAngle(fps);
      const nCos = Math.cos(needleAngle);
      const nSin = Math.sin(needleAngle);
      const needleLen = radius * 0.98;

      ctx.strokeStyle = "rgba(0, 0, 0, 0.55)";
      ctx.lineWidth = 7.5;
      ctx.beginPath();
      ctx.moveTo(cx + 3, cy + 3);
      ctx.lineTo(cx + needleLen * nCos + 3, cy + needleLen * nSin + 3);
      ctx.stroke();

      ctx.strokeStyle = "#22d3ee";
      ctx.lineWidth = 6.0;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx + needleLen * nCos, cy + needleLen * nSin);
      ctx.stroke();

      // Hub
      ctx.fillStyle = "rgba(2, 6, 23, 0.95)";
      ctx.beginPath();
      ctx.arc(cx, cy, 14, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = "rgba(100, 116, 139, 0.82)";
      ctx.beginPath();
      ctx.arc(cx, cy, 7, 0, Math.PI * 2);
      ctx.fill();
    }

    function drawStepGauge(canvas, stepsPerSec) {
      if (!canvas) return;

      const width = canvas.clientWidth || 360;
      const height = canvas.clientHeight || 160;
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);

      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);

      const cx = width * 0.5;
      const cy = height * 0.68;
      const radius = Math.max(44, Math.min(width * 0.30, height * 0.56));

      const startDeg = 150;
      const spanDeg = 240;
      const degToRad = (d) => (d * Math.PI) / 180.0;
      const clampSteps = (v) => Math.max(GAUGE_MIN_STEPS, Math.min(GAUGE_MAX_STEPS, Number(v) || 0));
      const valToAngle = (v) => {
        const t = (clampSteps(v) - GAUGE_MIN_STEPS) / (GAUGE_MAX_STEPS - GAUGE_MIN_STEPS);
        return degToRad(startDeg + spanDeg * t);
      };

      const trackW = Math.max(9, radius * 0.11);
      ctx.lineCap = "round";

      // Zone arcs: red [0,10], yellow (10,20], green (20,30].
      ctx.lineWidth = trackW + 1.0;
      ctx.strokeStyle = "rgba(239, 68, 68, 0.95)";
      ctx.beginPath();
      ctx.arc(cx, cy, radius, valToAngle(0), valToAngle(10), false);
      ctx.stroke();

      ctx.strokeStyle = "rgba(245, 158, 11, 0.95)";
      ctx.beginPath();
      ctx.arc(cx, cy, radius, valToAngle(10), valToAngle(20), false);
      ctx.stroke();

      ctx.strokeStyle = "rgba(34, 197, 94, 0.95)";
      ctx.beginPath();
      ctx.arc(cx, cy, radius, valToAngle(20), valToAngle(30), false);
      ctx.stroke();

      // Tick marks (major each 10, minor each 5)
      for (let v = GAUGE_MIN_STEPS; v <= GAUGE_MAX_STEPS; v += 5) {
        const isMajor = (v % 10) === 0;
        const a = valToAngle(v);
        const cosA = Math.cos(a);
        const sinA = Math.sin(a);

        const outer = radius + trackW * 0.38;
        const inner = outer - (isMajor ? trackW * 1.6 : trackW * 0.95);

        let tickColor = "rgba(239, 68, 68, 0.95)";
        if (v > 10 && v <= 20) tickColor = "rgba(245, 158, 11, 0.95)";
        else if (v > 20) tickColor = "rgba(34, 197, 94, 0.95)";

        ctx.strokeStyle = tickColor;
        ctx.lineWidth = isMajor ? 3.0 : 1.8;
        ctx.beginPath();
        ctx.moveTo(cx + outer * cosA, cy + outer * sinA);
        ctx.lineTo(cx + inner * cosA, cy + inner * sinA);
        ctx.stroke();
      }

      // Needle
      const needleAngle = valToAngle(stepsPerSec);
      const nCos = Math.cos(needleAngle);
      const nSin = Math.sin(needleAngle);
      const needleLen = radius * 0.98;

      ctx.strokeStyle = "rgba(0, 0, 0, 0.55)";
      ctx.lineWidth = 7.5;
      ctx.beginPath();
      ctx.moveTo(cx + 3, cy + 3);
      ctx.lineTo(cx + needleLen * nCos + 3, cy + needleLen * nSin + 3);
      ctx.stroke();

      ctx.strokeStyle = "#e2e8f0";
      ctx.lineWidth = 6.0;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx + needleLen * nCos, cy + needleLen * nSin);
      ctx.stroke();

      // Hub
      ctx.fillStyle = "rgba(2, 6, 23, 0.95)";
      ctx.beginPath();
      ctx.arc(cx, cy, 14, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = "rgba(100, 116, 139, 0.82)";
      ctx.beginPath();
      ctx.arc(cx, cy, 7, 0, Math.PI * 2);
      ctx.fill();
    }

    function drawChart(canvas, history, seriesDefs) {
      const points = history.slice(-maxPoints);
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

      // Time-compressed x-axis:
      // right quarter = recent 1x window
      // to half mark = 2x history
      // next quarter = 4x history
      // far left = 8x history
      const tsVals = points
        .map((p) => Number(p.ts))
        .filter((v) => Number.isFinite(v));
      const hasTimeAxis = tsVals.length >= 2;
      const newestTs = hasTimeAxis ? Math.max(...tsVals) : 0.0;
      const oldestTs = hasTimeAxis ? Math.min(...tsVals) : 0.0;
      const maxAge = hasTimeAxis ? Math.max(1e-6, newestTs - oldestTs) : 0.0;
      const baseAge = maxAge / 8.0;

      const xNormFromAge = (ageRaw) => {
        if (!hasTimeAxis || baseAge <= 0) return 1.0;
        const age = Math.max(0.0, Math.min(maxAge, ageRaw));
        if (age <= baseAge) {
          return 1.0 - ((age / baseAge) * 0.25);
        }
        if (age <= (2.0 * baseAge)) {
          return 0.75 - (((age - baseAge) / baseAge) * 0.25);
        }
        if (age <= (4.0 * baseAge)) {
          return 0.50 - (((age - (2.0 * baseAge)) / (2.0 * baseAge)) * 0.25);
        }
        return 0.25 - (((age - (4.0 * baseAge)) / (4.0 * baseAge)) * 0.25);
      };

      const ageFromXNorm = (xNormRaw) => {
        if (!hasTimeAxis || baseAge <= 0) return 0.0;
        const xn = Math.max(0.0, Math.min(1.0, xNormRaw));
        if (xn >= 0.75) {
          return (1.0 - xn) * 4.0 * baseAge;
        }
        if (xn >= 0.50) {
          return baseAge + ((0.75 - xn) * 4.0 * baseAge);
        }
        if (xn >= 0.25) {
          return (2.0 * baseAge) + ((0.50 - xn) * 8.0 * baseAge);
        }
        return (4.0 * baseAge) + ((0.25 - xn) * 16.0 * baseAge);
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
      for (let i = 0; i < 4; i++) {
        const y = padT + (plotH * i / 3.0);
        ctx.beginPath();
        ctx.moveTo(padL, y);
        ctx.lineTo(width - padR, y);
        ctx.stroke();
      }
      for (let i = 1; i < 4; i++) {
        const x = padL + (plotW * (i / 4.0));
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

          const labelText = Math.abs(tv) >= 100 ? `${Math.round(tv)}` : `${Number(tv).toFixed(0)}`;
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
        ctx.strokeStyle = s.color;
        ctx.lineWidth = 2.0;
        ctx.beginPath();
        let started = false;
        for (let i = 0; i < n; i++) {
          const val = seriesValue(points[i], s.key);
          if (!Number.isFinite(val)) continue;
          const x = xAt(i);
          const y = yAt(axis, Number(val));
          if (!started) {
            ctx.moveTo(x, y);
            started = true;
          } else {
            ctx.lineTo(x, y);
          }
        }
        ctx.stroke();
      }
    }

    function drawMiniChart(canvas, history, seriesDefs) {
      if (!canvas) return;
      const points = history.slice(-240);
      const width = canvas.clientWidth || 120;
      const height = canvas.clientHeight || 56;
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);

      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);
      if (!points.length) return;

      const padX = 4;
      const padY = 6;
      const plotW = width - (2 * padX);
      const plotH = height - (2 * padY);
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

      let minV = Math.min(...values);
      let maxV = Math.max(...values);
      if (maxV <= minV) {
        maxV = minV + 1.0;
      } else {
        const p = (maxV - minV) * 0.08;
        minV -= p;
        maxV += p;
      }

      const xAt = (i) => {
        const t = points.length <= 1 ? 1.0 : (i / (points.length - 1));
        return padX + (t * plotW);
      };
      const yAt = (v) => {
        const t = (v - minV) / (maxV - minV);
        return padY + ((1.0 - t) * plotH);
      };

      // Soft center guide.
      const yMid = padY + (plotH * 0.5);
      ctx.strokeStyle = "rgba(148, 163, 184, 0.18)";
      ctx.lineWidth = 1.0;
      ctx.beginPath();
      ctx.moveTo(padX, yMid);
      ctx.lineTo(width - padX, yMid);
      ctx.stroke();

      const n = points.length;
      for (const s of seriesDefs) {
        ctx.strokeStyle = s.color;
        ctx.globalAlpha = (s.key === "level_1m") ? 0.95 : 0.82;
        ctx.lineWidth = (s.key === "level_1m") ? 2.0 : 1.6;
        ctx.beginPath();
        let started = false;
        for (let i = 0; i < n; i++) {
          const val = Number(points[i][s.key]);
          if (!Number.isFinite(val)) continue;
          const x = xAt(i);
          const y = yAt(val);
          if (!started) {
            ctx.moveTo(x, y);
            started = true;
          } else {
            ctx.lineTo(x, y);
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

    function updateCards(now, smoothedSteps) {
      cards.frame.textContent = fmtInt(now.frame_count);
      cards.fps.textContent = fmtFloat(now.fps, 1);
      cards.steps.textContent = fmtFloat(smoothedSteps, 1);
      cards.clients.textContent = fmtInt(now.client_count);
      cards.web.textContent = fmtInt(now.web_client_count);
      cards.level.textContent = fmtFloat(now.average_level, 2);
      cards.inf.textContent = `${fmtFloat(now.avg_inf_ms, 2)}ms`;
      setInfLed(now.avg_inf_ms);
      cards.eps.textContent = fmtPct(now.epsilon);
      cards.xprt.textContent = fmtPct(now.expert_ratio);
      cards.rwrd.textContent = fmtInt(now.reward_total);
      cards.loss.textContent = fmtFloat(now.loss, 4);
      cards.grad.textContent = fmtFloat(now.grad_norm, 3);
      cards.buf.textContent = `${fmtInt(now.memory_buffer_k)}k (${fmtInt(now.memory_buffer_pct)}%)`;
      cards.lr.textContent = (now.lr === null || now.lr === undefined) ? "-" : Number(now.lr).toExponential(1);
      cards.q.textContent = (now.q_min === null || now.q_max === null)
        ? "-"
        : `[${fmtFloat(now.q_min, 1)}, ${fmtFloat(now.q_max, 1)}]`;
    }

    function render(payload) {
      if (!payload || !payload.now) return;
      const history = payload.history || [];
      const smoothedStepSpd = computeSmoothedStepSpd(payload.now, history);
      updateCards(payload.now, smoothedStepSpd);
      drawFpsGauge(fpsGaugeCanvas, payload.now.fps);
      drawStepGauge(stepGaugeCanvas, smoothedStepSpd);
      drawChart(charts.throughput.canvas, history, charts.throughput.series);
      drawChart(charts.rewards.canvas, history, charts.rewards.series);
      drawChart(charts.learning.canvas, history, charts.learning.series);
      drawChart(charts.dqn.canvas, history, charts.dqn.series);
      drawMiniChart(charts.level1m.canvas, history, charts.level1m.series);
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
        historyCache = history.slice(-maxPoints);
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
          if (historyCache.length > maxPoints) {
            historyCache.shift();
          }
          lastTs = ts;
          hasNewSample = true;
        }
        if (hasNewSample || !hadNow) {
          renderCurrent();
        }
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

    class DashboardHandler(BaseHTTPRequestHandler):
        def _send(self, payload: bytes, content_type: str = "text/plain", status: int = 200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

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
        history_limit: int = 900,
        open_browser: bool = True,
    ):
        self.metrics = metrics_obj
        self.agent = agent_obj
        self.host = host
        self.port = port
        # Cap sampler refresh at 10 Hz max.
        self.sample_interval = max(0.1, sample_interval)
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
