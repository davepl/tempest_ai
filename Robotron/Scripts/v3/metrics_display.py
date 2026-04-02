#!/usr/bin/env python3
"""Robotron AI v3 — Metrics display: rolling windows + tabular console output.

Ported from v2 metrics_display.py, adapted for PPO metrics.
"""

import sys
import time
import threading
from collections import deque

# ── Rolling reward windows (frame-weighted) ─────────────────────────────────

DQN100K_FRAMES = 100_000
DQN1M_FRAMES = 1_000_000
DQN5M_FRAMES = 5_000_000

_rwd100k: deque = deque()
_rwd100k_frames: int = 0

_rwd1m: deque = deque()
_rwd1m_frames: int = 0

_rwd5m: deque = deque()
_rwd5m_frames: int = 0

# Rolling episode-length windows
_eplen100k: deque = deque()
_eplen100k_frames: int = 0

_eplen1m: deque = deque()
_eplen1m_frames: int = 0

_windows_lock = threading.Lock()


def _add_to_window(buf: deque, frames_name: str, limit: int, reward: float, ep_len: int):
    """Add an episode to a frame-weighted rolling window."""
    if ep_len <= 0:
        return
    g = globals()
    buf.append((float(reward), int(ep_len)))
    g[frames_name] += ep_len
    while buf and g[frames_name] > limit:
        _, l = buf.popleft()
        g[frames_name] -= l


def add_episode_to_reward_windows(reward: float, ep_len: int):
    """Add an episode reward to all rolling reward windows."""
    if ep_len <= 0:
        return
    with _windows_lock:
        _add_to_window(_rwd100k, "_rwd100k_frames", DQN100K_FRAMES, reward, ep_len)
        _add_to_window(_rwd1m, "_rwd1m_frames", DQN1M_FRAMES, reward, ep_len)
        _add_to_window(_rwd5m, "_rwd5m_frames", DQN5M_FRAMES, reward, ep_len)


def add_episode_to_eplen_windows(ep_len: int):
    """Add an episode length to the 100K and 1M-frame rolling windows."""
    if ep_len <= 0:
        return
    with _windows_lock:
        _add_to_window(_eplen100k, "_eplen100k_frames", DQN100K_FRAMES, float(ep_len), ep_len)
        _add_to_window(_eplen1m, "_eplen1m_frames", DQN1M_FRAMES, float(ep_len), ep_len)


def _avg_window(win: deque) -> float:
    if not win:
        return 0.0
    return sum(r for r, _ in win) / len(win)


def get_reward_window_averages() -> tuple[float, float, float]:
    """Return (100K, 1M, 5M) average rewards."""
    with _windows_lock:
        return _avg_window(_rwd100k), _avg_window(_rwd1m), _avg_window(_rwd5m)


def get_eplen_100k_average() -> float:
    with _windows_lock:
        return _avg_window(_eplen100k)


def get_eplen_1m_average() -> float:
    with _windows_lock:
        return _avg_window(_eplen1m)


def export_window_state() -> dict:
    """Serialize window state for checkpointing."""
    with _windows_lock:
        return {
            "rwd100k": list(_rwd100k),
            "rwd100k_frames": _rwd100k_frames,
            "rwd1m": list(_rwd1m),
            "rwd1m_frames": _rwd1m_frames,
            "rwd5m": list(_rwd5m),
            "rwd5m_frames": _rwd5m_frames,
            "eplen100k": list(_eplen100k),
            "eplen100k_frames": _eplen100k_frames,
            "eplen1m": list(_eplen1m),
            "eplen1m_frames": _eplen1m_frames,
        }


def import_window_state(state: dict | None) -> None:
    """Restore window state from checkpoint."""
    global _rwd100k_frames, _rwd1m_frames, _rwd5m_frames
    global _eplen100k_frames, _eplen1m_frames
    if not isinstance(state, dict):
        return
    with _windows_lock:
        _rwd100k.clear()
        _rwd100k.extend((float(r), int(l)) for r, l in state.get("rwd100k", []))
        _rwd100k_frames = int(state.get("rwd100k_frames", 0))

        _rwd1m.clear()
        _rwd1m.extend((float(r), int(l)) for r, l in state.get("rwd1m", []))
        _rwd1m_frames = int(state.get("rwd1m_frames", 0))

        _rwd5m.clear()
        _rwd5m.extend((float(r), int(l)) for r, l in state.get("rwd5m", []))
        _rwd5m_frames = int(state.get("rwd5m_frames", 0))

        _eplen100k.clear()
        _eplen100k.extend((float(r), int(l)) for r, l in state.get("eplen100k", []))
        _eplen100k_frames = int(state.get("eplen100k_frames", 0))

        _eplen1m.clear()
        _eplen1m.extend((float(r), int(l)) for r, l in state.get("eplen1m", []))
        _eplen1m_frames = int(state.get("eplen1m_frames", 0))


# ── Tabular console display ────────────────────────────────────────────────

_row_counter = 0


def _write_console_line(text: str) -> None:
    """Start each metrics line at column 0 without inserting blank lines."""
    if sys.stdout.isatty():
        sys.stdout.write("\r")
    sys.stdout.write(text)
    sys.stdout.write("\n")
    sys.stdout.flush()


def display_metrics_header():
    global _row_counter
    _row_counter = 0
    hdr = (
        f"{'Frame':>11} {'FPS':>7} {'Epsi':>7} {'Xprt':>7} {'AvgScr':>7} "
        f"{'AvgRwd':>9} {'Rwd100K':>9} {'Rwd1M':>9} {'Rwd5M':>9} "
        f"{'Loss':>10} {'PiLoss':>8} {'VLoss':>8} {'Entropy':>8} "
        f"{'EpLen':>8} {'BCLoss':>8} {'BCWgt':>6} "
        f"{'Clnt':>4} {'Levl':>5} "
        f"{'GrNorm':>8} {'LR':>9}"
    )
    _write_console_line(hdr)
    _write_console_line("-" * len(hdr))


def display_metrics_row(server_metrics, agent):
    """Print one formatted row of training metrics.

    Args:
        server_metrics: The socket_server Metrics object.
        agent: The PPOAgent instance.
    """
    global _row_counter
    if _row_counter > 0 and _row_counter % 30 == 0:
        display_metrics_header()

    m = server_metrics
    rwd100k, rwd1m, rwd5m = get_reward_window_averages()

    expert_r = agent.get_expert_ratio()
    eps = agent.get_epsilon()
    bc_w = agent._get_bc_weight()
    lr = agent.optimizer.param_groups[0]["lr"]

    # Mark overridden values with '*'
    eps_mark = "*" if agent.is_epsilon_overridden() else "%"
    xprt_mark = "*" if agent.is_expert_overridden() else "%"
    train_mark = "" if agent.training_enabled else " T-OFF"

    def _fr(v, w=9):
        try:
            return f"{float(v):.1f}".rjust(w)
        except Exception:
            return "0.0".rjust(w)

    row = (
        f"{m.total_frames:>11,} {m.fps:>7.1f} "
        f"{eps*100:>6.1f}{eps_mark} {expert_r*100:>6.1f}{xprt_mark} "
        f"{int(round(m.avg_game_score)):>7,} "
        f"{_fr(m.avg_reward)} {_fr(rwd100k)} {_fr(rwd1m)} {_fr(rwd5m)} "
        f"{agent.last_loss:>10.6f} {agent.last_policy_loss:>8.5f} "
        f"{agent.last_value_loss:>8.5f} {agent.last_entropy:>8.5f} "
        f"{m.avg_ep_len:>8.1f} {agent.last_bc_loss:>8.5f} {bc_w:>6.3f} "
        f"{m.client_count:>4} {m.avg_level:>5.1f} "
        f"{agent.last_grad_norm:>8.3f} {lr:>9.1e}{train_mark}"
    )
    _write_console_line(row)
    _row_counter += 1
