#!/usr/bin/env python3
# ==================================================================================================================
# ||  TEMPEST AI v2 • METRICS DISPLAY                                                                            ||
# ||  Periodic header + row output for training telemetry.                                                        ||
# ==================================================================================================================
"""Metrics display for Tempest AI v2."""

if __name__ == "__main__":
    print("This is not the main application, run 'main.py' instead")
    exit(1)

import sys, time, math, threading
import numpy as np
from collections import deque

from config import metrics, IS_INTERACTIVE, RL_CONFIG

row_counter = 0

# Rolling DQN reward windows
DQN1K_FRAMES = 1_000
_dqn1k = deque()
_dqn1k_frames = 0

DQN1M_FRAMES = 1_000_000
_dqn1m = deque()
_dqn1m_frames = 0

DQN5M_FRAMES = 5_000_000
_dqn5m = deque()
_dqn5m_frames = 0

_dqn_windows_lock = threading.Lock()


def add_episode_to_dqn1k_window(dqn_reward: float, ep_len: int):
    global _dqn1k_frames
    if ep_len <= 0:
        return
    with _dqn_windows_lock:
        _dqn1k.append((float(dqn_reward), int(ep_len)))
        _dqn1k_frames += ep_len
        while _dqn1k and _dqn1k_frames > DQN1K_FRAMES:
            _, l = _dqn1k.popleft()
            _dqn1k_frames -= l


def add_episode_to_dqn1m_window(dqn_reward: float, ep_len: int):
    global _dqn1m_frames
    if ep_len <= 0:
        return
    with _dqn_windows_lock:
        _dqn1m.append((float(dqn_reward), int(ep_len)))
        _dqn1m_frames += ep_len
        while _dqn1m and _dqn1m_frames > DQN1M_FRAMES:
            _, l = _dqn1m.popleft()
            _dqn1m_frames -= l


def add_episode_to_dqn5m_window(dqn_reward: float, ep_len: int):
    global _dqn5m_frames
    if ep_len <= 0:
        return
    with _dqn_windows_lock:
        _dqn5m.append((float(dqn_reward), int(ep_len)))
        _dqn5m_frames += ep_len
        while _dqn5m and _dqn5m_frames > DQN5M_FRAMES:
            _, l = _dqn5m.popleft()
            _dqn5m_frames -= l


def _avg_window(win):
    if not win:
        return 0.0
    return sum(r for r, _ in win) / len(win)


def get_dqn_window_averages() -> tuple[float, float, float]:
    with _dqn_windows_lock:
        return _avg_window(_dqn1k), _avg_window(_dqn1m), _avg_window(_dqn5m)


def clear_screen():
    if IS_INTERACTIVE:
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()


def _print_line(msg, is_header=False):
    global row_counter
    if is_header:
        print(msg)
        print("-" * len(msg))
        row_counter = 0
    else:
        print(msg)
        row_counter += 1
    sys.stdout.flush()


def display_metrics_header():
    global row_counter
    row_counter = 0
    hdr = (
        f"{'Frame':>11} {'FPS':>7} {'Epsi':>7} {'Xprt':>7} "
        f"{'Rwrd':>9} {'Subj':>9} {'Obj':>9} {'DQN':>9} {'DQN1M':>9} {'DQN5M':>9} "
        f"{'Loss':>10} {'Agree%':>7} "
        f"{'EpLen':>8} {'BCLoss':>8} "
        f"{'Clnt':>4} {'Levl':>5} "
        f"{'AvgInf':>7} {'Steps/s':>8} {'GrNorm':>8} {'Q-Range':>14} {'Mem':>10} {'LR':>9}"
    )
    _print_line(hdr, is_header=True)
    try:
        now = time.time()
        with metrics.lock:
            if metrics.last_fps_time <= 0:
                metrics.last_fps_time = now
    except Exception:
        pass


def display_metrics_row(agent, kb_handler):
    global row_counter
    if row_counter > 0 and row_counter % 30 == 0:
        display_metrics_header()

    # ── Interval averages ───────────────────────────────────────────────
    mean_reward = mean_subj = mean_obj = mean_dqn = 0.0
    with metrics.lock:
        if metrics.reward_count_interval > 0:
            mean_reward = metrics.reward_sum_interval / max(1, metrics.reward_count_interval)
        if metrics.reward_count_interval_dqn > 0:
            mean_dqn = metrics.reward_sum_interval_dqn / max(1, metrics.reward_count_interval_dqn)
        if metrics.reward_count_interval_subj > 0:
            mean_subj = metrics.reward_sum_interval_subj / max(1, metrics.reward_count_interval_subj)
        if metrics.reward_count_interval_obj > 0:
            mean_obj = metrics.reward_sum_interval_obj / max(1, metrics.reward_count_interval_obj)
        # Reset
        metrics.reward_sum_interval = metrics.reward_count_interval = 0
        metrics.reward_sum_interval_dqn = metrics.reward_count_interval_dqn = 0
        metrics.reward_sum_interval_subj = metrics.reward_count_interval_subj = 0
        metrics.reward_sum_interval_obj = metrics.reward_count_interval_obj = 0

    # Fallback to deque
    if mean_reward == 0.0 and mean_dqn == 0.0:
        try:
            n = min(len(metrics.episode_rewards), len(metrics.dqn_rewards), 20)
            if n > 0:
                mean_reward = sum(list(metrics.episode_rewards)[-n:]) / n
                mean_dqn = sum(list(metrics.dqn_rewards)[-n:]) / n
                s = list(metrics.subj_rewards) if metrics.subj_rewards else []
                o = list(metrics.obj_rewards) if metrics.obj_rewards else []
                if s:
                    mean_subj = sum(s[-n:]) / min(n, len(s))
                if o:
                    mean_obj = sum(o[-n:]) / min(n, len(o))
        except Exception:
            pass

    # ── Loss / agreement / steps/s ──────────────────────────────────────
    loss_avg = 0.0
    agree_avg = 0.0
    steps_per_sec = 0.0
    avg_inf_ms = 0.0
    with metrics.lock:
        if metrics.total_inference_requests > 0:
            avg_inf_ms = (metrics.total_inference_time / metrics.total_inference_requests) * 1000
        metrics.total_inference_time = 0.0
        metrics.total_inference_requests = 0

        if metrics.loss_count_interval > 0:
            loss_avg = metrics.loss_sum_interval / max(1, metrics.loss_count_interval)
        if metrics.agree_count_interval > 0:
            agree_avg = metrics.agree_sum_interval / max(1, metrics.agree_count_interval)
        metrics.loss_sum_interval = metrics.loss_count_interval = 0
        metrics.agree_sum_interval = metrics.agree_count_interval = 0

        now = time.time()
        last_t = getattr(metrics, "_last_row_time", 0.0)
        steps_int = metrics.training_steps_interval
        elapsed = now - last_t if last_t > 0 else 1.0
        steps_per_sec = steps_int / max(0.001, elapsed)
        metrics.training_steps_interval = 0
        metrics.frames_count_interval = 0
        metrics._last_row_time = now

    # ── Episode length ──────────────────────────────────────────────────
    avg_ep_len = 0.0
    with metrics.lock:
        if metrics.episode_length_count_interval > 0:
            avg_ep_len = metrics.episode_length_sum_interval / max(1, metrics.episode_length_count_interval)
        metrics.episode_length_sum_interval = 0
        metrics.episode_length_count_interval = 0

    # ── Level ───────────────────────────────────────────────────────────
    display_level = metrics.average_level + 1.0

    # ── DQN windows ─────────────────────────────────────────────────────
    _, dqn1m, dqn5m = get_dqn_window_averages()

    # ── Q range ─────────────────────────────────────────────────────────
    q_range = "N/A"
    if agent:
        try:
            mn, mx = agent.get_q_value_range()
            if not (np.isnan(mn) or np.isnan(mx)):
                q_range = f"[{mn:.1f},{mx:.1f}]"
        except Exception:
            q_range = "err"

    mem_k = metrics.memory_buffer_size // 1000

    # ── Current LR ──────────────────────────────────────────────────────
    lr_str = ""
    if agent and hasattr(agent, "get_lr"):
        try:
            cur_lr = agent.get_lr()
            lr_str = f"{cur_lr:.1e}"
        except Exception:
            lr_str = "?"

    # ── Reward scaling for display ──────────────────────────────────────
    inv = 1.0 / max(1e-9, RL_CONFIG.obj_reward_scale)
    inv_s = 1.0 / max(1e-9, RL_CONFIG.subj_reward_scale)

    def _fr(v, w=9):
        try:
            return f"{float(v):.0f}".rjust(w)
        except Exception:
            return "0".rjust(w)

    eps_pct = f"{metrics.get_effective_epsilon()*100:.0f}%".rjust(7)
    xprt_pct = f"{metrics.get_expert_ratio()*100:.0f}%".rjust(7)

    row = (
        f"{metrics.frame_count:>11,} {metrics.fps:>7.1f} {eps_pct} {xprt_pct} "
        f"{_fr(mean_reward*inv)} {_fr(mean_subj*inv_s)} {_fr(mean_obj*inv)} {_fr(mean_dqn*inv)} "
        f"{_fr(dqn1m*inv)} {_fr(dqn5m*inv)} "
        f"{loss_avg:>10.6f} {agree_avg*100:>6.1f}% "
        f"{avg_ep_len:>8.1f} {metrics.last_bc_loss:>8.4f} "
        f"{metrics.client_count:>4} {display_level:>5.1f} "
        f"{avg_inf_ms:>7.2f} {steps_per_sec:>8.1f} "
        f"{metrics.last_grad_norm:>8.3f} {q_range:>14} {mem_k:>8}k {lr_str:>9}"
    )
    _print_line(row)
