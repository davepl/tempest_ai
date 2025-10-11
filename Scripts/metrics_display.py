#!/usr/bin/env python3
# ==================================================================================================================
# ||                                                                                                              ||
# ||                                 TEMPEST AI • METRICS DISPLAY AND REPORTING                                   ||
# ||                                                                                                              ||
# ||  FILE: Scripts/metrics_display.py                                                                            ||
# ||  ROLE: Prints periodic metrics header and rows; computes rolling DQN windows and training telemetry.          ||
# ||                                                                                                              ||
# ||  NEED TO KNOW:                                                                                               ||
# ||   - Header printed once; rows show FPS, epsilon, rewards, losses, Q-range, Train%, Training Stats.            ||
# ||   - Steps/s and Samples/s computed per-interval; Train% based on queue requested vs missed.                  ||
# ||   - DQN 1M/5M windows updated after first real training step.                                                ||
# ||                                                                                                              ||
# ||  CONSUMES: metrics, RL_CONFIG                                                                                ||
# ||  PRODUCES: Console-formatted performance summary                                                             ||
# ||                                                                                                              ||
# ==================================================================================================================
"""
Metrics display for Tempest AI.
"""

# Prevent direct execution
if __name__ == "__main__":
    print("This is not the main application, run 'main.py' instead")
    exit(1)

import os
import sys
import time
import threading
import numpy as np
from typing import Optional, List, Dict, Any
from collections import deque

# Import from config.py
from config import metrics, IS_INTERACTIVE, RL_CONFIG

# Add a counter to track the number of rows printed
row_counter = 0

# Rolling window for DQN reward over last 5M frames
DQN_WINDOW_FRAMES = 5_000_000
_dqn_window = deque()  # entries: (frames_in_interval: int, dqn_reward_mean: float, frame_end: int)
_dqn_window_frames = 0
_last_frame_count_seen = None

# Rolling window for DQN reward over last 1M frames
DQN1M_WINDOW_FRAMES = 1_000_000
_dqn1m_window = deque()  # entries: (frames_in_interval: int, dqn_reward_mean: float, frame_end: int)
_dqn1m_window_frames = 0
_last_frame_count_seen_1m = None

def _update_dqn_window(mean_dqn_reward: float):
    """Update the 5M-frames rolling window with the latest interval.

    Uses the number of frames progressed since the last row as the weight.
    """
    global _dqn_window_frames, _last_frame_count_seen
    current_frame = metrics.frame_count
    # Determine frames elapsed since last sample
    if _last_frame_count_seen is None:
        delta_frames = 0
    else:
        delta_frames = max(0, current_frame - _last_frame_count_seen)
    _last_frame_count_seen = current_frame

    # If no frame progress (e.g., first row), just return without adding
    if delta_frames <= 0:
        return

    # Append new interval
    _dqn_window.append((delta_frames, float(mean_dqn_reward), int(current_frame)))
    _dqn_window_frames += delta_frames

    # Trim window to last 5M frames (may need partial trim of the oldest bucket)
    while _dqn_window and _dqn_window_frames > DQN_WINDOW_FRAMES:
        overflow = _dqn_window_frames - DQN_WINDOW_FRAMES
        oldest_frames, oldest_val, oldest_end = _dqn_window[0]
        if oldest_frames <= overflow:
            _dqn_window.popleft()
            _dqn_window_frames -= oldest_frames
        else:
            # Partially trim the oldest bucket
            kept_frames = oldest_frames - overflow
            _dqn_window[0] = (kept_frames, oldest_val, oldest_end)
            _dqn_window_frames = DQN_WINDOW_FRAMES
            break

def _update_dqn1m_window(mean_dqn_reward: float):
    """Update the 1M-frames rolling window with the latest interval.

    Uses the number of frames progressed since the last row as the weight.
    """
    global _dqn1m_window_frames, _last_frame_count_seen_1m
    current_frame = metrics.frame_count
    # Determine frames elapsed since last sample
    if _last_frame_count_seen_1m is None:
        delta_frames = 0
    else:
        delta_frames = max(0, current_frame - _last_frame_count_seen_1m)
    _last_frame_count_seen_1m = current_frame

    # If no frame progress (e.g., first row), just return without adding
    if delta_frames <= 0:
        return

    # Append new interval
    _dqn1m_window.append((delta_frames, float(mean_dqn_reward), int(current_frame)))
    _dqn1m_window_frames += delta_frames

    # Trim window to last 1M frames (may need partial trim of the oldest bucket)
    while _dqn1m_window and _dqn1m_window_frames > DQN1M_WINDOW_FRAMES:
        overflow = _dqn1m_window_frames - DQN1M_WINDOW_FRAMES
        oldest_frames, oldest_val, oldest_end = _dqn1m_window[0]
        if oldest_frames <= overflow:
            _dqn1m_window.popleft()
            _dqn1m_window_frames -= oldest_frames
        else:
            # Partially trim the oldest bucket
            kept_frames = oldest_frames - overflow
            _dqn1m_window[0] = (kept_frames, oldest_val, oldest_end)
            _dqn1m_window_frames = DQN1M_WINDOW_FRAMES
            break

def _compute_dqn_window_stats():
    """Compute weighted average and weighted regression slope (per million frames) for the 5M-frame window.

    Returns (avg, slope_per_million).
    """
    if not _dqn_window or _dqn_window_frames <= 0:
        return 0.0, 0.0

    # Weighted average
    w_sum = float(_dqn_window_frames)
    wy_sum = sum(fr * val for fr, val, _ in _dqn_window)
    avg = wy_sum / w_sum if w_sum > 0 else 0.0

    # Weighted linear regression slope of y vs x, with weights = frames
    # Use frame_end as x for each bucket
    wx_sum = sum(fr * x for fr, _, x in _dqn_window)
    wxx_sum = sum(fr * (x * x) for fr, _, x in _dqn_window)
    wy_sum = sum(fr * y for fr, y, _ in _dqn_window)
    wxy_sum = sum(fr * x * y for fr, y, x in _dqn_window)

    denom = (wxx_sum - (wx_sum * wx_sum) / w_sum) if w_sum > 0 else 0.0
    if denom <= 0:
        slope = 0.0
    else:
        slope = (wxy_sum - (wx_sum * wy_sum) / w_sum) / denom

    slope_per_million = slope * 1_000_000.0
    return avg, slope_per_million

def _compute_dqn1m_window_stats():
    """Compute weighted average for the 1M-frame window.

    Returns avg.
    """
    if not _dqn1m_window or _dqn1m_window_frames <= 0:
        return 0.0

    # Weighted average
    w_sum = float(_dqn1m_window_frames)
    wy_sum = sum(fr * val for fr, val, _ in _dqn1m_window)
    avg = wy_sum / w_sum if w_sum > 0 else 0.0
    return avg

def clear_screen():
    """Clear the screen and move cursor to home position"""
    if IS_INTERACTIVE:
        # Clear screen and move to home
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()

def print_metrics_line(message, is_header=False):
    """Print a metrics line with proper formatting"""
    global row_counter
    
    if IS_INTERACTIVE:
        if is_header:
            # For header: print at current position
            print(message)
            print("-" * len(message))  # Add separator line
            # Reset row counter when header is printed
            row_counter = 0
        else:
            # For rows: just print at current position
            print(message)
            # Increment row counter
            row_counter += 1
        sys.stdout.flush()
    else:
        print(message)

def display_metrics_header():
    """Display the header for metrics output"""
    row_counter = 0
    # clear_screen()
    
    # Header with Q-Value Range moved before Training Stats, reward components removed
    header = (
        f"{'Frame':>11} {'FPS':>6} {'Epsi':>6} {'Xprt':>6} "
        f"{'Rwrd':>6} {'Subj':>6} {'Obj':>6} {'DQN':>6} {'DQN1M':>6} {'DQN5M':>6} {'DQNSlope':>9} "
        f"{'DLoss':>10} {'CLoss':>10} "
        f"{'Agree%':>7} {'Done%':>6} {'HMean':>6} "
        f"{'Clnt':>4} {'Levl':>5} {'OVR':>3} {'Expert':>6} {'Train':>5} "
        f"{'AvgInf':>7} {'Samp/s':>8} {'Steps/s':>8} {'GradNorm':>8} {'ClipΔ':>6} {'Q-Value Range':>14} {'Train%':>7} {'Training Stats':>15}"
    )
    
    print_metrics_line(header, is_header=True)

    # Initialize the timing anchor for Steps/s so the first row divides by real elapsed seconds
    try:
        now = time.time()
        with metrics.lock:
            last_t = getattr(metrics, 'last_metrics_row_time', 0.0)
            if not isinstance(last_t, (int, float)) or last_t <= 0.0:
                metrics.last_metrics_row_time = now
    except Exception:
        # If metrics.lock or attributes aren't available yet, skip init safely
        pass

def display_metrics_row(agent, kb_handler):
    """Display a row of metrics data"""
    global row_counter
    global _last_show_reward_cols
    
    # Check if we need to print the header (every 30th row)
    if row_counter > 0 and row_counter % 30 == 0:
        display_metrics_header()
    
    # Compute reward averages since last print; fallback to recent deque averages (aligned across all three)
    mean_reward = 0.0
    mean_subj_reward = 0.0
    mean_obj_reward = 0.0
    mean_dqn_reward = 0.0
    mean_expert_reward = 0.0
    used_fallback_aligned = False
    with metrics.lock:
        if getattr(metrics, 'reward_count_interval_total', 0) > 0:
            mean_reward = metrics.reward_sum_interval_total / max(metrics.reward_count_interval_total, 1)
        if getattr(metrics, 'reward_count_interval_dqn', 0) > 0:
            mean_dqn_reward = metrics.reward_sum_interval_dqn / max(metrics.reward_count_interval_dqn, 1)
        if getattr(metrics, 'reward_count_interval_expert', 0) > 0:
            mean_expert_reward = metrics.reward_sum_interval_expert / max(metrics.reward_count_interval_expert, 1)
        if getattr(metrics, 'reward_count_interval_subj', 0) > 0:
            mean_subj_reward = metrics.reward_sum_interval_subj / max(metrics.reward_count_interval_subj, 1)
        if getattr(metrics, 'reward_count_interval_obj', 0) > 0:
            mean_obj_reward = metrics.reward_sum_interval_obj / max(metrics.reward_count_interval_obj, 1)
        # Reset interval counters so next print is fresh
        metrics.reward_sum_interval_total = 0.0
        metrics.reward_count_interval_total = 0
        metrics.reward_sum_interval_dqn = 0.0
        metrics.reward_count_interval_dqn = 0
        metrics.reward_sum_interval_expert = 0.0
        metrics.reward_count_interval_expert = 0
        metrics.reward_sum_interval_subj = 0.0
        metrics.reward_count_interval_subj = 0
        metrics.reward_sum_interval_obj = 0.0
        metrics.reward_count_interval_obj = 0
    # Fallback if no interval episodes finished: compute aligned means across the same last-N episodes
    if mean_reward == 0.0 and mean_dqn_reward == 0.0 and mean_expert_reward == 0.0:
        try:
            total_q = list(metrics.episode_rewards) if metrics.episode_rewards else []
            subj_q = list(metrics.subj_rewards) if hasattr(metrics, 'subj_rewards') and metrics.subj_rewards else []
            obj_q = list(metrics.obj_rewards) if hasattr(metrics, 'obj_rewards') and metrics.obj_rewards else []
            dqn_q = list(metrics.dqn_rewards) if metrics.dqn_rewards else []
            exp_q = list(metrics.expert_rewards) if metrics.expert_rewards else []
            n = min(len(total_q), len(subj_q), len(obj_q), len(dqn_q), len(exp_q), 20)
            if n > 0:
                mean_reward = sum(total_q[-n:]) / float(n)
                mean_subj_reward = sum(subj_q[-n:]) / float(n) if subj_q else 0.0
                mean_obj_reward = sum(obj_q[-n:]) / float(n) if obj_q else 0.0
                mean_dqn_reward = sum(dqn_q[-n:]) / float(n)
                mean_expert_reward = sum(exp_q[-n:]) / float(n)
                used_fallback_aligned = True
        except Exception:
            pass
    
    # Get the latest loss value (fallback) and compute avg since last print; also compute Avg Inference time and Steps/s
    # Simplified loss reporting: only DLoss and CLoss; drop combined loss
    d_loss_avg = float(getattr(metrics, 'last_d_loss', 0.0) or 0.0)
    c_loss_avg = float(getattr(metrics, 'last_c_loss', 0.0) or 0.0)
    avg_inference_time_ms = 0.0
    steps_per_sec = 0.0
    samples_per_sec = 0.0
    steps_per_1k_frames = 0.0
    with metrics.lock:
        # Average inference time and reset
        if metrics.total_inference_requests > 0:
            avg_inference_time_ms = (metrics.total_inference_time / metrics.total_inference_requests) * 1000
        metrics.total_inference_time = 0.0
        metrics.total_inference_requests = 0

        # Component interval averages only (keep dloss and closs)
        try:
            if getattr(metrics, 'd_loss_count_interval', 0) > 0:
                d_loss_avg = metrics.d_loss_sum_interval / max(metrics.d_loss_count_interval, 1)
            if getattr(metrics, 'c_loss_count_interval', 0) > 0:
                c_loss_avg = metrics.c_loss_sum_interval / max(metrics.c_loss_count_interval, 1)
        except Exception:
            pass
        # Reset (we no longer use the combined loss columns).
        metrics.loss_sum_interval = 0.0
        metrics.loss_count_interval = 0
        metrics.d_loss_sum_interval = 0.0
        metrics.d_loss_count_interval = 0
        metrics.c_loss_sum_interval = 0.0
        metrics.c_loss_count_interval = 0

        # Steps/s: compute using time elapsed since last row
        now = time.time()
        last_t = getattr(metrics, 'last_metrics_row_time', 0.0)
        elapsed = now - last_t if last_t > 0.0 else None
        steps_int = int(getattr(metrics, 'training_steps_interval', 0))
        frames_int = int(getattr(metrics, 'frames_count_interval', 0))
        if elapsed and elapsed > 0:
            steps_per_sec = steps_int / elapsed
        else:
            steps_per_sec = float(steps_int)
        # Samples/s reflects batch_size * Steps/s; this makes batch size changes visible
        try:
            samples_per_sec = steps_per_sec * float(RL_CONFIG.batch_size)
        except Exception:
            samples_per_sec = 0.0
        # Interval Steps/1kF using the same interval counts
        denom_frames = max(1, frames_int)
        steps_per_1k_frames = (steps_int * 1000.0) / float(denom_frames)
        # Compute training completion percent over the interval
        req = int(getattr(metrics, 'training_steps_requested_interval', 0))
        missed = int(getattr(metrics, 'training_steps_missed_interval', 0))
        completed = max(0, req - missed)
        train_pct = (completed / req * 100.0) if req > 0 else 100.0
        # Reset intervals and update last row time
        metrics.training_steps_interval = 0
        metrics.training_steps_requested_interval = 0
        metrics.training_steps_missed_interval = 0
        metrics.frames_count_interval = 0
        metrics.last_metrics_row_time = now
    
    # Average level since last print, default to current snapshot; display as 1-based
    display_level = metrics.average_level + 1.0
    with metrics.lock:
        if getattr(metrics, 'level_count_interval', 0) > 0:
            avg_level_interval = metrics.level_sum_interval / max(metrics.level_count_interval, 1)
            display_level = avg_level_interval + 1.0
        # Reset interval accumulators
        metrics.level_sum_interval = 0.0
        metrics.level_count_interval = 0

    # Update 5M-frame DQN window stats only after training has actually started
    try:
        trained_once = getattr(metrics, 'total_training_steps', 0) > 0
    except Exception:
        trained_once = False
    if trained_once:
        try:
            _update_dqn_window(mean_dqn_reward)
            dqn5m_avg, dqn5m_slopeM = _compute_dqn_window_stats()
        except Exception:
            dqn5m_avg, dqn5m_slopeM = 0.0, 0.0
    else:
        dqn5m_avg, dqn5m_slopeM = 0.0, 0.0

    # Update 1M-frame DQN window stats only after training has actually started
    if trained_once:
        try:
            _update_dqn1m_window(mean_dqn_reward)
            dqn1m_avg = _compute_dqn1m_window_stats()
        except Exception:
            dqn1m_avg = 0.0
    else:
        dqn1m_avg = 0.0

    # Publish DQN5M stats to global metrics for gating logic elsewhere
    try:
        with metrics.lock:
            metrics.dqn5m_avg = float(dqn5m_avg)
            metrics.dqn5m_slopeM = float(dqn5m_slopeM)
    except Exception:
        pass

    # steps_per_1k_frames already computed above using the same interval counts
    
    # Calculate frames since last target update  
    frames_since_target_update = metrics.frame_count - metrics.last_target_update_frame
    
    # Format training stats: MemK/Steps/StepsPer1kF/TargetAge
    mem_k = getattr(metrics, 'memory_buffer_size', 0) // 1000
    training_stats = f"{mem_k}k/{metrics.total_training_steps}/{steps_per_1k_frames:.1f}/{frames_since_target_update//1000}k"

    # Get Q-value range from the agent only after training has started
    q_range = "N/A"
    if agent and trained_once:
        try:
            min_q, max_q = agent.get_q_value_range()
            if not (np.isnan(min_q) or np.isnan(max_q)):
                q_range = f"[{min_q:.2f}, {max_q:.2f}]"
        except Exception:
            q_range = "Error"

    # Compute frames/time since last inference sync and last target update
    now = time.time()
    # Frames since
    sync_df = metrics.frame_count - getattr(metrics, 'last_inference_sync_frame', 0)
    targ_df = metrics.frame_count - getattr(metrics, 'last_hard_target_update_frame', 0)

    # Seconds since (guard against unset timestamps which default to 0.0)
    last_sync_time = getattr(metrics, 'last_inference_sync_time', 0.0)
    last_targ_time = getattr(metrics, 'last_hard_target_update_time', 0.0)
    sync_dt = (now - last_sync_time) if last_sync_time > 0.0 else None
    targ_dt = (now - last_targ_time) if last_targ_time > 0.0 else None
    sync_col = f"{sync_df//1000}k/{(f'{sync_dt:>4.1f}s' if sync_dt is not None else 'n/a'):>6}"
    targ_col = f"{targ_df//1000}k/{(f'{targ_dt:>4.1f}s' if targ_dt is not None else 'n/a'):>6}"

    # Additional diagnostics for troubleshooting
    d_loss = d_loss_avg
    c_loss = c_loss_avg
    agree_pct = float(getattr(metrics, 'action_agree_pct', 0.0) or 0.0)
    done_pct = 100.0 * float(getattr(metrics, 'batch_done_frac', 0.0) or 0.0)
    h_mean = float(getattr(metrics, 'batch_h_mean', 1.0) or 1.0)

    # Base row text with Q-Value Range moved before Training Stats, reward components removed
    # Show effective epsilon (0.00 when epsilon override is ON)
    try:
        effective_eps = metrics.get_effective_epsilon()
    except Exception:
        effective_eps = metrics.epsilon

    row = (
        f"{metrics.frame_count:>11,} {metrics.fps:>6.1f} {effective_eps:>6.2f} "
    f"{metrics.expert_ratio*100:>5.1f}% {mean_reward:>6.2f} {mean_subj_reward:>6.2f} {mean_obj_reward:>6.2f} {mean_dqn_reward:>6.2f} {dqn1m_avg:>6.2f} {dqn5m_avg:>6.2f} {dqn5m_slopeM:>9.3f} "
    f"{d_loss:>10.6f} {c_loss:>10.6f} "
    f"{agree_pct:>7.1f} {done_pct:>6.1f} {h_mean:>6.2f} "
    f"{metrics.client_count:04d} {display_level:>5.1f} "
        f"{'ON' if metrics.override_expert else 'OFF':>3} "
        f"{'ON' if metrics.expert_mode else 'OFF':>6} "
        f"{'ON' if metrics.training_enabled else 'OFF':>5} "
        f"{avg_inference_time_ms:>7.2f} "
        f"{samples_per_sec:>8.0f} "
        f"{steps_per_sec:>8.1f} "
        f"{metrics.last_grad_norm:>8.3f} "
        f"{metrics.last_clip_delta:>6.3f} "
        f"{q_range:>14} {train_pct:>6.1f}% {training_stats:>15}"
    )
    
    print_metrics_line(row)

def run_stats_reporter(metrics):
    """Run the stats reporter in a loop"""
    display_metrics_header()
    
    while True:
        try:
            time.sleep(1)  # Update every second
            display_metrics_row(None, None)
        except Exception as e:
            print(f"Error in stats reporter: {e}")
            time.sleep(1)  # Wait a bit before retrying