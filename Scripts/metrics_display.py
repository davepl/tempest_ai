#!/usr/bin/env python3
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
    
    # Full header with all desired columns (aligned to match data column widths)
    header = (
        f"{'Frame':>11} {'FPS':>7} {'Epsi':>6} {'Xprt':>7} "
        f"{'Rwrd':>7} {'Subj':>7} {'Obj':>7} {'DQN':>7} {'DQN1M':>6} {'DQN5M':>6} {'DQNSlope':>9} {'Loss':>10} "
        f"{'Agree%':>7} "
        f"{'AvgEpLen':>8} {'Train%':>6} "
        f"{'Clnt':>4} {'Levl':>5} "
        f"{'AvgInf':>7} {'Samp/s':>9} {'Steps/s':>8} {'GradNorm':>8} {'ClipΔ':>6} {'Q-Range':>12} {'Stats':>26}"
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
    latest_loss = metrics.losses[-1] if metrics.losses else 0.0
    loss_avg = latest_loss
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

        # Average loss since last row and reset
        if getattr(metrics, 'loss_count_interval', 0) > 0:
            loss_avg = metrics.loss_sum_interval / max(metrics.loss_count_interval, 1)
        # Average agreement since last row and reset
        agree_avg = 0.0
        if getattr(metrics, 'agree_count_interval', 0) > 0:
            agree_avg = metrics.agree_sum_interval / max(metrics.agree_count_interval, 1)
        # Reset interval accumulators
        metrics.loss_sum_interval = 0.0
        metrics.loss_count_interval = 0
        metrics.d_loss_sum_interval = 0.0
        metrics.d_loss_count_interval = 0
        metrics.agree_sum_interval = 0.0
        metrics.agree_count_interval = 0

        # Steps/s: compute using time elapsed since last row
        now = time.time()
        last_t = getattr(metrics, 'last_metrics_row_time', 0.0)
        elapsed = now - last_t if last_t > 0.0 else None
        steps_int = int(getattr(metrics, 'training_steps_interval', 0))
        frames_int = int(getattr(metrics, 'frames_count_interval', 0))
        steps_missed_int = int(getattr(metrics, 'training_steps_missed_interval', 0))
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
        # Calculate training completion percentage against served + missed requests
        denom_steps = max(1.0, float(steps_int + steps_missed_int))
        train_pct = (100.0 * steps_int / denom_steps)
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

    # Update 5M-frame DQN window stats now that we have this row's mean_dqn_reward
    try:
        _update_dqn_window(mean_dqn_reward)
        dqn5m_avg, dqn5m_slopeM = _compute_dqn_window_stats()
    except Exception:
        dqn5m_avg, dqn5m_slopeM = 0.0, 0.0

    # Update 1M-frame DQN window stats
    try:
        _update_dqn1m_window(mean_dqn_reward)
        dqn1m_avg = _compute_dqn1m_window_stats()
    except Exception:
        dqn1m_avg = 0.0

    # Publish DQN5M stats to global metrics for gating logic elsewhere
    try:
        with metrics.lock:
            metrics.dqn5m_avg = float(dqn5m_avg)
            metrics.dqn5m_slopeM = float(dqn5m_slopeM)
    except Exception:
        pass

    # steps_per_1k_frames already computed above using the same interval counts
    
    # Calculate training steps since last target update
    steps_since_target_update = metrics.total_training_steps - getattr(metrics, 'last_target_update_step', 0)
    
    # Format training stats: MemK/Steps/P98-100%/P95-98%/P90-95%/Main%
    mem_k = getattr(metrics, 'memory_buffer_size', 0) // 1000
    
    # Get partition stats if agent is available
    # Show fill % for all N priority buckets plus main bucket
    bucket_fill_pcts = []
    if agent and hasattr(agent, 'memory') and hasattr(agent.memory, 'get_partition_stats'):
        try:
            pstats = agent.memory.get_partition_stats()
            if not pstats.get('priority_buckets_enabled', False):
                bucket_fill_pcts.append(f"{pstats.get('main_fill_pct', 0.0):.0f}%")
            else:
                bucket_names = list(pstats.get('bucket_labels', []))
                if not bucket_names:
                    for key in pstats.keys():
                        if key.startswith('p') and key.endswith('_fill_pct') and key != 'main_fill_pct':
                            bucket_names.append(key.replace('_fill_pct', ''))

                if bucket_names:
                    bucket_names = sorted(
                        bucket_names,
                        key=lambda x: int(x.split('_')[0][1:]) if x.startswith('p') else 0,
                        reverse=True,
                    )

                for name in bucket_names:
                    fill_pct = pstats.get(f'{name}_fill_pct', 0.0)
                    bucket_fill_pcts.append(f"{fill_pct:.0f}%")

                main_fill_pct = pstats.get('main_fill_pct', 0.0)
                bucket_fill_pcts.append(f"{main_fill_pct:.0f}%")
        except Exception:
            bucket_fill_pcts = ['--']
    else:
        bucket_fill_pcts = ['--']
    
    # Format: MemK/Steps/P98/P95/P90/Main (for N=3)
    training_stats = f"{mem_k}k/{metrics.total_training_steps}/{'/'.join(bucket_fill_pcts)}"

    # Get Q-value range from the agent
    q_range = "N/A"
    if agent:
        try:
            min_q, max_q = agent.get_q_value_range()
            if not (np.isnan(min_q) or np.isnan(max_q)):
                q_range = f"[{min_q:.1f},{max_q:.1f}]"
        except Exception:
            q_range = "Error"

    # Additional diagnostics for troubleshooting
    agree_pct = agree_avg  # Use interval-averaged agreement instead of snapshot
    
    # Calculate average episode length since last metrics print
    avg_episode_length = 0.0
    with metrics.lock:
        if getattr(metrics, 'episode_length_count_interval', 0) > 0:
            avg_episode_length = metrics.episode_length_sum_interval / max(metrics.episode_length_count_interval, 1)
        # Reset interval counters
        metrics.episode_length_sum_interval = 0
        metrics.episode_length_count_interval = 0

    # Show effective epsilon with OVR marker
    try:
        effective_eps = metrics.get_effective_epsilon()
        eps_display = f"{effective_eps:>6.2f}"
        if metrics.override_epsilon:
            eps_display = f"{eps_display}*"
    except Exception:
        eps_display = f"{metrics.epsilon:>6.2f}"

    # Expert ratio with OVR marker
    xprt_display = f"{metrics.expert_ratio*100:>6.1f}%"
    if metrics.override_expert:
        xprt_display = f"{xprt_display}*"
    elif metrics.expert_mode:
        xprt_display = f"{xprt_display}!"

    # Rewards with 2 decimal places, with OVR marker for expert mode
    rwrd_display = f"{mean_reward:>6.2f}"
    subj_display = f"{mean_subj_reward:>6.2f}"
    obj_display = f"{mean_obj_reward:>6.2f}"
    dqn_display = f"{mean_dqn_reward:>6.2f}"
    if metrics.expert_mode:
        rwrd_display = f"{rwrd_display}!"
        subj_display = f"{subj_display}!"
        obj_display = f"{obj_display}!"
        dqn_display = f"{dqn_display}!"

    row = (
        f"{metrics.frame_count:>11,} {metrics.fps:>7.1f} {eps_display} "
        f"{xprt_display:>7} {rwrd_display:>7} {subj_display:>7} {obj_display:>7} {dqn_display:>7} {dqn1m_avg:>6.2f} {dqn5m_avg:>6.2f} {dqn5m_slopeM:>9.3f} {loss_avg:>10.6f} "
        f"{agree_pct*100:>6.1f}% "
        f"{avg_episode_length:>8.1f} {train_pct:>6.1f} "
        f"{metrics.client_count:04d} {display_level:>5.1f} "
        f"{avg_inference_time_ms:>7.2f} "
        f"{samples_per_sec:>9.0f} "
        f"{steps_per_sec:>8.1f} "
        f"{metrics.last_grad_norm:>8.3f} "
        f"{metrics.last_clip_delta:>6.3f} "
        f"{q_range:>12} {training_stats:>18}"
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
