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
import math
import numpy as np
from typing import Optional, List, Dict, Any
from collections import deque

# Import from config.py
from config import metrics, IS_INTERACTIVE, RL_CONFIG

# Add a counter to track the number of rows printed
row_counter = 0

# Rolling window for DQN reward over last 5M frames
DQN_WINDOW_FRAMES = 5_000_000
_dqn_window = deque()  # entries: (episode_dqn_reward: float, episode_length: int)
_dqn_window_frames = 0

# Rolling window for DQN reward over last 1M frames
DQN1M_WINDOW_FRAMES = 1_000_000
_dqn1m_window = deque()  # entries: (episode_dqn_reward: float, episode_length: int)
_dqn1m_window_frames = 0

# Short-term DQN trend based on recent raw values
DQN_TREND_POINTS = 8
_dqn_trend_points = deque(maxlen=DQN_TREND_POINTS)

def _update_dqn_window(dqn_reward: float, episode_length: int):
    """Add a completed episode to the 5M-frames rolling window.
    
    Args:
        dqn_reward: Total DQN reward for the episode
        episode_length: Number of frames in the episode
    """
    global _dqn_window_frames
    
    # Skip if episode length is invalid
    if episode_length <= 0:
        return
    
    # Append new episode
    _dqn_window.append((float(dqn_reward), int(episode_length)))
    _dqn_window_frames += episode_length

    # Trim window to last 5M frames (remove full episodes only)
    while _dqn_window and _dqn_window_frames > DQN_WINDOW_FRAMES:
        oldest_reward, oldest_length = _dqn_window.popleft()
        _dqn_window_frames -= oldest_length

def _update_dqn1m_window(dqn_reward: float, episode_length: int):
    """Add a completed episode to the 1M-frames rolling window.
    
    Args:
        dqn_reward: Total DQN reward for the episode
        episode_length: Number of frames in the episode
    """
    global _dqn1m_window_frames
    
    # Skip if episode length is invalid
    if episode_length <= 0:
        return
    
    # Append new episode
    _dqn1m_window.append((float(dqn_reward), int(episode_length)))
    _dqn1m_window_frames += episode_length

    # Trim window to last 1M frames (remove full episodes only)
    while _dqn1m_window and _dqn1m_window_frames > DQN1M_WINDOW_FRAMES:
        oldest_reward, oldest_length = _dqn1m_window.popleft()
        _dqn1m_window_frames -= oldest_length

def _compute_dqn_window_stats():
    """Compute average DQN reward for the 5M-frame window.
    
    Returns average DQN reward per episode, weighted by episode length.
    """
    if not _dqn_window or _dqn_window_frames <= 0:
        return 0.0

    # Sum total DQN rewards and compute average per episode
    total_dqn_reward = sum(reward for reward, _ in _dqn_window)
    num_episodes = len(_dqn_window)
    avg = total_dqn_reward / num_episodes if num_episodes > 0 else 0.0
    return avg

def _compute_dqn1m_window_stats():
    """Compute weighted average for the 1M-frame window.
    
    Returns average DQN reward per episode, weighted by episode length.
    """
    if not _dqn1m_window or _dqn1m_window_frames <= 0:
        return 0.0

    # Sum total DQN rewards and total frames
    total_dqn_reward = sum(reward for reward, _ in _dqn1m_window)
    
    # Average reward per frame, then scale to per-episode basis
    # Actually, we want average episode reward, so just compute that directly
    num_episodes = len(_dqn1m_window)
    avg = total_dqn_reward / num_episodes if num_episodes > 0 else 0.0
    return avg

def add_episode_to_dqn1m_window(dqn_reward: float, episode_length: int):
    """Public function to add completed episodes to the 1M-frame DQN window.
    
    Call this when an episode completes to track DQN performance over the last 1M frames.
    
    Args:
        dqn_reward: Total DQN reward for the completed episode
        episode_length: Number of frames in the completed episode
    """
    try:
        _update_dqn1m_window(dqn_reward, episode_length)
    except Exception:
        pass

def add_episode_to_dqn5m_window(dqn_reward: float, episode_length: int):
    """Public function to add completed episodes to the 5M-frame DQN window.
    
    Call this when an episode completes to track DQN performance over the last 5M frames.
    
    Args:
        dqn_reward: Total DQN reward for the completed episode
        episode_length: Number of frames in the completed episode
    """
    try:
        _update_dqn_window(dqn_reward, episode_length)
    except Exception:
        pass

def _update_dqn_trend(frame_count: int, dqn_raw: float):
    """Maintain a small buffer of recent (frame, raw_dqn_reward) points."""
    try:
        frame = int(frame_count)
        value = float(dqn_raw)
        if not math.isfinite(value):
            return
    except Exception:
        return
    _dqn_trend_points.append((frame, value))

def _compute_dqn_trend_slope():
    """Compute linear trend (per million frames) over the recent raw DQN values."""
    if len(_dqn_trend_points) < 2:
        return 0.0

    base_frame = _dqn_trend_points[0][0]
    xs = [frame - base_frame for frame, _ in _dqn_trend_points]
    ys = [value for _, value in _dqn_trend_points]

    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xx = sum(x * x for x in xs)
    sum_xy = sum(x * y for x, y in zip(xs, ys))
    n = float(len(xs))

    denom = n * sum_xx - (sum_x * sum_x)
    if denom == 0.0:
        return 0.0

    slope_per_frame = (n * sum_xy - sum_x * sum_y) / denom
    return slope_per_frame * 1_000_000.0

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
        f"{'Frame':>11} {'FPS':>7} {'Epsi':>9} {'Xprt':>9} "
        f"{'Rwrd':>9} {'Subj':>9} {'Obj':>9} {'DQN':>9} {'DQN1M':>9} {'DQN5M':>9} {'DQNTrend':>9} {'Loss':>10} "
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

    # Compute DQN window stats (episodes are added via add_episode_to_dqn*m_window)
    try:
        dqn1m_avg = _compute_dqn1m_window_stats()
    except Exception:
        dqn1m_avg = 0.0
    
    try:
        dqn5m_avg = _compute_dqn_window_stats()
    except Exception:
        dqn5m_avg = 0.0

    # Publish DQN5M stats to global metrics for gating logic elsewhere
    try:
        with metrics.lock:
            metrics.dqn5m_avg = float(dqn5m_avg)
    except Exception:
        pass

    # steps_per_1k_frames already computed above using the same interval counts
    
    # Calculate training steps since last target update
    steps_since_target_update = metrics.total_training_steps - getattr(metrics, 'last_target_update_step', 0)
    
    # Format training stats: MemK/Steps/P98-100%/P95-98%/P90-95%/Main%
    mem_k = getattr(metrics, 'memory_buffer_size', 0) // 1000
    
    # Get partition stats if agent is available
    # Show fill % for Agent vs Expert buffers
    bucket_fill_pcts = []
    if agent and hasattr(agent, 'memory') and hasattr(agent.memory, 'get_partition_stats'):
        try:
            pstats = agent.memory.get_partition_stats()
            # Check for new StratifiedReplayBuffer stats
            if 'dqn' in pstats and 'expert' in pstats:
                dqn_count = pstats['dqn']
                expert_count = pstats['expert']
                total = pstats.get('total_size', 1)
                if total == 0: total = 1
                
                dqn_pct = (dqn_count / total) * 100
                expert_pct = (expert_count / total) * 100
                
                bucket_fill_pcts.append(f"A:{dqn_pct:.0f}%")
                bucket_fill_pcts.append(f"E:{expert_pct:.0f}%")
            
            # Fallback to old logic if not stratified
            elif not pstats.get('priority_buckets_enabled', False):
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

    def _safe_inverse(scale_value):
        try:
            scale_float = float(scale_value)
            if scale_float == 0.0:
                return 1.0
            return 1.0 / scale_float
        except Exception:
            return 1.0

    inv_obj = _safe_inverse(getattr(RL_CONFIG, 'obj_reward_scale', 1.0))
    inv_subj = _safe_inverse(getattr(RL_CONFIG, 'subj_reward_scale', 1.0))

    # Use fixed multiplier based on objective scale to ensure stability.
    # Dynamic calculation causes fluctuations when hidden rewards (like death penalty) 
    # affect the denominator (mean_reward) but not the numerator (total_raw).
    reward_multiplier = inv_obj if inv_obj != 1.0 else (inv_subj if inv_subj != 1.0 else 1.0)

    # Get Q-value range from the agent
    q_range = "N/A"
    if agent:
        try:
            min_q, max_q = agent.get_q_value_range()
            if not (np.isnan(min_q) or np.isnan(max_q)):
                # Apply multiplier to show in point units
                min_q_scaled = min_q * reward_multiplier
                max_q_scaled = max_q * reward_multiplier
                q_range = f"[{min_q_scaled:.1f},{max_q_scaled:.1f}]"
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

    obj_raw = mean_obj_reward * inv_obj
    subj_raw = mean_subj_reward * inv_subj
    total_raw = obj_raw + subj_raw

    dqn_raw = mean_dqn_reward * reward_multiplier
    
    # DQN1M and DQN5M are already stored as scaled rewards in the window buffers
    # (passed from aimodel.py -> socket_server.py where scaling is applied)
    # So we just need to apply the multiplier to convert back to display units
    dqn1m_raw = dqn1m_avg * reward_multiplier
    dqn5m_raw = dqn5m_avg * reward_multiplier

    dqn_trend_perM = 0.0
    try:
        _update_dqn_trend(metrics.frame_count, dqn_raw)
        dqn_trend_perM = _compute_dqn_trend_slope()
    except Exception:
        dqn_trend_perM = 0.0

    try:
        with metrics.lock:
            metrics.dqn_trend_perM = float(dqn_trend_perM)
    except Exception:
        pass

    def _format_reward(value, width=9, marker=""):
        try:
            val = float(value)
            if not math.isfinite(val):
                val = 0.0
        except Exception:
            val = 0.0
        base = f"{val:.0f}"
        if marker:
            base = f"{base}{marker}"
        return base.rjust(width)

    # Show effective epsilon with OVR marker (percentage with no decimals)
    try:
        effective_eps = metrics.get_effective_epsilon()
        eps_percent = float(effective_eps) * 100.0
    except Exception:
        eps_percent = float(metrics.epsilon) * 100.0
    eps_display = f"{eps_percent:.0f}%".rjust(9)
    if metrics.override_epsilon:
        eps_display = (eps_display.rstrip() + "*").rjust(9)

    # Expert ratio with OVR marker (percentage with no decimals)
    xprt_percent = metrics.expert_ratio * 100.0
    xprt_display = f"{xprt_percent:.0f}%".rjust(9)
    if metrics.override_expert:
        xprt_display = (xprt_display.rstrip() + "*").rjust(9)
    elif metrics.expert_mode:
        xprt_display = (xprt_display.rstrip() + "!").rjust(9)

    # Rewards shown in raw points (no decimals)
    expert_marker = "!" if metrics.expert_mode else ""
    rwrd_display = _format_reward(total_raw, marker=expert_marker)
    subj_display = _format_reward(subj_raw, marker=expert_marker)
    obj_display = _format_reward(obj_raw, marker=expert_marker)
    dqn_display = _format_reward(dqn_raw, marker=expert_marker)
    dqn1m_display = _format_reward(dqn1m_raw)
    dqn5m_display = _format_reward(dqn5m_raw)

    row = (
        f"{metrics.frame_count:>11,} {metrics.fps:>7.1f} {eps_display} "
        f"{xprt_display} {rwrd_display} {subj_display} {obj_display} {dqn_display} {dqn1m_display} {dqn5m_display} {dqn_trend_perM:>9.3f} {loss_avg:>10.6f} "
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
