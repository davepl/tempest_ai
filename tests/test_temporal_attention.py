#!/usr/bin/env python3
"""Tests for temporal enemy tokens, stack-skip assembly, and diagnose_attention()."""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Scripts"))

from aimodel import RainbowNet, RainbowAgent  # type: ignore
from config import SERVER_CONFIG, RL_CONFIG    # type: ignore


def _make_net():
    fs = max(1, RL_CONFIG.frame_stack)
    stacked = SERVER_CONFIG.params_count * fs
    return RainbowNet(stacked), fs, SERVER_CONFIG.params_count, stacked


# ── Temporal token shapes ────────────────────────────────────────────────────

def test_temporal_token_shape():
    """Enemy tokens should be (B, 7*fs, 15) with temporal index as feature 14."""
    net, fs, raw, stacked = _make_net()
    B = 2
    state = torch.randn(B, stacked)
    tokens, mask = net._build_enemy_tokens(state)
    assert tokens.shape == (B, 7 * fs, 15), f"Expected (B, {7*fs}, 15), got {tokens.shape}"
    assert mask.shape == (B, 7 * fs), f"Expected (B, {7*fs}), got {mask.shape}"


def test_temporal_index_values():
    """Temporal index (feature 14) should be 0.0 for current frame, ascending to 1.0."""
    net, fs, raw, stacked = _make_net()
    state = torch.randn(1, stacked)
    tokens, _ = net._build_enemy_tokens(state)
    # Feature 14 is the temporal index
    for f in range(fs):
        expected = f / max(1, fs - 1) if fs > 1 else 0.0
        actual = tokens[0, f * 7, 14].item()
        assert abs(actual - expected) < 1e-5, (
            f"Frame {f}: temporal_idx expected {expected:.4f}, got {actual:.4f}"
        )


def test_temporal_tokens_use_correct_frame_data():
    """Each frame's enemy tokens should reflect that frame's data, not frame 0's."""
    net, fs, raw, stacked = _make_net()
    if fs < 2:
        return  # nothing to test with single frame
    state = torch.zeros(1, stacked)
    # Put a distinctive depth in frame 0 slot 0 vs frame 1 slot 0
    state[0, 135] = 0.5        # frame 0, slot 0 depth
    state[0, raw + 135] = 0.8  # frame 1, slot 0 depth
    tokens, _ = net._build_enemy_tokens(state)
    # After per-frame sort, the active enemy lands in first slot of each frame's block.
    # Depth is feature index 7 in the 14-feature base (decoded(6) + seg(1) + depth(1)).
    depth_f0 = tokens[0, 0, 7].item()
    depth_f1 = tokens[0, 7, 7].item()
    assert abs(depth_f0 - 0.5) < 1e-5, f"Frame 0 depth: expected 0.5, got {depth_f0}"
    assert abs(depth_f1 - 0.8) < 1e-5, f"Frame 1 depth: expected 0.8, got {depth_f1}"


# ── Stack-skip assembly ─────────────────────────────────────────────────────

def test_stack_skip_frame_ordering():
    """_build_stacked_state should pick frames at t, t-skip, t-2*skip, ..."""
    # Import the server's stacking function
    from socket_server import SocketServer
    from collections import deque

    raw = SERVER_CONFIG.params_count
    fs = RL_CONFIG.frame_stack
    skip = RL_CONFIG.frame_stack_skip
    hist_len = 1 + (fs - 1) * skip if fs > 1 else 1

    cs = {
        "frame_history": deque(maxlen=hist_len),
        "frame_stack": fs,
        "frame_stack_skip": skip,
        "raw_state_size": raw,
    }

    # Feed numbered frames: frame i has all values = float(i)
    total_frames = hist_len + 5  # feed extra to fill history
    for i in range(total_frames):
        raw_state = np.full(raw, float(i), dtype=np.float32)
        stacked = SocketServer._build_stacked_state(raw_state, cs)

    # After feeding 'total_frames - 1' as last frame:
    last = total_frames - 1
    # stacked[0:raw] should be the current frame (last)
    assert stacked[0] == float(last), f"Frame 0 (current) should be {last}, got {stacked[0]}"
    if fs > 1:
        # stacked[raw:2*raw] should be frame at t - skip
        expected_f1 = float(last - skip)
        assert stacked[raw] == expected_f1, (
            f"Frame 1 should be {expected_f1}, got {stacked[raw]}"
        )
        if fs > 2:
            expected_f2 = float(last - 2 * skip)
            assert stacked[2 * raw] == expected_f2, (
                f"Frame 2 should be {expected_f2}, got {stacked[2 * raw]}"
            )


def test_stack_skip_zero_padding_early():
    """Early frames without enough history should zero-pad older slots."""
    from socket_server import SocketServer
    from collections import deque

    raw = SERVER_CONFIG.params_count
    fs = RL_CONFIG.frame_stack
    skip = RL_CONFIG.frame_stack_skip
    hist_len = 1 + (fs - 1) * skip if fs > 1 else 1

    cs = {
        "frame_history": deque(maxlen=hist_len),
        "frame_stack": fs,
        "frame_stack_skip": skip,
        "raw_state_size": raw,
    }

    # Feed only 1 frame
    raw_state = np.ones(raw, dtype=np.float32)
    stacked = SocketServer._build_stacked_state(raw_state, cs)
    assert stacked[0] == 1.0, "Current frame should be 1.0"
    if fs > 1:
        # All older slots should be zero-padded
        for f in range(1, fs):
            val = stacked[f * raw]
            assert val == 0.0, f"Frame {f} should be zero-padded, got {val}"


# ── diagnose_attention() crash regression ────────────────────────────────────

def test_diagnose_attention_no_crash():
    """diagnose_attention() must not raise IndexError with frame_stack > 1.

    Regression: enemy_segs was (B, 7) but loop iterated over S = 7 * frame_stack.
    """
    fs = max(1, RL_CONFIG.frame_stack)
    stacked = SERVER_CONFIG.params_count * fs
    agent = RainbowAgent(stacked)

    # Fill replay with enough samples for diagnose to run
    num = 300
    for _ in range(num):
        s = np.random.randn(stacked).astype(np.float32)
        ns = np.random.randn(stacked).astype(np.float32)
        agent.memory.add(s, 0, 0.1, ns, False, 1, False)

    # This should NOT raise
    result = agent.diagnose_attention(num_samples=64)
    assert isinstance(result, str)
    assert "LANE-CROSS-ATTENTION DIAGNOSTICS" in result
    # With temporal tokens, should include temporal section
    if fs > 1:
        assert "Temporal attention distribution" in result


def test_diagnose_attention_temporal_section():
    """Temporal distribution section should report per-frame percentages."""
    fs = max(1, RL_CONFIG.frame_stack)
    if fs <= 1:
        return
    stacked = SERVER_CONFIG.params_count * fs
    agent = RainbowAgent(stacked)

    num = 300
    for _ in range(num):
        s = np.random.randn(stacked).astype(np.float32)
        ns = np.random.randn(stacked).astype(np.float32)
        agent.memory.add(s, 0, 0.1, ns, False, 1, False)

    result = agent.diagnose_attention(num_samples=64)
    # Should contain per-frame labels
    assert "current" in result
    assert "t-1" in result
