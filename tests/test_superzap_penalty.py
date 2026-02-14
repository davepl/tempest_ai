#!/usr/bin/env python3
"""Unit tests for superzap penalty bookkeeping."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Scripts'))

from socket_server import SocketServer  # type: ignore  # pylint: disable=import-error
from config import RL_CONFIG  # type: ignore  # pylint: disable=import-error


def test_zap_penalty_accumulates_and_drains():
    state: dict[str, float] = {}
    original_penalty = getattr(RL_CONFIG, 'superzap_block_penalty', -0.05)
    try:
        RL_CONFIG.superzap_block_penalty = -0.1
        SocketServer._record_zap_block_penalty(state)
        SocketServer._record_zap_block_penalty(state)
        assert state['pending_zap_penalty'] == -0.2
        drained = SocketServer._drain_zap_block_penalty(state)
        assert drained == -0.2
        assert state['pending_zap_penalty'] == 0.0
    finally:
        RL_CONFIG.superzap_block_penalty = original_penalty
