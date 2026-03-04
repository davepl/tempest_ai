#!/usr/bin/env python3
"""Runs Lua-based avoidance regression tests."""

import os
import subprocess
import sys

TEST_SCRIPT = os.path.join(os.path.dirname(__file__), 'lua', 'test_avoidance_logic.lua')

def test_lua_avoidance_behavior():
    result = subprocess.run(['lua', TEST_SCRIPT], cwd=os.path.dirname(__file__), capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
    assert result.returncode == 0, 'Lua avoidance regression failed'
