#!/usr/bin/env python3
"""Runs Lua-based state extraction regression tests."""

import os
import subprocess
import sys

TEST_SCRIPT = os.path.join(os.path.dirname(__file__), "lua", "test_state_extraction.lua")


def test_lua_state_extraction_behavior():
    result = subprocess.run(["lua", TEST_SCRIPT], cwd=os.path.dirname(__file__), capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
    assert result.returncode == 0, "Lua state extraction regression failed"
