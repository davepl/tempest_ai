#!/usr/bin/env python3
"""
Quick test to verify verbose mode toggle functionality
"""

import sys
sys.path.insert(0, 'Scripts')

from config import metrics

def test_verbose_mode():
    """Test verbose mode toggle"""
    print("Testing verbose mode toggle...")
    
    # Initial state should be False
    assert metrics.verbose_mode == False, "Initial verbose_mode should be False"
    print("✓ Initial state: verbose_mode = False")
    
    # Toggle ON
    metrics.toggle_verbose_mode(None)
    assert metrics.verbose_mode == True, "After first toggle, verbose_mode should be True"
    print("✓ After toggle ON: verbose_mode = True")
    
    # Toggle OFF
    metrics.toggle_verbose_mode(None)
    assert metrics.verbose_mode == False, "After second toggle, verbose_mode should be False"
    print("✓ After toggle OFF: verbose_mode = False")
    
    # Toggle ON again
    metrics.toggle_verbose_mode(None)
    assert metrics.verbose_mode == True, "After third toggle, verbose_mode should be True"
    print("✓ After toggle ON again: verbose_mode = True")
    
    print("\n✅ All tests passed!")

if __name__ == "__main__":
    test_verbose_mode()
