#!/usr/bin/env python3
"""Test that effective expert ratio matches what's used for decisions"""

import sys
sys.path.insert(0, 'Scripts')

from config import MetricsData

def test_effective_expert_ratio():
    """Verify effective expert ratio behavior"""
    metrics = MetricsData()
    
    print("Testing effective expert ratio...")
    
    # Test 1: Normal operation (no override, no expert mode)
    metrics.expert_ratio = 0.5
    metrics.override_expert = False
    metrics.expert_mode = False
    
    raw = metrics.get_expert_ratio()
    effective = metrics.get_effective_expert_ratio()
    
    assert raw == 0.5, f"Expected raw ratio 0.5, got {raw}"
    assert effective == 0.5, f"Expected effective ratio 0.5, got {effective}"
    print(f"✓ Normal: raw={raw:.2f}, effective={effective:.2f}")
    
    # Test 2: Override ON (forces DQN, so effective should be 0.0)
    metrics.expert_ratio = 0.5
    metrics.override_expert = True
    metrics.expert_mode = False
    
    raw = metrics.get_expert_ratio()
    effective = metrics.get_effective_expert_ratio()
    is_override_active = metrics.is_override_active()
    
    assert raw == 0.5, f"Expected raw ratio 0.5, got {raw}"
    assert effective == 0.0, f"Expected effective ratio 0.0 when override ON, got {effective}"
    assert is_override_active == True, f"Expected override active"
    print(f"✓ Override ON: raw={raw:.2f}, effective={effective:.2f}, override_active={is_override_active}")
    
    # Test 3: Expert mode ON (forces expert, ratio becomes 1.0)
    metrics.override_expert = False
    metrics.expert_mode = False
    metrics.expert_ratio = 0.5
    metrics.saved_expert_ratio = 0.5
    
    # Simulate toggling expert mode on
    metrics.expert_mode = True
    metrics.saved_expert_ratio = metrics.expert_ratio
    metrics.expert_ratio = 1.0
    
    raw = metrics.get_expert_ratio()
    effective = metrics.get_effective_expert_ratio()
    
    assert raw == 1.0, f"Expected raw ratio 1.0 when expert mode ON, got {raw}"
    assert effective == 1.0, f"Expected effective ratio 1.0 when expert mode ON, got {effective}"
    print(f"✓ Expert mode ON: raw={raw:.2f}, effective={effective:.2f}")
    
    # Test 4: Both override and expert mode ON (override wins, forces DQN)
    metrics.expert_mode = True
    metrics.expert_ratio = 1.0
    metrics.override_expert = True
    
    raw = metrics.get_expert_ratio()
    effective = metrics.get_effective_expert_ratio()
    is_override_active = metrics.is_override_active()
    
    assert raw == 1.0, f"Expected raw ratio 1.0, got {raw}"
    assert effective == 0.0, f"Expected effective ratio 0.0 when override ON, got {effective}"
    assert is_override_active == True, f"Expected override active"
    print(f"✓ Both ON (override wins): raw={raw:.2f}, effective={effective:.2f}, override_active={is_override_active}")
    
    print("\n✅ All tests passed!")
    print("\nSummary:")
    print("- Normal operation: effective = raw ratio")
    print("- Override ON: effective = 0.0 (always DQN)")
    print("- Expert mode ON: raw = 1.0, effective = 1.0 (always expert)")
    print("- Both ON: override wins, effective = 0.0 (always DQN)")

if __name__ == '__main__':
    test_effective_expert_ratio()
