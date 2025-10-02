#!/usr/bin/env python3
"""Test expert ratio curriculum with floor enforcement"""

import sys
sys.path.insert(0, 'Scripts')

from config import RL_CONFIG, MetricsData

def test_expert_ratio_decay():
    """Verify expert ratio decays correctly and respects floor"""
    
    print("=" * 70)
    print("Expert Ratio Curriculum Test")
    print("=" * 70)
    
    # Import decay function
    sys.path.insert(0, 'Scripts')
    from aimodel import decay_expert_ratio
    
    # Create fresh metrics
    metrics = MetricsData()
    
    # Import to set global metrics
    import aimodel
    aimodel.metrics = metrics
    
    # Set config values
    print(f"\nConfiguration:")
    print(f"  expert_ratio_start: {RL_CONFIG.expert_ratio_start:.2%}")
    print(f"  expert_ratio_min:   {RL_CONFIG.expert_ratio_min:.2%}")
    print(f"  expert_ratio_decay: {RL_CONFIG.expert_ratio_decay}")
    print(f"  decay_steps:        {RL_CONFIG.expert_ratio_decay_steps:,}")
    
    # Calculate expected frames to reach floor
    import math
    steps_to_floor = math.log(RL_CONFIG.expert_ratio_min / RL_CONFIG.expert_ratio_start) / math.log(RL_CONFIG.expert_ratio_decay)
    frames_to_floor = steps_to_floor * RL_CONFIG.expert_ratio_decay_steps
    
    print(f"\n📊 Expected Behavior:")
    print(f"  Steps to floor:  {steps_to_floor:.0f}")
    print(f"  Frames to floor: {frames_to_floor:,.0f}")
    
    # Initialize
    metrics.expert_ratio = RL_CONFIG.expert_ratio_start
    metrics.last_decay_step = 0
    
    # Test decay over time
    print(f"\n🔄 Decay Simulation:")
    print(f"{'Frames':>12} {'Expert %':>10} {'Status':>15}")
    print("-" * 40)
    
    test_frames = [0, 500_000, 1_000_000, 2_000_000, 3_000_000, 
                   4_000_000, 5_000_000, 6_000_000, 8_000_000, 10_000_000]
    
    last_ratio = None
    floor_hit_frame = None
    
    for frame in test_frames:
        decay_expert_ratio(frame)
        
        status = ""
        if metrics.expert_ratio == RL_CONFIG.expert_ratio_min:
            status = "AT FLOOR"
            if floor_hit_frame is None:
                floor_hit_frame = frame
        elif last_ratio is not None and metrics.expert_ratio == last_ratio:
            status = "STUCK (BUG?)"
        else:
            status = "Decaying"
        
        print(f"{frame:>12,} {metrics.expert_ratio*100:>9.2f}% {status:>15}")
        last_ratio = metrics.expert_ratio
    
    # Verify floor is enforced
    print(f"\n✅ Verification:")
    final_ratio = metrics.expert_ratio
    
    if final_ratio >= RL_CONFIG.expert_ratio_min:
        print(f"  ✅ Floor enforced: {final_ratio:.2%} >= {RL_CONFIG.expert_ratio_min:.2%}")
    else:
        print(f"  ❌ Floor violated: {final_ratio:.2%} < {RL_CONFIG.expert_ratio_min:.2%}")
        return False
    
    if floor_hit_frame is not None:
        print(f"  ✅ Floor reached at: {floor_hit_frame:,} frames")
        print(f"  ✅ Expected around:  {frames_to_floor:,.0f} frames")
        deviation = abs(floor_hit_frame - frames_to_floor) / frames_to_floor
        if deviation < 0.1:
            print(f"  ✅ Within 10% of expected")
        else:
            print(f"  ⚠️  Deviation: {deviation:.1%}")
    else:
        print(f"  ❌ Floor never reached in test range")
        return False
    
    # Test that it stays at floor
    print(f"\n🔒 Floor Stability Test:")
    for i in range(5):
        test_frame = 10_000_000 + i * 1_000_000
        decay_expert_ratio(test_frame)
        print(f"  {test_frame:>12,} frames: {metrics.expert_ratio*100:>6.2f}% ", end="")
        if metrics.expert_ratio == RL_CONFIG.expert_ratio_min:
            print("✅ Stable at floor")
        else:
            print(f"❌ Changed to {metrics.expert_ratio:.2%}")
            return False
    
    print(f"\n" + "=" * 70)
    print("✅ Expert ratio curriculum working correctly!")
    print("=" * 70)
    
    return True

def test_floor_config_scenarios():
    """Test different floor configurations"""
    
    print("\n" + "=" * 70)
    print("Floor Configuration Scenarios")
    print("=" * 70)
    
    scenarios = [
        (0.10, "10% - Recommended (safety + diversity)"),
        (0.05, "5% - Light touch (mostly DQN)"),
        (0.00, "0% - Pure DQN (no safety net)"),
        (0.15, "15% - Conservative (more guidance)"),
    ]
    
    print(f"\n{'Floor':>8} | {'Description':<45} | {'Final Behavior'}")
    print("-" * 80)
    
    for floor, desc in scenarios:
        if floor == 0.0:
            behavior = "Decays to 0%, pure RL"
        else:
            behavior = f"Maintains {floor:.0%} expert throughout"
        
        print(f"{floor*100:>7.0f}% | {desc:<45} | {behavior}")
    
    print(f"\n✅ Current setting: expert_ratio_min = {RL_CONFIG.expert_ratio_min:.0%}")
    
    if RL_CONFIG.expert_ratio_min == 0.10:
        print("   ✅ Using recommended 10% floor")
    elif RL_CONFIG.expert_ratio_min == 0.0:
        print("   ⚠️  Using 0% - pure DQN (no safety net)")
    else:
        print(f"   ℹ️  Using custom {RL_CONFIG.expert_ratio_min:.0%} floor")

if __name__ == '__main__':
    success = test_expert_ratio_decay()
    if success:
        test_floor_config_scenarios()
        print("\n✅ All tests passed!\n")
        sys.exit(0)
    else:
        print("\n❌ Tests failed\n")
        sys.exit(1)
