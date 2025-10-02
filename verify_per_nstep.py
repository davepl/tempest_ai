#!/usr/bin/env python3
"""Verify that PER and N-Step are both active in the current configuration"""

import sys
sys.path.insert(0, 'Scripts')

from config import RL_CONFIG

def verify_per_nstep_config():
    """Check configuration and verify both features are enabled"""
    
    print("=" * 70)
    print("PER + N-Step Configuration Verification")
    print("=" * 70)
    
    # Check N-Step configuration
    n_step = getattr(RL_CONFIG, 'n_step', 1)
    n_step_enabled = getattr(RL_CONFIG, 'n_step_enabled', True)
    gamma = getattr(RL_CONFIG, 'gamma', 0.99)
    
    print("\n📊 N-Step Configuration:")
    print(f"  n_step:         {n_step}")
    print(f"  n_step_enabled: {n_step_enabled}")
    print(f"  gamma:          {gamma}")
    
    if n_step > 1 and n_step_enabled:
        print(f"  ✅ N-Step is ACTIVE (will accumulate {n_step} steps)")
        gamma_boot = gamma ** n_step
        print(f"  ✅ Bootstrap discount: γ^{n_step} = {gamma_boot:.6f}")
    elif n_step == 1:
        print(f"  ⚠️  N-Step is set to 1 (effectively single-step)")
    else:
        print(f"  ❌ N-Step is DISABLED")
    
    # Check PER configuration
    use_per = getattr(RL_CONFIG, 'use_per', False)
    per_alpha = getattr(RL_CONFIG, 'per_alpha', 0.6)
    per_beta_start = getattr(RL_CONFIG, 'per_beta_start', 0.4)
    per_beta_end = getattr(RL_CONFIG, 'per_beta_end', 1.0)
    per_eps = getattr(RL_CONFIG, 'per_eps', 1e-6)
    
    print("\n🎯 PER Configuration:")
    print(f"  use_per:        {use_per}")
    print(f"  per_alpha:      {per_alpha} (prioritization exponent)")
    print(f"  per_beta:       {per_beta_start} → {per_beta_end} (importance sampling)")
    print(f"  per_eps:        {per_eps} (minimum priority)")
    
    if use_per:
        print(f"  ✅ PER is ACTIVE")
        print(f"  ✅ Prioritization strength: {per_alpha:.1f} (0=uniform, 1=full)")
        print(f"  ✅ Importance sampling: β anneals {per_beta_start}→{per_beta_end}")
    else:
        print(f"  ❌ PER is DISABLED (using uniform sampling)")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    
    both_active = (n_step > 1 and n_step_enabled and use_per)
    
    if both_active:
        print("✅ ✅ ✅  BOTH N-Step AND PER are ACTIVE!")
        print("\nData Flow:")
        print("  Game → N-Step Buffer → PER Buffer → Training")
        print("\nBenefits:")
        print("  • N-Step: Faster credit assignment (multi-step returns)")
        print("  • PER: Better sample efficiency (prioritized sampling)")
        print("  • Combined: Optimal sample-efficient deep RL!")
        
        print("\nExpected Behavior:")
        print(f"  1. Raw transitions accumulated into {n_step}-step returns")
        print(f"  2. N-step experiences stored in PER with max priority")
        print(f"  3. Training samples high-TD-error experiences more often")
        print(f"  4. Targets computed as: R_{{t:t+{n_step}}} + γ^{n_step} * Q(s_{{t+{n_step}}}, a*)")
        
    elif n_step > 1 and n_step_enabled and not use_per:
        print("⚠️  N-Step is active but PER is disabled")
        print("   → Using standard uniform replay buffer with n-step returns")
        
    elif use_per and (n_step == 1 or not n_step_enabled):
        print("⚠️  PER is active but N-Step is disabled")
        print("   → Using prioritized sampling with single-step returns")
        
    else:
        print("❌ Neither N-Step nor PER is active")
        print("   → Using standard uniform replay with single-step returns")
    
    print("=" * 70)
    
    return both_active

if __name__ == '__main__':
    both_active = verify_per_nstep_config()
    sys.exit(0 if both_active else 1)
