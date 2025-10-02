# Expert Ratio Curriculum Strategy - Analysis & Recommendation

## Your Question
Should the expert ratio go to 0% after the 5M frame taper, or lock at 10%, or run DQN-only?

## Current Configuration
```python
expert_ratio_start = 0.95     # Start at 95%
expert_ratio_min = 0.10       # Configured minimum: 10%
expert_ratio_decay = 0.996    # Decay factor
expert_ratio_decay_steps = 10000  # Apply every 10k frames
```

### Current Decay Math
- Every 10k frames: ratio *= 0.996
- 95% → 10% takes: log(0.10/0.95) / log(0.996) ≈ **537 steps** ≈ **5.37M frames** ✅
- But: Code doesn't enforce `expert_ratio_min` floor! 🚨

## Three Strategy Options

### Option 1: Floor at 10% (Recommended ⭐)

**Pros:**
- ✅ **Safety net**: Expert handles rare edge cases DQN hasn't seen
- ✅ **Exploration diversity**: Expert provides different strategies than DQN
- ✅ **Catastrophic forgetting prevention**: Small expert signal maintains baseline
- ✅ **Non-stationary handling**: Later levels harder; expert helps
- ✅ **Research-backed**: DQfD, AlphaGo kept small expert ratios throughout

**Cons:**
- ⚠️ Final policy not "pure DQN" (but you can test with 'o' override)
- ⚠️ 10% expert still affects data distribution

**Implementation:**
```python
# In decay_expert_ratio()
metrics.expert_ratio = max(metrics.expert_ratio, RL_CONFIG.expert_ratio_min)
```

### Option 2: Decay to 0% (Pure DQN)

**Pros:**
- ✅ Agent must learn completely autonomously
- ✅ No expert bias in final policy
- ✅ Tests if DQN truly learned the task
- ✅ May discover novel strategies expert can't do

**Cons:**
- ❌ **Risk of catastrophic forgetting**: Performance may degrade
- ❌ **No safety net**: Bad habits can develop without correction
- ❌ **Local optima**: Can get stuck without expert diversity
- ❌ **Rare events**: Edge cases undertrained

**Implementation:**
```python
expert_ratio_min = 0.0  # Let it go to zero
```

### Option 3: Adaptive Floor (Advanced)

Floor based on DQN performance:
- If DQN 5M avg > 3.0: Floor at 5% (DQN is strong)
- If DQN 5M avg 2.5-3.0: Floor at 10% (DQN learning well)  
- If DQN 5M avg < 2.5: Floor at 20% (DQN needs more help)

**Implementation:**
```python
# In decay_expert_ratio()
dqn_perf = getattr(metrics, 'dqn_5m_avg', 0.0)
if dqn_perf > 3.0:
    floor = 0.05
elif dqn_perf > 2.5:
    floor = 0.10
else:
    floor = 0.20
metrics.expert_ratio = max(metrics.expert_ratio, floor)
```

---

## My Recommendation: **Floor at 10%** ⭐

### Reasoning for Tempest Specifically

1. **Game Complexity**
   - 16 levels with increasing difficulty
   - New patterns emerge at higher levels
   - Expert provides stability as DQN explores

2. **Expert System Characteristics**
   - Deterministic and consistent
   - Good safety baseline (avoids death)
   - Not necessarily optimal (DQN can beat it)
   - Fast and cheap (no inference cost)

3. **Risk Management**
   - 10% is enough for safety without dominating
   - Prevents complete policy collapse
   - Maintains exploration diversity
   - Can always test pure DQN with override ('o' hotkey)

4. **Research Best Practices**
   - DQfD: Kept 10-25% demonstrations throughout training
   - AlphaGo: Kept expert games in replay indefinitely
   - Most successful RL+IL systems maintain small expert signal

5. **Practical Benefits**
   - **Testing flexibility**: Can disable expert anytime with hotkey
   - **Diversity bonus synergy**: Expert explores different states
   - **PER benefit**: Expert experiences help populate rare state regions

### The Key Insight

The 10% expert isn't about "not trusting DQN" - it's about:
- **Exploration diversity**: Expert tries different things than ε-greedy
- **Safety baseline**: Prevents catastrophic failures
- **Curriculum regularization**: Maintains stability

Think of it like training wheels that gradually shrink but never fully disappear - they're not holding you back, they're providing a safety margin.

---

## Recommended Changes

### 1. Enforce the Floor (Required 🚨)

Your current code doesn't enforce `expert_ratio_min`! Add this:

```python
def decay_expert_ratio(current_step):
    """Update expert ratio periodically with floor enforcement."""
    # Skip decay if expert mode, override, or manual override is active
    if metrics.expert_mode or metrics.override_expert or getattr(metrics, 'manual_expert_override', False):
        return metrics.expert_ratio
    
    # DON'T auto-initialize to start value at frame 0 - respect loaded checkpoint values
    if current_step == 0 and (metrics.expert_ratio < 0 or metrics.expert_ratio > 1):
        metrics.expert_ratio = RL_CONFIG.expert_ratio_start
        metrics.last_decay_step = 0
        return metrics.expert_ratio

    step_interval = current_step // RL_CONFIG.expert_ratio_decay_steps

    # Apply scheduled decay when we cross an interval boundary
    if step_interval > metrics.last_decay_step:
        steps_to_apply = step_interval - metrics.last_decay_step
        for _ in range(steps_to_apply):
            metrics.expert_ratio *= RL_CONFIG.expert_ratio_decay
        metrics.last_decay_step = step_interval
    
    # ENFORCE MINIMUM FLOOR ← ADD THIS!
    metrics.expert_ratio = max(metrics.expert_ratio, RL_CONFIG.expert_ratio_min)

    return metrics.expert_ratio
```

### 2. Consider Adaptive Floor (Optional Enhancement)

If you want to experiment, you could make the floor adaptive:

```python
# After decay, before return
# Adaptive floor based on DQN performance
try:
    dqn_5m = float(getattr(metrics, 'dqn_5m_avg', 0.0))
    if dqn_5m > 3.0:
        adaptive_floor = max(0.05, RL_CONFIG.expert_ratio_min)
    elif dqn_5m > 2.5:
        adaptive_floor = max(0.10, RL_CONFIG.expert_ratio_min)
    else:
        adaptive_floor = max(0.15, RL_CONFIG.expert_ratio_min)
    metrics.expert_ratio = max(metrics.expert_ratio, adaptive_floor)
except Exception:
    # Fallback to configured minimum
    metrics.expert_ratio = max(metrics.expert_ratio, RL_CONFIG.expert_ratio_min)
```

### 3. Add Logging (Helpful for Analysis)

```python
# When floor is hit, log it once
if metrics.expert_ratio <= RL_CONFIG.expert_ratio_min and not getattr(metrics, '_expert_floor_logged', False):
    print(f"\n🎯 Expert ratio reached floor: {RL_CONFIG.expert_ratio_min:.1%} at {current_step:,} frames")
    metrics._expert_floor_logged = True
```

---

## Alternative: Easy A/B Testing

Add a config option to choose strategy:

```python
# In config.py RLConfigData
expert_floor_strategy: str = "fixed"  # Options: "fixed", "zero", "adaptive"
```

Then in decay function:
```python
# Apply floor based on strategy
strategy = getattr(RL_CONFIG, 'expert_floor_strategy', 'fixed')
if strategy == 'zero':
    # No floor - let it decay to 0
    pass
elif strategy == 'adaptive':
    # Use adaptive floor based on performance
    dqn_5m = float(getattr(metrics, 'dqn_5m_avg', 0.0))
    floor = 0.05 if dqn_5m > 3.0 else (0.10 if dqn_5m > 2.5 else 0.15)
    metrics.expert_ratio = max(metrics.expert_ratio, floor)
else:  # 'fixed' (default)
    metrics.expert_ratio = max(metrics.expert_ratio, RL_CONFIG.expert_ratio_min)
```

---

## Testing Your Choice

You can test any strategy at runtime:

### Test Pure DQN Anytime
```
Press 'o' - Override expert (forces 0% expert)
```
This lets you evaluate pure DQN performance without changing the curriculum.

### Test Different Floors
```
Press '+' repeatedly to manually set expert ratio
Press '-' to decrease
```
You can experiment with different values during training.

### Compare Strategies
Run multiple training sessions with different `expert_ratio_min`:
- Session 1: `expert_ratio_min = 0.10` (10% floor)
- Session 2: `expert_ratio_min = 0.05` (5% floor)  
- Session 3: `expert_ratio_min = 0.00` (pure DQN)

Compare final DQN performance, stability, and learning curves.

---

## My Final Recommendation

**Start with 10% floor, monitor, adjust if needed:**

1. **Immediate fix**: Add floor enforcement (your code is missing this!)
2. **Keep 10% floor** for first long run (5-10M frames)
3. **Monitor metrics**: Watch DQN rewards, stability, catastrophic forgetting
4. **Test pure DQN**: Use 'o' override periodically to check performance
5. **Adjust if needed**: 
   - If DQN dominates and expert hurts: Lower to 5%
   - If DQN struggles or forgets: Keep or raise to 15%
   - If very stable: Try pure DQN run

### Why 10% is the Sweet Spot

- **Small enough**: DQN drives 90% of behavior
- **Large enough**: Provides meaningful diversity and safety
- **Research-validated**: Proven in DQfD and similar approaches
- **Reversible**: Can always test pure DQN with override
- **Safe default**: Better to have it and not need it

---

## Curriculum Timeline Visualization

```
Frames:  0        1M       2M       3M       4M       5M       10M      15M
         |--------|--------|--------|--------|--------|--------|--------|
Expert:  95%      70%      50%      35%      20%      10%      10%      10%
         └────────────── Decay ──────────────┘        └─── Floor ───────┘
                                                              ↑
                                                         Stays at 10%
```

**Without floor enforcement (CURRENT BUG):**
```
Frames:  0        1M       2M       3M       4M       5M       10M      15M
         |--------|--------|--------|--------|--------|--------|--------|
Expert:  95%      70%      50%      35%      20%      10%      5%       1%
         └────────────────────────── Keeps Decaying ─────────────────────┘
                                                              ↑
                                                         Should floor!
```

---

## TL;DR

✅ **Fix the bug first**: Code doesn't enforce `expert_ratio_min` floor
✅ **Recommended**: Keep 10% floor for safety, diversity, and stability  
✅ **Test anytime**: Use 'o' override to evaluate pure DQN performance
✅ **Monitor & adjust**: Can always change based on observed behavior

**The 10% expert is insurance, not training wheels.** 🎯
