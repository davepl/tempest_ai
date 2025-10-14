# Spinner Agreement Degradation Fix

## Problem Discovered

Spinner agreement was **steadily declining** while discrete agreement improved:

```
Frame       Agree%  SpinAgr%  CLoss
2,498,280   17.8%   72.2%     0.0365
2,626,544   63.2%   71.4%     0.0360
2,797,832   75.8%   69.3%     0.0347
2,968,256   73.0%   66.7%     0.0388
3,142,232   82.1%   63.6%     0.0390
3,185,880   82.9%   63.5%     0.0390
```

**Trends:**
- ❌ **Spinner Agreement**: 72.2% → 63.5% (declining ~9%)
- ❌ **CLoss (Continuous Loss)**: 0.0365 → 0.0390 (increasing +7%)
- ✅ **Discrete Agreement**: 17.8% → 82.9% (improving +65%)

**Inverse correlation:** As spinner agreement drops, continuous loss rises. The continuous head is **unlearning** expert behavior.

## Root Cause

**Zero gradient on DQN frames** caused catastrophic forgetting:

### The Bad Logic (Before):

```python
continuous_targets = continuous_actions.clone()  # Start with taken actions

# For DQN samples, use predicted continuous as target
# For expert samples, use taken actions
if torch_mask_dqn.any():
    continuous_targets[torch_mask_dqn] = continuous_pred[torch_mask_dqn]
```

**What happened:**
```python
# Expert frames (50% of data):
c_loss = MSE(pred, expert_action) * advantage_weight
# → Learning happens ✅

# DQN frames (50% of data):
c_loss = MSE(pred, pred) * advantage_weight = 0
# → Zero gradient, NO LEARNING ❌
```

### Why This Caused Degradation:

1. **Only 50% of data taught spinner** (expert frames only)
2. **DQN frames wasted** (zero gradient from target=prediction)
3. **As discrete performance improved:**
   - DQN frames got higher rewards
   - Higher advantage weights on DQN frames
   - But DQN continuous gradient still ZERO
   - **Gradient imbalance worsened**
4. **Shared network trunk** optimized for discrete success
5. **Spinner head became irrelevant** to discrete performance
6. **Catastrophic forgetting**: Expert spinner knowledge degraded

### The Deadly Feedback Loop:

```
Better discrete actions (DQN)
  ↓
Higher DQN frame rewards
  ↓
Higher advantage weights on DQN frames
  ↓
More backprop through DQN gradients
  ↓
But DQN continuous gradient = 0!
  ↓
Network trunk optimizes for discrete only
  ↓
Spinner head features marginalized
  ↓
Spinner agreement degrades
```

## The Fix

**Self-imitation learning** on high-reward DQN frames:

### New Logic:

```python
continuous_targets = continuous_actions.clone()

# For expert frames: always learn from expert (supervised learning)
# For DQN frames: self-imitation on successful experiences
if torch_mask_dqn.any():
    dqn_rewards = rewards[torch_mask_dqn]
    dqn_median = dqn_rewards.median()
    
    for each DQN frame:
        if reward >= median:
            # High reward: learn from taken action (self-imitation)
            target = taken_action
        else:
            # Low reward: zero gradient (avoid learning bad behavior)
            target = prediction
```

### What This Achieves:

1. **Expert frames (50%)**: Learn expert spinner control ✅
2. **High-reward DQN frames (25%)**: Reinforce successful spinner actions ✅
3. **Low-reward DQN frames (25%)**: Zero gradient (don't learn mistakes) ✅
4. **Total learning**: 75% of data now teaches spinner (was 50%)

### Why This Works:

1. **Self-imitation**: When DQN performs well, learns to repeat that success
2. **Selective learning**: Only imitates high-reward actions, filters noise
3. **Gradient balance**: DQN frames contribute meaningful spinner gradients
4. **Network coherence**: Spinner and discrete heads both benefit from DQN success
5. **No forgetting**: Continuous learning throughout training, not just from experts

## Expected Results

After fix:
- ✅ **Spinner agreement should stabilize** (~65-70%)
- ✅ **Then gradually improve** as DQN discovers good spinner control
- ✅ **CLoss should decrease** (better predictions)
- ✅ **Discrete agreement preserved** (~82-92%)

### Monitoring:

Watch for these trends:
- **SpinAgr% flattens then increases** (stopping degradation)
- **CLoss stabilizes then decreases** (improving predictions)
- **Positive correlation emerges**: Higher DQN reward → higher spinner agreement

## Validation

Compare before/after over 500k frames:

**Before fix (expected):**
- SpinAgr%: 72% → 63% (declining)
- CLoss: 0.036 → 0.039 (increasing)

**After fix (expected):**
- SpinAgr%: 63% → 65% → 68% (recovering)
- CLoss: 0.039 → 0.037 → 0.035 (improving)

## Technical Details

### Self-Imitation Threshold:

Used **median reward** as cutoff:
- **Advantage**: Adapts to performance level automatically
- **Simple**: No hyperparameter tuning needed
- **Balanced**: Always 50% of DQN frames learn, 50% zero gradient

### Alternative Approaches Considered:

1. **Full continuous Q-learning** (DDPG/TD3):
   - Requires separate critic network
   - More complex, slower
   - ❌ Rejected: Too much complexity

2. **Policy gradient** (REINFORCE):
   - Use reward directly to guide spinner
   - Moderate variance
   - ❌ Rejected: Need baseline, more hyperparams

3. **Fixed threshold** (e.g., reward > 5.0):
   - Simpler but brittle
   - Breaks if reward scale changes
   - ❌ Rejected: Median more robust

4. **Top-k% of DQN frames** (e.g., top 25%):
   - Similar to median but more aggressive
   - Could work but median is safer
   - ❌ Rejected: Start conservative with 50%

### Future Improvements:

If spinner agreement still doesn't improve enough:
1. **Increase self-imitation threshold** to top 75% of DQN frames
2. **Add curiosity bonus** for spinner exploration
3. **Separate spinner optimizer** with different learning rate
4. **Continuous curriculum**: Start with expert-only, gradually add DQN

---
**Date:** October 14, 2025  
**Files Modified:** `Scripts/aimodel.py` (continuous target generation)  
**Impact:** Prevents spinner degradation by enabling learning from successful DQN experiences
