# Spinner Agreement Plateau Fix

## Problem: Spinner Stuck at 64%

After the initial self-imitation fix, spinner agreement improved from 61% → 64.6% but then **plateaued**:

```
Frame       Agree%  SpinAgr%  CLoss
1,481,816   91.9%   63.8%     0.0442
1,875,568   93.1%   64.2%     0.0422
2,217,512   93.4%   63.9%     0.0399
2,560,128   93.4%   64.1%     0.0384
2,773,392   93.3%   64.6%     0.0378
```

**Observations:**
- ✅ Discrete agreement excellent and stable (~93%)
- ❌ Spinner agreement plateaued at ~64%
- ⚠️ CLoss still decreasing (good) but spinner agreement not following

**Gap:** There's likely room for improvement to 70%+, but learning stalled.

---

## Root Cause: Conservative Self-Imitation Threshold

The initial fix used **median** (50th percentile) as the threshold:
- **Top 50%** of DQN frames: Learn from taken action ✅
- **Bottom 50%** of DQN frames: Zero gradient ❌

### Why This Was Too Conservative:

1. **Median might be high**: If median reward is good (e.g., 3.0), only excellent frames (>3.0) teach
2. **Missing "decent" frames**: Frames with reward 2.5-3.0 are still useful but ignored
3. **Limited data**: Only learning from Expert (50%) + DQN top 50% (25%) = **75% total**
4. **Plateau effect**: Not enough diversity in training signal

### The Math:
```
With 50% expert ratio and median DQN threshold:
- Expert frames: 50% of batch → learn spinner ✅
- DQN high-reward: 25% of batch → learn spinner ✅
- DQN low-reward: 25% of batch → zero gradient ❌

Total learning: 75% of data
Wasted: 25% of data
```

---

## The Fix: More Aggressive Learning

Changed threshold from **median (50th)** to **25th percentile**:

### Before:
```python
dqn_median = dqn_rewards.median()  # Top 50% learn
if rewards[idx] < dqn_median:
    continuous_targets[idx] = continuous_pred[idx]  # Bottom 50% zero gradient
```

### After:
```python
dqn_threshold = torch.quantile(dqn_rewards, 0.25)  # Top 75% learn
if rewards[idx] < dqn_threshold:
    continuous_targets[idx] = continuous_pred[idx]  # Bottom 25% zero gradient
```

### New Data Distribution:
```
With 50% expert ratio and 25th percentile DQN threshold:
- Expert frames: 50% of batch → learn spinner ✅
- DQN high-reward: 37.5% of batch → learn spinner ✅
- DQN low-reward: 12.5% of batch → zero gradient ❌

Total learning: 87.5% of data (was 75%)
Wasted: 12.5% of data (was 25%)
```

**Improvement:** +12.5% more data teaching spinner!

---

## Why This Should Work:

### 1. More Training Signal
- Learning from 87.5% of data instead of 75%
- More diverse DQN experiences included
- "Decent but not great" frames now contribute

### 2. Better Risk/Reward Balance
- Still filtering bottom 25% (truly bad actions)
- But embracing middle-tier performance (25th-50th percentile)
- These "okay" frames still contain useful spinner patterns

### 3. Faster Convergence
- More gradient flow → faster learning
- More examples of acceptable spinner control
- Less prone to overfitting to only "perfect" examples

### 4. Robustness
- Learning from broader range of situations
- Not just imitating best-case scenarios
- More realistic and generalizable spinner control

---

## Expected Results:

### Short Term (next 300k frames):
- **SpinAgr should break plateau**: 64.6% → 67-68%
- **CLoss continues decreasing**: 0.038 → 0.035
- **Discrete agreement stable**: ~93%

### Long Term (500k+ frames):
- **SpinAgr target**: 70-72%
- **Possible ceiling**: ~75% (limited by epsilon noise and expert quality)

### Monitoring:
Watch for:
- ✅ **SpinAgr upward trend** resuming
- ✅ **CLoss continuing down**
- ⚠️ **If SpinAgr drops**: Threshold too aggressive, revert to median

---

## Alternative Thresholds Considered:

| Threshold | DQN Learn | Total Learn | Risk | Decision |
|-----------|-----------|-------------|------|----------|
| **10th percentile** | 45% | 95% | High | Too risky - learning from bad frames |
| **25th percentile** | 37.5% | 87.5% | Low | **✅ CHOSEN - good balance** |
| **Median (50th)** | 25% | 75% | Very Low | ❌ Too conservative - caused plateau |
| **75th percentile** | 12.5% | 62.5% | Minimal | Too conservative - worse than median |

---

## Rollback Plan:

If spinner agreement **drops** below 63% after this change:

### Symptoms:
- SpinAgr: 64% → 62% (declining)
- CLoss: 0.038 → 0.042 (increasing)

### Action:
Revert to median threshold:
```python
dqn_threshold = dqn_rewards.median()  # Back to top 50%
```

Or try intermediate (33rd percentile = top 67%):
```python
dqn_threshold = torch.quantile(dqn_rewards, 0.33)  # Top 67%
```

---

## Technical Notes:

### Why Percentile Instead of Absolute Threshold?

**Good:** `torch.quantile(rewards, 0.25)` - Adaptive
- Adjusts to current performance level
- Always filters bottom 25% regardless of scale
- Robust to reward distribution changes

**Bad:** `reward > 3.0` - Fixed
- Breaks if reward scale changes
- Too strict when performance is low
- Too lenient when performance is high

### Performance Impact:

Minimal overhead:
- `torch.quantile()` is O(n log n) but n is small (batch_size/2 ≈ 1024)
- Called once per training step
- Negligible compared to forward/backward pass

### Why Not Learn From All DQN Frames?

**Risk of learning bad behavior:**
- Bottom 25% frames likely have mistakes
- Could include exploratory noise (epsilon=0.05)
- Could include genuinely bad spinner decisions
- Learning from these would **degrade** performance

**Better to:**
- Filter out worst frames
- Focus learning signal on success
- Let poor frames be forgotten (zero gradient)

---

## Success Criteria:

After 500k frames with this change:

**Minimum Success:**
- SpinAgr reaches 67% (breaking plateau)
- No degradation in discrete agreement (stays ~93%)

**Good Success:**
- SpinAgr reaches 70%
- CLoss drops to 0.033-0.035

**Excellent Success:**
- SpinAgr reaches 72%+
- Both metrics stable and improving

---

**Date:** October 14, 2025  
**Files Modified:** `Scripts/aimodel.py` (line ~1167: median → 25th percentile)  
**Impact:** Increases spinner learning from 75% to 87.5% of data, should break 64% plateau
