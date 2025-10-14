# Q-Value Rescaling Bug Fix

## Problem Discovered

After restart, **DLoss jumped 10-20x** and **agreement collapsed** from 93% → 17%, despite no code changes.

### Symptoms:
```
Before Restart (stable):
  DLoss: 0.028-0.034 (low, stable)
  Agree%: 92-94% (excellent)
  Q-values: [-53, +9] (reasonable)

After Restart (catastrophic):
  DLoss: 0.18 → 0.37 → 0.53 → 0.59 (20x increase!)
  Agree%: 17% → 40% → 55% → 63% (collapsed, slowly recovering)
  Q-values: [0.03, 0.08] → [-45, +64] (explosion from tiny to huge)
```

## Root Cause

**Q-value rescaling on model load** was catastrophically interfering with training:

### The Deadly Sequence:

1. **Model loads** with Q-values around [-10, -0.4]
2. **Rescaling triggers** (range ~10 > threshold of 10.0)
3. **Scales to target of 2.0**: Q-values become [0.03, 0.08]
4. **Replay buffer clears**: Fresh data with rewards 5-7 arrives
5. **Massive TD errors**: 
   - TD_error = reward + γ×Q_next - Q_current
   - TD_error ≈ 5 + 0.99×0.08 - 0.05 ≈ **5.0** (100x normal!)
6. **DLoss explodes**: Huber loss on 5.0 TD errors → 0.18-0.59
7. **Network scrambles**: Q-values race from 0.08 → 64 in 200k frames
8. **Agreement collapses**: Action preferences completely scrambled

### The Irony:

The rescaling was **designed to prevent Q-value explosion**, but instead it:
- Triggered on healthy Q-values ([-10, -0.4] is fine!)
- Created tiny Q-values incompatible with reward scale
- **CAUSED the very explosion it was meant to prevent**

## The Fix

**Disabled Q-value rescaling entirely** in `aimodel.py` lines ~1710-1720:

```python
# BEFORE (harmful):
if q_range > 10.0:
    print(f"WARNING: Loaded model has large Q-values [{qmin:.3f}, {qmax:.3f}]. Rescaling down...")
    scale = target_scale / q_range
    # ... rescale all parameters ...

# AFTER (safe):
# Just log Q-values for monitoring, but don't rescale
print(f"Loaded model Q-value range: [{qmin:.3f}, {qmax:.3f}]")
# Rescaling disabled - let Q-values naturally stabilize through training
```

## Why This Works

1. **Q-values find natural scale**: Through TD learning and target updates
2. **No artificial disruption**: Network weights stay consistent with replay buffer
3. **Gradual adaptation**: If Q-values need adjustment, happens smoothly over training
4. **No catastrophic resets**: Agreement and performance preserved across restarts

## Expected Behavior After Fix

On restart, you should see:
- ✅ **DLoss stays low**: ~0.03 (no 10-20x spike)
- ✅ **Agreement preserved**: Stays at 92-94% from first training row
- ✅ **Q-values stable**: Continue from previous range, no explosion
- ✅ **Smooth continuation**: Training picks up where it left off

## Validation

Monitor first few rows after restart:
- DLoss should be 0.025-0.040 (not 0.18+)
- Agree% should be 90%+ immediately (not 17%)
- Q-values should be similar to pre-restart (not [0.03, 0.08])

## Lessons Learned

1. **Trust the training process**: Q-values scale naturally through Bellman updates
2. **Beware "helpful" interventions**: Automatic rescaling can do more harm than good
3. **Test restarts thoroughly**: Training instability often shows up after model reload
4. **Log, don't modify**: Monitor Q-values but let the optimizer handle scaling

---
**Date:** October 14, 2025  
**Files Modified:** `Scripts/aimodel.py` (disabled Q-value rescaling)  
**Impact:** Prevents catastrophic DLoss spikes and agreement collapse on restart
