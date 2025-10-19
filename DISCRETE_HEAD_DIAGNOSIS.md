# Discrete Head Training Issues - Diagnosis & Fixes

## Observed Problems (1M frames of training)

### 1. **GradNorm Always 0.000** ❌ CRITICAL BUG
**Symptom**: GradNorm column shows 0.000 for every training step

**Root Cause**: Gradient norm was being calculated but never recorded
```python
# Before (BUG):
torch.nn.utils.clip_grad_norm_(agent.qnetwork_local.parameters(), 10.0)
# Result was discarded!
```

**Fix Applied**:
```python
# After (FIXED):
grad_norm = torch.nn.utils.clip_grad_norm_(agent.qnetwork_local.parameters(), 10.0)
metrics.last_grad_norm = float(grad_norm.item())
```

**Impact**: Now you'll be able to monitor gradient health and detect:
- Vanishing gradients (norm → 0)
- Exploding gradients (norm >> 10)
- Dead neurons (consistently low norms)

---

### 2. **DLoss Increasing** ⚠️ CONCERNING PATTERN
**Symptom**: Discrete loss trending upward over 1M frames
```
Frame     DLoss    Trend
36k      0.00722   (baseline)
79k      0.01572   +117% ↑
165k     0.01385   +92%  ↑
553k     0.01837   +154% ↑
1023k    0.01926   +167% ↑
```

**Meanwhile**: CLoss decreasing (good)
```
Frame     CLoss    Trend
36k      0.01889   (baseline)
1023k    0.01443   -24%  ↓ (learning well)
```

**Analysis**:
- Continuous head is learning successfully (CLoss ↓, SpinAgr% 60→66%)
- Discrete head is struggling (DLoss ↑, Agree% only 39→40%)
- This suggests a **discrete-specific issue**, not a general training problem

---

### 3. **Zero-Initialized Output Layer** ❌ DEAD NEURONS
**Root Cause**: Previous fix for action bias used zero weights
```python
# Before (CAUSES DEAD NEURONS):
torch.nn.init.constant_(self.discrete_out.weight, 0.0)
torch.nn.init.constant_(self.discrete_out.bias, 0.0)
```

**Problem**:
- Output = 0.0 * input + 0.0 = always 0.0
- Gradient through zero-weight layer is extremely weak
- Layer can't learn because there's no signal to differentiate actions
- Creates "symmetry breaking" problem

**Fix Applied**:
```python
# After (ALLOWS GRADIENT FLOW):
torch.nn.init.uniform_(self.discrete_out.weight, -0.003, 0.003)  # Small random
torch.nn.init.constant_(self.discrete_out.bias, 0.0)              # Zero bias
```

**Expected Impact**:
- Initial Q-values near 0 but with slight variation (-0.3 to +0.3 range)
- Gradients can flow and differentiate actions
- Still avoids strong action bias (range is only ~0.6)
- Should see DLoss start to decrease after retraining

---

## Why This Explains the Observations

### Q-Range Evolution
```
Frame     Q-Range      Analysis
0        [0.0, 0.0]   Zero init - all equal
36k      [0.0, 1.4]   Started learning, but slowly
165k     [0.4, 2.5]   Expanding, but maybe due to target network drift?
1023k    [-0.2, 2.3]  Range ~2.5, reasonable but learning seems slow
```

**Interpretation**: Q-values ARE changing, but:
- Change might be more from target network updates than true learning
- With zero output weights, the discrete_fc layer has to do ALL the work
- Gradients are weak, so learning is very slow

### Agreement Stagnation
```
Frame     Agree%   Expected    Gap
36k       38.7%    25% random  +13.7% ✓
165k      45.1%    50%+        -5%    ⚠️
1023k     40.2%    60%+        -20%   ❌ (actually decreased!)
```

**Interpretation**: 
- Initial improvement (38→45%) was promising
- Then **regressed** back to 40%
- Should be 50-60% by 1M frames with healthy learning
- Regression suggests the network is "forgetting" or unstable

### SpinAgr% vs Agree% Divergence
```
Metric       36k    1023k   Change
SpinAgr%    60.4%   65.8%   +5.4%  ✓ (healthy)
Agree%      38.7%   40.2%   +1.5%  ⚠️ (poor)
```

**Interpretation**:
- Continuous head learning normally (5.4% improvement)
- Discrete head barely learning (1.5% improvement)
- Confirms **discrete-specific issue**

---

## Root Cause Summary

The zero-initialized discrete output layer created a "dead neuron" problem:

1. **No gradient signal**: Zero weights → zero gradients → no learning
2. **Slow adaptation**: discrete_fc has to compensate, but it's hard
3. **Increasing loss**: As Q-targets grow (from rewards), predictions stay near zero
4. **Agreement stagnation**: Model can't differentiate actions effectively

Meanwhile, the continuous head works fine because it was initialized with small random weights (`gain=0.1`).

---

## Fixes Applied

### 1. ✓ Fixed GradNorm Tracking
**File**: `Scripts/training.py`
- Now captures and records gradient norm
- Will reveal gradient health in next training run

### 2. ✓ Fixed Discrete Output Initialization  
**File**: `Scripts/aimodel.py`
- Changed from zero weights to small random (-0.003, +0.003)
- Allows gradient flow while maintaining low initial bias
- Small enough to avoid action preference (range ~0.6)

---

## Expected Results After Fix

### Immediate (Next Training Run - First 100k Frames)
- **GradNorm**: Should show non-zero values (expect 0.5-5.0 range)
- **DLoss**: Should start higher due to random init, then **decrease**
- **Agree%**: May start slightly lower (35-40%) but should climb steadily
- **Q-Range**: Should expand more rapidly with better gradient flow

### Medium Term (100k-500k Frames)
- **DLoss**: Steady decrease toward CLoss levels
- **Agree%**: Should reach 50-60% (matching SpinAgr%)
- **GradNorm**: Should stabilize around 1.0-3.0
- **Learning rate**: Discrete and continuous heads should learn at similar rates

### Long Term (500k+ Frames)
- **DLoss**: Should be similar magnitude to CLoss (both ~0.01)
- **Agree%**: Should reach 60-70% (higher than SpinAgr% for discrete actions)
- **Q-Range**: Should stabilize at reasonable values (2-10 range)

---

## Monitoring Checklist

After restarting with fixes:

✅ **GradNorm > 0**: Confirms gradient flow
- If < 0.1: Weak gradients, may need higher learning rate for discrete
- If > 10: Exploding gradients, may need stronger clipping

✅ **DLoss Decreasing**: Confirms discrete head is learning
- Should drop from initial ~0.02 to ~0.01 or lower
- If still increasing: Check discrete_loss_weight (currently 10.0)

✅ **Agree% Increasing**: Confirms policy improvement
- Should cross 50% by 200k frames
- Should reach 60%+ by 500k frames
- If stagnant: Model might need more exploration (increase epsilon)

✅ **DLoss ≈ CLoss Eventually**: Confirms balanced learning
- Both should converge to similar values
- If DLoss >> CLoss: Increase discrete_loss_weight further
- If CLoss >> DLoss: Decrease discrete_loss_weight

---

## Retraining Recommendation

**Option 1 - Fresh Start (Recommended)**:
```bash
# Backup current model
cp models/tempest_model_latest.pt models/tempest_model_latest.pt.before_init_fix

# Set fresh model flag
# In Scripts/config.py: FORCE_FRESH_MODEL = True

# Restart training
python Scripts/main.py
```

**Option 2 - Continue Training**:
- The zero-weight layer has probably learned SOME representation by now
- But it's been learning inefficiently for 1M frames
- Continuing may be slower than restarting

**Why Fresh Start**:
- The discrete_fc layer may have "overfit" to compensate for dead output layer
- Fresh start with proper init will learn more efficiently
- You'll see healthy DLoss decrease from the beginning
- Only lose 1M frames, but gain proper learning trajectory

---

## Additional Observations

### Q-Range is Acceptable
- Range of 2.5 at 1M frames is reasonable for early training
- Negative Q-values are normal (negative rewards exist)
- Not growing too fast (no value explosion)
- **Verdict**: Q-range is healthy ✓

### Loss Weighting is Good
- `discrete_loss_weight = 10.0` gives discrete head strong signal
- This is appropriate given discrete task is harder
- Don't change unless DLoss becomes << CLoss

### Training Speed is Good
- 250 steps/sec is solid throughput
- ~500k samples/sec with batch_size=2048
- Buffer filling normally (1M entries at 1M frames)
- **Verdict**: Training infrastructure is healthy ✓

---

## Summary

**Main Issues**:
1. ❌ GradNorm not being recorded (FIXED)
2. ❌ Zero-initialized output causing dead neurons (FIXED)  
3. ⚠️ DLoss increasing instead of decreasing (should fix with above)

**Root Cause**: Zero-weight initialization prevented discrete head from learning efficiently

**Solution**: Use small random weights to enable gradient flow while maintaining balance

**Recommendation**: **Start fresh** with corrected initialization for fastest path to good performance
