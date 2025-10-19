# Agreement Fix Summary - October 18, 2025

## Problem
Agreement metric started at 13.4%, worse than the 25% random baseline for 4 discrete actions.

## Root Causes Identified and Fixed

### 1. ✓ FIXED: Superzap Gate Action Mismatch
**Problem**: Superzap gate was blocking 99% of zap actions, creating systematic mismatch between what model learns to predict vs what gets stored in replay buffer.

**Evidence**:
- Diagnostics showed model predicting 67.8% zap actions (actions 1 and 3)
- Superzap gate only allowed 1% success rate
- Result: ~67% of predicted actions were being converted to different actions in buffer
- This caused agreement to be worse than random (13.4% < 25%)

**Fix Applied**:
```python
# Scripts/config.py
enable_superzap_gate: bool = False  # Was True
```

**Impact**: Eliminates the 67% action mismatch, allows model to learn true value of zap actions without artificial constraints.

**Note**: Can re-enable superzap gate later for evaluation/competition once model is trained.

---

### 2. ✓ FIXED: Network Initialization Bias
**Problem**: Fresh models showed extreme action bias (32-93% preference for single actions) due to default PyTorch initialization of discrete Q-head.

**Evidence**:
- Fresh model test showed 93% preference for action 3 (FIRE+ZAP)
- Different initialization attempts showed consistent bias patterns
- Bias ratio: 32x to infinite (some actions never selected)

**Fix Applied**:
```python
# Scripts/aimodel.py - HybridDQN.__init__()

# Initialize shared trunk
for layer in self.shared_layers:
    if isinstance(layer, nn.Linear):
        torch.nn.init.xavier_uniform_(layer.weight, gain=1.0)
        torch.nn.init.constant_(layer.bias, 0.0)

# Initialize discrete Q-head with balanced initialization
torch.nn.init.xavier_uniform_(self.discrete_fc.weight, gain=1.0)
torch.nn.init.constant_(self.discrete_fc.bias, 0.0)
# Zero-initialize output layer so all Q-values start exactly equal
torch.nn.init.constant_(self.discrete_out.weight, 0.0)
torch.nn.init.constant_(self.discrete_out.bias, 0.0)
```

**Impact**: 
- All Q-values now start at exactly 0.0 (perfectly balanced)
- No action bias during cold start
- Model learns from scratch without artificial preferences

---

### 3. ✓ CONFIGURED: Loss Weight Balance
**Current Settings**:
```python
discrete_loss_weight: float = 10.0    # Increased from 1.0
continuous_loss_weight: float = 1.0   # Kept at 1.0
```

**Rationale**: With superzap gate disabled and initialization fixed, discrete actions need stronger learning signal to catch up to the already-trained continuous spinner (85-90% agreement).

---

## Expected Results

### Immediate (First 50k Frames)
- Agreement will start low (< 25%) during cold start - **this is normal**
- Fresh Q-values (all 0.0) need time to differentiate through learning
- Random exploration (epsilon=0.05) will dominate early actions

### Medium Term (50k-200k Frames)
- Agreement should rise **above 25%** as Q-values differentiate
- Without superzap gate mismatch, agreement should climb to 30-50%
- Discrete actions should learn faster with 10x loss weight

### Long Term (200k+ Frames)
- Agreement should stabilize at 40-60% (similar to spinner agreement)
- Model should learn strategic zap usage without artificial constraints
- Overall performance should improve as discrete and continuous heads balance

---

## Testing Performed

### 1. Fresh Model Initialization Test
```bash
python test_fresh_model_bias.py
```
**Result**: All Q-values exactly 0.0 (perfect balance) ✓

### 2. Discrete Storage Test
```bash
PYTHONPATH=Scripts python test_discrete_storage.py
```
**Result**: Actions stored correctly as int32, range [0-3] ✓

### 3. Action Bias Diagnostics
```bash
python diagnose_agreement.py
```
**Before Fix**: 93% action 3, 32.7x bias ratio, 67.8% zap predictions
**After Fix**: Initialization balanced, superzap gate disabled

---

## Files Modified

1. **Scripts/config.py**
   - Disabled `enable_superzap_gate` (False)
   - Set `discrete_loss_weight = 10.0` (was 1.0)

2. **Scripts/aimodel.py**
   - Added proper initialization to `HybridDQN.__init__()`
   - Initialized shared trunk with Xavier uniform
   - Zero-initialized discrete output layer for balanced Q-values

---

## Monitoring Recommendations

### Key Metrics to Watch
1. **Agree%**: Should rise above 25% within 100k frames
2. **DLoss**: Should decrease as discrete head learns
3. **Reward**: Overall performance should improve
4. **Q-Range**: Should expand from [0,0] as values differentiate

### Expected Progression
```
Frame     Agree%   Interpretation
------    ------   --------------
0-50k     10-20%   Cold start, Q-values still near 0.0
50-100k   20-30%   Q-values differentiating, learning accelerating  
100-200k  30-50%   Healthy learning, comparable to spinner
200k+     40-60%   Stable, model has learned discrete strategy
```

### Red Flags
- Agree% stays below 15% after 100k frames → investigate buffer composition
- Agree% doesn't improve at all → check if learning is happening (DLoss should decrease)
- Q-values stay at 0.0 → check gradients, may need to adjust discrete_loss_weight

---

## Reversion Plan (If Needed)

If disabling superzap gate causes problems:

```python
# Scripts/config.py
enable_superzap_gate: bool = True  # Re-enable
discrete_loss_weight: float = 1.0  # Reduce if continuous learning suffers
```

However, you'll need to accept that agreement will remain below 25% with superzap gate enabled due to systematic action mismatch.

---

## Conclusion

The fixes address both immediate (initialization bias) and systematic (superzap gate mismatch) causes of sub-random agreement. With these changes:

- ✓ Fresh models start with balanced Q-values (no action bias)
- ✓ No artificial action blocking to create systematic mismatch
- ✓ Stronger discrete learning signal (10x loss weight)
- ✓ Model can learn true strategic value of zap actions

Agreement should now improve naturally over time as the model learns from experience.
