# Training Diagnosis - Critical Issues

## Observed Problems

Based on the training output from frames 3M-4.5M:

### 1. Agreement Below Random Baseline (CRITICAL!)

**Observation**: Agreement is 7-23%, mostly 7-15%
**Expected**: ~25% for random baseline (1 in 4 actions), increasing to 60-80% as learning progresses
**Status**: 🔴 **BROKEN** - Agreement is below random chance!

**Possible Causes**:
- Agreement metric calculation is still broken
- Model is learning to anti-correlate with past actions (highly unlikely)
- Action encoding mismatch between training and replay
- DQN frame filtering is wrong (counting wrong frames)

### 2. Flat/Declining DQN Performance

**Observation**:
- DQN reward bouncing: 381 → 441 → 332 → 364 (no clear trend)
- DQN1M declining: 342.93 → 336.97 (getting worse!)
- DQN5M rising slowly: 289.25 → 306.99 (+17 over 1.5M frames)

**Expected**: Clear upward trend as model learns
**Status**: 🟡 **STAGNANT** - No meaningful improvement

### 3. Wildly Unstable Q-Values

**Observation**:
```
Frame 3,031,704: [10.35, 7197.81]   ← Max Q: 7197
Frame 3,282,384: [8.46, 9827.24]    ← Max Q: 9827 (explosion!)
Frame 3,500,936: [-2.57, 311.06]    ← Max Q: 311 (collapse)
Frame 3,717,296: [3.61, 322.32]     ← Max Q: 322 (stable-ish)
Frame 4,273,400: [2.65, 1109.57]    ← Max Q: 1109 (rising)
Frame 4,556,576: [4.53, 786.32]     ← Max Q: 786 (declining)
```

**Expected**: Gradual, stable growth in Q-value range
**Status**: 🔴 **UNSTABLE** - Q-values exploding and collapsing randomly

### 4. Training Hyperparameters

**Current Settings**:
- Learning rate: 0.00025
- Batch size: 2048
- Epsilon: 0.05 (5% exploration)
- Expert ratio: 60.0% (expert providing 60% of actions)
- TD target clip: Removed (was 10.0)
- Gradient clipping: Disabled

## Root Cause Analysis

### Agreement < 25% Indicates Fundamental Problem

If agreement is consistently below random baseline (25%), one of these is true:

1. **Metric is still broken**: The calculation isn't correctly comparing actions
2. **Model outputs are deterministic wrong**: Model always picks action opposite to what it did before
3. **Action encoding mismatch**: Actions stored in replay don't match current encoding
4. **Frame filtering bug**: We're comparing wrong subset of frames

### Q-Value Instability Suggests

1. **No TD target clipping**: After removing the 10.0 clip, Q-values can explode unconstrained
2. **Batch sampling issues**: High-reward frames causing large TD errors
3. **Learning rate too high**: 0.00025 might be too aggressive without target clipping
4. **Advantage weighting**: The 0.5 scaling on advantages might still cause issues

### Flat Learning Suggests

1. **Policy not improving**: DQN agent isn't learning effective strategy
2. **Exploration too low**: 5% epsilon might be insufficient
3. **Expert interference**: 60% expert ratio might be dominating learning
4. **Reward signal**: Rewards might not be shaped properly for learning

## Diagnostic Steps Added

Added debug output (prints every 100 training steps):

```python
[AGREE DEBUG] Step XXX:
  n_dqn=XXX, agree_pct=X.X%
  First 10 greedy: [...]     # Current policy choices
  First 10 replay: [...]     # Actions from replay buffer
  Greedy dist: [...]         # Distribution of greedy actions (0-3)
  Replay dist: [...]         # Distribution of replay actions (0-3)
```

This will show:
- How many DQN frames in batch
- What actions the model is choosing NOW
- What actions were chosen THEN
- If distributions are completely different

## Recommended Actions

### Immediate (Debug)

1. **Run with debug output** - Look at the agreement debug prints
2. **Check action distributions** - Are greedy and replay distributions similar?
3. **Verify n_dqn** - Are we actually getting DQN frames in batches?
4. **Check Q-value stats** - Print Q-value distributions, not just min/max

### Short Term (Fixes)

1. **Restore TD target clipping** - But much higher (50.0 or 100.0, not 10.0)
   ```python
   td_target_clip: float | None = 100.0  # Prevent explosion but allow growth
   ```

2. **Add gradient clipping back** - Stabilize training
   ```python
   torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), max_norm=10.0)
   ```

3. **Reduce learning rate** - More stable learning
   ```python
   lr: float = 0.0001  # Was 0.00025
   ```

4. **Increase epsilon** - More exploration
   ```python
   epsilon_min: float = 0.10  # Was 0.05
   ```

### Medium Term (Architecture)

1. **Review reward scaling** - Current rewards (600-1000) lead to huge Q-values
2. **Consider reward clipping** - Clip rewards to [-1, +1] range
3. **Review expert ratio decay** - 60% expert might be too high at 4M frames
4. **Audit action encoding** - Verify discrete actions are encoded consistently

## Expected Debug Output

### If Agreement Metric is Working

```
[AGREE DEBUG] Step 100:
  n_dqn=819, agree_pct=24.8%
  First 10 greedy: [2 1 2 3 2 2 1 2 3 2]
  First 10 replay: [1 2 3 2 1 3 2 1 2 3]
  Greedy dist: [0 204 410 205]  # Roughly balanced
  Replay dist: [205 204 205 205]  # Roughly balanced
```
- n_dqn > 0 (we have DQN frames)
- agree_pct ~25% (random baseline)
- Actions look random in both
- Distributions are balanced

### If Agreement Metric is Broken

```
[AGREE DEBUG] Step 100:
  n_dqn=0, agree_pct=0.0%
```
OR
```
[AGREE DEBUG] Step 100:
  n_dqn=819, agree_pct=7.2%
  First 10 greedy: [2 2 2 2 2 2 2 2 2 2]
  First 10 replay: [0 1 3 0 1 3 0 1 3 0]
  Greedy dist: [0 0 819 0]  # All action 2!
  Replay dist: [273 273 0 273]  # Never action 2!
```
- n_dqn might be 0 (no DQN frames being counted)
- Actions are completely different
- Model is stuck on one action
- Distributions don't overlap

## Priority

1. 🔴 **HIGHEST**: Debug agreement metric with new output
2. 🔴 **HIGH**: Restore TD target clipping (higher value)
3. 🟡 **MEDIUM**: Restore gradient clipping
4. 🟡 **MEDIUM**: Reduce learning rate
5. 🟢 **LOW**: Review reward scaling architecture

## Next Steps

1. Run training and capture debug output
2. Share first few agreement debug blocks
3. Analyze action distributions
4. Determine if metric is broken or model is broken
5. Apply appropriate fix based on diagnosis
