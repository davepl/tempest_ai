# CRITICAL BUG: Model Learning Anti-Pattern

## The Problem

Debug output shows the model has learned to predict **the opposite of correct actions**:

```
Greedy dist: [  0 704  86   0]  ← Model predicts: 89% action-1 (ZAP only)
Replay dist: [ 86   0 699   5]  ← Reality used:   88% action-2 (FIRE only)
Agreement: 10.4% (worse than random 25%!)
```

**Action Encoding**:
- Action 0: No fire, no zap
- Action 1: No fire, ZAP ← Model predicts this  
- Action 2: FIRE, no zap ← Gameplay uses this
- Action 3: Fire + zap

## Why This Happens

The model has learned that "action 1" is best, even though the replay buffer shows "action 2" was actually taken. This is **Q-value inversion** - the model has inverted which actions are good.

### Likely Causes

1. **Q-Value Explosion** (confirmed by Q-ranges: 322 → 9827 → 311)
   - Without TD clipping (removed at 10.0), Q-values explode
   - Exploded Q-values corrupt the policy
   - Model learns unstable/inverted value function

2. **Advantage Weighting Gone Wrong**
   - Advantage weights can amplify bad signals
   - If high-reward frames have wrong actions, model learns backwards

3. **Target Network Staleness**
   - Target updates every 200 steps might be too infrequent
   - Stale targets lead to value divergence

4. **Learning Rate Too High**
   - LR=0.00025 with unstable Q-values causes wild swings
   - Model overshoots and learns wrong patterns

## Immediate Fixes Required

### 1. Restore TD Target Clipping (CRITICAL)

```python
# In config.py:
td_target_clip: float | None = 50.0  # Restore clipping, but higher than 10.0
```

This prevents Q-values from exploding beyond [-50, +50], which is reasonable for your reward scale (600-1000 per episode).

### 2. Enable Gradient Clipping (CRITICAL)

```python
# In aimodel.py, around line 1337, UNCOMMENT:
max_norm = 10.0
torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), max_norm=max_norm)
```

This prevents gradient explosions that corrupt learning.

### 3. Reduce Learning Rate (HIGH PRIORITY)

```python
# In config.py:
lr: float = 0.0001  # Reduce from 0.00025
```

Slower learning = more stable convergence.

### 4. Increase Target Update Frequency (MEDIUM)

```python
# In config.py:
target_update_freq: int = 100  # Update more often (was 200)
```

More frequent target updates reduce divergence.

### 5. Reset Model Weights (NUCLEAR OPTION)

The current model has learned a completely inverted policy. You may need to:

```python
# In config.py:
FORCE_FRESH_MODEL: bool = True  # Start with fresh random weights
```

Then after starting training:
```python
FORCE_FRESH_MODEL: bool = False  # Disable after one run
```

## Why Agreement is Below 25%

Random baseline should be 25% (1 in 4 actions match by chance).

Agreement of 10% means the model is **anti-correlated**:
- When replay says "action 2", model says "action 1"  
- When replay says "action 0", model says "action 1"
- Model is systematically wrong

This is WORSE than random and indicates the model has learned backwards.

## Expected Behavior After Fixes

After applying the fixes, you should see:

### Immediately (First 10K steps)
```
[AGREE DEBUG] Step 100:
  Greedy dist: [512 512 512 512]  ← Random/balanced
  Replay dist: [ 86   0 699   5]  ← Still mostly action 2
  Agreement: 25-30%  ← Random baseline
```

### After Learning (50K+ steps)
```
[AGREE DEBUG] Step 50000:
  Greedy dist: [  50   20  950   10]  ← Model learns action 2!
  Replay dist: [ 86   0 699   5]     ← Matches reality
  Agreement: 70-85%  ← Model agrees with past
```

### Stable Learning (200K+ steps)
```
[AGREE DEBUG] Step 200000:
  Greedy dist: [  20   10  960   10]  ← Strong action 2 preference
  Replay dist: [ 86   0 699   5]
  Agreement: 85-95%  ← Very stable policy
```

## Implementation Priority

1. 🔴 **IMMEDIATE**: Enable TD target clipping (50.0)
2. 🔴 **IMMEDIATE**: Enable gradient clipping (10.0)
3. 🟡 **HIGH**: Reduce learning rate (0.0001)
4. 🟡 **HIGH**: Consider resetting model (FORCE_FRESH_MODEL=True)
5. 🟢 **MEDIUM**: Increase target update frequency (100)

## Verification

After applying fixes, look for:
- ✅ Agreement starts at ~25% (random baseline)
- ✅ Agreement gradually increases toward 60-80%
- ✅ Greedy distribution shifts toward action 2
- ✅ Q-value ranges stay bounded (not exploding to 9827)
- ✅ DQN reward shows upward trend

If agreement stays below 25%, there's still a bug in the calculation or encoding.
