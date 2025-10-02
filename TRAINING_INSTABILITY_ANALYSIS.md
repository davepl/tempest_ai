# Training Instability Analysis - Frame 20M

## Problem Statement
Training has stalled and is actually **degrading** despite 20M frames of experience.

---

## Critical Symptoms

### 1. Q-Value Divergence (SMOKING GUN) 🚨
```
Frame 17.7M: Q-range [0.15, 324.56]    (reasonable)
Frame 19.4M: Q-range [-12.91, 1024.10] (exploding)
Frame 19.6M: Q-range [-6.48, 1308.56]  (worse)
Frame 19.9M: Q-range [-12.89, 1779.30] (much worse)
Frame 20.4M: Q-range [-1.55, 1999.62]  (EXPLOSION!)
Frame 20.6M: Q-range [-3.37, 15.95]    (sudden collapse)
```

**Analysis**: Q-values are wildly unstable, jumping 100x in magnitude. This is **classic Q-value divergence** - the Bellman updates are not converging.

---

### 2. Performance Degradation 📉
```
DQN5M Slope (per million frames):
  Start (17.7M): +1.131  ← Improving nicely
  End   (20.0M): -0.190  ← Now declining!
```

**Analysis**: The network **was learning**, but now it's **unlearning**. Performance trend has reversed from positive to negative.

---

### 3. Loss Decreasing, Performance Degrading 🔴
```
Loss:        0.465 → 0.367  (decreasing = looks "good")
DQN5M Slope: +1.13 → -0.19  (declining = actually BAD)
```

**Analysis**: Classic **overfitting to wrong objective**. The network is minimizing loss but not improving at the task. Q-values have lost their meaning.

---

### 4. Gradient Instability
```
GradNorm: 4.764 to 8.787  (wild swings)
ClipΔ:    0.569 to 1.000  (sometimes clipping 43%, sometimes not at all)
```

**Analysis**: Optimizer is bouncing around, not converging smoothly.

---

### 5. Reward Instability
```
RwdSlope (per 20 episodes):
  -49.168, +50.312, -40.618, +48.981, etc.
```

**Analysis**: Individual episode performance is extremely noisy, making it hard for the network to learn consistent patterns.

---

## Root Cause Analysis

### Primary Cause: Learning Rate Too High

**Configuration:**
```python
lr: float = 0.003  # Very high for DQN
```

**Context:**
- Training at **~48 steps/second** (very fast!)
- With LR=0.003, each step makes LARGE parameter changes
- Network parameters oscillate instead of converging
- Q-values diverge because updates overshoot optimal values

**Evidence:**
- Q-values exploding (overshooting targets)
- Then collapsing (overcorrecting)
- Loss decreasing (network fitting training data)
- But performance degrading (fitted to wrong patterns)

---

### Contributing Factor 1: Soft Target Updates Too Aggressive

**Configuration:**
```python
tau: float = 0.012  # 1.2% blend per step
use_soft_target: bool = True
```

**Math at 48 steps/sec:**
```
After 100 steps (~2 seconds):
  Target = 0.988^100 × old + (1 - 0.988^100) × new
  Target = 0.297 × old + 0.703 × new
```

After just 2 seconds, target network is **70% updated**!

**Problem:**
- Target network should provide **stable bootstrapping**
- With tau=0.012 @ 48 steps/sec, target tracks local too closely
- Bootstrap targets are unstable (chasing a moving target)
- Q-learning requires stable targets for convergence

---

### Contributing Factor 2: N-Step Amplifies Errors

**Configuration:**
```python
n_step: int = 8
gamma: float = 0.995
# γ^8 = 0.961
```

**Problem:**
- 8-step returns use Q-values **8 steps ahead** for bootstrapping
- When those Q-values are exploding/diverging:
  - TD target = r + γ^8 × Q(s+8, a*)
  - If Q(s+8) = 1999, target explodes
  - If Q(s+8) = 10, target is too low
- N-step **amplifies** Q-value instability

---

### Contributing Factor 3: No Gradient Smoothing

**Configuration:**
```python
gradient_accumulation_steps: int = 1
training_steps_per_sample: int = 8
batch_size: int = 16384
```

**Problem:**
- 8 training steps per environment sample
- No gradient accumulation (each batch → immediate update)
- High-variance gradients with large LR = instability
- At 48 steps/sec, parameters change **very rapidly**

**What's needed:**
- Accumulate gradients over multiple batches
- Apply one smooth update instead of many noisy updates
- Reduces optimizer instability

---

### Contributing Factor 4: Fast Training + High Throughput

**Measured Performance:**
```
Steps/sec: ~48
Samples/sec: ~780,000 (batch_size × steps/sec)
Training speed: 4x faster than with PER
```

**Problem:**
- Faster training = more opportunities to diverge
- High LR + fast updates = recipe for instability
- Network doesn't have time to stabilize between updates

---

## Why This Happened Now (Not Earlier)

### Phase 1: Expert Curriculum (0-17M frames)
```
Expert ratio: 95% → 10%
```
- Expert provided good state coverage
- DQN learned from expert demonstrations
- Q-values stayed reasonable (expert is stable)
- **Worked well!**

### Phase 2: DQN Independence (17M-20M frames)
```
Expert ratio: 5% (at floor)
```
- DQN now driving 95% of actions
- No expert safety net
- DQN's Q-value errors **compound**
- With high LR + fast training → **divergence**

**Pattern**: Training was stable with expert scaffolding, unstable without it.

---

## Applied Fixes

### Fix 1: Lower Learning Rate ✅
```python
lr: float = 0.001  # Was 0.003 (3x reduction)
```

**Impact:**
- Smaller parameter updates per step
- Smoother convergence
- Less oscillation
- Q-values should stabilize

**Expected result**: Loss will decrease more slowly, but performance should improve steadily.

---

### Fix 2: Reduce Soft Target Update Rate ✅
```python
tau: float = 0.005  # Was 0.012 (2.4x slower)
```

**Impact:**
- Target network changes more slowly
- More stable bootstrap targets
- Better convergence guarantees

**Math at 48 steps/sec:**
```
After 100 steps (~2 seconds):
  Old tau=0.012: Target = 30% old + 70% new
  New tau=0.005: Target = 60% old + 40% new
```

Target network now provides **much more stable** bootstrapping.

---

### Fix 3: Reduce N-Step ✅
```python
n_step: int = 5  # Was 8
```

**Impact:**
- γ^5 = 0.975 (was γ^8 = 0.961)
- Bootstrap 5 steps ahead instead of 8
- Less error amplification from unstable Q-values
- More conservative credit assignment

**Tradeoff:**
- Slightly slower credit assignment
- But **much more stable** during recovery

Once Q-values stabilize, can increase back to 7-8.

---

### Fix 4: Add Gradient Smoothing ✅
```python
gradient_accumulation_steps: int = 4  # Was 1
```

**Impact:**
- Accumulate gradients over 4 micro-batches
- Apply one smooth update instead of 4 noisy updates
- Effective batch size: 16384 × 4 = **65,536 samples**
- Reduces gradient variance → more stable training

**Tradeoff:**
- 4x fewer optimizer steps per second
- But each step is **much better quality**
- Net result: better convergence

---

## What to Watch For (Next Run)

### Signs of Recovery ✅

1. **Q-value range stabilizes**
   - Should stay in reasonable range (e.g., [-10, 100])
   - Should not jump 100x in magnitude
   - Gradual growth is OK, explosive growth is BAD

2. **DQN5M Slope becomes positive**
   - Currently: -0.190 (declining)
   - Target: +0.5 to +1.5 (improving)
   - Sign of learning vs unlearning

3. **Loss stabilizes**
   - Less wild oscillation
   - Smooth downward trend
   - Not trying to fit noise

4. **Gradient ClipΔ consistent**
   - Should be close to 1.0 (not clipping much)
   - If frequently < 0.8, might need larger max_grad_norm
   - If always 1.0, clipping isn't helping (OK)

5. **Performance improves steadily**
   - DQN rewards trending up
   - DQN1M and DQN5M both increasing
   - Less episode-to-episode variance

---

### Signs of Continued Problems ⚠️

1. **Q-values still exploding**
   - Range still jumping wildly
   - Max Q > 1000
   - **Action**: Lower LR further (try 0.0005)

2. **DQN5M Slope still negative**
   - Still unlearning after 1-2M frames
   - **Action**: Increase expert ratio temporarily (10% → 20%)

3. **Loss oscillating**
   - Not converging smoothly
   - **Action**: Increase gradient_accumulation_steps to 8

4. **GradNorm still wild**
   - Swinging 2x or more
   - **Action**: Check for NaN/Inf values, reduce LR more

---

## Additional Recommendations

### Consider: Target Clamping (Emergency Brake)
```python
clamp_targets: bool = True   # Currently False
target_clamp_value: float = 50.0  # Reasonable for Tempest rewards
```

**When to enable:**
- If Q-values still explode after 1M frames
- Emergency stabilization measure
- Prevents runaway bootstrapping

**Downside:**
- Artificial ceiling on Q-values
- May limit long-term performance

---

### Consider: Reduce Training Intensity
```python
training_steps_per_sample: int = 4  # Was 8
```

**When to enable:**
- If still unstable after fixes
- Reduces update frequency
- Gives network more time to stabilize

**Tradeoff:**
- Slower learning (half the gradient updates)
- But more stable convergence

---

### Consider: Increase Expert Floor Temporarily
```python
expert_ratio_min: float = 0.20  # Was 0.10
```

**When to enable:**
- If DQN still degrading after 2M frames
- Gives more expert scaffolding during recovery
- Can reduce back to 10% once stable

---

### Monitor: Target Network Updates

**Expected behavior:**
```
Training Stats: xxxk/steps/xx.x/25k
                                  ^^^
                     Target age should increment
```

**If target age stays at 0k:**
- Hard refresh is firing too often
- Or display bug
- Check hard_target_refresh_every_steps setting

---

## Expected Timeline for Recovery

### Phase 1: Stabilization (0-2M frames)
- Q-values stop exploding
- Loss stabilizes
- DQN5M Slope approaches 0 (stops declining)

### Phase 2: Learning Resumes (2-5M frames)
- DQN5M Slope becomes positive
- Performance starts improving
- Q-values grow smoothly (not explosively)

### Phase 3: Acceleration (5M+ frames)
- Can potentially increase LR slightly (0.001 → 0.0015)
- Can increase n_step back to 7-8
- Can reduce gradient_accumulation_steps to 2

---

## Summary: What Changed

| Parameter | Old Value | New Value | Reason |
|-----------|-----------|-----------|--------|
| `lr` | 0.003 | 0.001 | Prevent parameter oscillation |
| `tau` | 0.012 | 0.005 | Stabilize bootstrap targets |
| `n_step` | 8 | 5 | Reduce error amplification |
| `gradient_accumulation_steps` | 1 | 4 | Smooth gradient updates |

---

## Prognosis

**With these fixes:**
- ✅ Q-value divergence should stop
- ✅ Training should stabilize within 1-2M frames
- ✅ Performance should resume improving
- ⚠️ Learning will be slower but **correct**

**Key insight**: Your previous "plateau breaker" (LR=0.003) actually **broke convergence** instead. Sometimes slower is faster in RL.

**Next checkpoint**: Review logs at 22M frames (2M from now). Should see clear signs of recovery.
