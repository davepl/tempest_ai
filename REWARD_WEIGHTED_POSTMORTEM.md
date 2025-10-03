# Reward-Weighted Learning Postmortem

## What Went Wrong

### **Observation** 🚨
After 8.5M frames of reward-weighted training (8.7M → 17.4M frames), the spinner **stopped moving and hunting**. Agent regressed to static/constant spinner predictions instead of dynamic behavior.

### **Root Cause: Regression to Mean**

**The Problem**:
```python
# Reward-weighted objective
Loss = Σ weight(reward) * (prediction - executed_action)²

# If reward isn't clearly correlated with specific spinner values,
# gradient descent finds a local minimum:
prediction = constant (mean of all actions)

# This minimizes loss but destroys dynamic behavior!
```

**Why it happened**:
1. **Weak Correlation**: Episode rewards come from many factors (fire/zap timing, enemy patterns, level design), not just spinner position
2. **Credit Assignment**: Network can't distinguish which spinner value caused the reward
3. **Optimization Path**: Easier to predict constant value than learn complex state→action mapping
4. **Loss Minimization**: Predicting mean minimizes weighted MSE loss mathematically

---

## Why Metrics Looked Good

### **Deceptive Indicators** ⚠️

| Metric | Value | Why It Looked Good | What Was Actually Happening |
|--------|-------|---------------------|----------------------------|
| **DQN5M** | 36.78 → 39.25 (+6.7%) | Performance improved! | **Discrete head** (fire/zap) was learning normally ✅ |
| **DQNSlope** | -7.8 → +0.02 | Trend turned positive! | Discrete decisions improved, masked continuous failure |
| **Loss** | 0.126 → 0.146 (stable) | Loss didn't explode | Predicting mean = low loss but bad behavior |
| **GradNorm** | 0.3-0.7 (healthy) | Gradients stable | Network converged to local minimum (mean) |
| **Training** | No divergence | System stable | Stable but wrong solution |

**Key Insight**: Discrete head improvement masked continuous head regression!

---

## The Failure Mode

### **Behavioral Cloning (Before)** ✅
```python
# Network learns: "In state S, spinner values range from -0.5 to +0.8"
# Predictions: Dynamic, state-dependent, maintains variety
# Behavior: Spinner actively hunts enemies, adjusts to situations
```

### **Reward-Weighted (After 8.5M frames)** ❌
```python
# Network learns: "Always predict 0.2 (the mean)"
# Predictions: Static, constant, no variety
# Behavior: Spinner frozen, no hunting, zombie-like movement

# Why? Because:
#   - High-reward action at 0.7: weight=2.0, loss=(0.2-0.7)²*2.0 = 0.50
#   - Low-reward action at -0.3: weight=0.5, loss=(0.2-(-0.3))²*0.5 = 0.125
#   - Mean action at 0.2: weight=1.0, loss=(0.2-0.2)²*1.0 = 0.0
#   
#   Total weighted loss minimized by predicting mean!
```

---

## Why This is Hard to Detect

### **Silent Degradation**

1. **No Error Signals**:
   - Loss stable (predicting mean minimizes loss)
   - Gradients healthy (converged to stable point)
   - No NaN/Inf warnings
   - Training metrics look normal

2. **Masked by Discrete Improvements**:
   - Fire/zap decisions getting better
   - Overall scores improving
   - DQN5M trending upward
   - System "appears" to be learning

3. **Requires Visual Inspection**:
   - Only observable by watching gameplay
   - Spinner behavior change is qualitative
   - Metrics don't capture "dynamism"

---

## Technical Deep Dive

### **Mathematical Analysis**

Given batch with actions and rewards:
```
State S: Enemy at position 0.7

Experience 1: action=-0.3, reward=10  → weight = 10/15 = 0.67
Experience 2: action=0.2,  reward=15  → weight = 15/15 = 1.00
Experience 3: action=0.7,  reward=20  → weight = 20/15 = 1.33
```

**Behavioral Cloning Loss** (uniform weights):
```
L_BC = (pred - (-0.3))² + (pred - 0.2)² + (pred - 0.7)²

Optimal: pred = mean(-0.3, 0.2, 0.7) = 0.2
Behavior: Learns distribution of actions (dynamic)
```

**Reward-Weighted Loss**:
```
L_RW = 0.67*(pred - (-0.3))² + 1.00*(pred - 0.2)² + 1.33*(pred - 0.7)²

Optimal: pred ≈ 0.32 (weighted toward high reward)
BUT: With noisy rewards, optimal converges to mean!
```

**Why Convergence to Mean Happens**:
- Rewards are noisy (same action → different rewards)
- State features don't clearly predict optimal spinner value
- Network can't learn: "In state X, action Y yields reward Z"
- Safest prediction: mean (minimizes expected loss across all samples)

---

## Credit Assignment Problem

### **The Core Issue**

```
Episode:
  Frame 1: state=S1, spinner=0.3, fire  → +10 (killed enemy)
  Frame 2: state=S2, spinner=0.7, move  → +5 (survived)
  Frame 3: state=S3, spinner=-0.2, zap  → +50 (cleared level)
  
  Total episode reward: +65

Question: Which spinner value was "good"?
  - 0.3? (helped kill enemy)
  - 0.7? (positioned well)
  - -0.2? (doesn't matter, level cleared due to zap timing)

Network receives: All actions get credit for +65 reward
Reality: Discrete actions (fire/zap) mattered most
```

**Result**: Reward signal is too noisy to learn spinner → reward correlation!

---

## Solution: Hybrid Mode

### **The Fix** ✅

Instead of pure reward weighting, use a **hybrid approach**:

```python
# HYBRID MODE: Blend behavioral cloning + reward bias

# Base weight: 1.0 (behavioral cloning, maintains diversity)
base_weight = 1.0

# Reward bias: scaled contribution (prevents mean regression)
reward_weight = (reward / reward_mean).clamp(0.5, 2.0)  # Narrow range!
scale = 0.3  # Conservative

# Final weight: 1.0 + 0.3 * (reward_weight - 1.0)
final_weight = base_weight + scale * (reward_weight - 1.0)

# Examples:
#   Low reward:  weight = 1.0 + 0.3*(0.5-1.0) = 0.85  (slight down-weight)
#   Mean reward: weight = 1.0 + 0.3*(1.0-1.0) = 1.00  (unchanged)
#   High reward: weight = 1.0 + 0.3*(2.0-1.0) = 1.30  (slight up-weight)
```

**Key Benefits**:
1. ✅ **Maintains Diversity**: Base weight 1.0 ensures all actions contribute
2. ✅ **Learns from Rewards**: Scale 0.3 adds gentle bias toward high-reward
3. ✅ **Prevents Mean Regression**: Narrow range (0.85-1.3) vs. wide (0.1-10.0)
4. ✅ **Stable Training**: Small adjustments won't collapse to constant

---

## Configuration Changes

### **Immediate Fix** (Applied)

```python
# config.py
continuous_learning_mode = 'behavioral_cloning'  # Rollback to restore dynamic behavior
continuous_reward_weight_scale = 0.3  # Reduced from 1.0 (conservative for future hybrid mode)
```

### **New Hybrid Mode** (Implemented, Not Enabled)

```python
# To try hybrid mode (after behavioral cloning recovers dynamics):
continuous_learning_mode = 'hybrid'  # Blend BC + reward bias
continuous_reward_weight_scale = 0.3  # Conservative scale (0.3 recommended)

# Weight range:
#   Low reward:  0.85x (slight penalty)
#   Mean reward: 1.00x (unchanged)
#   High reward: 1.30x (slight bonus)
```

---

## Lessons Learned

### **1. Reward Signal Quality Matters**

❌ **Problem**: Episode rewards aggregate many factors (fire/zap timing, enemy patterns, luck)
✅ **Need**: Per-action rewards or dense reward shaping for continuous control

### **2. Monitor Behavior, Not Just Metrics**

❌ **Missed**: Spinner stopped moving (qualitative observation)
✅ **Caught**: DQN5M improved (quantitative metric)
❌ **Lesson**: Visual inspection is critical for continuous actions!

### **3. Optimization Can Find "Wrong" Solutions**

❌ **Problem**: Predicting mean minimizes loss but destroys behavior
✅ **Insight**: Loss minimization ≠ good behavior always
✅ **Solution**: Regularization (entropy, diversity penalties)

### **4. Credit Assignment is Hard**

❌ **Problem**: Which spinner value caused the reward?
✅ **Reality**: Discrete actions (fire/zap) dominate rewards
✅ **Implication**: Continuous head needs different learning signal

---

## Recovery Plan

### **Phase 1: Restore Dynamics** (Current)

```bash
# Config: behavioral_cloning mode
# Goal: Spinner resumes hunting/movement
# Time: 1-2M frames to recover
# Watch: Spinner actively tracking enemies again
```

### **Phase 2: Validate Baseline** (After Recovery)

```bash
# Goal: Confirm behavioral cloning works
# Metrics: DQN5M stable, spinner dynamic
# Time: 2-3M frames to stabilize
```

### **Phase 3: Try Hybrid Mode** (Optional)

```bash
# Config: continuous_learning_mode = 'hybrid'
# Config: continuous_reward_weight_scale = 0.3
# Goal: Gentle reward bias without mean regression
# Watch: Spinner stays dynamic but biases toward high-reward strategies
# Time: 5M+ frames to evaluate
```

---

## Alternative Approaches

### **Option 1: Per-Action Reward Shaping** (Best, but requires game modification)

```python
# Instead of episode rewards, compute:
continuous_reward = distance_to_enemy_before - distance_to_enemy_after

# This gives immediate feedback:
#   Spinner moved toward enemy → positive reward
#   Spinner moved away from enemy → negative reward
```

**Pros**: Clear credit assignment, direct feedback
**Cons**: Requires game instrumentation, may not reflect true objectives

---

### **Option 2: Separate Continuous Critic** (Actor-Critic)

```python
# Train separate value network for continuous actions
continuous_value = critic(state, continuous_action)
policy_loss = -continuous_value.mean()

# This learns: "what continuous actions are valuable"
```

**Pros**: Theoretically sound, true RL
**Cons**: Complex, higher variance, needs careful tuning

---

### **Option 3: Curiosity/Entropy Regularization**

```python
# Add penalty for constant predictions
entropy_bonus = -std(continuous_predictions)
loss = mse_loss + lambda * entropy_bonus

# Encourages variety in predictions
```

**Pros**: Simple, prevents mean regression
**Cons**: Doesn't directly optimize for rewards

---

### **Option 4: Hybrid Mode** (Implemented) ✅

```python
# Blend behavioral cloning + reward bias
weight = 1.0 + 0.3 * (reward_weight - 1.0)

# Maintains diversity while learning from rewards
```

**Pros**: Simple, stable, combines benefits
**Cons**: Weaker learning signal than pure reward-weighted

---

## Metrics to Monitor Going Forward

### **Continuous Head Health Indicators**

1. **Visual Inspection** 👀
   - Spinner actively tracking enemies
   - Dynamic movement (not frozen at constant value)
   - Hunting behavior observable

2. **Prediction Variance** (add to metrics)
   ```python
   continuous_pred_std = continuous_predictions.std()
   # Healthy: >0.2 (dynamic)
   # Warning: <0.1 (static/frozen)
   ```

3. **Action Distribution**
   ```python
   # Plot histogram of continuous actions
   # Healthy: Spread across [-0.9, 0.9]
   # Warning: Spike at single value (mean)
   ```

4. **Performance Correlation**
   ```python
   # Does continuous action correlate with reward?
   correlation = np.corrcoef(continuous_actions, rewards)[0,1]
   # Healthy: >0.3 (some correlation)
   # Warning: <0.1 (no correlation, might as well be random)
   ```

---

## Summary

### **What Happened**
- ❌ Reward-weighted learning caused **regression to mean**
- ❌ Spinner **stopped moving**, became static
- ✅ Discrete head kept improving (masked the problem)

### **Why It Happened**
- ❌ Episode rewards don't clearly correlate with spinner values
- ❌ Credit assignment problem (which action caused reward?)
- ❌ Optimization found local minimum: predict mean = low loss

### **The Fix**
- ✅ **Immediate**: Rollback to behavioral_cloning mode
- ✅ **Future**: Hybrid mode (blend BC + reward bias at 0.3 scale)
- ✅ **Monitor**: Visual inspection + prediction variance metrics

### **Key Takeaway**
**Loss minimization ≠ Good behavior**. Always monitor actual agent behavior, not just metrics! 👀

---

## Timeline

| Frame Count | Mode | Status | Notes |
|-------------|------|--------|-------|
| 0-8.6M | Behavioral Cloning | ✅ Working | Dynamic spinner, baseline established |
| 8.6M-17.4M | Reward-Weighted | ❌ Failed | Regressed to mean, spinner froze |
| 17.4M+ | Behavioral Cloning | 🔄 Recovery | Restoring dynamic behavior |
| Future | Hybrid (0.3 scale) | 🔮 Planned | Conservative reward bias |

---

## Code Changes

### **Files Modified**

1. **config.py**:
   ```python
   continuous_learning_mode = 'behavioral_cloning'  # Rollback
   continuous_reward_weight_scale = 0.3  # Reduced for future hybrid
   ```

2. **aimodel.py**:
   - Added `'hybrid'` mode to continuous learning
   - Implements: `weight = 1.0 + 0.3 * (reward_weight - 1.0)`
   - Narrow range: 0.85-1.30 (vs. 0.1-10.0 in pure reward-weighted)

---

## References

**Regression to Mean in RL**:
- Known failure mode in reward-weighted regression
- Occurs when reward signal is noisy relative to action space
- Network learns to predict average to minimize loss across diverse experiences

**Credit Assignment Problem**:
- Classic RL challenge: which action caused which reward?
- Harder for continuous actions than discrete
- Requires dense rewards or sophisticated credit assignment (e.g., TD-learning, attention mechanisms)

**Hybrid Approaches**:
- Behavioral cloning provides supervised baseline
- Reward weighting adds RL signal
- Blending combines stability (BC) with optimization (RL)
