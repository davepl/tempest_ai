# N-Step Hyperparameter Analysis for Tempest

## Current Setting: n_step = 5

## Question
How high should we push n_step, and what are the tradeoffs?

---

## The N-Step Tradeoff

### Bias-Variance Spectrum

```
n=1                    n=5              n=10            n=20           n=∞
(1-step DQN)       (Current)        (Aggressive)    (Very Agg)   (Monte Carlo)
    │                  │                 │               │              │
    ├─ High Bias ──────┼─────────────────┼───────────────┼──── Low Bias┤
    ├─ Low Variance ───┼─────────────────┼───────────────┼─ High Var ──┤
    ├─ Most Stable ────┼─────────────────┼───────────────┼─ Unstable ──┤
    └─ Slow Learning ──┼─────────────────┼───────────────┼─ Fast Learn─┘
```

---

## Gamma Impact (γ = 0.995)

Your high gamma means n-step is practical at larger values:

| n-step | γ^n | Effective Horizon | Interpretation |
|--------|-----|-------------------|----------------|
| 1 | 0.995 | 1 step | Immediate bootstrap |
| 5 | 0.975 | 5 steps | Current setting ✅ |
| 10 | 0.951 | 10 steps | Still reasonable |
| 15 | 0.928 | 15 steps | Getting aggressive |
| 20 | 0.905 | 20 steps | Very aggressive |
| 30 | 0.861 | 30 steps | Too much discount |
| 50 | 0.778 | 50 steps | Monte Carlo territory |

**Key Insight**: γ=0.995 is high (long-term focused), so you can use larger n-step before over-discounting becomes a problem.

---

## Tempest-Specific Considerations

### Episode Structure
```
Typical Episode Timeline:
├─ 0-120 frames: Quick death (bad run)
├─ 120-1800 frames: Clear 1-2 levels (average)
└─ 1800+ frames: Multi-level success (good run)

Typical Engagement:
├─ 60-180 frames: Clear a segment (1-3 seconds)
└─ 1800-7200 frames: Complete a level (30-120 sec)
```

### Reward Structure
- **Immediate rewards**: Enemy kills (every few frames)
- **Delayed rewards**: Level completion bonuses
- **Diversity bonus**: Immediate exploration reward
- **Death penalty**: Sudden terminal reward

### Current n_step = 5 Analysis
- Looks ahead 5 transition steps in replay buffer
- With typical sampling, covers ~5 environment frames
- That's **< 0.1 seconds** of game time
- **Very short-term** credit assignment

---

## N-Step Value Analysis

### n=1 (Vanilla DQN)
**Characteristics:**
- Most stable (lowest variance)
- Most biased (heavy bootstrapping)
- Slowest learning (rewards propagate 1 step at a time)

**Use When:**
- Debugging issues
- Baseline comparison
- Very unstable environment

**Tempest Assessment:** Too conservative; you're leaving performance on the table

---

### n=3 (Conservative)
**Characteristics:**
- Very stable
- Slight improvement over n=1
- Standard in some Atari games

**Use When:**
- Initial training experiments
- Stability issues with higher n
- Conservative curriculum

**Tempest Assessment:** Could work but likely still too conservative

---

### n=5 (Current - Moderate) ✅
**Characteristics:**
- **Balanced bias/variance**
- Proven stable in many domains
- Good default for most RL tasks

**Gamma^5 = 0.975** (2.5% discount)

**Use When:**
- General purpose training
- First serious run
- Stability is important

**Tempest Assessment:** **Good starting point, proven stable**

---

### n=7-8 (Moderately Aggressive)
**Characteristics:**
- Faster credit assignment than n=5
- Still quite stable
- Good middle ground

**Gamma^8 = 0.961** (3.9% discount)

**Use When:**
- Training is stable at n=5
- Want slightly faster learning
- Reasonable next experiment

**Tempest Assessment:** **Recommended next step if n=5 stable**

---

### n=10 (Aggressive)
**Characteristics:**
- Significantly faster credit assignment
- Moderate variance increase
- Common in Atari with high gamma

**Gamma^10 = 0.951** (4.9% discount)

**Use When:**
- Very stable training
- Want faster convergence
- PER helps manage variance

**Tempest Assessment:** **Practical upper limit for initial experiments**

---

### n=15 (Very Aggressive)
**Characteristics:**
- Very fast credit assignment
- Higher variance (PER helps)
- Near Monte Carlo territory

**Gamma^15 = 0.928** (7.2% discount)

**Use When:**
- Training rock solid at n=10
- Short episodes
- Research/experimentation

**Tempest Assessment:** **Maximum recommended value**

---

### n=20+ (Monte Carlo Territory)
**Characteristics:**
- Fastest credit assignment
- Very high variance
- Risk of instability

**Gamma^20 = 0.905** (9.5% discount)
**Gamma^30 = 0.861** (13.9% discount)
**Gamma^50 = 0.778** (22.2% discount)

**Use When:**
- Pure research
- Very short episodes
- You really know what you're doing

**Tempest Assessment:** **Not recommended - too risky**

---

## Recommended N-Step Progression

### Phase 1: Establish Baseline (Current)
```python
n_step = 5  # Your current setting
```
**Goal**: Verify stable training, collect baseline metrics

**Expected**: Stable learning curve, gradual improvement

---

### Phase 2: Moderate Increase (If Stable)
```python
n_step = 8  # or 7
```
**Goal**: Faster credit assignment while maintaining stability

**Expected**: 10-20% faster learning, similar stability

**Monitor**:
- Loss stability (shouldn't oscillate wildly)
- Gradient norms (shouldn't explode)
- DQN performance trends (should improve steadily)

---

### Phase 3: Aggressive (If Still Stable)
```python
n_step = 10
```
**Goal**: Push toward faster convergence

**Expected**: Noticeably faster learning, acceptable variance

**Monitor**:
- Training stability (watch for divergence)
- Q-value range (shouldn't explode)
- PER priorities (check distribution)

---

### Phase 4: Maximum (Optional Experiment)
```python
n_step = 15
```
**Goal**: Test limits of n-step for Tempest

**Expected**: Maximum speed, higher variance

**Risky**: May destabilize training

---

## Benefits vs Tradeoffs Table

| n-step | Credit Assignment Speed | Variance | Stability | Bootstrap Bias | Recommended For |
|--------|------------------------|----------|-----------|----------------|-----------------|
| 1 | ⭐ Slowest | ⭐⭐⭐⭐⭐ Lowest | ⭐⭐⭐⭐⭐ Most | ❌ Highest | Debugging |
| 3 | ⭐⭐ Slow | ⭐⭐⭐⭐ Low | ⭐⭐⭐⭐ Very | ❌❌ High | Conservative |
| 5 | ⭐⭐⭐ Moderate | ⭐⭐⭐ Moderate | ⭐⭐⭐ Good | ⚠️ Moderate | **Current** ✅ |
| 8 | ⭐⭐⭐⭐ Fast | ⭐⭐ Moderate | ⭐⭐ Good | ✅ Low | **Next step** 🎯 |
| 10 | ⭐⭐⭐⭐ Fast | ⭐⭐ Higher | ⭐⭐ Acceptable | ✅ Low | Aggressive |
| 15 | ⭐⭐⭐⭐⭐ Fastest | ⭐ High | ⭐ Risky | ✅✅ Very Low | **Max limit** ⚠️ |
| 20+ | ⭐⭐⭐⭐⭐ Fastest | ❌ Very High | ❌ Unstable | ✅✅✅ Minimal | Not recommended |

---

## Why Not Higher Than 15-20?

### 1. **Discounting Over-Kills Long-Term Rewards**
```
γ^20 = 0.905 → 20-step-ahead reward worth 90.5%
γ^30 = 0.861 → 30-step-ahead reward worth 86.1%
γ^50 = 0.778 → 50-step-ahead reward worth 77.8%
```
You lose too much of the future value.

### 2. **Variance Increases Exponentially**
```
Var(n-step return) ≈ n × σ²
```
Each additional step multiplies variance.

### 3. **Episode Boundaries**
- Can't n-step past `done=True`
- Short episodes limit effective n
- Tempest deaths can be quick (60-120 frames)

### 4. **Target Network Staleness**
- Your target updates every 25k steps (hard refresh)
- Soft updates every step (τ=0.012)
- Very large n makes stale targets more problematic

### 5. **Diminishing Returns**
- Most benefit comes from n=1→5
- n=5→10 gives moderate gains
- n=10→20 gives smaller gains
- n=20+ minimal additional benefit

---

## Interaction with Other Hyperparameters

### With PER (use_per=True) ✅
**PER helps with larger n-step:**
- Importance sampling corrects for bias
- High-TD-error experiences sampled more
- Reduces effective variance

**Recommendation**: PER enables you to push n_step higher (8-10 more viable)

### With Gamma=0.995 ✅
**High gamma supports larger n:**
- γ^10 = 0.951 still reasonable
- Less over-discounting than γ=0.99
- Can safely go to n=10-15

**Recommendation**: Your high gamma is ideal for n=8-10

### With Batch Size=16384 ✅
**Large batch helps stability:**
- Averages out variance from high n
- More stable gradient estimates
- Supports aggressive n-step

**Recommendation**: Large batch enables n=10 safely

### With Mixed Precision (AMP) ✅
**AMP can interact with variance:**
- FP16 has limited precision
- High variance + low precision = instability risk
- Monitor for overflow/underflow

**Recommendation**: If n>10, watch for numerical issues

---

## Practical Experiment Plan

### Experiment 1: Current Baseline (n=5)
```python
n_step = 5  # Keep current
```
**Duration**: 5-10M frames
**Goal**: Establish stable baseline performance
**Metrics**: DQN rewards, loss, stability

---

### Experiment 2: Moderate Increase (n=8)
```python
n_step = 8
```
**Duration**: 5-10M frames  
**Goal**: Test faster credit assignment
**Compare**: Learning speed vs n=5
**Metrics**: Same as baseline

---

### Experiment 3: Aggressive (n=10)
```python
n_step = 10
```
**Duration**: 5-10M frames
**Goal**: Push toward optimal n-step
**Compare**: Stability vs speed tradeoff
**Metrics**: Plus gradient norms, Q-values

---

### Experiment 4: Limit Test (n=15)
```python
n_step = 15
```
**Duration**: 2-5M frames (may fail early)
**Goal**: Find breaking point
**Compare**: When does instability emerge?
**Metrics**: Watch for divergence signals

---

## Recommended Strategy

### Conservative Path (Recommended) 🎯
1. ✅ **Stay at n=5** for initial 10M frame run
2. If stable: Try **n=8** for next run
3. If still stable: Try **n=10**
4. **Don't exceed n=15**

### Aggressive Path (If You're Feeling Bold)
1. Try **n=8** immediately
2. If works: Jump to **n=10**
3. If works: Try **n=15** (carefully)
4. If fails: Back off to last stable value

### Research Path (For Science!)
1. Systematic sweep: n ∈ {3, 5, 7, 10, 15, 20}
2. Fixed compute budget per n
3. Compare learning curves
4. Find Pareto frontier (speed vs stability)

---

## Warning Signs of Too-High N-Step

Watch for these indicators that n is too large:

### Training Instability
- ❌ Loss oscillates wildly
- ❌ Gradient norms spike (>100)
- ❌ Q-values explode (>1000)
- ❌ Training diverges

### Performance Degradation
- ❌ DQN rewards decrease over time
- ❌ Agent forgets learned behaviors
- ❌ Erratic action selection

### PER Issues
- ❌ All priorities maxed out
- ❌ Priority distribution collapsed
- ❌ Sampling becomes uniform

**If you see these**: Reduce n_step immediately

---

## Summary & Recommendation

### Current: n_step = 5 ✅
**Status**: Good, stable, proven

### Recommended Range: **5-10**
- **n=5**: Safe baseline (current)
- **n=8**: Sweet spot for Tempest 🎯
- **n=10**: Aggressive but viable
- **n=15**: Maximum practical limit ⚠️
- **n=20+**: Not recommended ❌

### Next Step: Try n=8
**Rationale**:
- Your γ=0.995 supports it
- PER manages variance
- Large batch helps stability
- 60% faster credit assignment than n=5
- Still conservative enough to be safe

### Don't Exceed: n=15-20
**Rationale**:
- γ^20 = 0.905 (excessive discounting)
- Variance becomes problematic
- Diminishing returns beyond n=10
- Episode boundaries limit utility
- Risk of instability

---

## Final Answer

**Q:** How high should we push n_step?

**A:** 
- **Current (n=5)**: Good starting point ✅
- **Recommended (n=8)**: Optimal for Tempest 🎯
- **Maximum (n=10-15)**: Practical upper limit ⚠️
- **Avoid (n=20+)**: Too risky, diminishing returns ❌

**Your γ=0.995 and PER enable pushing to n=8-10 safely. Start with n=8 for your next serious run.** 🚀
