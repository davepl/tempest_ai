# N-Step Quick Reference Guide

## TL;DR

**Q: How high can we push n_step?**
**A: Practical max is n=10-15. Your current n=7 is optimal.**

---

## One-Page Summary

### Current Configuration ✅
```python
n_step = 7  # Well-tuned for Tempest
gamma = 0.995
batch_size = 16384
use_per = True
expert_ratio = 95% → 10% (floor)
```

### Performance Impact Table

| N-Step | Credit Assignment | Variance | Contamination @10% | Verdict |
|--------|------------------|----------|-------------------|---------|
| n=1 | Poor | Low | N/A | ❌ Too slow |
| n=3 | Fair | Low | 73% clean | ✅ Safe baseline |
| n=5 | Good | Medium | 59% clean | ✅ Balanced |
| **n=7** | **Very Good** | **Medium** | **48% clean** | ✅ **Current sweet spot** |
| n=10 | Excellent | High | 35% clean | ✅ Viable, test first |
| n=15 | Excellent+ | Very High | 20% clean | ⚠️ Risky, monitor closely |
| n=20 | Marginal gain | Extreme | 12% clean | ❌ Too risky |
| n=30+ | No gain | Unusable | <5% clean | ❌ Never use |

### Key Tradeoffs

**Benefits of Higher N:**
- ✅ Faster reward propagation (kills reward after 3-8 frames)
- ✅ Less bootstrap bias (more real rewards, less Q-estimate)
- ✅ Better sample efficiency

**Costs of Higher N:**
- ❌ Higher variance (grows ~linearly with n)
- ❌ More expert contamination ((1-expert_ratio)^n clean episodes)
- ❌ Lower effective planning horizon (γ^n in bootstrap)

---

## Recommendations by Training Phase

### Early Training (0-1M frames, expert_ratio≈95%)
**Recommendation:** Consider lowering to n=3-5
- Heavy expert contamination at n=7 (~0% clean episodes)
- Lower n reduces bias toward expert policy

### Mid Training (1M-6M frames, expert_ratio 95%→10%)
**Recommendation:** Keep n=7 (current)
- Balanced as expert ratio decreases
- Stable throughout transition

### Late Training (6M+ frames, expert_ratio=10%)
**Recommendation:** Could try n=10
- 35% clean episodes (acceptable)
- Better credit assignment
- Test on checkpoint before committing

---

## Decision Flowchart

```
Do you have instability (loss oscillations, Q-explosion)?
├─ YES → Reduce n_step to 3-5
└─ NO → Continue
    │
    Is credit assignment too slow (rewards not propagating)?
    ├─ YES → Increase n_step to 10
    └─ NO → Keep n=7 ✅
```

---

## Warning Signs

### Reduce N-Step If You See:
- Loss variance > 10
- Q-values growing unboundedly
- Episode rewards decreasing
- TD errors oscillating wildly

### Increase N-Step If You See:
- Agent can't learn delayed rewards
- Myopic behavior (only immediate rewards)
- Training is stable but slow

### Your Current Status:
- ✅ Stable loss (PER + large batch working)
- ✅ Reasonable Q-values
- ✅ Good reward progression
- **No changes needed** 🎯

---

## FAQ

**Q: Why not n=20 for maximum credit assignment?**
A: Variance grows ~20x, contamination at 88%, effective horizon shrinks to 10 steps. Costs exceed benefits.

**Q: Can I use different n_step during training?**
A: Yes, but requires code changes. Adaptive schedule (n=3 early, n=7 mid, n=10 late) is theoretically optimal but adds complexity.

**Q: Does n_step interact with other hyperparameters?**
A: Yes! With gamma (effective discount=γ^n), batch_size (variance tolerance), and expert_ratio (contamination risk).

**Q: What if I want to test higher n?**
A: Test on a checkpoint first:
1. Save current model
2. Try n=10 for 500K frames
3. Compare metrics (reward, loss variance, Q-values)
4. Revert if worse

**Q: My expert_ratio is high (95%). Should I lower n_step?**
A: Yes, consider n=3-5 during high expert_ratio phases to reduce contamination.

**Q: What's the theoretical maximum?**
A: Episode length (~500 frames), but practical max is n=15 due to variance/contamination.

---

## Experiment Protocol

### Test: Is n=10 better than n=7?

**Prerequisites:**
- Save checkpoint
- Stable training (loss not oscillating)
- Expert_ratio < 20%

**Procedure:**
1. Change config: `n_step = 10`
2. Restart training from checkpoint
3. Run 1M frames
4. Compare:
   - Average reward last 100 episodes
   - Loss std dev
   - Q-value magnitude
   - Wall time per million frames

**Decision criteria:**
- If reward +5% AND loss stable → keep n=10 ✅
- If reward +2% but loss +50% variance → questionable, prefer n=7 ⚠️
- If reward -2% OR loss unstable → revert to n=7 ❌

---

## Mathematical Summary

### N-Step Return
```
R_n = Σ(k=0 to n-1) γ^k * r_{t+k} + γ^n * max_a Q(s_{t+n}, a)
```

### Variance
```
Var[R_n] ≈ n * σ_r² + γ^(2n) * Var[V]
```

### Clean Episode Probability
```
P(all DQN) = (1 - expert_ratio)^n
```

At expert_ratio=10%:
- n=7: 48% clean
- n=10: 35% clean
- n=15: 20% clean

### Effective Discount
```
γ_eff = γ^n
```

With γ=0.995:
- n=7: γ_eff = 0.966
- n=10: γ_eff = 0.951
- n=20: γ_eff = 0.905

---

## Monitoring Commands

### Check Variance
```python
import numpy as np
print(f"Loss variance: {np.var(metrics.losses):.4f}")
# Healthy: < 1.0, Warning: 1-10, Critical: > 10
```

### Check Q-Values
```python
with torch.no_grad():
    q = agent.qnetwork_local(sample_states).max(dim=1).values
    print(f"Q-values: mean={q.mean():.2f}, max={q.max():.2f}")
# Healthy: mean 0-300, max < 1000
```

### Check TD Error
```python
td_err = abs(Q_pred - Q_target).detach()
print(f"TD error: mean={td_err.mean():.3f}, std={td_err.std():.3f}")
# Healthy: mean decreasing over time, std < 2*mean
```

---

## When to Change N-Step

### ✅ CHANGE if:
- Running controlled experiment
- Contamination is very high (expert_ratio > 50%)
- Have empirical evidence current n is suboptimal
- Testing adaptive schedule

### ❌ DON'T CHANGE if:
- Starting a long training run (stay conservative)
- System is stable and performing well
- No specific problem to solve
- Just curious (test on separate run instead)

---

## The Bottom Line

**Your n=7 is in the optimal range (5-10) for Tempest AI.**

**Maximum safe push:** n=10-12
**Maximum viable:** n=15 (with careful monitoring)
**Maximum theoretical:** n=50+ (academic interest only)

**Recommendation for long run:** Keep n=7 for stability. If you want to experiment, test n=10 on a checkpoint first.

**Bigger optimization opportunities:**
1. Expert ratio schedule (already optimized with 10% floor ✅)
2. Learning rate schedule (already optimized ✅)
3. Network architecture depth/width (potential experiments)
4. Reward shaping/clipping (already tuned ✅)
5. Exploration bonus tuning (diversity bonus already enabled ✅)

**Focus on those, not n_step.** 🎯
