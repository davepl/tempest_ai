# Per-Actor Advantage Computation - The Real Fix

## The Critical Bug

The actor attribution fix revealed a **deeper problem**: even with correct attribution, the advantage-weighted learning was **systematically suppressing DQN frames** by comparing them to expert frames.

## Root Cause: Cross-Contamination

### Old (Broken) Code
```python
# WRONG: Compute advantages across mixed expert+DQN batch
reward_mean = rewards.mean()      # Mean of expert AND DQN together
reward_std = rewards.std()        # Std of expert AND DQN together
advantages = (rewards - reward_mean) / reward_std
advantage_weights = torch.exp(advantages * 1.5).clamp(0.001, 100.0)
```

### The Problem

If expert frames have higher rewards than DQN frames (which they do):

**Batch composition: 6000 DQN (reward ~0.3) + 2000 expert (reward ~0.8)**

```
Overall mean: ~0.425
Overall std: ~0.25

DQN frame with reward 0.3:
  advantage = (0.3 - 0.425) / 0.25 = -0.5
  weight = exp(-0.5 * 1.5) = exp(-0.75) ≈ 0.47x
  → DQN gradient is SUPPRESSED by 53%!

Expert frame with reward 0.8:
  advantage = (0.8 - 0.425) / 0.25 = +1.5
  weight = exp(+1.5 * 1.5) = exp(+2.25) ≈ 9.5x
  → Expert gradient is AMPLIFIED 9.5x!
```

**Result:** Expert frames dominate learning, DQN frames are ignored. The network learns expert behavior but can't improve its own (DQN) decision-making.

### Why This Happened

The advantage weighting was designed to "overcome bad DQN dominance" based on the assumption that DQN would produce a mix of good and bad experiences, and we wanted to learn from the good ones.

**But it backfired:** When you mix expert (high reward) and DQN (lower reward) in the same batch, the global statistics make DQN look bad **relative to expert**, not relative to itself.

## The Fix: Per-Actor Advantages

### New (Correct) Code
```python
# CORRECT: Compute advantages SEPARATELY per actor type
advantage_weights = torch.ones_like(rewards)

# DQN advantages: compare DQN frames to OTHER DQN frames
if n_dqn > 1:
    dqn_rewards = rewards[actor_dqn_mask]
    dqn_mean = dqn_rewards.mean()         # Mean of DQN frames ONLY
    dqn_std = dqn_rewards.std() + 1e-8    # Std of DQN frames ONLY
    dqn_advantages = (dqn_rewards - dqn_mean) / dqn_std
    dqn_advantages = dqn_advantages.clamp(-3, 3)
    dqn_weights = torch.exp(dqn_advantages * 1.5).clamp(0.001, 100.0)
    advantage_weights[actor_dqn_mask] = dqn_weights

# Expert advantages: compare expert frames to OTHER expert frames
if n_expert > 1:
    exp_rewards = rewards[actor_expert_mask]
    exp_mean = exp_rewards.mean()         # Mean of expert frames ONLY
    exp_std = exp_rewards.std() + 1e-8    # Std of expert frames ONLY
    exp_advantages = (exp_rewards - exp_mean) / exp_std
    exp_advantages = exp_advantages.clamp(-3, 3)
    exp_weights = torch.exp(exp_advantages * 1.5).clamp(0.001, 100.0)
    advantage_weights[actor_expert_mask] = exp_weights
```

### How This Helps

**Same batch: 6000 DQN (rewards: 0.1 to 0.5) + 2000 expert (rewards: 0.6 to 1.0)**

```
DQN-only statistics:
  mean: ~0.3
  std: ~0.12

DQN frame with reward 0.1 (bad DQN play):
  advantage = (0.1 - 0.3) / 0.12 = -1.67
  weight = exp(-1.67 * 1.5) ≈ 0.07x
  → Bad DQN play is suppressed ✓

DQN frame with reward 0.5 (good DQN play):
  advantage = (0.5 - 0.3) / 0.12 = +1.67
  weight = exp(+1.67 * 1.5) ≈ 14.1x
  → Good DQN play is amplified 14x! ✓

Expert-only statistics:
  mean: ~0.8
  std: ~0.12

Expert frame with reward 0.6 (bad expert play):
  advantage = (0.6 - 0.8) / 0.12 = -1.67
  weight = exp(-1.67 * 1.5) ≈ 0.07x
  → Bad expert play is suppressed ✓

Expert frame with reward 1.0 (good expert play):
  advantage = (1.0 - 0.8) / 0.12 = +1.67
  weight = exp(+1.67 * 1.5) ≈ 14.1x
  → Good expert play is amplified 14x! ✓
```

**Result:** 
- DQN learns from **good DQN plays** vs **bad DQN plays**
- Expert provides **good expert examples** vs **bad expert examples**
- **No cross-contamination**: DQN isn't penalized for being worse than expert
- Both actor types contribute meaningfully to learning

## Why Attribution Is Now Critical

With per-actor advantages, actor attribution is **essential**:

1. **Prevents suppression** - DQN frames aren't compared to expert frames
2. **Enables learning** - Good DQN plays get amplified relative to bad DQN plays
3. **Balances sources** - Expert and DQN both contribute based on their own variance
4. **Allows improvement** - Network can learn from its own successes, not just imitate expert

## Expected Impact

### Before Fix (Cross-Contaminated Advantages)
- Expert frames: Strong gradients (rewards above global mean)
- DQN frames: Weak gradients (rewards below global mean)
- **Result:** Network learns expert behavior, ignores DQN experiences

### After Fix (Per-Actor Advantages)
- Expert frames: Strong gradients for good plays, weak for bad plays (relative to expert)
- DQN frames: Strong gradients for good plays, weak for bad plays (relative to DQN)
- **Result:** Network learns from both expert examples AND its own successful explorations

## Verification

Watch for these changes after applying the fix:

1. **Batch logs** - Should still show ~74% DQN / 26% expert
2. **TD errors** - `td_err_mean_dqn` should decrease over time (DQN predictions improving)
3. **Q-values** - `q_mean_dqn` should increase as network learns good DQN plays
4. **Rewards** - `reward_mean_dqn` should improve as DQN gets better at playing
5. **Learning** - DQN should start learning from its own experiences, not just imitating expert

## Why The Original Approach Failed

The original advantage-weighted approach assumed:
- All experiences come from the same policy (DQN)
- Good experiences have high rewards, bad have low rewards
- We want to learn from the good ones

**But with expert guidance:**
- Experiences come from TWO policies (DQN + expert)
- Expert experiences have systematically higher rewards
- Comparing across policies suppresses the weaker policy (DQN)

**Solution:** Compare within policies, not across them.

## Files Modified

- `Scripts/aimodel.py` - Changed advantage computation in `train_step()` to be per-actor

## Technical Notes

### Edge Cases Handled

1. **Single DQN/expert frame** - Falls back to weight=1.0 (no normalization possible)
2. **All DQN or all expert** - Computes advantages normally within that single group
3. **Missing actor masks** - Falls back to old cross-batch computation (graceful degradation)

### Performance Impact

Minimal - just two additional mean/std computations per batch (very fast on GPU).

### Alternative Approaches Considered

1. **Per-actor loss weighting** - Could boost DQN loss by constant factor, but doesn't differentiate good vs bad DQN plays
2. **Separate optimizers** - Overkill, adds complexity
3. **Remove advantage weighting** - Loses ability to emphasize successful explorations
4. **Per-actor batch sampling** - Could sample 50/50 DQN/expert, but wastes expert data

Per-actor advantages is the cleanest solution that preserves the benefits of advantage weighting while fixing the cross-contamination bug.
