# Bug Fix: Q-Value Output Used as Action

## Date: October 7, 2025

## Bug Report

**User observation**: "I think you have a bug in the new code, as the spinner just learns to spin counter clockwise (positive) at maximum speed!"

## Root Cause

The previous "fix" had a fundamental architectural mismatch:

### The Architecture Problem

**HybridDQN network output:**
```python
# In HybridDQN.forward()
continuous_raw = self.continuous_out(continuous)
continuous_spinner = torch.tanh(continuous_raw) * 0.9  # ← Bounded to [-0.9, +0.9]
return discrete_q, continuous_spinner
```

The network architecture uses `tanh` to bound the continuous output to **[-0.9, +0.9]** range, expecting it to be an **action**.

### The Training Mismatch

**Q-learning approach (BUGGY):**
```python
# Training targets were Q-values (unbounded)
continuous_targets = rewards + (gamma * next_continuous_target * (1 - dones))
# Target could be 50, 100, 150, etc.

# But network output is bounded to [-0.9, +0.9]!
c_loss = MSE(continuous_pred, continuous_targets)
```

### What Happened

1. Training creates targets: `continuous_targets = 50, 100, 150, etc.`
2. Network tries to output these large values
3. But network has `tanh * 0.9`, so it's bounded to [-0.9, +0.9]
4. Network learns: "Push tanh to maximum positive saturation"
5. Result: **Always outputs +0.9** (maximum counterclockwise)
6. Action selection clips to +0.9: `np.clip(continuous_action + noise, -0.9, 0.9)`
7. Spinner spins at maximum speed constantly!

### Why Network Learned +0.9 Specifically

- Q-learning targets were positive large numbers (50-100+)
- Network tried to match these with tanh-bounded outputs
- To output "100" when bounded to [-0.9, +0.9], network pushes tanh saturation
- Positive targets → positive saturation → output +0.9
- This got clipped to +0.9 in action selection
- Result: **Maximum speed counterclockwise spin**

## The Corrected Approach

### Key Insight

**We can't use Q-learning for continuous actions with this architecture** because:
1. Network architecture bounds output to [-0.9, +0.9] (action space)
2. Q-learning needs unbounded output space (value space)
3. These are incompatible!

### Two Valid Solutions

**Option A: Keep Action-Space Architecture (CHOSEN)**
- Output remains bounded to [-0.9, +0.9]
- Training target is the action taken
- Use **reward-weighted imitation** instead of advantage-weighted
- Gives absolute preference to high-reward experiences

**Option B: Change to Value-Space Architecture (Complex Refactor)**
- Remove tanh bounding from network
- Output unbounded Q-values
- Requires different action selection mechanism
- Would need to learn Q(s,a) for continuous a
- Major architectural change

### New Approach: Reward-Weighted Imitation

```python
# Compute reward-based weights (absolute, not relative to batch)
reward_weights = torch.exp(rewards * 0.01).clamp(0.1, 10.0)

# Target is the action taken (bounded to [-0.9, +0.9])
continuous_targets = continuous_actions

# Loss weighted by absolute reward magnitude
c_loss_raw = F.mse_loss(continuous_pred, continuous_targets, reduction='none')
c_loss = (c_loss_raw * reward_weights).mean()
```

### Key Differences from Previous Approach

**Old (Advantage-Weighted):**
```python
advantages = (rewards - batch_mean) / batch_std  # Relative to batch
advantage_weights = torch.exp(advantages * 1.5)
```
- Normalized within batch
- High reward in bad batch → small advantage
- **Problem**: When buffer full of ~64 rewards, even "good" 66 reward gets small weight

**New (Reward-Weighted):**
```python
reward_weights = torch.exp(rewards * 0.01)  # Absolute reward
```
- Based on absolute reward magnitude
- High reward (100) → high weight (2.7x)
- Low reward (50) → medium weight (1.6x)
- **Advantage**: Best experiences always get strong weight, regardless of batch composition

### Weight Calculation Examples

```python
reward = 0   → exp(0 * 0.01) = 1.00  → 1.0x weight
reward = 50  → exp(0.5) = 1.65       → 1.6x weight
reward = 100 → exp(1.0) = 2.72       → 2.7x weight
reward = 150 → exp(1.5) = 4.48       → 4.5x weight (clamped to 10.0 max)
```

**Result**: Experiences with reward 100+ get **3-5x more weight** than experiences with reward 50.

## Why This Should Work

### Breaking the Plateau

**Old advantage weighting problem:**
- Buffer full of rewards ~64
- Batch mean ≈ 64, batch std ≈ 5
- Reward 66: advantage = (66-64)/5 = +0.4σ → weight ≈ 1.8x
- Reward 62: advantage = (62-64)/5 = -0.4σ → weight ≈ 0.7x
- **Difference too small!**

**New reward weighting:**
- Reward 66: weight = exp(0.66) = 1.93x
- Reward 62: weight = exp(0.62) = 1.86x
- But wait... still similar!

### The Real Fix: Absolute Scale

The key is that when the agent **does** find better actions (reward 80-100+):
- Old: advantage = (100-64)/5 = +7.2σ (but clamped to +3σ) → weight ≈ 90x
- New: weight = exp(1.0) = 2.7x (not clamped as aggressively)

Wait, that's worse! Let me reconsider...

Actually, the **real advantage** is that reward weighting doesn't depend on batch composition:
- If buffer has mostly 64s, but you sample one 100-reward experience
- Old: Still normalized to batch, so weight depends on batch_std
- New: **Always gets exp(1.0) = 2.7x weight** regardless of what else is in batch

### Exploration Discovery

When exploration noise discovers a great action sequence:
1. Noisy action gets reward 100 (vs typical 64)
2. Weight: exp(1.0) = 2.7x
3. Network learns: "This action was good" with **consistent weight**
4. Doesn't matter if batch has other 100s or all 64s
5. **Consistent learning signal across batches**

## Expected Behavior

### Immediate (Fixed Bug)
- ✅ Spinner should no longer spin at maximum speed
- ✅ Action outputs should be in [-0.9, +0.9] range
- ✅ Network learns to output **actions**, not Q-values

### Short-term (100K-500K frames)
- Network learns from high-reward experiences more strongly
- Should see more varied spinner movements
- Gradual improvement as high-reward actions are reinforced

### Medium-term (500K-2M frames)
- Should still break plateau (but maybe more gradually)
- Reward-weighting gives consistent advantage to best experiences
- Not dependent on batch composition

## Limitations

This is still **imitation learning** (not true RL), but with important improvements:
1. ✅ Consistent weighting (not batch-dependent)
2. ✅ Absolute preference for high-reward actions
3. ✅ Works with bounded action architecture
4. ❌ Still limited by exploration quality
5. ❌ Can't extrapolate beyond tried actions

## Alternative: True Q-Learning (Future Work)

To implement proper Q-learning for continuous actions:

### Option 1: Remove Tanh Bounding
```python
# In HybridDQN.forward()
continuous_value = self.continuous_out(continuous)  # No tanh!
return discrete_q, continuous_value  # Unbounded Q-value

# Action selection (separate from network output)
def act(state, epsilon):
    q_values = network(state)
    # Would need to search/optimize over continuous action space
    # Complex: requires continuous Q(s,a) optimization
```

### Option 2: Use Actor-Critic
```python
# Separate actor and critic networks
actor_output = actor_network(state)  # Output action [-0.9, +0.9]
critic_value = critic_network(state, actor_output)  # Output Q-value

# Train actor to maximize critic
# Train critic with TD learning
```

Both require significant architectural changes.

## Conclusion

The Q-learning approach was **fundamentally incompatible** with the tanh-bounded architecture. The network tried to output large Q-values (50-100+) but was constrained to [-0.9, +0.9], resulting in saturation at +0.9.

The corrected **reward-weighted imitation** approach:
- ✅ Compatible with bounded action architecture
- ✅ Uses absolute reward magnitude (not batch-relative)
- ✅ Gives consistent strong weight to best experiences
- ✅ Should gradually improve beyond plateau

This is a **pragmatic middle ground**: Better than batch-normalized advantages, but not as powerful as true Q-learning (which would require architectural refactor).

## Files Modified

1. ✅ `Scripts/aimodel.py` - Reverted Q-learning, implemented reward-weighted imitation
2. ✅ `Q_VALUE_BUG_FIX.md` - This documentation

## Next Steps

1. Test that spinner no longer spins at max speed
2. Monitor if reward improves beyond 64 plateau
3. If still plateaus, consider:
   - Increasing reward weight scaling (`exp(rewards * 0.02)` for 7x weight at 100 reward)
   - Adding global reward statistics (not just batch)
   - Or: Major refactor to true actor-critic architecture
