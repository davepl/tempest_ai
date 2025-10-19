# Discrete Head Behavioral Cloning Fix

**Date**: October 18, 2025  
**Issue**: Discrete agreement degrading (46.7% → 35.5%) while continuous agreement improving (62.7% → 72.8%)

## Root Cause

**Asymmetric Training:**
- **Continuous head**: Uses expert spinner values as supervised targets → learning works
- **Discrete head**: Uses bootstrapped TD targets only → learning fails

The discrete head was doing pure reinforcement learning (learning from its own wrong predictions), while the continuous head was doing supervised learning (learning from expert demonstrations).

## Solution: Add Behavioral Cloning for Discrete Actions

Added a **cross-entropy loss** that teaches the discrete head to imitate expert action choices:

```python
# Discrete TD loss (original)
d_loss_td = F.huber_loss(discrete_q_selected, discrete_targets, reduction='mean')

# NEW: Behavioral cloning loss for expert actions
bc_loss = torch.tensor(0.0, device=device)
if actor_expert_mask is not None and n_expert > 0:
    exp_mask = torch.from_numpy(actor_expert_mask).to(device)
    expert_q_values = discrete_q_pred[exp_mask]  # [n_expert, 4]
    expert_actions = discrete_actions[exp_mask]  # [n_expert, 1]
    
    if len(expert_q_values) > 0:
        # Cross-entropy: maximize Q-value of expert's chosen action
        bc_loss = bc_weight * F.cross_entropy(expert_q_values, expert_actions.squeeze(1))

# Combined loss
d_loss = d_loss_td + bc_loss
```

## How It Works

**Cross-Entropy Loss:**
- Takes Q-values for all 4 actions: `[Q0, Q1, Q2, Q3]`
- Applies softmax to convert to probabilities
- Maximizes probability of expert's chosen action
- This directly teaches "pick action 2 (FIRE)" rather than "predict Q=5.3"

**Why This Works:**
1. Expert action distribution: ~95% action 2 (FIRE)
2. Cross-entropy directly optimizes discrete action selection
3. Combines with TD learning for reward-based refinement
4. Same paradigm as continuous head (expert supervision)

## Configuration

Added new parameter:
```python
discrete_bc_weight: float = 1.0  # Weight for behavioral cloning loss
```

Set to 1.0 to balance with TD loss. Can tune:
- **Higher (2-5)**: More imitation, less exploration
- **Lower (0.1-0.5)**: More RL, less imitation
- **0.0**: Disables BC (pure TD learning)

## Expected Results

With `discrete_bc_weight=1.0`:
- **Discrete agreement should improve** from 35% toward 60%+ 
- **Q-values should stabilize** (less overestimation)
- **Model should learn FIRE action** (action 2) as primary choice
- **Training should converge faster** with expert guidance

## Related Changes

Also made in same session:
1. Balanced loss weights: `continuous_loss_weight=1.0`, `discrete_loss_weight=1.0`
2. Q-value clipping: `max_q_value=50.0` to prevent overestimation spiral
3. Target update frequency: `target_update_freq=500` to balance stability

## Verification

To verify BC is working:
1. Check that `Agree%` starts increasing (not decreasing)
2. Monitor Q-Range stays within [-5, +10] (not exploding)
3. Verify DLoss decreases while Agree% improves (not diverging)
4. Expect action 2 (FIRE) to dominate model's choices

## References

- **DQfD**: Deep Q-Learning from Demonstrations (Hester et al., 2017)
- **SQIL**: Soft Q Imitation Learning (Reddy et al., 2019)
- Combines imitation learning (BC) with reinforcement learning (TD)
