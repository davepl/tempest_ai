# Behavioral Cloning Loss Implementation

## Problem: Agreement Stuck at 20% Despite 50% Expert Data

After 1M+ frames with 50% expert demonstrations, agreement was stuck at 15-28% (barely better than random 25%). The network wasn't learning to imitate expert action choices.

### Root Cause

**Q-learning on expert data teaches Q-value magnitudes, NOT action selection:**

```python
# What Q-learning teaches:
Expert sample: (state, action=2, reward=10)
Network learns: Q(state, action_2) = 10

# Problem: This doesn't teach preference!
If network currently has:
  Q(state, action_0) = 15  ← Still chosen (highest Q)
  Q(state, action_2) = 10  ← Expert's action (not chosen)
```

Q-learning only teaches "what reward to expect" but not "which action to choose". The network could have correct Q-values but still choose the wrong actions.

## Solution: Behavioral Cloning Loss

Added a **cross-entropy loss** on expert frames that directly teaches action selection:

```python
# For EXPERT frames:
log_probs = F.log_softmax(Q_values, dim=1)  # Convert Q-values to probabilities
bc_loss = F.nll_loss(log_probs, expert_actions)  # "Choose what expert chose"

# For DQN frames:
# Pure Q-learning (no BC loss) - free to explore and discover
```

### How It Works

1. **Softmax converts Q-values to action probabilities**:
   - Q = [2.0, 5.0, 3.0, 1.0] → P ≈ [0.04, 0.76, 0.11, 0.02]
   
2. **Cross-entropy loss maximizes probability of expert's action**:
   - Expert chose action 1 (Q=5.0)
   - Loss pushes P[action_1] → 1.0
   - This directly teaches: "Choose action 1 in this state"

3. **Combined with Q-learning**:
   - Q-learning teaches value estimation (how good is this action?)
   - BC loss teaches action selection (which action should I choose?)

## Configuration

Added to `config.py`:
```python
use_behavioral_cloning: bool = True   # Enable BC loss for expert frames
bc_loss_weight: float = 1.0           # Weight relative to Q-learning loss
```

Total loss for expert frames:
```python
total_loss = (discrete_loss_weight * Q_learning_loss) + 
             (bc_loss_weight * behavioral_cloning_loss) + 
             (continuous_loss_weight * continuous_loss)
```

## Expected Impact

### Before (Q-learning only):
- Agreement: 15-28% (stuck, not improving)
- Network learns Q-values but not action preferences
- Slow learning, relies purely on TD bootstrapping

### After (Q-learning + Behavioral Cloning):
- **Agreement: Should jump to 60-80%** on expert frames
- Network directly learns expert action preferences
- Much faster learning via imitation
- **DQN exploration unchanged** - only expert frames get BC loss

## Metrics Display

Added new column `BCLoss` showing behavioral cloning loss magnitude:
```
Frame    ... DLoss   CLoss   BCLoss  Agree% ...
1.2M         0.003   0.004   0.150   65.2
```

BC loss should:
- Start high (~1.0-2.0) as network is random
- Decrease as network learns to imitate (~0.1-0.3)
- Correlate inversely with Agreement% (lower BC loss = higher agreement)

## Why This Preserves Discovery

**DQN frames (50% of data):**
- Pure Q-learning: `Q(s,a) = r + γ*max Q(s',a')`
- Epsilon-greedy exploration (FAFO=0.05)
- **No imitation constraint** - free to discover novel strategies
- Learns from its own experience what works

**Expert frames (50% of data):**
- Q-learning: Teaches value estimation
- **Behavioral cloning: Teaches action selection directly**
- Provides "warm start" from expert knowledge
- Accelerates learning without constraining final policy

## Implementation Details

Files modified:
- `Scripts/config.py`: Added `use_behavioral_cloning` and `bc_loss_weight` flags
- `Scripts/aimodel.py`: Added BC loss calculation for expert frames
- `Scripts/metrics_display.py`: Added `BCLoss` column to display

The BC loss is computed only when:
1. `use_behavioral_cloning=True` in config
2. Expert frames are present in the batch (`n_expert > 0`)
3. Expert mask is valid (`torch_mask_exp` exists)

## References

This technique is standard in deep RL:
- AlphaGo: Pretrained policy network via supervised learning on expert games
- OpenAI Five: Combined self-play (RL) with human demonstrations (BC)
- DAgger: Iterative behavioral cloning with exploration

The key insight: **Imitation (BC) provides a strong prior, Exploration (RL) refines beyond expert performance**.
