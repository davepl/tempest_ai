# Behavioral Cloning Analysis: Can the DQN Learn Beyond the Expert?

## Executive Summary

**Verdict: ✅ YES - The DQN CAN learn beyond the expert, but with ONE MAJOR CAVEAT:**

- **Discrete head (fire/zap)**: ✅ **True RL** - Uses TD learning with rewards
- **Continuous head (spinner)**: ❌ **Pure behavioral cloning** - Supervised learning on executed actions

**The continuous head cannot discover better spinner strategies than what was executed (DQN or expert).**

---

## The Claim

Someone suggested the RL can never learn beyond the expert because one of the heads is set up as "behavioral cloning only."

---

## Investigation Findings

### 1. **Network Architecture** ✅

```python
class HybridDQN(nn.Module):
    """
    Two heads:
    - Discrete head: Q-values for 4 fire/zap combinations
    - Continuous head: Single spinner value [-0.9, +0.9]
    """
```

**Finding**: Two separate heads with different learning objectives.

---

### 2. **Loss Computation** 🔴 **CRITICAL ISSUE**

Located in `train_step()` at line ~1351:

```python
# Target computation (inside torch.no_grad())
discrete_targets = r + (gamma_boot * discrete_q_next_max * (1 - dones))
continuous_targets = continuous_actions  # ← BEHAVIORAL CLONING!

# Loss computation
d_loss = F.huber_loss(discrete_q_selected, discrete_targets, reduction='none')
c_loss = F.mse_loss(continuous_pred, continuous_targets, reduction='none')
```

**Finding**:

| Head | Target | Learning Type | Can Improve Beyond Data? |
|------|--------|---------------|--------------------------|
| **Discrete** | `r + γ * max Q(s', a')` | **TD Learning (RL)** | ✅ YES |
| **Continuous** | `continuous_actions` | **Behavioral Cloning (SL)** | ❌ NO |

---

### 3. **What Gets Stored in Replay Buffer** 🔍

From `socket_server.py` line ~471:

```python
# Action selection (15% expert, 85% DQN based on expert_ratio)
if use_expert:
    # Expert system chooses
    fire, zap, spin = get_expert_action(...)
    discrete_action = fire_zap_to_discrete(fire, zap)
    continuous_spinner = float(spin)
    action_source = 'expert'
else:
    # DQN chooses with epsilon-greedy
    da, ca = self.agent.act(frame.state, epsilon)
    discrete_action, continuous_spinner = int(da), float(ca)
    action_source = 'dqn'

# Store the ACTION THAT WAS ACTUALLY TAKEN
state['last_action_hybrid'] = (discrete_action, continuous_spinner)
```

Then stored in replay buffer (line ~321):

```python
self.agent.step(state['last_state'], int(da), float(ca), total_reward, 
                frame.state, bool(frame.done))
```

**Finding**: The replay buffer stores **whatever action was executed** (15% expert, 85% DQN + epsilon exploration).

---

### 4. **Replay Buffer Composition** 📊

Current setting: `expert_ratio = 0.15`, `epsilon = 0.15`

**Approximate buffer composition**:
```
15% Expert actions (both discrete + continuous)
85% DQN actions broken down as:
    ├─ ~72.25% DQN greedy actions (85% × 85%)
    └─ ~12.75% DQN random exploration (85% × 15%)
```

**For continuous head specifically**:
- 15% expert spinner values
- 72.25% DQN-predicted spinner values (potentially suboptimal)
- 12.75% random spinner values (uniform exploration in [-0.9, 0.9])

---

## The Problem: Behavioral Cloning Cannot Extrapolate

### **Discrete Head (Fire/Zap)** ✅

```python
# TD target uses REWARD + bootstrapped Q-value
discrete_targets = r + (gamma * max Q(s', a'))

# This enables:
# 1. Credit assignment: Good outcomes → higher Q-values
# 2. Value propagation: Future rewards flow backward
# 3. Discovery: Can learn Q(fire=1) > Q(fire=0) even if rarely tried
```

**Result**: The discrete head can discover that firing in situation X yields better rewards than not firing, even if the expert/past DQN rarely did it.

---

### **Continuous Head (Spinner)** ❌

```python
# Supervised target uses EXECUTED ACTION
continuous_targets = continuous_actions

# This means:
# Loss = MSE(predicted_spinner, executed_spinner)
```

**What this does**:
- **NOT**: "Learn to predict spinner values that maximize reward"
- **INSTEAD**: "Learn to mimic whatever spinner value was executed"

**Result**: The continuous head is just a function approximator trying to reproduce the distribution of executed spinner actions. It CANNOT discover that spinner=+0.5 is better than spinner=+0.3 unless that was tried AND stored in the buffer.

---

## Can It Learn Beyond Expert? Analysis by Component

### **Scenario 1: Discrete Actions (Fire/Zap)** ✅

**Question**: Can DQN learn to fire more aggressively than expert?

**Answer**: **YES!**

1. Epsilon exploration occasionally fires when expert wouldn't
2. If firing yields high reward, TD error is large: `r + γ*Q(s',a') >> Q(s,fire)`
3. Q(s, fire) gets updated upward via gradient descent
4. DQN learns to fire more often in that situation
5. Over time, DQN discovers superior fire/zap policy

**Evidence**: This IS standard Q-learning. Works as designed. ✅

---

### **Scenario 2: Continuous Actions (Spinner)** ❌

**Question**: Can DQN learn to spin faster/differently than expert?

**Answer**: **NO!** (with current setup)

**Why not?**

1. **Training Target**: `continuous_targets = continuous_actions`
   - This is the action that was EXECUTED (expert or DQN's past prediction)
   - No reward signal! No TD learning!

2. **What the network learns**:
   - "In state S, the executed spinner value was X"
   - Network becomes a **conditional average** of executed actions
   - If expert spins +0.3 and DQN explores +0.5, network learns ~+0.35
   - But WHY should it predict +0.3 vs +0.5? No gradient from reward!

3. **Exploration helps, but is limited**:
   - Epsilon exploration adds Gaussian noise to predictions: `action = pred + N(0, epsilon*0.9)`
   - This adds diverse spinner values to buffer (good for coverage!)
   - But loss is still `MSE(pred, executed_value)`, not reward-based
   - Network learns to reproduce the **distribution** of executed actions
   - High-reward actions appear more (survival bias) but gradient doesn't use rewards directly

4. **Example failure case**:
   ```
   State: Enemy at segment 5, player at 3
   Expert: spin = +0.2 (cautious)
   Optimal: spin = +0.7 (aggressive, yields +10 reward)
   
   Problem: DQN never learns spin=+0.7 is better because:
   - If DQN predicts 0.7, but buffer only has examples of 0.2
   - Loss = MSE(0.7, 0.2) = high → gradient pushes DOWN to 0.2
   - Reward signal doesn't factor in!
   ```

---

## Why Does It Work At All Then?

Despite the behavioral cloning issue, the system still functions reasonably well because:

1. **Discrete head is learning correctly** via TD learning
   - Fire/zap decisions improve over time
   - Most of the strategic value comes from knowing WHEN to fire

2. **Expert's spinner policy is decent**
   - 15% expert actions provide reasonable spinner examples
   - Continuous head mimics expert's spinner strategy

3. **Epsilon adds diversity AND creates survival bias**
   - 15% epsilon adds Gaussian noise: `action = pred + N(0, 0.135)` at ε=0.15
   - Explores action space continuously (not discrete jumps)
   - **Survival bias**: States where spinner values work well → longer episodes → more buffer samples
   - Network implicitly learns good spinner distributions through frequency weighting
   - BUT: Still no direct reward gradient (learns distribution, not reward maximization)

4. **Spinner might be less critical**
   - If optimal spinner value doesn't vary much by state, behavioral cloning suffices
   - Example: If spinner = +0.3 ± 0.2 is "good enough" for most states
   - Discrete head (fire/zap) may carry most of the strategic weight

---

## The User's Defense

> "But I maintain it learns from the replay buffer, which right now is a mix of 85% DQN and 15% expert, and that includes 15% epsilon, so it should 'discover' new behaviors."

**Evaluation**:

✅ **Correct for discrete head**: The discrete head DOES learn from the replay buffer via TD learning. It CAN discover that certain fire/zap combinations are better than what expert does.

❌ **Incorrect for continuous head**: The continuous head does NOT learn from rewards. It learns to mimic whatever actions are in the buffer. Epsilon exploration helps diversify the buffer, but the network is still just learning the **distribution** of executed actions, not the **reward-maximizing** action.

**Key insight**: Having diverse data in the buffer is necessary but NOT sufficient for RL. You also need the LOSS FUNCTION to incorporate rewards (TD error).

---

## Proof of Behavioral Cloning

Compare to true RL for continuous actions:

### **Option A: Current (Behavioral Cloning)**
```python
continuous_targets = continuous_actions  # Action that was executed
c_loss = F.mse_loss(continuous_pred, continuous_targets)
```

### **Option B: TD Learning for Continuous Actions**
```python
# Use a critic to evaluate continuous action quality
continuous_q_value = self.critic(state, continuous_pred)  # Predicted Q(s, a)
continuous_target_q = reward + gamma * max_Q(next_state)  # TD target
c_loss = F.mse_loss(continuous_q_value, continuous_target_q)  # TD error
```

### **Option C: Policy Gradient for Continuous Actions**
```python
# Train continuous head as policy, use discrete Q-value as advantage
log_prob = -F.mse_loss(continuous_pred, continuous_actions)  # Log probability
advantage = discrete_q_selected - discrete_q_selected.mean()  # Advantage estimate
c_loss = -(log_prob * advantage.detach()).mean()  # Policy gradient
```

**Current code uses Option A** → Behavioral cloning, not RL.

---

## Consequences

### What Works ✅

1. **Discrete head learns optimal fire/zap policy**
   - Can surpass expert in deciding when to shoot
   - TD learning enables discovery

2. **System stability**
   - Behavioral cloning is more stable than true continuous RL
   - No continuous action exploration causing catastrophic failures

3. **Decent performance**
   - If expert's spinner policy is reasonable, mimicking it works

### What Doesn't Work ❌

1. **Continuous head cannot discover better spinner strategies**
   - Cannot learn "spin faster here yields more points"
   - Stuck at expert/DQN's executed distribution

2. **Suboptimal spinner usage**
   - If optimal spinner = +0.8 but expert uses +0.3
   - DQN will learn +0.3 (weighted by frequency in buffer)
   - Even if +0.8 yields 2x more reward!

3. **Wasted learning potential**
   - Replay buffer contains reward signals
   - Continuous head ignores them

---

## Recommendations

### **Option 1: Keep Current (Conservative)**

**Pros**:
- Stable, works reasonably well
- Simpler than true continuous RL

**Cons**:
- Continuous head cannot improve beyond demonstrated actions

**When to use**: If spinner control is not critical to performance, or if expert's spinner policy is already near-optimal.

---

### **Option 2: Add Continuous TD Learning (Aggressive)**

**Modify train_step()** to compute TD-based continuous targets:

```python
# Current (behavioral cloning)
continuous_targets = continuous_actions

# Proposed (TD learning)
# Treat continuous action as part of state-action pair
# Use reward to train continuous head
with torch.no_grad():
    # Get Q-value for executed continuous action
    _, next_continuous_target = self.qnetwork_target(next_states)
    # Compute TD target: reward + gamma * Q(s', a')
    # For continuous, we need a critic or use discrete Q as proxy
    continuous_targets = rewards + (gamma_boot * discrete_q_next_max * (1 - dones))
    # This requires rethinking the architecture - continuous Q-function

# Alternative: Use discrete Q-value as "advantage" for continuous action
# This assumes discrete action quality is correlated with continuous quality
```

**Implementation complexity**: High (requires architecture changes)

**Reward**: Continuous head can learn reward-maximizing spinner policy

---

### **Option 3: Hybrid Approach (Balanced)**

**Mix behavioral cloning + reward signal**:

```python
# Compute both targets
bc_targets = continuous_actions  # Behavioral cloning
reward_signal = rewards  # Reward achieved

# Weighted combination
continuous_targets = 0.7 * bc_targets + 0.3 * reward_signal * continuous_pred.sign()
# This nudges continuous action toward higher rewards while staying close to executed actions
```

**Pros**:
- Adds reward signal without full RL complexity
- More stable than pure TD learning

**Cons**:
- Heuristic, not principled RL
- Unclear if it actually works

---

### **Option 4: Decouple Continuous Head (Advanced)**

Treat continuous head as separate RL problem:

```python
# Train continuous head with actor-critic
# Actor: continuous_pred (current)
# Critic: discrete Q-values (discrete head)

# Use discrete Q as baseline for continuous action
advantage = discrete_q_selected - discrete_q_selected.mean()
log_prob = -0.5 * ((continuous_pred - continuous_actions) ** 2)  # Gaussian policy
c_loss = -(log_prob * advantage.detach()).mean()  # REINFORCE with baseline
```

**Pros**:
- Principled policy gradient approach
- Continuous head learns from rewards

**Cons**:
- Requires careful tuning (policy gradient is unstable)
- May need separate optimizer for continuous head

---

## Validation: How to Test the Claim

### **Experiment 1: Pure Expert Spinner**

1. Force `continuous_actions` in replay buffer to always be expert values
2. Train for 10M frames
3. **Prediction**: Continuous head will mimic expert perfectly
4. **Result**: If this works, confirms behavioral cloning hypothesis

### **Experiment 2: Inverted Spinner**

1. Modify stored `continuous_actions` to be `-1.0 * executed_action`
2. Train for 5M frames
3. **Prediction**: Network will learn to predict inverted values
4. **Observation**: If loss decreases despite negative correlation with rewards, confirms no reward learning

### **Experiment 3: Continuous Q-Value Analysis**

1. After training, for a given state S:
   - Sweep continuous action from -0.9 to +0.9
   - Measure discrete Q-value for each
2. **Prediction (current)**: Continuous pred does NOT correlate with Q-value
3. **Prediction (true RL)**: Continuous pred WOULD maximize Q-value

---

## Conclusion

**The claim is PARTIALLY CORRECT:**

✅ **Discrete head (fire/zap)**: True RL via TD learning - CAN learn beyond expert  
❌ **Continuous head (spinner)**: Behavioral cloning - CANNOT learn beyond demonstrated actions

**User's argument**:
- Correct that replay buffer has 85% DQN + 15% expert + epsilon exploration
- **BUT** this doesn't help continuous head because loss function ignores rewards
- Diversity in buffer ≠ Learning from rewards

**Bottom line**:
- The system CAN learn better fire/zap policies than the expert (discrete head)
- The system CANNOT learn better spinner policies than what's executed (continuous head)
- Overall performance can still exceed expert if discrete improvements dominate

**If spinner control is critical**: Consider implementing Option 2 or 4 above to enable true RL for continuous actions.

---

## Code References

**Behavioral cloning evidence**:
- `aimodel.py:1351` - `continuous_targets = continuous_actions`
- `aimodel.py:1368` - `c_loss = F.mse_loss(continuous_pred, continuous_targets)`

**Action storage**:
- `socket_server.py:471` - `state['last_action_hybrid'] = (discrete_action, continuous_spinner)`
- `socket_server.py:321` - `self.agent.step(..., float(ca), ...)`

**Network architecture**:
- `aimodel.py:364-442` - `HybridDQN` class definition
- `aimodel.py:427-430` - Continuous head layers

---

**Authors Note**: This analysis is based on code inspection as of the current session. The finding is clear: continuous head uses supervised learning (behavioral cloning) on executed actions, not reinforcement learning on rewards.
