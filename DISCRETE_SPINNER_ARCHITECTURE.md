# Discrete 9-Action Spinner Architecture

## Overview

This document describes the **dual-head discrete DQN architecture** implemented for Tempest AI, replacing the previous hybrid discrete-continuous approach with a fully discrete action space.

## Motivation

The continuous spinner control suffered from fundamental learning challenges:
- **Credit assignment problem**: Episode rewards don't correlate with specific spinner values
- **Regression to mean**: Reward-weighted learning caused the network to predict constant mean values
- **Behavioral cloning limitations**: Without true RL, the agent couldn't improve beyond expert behavior

**Solution**: Convert spinner control to discrete Q-learning with 9 carefully chosen action values.

---

## Architecture

### Action Space

#### Fire/Zap Head (4 discrete actions)
```python
FIRE_ZAP_MAPPING = {
    0: (0, 0),  # No fire, no zap
    1: (1, 0),  # Fire, no zap  
    2: (0, 1),  # No fire, zap
    3: (1, 1),  # Fire, zap
}
```

#### Spinner Head (9 discrete actions)
```python
SPINNER_MAPPING = {
    0: -0.9,  # Full left
    1: -0.6,  # Medium left
    2: -0.3,  # Slow left
    3: -0.1,  # Micro left
    4: 0.0,   # Still
    5: 0.1,   # Micro right
    6: 0.3,   # Slow right
    7: 0.6,   # Medium right
    8: 0.9,   # Full right
}
```

**Total action combinations**: 4 × 9 = **36 discrete actions**

---

## Network Architecture

### HybridDQN Structure

```
Input State (171 features)
       ↓
  Shared Trunk (6 layers, 512→256→128→...)
       ↓
    ┌──┴──┐
    ↓      ↓
Fire/Zap  Spinner
  Head     Head
    ↓      ↓
  4 Q's  9 Q's
```

### Dueling Architecture (Optional)

Both heads support dueling architecture:
```
Shared Features
    ↓
┌───┴───┐
↓       ↓
Value  Advantage
  V(s)  A(s,a)
    ↓      ↓
Q(s,a) = V(s) + (A(s,a) - mean(A))
```

---

## Key Components

### 1. Expert Action Quantization

Expert system outputs continuous spinner values in `[-0.9, +0.9]`. These are quantized to nearest discrete actions:

```python
def quantize_spinner_action(spinner_value: float) -> int:
    """Quantize continuous expert spinner to nearest discrete action"""
    spinner_value = max(-0.9, min(0.9, spinner_value))
    
    # Find nearest discrete value
    min_dist = float('inf')
    best_action = 4  # Default to center
    
    for action, value in SPINNER_MAPPING.items():
        dist = abs(spinner_value - value)
        if dist < min_dist:
            min_dist = dist
            best_action = action
    
    return best_action
```

**Example quantizations:**
- Expert: `-0.75` → Action `1` (`-0.6`)
- Expert: `0.25` → Action `6` (`0.3`)
- Expert: `0.0` → Action `4` (`0.0`)

### 2. Action Selection (Epsilon-Greedy)

```python
def act(self, state, epsilon=0.0):
    """Select dual discrete actions via epsilon-greedy"""
    firezap_q, spinner_q = self.qnetwork_inference(state)
    
    # Fire/Zap selection
    if random.random() < epsilon:
        firezap_action = random.randint(0, 3)
    else:
        firezap_action = firezap_q.argmax()
    
    # Spinner selection (independent exploration)
    if random.random() < epsilon:
        spinner_action = random.randint(0, 8)
    else:
        spinner_action = spinner_q.argmax()
    
    # Map spinner action to continuous value for game
    spinner_value = SPINNER_MAPPING[spinner_action]
    
    return firezap_action, spinner_value
```

### 3. Training (Dual Double DQN)

Both heads use **Double DQN** with **Huber loss**:

```python
# Forward pass
firezap_q_pred, spinner_q_pred = qnetwork_local(states)
firezap_q_selected = firezap_q_pred.gather(1, firezap_actions)
spinner_q_selected = spinner_q_pred.gather(1, spinner_actions)

# Target computation (Double DQN for both heads)
with torch.no_grad():
    # Fire/Zap target
    next_firezap_q_local, _ = qnetwork_local(next_states)
    next_firezap_actions = next_firezap_q_local.argmax(dim=1, keepdim=True)
    next_firezap_q_target, _ = qnetwork_target(next_states)
    firezap_q_next = next_firezap_q_target.gather(1, next_firezap_actions)
    
    # Spinner target
    _, next_spinner_q_local = qnetwork_local(next_states)
    next_spinner_actions = next_spinner_q_local.argmax(dim=1, keepdim=True)
    _, next_spinner_q_target = qnetwork_target(next_states)
    spinner_q_next = next_spinner_q_target.gather(1, next_spinner_actions)
    
    # TD targets
    firezap_target = reward + gamma * firezap_q_next * (1 - done)
    spinner_target = reward + gamma * spinner_q_next * (1 - done)

# Loss computation
firezap_loss = F.huber_loss(firezap_q_selected, firezap_target)
spinner_loss = F.huber_loss(spinner_q_selected, spinner_target)
total_loss = firezap_loss + spinner_loss_weight * spinner_loss
```

### 4. Replay Buffer

Both PER and standard replay buffers store:
```python
(state, firezap_action, spinner_action, reward, next_state, done)
```

- `firezap_action`: int (0-3)
- `spinner_action`: int (0-8)

---

## Advantages Over Continuous Approach

### 1. **Clear Credit Assignment**
```
Discrete Q-learning: Q(s, a) directly estimates value of taking action a
Continuous: Network predicts value, unclear which value caused reward
```

### 2. **No Regression-to-Mean**
```
Discrete: Each action's Q-value updates independently
Continuous: MSE loss drives predictions toward mean
```

### 3. **True Reinforcement Learning**
```
Discrete: Network learns "in state X, action Y yields reward Z"
Continuous: Behavioral cloning with no clear improvement signal
```

### 4. **Stable Exploration**
```
Discrete: Epsilon-greedy samples uniformly from 9 actions
Continuous: Gaussian noise can concentrate around mean
```

### 5. **Manageable Action Space**
```
36 total actions (4 × 9) vs infinite continuous space
Still fine-grained: 9 spinner positions cover full ±0.9 range
```

---

## Configuration

### Key Parameters

```python
# config.py
discrete_action_size: int = 4      # Fire/zap combinations
spinner_action_size: int = 9       # Spinner positions
spinner_loss_weight: float = 1.0   # Equal weighting for both heads

# Spinner mapping
SPINNER_MAPPING = {0: -0.9, 1: -0.6, ..., 8: 0.9}
```

### Training Settings

```python
# Both heads use same hyperparameters
learning_rate: float = 0.001
gamma: float = 0.995
epsilon_start: float = 0.25
use_double_dqn: bool = True
use_dueling: bool = True
loss_type: str = 'huber'
```

---

## Expected Benefits

### Learning Quality
- **Better credit assignment**: Direct mapping from actions to rewards
- **No mean regression**: Each Q-value learns independently
- **True policy improvement**: Can discover better-than-expert strategies

### Exploration
- **Uniform sampling**: Epsilon-greedy explores all 9 spinner positions equally
- **Stable behavior**: No Gaussian noise instability
- **Interpretable**: "Agent chose action 7 (+0.6)" vs "predicted 0.583"

### Training Efficiency
- **Faster convergence**: Discrete Q-learning typically faster than continuous control
- **Lower variance**: Deterministic action selection (no sampling noise)
- **Simpler loss**: Standard Huber loss, no special weighting schemes

---

## Migration Notes

### From Continuous to Discrete

**Replay buffer**: Old checkpoints incompatible (different storage format)
- Old: `(state, discrete, continuous_float, ...)`
- New: `(state, firezap_int, spinner_int, ...)`

**Network architecture**: Incompatible (different output heads)
- Old: `(4 discrete Q's, 1 continuous value)`
- New: `(4 firezap Q's, 9 spinner Q's)`

**Recommendation**: Start fresh training with new architecture

### Expert System Compatibility

Expert actions are automatically quantized:
```python
expert_spinner = 0.75  # Continuous
quantized = quantize_spinner_action(0.75)  # Returns 7 (maps to 0.6)
```

Minimal information loss: 9 bins provide excellent coverage of ±0.9 range

---

## Testing & Validation

### Sanity Checks

1. **Action distribution**: Verify epsilon-greedy explores uniformly across 9 actions
2. **Quantization**: Check expert actions quantize to reasonable discrete values
3. **Q-value range**: Monitor Q-values don't explode or collapse
4. **Loss convergence**: Both heads should show decreasing loss

### Expected Behavior

- **Initial phase**: Random exploration, high epsilon
- **Learning phase**: Q-values diverge, policy improves
- **Convergence**: Low epsilon, stable Q-values, improved performance over expert

---

## Performance Monitoring

### Key Metrics

```python
# DQN reward slope (per million frames)
DQN5M_slopeM: Track improvement rate

# Head-specific losses
firezap_loss: Fire/zap head learning signal
spinner_loss: Spinner head learning signal

# Q-value statistics
mean_q: Average Q-value magnitude
q_variance: Q-value spread (should increase as learning progresses)
```

### Success Indicators

✅ **DQN reward slope positive** (improving beyond expert)  
✅ **Both head losses decreasing** (learning progressing)  
✅ **Q-values stable** (no explosion or collapse)  
✅ **Agent tries diverse actions** (not stuck in local optimum)

---

## Future Enhancements

### Potential Improvements

1. **Adaptive action space**: Dynamically adjust spinner granularity
2. **Hierarchical RL**: High-level strategy controller + low-level spinner
3. **Multi-head attention**: Separate heads for different enemy patterns
4. **Distributional RL**: Model full Q-value distributions (C51, QR-DQN)

### Alternative Architectures

- **Single head with 36 outputs**: Simpler but loses dual-head modularity
- **Finer granularity (16 actions)**: More precise but larger action space
- **Coarser granularity (5 actions)**: Faster learning but less control

---

## Summary

| Aspect | Continuous Approach | Discrete Approach |
|--------|-------------------|-------------------|
| **Action Space** | Infinite ([-0.9, +0.9]) | 9 discrete positions |
| **Credit Assignment** | ❌ Unclear | ✅ Direct Q(s,a) |
| **Learning Type** | Behavioral cloning | Pure RL |
| **Exploration** | Gaussian noise | Epsilon-greedy |
| **Loss Function** | MSE (regression) | Huber (TD) |
| **Stability** | ❌ Mean regression | ✅ Independent Q's |
| **Interpretability** | ❌ Raw values | ✅ Discrete actions |
| **Total Actions** | 4 × ∞ | 4 × 9 = 36 |

**Verdict**: Discrete approach provides **clearer learning signal**, **better credit assignment**, and **true RL** capabilities while maintaining **fine-grained control** through 9 carefully chosen spinner positions.

---

## References

- **Double DQN**: van Hasselt et al., "Deep Reinforcement Learning with Double Q-learning" (2015)
- **Dueling DQN**: Wang et al., "Dueling Network Architectures for Deep RL" (2016)
- **Prioritized Experience Replay**: Schaul et al., "Prioritized Experience Replay" (2015)

---

**Implementation Date**: October 3, 2025  
**Branch**: `discrete`  
**Status**: ✅ Complete and validated
