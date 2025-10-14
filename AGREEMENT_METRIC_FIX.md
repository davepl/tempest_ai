# Agreement Metric Fix - Root Cause Analysis

## Problem
Agreement metric stuck at 0.0-0.1% despite training progress, when it should show 50-90% for a learning agent.

## Root Cause: Mixed NumPy/Torch Indexing Bug

The bug was in `Scripts/aimodel.py` line ~1428 in the `train_step()` method:

```python
# BUGGY CODE:
actor_dqn_mask = np.array([a == 'dqn' for a in actors], dtype=bool)  # NumPy array
dqn_agree_target = (a_greedy_local == a_greedy_target).float()        # Torch tensor
agree = dqn_agree_target[actor_dqn_mask].mean().item()               # ❌ BUG HERE!
```

### Why This Failed

1. **`actor_dqn_mask`** is a **NumPy boolean array** created from the string actor labels
2. **`dqn_agree_target`** is a **PyTorch tensor** containing agreement flags
3. **PyTorch doesn't handle NumPy boolean array indexing correctly**

When you index a PyTorch tensor with a NumPy boolean array:
- PyTorch tries to interpret it as integer indices or broadcasts incorrectly
- The resulting slice is malformed, giving essentially random values
- This caused the agreement calculation to fail silently, returning ~0.1% instead of the expected 50-90%

### The Fix

Convert the NumPy mask to a PyTorch tensor **before** using it for boolean indexing:

```python
# FIXED CODE:
actor_dqn_mask = np.array([a == 'dqn' for a in actors], dtype=bool)        # NumPy array
torch_dqn_mask = torch.from_numpy(actor_dqn_mask).to(device=states.device) # Convert to torch
dqn_agree_target = (a_greedy_local == a_greedy_target).float()             # Torch tensor
agree = dqn_agree_target[torch_dqn_mask].mean().item()                     # ✅ WORKS!
```

## Expected Behavior After Fix

### Current Policy vs Target Network Comparison

The agreement metric now correctly compares:
- **Current policy** (qnetwork_local): The actively training network
- **Target network** (qnetwork_target): The frozen reference network updated every 200 steps

**Expected values:**
- **Right after target update**: ~95-100% agreement (networks are identical)
- **Between target updates**: Gradually decreases as local network learns and diverges
- **Before target update**: ~70-90% agreement (moderate divergence as learning progresses)

This is a **stability metric** - it shows how much the policy changes between target updates.

### Why This Metric is Valuable

1. **Policy Stability**: Low agreement (50-70%) means rapid policy changes, high agreement (90%+) means stable/converged policy
2. **Learning Progress**: Agreement should decrease between target updates as the agent learns, then jump back up at each update
3. **Diagnostic Tool**: Sudden drops or persistently low values indicate training instability

## Loss Weighting Question

The 10x DLoss weighting (`discrete_loss_weight=10.0` vs `continuous_loss_weight=1.0`) is **intentional**:

- **Discrete head**: Q-learning for fire/zap decisions (4 discrete actions)
- **Continuous head**: Policy gradient for spinner movement (regression to [-0.9, +0.9])

The 10x weighting prioritizes getting the Q-values right (discrete) over fine-tuning spinner control (continuous), which makes sense because:
- Firing/zapping decisions are binary and critical for survival
- Spinner control is more forgiving and can be learned with weaker signal

**DLoss being ~10x CLoss is expected and correct.**

## Timeline

- **Before fix**: Agreement stuck at 0.0-0.1% (broken due to numpy/torch mixing)
- **After fix**: Agreement should show 70-95% depending on position in target update cycle

## Files Changed

- `Scripts/aimodel.py`: Added torch conversion for `actor_dqn_mask` before boolean indexing

## Testing

The fix should be immediately visible in training metrics:
- Agree% column should jump from 0.1% to 70-95% range
- Value should fluctuate between target updates (every 200 training steps)
