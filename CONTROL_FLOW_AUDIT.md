# Control Flow Audit Report
## Tempest AI - Discrete Spinner Architecture

**Date**: October 3, 2025  
**Branch**: `discrete`  
**Auditor**: AI Assistant

---

## Executive Summary

### 🚨 **CRITICAL ISSUES FOUND**

1. **Action Storage Format Mismatch** ⚠️
   - Socket server stores `continuous_spinner` (float) in `last_action_hybrid`
   - Agent expects `spinner_action` (int 0-8) in replay buffer
   - **Impact**: Training on wrong data format, corrupted experiences

2. **Expert Action Not Quantized** ⚠️
   - Expert system outputs continuous spinner (-0.9 to +0.9)
   - Stored directly as float instead of being quantized to discrete action
   - **Impact**: Expert experiences stored in wrong format

3. **Diversity Bonus Using Wrong Format** ⚠️
   - `calculate_diversity_bonus()` called with continuous float
   - Function expects discrete actions for tracking
   - **Impact**: Diversity tracking ineffective/broken

4. **N-Step Buffer Format Inconsistency** ⚠️
   - N-step buffer receives continuous float via `aux_action` parameter
   - Should receive discrete spinner action
   - **Impact**: Multi-step returns use wrong action format

---

## Detailed Audit

### 1. OOB Boundary Receipt ✅ (Mostly OK)

**Location**: `parse_frame_data()` in `aimodel.py:2117`

```python
# Incoming data structure:
action=(bool(fire), bool(zap), spinner)  # spinner is float from game
```

**Status**: ✅ **Correct** - Game sends continuous spinner, parsed as float

**Note**: This is fine because the game itself outputs continuous values. Quantization should happen **after** receipt, not at boundary.

---

### 2. Action Generation ⚠️ (CRITICAL ISSUES)

**Location**: `socket_server.py:384-471`

#### Issue 2A: DQN Action Selection ✅ (Correct)

```python
# Line 440
da, ca = self.agent.act(frame.state, epsilon)
discrete_action, continuous_spinner = int(da), float(ca)
```

**Status**: ✅ **Correct** - Agent returns discrete action mapped to continuous value

**What happens inside `act()`**:
```python
# aimodel.py:1128
firezap_action = firezap_q.argmax()  # int 0-3
spinner_action = spinner_q.argmax()  # int 0-8
spinner_value = SPINNER_MAPPING[spinner_action]  # float -0.9 to +0.9
return int(firezap_action), float(spinner_value)
```

#### Issue 2B: Expert Action Selection ⚠️ **CRITICAL BUG**

```python
# Line 405-409
fire, zap, spin = get_expert_action(...)
discrete_action = fire_zap_to_discrete(fire, zap)
continuous_spinner = float(spin)  # ❌ BUG: Should quantize!
```

**Problem**: Expert outputs continuous spinner (-0.9 to +0.9) which is stored directly as float, but agent expects discrete action index (0-8).

**What should happen**:
```python
from config import quantize_spinner_action
firezap_action, spinner_action = get_hybrid_expert_action(...)  # Returns discrete actions
continuous_spinner = SPINNER_MAPPING[spinner_action]  # Map for game control
```

---

### 3. Action Storage ⚠️ **CRITICAL BUG**

**Location**: `socket_server.py:471`

```python
# Current (WRONG):
state['last_action_hybrid'] = (discrete_action, continuous_spinner)  # (int, float)

# What agent.step() expects:
# aimodel.py:1205
def step(self, state, firezap_action, spinner_action, reward, next_state, done):
    self.memory.push(state, firezap_action, spinner_action, ...)
    #                        ^int 0-3       ^int 0-8
```

**Problem**: Storing continuous float, but replay buffer expects discrete int!

**Impact**:
- Replay buffer receives float values like `0.6` instead of ints like `7`
- When sampled, these are cast to long tensors with wrong values
- Training learns on corrupted action representations

---

### 4. Replay Buffer Storage ⚠️ **DATA TYPE MISMATCH**

**Location**: 
- `PrioritizedReplayMemory.push()` - `aimodel.py:534`
- `HybridReplayBuffer.push()` - `aimodel.py:729`

```python
# Current buffer definition (CORRECT):
self.spinner_actions = np.empty((capacity,), dtype=np.int32)  # ✅

# What's being stored (WRONG):
spinner_idx = int(spinner_action)  # Casting 0.6 → 0, 0.9 → 0, etc.
```

**Problem**: When float is cast to int:
- `0.6` → `0` (should be `7`)
- `0.3` → `0` (should be `6`)
- `-0.6` → `0` (should be `1`)
- `-0.9` → `0` (should be `0`)

**Most spinner actions get stored as action 0 (-0.9)!**

---

### 5. N-Step Buffer ⚠️ **FORMAT INCONSISTENCY**

**Location**: `socket_server.py:295-309`

```python
# Line 295
experiences = state['nstep_buffer'].add(
    state['last_state'],
    int(da),                # ✅ Fire/zap action (correct)
    total_reward,
    frame.state,
    frame.done,
    aux_action=float(ca)    # ❌ Continuous spinner (wrong format)
)

# Line 309
self.agent.step(exp_state, exp_action, exp_continuous, ...)
#                          ^int        ^float (WRONG!)
```

**Problem**: N-step buffer stores continuous float as `aux_action`, but agent expects discrete int.

---

### 6. Diversity Bonus ⚠️ **WRONG PARAMETER TYPE**

**Location**: `socket_server.py:283-289`

```python
diversity_bonus = self.agent.calculate_diversity_bonus(
    state['last_state'], da, ca  # da=int, ca=float
)

# But function signature expects:
def calculate_diversity_bonus(self, state, firezap_action, spinner_value):
    # spinner_value gets rounded: round(float(0.6), 1) = 0.6
    # This should work, but tracking becomes inconsistent
```

**Problem**: Function receives continuous float, rounds it, but discrete tracking would be cleaner and match architecture.

---

## Root Cause Analysis

### The Disconnect

The architecture change from **continuous to discrete** was incomplete:

1. ✅ **Network**: Outputs discrete Q-values, maps to continuous for game control
2. ✅ **Replay Buffer**: Stores discrete int spinner actions  
3. ❌ **Socket Server**: Still treats spinner as continuous throughout
4. ❌ **Storage**: Stores continuous float instead of discrete int

### The Fix Strategy

**Store discrete actions internally, map to continuous only at game boundary:**

```
Expert/DQN → Discrete Action (int) → Store in Replay → Train on Discrete
                    ↓
            Map to Continuous (float) → Send to Game
```

---

## Required Fixes

### Fix 1: Update Expert Action Generation ⚠️ **HIGH PRIORITY**

**File**: `socket_server.py:405-409`

**Current**:
```python
fire, zap, spin = get_expert_action(...)
discrete_action = fire_zap_to_discrete(fire, zap)
continuous_spinner = float(spin)
```

**Fixed**:
```python
from config import quantize_spinner_action
fire, zap, spin = get_expert_action(...)
firezap_action = fire_zap_to_discrete(fire, zap)
spinner_action = quantize_spinner_action(float(spin))  # Quantize to 0-8
spinner_value = SPINNER_MAPPING[spinner_action]  # Map for game
```

### Fix 2: Update DQN Action Variable Names ⚠️ **MEDIUM PRIORITY**

**File**: `socket_server.py:440-460`

**Current**:
```python
da, ca = self.agent.act(frame.state, epsilon)
discrete_action, continuous_spinner = int(da), float(ca)
```

**Issue**: Variable name `continuous_spinner` is misleading - it's actually discrete action mapped to continuous value.

**Fixed** (for clarity):
```python
firezap_action, spinner_value = self.agent.act(frame.state, epsilon)
# act() already returns (int, float) where float is mapped from discrete action
```

### Fix 3: Update Action Storage ⚠️ **HIGH PRIORITY**

**File**: `socket_server.py:471`

**Current**:
```python
state['last_action_hybrid'] = (discrete_action, continuous_spinner)  # (int, float)
```

**Problem**: Need to store discrete spinner action, not continuous value.

**Fixed**:
```python
# Store discrete actions for training
state['last_action_hybrid'] = (firezap_action, spinner_action)  # (int, int)
# Also store continuous value for game control
state['last_spinner_value'] = spinner_value  # float for game
```

**Alternative** (simpler): Reconstruct discrete action from continuous when needed:
```python
state['last_action_hybrid'] = (firezap_action, spinner_action)  # Store discrete
# Reconstruct continuous when sending to game (already done at line 474)
```

### Fix 4: Update agent.step() Calls ⚠️ **HIGH PRIORITY**

**File**: `socket_server.py:276, 309, 321`

**Current**:
```python
da, ca = state['last_action_hybrid']  # ca is float
self.agent.step(state['last_state'], int(da), float(ca), ...)
```

**Fixed**:
```python
firezap_action, spinner_action = state['last_action_hybrid']  # Both int
self.agent.step(state['last_state'], firezap_action, spinner_action, ...)
```

### Fix 5: Update N-Step Buffer ⚠️ **HIGH PRIORITY**

**File**: `socket_server.py:295-309`

**Current**:
```python
experiences = state['nstep_buffer'].add(
    ...,
    aux_action=float(ca)  # float
)

# Later:
self.agent.step(exp_state, exp_action, exp_continuous, ...)
```

**Fixed**:
```python
firezap_action, spinner_action = state['last_action_hybrid']
experiences = state['nstep_buffer'].add(
    state['last_state'],
    int(firezap_action),
    total_reward,
    frame.state,
    frame.done,
    aux_action=int(spinner_action)  # Discrete spinner action
)

# Later:
for item in experiences:
    if len(item) == 6:
        exp_state, exp_firezap, exp_spinner, exp_reward, exp_next_state, exp_done = item
        self.agent.step(exp_state, exp_firezap, exp_spinner, exp_reward, exp_next_state, exp_done)
```

### Fix 6: Update Diversity Bonus Call ⚠️ **LOW PRIORITY**

**File**: `socket_server.py:283-289`

**Current**:
```python
diversity_bonus = self.agent.calculate_diversity_bonus(
    state['last_state'], da, ca  # ca is float
)
```

**Fixed** (after other fixes):
```python
firezap_action, spinner_action = state['last_action_hybrid']
# calculate_diversity_bonus already expects float spinner_value
spinner_value = SPINNER_MAPPING[spinner_action]
diversity_bonus = self.agent.calculate_diversity_bonus(
    state['last_state'], firezap_action, spinner_value
)
```

---

## Implementation Plan

### Phase 1: Critical Fixes (Required for Correct Training)

1. ✅ Import `SPINNER_MAPPING` and `quantize_spinner_action` at top of socket_server.py
2. ✅ Fix expert action quantization (Fix 1)
3. ✅ Update action storage to store discrete actions (Fix 3)
4. ✅ Update all agent.step() calls to pass discrete actions (Fix 4)
5. ✅ Fix N-step buffer to use discrete actions (Fix 5)

### Phase 2: Cleanup (Nice to Have)

6. Update variable names for clarity (Fix 2)
7. Update diversity bonus call (Fix 6)
8. Add assertions to verify discrete action ranges

### Phase 3: Validation

- Add logging to verify correct data types
- Check replay buffer contents
- Verify training metrics improve
- Test expert and DQN actions separately

---

## Testing Checklist

### Unit Tests

- [ ] Expert action quantization: `spin=0.75` → `action=7` → `value=0.6`
- [ ] DQN action already discrete: `action=7` → `value=0.6`
- [ ] Replay buffer stores int32: `spinner_actions.dtype == np.int32`
- [ ] Action retrieval: sampled actions are long tensors (0-8)

### Integration Tests

- [ ] Expert episode: all actions stored as discrete ints
- [ ] DQN episode: all actions stored as discrete ints  
- [ ] N-step buffer: multi-step returns use discrete actions
- [ ] Training step: no dtype errors, losses decrease

### Smoke Tests

- [ ] Run 1000 frames with expert=100%: no crashes
- [ ] Run 1000 frames with expert=0%: no crashes
- [ ] Check replay buffer: `memory.spinner_actions` contains 0-8 only
- [ ] Check training: loss values are reasonable (not NaN/Inf)

---

## Performance Impact

### Before Fixes (Current State)

- ❌ Most spinner actions stored as `0` (due to float→int truncation)
- ❌ Agent learns "always choose action 0" (-0.9 full left)
- ❌ Diversity bonus tracks wrong data
- ❌ N-step returns use corrupted actions

### After Fixes (Expected)

- ✅ All spinner actions stored correctly (0-8)
- ✅ Agent learns true Q-values for all 9 actions
- ✅ Diversity bonus tracks discrete actions properly
- ✅ N-step returns use correct discrete actions
- ✅ Training learns meaningful policies

**Expected improvement**: Significant! Going from mostly broken to fully functional.

---

## Additional Recommendations

### 1. Add Type Hints

```python
def handle_client_step(
    firezap_action: int,      # 0-3
    spinner_action: int,      # 0-8
    spinner_value: float      # -0.9 to +0.9
) -> Tuple[int, int, int]:  # game commands
    ...
```

### 2. Add Assertions

```python
assert 0 <= firezap_action <= 3, f"Invalid firezap: {firezap_action}"
assert 0 <= spinner_action <= 8, f"Invalid spinner: {spinner_action}"
assert -0.9 <= spinner_value <= 0.9, f"Invalid value: {spinner_value}"
```

### 3. Add Debug Logging

```python
if frame_counter % 1000 == 0:
    print(f"Action sample: firezap={firezap_action}, spinner={spinner_action} "
          f"(value={spinner_value:.2f}), source={action_source}")
```

### 4. Replay Buffer Inspection Tool

```python
def inspect_replay_buffer(memory):
    """Check replay buffer for data integrity"""
    if len(memory) > 0:
        unique_spinners = np.unique(memory.spinner_actions[:memory.size])
        print(f"Unique spinner actions: {sorted(unique_spinners)}")
        print(f"Expected: {list(range(9))}")
        
        if not np.all(np.isin(unique_spinners, range(9))):
            print("⚠️  WARNING: Invalid spinner actions detected!")
```

---

## Conclusion

### Severity: **CRITICAL** 🚨

The current implementation has a fundamental data format mismatch that corrupts all training data. The network is trained on discrete actions (0-8), but the replay buffer is being filled with continuous values incorrectly cast to integers (mostly zeros).

**This explains why the agent may not be learning properly!**

### Fix Priority: **IMMEDIATE**

All Phase 1 fixes should be implemented together as they form a cohesive update to the action handling pipeline. Testing should verify that:

1. Replay buffer contains only integers 0-8 in `spinner_actions`
2. Training proceeds without dtype errors
3. Agent learns diverse policies (not stuck on action 0)

### Estimated Impact

**Before**: 🔴 Training fundamentally broken (wrong data format)  
**After**: 🟢 Training should work as designed (discrete Q-learning)

---

**Audit Complete**  
**Status**: Ready for implementation  
**Next Step**: Apply fixes in socket_server.py
