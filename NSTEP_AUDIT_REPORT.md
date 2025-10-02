# N-Step Implementation Audit Report

## Executive Summary

**Status**: ⚠️ **TWO SEPARATE N-STEP SYSTEMS DETECTED**

Your codebase has **two different n-step implementations** that are **NOT integrated**:

1. **Active System**: `NStepReplayBuffer` in socket_server.py (working, uses `RL_CONFIG.n_step`)
2. **Inactive System**: `add_trajectory()` method in aimodel.py (never called, uses `RL_CONFIG.nstep_length`)

The diversity bonus IS working correctly, but the new `add_trajectory` method you recently added is **not being used**.

---

## Detailed Flow Analysis

### Config Variables

**File**: `Scripts/config.py`

```python
# Line 106 - Used by NStepReplayBuffer (ACTIVE)
n_step: int = 5

# Lines 152-153 - Used by add_trajectory (INACTIVE)
nstep_enabled: bool = True
nstep_length: int = 5
```

**Issue**: Two different variable names for the same concept.

---

### Flow 1: Active N-Step System (NStepReplayBuffer)

#### Step 1: Initialization
**File**: `Scripts/socket_server.py` lines 171-173

```python
'nstep_buffer': (
    NStepReplayBuffer(RL_CONFIG.n_step, RL_CONFIG.gamma, store_aux_action=True)
    if self._server_nstep_enabled() else None
)
```

- Creates `NStepReplayBuffer` with `RL_CONFIG.n_step` (value: 5)
- Only created if `_server_nstep_enabled()` returns True

#### Step 2: Enable Check
**File**: `Scripts/socket_server.py` lines 59-77

```python
def _server_nstep_enabled(self) -> bool:
    n = int(getattr(RL_CONFIG, 'n_step', 1) or 1)
    if n <= 1:
        return False
    if hasattr(self.agent, 'n_step_buffer') and getattr(self.agent, 'n_step_buffer') is not None:
        return False  # Avoid double application
    return True
```

**Current State**: Returns `True` (n_step=5, agent doesn't have n_step_buffer)

#### Step 3: Experience Collection
**File**: `Scripts/socket_server.py` lines 285-310

```python
if self._server_nstep_enabled() and state.get('nstep_buffer') is not None:
    # Add experience to n-step buffer and get matured experiences
    experiences = state['nstep_buffer'].add(
        state['last_state'],
        int(da),
        total_reward,  # Already includes diversity bonus
        frame.state,
        frame.done,
        aux_action=float(ca)
    )
    
    # Push all matured experiences to agent
    if self.agent and experiences:
        for item in experiences:
            if len(item) == 6:
                exp_state, exp_action, exp_continuous, exp_reward, exp_next_state, exp_done = item
                self.agent.step(exp_state, exp_action, exp_continuous, exp_reward, exp_next_state, exp_done)
```

**How It Works**:
1. Each frame adds transition to sliding window buffer
2. When buffer has ≥5 transitions, emits one matured experience with n-step return:
   - `R_n = r_t + γ*r_{t+1} + γ²*r_{t+2} + γ³*r_{t+3} + γ⁴*r_{t+4}`
3. Matured experience goes to `agent.step()` → `memory.push()` → replay buffer

#### Step 4: Storage
**File**: `Scripts/aimodel.py` line 1175

```python
def step(self, state, discrete_action, continuous_action, reward, next_state, done):
    """Add experience to memory and queue training"""
    self.memory.push(state, discrete_action, continuous_action, reward, next_state, done)
```

The `reward` parameter here is **already the n-step return** computed by NStepReplayBuffer.

#### Step 5: Training
**File**: `Scripts/aimodel.py` (train_step method samples from memory)

Samples random batch from `PrioritizedReplayMemory`, which contains experiences with n-step returns.

**✅ VERDICT: This system is WORKING CORRECTLY**

---

### Flow 2: Inactive N-Step System (add_trajectory)

#### Step 1: Agent Initialization
**File**: `Scripts/aimodel.py` lines 1086-1087

```python
self.nstep_enabled = getattr(RL_CONFIG, 'nstep_enabled', True)
self.nstep_length = getattr(RL_CONFIG, 'nstep_length', 5)
```

These variables are set but **never checked** anywhere in the code.

#### Step 2: Method Definition
**File**: `Scripts/aimodel.py` lines 630-667

```python
def add_trajectory(self, trajectory, n_step=5, gamma=0.995):
    """Add trajectory with n-step returns for better credit assignment"""
    n_step = int(n_step)  # Good: converts to int
    
    for i in range(len(trajectory)):
        state, discrete_action, continuous_action, reward, next_state, done = trajectory[i]
        
        # Calculate n-step return
        n_step_return = 0.0
        discount = 1.0
        final_next_state = next_state
        final_done = done
        
        # Look ahead up to n_step frames
        for j in range(min(n_step, len(trajectory) - i)):
            _, _, _, r, ns, d = trajectory[i + j]
            n_step_return += discount * r
            discount *= gamma
            final_next_state = ns
            final_done = d
            if d:
                break
        
        # Store with n-step return instead of single-step reward
        self.push(state, discrete_action, continuous_action, n_step_return, final_next_state, final_done)
```

**✅ Logic looks correct** (verified by your test suite - 6/6 tests passing)

#### Step 3: Usage
**Search Results**: `grep add_trajectory` found **0 call sites**

**❌ VERDICT: This method is NEVER CALLED**

---

### Flow 3: Diversity Bonus (Working)

#### Step 1: Calculation
**File**: `Scripts/socket_server.py` lines 273-280

```python
diversity_bonus = 0.0
if self.agent and hasattr(self.agent, 'calculate_diversity_bonus'):
    try:
        diversity_bonus = self.agent.calculate_diversity_bonus(
            state['last_state'], da, ca
        )
    except Exception:
        pass

total_reward = float(frame.reward) + diversity_bonus
```

#### Step 2: Integration
The diversity bonus is added to reward **before** n-step calculation:

```
Game Reward: 100
+ Diversity Bonus: 0.5
= Total Reward: 100.5
→ Passed to NStepReplayBuffer
→ Accumulated over n steps: R_n = 100.5 + γ*r_1 + γ²*r_2 + ...
```

**✅ VERDICT: Diversity bonus is correctly integrated with n-step returns**

---

## Issues Identified

### Issue 1: Duplicate N-Step Configuration ⚠️

**Problem**: Two sets of config variables
- `n_step` (used by active system)
- `nstep_enabled`, `nstep_length` (unused)

**Impact**: Confusing, potential for bugs if someone tries to use `nstep_length`

**Recommendation**: Remove unused variables or integrate them

### Issue 2: Unused add_trajectory Method ❌

**Problem**: Method exists, tests pass, but it's never called

**Impact**: 
- Code bloat
- User confusion (you added it thinking it would work)
- Tests verify logic that isn't executed

**Recommendation**: Either integrate it or remove it

### Issue 3: Inconsistent Naming Convention ⚠️

**Problem**: 
- `n_step` (with underscore)
- `nstep_enabled` (without underscore)
- `nstep_length` (without underscore)

**Impact**: Harder to grep/search, cognitive overhead

**Recommendation**: Standardize on one convention

### Issue 4: Agent Attributes Not Used ❌

**Problem**: Agent has `self.nstep_enabled` and `self.nstep_length` but they're never checked

**Impact**: Dead code, misleading state

---

## Current State Summary

### ✅ What's Working:

1. **N-step returns are active**: `NStepReplayBuffer` correctly computes 5-step returns
2. **Diversity bonus is active**: Novel actions get reward bonuses
3. **Integration is correct**: Diversity bonus included in n-step calculation
4. **Hotkeys work**: 'n' and 'd' keys toggle agent attributes (but 'n' doesn't affect behavior)

### ❌ What's Not Working:

1. **add_trajectory is unused**: Your recently added method isn't called
2. **nstep_length is ignored**: Config variable has no effect
3. **Toggle 'n' key does nothing**: Sets `agent.nstep_enabled = False` but nothing checks it
4. **Two parallel systems**: Confusion between NStepReplayBuffer and add_trajectory

### ⚠️ What's Misleading:

1. Test suite passes for add_trajectory but method is never used in production
2. Config has two n_step variables with different names
3. Agent has nstep attributes that don't affect behavior

---

## Recommendations

### Option A: Use Only NStepReplayBuffer (Simplest)

**Remove**:
- `add_trajectory()` method from `PrioritizedReplayMemory`
- `nstep_enabled`, `nstep_length` from config
- `self.nstep_enabled`, `self.nstep_length` from agent

**Keep**:
- `NStepReplayBuffer` (current working system)
- `RL_CONFIG.n_step` variable

**Hotkey Change**:
- Make 'n' key modify `RL_CONFIG.n_step` between 1 and 5

### Option B: Use Only add_trajectory (More Control)

**Remove**:
- `NStepReplayBuffer` usage from socket_server
- Server-side n-step processing

**Implement**:
- Buffer complete trajectories in socket_server
- Call `memory.add_trajectory()` on episode end
- Use `nstep_enabled` to gate this behavior

**Benefits**:
- Full trajectory visibility
- Can apply different n-step values per trajectory
- Simpler architecture (one system)

### Option C: Keep Both (Most Flexible, Most Complex)

**Implement**:
- Add toggle in socket_server: if `agent.nstep_enabled` use add_trajectory, else use NStepReplayBuffer
- Rename `n_step` → `server_nstep` and `nstep_length` → `agent_nstep`
- Make 'n' key switch between modes

**Benefits**:
- Can A/B test different n-step implementations
- Maximum flexibility

**Drawbacks**:
- More complex code
- Higher maintenance burden
- Potential for bugs

---

## Critical Finding: The 'n' Hotkey

**Current behavior**: Pressing 'n' toggles `agent.nstep_enabled` but **has no effect on training**

**Why**: Nothing checks `agent.nstep_enabled` - the NStepReplayBuffer is always active

**To make it work**:

```python
# In socket_server.py, replace _server_nstep_enabled:
def _server_nstep_enabled(self) -> bool:
    # Check agent preference first
    if hasattr(self.agent, 'nstep_enabled'):
        if not getattr(self.agent, 'nstep_enabled', True):
            return False
    
    n = int(getattr(RL_CONFIG, 'n_step', 1) or 1)
    if n <= 1:
        return False
    return True
```

This would make the 'n' key actually disable n-step processing.

---

## Conclusion

You have a **working n-step system** (NStepReplayBuffer) that correctly:
1. Accumulates 5-step returns
2. Integrates diversity bonuses
3. Stores experiences in replay buffer

You also have an **unused n-step system** (add_trajectory) that:
1. Has correct logic (tests pass)
2. Is never called
3. Creates confusion

**Recommended Action**: Choose Option A (simplest) or implement the hotkey fix to make 'n' functional.
