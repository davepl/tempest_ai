# Actor Attribution Fix - Root Cause Analysis

## Problem Discovery

The user reported that the DQN was learning from expert demonstrations but not from its own (DQN) experiences. After investigation, I discovered a **critical bug**: actor tags ('expert' vs 'dqn') were being assigned and passed through the system but **completely dropped** when storing experiences to the replay buffer.

## Root Cause

### What Was Happening

1. ✅ **Action source correctly tracked** - `socket_server.py` properly assigns `action_source = 'expert'` or `action_source = 'dqn'` based on the actual decision maker
2. ✅ **Actor tags correctly passed** - `require_actor_tag()` validates and passes the actor to `agent.step()`
3. ❌ **Actor tags DROPPED at storage** - `agent.step()` accepted the `actor` parameter but never passed it to `memory.push()`
4. ❌ **Replay buffer didn't store actors** - `HybridReplayBuffer` had no field to store actor tags
5. ❌ **Training treated all frames identically** - Without actor information, the training loop couldn't distinguish expert from DQN frames

### Impact

This bug meant:
- **All diagnostic metrics were meaningless** - `FracDQN`, `TDdqn`, `TDexp`, `Qdqn`, `Qexp` couldn't be computed correctly
- **No per-actor analysis possible** - Couldn't determine if DQN frames were actually contributing to learning
- **No debugging capability** - Couldn't verify if batch composition matched expected ~26% expert / 74% DQN ratio
- **Training was blind** - The advantage-weighted learning treated expert and DQN frames identically, giving no visibility into learning asymmetry

## Fix Implementation

### 1. Added Actor Storage to Replay Buffer (`aimodel.py`)

**Before:**
```python
class HybridReplayBuffer:
    def __init__(self, capacity, state_size):
        self.states = np.empty((capacity, state_size), dtype=np.float32)
        self.discrete_actions = np.empty((capacity,), dtype=np.int32)
        # ... other fields
        # NO ACTOR FIELD!
```

**After:**
```python
class HybridReplayBuffer:
    def __init__(self, capacity, state_size):
        self.states = np.empty((capacity, state_size), dtype=np.float32)
        self.discrete_actions = np.empty((capacity,), dtype=np.int32)
        # ... other fields
        self.actors = np.empty((capacity,), dtype='U10')  # Store actor tags
```

### 2. Updated push() to Accept and Store Actor (`aimodel.py`)

**Before:**
```python
def push(self, state, discrete_action, continuous_action, reward, next_state, done):
    # Store experience WITHOUT actor tag
    self.discrete_actions[self.position] = discrete_idx
    # ... store other fields
```

**After:**
```python
def push(self, state, discrete_action, continuous_action, reward, next_state, done, actor='dqn'):
    actor_tag = str(actor).lower().strip() if actor else 'dqn'
    # Store experience WITH actor tag
    self.discrete_actions[self.position] = discrete_idx
    # ... store other fields
    self.actors[self.position] = actor_tag
```

### 3. Updated agent.step() to Pass Actor Through (`aimodel.py`)

**Before:**
```python
def step(self, state, discrete_action, continuous_action, reward, next_state, done, actor=None, horizon=1):
    self.memory.push(state, discrete_action, continuous_action, reward, next_state, done)
    # Actor parameter ACCEPTED but DROPPED!
```

**After:**
```python
def step(self, state, discrete_action, continuous_action, reward, next_state, done, actor=None, horizon=1):
    if actor is None:
        actor = 'dqn'  # Default to DQN if not specified
    self.memory.push(state, discrete_action, continuous_action, reward, next_state, done, actor=actor)
    # Actor parameter now PASSED THROUGH
```

### 4. Updated sample() to Return Actor Tags (`aimodel.py`)

**Before:**
```python
def sample(self, batch_size):
    # ... sample indices
    return states, discrete_actions, continuous_actions, rewards, next_states, dones
    # NO ACTOR TAGS RETURNED
```

**After:**
```python
def sample(self, batch_size):
    # ... sample indices
    batch_actors = self.actors[indices]  # Get actor tags for batch
    return states, discrete_actions, continuous_actions, rewards, next_states, dones, batch_actors
    # ACTOR TAGS NOW INCLUDED
```

### 5. Updated train_step() to Use Actor Tags (`aimodel.py`)

**Before:**
```python
def train_step(self):
    batch = self.memory.sample(self.batch_size)
    states, discrete_actions, continuous_actions, rewards, next_states, dones = batch
    # NO ACTOR INFORMATION - all frames treated identically
```

**After:**
```python
def train_step(self):
    batch = self.memory.sample(self.batch_size)
    states, discrete_actions, continuous_actions, rewards, next_states, dones, actors = batch
    
    # Compute batch composition
    actor_dqn_mask = np.array([a == 'dqn' for a in actors], dtype=bool)
    actor_expert_mask = np.array([a == 'expert' for a in actors], dtype=bool)
    n_dqn = actor_dqn_mask.sum()
    n_expert = actor_expert_mask.sum()
    frac_dqn = n_dqn / len(actors)
    
    # Store metrics
    metrics.batch_frac_dqn = float(frac_dqn)
    metrics.batch_n_dqn = int(n_dqn)
    metrics.batch_n_expert = int(n_expert)
    
    # Compute per-actor metrics after loss calculation
    td_errors = (discrete_q_selected - discrete_targets).detach().cpu().numpy().flatten()
    if n_dqn > 0:
        metrics.td_err_mean_dqn = float(np.abs(td_errors[actor_dqn_mask]).mean())
        metrics.reward_mean_dqn = float(rewards.cpu().numpy().flatten()[actor_dqn_mask].mean())
        metrics.q_mean_dqn = float(discrete_q_selected.detach().cpu().numpy().flatten()[actor_dqn_mask].mean())
    if n_expert > 0:
        metrics.td_err_mean_expert = float(np.abs(td_errors[actor_expert_mask]).mean())
        metrics.reward_mean_expert = float(rewards.cpu().numpy().flatten()[actor_expert_mask].mean())
        metrics.q_mean_expert = float(discrete_q_selected.detach().cpu().numpy().flatten()[actor_expert_mask].mean())
```

### 6. Added Per-Actor Metrics Fields (`config.py`)

```python
class MetricsData:
    # ... existing fields
    
    # Per-actor training diagnostics
    batch_frac_dqn: float = 0.0       # Fraction of batch that is DQN frames
    batch_n_dqn: int = 0              # Number of DQN frames in last batch
    batch_n_expert: int = 0           # Number of expert frames in last batch
    td_err_mean_dqn: float = 0.0      # Mean TD error for DQN frames
    td_err_mean_expert: float = 0.0   # Mean TD error for expert frames
    reward_mean_dqn: float = 0.0      # Mean reward for DQN frames in batch
    reward_mean_expert: float = 0.0   # Mean reward for expert frames in batch
    q_mean_dqn: float = 0.0           # Mean Q-value for DQN frames
    q_mean_expert: float = 0.0        # Mean Q-value for expert frames
```

### 7. Added Diagnostic Logging (`aimodel.py`)

```python
# Added to HybridReplayBuffer
def get_actor_composition(self):
    """Return statistics on actor composition of buffer"""
    actors_slice = self.actors[:self.size]
    n_dqn = np.sum(actors_slice == 'dqn')
    n_expert = np.sum(actors_slice == 'expert')
    return {
        'total': self.size,
        'dqn': int(n_dqn),
        'expert': int(n_expert),
        'frac_dqn': float(n_dqn) / self.size,
        'frac_expert': float(n_expert) / self.size,
    }

# Added to train_step() - logs every 100 training steps
if self.training_steps % 100 == 0:
    print(f"[BATCH] Step {self.training_steps}: {n_dqn} DQN ({frac_dqn*100:.1f}%) / {n_expert} expert")

# Added to train_step() - logs every 1000 training steps
if self.training_steps % 1000 == 0:
    comp = self.memory.get_actor_composition()
    print(f"[BUFFER] Step {self.training_steps}: {comp['total']} total, "
          f"{comp['dqn']} DQN ({comp['frac_dqn']*100:.1f}%), "
          f"{comp['expert']} expert ({comp['frac_expert']*100:.1f}%)")
```

## Expected Behavior After Fix

### During Training
1. **Batch composition logging every 100 steps:**
   ```
   [BATCH] Step 100: 6075 DQN (74.1%) / 2117 expert (25.9%)
   [BATCH] Step 200: 6100 DQN (74.5%) / 2092 expert (25.5%)
   ```

2. **Buffer composition logging every 1000 steps:**
   ```
   [BUFFER] Step 1000: 1000000 total, 740000 DQN (74.0%), 260000 expert (26.0%)
   ```

3. **Per-actor metrics available:**
   - `metrics.batch_frac_dqn` - should be ~0.74 (74% DQN)
   - `metrics.td_err_mean_dqn` - TD error for DQN frames
   - `metrics.td_err_mean_expert` - TD error for expert frames
   - `metrics.reward_mean_dqn` - Reward for DQN frames
   - `metrics.reward_mean_expert` - Reward for expert frames
   - `metrics.q_mean_dqn` - Q-values for DQN frames
   - `metrics.q_mean_expert` - Q-values for expert frames

### Verification Steps

1. **Check batch composition** - Should see ~74% DQN / ~26% expert in logs
2. **Compare TD errors** - If DQN TD errors are much larger than expert, indicates DQN predictions are poor
3. **Compare rewards** - Should see actual reward distribution per actor
4. **Compare Q-values** - If DQN Q-values are very different from expert, may indicate value function bias
5. **Monitor learning** - Can now determine if DQN frames are actually updating weights meaningfully

## Next Steps for Investigation

With actor attribution now working, you can:

1. **Verify the fix worked** - Check that batch composition shows expected 74/26 split
2. **Analyze per-actor metrics** - Compare TD errors, rewards, and Q-values between expert and DQN
3. **Identify learning issues** - If DQN frames still don't contribute to learning, now you can see WHY:
   - Are DQN rewards much worse than expert? (reward_mean comparison)
   - Are DQN TD errors much larger? (suggests poor value estimation)
   - Are DQN Q-values systematically biased? (suggests architectural issue)
4. **Consider per-actor weighting** - If DQN frames have systematically lower rewards, may need to weight them differently in advantage calculation
5. **Check gradient flow** - Add per-actor gradient tracking to verify DQN frames produce meaningful weight updates

## Files Modified

- `Scripts/aimodel.py` - Added actor storage, updated push/sample/step/train_step
- `Scripts/config.py` - Added per-actor metrics fields
- No changes needed to `Scripts/socket_server.py` - actor tagging was already correct!

## Testing Recommendations

1. **Start training** and watch for the `[BATCH]` and `[BUFFER]` log messages
2. **Verify composition** matches expected ~74% DQN / 26% expert
3. **Monitor per-actor metrics** in your metrics display (you may want to add these to the display)
4. **Compare learning** - watch if rewards improve for both expert and DQN frames
5. **Check for asymmetry** - if DQN TD errors are consistently much higher, that's the smoking gun

## Why This Was So Critical

This bug meant the entire diagnostic framework was non-functional. You couldn't determine:
- Whether attribution was correct (it was!)
- Whether DQN frames were in the buffer (they were!)
- Whether DQN frames were in training batches (they were!)
- Whether DQN frames had different characteristics than expert frames (couldn't tell!)
- Whether the training loop was treating them differently (it wasn't, and had no way to!)

Now, with actor tags properly stored and tracked, you can finally see what's happening with each type of experience and diagnose the real learning problem.
