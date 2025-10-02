# N-Step Returns & Diversity Bonus Implementation

## Overview
This implementation addresses the fundamental RL learning problem where the agent "gets as good as the expert but no better" - unable to discover improvements through experimentation.

## Root Causes Identified
1. **Poor Credit Assignment**: Agent only sees immediate rewards, not multi-step outcomes
2. **No Counterfactual Exploration**: Agent doesn't explore "what if I tried different actions?"
3. **Policy Mismatch**: Expert actions stored with DQN Q-values creates supervised learning contamination

## Solution: Two Complementary Features

### 1. N-Step Returns (Credit Assignment Fix)
**Problem**: Agent doesn't understand multi-step consequences of actions
**Solution**: Look ahead 5 frames to calculate cumulative discounted rewards

**How it works**:
- Buffers trajectories of (state, action, reward, next_state, done) tuples
- Calculates n-step return: R_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ... + γⁿ*r_{t+n}
- Stores final state reached after n steps for bootstrapping
- Handles episode termination correctly (stops accumulation at done=True)

**Benefits**:
- Agent sees "what happened 5 frames after my action" instead of just immediate reward
- Better credit for actions that set up future success
- Helps discover strategies that have delayed payoffs

### 2. Action Diversity Bonus (Counterfactual Exploration)
**Problem**: Agent only tries actions expert would take, never explores alternatives
**Solution**: Reward trying novel actions in similar states

**How it works**:
- Clusters states by rounding first 20 dimensions to 0.1
- Tracks which (discrete_action, continuous_action) pairs tried per state cluster
- Awards bonus for first time trying each action: `bonus = weight / sqrt(num_actions_tried)`
- Bonus decays as more actions tried (1/√n encourages broad exploration)

**Benefits**:
- Actively encourages trying different strategies in familiar situations
- Counterfactual reasoning: "What if I fire instead of just move?"
- Prevents getting stuck in local optima of expert-like behavior

## Files Modified

### 1. `Scripts/aimodel.py`
**Changes**:
- Added `add_trajectory()` method to `PrioritizedReplayMemory` class (lines ~630)
  - Implements n-step return calculation with gamma discounting
  - Handles trajectories of arbitrary length
  - Correctly stops at episode termination
  
- Added diversity tracking to `HybridDQNAgent.__init__` (lines ~1082-1088):
  - `action_history`: dict tracking tried actions per state cluster
  - `diversity_bonus_enabled`: runtime toggle (default True)
  - `diversity_bonus_weight`: base bonus magnitude (default 0.5)
  - `nstep_enabled`: runtime toggle (default True)
  - `nstep_length`: lookahead steps (default 5)
  
- Added `calculate_diversity_bonus()` method (lines ~1130-1168):
  - State clustering via rounding to 0.1
  - Action tracking with discrete + continuous dimensions
  - 1/√n decay for bonus magnitude
  - Silent error handling to prevent training disruption
  
- Added `set_diversity_bonus_enabled(bool)` method (lines ~1170-1174)
- Added `set_nstep_enabled(bool)` method (lines ~1176-1180)

### 2. `Scripts/socket_server.py`
**Changes**:
- Added diversity bonus calculation in experience processing loop (lines ~225-232)
  - Calls `agent.calculate_diversity_bonus()` before storing experience
  - Adds bonus to reward before passing to n-step buffer or agent
  - Graceful fallback if method not available

### 3. `Scripts/main.py`
**Changes**:
- Added hotkey handler for 'n' key (lines ~126-133):
  - Toggles n-step learning on/off
  - Displays current status
  - Refreshes metrics display
  
- Added hotkey handler for 'd' key (lines ~134-141):
  - Toggles diversity bonus on/off
  - Displays current status
  - Refreshes metrics display

### 4. `Scripts/config.py`
**Changes**:
- Added configuration options (lines ~149-154):
  ```python
  nstep_enabled: bool = True            # Enable n-step returns
  nstep_length: int = 5                 # Look ahead 5 frames
  diversity_bonus_enabled: bool = True  # Reward novel actions
  diversity_bonus_weight: float = 0.5   # Base bonus (decays with √n)
  ```

### 5. `test_nstep_diversity.py` (NEW)
**Created comprehensive test suite**:
- `test_nstep_trajectory_calculation()`: Verifies n-step return math
- `test_nstep_terminal_state()`: Tests terminal state handling
- `test_diversity_bonus_novel_action()`: Tests bonus rewards novel actions
- `test_diversity_bonus_state_clustering()`: Verifies state clustering works
- `test_diversity_bonus_toggle()`: Tests runtime enable/disable
- `test_nstep_toggle()`: Tests runtime enable/disable

**Test Results**: ✅ All 6/6 tests passing

## Usage

### Runtime Hotkeys
- **'n'**: Toggle n-step learning on/off
- **'d'**: Toggle diversity bonus on/off

### Configuration
Edit `Scripts/config.py` to adjust parameters:
```python
# N-step settings
nstep_enabled = True          # Master switch
nstep_length = 5             # Increase for longer lookahead (1-10 recommended)

# Diversity bonus settings
diversity_bonus_enabled = True
diversity_bonus_weight = 0.5  # Increase for stronger exploration incentive
```

### Testing
Run the test suite:
```bash
python test_nstep_diversity.py
```

## Expected Behavior

### With N-Step Returns Enabled
- Agent should learn strategies with delayed rewards (e.g., positioning for future kills)
- Better credit assignment for setup actions vs finishing actions
- More stable learning when rewards are sparse

### With Diversity Bonus Enabled
- Agent should try more varied strategies in familiar states
- Exploration of counterfactual actions ("what if I tried X instead?")
- Potential to discover strategies superior to expert baseline

### Combined Effect
- Agent can learn beyond expert performance by:
  1. Understanding multi-step consequences (n-step returns)
  2. Actively exploring alternatives to expert actions (diversity bonus)
- Expect initial performance dip as agent explores, then potential breakthrough past expert plateau

## Implementation Details

### N-Step Return Calculation
```python
n_step_return = 0.0
discount = 1.0
for j in range(min(n_step, len(trajectory) - i)):
    _, _, _, reward, next_state, done = trajectory[i + j]
    n_step_return += discount * reward
    discount *= gamma
    if done:
        break
```

### Diversity Bonus Calculation
```python
# Cluster state by rounding first 20 dims
state_key = tuple(np.round(state[:20], 1))

# Track action (discrete, continuous)
action_taken = (int(discrete_action), round(float(continuous_action), 1))

# Award bonus if novel
if action_taken not in action_history[state_key]:
    action_history[state_key].add(action_taken)
    num_tried = len(action_history[state_key])
    bonus = weight / np.sqrt(num_tried)
```

## Debugging Tips

### If agent performance drops:
1. Check if diversity bonus weight is too high (try 0.3 instead of 0.5)
2. Temporarily disable diversity bonus ('d' key) to verify it's the cause
3. Reduce nstep_length to 3 for faster feedback

### If agent still plateaus:
1. Verify diversity bonus is calculating (check console for bonus values)
2. Increase diversity_bonus_weight to 1.0 for stronger exploration
3. Check that expert_ratio is decaying properly (agent needs exploration time)

### Monitoring:
- Watch for console messages: "Action diversity bonus enabled/disabled"
- Watch for console messages: "N-step learning enabled/disabled"
- Check that action_history dict is growing (sign of exploration)

## Theory Behind the Solution

### Credit Assignment Problem
Traditional 1-step TD learning: Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
- Only sees immediate reward r
- Relies on bootstrapping from Q(s',a') for future value
- Problem: Q(s',a') is inaccurate early in training

N-step returns: Q(s,a) ← Q(s,a) + α[R_t^(n) + γⁿ·max Q(s_{t+n},a') - Q(s,a)]
- Sees actual n-step return R_t^(n) = Σ γʲ·r_{t+j}
- Less reliance on bootstrapped value estimates
- More accurate credit for multi-step strategies

### Counterfactual Exploration Problem
Expert-guided learning creates "observational bias":
- Agent only sees outcomes of expert actions
- Never learns "what would happen if I did X instead?"
- Stuck in expert's policy basin of attraction

Diversity bonus creates "interventional learning":
- Explicitly rewards trying non-expert actions
- Discovers counterfactual outcomes through direct experience
- Can find strategies expert never tried

### Why This Fixes the Plateau
1. **Before**: Agent imitates expert → gets expert-level performance → no reason to explore further
2. **After**: 
   - Diversity bonus forces exploration beyond expert actions
   - N-step returns provide accurate credit for novel strategies
   - Agent discovers some novel strategies outperform expert
   - Learning continues beyond expert plateau

## Next Steps

### Potential Enhancements
1. **Adaptive diversity weight**: Decrease bonus as agent performance improves
2. **State-aware clustering**: Use learned embeddings instead of rounding
3. **Intrinsic motivation**: Add prediction error as exploration bonus
4. **Curriculum learning**: Start with high expert_ratio, decay as diversity exploration succeeds

### Monitoring Success
Watch for these signs that it's working:
- Agent tries obviously sub-optimal actions initially (exploration)
- Performance temporarily drops below expert baseline
- After exploration phase, performance surpasses expert baseline
- Action diversity stabilizes at higher level than pre-implementation

### Debugging Flags
If issues arise, can disable features independently:
```python
# In config.py
nstep_enabled = False         # Disable n-step, keep diversity
diversity_bonus_enabled = False  # Disable diversity, keep n-step
```

## References
- N-step TD Learning: Sutton & Barto, "Reinforcement Learning: An Introduction", Chapter 7
- Exploration Bonuses: Pathak et al., "Curiosity-driven Exploration by Self-supervised Prediction" (2017)
- Credit Assignment: Mnih et al., "Asynchronous Methods for Deep Reinforcement Learning" (2016)
