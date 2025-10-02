# N-Step & Diversity Bonus Quick Reference

## 🎮 Runtime Hotkeys

| Key | Function | Effect |
|-----|----------|--------|
| **'n'** | Toggle n-step | Switch between 1-step (TD) and 5-step returns |
| **'d'** | Toggle diversity bonus | Enable/disable exploration bonuses |

## ⚙️ Configuration

### In `Scripts/config.py`:

```python
# N-step returns (line 106)
n_step: int = 5                      # Lookahead steps for credit assignment

# Runtime toggles (line 153-155)
n_step_enabled: bool = True          # Enable n-step at startup
diversity_bonus_enabled: bool = True # Enable diversity bonus at startup
diversity_bonus_weight: float = 0.5  # Bonus magnitude (0.3-1.0 recommended)
```

## 📊 What Each Feature Does

### N-Step Returns (Credit Assignment)
- **Off (1-step)**: Q(s,a) learns from immediate reward only
- **On (5-step)**: Q(s,a) learns from 5-frame cumulative reward
- **When to disable**: If agent becomes unstable or too exploratory
- **When to enable**: To improve credit for multi-step strategies

### Diversity Bonus (Counterfactual Exploration)
- **Off**: Pure game rewards, no exploration incentive
- **On**: Bonus for trying novel actions in familiar states
- **When to disable**: If agent explores too much, hurts performance
- **When to enable**: To discover strategies beyond expert baseline

## 🔬 Monitoring

### Console Messages
```
N-step learning enabled/disabled     # Hotkey 'n' pressed
Action diversity bonus enabled/disabled  # Hotkey 'd' pressed
```

### What to Watch
- **Performance dip** when toggling diversity on = exploration phase
- **Performance recovery** after exploration = learning from discoveries
- **Stable with n-step off** but **plateau** = needs better credit assignment

## 🎯 Recommended Experiments

### Experiment 1: Pure N-Step
```
1. Disable diversity ('d' key)
2. Enable n-step ('n' key if off)
3. Watch for improved learning on delayed-reward scenarios
```

### Experiment 2: Pure Diversity
```
1. Disable n-step ('n' key)
2. Enable diversity ('d' key if off)
3. Watch for trying different strategies, potential breakthroughs
```

### Experiment 3: Combined (Default)
```
1. Both enabled (default)
2. Should see exploration + good credit assignment
3. Best chance to surpass expert plateau
```

### Experiment 4: Baseline
```
1. Both disabled
2. Pure 1-step TD learning with epsilon exploration
3. Compare to see what each feature contributes
```

## 📈 Expected Behavior

### With N-Step Enabled
✅ Better credit for actions that set up future rewards
✅ More stable learning on sparse reward tasks
✅ Longer-term strategy development
⚠️ Slightly higher variance in early training

### With Diversity Bonus Enabled
✅ Tries actions expert never tried
✅ Discovers counterfactual outcomes
✅ Can escape local optima
⚠️ Initial performance dip during exploration
⚠️ Higher memory usage (tracks action history)

### With Both Enabled (Recommended)
✅ Best of both: exploration + credit assignment
✅ Highest chance to surpass expert performance
✅ Addresses root causes of plateau
⚠️ Most complex to tune (two hyperparameters)

## 🐛 Troubleshooting

### Hotkey 'n' doesn't work?
- Check: `agent.n_step_enabled` should toggle True/False
- Check: Console should show "N-step learning enabled/disabled"
- Check: `_server_nstep_enabled()` should return False when disabled

### Hotkey 'd' doesn't work?
- Check: `agent.diversity_bonus_enabled` should toggle True/False
- Check: Console should show "Action diversity bonus enabled/disabled"
- Check: Diversity bonus should be 0.0 when disabled

### Agent explores too much?
- Lower `diversity_bonus_weight` from 0.5 to 0.3 in config
- Or temporarily disable with 'd' key

### Agent too conservative?
- Raise `diversity_bonus_weight` from 0.5 to 1.0 in config
- Or verify diversity bonus is enabled with 'd' key

### Performance unstable with n-step?
- Try lowering `n_step` from 5 to 3 in config
- Or temporarily test with 'n' key to disable

## 🔍 Technical Details

### How N-Step Works
```python
# 1-step (off): R = r_t
# 5-step (on):  R = r_t + γ*r_{t+1} + γ²*r_{t+2} + γ³*r_{t+3} + γ⁴*r_{t+4}
```

Stored in replay buffer as single experience with cumulative reward.

### How Diversity Bonus Works
```python
state_cluster = round(state[:20], 1)  # Group similar states
action_key = (discrete_action, round(continuous_action, 1))

if action_key not in tried_actions[state_cluster]:
    bonus = weight / sqrt(num_actions_tried)
    total_reward = game_reward + bonus
```

Bonus decays as more actions tried (encourages breadth).

## 📝 Test Suite

Run tests to verify everything works:
```bash
python test_nstep_diversity.py
```

Should see:
```
✓ Diversity Bonus Novel Actions
✓ Diversity Bonus State Clustering
✓ Diversity Bonus Toggle
✓ N-Step Toggle
Total: 4/4 tests passed
```

## 🚀 Quick Start

**Default settings (recommended)**:
- Both features enabled
- n_step = 5
- diversity_bonus_weight = 0.5

**To change at runtime**: Press 'n' or 'd' keys

**To change permanently**: Edit `Scripts/config.py`

**To verify working**: Run `python test_nstep_diversity.py`

## 📚 More Info

See detailed documentation:
- `NSTEP_FIX_SUMMARY.md` - Complete fix details
- `NSTEP_DIVERSITY_README.md` - Original implementation docs
- `NSTEP_AUDIT_REPORT.md` - Pre-fix audit findings
