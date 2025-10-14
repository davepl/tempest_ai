# Verbose Mode Feature

## Overview
Added keyboard shortcut `v` to toggle verbose debug output mode on/off. This controls whether detailed diagnostic output is displayed during training.

## Usage
- Press `v` during training to toggle verbose mode
- Default: OFF (clean output)
- When ON: Shows detailed agreement debugging and other diagnostic output

## What's Controlled by Verbose Mode

### Agreement Debug Output (AGREE DEBUG)
When verbose mode is ON, prints every 100 training steps:
- Agreement percentage between current policy and replay buffer actions
- Number of DQN frames in batch
- First 10 greedy actions vs first 10 replay actions
- Distribution of greedy actions [action 0, 1, 2, 3 counts]
- Distribution of replay buffer actions [action 0, 1, 2, 3 counts]

Example output:
```
[AGREE DEBUG] Step 5400:
  n_dqn=1014, agree_pct=76.8%
  First 10 greedy: [2 2 2 2 1 2 2 2 2 2]
  First 10 replay: [2 2 2 2 2 2 2 2 2 2]
  Greedy dist: [  0  66 948   0]
  Replay dist: [182   1 829   2]
```

### Future Extensibility
Additional debug output can be added and controlled by checking `metrics.verbose_mode`:

```python
if metrics.verbose_mode:
    print("[DEBUG] Your detailed diagnostic info here")
```

## Implementation Details

### Files Modified
1. **Scripts/config.py**
   - Added `verbose_mode: bool = False` to `MetricsData` class
   - Added `toggle_verbose_mode()` method to `MetricsData` class

2. **Scripts/main.py**
   - Added keyboard handler for 'v' key
   - Updated documentation comment to include 'v' in keyboard controls list

3. **Scripts/aimodel.py**
   - Modified agreement debug output to check `metrics.verbose_mode` before printing
   - Changed: `if self.training_steps % 100 == 0:` → `if metrics.verbose_mode and self.training_steps % 100 == 0:`

### Thread Safety
The `verbose_mode` flag is stored in the `MetricsData` class and protected by `metrics.lock` during toggles, ensuring thread-safe access from both the keyboard handler thread and training threads.

## Benefits
- **Cleaner Default Output**: No verbose debug spam during normal training
- **On-Demand Diagnostics**: Enable detailed output when investigating specific issues
- **Performance**: Eliminates overhead of string formatting and I/O when not needed
- **User-Friendly**: Simple toggle with immediate feedback

## Keyboard Controls Summary
After this change:
- `v` - Toggle verbose debug output (default: OFF)
- `s` - Save model
- `q` - Quit
- `o` - Toggle expert override
- `e` - Toggle expert mode
- `p` - Toggle epsilon override
- `t` - Toggle training mode
- `c` - Clear screen and redraw header
- `h` - Force hard target network update
- `4/6` - Decrease/increase epsilon
- `5` - Restore natural epsilon decay
- `7/9` - Decrease/increase expert ratio
- `8` - Restore natural expert ratio decay
- `l/L` - Adjust learning rate
- `Space` - Print single metrics row
