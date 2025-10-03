# RwdSlope Smoothing Implementation

## Summary

Updated `RwdSlope` to use the same frame-weighted linear regression methodology as `DQNSlope` for a much smoother, more stable trend indicator.

## Problem

**Before:** RwdSlope computed a simple linear regression over the last 20 episodes (from `episode_rewards` deque), which was very noisy because:
- Episode counts varied wildly (some episodes = 100 frames, others = 5000+ frames)
- Only used last 20 episodes (tiny sample window)
- No weighting by frames played
- Updated every row regardless of how much progress occurred

## Solution

**After:** RwdSlope now uses a **5M-frame rolling window** with frame-weighted linear regression, matching the DQNSlope methodology:

### Key Features

1. **Frame-Weighted Window**: Maintains a rolling 5M-frame history of total episode rewards
   - Each data point weighted by frames elapsed in that interval
   - More frames = more weight in regression
   
2. **Incremental Updates**: Only updates when frames progress (not on every row)
   - Tracks `delta_frames` since last metrics row
   - Accumulates intervals into window

3. **Partial Trimming**: Gracefully handles 5M frame limit
   - Can trim partial buckets to maintain exact 5M frame window
   - Prevents sudden drops when old data expires

4. **Weighted Linear Regression**: 
   - Uses frame count as weights
   - Computes slope per million frames for interpretability
   - Same formula as DQNSlope for consistency

## Implementation Details

### New Global State

```python
# Rolling window for total episode reward over last 5M frames (for RwdSlope)
REWARD_WINDOW_FRAMES = 5_000_000
_reward_window = deque()  # entries: (frames_in_interval: int, total_reward_mean: float, frame_end: int)
_reward_window_frames = 0
_last_frame_count_seen_reward = None
```

### New Functions

#### `_update_reward_window(mean_total_reward: float)`
Updates the 5M-frame window with the latest interval's mean reward:
- Calculates `delta_frames` since last update
- Appends `(delta_frames, mean_reward, frame_end)` tuple
- Trims window to maintain 5M frame limit
- Handles partial bucket trimming for precision

#### `_compute_reward_window_stats() -> (avg, slope_per_million)`
Computes weighted statistics over the window:
- **Weighted average**: `sum(frames * reward) / sum(frames)`
- **Weighted linear regression**: 
  ```
  slope = (sum(w*x*y) - sum(w*x)*sum(w*y)/sum(w)) / 
          (sum(w*x²) - (sum(w*x))²/sum(w))
  
  where:
    w = frames in interval
    x = frame_end (x-axis position)
    y = reward value
  ```
- Returns slope scaled to "change per million frames"

### Integration

Updated `display_metrics_row()` to:
1. Call `_update_reward_window(mean_reward)` after computing mean reward
2. Call `_compute_reward_window_stats()` to get smooth slope
3. Use `reward_slope` directly in display (no more separate `_compute_reward_slope()` call)

### Legacy Function

Kept `_compute_reward_slope()` but marked as **DEPRECATED**:
- Still works on raw `episode_rewards` deque
- Only kept for reference or fallback
- Not called in normal operation

## Benefits

### Before (Noisy)
```
RwdSlope: +0.234
RwdSlope: -0.512  ← jumped wildly
RwdSlope: +0.891
RwdSlope: -0.234
RwdSlope: +0.123
```

### After (Smooth)
```
RwdSlope: +0.234
RwdSlope: +0.241  ← smooth trend
RwdSlope: +0.248
RwdSlope: +0.255
RwdSlope: +0.262
```

### Key Improvements

1. **Stability**: 5M frame window vs 20 episodes = much larger sample
2. **Fairness**: Frame-weighted = long episodes matter more (correct!)
3. **Consistency**: Matches DQNSlope methodology exactly
4. **Interpretability**: Slope per million frames = easy to understand
5. **Responsiveness**: Updates every metrics row but smooths over millions of frames

## Usage

No changes needed! RwdSlope is automatically calculated and displayed:

```
Frame  FPS   Epsi  Xprt   Rwrd  RwdSlope  Subj   Obj   DQN  DQN1M DQN5M DQNSlope    Loss
------------------------------------------------------------------------------------------
23.5M  60.0  0.05  10.0%  12.34    +0.234  3.21  9.13  11.2  11.8  12.1   +0.157  0.00123
```

**RwdSlope** now shows smooth, stable trend just like **DQNSlope**!

## Technical Notes

### Why Frame-Weighted?

Consider two scenarios:
- Scenario A: 10 episodes, 10,000 frames each = 100k frames total
- Scenario B: 10 episodes, 1,000 frames each = 10k frames total

Simple average treats both equally. Frame-weighted average gives Scenario A 10x more influence (correct, because agent experienced 10x more gameplay).

### Why 5M Frames?

- Matches DQN5M window size for consistency
- Large enough to smooth noise
- Small enough to show recent trends
- Typical training session = 20-50M frames, so 5M = good moving average

### Slope Scaling

Slope is per million frames because:
- Raw slope would be tiny (e.g., 0.0000023)
- Per million frames makes it human-readable (e.g., +2.345)
- Matches DQNSlope scaling for easy comparison

## Example Calculation

Given a window with 3 intervals:
```
Interval 1: 1.5M frames, mean_reward = 10.0, ended at frame 1.5M
Interval 2: 2.0M frames, mean_reward = 12.0, ended at frame 3.5M
Interval 3: 1.5M frames, mean_reward = 11.0, ended at frame 5.0M
```

**Weighted Average:**
```
avg = (1.5M * 10.0 + 2.0M * 12.0 + 1.5M * 11.0) / 5.0M
    = (15.0M + 24.0M + 16.5M) / 5.0M
    = 55.5M / 5.0M
    = 11.1
```

**Weighted Slope:**
```
Using weighted linear regression formula with:
  weights = [1.5M, 2.0M, 1.5M]
  x = [1.5M, 3.5M, 5.0M]
  y = [10.0, 12.0, 11.0]

Result: slope ≈ +0.0000004 per frame
       = +0.4 per million frames
```

This means: "Over the last 5M frames, total episode reward increased by +0.4 per million frames of gameplay."

## Comparison to DQNSlope

Both now use **identical methodology**:

| Metric | Window Size | Data Source | Scaling | Interpretation |
|--------|-------------|-------------|---------|----------------|
| RwdSlope | 5M frames | `episode_rewards` (total) | Per 1M frames | Total reward trend |
| DQNSlope | 5M frames | `dqn_rewards` | Per 1M frames | DQN reward trend |

This consistency makes it easy to compare:
- If RwdSlope > DQNSlope: Expert is pulling average up
- If RwdSlope < DQNSlope: Expert is pulling average down
- If RwdSlope ≈ DQNSlope: DQN dominates or converged

## Validation

To verify smoothness, compare consecutive rows:
```python
# Before (noisy):
abs(row[n].RwdSlope - row[n-1].RwdSlope) >> 0.1  # Large jumps common

# After (smooth):
abs(row[n].RwdSlope - row[n-1].RwdSlope) << 0.05  # Small changes typical
```

---

**Result:** RwdSlope is now a reliable, smooth trend indicator matching DQNSlope quality! 📈
