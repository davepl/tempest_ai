# Negative Objective Reward Bug Fix

## Date: October 7, 2025

## Problem Discovery

User reported: "I'm still not sure why total objective rewards are negative most times - other than superzap, which should be edge-triggered and thus once per level, I don't see where we're deducting from the objective reward!"

## Root Cause Analysis

### The Bug
In `logic.lua` line 593, the score delta filter was:
```lua
if score_delta ~= 0 and score_delta < 1000 then
```

**This condition has a critical flaw**: it allows **negative score deltas** to pass through!

### Why This Happens

Mathematical logic error:
- **Intended**: Filter out large completion bonuses (>1000 points)
- **Actual**: The condition `score_delta < 1000` is TRUE for ALL negative numbers!
  - Example: `-1000 < 1000` → **TRUE** ✓
  - Example: `-500 < 1000` → **TRUE** ✓
  - Example: `-50000 < 1000` → **TRUE** ✓

### Example Scenario

```lua
Frame N:   score = 50,000
Frame N+1: score = 49,000  (decrease during game transition/reset)

score_delta = 49,000 - 50,000 = -1,000

Check: score_delta ~= 0 and score_delta < 1000
       -1,000 ~= 0 → TRUE
       -1,000 < 1000 → TRUE
       
Result: obj_reward += -1,000  ← MASSIVE NEGATIVE REWARD!
```

## Why Score Might Decrease

Several legitimate scenarios cause score to decrease frame-to-frame:

1. **New Game Start**: Score resets to 0 while `previous_score` still holds old value
2. **Game State Transitions**: 
   - Attract mode → gameplay
   - High score entry → new game
   - Death → respawn
3. **BCD Reading Timing**: Score read mid-update during memory writes
4. **Level Completion**: Brief moments where score display changes
5. **Demo Mode**: Score resets frequently

## The Fix

Change the condition to **only accept POSITIVE score deltas**:

```lua
if score_delta > 0 and score_delta < 1000 then  -- Filter out large bonuses AND negative deltas
```

This ensures:
- ✅ Positive score increases (1-999 pts) → added to obj_reward
- ✅ Large bonuses (≥1000 pts) → ignored (level completion)
- ✅ **Negative deltas → ignored** (score decreases don't penalize)
- ✅ Zero deltas → ignored (no score change)

## Impact

### Before Fix
- Score decreases caused large negative objective rewards
- Training received random punishment during game transitions
- Objective reward could be -1000 or worse per frame
- Agent learned to avoid situations where score might "appear" to decrease

### After Fix
- Score decreases are simply ignored (no reward/penalty)
- Objective reward only reflects actual point gains
- Training signal is clean and consistent
- Agent focuses on maximizing score increases, not avoiding transitions

## Additional Considerations

### Should We Reset previous_score?

Currently `previous_score` persists across game boundaries. Consider resetting it when:
- `gamestate` changes (new game detected)
- Player death (though death already bypasses this code path)
- Attract mode transitions

**Current Approach**: Ignoring negative deltas is safer than trying to detect all reset conditions, as it handles:
- All game state transitions automatically
- BCD read timing issues
- Unknown edge cases

### Alternative Fixes Considered

1. ❌ **Reset previous_score on state changes**: Complex, might miss edge cases
2. ❌ **Clamp to zero**: `obj_reward += max(0, score_delta)` - works but less clear intent
3. ✅ **Filter positives only**: `score_delta > 0` - clear, simple, correct

## Testing Validation

To verify the fix works:
1. Monitor obj_reward values - should no longer see large negatives
2. Check during game transitions - score decreases should be ignored
3. Verify normal gameplay - small score gains still rewarded
4. Test level completion - large bonuses (≥1000) still filtered

## Code Changes

**File**: `Scripts/logic.lua`  
**Line**: 593

**Before**:
```lua
if score_delta ~= 0 and score_delta < 1000 then
```

**After**:
```lua
if score_delta > 0 and score_delta < 1000 then
```

**Comment Updated**:
```lua
-- Filter out large bonuses AND negative deltas
```

## Conclusion

This was a subtle but significant bug where the filter intended to remove large positive bonuses inadvertently **allowed all negative values** through due to basic mathematical comparison logic. The fix ensures objective rewards only reflect genuine score increases within the expected range.
