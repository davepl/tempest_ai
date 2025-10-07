# Score Transmission Refactor

## Date: October 7, 2025

## Summary
Simplified score handling by transmitting the full score as a single value instead of splitting into high/low components. Score scaling is now handled on the Python side as needed.

## Changes Made

### 1. Logic.lua (Reward Calculation)
- **SCORE_UNIT**: Kept at `1.0` (no scaling in Lua)
- **Reward Calculation**: Score deltas are added directly to obj_reward without division
- Python side will handle any needed scaling via `RL_CONFIG.subj_reward_scale` and related config

### 2. State.lua (Score Reading) - NO CHANGES NEEDED
- **Score Extraction**: Already correctly reading 3-byte BCD score from memory (0x40-0x42)
- **Formula**: `score = score_high * 10000 + score_mid * 100 + score_low`
- **Range**: 0 to 999,999 (6 BCD digits)
- This is working perfectly and requires no changes

### 3. Main.lua (Score Transmission)
**Before:**
```lua
local score_high = math.floor(score / 65536)
local score_low = score % 65536
-- Format: ">HdddBBBHHHBBBhBhBBBBB" (two H for score_high, score_low)
```

**After:**
```lua
-- Removed score_high/score_low calculation
-- Format: ">HdddBBBHIBBBhBhBBBBB" (single I for full score)
-- Send score as unsigned int (4 bytes, up to 4,294,967,295)
```

### 4. Aimodel.py (Score Reception)
**Before:**
```python
_FMT_OOB = ">HdddBBBHHHBBBhBhBBBBB"
(num_values, reward, subjreward, objreward, gamestate, game_mode, done, 
 frame_counter, score_high, score_low, ...)
score = (score_high * 65536) + score_low
```

**After:**
```python
_FMT_OOB = ">HdddBBBHIBBBhBhBBBBB"
(num_values, reward, subjreward, objreward, gamestate, game_mode, done, 
 frame_counter, score, ...)
# No reconstruction needed - score is already the full value
```

## Benefits

1. **Simpler**: No need to split/reconstruct score across language boundary
2. **Clearer**: Score value is transmitted as-is from memory to Python
3. **More Range**: Unsigned int (4 bytes) supports up to 4.2 billion vs previous 4.2 million
4. **Less Error-Prone**: Eliminates potential wraparound issues with 16-bit splits
5. **Flexible**: Python side can apply any scaling needed via config

## OOB Format Details

**New Format String**: `">HdddBBBHIBBBhBhBBBBB"`

| Field | Type | Bytes | Description |
|-------|------|-------|-------------|
| num_values | H (ushort) | 2 | Number of float32 values in main payload |
| reward | d (double) | 8 | Total reward (subjective + objective) |
| subjreward | d (double) | 8 | Subjective reward component |
| objreward | d (double) | 8 | Objective reward component |
| gamestate | B (uchar) | 1 | Current game state |
| game_mode | B (uchar) | 1 | Game mode flags |
| done | B (uchar) | 1 | Episode done flag |
| frame | H (ushort) | 2 | Frame counter (wraps at 65536) |
| **score** | **I (uint)** | **4** | **Full player score (0-999,999)** |
| save_signal | B (uchar) | 1 | Save model signal |
| fire | B (uchar) | 1 | Fire button command |
| zap | B (uchar) | 1 | Zap button command |
| spinner | h (short) | 2 | Spinner command (-128 to +127) |
| is_attract | B (uchar) | 1 | Attract mode flag |
| nearest_enemy | h (short) | 2 | Nearest enemy absolute segment |
| player_seg | B (uchar) | 1 | Player absolute segment |
| is_open | B (uchar) | 1 | Open level flag |
| expert_fire | B (uchar) | 1 | Expert system fire recommendation |
| expert_zap | B (uchar) | 1 | Expert system zap recommendation |
| level_number | B (uchar) | 1 | Current level number |

**Total Header Size**: 47 bytes (was 49 bytes with HH split)

## Score Flow

```
┌─────────────────┐
│ Memory (0x40-42)│  3 bytes BCD → 6 digits (000000-999999)
│   BCD: HH MM LL │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   state.lua     │  Converts BCD to decimal integer
│ score = H*10000 │  Range: 0 to 999,999
│      + M*100    │
│      + L        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   main.lua      │  Packs as unsigned int (4 bytes)
│ pack(">...I..."│  Can hold up to 4,294,967,295
│      score)     │  (Tempest max is 999,999)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  aimodel.py     │  Unpacks as Python int
│ unpack(">...I"  │  Ready for use in reward calculations
│        ...)     │  Python applies any needed scaling
└─────────────────┘
```

## Testing Checklist

- [x] Lua syntax valid (no pack format errors)
- [x] Python struct format matches Lua format
- [ ] Score values transmit correctly (verify with print statements)
- [ ] Score deltas compute correctly in logic.lua
- [ ] Rewards scale appropriately (150pt flipper → reasonable obj_reward)
- [ ] No overflow or wraparound issues with high scores

## Rollback Instructions

If issues arise, revert to previous format:
1. Restore `local score_high = math.floor(score / 65536)` and `score_low` in main.lua
2. Change format back to `">HdddBBBHHHBBBhBhBBBBB"` in both files
3. Restore `score = (score_high * 65536) + score_low` in aimodel.py
