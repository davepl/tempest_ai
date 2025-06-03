# Tanker Reward System Implementation

## Overview
Implemented a reward system that encourages the AI to flee from dangerous tankers carrying fuseball payloads while hunting/staying near safe tankers without payloads.

## Implementation Details

### 1. Payload Detection
- **Method**: Uses the `enemy_split_behavior` field from the enemy state byte
- **Logic**: `split_behavior > 0` indicates a tanker carrying a fuseball payload (dangerous)
- **Logic**: `split_behavior == 0` indicates a safe tanker (good hunting target)

### 2. Reward System Location
- **File**: `/home/dave/source/repos/tempest_ai/Scripts/logic.lua`
- **Function**: `M.calculate_reward()`
- **Position**: Added after the enemy shot reward logic (around line 554-612)

### 3. Constants Added
```lua
local TANKER_DANGER_THRESHOLD = 0x80 -- Tanker danger threshold for proximity rewards
```

### 4. Reward Logic

#### For Dangerous Tankers (with payload):
- **Fleeing Reward**: +50 to +100 points for moving away from dangerous tankers
- **Approaching Penalty**: -30 to -60 points for moving toward dangerous tankers  
- **Staying Still Penalty**: -40 to -80 points for staying near dangerous tankers

#### For Safe Tankers (no payload):
- **Hunting Reward**: +30 to +60 points for moving toward safe tankers
- **Staying Near Reward**: +20 to +40 points for staying near safe tankers
- **Fleeing Penalty**: -15 to -30 points for moving away from safe tankers

### 5. Distance and Proximity Logic
- **Range**: Only applies to tankers within 3 segments of the player
- **Proximity Multiplier**: `max(1, (4 - distance) / 2)` - closer tankers have higher impact
- **Depth Threshold**: Only considers tankers with depth > `TANKER_DANGER_THRESHOLD` (0x80)

### 6. Movement Detection
- **Moving Away**: Detected when relative distance and spinner direction have opposite signs
- **Moving Toward**: Detected when relative distance and spinner direction have same signs
- **Staying Still**: When `detected_spinner == 0`

## Key Features

1. **Payload Detection**: Automatically detects whether a tanker is carrying a fuseball payload
2. **Dynamic Behavior**: Encourages opposite behaviors for dangerous vs safe tankers
3. **Proximity-Based**: Rewards/penalties scale with distance to tanker
4. **Movement-Aware**: Different rewards based on whether player is moving toward, away, or staying still
5. **Performance Optimized**: Only processes closest tanker to avoid over-rewarding

## Testing Notes

The system should encourage the AI to:
- Flee from tankers showing `split_behavior > 0` (carrying fuseball payloads)
- Hunt/stay near tankers showing `split_behavior == 0` (safe targets)
- Show more pronounced behavior when tankers are closer
- Balance tanker behavior with other existing reward systems

## Integration

This system integrates seamlessly with the existing reward calculation and follows the same patterns as:
- Fuseball charging lane rewards
- Enemy shot danger rewards  
- Pulsar lane penalty system

The implementation preserves all existing functionality while adding the new tanker-specific reward logic.
