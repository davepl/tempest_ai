# Reward Tracking Investigation

## Issue
The user reported that `DQN1M` and `DQN5M` statistics are negative (~-9000) despite the Objective reward (`Obj`) being positive (~1150).

## Root Cause Analysis
The discrepancy is caused by the interaction between the **Reward Scaling** and the **Death Penalty**.

### 1. Reward Scaling
- **Objective Reward Scale**: `0.0001` (1 point = 0.0001 reward units).
- **Display Logic**: The metrics display multiplies the internal reward units by `10,000` (inverse of scale) to show "points".
- **Example**: An episode with 1150 points results in `0.115` internal reward units. The display correctly shows `1150`.

### 2. Death Penalty
- **Configuration**: `death_penalty` is set to `-1.0` in `Scripts/config.py`.
- **Application**: This penalty is added directly to the *scaled* reward when `frame.done` is True (in `Scripts/socket_server.py`).
- **Magnitude**: A penalty of `-1.0` internal units is equivalent to **-10,000 points**.

### 3. The Calculation
For a typical episode:
- **Objective Score**: 1150 points -> `0.115` reward units.
- **Death Penalty**: -1.0 reward units.
- **Total Episode Reward**: `0.115 - 1.0 = -0.885` reward units.
- **Displayed DQN Reward**: `-0.885 * 10,000 = -8850`.

This explains why the DQN stats are negative (~-9000) while the Objective stats are positive (~1150). The agent is effectively being penalized 10,000 points for dying, which outweighs the points earned in early gameplay.

## Secondary Findings
- **Superzap Penalty**: `superzap_block_penalty` (-0.05) appears to be defined in `config.py` but is **unused** in the current codebase (`socket_server.py`, `aimodel.py`, `training.py`). It does not contribute to the negative score.

## Recommendations
1. **Adjust Death Penalty**: If a death penalty of -10,000 points is too harsh, reduce `death_penalty` in `Scripts/config.py`.
   - Current: `-1.0` (-10,000 points)
   - Suggested: `-0.1` (-1,000 points) or `-0.05` (-500 points).
2. **Clarify Units**: Add comments in `config.py` clarifying that `death_penalty` is in *post-scaled* reward units.
