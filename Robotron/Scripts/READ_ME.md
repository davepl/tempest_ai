# Robotron AI Baseline

This `Robotron/Scripts` project is now a stripped baseline for Robotron-specific bring-up.

## Current Scope

- Lua sends a **2-value state vector** each frame:
  - `PlayerAlive` (dummy placeholder)
  - `Score` (dummy placeholder)
- Python returns **dual 8-way joystick actions**:
  - movement direction index `0..7`
  - firing direction index `0..7`
- Replay/training pipeline remains active with this reduced state/action space.

## Protocol (Lua -> Python)

- Header format: `>HddBIBB`
  - `H`: number of float state values (currently `2`)
  - `d`: subjective reward
  - `d`: objective reward
  - `B`: done flag
  - `I`: score
  - `B`: player alive flag
  - `B`: save signal
- Followed by `N` big-endian float32 state values.

## Protocol (Python -> Lua)

- Action format: `bb`
  - movement direction index `0..7`
  - firing direction index `0..7`

## TODO (Known Missing Robotron Wiring)

- Exact MAME input field names for:
  - Start button
  - Coin insert
- Real Robotron memory extraction for:
  - `PlayerAlive`
  - `Score`

Until those are wired, the script uses placeholders so the training pipeline can run end-to-end.
