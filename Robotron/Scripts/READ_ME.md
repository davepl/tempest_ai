# Robotron AI Baseline

This `Robotron/Scripts` project is now a stripped baseline for Robotron-specific bring-up.

## Current Scope

- Lua sends a **2210-value hybrid state vector** each frame:
  - 98 global features
    - alive / score / replay / lasers / wave
    - player position and velocity
    - `ZP1ENM` enemy-state bag (50 normalized bytes)
    - per-category counts / presence / nearest-distance summaries
    - quadrant danger / rescue summaries
    - wall proximity
  - `12 x 12 x 8` player-centered spatial grid
    - local danger, projectile, brute, human, obstacle, wall, density, and approach channels
  - `64 x 15` object tokens
    - salient objects from `OPTR`, `HPTR`, `RPTR`, `PPTR`
    - relative position, true velocity, distance, direction, threat, size, type flags
- Python returns **dual 8-way joystick actions**:
  - movement direction index `0..7`
  - firing direction index `0..7`
- Replay/training pipeline remains active with this expanded state vector.

## Protocol (Lua -> Python)

- Header format: `>HddBIBBIBB`
  - `H`: number of float state values (currently `2210`)
  - `d`: subjective reward
  - `d`: objective reward
  - `B`: done flag
  - `I`: score (decoded from `ZP1SCR`)
  - `B`: player alive flag
  - `B`: save signal
  - `I`: next replay level (decoded from `ZP1RP`)
  - `B`: number of lasers (`ZP1LAS`)
  - `B`: wave number (`ZP1WAV`)
- Followed by `N` big-endian float32 state values.

## Protocol (Python -> Lua)

- Action format: `bb`
  - movement direction index `0..7`
  - firing direction index `0..7`

## Startup Diagnostics

- Run foreground diagnostics:
  - `cd Robotron`
  - `./startmame.sh --fg`
- Background mode now reports explicit process liveness:
  - `./startmame.sh`
- Startup trace output is written to:
  - terminal
  - `Robotron/logs/startup_trace.log`

Lua debug controls in `Robotron/Scripts/main.lua`:

- `DEBUG_STARTUP_TRACE` (default `false`)
- `DEBUG_TRACE_FRAMES` (default `10`)
- `DEBUG_BYPASS_SOCKET_FOR_FRAMES` (default `0`)
  - Set to `10` to skip socket send/recv for first 10 frames (neutral action), for A/B isolation.

Interpretation guide:

- Failure before `socket_write_ok`:
  - memory extraction / frame serialization path issue.
- Failure after `socket_write_begin` but before `socket_read_ok`:
  - socket exchange/timeout path issue.
- Failure after `apply_action`:
  - likely game runtime/ROM/input interaction issue.

## Remote Preview (WebRTC TURN/STUN)

The dashboard preview card can use WebRTC video streaming when `aiortc`, `av`, and `numpy` are installed.
For reliable mobile/remote viewing (5G, cross-country/international), configure TURN/STUN via:

- `ROBOTRON_WEBRTC_ICE_SERVERS` (JSON array of ICE server objects)

Example:

```bash
export ROBOTRON_WEBRTC_ICE_SERVERS='[
  {"urls":["stun:stun.l.google.com:19302"]},
  {"urls":["turn:turn.example.com:3478?transport=udp","turn:turn.example.com:3478?transport=tcp"],"username":"robotron","credential":"YOUR_SECRET"}
]'
```

If unset or invalid, dashboard uses built-in ICE defaults from
`Robotron/Scripts/config.py` (`WEBRTC_ICE_SERVERS`).

## TODO (Known Missing Robotron Wiring)

- Exact MAME input field names for:
  - Start button
  - Coin insert
- Deeper Robotron feature extraction beyond PLDATA/ELIST mirror bytes.

The script now uses real RAM extraction for `PlayerAlive`, PLDATA fields, and enemy-state bag bytes.
