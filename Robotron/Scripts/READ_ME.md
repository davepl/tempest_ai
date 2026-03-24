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
- Add an escape vector derived from local threat density so the agent/player can bias movement toward lower-pressure space instead of using a naive enemy centroid.
- Replace large per-frame enemy tables with a smaller persistent active roster chosen by salience, e.g. nearest/most important non-hulks, hulks, and electrodes, while preserving slot identity with hysteresis to avoid roster churn.
- Move toward a more Tempest-like compact state: roughly 24 salient object slots with about 8 attributes each, plus a small global summary block (player state, nearest-class distances, 4-way directional threat/openness, crowding/projectile pressure, and escape vector) instead of heavier lane/set/grid representations.
- Update the frame extractor/state schema to emit the compact global frame block explicitly: alive, wave, laser count, optional score/score delta, player position/velocity, nearest distances by class (enemy/hulk/electrode/human), 4-way threat, 4-way openness, crowding score, local projectile pressure, and escape vector.
- Add the selection/scoring code needed to support that compact state efficiently: salience scoring for object promotion, hysteresis/stickiness for stable slot identity, and cheap per-frame aggregation for directional threat/openness and escape-pressure summaries without lane/set layers.
- Stack `N` compact frames for temporal context, with `N=2` initially, and/or add short-horizon deltas where needed so the policy can infer motion and pressure changes without a large spatial history.
- Include a small action/history context block if needed for stability, such as previous move direction, previous fire direction, recent damage/death indicator, and recent rescue indicator.
- Add at least one compact opportunity signal alongside danger signals, e.g. human rescue opportunity and/or safe-fire opportunity, so the policy does not collapse into pure evasion.
- Normalize the compact state consistently across all fields and wave conditions so type, threat, distance, and pressure features stay stationary enough for learning.
- Build debug visualization for the compact representation: tracked roster slots, per-slot salience, directional threat/openness summaries, and the escape vector, so state extraction can be validated against live gameplay.
- Add ablation/config hooks to toggle major compact-state components independently, including the active roster, escape vector, directional summaries, projectile pressure, opportunity signals, and frame-stack depth.
- Add replay/frame inspection tooling to dump selected slots, top rejected candidates, global summary values, and chosen actions for failure analysis on dense waves.
- Evaluate and tune the compact system specifically on late-wave/high-density scenes so off-roster enemy mass and slot-churn failures are caught early.

The script now uses real RAM extraction for `PlayerAlive`, PLDATA fields, and enemy-state bag bytes.
