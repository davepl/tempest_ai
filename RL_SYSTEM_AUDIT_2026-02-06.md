# Tempest RL System Audit (2026-02-06)

## Executive Summary

The existing learner was bottlenecked by both implementation bugs and architectural constraints:

- The model treated `fire/zap` and `spinner` as independent Q-problems, even though reward depends on their joint effect.
- n-step propagation existed in the repo but was not active in the live training path.
- Replay capacity accounting was wrong (effective total capacity doubled), and replay sampling could return undersized batches.
- Several schedule/config controls were disconnected from runtime behavior.

To address this within current scope, the Python learner was refactored to a **joint-action Double DQN** with **per-client n-step rollout integration** and replay/training plumbing fixes, while preserving Lua and socket protocol behavior.

## Key Weaknesses Found

### Code-Level

1. `Scripts/aimodel.py`
- Replay capacity was effectively `2x` requested due two partitions each allocated at full capacity.
- Replay sampling did not fully backfill from available partitions, so effective batch size could shrink.

2. `Scripts/socket_server.py`
- n-step returns were not being applied, despite `n_step` configuration and existing support code.
- Superzap gate controls in config were not applied in action path.

3. `Scripts/config.py` + `Scripts/aimodel.py`
- `decay_epsilon` / `decay_expert_ratio` callbacks referenced by metrics controls were missing.

### Architecture-Level

1. Two-head independence assumption
- Separate heads for `fire/zap` and `spinner` assume additive/separable action value structure.
- In Tempest, this coupling is strong (movement and fire timing interact); a joint-action learner is more appropriate.

2. Weak credit assignment for delayed rewards
- Without n-step in the live path, sparse/delayed score outcomes backpropagate slowly.

3. Ambiguous instrumentation
- Agreement values could look acceptable while still missing joint-action correctness.

## Changes Implemented

### 1) Joint-Action Q Architecture

File: `Scripts/aimodel.py`

- Replaced dual-head output with a single joint Q head over 256 actions (`4 x 64`).
- Added helper mapping functions:
  - `combine_action_indices`
  - `split_joint_action`
  - plus compatibility aliases (`compose_action_index`, `decompose_action_index`, etc.).
- Wired hidden size / depth / dueling / layer-norm config into model construction.

### 2) Replay Buffer Fixes

File: `Scripts/aimodel.py`

- Replay now enforces true total capacity (not duplicated per partition).
- Experience schema now stores:
  - `action_idx` (joint),
  - `reward`,
  - `next_state`,
  - `done`,
  - `horizon`.
- Sampling now backfills from the other partition to keep batch size as full as possible.

### 3) Live n-step Integration

File: `Scripts/socket_server.py`

- Added per-client `NStepReplayBuffer`.
- On each transition:
  - maturity outputs are generated and pushed asynchronously to replay,
  - final transition on terminal frames flushes tail transitions.
- Horizons are now carried into replay for correct n-step target discounting.

### 4) Training Loop Refactor

File: `Scripts/training.py`

- Refactored to joint-action Double DQN training:
  - `Q(s, a_joint)` gather,
  - `argmax_a Q_local(s', a)` for action selection,
  - `Q_target(s', argmax)` for target value.
- Added horizon-aware target:
  - `r_n + gamma^n * max_a' Q_target(s', a')`.
- Kept optional expert imitation as joint-action CE on expert-labeled samples.
- Updated agreement metrics:
  - joint agreement,
  - per-head derived agreement (`fire/zap`, `spinner`) for interpretability.

### 5) Runtime Control/Schedule Plumbing

File: `Scripts/aimodel.py`

- Implemented missing `decay_epsilon` and `decay_expert_ratio`.
- Safe metrics wrapper now respects effective epsilon/expert getters when available.

### 6) Superzap Gate Activation

File: `Scripts/socket_server.py`

- Applied `enable_superzap_gate` and `superzap_prob` in action dispatch path.
- If blocked, configured penalty is deferred and added to the next transition reward.

### 7) Startup Reporting Updates

File: `Scripts/main.py`

- Updated architecture printout to reflect joint-action Q-head and current optimization setup.

## Validation Run

Performed:

- `python3 -m py_compile Scripts/aimodel.py Scripts/training.py Scripts/socket_server.py Scripts/main.py Scripts/config.py`
- Synthetic smoke run: instantiate agent, fill replay, run train step successfully.
- `python3 Scripts/nstep_smoketest.py` passed all cases.

Not performed:

- `pytest` suite (module not installed in this environment).

## Remaining Gaps / Next Actions

1. Evaluation protocol
- Add fixed-seed, no-expert, no-exploration evaluation episodes and track win/score/level progression versus expert baseline.

2. Expert policy quality
- Python-side expert spinner remains heuristic from nearest enemy metadata rather than full Lua expert targeting logic.
- If you want strict teacher quality, mirror expert target output into Python action supervision (without changing transport framing unless necessary).

3. Offline warm start
- Build a short expert-only dataset pass (behavior cloning pretrain) before RL updates each run to reduce early instability.

4. Replay diagnostics
- Surface async replay dropped item count and actor composition time-series directly in metrics row.

5. Confidence-gated expert decay
- Decay expert ratio based on rolling DQN evaluation performance, not just frame count.
