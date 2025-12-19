# DQN Learning Stagnation Analysis

## Symptoms
- **DQN1M (1M-frame average reward)** is flatlining around 2300.
- **Loss** is extremely low (~0.002).
- **Gradient Norm** is effectively zero (~0.000 - 0.005).
- **Agreement** is low (~16%).
- **Q-Range** is stable but low (`[400, 550]` scaled, `[0.04, 0.05]` internal).

## Diagnosis
The agent is suffering from **Vanishing Gradients due to Reward Scaling**.

### The Math
1.  **Reward Scaling**: `0.0001` (1e-4).
    - A 3000-point episode results in `0.3` total reward units.
    - Per-step reward is `0.3 / 700 steps` ≈ `4e-4`.
2.  **Q-Values**: With `gamma=0.99`, Q-values accumulate to `~0.04` (internal units).
3.  **Loss Calculation**: `SmoothL1Loss(Q, Target)`.
    - If error is `10%` of Q-value (`0.004`), the loss is `0.5 * (0.004)^2` ≈ `8e-6`.
    - This is **tiny**.
4.  **Gradient Calculation**:
    - Gradients are proportional to the error (`Q - Target`).
    - Error magnitude is `~0.004`.
    - `Adam` optimizer updates weights by `lr * gradient`.
    - `1e-4 * 0.004` = `4e-7`.
    - This update is so small it likely falls below floating point precision or is just ineffective.

### Why Loss appears as 0.002?
The user's log shows `Loss` as `0.002`.
If `Loss` is `0.002`, then `0.5 * delta^2 = 0.002` -> `delta^2 = 0.004` -> `delta = 0.06`.
A delta of `0.06` is actually **larger** than the estimated Q-value (`0.04`).
This contradicts the "tiny gradient" theory if the loss is truly `0.002`.

**However**, if the `GradNorm` is `0.000`, it means the backpropagation is finding almost no slope.
This happens if:
1.  **Dead ReLUs**: The network outputs are constant 0.
    - But Q-values are non-zero (`0.04`).
2.  **Flat Plateau**: The network is in a region where changing weights doesn't affect loss (unlikely with ReLUs).
3.  **Clipping**: Is `GradNorm` calculated *before* or *after* clipping?
    - In `training.py`: `total_norm` is calculated *before* clipping.

**Alternative Theory: Learning Rate is too low for the scale.**
If the gradients are naturally small (due to small rewards), we need a larger Learning Rate to make meaningful updates.
Or, we need to scale rewards up.

## Solution
We should **increase the Reward Scale** to bring Q-values and Gradients into a healthier range (e.g., Q-values around 1.0 - 10.0).

### Proposed Changes
1.  **Increase `obj_reward_scale`**: From `0.0001` to `0.01` (100x increase).
    - 1 point = 0.01 reward units.
    - 3000 points = 30.0 reward units.
    - Q-values ≈ 3.0 - 5.0.
    - Gradients will be ~100x larger.
2.  **Adjust `subj_reward_scale`**: Maintain the ratio (currently 1/4 of obj).
    - From `0.000025` to `0.0025`.
3.  **Adjust `death_penalty`**:
    - Previous: `-0.1` (1000 points).
    - New scale: `-10.0` (1000 points * 0.01).

This will make the "internal units" much larger, resulting in larger gradients and faster learning.
