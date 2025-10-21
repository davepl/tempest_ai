# N-Bucket Replay Buffer Test Suite

## Overview

Comprehensive unit tests for the N-bucket stratified replay buffer implementation in `Scripts/aimodel.py`. The test suite validates all major functionality of the TD-error based priority bucketing system.

## Test File

`test_nbucket_replay.py` - 850+ lines, 29 comprehensive tests

## Running Tests

```bash
# Using configured virtual environment
/home/dave/source/repos/tempest_ai/.venv/bin/python test_nbucket_replay.py

# Or activate venv first
source .venv/bin/activate
python test_nbucket_replay.py
```

## Test Coverage

### Core Functionality Tests (`TestNBucketReplayBuffer` - 26 tests)

#### Initialization & Configuration
- ✅ `test_initialization` - Buffer setup, capacity calculation, storage allocation
- ✅ `test_bucket_configuration` - Percentile ranges (90-100%, 80-90%, etc.)

#### Push Operations
- ✅ `test_push_basic` - Single experience storage
- ✅ `test_push_multiple` - Batch experience insertion
- ✅ `test_continuous_action_clamping` - Action range validation [-0.9, +0.9]
- ✅ `test_discrete_action_range` - Valid discrete actions (0-3)
- ✅ `test_invalid_actor_tag` - Actor tag validation (must be 'dqn' or 'expert')
- ✅ `test_invalid_horizon` - Horizon validation (must be >= 1)
- ✅ `test_state_size_mismatch_handling` - State size padding/truncation

#### TD-Error & Bucket Routing
- ✅ `test_bucket_routing_low_td_error` - Low TD-error → main bucket
- ✅ `test_bucket_routing_high_td_error` - High TD-error → priority buckets
- ✅ `test_bucket_fill_progression` - Bucket fill order by TD-error
- ✅ `test_percentile_threshold_update` - Threshold adaptation (50th-90th percentiles)
- ✅ `test_td_error_window_size` - Rolling window (50K max) maintenance

#### Sampling
- ✅ `test_sample_before_enough_data` - Insufficient data handling (returns None)
- ✅ `test_sample_basic` - Batch sampling structure and types
- ✅ `test_sample_uniform_distribution` - Uniform sampling across buffer
- ✅ `test_multiple_sample_batches` - Independent batch sampling

#### Ring Buffer Behavior
- ✅ `test_ring_buffer_wraparound` - Buffer wraparound at capacity
- ✅ `test_terminal_state_handling` - Done flag storage
- ✅ `test_horizon_values` - N-step horizon tracking

#### Statistics & Monitoring
- ✅ `test_actor_composition_dqn_only` - DQN-only composition tracking
- ✅ `test_actor_composition_expert_only` - Expert-only composition tracking
- ✅ `test_actor_composition_mixed` - Mixed actor composition
- ✅ `test_partition_stats` - Bucket stats reporting
- ✅ `test_empty_buffer_stats` - Empty buffer edge case

### Integration Tests (`TestNBucketIntegration` - 3 tests)

#### Realistic Scenarios
- ✅ `test_realistic_training_scenario` - Training with exponential TD-error distribution
- ✅ `test_expert_to_dqn_transition` - Expert → DQN transition with buffer wraparound
- ✅ `test_varying_horizons_distribution` - Mixed n-step horizons (1-10)

## Key Features Validated

### 1. Bucket Architecture
- 5 priority buckets (250K each) for 50th-100th percentile TD-errors
- 1 main bucket (1M) for <50th percentile TD-errors
- Contiguous storage for O(1) sampling
- Independent ring buffers per bucket

### 2. TD-Error Stratification
- Rolling 50K TD-error window for threshold calculation
- Percentile thresholds update every 1000 insertions
- Bucket routing based on 90th, 80th, 70th, 60th, 50th percentiles
- Natural oversampling of high-error experiences

### 3. Data Integrity
- State size validation (padding/truncation)
- Continuous action clamping [-0.9, +0.9]
- Discrete action range [0-3]
- Actor tag enforcement ('dqn', 'expert')
- Horizon validation (>= 1)

### 4. Sampling & Performance
- Uniform sampling from contiguous storage
- O(1) sample operation (no tree structures)
- Proper tensor/numpy conversion
- CUDA pinned memory support

### 5. Monitoring & Telemetry
- Per-bucket fill statistics
- Actor composition tracking
- TD-error threshold reporting
- Terminal state counts

## Test Configuration

Tests use smaller buffer sizes for performance:
- Priority buckets: 1,000 capacity each (vs 250K production)
- Main bucket: 5,000 capacity (vs 1M production)
- Total test capacity: 10,000 (vs 2.25M production)

Integration tests use larger buffers (100K total) to validate wraparound behavior.

## Test Results

**All 29 tests pass** ✅

```
Ran 29 tests in 3.3s
Successes: 29
Failures: 0
Errors: 0
```

## Key Test Insights

1. **Threshold Establishment**: Buckets start with zero thresholds; need ~1500 varied TD-error experiences to establish meaningful percentiles

2. **Bucket Fill Order**: Without established thresholds, all experiences go to first bucket. After threshold calibration, routing becomes TD-error dependent

3. **Buffer Wraparound**: Each bucket wraps independently. Total size = sum of individual bucket sizes (not necessarily total capacity until all buckets fill)

4. **Actor Composition**: Reflects all stored experiences, not just recent ones, until buffer wraps

5. **Percentile Precision**: Thresholds approximate numpy percentiles within ~10% (sufficient for prioritization)

## Usage Example

```python
from aimodel import HybridReplayBuffer

# Create buffer
buffer = HybridReplayBuffer(capacity=100000, state_size=128)

# Push experience with TD-error
buffer.push(
    state=state,
    discrete_action=action_idx,
    continuous_action=spinner_val,
    reward=reward,
    next_state=next_state,
    done=done,
    actor='dqn',
    horizon=1,
    td_error=abs_td_error
)

# Sample batch
batch = buffer.sample(batch_size=64)
if batch is not None:
    states, discrete_actions, continuous_actions, rewards, next_states, dones, actors, horizons = batch

# Get statistics
stats = buffer.get_partition_stats()
print(f"P90-100 bucket: {stats['p90_100_fill_pct']:.1f}% full")
print(f"Main bucket: {stats['main_fill_pct']:.1f}% full")
```

## Future Test Ideas

- [ ] Stress test with millions of experiences
- [ ] Benchmark insertion/sampling performance
- [ ] Test with real DQN network TD-error computation
- [ ] Validate bucket distribution matches expected percentiles over long runs
- [ ] Test with different N (number of priority buckets)
- [ ] Concurrent access patterns (if multithreading added)
- [ ] Memory profiling under high load

## Notes

- Tests restore original config after each test (via setUp/tearDown)
- Random seed not fixed - tests are probabilistic but robust
- GPU tests run on CPU (no CUDA required)
- TD-error computation approximation validated (close enough for bucketing)
