# Pre-Death Partition Implementation

## Overview
Added a third partition to the replay buffer to oversample frames near death (terminal states). This ensures critical learning moments get adequate representation in training batches.

## Partition Configuration

**Before**: 2 partitions (30/70 split)
- High-reward: 30%
- Regular: 70%
- Sampling: 50% / 50%

**After**: 3 partitions (25/25/50 split)
- High-reward: 25% (reward >= 75th percentile)
- Pre-death: 25% (via terminal index tracking)
- Regular: 50% (everything else)
- Sampling: 25% / 25% / 50%

## Why Pre-Death Frames Matter

1. **Final Mistakes**: Last 5-10 frames before death show exactly what went wrong
2. **Proper TD Learning**: Terminal states (done=True) require special handling - no future value to bootstrap
3. **Rare Events**: Deaths are relatively rare, so they naturally get undersampled
4. **Credit Assignment**: Learning "don't do X near enemies" requires seeing the death outcome

## Implementation Details

### Terminal Index Tracking
```python
# In __init__:
self.terminal_indices = deque(maxlen=10000)  # Track last 10K deaths

# In push():
if done:
    self.terminal_indices.append(idx)
```

### Pre-Death Sampling Strategy
When sampling 25% pre-death frames:
1. Pick random terminal indices from `self.terminal_indices`
2. For each terminal, sample backwards by random lookback (5-10 frames from config)
3. Use modulo arithmetic to handle ring buffer wraparound
4. Validate indices are within valid buffer range

### Lookback Configuration
From `config.py`:
```python
replay_terminal_lookback_min: int = 5  # Minimum frames before death
replay_terminal_lookback_max: int = 10  # Maximum frames before death
```

### Performance
- **O(1) per sample**: Simple index arithmetic and modulo operations
- **No numpy scans**: All operations use integer indexing
- **Minimal memory**: Deque holds at most 10K terminal indices (~80KB)
- **Ring buffer safe**: Modulo arithmetic handles wraparound correctly

## Expected Impact

### Before
- Deaths might appear in ~0.1-1% of batches (depending on death frequency)
- Critical pre-death frames get same sampling probability as boring gameplay
- Agent might not learn danger signals effectively

### After
- Pre-death frames guaranteed in 25% of batch
- Each death contributes 5-10 pre-death frames to sampling pool
- Agent gets 25x+ more exposure to critical failure moments
- Should improve danger avoidance and survival behavior

## Monitoring

Check partition stats via `buffer.get_partition_stats()`:
```python
{
    'high_reward': 500000,           # High-reward partition size
    'pre_death': 8432,               # Number of terminal indices tracked
    'regular': 1500000,              # Regular partition size
    'high_reward_threshold': 2.5,   # Current reward threshold
    'terminal_count': 8432,          # Terminal indices available
}
```

## Validation

To verify pre-death sampling is working:
1. Monitor `terminal_count` - should grow as agent dies
2. Check batch composition - look for done=True frames in ~25% of batches
3. Verify agent learns to avoid obvious deaths (e.g., not moving into enemies)
4. Watch for improved survival time on early levels

## Tuning

If needed, adjust via config:
- `replay_terminal_lookback_min/max`: Change how far back to sample before death
- Partition ratios: Modify split in buffer init (currently 25/25/50)
- Terminal deque size: Adjust `maxlen` if you want more/fewer death history

## Notes

- Ring buffer wraparound handled by modulo: `(terminal_idx - lookback) % capacity`
- Validation check ensures sampled indices are within valid buffer range
- Falls back to uniform sampling if not enough terminal indices available
- Zero performance overhead - all O(1) operations
