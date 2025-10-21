# N-Bucket Replay Buffer Hotkey Feature

## Overview

Added hotkey **'b'** to display detailed N-bucket replay buffer statistics during training.

## Usage

While the training script is running, press **'b'** to display a comprehensive statistics table showing:

### Overall Statistics
- **Total Size**: Current buffer size vs capacity with percentage
- **DQN Experiences**: Count and percentage of DQN-generated experiences
- **Expert Experiences**: Count and percentage of expert-generated experiences

### Priority Bucket Breakdown
Visual table showing for each bucket:
- **Bucket name** (p90_100, p80_90, etc., and main)
- **Percentile range** (what TD-error percentile it represents)
- **Size** (current fill vs capacity)
- **Fill percentage** with visual bar graph using block characters (█ filled, ░ empty)

### TD-Error Percentile Thresholds
Current threshold values for bucket classification:
- 90th, 80th, 70th, 60th, and 50th percentiles
- These values are auto-updated every 1000 insertions based on rolling 50K TD-error window

### Sampling Metrics
- Batch size configuration
- Total training steps completed
- Total experiences added to buffer

## Example Output

```
==========================================================================================
                              N-BUCKET REPLAY BUFFER STATISTICS
==========================================================================================

OVERALL STATISTICS
------------------------------------------------------------------------------------------
  Total Size:                 5,000 /    2,250,000 (  0.2%)
  DQN Experiences:            1,571   ( 31.4%)
  Expert Experiences:         1,107   ( 22.1%)

PRIORITY BUCKET BREAKDOWN
------------------------------------------------------------------------------------------
  Bucket          Percentile      Size                 Capacity             Fill %
------------------------------------------------------------------------------------------
  p90_100         90-100%            2,678 / 250,000      1.1%  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]
  p80_90          80-90%               561 / 250,000      0.2%  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]
  p70_80          70-80%               429 / 250,000      0.2%  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]
  p60_70          60-70%               297 / 250,000      0.1%  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]
  p50_60          50-60%               234 / 250,000      0.1%  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]
  main            <50%                 801 / 1,000,000     0.1%  [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]

TD-ERROR PERCENTILE THRESHOLDS
------------------------------------------------------------------------------------------
  90th percentile:       3.4555
  80th percentile:       2.5479
  70th percentile:       1.9580
  60th percentile:       1.4724
  50th percentile:       1.0291

SAMPLING METRICS
------------------------------------------------------------------------------------------
  Recent batch size:   64
  Training steps:                 0
  Experiences added:          5,000

==========================================================================================
```

## Implementation Details

### Files Modified

1. **Scripts/main.py**:
   - Added `print_bucket_stats()` function to format and display statistics
   - Added 'b' hotkey handler in `keyboard_input_handler()`
   - Updated network config display to show keyboard controls including 'b'
   - Updated header comment to mention new hotkey

### Function Signature

```python
def print_bucket_stats(agent, kb_handler):
    """Print detailed N-bucket replay buffer statistics table.
    
    Args:
        agent: HybridDQNAgent instance with replay buffer
        kb_handler: KeyboardHandler for terminal restoration
    """
```

### Error Handling

- Checks if agent has replay buffer before attempting to display stats
- Wraps entire display in try-except to handle any errors gracefully
- Restores terminal to raw mode after display (for continued keyboard input)

### Visual Features

- 30-character wide progress bars using Unicode block characters
- Comma-separated thousands for better readability
- Aligned columns for clean table layout
- Color-coded sections with separators

## Debugging Use Cases

The bucket stats display is useful for:

1. **Verifying Bucket Distribution**: Check if experiences are being routed to correct buckets based on TD-error
2. **Monitoring Fill Rates**: See which buckets are filling fastest (indicates TD-error distribution)
3. **Threshold Calibration**: Verify that percentile thresholds are updating reasonably
4. **Actor Balance**: Check DQN vs Expert experience ratio in buffer
5. **Capacity Planning**: Monitor total buffer utilization to tune bucket sizes

## Related Files

- `test_bucket_hotkey.py` - Test script demonstrating bucket stats display
- `test_nbucket_replay.py` - Comprehensive unit tests for N-bucket system
- `TEST_NBUCKET_README.md` - Full documentation of N-bucket replay buffer

## Performance Notes

- Display operation is O(1) - just reads statistics from buffer
- No impact on training performance (only runs when hotkey pressed)
- Terminal restoration ensures keyboard continues working after display
