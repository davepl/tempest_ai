# Startup Display Documentation

**Date**: October 18, 2025  
**Feature**: Network configuration and hyperparameter display at startup

## What Was Added

Added `print_network_config()` function to `Scripts/main.py` that displays comprehensive information about:
- Network architecture (layers, dimensions, parameters)
- Training hyperparameters (learning rate, batch size, etc.)
- Loss configuration (weights, clipping, BC settings)
- Expert guidance settings
- Optimization settings

## Example Output

```
====================================================================================================
                              TEMPEST AI - NETWORK CONFIGURATION                                    
====================================================================================================

📐 NETWORK ARCHITECTURE:
   State Size:        128
   Discrete Actions:  4 (FIRE/ZAP combinations)
   Continuous Output: 1 (Spinner: -0.9 to +0.9)

   Shared Trunk:      5 layers
      Layer 1:        128 → 512
      Layer 2:        512 → 512
      Layer 3:        512 → 256
      Layer 4:        256 → 256
      Layer 5:        256 → 128

   Discrete Head:     128 → 64 → 4
   Continuous Head:   128 → 64 → 32 → 1

   Total Parameters:  534,916
   Trainable:         534,916

⚙️  TRAINING HYPERPARAMETERS:
   Learning Rate:     0.000250
   Batch Size:        2,048
   Gamma (γ):         0.99
   Epsilon (ε):       0.05 → 0.05 (exploration)
   Memory Size:       2,000,000 transitions
   Target Update:     Every 500 steps

⚖️  LOSS CONFIGURATION:
   Discrete Loss:     TD (Huber) + BC (Cross-Entropy)
   Continuous Loss:   MSE with advantage weighting
   Loss Weights:      Discrete=1.0, Continuous=1.0
   BC Weight:         1.0 (behavioral cloning)
   Max Q-Value Clip:  50.0
   TD Target Clip:    1500.0

🎓 EXPERT GUIDANCE:
   Expert Ratio:      50%
   Superzap Gate:     Disabled

🚀 OPTIMIZATION:
   Gradient Clip:     10.0 (max norm)
   N-Step Returns:    1-step
   Training Workers:  4

====================================================================================================

✓ Loaded model from: models/tempest_model_latest.pt
```

## Key Information Displayed

### Network Architecture
- **State Size**: Input dimension (128 features)
- **Action Space**: 4 discrete + 1 continuous
- **Layer Structure**: Shows exact dimensions of each layer in shared trunk
- **Head Architecture**: Separate discrete and continuous paths
- **Parameter Count**: Total trainable parameters

### Training Hyperparameters
- **Learning Rate**: 0.00025 (standard Atari DQN rate)
- **Batch Size**: 2048 samples per update
- **Gamma**: Discount factor for future rewards
- **Epsilon**: Exploration rate
- **Memory Size**: Replay buffer capacity
- **Target Update Frequency**: How often target network syncs

### Loss Configuration
- **Discrete Loss**: Combination of TD learning (Huber) and Behavioral Cloning (Cross-Entropy)
- **Continuous Loss**: MSE with advantage-based importance sampling
- **Loss Weights**: Currently balanced at 1.0 each
- **BC Weight**: Strength of expert imitation signal
- **Clipping**: Q-value and TD target clipping thresholds

### Expert Guidance
- **Expert Ratio**: Percentage of expert data in replay buffer
- **Superzap Gate**: Whether expert zap actions are gated by game logic

### Optimization
- **Gradient Clipping**: Maximum gradient norm before clipping
- **N-Step**: Horizon for n-step TD returns
- **Workers**: Number of parallel training threads

## Benefits

1. **Transparency**: See exactly what settings are active at startup
2. **Debugging**: Quickly verify configuration matches expectations
3. **Documentation**: Self-documenting runs with full configuration visible
4. **Comparison**: Easy to compare different experimental runs
5. **Troubleshooting**: Immediately spot misconfigured parameters

## When It Displays

The configuration display appears:
1. After agent creation
2. Before model loading
3. Once at startup (not repeated)

## Files Modified

- `Scripts/main.py`: Added `print_network_config()` function and call in `main()`
- Added `import torch` for layer type checking
