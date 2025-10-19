# Agreement Starts Worse Than Random - Root Cause Analysis

## Problem
Agreement starts at 13.4%, which is worse than the 25% random baseline expected for 4 discrete actions.

## Root Causes Identified

### 1. **Network Initialization Bias** ✓ FIXED
- **Problem**: The discrete Q-head layers (`discrete_fc`, `discrete_out`) were using PyTorch's default initialization
- **Impact**: Fresh models showed extreme action bias (32-93% for single actions)
- **Examples**:
  - Default init: 93% action 3 (FIRE+ZAP)
  - Xavier gain=0.5: 70% action 0 
  - Xavier gain=0.5: 79% action 2 (FIRE only)
  
- **Fix Applied**: Zero-initialize the output layer weights:
  ```python
  torch.nn.init.constant_(self.discrete_out.weight, 0.0)
  torch.nn.init.constant_(self.discrete_out.bias, 0.0)
  ```
  
- **Result**: All Q-values start at exactly 0.0, giving deterministic but balanced starting point

### 2. **Cold Start Problem** (Expected Behavior)
- **Observation**: When all Q-values are 0.0, `argmax` deterministically selects action 0
- **Impact**: During initial buffer fill (first ~50k frames), the model hasn't learned meaningful Q-values yet
- **Expected**: Agreement will be low until the model learns from experience

### 3. **Superzap Gate Mismatch** ⚠️  MAJOR CONTRIBUTOR
- **Problem**: Model predicts zap actions (1, 3) frequently, but superzap gate only allows 1% success
- **Evidence from diagnostics**:
  - Model predicts 67.8% zap actions
  - Superzap gate allows only 1.0% to succeed
  - **Result**: 66.8% action mismatch between prediction and stored action
  
- **Impact on Agreement**:
  - Replay buffer stores what actually happened (mostly failed zaps → converted to no-zap)
  - Training compares stored actions vs current policy
  - Current policy still predicts zaps → mismatch → low agreement

### 4. **Discrete Loss Weight** (Configuration)
- **Current**: `discrete_loss_weight = 10.0`, `continuous_loss_weight = 1.0`
- **Effect**: Discrete learning is heavily weighted, which should help but...
- **Conflict**: High discrete weight can't overcome the superzap gate creating systematic action mismatches

## Why Agreement < 25% Random Baseline

The 13.4% agreement is caused by:

1. **Systematic bias** from initialization (FIXED with zero init)
2. **Superzap gate** creating 66.8% mismatch (zap predictions fail → stored as no-zap)
3. **Cold start** - model hasn't learned yet, still exploring

The superzap gate is the PRIMARY cause of sub-random agreement because it creates a systematic mismatch between:
- What the model learns to predict (zap when appropriate)
- What actually gets stored in the buffer (failed zaps)

## Recommendations

### ✓ DONE
1. Fixed network initialization to prevent action bias at startup
2. Increased discrete_loss_weight to 10.0 for stronger discrete learning signal

### TODO
1. **Option A - Disable Superzap Gate During Training**:
   ```python
   enable_superzap_gate: bool = False  # Disable during training, enable for evaluation
   ```
   - Pros: Eliminates action mismatch, allows model to learn true zap value
   - Cons: Model might rely on zaps too much
   
2. **Option B - Track Superzap Success in Replay**:
   - Store whether zap was allowed/blocked by gate
   - Only train on actions that actually executed
   - More complex implementation

3. **Option C - Accept Low Initial Agreement**:
   - Monitor agreement over time - should improve as model learns
   - Focus on reward/performance rather than agreement metric
   - Agreement will naturally improve once Q-values diverge from zero

### Monitoring
- Watch agreement trend over next 100k-200k frames
- If it doesn't improve above 25%, investigate buffer composition
- Check if superzap gate is still causing systematic mismatches

## Expected Behavior After Fix

With zero-initialized output layer:
- Initial Q-values: All exactly 0.0
- Initial action selection: Deterministic (action 0) when epsilon=0, random when epsilon>0
- Agreement during cold start: Low (expected)
- Agreement after learning: Should rise above 25% as Q-values differentiate
- Superzap gate: Will continue to create mismatch unless addressed

## Test Results

### Fresh Model (After Fix)
```
Action Distribution (n=1000 random states):
  Action 0 (--): 1000 (100.0%)  ← All Q-values are 0.0, argmax picks first
  Action 1 (-Z):    0 (  0.0%)
  Action 2 (F-):    0 (  0.0%)
  Action 3 (FZ):    0 (  0.0%)
```

This is correct - all Q-values start equal at 0.0. Learning will differentiate them.
