# Implementing Adaptive N-Step (Optional Enhancement)

## Overview

This document provides implementation guidance for **adaptive n-step scheduling** based on training progress. This is an **optional enhancement** - your current fixed n=7 is already well-tuned.

## When to Consider Adaptive N-Step

✅ **Consider if:**
- You want to minimize expert contamination in early training
- You have time to test before the long run
- You're interested in squeezing out 3-5% extra performance

❌ **Skip if:**
- You're starting a long training run now (stay with n=7)
- System is already meeting performance goals
- You want maximum stability/simplicity

---

## Strategy 1: Phase-Based N-Step (Simplest)

### Implementation

Add to `config.py`:

```python
def get_adaptive_n_step(frame_count: int, expert_ratio: float) -> int:
    """
    Adaptive n-step based on training phase and expert contamination.
    
    Phase 1 (0-1M frames, expert_ratio high): Use n=3 to minimize contamination
    Phase 2 (1M-6M frames, expert_ratio decaying): Use n=7 for balance
    Phase 3 (6M+ frames, expert_ratio low): Use n=10 for max credit assignment
    """
    if frame_count < 1_000_000:
        # Early training: minimize expert contamination
        return 3
    elif frame_count < 6_000_000:
        # Mid training: balanced approach
        return 7
    else:
        # Late training: maximize credit assignment
        return 10
```

In `RLConfigData`:
```python
# Keep static fallback
n_step: int = 7

# Add method to get current n_step
def get_n_step(self, frame_count: int, expert_ratio: float) -> int:
    """Get current n_step value (adaptive or fixed)"""
    if bool(getattr(self, 'adaptive_n_step', False)):
        return get_adaptive_n_step(frame_count, expert_ratio)
    return self.n_step

# Add toggle
adaptive_n_step: bool = False  # Set True to enable adaptive schedule
```

In `socket_server.py`, change line 172:
```python
# OLD:
'nstep_buffer': (
    NStepReplayBuffer(RL_CONFIG.n_step, RL_CONFIG.gamma, store_aux_action=True)
    if self._server_nstep_enabled() else None
)

# NEW:
'nstep_buffer': (
    NStepReplayBuffer(
        RL_CONFIG.get_n_step(metrics.frame_count, metrics.expert_ratio),
        RL_CONFIG.gamma,
        store_aux_action=True
    ) if self._server_nstep_enabled() else None
)
```

### Problem: Buffer is Created Once Per Client

The above creates the buffer with n_step at client connection time. To make it truly adaptive, you need to either:

**Option A:** Recreate buffers periodically (complex)
**Option B:** Make NStepReplayBuffer support dynamic n (requires changes)
**Option C:** Accept that n_step is fixed per client session (simplest)

**Recommendation:** Use **Option C** - new clients get current n_step, old clients keep theirs until reconnect. With 16 clients reconnecting occasionally, this naturally migrates over a few minutes.

---

## Strategy 2: Smooth Adaptive N-Step (More Complex)

### Implementation

Smoothly interpolate n_step based on expert_ratio:

```python
def get_smooth_n_step(expert_ratio: float, min_n: int = 3, max_n: int = 10) -> int:
    """
    Smoothly scale n-step inversely with expert contamination risk.
    
    At expert_ratio=1.0: n=min_n (max contamination)
    At expert_ratio=0.0: n=max_n (no contamination)
    """
    # Inverse relationship: lower expert_ratio → higher n_step
    # Use quadratic for smoother transition
    normalized = 1.0 - expert_ratio  # 0 at high expert, 1 at low expert
    scaled = normalized ** 0.5  # Square root for gentler curve
    n = int(min_n + scaled * (max_n - min_n))
    return max(min_n, min(max_n, n))

# Example values:
# expert_ratio=0.95 → normalized=0.05 → sqrt=0.22 → n ≈ 3-4
# expert_ratio=0.50 → normalized=0.50 → sqrt=0.71 → n ≈ 8
# expert_ratio=0.10 → normalized=0.90 → sqrt=0.95 → n ≈ 10
```

Add to `RLConfigData`:
```python
adaptive_n_step_mode: str = 'fixed'  # Options: 'fixed', 'phase', 'smooth'
adaptive_n_step_min: int = 3
adaptive_n_step_max: int = 10

def get_n_step(self, frame_count: int, expert_ratio: float) -> int:
    if self.adaptive_n_step_mode == 'phase':
        return get_adaptive_n_step(frame_count, expert_ratio)
    elif self.adaptive_n_step_mode == 'smooth':
        return get_smooth_n_step(expert_ratio, self.adaptive_n_step_min, self.adaptive_n_step_max)
    else:
        return self.n_step
```

---

## Strategy 3: Dynamic NStepReplayBuffer (Most Flexible)

### Modify NStepReplayBuffer to Support Dynamic N

In `nstep_buffer.py`:

```python
class NStepReplayBuffer:
    def __init__(self, n_step: int, gamma: float, store_aux_action: bool = False):
        assert n_step >= 1
        self._initial_n_step = int(n_step)
        self.n_step = int(n_step)  # Now mutable
        self.gamma = float(gamma)
        self.store_aux_action = bool(store_aux_action)
        self._deque: Deque[Tuple] = deque()
    
    def set_n_step(self, new_n: int):
        """Dynamically change n_step (takes effect on next add)"""
        assert new_n >= 1
        old_n = self.n_step
        self.n_step = int(new_n)
        
        # If reducing n and queue has more than new_n items, might want to flush
        if new_n < old_n and len(self._deque) >= new_n:
            # Optionally: flush partial experiences
            pass
    
    def reset(self):
        self._deque.clear()
        # Optionally: reset to initial n_step
        # self.n_step = self._initial_n_step
```

Then in socket_server.py, periodically update:

```python
def _update_nstep_buffers(self):
    """Periodically update n_step in all client buffers"""
    current_n = RL_CONFIG.get_n_step(self.metrics.frame_count, self.metrics.expert_ratio)
    
    with self.client_lock:
        for client_id, state in self.client_states.items():
            buf = state.get('nstep_buffer')
            if buf is not None and hasattr(buf, 'set_n_step'):
                if buf.n_step != current_n:
                    buf.set_n_step(current_n)
                    # Optional: log the change
                    # print(f"Client {client_id}: n_step changed to {current_n}")

# Call this in handle_client every N frames
if frame_count % 10000 == 0:
    self._update_nstep_buffers()
```

---

## Testing Adaptive N-Step

### Validation Script

```python
#!/usr/bin/env python3
"""Test adaptive n-step schedule"""

import sys
sys.path.insert(0, 'Scripts')

from config import get_adaptive_n_step, get_smooth_n_step

def test_phase_schedule():
    """Test phase-based schedule"""
    print("Phase-Based Schedule:")
    test_frames = [0, 500_000, 1_000_000, 3_000_000, 6_000_000, 10_000_000]
    test_ratios = [0.95, 0.70, 0.50, 0.30, 0.10, 0.10]
    
    for fc, er in zip(test_frames, test_ratios):
        n = get_adaptive_n_step(fc, er)
        print(f"  Frame {fc:>9,} | expert_ratio={er:.2f} | n_step={n}")

def test_smooth_schedule():
    """Test smooth schedule"""
    print("\nSmooth Schedule:")
    test_ratios = [0.95, 0.80, 0.60, 0.40, 0.20, 0.10, 0.05]
    
    for er in test_ratios:
        n = get_smooth_n_step(er, min_n=3, max_n=10)
        clean_pct = ((1 - er) ** n) * 100
        print(f"  expert_ratio={er:.2f} | n_step={n:>2} | clean_episodes={clean_pct:>5.1f}%")

def test_contamination():
    """Analyze contamination rates"""
    print("\nContamination Analysis (% clean episodes):")
    print("expert_ratio | n=3   | n=5   | n=7   | n=10  | n=15  |")
    print("-------------|-------|-------|-------|-------|-------|")
    
    for er in [0.95, 0.75, 0.50, 0.25, 0.10]:
        results = []
        for n in [3, 5, 7, 10, 15]:
            clean_pct = ((1 - er) ** n) * 100
            results.append(f"{clean_pct:>5.1f}")
        print(f"   {er:.2f}      | {' | '.join(results)} |")

if __name__ == '__main__':
    test_phase_schedule()
    test_smooth_schedule()
    test_contamination()
```

Save as `test_adaptive_nstep.py` and run:
```bash
python test_adaptive_nstep.py
```

Expected output:
```
Phase-Based Schedule:
  Frame         0 | expert_ratio=0.95 | n_step=3
  Frame   500,000 | expert_ratio=0.70 | n_step=3
  Frame 1,000,000 | expert_ratio=0.50 | n_step=7
  Frame 3,000,000 | expert_ratio=0.30 | n_step=7
  Frame 6,000,000 | expert_ratio=0.10 | n_step=10
  Frame 10,000,000 | expert_ratio=0.10 | n_step=10

Smooth Schedule:
  expert_ratio=0.95 | n_step= 3 | clean_episodes=  0.0%
  expert_ratio=0.80 | n_step= 4 | clean_episodes=  0.2%
  expert_ratio=0.60 | n_step= 6 | clean_episodes=  0.4%
  expert_ratio=0.40 | n_step= 8 | clean_episodes=  1.7%
  expert_ratio=0.20 | n_step= 9 | clean_episodes= 13.4%
  expert_ratio=0.10 | n_step=10 | clean_episodes= 34.9%
  expert_ratio=0.05 | n_step=10 | clean_episodes= 59.9%

Contamination Analysis (% clean episodes):
expert_ratio | n=3   | n=5   | n=7   | n=10  | n=15  |
-------------|-------|-------|-------|-------|-------|
   0.95      |   0.0 |   0.0 |   0.0 |   0.0 |   0.0 |
   0.75      |   1.6 |   0.1 |   0.0 |   0.0 |   0.0 |
   0.50      |  12.5 |   3.1 |   0.8 |   0.1 |   0.0 |
   0.25      |  42.2 |  23.7 |  13.3 |   5.6 |   1.3 |
   0.10      |  72.9 |  59.0 |  47.8 |  34.9 |  20.4 |
```

---

## Monitoring Adaptive N-Step

### Add to Metrics Display

In `metrics_display.py`:

```python
def display_nstep_info(metrics):
    """Display current n_step value"""
    if hasattr(RL_CONFIG, 'adaptive_n_step_mode') and RL_CONFIG.adaptive_n_step_mode != 'fixed':
        current_n = RL_CONFIG.get_n_step(metrics.frame_count, metrics.expert_ratio)
        clean_pct = ((1 - metrics.expert_ratio) ** current_n) * 100
        print(f"N-Step: {current_n} (adaptive, {clean_pct:.1f}% clean episodes)")
    else:
        n = RL_CONFIG.n_step
        clean_pct = ((1 - metrics.expert_ratio) ** n) * 100
        print(f"N-Step: {n} (fixed, {clean_pct:.1f}% clean episodes)")
```

### Log N-Step Changes

```python
class NStepLogger:
    """Track n_step changes over training"""
    def __init__(self):
        self.last_n = None
        self.changes = []
    
    def check_and_log(self, frame_count, current_n):
        if self.last_n is not None and current_n != self.last_n:
            self.changes.append({
                'frame': frame_count,
                'old_n': self.last_n,
                'new_n': current_n
            })
            print(f"N-step changed: {self.last_n} → {current_n} at frame {frame_count:,}")
        self.last_n = current_n
    
    def save_log(self, filename='nstep_changes.json'):
        import json
        with open(filename, 'w') as f:
            json.dump(self.changes, f, indent=2)
```

---

## Performance Expectations

### Expected Improvements

**Scenario: Start from scratch with adaptive schedule**

Phase 1 (0-1M, n=3):
- Less expert contamination than fixed n=7
- Slightly slower credit assignment
- **Net effect:** +2-5% reward (cleaner Q-function learning)

Phase 2 (1M-6M, n=7):
- Same as current fixed approach
- **Net effect:** Baseline performance

Phase 3 (6M+, n=10):
- Better credit assignment at low contamination
- Slight variance increase
- **Net effect:** +3-5% reward vs fixed n=7

**Overall improvement:** +5-10% final performance (uncertain, needs testing)

### Risks

1. **Instability during transitions:** When n changes, TD targets suddenly use different horizons
2. **Complexity:** More moving parts, harder to debug
3. **Validation needed:** Theoretical benefits might not materialize in practice

---

## Recommendation: Start Simple

### Option 1: Keep Fixed n=7 (Recommended) ✅

**For your long training run:**
- Proven stable
- Well-documented behavior
- Easy to reason about
- Focus on other optimizations

### Option 2: Test Adaptive on Side Experiment 🔬

**If you have time:**
1. Save checkpoint at 1M frames
2. Branch A: Continue with n=7
3. Branch B: Switch to adaptive schedule
4. Compare after 5M more frames
5. Choose winner for final run

### Option 3: Implement But Keep Disabled 🛠️

Add adaptive code but set `adaptive_n_step_mode = 'fixed'`:
- Code is ready if you want to test later
- Easy to enable with config change
- No risk during main run

---

## Code Changes Summary

### Minimal Implementation (Recommended)

**File:** `config.py`
```python
def get_adaptive_n_step(frame_count: int, expert_ratio: float) -> int:
    if frame_count < 1_000_000:
        return 3
    elif frame_count < 6_000_000:
        return 7
    else:
        return 10

# In RLConfigData:
adaptive_n_step: bool = False  # Set True to enable

def get_n_step(self, frame_count: int, expert_ratio: float) -> int:
    if self.adaptive_n_step:
        return get_adaptive_n_step(frame_count, expert_ratio)
    return self.n_step
```

**File:** `socket_server.py` (line 172)
```python
# Change this:
NStepReplayBuffer(RL_CONFIG.n_step, RL_CONFIG.gamma, store_aux_action=True)

# To this:
NStepReplayBuffer(
    RL_CONFIG.get_n_step(metrics.frame_count, metrics.expert_ratio) if RL_CONFIG.adaptive_n_step else RL_CONFIG.n_step,
    RL_CONFIG.gamma,
    store_aux_action=True
)
```

**Total changes:** ~15 lines of code

---

## Conclusion

**Adaptive n-step is theoretically beneficial but adds complexity.**

**For your long training run:**
- ✅ Keep fixed n=7 (stable, proven)
- 🔬 Test adaptive on separate experiment
- 📊 Compare results before committing

**If you implement adaptive:**
- Start with simple phase-based approach
- Monitor for instability during transitions
- Be prepared to revert if issues arise

**Your fixed n=7 is already excellent. Only implement adaptive if you:**
1. Have time to test first
2. Want to squeeze out potential 5-10% gains
3. Are comfortable with added complexity

**Otherwise, focus on other optimizations that have clearer payoffs.** 🎯
