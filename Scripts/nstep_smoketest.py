#!/usr/bin/env python3
"""
Quick smoke tests for NStepReplayBuffer without requiring the full training stack.
Validates:
- Discounted n-step return R_n computation
- Horizon (steps_used) values
- Actor boundary split (no cross-actor aggregation)
- Terminal flush behavior (no transitions lost on done)

Run: python Scripts/nstep_smoketest.py
Expected: All tests print PASS with computed values.
"""
from __future__ import annotations
import math
from typing import List, Tuple

from nstep_buffer import NStepReplayBuffer


def approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


def compute_expected_rn(rewards: List[float], gamma: float) -> float:
    return sum((gamma ** i) * r for i, r in enumerate(rewards))


def case_simple_sequence() -> None:
    gamma = 0.99
    n = 3
    buf = NStepReplayBuffer(n_step=n, gamma=gamma, store_aux_action=True)

    # Feed a simple 5-step sequence with constant rewards, single actor
    outs = []
    # state is s[i], action i, aux_action = i * 0.1 for traceability
    for i in range(5):
        res = buf.add(
            state=f"s{i}",
            action=i,
            reward=1.0,
            next_state=f"s{i+1}",
            done=False,
            aux_action=i * 0.1,
            actor="dqn",
        )
        outs.extend(res)

    # No terminal yet -> expect (len - n + 1) matured items = 5 - 3 + 1 = 3
    assert len(outs) == 3, f"expected 3 matured, got {len(outs)}"
    # First output should aggregate steps [0,1,2]
    s0, a0, aux0, R0, s3, done0, h0, actor0, *rest = outs[0]
    exp_R0 = compute_expected_rn([1.0, 1.0, 1.0], gamma)
    assert s0 == "s0" and a0 == 0 and approx(R0, exp_R0) and s3 == "s3" and not done0 and h0 == 3 and actor0 == "dqn"
    # Second output should aggregate steps [1,2,3]
    s1, a1, aux1, R1, s4, done1, h1, actor1, *rest = outs[1]
    exp_R1 = compute_expected_rn([1.0, 1.0, 1.0], gamma)
    assert s1 == "s1" and a1 == 1 and approx(R1, exp_R1) and s4 == "s4" and not done1 and h1 == 3 and actor1 == "dqn"
    # Third output should aggregate steps [2,3,4]
    s2, a2, aux2, R2, s5, done2, h2, actor2, *rest = outs[2]
    exp_R2 = compute_expected_rn([1.0, 1.0, 1.0], gamma)
    assert s2 == "s2" and a2 == 2 and approx(R2, exp_R2) and s5 == "s5" and not done2 and h2 == 3 and actor2 == "dqn"


def case_terminal_flush() -> None:
    gamma = 0.9
    n = 3
    buf = NStepReplayBuffer(n_step=n, gamma=gamma, store_aux_action=False)

    # Sequence ends with done at step 2
    outs = []
    outs.extend(buf.add("s0", 0, 2.0, "s1", False, actor="expert"))
    outs.extend(buf.add("s1", 1, 2.0, "s2", False, actor="expert"))
    outs.extend(buf.add("s2", 2, 2.0, "s3", True, actor="expert"))  # terminal -> flush remaining

    # With done at i=2, we should flush 3 experiences: start at 0, then 1, then 2
    assert len(outs) == 3, f"expected 3 outputs on terminal flush, got {len(outs)}"
    # First: rewards [2,2,2], done True at horizon 3
    s0, a0, R0, s3, d0, h0, actor0, *rest = outs[0]
    assert d0 and h0 == 3 and approx(R0, compute_expected_rn([2,2,2], gamma))
    # Second: start at 1 -> rewards [2,2], done True, horizon 2
    s1, a1, R1, s3b, d1, h1, actor1, *rest = outs[1]
    assert d1 and h1 == 2 and approx(R1, compute_expected_rn([2,2], gamma)) and s3b == "s3"
    # Third: start at 2 -> rewards [2], done True, horizon 1
    s2, a2, R2, s3c, d2, h2, actor2, *rest = outs[2]
    assert d2 and h2 == 1 and approx(R2, 2.0) and s3c == "s3"


def case_actor_boundary_split() -> None:
    gamma = 0.95
    n = 3
    buf = NStepReplayBuffer(n_step=n, gamma=gamma, store_aux_action=True)

    outs = []
    # Step 0: DQN
    outs.extend(buf.add("s0", 0, 1.0, "s1", False, aux_action=0.0, actor="dqn"))
    # Step 1: EXPERT (boundary)
    outs.extend(buf.add("s1", 1, 1.0, "s2", False, aux_action=0.1, actor="expert"))
    # Step 2: EXPERT
    outs.extend(buf.add("s2", 2, 1.0, "s3", False, aux_action=0.2, actor="expert"))
    # Step 3: EXPERT
    outs.extend(buf.add("s3", 3, 1.0, "s4", False, aux_action=0.3, actor="expert"))

    # On boundary at step 1, the first output should have been emitted already for start=0 (only first step due to actor split)
    assert len(outs) >= 1
    # First output corresponds to start at 0 with horizon 1, next_state s1, done False
    s0, a0, aux0, R0, sn, d0, h0, actor0, *rest = outs[0]
    assert s0 == "s0" and a0 == 0 and aux0 == 0.0 and h0 == 1 and not d0 and sn == "s1" and approx(R0, 1.0)

    # After the sequence, there should be enough EXPERT steps to emit at least one matured of horizon 3
    # The remaining outputs should belong to actor "expert"
    for tup in outs[1:]:
        actor_idx = 7 if len(tup) >= 8 else -1
        assert tup[actor_idx] == "expert", f"unexpected actor in subsequent outputs: {tup[actor_idx]}"

    # Final check: ensure one matured with horizon>=2 exists for expert
    horizons = [t[6] for t in outs[1:]]
    assert any(h >= 2 for h in horizons), f"expected at least one expert horizon >=2, got {horizons}"


def main():
    case_simple_sequence()
    print("[PASS] case_simple_sequence")
    case_terminal_flush()
    print("[PASS] case_terminal_flush")
    case_actor_boundary_split()
    print("[PASS] case_actor_boundary_split")
    print("All NStep smoke tests passed.")


if __name__ == "__main__":
    main()
