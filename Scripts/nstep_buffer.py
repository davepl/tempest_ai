#!/usr/bin/env python3
# ==================================================================================================================
# ||  TEMPEST AI v2 • N-STEP RETURN BUFFER                                                                       ||
# ||  Lightweight sliding-window preprocessor (unchanged interface for socket_server).                            ||
# ==================================================================================================================
"""N-step return preprocessor. Dependency-light for fast unit tests."""

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, List, Optional, Tuple
import numpy as np


@dataclass
class _PendingStep:
    state: Any
    action: int
    reward: float
    priority_reward: float
    next_state: Any
    done: bool
    actor: str


class NStepReplayBuffer:
    """Sliding-window n-step return preprocessor.

    add() returns a list of matured experiences to push into the main replay buffer.
    On terminal, flushes the remaining tail so no transitions are lost.
    Actor labels are retained for the first action but do not truncate return rollout.
    """

    def __init__(self, n_step: int, gamma: float):
        assert n_step >= 1
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self._deque: Deque[_PendingStep] = deque()

    def reset(self):
        self._deque.clear()

    def _make_experience(self):
        R = 0.0
        priority_R = 0.0
        done_flag = False
        last_next = None
        first = self._deque[0]
        s0, a0, actor0 = first.state, first.action, first.actor
        steps = 0

        for i in range(min(self.n_step, len(self._deque))):
            step = self._deque[i]
            R += (self.gamma ** i) * step.reward
            priority_R += (self.gamma ** i) * step.priority_reward
            last_next = step.next_state
            steps = i + 1
            if step.done:
                done_flag = True
                break

        return (s0, a0, R, priority_R, last_next, done_flag, max(1, steps), actor0)

    def _should_emit(self) -> bool:
        if not self._deque:
            return False
        if len(self._deque) >= self.n_step:
            return True
        for i in range(min(self.n_step, len(self._deque))):
            step = self._deque[i]
            if step.done:
                return True
        return False

    def add(self, state, action, reward, next_state, done,
            actor: Optional[str] = None, priority_reward: Optional[float] = None):
        pr = priority_reward if priority_reward is not None else reward
        self._deque.append(_PendingStep(state, int(action), float(reward),
                                         float(pr), next_state, bool(done),
                                         actor or "dqn"))
        results = []
        while self._should_emit():
            results.append(self._make_experience())
            self._deque.popleft()
        # Flush on terminal
        if done:
            while self._deque:
                results.append(self._make_experience())
                self._deque.popleft()
        return results
