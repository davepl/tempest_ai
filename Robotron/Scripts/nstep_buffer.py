#!/usr/bin/env python3
# ==================================================================================================================
# ||  ROBOTRON AI v2 • N-STEP RETURN BUFFER                                                                      ||
# ||  Lightweight sliding-window preprocessor (unchanged interface for socket_server).                            ||
# ==================================================================================================================
"""N-step return preprocessor. Dependency-light for fast unit tests."""

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, List, Optional, Tuple


@dataclass
class _PendingStep:
    state: Any
    action: int
    reward: float
    priority_reward: float
    next_state: Any
    done: bool
    actor: str
    wave_number: int
    start_wave: int


class NStepReplayBuffer:
    """Sliding-window n-step return preprocessor.

    add() returns a list of matured experiences to push into the main replay buffer.
    On terminal, flushes the remaining tail so no transitions are lost.
    """

    def __init__(self, n_step: int, gamma: float, cross_actor: bool = False):
        assert n_step >= 1
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self.cross_actor = bool(cross_actor)
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
        wave0, start0 = first.wave_number, first.start_wave
        steps = 0

        for i in range(min(self.n_step, len(self._deque))):
            step = self._deque[i]
            # Optionally truncate returns at actor (expert/DQN) boundaries.
            if not self.cross_actor and i > 0 and step.actor != actor0:
                break
            R += (self.gamma ** i) * step.reward
            priority_R += (self.gamma ** i) * step.priority_reward
            last_next = step.next_state
            steps = i + 1
            if step.done:
                done_flag = True
                break

        return (s0, a0, R, priority_R, last_next, done_flag, max(1, steps), actor0, wave0, start0)

    def _should_emit(self) -> bool:
        if not self._deque:
            return False
        actor0 = self._deque[0].actor
        for i in range(min(self.n_step, len(self._deque))):
            step = self._deque[i]
            if not self.cross_actor and i > 0 and step.actor != actor0:
                return True
            if step.done:
                return True
        return len(self._deque) >= self.n_step

    def add(self, state, action, reward, next_state, done,
            actor: Optional[str] = None, priority_reward: Optional[float] = None,
            wave_number: int = 1, start_wave: int = 1):
        pr = priority_reward if priority_reward is not None else reward
        self._deque.append(_PendingStep(state, int(action), float(reward),
                                         float(pr), next_state, bool(done),
                                         actor or "dqn",
                                         max(1, int(wave_number)),
                                         max(1, int(start_wave))))
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
