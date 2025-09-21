#!/usr/bin/env python3
"""
Lightweight n-step return preprocessor used by DQNAgent.
This module is intentionally dependency-light to make unit tests fast.
"""
from collections import deque
from typing import Deque, List, Tuple
import numpy as np


class NStepReplayBuffer:
    """
    Sliding-window n-step return preprocessor.
    - add(s, a, r, s_next, done) returns a list of 0..n experiences to push into the main replay buffer.
    - On each step (non-terminal), when we have at least n items, we emit exactly one matured experience.
    - On terminal, we flush the remaining tail so no transitions are lost across episode boundaries.
    Contract:
      Input: (state, action, reward, next_state, done)
      Output: List[Tuple(state, action, R_n, next_state_n, done_n)]
    """
    def __init__(self, n_step: int, gamma: float):
        assert n_step >= 1
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self._deque: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque()

    def reset(self):
        self._deque.clear()

    def _make_experience_from_start(self) -> Tuple[np.ndarray, int, float, np.ndarray, bool]:
        R = 0.0
        done_flag = False
        last_next_state = None

        s0, a0, _, _, _ = self._deque[0]

        for i in range(self.n_step):
            if i >= len(self._deque):
                break
            _, _, r, ns, d = self._deque[i]
            R += (self.gamma ** i) * float(r)
            last_next_state = ns
            if d:
                done_flag = True
                break

        assert last_next_state is not None
        return (s0, a0, R, last_next_state, done_flag)

    def add(self, state, action, reward, next_state, done) -> List[Tuple[np.ndarray, int, float, np.ndarray, bool]]:
        # Normalize action to int
        try:
            if isinstance(action, np.ndarray):
                a_idx = int(action.reshape(-1)[0])
            elif isinstance(action, (list, tuple)):
                a_idx = int(action[0])
            else:
                a_idx = int(action)
        except Exception:
            a_idx = int(action)

        self._deque.append((state, a_idx, float(reward), next_state, bool(done)))

        outputs: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = []

        if not done:
            if len(self._deque) >= self.n_step:
                outputs.append(self._make_experience_from_start())
                self._deque.popleft()
        else:
            while len(self._deque) > 0:
                outputs.append(self._make_experience_from_start())
                self._deque.popleft()

        return outputs
