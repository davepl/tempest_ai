#!/usr/bin/env python3
"""
Lightweight n-step return preprocessor used by DQNAgent.
This module is intentionally dependency-light to make unit tests fast.
"""
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
    """
    Sliding-window n-step return preprocessor.
    - add(s, a, r, s_next, done) returns a list of 0..n experiences to push into the main replay buffer.
    - On each step (non-terminal), when we have at least n items, we emit exactly one matured experience.
    - On terminal, we flush the remaining tail so no transitions are lost across episode boundaries.
    Contract:
      Input: (state, action, reward, next_state, done)
    Output: List[Tuple(state, action, R_n, next_state_n, done_n, actor)]
    """
    def __init__(self, n_step: int, gamma: float):
        assert n_step >= 1
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self._deque: Deque[_PendingStep] = deque()

    def reset(self):
        self._deque.clear()

    def _make_experience_from_start(self):
        R = 0.0
        priority_R = 0.0
        done_flag = False
        last_next_state = None
        first = self._deque[0]
        s0 = first.state
        a0 = first.action
        start_actor = first.actor
        steps_used = 0

        for i in range(self.n_step):
            if i >= len(self._deque):
                break
            step = self._deque[i]
            if i > 0 and step.actor != start_actor:
                break
            R += (self.gamma ** i) * float(step.reward)
            priority_R += (self.gamma ** i) * float(step.priority_reward)
            last_next_state = step.next_state
            steps_used = i + 1
            if step.done:
                done_flag = True
                break

        # Ensure we always report at least one step consumed
        if steps_used <= 0:
            steps_used = 1

        assert last_next_state is not None
        return (s0, a0, R, priority_R, last_next_state, done_flag, steps_used, start_actor)

    def _should_emit(self) -> bool:
        if not self._deque:
            return False

        if len(self._deque) >= self.n_step:
            return True

        start_actor = self._deque[0].actor
        max_len = min(self.n_step, len(self._deque))
        for i in range(max_len):
            step = self._deque[i]
            if i > 0 and step.actor != start_actor:
                return True
            if step.done:
                return True
        return False

    def add(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        actor: Optional[str] = None,
        priority_reward: Optional[float] = None,
    ):
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

        if actor is None:
            raise ValueError("NStepReplayBuffer.add requires an explicit actor tag")
        actor_tag = str(actor).strip().lower()
        if not actor_tag:
            raise ValueError("NStepReplayBuffer.add received blank actor tag")
        if actor_tag in ('unknown', 'none', 'random'):
            raise ValueError(f"NStepReplayBuffer.add received invalid actor tag '{actor_tag}'")
        priority_val = float(priority_reward) if priority_reward is not None else float(reward)

        self._deque.append(
            _PendingStep(
                state=state,
                action=a_idx,
                reward=float(reward),
                priority_reward=priority_val,
                next_state=next_state,
                done=bool(done),
                actor=actor_tag,
            )
        )

        outputs: List[Tuple] = []

        if done:
            while self._deque:
                outputs.append(self._make_experience_from_start())
                self._deque.popleft()
            return outputs

        while self._should_emit():
            outputs.append(self._make_experience_from_start())
            self._deque.popleft()

        return outputs
