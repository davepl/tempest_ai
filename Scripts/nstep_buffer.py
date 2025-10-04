#!/usr/bin/env python3
"""
Lightweight n-step return preprocessor used by DQNAgent.
This module is intentionally dependency-light to make unit tests fast.
"""
from collections import deque
from typing import List, Tuple
import numpy as np


class NStepReplayBuffer:
    """
    Sliding-window n-step return preprocessor.

    Features:
    - Always accumulates the primary reward plus optional subjective/objective components.
    - Returns matured experiences containing the discounted totals for each component.
    - Supports optional auxiliary action storage (e.g., spinner head index).

    Contract:
        Input: add(state, action, reward, next_state, done, *, subjreward=None, objreward=None, aux_action=None)
        Output when store_aux_action is False:
            (state, action, R_total, R_subj, R_obj, next_state_n, done_n)
        Output when store_aux_action is True:
            (state, action, aux_action, R_total, R_subj, R_obj, next_state_n, done_n)
    """

    def __init__(self, n_step: int, gamma: float, store_aux_action: bool = False, track_reward_components: bool = True):
        assert n_step >= 1
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        # When store_aux_action=True, we will store an extra per-step auxiliary action
        # (e.g., a discrete spinner action) alongside the discrete action and return it in outputs.
        self.store_aux_action = bool(store_aux_action)
        self.track_reward_components = bool(track_reward_components)
        self._deque = deque()

    def reset(self):
        self._deque.clear()

    def _make_experience_from_start(self):
        R = 0.0
        R_subj = 0.0
        R_obj = 0.0
        done_flag = False
        last_next_state = None
        if self.store_aux_action:
            s0, a0, aux0, _, _, _, _, _ = self._deque[0]
        else:
            s0, a0, _, _, _, _, _ = self._deque[0]

        for i in range(self.n_step):
            if i >= len(self._deque):
                break
            if self.store_aux_action:
                _, _, _, r, r_subj, r_obj, ns, d = self._deque[i]
            else:
                _, _, r, r_subj, r_obj, ns, d = self._deque[i]
            discount = (self.gamma ** i)
            R += discount * float(r)
            R_subj += discount * float(r_subj)
            R_obj += discount * float(r_obj)
            last_next_state = ns
            if d:
                done_flag = True
                break

        assert last_next_state is not None
        if self.store_aux_action:
            if self.track_reward_components:
                return (s0, a0, aux0, R, R_subj, R_obj, last_next_state, done_flag)
            return (s0, a0, aux0, R, last_next_state, done_flag)
        if self.track_reward_components:
            return (s0, a0, R, R_subj, R_obj, last_next_state, done_flag)
        return (s0, a0, R, last_next_state, done_flag)

    def add(self, state, action, reward, next_state, done, *, subjreward=None, objreward=None, aux_action=None):
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

        subj_val = float(reward if subjreward is None else subjreward)
        obj_val = float(reward if objreward is None else objreward)

        if self.store_aux_action:
            self._deque.append((state, a_idx, float(aux_action) if aux_action is not None else 0.0,
                                float(reward), subj_val, obj_val, next_state, bool(done)))
        else:
            self._deque.append((state, a_idx, float(reward), subj_val, obj_val, next_state, bool(done)))

        outputs: List[Tuple] = []

        if not done:
            if len(self._deque) >= self.n_step:
                outputs.append(self._make_experience_from_start())
                self._deque.popleft()
        else:
            while len(self._deque) > 0:
                outputs.append(self._make_experience_from_start())
                self._deque.popleft()

        return outputs
