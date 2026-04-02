#!/usr/bin/env python3
"""Robotron AI v3 — Rollout buffer for PPO.

On-policy PPO requires storing complete rollouts from actors, then
computing GAE advantages and discounted returns before each update.
This buffer collects transitions from multiple parallel actors and
provides mini-batch iteration for PPO epochs.
"""

import torch
import numpy as np
from typing import Optional, Iterator
from .config import CONFIG


class RolloutBuffer:
    """Fixed-size rollout storage for PPO with GAE.

    Stores rollout_length steps from num_actors parallel environments.
    After filling, computes advantages via GAE and provides shuffled
    mini-batch iteration for PPO epochs.

    Storage layout: all tensors are (num_steps, num_actors, ...) so
    actor-parallel operations are contiguous in memory.
    """

    def __init__(
        self,
        rollout_length: int = None,
        num_actors: int = None,
        max_entities: int = None,
        entity_feature_dim: int = None,
        global_context_dim: int = None,
        frame_stack: int = None,
        gamma: float = None,
        gae_lambda: float = None,
        device: torch.device = None,
    ):
        cfg = CONFIG.model
        tcfg = CONFIG.train

        self.rollout_length = rollout_length or tcfg.rollout_length
        self.num_actors = num_actors or tcfg.num_actors
        self.max_entities = max_entities or cfg.max_entities
        self.entity_feature_dim = entity_feature_dim or cfg.entity_feature_dim
        self.global_context_dim = global_context_dim or cfg.global_context_dim
        self.frame_stack = frame_stack or cfg.frame_stack
        self.gamma = gamma or tcfg.gamma
        self.gae_lambda = gae_lambda or tcfg.gae_lambda
        self.device = device or torch.device("cpu")

        T = self.rollout_length
        A = self.num_actors
        N = self.max_entities
        F = self.entity_feature_dim
        G = self.global_context_dim
        FS = self.frame_stack

        # Observations: per-frame entity set + global context
        # Stored as stacked frames: (T, A, FS, N, F) and (T, A, FS, G)
        self.entity_features = torch.zeros(T, A, FS, N, F, device=self.device)
        self.entity_masks = torch.ones(T, A, FS, N, dtype=torch.bool, device=self.device)
        self.global_contexts = torch.zeros(T, A, FS, G, device=self.device)

        # Actions
        self.move_actions = torch.zeros(T, A, dtype=torch.long, device=self.device)
        self.fire_actions = torch.zeros(T, A, dtype=torch.long, device=self.device)

        # PPO data
        self.log_probs = torch.zeros(T, A, device=self.device)
        self.values = torch.zeros(T, A, device=self.device)
        self.has_value = torch.zeros(T, A, dtype=torch.bool, device=self.device)
        self.rewards = torch.zeros(T, A, device=self.device)
        self.dones = torch.zeros(T, A, dtype=torch.bool, device=self.device)

        # Computed after rollout
        self.advantages = torch.zeros(T, A, device=self.device)
        self.returns = torch.zeros(T, A, device=self.device)

        # Expert BC targets (if expert action was available)
        self.expert_move = torch.zeros(T, A, dtype=torch.long, device=self.device)
        self.expert_fire = torch.zeros(T, A, dtype=torch.long, device=self.device)
        self.is_expert = torch.zeros(T, A, dtype=torch.bool, device=self.device)
        self.policy_sampled = torch.zeros(T, A, dtype=torch.bool, device=self.device)
        self.fire_locked = torch.zeros(T, A, dtype=torch.bool, device=self.device)

        self.step = 0
        self.ready = False

    def reset(self):
        """Reset buffer for next rollout collection."""
        self.step = 0
        self.ready = False

    def add(
        self,
        actor_id: int,
        entity_features: torch.Tensor,     # (FS, N, F)
        entity_mask: torch.Tensor,          # (FS, N)
        global_context: torch.Tensor,       # (FS, G)
        move_action: int,
        fire_action: int,
        log_prob: float,
        value: float,
        has_value: bool,
        reward: float,
        done: bool,
        expert_move: int = 8,
        expert_fire: int = 8,
        is_expert: bool = False,
        policy_sampled: bool = False,
        fire_locked: bool = False,
    ):
        """Add a single step for one actor."""
        t = self.step
        a = actor_id

        self.entity_features[t, a] = entity_features
        self.entity_masks[t, a] = entity_mask
        self.global_contexts[t, a] = global_context
        self.move_actions[t, a] = move_action
        self.fire_actions[t, a] = fire_action
        self.log_probs[t, a] = log_prob
        self.values[t, a] = value
        self.has_value[t, a] = has_value
        self.rewards[t, a] = reward
        self.dones[t, a] = done
        self.expert_move[t, a] = expert_move
        self.expert_fire[t, a] = expert_fire
        self.is_expert[t, a] = is_expert
        self.policy_sampled[t, a] = policy_sampled
        self.fire_locked[t, a] = fire_locked

    def advance(self):
        """Advance the step counter after all actors have added their step."""
        self.step += 1
        if self.step >= self.rollout_length:
            self.ready = True

    def compute_gae(self, last_values: torch.Tensor):
        """Compute GAE advantages and discounted returns.

        Args:
            last_values: (num_actors,) value estimates for the state
                         AFTER the last step in the rollout (bootstrap).
        """
        T = self.rollout_length
        gamma = self.gamma
        lam = self.gae_lambda

        last_gae = torch.zeros(self.num_actors, device=self.device)
        next_values = last_values

        for t in reversed(range(T)):
            non_terminal = (~self.dones[t]).float()
            delta = self.rewards[t] + gamma * next_values * non_terminal - self.values[t]
            last_gae = delta + gamma * lam * non_terminal * last_gae
            self.advantages[t] = last_gae
            next_values = self.values[t]

        self.returns = self.advantages + self.values

    def _flatten(self) -> dict[str, torch.Tensor]:
        """Flatten (T, A, ...) → (T*A, ...) for mini-batch sampling."""
        T, A = self.rollout_length, self.num_actors
        N = T * A
        return {
            "entity_features": self.entity_features.reshape(N, self.frame_stack, self.max_entities, self.entity_feature_dim),
            "entity_masks": self.entity_masks.reshape(N, self.frame_stack, self.max_entities),
            "global_contexts": self.global_contexts.reshape(N, self.frame_stack, self.global_context_dim),
            "move_actions": self.move_actions.reshape(N),
            "fire_actions": self.fire_actions.reshape(N),
            "log_probs": self.log_probs.reshape(N),
            "values": self.values.reshape(N),
            "has_value": self.has_value.reshape(N),
            "advantages": self.advantages.reshape(N),
            "returns": self.returns.reshape(N),
            "expert_move": self.expert_move.reshape(N),
            "expert_fire": self.expert_fire.reshape(N),
            "is_expert": self.is_expert.reshape(N),
            "policy_sampled": self.policy_sampled.reshape(N),
            "fire_locked": self.fire_locked.reshape(N),
        }

    def iterate_minibatches(
        self,
        mini_batch_size: int = None,
        num_epochs: int = 1,
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Yield shuffled mini-batches for PPO epochs.

        Generates num_epochs passes over the full rollout, each time
        shuffling and splitting into mini-batches.
        """
        mbs = mini_batch_size or CONFIG.train.mini_batch_size
        flat = self._flatten()
        N = flat["move_actions"].shape[0]

        for _ in range(num_epochs):
            indices = torch.randperm(N, device=self.device)
            for start in range(0, N, mbs):
                end = min(start + mbs, N)
                idx = indices[start:end]
                yield {k: v[idx] for k, v in flat.items()}
