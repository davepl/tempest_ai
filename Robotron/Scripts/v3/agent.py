#!/usr/bin/env python3
"""Robotron AI v3 — PPO Agent.

High-level agent wrapping the Set Transformer network, PPO training,
expert system, and state processing. Provides the same external API
that the socket server expects: act(), step(), save(), load().
"""

import os
import time
import math
import random
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from collections import deque
from pathlib import Path

from .config import CONFIG, MODEL_DIR, CHECKPOINT_PATH, GAME_SETTINGS
from .model import RobotronPPONet
from .state_processor import StateProcessor, extract_entities, extract_global_context
from .expert import PotentialFieldExpert, get_expert_action
from .reward import shape_reward
from .rollout_buffer import RolloutBuffer


class PPOAgent:
    """PPO agent for Robotron with Set Transformer.

    Manages:
      - Network (policy + value + auxiliary heads)
      - Per-client frame stacking
      - Action selection (policy / expert / epsilon)
      - PPO training loop
      - Checkpoint save/load
    """

    def __init__(self, device: str = "auto"):
        cfg = CONFIG.model
        tcfg = CONFIG.train

        # Device selection
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Network
        self.net = RobotronPPONet(
            entity_feature_dim=cfg.entity_feature_dim,
            max_entities=cfg.max_entities,
            embed_dim=cfg.embed_dim,
            num_isab_layers=cfg.num_isab_layers,
            num_heads=cfg.num_heads,
            num_inducing=cfg.num_inducing_points,
            global_context_dim=cfg.global_context_dim,
            frame_stack=cfg.frame_stack,
            fusion_hidden=cfg.fusion_hidden,
            fusion_layers=cfg.fusion_layers,
            num_move_actions=cfg.num_move_actions,
            num_fire_actions=cfg.num_fire_actions,
            use_auxiliary_head=cfg.use_auxiliary_head,
            auxiliary_predict_steps=cfg.auxiliary_predict_steps,
            dropout=cfg.dropout,
        ).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=tcfg.lr,
            eps=tcfg.adam_eps,
            weight_decay=tcfg.weight_decay,
        )

        # LR scheduler: linear warmup then cosine decay
        self.lr_scheduler = None  # built on first training step
        self._training_steps = 0

        # State processor
        self.state_processor = StateProcessor()
        self.expert = PotentialFieldExpert()

        # Per-client frame buffers: client_id → deque of processed frames
        self._frame_buffers: dict[int, deque] = {}
        self._buffer_lock = threading.Lock()

        # Rollout buffer
        self.rollout = RolloutBuffer(device=self.device)

        # Training state
        self.total_frames = 0
        self.last_loss = 0.0
        self.last_policy_loss = 0.0
        self.last_value_loss = 0.0
        self.last_entropy = 0.0
        self.last_bc_loss = 0.0
        self.last_grad_norm = 0.0
        self._save_lock = threading.Lock()

        print(f"PPO Agent initialized on {self.device}")
        print(f"  Network params: {sum(p.numel() for p in self.net.parameters()):,}")
        print(f"  Entity feature dim: {cfg.entity_feature_dim}")
        print(f"  Max entities: {cfg.max_entities}")
        print(f"  Embed dim: {cfg.embed_dim}")
        print(f"  ISAB layers: {cfg.num_isab_layers}")
        print(f"  Inducing points: {cfg.num_inducing_points}")
        print(f"  Frame stack: {cfg.frame_stack}")

    # ── Frame processing ────────────────────────────────────────────────

    def _get_frame_buffer(self, client_id: int) -> deque:
        with self._buffer_lock:
            if client_id not in self._frame_buffers:
                self._frame_buffers[client_id] = deque(maxlen=CONFIG.model.frame_stack)
            return self._frame_buffers[client_id]

    def _reset_frame_buffer(self, client_id: int):
        with self._buffer_lock:
            if client_id in self._frame_buffers:
                self._frame_buffers[client_id].clear()

    def _process_and_stack(
        self,
        wire_state: np.ndarray,
        client_id: int,
    ) -> dict[str, torch.Tensor]:
        """Process wire state and return stacked tensors ready for the network."""
        buf = self._get_frame_buffer(client_id)

        # Process current frame
        frame = self.state_processor.process_frame(wire_state)
        buf.append(frame)

        # Pad with copies of first frame if buffer isn't full
        while len(buf) < CONFIG.model.frame_stack:
            buf.appendleft(frame.copy() if isinstance(frame, dict) else frame)

        # Stack frames
        stacked = self.state_processor.stack_frames(list(buf))

        # Convert to tensors with batch dim
        tensors = self.state_processor.to_tensors(stacked, self.device)
        return {k: v.unsqueeze(0) for k, v in tensors.items()}  # add batch dim

    # ── Action selection ────────────────────────────────────────────────

    @torch.no_grad()
    def act(
        self,
        wire_state: np.ndarray,
        epsilon: float = 0.0,
        client_id: int = 0,
        locked_fire: Optional[int] = None,
    ) -> tuple[int, int, bool]:
        """Select action for one frame.

        Returns: (move_dir, fire_dir, is_epsilon)
          - move_dir: 0-8 (8=idle)
          - fire_dir: 0-8 (8=idle)
          - is_epsilon: True if action was random
        """
        # Epsilon-greedy exploration
        if random.random() < epsilon:
            move = random.randrange(CONFIG.model.num_move_actions)
            fire = random.randrange(CONFIG.model.num_fire_actions)
            if locked_fire is not None:
                fire = locked_fire
            return move, fire, True

        # Policy action
        tensors = self._process_and_stack(wire_state, client_id)

        self.net.eval()
        out = self.net(
            tensors["entity_features"],
            tensors["entity_mask"],
            tensors["global_context"],
        )

        move_probs = F.softmax(out["move_logits"][0], dim=-1)
        fire_probs = F.softmax(out["fire_logits"][0], dim=-1)

        move = torch.multinomial(move_probs, 1).item()

        if locked_fire is not None:
            fire = locked_fire
        else:
            fire = torch.multinomial(fire_probs, 1).item()

        return int(move), int(fire), False

    @torch.no_grad()
    def act_greedy(
        self,
        wire_state: np.ndarray,
        client_id: int = 0,
        locked_fire: Optional[int] = None,
    ) -> tuple[int, int]:
        """Greedy (argmax) action selection for evaluation."""
        tensors = self._process_and_stack(wire_state, client_id)

        self.net.eval()
        out = self.net(
            tensors["entity_features"],
            tensors["entity_mask"],
            tensors["global_context"],
        )

        move = out["move_logits"][0].argmax().item()
        fire = out["fire_logits"][0].argmax().item() if locked_fire is None else locked_fire

        return int(move), int(fire)

    @torch.no_grad()
    def act_with_value(
        self,
        wire_state: np.ndarray,
        epsilon: float = 0.0,
        client_id: int = 0,
        locked_fire: Optional[int] = None,
    ) -> tuple[int, int, float, float, bool, dict]:
        """Select action and return value estimate + log_prob for rollout storage.

        Returns: (move, fire, log_prob, value, is_epsilon, tensors_dict)
        """
        if random.random() < epsilon:
            move = random.randrange(CONFIG.model.num_move_actions)
            fire = random.randrange(CONFIG.model.num_fire_actions)
            if locked_fire is not None:
                fire = locked_fire
            # Still compute value for GAE even on random actions
            tensors = self._process_and_stack(wire_state, client_id)
            self.net.eval()
            value = self.net.get_value(
                tensors["entity_features"],
                tensors["entity_mask"],
                tensors["global_context"],
            ).item()
            return move, fire, 0.0, value, True, self._detach_tensors(tensors)

        tensors = self._process_and_stack(wire_state, client_id)

        self.net.eval()
        move_a, fire_a, log_prob, entropy, value = self.net.get_action_and_value(
            tensors["entity_features"],
            tensors["entity_mask"],
            tensors["global_context"],
        )

        move = move_a[0].item()
        fire = fire_a[0].item() if locked_fire is None else locked_fire
        lp = log_prob[0].item()
        val = value[0].item()

        return int(move), int(fire), lp, val, False, self._detach_tensors(tensors)

    def _detach_tensors(self, tensors: dict) -> dict:
        """Move tensors to CPU for storage in rollout buffer."""
        return {k: v.squeeze(0).cpu() for k, v in tensors.items()}

    # ── Training ────────────────────────────────────────────────────────

    def train_step(self, rollout: RolloutBuffer) -> dict[str, float]:
        """Run PPO update on a filled rollout buffer.

        Returns dict of loss components for logging.
        """
        tcfg = CONFIG.train
        self.net.train()

        # Build LR scheduler on first step
        if self.lr_scheduler is None:
            self.lr_scheduler = self._build_lr_scheduler()

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        total_bc_loss = 0.0
        total_aux_loss = 0.0
        num_batches = 0

        for batch in rollout.iterate_minibatches(
            mini_batch_size=tcfg.mini_batch_size,
            num_epochs=tcfg.num_epochs,
        ):
            # Move batch to device
            entity_features = batch["entity_features"].to(self.device)
            entity_masks = batch["entity_masks"].to(self.device)
            global_contexts = batch["global_contexts"].to(self.device)
            old_move = batch["move_actions"].to(self.device)
            old_fire = batch["fire_actions"].to(self.device)
            old_log_probs = batch["log_probs"].to(self.device)
            advantages = batch["advantages"].to(self.device)
            returns = batch["returns"].to(self.device)
            expert_move = batch["expert_move"].to(self.device)
            expert_fire = batch["expert_fire"].to(self.device)
            is_expert = batch["is_expert"].to(self.device)

            # Normalize advantages
            adv_mean = advantages.mean()
            adv_std = advantages.std() + 1e-8
            advantages = (advantages - adv_mean) / adv_std

            # Forward pass: evaluate old actions
            _, _, new_log_probs, entropy, new_values = self.net.get_action_and_value(
                entity_features, entity_masks, global_contexts,
                move_action=old_move, fire_action=old_fire,
            )

            # PPO clipped objective
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - tcfg.clip_epsilon, 1.0 + tcfg.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss (clipped)
            value_pred = new_values
            value_clipped = batch["values"].to(self.device) + torch.clamp(
                value_pred - batch["values"].to(self.device),
                -tcfg.clip_value, tcfg.clip_value,
            )
            value_loss1 = F.mse_loss(value_pred, returns)
            value_loss2 = F.mse_loss(value_clipped, returns)
            value_loss = torch.max(value_loss1, value_loss2)

            # Entropy bonus
            entropy_loss = -entropy.mean()

            # Behavioral cloning loss (if expert demonstrations present)
            bc_loss = torch.tensor(0.0, device=self.device)
            has_expert = is_expert.any()
            bc_weight = self._get_bc_weight()
            if has_expert and bc_weight > 0:
                out = self.net(entity_features[is_expert], entity_masks[is_expert], global_contexts[is_expert])
                bc_move_loss = F.cross_entropy(out["move_logits"], expert_move[is_expert])
                bc_fire_loss = F.cross_entropy(out["fire_logits"], expert_fire[is_expert])
                bc_loss = bc_move_loss + bc_fire_loss

            # Total loss
            loss = (
                policy_loss
                + tcfg.value_coeff * value_loss
                + tcfg.entropy_coeff * entropy_loss
                + bc_weight * bc_loss
            )

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            grad_norm = nn.utils.clip_grad_norm_(
                self.net.parameters(), tcfg.max_grad_norm
            )

            # Skip if non-finite
            if not torch.isfinite(torch.tensor(loss.item())):
                continue

            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            self._training_steps += 1
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy_loss += entropy_loss.item()
            total_bc_loss += bc_loss.item()
            num_batches += 1

        if num_batches > 0:
            self.last_policy_loss = total_policy_loss / num_batches
            self.last_value_loss = total_value_loss / num_batches
            self.last_entropy = -total_entropy_loss / num_batches
            self.last_bc_loss = total_bc_loss / num_batches
            self.last_loss = (
                self.last_policy_loss
                + CONFIG.train.value_coeff * self.last_value_loss
            )
            self.last_grad_norm = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

        return {
            "policy_loss": self.last_policy_loss,
            "value_loss": self.last_value_loss,
            "entropy": self.last_entropy,
            "bc_loss": self.last_bc_loss,
            "grad_norm": self.last_grad_norm,
            "total_loss": self.last_loss,
            "lr": self.optimizer.param_groups[0]["lr"],
            "bc_weight": self._get_bc_weight(),
            "training_steps": self._training_steps,
        }

    def _get_bc_weight(self) -> float:
        """Compute current BC weight from decay schedule."""
        tcfg = CONFIG.train
        if self.total_frames <= tcfg.bc_decay_start_frame:
            return tcfg.bc_weight_initial
        if self.total_frames >= tcfg.bc_decay_end_frame:
            return tcfg.bc_weight_floor
        frac = (self.total_frames - tcfg.bc_decay_start_frame) / max(
            1, tcfg.bc_decay_end_frame - tcfg.bc_decay_start_frame
        )
        return tcfg.bc_weight_initial + frac * (tcfg.bc_weight_floor - tcfg.bc_weight_initial)

    def _build_lr_scheduler(self):
        """Linear warmup then cosine decay."""
        tcfg = CONFIG.train
        warmup = tcfg.lr_warmup_steps
        total = tcfg.lr_decay_steps

        def lr_lambda(step):
            if step < warmup:
                return max(0.01, step / max(1, warmup))
            progress = min(1.0, (step - warmup) / max(1, total - warmup))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            lr_range = 1.0 - (tcfg.lr_min / tcfg.lr)
            return tcfg.lr_min / tcfg.lr + lr_range * cosine

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def get_expert_ratio(self) -> float:
        """Current expert action mixing ratio from decay schedule."""
        tcfg = CONFIG.train
        if self.total_frames >= tcfg.expert_ratio_decay_frames:
            return tcfg.expert_ratio_final
        frac = self.total_frames / max(1, tcfg.expert_ratio_decay_frames)
        return tcfg.expert_ratio_initial + frac * (tcfg.expert_ratio_final - tcfg.expert_ratio_initial)

    def get_epsilon(self) -> float:
        """Current exploration epsilon from decay schedule."""
        tcfg = CONFIG.train
        if self.total_frames >= tcfg.epsilon_decay_frames:
            return tcfg.epsilon_final
        frac = self.total_frames / max(1, tcfg.epsilon_decay_frames)
        return tcfg.epsilon_initial + frac * (tcfg.epsilon_final - tcfg.epsilon_initial)

    # ── Checkpoint ──────────────────────────────────────────────────────

    def save(self, path: str = None) -> bool:
        """Save model checkpoint."""
        with self._save_lock:
            try:
                save_path = Path(path) if path else CHECKPOINT_PATH
                save_path.parent.mkdir(parents=True, exist_ok=True)

                torch.save({
                    "net_state_dict": self.net.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
                    "training_steps": self._training_steps,
                    "total_frames": self.total_frames,
                    "config": {
                        "model": CONFIG.model.__dict__,
                        "train": CONFIG.train.__dict__,
                    },
                }, str(save_path))

                GAME_SETTINGS.total_frames = self.total_frames
                GAME_SETTINGS.save()
                return True
            except Exception as e:
                print(f"Save failed: {e}")
                return False

    def load(self, path: str = None) -> bool:
        """Load model checkpoint."""
        try:
            load_path = Path(path) if path else CHECKPOINT_PATH
            if not load_path.exists():
                print(f"No checkpoint found at {load_path}")
                return False

            checkpoint = torch.load(str(load_path), map_location=self.device, weights_only=False)

            self.net.load_state_dict(checkpoint["net_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            if checkpoint.get("scheduler_state_dict") and self.lr_scheduler:
                self.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            self._training_steps = checkpoint.get("training_steps", 0)
            self.total_frames = checkpoint.get("total_frames", 0)

            GAME_SETTINGS.load()
            GAME_SETTINGS.total_frames = self.total_frames

            print(f"Loaded checkpoint: {self.total_frames:,} frames, {self._training_steps:,} training steps")
            return True
        except Exception as e:
            print(f"Load failed: {e}")
            return False

    def stop(self):
        """Cleanup on shutdown."""
        self.save()
