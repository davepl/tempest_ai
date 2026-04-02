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


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        return max(1, int(raw))
    except Exception:
        return int(default)


def _configure_cpu_torch_threads() -> tuple[int, int]:
    """Keep CPU Torch work from exploding into hundreds of threads.

    V2 already forced CPU inference workers to a single Torch thread. Without
    that, small per-frame value passes on high-core-count hosts become
    dramatically slower because every client thread can trigger a huge CPU
    threadpool fan-out.
    """
    intra_threads = _env_int("ROBOTRON_CPU_THREADS", 1)
    interop_threads = _env_int("ROBOTRON_CPU_INTEROP_THREADS", 1)

    os.environ.setdefault("OMP_NUM_THREADS", str(intra_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(intra_threads))

    try:
        torch.set_num_threads(intra_threads)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(interop_threads)
    except Exception:
        pass

    return intra_threads, interop_threads


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

        env_device = (os.getenv("ROBOTRON_DEVICE") or "").strip()
        if device == "auto" and env_device:
            device = env_device

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

        self.cpu_intra_threads = None
        self.cpu_interop_threads = None
        if self.device.type == "cpu":
            self.cpu_intra_threads, self.cpu_interop_threads = _configure_cpu_torch_threads()

        # ── Multi-GPU / CUDA stream setup ───────────────────────────────
        # If ≥2 CUDA GPUs: inference on GPU 0, training on GPU 1.
        # If 1 CUDA GPU: both on same GPU but separate CUDA streams.
        # Otherwise (CPU/MPS): single device, no streams.
        self.train_device = self.device
        self.infer_device = self.device
        self.infer_stream: Optional[torch.cuda.Stream] = None
        self.train_stream: Optional[torch.cuda.Stream] = None
        self._multi_gpu = False

        if self.device.type == "cuda":
            num_gpus = torch.cuda.device_count()
            env_infer_gpu = (os.getenv("ROBOTRON_INFER_GPU") or "").strip()
            env_train_gpu = (os.getenv("ROBOTRON_TRAIN_GPU") or "").strip()

            if env_infer_gpu and env_train_gpu:
                # Explicit GPU assignment
                self.infer_device = torch.device(f"cuda:{int(env_infer_gpu)}")
                self.train_device = torch.device(f"cuda:{int(env_train_gpu)}")
                self._multi_gpu = (self.infer_device != self.train_device)
            elif num_gpus >= 2:
                # Auto: inference on GPU 0, training on GPU 1
                self.infer_device = torch.device("cuda:0")
                self.train_device = torch.device("cuda:1")
                self._multi_gpu = True
            else:
                # Single GPU — use streams for overlap
                self.infer_device = self.device
                self.train_device = self.device

            self.infer_stream = torch.cuda.Stream(device=self.infer_device)
            self.train_stream = torch.cuda.Stream(device=self.train_device)

        # After multi-GPU setup, alias self.device to train_device so that
        # train_step() and other methods that use self.device send tensors
        # to the correct device (where self.net lives).
        self.device = self.train_device

        # Network (lives on train_device — this is the "authority" copy)
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
        ).to(self.train_device)

        # Separate inference copy (on infer_device) when multi-GPU
        self.infer_net: Optional[RobotronPPONet] = None
        if self._multi_gpu:
            import copy
            self.infer_net = copy.deepcopy(self.net).to(self.infer_device)
            self.infer_net.eval()
            self.infer_net.requires_grad_(False)
        # Weight sync event — set after each training step to signal
        # the inference batcher that fresh weights are available.
        self._weights_updated = threading.Event()

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

        # Manual override state (keyboard / dashboard controls)
        self._override_lock = threading.RLock()
        self.training_enabled = True
        self.verbose_mode = False
        self.override_expert = False          # 'o' key: force expert ratio to 0
        self.expert_mode = False              # 'e' key: force expert ratio to 1.0
        self.manual_expert_override = False   # 7/9 keys: manual expert pct
        self._manual_expert_ratio = 0.0       # value set by 7/9
        self._saved_expert_ratio = 0.0        # stash for o/e toggles
        self.manual_epsilon_override = False  # 4/6 keys: manual epsilon pct
        self._manual_epsilon = 0.0            # value set by 4/6

        print(f"PPO Agent initialized on {self.device}")
        print(f"  Network params: {sum(p.numel() for p in self.net.parameters()):,}")
        print(f"  Entity feature dim: {cfg.entity_feature_dim}")
        print(f"  Max entities: {cfg.max_entities}")
        print(f"  Embed dim: {cfg.embed_dim}")
        print(f"  ISAB layers: {cfg.num_isab_layers}")
        print(f"  Inducing points: {cfg.num_inducing_points}")
        print(f"  Frame stack: {cfg.frame_stack}")
        if self._multi_gpu:
            print(f"  Multi-GPU: infer={self.infer_device}, train={self.train_device}")
        elif self.infer_stream is not None:
            print(f"  CUDA streams: inference + training on {self.device}")
        if self.device.type == "cpu":
            print(f"  Torch CPU threads: intra={self.cpu_intra_threads} interop={self.cpu_interop_threads}")

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
        device: Optional[torch.device] = None,
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

        # Keep per-frame preprocessing on CPU unless the caller explicitly
        # requests a device. The batched inference path moves whole batches.
        tensors = self.state_processor.to_tensors(stacked, device)
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
        tensors = self._process_and_stack(wire_state, client_id, device=self.device)

        self.net.eval()
        out = self.net(
            tensors["entity_features"],
            tensors["entity_mask"],
            tensors["global_context"],
        )

        move_logits = out["move_logits"][0].clamp(-50, 50)
        fire_logits = out["fire_logits"][0].clamp(-50, 50)
        move_probs = F.softmax(move_logits, dim=-1)
        fire_probs = F.softmax(fire_logits, dim=-1)

        # Guard against nan/inf from degenerate network outputs
        if torch.isnan(move_probs).any() or (move_probs < 0).any():
            move_probs = torch.ones_like(move_probs) / move_probs.numel()
        if torch.isnan(fire_probs).any() or (fire_probs < 0).any():
            fire_probs = torch.ones_like(fire_probs) / fire_probs.numel()

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
        tensors = self._process_and_stack(wire_state, client_id, device=self.device)

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
            tensors = self._process_and_stack(wire_state, client_id, device=self.device)
            self.net.eval()
            value = self.net.get_value(
                tensors["entity_features"],
                tensors["entity_mask"],
                tensors["global_context"],
            ).item()
            return move, fire, 0.0, value, True, self._detach_tensors(tensors)

        tensors = self._process_and_stack(wire_state, client_id, device=self.device)

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
        detached = {}
        for k, v in tensors.items():
            out = v.detach().squeeze(0)
            detached[k] = out if out.device.type == "cpu" else out.cpu()
        return detached

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
        total_total_loss = 0.0
        num_batches = 0
        last_finite_grad_norm = self.last_grad_norm if math.isfinite(float(self.last_grad_norm)) else 0.0

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
            has_value = batch["has_value"].to(self.device)
            advantages = batch["advantages"].to(self.device)
            returns = batch["returns"].to(self.device)
            expert_move = batch["expert_move"].to(self.device)
            expert_fire = batch["expert_fire"].to(self.device)
            is_expert = batch["is_expert"].to(self.device)
            policy_sampled = batch["policy_sampled"].to(self.device)
            fire_locked = batch["fire_locked"].to(self.device)
            any_policy = bool(policy_sampled.any().item())
            any_value = bool(has_value.any().item())
            any_expert = bool(is_expert.any().item())

            # Forward pass: evaluate the stored actions under the current policy.
            out = self.net(entity_features, entity_masks, global_contexts)
            move_logits = out["move_logits"].clamp(-50.0, 50.0)
            fire_logits = out["fire_logits"].clamp(-50.0, 50.0)
            if torch.isnan(move_logits).any() or torch.isinf(move_logits).any():
                move_logits = torch.zeros_like(move_logits)
            if torch.isnan(fire_logits).any() or torch.isinf(fire_logits).any():
                fire_logits = torch.zeros_like(fire_logits)

            new_values = out["value"]

            # PPO clipped objective
            if any_policy:
                move_dist = torch.distributions.Categorical(logits=move_logits)
                fire_dist = torch.distributions.Categorical(logits=fire_logits)
                move_log_probs = move_dist.log_prob(old_move)
                fire_log_probs = fire_dist.log_prob(old_fire)
                new_log_probs = move_log_probs + torch.where(
                    fire_locked,
                    torch.zeros_like(fire_log_probs),
                    fire_log_probs,
                )
                entropy = move_dist.entropy() + torch.where(
                    fire_locked,
                    torch.zeros_like(fire_logits[:, 0]),
                    fire_dist.entropy(),
                )
                adv_policy = advantages[policy_sampled]
                adv_policy_std = adv_policy.std(unbiased=False).clamp_min(1e-8)
                adv_policy = (adv_policy - adv_policy.mean()) / adv_policy_std
                ratio = torch.exp(new_log_probs[policy_sampled] - old_log_probs[policy_sampled])
                surr1 = ratio * adv_policy
                surr2 = torch.clamp(ratio, 1.0 - tcfg.clip_epsilon, 1.0 + tcfg.clip_epsilon) * adv_policy
                policy_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -entropy[policy_sampled].mean()
            else:
                policy_loss = torch.zeros((), device=self.device)
                entropy_loss = torch.zeros((), device=self.device)

            # Value loss (clipped)
            if any_value:
                old_values = batch["values"].to(self.device)[has_value]
                value_pred = new_values[has_value]
                returns_value = returns[has_value]
                value_clipped = old_values + torch.clamp(
                    value_pred - old_values,
                    -tcfg.clip_value, tcfg.clip_value,
                )
                value_loss1 = F.mse_loss(value_pred, returns_value)
                value_loss2 = F.mse_loss(value_clipped, returns_value)
                value_loss = torch.max(value_loss1, value_loss2)
            else:
                value_loss = torch.zeros((), device=self.device)

            # Behavioral cloning loss (if expert demonstrations present)
            bc_loss = torch.tensor(0.0, device=self.device)
            bc_weight = self._get_bc_weight()
            if any_expert and bc_weight > 0:
                expert_move_logits = move_logits[is_expert]
                expert_fire_logits = fire_logits[is_expert]
                bc_move_loss = F.cross_entropy(expert_move_logits, expert_move[is_expert])
                expert_fire_locked = fire_locked[is_expert]
                if (~expert_fire_locked).any():
                    bc_fire_loss = F.cross_entropy(
                        expert_fire_logits[~expert_fire_locked],
                        expert_fire[is_expert][~expert_fire_locked],
                    )
                    bc_loss = bc_move_loss + bc_fire_loss
                else:
                    bc_loss = bc_move_loss

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

            # Skip update if loss or grad norm is non-finite
            grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
            if not (math.isfinite(loss.item()) and math.isfinite(grad_norm_val)):
                self.optimizer.zero_grad()
                continue

            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            self._training_steps += 1
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy_loss += entropy_loss.item()
            total_bc_loss += bc_loss.item()
            total_total_loss += loss.item()
            last_finite_grad_norm = grad_norm_val
            num_batches += 1

        if num_batches > 0:
            self.last_policy_loss = total_policy_loss / num_batches
            self.last_value_loss = total_value_loss / num_batches
            self.last_entropy = -total_entropy_loss / num_batches
            self.last_bc_loss = total_bc_loss / num_batches
            self.last_loss = total_total_loss / num_batches
            self.last_grad_norm = last_finite_grad_norm

        if not math.isfinite(float(self.last_grad_norm)):
            self.last_grad_norm = 0.0
        if not math.isfinite(float(self.last_loss)):
            self.last_loss = 0.0

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

    def sync_inference_weights(self):
        """Copy trained weights to the inference network.

        Multi-GPU: copies state_dict from train_device → infer_device.
        Single-GPU with streams: no-op (same net object, streams handle sync).
        """
        if self.infer_net is None:
            # Single-GPU or CPU — inference uses self.net directly.
            # Signal the batcher that weights changed (for stream sync).
            self._weights_updated.set()
            return
        # Multi-GPU: copy state_dict across devices
        src = self.net.state_dict()
        dst = {k: v.to(self.infer_device, non_blocking=True) for k, v in src.items()}
        self.infer_net.load_state_dict(dst, strict=True)
        self._weights_updated.set()

    def get_inference_net(self) -> RobotronPPONet:
        """Return the network copy used for inference (may differ from training net)."""
        return self.infer_net if self.infer_net is not None else self.net

    def get_expert_ratio(self) -> float:
        """Current expert action mixing ratio, respecting manual overrides."""
        with self._override_lock:
            if self.expert_mode:
                return 1.0
            if self.override_expert:
                return 0.0
            if self.manual_expert_override:
                return self._manual_expert_ratio
        # Natural decay schedule
        tcfg = CONFIG.train
        if self.total_frames >= tcfg.expert_ratio_decay_frames:
            return tcfg.expert_ratio_final
        frac = self.total_frames / max(1, tcfg.expert_ratio_decay_frames)
        return tcfg.expert_ratio_initial + frac * (tcfg.expert_ratio_final - tcfg.expert_ratio_initial)

    def get_epsilon(self) -> float:
        """Current exploration epsilon, respecting manual overrides."""
        with self._override_lock:
            if self.manual_epsilon_override:
                return self._manual_epsilon
        # Natural decay schedule
        tcfg = CONFIG.train
        if self.total_frames >= tcfg.epsilon_decay_frames:
            return tcfg.epsilon_final
        frac = self.total_frames / max(1, tcfg.epsilon_decay_frames)
        return tcfg.epsilon_initial + frac * (tcfg.epsilon_final - tcfg.epsilon_initial)

    # ── Interactive parameter controls ──────────────────────────────────

    def toggle_override(self) -> str:
        """Toggle expert override (force expert ratio to 0%)."""
        with self._override_lock:
            self.override_expert = not self.override_expert
            self.expert_mode = False
            self.manual_expert_override = False
            return f"Expert override {'ON (0%)' if self.override_expert else 'OFF (natural)'}"

    def toggle_expert_mode(self) -> str:
        """Toggle 100% expert mode."""
        with self._override_lock:
            self.expert_mode = not self.expert_mode
            self.override_expert = False
            self.manual_expert_override = False
            return f"Expert mode {'ON (100%)' if self.expert_mode else 'OFF (natural)'}"

    def toggle_training(self) -> str:
        """Toggle training on/off."""
        self.training_enabled = not self.training_enabled
        return f"Training {'ENABLED' if self.training_enabled else 'DISABLED'}"

    def toggle_verbose(self) -> str:
        """Toggle verbose output."""
        self.verbose_mode = not self.verbose_mode
        return f"Verbose {'ON' if self.verbose_mode else 'OFF'}"

    def increase_expert_ratio(self) -> str:
        """Increase expert ratio by 1% (if <10%) or 5%."""
        with self._override_lock:
            cur = self.get_expert_ratio() if not self.manual_expert_override else self._manual_expert_ratio
            p = int(cur * 100)
            p = min(100, p + (1 if p < 10 else 5))
            self._manual_expert_ratio = p / 100.0
            self.manual_expert_override = True
            self.override_expert = False
            self.expert_mode = False
            return f"Expert ratio → {p}%*"

    def decrease_expert_ratio(self) -> str:
        """Decrease expert ratio by 1% (if <=10%) or 5%."""
        with self._override_lock:
            cur = self.get_expert_ratio() if not self.manual_expert_override else self._manual_expert_ratio
            p = int(cur * 100)
            p = max(0, p - (1 if p <= 10 else 5))
            self._manual_expert_ratio = p / 100.0
            self.manual_expert_override = True
            self.override_expert = False
            self.expert_mode = False
            return f"Expert ratio → {p}%*"

    def restore_natural_expert_ratio(self) -> str:
        """Restore expert ratio to natural decay schedule."""
        with self._override_lock:
            self.manual_expert_override = False
            self.override_expert = False
            self.expert_mode = False
        r = self.get_expert_ratio()
        return f"Expert ratio → {r:.1%} (natural)"

    def increase_epsilon(self) -> str:
        """Increase epsilon by 1% (if <10%) or 5%."""
        with self._override_lock:
            cur = self.get_epsilon() if not self.manual_epsilon_override else self._manual_epsilon
            p = int(cur * 100)
            p = min(100, p + (1 if p < 10 else 5))
            self._manual_epsilon = p / 100.0
            self.manual_epsilon_override = True
            return f"Epsilon → {p}%*"

    def decrease_epsilon(self) -> str:
        """Decrease epsilon by 1% (if <=10%) or 5%."""
        with self._override_lock:
            cur = self.get_epsilon() if not self.manual_epsilon_override else self._manual_epsilon
            p = int(cur * 100)
            p = max(0, p - (1 if p <= 10 else 5))
            self._manual_epsilon = p / 100.0
            self.manual_epsilon_override = True
            return f"Epsilon → {p}%*"

    def restore_natural_epsilon(self) -> str:
        """Restore epsilon to natural decay schedule."""
        with self._override_lock:
            self.manual_epsilon_override = False
        e = self.get_epsilon()
        return f"Epsilon → {e:.3f} (natural)"

    def is_expert_overridden(self) -> bool:
        """True if expert ratio is manually overridden (for display markers)."""
        return self.override_expert or self.expert_mode or self.manual_expert_override

    def is_epsilon_overridden(self) -> bool:
        """True if epsilon is manually overridden (for display markers)."""
        return self.manual_epsilon_override

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

            checkpoint = torch.load(str(load_path), map_location=self.train_device, weights_only=False)

            self.net.load_state_dict(checkpoint["net_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            self._training_steps = checkpoint.get("training_steps", 0)
            self.total_frames = checkpoint.get("total_frames", 0)

            if checkpoint.get("scheduler_state_dict"):
                if self.lr_scheduler is None:
                    self.lr_scheduler = self._build_lr_scheduler()
                self.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            GAME_SETTINGS.load()
            GAME_SETTINGS.total_frames = self.total_frames

            # Sync inference copy with freshly loaded weights
            self.sync_inference_weights()

            print(f"Loaded checkpoint: {self.total_frames:,} frames, {self._training_steps:,} training steps")
            return True
        except Exception as e:
            print(f"Load failed: {e}")
            return False

    def stop(self):
        """Cleanup on shutdown."""
        self.save()
