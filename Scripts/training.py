#!/usr/bin/env python3
# ==================================================================================================================
# ||  TEMPEST AI v2 • TRAINING STEP                                                                              ||
# ||  C51 distributional Bellman update with PER and optional BC loss.                                            ||
# ==================================================================================================================
"""Single training step for Rainbow-lite agent."""

import time, math
from contextlib import nullcontext
import numpy as np
import torch
import torch.nn.functional as F

try:
    from config import RL_CONFIG, metrics
except ImportError:
    from Scripts.config import RL_CONFIG, metrics

# Device (mirrors aimodel.py)
if torch.cuda.is_available():
    n = torch.cuda.device_count()
    idx = int(getattr(RL_CONFIG, "train_cuda_device_index", 0))
    if idx < 0 or idx >= n:
        idx = 0
    device = torch.device(f"cuda:{idx}")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def _beta_schedule(frame_count: int) -> float:
    """Anneal PER beta from start → 1.0."""
    progress = min(1.0, frame_count / max(1, RL_CONFIG.priority_beta_frames))
    return RL_CONFIG.priority_beta_start + progress * (1.0 - RL_CONFIG.priority_beta_start)


def _bc_weight_schedule(frame_count: int) -> float:
    """Anneal behavioural-cloning weight."""
    cfg = RL_CONFIG
    if frame_count < cfg.expert_bc_decay_start:
        return cfg.expert_bc_weight
    progress = min(1.0, (frame_count - cfg.expert_bc_decay_start) / max(1, cfg.expert_bc_decay_frames))
    return cfg.expert_bc_weight + progress * (cfg.expert_bc_min_weight - cfg.expert_bc_weight)


def train_step(agent, prefetched_batch=None) -> float | None:
    """Run one C51 distributional training step.

    Returns the scalar loss value, or None if training was skipped.
    """
    if not getattr(metrics, "training_enabled", True) or not agent.training_enabled:
        return None

    if len(agent.memory) < max(RL_CONFIG.min_replay_to_train, RL_CONFIG.batch_size):
        return None

    # Keep replay pressure bounded so optimization does not outrun data refresh.
    try:
        with metrics.lock:
            frame_count = int(metrics.frame_count)
            loaded_frame_count = int(getattr(metrics, "loaded_frame_count", 0))
        loaded_training_steps = int(getattr(agent, "loaded_training_steps", 0))
        max_spf = float(getattr(RL_CONFIG, "max_samples_per_frame", 0.0))
        if max_spf > 0.0 and frame_count > 0:
            recent_frames = max(1, frame_count - loaded_frame_count)
            recent_steps = max(0, int(agent.training_steps) - loaded_training_steps)
            sampled_per_frame = (float(recent_steps) * float(RL_CONFIG.batch_size)) / float(recent_frames)
            if sampled_per_frame >= max_spf:
                return None
    except Exception:
        pass

    # ── Sample ──────────────────────────────────────────────────────────
    beta = _beta_schedule(metrics.frame_count)
    batch = prefetched_batch if prefetched_batch is not None else agent.memory.sample(RL_CONFIG.batch_size, beta=beta)
    if batch is None:
        return None

    states, actions, rewards, next_states, dones, horizons, is_expert, indices, weights = batch

    states_t      = torch.from_numpy(states).float().to(device)
    actions_t     = torch.from_numpy(actions).long().to(device)
    rewards_t     = torch.from_numpy(rewards).float().to(device)
    next_states_t = torch.from_numpy(next_states).float().to(device)
    dones_t       = torch.from_numpy(dones).float().to(device)
    horizons_t    = torch.from_numpy(horizons.astype(np.float32)).float().to(device)
    weights_t     = torch.from_numpy(weights).float().to(device)
    is_expert_t   = torch.from_numpy(is_expert).bool().to(device)

    B = states_t.shape[0]
    cfg = RL_CONFIG

    agent.online_net.train()
    agent._update_lr()

    use_amp = agent.use_amp and device.type == "cuda"
    scaler = agent.grad_scaler
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()

    # ── C51 distributional update ───────────────────────────────────────
    num_atoms = cfg.num_atoms
    v_min, v_max = cfg.v_min, cfg.v_max
    delta_z = (v_max - v_min) / (num_atoms - 1)
    support = agent.online_net.support  # (num_atoms,)

    with amp_ctx:
        # Current distribution
        log_p = agent.online_net(states_t, log=True)       # (B, A, N)
        log_p_a = log_p[torch.arange(B, device=device), actions_t]  # (B, N)

        # Target distribution (Double-DQN style: online selects, target evaluates)
        with torch.no_grad():
            # Select actions with online net
            q_next = agent.online_net.q_values(next_states_t)
            best_next = q_next.argmax(dim=1)                # (B,)

            # Get target distribution for those actions
            target_p = agent.target_net(next_states_t, log=False)  # (B, A, N)
            target_p_a = target_p[torch.arange(B, device=device), best_next]  # (B, N)

            # Compute Tz (projected Bellman update with n-step)
            gamma_n = cfg.gamma ** horizons_t       # (B,)
            Tz = rewards_t.unsqueeze(1) + (1.0 - dones_t.unsqueeze(1)) * gamma_n.unsqueeze(1) * support.unsqueeze(0)
            Tz = Tz.clamp(v_min, v_max)

            # Project onto support
            b = (Tz - v_min) / delta_z               # (B, N)
            l = b.floor().long()
            u = b.ceil().long()
            # Clamp to valid range
            l = l.clamp(0, num_atoms - 1)
            u = u.clamp(0, num_atoms - 1)

            # Distribute probability
            m = torch.zeros(B, num_atoms, device=device, dtype=torch.float32)
            offset = torch.linspace(0, (B - 1) * num_atoms, B, device=device, dtype=torch.long).unsqueeze(1).expand_as(l)

            # When l == u, both (u-b) and (b-l) are 0 → mass is lost.
            # Fix: assign full mass to that bin directly.
            eq_mask = (l == u)
            neq_mask = ~eq_mask

            m.view(-1).index_add_(0, (l + offset).view(-1), (target_p_a * (u.float() - b) * neq_mask.float()).view(-1))
            m.view(-1).index_add_(0, (u + offset).view(-1), (target_p_a * (b - l.float()) * neq_mask.float()).view(-1))
            m.view(-1).index_add_(0, (l + offset).view(-1), (target_p_a * eq_mask.float()).view(-1))

        # Cross-entropy loss (weighted by IS weights)
        ce_loss = -(m * log_p_a).sum(dim=1)             # (B,)
        weighted_loss = (weights_t * ce_loss).mean()

    # ── Optional BC loss on expert transitions ─────────────────────────
    bc_loss_val = 0.0
    bc_w = _bc_weight_schedule(metrics.frame_count)
    if bc_w > 0.0 and is_expert_t.any():
        with amp_ctx:
            expert_idx = is_expert_t.nonzero(as_tuple=True)[0]
            if expert_idx.numel() > 0:
                q_expert = agent.online_net.q_values(states_t[expert_idx])
                bc_loss = F.cross_entropy(q_expert, actions_t[expert_idx])
                # Scale BC by sampled expert fraction to avoid over-weighting when
                # expert transitions are sparse but present in most batches.
                bc_scale = float(expert_idx.numel()) / float(B)
                weighted_loss = weighted_loss + (bc_w * bc_scale) * bc_loss
                bc_loss_val = float(bc_loss.detach().item())

    # ── NaN / Inf guard ───────────────────────────────────────────────────
    if not torch.isfinite(weighted_loss):
        print(f"[WARN] Non-finite loss detected ({weighted_loss.item():.4g}), skipping step")
        return None

    # ── Optimise ────────────────────────────────────────────────────────
    try:
        agent.optimizer.zero_grad(set_to_none=True)
    except TypeError:
        agent.optimizer.zero_grad()

    clip_norm = cfg.grad_clip_norm
    if use_amp and scaler is not None:
        scaler.scale(weighted_loss).backward()
        scaler.unscale_(agent.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(agent.online_net.parameters(), clip_norm)
        scaler.step(agent.optimizer)
        scaler.update()
    else:
        weighted_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(agent.online_net.parameters(), clip_norm)
        agent.optimizer.step()

    # ── Update priorities ───────────────────────────────────────────────
    td_errors = ce_loss.detach().cpu().numpy()
    agent.memory.update_priorities(indices, td_errors)

    # ── Target network update ───────────────────────────────────────────
    agent.training_steps += 1
    if agent.training_steps % cfg.target_update_period == 0:
        agent.update_target()

    # ── Sync inference model ────────────────────────────────────────────
    agent._sync_inference(force=False)

    # ── Metrics ─────────────────────────────────────────────────────────
    try:
        loss_val = float(weighted_loss.detach().item())
        gn = float(grad_norm.item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm)

        metrics.total_training_steps += 1
        if hasattr(metrics, "training_steps_interval"):
            metrics.training_steps_interval += 1
        metrics.memory_buffer_size = len(agent.memory)
        metrics.losses.append(loss_val)
        metrics.last_loss = loss_val
        metrics.last_grad_norm = gn
        metrics.last_bc_loss = bc_loss_val
        metrics.last_priority_mean = float(np.mean(td_errors))

        # Agreement (Q-values for display)
        with torch.no_grad():
            q_all = agent.online_net.q_values(states_t)
            pred = q_all.argmax(dim=1)
            agree = (pred == actions_t).float().mean().item()
            metrics.last_q_mean = float(q_all.mean().item())

        if hasattr(metrics, "agree_sum_interval"):
            metrics.agree_sum_interval += agree
            metrics.agree_count_interval += 1
        if hasattr(metrics, "loss_sum_interval"):
            metrics.loss_sum_interval += loss_val
            metrics.loss_count_interval += 1
    except Exception:
        pass

    return loss_val
