#!/usr/bin/env python3
# ==================================================================================================================
# ||  ROBOTRON AI v2 • TRAINING STEP                                                                             ||
# ||  C51 distributional Bellman update with PER and optional BC loss.                                            ||
# ==================================================================================================================
"""Single training step for Rainbow-lite agent."""

import time, math
from contextlib import nullcontext
import numpy as np
import torch
import torch.nn.functional as F

try:
    from config import (
        RL_CONFIG,
        metrics,
        LEGACY_CORE_FEATURES,
        LEGACY_ELIST_FEATURES,
        LEGACY_SLOT_STATE_FEATURES,
        TACTICAL_LANE_COUNT,
        TACTICAL_LANE_FEATURES,
        TACTICAL_POOL_DEFS,
        UNIFIED_TYPE_NAMES,
        UNIFIED_NUM_TYPES,
        UNIFIED_HUMAN_TYPE_ID,
        UNIFIED_ELECTRODE_TYPE_ID,
    )
except ImportError:
    from Scripts.config import (
        RL_CONFIG,
        metrics,
        LEGACY_CORE_FEATURES,
        LEGACY_ELIST_FEATURES,
        LEGACY_SLOT_STATE_FEATURES,
        TACTICAL_LANE_COUNT,
        TACTICAL_LANE_FEATURES,
        TACTICAL_POOL_DEFS,
        UNIFIED_TYPE_NAMES,
        UNIFIED_NUM_TYPES,
        UNIFIED_HUMAN_TYPE_ID,
        UNIFIED_ELECTRODE_TYPE_ID,
    )

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


_PROJECTILE_TYPE_ID = UNIFIED_TYPE_NAMES.index("projectile") if "projectile" in UNIFIED_TYPE_NAMES else -1


def _latest_frame_danger(states_t: torch.Tensor) -> torch.Tensor:
    """Estimate tactical danger from the latest frame of the stacked state.

    This is intentionally simple and cheap: move supervision is emphasized when
    the recent state contains nearby high-threat entities, especially
    projectiles and electrodes.
    """
    batch = int(states_t.shape[0]) if states_t.ndim >= 1 else 0
    if batch <= 0:
        return torch.zeros(0, device=states_t.device, dtype=torch.float32)

    base_state_size = int(getattr(RL_CONFIG, "base_state_size", 0) or 0)
    lane_count = int(getattr(RL_CONFIG, "lane_token_count", TACTICAL_LANE_COUNT) or TACTICAL_LANE_COUNT)
    lane_features = int(getattr(RL_CONFIG, "lane_token_features", TACTICAL_LANE_FEATURES) or TACTICAL_LANE_FEATURES)
    tactical_grid_w = int(getattr(RL_CONFIG, "tactical_grid_width", 0) or 0)
    tactical_grid_h = int(getattr(RL_CONFIG, "tactical_grid_height", 0) or 0)
    tactical_grid_c = int(getattr(RL_CONFIG, "tactical_grid_channels", 0) or 0)
    pool_defs = getattr(RL_CONFIG, "state_role_pools", TACTICAL_POOL_DEFS) or TACTICAL_POOL_DEFS
    if base_state_size <= 0:
        return torch.zeros(batch, device=states_t.device, dtype=torch.float32)

    latest = states_t[:, -base_state_size:]
    offset = int(
        LEGACY_CORE_FEATURES
        + LEGACY_ELIST_FEATURES
        + (lane_count * lane_features)
        + (tactical_grid_w * tactical_grid_h * tactical_grid_c)
    )
    if latest.shape[1] < offset:
        return torch.zeros(batch, device=states_t.device, dtype=torch.float32)

    score_parts = []
    mask_parts = []
    close_parts = []

    for pool_name, slots, slot_features in pool_defs:
        slots_i = int(slots)
        feat_i = int(slot_features)
        width = slots_i * feat_i
        if latest.shape[1] < (offset + 1 + width):
            return torch.zeros(batch, device=states_t.device, dtype=torch.float32)

        block = latest[:, offset + 1: offset + 1 + width].reshape(batch, slots_i, feat_i)
        present = block[:, :, 0] > 0.5

        if pool_name == "projectile":
            dist = block[:, :, 3].clamp(min=0.0, max=2.0)
            threat = block[:, :, 6].clamp(min=0.0, max=1.0)
            ttc = block[:, :, 7].clamp(min=0.0, max=1.0)
            closest_pass = block[:, :, 8].clamp(min=0.0, max=2.0)
            approach = block[:, :, 9].clamp(min=-1.0, max=1.0)
            is_projectile = torch.ones_like(present)
            is_electrode = torch.zeros_like(present)
        elif pool_name == "danger":
            dist = block[:, :, 3].clamp(min=0.0, max=2.0)
            threat = block[:, :, 6].clamp(min=0.0, max=1.0)
            approach = block[:, :, 7].clamp(min=-1.0, max=1.0)
            ttc = block[:, :, 8].clamp(min=0.0, max=1.0)
            closest_pass = dist
            type_norm = block[:, :, 9].clamp(min=0.0, max=1.0)
            type_idx = torch.round(type_norm * float(max(1, UNIFIED_NUM_TYPES - 1))).long()
            type_idx = type_idx.clamp(0, max(0, UNIFIED_NUM_TYPES - 1))
            is_projectile = (type_idx == int(_PROJECTILE_TYPE_ID)) if _PROJECTILE_TYPE_ID >= 0 else torch.zeros_like(present)
            is_electrode = type_idx == int(UNIFIED_ELECTRODE_TYPE_ID)
        elif pool_name == "electrode":
            dist = block[:, :, 3].clamp(min=0.0, max=2.0)
            threat = block[:, :, 4].clamp(min=0.0, max=1.0)
            approach = torch.zeros_like(dist)
            ttc = torch.ones_like(dist)
            closest_pass = dist
            is_projectile = torch.zeros_like(present)
            is_electrode = torch.ones_like(present)
        else:
            offset += 1 + width
            continue

        closeness = (1.0 - dist).clamp(min=0.0, max=1.0)
        approach_score = approach.clamp(min=0.0, max=1.0)
        imminence = (1.0 - ttc).clamp(min=0.0, max=1.0)
        pass_score = (1.0 - closest_pass).clamp(min=0.0, max=1.0)
        danger_score = (0.35 * closeness) + (0.30 * threat) + (0.15 * approach_score) + (0.20 * imminence)
        danger_score = danger_score + (0.20 * is_projectile.float() * pass_score)
        danger_score = danger_score + (0.20 * is_electrode.float() * closeness)
        danger_score = torch.where(present, danger_score, torch.zeros_like(danger_score))

        score_parts.append(danger_score)
        mask_parts.append(present)
        close_parts.append(closeness)
        offset += 1 + width

    if not score_parts:
        return torch.zeros(batch, device=states_t.device, dtype=torch.float32)

    all_scores = torch.cat(score_parts, dim=1)
    all_masks = torch.cat(mask_parts, dim=1)
    all_close = torch.cat(close_parts, dim=1)

    max_score = all_scores.max(dim=1).values
    nearby_count = (all_masks & (all_close > 0.35)).float().sum(dim=1).clamp(max=3.0) / 3.0
    return (max_score + (0.25 * nearby_count)).clamp(min=0.0, max=1.5)


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

    if len(batch) == 12:
        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            horizons,
            is_expert,
            is_self_imitation,
            _wave_numbers,
            _start_waves,
            indices,
            weights,
        ) = batch
    elif len(batch) == 11:
        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            horizons,
            is_expert,
            _wave_numbers,
            _start_waves,
            indices,
            weights,
        ) = batch
        is_self_imitation = np.zeros_like(is_expert, dtype=np.bool_)
    elif len(batch) == 9:
        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            horizons,
            is_expert,
            indices,
            weights,
        ) = batch
        _wave_numbers = None
        _start_waves = None
        is_self_imitation = np.zeros_like(is_expert, dtype=np.bool_)
    else:
        raise ValueError(f"Unexpected replay batch shape: expected 9, 11, or 12 items, got {len(batch)}")

    states_t      = torch.from_numpy(states).float().to(device)
    actions_t     = torch.from_numpy(actions).long().to(device)
    rewards_t     = torch.from_numpy(rewards).float().to(device)
    next_states_t = torch.from_numpy(next_states).float().to(device)
    dones_t       = torch.from_numpy(dones).float().to(device)
    horizons_t    = torch.from_numpy(horizons.astype(np.float32)).float().to(device)
    weights_t     = torch.from_numpy(weights).float().to(device)
    is_expert_t   = torch.from_numpy(is_expert).bool().to(device)
    is_self_imitation_t = torch.from_numpy(np.asarray(is_self_imitation, dtype=np.bool_)).bool().to(device)

    B = states_t.shape[0]
    cfg = RL_CONFIG

    agent.online_net.train()
    agent._update_lr()

    use_amp = agent.use_amp and device.type == "cuda"
    scaler = agent.grad_scaler
    amp_dtype = getattr(agent, "amp_dtype", torch.float16)
    amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()

    # ── C51 distributional update ───────────────────────────────────────
    num_atoms = cfg.num_atoms
    v_min, v_max = cfg.v_min, cfg.v_max
    delta_z = (v_max - v_min) / (num_atoms - 1)
    support = agent.online_net.support  # (num_atoms,)

    with amp_ctx:
        # Current distribution
        log_p = agent.online_net(states_t, log=True)       # (B, A, N)
        log_p_a = log_p[torch.arange(B, device=device), actions_t]  # (B, N)
        q_all = (log_p.exp() * support.unsqueeze(0).unsqueeze(0)).sum(dim=2)  # (B, A)

        # Target distribution (Double-DQN style: online selects, target evaluates)
        with torch.no_grad():
            # Select actions with online net using a single distributional forward.
            online_next_p = agent.online_net(next_states_t, log=False)
            q_next = (online_next_p * support.unsqueeze(0).unsqueeze(0)).sum(dim=2)
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
        # Clamp log_p_a to prevent -inf × nonzero target mass from producing
        # inf gradients.  log_softmax can hit -inf when an atom's probability
        # collapses toward zero (e.g. death transitions with extreme rewards).
        # -30 ≈ log(1e-13), well below any meaningful probability.
        log_p_a_safe = log_p_a.clamp(min=-30.0)
        ce_loss = -(m * log_p_a_safe).sum(dim=1)        # (B,)
        weighted_loss = (weights_t * ce_loss).mean()

    move_targets = torch.div(actions_t, cfg.num_fire_actions, rounding_mode="floor")
    fire_targets = torch.remainder(actions_t, cfg.num_fire_actions)
    q_joint = q_all.view(B, cfg.num_move_actions, cfg.num_fire_actions)
    move_scores = q_joint.max(dim=2).values
    fire_scores = q_joint.max(dim=1).values

    # ── Optional BC loss on expert/self-imitation transitions ──────────
    bc_loss_val = 0.0
    bc_w = _bc_weight_schedule(metrics.frame_count)
    imitation_mask = is_expert_t | is_self_imitation_t
    if bc_w > 0.0 and imitation_mask.any():
        imitation_idx = imitation_mask.nonzero(as_tuple=True)[0]
        if imitation_idx.numel() > 0:
            imitation_scale = torch.where(
                is_expert_t[imitation_idx],
                torch.ones_like(imitation_idx, dtype=torch.float32, device=device),
                torch.full_like(imitation_idx, float(getattr(cfg, "factorized_bc_self_imitation_scale", 0.75)), dtype=torch.float32, device=device),
            )

            if bool(getattr(cfg, "factorized_bc_enabled", False)):
                move_w = float(getattr(cfg, "factorized_bc_move_weight", 1.0) or 1.0)
                fire_w = float(getattr(cfg, "factorized_bc_fire_weight", 1.0) or 1.0)
                danger_scale = float(getattr(cfg, "factorized_bc_danger_scale", 0.0) or 0.0)
                danger = _latest_frame_danger(states_t)[imitation_idx]
                move_sample_w = imitation_scale * (1.0 + (danger_scale * danger))
                fire_sample_w = imitation_scale

                move_ce = F.cross_entropy(move_scores[imitation_idx].float(), move_targets[imitation_idx], reduction="none")
                fire_ce = F.cross_entropy(fire_scores[imitation_idx].float(), fire_targets[imitation_idx], reduction="none")
                move_bc = (move_ce * move_sample_w).sum() / move_sample_w.sum().clamp(min=1e-6)
                fire_bc = (fire_ce * fire_sample_w).sum() / fire_sample_w.sum().clamp(min=1e-6)
                bc_loss = (move_w * move_bc) + (fire_w * fire_bc)
            else:
                bc_ce = F.cross_entropy(q_all[imitation_idx].float(), actions_t[imitation_idx], reduction="none")
                bc_loss = (bc_ce * imitation_scale).sum() / imitation_scale.sum().clamp(min=1e-6)

            bc_scale = float(imitation_idx.numel()) / float(B)
            weighted_loss = weighted_loss + (bc_w * bc_scale) * bc_loss
            bc_loss_val = float(bc_loss.detach().item())

    # ── Reward-weighted BC for DQN discoveries ────────────────────────
    # Give positive-reward DQN frames a direct "do this action" signal
    # proportional to reward, so useful behaviours found by exploration
    # are reinforced immediately rather than only via the slow C51 path.
    dqn_bc_w = float(getattr(cfg, "dqn_reward_bc_weight", 0.0) or 0.0)
    if dqn_bc_w > 0.0:
        dqn_mask = ~is_expert_t & ~is_self_imitation_t
        dqn_thresh = float(getattr(cfg, "dqn_reward_bc_threshold", 0.5) or 0.5)
        dqn_positive = dqn_mask & (rewards_t > dqn_thresh)
        if dqn_positive.any():
            dqn_idx = dqn_positive.nonzero(as_tuple=True)[0]
            dqn_max_w = float(getattr(cfg, "dqn_reward_bc_max_weight", 3.0) or 3.0)
            reward_w = rewards_t[dqn_idx].clamp(min=0.0, max=dqn_max_w)
            if bool(getattr(cfg, "factorized_bc_enabled", False)):
                move_w_f = float(getattr(cfg, "factorized_bc_move_weight", 1.0) or 1.0)
                fire_w_f = float(getattr(cfg, "factorized_bc_fire_weight", 1.0) or 1.0)
                dqn_move_ce = F.cross_entropy(move_scores[dqn_idx].float(), move_targets[dqn_idx], reduction="none")
                dqn_fire_ce = F.cross_entropy(fire_scores[dqn_idx].float(), fire_targets[dqn_idx], reduction="none")
                dqn_move_bc = (dqn_move_ce * reward_w).sum() / reward_w.sum().clamp(min=1e-6)
                dqn_fire_bc = (dqn_fire_ce * reward_w).sum() / reward_w.sum().clamp(min=1e-6)
                dqn_bc_loss = (move_w_f * dqn_move_bc) + (fire_w_f * dqn_fire_bc)
            else:
                dqn_bc_ce = F.cross_entropy(q_all[dqn_idx].float(), actions_t[dqn_idx], reduction="none")
                dqn_bc_loss = (dqn_bc_ce * reward_w).sum() / reward_w.sum().clamp(min=1e-6)
            dqn_bc_scale = float(dqn_idx.numel()) / float(B)
            weighted_loss = weighted_loss + (dqn_bc_w * dqn_bc_scale) * dqn_bc_loss

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
    grad_norm = None
    if use_amp and scaler is not None:
        scaler.scale(weighted_loss).backward()
        scaler.unscale_(agent.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(agent.online_net.parameters(), clip_norm)
        if not torch.isfinite(grad_norm):
            print(f"[WARN] Non-finite grad norm detected ({float(grad_norm):.4g}), skipping step")
            try:
                agent.optimizer.zero_grad(set_to_none=True)
            except TypeError:
                agent.optimizer.zero_grad()
            try:
                cur = float(scaler.get_scale())
                scaler.update(new_scale=max(1.0, cur * 0.5))
            except Exception:
                pass
            return None
        scaler.step(agent.optimizer)
        scaler.update()
    else:
        weighted_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(agent.online_net.parameters(), clip_norm)
        if not torch.isfinite(grad_norm):
            print(f"[WARN] Non-finite grad norm detected ({float(grad_norm):.4g}), skipping step")
            try:
                agent.optimizer.zero_grad(set_to_none=True)
            except TypeError:
                agent.optimizer.zero_grad()
            return None
        agent.optimizer.step()

    # ── Update priorities ───────────────────────────────────────────────
    td_errors = ce_loss.detach().cpu().numpy()
    if np.isfinite(td_errors).all():
        agent.memory.update_priorities(indices, td_errors)
    else:
        print("[WARN] Non-finite TD errors detected, skipping priority update")

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

        # Agreement metric: exact joint-action match rate.
        with torch.no_grad():
            pred = q_all.argmax(dim=1)
            move_pred = move_scores.argmax(dim=1)
            fire_pred = fire_scores.argmax(dim=1)
            metrics.last_q_mean = float(q_all.mean().item())
            agree = (pred == actions_t).float().mean().item()
            move_agree = (move_pred == move_targets).float().mean().item()
            fire_agree = (fire_pred == fire_targets).float().mean().item()
            metrics.last_agreement = agree
            metrics.last_move_agreement = move_agree
            metrics.last_fire_agreement = fire_agree
            # Debug: print once every 1000 steps only in verbose mode.
            if getattr(metrics, "verbose_mode", False) and agent.training_steps % 1000 == 0:
                print(
                    f"[DEBUG] step={agent.training_steps} agree={agree:.4f} "
                    f"move_agree={move_agree:.4f} fire_agree={fire_agree:.4f} "
                    f"pred_sample={pred[:5].tolist()} act_sample={actions_t[:5].tolist()}"
                )

        if hasattr(metrics, "agree_sum_interval"):
            metrics.agree_sum_interval += agree
            metrics.agree_count_interval += 1
        if hasattr(metrics, "loss_sum_interval"):
            metrics.loss_sum_interval += loss_val
            metrics.loss_count_interval += 1
    except Exception as e:
        import traceback
        print(f"[train_step] metrics exception: {e}")
        traceback.print_exc()

    return loss_val
