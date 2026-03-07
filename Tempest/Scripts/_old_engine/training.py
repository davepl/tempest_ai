#!/usr/bin/env python3
"""Joint-action Double DQN training loop for Tempest."""

import time
from contextlib import nullcontext
import numpy as np
import torch
import torch.nn.functional as F

try:
    from config import RL_CONFIG, metrics
except ImportError:
    from Scripts.config import RL_CONFIG, metrics

if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

NUM_SPINNER_BUCKETS = 64

def _expert_mask_from_actors(actors):
    """
    Build a boolean expert mask from replay actor tags.
    Supports:
      - uint8/bool flags (1=expert, 0=dqn) from current replay
      - string tags ("expert"/"dqn") from legacy replay
    """
    if actors is None:
        return None

    arr = np.asarray(actors)
    if arr.size == 0:
        return np.zeros((0,), dtype=bool)

    try:
        if np.issubdtype(arr.dtype, np.number) or arr.dtype == np.bool_:
            return arr.astype(np.uint8, copy=False) != 0
    except Exception:
        pass

    if arr.dtype.kind == "S":
        try:
            arr = arr.astype("U", copy=False)
        except Exception:
            pass
    if arr.dtype.kind == "U":
        lowered = np.char.lower(arr)
        return (lowered == "expert") | (lowered == "1") | (lowered == "true")

    # Object fallback for mixed/older actor payloads.
    flat = arr.ravel()
    out = np.zeros(flat.shape, dtype=bool)
    for i, val in enumerate(flat):
        if isinstance(val, (bytes, bytearray)):
            try:
                val = val.decode("utf-8", errors="ignore")
            except Exception:
                pass
        if isinstance(val, str):
            sval = val.strip().lower()
            out[i] = (sval == "expert") or (sval == "1") or (sval == "true")
        else:
            out[i] = bool(val == 1 or val is True)
    return out.reshape(arr.shape)

def _unpack_batch(batch):
    """
    Normalize replay outputs to:
      (states, action_idxs, rewards, next_states, dones, horizons, actors)
    Handles both the current joint-action schema and older replay tuples.
    """
    if batch is None:
        return None

    # New schema:
    # (states, action_idxs, rewards, next_states, dones, horizons, actors)
    if len(batch) == 7 and isinstance(batch[3], np.ndarray) and batch[3].ndim == 2:
        states, action_idxs, rewards, next_states, dones, horizons, actors = batch
        return states, action_idxs, rewards, next_states, dones, horizons, actors

    # Legacy two-head schema:
    # (states, fz_idxs, sp_idxs, rewards, next_states, dones, actors)
    if len(batch) == 7:
        states, fz_idxs, sp_idxs, rewards, next_states, dones, actors = batch
        action_idxs = (np.array(fz_idxs, dtype=np.int64) * NUM_SPINNER_BUCKETS) + np.array(sp_idxs, dtype=np.int64)
        horizons = np.ones_like(action_idxs, dtype=np.int64)
        return states, action_idxs, rewards, next_states, dones, horizons, actors

    # Very old fallback (no actor tags)
    if len(batch) == 6:
        states, action_idxs, rewards, next_states, dones, horizons = batch
        actors = None
        return states, action_idxs, rewards, next_states, dones, horizons, actors

    return None

def train_step(agent):
    """Run a single optimizer step for the joint-action DQN agent."""
    if not getattr(metrics, "training_enabled", True) or not agent.training_enabled:
        return None

    if len(agent.memory) < agent.batch_size:
        return None

    batch = agent.memory.sample(agent.batch_size, metrics.expert_ratio)
    unpacked = _unpack_batch(batch)
    if unpacked is None:
        return None

    states, action_idxs, rewards, next_states, dones, horizons, actors = unpacked

    # Convert to tensors
    states_t = torch.from_numpy(np.asarray(states, dtype=np.float32)).to(device)
    action_t = torch.from_numpy(np.asarray(action_idxs, dtype=np.int64)).unsqueeze(1).to(device)
    rewards_t = torch.from_numpy(np.asarray(rewards, dtype=np.float32)).unsqueeze(1).to(device)
    next_states_t = torch.from_numpy(np.asarray(next_states, dtype=np.float32)).to(device)
    dones_t = torch.from_numpy(np.asarray(dones, dtype=np.float32)).unsqueeze(1).to(device)
    horizons_t = torch.from_numpy(np.asarray(horizons, dtype=np.float32)).unsqueeze(1).to(device)

    agent.qnetwork_local.train()

    use_amp = bool(getattr(agent, "use_amp", False)) and (device.type == "cuda")
    scaler = getattr(agent, "grad_scaler", None)
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()

    # Forward
    max_q_cfg = getattr(RL_CONFIG, "max_q_value", None)
    try:
        max_q = float(max_q_cfg) if max_q_cfg is not None else None
    except Exception:
        max_q = None
    if max_q is not None and max_q <= 0.0:
        max_q = None

    with autocast_ctx:
        q_values = agent.qnetwork_local(states_t)
        if max_q is not None:
            q_values = torch.clamp(q_values, -max_q, max_q)
        q_selected = q_values.gather(1, action_t)

        # Double DQN Targets (horizon-aware for n-step)
        with torch.no_grad():
            next_q_local = agent.qnetwork_local(next_states_t)
            if max_q is not None:
                next_q_local = torch.clamp(next_q_local, -max_q, max_q)
            best_actions = next_q_local.argmax(dim=1, keepdim=True)
            next_q_target = agent.qnetwork_target(next_states_t)
            if max_q is not None:
                next_q_target = torch.clamp(next_q_target, -max_q, max_q)
            next_val = next_q_target.gather(1, best_actions)

            gamma_base = torch.full_like(horizons_t, float(agent.gamma))
            gamma_pow = torch.pow(gamma_base, horizons_t)
            target_q = rewards_t + (1.0 - dones_t) * gamma_pow * next_val

            td_clip = getattr(RL_CONFIG, "td_target_clip", None)
            if td_clip is not None:
                target_q = torch.clamp(target_q, -td_clip, td_clip)

        # Keep loss in fp32 for stability even when autocast is enabled.
        loss_q = F.smooth_l1_loss(q_selected.float(), target_q.float())
    w_disc = float(getattr(RL_CONFIG, "discrete_loss_weight", 1.0) or 1.0)
    total_loss = w_disc * loss_q

    # Optional conservative regularization and expert imitation.
    cql_loss_tensor = torch.zeros((), device=device, dtype=torch.float32)
    try:
        cql_alpha = float(getattr(RL_CONFIG, "cql_alpha", 0.0) or 0.0)
    except Exception:
        cql_alpha = 0.0
    if cql_alpha > 0.0:
        cql_loss = (torch.logsumexp(q_values.float(), dim=1, keepdim=True) - q_selected.float()).mean()
        cql_term = cql_alpha * cql_loss
        total_loss = total_loss + cql_term
        cql_loss_tensor = cql_term.detach().float()

    # Optional expert imitation on the joint action label.
    supervised_loss_tensor = torch.zeros((), device=device, dtype=torch.float32)
    try:
        w_sup = float(getattr(RL_CONFIG, "expert_supervision_weight", 0.0) or 0.0)
        w_spin = float(getattr(RL_CONFIG, "spinner_supervision_weight", 0.0) or 0.0)
    except Exception:
        w_sup = 0.0
        w_spin = 0.0

    sup_scale = 1.0
    try:
        decay_start = int(getattr(RL_CONFIG, "supervision_decay_start", 0) or 0)
        decay_frames = int(getattr(RL_CONFIG, "supervision_decay_frames", 1) or 1)
        decay_frames = max(1, decay_frames)
        min_sup = float(getattr(RL_CONFIG, "min_supervision_weight", 0.0) or 0.0)
        frame_now = int(getattr(metrics, "frame_count", 0))
        if frame_now > decay_start:
            progress = min(1.0, (frame_now - decay_start) / float(decay_frames))
            sup_scale = 1.0 - progress * (1.0 - min_sup)
    except Exception:
        sup_scale = 1.0

    w_bc = (w_sup + w_spin) * sup_scale
    if actors is not None and w_bc > 0.0:
        try:
            expert_mask_np = _expert_mask_from_actors(actors)
            if expert_mask_np is not None and expert_mask_np.any():
                expert_idx = torch.from_numpy(np.nonzero(expert_mask_np)[0].astype(np.int64)).to(device)
                if expert_idx.numel() > 0:
                    ce_joint = F.cross_entropy(q_values[expert_idx].float(), action_t.squeeze(1)[expert_idx])
                    sup_term = w_bc * ce_joint
                    total_loss = total_loss + sup_term
                    supervised_loss_tensor = sup_term.detach().float()
        except Exception:
            pass

    # Optimize
    try:
        agent.optimizer.zero_grad(set_to_none=True)
    except TypeError:
        agent.optimizer.zero_grad()

    clip_norm = float(getattr(RL_CONFIG, "grad_clip_norm", 10.0) or 10.0)
    if use_amp and scaler is not None:
        scaler.scale(total_loss).backward()
        scaler.unscale_(agent.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(agent.qnetwork_local.parameters(), clip_norm)
        scaler.step(agent.optimizer)
        scaler.update()
    else:
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(agent.qnetwork_local.parameters(), clip_norm)
        agent.optimizer.step()

    # Target update
    now = time.time()
    use_soft = bool(getattr(RL_CONFIG, "use_soft_target_update", True))
    if use_soft:
        tau = float(getattr(RL_CONFIG, "soft_target_tau", 1e-3) or 1e-3)
        tau = max(0.0, min(1.0, tau))
        for target_param, local_param in zip(agent.qnetwork_target.parameters(), agent.qnetwork_local.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
        try:
            metrics.last_target_update_step = int(getattr(metrics, "total_training_steps", 0))
            metrics.last_target_update_time = now
        except Exception:
            pass
    else:
        freq = int(getattr(RL_CONFIG, "target_update_freq", 0) or 0)
        if freq > 0 and (agent.training_steps % freq == 0):
            agent.qnetwork_target.load_state_dict(agent.qnetwork_local.state_dict())
            try:
                metrics.last_target_update_step = int(getattr(metrics, "total_training_steps", 0))
                metrics.last_target_update_time = now
            except Exception:
                pass

    agent.training_steps += 1
    try:
        agent.sync_inference_model(force=False)
    except Exception:
        pass

    # Metrics
    try:
        metrics.total_training_steps += 1
        if hasattr(metrics, "training_steps_interval"):
            metrics.training_steps_interval += 1
        try:
            metrics.memory_buffer_size = len(agent.memory)
        except Exception:
            pass

        metrics_interval = int(max(1, int(getattr(agent, "train_metrics_interval_steps", 1) or 1)))
        if (agent.training_steps % metrics_interval) == 0:
            loss_item = float(total_loss.detach().item())
            metrics.losses.append(loss_item)
            metrics.last_d_loss = loss_item
            grad_norm_val = float(grad_norm.item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
            metrics.last_grad_norm = grad_norm_val
            metrics.last_clip_delta = max(0.0, grad_norm_val - clip_norm)
            metrics.last_supervised_loss = float(supervised_loss_tensor.item())
            metrics.last_spinner_loss = 0.0
            metrics.last_cql_loss = float(cql_loss_tensor.item())

            with torch.no_grad():
                pred_action = q_values.argmax(dim=1)
                true_action = action_t.squeeze(1)

                agree_joint = (pred_action == true_action).float().mean().item()

                pred_fz = torch.div(pred_action, NUM_SPINNER_BUCKETS, rounding_mode='floor')
                pred_sp = pred_action % NUM_SPINNER_BUCKETS
                true_fz = torch.div(true_action, NUM_SPINNER_BUCKETS, rounding_mode='floor')
                true_sp = true_action % NUM_SPINNER_BUCKETS

                agree_fz = (pred_fz == true_fz).float().mean().item()
                agree_sp = (pred_sp == true_sp).float().mean().item()

                metrics.agreement_rate = (agree_fz + agree_sp) / 2.0
                metrics.agreement_rate_fz = float(agree_fz)
                metrics.agreement_rate_sp = float(agree_sp)
                metrics.agreement_rate_joint = float(agree_joint)

                if hasattr(metrics, "agree_sum_interval"):
                    metrics.agree_sum_interval += metrics.agreement_rate
                    metrics.agree_count_interval += 1

                if hasattr(metrics, "loss_sum_interval"):
                    metrics.loss_sum_interval += loss_item
                    metrics.loss_count_interval += 1

                if actors is not None:
                    try:
                        expert_mask_np = _expert_mask_from_actors(actors)
                        if expert_mask_np is not None:
                            n_expert = int(expert_mask_np.sum())
                            n_dqn = int(expert_mask_np.size - n_expert)
                        else:
                            n_expert = 0
                            n_dqn = 0
                        metrics.batch_n_expert = n_expert
                        metrics.batch_n_dqn = n_dqn
                        metrics.batch_frac_dqn = (n_dqn / max(1, n_dqn + n_expert))
                    except Exception:
                        pass
    except Exception:
        pass

    return 0.0
