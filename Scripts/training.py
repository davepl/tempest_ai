#!/usr/bin/env python3
"""Simplified training loop for the Tempest Discrete DQN agent (2-head)."""

import math
import time
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

def train_step(agent):
    """Run a single optimizer step for the DiscreteDQN agent."""
    if not getattr(metrics, "training_enabled", True) or not agent.training_enabled:
        return None

    if len(agent.memory) < agent.batch_size:
        return None

    # Sample from StratifiedReplayBuffer
    # Returns: (states, fz_idxs, sp_idxs, rewards, next_states, dones, actors)
    batch = agent.memory.sample(agent.batch_size, metrics.expert_ratio)
    if batch is None:
        return None

    if len(batch) == 7:
        states, fz_idxs, sp_idxs, rewards, next_states, dones, actors = batch
    else:
        # Backward-compat fallback if replay buffer doesn't provide actor tags
        states, fz_idxs, sp_idxs, rewards, next_states, dones = batch
        actors = None

    # Convert to tensors
    states = torch.from_numpy(states).float().to(device)
    fz_idxs = torch.from_numpy(fz_idxs).long().unsqueeze(1).to(device)
    sp_idxs = torch.from_numpy(sp_idxs).long().unsqueeze(1).to(device)
    rewards = torch.from_numpy(rewards).float().unsqueeze(1).to(device)
    next_states = torch.from_numpy(next_states).float().to(device)
    dones = torch.from_numpy(dones).float().unsqueeze(1).to(device)

    agent.qnetwork_local.train()
    
    # Forward pass
    q_fz, q_sp = agent.qnetwork_local(states)
    
    # Double DQN Targets
    with torch.no_grad():
        # Get best actions from local network
        next_q_fz_local, next_q_sp_local = agent.qnetwork_local(next_states)
        best_fz_actions = next_q_fz_local.argmax(dim=1, keepdim=True)
        best_sp_actions = next_q_sp_local.argmax(dim=1, keepdim=True)
        
        # Get values from target network
        next_q_fz_target, next_q_sp_target = agent.qnetwork_target(next_states)
        next_val_fz = next_q_fz_target.gather(1, best_fz_actions)
        next_val_sp = next_q_sp_target.gather(1, best_sp_actions)
        
        # Compute targets
        target_fz = rewards + (1.0 - dones) * agent.gamma * next_val_fz
        target_sp = rewards + (1.0 - dones) * agent.gamma * next_val_sp

        # Clip targets if configured
        td_clip = getattr(RL_CONFIG, 'td_target_clip', None)
        if td_clip is not None:
            target_fz = torch.clamp(target_fz, -td_clip, td_clip)
            target_sp = torch.clamp(target_sp, -td_clip, td_clip)

    # Compute Losses
    loss_fz = F.smooth_l1_loss(q_fz.gather(1, fz_idxs), target_fz)
    loss_sp = F.smooth_l1_loss(q_sp.gather(1, sp_idxs), target_sp)
    
    w_disc = float(getattr(RL_CONFIG, 'discrete_loss_weight', 1.0) or 1.0)
    total_loss = w_disc * (loss_fz + loss_sp)

    # Optional expert imitation losses (helps bootstrap spinner policy)
    supervised_loss_item = 0.0
    spinner_loss_item = 0.0
    try:
        w_sup = float(getattr(RL_CONFIG, "expert_supervision_weight", 0.0) or 0.0)
        w_spin = float(getattr(RL_CONFIG, "spinner_supervision_weight", w_sup) or 0.0)
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

    w_sup_eff = w_sup * sup_scale
    w_spin_eff = w_spin * sup_scale

    expert_mask = None
    if actors is not None and (w_sup_eff > 0.0 or w_spin_eff > 0.0):
        try:
            actors_np = np.array(actors, dtype=object)
            expert_mask_np = actors_np == "expert"
            if expert_mask_np.any():
                expert_mask = torch.from_numpy(expert_mask_np.astype(np.bool_)).to(device)
                expert_idx = torch.nonzero(expert_mask, as_tuple=False).squeeze(1)
                if expert_idx.numel() > 0:
                    if w_sup_eff > 0.0:
                        ce_fz = F.cross_entropy(q_fz[expert_idx], fz_idxs.squeeze(1)[expert_idx])
                        supervised_term = w_sup_eff * ce_fz
                        total_loss = total_loss + supervised_term
                        supervised_loss_item += float(supervised_term.item())
                    if w_spin_eff > 0.0:
                        ce_sp = F.cross_entropy(q_sp[expert_idx], sp_idxs.squeeze(1)[expert_idx])
                        spinner_term = w_spin_eff * ce_sp
                        total_loss = total_loss + spinner_term
                        spinner_loss_item = float(spinner_term.item())
                        supervised_loss_item += spinner_loss_item
        except Exception:
            expert_mask = None

    # Optimize
    try:
        agent.optimizer.zero_grad(set_to_none=True)
    except TypeError:
        agent.optimizer.zero_grad()
    total_loss.backward()

    clip_norm = float(getattr(RL_CONFIG, "grad_clip_norm", 10.0) or 10.0)
    grad_norm = torch.nn.utils.clip_grad_norm_(agent.qnetwork_local.parameters(), clip_norm)
    agent.optimizer.step()

    # Target network update strategy
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

    # Metrics
    try:
        metrics.total_training_steps += 1
        if hasattr(metrics, 'training_steps_interval'):
            metrics.training_steps_interval += 1
        try:
            metrics.memory_buffer_size = len(agent.memory)
        except Exception:
            pass
            
        metrics.losses.append(total_loss.item())
        metrics.last_d_loss = total_loss.item()
        grad_norm_val = float(grad_norm.item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
        metrics.last_grad_norm = grad_norm_val
        metrics.last_clip_delta = max(0.0, grad_norm_val - clip_norm)
        metrics.last_supervised_loss = float(supervised_loss_item)
        metrics.last_spinner_loss = float(spinner_loss_item)
        
        # Calculate agreement (accuracy)
        with torch.no_grad():
            pred_fz = q_fz.argmax(dim=1, keepdim=True)
            pred_sp = q_sp.argmax(dim=1, keepdim=True)
            agree_fz = (pred_fz == fz_idxs).float().mean().item()
            agree_sp = (pred_sp == sp_idxs).float().mean().item()
            
            metrics.agreement_rate = (agree_fz + agree_sp) / 2.0
            # Store per-head agreement for debugging (not displayed by default)
            metrics.agreement_rate_fz = float(agree_fz)
            metrics.agreement_rate_sp = float(agree_sp)
            
            # Update interval stats for display
            if hasattr(metrics, 'agree_sum_interval'):
                metrics.agree_sum_interval += metrics.agreement_rate
                metrics.agree_count_interval += 1
            
            if hasattr(metrics, 'loss_sum_interval'):
                metrics.loss_sum_interval += total_loss.item()
                metrics.loss_count_interval += 1

            # Per-actor batch composition diagnostics (helps spot “no expert in batch” issues)
            if actors is not None:
                try:
                    actors_np = np.array(actors, dtype=object)
                    n_expert = int((actors_np == "expert").sum())
                    n_dqn = int((actors_np == "dqn").sum())
                    metrics.batch_n_expert = n_expert
                    metrics.batch_n_dqn = n_dqn
                    metrics.batch_frac_dqn = (n_dqn / max(1, len(actors_np)))
                except Exception:
                    pass
            
    except Exception:
        pass

    return total_loss.item()
