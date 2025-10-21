#!/usr/bin/env python3
"""Training logic for hybrid DQN agent."""

import time
import torch
import torch.nn.functional as F
import numpy as np

# Import required modules
try:
    from config import RL_CONFIG, metrics
except ImportError:
    from Scripts.config import RL_CONFIG, metrics

# Define device locally to avoid circular import
if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def train_step(agent):
    """Perform one optimizer update for hybrid DQN.

    Simplified version focusing on core training logic.
    """
    # Global training gate
    if not getattr(metrics, 'training_enabled', True) or not agent.training_enabled:
        return None

    # Post-load burn-in check
    try:
        loaded_fc = int(getattr(metrics, 'loaded_frame_count', 0) or 0)
        require_new = int(getattr(RL_CONFIG, 'min_new_frames_after_load_to_train', 0) or 0)
        if loaded_fc > 0 and (metrics.frame_count - loaded_fc) < require_new:
            return None
    except Exception:
        pass

    # Require minimum data
    if len(agent.memory) < agent.batch_size:
        return None

    # Sample batch
    batch = agent.memory.sample(agent.batch_size)
    if batch is None:
        return None

    states, discrete_actions, continuous_actions, rewards, next_states, dones, actors, horizons = batch

    # Actor composition analysis
    try:
        actor_dqn_mask = np.array([a == 'dqn' for a in actors], dtype=bool)
        actor_expert_mask = np.array([a == 'expert' for a in actors], dtype=bool)
        n_dqn = actor_dqn_mask.sum()
        n_expert = actor_expert_mask.sum()

        # Store batch metrics
        metrics.batch_frac_dqn = float(n_dqn / len(actors)) if len(actors) > 0 else 0.0
        metrics.batch_n_dqn = int(n_dqn)
        metrics.batch_n_expert = int(n_expert)
    except Exception:
        actor_dqn_mask = None
        actor_expert_mask = None

    # Forward pass
    discrete_q_pred, continuous_pred = agent.qnetwork_local(states)
    
    # NOTE: Do NOT clamp predictions before computing loss. Clamping here zeroes gradients
    # outside the bounds and stalls learning. Keep predictions unconstrained for loss,
    # guard stability via target clipping and gradient clipping instead.
    max_q = getattr(RL_CONFIG, 'max_q_value', None)

    discrete_q_selected = discrete_q_pred.gather(1, discrete_actions)

    # Target computation with Double DQN
    with torch.no_grad():
        # Select actions with local network (no clamp needed for argmax)
        next_q_local, _ = agent.qnetwork_local(next_states)
        
        best_actions = next_q_local.max(1)[1].unsqueeze(1)

        # Evaluate with target network
        next_q_target, _ = agent.qnetwork_target(next_states)
        
        # Optional clamp on target values to bound bootstrap magnitude
        if max_q is not None:
            next_q_target = next_q_target.clamp(-float(max_q), float(max_q))
        
        discrete_q_next_max = next_q_target.gather(1, best_actions)

        # Apply n-step returns and build TD targets
        gamma_h = torch.pow(agent.gamma, horizons.float())
        discrete_targets = rewards + (gamma_h * discrete_q_next_max * (1 - dones))

        # Telemetry: record n-step horizon stats and TD target scale for debugging
        try:
            # Basic horizon stats
            metrics.horizon_mean = float(horizons.mean().item())
            metrics.horizon_min = float(horizons.min().item())
            metrics.horizon_max = float(horizons.max().item())
            # Effective discount applied on bootstrap
            metrics.gamma_h_mean = float(gamma_h.mean().item())

            # TD target scale (helps spot clipping or saturation)
            metrics.target_mean = float(discrete_targets.mean().item())
            metrics.target_min = float(discrete_targets.min().item())
            metrics.target_max = float(discrete_targets.max().item())
        except Exception:
            pass

        # Optional TD target clipping (record clipped fraction for diagnostics)
        td_clip = getattr(RL_CONFIG, 'td_target_clip', None)
        if td_clip is not None:
            try:
                tgt_pre = discrete_targets
                clipped_low = (tgt_pre < -float(td_clip)).float().mean().item()
                clipped_high = (tgt_pre > float(td_clip)).float().mean().item()
                metrics.target_clip_frac_low = float(clipped_low)
                metrics.target_clip_frac_high = float(clipped_high)
            except Exception:
                pass
            discrete_targets = discrete_targets.clamp(-float(td_clip), float(td_clip))

        # Simplified advantage weighting for continuous actions
        advantage_weights = torch.ones_like(rewards)
        if actor_dqn_mask is not None and actor_expert_mask is not None:
            try:
                # Separate advantage computation per actor type
                if n_dqn > 1:
                    dqn_rewards = rewards[torch.from_numpy(actor_dqn_mask).to(device)]
                    dqn_mean = dqn_rewards.mean()
                    dqn_std = dqn_rewards.std() + 1e-8
                    dqn_advantages = (dqn_rewards - dqn_mean) / dqn_std
                    dqn_weights = torch.exp(dqn_advantages * 0.5).clamp(0.1, 5.0)
                    advantage_weights[torch.from_numpy(actor_dqn_mask).to(device)] = dqn_weights

                if n_expert > 1:
                    exp_rewards = rewards[torch.from_numpy(actor_expert_mask).to(device)]
                    exp_mean = exp_rewards.mean()
                    exp_std = exp_rewards.std() + 1e-8
                    exp_advantages = (exp_rewards - exp_mean) / exp_std
                    exp_weights = torch.exp(exp_advantages * 0.5).clamp(0.1, 5.0)
                    advantage_weights[torch.from_numpy(actor_expert_mask).to(device)] = exp_weights
            except Exception:
                pass

        # Continuous targets with expert supervision
        continuous_targets = continuous_actions.clone()

        # Expert spinner annealing (simplified)
        try:
            anneal_frames = float(getattr(RL_CONFIG, 'continuous_expert_weight_frames', 1_000_000) or 1_000_000)
            fc = float(getattr(metrics, 'frame_count', 0))
            if anneal_frames > 0:
                prog = min(1.0, fc / anneal_frames)
                expert_w = 1.0 + (0.5 - 1.0) * prog  # Anneal from 1.0 to 0.5
            else:
                expert_w = 0.5
            
            # Track dynamic expert weight for metrics
            metrics.dynamic_expert_weight = expert_w

            if actor_expert_mask is not None and actor_expert_mask.any():
                exp_mask = torch.from_numpy(actor_expert_mask).to(device)
                blended = (expert_w * continuous_actions[exp_mask]) + ((1.0 - expert_w) * continuous_pred[exp_mask])
                continuous_targets[exp_mask] = blended
        except Exception:
            pass

    # Loss computation
    w_cont = float(getattr(RL_CONFIG, 'continuous_loss_weight', 1.0) or 1.0)
    w_disc = float(getattr(RL_CONFIG, 'discrete_loss_weight', 1.0) or 1.0)

    # Discrete TD loss (respect configured loss type)
    loss_type = str(getattr(RL_CONFIG, 'loss_type', 'huber') or 'huber').lower()
    if loss_type == 'mse':
        d_loss_td = F.mse_loss(discrete_q_selected, discrete_targets, reduction='mean')
        d_loss_td_per = F.mse_loss(discrete_q_selected, discrete_targets, reduction='none')
    else:
        d_loss_td = F.huber_loss(discrete_q_selected, discrete_targets, reduction='mean')
        d_loss_td_per = F.huber_loss(discrete_q_selected, discrete_targets, reduction='none')
    
    # Behavioral cloning loss for expert actions
    bc_loss = torch.tensor(0.0, device=device)
    # Gate BC by config flag and optional Q-filter margin
    if bool(getattr(RL_CONFIG, 'use_behavioral_cloning', False)):
        if actor_expert_mask is not None and n_expert > 0:
            bc_weight = float(getattr(RL_CONFIG, 'discrete_bc_weight', 0.0))
            if bc_weight > 0.0:
                exp_mask = torch.from_numpy(actor_expert_mask).to(device)
                expert_q_values = discrete_q_pred[exp_mask]  # logits for CE
                expert_actions = discrete_actions[exp_mask]

                # Optional Q-filter: only apply BC where expert action is near-best
                try:
                    margin = float(getattr(RL_CONFIG, 'bc_q_filter_margin', 0.0) or 0.0)
                except Exception:
                    margin = 0.0
                if expert_q_values.numel() > 0:
                    if margin <= 0.0:
                        # Strict: only if expert is argmax
                        q_max, a_max = expert_q_values.max(dim=1)
                        allow = (a_max == expert_actions.squeeze(1))
                    else:
                        # Lenient: allow when Q(exp) >= maxQ - margin
                        q_max, _ = expert_q_values.max(dim=1)
                        q_exp = expert_q_values.gather(1, expert_actions).squeeze(1)
                        allow = (q_exp >= (q_max - margin))
                    # Compute CE only on allowed samples
                    if allow.any():
                        bc_loss = bc_weight * F.cross_entropy(
                            expert_q_values[allow], expert_actions.squeeze(1)[allow]
                        )
            
    # Combined discrete loss: TD learning + behavioral cloning
    d_loss = d_loss_td + bc_loss

    # Continuous loss with advantage weighting
    c_loss_raw = F.mse_loss(continuous_pred, continuous_targets, reduction='none')
    c_loss = (c_loss_raw * advantage_weights).mean()

    # Combine losses
    total_loss = (w_disc * d_loss) + (w_cont * c_loss)

    # Backward pass
    agent.optimizer.zero_grad(set_to_none=True)
    total_loss.backward()

    # Gradient clipping and tracking
    grad_norm = torch.nn.utils.clip_grad_norm_(agent.qnetwork_local.parameters(), 10.0)
    
    # Record gradient norm
    try:
        metrics.last_grad_norm = float(grad_norm.item())
    except Exception:
        pass

    # Optimizer step
    agent.optimizer.step()

    # Update counters
    agent.training_steps += 1
    try:
        metrics.total_training_steps += 1
        metrics.training_steps_interval += 1
        metrics.memory_buffer_size = len(agent.memory)
    except Exception:
        pass

    # Track losses
    try:
        loss_val = float(total_loss.item())
        metrics.losses.append(loss_val)
        metrics.loss_sum_interval += loss_val
        metrics.loss_count_interval += 1

        metrics.last_d_loss = float((w_disc * d_loss).item())
        metrics.last_c_loss = float((w_cont * c_loss).item())
        try:
            metrics.last_bc_loss = float(bc_loss.item()) if isinstance(bc_loss, torch.Tensor) else float(bc_loss)
        except Exception:
            pass

        # Accumulate interval losses
        try:
            metrics.d_loss_sum_interval += metrics.last_d_loss
            metrics.d_loss_count_interval += 1
            metrics.c_loss_sum_interval += metrics.last_c_loss
            metrics.c_loss_count_interval += 1
            metrics.bc_loss_sum_interval += metrics.last_bc_loss
            metrics.bc_loss_count_interval += 1
        except Exception:
            pass
        
        # Track batch terminal state fraction
        try:
            terminal_count = dones.sum().item()
            batch_size_actual = dones.shape[0]
            metrics.batch_done_frac = terminal_count / batch_size_actual if batch_size_actual > 0 else 0.0
        except Exception:
            metrics.batch_done_frac = 0.0

        # TD-error diagnostics and per-actor breakdown
        try:
            # Detach to compute stats only
            td_err = (discrete_q_selected.detach() - discrete_targets.detach()).abs()
            metrics.td_err_mean = float(td_err.mean().item())
            # Percentiles
            td_cpu = td_err.view(-1).detach().cpu()
            metrics.td_err_p90 = float(torch.quantile(td_cpu, torch.tensor(0.90)).item()) if td_cpu.numel() > 0 else 0.0
            metrics.td_err_p99 = float(torch.quantile(td_cpu, torch.tensor(0.99)).item()) if td_cpu.numel() > 0 else 0.0

            # Per-actor TD loss means
            if actor_dqn_mask is not None and actor_expert_mask is not None:
                d_per = d_loss_td_per.detach().view(-1)
                mask_dqn = torch.from_numpy(actor_dqn_mask).to(d_per.device)
                mask_exp = torch.from_numpy(actor_expert_mask).to(d_per.device)
                if mask_dqn.any():
                    metrics.d_loss_mean_dqn = float(d_per[mask_dqn].mean().item())
                if mask_exp.any():
                    metrics.d_loss_mean_expert = float(d_per[mask_exp].mean().item())
                # Selected Q and targets per-actor
                q_sel = discrete_q_selected.detach().view(-1)
                tgt = discrete_targets.detach().view(-1)
                if mask_dqn.any():
                    metrics.q_sel_mean_dqn = float(q_sel[mask_dqn].mean().item())
                    metrics.q_tgt_mean_dqn = float(tgt[mask_dqn].mean().item())
                if mask_exp.any():
                    metrics.q_sel_mean_expert = float(q_sel[mask_exp].mean().item())
                    metrics.q_tgt_mean_expert = float(tgt[mask_exp].mean().item())
        except Exception:
            pass
        
        # Track agreement metrics (how often taken actions match current policy)
        try:
            with torch.no_grad():
                # Get current policy predictions
                network_output = agent.qnetwork_local(states)
                
                # Handle both hybrid (tuple) and discrete-only (tensor) architectures
                if isinstance(network_output, tuple):
                    curr_discrete_q, curr_continuous_pred = network_output
                else:
                    curr_discrete_q = network_output
                    curr_continuous_pred = None
                
                # Filter to only DQN-generated actions for agreement calculation
                dqn_indices = [i for i, actor in enumerate(actors) if actor == 'dqn']
                dqn_count = len(dqn_indices)
                
                if dqn_count > 0:
                    # Convert dqn_indices list to tensor for proper indexing
                    dqn_tensor = torch.tensor(dqn_indices, dtype=torch.long, device=curr_discrete_q.device)
                    
                    # Discrete agreement: check if argmax matches taken action (DQN only)
                    curr_discrete_actions = curr_discrete_q.argmax(dim=1, keepdim=True)
                    curr_dqn = curr_discrete_actions[dqn_tensor]
                    actions_dqn = discrete_actions[dqn_tensor]
                    
                    dqn_discrete_matches = (curr_dqn == actions_dqn).float()
                    discrete_agree_frac = dqn_discrete_matches.mean().item()
                    
                    # Continuous agreement (if hybrid network)
                    if curr_continuous_pred is not None:
                        continuous_diff = torch.abs(curr_continuous_pred[dqn_tensor] - continuous_actions[dqn_tensor])
                        dqn_continuous_matches = (continuous_diff < 0.1).float()
                        continuous_agree_frac = dqn_continuous_matches.mean().item()
                    else:
                        continuous_agree_frac = 0.0
                else:
                    # Fallback if no DQN actions in batch
                    discrete_agree_frac = 0.0
                    continuous_agree_frac = 0.0
                
                # Update interval accumulators
                metrics.agree_sum_interval += discrete_agree_frac
                metrics.agree_count_interval += 1
                if curr_continuous_pred is not None:
                    metrics.spinner_agree_sum_interval += continuous_agree_frac
                    metrics.spinner_agree_count_interval += 1
        except Exception as e:
            # Debug: print if agreement calculation fails
            print(f"Warning: Agreement calculation failed: {e}")
            pass
    except Exception:
        pass

    # Target network update
    if getattr(RL_CONFIG, 'use_soft_target_update', False):
        tau = float(getattr(RL_CONFIG, 'soft_target_tau', 0.005) or 0.005)
        with torch.no_grad():
            for tgt_p, src_p in zip(agent.qnetwork_target.parameters(), agent.qnetwork_local.parameters()):
                tgt_p.data.mul_(1.0 - tau).add_(src_p.data, alpha=tau)
    else:
        if agent.training_steps % RL_CONFIG.target_update_freq == 0:
            agent.qnetwork_target.load_state_dict(agent.qnetwork_local.state_dict())

    return float(total_loss.item())