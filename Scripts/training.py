#!/usr/bin/env python3
"""Simplified training loop for the Tempest hybrid DQN agent."""

import time
from typing import Sequence

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


def _to_tensor_bool(mask: Sequence[bool], length: int, *, device: torch.device) -> torch.Tensor:
    if len(mask) != length:
        raise ValueError("mask length mismatch")
    return torch.tensor(mask, dtype=torch.bool, device=device)


_SPINNER_SIGN_CACHE: dict[tuple[str, int], torch.Tensor] = {}


def _get_spinner_signs(device: torch.device, spinner_actions: int) -> torch.Tensor:
    """Return a tensor of spinner signs (-1, 0, 1) for each bucket on the requested device."""
    key = (device.type, int(getattr(device, "index", -1) or -1))
    cached = _SPINNER_SIGN_CACHE.get(key)
    if cached is None or cached.numel() < spinner_actions:
        levels = tuple(getattr(RL_CONFIG, "spinner_command_levels", (0,)))
        if not levels:
            levels = (0,)
        sign_tensor = torch.sign(torch.tensor(levels, dtype=torch.float32, device=device))
        if sign_tensor.numel() < spinner_actions:
            pad = torch.zeros(spinner_actions - sign_tensor.numel(), dtype=torch.float32, device=device)
            sign_tensor = torch.cat([sign_tensor, pad], dim=0)
        _SPINNER_SIGN_CACHE[key] = sign_tensor
        cached = sign_tensor
    return cached[:spinner_actions]


def compute_action_agreement(
    greedy_actions: torch.Tensor,
    taken_actions: torch.Tensor,
    spinner_actions: int,
    device: torch.device,
) -> torch.Tensor:
    """Return boolean tensor where True means actions agree under sign-aware spinner matching."""
    if greedy_actions.ndim > 1:
        greedy_flat = greedy_actions.reshape(-1)
    else:
        greedy_flat = greedy_actions
    if taken_actions.ndim > 1:
        taken_flat = taken_actions.reshape(-1)
    else:
        taken_flat = taken_actions

    exact_match = greedy_flat == taken_flat
    spinner_actions = max(1, int(spinner_actions))

    greedy_fire_zap = torch.div(greedy_flat, spinner_actions, rounding_mode="floor")
    taken_fire_zap = torch.div(taken_flat, spinner_actions, rounding_mode="floor")
    fire_zap_match = greedy_fire_zap == taken_fire_zap

    sign_tensor = _get_spinner_signs(device, spinner_actions)
    greedy_spinner_idx = torch.remainder(greedy_flat, spinner_actions).long()
    taken_spinner_idx = torch.remainder(taken_flat, spinner_actions).long()
    greedy_sign = sign_tensor[greedy_spinner_idx]
    taken_sign = sign_tensor[taken_spinner_idx]

    sign_match = greedy_sign == taken_sign
    return exact_match | (fire_zap_match & sign_match)


def train_step(agent):
    """Run a single optimizer step for the simplified hybrid agent."""
    if not getattr(metrics, "training_enabled", True) or not agent.training_enabled:
        return None

    if len(agent.memory) < agent.batch_size:
        return None

    batch = agent.memory.sample(agent.batch_size)
    if batch is None:
        return None

    (
        states,
        discrete_actions,
        rewards,
        next_states,
        dones,
        actors,
        horizons,
    ) = batch

    discrete_actions = discrete_actions.long()
    rewards = rewards.float()
    dones = dones.float()
    horizons = horizons.float()

    agent.qnetwork_local.train()

    # Forward pass for current states
    q_values = agent.qnetwork_local(states)
    selected_q = q_values.gather(1, discrete_actions)

    # Build Double DQN targets
    with torch.no_grad():
        next_q_local = agent.qnetwork_local(next_states)
        best_next_actions = next_q_local.argmax(dim=1, keepdim=True)

        next_q_target = agent.qnetwork_target(next_states)
        next_values = next_q_target.gather(1, best_next_actions)

        gamma_h = torch.pow(torch.full_like(horizons, agent.gamma), horizons)
        targets = rewards + (1.0 - dones) * gamma_h * next_values

    td_loss = F.smooth_l1_loss(selected_q, targets)

    expert_mask_np = [actor == "expert" for actor in actors]
    expert_mask = _to_tensor_bool(expert_mask_np, len(actors), device=states.device)

    w_disc = float(getattr(RL_CONFIG, "discrete_loss_weight", 1.0))
    total_loss = w_disc * td_loss

    supervised_loss_item = 0.0
    spinner_loss_item = 0.0
    w_sup = float(getattr(RL_CONFIG, "expert_supervision_weight", 0.0) or 0.0)
    w_spin = float(getattr(RL_CONFIG, "spinner_supervision_weight", w_sup) or 0.0)
    expert_any = bool(expert_mask.any().item())

    if expert_any and (w_sup > 0.0 or w_spin > 0.0):
        log_probs = F.log_softmax(q_values, dim=1)
        spinner_actions = max(1, int(getattr(agent, "spinner_actions", 1) or 1))
        fire_zap_actions = max(1, int(getattr(agent, "fire_zap_actions", 1) or 1))
        taken_flat = discrete_actions.squeeze(1)
        expert_indices = torch.nonzero(expert_mask, as_tuple=False).squeeze(1)

        if expert_indices.numel() > 0 and w_sup > 0.0:
            total_actions = log_probs.size(1)
            action_indices = torch.arange(total_actions, device=log_probs.device)
            fire_zap_indices = torch.div(action_indices, spinner_actions, rounding_mode="floor")
            fire_mask = ((fire_zap_indices >> 1) & 1).bool()
            zap_mask = (fire_zap_indices & 1).bool()

            log_prob_fire1 = torch.logsumexp(log_probs[:, fire_mask], dim=1)
            log_prob_fire0 = torch.logsumexp(log_probs[:, ~fire_mask], dim=1)
            log_prob_zap1 = torch.logsumexp(log_probs[:, zap_mask], dim=1)
            log_prob_zap0 = torch.logsumexp(log_probs[:, ~zap_mask], dim=1)

            fire_zap_taken = torch.div(taken_flat, spinner_actions, rounding_mode="floor")
            fire_targets = ((fire_zap_taken >> 1) & 1).float()
            zap_targets = (fire_zap_taken & 1).float()

            fire_log_prob = torch.where(fire_targets > 0.5, log_prob_fire1, log_prob_fire0)
            zap_log_prob = torch.where(zap_targets > 0.5, log_prob_zap1, log_prob_zap0)

            fire_loss = -fire_log_prob[expert_indices]
            zap_loss = -zap_log_prob[expert_indices]
            imitation_loss = (fire_loss.mean() + zap_loss.mean()) * 0.5
            supervised_term = imitation_loss * w_sup
            total_loss = total_loss + supervised_term
            supervised_loss_item += float(supervised_term.item())

        if expert_indices.numel() > 0 and w_spin > 0.0:
            log_probs_reshaped = log_probs.reshape(-1, fire_zap_actions, spinner_actions)
            spinner_log_probs = torch.logsumexp(log_probs_reshaped, dim=1)
            spinner_taken = torch.remainder(taken_flat, spinner_actions).long()
            spinner_log_prob = spinner_log_probs.gather(1, spinner_taken.unsqueeze(1)).squeeze(1)
            spinner_loss = -spinner_log_prob[expert_indices].mean() * w_spin
            total_loss = total_loss + spinner_loss
            spinner_loss_item = float(spinner_loss.item())
            supervised_loss_item += spinner_loss_item

    agent.optimizer.zero_grad(set_to_none=True)
    total_loss.backward()

    clip_norm = float(getattr(RL_CONFIG, "grad_clip_norm", 10.0) or 10.0)
    grad_norm = torch.nn.utils.clip_grad_norm_(agent.qnetwork_local.parameters(), clip_norm)

    agent.optimizer.step()

    agent.training_steps += 1
    agent._apply_target_update()

    loss_value = float(total_loss.item())

    # Metric updates ---------------------------------------------------------
    try:
        metrics.total_training_steps += 1
        metrics.training_steps_interval += 1
        metrics.memory_buffer_size = len(agent.memory)

        metrics.losses.append(loss_value)
        metrics.loss_sum_interval += loss_value
        metrics.loss_count_interval += 1

        metrics.last_d_loss = float((w_disc * td_loss).item())
        metrics.last_supervised_loss = float(supervised_loss_item)
        metrics.last_spinner_loss = float(spinner_loss_item)

        metrics.d_loss_sum_interval += metrics.last_d_loss
        metrics.d_loss_count_interval += 1

        grad_norm_val = float(grad_norm.item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
        metrics.last_grad_norm = grad_norm_val
        metrics.last_clip_delta = max(0.0, grad_norm_val - clip_norm)

        metrics.batch_done_frac = float(dones.mean().item())
        metrics.batch_h_mean = float(horizons.mean().item())

        actors_np = np.array(actors)
        dqn_mask_np = actors_np == "dqn"
        expert_mask_np = actors_np == "expert"
        n_dqn = int(dqn_mask_np.sum())
        n_expert = int(expert_mask_np.sum())

        metrics.batch_n_dqn = n_dqn
        metrics.batch_n_expert = n_expert
        metrics.batch_frac_dqn = (n_dqn / len(actors_np)) if len(actors_np) else 0.0

        rewards_np = rewards.detach().cpu().numpy().reshape(-1)
        if n_dqn > 0:
            metrics.reward_mean_dqn = float(rewards_np[dqn_mask_np].mean())
        if n_expert > 0:
            metrics.reward_mean_expert = float(rewards_np[expert_mask_np].mean())

        with torch.no_grad():
            policy_q = q_values.detach()
            greedy_actions = policy_q.argmax(dim=1, keepdim=True)
            spinner_actions = getattr(agent, "spinner_actions", 1)
            combined_match = compute_action_agreement(
                greedy_actions,
                discrete_actions,
                spinner_actions,
                states.device,
            )
            action_matches = combined_match.float()

            if n_dqn > 0:
                dqn_indices = torch.tensor(np.nonzero(dqn_mask_np)[0], dtype=torch.long, device=states.device)
                agree_mean = float(action_matches[dqn_indices].mean().item())

                # Update agreement metrics atomically to avoid race-induced >100% readings
                with metrics.lock:
                    metrics.agree_sum_interval += agree_mean
                    metrics.agree_count_interval += 1

        metrics.last_optimizer_step_time = time.time()
    except Exception:
        pass
    # ------------------------------------------------------------------------

    return loss_value
