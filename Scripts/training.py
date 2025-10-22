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
        continuous_actions,
        rewards,
        next_states,
        dones,
        actors,
        horizons,
    ) = batch

    discrete_actions = discrete_actions.long()
    continuous_actions = continuous_actions.float()
    rewards = rewards.float()
    dones = dones.float()
    horizons = horizons.float()

    agent.qnetwork_local.train()

    # Forward pass for current states
    discrete_q, spinner_pred = agent.qnetwork_local(states)
    selected_q = discrete_q.gather(1, discrete_actions)

    # Build Double DQN targets
    with torch.no_grad():
        next_q_local, _ = agent.qnetwork_local(next_states)
        best_next_actions = next_q_local.argmax(dim=1, keepdim=True)

        next_q_target, _ = agent.qnetwork_target(next_states)
        next_values = next_q_target.gather(1, best_next_actions)

        gamma_h = torch.pow(torch.full_like(horizons, agent.gamma), horizons)
        targets = rewards + (1.0 - dones) * gamma_h * next_values

    td_loss = F.smooth_l1_loss(selected_q, targets)
    spinner_loss = F.mse_loss(spinner_pred, continuous_actions)

    expert_mask_np = [actor == "expert" for actor in actors]
    expert_mask = _to_tensor_bool(expert_mask_np, len(actors), device=states.device)

    if expert_mask.any():
        bc_loss_raw = F.cross_entropy(discrete_q[expert_mask], discrete_actions[expert_mask].squeeze(1))
    else:
        bc_loss_raw = torch.tensor(0.0, device=states.device)

    w_disc = float(getattr(RL_CONFIG, "discrete_loss_weight", 1.0))
    w_cont = float(getattr(RL_CONFIG, "continuous_loss_weight", 1.0))
    bc_weight = float(getattr(RL_CONFIG, "discrete_bc_weight", 0.0))

    total_loss = (w_disc * td_loss) + (w_cont * spinner_loss) + (bc_weight * bc_loss_raw)

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
        metrics.last_c_loss = float((w_cont * spinner_loss).item())
        metrics.last_bc_loss = float((bc_weight * bc_loss_raw).item())

        metrics.d_loss_sum_interval += metrics.last_d_loss
        metrics.d_loss_count_interval += 1
        metrics.c_loss_sum_interval += metrics.last_c_loss
        metrics.c_loss_count_interval += 1
        metrics.bc_loss_sum_interval += metrics.last_bc_loss
        metrics.bc_loss_count_interval += 1

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
            policy_q, policy_spinner = agent.qnetwork_local(states)
            greedy_actions = policy_q.argmax(dim=1, keepdim=True)
            action_matches = (greedy_actions == discrete_actions).float().squeeze(1)

            if n_dqn > 0:
                dqn_indices = torch.tensor(np.nonzero(dqn_mask_np)[0], dtype=torch.long, device=states.device)
                agree_mean = float(action_matches[dqn_indices].mean().item())

                spinner_diff = torch.abs(policy_spinner[dqn_indices] - continuous_actions[dqn_indices])
                spinner_agree_mean = float((spinner_diff < 0.1).float().mean().item())

                # Update agreement metrics atomically to avoid race-induced >100% readings
                with metrics.lock:
                    metrics.agree_sum_interval += agree_mean
                    metrics.agree_count_interval += 1
                    metrics.spinner_agree_sum_interval += spinner_agree_mean
                    metrics.spinner_agree_count_interval += 1

        metrics.last_optimizer_step_time = time.time()
    except Exception:
        pass
    # ------------------------------------------------------------------------

    return loss_value
