#!/usr/bin/env python3
# ==================================================================================================================
# ||  TEMPEST AI v2 • CONFIGURATION                                                                              ||
# ||  Rainbow-Attention engine with factored action heads                                                         ||
# ==================================================================================================================
"""Central configuration: server, RL hyper-parameters, metrics."""

if __name__ == "__main__":
    print("This is not the main application, run 'main.py' instead")
    exit(1)

import os, sys, time, threading, math
from dataclasses import dataclass, field
from typing import Deque
from collections import deque

IS_INTERACTIVE = sys.stdin.isatty()
RESET_METRICS = False
FORCE_FRESH_MODEL = False

MODEL_DIR = "models"
LATEST_MODEL_PATH = f"{MODEL_DIR}/tempest_model_latest.pt"

# ---------------------------------------------------------------------------
@dataclass
class ServerConfigData:
    host: str = "0.0.0.0"
    port: int = 9999
    max_clients: int = 36
    params_count: int = 195

SERVER_CONFIG = ServerConfigData()

# ---------------------------------------------------------------------------
@dataclass
class RLConfigData:
    # ── state / action ──────────────────────────────────────────────────
    state_size: int = SERVER_CONFIG.params_count

    # Factored action space  (4 fire/zap × 11 spinner = 44 actions)
    num_firezap_actions: int = 4
    spinner_command_levels: tuple[int, ...] = (0, 12, 9, 6, 3, 1, -1, -3, -6, -9, -12)

    @property
    def num_spinner_actions(self) -> int:
        return len(self.spinner_command_levels)

    @property
    def num_joint_actions(self) -> int:
        return self.num_firezap_actions * self.num_spinner_actions

    # ── network architecture ────────────────────────────────────────────
    trunk_hidden: int = 512
    trunk_layers: int = 3
    use_layer_norm: bool = True
    dropout: float = 0.0

    # Attention over enemy slots  (7 × 6)
    use_enemy_attention: bool = True
    enemy_slots: int = 7
    enemy_features: int = 6
    attn_heads: int = 4
    attn_dim: int = 64

    # Distributional C51
    # Support scaled to match Rainbow's 20:1 ratio (support range / reward_clip).
    # Old [-50,50] was 10:1 — too tight, causing Bellman target clipping on
    # kill rewards above 245 points.  New [-100,100] eliminates clipping for
    # kills under 490 pts and gives ~5× headroom for Q-value growth.
    use_distributional: bool = True
    num_atoms: int = 51
    v_min: float = -100.0
    v_max: float = 100.0

    use_dueling: bool = True

    # ── training ────────────────────────────────────────────────────────
    batch_size: int = 512
    lr: float = 6.25e-5
    lr_min: float = 6.25e-5              # Same as lr = fixed LR, no cosine restarts
    lr_warmup_steps: int = 5_000
    lr_cosine_period: int = 100_000        # (inert when lr == lr_min)
    gamma: float = 0.99
    n_step: int = 10                        # Doubled from 5 for wider death attribution window

    # Replay (PER with proportional priorities)
    memory_size: int = 2_000_000
    priority_alpha: float = 0.7
    priority_beta_start: float = 0.4
    priority_beta_frames: int = 10_000_000
    priority_eps: float = 1e-6
    min_replay_to_train: int = 10_000

    # Target network (soft Polyak averaging)
    target_update_period: int = 1
    target_tau: float = 0.001

    # Gradient
    grad_clip_norm: float = 10.0

    # ── exploration ─────────────────────────────────────────────────────
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_frames: int = 1_000_000
    epsilon: float = 1.0

    # Expert guidance
    expert_ratio_start: float = 0.50
    expert_ratio_end: float = 0.0
    expert_ratio_decay_frames: int = 5_000_000
    expert_ratio: float = 0.50
    # During tube zoom (gamestate 0x20), temporarily boost expert usage.
    expert_ratio_zoom_multiplier: float = 2.0
    expert_ratio_zoom_gamestate: int = 0x20

    # Expert BC
    expert_bc_weight: float = 1.0
    expert_bc_decay_start: int = 500_000
    expert_bc_decay_frames: int = 2_000_000
    # Keep a small non-zero floor to reduce late-training policy drift.
    expert_bc_min_weight: float = 0.1

    # ── reward ──────────────────────────────────────────────────────────
    obj_reward_scale: float = 0.01
    subj_reward_scale: float = 0.01
    reward_clip: float = 10.0
    death_reward_clip: float = 25.0        # 2.5× normal clip — death is special but doesn't crush C51 support

    # ── death attribution ───────────────────────────────────────────────
    death_priority_boost: float = 10.0     # Minimum PER priority for terminal transitions

    # ── inference ───────────────────────────────────────────────────────
    use_separate_inference_model: bool = True
    inference_on_cpu: bool = True
    inference_sync_steps: int = 100

    # ── background training ─────────────────────────────────────────────
    training_steps_per_cycle: int = 8
    save_interval: int = 10_000

    enable_amp: bool = True


RL_CONFIG = RLConfigData()

# ---------------------------------------------------------------------------
#  Metrics
# ---------------------------------------------------------------------------
@dataclass
class MetricsData:
    frame_count: int = 0
    total_controls: int = 0
    total_training_steps: int = 0
    memory_buffer_size: int = 0
    client_count: int = 0

    epsilon: float = RL_CONFIG.epsilon_start
    expert_ratio: float = RL_CONFIG.expert_ratio_start

    episode_rewards: Deque[float] = field(default_factory=lambda: deque(maxlen=50))
    dqn_rewards: Deque[float] = field(default_factory=lambda: deque(maxlen=50))
    expert_rewards: Deque[float] = field(default_factory=lambda: deque(maxlen=50))
    subj_rewards: Deque[float] = field(default_factory=lambda: deque(maxlen=50))
    obj_rewards: Deque[float] = field(default_factory=lambda: deque(maxlen=50))
    losses: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))

    fps: float = 0.0
    frames_last_second: int = 0
    last_fps_time: float = 0.0

    # Interval accumulators (reset each display row)
    loss_sum_interval: float = 0.0
    loss_count_interval: int = 0
    agree_sum_interval: float = 0.0
    agree_count_interval: int = 0
    reward_sum_interval: float = 0.0
    reward_count_interval: int = 0
    reward_sum_interval_dqn: float = 0.0
    reward_count_interval_dqn: int = 0
    reward_sum_interval_subj: float = 0.0
    reward_count_interval_subj: int = 0
    reward_sum_interval_obj: float = 0.0
    reward_count_interval_obj: int = 0
    training_steps_interval: int = 0
    frames_count_interval: int = 0
    episode_length_sum_interval: int = 0
    episode_length_count_interval: int = 0
    level_sum_interval: float = 0.0
    level_count_interval: int = 0

    total_inference_time: float = 0.0
    total_inference_requests: int = 0

    last_grad_norm: float = 0.0
    last_loss: float = 0.0
    last_q_mean: float = 0.0
    last_bc_loss: float = 0.0
    last_priority_mean: float = 0.0

    average_level: float = 0.0
    last_target_update_step: int = 0
    last_target_update_time: float = 0.0
    loaded_frame_count: int = 0

    # UI toggles
    override_expert: bool = False
    expert_mode: bool = False
    manual_expert_override: bool = False
    override_epsilon: bool = False
    manual_epsilon_override: bool = False
    training_enabled: bool = True
    verbose_mode: bool = False
    saved_expert_ratio: float = 0.50

    global_server: object = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    # ── helpers ─────────────────────────────────────────────────────────
    def update_frame_count(self, delta: int = 1):
        with self.lock:
            d = max(1, delta)
            self.frame_count += d
            self.frames_count_interval += d
            self.frames_last_second += d
            now = time.time()
            if self.last_fps_time == 0:
                self.last_fps_time = now
            elapsed = now - self.last_fps_time
            if elapsed >= 1.0:
                self.fps = self.frames_last_second / elapsed
                self.frames_last_second = 0
                self.last_fps_time = now

    def get_epsilon(self):
        with self.lock:
            return float(self.epsilon)

    def get_effective_epsilon(self) -> float:
        with self.lock:
            return 0.0 if self.override_epsilon else float(self.epsilon)

    def update_epsilon(self):
        with self.lock:
            if self.manual_epsilon_override:
                return self.epsilon
            progress = min(1.0, self.frame_count / max(1, RL_CONFIG.epsilon_decay_frames))
            self.epsilon = RL_CONFIG.epsilon_start + progress * (RL_CONFIG.epsilon_end - RL_CONFIG.epsilon_start)
            return self.epsilon

    def get_expert_ratio(self):
        with self.lock:
            return float(self.expert_ratio)

    def update_expert_ratio(self):
        with self.lock:
            if self.expert_mode or self.override_expert or self.manual_expert_override:
                return self.expert_ratio
            progress = min(1.0, self.frame_count / max(1, RL_CONFIG.expert_ratio_decay_frames))
            self.expert_ratio = RL_CONFIG.expert_ratio_start + progress * (RL_CONFIG.expert_ratio_end - RL_CONFIG.expert_ratio_start)
            return self.expert_ratio

    def add_episode_reward(self, total, dqn, expert, subj=None, obj=None, length=0):
        with self.lock:
            self.episode_rewards.append(float(total))
            self.dqn_rewards.append(float(dqn))
            self.expert_rewards.append(float(expert))
            if subj is not None:
                self.subj_rewards.append(float(subj))
            if obj is not None:
                self.obj_rewards.append(float(obj))
            self.reward_sum_interval += float(total)
            self.reward_count_interval += 1
            self.reward_sum_interval_dqn += float(dqn)
            self.reward_count_interval_dqn += 1
            if subj is not None:
                self.reward_sum_interval_subj += float(subj)
                self.reward_count_interval_subj += 1
            if obj is not None:
                self.reward_sum_interval_obj += float(obj)
                self.reward_count_interval_obj += 1
            if length > 0:
                self.episode_length_sum_interval += length
                self.episode_length_count_interval += 1

    def increment_total_controls(self):
        with self.lock:
            self.total_controls += 1

    def update_game_state(self, enemy_seg, open_level):
        pass  # compat stub

    def add_inference_time(self, t: float):
        with self.lock:
            self.total_inference_time += t
            self.total_inference_requests += 1

    # ── UI toggle methods ───────────────────────────────────────────────
    def toggle_override(self, kb=None):
        with self.lock:
            self.override_expert = not self.override_expert
            if self.override_expert:
                self.saved_expert_ratio = self.expert_ratio
                self.expert_ratio = 0.0
            else:
                self.expert_ratio = self.saved_expert_ratio

    def toggle_expert_mode(self, kb=None):
        with self.lock:
            self.expert_mode = not self.expert_mode
            if self.expert_mode:
                self.saved_expert_ratio = self.expert_ratio
                self.expert_ratio = 1.0
            else:
                self.expert_ratio = self.saved_expert_ratio

    def toggle_training_mode(self, kb=None):
        with self.lock:
            self.training_enabled = not self.training_enabled

    def toggle_epsilon_override(self, kb=None):
        with self.lock:
            self.override_epsilon = not self.override_epsilon

    def toggle_verbose_mode(self, kb=None):
        with self.lock:
            self.verbose_mode = not self.verbose_mode

    def increase_expert_ratio(self, kb=None):
        with self.lock:
            p = int(self.expert_ratio * 100)
            p = min(100, p + (1 if p < 10 else 5))
            self.expert_ratio = p / 100.0
            self.manual_expert_override = True

    def decrease_expert_ratio(self, kb=None):
        with self.lock:
            p = int(self.expert_ratio * 100)
            p = max(0, p - (1 if p <= 10 else 5))
            self.expert_ratio = p / 100.0
            self.manual_expert_override = True

    def restore_natural_expert_ratio(self, kb=None):
        with self.lock:
            self.manual_expert_override = False
            progress = min(1.0, self.frame_count / max(1, RL_CONFIG.expert_ratio_decay_frames))
            self.expert_ratio = RL_CONFIG.expert_ratio_start + progress * (RL_CONFIG.expert_ratio_end - RL_CONFIG.expert_ratio_start)

    def increase_epsilon(self, kb=None):
        with self.lock:
            p = int(self.epsilon * 100)
            p = min(100, p + (1 if p < 10 else 5))
            self.epsilon = p / 100.0
            self.manual_epsilon_override = True

    def decrease_epsilon(self, kb=None):
        with self.lock:
            p = int(self.epsilon * 100)
            p = max(0, p - (1 if p <= 10 else 5))
            self.epsilon = p / 100.0
            self.manual_epsilon_override = True

    def restore_natural_epsilon(self, kb=None):
        with self.lock:
            self.manual_epsilon_override = False
            progress = min(1.0, self.frame_count / max(1, RL_CONFIG.epsilon_decay_frames))
            self.epsilon = RL_CONFIG.epsilon_start + progress * (RL_CONFIG.epsilon_end - RL_CONFIG.epsilon_start)


metrics = MetricsData()
