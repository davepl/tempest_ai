#!/usr/bin/env python3
"""
Configuration and shared types for Tempest AI.
"""

# Prevent direct execution
if __name__ == "__main__":
    print("This is not the main application, run 'main.py' instead")
    exit(1)

import os
import sys
import time
import threading
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple
from collections import deque

# Global flags
IS_INTERACTIVE = sys.stdin.isatty()

# Flag to control metric reset on load
RESET_METRICS = False  # Set to True to ignore saved epsilon/expert ratio - FRESH START
FORCE_FRESH_MODEL = False # Set to True to completely ignore saved model and start fresh

# Directory paths
MODEL_DIR = "models"
LATEST_MODEL_PATH = f"{MODEL_DIR}/tempest_model_latest.pt"

@dataclass
class ServerConfigData:
    """Configuration for socket server"""
    host: str = "0.0.0.0"  # Listen on all interfaces
    port: int = 9999
    max_clients: int = 36
    params_count: int = 171
    reset_frame_count: bool = False   # Resume from checkpoint - don't reset frame count
    reset_expert_ratio: bool = False  # Resume from checkpoint - don't reset expert ratio  
    reset_epsilon: bool = True       # Resume from checkpoint - don't reset epsilon
     
    force_expert_ratio_recalc: bool = False  # Don't force recalculation of expert ratio

# Create instance of ServerConfigData first
SERVER_CONFIG = ServerConfigData()

@dataclass
class RLConfigData:
    """Reinforcement Learning Configuration"""
    state_size: int = SERVER_CONFIG.params_count  # Use value from ServerConfigData
    # Hybrid action space: 4 discrete fire/zap combinations + 1 continuous spinner
    discrete_action_size: int = 4  # fire/zap combinations: (0,0), (1,0), (0,1), (1,1)
    continuous_action_size: int = 1  # spinner value in [-0.3, +0.3]
    # Legacy removed: discrete 18-action size (pure hybrid model)
    # Phase 1 Optimization: Larger batch + accumulation for better GPU utilization
    batch_size: int = 16384               # Increased for better GPU utilization with AMP enabled
    lr: float = 0.003                     # PLATEAU BREAKER: Double LR from 0.0025 to escape local optimum
    gradient_accumulation_steps: int = 1  # Increased to simulate 131k effective batch for throughput
    gamma: float = 0.995                   # Reverted from 0.92 - lower gamma made plateau worse
    epsilon: float = 0.25                 # Next-run start: exploration rate (see decay schedule below)
    epsilon_start: float = 0.25           # Start at 0.20 on next run
    # Quick Win: keep a bit more random exploration while DQN catches up
    epsilon_min: float = 0.25            # Floor for exploration
    epsilon_end: float = 0.25            # Target floor
    epsilon_decay_steps: int = 10000     # Decay applied every 10k frames
    epsilon_decay_factor: float = 0.995
    # Expert guidance ratio schedule (moved here next to epsilon for unified exploration control)
    expert_ratio_start: float = 0.95      # Initial probability of expert control
    # During GS_ZoomingDown (0x20), exploration is disruptive; scale epsilon down at inference time
    zoom_epsilon_scale: float = 0.25
    expert_ratio_min: float = 0.10        # Minimum expert control probability
    expert_ratio_decay: float = 0.996     # Multiplicative decay factor per step interval
    expert_ratio_decay_steps: int = 10000 # Step interval for applying decay
    memory_size: int = 5000000           # Balanced buffer size (was 4000000)
    hidden_size: int = 512               # More moderate size - 2048 too slow for rapid experimentation
    num_layers: int = 6                  
    target_update_freq: int = 2000        # Reverted from 1000 - more frequent updates destabilized learning
    update_target_every: int = 2000       # Reverted - more frequent target updates made plateau worse
    save_interval: int = 10000            # Model save frequency
    use_noisy_nets: bool = False          # DISABLED: Use pure epsilon-greedy exploration for debugging
    
    # Prioritized Experience Replay (PER) settings
    use_per: bool = True                  # Enable Prioritized Experience Replay for better sample efficiency
    per_alpha: float = 0.6                # Prioritization exponent (0=uniform, 1=fully prioritized)
    per_beta_start: float = 0.4           # Initial importance sampling correction exponent
    per_beta_end: float = 1.0             # Final importance sampling correction exponent
    per_beta_decay_steps: int = 1000000   # Steps to linearly anneal beta from start to end
    per_eps: float = 1e-6                 # Small constant to prevent zero priorities
      
    # NOTE: gradient_accumulation_steps is defined above and should remain 2 for responsiveness
    use_mixed_precision: bool = True      # Enable automatic mixed precision for better performance  
    # Phase 1 Optimization: More frequent updates for faster convergence
    training_steps_per_sample: int = 8    # Reduced from 8 to 2 to prevent queue overflow
    training_workers: int = 4             # Increased from 8 to 16 for better queue processing
    use_torch_compile: bool = True        # ENABLED - torch.compile for loss computation (safe in single-threaded training)
    use_soft_target: bool = True          # Enable soft target updates for stability
    tau: float = 0.012                    # Slight bump for more responsive soft target tracking
    # Optional hard refresh of target net even when using soft updates
    hard_target_refresh_every_steps: int = 25000  # Slightly faster hard refresh cadence to re-anchor periodically
    # Warmup hard-refresh policy (active only while total_training_steps < warmup_until_steps)
    warmup_hard_refresh_until_steps: int = 2000
    warmup_hard_refresh_every_steps: int = 200
    # Watchdog: ensure a hard refresh happens at least every N seconds once training is active
    hard_update_watchdog_seconds: float = 3600.0     # Once per hour; rely on soft targets primarily
    # Legacy setting removed: zap_random_scale used only by legacy discrete agent
    # Modest n-step to aid credit assignment without destabilizing
    n_step: int = 5
    # Enable dueling architecture for better value/advantage separation
    use_dueling: bool = True              # ENABLED: Deeper network can benefit from dueling streams             
    # Loss function type: 'mse' for vanilla DQN, 'huber' for more robust training
    loss_type: str = 'huber'              # Use Huber for robustness to outliers
    # Gradient clipping configuration
    max_grad_norm: float = 5.0            # Clip threshold for total grad norm (L2)
    # Target clamp to stabilize bootstrapping near plateaus - DISABLED to observe natural Q-value range with gamma=0.95
    clamp_targets: bool = False           # DISABLED: Let Q-values grow naturally to detect any remaining inflation
    target_clamp_value: float = 8.0       # Value preserved for potential re-enable if needed
    # Require fresh frames after load before resuming training
    min_new_frames_after_load_to_train: int = 50000

    # Optimization: learning-rate schedule (frame-based)
    # Options: 'none', 'cosine'
    lr_schedule: str = 'cosine'
    lr_base: float = 0.001
    lr_min: float = 5e-05
    lr_warmup_frames: int = 250_000       # linear warmup from lr_min -> lr_base
    lr_hold_until_frames: int = 1_000_000 # hold at lr_base until this frame
    lr_decay_until_frames: int = 12_000_000 # cosine decay from hold -> this frame

    # Targets: use Double DQN bootstrapping for discrete head
    use_double_dqn: bool = True

    # Replay sampling bias toward most recent data (windowed uniform)
    # If bias > 0, sample this fraction from the most recent (window_frac) of the buffer
    recent_sample_bias: float = 0.5       # 0.0 disables; 0.5 = half recent, half global
    recent_window_frac: float = 0.25      # last 25% of buffer considered "recent"

    # Exploration policy: allow adaptive epsilon floor adjustments as performance improves
    adaptive_epsilon_floor: bool = True
    # Expert policy: allow adaptive expert-ratio floor adjustments based on performance trend
    adaptive_expert_floor: bool = False

    # Loss weighting: balance continuous head relative to discrete head
    continuous_loss_weight: float = 0.5

    # Reward shaping/normalization controls (to stabilize targets when external reward scale changes)
    reward_scale: float = 0.1            # Multiply incoming rewards by this factor before TD target
    reward_clamp_abs: float = 0.0        # If > 0, clamp rewards to [-reward_clamp_abs, +reward_clamp_abs]
    reward_tanh: bool = False            # If True, apply tanh to (scaled) rewards

    # Subjective reward scaling (for movement/aiming rewards)
    subj_reward_scale: float = 0.35       # Scale factor applied to subjective rewards from OOB
    
    # N-step returns: runtime toggle and diversity bonus settings
    n_step_enabled: bool = True           # Runtime toggle for n-step returns (hotkey 'n')
    diversity_bonus_enabled: bool = True  # Reward trying different actions in similar states (hotkey 'd')
    diversity_bonus_weight: float = 0.5   # Base reward bonus for novel actions (decays with 1/sqrt(n))

    # Optional gradient value clamp (0.0 disables)
    max_grad_value: float = 0.0

    # Superzap gate: probability that a DQN-selected zap is actually executed (0..1)
    # Expert-directed zaps are not gated.
    superzap_prob: float = 0.01

# Create instance of RLConfigData after its definition
RL_CONFIG = RLConfigData()

@dataclass
class MetricsData:
    """Metrics tracking for training progress"""
    frame_count: int = 0
    guided_count: int = 0
    total_controls: int = 0
    episode_rewards: Deque[float] = field(default_factory=lambda: deque(maxlen=20))
    dqn_rewards: Deque[float] = field(default_factory=lambda: deque(maxlen=20))
    expert_rewards: Deque[float] = field(default_factory=lambda: deque(maxlen=20))
    subj_rewards: Deque[float] = field(default_factory=lambda: deque(maxlen=20))  # Subjective rewards (movement/aiming)
    obj_rewards: Deque[float] = field(default_factory=lambda: deque(maxlen=20))   # Objective rewards (scoring)
    losses: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    epsilon: float = field(default_factory=lambda: RL_CONFIG.epsilon_start)
    expert_ratio: float = RL_CONFIG.expert_ratio_start
    last_decay_step: int = 0
    last_epsilon_decay_step: int = 0 # Added tracker for epsilon decay
    enemy_seg: int = -1
    open_level: bool = False
    override_expert: bool = False
    saved_expert_ratio: float = 0.75
    expert_mode: bool = False
    manual_expert_override: bool = False  # Track if manual +/- override is active
    manual_epsilon_override: bool = False  # Track if manual epsilon override is active
    last_action_source: str = ""
    frames_last_second: int = 0
    last_fps_time: float = 0
    fps: float = 0.0
    client_count: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)
    total_inference_time: float = 0.0
    total_inference_requests: int = 0
    average_level: float = 0  # Average level number across all clients
    # PER-specific metrics removed
    # Gradient clipping diagnostics
    grad_clip_delta: float = 1.0   # post-clip L2 norm divided by pre-clip L2 norm (should be ~1.0 if not clipping)
    grad_norm: float = 0.0         # pre-clip gradient L2 norm magnitude for monitoring learning dynamics
    # Loss averaging since last metrics print
    loss_sum_interval: float = 0.0
    loss_count_interval: int = 0
    # Training steps since last metrics print and when last row printed
    training_steps_interval: int = 0
    last_metrics_row_time: float = 0.0
    # Frames since last metrics print
    frames_count_interval: int = 0
    
    # Training-specific metrics
    memory_buffer_size: int = 0  # Current replay buffer size
    total_training_steps: int = 0  # Total training steps completed
    last_target_update_frame: int = 0  # Frame count when target network was last updated
    last_inference_sync_frame: int = 0 # Frame count when inference net was last synced
    last_target_update_time: float = 0.0   # Wall time of last target update
    last_inference_sync_time: float = 0.0  # Wall time of last inference sync
    # Hard target update telemetry (full copy)
    last_hard_target_update_frame: int = 0
    last_hard_target_update_time: float = 0.0
    # Rolling performance metrics (from metrics_display)
    dqn5m_avg: float = 0.0  # Weighted average DQN reward across last 5M frames
    dqn5m_slopeM: float = 0.0  # Weighted regression slope per million frames
    # Frame count at load time for enforcing post-load burn-in
    loaded_frame_count: int = 0
    
    # Reward component tracking (for analysis and display)
    # State summary stats (rolling)
    state_mean: float = 0.0
    state_std: float = 0.0
    state_min: float = 0.0
    state_max: float = 0.0
    state_invalid_frac: float = 0.0  # Fraction of entries equal to -1.0 (our invalid sentinel)
    # Level averaging since last metrics print (0-based levels)
    level_sum_interval: float = 0.0
    level_count_interval: int = 0
    # Reward averaging since last metrics print
    reward_sum_interval_total: float = 0.0
    reward_count_interval_total: int = 0
    reward_sum_interval_dqn: float = 0.0
    reward_count_interval_dqn: int = 0
    reward_sum_interval_expert: float = 0.0
    reward_count_interval_expert: int = 0
    reward_sum_interval_subj: float = 0.0
    reward_count_interval_subj: int = 0
    reward_sum_interval_obj: float = 0.0
    reward_count_interval_obj: int = 0
    # Training enable/disable (UI toggle). When False, background workers do no training.
    training_enabled: bool = True
    # Epsilon override: when True, force epsilon=0.0 (pure greedy) regardless of other overrides
    override_epsilon: bool = False
    
    def update_frame_count(self, delta: int = 1):
        """Update frame count and FPS tracking"""
        with self.lock:
            # Update total frame count
            if delta < 1:
                delta = 1
            self.frame_count += delta
            # Track interval frames for rate calculations
            try:
                self.frames_count_interval += delta
            except Exception:
                pass
            
            # Update FPS tracking
            current_time = time.time()
            
            # Initialize last_fps_time if this is the first frame
            if self.last_fps_time == 0:
                self.last_fps_time = current_time
                
            # Count frames for this second
            self.frames_last_second += delta
            
            # Calculate FPS every second
            elapsed = current_time - self.last_fps_time
            if elapsed >= 1.0:
                # Calculate frames per second with more accuracy
                new_fps = self.frames_last_second / elapsed
                
                # Store the new FPS value
                self.fps = new_fps
                
                # Reset counters
                self.frames_last_second = 0
                self.last_fps_time = current_time
                
            return self.frame_count
    
    def get_epsilon(self):
        """Get current epsilon value"""
        with self.lock:
            return self.epsilon

    def get_effective_epsilon(self) -> float:
        """Return the epsilon value that will actually be used for action selection.

        When override_epsilon is ON, this returns 0.0 (pure greedy) regardless of other modes.
        Otherwise, returns the current decayed epsilon.
        """
        with self.lock:
            return 0.0 if self.override_epsilon else float(self.epsilon)
    
    def update_epsilon(self):
        """Update epsilon based on frame count"""
        with self.lock:
            # Import here to avoid circular imports
            from aimodel import decay_epsilon
            self.epsilon = decay_epsilon(self.frame_count)
            return self.epsilon
    
    def update_expert_ratio(self):
        """Update expert ratio based on frame count"""
        with self.lock:
            # Import here to avoid circular imports
            from aimodel import decay_expert_ratio
            # Skip decay if expert mode, override mode, or manual override is active
            if self.expert_mode or self.override_expert or self.manual_expert_override:
                return self.expert_ratio
            decay_expert_ratio(self.frame_count)
            return self.expert_ratio
    
    def add_episode_reward(self, total_reward, dqn_reward, expert_reward, subj_reward=None, obj_reward=None):
        """Add episode rewards to tracking (include negatives/zeros for accurate means)"""
        with self.lock:
            self.episode_rewards.append(float(total_reward))
            self.dqn_rewards.append(float(dqn_reward))
            self.expert_rewards.append(float(expert_reward))
            # Track subjective and objective rewards if provided
            if subj_reward is not None:
                self.subj_rewards.append(float(subj_reward))
            if obj_reward is not None:
                self.obj_rewards.append(float(obj_reward))
            # Track interval reward averages
            try:
                self.reward_sum_interval_total += float(total_reward)
                self.reward_count_interval_total += 1
                self.reward_sum_interval_dqn += float(dqn_reward)
                self.reward_count_interval_dqn += 1
                self.reward_sum_interval_expert += float(expert_reward)
                self.reward_count_interval_expert += 1
                if subj_reward is not None:
                    self.reward_sum_interval_subj += float(subj_reward)
                    self.reward_count_interval_subj += 1
                if obj_reward is not None:
                    self.reward_sum_interval_obj += float(obj_reward)
                    self.reward_count_interval_obj += 1
            except Exception:
                pass
    
    def increment_guided_count(self):
        """Increment guided count"""
        with self.lock:
            self.guided_count += 1
    
    def increment_total_controls(self):
        """Increment total controls"""
        with self.lock:
            self.total_controls += 1
    
    def update_action_source(self, source):
        """Update last action source"""
        with self.lock:
            self.last_action_source = source
    
    def update_game_state(self, enemy_seg, open_level):
        """Update game state"""
        with self.lock:
            self.enemy_seg = enemy_seg
            self.open_level = open_level
    
    def get_expert_ratio(self):
        """Get current expert ratio"""
        with self.lock:
            return self.expert_ratio
    
    def get_effective_expert_ratio(self):
        """Get the effective expert ratio used for decisions (0.0 when override is ON)"""
        with self.lock:
            if self.override_expert:
                return 0.0
            return self.expert_ratio
    
    def is_override_active(self):
        """Check if override is active"""
        with self.lock:
            return self.override_expert
    
    def get_fps(self):
        """Get current FPS"""
        with self.lock:
            return self.fps
    
    
    def get_reward_component_averages(self) -> Dict[str, float]:
        """Get recent averages of reward components"""
        with self.lock:
            averages = {}
            for component, history in self.reward_component_history.items():
                if history:
                    averages[component] = sum(history) / len(history)
                else:
                    averages[component] = 0.0
            return averages
    
    def toggle_override(self, kb_handler=None):
        """Toggle override mode"""
        with self.lock:
            self.override_expert = not self.override_expert
            if self.override_expert:
                self.saved_expert_ratio = self.expert_ratio
                self.expert_ratio = 0.0
            else:
                self.expert_ratio = self.saved_expert_ratio
            if kb_handler and IS_INTERACTIVE:
                # Import here to avoid circular import at top level
                from aimodel import print_with_terminal_restore
                print_with_terminal_restore(kb_handler, f"\nOverride mode: {'ON' if self.override_expert else 'OFF'}\r")
    
    def toggle_expert_mode(self, kb_handler=None):
        """Toggle expert mode"""
        with self.lock:
            self.expert_mode = not self.expert_mode
            if self.expert_mode:
                # Save current expert ratio and set to 1.0 (100%) when expert mode is ON
                self.saved_expert_ratio = self.expert_ratio
                self.expert_ratio = 1.0
            else:
                # Restore the saved expert ratio when expert mode is OFF
                self.expert_ratio = self.saved_expert_ratio
            if kb_handler and IS_INTERACTIVE:
                # Import here to avoid circular import at top level
                from aimodel import print_with_terminal_restore
                print_with_terminal_restore(kb_handler, f"\nExpert mode: {'ON' if self.expert_mode else 'OFF'}\r")

    def toggle_training_mode(self, kb_handler=None):
        """Toggle training enable/disable (does not affect data collection)."""
        with self.lock:
            self.training_enabled = not self.training_enabled
            status = 'ON' if self.training_enabled else 'OFF'
            if kb_handler and IS_INTERACTIVE:
                from aimodel import print_with_terminal_restore
                print_with_terminal_restore(kb_handler, f"\nTrain: {status}\r")

    def toggle_epsilon_override(self, kb_handler=None):
        """Toggle epsilon override. When ON, epsilon is treated as 0.0 everywhere (pure greedy)."""
        with self.lock:
            self.override_epsilon = not self.override_epsilon
            status = 'ON' if self.override_epsilon else 'OFF'
            if kb_handler and IS_INTERACTIVE:
                from aimodel import print_with_terminal_restore
                print_with_terminal_restore(kb_handler, f"\nEpsilon override: {status}\r")
    
    def increase_expert_ratio(self, kb_handler=None):
        """Increase expert ratio with smart stepping: 0.01 in decimals (0.00-0.09), 0.05 in tenths (0.10+)"""
        with self.lock:
            current_percent = int(self.expert_ratio * 100)
            
            if current_percent < 10:
                # Single digits: step by 1%
                next_percent = current_percent + 1
            else:
                # Double digits: step by 5% (round up to next multiple of 5)
                next_percent = ((current_percent + 5) // 5) * 5
            
            # Cap at 100%
            next_percent = min(next_percent, 100)
            self.expert_ratio = next_percent / 100.0
            self.manual_expert_override = True
            # Auto-disable override_expert when manually setting ratio > 0
            if self.override_expert and next_percent > 0:
                self.override_expert = False
            if kb_handler and IS_INTERACTIVE:
                from aimodel import print_with_terminal_restore
                print_with_terminal_restore(kb_handler, f"\nExpert ratio: {next_percent}% (manual override)\r")
    
    def decrease_expert_ratio(self, kb_handler=None):
        """Decrease expert ratio with smart stepping: 0.01 in decimals (0.00-0.09), 0.05 in tenths (0.10+)"""
        with self.lock:
            current_percent = int(self.expert_ratio * 100)
            
            if current_percent <= 10:
                # Single digits and 10%: step by 1%
                next_percent = current_percent - 1
            else:
                # Above 10%: step by 5% (round down to previous multiple of 5)
                next_percent = ((current_percent - 1) // 5) * 5
            
            # Floor at 0%
            next_percent = max(next_percent, 0)
            self.expert_ratio = next_percent / 100.0
            self.manual_expert_override = True
            # Auto-disable override_expert when manually setting ratio > 0
            if self.override_expert and next_percent > 0:
                self.override_expert = False
            if kb_handler and IS_INTERACTIVE:
                from aimodel import print_with_terminal_restore
                print_with_terminal_restore(kb_handler, f"\nExpert ratio: {next_percent}% (manual override)\r")
    
    def restore_natural_expert_ratio(self, kb_handler=None):
        """Restore natural decaying expert ratio (=key)"""
        with self.lock:
            self.manual_expert_override = False
            # Recalculate the natural expert ratio based on current frame count
            from aimodel import decay_expert_ratio
            # Temporarily disable override to allow natural calculation
            old_override = self.override_expert
            old_expert_mode = self.expert_mode
            self.override_expert = False
            self.expert_mode = False
            decay_expert_ratio(self.frame_count)
            # Restore previous override states
            self.override_expert = old_override
            self.expert_mode = old_expert_mode
            if kb_handler and IS_INTERACTIVE:
                from aimodel import print_with_terminal_restore
                print_with_terminal_restore(kb_handler, f"\nExpert ratio: {int(self.expert_ratio * 100)}% (natural decay)\r")
    
    def increase_epsilon(self, kb_handler=None):
        """Increase epsilon with smart stepping: 0.01 in decimals (0.00-0.09), 0.05 in tenths (0.10+)"""
        with self.lock:
            current_percent = int(self.epsilon * 100)
            
            if current_percent < 10:
                # At or under 9%: step by 0.01
                next_percent = current_percent + 1
            else:
                # 10% and above: step by 0.05 (round up to next multiple of 5)
                next_percent = ((current_percent + 5) // 5) * 5
            
            # Cap at 100%
            next_percent = min(next_percent, 100)
            self.epsilon = next_percent / 100.0
            self.manual_epsilon_override = True
            # Auto-disable override_epsilon when manually setting epsilon > 0
            if self.override_epsilon and next_percent > 0:
                self.override_epsilon = False
            if kb_handler and IS_INTERACTIVE:
                from aimodel import print_with_terminal_restore
                print_with_terminal_restore(kb_handler, f"\nEpsilon: {self.epsilon:.3f} (manual override)\r")
    
    def decrease_epsilon(self, kb_handler=None):
        """Decrease epsilon with smart stepping: 0.01 in decimals (0.00-0.09), 0.05 in tenths (0.10+)"""
        with self.lock:
            current_percent = int(self.epsilon * 100)
            
            if current_percent <= 10:
                # At or under 10%: step by 0.01
                next_percent = current_percent - 1
            else:
                # Above 10%: step by 0.05 (round down to previous multiple of 5)
                next_percent = ((current_percent - 1) // 5) * 5
            
            # Floor at 0%
            next_percent = max(next_percent, 0)
            self.epsilon = next_percent / 100.0
            self.manual_epsilon_override = True
            # Auto-disable override_epsilon when manually setting epsilon > 0
            if self.override_epsilon and next_percent > 0:
                self.override_epsilon = False
            if kb_handler and IS_INTERACTIVE:
                from aimodel import print_with_terminal_restore
                print_with_terminal_restore(kb_handler, f"\nEpsilon: {self.epsilon:.3f} (manual override)\r")
    
    def restore_natural_epsilon(self, kb_handler=None):
        """Restore natural decaying epsilon (=key)"""
        with self.lock:
            self.manual_epsilon_override = False
            # Recalculate the natural epsilon based on current frame count
            from aimodel import decay_epsilon
            # Temporarily disable override to allow natural calculation
            old_override = self.override_epsilon
            self.override_epsilon = False
            decay_epsilon(self.frame_count)
            # Restore previous override state
            self.override_epsilon = old_override
            if kb_handler and IS_INTERACTIVE:
                from aimodel import print_with_terminal_restore
                print_with_terminal_restore(kb_handler, f"\nEpsilon: {self.epsilon:.3f} (natural decay)\r")
    
    def increase_learning_rate(self, kb_handler=None, agent=None):
        """Increase learning rate by 0.0005 (L key)"""
        with self.lock:
            RL_CONFIG.lr += 0.0005
            # Apply to agent's optimizer if available
            if agent:
                try:
                    for param_group in agent.optimizer.param_groups:
                        param_group['lr'] = RL_CONFIG.lr
                except Exception:
                    pass
            if kb_handler and IS_INTERACTIVE:
                from aimodel import print_with_terminal_restore
                print_with_terminal_restore(kb_handler, f"\nLearning Rate: {RL_CONFIG.lr:.5f}\r")
    
    def decrease_learning_rate(self, kb_handler=None, agent=None):
        """Decrease learning rate by 0.0005, floor at 0.0001 (l key)"""
        with self.lock:
            RL_CONFIG.lr = max(0.0001, RL_CONFIG.lr - 0.0005)
            # Apply to agent's optimizer if available
            if agent:
                try:
                    for param_group in agent.optimizer.param_groups:
                        param_group['lr'] = RL_CONFIG.lr
                except Exception:
                    pass
            if kb_handler and IS_INTERACTIVE:
                from aimodel import print_with_terminal_restore
                print_with_terminal_restore(kb_handler, f"\nLearning Rate: {RL_CONFIG.lr:.5f}\r")

# Define hybrid action space
# Discrete fire/zap combinations (4 total)
FIRE_ZAP_MAPPING = {
    0: (0, 0),  # No fire, no zap
    1: (1, 0),  # Fire, no zap  
    2: (0, 1),  # No fire, zap
    3: (1, 1),  # Fire, zap
}

# Continuous spinner range - matches expert system full range
SPINNER_MIN = -0.9
SPINNER_MAX = 0.9

# Legacy discrete action mapping removed (pure hybrid model)

# Create instances of config classes
metrics = MetricsData()

# # Import print_with_terminal_restore from metrics_display to avoid circular imports
# # # DEF print_with_terminal_restore(kb_handler, *args, **kwargs):
# # #     \"\"\"Print with terminal restore if in interactive mode\"\"\"
# # #     if IS_INTERACTIVE and kb_handler:
# # #         # Import here to avoid circular imports
# # #         from metrics_display import print_with_terminal_restore as _print
# # #         _print(*args, **kwargs)
# # #     else:
# # #         print(*args, **kwargs)
