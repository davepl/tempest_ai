from collections import deque
import time
import threading
import numpy as np
from typing import Deque

# Define a max length for the deques holding recent values
METRICS_DEQUE_SIZE = 1000

class Metrics:
    """Track various metrics for the session"""
    def __init__(self):
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0.0
        # FPS interval tracking
        self.last_fps_calc_time = time.time()
        self.frames_at_last_calc = 0
        
        self.epsilon = 1.0 # Start with epsilon from config
        self.expert_ratio = 1.0 # Start with expert ratio from config
        self.last_decay_step = -1
        self.last_action_source = "init"
        self.override_expert = False # Manual override flag
        self.expert_mode = False # Automatic expert mode flag
        self.client_count = 0
        self.episode_rewards = deque(maxlen=METRICS_DEQUE_SIZE) # Store last 1000 episode rewards
        self.dqn_rewards = deque(maxlen=METRICS_DEQUE_SIZE) # Store DQN-only rewards
        self.expert_rewards = deque(maxlen=METRICS_DEQUE_SIZE) # Store expert-only rewards
        self.losses = deque(maxlen=METRICS_DEQUE_SIZE * 2) # Store recent loss values
        self.guided_count = 0
        self.total_controls = 0
        self.avg_dqn_inf_time = 0.0 # Will store the interval average
        self.avg_expert_inf_time = 0.0 # Could do the same for expert if needed
        self.dqn_inf_times = deque(maxlen=METRICS_DEQUE_SIZE) # Keep for potential other uses
        self.expert_inf_times = deque(maxlen=METRICS_DEQUE_SIZE)
        # Inference Time interval tracking
        self.interval_inf_time_sum = 0.0
        self.interval_inf_time_count = 0
        
        self.last_agent_save_time = 0.0 # Timestamp of last save triggered by Lua
        
        # DQN frame reward tracking (Cumulative)
        self.dqn_frame_count = 0
        self.dqn_total_reward_on_frames = 0.0
        # ---> Add Interval-specific DQN frame reward tracking <---
        self.interval_dqn_frame_count = 0
        self.interval_dqn_total_reward = 0.0
        self.avg_dqn_reward_per_frame = float('nan') # Stores the last interval average
        
        self.lock = threading.Lock()
        # Add placeholders for state values needed by metrics display
        self.enemy_seg = -1
        self.open_level = False
        # Reference to server instance (optional, for client count)
        self.global_server = None
        # --- Add deque for inference times ---
        self.dqn_inference_times: Deque[float] = deque(maxlen=METRICS_DEQUE_SIZE)

    def update_frame_count(self):
        # Only increment frame count here
        with self.lock:
            self.frame_count += 1
            # FPS calculation moved to calculate_interval_fps
            return self.frame_count

    def calculate_interval_fps(self):
        """Calculate FPS based on frames/time since last call."""
        with self.lock:
            current_time = time.time()
            current_frames = self.frame_count
            
            elapsed_time = current_time - self.last_fps_calc_time
            elapsed_frames = current_frames - self.frames_at_last_calc
            
            if elapsed_time > 0.01: # Avoid division by zero or tiny intervals
                self.fps = elapsed_frames / elapsed_time
            # else: keep the previous fps value if interval is too short

            # Reset for next interval
            self.last_fps_calc_time = current_time
            self.frames_at_last_calc = current_frames
            return self.fps # Return the calculated value

    def get_epsilon(self):
        with self.lock:
            return self.epsilon

    def update_epsilon(self, epsilon_value=None):
        # This might be called by main loop based on frame count decay
        # Or directly set if managed elsewhere
        with self.lock:
             if epsilon_value is not None:
                  self.epsilon = epsilon_value
             # Add decay logic here if needed, or confirm it's handled externally
             pass # Assuming decay handled elsewhere

    def update_expert_ratio(self, expert_ratio_value=None):
        with self.lock:
             if expert_ratio_value is not None:
                  self.expert_ratio = expert_ratio_value
             # Add decay logic here if needed, or confirm it's handled externally
             pass # Assuming decay handled elsewhere

    def add_episode_reward(self, total_reward, dqn_reward, expert_reward):
        with self.lock:
            self.episode_rewards.append(total_reward)
            self.dqn_rewards.append(dqn_reward)
            self.expert_rewards.append(expert_reward)

    def add_loss(self, loss_value):
        """Add a loss value from the training process."""
        with self.lock:
            self.losses.append(loss_value)

    def increment_guided_count(self):
        with self.lock:
            self.guided_count += 1

    def increment_total_controls(self):
        with self.lock:
            self.total_controls += 1
            
    def update_action_source(self, source):
        with self.lock:
            self.last_action_source = source
            
    def update_game_state(self, enemy_seg, open_level):
        # Used for display
        with self.lock:
            self.enemy_seg = enemy_seg
            self.open_level = open_level

    def update_dqn_inference_time(self, inf_time_ms):
        with self.lock:
             # Keep rolling deque (optional)
             self.dqn_inf_times.append(inf_time_ms)
             # Update interval accumulators
             self.interval_inf_time_sum += inf_time_ms
             self.interval_inf_time_count += 1
             # Note: self.avg_dqn_inf_time is updated by calculate_interval_avg_inf_time

    def update_expert_inference_time(self, inf_time_ms):
        with self.lock:
             self.expert_inf_times.append(inf_time_ms)
             if self.expert_inf_times:
                  self.avg_expert_inf_time = np.mean(list(self.expert_inf_times))
    
    def calculate_interval_avg_inf_time(self):
        """Calculate average DQN inference time over the last interval."""
        with self.lock:
            if self.interval_inf_time_count > 0:
                 # Calculate average for the interval
                 interval_avg = self.interval_inf_time_sum / self.interval_inf_time_count
                 # Update the main attribute used for display
                 self.avg_dqn_inf_time = interval_avg
            # else: If count is 0, keep the previous avg_dqn_inf_time value
            
            # Reset accumulators for the next interval
            self.interval_inf_time_sum = 0.0
            self.interval_inf_time_count = 0
            
            return self.avg_dqn_inf_time # Return the calculated interval average

    def get_expert_ratio(self):
        # Returns effective ratio considering override
        with self.lock:
            if self.override_expert:
                return 0.0 # Override forces DQN (0% expert)
            elif self.expert_mode:
                 return 1.0 # Treat expert mode as 100% expert
            else:
                 # Return the base, potentially decayed, ratio
                 return self.expert_ratio
            
    def is_override_active(self):
        with self.lock:
            return self.override_expert

    def get_fps(self):
        with self.lock:
            return self.fps

    def get_client_count(self):
         # Optionally get count from server reference
         if self.global_server:
              with self.global_server.client_lock:
                   self.client_count = len(self.global_server.client_states)
         return self.client_count

    def get_decision_params(self):
        """Get parameters needed for action decision under a single lock."""
        with self.lock:
            effective_ratio = self.expert_ratio # Base ratio
            override = self.override_expert
            expert_mode_active = self.expert_mode
            current_epsilon = self.epsilon
            
            # Calculate effective ratio based on modes
            if override:
                 effective_ratio = 0.0
            elif expert_mode_active:
                 effective_ratio = 1.0
                 
            return effective_ratio, override, expert_mode_active, current_epsilon

    # --- Add method to store inference time ---
    def add_inference_time(self, time_sec: float):
        """Add a single DQN inference time measurement."""
        with self.lock:
            self.dqn_inference_times.append(time_sec)

    def get_state_for_save(self) -> dict:
        """Returns a dictionary of metrics relevant for saving checkpoints."""
        with self.lock:
            return {
                'frame_count': self.frame_count,
                'epsilon': self.epsilon,
                'expert_ratio': self.expert_ratio,
                'last_decay_step': self.last_decay_step,
                # Add other relevant states if needed
            }

    def restore_state_from_save(self, state: dict):
        """Restores metrics from a loaded checkpoint state."""
        with self.lock:
            self.frame_count = state.get('frame_count', self.frame_count)
            self.epsilon = state.get('epsilon', self.epsilon)
            self.expert_ratio = state.get('expert_ratio', self.expert_ratio)
            self.last_decay_step = state.get('last_decay_step', self.last_decay_step)
            print(f"[Metrics] Restored state: Frame={self.frame_count}, Epsilon={self.epsilon:.4f}, ExpertRatio={self.expert_ratio:.4f}")

    # --- Method to add reward specifically for DQN-controlled frames --- 
    def add_dqn_frame_reward(self, reward: float):
        """Increment count and total reward for DQN frames."""
        with self.lock:
            self.dqn_frame_count += 1
            self.dqn_total_reward_on_frames += reward
            # ---> Increment interval counters as well <---            
            self.interval_dqn_frame_count += 1
            self.interval_dqn_total_reward += reward
    # --- End method ---

    # ---> Add method to calculate and reset interval average <--- 
    def calculate_interval_avg_dqn_frame_reward(self) -> float:
        """Calculate avg reward per DQN frame for the interval and reset counters."""
        with self.lock:
            if self.interval_dqn_frame_count > 0:
                self.avg_dqn_reward_per_frame = self.interval_dqn_total_reward / self.interval_dqn_frame_count
            else:
                self.avg_dqn_reward_per_frame = float('nan') # Or 0.0 if preferred for no DQN frames
            
            # Reset for the next interval
            self.interval_dqn_frame_count = 0
            self.interval_dqn_total_reward = 0.0
            
            return self.avg_dqn_reward_per_frame
            
# Global instance
metrics = Metrics() 