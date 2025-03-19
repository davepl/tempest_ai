#!/usr/bin/env python3
"""
Tempest AI Model: Combines Behavioral Cloning (BC) in attract mode with Reinforcement Learning (RL) in gameplay.
- BC learns from game AI demonstrations; RL (SAC) optimizes during player control with robust training architecture.
- Communicates with Tempest via Lua pipes; saves/loads models for persistence.
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import time
import struct
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3 import SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.logger import Logger
import gymnasium as gym
from gymnasium import spaces
import threading
import queue
from collections import deque

# Constants
NumberOfParams = 247
LUA_TO_PY_PIPE = "/tmp/lua_to_py"
PY_TO_LUA_PIPE = "/tmp/py_to_lua"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
LATEST_MODEL_PATH = f"{MODEL_DIR}/tempest_model_latest.zip"
BC_MODEL_PATH = f"{MODEL_DIR}/tempest_bc_model.pt"

# Parse command line arguments
parser = argparse.ArgumentParser(description='Tempest AI Model')
parser.add_argument('--replay', type=str, help='Path to a log file to replay for BC training')
args = parser.parse_args()

# Device selection: CUDA > MPS > CPU
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device.type.upper()}")

# Training metrics buffers
actor_losses = deque(maxlen=100)
critic_losses = deque(maxlen=100)
ent_coefs = deque(maxlen=100)
mean_rewards = deque(maxlen=100)
training_queue = queue.Queue()

class TempestEnv(gym.Env):
    """Custom Gym environment interfacing with Tempest game via Lua pipes."""
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(low=np.array([0.0, 0.0, -1.0], dtype=np.float32), 
                                       high=np.array([1.0, 1.0, 1.0], dtype=np.float32), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(NumberOfParams,), dtype=np.float32)
        self.state = np.zeros(NumberOfParams, dtype=np.float32)
        self.reward = np.float32(0.0)
        self.done = False
        self.is_attract_mode = False
        self.prev_state = None
        self.prev_action = None

    def reset(self, seed=None, options=None):
        """Reset environment state, reward, and flags."""
        super().reset(seed=seed)
        self.state = np.zeros(NumberOfParams, dtype=np.float32)
        self.reward = np.float32(0.0)
        self.done = False
        self.is_attract_mode = False
        self.prev_state = None
        self.prev_action = None
        return self.state, {}

    def step(self, action):
        """Process action, update prev_action, and return current state/reward."""
        self.prev_action = np.array(action, dtype=np.float32)
        fire, zap, spinner = self.decode_action(action)
        return self.state, self.reward, self.done, False, {"action_taken": (fire, zap, spinner)}

    def update_state(self, game_state, reward, game_action=None, done=False):
        """Update state with Lua data, preserving previous state."""
        self.prev_state = self.state.copy()
        self.state = game_state.astype(np.float32)
        self.reward = np.float32(reward)
        self.done = bool(done)
        return self.state

    def decode_action(self, action):
        """Decode RL actions to game inputs."""
        fire = 1 if action[0] > 0.5 else 0
        zap = 1 if action[1] > 0.5 else 0
        spinner = int(round(action[2] * 127.0))
        return fire, zap, max(-127, min(127, spinner))

class TempestFeaturesExtractor(BaseFeaturesExtractor):
    """Feature extractor for SAC, reducing observation dimensionality."""
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        self.feature_extractor = nn.Sequential(
            nn.Linear(observation_space.shape[0], 256), nn.ReLU(),
            nn.Linear(256, features_dim), nn.ReLU()
        ).to(device)

    def forward(self, observations):
        # Ensure observations are on the correct device
        if observations.device != device:
            observations = observations.to(device)
        return self.feature_extractor(observations)

class BCModel(nn.Module):
    """BC model to mimic game AI actions in attract mode."""
    def __init__(self, input_size=NumberOfParams):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU()
        )
        self.fire_output = nn.Linear(64, 1)
        self.zap_output = nn.Linear(64, 1)
        self.spinner_output = nn.Linear(64, 1)
        for layer in [self.net[0], self.net[2], self.fire_output, self.zap_output, self.spinner_output]:
            nn.init.xavier_uniform_(layer.weight)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.to(device)

    def forward(self, x):
        # Ensure input is on the correct device
        if x.device != device:
            x = x.to(device)
        h = self.net(x)
        return torch.cat([
            torch.sigmoid(self.fire_output(h)),
            torch.sigmoid(self.zap_output(h)),
            torch.tanh(self.spinner_output(h))
        ], dim=1)

def process_frame_data(data):
    """Unpack and normalize game frame data from Lua."""
    try:
        header = struct.unpack(">IdBBBIIBBBh", data[:struct.calcsize(">IdBBBIIBBBh")])
        num_values, reward, game_action, game_mode, is_done, frame_counter, score, save_signal, fire, zap, spinner = header
        game_data = np.frombuffer(data[struct.calcsize(">IdBBBIIBBBh"):], dtype=np.uint16).astype(np.float32) - 32768.0
        state = game_data / np.where(game_data > 0, 32767.0, 32768.0).astype(np.float32)
        if len(state) != NumberOfParams:
            state = np.pad(state, (0, NumberOfParams - len(state)), 'constant').astype(np.float32)[:NumberOfParams]
        return state, frame_counter, float(reward), (fire, zap, spinner), not (game_mode & 0x80), is_done, save_signal
    except Exception as e:
        print(f"Error processing frame data: {e}")
        return None, 0, 0.0, None, False, False, 0

def train_bc(model, state, fire_target, zap_target, spinner_target):
    """Train BC model on a single demonstration frame with enhanced spinner value handling."""
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Apply a bias towards smaller spinner values in training data
    normalized_spinner = max(-127, min(127, spinner_target)) / 127.0
    
    # Higher weight for spinner component to emphasize its importance
    spinner_weight = 1.5
    
    targets = torch.tensor([
        [float(fire_target), float(zap_target), normalized_spinner]], 
        dtype=torch.float32
    ).to(device)
    
    preds = model(state_tensor)
    
    # Weighted loss: emphasize spinner accuracy
    fire_zap_loss = F.binary_cross_entropy(preds[:, :2], targets[:, :2])
    spinner_loss = F.mse_loss(preds[:, 2:], targets[:, 2:]) * spinner_weight
    loss = fire_zap_loss + spinner_loss
    
    model.optimizer.zero_grad()
    loss.backward()
    model.optimizer.step()
    
    actions = preds.detach().cpu().numpy()[0].astype(np.float32)
    
    # Apply the easing function to prefer smaller spinner values during inference
    spinner_action = int(round(actions[2] * 127.0))
    if abs(spinner_action) > 20:
        spinner_action = int(np.sign(spinner_action) * (20 + (abs(spinner_action) - 20) * 0.5))
    
    return loss.item(), encode_action(int(actions[0] > 0.5), int(actions[1] > 0.5), spinner_action)

def safe_add_to_buffer(buffer, obs, next_obs, action, reward, done):
    """Add experience to RL replay buffer with validation."""
    try:
        # Normalize input arrays
        obs = np.asarray(obs, dtype=np.float32).flatten()
        next_obs = np.asarray(next_obs, dtype=np.float32).flatten()
        action = np.asarray(action, dtype=np.float32).flatten()
        
        # Ensure correct shapes
        if obs.shape != (NumberOfParams,) or next_obs.shape != (NumberOfParams,) or action.shape != (3,):
            obs = np.resize(obs, NumberOfParams)
            next_obs = np.resize(next_obs, NumberOfParams)
            action = np.resize(action, 3)
        
        # Convert to scalar values for reward and done
        reward_scalar = float(reward)
        done_scalar = bool(done)
        
        # For newer SB3 versions with tensor-based replay buffer
        if hasattr(buffer, 'device') and hasattr(buffer, '_store_transition'):
            # Make sure the device for the replay buffer data is consistent
            obs_tensor = torch.as_tensor(obs, device=buffer.device)
            next_obs_tensor = torch.as_tensor(next_obs, device=buffer.device)
            action_tensor = torch.as_tensor(action, device=buffer.device)
            reward_tensor = torch.as_tensor(reward_scalar, device=buffer.device)
            done_tensor = torch.as_tensor(done_scalar, device=buffer.device)
            
            # Use scalar tensors for reward and done (no extra dimensions)
            buffer._store_transition(
                obs_tensor, next_obs_tensor, action_tensor, 
                reward_tensor, done_tensor, [{}]
            )
        else:
            # For older SB3 versions with numpy-based replay buffer
            buffer.add(
                obs, next_obs, action, reward_scalar, done_scalar,
                [{}] if hasattr(buffer, 'handle_timeout_termination') else None
            )
        
        return True
    except Exception as e:
        print(f"Error adding to buffer: {e}")
        return False

def apply_minimal_compatibility_patches(model):
    """Ensure SB3 model has a logger and moves all tensors to the correct device."""
    if not hasattr(model, '_logger'):
        model._logger = Logger(folder=None, output_formats=[])
    
    # Move all model components to the specified device
    model.policy.to(device)
    model.actor.to(device)
    model.critic.to(device)
    
    # Fix potential shape issues in critic target networks
    if hasattr(model, 'critic_target'):
        model.critic_target.to(device)
    
    # Set replay buffer device
    if hasattr(model.replay_buffer, 'device'):
        model.replay_buffer.device = device
    
    # For SB3 v1.8.0+, ensure all tensors in the replay buffer are on the correct device
    if hasattr(model.replay_buffer, '_tensor_names'):
        for tensor_name in model.replay_buffer._tensor_names:
            if hasattr(model.replay_buffer, tensor_name):
                tensor = getattr(model.replay_buffer, tensor_name)
                if isinstance(tensor, torch.Tensor) and tensor.device != device:
                    setattr(model.replay_buffer, tensor_name, tensor.to(device))
    
    # Handle SAC-specific parameters
    if hasattr(model, 'ent_coef_optimizer'):
        # Ensure log_ent_coef is properly initialized and on correct device
        if hasattr(model, 'log_ent_coef') and isinstance(model.log_ent_coef, torch.nn.Parameter):
            model.log_ent_coef.data = model.log_ent_coef.data.to(device)
    
    # Check if there are any issues with target entropy shape
    if hasattr(model, 'target_entropy'):
        # Make sure target_entropy is a scalar and not an array
        if isinstance(model.target_entropy, (torch.Tensor, np.ndarray)) and hasattr(model.target_entropy, 'shape'):
            if len(model.target_entropy.shape) > 0 and model.target_entropy.shape[0] > 1:
                # Convert to scalar if it's not already
                if isinstance(model.target_entropy, torch.Tensor):
                    model.target_entropy = model.target_entropy[0].item()
                else:
                    model.target_entropy = float(model.target_entropy[0])

def replay_log_file(log_file_path, bc_model):
    """Train BC model by replaying demonstration log file."""
    if not os.path.exists(log_file_path):
        print(f"Error: Log file {log_file_path} not found")
        return False
    print(f"Replaying {log_file_path}")
    frame_count = 0
    with open(log_file_path, 'rb') as f:
        while frame_count < 10000:  # Limit replay frames
            header_size_bytes = f.read(4)
            if not header_size_bytes or len(header_size_bytes) < 4:
                break
            header_size = struct.unpack(">I", header_size_bytes)[0]
            payload_size_bytes = f.read(4)
            if not payload_size_bytes or len(payload_size_bytes) < 4:
                break
            payload_size = struct.unpack(">I", payload_size_bytes)[0]
            payload = f.read(payload_size)
            if len(payload) < payload_size:
                break
            state, _, _, game_action, _, _, _ = process_frame_data(payload)
            if state is not None and game_action:
                loss, _ = train_bc(bc_model, state, *game_action)
                frame_count += 1
                if frame_count % 1000 == 0:
                    print(f"Processed {frame_count} frames, BC Loss: {loss:.6f}")
    print(f"Replay complete: {frame_count} frames")
    torch.save(bc_model.state_dict(), BC_MODEL_PATH)
    return True

def background_train(rl_model):
    """Background RL training thread with queue-based triggering."""
    while True:
        try:
            training_queue.get(timeout=1.0)
            if rl_model.replay_buffer.pos > rl_model.learning_starts:
                # Ensure all components are on the specified device before training
                apply_minimal_compatibility_patches(rl_model)
                
                # Start with smaller batch size and gradually increase to avoid broadcasting issues
                buffer_size = rl_model.replay_buffer.pos
                # Use a smaller batch size to start, ensuring we don't have dimension mismatch
                if buffer_size < 100:
                    batch_size = 8
                elif buffer_size < 500:
                    batch_size = 16
                elif buffer_size < 1000:
                    batch_size = 32
                else:
                    batch_size = 64
                
                try:
                    # Wrap training in another try-except to catch tensor shape issues
                    rl_model.train(gradient_steps=10, batch_size=batch_size)
                    
                    if hasattr(rl_model, 'logger'):
                        log_vals = rl_model.logger.name_to_value
                        actor_losses.append(log_vals.get('train/actor_loss', float('nan')))
                        critic_losses.append(log_vals.get('train/critic_loss', float('nan')))
                        ent_coefs.append(log_vals.get('train/ent_coef', float('nan')))
                except RuntimeError as rt_err:
                    if "doesn't match the broadcast shape" in str(rt_err):
                        print(f"Broadcasting error with batch size {batch_size}, retrying with smaller batch...")
                        # Retry with smaller batch size
                        if batch_size > 8:
                            reduced_batch = max(8, batch_size // 2)
                            try:
                                rl_model.train(gradient_steps=5, batch_size=reduced_batch)
                            except Exception as retry_err:
                                print(f"Still failed with reduced batch size: {retry_err}")
                    else:
                        print(f"Training runtime error: {rt_err}")
            
            training_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Training error: {e}")
            # Try to fix device issues on error
            try:
                apply_minimal_compatibility_patches(rl_model)
            except Exception as fix_error:
                print(f"Error fixing device issues: {fix_error}")
            training_queue.task_done()

def main():
    """Main execution loop for Tempest AI."""
    env = TempestEnv()
    bc_model, rl_model = initialize_models(env)
    threading.Thread(target=background_train, args=(rl_model,), daemon=True).start()

    if args.replay:
        replay_log_file(args.replay, bc_model)
        if not os.path.exists(LUA_TO_PY_PIPE):
            return

    for pipe in [LUA_TO_PY_PIPE, PY_TO_LUA_PIPE]:
        if os.path.exists(pipe): os.unlink(pipe)
        os.mkfifo(pipe)
        os.chmod(pipe, 0o666)

    print("Pipes created. Waiting for Lua...")
    frame_count = 0
    with os.fdopen(os.open(LUA_TO_PY_PIPE, os.O_RDONLY | os.O_NONBLOCK), "rb") as lua_to_py, \
         open(PY_TO_LUA_PIPE, "wb") as py_to_lua:
        print("✔️ Lua connected.")
        
        # Track spinner movements for analytics
        spinner_history = deque(maxlen=100)
        
        while True:
            try:
                data = lua_to_py.read()
                if not data:
                    time.sleep(0.01)
                    continue
                state, _, reward, game_action, is_attract, done, save_signal = process_frame_data(data)
                if state is None:
                    continue

                env.update_state(state, reward, game_action, done)
                if done:
                    env.reset()
                    frame_count = 0

                # Add to replay buffer with careful device handling
                if not is_attract and env.prev_state is not None and env.prev_action is not None:
                    safe_add_to_buffer(rl_model.replay_buffer, env.prev_state, env.state, env.prev_action, reward, done)

                if is_attract:
                    action = train_bc(bc_model, state, *game_action)[1] if game_action else encode_action(0, 0, 0)
                else:
                    # Get BC or RL prediction with device consistency and bias towards smaller spinner movements
                    if frame_count < 5000 and random.random() < 0.2:
                        # BC action for exploration
                        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                        action = bc_model(state_tensor).detach().cpu().numpy()[0]
                    else:
                        # RL action with exploration noise biased towards smaller spinner deltas
                        state_tensor = torch.tensor(state.reshape(1, -1), dtype=torch.float32)
                        raw_action = rl_model.predict(state_tensor, deterministic=frame_count > 10000)[0].flatten()
                        
                        # Apply additional spinner bias based on observed game state patterns
                        # This creates a "guided exploration" that focuses on smaller movements
                        if not rl_model.predict(state_tensor, deterministic=True)[0].flatten()[2] == raw_action[2]:
                            # We're in exploration mode for spinner
                            # Bias exploration toward smaller deltas (less than 20) using a beta distribution
                            spinner_scale = 0.15  # Controls the scale of typical random movements
                            spinner_delta = np.random.beta(1.5, 3.0) * 2.0 - 1.0  # Beta dist centered and scaled
                            raw_action[2] = spinner_delta * spinner_scale
                            
                        action = raw_action

                env.step(action)
                fire, zap, spinner = env.decode_action(action)
                
                # Track spinner actions for analytics
                spinner_history.append(spinner)
                
                py_to_lua.write(struct.pack("bbb", fire, zap, spinner))
                py_to_lua.flush()

                if frame_count % 10 == 0 and rl_model.replay_buffer.pos > rl_model.learning_starts:
                    training_queue.put(True)
                    mean_rewards.append(reward)

                if frame_count % 100 == 0:
                    print(f"Frame {frame_count}, Reward: {reward}, Done: {done}, Buffer Size: {rl_model.replay_buffer.pos}")
                    actor_loss_mean = np.nanmean(actor_losses) if actor_losses else np.nan
                    critic_loss_mean = np.nanmean(critic_losses) if critic_losses else np.nan
                    ent_coef_mean = np.nanmean(ent_coefs) if ent_coefs else np.nan
                    reward_mean = np.nanmean(mean_rewards) if mean_rewards else np.nan
                    
                    # Report spinner statistics
                    if spinner_history:
                        spinner_abs = [abs(s) for s in spinner_history]
                        spinner_avg = np.mean(spinner_abs)
                        spinner_max = np.max(spinner_abs)
                        print(f"Spinner stats - Avg: {spinner_avg:.2f}, Max: {spinner_max}, "
                              f"% small moves (<10): {sum(1 for s in spinner_abs if s < 10)/len(spinner_abs)*100:.1f}%")
                    
                    print(f"Metrics - Actor Loss: {actor_loss_mean:.6f}, Critic Loss: {critic_loss_mean:.6f}, "
                          f"Entropy Coef: {ent_coef_mean:.6f}, Mean Reward: {reward_mean:.2f}")

                if save_signal:
                    save_models(rl_model, bc_model)
                frame_count += 1

            except KeyboardInterrupt:
                save_models(rl_model, bc_model)
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(5)

def initialize_models(env):
    """Initialize models with improved SAC parameters."""
    bc_model = BCModel()
    if os.path.exists(BC_MODEL_PATH):
        bc_model.load_state_dict(torch.load(BC_MODEL_PATH, map_location=device))
        print(f"Loaded BC model from {BC_MODEL_PATH}")

    # Set replay buffer's device during initialization
    rl_model = SAC("MlpPolicy", env, policy_kwargs={
        "features_extractor_class": TempestFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": 128},
        "net_arch": dict(pi=[256, 128], qf=[256, 128])
    }, learning_rate=0.001, buffer_size=100000, learning_starts=1000, batch_size=64,
                   train_freq=(10, "step"), gradient_steps=10, device=device, verbose=1, tau=0.005)
    
    if os.path.exists(LATEST_MODEL_PATH):
        rl_model = SAC.load(LATEST_MODEL_PATH, env=env, device=device)
        print(f"Loaded RL model from {LATEST_MODEL_PATH}")
    
    # Make sure the model is properly set up with consistent device placement
    apply_minimal_compatibility_patches(rl_model)
    return bc_model, rl_model

def save_models(rl_model, bc_model):
    """Save RL and BC models to disk."""
    try:
        # Ensure consistent device state before saving
        apply_minimal_compatibility_patches(rl_model)
        rl_model.save(LATEST_MODEL_PATH)
        torch.save(bc_model.state_dict(), BC_MODEL_PATH)
        print(f"Models saved to {MODEL_DIR}")
    except Exception as e:
        print(f"Error saving models: {e}")

def encode_action(fire, zap, spinner_delta):
    """Encode actions for RL model consistency with easing function for spinner values."""
    # Apply an easing function to encourage smaller spinner deltas
    # Use tanh to squash large values while preserving sign
    normalized_spinner = np.tanh(spinner_delta / 20.0) * 127.0
    return np.array([float(fire), float(zap), max(-1.0, min(1.0, normalized_spinner / 127.0))], dtype=np.float32)

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    main()