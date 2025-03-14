#!/usr/bin/env python3
"""
Tempest AI Model with BC-to-RL Transition
Author: Dave Plummer (davepl) and various AI assists
Date: 2023-03-06 (Updated)

This script implements a hybrid AI model for the Tempest arcade game that:
1. Uses Behavioral Cloning (BC) during attract mode to learn from the game's AI
2. Uses Reinforcement Learning (RL) during actual gameplay
3. Transfers knowledge from BC to RL for efficient learning
"""

import os
import sys
import time
import struct
import random
import stat
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
# Add imports for Stable Baselines3
from stable_baselines3 import DQN
from stable_baselines3.common.buffers import ReplayBuffer as SB3ReplayBuffer
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
import datetime

# Define global shutdown tracking variable
shutdown_requested = False

# Define the paths to the named pipes
LUA_TO_PY_PIPE = "/tmp/lua_to_py"
PY_TO_LUA_PIPE = "/tmp/py_to_lua"

# Define action mapping
ACTION_MAP = {
    0: "fire",
    1: "zap",
    2: "left",
    3: "right",
    4: "none"
}

# Create a directory for model checkpoints
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
if not os.path.exists(MODEL_DIR):
    try:
        os.makedirs(MODEL_DIR)
        print(f"Created model directory at {MODEL_DIR}")
    except Exception as e:
        print(f"Warning: Could not create models directory: {e}")
        # Fall back to using a directory in the user's home folder
        MODEL_DIR = os.path.expanduser("~/tempest_models")
        if not os.path.exists(MODEL_DIR):
            try:
                os.makedirs(MODEL_DIR)
                print(f"Created fallback model directory at {MODEL_DIR}")
            except Exception as e2:
                print(f"Critical error: Could not create fallback directory: {e2}")
                # Last resort - use /tmp which should be writable
                MODEL_DIR = "/tmp/tempest_models"
                if not os.path.exists(MODEL_DIR):
                    os.makedirs(MODEL_DIR)
                print(f"Using temporary directory for models: {MODEL_DIR}")

# Define paths for latest models and checkpoints
LATEST_MODEL_PATH = os.path.join(MODEL_DIR, "tempest_model_latest.zip")
BC_MODEL_PATH = os.path.join(MODEL_DIR, "tempest_bc_model.pt")

# Add model versioning to track compatibility
MODEL_VERSION = "1.0.0"

# Add diagnostic filenames for troubleshooting
MODEL_LOAD_DIAGNOSTICS = os.path.join(MODEL_DIR, "model_load_diagnostics.txt")

# Print model paths for debugging
print(f"Model paths:\n  RL: {LATEST_MODEL_PATH}\n  BC: {BC_MODEL_PATH}")

# Function to write diagnostic information for troubleshooting
def write_diagnostic_info(message, error=None, model_path=None):
    """Write detailed diagnostic information to help troubleshoot model loading issues"""
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(MODEL_LOAD_DIAGNOSTICS, "a") as f:
            f.write(f"\n\n=== {timestamp} ===\n")
            f.write(f"{message}\n")
            
            if error:
                f.write(f"Error: {str(error)}\n")
                if hasattr(error, "__traceback__"):
                    import traceback
                    tb_str = "".join(traceback.format_exception(type(error), error, error.__traceback__))
                    f.write(f"Traceback:\n{tb_str}\n")
            
            if model_path and os.path.exists(model_path):
                file_size = os.path.getsize(model_path)
                file_time = datetime.datetime.fromtimestamp(os.path.getmtime(model_path))
                f.write(f"Model file: {model_path}\n")
                f.write(f"  - Size: {file_size} bytes\n")
                f.write(f"  - Modified: {file_time}\n")
                
                # For zip files, try to list contents
                if model_path.endswith(".zip"):
                    try:
                        import zipfile
                        with zipfile.ZipFile(model_path, 'r') as zip_ref:
                            f.write("  - Contents:\n")
                            for info in zip_ref.infolist():
                                f.write(f"    - {info.filename} ({info.file_size} bytes)\n")
                    except Exception as zip_error:
                        f.write(f"  - Error reading zip contents: {zip_error}\n")
        
        print(f"Diagnostic information written to {MODEL_LOAD_DIAGNOSTICS}")
    except Exception as diag_error:
        print(f"Error writing diagnostics: {diag_error}")

class TempestEnv(gym.Env):
    """
    Custom Gymnasium environment for Tempest arcade game.
    """
    metadata = {"render_modes": ["human"], "render_fps": 60}
    
    def __init__(self):
        super().__init__()
        
        # Define action space: fire, zap, left, right, none
        self.action_space = spaces.Discrete(5)
        
        # Define observation space - 243 features based on game state
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(243,), dtype=np.float32)
        
        # Initialize state
        self.state = np.zeros(243, dtype=np.float32)
        self.reward = 0
        self.done = False
        self.info = {}
        self.episode_step = 0
        self.total_reward = 0
        self.is_attract_mode = False
        self.prev_state = None
        
        print("Tempest Gymnasium environment initialized")
    
    def reset(self, seed=None, options=None):
        """Reset the environment to start a new episode."""
        super().reset(seed=seed)
        
        self.state = np.zeros(243, dtype=np.float32)
        self.reward = 0
        self.done = False
        self.episode_step = 0
        self.total_reward = 0
        self.prev_state = None
        
        return self.state, self.info
    
    def step(self, action):
        """
        Take a step in the environment using the given action.
        """
        self.episode_step += 1
        self.total_reward += self.reward
        
        terminated = self.done  # Use Lua-provided done flag
        truncated = self.episode_step >= 10000  # Episode too long
        
        self.info = {
            "action_taken": ACTION_MAP[action],
            "episode_step": self.episode_step,
            "total_reward": self.total_reward,
            "attract_mode": self.is_attract_mode
        }
        
        return self.state, self.reward, terminated, truncated, self.info
    
    def update_state(self, new_state, reward, game_action=None, done=False):
        """
        Update the environment state with new data from the game.
        """
        self.prev_state = self.state.copy() if self.state is not None else None
        self.state = new_state
        self.reward = reward
        self.done = done
        if game_action is not None:
            self.info["game_action"] = game_action
        return self.state

class CustomReplayBuffer:
    """Custom replay buffer that can handle both BC and RL transitions"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.bc_buffer = deque(maxlen=capacity // 2)  # Separate buffer for BC samples
    
    def add(self, state, action, reward, next_state, done, is_bc=False):
        # Store in appropriate buffer
        if is_bc:
            self.bc_buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size, bc_ratio=0.3):
        """Sample from both buffers with a given BC ratio"""
        if len(self.bc_buffer) < batch_size * bc_ratio or len(self.buffer) < batch_size * (1 - bc_ratio):
            # Not enough samples in one of the buffers, sample from what's available
            all_samples = list(self.buffer) + list(self.bc_buffer)
            if len(all_samples) < batch_size:
                return None  # Not enough samples total
            samples = random.sample(all_samples, batch_size)
        else:
            # Sample from both buffers according to the ratio
            bc_samples = random.sample(list(self.bc_buffer), int(batch_size * bc_ratio))
            rl_samples = random.sample(list(self.buffer), batch_size - len(bc_samples))
            samples = bc_samples + rl_samples
        
        states, actions, rewards, next_states, dones = zip(*samples)
        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(next_states), np.array(dones, dtype=np.float32))
    
    def __len__(self):
        return len(self.buffer) + len(self.bc_buffer)

# Custom feature extractor for Stable Baselines3
class TempestFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        self.feature_extractor = nn.Sequential(
            nn.Linear(observation_space.shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations):
        return self.feature_extractor(observations)

# Custom BC model
class BCModel(nn.Module):
    def __init__(self, input_size=243, output_size=5):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x):
        return self.model(x)
    
    def predict(self, state):
        with torch.no_grad():
            logits = self.forward(torch.FloatTensor(state))
            return logits.argmax().item()

# Custom callback for saving the model with compatibility patches
class SaveOnSignalCallback(BaseCallback):
    def __init__(self, save_path, bc_model=None, verbose=1):
        # Safe initialization for logger compatibility across SB3 versions
        try:
            super().__init__(verbose)
        except AttributeError as e:
            if "property 'logger' of 'BaseCallback' object has no setter" in str(e):
                # Initialize without calling parent constructor
                self._verbose = verbose
                # Create a minimal logger implementation that supports record() but does nothing
                from stable_baselines3.common.logger import Logger
                self.__dict__['_logger'] = Logger(folder=None, output_formats=[])
            else:
                raise
                
        self.save_path = save_path
        self.bc_model = bc_model
        self.force_save = False
        self.model = None  # Will be set when callback is initialized
    
    def _on_step(self):
        if self.force_save:
            self._save_models()
            self.force_save = False
        return True
    
    def _save_models(self):
        """Internal method to actually save the models"""
        print("\n" + "=" * 50)
        print("SAVE OPERATION STARTING")
        print(f"RL model path: {self.save_path}")
        print(f"BC model path: {BC_MODEL_PATH}")
        print(f"Models directory: {MODEL_DIR}")
        print("=" * 50 + "\n")
        
        # Check if directory exists and is writable
        if not os.path.exists(MODEL_DIR):
            print(f"ERROR: Model directory {MODEL_DIR} does not exist!")
            try:
                os.makedirs(MODEL_DIR, exist_ok=True)
                print(f"Created directory {MODEL_DIR}")
            except Exception as e:
                print(f"ERROR: Could not create directory: {e}")
        
        if not os.access(MODEL_DIR, os.W_OK):
            print(f"ERROR: Model directory {MODEL_DIR} is not writable!")
        
        try:
            # Save RL model
            print("Attempting to save RL model...")
            
            # Verify model is valid before saving
            if self.model is None:
                raise ValueError("Model reference is None - cannot save")
            
            # Add version metadata before saving
            if not hasattr(self.model, 'model_version'):
                self.model.model_version = MODEL_VERSION
            if not hasattr(self.model, 'model_timestamp'):
                self.model.model_timestamp = datetime.datetime.now().isoformat()
            
            # Create a temporary file first to avoid corrupting existing model
            temp_path = f"{self.save_path}.temp"
            self.model.save(temp_path)
            
            # Verify the temporary file
            if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                raise ValueError(f"Failed to create valid temporary model file at {temp_path}")
            
            # Now safely move it to the real location
            import shutil
            shutil.move(temp_path, self.save_path)
            
            print(f"RL model saved to {self.save_path}")
            
            # Verify RL model was saved
            if os.path.exists(self.save_path):
                print(f"VERIFIED: RL model file exists at {self.save_path}")
                print(f"File size: {os.path.getsize(self.save_path)} bytes")
            else:
                print(f"ERROR: RL model file does not exist after save attempt!")
            
            # Also save a timestamped backup
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{self.save_path.replace('.zip', '')}_{timestamp}.zip"
            
            # Use direct copy for backup to avoid potential model corruption
            shutil.copy2(self.save_path, backup_path)
            print(f"RL model backup saved to {backup_path}")
            
            # Save BC model if provided
            if self.bc_model is not None:
                print("Attempting to save BC model...")
                bc_path = BC_MODEL_PATH
                try:
                    # Save to temporary file first
                    temp_bc_path = f"{bc_path}.temp"
                    torch.save(self.bc_model.state_dict(), temp_bc_path)
                    
                    # Verify temp file is valid
                    if not os.path.exists(temp_bc_path) or os.path.getsize(temp_bc_path) == 0:
                        raise ValueError(f"Failed to create valid temporary BC model file")
                    
                    # Move to final location
                    shutil.move(temp_bc_path, bc_path)
                    print(f"BC model saved to {bc_path}")
                    
                    # Verify BC model was saved
                    if os.path.exists(bc_path):
                        print(f"VERIFIED: BC model file exists at {bc_path}")
                        print(f"File size: {os.path.getsize(bc_path)} bytes")
                    else:
                        print(f"ERROR: BC model file does not exist after save attempt!")
                    
                    # Also save a timestamped backup for BC model
                    bc_backup_path = f"{bc_path.replace('.pt', '')}_{timestamp}.pt"
                    shutil.copy2(bc_path, bc_backup_path)
                    print(f"BC model backup saved to {bc_backup_path}")
                except Exception as bc_error:
                    print(f"ERROR saving BC model: {bc_error}")
                    import traceback
                    traceback.print_exc()
            
            # Write a marker file to confirm saving worked and include model version info
            marker_file = os.path.join(MODEL_DIR, f"save_confirmed_{timestamp}.txt")
            try:
                with open(marker_file, "w") as f:
                    f.write(f"Models saved successfully at {datetime.datetime.now().isoformat()}\n")
                    f.write(f"Model Version: {MODEL_VERSION}\n")
                    f.write(f"RL Model: {self.save_path}\n")
                    f.write(f"RL File Size: {os.path.getsize(self.save_path)} bytes\n")
                    if self.bc_model is not None:
                        f.write(f"BC Model: {bc_path}\n")
                        f.write(f"BC File Size: {os.path.getsize(bc_path)} bytes\n")
                print(f"Confirmation file written to {marker_file}")
            except Exception as marker_error:
                print(f"ERROR writing confirmation file: {marker_error}")
            
            # List all files in the models directory
            print("\nCurrent files in models directory:")
            try:
                files = os.listdir(MODEL_DIR)
                for file in files:
                    file_path = os.path.join(MODEL_DIR, file)
                    file_size = os.path.getsize(file_path)
                    print(f"  - {file} ({file_size} bytes)")
            except Exception as e:
                print(f"ERROR listing directory contents: {e}")
            
            print("\nSAVE COMPLETED SUCCESSFULLY")
        except Exception as e:
            print(f"ERROR saving models: {e}")
            import traceback
            traceback.print_exc()
            
            # Record diagnostic information about the failed save
            write_diagnostic_info("Failed to save models", error=e)
            
            # Try an alternative save location
            alt_dir = os.path.expanduser("~/tempest_models_fallback")
            print(f"\nAttempting emergency save to alternative location: {alt_dir}")
            try:
                os.makedirs(alt_dir, exist_ok=True)
                alt_rl_path = os.path.join(alt_dir, "tempest_model_emergency.zip")
                alt_bc_path = os.path.join(alt_dir, "tempest_bc_model_emergency.pt")
                
                if self.model is not None:
                    self.model.save(alt_rl_path)
                if self.bc_model is not None:
                    torch.save(self.bc_model.state_dict(), alt_bc_path)
                
                print(f"Emergency save successful to {alt_dir}")
                print(f"  - RL model: {alt_rl_path}")
                print(f"  - BC model: {alt_bc_path}")
            except Exception as alt_error:
                print(f"ERROR during emergency save: {alt_error}")
    
    def signal_save(self):
        """Signal that models should be saved - now saves immediately"""
        print("Save signal received - saving models immediately")
        self._save_models()  # Save immediately instead of waiting for next step
        self.force_save = False  # Just in case _on_step is called

def process_frame_data(data):
    """
    Process the binary frame data received from Lua.
    
    Args:
        data (bytes): Binary data containing OOB header and game state information
        
    Returns:
        tuple: (processed_data, frame_counter, reward, game_action, is_attract, done, save_signal)
    """
    global shutdown_requested
    
    if len(data) < 24:  # Header (4+8) + action (1) + mode (1) + done (1) + frame_counter (4) + score (4) + save_signal (1)
        print(f"Warning: Data too small ({len(data)} bytes)")
        return None, 0, 0.0, None, False, False, False
    
    try:
        # Extract out-of-band information
        num_oob_values = struct.unpack(">I", data[0:4])[0]
        reward = struct.unpack(">d", data[4:12])[0]
        game_action = struct.unpack(">B", data[12:13])[0]
        game_mode = struct.unpack(">B", data[13:14])[0]
        done = struct.unpack(">B", data[14:15])[0] != 0
        frame_counter = struct.unpack(">I", data[15:19])[0]  # Extract 32-bit frame counter
        score = struct.unpack(">I", data[19:23])[0]  # Extract 32-bit score
        save_signal = struct.unpack(">B", data[23:24])[0] != 0  # Read dedicated save signal byte
        
        # Debug output for game mode occasionally
        if random.random() < 0.01 or save_signal:  # Show debug info about 1% of the time or if save signal
            print(f"Game Mode: 0x{game_mode:02X}, Is Attract Mode: {(game_mode & 0x80) == 0}, Save Signal: {save_signal}")
            print(f"OOB Data: values={num_oob_values}, reward={reward:.2f}, action={game_action}, done={done}")
            print(f"Frame Counter: {frame_counter}, Score: {score}")
        
        # Make save signal very visible when it happens
        if save_signal:
            print("\n" + "!" * 50)
            print("!!! SAVE SIGNAL RECEIVED FROM LUA !!!")
            print("!" * 50 + "\n")
            
            # If this save signal is from an on_mame_exit callback, mark it as a shutdown request
            if done or frame_counter % 100 == 99:  # Simple heuristic to detect shutdown saves (done or certain frame patterns)
                shutdown_requested = True
                print("DETECTED POSSIBLE SHUTDOWN - Models will be saved immediately")
        
        # Calculate header size: 4 bytes for count + (num_oob_values * 8) bytes for values + 3 bytes for extra data + 8 bytes for frame_counter and score + 1 byte for save signal
        header_size = 4 + (num_oob_values * 8) + 3 + 8 + 1
        
        # Extract game state data (everything after the header)
        game_data = data[header_size:]
        
        # Calculate how many 16-bit integers we have in the game data
        num_ints = len(game_data) // 2
        
        # Unpack the binary data into integers
        unpacked_data = []
        for i in range(num_ints):
            value = struct.unpack(">H", game_data[i*2:i*2+2])[0]
            # Convert from offset encoding (values were sent with +32768)
            value = value - 32768
            unpacked_data.append(value)
        
        # Normalize the data to -1 to 1 range for the neural network
        normalized_data = np.array([float(x) / 32767.0 if x > 0 else float(x) / 32768.0 for x in unpacked_data], dtype=np.float32)
        
        # Check if we're in attract mode (bit 0x80 of game_mode is clear)
        is_attract = (game_mode & 0x80) == 0
        
        # Debug output for attract mode transitions
        if hasattr(process_frame_data, 'last_attract_mode') and process_frame_data.last_attract_mode != is_attract:
            print(f"ATTRACT MODE TRANSITION: {'Attract → Play' if not is_attract else 'Play → Attract'}")
            print(f"Game Mode: 0x{game_mode:02X}, Is Attract: {is_attract}")
            print(f"Frame: {frame_counter}, Score: {score}")
        
        # Store for next comparison
        process_frame_data.last_attract_mode = is_attract
        
        # Pad or truncate to match the expected observation space size
        expected_size = 243  # Updated to match new state size
        if len(normalized_data) < expected_size:
            padded_data = np.zeros(expected_size, dtype=np.float32)
            padded_data[:len(normalized_data)] = normalized_data
            normalized_data = padded_data
        elif len(normalized_data) > expected_size:
            normalized_data = normalized_data[:expected_size]
        
        return normalized_data, frame_counter, reward, game_action, is_attract, done, save_signal
    
    except Exception as e:
        print(f"Error processing frame data: {e}")
        import traceback
        traceback.print_exc()
        return None, 0, 0.0, None, False, False, False

# Initialize static variable for process_frame_data
process_frame_data.last_attract_mode = True

def train_bc(model, state, action):
    """Train the BC model using demonstration data"""
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    action_tensor = torch.LongTensor([action])
    
    # Forward pass
    logits = model(state_tensor)
    
    # Calculate loss
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits, action_tensor)
    
    # Backpropagation
    model.optimizer.zero_grad()
    loss.backward()
    model.optimizer.step()
    
    return loss.item()

def initialize_models():
    """Initialize both BC and RL models with robust error handling and diagnostics"""
    # Create the environment
    env = TempestEnv()
    
    print("\n===== MODEL INITIALIZATION =====")
    print(f"Model version: {MODEL_VERSION}")
    print(f"Models directory: {MODEL_DIR}")
    
    # Save backup of any existing models before loading
    backup_existing_models = False  # Set to True if you want automatic backups before loading
    if backup_existing_models:
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            if os.path.exists(LATEST_MODEL_PATH):
                backup_rl = f"{LATEST_MODEL_PATH}.bak_{timestamp}"
                import shutil
                shutil.copy2(LATEST_MODEL_PATH, backup_rl)
                print(f"Backed up RL model to {backup_rl}")
            if os.path.exists(BC_MODEL_PATH):
                backup_bc = f"{BC_MODEL_PATH}.bak_{timestamp}"
                shutil.copy2(BC_MODEL_PATH, backup_bc)
                print(f"Backed up BC model to {backup_bc}")
        except Exception as backup_error:
            print(f"Warning: Could not create backups: {backup_error}")
    
    # Initialize the BC model
    bc_model = BCModel(input_size=243)
    bc_model.optimizer = optim.Adam(bc_model.parameters(), lr=0.001)
    bc_loaded_successfully = False
    
    # Try to load existing BC model
    if os.path.exists(BC_MODEL_PATH):
        print(f"Found BC model file at {BC_MODEL_PATH} - Attempting to load...")
        try:
            # First verify file integrity
            if os.path.getsize(BC_MODEL_PATH) == 0:
                raise ValueError("BC model file is empty (0 bytes)")
            
            # Record file stats for diagnostics
            file_stats = f"File size: {os.path.getsize(BC_MODEL_PATH)} bytes, " \
                        f"Modified: {datetime.datetime.fromtimestamp(os.path.getmtime(BC_MODEL_PATH))}"
            print(f"BC model file stats: {file_stats}")
            
            # Attempt to load model with error handling
            start_time = time.time()
            state_dict = torch.load(BC_MODEL_PATH)
            
            # Optional: Validate state dict structure to prevent partial loads
            expected_keys = {'model.0.weight', 'model.0.bias', 'model.2.weight', 'model.2.bias', 
                            'model.4.weight', 'model.4.bias'}
            missing_keys = expected_keys - set(state_dict.keys())
            
            if missing_keys:
                raise ValueError(f"BC model is missing expected keys: {missing_keys}")
            
            # Apply state dict to model
            bc_model.load_state_dict(state_dict)
            load_time = time.time() - start_time
            
            print(f"Successfully loaded BC model from {BC_MODEL_PATH} in {load_time:.2f} seconds")
            
            # Additional validation - perform a test forward pass
            test_input = torch.zeros(1, 243, dtype=torch.float32)
            with torch.no_grad():
                test_output = bc_model(test_input)
                if test_output.shape != (1, 5):
                    raise ValueError(f"Model produced incorrect output shape: {test_output.shape}, expected (1, 5)")
            
            bc_loaded_successfully = True
            write_diagnostic_info(f"BC model loaded successfully", model_path=BC_MODEL_PATH)
            
        except Exception as e:
            print(f"ERROR loading BC model: {e}")
            import traceback
            traceback.print_exc()
            write_diagnostic_info("Failed to load BC model", error=e, model_path=BC_MODEL_PATH)
            print("Initializing fresh BC model due to loading error")
            # Reinitialize the model since loading failed
            bc_model = BCModel(input_size=243)
            bc_model.optimizer = optim.Adam(bc_model.parameters(), lr=0.001)
    else:
        print(f"No existing BC model found at {BC_MODEL_PATH}, starting fresh")
        write_diagnostic_info("No BC model file found - starting with new model")
    
    # Initialize the RL model with Stable Baselines3
    policy_kwargs = dict(
        features_extractor_class=TempestFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=[128, 64]
    )
    
    # Create checkpoint callback for regular saving
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,  # Save every 10000 steps
        save_path=os.path.join(MODEL_DIR, "checkpoints"),
        name_prefix="tempest_dqn"
    )
    
    # Create our custom save-on-signal callback with BC model reference
    save_signal_callback = SaveOnSignalCallback(save_path=LATEST_MODEL_PATH, bc_model=bc_model)
    
    # Create initial DQN model
    rl_model = DQN(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=0.0001,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.5,
        exploration_initial_eps=0.5,
        exploration_final_eps=0.1,
        tensorboard_log=os.path.join(MODEL_DIR, "tensorboard_logs"),
        verbose=1
    )
    
    # Store the model parameters before loading for comparison
    rl_model_init_params = str(rl_model.policy)
    rl_loaded_successfully = False
    
    # Try to load existing RL model
    if os.path.exists(LATEST_MODEL_PATH):
        print(f"Found RL model file at {LATEST_MODEL_PATH} - Attempting to load...")
        try:
            # Verify file integrity
            if os.path.getsize(LATEST_MODEL_PATH) == 0:
                raise ValueError("RL model file is empty (0 bytes)")
            
            # Record file stats for diagnostics
            file_stats = f"File size: {os.path.getsize(LATEST_MODEL_PATH)} bytes, " \
                        f"Modified: {datetime.datetime.fromtimestamp(os.path.getmtime(LATEST_MODEL_PATH))}"
            print(f"RL model file stats: {file_stats}")
            
            # Try to load the zip file contents for diagnostics
            try:
                import zipfile
                with zipfile.ZipFile(LATEST_MODEL_PATH, 'r') as zip_ref:
                    print(f"RL model zip contents: {', '.join(zip_ref.namelist())}")
            except Exception as zip_error:
                print(f"Warning: Could not inspect zip contents: {zip_error}")
            
            # Attempt to load model with error handling
            start_time = time.time()
            loaded_model = DQN.load(LATEST_MODEL_PATH, env=env)
            
            # Validate the loaded model by checking a few expected attributes
            expected_attrs = ['policy', 'replay_buffer', 'observation_space', 'action_space']
            for attr in expected_attrs:
                if not hasattr(loaded_model, attr):
                    raise ValueError(f"Loaded model missing expected attribute: {attr}")
            
            load_time = time.time() - start_time
            print(f"Successfully loaded RL model from {LATEST_MODEL_PATH} in {load_time:.2f} seconds")
            
            # Replace our model with the loaded one
            rl_model = loaded_model
            
            # Compare model architecture before and after loading
            rl_model_loaded_params = str(rl_model.policy)
            is_architecture_same = rl_model_init_params.split('\n')[0] == rl_model_loaded_params.split('\n')[0]
            print(f"Model architecture unchanged: {is_architecture_same}")
            
            rl_loaded_successfully = True
            write_diagnostic_info("RL model loaded successfully", model_path=LATEST_MODEL_PATH)
            
        except Exception as e:
            print(f"ERROR loading RL model: {e}")
            import traceback
            traceback.print_exc()
            write_diagnostic_info("Failed to load RL model", error=e, model_path=LATEST_MODEL_PATH)
            print("Using fresh RL model due to loading error")
            # Continue with the initial model
    else:
        print(f"No existing RL model found at {LATEST_MODEL_PATH}, starting fresh")
        write_diagnostic_info("No RL model file found - starting with new model")
    
    # Set up callbacks
    rl_model.set_env(env)
    callbacks = [checkpoint_callback, save_signal_callback]
    
    # Add a version timestamp to the model to track when it was initialized/loaded
    save_signal_callback.model = rl_model
    rl_model.model_version = MODEL_VERSION
    rl_model.model_timestamp = datetime.datetime.now().isoformat()
    
    # Save metadata for debugging - create a readable file with model info
    try:
        metadata_file = os.path.join(MODEL_DIR, "model_metadata.txt")
        with open(metadata_file, "w") as f:
            f.write(f"Model Metadata - Generated: {datetime.datetime.now().isoformat()}\n")
            f.write(f"Model Version: {MODEL_VERSION}\n")
            f.write(f"RL Model Path: {LATEST_MODEL_PATH}\n")
            f.write(f"BC Model Path: {BC_MODEL_PATH}\n")
            f.write(f"RL Model Loaded Successfully: {rl_loaded_successfully}\n")
            f.write(f"BC Model Loaded Successfully: {bc_loaded_successfully}\n")
            f.write(f"RL Policy: {str(rl_model.policy)}\n")
            f.write(f"BC Model: {str(bc_model)}\n")
        print(f"Model metadata written to {metadata_file}")
    except Exception as metadata_error:
        print(f"Warning: Could not write model metadata: {metadata_error}")
    
    # Final status summary
    print("\n===== MODEL STATUS SUMMARY =====")
    print(f"BC Model: {'Loaded from file' if bc_loaded_successfully else 'New instance'}")
    print(f"RL Model: {'Loaded from file' if rl_loaded_successfully else 'New instance'}")
    print("===============================\n")
    
    # Apply Logger compatibility patches for stable_baselines3
    # This ensures we can work with different SB3 versions
    try:
        # Check if the model has a logger using hasattr (doesn't trigger property)
        has_logger = '_logger' in rl_model.__dict__ or (hasattr(rl_model, 'logger') and not isinstance(rl_model.__class__.logger, property))
        
        if not has_logger:
            # Create a minimal logger and attach it safely without using properties
            from stable_baselines3.common.logger import Logger
            dummy_logger = Logger(folder=None, output_formats=[])
            
            # Try to detect if we need _logger or logger
            if hasattr(rl_model.__class__, 'logger') and isinstance(rl_model.__class__.logger, property):
                # Logger is a property that probably relies on _logger
                print("Applying _logger compatibility patch")
                rl_model.__dict__['_logger'] = dummy_logger
            else:
                # Direct logger attribute
                print("Adding logger attribute directly")
                rl_model.__dict__['logger'] = dummy_logger
        
        # Create monkey-patched utility methods to avoid logger issues
        original_update_lr = rl_model._update_learning_rate if hasattr(rl_model, '_update_learning_rate') else None
        
        def safe_update_lr(optimizer):
            """Safe version of _update_learning_rate that doesn't require logger"""
            try:
                # Try original method first
                if original_update_lr:
                    return original_update_lr(optimizer)
                
                # Fallback implementation if original fails or doesn't exist
                if hasattr(rl_model, 'lr_schedule') and callable(rl_model.lr_schedule):
                    new_lr = rl_model.lr_schedule(rl_model._current_progress_remaining 
                                                 if hasattr(rl_model, '_current_progress_remaining') else 1.0)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr
            except Exception as lr_error:
                print(f"Warning: Learning rate update failed: {lr_error}")
            return
        
        # Store original method for potential restoration
        rl_model.__dict__['_original_update_lr'] = original_update_lr
        # Apply safe method
        rl_model.__dict__['_update_learning_rate'] = safe_update_lr
        
        print("Applied SB3 compatibility patches")
    except Exception as patch_error:
        print(f"Warning: Failed to apply compatibility patches: {patch_error}")
        import traceback
        traceback.print_exc()
    
    return env, bc_model, rl_model, save_signal_callback

def main():
    """
    Main function that handles the communication with Lua and processes game frames.
    """
    global shutdown_requested
    
    print("Python AI model starting with bidirectional BC/RL knowledge transfer using Stable Baselines3...")
    print(f"Models will be saved to directory: {MODEL_DIR}")
    
    # Print information about directory access
    try:
        if os.path.exists(MODEL_DIR):
            print(f"Model directory exists: {MODEL_DIR}")
            
            # Check if directory is writable
            if os.access(MODEL_DIR, os.W_OK):
                print("Model directory is writable")
                
                # Try creating a test file
                test_file = os.path.join(MODEL_DIR, "write_test.txt")
                try:
                    with open(test_file, "w") as f:
                        f.write("Write test")
                    print(f"Successfully created test file: {test_file}")
                    os.remove(test_file)
                    print("Successfully removed test file")
                except Exception as e:
                    print(f"ERROR: Could not write test file: {e}")
            else:
                print("WARNING: Model directory is NOT writable!")
        else:
            print(f"WARNING: Model directory does not exist: {MODEL_DIR}")
            try:
                os.makedirs(MODEL_DIR, exist_ok=True)
                print(f"Created directory: {MODEL_DIR}")
            except Exception as e:
                print(f"ERROR: Could not create model directory: {e}")
    except Exception as e:
        print(f"ERROR checking directory access: {e}")

    # Initialize models and environment
    env, bc_model, rl_model, save_signal_callback = initialize_models()
    
    # Create a custom replay buffer for mixed BC and RL transitions
    custom_buffer = CustomReplayBuffer(capacity=100000)
    
    # Stats tracking
    bc_episodes = 0
    rl_episodes = 0
    bc_losses = []
    rewards_history = []
    
    # Track mode transitions for logging
    last_mode_was_attract = True
    mode_transitions = 0
    
    # Create the pipes
    for pipe_path in [LUA_TO_PY_PIPE, PY_TO_LUA_PIPE]:
        try:
            if os.path.exists(pipe_path):
                os.unlink(pipe_path)
            os.mkfifo(pipe_path)
            os.chmod(pipe_path, 0o666)
            print(f"Created pipe: {pipe_path}")
        except OSError as e:
            print(f"Error with pipe {pipe_path}: {e}")
            sys.exit(1)
    
    print("Pipes created successfully. Waiting for Lua connection...")
    
    # Connection retry loop
    while True:
        try:
            # Open pipes in non-blocking mode to avoid deadlock
            fd = os.open(LUA_TO_PY_PIPE, os.O_RDONLY | os.O_NONBLOCK)
            lua_to_py = os.fdopen(fd, "rb")
            print("Input pipe opened successfully")
            
            py_to_lua = open(PY_TO_LUA_PIPE, "w")
            print("Output pipe opened successfully")
            
            try:
                frame_count = 0
                last_frame_time = time.time()
                fps = 0
                last_save_time = time.time()
                save_interval = 300  # Save every 5 minutes

                while True:
                    try:
                        # Read from pipe
                        data = lua_to_py.read()
                        
                        if not data:
                            time.sleep(0.01)
                            continue
                        
                        # Process the frame data
                        result = process_frame_data(data)
                        if result is None or result[0] is None:
                            print("Error processing frame data, skipping frame")
                            continue
                        
                        processed_data, frame_counter, reward, game_action, is_attract, done, save_signal = result
                        
                        # Update the environment state
                        env.update_state(processed_data, reward, game_action, done)
                        env.is_attract_mode = is_attract
                        
                        # Detect mode transition
                        if is_attract != last_mode_was_attract:
                            mode_transitions += 1
                            print(f"\n*** MODE TRANSITION #{mode_transitions}: {'Attract → Play' if not is_attract else 'Play → Attract'} ***")
                            print(f"Frame: {frame_counter}, Is Attract: {is_attract}, Game Action: {game_action}")
                            last_mode_was_attract = is_attract
                        
                        # Process save signal
                        if save_signal:
                            print("\n" + "!" * 50)
                            print("!!! SAVE SIGNAL RECEIVED FROM LUA !!!")
                            print("!" * 50 + "\n")
                            
                            # If this is a shutdown signal, set the flag
                            if done or frame_counter % 100 == 99:  # Simple heuristic to detect shutdown
                                shutdown_requested = True
                                print("DETECTED POSSIBLE SHUTDOWN")
                            
                            # Set model reference on the callback if not already set
                            if save_signal_callback.model is None:
                                save_signal_callback.model = rl_model
                            
                            # Signal immediate save
                            save_signal_callback.signal_save()
                        
                        # Different behavior based on mode
                        if is_attract:
                            # Behavioral Cloning mode
                            if game_action is not None and game_action < 5:
                                # Store transition for future mixed batches
                                if env.prev_state is not None:
                                    custom_buffer.add(env.prev_state, game_action, reward, processed_data, done, is_bc=True)
                                
                                # Learn from the game's action using BC
                                loss = train_bc(bc_model, processed_data, game_action)
                                bc_losses.append(loss)
                                
                                # Log progress occasionally
                                if frame_count % 100 == 0:
                                    print(f"BC training - Frame {frame_counter}, Action: {ACTION_MAP[game_action]}, Loss: {loss:.6f}")
                                
                                # In attract mode, use the game's action
                                action = game_action
                            else:
                                action = random.randint(0, 4)
                            
                            # Track BC episodes
                            if done:
                                bc_episodes += 1
                                avg_loss = np.mean(bc_losses[-100:]) if bc_losses else 0
                                print(f"BC Episode {bc_episodes} completed, avg loss: {avg_loss:.6f}")
                        else:
                            # Reinforcement Learning mode
                            # Add to SB3's replay buffer if we have a previous state
                            if env.prev_state is not None:
                                # Get action from policy
                                action_tensor, _ = rl_model.predict(processed_data, deterministic=False)
                                action = action_tensor
                                
                                # Add to custom buffer for mixed learning
                                custom_buffer.add(env.prev_state, action, reward, processed_data, done, is_bc=False)
                                
                                # Also add directly to SB3's buffer
                                try:
                                    # Create info dict for compatibility with newer SB3 versions
                                    info_dict = [{"terminal_observation": processed_data if done else None}]
                                    
                                    # Try with the new API that requires infos
                                    rl_model.replay_buffer.add(
                                        np.array([env.prev_state]),
                                        np.array([processed_data]),
                                        np.array([[action]]),
                                        np.array([reward]),
                                        np.array([float(done)]),
                                        info_dict
                                    )
                                except TypeError as e:
                                    if "missing 1 required positional argument: 'infos'" in str(e):
                                        print("Detected older SB3 version, adjusting API call")
                                        # Fallback for older SB3 versions that don't use infos
                                        rl_model.replay_buffer.add(
                                            np.array([env.prev_state]),
                                            np.array([processed_data]),
                                            np.array([[action]]),
                                            np.array([reward]),
                                            np.array([float(done)])
                                        )
                                    else:
                                        # Some other TypeError occurred, re-raise it
                                        raise
                            else:
                                # First frame or after reset, get action from policy
                                action_tensor, _ = rl_model.predict(processed_data, deterministic=False)
                            
                            # Train the model - check replay buffer size with compatibility for different SB3 versions
                            try:
                                # Try the standard length check first
                                buffer_size = len(rl_model.replay_buffer)
                            except (TypeError, AttributeError):
                                try:
                                    # Next try the size() method if available
                                    buffer_size = rl_model.replay_buffer.size()
                                except (TypeError, AttributeError):
                                    try:
                                        # Try accessing the pos attribute (index of the next element to insert)
                                        buffer_size = rl_model.replay_buffer.pos
                                    except (TypeError, AttributeError):
                                        # Fallback: assume buffer has data if we've added anything
                                        print("Warning: Could not determine replay buffer size, assuming it's ready for training")
                                        buffer_size = rl_model.learning_starts + 1
                            
                            # Only train if we have enough data
                            if buffer_size > rl_model.learning_starts:
                                rl_model._n_updates += 1
                                # Train for a single gradient step with error handling for different SB3 versions
                                try:
                                    # Standard training call
                                    rl_model.train(gradient_steps=1, batch_size=rl_model.batch_size)
                                except AttributeError as e:
                                    if "'DQN' object has no attribute '_logger'" in str(e):
                                        # Fix for logger attribute mismatch between SB3 versions
                                        print("Detected missing _logger attribute, applying fix...")
                                        
                                        try:
                                            # If logger exists, monkey patch _logger
                                            if hasattr(rl_model, 'logger'):
                                                print("Using existing logger")
                                                # Using __dict__ instead of direct assignment to bypass property restrictions
                                                rl_model.__dict__['_logger'] = rl_model.logger
                                            else:
                                                # Create a minimal dummy logger
                                                print("Creating dummy logger")
                                                from stable_baselines3.common.logger import Logger
                                                rl_model.__dict__['_logger'] = Logger(folder=None, output_formats=[])
                                            
                                            # Try training again with patched _logger
                                            rl_model.train(gradient_steps=1, batch_size=rl_model.batch_size)
                                        except Exception as logger_fix_error:
                                            print(f"WARNING: Logger fix failed: {logger_fix_error}")
                                            # Create dummy train method as absolute fallback
                                            print("Using fallback minimal training (update counter only)")
                                            # Just increment the update counter without actual training
                                            rl_model._n_updates += 1
                                    elif "property 'logger' of 'DQN' object has no setter" in str(e):
                                        print("Detected read-only logger property, applying alternative fix...")
                                        try:
                                            # Only create _logger without touching logger property
                                            if not hasattr(rl_model, '_logger'):
                                                print("Creating _logger without modifying logger property")
                                                from stable_baselines3.common.logger import Logger
                                                # Use __dict__ to bypass property restrictions
                                                rl_model.__dict__['_logger'] = Logger(folder=None, output_formats=[])
                                            
                                            # Try monkey patching the _update_learning_rate method to avoid logger usage
                                            original_update_lr = rl_model._update_learning_rate
                                            def patched_update_lr(optimizer):
                                                # Simple implementation that skips logger.record calls
                                                if hasattr(rl_model, 'lr_schedule') and callable(rl_model.lr_schedule):
                                                    new_lr = rl_model.lr_schedule(rl_model._current_progress_remaining)
                                                    for param_group in optimizer.param_groups:
                                                        param_group['lr'] = new_lr
                                                return
                                            
                                            # Replace the method temporarily
                                            rl_model._update_learning_rate = patched_update_lr
                                            
                                            # Try training again with the patched method
                                            rl_model.train(gradient_steps=1, batch_size=rl_model.batch_size)
                                            
                                            # Restore original method after training
                                            rl_model._update_learning_rate = original_update_lr
                                        except Exception as alt_fix_error:
                                            print(f"WARNING: Alternative logger fix failed: {alt_fix_error}")
                                            # Just increment the update counter without actual training
                                            print("Using fallback minimal training (update counter only)")
                                            rl_model._n_updates += 1
                                    else:
                                        # Some other AttributeError, re-raise
                                        raise
                                except Exception as train_error:
                                    print(f"Error during training: {train_error}")
                                    print("Using minimal fallback (incrementing counters only)")
                                    # Increment counter but don't perform actual training
                                    rl_model._n_updates += 1
                        
                        # Convert action to string for sending back to Lua
                        if isinstance(action, np.ndarray):
                            action = action.item()
                        action_str = ACTION_MAP[action]
                        
                        # Write the action back to Lua
                        py_to_lua.write(action_str + "\n")
                        py_to_lua.flush()
                        
                        # Calculate and display FPS occasionally
                        frame_count += 1
                        if frame_count % 100 == 0:
                            current_time = time.time()
                            fps = 100 / (current_time - last_frame_time)
                            last_frame_time = current_time
                            
                            # Log overall statistics
                            bc_loss = np.mean(bc_losses[-100:]) if bc_losses else 0
                            print(f"Stats: {frame_count} frames, {fps:.1f} FPS, Mode transitions: {mode_transitions}")
                            print(f"       BC episodes: {bc_episodes}, BC loss: {bc_loss:.6f}")
                            print(f"       RL episodes: {rl_episodes}, Buffer size: {len(custom_buffer)}")
                            print(f"       Model save locations: {MODEL_DIR}")
                    
                    except BlockingIOError:
                        # Expected in non-blocking mode
                        time.sleep(0.01)
                    
                    except Exception as e:
                        print(f"Error during frame processing: {e}")
                        import traceback
                        traceback.print_exc()
            
            finally:
                print("Pipe connection ended - Performing emergency save before exit")
                try:
                    # Emergency save before exit
                    rl_model.save(LATEST_MODEL_PATH)
                    torch.save(bc_model.state_dict(), BC_MODEL_PATH)
                    print(f"Emergency save completed to {MODEL_DIR}")
                    
                    # Write confirmation
                    with open(os.path.join(MODEL_DIR, "emergency_save_confirmed.txt"), "w") as f:
                        f.write(f"Emergency save completed at {datetime.datetime.now().isoformat()}")
                except Exception as e:
                    print(f"Failed emergency save: {e}")
                
                lua_to_py.close()
                py_to_lua.close()
                print("Pipes closed, reconnecting...")
        
        except KeyboardInterrupt:
            print("Interrupted by user")
            # Final save before exit
            try:
                save_signal_callback.force_save = True
                save_signal_callback._on_step()  # Force immediate save
                print("Final keyboard interrupt save complete")
            except Exception as e:
                print(f"Error during keyboard interrupt save: {e}")
                # Direct emergency save
                rl_model.save(LATEST_MODEL_PATH.replace(".zip", "_error.zip"))
                torch.save(bc_model.state_dict(), BC_MODEL_PATH.replace(".pt", "_error.pt"))
            break
        
        except Exception as e:
            print(f"Error: {e}")
            # Try emergency save on unexpected error
            try:
                rl_model.save(LATEST_MODEL_PATH.replace(".zip", "_error.zip"))
                torch.save(bc_model.state_dict(), BC_MODEL_PATH.replace(".pt", "_error.pt"))
                print("Emergency save on error completed")
            except Exception as save_error:
                print(f"Emergency save failed: {save_error}")
            time.sleep(5)
    
    print("Python AI model shutting down")

if __name__ == "__main__":
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    main()
        

