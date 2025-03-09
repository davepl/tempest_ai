#!/usr/bin/env python3
"""
Tempest AI Model
Author: Dave Plummer (davepl) and various AI assists
Date: 2023-03-06

This script implements a simple AI model for the Tempest arcade game.
It receives game state data from a Lua script running in MAME via a named pipe,
processes the data, and returns actions to control the game.

The script uses a named pipe for communication with the Lua script.
"""

import os
import sys
import time
import struct
import random
import stat

# Define the paths to the named pipes
LUA_TO_PY_PIPE = "/tmp/lua_to_py"
PY_TO_LUA_PIPE = "/tmp/py_to_lua"

def process_frame_data(data):
    """
    Process the binary frame data received from Lua.
    
    Args:
        data (bytes): Binary data containing game state information
        
    Returns:
        list: Processed game state as a list of normalized values
    """
    # Calculate how many 16-bit integers we have
    num_ints = len(data) // 2
    
    # Unpack the binary data into 16-bit signed integers (big-endian)
    unpacked_data = []
    for i in range(num_ints):
        value = struct.unpack(">h", data[i*2:i*2+2])[0]
        unpacked_data.append(value)
    
    # Normalize the data to 0-1 range for the neural network
    # This is a simplified example - you would need to adjust based on actual data ranges
    normalized_data = [float(x) / 32767.0 if x > 0 else float(x) / 32768.0 for x in unpacked_data]
    
    return normalized_data

def ai_model(game_state):
    """
    Simple AI model that determines the action based on the game state.
    
    Args:
        game_state (list): Processed game state data
        
    Returns:
        str: Action to take (fire, zap, left, right, none)
    """
    # For now, just return a random action
    actions = ["fire", "zap", "left", "right", "none"]
    
    # Add weights to make some actions more common than others
    # For example, fire more often, use zap rarely
    weights = [0.4, 0.05, 0.25, 0.25, 0.05]  # 40% fire, 5% zap, 25% left, 25% right, 5% none
    
    # Choose a random action based on weights
    action = random.choices(actions, weights=weights, k=1)[0]
    
    # Log the action (for debugging)
    print(f"AI choosing action: {action}")
    
    return action

def main():
    """
    Main function that handles the communication with Lua and processes game frames.
    """
    print("Python AI model starting...")
    
    # Remove existing pipes to ensure clean state
    for pipe_path in [LUA_TO_PY_PIPE, PY_TO_LUA_PIPE]:
        try:
            os.unlink(pipe_path)
            print(f"Removed existing pipe: {pipe_path}")
        except FileNotFoundError:
            print(f"Pipe {pipe_path} did not exist, no need to remove")
    
    # Create the pipes
    for pipe_path in [LUA_TO_PY_PIPE, PY_TO_LUA_PIPE]:
        try:
            os.mkfifo(pipe_path)
            # Set permissions to ensure they're readable/writable
            os.chmod(pipe_path, 0o666)
            print(f"Created pipe: {pipe_path}")
        except OSError as e:
            print(f"Error creating pipe {pipe_path}: {e}")
            sys.exit(1)
    
    # Verify pipes exist
    for pipe_path in [LUA_TO_PY_PIPE, PY_TO_LUA_PIPE]:
        if os.path.exists(pipe_path):
            mode = os.stat(pipe_path).st_mode
            if stat.S_ISFIFO(mode):
                print(f"Verified {pipe_path} exists and is a named pipe")
            else:
                print(f"Warning: {pipe_path} exists but is not a named pipe!")
        else:
            print(f"Error: {pipe_path} does not exist after creation!")
            sys.exit(1)
    
    print("Pipes created successfully. Waiting for Lua connection...")
    
    # Connection retry loop
    while True:
        try:
            # IMPORTANT: Open pipes in the correct order to avoid deadlock
            # First, open the reading pipe (lua_to_py) in non-blocking mode
            # This is critical because opening a pipe for reading normally blocks until someone opens it for writing
            print("Opening input pipe (lua_to_py) in non-blocking mode...")
            fd = os.open(LUA_TO_PY_PIPE, os.O_RDONLY | os.O_NONBLOCK)
            lua_to_py = os.fdopen(fd, "rb")
            print("Input pipe opened successfully in non-blocking mode")
            
            # Then open the writing pipe (py_to_lua)
            print("Opening output pipe (py_to_lua)...")
            py_to_lua = open(PY_TO_LUA_PIPE, "w")
            print("Output pipe opened successfully")
            
            print("Connected to Lua pipes! Ready to process game frames.")
            
            try:
                frame_count = 0
                while True:
                    try:
                        # Try to read from the pipe (may return empty if no data available)
                        data = lua_to_py.read()
                        
                        if not data:
                            # In non-blocking mode, this could mean no data yet
                            # Small delay and continue
                            time.sleep(0.01)
                            continue
                        
                        # Process the frame data
                        processed_data = process_frame_data(data)
                        
                        # Get the action from the AI model
                        action = ai_model(processed_data)
                        
                        # Write the action back to Lua
                        py_to_lua.write(action + "\n")
                        py_to_lua.flush()  # Make sure the data is sent immediately
                        
                        # Log every 100 frames to avoid excessive output
                        frame_count += 1
                        if frame_count % 100 == 0:
                            print(f"Processed {frame_count} frames, last action: {action}")
                    
                    except BlockingIOError:
                        # This is expected in non-blocking mode when no data is available
                        time.sleep(0.01)
                        continue
            
            finally:
                # Clean up resources
                lua_to_py.close()
                py_to_lua.close()
                print("Pipes closed, attempting to reconnect...")
        
        except KeyboardInterrupt:
            print("Interrupted by user")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Waiting 5 seconds before retry...")
            time.sleep(5)
    
    print("Python AI model shutting down")

if __name__ == "__main__":
    # Set seed for reproducibility
    random.seed(42)
    main()
        

