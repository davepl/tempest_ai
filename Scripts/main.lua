--[[
    Tempest AI Lua Script for MAME
    Author: Dave Plummer (davepl) and various AI assists
    Date: [2025-03-06]

    Overview:
    This script is part of an AI project to play the classic arcade game Tempest using reinforcement learning.
    It runs within MAME's Lua environment, collecting game state data each frame and applying actions to control the game.
    The script uses a class-based structure (game, level, player, enemy, controls) for modularity and extensibility.

    Key Features:
    - Captures game state (credits, level, player position, alive status, enemy count) from MAME memory.
    - Implements random actions (fire, zap, move left, move right, idle) to simulate gameplay.
    - Outputs a concise frame-by-frame summary of key stats for debugging and analysis.

    Usage:
    - Launch with MAME: `mame tempest -autoboot_script main.lua`
    - Customize memory addresses and input field names as needed for your Tempest version.
    - Everything I've done is based on the original Tempest ROM set, not later revisions.

    Notes:
    - Memory addresses are placeholders; update them based on the game's memory map.
    - Input field names (e.g., "Fire", "Superzapper") must align with the game's input config in MAME.
    - Designed for educational use; extendable for full AI integration (e.g., RL model).

    Dependencies:
    - MAME with Lua scripting support enabled.
    - Tempest ROM set loaded in MAME.
--]]

-- Add the Scripts directory to Lua's package path
package.path = package.path .. ";/Users/dave/source/repos/tempest/Scripts/?.lua"
-- Now require the module by name only (without path or extension)

-- Check command line arguments for -logonly mode
local LOG_ONLY_MODE = false

local function clear_screen()
    io.write("\027[2J\027[H")
end

-- Function to move the cursor to the home position using ANSI escape codes
local function move_cursor_home()
    io.write("\027[H")
end

-- Global pipe variables
local pipe_out = nil
local pipe_in = nil
-- Log file for storing frame data
local log_file = nil  -- Will hold the file handle, not the path
local log_file_path = "/Users/dave/mame/tempest.log"  -- Store the path separately
local log_file_opened = false  -- Track if we've already opened the file

-- Add this near the top of the file with other global variables
local frame_count = 0  -- Initialize frame counter

-- Function to open pipes 
local function open_pipes()
    if (LOG_ONLY_MODE) then

        -- Only open the log file once to prevent truncation!
        if log_file_opened then
            return true
        end
        
        -- Ensure log file is opened properly in binary mode
        local log_success, log_err = pcall(function()
            -- Close the log file if it's already open
            if log_file then
                log_file:close()
                log_file = nil
            end
            
            -- Open log file in binary mode, with explicit path - ONLY ONCE
            log_file = io.open(log_file_path, "wb")
            
            if not log_file then
                print("Failed to open log file: file handle is nil")
                return false
            end
            
            print("Successfully opened log file for writing")
            log_file_opened = true  -- Mark as opened so we don't reopen
        end)
        
        if not log_success then
            print("Error opening log file: " .. tostring(log_err))
            return false
        end
        
        return true
    end

    -- Try to open the output pipe
    local open_success, err = pcall(function()
        pipe_out = io.open("/tmp/lua_to_py", "wb")
    end)
    
    if not open_success or not pipe_out then
        print("Failed to open output pipe: " .. tostring(err))
        return false
    end
    
    -- Try to open the input pipe
    open_success, err = pcall(function()
        pipe_in = io.open("/tmp/py_to_lua", "rb")
    end)
    
    if not open_success or not pipe_in then
        print("Failed to open input pipe: " .. (err or "Unknown error"))
        if pipe_out then pipe_out:close(); pipe_out = nil end
        return false
    end
    
    -- Open log file for appending in binary mode
    if not log_file then
        local log_success, log_err = pcall(function()
            log_file = io.open("tempest.log", "ab")
        end)
        
        if not log_success or not log_file then
            print("Warning: Failed to open log file: " .. tostring(log_err))
        else
            print("Opened log file for frame data recording")
        end
    end
    
    return true
end

-- Create a variable to track pipe retry attempts
local pipe_retry_count = 0

-- Declare global variables for reward calculation
local previous_score = 0
local previous_level = 0
local avg_score_per_frame = 0
local AGGRESSION_DECAY = 0.99
local previous_alive_state = 1  -- Track previous alive state, initialize as alive

-- Declare a global variable to store the last reward state
local LastRewardState = 0

-- Add this at the top of the file, near other global variables
local shutdown_requested = false

-- Add these global variables near the top of the script with other globals
local model_fire = 0
local model_zap = 0
local model_spinner = 0
local last_display_update = 0  -- Timestamp of last display update
local DISPLAY_UPDATE_INTERVAL = 0.1  -- Update display every 0.1 seconds (10 times per second)

-- Function to calculate reward for the current frame
local function calculate_reward(game_state, level_state, player_state)
    local reward = 0
    
    -- 1. Survival reward: 1 point per frame for staying alive
    -- 2. Death penalty: -10 points only when transitioning from alive to dead
    if player_state.alive == 1 then
        -- Player is alive
        reward = reward + 1
    else
        -- Player is dead, but only apply penalty on transition
        if previous_alive_state == 1 then
            -- Just died - apply death penalty once
            reward = reward - 10
            print("Death penalty applied: -10")
        end
        -- No penalty on subsequent dead frames
    end
    
    -- 2. Score reward: Add the score delta from the last frame
    local score_delta = player_state.score - previous_score
    -- Debug output if score changes
    if score_delta > 0 then
        reward = reward + score_delta
    end
    
    -- 3. Level completion reward: 1000 * new level number when level increases
    if level_state.level_number ~= previous_level then
        reward = reward + (1000 * previous_level)
    end
    
    -- 4. Aggression reward: Use weighted average of score per frame
    -- Update the weighted average with the current score delta
    -- Formula: new_avg = old_avg * decay + current_value * (1 - decay)

    avg_score_per_frame = avg_score_per_frame * AGGRESSION_DECAY + score_delta * (1 - AGGRESSION_DECAY)
    
    -- Add aggression bonus (scaled to be meaningful but not dominant)
    -- You could multiply by 10 to make it more significant
    -- reward = reward + (avg_score_per_frame)
    
    -- 6. Spinner stasis reward: 128 - abs(spinner_delta) / 50

    local spinner_abs = math.min(127, math.abs(player_state.SpinnerDelta))
    reward = reward + ((128 - spinner_abs) / 100)

    -- Update previous values for next frame
    previous_score = player_state.score
    previous_level = level_state.level_number
    previous_alive_state = player_state.alive  -- Update previous alive state

    -- Update the LastRewardState with the current reward
    LastRewardState = reward
    
    return reward
end

-- Function to send parameters and get action each frame
local function process_frame(params, player_state, controls, reward)
    -- In log-only mode, we only write to the log file and don't communicate with Python

    -- Get the size of the header in bytes 
    local header_size = 4 + 4 + 4 + 1 + 1 + 1
    -- Get the size of the payload in bytes
    local payload_size = #params

    if LOG_ONLY_MODE then
        -- Log the frame data to file
        if log_file then
            local success, err = pcall(function()
                -- Convert reward to integer with 3 decimal precision
                local int_reward = math.floor((reward or 0) * 1000)
                
                -- Print debugging output every 100 frames to avoid spam
                if frame_count % 100 == 0 then
                    print(string.format("Writing frame %d - Header size: %d, Payload size: %d, Reward: %d/1000", 
                          frame_count, header_size, payload_size, int_reward))
                end
                
                assert(controls.spinner_delta > -127 and controls.spinner_delta < 128)
                
                -- Preamble - header and payload sizes
                log_file:write(string.pack(">I4", header_size))                 -- Write 32-bit header size
                log_file:write(string.pack(">I4", payload_size))                -- Write 32-bit payload size
                -- Header
                log_file:write(string.pack(">i4", int_reward))                  -- Write 32-bit reward (multiplied by 1000)
                log_file:write(string.pack(">I1", controls.zap_commanded))      -- Write 8-bit zap
                log_file:write(string.pack(">I1", controls.fire_commanded))     -- Write 8-bit fire
                log_file:write(string.pack(">i1", controls.spinner_delta))  -- Write 8-bit SIGNED spinner (lowercase i)
                -- Write the raw data   
                log_file:write(params)

                log_file:flush()  -- Ensure data is written to disk
            end)
            
            if not success then
                print("ERROR writing to log file: " .. tostring(err))
            end
        else
            -- Log file not open, attempting to open it
            print("Log file not open, attempting to reopen")
            open_pipes()
        end
        
        -- Increment frame count after processing
        frame_count = frame_count + 1
        
        return controls.zap_commanded, controls.fire_commanded, controls.spinner_delta
    end
    
    -- Check if pipes are open, try to reopen if not
    if not pipe_out or not pipe_in then
        if not open_pipes() then
            return 0, 0, 0  -- Return zeros for fire, zap, spinner_delta on pipe error
        end
    end
  
    -- Send data to Python

    local int_reward = math.floor((reward or 0) * 1000)

    pipe_out:write(string.pack(">I4", header_size))                         -- Write 32-bit header size    
    pipe_out:write(string.pack(">I4", payload_size))                        -- Write 32-bit payload size
    pipe_out:write(string.pack(">I4", int_reward))                              -- Write 32-bit reward 
    pipe_out:write(string.pack(">I1", controls.zap_commanded))              -- Write 8-bit zap
    pipe_out:write(string.pack(">I1", controls.fire_commanded))             -- Write 8-bit fire
    pipe_out:write(string.pack(">i1", controls.spinner_delta))          -- Write 8-bit SIGNED spinner (lowercase i)

    -- Try to write to pipe, handle errors
    local success, err = pcall(function()
        pipe_out:write(params)
        pipe_out:flush()
    end)
    
    if not success then
        print("Error writing to pipe:", err)
        -- Close and attempt to reopen pipes
        if pipe_out then pipe_out:close(); pipe_out = nil end
        if pipe_in then pipe_in:close(); pipe_in = nil end
        open_pipes()
        return 0, 0, 0  -- Return zeros for fire, zap, spinner_delta on write error
    end
    
    -- Try to read from pipe, handle errors
    local fire, zap, spinner = 0, 0, 0  -- Default values
    success, err = pcall(function()
        -- Use a timeout to avoid hanging indefinitely
        local start_time = os.time()
        local timeout = 2  -- 2 second timeout
        
        while os.time() - start_time < timeout do
            -- Read exactly 3 bytes for the three i8 values
            local action_bytes = pipe_in:read(3)
            
            if action_bytes and #action_bytes == 3 then
                -- Unpack the three signed 8-bit integers
                fire, zap, spinner = string.unpack("bbb", action_bytes)
                
                -- Store the values globally for display
                model_fire = fire
                model_zap = zap  
                model_spinner = spinner
                
                break  -- Exit the loop once we've successfully read and unpacked the data
            else
                print("Sleeping...")
                -- Sleep briefly to avoid busy-waiting
                os.execute("sleep 0.01")
            end
        end
    end)
    
    if not success then
        print("Error reading from pipe:", err)
        -- Close and attempt to reopen pipes
        if pipe_out then pipe_out:close(); pipe_out = nil end
        if pipe_in then pipe_in:close(); pipe_in = nil end
        open_pipes()
        return 0, 0, 0  -- Return zeros for fire, zap, spinner_delta on read error
    end
    
    -- Return the three components directly
    return fire, zap, spinner
end

-- Call this during initialization
local function start_python_script()
    -- In log-only mode, we don't need to start the Python script
    if LOG_ONLY_MODE then
        print("Log-only mode: Skipping Python script startup")
        
        -- Still open the log file
        local log_success, log_err = pcall(function()
            log_file = io.open("tempest.log", "ab")
        end)
        
        if not log_success or not log_file then
            print("Warning: Failed to open log file: " .. tostring(log_err))
        else
            print("Opened log file for frame data recording")
        end
        
        return true  -- Return success
    end
    
    -- print("Starting Python script")
    
    -- Comment these out if you're trying to debug the Python side...

    -- Kill any existing Python script instances
    -- os.execute("pkill -f 'python.*aimodel.py' 2>/dev/null")
    
    -- Remove existing pipes to ensure clean state
    -- os.execute("rm -f /tmp/lua_to_py /tmp/py_to_lua")
    
    -- os.execute("rm -f /Users/dave/source/repos/tempest/Scripts/models/*")

    -- Launch Python script in the background with proper error handling
    local cmd = "python /Users/dave/source/repos/tempest/Scripts/aimodel.py >/tmp/python_output.log 2>&1 &"
    local result = os.execute(cmd)
    
    if result ~= 0 then
        print("Warning: Failed to start Python script")
        return false
    end
    
    -- Give Python script a moment to start up and create pipes
    -- print("Waiting for Python script to initialize and create pipes...")
    os.execute("sleep 3")
    
    -- Check if Python script is running
    local python_running = os.execute("pgrep -f 'python.*aimodel.py' >/dev/null") == 0
    if not python_running then
        print("Warning: Python script failed to start or terminated early")
        print("Check /tmp/python_output.log for errors")
        return false
    end
    
    -- print("Python script started successfully")
    
    -- Try to open pipes immediately
    local pipes_opened = open_pipes()
    if not pipes_opened then
        print("Initial pipe opening failed, will retry during frame callback")
    end
    
    return true
end

start_python_script()
clear_screen()

-- Add after the initial requires, before the GameState class

-- Seed the random number generator once at script start
math.randomseed(os.time())

-- Access the main CPU and memory space
local mainCpu = manager.machine.devices[":maincpu"]
if not mainCpu then
    print("Error: Main CPU not found")
    return
end

local mem = mainCpu.spaces["program"]
if not mem then
    print("Error: Program memory space not found")
    return
end

-- Helper function for calculating relative positions
local function calculate_relative_position(player_pos, target_pos, is_open)
    local direct_diff = target_pos - player_pos
    
    if is_open then
        -- For open levels, just return the direct difference
        return direct_diff
    else
        -- For closed levels, find the shortest path around the circle
        -- Total segments is 16
        local clockwise_diff = direct_diff
        local counter_clockwise_diff = direct_diff
        
        if direct_diff > 8 then
            clockwise_diff = direct_diff - 16
        elseif direct_diff < -8 then
            counter_clockwise_diff = direct_diff + 16
        end
        
        -- Return the smallest absolute difference
        if math.abs(clockwise_diff) < math.abs(counter_clockwise_diff) then
            return clockwise_diff
        else
            return counter_clockwise_diff
        end
    end
end

-- **GameState Class**
GameState = {}
GameState.__index = GameState

function GameState:new()
    local self = setmetatable({}, GameState)
    self.credits = 0
    self.p1_level = 0
    self.p1_lives = 0
    self.gamestate = 0    -- Game state from address 0
    self.game_mode = 0    -- Game mode from address 5
    self.countdown_timer = 0  -- Countdown timer from address 4
    self.frame_counter = 0  -- Frame counter for tracking progress
    self.last_save_time = os.time()  -- Track when we last sent save signal
    self.save_interval = 300  -- Send save signal every 5 minutes (300 seconds)
    return self
end

function GameState:update(mem)
    self.gamestate = mem:read_u8(0x0000)  -- Game state at address 0
    self.game_mode = mem:read_u8(0x0005)  -- Game mode at address 5
    self.countdown_timer = mem:read_u8(0x0004)  -- Countdown timer from address 4
    self.credits = mem:read_u8(0x0006)    -- Credits
    self.p1_level = mem:read_u8(0x0046)   -- Player 1 level
    self.p1_lives = mem:read_u8(0x0048)   -- Player 1 lives
    self.frame_counter = self.frame_counter + 1  -- Increment frame counter
end

-- **LevelState Class**
LevelState = {}
LevelState.__index = LevelState

function LevelState:new()
    local self = setmetatable({}, LevelState)
    self.level_number = 0
    self.spike_heights = {}  -- Array of 16 spike heights
    self.level_type = 0     -- 00 = closed, FF = open
    self.level_angles = {}  -- Array of 16 tube angles
    self.level_shape = 0    -- Level shape (level_number % 16)
    return self
end

function LevelState:update(mem)
    self.level_number = mem:read_u8(0x009F)  -- Example address for level number
    self.level_type = mem:read_u8(0x0111)    -- Level type (00=closed, FF=open)
    self.level_shape = self.level_number % 16  -- Calculate level shape
    local player_pos = mem:read_u8(0x0200)   -- Player position
    local is_open = self.level_type == 0xFF
    
    -- Read spike heights for all 16 segments and store them relative to player
    self.spike_heights = {}
    for i = 0, 15 do  -- Use 0-based indexing to match game's segment numbering
        local height = mem:read_u8(0x03AC + i)
        -- Adjust player_pos to 0-based indexing to match our loop
        local rel_pos = calculate_relative_position(player_pos & 0x0F, i, is_open)
        self.spike_heights[rel_pos] = height
    end
    
    -- Read tube angles for all 16 segments
    self.level_angles = {}
    for i = 0, 15 do
        self.level_angles[i] = mem:read_u8(0x03EE + i)
    end
end

-- **PlayerState Class**
PlayerState = {}
PlayerState.__index = PlayerState

function PlayerState:new()
    local self = setmetatable({}, PlayerState)
    self.position = 0
    self.alive = 0
    self.score = 0
    self.superzapper_uses = 0
    self.superzapper_active = 0
    self.player_depth = 0     -- New field for player depth along the tube
    self.player_state = 0     -- New field for player state from $201
    self.shot_segments = {0, 0, 0, 0, 0, 0, 0, 0}  -- 8 shot segments
    self.shot_positions = {0, 0, 0, 0, 0, 0, 0, 0}  -- 8 shot positions (depth)
    self.shot_count = 0
    self.debounce = 0
    self.fire_detected = 0
    self.zap_detected = 0
    self.SpinnerAccum = 0
    self.SpinnerDelta = 0
    self.prevSpinnerAccum = 0
    self.inferredSpinnerDelta = 0
    return self
end

function PlayerState:update(mem)
    self.position = mem:read_u8(0x0200)          -- Player position
    self.player_state = mem:read_u8(0x0201)      -- Player state value at $201
    self.player_depth = mem:read_u8(0x0202)      -- Player depth along the tube
    
    -- Check if player state doesn't have the high bit set
    -- When high bit is set (0x80), player is dead
    self.alive = ((self.player_state & 0x80) == 0) and 1 or 0

    local function bcd_to_decimal(bcd)
        return ((bcd >> 4) * 10) + (bcd & 0x0F)
    end

    -- Extract the score from the BCD value in memory
    
    local score_low = bcd_to_decimal(mem:read_u8(0x0040))
    local score_mid = bcd_to_decimal(mem:read_u8(0x0041))
    local score_high = bcd_to_decimal(mem:read_u8(0x0042))
    self.score = score_high * 10000 + score_mid * 100 + score_low

    self.superzapper_uses = mem:read_u8(0x03AA)        -- Superzapper availability
    self.superzapper_active = mem:read_u8(0x0125)           -- Superzapper active status
    self.shot_count = mem:read_u8(0x0135)                   -- Number of active player shots

    -- Read all 8 shot positions and segments
    local is_open = mem:read_u8(0x0111) == 0xFF
    for i = 1, 8 do
        -- Read depth (position along the tube)
        self.shot_positions[i] = mem:read_u8(0x02D3 + i - 1)  -- PlayerShotPositions
        
        -- Read segment and make it relative to player position only if shot is active
        local abs_segment = mem:read_u8(0x02AD + i - 1)       -- PlayerShotSegments
        if self.shot_positions[i] == 0 or abs_segment == 0 then
            self.shot_segments[i] = 0  -- Shot not active
        else
            abs_segment = abs_segment & 0x0F  -- Mask to get valid segment
            self.shot_segments[i] = calculate_relative_position(self.position, abs_segment, is_open)
        end
    end

    -- Update detected input states
    self.debounce = mem:read_u8(0x004D)
    self.fire_detected = (self.debounce & 0x10) ~= 0 and 1 or 0
    self.zap_detected = (self.debounce & 0x08) ~= 0 and 1 or 0
    
    -- Store previous spinner position
    local currentSpinnerAccum = mem:read_u8(0x0051)
    self.SpinnerDelta = mem:read_u8(0x0050)
    
    -- Calculate inferred delta by comparing with previous position
    local rawDelta = currentSpinnerAccum - self.prevSpinnerAccum
    
    -- Handle 8-bit wrap-around
    if rawDelta > 127 then
        rawDelta = rawDelta - 256
    elseif rawDelta < -128 then
        rawDelta = rawDelta + 256
    end
    
    self.inferredSpinnerDelta = rawDelta
    self.SpinnerAccum = currentSpinnerAccum
    self.prevSpinnerAccum = currentSpinnerAccum
end

-- **EnemiesState Class**
EnemiesState = {}
EnemiesState.__index = EnemiesState

function EnemiesState:new()
    local self = setmetatable({}, EnemiesState)
    -- Active enemies (currently on screen)
    self.active_flippers = 0
    self.active_pulsars = 0
    self.active_tankers = 0
    self.active_spikers = 0
    self.active_fuseballs = 0
    self.pulse_beat = 0    -- Add pulse beat counter
    self.pulsing = 0      -- Add pulsing state
    -- Available spawn slots (how many more can be created)
    self.spawn_slots_flippers = 0
    self.spawn_slots_pulsars = 0
    self.spawn_slots_tankers = 0
    self.spawn_slots_spikers = 0
    self.spawn_slots_fuseballs = 0
    -- Enemy info arrays
    self.enemy_type_info = {0, 0, 0, 0, 0, 0, 0}    -- 7 enemy type slots
    self.active_enemy_info = {0, 0, 0, 0, 0, 0, 0}  -- 7 active enemy slots
    self.enemy_segments = {0, 0, 0, 0, 0, 0, 0}     -- 7 enemy segment numbers
    self.enemy_depths = {0, 0, 0, 0, 0, 0, 0}       -- 7 enemy depth positions
    self.shot_positions = {}  -- Will store nil for inactive shots
    self.pending_vid = {}  -- 64-byte table
    self.pending_seg = {}  -- 64-byte table
    -- Enemy shot segments parameters extracted from memory address 02B5
    self.enemy_shot_segments = {
        { address = 0x02B5, value = 0 },
        { address = 0x02B6, value = 0 },
        { address = 0x02B7, value = 0 },
        { address = 0x02B8, value = 0 }
    }
    return self
end

function EnemiesState:update(mem)
    -- Read active enemies (currently on screen)
    self.active_flippers = mem:read_u8(0x0142)   -- n_flippers - current active count
    self.active_pulsars = mem:read_u8(0x0143)    -- n_pulsars
    self.active_tankers = mem:read_u8(0x0144)    -- n_tankers
    self.active_spikers = mem:read_u8(0x0145)    -- n_spikers
    self.active_fuseballs = mem:read_u8(0x0146)  -- n_fuseballs
    self.pulse_beat = mem:read_u8(0x0147)        -- pulse_beat counter
    self.pulsing = mem:read_u8(0x0148)          -- pulsing state

    -- Update enemy shot segments from memory
    for i = 1, 4 do
        self.enemy_shot_segments[i].value = mem:read_u8(self.enemy_shot_segments[i].address)
    end

    -- Get player position and level type for relative calculations
    local player_pos = mem:read_u8(0x0200)
    local is_open = mem:read_u8(0x0111) == 0xFF

    -- Read available spawn slots (how many more can be created)
    self.spawn_slots_flippers = mem:read_u8(0x013D)   -- avl_flippers - spawn slots available
    self.spawn_slots_pulsars = mem:read_u8(0x013E)    -- avl_pulsars
    self.spawn_slots_tankers = mem:read_u8(0x013F)    -- avl_tankers
    self.spawn_slots_spikers = mem:read_u8(0x0140)    -- avl_spikers
    self.spawn_slots_fuseballs = mem:read_u8(0x0141)  -- avl_fuseballs

    -- Read enemy type and state info
    for i = 1, 7 do
        self.enemy_type_info[i] = mem:read_u8(0x0283 + i - 1)    -- Enemy type info at $0283
        self.active_enemy_info[i] = mem:read_u8(0x028A + i - 1)  -- Active enemy info at $028A
        
        -- Get absolute segment number and convert to relative
        local abs_segment = mem:read_u8(0x0291 + i - 1) & 0x0F
        self.enemy_segments[i] = calculate_relative_position(player_pos, abs_segment, is_open)
        
        -- Get main position value and LSB for more precision
        local pos = mem:read_u8(0x02DF + i - 1)       -- enemy_along at $02DF - main position
        local lsb = mem:read_u8(0x029F + i - 1)       -- enemy_along_lsb at $029F - fractional part
        -- Store both values for display
        self.enemy_depths[i] = {pos = pos, frac = lsb}
    end

    -- Read all 4 enemy shot positions and convert to relative positions
    for i = 1, 4 do
        local raw_pos = mem:read_u8(0x02DB + i - 1)
        if raw_pos == 0 then
            self.shot_positions[i] = nil  -- Mark inactive shots as nil
        else
            local abs_pos = raw_pos & 0x0F
            self.shot_positions[i] = calculate_relative_position(player_pos, abs_pos, is_open)
        end
    end

    -- Read pending_seg (64 bytes starting at 0x0203)
    for i = 1, 64 do
        self.pending_seg[i] = mem:read_u8(0x0203 + i - 1)
    end

    -- Read pending_vid (64 bytes starting at 0x0243)
    for i = 1, 64 do
        self.pending_vid[i] = mem:read_u8(0x0243 + i - 1)
    end
end

-- Helper function to decode enemy type info
function EnemiesState:decode_enemy_type(type_byte)
    local enemy_type = type_byte & 0x07
    local between_segments = (type_byte & 0x80) ~= 0
    local segment_increasing = (type_byte & 0x40) ~= 0
    return string.format("%d%s%s", 
        enemy_type,
        between_segments and "B" or "-",
        segment_increasing and "+" or "-"
    )
end

-- Helper function to decode enemy state info
function EnemiesState:decode_enemy_state(state_byte)
    local split_behavior = state_byte & 0x03
    local can_shoot = (state_byte & 0x40) ~= 0
    local moving_away = (state_byte & 0x80) ~= 0
    return string.format("%s%s%s",
        moving_away and "A" or "T",
        can_shoot and "S" or "-",
        split_behavior
    )
end

function EnemiesState:get_total_active()
    return self.active_flippers + self.active_pulsars + self.active_tankers + 
           self.active_spikers + self.active_fuseballs
end

-- **Controls Class**
Controls = {}
Controls.__index = Controls

function Controls:new()
    local self = setmetatable({}, Controls)

    -- Debug: List all available ports and fields
    for port_tag, port in pairs(manager.machine.ioport.ports) do
        print("Port: " .. port_tag)
        for field_name, field in pairs(port.fields) do
            print("  Field: " .. field_name)
        end
    end

    -- Get button ports
    self.button_port = manager.machine.ioport.ports[":BUTTONSP1"]
    print("Button port found: " .. (self.button_port and "Yes" or "No"))
    if self.button_port then
        print("Fire field found: " .. (self.button_port.fields["P1 Button 1"] and "Yes" or "No"))
        if self.button_port.fields["P1 Button 1"] then
            print("Fire field type: " .. type(self.button_port.fields["P1 Button 1"]))
        end
    end
    
    -- Get spinner/dial port
    self.spinner_port = manager.machine.ioport.ports[":KNOBP1"]
    print("Spinner port found: " .. (self.spinner_port and "Yes" or "No"))
    if self.spinner_port then
        print("Dial field found: " .. (self.spinner_port.fields["Dial"] and "Yes" or "No"))
        if self.spinner_port.fields["Dial"] then
            print("Dial field type: " .. type(self.spinner_port.fields["Dial"]))
        end
    end
    
    -- Set up button fields
    self.fire_field = self.button_port and self.button_port.fields["P1 Button 1"] or nil
    self.zap_field = self.button_port and self.button_port.fields["P1 Button 2"] or nil
    
    -- Set up spinner field
    self.spinner_field = self.spinner_port and self.spinner_port.fields["Dial"] or nil
    
    -- Track commanded states
    self.fire_commanded = 0
    self.zap_commanded = 0
    self.spinner_delta = 0
    
    return self
end

function Controls:apply_action(fire, zap, spinner, game_state, player_state)
    -- Fix the attract mode check - bit 0x80 being CLEAR indicates attract mode
    local is_attract_mode = (game_state.game_mode & 0x80) == 0
    
    if is_attract_mode then
        -- In attract mode:
        -- 1. Fire is always true regardless of physical control existence
        self.fire_commanded = 1  -- Set this first, unconditionally
        
        -- Now set the physical control if it exists
        if self.fire_field then 
            self.fire_field:set_value(1)
        end
        
        -- 2. Zap is always false (already set to 0 above)
        
        -- 3. Use inferred spinner delta
        self.spinner_delta = player_state.inferredSpinnerDelta
        
    else
        -- In actual gameplay:
        -- Always use the inferred spinner delta for display
        self.spinner_delta = player_state.inferredSpinnerDelta
        
        -- But apply the model's fire, zap, and spinner values to the physical controls
        self.fire_commanded = fire
        self.zap_commanded = zap

        if self.fire_field then self.fire_field:set_value(fire) end
        if self.zap_field then self.zap_field:set_value(zap) end
        
        -- Apply the model's spinner value to the game

        mem:write_u8(0x0050, spinner)
    end
end

-- Instantiate state objects
local game_state = GameState:new()
local level_state = LevelState:new()
local player_state = PlayerState:new()
local enemies_state = EnemiesState:new()
local controls = Controls:new()

-- Remove all ncurses-related code and replace with this display function
local function format_section(title, metrics)
    local width = 40
    local separator = string.rep("-", width - 4)
    local result = string.format("--[ %s ]%s\n", title, separator)
    
    -- Find the longest key for alignment
    local max_key_length = 0
    for key, _ in pairs(metrics) do
        max_key_length = math.max(max_key_length, string.len(key))
    end
    
    -- Format each metric
    for key, value in pairs(metrics) do
        result = result .. string.format("  %-" .. max_key_length .. "s : %s\n", key, tostring(value))
    end
    
    return result .. "\n"
end

-- Function to move the cursor to a specific row
local function move_cursor_to_row(row)
    io.write(string.format("\027[%d;0H", row))
end

-- Global variables for tracking bytes sent and FPS
local total_bytes_sent = 0
local last_fps_time = os.time()
local current_fps = 0

-- Function to flatten and serialize the game state data to signed 16-bit integers
local function flatten_game_state_to_binary(game_state, level_state, player_state, enemies_state)
    -- Create a consistent data structure with fixed sizes
    local data = {}
    
    -- Game state (5 values, frame counter is now in OOB data)
    table.insert(data, game_state.gamestate)
    table.insert(data, game_state.game_mode)
    table.insert(data, game_state.countdown_timer)
    table.insert(data, game_state.credits)
    table.insert(data, game_state.p1_lives)
    table.insert(data, game_state.p1_level)
    
    -- Player state (5 values + arrays, score is now in OOB data)
    table.insert(data, player_state.position)
    table.insert(data, player_state.alive)
    table.insert(data, player_state.player_state)  -- Add player state to serialized data 
    table.insert(data, player_state.player_depth)  -- Add player depth to serialized data
    table.insert(data, player_state.superzapper_uses)
    table.insert(data, player_state.superzapper_active)
    table.insert(data, player_state.shot_count)
    
    -- Player shot positions (fixed size: 8)
    for i = 1, 8 do
        table.insert(data, player_state.shot_positions[i] or 0)
    end
    
    -- Player shot segments (fixed size: 8)
    for i = 1, 8 do
        table.insert(data, player_state.shot_segments[i] or 0)
    end
    
    -- Level state (3 values + arrays)
    table.insert(data, level_state.level_number)
    table.insert(data, level_state.level_type)
    table.insert(data, level_state.level_shape)
    
    -- Spike heights (fixed size: 16)
    for i = -8, 7 do  -- Relative positions from -8 to 7
        table.insert(data, level_state.spike_heights[i] or 0)
    end
    
    -- Level angles (fixed size: 16)
    for i = 0, 15 do
        table.insert(data, level_state.level_angles[i] or 0)
    end
    
    -- Enemies state (counts: 10 values)
    table.insert(data, enemies_state.active_flippers)
    table.insert(data, enemies_state.active_pulsars)
    table.insert(data, enemies_state.active_tankers)
    table.insert(data, enemies_state.active_spikers)
    table.insert(data, enemies_state.active_fuseballs)
    table.insert(data, enemies_state.spawn_slots_flippers)
    table.insert(data, enemies_state.spawn_slots_pulsars)
    table.insert(data, enemies_state.spawn_slots_tankers)
    table.insert(data, enemies_state.spawn_slots_spikers)
    table.insert(data, enemies_state.spawn_slots_fuseballs)
    
    -- Enemy type info (fixed size: 7)
    for i = 1, 7 do
        table.insert(data, enemies_state.enemy_type_info[i] or 0)
    end
    
    -- Active enemy info (fixed size: 7)
    for i = 1, 7 do
        table.insert(data, enemies_state.active_enemy_info[i] or 0)
    end
    
    -- Enemy segments (fixed size: 7)
    for i = 1, 7 do
        table.insert(data, enemies_state.enemy_segments[i] or 0)
    end
    
    -- Enemy depths (fixed size: 14 - 7 positions and 7 fractions)
    for i = 1, 7 do
        if type(enemies_state.enemy_depths[i]) == "table" then
            table.insert(data, enemies_state.enemy_depths[i].pos or 0)
            table.insert(data, enemies_state.enemy_depths[i].frac or 0)
        else
            table.insert(data, 0)  -- Position
            table.insert(data, 0)  -- Fraction
        end
    end
    
    -- Enemy shot positions (fixed size: 4)
    for i = 1, 4 do
        table.insert(data, enemies_state.shot_positions[i] or 0)
    end
    
    -- Enemy shot segments (fixed size: 4)
    for i = 1, 4 do
        table.insert(data, enemies_state.enemy_shot_segments[i].value)
    end
    
    -- Additional game state (pulse beat, pulsing)
    table.insert(data, enemies_state.pulse_beat or 0)
    table.insert(data, enemies_state.pulsing or 0)
    
    -- Add pending_vid (64 bytes)
    for i = 1, 64 do
        table.insert(data, enemies_state.pending_vid[i] or 0)
    end
    
    -- Add pending_seg (64 bytes)
    for i = 1, 64 do
        table.insert(data, enemies_state.pending_seg[i] or 0)
    end
    
    -- Serialize the data to a binary string
    local binary_data = ""
    for i, value in ipairs(data) do
        -- Apply offset encoding to handle negative values
        local encoded_value = value + 32768
        -- Check if the encoded value would overflow
        if encoded_value > 65535 then
            print("Warning: Value at index " .. i .. " is too large: " .. value)
            encoded_value = 65535
        elseif encoded_value < 0 then
            print("Warning: Value at index " .. i .. " is too negative: " .. value)
            encoded_value = 0
        end
        binary_data = binary_data .. string.pack(">I2", encoded_value)
    end
  
    -- Check if it's time to send a save signal
    local current_time = os.time()
    local save_signal = 0
    
    -- Send save signal every N seconds or when exiting play mode or when shutdown requested
    if current_time - game_state.last_save_time >= game_state.save_interval or shutdown_requested then
        save_signal = 1
        game_state.last_save_time = current_time
        print("Sending save signal to Python script")
        
        -- If this is a shutdown save, add extra debug info
        if shutdown_requested then
            print("SHUTDOWN SAVE: Final save before MAME exits")
        end
    end
    
    -- Debug output for game mode value occasionally
    if game_state.frame_counter % 60 == 0 then
--        print(string.format("Game Mode: 0x%02X, Is Attract Mode: %s", game_state.game_mode, (game_state.game_mode & 0x80) == 0 and "true" or "false"))
    end
    
    -- Create out-of-band context information structure
    -- Pack: num_values (uint32), reward (double), game_action (byte), game_mode (byte), 
    -- done flag (byte), frame_counter (uint32), score (uint32), save_signal (byte),
    -- fire_commanded (byte), zap_commanded (byte), spinner_delta (int8)
    local is_done = 0  -- We don't currently track if the game is done
    local oob_data = string.pack(">IdBBBIIBBBh", 
        1,                    -- num_values
        LastRewardState,      -- reward
        0,                    -- game_action
        game_state.game_mode, -- game_mode
        is_done,              -- done flag
        game_state.frame_counter, -- frame_counter
        player_state.score,   -- score
        save_signal,          -- save_signal
        controls.fire_commanded, -- fire_commanded (added)
        controls.zap_commanded,  -- zap_commanded (added)
        controls.spinner_delta   -- spinner_delta (added)
    )
    
    -- Combine out-of-band header with game state data
    local final_data = oob_data .. binary_data
    
    return final_data, #data  -- Return the number of 16-bit integers
end

-- Update the frame_callback function to track bytes sent and calculate FPS
local frame_counter = 0

local function frame_callback()
    -- Declare num_values at the start of the function
    local num_values = 0

    -- Check if pipes are open, try to reopen if not
    if not pipe_out or not pipe_in then
        if not open_pipes() then
            -- If pipes aren't open yet, just update the display without sending data
            -- Update all state objects
            game_state:update(mem)
            level_state:update(mem)
            player_state:update(mem)
            enemies_state:update(mem)
            
            -- Update the display with empty controls
            update_display("Waiting for Python connection...", game_state, level_state, player_state, enemies_state, nil, num_values)
            return true
        end
    end

    -- Reset the countdown timer to zero all the time
    mem:write_u8(0x0004, 0)

    -- NOP out the jump that skips scoring in attract mode
    mem:write_direct_u8(0xCA6F, 0xEA)
    mem:write_direct_u8(0xCA70, 0xEA)
    
    -- NOP out the damage the copy protection code does to memory when it detects a bad checksum
    mem:write_direct_u8(0xA591, 0xEA)
    mem:write_direct_u8(0xA592, 0xEA)

    -- Increase the maximum level for demo mode
    mem:write_direct_u8(0x9196, 0x0F)


    -- Update all state objects
    game_state:update(mem)
    level_state:update(mem)
    player_state:update(mem)
    enemies_state:update(mem)

    -- Initialize action to "none" as default
    local action = "none"
    local status_message = ""

    -- Massage game state to keep it out of the high score and banner modes
    if game_state.countdown_timer > 0 then
        -- Write 0 to memory location 4
        mem:write_u8(0x0004, 0)
        game_state.countdown_timer = 0
    end

    local is_attract_mode = (game_state.game_mode & 0x80) == 0

    -- In attract mode, zero the score if we're dead or zooming down the tube

    if is_attract_mode then
        if game_state.gamestate == 0x06 or game_state.gamestate == 0x20 then
            mem:write_direct_u8(0x0040, 0x00)
            mem:write_direct_u8(0x0041, 0x00)
            mem:write_direct_u8(0x0042, 0x00)
        end
    else
        -- Four lives at all times
        mem:write_direct_u8(0x0048, 0x04)
    end

    -- We only control the game in regular play mode (04) and zooming down the tube (20)
    if game_state.gamestate == 0x04 or game_state.gamestate == 0x20 then


        -- NOP out the jump that skips scoring in attract mode
        mem:write_direct_u8(0xCA6F, 0xEA)
        mem:write_direct_u8(0xCA70, 0xEA)

        -- NOP out the clearing of zap_fire_new
        mem:write_direct_u8(0x976E, 0x00)
        mem:write_direct_u8(0x976F, 0x00)
        
        -- Calculate the reward for the current frame - do this ONCE per frame
        local reward = calculate_reward(game_state, level_state, player_state)
        
        -- Add periodic save mechanism based on frame counter instead of key press
        -- This will trigger a save every 30,000 frames (approximately 8 minutes at 60fps)
        if game_state.frame_counter % 30000 == 0 then
            print("Frame counter triggered save at frame " .. game_state.frame_counter)
            game_state.last_save_time = 0  -- Force save on next frame
        end
        
        -- Try to detect ESC key press using a more reliable method
        -- Check for ESC key in all available ports
        local esc_pressed = false
        for port_name, port in pairs(manager.machine.ioport.ports) do
            for field_name, field in pairs(port.fields) do
                if field_name:find("ESC") or field_name:find("Escape") then
                    if field.pressed then
                        print("ESC key detected - Triggering save")
                        game_state.last_save_time = 0  -- Force save on next frame
                        esc_pressed = true
                        break
                    end
                end
            end
            if esc_pressed then break end
        end
        
        -- Flatten and serialize the game state data
        local frame_data
        frame_data, num_values = flatten_game_state_to_binary(game_state, level_state, player_state, enemies_state)

        -- Send the serialized data to the Python script and get the components
        local fire, zap, spinner = process_frame(frame_data, player_state, controls, reward)

        -- Update total bytes sent
        total_bytes_sent = total_bytes_sent + #frame_data

        -- Calculate FPS
        frame_count = frame_count + 1
        local current_time = os.time()
        if current_time > last_fps_time then
            current_fps = frame_count / (current_time - last_fps_time)
            frame_count = 0
            last_fps_time = current_time
        end

        -- In LOG_ONLY_MODE, limit display updates to 10 times per second
        local current_time_high_res = os.clock()
        local should_update_display = not LOG_ONLY_MODE or 
                                     (current_time_high_res - last_display_update) >= DISPLAY_UPDATE_INTERVAL

        if should_update_display then
            -- Update the display with the current action and metrics
            update_display(status_message, game_state, level_state, player_state, enemies_state, action, num_values, reward)
            last_display_update = current_time_high_res
        end

        -- Apply the action to MAME controls
        controls:apply_action(fire, zap, spinner, game_state, player_state)
    end

    

    return true
end

-- Update the update_display function to show total bytes sent and FPS
function update_display(status, game_state, level_state, player_state, enemies_state, current_action, num_values, reward)
    clear_screen()
    move_cursor_to_row(1)

    -- Format and print game state in 3 columns at row 1
    
    -- Create game metrics in a more organized way for 3-column display
    local game_metrics = {
        {"Gamestate", string.format("0x%02X", game_state.gamestate)},
        {"Game Mode", string.format("0x%02X", game_state.game_mode)},
        {"Countdown", string.format("0x%02X", game_state.countdown_timer)},
        {"Credits", game_state.credits},
        {"P1 Lives", game_state.p1_lives},
        {"P1 Level", game_state.p1_level},
        {"Frame", game_state.frame_counter},
        {"Bytes Sent", total_bytes_sent},
        {"FPS", string.format("%.2f", current_fps)},
        {"Payload Size", num_values},
        {"Last Reward", string.format("%.2f", LastRewardState)},  -- Add last reward to game metrics
        {"Log Only Mode", LOG_ONLY_MODE and "ENABLED" or "OFF"}   -- Show log-only mode status
    }
    
    -- Print game metrics in 3 columns
    print("--[ Game State ]--------------------------------------")
    local col_width = 35  -- Width for each column
    
    -- Calculate how many rows we need (ceiling of items/3)
    local rows = math.ceil(#game_metrics / 3)
    
    for row = 1, rows do
        local line = "  "
        for col = 1, 3 do
            local idx = (row - 1) * 3 + col
            if idx <= #game_metrics then
                local key, value = table.unpack(game_metrics[idx])
                -- Format each metric with fixed width
                line = line .. string.format("%-12s: %s ", key, value)
            end
        end
        print(line)
    end
    print("")  -- Empty line after section

    -- Format and print player state in 3 columns at row 6
    move_cursor_to_row(7)
    
    -- Create player metrics in a more organized way for 3-column display
    local player_metrics = {
        {"Position", player_state.position .. " "},
        {"State", string.format("0x%02X", player_state.player_state) .. " "},
        {"Depth", player_state.player_depth .. " "},
        {"Alive", player_state.alive .. " "},
        {"Score", player_state.score .. " "},
        {"Szapper Uses", player_state.superzapper_uses .. " "},
        {"Szapper Active", player_state.superzapper_active .. " "},
        {"Shot Count", player_state.shot_count .. " "},
        {"Debounce", player_state.debounce .. " "},
        {"Fire Detected", player_state.fire_detected .. " "},
        {"Zap Detected", player_state.zap_detected .. " "},
        {"SpinnerAccum", player_state.SpinnerAccum .. " "},
        {"SpinnerDelta", player_state.SpinnerDelta .. " "},
        {"InferredDelta", player_state.inferredSpinnerDelta .. " "}
    }
    
    -- Print player metrics in 3 columns
    print("--[ Player State ]------------------------------------")
    
    -- Calculate how many rows we need (ceiling of items/3)
    local rows = math.ceil(#player_metrics / 3)
    
    for row = 1, rows do
        local line = "  "
        for col = 1, 3 do
            local idx = (row - 1) * 3 + col
            if idx <= #player_metrics then
                local key, value = table.unpack(player_metrics[idx])
                -- Format each metric with fixed width
                line = line .. string.format("%-12s: %s", key, value)
            end
        end
        print(line)
    end
    
    -- Add shot positions on its own line
    local shots_str = ""
    for i = 1, 8 do
        local segment_str = string.format("%+02d", player_state.shot_segments[i])
        shots_str = shots_str .. string.format("%02X-%s ", player_state.shot_positions[i], segment_str)
    end
    print("  Shot Positions: " .. shots_str)
    
    -- Display enemy shot segments
    local shot_segments_str = ""
    for i = 1, 4 do
        shot_segments_str = shot_segments_str .. string.format("%02X ", enemies_state.enemy_shot_segments[i].value)
    end
    print("  Shot Segments : " .. shot_segments_str)
    
    print("")  -- Empty line after section

    -- Format and print player controls at row 14
    move_cursor_to_row(15)
    local is_attract_mode = (game_state.game_mode & 0x80) == 0
    local controls_metrics = {
        ["Fire (Inferred)"] = controls.fire_commanded,
        ["Superzapper (Inferred)"] = controls.zap_commanded,
        ["Spinner Delta (Inferred)"] = controls.spinner_delta,
        ["Attract Mode"] = is_attract_mode and "Active" or "Inactive"
    }
    print(format_section("Player Controls (Game Inferred Values)", controls_metrics))

    -- Add new Model State section
    move_cursor_to_row(21)
    local model_metrics = {
        ["Model Fire"] = model_fire,
        ["Model Zap"] = model_zap,
        ["Model Spinner"] = model_spinner
    }
    print(format_section("Model Output", model_metrics))

    -- Format and print level state in 3 columns (adjust row number)
    move_cursor_to_row(26)  -- Changed from 23 to make room for Model State
    
    -- Print level metrics in 3 columns
    print("--[ Level State ]-------------------------------------")

    -- Create level metrics in a more organized way for 3-column display
    local level_metrics_list = {
        {"Level Number", level_state.level_number},
        {"Level Type", level_state.level_type == 0xFF and "Open" or "Closed"},
        {"Level Shape", level_state.level_shape}
    }
    
    -- Calculate how many rows we need (ceiling of items/3)
    local rows = math.ceil(#level_metrics_list / 3)
    
    for row = 1, rows do
        local line = "  "
        for col = 1, 3 do
            local idx = (row - 1) * 3 + col
            if idx <= #level_metrics_list then
                local key, value = table.unpack(level_metrics_list[idx])
                -- Format each metric with fixed width
                line = line .. string.format("%-12s: %s", key, value)
            end
        end
        print(line)
    end
    
    -- Add spike heights on its own line
    local heights_str = ""
    -- For open levels, show 0-15, for closed levels show -8 to +7 but just display the values
    for i = -8, 7 do
        if level_state.spike_heights[i] then
            heights_str = heights_str .. string.format("%02X ", level_state.spike_heights[i])
        else
            heights_str = heights_str .. "-- "
        end
    end
    print("  Spike Heights: " .. heights_str)
    
    -- Add level angles on its own line
    local angles_str = ""
    for i = 0, 15 do
        angles_str = angles_str .. string.format("%02X ", level_state.level_angles[i])
    end
    print("  Level Angles : " .. angles_str)
    print("")  -- Empty line after section

    -- Format and print enemies state at row 31
    move_cursor_to_row(31)
    local enemy_types = {}
    local enemy_states = {}
    local enemy_segs = {}
    local enemy_depths = {}
    for i = 1, 7 do
        enemy_types[i] = enemies_state:decode_enemy_type(enemies_state.enemy_type_info[i])
        enemy_states[i] = enemies_state:decode_enemy_state(enemies_state.active_enemy_info[i])
        enemy_segs[i] = string.format("%+3d", enemies_state.enemy_segments[i])
        enemy_depths[i] = string.format("%02X.%02X", enemies_state.enemy_depths[i].pos, enemies_state.enemy_depths[i].frac)
    end

    local enemies_metrics = {
        ["Flippers"] = string.format("%d active, %d spawn slots", enemies_state.active_flippers, enemies_state.spawn_slots_flippers),
        ["Pulsars"] = string.format("%d active, %d spawn slots", enemies_state.active_pulsars, enemies_state.spawn_slots_pulsars),
        ["Tankers"] = string.format("%d active, %d spawn slots", enemies_state.active_tankers, enemies_state.spawn_slots_tankers),
        ["Spikers"] = string.format("%d active, %d spawn slots", enemies_state.active_spikers, enemies_state.spawn_slots_spikers),
        ["Fuseballs"] = string.format("%d active, %d spawn slots", enemies_state.active_fuseballs, enemies_state.spawn_slots_fuseballs),
        ["Total"] = string.format("%d active, %d spawn slots", 
            enemies_state:get_total_active(),
            enemies_state.spawn_slots_flippers + enemies_state.spawn_slots_pulsars + 
            enemies_state.spawn_slots_tankers + enemies_state.spawn_slots_spikers + 
            enemies_state.spawn_slots_fuseballs),
        ["Pulse State"] = string.format("beat:%02X charge:%02X/FF", enemies_state.pulse_beat, enemies_state.pulsing),
        ["Enemy Types"] = table.concat(enemy_types, " "),
        ["Enemy States"] = table.concat(enemy_states, " ")
    }

    print(format_section("Enemies State", enemies_metrics))

    -- move_cursor_to_row(40)  -- Move Enemy Segments display up one line   
    -- Add enemy segments and depths on their own lines
    print("  Enemy Segments: " .. table.concat(enemy_segs, " "))
    print("  Enemy Depths  : " .. table.concat(enemy_depths, " "))
    
    -- Add enemy shot positions in a simple fixed format
    local shot_positions_str = ""
    for i = 1, 4 do
        local pos_value = enemies_state.shot_positions[i] or 0
        -- Mask to 8 bits (0-255) to display only one byte
        pos_value = pos_value & 0xFF
        shot_positions_str = shot_positions_str .. string.format("%02X ", pos_value)
    end
    print("  Shot Positions: " .. shot_positions_str)
    
    -- Display enemy shot segments
    local shot_segments_str = ""
    for i = 1, 4 do
        shot_segments_str = shot_segments_str .. string.format("%02X ", enemies_state.enemy_shot_segments[i].value)
    end
    print("  Shot Segments : " .. shot_segments_str)
    
    print("")  -- Empty line after section

    -- Display pending_vid (64 bytes)
    local pending_vid_str = ""
    for i = 1, 64 do
        pending_vid_str = pending_vid_str .. string.format("%02X ", enemies_state.pending_vid[i])
        if i % 16 == 0 then pending_vid_str = pending_vid_str .. "\n  " end
    end
    print("  Pending VID   : " .. pending_vid_str)

    -- Display pending_seg similarly
    local pending_seg_str = ""
    for i = 1, 64 do
        pending_seg_str = pending_seg_str .. string.format("%02X ", enemies_state.pending_seg[i])
        if i % 16 == 0 then pending_seg_str = pending_seg_str .. "\n  " end
    end
    print("  Pending SEG   : " .. pending_seg_str)
end

-- Function to be called when MAME is shutting down
local function on_mame_exit()
    print("MAME is shutting down - Sending final save signal")
    
    -- Set shutdown flag to trigger save in next frame
    shutdown_requested = true
    
    -- Force one more frame update to ensure save
    -- We can use the existing objects since they should still be valid
    if game_state and level_state and player_state and enemies_state then
        -- Calculate reward one more time
        local reward = calculate_reward(game_state, level_state, player_state)
        
        -- Get final frame data with save signal
        local frame_data = flatten_game_state_to_binary(game_state, level_state, player_state, enemies_state)
        
        -- Send one last time
        if pipe_out and pipe_in then
            local result = process_frame(frame_data, player_state, controls, reward)
            print("Final save complete, response: " .. (result or "none"))
        end
    end
    
    -- Close pipes as in the existing cleanup function
    if pipe_out then pipe_out:close() end
    if pipe_in then pipe_in:close() end
    print("Pipes closed during MAME shutdown")
    
    -- Close log file
    if log_file then
        log_file:close()
        log_file = nil
        print("Log file closed during MAME shutdown")
    end
end

-- Start the Python script but don't wait for pipes to open
start_python_script()
clear_screen()

-- Register the frame callback with MAME.  Keep a reference to the callback or it will be garbage collected
callback_ref = emu.add_machine_frame_notifier(frame_callback)

-- Register the shutdown callback
emu.register_stop(on_mame_exit)

