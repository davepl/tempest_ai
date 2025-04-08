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

-- Define constants
local INVALID_SEGMENT = -32768  -- Used as sentinel value for invalid segments

local LOG_ONLY_MODE           = false
local AUTO_PLAY_MODE          = not LOG_ONLY_MODE
local SHOW_DISPLAY            = true
local DISPLAY_UPDATE_INTERVAL = 0.05  

local function clear_screen()
    io.write("\027[2J\027[H")
end

-- Function to move the cursor to the home position using ANSI escape codes
local function move_cursor_home()
    io.write("\027[H")
end

-- Log file for storing frame data
local log_file = nil  -- Will hold the file handle, not the path
local log_file_path = "/Users/dave/mame/tempest.log"  -- Store the path separately
local log_file_opened = false  -- Track if we've already opened the file

-- Global socket variable
local socket = nil

-- Add this near the top of the file with other global variables
local frame_count = 0  -- Initialize frame counter

-- Global variables for tracking bytes sent and FPS
local total_bytes_sent = 0

-- Function to open socket connection
local function open_socket()
    
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

    -- Try to open socket connection
    local socket_success, err = pcall(function()
        -- Close existing socket if any
        if socket then
            socket:close()
            socket = nil 
        end
        
        -- Create a new socket connection
        socket = emu.file("rw")  -- "rw" mode for read/write
        local result = socket:open("socket.192.168.1.248:9999")
        
        if result == nil then
            print("Successfully opened socket connection to localhost:9999")
            
            -- Send initial 4-byte ping for handshake
            local ping_data = string.pack(">H", 0)  -- 2-byte integer with value 0
            socket:write(ping_data)
            print("Initial handshake ping sent")
        else
            print("Failed to open socket connection: " .. tostring(result))
            socket = nil
            return false
        end
    end)
    
    if not socket_success or not socket then
        print("Error opening socket connection: " .. tostring(err or "unknown error"))
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


-- Declare global variables for reward calculation
local previous_score = 0
local previous_level = 0
local previous_alive_state = 1  -- Track previous alive state, initialize as alive

-- Declare a global variable to store the last reward state
local LastRewardState = 0

local shutdown_requested = false

local last_display_update = 0  -- Timestamp of last display update

-- Function to calculate reward for the current frame
local function calculate_reward(game_state, level_state, player_state, enemies_state)
    local reward = 0
    local bDone = false

    -- We want as many shots active as possible, but only up to 7, so that we have one in reserve for close enemies
    
    if (player_state.shot_count < 8) then
        reward = reward + player_state.shot_count
    end

    -- Base survival reward - make staying alive more valuable
    
    if player_state.alive == 1 then
        reward = reward + 10  -- Constant reward for being alive each frame

        -- Stronger reward for maintaining lives
        if player_state.player_lives ~= nil then
            reward = reward + (player_state.player_lives)
        end

        -- Score-based reward (keep this as a strong motivator)
        local score_delta = player_state.score - previous_score
        if score_delta > 0 then
            reward = reward + (score_delta * 2)  -- Amplify score impact
        end

        -- Level completion bonus
        if level_state.level_number ~= previous_level then
            reward = reward + (50 * previous_level)  -- Increased bonus for progression
        end

        -- Enemy targeting logic
        local target_segment = enemies_state:nearest_enemy_segment()
        local player_segment = player_state.position & 0x0F

        if target_segment < 0 or game_state.gamestate == 0x20 then
            -- No enemies: reward staying still more strongly
            reward = reward + (player_state.SpinnerDelta == 0 and 50 or -20)
        else
            -- Get desired spinner direction, segment distance, AND enemy depth
            local desired_spinner, segment_distance, enemy_depth = direction_to_nearest_enemy(game_state, level_state, player_state, enemies_state)

            -- Check alignment based on actual segment distance
            if segment_distance == 0 then 
                -- Massive reward for alignment + firing incentive
                reward = reward + 250
                -- Bonus for shooting when aligned
                if player_state.shot_count > 0 then
                    reward = reward + 100
                end
                -- Small penalty for unnecessary movement when aligned (but could be moving to next enemy after firing, so not that bad)
                if player_state.SpinnerDelta ~= 0 then
                    reward = reward - 10
                end
            else 
                -- Misaligned case (segment_distance > 0)
                -- Enemies at the top of tube should be shot when close (using segment distance)
                if (segment_distance < 2) then -- Check using actual segment distance
                    -- Use the depth returned by direction_to_nearest_enemy
                    if (enemy_depth <= 0x20) then 
                        if player_state.fire_commanded == 1 then
                            -- Strong reward for firing at close enemies
                            reward = reward + 250
                        else
                            -- Moderate penalty for not firing at close enemies
                            print("*")
                            reward = reward - 50
                        end
                    end
                end

                -- Graduated reward for proximity (higher reward for smaller segment distance)
                reward = reward + (10 - segment_distance) -- Simple linear reward for proximity
                
                -- Movement incentives (using desired_spinner direction)
                if desired_spinner * player_state.SpinnerDelta < 0 then
                    -- Strong reward for correct movement (signs match)
                    reward = reward + 50
                elseif player_state.SpinnerDelta ~= 0 then
                    -- Strong penalty for wrong movement (signs mismatch)
                    reward = reward - 50
                end
                
                -- Encourage maintaining shots in reserve
                if player_state.shot_count < 2 or player_state.shot_count > 7 then
                    reward = reward - 20  -- Penalty for not having shots ready
                elseif player_state.shot_count >= 4 then
                    reward = reward + 20  -- Bonus for good ammo management
                end
            end
        end

    else
        -- Massive penalty for death to prioritize survival, equal to the cost of a bonus life in points

        if previous_alive_state == 1 then
            reward = reward - 20000
            bDone = true
        end
    end

    -- Update previous values
    previous_score = player_state.score
    previous_level = level_state.level_number
    previous_alive_state = player_state.alive

    LastRewardState = reward

    return reward, bDone
end

-- Function to send parameters and get action each frame
local function process_frame(rawdata, player_state, controls, reward, bDone, bAttractMode)
    -- In log-only mode, we only write to the log file and don't communicate with Python

    -- Log the frame data to file
    if log_file then
        local success, err = pcall(function()
            -- Write the raw data   
            log_file:write(rawdata)
            log_file:flush()  -- Ensure data is written to disk
        end)
        
        if not success then
            print("ERROR writing to log file: " .. tostring(err))
        end
    else
        -- Log file not open, attempting to open it
        print("Log file not open, attempting to reopen")
        open_socket()
    end
    
    -- Increment frame count after processing
    frame_count = frame_count + 1

    if LOG_ONLY_MODE then
        return controls.fire_commanded, controls.zap_commanded, controls.spinner_delta
    end
    
    -- Check if socket is open, try to reopen if not
    if not socket then
        if not open_socket() then
            return 0, 0, 0  -- Return zeros for fire, zap, spinner_delta on socket error
        end
    end
  
    -- Try to write to socket, handle errors
    local success, err = pcall(function()
        -- Add 4-byte length header to rawdata
        local data_length = #rawdata
        local length_header = string.pack(">H", data_length)
        
        -- Write length header followed by data
        socket:write(length_header)
        socket:write(rawdata)
    end)
    
    if not success then
        print("Error writing to socket:", err)
        -- Close and attempt to reopen socket
        if socket then socket:close(); socket = nil end
        open_socket()
        return 0, 0, 0  -- Return zeros for fire, zap, spinner_delta on write error
    end
    
    -- Try to read from socket with timeout protection
    local fire, zap, spinner = 0, 0, 0  -- Default values
    local read_start_time = os.clock()
    local read_timeout = 0.5  -- 500ms timeout for socket read
    
    success, err = pcall(function()
        -- Use non-blocking approach with timeout
        local action_bytes = nil
        local elapsed = 0
        
        while not action_bytes and elapsed < read_timeout do
            action_bytes = socket:read(3)
            
            if not action_bytes or #action_bytes < 3 then
                -- If read fails, sleep a tiny bit and try again
                -- Use non-blocking sleep
                local wait_start = os.clock()
                while os.clock() - wait_start < 0.01 do
                    -- Busy wait instead of os.execute which can freeze MAME
                end
                elapsed = os.clock() - read_start_time
                action_bytes = nil
            end
        end
        
        if action_bytes and #action_bytes == 3 then
            -- Unpack the three signed 8-bit integers
            fire, zap, spinner = string.unpack("bbb", action_bytes)
        else
            -- Default action if read fails or times out
            print("Failed to read action from socket after " .. elapsed .. "s, got " .. 
                  (action_bytes and #action_bytes or 0) .. " bytes")
            fire, zap, spinner = 0, 0, 0
            
            -- If we timed out, reconnect
            if elapsed >= read_timeout then
                error("Socket read timeout exceeded")
            end
        end
    end)
    
    if not success then
        print("Error reading from socket:", err)
        -- Close and attempt to reopen socket
        if socket then socket:close(); socket = nil end
        open_socket()
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
    
    -- Launch Python script in the background with proper error handling
    local cmd = "python /Users/dave/source/repos/tempest/Scripts/aimodel.py >/tmp/python_output.log 2>&1 &"
    local result = os.execute(cmd)
    
    if result ~= 0 then
        print("Warning: Failed to start Python script")
        return false
    end
    
    -- Give Python script a moment to start up and create socket server
    -- print("Waiting for Python script to initialize...")
    os.execute("sleep 3")
    
    -- Check if Python script is running
    local python_running = os.execute("pgrep -f 'python.*aimodel.py' >/dev/null") == 0
    if not python_running then
        print("Warning: Python script failed to start or terminated early")
        print("Check /tmp/python_output.log for errors")
        return false
    end
    
    -- print("Python script started successfully")
    
    -- Try to open socket immediately
    local socket_opened = open_socket()
    if not socket_opened then
        print("Initial socket connection failed, will retry during frame callback")
    end
    
    return true
end

start_python_script()
clear_screen()

-- Add after the initial requires, before the GameState class

-- Seed the random number generator once at script start
math.randomseed(os.time())

-- Access the main CPU and memory space
local mainCpu = nil
local mem = nil

-- Try to access machine using the manager API with proper error handling
local success, err = pcall(function()
    -- First, check if manager is available
    if not manager then
        error("MAME manager API not available")
    end
    
    -- Then check if machine is available on manager
    if not manager.machine then
        error("manager.machine not available")
    end
    
    -- Finally, access the devices
    mainCpu = manager.machine.devices[":maincpu"]
    if not mainCpu then
        error("Main CPU not found")
    end
    
    mem = mainCpu.spaces["program"]
    if not mem then
        error("Program memory space not found")
    end
end)

if not success then
    print("Error accessing MAME machine: " .. tostring(err))
    print("Attempting alternative access method...")
    
    -- Try alternative access methods
    success, err = pcall(function()
        -- Try accessing machine directly if available
        if machine then
            mainCpu = machine.devices[":maincpu"]
            if not mainCpu then
                error("Main CPU not found via machine")
            end
            
            mem = mainCpu.spaces["program"]
            if not mem then
                error("Program memory space not found via machine")
            end
        else
            error("Neither manager.machine nor machine is available")
        end
    end)
    
    if not success then
        print("Error with alternative access method: " .. tostring(err))
        print("FATAL: Cannot access MAME memory")
        return
    end
end

print("Successfully accessed MAME memory interface")

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
    
    -- FPS tracking (now handled at global level, not in GameState)
    self.current_fps = 0  -- Store the FPS value for display
    
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
    
    -- The current_fps is now only updated when FPS is calculated in frame_callback
end

-- Store the last time a full debug dump was performed
local last_debug_time = os.clock()

-- Enhanced debug print for the frame callback
local function debug_dump_state()
    local current_time = os.clock()
    if current_time - last_debug_time < 5.0 then
        return  -- Only dump debug info every 5 seconds
    end
    
    last_debug_time = current_time
    
    print("\n----- DEBUG STATE DUMP -----")
    print(string.format("Game counter: %d, Reported FPS: %.2f", 
        game_state.frame_counter, game_state.current_fps))
    
    -- Dump timing data
    print(string.format("Current time: %.6f, Last FPS time: %.6f, Diff: %.6f", 
        current_time, game_state.last_fps_time, current_time - game_state.last_fps_time))
    print(string.format("Frame count accum: %d", game_state.frame_count_accum))
    print(string.format("Current timer value: %d", mem:read_u8(0x0003)))
    print("---------------------------\n")
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
    
    -- Read spike heights for all 16 segments and store them using absolute positions
    self.spike_heights = {}
    for i = 0, 15 do  -- Use 0-based indexing to match game's segment numbering
        local height = mem:read_u8(0x03AC + i)
        -- Store absolute position directly
        self.spike_heights[i] = height
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
    self.fire_commanded = 0   -- Add fire_commanded field
    self.zap_commanded = 0    -- Add zap_commanded field
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
    local player_abs_segment = self.position & 0x0F
    for i = 1, 8 do
        -- Read depth (position along the tube)
        self.shot_positions[i] = mem:read_u8(0x02D3 + i - 1)  -- PlayerShotPositions

        -- If shot position is 0, the shot is inactive, regardless of segment memory
        if self.shot_positions[i] == 0 then
            self.shot_segments[i] = INVALID_SEGMENT
        else
            -- Position is non-zero, now check the segment memory
            local abs_segment = mem:read_u8(0x02AD + i - 1)
            if abs_segment == 0 then -- Also treat segment 0 as inactive/invalid
                 self.shot_segments[i] = INVALID_SEGMENT
            else
                 -- Valid position and valid segment read, calculate relative segment
                 abs_segment = abs_segment & 0x0F  -- Mask to get valid segment 0-15
                 self.shot_segments[i] = absolute_to_relative_segment(player_abs_segment, abs_segment, is_open)
            end
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
    self.pulsar_fliprate = 0 -- NEW: Add pulsar fliprate
    self.num_enemies_in_tube = 0
    self.num_enemies_on_top = 0
    self.enemies_pending = 0
    self.nearest_enemy_seg = INVALID_SEGMENT  -- Initialize relative nearest enemy segment with sentinel
    
    -- Enemy info arrays (Original)
    self.enemy_type_info = {0, 0, 0, 0, 0, 0, 0}    -- Raw type byte from $0169 (or similar)
    self.active_enemy_info = {0, 0, 0, 0, 0, 0, 0}  -- Raw state byte from $0170 (or similar)
    self.enemy_segments = {0, 0, 0, 0, 0, 0, 0}     -- 7 enemy segment numbers ($02B9)
    self.enemy_depths = {0, 0, 0, 0, 0, 0, 0}       -- 7 enemy depth positions ($02DF)
    -- LSB/Display list related fields remain...
    self.enemy_depths_lsb = {0, 0, 0, 0, 0, 0, 0}   -- 7 enemy depth positions LSB
    self.enemy_shot_lsb = {0, 0, 0, 0, 0, 0, 0}     -- 7 enemy shot LSB values at $02E6

    -- NEW: Decoded Enemy Info Tables (Size 7)
    self.enemy_core_type = {0, 0, 0, 0, 0, 0, 0}        -- Bits 0-2 from type byte
    self.enemy_direction_moving = {0, 0, 0, 0, 0, 0, 0} -- Bit 6 from type byte (0/1)
    self.enemy_between_segments = {0, 0, 0, 0, 0, 0, 0} -- Bit 7 from type byte (0/1)
    self.enemy_moving_away = {0, 0, 0, 0, 0, 0, 0}      -- Bit 7 from state byte (0/1)
    self.enemy_can_shoot = {0, 0, 0, 0, 0, 0, 0}        -- Bit 6 from state byte (0/1)
    self.enemy_split_behavior = {0, 0, 0, 0, 0, 0, 0}   -- Bits 0-1 from state byte

    -- Convert shot positions to simple array like other state values
    self.shot_positions = {0, 0, 0, 0}  -- 4 shot positions
    
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
    -- First, initialize/reset all arrays at the beginning
    -- Reset enemy arrays
    -- Keep resetting original raw arrays
    self.enemy_type_info = {0, 0, 0, 0, 0, 0, 0}
    self.active_enemy_info = {0, 0, 0, 0, 0, 0, 0}
    self.enemy_segments = {0, 0, 0, 0, 0, 0, 0}
    self.enemy_depths = {0, 0, 0, 0, 0, 0, 0}
    self.enemy_depths_lsb = {0, 0, 0, 0, 0, 0, 0}
    self.enemy_shot_lsb = {0, 0, 0, 0, 0, 0, 0}
    self.enemy_move_vectors = {0, 0, 0, 0, 0, 0, 0} -- Keep this reset if used elsewhere
    self.enemy_state_flags = {0, 0, 0, 0, 0, 0, 0}  -- Keep this reset if used elsewhere
    -- NEW: Reset decoded tables
    self.enemy_core_type = {0, 0, 0, 0, 0, 0, 0}
    self.enemy_direction_moving = {0, 0, 0, 0, 0, 0, 0}
    self.enemy_between_segments = {0, 0, 0, 0, 0, 0, 0}
    self.enemy_moving_away = {0, 0, 0, 0, 0, 0, 0}
    self.enemy_can_shoot = {0, 0, 0, 0, 0, 0, 0}
    self.enemy_split_behavior = {0, 0, 0, 0, 0, 0, 0}

    -- Read active enemies counts, pulse state, etc.
    self.active_flippers  = mem:read_u8(0x0142)   -- n_flippers - current active count
    self.active_pulsars   = mem:read_u8(0x0143)    -- n_pulsars
    self.active_tankers   = mem:read_u8(0x0144)    -- n_tankers
    self.active_spikers   = mem:read_u8(0x0145)    -- n_spikers
    self.active_fuseballs = mem:read_u8(0x0146)  -- n_fuseballs
    self.pulse_beat       = mem:read_u8(0x0147)        -- pulse_beat counter
    self.pulsing          = mem:read_u8(0x0148)          -- pulsing state
    self.pulsar_fliprate  = mem:read_u8(0x00B2)          -- NEW: Pulsar flip rate at $B2
    self.num_enemies_in_tube = mem:read_u8(0x0108)
    self.num_enemies_on_top = mem:read_u8(0x0109)
    self.enemies_pending = mem:read_u8(0x03AB)

    -- Update enemy shot segments from memory (store relative distances)
    local player_abs_segment = mem:read_u8(0x0200) & 0x0F -- Get current player absolute segment
    local is_open = mem:read_u8(0x0111) == 0xFF

    for i = 1, 4 do
        local abs_segment = mem:read_u8(self.enemy_shot_segments[i].address)
        if abs_segment == 0 then
            self.enemy_shot_segments[i].value = INVALID_SEGMENT  -- Not active, use sentinel
        else
            local segment = abs_segment & 0x0F  -- Mask to ensure 0-15
            self.enemy_shot_segments[i].value = absolute_to_relative_segment(player_abs_segment, segment, is_open)
        end
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

    local activeEnemies = self.num_enemies_in_tube + self.num_enemies_on_top
    
    -- Read standard enemy segments and depths first, store relative segments
    for i = 1, 7 do
        local abs_segment = mem:read_u8(0x02B9 + i - 1)
        self.enemy_depths[i] = mem:read_u8(0x02DF + i - 1)

        if (self.enemy_depths[i] == 0 or abs_segment == 0) then
            self.enemy_segments[i] = INVALID_SEGMENT  -- Not active, use sentinel
        else
            local segment = abs_segment & 0x0F  -- Mask to ensure 0-15
            -- Store relative segment distance
            self.enemy_segments[i] = absolute_to_relative_segment(player_abs_segment, segment, is_open)
        end
    end

    -- Read raw type/state bytes for all 7 slots THEN decode based on activity check using depth
    local raw_type_bytes = {}
    local raw_state_bytes = {}
    for i = 1, 7 do
        -- Read raw bytes (Type confirmed at $0283, State confirmed at $028A)
        raw_type_bytes[i] = mem:read_u8(0x0283 + i - 1) 
        raw_state_bytes[i] = mem:read_u8(0x028A + i - 1) 
        -- Store raw bytes as well, might be useful
        self.enemy_type_info[i] = raw_type_bytes[i]
        self.active_enemy_info[i] = raw_state_bytes[i]
    end
    
    -- Now loop again, decoding only if the corresponding depth indicates activity
    for i = 1, 7 do
        if self.enemy_depths[i] > 0 then -- Check activity using depth for this index
            local type_byte = raw_type_bytes[i]
            local state_byte = raw_state_bytes[i]
            
            -- Decode Type Byte ($0283)
            self.enemy_core_type[i] = type_byte & 0x07
            self.enemy_direction_moving[i] = (type_byte & 0x40) ~= 0 and 1 or 0 -- Bit 6: Segment increasing?
            self.enemy_between_segments[i] = (type_byte & 0x80) ~= 0 and 1 or 0 -- Bit 7: Between segments?

            -- Decode State Byte ($028A)
            self.enemy_moving_away[i] = (state_byte & 0x80) ~= 0 and 1 or 0 -- Bit 7: Moving Away?
            self.enemy_can_shoot[i] = (state_byte & 0x40) ~= 0 and 1 or 0   -- Bit 6: Can Shoot?
            self.enemy_split_behavior[i] = state_byte & 0x03               -- Bits 0-1: Split Behavior
        else
            -- Zero out decoded values for inactive slots (based on depth)
            self.enemy_core_type[i] = 0
            self.enemy_direction_moving[i] = 0
            self.enemy_between_segments[i] = 0
            self.enemy_moving_away[i] = 0
            self.enemy_can_shoot[i] = 0
            self.enemy_split_behavior[i] = 0
        end
    end

    -- Read all 4 enemy shot positions and store absolute positions
    for i = 1, 4 do
        local raw_pos = mem:read_u8(0x02DB + i - 1)
        self.shot_positions[i] = raw_pos  -- Store full raw position value
    end

    -- Read pending_seg (64 bytes starting at 0x0203), store relative
    for i = 1, 64 do
        local abs_segment = mem:read_u8(0x0203 + i - 1)
        if abs_segment == 0 then
            self.pending_seg[i] = INVALID_SEGMENT  -- Not active, use sentinel
        else
            local segment = abs_segment & 0x0F  -- Mask to ensure 0-15
            -- Store relative segment distance
            self.pending_seg[i] = absolute_to_relative_segment(player_abs_segment, segment, is_open)
        end
    end

    -- Read pending_vid (64 bytes starting at 0x0243)
    for i = 1, 64 do
        self.pending_vid[i] = mem:read_u8(0x0243 + i - 1)
    end

    -- Scan the display list region for additional enemy data
    self.display_list = {}
    for i = 0, 31 do  -- Just scan part of it for efficiency
        local command = mem:read_u8(0x0300 + i * 4)
        local abs_segment = mem:read_u8(0x0301 + i * 4) & 0x0F
        local depth = mem:read_u8(0x0302 + i * 4)
        local type_val = mem:read_u8(0x0303 + i * 4)

        -- Calculate relative segment if command is active
        local rel_segment = INVALID_SEGMENT
        if command ~= 0 and depth ~= 0 and abs_segment >= 0 and abs_segment <= 15 then
            rel_segment = absolute_to_relative_segment(player_abs_segment, abs_segment, is_open)
        end

        self.display_list[i] = {
            command = command,
            segment = rel_segment, -- Store relative segment
            depth = depth,
            type = type_val
        }
    end

    -- Calculate and store the relative nearest enemy segment for internal use
    local nearest_abs_seg, _ = self:nearest_enemy_segment() -- Call the function that finds absolute
    if nearest_abs_seg == -1 then
        self.nearest_enemy_seg = INVALID_SEGMENT
    else
        self.nearest_enemy_seg = absolute_to_relative_segment(player_abs_segment, nearest_abs_seg, is_open)
    end
end

-- Find the *absolute* segment and depth of the enemy closest to the top of the tube
-- This is used primarily for OOB data packing and targeting decisions that need absolute values.
function EnemiesState:nearest_enemy_segment()
    local min_depth = 255
    local closest_absolute_segment = -1 -- Use -1 as sentinel for *absolute* segment not found

    -- Check standard enemy table (7 slots)
    for i = 1, 7 do
        -- Read the absolute segment and depth directly from memory for this calculation
        local abs_segment = mem:read_u8(0x02B9 + i - 1) & 0x0F -- Mask to 0-15
        local depth = mem:read_u8(0x02DF + i - 1)

        -- Only consider active enemies with valid segments (0-15)
        if depth > 0 and abs_segment >= 0 and abs_segment <= 15 then
            if depth < min_depth then
                min_depth = depth
                closest_absolute_segment = abs_segment
            end
        end
    end

    -- Return the absolute segment (-1 if none found) and its depth
    return closest_absolute_segment, min_depth
end

-- function EnemiesState:depth_of_top_enemy() -- This will be removed later

function direction_to_nearest_enemy(game_state, level_state, player_state, enemies_state)
    -- Get the *absolute* segment AND depth of nearest enemy from the dedicated function
    local enemy_abs_seg, enemy_depth = enemies_state:nearest_enemy_segment()
    local player_abs_seg = player_state.position & 0x0F
    local is_open = level_state.level_type == 0xFF

    -- If no enemy was found (absolute segment is -1)
    if enemy_abs_seg == -1 then
        return 0, 0, 255 -- No enemy, return spinner 0, distance 0, max depth
    end

    -- Calculate the relative segment distance using the helper function
    local enemy_relative_dist = absolute_to_relative_segment(player_abs_seg, enemy_abs_seg, is_open)

    -- If already aligned (relative distance is 0)
    if enemy_relative_dist == 0 then
        return 0, 0, enemy_depth -- Aligned, return spinner 0, distance 0, current depth
    end

    local intensity
    local spinner
    local actual_segment_distance

    -- Calculate actual segment distance based on relative distance
    actual_segment_distance = math.abs(enemy_relative_dist)

    -- Set intensity based on distance
    intensity = math.min(0.9, 0.3 + (actual_segment_distance * 0.05))

    -- Set spinner direction based on the sign of the relative distance
    -- The absolute_to_relative_segment function handles open/closed logic correctly
    if is_open then
        -- Open Level: Positive relative distance means enemy is clockwise (higher segment index)
        -- We want to move counter-clockwise (negative spinner) towards it.
        spinner = enemy_relative_dist > 0 and -intensity or intensity
    else
        -- Closed Level: Positive relative distance means enemy is clockwise.
        -- We want to move clockwise (positive spinner) towards it.
        spinner = enemy_relative_dist > 0 and intensity or -intensity
    end

    return spinner, actual_segment_distance, enemy_depth -- Return spinner, distance, AND depth
end


-- Function to decode enemy type info
function EnemiesState:decode_enemy_type(type_byte)
    local enemy_type = type_byte & 0x07
    local between_segments = (type_byte & 0x80) ~= 0
    local segment_increasing = (type_byte & 0x40) ~= 0
    return string.format("%d%s%s", 
        enemy_type,
        between_segments and "B" or "-",
        segment_increasing and "+" or ""  -- Remove the '+' sign for segment numbers
    )
end

-- Function to decode enemy state info
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

-- Instantiate state objects - AFTER defining all classes
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

-- Function to flatten and serialize the game state data to signed 16-bit integers
local function flatten_game_state_to_binary(reward, game_state, level_state, player_state, enemies_state, bDone)
    -- Create a consistent data structure with fixed sizes
    local data = {}

    -- Game state (5 values, frame counter is now in OOB data)
    table.insert(data, game_state.gamestate)
    table.insert(data, game_state.game_mode)
    table.insert(data, game_state.countdown_timer)
    table.insert(data, game_state.p1_lives)
    table.insert(data, game_state.p1_level)

    -- Get nearest enemy relative segment (or INVALID_SEGMENT) stored in state
    local nearest_relative_seg = enemies_state.nearest_enemy_seg -- This is relative or INVALID_SEGMENT
    local segment_delta = 0

    -- Use the relative segment directly as the delta if valid
    if nearest_relative_seg ~= INVALID_SEGMENT then
        segment_delta = nearest_relative_seg
    end

    -- Insert relative segment and delta into main payload
    -- Use INVALID_SEGMENT sentinel for nearest_relative_seg if no enemy
    table.insert(data, nearest_relative_seg)
    table.insert(data, segment_delta)      -- Relative distance (-7 to +8 or -15 to +15) or 0

    -- Player state (5 values + arrays, score is now in OOB data)
    table.insert(data, player_state.position) -- Keep absolute player position here if needed, or mask if only segment used
    table.insert(data, player_state.alive)
    table.insert(data, player_state.player_state)
    table.insert(data, player_state.player_depth)
    table.insert(data, player_state.superzapper_uses)
    table.insert(data, player_state.superzapper_active)
    table.insert(data, player_state.shot_count)

    -- Player shot positions (fixed size: 8) - Absolute depth
    for i = 1, 8 do
        table.insert(data, player_state.shot_positions[i] or 0)
    end

    -- Player shot segments (fixed size: 8) - Relative segments
    for i = 1, 8 do
        table.insert(data, player_state.shot_segments[i] or INVALID_SEGMENT) -- Use sentinel
    end

    -- Level state (3 values + arrays)
    table.insert(data, level_state.level_number)
    table.insert(data, level_state.level_type)
    table.insert(data, level_state.level_shape)

    -- Spike heights (fixed size: 16) - Absolute heights indexed 0-15 (as per user's accepted diff)
    for i = 0, 15 do
        table.insert(data, level_state.spike_heights[i] or 0)
    end

    -- Level angles (fixed size: 16) - Absolute angles
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
    table.insert(data, enemies_state.num_enemies_in_tube)
    table.insert(data, enemies_state.num_enemies_on_top)
    table.insert(data, enemies_state.enemies_pending)
    table.insert(data, enemies_state.pulsar_fliprate)

    -- Decoded Enemy Type/State Info (Size 7 each) - Absolute info
    for i = 1, 7 do
        table.insert(data, enemies_state.enemy_core_type[i] or 0)
        table.insert(data, enemies_state.enemy_direction_moving[i] or 0)
        table.insert(data, enemies_state.enemy_between_segments[i] or 0)
        table.insert(data, enemies_state.enemy_moving_away[i] or 0)
        table.insert(data, enemies_state.enemy_can_shoot[i] or 0)
        table.insert(data, enemies_state.enemy_split_behavior[i] or 0)
    end

    -- Enemy segments (fixed size: 7) - Relative segments
    for i = 1, 7 do
        table.insert(data, enemies_state.enemy_segments[i] or INVALID_SEGMENT) -- Use sentinel
    end

    -- Enemy depths (fixed size: 7 - 16bit positions) - Absolute depth
    for i = 1, 7 do
        table.insert(data, enemies_state.enemy_depths[i] or 0)
    end

    -- Enemy depths LSB (fixed size: 7 - 16bit positions) - Absolute LSB
    for i = 1, 7 do
        table.insert(data, enemies_state.enemy_depths_lsb[i] or 0)
    end

    -- Enemy shot positions (fixed size: 4) - Absolute depth
    for i = 1, 4 do
        table.insert(data, enemies_state.shot_positions[i])
    end

    -- Enemy shot positions LSB (fixed size: 4) - Absolute LSB
    for i = 1, 4 do
        table.insert(data, enemies_state.enemy_shot_lsb[i] or 0)
    end

    -- Enemy shot segments (fixed size: 4) - Relative segments
    for i = 1, 4 do
        table.insert(data, enemies_state.enemy_shot_segments[i].value or INVALID_SEGMENT) -- Use sentinel
    end

    -- Additional game state (pulse beat, pulsing)
    table.insert(data, enemies_state.pulse_beat or 0)
    table.insert(data, enemies_state.pulsing or 0)

    -- Add pending_vid (64 bytes) - Absolute info
    for i = 1, 64 do
        table.insert(data, enemies_state.pending_vid[i] or 0)
    end

    -- Add pending_seg (64 bytes) - Relative segments
    for i = 1, 64 do
        table.insert(data, enemies_state.pending_seg[i] or INVALID_SEGMENT) -- Use sentinel
    end

    -- Serialize the main data payload to a binary string. Convert all values to 16-bit integers.
    local binary_data = ""
    for i, value in ipairs(data) do
        -- Ensure value is treated as signed short for packing if negative
        local packed_value
        if type(value) == "number" and value < 0 then
             -- Convert negative numbers to their two's complement 16-bit representation
             packed_value = 0xFFFF + value + 1
        else
             -- Handle positive numbers and non-numbers (like nil, defaulting to 0)
             packed_value = (value or 0) & 0xFFFF
        end
        binary_data = binary_data .. string.pack(">H", packed_value) -- Pack as unsigned short (H)
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

    -- --- OOB Data Packing ---
    -- This section MUST remain unchanged to maintain compatibility with the Python model.
    -- It requires specific ABSOLUTE values.

    local is_attract_mode = (game_state.game_mode & 0x80) == 0
    local is_open_level = level_state.level_type == 0xFF

    -- Get the ABSOLUTE nearest enemy segment (-1 sentinel) and depth for OOB packing
    local nearest_abs_seg_oob, enemy_depth_oob = enemies_state:nearest_enemy_segment()
    local player_abs_seg_oob = player_state.position & 0x0F -- Use absolute player segment 0-15

    local is_enemy_present_oob = (nearest_abs_seg_oob ~= -1) and 1 or 0 -- Check against -1 sentinel

    -- Pack header data using 2-byte integers where possible
    local score = player_state.score or 0
    local score_high = math.floor(score / 65536)  -- High 16 bits
    local score_low = score % 65536               -- Low 16 bits

    -- Mask frame counter to 16 bits to prevent overflow
    local frame = game_state.frame_counter % 65536

    -- Ensure the format string and order match EXACTLY the previous working version.
    -- H = unsigned short (uint16)
    -- d = double
    -- B = unsigned char (uint8)
    -- h = signed short (int16)
    local oob_data = string.pack(">HdBBBHHHBBBhBhBB",
        #data,                          -- H num_values (size of main payload in 16-bit words)
        reward,                         -- d reward
        0,                              -- B game_action (placeholder)
        game_state.game_mode,           -- B game_mode
        bDone and 1 or 0,               -- B done flag
        frame,                          -- H frame_counter (16-bit)
        score_high,                     -- H score high 16 bits
        score_low,                      -- H score low 16 bits
        save_signal,                    -- B save_signal
        controls.fire_commanded,        -- B fire_commanded
        controls.zap_commanded,         -- B zap_commanded
        controls.spinner_delta,         -- h spinner_delta (model output, int8 range but packed as int16)
        is_attract_mode and 1 or 0,     -- B is_attract_mode
        nearest_abs_seg_oob,            -- h nearest_enemy_segment (ABSOLUTE, -1 sentinel, packed as int16)
        player_abs_seg_oob,             -- B player_segment (ABSOLUTE 0-15)
        is_open_level and 1 or 0        -- B is_open_level
    )
    -- --- End OOB Data Packing ---

    -- Combine out-of-band header with game state data
    local final_data = oob_data .. binary_data

    return final_data, #data  -- Return the combined data and the size of the main payload
end

-- Update the frame_callback function to track bytes sent and calculate FPS
local frame_counter = 0
local lastTimer = 0
local last_fps_time = os.time()  -- Use os.time() for wall clock time

-- Add a global tracker for timer value changes
local timer_changes = 0
local timer_check_start_time = os.clock()

-- Add a last update time tracker for precise intervals
local last_update_time = os.clock()

-- Add tracking for different frame detection methods
local last_player_position = nil
local last_timer_value = nil
local method_fps_counter = {0, 0, 0}  -- For 3 different methods
local method_start_time = os.clock()

local function frame_callback()
    -- Check the time counter at address 0x0003
    local currentTimer = mem:read_u8(0x0003)
    
    -- Check if the timer changed
    if currentTimer == lastTimer then
        return true
    end
    lastTimer = currentTimer
    
    -- Increment frame count and track FPS
    frame_count = frame_count + 1
    local current_time = os.time()
    
    -- Calculate FPS every second
    if current_time > last_fps_time then
        -- Update the FPS display only when we calculate it
        game_state.current_fps = frame_count  -- Update in GameState for display
        frame_count = 0
        last_fps_time = current_time
    end
    
    -- Update game state first
    game_state:update(mem)
    
    -- Update level state next
    level_state:update(mem)
    
    -- Update player state before enemies state
    player_state:update(mem)
    
    -- Update enemies state last to ensure all references are correct
    enemies_state:update(mem)
    
    -- Check if the game mode is in high score entry, mash AAA if it is
    if game_state.game_mode == 0x80 then
        if game_state.frame_counter % 60 == 0 then
            controls.fire_field:set_value(1)
            print("Pressing Fire")
        elseif game_state.frame_counter % 60 == 30 then
            controls.fire_field:set_value(0)
            print("Releasing Fire")
        end
        return true
    end

    -- Declare num_values at the start of the function
    local num_values = 0
    local bDone = false

    -- Check if socket is open, try to reopen if not
    if not socket then
        if not open_socket() then
            -- If socket isn't open yet, just update the display without sending data
            -- Update the display with empty controls
            update_display("Waiting for Python connection...", game_state, level_state, player_state, enemies_state, nil, num_values)
            return true
        end
    end

    -- 2 Credits
    mem:write_u8(0x0006, 2)

    -- Reset the countdown timer to zero all the time
    mem:write_u8(0x0004, 0)

    -- NOP out the jump that skips scoring in attract mode
    mem:write_direct_u8(0xCA6F, 0xEA)
    mem:write_direct_u8(0xCA70, 0xEA)
    
    -- NOP out the damage the copy protection code does to memory when it detects a bad checksum
    mem:write_direct_u8(0xA591, 0xEA)
    mem:write_direct_u8(0xA592, 0xEA)

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
        if AUTO_PLAY_MODE then
            local port = manager.machine.ioport.ports[":IN2"]
            -- Press P1 Start in MAME with proper press/release simulation
            if game_state.frame_counter % 60 == 0 then
                -- Try different possible field names
                local startField = port.fields["1 Player Start"] or 
                                   port.fields["P1 Start"] or 
                                   port.fields["Start 1"]
                
                if startField then
                    -- Press the button
                    startField:set_value(1)
                    print("Pressing 1 Player Start button in attract mode")
                else
                    print("Error: Could not find start button field")
                end
            elseif game_state.frame_counter % 60 == 5 then
                -- Release the button 5 frames later
                local startField = port.fields["1 Player Start"] or 
                                   port.fields["P1 Start"] or 
                                   port.fields["Start 1"]
                
                if startField then
                    startField:set_value(0)
                    print("Releasing 1 Player Start button")
                end
            end
        end
    else
        -- Four lives at all times
        -- mem:write_direct_u8(0x0048, 0x04)
    end

    -- Calculate the reward for the current frame - do this ONCE per frame
    local reward, bDone = calculate_reward(game_state, level_state, player_state, enemies_state)

    -- NOP out the jump that skips scoring in attract mode
    mem:write_direct_u8(0xCA6F, 0xEA)
    mem:write_direct_u8(0xCA70, 0xEA)

    -- NOP out the clearing of zap_fire_new
    mem:write_direct_u8(0x976E, 0x00)
    mem:write_direct_u8(0x976F, 0x00)
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
    frame_data, num_values = flatten_game_state_to_binary(reward, game_state, level_state, player_state, enemies_state, bDone)

    -- Send the serialized data to the Python script and get the components
    local fire, zap, spinner = process_frame(frame_data, player_state, controls, reward, bDone, is_attract_mode)

    -- BOT: Calculate spinner value based on direction to nearest enemy, it will play like demo mode
    -- Calculate spinner value based on direction to nearest enemy, override what the model said above
    -- spinner = 9 * direction_to_nearest_enemy(game_state, level_state, player_state, enemies_state)


    player_state.fire_commanded = fire
    player_state.zap_commanded = zap
    player_state.SpinnerDelta = spinner

    -- Update total bytes sent
    total_bytes_sent = total_bytes_sent + #frame_data


    -- In LOG_ONLY_MODE, limit display updates to 10 times per second
    local current_time_high_res = os.clock()
    local should_update_display = not LOG_ONLY_MODE or 
                                    (current_time_high_res - last_display_update) >= DISPLAY_UPDATE_INTERVAL

    if should_update_display and SHOW_DISPLAY then
        -- Update the display with the current action and metrics
        update_display(status_message, game_state, level_state, player_state, enemies_state, action, num_values, reward)
        last_display_update = current_time_high_res
    end

    -- We only control the game in regular play mode (04), zooming down the tube (20), AND High Score Entry (24)
    if game_state.gamestate == 0x04 or game_state.gamestate == 0x20 or game_state.gamestate == 0x24 then -- Added 0x24
        -- Apply the action determined by the AI (Python script) to MAME controls
        controls:apply_action(fire, zap, spinner, game_state, player_state)
    elseif game_state.gamestate == 0x12 then -- This state might need verification too
        -- Apply the action to MAME controls (Seems like hardcoded zap?)
        controls:apply_action(0, frame_count % 2, 0, game_state, player_state)
    end

    return true
end

-- Helper function to format segment values for display
local function format_segment(value)
    if value == INVALID_SEGMENT then
        return "---"
    else
        -- Use %+03d: this ensures a sign and pads with 0 to a width of 2 digits (total 3 chars like +01, -07)
        return string.format("%+03d", value)
    end
end

-- Update the update_display function to show total bytes sent and FPS
function update_display(status, game_state, level_state, player_state, enemies_state, current_action, num_values, reward)
    clear_screen()
    move_cursor_to_row(1)

    -- Format and print game state in 3 columns at row 1
    print("--[ Game State ]--------------------------------------")
    
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
        {"FPS", string.format("%.2f", game_state.current_fps)},
        {"Payload Size", num_values},
        {"Last Reward", string.format("%.2f", LastRewardState)},
        {"Log Only Mode", LOG_ONLY_MODE and "ENABLED" or "OFF"}
    }
    
    -- Calculate how many rows we need (ceiling of items/3)
    local rows = math.ceil(#game_metrics / 3)
    
    for row = 1, rows do
        local line = "  "
        for col = 1, 3 do
            local idx = (row - 1) * 3 + col
            if idx <= #game_metrics then
                local key, value = table.unpack(game_metrics[idx])
                -- Format each metric with fixed width to fit in 80 columns
                line = line .. string.format("%-14s: %-10s", key, tostring(value))
            end
        end
        print(line)
    end
    print("")  -- Empty line after section

    -- Format and print player state
    print("--[ Player State ]------------------------------------")
    
    -- Create player metrics in a more organized way for 3-column display
    local player_metrics = {
        {"Position", string.format("%d", player_state.position)},
        {"State", string.format("0x%02X", player_state.player_state)},
        {"Depth", string.format("%d", player_state.player_depth)},
        {"Alive", string.format("%d", player_state.alive)},
        {"Score", string.format("%d", player_state.score)},
        {"Szapper Uses", string.format("%d", player_state.superzapper_uses)},
        {"Szapper Active", string.format("%d", player_state.superzapper_active)},
        {"Shot Count", string.format("%d", player_state.shot_count)},
        {"Debounce", string.format("%d", player_state.debounce)},
        {"Fire Detected", string.format("%d", player_state.fire_detected)},
        {"Zap Detected", string.format("%d", player_state.zap_detected)},
        {"SpinnerAccum", string.format("%d", player_state.SpinnerAccum)},
        {"SpinnerDelta", string.format("%d", player_state.SpinnerDelta)},
        {"InferredDelta", string.format("%d", player_state.inferredSpinnerDelta)}
    }
    
    -- Calculate how many rows we need (ceiling of items/3)
    local rows = math.ceil(#player_metrics / 3)
    
    for row = 1, rows do
        local line = "  "
        for col = 1, 3 do
            local idx = (row - 1) * 3 + col
            if idx <= #player_metrics then
                local key, value = table.unpack(player_metrics[idx])
                -- Format each metric with fixed width to fit in 80 columns
                line = line .. string.format("%-14s: %-10s", key, value)
            end
        end
        print(line)
    end
    
    -- Add shot positions on its own line
    local shots_str = ""
    for i = 1, 8 do
        shots_str = shots_str .. string.format(" %02X ", player_state.shot_positions[i])
    end
    print("  Shot Positions: " .. shots_str)
    
    -- Display player shot segments as a separate line
    local player_shot_segments_str = ""
    for i = 1, 8 do
        player_shot_segments_str = player_shot_segments_str .. format_segment(player_state.shot_segments[i]) .. " "
    end
    print("  Shot Segments : " .. player_shot_segments_str)
    
    print("")  -- Empty line after section

    -- Format and print player controls
    print("--[ Player Controls (Game Inferred Values) ]----------")
    local is_attract_mode = (game_state.game_mode & 0x80) == 0
    print(string.format("  %-25s: %d", "Fire (Inferred)", controls.fire_commanded))
    print(string.format("  %-25s: %d", "Superzapper (Inferred)", controls.zap_commanded))
    print(string.format("  %-25s: %d", "Spinner Delta (Inferred)", controls.spinner_delta))
    print(string.format("  %-25s: %s", "Attract Mode", is_attract_mode and "Active" or "Inactive"))
    print("")

    -- Add Model State section
    print("--[ Model Output ]------------------------------------")
    print(string.format("  %-25s: %d", "Model Fire", player_state.fire_commanded))
    print(string.format("  %-25s: %d", "Model Zap", player_state.zap_commanded))
    print(string.format("  %-25s: %d", "Model Spinner", player_state.SpinnerDelta))
    print("")

    -- Format and print level state
    print("--[ Level State ]-------------------------------------")
    print(string.format("  %-14s: %-10s  %-14s: %-10s  %-14s: %s",
        "Level Number", level_state.level_number,
        "Level Type", level_state.level_type == 0xFF and "Open" or "Closed",
        "Level Shape", level_state.level_shape))
    
    -- Add spike heights on its own line
    local heights_str = ""
    for i = 0, 15 do
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
    print("")

    -- Format and print enemies state at row 31
    move_cursor_to_row(31)
    local enemy_types = {}
    local enemy_states = {}
    local enemy_segs = {}
    local enemy_depths = {}
    local enemy_lsbs = {}
    local enemy_shot_lsbs = {}  -- New array for shot LSBs
    for i = 1, 7 do
        enemy_types[i] = enemies_state:decode_enemy_type(enemies_state.enemy_type_info[i])
        enemy_states[i] = enemies_state:decode_enemy_state(enemies_state.active_enemy_info[i])
        -- Format enemy segments using the helper function
        enemy_segs[i] = format_segment(enemies_state.enemy_segments[i])
        -- Display enemy depths as 2-digit hex values
        enemy_depths[i] = string.format("%02X", enemies_state.enemy_depths[i])
        -- Display enemy LSBs as 2-digit hex values
        enemy_lsbs[i] = string.format("%02X", enemies_state.enemy_depths_lsb[i])
        -- Display enemy shot LSBs as 2-digit hex values
        enemy_shot_lsbs[i] = string.format("%02X", enemies_state.enemy_shot_lsb[i])
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
        ["Flip Rate"] = string.format("%02X", enemies_state.pulsar_fliprate), -- NEW: Display Pulsar Flip Rate
        ["In Tube"] = string.format("%d enemies", enemies_state.num_enemies_in_tube),
        -- Display relative nearest enemy segment using format_segment
        ["Nearest Enemy"] = string.format("segment %s", format_segment(enemies_state.nearest_enemy_seg)),
        ["On Top"] = string.format("%d enemies", enemies_state.num_enemies_on_top),
        ["Pending"] = string.format("%d enemies", enemies_state.enemies_pending),
        ["Enemy Types"] = table.concat(enemy_types, " "),
        ["Enemy States"] = table.concat(enemy_states, " ")
    }

    print(format_section("Enemies State", enemies_metrics))

    -- Add enemy segments and depths on their own lines
    print("  Enemy Segments: " .. table.concat(enemy_segs, " "))
    print("  Enemy Depths  : " .. table.concat(enemy_depths, " "))
    print("  Enemy LSBs    : " .. table.concat(enemy_lsbs, " "))
    print("  Shot LSBs     : " .. table.concat(enemy_shot_lsbs, " "))
    
    -- NEW: Display decoded enemy info tables
    local enemy_core_types_str = table.concat(enemies_state.enemy_core_type, " ")
    local enemy_dir_mov_str = table.concat(enemies_state.enemy_direction_moving, " ")
    local enemy_between_str = table.concat(enemies_state.enemy_between_segments, " ")
    local enemy_mov_away_str = table.concat(enemies_state.enemy_moving_away, " ")
    local enemy_can_shoot_str = table.concat(enemies_state.enemy_can_shoot, " ")
    local enemy_split_str = table.concat(enemies_state.enemy_split_behavior, " ")
    
    print("  Enemy Core Type: " .. enemy_core_types_str)
    print("  Enemy Dir Mov  : " .. enemy_dir_mov_str)
    print("  Enemy Between  : " .. enemy_between_str)
    print("  Enemy Mov Away : " .. enemy_mov_away_str)
    print("  Enemy Can Shoot: " .. enemy_can_shoot_str)
    print("  Enemy Split Bhv: " .. enemy_split_str)
    
    -- Add enemy shot positions in a simple fixed format
    local shot_positions_str = ""
    for i = 1, 4 do
        local pos_value = enemies_state.shot_positions[i] or 0
        -- Mask to 8 bits (0-255) to display only one byte
        pos_value = pos_value & 0xFF
        shot_positions_str = shot_positions_str .. string.format(" %02X ", pos_value)
    end
    print("  Shot Positions: " .. shot_positions_str)
    
    -- Display enemy shot segments using format_segment
    local shot_segments_str = ""
    for i = 1, 4 do
        shot_segments_str = shot_segments_str .. format_segment(enemies_state.enemy_shot_segments[i].value) .. " "
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
        pending_seg_str = pending_seg_str .. format_segment(enemies_state.pending_seg[i]) .. " "
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
        local reward = calculate_reward(game_state, level_state, player_state, enemies_state)
        
        -- Get final frame data with save signal
        local frame_data, num_values = flatten_game_state_to_binary(reward, game_state, level_state, player_state, enemies_state, true)
        
        -- Send one last time
        if socket then
            local result = process_frame(frame_data, player_state, controls, reward, true)
            print("Final save complete, response: " .. (result or "none"))
        end
    end
    
    -- Close socket
    if socket then socket:close(); socket = nil end
    print("Socket closed during MAME shutdown")
    
    -- Close log file
    if log_file then
        log_file:close()
        log_file = nil
        print("Log file closed during MAME shutdown")
    end
end

-- Start the Python script but don't wait for socket to open
start_python_script()
clear_screen()

-- Register the frame callback with MAME.  Keep a reference to the callback or it will be garbage collected
callback_ref = emu.add_machine_frame_notifier(frame_callback)

-- Register the shutdown callback
emu.register_stop(on_mame_exit)

-- Function to get the relative distance to a target segment
function absolute_to_relative_segment(current_abs_segment, target_abs_segment, is_open_level)
    -- Mask inputs to ensure they are within 0-15 range
    current_abs_segment = current_abs_segment & 0x0F
    target_abs_segment = target_abs_segment & 0x0F
    
    -- Check if target segment is valid (e.g., not 0 if 0 represents inactive)
    -- Note: Caller should handle cases where target_abs_segment might represent an invalid state before calling.
    -- We assume 0-15 are potentially valid target segments here.

    -- Get segment distance based on level type
    if is_open_level then
        -- Open level: simple distance calculation (-15 to +15)
        return target_abs_segment - current_abs_segment
    else
        -- Closed level: find shortest path around the circle (-7 to +8)
        -- Note: The range is -7 to +8, not -8 to +7, because a distance of 8 can be reached
        -- clockwise or counter-clockwise, but we need a convention. We'll return +8.
        local diff = target_abs_segment - current_abs_segment
        if diff > 8 then
            return diff - 16 -- Wrap around (e.g., 1 -> 15 is -2)
        elseif diff <= -8 then
            return diff + 16 -- Wrap around (e.g., 15 -> 1 is +2)
        else
            return diff
        end
    end
end
