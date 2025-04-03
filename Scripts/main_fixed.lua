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
local pipe_out = nil

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
        local result = socket:open("socket.127.0.0.1:9999")
        
        if result == nil then
            print("Successfully opened socket connection to localhost:9999")
            
            -- Send initial 4-byte ping for handshake
            local ping_data = string.pack(">I", 0)  -- 4-byte integer with value 0
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
--
-- Return reward and bDone, where bDone is true if the episode is done

local function calculate_reward(game_state, level_state, player_state, enemies_state)
    
    local reward = 0
    local bDone = false             -- Track if the episode (life) is done

    -- Survival reward
    if player_state.alive == 1 then

        reward = reward + 1

        -- Score reward
        local score_delta = player_state.score - previous_score
        if score_delta > 0 then
            reward = reward + score_delta
        end

        -- Level completion reward
        if level_state.level_number ~= previous_level then
            reward = reward + (100 * previous_level)
        end

        -- UPDATED: Enemy positioning reward based on the demoplay logic
        -- Find the enemy segment closest to the top of the tube
        local target_segment = enemies_state:nearest_enemy_segment()
        
        -- Only proceed with target-based rewards if there's at least one enemy
        if target_segment >= 0 then
            local player_segment = player_state.position & 0x0F  -- Apply mask to ensure 0-15 range
            local is_open = level_state.level_type == 0xFF
            
            -- Calculate the direction and distance to the nearest enemy
            local clockwise_distance = 0
            local counterclockwise_distance = 0
            
            if target_segment > player_segment then
                clockwise_distance = target_segment - player_segment
                counterclockwise_distance = player_segment + (16 - target_segment)
            else
                clockwise_distance = target_segment + (16 - player_segment)
                counterclockwise_distance = player_segment - target_segment
            end
            
            -- Find the minimum distance considering wraparound
            local min_distance = math.min(clockwise_distance, counterclockwise_distance)
            
            -- Larger reward for being directly lined up with an enemy
            if min_distance == 0 then
                -- Extra reward if player is not moving while lined up (ready to fire)
                if player_state.SpinnerDelta == 0 then
                    reward = reward + 15  -- Significant reward for being lined up and still
                else
                    reward = reward + 10   -- Good reward just for being lined up
                end
            else
                -- Give decreasing rewards as distance increases
                -- The demoplay routine wants to minimize the distance to the closest enemy
                local alignment_reward = 10 - min_distance  -- Up to 9 points for being 1 segment away, down to 0 for 10+ segments away
                if alignment_reward > 0 then
                    reward = reward + alignment_reward
                end
                
                -- Small bonus for moving in the correct direction toward the target
                local move_direction = direction_to_nearest_enemy(game_state, level_state, player_state, enemies_state)
                if move_direction ~= 0 then
                    -- Check if player is moving in the direction that will reach the target faster
                    -- direction < 0 means go clockwise (need negative spinner)
                    -- direction > 0 means go counterclockwise (need positive spinner)
                    if (move_direction < 0 and player_state.SpinnerDelta < 0) or 
                       (move_direction > 0 and player_state.SpinnerDelta > 0) then
                        reward = reward + 5  -- Small bonus for moving the right way
                    elseif player_state.SpinnerDelta ~= 0 then
                        reward = reward - 1  -- Small penalty for moving the wrong way
                    end
                end

                -- If there are no enemies, or if we're zooming down the tube, don't move around

                if (target_segment < 0 or game_state.gamestate == 0x20) then
                    if (player_state.SpinnerDelta == 0) then
                        reward = reward + 10
                    else
                        reward = reward - math.abs(player_state.SpinnerDelta)
                    end
                end
            end
        end
    else
        if previous_alive_state == 1 then
            reward = reward - 1000
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
local function process_frame(params, player_state, controls, reward, bDone, bAttractMode)
    -- In log-only mode, we only write to the log file and don't communicate with Python

    -- Log the frame data to file
    if log_file then
        local success, err = pcall(function()
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
        open_socket()
    end
    
    -- Increment frame count after processing
    frame_count = frame_count + 1

    if LOG_ONLY_MODE then
        return controls.zap_commanded, controls.fire_commanded, controls.spinner_delta
    end
    
    -- Check if socket is open, try to reopen if not
    if not socket then
        if not open_socket() then
            return 0, 0, 0  -- Return zeros for fire, zap, spinner_delta on socket error
        end
    end
  
    -- Try to write to socket, handle errors
    local success, err = pcall(function()
        -- Add 4-byte length header to params
        local data_length = #params
        local length_header = string.pack(">I", data_length)
        
        -- Write length header followed by data
        socket:write(length_header)
        socket:write(params)
    end)
    
    if not success then
        print("Error writing to socket:", err)
        -- Close and attempt to reopen socket
        if socket then socket:close(); socket = nil end
        open_socket()
        return 0, 0, 0  -- Return zeros for fire, zap, spinner_delta on write error
    end
    
    -- Try to read from socket, handle errors
    local fire, zap, spinner = 0, 0, 0  -- Default values
    success, err = pcall(function()
        -- Read exactly 3 bytes for the three i8 values
        local action_bytes = socket:read(3)
        
        if action_bytes and #action_bytes == 3 then
            -- Unpack the three signed 8-bit integers
            fire, zap, spinner = string.unpack("bbb", action_bytes)
        else
            -- Default action if read fails
            print("Failed to read action from socket, got " .. (action_bytes and #action_bytes or 0) .. " bytes")
            fire, zap, spinner = 0, 0, 0
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
        
        -- Read segment and store absolute position
        local abs_segment = mem:read_u8(0x02AD + i - 1)       -- PlayerShotSegments
        if self.shot_positions[i] == 0 or abs_segment == 0 then
            self.shot_segments[i] = 0  -- Shot not active
        else
            abs_segment = abs_segment & 0x0F  -- Mask to get valid segment
            self.shot_segments[i] = abs_segment
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
    self.num_enemies_in_tube = 0
    self.num_enemies_on_top = 0
    self.enemies_pending = 0
    self.nearest_enemy_seg = -1  -- Track the segment of the nearest enemy
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
    self.enemy_depths_lsb = {0, 0, 0, 0, 0, 0, 0}   -- 7 enemy depth positions LSB
    self.enemy_shot_lsb = {0, 0, 0, 0, 0, 0, 0}     -- 7 enemy shot LSB values at $02E6

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
    -- First, initialize/reset all arrays at the beginning
    -- Reset enemy arrays
    self.enemy_type_info = {0, 0, 0, 0, 0, 0, 0}    -- 7 enemy type slots
    self.active_enemy_info = {0, 0, 0, 0, 0, 0, 0}  -- 7 active enemy slots
    self.enemy_segments = {0, 0, 0, 0, 0, 0, 0}     -- 7 enemy segment numbers
    self.enemy_depths = {0, 0, 0, 0, 0, 0, 0}       -- 7 enemy depth positions
    self.enemy_depths_lsb = {0, 0, 0, 0, 0, 0, 0}   -- 7 enemy depth positions LSB
    self.enemy_shot_lsb = {0, 0, 0, 0, 0, 0, 0}     -- Reset enemy shot LSB values
    self.enemy_move_vectors = {0, 0, 0, 0, 0, 0, 0} -- Movement vectors
    self.enemy_state_flags = {0, 0, 0, 0, 0, 0, 0}  -- State flags
    
    -- Read active enemies (currently on screen)
    self.active_flippers  = mem:read_u8(0x0142)   -- n_flippers - current active count
    self.active_pulsars   = mem:read_u8(0x0143)    -- n_pulsars
    self.active_tankers   = mem:read_u8(0x0144)    -- n_tankers
    self.active_spikers   = mem:read_u8(0x0145)    -- n_spikers
    self.active_fuseballs = mem:read_u8(0x0146)  -- n_fuseballs
    self.pulse_beat       = mem:read_u8(0x0147)        -- pulse_beat counter
    self.pulsing          = mem:read_u8(0x0148)          -- pulsing state
    self.num_enemies_in_tube = mem:read_u8(0x0108)
    self.num_enemies_on_top = mem:read_u8(0x0109)
    self.enemies_pending = mem:read_u8(0x03AB)

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

    local activeEnemies = self.num_enemies_in_tube + self.num_enemies_on_top
    
    -- Read enemy type and state info
    for i = 1, 7 do
        -- Get raw values from memory
        self.enemy_type_info[i] = mem:read_u8(0x0283 + i - 1)    -- Enemy type info at $0283
        self.active_enemy_info[i] = mem:read_u8(0x028A + i - 1)  -- Active enemy info at $028A
        self.enemy_segments[i] = mem:read_u8(0x02B9 + i - 1) & 0x0F
        
        -- Also read state flags and movement vectors here
        local state_flags = mem:read_u8(0x0275 + i - 1)
        self.enemy_state_flags[i] = state_flags
        
        -- Read movement vector and handle two's complement
        local move_vector = mem:read_u8(0x0291 + i - 1)
        if move_vector > 127 then
            move_vector = move_vector - 256  -- Convert to signed
        end
        self.enemy_move_vectors[i] = move_vector
        
        -- Check if enemy is active using multiple indicators
        local is_active = false
        
        -- Main activity test: Active info bit 7 is clear (0) AND either type or depth is non-zero
        if (self.active_enemy_info[i] & 0x80) == 0 then
            local raw_depth = mem:read_u8(0x02DF + i - 1)
            if raw_depth > 0 or (self.enemy_type_info[i] & 0x07) > 0 then
                is_active = true
            end
        end
        
        -- State flags can confirm activity - bit 1 (0x02) indicates movement
        if (state_flags & 0x02) ~= 0 then
            is_active = true
        end
        
        -- For pulsars specifically, look at the pulse state
        if (self.enemy_type_info[i] & 0x07) == 2 and self.pulsing > 0 then
            is_active = true
        end
        
        -- Only populate data if the enemy is active
        if is_active then
            -- Get main position value and LSB for more precision
            local pos = mem:read_u8(0x02DF + i - 1)       -- enemy_along at $02DF - main position
            local lsb = mem:read_u8(0x029F + i - 1)       -- enemy_along_lsb at $029F - fractional part
            
            -- Store values from memory
            self.enemy_depths[i] = pos
            self.enemy_depths_lsb[i] = lsb
            
            -- Track shot LSB
            self.enemy_shot_lsb[i] = mem:read_u8(0x02E6 + i - 1)
        else
            -- Zero out ALL values for inactive enemies to avoid stale data
            self.enemy_depths[i] = 0
            self.enemy_depths_lsb[i] = 0
            self.enemy_segments[i] = 0  -- Important: Zero out segment for inactive enemies
            self.enemy_move_vectors[i] = 0
            self.enemy_state_flags[i] = 0
            self.enemy_shot_lsb[i] = 0
        end
    end
    
    -- Read all 4 enemy shot positions and store absolute positions
    for i = 1, 4 do
        local raw_pos = mem:read_u8(0x02DB + i - 1)
        if raw_pos == 0 then
            self.shot_positions[i] = nil  -- Mark inactive shots as nil
        else
            local abs_pos = raw_pos & 0x0F
            self.shot_positions[i] = abs_pos
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

    -- Scan the display list region for additional enemy data
    self.display_list = {}
    for i = 0, 31 do  -- Just scan part of it for efficiency
        self.display_list[i] = {
            command = mem:read_u8(0x0300 + i * 4),
            segment = mem:read_u8(0x0301 + i * 4) & 0x0F,
            depth = mem:read_u8(0x0302 + i * 4),
            type = mem:read_u8(0x0303 + i * 4)
        }
    end
    
    -- Update nearest enemy segment
    self.nearest_enemy_seg = self:nearest_enemy_segment()
end

-- Find the segment with the enemy closest to the top of the tube
function EnemiesState:nearest_enemy_segment()
    local min_depth = 255  -- Initialize with maximum possible depth
    local closest_segment = -1  -- Initialize with invalid segment
    
    -- Check the standard enemy table for non-zero depths
    for i = 1, 7 do
        -- Only consider enemies with a valid depth and segment
        if self.enemy_depths[i] > 0 and 
           self.enemy_segments[i] >= 0 and self.enemy_segments[i] <= 15 then
            if self.enemy_depths[i] < min_depth then
                min_depth = self.enemy_depths[i]
                closest_segment = self.enemy_segments[i]
            end
        end
    end
    
    return closest_segment  -- Returns the segment number or -1 if no enemies found
end

-- Function to calculate the direction to the nearest enemy segment
function direction_to_nearest_enemy(game_state, level_state, player_state, enemies_state)
    local target_segment = enemies_state:nearest_enemy_segment()
    if target_segment < 0 then return 0 end -- No enemy

    local player_segment = player_state.position
    local delta = target_segment - player_segment
    if delta == 0 then return 0 end -- Already aligned

    if level_state.level_type == 0xFF then -- Open Level
        -- Open Level Logic: Return negative delta to match spinner direction needed
        return -delta 
    else -- Closed Level
        -- Closed Level Logic: Calculate shortest signed distance with wraparound
        -- Result is in range [-8, 7], negative for CW, positive for CCW
        return (delta + 8) % 16 - 8
    end
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
    table.insert(data, game_state.credits)
    table.insert(data, game_state.p1_lives)
    table.insert(data, game_state.p1_level)
    
    -- Add nearest enemy segment and segment delta
    local nearest_enemy_seg = enemies_state.nearest_enemy_seg
    local player_segment = player_state.position & 0x0F
    local segment_delta = 0
    
    -- Only calculate delta if we have a valid nearest enemy
    if nearest_enemy_seg >= 0 then
        -- Calculate clockwise distance to nearest enemy
        if nearest_enemy_seg > player_segment then
            segment_delta = nearest_enemy_seg - player_segment
        else
            segment_delta = nearest_enemy_seg + (16 - player_segment)
        end
        -- Convert to shortest path (clockwise or counterclockwise)
        if segment_delta > 8 then
            segment_delta = -(16 - segment_delta)
        end
    end
    
    table.insert(data, nearest_enemy_seg)  -- -1 if no enemy
    table.insert(data, segment_delta)      -- Shortest path delta (-8 to +8)
    
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
    for i = 0, 15 do  
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
    table.insert(data, enemies_state.num_enemies_in_tube)
    table.insert(data, enemies_state.num_enemies_on_top)
    table.insert(data, enemies_state.enemies_pending)
    
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
    
    -- Enemy depths (fixed size: 7 - 16bit positions)
    for i = 1, 7 do
        table.insert(data, enemies_state.enemy_depths[i] or 0)
    end

    -- Enemy depths (fixed size: 7 - 16bit positions)
    for i = 1, 7 do
        table.insert(data, enemies_state.enemy_depths_lsb[i] or 0)
    end
    
    -- Enemy shot positions (fixed size: 4)
    for i = 1, 4 do
        table.insert(data, enemies_state.shot_positions[i] or 0)
    end

    -- Enemy shot positions (fixed size: 4)
    for i = 1, 4 do
        table.insert(data, enemies_state.enemy_shot_lsb[i] or 0)
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
--        table.insert(data, enemies_state.pending_vid[i] or 0)
    end
    
    -- Add pending_seg (64 bytes)
    for i = 1, 64 do
--        table.insert(data, enemies_state.pending_seg[i] or 0)
    end
    
    -- Serialize the data to a binary string.  We will convert all values to 16-bit signed integers
    -- and then pack them into a binary string.
    
    local binary_data = ""
    for i, value in ipairs(data) do
       
        encoded_value = i & 0xFFFF
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
    
    local is_attract_mode = (game_state.game_mode & 0x80) == 0
    is_attract_mode = is_attract_mode and 1 or 0
    
    -- Determine if this is an open level (FF) or closed level (00)
    local is_open_level = level_state.level_type == 0xFF
    is_open_level = is_open_level and 1 or 0

    -- Create out-of-band context information structure
    -- Pack: num_values (uint32), reward (double), game_action (byte), game_mode (byte), 
    -- done flag (byte), frame_counter (uint32), score (uint32), save_signal (byte),
    -- fire_commanded (byte), zap_commanded (byte), spinner_delta (int8),
    -- is_attract (byte), nearest_enemy_segment (byte), player_segment (byte),
    -- is_open_level (byte)

    local oob_data = string.pack(">IdBBBIIBBBhBhBB", 
        #data,                          -- I num_values
        reward,                         -- d reward
        0,                              -- B game_action
        game_state.game_mode,           -- B game_mode
        bDone and 1 or 0,               -- B done flag
        game_state.frame_counter,       -- I frame_counter
        player_state.score,             -- I score
        save_signal,                    -- B save_signal
        controls.fire_commanded,        -- B fire_commanded (added)
        controls.zap_commanded,         -- B zap_commanded (added)
        controls.spinner_delta,         -- h spinner_delta (added)
        is_attract_mode,                -- B is_attract_mode
        enemies_state:nearest_enemy_segment(), -- h nearest_enemy_segment
        player_state.position,          -- B player segment
        is_open_level                   -- B is_open_level (added)
    )
    
    -- Combine out-of-band header with game state data
    local final_data = oob_data .. binary_data
    
    return final_data, #data  -- Return the number of 16-bit integers
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
        shots_str = shots_str .. string.format("%02X ", player_state.shot_positions[i])
    end
    print("  Shot Positions: " .. shots_str)
    
    -- Display player shot segments as a separate line
    local player_shot_segments_str = ""
    for i = 1, 8 do
        player_shot_segments_str = player_shot_segments_str .. string.format("%02X ", player_state.shot_segments[i])
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
        -- Format enemy segments as two-digit hexadecimal numbers
        enemy_segs[i] = string.format("%02X", enemies_state.enemy_segments[i])
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
        ["In Tube"] = string.format("%d enemies", enemies_state.num_enemies_in_tube),
        ["Nearest Enemy"] = string.format("segment %d", enemies_state.nearest_enemy_seg),
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
