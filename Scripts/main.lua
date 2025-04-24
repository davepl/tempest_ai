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
package.path                  = package.path .. ";/Users/dave/source/repos/tempest/Scripts/?.lua"
-- Now require the module by name only (without path or extension)

-- Define constants
local INVALID_SEGMENT         = -32768 -- Used as sentinel value for invalid segments

local SHOW_DISPLAY            = true
local DISPLAY_UPDATE_INTERVAL = 0.02

-- Access the main CPU and memory space
local mainCpu                 = nil
local mem                     = nil

local function clear_screen()
    io.write("\027[2J\027[H")
end

-- Function to move the cursor to the home position using ANSI escape codes
local function move_cursor_home()
    io.write("\027[H")
end

-- Global socket variable
local socket = nil

-- Add this near the top of the file with other global variables
local frame_count = 0 -- Initialize frame counter

-- Global variables for tracking bytes sent and FPS
local total_bytes_sent = 0

-- Add near the top with other global variables
local level_select_counter = 0

-- Function to open socket connection
local function open_socket()
    -- Try to open socket connection
    local socket_success, err = pcall(function()
        -- Close existing socket if any
        if socket then
            socket:close()
            socket = nil
        end

        -- Create a new socket connection
        socket = emu.file("rw") -- "rw" mode for read/write
        local result = socket:open("socket.m2macpro.local:9999")

        if result == nil then
            print("Successfully opened socket connection to localhost:9999")

            -- Send initial 4-byte ping for handshake
            local ping_data = string.pack(">H", 0) -- 2-byte integer with value 0
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


    return true
end


-- Declare global variables for reward calculation
local previous_score = 0
local previous_level = 0
local previous_alive_state = 1 -- Track previous alive state, initialize as alive

-- Declare a global variable to store the last reward state
local LastRewardState = 0

local shutdown_requested = false

local last_display_update = 0 -- Timestamp of last display update


-- NEW Function: Find the topmost entity (enemy or shot) in a specific absolute segment
-- Returns: depth (0-255) and type (0-7 for enemies, 8 for shots, -1 if lane clear)
function top_enemy_in_segment(target_abs_segment, level_state, enemies_state)
    -- Determine level type internally
    local is_open_level = (level_state.level_type == 0xFF)

    -- Validate segment based on level type BEFORE masking/wrapping
    if is_open_level then
        if target_abs_segment < 0 or target_abs_segment > 15 then
            return 255, 0
        end
    end

    -- Ensure target segment is valid (handles wrapping for closed levels)
    target_abs_segment = target_abs_segment & 0x0F

    -- 1. Check Enemy Shots first (they're highest priority)
    local min_depth = 256 -- Initialize with a value greater than max depth
    local found_type = -1 -- Sentinel for 'nothing found'

    for i = 1, 4 do
        -- Use pre-calculated absolute segment and depth
        local shot_abs_segment = enemies_state.enemy_shot_abs_segments[i]
        local shot_depth = enemies_state.shot_positions[i]

        -- Consider only active shots (segment ~= INVALID) in the target segment
        if shot_abs_segment ~= INVALID_SEGMENT and shot_abs_segment == target_abs_segment then
            min_depth = shot_depth
            found_type = 8 -- Assign type 8 for enemy shots
        end
    end

    -- 2. Then check Enemies (Slots 1-7)
    for i = 1, 7 do
        -- Use pre-calculated absolute segment and depth
        local enemy_abs_segment = enemies_state.enemy_abs_segments[i]
        local enemy_depth = enemies_state.enemy_depths[i]
        local enemy_type = enemies_state.enemy_core_type[i]

        -- Consider all active enemies in the target segment
        if enemy_abs_segment ~= INVALID_SEGMENT and enemy_abs_segment == target_abs_segment then
            -- If we haven't found anything yet, or if this enemy is closer
            if found_type == -1 or enemy_depth < min_depth then
                min_depth = enemy_depth
                found_type = enemy_type
            end
        end
    end

    -- 3. Return results
    if min_depth > 255 then -- If min_depth wasn't updated, nothing was found
        return 255, 0
    else
        return min_depth, found_type
    end
end

-- Function to calculate reward for the current frame
local function calculate_reward(game_state, level_state, player_state, enemies_state, commanded_spinner)
    local reward = 0
    local bDone = false
    local should_fire = false

    -- Base survival reward - make staying alive more valuable
    if player_state.alive == 1 then
        -- Score-based reward (keep this as a strong motivator).  Filter out large bonus awards.
        local score_delta = player_state.score - previous_score
        if score_delta > 0 and score_delta <= 1000 then
            reward = reward + (score_delta)
        end

        -- Encourage maintaining shots in reserve.  Penalize 0 or 8, graduated reward for 1-7
        local sc = player_state.shot_count
        if sc == 0 then
            reward = reward - 50
        elseif sc == 4 then
            reward = reward + 5
        elseif sc == 5 then
            reward = reward + 10
        elseif sc == 6 then
            reward = reward + 15
        elseif sc == 7 then
            reward = reward + 20
        elseif sc >= 8 then -- Max shots is 8, handle this case
            reward = reward - 50
        end

        -- Penalize using superzapper; only in play mode, since it's also set during zoom (0x020)
        if (game_state.gamestate == 0x04) then
            if (player_state.superzapper_active ~= 0) then
                reward = reward - 500
            end
        end

        -- Enemy targeting logic
        local target_segment, target_depth, target_should_fire = enemies_state:target_segment(game_state, player_state,
            level_state)
        should_fire = target_should_fire -- Capture the firing recommendation
        local player_segment = player_state.position & 0x0F

        -- Tube Zoom logic adjustment
        if game_state.gamestate == 0x20 then -- In tube zoom
            -- Reward based on inverse spike height (higher reward for shorter spikes)
            local spike_h = level_state.spike_heights[player_segment] or 0
            if spike_h > 0 then
                local effective_spike_length = 255 - spike_h                       -- Shorter spike = higher value
                -- Scale reward: Max 150 for no spike (height 0 -> length 255), less for longer spikes
                reward = reward + math.max(0, (effective_spike_length / 2) - 27.5) -- Scaled reward
            else
                reward = reward + (commanded_spinner == 0 and 250 or -50)          -- Max reward if no spike
            end

            if (player_state.fire_commanded == 1) then
                reward = reward + 200
            end
        elseif target_segment < 0 then
            -- No enemies: reward staying still more strongly
            -- Use commanded_spinner here to check if model *intended* to stay still
            reward = reward + (commanded_spinner == 0 and 150 or -20)
        else
            -- Get desired spinner direction, segment distance, AND enemy depth
            local desired_spinner, segment_distance, enemy_depth = direction_to_nearest_enemy(game_state, level_state,
                player_state, target_segment)

            -- Check alignment based on actual segment distance
            if segment_distance == 0 then
                -- Big reward for alignment + firing incentive
                if commanded_spinner == 0 then
                    reward = reward + 250
                else
                    reward = reward - segment_distance + 10
                end

                if player_state.fire_commanded then
                    reward = reward + 50
                end
            else
                -- MISALIGNED CASE (segment_distance > 0)
                -- Enemies at the top of tube should be shot when close (using segment distance)
                if (segment_distance < 2) then -- Check using actual segment distance
                    -- Use the depth returned by direction_to_nearest_enemy
                    if (enemy_depth <= 0x20) then
                        if player_state.fire_commanded == 1 then
                            -- Strong reward for firing at close enemies
                            reward = reward + 150
                        else
                            -- Moderate penalty for not firing at close enemies
                            reward = reward - 50
                        end
                    end
                end

                -- Graduated reward for proximity (higher reward for smaller segment distance)
                reward = reward + (10 - segment_distance) * 5 -- Simple linear reward for proximity

                -- Movement incentives (using desired_spinner direction and commanded_spinner)
                -- Reward if the COMMANDED movement (commanded_spinner) is IN THE SAME direction as the desired direction.
                if desired_spinner * commanded_spinner > 0 then
                    -- Reward for moving TOWARDS the target.
                    reward = reward + 25
                    -- Penalize if the COMMANDED movement is OPPOSITE to the desired direction.
                elseif desired_spinner * commanded_spinner < 0 then
                    -- Stronger penalty for moving AWAY from the target.
                    reward = reward - 50 -- Increased penalty
                    -- Penalize staying still when misaligned
                elseif commanded_spinner == 0 and desired_spinner ~= 0 then
                    reward = reward - 15  -- Small penalty for staying still when misaligned
                end
            end
        end
    else
        -- Large penalty for death to prioritize survival
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

    return reward, bDone, should_fire
end

-- Function to send parameters and get action each frame
local function process_frame(rawdata, player_state, controls, reward, bDone, bAttractMode)
    -- Increment frame count after processing
    frame_count = frame_count + 1

    -- Check if socket is open, try to reopen if not
    if not socket then
        if not open_socket() then
            return 0, 0, 0 -- Return zeros for fire, zap, spinner_delta on socket error
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
        if socket then
            socket:close(); socket = nil
        end
        open_socket()
        return 0, 0, 0 -- Return zeros for fire, zap, spinner_delta on write error
    end

    -- Try to read from socket with timeout protection
    local fire, zap, spinner = 0, 0, 0 -- Default values
    local read_start_time = os.clock()
    local read_timeout = 0.5           -- 500ms timeout for socket read

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
        if socket then
            socket:close(); socket = nil
        end
        open_socket()
        return 0, 0, 0 -- Return zeros for fire, zap, spinner_delta on read error
    end

    -- Return the three components directly
    return fire, zap, spinner
end

clear_screen()

-- Add after the initial requires, before the GameState class

-- Seed the random number generator once at script start
math.randomseed(os.time())

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
    self.gamestate = 0              -- Game state from address 0
    self.game_mode = 0              -- Game mode from address 5
    self.countdown_timer = 0        -- Countdown timer from address 4
    self.frame_counter = 0          -- Frame counter for tracking progress
    self.last_save_time = os.time() -- Track when we last sent save signal
    self.save_interval = 300        -- Send save signal every 5 minutes (300 seconds)
    self.start_delay = nil          -- Will be set to a random value between 0-59 in attract mode

    -- FPS tracking (now handled at global level, not in GameState)
    self.current_fps = 0 -- Store the FPS value for display

    return self
end

function GameState:update(mem)
    self.gamestate = mem:read_u8(0x0000)        -- Game state at address 0
    self.game_mode = mem:read_u8(0x0005)        -- Game mode at address 5
    self.countdown_timer = mem:read_u8(0x0004)  -- Countdown timer from address 4
    self.credits = mem:read_u8(0x0006)          -- Credits
    self.p1_level = mem:read_u8(0x0046)         -- Player 1 level
    self.p1_lives = mem:read_u8(0x0048)         -- Player 1 lives
    self.frame_counter = self.frame_counter + 1 -- Increment frame counter

    -- The current_fps is now only updated when FPS is calculated in frame_callback
end

-- **LevelState Class**
LevelState = {}
LevelState.__index = LevelState

function LevelState:new()
    local self = setmetatable({}, LevelState)
    self.level_number = 0
    self.spike_heights = {} -- Array of 16 spike heights
    self.level_type = 0     -- 00 = closed, FF = open
    self.level_angles = {}  -- Array of 16 tube angles
    self.level_shape = 0    -- Level shape (level_number % 16)
    return self
end

function LevelState:update(mem)
    self.level_number = mem:read_u8(0x009F)   -- Example address for level number
    self.level_type = mem:read_u8(0x0111)     -- Level type (00=closed, FF=open)
    self.level_shape = self.level_number % 16 -- Calculate level shape
    local player_pos = mem:read_u8(0x0200)    -- Player position
    local is_open = self.level_type == 0xFF

    -- Read spike heights for all 16 segments and store them using absolute positions
    self.spike_heights = {}
    for i = 0, 15 do -- Use 0-based indexing to match game's segment numbering
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
    self.player_depth = 0                          -- New field for player depth along the tube
    self.player_state = 0                          -- New field for player state from $201
    self.shot_segments = { 0, 0, 0, 0, 0, 0, 0, 0 } -- 8 shot segments
    self.shot_positions = { 0, 0, 0, 0, 0, 0, 0, 0 } -- 8 shot positions (depth)
    self.shot_count = 0
    self.debounce = 0
    self.fire_detected = 0
    self.zap_detected = 0
    self.SpinnerAccum = 0
    self.prevSpinnerAccum = 0
    self.fire_commanded = 0    -- Add fire_commanded field
    self.zap_commanded = 0     -- Add zap_commanded field
    self.spinner_commanded = 0 -- Renamed from SpinnerDelta
    return self
end

function PlayerState:update(mem)
    self.position = mem:read_u8(0x0200)     -- Player position
    self.player_state = mem:read_u8(0x0201) -- Player state value at $201
    self.player_depth = mem:read_u8(0x0202) -- Player depth along the tube

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

    self.superzapper_uses = mem:read_u8(0x03AA)   -- Superzapper availability
    self.superzapper_active = mem:read_u8(0x0125) -- Superzapper active status
    self.shot_count = mem:read_u8(0x0135)         -- Number of active player shots

    -- Read all 8 shot positions and segments
    local is_open = mem:read_u8(0x0111) == 0xFF
    local player_abs_segment = self.position & 0x0F
    for i = 1, 8 do
        -- Read depth (position along the tube)
        self.shot_positions[i] = mem:read_u8(0x02D3 + i - 1) -- PlayerShotPositions

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
    -- Read the spinner delta from memory into the renamed field
    self.spinner_commanded = mem:read_u8(0x0050) -- Value read here is overwritten later

    -- Calculate inferred delta by comparing with previous position
    local rawDelta = currentSpinnerAccum - self.prevSpinnerAccum

    -- Handle 8-bit wrap-around
    if rawDelta > 127 then
        rawDelta = rawDelta - 256
    elseif rawDelta < -128 then
        rawDelta = rawDelta + 256
    end

    -- Rename spinner_delta to spinner_detected
    self.spinner_detected = rawDelta

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
    self.pulse_beat = 0      -- Add pulse beat counter
    self.pulsing = 0         -- Add pulsing state
    self.pulsar_fliprate = 0 -- NEW: Add pulsar fliprate
    self.num_enemies_in_tube = 0
    self.num_enemies_on_top = 0
    self.enemies_pending = 0
    self.nearest_enemy_seg = INVALID_SEGMENT -- Initialize relative nearest enemy segment with sentinel
    self.enemy_abs_segments = {}             -- NEW: Store absolute segments for enemies
    self.enemy_shot_abs_segments = {}        -- NEW: Store absolute segments for enemy shots

    -- NEW Engineered Features for Targeting/Aiming
    self.is_aligned_with_nearest = 0.0
    self.nearest_enemy_depth_raw = 255 -- Sentinel value (max depth)
    self.alignment_error_magnitude = 0.0

    -- NEW: Array to track charging fuseballs by absolute segment (0-15 -> index 1-16)
    self.charging_fuseball_segments = {}

    -- Enemy info arrays (Original)
    self.enemy_type_info = { 0, 0, 0, 0, 0, 0, 0 } -- Raw type byte from $0169 (or similar)
    self.active_enemy_info = { 0, 0, 0, 0, 0, 0, 0 } -- Raw state byte from $0170 (or similar)
    self.enemy_segments = { 0, 0, 0, 0, 0, 0, 0 }  -- 7 enemy segment numbers ($02B9)
    self.enemy_depths = { 0, 0, 0, 0, 0, 0, 0 }    -- 7 enemy depth positions ($02DF)
    -- LSB/Display list related fields remain...
    self.enemy_depths_lsb = { 0, 0, 0, 0, 0, 0, 0 } -- 7 enemy depth positions LSB
    self.enemy_shot_lsb = { 0, 0, 0, 0, 0, 0, 0 }  -- 7 enemy shot LSB values at $02E6

    -- NEW: Decoded Enemy Info Tables (Size 7)
    self.enemy_core_type = { 0, 0, 0, 0, 0, 0, 0 }      -- Bits 0-2 from type byte
    self.enemy_direction_moving = { 0, 0, 0, 0, 0, 0, 0 } -- Bit 6 from type byte (0/1)
    self.enemy_between_segments = { 0, 0, 0, 0, 0, 0, 0 } -- Bit 7 from type byte (0/1)
    self.enemy_moving_away = { 0, 0, 0, 0, 0, 0, 0 }    -- Bit 7 from state byte (0/1)
    self.enemy_can_shoot = { 0, 0, 0, 0, 0, 0, 0 }      -- Bit 6 from state byte (0/1)
    self.enemy_split_behavior = { 0, 0, 0, 0, 0, 0, 0 } -- Bits 0-1 from state byte

    -- Convert shot positions to simple array like other state values
    self.shot_positions = { 0, 0, 0, 0 } -- 4 shot positions

    self.pending_vid = {}              -- 64-byte table
    self.pending_seg = {}              -- 64-byte table

    -- Enemy shot segments parameters extracted from memory address 02B5
    self.enemy_shot_segments = {
        { address = 0x02B5, value = 0 },
        { address = 0x02B6, value = 0 },
        { address = 0x02B7, value = 0 },
        { address = 0x02B8, value = 0 }
    }
    return self
end

function EnemiesState:update(mem, game_state, player_state, level_state)
    -- First, initialize/reset all arrays at the beginning
    -- Reset enemy arrays
    -- Keep resetting original raw arrays
    self.enemy_type_info         = { 0, 0, 0, 0, 0, 0, 0 }
    self.active_enemy_info       = { 0, 0, 0, 0, 0, 0, 0 }
    self.enemy_segments          = { 0, 0, 0, 0, 0, 0, 0 }
    self.enemy_abs_segments      = { INVALID_SEGMENT, INVALID_SEGMENT, INVALID_SEGMENT, INVALID_SEGMENT, INVALID_SEGMENT,
        INVALID_SEGMENT, INVALID_SEGMENT }                                                                                                            -- Reset new table
    self.enemy_depths            = { 0, 0, 0, 0, 0, 0, 0 }
    self.enemy_depths_lsb        = { 0, 0, 0, 0, 0, 0, 0 }
    self.enemy_shot_lsb          = { 0, 0, 0, 0, 0, 0, 0 }
    self.enemy_shot_abs_segments = { INVALID_SEGMENT, INVALID_SEGMENT, INVALID_SEGMENT, INVALID_SEGMENT } -- Reset new table
    self.enemy_move_vectors      = { 0, 0, 0, 0, 0, 0, 0 }                                              -- Keep this reset if used elsewhere
    self.enemy_state_flags       = { 0, 0, 0, 0, 0, 0, 0 }                                              -- Keep this reset if used elsewhere
    -- NEW: Reset decoded tables
    self.enemy_core_type         = { 0, 0, 0, 0, 0, 0, 0 }
    self.enemy_direction_moving  = { 0, 0, 0, 0, 0, 0, 0 }
    self.enemy_between_segments  = { 0, 0, 0, 0, 0, 0, 0 }
    self.enemy_moving_away       = { 0, 0, 0, 0, 0, 0, 0 }
    self.enemy_can_shoot         = { 0, 0, 0, 0, 0, 0, 0 }
    self.enemy_split_behavior    = { 0, 0, 0, 0, 0, 0, 0 }

    -- Read active enemies counts, pulse state, etc.
    self.active_flippers         = mem:read_u8(0x0142) -- n_flippers - current active count
    self.active_pulsars          = mem:read_u8(0x0143) -- n_pulsars
    self.active_tankers          = mem:read_u8(0x0144) -- n_tankers
    self.active_spikers          = mem:read_u8(0x0145) -- n_spikers
    self.active_fuseballs        = mem:read_u8(0x0146) -- n_fuseballs
    self.pulse_beat              = mem:read_u8(0x0147) -- pulse_beat counter
    self.pulsing                 = mem:read_u8(0x0148) -- pulsing state
    self.pulsar_fliprate         = mem:read_u8(0x00B2) -- NEW: Pulsar flip rate at $B2
    self.num_enemies_in_tube     = mem:read_u8(0x0108)
    self.num_enemies_on_top      = mem:read_u8(0x0109)
    self.enemies_pending         = mem:read_u8(0x03AB)

    -- Update enemy shot segments from memory (store relative distances)
    local player_abs_segment     = mem:read_u8(0x0200) & 0x0F -- Get current player absolute segment
    local is_open                = mem:read_u8(0x0111) == 0xFF

    for i = 1, 4 do
        local abs_segment = mem:read_u8(self.enemy_shot_segments[i].address)
        if abs_segment == 0 then
            self.enemy_shot_segments[i].value = INVALID_SEGMENT                                                    -- Relative segment (for display/OOB)
            self.enemy_shot_abs_segments[i] = INVALID_SEGMENT                                                      -- Absolute segment (for internal use)
        else
            local segment = abs_segment & 0x0F                                                                     -- Mask to ensure 0-15
            self.enemy_shot_segments[i].value = absolute_to_relative_segment(player_abs_segment, segment, is_open) -- Relative
            self.enemy_shot_abs_segments[i] = segment                                                              -- Absolute
        end
    end

    -- Get player position and level type for relative calculations
    local player_pos = mem:read_u8(0x0200)
    local is_open = mem:read_u8(0x0111) == 0xFF

    -- Read available spawn slots (how many more can be created)
    self.spawn_slots_flippers = mem:read_u8(0x013D)  -- avl_flippers - spawn slots available
    self.spawn_slots_pulsars = mem:read_u8(0x013E)   -- avl_pulsars
    self.spawn_slots_tankers = mem:read_u8(0x013F)   -- avl_tankers
    self.spawn_slots_spikers = mem:read_u8(0x0140)   -- avl_spikers
    self.spawn_slots_fuseballs = mem:read_u8(0x0141) -- avl_fuseballs

    local activeEnemies = self.num_enemies_in_tube + self.num_enemies_on_top

    -- Read standard enemy segments and depths first, store relative segments
    for i = 1, 7 do
        local abs_segment_raw = mem:read_u8(0x02B9 + i - 1)
        self.enemy_depths[i] = mem:read_u8(0x02DF + i - 1)

        if (self.enemy_depths[i] == 0 or abs_segment_raw == 0) then
            self.enemy_segments[i] = INVALID_SEGMENT     -- Relative segment
            self.enemy_abs_segments[i] = INVALID_SEGMENT -- Absolute segment
        else
            local abs_segment = abs_segment_raw & 0x0F   -- Mask to ensure 0-15
            self.enemy_abs_segments[i] = abs_segment     -- Store absolute segment
            -- Store relative segment distance
            self.enemy_segments[i] = absolute_to_relative_segment(player_abs_segment, abs_segment, is_open)
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
            self.enemy_split_behavior[i] = state_byte & 0x03                -- Bits 0-1: Split Behavior
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

    -- NEW: Calculate charging Fuseball segments
    -- Initialize with 16 zeros (for absolute segments 0-15 -> indices 1-16)
    self.charging_fuseball_segments = {}
    for i = 1, 7 do
        -- Check if it's a Fuseball (type 4) and moving towards player (bit 7 of state byte is clear)
        if self.enemy_core_type[i] == 4 and (self.active_enemy_info[i] & 0x80) == 0 then
            -- Read the absolute segment directly from memory
            local abs_segment = mem:read_u8(0x02B9 + i - 1) & 0x0F -- Mask to 0-15
            -- Set the flag in our table (use 1-based index)
            self.charging_fuseball_segments[abs_segment + 1] = 1
        end
    end

    -- Read all 4 enemy shot positions and store absolute positions
    for i = 1, 4 do
        local raw_pos = mem:read_u8(0x02DB + i - 1)
        self.shot_positions[i] = raw_pos -- Store full raw position value

        -- Invalidate the segment values for any shots that are zeroed
        if (self.shot_positions[i] == 0) then
            self.enemy_shot_segments[i].value = INVALID_SEGMENT
        end
    end

    -- Read pending_seg (64 bytes starting at 0x0203), store relative
    for i = 1, 64 do
        local abs_segment = mem:read_u8(0x0203 + i - 1)
        if abs_segment == 0 then
            self.pending_seg[i] = INVALID_SEGMENT -- Not active, use sentinel
        else
            local segment = abs_segment & 0x0F    -- Mask to ensure 0-15
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
    for i = 0, 31 do -- Just scan part of it for efficiency
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
    -- Capture BOTH return values: absolute segment and depth
    local nearest_abs_seg, nearest_depth = self:target_segment(game_state, player_state, level_state)
    if nearest_abs_seg == -1 then
        self.nearest_enemy_seg = INVALID_SEGMENT
        -- Set default values for engineered features when no enemy
        self.is_aligned_with_nearest = 0.0
        self.nearest_enemy_depth_raw = 255 -- Use max depth as sentinel
        self.alignment_error_magnitude = 0.0
    else
        local nearest_rel_seg = absolute_to_relative_segment(player_abs_segment, nearest_abs_seg, is_open)
        self.nearest_enemy_seg = nearest_rel_seg     -- Store relative for internal use/display
        self.nearest_enemy_depth_raw = nearest_depth -- Store raw depth

        -- Calculate Is_Aligned
        self.is_aligned_with_nearest = (nearest_rel_seg == 0) and 1.0 or 0.0

        -- Calculate Alignment_Error_Magnitude (Scaled to 0-10000 for packing)
        local error_abs = math.abs(nearest_rel_seg)
        local normalized_error = 0.0
        if is_open then
            -- Normalize based on max possible distance in open level (15)
            if error_abs > 0 then normalized_error = error_abs / 15.0 end
        else
            -- Normalize based on max possible distance in closed level (8)
            if error_abs > 0 then normalized_error = error_abs / 8.0 end
        end
        -- Scale to 0-10000 and store as integer
        self.alignment_error_magnitude = math.floor(normalized_error * 10000.0)
    end
end

-- Function to check if we should fire at a segment
local function check_segment_threat(segment, level_state, enemies_state)
    local depth, enemy_type = top_enemy_in_segment(segment, level_state, enemies_state)

    -- If nothing found in segment
    if depth == 255 then
        return false
    end

    -- Never aim at fuseballs
    if enemy_type == 4 then
        return false
    end

    -- Always fire at enemy shots (type 8)
    if enemy_type == 8 then
        return true
    end

    -- Fire at any enemy in our segment (except fuseballs)
    return true
end

function EnemiesState:target_segment(game_state, player_state, level_state)
    -- Initialize return values
    local best_segment = -1 -- Use -1 as sentinel for no target
    local best_depth = 255  -- Use 255 as sentinel for max depth
    local should_fire = false

    -- During tube transition, check for better lanes
    if game_state and game_state.gamestate == 0x20 then
        -- Get current player segment and check adjacent lanes
        local player_abs_segment = player_state.position & 0x0F
        local is_open = level_state.level_type == 0xFF

        -- Initialize with current lane's spike height
        local shortest_height = level_state.spike_heights[player_abs_segment] or 255
        local best_lane = player_abs_segment

        -- Check left lane
        if player_abs_segment > 0 or not is_open then
            local left_segment = (player_abs_segment - 1) & 0x0F
            local left_height = level_state.spike_heights[left_segment] or 255
            if left_height < shortest_height then
                shortest_height = left_height
                best_lane = left_segment
            end
        end

        -- Check right lane
        if player_abs_segment < 15 or not is_open then
            local right_segment = (player_abs_segment + 1) & 0x0F
            local right_height = level_state.spike_heights[right_segment] or 255
            if right_height < shortest_height then
                shortest_height = right_height
                best_lane = right_segment
            end
        end

        -- Return the best lane and always fire during zoom
        return best_lane, shortest_height, true
    end

    -- Normal gameplay enemy targeting starts here
    local player_abs_segment = player_state.position & 0x0F
    local is_open = level_state.level_type == 0xFF

    -- First check for dangerous Fuseballs
    for i = 1, 7 do
        if self.enemy_core_type[i] == 4 and        -- Is Fuseball
            ((self.enemy_depths[i] == 0x10) or     -- At top rail
                (self.enemy_depths[i] < 0x40 and   -- OR close and
                    self.enemy_moving_away[i] == 0)) then -- moving towards us
            local fuseball_segment = self.enemy_abs_segments[i]
            if fuseball_segment >= 0 and fuseball_segment <= 15 then
                best_segment = fuseball_segment
                best_depth = self.enemy_depths[i]
                should_fire = false -- Don't fire at fuseballs
                return best_segment, best_depth, should_fire
            end
        end
    end

    -- Check for top rail enemies
    for i = 1, 7 do
        if self.enemy_depths[i] == 0x10 then
            local enemy_segment = self.enemy_abs_segments[i]
            if enemy_segment >= 0 and enemy_segment <= 15 then
                best_segment = enemy_segment
                best_depth = self.enemy_depths[i]
                break
            end
        end
    end

    -- Check if we should fire at current or adjacent segments
    should_fire = check_segment_threat(player_abs_segment, level_state, self)                  -- Current lane
    if not should_fire and (player_abs_segment > 0 or not is_open) then
        should_fire = check_segment_threat((player_abs_segment - 1) & 0x0F, level_state, self) -- Left lane
    end
    if not should_fire and (player_abs_segment < 15 or not is_open) then
        should_fire = check_segment_threat((player_abs_segment + 1) & 0x0F, level_state, self) -- Right lane
    end

    -- If we haven't found a best segment yet, look for any enemy
    if best_segment == -1 then
        -- Find closest enemy
        local closest_depth = 255
        for i = 1, 7 do
            if self.enemy_depths[i] < closest_depth and self.enemy_abs_segments[i] >= 0 and self.enemy_abs_segments[i] <= 15 then
                closest_depth = self.enemy_depths[i]
                best_segment = self.enemy_abs_segments[i]
                best_depth = self.enemy_depths[i]
            end
        end
    end

    return best_segment, best_depth, should_fire
end

function direction_to_nearest_enemy(game_state, level_state, player_state, target_abs_segment)
    -- Get the player's current absolute segment
    local player_abs_seg = player_state.position & 0x0F
    local is_open = level_state.level_type == 0xFF

    -- If no target was provided (target_abs_segment is -1)
    if target_abs_segment == -1 then
        return 0, 0, 255 -- No target, return spinner 0, distance 0, max depth
    end

    -- Calculate the relative segment distance using the helper function
    local relative_dist = absolute_to_relative_segment(player_abs_seg, target_abs_segment, is_open)

    -- If already aligned (relative distance is 0)
    if relative_dist == 0 then
        return 0, 0, 0 -- Aligned, return spinner 0, distance 0, depth 0
    end

    -- Calculate actual segment distance and intensity
    local actual_segment_distance = math.abs(relative_dist)
    local intensity = math.min(0.9, 0.3 + (actual_segment_distance * 0.05))

    -- Set spinner direction based on the sign of the relative distance
    -- The absolute_to_relative_segment function handles open/closed logic correctly
    local spinner = relative_dist > 0 and intensity or -intensity

    return spinner, actual_segment_distance, 0 -- Return spinner, distance, and depth 0 (depth no longer relevant)
end

-- Function to decode enemy type info
function EnemiesState:decode_enemy_type(type_byte)
    local enemy_type = type_byte & 0x07
    local between_segments = (type_byte & 0x80) ~= 0
    local segment_increasing = (type_byte & 0x40) ~= 0
    return string.format("%d%s%s",
        enemy_type,
        between_segments and "B" or "-",
        segment_increasing and "+" or "" -- Remove the '+' sign for segment numbers
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

    -- Get Start button port and field
    self.in2_port = manager.machine.ioport.ports[":IN2"]
    self.p1_start_field = nil
    if self.in2_port then
        self.p1_start_field = self.in2_port.fields["1 Player Start"] or
            self.in2_port.fields["P1 Start"] or
            self.in2_port.fields["Start 1"]
        if not self.p1_start_field then
            print("Warning: Could not find P1 Start button field in :IN2 port.")
        end
    else
        print("Warning: Could not find :IN2 port for Start button.")
    end

    return self
end

-- Apply received AI action (fire, zap, spinner) and p1_start to game controls
function Controls:apply_action(fire, zap, spinner, p1_start, game_state, player_state)
    player_state.fire_commanded = fire
    player_state.zap_commanded = zap
    player_state.spinner_commanded = spinner

    self.fire_field:set_value(fire)
    self.zap_field:set_value(zap)
    self.p1_start_field:set_value(p1_start)

    -- Write AI spinner value to memory only if non-zero to potentially reduce noise/interference
    if spinner ~= 0 then
        mem:write_u8(0x0050, spinner)
    end
end

-- Instantiate state objects - AFTER defining all classes but BEFORE functions using them
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
local function flatten_game_state_to_binary(reward, game_state, level_state, player_state, enemies_state, bDone,
                                            should_fire)
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
    table.insert(data, segment_delta)                           -- Relative distance (-7 to +8 or -15 to +15) or 0
    -- Add NEW Engineered Features (Targeting/Aiming)
    table.insert(data, enemies_state.nearest_enemy_depth_raw)   -- Raw depth (0-255)
    table.insert(data, enemies_state.is_aligned_with_nearest)   -- Float (0.0 or 1.0)
    table.insert(data, enemies_state.alignment_error_magnitude) -- Float (0.0-1.0)

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

    -- NEW: Top Enemy Segments (fixed size: 7) - Relative segments only for enemies at depth 0x10
    for i = 1, 7 do
        if enemies_state.enemy_depths[i] == 0x10 then
            table.insert(data, enemies_state.enemy_segments[i]) -- Insert the relative segment
        else
            table.insert(data, INVALID_SEGMENT)                 -- Use sentinel if not at depth 0x10
        end
    end

    -- Enemy depths (fixed size: 7) - Absolute depth combined with LSB
    for i = 1, 7 do
        local depth = enemies_state.enemy_depths[i] or 0
        local lsb = enemies_state.enemy_depths_lsb[i] or 0
        -- Combine MSB depth and LSB (scaled 0-1) into a float
        table.insert(data, depth + (lsb / 255.0))
    end

    -- Enemy shot positions (fixed size: 4) - Absolute depth
    for i = 1, 4 do
        table.insert(data, enemies_state.shot_positions[i])
    end

    -- Enemy shot segments (fixed size: 4) - Relative segments
    for i = 1, 4 do
        table.insert(data, enemies_state.enemy_shot_segments[i].value or INVALID_SEGMENT) -- Use sentinel
    end

    -- NEW: Charging Fuseball flags per absolute segment (fixed size: 16)
    for i = 1, 16 do
        -- Insert 1 if a fuseball is charging towards the rim in this segment, 0 otherwise
        table.insert(data, enemies_state.charging_fuseball_segments[i] or 0)
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
    local nearest_abs_seg_oob, enemy_depth_oob = enemies_state:target_segment(game_state, player_state, level_state)
    local player_abs_seg_oob = player_state.position & 0x0F             -- Use absolute player segment 0-15

    local is_enemy_present_oob = (nearest_abs_seg_oob ~= -1) and 1 or 0 -- Check against -1 sentinel

    -- Calculate expert recommendations
    -- Use the should_fire value passed from calculate_reward
    local expert_should_fire = should_fire and 1 or 0 -- Convert boolean to 0/1
    local should_zap = 0

    -- Pack header data using 2-byte integers where possible
    local score = player_state.score or 0
    local score_high = math.floor(score / 65536) -- High 16 bits
    local score_low = score % 65536              -- Low 16 bits

    -- Mask frame counter to 16 bits to prevent overflow
    local frame = game_state.frame_counter % 65536

    -- Ensure the format string and order match EXACTLY the previous working version.
    -- H = unsigned short (uint16)
    -- d = double
    -- B = unsigned char (uint8)
    -- h = signed short (int16)
    local oob_data = string.pack(">HdBBBHHHBBBhBhBBBB",
        #data,                          -- H num_values (size of main payload in 16-bit words)
        reward,                         -- d reward
        0,                              -- B game_action (placeholder)
        game_state.game_mode,           -- B game_mode
        bDone and 1 or 0,               -- B done flag
        frame,                          -- H frame_counter (16-bit)
        score_high,                     -- H score high 16 bits
        score_low,                      -- H score low 16 bits
        save_signal,                    -- B save_signal
        player_state.fire_commanded,    -- B fire_commanded
        player_state.zap_commanded,     -- B zap_commanded
        player_state.spinner_commanded, -- h spinner_delta (model output, int8 range but packed as int16)
        is_attract_mode and 1 or 0,     -- B is_attract_mode
        nearest_abs_seg_oob,            -- h target_segment (ABSOLUTE, -1 sentinel, packed as int16)
        player_abs_seg_oob,             -- B player_segment (ABSOLUTE 0-15)
        is_open_level and 1 or 0,       -- B is_open_level
        expert_should_fire,             -- B expert fire recommendation (using should_fire from calculate_reward)
        should_zap                      -- B expert zap recommendation
    )
    -- --- End OOB Data Packing ---

    -- Combine out-of-band header with game state data
    local final_data = oob_data .. binary_data

    return final_data, #data -- Return the combined data and the size of the main payload
end

-- Update the frame_callback function to track bytes sent and calculate FPS
local frame_counter = 0
local lastTimer = 0
local last_fps_time = os.time() -- Use os.time() for wall clock time

-- Add a global tracker for timer value changes
local timer_changes = 0
local timer_check_start_time = os.clock()

-- Add a last update time tracker for precise intervals
local last_update_time = os.clock()

-- Add tracking for different frame detection methods
local last_player_position = nil
local last_timer_value = nil
local method_fps_counter = { 0, 0, 0 } -- For 3 different methods
local method_start_time = os.clock()
local last_frame_counter = 0         -- Initialize counter for FPS calculation
local frames_to_wait = 0
local frames_waited = 0

local function frame_callback()
    -- Check the time counter at address 0x0003
    local currentTimer = mem:read_u8(0x0003)

    -- Check if the timer changed
    if currentTimer == lastTimer then
        return true
    end
    lastTimer = currentTimer

    frames_waited = frames_waited + 1
    if frames_waited <= frames_to_wait then
        return true
    end

    -- Track FPS using frame_counter instead of separate frame_count
    local current_time = os.time()

    -- Calculate FPS every second
    if current_time > last_fps_time then
        -- Update the FPS display using frame_counter
        game_state.current_fps = game_state.frame_counter - last_frame_counter
        last_frame_counter = game_state.frame_counter
        last_fps_time = current_time
    end

    -- Update game state first
    game_state:update(mem)

    -- Reset start_delay when entering attract mode
    if (game_state.game_mode & 0x80) == 0 and game_state.start_delay ~= nil then
        game_state.start_delay = nil -- Reset delay so it will be regenerated
        print("Reset start delay for new attract mode session")
    end

    -- Update level state next
    level_state:update(mem)

    -- Update player state before enemies state
    player_state:update(mem)

    -- Update enemies state last to ensure all references are correct
    enemies_state:update(mem, game_state, player_state, level_state)


    -- Declare num_values at the start of the function
    local num_values = 0
    local bDone = false

    -- Check if socket is open, try to reopen if not
    if not socket then
        if not open_socket() then
            -- If socket isn't open yet, just update the display without sending data
            -- Update the display with empty controls
            update_display("Waiting for Python connection...", game_state, level_state, player_state, enemies_state, nil,
                num_values)
        end
    end

    -- 2 Credits
    mem:write_u8(0x0006, 2)

    -- Reset the countdown timer to zero all the time in order to move quickly through attract mode
    -- mem:write_u8(0x0004, 0)
    -- NOP out the jump that skips scoring in attract mode
    -- mem:write_direct_u8(0xCA6F, 0xEA)
    -- mem:write_direct_u8(0xCA70, 0xEA)

    -- NOP out the damage the copy protection code does to memory when it detects a bad checksum
    mem:write_direct_u8(0xA591, 0xEA)
    mem:write_direct_u8(0xA592, 0xEA)

    local status_message = ""

    local is_attract_mode = (game_state.game_mode & 0x80) == 0
    local is_level_select = (game_state.game_state == 0x16)

    -- Calculate the reward for the current frame - do this ONCE per frame
    -- Note: We calculate reward *before* getting the commanded spinner for *this* frame.
    --       To use the commanded spinner *in* the reward calculation, we need to get it first.
    --       (Refactoring needed - see below)

    -- NOP out the clearing of zap_fire_new
    -- mem:write_direct_u8(0x976E, 0x00)
    -- mem:write_direct_u8(0x976F, 0x00)

    -- Add periodic save mechanism based on frame counter instead of key press
    -- This will trigger a save every 30,000 frames (approximately 8 minutes at 60fps)
    if game_state.frame_counter % 30000 == 0 then
        print("Frame counter triggered save at frame " .. game_state.frame_counter)
        game_state.last_save_time = 0 -- Force save on next frame
    end

    -- Try to detect ESC key press using a more reliable method
    -- Check for ESC key in all available ports
    local esc_pressed = false
    for port_name, port in pairs(manager.machine.ioport.ports) do
        for field_name, field in pairs(port.fields) do
            if field_name:find("ESC") or field_name:find("Escape") then
                if field.pressed then
                    print("ESC key detected - Triggering save")
                    game_state.last_save_time = 0 -- Force save on next frame
                    esc_pressed = true
                    break
                end
            end
        end
        if esc_pressed then break end
    end

    -- 1. Get the action COMMANDED by the AI for the CURRENT state (from Python)
    --    To do this, we first need to send the *previous* frame's state/reward/done.
    --    The flatten function packs the reward/done for the *current* frame into the OOB header,
    --    so we need a temporary flatten/send just to get the action.

    -- Calculate the reward based on the state *before* the commanded action is known.
    -- This is necessary because the reward is packed in the data sent *to* python.
    -- We will pass 0 for commanded_spinner here as it's not known yet.
    -- NOTE: This means the reward calculation CANNOT depend on the commanded spinner.
    --       If it MUST, the protocol needs changing.
    --       Let's assume for now it's okay to use the *previous* frame's commanded spinner (in player_state.SpinnerDelta)
    --       or just ignore commanded spinner's effect on reward.
    --       We'll revert calculate_reward temporarily to not require commanded_spinner.
    -- TODO: Re-evaluate if calculate_reward *truly* needs commanded_spinner or if previous frame's effect is ok.

    -- Calculate reward based on *detected* movement from the previous frame
    local reward, bDone, should_fire = calculate_reward(game_state, level_state, player_state, enemies_state,
        player_state.spinner_detected)

    -- Flatten and serialize the game state data (using the calculated reward/bDone for OOB header)
    local frame_data
    frame_data, num_values = flatten_game_state_to_binary(reward, game_state, level_state, player_state, enemies_state,
        bDone, should_fire)                                                                                                                 -- Pass should_fire here

    -- Send the current state data (s') and reward/done (r, d) to Python,
    -- and get the action (a) for this state (s') back.
    local fire, zap, spinner = process_frame(frame_data, player_state, controls, reward, bDone, is_attract_mode)

    -- Now we have the commanded spinner for THIS frame (stored in local var 'spinner')

    -- Update total bytes sent
    total_bytes_sent = total_bytes_sent + #frame_data
    local current_time_high_res = os.clock()
    local should_update_display = (current_time_high_res - last_display_update) >= DISPLAY_UPDATE_INTERVAL

    controls:apply_action(0, 0, 0, 0, game_state, player_state)

    -- Apply the action to the controls, which will also update the player_state to reflect the commanded values
    if game_state.gamestate == 0x12 then -- High Score Entry Mode (0x12 = 18)
        -- Press fire only every 30 frames instead of every other frame
        controls:apply_action((game_state.frame_counter % 10) == 0 and 1 or 0, 0, 0, 0, game_state, player_state)
    elseif game_state.gamestate == 0x16 then                             -- Level Select Mode
        if level_select_counter < 60 then                                -- Spin for 60 frames
            controls:apply_action(0, 0, 31, 0, game_state, player_state) -- No fire, no P1 Start during spin
            level_select_counter = level_select_counter + 1
        elseif level_select_counter == 60 then                           -- Press fire exactly once
            controls:apply_action(1, 0, 0, 0, game_state, player_state)  -- Press fire once, no P1 Start
            level_select_counter = 0
        else
            controls:apply_action(0, 0, 0, 0, game_state, player_state) -- No inputs after fire pressed
        end
    else
        if (game_state.game_mode & 0x80) == 0 then -- Attract Mode
            -- Only push start after our random delay
            local should_push_start = (game_state.frame_counter % 50 == 0) and 1 or 0
            controls:apply_action(0, 0, 0, should_push_start, game_state, player_state)
        elseif game_state.gamestate == 0x04 or game_state.gamestate == 0x20 then -- Play mode and other states
            -- Only apply the fire command if we have 6 or fewer shots
            if player_state.shot_count > 6 then
                fire = 0 -- Too many shots, don't fire
            end
            controls:apply_action(fire, zap, spinner, 0, game_state, player_state)
        end
    end

    -- Update the display with the current action and metrics
    if should_update_display and SHOW_DISPLAY then
        -- Update the display with the current action and metrics
        update_display(status_message, game_state, level_state, player_state, enemies_state, action, num_values, reward)
        last_display_update = current_time_high_res
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
        { "Gamestate",    string.format("0x%02X", game_state.gamestate) },
        { "Game Mode",    string.format("0x%02X", game_state.game_mode) },
        { "Countdown",    string.format("0x%02X", game_state.countdown_timer) },
        { "Credits",      game_state.credits },
        { "P1 Lives",     game_state.p1_lives },
        { "P1 Level",     game_state.p1_level },
        { "Frame",        game_state.frame_counter },
        { "Bytes Sent",   total_bytes_sent },
        { "FPS",          string.format("%.2f", game_state.current_fps) },
        { "Payload Size", num_values },
        { "Last Reward",  string.format("%d", math.floor(LastRewardState + 0.5)) }, -- Changed format to integer
    }

    -- Calculate how many rows we need (ceiling of items/3)
    local rows = math.ceil(#game_metrics / 3)

    for row = 1, rows do
        local line = "  "
        for col = 1, 3 do
            local idx = (row - 1) * 3 + col
            if idx <= #game_metrics then
                local key, value = table.unpack(game_metrics[idx])
                -- Format each metric with fixed width, right-align value
                line = line .. string.format("%-12s: %10s   ", key, tostring(value)) -- Changed to right-align value
            end
        end
        print(line)
    end
    print("") -- Empty line after section

    -- Format and print player state
    print("--[ Player State ]------------------------------------")

    -- Create player metrics in a more organized way for 3-column display
    local player_metrics = {
        { "Position",       string.format("%d", player_state.position) },
        { "State",          string.format("0x%02X", player_state.player_state) },
        { "Depth",          string.format("%d", player_state.player_depth) },
        { "Alive",          string.format("%d", player_state.alive) },
        { "Score",          string.format("%d", player_state.score) },
        { "Szapper Uses",   string.format("%d", player_state.superzapper_uses) },
        { "Szapper Active", string.format("%d", player_state.superzapper_active) },
        { "Shot Count",     string.format("%d", player_state.shot_count) },
        { "Debounce",       string.format("%d", player_state.debounce) },
        { "Fire Detected",  string.format("%d", player_state.fire_detected) },
        { "Zap Detected",   string.format("%d", player_state.zap_detected) },
        { "SpinnerAccum",   string.format("%d", player_state.SpinnerAccum) },
        { "SpinnerCmd",     string.format("%d", player_state.spinner_commanded) }, -- Commanded value
        { "SpinnerDet",     string.format("%d", player_state.spinner_detected) } -- Detected value (renamed)
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

    print("") -- Empty line after section

    -- Level State Section
    print("--[ Level State ]--------------------------------------------------------------------")
    local spike_heights_str = ""
    for i = 0, 15 do
        spike_heights_str = spike_heights_str .. string.format("%02X ", level_state.spike_heights[i] or 0)
    end
    print("  Spike Heights: " .. spike_heights_str)
    print(string.format("  Level Num: %d Type: %s Shape: %d", level_state.level_number,
        (level_state.level_type == 0xFF and "Open" or "Closed"), level_state.level_shape))
    print("") -- Empty line after section

    -- Controls/AI State Section
    print("--[ Controls & AI ]--------------------------------------------------------------")
    print(string.format("  %-25s: %d", "Model Fire", player_state.fire_commanded))
    print(string.format("  %-25s: %d", "Model Zap", player_state.zap_commanded))
    print(string.format("  %-25s: %d", "Model Spinner", player_state.spinner_commanded))
    print("")

    -- move_cursor_to_row(21)
    local enemy_types = {}
    local enemy_states = {}
    local enemy_segs = {}
    local enemy_depths = {}
    for i = 1, 7 do
        enemy_types[i] = enemies_state:decode_enemy_type(enemies_state.enemy_type_info[i])
        enemy_states[i] = enemies_state:decode_enemy_state(enemies_state.active_enemy_info[i])
        -- Format enemy segments using the helper function
        enemy_segs[i] = format_segment(enemies_state.enemy_segments[i])
        -- Display enemy depths as 2-digit hex values
        enemy_depths[i] = string.format("%02X", enemies_state.enemy_depths[i])
    end

    local enemies_metrics = {
        ["Flippers"] = string.format("%d active, %d spawn slots", enemies_state.active_flippers,
            enemies_state.spawn_slots_flippers),
        ["Pulsars"] = string.format("%d active, %d spawn slots", enemies_state.active_pulsars,
            enemies_state.spawn_slots_pulsars),
        ["Tankers"] = string.format("%d active, %d spawn slots", enemies_state.active_tankers,
            enemies_state.spawn_slots_tankers),
        ["Spikers"] = string.format("%d active, %d spawn slots", enemies_state.active_spikers,
            enemies_state.spawn_slots_spikers),
        ["Fuseballs"] = string.format("%d active, %d spawn slots", enemies_state.active_fuseballs,
            enemies_state.spawn_slots_fuseballs),
        ["Total"] = string.format("%d active, %d spawn slots",
            enemies_state:get_total_active(),
            enemies_state.spawn_slots_flippers + enemies_state.spawn_slots_pulsars +
            enemies_state.spawn_slots_tankers + enemies_state.spawn_slots_spikers +
            enemies_state.spawn_slots_fuseballs),
        ["Pulse State"] = string.format("beat:%02X charge:%02X/FF", enemies_state.pulse_beat, enemies_state.pulsing),
        ["Flip Rate"] = string.format("%02X", enemies_state.pulsar_fliprate), -- NEW: Display Pulsar Flip Rate
        ["In Tube"] = string.format("%d enemies", enemies_state.num_enemies_in_tube),
        ["Nearest Enemy"] = string.format("segment %s", format_segment(enemies_state.nearest_enemy_seg)),
        ["On Top"] = string.format("%d enemies", enemies_state.num_enemies_on_top),
        ["Pending"] = string.format("%d enemies", enemies_state.enemies_pending),
        ["Enemy Types"] = table.concat(enemy_types, " "),
        ["Enemy States"] = table.concat(enemy_states, " ")
    }

    print(format_section("Enemies State", enemies_metrics))

    -- Add enemy segments and depths on their own lines
    print("  Enemy Segments: " .. table.concat(enemy_segs, " "))

    -- NEW: Display segments of enemies specifically at depth 0x10
    local top_enemy_segs = {}
    for i = 1, 7 do
        if enemies_state.enemy_depths[i] == 0x10 then
            top_enemy_segs[i] = format_segment(enemies_state.enemy_segments[i])
        else
            top_enemy_segs[i] = format_segment(INVALID_SEGMENT)
        end
    end
    print("  Enemies On Top: " .. table.concat(top_enemy_segs, " "))
    print("  Enemy Depths  : " .. table.concat(enemy_depths, " "))

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

    print("") -- Empty line after section

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

    -- NEW: Display charging fuseball flags per absolute segment
    local charging_fuseball_str = {}
    for i = 1, 16 do
        if enemies_state.charging_fuseball_segments[i] == 1 then
            table.insert(charging_fuseball_str, "*")
        else
            table.insert(charging_fuseball_str, "-")
        end
    end
    print("  Fuseball Chrg : " .. table.concat(charging_fuseball_str, " "))

    print("") -- Empty line after section
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
        local reward = calculate_reward(game_state, level_state, player_state, enemies_state, player_state.spinner_detected)
        
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
end

-- Start the Python script but don't wait for socket to open
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

--[[
    Main Frame Callback Logic
--]]
local frame_counter = 0
