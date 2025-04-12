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

-- Define the server address as a global variable
local SERVER_ADDRESS = "socket.localhost:9999"

-- Add the Scripts directory to Lua's package path
package.path = package.path .. ";/Users/dave/source/repos/tempest_ai/Scripts/?.lua"
-- Now require the module by name only (without path or extension)

-- Define constants
local INVALID_SEGMENT = -32768  -- Used as sentinel value for invalid segments

local AUTO_START_GAME         = true -- flag to control auto-starting during attract mode

-- Timer and FPS tracking variables (Initialize globally)
local lastTimerValue = 0 
local lastFPSTime = os.time()
local frameCountSinceLastFPS = 0 

-- Load the display module
local Display = require("display")

-- Global socket variable
local socket = nil

-- Add this near the top of the file with other global variables
local frame_count = 0  -- Initialize frame counter

-- Global variables for tracking bytes sent and FPS
local total_bytes_sent = 0

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
        socket = emu.file("rw")  -- "rw" mode for read/write
        -- Use the global SERVER_ADDRESS variable
        local result = socket:open(SERVER_ADDRESS) 
        
        if result == nil then
            -- Update print statement to reflect the actual address used
            print("Successfully opened socket connection to " .. SERVER_ADDRESS) 
            
            -- Send initial 4-byte ping for handshake
            local ping_data = string.pack(">H", 0)  -- 2-byte integer with value 0
            socket:write(ping_data)
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
local previous_alive_state = 1  -- Track previous alive state, initialize as alive

-- Declare a global variable to store the last reward state
local LastRewardState = 0

local shutdown_requested = false

local last_display_update = 0  -- Timestamp of last display update

-- Function to calculate reward for the current frame
local function calculate_reward(game_state, level_state, player_state, enemies_state, commanded_spinner)
    local reward = 0
    local bDone = false

    -- We want as many shots active as possible, but only up to 7, so that we have one in reserve for close enemies
    
    if (player_state.shot_count < 8) then
        reward = reward + player_state.shot_count
    end

    -- Base survival reward - make staying alive more valuable
    
    if player_state.alive == 1 then

        -- Stronger reward for maintaining lives
        if player_state.player_lives ~= nil then
            reward = reward + (player_state.player_lives)
        end

        -- Score-based reward (keep this as a strong motivator).  Filter out large bonus awards.
        local score_delta = player_state.score - previous_score
        if score_delta > 0 and score_delta < 5000 then
            reward = reward + (score_delta * 5)  -- Amplify score impact
        end

        -- Level completion bonus removed since it would come after end of the episode
        -- if level_state.level_number ~= previous_level then
        --     reward = reward + (500 * previous_level)  -- Increased bonus for progression
        --     bDone = true
        -- end

        -- Penalize using superzapper; only in play mode, since it's also set during zoom (0x020)

        if (game_state.gamestate == 0x04) then
            if (player_state.superzapper_active ~= 0) then
                reward = reward - 250
            end
        end
                
        -- Enemy targeting logic
        local target_segment = enemies_state.nearest_enemy_seg 
        local player_segment = player_state.position & 0x0F

        -- Check to see if any enemy shots are active in our lane, and if so check the distance.  If it's less than or equal to 0x24
        -- reward the player for moving out of the way of the enemy shot.

        for i = 1, 4 do
            if (enemies_state.enemy_shot_segments[i].value == player_segment) then
                if (enemies_state.shot_positions[i] <= 0x24) then
                    -- BUGBUG we should really check to see the commanded_spinner is in a direction
                    --        that would have avoided the shot, but this is a good start.
                    if (commanded_spinner == 0) then 
                        reward = reward - 100
                    else
                        reward = reward + commanded_spinner * 100
                    end
                end
            end
        end

        if target_segment == INVALID_SEGMENT or game_state.gamestate == 0x20 then 
            -- No enemies: reward staying still more strongly
            -- Use commanded_spinner here to check if model *intended* to stay still
            reward = reward + (commanded_spinner == 0 and 50 or -20)
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
                -- REMOVED penalty for movement when aligned; the alignment bonus incentivizes staying put.
            else 
                -- MISALIGNED CASE (segment_distance > 0)
                -- Enemies at the top of tube should be shot when close (using segment distance)
                if (segment_distance < 2) then -- Check using actual segment distance
                    -- Use the depth returned by direction_to_nearest_enemy
                    if (enemy_depth <= 0x20) then 
                        if player_state.fire_commanded == 1 then
                            -- Strong reward for firing at close enemies
                            reward = reward + 250
                        else
                            -- Moderate penalty for not firing at close enemies
                            reward = reward - 50
                        end
                    end
                end

                -- Graduated reward for proximity (higher reward for smaller segment distance)
                reward = reward + (10 - segment_distance) -- Simple linear reward for proximity
                
                -- Movement incentives (using desired_spinner direction and commanded_spinner)
                -- Penalize if the COMMANDED movement (commanded_spinner) is OPPOSITE to the desired direction.
                if desired_spinner * commanded_spinner < 0 then
                    -- Strong penalty for moving AWAY from the target.
                    reward = reward - 50
                -- No explicit reward for moving towards, let proximity handle that.
                -- No penalty for staying still when misaligned (proximity reward decreases naturally).
                end
                
                -- Encourage maintaining shots in reserve
                if player_state.shot_count < 2 or player_state.shot_count > 7 then
                    reward = reward - 20  -- Penalty for not having shots ready
                elseif player_state.shot_count >= 5 then
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
    -- Increment frame count after processing
    frame_count = frame_count + 1

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

-- Use the global SERVER_ADDRESS variable for the print statement
print("Connecting to server at " .. SERVER_ADDRESS) 

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

-- If the primary access method failed, print the error and stop the script.
if not success then
    print("Error accessing MAME machine via manager: " .. tostring(err))
    print("FATAL: Cannot access MAME memory")
    return -- Stop script execution
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
    self.gamestate = 0                  -- Game state from address 0
    self.game_mode = 0                  -- Game mode from address 5
    self.countdown_timer = 0            -- Countdown timer from address 4
    self.frame_counter = 0              -- Frame counter for tracking progress
    self.last_save_time = os.time()     -- Track when we last sent save signal
    self.save_interval = 300            -- Send save signal every 5 minutes (300 seconds)
    
    -- FPS tracking (now handled at global level, not in GameState)
    self.current_fps = 0                -- Store the FPS value for display
    
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

-- **LevelState Class**
LevelState = {}
LevelState.__index = LevelState

function LevelState:new()
    local self = setmetatable({}, LevelState)
    self.level_number = 0
    self.spike_heights = {}  -- Array of 16 spike heights
    self.is_open_level = false -- NEW field to track open level status
    self.level_angles = {}  -- Array of 16 tube angles
    self.level_shape = 0    -- Level shape (level_number % 16)
    return self
end

function LevelState:update(mem)
    self.level_number = mem:read_u8(0x009F)  -- Example address for level number
    self.is_open_level = (mem:read_u8(0x0111) == 0xFF) -- Update the boolean field directly
    self.level_shape = self.level_number % 16  -- Calculate level shape
    
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

-- Update PlayerState:update to accept level_state
function PlayerState:update(mem, level_state)
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
    -- Use level_state.is_open_level instead of reading memory again
    local is_open = level_state.is_open_level
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
                 -- Pass is_open from level_state
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
    -- NEW Engineered Features for Targeting/Aiming
    self.is_aligned_with_nearest = 0.0
    self.nearest_enemy_depth_raw = 255 -- Sentinel value (max depth)
    self.alignment_error_magnitude = 0.0

    -- NEW: Array to track charging fuseballs by absolute segment (0-15 -> index 1-16)
    self.charging_fuseball_segments = {}

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

-- Update EnemiesState:update to accept level_state
function EnemiesState:update(mem, level_state)
    -- First, initialize/reset all arrays at the beginning
    -- Reset enemy arrays
    -- Keep resetting original raw arrays
    self.enemy_type_info        = {0, 0, 0, 0, 0, 0, 0}
    self.active_enemy_info      = {0, 0, 0, 0, 0, 0, 0}
    self.enemy_segments         = {0, 0, 0, 0, 0, 0, 0}
    self.enemy_depths           = {0, 0, 0, 0, 0, 0, 0}
    self.enemy_depths_lsb       = {0, 0, 0, 0, 0, 0, 0}
    self.enemy_shot_lsb         = {0, 0, 0, 0, 0, 0, 0}
    self.enemy_move_vectors     = {0, 0, 0, 0, 0, 0, 0} -- Keep this reset if used elsewhere
    self.enemy_state_flags      = {0, 0, 0, 0, 0, 0, 0}  -- Keep this reset if used elsewhere
    -- NEW: Reset decoded tables
    self.enemy_core_type        = {0, 0, 0, 0, 0, 0, 0}
    self.enemy_direction_moving = {0, 0, 0, 0, 0, 0, 0}
    self.enemy_between_segments = {0, 0, 0, 0, 0, 0, 0}
    self.enemy_moving_away      = {0, 0, 0, 0, 0, 0, 0}
    self.enemy_can_shoot        = {0, 0, 0, 0, 0, 0, 0}
    self.enemy_split_behavior   = {0, 0, 0, 0, 0, 0, 0}

    -- Read active enemies counts, pulse state, etc.
    self.active_flippers        = mem:read_u8(0x0142)       -- n_flippers - current active count
    self.active_pulsars         = mem:read_u8(0x0143)       -- n_pulsars
    self.active_tankers         = mem:read_u8(0x0144)       -- n_tankers
    self.active_spikers         = mem:read_u8(0x0145)       -- n_spikers
    self.active_fuseballs       = mem:read_u8(0x0146)       -- n_fuseballs
    self.pulse_beat             = mem:read_u8(0x0147)       -- pulse_beat counter
    self.pulsing                = mem:read_u8(0x0148)       -- pulsing state
    self.pulsar_fliprate        = mem:read_u8(0x00B2)       -- pulsar flip rate
    self.num_enemies_in_tube    = mem:read_u8(0x0108)
    self.num_enemies_on_top     = mem:read_u8(0x0109)
    self.enemies_pending        = mem:read_u8(0x03AB)

    -- Update enemy shot segments from memory (store relative distances)
    local player_abs_segment = mem:read_u8(0x0200) & 0x0F -- Get current player absolute segment
    -- Use level_state.is_open_level
    local is_open = level_state.is_open_level

    for i = 1, 4 do
        local abs_segment = mem:read_u8(self.enemy_shot_segments[i].address)
        if abs_segment == 0 then
            self.enemy_shot_segments[i].value = INVALID_SEGMENT  -- Not active, use sentinel
        else
            local segment = abs_segment & 0x0F  -- Mask to ensure 0-15
            -- Pass is_open from level_state
            self.enemy_shot_segments[i].value = absolute_to_relative_segment(player_abs_segment, segment, is_open)
        end
    end

    -- Get player position and level type for relative calculations
    local player_pos = mem:read_u8(0x0200)
    -- Use level_state.is_open_level
    -- local is_open = mem:read_u8(0x0111) == 0xFF -- No longer needed

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
            -- Pass is_open from level_state
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
        self.shot_positions[i] = raw_pos  -- Store full raw position value

        -- Invalidate the segment values for any shots that are zeroed
        if (self.shot_positions[i] == 0) then
            self.enemy_shot_segments[i].value = INVALID_SEGMENT
        end
    end

    -- Read pending_seg (64 bytes starting at 0x0203), store relative
    for i = 1, 64 do
        local abs_segment = mem:read_u8(0x0203 + i - 1)
        if abs_segment == 0 then
            self.pending_seg[i] = INVALID_SEGMENT  -- Not active, use sentinel
        else
            local segment = abs_segment & 0x0F  -- Mask to ensure 0-15
            -- Store relative segment distance
            -- Pass is_open from level_state
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
            -- Pass is_open from level_state
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
    -- Pass level_state to nearest_enemy_segment
    local nearest_abs_seg, nearest_depth = self:nearest_enemy_segment(level_state)
    if nearest_abs_seg == -1 then
        self.nearest_enemy_seg = INVALID_SEGMENT
        -- Set default values for engineered features when no enemy
        self.is_aligned_with_nearest = 0.0
        self.nearest_enemy_depth_raw = 255 -- Use max depth as sentinel
        self.alignment_error_magnitude = 0.0
    else
        -- Pass is_open from level_state
        local nearest_rel_seg = absolute_to_relative_segment(player_abs_segment, nearest_abs_seg, level_state.is_open_level)
        self.nearest_enemy_seg = nearest_rel_seg -- Store relative for internal use/display
        self.nearest_enemy_depth_raw = nearest_depth -- Store raw depth

        -- Calculate Is_Aligned
        self.is_aligned_with_nearest = (nearest_rel_seg == 0) and 1.0 or 0.0

        -- Calculate Alignment_Error_Magnitude (Scaled to 0-10000 for packing)
        local error_abs = math.abs(nearest_rel_seg)
        local normalized_error = 0.0
        -- Use level_state.is_open_level
        if level_state.is_open_level then
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

-- Find the *absolute* segment and depth of the enemy closest to the top of the tube
-- This is used primarily for OOB data packing and targeting decisions that need absolute values.
-- If multiple enemies share the minimum depth, it chooses the one closest in segment distance.
-- Update EnemiesState:nearest_enemy_segment to accept level_state
function EnemiesState:nearest_enemy_segment(level_state)
    -- Add check for nil level_state at the beginning of the function
    if not level_state then
        error("FATAL: level_state argument is nil inside EnemiesState:nearest_enemy_segment", 2) -- Level 2 error points to the caller
    end

    local min_depth = 255
    local closest_absolute_segment = -1 -- Use -1 as sentinel for *absolute* segment not found
    local min_relative_distance_abs = 17 -- Sentinel for minimum absolute relative distance (max possible is 15 or 8)

    -- Get player state needed for relative calculations
    local player_abs_segment = mem:read_u8(0x0200) & 0x0F
    -- Use level_state.is_open_level
    local is_open = level_state.is_open_level

    -- Check standard enemy table (7 slots)
    for i = 1, 7 do
        -- Read the absolute segment and depth directly from memory for this calculation
        local current_abs_segment = mem:read_u8(0x02B9 + i - 1) & 0x0F -- Mask to 0-15
        local current_depth = mem:read_u8(0x02DF + i - 1)

        -- Only consider active enemies with valid segments (0-15)
        if current_depth > 0 and current_abs_segment >= 0 and current_abs_segment <= 15 then
            -- Calculate relative distance for this enemy
            -- Pass is_open from level_state
            local current_relative_distance = absolute_to_relative_segment(player_abs_segment, current_abs_segment, is_open)
            local current_relative_distance_abs = math.abs(current_relative_distance)

            -- Priority 1: Closer depth always wins
            if current_depth < min_depth then
                min_depth = current_depth
                closest_absolute_segment = current_abs_segment
                min_relative_distance_abs = current_relative_distance_abs
            -- Priority 2: Same depth, closer segment wins
            elseif current_depth == min_depth then
                if current_relative_distance_abs < min_relative_distance_abs then
                    closest_absolute_segment = current_abs_segment
                    min_relative_distance_abs = current_relative_distance_abs
                end
            end
        end
    end

    -- Return the absolute segment (-1 if none found) and its depth
    return closest_absolute_segment, min_depth
end

function direction_to_nearest_enemy(game_state, level_state, player_state, enemies_state)
    -- Get the *absolute* segment AND depth of nearest enemy from the dedicated function
    -- Pass level_state
    local enemy_abs_seg, enemy_depth = enemies_state:nearest_enemy_segment(level_state)
    local player_abs_seg = player_state.position & 0x0F
    -- Use level_state.is_open_level
    local is_open = level_state.is_open_level

    -- If no enemy was found (absolute segment is -1)
    if enemy_abs_seg == -1 then
        return 0, 0, 255 -- No enemy, return spinner 0, distance 0, max depth
    end

    -- Calculate the relative segment distance using the helper function
    -- Pass is_open from level_state
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
    -- Use is_open derived from level_state
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

    -- Get button ports
    self.button_port = manager.machine.ioport.ports[":BUTTONSP1"]
    
    -- Get spinner/dial port
    self.spinner_port = manager.machine.ioport.ports[":KNOBP1"]
    
    -- Set up button fields
    self.fire_field = self.button_port and self.button_port.fields["P1 Button 1"] or nil
    self.zap_field = self.button_port and self.button_port.fields["P1 Button 2"] or nil
    
    -- Set up spinner field
    self.spinner_field = self.spinner_port and self.spinner_port.fields["Dial"] or nil
    
    -- Track commanded states (consider if these are needed if player_state already tracks them)
    self.fire_commanded = 0
    self.zap_commanded = 0
    self.spinner_delta = 0

    -- Validation prints during initialization (can be removed later if desired)
    print("Button port found: " .. (self.button_port and "Yes" or "No"))
    if self.button_port then
        print("  Fire field found: " .. (self.fire_field and "Yes" or "No"))
        print("  Zap field found: " .. (self.zap_field and "Yes" or "No"))
    end
    print("Spinner port found: " .. (self.spinner_port and "Yes" or "No"))
    if self.spinner_port then
        print("  Dial field found: " .. (self.spinner_field and "Yes" or "No"))
    end
    
    return self
end

-- Controls:apply_action: Use player_state directly for inferred values
function Controls:apply_action(fire, zap, spinner, game_state, player_state)
    -- Fix the attract mode check - bit 0x80 being CLEAR indicates attract mode
    local is_attract_mode = (game_state.game_mode & 0x80) == 0
    
    if is_attract_mode then
        -- In attract mode:
        -- 1. Set fire command in player_state (model doesn't control attract mode fire)
        player_state.fire_commanded = 1 
        if self.fire_field then self.fire_field:set_value(1) end
        
        -- 2. Ensure zap is off
        player_state.zap_commanded = 0
        if self.zap_field then self.zap_field:set_value(0) end

        -- 3. Spinner delta is inferred (game controls it), update player_state.SpinnerDelta
        player_state.SpinnerDelta = player_state.inferredSpinnerDelta 
        
    else
        -- In actual gameplay:
        -- Apply the model's fire, zap, and spinner values to the physical controls
        -- Update player_state with the commanded values
        player_state.fire_commanded = fire
        player_state.zap_commanded = zap
        player_state.SpinnerDelta = spinner -- This is the commanded spinner from Python

        if self.fire_field then self.fire_field:set_value(fire) end
        if self.zap_field then self.zap_field:set_value(zap) end
        
        -- Apply the model's spinner value to the game's spinner delta input
        if self.spinner_field then mem:write_u8(0x0050, spinner) end
    end
end

-- Instantiate state objects - AFTER defining all classes
local game_state = GameState:new()
local level_state = LevelState:new()
local player_state = PlayerState:new()
local enemies_state = EnemiesState:new()
local controls = Controls:new()

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
    -- Add NEW Engineered Features (Targeting/Aiming)
    table.insert(data, enemies_state.nearest_enemy_depth_raw) -- Raw depth (0-255)
    table.insert(data, enemies_state.is_aligned_with_nearest) -- Float (0.0 or 1.0)
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

    -- Level state (2 values + arrays) - Removed level_type
    table.insert(data, level_state.level_number)
    -- Remove level_type from data payload
    -- table.insert(data, level_state.level_type)
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
            table.insert(data, INVALID_SEGMENT) -- Use sentinel if not at depth 0x10
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
    local is_open_level = level_state.is_open_level

    -- Get the ABSOLUTE nearest enemy segment (-1 sentinel) and depth for OOB packing
    -- Pass level_state
    local nearest_abs_seg_oob, enemy_depth_oob = enemies_state:nearest_enemy_segment(level_state)
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
local function frame_callback()
    -- Check the time counter at address 0x0003
    local currentTimer = mem:read_u8(0x0003)
    
    -- Check if the timer changed
    if currentTimer == lastTimerValue then -- Use the global variable
        return true
    end
    lastTimerValue = currentTimer -- Update the global variable
    
    -- Increment frame count and track FPS
    frameCountSinceLastFPS = frameCountSinceLastFPS + 1 -- Use the global variable
    local currentTime = os.time()
    
    -- Calculate FPS every second
    if currentTime > lastFPSTime then -- Use the global variable
        -- Ensure division by zero doesn't happen if time hasn't advanced
        local time_diff = currentTime - lastFPSTime
        if time_diff > 0 then
            game_state.current_fps = frameCountSinceLastFPS / time_diff -- More accurate FPS
        else
            game_state.current_fps = 0 -- Or some default if time hasn't advanced
        end
        frameCountSinceLastFPS = 0 -- Reset the global variable
        lastFPSTime = currentTime -- Update the global variable
    end
    
    -- Update game state first
    game_state:update(mem)
    
    -- Update level state next (Crucial: sets level_state.is_open_level)
    level_state:update(mem) 
    
    -- Update player state, passing the updated level_state
    player_state:update(mem, level_state)
    
    -- Update enemies state last, passing the updated level_state
    enemies_state:update(mem, level_state)
    
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
    elseif game_state.gamestate == 0x16 then
        -- Game is in level select mode, advance selection 
        -- controls:apply_action(0, 0, 9, game_state, player_state)
        return true
    end

    -- Check if socket is open, try to reopen if not
    if not socket then
        if not open_socket() then
            -- If socket isn't open yet, display waiting message and try again next frame
            if Display.SHOW_DISPLAY then 
                Display.update_display("Waiting for Python connection...", game_state, level_state, player_state, enemies_state, nil, 0, nil, total_bytes_sent, LastRewardState)
            end
            -- Do NOT return true here. Let the callback attempt connection again next frame.
            -- The rest of the function (requiring the socket) will be skipped implicitly.
            return true -- Still need to return true to MAME, but skip processing this frame's data
        end
        -- If open_socket succeeded, socket is now not nil, and we proceed.
    end

    -- If we reach here, the socket should be open.

    -- Declare num_values (will be set by flatten_game_state_to_binary)
    local num_values = 0 
    local bDone = false

    -- 2 Credits (Consider if this needs to be set every frame)
    mem:write_u8(0x0006, 2)

    -- Reset the countdown timer to zero all the time (Consider if this needs to be set every frame)
    mem:write_u8(0x0004, 0)

    -- NOP out the jump that skips scoring in attract mode (Can likely be done once at init)
    mem:write_direct_u8(0xCA6F, 0xEA)
    mem:write_direct_u8(0xCA70, 0xEA)
    
    -- NOP out the damage the copy protection code does to memory when it detects a bad checksum (Can likely be done once at init)
    mem:write_direct_u8(0xA591, 0xEA)
    mem:write_direct_u8(0xA592, 0xEA)

    -- Initialize action to "none" as default (Redundant if 'action' var isn't used before assignment below?)
    -- local action = "none" 
    local status_message = ""

    -- Massage game state to keep it out of the high score and banner modes
    if game_state.countdown_timer > 0 then
        -- Write 0 to memory location 4
        mem:write_u8(0x0004, 0)
        game_state.countdown_timer = 0
    end

    local is_attract_mode = (game_state.game_mode & 0x80) == 0

    -- Handle Attract Mode (P1 Start Button, Score Reset)
    if is_attract_mode then
        -- Zero score if dead or zooming
        if game_state.gamestate == 0x06 or game_state.gamestate == 0x20 then
            mem:write_direct_u8(0x0040, 0x00)
            mem:write_direct_u8(0x0041, 0x00)
            mem:write_direct_u8(0x0042, 0x00)
        end
        -- Auto-start logic
        if AUTO_START_GAME then 
            local port = manager.machine.ioport.ports[":IN2"]
            if port then -- Check if port exists
                local startField = port.fields["1 Player Start"] or 
                                   port.fields["P1 Start"] or 
                                   port.fields["Start 1"]
                
                if startField then
                    if game_state.frame_counter % 60 == 0 then
                        startField:set_value(1)
                    elseif game_state.frame_counter % 60 == 5 then
                        startField:set_value(0)
                    end
                else
                     print("Error: Could not find start button field in :IN2")
                end
            else
                print("Error: Could not find port :IN2")
            end
        end
        -- In attract mode, we don't need to calculate reward or get actions from Python
        -- We can potentially return early after handling display updates
        
    else 
        -- Gameplay Mode Logic
        -- Calculate reward based on previous action's effect
        local reward, bDoneCurrent = calculate_reward(game_state, level_state, player_state, enemies_state, player_state.inferredSpinnerDelta)
        bDone = bDoneCurrent -- Update the main bDone flag

        -- Flatten and serialize the game state data
        local frame_data
        frame_data, num_values = flatten_game_state_to_binary(reward, game_state, level_state, player_state, enemies_state, bDone)

        -- Send state and get action from Python
        local fire, zap, spinner = process_frame(frame_data, player_state, controls, reward, bDone, is_attract_mode)

        -- Update player_state with commanded actions
        player_state.fire_commanded = fire
        player_state.zap_commanded = zap
        player_state.SpinnerDelta = spinner 

        -- Update total bytes sent
        total_bytes_sent = total_bytes_sent + #frame_data

        -- Apply actions only in specific gameplay states
        if game_state.gamestate == 0x04 or game_state.gamestate == 0x20 or game_state.gamestate == 0x24 then
            controls:apply_action(fire, zap, spinner, game_state, player_state)
        end
    end

    -- Update Display (if enabled and interval met)
    local current_time_high_res = os.clock()
    local should_update_display = (current_time_high_res - last_display_update) >= Display.DISPLAY_UPDATE_INTERVAL 
    if should_update_display and Display.SHOW_DISPLAY then 
        -- The 'action' variable was never assigned, using placeholder 'N/A'
        Display.update_display(status_message, game_state, level_state, player_state, enemies_state, "N/A", num_values, LastRewardState, total_bytes_sent, LastRewardState) 
        last_display_update = current_time_high_res
    end

    return true -- Always return true to MAME to continue emulation
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
        -- Pass 0 for commanded_spinner as it's shutdown
        local reward = calculate_reward(game_state, level_state, player_state, enemies_state, 0) 
        
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
