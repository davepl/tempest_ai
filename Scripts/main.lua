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

local function clear_screen()
    io.write("\027[2J\027[H")
end

-- Global pipe variables
local pipe_out = nil
local pipe_in = nil

-- Function to open pipes (called once at startup)
local function open_pipes()
    -- Close existing pipes if they're open
    if pipe_out then pipe_out:close(); pipe_out = nil end
    if pipe_in then pipe_in:close(); pipe_in = nil end
    
    print("Attempting to open pipes...")
    
    -- Try to open the pipes directly without checking existence first
    -- First open the output pipe (this is the one Python should be reading from)
    print("Opening output pipe (lua_to_py)...")
    local open_success, err = pcall(function()
        pipe_out = io.open("/tmp/lua_to_py", "wb")
    end)
    
    if not open_success or not pipe_out then
        print("Failed to open output pipe: " .. tostring(err))
        return false
    end
    print("Output pipe opened successfully")
    
    -- Then open input pipe (this is the one Python should be writing to)
    print("Opening input pipe (py_to_lua)...")
    open_success, err = pcall(function()
        pipe_in = io.open("/tmp/py_to_lua", "r")
    end)
    
    if not open_success or not pipe_in then
        print("Failed to open input pipe: " .. tostring(err))
        if pipe_out then pipe_out:close(); pipe_out = nil end
        return false
    end
    print("Input pipe opened successfully")
    
    print("Successfully opened both pipes")
    return true
end

-- Create a variable to track pipe retry attempts
local pipe_retry_count = 0

-- Function to send parameters and get action each frame
local function process_frame(params)
    -- Check if pipes are open, try to reopen if not
    if not pipe_out or not pipe_in then
        pipe_retry_count = pipe_retry_count + 1
        
        -- Only try to open pipes every 60 frames (about once per second)
        if pipe_retry_count % 60 == 0 then
            print("Attempt #" .. pipe_retry_count / 60 .. " to open pipes")
            if not open_pipes() then
                return "pipe error"
            end
        else
            return "pipe error"
        end
    end
    
    -- Reset retry count once pipes are open
    pipe_retry_count = 0
    
    -- Try to write to pipe, handle errors
    local success, err = pcall(function()
        pipe_out:write(params)
        pipe_out:flush()
    end)
    
    if not success then
        print("Error writing to pipe:", err)
        open_pipes()  -- Try to recover
        return "write error"
    end
    
    -- Try to read from pipe, handle errors
    local action = nil
    success, err = pcall(function()
        action = pipe_in:read("*line")
    end)
    
    if not success then
        print("Error reading from pipe:", err)
        open_pipes()  -- Try to recover
        return "read error"
    end
    
    return action or "no response"
end

-- Call this during initialization
local function start_python_script()
    print("Starting Python script")
    
    -- Kill any existing Python script instances
    os.execute("pkill -f 'python.*aimodel.py' 2>/dev/null")
    
    -- Remove existing pipes to ensure clean state
    os.execute("rm -f /tmp/lua_to_py /tmp/py_to_lua")
    
    -- Launch Python script in the background with proper error handling
    local cmd = "python /Users/dave/source/repos/tempest/Scripts/aimodel.py >/tmp/python_output.log 2>&1 &"
    local result = os.execute(cmd)
    
    if result ~= 0 then
        print("Warning: Failed to start Python script")
        return false
    end
    
    -- Give Python script a moment to start up and create pipes
    print("Waiting for Python script to initialize and create pipes...")
    os.execute("sleep 3")
    
    -- Check if Python script is running
    local python_running = os.execute("pgrep -f 'python.*aimodel.py' >/dev/null") == 0
    if not python_running then
        print("Warning: Python script failed to start or terminated early")
        print("Check /tmp/python_output.log for errors")
        return false
    end
    
    print("Python script started successfully")
    
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
    return self
end

function GameState:update(mem)
    self.credits = mem:read_u8(0x0006)  -- Example address for credits
    self.p1_level = mem:read_u8(0x0046)  -- Player 1 level
    self.p1_lives = mem:read_u8(0x0048)  -- Player 1 lives
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
    return self
end

function LevelState:update(mem)
    self.level_number = mem:read_u8(0x009F)  -- Example address for level number
    self.level_type = mem:read_u8(0x0111)    -- Level type (00=closed, FF=open)
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
    self.angle = 0
    self.superzapper_uses = 0
    self.superzapper_active = 0
    self.shot_segments = {0, 0, 0, 0, 0, 0, 0, 0}  -- 8 shot segments
    self.shot_positions = {0, 0, 0, 0, 0, 0, 0, 0}  -- 8 shot positions (depth)
    self.shot_count = 0
    return self
end

function PlayerState:update(mem)
    self.position = mem:read_u8(0x0200)          -- Player position
    local status_flags = mem:read_u8(0x0005)     -- Status flags
    self.alive = (status_flags & 0x40) ~= 0 and 1 or 0  -- Bit 6 for alive status

    local function bcd_to_decimal(bcd)
        return ((bcd >> 4) * 10) + (bcd & 0x0F)
    end

    local score_low = bcd_to_decimal(mem:read_u8(0x0040))
    local score_mid = bcd_to_decimal(mem:read_u8(0x0041))
    local score_high = bcd_to_decimal(mem:read_u8(0x0042))
    self.score = score_high * 10000 + score_mid * 100 + score_low

    self.angle = mem:read_u8(0x03EE)                        -- Player angle/orientation
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
    self.port = manager.machine.ioport.ports[":IN0"]
    self.fire_field = self.port and self.port.fields["Fire"] or nil
    self.zap_field = self.port and self.port.fields["Superzapper"] or nil
    self.left_field = self.port and self.port.fields["Left"] or nil
    self.right_field = self.port and self.port.fields["Right"] or nil
    return self
end

function Controls:apply_action(action)
    -- Reset all controls to 0
    if self.fire_field then self.fire_field:set_value(0) end
    if self.zap_field then self.zap_field:set_value(0) end
    if self.left_field then self.left_field:set_value(0) end
    if self.right_field then self.right_field:set_value(0) end

    -- Apply the selected action
    if action == "fire" and self.fire_field then
        self.fire_field:set_value(1)
    elseif action == "zap" and self.zap_field then
        self.zap_field:set_value(1)
    elseif action == "left" and self.left_field then
        self.left_field:set_value(1)
    elseif action == "right" and self.right_field then
        self.right_field:set_value(1)
    end
    -- "none" results in no inputs being set to 1
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

function update_display(status, game_state, level_state, player_state, enemies_state, current_action)
    -- Clear screen
    -- clear_screen()

    -- Move cursor to row 0 and print status
    move_cursor_to_row(0)
    print(status)

    -- Format and print game state at row 1
    move_cursor_to_row(1)
    local game_metrics = {
        ["Credits"] = game_state.credits,
        ["P1 Lives"] = game_state.p1_lives,
        ["P1 Level"] = game_state.p1_level
    }
    print(format_section("Game State", game_metrics))

    -- Format and print player state at row 5
    move_cursor_to_row(5)
    local player_metrics = {
        ["Position"] = player_state.position,
        ["Alive"] = player_state.alive,
        ["Score"] = player_state.score,
        ["Angle"] = player_state.angle,
        ["Superzapper Uses"] = player_state.superzapper_uses,
        ["Superzapper Active"] = player_state.superzapper_active,
        ["Shot Count"] = player_state.shot_count,
        ["Shot Positions"] = (function()
            local shots = {}
            for i = 1, 8 do
                local segment_str = string.format("%+02d", player_state.shot_segments[i])
                shots[i] = string.format("%02X-%s", player_state.shot_positions[i], segment_str)
            end
            return table.concat(shots, " ")
        end)()
    }
    print(format_section("Player State", player_metrics))

    -- Format and print player controls at row 13
    move_cursor_to_row(13)
    local controls_metrics = {
        ["Current Action"] = current_action or "none",
        ["Fire"] = (current_action == "fire") and "ACTIVE" or "inactive",
        ["Superzapper"] = (current_action == "zap") and "ACTIVE" or "inactive",
        ["Left"] = (current_action == "left") and "ACTIVE" or "inactive",
        ["Right"] = (current_action == "right") and "ACTIVE" or "inactive"
    }
    print(format_section("Player Controls", controls_metrics))

    -- Format and print level state at row 20
    move_cursor_to_row(20)
    local level_metrics = {
        ["Level Number"] = level_state.level_number,
        ["Level Type"] = level_state.level_type == 0xFF and "Open" or "Closed",
        ["Spike Heights"] = (function()
            local heights = {}
            local positions = {}
            for pos, height in pairs(level_state.spike_heights) do
                table.insert(positions, pos)
            end
            table.sort(positions)
            for _, pos in ipairs(positions) do
                table.insert(heights, string.format("%+2d:%02X", pos, level_state.spike_heights[pos]))
            end
            return table.concat(heights, " ")
        end)(),
        ["Level Angles"] = (function()
            local angles = {}
            for i = 0, 15 do
                table.insert(angles, string.format("%02X", level_state.level_angles[i]))
            end
            return table.concat(angles, " ")
        end)()
    }
    print(format_section("Level State", level_metrics))

    -- Format and print enemies state at row 30
    move_cursor_to_row(30)
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
        ["Enemy States"] = table.concat(enemy_states, " "),
        ["Enemy Segments"] = table.concat(enemy_segs, " "),
        ["Enemy Depths"] = table.concat(enemy_depths, " "),
        ["Shot Positions"] = (function()
            local shots = {}
            for i = 1, 4 do
                if enemies_state.shot_positions[i] then
                    table.insert(shots, string.format("%+d", enemies_state.shot_positions[i]))
                end
            end
            return #shots > 0 and table.concat(shots, " ") or "none"
        end)()
    }

    print(format_section("Enemies State", enemies_metrics))
end

-- Function to move the cursor to the home position using ANSI escape codes
local function move_cursor_home()
    io.write("\027[H")
end

-- Function to flatten and serialize the game state data to signed 16-bit integers
local function flatten_game_state_to_binary(game_state, level_state, player_state, enemies_state)
    local data = {
        credits = game_state.credits,
        p1_lives = game_state.p1_lives,
        p1_level = game_state.p1_level,
        player_position = player_state.position,
        player_alive = player_state.alive,
        score_low = player_state.score & 0xFFFF,  -- Lower 16 bits of score
        score_high = (player_state.score >> 16) & 0xFFFF,  -- Upper 16 bits of score
        player_angle = player_state.angle,
        superzapper_uses = player_state.superzapper_uses,
        superzapper_active = player_state.superzapper_active,
        shot_count = player_state.shot_count,
        shot_positions = player_state.shot_positions,
        shot_segments = player_state.shot_segments,
        level_number = level_state.level_number,
        level_type = level_state.level_type,
        spike_heights = level_state.spike_heights,
        level_angles = level_state.level_angles,  -- Add level angles to the data
        active_flippers = enemies_state.active_flippers,
        active_pulsars = enemies_state.active_pulsars,
        active_tankers = enemies_state.active_tankers,
        active_spikers = enemies_state.active_spikers,
        active_fuseballs = enemies_state.active_fuseballs,
        spawn_slots_flippers = enemies_state.spawn_slots_flippers,
        spawn_slots_pulsars = enemies_state.spawn_slots_pulsars,
        spawn_slots_tankers = enemies_state.spawn_slots_tankers,
        spawn_slots_spikers = enemies_state.spawn_slots_spikers,
        spawn_slots_fuseballs = enemies_state.spawn_slots_fuseballs,
        enemy_segments = enemies_state.enemy_segments,
        shot_positions_enemy = enemies_state.shot_positions
    }

    -- Serialize the data to a binary string
    local binary_data = ""
    for name, value in pairs(data) do
        if type(value) == "number" then
            binary_data = binary_data .. string.pack(">i2", value)  -- Pack as signed 16-bit integer
        elseif type(value) == "table" then
            for i, v in ipairs(value) do
                binary_data = binary_data .. string.pack(">i2", v)  -- Pack as signed 16-bit integer
            end
        else
            print("Non-numeric value found:", name, "value:", value)  -- Debugging output
        end
    end

    -- Handle enemy_depths separately as it contains nested tables
    for i, depth in ipairs(enemies_state.enemy_depths) do
        if type(depth) == "table" then
            binary_data = binary_data .. string.pack(">i2", depth.pos)  -- Pack main position
            binary_data = binary_data .. string.pack(">i2", depth.frac) -- Pack fractional part
        else
            print("Unexpected non-table value in enemy_depths:", depth)
        end
    end

    return binary_data
end

-- Update the frame callback function
local function frame_callback()
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
            update_display("Waiting for Python connection...", game_state, level_state, player_state, enemies_state, "none")
            return
        end
    end
    
    -- Clear the screen at the start of each frame
    move_cursor_home()

    -- Update all state objects
    game_state:update(mem)
    level_state:update(mem)
    player_state:update(mem)
    enemies_state:update(mem)

    -- Flatten and serialize the game state data
    local frame_data = flatten_game_state_to_binary(game_state, level_state, player_state, enemies_state)

    -- Send the serialized data to the Python script
    local result = process_frame(frame_data)
    
    -- Use the action from Python if valid, otherwise use random
    local action = result
    if not (action == "fire" or action == "zap" or action == "left" or action == "right" or action == "none") then
        -- If Python returns an invalid action or error, use random
        local actions = {"fire", "zap", "left", "right", "none"}
        action = actions[math.random(#actions)]
    end

    -- Update the display with the current action
    update_display(result, game_state, level_state, player_state, enemies_state, action)

    -- Apply the action to MAME controls
    controls:apply_action(action)
end

-- Start the Python script but don't wait for pipes to open
start_python_script()
clear_screen()

-- Register the frame callback with MAME
emu.register_frame(frame_callback)

local function cleanup()
    if pipe_out then pipe_out:close() end
    if pipe_in then pipe_in:close() end
    print("Pipes closed")
end

-- Register cleanup with MAME's exit handler if available
emu.register_stop(cleanup)

