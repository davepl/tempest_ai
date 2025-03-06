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
    return self
end

function GameState:update(mem)
    self.credits = mem:read_u8(0x0006)  -- Example address for credits
    self.p1_level = mem:read_u8(0x00B9)  -- Player 1 level
    self.p1_lives = mem:read_u8(0x00A1)  -- Player 1 lives
end

-- **LevelState Class**
LevelState = {}
LevelState.__index = LevelState

function LevelState:new()
    local self = setmetatable({}, LevelState)
    self.level_number = 0
    return self
end

function LevelState:update(mem)
    self.level_number = mem:read_u8(0x0007)  -- Example address for level number
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
    self.superzapper_available = 0
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

    self.angle = mem:read_u8(0x00B0)            -- Player angle/orientation
    self.superzapper_available = mem:read_u8(0x00C0)  -- Superzapper availability
end

-- **EnemiesState Class**
EnemiesState = {}
EnemiesState.__index = EnemiesState

function EnemiesState:new()
    local self = setmetatable({}, EnemiesState)
    self.active_flippers = 0
    self.active_pulsars = 0
    self.active_tankers = 0
    self.active_spikers = 0
    self.active_fuseballs = 0
    return self
end

function EnemiesState:update(mem)
    self.active_flippers = mem:read_u8(0x0142)   -- Example addresses for active enemies
    self.active_pulsars = mem:read_u8(0x0143)
    self.active_tankers = mem:read_u8(0x0144)
    self.active_spikers = mem:read_u8(0x0145)
    self.active_fuseballs = mem:read_u8(0x0146)
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

-- Frame callback function
local function frame_callback()
    -- Update all state objects
    game_state:update(mem)
    level_state:update(mem)
    player_state:update(mem)
    enemies_state:update(mem)

    -- Calculate total active enemies
    local total_active_enemies = enemies_state:get_total_active()

    -- Randomly select an action
    local actions = {"fire", "zap", "left", "right", "none"}
    local action = actions[math.random(#actions)]

    -- Print a terse line with key stats and the action
    print(string.format("Credits: %d, Level: %d, Player Pos: %d, Alive: %d, Score: %d, Angle: %d, Superzapper: %d, Active Enemies: %d, Action: %s",
        game_state.credits,
        level_state.level_number,
        player_state.position,
        player_state.alive,
        player_state.score,
        player_state.angle,
        player_state.superzapper_available,
        total_active_enemies,
        action
    ))

    -- Apply the action to MAME controls
    controls:apply_action(action)
end

-- Register the frame callback with MAME
emu.register_frame(frame_callback)