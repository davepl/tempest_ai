--[[
    state.lua
    Contains class definitions for game state objects (Game, Level, Player, Enemies)
    and related helper functions for the Tempest AI project.
--]]

local M = {} -- Module table to hold exported classes and functions
local helpers = require("helpers")

-- Define constants
local INVALID_SEGMENT = -32768 -- Used as sentinel value for invalid segments
M.INVALID_SEGMENT = INVALID_SEGMENT -- Export if needed elsewhere

-- Correct Enemy Type Constants (from assembly)
local ENEMY_TYPE_FLIPPER  = 0
local ENEMY_TYPE_PULSAR   = 1
local ENEMY_TYPE_TANKER   = 2
local ENEMY_TYPE_SPIKER   = 3
local ENEMY_TYPE_FUSEBALL = 4
local ENEMY_TYPE_MASK = 0x07 -- <<< ADDED THIS DEFINITION (%00000111)

-- Helper function for BCD conversion (local to this module)
local function bcd_to_decimal(bcd)
    -- Handle potential non-number inputs gracefully
    if type(bcd) ~= 'number' then return 0 end
    return math.floor(((bcd / 16) % 16) * 10 + (bcd % 16))
end

-- Forward declarations for helper functions used within classes/other helpers
-- Note: abs_to_rel_func will be passed in from main.lua where needed.
local find_nearest_enemy_of_type
local hunt_enemies
local find_target_segment
local find_forbidden_segments
local find_nearest_safe_segment

-- ====================
-- Helper Functions (Internal to this State Module)
-- ====================



-- NEW Helper: Identify forbidden segments
find_forbidden_segments = function(enemies_state, level_state, player_state)
    local forbidden = {} -- Use a table as a set (keys are forbidden segments 0-15)
    local is_pulsing = enemies_state.pulsing ~= 0
    -- print(string.format("FIND_FORBIDDEN: Pulsing active = %s", tostring(is_pulsing))) -- DEBUG

    -- Check enemies
    for i = 1, 7 do
        local core_type = enemies_state.enemy_core_type[i]
        local abs_seg = enemies_state.enemy_abs_segments[i]
        local depth = enemies_state.enemy_depths[i]

        if abs_seg ~= INVALID_SEGMENT and depth > 0 then
            -- 1. Pulsing Pulsar Lane (Check for Pulsar type 1)
            if core_type == ENEMY_TYPE_PULSAR and is_pulsing then -- <<< CORRECTED TYPE
                -- print(string.format("  -> FORBIDDEN (Pulsing Pulsar): Slot %d, Seg %d", i, abs_seg)) -- DEBUG
                forbidden[abs_seg] = true
            end
            -- 2. Top-level enemies (depth <= 0x20)
            -- Includes Flippers, Pulsars, Tankers, Fuseballs, Spikers if they are close
            if depth <= 0x20 then
                 -- print(string.format("  -> FORBIDDEN (Top Enemy): Slot %d, Type %d, Seg %d, Depth %02X", i, core_type, abs_seg, depth)) -- DEBUG
                 forbidden[abs_seg] = true
            end
        end
    end

    -- Check enemy shots
    local has_ammo = player_state.shot_count < 8 -- Check if player can shoot back
    -- print(string.format("FIND_FORBIDDEN: Player can shoot = %s (Shot count %d)", tostring(has_ammo), player_state.shot_count)) -- DEBUG
    for i = 1, 4 do
        local shot_abs_seg = enemies_state.enemy_shot_abs_segments[i]
        local shot_depth = enemies_state.shot_positions[i]
        -- Mark forbidden if shot is close AND player cannot shoot back
        if shot_abs_seg ~= INVALID_SEGMENT and shot_depth > 0 and shot_depth <= 0x20 and not has_ammo then
            -- print(string.format("  -> FORBIDDEN (Enemy Shot): Shot %d, Seg %d, Depth %02X", i, shot_abs_seg, shot_depth)) -- DEBUG
            forbidden[shot_abs_seg] = true
        end
    end
    return forbidden
end

-- NEW Helper: Find nearest safe segment if current is forbidden
find_nearest_safe_segment = function(player_abs_seg, is_open, forbidden_segments, abs_to_rel_func)
    -- Search outwards from the current segment
    local max_dist = is_open and 15 or 8
    for dist = 1, max_dist do
        -- Check Left
        local left_target_seg = -1
        if is_open then
            if player_abs_seg - dist >= 0 then left_target_seg = player_abs_seg - dist end
        else -- Closed
            left_target_seg = (player_abs_seg - dist + 16) % 16
        end

        if left_target_seg ~= -1 and not forbidden_segments[left_target_seg] then
            return left_target_seg -- Found safe segment to the left
        end

        -- Check Right
        local right_target_seg = -1
         if is_open then
            if player_abs_seg + dist <= 15 then right_target_seg = player_abs_seg + dist end
        else -- Closed
            right_target_seg = (player_abs_seg + dist) % 16
         end

         if right_target_seg ~= -1 and not forbidden_segments[right_target_seg] then
            return right_target_seg -- Found safe segment to the right
        end
    end

    -- If no safe segment found (highly unlikely unless all are forbidden)
    print("WARNING: No safe segment found to flee to! Staying put.")
    return player_abs_seg
end

-- Helper function to find nearest enemy of a specific type
-- Needs abs_to_rel_func, is_open, and forbidden_segments passed in
find_nearest_enemy_of_type = function(enemies_state, player_abs_segment, is_open, type_id, abs_to_rel_func, forbidden_segments)
    local nearest_seg_abs = -1
    local nearest_depth = 255
    local min_distance = 255 -- Use a large initial distance

    -- if type_id == ENEMY_TYPE_TANKER then print(string.format("FIND_NEAREST DEBUG: Hunting Tankers (Type %d)", type_id)) end -- DEBUG

    for i = 1, 7 do
        local core_type = enemies_state.enemy_core_type[i]
        local enemy_abs_seg = enemies_state.enemy_abs_segments[i]
        local enemy_depth = enemies_state.enemy_depths[i]
        local is_forbidden = (enemy_abs_seg ~= INVALID_SEGMENT) and forbidden_segments[enemy_abs_seg] or false

        --[[ DEBUG specific to Tankers
        if type_id == ENEMY_TYPE_TANKER then
            print(string.format(
                "  Slot %d: Type=%d, AbsSeg=%s, Depth=%02X, IsForbidden=%s, PassesChecks=%s",
                i, core_type, tostring(enemy_abs_seg), enemy_depth, tostring(is_forbidden),
                tostring(core_type == type_id and enemy_abs_seg ~= INVALID_SEGMENT and enemy_depth > 0x30 and not is_forbidden)
            ))
        end
        --]]

        -- Check if this is the enemy type we're looking for, if it's active,
        -- if its depth is > 0x30, AND if the segment is NOT forbidden
        if core_type == type_id and
           enemy_abs_seg ~= INVALID_SEGMENT and
           enemy_depth > 0x30 and
           not is_forbidden then

            -- Calculate distance using the provided function
            local rel_dist = abs_to_rel_func(player_abs_segment, enemy_abs_seg, is_open)
            local abs_dist = math.abs(rel_dist)

            -- Check if this enemy is closer than the current nearest
            if abs_dist < min_distance then
                min_distance = abs_dist
                nearest_seg_abs = enemy_abs_seg
                nearest_depth = enemy_depth
            -- Optional: Prioritize closer depth if distances are equal
            elseif abs_dist == min_distance and enemy_depth < nearest_depth then
                nearest_seg_abs = enemy_abs_seg
                nearest_depth = enemy_depth
            end
        end
    end

    return nearest_seg_abs, nearest_depth
end

-- Helper function to hunt enemies in preference order
-- Needs abs_to_rel_func, is_open, and forbidden_segments passed in
hunt_enemies = function(enemies_state, player_abs_segment, is_open, abs_to_rel_func, forbidden_segments)
    -- Corrected Hunt Order (based on assembly types): Fuseball(4), Pulsar(1), Tanker(2), Flipper(0), Spiker(3)
    local hunt_order = {
        ENEMY_TYPE_FUSEBALL, -- 4
        ENEMY_TYPE_PULSAR,   -- 1
        ENEMY_TYPE_TANKER,   -- 2
        ENEMY_TYPE_FLIPPER,  -- 0
        ENEMY_TYPE_SPIKER    -- 3
    }

    for _, enemy_type in ipairs(hunt_order) do
        local target_seg_abs, target_depth = find_nearest_enemy_of_type(enemies_state, player_abs_segment, is_open, enemy_type, abs_to_rel_func, forbidden_segments)
        if target_seg_abs ~= -1 then
            -- Check for Top Rail Flipper(0)/Pulsar(1) Avoidance
            if (enemy_type == ENEMY_TYPE_FLIPPER or enemy_type == ENEMY_TYPE_PULSAR) and target_depth <= 0x10 then -- <<< CORRECTED TYPE
                local rel_dist = abs_to_rel_func(player_abs_segment, target_seg_abs, is_open)
                if math.abs(rel_dist) <= 1 then -- If aligned or adjacent
                    local safe_adjacent_seg
                    if rel_dist <= 0 then -- Threat is left or aligned, move right
                        safe_adjacent_seg = (player_abs_segment + 1) % 16
                    else -- Threat is right, move left
                        safe_adjacent_seg = (player_abs_segment - 1 + 16) % 16
                    end
                    -- print(string.format("HUNT AVOID: Top Type %d at %d (Rel %d). Targeting adjacent safe %d", enemy_type, target_seg_abs, rel_dist, safe_adjacent_seg))
                    if forbidden_segments[safe_adjacent_seg] then
                         -- print("HUNT AVOID: Adjacent safe segment " .. safe_adjacent_seg .. " is forbidden! Staying put.")
                         return player_abs_segment, target_depth, true -- Stay put, but mark as avoiding
                    else
                         return safe_adjacent_seg, target_depth, true -- Target safe adjacent, mark as avoiding
                    end
                else
                    -- Top Flipper/Pulsar is not adjacent, target directly
                    return target_seg_abs, target_depth, false
                end
            else
                -- Not a top-rail Flipper/Pulsar, target directly
                return target_seg_abs, target_depth, false
            end
        end
    end

    -- If no enemies from the hunt order are found
    return -1, 255, false
end

-- Function to determine target segment, depth, and firing decision
find_target_segment = function(game_state, player_state, level_state, enemies_state, abs_to_rel_func, is_open)
    local logic = require("logic")
    return logic.find_target_segment(game_state, player_state, level_state, enemies_state, abs_to_rel_func)
end


-- ====================
-- GameState Class
-- ====================
M.GameState = {}
M.GameState.__index = M.GameState

function M.GameState:new()
    local self = setmetatable({}, M.GameState)
    self.credits = 0
    self.p1_level = 0
    self.p1_lives = 0
    self.gamestate = 0              -- Game state from address 0
    self.game_mode = 0              -- Game mode from address 5
    self.countdown_timer = 0        -- Countdown timer from address 4
    self.frame_counter = 0          -- Frame counter for tracking progress
    self.last_save_time = os.time() -- Track when we last sent save signal
    self.save_interval = 300        -- Send save signal every 5 minutes (300 seconds)
    self.start_delay = nil          -- Was used for random start delay in attract mode (currently unused)
    self.current_fps = 0            -- Store the calculated FPS value for display
    return self
end

function M.GameState:update(mem)
    self.gamestate = mem:read_u8(0x0000)        -- Game state at address 0
    self.game_mode = mem:read_u8(0x0005)        -- Game mode at address 5
    self.countdown_timer = mem:read_u8(0x0004)  -- Countdown timer from address 4
    self.credits = mem:read_u8(0x0006)          -- Credits
    self.p1_level = mem:read_u8(0x0046)         -- Player 1 level
    self.p1_lives = mem:read_u8(0x0048)         -- Player 1 lives
    self.frame_counter = self.frame_counter + 1 -- Increment frame counter
end

-- ====================
-- LevelState Class
-- ====================
M.LevelState = {}
M.LevelState.__index = M.LevelState

function M.LevelState:new()
    local self = setmetatable({}, M.LevelState)
    self.level_number = 0
    self.spike_heights = {} -- Array of 16 spike heights (0-15 index)
    self.level_type = 0     -- 00 = closed, FF = open (Read from memory, but might be unreliable)
    self.level_angles = {}  -- Array of 16 tube angles (0-15 index)
    self.level_shape = 0    -- Level shape (level_number % 16)
    -- Initialize tables
    for i = 0, 15 do
        self.spike_heights[i] = 0
        self.level_angles[i] = 0
    end
    return self
end

function M.LevelState:update(mem)
    self.level_number = mem:read_u8(0x009F)   -- Level number
    self.level_type = mem:read_u8(0x0111)     -- Level type (00=closed, FF=open)
    self.level_shape = self.level_number % 16 -- Calculate level shape

    -- Read spike heights for all 16 segments and store them indexed by absolute segment number (0-15)
    for i = 0, 15 do
        self.spike_heights[i] = mem:read_u8(0x03AC + i)
    end

    -- Read tube angles for all 16 segments indexed by absolute segment number (0-15)
    for i = 0, 15 do
        self.level_angles[i] = mem:read_u8(0x03EE + i)
    end
end

-- ====================
-- PlayerState Class
-- ====================
M.PlayerState = {}
M.PlayerState.__index = M.PlayerState

function M.PlayerState:new()
    local self = setmetatable({}, M.PlayerState)
    self.position = 0           -- Raw position byte from $0200
    self.alive = 0              -- 1 if alive, 0 if dead
    self.score = 0              -- Player score (decimal)
    self.superzapper_uses = 0
    self.superzapper_active = 0
    self.player_depth = 0       -- Player depth along the tube ($0202)
    self.player_state = 0       -- Player state byte from $0201
    self.shot_segments = {}     -- Table for 8 shot relative segments (1-8 index, or INVALID_SEGMENT)
    self.shot_positions = {}    -- Table for 8 shot positions/depths (1-8 index)
    self.shot_count = 0         -- Number of active shots ($0135)
    self.debounce = 0           -- Input debounce state ($004D)
    self.fire_detected = 0      -- Fire button state detected from debounce
    self.zap_detected = 0       -- Zap button state detected from debounce
    self.SpinnerAccum = 0       -- Raw spinner accumulator ($0051)
    self.prevSpinnerAccum = 0   -- Previous frame's spinner accumulator
    self.spinner_detected = 0   -- Calculated spinner delta based on accumulator change
    self.fire_commanded = 0     -- Fire action commanded by AI
    self.zap_commanded = 0      -- Zap action commanded by AI
    self.spinner_commanded = 0  -- Spinner action commanded by AI

    -- Initialize shot tables
    for i = 1, 8 do
        self.shot_segments[i] = INVALID_SEGMENT
        self.shot_positions[i] = 0
    end

    return self
end

-- PlayerState update needs the absolute_to_relative_segment function passed in
function M.PlayerState:update(mem, abs_to_rel_func)
    self.position = mem:read_u8(0x0200) -- Player position byte
    self.player_state = mem:read_u8(0x0201) -- Player state value at $201
    self.player_depth = mem:read_u8(0x0202) -- Player depth along the tube

    -- Player alive state: High bit of player_state ($201) is set when dead
    self.alive = ((self.player_state & 0x80) == 0) and 1 or 0

    -- Read and convert score from BCD using local helper
    local score_low = bcd_to_decimal(mem:read_u8(0x0040))
    local score_mid = bcd_to_decimal(mem:read_u8(0x0041))
    local score_high = bcd_to_decimal(mem:read_u8(0x0042))
    self.score = score_high * 10000 + score_mid * 100 + score_low

    self.superzapper_uses = mem:read_u8(0x03AA)   -- Superzapper availability
    self.superzapper_active = mem:read_u8(0x0125) -- Superzapper active status
    self.shot_count = mem:read_u8(0x0135)         -- Number of active player shots ($0135)

    -- Read all 8 shot positions and segments
    -- Determine if level is open based on the level type flag (might be unreliable)
    local level_type_flag = mem:read_u8(0x0111)
    local is_open = (level_type_flag == 0xFF)
    -- Or use level number pattern if flag is known bad:
    -- local level_num_zero_based = (mem:read_u8(0x009F) - 1)
    -- local is_open = (level_num_zero_based % 4 == 2)

    local player_abs_segment = self.position & 0x0F
    for i = 1, 8 do
        -- Read depth (position along the tube) from PlayerShotPositions ($02D3 - $02DA)
        self.shot_positions[i] = mem:read_u8(0x02D3 + i - 1)

        -- Shot is inactive if depth is 0
        if self.shot_positions[i] == 0 then
            self.shot_segments[i] = INVALID_SEGMENT
        else
            -- Read absolute segment from PlayerShotSegments ($02AD - $02B4)
            local abs_segment = mem:read_u8(0x02AD + i - 1)

            -- Shot is also inactive if segment byte is 0
            if abs_segment == 0 then
                self.shot_segments[i] = INVALID_SEGMENT
                self.shot_positions[i] = 0 -- Ensure position is also zeroed if segment is invalid
            else
                -- Valid position and valid segment read, calculate relative segment using passed function
                abs_segment = abs_segment & 0x0F  -- Mask to get valid segment 0-15
                self.shot_segments[i] = abs_to_rel_func(player_abs_segment, abs_segment, is_open)
            end
        end
    end

    -- Update detected input states from debounce byte ($004D)
    self.debounce = mem:read_u8(0x004D)
    self.fire_detected = (self.debounce & 0x10) ~= 0 and 1 or 0 -- Bit 4 for Fire
    self.zap_detected = (self.debounce & 0x08) ~= 0 and 1 or 0  -- Bit 3 for Zap

    -- Update spinner state
    local currentSpinnerAccum = mem:read_u8(0x0051) -- Read current accumulator value ($0051)
    -- self.spinner_commanded is updated in Controls:apply_action in main.lua

    -- Calculate inferred spinner movement delta by comparing current accumulator with previous
    local rawDelta = currentSpinnerAccum - self.prevSpinnerAccum

    -- Handle 8-bit wrap-around
    if rawDelta > 127 then
        rawDelta = rawDelta - 256
    elseif rawDelta < -128 then
        rawDelta = rawDelta + 256
    end
    self.spinner_detected = rawDelta -- Store the calculated delta

    -- Update accumulator values for next frame
    self.SpinnerAccum = currentSpinnerAccum
    self.prevSpinnerAccum = currentSpinnerAccum
end

-- ====================
-- EnemiesState Class
-- ====================
M.pulsar_segments = {}
M.EnemiesState = {}
M.EnemiesState.__index = M.EnemiesState

function M.EnemiesState:new()
    local self = setmetatable({}, M.EnemiesState)
    -- Active enemy counts
    self.active_flippers = 0
    self.active_pulsars = 0
    self.active_tankers = 0
    self.active_spikers = 0
    self.active_fuseballs = 0
    -- Available spawn slots
    self.spawn_slots_flippers = 0
    self.spawn_slots_pulsars = 0
    self.spawn_slots_tankers = 0
    self.spawn_slots_spikers = 0
    self.spawn_slots_fuseballs = 0
    -- Other enemy state
    self.pulse_beat = 0      -- Pulsar pulse beat counter ($0147)
    self.pulsing = 0         -- Pulsar pulsing state ($0148)
    self.pulsar_fliprate = 0 -- Pulsar flip rate ($00B2)
    self.num_enemies_in_tube = 0 -- ($0108)
    self.num_enemies_on_top = 0  -- ($0109)
    self.enemies_pending = 0 -- ($03AB)

    -- Enemy info arrays (Size 7, for enemy slots 1-7)
    self.enemy_type_info = {} -- Raw type byte ($0283 + i - 1)
    self.active_enemy_info = {} -- Raw state byte ($028A + i - 1)
    self.enemy_segments = {}  -- Relative segment (-7 to +8 or -15 to +15, or INVALID_SEGMENT)
    self.enemy_abs_segments = {} -- Absolute segment (0-15, or INVALID_SEGMENT)
    self.enemy_depths = {}    -- Enemy depth/position ($02DF + i - 1)

    -- Decoded Enemy Info Tables (Size 7)
    self.enemy_core_type = {}      -- Bits 0-2 from type byte (0-4 based on assembly)
    self.enemy_direction_moving = {} -- Bit 6 from type byte (0/1)
    self.enemy_between_segments = {} -- Bit 7 from type byte (0/1)
    self.enemy_moving_away = {}    -- Bit 7 from state byte (0/1)
    self.enemy_can_shoot = {}      -- Bit 6 from state byte (0/1)
    self.enemy_split_behavior = {} -- Bits 0-1 from state byte

    -- Add missing initialization for more_enemy_info (Size 7)
    self.more_enemy_info = {}

    -- Enemy Shot Info (Size 4)
    self.shot_positions = {}          -- Absolute depth/position ($02DB + i - 1)
    self.enemy_shot_segments = {}     -- Relative segment (-7 to +8 or -15 to +15, or INVALID_SEGMENT)
    self.enemy_shot_abs_segments = {} -- Absolute segment (0-15, or INVALID_SEGMENT)

    -- Pending enemy data (Size 64)
    self.pending_vid = {}              -- ($0243 + i - 1)
    self.pending_seg = {}              -- Relative segment ($0203 + i - 1, or INVALID_SEGMENT)

    -- Charging Fuseball Tracking (Size 16, indexed 1-16 for abs seg 0-15)
    self.charging_fuseball_segments = {} -- 0 if none, else depth of topmost fuseball
    self.top_pulsar_depth = {} -- 0 if none, else depth of topmost pulsar

    -- Engineered Features for AI (Calculated in update)
    self.nearest_enemy_seg = INVALID_SEGMENT        -- Relative segment of nearest target enemy
    self.nearest_enemy_abs_seg_internal = -1        -- Absolute segment of nearest target enemy (-1 if none)
    self.nearest_enemy_should_fire = false          -- Whether expert logic recommends firing at target
    self.nearest_enemy_should_zap = false           -- Whether expert logic recommends zapping
    self.is_aligned_with_nearest = 0.0              -- 1.0 if aligned, 0.0 otherwise
    self.nearest_enemy_depth_raw = 255              -- Depth of nearest target enemy (0-255)
    self.alignment_error_magnitude = 0.0            -- Normalized alignment error (0.0-1.0) scaled later

    -- Initialize tables
    for i = 1, 7 do
        self.enemy_type_info[i] = 0
        self.active_enemy_info[i] = 0
        self.enemy_segments[i] = INVALID_SEGMENT
        self.enemy_abs_segments[i] = INVALID_SEGMENT
        self.enemy_depths[i] = 0
        self.enemy_core_type[i] = 0
        self.enemy_direction_moving[i] = 0
        self.enemy_between_segments[i] = 0
        self.enemy_moving_away[i] = 0
        self.enemy_can_shoot[i] = 0
        self.enemy_split_behavior[i] = 0
        self.more_enemy_info[i] = 0 -- Ensure more_enemy_info is always initialized
    end
    for i = 1, 4 do
        self.shot_positions[i] = 0
        self.enemy_shot_segments[i] = INVALID_SEGMENT
        self.enemy_shot_abs_segments[i] = INVALID_SEGMENT
    end
     for i = 1, 64 do
        self.pending_vid[i] = 0
        self.pending_seg[i] = INVALID_SEGMENT
    end
    for i = 1, 16 do
        self.charging_fuseball_segments[i] = 0
        self.top_pulsar_depth[i] = 0
    end

    return self
end

-- EnemiesState update needs game_state, player_state, level_state, and abs_to_rel_func
function M.EnemiesState:update(mem, game_state, player_state, level_state, abs_to_rel_func)
    -- Get player position and level type for relative calculations
    local player_abs_segment = player_state.position & 0x0F -- Get current player absolute segment
    -- Determine if level is open based *only* on the memory flag now
    local is_open = (level_state.level_type == 0xFF)
    -- Re-enable debug print to monitor memory flag and resulting is_open

    -- Read active enemy counts and related state
    self.active_flippers         = mem:read_u8(0x0142) -- n_flippers
    self.active_pulsars          = mem:read_u8(0x0143) -- n_pulsars
    self.active_tankers          = mem:read_u8(0x0144) -- n_tankers
    self.active_spikers          = mem:read_u8(0x0145) -- n_spikers
    self.active_fuseballs        = mem:read_u8(0x0146) -- n_fuseballs
    self.pulse_beat              = mem:read_u8(0x0147) -- pulse_beat
    self.pulsing                 = mem:read_u8(0x0148) -- pulsing state
    self.pulsar_fliprate         = mem:read_u8(0x00B2) -- Pulsar flip rate
    self.num_enemies_in_tube     = mem:read_u8(0x0108) -- NumInTube
    self.num_enemies_on_top      = mem:read_u8(0x0109) -- NumOnTop
    self.enemies_pending         = mem:read_u8(0x03AB) -- PendingEnemies

    -- Read available spawn slots
    self.spawn_slots_flippers = mem:read_u8(0x013D)  -- avl_flippers
    self.spawn_slots_pulsars = mem:read_u8(0x013E)   -- avl_pulsars
    self.spawn_slots_tankers = mem:read_u8(0x013F)   -- avl_tankers
    self.spawn_slots_spikers = mem:read_u8(0x0140)   -- avl_spikers
    self.spawn_slots_fuseballs = mem:read_u8(0x0141) -- avl_fuseballs

    -- Read and process enemy slots (1-7)
    for i = 1, 7 do
        -- Read depth and segment first to determine activity
        local enemy_depth_raw = mem:read_u8(0x02DF + i - 1) -- EnemyPositions ($02DF-$02E5)
        local abs_segment_raw = mem:read_u8(0x02B9 + i - 1) -- EnemySegments ($02B9-$02BF)

        -- Reset decoded info for this slot
        self.enemy_core_type[i] = 0
        self.enemy_direction_moving[i] = 0
        self.enemy_between_segments[i] = 0
        self.enemy_moving_away[i] = 0
        self.enemy_can_shoot[i] = 0
        self.enemy_split_behavior[i] = 0
        self.enemy_segments[i] = INVALID_SEGMENT
        self.enemy_abs_segments[i] = INVALID_SEGMENT
        self.enemy_depths[i] = 0
        self.enemy_type_info[i] = 0
        self.active_enemy_info[i] = 0

        -- Check if enemy is active (depth > 0 and segment raw byte > 0)
        if enemy_depth_raw > 0 and abs_segment_raw > 0 then
            local abs_segment = abs_segment_raw & 0x0F -- Mask to 0-15
            self.enemy_abs_segments[i] = abs_segment
            self.enemy_segments[i] = abs_to_rel_func(player_abs_segment, abs_segment, is_open)
            self.enemy_depths[i] = enemy_depth_raw

            -- Read raw type/state bytes only for active enemies
            local type_byte = mem:read_u8(0x0283 + i - 1) -- EnemyTypeInfo ($0283-$0289)
            local state_byte = mem:read_u8(0x028A + i - 1) -- ActiveEnemyInfo ($028A-$0290)
            self.enemy_type_info[i] = type_byte
            self.active_enemy_info[i] = state_byte

            -- Decode Type Byte (Use assembly mask)
            self.enemy_core_type[i] = type_byte & ENEMY_TYPE_MASK -- Apply the mask to get 0-4
            self.enemy_direction_moving[i] = (type_byte & 0x40) ~= 0 and 1 or 0 -- Bit 6: Segment increasing?
            self.enemy_between_segments[i] = (type_byte & 0x80) ~= 0 and 1 or 0 -- Bit 7: Between segments?

            -- Decode State Byte
            self.enemy_moving_away[i] = (state_byte & 0x80) ~= 0 and 1 or 0 -- Bit 7: Moving Away?
            self.enemy_can_shoot[i] = (state_byte & 0x40) ~= 0 and 1 or 0   -- Bit 6: Can Shoot?
            self.enemy_split_behavior[i] = state_byte & 0x03                -- Bits 0-1: Split Behavior
        end -- End if enemy active
    end -- End enemy slot loop

    -- Read more_enemy_info (7 bytes at 0x02CC)
    for i = 1, 7 do
        self.more_enemy_info[i] = mem:read_u8(0x02CC + i - 1)
    end

    -- Calculate charging Fuseball segments (now stores depth of topmost fuseball, or 0)
    for seg = 1, 16 do self.charging_fuseball_segments[seg] = 0 end
    for i = 1, 7 do
        if self.enemy_core_type[i] == ENEMY_TYPE_FUSEBALL and self.enemy_abs_segments[i] ~= INVALID_SEGMENT and (self.active_enemy_info[i] & 0x80) == 0 then
            local abs_segment_idx = self.enemy_abs_segments[i] + 1
            if abs_segment_idx >= 1 and abs_segment_idx <= 16 then
                local depth = self.enemy_depths[i]
                if self.charging_fuseball_segments[abs_segment_idx] == 0 or depth < self.charging_fuseball_segments[abs_segment_idx] then
                    self.charging_fuseball_segments[abs_segment_idx] = depth
                end
            end
        end
    end
    -- Calculate topmost pulsar depth per segment (1-16)
    -- Ensure pulsar_depth_by_segment is always filled out (1-16)
    if not self.pulsar_depth_by_segment then self.pulsar_depth_by_segment = {} end
    for seg = 1, 16 do self.pulsar_depth_by_segment[seg] = 0 end
    for i = 1, 7 do
        if self.enemy_core_type[i] == ENEMY_TYPE_PULSAR and self.enemy_abs_segments[i] ~= INVALID_SEGMENT then
            local abs_segment_idx = self.enemy_abs_segments[i] + 1
            if abs_segment_idx >= 1 and abs_segment_idx <= 16 then
                local depth = self.enemy_depths[i]
                if self.pulsar_depth_by_segment[abs_segment_idx] == 0 or depth < self.pulsar_depth_by_segment[abs_segment_idx] then
                    self.pulsar_depth_by_segment[abs_segment_idx] = depth
                end
            end
        end
    end
end

-- Helper functions for display (can be called on an instance)
function M.EnemiesState:decode_enemy_type(type_byte)
    local enemy_type = type_byte & ENEMY_TYPE_MASK -- Use the mask
    local between_segments = (type_byte & 0x80) ~= 0
    local segment_increasing = (type_byte & 0x40) ~= 0
    return string.format("%d%s%s",
        enemy_type,
        between_segments and "B" or "-",
        segment_increasing and "+" or "-" -- Use '-' if not increasing
    )
end

function M.EnemiesState:decode_enemy_state(state_byte)
    local split_behavior = state_byte & 0x03
    local can_shoot = (state_byte & 0x40) ~= 0
    local moving_away = (state_byte & 0x80) ~= 0
    return string.format("%s%s%d", -- Show split behavior as number
        moving_away and "A" or "T", -- Away / Towards
        can_shoot and "S" or "-",   -- Can Shoot / Cannot Shoot
        split_behavior
    )
end

function M.EnemiesState:get_total_active()
    return self.active_flippers + self.active_pulsars + self.active_tankers +
        self.active_spikers + self.active_fuseballs
end

-- Helper: Get fractional segment for an enemy index (1-based, 1-7)
function M.EnemiesState:get_fractional_segment_for_enemy(enemy_index)
    local base_segment = self.enemy_abs_segments[enemy_index]
    local info = self.more_enemy_info[enemy_index]
    if base_segment == nil or base_segment == INVALID_SEGMENT then
        return base_segment -- Always return a number (may be INVALID_SEGMENT)
    end
    -- Only use fractional info if top bit is set
    if info and (info & 0x80) ~= 0 then
        local fraction = (info & 0x0F) / 16.0
        return base_segment - fraction
    else
        return base_segment -- Always return a number
    end
end

-- Return the module table
return M