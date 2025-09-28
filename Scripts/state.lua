--[[
    state.lua
    Contains class definitions for game state objects (Game, Level, Player, Enemies)
    and related helper functions for the Tempest AI project.
--]]

local M = {} -- Module table to hold exported classes and functions

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
            -- 2. Top-level enemies (depth <= 0x10)
            -- Includes Flippers, Pulsars, Tankers, Fuseballs, Spikers if they are close
            if depth <= 0x10 then
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
        if shot_abs_seg ~= INVALID_SEGMENT and shot_depth > 0 and shot_depth <= 0x30 and not has_ammo then
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
        ENEMY_TYPE_TANKER,   -- 2
        ENEMY_TYPE_FLIPPER,  -- 0
        ENEMY_TYPE_SPIKER,   -- 3
        ENEMY_TYPE_PULSAR    -- 1
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
    local player_abs_seg = player_state.position & 0x0F

    local initial_target_seg_abs
    local target_depth
    local should_fire = false
    local should_zap = false
    local did_flee = false
    local hunting_target_info = "N/A"

    -- Check for Tube Zoom state first
    if game_state.gamestate == 0x20 then
        -- Spike heights are depths: 0=clear/no spike, >0 means spike exists, smaller number = longer spike
        local current_spike_h = level_state.spike_heights[player_abs_seg]
        if current_spike_h == 0 then return player_abs_seg, 0, true, false end
        local left_neighbour_seg = -1
        local right_neighbour_seg = -1
        if is_open then
            if player_abs_seg > 0 then left_neighbour_seg = player_abs_seg - 1 end
            if player_abs_seg < 15 then right_neighbour_seg = player_abs_seg + 1 end
        else
            left_neighbour_seg = (player_abs_seg - 1 + 16) % 16
            right_neighbour_seg = (player_abs_seg + 1) % 16
        end
        local left_spike_h = -1; if left_neighbour_seg ~= -1 then left_spike_h = level_state.spike_heights[left_neighbour_seg] end
        local right_spike_h = -1; if right_neighbour_seg ~= -1 then right_spike_h = level_state.spike_heights[right_neighbour_seg] end
        if left_spike_h == 0 then return left_neighbour_seg, 0, true, false end
        if right_spike_h == 0 then return right_neighbour_seg, 0, true, false end
        local temp_target = player_abs_seg
        local is_left_better = (left_spike_h > current_spike_h)
        local is_right_better = (right_spike_h > current_spike_h)
        if is_left_better and is_right_better then temp_target = (left_spike_h >= right_spike_h) and left_neighbour_seg or right_neighbour_seg
        elseif is_left_better then temp_target = left_neighbour_seg
        elseif is_right_better then temp_target = right_neighbour_seg
        end
        initial_target_seg_abs, target_depth, should_fire, should_zap = (function()
           -- ... (spike logic returns temp_target, 0, true, false)
           return temp_target, 0, true, false
        end)()
    -- Check Flee/Hunt Logic (only in normal play mode)
    elseif game_state.gamestate == 0x04 then
        local forbidden_segments = find_forbidden_segments(enemies_state, level_state, player_state)
        local current_segment_is_forbidden = forbidden_segments[player_abs_seg] or false

        if current_segment_is_forbidden then
            did_flee = true
            initial_target_seg_abs = find_nearest_safe_segment(player_abs_seg, is_open, forbidden_segments, abs_to_rel_func)
            target_depth = 0 -- Depth isn't the focus when fleeing
            should_fire = false
            should_zap = false
        else -- Current segment is SAFE, proceed to HUNT
            -- Pass forbidden_segments to hunt_enemies
            local hunt_target_seg, hunt_target_depth, should_avoid = hunt_enemies(enemies_state, player_abs_seg, is_open, abs_to_rel_func, forbidden_segments)
            hunting_target_info = string.format("HuntTgt=%d, HuntDepth=%02X", hunt_target_seg, hunt_target_depth) -- DEBUG

            if hunt_target_seg ~= -1 then
                initial_target_seg_abs = hunt_target_seg
                target_depth = hunt_target_depth
                local rel_dist = abs_to_rel_func(player_abs_seg, initial_target_seg_abs, is_open)
                should_fire = (rel_dist <= 1) -- Initial fire recommendation if aligned
            else
                initial_target_seg_abs = player_abs_seg -- Stay put if no hunt target
                target_depth = 0x10
                should_fire = false
            end
        end
    else -- Other game states: Stay put
        initial_target_seg_abs = player_abs_seg
        target_depth = player_state.player_depth
        should_fire = false
        should_zap = false
    end

    -- Apply panic braking
    local final_target_seg_abs = initial_target_seg_abs
    local did_brake = false
    if final_target_seg_abs ~= player_abs_seg and game_state.gamestate == 0x04 then
        local initial_relative_dist = abs_to_rel_func(player_abs_seg, final_target_seg_abs, is_open)
        local next_segment_abs = -1

        if initial_relative_dist > 0 then -- Moving right (positive relative dist)
            if is_open then
                if player_abs_seg < 15 then next_segment_abs = player_abs_seg + 1 end
            else -- Closed
                next_segment_abs = (player_abs_seg + 1) % 16
            end
        elseif initial_relative_dist < 0 then -- Moving left (negative relative dist)
             if is_open then
                if player_abs_seg > 0 then next_segment_abs = player_abs_seg - 1 end
            else -- Closed
                next_segment_abs = (player_abs_seg - 1 + 16) % 16
            end
        end

        -- If there is a valid next segment to check
        if next_segment_abs ~= -1 then
            local brake_condition_met = false
            -- Check enemy shots in the next segment
            for i = 1, 4 do
                if enemies_state.enemy_shot_abs_segments[i] == next_segment_abs and
                   enemies_state.shot_positions[i] > 0 and
                   enemies_state.shot_positions[i] <= 0x30 then
                    brake_condition_met = true; break
                end
            end
            -- Check Flippers (0) and Pulsars (1) in the next segment if no shot found yet
            if not brake_condition_met then
                for i = 1, 7 do
                    -- Check for Flipper OR Pulsar if close
                    if (enemies_state.enemy_core_type[i] == ENEMY_TYPE_FLIPPER or enemies_state.enemy_core_type[i] == ENEMY_TYPE_PULSAR) and -- <<< CORRECTED TYPE
                       enemies_state.enemy_abs_segments[i] == next_segment_abs and
                       enemies_state.enemy_depths[i] > 0 and
                       enemies_state.enemy_depths[i] <= 0x30 then
                        brake_condition_met = true; break
                    end
                end
            end

            -- If brake condition met, override target to stay put
            if brake_condition_met then
                did_brake = true -- Record brake engagement
                final_target_seg_abs = player_abs_seg -- Override the target
                target_depth = player_state.player_depth -- Use current depth (might not matter)
                should_fire = false -- Don't fire if braking
                should_zap = false
            end
        end
    end

    -- NEW: Fuseball Avoidance Logic (after panic brake)
    local fuseball_avoid_target = -1
    local min_fuseball_dist = 3 -- Minimum desired distance from a top-level fuseball

    -- Check if the current target is too close to a top-level fuseball
    local too_close = false
    for i = 1, 7 do
        if enemies_state.enemy_core_type[i] == ENEMY_TYPE_FUSEBALL and -- Is it a Fuseball? (Correct type 4)
           enemies_state.enemy_depths[i] <= 0x10 and -- Is it at the top?
           enemies_state.enemy_abs_segments[i] ~= INVALID_SEGMENT then

            local fuseball_abs_seg = enemies_state.enemy_abs_segments[i]
            local dist_to_fuseball = math.abs(abs_to_rel_func(final_target_seg_abs, fuseball_abs_seg, is_open))

            if dist_to_fuseball < min_fuseball_dist then
                too_close = true
                break -- Found one too close, no need to check others
            end
        end
    end

    -- If too close, find the nearest segment >= min_fuseball_dist away from ALL top fuseballs
    if too_close then
        print("FUSEBALL AVOID: Target " .. final_target_seg_abs .. " too close. Finding alternative.")
        local best_safe_seg = -1
        local search_max_dist = is_open and 15 or 8

        for search_dist = 0, search_max_dist do -- Start search from current pos (dist 0)
            local segments_to_check = {}
            if search_dist == 0 then
                segments_to_check = {player_abs_seg}
            else
                -- Check Left
                local left_check = -1
                if is_open then
                    if player_abs_seg - search_dist >= 0 then left_check = player_abs_seg - search_dist end
                else left_check = (player_abs_seg - search_dist + 16) % 16 end
                if left_check ~= -1 then table.insert(segments_to_check, left_check) end

                -- Check Right
                local right_check = -1
                 if is_open then
                    if player_abs_seg + search_dist <= 15 then right_check = player_abs_seg + search_dist end
                else right_check = (player_abs_seg + search_dist) % 16 end
                 if right_check ~= -1 and right_check ~= left_check then table.insert(segments_to_check, right_check) end
            end

            for _, check_seg in ipairs(segments_to_check) do
                local is_seg_safe = true
                -- Check this segment against ALL top-level fuseballs
                for i = 1, 7 do
                     if enemies_state.enemy_core_type[i] == ENEMY_TYPE_FUSEBALL and enemies_state.enemy_depths[i] <= 0x10 and enemies_state.enemy_abs_segments[i] ~= INVALID_SEGMENT then -- Correct type 4
                        local fuseball_abs_seg = enemies_state.enemy_abs_segments[i]
                        local dist = math.abs(abs_to_rel_func(check_seg, fuseball_abs_seg, is_open))
                        if dist < min_fuseball_dist then
                            is_seg_safe = false
                            break -- This segment is too close to this fuseball
                        end
                    end
                end

                if is_seg_safe then
                    best_safe_seg = check_seg
                    goto found_safe_segment -- Exit outer loops once the *nearest* safe segment is found
                end
            end
        end
        ::found_safe_segment::

        if best_safe_seg ~= -1 then
            print("FUSEBALL AVOID: New target = " .. best_safe_seg)
            final_target_seg_abs = best_safe_seg
            target_depth = 0 -- Reset depth indication
            should_fire = false -- Don't fire when avoiding
            should_zap = false
        else
             print("WARNING: Fuseball avoidance could not find any safe segment!")
             -- Keep original target if no safe alternative found (might be stuck)
        end
    end

    -- Apply nearby Flipper firing override (Run this check regardless of brake/avoidance, but before shot count limit)
    local initial_should_fire = should_fire -- Store initial recommendation before override
    if not initial_should_fire then -- Only override if not already firing
        for i = 1, 7 do
            if enemies_state.enemy_depths[i] > 0 and enemies_state.enemy_depths[i] <= 0x30 then -- Is it close vertically?
                local threat_abs_seg = enemies_state.enemy_abs_segments[i]
                if threat_abs_seg ~= INVALID_SEGMENT then
                    local threat_rel_seg = abs_to_rel_func(player_abs_seg, threat_abs_seg, is_open)
                    if math.abs(threat_rel_seg) <= 1 then -- Is it close laterally (or aligned)?
                        -- Always recommend firing at close threats
                        should_fire = true

                        -- Expert tweak: if shot buffer is full, still press FIRE but move one segment away
                        -- Apply only for Flipper/Pulsar top-rail threats per design intent
                        if player_state.shot_count >= 8 then
                            local core_type = enemies_state.enemy_core_type[i]
                            if core_type == ENEMY_TYPE_FLIPPER or core_type == ENEMY_TYPE_PULSAR then
                                -- Move away by one segment opposite the threat direction.
                                local move_right = (threat_rel_seg <= 0) -- threat at/aligned-left -> move right
                                local candidate = -1
                                if move_right then
                                    if is_open then
                                        if player_abs_seg < 15 then candidate = player_abs_seg + 1 end
                                    else
                                        candidate = (player_abs_seg + 1) % 16
                                    end
                                else
                                    if is_open then
                                        if player_abs_seg > 0 then candidate = player_abs_seg - 1 end
                                    else
                                        candidate = (player_abs_seg - 1 + 16) % 16
                                    end
                                end
                                if candidate ~= -1 then
                                    final_target_seg_abs = candidate
                                    target_depth = 0 -- Depth not critical for lateral dodge
                                end
                            end
                        end

                        break -- Found a dangerous close threat, no need to check others
                    end
                end
            end
        end
    end

    -- Apply shot count override (happens last)
    should_fire = should_fire or player_state.shot_count < 3

    return final_target_seg_abs, target_depth, should_fire, should_zap
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
    self.death_reason = 0       -- Death reason code from $013B (signed byte semantics)
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
    -- Read death reason byte ($013B). Assembly uses values like $FF (enemy shot), $09 (collision), $07 (spike/pulsar).
    -- We keep it as 0..255 here; packing uses signed byte semantics so $FF will be -1 on Python side.
    self.death_reason = mem:read_u8(0x013B)

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
    -- Enemy speed parameters (level-dependent)
    self.fuse_move_prb = 0         -- Fuseball movement probability
    self.spd_flipper_lsb = 0       -- Flipper speed LSB
    self.spd_pulsar_lsb = 0        -- Pulsar speed LSB
    self.spd_tanker_lsb = 0        -- Tanker speed LSB
    self.spd_spiker_lsb = 0        -- Spiker speed LSB
    self.spd_fuseball_lsb = 0      -- Fuseball speed LSB
    self.spd_flipper_msb = 0       -- Flipper speed MSB
    self.spd_pulsar_msb = 0        -- Pulsar speed MSB
    self.spd_tanker_msb = 0        -- Tanker speed MSB
    self.spd_spiker_msb = 0        -- Spiker speed MSB
    self.spd_fuseball_msb = 0      -- Fuseball speed MSB
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

    -- Raw angle/progress per enemy (Size 7)
    -- Mirrors more_enemy_info at $02CC..$02D2; low nibble (0-15) increments/decrements while between segments
    self.more_enemy_info = {}

    -- Enemy Shot Info (Size 4)
    self.shot_positions = {}          -- Absolute depth/position ($02DB + i - 1)
    self.shot_positions_lsb = {}      -- Fractional LSB for enemy shots ($02E6 + i - 1)
    self.enemy_shot_segments = {}     -- Relative segment (-7 to +8 or -15 to +15, or INVALID_SEGMENT)
    self.enemy_shot_abs_segments = {} -- Absolute segment (0-15, or INVALID_SEGMENT)

    -- Pending enemy data (Size 64)
    self.pending_vid = {}              -- ($0243 + i - 1)
    self.pending_seg = {}              -- Relative segment ($0203 + i - 1, or INVALID_SEGMENT)

    -- Charging Fuseball Tracking (Size 7, one per enemy slot that can be a fuseball)
    -- Array contains relative segment positions (-7 to +8 or -15 to +15, or 0 when no charging fuseball)
    self.charging_fuseball = {} -- Relative segments of charging fuseballs
    
    -- Pulsar Tracking (Size 7, one per enemy slot that can be a pulsar)
    -- Array contains relative segment positions (-7 to +8 or -15 to +15, or 0 when no pulsar)
    self.active_pulsar = {} -- Relative segments of pulsars
    
    -- Top Rail Enemy Tracking (Size 7, one per enemy slot that can be a pulsar or flipper at depth 0x20)
    -- Array contains relative segment positions (-7 to +8 or -15 to +15, or 0 when no valid enemy)
    self.active_top_rail_enemies = {} -- Relative segments of pulsars/flippers at the top rail
    
    -- NEW: Fractional Enemy Segments By Slot (Size 7, indexed 1-7)
    -- Stores fractional segment position as 12-bit integer
    self.fractional_enemy_segments_by_slot = {}

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
        self.more_enemy_info[i] = 0
    end
    for i = 1, 4 do
        self.shot_positions[i] = 0
        self.shot_positions_lsb[i] = 0
        self.enemy_shot_segments[i] = INVALID_SEGMENT
        self.enemy_shot_abs_segments[i] = INVALID_SEGMENT
    end
     for i = 1, 64 do
        self.pending_vid[i] = 0
        self.pending_seg[i] = INVALID_SEGMENT
    end
    for i = 1, 7 do
        self.charging_fuseball[i] = 0
        self.active_pulsar[i] = 0
        self.active_top_rail_enemies[i] = 0
        self.fractional_enemy_segments_by_slot[i] = 0
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

    -- Read enemy speed parameters (level-dependent)
    self.fuse_move_prb = mem:read_u8(0x015F)         -- Fuseball movement probability
    self.spd_flipper_lsb = mem:read_u8(0x0160)       -- Flipper speed LSB
    self.spd_pulsar_lsb = mem:read_u8(0x0161)        -- Pulsar speed LSB
    self.spd_tanker_lsb = mem:read_u8(0x0162)        -- Tanker speed LSB
    self.spd_spiker_lsb = mem:read_u8(0x0163)        -- Spiker speed LSB
    self.spd_fuseball_lsb = mem:read_u8(0x0164)      -- Fuseball speed LSB
    self.spd_flipper_msb = mem:read_u8(0x0165)       -- Flipper speed MSB
    self.spd_pulsar_msb = mem:read_u8(0x0166)        -- Pulsar speed MSB
    self.spd_tanker_msb = mem:read_u8(0x0167)        -- Tanker speed MSB
    self.spd_spiker_msb = mem:read_u8(0x0168)        -- Spiker speed MSB
    self.spd_fuseball_msb = mem:read_u8(0x0169)      -- Fuseball speed MSB

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

            -- Read raw more_enemy_info angle/progress byte ($02CC-$02D2)
            self.more_enemy_info[i] = mem:read_u8(0x02CC + i - 1)
        end -- End if enemy active
    end -- End enemy slot loop

    -- Calculate charging Fuseball segments (reset first)
    for i = 1, 7 do 
        self.charging_fuseball[i] = 0
        self.active_pulsar[i] = 0
        self.active_top_rail_enemies[i] = 0
    end
    
    for i = 1, 7 do
        -- Check if it's an active Fuseball (type 4) moving towards player (bit 7 of state byte is clear)
        if self.enemy_core_type[i] == ENEMY_TYPE_FUSEBALL and self.enemy_abs_segments[i] ~= INVALID_SEGMENT and (self.active_enemy_info[i] & 0x80) == 0 then -- Correct type 4
            self.charging_fuseball[i] = self.enemy_segments[i]
        end
        
        -- Check if it's an active Pulsar (type 1)
        if self.enemy_core_type[i] == ENEMY_TYPE_PULSAR and self.enemy_abs_segments[i] ~= INVALID_SEGMENT then -- Pulsar type 1
            self.active_pulsar[i] = self.enemy_segments[i]
        end

        -- Check if it's a top rail Pulsar or Flipper (close to top)
        if (self.enemy_abs_segments[i] ~= INVALID_SEGMENT and (self.enemy_core_type[i] == ENEMY_TYPE_PULSAR or self.enemy_core_type[i] == ENEMY_TYPE_FLIPPER) and self.enemy_depths[i] <= 0x30) then
            -- Build absolute floating segment using between-segment progress (from more_enemy_info low nibble)
            local abs_int = self.enemy_abs_segments[i]
            local angle_nibble = self.more_enemy_info[i] & 0x0F -- 0..15
            -- Compute a monotonically increasing progress [0,1) based on direction semantics:
            -- Direction bit set (increasing): nibble DECREMENTS each tick -> progress = (15 - nibble)/16
            -- Direction bit clear (decreasing): nibble INCREMENTS each tick -> progress = nibble/16
            local progress = 0.0
            if self.enemy_between_segments[i] == 1 then
                if self.enemy_direction_moving[i] == 1 then
                    progress = (15.0 - angle_nibble) / 16.0
                else
                    progress = angle_nibble / 16.0
                end
            end

            -- Absolute float using StartFlip semantics:
            -- When increasing, enemy_seg already points to the DESTINATION segment; position is (dest - (1 - progress)).
            -- When decreasing, enemy_seg points to the SOURCE segment; position is (source - progress).
            local abs_float
            if self.enemy_between_segments[i] == 1 then
                if self.enemy_direction_moving[i] == 1 then
                    abs_float = abs_int + progress - 1.0
                else
                    abs_float = abs_int - progress
                end
            else
                abs_float = abs_int
            end

            -- Normalize to [0,16)
            abs_float = abs_float % 16.0
            if abs_float < 0 then abs_float = abs_float + 16.0 end

            -- Convert to relative float
            local rel_float
            if is_open then
                rel_float = abs_float - player_abs_segment
            else
                -- Wrap into [-8,+8]
                rel_float = abs_float - player_abs_segment
                while rel_float > 8.0 do rel_float = rel_float - 16.0 end
                while rel_float < -8.0 do rel_float = rel_float + 16.0 end
            end

            self.active_top_rail_enemies[i] = rel_float
            -- print("Top rail at: " .. string.format("%.3f", rel_float) .. " (slot " .. i .. ")")
        end
    end

    -- Read and process enemy shots (1-4)
    for i = 1, 4 do
        -- Read integer depth/position and fractional LSB
        local pos_int = mem:read_u8(0x02DB + i - 1)  -- EnemyShotPositions ($02DB-$02DE)
        local pos_lsb = mem:read_u8(0x02E6 + i - 1)  -- enm_shot_lsb ($02E6-$02E9)

        -- If integer position is 0, shot is inactive regardless of LSB
        if pos_int == 0 then
            self.enemy_shot_segments[i] = INVALID_SEGMENT
            self.enemy_shot_abs_segments[i] = INVALID_SEGMENT
            self.shot_positions_lsb[i] = 0
            self.shot_positions[i] = 0
        else
            -- Read shot segment byte
            local abs_segment_raw = mem:read_u8(0x02B5 + i - 1) -- EnemyShotSegments ($02B5-$02B8)
             -- Also inactive if segment byte is 0
            if abs_segment_raw == 0 then
                self.enemy_shot_segments[i] = INVALID_SEGMENT
                self.enemy_shot_abs_segments[i] = INVALID_SEGMENT
                self.shot_positions_lsb[i] = 0
                self.shot_positions[i] = 0 -- Ensure position is zeroed
            else
                local abs_segment = abs_segment_raw & 0x0F -- Mask to 0-15
                self.enemy_shot_abs_segments[i] = abs_segment
                self.enemy_shot_segments[i] = abs_to_rel_func(player_abs_segment, abs_segment, is_open)
                -- Combine integer position with LSB to form full 8.8 fixed-point as a float
                -- Range remains < 256.0 (pos_int in [1,255], pos_lsb in [0,255])
                self.shot_positions_lsb[i] = pos_lsb
                self.shot_positions[i] = pos_int + (pos_lsb / 256.0)
            end
        end
    end

    -- Read pending_seg (64 bytes starting at 0x0203), store relative
    for i = 1, 64 do
        local abs_segment_raw = mem:read_u8(0x0203 + i - 1)
        if abs_segment_raw == 0 then
            self.pending_seg[i] = INVALID_SEGMENT -- Not active, use sentinel
        else
            local segment = abs_segment_raw & 0x0F    -- Mask to ensure 0-15
            self.pending_seg[i] = abs_to_rel_func(player_abs_segment, segment, is_open)
        end
    end

    -- Read pending_vid (64 bytes starting at 0x0243)
    for i = 1, 64 do
        self.pending_vid[i] = mem:read_u8(0x0243 + i - 1)
    end
    
    -- Calculate true between-segment progress per slot (scaled to 12-bit, 0..4095)
    for i = 1, 7 do
        if self.enemy_abs_segments[i] == INVALID_SEGMENT then
            self.fractional_enemy_segments_by_slot[i] = INVALID_SEGMENT
        else
            -- Use the same direction-aware progress as above
            local progress = 0.0
            if self.enemy_between_segments[i] == 1 then
                local angle_nibble = self.more_enemy_info[i] & 0x0F -- 0..15
                if self.enemy_direction_moving[i] == 1 then
                    progress = (15.0 - angle_nibble) / 16.0
                else
                    progress = angle_nibble / 16.0
                end
            end
            local scaled_value = math.floor(progress * 4096.0 + 0.5) -- Round to nearest
            if scaled_value >= 4096 then scaled_value = 4095 end
            if scaled_value < 0 then scaled_value = 0 end
            self.fractional_enemy_segments_by_slot[i] = scaled_value
        end
    end

    -- Compute fractional relative positions for all enemies to enable precise timing learning
    for i = 1, 7 do
        if self.enemy_abs_segments[i] ~= INVALID_SEGMENT then
            local abs_int = self.enemy_abs_segments[i]
            local progress = 0.0
            if self.enemy_between_segments[i] == 1 then
                local angle_nibble = self.more_enemy_info[i] & 0x0F
                if self.enemy_direction_moving[i] == 1 then
                    progress = (15.0 - angle_nibble) / 16.0
                else
                    progress = angle_nibble / 16.0
                end
            end
            local abs_float
            if self.enemy_between_segments[i] == 1 then
                if self.enemy_direction_moving[i] == 1 then
                    abs_float = abs_int + progress - 1.0
                else
                    abs_float = abs_int - progress
                end
            else
                abs_float = abs_int
            end
            abs_float = abs_float % 16.0
            if abs_float < 0 then abs_float = abs_float + 16.0 end
            local rel_float
            if is_open then
                rel_float = abs_float - player_abs_segment
            else
                rel_float = abs_float - player_abs_segment
                while rel_float > 8.0 do rel_float = rel_float - 16.0 end
                while rel_float < -8.0 do rel_float = rel_float + 16.0 end
            end
            self.enemy_segments[i] = rel_float  -- Update to fractional relative position
        end
    end

    -- === Calculate and store nearest enemy segment and engineered features ===
    -- Reset zap recommendation before calculation
    self.nearest_enemy_should_zap = false
    -- Use the internal helper function find_target_segment, passing the calculated is_open
    local nearest_abs_seg, nearest_depth, should_fire_target, should_zap_target = find_target_segment(game_state, player_state, level_state, self, abs_to_rel_func, is_open)

    -- Store results in the object's fields
    self.nearest_enemy_abs_seg_internal = nearest_abs_seg -- Store absolute segment (-1 if none)
    self.nearest_enemy_should_fire = should_fire_target   -- Store firing recommendation
    self.nearest_enemy_should_zap = should_zap_target     -- Store zapping recommendation

    -- Add debug print for hints
    -- print(string.format("STATE HINTS DEBUG: Abs=%d, Fire=%s, Zap=%s", self.nearest_enemy_abs_seg_internal, tostring(self.nearest_enemy_should_fire), tostring(self.nearest_enemy_should_zap)))

    if nearest_abs_seg == -1 then -- No target found
        self.nearest_enemy_seg = INVALID_SEGMENT
        self.is_aligned_with_nearest = 0.0
        self.nearest_enemy_depth_raw = 255 -- Use max depth as sentinel
        self.alignment_error_magnitude = 0.0
    else -- Valid target found
        -- Use the is_open calculated at the start of this function
        local nearest_rel_seg = abs_to_rel_func(player_abs_segment, nearest_abs_seg, is_open)
        self.nearest_enemy_seg = nearest_rel_seg     -- Store relative segment
        self.nearest_enemy_depth_raw = nearest_depth -- Store raw depth

        -- Calculate Is_Aligned
        self.is_aligned_with_nearest = (nearest_rel_seg == 0) and 1.0 or 0.0

        -- Calculate Alignment_Error_Magnitude (Normalized 0.0-1.0)
        local error_abs = math.abs(nearest_rel_seg)
        local max_error = is_open and 15.0 or 8.0 -- Max possible distance
        self.alignment_error_magnitude = (error_abs > 0) and (error_abs / max_error) or 0.0
        -- Scaling happens during packing
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

-- Return the module table
return M