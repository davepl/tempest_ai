local state_defs = require("state") -- Assuming state.lua is in the same dir

-- Define enemy type constants (copy from main.lua)
local ENEMY_TYPE_FLIPPER = 0
local ENEMY_TYPE_PULSAR = 1
local ENEMY_TYPE_TANKER = 2
local ENEMY_TYPE_SPIKER = 3
local ENEMY_TYPE_FUSEBALL = 4

-- Define constants (copy from main.lua)
local INVALID_SEGMENT = state_defs.INVALID_SEGMENT

-- New constants for top rail logic
local TOP_RAIL_DEPTH = 0x15
local SAFE_DISTANCE = 1
local FLIPPER_WAIT_DISTANCE = 5 -- segments within which we prefer to wait and conserve shots on top rail
local FREEZE_FIRE_PRIO_LOW = 2
local FREEZE_FIRE_PRIO_HIGH = 8
local AVOID_FIRE_PRIORITY = 3
local PULSAR_THRESHOLD = 0xE0 -- Pulsing threshold for avoidance (match dangerous pulsar threshold)
-- Open-level tuning: react slightly sooner to top-rail flippers using fractional distance
local OPEN_FLIPPER_REACT_DISTANCE = 1.10
-- Pulsar target offset (segments away from pulsar when hunting/avoiding)
local PULSAR_TARGET_DISTANCE = 2
-- Optional: Conserve fire mode (hold fire/movement until react distance)
local CONSERVE_FIRE_MODE = false
local CONSERVE_REACT_DISTANCE = 1.10
-- Configurable pulsar hunting preferences
local PULSAR_PREF_DISTANCE = 1.0   -- desired lanes away from the pulsar (can be fractional for hold/fire logic)
local PULSAR_PREF_TOLERANCE = 0.15 -- acceptable window around preferred distance to hold and fire

local M = {} -- Module table

-- Global variables needed by calculate_reward (scoped within this module)
local previous_score = 0
local previous_level = 0
local previous_alive_state = 1 -- Track previous alive state, initialize as alive
local LastRewardState = 0
-- Track superzapper usage and activation edge per level (for one-time charging)
local previous_superzapper_active = 0
local previous_superzapper_uses_in_level = 0

-- Helper function to find nearest enemy of a specific type (copied from state.lua for locality within logic)
-- NOTE: Duplicated from state.lua for now to keep logic self-contained. Consider unifying later.
local function find_nearest_enemy_of_type(enemies_state, player_abs_segment, is_open, enemy_type, abs_to_rel_func)
    local nearest_seg_abs = -1
    local min_dist = 255
    local nearest_depth = 255

    for i = 1, 7 do
        if enemies_state.enemy_core_type[i] == enemy_type and enemies_state.enemy_depths[i] > 0 then
            local enemy_abs_seg = enemies_state.enemy_abs_segments[i]
            if enemy_abs_seg ~= INVALID_SEGMENT then
                local rel_dist = abs_to_rel_func(player_abs_segment, enemy_abs_seg, is_open)
                local abs_dist = math.abs(rel_dist)

                if abs_dist < min_dist then
                    min_dist = abs_dist
                    nearest_seg_abs = enemy_abs_seg
                    nearest_depth = enemies_state.enemy_depths[i]
                -- Tie-breaking: shallower enemy is preferred
                elseif abs_dist == min_dist and enemies_state.enemy_depths[i] < nearest_depth then
                    nearest_seg_abs = enemy_abs_seg
                    nearest_depth = enemies_state.enemy_depths[i]
                end
            end
        end
    end
    return nearest_seg_abs, nearest_depth
end


-- Function to get the relative distance to a target segment
function M.absolute_to_relative_segment(current_abs_segment, target_abs_segment, is_open_level)
    current_abs_segment = tonumber(current_abs_segment) or 0
    target_abs_segment = tonumber(target_abs_segment) or 0
    current_abs_segment = math.floor(current_abs_segment) % 16
    target_abs_segment = math.floor(target_abs_segment) % 16

    if is_open_level then
        return target_abs_segment - current_abs_segment
    else
        local diff = target_abs_segment - current_abs_segment
        if diff > 8 then return diff - 16
        elseif diff <= -8 then return diff + 16
        else return diff end
    end
end

-- Helper function to hunt enemies in preference order
function M.hunt_enemies(enemies_state, player_abs_segment, is_open, abs_to_rel_func, forbidden_segments)
    local hunt_order = {
        ENEMY_TYPE_FUSEBALL, ENEMY_TYPE_FLIPPER, ENEMY_TYPE_TANKER, ENEMY_TYPE_SPIKER
    }
    for _, enemy_type in ipairs(hunt_order) do
        local target_seg_abs, target_depth = find_nearest_enemy_of_type(enemies_state, player_abs_segment, is_open, enemy_type, abs_to_rel_func)
        if target_seg_abs ~= -1 then
            -- NOTE: Top rail flipper avoidance (depth 0x10) is removed here, handled by new logic in find_target_segment
            -- if target_depth <= 0x10 and enemy_type == ENEMY_TYPE_FLIPPER then ... end
            return target_seg_abs, target_depth, false -- Returns target_abs, depth, should_fire (always false from hunt)
        end
    end
    return -1, 255, false
end

-- Function to handle tube zoom state (0x20)
function M.zoom_down_tube(player_abs_seg, level_state, is_open)
    local current_spike_h = level_state.spike_heights[player_abs_seg]
    if current_spike_h == 0 then return player_abs_seg, 0, true, false end

    local left_neighbour_seg, right_neighbour_seg = -1, -1
    if is_open then
        if player_abs_seg > 0 then left_neighbour_seg = player_abs_seg - 1 end
        if player_abs_seg < 15 then right_neighbour_seg = player_abs_seg + 1 end
    else
        left_neighbour_seg = (player_abs_seg - 1 + 16) % 16
        right_neighbour_seg = (player_abs_seg + 1) % 16
    end

    local left_spike_h = (left_neighbour_seg ~= -1) and level_state.spike_heights[left_neighbour_seg] or -1
    local right_spike_h = (right_neighbour_seg ~= -1) and level_state.spike_heights[right_neighbour_seg] or -1

    if left_spike_h == 0 then return left_neighbour_seg, 0, true, false end
    if right_spike_h == 0 then return right_neighbour_seg, 0, true, false end

    local temp_target = player_abs_seg
    local is_left_better = (left_spike_h > current_spike_h)
    local is_right_better = (right_spike_h > current_spike_h)

    if is_left_better and is_right_better then
        temp_target = (left_spike_h >= right_spike_h) and left_neighbour_seg or right_neighbour_seg
    elseif is_left_better then temp_target = left_neighbour_seg
    elseif is_right_better then temp_target = right_neighbour_seg end

    return temp_target, 0, true, false
end

-- Function to check for fuseball threats
function M.fuseball_check(player_abs_seg, enemies_state, is_open, abs_to_rel_func)
    local fuseball_threat_nearby = false
    local escape_target_seg = -1
    for i = 1, 7 do
        if enemies_state.enemy_core_type[i] == ENEMY_TYPE_FUSEBALL and
           enemies_state.enemy_depths[i] <= 0x40 and
           enemies_state.enemy_abs_segments[i] ~= INVALID_SEGMENT then
            local fuseball_abs_seg = enemies_state.enemy_abs_segments[i]
            local rel_dist = abs_to_rel_func(player_abs_seg, fuseball_abs_seg, is_open)
            if math.abs(rel_dist) <= 2 then
                fuseball_threat_nearby = true
                escape_target_seg = (rel_dist <= 0) and ((player_abs_seg + 3) % 16) or ((player_abs_seg - 3 + 16) % 16)
                break
            end
        end
    end
    return fuseball_threat_nearby, escape_target_seg
end

-- Function to check for pulsar threats
function M.pulsar_check(player_abs_seg, enemies_state, is_open, abs_to_rel_func, forbidden_segments)
    if enemies_state.pulsing < PULSAR_THRESHOLD then return false, player_abs_seg, 0, false, false end

    local is_in_pulsar_lane = false
    local current_pulsar_seg = -1
    for i = 1, 7 do
        if enemies_state.enemy_core_type[i] == ENEMY_TYPE_PULSAR and
           enemies_state.enemy_depths[i] > 0 then
            if enemies_state.enemy_abs_segments[i] == player_abs_seg then
                is_in_pulsar_lane = true
                current_pulsar_seg = player_abs_seg
                break
            end
        end
    end

    if not is_in_pulsar_lane then return false, player_abs_seg, 0, false, false end

    local adj = adjacent_to_pulsar_closest_to_player(current_pulsar_seg, player_abs_seg, is_open, abs_to_rel_func)
    if adj == -1 then
        adj = find_nearest_non_pulsar_segment(player_abs_seg, enemies_state, is_open)
    end
    return true, adj, 0, false, false
end

-- Function to check for immediate threats in a segment
function M.check_segment_threat(segment, enemies_state)
    for i = 1, 4 do -- Check shots
        if enemies_state.enemy_shot_abs_segments[i] == segment and enemies_state.shot_positions[i] > 0 and enemies_state.shot_positions[i] <= 0x30 then return true end
    end
    for i = 1, 7 do -- Check enemies
        if enemies_state.enemy_abs_segments[i] == segment and enemies_state.enemy_depths[i] > 0 and enemies_state.enemy_depths[i] <= 0x30 then return true end
    end
    for i = 1, 7 do -- Check pulsars (regardless of pulsing state for general threat)
        if enemies_state.enemy_core_type[i] == ENEMY_TYPE_PULSAR and enemies_state.enemy_abs_segments[i] == segment and enemies_state.enemy_depths[i] > 0 then return true end
    end
    return false
end

-- Returns true if the segment is a danger lane (more specific criteria for immediate action)
function M.is_danger_lane(segment, enemies_state)
    if enemies_state.pulsing >= PULSAR_THRESHOLD then -- Check dangerous pulsars first
        for i = 1, 7 do
            if enemies_state.enemy_core_type[i] == ENEMY_TYPE_PULSAR and enemies_state.enemy_abs_segments[i] == segment and enemies_state.enemy_depths[i] > 0 then return true end
        end
    end
    for i = 1, 7 do -- Check enemies with type-specific danger distances
        if enemies_state.enemy_abs_segments[i] == segment and enemies_state.enemy_depths[i] > 0 then
            local enemy_type = enemies_state.enemy_core_type[i]
            local depth = enemies_state.enemy_depths[i]
            -- Fuseballs are fatal on contact, so they're dangerous at greater distances on top rail
            if enemy_type == ENEMY_TYPE_FUSEBALL and depth <= TOP_RAIL_DEPTH then return true end
            -- Other enemies are dangerous when close
            if depth <= 0x20 then return true end
        end
    end
    for i = 1, 4 do -- Check close enemy shots (depth <= 0x30)
        if enemies_state.enemy_shot_abs_segments[i] == segment and enemies_state.shot_positions[i] > 0 and enemies_state.shot_positions[i] <= 0x30 then return true end
    end
    return false
end

-- Helper function to find the segment of the nearest enemy at depth 0x10
function M.find_nearest_top_rail_enemy_seg(player_abs_seg, enemies_state, abs_to_rel_func, is_open)
    local nearest_enemy_seg, min_dist = -1, 255
    for i = 1, 7 do
        if enemies_state.enemy_depths[i] == 0x10 then
            local enemy_seg = enemies_state.enemy_abs_segments[i]
            if enemy_seg ~= INVALID_SEGMENT then
                local abs_dist = math.abs(abs_to_rel_func(player_abs_seg, enemy_seg, is_open))
                if abs_dist < min_dist then min_dist = abs_dist; nearest_enemy_seg = enemy_seg end
            end
        end
    end
    return nearest_enemy_seg
end

-- Returns the highest priority enemy type and its priority value in a segment
function M.get_enemy_priority(segment, enemies_state)
    local best_priority = 100
    local best_type = nil
    local priority_map = {[ENEMY_TYPE_PULSAR]=1, [ENEMY_TYPE_FLIPPER]=2, [ENEMY_TYPE_TANKER]=3, [ENEMY_TYPE_FUSEBALL]=4, [ENEMY_TYPE_SPIKER]=5}
    for i = 1, 7 do
        if enemies_state.enemy_abs_segments[i] == segment and enemies_state.enemy_depths[i] > 0 then
            local t = enemies_state.enemy_core_type[i]
            local p = priority_map[t] or 99
            if p < best_priority then best_priority = p; best_type = t end
        end
    end
    return best_type, best_priority
end

-- Public API: configure conserve fire mode
function M.set_conserve_fire_mode(enabled, react_distance)
    CONSERVE_FIRE_MODE = not not enabled
    if type(react_distance) == "number" and react_distance >= 0.5 then
        CONSERVE_REACT_DISTANCE = react_distance
    end
end

function M.get_conserve_fire_mode()
    return CONSERVE_FIRE_MODE, CONSERVE_REACT_DISTANCE
end

-- Helper to find the nearest safe segment (not a danger lane)
local function find_nearest_safe_segment(start_seg, enemies_state, is_open)
    if not M.is_danger_lane(start_seg, enemies_state) then return start_seg end

    for d = 1, 8 do -- Search radius up to 8
        local left_seg = (start_seg - d + 16) % 16
        if not M.is_danger_lane(left_seg, enemies_state) then return left_seg end

        local right_seg = (start_seg + d + 16) % 16
        if not M.is_danger_lane(right_seg, enemies_state) then return right_seg end
    end
    return start_seg -- Fallback: stay put if no safe found nearby
end

-- NEW Helper: Check if a segment contains an active Pulsar
local function is_pulsar_lane(segment, enemies_state)
    for i = 1, 7 do
        if enemies_state.enemy_core_type[i] == ENEMY_TYPE_PULSAR and
           enemies_state.enemy_abs_segments[i] == segment and
           enemies_state.enemy_depths[i] > 0 then
            return true
        end
    end
    return false
end

-- Public API: configure pulsar hunting preferences
function M.set_pulsar_preference(distance, tolerance)
    if type(distance) == "number" and distance >= 1 then PULSAR_PREF_DISTANCE = distance end
    if type(tolerance) == "number" and tolerance >= 0 then PULSAR_PREF_TOLERANCE = tolerance end
end

function M.get_pulsar_preference()
    return PULSAR_PREF_DISTANCE, PULSAR_PREF_TOLERANCE
end

-- Helper: Get the lane adjacent to a given pulsar that's closest to the player
local function adjacent_to_pulsar_closest_to_player(pulsar_seg, player_seg, is_open, abs_to_rel_func)
    local left_adj = (pulsar_seg - 1 + 16) % 16
    local right_adj = (pulsar_seg + 1) % 16
    if is_open then
        -- Clamp for open levels
        left_adj = (left_adj >= 0 and left_adj <= 15) and left_adj or -1
        right_adj = (right_adj >= 0 and right_adj <= 15) and right_adj or -1
    end
    local best = -1
    local best_dist = 999
    if left_adj ~= -1 then
        local d = math.abs(abs_to_rel_func(player_seg, left_adj, is_open))
        if d < best_dist then best, best_dist = left_adj, d end
    end
    if right_adj ~= -1 then
        local d = math.abs(abs_to_rel_func(player_seg, right_adj, is_open))
        if d < best_dist then best, best_dist = right_adj, d end
    end
    return best
end

-- Helper: Find nearest segment that is NOT a pulsar lane (ignores other dangers by design)
local function find_nearest_non_pulsar_segment(start_seg, enemies_state, is_open)
    if not is_pulsar_lane(start_seg, enemies_state) then return start_seg end
    for d = 1, 8 do
        local left_seg = is_open and (start_seg - d) or ((start_seg - d + 16) % 16)
        local right_seg = is_open and (start_seg + d) or ((start_seg + d) % 16)
        if left_seg >= 0 and left_seg <= 15 and not is_pulsar_lane(left_seg, enemies_state) then return left_seg end
        if right_seg >= 0 and right_seg <= 15 and not is_pulsar_lane(right_seg, enemies_state) then return right_seg end
    end
    return start_seg -- fallback (should be rare)
end

-- NEW Helper: Find nearest safe segment that also respects distance from a constraint segment
local function find_nearest_constrained_safe_segment(start_seg, enemies_state, is_open, constraint_seg, abs_to_rel_func)
    -- Search outwards from the start segment
    for d = 0, 8 do -- Check current segment first (d=0), then outwards
        local segments_to_check = {}
        if d == 0 then
            segments_to_check = {start_seg}
        else
            local left_seg = (start_seg - d + 16) % 16
            local right_seg = (start_seg + d + 16) % 16
            segments_to_check = {left_seg, right_seg}
        end

        for _, check_seg in ipairs(segments_to_check) do
            if not M.is_danger_lane(check_seg, enemies_state) then
                local dist_to_constraint = math.abs(abs_to_rel_func(check_seg, constraint_seg, is_open))
                if dist_to_constraint >= SAFE_DISTANCE then
                    -- Found a segment that is safe AND respects the distance constraint
                    return check_seg
                end
            end
        end
    end

    -- Fallback: If no segment satisfies both, return the simple nearest safe segment
    return start_seg -- NEW FALLBACK: Prefer original unsafe target over potentially worse simple safe target
end

-- Function to find the target segment and recommended action (expert policy)
function M.find_target_segment(game_state, player_state, level_state, enemies_state, abs_to_rel_func)
    local is_open = (level_state.level_type == 0xFF)
    local player_abs_seg = math.floor(player_state.position) % 16
    local shot_count = player_state.shot_count or 0

    -- Default return values
    local proposed_target_seg = player_abs_seg
    local proposed_fire_prio = 6 -- Default fire priority
    local final_should_fire = false

    if game_state.gamestate == 0x20 then -- Tube Zoom
        proposed_target_seg, _, final_should_fire, _ = M.zoom_down_tube(player_abs_seg, level_state, is_open)
        -- In tube zoom, fire priority isn't calculated the same way, should_fire is directly returned
        -- We will bypass the final Pulsar check for Tube Zoom state
        return proposed_target_seg, 0, final_should_fire, false

    elseif game_state.gamestate == 0x04 then -- Normal Gameplay
        -- === Step 1: Determine proposed target based on Top Rail / Hunt ===

        -- Scan Top Rail Enemies
    local nl_seg, nr_seg = nil, nil
    local nl_dist, nr_dist = 100, 100
    local nl_dist_float, nr_dist_float = 100, 100
        local enemy_left_exists, enemy_right_exists = false, false
        local any_enemy_proximate = false
    local top_rail_enemies = {}
    local nearest_flipper_seg = nil -- Specifically track nearest flipper
    local nearest_flipper_rel = nil
    local min_flipper_abs_rel = 100
    local nearest_flipper_frac_pos = 0 -- Track fractional position
    local nearest_pulsar_seg = nil -- Track nearest top-rail pulsar
    local nearest_pulsar_rel = nil
    local min_pulsar_abs_rel = 100

        for i = 1, 7 do
            local depth = enemies_state.enemy_depths[i]
            if depth > 0 and depth <= TOP_RAIL_DEPTH then
                local seg = enemies_state.enemy_abs_segments[i]
                if seg ~= INVALID_SEGMENT then
                    -- Get fractional position (scaled to 0-1 range)
                    local frac_pos = 0
                    if enemies_state.fractional_enemy_segments_by_slot[i] ~= INVALID_SEGMENT then
                        frac_pos = enemies_state.fractional_enemy_segments_by_slot[i] / 4096.0
                    end
                    
                    -- Get both integer and floating point relative positions
                    local rel_int = abs_to_rel_func(player_abs_seg, seg, is_open)
                    
                    -- Store fractional position for detailed calculations but use integer position for main logic
                    local rel_float = rel_int
                    if rel_int > 0 then
                        rel_float = rel_int + frac_pos
                    elseif rel_int < 0 then
                        rel_float = rel_int - (1 - frac_pos)  -- Adjust for negative direction
                    end
                    
                    -- Use integer-based absolute distance for coarse comparisons; keep float for precision
                    local abs_rel_int = math.abs(rel_int)
                    local abs_rel_float = math.abs(rel_float)
                    local core_type = enemies_state.enemy_core_type[i]

                    if is_open then
                        if abs_rel_float <= OPEN_FLIPPER_REACT_DISTANCE then any_enemy_proximate = true end
                    else
                        if abs_rel_int <= 1 then any_enemy_proximate = true end
                    end
                    table.insert(top_rail_enemies, {
                        seg = seg, 
                        rel = rel_int,  -- Store integer position for main comparisons
                        rel_float = rel_float, -- Store float for detailed calculations
                        abs_rel = abs_rel_int, -- Use integer-based distance
                        type = core_type,
                        frac_pos = frac_pos
                    })

                    -- Update nearest overall left/right (using integer positions for critical logic)
                    if rel_int > 0 and abs_rel_int < nr_dist then 
                        nr_dist = abs_rel_int
                        nr_dist_float = abs_rel_float
                        nr_seg = seg
                        enemy_right_exists = true 
                    end
                    
                    if rel_int < 0 and abs_rel_int < nl_dist then 
                        nl_dist = abs_rel_int
                        nl_dist_float = abs_rel_float
                        nl_seg = seg
                        enemy_left_exists = true 
                    end

                    -- Update nearest flipper specifically - use floating point for detailed positioning
                    if core_type == ENEMY_TYPE_FLIPPER then
                        if abs_rel_float < min_flipper_abs_rel then
                             min_flipper_abs_rel = abs_rel_float
                             nearest_flipper_seg = seg
                             nearest_flipper_rel = rel_float
                             nearest_flipper_frac_pos = frac_pos
                        end
                    elseif core_type == ENEMY_TYPE_PULSAR then
                        if abs_rel_float < min_pulsar_abs_rel then
                            min_pulsar_abs_rel = abs_rel_float
                            nearest_pulsar_seg = seg
                            nearest_pulsar_rel = rel_float
                        end
                    end
                end
            end
        end

    -- === Step 1A: HIGHEST Priority Fuseball Avoidance ===
        -- Check for any fuseballs on top rail and avoid them immediately (fatal on contact)
        local nearest_fuseball_seg = nil
        local nearest_fuseball_rel = nil
        local min_fuseball_distance = 100
        
        for _, enemy in ipairs(top_rail_enemies) do
            if enemy.type == ENEMY_TYPE_FUSEBALL then
                local fuseball_distance = math.abs(enemy.rel)
                if fuseball_distance < min_fuseball_distance then
                    min_fuseball_distance = fuseball_distance
                    nearest_fuseball_seg = enemy.seg
                    nearest_fuseball_rel = enemy.rel
                end
            end
        end
        
        if nearest_fuseball_seg and min_fuseball_distance <= 2 then -- Fuseballs are dangerous within 2 segments
            -- Move at least 3 segments away from the fuseball
            local direction_away_from_fuseball = (nearest_fuseball_rel == 0) and 1 or (nearest_fuseball_rel > 0 and -1 or 1)
            local target_seg = (nearest_fuseball_seg + direction_away_from_fuseball * 3 + 16) % 16
            
            -- Safety check the escape target
            if M.is_danger_lane(target_seg, enemies_state) then
                target_seg = find_nearest_safe_segment(target_seg, enemies_state, is_open)
            end
            
            local fire_priority = AVOID_FIRE_PRIORITY -- Use specific avoid priority
            local should_fire = fire_priority > shot_count
            return target_seg, 0, should_fire, false -- Return early - fuseball avoidance is critical
        end

        -- Conserve Fire Mode override: when top-rail flippers/pulsars are present, hold position and fire only when within react distance
        local conserve_override_active = false
        if CONSERVE_FIRE_MODE and (nearest_flipper_seg or nearest_pulsar_seg) then
            conserve_override_active = true
            proposed_target_seg = player_abs_seg -- hold position; safety overrides may still move us later
            local nearest_abs = 999
            if nearest_flipper_seg then nearest_abs = math.min(nearest_abs, min_flipper_abs_rel) end
            if nearest_pulsar_seg then nearest_abs = math.min(nearest_abs, min_pulsar_abs_rel) end
            local threat_near_lane = M.check_segment_threat(player_abs_seg, enemies_state)
            local within_react = nearest_abs <= CONSERVE_REACT_DISTANCE
            -- Stop hunting/firing unless threat in our lane or within react distance
            proposed_fire_prio = (threat_near_lane or within_react) and FREEZE_FIRE_PRIO_HIGH or 0
        end

    -- === Step 1B: High Priority Flipper/Pulsar Avoidance ===
        if not conserve_override_active and nearest_flipper_seg and min_flipper_abs_rel < SAFE_DISTANCE then

            -- Calculate target D segments away from the flipper, in the direction opposite the flipper from the player
            local direction_away_from_flipper = (nearest_flipper_rel == 0) and 1 or (nearest_flipper_rel > 0 and -1 or 1)
            local target_seg = (nearest_flipper_seg + direction_away_from_flipper * SAFE_DISTANCE + 16) % 16

            -- Safety Check for the calculated avoidance target
            if M.is_danger_lane(target_seg, enemies_state) then
                 target_seg = find_nearest_constrained_safe_segment(target_seg, enemies_state, is_open, nearest_flipper_seg, abs_to_rel_func)
            end
            local fire_priority = AVOID_FIRE_PRIORITY -- Use specific avoid priority
            local should_fire = fire_priority > shot_count
            return target_seg, 0, should_fire, false -- Return early
        elseif not conserve_override_active and nearest_pulsar_seg and min_pulsar_abs_rel < SAFE_DISTANCE then
            -- Calculate target D segments away from the pulsar, in the direction opposite the pulsar from the player
            local direction_away_from_pulsar = (nearest_pulsar_rel == 0) and 1 or (nearest_pulsar_rel > 0 and -1 or 1)
            local target_seg = (nearest_pulsar_seg + direction_away_from_pulsar * PULSAR_TARGET_DISTANCE + 16) % 16

            -- Safety check the avoidance target
            if M.is_danger_lane(target_seg, enemies_state) then
                target_seg = find_nearest_constrained_safe_segment(target_seg, enemies_state, is_open, nearest_pulsar_seg, abs_to_rel_func)
            end
            -- Never move onto a pulsar lane
            if is_pulsar_lane(target_seg, enemies_state) then
                local alt = (direction_away_from_pulsar == 1) and ((nearest_pulsar_seg - PULSAR_TARGET_DISTANCE + 16) % 16) or ((nearest_pulsar_seg + PULSAR_TARGET_DISTANCE) % 16)
                if not is_pulsar_lane(alt, enemies_state) then target_seg = alt
                else target_seg = find_nearest_non_pulsar_segment(target_seg, enemies_state, is_open) end
            end
            local fire_priority = AVOID_FIRE_PRIORITY
            local should_fire = fire_priority > shot_count
            return target_seg, 0, should_fire, false -- Return early
        end

    -- === Step 1C: Top-Rail Flipper/Pulsar Hunt-to-Adjacency (Closed levels) ===
    -- On closed levels, proactively move to be exactly one segment away from the nearest top-rail flipper or pulsar.
    -- For pulsars specifically, NEVER target the pulsar lane; target the adjacent lane closest to the player.
    if not conserve_override_active and not is_open then
            local target_threat_seg, target_threat_rel, target_threat_type, target_abs_rel = nil, nil, nil, 100
            if nearest_flipper_seg and min_flipper_abs_rel < target_abs_rel then
                target_threat_seg = nearest_flipper_seg
                target_threat_rel = nearest_flipper_rel
                target_threat_type = ENEMY_TYPE_FLIPPER
                target_abs_rel = min_flipper_abs_rel
            end
            if nearest_pulsar_seg and min_pulsar_abs_rel < target_abs_rel then
                target_threat_seg = nearest_pulsar_seg
                target_threat_rel = nearest_pulsar_rel
                target_threat_type = ENEMY_TYPE_PULSAR
                target_abs_rel = min_pulsar_abs_rel
            end

        if target_threat_seg then
                local target_seg = -1
                if target_threat_type == ENEMY_TYPE_PULSAR then
                    -- Hunt pulsar from fixed offset (segments away) toward the player side; never target pulsar lane
                    local dir_to_player = abs_to_rel_func(target_threat_seg, player_abs_seg, is_open)
                    local dir_sign = (dir_to_player > 0) and 1 or (dir_to_player < 0 and -1 or 1)
                    target_seg = (target_threat_seg + dir_sign * PULSAR_TARGET_DISTANCE + 16) % 16
                else
                    -- Flipper: position one away opposite the threat direction
                    local dir_toward_threat = (target_threat_rel > 0) and 1 or (target_threat_rel < 0 and -1 or 0)
                    local one_away_dir = (dir_toward_threat > 0) and -1 or (dir_toward_threat < 0 and 1 or 1)
                    target_seg = (target_threat_seg + one_away_dir + 16) % 16
                end

                if M.is_danger_lane(target_seg, enemies_state) then
                    -- Keep at least SAFE_DISTANCE from the threat if possible
                    target_seg = find_nearest_constrained_safe_segment(target_seg, enemies_state, is_open, target_threat_seg, abs_to_rel_func)
                end
                -- Never target a pulsar lane
                if is_pulsar_lane(target_seg, enemies_state) then
                    target_seg = find_nearest_non_pulsar_segment(target_seg, enemies_state, is_open)
                end

                local fire_priority
                if target_threat_type == ENEMY_TYPE_PULSAR then
                    -- Use fractional distance to pulsar to decide hold/fire vs reposition
                    local current_dist_float = math.abs(target_threat_rel or 0)
                    local within = math.abs(current_dist_float - PULSAR_PREF_DISTANCE) <= PULSAR_PREF_TOLERANCE
                    fire_priority = within and FREEZE_FIRE_PRIO_HIGH or 6
                else
                    fire_priority = (target_abs_rel <= 1.05) and FREEZE_FIRE_PRIO_HIGH or 6
                end
                local should_fire = fire_priority > shot_count
                return target_seg, 0, should_fire, false -- Early return to enforce adjacency strategy
            end
        end

        -- === Step 1B: Determine proposed target based on Top Rail (Presence) / Hunt ===
        local base_fire_priority = any_enemy_proximate and FREEZE_FIRE_PRIO_HIGH or FREEZE_FIRE_PRIO_LOW
        if not conserve_override_active then
            proposed_fire_prio = base_fire_priority -- Start with base priority (might be overridden by safety checks below)
        end

    if not conserve_override_active and enemy_left_exists and enemy_right_exists then
            -- Case 1: Both Sides - Pin to nearest threat strategy
            -- Find the nearest threat (fuseball, pulsar, or flipper - fuseballs have highest priority)
            local nearest_threat_seg = nil
            local nearest_threat_rel = nil
            local min_threat_distance = 100
            local nearest_threat_type = nil
            
            for _, enemy in ipairs(top_rail_enemies) do
                if enemy.type == ENEMY_TYPE_FUSEBALL or enemy.type == ENEMY_TYPE_FLIPPER or enemy.type == ENEMY_TYPE_PULSAR then
                    local threat_distance = math.abs(enemy.rel)
                    -- Fuseballs get priority due to fatal contact, then closest distance
                    local is_higher_priority = false
                    if enemy.type == ENEMY_TYPE_FUSEBALL and nearest_threat_type ~= ENEMY_TYPE_FUSEBALL then
                        is_higher_priority = true
                    elseif enemy.type == ENEMY_TYPE_FUSEBALL and nearest_threat_type == ENEMY_TYPE_FUSEBALL and threat_distance < min_threat_distance then
                        is_higher_priority = true
                    elseif nearest_threat_type ~= ENEMY_TYPE_FUSEBALL and threat_distance < min_threat_distance then
                        is_higher_priority = true
                    end
                    
                    if is_higher_priority then
                        min_threat_distance = threat_distance
                        nearest_threat_seg = enemy.seg
                        nearest_threat_rel = enemy.rel
                        nearest_threat_type = enemy.type
                    end
                end
            end
            
            if nearest_threat_seg then
                -- Pin to appropriate distance from nearest threat on the near side
                if nearest_threat_type == ENEMY_TYPE_PULSAR then
                    -- Hunt pulsar from fixed offset toward the player side
                    local dir_to_player = abs_to_rel_func(nearest_threat_seg, player_abs_seg, is_open)
                    local dir_sign = (dir_to_player > 0) and 1 or (dir_to_player < 0 and -1 or 1)
                    proposed_target_seg = (nearest_threat_seg + dir_sign * PULSAR_TARGET_DISTANCE + 16) % 16
                else
                    local safe_distance = (nearest_threat_type == ENEMY_TYPE_FUSEBALL) and 2 or 1 -- Fuseballs need more distance
                    if nearest_threat_rel > 0 then
                        -- Threat is to the right, pin safe_distance segments to the left of it
                        proposed_target_seg = (nearest_threat_seg - safe_distance + 16) % 16
                    elseif nearest_threat_rel < 0 then
                        -- Threat is to the left, pin safe_distance segments to the right of it
                        proposed_target_seg = (nearest_threat_seg + safe_distance) % 16
                    else
                        proposed_target_seg = find_nearest_safe_segment(player_abs_seg, enemies_state, is_open)
                    end
                end
                
                -- Safety check the pinned position
                if M.is_danger_lane(proposed_target_seg, enemies_state) then
                    proposed_target_seg = find_nearest_safe_segment(proposed_target_seg, enemies_state, is_open)
                end
                -- Never target pulsar lane
                if is_pulsar_lane(proposed_target_seg, enemies_state) then
                    proposed_target_seg = find_nearest_non_pulsar_segment(proposed_target_seg, enemies_state, is_open)
                end
            else
                -- No critical threats found, fall back to safety-based positioning
                if M.is_danger_lane(player_abs_seg, enemies_state) then
                    proposed_target_seg = find_nearest_safe_segment(player_abs_seg, enemies_state, is_open)
                else
                    proposed_target_seg = player_abs_seg
                end
            end
    elseif not conserve_override_active and enemy_right_exists then
            -- Case 2: Right Only - Use adaptive distance approach
            if is_open then 
                -- Use fractional distance to improve timing on open levels
        proposed_target_seg = (nr_dist_float <= OPEN_FLIPPER_REACT_DISTANCE) and 0 or 1
            else 
                if nr_dist < SAFE_DISTANCE then
                    -- Use alternating safe distance based on distance
                    local safe_offset = math.floor(nr_dist * 2) + 1  -- Calculate offset based on distance
                    safe_offset = math.max(1, math.min(2, safe_offset))  -- Keep between 1-2
                    proposed_target_seg = (nr_seg - safe_offset + 16) % 16
                else 
                    proposed_target_seg = player_abs_seg 
                end 
            end
    elseif not conserve_override_active and enemy_left_exists then
            -- Case 3: Left Only - Mirror the right-only strategy for consistency  
            if is_open then 
                -- Revert to integer distance for left-only on open fields (works better for lane-1 behavior)
        proposed_target_seg = (nl_dist_float <= OPEN_FLIPPER_REACT_DISTANCE) and 13 or 14
            else 
                if nl_dist < SAFE_DISTANCE then
                    -- Use alternating safe distance based on distance
                    local safe_offset = math.floor(nl_dist * 2) + 1  -- Calculate offset based on distance
                    safe_offset = math.max(1, math.min(2, safe_offset))  -- Keep between 1-2
                    proposed_target_seg = (nl_seg + safe_offset) % 16
                else 
                    proposed_target_seg = player_abs_seg 
                end 
            end
        else
            -- Case 4: No Top Rail -> Standard Hunt Logic
            if not conserve_override_active and M.is_danger_lane(player_abs_seg, enemies_state) then
                 proposed_target_seg = find_nearest_safe_segment(player_abs_seg, enemies_state, is_open)
                 proposed_fire_prio = AVOID_FIRE_PRIORITY -- Lower priority when escaping
            elseif not conserve_override_active and M.get_enemy_priority(player_abs_seg, enemies_state) then
                 proposed_target_seg = player_abs_seg
                 proposed_fire_prio = 6
            elseif not conserve_override_active then -- Search outwards
                local fallback_safe, best_enemy_prio, found_target = nil, 100, false
                local search_target = player_abs_seg
                for sign = -1, 1, 2 do
                    for d = 1, 8 do
                        local seg = (player_abs_seg + sign * d + 16) % 16
                        if M.is_danger_lane(seg, enemies_state) then break end
                        local type, prio = M.get_enemy_priority(seg, enemies_state)
                        if type and prio < best_enemy_prio then search_target, best_enemy_prio, found_target, proposed_fire_prio = seg, prio, true, 6 end
                        if not type and not found_target and fallback_safe == nil then fallback_safe = seg end
                    end
                end
                if not found_target and fallback_safe then proposed_target_seg = fallback_safe; proposed_fire_prio = 4
                elseif not found_target then proposed_target_seg = player_abs_seg; proposed_fire_prio = 4
                else proposed_target_seg = search_target end -- Use found target
            else
                -- Conserve mode active and no top-rail flippers/pulsars: revert to base priority
                proposed_target_seg = player_abs_seg
                if proposed_fire_prio == nil then proposed_fire_prio = base_fire_priority end
            end
        end

        -- === Step 2: Apply Safety Overrides (including Pulsar) ===
        local final_target_seg = proposed_target_seg
        local final_fire_priority = proposed_fire_prio
        local pulsar_override_active = false

        -- ** Pulsar Policy (Highest Priority): never target pulsar lanes; evacuate hot pulsar lanes immediately **
        if enemies_state.pulsing >= PULSAR_THRESHOLD and is_pulsar_lane(player_abs_seg, enemies_state) then
            final_target_seg = find_nearest_non_pulsar_segment(player_abs_seg, enemies_state, is_open)
            final_fire_priority = AVOID_FIRE_PRIORITY
            pulsar_override_active = true
        end
        -- Never target a pulsar lane as the final target
        if not pulsar_override_active and is_pulsar_lane(final_target_seg, enemies_state) then
            final_target_seg = find_nearest_non_pulsar_segment(final_target_seg, enemies_state, is_open)
        end
        -- When moving, if path into a pulsar lane, stop right before it
        if not pulsar_override_active and final_target_seg ~= player_abs_seg then
            local relative_dist = abs_to_rel_func(player_abs_seg, final_target_seg, is_open)
            local dir = (relative_dist > 0) and 1 or -1
            local steps = math.abs(relative_dist)
            for d = 1, steps do
                local check_seg = (player_abs_seg + dir * d + 16) % 16
                if is_pulsar_lane(check_seg, enemies_state) then
                    final_target_seg = (player_abs_seg + dir * (d - 1) + 16) % 16
                    final_fire_priority = AVOID_FIRE_PRIORITY
                    break
                end
            end
        end

        -- ** General Danger Lane Check (Lower priority than Pulsar) **
        -- Apply only if Pulsar override didn't happen AND the proposed target requires moving
        if not pulsar_override_active and final_target_seg ~= player_abs_seg and M.is_danger_lane(final_target_seg, enemies_state) then
            final_target_seg = find_nearest_safe_segment(final_target_seg, enemies_state, is_open)
            final_fire_priority = AVOID_FIRE_PRIORITY
        end

        -- === Step 3: Final Return ===
        -- Final pulsar-lane sanitization
        if is_pulsar_lane(final_target_seg, enemies_state) then
            final_target_seg = find_nearest_non_pulsar_segment(final_target_seg, enemies_state, is_open)
        end
        final_should_fire = final_fire_priority > shot_count
        return final_target_seg, 0, final_should_fire, false

    else -- Other game states
        return player_abs_seg, 0, false, false
    end
end

-- Function to calculate desired spinner direction and distance to target enemy
function M.direction_to_nearest_enemy(game_state, level_state, player_state, enemies_state, abs_to_rel_func)
    local player_abs_seg = math.floor(player_state.position) % 16
    local is_open = (level_state.level_type == 0xFF)
    local target_abs_segment = enemies_state.nearest_enemy_abs_seg_internal or -1

    if target_abs_segment == -1 then return 0, 0, 255 end -- No target

    local relative_dist = abs_to_rel_func(player_abs_seg, target_abs_segment, is_open)
    if relative_dist == 0 then return 0, 0, enemies_state.nearest_enemy_depth_raw end -- Aligned

    local distance = math.abs(relative_dist)
    local intensity = math.min(0.9, 0.3 + (distance * 0.05))
    local spinner = (relative_dist > 0) and intensity or -intensity
    return spinner, distance, 255 -- Misaligned (depth 255 indicates not aligned)
end

-- Function to calculate reward for the current frame
function M.calculate_reward(game_state, level_state, player_state, enemies_state, abs_to_rel_func)
    local reward, bDone = 0, false
    local detected_spinner = player_state.spinner_detected

    if player_state.alive == 1 then

        -- Level completion bonus and per-level resets
        if level_state.level_number > previous_level then
            reward = reward + 8000 -- explicit level completion reward (tuned)
            previous_superzapper_uses_in_level = 0 -- reset zap uses for new level
        end

        -- Player is alive; check to see if they are on a pulsar segment
        local player_segment = math.floor(player_state.position) % 16
    if is_pulsar_lane(player_segment, enemies_state) and enemies_state.pulsing >= PULSAR_THRESHOLD then
            reward = reward - 100 -- Deduct 100 reward if on a pulsar segment
        end

        -- If this is a fusebal charging lane, reward large spinner movement to escape

        if enemies_state.fuseball_threat_nearby then
            local fuseball_escape_target = enemies_state.fuseball_escape_target
            if fuseball_escape_target ~= -1 then
                local is_open = (level_state.level_type == 0xFF)
                local rel_dist = abs_to_rel_func(player_segment, fuseball_escape_target, is_open)
                -- Directional shaping: reward moving toward escape lane, penalize moving away or idling
                if rel_dist ~= 0 then
                    local need_dir = (rel_dist > 0) and 1 or -1
                    if (detected_spinner > 0 and need_dir > 0) or (detected_spinner < 0 and need_dir < 0) then
                        reward = reward + 60 -- moving in the right direction (tuned)
                    elseif detected_spinner == 0 then
                        reward = reward - 30 -- idle while threatened
                    else
                        reward = reward - 50 -- moving the wrong way (tuned)
                    end
                else
                    reward = reward + 50 -- reached/arrived at the escape lane
                end
                -- Small terminal bonus for being close to the escape lane
                if math.abs(rel_dist) <= 2 then reward = reward + 100 end
            end
        end

        


    local score_delta = player_state.score - previous_score
    if score_delta > 0 then reward = reward + (score_delta * 0.2) end -- scale points (tuned)

        local sc = player_state.shot_count
        if sc == 0 or sc >= 8 then reward = reward - 50
        elseif sc == 4 then reward = reward + 5; elseif sc == 5 then reward = reward + 10
        elseif sc == 6 then reward = reward + 15; elseif sc == 7 then reward = reward + 20 end

        -- Superzapper one-time charging per activation: -500 on first use in a level, -50 on second
        if game_state.gamestate == 0x04 then
            local sz_active = player_state.superzapper_active or 0
            if previous_superzapper_active == 0 and sz_active ~= 0 then
                local charge = (previous_superzapper_uses_in_level == 0) and 500 or 50
                reward = reward - charge
                previous_superzapper_uses_in_level = math.min(2, previous_superzapper_uses_in_level + 1)
            end
            previous_superzapper_active = sz_active
        end

        local target_abs_segment = enemies_state.nearest_enemy_abs_seg_internal
        local target_depth = enemies_state.nearest_enemy_depth_raw
        local player_segment = math.floor(player_state.position) % 16

        if game_state.gamestate == 0x20 then -- Tube Zoom Reward
            local spike_h = level_state.spike_heights[player_segment]
            if spike_h > 0 then reward = reward + math.max(0, ((255 - spike_h) / 2) - 27.5)
            else reward = reward + (detected_spinner == 0 and 250 or -50) end
            if player_state.fire_commanded == 1 then reward = reward + 200 end

        elseif target_abs_segment < 0 then -- No Enemies Reward
            reward = reward + (detected_spinner == 0 and 150 or -20)
            if player_state.fire_commanded == 1 then reward = reward - 100 end

        else -- Enemies Present Reward
            local desired_spinner, segment_distance, _ = M.direction_to_nearest_enemy(game_state, level_state, player_state, enemies_state, abs_to_rel_func)
            if segment_distance == 0 then -- Aligned Reward
                reward = reward + (detected_spinner == 0 and 250 or -10) -- relax penalty while aligned
                if player_state.fire_commanded == 1 then reward = reward + 50 end
            else -- Misaligned Reward
                if segment_distance < 2 and target_depth <= 0x20 then reward = reward + (player_state.fire_commanded == 1 and 150 or -50)
                else if player_state.fire_commanded == 1 then reward = reward - (segment_distance < 2 and 20 or 30) end end
                -- Movement reward
        if desired_spinner * detected_spinner > 0 then reward = reward + 50
        elseif desired_spinner * detected_spinner < 0 then reward = reward - 40
        elseif detected_spinner == 0 and desired_spinner ~= 0 then reward = reward - 20 end
            end
        end
    else -- Player Died Penalty
    if previous_alive_state == 1 then reward = reward - 16000; bDone = true end -- tuned
    end

    previous_score = player_state.score
    previous_level = level_state.level_number
    previous_alive_state = player_state.alive
    LastRewardState = reward

    return reward, bDone
end

-- Function to retrieve the last calculated reward (for display)
function M.getLastReward()
    return LastRewardState
end

return M 