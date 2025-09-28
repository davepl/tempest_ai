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
local TOP_RAIL_DEPTH = 0x20
local SAFE_DISTANCE = 1
local FLIPPER_WAIT_DISTANCE = 5 -- segments within which we prefer to wait and conserve shots on top rail
local FLIPPER_REACT_DISTANCE_R = 2.0 -- distance at which we move one segment and fire (right-side, float)
local FLIPPER_REACT_DISTANCE_L = 2.0 -- distance at which we move one segment and fire (left-side, float)
local FREEZE_FIRE_PRIO_LOW = 2
local FREEZE_FIRE_PRIO_HIGH = 8
local AVOID_FIRE_PRIORITY = 3
local PULSAR_THRESHOLD = 0xE0 -- Pulsing threshold for avoidance (match dangerous pulsar threshold)
-- Open-level tuning: react slightly sooner to top-rail flippers using fractional distance
local OPEN_FLIPPER_REACT_DISTANCE = 1.10
-- Retreat positions for open level flipper handling
local RIGHT_RETREAT_SEGMENT = 1   -- retreat to segment 1 when flippers to the right
local LEFT_RETREAT_SEGMENT = 13  -- retreat to segment 13 when flippers to the left  
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
-- Reward shaping parameters (tunable)
local SCORE_UNIT = 10000.0           -- 10k points ~= 1 life worth of reward
local LEVEL_COMPLETION_BONUS = 2.0   -- Edge-triggered bonus when level increments
local DEATH_PENALTY = 0.3            -- Edge-triggered penalty when dying (raised to better balance vs completion)
local ZAP_COST = 0.2                 -- Edge-triggered Small cost per zap frame
 
local previous_score = 0
local previous_level = 0
local previous_alive_state = 1 -- Track previous alive state, initialize as alive
local LastRewardState = 0
-- Track superzapper usage and activation edge per level (for one-time charging)
local previous_superzapper_active = 0
local previous_superzapper_uses_in_level = 0
local previous_zap_detected = 0
local previous_fire_detected = 0
-- Track previous player position for positioning reward
local previous_player_position = 0
-- Track score at the beginning of the current level (for optional ratio-based bonuses)
local score_at_level_start = 0

-- Track previous top-rail alignment (for progress reward)
local previous_toprail_min_abs_rel = nil

-- Track flipper count for elimination detection
local previous_flipper_count = nil

-- Track episode termination to prevent multiple bDone signals per episode
local episode_ended = false

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
        -- Wrap into [-8, +8] range, but treat exact ties (±8) neutrally
        if diff > 8 then
            diff = diff - 16
        elseif diff < -8 then
            diff = diff + 16
        end
        -- Neutral tie-breaker: if exactly opposite (distance 8), randomly pick direction
        if diff == 8 or diff == -8 then
            diff = (math.random() < 0.5) and 8 or -8
        end
        return diff
    end
end

-- NEW: Function to calculate effective angular distance between two segments
-- Returns effective distance where 1.0 = parallel, 0.5 = perpendicular, 0.0 = opposite
function M.calculate_angular_distance(player_abs_seg, enemy_abs_seg, level_state, is_open)
    -- Get angle values for both segments
    local player_angle = level_state.level_angles[player_abs_seg] or 0
    local enemy_angle = level_state.level_angles[enemy_abs_seg] or 0
    
    -- Calculate minimum angular difference (circular)
    local angle_diff = math.abs(player_angle - enemy_angle)
    if angle_diff > 8 then
        angle_diff = 16 - angle_diff
    end
    
    -- Convert to effective distance: 1.0 - (diff/8.0)
    -- This gives: 0° diff = 1.0, 90° diff = 0.5, 180° diff = 0.0
    local effective_distance = 1.0 - (angle_diff / 8.0)
    
    return effective_distance, angle_diff
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
    -- Simplified targeting logic per spec
    local is_open = (level_state.level_type == 0xFF)
    local player_abs_seg = math.floor(player_state.position) % 16
    local shot_count = player_state.shot_count or 0

    -- Tube Zoom behavior unchanged
    if game_state.gamestate == 0x20 then
        local seg, _, should_fire, _ = M.zoom_down_tube(player_abs_seg, level_state, is_open)
        return seg, 0, should_fire, false
    end
    if game_state.gamestate ~= 0x04 then
        return player_abs_seg, 0, false, false
    end

    -- Immediate fuseball avoidance: if a charging fuseball is in our lane or adjacent and near the top,
    -- move one segment away and keep firing. This preempts other targeting logic.
    do
        local FUSEBALL_NEAR_DEPTH = 0x50 -- consider near-top fuseballs (<= 0x50) as immediate threats
        local best_threat_rel = nil
        local best_threat_abs_seg = -1
        for i = 1, 7 do
            if enemies_state.enemy_core_type[i] == ENEMY_TYPE_FUSEBALL and
               enemies_state.enemy_abs_segments[i] ~= INVALID_SEGMENT then
                local depth = enemies_state.enemy_depths[i]
                local moving_away = (enemies_state.active_enemy_info and ((enemies_state.active_enemy_info[i] or 0) & 0x80) ~= 0) or false
                if depth > 0 and depth <= FUSEBALL_NEAR_DEPTH and not moving_away then
                    local rel = abs_to_rel_func(player_abs_seg, enemies_state.enemy_abs_segments[i], is_open)
                    local abs_rel = math.abs(rel)
                    if abs_rel <= 1 then
                        -- choose the closest such threat
                        if not best_threat_rel or abs_rel < math.abs(best_threat_rel) then
                            best_threat_rel = rel
                            best_threat_abs_seg = enemies_state.enemy_abs_segments[i]
                        end
                    end
                end
            end
        end
        if best_threat_rel ~= nil then
            local move_right = (best_threat_rel <= 0) -- threat aligned/left -> move right
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
            -- Fallback to the opposite side if open edge blocked
            if candidate == -1 then
                if move_right then
                    if is_open then
                        if player_abs_seg > 0 then candidate = player_abs_seg - 1 end
                    else
                        candidate = (player_abs_seg - 1 + 16) % 16
                    end
                else
                    if is_open then
                        if player_abs_seg < 15 then candidate = player_abs_seg + 1 end
                    else
                        candidate = (player_abs_seg + 1) % 16
                    end
                end
            end
            if candidate ~= -1 then
                return candidate, 0, true, false
            end
        end
    end

    -- Scan top-rail threats (flippers and pulsars are equivalent once on top rail)
    -- UPDATED: Now uses angular distances instead of simple segment distances
    local nr_seg, nl_seg = nil, nil
    local nr_angular_dist, nl_angular_dist = 999, 999
    local nr_dist_float, nl_dist_float = 999, 999
    local right_exists, left_exists = false, false
    local min_angular_effective = 999 -- nearest top-rail flipper/pulsar angular effective distance
    local min_angle_diff = 999 -- corresponding angular difference for head start adjustment
    for i = 1, 7 do
        local depth = enemies_state.enemy_depths[i]
        if depth > 0 and depth <= TOP_RAIL_DEPTH then
            local t = enemies_state.enemy_core_type[i]
            if t == ENEMY_TYPE_FLIPPER or t == ENEMY_TYPE_PULSAR then
                local seg = enemies_state.enemy_abs_segments[i]
                if seg ~= INVALID_SEGMENT then
                    -- Calculate angular distance instead of simple segment distance
                    local angular_effective, angular_diff = M.calculate_angular_distance(player_abs_seg, seg, level_state, is_open)
                    local rel_int = abs_to_rel_func(player_abs_seg, seg, is_open)
                    local rel_float = enemies_state.active_top_rail_enemies[i]
                    if rel_float == 0 and enemies_state.enemy_between_segments[i] == 0 then
                        rel_float = rel_int
                    end
                    local abs_int = math.abs(rel_int)
                    local abs_float = math.abs(rel_float)
                    
                    -- Use angular effective distance for primary threat assessment
                    if rel_int > 0 and angular_effective < nr_angular_dist then 
                        nr_angular_dist, nr_dist_float, nr_seg, right_exists = angular_effective, abs_float, seg, true 
                    end
                    if rel_int < 0 and angular_effective < nl_angular_dist then 
                        nl_angular_dist, nl_dist_float, nl_seg, left_exists = angular_effective, abs_float, seg, true 
                    end
                    if angular_effective < min_angular_effective then 
                        min_angular_effective = angular_effective 
                        min_angle_diff = angular_diff
                    end
                end
            end
        end
    end

    -- Open-level fixed retreats (right-only -> 1, left-only -> 13). Both sides falls through to closed-level logic
    local target_seg = player_abs_seg
    if is_open then
        if right_exists and not left_exists then
            target_seg = RIGHT_RETREAT_SEGMENT -- 1
        elseif left_exists and not right_exists then
            target_seg = LEFT_RETREAT_SEGMENT -- 13
        end
    end

    -- Closed-level (and open with both sides) movement policy
    -- UPDATED: Now uses angular distances for threat assessment
    -- Accounts for "head start" at narrower angles by adjusting movement thresholds
    if (not is_open) or (is_open and right_exists and left_exists) then
        -- Calculate dynamic thresholds based on head start
        local base_move_threshold = 0.8  -- Base threshold for ~60° angular proximity
        local _, right_angle_diff = nr_seg and M.calculate_angular_distance(player_abs_seg, nr_seg, level_state, is_open) or 0, 8
        local _, left_angle_diff = nl_seg and M.calculate_angular_distance(player_abs_seg, nl_seg, level_state, is_open) or 0, 8
        local right_head_start_bonus = (8 - right_angle_diff) / 16.0
        local left_head_start_bonus = (8 - left_angle_diff) / 16.0
        
        if right_exists and not left_exists then
            -- Right-only: move left if angular distance indicates close threat
            -- Lower angular distance = closer angular proximity = higher threat
            local right_threshold = base_move_threshold + right_head_start_bonus
            if nr_angular_dist < right_threshold then  -- Threat within adjusted angular proximity
                target_seg = (nr_seg - 1 + 16) % 16
            else
                target_seg = player_abs_seg
            end
        elseif left_exists and not right_exists then
            -- Left-only: move right if angular distance indicates close threat
            local left_threshold = base_move_threshold + left_head_start_bonus
            if nl_angular_dist < left_threshold then  -- Threat within adjusted angular proximity
                target_seg = (nl_seg + 1) % 16
            else
                target_seg = player_abs_seg
            end
        elseif right_exists and left_exists then
            -- Both sides: freeze (hold position) - threats on both sides
            target_seg = player_abs_seg
        else
            -- No top-rail flippers/pulsars: hold
            target_seg = player_abs_seg
        end
    end

    -- Firing policy
    -- UPDATED: Now uses angular effective distance for shooting decisions
    -- Accounts for "head start" at narrower angles by adjusting threshold dynamically
    -- Shoot if: something is in our lane OR any top-rail flipper/pulsar within angular shooting distance
    local base_shoot_dist = 0.75  -- Base angular distance threshold (corresponds to ~45° difference)
    -- Adjust threshold for head start: narrower angles (smaller diff) get higher threshold (shoot from farther)
    local head_start_bonus = (8 - min_angle_diff) / 16.0  -- Max bonus 0.5 for 0° diff
    local ANGULAR_SHOOT_DIST = base_shoot_dist + head_start_bonus
    local lane_has_threat = M.check_segment_threat(player_abs_seg, enemies_state)
    local within_angular_shooting_distance = (min_angular_effective <= ANGULAR_SHOOT_DIST)

    local should_fire
    if lane_has_threat or within_angular_shooting_distance then
        should_fire = true
    else
        -- Keep only 3 shots onscreen
        should_fire = (shot_count < 3)
    end

    -- Expert tweak: When we would fire at a top-rail flipper/pulsar but the shot buffer is full (>=8),
    -- still recommend FIRE and also MOVE AWAY by one segment opposite the nearest threat.
    -- UPDATED: Now uses angular distances for threat assessment
    -- Applies only if there is at least one top-rail flipper/pulsar detected (right_exists/left_exists)
    if should_fire and shot_count >= 8 and (right_exists or left_exists) then
        -- Determine nearest side using angular effective distances
        local move_right -- boolean: true => move to player_abs_seg+1, false => -1
        if right_exists and left_exists then
            -- Choose the side with lower angular distance (higher threat)
            move_right = (nr_angular_dist > nl_angular_dist)  -- Move away from closer threat
        elseif right_exists then
            -- Threat only on the right -> move left (away from threat)
            move_right = false
        else -- left_exists only
            -- Threat only on the left -> move right (away from threat)
            move_right = true
        end

        -- Compute candidate adjacent segment respecting open/closed topology
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
            target_seg = candidate
        end
    end

    -- Superzap heuristic retained (3+ top-rail enemies)
    local top_rail_count = 0
    for i = 1, 7 do
        local depth = enemies_state.enemy_depths[i]
        local seg = enemies_state.enemy_abs_segments[i]
        if depth > 0 and depth <= TOP_RAIL_DEPTH and seg ~= INVALID_SEGMENT then
            top_rail_count = top_rail_count + 1
        end
    end
    local superzapper_available = (player_state.superzapper_uses or 0) < 2
    local should_superzap = superzapper_available and (top_rail_count >= 3)

    return target_seg, 0, should_fire, should_superzap
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
    local reward, bDone = 0.0, false

    -- Terminal: death (edge-triggered) - Scaled to match 1 life = 1.0 reward unit
    if player_state.alive == 0 and previous_alive_state == 1 and not episode_ended then
        reward = reward - DEATH_PENALTY
        bDone = true
        episode_ended = true
    else
        -- Reset episode_ended flag when player respawns (alive goes from 0 to 1)
        if player_state.alive == 1 and previous_alive_state == 0 then
            episode_ended = false
        end
        -- Primary dense signal: scaled/clipped score delta
        local score_delta = (player_state.score or 0) - (previous_score or 0)
        if score_delta ~= 0 and score_delta < 1000 then                         -- Filter our large completion bonuses
            local r_score = score_delta / SCORE_UNIT                            -- Scaled: 20k points = 1.0 reward unit
            if r_score > 1.0 then r_score = 1.0 end
            if r_score < -1.0 then r_score = -1.0 end
            reward = reward + r_score
        end

        -- Level completion bonus (edge-triggered) - Scaled to match death penalty magnitude
        if (level_state.level_number or 0) > (previous_level or 0) then
            -- Fixed completion bonus; optional ratio-based scaling can be added later using score_at_level_start
            reward = reward + LEVEL_COMPLETION_BONUS
            -- bDone = true  -- REMOVED: Episodes now terminate only on death for longer trajectories
        end

        -- Zap cost (edge-triggered on button press)
        local zap_now = player_state.zap_detected or 0
        if zap_now == 1 and previous_zap_detected == 0 then
            reward = reward - ZAP_COST
        end

        -- === OBJECTIVE DENSE REWARD COMPONENTS ===
        -- Only apply during active gameplay (not tube zoom, high score entry, etc.)
        if game_state.gamestate == 0x04 or game_state.gamestate == 0x20 then
            local player_abs_seg = player_state.position & 0x0F
            local is_open = (level_state.level_type == 0xFF)
            
            -- Level-specific scaling factors
            local level_type = level_state.level_type
            local proximity_scale = 1.0
            local safety_scale = 1.0
            
            if level_type == 0xFF then -- Open levels
                proximity_scale = 1.2  -- Increase proximity rewards on open levels
                safety_scale = 0.8     -- Reduce safety emphasis (more space available)
            elseif level_type == 0x00 then -- Closed levels
                proximity_scale = 0.9  -- Slightly reduce proximity rewards on closed levels
                safety_scale = 1.1     -- Increase safety emphasis on closed levels
            end

            -- Compute expert target once for reward gating (avoid rewarding moves toward danger)
            local expert_target_seg_cached = -1
            do
                local ets, _, _, _ = M.find_target_segment(game_state, player_state, level_state, enemies_state, abs_to_rel_func)
                expert_target_seg_cached = ets or -1
            end
            
            -- 1. DANGER AVOIDANCE REWARD (Penalties removed per spec)
            -- No penalty for being in dangerous positions; optional safe bonus remains disabled
            
            -- 2. PROXIMITY OPTIMIZATION REWARD (Distance-based Positioning)
            -- Reward optimal distance to nearest enemy (not too close, not too far)
            -- IMPORTANT: If expert target lane is dangerous, skip proximity shaping entirely.
            do
                local danger_target_lane = (expert_target_seg_cached ~= -1) and M.is_danger_lane(expert_target_seg_cached, enemies_state)
                local nearest_distance = enemies_state.alignment_error_magnitude or 1.0
                if not danger_target_lane and enemies_state.nearest_enemy_seg ~= INVALID_SEGMENT then
                    -- Convert normalized distance (0-1) to segments for clearer logic
                    local max_dist = is_open and 15.0 or 8.0
                    local distance_segments = nearest_distance * max_dist
                    
                    -- Optimal range: 1-3 segments (allows reaction time but maintains offensive capability)
                    local optimal_min, optimal_max = 1.0, 3.0
                    local prox_reward = 0.0
                    if distance_segments >= optimal_min and distance_segments <= optimal_max then
                        prox_reward = 0.10 * proximity_scale  -- Good positioning bonus
                    else
                        prox_reward = 0.0 -- Penalties removed
                    end
                    -- Neutral reward for 3-5 segments (acceptable range)
                    reward = reward + prox_reward
                end
            end
            
            -- 3. STRATEGIC SHOT MANAGEMENT REWARD (Smart Resource Management)
            -- Requested policy:
            --   0-2 shots: penalty (too few shots)
            --   4-7 shots: reward (good management)
            --   8 shots: penalty (overuse)
            do
                local shot_count = player_state.shot_count or 0
                local shot_reward = 0.0
                if shot_count <= 2 then
                    shot_reward = 0 -- Penalty removed
                elseif shot_count >= 4 and shot_count <= 7 then
                    shot_reward = 1
                elseif shot_count >= 8 then
                    shot_reward = 0 -- Penalty removed
                end
                reward = reward + shot_reward / SCORE_UNIT
            end
            
            -- 4. THREAT RESPONSIVENESS REWARD (Reaction to Immediate Dangers)
            -- Bonus for appropriate responses to critical threats
            local critical_threats = 0
            for i = 1, 7 do
                local abs_seg = enemies_state.enemy_abs_segments[i]
                local depth = enemies_state.enemy_depths[i]
                local enemy_type = enemies_state.enemy_core_type[i]
                
                if abs_seg == player_abs_seg and depth > 0 and depth <= 0x30 then
                    critical_threats = critical_threats + 1
                end
            end
            
            -- Threat penalties removed per spec (no subtraction)
            
            -- 5. PULSAR SAFETY REWARD (Objective Hazard Assessment)
            -- Extra penalty for being in pulsing pulsar lanes (objectively lethal)
            -- Pulsar safety penalties removed per spec

            -- 6. EXPERT POSITIONING REWARD (Follow Expert System Guidance)
            -- Reward the DQN for following expert system's strategic positioning recommendations
            -- IMPORTANT: If expert target lane is dangerous, skip positioning shaping entirely.
            do
                local positioning_reward = 0.0
                if expert_target_seg_cached ~= -1 then
                    local danger_target_lane = M.is_danger_lane(expert_target_seg_cached, enemies_state)
                    if not danger_target_lane then
                        local current_player_seg = player_abs_seg
                        local prev_player_seg = previous_player_position & 0x0F
                        
                        -- Calculate distances to expert target (current and previous)
                        local current_distance = math.abs(abs_to_rel_func(current_player_seg, expert_target_seg_cached, is_open))
                        local previous_distance = math.abs(abs_to_rel_func(prev_player_seg, expert_target_seg_cached, is_open))
                        
                        -- Award for being on target segment
                        if current_distance == 0 then
                            -- Worth ~10 points when perfectly aligned (10 / SCORE_UNIT)
                            positioning_reward = 10.0 / SCORE_UNIT
                        -- Award for moving toward target
                        elseif current_distance < previous_distance then
                            local progress = previous_distance - current_distance
                            -- Scale small progress reward so it remains below score/death signals
                            positioning_reward = 0.5 * (progress / 8.0) * (10.0 / SCORE_UNIT)
                        -- Small penalty for moving away from target or not moving when needed
                        elseif current_distance > 0 then
                            -- Penalties removed for moving away or not moving
                        end
                        
                        -- Apply level-specific scaling
                        if level_type == 0xFF then -- Open levels
                            positioning_reward = positioning_reward * 1.1  -- Slightly increase positioning importance on open levels
                        elseif level_type == 0x00 then -- Closed levels
                            positioning_reward = positioning_reward * 0.9  -- Slightly reduce on closed levels
                        end
                    end
                end
                reward = reward + positioning_reward
            end

            -- 8. TOP-RAIL FLIPPER ENGAGEMENT REWARD (dense shaping using fractional rel positions)
            do
                local min_abs_rel_float = nil
                local aligned_fire_bonus = 0.0
                local hold_penalty = 0.0

                -- Find nearest top-rail flipper using authoritative fractional relative pos
                for i = 1, 7 do
                    if enemies_state.enemy_core_type[i] == ENEMY_TYPE_FLIPPER and
                       enemies_state.enemy_depths[i] > 0 and enemies_state.enemy_depths[i] <= 0x40 then
                        local relf = enemies_state.active_top_rail_enemies[i]
                        if relf ~= 0 or enemies_state.enemy_between_segments[i] == 1 then
                            local d = math.abs(relf)
                            if not min_abs_rel_float or d < min_abs_rel_float then
                                min_abs_rel_float = d
                            end
                        end
                    end
                end

                if min_abs_rel_float then
                    -- Progress toward alignment (reward getting closer)
                    if previous_toprail_min_abs_rel then
                        local delta = previous_toprail_min_abs_rel - min_abs_rel_float
                        if delta > 0 then
                            -- Small reward proportional to progress, capped and scaled under 10 pts
                            local progress_bonus = math.min(1.0, delta) * (5.0 / SCORE_UNIT)
                            reward = reward + progress_bonus
                        end
                    end

                    -- Reward firing when well aligned (tight aim window)
                    local fire_now = (player_state.fire_detected or 0)
                    local fire_edge = (fire_now == 1 and previous_fire_detected == 0)
                    if min_abs_rel_float <= 0.30 and fire_edge then
                        reward = reward + (8.0 / SCORE_UNIT) -- increased bonus (~8 pts) for well-timed shot
                    end

                    -- Penalty for not firing while very close removed per spec
                end
            end

            -- 9. FLIPPER SAFETY REWARD (Encourage stepping away when shots depleted)
            -- Penalize proximity to top-rail flippers when all shots are in use (shot_count >= 8)
            do
                local shot_count = player_state.shot_count or 0
                if shot_count >= 8 then
                    local nearest_flipper_dist = nil
                    -- Find nearest top-rail flipper distance (extended to 0x40 depth for earlier detection)
                    for i = 1, 7 do
                        if enemies_state.enemy_core_type[i] == ENEMY_TYPE_FLIPPER and
                           enemies_state.enemy_depths[i] > 0 and enemies_state.enemy_depths[i] <= 0x40 then
                            local relf = enemies_state.active_top_rail_enemies[i]
                            if relf ~= 0 or enemies_state.enemy_between_segments[i] == 1 then
                                local d = math.abs(relf)
                                if not nearest_flipper_dist or d < nearest_flipper_dist then
                                    nearest_flipper_dist = d
                                end
                            end
                        end
                    end

                    -- Apply safety reward/penalty based on proximity when shots are depleted
                    if nearest_flipper_dist then
                        local safety_signal = 0.0
                        if nearest_flipper_dist >= 3.0 then
                            -- Positive reward for maintaining safe distance (3+ segments away)
                            safety_signal = 0.002 * safety_scale
                        elseif nearest_flipper_dist >= 2.0 then
                            -- Neutral for moderate distance
                            safety_signal = 0.0
                        elseif nearest_flipper_dist >= 1.0 then
                            -- Light penalty for close proximity
                            safety_signal = -0.001 * safety_scale
                        else
                            -- Stronger penalty when very close
                            safety_signal = -0.003 * safety_scale
                        end
                        reward = reward + safety_signal
                    end
                end
            end

            -- 9.5. GENERAL FLIPPER PROXIMITY MANAGEMENT (Always active, even with shots available)
            -- Encourage safer positioning around flippers overall, not just when shots depleted
            do
                local nearest_flipper_dist = nil
                local nearest_flipper_depth = nil
                -- Find nearest top-rail flipper distance and depth
                for i = 1, 7 do
                    if enemies_state.enemy_core_type[i] == ENEMY_TYPE_FLIPPER and
                       enemies_state.enemy_depths[i] > 0 and enemies_state.enemy_depths[i] <= TOP_RAIL_DEPTH then
                        local relf = enemies_state.active_top_rail_enemies[i]
                        if relf ~= 0 or enemies_state.enemy_between_segments[i] == 1 then
                            local d = math.abs(relf)
                            if not nearest_flipper_dist or d < nearest_flipper_dist then
                                nearest_flipper_dist = d
                                nearest_flipper_depth = enemies_state.enemy_depths[i]
                            end
                        end
                    end
                end

                -- Apply general proximity management (lighter penalties than depleted-shots case)
                if nearest_flipper_dist then
                    local proximity_penalty = 0.0
                    if nearest_flipper_dist < 1.0 then
                        -- Light penalty for being in the same segment as a flipper
                        proximity_penalty = -0.0005 * safety_scale
                    elseif nearest_flipper_dist < 2.0 then
                        -- Very light penalty for adjacent segments
                        proximity_penalty = -0.0002 * safety_scale
                    end
                    -- No penalty for 2+ segments away - encourages strategic positioning
                    reward = reward + proximity_penalty
                end
            end

            -- 9.6. FLIPPER ELIMINATION BONUS (Reward successful kills)
            -- Track flipper disappearances and reward eliminations more than movement costs
            do
                -- Track previous flipper count for edge detection
                local current_flipper_count = 0
                for i = 1, 7 do
                    if enemies_state.enemy_core_type[i] == ENEMY_TYPE_FLIPPER and
                       enemies_state.enemy_depths[i] > 0 and enemies_state.enemy_depths[i] <= TOP_RAIL_DEPTH then
                        current_flipper_count = current_flipper_count + 1
                    end
                end

                -- Compare to previous count
                if previous_flipper_count and current_flipper_count < previous_flipper_count then
                    -- Flipper was eliminated - give bonus reward (~15 points worth)
                    local elimination_bonus = 15.0 / SCORE_UNIT
                    reward = reward + elimination_bonus
                end
            end

            -- 9.7. STRATEGIC RETREAT REWARD (Reward moving away from close flippers)
            -- When shots are available but flipper is very close, reward retreating
            do
                local shot_count = player_state.shot_count or 0
                if shot_count < 8 then  -- Shots available
                    local nearest_flipper_dist = nil
                    local nearest_flipper_rel = nil
                    -- Find nearest top-rail flipper
                    for i = 1, 7 do
                        if enemies_state.enemy_core_type[i] == ENEMY_TYPE_FLIPPER and
                           enemies_state.enemy_depths[i] > 0 and enemies_state.enemy_depths[i] <= TOP_RAIL_DEPTH then
                            local relf = enemies_state.active_top_rail_enemies[i]
                            if relf ~= 0 or enemies_state.enemy_between_segments[i] == 1 then
                                local d = math.abs(relf)
                                if not nearest_flipper_dist or d < nearest_flipper_dist then
                                    nearest_flipper_dist = d
                                    nearest_flipper_rel = relf
                                end
                            end
                        end
                    end

                    -- Reward retreating from very close flippers when shots available
                    if nearest_flipper_dist and nearest_flipper_dist < 1.5 then
                        local current_pos = player_abs_seg
                        local prev_pos = previous_player_position & 0x0F
                        local pos_change = abs_to_rel_func(current_pos, prev_pos, is_open)
                        
                        -- Check if we moved away from the flipper
                        -- If flipper is to the right (rel > 0), moving left (pos_change < 0) is retreat
                        -- If flipper is to the left (rel < 0), moving right (pos_change > 0) is retreat
                        if (nearest_flipper_rel > 0 and pos_change < 0) or (nearest_flipper_rel < 0 and pos_change > 0) then
                            -- Successful retreat from close flipper (~5 points)
                            local retreat_bonus = 5.0 / SCORE_UNIT
                            reward = reward + retreat_bonus
                        end
                    end
                end
            end
            -- 10. USELESS MOVEMENT PENALTY (Discourage random spinner when not needed)
            -- Apply a small penalty when spinning while already aligned (or no target)
            -- and not in immediate danger. Keeps behavior calm instead of fidgety.
            do
                local spin_delta = math.abs(tonumber(player_state.spinner_detected or 0))
                if spin_delta > 0 then
                    local player_abs_seg2 = player_state.position & 0x0F
                    local nearest_abs = enemies_state.nearest_enemy_abs_seg_internal or -1
                    local is_open2 = (level_state.level_type == 0xFF)
                    local need_move = false
                    if nearest_abs ~= -1 then
                        local rel = abs_to_rel_func(player_abs_seg2, nearest_abs, is_open2)
                        need_move = math.abs(rel) > 1 -- require movement if more than one segment off
                    end
                    local in_danger = M.is_danger_lane(player_abs_seg2, enemies_state)
                    if not need_move and not in_danger then
                        -- Scale a gentle penalty with movement magnitude, capped
                        local units = math.min(4, spin_delta)
                        local move_penalty = -0.0002 * units
                        reward = reward + move_penalty
                        -- Attribute to proximity shaping bucket for metrics
                    end
                end
            end
        end
    end

    -- State updates
    -- Detect level increment to reset per-level trackers
    local current_level = level_state.level_number or 0
    local current_score = player_state.score or 0
    if current_level > (previous_level or 0) then
        score_at_level_start = current_score
    end
    previous_score = current_score
    previous_level = current_level
    previous_alive_state = player_state.alive or 0
    previous_zap_detected = player_state.zap_detected or 0
    previous_fire_detected = player_state.fire_detected or 0
    previous_player_position = player_state.position or 0
    -- Update top-rail tracking
    do
        local min_abs_rel_float = nil
        for i = 1, 7 do
            if enemies_state.enemy_core_type[i] == ENEMY_TYPE_FLIPPER and
               enemies_state.enemy_depths[i] > 0 and enemies_state.enemy_depths[i] <= 0x40 then
                local relf = enemies_state.active_top_rail_enemies[i]
                if relf ~= 0 or enemies_state.enemy_between_segments[i] == 1 then
                    local d = math.abs(relf)
                    if not min_abs_rel_float or d < min_abs_rel_float then
                        min_abs_rel_float = d
                    end
                end
            end
        end
        previous_toprail_min_abs_rel = min_abs_rel_float
    end

    -- Update flipper count tracking
    do
        local current_flipper_count = 0
        for i = 1, 7 do
            if enemies_state.enemy_core_type[i] == ENEMY_TYPE_FLIPPER and
               enemies_state.enemy_depths[i] > 0 and enemies_state.enemy_depths[i] <= TOP_RAIL_DEPTH then
                current_flipper_count = current_flipper_count + 1
            end
        end
        previous_flipper_count = current_flipper_count
    end
    LastRewardState = reward

    return reward, bDone
end

-- Function to retrieve the last calculated reward (for display)
function M.getLastReward()
    return LastRewardState
end


return M 