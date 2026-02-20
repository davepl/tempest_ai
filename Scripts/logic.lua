local state_defs = require("state") -- Assuming state.lua is in the same dir

-- Define enemy type constants (copy from main.lua)
local ENEMY_TYPE_FLIPPER = 0
local ENEMY_TYPE_PULSAR = 1
local ENEMY_TYPE_TANKER = 2
local ENEMY_TYPE_SPIKER = 3
local ENEMY_TYPE_FUSEBALL = 4

-- Define constants (copy from main.lua)
local INVALID_SEGMENT = state_defs.INVALID_SEGMENT
local TOP_RAIL_ABSENT = state_defs.TOP_RAIL_ABSENT or 255

-- New constants for top rail logic
local TOP_RAIL_DEPTH = 0x15
local TOP_RAIL_AVOID_DEPTH = 0x60
local SAFE_DISTANCE = 2
local FLIPPER_WAIT_DISTANCE = 5 -- segments within which we prefer to wait and conserve shots on top rail
local FLIPPER_REACT_DISTANCE_R = 2.0 -- distance at which we move one segment and fire (right-side, float)
local FLIPPER_REACT_DISTANCE_L = 2.0 -- distance at which we move one segment and fire (left-side, float)
local FREEZE_FIRE_PRIO_LOW = 2
local FREEZE_FIRE_PRIO_HIGH = 8
local AVOID_FIRE_PRIORITY = 3
local PULSAR_THRESHOLD = 0xE0 -- DEPRECATED: see assembly-correct pulsing checks below
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
-- Expert fire control: per-frame Bernoulli probability for emitting fire.
local EXPERT_FIRE_PROBABILITY = 0.95

local M = {} -- Module table

-- Global variables needed by calculate_reward (scoped within this module)
-- Reward shaping parameters (tunable)

local DEATH_PENALTY = 1000           -- Edge-triggered penalty when dying (same clip as normal rewards)
local DANGER_DEPTH = 0x80            -- Depth threshold for nearby threats/safety shaping
local SAFE_LANE_REWARD = 2.0         -- Base reward when a lane is clear of nearby threats
local DANGER_LANE_PENALTY = 2.0      -- Base penalty when a lane contains nearby threats
local ZAP_CONSERVATION_BONUS = 0.3   -- Per-frame bonus per remaining superzapper charge

local previous_score = 0
local previous_level = 0
local previous_alive_state = 1 -- Track previous alive state, initialize as alive
local LastRewardState = 0
local LastSubjRewardState = 0
local LastObjRewardState = 0
local previous_player_position = 0

local function sample_expert_fire()
    return math.random() < EXPERT_FIRE_PROBABILITY
end

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
    -- Use adjusted spike heights (0 = no spike, low = shorter/safer, high = longer/more dangerous).
    -- Pick the globally best *near* lane:
    -- 1) Any spike-free lane (height 0), nearest first
    -- 2) Otherwise lowest adjusted spike height, nearest as tie-breaker
    local best_seg = player_abs_seg
    local best_spike_h = level_state.spike_heights[player_abs_seg] or 0
    local best_dist = 0

    for seg = 0, 15 do
        local spike_h = level_state.spike_heights[seg] or 0
        local rel = M.absolute_to_relative_segment(player_abs_seg, seg, is_open)
        local dist = math.abs(rel)

        local better = false
        if spike_h == 0 then
            if best_spike_h ~= 0 then
                better = true
            elseif dist < best_dist then
                better = true
            end
        elseif best_spike_h ~= 0 then
            if spike_h < best_spike_h then
                better = true
            elseif spike_h == best_spike_h and dist < best_dist then
                better = true
            end
        end

        if better then
            best_seg = seg
            best_spike_h = spike_h
            best_dist = dist
        end
    end

    return best_seg, 0, true, false
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
    -- Assembly: pulsing bit 7 CLEAR ($01-$7F) = dangerous, bit 7 SET or 0 = safe
    if enemies_state.pulsing == 0 or enemies_state.pulsing >= 0x80 then return false, player_abs_seg, 0, false, false end

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
        if enemies_state.enemy_shot_abs_segments[i] == segment and enemies_state.shot_positions[i] > 0 and enemies_state.shot_positions[i] <= TOP_RAIL_AVOID_DEPTH then return true end
    end
    for i = 1, 7 do -- Check enemies
        if enemies_state.enemy_abs_segments[i] == segment and enemies_state.enemy_depths[i] > 0 and enemies_state.enemy_depths[i] <= TOP_RAIL_AVOID_DEPTH then return true end
    end
    for i = 1, 7 do -- Check pulsars (regardless of pulsing state for general threat)
        if enemies_state.enemy_core_type[i] == ENEMY_TYPE_PULSAR and enemies_state.enemy_abs_segments[i] == segment and enemies_state.enemy_depths[i] > 0 then return true end
    end
    return false
end

-- Returns true if the segment is a danger lane (more specific criteria for immediate action)
function M.is_danger_lane(segment, enemies_state)
    -- Assembly: pulsing bit 7 CLEAR ($01-$7F) = dangerous
    if enemies_state.pulsing > 0 and enemies_state.pulsing < 0x80 then -- Check dangerous pulsars first
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
    for i = 1, 4 do -- Check close enemy shots (depth <= TOP_RAIL_AVOID_DEPTH)
        if enemies_state.enemy_shot_abs_segments[i] == segment and enemies_state.shot_positions[i] > 0 and enemies_state.shot_positions[i] <= TOP_RAIL_AVOID_DEPTH then return true end
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

function M.set_expert_fire_probability(probability)
    if type(probability) ~= "number" then
        return
    end
    if probability < 0.0 then
        probability = 0.0
    elseif probability > 1.0 then
        probability = 1.0
    end
    EXPERT_FIRE_PROBABILITY = probability
end

function M.get_expert_fire_probability()
    return EXPERT_FIRE_PROBABILITY
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
    local is_open = (level_state.level_type ~= 0x00) -- Assembly: $00=closed, $FF=open
    local player_abs_seg = math.floor(player_state.position) % 16
    local shot_count = player_state.shot_count or 0
    local min_abs_rel_float = nil

    -- During tube zoom, spike avoidance owns steering.
    if game_state.gamestate == 0x20 then
        local seg = M.zoom_down_tube(player_abs_seg, level_state, is_open)
        return seg, 0, sample_expert_fire(), false
    end

    local function try_offset(base_seg, delta)
        if is_open then
            local candidate = base_seg + delta
            if candidate >= 0 and candidate <= 15 then
                return candidate
            end
            return nil
        else
            return (base_seg + delta + 16) % 16
        end
    end

    local function scan_top_threats()
        local flee_rel = nil
        local flee_abs = nil
        local shot_threat = false

        local function consider_top(rel_value)
            if rel_value == nil then
                return
            end
            local abs_rel = math.abs(rel_value)
            if min_abs_rel_float == nil or abs_rel < min_abs_rel_float then
                min_abs_rel_float = abs_rel
            end
            if abs_rel <= SAFE_DISTANCE and (flee_abs == nil or abs_rel < flee_abs) then
                flee_abs = abs_rel
                flee_rel = rel_value
            end
        end

        local function consider_shot(rel_value)
            if rel_value == nil or rel_value == INVALID_SEGMENT then
                return
            end
            local abs_rel = math.abs(rel_value)
            if min_abs_rel_float == nil or abs_rel < min_abs_rel_float then
                min_abs_rel_float = abs_rel
            end
            if abs_rel <= SAFE_DISTANCE then
                shot_threat = true
                if flee_abs == nil or abs_rel < flee_abs then
                    flee_abs = abs_rel
                    flee_rel = rel_value
                end
            end
        end

        for i = 1, 7 do
            local depth = enemies_state.enemy_depths[i]
            if depth > 0 and depth <= TOP_RAIL_AVOID_DEPTH then
                local t = enemies_state.enemy_core_type[i]
                if t == ENEMY_TYPE_FLIPPER or t == ENEMY_TYPE_PULSAR then
                    local seg = enemies_state.enemy_abs_segments[i]
                    if seg ~= INVALID_SEGMENT then
                        local rel_int = abs_to_rel_func(player_abs_seg, seg, is_open)
                        local rel_float = enemies_state.active_top_rail_enemies[i]
                        if rel_float == nil or rel_float == TOP_RAIL_ABSENT then
                            rel_float = rel_int
                        end
                        consider_top(rel_float)
                    end
                end
            end
        end

        for i = 1, 4 do
            local shot_depth = enemies_state.shot_positions[i]
            if shot_depth and shot_depth > 0 and shot_depth <= TOP_RAIL_AVOID_DEPTH then
                consider_shot(enemies_state.enemy_shot_segments[i])
            end
        end

        return flee_rel, shot_threat
    end

    local flee_rel, shot_threat_near = scan_top_threats()
    if flee_rel ~= nil then
        local flee_dir = (flee_rel >= 0) and -1 or 1
        local flee_target = try_offset(player_abs_seg, flee_dir)
        if flee_target == nil then
            flee_target = try_offset(player_abs_seg, -flee_dir) or player_abs_seg
        end
        return flee_target, 0, sample_expert_fire(), false
    end

    if game_state.gamestate ~= 0x04 then
        return player_abs_seg, 0, sample_expert_fire(), false
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
                return candidate, 0, sample_expert_fire(), false
            end
        end
    end

    -- Default to hunting the next enemy when no imminent top threat is present
    local target_seg = player_abs_seg
    do
        local hunt_target_seg, _, _ = M.hunt_enemies(enemies_state, player_abs_seg, is_open, abs_to_rel_func, {})
        if hunt_target_seg ~= -1 then
            target_seg = hunt_target_seg
        end
    end

    local shots_remaining = math.max(0, 8 - shot_count)
    -- Keep lane-hunting active even when ammo is low.
    -- Previously we froze target_seg to the current lane at low ammo, which
    -- effectively disabled spinner movement unless a top-rail flee case fired.
    if shot_threat_near then
        target_seg = player_abs_seg
    end

    -- Superzap heuristic retained (3+ top-rail enemies)
    local top_rail_count = 0
    for i = 1, 7 do
        local depth = enemies_state.enemy_depths[i]
        local seg = enemies_state.enemy_abs_segments[i]
        if depth > 0 and depth <= TOP_RAIL_AVOID_DEPTH and seg ~= INVALID_SEGMENT then
            top_rail_count = top_rail_count + 1
        end
    end
    local superzapper_available = (player_state.superzapper_uses or 0) < 2
    local should_superzap = superzapper_available and (top_rail_count >= 3)

    return target_seg, 0, sample_expert_fire(), should_superzap
end

-- Function to calculate desired spinner direction and distance to target enemy
function M.direction_to_nearest_enemy(game_state, level_state, player_state, enemies_state, abs_to_rel_func)
    local player_abs_seg = math.floor(player_state.position) % 16
    local is_open = (level_state.level_type ~= 0x00) -- Assembly: $00=closed, $FF=open
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
    local reward, subj_reward, obj_reward, bDone = 0.0, 0.0, 0.0, false

    -- Terminal: death (edge-triggered) - Penalty applied to objective reward
    if player_state.alive == 0 and previous_alive_state == 1 then
        obj_reward = obj_reward - DEATH_PENALTY
        bDone = true
    else
        -- Primary dense signal: scaled/clipped score delta
        local score_delta = (player_state.score or 0) - (previous_score or 0)
        if score_delta > 0 and score_delta < 1000 then                         -- Filter out large bonuses AND negative deltas
            local r_score = score_delta 
            obj_reward = obj_reward + r_score
        end

        -- Subjective shaping: lane safety/danger reward
        -- Only apply during active gameplay (not tube zoom, high score entry, etc.)
        if game_state.gamestate == 0x04 or game_state.gamestate == 0x20 then
            -- Superzapper conservation: small bonus for each zap held in reserve
            local zap_remaining = 2 - (player_state.superzapper_uses or 0)
            if zap_remaining > 0 then
                subj_reward = subj_reward + (zap_remaining * ZAP_CONSERVATION_BONUS)
            end

            local player_abs_seg = player_state.position & 0x0F
            local prev_player_seg = math.floor(previous_player_position or 0) % 16
            local is_open = (level_state.level_type ~= 0x00) -- Assembly: $00=closed, $FF=open

            -- Apply safety shaping EVERY frame so the agent feels ongoing danger,
            -- not just on lane changes.  Scale slightly smaller when stationary
            -- to keep a lane-change incentive.
            do
                local shaping_scale = (player_abs_seg ~= prev_player_seg) and 1.0 or 0.5
                -- First pass: map threats (enemies or shots) within DANGER_DEPTH by lane
                local lane_threat = {}
                local function mark_threat(seg)
                    if seg == nil or seg == INVALID_SEGMENT then return end
                    if seg < 0 or seg > 15 then return end
                    lane_threat[seg] = true
                end

                for i = 1, 7 do
                    local depth = enemies_state.enemy_depths[i]
                    local seg = enemies_state.enemy_abs_segments[i]
                    if depth and depth > 0 and depth <= DANGER_DEPTH then
                        mark_threat(seg)
                    end
                end

                for i = 1, 4 do
                    local shot_depth = enemies_state.shot_positions[i]
                    local seg = enemies_state.enemy_shot_abs_segments[i]
                    if shot_depth and shot_depth > 0 and shot_depth <= DANGER_DEPTH then
                        mark_threat(seg)
                    end
                end

                local offsets = {0, 1, -1, 2, -2}
                local seen = {}
                local function offset_to_lane(offset)
                    if offset == 0 then
                        return player_abs_seg
                    end
                    if is_open then
                        local candidate = player_abs_seg + offset
                        if candidate < 0 or candidate > 15 then
                            return nil -- Off the board on open levels
                        end
                        return candidate
                    else
                        return (player_abs_seg + offset + 16) % 16
                    end
                end

                for _, offset in ipairs(offsets) do
                    local lane = offset_to_lane(offset)
                    if lane ~= nil and not seen[lane] then
                        seen[lane] = true

                        local distance = math.abs(offset)
                        local weight = 1.0
                        if distance == 1 then
                            weight = 0.5
                        elseif distance == 2 then
                            weight = 0.25
                        end

                        if lane_threat[lane] then
                            subj_reward = subj_reward - (DANGER_LANE_PENALTY * weight * shaping_scale)
                        else
                            subj_reward = subj_reward + (SAFE_LANE_REWARD * weight * shaping_scale)
                        end
                    end
                end
            end
        end
    end
    
    -- State updates
    previous_score = player_state.score or 0
    previous_level = level_state.level_number or 0
    previous_alive_state = player_state.alive or 0
    previous_player_position = player_state.position or 0

    -- Calculate total reward as sum of subjective and objective components
    reward = subj_reward + obj_reward

    LastRewardState = reward
    LastSubjRewardState = subj_reward
    LastObjRewardState = obj_reward

    return reward, subj_reward, obj_reward, bDone
end

-- Function to retrieve the last calculated reward (for display)
function M.getLastReward()
    return LastRewardState
end

-- Function to retrieve the last calculated reward components
function M.getLastRewardComponents()
    return LastSubjRewardState, LastObjRewardState
end


return M 
 