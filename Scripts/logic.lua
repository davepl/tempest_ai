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
local FREEZE_FIRE_PRIO_LOW = 2
local FREEZE_FIRE_PRIO_HIGH = 8
local AVOID_FIRE_PRIORITY = 3
local PULSAR_THRESHOLD = 0x00 -- Pulsing threshold for avoidance

local M = {} -- Module table

-- Global variables needed by calculate_reward (scoped within this module)
local previous_score = 0
local previous_level = 0
local previous_alive_state = 1 -- Track previous alive state, initialize as alive
local LastRewardState = 0

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
    current_abs_segment = current_abs_segment & 0x0F
    target_abs_segment = target_abs_segment & 0x0F

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
    if enemies_state.pulsing <= 0xE0 then return false, player_abs_seg, 0, false, false end

    local is_in_pulsar_lane = false
    for i = 1, 7 do
        if enemies_state.enemy_core_type[i] == ENEMY_TYPE_PULSAR and
           enemies_state.enemy_abs_segments[i] == player_abs_seg and
           enemies_state.enemy_depths[i] > 0 then
            is_in_pulsar_lane = true; break
        end
    end
    if not is_in_pulsar_lane then return false, player_abs_seg, 0, false, false end

    local nearest_safe, min_dist = -1, 255
    for target_seg = 0, 15 do
        local is_safe = true
        for i = 1, 7 do
            if enemies_state.enemy_core_type[i] == ENEMY_TYPE_PULSAR and
               enemies_state.enemy_abs_segments[i] == target_seg and
               enemies_state.enemy_depths[i] > 0 then
                is_safe = false; break
            end
        end
        if is_safe then
            local abs_dist = math.abs(abs_to_rel_func(player_abs_seg, target_seg, is_open))
            if abs_dist < min_dist then min_dist = abs_dist; nearest_safe = target_seg end
        end
    end

    if nearest_safe ~= -1 then return true, nearest_safe, 0, false, false
    else return true, player_abs_seg, 0, false, false end -- Stay put if no safe lane
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
    if enemies_state.pulsing > 0xE0 then -- Check dangerous pulsars first
        for i = 1, 7 do
            if enemies_state.enemy_core_type[i] == ENEMY_TYPE_PULSAR and enemies_state.enemy_abs_segments[i] == segment and enemies_state.enemy_depths[i] > 0 then return true end
        end
    end
    for i = 1, 7 do -- Check close enemies (depth <= 0x20)
        if enemies_state.enemy_abs_segments[i] == segment and enemies_state.enemy_depths[i] > 0 and enemies_state.enemy_depths[i] <= 0x20 then return true end
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
    local player_abs_seg = player_state.position & 0x0F
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
        local enemy_left_exists, enemy_right_exists = false, false
        local any_enemy_proximate = false
        local top_rail_enemies = {}
        local nearest_flipper_seg = nil -- Specifically track nearest flipper
        local nearest_flipper_rel = nil
        local min_flipper_abs_rel = 100

        for i = 1, 7 do
            local depth = enemies_state.enemy_depths[i]
            if depth > 0 and depth <= TOP_RAIL_DEPTH then
                local seg = enemies_state.enemy_abs_segments[i]
                if seg ~= INVALID_SEGMENT then
                    local rel = abs_to_rel_func(player_abs_seg, seg, is_open)
                    local abs_rel = math.abs(rel)
                    local core_type = enemies_state.enemy_core_type[i]

                    if abs_rel <= 1 then any_enemy_proximate = true end
                    table.insert(top_rail_enemies, {seg = seg, rel = rel, abs_rel = abs_rel, type = core_type})

                    -- Update nearest overall left/right
                    if rel > 0 and abs_rel < nr_dist then nr_dist, nr_seg, enemy_right_exists = abs_rel, seg, true end
                    if rel < 0 and abs_rel < nl_dist then nl_dist, nl_seg, enemy_left_exists = abs_rel, seg, true end

                    -- Update nearest flipper specifically
                    if core_type == ENEMY_TYPE_FLIPPER then
                        if abs_rel < min_flipper_abs_rel then
                             min_flipper_abs_rel = abs_rel
                             nearest_flipper_seg = seg
                             nearest_flipper_rel = rel
                        end
                    end
                end
            end
        end

        -- === Step 1A: High Priority Flipper Avoidance ===
        if nearest_flipper_seg and min_flipper_abs_rel < SAFE_DISTANCE then

            -- Calculate target D segments away from the flipper, in the direction opposite the player
            local direction_away_from_player = (nearest_flipper_rel == 0) and 1 or (nearest_flipper_rel > 0 and -1 or 1)
            local target_seg = (nearest_flipper_seg + direction_away_from_player * SAFE_DISTANCE + 16) % 16

            -- Safety Check for the calculated avoidance target
            if M.is_danger_lane(target_seg, enemies_state) then
                 target_seg = find_nearest_constrained_safe_segment(target_seg, enemies_state, is_open, nearest_flipper_seg, abs_to_rel_func)
            end
            local fire_priority = AVOID_FIRE_PRIORITY -- Use specific avoid priority
            local should_fire = fire_priority > shot_count
            return target_seg, 0, should_fire, false -- Return early
        end

        -- === Step 1B: Determine proposed target based on Top Rail (Presence) / Hunt ===
        local base_fire_priority = any_enemy_proximate and FREEZE_FIRE_PRIO_HIGH or FREEZE_FIRE_PRIO_LOW
        proposed_fire_prio = base_fire_priority -- Start with base priority (might be overridden by safety checks below)

        if enemy_left_exists and enemy_right_exists then
            -- Case 1: Both Sides - Find nearest threat, target inside or midpoint fallback
            local nearest_threat_seg, nearest_threat_rel, min_threat_abs_rel = nil, nil, 100
            for _, enemy in ipairs(top_rail_enemies) do
                if enemy.type == ENEMY_TYPE_FLIPPER or enemy.type == ENEMY_TYPE_PULSAR then
                    if enemy.abs_rel < min_threat_abs_rel then min_threat_abs_rel, nearest_threat_seg, nearest_threat_rel = enemy.abs_rel, enemy.seg, enemy.rel end
                end
            end
            if nearest_threat_seg then
                if nearest_threat_rel > 0 then proposed_target_seg = (nearest_threat_seg - SAFE_DISTANCE + 16) % 16
                elseif nearest_threat_rel < 0 then proposed_target_seg = (nearest_threat_seg + SAFE_DISTANCE + 16) % 16
                else proposed_target_seg = player_abs_seg end
            else -- Fallback: No Flippers/Pulsars, use midpoint
                 if is_open then local mid = math.floor((nl_seg + nr_seg) / 2); proposed_target_seg = any_enemy_proximate and mid or math.min(15, mid + 1)
                 else local dist = (nr_seg - nl_seg + 16) % 16; proposed_target_seg = (nl_seg + math.floor(dist / 2) + 16) % 16 end
            end
        elseif enemy_right_exists then
            -- Case 2: Right Only
            if is_open then proposed_target_seg = (nr_dist <= 1) and 0 or 1
            else if nr_dist < SAFE_DISTANCE then proposed_target_seg = (nr_seg - SAFE_DISTANCE + 16) % 16 else proposed_target_seg = player_abs_seg end end
        elseif enemy_left_exists then
             -- Case 3: Left Only
            if is_open then proposed_target_seg = (nl_dist <= 1) and 14 or 13
            else if nl_dist < SAFE_DISTANCE then proposed_target_seg = (nl_seg + SAFE_DISTANCE) % 16 else proposed_target_seg = player_abs_seg end end
        else
            -- Case 4: No Top Rail -> Standard Hunt Logic
            if M.is_danger_lane(player_abs_seg, enemies_state) then
                 proposed_target_seg = find_nearest_safe_segment(player_abs_seg, enemies_state, is_open)
                 proposed_fire_prio = AVOID_FIRE_PRIORITY -- Lower priority when escaping
            elseif M.get_enemy_priority(player_abs_seg, enemies_state) then
                 proposed_target_seg = player_abs_seg
                 proposed_fire_prio = 6
            else -- Search outwards
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
            end
        end

        -- === Step 2: Apply Safety Overrides (including Pulsar) ===
        local final_target_seg = proposed_target_seg
        local final_fire_priority = proposed_fire_prio
        local pulsar_override_active = false

        -- ** Pulsar Check (Highest Priority) **
        if enemies_state.pulsing > PULSAR_THRESHOLD then
            if is_pulsar_lane(player_abs_seg, enemies_state) then -- Currently ON a pulsar lane
                local safe_target = find_nearest_safe_segment(player_abs_seg, enemies_state, is_open)
                final_target_seg = safe_target
                final_fire_priority = AVOID_FIRE_PRIORITY
                pulsar_override_active = true
            elseif final_target_seg ~= player_abs_seg then -- Moving, check path
                local original_proposed_target = final_target_seg -- Store original proposed target before path check
                local relative_dist = abs_to_rel_func(player_abs_seg, original_proposed_target, is_open)
                local dir = (relative_dist > 0) and 1 or -1
                local steps = math.abs(relative_dist)
                for d = 1, steps do
                    local check_seg = (player_abs_seg + dir * d + 16) % 16
                    if is_pulsar_lane(check_seg, enemies_state) then
                        local safe_stop_seg = (player_abs_seg + dir * (d - 1) + 16) % 16
                        final_target_seg = safe_stop_seg
                        final_fire_priority = AVOID_FIRE_PRIORITY
                        pulsar_override_active = true
                        break
                    end
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
        final_should_fire = final_fire_priority > shot_count
        return final_target_seg, 0, final_should_fire, false

    else -- Other game states
        return player_abs_seg, 0, false, false
    end
end

-- Function to calculate desired spinner direction and distance to target enemy
function M.direction_to_nearest_enemy(game_state, level_state, player_state, enemies_state, abs_to_rel_func)
    local player_abs_seg = player_state.position & 0x0F
    local is_open = (level_state.level_number - 1) % 4 == 2
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

        -- Player is alive; check to see if they are on a pulsar segment
        local player_segment = player_state.position & 0x0F
        if is_pulsar_lane(player_segment, enemies_state) and enemies_state.pulsing > PULSAR_THRESHOLD then
            reward = reward - 100 -- Deduct 100 reward if on a pulsar segment
        end

        -- If this is a fusebal charging lane, reward large spinner movement to escape

        if enemies_state.fuseball_threat_nearby then
            local fuseball_escape_target = enemies_state.fuseball_escape_target
            if fuseball_escape_target ~= -1 then
                local rel_dist = abs_to_rel_func(player_segment, fuseball_escape_target, level_state.level_type == 0xFF)
                if math.abs(rel_dist) <= 2 then reward = reward + 100 end -- Reward for escaping fuseball threat
            end
        end

        


        local score_delta = player_state.score - previous_score
        if score_delta > 0 and score_delta <= 1000 then reward = reward + score_delta end

        local sc = player_state.shot_count
        if sc == 0 or sc >= 8 then reward = reward - 50
        elseif sc == 4 then reward = reward + 5; elseif sc == 5 then reward = reward + 10
        elseif sc == 6 then reward = reward + 15; elseif sc == 7 then reward = reward + 20 end

        if game_state.gamestate == 0x04 and player_state.superzapper_active ~= 0 then reward = reward - 500 end

        local target_abs_segment = enemies_state.nearest_enemy_abs_seg_internal
        local target_depth = enemies_state.nearest_enemy_depth_raw
        local player_segment = player_state.position & 0x0F

        if game_state.gamestate == 0x20 then -- Tube Zoom Reward
            local spike_h = level_state.spike_heights[player_segment] or 0
            if spike_h > 0 then reward = reward + math.max(0, ((255 - spike_h) / 2) - 27.5)
            else reward = reward + (detected_spinner == 0 and 250 or -50) end
            if player_state.fire_commanded == 1 then reward = reward + 200 end

        elseif target_abs_segment < 0 then -- No Enemies Reward
            reward = reward + (detected_spinner == 0 and 150 or -20)
            if player_state.fire_commanded == 1 then reward = reward - 100 end

        else -- Enemies Present Reward
            local desired_spinner, segment_distance, _ = M.direction_to_nearest_enemy(game_state, level_state, player_state, enemies_state, abs_to_rel_func)
            if segment_distance == 0 then -- Aligned Reward
                reward = reward + (detected_spinner == 0 and 250 or -50)
                if player_state.fire_commanded == 1 then reward = reward + 50 end
            else -- Misaligned Reward
                if segment_distance < 2 and target_depth <= 0x20 then reward = reward + (player_state.fire_commanded == 1 and 150 or -50)
                else if player_state.fire_commanded == 1 then reward = reward - (segment_distance < 2 and 20 or 30) end end
                -- Movement reward
                if desired_spinner * detected_spinner > 0 then reward = reward + 40
                elseif desired_spinner * detected_spinner < 0 then reward = reward - 50
                elseif detected_spinner == 0 and desired_spinner ~= 0 then reward = reward - 15 end
            end
        end
    else -- Player Died Penalty
        if previous_alive_state == 1 then reward = reward - 20000; bDone = true end
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