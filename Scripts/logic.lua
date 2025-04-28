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
local TOP_RAIL_DEPTH = 0x30
local SAFE_DISTANCE = 2
local FREEZE_FIRE_PRIO_LOW = 4
local FREEZE_FIRE_PRIO_HIGH = 8
local AVOID_FIRE_PRIORITY = 4

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
           enemies_state.enemy_depths[i] <= 0x10 and
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

-- Function to find the target segment and recommended action (expert policy)
function M.find_target_segment(game_state, player_state, level_state, enemies_state, abs_to_rel_func, is_open)
    local player_abs_seg = player_state.position & 0x0F
    local shot_count = player_state.shot_count or 0
    local best_target = player_abs_seg -- Default target
    local should_fire = false -- Default fire state
    local fire_priority = 4 -- Default fire priority

    if game_state.gamestate == 0x20 then -- Tube Zoom
        return M.zoom_down_tube(player_abs_seg, level_state, is_open)

    elseif game_state.gamestate == 0x04 then -- Normal Gameplay
        -- --- Top Rail Logic --- Priority 1
        local top_rail_enemies = {}
        local top_rail_count = 0
        for i = 1, 7 do
            local depth = enemies_state.enemy_depths[i]
            if depth > 0 and depth <= TOP_RAIL_DEPTH then
                local seg = enemies_state.enemy_abs_segments[i]
                if seg ~= INVALID_SEGMENT then
                    top_rail_count = top_rail_count + 1
                    table.insert(top_rail_enemies, {seg = seg, rel = abs_to_rel_func(player_abs_seg, seg, is_open)})
                end
            end
        end

        if top_rail_count >= 2 then -- FREEZE behaviour
            best_target = find_nearest_safe_segment(player_abs_seg, enemies_state, is_open)
            local enemy_close = false
            for _, enemy in ipairs(top_rail_enemies) do
                if math.abs(enemy.rel) <= 1 then
                    enemy_close = true
                    break
                end
            end
            fire_priority = enemy_close and FREEZE_FIRE_PRIO_HIGH or FREEZE_FIRE_PRIO_LOW
            should_fire = fire_priority > shot_count
            return best_target, 0, should_fire, false -- Return early

        elseif top_rail_count == 1 then -- AVOID/CONSTRAINED HUNT behaviour
            local tr_enemy = top_rail_enemies[1]
            local rel_dist_tr = tr_enemy.rel

            if math.abs(rel_dist_tr) < SAFE_DISTANCE then -- Too close, must avoid
                -- Calculate target segment that is SAFE_DISTANCE away
                local move_dir = (rel_dist_tr == 0) and 1 or (rel_dist_tr > 0 and -1 or 1) -- Move away from enemy
                local target_dist = SAFE_DISTANCE
                best_target = (player_abs_seg + move_dir * target_dist + 16) % 16

                -- Check if target is safe, if not, try other direction or stay put
                if M.is_danger_lane(best_target, enemies_state) then
                    best_target = (player_abs_seg - move_dir * target_dist + 16) % 16 -- Try other side
                    if M.is_danger_lane(best_target, enemies_state) then
                         best_target = find_nearest_safe_segment(player_abs_seg, enemies_state, is_open) -- Fallback to nearest safe
                    end
                end
                fire_priority = AVOID_FIRE_PRIORITY
                should_fire = fire_priority > shot_count
                return best_target, 0, should_fire, false -- Return early
            else
                -- Safe distance maintained, proceed to normal hunt BUT check target constraint
                -- Fall through to the standard logic below, but remember tr_enemy.seg
            end
        -- elseif top_rail_count == 0 then
            -- No top rail enemies, proceed to standard logic
            -- Fall through to the standard logic below
        end

        -- --- Standard Gameplay Logic (Danger Check, Hunt, Search) --- Executed if top_rail_count == 0 OR (count == 1 AND distance >= SAFE_DISTANCE)
        local constraint_tr_seg = (top_rail_count == 1) and top_rail_enemies[1].seg or nil

        -- 1. Check for Immediate Danger in Current Lane
        if M.is_danger_lane(player_abs_seg, enemies_state) then
            fire_priority = 8 -- Max priority when escaping
            local best_safe_left, best_safe_right, min_dist_l, min_dist_r = -1, -1, 16, 16
            for d = 1, 8 do
                 local search_l, search_r = (d <= min_dist_l), (d <= min_dist_r)
                 if not (search_l or search_r) then break end
                 if search_l then
                     local left_seg = (player_abs_seg - d + 16) % 16
                     if not M.is_danger_lane(left_seg, enemies_state) and d < min_dist_l then best_safe_left = left_seg; min_dist_l = d end
                 end
                 if search_r then
                     local right_seg = (player_abs_seg + d + 16) % 16
                     if not M.is_danger_lane(right_seg, enemies_state) and d < min_dist_r then best_safe_right = right_seg; min_dist_r = d end
                 end
            end
            if min_dist_l <= min_dist_r and best_safe_left ~= -1 then best_target = best_safe_left
            elseif best_safe_right ~= -1 then best_target = best_safe_right
            else best_target = player_abs_seg end -- Stay put if no safe lane

        -- 2. Check for Enemy in Current Lane (if not danger)
        elseif M.get_enemy_priority(player_abs_seg, enemies_state) then
            fire_priority = 6 -- High priority to shoot enemy in current lane
            best_target = player_abs_seg -- Stay and shoot

        -- 3. Search Outwards for Targets or Safe Lanes
        else
            local fallback_safe, best_enemy_prio, found_target = nil, 100, false
            for sign = -1, 1, 2 do
                for d = 1, 8 do -- Search radius
                    local seg = (player_abs_seg + sign * d + 16) % 16
                    if M.is_danger_lane(seg, enemies_state) then break end -- Stop searching this direction

                    local type, prio = M.get_enemy_priority(seg, enemies_state)
                    if type and prio < best_enemy_prio then -- Found a better enemy target
                        best_target, best_enemy_prio = seg, prio
                        fire_priority = 6 -- Target enemy priority
                        found_target = true
                    elseif not type and not found_target and fallback_safe == nil then -- Found a potential safe fallback
                        fallback_safe = seg
                    end
                end
            end
            if not found_target and fallback_safe then -- No enemy found, use fallback
                best_target = fallback_safe
                fire_priority = 4 -- Default priority for fallback
            elseif not found_target then -- No enemy, no fallback nearby
                 best_target = player_abs_seg -- Stay put
                 fire_priority = 4
            end
             -- If found_target is true, best_target and fire_priority=6 are already set
        end

        -- Apply Top Rail Constraint (if count == 1)
        if constraint_tr_seg then
            local rel_dist_target_tr = abs_to_rel_func(best_target, constraint_tr_seg, is_open)
            if math.abs(rel_dist_target_tr) < SAFE_DISTANCE then
                -- Target violates constraint, override!
                -- Try staying put first if it's safe and respects distance
                if not M.is_danger_lane(player_abs_seg, enemies_state) and math.abs(abs_to_rel_func(player_abs_seg, constraint_tr_seg, is_open)) >= SAFE_DISTANCE then
                     best_target = player_abs_seg
                else
                    -- Find nearest safe segment that respects distance
                    local safe_override = -1
                    local min_safe_dist = 16
                    for d = 0, 8 do -- Check current seg then expand outwards
                        local found_safe = false
                        local seg_options = (d == 0) and {player_abs_seg} or {(player_abs_seg - d + 16) % 16, (player_abs_seg + d + 16) % 16}
                        for _, check_seg in ipairs(seg_options) do
                            if not M.is_danger_lane(check_seg, enemies_state) then
                                local dist_to_tr = math.abs(abs_to_rel_func(check_seg, constraint_tr_seg, is_open))
                                if dist_to_tr >= SAFE_DISTANCE then
                                    safe_override = check_seg
                                    found_safe = true
                                    break
                                end
                            end
                        end
                        if found_safe then break end
                    end
                    best_target = (safe_override ~= -1) and safe_override or player_abs_seg -- Fallback stay put if absolutely nothing found
                end
                fire_priority = AVOID_FIRE_PRIORITY -- Low fire priority when avoiding constraint
            end
        end

        -- Final calculation for firing
        should_fire = fire_priority > shot_count
        return best_target, 0, should_fire, false

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