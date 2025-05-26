local state_defs = require("state") -- Assuming state.lua is in the same dir

-- Define enemy type constants (copy from main.lua)
local ENEMY_TYPE_FLIPPER = 0
local ENEMY_TYPE_PULSAR = 1
local ENEMY_TYPE_TANKER = 2
local ENEMY_TYPE_SPIKER = 3
local ENEMY_TYPE_FUSEBALL = 4

-- Define constants (copy from main.lua)
local INVALID_SEGMENT = state_defs.INVALID_SEGMENT

-- Constants for danger-based system
local SAFE_DANGER_THRESHOLD = 0.5 -- When danger >= 0.5, seek safer alternatives
local TOP_RAIL_DEPTH = 0x15
local PULSAR_THRESHOLD = 0x80 -- Pulsing threshold for pulsar danger

local M = {} -- Module table

-- Global variables needed by calculate_reward (scoped within this module)
local previous_score = 0
local previous_level = 0
local previous_alive_state = 1 -- Track previous alive state, initialize as alive
local LastRewardState = 0

-- Build danger map for all segments (0-15, danger value 0-1.0)
function M.build_danger_map(enemies_state, is_open)
    local danger_map = {}
    -- Initialize all segments to 0 danger
    for i = 0, 15 do
        danger_map[i] = 0.0
    end
    
    -- Process enemy shots - highest immediate danger
    for i = 1, 4 do
        local shot_seg = enemies_state.enemy_shot_abs_segments[i]
        local shot_depth = enemies_state.shot_positions[i]
        if shot_seg ~= INVALID_SEGMENT and shot_depth > 0 then
            -- Shot danger decreases with depth
            local shot_danger = math.max(0.0, (0x30 - shot_depth) / 0x30)
            danger_map[shot_seg] = math.max(danger_map[shot_seg], shot_danger)
        end
    end
    
    -- Process enemies by type with specific danger calculations
    for i = 1, 7 do
        local enemy_type = enemies_state.enemy_core_type[i]
        local enemy_seg = enemies_state.enemy_abs_segments[i]
        local enemy_depth = enemies_state.enemy_depths[i]
        
        if enemy_seg ~= INVALID_SEGMENT and enemy_depth > 0 then
            local base_danger = 0.0
            
            -- Type-specific danger calculations
            if enemy_type == ENEMY_TYPE_FUSEBALL then
                -- High danger, decreases with depth
                base_danger = math.max(0.3, (0x25 - enemy_depth) / 0x25)
            elseif enemy_type == ENEMY_TYPE_FLIPPER then
                -- Medium danger, more dangerous when close
                base_danger = math.max(0.2, (0x20 - enemy_depth) / 0x20)
            elseif enemy_type == ENEMY_TYPE_PULSAR then
                -- Danger depends on pulsing state
                if enemies_state.pulsing > PULSAR_THRESHOLD then
                    base_danger = 0.8 -- Very dangerous when pulsing
                else
                    base_danger = math.max(0.1, (0x20 - enemy_depth) / 0x20)
                end
            elseif enemy_type == ENEMY_TYPE_TANKER then
                -- Medium danger, considers split behavior
                base_danger = math.max(0.2, (0x20 - enemy_depth) / 0x20)
                -- Add extra danger for tankers that will spawn enemies
                local split_behavior = enemies_state.enemy_split_behavior and enemies_state.enemy_split_behavior[i] or 0
                if split_behavior > 0 then
                    base_danger = base_danger + 0.2
                end
            elseif enemy_type == ENEMY_TYPE_SPIKER then
                -- Lower danger
                base_danger = math.max(0.1, (0x18 - enemy_depth) / 0x18)
            end
            
            -- Clamp danger to [0.0, 1.0]
            base_danger = math.max(0.0, math.min(1.0, base_danger))
            danger_map[enemy_seg] = math.max(danger_map[enemy_seg], base_danger)
        end
    end
    
    return danger_map
end

-- Find hunting target based on priority order: fuseball, tanker, flipper, spiker, pulsar
function M.find_hunting_target(enemies_state, player_abs_segment, is_open, abs_to_rel_func)
    local hunt_priority = {
        ENEMY_TYPE_FUSEBALL,
        ENEMY_TYPE_TANKER, 
        ENEMY_TYPE_FLIPPER,
        ENEMY_TYPE_SPIKER,
        ENEMY_TYPE_PULSAR
    }
    
    for _, target_type in ipairs(hunt_priority) do
        local best_seg = -1
        local min_distance = 255
        local best_depth = 255
        
        for i = 1, 7 do
            if enemies_state.enemy_core_type[i] == target_type and 
               enemies_state.enemy_depths[i] > 0 and
               enemies_state.enemy_abs_segments[i] ~= INVALID_SEGMENT then
                
                local enemy_seg = enemies_state.enemy_abs_segments[i]
                local enemy_depth = enemies_state.enemy_depths[i]
                local rel_dist = abs_to_rel_func(player_abs_segment, enemy_seg, is_open)
                local abs_dist = math.abs(rel_dist)
                
                -- Prefer closer enemies, with depth as tiebreaker
                if abs_dist < min_distance or (abs_dist == min_distance and enemy_depth < best_depth) then
                    min_distance = abs_dist
                    best_seg = enemy_seg
                    best_depth = enemy_depth
                end
            end
        end
        
        if best_seg ~= -1 then
            return best_seg, best_depth
        end
    end
    
    return -1, 255 -- No hunting target found
end

-- Find safest segment from danger map
function M.find_safest_segment(danger_map, current_seg, is_open, abs_to_rel_func)
    local safest_seg = current_seg
    local min_danger = danger_map[current_seg] or 1.0
    
    -- Search all segments for the safest one, preferring closer segments
    for seg = 0, 15 do
        local danger = danger_map[seg] or 1.0
        local distance = math.abs(abs_to_rel_func(current_seg, seg, is_open))
        
        -- Prefer lower danger, with distance as tiebreaker
        if danger < min_danger or (danger == min_danger and distance < math.abs(abs_to_rel_func(current_seg, safest_seg, is_open))) then
            min_danger = danger
            safest_seg = seg
        end
    end
    
    return safest_seg, min_danger
end


-- Preserved: Open level top rail enemy positioning logic
function M.handle_open_level_top_rail(enemies_state, player_abs_segment, is_open, abs_to_rel_func)
    if not is_open then
        return nil -- Only applies to open levels
    end
    
    local enemies_left = false
    local enemies_right = false
    local leftmost_seg = 16
    local rightmost_seg = -1
    
    -- Scan for top rail enemies (depth <= TOP_RAIL_DEPTH)
    for i = 1, 7 do
        local depth = enemies_state.enemy_depths[i]
        local seg = enemies_state.enemy_abs_segments[i]
        
        if depth > 0 and depth <= TOP_RAIL_DEPTH and seg ~= INVALID_SEGMENT then
            if seg < leftmost_seg then
                leftmost_seg = seg
            end
            if seg > rightmost_seg then
                rightmost_seg = seg
            end
            
            local rel_dist = abs_to_rel_func(player_abs_segment, seg, is_open)
            if rel_dist < 0 then
                enemies_left = true
            elseif rel_dist > 0 then
                enemies_right = true
            end
        end
    end
    
    -- Apply the preserved logic
    if enemies_right and not enemies_left then
        -- Enemies right only → segment 1
        return 1
    elseif enemies_left and not enemies_right then
        -- Enemies left only → segment 13
        return 13
    end
    
    return nil -- No special positioning needed
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
-- Main function: Find target segment using danger-based movement strategy
function M.find_target_segment(game_state, player_state, level_state, enemies_state, abs_to_rel_func)
    local is_open = (level_state.level_type == 0xFF)
    local player_abs_seg = player_state.position & 0x0F
    local shot_count = player_state.shot_count or 0

    -- Handle tube zoom state
    if game_state.gamestate == 0x20 then
        return M.zoom_down_tube(player_abs_seg, level_state, is_open)
    end

    -- Only proceed with normal gameplay
    if game_state.gamestate ~= 0x04 then
        return player_abs_seg, 0, false, false
    end

    -- Build danger map for all segments
    local danger_map = M.build_danger_map(enemies_state, is_open)
    local current_danger = danger_map[player_abs_seg] or 0.0

    -- Check for preserved open level top rail positioning
    local top_rail_target = M.handle_open_level_top_rail(enemies_state, player_abs_seg, is_open, abs_to_rel_func)
    if top_rail_target then
        local top_rail_danger = danger_map[top_rail_target] or 0.0
        if top_rail_danger < SAFE_DANGER_THRESHOLD then
            -- Safe to use top rail positioning
            local fire_priority = (current_danger >= SAFE_DANGER_THRESHOLD) and 8 or 6
            local should_fire = fire_priority > shot_count
            return top_rail_target, 0, should_fire, false
        end
        -- Top rail target too dangerous, fall through to hunting logic
    end

    -- Safety override: If current position is dangerous, seek safety first
    if current_danger >= SAFE_DANGER_THRESHOLD then
        local safe_seg, safe_danger = M.find_safest_segment(danger_map, player_abs_seg, is_open, abs_to_rel_func)
        local fire_priority = 4 -- Lower priority when escaping danger
        local should_fire = fire_priority > shot_count
        return safe_seg, 0, should_fire, false
    end

    -- Hunting logic: Find target based on priority order
    local hunt_target, hunt_depth = M.find_hunting_target(enemies_state, player_abs_seg, is_open, abs_to_rel_func)
    
    if hunt_target ~= -1 then
        local hunt_danger = danger_map[hunt_target] or 0.0
        
        if hunt_danger < SAFE_DANGER_THRESHOLD then
            -- Safe to hunt this target
            local fire_priority = 6
            local should_fire = fire_priority > shot_count
            return hunt_target, 0, should_fire, false
        else
            -- Hunting target is too dangerous, find safer alternative
            local safe_seg, safe_danger = M.find_safest_segment(danger_map, player_abs_seg, is_open, abs_to_rel_func)
            local fire_priority = 4
            local should_fire = fire_priority > shot_count
            return safe_seg, 0, should_fire, false
        end
    end

    -- No hunting targets found, stay in current position if safe
    local fire_priority = 4
    local should_fire = fire_priority > shot_count
    return player_abs_seg, 0, should_fire, false
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

        -- === NEW: Penalize being in a dangerous pulsar lane ===
        if enemies_state.pulsing > 0xE0 then
            -- Check if the player is in a lane with a dangerous pulsar
            for i = 1, 7 do
                if enemies_state.enemy_core_type[i] == ENEMY_TYPE_PULSAR and
                   enemies_state.enemy_abs_segments[i] == player_segment and
                   enemies_state.enemy_depths[i] > 0 then
                    reward = reward - 50 -- Penalty for being in a dangerous pulsar lane
                    break
                end
            end
        end

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