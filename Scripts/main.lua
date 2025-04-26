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
    - Applies actions received from an external process via sockets.
    - Outputs a concise frame-by-frame summary of key stats for debugging and analysis.

    Usage:
    - Launch with MAME: `mame tempest -autoboot_script main.lua`
    - Ensure the Python AI server is running and accessible at socket.m2macpro.local:9999.
    - Memory addresses and input field names are based on the original Tempest ROM set.

    Notes:
    - Designed for educational use; extendable for full AI integration (e.g., RL model).

    Dependencies:
    - MAME with Lua scripting support enabled.
    - Tempest ROM set loaded in MAME.
--]]

--[[ package.path                  = package.path .. ";/Users/dave/source/repos/tempest/Scripts/?.lua" ]] -- Commented out hardcoded path

-- Dynamically add the script's directory to the package path
local script_path = debug.getinfo(1,"S").source:sub(2) -- Get source path, remove leading '@'
local script_dir = script_path:match("(.*[/\\])") or "./" -- Match up to last / OR \\ 
print("Detected script directory: " .. script_dir) -- Add print for verification
package.path = package.path .. ";" .. script_dir .. "?.lua"

-- Now require the module by name only (without path or extension)
local display = require("display") -- REVERTED: Require by module name only
local state_defs = require("state") -- ADDED: Require the new state module

-- Define constants
local INVALID_SEGMENT         = state_defs.INVALID_SEGMENT -- UPDATED: Get from state module

local SHOW_DISPLAY            = true
local DISPLAY_UPDATE_INTERVAL = 0.02

-- Access the main CPU and memory space
local mainCpu                 = nil
local mem                     = nil

-- Global socket variable
local socket = nil

-- Global variables for tracking bytes sent and FPS
local total_bytes_sent = 0

-- Level select counter
local level_select_counter = 0

-- Function to open socket connection
local function open_socket()
    -- Try to open socket connection
    local socket_success, err = pcall(function()
        -- Close existing socket if any
        if socket then
            socket:close()
            socket = nil
        end

        -- Create a new socket connection
        socket = emu.file("rw") -- "rw" mode for read/write
        local result = socket:open("socket.m2macpro.local:9999")

        if result == nil then
            -- Send initial 4-byte ping for handshake
            local ping_data = string.pack(">H", 0) -- 2-byte integer with value 0
            socket:write(ping_data)
        else
            print("Failed to open socket connection: " .. tostring(result))
            socket = nil
            return false
        end
    end)

    if not socket_success or not socket then
        print("Error opening socket connection: " .. tostring(err or "unknown error"))
        return false
    end


    return true
end

-- Declare global variables for reward calculation
local previous_score = 0
local previous_level = 0
local previous_alive_state = 1 -- Track previous alive state, initialize as alive

-- Declare a global variable to store the last reward state
local LastRewardState = 0

local shutdown_requested = false

local last_display_update = 0 -- Timestamp of last display update

-- Function to get the relative distance to a target segment
local function absolute_to_relative_segment(current_abs_segment, target_abs_segment, is_open_level)
    -- Ensure inputs are numbers, not tables
    current_abs_segment = tonumber(current_abs_segment) or 0
    target_abs_segment = tonumber(target_abs_segment) or 0

    -- Mask inputs to ensure they are within 0-15 range
    current_abs_segment = current_abs_segment & 0x0F
    target_abs_segment = target_abs_segment & 0x0F

    -- Get segment distance based on level type
    if is_open_level then
        -- Open level: simple distance calculation (-15 to +15)
        return target_abs_segment - current_abs_segment
    else
        -- Closed level: find shortest path around the circle (-7 to +8)
        local diff = target_abs_segment - current_abs_segment
        if diff > 8 then
            return diff - 16 -- Wrap around (e.g., 1 -> 15 is -2)
        elseif diff <= -8 then
            return diff + 16 -- Wrap around (e.g., 15 -> 1 is +2)
        else
            return diff
        end
    end
end

-- Forward declaration needed because calculate_reward uses find_target_segment,
-- which uses hunt_enemies, which uses find_nearest_enemy_of_type.
-- MOVED TO state.lua

-- Helper function to find nearest enemy of a specific type
-- MOVED TO state.lua

-- Helper function to hunt enemies in preference order
-- Needs abs_to_rel_func, is_open, and forbidden_segments passed in
hunt_enemies = function(enemies_state, player_abs_segment, is_open, abs_to_rel_func, forbidden_segments)
    -- Corrected Hunt Order: Fuseball(4), Pulsar(1), Flipper(0), Tanker(2), Spiker(3)
    local hunt_order = {
        M.ENEMY_TYPE_FUSEBALL, -- 4
        M.ENEMY_TYPE_PULSAR,   -- 1
        M.ENEMY_TYPE_FLIPPER,  -- 0
        M.ENEMY_TYPE_TANKER,   -- 2
        M.ENEMY_TYPE_SPIKER    -- 3
    }

    for _, enemy_type in ipairs(hunt_order) do
        local target_seg_abs, target_depth = find_nearest_enemy_of_type(enemies_state, player_abs_segment, is_open, enemy_type, abs_to_rel_func, forbidden_segments)

        if target_seg_abs ~= -1 then
            -- Check for Top Rail Avoidance Logic (ONLY for Flipper/Pulsar now)
            if target_depth <= 0x10 and (enemy_type == M.ENEMY_TYPE_FLIPPER or enemy_type == M.ENEMY_TYPE_PULSAR) then
                local rel_dist = abs_to_rel_func(player_abs_segment, target_seg_abs, is_open)
                if math.abs(rel_dist) <= 1 then -- If aligned or adjacent
                    local safe_adjacent_seg
                    if rel_dist <= 0 then safe_adjacent_seg = (player_abs_segment + 1) % 16 -- Move Right
                    else safe_adjacent_seg = (player_abs_segment - 1 + 16) % 16 end -- Move Left

                    if forbidden_segments[safe_adjacent_seg] then
                        print(string.format("AVOID F/P: Adjacent safe %d forbidden! Staying put.", safe_adjacent_seg))
                        return player_abs_segment, target_depth, true -- Stay put
                    else
                        print(string.format("AVOID F/P: Top Type %d at %d (Rel %d). Targeting adjacent safe %d", enemy_type, target_seg_abs, rel_dist, safe_adjacent_seg))
                        return safe_adjacent_seg, target_depth, true -- Target safe adjacent
                    end
                end
            end -- End Top Rail Flipper/Pulsar Check

            -- If no F/P avoidance was triggered, return the original hunt target
            -- Fuseball avoidance is handled globally before this function is called
            return target_seg_abs, target_depth, false

        end -- End if target_seg_abs ~= -1
    end -- End hunt_order loop

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
    local forbidden_segments = {} -- Define upfront for later use

    -- === HIGHEST PRIORITY: Check for immediate Fuseball threat ===
    if game_state.gamestate == 0x04 then -- Only check during normal gameplay
        local min_fuseball_dist = 3 -- Target distance is >= 3 (meaning <= 2 is too close)
        local fuseball_threat_nearby = false
        local escape_target_seg = -1

        for i = 1, 7 do
            if enemies_state.enemy_core_type[i] == M.ENEMY_TYPE_FUSEBALL and
               enemies_state.enemy_depths[i] <= 0x10 and -- Top rail only
               enemies_state.enemy_abs_segments[i] ~= M.INVALID_SEGMENT then

                local fuseball_abs_seg = enemies_state.enemy_abs_segments[i]
                local rel_dist = abs_to_rel_func(player_abs_seg, fuseball_abs_seg, is_open)

                if math.abs(rel_dist) <= 2 then -- Is player currently within 2 segments?
                    fuseball_threat_nearby = true
                    print(string.format("!!! FUSEBALL ALERT: Player at %d too close (Rel %d) to Fuseball at %d", player_abs_seg, rel_dist, fuseball_abs_seg))

                    -- Calculate escape segment (3 steps away)
                    if rel_dist <= 0 then -- Fuseball left or aligned, target 3 steps right
                        escape_target_seg = (player_abs_seg + 3) % 16
                    else -- Fuseball right, target 3 steps left
                         escape_target_seg = (player_abs_seg - 3 + 16) % 16
                    end
                    break -- Found a threat, no need to check further fuseballs
                end
            end
        end

        -- If a threat was found, determine the actual target (escape or stay put)
        if fuseball_threat_nearby then
            forbidden_segments = find_forbidden_segments(enemies_state, level_state, player_state) -- Need forbidden segments now
            if forbidden_segments[escape_target_seg] then
                 print(string.format("FUSEBALL AVOID: Escape seg %d forbidden! Staying put at %d.", escape_target_seg, player_abs_seg))
                 return player_abs_seg, 0, false, false -- Stay put, don't fire/zap
            else
                 print(string.format("FUSEBALL AVOID: Targeting escape seg %d.", escape_target_seg))
                 return escape_target_seg, 0, false, false -- Target escape, don't fire/zap
            end
        end
        -- If no immediate fuseball threat, continue to regular logic...
    end
    -- === END FUSEBALL PRIORITY CHECK ===


    -- Check for Tube Zoom state first
    if game_state.gamestate == 0x20 then
        -- ... (existing spike logic remains the same) ...
        local current_spike_h = level_state.spike_heights[player_abs_seg]
        if current_spike_h == 0 then return player_abs_seg, 0, true, false end
        local left_neighbour_seg = -1
        local right_neighbour_seg = -1
        if is_open then
            if player_abs_seg > 0 then left_neighbour_seg = player_abs_seg - 1 end
            if player_abs_seg < 15 then right_neighbour_seg = player_abs_seg + 1 end
        else
            left_neighbour_seg = (player_abs_seg - 1 + 16) % 16
            right_neighbour_seg = (player_abs_segment + 1) % 16
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
           return temp_target, 0, true, false
        end)()
    -- Check Flee/Hunt Logic (only in normal play mode, if no fuseball threat caused early return)
    elseif game_state.gamestate == 0x04 then
        -- Calculate forbidden segments if not already done by fuseball check
        if not next(forbidden_segments) then -- Check if table is empty
             forbidden_segments = find_forbidden_segments(enemies_state, level_state, player_state)
        end
        local current_segment_is_forbidden = forbidden_segments[player_abs_seg] or false

        if current_segment_is_forbidden then
            did_flee = true
            initial_target_seg_abs = find_nearest_safe_segment(player_abs_seg, is_open, forbidden_segments, abs_to_rel_func)
            target_depth = 0 -- Depth isn't the focus when fleeing
            should_fire = false
            should_zap = false
        else -- Current segment is SAFE, proceed to HUNT
            -- hunt_enemies now only handles Flipper/Pulsar top-rail avoidance
            local hunt_target_seg, hunt_target_depth, did_avoid = hunt_enemies(enemies_state, player_abs_seg, is_open, abs_to_rel_func, forbidden_segments)
            hunting_target_info = string.format("HuntTgt=%d, HuntDepth=%02X, Avoid=%s", hunt_target_seg, hunt_target_depth, tostring(did_avoid)) -- DEBUG

            if hunt_target_seg ~= -1 then
                initial_target_seg_abs = hunt_target_seg
                target_depth = hunt_target_depth
                -- Only recommend firing initially if NOT avoiding and aligned
                if not did_avoid then
                    local rel_dist = abs_to_rel_func(player_abs_seg, initial_target_seg_abs, is_open)
                    should_fire = (rel_dist == 0)
                else
                    should_fire = false -- Don't fire if avoidance maneuver was chosen
                end
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

    -- Apply panic braking (Checks next segment for Flipper/Pulsar/Shots/Fuseballs)
    local final_target_seg_abs = initial_target_seg_abs
    local did_brake = false
    if final_target_seg_abs ~= player_abs_seg and game_state.gamestate == 0x04 then
        local initial_relative_dist = abs_to_rel_func(player_abs_seg, final_target_seg_abs, is_open)
        local next_segment_abs = -1

        if initial_relative_dist > 0 then -- Moving right
            if is_open then
                if player_abs_seg < 15 then next_segment_abs = player_abs_seg + 1 end
            else next_segment_abs = (player_abs_seg + 1) % 16 end
        elseif initial_relative_dist < 0 then -- Moving left
             if is_open then
                if player_abs_seg > 0 then next_segment_abs = player_abs_seg - 1 end
            else next_segment_abs = (player_abs_seg - 1 + 16) % 16 end
        end

        if next_segment_abs ~= -1 then
            local brake_condition_met = false
            local braking_enemy_type = -1 -- For debugging

            -- Check enemy shots
            if not brake_condition_met then
                for i = 1, 4 do
                    if enemies_state.enemy_shot_abs_segments[i] == next_segment_abs and
                       enemies_state.shot_positions[i] > 0 and
                       enemies_state.shot_positions[i] <= 0x30 then
                        brake_condition_met = true; braking_enemy_type = -2; break -- Indicate shot
                    end
                end
            end

            -- Check Enemies (Flippers, Pulsars, AND top-rail Fuseballs)
            if not brake_condition_met then
                for i = 1, 7 do
                    local enemy_type = enemies_state.enemy_core_type[i]
                    local enemy_seg = enemies_state.enemy_abs_segments[i]
                    local enemy_depth = enemies_state.enemy_depths[i]

                    -- Check if enemy is in the next segment and close vertically
                    if enemy_seg == next_segment_abs and enemy_depth > 0 and enemy_depth <= 0x30 then
                        -- Brake for Flippers OR Pulsars at this depth
                        if (enemy_type == M.ENEMY_TYPE_FLIPPER or enemy_type == M.ENEMY_TYPE_PULSAR) then
                            brake_condition_met = true; braking_enemy_type = enemy_type; break
                        -- Brake for Fuseballs ONLY if they are very close (e.g., <= 0x10)
                        elseif (enemy_type == M.ENEMY_TYPE_FUSEBALL and enemy_depth <= 0x10) then
                            brake_condition_met = true; braking_enemy_type = enemy_type; break
                        end
                    end
                end
            end

            -- Apply brake if condition met
            if brake_condition_met then
                did_brake = true
                final_target_seg_abs = player_abs_seg
                target_depth = player_state.player_depth
                should_fire = false
                should_zap = false
                print(string.format("PANIC BRAKE triggered! Reason: Type %d in next segment %d", braking_enemy_type, next_segment_abs)) -- DEBUG
            end
        end
    end

    -- Apply nearby Flipper firing override (if not braking or avoiding)
    local initial_should_fire_flag = should_fire -- Capture should_fire before override
    if not initial_should_fire_flag and not did_brake and game_state.gamestate == 0x04 then
         -- Check if Hunt decided to avoid (might be redundant if fuseball check returned early, but safe)
         local _, _, did_avoid_hunt = hunt_enemies(enemies_state, player_abs_seg, is_open, abs_to_rel_func, forbidden_segments)
         if not did_avoid_hunt then
             for i = 1, 7 do
                 if enemies_state.enemy_core_type[i] == M.ENEMY_TYPE_FLIPPER and
                    enemies_state.enemy_depths[i] > 0 and enemies_state.enemy_depths[i] <= 0x30 then
                     local flipper_abs_seg = enemies_state.enemy_abs_segments[i]
                     if flipper_abs_seg ~= M.INVALID_SEGMENT then
                         local flipper_rel_seg = abs_to_rel_func(player_abs_seg, flipper_abs_seg, is_open)
                         if math.abs(flipper_rel_seg) <= 1 then
                             should_fire = true
                             print("FLIPPER OVERRIDE: Firing due to nearby flipper.")
                             break
                         end
                     end
                 end
             end
         end
    end

    -- Apply shot count override (happens last, only if not braking/avoiding)
    if game_state.gamestate == 0x04 and not did_brake then
        -- Check if Hunt decided to avoid
        local _, _, did_avoid_hunt = hunt_enemies(enemies_state, player_abs_seg, is_open, abs_to_rel_func, forbidden_segments)
        -- Also check if the initial Fuseball check forced an avoidance (which sets should_fire=false)
        -- We can infer this if the initial_should_fire_flag is false and the target is not the player's current segment
        -- Or simply check if the initial_should_fire_flag was already true
        if not did_avoid_hunt and not initial_should_fire_flag then
             should_fire = should_fire or player_state.shot_count < 5
        elseif initial_should_fire_flag then -- If hunt logic decided to fire, allow override
             should_fire = should_fire or player_state.shot_count < 5
        end
        -- If initial fuseball check or hunt avoidance happened, should_fire remains false
    end

    return final_target_seg_abs, target_depth, should_fire, should_zap
end

-- Function to calculate desired spinner direction and distance to target
local function direction_to_nearest_enemy(game_state, level_state, player_state, enemies_state)
    -- Get the player's current absolute segment
    local player_abs_seg = player_state.position & 0x0F
    -- Use reliable level number pattern to determine if level is open
    local level_num_zero_based = level_state.level_number - 1
    local is_open = (level_num_zero_based % 4 == 2)
    -- Get target directly from enemies_state
    local target_abs_segment = enemies_state.nearest_enemy_abs_seg_internal or -1

    -- If no target was provided (target_abs_segment is -1)
    if target_abs_segment == -1 then
        return 0, 0, 255 -- No target, return spinner 0, distance 0, max depth
    end

    -- Calculate the relative segment distance using the helper function
    local relative_dist = absolute_to_relative_segment(player_abs_seg, target_abs_segment, is_open)

    -- If already aligned (relative distance is 0)
    if relative_dist == 0 then
         -- Find depth (re-fetch from state)
         return 0, 0, enemies_state.nearest_enemy_depth_raw -- Returns 0, 0, Number
    end

    -- Calculate actual segment distance and intensity
    local actual_segment_distance = math.abs(relative_dist)
    local intensity = math.min(0.9, 0.3 + (actual_segment_distance * 0.05))
    -- Set spinner direction based on the sign of the relative distance
    local spinner = relative_dist > 0 and intensity or -intensity

    -- Depth isn't directly relevant when misaligned for spinner calculation
    return spinner, actual_segment_distance, 255 -- Returns Number, Number, 255
end

-- Function to calculate reward for the current frame
local function calculate_reward(game_state, level_state, player_state, enemies_state)
    local reward = 0
    local bDone = false
    local detected_spinner = player_state.spinner_detected -- Use detected spinner for reward

    -- Safety check for potentially uninitialized state (shouldn't happen in normal flow)
    if not enemies_state or enemies_state.nearest_enemy_abs_seg_internal == nil then
        print("WARNING: calculate_reward called with invalid enemies_state or nearest_enemy_abs_seg_internal is nil. Defaulting target.")
        enemies_state = enemies_state or {}
        enemies_state.nearest_enemy_abs_seg_internal = -1 -- Default to no target
        enemies_state.nearest_enemy_depth_raw = 255
        enemies_state.nearest_enemy_should_fire = false
    end

    -- Base survival reward - make staying alive more valuable
    if player_state.alive == 1 then
        -- Score-based reward (keep this as a strong motivator). Filter out large bonus awards.
        local score_delta = player_state.score - previous_score
        if score_delta > 0 and score_delta <= 1000 then
            reward = reward + (score_delta)
        end

        -- Encourage maintaining shots in reserve. Penalize 0 or 8, graduated reward for 1-7
        local sc = player_state.shot_count
        if sc == 0 then
            reward = reward - 50
        elseif sc == 4 then
            reward = reward + 5
        elseif sc == 5 then
            reward = reward + 10
        elseif sc == 6 then
            reward = reward + 15
        elseif sc == 7 then
            reward = reward + 20
        elseif sc >= 8 then -- Max shots is 8, handle this case
            reward = reward - 50
        end

        -- Penalize using superzapper; only in play mode, since it's also set during zoom (0x020)
        if (game_state.gamestate == 0x04) then
            if (player_state.superzapper_active ~= 0) then
                reward = reward - 500
            end
        end

        -- Enemy targeting logic - Use results stored in enemies_state by its update method
        local target_abs_segment = enemies_state.nearest_enemy_abs_seg_internal -- Use the internally stored absolute segment
        local target_depth = enemies_state.nearest_enemy_depth_raw
        local expert_should_fire = enemies_state.nearest_enemy_should_fire -- Get firing recommendation from state

        local player_segment = player_state.position & 0x0F

        -- Tube Zoom logic adjustment
        if game_state.gamestate == 0x20 then -- In tube zoom
            -- Reward based on inverse spike height (higher reward for shorter spikes)
            local spike_h = level_state.spike_heights[player_segment] or 0
            if spike_h > 0 then
                local effective_spike_length = 255 - spike_h                       -- Shorter spike = higher value
                reward = reward + math.max(0, (effective_spike_length / 2) - 27.5) -- Scaled reward
            else
                -- Use detected spinner for reward calculation
                reward = reward + (detected_spinner == 0 and 250 or -50)          -- Max reward if no spike
            end

            if (player_state.fire_commanded == 1) then -- Keep using commanded for checking fire intent
                reward = reward + 200
            end
        elseif target_abs_segment < 0 then -- This comparison should now be safe
            -- No enemies: reward staying still more strongly
            reward = reward + (detected_spinner == 0 and 150 or -20)
            if player_state.fire_commanded == 1 then reward = reward - 100 end -- Penalize firing at nothing
        else
            -- Get desired spinner direction, segment distance, AND enemy depth
            local desired_spinner, segment_distance, _ = direction_to_nearest_enemy(game_state, level_state, player_state, enemies_state)

            -- Check alignment based on actual segment distance
            if segment_distance == 0 then
                -- Big reward for alignment + firing incentive
                -- Use detected_spinner for checking if player is still
                if detected_spinner == 0 then
                    reward = reward + 250
                else
                    reward = reward - 50 -- Penalize moving when aligned
                end

                if player_state.fire_commanded then -- Keep checking commanded fire
                    reward = reward + 50
                end
            else
                -- MISALIGNED CASE (segment_distance > 0)
                -- Enemies at the top of tube should be shot when close (using segment distance)
                if (segment_distance < 2) then -- Check using actual segment distance
                    -- Use the depth stored in enemies_state
                    if (target_depth <= 0x20) then
                        if player_state.fire_commanded == 1 then -- Check commanded fire
                            reward = reward + 150
                        else
                            reward = reward - 50
                        end
                    else -- Close laterally, far depth
                         if player_state.fire_commanded == 1 then reward = reward - 20 end
                    end
                else -- Far laterally
                     if player_state.fire_commanded == 1 then reward = reward - 30 end
                end

                -- Movement incentives (using desired_spinner direction and *detected* spinner)
                -- Reward if the DETECTED movement (detected_spinner) is IN THE SAME direction as the desired direction.
                if desired_spinner * detected_spinner > 0 then
                    reward = reward + 40
                elseif desired_spinner * detected_spinner < 0 then
                    reward = reward - 50
                elseif detected_spinner == 0 and desired_spinner ~= 0 then
                    reward = reward - 15
                end
            end
        end
    else
        -- Large penalty for death to prioritize survival
        if previous_alive_state == 1 then
            reward = reward - 20000
            bDone = true
        end
    end

    -- Update previous values
    previous_score = player_state.score
    previous_level = level_state.level_number
    previous_alive_state = player_state.alive

    LastRewardState = reward

    -- Return only reward and done flag now
    return reward, bDone
end

-- Function to send parameters and get action each frame
local function process_frame(rawdata)
    -- Check if socket is open, try to reopen if not
    if not socket then
        if not open_socket() then
            return 0, 0, 0 -- Return zeros for fire, zap, spinner_delta on socket error
        end
    end

    -- Try to write to socket, handle errors
    local success, err = pcall(function()
        -- Add 4-byte length header to rawdata
        local data_length = #rawdata
        local length_header = string.pack(">H", data_length)

        -- Write length header followed by data
        socket:write(length_header)
        socket:write(rawdata)
    end)

    if not success then
        print("Error writing to socket:", err)
        -- Close and attempt to reopen socket
        if socket then
            socket:close(); socket = nil
        end
        open_socket()
        return 0, 0, 0 -- Return zeros for fire, zap, spinner_delta on write error
    end

    -- Try to read from socket with timeout protection
    local fire, zap, spinner = 0, 0, 0 -- Default values
    local read_start_time = os.clock()
    local read_timeout = 0.5           -- 500ms timeout for socket read

    success, err = pcall(function()
        -- Use non-blocking approach with timeout
        local action_bytes = nil
        local elapsed = 0

        while not action_bytes and elapsed < read_timeout do
            action_bytes = socket:read(3)

            if not action_bytes or #action_bytes < 3 then
                -- If read fails, sleep a tiny bit and try again
                -- Use non-blocking sleep
                local wait_start = os.clock()
                while os.clock() - wait_start < 0.01 do
                    -- Busy wait instead of os.execute which can freeze MAME
                end
                elapsed = os.clock() - read_start_time
                action_bytes = nil -- Ensure loop condition check uses updated elapsed
            else
                -- Ensure we read exactly 3 bytes
                if #action_bytes ~= 3 then
                     print("Warning: Expected 3 bytes from socket, received " .. #action_bytes)
                     action_bytes = nil -- Treat as incomplete read
                     -- Sleep briefly before retrying
                     local wait_start = os.clock()
                     while os.clock() - wait_start < 0.01 do end
                     elapsed = os.clock() - read_start_time
                end
            end
        end

        if action_bytes and #action_bytes == 3 then
            -- Unpack the three signed 8-bit integers
            fire, zap, spinner = string.unpack("bbb", action_bytes)
        else
            -- Default action if read fails or times out
            print("Failed to read action from socket after " .. string.format("%.2f", elapsed) .. "s, got " ..
                (action_bytes and #action_bytes or 0) .. " bytes. Defaulting action.")
            fire, zap, spinner = 0, 0, 0

            -- If we timed out, attempt to reconnect
            if elapsed >= read_timeout then
                print("Socket read timeout exceeded, attempting reconnect...")
                if socket then socket:close(); socket = nil end
                open_socket() -- Attempt reconnect immediately
            end
        end
    end)

    if not success then
        print("Error reading from socket:", err)
        -- Close and attempt to reopen socket
        if socket then
            socket:close(); socket = nil
        end
        open_socket()
        return 0, 0, 0 -- Return zeros for fire, zap, spinner_delta on read error
    end

    -- Return the three components directly
    return fire, zap, spinner
end

-- Seed the random number generator once at script start
math.randomseed(os.time())

-- Try to access machine using the manager API with proper error handling
local success, err = pcall(function()
    if not manager then error("MAME manager API not available") end
    if not manager.machine then error("manager.machine not available") end
    mainCpu = manager.machine.devices[":maincpu"]
    if not mainCpu then error("Main CPU not found") end
    mem = mainCpu.spaces["program"]
    if not mem then error("Program memory space not found") end
end)

if not success then
    print("Error accessing MAME machine via manager: " .. tostring(err))
    print("Attempting alternative access method...")
    success, err = pcall(function()
        if not machine then error("Neither manager.machine nor machine is available") end
        mainCpu = machine.devices[":maincpu"]
        if not mainCpu then error("Main CPU not found via machine") end
        mem = mainCpu.spaces["program"]
        if not mem then error("Program memory space not found via machine") end
    end)

    if not success then
        print("Error with alternative access method: " .. tostring(err))
        print("FATAL: Cannot access MAME memory")
        -- Potentially add emu.pause() or similar if running directly causes issues
        return -- Stop script execution if memory isn't accessible
    end
end

-- **GameState Class**
-- MOVED TO state.lua

-- **LevelState Class**
-- MOVED TO state.lua

-- **PlayerState Class**
-- MOVED TO state.lua

-- **EnemiesState Class**
-- MOVED TO state.lua

-- **Controls Class**
Controls = {}
Controls.__index = Controls

function Controls:new()
    local self = setmetatable({}, Controls)

    -- Find required input ports and fields
    self.button_port = manager.machine.ioport.ports[":BUTTONSP1"]
    self.spinner_port = manager.machine.ioport.ports[":KNOBP1"]
    self.in2_port = manager.machine.ioport.ports[":IN2"]

    -- Error checking for ports
    if not self.button_port then print("Warning: Could not find :BUTTONSP1 port.") end
    if not self.spinner_port then print("Warning: Could not find :KNOBP1 port.") end
    if not self.in2_port then print("Warning: Could not find :IN2 port.") end

    -- Find button fields
    self.fire_field = self.button_port and self.button_port.fields["P1 Button 1"] or nil
    self.zap_field = self.button_port and self.button_port.fields["P1 Button 2"] or nil

    -- Find spinner field
    self.spinner_field = self.spinner_port and self.spinner_port.fields["Dial"] or nil

    -- Find Start button field (try common names)
    self.p1_start_field = nil
    if self.in2_port then
        self.p1_start_field = self.in2_port.fields["1 Player Start"] or
            self.in2_port.fields["P1 Start"] or
            self.in2_port.fields["Start 1"]
    end

    -- Error checking for fields
    if not self.fire_field then print("Warning: Could not find 'P1 Button 1' field in :BUTTONSP1 port.") end
    if not self.zap_field then print("Warning: Could not find 'P1 Button 2' field in :BUTTONSP1 port.") end
    if not self.spinner_field then print("Warning: Could not find 'Dial' field in :KNOBP1 port.") end
    if not self.p1_start_field then print("Warning: Could not find P1 Start button field in :IN2 port.") end

    return self
end


-- Apply received AI action (fire, zap, spinner) and p1_start to game controls
function Controls:apply_action(fire, zap, spinner, p1_start)
    -- player_state commanded values are now set earlier in frame_callback
    -- player_state.fire_commanded = fire
    -- player_state.zap_commanded = zap
    -- player_state.spinner_commanded = spinner

    -- Apply actions to MAME input fields if they exist
    self.fire_field:set_value(fire)
    self.zap_field:set_value(zap)
    self.p1_start_field:set_value(p1_start)

    -- Ensure spinner value is within signed byte range (-128 to 127)
    local spinner_val = math.max(-128, math.min(127, spinner or 0))
    -- Use write_u8, as write_s8 might not exist. write_u8 handles the byte pattern correctly.
    mem:write_u8(0x0050, spinner_val) -- *** WRITE TO 0x0050 HAPPENS HERE ***
end


-- Instantiate state objects - AFTER defining all classes but BEFORE functions using them
-- UPDATED to use state_defs from require('state')
local game_state = state_defs.GameState:new()
local level_state = state_defs.LevelState:new()
local player_state = state_defs.PlayerState:new()
local enemies_state = state_defs.EnemiesState:new()
local controls = Controls:new() -- Instantiate controls after MAME interface is confirmed

-- Function to format section for display
-- MOVED TO display.lua

-- Function to move the cursor to a specific row (using ANSI escape code)
-- MOVED TO display.lua

-- Function to flatten and serialize the game state data to binary
local function flatten_game_state_to_binary(reward, game_state, level_state, player_state, enemies_state, bDone)
    -- Create a consistent data structure with fixed sizes
    local data = {}

    -- Game state (5 values)
    table.insert(data, game_state.gamestate)
    table.insert(data, game_state.game_mode)
    table.insert(data, game_state.countdown_timer)
    table.insert(data, game_state.p1_lives)
    table.insert(data, game_state.p1_level)

    -- Targeting Info / Engineered Features (5 values)
    table.insert(data, enemies_state.nearest_enemy_seg)           -- Relative segment (-7..8 / -15..15 or INVALID)
    table.insert(data, (enemies_state.nearest_enemy_seg ~= INVALID_SEGMENT) and enemies_state.nearest_enemy_seg or 0) -- Relative delta (use 0 if no target)
    table.insert(data, enemies_state.nearest_enemy_depth_raw)     -- Raw depth (0-255)
    -- Pack alignment as 0 or 10000 for integer representation
    table.insert(data, enemies_state.is_aligned_with_nearest > 0 and 10000 or 0)
    -- Pack error magnitude scaled to 0-10000 integer
    table.insert(data, math.floor(enemies_state.alignment_error_magnitude * 10000.0))

    -- Player state (7 values + arrays)
    table.insert(data, player_state.position) -- Absolute player position/segment byte
    table.insert(data, player_state.alive)
    table.insert(data, player_state.player_state)
    table.insert(data, player_state.player_depth)
    table.insert(data, player_state.superzapper_uses)
    table.insert(data, player_state.superzapper_active)
    table.insert(data, player_state.shot_count)

    -- Player shot positions (fixed size: 8) - Absolute depth
    for i = 1, 8 do
        table.insert(data, player_state.shot_positions[i]) -- Already initialized to 0 if unused
    end

    -- Player shot segments (fixed size: 8) - Relative segments
    for i = 1, 8 do
        table.insert(data, player_state.shot_segments[i]) -- Already initialized to INVALID_SEGMENT
    end

    -- Level state (3 values + arrays)
    table.insert(data, level_state.level_number)
    table.insert(data, level_state.level_type)
    table.insert(data, level_state.level_shape)

    -- Spike heights (fixed size: 16) - Absolute heights indexed 0-15
    for i = 0, 15 do
        table.insert(data, level_state.spike_heights[i] or 0)
    end

    -- Level angles (fixed size: 16) - Absolute angles indexed 0-15
    for i = 0, 15 do
        table.insert(data, level_state.level_angles[i] or 0)
    end

    -- Enemies state (counts: 10 values + other state)
    table.insert(data, enemies_state.active_flippers)
    table.insert(data, enemies_state.active_pulsars)
    table.insert(data, enemies_state.active_tankers)
    table.insert(data, enemies_state.active_spikers)
    table.insert(data, enemies_state.active_fuseballs)
    table.insert(data, enemies_state.spawn_slots_flippers)
    table.insert(data, enemies_state.spawn_slots_pulsars)
    table.insert(data, enemies_state.spawn_slots_tankers)
    table.insert(data, enemies_state.spawn_slots_spikers)
    table.insert(data, enemies_state.spawn_slots_fuseballs)
    table.insert(data, enemies_state.num_enemies_in_tube)
    table.insert(data, enemies_state.num_enemies_on_top)
    table.insert(data, enemies_state.enemies_pending)
    table.insert(data, enemies_state.pulsar_fliprate)
    table.insert(data, enemies_state.pulse_beat)
    table.insert(data, enemies_state.pulsing)

    -- Decoded Enemy Info (Size 7 slots * 6 fields = 42 values) - Absolute info
    for i = 1, 7 do
        table.insert(data, enemies_state.enemy_core_type[i])
        table.insert(data, enemies_state.enemy_direction_moving[i])
        table.insert(data, enemies_state.enemy_between_segments[i])
        table.insert(data, enemies_state.enemy_moving_away[i])
        table.insert(data, enemies_state.enemy_can_shoot[i])
        table.insert(data, enemies_state.enemy_split_behavior[i])
    end

    -- Enemy segments (fixed size: 7) - Relative segments
    for i = 1, 7 do
        table.insert(data, enemies_state.enemy_segments[i]) -- Already INVALID_SEGMENT if inactive
    end

    -- Enemy depths (fixed size: 7) - Absolute depth
    -- Combining MSB/LSB logic removed as LSB seemed unused/unverified
    for i = 1, 7 do
        table.insert(data, enemies_state.enemy_depths[i]) -- Already 0 if inactive
    end

    -- Top Enemy Segments (fixed size: 7) - Relative segments only for enemies at depth 0x10
    for i = 1, 7 do
        if enemies_state.enemy_depths[i] == 0x10 then
            table.insert(data, enemies_state.enemy_segments[i]) -- Insert the relative segment
        else
            table.insert(data, INVALID_SEGMENT)                 -- Use sentinel if not at depth 0x10 or inactive
        end
    end

    -- Enemy shot positions (fixed size: 4) - Absolute depth
    for i = 1, 4 do
        table.insert(data, enemies_state.shot_positions[i]) -- Already 0 if inactive
    end

    -- Enemy shot segments (fixed size: 4) - Relative segments
    for i = 1, 4 do
        table.insert(data, enemies_state.enemy_shot_segments[i]) -- Already INVALID_SEGMENT if inactive
    end

    -- Charging Fuseball flags per absolute segment (fixed size: 16, indexed 1-16)
    for i = 1, 16 do
        table.insert(data, enemies_state.charging_fuseball_segments[i]) -- Already 0 if not charging
    end

    -- Add pending_vid (64 bytes) - Absolute info
    for i = 1, 64 do
        table.insert(data, enemies_state.pending_vid[i]) -- Already 0 if unused
    end

    -- Add pending_seg (64 bytes) - Relative segments
    for i = 1, 64 do
        table.insert(data, enemies_state.pending_seg[i]) -- Already INVALID_SEGMENT if unused
    end

    -- Serialize the main data payload to a binary string. Convert all values to signed 16-bit integers.
    local binary_data = ""
    local num_values_packed = 0
    for i, value in ipairs(data) do
        local packed_value
        -- Ensure value is a number, default to 0 if not
        local num_value = tonumber(value) or 0

        -- Pack as signed 16-bit integer (little endian short 'h')
        -- Need to handle range: -32768 to 32767
        if num_value > 32767 then
            -- print(string.format("Warning: Value %d at index %d exceeds int16 max. Clamping.", num_value, i))
            packed_value = 32767
        elseif num_value < -32768 then
             -- print(string.format("Warning: Value %d at index %d exceeds int16 min. Clamping.", num_value, i))
             packed_value = -32768
        else
             packed_value = num_value
        end

        -- Use format '<h' for signed 16-bit little-endian
        -- NOTE: The original format was '>H' (unsigned big-endian). Keeping big-endian for compatibility, but using signed '<h' logic for packing.
        -- Let's stick to '>h' (signed big-endian) for consistency if Python expects that. Assuming '>h'.
        binary_data = binary_data .. string.pack(">h", packed_value)
        num_values_packed = num_values_packed + 1
    end

    -- --- OOB Data Packing ---
    local is_attract_mode = (game_state.game_mode & 0x80) == 0
    local is_open_level = level_state.level_type == 0xFF

    -- Get ABSOLUTE nearest enemy segment for OOB packing (-1 sentinel if none)
    local nearest_abs_seg_oob = enemies_state.nearest_enemy_abs_seg_internal or -1 -- CORRECTED: Use internal absolute value

    -- Expert recommendations (use state calculated in EnemiesState:update)
    local expert_should_fire = enemies_state.nearest_enemy_should_fire and 1 or 0 -- Read from state
    local expert_should_zap = enemies_state.nearest_enemy_should_zap and 1 or 0   -- Read from state

    -- Score packing
    local score = player_state.score or 0
    local score_high = math.floor(score / 65536) -- High 16 bits
    local score_low = score % 65536              -- Low 16 bits

    -- Frame counter packing (masked to 16 bits)
    local frame = game_state.frame_counter % 65536

    -- Check for save signal (moved near other OOB prep)
    local current_time = os.time()
    local save_signal = 0
    if shutdown_requested or current_time - game_state.last_save_time >= game_state.save_interval then
        save_signal = 1
        game_state.last_save_time = current_time
        if shutdown_requested then print("SHUTDOWN SAVE: Sending final save signal before MAME exits")
        else print("Periodic Save: Sending save signal to Python script") end
    end

    -- Pack OOB data using the format string expected by Python: >HdBBBHHHBBBhBhBBBB
    local oob_format = ">HdBBBHHHBBBhBhBBBB" -- 17 codes

    local oob_data = string.pack(oob_format,
        num_values_packed,              -- H: ushort (Should be 299)
        reward,                         -- d: double
        0,                              -- B: Placeholder (game_action?)
        game_state.game_mode,           -- B: uchar
        bDone and 1 or 0,               -- B: uchar (0/1)
        frame,                          -- H: ushort
        score_high,                     -- H: ushort
        score_low,                      -- H: ushort
        save_signal,                    -- B: uchar (0/1)
        player_state.fire_commanded,    -- B: uchar
        player_state.zap_commanded,     -- B: uchar
        player_state.spinner_commanded, -- h: short
        is_attract_mode and 1 or 0,     -- B: uchar (0/1)
        nearest_abs_seg_oob,            -- h: short (-1 to 15, packed as short)
        player_state.position & 0x0F,   -- B: uchar (0-15)
        is_open_level and 1 or 0,       -- B: uchar (0/1)
        expert_should_fire,             -- B: uchar (0/1) - From state
        expert_should_zap               -- B: uchar (0/1) - From state
    )

    -- Combine out-of-band header with game state data
    local final_data = oob_data .. binary_data

    return final_data, num_values_packed -- Return the combined data and the size of the main payload
end


-- Variables for frame timing and FPS calculation
local lastTimer = -1 -- Initialize to -1 to ensure first frame runs
local last_fps_time = os.time()
local last_frame_counter_for_fps = 0 -- Use a separate counter for FPS calc


-- Variables for frame skipping logic
local frames_to_wait = 0 -- How many timer ticks to wait before processing a frame (0 = process every tick)
local frames_waited = 0

-- Main frame callback for MAME
local function frame_callback()
    -- Frame skipping logic based on timer ticks at $0003
    local currentTimer = mem:read_u8(0x0003)
    if currentTimer == lastTimer then
        -- Timer hasn't changed, do nothing for this MAME frame
        return true
    end
    -- Timer has changed, update lastTimer
    lastTimer = currentTimer

    -- Increment wait counter and check if we should process this tick
    frames_waited = frames_waited + 1
    if frames_waited <= frames_to_wait then
        -- Skip processing this timer tick
        return true
    end
    -- Reset wait counter if we process this tick
    frames_waited = 0

    -- --- Start Processing Game Frame ---

    -- Calculate FPS (once per second)
    local current_time = os.time()
    if current_time > last_fps_time then
        game_state.current_fps = game_state.frame_counter - last_frame_counter_for_fps
        last_frame_counter_for_fps = game_state.frame_counter
        last_fps_time = current_time
    end

    -- Update game state objects (Order: Game -> Level -> Player -> Enemies)
    game_state:update(mem)
    level_state:update(mem)
    player_state:update(mem, absolute_to_relative_segment) -- Pass helper function
    enemies_state:update(mem, game_state, player_state, level_state, absolute_to_relative_segment) -- Pass dependencies & helper

    local bDone = false -- Indicates if the episode ended this frame

    -- Ensure socket connection is open
    if not socket then
        if not open_socket() then
            -- Socket failed, potentially update display and skip AI interaction
            display.update("Waiting for Python connection...", game_state, level_state, player_state, enemies_state, 0, 0, 0)
            -- Decide how to handle controls when no AI: maybe do nothing? Or basic attract mode logic?
            -- For now, just return without applying actions.
            return true
        end
    end

    -- --- Overrides / Cheats ---
    -- Set credits to 2 (prevents needing coins)
    mem:write_u8(0x0006, 2)
    -- NOP out copy protection memory corruption ($A591/$A592)
    mem:write_direct_u8(0xA591, 0xEA) -- NOP
    mem:write_direct_u8(0xA592, 0xEA) -- NOP

    -- --- AI Interaction ---
    local is_attract_mode = (game_state.game_mode & 0x80) == 0

    -- Calculate reward based on the *current* state and the *detected* spinner movement
    local reward, episode_done = calculate_reward(game_state, level_state, player_state, enemies_state) -- Correct args
    bDone = episode_done -- Capture if the episode ended

    -- Flatten and serialize the *current* game state (s') including reward info
    -- The flatten function now reads expert hints directly from enemies_state
    local frame_data, num_values = flatten_game_state_to_binary(reward, game_state, level_state, player_state, enemies_state, bDone) -- Correct args

    -- Send current state (s'), reward (r), done (d) to AI; receive action (a) for s'
    local received_fire_cmd, received_zap_cmd, received_spinner_cmd = process_frame(frame_data) -- Correct args

    -- Store received commands in player state immediately
    player_state.fire_commanded = received_fire_cmd
    player_state.zap_commanded = received_zap_cmd
    player_state.spinner_commanded = received_spinner_cmd

    -- Update total bytes sent (for display)
    total_bytes_sent = total_bytes_sent + #frame_data

    -- --- Apply AI/Manual Action ---
    -- Start with AI commands as base
    local final_fire_cmd = received_fire_cmd
    local final_zap_cmd = received_zap_cmd
    local final_spinner_cmd = received_spinner_cmd
    local final_p1_start_cmd = 0 -- Default P1 start to 0

    -- Handle specific game states / modes (override final commands)
    if game_state.gamestate == 0x12 then -- High Score Entry Mode
        final_fire_cmd = (game_state.frame_counter % 10 == 0) and 1 or 0
        final_zap_cmd = 0
        final_spinner_cmd = 0
    elseif game_state.gamestate == 0x16 then -- Level Select Mode
        if level_select_counter < 60 then
            final_spinner_cmd = 18; final_fire_cmd = 0; final_zap_cmd = 0
            level_select_counter = level_select_counter + 1
        elseif level_select_counter == 60 then
            final_fire_cmd = 1; final_spinner_cmd = 0; final_zap_cmd = 0
            level_select_counter = 61 -- Prevent re-pressing fire
        else
            final_fire_cmd = 0; 
            final_zap_cmd = 0; 
            final_spinner_cmd = 0
            level_select_counter = 0
        end
    elseif is_attract_mode then -- Attract Mode
        final_p1_start_cmd = (game_state.frame_counter % 50 == 0) and 1 or 0
        final_fire_cmd = 0; final_zap_cmd = 0; final_spinner_cmd = 0
        level_select_counter = 0
    elseif (game_state.gamestate == 0x04) or (game_state.gamestate == 0x20) then
        final_fire_cmd = player_state.fire_commanded
        final_zap_cmd = player_state.zap_commanded
        final_spinner_cmd = player_state.spinner_commanded
    else
        final_fire_cmd = 0; final_zap_cmd = 0; final_spinner_cmd = 0
    end

    -- Apply the final determined actions (AI or state-based overrides)

    if (final_p1_start_cmd == 1) then
        print("PRESSING START with fire=" .. final_fire_cmd .. " zap=" .. final_zap_cmd .. " spinner=" .. final_spinner_cmd)
    end
    
    controls:apply_action(final_fire_cmd, final_zap_cmd, final_spinner_cmd, final_p1_start_cmd)

    -- --- Update Display ---
    local current_time_high_res = os.clock()
    if SHOW_DISPLAY and (current_time_high_res - last_display_update) >= DISPLAY_UPDATE_INTERVAL then
        -- Use LastRewardState which was updated in calculate_reward
        -- Call the update function from the display module
        display.update("Running", game_state, level_state, player_state, enemies_state, num_values, LastRewardState, total_bytes_sent)
        last_display_update = current_time_high_res
    end

    return true -- Indicate success to MAME
end


-- Helper function to format segment values for display
-- MOVED TO display.lua

-- Function to update the console display
-- MOVED TO display.lua

-- Function to be called when MAME is shutting down
local function on_mame_exit()
    print("MAME is shutting down - Sending final save signal")

    -- Set shutdown flag to trigger save signal in flatten_game_state_to_binary
    shutdown_requested = true

    -- Try to process one final frame to send the save signal
    if mainCpu and mem and game_state and level_state and player_state and enemies_state and controls and socket then
        print("Processing final frame for save...")
        -- Update state objects one last time
        game_state:update(mem)
        level_state:update(mem)
        player_state:update(mem, absolute_to_relative_segment) -- Pass helper
        enemies_state:update(mem, game_state, player_state, level_state, absolute_to_relative_segment) -- Pass dependencies & helper

        -- Calculate final reward (value might not matter much here)
        local reward, _ = calculate_reward(game_state, level_state, player_state, enemies_state) -- Correct args

        -- Flatten state with shutdown_requested = true (handled inside flatten), bDone=true
        local frame_data, num_values = flatten_game_state_to_binary(reward, game_state, level_state, player_state, enemies_state, true) -- Correct args

        -- Send one last time using process_frame (ignore received action)
        local success, err = pcall(process_frame, frame_data) -- Correct args
        if success then
             print("Final save frame processed.")
        else
             print("Error processing final save frame: " .. tostring(err))
        end
    else
         print("Could not process final frame: required objects or socket not available.")
    end

    -- Close socket
    if socket then
        socket:close()
        socket = nil
        print("Socket closed during MAME shutdown.")
    end
end

-- Register callbacks with MAME
-- Keep references to prevent garbage collection
callback_ref = emu.add_machine_frame_notifier(frame_callback)
emu.add_machine_stop_notifier(on_mame_exit)

print("Tempest AI script initialized and callbacks registered.")

--[[ End of main.lua ]]--
