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

-- Define the server address as a global variable
local SERVER_ADDRESS = "socket.m2macpro:9999"

-- Get the directory of the current script
local script_path = debug.getinfo(1,"S").source:match("@?(.*[/\\])")
if script_path then
  -- Prepend the script's directory to package.path
  package.path = script_path .. "?.lua;" .. package.path
  print("Added script directory to package.path: " .. script_path)
else
  print("Warning: Could not determine script directory.")
end

local AUTO_START_GAME         = true -- flag to control auto-starting during attract mode

-- Timer and FPS tracking variables 
local lastFPSTime            = os.time()
local frameCountSinceLastFPS = 0 
local socket                 = nil
local frame_count            = 0 
local total_bytes_sent       = 0

-- Access the main CPU and memory space
local mainCpu = nil
local mem = nil

-- Tracking variables
local previous_score = 0
local previous_level = 0
local previous_alive_state = 1  -- Track previous alive state, initialize as alive
local LastRewardState = 0
local shutdown_requested = false
local last_display_update = 0  -- Timestamp of last display update

-- Load the required modules
local Display       = require("display")
local SegmentUtils  = require("segment")
local StateUtils    = require("state")


-- Function to open socket connection
local function open_socket()

    -- Try to open socket connection
    local socket_success, err = pcall(function()
        -- Close existing socket if any
        if socket then
            socket:close()
            socket = nil 
        end
        
        socket = emu.file("rw")  -- "rw" mode for read/write
        local result = socket:open(SERVER_ADDRESS) 
        
        if result == nil then
            print("Successfully opened socket connection to " .. SERVER_ADDRESS) 
            -- Send initial 4-byte ping for handshake
            local ping_data = string.pack(">H", 0)  -- 2-byte integer with value 0
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


-- Function to calculate reward for the current frame
local function calculate_reward(game_state, level_state, player_state, enemies_state)
    local reward = 0
    local bDone = false

    if player_state.alive == 1 then

        -- Score-based reward (keep this as a strong motivator).  Filter out large bonus awards.
        local score_delta = player_state.score - previous_score
        if score_delta > 0 and score_delta < 5000 then
            reward = reward + (score_delta)  -- Amplify score impact
        end

        -- Penalize using superzapper; only in play mode, since it's also set during zoom (0x020)
        if (game_state.gamestate == 0x04) then
            if (player_state.superzapper_active ~= 0) then
                reward = reward - 100
            end
        end
                
        -- Enemy targeting logic
        local target_segment = enemies_state.nearest_enemy_seg 
        local player_segment = player_state.position & 0x0F

        -- Check enemy shots ONLY if not in zoom state (0x20)
        if game_state.gamestate ~= 0x20 then
            for i = 1, 4 do
                if enemies_state.enemy_shot_segments[i].value ~= SegmentUtils.INVALID_SEGMENT and enemies_state.enemy_shot_segments[i].value == player_segment then
                    
                    local shot_pos = enemies_state.shot_positions[i]
                    -- Check type is number AND value is not nil
                    if type(shot_pos) ~= "number" then 
                        error("FATAL: enemies_state.shot_positions[" .. tostring(i) .. "] is not a number. Type: " .. type(shot_pos) .. ". Value: " .. tostring(shot_pos), 0)
                    end

                    -- Comparison line (approx 206)
                    if (shot_pos <= 0x24) then 
                        if (player_state.spinner_commanded == 0) then 
                            reward = reward - 100
                        else
                            reward = reward + math.abs(player_state.spinner_commanded * 100) 
                        end
                    end
                end
            end
        end -- End of check for game_state ~= 0x20

        -- Reward shaping for avoiding close Fuseballs moving towards player
        for i = 1, 7 do
            -- Check if it's an active Fuseball (type 4) moving towards the player (moving_away == 0)
            if enemies_state.enemy_core_type[i] == 4 and enemies_state.enemy_moving_away[i] == 0 and enemies_state.enemy_depths[i] > 0 then
                local fuseball_depth = enemies_state.enemy_depths[i]
                local fuseball_rel_seg = enemies_state.enemy_segments[i] -- Use RELATIVE segment
                
                -- Check if player is aligned with this fuseball (relative segment is 0)
                if math.abs(fuseball_rel_seg) <= 1 then
                    -- Apply proximity penalties/rewards similar to shots, slightly stronger
                    if (fuseball_depth <= 0x24) then 
                        if (player_state.spinner_commanded == 0) then 
                            reward = reward - 150 -- Penalty for staying still near close fuseball
                        else
                            reward = reward + math.abs(player_state.spinner_commanded * 150) -- Reward for moving away
                        end
                    end
                    if (fuseball_depth < 0x80) then
                        reward = reward - (255 - fuseball_depth) / 8 -- General penalty for proximity
                    end
                end
            end
        end -- End of Fuseball check

        -- Reward for avoiding close fuseballs that are approaching
        
        if target_segment == SegmentUtils.INVALID_SEGMENT or game_state.gamestate == 0x20 then 
            -- No enemies or zooming: reward staying still more strongly
            reward = reward + (player_state.spinner_commanded == 0 and 50 or -20)
        else
            -- Get absolute segment and depth of nearest enemy (stored during EnemiesState:update)
            local nearest_abs_seg = enemies_state.nearest_enemy_abs_seg_raw 
            local enemy_depth = enemies_state.nearest_enemy_depth_raw -- Get stored depth

            -- Check if nearest_abs_seg is valid before proceeding (-1 means none found)
            if nearest_abs_seg ~= -1 then 
                 -- Calculate desired spinner and segment distance using ABSOLUTE segments
                local desired_spinner, segment_distance = SegmentUtils.calculate_direction_intensity(player_segment, nearest_abs_seg, level_state.is_open_level)

                -- Check alignment based on actual segment distance
                if segment_distance == 0 then 
                    reward = reward + 250
                    if player_state.shot_count > 0 then
                        reward = reward + 100
                    end
                else 
                    -- MISALIGNED CASE 
                    if (segment_distance < 2) then 
                        -- Check depth separately using the stored enemy_depth
                        if (enemy_depth <= 0x20) then 
                            if player_state.fire_commanded ~= 0 then
                                reward = reward + 250
                            else
                                reward = reward - 50
                            end
                        end
                    end
                    reward = reward + (8 - segment_distance) * 10 -- Scaled Proximity Reward
                    if desired_spinner * player_state.spinner_commanded < 0 then -- Movement penalty
                        reward = reward - 50
                    elseif desired_spinner ~= 0 and player_state.spinner_commanded == 0 then -- Inaction penalty when misaligned
                        reward = reward - 25 -- Reduced inaction penalty
                    end
                end
                
                -- Encourage maintaining shots in reserve (outside alignment check)
                if player_state.shot_count < 2 or player_state.shot_count > 7 then
                    reward = reward - 20
                elseif player_state.shot_count >= 5 then
                    reward = reward + 20
                end
            else
                 -- Handle case where nearest_abs_seg might be -1 (e.g., if enemy just died)
                 -- Default to no-enemy behavior
                 reward = reward + (player_state.spinner_commanded == 0 and 50 or -20) 
            end
        end

        -- Reward/penalty for being in/out of Pulsar lanes
        local player_abs_segment = player_state.position & 0x0F
        if enemies_state.pulsar_lanes[player_abs_segment + 1] == 1 then
            reward = reward - 50 -- Penalty for being in a pulsar lane
        end

    else -- Player not alive
        -- Major penalty for death to prioritize survival, equal to the cost of a bonus life in points
        if previous_alive_state == 1 then
            reward = reward - 2500
            bDone = true
        end
    end

    -- Update previous values
    previous_score = player_state.score
    previous_level = level_state.level_number
    previous_alive_state = player_state.alive

    LastRewardState = reward

    return reward, bDone
end

-- Function to send parameters and get action each frame
-- Remove unused 'controls' parameter
local function process_frame(rawdata) 
    -- Increment frame count after processing
    frame_count = frame_count + 1

    -- Check if socket is open, try to reopen if not
    if not socket then
        if not open_socket() then
            return 0, 0, 0  -- Return zeros for fire, zap, spinner_delta on socket error
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
        if socket then socket:close(); socket = nil end
        open_socket()
        return 0, 0, 0  -- Return zeros for fire, zap, spinner_delta on write error
    end
    
    -- Try to read from socket with timeout protection
    local fire, zap, spinner = 0, 0, 0  -- Default values
    local read_start_time = os.clock()
    local read_timeout = 10.0  -- 2000ms timeout for socket read
    
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
                action_bytes = nil
            end
        end
        
        if action_bytes and #action_bytes == 3 then
            -- Unpack the three signed 8-bit integers
            fire, zap, spinner = string.unpack("bbb", action_bytes)
        else
            -- Default action if read fails or times out
            print("Failed to read action from socket after " .. elapsed .. "s, got " .. 
                  (action_bytes and #action_bytes or 0) .. " bytes")
            fire, zap, spinner = 0, 0, 0
            
            -- If we timed out, reconnect
            if elapsed >= read_timeout then
                error("Socket read timeout exceeded")
            end
        end
    end)
    
    if not success then
        print("Error reading from socket:", err)
        -- Close and attempt to reopen socket
        if socket then socket:close(); socket = nil end
        open_socket()
        return 0, 0, 0  -- Return zeros for fire, zap, spinner_delta on read error
    end
    
    -- Return the three components directly
    return fire, zap, spinner
end

-- Try to access machine using the manager API with proper error handling
local success, err = pcall(function()
    -- First, check if manager is available
    if not manager then
        error("MAME manager API not available")
    end
    
    -- Then check if machine is available on manager
    if not manager.machine then
        error("manager.machine not available")
    end
    
    -- Finally, access the devices
    mainCpu = manager.machine.devices[":maincpu"]
    if not mainCpu then
        error("Main CPU not found")
    end
    
    mem = mainCpu.spaces["program"]
    if not mem then
        error("Program memory space not found")
    end
end)

-- Instantiate state objects 
local game_state    = StateUtils.GameState:new()
local level_state   = StateUtils.LevelState:new()
local player_state  = StateUtils.PlayerState:new()
local enemies_state = StateUtils.EnemiesState:new()
local controls      = StateUtils.ControlState:new() 

-- Update the frame_callback function
local function frame_callback()
    -- Check the time counter at address 0x0003
    local currentTimer = mem:read_u8(0x0003)
    
    -- Check if the timer changed
    if currentTimer == lastTimerValue then -- Use the global variable
        return true
    end
    lastTimerValue = currentTimer -- Update the global variable
    
    -- Increment frame count and track FPS
    frameCountSinceLastFPS = frameCountSinceLastFPS + 1 -- Use the global variable
    local currentTime = os.time()
    
    -- Calculate FPS every second
    if currentTime > lastFPSTime then -- Use the global variable
        -- Ensure division by zero doesn't happen if time hasn't advanced
        local time_diff = currentTime - lastFPSTime
        if time_diff > 0 then
            game_state.current_fps = frameCountSinceLastFPS / time_diff -- More accurate FPS
        else
            game_state.current_fps = 0 -- Or some default if time hasn't advanced
        end
        frameCountSinceLastFPS = 0 -- Reset the global variable
        lastFPSTime = currentTime -- Update the global variable
    end
    
    -- Update game state first
    game_state:update(mem)
    level_state:update(mem) 
    player_state:update(mem, level_state)
    enemies_state:update(mem, level_state)
    
    -- Check if the game mode is in high score entry, mash AAA if it is
    if game_state.game_mode == 0x80 then
        if game_state.frame_counter % 60 == 0 then
            controls.fire_field:set_value(1)
            print("Pressing Fire")
        elseif game_state.frame_counter % 60 == 30 then
            controls.fire_field:set_value(0)
            print("Releasing Fire")
        end
    elseif game_state.gamestate == 0x16 then
        -- Game is in level select mode, advance selection 
        -- controls:apply_action(mem, 0, 0, 9, game_state, player_state)
    end

    local num_values = 0 
    local bDone = false
    local status_message = ""
    local is_attract_mode = (game_state.game_mode & 0x80) == 0

    -- NOPs and game state massage
    mem:write_u8(0x0006, 2) -- Credits
    -- mem:write_direct_u8(0xCA6F, 0xEA); mem:write_direct_u8(0xCA70, 0xEA) -- Skip score NOP
    mem:write_direct_u8(0xA591, 0xEA); mem:write_direct_u8(0xA592, 0xEA) -- Copy protection NOP

    mem:write_u8(0x0004, 0) -- Countdown timer to zero, forces us through the attract mode stages
    if game_state.countdown_timer > 0 then game_state.countdown_timer = 0 end

    -- Handle Attract Mode vs Gameplay Mode
    if is_attract_mode then
        -- Auto-start logic
        if AUTO_START_GAME then 
            local port = manager.machine.ioport.ports[":IN2"]
            if port then -- Check if port exists
                local startField = port.fields["1 Player Start"] or 
                                   port.fields["P1 Start"] or 
                                   port.fields["Start 1"]
                
                if startField then
                    if game_state.frame_counter % 10 == 0 then
                        startField:set_value(1)
                    elseif game_state.frame_counter % 10 == 5 then
                        startField:set_value(0)
                    end
                else
                     print("Error: Could not find start button field in :IN2")
                end
            else
                print("Error: Could not find port :IN2")
            end
        end
    end 

        -- In attract mode, we don't need to calculate reward or get actions from Python
        -- We can potentially return early after handling display updates
        
    -- Gameplay Mode Logic
    local reward, bDoneCurrent = calculate_reward(game_state, level_state, player_state, enemies_state)
    bDone = bDoneCurrent

    -- Flatten state using StateUtils function, passing controls
    local frame_data
    frame_data, num_values = StateUtils.flatten_game_state_to_binary(reward, game_state, level_state, player_state, enemies_state, controls, bDone, shutdown_requested)

    -- Send state and get action (remove controls argument)
    local fire, zap, spinner = process_frame(frame_data)

    -- Update player_state with commanded actions
    player_state.fire_commanded = fire
    player_state.zap_commanded = zap
    player_state.spinner_commanded = spinner 

    -- Check if socket is open, try to reopen if not
    if not socket then
        if not open_socket() then
            if Display.SHOW_DISPLAY then 
                -- Pass reward=nil explicitly when waiting
                Display.update_display("Waiting for Python connection...", game_state, level_state, player_state, enemies_state, nil, 0, nil, total_bytes_sent, LastRewardState) 
            end
        end
    end

    -- If we reach here, the socket should be open.
    
    total_bytes_sent = total_bytes_sent + #frame_data

    -- Apply actions, passing mem to the controls object method
    if game_state.gamestate == 0x04 or game_state.gamestate == 0x20 or game_state.gamestate == 0x24 then
        controls:apply_action(mem, fire, zap, spinner, game_state, player_state) 
    end

    -- Update Display
    local current_time_high_res = os.clock()
    local should_update_display = (current_time_high_res - last_display_update) >= Display.DISPLAY_UPDATE_INTERVAL 
    if should_update_display and Display.SHOW_DISPLAY then 
        -- ADD DEBUG PRINT HERE
        -- Pass LastRewardState for both reward params as 'reward' is local to gameplay block
        Display.update_display(status_message, game_state, level_state, player_state, enemies_state, "N/A", num_values, LastRewardState, total_bytes_sent, LastRewardState) 
        last_display_update = current_time_high_res
    end

    return true 
end

-- Function to be called when MAME is shutting down

local function on_mame_exit()
    print("MAME is shutting down - Sending final save signal")
    shutdown_requested = true
        
    if game_state and level_state and player_state and enemies_state then
        local reward = calculate_reward(game_state, level_state, player_state, enemies_state) 
        -- Use StateUtils flatten, passing controls and true for shutdown
        local frame_data, num_values = StateUtils.flatten_game_state_to_binary(reward, game_state, level_state, player_state, enemies_state, controls, true, true) 
        
        if socket then
            -- Use updated process_frame call signature
            local result = process_frame(frame_data) 
            print("Final save complete, response: " .. (result or "none"))
        end
    end
    
    if socket then socket:close(); socket = nil end
    print("Socket closed during MAME shutdown")    
end

-- Register callbacks
print("Registering callbacks with MAME")
callback_ref = emu.add_machine_frame_notifier(frame_callback)
emu.register_stop(on_mame_exit)

print("Tempest AI Script Initialized.")
