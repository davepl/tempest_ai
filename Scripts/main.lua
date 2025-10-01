--[[
    Tempest AI Lua Script for MAME - Refactored
    Author: Dave Plummer (davepl) and various AI assists
    Date: [2025-03-06] (Refactored Date)

    Overview: Refactored version focusing on modularity and clarity.
--]]

-- Dynamically add the script's directory to the package path
local script_path = debug.getinfo(1,"S").source:sub(2)
local script_dir = script_path:match("(.*[/\\])") or "./"
package.path = package.path .. ";" .. script_dir .. "?.lua"

-- Require modules
local display = require("display")
local state_defs = require("state")
local logic = require("logic") -- ADDED: Require the new logic module
local unpack = table.unpack or unpack -- Compatibility for unpack function

-- Constants
local SHOW_DISPLAY            = false
local START_ADVANCED          = false
local START_LEVEL_MIN         = 9
local DISPLAY_UPDATE_INTERVAL = 0.02
local SOCKET_ADDRESS          = "socket.ubdellamd:9999"
local SOCKET_READ_TIMEOUT_S   = 3.5  
local SOCKET_RETRY_WAIT_S     = 0.01
local CONNECTION_RETRY_INTERVAL_S = 5 -- How often to retry connecting (seconds)

-- MAME Interface Globals (initialized later)
local mainCpu                 = nil
local mem                     = nil

-- Global State
local current_socket = nil 
local total_bytes_sent = 0
local level_select_counter = 0
local shutdown_requested = false
local last_display_update = 0 -- Timestamp of last display update
local last_connection_attempt_time = 0 -- Timestamp of last connection attempt

-- Frame Skipping State
local last_timer_tick = -1
local frames_to_wait = 0 -- Process every tick by default
local frames_waited = 0

-- FPS Calculation State
local last_fps_time = os.time()
local last_frame_counter_for_fps = 0

-- Initialize MAME Interface (CPU and Memory)
local function initialize_mame_interface()
    local success, err = pcall(function()
        if not manager or not manager.machine then error("MAME manager.machine not available") end
        mainCpu = manager.machine.devices[":maincpu"]
        if not mainCpu then error("Main CPU not found") end
        mem = mainCpu.spaces["program"]
        if not mem then error("Program memory space not found") end
    end)

    if not success then
        print("Error accessing MAME via manager: " .. tostring(err))
        print("Attempting alternative access...")
        success, err = pcall(function()
            if not machine then error("Neither manager.machine nor machine is available") end
            mainCpu = machine.devices[":maincpu"]
            if not mainCpu then error("Main CPU not found via machine") end
            mem = mainCpu.spaces["program"]
            if not mem then error("Program memory space not found via machine") end
        end)

        if not success then
            print("Error with alternative access: " .. tostring(err))
            print("FATAL: Cannot access MAME memory.")
            return false -- Indicate failure
        end
    end
    print("MAME interface initialized successfully.")
    return true -- Indicate success
end

-- Socket Management
local function close_socket()
    if current_socket then
        current_socket:close()
        current_socket = nil
        -- print("Socket closed.") -- Optional: uncomment for debug
    end
end

local function open_socket()
    close_socket() -- Ensure any existing socket is closed first

    local socket_success, err = pcall(function()
        local sock = emu.file("rw")
        local result = sock:open(SOCKET_ADDRESS)
        if result == nil then
            -- Send initial 2-byte handshake message (required by server)
            local handshake_data = string.pack(">H", 0) -- 2-byte unsigned short, big-endian, value 0
            sock:write(handshake_data)

            current_socket = sock -- Assign to global only on success
            print("Socket connection opened to " .. SOCKET_ADDRESS)
            return true
        else
            print("Failed to open socket connection: " .. tostring(result))
            sock:close() -- Close the file handle if open failed
            return false
        end
    end)

    if not socket_success or not current_socket then
        print("Error during socket opening: " .. tostring(err or "unknown error"))
        close_socket() -- Ensure cleanup
        return false
    end
    return true
end

-- Controls Class (Simplified initialization)
local Controls = {}
Controls.__index = Controls

function Controls:new(mame_manager)
    local self = setmetatable({}, Controls)
    local ioport = mame_manager.machine.ioport

    local function find_port_field(port_name, field_name_options)
        local port = ioport.ports[port_name]
        if not port then print("Warning: Could not find port: " .. port_name); return nil end
        for _, field_name in ipairs(field_name_options) do
            local field = port.fields[field_name]
            if field then return field end
        end
        print("Warning: Could not find field " .. table.concat(field_name_options, "/") .. " in port " .. port_name)
        return nil
    end

    self.fire_field = find_port_field(":BUTTONSP1", {"P1 Button 1"})
    self.zap_field = find_port_field(":BUTTONSP1", {"P1 Button 2"})
    self.spinner_field = find_port_field(":KNOBP1", {"Dial"}) -- Spinner value is written directly to memory
    self.p1_start_field = find_port_field(":IN2", {"1 Player Start", "P1 Start", "Start 1"})

    return self
end

-- Apply received AI action and overrides to game controls
function Controls:apply_action(fire, zap, spinner, p1_start, memory)
    -- Debug values just before setting (simplified)
    -- print(string.format("[DEBUG apply_action] p1_start_val=%s, p1_start_field_valid=%s",
    --    tostring(p1_start),
    --    tostring(self.p1_start_field ~= nil)))

    if self.fire_field then self.fire_field:set_value(fire) end
    if self.zap_field then self.zap_field:set_value(zap) end

    if self.p1_start_field then
        -- Debug print *only* when attempting to set start=1
        -- if p1_start == 1 then
        --    print("[DEBUG apply_action] Attempting to set P1 Start = 1")
        -- end
    self.p1_start_field:set_value(p1_start)
    end

    -- Apply spinner value directly to memory (as before)
    local spinner_val = math.max(-128, math.min(127, spinner or 0))
    memory:write_u8(0x0050, spinner_val)
end

-- Instantiate state objects using state_defs
local game_state = state_defs.GameState:new()
local level_state = state_defs.LevelState:new()
local player_state = state_defs.PlayerState:new()
local enemies_state = state_defs.EnemiesState:new()
local controls = nil -- Initialized after MAME interface confirmed

-- Flatten game state to binary format for sending over socket
local function flatten_game_state_to_binary(reward, subj_reward, obj_reward, gs, ls, ps, es, bDone, expert_target_seg, expert_fire_packed, expert_zap_packed)
    local insert = table.insert -- Local alias for performance

    -- Helpers for normalized 8-bit integer packing - fail fast on out-of-range values!
    local function assert_range(v, lo, hi, context)
        if v < lo or v > hi then
            error(string.format("Value %g out of range [%g,%g] in %s", v, lo, hi, context or "unknown"))
        end
        return v
    end
    
    local function push_int8(parts, v)
        local val = tonumber(v) or 0
        -- CRITICAL: Assert final normalized value is within expected range [-128,127]
        if val < -128 or val > 127 then
            error(string.format("FINAL NORMALIZED VALUE %g out of range [-128,127] before serialization!", val))
        end
        insert(parts, string.pack(">b", val))  -- Big-endian int8
        return 1
    end

    local function push_float32(parts, v)
        local val = tonumber(v) or 0.0
        -- CRITICAL: Assert final normalized value is within expected range [-1,1]
        if val < -1.0 or val > 1.0 then
            error(string.format("FINAL NORMALIZED VALUE %g out of range [-1,1] before serialization!", val))
        end
        insert(parts, string.pack(">f", val))  -- Big-endian float32
        return 1
    end

    -- Normalize natural 8-bit values [0,255] to [0,1]
    local function push_natural_norm(parts, v)
        local num = tonumber(v) or 0
        local validated = assert_range(num, 0, 255, "natural_8bit")
        local val = validated / 255.0
        return push_float32(parts, val)
    end
    


    -- Normalize relative segments with proper Tempest range handling
    local INVALID_SEGMENT = state_defs.INVALID_SEGMENT or -32768
    local function push_relative_norm(parts, v)
        local num = tonumber(v) or 0
        if num == INVALID_SEGMENT then
            return push_float32(parts, -1.0)  -- INVALID sentinel
        end
        -- Assert actual Tempest range and normalize to [-1,+1]
        local validated = assert_range(num, -15, 15, "relative_segment")
        local normalized = validated / 15.0  -- [-15,+15] → [-1,+1]
        return push_float32(parts, normalized)
    end
    
    -- Normalize boolean to [0,1]
    local function push_bool_norm(parts, v)
        local val = v and 1.0 or 0.0
        return push_float32(parts, val)
    end
    
    -- Normalize unit float [0,1] (should already be normalized - assert if not)
    local function push_unit_norm(parts, v)
        local num = tonumber(v) or 0
        local validated = assert_range(num, 0.0, 1.0, "unit_float")
        return push_float32(parts, validated)
    end

    local binary_data_parts = {}
    local num_values_packed = 0

    -- Game state (5) - all natural values
    num_values_packed = num_values_packed + push_natural_norm(binary_data_parts, gs.gamestate)
    num_values_packed = num_values_packed + push_natural_norm(binary_data_parts, gs.game_mode)
    num_values_packed = num_values_packed + push_natural_norm(binary_data_parts, gs.countdown_timer)
    num_values_packed = num_values_packed + push_natural_norm(binary_data_parts, gs.p1_lives)
    num_values_packed = num_values_packed + push_natural_norm(binary_data_parts, gs.p1_level)

    -- Targeting / Engineered Features (4) - FIXED: Removed duplicate nearest_enemy_seg
    -- num_values_packed = num_values_packed + push_relative_norm(binary_data_parts, es.nearest_enemy_seg)
    -- num_values_packed = num_values_packed + push_natural_norm(binary_data_parts, es.nearest_enemy_depth_raw)
    -- num_values_packed = num_values_packed + push_bool_norm(binary_data_parts, es.is_aligned_with_nearest > 0)
    -- num_values_packed = num_values_packed + push_unit_norm(binary_data_parts, es.alignment_error_magnitude)

    -- Player state (7 + 8 + 8 = 23)
    num_values_packed = num_values_packed + push_natural_norm(binary_data_parts, ps.position)
    num_values_packed = num_values_packed + push_natural_norm(binary_data_parts, ps.alive)
    num_values_packed = num_values_packed + push_natural_norm(binary_data_parts, ps.player_state)
    num_values_packed = num_values_packed + push_natural_norm(binary_data_parts, ps.player_depth)
    num_values_packed = num_values_packed + push_natural_norm(binary_data_parts, ps.superzapper_uses)
    num_values_packed = num_values_packed + push_natural_norm(binary_data_parts, ps.superzapper_active)
    num_values_packed = num_values_packed + push_natural_norm(binary_data_parts, 8 - ps.shot_count)
    for i = 1, 8 do
        num_values_packed = num_values_packed + push_natural_norm(binary_data_parts, ps.shot_positions[i])
    end
    for i = 1, 8 do
        num_values_packed = num_values_packed + push_relative_norm(binary_data_parts, ps.shot_segments[i])
    end

    -- Level state (3 + 16 + 16 = 35) - all natural values
    num_values_packed = num_values_packed + push_natural_norm(binary_data_parts, ls.level_number)
    num_values_packed = num_values_packed + push_natural_norm(binary_data_parts, ls.level_type)
    num_values_packed = num_values_packed + push_natural_norm(binary_data_parts, ls.level_shape)
    for i = 0, 15 do
        num_values_packed = num_values_packed + push_natural_norm(binary_data_parts, ls.spike_heights[i] or 0)
    end
    for i = 0, 15 do
        num_values_packed = num_values_packed + push_natural_norm(binary_data_parts, ls.level_angles[i] or 0)
    end

    -- Enemies state (counts: 10 + other: 6 = 16) - all natural values
    num_values_packed = num_values_packed + push_natural_norm(binary_data_parts, es.active_flippers)
    num_values_packed = num_values_packed + push_natural_norm(binary_data_parts, es.active_pulsars)
    num_values_packed = num_values_packed + push_natural_norm(binary_data_parts, es.active_tankers)
    num_values_packed = num_values_packed + push_natural_norm(binary_data_parts, es.active_spikers)
    num_values_packed = num_values_packed + push_natural_norm(binary_data_parts, es.active_fuseballs)
    num_values_packed = num_values_packed + push_natural_norm(binary_data_parts, es.spawn_slots_flippers)
    num_values_packed = num_values_packed + push_natural_norm(binary_data_parts, es.spawn_slots_pulsars)
    num_values_packed = num_values_packed + push_natural_norm(binary_data_parts, es.spawn_slots_tankers)
    num_values_packed = num_values_packed + push_natural_norm(binary_data_parts, es.spawn_slots_spikers)
    num_values_packed = num_values_packed + push_natural_norm(binary_data_parts, es.spawn_slots_fuseballs)
    num_values_packed = num_values_packed + push_natural_norm(binary_data_parts, es.num_enemies_in_tube)
    num_values_packed = num_values_packed + push_natural_norm(binary_data_parts, es.num_enemies_on_top)
    num_values_packed = num_values_packed + push_natural_norm(binary_data_parts, es.enemies_pending)
    num_values_packed = num_values_packed + push_natural_norm(binary_data_parts, es.pulsar_fliprate)
    num_values_packed = num_values_packed + push_natural_norm(binary_data_parts, es.pulse_beat)
    num_values_packed = num_values_packed + push_natural_norm(binary_data_parts, es.pulsing)

    -- Decoded Enemy Info (7 * 6 = 42) - all natural values
    for i = 1, 7 do
        num_values_packed = num_values_packed + push_natural_norm(binary_data_parts, es.enemy_core_type[i])
        num_values_packed = num_values_packed + push_natural_norm(binary_data_parts, es.enemy_direction_moving[i])
        num_values_packed = num_values_packed + push_natural_norm(binary_data_parts, es.enemy_between_segments[i])
        num_values_packed = num_values_packed + push_natural_norm(binary_data_parts, es.enemy_moving_away[i])
        num_values_packed = num_values_packed + push_natural_norm(binary_data_parts, es.enemy_can_shoot[i])
        num_values_packed = num_values_packed + push_natural_norm(binary_data_parts, es.enemy_split_behavior[i])
    end

    -- Enemy segments (7) - relative values
    for i = 1, 7 do
        num_values_packed = num_values_packed + push_relative_norm(binary_data_parts, es.enemy_segments[i])
    end
    -- Enemy depths (7) - natural values
    for i = 1, 7 do
        num_values_packed = num_values_packed + push_natural_norm(binary_data_parts, es.enemy_depths[i])
    end
    -- Top Enemy Segments (7) - relative values
    for i = 1, 7 do
        local seg = (es.enemy_depths[i] == 0x10) and es.enemy_segments[i] or INVALID_SEGMENT
        num_values_packed = num_values_packed + push_relative_norm(binary_data_parts, seg)
    end
    -- Enemy shot positions (4) - natural values
    for i = 1, 4 do
        num_values_packed = num_values_packed + push_natural_norm(binary_data_parts, es.shot_positions[i])
    end
    -- Enemy shot segments (4) - relative values
    for i = 1, 4 do
        num_values_packed = num_values_packed + push_relative_norm(binary_data_parts, es.enemy_shot_segments[i])
    end
    -- Charging Fuseball segments (7) - relative values
    for i = 1, 7 do
        num_values_packed = num_values_packed + push_relative_norm(binary_data_parts, es.charging_fuseball[i])
    end
    -- Active Pulsar segments (7) - relative values
    for i = 1, 7 do
        num_values_packed = num_values_packed + push_relative_norm(binary_data_parts, es.active_pulsar[i])
    end
    -- Top Rail Enemy segments (7) - relative values
    for i = 1, 7 do
        num_values_packed = num_values_packed + push_relative_norm(binary_data_parts, es.active_top_rail_enemies[i])
    end

    -- Total main payload size: 175

    -- Serialize main data to binary string (float32 values)
    local binary_data = table.concat(binary_data_parts)

    -- VALIDATION: Debug print to verify key segment encodings (first few frames only)
    if gs.frame_counter < 5 then
        print(string.format("[DEBUG] Frame %d: Nearest enemy seg=%.3f, Player shot segs=[%.3f,%.3f,%.3f,%.3f]",
            gs.frame_counter,
            (es.nearest_enemy_seg == INVALID_SEGMENT) and -1.0 or (es.nearest_enemy_seg / 15.0),
            (ps.shot_segments[1] == INVALID_SEGMENT) and -1.0 or (ps.shot_segments[1] / 15.0),
            (ps.shot_segments[2] == INVALID_SEGMENT) and -1.0 or (ps.shot_segments[2] / 15.0),
            (ps.shot_segments[3] == INVALID_SEGMENT) and -1.0 or (ps.shot_segments[3] / 15.0),
            (ps.shot_segments[4] == INVALID_SEGMENT) and -1.0 or (ps.shot_segments[4] / 15.0)
        ))
    end

    -- --- OOB Data Packing ---
    local is_attract_mode = (gs.game_mode & 0x80) == 0
    local is_open_level = ls.level_type == 0xFF
    local score = ps.score or 0
    local score_high = math.floor(score / 65536)
    local score_low = score % 65536
    local frame = gs.frame_counter % 65536

    -- Save signal logic
    local current_time = os.time()
    local save_signal = 0
    if shutdown_requested or current_time - gs.last_save_time >= gs.save_interval then
        save_signal = 1
        gs.last_save_time = current_time
        if shutdown_requested then print("SHUTDOWN SAVE: Sending final save signal.")
        else print("Periodic Save: Sending save signal.") end
    end

    -- Fetch last reward components to transmit out-of-band (avoid recomputation in Python)
    -- Pack OOB data (header only, reward components removed for simplicity)
    -- NOTE: The short (h) after attract encodes the NEAREST ENEMY ABSOLUTE SEGMENT for Python expert steering.
    -- Format legend:
    --   >HdddBBBHHHBBBhBhBBBBB
    --   H: num_values, ddd: (reward, subj_reward, obj_reward), BBB: (gamestate, game_mode, done), HHH: (frame, score_hi, score_lo),
    --   BBB: (save, fire, zap), h: spinner, B: attract, h: nearest_enemy_abs_seg, B: player_seg, B: is_open,
    --   BB: (expert_fire, expert_zap), B: level_number
    local oob_format = ">HdddBBBHHHBBBhBhBBBBB"
    -- Determine nearest enemy absolute segment to transmit (or -1 if none)
    local oob_nearest_enemy_abs_seg = es.nearest_enemy_abs_seg_internal or -1
    local oob_data = string.pack(oob_format,
        num_values_packed,          -- H: Number of values in main payload (ushort)
        reward,                     -- d: Total reward (double)
        subj_reward,                -- d: Subjective reward (double)
        obj_reward,                 -- d: Objective reward (double)
        gs.gamestate,               -- B: Gamestate (uchar) - was placeholder
        gs.game_mode,               -- B: Game Mode (uchar)
        bDone and 1 or 0,           -- B: Done flag (uchar)
        frame,                      -- H: Frame counter (ushort)
        score_high,                 -- H: Score High (ushort)
        score_low,                  -- H: Score Low (ushort)
        save_signal,                -- B: Save Signal (uchar)
        ps.fire_commanded,          -- B: Commanded Fire (uchar)
        ps.zap_commanded,           -- B: Commanded Zap (uchar)
        ps.spinner_commanded,       -- h: Commanded Spinner (short)o
        is_attract_mode and 1 or 0, -- B: Is Attract Mode (uchar)
        oob_nearest_enemy_abs_seg,  -- h: Nearest Enemy ABS Segment (short)
        ps.position & 0x0F,         -- B: Player Abs Segment (uchar)
        is_open_level and 1 or 0,   -- B: Is Open Level (uchar)
        expert_fire_packed,         -- B: Expert Fire (uchar)
        expert_zap_packed,          -- B: Expert Zap (uchar)
        ls.level_number             -- B: Current Level Number (uchar)
    )

    -- Combine OOB header + main data
    local final_data = oob_data .. binary_data

    -- DEBUG: Updated length info for float32 payload: OOB ~32 bytes, Main=704 bytes -> Total ~736 bytes  
    -- print(string.format("Packed lengths: OOB~%d, Main=%d, Total=%d, Num values: %d", 32, #binary_data, #final_data, num_values_packed))

    return final_data, num_values_packed
end


-- Send state and receive action via socket
local function process_frame_via_socket(rawdata)
    -- Ensure socket connection
    if not current_socket then
        if not open_socket() then
            return 0, 0, 0, false -- Return zeros and error flag
        end
    end

    -- Attempt to write data with length header
    local write_success, write_err = pcall(function()
        local data_length = #rawdata
        local length_header = string.pack(">H", data_length) -- Unsigned short, big-endian length
        current_socket:write(length_header)
        current_socket:write(rawdata)
    end)

    if not write_success then
        print("Error writing to socket: " .. tostring(write_err) .. ". Attempting reconnect.")
        close_socket()
        open_socket() -- Try immediate reconnect
        return 0, 0, 0, false -- Return zeros and error flag
    end

    -- Attempt to read action with timeout
    local fire, zap, spinner = 0, 0, 0
    local read_success, read_result = pcall(function()
        local action_bytes = nil
        local read_start_time = os.clock()
        local elapsed = 0

        while elapsed < SOCKET_READ_TIMEOUT_S do
            -- Try reading 3 bytes for the action (b, b, b)
            action_bytes = current_socket:read(3)

            if action_bytes and #action_bytes == 3 then
                -- Successfully read 3 bytes
                 return { string.unpack("bbb", action_bytes) } -- Return unpacked values
            end

            -- If read failed or got partial data, just loop and rely on main timeout
            -- No explicit wait here; os.clock() check handles timing.
            elapsed = os.clock() - read_start_time
        end

        -- Loop finished without getting 3 bytes (Timeout)
        print("Socket read timeout after " .. string.format("%.3f", elapsed) .. "s. Expected 3 bytes.")
        if elapsed >= SOCKET_READ_TIMEOUT_S then
             print("Socket read timeout exceeded, attempting reconnect...")
             close_socket()
             open_socket()
        end
        return { 0, 0, 0 } -- Default action on timeout

    end)

    if not read_success then
        print("Error reading from socket: " .. tostring(read_result) .. ". Attempting reconnect.")
        close_socket()
        open_socket()
        return 0, 0, 0, false -- Return zeros and error flag
    end

    -- Return the received action values and success flag
    fire, zap, spinner = unpack(read_result) -- Unpack results from the table returned by pcall
    return fire, zap, spinner, true
end

-- Update all game state objects
local function update_game_states(memory)
    game_state:update(memory)
    level_state:update(memory)
    player_state:update(memory, logic.absolute_to_relative_segment) -- Pass helper from logic module
    enemies_state:update(memory, game_state, player_state, level_state, logic.absolute_to_relative_segment) -- Pass dependencies & helper
    -- DEBUG: Print game_mode immediately after update
    -- print(string.format("[DEBUG state update] Frame: %d, game_mode: 0x%02X", game_state.frame_counter, game_state.game_mode))
end

-- Perform AI interaction (calculate reward, expert advice, send state, receive action)
local function handle_ai_interaction()
    -- Calculate reward based on current state and detected actions
    local reward, subj_reward, obj_reward, episode_done = logic.calculate_reward(game_state, level_state, player_state, enemies_state, logic.absolute_to_relative_segment)
    
    -- NOTE: Removed reward clamping [-1,1] since rewards are now properly scaled in logic.calculate_reward()    -- Calculate expert advice (target segment, fire, zap)
    local is_open_level = (level_state.level_number - 1) % 4 == 2
    local expert_target_seg, _, expert_should_fire_lua, expert_should_zap_lua = logic.find_target_segment(
        game_state, player_state, level_state, enemies_state, logic.absolute_to_relative_segment, is_open_level
    )
    local expert_fire_packed = expert_should_fire_lua and 1 or 0
    local expert_zap_packed = expert_should_zap_lua and 1 or 0

    -- Default values if socket is not connected
    local received_fire_cmd, received_zap_cmd, received_spinner_cmd = 0, 0, 0
    local socket_ok = false
    local num_values = 0 -- Default if not sending

    -- Only attempt network ops if socket exists and seems valid
    if current_socket then
        -- Flatten current state (s') including reward (r) and done (d)
        local frame_data -- Declare frame_data here
        frame_data, num_values = flatten_game_state_to_binary(reward, subj_reward, obj_reward, game_state, level_state, player_state, enemies_state, episode_done, expert_target_seg, expert_fire_packed, expert_zap_packed)

        -- Send s', r, d; Receive action a for s'
        received_fire_cmd, received_zap_cmd, received_spinner_cmd, socket_ok = process_frame_via_socket(frame_data)

        -- Update total bytes sent (only if socket write was likely successful)
        if socket_ok then -- Assuming socket_ok implies write likely succeeded before read attempt
            total_bytes_sent = total_bytes_sent + #frame_data
        end
    else
        -- Socket doesn't exist, attempt to open it periodically
        local current_time = os.time()
        if current_time - last_connection_attempt_time > CONNECTION_RETRY_INTERVAL_S then
            -- print(string.format("[handle_ai_interaction] No active socket, attempting connect retry (Last attempt: %ds ago).", current_time - last_connection_attempt_time))
            last_connection_attempt_time = current_time -- Update time *before* attempting
            open_socket() -- Attempt connection
        end
    end

    -- Store received commands (will be 0s if no socket or read failed)
    player_state.fire_commanded = received_fire_cmd
    player_state.zap_commanded = received_zap_cmd
    player_state.spinner_commanded = received_spinner_cmd

    return episode_done, socket_ok, num_values -- Return done flag, socket status, and num_values for display
end

-- Determine the final action based on game state and AI commands (Returns: fire, zap, spinner, start)
local function determine_final_actions()
    -- Initialize all commands to 0, apply overrides below
    local final_fire_cmd = 0
    local final_zap_cmd = 0
    local final_spinner_cmd = 0
    local final_p1_start_cmd = 0
    local is_attract_mode = (game_state.game_mode & 0x80) == 0

    -- Override based on game state
    if game_state.gamestate == 0x12 then -- High Score Entry
        final_fire_cmd = (game_state.frame_counter % 10 == 0) and 1 or 0
        -- Zap, Spinner, Start remain 0
    elseif game_state.gamestate == 0x16 then -- Level Select
        -- DEBUG: Log Level Select State
        -- print(string.format("[DEBUG LevelSelect] Frame: %d, Timer: %d, Counter: %d", game_state.frame_counter, mem and mem:read_u8(0x0003) or -1, level_select_counter))

        if level_select_counter < 60 then
            if (START_ADVANCED) then
                final_spinner_cmd = 18; -- Fire, Zap, Start remain 0
            end
            level_select_counter = level_select_counter + 1
            -- print("  -> Spinning knob (Counter now " .. level_select_counter .. ")")
        elseif level_select_counter == 60 then
            final_fire_cmd = 1; -- Spinner, Zap, Start remain 0
            level_select_counter = level_select_counter + 1 -- Increment past 60 immediately
            -- print("  -> Pressing Fire! (Counter now " .. level_select_counter .. ")")
        else -- Counter is > 60, just do nothing, don't reset here
             -- All commands remain 0
             -- print("  -> Level selected, doing nothing.")
        end
        -- Resetting the counter should ONLY happen in attract mode.
    elseif is_attract_mode then -- Attract Mode
        local should_press_start = (game_state.frame_counter % 50 == 0)
        final_p1_start_cmd = should_press_start and 1 or 0
        -- Fire, Zap, Spinner remain 0

        -- DEBUG: Log counter reset
        -- if level_select_counter ~= 0 then
        --    print("[DEBUG Attract] Resetting level_select_counter from " .. level_select_counter .. " to 0")
        -- end
        level_select_counter = 0 -- Reset level select counter here

    elseif game_state.gamestate == 0x04 or game_state.gamestate == 0x20 or game_state.gamestate == 0x20 then -- Normal Play or Tube Zoom
        -- Use AI commands stored in player_state
        final_fire_cmd = player_state.fire_commanded
        final_zap_cmd = player_state.zap_commanded
        final_spinner_cmd = player_state.spinner_commanded
        -- Start remains 0
    else
        -- Unknown state, all commands default to 0
        -- print(string.format("[WARN] Unknown game state 0x%02X encountered in determine_final_actions", game_state.gamestate))
    end

    return final_fire_cmd, final_zap_cmd, final_spinner_cmd, final_p1_start_cmd
end

-- Update the console display if enabled and interval has passed
local function update_display_if_needed(num_values_packed)
    local current_time_high_res = os.clock()
    if SHOW_DISPLAY and (current_time_high_res - last_display_update) >= DISPLAY_UPDATE_INTERVAL then
        display.update("Running", game_state, level_state, player_state, enemies_state, num_values_packed, logic.getLastReward(), total_bytes_sent)
        last_display_update = current_time_high_res
    end
end

-- Apply cheats/overrides
local function apply_overrides(memory)
    memory:write_u8(0x0006, 2) -- Credits
    memory:write_direct_u8(0xA591, 0xEA) -- NOP Copy Prot
    memory:write_direct_u8(0xA592, 0xEA) -- NOP Copy Prot

    -- NOP out the start level check
    -- memory:write_direct_u8(0x90CD, 0xEA) -- NOP
    -- memory:write_direct_u8(0x90CE, 0xEA) -- NOP

    if (memory:read_u8(0x0126) < START_LEVEL_MIN) then
        memory:write_direct_u8(0x0126, START_LEVEL_MIN) -- NOP out the "Level Select" check
    end
end


-- Main frame callback for MAME
local function frame_callback()
    -- Frame skipping logic
    local currentTimer = mem:read_u8(0x0003)
    if currentTimer == last_timer_tick then return true end
    last_timer_tick = currentTimer
    frames_waited = frames_waited + 1
    if frames_waited <= frames_to_wait then return true end
    frames_waited = 0

    -- Calculate FPS
    local current_time = os.time()
    if current_time > last_fps_time then
        game_state.current_fps = game_state.frame_counter - last_frame_counter_for_fps
        last_frame_counter_for_fps = game_state.frame_counter
        last_fps_time = current_time
    end

    -- Update state from MAME memory
    update_game_states(mem)
  
    -- Apply overrides/cheats
    apply_overrides(mem)

    -- Handle AI Interaction (Send state s', get action a)
    local episode_done, socket_ok, num_values_packed = handle_ai_interaction()

    -- Determine final action based on AI input and game state
    local final_fire, final_zap, final_spinner, final_p1_start = determine_final_actions()

    -- DEBUG: Print final commands before applying
    -- print(string.format("[DEBUG Final Apply] Frame=%d, State=0x%02X, Fire=%d, Zap=%d, Spin=%d, Start=%d",
    --    game_state.frame_counter, game_state.gamestate, final_fire, final_zap, final_spinner, final_p1_start))

    -- Apply actions to controls
    controls:apply_action(final_fire, final_zap, final_spinner, final_p1_start, mem)

    -- Update console display periodically
    update_display_if_needed(num_values_packed)

    return true -- Indicate success to MAME
end

-- Function called when MAME is shutting down
local function on_mame_exit()
    print("MAME is shutting down...")
    shutdown_requested = true -- Signal for final save

    -- Try to process one final frame to send save signal if possible
    if mainCpu and mem and controls and current_socket then
        print("Processing final frame for save signal...")
        update_game_states(mem) -- Get final state
        -- Call AI handler - this calculates reward and flattens state with save signal
        handle_ai_interaction() -- Ignore return values, just need to send state
        print("Final frame processed and sent.")
    else
         print("Could not process final frame: MAME interface, controls, or socket not available.")
    end

    close_socket() -- Ensure socket is closed
    print("Shutdown complete.")
end

-- --- Script Initialization ---
math.randomseed(os.time())

-- Initialize MAME interface first
if not initialize_mame_interface() then
    return -- Stop script if MAME interface failed
end

-- Initialize controls now that MAME interface is confirmed
controls = Controls:new(manager)

-- Attempt initial socket connection
open_socket()

-- Register callbacks with MAME
-- Store reference globally like original script, in case of MAME GC quirks
global_callback_ref = emu.add_machine_frame_notifier(frame_callback)
emu.add_machine_stop_notifier(on_mame_exit)

print("Tempest AI script initialized and callbacks registered.")
--[[ End of main.lua ]]--



