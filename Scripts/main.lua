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
local START_LEVEL_MIN         = 9      -- Desired 1-based starting level (1 = default, >1 = skip ahead)
local DISPLAY_UPDATE_INTERVAL = 0.02
local SOCKET_ADDRESS          = "socket.ubvmdell:9999"
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

-- Tempest startlevtbl: maps selection index (0-based) → 0-based level number.
-- Mirrors the ROM table at startlevtbl in tempest.asm (standard table, !ALT_START_TABLE).
local startlevtbl = {
    [0]=0, 2, 4, 6, 8, 10, 12, 14, 16, 19, 21, 23, 25, 27, 30, 32,
    35, 39, 43, 46, 48, 51, 55, 59, 62, 64, 72, 80
}

--- Find the startlevtbl index whose level is closest to (but not exceeding)
--- the desired 1-based level.  Returns 0-based index.
local function level_to_select_index(desired_level_1based)
    local target = math.max(0, desired_level_1based - 1)  -- convert to 0-based
    local best_idx = 0
    for idx = 0, #startlevtbl do
        if startlevtbl[idx] <= target then
            best_idx = idx
        else
            break
        end
    end
    return best_idx
end
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
        if v < 0 or v > 255 then
            error(string.format("Value %g out of range [0,255] in natural_8bit", v))
        end
        local num = tonumber(v) or 0
        local validated = assert_range(num, 0, 255, "natural_8bit")
        local val = validated / 255.0
        return push_float32(parts, val)
    end

    -- Normalize precomputed spike heights to [0,1].
    -- LevelState builds these per lane as:
    --   depth == 0 and 0 or (255 - depth)
    local function push_spike_norm(parts, v)
        local num = tonumber(v) or 0
        if num < 0 or num > 255 then
            error(string.format("Value %g out of range [0,255] in spike_norm", num))
        end
        local val = num / 255.0
        return push_float32(parts, val)
    end

    -- Normalize 8.8 fixed point values [0,255.996] to [0,1]
    local function push_fixed_norm(parts, v)
        local num = tonumber(v) or 0
        local validated = assert_range(num, 0, 256, "fixed_8_8")  -- Allow up to 256
        local val = validated / 256.0
        return push_float32(parts, val)
    end
    

    -- Normalize relative segments with proper Tempest range handling
    local INVALID_SEGMENT = state_defs.INVALID_SEGMENT or -32768
    local TOP_RAIL_ABSENT = state_defs.TOP_RAIL_ABSENT or 255
    local function push_relative_norm(parts, v)
        local num = tonumber(v) or 0
        if num == INVALID_SEGMENT then
            return push_float32(parts, -1.0)  -- INVALID sentinel
        end
        -- Clamp to [-15,+15] — fractional sub-segment positions on open levels
        -- can slightly exceed 15 (e.g. 15.3125), so clamp rather than crash
        local clamped = math.max(-15.0, math.min(15.0, num))
        local normalized = clamped / 15.0  -- [-15,+15] → [-1,+1]
        return push_float32(parts, normalized)
    end

    -- Top-rail synthetic rel segments use 255 as "not present" default.
    local function push_toprail_relative_norm(parts, v)
        local num = tonumber(v) or TOP_RAIL_ABSENT
        if num == TOP_RAIL_ABSENT then
            return push_float32(parts, -1.0)
        end
        return push_relative_norm(parts, num)
    end
    
    -- Normalize boolean to [0,1]
    local function push_bool_norm(parts, v)
        local val
        if type(v) == "number" then
            val = (v ~= 0) and 1.0 or 0.0
        else
            val = v and 1.0 or 0.0
        end
        return push_float32(parts, val)
    end

    -- Normalize small enums/bitfields [0,maxv] to [0,1] so low-cardinality
    -- semantic features (type/bits) are not compressed into ~0.
    -- Clamps to [0,maxv] since raw memory can hold garbage during attract/init.
    local function push_small_enum_norm(parts, v, maxv, context)
        local num = tonumber(v) or 0
        if num < 0 then num = 0 end
        if num > maxv then num = maxv end
        if maxv == 0 then
            return push_float32(parts, 0.0)
        end
        return push_float32(parts, num / maxv)
    end
    
    -- Normalize raw Tempest depth values [0, 255] to [0, 1].
    -- IMPORTANT: TOP_OF_TUNNEL is 0x10 in Tempest; keep raw values so top-rail
    -- enemies do not collapse to 0 (which is reserved for inactive slots).
    local function push_depth_norm(parts, v)
        local num = tonumber(v) or 0
        if num < 0 then num = 0 end
        local validated = assert_range(num, 0, 255, "depth")
        local val = validated / 255.0  
        return push_float32(parts, val)
    end

    -- Normalize a signed value with known magnitude range to [-1, +1].
    -- Used for pulse_beat which oscillates e.g. +12 to -12.
    local function push_signed_norm(parts, v, max_magnitude, context)
        local num = tonumber(v) or 0
        if max_magnitude <= 0 then return push_float32(parts, 0.0) end
        local clamped = math.max(-max_magnitude, math.min(max_magnitude, num))
        local val = clamped / max_magnitude
        return push_float32(parts, val)
    end

    local binary_data_parts = {}
    local num_values_packed = 0

    -- Game state (5) - proper range normalization
    num_values_packed = num_values_packed + push_small_enum_norm(binary_data_parts, gs.gamestate, 36, "gamestate") -- even values 0-36 ($00-$24)
    num_values_packed = num_values_packed + push_natural_norm(binary_data_parts, gs.game_mode)  -- bitmask, full byte
    num_values_packed = num_values_packed + push_natural_norm(binary_data_parts, gs.countdown_timer) -- BCD or binary, keep raw
    num_values_packed = num_values_packed + push_small_enum_norm(binary_data_parts, gs.p1_lives, 6, "lives") -- 0-6 (capped at 6)
    num_values_packed = num_values_packed + push_small_enum_norm(binary_data_parts, math.min(gs.p1_level, 98), 98, "p1_level") -- 0-98 max, clamp 255 during attract

    -- Targeting / Engineered Features (4) - FIXED: Removed duplicate nearest_enemy_seg
    -- num_values_packed = num_values_packed + push_relative_norm(binary_data_parts, es.nearest_enemy_seg)
    -- num_values_packed = num_values_packed + push_natural_norm(binary_data_parts, es.nearest_enemy_depth_raw)
    -- num_values_packed = num_values_packed + push_bool_norm(binary_data_parts, es.is_aligned_with_nearest > 0)
    -- num_values_packed = num_values_packed + push_unit_norm(binary_data_parts, es.alignment_error_magnitude)

    -- Player state (7 + 8 + 8 = 23)
    num_values_packed = num_values_packed + push_small_enum_norm(binary_data_parts, ps.position, 15, "position") -- segment 0-15
    num_values_packed = num_values_packed + push_bool_norm(binary_data_parts, ps.alive)  -- 0 or 1
    num_values_packed = num_values_packed + push_small_enum_norm(binary_data_parts, ps.player_state & 0x0F, 15, "player_state_nibble") -- low nibble = next seg (0-15)
    -- Player depth: raw value from Tempest memory ($0202)
    num_values_packed = num_values_packed + push_depth_norm(binary_data_parts, ps.player_depth)
    num_values_packed = num_values_packed + push_small_enum_norm(binary_data_parts, ps.superzapper_uses, 2, "zap_uses") -- 0-2 uses consumed
    num_values_packed = num_values_packed + push_small_enum_norm(binary_data_parts, math.min(ps.superzapper_active, 19), 19, "zap_active") -- 0-19 countdown, $FF clamped
    num_values_packed = num_values_packed + push_small_enum_norm(binary_data_parts, 8 - ps.shot_count, 8, "remaining_shots") -- 0-8 remaining
    for i = 1, 8 do
        num_values_packed = num_values_packed + push_fixed_norm(binary_data_parts, ps.shot_positions[i])
    end
    for i = 1, 8 do
        num_values_packed = num_values_packed + push_relative_norm(binary_data_parts, ps.shot_segments[i])
    end

    -- Level state (3 + 16 + 16 = 35)
    num_values_packed = num_values_packed + push_small_enum_norm(binary_data_parts, math.min(ls.level_number, 98), 98, "level_number") -- 0-98 max, clamp 255 during attract
    num_values_packed = num_values_packed + push_bool_norm(binary_data_parts, ls.level_type)  -- $00=closed(0), $FF=open(1)
    num_values_packed = num_values_packed + push_small_enum_norm(binary_data_parts, ls.level_shape, 15, "level_shape") -- 0-15
    for i = 0, 15 do
        num_values_packed = num_values_packed + push_spike_norm(binary_data_parts, ls.spike_heights[i] or 0)
    end
    for i = 0, 15 do
        num_values_packed = num_values_packed + push_small_enum_norm(binary_data_parts, ls.level_angles[i] or 0, 15, "level_angle") -- 0-15 nibble
    end

    -- Enemies state — counts use proper ranges, speeds/misc keep raw byte
    num_values_packed = num_values_packed + push_small_enum_norm(binary_data_parts, es.active_flippers, 7, "active_flippers")
    num_values_packed = num_values_packed + push_small_enum_norm(binary_data_parts, es.active_pulsars, 7, "active_pulsars")
    num_values_packed = num_values_packed + push_small_enum_norm(binary_data_parts, es.active_tankers, 7, "active_tankers")
    num_values_packed = num_values_packed + push_small_enum_norm(binary_data_parts, es.active_spikers, 7, "active_spikers")
    num_values_packed = num_values_packed + push_small_enum_norm(binary_data_parts, es.active_fuseballs, 7, "active_fuseballs")
    num_values_packed = num_values_packed + push_small_enum_norm(binary_data_parts, es.spawn_slots_flippers, 7, "spawn_flippers") -- DUBIOUS: not displayed to player
    num_values_packed = num_values_packed + push_small_enum_norm(binary_data_parts, es.spawn_slots_pulsars, 7, "spawn_pulsars")    -- DUBIOUS
    num_values_packed = num_values_packed + push_small_enum_norm(binary_data_parts, es.spawn_slots_tankers, 7, "spawn_tankers")    -- DUBIOUS
    num_values_packed = num_values_packed + push_small_enum_norm(binary_data_parts, es.spawn_slots_spikers, 7, "spawn_spikers")    -- DUBIOUS
    num_values_packed = num_values_packed + push_small_enum_norm(binary_data_parts, es.spawn_slots_fuseballs, 7, "spawn_fuseballs")-- DUBIOUS
    num_values_packed = num_values_packed + push_small_enum_norm(binary_data_parts, es.num_enemies_in_tube, 7, "num_in_tube")
    num_values_packed = num_values_packed + push_small_enum_norm(binary_data_parts, es.num_enemies_on_top, 7, "num_on_top")
    num_values_packed = num_values_packed + push_small_enum_norm(binary_data_parts, es.enemies_pending, 63, "enemies_pending") -- DUBIOUS: not displayed to player
    num_values_packed = num_values_packed + push_small_enum_norm(binary_data_parts, math.min(es.pulsar_fliprate, 40), 40, "pulsar_fliprate") -- 10-40 active, clamp 255→40
    -- pulse_beat is now signed (-12..+12); use signed normalization
    num_values_packed = num_values_packed + push_signed_norm(binary_data_parts, es.pulse_beat, 12, "pulse_beat")
    num_values_packed = num_values_packed + push_natural_norm(binary_data_parts, es.pulsing)  -- oscillator phase 0-255, keep raw
    num_values_packed = num_values_packed + push_small_enum_norm(binary_data_parts, es.flipper_move or 0, 135, "flipper_move") -- p-code PC, max 135
    num_values_packed = num_values_packed + push_natural_norm(binary_data_parts, es.fuse_move_prb or 0) -- probability threshold, /255 OK
    -- Combined 16-bit signed speeds: MSB*256+LSB, sign-extended.
    -- All crack_speed outputs are negative (enemies move toward player).
    -- Max magnitude 2560 = fuseball (flipper_speed × 2, worst case -1280 × 2).
    local function push_combined_speed(parts, lsb, msb)
        local combined = (msb or 0) * 256 + (lsb or 0)
        if combined > 32767 then combined = combined - 65536 end -- sign extend
        return push_signed_norm(parts, combined, 2560, "enemy_speed")
    end
    num_values_packed = num_values_packed + push_combined_speed(binary_data_parts, es.spd_flipper_lsb, es.spd_flipper_msb)
    num_values_packed = num_values_packed + push_combined_speed(binary_data_parts, es.spd_pulsar_lsb, es.spd_pulsar_msb)
    num_values_packed = num_values_packed + push_combined_speed(binary_data_parts, es.spd_tanker_lsb, es.spd_tanker_msb)
    num_values_packed = num_values_packed + push_combined_speed(binary_data_parts, es.spd_spiker_lsb, es.spd_spiker_msb)
    num_values_packed = num_values_packed + push_combined_speed(binary_data_parts, es.spd_fuseball_lsb, es.spd_fuseball_msb)

    -- Decoded Enemy Info (7 * 6 = 42) - keep meaningful dynamic range
    for i = 1, 7 do
        -- Enemy core type is decoded with mask 0x07, so allow full 0..7 range.
        num_values_packed = num_values_packed + push_small_enum_norm(binary_data_parts, es.enemy_core_type[i], 7, "enemy_core_type")
        num_values_packed = num_values_packed + push_small_enum_norm(binary_data_parts, es.enemy_direction_moving[i], 1, "enemy_direction_moving")
        num_values_packed = num_values_packed + push_small_enum_norm(binary_data_parts, es.enemy_between_segments[i], 1, "enemy_between_segments")
        num_values_packed = num_values_packed + push_small_enum_norm(binary_data_parts, es.enemy_moving_away[i], 1, "enemy_moving_away")
        num_values_packed = num_values_packed + push_small_enum_norm(binary_data_parts, es.enemy_can_shoot[i], 1, "enemy_can_shoot") -- DUBIOUS: internal flag
        num_values_packed = num_values_packed + push_small_enum_norm(binary_data_parts, es.enemy_split_behavior[i], 3, "enemy_split_behavior")
    end

    -- Enemy segments (7) - relative values (fractional when between segments)
    for i = 1, 7 do
        local seg = es.enemy_segments_fractional[i] or es.enemy_segments[i]
        num_values_packed = num_values_packed + push_relative_norm(binary_data_parts, seg)
    end
    -- Enemy depths (7) - raw Tempest depth (0 means inactive)
    for i = 1, 7 do
        num_values_packed = num_values_packed + push_depth_norm(binary_data_parts, es.enemy_depths[i])
    end
    -- Top Enemy Segments (7) - relative values (enemies at collision depth 0x10)
    for i = 1, 7 do
        local rel_seg = es.enemy_segments_fractional[i] or es.enemy_segments[i]
        local seg = (es.enemy_depths[i] == 0x10) and rel_seg or INVALID_SEGMENT
        num_values_packed = num_values_packed + push_relative_norm(binary_data_parts, seg)
    end
    -- Enemy shot positions (4) - fixed point values
    for i = 1, 4 do
        num_values_packed = num_values_packed + push_fixed_norm(binary_data_parts, es.shot_positions[i])
    end
    -- Enemy shot segments (4) - relative values
    for i = 1, 4 do
        num_values_packed = num_values_packed + push_relative_norm(binary_data_parts, es.enemy_shot_segments[i])
    end
    -- Pulsar depths table (7) - raw depth for active pulsars, 0 otherwise.
    -- Reuses this 7-wide block so payload width stays fixed.
    for i = 1, 7 do
        num_values_packed = num_values_packed + push_depth_norm(binary_data_parts, es.active_pulsar_depths[i] or 0)
    end
    -- Pulsar lanes table (7) - relative segment for active pulsars.
    for i = 1, 7 do
        num_values_packed = num_values_packed + push_relative_norm(binary_data_parts, es.active_pulsar[i])
    end
    -- Top Rail Enemy segments (7) - relative values
    for i = 1, 7 do
        num_values_packed = num_values_packed + push_toprail_relative_norm(binary_data_parts, es.active_top_rail_enemies[i])
    end

    -- ── Danger Proximity Features (3) ──────────────────────────────────
    -- Pre-computed nearest-threat depth for player's lane and ±1 adjacent lanes.
    -- Gives the network direct "danger here / escape there" signals without
    -- needing to cross-reference segment + depth arrays itself.
    -- Encoding: 0.0 = threat at rim (imminent danger), ~1.0 = no threat (safe)
    local player_abs_seg = ps.position & 0x0F
    local is_open = (ls.level_type ~= 0x00)

    -- Adjacent segments with open/closed topology awareness
    local adj_left, adj_right
    if is_open then
        adj_left  = (player_abs_seg > 0)  and (player_abs_seg - 1) or -1
        adj_right = (player_abs_seg < 15) and (player_abs_seg + 1) or -1
    else
        adj_left  = (player_abs_seg - 1) % 16
        adj_right = (player_abs_seg + 1) % 16
    end

    -- Scan enemies, enemy shots, AND spikes for a lane, return min depth [1..255] or 255 (safe)
    local function nearest_threat_in_lane(target_seg)
        if target_seg < 0 then return 255 end  -- open-level edge: no lane exists → safe
        local best = 255
        -- Include spike depth: raw depth in same coordinate space as enemy depths.
        -- Lower depth = spike extends closer to tube rim = more dangerous.
        -- Critical during zoom when spikes are the ONLY threat.
        local sd = ls.spike_depths[target_seg] or 0
        if sd > 0 and sd < best then
            best = sd
        end
        for i = 1, 7 do
            local d = es.enemy_depths[i]
            if d > 0 and es.enemy_abs_segments[i] == target_seg and d < best then
                best = d
            end
        end
        for i = 1, 4 do
            if es.enemy_shot_abs_segments[i] == target_seg then
                local d = math.floor(es.shot_positions[i])
                if d > 0 and d < best then best = d end
            end
        end
        return best
    end

    num_values_packed = num_values_packed + push_depth_norm(binary_data_parts, nearest_threat_in_lane(player_abs_seg))
    num_values_packed = num_values_packed + push_depth_norm(binary_data_parts, nearest_threat_in_lane(adj_left))
    num_values_packed = num_values_packed + push_depth_norm(binary_data_parts, nearest_threat_in_lane(adj_right))

    -- ── Enemy Velocity Features (14) ─────────────────────────────────
    -- Per-slot segment delta and depth delta from previous frame.
    -- Gives the network direct velocity signals for each enemy.
    for i = 1, 7 do
        num_values_packed = num_values_packed + push_signed_norm(binary_data_parts, es.enemy_delta_seg[i], 8, "enemy_delta_seg")
    end
    for i = 1, 7 do
        num_values_packed = num_values_packed + push_signed_norm(binary_data_parts, es.enemy_delta_depth[i], 128, "enemy_delta_depth")
    end

    -- ── Fuseball Phase Features (7) ──────────────────────────────────
    -- Per-slot fuseball lifecycle phase:
    --   0.0 = not a fuseball,  1.0 = charging,  0.5 = mid-flip,  -1.0 = retreating
    for i = 1, 7 do
        num_values_packed = num_values_packed + push_float32(binary_data_parts, es.enemy_fuseball_phase[i])
    end

    -- ── Enemy Shot Speed (1) ─────────────────────────────────────────
    -- Global enemy shot velocity (all shots share one speed per level).
    -- Combined 16-bit signed, same scale as other enemy speeds.
    num_values_packed = num_values_packed + push_combined_speed(binary_data_parts, es.enm_shotspd_lsb, es.enm_shotspd_msb)

    -- ── Per-slot Shot Delay (7) ──────────────────────────────────────
    -- DUBIOUS: internal countdown not visible to player
    -- Countdown timer until this enemy can fire; 0 = ready to shoot.
    -- Normalised by shot_holdoff (global max) when > 0, else /255 fallback.
    local holdoff = (es.shot_holdoff > 0) and es.shot_holdoff or 255
    for i = 1, 7 do
        local delay = es.enemy_shot_delay[i] or 0
        num_values_packed = num_values_packed + push_float32(binary_data_parts, math.min(delay / holdoff, 1.0))
    end

    -- ── Shot Holdoff (1) ─────────────────────────────────────────────
    -- DUBIOUS: internal cooldown ceiling not visible to player
    -- Global cooldown ceiling for all enemy shots. Higher = slower firing.
    num_values_packed = num_values_packed + push_depth_norm(binary_data_parts, es.shot_holdoff or 0)

    -- ── Per-slot P-code Program Counter (7) ──────────────────────────
    -- DUBIOUS: internal movement script state not visible to player
    -- Current instruction index in the movement P-code interpreter.
    -- Max observed is ~151 (ALVROM table size). Normalise /151.
    for i = 1, 7 do
        local pc = es.enemy_pcode_pc[i] or 0
        num_values_packed = num_values_packed + push_float32(binary_data_parts, math.min(pc / 151.0, 1.0))
    end

    -- ── Per-slot P-code Storage / Loop Counter (7) ───────────────────
    -- DUBIOUS: internal movement script state not visible to player
    -- Loop iteration counter used by P-code repeat instructions.
    -- Range 0-8 (max repeat count). Normalise /8.
    for i = 1, 7 do
        local stor = es.enemy_pcode_storage[i] or 0
        num_values_packed = num_values_packed + push_float32(binary_data_parts, math.min(stor / 8.0, 1.0))
    end

    -- ── Lethal Distance (1) ──────────────────────────────────────────
    -- DUBIOUS: internal threshold not visible to player
    -- Depth threshold below which pulsars electrocute the player.
    num_values_packed = num_values_packed + push_depth_norm(binary_data_parts, es.lethal_distance or 0)

    -- ── Pulse Field Active (1) ───────────────────────────────────────
    -- Binary: 1.0 if pulsar lanes are currently electrified, 0.0 otherwise.
    num_values_packed = num_values_packed + push_float32(binary_data_parts, es.pulse_field_active)

    -- ── Soonest Spawn (2) ────────────────────────────────────────────
    -- Timer: normalised countdown [0..1] of nearest pending spawn (0 = imminent).
    -- Lane: relative segment of that spawn (INVALID_SEGMENT → sentinel).
    num_values_packed = num_values_packed + push_float32(binary_data_parts, es.soonest_spawn_timer)
    num_values_packed = num_values_packed + push_relative_norm(binary_data_parts, es.soonest_spawn_lane)

    -- Total main payload size: 229

    -- Serialize main data to binary string (float32 values)
    local binary_data = table.concat(binary_data_parts)

    -- VALIDATION: Debug print to verify key segment encodings (first few frames only)
    -- if gs.frame_counter < 5 then
    --     print(string.format("[DEBUG] Frame %d: Nearest enemy seg=%.3f, Player shot segs=[%.3f,%.3f,%.3f,%.3f]",
    --         gs.frame_counter,
    --         (es.nearest_enemy_seg == INVALID_SEGMENT) and -1.0 or (es.nearest_enemy_seg / 15.0),
    --         (ps.shot_segments[1] == INVALID_SEGMENT) and -1.0 or (ps.shot_segments[1] / 15.0),
    --         (ps.shot_segments[2] == INVALID_SEGMENT) and -1.0 or (ps.shot_segments[2] / 15.0),
    --         (ps.shot_segments[3] == INVALID_SEGMENT) and -1.0 or (ps.shot_segments[3] / 15.0),
    --         (ps.shot_segments[4] == INVALID_SEGMENT) and -1.0 or (ps.shot_segments[4] / 15.0)
    --     ))
    -- end

    -- --- OOB Data Packing ---
    -- Python expects OOB header format: >HddBBBHIBBBhhBBBBB (no total reward, no attract byte)
    local is_open_level = ls.level_type ~= 0x00 -- Assembly: $00=closed, $FF=open
    local score = ps.score or 0
    local frame = gs.frame_counter % 65536

    -- Save signal logic
    local current_time = os.time()
    local save_signal = 0
    if shutdown_requested or current_time - gs.last_save_time >= gs.save_interval then
        save_signal = 1
        gs.last_save_time = current_time
        if shutdown_requested then print("SHUTDOWN SAVE: Sending final save signal.") end
    end

    -- Pack OOB data matching Python (reward total not sent; only subj,obj). Attract not sent.
    -- Format legend:
    --   >HddBBBHIBBBhhBBBBB
    --   H: num_values, dd: (subj_reward, obj_reward), BBB: (gamestate, game_mode, done), HI: (frame, score),
    --   BBB: (save, fire, zap), h: spinner, h: expert_target_abs_seg, B: player_seg, B: is_open,
    --   BB: (expert_fire, expert_zap), B: level_number
    local oob_format = ">HddBBBHIBBBhhBBBBB"
    -- Use logic.find_target_segment() output as the authoritative expert target segment.
    local oob_expert_target_abs_seg = expert_target_seg or -1
    local oob_data = string.pack(oob_format,
        num_values_packed,          -- H: Number of values in main payload (ushort)
        subj_reward,                -- d: Subjective reward (double)
        obj_reward,                 -- d: Objective reward (double)
        gs.gamestate,               -- B: Gamestate (uchar)
        gs.game_mode,               -- B: Game Mode (uchar)
        bDone and 1 or 0,           -- B: Done flag (uchar)
        frame,                      -- H: Frame counter (ushort)
        score,                      -- I: Score (uint)
        save_signal,                -- B: Save Signal (uchar)
        ps.fire_commanded,          -- B: Commanded Fire (uchar)
        ps.zap_commanded,           -- B: Commanded Zap (uchar)
        ps.spinner_commanded,       -- h: Commanded Spinner (short)
        oob_expert_target_abs_seg,  -- h: Expert target ABS Segment (short)
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
    local r1, r2, r3, r4 = logic.calculate_reward(game_state, level_state, player_state, enemies_state, logic.absolute_to_relative_segment)
    local reward, subj_reward, obj_reward, episode_done
    if r4 ~= nil then
        -- Signature: reward, subj, obj, done
        reward, subj_reward, obj_reward, episode_done = r1, r2, r3, r4
    else
        -- Signature: reward, done
        reward, episode_done = r1, r2
        subj_reward = reward
        obj_reward = 0.0
    end
    
    -- NOTE: Removed reward clamping [-1,1] since rewards are now properly scaled in logic.calculate_reward()    -- Calculate expert advice (target segment, fire, zap)
    local is_open_level = (level_state.level_type ~= 0x00)
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
        -- Poke the desired start level directly into the selection cursor.
        -- $0200 (player_seg) is the selection index during level select.
        -- $0127 is max selectable index — ensure it allows our choice.
        if level_select_counter < 10 then
            -- Wait a few frames for the level select screen to initialise,
            -- then poke the desired index and max selectable each frame until we fire.
            if START_LEVEL_MIN > 1 then
                local desired_idx = level_to_select_index(START_LEVEL_MIN)
                mem:write_u8(0x0200, desired_idx)           -- selection cursor
                mem:write_u8(0x0127, math.max(desired_idx, mem:read_u8(0x0127))) -- ensure max allows it
            end
            level_select_counter = level_select_counter + 1
        elseif level_select_counter == 10 then
            final_fire_cmd = 1  -- Press fire to confirm selection
            level_select_counter = level_select_counter + 1
        else -- Counter is > 10, selection confirmed, wait for state transition
             -- All commands remain 0
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

    elseif game_state.gamestate == 0x04 or game_state.gamestate == 0x20 then -- Normal Play or Tube Zoom
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
