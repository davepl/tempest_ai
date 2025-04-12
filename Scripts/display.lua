-- display.lua: Module for handling console display output

local Display = {}

-- Define constants locally within the module
local INVALID_SEGMENT = -32768  -- Used as sentinel value for invalid segments
local SHOW_DISPLAY = true
local DISPLAY_UPDATE_INTERVAL = 0.05  

-- Export constants
Display.SHOW_DISPLAY = SHOW_DISPLAY
Display.DISPLAY_UPDATE_INTERVAL = DISPLAY_UPDATE_INTERVAL

-- Local helper functions (not exported directly)
local function clear_screen()
    io.write("\027[2J\027[H")
end

local function move_cursor_home()
    io.write("\027[H")
end

-- Helper function to format segment values for display
local function format_segment(value)
    if value == INVALID_SEGMENT then
        return "---"
    else
        -- Use %+03d: this ensures a sign and pads with 0 to a width of 2 digits (total 3 chars like +01, -07)
        return string.format("%+03d", value)
    end
end

-- Moved from EnemiesState
local function decode_enemy_type(type_byte)
    local enemy_type = type_byte & 0x07
    local between_segments = (type_byte & 0x80) ~= 0
    local segment_increasing = (type_byte & 0x40) ~= 0
    return string.format("%d%s%s", 
        enemy_type,
        between_segments and "B" or "-",
        segment_increasing and "+" or ""  -- Keep consistent with previous
    )
end

-- Moved from EnemiesState
local function decode_enemy_state(state_byte)
    local split_behavior = state_byte & 0x03
    local can_shoot = (state_byte & 0x40) ~= 0
    local moving_away = (state_byte & 0x80) ~= 0
    return string.format("%s%s%s",
        moving_away and "A" or "T",
        can_shoot and "S" or "-",
        split_behavior
    )
end

local function format_section(title, metrics)
    local width = 40
    local separator = string.rep("-", width - 4)
    local result = string.format("--[ %s ]%s\n", title, separator)
    
    -- Find the longest key for alignment
    local max_key_length = 0
    for key, _ in pairs(metrics) do
        max_key_length = math.max(max_key_length, string.len(key))
    end
    
    -- Format each metric
    for key, value in pairs(metrics) do
        result = result .. string.format("  %-" .. max_key_length .. "s : %s\n", key, tostring(value))
    end
    
    return result
end

-- Function to move the cursor to a specific row
local function move_cursor_to_row(row)
    io.write(string.format("\027[%d;0H", row))
end

-- Exported functions
Display.clear_screen = clear_screen
Display.move_cursor_home = move_cursor_home

-- Update the update_display function to show total bytes sent and FPS
function Display.update_display(status, game_state, level_state, player_state, enemies_state, current_action, num_values, reward, total_bytes_sent, last_reward)
    clear_screen() -- Call local clear_screen
    move_cursor_to_row(1) -- Call local move_cursor_to_row

    -- Format and print game state in 3 columns at row 1
    print("--[ Game State ]--------------------------------------")
    
    -- Access global total_bytes_sent (needs careful consideration or passing)
    -- For now, assuming it's accessible or we adjust later
    local game_metrics = {
        {"Gamestate", string.format("0x%02X", game_state.gamestate)},
        {"Game Mode", string.format("0x%02X", game_state.game_mode)},
        {"Countdown", string.format("0x%02X", game_state.countdown_timer)},
        {"Credits", game_state.credits},
        {"P1 Lives", game_state.p1_lives},
        {"P1 Level", game_state.p1_level},
        {"Frame", game_state.frame_counter},
        {"Bytes Sent", total_bytes_sent}, -- Use passed parameter
        {"FPS", string.format("%.2f", game_state.current_fps)},
        {"Payload Size", num_values},
        {"Last Reward", string.format("%.2f", last_reward)} -- Use passed parameter
    }
    
    -- Calculate how many rows we need (ceiling of items/3)
    local rows = math.ceil(#game_metrics / 3)
    
    for row = 1, rows do
        local line = "  "
        for col = 1, 3 do
            local idx = (row - 1) * 3 + col
            if idx <= #game_metrics then
                local key, value = table.unpack(game_metrics[idx])
                -- Format each metric with fixed width to fit in 80 columns
                line = line .. string.format("%-14s: %-10s", key, tostring(value))
            end
        end
        print(line)
    end
    print("")  -- Empty line after section

    -- Format and print player state
    print("--[ Player State ]------------------------------------")
    
    -- Create player metrics in a more organized way for 3-column display
    local player_metrics = {
        {"Position", string.format("%d", player_state.position)},
        {"State", string.format("0x%02X", player_state.player_state)},
        {"Depth", string.format("%d", player_state.player_depth)},
        {"Alive", string.format("%d", player_state.alive)},
        {"Score", string.format("%d", player_state.score)},
        {"Szapper Uses", string.format("%d", player_state.superzapper_uses)},
        {"Szapper Active", string.format("%d", player_state.superzapper_active)},
        {"Shot Count", string.format("%d", player_state.shot_count)},
        {"Debounce", string.format("%d", player_state.debounce)},
        {"Fire Detected", string.format("%d", player_state.fire_detected)},
        {"Zap Detected", string.format("%d", player_state.zap_detected)},
        {"SpinnerAccum", string.format("%d", player_state.SpinnerAccum)},
        {"SpinnerDelta", string.format("%d", player_state.SpinnerDelta)},
        {"InferredDelta", string.format("%d", player_state.inferredSpinnerDelta)}
    }
    
    -- Calculate how many rows we need (ceiling of items/3)
    rows = math.ceil(#player_metrics / 3)
    
    for row = 1, rows do
        local line = "  "
        for col = 1, 3 do
            local idx = (row - 1) * 3 + col
            if idx <= #player_metrics then
                local key, value = table.unpack(player_metrics[idx])
                -- Format each metric with fixed width to fit in 80 columns
                line = line .. string.format("%-14s: %-10s", key, value)
            end
        end
        print(line)
    end
    
    -- Add shot positions on its own line
    local shots_str = ""
    for i = 1, 8 do
        shots_str = shots_str .. string.format(" %02X ", player_state.shot_positions[i])
    end
    print("  Shot Positions: " .. shots_str)
    
    -- Display player shot segments as a separate line
    local player_shot_segments_str = ""
    for i = 1, 8 do
        player_shot_segments_str = player_shot_segments_str .. format_segment(player_state.shot_segments[i]) .. " " -- Call local format_segment
    end
    print("  Shot Segments : " .. player_shot_segments_str)
    
    print("")  -- Empty line after section

    -- Format and print player controls (Reflect player_state commanded values)
    print("--[ Player Controls (Commanded Values) ]----------")
    print(string.format("  %-25s: %d", "Fire Commanded", player_state.fire_commanded))
    print(string.format("  %-25s: %d", "Superzapper Commanded", player_state.zap_commanded))
    print(string.format("  %-25s: %d", "Spinner Delta Commanded", player_state.SpinnerDelta)) 
    -- Display attract mode status based on game_state
    local is_attract_mode = (game_state.game_mode & 0x80) == 0
    print(string.format("  %-25s: %s", "Attract Mode", is_attract_mode and "Active" or "Inactive"))
    print("")

    -- Format and print level state 
    local enemy_types = {}
    local enemy_states = {}
    local enemy_segs = {}
    local enemy_depths = {}
    local enemy_lsbs = {}
    local enemy_shot_lsbs = {}  -- New array for shot LSBs
    for i = 1, 7 do
        -- Need access to enemies_state:decode_enemy_type/state
        -- These should ideally be moved here too or passed in
        -- For now, assume they are available globally or refactor later
        enemy_types[i] = decode_enemy_type(enemies_state.enemy_type_info[i]) -- Use local decode function
        enemy_states[i] = decode_enemy_state(enemies_state.active_enemy_info[i]) -- Use local decode function
        enemy_segs[i] = format_segment(enemies_state.enemy_segments[i]) -- Call local format_segment
        enemy_depths[i] = string.format("%02X", enemies_state.enemy_depths[i])
        enemy_lsbs[i] = string.format("%02X", enemies_state.enemy_depths_lsb[i])
        enemy_shot_lsbs[i] = string.format("%02X", enemies_state.enemy_shot_lsb[i])
    end

    local enemies_metrics = {
        ["Flippers"] = string.format("%d active, %d spawn slots", enemies_state.active_flippers, enemies_state.spawn_slots_flippers),
        ["Pulsars"] = string.format("%d active, %d spawn slots", enemies_state.active_pulsars, enemies_state.spawn_slots_pulsars),
        ["Tankers"] = string.format("%d active, %d spawn slots", enemies_state.active_tankers, enemies_state.spawn_slots_tankers),
        ["Spikers"] = string.format("%d active, %d spawn slots", enemies_state.active_spikers, enemies_state.spawn_slots_spikers),
        ["Fuseballs"] = string.format("%d active, %d spawn slots", enemies_state.active_fuseballs, enemies_state.spawn_slots_fuseballs),
        ["Total"] = string.format("%d active, %d spawn slots", 
            enemies_state:get_total_active(),
            enemies_state.spawn_slots_flippers + enemies_state.spawn_slots_pulsars + 
            enemies_state.spawn_slots_tankers + enemies_state.spawn_slots_spikers + 
            enemies_state.spawn_slots_fuseballs),
        ["Pulse State"] = string.format("beat:%02X charge:%02X/FF", enemies_state.pulse_beat, enemies_state.pulsing),
        ["Flip Rate"] = string.format("%02X", enemies_state.pulsar_fliprate),
        ["In Tube"] = string.format("%d enemies", enemies_state.num_enemies_in_tube),
        ["Nearest Enemy"] = string.format("segment %s", format_segment(enemies_state.nearest_enemy_seg)), -- Call local format_segment
        ["On Top"] = string.format("%d enemies", enemies_state.num_enemies_on_top),
        ["Pending"] = string.format("%d enemies", enemies_state.enemies_pending),
        ["Enemy Types"] = table.concat(enemy_types, " "),
        ["Enemy States"] = table.concat(enemy_states, " ")
    }

    print(format_section("Enemies State", enemies_metrics)) -- Call local format_section

    -- Add enemy segments and depths on their own lines
    print("  Enemy Segments: " .. table.concat(enemy_segs, " "))

    -- Display segments of enemies specifically at depth 0x10
    local top_enemy_segs = {}
    for i = 1, 7 do
        if enemies_state.enemy_depths[i] == 0x10 then
            top_enemy_segs[i] = format_segment(enemies_state.enemy_segments[i]) -- Call local format_segment
        else
            top_enemy_segs[i] = format_segment(INVALID_SEGMENT) -- Use local constant
        end
    end
    print("  Enemies On Top: " .. table.concat(top_enemy_segs, " "))

    print("  Enemy Depths  : " .. table.concat(enemy_depths, " "))
    print("  Enemy LSBs    : " .. table.concat(enemy_lsbs, " "))
    print("  Shot LSBs     : " .. table.concat(enemy_shot_lsbs, " "))
    
    -- Display decoded enemy info tables
    local enemy_core_types_str = table.concat(enemies_state.enemy_core_type, " ")
    local enemy_dir_mov_str = table.concat(enemies_state.enemy_direction_moving, " ")
    local enemy_between_str = table.concat(enemies_state.enemy_between_segments, " ")
    local enemy_mov_away_str = table.concat(enemies_state.enemy_moving_away, " ")
    local enemy_can_shoot_str = table.concat(enemies_state.enemy_can_shoot, " ")
    local enemy_split_str = table.concat(enemies_state.enemy_split_behavior, " ")
    
    print("  Enemy Core Type: " .. enemy_core_types_str)
    print("  Enemy Dir Mov  : " .. enemy_dir_mov_str)
    print("  Enemy Between  : " .. enemy_between_str)
    print("  Enemy Mov Away : " .. enemy_mov_away_str)
    print("  Enemy Can Shoot: " .. enemy_can_shoot_str)
    print("  Enemy Split Bhv: " .. enemy_split_str)
    
    -- Add enemy shot positions in a simple fixed format
    local shot_positions_str = ""
    for i = 1, 4 do
        local pos_value = enemies_state.shot_positions[i] or 0
        pos_value = pos_value & 0xFF
        shot_positions_str = shot_positions_str .. string.format(" %02X ", pos_value)
    end
    print("  Shot Positions: " .. shot_positions_str)
    
    -- Display enemy shot segments using format_segment
    local shot_segments_str = ""
    for i = 1, 4 do
        shot_segments_str = shot_segments_str .. format_segment(enemies_state.enemy_shot_segments[i].value) .. " " -- Call local format_segment
    end
    print("  Shot Segments : " .. shot_segments_str)
    
    print("")  -- Empty line after section

    -- Display pending_vid (64 bytes)
    local pending_vid_str = ""
    for i = 1, 64 do
        pending_vid_str = pending_vid_str .. string.format("%02X ", enemies_state.pending_vid[i])
        if i % 16 == 0 then pending_vid_str = pending_vid_str .. "\n  " end
    end
    print("  Pending VID   : ")
    print("  " .. pending_vid_str)

    -- Display pending_seg similarly
    local pending_seg_str = ""
    for i = 1, 64 do
        pending_seg_str = pending_seg_str .. format_segment(enemies_state.pending_seg[i]) .. " " -- Call local format_segment
        if i % 16 == 0 then pending_seg_str = pending_seg_str .. "\n  " end
    end
    print("  Pending SEG   : ")
    print("  " .. pending_seg_str)

    -- Display charging fuseball flags per absolute segment
    local charging_fuseball_str = {}
    for i = 1, 16 do
        if enemies_state.charging_fuseball_segments[i] == 1 then
            table.insert(charging_fuseball_str, "*")
        else
            table.insert(charging_fuseball_str, "-")
        end
    end
    print("  Fuseball Chrg : " .. table.concat(charging_fuseball_str, " "))
    
    print("")  -- Empty line after section
end

return Display 