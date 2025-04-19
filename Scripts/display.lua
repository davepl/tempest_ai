-- display.lua: Module for handling console display output

local Display = {}
local SegmentUtils = require("segment") -- Require segment utils

-- Define constants locally within the module
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
    -- Use constant from SegmentUtils
    if value == SegmentUtils.INVALID_SEGMENT then 
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
    clear_screen()
    move_cursor_to_row(1)

    -- Format and print game state in 3 columns at row 1
    print("--[ Game State ]--------------------------------------")
    
    -- Pre-format all game metrics
    local game_metrics = {
        {"Gamestate", string.format("0x%02X", game_state.gamestate)},
        {"Game Mode", string.format("0x%02X", game_state.game_mode)},
        {"Countdown", string.format("0x%02X", game_state.countdown_timer)},
        {"Credits", game_state.credits},
        {"P1 Lives", game_state.p1_lives},
        {"P1 Level", game_state.p1_level},
        {"Frame", game_state.frame_counter},
        {"Bytes Sent", total_bytes_sent},
        {"FPS", string.format("%.2f", game_state.current_fps)},
        {"Payload Size", num_values},
        {"Last Reward", string.format("%.2f", last_reward)}
    }
    
    -- Calculate and add Time (DDd HH:MM format @ 30 FPS)
    local total_seconds = game_state.frame_counter / 30
    local days = math.floor(total_seconds / 86400)
    local hours = math.floor((total_seconds % 86400) / 3600)
    local minutes = math.floor((total_seconds % 3600) / 60)
    local time_str = string.format("%02dd %02d:%02d", days, hours, minutes)
    table.insert(game_metrics, {"Time", time_str})
    
    -- Build display lines more efficiently
    local display_lines = {}
    local rows = math.ceil(#game_metrics / 3)
    for row = 1, rows do
        local line_parts = {"  "}
        for col = 1, 3 do
            local idx = (row - 1) * 3 + col
            if idx <= #game_metrics then
                local key, value = table.unpack(game_metrics[idx])
                table.insert(line_parts, string.format("%-14s: %-10s", key, tostring(value)))
            end
        end
        table.insert(display_lines, table.concat(line_parts))
    end
    print(table.concat(display_lines, "\n"))
    print("")

    -- Format and print player state
    print("--[ Player State (Detected) ]---------------------------")
    
    -- Pre-format player metrics
    local player_metrics = {
        {"Position", string.format("0x%02X", player_state.position)},
        {"Alive", player_state.alive},
        {"Score", player_state.score},
        -- {"Lives", player_state.player_lives}, -- Remove
        -- {"Level", player_state.level}, -- Remove
        {"Shot Count", player_state.shot_count},
        {"Superzapper", player_state.superzapper_active},
    }
    
    -- Build player state lines
    local player_lines = {}
    for _, metric in ipairs(player_metrics) do
        table.insert(player_lines, string.format("  %-25s: %s", metric[1], tostring(metric[2])))
    end
    print(table.concat(player_lines, "\n"))
    
    -- Format shot segments more efficiently
    local shot_segments = {}
    for i = 1, 8 do
        table.insert(shot_segments, format_segment(player_state.shot_segments[i]))
    end
    print("  Shot Segments  : " .. table.concat(shot_segments, " "))

    -- Format shot positions (depths)
    local shot_positions = {}
    for i = 1, 8 do
        table.insert(shot_positions, string.format("%02X", player_state.shot_positions[i] & 0xFF))
    end
    print("  Shot Positions :  " .. table.concat(shot_positions, "  "))
    print("")

    -- Format and print player controls
    print("--[ Player Controls (Commanded Values) ]----------")
    local control_lines = {
        string.format("  %-25s: %d", "Fire Commanded", player_state.fire_commanded),
        string.format("  %-25s: %d", "Superzapper Commanded", player_state.zap_commanded),
        string.format("  %-25s: %d", "Spinner Delta Commanded", player_state.spinner_commanded),
        string.format("  %-25s: %s", "Attract Mode", (game_state.game_mode & 0x80) == 0 and "Active" or "Inactive")
    }
    print(table.concat(control_lines, "\n"))
    print("")

    -- Format and print level state
    print("--[ Level State ]------------------------------------")
    local level_metrics = {
        {"Level Number", level_state.level_number},
        {"Is Open Level", level_state.is_open_level and "Yes" or "No"},
        {"Enemies In Tube", enemies_state.num_enemies_in_tube},
        {"Enemies On Top", enemies_state.num_enemies_on_top}
    }
    
    local level_lines = {}
    for _, metric in ipairs(level_metrics) do
        table.insert(level_lines, string.format("  %-25s: %s", metric[1], tostring(metric[2])))
    end
    print(table.concat(level_lines, "\n"))

    -- Format and print spike heights
    local spike_height_vals = {}
    for i = 1, 16 do
        -- Format as 2-digit hex, handle potential nil with 'or 0'
        table.insert(spike_height_vals, string.format("%02X", level_state.spike_heights[i] or 0))
    end
    print("  Spike Heights  : " .. table.concat(spike_height_vals, " "))
    print("")

    -- Format and print enemy state
    print("--[ Enemy State ]------------------------------------")
    
    -- Pre-format enemy data
    local enemy_types = {}
    local enemy_states = {}
    local enemy_segs = {}
    local enemy_depths = {}
    local enemy_lsbs = {}
    
    for i = 1, 7 do
        enemy_types[i] = decode_enemy_type(enemies_state.enemy_type_info[i])
        enemy_states[i] = decode_enemy_state(enemies_state.active_enemy_info[i])
        enemy_segs[i] = format_segment(enemies_state.enemy_segments[i])
        enemy_depths[i] = string.format(" %02X", enemies_state.enemy_depths[i])
        enemy_lsbs[i] = string.format(" %02X", enemies_state.enemy_depths_lsb[i])
    end

    -- Build enemy state lines
    local enemy_lines = {
        "  Enemy Types    : " .. table.concat(enemy_types, " "),
        "  Enemy States   : " .. table.concat(enemy_states, " "),
        "  Enemy Segments : " .. table.concat(enemy_segs, " "),
        "  Enemy Depths   : " .. table.concat(enemy_depths, " "),
        "  Enemy LSBs     : " .. table.concat(enemy_lsbs, " ")
    }
    print(table.concat(enemy_lines, "\n"))
    print("")

    -- Format charging fuseball flags
    local charging_fuseball = {}
    for i = 1, 16 do
        table.insert(charging_fuseball, enemies_state.charging_fuseball_segments[i] == 1 and "F" or "-")
    end
    print("  Charge Fuseball: " .. table.concat(charging_fuseball, " "))

    -- Format pulsar lanes
    local pulsar_lanes = {}
    for i = 1, 16 do
        table.insert(pulsar_lanes, enemies_state.pulsar_lanes[i] == 1 and "P" or "-")
    end
    print("  Pulsar Lanes   : " .. table.concat(pulsar_lanes, " "))
    print("")

    -- Format shot positions and segments
    local shot_positions = {}
    local shot_segments = {}
    for i = 1, 4 do
        table.insert(shot_positions, string.format("%02X", enemies_state.shot_positions[i] & 0xFF))
        table.insert(shot_segments, format_segment(enemies_state.enemy_shot_segments[i].value))
    end
    print("  Shot Positions : " .. table.concat(shot_positions, " "))
    print("  Shot Segments  : " .. table.concat(shot_segments, " "))
    print("")

    -- Format pending data
    print("  Pending VID   : ")
    local pending_vid_lines = {}
    for i = 1, 64, 16 do
        local line = {}
        for j = 0, 15 do
            table.insert(line, string.format("%02X ", enemies_state.pending_vid[i + j]))
        end
        table.insert(pending_vid_lines, "  " .. table.concat(line))
    end
    print(table.concat(pending_vid_lines, "\n"))
    print(" ")
    print("  Pending SEG   : ")
    -- Removing Pending SEG display as it's less useful than VID and clutters
    -- local pending_seg_lines = {}
    -- for i = 1, 64, 16 do
    --     local line = {}
    --     for j = 0, 15 do
    --         table.insert(line, format_segment(enemies_state.pending_seg[i + j]) .. " ")
    --     end
    --     table.insert(pending_seg_lines, "  " .. table.concat(line))
    -- end
    -- print(table.concat(pending_seg_lines, "\n"))
end

return Display 