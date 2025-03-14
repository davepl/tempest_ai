--[[
    Update Display Module for Tempest AI
    Handles all console display functionality 
    Author: Dave Plummer (davepl) and AI assists
    Date: 2025-03-06
]]

local update = {}

-- Terminal control functions
function update.clear_screen()
    io.write("\027[2J\027[H")
end

function update.move_cursor_home()
    io.write("\027[H")
end

function update.move_cursor_to_row(row)
    io.write(string.format("\027[%d;0H", row))
end

-- Format a section with title and metrics
function update.format_section(title, metrics)
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
    
    return result .. "\n"
end

-- Main display update function
function update.update_display(status, game_state, level_state, player_state, enemies_state, current_action, num_values, reward)
    update.clear_screen()
    update.move_cursor_to_row(1)

    -- Format and print game state in 3 columns at row 1
    
    -- Create game metrics in a more organized way for 3-column display
    local game_metrics = {
        {"Gamestate", string.format("0x%02X", game_state.gamestate)},
        {"Game Mode", string.format("0x%02X", game_state.game_mode)},
        {"Countdown", string.format("0x%02X", game_state.countdown_timer)},
        {"Credits", game_state.credits},
        {"P1 Lives", game_state.p1_lives},
        {"P1 Level", game_state.p1_level},
        {"Frame", game_state.frame_counter},
        {"Bytes Sent", total_bytes_sent},
        {"FPS", string.format("%.2f", current_fps)},
        {"Payload Size", num_values},
        {"Last Reward", string.format("%.2f", LastRewardState)}  -- Add last reward to game metrics
    }
    
    -- Print game metrics in 3 columns
    print("--[ Game State ]--------------------------------------")
    local col_width = 35  -- Width for each column
    
    -- Calculate how many rows we need (ceiling of items/3)
    local rows = math.ceil(#game_metrics / 3)
    
    for row = 1, rows do
        local line = "  "
        for col = 1, 3 do
            local idx = (row - 1) * 3 + col
            if idx <= #game_metrics then
                local key, value = table.unpack(game_metrics[idx])
                -- Format each metric with fixed width
                line = line .. string.format("%-12s: %s ", key, value)
            end
        end
        print(line)
    end
    print("")  -- Empty line after section

    -- Format and print player state in 3 columns at row 6
    update.move_cursor_to_row(7)
    
    -- Create player metrics in a more organized way for 3-column display
    local player_metrics = {
        {"Position", player_state.position .. " "},
        {"State", string.format("0x%02X", player_state.player_state) .. " "},
        {"Depth", player_state.player_depth .. " "},
        {"Alive", player_state.alive .. " "},
        {"Score", player_state.score .. " "},
        {"Szapper Uses", player_state.superzapper_uses .. " "},
        {"Szapper Active", player_state.superzapper_active .. " "},
        {"Shot Count", player_state.shot_count .. " "},
        {"Debounce", player_state.debounce .. " "},
        {"Fire Detected", player_state.fire_detected .. " "},
        {"Zap Detected", player_state.zap_detected .. " "},
        {"SpinnerAccum", player_state.SpinnerAccum .. " "},
        {"SpinnerDelta", player_state.SpinnerDelta .. " "},
        {"InferredDelta", player_state.inferredSpinnerDelta .. " "},
        {"ZapFireNew", player_state.ZapFireNew .. " "},
        {"ZapFireStarts", player_state.zap_fire_starts .. " "}
    }
    
    -- Print player metrics in 3 columns
    print("--[ Player State ]------------------------------------")
    
    -- Calculate how many rows we need (ceiling of items/3)
    local rows = math.ceil(#player_metrics / 3)
    
    for row = 1, rows do
        local line = "  "
        for col = 1, 3 do
            local idx = (row - 1) * 3 + col
            if idx <= #player_metrics then
                local key, value = table.unpack(player_metrics[idx])
                -- Format each metric with fixed width
                line = line .. string.format("%-12s: %s", key, value)
            end
        end
        print(line)
    end
    
    -- Add shot positions on its own line
    local shots_str = ""
    for i = 1, 8 do
        local segment_str = string.format("%+02d", player_state.shot_segments[i])
        shots_str = shots_str .. string.format("%02X-%s ", player_state.shot_positions[i], segment_str)
    end
    print("  Shot Positions: " .. shots_str)
    print("")  -- Empty line after section

    -- Format and print player controls at row 14
    update.move_cursor_to_row(16)
    local controls_metrics = {
        ["Fire"] = controls.fire_commanded,
        ["Superzapper"] = controls.zap_commanded,
        ["Left"] = controls.left_commanded,
        ["Right"] = controls.right_commanded,
        ["Current Action"] = current_action or "none"
    }
    print(update.format_section("Player Controls", controls_metrics))

    -- Format and print level state in 3 columns
    update.move_cursor_to_row(23)
    
    -- Print level metrics in 3 columns
    print("--[ Level State ]-------------------------------------")

    -- Create level metrics in a more organized way for 3-column display
    local level_metrics_list = {
        {"Level Number", level_state.level_number},
        {"Level Type", level_state.level_type == 0xFF and "Open" or "Closed"},
        {"Level Shape", level_state.level_shape}
    }
    
    -- Calculate how many rows we need (ceiling of items/3)
    local rows = math.ceil(#level_metrics_list / 3)
    
    for row = 1, rows do
        local line = "  "
        for col = 1, 3 do
            local idx = (row - 1) * 3 + col
            if idx <= #level_metrics_list then
                local key, value = table.unpack(level_metrics_list[idx])
                -- Format each metric with fixed width
                line = line .. string.format("%-12s: %s", key, value)
            end
        end
        print(line)
    end
    
    -- Add spike heights on its own line
    local heights_str = ""
    -- For open levels, show 0-15, for closed levels show -8 to +7 but just display the values
    for i = -8, 7 do
        if level_state.spike_heights[i] then
            heights_str = heights_str .. string.format("%02X ", level_state.spike_heights[i])
        else
            heights_str = heights_str .. "-- "
        end
    end
    print("  Spike Heights: " .. heights_str)
    
    -- Add level angles on its own line
    local angles_str = ""
    for i = 0, 15 do
        angles_str = angles_str .. string.format("%02X ", level_state.level_angles[i])
    end
    print("  Level Angles : " .. angles_str)
    print("")  -- Empty line after section

    -- Format and print enemies state at row 31
    update.move_cursor_to_row(28)
    local enemy_types = {}
    local enemy_states = {}
    local enemy_segs = {}
    local enemy_depths = {}
    for i = 1, 7 do
        enemy_types[i] = enemies_state:decode_enemy_type(enemies_state.enemy_type_info[i])
        enemy_states[i] = enemies_state:decode_enemy_state(enemies_state.active_enemy_info[i])
        enemy_segs[i] = string.format("%+3d", enemies_state.enemy_segments[i])
        enemy_depths[i] = string.format("%02X.%02X", enemies_state.enemy_depths[i].pos, enemies_state.enemy_depths[i].frac)
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
        ["Enemy Types"] = table.concat(enemy_types, " "),
        ["Enemy States"] = table.concat(enemy_states, " ")
    }

    print(update.format_section("Enemies State", enemies_metrics))

    -- Add enemy segments and depths on their own lines
    print("  Enemy Segments: " .. table.concat(enemy_segs, " "))
    print("  Enemy Depths  : " .. table.concat(enemy_depths, " "))
    
    -- Add enemy shot positions on its own line
    local shots_str = "none"
    local shots = {}
    for i = 1, 4 do
        if enemies_state.shot_positions[i] then
            table.insert(shots, string.format("%+d", enemies_state.shot_positions[i]))
        end
    end
    if #shots > 0 then
        shots_str = table.concat(shots, " ")
    end
    print("  Shot Positions: " .. shots_str)
    print("")  -- Empty line after section
end

return update 