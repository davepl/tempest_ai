--[[
    display.lua
    Handles the on-screen display logic for the Tempest AI MAME script.
--]]

local M = {} -- Module table for export

-- Constants
local INVALID_SEGMENT = -32768
local is_first_display = true -- Flag to clear screen only on first update

-- Fixed column configuration (3 columns)
-- Widths: Label + " : " + Value = 12 + 3 + 10 = 25 per column
-- Total: 25 * 3 + 2 * 2 (padding) = 75 + 4 = 79 chars
local COL_CONFIG_3 = {
    { label = 12, value = 10 }, -- Col 1
    { label = 12, value = 10 }, -- Col 2
    { label = 12, value = 10 }  -- Col 3
}

-- Helper function to format segment values for display
local function format_segment(value)
    if value == INVALID_SEGMENT then
        return "---"
    else
        -- Use %+03d: sign, pad with 0 to width 2 (total 3 chars like +01, -07)
        return string.format("%+03d", value)
    end
end

-- NEW Function to format section with multiple columns and fixed widths
local function format_multi_column_section(title, metrics_list, num_cols, col_config)
    local width = 80 -- Adjusted width for standard terminal
    local title_padding = math.floor((width - #title - 4) / 2)
    local separator = string.rep("-", width)
    local result = string.format("%s--[ %s ]%s\\n", string.rep("-", title_padding), title, string.rep("-", width - title_padding - #title - 4))

    -- Pre-calculate column format strings and total width
    local col_formats = {}
    local total_line_width = 0
    for i = 1, num_cols do
        local cfg = col_config[i]
        -- Format: Left-aligned label (width cfg.label), Right-aligned value (width cfg.value)
        col_formats[i] = string.format("%%-%ds : %%ds", cfg.label, cfg.value)
        total_line_width = total_line_width + cfg.label + 3 + cfg.value -- label + " : " + value
    end
    total_line_width = total_line_width + (num_cols - 1) * 2 -- Add "  " padding between columns

    local num_metrics = #metrics_list
    local num_rows = math.ceil(num_metrics / num_cols)

    for r = 1, num_rows do
        local line_parts = {}
        for c = 1, num_cols do
            local metric_index = (r - 1) * num_cols + c
            local cfg = col_config[c]
            local col_width = cfg.label + 3 + cfg.value
            if metric_index <= num_metrics then
                local metric = metrics_list[metric_index]
                -- Format the key/value pair using the pre-calculated format string for this column
                -- Ensure value is converted to string
                local formatted_metric = string.format(col_formats[c], metric.key or "", tostring(metric.value or ""))
                -- Truncate if necessary (shouldn't happen with planned widths, but safety)
                formatted_metric = string.sub(formatted_metric, 1, col_width)
                table.insert(line_parts, formatted_metric)
            else
                -- Add padding if last row is not full for this column
                table.insert(line_parts, string.rep(" ", col_width))
            end
        end
        -- Join columns with "  " padding and add to result
        result = result .. table.concat(line_parts, "  ") .. "\\n"
    end

    -- Add padding line to ensure previous content is overwritten if needed (though fixed widths should handle it)
    -- result = result .. string.rep(" ", 80) .. "\\n"

    return result .. separator .. "\\n"
end

-- Function to move the cursor to a specific row (using ANSI escape code)
local function move_cursor_to_row(row)
    io.write(string.format("\\027[%d;1H", row)) -- Use 1H for column 1
end

-- Main display update function (exported as M.update)
-- Signature matches the call in the original main.lua frame_callback
function M.update(status_message, game_state, level_state, player_state, enemies_state, num_values, last_reward) -- Note: Original didn't pass total_bytes_sent

    if is_first_display then
        io.write("\\027[2J") -- ANSI code to clear screen
        is_first_display = false
    end

    move_cursor_to_row(1) -- Move to top-left corner

    local display_str = ""

    -- Game State Section (List format for multi-column)
    local game_metrics_list = {
        { key="Status", value=status_message },
        { key="Gamestate", value=string.format("0x%02X", game_state.gamestate) },
        { key="Game Mode", value=string.format("0x%02X", game_state.game_mode) }, -- Shortened
        { key="P1 Lives", value=game_state.p1_lives },
        { key="P1 Level", value=game_state.p1_level },
        { key="Frame", value=game_state.frame_counter },
        { key="Credits", value=game_state.credits },
        { key="Countdown", value=string.format("0x%02X", game_state.countdown_timer) },
        { key="FPS", value=string.format("%.1f", game_state.current_fps) },
        { key="Last Reward", value=string.format("%.1f", last_reward) },
        { key="Data Size", value=string.format("%d", num_values) }, -- Simplified
        { key="Game Mode D", value=((game_state.game_mode & 0x80 == 0) and "Attract" or "Play") }, -- Mode Description
    }
    display_str = display_str .. format_multi_column_section("Game State", game_metrics_list, 3, COL_CONFIG_3)

    -- Player State Section (List format for multi-column)
    local player_metrics_list = {
        { key="Position", value=string.format("%d", player_state.position) }, -- Raw Pos
        { key="Abs Segment", value=string.format("%d", player_state.position & 0x0F) },
        { key="State", value=string.format("0x%02X", player_state.player_state) },
        { key="Depth", value=string.format("0x%02X", player_state.player_depth) },
        { key="Alive", value=(player_state.alive == 1) and "Yes" or "No" },
        { key="Score", value=player_state.score },
        { key="Zapper Uses", value=player_state.superzapper_uses },
        { key="Zapper Actv", value=(player_state.superzapper_active ~= 0) and string.format("%d", player_state.superzapper_active) or "No" }, -- Simplified Active
        { key="Shot Count", value=player_state.shot_count },
        { key="Fire Cmd", value=player_state.fire_commanded }, -- Commanded Fire
        { key="Zap Cmd", value=player_state.zap_commanded },   -- Commanded Zap
        { key="Spinner Cmd", value=player_state.spinner_commanded },
        { key="Fire Detect", value=player_state.fire_detected },
        { key="Zap Detect", value=player_state.zap_detected },
        { key="Spinner Acc", value=player_state.SpinnerAccum },
        { key="Spinner Del", value=player_state.spinner_detected }, -- Spinner Delta
        { key="Debounce", value=string.format("0x%02X", player_state.debounce) },
    }
    display_str = display_str .. format_multi_column_section("Player State", player_metrics_list, 3, COL_CONFIG_3)

    -- Player Shots (Keep as is for now)
    local shots_pos_str = ""
    local shots_seg_str = ""
    for i = 1, 8 do
        shots_pos_str = shots_pos_str .. string.format(" %02X", player_state.shot_positions[i])
        shots_seg_str = shots_seg_str .. " " .. format_segment(player_state.shot_segments[i])
    end
    display_str = display_str .. string.format("%-16s:%s\\n", "Player Shots Pos", shots_pos_str)
    display_str = display_str .. string.format("%-16s:%s\\n\\n", "Player Shots Seg", shots_seg_str)

    -- Level State Section (Keep single column for now or make 2-col?)
    -- Let's keep it simpler for now.
    local level_metrics_list = {
        { key="Level Num", value=level_state.level_number },
        { key="Level Type", value=string.format("0x%02X (%s)", level_state.level_type, (level_state.level_type == 0xFF) and "Open" or "Closed") },
        { key="Level Shape", value=level_state.level_shape },
    }
    -- Use multi-column with 1 column to reuse formatting logic easily
    display_str = display_str .. format_multi_column_section("Level State", level_metrics_list, 1, {{label=16, value=20}}) -- Adjusted config for single wide column

    local spike_heights_str = ""
    for i = 0, 15 do spike_heights_str = spike_heights_str .. string.format("%02X ", level_state.spike_heights[i] or 0) end
    display_str = display_str .. string.format("%-16s: %s\\n\\n", "Spike Heights", spike_heights_str)

    -- Enemies State Section (List format for multi-column)
    local enemies_metrics_list = {
        { key="Flippers", value=string.format("%d/%d", enemies_state.active_flippers, enemies_state.spawn_slots_flippers) },
        { key="Pulsars", value=string.format("%d/%d", enemies_state.active_pulsars, enemies_state.spawn_slots_pulsars) },
        { key="Tankers", value=string.format("%d/%d", enemies_state.active_tankers, enemies_state.spawn_slots_tankers) },
        { key="Spikers", value=string.format("%d/%d", enemies_state.active_spikers, enemies_state.spawn_slots_spikers) },
        { key="Fuseballs", value=string.format("%d/%d", enemies_state.active_fuseballs, enemies_state.spawn_slots_fuseballs) },
        { key="Total Active", value=enemies_state:get_total_active() },
        { key="In Tube", value=enemies_state.num_enemies_in_tube },
        { key="On Top", value=enemies_state.num_enemies_on_top },
        { key="Pending", value=enemies_state.enemies_pending },
        { key="Pulse Beat", value=string.format("0x%02X", enemies_state.pulse_beat) },
        { key="Pulsing", value=string.format("0x%02X", enemies_state.pulsing) },
        { key="Pulse Rate", value=string.format("0x%02X", enemies_state.pulsar_fliprate) },
        { key="Near Seg", value=format_segment(enemies_state.nearest_enemy_seg) },
        { key="Near Depth", value=string.format("0x%02X", enemies_state.nearest_enemy_depth_raw) },
        { key="Aligned", value=string.format("%.0f%%", enemies_state.is_aligned_with_nearest * 100) },
        { key="Align Error", value=string.format("%.0f%%", enemies_state.alignment_error_magnitude * 100) },
    }
    display_str = display_str .. format_multi_column_section("Enemies State (Active/Spawn)", enemies_metrics_list, 3, COL_CONFIG_3)

    -- Enemy Slots Details (Keep multi-line format for clarity)
    local enemy_details_title = "--[ Enemy Slots (1-7) ]"
    local title_pad_len = math.floor((80 - #enemy_details_title) / 2)
    display_str = display_str .. string.rep("-", title_pad_len) .. enemy_details_title .. string.rep("-", 80 - title_pad_len - #enemy_details_title) .. "\\n"
    local enemy_details = { "Slot: ", "Type: ", "State:", "AbsSeg:", "RelSeg:", "Depth:" }
    local field_width = 6 -- Adjusted width for enemy details columns
    for i = 1, 7 do
        enemy_details[1] = enemy_details[1] .. string.format(" %" .. field_width .. "d", i)
        local type_str = enemies_state.decode_enemy_type and enemies_state:decode_enemy_type(enemies_state.enemy_type_info[i]) or string.format("%d", enemies_state.enemy_type_info[i])
        local state_str = enemies_state.decode_enemy_state and enemies_state:decode_enemy_state(enemies_state.active_enemy_info[i]) or string.format("0x%02X", enemies_state.active_enemy_info[i])
        enemy_details[2] = enemy_details[2] .. " " .. string.format("%" .. field_width .. "s", type_str)
        enemy_details[3] = enemy_details[3] .. " " .. string.format("%" .. field_width .. "s", state_str)
        enemy_details[4] = enemy_details[4] .. " " .. string.format("%" .. field_width .. "s", (enemies_state.enemy_abs_segments[i] ~= INVALID_SEGMENT) and enemies_state.enemy_abs_segments[i] or "---")
        enemy_details[5] = enemy_details[5] .. " " .. string.format("%" .. field_width .. "s", format_segment(enemies_state.enemy_segments[i]))
        enemy_details[6] = enemy_details[6] .. " " .. string.format("0x%02X", enemies_state.enemy_depths[i]) -- Keep hex depth, adjust spacing if needed
        -- Pad depth to match field_width approx
        enemy_details[6] = enemy_details[6] .. string.rep(" ", field_width - 4)
    end
    display_str = display_str .. table.concat(enemy_details, "\\n") .. "\\n" .. string.rep("-", 80) .. "\\n" -- Separator after

    -- Enemy Shots Details (Keep as is)
    local e_shots_pos_str = ""
    local e_shots_seg_str = ""
    for i = 1, 4 do
        e_shots_pos_str = e_shots_pos_str .. string.format(" %02X ", enemies_state.shot_positions[i])
        e_shots_seg_str = e_shots_seg_str .. " " .. format_segment(enemies_state.enemy_shot_segments[i])
    end
    display_str = display_str .. string.format("%-16s:%s\\n", "Enemy Shots Pos", e_shots_pos_str)
    display_str = display_str .. string.format("%-16s:%s\\n\\n", "Enemy Shots Seg", e_shots_seg_str)

    -- More Enemy Info Details (Keep as is)
    local more_info_str = ""
    for i = 1, 7 do
        local value = enemies_state.more_enemy_info[i]
        if (value & 0x80) ~= 0 then -- Check top bit
            more_info_str = more_info_str .. string.format(" %02X", value & 0x7F)
        else
            more_info_str = more_info_str .. " --"
        end
    end
    display_str = display_str .. string.format("%-16s:%s\\n\\n", "More Enemy Info", more_info_str)

    -- Charging Fuseballs (Keep as is)
    local charging_fuseball_str = {}
    for i = 1, 16 do table.insert(charging_fuseball_str, enemies_state.charging_fuseball_segments[i] == 1 and "*" or "-") end
    display_str = display_str .. string.format("%-16s: %s\\n\\n", "Fuseball Chrg", table.concat(charging_fuseball_str, " "))

    -- Pending Data (Keep as is, maybe shorten label)
    local pending_vid_str = ""
    local pending_seg_str = ""
    for i = 1, 16 do
        pending_vid_str = pending_vid_str .. string.format("%02X ", enemies_state.pending_vid[i])
        pending_seg_str = pending_seg_str .. " " .. format_segment(enemies_state.pending_seg[i])
    end
    display_str = display_str .. string.format("%-16s: %s...\\n", "Pending VID", pending_vid_str)
    display_str = display_str .. string.format("%-16s: %s...\\n", "Pending SEG", pending_seg_str)

    -- Add padding to overwrite previous longer lines at the end
    display_str = display_str .. string.rep(" ", 80 * 3) -- Add blank lines to clear potential leftover lines

    -- Write the entire display string at once
    io.write(display_str)
    io.flush() -- Ensure output is written immediately
end

return M 