--[[
    display.lua
    Handles the on-screen display logic for the Tempest AI MAME script.
--]]

local M = {} -- Module table for export

local display_width = 79 -- Fixed width for all display lines
local first_run = true     -- Flag for initial screen clear

-- Constants (Copied from main.lua\'s original display logic context)
local INVALID_SEGMENT = -32768

-- Helper function to format segment values for display
local function format_segment(value)
    if value == INVALID_SEGMENT then
        return "---"
    else
        -- Use %+03d: sign, pad with 0 to width 2 (total 3 chars like +01, -07)
        return string.format("%+03d", value)
    end
end

-- Function to format a section with a title and key-value pairs in three columns
local function format_three_column_section(title, metrics, base_width)
    local lines = {}
    local kv_format = "%-10s: %-12.12s" -- Key: 10, Val: 12 (truncated), Total: 24 with ": "
    local col_sep = " | "
    local num_cols = 3
    local col_width = 24 -- Must match the total width of kv_format
    -- local content_width_for_cols = num_cols * col_width + (num_cols - 1) * #col_sep -- 3*24 + 2*3 = 72 + 6 = 78

    -- Title line
    local title_padding_val = math.max(0, math.floor((base_width - (#title + 4)) / 2)) -- +4 for "--[ ]"
    local title_line_content = string.format("%s--[ %s ]%s",
        string.rep("-", title_padding_val),
        title,
        string.rep("-", math.max(0, base_width - title_padding_val - #title - 4))
    )
    table.insert(lines, title_line_content)

    local sorted_keys = {}
    for key, _ in pairs(metrics) do table.insert(sorted_keys, key) end
    table.sort(sorted_keys)

    local num_keys = #sorted_keys
    for i = 1, num_keys, num_cols do
        local line_parts = {}
        for j = 0, num_cols - 1 do
            if sorted_keys[i+j] then
                local key_str = sorted_keys[i+j]
                local val_str = tostring(metrics[key_str])
                table.insert(line_parts, string.format(kv_format, key_str, val_str))
            else
                table.insert(line_parts, string.rep(" ", col_width)) -- Empty column placeholder
            end
        end
        
        local row_content = table.concat(line_parts, col_sep)
        table.insert(lines, string.format("%-" .. base_width .. "s", row_content))
    end

    -- Separator line
    table.insert(lines, string.rep("-", base_width))
    return lines
end

-- Main display update function (exported as M.update)
-- Signature matches the call in the original main.lua frame_callback
function M.update(status_message, game_state, level_state, player_state, enemies_state, num_values, last_reward) -- Note: Original didn't pass total_bytes_sent

    if first_run then
        io.write("\027[2J") -- Clear entire screen
        first_run = false
    end
    io.write("\027[1;1H") -- Move cursor to top-left

    local all_display_lines = {}

    -- Helper to add multiple lines from a pre-formatted section
    local function add_section_lines(section_lines)
        for _, line in ipairs(section_lines) do
            table.insert(all_display_lines, line)
        end
    end
    
    -- Helper to add a single custom line, padded to display_width
    local function add_custom_line(content)
        table.insert(all_display_lines, string.format("%-" .. display_width .. "s", content or ""))
    end
    
    local function add_blank_line()
        table.insert(all_display_lines, string.rep(" ", display_width))
    end

    -- Game State Section
    local game_metrics = {
        ["Status"] = status_message,
        ["Gamestate"] = string.format("0x%02X", game_state.gamestate),
        ["Game Mode"] = string.format("0x%02X", game_state.game_mode), -- Simplified, attract/play can be inferred or is often obvious
        ["Countdown"] = string.format("0x%02X", game_state.countdown_timer),
        ["Credits"] = game_state.credits,
        ["P1 Lives"] = game_state.p1_lives,
        ["P1 Level"] = game_state.p1_level,
        ["Frame"] = game_state.frame_counter,
        ["FPS"] = string.format("%.1f", game_state.current_fps),
        ["Data Size"] = string.format("%d v", num_values), -- Shorter "vals"
        ["Lst Reward"] = string.format("%.1f", last_reward), -- Shorter "Last"
        ["Mode Txt"] = (game_state.game_mode < 0x80) and "Attract" or "Play", -- Replaced bit32.band
    }
    add_section_lines(format_three_column_section("Game State", game_metrics, display_width))

    -- Player State Section
    local player_metrics = {
        ["Position"] = string.format("%d(S%d)", player_state.position, (player_state.position % 16)), -- Replaced bit32.band, Compact
        ["State"] = string.format("0x%02X", player_state.player_state),
        ["Depth"] = string.format("0x%02X", player_state.player_depth),
        ["Alive"] = (player_state.alive == 1) and "Yes" or "No",
        ["Score"] = player_state.score,
        ["Zpr Uses"] = player_state.superzapper_uses, -- Shorter
        ["Zpr Active"] = (player_state.superzapper_active ~= 0) and string.format("Y(%d)", player_state.superzapper_active) or "No", -- Compact
        ["Shot Cnt"] = player_state.shot_count, -- Shorter
        ["Debounce"] = string.format("0x%02X", player_state.debounce),
        ["Fire Det"] = player_state.fire_detected, -- Shorter
        ["Zap Det"] = player_state.zap_detected,   -- Shorter
        ["Spin Accum"] = player_state.SpinnerAccum, -- Shorter
        ["Spin Cmd"] = player_state.spinner_commanded,   -- Shorter
        ["Spin Det"] = player_state.spinner_detected,  -- Shorter
    }
    add_section_lines(format_three_column_section("Player State", player_metrics, display_width))

    -- Level State Section
    local level_metrics = {
         ["Lvl Num"] = level_state.level_number, -- Shorter
         ["Lvl Type"] = string.format("0x%02X", level_state.level_type), -- Simpler
         ["Lvl Shape"] = level_state.level_shape, -- Shorter
         ["Type Txt"] = (level_state.level_type == 0xFF) and "Open" or "Closed",
    }
    add_section_lines(format_three_column_section("Level State", level_metrics, display_width))

    -- Enemies State (Counts) Section
    local enemies_counts_metrics = {
        ["Flippers"] = string.format("%d/%d", enemies_state.active_flippers, enemies_state.spawn_slots_flippers),
        ["Pulsars"] = string.format("%d/%d", enemies_state.active_pulsars, enemies_state.spawn_slots_pulsars),
        ["Tankers"] = string.format("%d/%d", enemies_state.active_tankers, enemies_state.spawn_slots_tankers),
        ["Spikers"] = string.format("%d/%d", enemies_state.active_spikers, enemies_state.spawn_slots_spikers),
        ["Fuseballs"] = string.format("%d/%d", enemies_state.active_fuseballs, enemies_state.spawn_slots_fuseballs),
        ["Total Act"] = enemies_state:get_total_active(), -- Shorter
        ["In Tube"] = enemies_state.num_enemies_in_tube,
        ["On Top"] = enemies_state.num_enemies_on_top,
        ["Pending"] = enemies_state.enemies_pending,
    }
    add_section_lines(format_three_column_section("Enemies (Counts)", enemies_counts_metrics, display_width))

    -- Enemy Details (Pulse State, Nearest Target) - Full width
    local enemy_pulse_state_str = string.format("Pulse State: Beat:%02X Pulse:%02X Rate:%02X", enemies_state.pulse_beat, enemies_state.pulsing, enemies_state.pulsar_fliprate)
    local enemy_nearest_target_str = string.format("Nearest Target: Seg:%s Depth:%02X Align:%.0f%% Err:%.0f%%", -- Added % symbols
                                    format_segment(enemies_state.nearest_enemy_seg),
                                    enemies_state.nearest_enemy_depth_raw,
                                    enemies_state.is_aligned_with_nearest * 100,
                                    enemies_state.alignment_error_magnitude * 100)
    do -- Block for Enemy Details title
        local title_text = "Enemy Details"
        local title_padding = math.max(0, math.floor((display_width - (#title_text + 4)) / 2))
        add_custom_line(string.format("%s--[ %s ]%s", string.rep("-", title_padding), title_text, string.rep("-", math.max(0, display_width - title_padding - #title_text - 4))))
    end
    add_custom_line(enemy_pulse_state_str)
    add_custom_line(enemy_nearest_target_str)
    add_custom_line(string.rep("-", display_width)) -- Separator

    -- Player Shots
    local shots_pos_str = ""
    local shots_seg_str = ""
    for i = 1, 8 do -- Assuming player_state.shot_positions and .shot_segments are 1-indexed tables of size 8
        shots_pos_str = shots_pos_str .. string.format(" %02X", player_state.shot_positions[i] or 0)
        shots_seg_str = shots_seg_str .. " " .. format_segment(player_state.shot_segments[i] or INVALID_SEGMENT)
    end
    add_custom_line("Player Shots Pos:" .. shots_pos_str)
    add_custom_line("Player Shots Seg:" .. shots_seg_str)
    add_blank_line()

    -- Spike Heights
    local spike_heights_str = ""
    -- Assuming level_state.spike_heights is 0-indexed or needs adjustment if 1-indexed in Lua
    -- Original loop was for i = 0, 15. If spike_heights is Lua 1-indexed, this should be 1, 16
    -- For now, sticking to original logic, assuming it handles indexing correctly.
    for i = 0, 15 do spike_heights_str = spike_heights_str .. string.format("%02X ", (level_state.spike_heights[i] or level_state.spike_heights[i+1]) or 0) end -- Tentative fix for 0/1 indexing
    add_custom_line("Spike Heights: " .. spike_heights_str)
    add_blank_line()

    -- Enemy Slots Details
    local enemy_details_lines = { "Slot:", "Type:", "State:", "AbsSeg:", "RelSeg:", "Depth:" }
    for i = 1, 7 do -- Assuming 7 enemy slots
        enemy_details_lines[1] = enemy_details_lines[1] .. string.format(" %4d", i)
        local type_val = enemies_state.enemy_type_info and enemies_state.enemy_type_info[i]
        local active_val = enemies_state.active_enemy_info and enemies_state.active_enemy_info[i]
        local abs_seg_val = enemies_state.enemy_abs_segments and enemies_state.enemy_abs_segments[i]
        local seg_val = enemies_state.enemy_segments and enemies_state.enemy_segments[i]
        local depth_val = enemies_state.enemy_depths and enemies_state.enemy_depths[i]

        local type_str = "???"
        if type_val then type_str = enemies_state.decode_enemy_type and enemies_state:decode_enemy_type(type_val) or string.format("%d", type_val) end
        
        local state_str = "???"
        if active_val then state_str = enemies_state.decode_enemy_state and enemies_state:decode_enemy_state(active_val) or string.format("0x%02X", active_val) end
        
        enemy_details_lines[2] = enemy_details_lines[2] .. " " .. string.format("%4s", type_str)
        enemy_details_lines[3] = enemy_details_lines[3] .. " " .. string.format("%4s", state_str)
        enemy_details_lines[4] = enemy_details_lines[4] .. string.format(" %4s", (abs_seg_val and abs_seg_val ~= INVALID_SEGMENT) and abs_seg_val or "---")
        enemy_details_lines[5] = enemy_details_lines[5] .. " " .. string.format("%4s", format_segment(seg_val or INVALID_SEGMENT))
        enemy_details_lines[6] = enemy_details_lines[6] .. string.format("  %02X ", depth_val or 0) -- Adjusted spacing for alignment
    end
    do -- Block for Enemy Slots title
        local title_text = "Enemy Slots"
        local title_padding = math.max(0, math.floor((display_width - (#title_text + 4)) / 2))
        add_custom_line(string.format("%s--[ %s ]%s", string.rep("-", title_padding), title_text, string.rep("-", math.max(0, display_width - title_padding - #title_text - 4))))
    end
    for _, detail_line_content in ipairs(enemy_details_lines) do
        add_custom_line(detail_line_content)
    end
    add_custom_line(string.rep("-", display_width)) -- Separator
    add_blank_line()

    -- Enemy Shots Details
    local e_shots_pos_str = ""
    local e_shots_seg_str = ""
    for i = 1, 4 do -- Assuming 4 enemy shot slots
        e_shots_pos_str = e_shots_pos_str .. string.format(" %02X ", (enemies_state.shot_positions and enemies_state.shot_positions[i]) or 0)
        e_shots_seg_str = e_shots_seg_str .. " " .. format_segment((enemies_state.enemy_shot_segments and enemies_state.enemy_shot_segments[i]) or INVALID_SEGMENT)
    end
    add_custom_line("Enemy Shots Pos:" .. e_shots_pos_str)
    add_custom_line("Enemy Shots Seg:" .. e_shots_seg_str)
    add_blank_line()

    -- Fuseball Lane Depths
    local fuseball_lane_depth_parts = {}
    for i = 1, 16 do -- Assuming 16 lanes
        -- Assuming enemies_state.fuseball_lane_depths is a 1-indexed table
        local depth = (enemies_state.fuseball_lane_depths and enemies_state.fuseball_lane_depths[i]) or 0
        if depth == 0 then
            table.insert(fuseball_lane_depth_parts, "--")
        else
            table.insert(fuseball_lane_depth_parts, string.format("%02X", depth))
        end
    end
    add_custom_line("Fuseball Lanes: " .. table.concat(fuseball_lane_depth_parts, " "))

    -- Charging Fuseball Segments (Enemy-based)
    local charging_fuseball_parts = {}
    for i = 1, 7 do -- 7 enemy slots
        local segment = (enemies_state.charging_fuseball_segments and enemies_state.charging_fuseball_segments[i]) or INVALID_SEGMENT
        if segment == INVALID_SEGMENT then
            table.insert(charging_fuseball_parts, "--")
        else
            table.insert(charging_fuseball_parts, string.format("%02d", segment))
        end
    end
    add_custom_line("Charging FB Segs:" .. string.rep(" ", 1) .. table.concat(charging_fuseball_parts, " "))

    -- Pulsar Segments (Enemy-based) 
    local pulsar_segment_parts = {}
    for i = 1, 7 do -- 7 enemy slots
        local segment = (enemies_state.pulsar_lanes and enemies_state.pulsar_lanes[i]) or INVALID_SEGMENT
        if segment == INVALID_SEGMENT then
            table.insert(pulsar_segment_parts, "--")
        else
            table.insert(pulsar_segment_parts, string.format("%02d", segment))
        end
    end
    add_custom_line("Pulsar Segments :" .. string.rep(" ", 1) .. table.concat(pulsar_segment_parts, " "))

    -- Top Rail Fuseball Segments (Enemy-based)
    local top_rail_fuseball_parts = {}
    for i = 1, 7 do -- 7 enemy slots
        local segment = (enemies_state.top_rail_fuseball_segments and enemies_state.top_rail_fuseball_segments[i]) or INVALID_SEGMENT
        if segment == INVALID_SEGMENT then
            table.insert(top_rail_fuseball_parts, "--")
        else
            table.insert(top_rail_fuseball_parts, string.format("%02d", segment))
        end
    end
    add_custom_line("Top Rail FB Segs:" .. string.rep(" ", 1) .. table.concat(top_rail_fuseball_parts, " "))

    -- Top Rail Other Enemy Segments (Enemy-based)
    local top_rail_other_parts = {}
    for i = 1, 7 do -- 7 enemy slots
        local segment = (enemies_state.top_rail_other_segments and enemies_state.top_rail_other_segments[i]) or INVALID_SEGMENT
        if segment == INVALID_SEGMENT then
            table.insert(top_rail_other_parts, "--")
        else
            table.insert(top_rail_other_parts, string.format("%02d", segment))
        end
    end
    add_custom_line("Top Rail Oth Segs:" .. string.rep(" ", 1) .. table.concat(top_rail_other_parts, " "))

    -- Enemy Shot Lane Depths
    local enemy_shot_lane_depth_parts = {}
    for i = 1, 16 do -- Assuming 16 lanes for enemy shots
        -- Assuming enemies_state.enemy_shot_depths_by_lane is a 1-indexed table
        local depth = (enemies_state.enemy_shot_depths_by_lane and enemies_state.enemy_shot_depths_by_lane[i]) or 0
        if depth == 0 then
            table.insert(enemy_shot_lane_depth_parts, "--")
        else
            table.insert(enemy_shot_lane_depth_parts, string.format("%02X", depth))
        end
    end
    add_custom_line("Enemy Shot Ln : " .. table.concat(enemy_shot_lane_depth_parts, " "))
    add_blank_line()

    -- Pending Data (Show first 16 for brevity)
    local pending_vid_str = ""
    local pending_seg_str = ""
    for i = 1, 16 do
        pending_vid_str = pending_vid_str .. string.format("%02X ", (enemies_state.pending_vid and enemies_state.pending_vid[i]) or 0)
        pending_seg_str = pending_seg_str .. " " .. format_segment((enemies_state.pending_seg and enemies_state.pending_seg[i]) or INVALID_SEGMENT)
    end
    add_custom_line("Pending VID:   " .. pending_vid_str .. "...")
    add_custom_line("Pending SEG:   " .. pending_seg_str .. "...")
    add_blank_line()

    -- Write the entire display string at once
    io.write(table.concat(all_display_lines, "\n"))
    io.write("\027[J") -- Clear from cursor to end of screen
    io.flush() -- Ensure output is written immediately
end

return M