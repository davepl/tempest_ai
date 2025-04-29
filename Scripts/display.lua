--[[
    display.lua
    Handles the on-screen display logic for the Tempest AI MAME script.
--]]

local M = {} -- Module table for export

-- Constants (Copied from main.lua's original display logic context)
local INVALID_SEGMENT = -32768
local is_first_display = true -- Flag to clear screen only on first update

-- Helper function to format segment values for display
local function format_segment(value)
    if value == INVALID_SEGMENT then
        return "---"
    else
        -- Use %+03d: sign, pad with 0 to width 2 (total 3 chars like +01, -07)
        return string.format("%+03d", value)
    end
end

-- Function to format section for display
local function format_section(title, metrics)
    local width = 80 -- Adjusted width for standard terminal
    local title_padding = math.floor((width - #title - 4) / 2)
    local separator = string.rep("-", width)
    local result = string.format("%s--[ %s ]%s\n", string.rep("-", title_padding), title, string.rep("-", width - title_padding - #title - 4))


    -- Find the longest key for alignment
    local max_key_length = 0
    for key, _ in pairs(metrics) do
        max_key_length = math.max(max_key_length, string.len(key))
    end

    -- Format each metric
    local metric_lines = {}
    local sorted_keys = {}
    for key, _ in pairs(metrics) do table.insert(sorted_keys, key) end
    table.sort(sorted_keys) -- Sort keys alphabetically

    for _, key in ipairs(sorted_keys) do
         local value = metrics[key]
         table.insert(metric_lines, string.format("  %-" .. max_key_length .. "s : %s", key, tostring(value)))
    end

    return result .. table.concat(metric_lines, "\n") .. "\n" .. separator .. "\n"
end


-- Function to move the cursor to a specific row (using ANSI escape code)
local function move_cursor_to_row(row)
    io.write(string.format("\027[%d;1H", row)) -- Use 1H for column 1
end

-- Main display update function (exported as M.update)
-- Signature matches the call in the original main.lua frame_callback
function M.update(status_message, game_state, level_state, player_state, enemies_state, num_values, last_reward) -- Note: Original didn't pass total_bytes_sent

    if is_first_display then
        io.write("\027[2J") -- ANSI code to clear screen
        is_first_display = false
    end

    move_cursor_to_row(1) -- Move to top-left corner

    local display_str = ""

    -- Game State Section
    local game_metrics = {
        ["Status"] = status_message,
        ["Gamestate"] = string.format("0x%02X", game_state.gamestate),
        ["Game Mode"] = string.format("0x%02X (%s)", game_state.game_mode, (game_state.game_mode & 0x80 == 0) and "Attract" or "Play"),
        ["Countdown"] = string.format("0x%02X", game_state.countdown_timer),
        ["Credits"] = game_state.credits,
        ["P1 Lives"] = game_state.p1_lives,
        ["P1 Level"] = game_state.p1_level,
        ["Frame"] = game_state.frame_counter,
        ["FPS"] = string.format("%.1f", game_state.current_fps),
        ["Data Size"] = string.format("%d vals", num_values),
        -- ["Bytes Sent"] = total_bytes_sent, -- total_bytes_sent is not passed in the original call
        ["Last Reward"] = string.format("%.1f", last_reward), -- Show reward with decimals
    }
    display_str = display_str .. format_section("Game State", game_metrics)

    -- Player State Section
    local player_metrics = {
        ["Position"] = string.format("%d (Seg %d)", player_state.position, player_state.position & 0x0F),
        ["State"] = string.format("0x%02X", player_state.player_state),
        ["Depth"] = string.format("0x%02X", player_state.player_depth),
        ["Alive"] = (player_state.alive == 1) and "Yes" or "No",
        ["Score"] = player_state.score,
        ["Zapper Uses"] = player_state.superzapper_uses,
        ["Zapper Active"] = (player_state.superzapper_active ~= 0) and string.format("Yes (%d)", player_state.superzapper_active) or "No", -- Show countdown
        ["Shot Count"] = player_state.shot_count,
        ["Debounce"] = string.format("0x%02X", player_state.debounce),
        ["Fire Detect"] = player_state.fire_detected,
        ["Zap Detect"] = player_state.zap_detected,
        ["Spinner Accum"] = player_state.SpinnerAccum,
        ["Spinner Cmd"] = player_state.spinner_commanded,
        ["Spinner Detect"] = player_state.spinner_detected,
    }
    display_str = display_str .. format_section("Player State", player_metrics)

    -- Player Shots
    local shots_pos_str = ""
    local shots_seg_str = ""
    for i = 1, 8 do
        shots_pos_str = shots_pos_str .. string.format(" %02X", player_state.shot_positions[i])
        shots_seg_str = shots_seg_str .. " " .. format_segment(player_state.shot_segments[i])
    end
    display_str = display_str .. "Player Shots Pos:" .. shots_pos_str .. "\n"
    display_str = display_str .. "Player Shots Seg:" .. shots_seg_str .. "\n\n"

    -- Level State Section
    local level_metrics = {
         ["Level Num"] = level_state.level_number,
         ["Level Type"] = string.format("0x%02X (%s)", level_state.level_type, (level_state.level_type == 0xFF) and "Open" or "Closed"),
         ["Level Shape"] = level_state.level_shape,
    }
    display_str = display_str .. format_section("Level State", level_metrics)
    local spike_heights_str = ""
    for i = 0, 15 do spike_heights_str = spike_heights_str .. string.format("%02X ", level_state.spike_heights[i] or 0) end
    display_str = display_str .. "Spike Heights: " .. spike_heights_str .. "\n\n"

    -- Enemies State Section
    local enemies_metrics = {
        ["Flippers"] = string.format("%d/%d", enemies_state.active_flippers, enemies_state.spawn_slots_flippers),
        ["Pulsars"] = string.format("%d/%d", enemies_state.active_pulsars, enemies_state.spawn_slots_pulsars),
        ["Tankers"] = string.format("%d/%d", enemies_state.active_tankers, enemies_state.spawn_slots_tankers),
        ["Spikers"] = string.format("%d/%d", enemies_state.active_spikers, enemies_state.spawn_slots_spikers),
        ["Fuseballs"] = string.format("%d/%d", enemies_state.active_fuseballs, enemies_state.spawn_slots_fuseballs),
        ["Total Active"] = enemies_state:get_total_active(),
        ["In Tube"] = enemies_state.num_enemies_in_tube,
        ["On Top"] = enemies_state.num_enemies_on_top,
        ["Pending"] = enemies_state.enemies_pending,
        ["Pulse State"] = string.format("Beat:%02X Pulse:%02X Rate:%02X", enemies_state.pulse_beat, enemies_state.pulsing, enemies_state.pulsar_fliprate),
        ["Nearest Target"] = string.format("Seg:%s Depth:%02X Align:%.0f Err:%.0f",
                                    format_segment(enemies_state.nearest_enemy_seg),
                                    enemies_state.nearest_enemy_depth_raw,
                                    enemies_state.is_aligned_with_nearest * 100, -- Show as percentage 0 or 100
                                    enemies_state.alignment_error_magnitude * 100), -- Show as percentage
    }
    display_str = display_str .. format_section("Enemies State (Active/Spawnable)", enemies_metrics)

    -- Enemy Slots Details
    local enemy_details = { "Slot:", "Type:", "State:", "AbsSeg:", "RelSeg:", "Depth:" }
    for i = 1, 7 do
        enemy_details[1] = enemy_details[1] .. string.format(" %4d", i)
        -- Use decode methods if they exist on the enemies_state object
        local type_str = enemies_state.decode_enemy_type and enemies_state:decode_enemy_type(enemies_state.enemy_type_info[i]) or string.format("%d", enemies_state.enemy_type_info[i])
        local state_str = enemies_state.decode_enemy_state and enemies_state:decode_enemy_state(enemies_state.active_enemy_info[i]) or string.format("0x%02X", enemies_state.active_enemy_info[i])
        enemy_details[2] = enemy_details[2] .. " " .. string.format("%4s", type_str)
        enemy_details[3] = enemy_details[3] .. " " .. string.format("%4s", state_str)
        enemy_details[4] = enemy_details[4] .. string.format(" %4s", (enemies_state.enemy_abs_segments[i] ~= INVALID_SEGMENT) and enemies_state.enemy_abs_segments[i] or "---")
        enemy_details[5] = enemy_details[5] .. " " .. string.format("%4s", format_segment(enemies_state.enemy_segments[i]))
        enemy_details[6] = enemy_details[6] .. string.format(" %02X  ", enemies_state.enemy_depths[i]) -- Use 4 chars width
    end
    display_str = display_str .. table.concat(enemy_details, "\n") .. "\n\n"

    -- Enemy Shots Details
    local e_shots_pos_str = ""
    local e_shots_seg_str = ""
    for i = 1, 4 do
        e_shots_pos_str = e_shots_pos_str .. string.format(" %02X ", enemies_state.shot_positions[i])
        e_shots_seg_str = e_shots_seg_str .. " " .. format_segment(enemies_state.enemy_shot_segments[i])
    end
    display_str = display_str .. "Enemy Shots Pos:" .. e_shots_pos_str .. "\n"
    display_str = display_str .. "Enemy Shots Seg:" .. e_shots_seg_str .. "\n\n"

    -- More Enemy Info Details
    local more_info_str = ""
    for i = 1, 7 do
        local value = enemies_state.more_enemy_info[i]
        if (value & 0x80) ~= 0 then -- Check top bit
            -- Valid: show lower 7 bits as 2-digit hex
            more_info_str = more_info_str .. string.format(" %02X", value & 0x7F)
        else
            -- Invalid: show --
            more_info_str = more_info_str .. " --"
        end
    end
    display_str = display_str .. "More Enemy Info:" .. more_info_str .. "\n\n"

    -- Charging Fuseballs
    local charging_fuseball_str = {}
    for i = 1, 16 do table.insert(charging_fuseball_str, enemies_state.charging_fuseball_segments[i] == 1 and "*" or "-") end
    display_str = display_str .. "Fuseball Chrg: " .. table.concat(charging_fuseball_str, " ") .. "\n\n"

    -- Pending Data (Show first 16 for brevity)
    local pending_vid_str = ""
    local pending_seg_str = ""
    for i = 1, 16 do
        pending_vid_str = pending_vid_str .. string.format("%02X ", enemies_state.pending_vid[i])
        pending_seg_str = pending_seg_str .. " " .. format_segment(enemies_state.pending_seg[i])
    end
    display_str = display_str .. "Pending VID:   " .. pending_vid_str .. "...\n"
    display_str = display_str .. "Pending SEG:   " .. pending_seg_str .. "...\n"

    -- Add padding to overwrite previous longer lines at the end
    display_str = display_str .. string.rep(" ", 80 * 5) -- Add blank lines to clear potential leftover lines

    -- Write the entire display string at once
    io.write(display_str)
    io.flush() -- Ensure output is written immediately
end

return M 