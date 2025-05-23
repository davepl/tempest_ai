--[[
    display.lua
    Handles the on-screen display logic for the Tempest AI MAME script.
--]]

local M = {} -- Module table for export

-- Helper functions for bitwise operations
local function band(a, b) -- bitwise AND (copied from state.lua for now)
    local result = 0
    local bit_val = 1
    while a > 0 and b > 0 do
        if a % 2 == 1 and b % 2 == 1 then
            result = result + bit_val
        end
        a = math.floor(a / 2)
        b = math.floor(b / 2)
        bit_val = bit_val * 2
    end
    return result
end

local function rshift(num, bits) -- bitwise RIGHT SHIFT
    return math.floor(num / (2^bits))
end

-- Constants (Copied from main.lua's original display logic context)
local INVALID_SEGMENT = -32768
local SCREEN_WIDTH = 80
local MAX_COLS = 3

-- Helper function to format segment values for display
local function format_segment(value)
    if value == INVALID_SEGMENT then
        return "---"
    else
        -- Use %+03d: sign, pad with 0 to width 2 (total 3 chars like +01, -07)
        return string.format("%+03d", value)
    end
end

-- Helper function to format a fixed-width segment value for our enemy tables
local function format_enemy_segment(value)
    if value == 0 then
        return "---"
    else
        -- Use %+03d: always show sign, pad with 0 to total width 3
        return string.format("%+03d", value)
    end
end

-- Function to create a section header (centered title with borders)
local function format_header(title)
    local title_padding = math.floor((SCREEN_WIDTH - #title - 4) / 2)
    return string.format("%s--[ %s ]%s", 
        string.rep("-", title_padding), 
        title, 
        string.rep("-", SCREEN_WIDTH - title_padding - #title - 4)
    )
end

-- Function to create a compact 3-column display
local function format_columns(items, col_width)
    local result = {}
    local item_count = #items
    local rows_needed = math.ceil(item_count / MAX_COLS)
    
    for row = 1, rows_needed do
        local line = ""
        for col = 1, MAX_COLS do
            local idx = (row - 1) * MAX_COLS + col
            if idx <= item_count then
                -- Format as "key: value", right-padded to column width
                line = line .. string.format("%-" .. col_width .. "s", items[idx])
            end
        end
        table.insert(result, line)
    end
    
    return result
end


-- Function to clear the screen and move cursor to top-left (ANSI escape code)
local function clear_screen()
    io.write("\027[2J\027[1;1H") -- Clear screen and move to top-left
    io.flush()
end

-- Format a key-value pair for display with consistent width
local function format_item(key, value, width)
    return string.format("%s: %-" .. (width - #key - 2) .. "s", key, tostring(value))
end

-- Main display update function (exported as M.update)
-- Signature matches the call in the original main.lua frame_callback
function M.update(status_message, game_state, level_state, player_state, enemies_state, num_values, last_reward)
    -- Clear the screen each time to ensure a fresh display
    clear_screen()
    
    local display_lines = {}
    local col_width = math.floor(SCREEN_WIDTH / MAX_COLS)
    
    -- Create compact items for Game State section
    table.insert(display_lines, format_header("Game State"))
    
    local game_items = {
        format_item("Status", status_message, col_width),
        format_item("GState", string.format("0x%02X", game_state.gamestate), col_width),
        format_item("Mode", (band(game_state.game_mode, 0x80) == 0) and "Attract" or "Play", col_width),
        format_item("Lives", game_state.p1_lives, col_width),
        format_item("Level", game_state.p1_level, col_width),
        format_item("Frame", game_state.frame_counter, col_width),
        format_item("FPS", string.format("%.1f", game_state.current_fps), col_width),
        format_item("Values", num_values, col_width),
        format_item("Reward", string.format("%.1f", last_reward), col_width)
    }
    
    -- Add the game state items in a grid layout
    local game_lines = format_columns(game_items, col_width)
    for _, line in ipairs(game_lines) do
        table.insert(display_lines, line)
    end
    table.insert(display_lines, "")

    -- Player State Section
    table.insert(display_lines, format_header("Player State"))
    
    local player_items = {
        format_item("Pos", string.format("%d (Seg %d)", player_state.position, band(player_state.position, 0x0F)), col_width),
        format_item("State", string.format("0x%02X", player_state.player_state), col_width),
        format_item("Depth", string.format("0x%02X", player_state.player_depth), col_width),
        format_item("Alive", (player_state.alive == 1) and "Yes" or "No", col_width),
        format_item("Score", player_state.score, col_width),
        format_item("Z-Uses", player_state.superzapper_uses, col_width),
        format_item("Z-Active", (player_state.superzapper_active ~= 0) and "Yes" or "No", col_width),
        format_item("Shots", player_state.shot_count, col_width),
        format_item("Fire", player_state.fire_detected and "Yes" or "No", col_width),
        format_item("Zap", player_state.zap_detected and "Yes" or "No", col_width),
        format_item("Spinner", player_state.spinner_detected, col_width),
        format_item("SpinCmd", player_state.spinner_commanded, col_width)
    }
    
    local player_lines = format_columns(player_items, col_width)
    for _, line in ipairs(player_lines) do
        table.insert(display_lines, line)
    end
    
    -- Player Shots (Compact single line format)
    local shots_str = "Shots: "
    for i = 1, 8 do
        if player_state.shot_segments[i] ~= INVALID_SEGMENT then
            shots_str = shots_str .. format_segment(player_state.shot_segments[i]) .. " "
        end
    end
    table.insert(display_lines, shots_str)
    table.insert(display_lines, "")
    
    -- Level State Section
    table.insert(display_lines, format_header("Level State"))
    
    local level_items = {
        format_item("Level", level_state.level_number, col_width),
        format_item("Type", (level_state.level_type == 0xFF) and "Open" or "Closed", col_width),
        format_item("Shape", level_state.level_shape, col_width)
    }
    
    local level_lines = format_columns(level_items, col_width)
    for _, line in ipairs(level_lines) do
        table.insert(display_lines, line)
    end
    
    -- Spikes summary (compact format)
    local spike_str = "Spikes: "
    for i = 0, 15 do 
        if i % 8 == 0 and i > 0 then spike_str = spike_str .. " | " end
        spike_str = spike_str .. string.format("%01X", rshift(level_state.spike_heights[i] or 0, 4))
    end
    table.insert(display_lines, spike_str)
    table.insert(display_lines, "")
    -- Enemies State Section
    table.insert(display_lines, format_header("Enemy State"))
    
    local enemies_items = {
        format_item("Flippers", string.format("%d/%d", enemies_state.active_flippers, enemies_state.spawn_slots_flippers), col_width),
        format_item("Pulsars", string.format("%d/%d", enemies_state.active_pulsars, enemies_state.spawn_slots_pulsars), col_width),
        format_item("Tankers", string.format("%d/%d", enemies_state.active_tankers, enemies_state.spawn_slots_tankers), col_width),
        format_item("Spikers", string.format("%d/%d", enemies_state.active_spikers, enemies_state.spawn_slots_spikers), col_width),
        format_item("Fuseballs", string.format("%d/%d", enemies_state.active_fuseballs, enemies_state.spawn_slots_fuseballs), col_width),
        format_item("Total", enemies_state:get_total_active(), col_width),
        format_item("In Tube", enemies_state.num_enemies_in_tube, col_width),
        format_item("On Top", enemies_state.num_enemies_on_top, col_width),
        format_item("Pending", enemies_state.enemies_pending, col_width),
        format_item("Pulsing", enemies_state.pulsing > 0 and "Yes" or "No", col_width),
        format_item("Beat", string.format("0x%02X", enemies_state.pulse_beat), col_width),
        format_item("Rate", string.format("0x%02X", enemies_state.pulsar_fliprate), col_width)
    }
    
    local enemies_lines = format_columns(enemies_items, col_width)
    for _, line in ipairs(enemies_lines) do
        table.insert(display_lines, line)
    end
    
    -- Nearest enemy info
    local nearest = string.format("Nearest: Seg=%s Depth=0x%02X Aligned=%s Error=%.0f%%",
                                format_segment(enemies_state.nearest_enemy_seg),
                                enemies_state.nearest_enemy_depth_raw,
                                enemies_state.is_aligned_with_nearest > 0 and "Yes" or "No",
                                enemies_state.alignment_error_magnitude * 100)
    table.insert(display_lines, nearest)
    table.insert(display_lines, "")

    -- Enemy Slots Table (compact format)
    table.insert(display_lines, format_header("Enemy Slots"))
    
    -- Create a compact header for the enemy slots table
    table.insert(display_lines, "  # | Type | State | Seg      | Depth | FracP ") -- Adjusted header for 8-char Seg
    table.insert(display_lines, "----+------+-------+----------+-------+-------") -- Adjusted separator for 8-char Seg
    
    -- Display each enemy slot in a compact tabular format
    for i = 1, 7 do
        local type_str = enemies_state.decode_enemy_type and 
                         enemies_state:decode_enemy_type(enemies_state.enemy_type_info[i]) or 
                         string.format("%d", enemies_state.enemy_type_info[i])
        
        local state_str = enemies_state.decode_enemy_state and 
                         enemies_state:decode_enemy_state(enemies_state.active_enemy_info[i]) or 
                         string.format("0x%02X", enemies_state.active_enemy_info[i])
        
        local rel_seg_val = enemies_state.enemy_segments[i]
        local frac_progress = enemies_state.enemy_fractional_progress[i]
        local is_between = enemies_state.enemy_between_segments[i]
        local dir_moving = enemies_state.enemy_direction_moving[i]
        
        local display_seg_str -- For "Seg" column
        if rel_seg_val == INVALID_SEGMENT then
            display_seg_str = " --.--- " -- 8 chars for invalid
        else
            local effective_display_val = rel_seg_val + 0.0 -- Ensure float type
            local tolerance = 0.0001 
            -- Only apply fractional part if enemy is marked as 'between_segments'
            -- and frac_progress is not essentially 0 or 1.
            if is_between == 1 and frac_progress > tolerance and frac_progress < (1.0 - tolerance) then
                if dir_moving == 1 then -- Moving towards more positive relative segment
                    effective_display_val = rel_seg_val + frac_progress
                else -- Moving towards more negative relative segment
                    effective_display_val = rel_seg_val - frac_progress
                end
            end
            display_seg_str = string.format("%+08.3f", effective_display_val) -- Format to 8 chars like +00.123 or +12.345
        end

        local depth_val = enemies_state.enemy_depths[i]
        local depth_str -- For "Depth" column
        if rel_seg_val == INVALID_SEGMENT then
            depth_str = "  --   " -- 7 chars
        else
            depth_str = string.format("  %02X   ", depth_val) -- e.g., "  1A   " (7 chars)
        end

        local frac_column_str -- For "FracP" column
        if rel_seg_val == INVALID_SEGMENT then
            frac_column_str = "  ---- " 
        else
            frac_column_str = string.format(" %.3f ", frac_progress) 
        end
        
        local slot_line = string.format("  %d | %-4s | %-5s | %-8s | %-7s | %-7s", 
            i, type_str, state_str, display_seg_str, depth_str, frac_column_str)
        
        table.insert(display_lines, slot_line)
    end
    table.insert(display_lines, "")
    
    -- Enemy Shots (compact format)
    local e_shots_str = "Enemy Shots: "
    for i = 1, 4 do
        if enemies_state.enemy_shot_segments[i] ~= INVALID_SEGMENT then
            e_shots_str = e_shots_str .. format_segment(enemies_state.enemy_shot_segments[i]) .. " "
        end
    end
    table.insert(display_lines, e_shots_str)
    
    -- Enemy arrays summary (compact one-line format for each)
    local fuseball_str = "Fuseballs: "
    local pulsar_str = "Pulsars:   "
    local top_rail_str = "Top Rail:  "
    
    for i = 1, 7 do
        fuseball_str = fuseball_str .. format_enemy_segment(enemies_state.charging_fuseball[i]) .. " "
        pulsar_str = pulsar_str .. format_enemy_segment(enemies_state.active_pulsar[i]) .. " "
        top_rail_str = top_rail_str .. format_enemy_segment(enemies_state.active_top_rail_enemies[i]) .. " "
    end
    
    table.insert(display_lines, fuseball_str)
    table.insert(display_lines, pulsar_str)
    table.insert(display_lines, top_rail_str)
    
    -- Fractional positions
    local frac_display_str = "FracProg: " -- Renamed to avoid conflict with frac_str inside loop
    for i = 1, 7 do
        local progress_val = enemies_state.enemy_fractional_progress[i]
        if progress_val > 0.005 and progress_val < 0.995 then -- Only display if meaningful
            frac_display_str = frac_display_str .. string.format("%.2f ", progress_val)
        else
            frac_display_str = frac_display_str .. "---- "
        end
    end
    table.insert(display_lines, frac_display_str) -- Use the new variable name
    table.insert(display_lines, "")
    
    -- Pending data summary
    table.insert(display_lines, format_header("Pending Data"))
    
    -- Show first few pending enemies in a compact format
    local pending_count = math.min(6, #enemies_state.pending_vid)
    if pending_count > 0 then
        local pending_str = "Pending: "
        for i = 1, pending_count do
            pending_str = pending_str .. string.format("V:%02X S:%s | ", 
                enemies_state.pending_vid[i], 
                format_segment(enemies_state.pending_seg[i]))
            
            -- Break into multiple lines if needed
            if i % 3 == 0 and i < pending_count then
                table.insert(display_lines, pending_str)
                pending_str = "         "
            end
        end
        table.insert(display_lines, pending_str)
    else
        table.insert(display_lines, "No pending enemies")
    end
    
    -- Join all the lines
    local display_str = table.concat(display_lines, "\n")
    
    -- Write the entire display at once
    io.write(display_str)
    io.flush() -- Ensure output is written immediately
end

return M