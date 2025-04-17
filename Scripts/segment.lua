-- segment.lua: Module for segment calculations and utilities

local SegmentUtils = {}

-- Define and export constants
SegmentUtils.INVALID_SEGMENT = -128

-- Function to get the relative distance to a target segment
-- Exported as it's used by various state update functions
function SegmentUtils.absolute_to_relative_segment(current_abs_segment, target_abs_segment, is_open_level)
    -- Input validation: Check for nil or invalid target segment BEFORE bitwise operations
    if type(target_abs_segment) ~= "number" or target_abs_segment < 0 or target_abs_segment > 15 then
        return SegmentUtils.INVALID_SEGMENT -- Return invalid if target is nil, negative, or > 15
    end

    -- Mask inputs to ensure they are within 0-15 range (now safe after validation)
    current_abs_segment = current_abs_segment & 0x0F
    target_abs_segment = target_abs_segment & 0x0F -- Line 13 (now safe)

    -- Get segment distance based on level type
    if is_open_level then
        -- Open level: simple distance calculation (-15 to +15)
        return target_abs_segment - current_abs_segment
    else
        -- Closed level: find shortest path around the circle (-7 to +8)
        local diff = target_abs_segment - current_abs_segment
        if diff > 8 then
            return diff - 16 -- Wrap around (e.g., 1 -> 15 is -2)
        elseif diff <= -8 then
            return diff + 16 -- Wrap around (e.g., 15 -> 1 is +2)
        else
            return diff -- Includes 0 and +8
        end
    end
end

-- Find the *absolute* segment and depth of the enemy closest to the player
-- Takes pre-read absolute segments and depths for the 7 enemy slots
function SegmentUtils.find_nearest_enemy_segment(player_abs_segment, enemy_abs_segments, enemy_depths, is_open_level)
    local min_depth = 255
    local closest_absolute_segment = -1 -- Use -1 as sentinel for not found
    local min_relative_distance_abs = 17 -- Max possible relative distance + 1

    player_abs_segment = player_abs_segment & 0x0F -- Ensure player segment is masked

    for i = 1, 7 do
        local current_abs_segment = enemy_abs_segments[i]
        local current_depth = enemy_depths[i]

        -- Only consider active enemies (depth > 0) with valid segments (0-15)
        if current_depth and current_depth > 0 and current_abs_segment and current_abs_segment >= 0 and current_abs_segment <= 15 then
            -- Calculate relative distance for this enemy using the module's function
            local current_relative_distance = SegmentUtils.absolute_to_relative_segment(player_abs_segment, current_abs_segment, is_open_level)
            local current_relative_distance_abs = math.abs(current_relative_distance)

            -- Priority 1: Closer depth always wins
            if current_depth < min_depth then
                min_depth = current_depth
                closest_absolute_segment = current_abs_segment
                min_relative_distance_abs = current_relative_distance_abs
            -- Priority 2: Same depth, closer segment wins
            elseif current_depth == min_depth then
                if current_relative_distance_abs < min_relative_distance_abs then
                    closest_absolute_segment = current_abs_segment
                    min_relative_distance_abs = current_relative_distance_abs
                end
            end
        end
    end

    -- Return the absolute segment (-1 if none found) and its depth
    return closest_absolute_segment, min_depth
end


-- Calculates desired spinner direction/intensity and segment distance to a target enemy
function SegmentUtils.calculate_direction_intensity(player_abs_segment, enemy_abs_segment, is_open_level)
    player_abs_segment = player_abs_segment & 0x0F -- Ensure player segment is masked

    -- Calculate the relative segment distance first
    local enemy_relative_dist = SegmentUtils.absolute_to_relative_segment(player_abs_segment, enemy_abs_segment, is_open_level)
    local actual_segment_distance = math.abs(enemy_relative_dist)

    -- If already aligned (relative distance is 0)
    if enemy_relative_dist == 0 then
        return 0, 0 -- Spinner 0, distance 0
    end

    -- Calculate intensity based on distance
    -- Intensity ramps up faster now
    local intensity = math.min(0.9, 0.1 + (actual_segment_distance * 0.1)) 

    local spinner
    -- Set spinner direction based on the sign of the relative distance
    if is_open_level then
        -- Open Level: Positive relative distance means enemy is clockwise (higher segment index)
        -- We want to move counter-clockwise (negative spinner) towards it.
        spinner = enemy_relative_dist > 0 and -intensity or intensity
    else
        -- Closed Level: Positive relative distance means enemy is clockwise.
        -- We want to move clockwise (positive spinner) towards it.
        spinner = enemy_relative_dist > 0 and intensity or -intensity
    end

    return spinner, actual_segment_distance -- Return spinner value and distance
end


return SegmentUtils 