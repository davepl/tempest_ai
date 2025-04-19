-- state.lua: Module for game state classes and serialization

local StateUtils = {}

-- Require dependencies
local SegmentUtils = require("segment")

-- Assume 'mem' is accessible as a global from main.lua for updates
-- If this causes issues, 'mem' should be passed explicitly to update methods

-- **GameState Class**
local GameState = {}
GameState.__index = GameState

function GameState:new()
    local self = setmetatable({}, GameState)
    self.credits = 0
    self.p1_level = 0
    self.p1_lives = 0
    self.gamestate = 0                  -- Game state from address 0
    self.game_mode = 0                  -- Game mode from address 5
    self.countdown_timer = 0            -- Countdown timer from address 4
    self.frame_counter = 0              -- Frame counter for tracking progress
    self.last_save_time = os.time()     -- Track when we last sent save signal
    self.save_interval = 300            -- Send save signal every 5 minutes (300 seconds)
    self.current_fps = 0                -- Store the FPS value for display
    return self
end

function GameState:update(mem)
    self.gamestate = mem:read_u8(0x0000)
    self.game_mode = mem:read_u8(0x0005)

    self.countdown_timer = mem:read_u8(0x0004)
    self.credits = mem:read_u8(0x0006)
    self.p1_level = mem:read_u8(0x0046)
    self.p1_lives = mem:read_u8(0x0048)
    self.frame_counter = self.frame_counter + 1
    
    -- Calculate time from frames (assuming 60 FPS)
    local total_seconds = self.frame_counter / 60
    self.time_days = math.floor(total_seconds / 86400)
    self.time_hours = math.floor((total_seconds % 86400) / 3600)
    self.time_mins = math.floor((total_seconds % 3600) / 60)
end

StateUtils.GameState = GameState


-- **LevelState Class**
local LevelState = {}
LevelState.__index = LevelState

function LevelState:new()
    local self = setmetatable({}, LevelState)
    self.level_number = 0
    self.spike_heights = {}
    self.is_open_level = false
    self.level_angles = {}
    self.level_shape = 0
    return self
end

function LevelState:update(mem) -- Keep mem parameter
    self.level_number = mem:read_u8(0x009F)
    self.is_open_level = (mem:read_u8(0x0111) == 0xFF)
    self.level_shape = self.level_number % 16
    
    -- Read spike heights using read_range (16 bytes from 0x03AC to 0x03BB)
    self.spike_heights = {}
    local spike_data = mem:read_range(0x03AC, 0x03BB, 8)
    for i = 1, 16 do
        self.spike_heights[i] = string.byte(spike_data, i)
    end
    
    -- Read level angles using read_range (16 bytes from 0x03EE to 0x03FD)
    self.level_angles = {}
    local angle_data = mem:read_range(0x03EE, 0x03FD, 8)
    for i = 1, 16 do
        self.level_angles[i] = string.byte(angle_data, i)
    end
end

StateUtils.LevelState = LevelState


-- **PlayerState Class**
local PlayerState = {}
PlayerState.__index = PlayerState

function PlayerState:new()
    local self = setmetatable({}, PlayerState)
    self.position = 0; self.alive = 0; self.score = 0
    self.superzapper_uses = 0; self.superzapper_active = 0
    self.player_depth = 0; self.player_state = 0
    self.shot_segments = {0, 0, 0, 0, 0, 0, 0, 0}
    self.shot_positions = {0, 0, 0, 0, 0, 0, 0, 0}
    self.shot_count = 0; self.debounce = 0
    self.fire_detected = 0; self.zap_detected = 0
    self.SpinnerAccum = 0; self.spinner_commanded = 0
    self.prevSpinnerAccum = 0;
    self.fire_commanded = 0; self.zap_commanded = 0
    return self
end

local function bcd_to_decimal(bcd)
    return ((bcd >> 4) * 10) + (bcd & 0x0F)
end

function PlayerState:update(mem, level_state) -- Needs level_state
    self.position = mem:read_u8(0x0200)
    self.player_state = mem:read_u8(0x0201)
    self.player_depth = mem:read_u8(0x0202)
    self.alive = ((self.player_state & 0x80) == 0) and 1 or 0

    local score_low = bcd_to_decimal(mem:read_u8(0x0040))
    local score_mid = bcd_to_decimal(mem:read_u8(0x0041))
    local score_high = bcd_to_decimal(mem:read_u8(0x0042))
    self.score = score_high * 10000 + score_mid * 100 + score_low

    self.superzapper_uses = mem:read_u8(0x03AA)
    self.superzapper_active = mem:read_u8(0x0125)
    self.shot_count = mem:read_u8(0x0135)

    local is_open = level_state.is_open_level
    local player_abs_segment = self.position & 0x0F
    for i = 1, 8 do
        self.shot_positions[i] = mem:read_u8(0x02D3 + i - 1)
        if self.shot_positions[i] == 0 then
            self.shot_segments[i] = SegmentUtils.INVALID_SEGMENT
        else
            local abs_segment = mem:read_u8(0x02AD + i - 1)
            if abs_segment == 0 then
                 self.shot_segments[i] = SegmentUtils.INVALID_SEGMENT
            else
                 self.shot_segments[i] = SegmentUtils.absolute_to_relative_segment(player_abs_segment, abs_segment & 0x0F, is_open)
            end
        end
    end

    self.debounce = mem:read_u8(0x004D)
    self.fire_detected = (self.debounce & 0x10) ~= 0 and 1 or 0
    self.zap_detected = (self.debounce & 0x08) ~= 0 and 1 or 0
    
    local currentSpinnerAccum = mem:read_u8(0x0051)
    local rawDelta = currentSpinnerAccum - self.prevSpinnerAccum
    if rawDelta > 127 then rawDelta = rawDelta - 256 elseif rawDelta < -128 then rawDelta = rawDelta + 256 end

    self.spinner_commanded = rawDelta

    self.SpinnerAccum = currentSpinnerAccum
    self.prevSpinnerAccum = currentSpinnerAccum
end

StateUtils.PlayerState = PlayerState

-- **EnemiesState Class**
local EnemiesState = {}
EnemiesState.__index = EnemiesState

function EnemiesState:new()
    local self = setmetatable({}, EnemiesState)
    self.active_flippers = 0; self.active_pulsars = 0; self.active_tankers = 0;
    self.active_spikers = 0; self.active_fuseballs = 0; self.pulse_beat = 0;
    self.pulsing = 0; self.pulsar_fliprate = 0; self.num_enemies_in_tube = 0;
    self.num_enemies_on_top = 0; self.enemies_pending = 0;
    self.spawn_slots_flippers = 0; self.spawn_slots_pulsars = 0; self.spawn_slots_tankers = 0;
    self.spawn_slots_spikers = 0; self.spawn_slots_fuseballs = 0;
    self.shot_positions = {0,0,0,0} -- Enemy shot positions (depth)
    self.enemy_type_info = {}; self.active_enemy_info = {}
    self.enemy_segments = {}; self.enemy_depths = {}; self.enemy_depths_lsb = {}
    self.enemy_shot_lsb = {0,0,0,0,0,0,0} -- Keep size 7 for consistency, only use 1-4
    self.enemy_core_type = {}; self.enemy_direction_moving = {}
    self.enemy_between_segments = {}; self.enemy_moving_away = {}
    self.enemy_can_shoot = {}; self.enemy_split_behavior = {}
    self.pending_seg = {}; self.pending_vid = {}
    self.display_list = {}
    self.nearest_enemy_seg = SegmentUtils.INVALID_SEGMENT
    self.nearest_enemy_depth_raw = 0
    self.nearest_enemy_abs_seg_raw = -1 -- Raw absolute segment
    self.is_aligned_with_nearest = 0.0
    self.charging_fuseball_segments = {} -- Tracks segments with charging fuseballs
    self.pulsar_lanes = {} -- Tracks segments with active pulsars
    
    -- Addresses for enemy shot segments (read later)
    self.enemy_shot_segments = {
        { address = 0x02C0, value = SegmentUtils.INVALID_SEGMENT }, -- shot 1 seg
        { address = 0x02C1, value = SegmentUtils.INVALID_SEGMENT }, -- shot 2 seg
        { address = 0x02C2, value = SegmentUtils.INVALID_SEGMENT }, -- shot 3 seg
        { address = 0x02C3, value = SegmentUtils.INVALID_SEGMENT }  -- shot 4 seg
    }
    return self
end

-- Find the *absolute* segment and depth of the enemy closest to the top of the tube
-- Keep this method within EnemiesState as it reads directly from memory
function EnemiesState:nearest_enemy_segment(mem, level_state)
    if not level_state then
        error("FATAL: level_state argument is nil inside EnemiesState:nearest_enemy_segment", 2) 
    end
    local player_abs_segment = mem:read_u8(0x0200) & 0x0F
    local is_open = level_state.is_open_level
    local current_enemy_abs_segments = {}
    local current_enemy_depths = {}
    for i = 1, 7 do
        current_enemy_abs_segments[i] = mem:read_u8(0x02B9 + i - 1) & 0x0F
        current_enemy_depths[i] = mem:read_u8(0x02DF + i - 1)
    end
    -- Call the utility function with the gathered data
    return SegmentUtils.find_nearest_enemy_segment(player_abs_segment, current_enemy_abs_segments, current_enemy_depths, is_open)
end

function EnemiesState:get_total_active()
    return self.active_flippers + self.active_pulsars + self.active_tankers + 
           self.active_spikers + self.active_fuseballs
end


function EnemiesState:update(mem, level_state) -- Needs level_state
    -- Reset arrays/tables that are fully recalculated
    self.enemy_type_info = {0,0,0,0,0,0,0}; self.active_enemy_info = {0,0,0,0,0,0,0}
    self.enemy_segments = {0,0,0,0,0,0,0}; self.enemy_depths = {0,0,0,0,0,0,0}
    self.enemy_depths_lsb = {0,0,0,0,0,0,0}; 
    self.enemy_shot_lsb = {0,0,0,0,0,0,0}; -- Keep size 7 for consistency, but only use 1-4
    self.enemy_core_type = {0,0,0,0,0,0,0}; self.enemy_direction_moving = {0,0,0,0,0,0,0}
    self.enemy_between_segments = {0,0,0,0,0,0,0}; self.enemy_moving_away = {0,0,0,0,0,0,0}
    self.enemy_can_shoot = {0,0,0,0,0,0,0}; self.enemy_split_behavior = {0,0,0,0,0,0,0}
    self.charging_fuseball_segments = {} -- Reset derived tables
    self.pulsar_lanes = {} -- Reset derived tables
    -- Initialize derived per-segment tables
    for i = 1, 16 do
        self.charging_fuseball_segments[i] = 0
        self.pulsar_lanes[i] = 0
    end

    -- Read enemy counts (9 bytes from 0x0142 to 0x014A)
    local counts = mem:read_range(0x0142, 0x014A, 8)
    self.active_flippers = string.byte(counts, 1)
    self.active_pulsars = string.byte(counts, 2)
    self.active_tankers = string.byte(counts, 3)
    self.active_spikers = string.byte(counts, 4)
    self.active_fuseballs = string.byte(counts, 5)
    self.pulse_beat = string.byte(counts, 6)
    self.pulsing = string.byte(counts, 7)
    self.pulsar_fliprate = string.byte(counts, 8)
    -- self.num_enemies_in_tube = string.byte(counts, 9) -- Incorrectly read from 0x014A via batch

    -- Read spawn slots (5 bytes from 0x013D to 0x0141)
    local spawn_slots = mem:read_range(0x013D, 0x0141, 8)
    self.spawn_slots_flippers = string.byte(spawn_slots, 1)
    self.spawn_slots_pulsars = string.byte(spawn_slots, 2)
    self.spawn_slots_tankers = string.byte(spawn_slots, 3)
    self.spawn_slots_spikers = string.byte(spawn_slots, 4)
    self.spawn_slots_fuseballs = string.byte(spawn_slots, 5)

    -- Read additional counts directly
    self.num_enemies_in_tube = mem:read_u8(0x0108) 
    self.num_enemies_on_top = mem:read_u8(0x0109) 
    self.enemies_pending = mem:read_u8(0x03AB)

    local player_abs_segment = mem:read_u8(0x0200) & 0x0F
    local is_open = level_state.is_open_level
    
    -- Batch read shot positions (4 bytes from 0x02DB to 0x02DE)
    local shot_positions = mem:read_range(0x02DB, 0x02DE, 8)
    for i = 1, 4 do self.shot_positions[i] = string.byte(shot_positions, i) end

    -- Update enemy shot segments (now safe to use self.shot_positions)
    for i = 1, 4 do
        local abs_segment = mem:read_u8(self.enemy_shot_segments[i].address)
        if abs_segment == 0 or self.shot_positions[i] == 0 then
            self.enemy_shot_segments[i].value = SegmentUtils.INVALID_SEGMENT
        else
            self.enemy_shot_segments[i].value = SegmentUtils.absolute_to_relative_segment(player_abs_segment, abs_segment & 0x0F, is_open)
        end
    end
    
    -- Batch read enemy data (7 bytes from 0x02B9 to 0x02BF)
    local enemy_data = mem:read_range(0x02B9, 0x02BF, 8)
    -- Batch read enemy depths (7 bytes from 0x02DF to 0x02E5)
    local enemy_depths = mem:read_range(0x02DF, 0x02E5, 8)
    
    for i = 1, 7 do
        local abs_segment = string.byte(enemy_data, i) & 0x0F
        self.enemy_depths[i] = string.byte(enemy_depths, i)
        
        if (self.enemy_depths[i] == 0 or abs_segment == 0) then -- Check abs_segment too
            self.enemy_segments[i] = SegmentUtils.INVALID_SEGMENT
        else
            self.enemy_segments[i] = SegmentUtils.absolute_to_relative_segment(player_abs_segment, abs_segment, is_open)
        end
    end

    -- Find nearest enemy using the class method (which calls SegmentUtils)
    local nearest_abs_seg, nearest_depth = self:nearest_enemy_segment(mem, level_state)
    self.nearest_enemy_abs_seg_raw = nearest_abs_seg -- Store raw absolute segment
    self.nearest_enemy_depth_raw = nearest_depth

    -- Update relative nearest segment and engineered features
    if nearest_abs_seg == -1 then
        self.nearest_enemy_seg = SegmentUtils.INVALID_SEGMENT
        self.is_aligned_with_nearest = 0.0; 
    else
        local nearest_rel_seg = SegmentUtils.absolute_to_relative_segment(player_abs_segment, nearest_abs_seg, is_open)
        self.nearest_enemy_seg = nearest_rel_seg
        self.is_aligned_with_nearest = (nearest_rel_seg == 0) and 1.0 or 0.0
    end

    -- Batch read enemy type info (7 bytes from 0x0283 to 0x0289)
    local type_info = mem:read_range(0x0283, 0x0289, 8)
    -- Batch read active enemy info (7 bytes from 0x028A to 0x0290)
    local active_info = mem:read_range(0x028A, 0x0290, 8)
    
    for i = 1, 7 do
        self.enemy_type_info[i] = string.byte(type_info, i)
        self.active_enemy_info[i] = string.byte(active_info, i)
        local current_abs_segment_local = string.byte(enemy_data, i) & 0x0F -- Need local absolute for charging/pulsar update

        if self.enemy_depths[i] > 0 and current_abs_segment_local >= 0 and current_abs_segment_local <= 15 then -- Check active and valid segment
            local type_byte = self.enemy_type_info[i]
            local state_byte = self.active_enemy_info[i]
            self.enemy_core_type[i] = type_byte & 0x07
            self.enemy_direction_moving[i] = (type_byte & 0x40) ~= 0 and 1 or 0
            self.enemy_between_segments[i] = (type_byte & 0x80) ~= 0 and 1 or 0
            self.enemy_moving_away[i] = (state_byte & 0x80) ~= 0 and 1 or 0
            self.enemy_can_shoot[i] = (state_byte & 0x40) ~= 0 and 1 or 0
            self.enemy_split_behavior[i] = state_byte & 0x03
            
            -- Update charging fuseball segments
            if self.enemy_core_type[i] == 4 and self.enemy_moving_away[i] == 0 then -- Type 4 = Fuseball
                 self.charging_fuseball_segments[current_abs_segment_local + 1] = 1
            end
            
            -- Update pulsar lanes (Assuming Pulsar core type is 1)
            if self.enemy_core_type[i] == 1 then 
                self.pulsar_lanes[current_abs_segment_local + 1] = 1
            end
        else 
            -- Explicitly zero out decoded info for inactive enemies
            self.enemy_core_type[i] = 0
            self.enemy_direction_moving[i] = 0
            self.enemy_between_segments[i] = 0
            self.enemy_moving_away[i] = 0
            self.enemy_can_shoot[i] = 0
            self.enemy_split_behavior[i] = 0
            -- Also ensure segments are invalid if depth is 0
            self.enemy_segments[i] = SegmentUtils.INVALID_SEGMENT
        end
    end
    
    -- Read Enemy Shot LSBs (only 4 shots exist)
    local shot_lsbs = mem:read_range(0x02E6, 0x02E9, 8)
    for i = 1, 4 do -- Loop only 4 times
        if self.shot_positions[i] > 0 then -- Check if the corresponding shot is active
            self.enemy_shot_lsb[i] = string.byte(shot_lsbs, i)
        else
            self.enemy_shot_lsb[i] = 0 -- Ensure LSB is 0 for inactive shots
        end
    end

    -- Batch read pending seg/vid (64 bytes each)
    local pending_seg = mem:read_range(0x0203, 0x0242, 8)
    local pending_vid = mem:read_range(0x0243, 0x0282, 8)
    for i = 1, 64 do
        local abs_segment = string.byte(pending_seg, i)
        if abs_segment == 0 then 
            self.pending_seg[i] = SegmentUtils.INVALID_SEGMENT
        else 
            self.pending_seg[i] = SegmentUtils.absolute_to_relative_segment(player_abs_segment, abs_segment & 0x0F, is_open) 
        end
        self.pending_vid[i] = string.byte(pending_vid, i)
    end

    -- Scan display list
    self.display_list = {}
    for i = 0, 31 do
        local command = mem:read_u8(0x0300 + i * 4)
        local abs_segment = mem:read_u8(0x0301 + i * 4) & 0x0F
        local depth = mem:read_u8(0x0302 + i * 4)
        local type_val = mem:read_u8(0x0303 + i * 4)
        local rel_segment = SegmentUtils.INVALID_SEGMENT
        if command ~= 0 and depth ~= 0 and abs_segment >= 0 and abs_segment <= 15 then
            rel_segment = SegmentUtils.absolute_to_relative_segment(player_abs_segment, abs_segment, is_open)
        end
        self.display_list[i] = { command=command, segment=rel_segment, depth=depth, type=type_val }
    end
end

StateUtils.EnemiesState = EnemiesState


-- **ControlState Class** (Moved from main.lua and renamed)
local ControlState = {}
ControlState.__index = ControlState

function ControlState:new()
    local self = setmetatable({}, ControlState)

    -- Get button ports (needs access to manager)
    -- Assuming 'manager' is accessible globally for now
    print("DEBUG state.lua: Type of 'manager' in ControlState:new() = ", type(manager)) -- Added Debug Print
    self.button_port = manager.machine.ioport.ports[":BUTTONSP1"]
    self.spinner_port = manager.machine.ioport.ports[":KNOBP1"]
    
    -- Set up button fields
    self.fire_field = self.button_port and self.button_port.fields["P1 Button 1"] or nil
    self.zap_field = self.button_port and self.button_port.fields["P1 Button 2"] or nil
    self.spinner_field = self.spinner_port and self.spinner_port.fields["Dial"] or nil
    
    -- Track commanded states (These are also in player_state, consider consolidation)
    self.fire_commanded = 0
    self.zap_commanded = 0
    
    -- Keep validation prints for now
    print("Button port found: " .. (self.button_port and "Yes" or "No"))
    if self.button_port then
        print("  Fire field found: " .. (self.fire_field and "Yes" or "No"))
        print("  Zap field found: " .. (self.zap_field and "Yes" or "No"))
    end
    print("Spinner port found: " .. (self.spinner_port and "Yes" or "No"))
    if self.spinner_port then
        print("  Dial field found: " .. (self.spinner_field and "Yes" or "No"))
    end

    return self
end

-- Updated to accept 'mem' argument
function ControlState:apply_action(mem, fire, zap, spinner, game_state, player_state) 
    -- Directly apply commanded actions
    player_state.fire_commanded = fire
    player_state.zap_commanded = zap
    player_state.spinner_commanded = spinner 

    if self.fire_field then self.fire_field:set_value(fire) end
    if self.zap_field then self.zap_field:set_value(zap) end
    if self.spinner_field then mem:write_u8(0x0050, spinner) end 
end

StateUtils.ControlState = ControlState -- Export the new class


-- Flattens the entire game state into a single table of numbers for sending
function StateUtils.flatten_game_state_to_binary(reward, game_state, level_state, player_state, enemies_state, controls, bDone, shutdown_requested)
    -- Use a single table for all state data
    local data = {}

    -- 1. Game State
    table.insert(data, game_state.gamestate)
    table.insert(data, game_state.game_mode)
    table.insert(data, game_state.countdown_timer)
    table.insert(data, game_state.p1_lives)
    table.insert(data, game_state.p1_level)
    table.insert(data, game_state.credits)

    -- 2. Level State
    table.insert(data, level_state.level_number)
    table.insert(data, level_state.level_shape)
    table.insert(data, level_state.is_open_level and 1 or 0)
    for i = 1, 16 do table.insert(data, level_state.spike_heights[i]) end
    for i = 1, 16 do table.insert(data, level_state.level_angles[i]) end

    -- 3. Player State
    table.insert(data, player_state.position)
    table.insert(data, player_state.player_depth)
    table.insert(data, player_state.player_state)
    table.insert(data, player_state.alive and 1 or 0)
    table.insert(data, player_state.superzapper_uses)
    table.insert(data, player_state.superzapper_active and 1 or 0)
    table.insert(data, player_state.shot_count)
    table.insert(data, player_state.inferredSpinnerDelta)

    for i = 1, 8 do table.insert(data, player_state.shot_segments[i]) end
    for i = 1, 8 do table.insert(data, player_state.shot_positions[i]) end

    -- 4. Enemy State
    table.insert(data, enemies_state.active_flippers)
    table.insert(data, enemies_state.active_pulsars)
    table.insert(data, enemies_state.active_tankers)
    table.insert(data, enemies_state.active_spikers)
    table.insert(data, enemies_state.active_fuseballs)
    table.insert(data, enemies_state.num_enemies_in_tube)
    table.insert(data, enemies_state.num_enemies_on_top)
    table.insert(data, enemies_state.enemies_pending)
    table.insert(data, enemies_state.spawn_slots_flippers)
    table.insert(data, enemies_state.spawn_slots_pulsars)
    table.insert(data, enemies_state.spawn_slots_tankers)
    table.insert(data, enemies_state.spawn_slots_spikers)
    table.insert(data, enemies_state.spawn_slots_fuseballs)
    table.insert(data, enemies_state.pulsar_fliprate)
    table.insert(data, enemies_state.pulse_beat)
    table.insert(data, enemies_state.pulsing)

    -- Nearest Enemy
    table.insert(data, enemies_state.nearest_enemy_seg)
    table.insert(data, enemies_state.nearest_enemy_depth_raw)
    table.insert(data, enemies_state.is_aligned_with_nearest)

    -- Enemy Segments/Depths (7 enemies)
    for i = 1, 7 do table.insert(data, enemies_state.enemy_segments[i]) end
    for i = 1, 7 do table.insert(data, enemies_state.enemy_depths[i]) end

    -- Enemy Shot Info (4 shots)
    for i = 1, 4 do table.insert(data, enemies_state.enemy_shot_segments[i].value) end
    for i = 1, 4 do table.insert(data, enemies_state.shot_positions[i]) end
    for i = 1, 4 do table.insert(data, enemies_state.enemy_shot_lsb[i]) end

    -- Enemy Flags (7 enemies)
    for i = 1, 7 do table.insert(data, enemies_state.enemy_moving_away[i]) end
    for i = 1, 7 do table.insert(data, enemies_state.enemy_can_shoot[i]) end
    for i = 1, 7 do table.insert(data, enemies_state.enemy_split_behavior[i]) end
    for i = 1, 7 do table.insert(data, enemies_state.enemy_core_type[i]) end
    for i = 1, 7 do table.insert(data, enemies_state.enemy_direction_moving[i]) end
    for i = 1, 7 do table.insert(data, enemies_state.enemy_between_segments[i]) end

    -- Per-Segment Flags (16 segments)
    for i = 1, 16 do table.insert(data, enemies_state.charging_fuseball_segments[i]) end
    for i = 1, 16 do table.insert(data, enemies_state.pulsar_lanes[i]) end

    -- 5. Pending Spawns (Lookahead)
    for i = 1, 64 do table.insert(data, enemies_state.pending_seg[i]) end
    for i = 1, 64 do table.insert(data, enemies_state.pending_vid[i]) end


    local num_total_values = #data

    -- Prepare OOB Header values
    local lua_frame_count_val = game_state.frame_counter -- Get the frame counter
    local nearest_enemy_byte_val = (enemies_state.nearest_enemy_abs_seg_raw == SegmentUtils.INVALID_SEGMENT) and -1 or enemies_state.nearest_enemy_abs_seg_raw
    local player_abs_seg_oob = player_state.position & 0x0F
    local is_open_level_byte_val = level_state.is_open_level and 1 or 0
    local reward_val = math.floor(reward)
    local game_mode_val = game_state.game_mode
    local bDone_val = bDone and 1 or 0
    local save_signal_val = (shutdown_requested) and 1 or 0
    local fire_cmd_val = player_state.fire_commanded
    local zap_cmd_val = player_state.zap_commanded
    local spinner_cmd_val = player_state.spinner_commanded

    -- Prepend '>I' for the 32-bit unsigned frame counter
    local oob_format_str = ">IHdBBBBBhbBB" -- New 12-field format

    -- Pack OOB header directly using variables
    local oob_data = string.pack(oob_format_str,
        lua_frame_count_val, -- 1. Lua frame count (New)
        num_total_values,    -- 2. State vector size
        reward_val,          -- 3. Reward
        game_mode_val,       -- 4. Game Mode
        bDone_val,           -- 5. Done flag
        save_signal_val,     -- 6. Save signal
        fire_cmd_val,        -- 7. Fire commanded
        zap_cmd_val,         -- 8. Zap commanded
        spinner_cmd_val,     -- 9. Spinner commanded
        nearest_enemy_byte_val,
        player_abs_seg_oob,
        is_open_level_byte_val
    )

    -- Pack the main state data table into a binary string using SIGNED SHORT '>h'
    local binary_data = ""
    for i, value in ipairs(data) do
        local num_value = tonumber(value)
        assert(num_value >= -128 and num_value <= 255, string.format("Value out of range at index %d: %s", i, tostring(value)))
        local packed_value = math.max(-32768, math.min(32767, math.floor(num_value)))
        binary_data = binary_data .. string.pack(">h", packed_value)
    end

    return oob_data .. binary_data, num_total_values
end

return StateUtils 