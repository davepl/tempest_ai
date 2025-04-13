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

function GameState:update(mem) -- Keep mem parameter for clarity, even if relying on global
    self.gamestate = mem:read_u8(0x0000)
    self.game_mode = mem:read_u8(0x0005)
    self.countdown_timer = mem:read_u8(0x0004)
    self.credits = mem:read_u8(0x0006)
    self.p1_level = mem:read_u8(0x0046)
    self.p1_lives = mem:read_u8(0x0048)
    self.frame_counter = self.frame_counter + 1
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
    
    self.spike_heights = {}
    for i = 0, 15 do self.spike_heights[i] = mem:read_u8(0x03AC + i) end
    self.level_angles = {}
    for i = 0, 15 do self.level_angles[i] = mem:read_u8(0x03EE + i) end
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
    self.SpinnerAccum = 0; self.SpinnerDelta = 0
    self.prevSpinnerAccum = 0; self.inferredSpinnerDelta = 0
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
    self.SpinnerDelta = mem:read_u8(0x0050)
    local rawDelta = currentSpinnerAccum - self.prevSpinnerAccum
    if rawDelta > 127 then rawDelta = rawDelta - 256 elseif rawDelta < -128 then rawDelta = rawDelta + 256 end
    self.inferredSpinnerDelta = rawDelta
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
    self.nearest_enemy_seg = SegmentUtils.INVALID_SEGMENT;
    self.nearest_enemy_abs_seg_raw = -1; -- Added in main.lua previously
    self.is_aligned_with_nearest = 0.0; self.nearest_enemy_depth_raw = 255;
    self.alignment_error_magnitude = 0.0;
    self.charging_fuseball_segments = {};
    self.enemy_type_info = {0,0,0,0,0,0,0}; self.active_enemy_info = {0,0,0,0,0,0,0};
    self.enemy_segments = {0,0,0,0,0,0,0}; self.enemy_depths = {0,0,0,0,0,0,0};
    self.enemy_depths_lsb = {0,0,0,0,0,0,0}; 
    self.enemy_shot_lsb = {0,0,0,0,0,0,0}; -- Keep size 7 for consistency, but only use 1-4
    self.enemy_core_type = {0,0,0,0,0,0,0}; self.enemy_direction_moving = {0,0,0,0,0,0,0};
    self.enemy_between_segments = {0,0,0,0,0,0,0}; self.enemy_moving_away = {0,0,0,0,0,0,0};
    self.enemy_can_shoot = {0,0,0,0,0,0,0}; self.enemy_split_behavior = {0,0,0,0,0,0,0};
    self.shot_positions = {0, 0, 0, 0};
    self.pending_vid = {}; self.pending_seg = {};
    self.enemy_shot_segments = { { address=0x02B5, value=0 }, { address=0x02B6, value=0 }, { address=0x02B7, value=0 }, { address=0x02B8, value=0 } };
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
    -- Reset arrays
    self.enemy_type_info = {0,0,0,0,0,0,0}; self.active_enemy_info = {0,0,0,0,0,0,0}
    self.enemy_segments = {0,0,0,0,0,0,0}; self.enemy_depths = {0,0,0,0,0,0,0}
    self.enemy_depths_lsb = {0,0,0,0,0,0,0}; 
    self.enemy_shot_lsb = {0,0,0,0,0,0,0}; -- Keep size 7 for consistency, but only use 1-4
    self.enemy_core_type = {0,0,0,0,0,0,0}; self.enemy_direction_moving = {0,0,0,0,0,0,0}
    self.enemy_between_segments = {0,0,0,0,0,0,0}; self.enemy_moving_away = {0,0,0,0,0,0,0}
    self.enemy_can_shoot = {0,0,0,0,0,0,0}; self.enemy_split_behavior = {0,0,0,0,0,0,0}
    self.charging_fuseball_segments = {}

    -- Read counts
    self.active_flippers=mem:read_u8(0x0142); self.active_pulsars=mem:read_u8(0x0143); self.active_tankers=mem:read_u8(0x0144);
    self.active_spikers=mem:read_u8(0x0145); self.active_fuseballs=mem:read_u8(0x0146); self.pulse_beat=mem:read_u8(0x0147);
    self.pulsing=mem:read_u8(0x0148); self.pulsar_fliprate=mem:read_u8(0x00B2); self.num_enemies_in_tube=mem:read_u8(0x0108);
    self.num_enemies_on_top=mem:read_u8(0x0109); self.enemies_pending=mem:read_u8(0x03AB);
    self.spawn_slots_flippers=mem:read_u8(0x013D); self.spawn_slots_pulsars=mem:read_u8(0x013E);
    self.spawn_slots_tankers=mem:read_u8(0x013F); self.spawn_slots_spikers=mem:read_u8(0x0140);
    self.spawn_slots_fuseballs=mem:read_u8(0x0141);

    local player_abs_segment = mem:read_u8(0x0200) & 0x0F
    local is_open = level_state.is_open_level
    
    -- Read enemy shot positions first
    for i = 1, 4 do self.shot_positions[i] = mem:read_u8(0x02DB + i - 1) end

    -- Update enemy shot segments (now safe to use self.shot_positions)
    for i = 1, 4 do
        local abs_segment = mem:read_u8(self.enemy_shot_segments[i].address)
        if abs_segment == 0 or self.shot_positions[i] == 0 then
            self.enemy_shot_segments[i].value = SegmentUtils.INVALID_SEGMENT
        else
            self.enemy_shot_segments[i].value = SegmentUtils.absolute_to_relative_segment(player_abs_segment, abs_segment & 0x0F, is_open)
        end
    end
    
    -- Read raw enemy data for segment calculations
    local raw_enemy_abs_segments = {}
    for i = 1, 7 do
        local abs_segment = mem:read_u8(0x02B9 + i - 1)
        self.enemy_depths[i] = mem:read_u8(0x02DF + i - 1)
        raw_enemy_abs_segments[i] = abs_segment & 0x0F -- Store masked absolute segment

        if (self.enemy_depths[i] == 0 or abs_segment == 0) then
            self.enemy_segments[i] = SegmentUtils.INVALID_SEGMENT
        else
            self.enemy_segments[i] = SegmentUtils.absolute_to_relative_segment(player_abs_segment, raw_enemy_abs_segments[i], is_open)
        end
    end

    -- Find nearest enemy using the class method (which calls SegmentUtils)
    local nearest_abs_seg, nearest_depth = self:nearest_enemy_segment(mem, level_state)
    self.nearest_enemy_abs_seg_raw = nearest_abs_seg -- Store raw absolute segment
    self.nearest_enemy_depth_raw = nearest_depth

    -- Update relative nearest segment and engineered features
    if nearest_abs_seg == -1 then
        self.nearest_enemy_seg = SegmentUtils.INVALID_SEGMENT
        self.is_aligned_with_nearest = 0.0; self.alignment_error_magnitude = 0.0;
    else
        local nearest_rel_seg = SegmentUtils.absolute_to_relative_segment(player_abs_segment, nearest_abs_seg, is_open)
        self.nearest_enemy_seg = nearest_rel_seg
        self.is_aligned_with_nearest = (nearest_rel_seg == 0) and 1.0 or 0.0
        local error_abs = math.abs(nearest_rel_seg)
        local max_dist = is_open and 15.0 or 8.0
        local normalized_error = (error_abs > 0 and max_dist > 0) and (error_abs / max_dist) or 0.0
        self.alignment_error_magnitude = math.floor(normalized_error * 10000.0)
    end

    -- Read and decode Type/State bytes
    for i = 1, 7 do
        self.enemy_type_info[i] = mem:read_u8(0x0283 + i - 1) 
        self.active_enemy_info[i] = mem:read_u8(0x028A + i - 1) 
        if self.enemy_depths[i] > 0 then
            local type_byte = self.enemy_type_info[i]
            local state_byte = self.active_enemy_info[i]
            self.enemy_core_type[i] = type_byte & 0x07
            self.enemy_direction_moving[i] = (type_byte & 0x40) ~= 0 and 1 or 0
            self.enemy_between_segments[i] = (type_byte & 0x80) ~= 0 and 1 or 0
            self.enemy_moving_away[i] = (state_byte & 0x80) ~= 0 and 1 or 0
            self.enemy_can_shoot[i] = (state_byte & 0x40) ~= 0 and 1 or 0
            self.enemy_split_behavior[i] = state_byte & 0x03
            if self.enemy_core_type[i] == 4 and self.enemy_moving_away[i] == 0 then
                 self.charging_fuseball_segments[(mem:read_u8(0x02B9 + i - 1) & 0x0F) + 1] = 1
            end
        else 
            -- Explicitly zero out decoded info for inactive enemies
            self.enemy_core_type[i] = 0
            self.enemy_direction_moving[i] = 0
            self.enemy_between_segments[i] = 0
            self.enemy_moving_away[i] = 0
            self.enemy_can_shoot[i] = 0
            self.enemy_split_behavior[i] = 0
            -- Also ensure relative segment is invalid if depth is 0
            self.enemy_segments[i] = SegmentUtils.INVALID_SEGMENT
        end
    end
    
    -- Read Enemy Shot LSBs (only 4 shots exist)
    for i = 1, 4 do -- Loop only 4 times
        if self.shot_positions[i] > 0 then -- Check if the corresponding shot is active
            self.enemy_shot_lsb[i] = mem:read_u8(0x02E6 + i - 1) 
        else
            self.enemy_shot_lsb[i] = 0 -- Ensure LSB is 0 for inactive shots
        end
    end
     -- Ensure remaining LSB slots (5-7) are 0 (already handled by reset at top)

    -- Read pending seg/vid
    for i = 1, 64 do
        local abs_segment = mem:read_u8(0x0203 + i - 1)
        if abs_segment == 0 then self.pending_seg[i] = SegmentUtils.INVALID_SEGMENT
        else self.pending_seg[i] = SegmentUtils.absolute_to_relative_segment(player_abs_segment, abs_segment & 0x0F, is_open) end
        self.pending_vid[i] = mem:read_u8(0x0243 + i - 1)
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
    self.button_port = manager.machine.ioport.ports[":BUTTONSP1"]
    self.spinner_port = manager.machine.ioport.ports[":KNOBP1"]
    
    -- Set up button fields
    self.fire_field = self.button_port and self.button_port.fields["P1 Button 1"] or nil
    self.zap_field = self.button_port and self.button_port.fields["P1 Button 2"] or nil
    self.spinner_field = self.spinner_port and self.spinner_port.fields["Dial"] or nil
    
    -- Track commanded states (These are also in player_state, consider consolidation)
    self.fire_commanded = 0
    self.zap_commanded = 0
    self.spinner_delta = 0
    
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
    local is_attract_mode = (game_state.game_mode & 0x80) == 0
    
    if is_attract_mode then
        player_state.fire_commanded = 1 
        if self.fire_field then self.fire_field:set_value(1) end
        player_state.zap_commanded = 0
        if self.zap_field then self.zap_field:set_value(0) end
        player_state.SpinnerDelta = player_state.inferredSpinnerDelta 
    else
        player_state.fire_commanded = fire
        player_state.zap_commanded = zap
        player_state.SpinnerDelta = spinner 

        if self.fire_field then self.fire_field:set_value(fire) end
        if self.zap_field then self.zap_field:set_value(zap) end
        
        -- Use the passed 'mem' object to write spinner delta
        if self.spinner_field then mem:write_u8(0x0050, spinner) end 
    end
end

StateUtils.ControlState = ControlState -- Export the new class


-- Flatten game state to binary (moved from main.lua)
-- Requires controls object to be passed (Now expecting a ControlState instance)
function StateUtils.flatten_game_state_to_binary(reward, game_state, level_state, player_state, enemies_state, controls, bDone, shutdown_requested)
    local data = {}
    -- Game state (Integers)
    table.insert(data, game_state.gamestate); table.insert(data, game_state.game_mode); 
    table.insert(data, game_state.countdown_timer); table.insert(data, game_state.p1_lives); 
    table.insert(data, game_state.p1_level);

    -- Nearest enemy relative segment and delta (Integers)
    local nearest_relative_seg = enemies_state.nearest_enemy_seg 
    local segment_delta = (nearest_relative_seg == SegmentUtils.INVALID_SEGMENT) and 0 or nearest_relative_seg
    table.insert(data, nearest_relative_seg) 
    table.insert(data, segment_delta)      
    table.insert(data, enemies_state.nearest_enemy_depth_raw) -- Integer (0-255)
    -- Convert potential floats to integers before inserting
    table.insert(data, math.floor(enemies_state.is_aligned_with_nearest)) -- Explicitly floor 0.0/1.0
    table.insert(data, enemies_state.alignment_error_magnitude) -- Already floored in update

    -- Player state (Integers/Booleans treated as 0/1 later)
    table.insert(data, player_state.position); table.insert(data, player_state.alive); 
    table.insert(data, player_state.player_state); table.insert(data, player_state.player_depth); 
    table.insert(data, player_state.superzapper_uses); table.insert(data, player_state.superzapper_active); 
    table.insert(data, player_state.shot_count);
    for i = 1, 8 do table.insert(data, player_state.shot_positions[i] or 0) end
    for i = 1, 8 do table.insert(data, player_state.shot_segments[i] or SegmentUtils.INVALID_SEGMENT) end

    -- Level state (Integers)
    table.insert(data, level_state.level_number); table.insert(data, level_state.level_shape); 
    for i = 0, 15 do table.insert(data, level_state.spike_heights[i] or 0) end
    for i = 0, 15 do table.insert(data, level_state.level_angles[i] or 0) end

    -- Enemies state counts (Integers)
    table.insert(data, enemies_state.active_flippers); table.insert(data, enemies_state.active_pulsars);
    table.insert(data, enemies_state.active_tankers); table.insert(data, enemies_state.active_spikers);
    table.insert(data, enemies_state.active_fuseballs); table.insert(data, enemies_state.spawn_slots_flippers);
    table.insert(data, enemies_state.spawn_slots_pulsars); table.insert(data, enemies_state.spawn_slots_tankers);
    table.insert(data, enemies_state.spawn_slots_spikers); table.insert(data, enemies_state.spawn_slots_fuseballs);
    table.insert(data, enemies_state.num_enemies_in_tube); table.insert(data, enemies_state.num_enemies_on_top);
    table.insert(data, enemies_state.enemies_pending); table.insert(data, enemies_state.pulsar_fliprate);

    -- Decoded Enemy Info (Integers/Booleans)
    for i = 1, 7 do
        table.insert(data, enemies_state.enemy_core_type[i] or 0)
        table.insert(data, enemies_state.enemy_direction_moving[i] or 0)
        table.insert(data, enemies_state.enemy_between_segments[i] or 0)
        table.insert(data, enemies_state.enemy_moving_away[i] or 0)
        table.insert(data, enemies_state.enemy_can_shoot[i] or 0)
        table.insert(data, enemies_state.enemy_split_behavior[i] or 0)
    end
    -- Enemy relative segments (Integers)
    for i = 1, 7 do table.insert(data, enemies_state.enemy_segments[i] or SegmentUtils.INVALID_SEGMENT) end
    -- Top Enemy Segments (Integers)
    for i = 1, 7 do 
        if enemies_state.enemy_depths[i] == 0x10 then table.insert(data, enemies_state.enemy_segments[i]) else table.insert(data, SegmentUtils.INVALID_SEGMENT) end 
    end
    -- Enemy depths - Remove LSB calculation, just use MSB
    for i = 1, 7 do 
        -- Explicitly use the integer depth value
        table.insert(data, enemies_state.enemy_depths[i] or 0) 
    end
    -- Enemy shot positions/segments (Integers)
    for i = 1, 4 do table.insert(data, enemies_state.shot_positions[i]) end
    for i = 1, 4 do table.insert(data, enemies_state.enemy_shot_segments[i].value or SegmentUtils.INVALID_SEGMENT) end
    -- Charging fuseballs (Integers/Booleans)
    for i = 1, 16 do table.insert(data, enemies_state.charging_fuseball_segments[i] or 0) end
    -- Pulse state (Integers)
    table.insert(data, enemies_state.pulse_beat or 0); table.insert(data, enemies_state.pulsing or 0);
    -- Pending vid/seg (Integers)
    for i = 1, 64 do table.insert(data, enemies_state.pending_vid[i] or 0) end
    for i = 1, 64 do table.insert(data, enemies_state.pending_seg[i] or SegmentUtils.INVALID_SEGMENT) end

    -- Serialize (as before)
    local binary_data = ""
    for i, value in ipairs(data) do
        local packed_value
        -- Ensure value is treated as a number before bitwise op (should be guaranteed now)
        local num_value = tonumber(value) or 0 
        if num_value < 0 then
             -- Handle negative numbers correctly for packing as unsigned short
             -- Calculate two's complement for 16 bits
             packed_value = 65536 + num_value 
        else
             packed_value = num_value & 0xFFFF
        end
        -- Ensure packed_value is within 0-65535 range before packing
        packed_value = math.max(0, math.min(packed_value, 65535)) 
        binary_data = binary_data .. string.pack(">H", packed_value)
    end

    -- Save signal logic
    local current_time = os.time()
    local save_signal = 0
    if current_time - game_state.last_save_time >= game_state.save_interval or shutdown_requested then
        save_signal = 1; game_state.last_save_time = current_time
        print("Sending save signal to Python script")
        if shutdown_requested then print("SHUTDOWN SAVE: Final save before MAME exits") end
    end

    -- OOB Data Packing
    local is_attract_mode = (game_state.game_mode & 0x80) == 0
    local is_open_level = level_state.is_open_level
    local nearest_abs_seg_oob = enemies_state.nearest_enemy_abs_seg_raw 
    local player_abs_seg_oob = player_state.position & 0x0F 
    local is_enemy_present_oob = (nearest_abs_seg_oob ~= -1) and 1 or 0 
    local score = player_state.score or 0
    local score_high = math.floor(score / 65536); local score_low = score % 65536              
    local frame = game_state.frame_counter % 65536

    -- Use controls object passed as argument
    local oob_data = string.pack(">HdBBBHHHBBBhBhBB",
        #data, reward, 0, game_state.game_mode, bDone and 1 or 0, frame, score_high, score_low,
        save_signal, controls.fire_commanded, controls.zap_commanded, controls.spinner_delta, 
        is_attract_mode and 1 or 0, nearest_abs_seg_oob, player_abs_seg_oob, is_open_level and 1 or 0)
    
    return oob_data .. binary_data, #data
end


return StateUtils 