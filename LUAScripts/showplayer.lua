-- Access the main CPU device
local maincpu = manager.machine.devices[":maincpu"]
if not maincpu then
    print("Error: Main CPU not found")
    return
end

-- Access the program memory space of the main CPU
local mem = maincpu.spaces["program"]
if not mem then
    print("Error: Program memory space not found")
    return
end

-- Define memory addresses for player information
local gamestate_addr    = 0x0000        -- Game state
local player_seg_addr   = 0x0200        -- Player's segment position (0-15)
local status_flags_addr = 0x0005        -- Status flags (bit 6 indicates if player is alive)
local zap_uses_addr     = 0x03aa        -- Superzapper uses (0-2)
local ply_shotcnt_addr  = 0x0135        -- Number of active player shots (0-8)
local player_state      = 0x0201        -- Player state
local player_depth      = 0x0202        -- Player depth in tube

local remain_flippers   = 0x013D        -- NUmber of flippers yet to spawn
local remain_pulsars    = 0x013E        -- Number of pulsars yet to spawn
local remain_tankers    = 0x013F        -- Number of tankers yet to spawn
local remain_spikers    = 0x0140        -- Number of spikers yet to spawn
local remain_fuseballs  = 0x0141        -- Number of fuseballs yet to spawn

local active_flippers   = 0x0142        -- Number of active flippers
local active_pulsars    = 0x0143        -- Number of active pulsars
local active_tankers    = 0x0144        -- Number of active tankers
local active_spikers    = 0x0145        -- Number of active spikers
local active_fuseballs  = 0x0146        -- Number of active fuseballs

local pulse_beat        = 0x0147        -- Pulse beat counter

-- Function to read and print player information
local function print_player_info()
    -- Read values from memory
    local gamestate = mem:read_u8(gamestate_addr)
    local player_position = mem:read_u8(player_seg_addr)
    local status_flags = mem:read_u8(status_flags_addr)
    local alive = (status_flags & 0x40) ~= 0 and 1 or 0  -- Check bit 6 for alive status
    local superzapper_uses = mem:read_u8(zap_uses_addr)
    local active_shots = mem:read_u8(ply_shotcnt_addr)
    
    -- Print the information in a formatted string
    print(string.format("Gamestate=0x%02X, Position=%d, Alive=%d, SZ Uses=%d, Active Shots=%d",
        gamestate, player_position, alive, superzapper_uses, active_shots))
end

-- Register the function to run every frame
emu.register_frame(print_player_info, "frame")