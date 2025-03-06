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
local gamestate_addr = 0x0000        -- Game state
local player_seg_addr = 0x0200       -- Player's segment position (0-15)
local status_flags_addr = 0x0005     -- Status flags (bit 6 indicates if player is alive)
local zap_uses_addr = 0x03aa         -- Superzapper uses (0-2)
local ply_shotcnt_addr = 0x0135      -- Number of active player shots (0-8)

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
    print(string.format("Gamestate=0x%02X, Position=%d, Alive=%d, Superzapper Uses=%d, Active Shots=%d",
        gamestate, player_position, alive, superzapper_uses, active_shots))
end

-- Register the function to run every frame
emu.register_frame(print_player_info, "frame")