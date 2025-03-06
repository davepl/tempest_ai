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

-- Define the memory address where credits are stored
local credit_address = 0x0006

-- Function to read and print the current number of credits
local function print_credits()
    local credits = mem:read_u8(credit_address)
    print("Credits: " .. credits)
end

-- Register the function to run every frame
emu.register_frame(print_credits, "frame")