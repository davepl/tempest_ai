-- Initialize variables
local frameCount = 0
local start_frame = 100  -- Start inserting coins at frame 100
local interval = 50      -- Insert a coin every 50 frames
local press_duration = 10 -- Hold coin input active for 10 frames

-- Function to manage coin input
local function manageCoinInput()
    -- Access the input port ":IN0" and the "Coin 1" field
    local port = manager.machine.ioport.ports[":IN0"]
    local coinField = port.fields["Coin 1"]

    -- Check if the field exists
    if not coinField then
        print("Error: 'Coin 1' field not found")
        return
    end

    -- Manage coin input after start_frame
    if frameCount >= start_frame then
        local offset = (frameCount - start_frame) % interval
        if offset < press_duration then
            coinField:set_value(1)  -- Coin active
            if offset == 0 then
                print("Coin 1 pressed at frame " .. frameCount)
            end
        else
            coinField:set_value(0)  -- Coin inactive
            if offset == press_duration then
                print("Coin 1 released at frame " .. frameCount)
            end
        end
    end

    -- Increment frame counter
    frameCount = frameCount + 1
end

-- Register the function to run every frame
emu.register_frame(manageCoinInput, "frame")