--[[
    Robotron AI Lua script for MAME (baseline bring-up).

    Current scope:
      - Sends a minimal 2-value state vector: [PlayerAlive, Score]
      - Receives dual 8-way joystick commands: movement_dir, firing_dir
      - Uses dummy/startup-safe placeholders where Robotron wiring is still unknown
--]]

local SOCKET_ADDRESS = "socket.ubvmdell:9999"
local SOCKET_READ_TIMEOUT_S = 3.5
local CONNECTION_RETRY_INTERVAL_S = 5.0
local SAVE_INTERVAL_S = 300
local unpack = table.unpack or unpack

local mainCpu = nil
local mem = nil
local controls = nil

local current_socket = nil
local last_connection_attempt_time = 0
local shutdown_requested = false

local frame_counter = 0
local last_save_time = 0
local previous_player_alive = 1
local previous_score = 0

-- Dummy placeholders for Robotron memory extraction.
-- These intentionally keep the pipeline alive until real memory mapping is wired.
local DUMMY_PLAYER_ALIVE = 1
local DUMMY_SCORE = 0

local function read_player_alive(_memory)
    return DUMMY_PLAYER_ALIVE
end

local function read_player_score(_memory)
    return DUMMY_SCORE
end

local function initialize_mame_interface()
    local success, err = pcall(function()
        if not manager or not manager.machine then
            error("MAME manager.machine not available")
        end
        mainCpu = manager.machine.devices[":maincpu"]
        if not mainCpu then
            error("Main CPU not found")
        end
        mem = mainCpu.spaces["program"]
        if not mem then
            error("Program memory space not found")
        end
    end)

    if not success then
        print("Error accessing MAME interface: " .. tostring(err))
        return false
    end

    print("MAME interface initialized.")
    return true
end

local function close_socket()
    if current_socket then
        current_socket:close()
        current_socket = nil
    end
end

local function open_socket()
    close_socket()

    local ok, err = pcall(function()
        local sock = emu.file("rw")
        local result = sock:open(SOCKET_ADDRESS)
        if result == nil then
            -- Required 2-byte handshake.
            sock:write(string.pack(">H", 0))
            current_socket = sock
        else
            sock:close()
        end
    end)

    if not ok or not current_socket then
        if err then
            print("Socket open error: " .. tostring(err))
        end
        close_socket()
        return false
    end

    print("Socket connection opened to " .. SOCKET_ADDRESS)
    return true
end

local function find_field(ioport, field_name_options)
    local candidate_ports = {
        ":IN0", ":IN1", ":IN2", ":IN3", ":IN4", ":IN5",
        ":P1", ":P1JOY", ":P1BUTTONS", ":JOYSTICK1", ":JOYSTICK2", ":BUTTONSP1"
    }

    for _, port_name in ipairs(candidate_ports) do
        local port = ioport.ports[port_name]
        if port then
            for _, field_name in ipairs(field_name_options) do
                local field = port.fields[field_name]
                if field then
                    return field
                end
            end
        end
    end

    return nil
end

local Controls = {}
Controls.__index = Controls

function Controls:new(mame_manager)
    local self = setmetatable({}, Controls)
    local ioport = mame_manager.machine.ioport

    self.move_up = find_field(ioport, {"P1 Move Up", "P1 Left Up", "P1 Joystick Up"})
    self.move_down = find_field(ioport, {"P1 Move Down", "P1 Left Down", "P1 Joystick Down"})
    self.move_left = find_field(ioport, {"P1 Move Left", "P1 Left Left", "P1 Joystick Left"})
    self.move_right = find_field(ioport, {"P1 Move Right", "P1 Left Right", "P1 Joystick Right"})

    self.fire_up = find_field(ioport, {"P1 Fire Up", "P1 Right Up"})
    self.fire_down = find_field(ioport, {"P1 Fire Down", "P1 Right Down"})
    self.fire_left = find_field(ioport, {"P1 Fire Left", "P1 Right Left"})
    self.fire_right = find_field(ioport, {"P1 Fire Right", "P1 Right Right"})

    self.p1_start = find_field(ioport, {"1 Player Start", "P1 Start", "Start 1"})
    self.coin_1 = find_field(ioport, {"Coin 1", "P1 Coin", "Insert Coin"})

    if not self.p1_start then
        print("TODO: Robotron start input field not mapped yet (need exact MAME field name).")
    end
    if not self.coin_1 then
        print("TODO: Robotron coin input field not mapped yet (need exact MAME field name).")
    end

    return self
end

local DIR_AXES = {
    [0] = {1, 0, 0, 0}, -- up
    [1] = {1, 0, 0, 1}, -- up-right
    [2] = {0, 0, 0, 1}, -- right
    [3] = {0, 1, 0, 1}, -- down-right
    [4] = {0, 1, 0, 0}, -- down
    [5] = {0, 1, 1, 0}, -- down-left
    [6] = {0, 0, 1, 0}, -- left
    [7] = {1, 0, 1, 0}, -- up-left
}

local function apply_direction(up_field, down_field, left_field, right_field, dir_idx)
    local axis = DIR_AXES[math.max(0, math.min(7, dir_idx or 0))] or DIR_AXES[0]

    if up_field then up_field:set_value(axis[1]) end
    if down_field then down_field:set_value(axis[2]) end
    if left_field then left_field:set_value(axis[3]) end
    if right_field then right_field:set_value(axis[4]) end
end

function Controls:apply_action(move_dir, fire_dir, start_cmd, coin_cmd)
    apply_direction(self.move_up, self.move_down, self.move_left, self.move_right, move_dir)
    apply_direction(self.fire_up, self.fire_down, self.fire_left, self.fire_right, fire_dir)

    if self.p1_start then
        self.p1_start:set_value(start_cmd)
    end
    if self.coin_1 then
        self.coin_1:set_value(coin_cmd)
    end
end

local function serialize_frame(player_alive, score, done, subj_reward, obj_reward, save_signal)
    local num_values = 2
    local score_u32 = math.max(0, math.min(4294967295, math.floor(score or 0)))

    local header = string.pack(
        ">HddBIBB",
        num_values,
        subj_reward,
        obj_reward,
        done and 1 or 0,
        score_u32,
        player_alive,
        save_signal
    )

    local state_payload = string.pack(">ff", player_alive, score)
    return header .. state_payload
end

local function process_frame_via_socket(frame_payload)
    if not current_socket then
        if not open_socket() then
            return 0, 0, false
        end
    end

    local write_ok, write_err = pcall(function()
        local length_header = string.pack(">H", #frame_payload)
        current_socket:write(length_header .. frame_payload)
    end)

    if not write_ok then
        print("Socket write error: " .. tostring(write_err))
        close_socket()
        return 0, 0, false
    end

    local read_ok, read_result = pcall(function()
        local started = os.clock()
        while (os.clock() - started) < SOCKET_READ_TIMEOUT_S do
            local action_bytes = current_socket:read(2)
            if action_bytes and #action_bytes == 2 then
                local move_dir, fire_dir = string.unpack("bb", action_bytes)
                return {move_dir, fire_dir}
            end
        end
        return {0, 0}
    end)

    if not read_ok then
        print("Socket read error: " .. tostring(read_result))
        close_socket()
        return 0, 0, false
    end

    local move_dir, fire_dir = unpack(read_result)
    return move_dir or 0, fire_dir or 0, true
end

local function determine_meta_commands(player_alive, frame_idx)
    local start_cmd = 0
    local coin_cmd = 0

    -- TODO: Robotron attract/game-state memory mapping is still missing.
    -- Placeholder behavior: if player is not alive, pulse coin/start periodically.
    if player_alive == 0 then
        if frame_idx % 180 == 0 then coin_cmd = 1 end
        if frame_idx % 120 == 0 then start_cmd = 1 end
    end

    return start_cmd, coin_cmd
end

local function frame_callback()
    if not mem or not controls then
        return true
    end

    local player_alive = (read_player_alive(mem) ~= 0) and 1 or 0
    local score = math.max(0, math.floor(read_player_score(mem) or 0))

    local done = (previous_player_alive == 1 and player_alive == 0)
    local obj_reward = score - previous_score
    local subj_reward = 0.0

    local now = os.time()
    local save_signal = 0
    if shutdown_requested or (now - last_save_time) >= SAVE_INTERVAL_S then
        save_signal = 1
        last_save_time = now
    end

    local payload = serialize_frame(player_alive, score, done, subj_reward, obj_reward, save_signal)

    local move_cmd, fire_cmd = 0, 0
    local socket_ok = false
    if current_socket then
        move_cmd, fire_cmd, socket_ok = process_frame_via_socket(payload)
    else
        if (now - last_connection_attempt_time) >= CONNECTION_RETRY_INTERVAL_S then
            last_connection_attempt_time = now
            open_socket()
        end
        move_cmd, fire_cmd = 0, 0
    end

    if not socket_ok then
        move_cmd, fire_cmd = 0, 0
    end

    local start_cmd, coin_cmd = determine_meta_commands(player_alive, frame_counter)
    controls:apply_action(move_cmd, fire_cmd, start_cmd, coin_cmd)

    previous_player_alive = player_alive
    previous_score = score
    frame_counter = frame_counter + 1

    return true
end

local function on_mame_exit()
    shutdown_requested = true
    close_socket()
    print("Robotron AI Lua script shutting down.")
end

math.randomseed(os.time())

if not initialize_mame_interface() then
    return
end

controls = Controls:new(manager)
open_socket()

last_save_time = os.time()
previous_player_alive = (read_player_alive(mem) ~= 0) and 1 or 0
previous_score = math.max(0, math.floor(read_player_score(mem) or 0))

global_callback_ref = emu.add_machine_frame_notifier(frame_callback)
emu.add_machine_stop_notifier(on_mame_exit)

print("Robotron AI Lua script initialized.")
