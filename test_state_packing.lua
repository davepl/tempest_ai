-- Simple test script to trigger state packing and see debug output
package.path = package.path .. ";Scripts/?.lua"

-- Mock state objects with extreme values to test normalization
local mock_gs = {
    gamestate = -1.62e+32,  -- Extreme negative value
    game_mode = 0,
    countdown_timer = 0,
    p1_lives = 0,
    p1_level = -8.0  -- Negative value
}

local mock_ps = {
    position = 0,
    alive = 0,
    player_state = 0,
    player_depth = 0,
    superzapper_uses = 0,
    superzapper_active = 0,
    shot_count = 0,
    shot_positions = { -11.8, 0, 0, 0, 0, 0, 0, 0 },  -- Extreme negative
    shot_segments = { 0, 0, 0, 0, 0, 0, 0, 0 }
}

local mock_ls = {
    level_number = 0,
    level_type = 0,
    level_shape = 0,
    spike_heights = {},
    level_angles = {}
}
for i = 0, 15 do
    mock_ls.spike_heights[i] = 0
    mock_ls.level_angles[i] = 0
end
-- Set some extreme values for level_angles
mock_ls.level_angles[4] = 1.62e+32  -- Extreme positive
mock_ls.level_angles[5] = -1.62e+32  -- Extreme negative

local mock_es = {
    active_flippers = 0,
    active_pulsars = -8.0,  -- Negative value
    active_tankers = 0,
    active_spikers = 0,
    active_fuseballs = 0,
    spawn_slots_flippers = 0,
    spawn_slots_pulsars = 0,
    spawn_slots_tankers = 0,
    spawn_slots_spikers = 0,
    spawn_slots_fuseballs = 0,
    num_enemies_in_tube = 0,
    num_enemies_on_top = 0,
    enemies_pending = 0,
    pulsar_fliprate = 0,
    pulse_beat = 0,
    pulsing = 0,
    enemy_core_type = {},
    enemy_direction_moving = {},
    enemy_between_segments = {},
    enemy_moving_away = {},
    enemy_can_shoot = {},
    enemy_split_behavior = {},
    enemy_segments = {},
    enemy_depths = {},
    enemy_shot_segments = {},
    charging_fuseball = {},
    active_pulsar = {}
}

for i = 1, 7 do
    mock_es.enemy_core_type[i] = 0
    mock_es.enemy_direction_moving[i] = 0
    mock_es.enemy_between_segments[i] = 0
    mock_es.enemy_moving_away[i] = 0
    mock_es.enemy_can_shoot[i] = 0
    mock_es.enemy_split_behavior[i] = 0
    mock_es.enemy_segments[i] = 0
    mock_es.enemy_depths[i] = 0
    mock_es.enemy_shot_segments[i] = 0
    mock_es.charging_fuseball[i] = 0
    mock_es.active_pulsar[i] = 0
end

-- Load the main script
local main = require("main")

-- Try to call the flatten function
local success, result = pcall(main.flatten_game_state_to_binary, 0, 0, 0, mock_gs, mock_ls, mock_ps, mock_es, false, 0, 0, 0)
if not success then
    print("Error calling flatten_game_state_to_binary: " .. result)
else
    print("State packing completed successfully")
end