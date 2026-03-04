package.path = package.path .. ';Scripts/?.lua'

local logic = require('logic')
local state_defs = require('state')

local function make_enemies()
    local enemies = {
        enemy_depths = {},
        enemy_abs_segments = {},
        enemy_core_type = {},
        enemy_segments = {},
        enemy_between_segments = {},
        active_top_rail_enemies = {},
        enemy_shot_abs_segments = {},
        enemy_shot_segments = {},
        shot_positions = {},
    }
    for i = 1, 7 do
        enemies.enemy_depths[i] = 0
        enemies.enemy_abs_segments[i] = state_defs.INVALID_SEGMENT
        enemies.enemy_core_type[i] = 0
        enemies.enemy_segments[i] = 0
        enemies.enemy_between_segments[i] = 0
        enemies.active_top_rail_enemies[i] = 0
    end
    for i = 1, 4 do
        enemies.enemy_shot_abs_segments[i] = state_defs.INVALID_SEGMENT
        enemies.enemy_shot_segments[i] = state_defs.INVALID_SEGMENT
        enemies.shot_positions[i] = 0
    end
    return enemies
end

local function base_states()
    local game_state = { gamestate = 0x04 }
    local level_state = { level_type = 0xFF, level_number = 1 }
    local player_state = { position = 0, shot_count = 0, superzapper_uses = 0 }
    local enemies_state = make_enemies()
    return game_state, level_state, player_state, enemies_state
end

-- Test 1: Top-rail flipper on the right should cause a left sidestep
local game_state, level_state, player_state, enemies_state = base_states()
enemies_state.enemy_depths[1] = 0x10
enemies_state.enemy_abs_segments[1] = 1
enemies_state.enemy_segments[1] = 1
enemies_state.enemy_core_type[1] = 0
enemies_state.active_top_rail_enemies[1] = 1
local target_seg = select(1, logic.find_target_segment(game_state, player_state, level_state, enemies_state, logic.absolute_to_relative_segment))
assert(target_seg == 15, string.format('Expected sidestep to segment 15, got %d', target_seg))

-- Test 2: Low ammo should keep the player in place even if a hunt target exists
local game_state2, level_state2, player_state2, enemies_state2 = base_states()
player_state2.shot_count = 7  -- Only one shot remaining
enemies_state2.enemy_depths[2] = 0x40
enemies_state2.enemy_abs_segments[2] = 5
enemies_state2.enemy_segments[2] = 5
enemies_state2.enemy_core_type[2] = 2
local target_seg_low_ammo = select(1, logic.find_target_segment(game_state2, player_state2, level_state2, enemies_state2, logic.absolute_to_relative_segment))
assert(target_seg_low_ammo == player_state2.position, 'Low ammo should keep the player anchored')

-- Test 3: Incoming shot near the player triggers defensive behavior
local game_state3, level_state3, player_state3, enemies_state3 = base_states()
enemies_state3.enemy_shot_abs_segments[1] = player_state3.position
enemies_state3.enemy_shot_segments[1] = 0
enemies_state3.shot_positions[1] = 0x20
local target_seg_shot = select(1, logic.find_target_segment(game_state3, player_state3, level_state3, enemies_state3, logic.absolute_to_relative_segment))
assert(target_seg_shot ~= player_state3.position, 'Shot threat should force a sidestep')

print('avoidance logic tests passed')
