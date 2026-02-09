package.path = package.path .. ';Scripts/?.lua'

local logic = require('logic')
local state_defs = require('state')

local function make_mem(values)
    return {
        read_u8 = function(_, addr)
            local v = values[addr]
            if v == nil then
                return 0
            end
            return v
        end
    }
end

local values = {}

-- Basic game/player context
values[0x0000] = 0x04 -- Playing
values[0x0005] = 0x80 -- Not attract mode
values[0x0046] = 1
values[0x0048] = 3
values[0x009F] = 1
values[0x0111] = 0xFF -- Closed level
values[0x0200] = 0x00 -- Player at segment 0
values[0x0201] = 0x00
values[0x0202] = 0x10 -- Player at top of tube
values[0x0135] = 1

-- Player shot #1: active with segment 0 (must be treated as valid)
values[0x02D3] = 0x40 -- PlayerShotPositions[1]
values[0x02AD] = 0x00 -- PlayerShotSegments[1]

-- Enemy slot #1: active with segment 0 (must be treated as valid)
values[0x02DF] = 0x10 -- enemy_along[1]
values[0x02B9] = 0x00 -- enemy_seg[1]
values[0x0283] = 0x00 -- enemy_type_info[1] (flipper)
values[0x028A] = 0x40 -- active_enemy_info[1]
values[0x02CC] = 0x00 -- more_enemy_info[1]

-- Enemy shot #1: active with segment 0 (must be treated as valid)
values[0x02DB] = 0x20 -- EnemyShotPositions[1]
values[0x02E6] = 0x80 -- enm_shot_lsb[1]
values[0x02B5] = 0x00 -- EnemyShotSegments[1]

local mem = make_mem(values)

local game_state = state_defs.GameState:new()
local level_state = state_defs.LevelState:new()
local player_state = state_defs.PlayerState:new()
local enemies_state = state_defs.EnemiesState:new()

game_state:update(mem)
level_state:update(mem)
player_state:update(mem, logic.absolute_to_relative_segment)
enemies_state:update(mem, game_state, player_state, level_state, logic.absolute_to_relative_segment)

assert(player_state.shot_positions[1] == 0x40, 'Player shot position should remain active')
assert(player_state.shot_segments[1] == 0, string.format('Expected player shot rel segment 0, got %s', tostring(player_state.shot_segments[1])))

assert(enemies_state.enemy_abs_segments[1] == 0, string.format('Expected enemy abs segment 0, got %s', tostring(enemies_state.enemy_abs_segments[1])))
assert(enemies_state.enemy_segments[1] == 0, string.format('Expected enemy rel segment 0, got %s', tostring(enemies_state.enemy_segments[1])))
assert(enemies_state.enemy_depths[1] == 0x10, string.format('Expected enemy depth 0x10, got 0x%02X', enemies_state.enemy_depths[1] or 0))

assert(enemies_state.enemy_shot_abs_segments[1] == 0, string.format('Expected enemy shot abs segment 0, got %s', tostring(enemies_state.enemy_shot_abs_segments[1])))
assert(enemies_state.enemy_shot_segments[1] == 0, string.format('Expected enemy shot rel segment 0, got %s', tostring(enemies_state.enemy_shot_segments[1])))
assert(enemies_state.shot_positions[1] > 0, 'Enemy shot position should remain active')

print('state extraction tests passed')
