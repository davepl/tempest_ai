-- Robotron baseline logic module (stub).
-- Detailed reward shaping and expert logic will be added after real memory extraction is wired.

local M = {}

local _last_reward = 0.0

function M.absolute_to_relative_segment(_a, _b, _is_open)
    return 0
end

function M.find_target_segment(_game_state, _player_state, _level_state, _enemies_state, _abs_to_rel_func, _is_open)
    return -1, 255, false, false
end

function M.calculate_reward(_game_state, _level_state, _player_state, _enemies_state, _abs_to_rel_func)
    _last_reward = 0.0
    return 0.0, 0.0, 0.0, false
end

function M.getLastReward()
    return _last_reward
end

return M
