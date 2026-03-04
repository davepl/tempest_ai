-- Robotron baseline state module (stub).
-- This intentionally exposes only minimal objects while Robotron memory mapping is in progress.

local M = {}

M.INVALID_SEGMENT = -32768

M.GameState = {}
M.GameState.__index = M.GameState

function M.GameState:new()
    local self = setmetatable({}, M.GameState)
    self.frame_counter = 0
    self.current_fps = 0
    self.last_save_time = os.time()
    self.save_interval = 300
    return self
end

function M.GameState:update(_mem)
    self.frame_counter = self.frame_counter + 1
end

M.LevelState = {}
M.LevelState.__index = M.LevelState

function M.LevelState:new()
    local self = setmetatable({}, M.LevelState)
    self.level_number = 0
    self.level_type = 0
    return self
end

function M.LevelState:update(_mem)
end

M.PlayerState = {}
M.PlayerState.__index = M.PlayerState

function M.PlayerState:new()
    local self = setmetatable({}, M.PlayerState)
    self.alive = 1
    self.score = 0
    self.move_direction_commanded = 0
    self.fire_direction_commanded = 0
    return self
end

function M.PlayerState:update(_mem, _abs_to_rel_func)
end

M.EnemiesState = {}
M.EnemiesState.__index = M.EnemiesState

function M.EnemiesState:new()
    local self = setmetatable({}, M.EnemiesState)
    return self
end

function M.EnemiesState:update(_mem, _game_state, _player_state, _level_state, _abs_to_rel_func)
end

return M
