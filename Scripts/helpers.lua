-- Shared constants and helper functions for Tempest AI

local helpers = {}

helpers.ENEMY_TYPE_FLIPPER  = 0
helpers.ENEMY_TYPE_PULSAR   = 1
helpers.ENEMY_TYPE_TANKER   = 2
helpers.ENEMY_TYPE_SPIKER   = 3
helpers.ENEMY_TYPE_FUSEBALL = 4
helpers.ENEMY_TYPE_MASK = 0x07
helpers.INVALID_SEGMENT = -32768

-- Helper: BCD to decimal
function helpers.bcd_to_decimal(bcd)
    if type(bcd) ~= 'number' then return 0 end
    return math.floor(((bcd / 16) % 16) * 10 + (bcd % 16))
end

return helpers
