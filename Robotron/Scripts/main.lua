--[[
    Robotron AI Lua script for MAME.

    Current scope:
      - Sends a 644-value state vector:
        5 core stats (alive, score, replay, lasers, wave)
        + 2 player position (x16, y16)
        + 2 player velocity (xv, yv)
        + 50 ELIST enemy state bytes
        + 9 per-type entity categories, each: 1 occupancy + N slots × 4 features
          Per-slot features: present, x, y, distance
          Slots sorted by distance to player (nearest first).
      - Receives joystick commands: movement_dir (-1 neutral or 0..7) and firing_dir (0..7)

    Entity categories (type is implicit in category position):
      0. grunt      (40 slots) - grunts                          peak 80
      1. hulk       (16 slots) - indestructible hulks             peak 25
      2. brain      (16 slots) - brains                          peak 25
      3. tank       ( 8 slots) - tanks (growing + full)          peak ~14
      4. spawner    ( 8 slots) - circles, squares/quarks         peak 14
      5. enforcer   (12 slots) - enforcers                       peak ~10
      6. projectile (12 slots) - sparks, shells, cruise, progs
      7. human      (16 slots) - mom, dad, kid                   peak 30
      8. electrode  (16 slots) - electrodes/posts                peak 25

    Entity type is determined by OCVECT (collision handler address at
    object offset $08), which is stable for an entity's lifetime and
    unique per type.  Classification is auto-discovered at runtime.
--]]

local RAW_SOCKET_ADDRESS = os.getenv("ROBOTRON_SOCKET_ADDRESS") or "ubvmdell:9998"
local SOCKET_ADDRESS = RAW_SOCKET_ADDRESS
if string.sub(SOCKET_ADDRESS, 1, 7) ~= "socket." then
    SOCKET_ADDRESS = "socket." .. SOCKET_ADDRESS
end
local SOCKET_READ_TIMEOUT_S = 3.5
local CONNECTION_RETRY_INTERVAL_S = 1.0
local SAVE_INTERVAL_S = 300
local unpack = table.unpack or unpack

-- Startup diagnostics (bounded, opt-in style flags kept local to script).
local DEBUG_STARTUP_TRACE = false
local DEBUG_TRACE_FRAMES = 10
local DEBUG_BYPASS_SOCKET_FOR_FRAMES = 0
local DEBUG_TRACE_FILE = "logs/startup_trace.log"
local DEBUG_FORCE_ACTION_FRAMES = 0
local DEBUG_FORCE_MOVE_DIR = 2  -- right
local DEBUG_FORCE_FIRE_DIR = 2  -- right
local DEATH_PENALTY_POINTS = 2500
-- Subjective shaping rewards (raw points; scaled in Python by subj_reward_scale).
-- Goal: densify survival signal without dominating objective score rewards.
local SUBJ_ENEMY_WEIGHT = 8.0
local SUBJ_HUMAN_WEIGHT = 12.0
local SUBJ_SURVIVAL_BONUS = 2.0
local SUBJ_DEATH_PENALTY = 25.0
local SUBJ_ENEMY_NEAR_NORM = 0.035
local SUBJ_ENEMY_FAR_NORM = 0.200
local SUBJ_HUMAN_NEAR_NORM = 0.120

-- Aiming reward: bonus for firing toward aligned enemies/obstacles.
local SUBJ_AIM_WEIGHT = 15.0        -- reward per frame when correctly aimed
local AIM_CROSS_THRESHOLD = 2048     -- 8 screen-pixels in x16 units (8 * 256)
local AIM_MIN_FORWARD = 1024         -- ~4 screen-pixels minimum forward distance
-- Categories that count as "targets" for aim reward (everything but humans).
local AIM_TARGET_CATS = {
    grunt = true, hulk = true, brain = true, tank = true,
    spawner = true, enforcer = true, projectile = true, electrode = true,
}
-- Direction unit vectors for 8-way fire (dx, dy in x16 coordinates).
-- Screen/world coordinates are y-down (larger y = lower on screen):
-- 0=up (-y), 1=up-right, 2=right, 3=down-right, 4=down (+y), 5=down-left, 6=left, 7=up-left
local FIRE_DIR_VEC = {
    [0] = { 0, -1},   -- up
    [1] = { 1, -1},   -- up-right
    [2] = { 1,  0},   -- right
    [3] = { 1,  1},   -- down-right
    [4] = { 0,  1},   -- down
    [5] = {-1,  1},   -- down-leftf
    [6] = {-1,  0},   -- left
    [7] = {-1, -1},   -- up-left
}
-- Evasion reward: bonus for moving away from nearest enemy when close.
local SUBJ_EVADE_WEIGHT = 10.0       -- reward when moving away from nearest threat
local EVADE_DANGER_NORM  = 0.08      -- only reward evasion when enemy within this normalised dist
local MOVE_DIR_VEC = FIRE_DIR_VEC    -- same 8-way mapping for move directions

-- Wall-hugging penalty: per-axis penalty when within 16 px of a wall.
-- Stacks additively so a corner costs double.
local SUBJ_WALL_PENALTY  = 5.0       -- penalty per wall axis per frame
-- WALL_MARGIN_NORM_X/Y defined after POS_X/Y_RANGE (see below).

-- Abandoned-human penalty: one-shot penalty per surviving human when a wave
-- is cleared.  Encourages the AI to rescue first, kill last.
local SUBJ_ABANDONED_HUMAN = 15.0     -- penalty per unrescued human on wave end

local mainCpu = nil
local mem = nil
local controls = nil

local current_socket = nil
local last_connection_attempt_time = 0
local shutdown_requested = false

local frame_counter = 0
local dead_frame_counter = 0
local last_save_time = 0
local previous_player_alive = 1
local previous_score = 0
local previous_wave_number = 0
local prev_num_humans = 0
local prev_fire_cmd = -1          -- fire direction from previous frame
local prev_move_cmd = -1          -- move direction from previous frame
local prev_aim_objects = nil      -- classified objects from previous frame
local last_action_source = 0      -- 0=none, 1=dqn, 2=epsilon, 3=expert, 4=forced_random

-- Fire-hold state:  The game's LSPROC laser routine (RRG23.ASM) requires
-- the fire joystick to stay in the SAME direction for 3 consecutive frames
-- before it creates a laser.  At high epsilon, random fire directions change
-- every frame, so shots almost never fire.  We hold each fire direction for
-- a minimum of FIRE_HOLD_FRAMES before accepting a new one.
local FIRE_HOLD_FRAMES = 4       -- frames to lock each fire direction (3 = minimum for 1 shot)
local fire_hold_dir   = -1       -- direction currently being held
local fire_hold_count = 0        -- frames remaining in current hold
local prev_aim_px16 = nil         -- player x16 from previous frame
local prev_aim_py16 = nil         -- player y16 from previous frame
local prev_nearest_enemy_x16 = nil
local prev_nearest_enemy_y16 = nil
local prev_nearest_enemy_dist = nil

-- Autoboot input sequence (MAME input level, no game-specific memory logic required).
-- Every cycle: pulse Coin 1, then pulse 1P Start shortly after.
local AUTOBOOT_ENABLED = true
local AUTOBOOT_CYCLE_FRAMES = 300
local AUTOBOOT_COIN_PULSE_FRAMES = 3
local AUTOBOOT_START_DELAY_FRAMES = 18
local AUTOBOOT_START_PULSE_FRAMES = 3

-- Robotron RAM symbols (from Williams map):
-- STATUS = 0x9859
-- In PLAYRV, bit0 gates player control update; on player death PLEND sets STATUS to 0x1B.
local STATUS_ADDR = 0x9859
local STATUS_PLAYER_INACTIVE_MASK = 0x01

-- Player PLDATA symbols (from Williams map generated by lwasm):
--   ZP1SCR = 0xBDE4 (4 bytes packed BCD, MSB first)
--   ZP1RP  = 0xBDE8 (4 bytes packed BCD, MSB first)
--   ZP1LAS = 0xBDEC (1 byte)
--   ZP1WAV = 0xBDED (1 byte)
--   ZP1ENM = 0xBDEE (50 bytes; first 22 mirror ELIST fields, rest reserved)
local ZP1SCR_ADDR = 0xBDE4
local ZP1RP_ADDR = 0xBDE8
local ZP1LAS_ADDR = 0xBDEC
local ZP1WAV_ADDR = 0xBDED
local ZP1ENM_ADDR = 0xBDEE
local ZP1ENM_SIZE = 50

-- Active object-list heads (from Williams base-page RAM map / RRF.ASM).
-- Address layout: OPTR($9817), OBPTR($9819), OFREE($981B), SPFREE($981D),
--                 HPTR($981F), RPTR($9821), PPTR($9823)
local OPTR_ADDR = 0x9817  -- motion objects (enforcers, sparks, circles, squares, shells, player lasers)
local HPTR_ADDR = 0x981F  -- humans to rescue (mom, dad, kid)
local RPTR_ADDR = 0x9821  -- robots/enemies (grunts, brains, hulks, tanks, progs, cruise missiles)
local PPTR_ADDR = 0x9823  -- fatal obstacles (electrodes)

-- Player object structure (PLOBJ at $985A from RRF.ASM).
local PLOBJ_ADDR = 0x985A
local PX16_ADDR = 0x9864   -- player 16-bit X world coordinate
local PY16_ADDR = 0x9866   -- player 16-bit Y world coordinate
local PXV_ADDR = 0x9868    -- player X velocity
local PYV_ADDR = 0x986A    -- player Y velocity

-- Object pool geometry (master list @ OLIST).
local OLIST_START = 0x9900
local OLIST_ENTRY_SIZE = 0x18
local OLIST_CAPACITY = 180
local OLIST_END = OLIST_START + (OLIST_ENTRY_SIZE * OLIST_CAPACITY)

-- Object entry offsets.
local OLINK_OFF = 0x00
local OPICT_OFF = 0x02
local OBJX_OFF = 0x04
local OBJY_OFF = 0x05
local OX16_OFF = 0x0A
local OY16_OFF = 0x0C

local MAX_LIST_WALK = 256
local OCVECT_OFF = 0x08  -- collision routine address (stable per entity type)

-- Object list head pointers (used to walk linked lists).
local ACTIVE_LISTS = {
    {name = "optr", addr = OPTR_ADDR},   -- motion objects: enforcers, sparks, circles, squares, shells, lasers
    {name = "hptr", addr = HPTR_ADDR},   -- humans: mom, dad, kid
    {name = "rptr", addr = RPTR_ADDR},   -- robots: grunts, brains, hulks, tanks, progs, cruise missiles
    {name = "pptr", addr = PPTR_ADDR},   -- fatal: electrodes
}

-- Per-type entity categories.  Order MUST match Python config/aimodel.py.
local ENTITY_CATEGORIES = {
    {name = "grunt",      slots = 40},
    {name = "hulk",       slots = 16},
    {name = "brain",      slots = 16},
    {name = "tank",       slots =  8},
    {name = "spawner",    slots =  8},
    {name = "enforcer",   slots = 12},
    {name = "projectile", slots = 12},
    {name = "human",      slots = 16},
    {name = "electrode",  slots = 16},
}
local ENTITY_FEATURES_PER_SLOT = 4          -- present, x, y, distance
local ENTITY_TOTAL_SLOTS = 0
local ENTITY_TOTAL_FEATURES = 0
for _, cat in ipairs(ENTITY_CATEGORIES) do
    ENTITY_TOTAL_SLOTS = ENTITY_TOTAL_SLOTS + cat.slots
    ENTITY_TOTAL_FEATURES = ENTITY_TOTAL_FEATURES + 1 + cat.slots * ENTITY_FEATURES_PER_SLOT
end
-- State vector: 9 core + 50 ELIST + entity features (585) = 644 floats
local EXPECTED_STATE_VALUES = 9 + ZP1ENM_SIZE + ENTITY_TOTAL_FEATURES

-- OCVECT-based entity classification (auto-discovered at runtime).
local ocvect_category_cache = {}       -- OCVECT address → category name | "skip"
local discovered_tank_ocvect = nil     -- TNKIL address once discovered via growing phase
local unresolved_7x16 = {}             -- {[ocvect] = true} for ambiguous 7×16 on RPTR
local DEBUG_LOG_DISCOVERY = false      -- set true to log OCVECT classification discoveries

-- Debug HUD overlay state (draws entity letters on screen each frame).
local mame_screen = nil                -- MAME screen device for draw_text
local hud_objects = nil                -- last frame's classified object list (reference)
local hud_player_x16 = nil
local hud_player_y16 = nil
local DEBUG_HUD_ENABLED = true         -- set false to disable overlay
local hud_key_code = nil               -- MAME input code for 'H' key (lazy-init)
local hud_key_was_down = false         -- edge-detect so hold doesn't strobe

local function trace_enabled_for_frame(frame_idx)
    if not DEBUG_STARTUP_TRACE then
        return false
    end
    if frame_idx == nil then
        return false
    end
    return frame_idx < DEBUG_TRACE_FRAMES
end

local function trace_log(frame_idx, phase, detail, force)
    if not DEBUG_STARTUP_TRACE then
        return
    end
    local enabled = force or trace_enabled_for_frame(frame_idx)
    if not enabled then
        return
    end
    local ts = os.date("%Y-%m-%d %H:%M:%S")
    local frame_text = (frame_idx == nil) and "-" or tostring(frame_idx)
    local line = string.format("[trace %s] frame=%s phase=%s %s", ts, frame_text, tostring(phase), tostring(detail or ""))
    print(line)

    local fh = io.open(DEBUG_TRACE_FILE, "a")
    if fh then
        fh:write(line .. "\n")
        fh:close()
    end
end

local function read_player_alive(memory)
    local status = memory:read_u8(STATUS_ADDR)
    if (status & STATUS_PLAYER_INACTIVE_MASK) ~= 0 then
        return 0
    end
    return 1
end

local function decode_bcd4(memory, addr)
    local value = 0
    for i = 0, 3 do
        local byte = memory:read_u8(addr + i)
        local hi = (byte >> 4) & 0x0F
        local lo = byte & 0x0F
        -- Treat non-BCD digits as invalid instead of clamping to 9,
        -- which can fabricate very large scores from transient RAM.
        if hi > 9 or lo > 9 then
            return nil
        end
        value = (value * 10) + hi
        value = (value * 10) + lo
    end
    return value
end

local function read_player_score(memory)
    return decode_bcd4(memory, ZP1SCR_ADDR)
end

local function read_next_replay_level(memory)
    return decode_bcd4(memory, ZP1RP_ADDR)
end

local function read_num_lasers(memory)
    return memory:read_u8(ZP1LAS_ADDR)
end

local function read_wave_number(memory)
    return memory:read_u8(ZP1WAV_ADDR)
end

local function read_enemy_state(memory)
    local raw = {}
    for i = 0, ZP1ENM_SIZE - 1 do
        raw[i + 1] = memory:read_u8(ZP1ENM_ADDR + i)
    end

    -- ELIST mirror (first 22 bytes) in order from RRF.ASM.
    local enemy_state = {
        raw = raw,
        robspd = raw[1], rmxspd = raw[2], enfnum = raw[3], enstim = raw[4],
        cdptim = raw[5], hlkspd = raw[6], bshtim = raw[7], brnspd = raw[8],
        tnksht = raw[9], shlspd = raw[10], tdptim = raw[11], sqspd = raw[12],
        robcnt = raw[13], pstcnt = raw[14], momcnt = raw[15], dadcnt = raw[16],
        kidcnt = raw[17], hlkcnt = raw[18], brncnt = raw[19], circnt = raw[20],
        sqcnt = raw[21], tnkcnt = raw[22],
    }

    return enemy_state
end

local function read_u16_be(memory, addr)
    local hi = memory:read_u8(addr)
    local lo = memory:read_u8(addr + 1)
    return ((hi << 8) | lo) & 0xFFFF
end

local function u16_to_i16(v)
    local u = v & 0xFFFF
    if u >= 0x8000 then
        return u - 0x10000
    end
    return u
end

local function norm_i16(v)
    local x = math.max(-32768, math.min(32767, v or 0))
    return (x + 32768) / 65535.0
end

local function clamp01(v)
    if v <= 0.0 then
        return 0.0
    end
    if v >= 1.0 then
        return 1.0
    end
    return v
end

local function clamp11(v)
    if v <= -1.0 then
        return -1.0
    end
    if v >= 1.0 then
        return 1.0
    end
    return v
end

-- Game playfield bounds from RRF.ASM.  Coordinates are 8.8 fixed-point
-- (high byte = screen pixel, low byte = sub-pixel fraction).  Positions
-- are UNSIGNED 16-bit values; treating them as signed creates a discontinuity
-- at pixel 128 that cuts right through the playfield.
local GAME_XMIN   = 7      -- XMIN EQU 7
local GAME_XMAX   = 0x8F   -- XMAX EQU $8F = 143
local GAME_YMIN   = 24     -- YMIN EQU 24
local GAME_YMAX   = 234    -- YMAX EQU 234
local POS_X_MIN   = GAME_XMIN * 256                        -- 1792
local POS_X_RANGE = (GAME_XMAX - GAME_XMIN) * 256          -- 34816
local POS_Y_MIN   = GAME_YMIN * 256                        -- 6144
local POS_Y_RANGE = (GAME_YMAX - GAME_YMIN) * 256          -- 53760
local POS_MAX_DIAG = math.sqrt(POS_X_RANGE * POS_X_RANGE
                             + POS_Y_RANGE * POS_Y_RANGE)  -- ≈64022

local WALL_MARGIN_NORM_X = 4096.0 / POS_X_RANGE  -- 16 px normalised (~0.118)
local WALL_MARGIN_NORM_Y = 4096.0 / POS_Y_RANGE  -- 16 px normalised (~0.076)

local function norm_pos_x(u16)
    return clamp01(((u16 or 0) - POS_X_MIN) / POS_X_RANGE)
end

local function norm_pos_y(u16)
    return clamp01(((u16 or 0) - POS_Y_MIN) / POS_Y_RANGE)
end

-- Relative position: (entity - player) normalised to [-1, +1] over playfield.
-- Enemies to the right/below the player are positive; left/above are negative.
local function rel_pos_x(entity_u16, player_u16)
    return clamp11(((entity_u16 or 0) - (player_u16 or 0)) / POS_X_RANGE)
end

local function rel_pos_y(entity_u16, player_u16)
    return clamp11(((entity_u16 or 0) - (player_u16 or 0)) / POS_Y_RANGE)
end

local function dist_norm(x1, y1, x2, y2)
    local dx = (x2 or 0) - (x1 or 0)
    local dy = (y2 or 0) - (y1 or 0)
    local d2 = (dx * dx) + (dy * dy)
    if d2 <= 0 then
        return 0.0
    end
    return clamp01(math.sqrt(d2) / POS_MAX_DIAG)
end

local function enemy_spacing_score(nearest_dist_norm)
    if nearest_dist_norm == nil then
        return 1.0
    end
    local span = math.max(1e-6, SUBJ_ENEMY_FAR_NORM - SUBJ_ENEMY_NEAR_NORM)
    local t = clamp01((nearest_dist_norm - SUBJ_ENEMY_NEAR_NORM) / span)
    return (2.0 * t) - 1.0
end

local function human_proximity_score(nearest_dist_norm)
    if nearest_dist_norm == nil then
        return 0.0
    end
    return 1.0 - clamp01(nearest_dist_norm / math.max(1e-6, SUBJ_HUMAN_NEAR_NORM))
end

local function compute_aim_reward(fire_cmd, px16, py16, objects)
    -- Returns 0..1 aim score: 1 if fire_cmd is toward at least one aligned target.
    -- Only evaluates if fire_cmd is a valid direction (0-7).
    if fire_cmd == nil or fire_cmd < 0 or fire_cmd > 7 then
        return 0.0
    end
    if px16 == nil or py16 == nil or objects == nil then
        return 0.0
    end

    local vec = FIRE_DIR_VEC[fire_cmd]
    if vec == nil then return 0.0 end
    local vx, vy = vec[1], vec[2]

    -- For diagonal directions the cross-threshold is wider by ~1.414
    local is_diagonal = (vx ~= 0 and vy ~= 0)
    local cross_thresh = is_diagonal and (AIM_CROSS_THRESHOLD * 1.414) or AIM_CROSS_THRESHOLD

    local best_score = 0.0
    for _, obj in ipairs(objects) do
        if obj.category and AIM_TARGET_CATS[obj.category] then
            local dx = obj.x16 - px16
            local dy = obj.y16 - py16

            -- Forward component (dot product with direction vector, unnormalised)
            local forward = dx * vx + dy * vy
            -- Cross component (absolute perpendicular distance, unnormalised)
            local cross = math.abs(dx * vy - dy * vx)

            if forward >= AIM_MIN_FORWARD and cross <= cross_thresh then
                -- Score by inverse distance: closer targets give higher reward
                local dist = math.sqrt(dx * dx + dy * dy)
                local score = clamp01(1.0 - dist / 32768.0)
                if score > best_score then
                    best_score = score
                end
            end
        end
    end
    return best_score
end

local function compute_evasion_reward(move_cmd, px16, py16, enemy_x16, enemy_y16, enemy_dist_norm)
    -- Returns 0..1 evasion score: reward for moving away from nearest enemy when close.
    if move_cmd == nil or move_cmd < 0 or move_cmd > 7 then
        return 0.0
    end
    if px16 == nil or py16 == nil or enemy_x16 == nil or enemy_y16 == nil then
        return 0.0
    end
    if enemy_dist_norm == nil or enemy_dist_norm > EVADE_DANGER_NORM then
        return 0.0  -- not in danger zone, no evasion reward
    end

    local vec = MOVE_DIR_VEC[move_cmd]
    if vec == nil then return 0.0 end

    -- Flee vector: from enemy toward player (direction we WANT to move)
    local flee_x = px16 - enemy_x16
    local flee_y = py16 - enemy_y16
    local flee_len = math.sqrt(flee_x * flee_x + flee_y * flee_y)
    if flee_len < 1.0 then return 0.0 end

    -- Normalise flee vector
    flee_x = flee_x / flee_len
    flee_y = flee_y / flee_len

    -- Dot product with move direction (move vecs are already unit/sqrt2)
    local dot = vec[1] * flee_x + vec[2] * flee_y
    -- dot ranges from -1.414 to +1.414 for diagonals; normalise to 0..1
    local score = clamp01(dot / 1.414)

    -- Scale by proximity: closer = more reward for correct evasion
    local proximity = clamp01(1.0 - enemy_dist_norm / EVADE_DANGER_NORM)
    return score * proximity
end

local function read_player_position(memory)
    -- Positions are UNSIGNED 8.8 fixed-point (high byte = screen pixel).
    -- Velocity fields (OXV/OYV at PLOBJ+$0E/+$10) are never written by
    -- PLAYRV, so they are always zero.  Velocity is computed as frame-to-
    -- frame position delta in serialize_frame() instead.
    local px16 = read_u16_be(memory, PX16_ADDR)   -- unsigned
    local py16 = read_u16_be(memory, PY16_ADDR)   -- unsigned
    return px16, py16
end

-- ── Per-Type Entity Classification ──────────────────────────────────────

local function is_valid_object_ptr(ptr)
    if ptr == nil or ptr == 0 then
        return false
    end
    if ptr < OLIST_START or ptr >= OLIST_END then
        return false
    end
    return ((ptr - OLIST_START) % OLIST_ENTRY_SIZE) == 0
end

local function classify_by_heuristic(list_name, width, height)
    -- Classify an unknown OCVECT by list membership + picture dimensions.
    -- Returns category name, "skip" (player laser), or nil (ambiguous 7×16).
    if list_name == "pptr" then return "electrode" end
    if list_name == "hptr" then return "human" end

    if list_name == "rptr" then
        if width == 5 and height == 13 then return "grunt" end
        if width == 3 and height == 4 then return "projectile" end   -- cruise missile
        -- Growing tank phases: 2×4, 4×7, 4×8, 6×12
        if (width == 2 and height == 4) or
           (width == 4 and (height == 7 or height == 8)) or
           (width == 6 and height == 12) then
            return "tank"
        end
        if width == 7 and height == 16 then return nil end  -- hulk/brain/tank: ambiguous
        return "projectile"  -- prog or other unknown RPTR object
    end

    if list_name == "optr" then
        if width <= 3 and height <= 6 then return "skip" end   -- player laser
        if width == 8 and height == 15 then return "spawner" end
        if width == 4 and height == 7 then return "projectile" end  -- spark / shell
        return "enforcer"  -- growing or full enforcer
    end

    return "skip"
end

local function try_resolve_7x16(all_objects, enemy_state)
    -- Try to resolve ambiguous 7×16 RPTR objects (hulk / brain / tank)
    -- by matching per-OCVECT counts to ELIST counters.
    -- Returns true if any new assignments were made.

    -- Count 7×16 objects on RPTR grouped by OCVECT (excluding already-resolved)
    local ocv_counts = {}
    for _, obj in ipairs(all_objects) do
        if obj.list_name == "rptr" and obj.width == 7 and obj.height == 16
           and unresolved_7x16[obj.ocvect] then
            ocv_counts[obj.ocvect] = (ocv_counts[obj.ocvect] or 0) + 1
        end
    end

    -- Exclude already-discovered tank OCVECT
    local candidates = {}
    for ocv, count in pairs(ocv_counts) do
        if ocv ~= discovered_tank_ocvect then
            candidates[#candidates + 1] = {ocvect = ocv, count = count}
        else
            -- This 7×16 is actually a full-grown tank
            ocvect_category_cache[ocv] = "tank"
            unresolved_7x16[ocv] = nil
            if DEBUG_LOG_DISCOVERY then print(string.format("[DISCOVERY] OCVECT 0x%04X → tank (matches growing-tank OCVECT)", ocv)) end
        end
    end

    if #candidates == 0 then return false end

    local hlkcnt = enemy_state.hlkcnt or 0
    local brncnt = enemy_state.brncnt or 0
    local tnkcnt = enemy_state.tnkcnt or 0

    local changed = false

    -- Fast path: if only one type of 7x16 enemy is present this wave,
    -- we can resolve all candidates directly.
    local only_hulks  = (hlkcnt > 0 and brncnt == 0 and tnkcnt == 0)
    local only_brains = (brncnt > 0 and hlkcnt == 0 and tnkcnt == 0)
    if only_hulks then
        for _, c in ipairs(candidates) do
            ocvect_category_cache[c.ocvect] = "hulk"
            unresolved_7x16[c.ocvect] = nil
            if DEBUG_LOG_DISCOVERY then print(string.format("[DISCOVERY] OCVECT 0x%04X → hulk (only hulks this wave, HLKCNT=%d)", c.ocvect, hlkcnt)) end
        end
        return true
    end
    if only_brains then
        for _, c in ipairs(candidates) do
            ocvect_category_cache[c.ocvect] = "brain"
            unresolved_7x16[c.ocvect] = nil
            if DEBUG_LOG_DISCOVERY then print(string.format("[DISCOVERY] OCVECT 0x%04X → brain (only brains this wave, BRNCNT=%d)", c.ocvect, brncnt)) end
        end
        return true
    end

    -- General case: try to match individual candidate counts to ELIST counters.
    -- Build a set of unresolved total to compare against.
    local total_unresolved = 0
    for _, c in ipairs(candidates) do
        total_unresolved = total_unresolved + c.count
    end

    if #candidates == 1 then
        local c = candidates[1]
        if c.count == hlkcnt and (brncnt == 0 or c.count ~= brncnt) then
            ocvect_category_cache[c.ocvect] = "hulk"
            unresolved_7x16[c.ocvect] = nil
            if DEBUG_LOG_DISCOVERY then print(string.format("[DISCOVERY] OCVECT 0x%04X → hulk (count=%d matches HLKCNT)", c.ocvect, c.count)) end
            changed = true
        elseif c.count == brncnt and (hlkcnt == 0 or c.count ~= hlkcnt) then
            ocvect_category_cache[c.ocvect] = "brain"
            unresolved_7x16[c.ocvect] = nil
            if DEBUG_LOG_DISCOVERY then print(string.format("[DISCOVERY] OCVECT 0x%04X → brain (count=%d matches BRNCNT)", c.ocvect, c.count)) end
            changed = true
        elseif c.count == tnkcnt and (hlkcnt == 0 or c.count ~= tnkcnt) and (brncnt == 0 or c.count ~= tnkcnt) then
            ocvect_category_cache[c.ocvect] = "tank"
            unresolved_7x16[c.ocvect] = nil
            if DEBUG_LOG_DISCOVERY then print(string.format("[DISCOVERY] OCVECT 0x%04X → tank (count=%d matches TNKCNT)", c.ocvect, c.count)) end
            changed = true
        end
    elseif #candidates == 2 then
        local a, b = candidates[1], candidates[2]
        -- Try all permutations of (hulk, brain, tank) for 2 candidates.
        local function try_assign(c1, type1, cnt1, c2, type2, cnt2)
            if c1.count == cnt1 and c2.count == cnt2 then
                ocvect_category_cache[c1.ocvect] = type1
                ocvect_category_cache[c2.ocvect] = type2
                unresolved_7x16[c1.ocvect] = nil
                unresolved_7x16[c2.ocvect] = nil
                if DEBUG_LOG_DISCOVERY then print(string.format("[DISCOVERY] OCVECT 0x%04X → %s, 0x%04X → %s (count match)",
                    c1.ocvect, type1, c2.ocvect, type2)) end
                return true
            end
            return false
        end
        changed = try_assign(a, "hulk", hlkcnt, b, "brain", brncnt)
            or try_assign(a, "brain", brncnt, b, "hulk", hlkcnt)
            or try_assign(a, "hulk", hlkcnt, b, "tank", tnkcnt)
            or try_assign(a, "tank", tnkcnt, b, "hulk", hlkcnt)
            or try_assign(a, "brain", brncnt, b, "tank", tnkcnt)
            or try_assign(a, "tank", tnkcnt, b, "brain", brncnt)
    elseif #candidates == 3 then
        -- Hulk + brain + tank: try matching all 6 permutations
        local perms = {
            {"hulk", hlkcnt, "brain", brncnt, "tank", tnkcnt},
            {"hulk", hlkcnt, "tank", tnkcnt, "brain", brncnt},
            {"brain", brncnt, "hulk", hlkcnt, "tank", tnkcnt},
            {"brain", brncnt, "tank", tnkcnt, "hulk", hlkcnt},
            {"tank", tnkcnt, "hulk", hlkcnt, "brain", brncnt},
            {"tank", tnkcnt, "brain", brncnt, "hulk", hlkcnt},
        }
        local a, b, c = candidates[1], candidates[2], candidates[3]
        for _, p in ipairs(perms) do
            if a.count == p[2] and b.count == p[4] and c.count == p[6] then
                ocvect_category_cache[a.ocvect] = p[1]
                ocvect_category_cache[b.ocvect] = p[3]
                ocvect_category_cache[c.ocvect] = p[5]
                unresolved_7x16[a.ocvect] = nil
                unresolved_7x16[b.ocvect] = nil
                unresolved_7x16[c.ocvect] = nil
                if DEBUG_LOG_DISCOVERY then print(string.format("[DISCOVERY] OCVECT 0x%04X → %s, 0x%04X → %s, 0x%04X → %s (count match)",
                    a.ocvect, p[1], b.ocvect, p[3], c.ocvect, p[5])) end
                changed = true
                break
            end
        end
    end

    return changed
end

local function extract_typed_entities(memory, player_x16, player_y16, enemy_state)
    -- Walk all 4 linked lists, classify each object by OCVECT into 9 typed
    -- categories, sort each by distance, and produce a flat feature vector.

    -- Phase 1: collect raw objects ------------------------------------------
    local all_objects = {}
    for _, list_def in ipairs(ACTIVE_LISTS) do
        local ptr = read_u16_be(memory, list_def.addr)
        local seen = {}
        local steps = 0
        while ptr ~= 0 and steps < MAX_LIST_WALK do
            if seen[ptr] then break end
            seen[ptr] = true
            steps = steps + 1
            if not is_valid_object_ptr(ptr) then break end

            local ocvect = read_u16_be(memory, ptr + OCVECT_OFF)
            -- Skip objects with zero OCVECT (transient/uninitialized)
            if ocvect == 0 then
                ptr = read_u16_be(memory, ptr + OLINK_OFF)
            else

            local x16 = read_u16_be(memory, ptr + OX16_OFF)   -- unsigned 8.8 fixed-point
            local y16 = read_u16_be(memory, ptr + OY16_OFF)
            local pict_ptr = read_u16_be(memory, ptr + OPICT_OFF)

            local width, height = 0, 0
            if pict_ptr > 0 then
                width = memory:read_u8(pict_ptr)
                height = memory:read_u8(pict_ptr + 1)
            end

            local dnorm = 1.0
            if player_x16 and player_y16 then
                dnorm = dist_norm(player_x16, player_y16, x16, y16)
            end

            all_objects[#all_objects + 1] = {
                list_name = list_def.name,
                x16 = x16,
                y16 = y16,
                ocvect = ocvect,
                width = width,
                height = height,
                dist_norm = dnorm,
            }

            ptr = read_u16_be(memory, ptr + OLINK_OFF)

            end  -- end of ocvect ~= 0 guard
        end
    end

    -- Phase 2: classify each object -----------------------------------------
    for _, obj in ipairs(all_objects) do
        local cat = ocvect_category_cache[obj.ocvect]
        if cat == nil then
            cat = classify_by_heuristic(obj.list_name, obj.width, obj.height)
            if cat ~= nil then
                ocvect_category_cache[obj.ocvect] = cat
                if cat == "tank" and obj.width ~= 7 then
                    discovered_tank_ocvect = obj.ocvect
                end
                if cat ~= "skip" then
                    if DEBUG_LOG_DISCOVERY then print(string.format("[DISCOVERY] OCVECT 0x%04X → %s (list=%s dim=%dx%d)",
                                        obj.ocvect, cat, obj.list_name, obj.width, obj.height)) end
                end
            else
                unresolved_7x16[obj.ocvect] = true
            end
        end
        obj.category = ocvect_category_cache[obj.ocvect]
    end

    -- Phase 3: resolve pending 7×16 ambiguities -----------------------------
    if next(unresolved_7x16) then
        if try_resolve_7x16(all_objects, enemy_state) then
            for _, obj in ipairs(all_objects) do
                if obj.category == nil then
                    obj.category = ocvect_category_cache[obj.ocvect]
                end
            end
        end
    end
    -- Any still-unresolved 7×16 → treat as hulk (conservative: avoid)
    for _, obj in ipairs(all_objects) do
        if obj.category == nil and obj.list_name == "rptr" then
            obj.category = "hulk"
        end
    end

    -- Phase 4: bucket, sort, and build features ----------------------------
    local buckets = {}
    for _, cat in ipairs(ENTITY_CATEGORIES) do
        buckets[cat.name] = {}
    end
    for _, obj in ipairs(all_objects) do
        if obj.category and obj.category ~= "skip" and buckets[obj.category] then
            local bucket = buckets[obj.category]
            bucket[#bucket + 1] = obj
        end
    end
    for _, cat in ipairs(ENTITY_CATEGORIES) do
        table.sort(buckets[cat.name], function(a, b)
            return a.dist_norm < b.dist_norm
        end)
        -- Stamp 1-based distance rank so the HUD can display it.
        for i, obj in ipairs(buckets[cat.name]) do
            obj.rank = i
        end
    end

    -- Phase 5: emit feature vector ------------------------------------------
    local features = {}
    local nearest_enemy_dist = nil
    local nearest_human_dist = nil

    for _, cat in ipairs(ENTITY_CATEGORIES) do
        local bucket = buckets[cat.name]
        features[#features + 1] = math.min(1.0, #bucket / cat.slots)
        for i = 1, cat.slots do
            local obj = bucket[i]
            if obj then
                features[#features + 1] = 1.0
                -- Relative position: (entity - player), normalised over playfield range.
                -- Gives direct player-relative direction without the network having
                -- to learn to subtract player_pos from every slot.  Ranges ~[-1,+1].
                features[#features + 1] = rel_pos_x(obj.x16, player_x16)
                features[#features + 1] = rel_pos_y(obj.y16, player_y16)
                features[#features + 1] = obj.dist_norm
            else
                features[#features + 1] = 0.0
                features[#features + 1] = 0.0
                features[#features + 1] = 0.0
                features[#features + 1] = 0.0
            end
        end
        -- Track nearest distances for reward shaping
        if #bucket > 0 then
            local nd = bucket[1].dist_norm
            if cat.name == "human" then
                nearest_human_dist = nd
            elseif nearest_enemy_dist == nil or nd < nearest_enemy_dist then
                nearest_enemy_dist = nd
                nearest_enemy_x16 = bucket[1].x16
                nearest_enemy_y16 = bucket[1].y16
            end
        end
    end

    -- Stash classified objects for the debug HUD (zero-alloc reference swap).
    hud_objects = all_objects
    hud_player_x16 = player_x16
    hud_player_y16 = player_y16

    local num_humans = #buckets["human"]
    return features, nearest_enemy_dist, nearest_human_dist, nearest_enemy_x16, nearest_enemy_y16, num_humans
end

-- ── Debug HUD: draw coloured rings + rank numbers on MAME screen ────────

local CAT_HUD_COLOR = {
    grunt      = 0xFFFF0000,   -- red
    hulk       = 0xFF00FF00,   -- green
    brain      = 0xFFFFFF00,   -- yellow
    tank       = 0xFFFF8000,   -- orange
    spawner    = 0xFFFF00FF,   -- magenta
    enforcer   = 0xFFFFFFFF,   -- white
    projectile = 0xFFFF8080,   -- light red
    human      = 0xFF4080FF,   -- blue
    electrode  = 0xFF808080,   -- grey
}
local HUD_PLAYER_COLOR = 0xFFFFFFFF   -- white ring for player
local HUD_RING_RADIUS  = 8           -- half-size of ring box in screen pixels

local function draw_debug_hud()
    -- Toggle HUD on/off with the 'H' key (edge-triggered).
    local ok_input, inp = pcall(function() return manager.machine.input end)
    if ok_input and inp then
        if not hud_key_code then
            local ok_code, code = pcall(function()
                return inp:code_from_token("KEYCODE_H")
            end)
            if ok_code and code then
                hud_key_code = code
            end
        end
        if hud_key_code then
            local ok_pressed, down = pcall(function()
                return inp:code_pressed(hud_key_code)
            end)
            if ok_pressed then
                if down and not hud_key_was_down then
                    DEBUG_HUD_ENABLED = not DEBUG_HUD_ENABLED
                    print("[HUD] toggled " .. (DEBUG_HUD_ENABLED and "ON" or "OFF"))
                end
                hud_key_was_down = down
            end
        end
    end

    if not DEBUG_HUD_ENABLED then return end

    -- Lazy-init: grab the screen device on first use.
    if not mame_screen then
        local ok, s = pcall(function()
            return manager.machine.screens[":screen"]
        end)
        if ok and s then
            mame_screen = s
            print("[HUD] Screen device acquired: :screen")
        else
            local ok2, s2 = pcall(function()
                for tag, scr in pairs(manager.machine.screens) do
                    print("[HUD] Found screen: " .. tostring(tag))
                    return scr
                end
            end)
            if ok2 and s2 then
                mame_screen = s2
                print("[HUD] Screen device acquired via fallback")
            else
                return
            end
        end
    end

    -- "HUD ACTIVE" banner at top.
    local ok_banner = pcall(function()
        mame_screen:draw_text("center", 0, "HUD ACTIVE", 0xFF00FF00, 0xC0000000)
    end)
    if not ok_banner then
        pcall(function()
            mame_screen:draw_text(100, 0, "HUD ACTIVE", 0xFF00FF00, 0xC0000000)
        end)
    end

    local r = HUD_RING_RADIUS

    -- Helper: draw a diamond (4 lines) to approximate a circle outline.
    -- Draw a diamond with separate horizontal/vertical radii to match sprite aspect ratio.
    local function draw_diamond(cx, cy, rx, ry, color)
        mame_screen:draw_line(cx, cy - ry, cx + rx, cy, color)   -- top to right
        mame_screen:draw_line(cx + rx, cy, cx, cy + ry, color)   -- right to bottom
        mame_screen:draw_line(cx, cy + ry, cx - rx, cy, color)   -- bottom to left
        mame_screen:draw_line(cx - rx, cy, cx, cy - ry, color)   -- left to top
    end

    local PAD = 2   -- extra pixels of clearance around the sprite

    -- Player ring (player sprite is roughly 5×13 pixels)
    -- Color by action source: green=DQN, red=epsilon, blue=expert, white=other
    local player_color = HUD_PLAYER_COLOR
    if last_action_source == 1 then
        player_color = 0xFF00FF00   -- green: DQN
    elseif last_action_source == 2 or last_action_source == 4 then
        player_color = 0xFFFF0000   -- red: epsilon / forced random
    elseif last_action_source == 3 then
        player_color = 0xFF4488FF   -- blue: expert
    end
    if hud_player_x16 and hud_player_y16 then
        local px = ((hud_player_x16 >> 8) & 0xFF) * 2
        local py = (hud_player_y16 >> 8) & 0xFF
        -- Use known player sprite 5×13
        local prx = 5 + PAD
        local pry = math.floor((13 + PAD) / 2)
        pcall(function() draw_diamond(px, py, prx, pry, player_color) end)
    end

    -- Entity rings + rank numbers
    if hud_objects then
        for _, obj in ipairs(hud_objects) do
            if obj.category and obj.category ~= "skip" then
                local color = CAT_HUD_COLOR[obj.category]
                if color then
                    local sx = ((obj.x16 >> 8) & 0xFF) * 2
                    local sy = (obj.y16 >> 8) & 0xFF
                    -- Size diamond to sprite dimensions
                    local w = math.max(obj.width or 4, 4)
                    local h = math.max(obj.height or 4, 4)
                    local rx = w + PAD
                    local ry = math.floor((h + PAD) / 2)
                    pcall(function() draw_diamond(sx, sy, rx, ry, color) end)
                    -- Distance rank number just below the diamond
                    if obj.rank then
                        pcall(function()
                            mame_screen:draw_text(sx - 3, sy + ry + 1,
                                                  tostring(obj.rank), color, 0x00000000)
                        end)
                    end
                end
            end
        end
    end
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

    local open_result = nil
    local ok, err = pcall(function()
        local sock = emu.file("rw")
        local result = sock:open(SOCKET_ADDRESS)
        if result == nil then
            -- Required 2-byte handshake.
            sock:write(string.pack(">H", 0))
            current_socket = sock
        else
            open_result = tostring(result)
            sock:close()
        end
    end)

    if not ok or not current_socket then
        local reason = tostring(err or open_result or "unknown error")
        trace_log(nil, "socket_open_failed", "addr=" .. SOCKET_ADDRESS .. " reason=" .. reason, true)
        print("Socket open failed: " .. SOCKET_ADDRESS .. " (" .. reason .. ")")
        close_socket()
        return false
    end

    trace_log(nil, "socket_open_ok", "addr=" .. SOCKET_ADDRESS, true)
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
                    return field, field_name
                end
            end
        end
    end

    return nil, nil
end

local function find_field_fuzzy(ioport, name_fragments)
    local function label_matches_fragment(label, fragment)
        local ll = string.lower(label or "")
        local ff = string.lower(fragment or "")
        if ff == "" then
            return false
        end
        if string.find(ll, ff, 1, true) then
            return true
        end

        -- Token-aware fallback so "right up" can match "right joystick up".
        -- Count occurrences to avoid false positives like:
        --   fragment="right right" matching label="move right".
        local label_words = {}
        for w in string.gmatch(ll, "%w+") do
            label_words[w] = (label_words[w] or 0) + 1
        end
        local frag_words = {}
        for w in string.gmatch(ff, "%w+") do
            frag_words[w] = (frag_words[w] or 0) + 1
        end
        local have_tokens = false
        for w, count in pairs(frag_words) do
            have_tokens = true
            if (label_words[w] or 0) < count then
                return false
            end
        end
        return have_tokens
    end

    local function any_match(label)
        local ll = string.lower(label or "")
        -- This script is strictly Player-1 control; avoid accidental P2 binds.
        if string.find(ll, "p2", 1, true) then
            return false
        end
        for _, frag in ipairs(name_fragments) do
            if label_matches_fragment(label, frag) then
                return true
            end
        end
        return false
    end

    local candidate_ports = {":IN0", ":IN1", ":IN2", ":IN3", ":IN4", ":IN5", ":P1", ":P2"}
    for _, port_name in ipairs(candidate_ports) do
        local port = ioport.ports[port_name]
        if port and port.fields then
            for label, field in pairs(port.fields) do
                if any_match(label) then
                    return field, label
                end
            end
        end
    end

    for _, port in pairs(ioport.ports) do
        if port and port.fields then
            for label, field in pairs(port.fields) do
                if any_match(label) then
                    return field, label
                end
            end
        end
    end

    return nil, nil
end

local function bind_control(ioport, exact_names, fuzzy_names, tag)
    local field, label = find_field(ioport, exact_names)
    if field then
        print(string.format("Mapped %-10s => %s", tag, tostring(label)))
        return field, label
    end
    field, label = find_field_fuzzy(ioport, fuzzy_names)
    if field then
        print(string.format("Mapped %-10s => %s (fuzzy)", tag, tostring(label)))
        return field, label
    end
    print(string.format("WARN: unmapped control %-10s", tag))
    return nil, nil
end

local Controls = {}
Controls.__index = Controls

function Controls:new(mame_manager)
    local self = setmetatable({}, Controls)
    local ioport = mame_manager.machine.ioport

    self.move_up, self.move_up_label = bind_control(
        ioport,
        {"Move Up", "P1 Move Up", "P1 Left Up", "P1 Joystick Up", "P1 Left Joystick Up", "P1 Left Stick Up"},
        {"move up", "left up", "left stick up", "joystick up"},
        "move_up"
    )
    self.move_down, self.move_down_label = bind_control(
        ioport,
        {"Move Down", "P1 Move Down", "P1 Left Down", "P1 Joystick Down", "P1 Left Joystick Down", "P1 Left Stick Down"},
        {"move down", "left down", "left stick down", "joystick down"},
        "move_down"
    )
    self.move_left, self.move_left_label = bind_control(
        ioport,
        {"Move Left", "P1 Move Left", "P1 Left Left", "P1 Joystick Left", "P1 Left Joystick Left", "P1 Left Stick Left"},
        {"move left", "left left", "left stick left", "joystick left"},
        "move_left"
    )
    self.move_right, self.move_right_label = bind_control(
        ioport,
        {"Move Right", "P1 Move Right", "P1 Left Right", "P1 Joystick Right", "P1 Left Joystick Right", "P1 Left Stick Right"},
        {"move right", "left right", "left stick right", "joystick right"},
        "move_right"
    )

    self.fire_up, self.fire_up_label = bind_control(
        ioport,
        {"Fire Up", "P1 Fire Up", "P1 Right Up", "P1 Right Joystick Up", "P1 Right Stick Up"},
        {"fire up", "right up", "right stick up"},
        "fire_up"
    )
    self.fire_down, self.fire_down_label = bind_control(
        ioport,
        {"Fire Down", "P1 Fire Down", "P1 Right Down", "P1 Right Joystick Down", "P1 Right Stick Down"},
        {"fire down", "right down", "right stick down"},
        "fire_down"
    )
    self.fire_left, self.fire_left_label = bind_control(
        ioport,
        {"Fire Left", "P1 Fire Left", "P1 Right Left", "P1 Right Joystick Left", "P1 Right Stick Left"},
        {"fire left", "right left", "right stick left"},
        "fire_left"
    )
    self.fire_right, self.fire_right_label = bind_control(
        ioport,
        {"Fire Right", "P1 Fire Right", "P1 Right Right", "P1 Right Joystick Right", "P1 Right Stick Right"},
        {"fire right", "right right", "right stick right"},
        "fire_right"
    )

    self.p1_start = bind_control(
        ioport,
        {"1 Player Start", "P1 Start", "Start 1"},
        {"1 player start", "start 1", "p1 start"},
        "start"
    )
    self.coin_1 = bind_control(
        ioport,
        {"Coin 1", "P1 Coin", "Insert Coin"},
        {"coin 1", "insert coin"},
        "coin_1"
    )

    local missing_move = (not self.move_up) or (not self.move_down) or (not self.move_left) or (not self.move_right)
    local missing_fire = (not self.fire_up) or (not self.fire_down) or (not self.fire_left) or (not self.fire_right)
    if missing_move or missing_fire then
        print("FATAL: required joystick mappings are incomplete.")
        print("  This run would generate mostly non-causal training data.")
        print("  Confirm MAME input labels for left/right sticks and update bind_control names.")
        return nil
    end

    local move_fields = {self.move_up, self.move_down, self.move_left, self.move_right}
    local move_labels = {self.move_up_label, self.move_down_label, self.move_left_label, self.move_right_label}
    local fire_fields = {self.fire_up, self.fire_down, self.fire_left, self.fire_right}
    local fire_labels = {self.fire_up_label, self.fire_down_label, self.fire_left_label, self.fire_right_label}
    local dir_names = {"up", "down", "left", "right"}

    for i = 1, 4 do
        for j = i + 1, 4 do
            if move_fields[i] == move_fields[j] then
                print(string.format(
                    "FATAL: move_%s and move_%s mapped to same field (%s / %s).",
                    dir_names[i], dir_names[j], tostring(move_labels[i]), tostring(move_labels[j])
                ))
                return nil
            end
            if fire_fields[i] == fire_fields[j] then
                print(string.format(
                    "FATAL: fire_%s and fire_%s mapped to same field (%s / %s).",
                    dir_names[i], dir_names[j], tostring(fire_labels[i]), tostring(fire_labels[j])
                ))
                return nil
            end
        end
    end
    for i = 1, 4 do
        for j = 1, 4 do
            if move_fields[i] == fire_fields[j] then
                print(string.format(
                    "FATAL: move_%s and fire_%s share the same mapped field (%s / %s).",
                    dir_names[i], dir_names[j], tostring(move_labels[i]), tostring(fire_labels[j])
                ))
                return nil
            end
        end
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
local DIR_NEUTRAL = {0, 0, 0, 0}

local function apply_direction(up_field, down_field, left_field, right_field, dir_idx)
    local axis = DIR_NEUTRAL
    if dir_idx ~= nil and dir_idx >= 0 and dir_idx <= 7 then
        axis = DIR_AXES[dir_idx] or DIR_NEUTRAL
    end

    if up_field then up_field:set_value(axis[1]) end
    if down_field then down_field:set_value(axis[2]) end
    if left_field then left_field:set_value(axis[3]) end
    if right_field then right_field:set_value(axis[4]) end
end

--- Apply fire-hold: keep each fire direction stable for FIRE_HOLD_FRAMES.
-- Returns the effective fire direction actually sent to the game.
local function apply_fire_hold(requested_dir)
    if requested_dir < 0 then
        -- No fire requested (e.g. player dead) – release immediately
        fire_hold_dir   = -1
        fire_hold_count = 0
        return -1
    end
    if requested_dir == fire_hold_dir then
        -- Same direction requested – keep holding, reset counter
        fire_hold_count = FIRE_HOLD_FRAMES
        return fire_hold_dir
    end
    if fire_hold_count > 0 then
        -- Still in hold period for previous direction, decrement and keep old
        fire_hold_count = fire_hold_count - 1
        return fire_hold_dir
    end
    -- Hold expired (or first fire) – accept new direction
    fire_hold_dir   = requested_dir
    fire_hold_count = FIRE_HOLD_FRAMES - 1   -- this frame counts as 1
    return requested_dir
end

function Controls:apply_action(move_dir, fire_dir, start_cmd, coin_cmd)
    apply_direction(self.move_up, self.move_down, self.move_left, self.move_right, move_dir)
    -- Fire hold is now applied Python-side (socket_server.py) so the replay
    -- buffer stores the effective action, not the model's raw request.
    -- The fire_dir received here IS the effective (held) direction.
    apply_direction(self.fire_up, self.fire_down, self.fire_left, self.fire_right, fire_dir)

    if self.p1_start then
        self.p1_start:set_value(start_cmd)
    end
    if self.coin_1 then
        self.coin_1:set_value(coin_cmd)
    end
    return fire_dir   -- already the effective direction from Python
end

local function serialize_frame(player_alive, score, replay_level, num_lasers, wave_number,
                               player_x16, player_y16,
                               enemy_state, list_state_values,
                               done, subj_reward, obj_reward, save_signal)
    local score_u32 = math.max(0, math.min(4294967295, math.floor(score or 0)))
    local replay_u32 = math.max(0, math.min(4294967295, math.floor(replay_level or 0)))
    local lasers_u8 = math.max(0, math.min(255, math.floor(num_lasers or 0)))
    local wave_u8 = math.max(0, math.min(255, math.floor(wave_number or 0)))
    -- Keep scalar core features tightly bounded to avoid dominating slot signals.
    -- Replay is an 8-digit BCD field; map to ~[0,1] even if game-specific semantics vary.
    local replay_norm = math.min(1.0, replay_u32 / 99999999.0)
    -- Laser and wave counters can exceed expected gameplay ranges transiently.
    local lasers_norm = math.min(1.0, lasers_u8 / 9.0)
    local wave_norm = math.min(1.0, wave_u8 / 40.0)

    local state_values = {}
    -- 5 core stats
    state_values[#state_values + 1] = ((player_alive or 0) ~= 0) and 1.0 or 0.0
    state_values[#state_values + 1] = score_u32 / 1000000.0
    state_values[#state_values + 1] = replay_norm
    state_values[#state_values + 1] = lasers_norm
    state_values[#state_values + 1] = wave_norm
    -- 2 player position values (unsigned 8.8 fixed-point, screen-bound normalized 0..1)
    state_values[#state_values + 1] = norm_pos_x(player_x16 or 0)
    state_values[#state_values + 1] = norm_pos_y(player_y16 or 0)
    -- 2 player velocity values: frame-to-frame position delta, normalised over
    -- playfield range.  ~[-1,+1] with typical per-frame deltas near ±0.004 (X)
    -- and ±0.005 (Y).  Zero on first frame or after death.
    local vel_x = 0.0
    local vel_y = 0.0
    if prev_player_x16 and prev_player_y16 and player_alive ~= 0 then
        vel_x = clamp11(((player_x16 or 0) - prev_player_x16) / POS_X_RANGE)
        vel_y = clamp11(((player_y16 or 0) - prev_player_y16) / POS_Y_RANGE)
    end
    state_values[#state_values + 1] = vel_x
    state_values[#state_values + 1] = vel_y
    -- 50 ELIST enemy state bytes
    for i = 1, ZP1ENM_SIZE do
        state_values[#state_values + 1] = (enemy_state.raw[i] or 0) / 255.0
    end
    -- 4 active lists × 65 features each
    for i = 1, #list_state_values do
        state_values[#state_values + 1] = list_state_values[i]
    end
    local num_values = #state_values
    if num_values ~= EXPECTED_STATE_VALUES then
        error(string.format("state size mismatch: got=%d expected=%d", num_values, EXPECTED_STATE_VALUES))
    end

    local header = string.pack(
        ">HddBIBBIBB",
        num_values,
        subj_reward,
        obj_reward,
        done and 1 or 0,
        score_u32,
        player_alive,
        save_signal,
        replay_u32,
        lasers_u8,
        wave_u8
    )

    local state_payload_parts = {}
    for i = 1, num_values do
        state_payload_parts[#state_payload_parts + 1] = string.pack(">f", state_values[i])
    end
    local state_payload = table.concat(state_payload_parts)
    return header .. state_payload
end

local function process_frame_via_socket(frame_payload, frame_idx)
    if not current_socket then
        trace_log(frame_idx, "socket_open_needed", "no active socket")
        if not open_socket() then
            trace_log(frame_idx, "socket_open_failed", "open_socket() returned false")
            return -1, -1, false
        end
        trace_log(frame_idx, "socket_open_ok", "socket ready")
    end

    local write_ok, write_err = pcall(function()
        trace_log(frame_idx, "socket_write_begin", "payload_bytes=" .. tostring(#frame_payload))
        local length_header = string.pack(">H", #frame_payload)
        current_socket:write(length_header .. frame_payload)
    end)

    if not write_ok then
        trace_log(frame_idx, "socket_write_error", tostring(write_err))
        print("Socket write error: " .. tostring(write_err))
        close_socket()
        return -1, -1, false
    end
    trace_log(frame_idx, "socket_write_ok", "payload sent")

    local read_ok, read_result = pcall(function()
        trace_log(frame_idx, "socket_read_begin", "waiting for 3-byte action")
        local started = os.clock()
        while (os.clock() - started) < SOCKET_READ_TIMEOUT_S do
            local action_bytes = current_socket:read(3)
            if action_bytes and #action_bytes == 3 then
                local move_dir, fire_dir, source = string.unpack("bbb", action_bytes)
                trace_log(frame_idx, "socket_read_ok", string.format("move=%d fire=%d src=%d", move_dir, fire_dir, source))
                return {move_dir, fire_dir, source}
            end
        end
        trace_log(frame_idx, "socket_read_timeout", "using neutral action")
        return {-1, -1}
    end)

    if not read_ok then
        trace_log(frame_idx, "socket_read_error", tostring(read_result))
        print("Socket read error: " .. tostring(read_result))
        close_socket()
        return -1, -1, false
    end

    local move_dir, fire_dir, source = unpack(read_result)
    last_action_source = source or 0
    return move_dir or -1, fire_dir or -1, true
end

local function determine_meta_commands(dead_frames, player_alive)
    local start_cmd = 0
    local coin_cmd = 0

    -- Keep autoboot inputs out of active gameplay to avoid contaminating control/reward dynamics.
    if AUTOBOOT_ENABLED and (player_alive == 0) then
        local cycle_pos = dead_frames % AUTOBOOT_CYCLE_FRAMES
        if cycle_pos < AUTOBOOT_COIN_PULSE_FRAMES then
            coin_cmd = 1
        end
        if cycle_pos >= AUTOBOOT_START_DELAY_FRAMES and
           cycle_pos < (AUTOBOOT_START_DELAY_FRAMES + AUTOBOOT_START_PULSE_FRAMES) then
            start_cmd = 1
        end
    end

    return start_cmd, coin_cmd
end

local function frame_callback()
    if not mem or not controls then
        return true
    end

    trace_log(frame_counter, "frame_begin", "callback entered")

    local ok_alive, alive_or_err = pcall(read_player_alive, mem)
    if not ok_alive then
        trace_log(frame_counter, "read_player_alive_error", tostring(alive_or_err), true)
        return true
    end
    local player_alive = (alive_or_err ~= 0) and 1 or 0
    trace_log(frame_counter, "read_player_alive", "alive=" .. tostring(player_alive))

    local ok_core, score, replay_level, num_lasers, wave_number = pcall(function()
        local raw_score = read_player_score(mem)
        if raw_score == nil then
            -- Keep previous score on transient invalid BCD reads to avoid reward spikes.
            raw_score = previous_score
        end
        local score = math.max(0, math.floor(raw_score or 0))
        local replay_level = math.max(0, math.floor(read_next_replay_level(mem) or 0))
        local num_lasers = math.max(0, math.floor(read_num_lasers(mem) or 0))
        local wave_number = math.max(0, math.floor(read_wave_number(mem) or 0))
        return score, replay_level, num_lasers, wave_number
    end)
    if not ok_core then
        trace_log(frame_counter, "read_core_stats_error", tostring(score), true)
        return true
    end
    trace_log(
        frame_counter,
        "read_core_stats",
        string.format("score=%d replay=%d lasers=%d wave=%d", score, replay_level, num_lasers, wave_number)
    )

    local ok_enemy, enemy_or_err = pcall(read_enemy_state, mem)
    if not ok_enemy then
        trace_log(frame_counter, "read_enemy_state_error", tostring(enemy_or_err), true)
        return true
    end
    local enemy_state = enemy_or_err
    trace_log(frame_counter, "read_enemy_state", "bytes=" .. tostring(#enemy_state.raw))

    local ok_player_pos, player_x16, player_y16 = pcall(read_player_position, mem)
    if not ok_player_pos then
        trace_log(frame_counter, "read_player_position_error", tostring(player_x16), true)
        return true
    end
    trace_log(frame_counter, "read_player_position", string.format("x16=%d y16=%d", player_x16 or 0, player_y16 or 0))

    local ok_entities, entity_features_or_err, nearest_enemy_dist, nearest_human_dist, nearest_enemy_x16, nearest_enemy_y16, num_humans = pcall(
        extract_typed_entities, mem, player_x16, player_y16, enemy_state
    )
    if not ok_entities then
        trace_log(frame_counter, "extract_typed_entities_error", tostring(entity_features_or_err), true)
        return true
    end
    local list_state_values = entity_features_or_err
    num_humans = num_humans or 0
    trace_log(frame_counter, "extract_typed_entities", "features=" .. tostring(#list_state_values))

    local done = (previous_player_alive == 1 and player_alive == 0)
    local score_delta = score - previous_score
    if score_delta < 0 then
        score_delta = 0
    end
    local obj_reward = score_delta
    if done then
        obj_reward = obj_reward - DEATH_PENALTY_POINTS
    end
    if player_alive == 0 then
        dead_frame_counter = dead_frame_counter + 1
    else
        dead_frame_counter = 0
    end

    local spacing_score = enemy_spacing_score(nearest_enemy_dist)
    local rescue_score = human_proximity_score(nearest_human_dist)
    local aim_score = compute_aim_reward(prev_fire_cmd, prev_aim_px16, prev_aim_py16, prev_aim_objects)
    local evade_score = compute_evasion_reward(prev_move_cmd, prev_aim_px16, prev_aim_py16,
        prev_nearest_enemy_x16, prev_nearest_enemy_y16, prev_nearest_enemy_dist)
    local survival_bonus = (player_alive == 1) and SUBJ_SURVIVAL_BONUS or 0.0

    -- Wall penalty: penalise each axis independently; corners stack.
    local wall_penalty = 0.0
    if player_alive == 1 and player_x16 then
        local px = norm_pos_x(player_x16)
        local py = norm_pos_y(player_y16)
        if px < WALL_MARGIN_NORM_X or px > (1.0 - WALL_MARGIN_NORM_X) then
            wall_penalty = wall_penalty + SUBJ_WALL_PENALTY
        end
        if py < WALL_MARGIN_NORM_Y or py > (1.0 - WALL_MARGIN_NORM_Y) then
            wall_penalty = wall_penalty + SUBJ_WALL_PENALTY
        end
    end

    -- Abandoned-human penalty: on successful wave clear, penalise each
    -- human that was still alive on the previous frame (the last frame of
    -- the old wave).  Do NOT apply on death (already penalised separately).
    local abandoned_penalty = 0.0
    if player_alive == 1 and wave_number > previous_wave_number
       and previous_wave_number > 0 and prev_num_humans > 0 then
        abandoned_penalty = prev_num_humans * SUBJ_ABANDONED_HUMAN
        trace_log(frame_counter, "abandoned_humans",
            string.format("wave %d->%d humans_left=%d penalty=%.1f",
                previous_wave_number, wave_number, prev_num_humans, abandoned_penalty))
    end

    local subj_reward = survival_bonus
        + (spacing_score * SUBJ_ENEMY_WEIGHT)
        + (rescue_score * SUBJ_HUMAN_WEIGHT)
        + (aim_score * SUBJ_AIM_WEIGHT)
        + (evade_score * SUBJ_EVADE_WEIGHT)
        - wall_penalty
        - abandoned_penalty
    if done then
        subj_reward = subj_reward - SUBJ_DEATH_PENALTY
    end

    trace_log(
        frame_counter,
        "reward_calc",
        string.format(
            "score_delta=%d done=%s obj_reward=%.1f subj_reward=%.2f enemy_dist=%s human_dist=%s",
            score_delta,
            tostring(done),
            obj_reward,
            subj_reward,
            nearest_enemy_dist and string.format("%.4f", nearest_enemy_dist) or "nil",
            nearest_human_dist and string.format("%.4f", nearest_human_dist) or "nil"
        )
    )

    local now = os.time()
    local save_signal = 0
    if shutdown_requested or (now - last_save_time) >= SAVE_INTERVAL_S then
        save_signal = 1
        last_save_time = now
    end

    local ok_payload, payload_or_err = pcall(
        serialize_frame,
        player_alive, score, replay_level, num_lasers, wave_number,
        player_x16, player_y16,
        enemy_state, list_state_values,
        done, subj_reward, obj_reward, save_signal
    )
    if not ok_payload then
        trace_log(frame_counter, "serialize_frame_error", tostring(payload_or_err), true)
        return true
    end
    local payload = payload_or_err
    local payload_count = 9 + ZP1ENM_SIZE + #list_state_values
    trace_log(frame_counter, "serialize_frame", "num_values=" .. tostring(payload_count) .. " bytes=" .. tostring(#payload))

    local move_cmd, fire_cmd = -1, -1
    local socket_ok = false
    if DEBUG_BYPASS_SOCKET_FOR_FRAMES > 0 and frame_counter < DEBUG_BYPASS_SOCKET_FOR_FRAMES then
        trace_log(frame_counter, "socket_bypass", "bypassing exchange; neutral action")
        move_cmd, fire_cmd, socket_ok = -1, -1, true
    elseif current_socket then
        move_cmd, fire_cmd, socket_ok = process_frame_via_socket(payload, frame_counter)
    else
        if (now - last_connection_attempt_time) >= CONNECTION_RETRY_INTERVAL_S then
            trace_log(frame_counter, "socket_retry", "attempting reconnect")
            last_connection_attempt_time = now
            open_socket()
        end
        move_cmd, fire_cmd = -1, -1
    end

    if not socket_ok then
        move_cmd, fire_cmd = -1, -1
    end

    if DEBUG_FORCE_ACTION_FRAMES > 0 and frame_counter < DEBUG_FORCE_ACTION_FRAMES then
        move_cmd = DEBUG_FORCE_MOVE_DIR
        fire_cmd = DEBUG_FORCE_FIRE_DIR
        trace_log(
            frame_counter,
            "force_action",
            string.format("move=%d fire=%d", move_cmd, fire_cmd)
        )
    end

    local start_cmd, coin_cmd = determine_meta_commands(dead_frame_counter, player_alive)
    local effective_fire = fire_cmd  -- fallback if pcall fails
    local ok_apply, apply_result = pcall(controls.apply_action, controls, move_cmd, fire_cmd, start_cmd, coin_cmd)
    if not ok_apply then
        trace_log(frame_counter, "apply_action_error", tostring(apply_result), true)
        return true
    end
    effective_fire = apply_result or fire_cmd  -- apply_action returns the held fire direction
    trace_log(
        frame_counter,
        "apply_action",
        string.format("move=%d fire=%d eff_fire=%d start=%d coin=%d socket_ok=%s",
            move_cmd, fire_cmd, effective_fire, start_cmd, coin_cmd, tostring(socket_ok))
    )

    previous_player_alive = player_alive
    previous_score = score
    previous_wave_number = wave_number
    prev_num_humans = num_humans or 0

    -- Stash position for next-frame velocity computation
    if player_alive == 1 then
        prev_player_x16 = player_x16
        prev_player_y16 = player_y16
    else
        prev_player_x16 = nil
        prev_player_y16 = nil
    end

    -- Stash aim-reward data for use on the NEXT frame (reward for this frame's action).
    -- Use effective_fire (the direction actually sent to the game after hold) for correct
    -- aim-reward attribution.
    prev_fire_cmd = effective_fire
    prev_move_cmd = move_cmd
    prev_aim_objects = hud_objects   -- reuse the same reference (set in extract_typed_entities)
    prev_aim_px16 = player_x16
    prev_aim_py16 = player_y16
    prev_nearest_enemy_x16 = nearest_enemy_x16
    prev_nearest_enemy_y16 = nearest_enemy_y16
    prev_nearest_enemy_dist = nearest_enemy_dist

    -- Draw debug HUD overlay (entity letters on screen)
    draw_debug_hud()

    trace_log(frame_counter, "frame_end", "done=" .. tostring(done))
    frame_counter = frame_counter + 1

    return true
end

local function on_mame_exit()
    shutdown_requested = true
    trace_log(frame_counter, "session_stop", "machine stop notifier", true)
    close_socket()
    print("Robotron AI Lua script shutting down.")
end

math.randomseed(os.time())

if not initialize_mame_interface() then
    return
end

print("Robotron socket target: " .. SOCKET_ADDRESS)
controls = Controls:new(manager)
if not controls then
    print("Robotron AI Lua script aborted due to missing control mappings.")
    return
end
open_socket()

trace_log(nil, "session_start", "robotron lua init", true)
last_save_time = os.time()
previous_player_alive = (read_player_alive(mem) ~= 0) and 1 or 0
previous_score = math.max(0, math.floor(read_player_score(mem) or 0))
previous_wave_number = math.max(0, math.floor(read_wave_number(mem) or 0))

global_callback_ref = emu.add_machine_frame_notifier(frame_callback)

-- Register HUD drawing AFTER video rendering so our overlay is not overwritten.
emu.register_frame_done(draw_debug_hud)
print("[HUD] Registered frame_done callback for debug overlay")

emu.add_machine_stop_notifier(on_mame_exit)

print("Robotron AI Lua script initialized.")
