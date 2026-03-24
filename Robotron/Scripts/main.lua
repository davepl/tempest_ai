--[[
    Robotron AI Lua script for MAME.

    Current scope:
      - Sends a compact RL state vector:
        + 18 core/player values
        + 22 ELIST bytes (first 22 of 50; rest are reserved padding)
        + 1 unified entity pool: 1 occupancy + 64 slots × 11 features
          storing present, dx, dy, dist, vx, vy, threat, approach,
          hit_w, hit_h, type_id
      - Receives joystick commands: movement_dir (-1 neutral or 0..7) and firing_dir (0..7)
--]]

RAW_SOCKET_ADDRESS = os.getenv("ROBOTRON_SOCKET_ADDRESS") or "ubvmdell:9998"
PREVIEW_CLIENT_FLAG = (os.getenv("ROBOTRON_PREVIEW_CLIENT") == "1") and 1 or 0
CLIENT_SLOT = math.max(0, math.floor(tonumber(os.getenv("ROBOTRON_CLIENT_SLOT") or "0") or 0))
SOCKET_ADDRESS = RAW_SOCKET_ADDRESS
if string.sub(SOCKET_ADDRESS, 1, 7) ~= "socket." then
    SOCKET_ADDRESS = "socket." .. SOCKET_ADDRESS
end
SOCKET_READ_TIMEOUT_S = 3.5
CONNECTION_RETRY_INTERVAL_S = 1.0
SAVE_INTERVAL_S = 300
unpack = table.unpack or unpack

function env_number(name, default)
    local raw = os.getenv(name)
    if raw == nil or raw == "" then
        return default
    end
    local val = tonumber(raw)
    if val == nil then
        return default
    end
    return val
end

function env_flag(name, default)
    local raw = os.getenv(name)
    if raw == nil or raw == "" then
        return default
    end
    raw = string.lower(tostring(raw))
    return not (raw == "0" or raw == "false" or raw == "off" or raw == "no")
end

RRCHRIS_PATCH_ENABLED = env_flag("ROBOTRON_ENABLE_RRCHRIS_PATCH", false)
RRCHRIS_PATCH_REGION = ":maincpu"
ROMTAB_BASE_ADDR = 0xFFB5
RRCHRIS_PATCH_CHUNKS = {
    { addr = 0x4990, bytes = {0x7E, 0x4A, 0x97} },
    { addr = 0x49B7, bytes = {0x7E, 0x4A, 0xA0} },
    { addr = 0x49E2, bytes = {0x7E, 0x4A, 0xA9} },
    {
        addr = 0x4A97,
        bytes = {
            0x0A, 0xC2, 0x27, 0xBE, 0x0A, 0xB9, 0x7E, 0x49, 0x94,
            0x0A, 0xC2, 0x27, 0xB5, 0x0C, 0xB8, 0x7E, 0x49, 0xBB,
            0x0A, 0xC2, 0x27, 0xAC, 0x0C, 0xB8, 0x7E, 0x49, 0xE6,
            0x30, 0x35, 0x2F, 0x31, 0x39, 0x38, 0x37, 0x20, 0x43,
            0x48, 0x52, 0x2E, 0x47, 0x2E,
        },
    },
    { addr = 0x4AD7, bytes = {0x4C} },
}
rrchris_patch_applied = false

-- Startup diagnostics (bounded, opt-in style flags kept local to script).
DEBUG_STARTUP_TRACE = false
DEBUG_TRACE_FRAMES = 10
DEBUG_BYPASS_SOCKET_FOR_FRAMES = 0
DEBUG_TRACE_FILE = "logs/startup_trace.log"
DEBUG_FORCE_ACTION_FRAMES = 0
DEBUG_FORCE_MOVE_DIR = 2  -- right
DEBUG_FORCE_FIRE_DIR = 2  -- right
DEATH_PENALTY_POINTS = 25000
-- Subjective shaping rewards (raw points; scaled in Python by subj_reward_scale).
-- Goal: densify survival signal without dominating objective score rewards.
SUBJ_ENEMY_WEIGHT = 8.0
SUBJ_HUMAN_WEIGHT = 12.0
SUBJ_SURVIVAL_BONUS = 2.0
-- Survival shaping is only awarded while there are humans left to rescue.
SUBJ_SURVIVAL_REQUIRE_HUMANS = true
-- Per-frame penalty when alive but no humans remain; helps avoid end-of-wave stalling.
SUBJ_NO_HUMANS_EXISTENCE_PENALTY = 2.0
SUBJ_DEATH_PENALTY = 25.0
SUBJ_ENEMY_NEAR_NORM = 0.035
SUBJ_ENEMY_FAR_NORM = 0.200
SUBJ_HUMAN_NEAR_NORM = 0.120
ADVANCED_SHAPING = {
    priority_aim_weight = 10.0,
    brain_guard_weight = 8.0,
    high_wave_threshold = 6,
    brain_guard_wave = 4,
    center_pressure_norm = 0.13,
    center_pull_margin_x = 0.16,
    center_pull_margin_y = 0.12,
}

-- Aiming reward: bonus for firing toward aligned enemies/obstacles.
SUBJ_AIM_WEIGHT = 15.0        -- reward per frame when correctly aimed
AIM_CROSS_THRESHOLD = 2048     -- 8 screen-pixels in x16 units (8 * 256)
AIM_MIN_FORWARD = 1024         -- ~4 screen-pixels minimum forward distance
-- Categories that count as "targets" for aim reward (everything but humans).
AIM_TARGET_CATS = {
    grunt = true, hulk = true, brain = true, tank = true,
    spawner = true, enforcer = true, projectile = true, electrode = true,
}
ADVANCED_SHAPING.priority_aim_bonus = {
    grunt = 1.00,
    hulk = 0.85,
    brain = 1.45,
    tank = 1.15,
    spawner = 1.20,
    enforcer = 1.25,
    projectile = 1.50,
    electrode = 0.75,
}
-- Direction unit vectors for 8-way fire (dx, dy in x16 coordinates).
-- Screen/world coordinates are y-down (larger y = lower on screen):
-- 0=up (-y), 1=up-right, 2=right, 3=down-right, 4=down (+y), 5=down-left, 6=left, 7=up-left
FIRE_DIR_VEC = {
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
SUBJ_EVADE_WEIGHT = 10.0       -- reward when moving away from nearest threat
EVADE_DANGER_NORM  = 0.08      -- only reward evasion when enemy within this normalised dist
MOVE_DIR_VEC = FIRE_DIR_VEC    -- same 8-way mapping for move directions

-- Wall-hugging penalty: per-axis penalty when within 16 px of a wall.
-- Stacks additively so a corner costs double.
SUBJ_WALL_PENALTY  = 15.0      -- penalty per wall axis per frame
-- WALL_MARGIN_NORM_X/Y defined after POS_X/Y_RANGE (see below).

-- Abandoned-human penalty: one-shot penalty per surviving human when a wave
-- is cleared.  Encourages the AI to rescue first, kill last.
SUBJ_ABANDONED_HUMAN = 15.0     -- penalty per unrescued human on wave end

mainCpu = nil
mem = nil
controls = nil

current_socket = nil
last_connection_attempt_time = 0
shutdown_requested = false

frame_counter = 0
dead_frame_counter = 0
last_save_time = 0
previous_player_alive = 1
previous_score = 0
previous_wave_number = 0
prev_num_humans = 0
prev_fire_cmd = -1          -- fire direction from previous frame
prev_move_cmd = -1          -- move direction from previous frame
prev_aim_objects = nil      -- classified objects from previous frame
last_action_source = 0      -- 0=none, 1=dqn, 2=epsilon, 3=expert, 4=forced_random

-- Fire-hold state:  The game's LSPROC laser routine (RRG23.ASM) requires
-- the fire joystick to stay in the SAME direction for 3 consecutive frames
-- before it creates a laser.  At high epsilon, random fire directions change
-- every frame, so shots almost never fire.  We hold each fire direction for
-- a minimum of FIRE_HOLD_FRAMES before accepting a new one.
FIRE_HOLD_FRAMES = 4       -- frames to lock each fire direction (3 = minimum for 1 shot)
fire_hold_dir   = -1       -- direction currently being held
fire_hold_count = 0        -- frames remaining in current hold
prev_aim_px16 = nil         -- player x16 from previous frame
prev_aim_py16 = nil         -- player y16 from previous frame
prev_nearest_enemy_x16 = nil
prev_nearest_enemy_y16 = nil
prev_nearest_enemy_dist = nil

-- Autoboot input sequence (MAME input level, no game-specific memory logic required).
-- Every cycle: pulse Coin 1, then pulse 1P Start shortly after.
AUTOBOOT_ENABLED = true
AUTOBOOT_CYCLE_FRAMES = 300
AUTOBOOT_COIN_PULSE_FRAMES = 3
AUTOBOOT_START_DELAY_FRAMES = 18
AUTOBOOT_START_PULSE_FRAMES = 3

-- Robotron RAM symbols (from Williams map):
-- STATUS = 0x9859
-- In PLAYRV, bit0 gates player control update; on player death PLEND sets STATUS to 0x1B.
STATUS_ADDR = 0x9859
STATUS_PLAYER_INACTIVE_MASK = 0x01

-- Player PLDATA symbols (from Williams map generated by lwasm):
--   ZP1SCR = 0xBDE4 (4 bytes packed BCD, MSB first)
--   ZP1RP  = 0xBDE8 (4 bytes packed BCD, MSB first)
--   ZP1LAS = 0xBDEC (1 byte)
--   ZP1WAV = 0xBDED (1 byte)
--   ZP1ENM = 0xBDEE (50 bytes; first 22 mirror ELIST fields, rest reserved)
ZP1SCR_ADDR = 0xBDE4
ZP1RP_ADDR = 0xBDE8
ZP1LAS_ADDR = 0xBDEC
ZP1WAV_ADDR = 0xBDED
ZP1ENM_ADDR = 0xBDEE
ZP1ENM_SIZE = 50         -- total bytes to read from memory
ZP1ENM_EMIT_COUNT = 22   -- only emit the 22 meaningful ELIST fields to Python

-- Active object-list heads (from Williams base-page RAM map / RRF.ASM).
-- Address layout: OPTR($9817), OBPTR($9819), OFREE($981B), SPFREE($981D),
--                 HPTR($981F), RPTR($9821), PPTR($9823)
OPTR_ADDR = 0x9817  -- motion objects (enforcers, sparks, circles, squares, shells, player lasers)
HPTR_ADDR = 0x981F  -- humans to rescue (mom, dad, kid)
RPTR_ADDR = 0x9821  -- robots/enemies (grunts, brains, hulks, tanks, progs, cruise missiles)
PPTR_ADDR = 0x9823  -- fatal obstacles (electrodes)

-- Player object structure (PLOBJ at $985A from RRF.ASM).
PLOBJ_ADDR = 0x985A
PX16_ADDR = 0x9864   -- player 16-bit X world coordinate
PY16_ADDR = 0x9866   -- player 16-bit Y world coordinate
PXV_ADDR = 0x9868    -- player X velocity
PYV_ADDR = 0x986A    -- player Y velocity

-- Object pool geometry (master list @ OLIST).
OLIST_START = 0x9900
OLIST_ENTRY_SIZE = 0x18
OLIST_CAPACITY = 180
OLIST_END = OLIST_START + (OLIST_ENTRY_SIZE * OLIST_CAPACITY)

-- Object entry offsets.
OLINK_OFF = 0x00
OPICT_OFF = 0x02
OBJX_OFF = 0x04
OBJY_OFF = 0x05
OX16_OFF = 0x0A
OY16_OFF = 0x0C
FONIPC_OFF = 0x16

MAX_LIST_WALK = 256
OCVECT_OFF = 0x08  -- collision routine address (stable per entity type)

-- Object list head pointers (used to walk linked lists).
ACTIVE_LISTS = {
    {name = "optr", addr = OPTR_ADDR},   -- motion objects: enforcers, sparks, circles, squares, shells, lasers
    {name = "hptr", addr = HPTR_ADDR},   -- humans: mom, dad, kid
    {name = "rptr", addr = RPTR_ADDR},   -- robots: grunts, brains, hulks, tanks, progs, cruise missiles
    {name = "pptr", addr = PPTR_ADDR},   -- fatal: electrodes
}

-- Per-type entity categories. Order MUST match Python config/aimodel.py.
ENTITY_CATEGORIES = {
    {name = "grunt",      slots = 40, peak = 80},
    {name = "hulk",       slots = 16, peak = 25},
    {name = "brain",      slots = 16, peak = 25},
    {name = "tank",       slots =  8, peak = 14},
    {name = "spawner",    slots =  8, peak = 14},
    {name = "enforcer",   slots = 12, peak = 10},
    {name = "projectile", slots = 12, peak = 20},
    {name = "human",      slots = 16, peak = 30},
    {name = "electrode",  slots = 16, peak = 25},
}
ENTITY_CATEGORY_INDEX = {}
for idx, cat in ipairs(ENTITY_CATEGORIES) do
    ENTITY_CATEGORY_INDEX[cat.name] = idx - 1
end

COMPACT_GLOBAL_FEATURES = 28
COMPACT_SLOT_STATE_FEATURES = 8
COMPACT_NON_HULK_SLOTS = 16
COMPACT_HULK_SLOTS = 4
COMPACT_ELECTRODE_SLOTS = 4
COMPACT_TOTAL_SLOTS = COMPACT_NON_HULK_SLOTS + COMPACT_HULK_SLOTS + COMPACT_ELECTRODE_SLOTS
EXPECTED_STATE_VALUES = COMPACT_GLOBAL_FEATURES + (COMPACT_TOTAL_SLOTS * COMPACT_SLOT_STATE_FEATURES)

-- Type ID mapping for unified pool (0-8, normalized by /8.0 in emission)
UNIFIED_TYPE_ID = {
    grunt = 0,
    hulk = 1,
    brain = 2,
    tank = 3,
    spawner = 4,
    enforcer = 5,
    projectile = 6,
    human = 7,
    electrode = 8,
}
UNIFIED_NUM_TYPES = 9   -- for normalization: type_id / (UNIFIED_NUM_TYPES - 1)

local CATEGORY_THREAT_WEIGHT = {
    grunt = 0.55,
    hulk = 1.00,
    brain = 0.95,
    tank = 0.95,
    spawner = 0.85,
    enforcer = 0.85,
    projectile = 0.90,
    human = 0.35,
    electrode = 0.75,
}

local CATEGORY_IS_DANGEROUS = {
    grunt = true, hulk = true, brain = true, tank = true,
    spawner = true, enforcer = true, projectile = true, electrode = true,
}

local CATEGORY_IS_STATIC = {
    electrode = true,
}

local prev_object_sample_x = {}
local prev_object_sample_y = {}
local prev_compact_slot_ptrs = {
    non_hulk = {},
    hulk = {},
    electrode = {},
}

local function _reset_compact_slot_assignments()
    prev_compact_slot_ptrs = {
        non_hulk = {},
        hulk = {},
        electrode = {},
    }
    for i = 1, COMPACT_NON_HULK_SLOTS do
        prev_compact_slot_ptrs.non_hulk[i] = 0
    end
    for i = 1, COMPACT_HULK_SLOTS do
        prev_compact_slot_ptrs.hulk[i] = 0
    end
    for i = 1, COMPACT_ELECTRODE_SLOTS do
        prev_compact_slot_ptrs.electrode[i] = 0
    end
end

_reset_compact_slot_assignments()

local function _stable_assign_bucket_slots(cat_name, bucket, slot_count)
    local assigned = {}
    local selected = {}
    local selected_by_ptr = {}
    local prev_slots = prev_category_slot_ptrs[cat_name] or {}
    local next_slots = {}
    local selected_n = 0

    -- Keep only the nearest-K candidates for slot assignment without sorting the
    -- full bucket. This avoids an O(n log n) table.sort per category every frame.
    for i = 1, #bucket do
        local obj = bucket[i]
        local dist = obj.dist_norm or 1.0
        local keep_candidate = true
        if selected_n < slot_count then
            selected_n = selected_n + 1
        else
            local worst = selected[selected_n]
            if worst ~= nil and dist >= (worst.dist_norm or 1.0) then
                keep_candidate = false
            end
        end

        if keep_candidate then
            local insert_pos = math.min(selected_n, slot_count)
            while insert_pos > 1 do
                local prev_obj = selected[insert_pos - 1]
                local prev_dist = prev_obj and (prev_obj.dist_norm or 1.0) or 1.0
                if dist >= prev_dist then
                    break
                end
                selected[insert_pos] = prev_obj
                insert_pos = insert_pos - 1
            end
            selected[insert_pos] = obj
        end
    end

    for i = 1, selected_n do
        local obj = selected[i]
        selected_by_ptr[obj.ptr] = obj
    end

    -- Preserve any previously assigned slot whose object is still in the
    -- nearest-K candidate set, then fill remaining holes by proximity.
    for slot_idx = 1, slot_count do
        local prev_ptr = prev_slots[slot_idx]
        local obj = prev_ptr and selected_by_ptr[prev_ptr] or nil
        if obj ~= nil then
            assigned[slot_idx] = obj
            selected_by_ptr[prev_ptr] = nil
        end
    end

    local fill_idx = 1
    for slot_idx = 1, slot_count do
        if assigned[slot_idx] == nil then
            while fill_idx <= selected_n do
                local obj = selected[fill_idx]
                fill_idx = fill_idx + 1
                if obj ~= nil and selected_by_ptr[obj.ptr] ~= nil then
                    assigned[slot_idx] = obj
                    selected_by_ptr[obj.ptr] = nil
                    break
                end
            end
        end
        next_slots[slot_idx] = assigned[slot_idx] and assigned[slot_idx].ptr or 0
    end

    return assigned, next_slots, selected_n
end

-- Stable-assign a flat list of entities into the unified pool.
-- Partition into dangerous / human / electrode tiers, keep only the top-K
-- entities needed from each tier, then merge with guaranteed minimum quotas so
-- humans and electrodes are never starved on crowded waves.
local MIN_HUMAN_RESERVE    = 10    -- guaranteed human slots (if that many exist)
local MIN_ELECTRODE_RESERVE = 6    -- guaranteed electrode slots (if that many exist)

local function _select_top_k_sorted(bucket, limit, better_fn)
    local selected = {}
    local selected_n = 0
    limit = math.max(0, math.floor(tonumber(limit) or 0))
    if limit <= 0 then
        return selected, 0
    end

    for i = 1, #bucket do
        local obj = bucket[i]
        if selected_n < limit then
            selected_n = selected_n + 1
            local insert_pos = selected_n
            while insert_pos > 1 and better_fn(obj, selected[insert_pos - 1]) do
                selected[insert_pos] = selected[insert_pos - 1]
                insert_pos = insert_pos - 1
            end
            selected[insert_pos] = obj
        else
            local worst = selected[selected_n]
            if better_fn(obj, worst) then
                local insert_pos = selected_n
                while insert_pos > 1 and better_fn(obj, selected[insert_pos - 1]) do
                    selected[insert_pos] = selected[insert_pos - 1]
                    insert_pos = insert_pos - 1
                end
                selected[insert_pos] = obj
            end
        end
    end

    return selected, selected_n
end

local function _dangerous_priority_better(a, b)
    if b == nil then return true end
    local ta, tb = tonumber(a.threat) or 0.0, tonumber(b.threat) or 0.0
    if ta ~= tb then
        return ta > tb
    end
    return (tonumber(a.dist_norm) or 1.0) < (tonumber(b.dist_norm) or 1.0)
end

local function _nearest_distance_better(a, b)
    if b == nil then return true end
    return (tonumber(a.dist_norm) or 1.0) < (tonumber(b.dist_norm) or 1.0)
end

local function _stable_assign_unified_pool(dangerous, humans, electrodes)
    -- Allocate slots with guaranteed minimums for humans/electrodes
    local slot_count      = UNIFIED_POOL_SLOTS
    local human_quota     = math.min(#humans, MIN_HUMAN_RESERVE)
    local electrode_quota = math.min(#electrodes, MIN_ELECTRODE_RESERVE)
    local dangerous_quota = 0
    local remaining = slot_count - human_quota - electrode_quota

    local extra_d = math.min(math.max(#dangerous - dangerous_quota, 0), remaining)
    dangerous_quota = dangerous_quota + extra_d
    remaining = remaining - extra_d

    local extra_h = math.min(math.max(#humans - human_quota, 0), remaining)
    human_quota = human_quota + extra_h
    remaining = remaining - extra_h

    local extra_e = math.min(math.max(#electrodes - electrode_quota, 0), remaining)
    electrode_quota = electrode_quota + extra_e

    local selected_dangerous, selected_dangerous_n = _select_top_k_sorted(dangerous, dangerous_quota, _dangerous_priority_better)
    local selected_humans, selected_humans_n = _select_top_k_sorted(humans, human_quota, _nearest_distance_better)
    local selected_electrodes, selected_electrodes_n = _select_top_k_sorted(electrodes, electrode_quota, _nearest_distance_better)

    -- Build selected list: dangerous first, then humans, then electrodes
    local selected = {}
    local selected_n = 0
    for i = 1, selected_dangerous_n do
        selected_n = selected_n + 1
        selected[selected_n] = selected_dangerous[i]
    end
    for i = 1, selected_humans_n do
        selected_n = selected_n + 1
        selected[selected_n] = selected_humans[i]
    end
    for i = 1, selected_electrodes_n do
        selected_n = selected_n + 1
        selected[selected_n] = selected_electrodes[i]
    end

    -- Stable pointer matching — preserve slot assignments across frames
    local assigned = {}
    local selected_by_ptr = {}
    for i = 1, selected_n do
        selected_by_ptr[selected[i].ptr] = selected[i]
    end

    local prev_slots = prev_unified_slot_ptrs
    for slot_idx = 1, slot_count do
        local prev_ptr = prev_slots[slot_idx]
        local obj = prev_ptr and prev_ptr ~= 0 and selected_by_ptr[prev_ptr] or nil
        if obj ~= nil then
            assigned[slot_idx] = obj
            selected_by_ptr[prev_ptr] = nil
        end
    end

    -- Fill remaining holes in priority order
    local fill_idx = 1
    for slot_idx = 1, slot_count do
        if assigned[slot_idx] == nil then
            while fill_idx <= selected_n do
                local obj = selected[fill_idx]
                fill_idx = fill_idx + 1
                if obj ~= nil and selected_by_ptr[obj.ptr] ~= nil then
                    assigned[slot_idx] = obj
                    selected_by_ptr[obj.ptr] = nil
                    break
                end
            end
        end
    end

    -- Update slot pointers for next frame
    local next_slots = {}
    for i = 1, slot_count do
        next_slots[i] = assigned[i] and assigned[i].ptr or 0
    end
    prev_unified_slot_ptrs = next_slots

    return assigned, selected_n
end

-- OCVECT-based entity classification (auto-discovered at runtime).
local ocvect_category_cache = {}       -- OCVECT address → category name | "skip"
local discovered_tank_ocvect = nil     -- TNKIL address once discovered via growing phase
local unresolved_7x16 = {}             -- {[ocvect] = true} for ambiguous 7×16 on RPTR
local DEBUG_LOG_DISCOVERY = false      -- set true to log OCVECT classification discoveries

-- Debug HUD overlay state (draws entity letters on screen each frame).
mame_screen = nil                -- MAME screen device for draw_text
hud_objects = nil                -- last frame's classified object list (reference)
hud_player_x16 = nil
hud_player_y16 = nil
hud_player_box = nil
DEBUG_HUD_ENABLED = true         -- default ON; toggle with H hotkey
hud_key_code = nil               -- MAME input code for 'H' key (lazy-init)
hud_key_was_down = false         -- edge-detect so hold doesn't strobe
PREVIEW_FORMAT_RGB565 = 1
PREVIEW_FORMAT_RGB565_LZSS = 2
PREVIEW_FORMAT_RGB565_RLE = 3
-- Enable preview support; server controls streaming per-client via action source flags.
PREVIEW_CAPTURE_ENABLED = true
PREVIEW_FPS = math.max(1, math.floor(env_number("ROBOTRON_PREVIEW_FPS", 30) or 30))
PREVIEW_MIN_INTERVAL_S = (1.0 / PREVIEW_FPS)
-- Capture near dashboard size at the source; sending full-resolution snapshots
-- through Lua was the dominant cost for the preview client.
PREVIEW_MAX_WIDTH = math.max(64, math.floor(env_number("ROBOTRON_PREVIEW_MAX_WIDTH", 320) or 320))
PREVIEW_MAX_HEIGHT = math.max(64, math.floor(env_number("ROBOTRON_PREVIEW_MAX_HEIGHT", 240) or 240))
PREVIEW_TRY_RLE = env_flag("ROBOTRON_PREVIEW_RLE", false)
PREVIEW_RLE_MIN_SAVINGS_BYTES = math.max(64, math.floor(env_number("ROBOTRON_PREVIEW_RLE_MIN_SAVINGS", 256) or 256))
PREVIEW_MAX_BYTES = 2000000
SOCKET_MAX_PAYLOAD_BYTES = 4194304
preview_stream_enabled = false
pending_preview_blob = nil
pending_preview_w = 0
pending_preview_h = 0
pending_preview_fmt = PREVIEW_FORMAT_RGB565
last_preview_capture_time_s = -1000000.0
preview_source_w = 0
preview_source_h = 0
preview_target_w = 0
preview_target_h = 0
picture_bounds_cache = {}
START_ADVANCED = false
START_LEVEL_MIN = 1
ACTION_RX_BUFFER = ""
ROBOTRON_WAVE_FIELD_CSV = {
    "20,15,15,15,15,15,15,15,15,15,14,14,14,14,14,13,13,13,13,13,14,14,14,14,14,14,13,13,13,13,13,13,12,12,12,12,12,12,15,12",
    "9,7,6,5,5,5,5,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,3,4,3,3,3,3,3,3,3,3,3,4,3",
    "10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11",
    "30,28,26,24,22,20,18,18,16,14,14,14,14,14,14,14,14,14,14,14,15,15,15,15,15,15,15,15,15,14,14,14,14,14,14,14,14,14,14,14",
    "30,28,26,24,30,20,18,16,18,25,12,12,12,25,25,12,12,12,18,20,14,14,14,14,14,25,14,14,18,25,12,12,12,12,25,12,12,12,18,20",
    "8,8,7,7,7,7,7,6,6,6,6,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5",
    "64,64,64,64,64,40,40,38,38,38,38,38,38,38,38,38,36,36,36,36,32,32,32,32,32,32,32,30,30,30,30,30,25,25,25,25,25,25,25,25",
    "8,8,8,8,8,7,7,7,7,7,7,7,7,7,7,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6",
    "32,32,32,32,32,32,32,30,30,30,30,30,30,28,28,28,28,28,28,28,30,30,30,30,30,30,28,28,28,28,28,26,26,26,26,26,24,24,24,24",
    "176,176,176,176,176,176,176,176,176,176,176,176,176,176,176,176,176,176,176,176,184,184,184,184,184,184,184,184,184,184,192,192,192,192,192,192,192,192,192,192",
    "16,16,16,16,16,16,16,16,16,16,16,16,16,16,15,15,15,15,15,15,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14",
    "50,50,50,50,50,50,50,50,50,50,50,50,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,56,60,60,60,60,60,60,60,60,60,60,60,60",
}
ROBOTRON_WAVE_COUNT_CSV = {
    "15,17,22,34,20,32,0,35,60,25,35,0,35,27,25,35,0,35,70,25,35,0,35,0,25,35,0,35,75,25,35,0,35,30,27,35,0,35,80,30",
    "5,15,25,25,20,25,0,25,0,20,25,0,25,5,20,25,0,25,0,20,25,0,25,0,20,25,0,25,0,20,25,0,25,0,15,25,0,25,0,15",
    "1,1,2,2,15,3,4,3,3,0,3,3,3,5,0,3,3,3,3,8,3,3,3,3,25,3,3,3,3,0,3,3,3,3,0,3,3,3,3,10",
    "1,1,2,2,0,3,4,3,3,22,3,3,3,5,0,3,3,3,3,8,3,3,3,3,0,3,3,3,3,25,3,3,3,3,0,3,3,3,3,10",
    "0,1,2,2,1,3,4,3,3,0,3,3,3,5,22,3,3,3,3,8,3,3,3,3,1,3,3,3,3,0,3,3,3,3,25,3,3,3,3,10",
    "0,5,6,7,0,7,12,8,4,0,8,13,8,20,2,3,14,8,3,2,8,15,8,13,1,8,16,8,4,1,8,16,8,25,2,8,16,8,6,2",
    "0,0,0,0,15,0,0,0,0,20,0,0,0,0,20,0,0,0,0,20,0,0,0,0,21,0,0,0,0,22,0,0,0,0,23,0,0,0,0,25",
    "0,1,3,4,1,4,0,5,5,1,5,0,5,2,1,5,0,5,5,2,5,0,5,6,1,5,0,5,5,1,5,0,5,2,1,5,0,5,5,1",
    "0,0,0,0,0,0,10,0,0,0,0,12,0,0,0,0,12,0,0,0,0,12,0,7,0,0,12,1,1,1,1,13,1,2,2,2,14,2,1,1",
}
ROBOTRON_WAVE_FIELD_TABLES = nil
ROBOTRON_WAVE_COUNT_TABLES = nil

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

function parse_csv_u8_list(src)
    local out = {}
    if not src or src == "" then
        return out
    end
    for token in string.gmatch(src, "([^,]+)") do
        local value = math.floor(tonumber(token) or 0)
        if value < 0 then
            value = 0
        elseif value > 255 then
            value = 255
        end
        out[#out + 1] = value
    end
    return out
end

function ensure_robotron_wave_tables()
    if ROBOTRON_WAVE_FIELD_TABLES and ROBOTRON_WAVE_COUNT_TABLES then
        return
    end

    ROBOTRON_WAVE_FIELD_TABLES = {}
    for i = 1, #ROBOTRON_WAVE_FIELD_CSV do
        ROBOTRON_WAVE_FIELD_TABLES[i] = parse_csv_u8_list(ROBOTRON_WAVE_FIELD_CSV[i])
    end

    ROBOTRON_WAVE_COUNT_TABLES = {}
    for i = 1, #ROBOTRON_WAVE_COUNT_CSV do
        ROBOTRON_WAVE_COUNT_TABLES[i] = parse_csv_u8_list(ROBOTRON_WAVE_COUNT_CSV[i])
    end
end

function robotron_wave_table_index(wave_number)
    local wave = math.max(1, math.floor(tonumber(wave_number) or 1))
    while wave > 40 do
        wave = wave - 20
    end
    return wave
end

function robotron_build_enemy_bytes_for_wave(wave_number)
    ensure_robotron_wave_tables()

    local idx = robotron_wave_table_index(wave_number)
    local bytes = {}

    for table_idx = 1, #ROBOTRON_WAVE_FIELD_TABLES do
        local values = ROBOTRON_WAVE_FIELD_TABLES[table_idx]
        bytes[#bytes + 1] = values[idx] or values[#values] or 0
    end

    for table_idx = 1, #ROBOTRON_WAVE_COUNT_TABLES do
        local values = ROBOTRON_WAVE_COUNT_TABLES[table_idx]
        bytes[#bytes + 1] = values[idx] or values[#values] or 0
    end

    bytes[#bytes + 1] = 0
    return bytes
end

function robotron_apply_start_wave_patch(memory, player_alive, score, wave_number)
    if not START_ADVANCED then
        return nil
    end

    local desired_wave = math.max(1, math.min(81, math.floor(tonumber(START_LEVEL_MIN) or 1)))
    if desired_wave <= 1 then
        return nil
    end
    if (player_alive or 0) ~= 0 then
        return nil
    end
    if (score or 0) ~= 0 then
        return nil
    end
    if math.floor(tonumber(wave_number) or 0) ~= 1 then
        return nil
    end

    local bytes = robotron_build_enemy_bytes_for_wave(desired_wave)
    if #bytes < 22 then
        return nil
    end

    memory:write_u8(ZP1WAV_ADDR, desired_wave)
    for i = 1, 22 do
        memory:write_u8(ZP1ENM_ADDR + (i - 1), bytes[i])
    end

    return desired_wave
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

function movement_alignment_score(move_cmd, target_x, target_y)
    if move_cmd == nil or move_cmd < 0 or move_cmd > 7 then
        return 0.0
    end
    local vec = MOVE_DIR_VEC[move_cmd]
    if vec == nil then return 0.0 end
    local len = math.sqrt((target_x * target_x) + (target_y * target_y))
    if len < 1.0 then
        return 0.0
    end
    local tx = target_x / len
    local ty = target_y / len
    return clamp01((vec[1] * tx + vec[2] * ty) / 1.414)
end

function priority_target_bonus(category, wave_number, num_humans, dist_norm)
    local bonus = ADVANCED_SHAPING.priority_aim_bonus[category] or 1.0
    local wave = math.max(0, math.floor(tonumber(wave_number) or 0))
    local humans = math.max(0, math.floor(tonumber(num_humans) or 0))
    local dist = tonumber(dist_norm) or 1.0
    if category == "brain" and humans > 0 then
        bonus = bonus + 0.25
    elseif category == "projectile" and dist < 0.10 then
        bonus = bonus + 0.20
    elseif category == "enforcer" and wave >= ADVANCED_SHAPING.high_wave_threshold then
        bonus = bonus + 0.15
    elseif category == "spawner" and wave >= (ADVANCED_SHAPING.high_wave_threshold + 2) then
        bonus = bonus + 0.10
    end
    return math.max(0.35, bonus)
end

function compute_priority_aim_reward(fire_cmd, px16, py16, objects, wave_number, num_humans)
    if fire_cmd == nil or fire_cmd < 0 or fire_cmd > 7 then
        return 0.0
    end
    if px16 == nil or py16 == nil or objects == nil then
        return 0.0
    end

    local vec = FIRE_DIR_VEC[fire_cmd]
    if vec == nil then return 0.0 end
    local vx, vy = vec[1], vec[2]
    local is_diagonal = (vx ~= 0 and vy ~= 0)
    local cross_thresh = is_diagonal and (AIM_CROSS_THRESHOLD * 1.414) or AIM_CROSS_THRESHOLD

    local best_score = 0.0
    for _, obj in ipairs(objects) do
        local cat = obj.category
        if cat and AIM_TARGET_CATS[cat] then
            local dx = obj.x16 - px16
            local dy = obj.y16 - py16
            local forward = dx * vx + dy * vy
            local cross = math.abs(dx * vy - dy * vx)
            if forward >= AIM_MIN_FORWARD and cross <= cross_thresh then
                local dist = math.sqrt(dx * dx + dy * dy)
                local base_score = clamp01(1.0 - dist / 32768.0)
                local threat = clamp01(obj.threat or 0.0)
                local bonus = priority_target_bonus(cat, wave_number, num_humans, obj.dist_norm)
                local score = clamp01((0.55 * base_score + 0.45 * threat) * bonus)
                if score > best_score then
                    best_score = score
                end
            end
        end
    end
    return best_score
end

function compute_brain_guard_reward(move_cmd, fire_cmd, px16, py16, objects, wave_number, num_humans)
    if px16 == nil or py16 == nil or objects == nil then
        return 0.0
    end
    local wave = math.max(0, math.floor(tonumber(wave_number) or 0))
    local humans = math.max(0, math.floor(tonumber(num_humans) or 0))
    if wave < ADVANCED_SHAPING.brain_guard_wave or humans <= 0 then
        return 0.0
    end

    local best = nil
    local best_score = -1.0
    for _, obj in ipairs(objects) do
        if obj.category == "brain" then
            local dist = tonumber(obj.dist_norm) or 1.0
            local score = clamp01(1.0 - dist) + clamp01(obj.threat or 0.0)
            if best == nil or score > best_score then
                best = obj
                best_score = score
            end
        end
    end
    if best == nil then
        return 0.0
    end

    local move_score = movement_alignment_score(move_cmd, best.x16 - px16, best.y16 - py16)
    local fire_score = 0.0
    if fire_cmd ~= nil and fire_cmd >= 0 and fire_cmd <= 7 then
        local vec = FIRE_DIR_VEC[fire_cmd]
        if vec ~= nil then
            local dx = best.x16 - px16
            local dy = best.y16 - py16
            local forward = dx * vec[1] + dy * vec[2]
            local cross = math.abs(dx * vec[2] - dy * vec[1])
            local cross_thresh = ((vec[1] ~= 0 and vec[2] ~= 0) and (AIM_CROSS_THRESHOLD * 1.414)) or AIM_CROSS_THRESHOLD
            if forward >= AIM_MIN_FORWARD and cross <= cross_thresh then
                local dist = math.sqrt(dx * dx + dy * dy)
                fire_score = clamp01(1.0 - dist / 32768.0)
            end
        end
    end

    return math.max(move_score * 0.75, fire_score) * clamp01(1.15 - (tonumber(best.dist_norm) or 1.0))
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

function picture_collision_bounds(memory, pict_ptr)
    if pict_ptr == nil or pict_ptr <= 0 then
        return 0, 0, 1, 1
    end
    local cached = picture_bounds_cache[pict_ptr]
    if cached then
        return cached[1], cached[2], cached[3], cached[4]
    end

    local width = memory:read_u8(pict_ptr) or 0
    local height = memory:read_u8(pict_ptr + 1) or 0
    local data_ptr = read_u16_be(memory, pict_ptr + 2)
    local off_x = 0
    local off_y = 0
    local box_w = math.max(1, width * HUD_SCREEN_X_SCALE)
    local box_h = math.max(1, height)

    if width > 0 and height > 0 and data_ptr and data_ptr > 0 then
        local found = false
        local min_col = width * HUD_SCREEN_X_SCALE
        local max_col = -1
        local min_row = height
        local max_row = 0
        for row = 0, height - 1 do
            local row_base = data_ptr + (row * width)
            for col = 0, width - 1 do
                local byte = memory:read_u8(row_base + col) or 0
                local x0 = col * HUD_SCREEN_X_SCALE
                if ((byte >> 4) & 0x0F) ~= 0 then
                    found = true
                    if x0 < min_col then min_col = x0 end
                    if x0 > max_col then max_col = x0 end
                    if row < min_row then min_row = row end
                    if row > max_row then max_row = row end
                end
                if (byte & 0x0F) ~= 0 then
                    local x1 = x0 + 1
                    found = true
                    if x1 < min_col then min_col = x1 end
                    if x1 > max_col then max_col = x1 end
                    if row < min_row then min_row = row end
                    if row > max_row then max_row = row end
                end
            end
        end
        if found then
            off_x = min_col
            off_y = min_row
            box_w = (max_col - min_col) + 1
            box_h = (max_row - min_row) + 1
        end
    end

    picture_bounds_cache[pict_ptr] = {off_x, off_y, box_w, box_h}
    return off_x, off_y, box_w, box_h
end

local function collision_center_x16(base_x16, off_x_px, box_w_px)
    local shift_px = (tonumber(off_x_px) or 0.0) + (0.5 * math.max(1.0, tonumber(box_w_px) or 1.0)) - 6.0
    return math.floor((tonumber(base_x16) or 0.0) + (shift_px * 256.0) + 0.5)
end

local function collision_center_y16(base_y16, off_y_px, box_h_px)
    local shift_px = (tonumber(off_y_px) or 0.0) + (0.5 * math.max(1.0, tonumber(box_h_px) or 1.0)) - 7.0
    return math.floor((tonumber(base_y16) or 0.0) + (shift_px * 256.0) + 0.5)
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

local function _grid_index(ix, iy, ch)
    return ((ch * GRID_H + iy) * GRID_W + ix) + 1
end

function _clamp_grid(v, lo, hi)
    if v < lo then return lo end
    if v > hi then return hi end
    return v
end

local function _object_threat_score(obj)
    local base = CATEGORY_THREAT_WEIGHT[obj.category] or 0.4
    local proximity = 1.0 - clamp01(obj.dist_norm or 1.0)
    local approach = clamp01((obj.approach or 0.0 + 1.0) * 0.5)
    if obj.category == "human" then
        return base * (0.35 + proximity)
    end
    return clamp01(base * (0.55 + 0.45 * proximity) + (0.25 * approach))
end

local function _salience_score(obj, roster_name)
    local threat = clamp01(tonumber(obj.threat) or 0.0)
    local proximity = clamp01(1.0 - (tonumber(obj.dist_norm) or 1.0))
    local approach = clamp01(((tonumber(obj.approach) or 0.0) + 1.0) * 0.5)
    local score = (0.55 * threat) + (0.30 * proximity) + (0.15 * approach)
    if roster_name == "non_hulk" then
        if obj.category == "projectile" then
            score = score + 0.28
        elseif obj.category == "brain" then
            score = score + 0.18
        elseif obj.category == "tank" or obj.category == "spawner" or obj.category == "enforcer" then
            score = score + 0.12
        end
    elseif roster_name == "hulk" then
        score = score + 0.22
    elseif roster_name == "electrode" then
        score = score + 0.10
    end
    return score
end

local function _assign_persistent_roster(bucket, prev_slots, slot_count, roster_name)
    local sorted = {}
    local by_ptr = {}
    for i = 1, #bucket do
        local obj = bucket[i]
        obj._salience = _salience_score(obj, roster_name)
        sorted[#sorted + 1] = obj
        by_ptr[obj.ptr] = obj
    end
    table.sort(sorted, function(a, b)
        local sa = tonumber(a._salience) or 0.0
        local sb = tonumber(b._salience) or 0.0
        if sa == sb then
            return (tonumber(a.dist_norm) or 1.0) < (tonumber(b.dist_norm) or 1.0)
        end
        return sa > sb
    end)

    local keep_threshold = 0.18
    local assigned = {}
    local used = {}
    local next_slots = {}

    for slot_idx = 1, slot_count do
        local prev_ptr = prev_slots[slot_idx] or 0
        local obj = by_ptr[prev_ptr]
        if obj ~= nil and (tonumber(obj._salience) or 0.0) >= keep_threshold then
            assigned[slot_idx] = obj
            used[prev_ptr] = true
        end
    end

    local cursor = 1
    for slot_idx = 1, slot_count do
        if assigned[slot_idx] == nil then
            while cursor <= #sorted do
                local obj = sorted[cursor]
                cursor = cursor + 1
                if obj ~= nil and not used[obj.ptr] then
                    assigned[slot_idx] = obj
                    used[obj.ptr] = true
                    break
                end
            end
        end
        next_slots[slot_idx] = assigned[slot_idx] and assigned[slot_idx].ptr or 0
    end

    return assigned, next_slots
end

local function _pressure_to_unit(v)
    local x = math.max(0.0, tonumber(v) or 0.0)
    return clamp01(x / (x + 1.25))
end

local function _emit_compact_slot_features(out, slots)
    local type_denom = math.max(1, UNIFIED_NUM_TYPES - 1)
    for i = 1, #slots do
        local obj = slots[i]
        if obj ~= nil then
            out[#out + 1] = 1.0
            out[#out + 1] = clamp11(obj.dx or 0.0)
            out[#out + 1] = clamp11(obj.dy or 0.0)
            out[#out + 1] = clamp11(obj.vx or 0.0)
            out[#out + 1] = clamp11(obj.vy or 0.0)
            out[#out + 1] = clamp01(obj.dist_norm or 1.0)
            out[#out + 1] = clamp01(obj.threat or 0.0)
            out[#out + 1] = (UNIFIED_TYPE_ID[obj.category] or 0) / type_denom
        else
            for _ = 1, COMPACT_SLOT_STATE_FEATURES do
                out[#out + 1] = 0.0
            end
        end
    end
end

local function extract_world_features(memory, player_x16, player_y16, _enemy_state)
    -- Walk live objects, score a persistent 24-slot salient roster, and emit a
    -- compact per-frame summary block for the Python learner.

    local player_pict_ptr = read_u16_be(memory, PLOBJ_ADDR + OPICT_OFF)
    local player_hit_off_x, player_hit_off_y, player_hit_w, player_hit_h = picture_collision_bounds(memory, player_pict_ptr)
    local player_center_x16 = collision_center_x16(player_x16, player_hit_off_x, player_hit_w)
    local player_center_y16 = collision_center_y16(player_y16, player_hit_off_y, player_hit_h)

    local all_objects = {}
    local hud_enabled = DEBUG_HUD_ENABLED
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
            if ocvect == 0 then
                ptr = read_u16_be(memory, ptr + OLINK_OFF)
            else
                local raw_x16 = read_u16_be(memory, ptr + OX16_OFF)
                local raw_y16 = read_u16_be(memory, ptr + OY16_OFF)
                local objx = hud_enabled and (memory:read_u8(ptr + OBJX_OFF) or 0) or nil
                local objy = hud_enabled and (memory:read_u8(ptr + OBJY_OFF) or 0) or nil
                local pict_ptr = read_u16_be(memory, ptr + OPICT_OFF)
                local fonipc_ptr = read_u16_be(memory, ptr + FONIPC_OFF)
                local width, height = 0, 0
                if pict_ptr > 0 then
                    width = memory:read_u8(pict_ptr)
                    height = memory:read_u8(pict_ptr + 1)
                end
                local collision_pict_ptr = (fonipc_ptr ~= nil and fonipc_ptr ~= 0) and fonipc_ptr or pict_ptr
                local hit_off_x, hit_off_y, hit_w, hit_h = picture_collision_bounds(memory, collision_pict_ptr)
                local x16 = collision_center_x16(raw_x16, hit_off_x, hit_w)
                local y16 = collision_center_y16(raw_y16, hit_off_y, hit_h)

                local rel_dx16 = 0
                local rel_dy16 = 0
                local dist_world = 0.0
                local dnorm = 1.0
                if player_center_x16 and player_center_y16 then
                    rel_dx16 = x16 - player_center_x16
                    rel_dy16 = y16 - player_center_y16
                    local d2 = (rel_dx16 * rel_dx16) + (rel_dy16 * rel_dy16)
                    if d2 > 0 then
                        dist_world = math.sqrt(d2)
                        dnorm = clamp01(dist_world / POS_MAX_DIAG)
                    else
                        dnorm = 0.0
                    end
                end

                local cat = ocvect_category_cache[ocvect]
                if cat == nil then
                    cat = classify_by_heuristic(list_def.name, width, height)
                    if cat ~= nil then
                        ocvect_category_cache[ocvect] = cat
                        if cat == "tank" and width ~= 7 then
                            discovered_tank_ocvect = ocvect
                        end
                        if cat ~= "skip" and DEBUG_LOG_DISCOVERY then
                            print(string.format("[DISCOVERY] OCVECT 0x%04X → %s (list=%s dim=%dx%d)",
                                ocvect, cat, list_def.name, width, height))
                        end
                    else
                        unresolved_7x16[ocvect] = true
                    end
                end

                all_objects[#all_objects + 1] = {
                    ptr = ptr,
                    list_name = list_def.name,
                    x16 = x16,
                    y16 = y16,
                    ocvect = ocvect,
                    width = width,
                    height = height,
                    hit_w = hit_w,
                    hit_h = hit_h,
                    rel_dx16 = rel_dx16,
                    rel_dy16 = rel_dy16,
                    dist_world = dist_world,
                    dist_norm = dnorm,
                    category = cat,
                }
                if hud_enabled then
                    local obj = all_objects[#all_objects]
                    obj.objx = objx
                    obj.objy = objy
                    obj.hit_off_x = hit_off_x
                    obj.hit_off_y = hit_off_y
                end
                ptr = read_u16_be(memory, ptr + OLINK_OFF)
            end
        end
    end

    if next(unresolved_7x16) then
        if try_resolve_7x16(all_objects, enemy_state) then
            for _, obj in ipairs(all_objects) do
                if obj.category == nil then
                    obj.category = ocvect_category_cache[obj.ocvect]
                end
            end
        end
    end

    local buckets = {}
    local counts = {}
    local classified_objects = hud_enabled and {} or nil
    local object_count = 0
    local nearest_enemy_dist = nil
    local nearest_human_dist = nil
    local nearest_hulk_dist = nil
    local nearest_electrode_dist = nil
    local nearest_enemy_x16 = nil
    local nearest_enemy_y16 = nil
    local nearest_human_x16 = nil
    local nearest_human_y16 = nil
    for _, cat in ipairs(ENTITY_CATEGORIES) do
        buckets[cat.name] = {}
        counts[cat.name] = 0
    end

    local dangerous_bucket = {}
    local human_bucket = buckets["human"]
    local electrode_bucket = buckets["electrode"]
    local non_hulk_bucket = {}
    local hulk_bucket = buckets["hulk"]
    local current_sample_x = {}
    local current_sample_y = {}
    for _, obj in ipairs(all_objects) do
        if obj.category == nil and obj.list_name == "rptr" then
            obj.category = "hulk"
        end
        if obj.category and obj.category ~= "skip" and buckets[obj.category] then
            local vx = 0.0
            local vy = 0.0
            local prev_x = prev_object_sample_x[obj.ptr]
            if prev_x ~= nil then
                vx = clamp11((obj.x16 - prev_x) / POS_X_RANGE)
                vy = clamp11((obj.y16 - (prev_object_sample_y[obj.ptr] or obj.y16)) / POS_Y_RANGE)
            end
            obj.vx = vx
            obj.vy = vy
            obj.dx = clamp11((obj.rel_dx16 or 0.0) / POS_X_RANGE)
            obj.dy = clamp11((obj.rel_dy16 or 0.0) / POS_Y_RANGE)
            local dist_world = obj.dist_world or 0.0
            if dist_world > 1.0 then
                obj.dir_x = clamp11((obj.rel_dx16 or 0.0) / dist_world)
                obj.dir_y = clamp11((obj.rel_dy16 or 0.0) / dist_world)
            else
                obj.dir_x = 0.0
                obj.dir_y = 0.0
            end
            local radial = -((obj.vx * obj.dir_x) + (obj.vy * obj.dir_y))
            obj.approach = clamp11(radial * 2.0)
            obj.threat = _object_threat_score(obj)

            counts[obj.category] = counts[obj.category] + 1
            current_sample_x[obj.ptr] = obj.x16
            current_sample_y[obj.ptr] = obj.y16
            buckets[obj.category][#buckets[obj.category] + 1] = obj
            if obj.category ~= "human" and obj.category ~= "electrode" then
                dangerous_bucket[#dangerous_bucket + 1] = obj
            end
            if obj.category == "hulk" then
                hulk_bucket[#hulk_bucket + 1] = obj
            elseif obj.category ~= "human" and obj.category ~= "electrode" then
                non_hulk_bucket[#non_hulk_bucket + 1] = obj
            end
            object_count = object_count + 1
            if hud_enabled then
                classified_objects[#classified_objects + 1] = obj
            end

            if obj.category == "human" then
                if nearest_human_dist == nil or obj.dist_norm < nearest_human_dist then
                    nearest_human_dist = obj.dist_norm
                    nearest_human_x16 = obj.x16
                    nearest_human_y16 = obj.y16
                end
            elseif obj.category == "hulk" then
                if nearest_hulk_dist == nil or obj.dist_norm < nearest_hulk_dist then
                    nearest_hulk_dist = obj.dist_norm
                end
                if nearest_enemy_dist == nil or obj.dist_norm < nearest_enemy_dist then
                    nearest_enemy_dist = obj.dist_norm
                    nearest_enemy_x16 = obj.x16
                    nearest_enemy_y16 = obj.y16
                end
            elseif obj.category == "electrode" then
                if nearest_electrode_dist == nil or obj.dist_norm < nearest_electrode_dist then
                    nearest_electrode_dist = obj.dist_norm
                end
            elseif CATEGORY_IS_DANGEROUS[obj.category] then
                if nearest_enemy_dist == nil or obj.dist_norm < nearest_enemy_dist then
                    nearest_enemy_dist = obj.dist_norm
                    nearest_enemy_x16 = obj.x16
                    nearest_enemy_y16 = obj.y16
                end
            end
        end
    end

    prev_object_sample_x = current_sample_x
    prev_object_sample_y = current_sample_y

    local non_hulk_slots, next_non_hulk = _assign_persistent_roster(
        non_hulk_bucket,
        prev_compact_slot_ptrs.non_hulk,
        COMPACT_NON_HULK_SLOTS,
        "non_hulk"
    )
    local hulk_slots, next_hulk = _assign_persistent_roster(
        hulk_bucket,
        prev_compact_slot_ptrs.hulk,
        COMPACT_HULK_SLOTS,
        "hulk"
    )
    local electrode_slots, next_electrode = _assign_persistent_roster(
        electrode_bucket,
        prev_compact_slot_ptrs.electrode,
        COMPACT_ELECTRODE_SLOTS,
        "electrode"
    )
    prev_compact_slot_ptrs.non_hulk = next_non_hulk
    prev_compact_slot_ptrs.hulk = next_hulk
    prev_compact_slot_ptrs.electrode = next_electrode

    if hud_enabled then
        for _, cat in ipairs(ENTITY_CATEGORIES) do
            local bucket = buckets[cat.name]
            if #bucket > 0 then
                local ranked = {}
                for i = 1, #bucket do
                    ranked[i] = bucket[i]
                end
                table.sort(ranked, function(a, b)
                    if a.threat == b.threat then
                        return a.dist_norm < b.dist_norm
                    end
                    return a.threat > b.threat
                end)
                for i, obj in ipairs(ranked) do
                    obj.rank = i
                end
            end
        end
        hud_objects = classified_objects
    else
        hud_objects = nil
    end

    local threat_left = 0.0
    local threat_right = 0.0
    local threat_up = 0.0
    local threat_down = 0.0
    local crowding_score = 0.0
    local projectile_pressure = 0.0
    local escape_x = 0.0
    local escape_y = 0.0
    local safe_fire_opportunity = 0.0

    for _, obj in ipairs(dangerous_bucket) do
        local dir_x = tonumber(obj.dir_x) or 0.0
        local dir_y = tonumber(obj.dir_y) or 0.0
        local pressure = clamp01((0.45 * (tonumber(obj.threat) or 0.0)) + (0.55 * (1.0 - (tonumber(obj.dist_norm) or 1.0))))
        threat_left = threat_left + (pressure * math.max(0.0, -dir_x))
        threat_right = threat_right + (pressure * math.max(0.0, dir_x))
        threat_up = threat_up + (pressure * math.max(0.0, -dir_y))
        threat_down = threat_down + (pressure * math.max(0.0, dir_y))
        crowding_score = crowding_score + pressure
        escape_x = escape_x - (dir_x * pressure)
        escape_y = escape_y - (dir_y * pressure)
        if obj.category == "projectile" then
            projectile_pressure = projectile_pressure + pressure
        end
        if obj.category ~= "hulk" then
            local align = math.max(
                clamp01(1.0 - (math.abs(obj.dx or 0.0) / 0.12)),
                clamp01(1.0 - (math.abs(obj.dy or 0.0) / 0.12))
            )
            local shot_window = clamp01(((tonumber(obj.dist_norm) or 1.0) - 0.02) / 0.20)
            safe_fire_opportunity = math.max(
                safe_fire_opportunity,
                clamp01(align * shot_window * (1.0 - _pressure_to_unit(projectile_pressure)))
            )
        end
    end
    for _, obj in ipairs(electrode_bucket) do
        local dir_x = tonumber(obj.dir_x) or 0.0
        local dir_y = tonumber(obj.dir_y) or 0.0
        local pressure = clamp01((0.35 * (tonumber(obj.threat) or 0.0)) + (0.65 * (1.0 - (tonumber(obj.dist_norm) or 1.0))))
        threat_left = threat_left + (pressure * math.max(0.0, -dir_x))
        threat_right = threat_right + (pressure * math.max(0.0, dir_x))
        threat_up = threat_up + (pressure * math.max(0.0, -dir_y))
        threat_down = threat_down + (pressure * math.max(0.0, dir_y))
        crowding_score = crowding_score + (pressure * 0.65)
        escape_x = escape_x - (dir_x * pressure)
        escape_y = escape_y - (dir_y * pressure)
    end

    local threat_left_n = _pressure_to_unit(threat_left)
    local threat_right_n = _pressure_to_unit(threat_right)
    local threat_up_n = _pressure_to_unit(threat_up)
    local threat_down_n = _pressure_to_unit(threat_down)

    local px_norm = norm_pos_x(player_x16 or 0)
    local py_norm = norm_pos_y(player_y16 or 0)
    local wall_left = clamp01((WALL_MARGIN_NORM_X - px_norm) / math.max(WALL_MARGIN_NORM_X, 1e-6))
    local wall_right = clamp01((px_norm - (1.0 - WALL_MARGIN_NORM_X)) / math.max(WALL_MARGIN_NORM_X, 1e-6))
    local wall_up = clamp01((WALL_MARGIN_NORM_Y - py_norm) / math.max(WALL_MARGIN_NORM_Y, 1e-6))
    local wall_down = clamp01((py_norm - (1.0 - WALL_MARGIN_NORM_Y)) / math.max(WALL_MARGIN_NORM_Y, 1e-6))

    local open_left = clamp01(1.0 - threat_left_n - (0.65 * wall_left))
    local open_right = clamp01(1.0 - threat_right_n - (0.65 * wall_right))
    local open_up = clamp01(1.0 - threat_up_n - (0.65 * wall_up))
    local open_down = clamp01(1.0 - threat_down_n - (0.65 * wall_down))

    local escape_len = math.sqrt((escape_x * escape_x) + (escape_y * escape_y))
    if escape_len > 1e-6 then
        escape_x = clamp11(escape_x / escape_len)
        escape_y = clamp11(escape_y / escape_len)
    else
        escape_x = 0.0
        escape_y = 0.0
    end

    local human_count_norm = clamp01((counts["human"] or 0) / 16.0)
    local rescue_opportunity = 0.0
    if nearest_human_dist ~= nil then
        rescue_opportunity = clamp01((1.0 - nearest_human_dist) * (1.0 - _pressure_to_unit(crowding_score)))
    end

    local state_features = {
        1.0, -- alive flag overwritten in serialize_frame
        clamp01((tonumber(previous_wave_number or 0) or 0.0) / 40.0), -- overwritten in serialize_frame
        0.0, -- lasers overwritten in serialize_frame
        px_norm,
        py_norm,
        0.0, -- vel_x overwritten in serialize_frame
        0.0, -- vel_y overwritten in serialize_frame
        clamp01(nearest_enemy_dist or 1.0),
        clamp01(nearest_hulk_dist or 1.0),
        clamp01(nearest_electrode_dist or 1.0),
        clamp01(nearest_human_dist or 1.0),
        clamp11(rel_pos_x(nearest_human_x16 or player_x16, player_x16)),
        clamp11(rel_pos_y(nearest_human_y16 or player_y16, player_y16)),
        human_count_norm,
        threat_left_n,
        threat_right_n,
        threat_up_n,
        threat_down_n,
        open_left,
        open_right,
        open_up,
        open_down,
        _pressure_to_unit(crowding_score),
        _pressure_to_unit(projectile_pressure),
        escape_x,
        escape_y,
        rescue_opportunity,
        clamp01(safe_fire_opportunity),
    }

    local slot_features = {}
    _emit_compact_slot_features(slot_features, non_hulk_slots)
    _emit_compact_slot_features(slot_features, hulk_slots)
    _emit_compact_slot_features(slot_features, electrode_slots)

    for i = 1, #slot_features do
        state_features[#state_features + 1] = slot_features[i]
    end

    local num_humans = counts["human"] or 0

    hud_player_x16 = memory:read_u8(PLOBJ_ADDR + OBJX_OFF)
    hud_player_y16 = memory:read_u8(PLOBJ_ADDR + OBJY_OFF)
    hud_player_box = {x = player_hit_off_x, y = player_hit_off_y, w = player_hit_w, h = player_hit_h}

    return {
        object_count = object_count,
        nearest_enemy_dist = nearest_enemy_dist,
        nearest_human_dist = nearest_human_dist,
        nearest_enemy_x16 = nearest_enemy_x16,
        nearest_enemy_y16 = nearest_enemy_y16,
        nearest_human_x16 = nearest_human_x16,
        nearest_human_y16 = nearest_human_y16,
        player_center_x16 = player_center_x16,
        player_center_y16 = player_center_y16,
        num_humans = num_humans,
        state_features = state_features,
    }
end

-- ── Debug HUD: draw coloured rings + rank numbers on MAME screen ────────

CAT_HUD_COLOR = {
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
HUD_PLAYER_COLOR = 0xFFFFFFFF   -- white box for player
HUD_PLAYER_BOX_W = 4           -- matches expert collision box width in pixels
HUD_PLAYER_BOX_H = 12          -- matches expert collision box height in pixels
HUD_SCREEN_X_SCALE = 2         -- Robotron pixels are doubled horizontally on screen

function draw_debug_hud()
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

    -- Helper: draw the tight non-zero bitmap bounds used by collision art.
    local function draw_hitbox(x_px, y_px, off_x_px, off_y_px, w_px, h_px, color)
        local left = (x_px * HUD_SCREEN_X_SCALE) + off_x_px - 6
        local top = y_px + off_y_px - 7
        local right = left + math.max(1, w_px) - 1
        local bottom = top + math.max(1, h_px) - 1
        mame_screen:draw_line(left, top, right, top, color)
        mame_screen:draw_line(right, top, right, bottom, color)
        mame_screen:draw_line(right, bottom, left, bottom, color)
        mame_screen:draw_line(left, bottom, left, top, color)
    end

    -- Player hit box
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
        local px = hud_player_x16
        local py = hud_player_y16
        local player_box = hud_player_box or {x = 0, y = 0, w = HUD_PLAYER_BOX_W, h = HUD_PLAYER_BOX_H}
        pcall(function() draw_hitbox(px, py, player_box.x or 0, player_box.y or 0, player_box.w or HUD_PLAYER_BOX_W, player_box.h or HUD_PLAYER_BOX_H, player_color) end)
    end

    -- Entity hit boxes + rank numbers
    if hud_objects then
        for _, obj in ipairs(hud_objects) do
            if obj.category and obj.category ~= "skip" then
                local color = CAT_HUD_COLOR[obj.category]
                if color then
                    local sx = obj.objx or 0
                    local sy = obj.objy or 0
                    local off_x = obj.hit_off_x or 0
                    local off_y = obj.hit_off_y or 0
                    local w = math.max(obj.hit_w or obj.width or 1, 1)
                    local h = math.max(obj.hit_h or obj.height or 1, 1)
                    pcall(function() draw_hitbox(sx, sy, off_x, off_y, w, h, color) end)
                    if obj.rank then
                        pcall(function()
                            mame_screen:draw_text((sx * HUD_SCREEN_X_SCALE) + off_x - 9, sy + off_y + h - 7,
                                                  tostring(obj.rank), color, 0x00000000)
                        end)
                    end
                end
            end
        end
    end
end

function clear_pending_preview()
    pending_preview_blob = nil
    pending_preview_w = 0
    pending_preview_h = 0
    pending_preview_fmt = PREVIEW_FORMAT_RGB565
end

local function lzss_compress_bytes(data)
    local n = #data
    if n <= 0 then
        return ""
    end
    local src = {string.byte(data, 1, n)}
    local dict = {}
    local out = {}
    local i = 1

    local function dict_push(pos)
        if (pos + 2) > n then
            return
        end
        local k = src[pos] * 65536 + src[pos + 1] * 256 + src[pos + 2]
        local bucket = dict[k]
        if not bucket then
            dict[k] = {pos}
            return
        end
        bucket[#bucket + 1] = pos
        if #bucket > 32 then
            table.remove(bucket, 1)
        end
    end

    while i <= n do
        local flag_idx = #out + 1
        out[flag_idx] = string.char(0)
        local flags = 0

        for bit = 0, 7 do
            if i > n then
                break
            end

            local best_len = 0
            local best_dist = 0
            if (i + 2) <= n then
                local k = src[i] * 65536 + src[i + 1] * 256 + src[i + 2]
                local bucket = dict[k]
                if bucket then
                    local checks = 0
                    for bi = #bucket, 1, -1 do
                        local p = bucket[bi]
                        local dist = i - p
                        if dist > 0 and dist <= 4095 then
                            local ml = 0
                            local maxl = math.min(18, n - i + 1)
                            while ml < maxl and src[p + ml] == src[i + ml] do
                                ml = ml + 1
                            end
                            if ml > best_len and ml >= 3 then
                                best_len = ml
                                best_dist = dist
                                if best_len >= 18 then
                                    break
                                end
                            end
                            checks = checks + 1
                            if checks >= 16 then
                                break
                            end
                        end
                    end
                end
            end

            if best_len >= 3 then
                flags = flags | (1 << bit)
                local b1 = ((best_len - 3) << 4) | ((best_dist >> 8) & 0x0F)
                local b2 = best_dist & 0xFF
                out[#out + 1] = string.char(b1, b2)
                for p = i, (i + best_len - 1) do
                    dict_push(p)
                end
                i = i + best_len
            else
                out[#out + 1] = string.char(src[i])
                dict_push(i)
                i = i + 1
            end
        end

        out[flag_idx] = string.char(flags)
    end

    return table.concat(out)
end

local function rle_compress_rgb565_words(blob)
    if not blob or #blob < 4 or (#blob % 2) ~= 0 then
        return nil
    end
    local n = #blob
    local out = {}
    local oi = 1
    local i = 1

    while i <= n do
        local word = string.sub(blob, i, i + 1)
        local run_words = 1
        local j = i + 2
        while run_words < 128 and j <= n do
            if string.sub(blob, j, j + 1) ~= word then
                break
            end
            run_words = run_words + 1
            j = j + 2
        end

        if run_words >= 2 then
            out[oi] = string.char(0x80 | (run_words - 1))
            oi = oi + 1
            out[oi] = word
            oi = oi + 1
            i = i + (run_words * 2)
        else
            local lit_start = i
            local lit_words = 1
            i = i + 2
            while lit_words < 128 and i <= n do
                if (i + 3) <= n and string.sub(blob, i, i + 1) == string.sub(blob, i + 2, i + 3) then
                    break
                end
                lit_words = lit_words + 1
                i = i + 2
            end
            out[oi] = string.char(lit_words - 1)
            oi = oi + 1
            out[oi] = string.sub(blob, lit_start, lit_start + (lit_words * 2) - 1)
            oi = oi + 1
        end
    end

    return table.concat(out)
end

local function capture_game_preview()
    if (not PREVIEW_CAPTURE_ENABLED) or (not preview_stream_enabled) then
        clear_pending_preview()
        return
    end
    local now_s = os.clock()
    if (now_s - last_preview_capture_time_s) < PREVIEW_MIN_INTERVAL_S then
        return
    end
    last_preview_capture_time_s = now_s

    local ok, result = pcall(function()
        local vw, vh = manager.machine.video:snapshot_size()
        if not vw or not vh or vw <= 0 or vh <= 0 then
            return nil
        end
        local src = manager.machine.video:snapshot_pixels()
        if not src or #src < (vw * vh * 4) then
            return nil
        end

        if vw ~= preview_source_w or vh ~= preview_source_h or preview_target_w <= 0 or preview_target_h <= 0 then
            preview_source_w = vw
            preview_source_h = vh
            local scale = math.min(1.0, PREVIEW_MAX_WIDTH / vw, PREVIEW_MAX_HEIGHT / vh)
            preview_target_w = math.max(1, math.floor(vw * scale + 0.5))
            preview_target_h = math.max(1, math.floor(vh * scale + 0.5))
        end

        local tw = preview_target_w
        local th = preview_target_h

        local out = {}
        local oi = 1
        for ty = 0, th - 1 do
            local sy = math.floor(((ty + 0.5) * vh) / th)
            if sy >= vh then sy = vh - 1 end
            local row_base = sy * vw
            for tx = 0, tw - 1 do
                local sx = math.floor(((tx + 0.5) * vw) / tw)
                if sx >= vw then sx = vw - 1 end
                local px_off = ((row_base + sx) * 4) + 1
                local px = string.unpack("=I4", src, px_off)
                local r = (px >> 16) & 0xFF
                local g = (px >> 8) & 0xFF
                local b = px & 0xFF
                local rgb565 = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)
                out[oi] = string.pack(">I2", rgb565)
                oi = oi + 1
            end
        end

        local blob = table.concat(out)
        local fmt = PREVIEW_FORMAT_RGB565
        if PREVIEW_TRY_RLE and #blob >= 4096 then
            local rle = rle_compress_rgb565_words(blob)
            if rle and #rle > 0 and (#blob - #rle) >= PREVIEW_RLE_MIN_SAVINGS_BYTES then
                fmt = PREVIEW_FORMAT_RGB565_RLE
                blob = rle
            end
        end
        if #blob <= 0 or #blob > PREVIEW_MAX_BYTES then
            return nil
        end
        return {w = tw, h = th, fmt = fmt, blob = blob}
    end)

    if ok and result and result.blob then
        pending_preview_w = result.w
        pending_preview_h = result.h
        pending_preview_fmt = result.fmt or PREVIEW_FORMAT_RGB565
        pending_preview_blob = result.blob
    end
end

local function frame_done_callback()
    draw_debug_hud()
    if PREVIEW_CAPTURE_ENABLED then
        capture_game_preview()
    else
        clear_pending_preview()
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

local function rom_region_offset_for_cpu_addr(addr)
    if addr >= 0x0000 and addr <= 0x8FFF then
        return 0x10000 + addr
    end
    if addr >= 0xD000 and addr <= 0xFFFF then
        return addr
    end
    return nil
end

local function compute_rom_page_sum(rom, page_base_addr)
    local region_base = rom_region_offset_for_cpu_addr(page_base_addr)
    if region_base == nil then
        error(string.format("unsupported ROMTAB page base $%04X", page_base_addr))
    end
    local total = 0
    for i = 0, 0x0FFF do
        total = (total + (rom:read_u8(region_base + i) or 0)) & 0xFF
    end
    return total
end

local function romtab_sum_addr_for_page(page_index)
    return ROMTAB_BASE_ADDR + (page_index * 2) + 1
end

local function rebalance_f000_page_checksum(rom)
    local f_page_index = 0xF
    local f_page_base = f_page_index * 0x1000
    local expected_sum_addr = romtab_sum_addr_for_page(f_page_index)
    local expected_sum_region_addr = rom_region_offset_for_cpu_addr(expected_sum_addr)
    local fudger_region_addr = rom_region_offset_for_cpu_addr(0xFFD6)
    if expected_sum_region_addr == nil or fudger_region_addr == nil then
        error("unable to locate F000 ROMTAB checksum fields")
    end

    local target_sum = rom:read_u8(expected_sum_region_addr) or 0
    local current_sum = compute_rom_page_sum(rom, f_page_base)
    local delta = (target_sum - current_sum) & 0xFF
    if delta ~= 0 then
        local old_fudger = rom:read_u8(fudger_region_addr) or 0
        rom:write_u8(fudger_region_addr, (old_fudger + delta) & 0xFF)
    end

    local final_sum = compute_rom_page_sum(rom, f_page_base)
    if final_sum ~= target_sum then
        error(string.format(
            "F000 checksum rebalance failed: sum=%02X target=%02X",
            final_sum,
            target_sum
        ))
    end

    return delta ~= 0
end

local function apply_rrchris_patch()
    if not RRCHRIS_PATCH_ENABLED then
        print("[PATCH] RRCHRIS runtime ROM patch disabled.")
        return true
    end
    if rrchris_patch_applied then
        return true
    end

    local success, result = pcall(function()
        local regions = manager.machine.memory.regions
        if not regions then
            error("MAME memory regions not available")
        end
        local rom = regions[RRCHRIS_PATCH_REGION]
        if not rom then
            error("ROM region not found: " .. RRCHRIS_PATCH_REGION)
        end

        local changed_bytes = 0
        local already_applied = true

        for _, chunk in ipairs(RRCHRIS_PATCH_CHUNKS) do
            local base = rom_region_offset_for_cpu_addr(chunk.addr)
            if base == nil then
                error(string.format("unsupported CPU patch address $%04X", chunk.addr))
            end
            if (base + #chunk.bytes - 1) >= rom.size then
                error(string.format("patch address $%04X exceeds region bounds", chunk.addr))
            end
            for i, expected in ipairs(chunk.bytes) do
                local cur = rom:read_u8(base + i - 1)
                if cur ~= expected then
                    already_applied = false
                    changed_bytes = changed_bytes + 1
                end
            end
        end

        if already_applied then
            return { already_applied = true, changed_bytes = 0 }
        end

        for _, chunk in ipairs(RRCHRIS_PATCH_CHUNKS) do
            local base = rom_region_offset_for_cpu_addr(chunk.addr)
            for i, value in ipairs(chunk.bytes) do
                rom:write_u8(base + i - 1, value)
            end
        end

        local page_index = 0x4
        local page_base_addr = page_index * 0x1000
        local romtab_sum_addr = romtab_sum_addr_for_page(page_index)
        local romtab_region_addr = rom_region_offset_for_cpu_addr(romtab_sum_addr)
        if romtab_region_addr == nil then
            error(string.format("unsupported ROMTAB sum address $%04X", romtab_sum_addr))
        end
        local expected_sum = compute_rom_page_sum(rom, page_base_addr)
        if rom:read_u8(romtab_region_addr) ~= expected_sum then
            changed_bytes = changed_bytes + 1
            rom:write_u8(romtab_region_addr, expected_sum)
        end
        if rebalance_f000_page_checksum(rom) then
            changed_bytes = changed_bytes + 1
        end

        for _, chunk in ipairs(RRCHRIS_PATCH_CHUNKS) do
            local base = rom_region_offset_for_cpu_addr(chunk.addr)
            for i, expected in ipairs(chunk.bytes) do
                local actual = rom:read_u8(base + i - 1)
                if actual ~= expected then
                    error(string.format(
                        "verification failed at $%04X: expected %02X got %02X",
                        chunk.addr + i - 1, expected, actual
                    ))
                end
            end
        end

        local page_sum_check = compute_rom_page_sum(rom, 0x4000)
        local romtab_expected = rom:read_u8(romtab_region_addr)
        if page_sum_check ~= romtab_expected then
            error(string.format(
                "ROMTAB verification failed for $4000 page: sum=%02X table=%02X",
                page_sum_check,
                romtab_expected
            ))
        end
        local f000_expected = rom:read_u8(rom_region_offset_for_cpu_addr(0xFFD4))
        local f000_sum_check = compute_rom_page_sum(rom, 0xF000)
        if f000_sum_check ~= f000_expected then
            error(string.format(
                "ROMTAB verification failed for $F000 page: sum=%02X table=%02X",
                f000_sum_check,
                f000_expected
            ))
        end

        return { already_applied = false, changed_bytes = changed_bytes }
    end)

    if not success then
        print("[PATCH] RRCHRIS patch failed: " .. tostring(result))
        return false
    end

    rrchris_patch_applied = true
    if result.already_applied then
        print("[PATCH] RRCHRIS enforcer fix already present in " .. RRCHRIS_PATCH_REGION .. ".")
    else
        print(string.format(
            "[PATCH] Applied RRCHRIS enforcer fix to %s (%d byte updates).",
            RRCHRIS_PATCH_REGION,
            result.changed_bytes
        ))
    end
    return true
end

local function close_socket()
    if current_socket then
        current_socket:close()
        current_socket = nil
    end
    ACTION_RX_BUFFER = ""
    preview_stream_enabled = false
end

local function open_socket()
    close_socket()

    local open_result = nil
    local ok, err = pcall(function()
        local sock = emu.file("rw")
        local result = sock:open(SOCKET_ADDRESS)
        if result == nil then
            -- Required 2-byte handshake:
            --   bit0     = preview-capable flag
            --   bits1-15 = launcher slot (stable audio/video identity)
            local preview_capable = (PREVIEW_CLIENT_FLAG ~= 0) and 1 or 0
            local handshake_u16 = math.max(0, math.min(65535, (CLIENT_SLOT * 2) + preview_capable))
            sock:write(string.pack(">H", handshake_u16))
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

DIR_AXES = {
    [0] = {1, 0, 0, 0}, -- up
    [1] = {1, 0, 0, 1}, -- up-right
    [2] = {0, 0, 0, 1}, -- right
    [3] = {0, 1, 0, 1}, -- down-right
    [4] = {0, 1, 0, 0}, -- down
    [5] = {0, 1, 1, 0}, -- down-left
    [6] = {0, 0, 1, 0}, -- left
    [7] = {1, 0, 1, 0}, -- up-left
}
DIR_NEUTRAL = {0, 0, 0, 0}

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
                               player_x16, player_y16, state_values,
                               done, subj_reward, obj_reward, save_signal, start_cmd,
                               preview_w, preview_h, preview_fmt, preview_blob)
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

    local values = {}
    for i = 1, #(state_values or {}) do
        values[i] = state_values[i]
    end
    values[1] = ((player_alive or 0) ~= 0) and 1.0 or 0.0
    values[2] = wave_norm
    values[3] = lasers_norm
    values[4] = norm_pos_x(player_x16 or 0)
    values[5] = norm_pos_y(player_y16 or 0)
    local vel_x = 0.0
    local vel_y = 0.0
    if prev_player_x16 and prev_player_y16 and player_alive ~= 0 then
        vel_x = clamp11(((player_x16 or 0) - prev_player_x16) / POS_X_RANGE)
        vel_y = clamp11(((player_y16 or 0) - prev_player_y16) / POS_Y_RANGE)
    end
    values[6] = vel_x
    values[7] = vel_y
    local num_values = #values
    if num_values ~= EXPECTED_STATE_VALUES then
        error(string.format("state size mismatch: got=%d expected=%d", num_values, EXPECTED_STATE_VALUES))
    end

    local header = string.pack(
        ">HddBIBBBIBB",
        num_values,
        subj_reward,
        obj_reward,
        done and 1 or 0,
        score_u32,
        player_alive,
        save_signal,
        math.max(0, math.min(1, math.floor(start_cmd or 0))),
        replay_u32,
        lasers_u8,
        wave_u8
    )

    local state_payload_parts = {}
    for i = 1, num_values do
        state_payload_parts[#state_payload_parts + 1] = string.pack(">f", values[i])
    end
    local state_payload = table.concat(state_payload_parts)

    local preview_chunk = ""
    local pw = math.max(0, math.floor(preview_w or 0))
    local ph = math.max(0, math.floor(preview_h or 0))
    local pf = math.max(0, math.min(255, math.floor(preview_fmt or PREVIEW_FORMAT_RGB565)))
    if preview_blob and pw > 0 and ph > 0 and #preview_blob > 0 and #preview_blob <= PREVIEW_MAX_BYTES then
        preview_chunk = string.pack(">HHB", pw, ph, pf) .. preview_blob
    end
    local preview_len = #preview_chunk
    local payload = header .. state_payload .. string.pack(">I4", preview_len) .. preview_chunk
    if #payload > SOCKET_MAX_PAYLOAD_BYTES then
        payload = header .. state_payload .. string.pack(">I4", 0)
    end
    return payload
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
        local length_header = string.pack(">I4", #frame_payload)
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
        trace_log(frame_idx, "socket_read_begin", "waiting for 3-byte or 5-byte action")
        local started = os.clock()
        local legacy_buffer_ready_at = nil

        while (os.clock() - started) < SOCKET_READ_TIMEOUT_S do
            if #ACTION_RX_BUFFER >= 5 then
                local action_bytes = string.sub(ACTION_RX_BUFFER, 1, 5)
                ACTION_RX_BUFFER = string.sub(ACTION_RX_BUFFER, 6)
                local move_dir, fire_dir, source, start_advanced, start_level_min = string.unpack("bbBBB", action_bytes)
                START_ADVANCED = (start_advanced or 0) ~= 0
                START_LEVEL_MIN = math.max(1, math.min(81, math.floor(start_level_min or 1)))
                trace_log(
                    frame_idx,
                    "socket_read_ok",
                    string.format(
                        "move=%d fire=%d src=%d adv=%d level=%d",
                        move_dir, fire_dir, source, start_advanced or 0, START_LEVEL_MIN
                    )
                )
                return {move_dir, fire_dir, source}
            end

            if #ACTION_RX_BUFFER == 3 then
                if legacy_buffer_ready_at == nil then
                    legacy_buffer_ready_at = os.clock() + 0.003
                elseif os.clock() >= legacy_buffer_ready_at then
                    local action_bytes = ACTION_RX_BUFFER
                    ACTION_RX_BUFFER = ""
                    local move_dir, fire_dir, source = string.unpack("bbB", action_bytes)
                    trace_log(frame_idx, "socket_read_ok_legacy", string.format("move=%d fire=%d src=%d", move_dir, fire_dir, source))
                    return {move_dir, fire_dir, source}
                end
            else
                legacy_buffer_ready_at = nil
            end

            local need = 5 - #ACTION_RX_BUFFER
            if need < 1 then
                need = 1
            end
            local chunk = current_socket:read(need)
            if chunk and #chunk > 0 then
                ACTION_RX_BUFFER = ACTION_RX_BUFFER .. chunk
            end
        end

        ACTION_RX_BUFFER = ""
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
    local source_u8 = (source or 0) & 0xFF
    if PREVIEW_CAPTURE_ENABLED then
        preview_stream_enabled = (source_u8 & 0x40) ~= 0
    else
        preview_stream_enabled = false
    end
    -- Source byte bits:
    --   low nibble = action source
    --   0x40 = preview enabled
    --   0x80 = HUD enabled
    DEBUG_HUD_ENABLED = (source_u8 & 0x80) ~= 0
    last_action_source = source_u8 & 0x0F
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

function read_frame_observation()
    local ok_alive, alive_or_err = pcall(read_player_alive, mem)
    if not ok_alive then
        return nil, "read_player_alive_error", tostring(alive_or_err)
    end
    local player_alive = (alive_or_err ~= 0) and 1 or 0
    trace_log(frame_counter, "read_player_alive", "alive=" .. tostring(player_alive))

    local ok_core, score, replay_level, num_lasers, wave_number = pcall(function()
        local raw_score = read_player_score(mem)
        if raw_score == nil then
            raw_score = previous_score
        end
        return math.max(0, math.floor(raw_score or 0)),
               math.max(0, math.floor(read_next_replay_level(mem) or 0)),
               math.max(0, math.floor(read_num_lasers(mem) or 0)),
               math.max(0, math.floor(read_wave_number(mem) or 0))
    end)
    if not ok_core then
        return nil, "read_core_stats_error", tostring(score)
    end
    trace_log(frame_counter, "read_core_stats",
        string.format("score=%d replay=%d lasers=%d wave=%d", score, replay_level, num_lasers, wave_number))

    local ok_enemy, enemy_or_err = pcall(read_enemy_state, mem)
    if not ok_enemy then
        return nil, "read_enemy_state_error", tostring(enemy_or_err)
    end
    local enemy_state = enemy_or_err
    trace_log(frame_counter, "read_enemy_state", "bytes=" .. tostring(#enemy_state.raw))

    local ok_player_pos, player_x16, player_y16 = pcall(read_player_position, mem)
    if not ok_player_pos then
        return nil, "read_player_position_error", tostring(player_x16)
    end
    trace_log(frame_counter, "read_player_position", string.format("x16=%d y16=%d", player_x16 or 0, player_y16 or 0))

    local ok_entities, obs_or_err = pcall(extract_world_features, mem, player_x16, player_y16, enemy_state)
    if not ok_entities then
        return nil, "extract_world_features_error", tostring(obs_or_err)
    end
    local obs = obs_or_err
    trace_log(frame_counter, "extract_world_features",
        string.format("objects=%d compact=%d", tonumber(obs.object_count or 0), #(obs.state_features or {})))

    return {
        player_alive = player_alive,
        score = score,
        replay_level = replay_level,
        num_lasers = num_lasers,
        wave_number = wave_number,
        enemy_state = enemy_state,
        player_x16 = obs.player_center_x16 or player_x16,
        player_y16 = obs.player_center_y16 or player_y16,
        obs = obs,
        num_humans = obs.num_humans or 0,
    }
end

function compute_frame_rewards(frame)
    local player_alive = frame.player_alive
    local player_x16 = frame.player_x16
    local done = (previous_player_alive == 1 and player_alive == 0)
    local score_delta = frame.score - previous_score
    if score_delta < 0 then
        score_delta = 0
    end
    local obj_reward = score_delta
    if done then
        obj_reward = obj_reward - DEATH_PENALTY_POINTS
    end

    local spacing_score = enemy_spacing_score(frame.obs.nearest_enemy_dist)
    local rescue_score = human_proximity_score(frame.obs.nearest_human_dist)
    local aim_score = compute_aim_reward(prev_fire_cmd, prev_aim_px16, prev_aim_py16, prev_aim_objects)
    local evade_score = compute_evasion_reward(prev_move_cmd, prev_aim_px16, prev_aim_py16,
        prev_nearest_enemy_x16, prev_nearest_enemy_y16, prev_nearest_enemy_dist)
    local survival_bonus = (player_alive == 1) and SUBJ_SURVIVAL_BONUS or 0.0

    local wall_penalty = 0.0
    if player_alive == 1 and player_x16 then
        local px = norm_pos_x(player_x16)
        local py = norm_pos_y(frame.player_y16)
        if px < WALL_MARGIN_NORM_X or px > (1.0 - WALL_MARGIN_NORM_X) then
            wall_penalty = wall_penalty + SUBJ_WALL_PENALTY
        end
        if py < WALL_MARGIN_NORM_Y or py > (1.0 - WALL_MARGIN_NORM_Y) then
            wall_penalty = wall_penalty + SUBJ_WALL_PENALTY
        end
    end

    local subj_reward = survival_bonus
        + (spacing_score * SUBJ_ENEMY_WEIGHT)
        + (rescue_score * SUBJ_HUMAN_WEIGHT)
        + (aim_score * SUBJ_AIM_WEIGHT)
        + (evade_score * SUBJ_EVADE_WEIGHT)
        - wall_penalty
    if done then
        subj_reward = subj_reward - SUBJ_DEATH_PENALTY
    end

    trace_log(frame_counter, "reward_calc",
        string.format("score_delta=%d done=%s obj_reward=%.1f subj_reward=%.2f enemy_dist=%s human_dist=%s",
            score_delta, tostring(done), obj_reward, subj_reward,
            frame.obs.nearest_enemy_dist and string.format("%.4f", frame.obs.nearest_enemy_dist) or "nil",
            frame.obs.nearest_human_dist and string.format("%.4f", frame.obs.nearest_human_dist) or "nil"))

    return {
        done = done,
        obj_reward = obj_reward,
        subj_reward = subj_reward,
    }
end

function capture_preview_payload()
    if preview_stream_enabled and pending_preview_blob then
        local out = {
            w = pending_preview_w,
            h = pending_preview_h,
            fmt = pending_preview_fmt or PREVIEW_FORMAT_RGB565,
            blob = pending_preview_blob,
        }
        clear_pending_preview()
        return out
    end
    return {w = 0, h = 0, fmt = PREVIEW_FORMAT_RGB565, blob = nil}
end

function frame_callback()
    if not mem or not controls then
        return true
    end

    trace_log(frame_counter, "frame_begin", "callback entered")

    local frame, err_phase, err_detail = read_frame_observation()
    if not frame then
        trace_log(frame_counter, err_phase or "frame_read_error", tostring(err_detail), true)
        return true
    end

    local ok_patch, patched_wave_or_err = pcall(
        robotron_apply_start_wave_patch,
        mem,
        frame.player_alive,
        frame.score,
        frame.wave_number
    )
    if not ok_patch then
        trace_log(frame_counter, "curriculum_patch_error", tostring(patched_wave_or_err), true)
    elseif patched_wave_or_err then
        frame.wave_number = patched_wave_or_err
        trace_log(frame_counter, "curriculum_patch", "applied start wave " .. tostring(patched_wave_or_err))
    end

    local player_alive = frame.player_alive
    local rewards = compute_frame_rewards(frame)
    if player_alive == 0 then
        dead_frame_counter = dead_frame_counter + 1
        prev_object_sample_x = {}
        prev_object_sample_y = {}
        _reset_compact_slot_assignments()
    else
        dead_frame_counter = 0
    end
    local start_cmd, coin_cmd = determine_meta_commands(dead_frame_counter, player_alive)

    local now = os.time()
    local save_signal = 0
    if shutdown_requested or (now - last_save_time) >= SAVE_INTERVAL_S then
        save_signal = 1
        last_save_time = now
    end

    local preview = capture_preview_payload()

    local ok_payload, payload_or_err = pcall(
        serialize_frame,
        frame.player_alive, frame.score, frame.replay_level, frame.num_lasers, frame.wave_number,
        frame.player_x16, frame.player_y16, frame.obs.state_features,
        rewards.done, rewards.subj_reward, rewards.obj_reward, save_signal, start_cmd,
        preview.w, preview.h, preview.fmt, preview.blob
    )
    if not ok_payload then
        trace_log(frame_counter, "serialize_frame_error", tostring(payload_or_err), true)
        return true
    end
    local payload = payload_or_err
    local payload_count = #(frame.obs.state_features or {})
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
    previous_score = frame.score
    previous_wave_number = frame.wave_number
    prev_num_humans = frame.num_humans or 0

    -- Stash position for next-frame velocity computation
    if player_alive == 1 then
        prev_player_x16 = frame.player_x16
        prev_player_y16 = frame.player_y16
    else
        prev_player_x16 = nil
        prev_player_y16 = nil
    end

    -- Stash aim-reward data for use on the NEXT frame (reward for this frame's action).
    -- Use effective_fire (the direction actually sent to the game after hold) for correct
    -- aim-reward attribution.
    prev_fire_cmd = effective_fire
    prev_move_cmd = move_cmd
    prev_aim_objects = hud_objects   -- reuse the same reference (set in extract_world_features)
    prev_aim_px16 = frame.player_x16
    prev_aim_py16 = frame.player_y16
    prev_nearest_enemy_x16 = frame.obs.nearest_enemy_x16
    prev_nearest_enemy_y16 = frame.obs.nearest_enemy_y16
    prev_nearest_enemy_dist = frame.obs.nearest_enemy_dist

    trace_log(frame_counter, "frame_end", "done=" .. tostring(rewards.done))
    frame_counter = frame_counter + 1

    return true
end

function on_mame_exit()
    shutdown_requested = true
    trace_log(frame_counter, "session_stop", "machine stop notifier", true)
    close_socket()
    print("Robotron AI Lua script shutting down.")
end

function register_frame_callback(cb)
    if emu.add_machine_frame_notifier then
        return emu.add_machine_frame_notifier(cb)
    end
    if emu.register_frame then
        return emu.register_frame(cb)
    end
    error("No supported frame callback API found (expected emu.add_machine_frame_notifier or emu.register_frame)")
end

function register_frame_done_callback(cb)
    if emu.register_frame_done then
        return emu.register_frame_done(cb)
    end
    -- Older/minimal MAME builds may not expose a frame-done hook.
    return nil
end

function register_stop_callback(cb)
    if emu.add_machine_stop_notifier then
        return emu.add_machine_stop_notifier(cb)
    end
    if emu.register_stop then
        return emu.register_stop(cb)
    end
    return nil
end

math.randomseed(os.time())

if not initialize_mame_interface() then
    return
end

if not apply_rrchris_patch() then
    print("Robotron AI Lua script aborted due to RRCHRIS patch failure.")
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

global_callback_ref = register_frame_callback(frame_callback)

-- All preview-capable instances register the frame_done callback.  The server
-- decides per-frame which client should actually capture/stream preview data by
-- toggling the preview flag in the action source byte.
if PREVIEW_CLIENT_FLAG == 1 then
    print(string.format(
        "[HUD] Preview capture configured: %dfps max=%dx%d rle=%s",
        PREVIEW_FPS, PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT, tostring(PREVIEW_TRY_RLE)
    ))
    if register_frame_done_callback(frame_done_callback) ~= nil or emu.register_frame_done ~= nil then
        print("[HUD] Registered frame_done callback for debug overlay + preview capture")
    else
        print("[HUD] Frame-done callback unavailable in this MAME build; preview capture disabled")
    end
end

register_stop_callback(on_mame_exit)

print("Robotron AI Lua script initialized.")
