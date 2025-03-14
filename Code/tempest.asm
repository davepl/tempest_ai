;-----------------------------------------------------------------------------
;
; Tempest (c) 1981 Atari Corporation - Original Game author Dave Theurer
;
; Code analysis by Davepl
;
; This code analysis is purely archaeological; commercial use prohibited.
;
; You can and should assume large portions of the comments and naming of
; variables within is incorrect.  Absolutely no warranty nor suitability
; for any purpose is implied.  Running this or derivitive code in a real
; Tempest machine may cause hardware damage, personal injury, or death.
; If I have seen further, it's becasue I stood on the shoulders of giants
; such as Josh McCormick and der Mouse (whoever that is!) that got the
; ball rolling with their original hacking.  Much credit is due to them!
; 
;
; Orientation:   Vertical
; Type:          Color Vector
; CRT:           Color, 19-inc Wells Gardner 19K6100 (most machines)
; CPU:           6502A @ 1.5MHz
; ROM:           24K
; RAM:           6K
;
;-----------------------------------------------------------------------------
; As of 1-22-2017 this code assembles with the HXA assembler from:
;   http://home.earthlink.net/~hxa/
; As of 6-20-020 that project is now located on github and works ok:
;   https://github.com/AntonTreuenfels/HXA_Cross_Assembler
;
; It will produce the original factory ROM image as a single OBJ that can be
; split into 2K sections to burn onto EPROMs. Any reduction in code
; size is automatically padded to fill the full 20K so the 6502 vectors wind
; up in the right place (but only because the p1 ROM is mirrored up at FFFx).
; Please ensure you have the legal rights to do so before making any ROM chips.
;
; 136002-133.d1  // 0x9000
; 136002-134.f1  // 0xA000
; 136002-235.j1  // 0xB000
; 136002-136.lm1 // 0xC000
; 136002-237.p1  // 0xD000 and loaded again at 0xF000
;
; There is a vector ROM (136002-138.np3) that is loaded at 0x3000 which is
; not part of this project.
;-----------------------------------------------------------------------------
;
;    HEX        R/W   D7 D6 D5 D4 D3 D2 D2 D0  function
;    0000-07FF  R/W   D  D  D  D  D  D  D  D   program ram (2K)
;    0800-080F   W                D  D  D  D   Colour ram
;
;    0C00        R                         D   Right coin sw
;    0C00        R                      D      Center coin sw
;    0C00        R                   D         Left coin sw
;    0C00        R                D            Slam sw
;    0C00        R             D               Self test sw
;    0C00        R          D                  Diagnostic step sw
;    0C00        R       D                     Halt
;    0C00        R    D                        3kHz ??
;    0D00        R    D  D  D  D  D  D  D  D   option switches
;    0E00        R    D  D  D  D  D  D  D  D   option switches
;
;    2000-2FFF  R/W   D  D  D  D  D  D  D  D   Vector Ram (4K)
;    3000-3FFF   R    D  D  D  D  D  D  D  D   Vector Rom (4K)
;
;    4000        W                         D   Right coin counter
;    4000        W                      D      left  coin counter
;    4000        W                D            Video invert - x
;    4000        W             D               Video invert - y
;    4800        W                             Vector generator GO
;    5000        W                             WD clear
;    5800        W                             Vect gen reset
;
;    6000-603F   W    D  D  D  D  D  D  D  D   EAROM write
;    6040        W    D  D  D  D  D  D  D  D   EAROM control
;    6040        R    D                        Mathbox status
;    6050        R    D  D  D  D  D  D  D  D   EAROM read
;
;    6060        R    D  D  D  D  D  D  D  D   Mathbox read
;    6070        R    D  D  D  D  D  D  D  D   Mathbox read
;    6080-609F   W    D  D  D  D  D  D  D  D   Mathbox start
;
;    60C0-60CF  R/W   D  D  D  D  D  D  D  D   Custom audio chip 1
;    60D0-60DF  R/W   D  D  D  D  D  D  D  D   Custom audio chip 2
;
;    60E0        R                         D   one player start LED
;    60E0        R                      D      two player start LED
;    60E0        R                   D         FLIP
;
;    9000-DFFF  R     D  D  D  D  D  D  D  D   Program ROM (20K)
;
;-----------------------------------------------------------------------------
;
;    GAME OPTIONS:
;    (8-position switch at L12 on Analog Vector-Generator PCB)
;
;    1   2   3   4   5   6   7   8   Meaning
;    -------------------------------------------------------------------------
;    Off Off                         2 lives per game
;    On  On                          3 lives per game
;    On  Off                         4 lives per game
;    Off On                          5 lives per game
;            On  On  Off             Bonus life every 10000 pts
;            On  On  On              Bonus life every 20000 pts
;            On  Off On              Bonus life every 30000 pts
;            On  Off Off             Bonus life every 40000 pts
;            Off On  On              Bonus life every 50000 pts
;            Off On  Off             Bonus life every 60000 pts
;            Off Off On              Bonus life every 70000 pts
;            Off Off Off             No bonus lives
;                        On  On      English
;                        On  Off     French
;                        Off On      German
;                        Off Off     Spanish
;                                On  1-credit minimum
;                                Off 2-credit minimum
;
;
;    GAME OPTIONS:
;    (4-position switch at D/E2 on Math Box PCB)
;
;    1   2   3   4                   Meaning
;    -------------------------------------------------------------------------
;        Off                         Minimum rating range: 1, 3, 5, 7, 9
;        On                          Minimum rating range tied to high score
;            Off Off                 Medium difficulty (see notes)
;            Off On                  Easy difficulty (see notes)
;            On  Off                 Hard difficulty (see notes)
;            On  On                  Medium difficulty (see notes)
;
;
;    PRICING OPTIONS:
;    (8-position switch at N13 on Analog Vector-Generator PCB)
;
;    1   2   3   4   5   6   7   8   Meaning
;    -------------------------------------------------------------------------
;    On  On  On                      No bonus coins
;    On  On  Off                     For every 2 coins, game adds 1 more coin
;    On  Off On                      For every 4 coins, game adds 1 more coin
;    On  Off Off                     For every 4 coins, game adds 2 more coins
;    Off On  On                      For every 5 coins, game adds 1 more coin
;    Off On  Off                     For every 3 coins, game adds 1 more coin
;    On  Off                 Off On  Demonstration Mode (see notes)
;    Off Off                 Off On  Demonstration-Freeze Mode (see notes)
;                On                  Left coin mech * 1
;                Off                 Left coin mech * 2
;                    On  On          Right coin mech * 1
;                    On  Off         Right coin mech * 4
;                    Off On          Right coin mech * 5
;                    Off Off         Right coin mech * 6
;                            Off On  Free Play
;                            Off Off 1 coin 2 plays
;                            On  On  1 coin 1 play
;                            On  Off 2 coins 1 play
;
;-----------------------------------------------------------------------------


.cpu 6502
.OBJFILE <tempest.obj>
.LISTON
.LISTFILE <tempest.lst>

;-----------------------------------------------------------------------------
; BUILD CUSTOMIZATIONS
;-----------------------------------------------------------------------------
;
; The following build definitions modify how the ROMs are built.  Setting all
; to 0 will generate a completely original ROM set.
;
; All of thge build flags are only set if not previously defined.  This allows
; us to have a file called 'original.asm' that defines them all to 0 before
; including this main source file and in turn produces an original ROM set.
;
;-----------------------------------------------------------------------------

.ifndef DAVEPL_MSG
DAVEPL_MSG = 0                          ; If set to 1, proof-of-life customization
.endif

.ifndef REMOVE_SELFTEST
REMOVE_SELFTEST  = 0                    ; Set to 1 to remove self-test code in
.endif                                  ;   order to make more space           

.ifndef OPTIMIZE
OPTIMIZE         = 0                   ; Prune dead code and tables
.endif

.ifndef ALT_START_TABLE
ALT_START_TABLE  = 0                    ; Set to 1 and you get starting levels on
.endif                                  ;   each color and then many in the black
                                        ;   and green rather than all down low
                                        ;   Requires room, so remove selftest, etc
.ifndef REMOVE_LANGUAGES
REMOVE_LANGUAGES = 0                    ; Remove non-English text to save space
.endif

.ifndef ADD_LEVEL
ADD_LEVEL        = 0                    ; Adds purple levels above green.  Works OK
.endif                                  ;   but level display is alawys still 2 digit

;-----------------------------------------------------------------------------

MAX_CREDITS         = 40                ; Credit limit for game

.if !ADD_LEVEL
  HIGHEST_LEVEL     = 98                ; After that, levels are randomized
  LAST_SHAPE_LEVEL  = 95                ; Last level part of a group of 16
  MAX_LEVEL         = 99                ; Max possible level
  LAST_GREEN        = 99                ; Last green level
.else
  HIGHEST_LEVEL     = 112
  LAST_SHAPE_LEVEL  = 112
  MAX_LEVEL         = 113
  LAST_GREEN        = 96    
.endif

TOP_OF_TUNNEL       = $10               ; Constant used to indicate top of tunnel
END_OF_TUNNEL       = $F0               ; Constant used to indicate made it to end
MAX_ZAP_USES        = 2                 ; Number of superzap uses per level
MAX_ENEMY_SHOTS     = 4                 ; Most enemy shots onscreen at once
MAX_ACTIVE_ENEMIES  = 7                 ; Most enemies onscreen at once
MAX_PLAYER_SHOTS    = 8                 ; Max onscreen shots from player

MAX_TOTAL_SHOTS     = MAX_PLAYER_SHOTS + MAX_ENEMY_SHOTS

ENEMY_TYPE_MASK     = %00000111         ; Bottom 3 bits indicate which enemy
ENEMY_TYPE_FLIPPER  = 0
ENEMY_TYPE_PULSAR   = 1
ENEMY_TYPE_TANKER   = 2
ENEMY_TYPE_SPIKER   = 3
ENEMY_TYPE_FUSEBALL = 4

;-----------------------------------------------------------------------------
; Game States
;-----------------------------------------------------------------------------
; When in Self-Test:
;
; $02 = first selftest screen (config bits, spinner line)
; $04 = second selftest screen (diagonal grid, line of characters)
; $06 = third selftest screen (crosshair, shrinking rectangle)
; $08 = fourth selftest screen (coloured lines)
; $0a = fifth selftest screen, grid with colour depending on spinner
; $0c = sixth selftest screen, blank rectangle
; 
; Not in selftest as follows:
;-----------------------------------------------------------------------------

GS_GameStartup          = $00       ; $00 = entered briefly at game startup
GS_LevelStartup         = $02       ; $02 = entered briefly at the beginning of first level of a game
GS_Playing              = $04       ; $04 = playing (including attract mode)
GS_Death                = $06       ; $06 = entered briefly on player death (game-ending or not)
GS_LevelBegin           = $08       ; $08 = set briefly at the beginning of each level?
GS_Delay                = $0A       ; $0a = eg AVOID SPIKES: $1e->$04, $0a->gamestate, $20->state_after_delay, $80->$0123
                                    ; $0c = unused? (jump table holds $0000)
GS_ZoomOffEnd           = $0E       ; $0e = entered briefly when zooming off the end of a level
GS_Unknown10            = $10       
GS_EnterInitials        = $12       ; $12 = entering initials
GS_Unknown14            = $14
GS_LevelSelect          = $16       ; $16 = starting level selection screen
GS_ZoomOntoNew          = $18       ; $18 = zooming new level in
GS_Unknown1A            = $1A       ; $1a = unknown
GS_Unknown1C            = $1C       ; $1c = unknown
GS_DelayThenPlay        = $1E       ; $1e = Brief pause, then switch to Playing mode
GS_ZoomingDown          = $20       ; $20 = descending down tube at level end
GS_ServiceDisplay       = $22       ; $22 = non-selftest service mode display
GS_HighScoreExplosion   = $24       ; $24 = high-score explosion


; Color Definitions.  A "sparkle" bit can be used in the high nibble (and which is subsequently
; parsed out to the bytes at x0808-0x80F), by adding it to the base color.  I do not yet know
; what adding $40 to the MSB does... maybe less sparkly?  It sounds like X flip according to
; MAME, but that doesn't make sense in Tempest.  My guess (davepl) is it's a Space Duel thing that
; the emulator supports but that the real hardware didn't yet.
;
; Approximate color formula is: 
;
; bit3 = (~data >> 3) & 1
; bit2 = (~data >> 2) & 1
; bit1 = (~data >> 1) & 1
; bit0 = (~data >> 0) & 1

; r = bit1 * 0xf3 + bit0 * 0x0c;
; b = bit2 * 0xf3;
; g = bit3 * 0xf3;
        
;                                        Data    GBRr       (Where R = major red, r = minor red)
White                   = $00           ;0000    1111
Cyan                    = $03           ;0011    1100
Yellow                  = $04           ;0010    1101
Green                   = $07           ;0111    1000
Purple                  = $08           ;1000    0111
Blue                    = $0B           ;1011    0100
Red                     = $0C           ;1100    0011
Black                   = $0F           ;1111    0000
Sparkle                 = $C0
Unk04                   = $40

;**** Zero-Page Definitions

.org $0000

; I could have made these ZP all EQU symbols but I felt it was more likely the original
; source would have been symbol, using data definition and not hardcoded addresses.

gamestate               .ds 1           ; $00
unknown_state           .ds 1           ; $01
state_after_delay       .ds 1           ; $02
timectr                 .ds 1           ; $03       Regularly incrementing time based counter
countdown_timer         .ds 1           ; $04
game_mode               .ds 1           ; $05       High bit seems to indicate attract mode
credits                 .ds 1           ; $06
                        .ds 1           ; $07
zap_fire_shadow         .ds 1           ; $08
coinage_shadow          .ds 1           ; $09       Shadow of optsw1.  Note value is XORed with $02 before storing
optsw2_shadow           .ds 1           ; $0A       Soft shadow of optsw2

                        .ds 11

coin_string             .ds 1           ; $16
uncredited              .ds 1           ; $17       Soft shadow of optsw2

                        .ds 20

zpPtrL                  .ds 1           ; $2A
zpPtrM                  .ds 1           ; $2B
zpPtrH                  .ds 1           ; $2C

                        .ds 14

curplayer               .ds 1           ; $3D
twoplayer               .ds 1           ; $3E
                        .ds 1           ; $3F
p1_score_l              .ds 1           ; $40
p1_score_m              .ds 1           ; $41
p1_score_h              .ds 1           ; $42
p2_score_l              .ds 1           ; $43
p2_score_m              .ds 1           ; $44
p2_score_h              .ds 1           ; $45
p1_level                .ds 1           ; $46
p2_level                .ds 1           ; $47
p1_lives                .ds 1           ; $48
p2_lives                .ds 1           ; $49
                        .ds 1           ; $4A
                        .ds 1           ; $4B

                        ; zap_fire_tmp1
                        ; 
                        ; Bit positions:
                        ; $08 = zap
                        ; $10 = fire
                        ; $20 = start 1
                        ; $40 = start 2
                        ; $80 = unknown (cleared at various points)

zap_fire_tmp1           .ds 1           ; $4c
zap_fire_debounce       .ds 1           ; $4d
zap_fire_new            .ds 1           ; $4e
zap_fire_tmp2           .ds 1           ; $4f
                        .ds 9           ; $50-58
fscale                  .ds 1           ; $59       copied from lev_fscale[]/lev_fscale2[]
                        .ds 6
y3d                     .ds 1           ; $60       copied from lev_y3d[]
                        .ds 17          ; $61-71
curscale                .ds 1           ; $72
draw_z                  .ds 1           ; $73
vidptr_l                .ds 1           ; $74
vidptr_h                .ds 1           ; $75
                        .ds 35
rgr_pt_inx              .ds 1           ; $99
                        .ds 4
curcolor                .ds 1
curlevel                .ds 1           ; $9F
                        .ds 6
EnemyShotCount          .ds 1           ; $A6
                        .ds 5
strtbl                  .ds 1           ; $AC
                        .ds 5
pulsar_fliprate         .ds 1           ; $B2       Number of movement ticks between pulsar flips


; Ratio by which flipper flips at top-of-tube are accelerated relative to
; flipper flips in the tube.  If this number is 2, they are twice as fast,
; etc.  See $a141 for more.

flip_top_accel          .ds 1           ; $B3
                        .ds 1
copyr_cksum             .ds 1           ; $B5
copyr_vid_loc           .ds 1           ; $B6
                        .ds 6   

; Pointer to RAM version of current EAROM block.  See $de{64,9a,9c,d8,ee}.

earom_memptr            .ds 1           ; $BD
                        .ds 66

;--- Page One ---------------------------------------------------------------------------
.org $0100
;----------------------------------------------------------------------------------------
                        .ds 2
p1_startchoice          .ds 1           ; $0102
p2_startchoice          .ds 1           ; $0103
zoomspd_lsb             .ds 1           ; $0104
zoomspd_msb             .ds 1           ; $0105
                        .ds 1
along_lsb               .ds 1           ; $0107
NumEnemiesInTube        .ds 1           ; $0108
NumEnemiesOnTop         .ds 1           ; $0109
pcode_run               .ds 1           ; $010A
pcode_pc                .ds 1           ; $010B

open_level          = $0111     ; $00 if current level closed, $ff if open after remap[] has been applied
curtube             = $0112
flagbits            = $0117
enm_shotspd_msb     = $0118

; Shot holdoff time.  After firing, an enemy cannot fire until at least
; this many ticks have passed.

shot_holdoff        = $0119
MaxEnemyShots       = $011A     ; Maximum number of enemy shots less one
copyr_vid_cksum1    = $011B
MaxActiveEnemies    = $011C

enm_shotspd_lsb     = $0120

zap_running         = $0125

; Minimum enemy counts for each enement type (calculated)

min_enemy_by_type   = $0129
min_flippers        = $0129     ; flipper.  1-4:1, 5-99:0
min_pulsars         = $012A     ; pulsar.   1-16:0, 17-32:3, 33-99:1
min_tankers         = $012B     ; tanker.   1-2:0, 3:1, 4:0, 5-99:1
min_spikers         = $012C     ; spiker.   1-3:0, 4:1, 5-16:2, 17-19:0, 20-99:1
min_fuseballs       = $012D     ; fuseball. 1 except 1-10, 17-19, 26, where 0

; Indexed by enemy type.  Max number of a given type in the tube 

max_enemy_by_type   = $012E
max_flippers        = $012E
max_pulsars         = $012F
max_tankers         = $0130
max_spikers         = $0131
max_fuseballs       = $0132

PlayerShotCount     = $0135

; Maximum less current for each enemy type (ie: number remaining that can
; be created).  Computed and used during creation of new enemies.

avl_enemy_by_type   = $013D
avl_flippers        = $013D
avl_pulsars         = $013E
avl_tankers         = $013F
avl_spikers         = $0140
avl_fuseballs       = $0141

n_enemy_by_type     = $0142
n_flippers          = $0142
n_pulsars           = $0143
n_tankers           = $0144
n_spikers           = $0145
n_fuseballs         = $0146

; Initially calculated at $9475.  $9b5a uses this to update $8148
; $9b8c/$9b9f negate this value when $0148 hits 0x0F and 0xC1, ascending and 
; descending respectively.

pulse_beat          = $0147

; Set to 0xFF during init.  Once per cycle, after all enement movement has
; been completed, the computation $0147 + $0147 is done, and if the high bit of
; $0148 goes from 0 to 1 as a result, $cd06 is called (see also $9b56)

pulsing             = $0148
tanker_load         = $0149

hit_tol_by_enm_type = $0151
hit_tol_flipper     = $0151
hit_tol_pulsar      = $0152
hit_tol_tanker      = $0153
hit_tol_spiker      = $0154
hit_tol_fuseball    = $0155

bonus_life_each     = $0156
lethal_distance     = $0157
init_lives          = $0158

; Two flags that control fuseball motion are kept here (in the 0x40 and 0x80 bits)
; See $9607_6b for setting, $9f2c/$9f4a/$9f6e for use

fuse_move_flg       = $0159

wave_spikeht        = $015A
wave_enemies        = $015B

;Movement type for flippers - See $9aa2 for use, 9607_5b for computation.

flipper_move        = $015D

; Fuseball movement probabilities
;
;  The chance of fuzzballs doing something in certain regions of the tube;
;  see 9607_6f for computation, $9f69 for use.

fuse_move_prb       = $015F
spd_flipper_lsb     = $0160
spd_pulsar_lsb      = $0161
spd_tanker_lsb      = $0162
spd_spiker_lsb      = $0163
spd_fuseball_lsb    = $0164
spd_flipper_msb     = $0165
spd_pulsar_msb      = $0166
spd_tanker_msb      = $0167
spd_spiker_msb      = $0168
spd_fuseball_msb    = $0169

; Difficulty Rating Bits
;
;  00000011 - 00 Medium, 01 Easy, 02 Hard, 03 Medium
;  00000100 - Rating (0 = Start 1-9, 1 = Tied to High Score)
;  00001000 - Unknown, comes from 0x20 bit of spinner/cabinet type

diff_bits           = $016A

copyr_disp_cksum1   = $016C
pulsar_fire         = $016D

earom_clr           = $01C6

; EAROM Stuff
;
;  00000011 Bits
;  --------
;  00000001  Top three sets of initials needs initializing
;  00000010  Top three scores need initializing

hs_initflag         = $01C9

earom_op            = $01CA
earom_blkoff        = $01CB
earom_ptr           = $01CC
earom_blkend        = $01CD

earom_cksum         = $01CF 

player_seg          = $0200

; Player State
;
; Usually, this is is (player_seg+1)&0x0f.  But it's set to other values
; sometimes; the $80 bit seems to indicate "death in progress".
; $80 - player grabbed by flipper/pulsar
; $81 - player hit by enemy shot
; $81 - player spiked while going down tube
; I suspect that $80 means "death in progress" and, when so, $01 means 
; don't display player anyway".

player_state        = $0201
player_along        = $0202             

; Segment numbers for pending enemies. Set to random values $0-$f; see $9250.

                    .org $0203
pending_seg         .ds  64             ; 16 x 4 = 64 Bytes ($40)


; Video display information, one per pending enemy.
;
; $00 here means "no pending enemy in this slot".
; This byte breaks down as BBLLLLLL, and is used as a vscale value with
; b=BB+1 and l=LLLLLL.  This is initialized to the pending_seg value in
; the low nibble and the offset's low nibble in the high nibble, except
; that if that would give $00, $0f is used instead.  See $9250.


pending_vid         .ds  64              ; 16 x 4 = 64 Bytes ($40)


; Active Enemy Type Information 
;
; $07 bits hold enemy type (0-4).
; $18 bits apparently mean something; see $b5bf.
; $40 bit set -> enemy_seg value increasing; clr -> decreasing (see $9eab)
; $80 bit set -> between segments (eg, flipper flipping)

enemy_type_info     = $0283             ; 7 Bytes - Indexed by enemy number.

; Active Enemy State Information 
;
; $80 bit set -> moving away from player, clear -> towards
; $40 means the enemy can shoot
; $03 bits determine what happens when enemy gets below $20:
; $00 = no special action (as if $20 weren't special)
; $01 = split into two flippers
; $02 = split into two pulsars
; $03 = split into two fuzzballs

active_enemy_info   = $028A         ; 7 Bytes - Indexed by active enemy number

enm_move_pc         = $0291         ; 7 Bytes - Indexed by active enemy number - PCode program counter for each enemy
enm_pc_storage      = $0298         ; 7 Bytes - Indexed by active enemy number - PCode storage 'register' for each enemy
enemy_along_lsb     = $029F

shot_delay          = $02A6         ; 7 Bytes - Indexed by active enemy number

.org $02AD
PlayerShotSegments  .ds MAX_PLAYER_SHOTS
EnemyShotSegments   .ds MAX_ENEMY_SHOTS
enemy_seg           .ds MAX_ACTIVE_ENEMIES


; More Enemy Info
;
; Flipper flipping: $80 plus current angle
; Flipper not flipping: segment number last flipped from
; Fuseballs store $81 or $87 here, depending on the $40 bit of enemy_type_info,x

                        .org $02cc
more_enemy_info         .ds  MAX_ACTIVE_ENEMIES         ; 7 Bytes - Indexed by enemy number
                      
                        .org $02D3  
PlayerShotPositions:    .ds MAX_PLAYER_SHOTS            ; 8 Bytes - Indexed by player shot number
EnemyShotPositions:     .ds MAX_ENEMY_SHOTS             ; 4 Bytes - Indexed by enemy shot number
enemy_along:            .ds MAX_ACTIVE_ENEMIES          ; 7 Bytes - Indexed by enemy number 
enm_shot_lsb:           .ds MAX_ENEMY_SHOTS             ; 4 Bytes - Indexed by enemy shot number            

                        .org $03AA
PlayerState:
zap_uses                .ds 1
enemies_pending         .ds 1
lane_spike_height       .ds 16
PlayerStateLen          = * - PlayerState
OtherPlayerState        .ds 1

tube_x              = $03CE

tube_y              = $03DE

tube_angle          = $03EE

on_time_l           = $0406
on_time_m           = $0407
on_time_h           = $0408
play_time_l         = $0409
play_time_m         = $040A
play_time_h         = $040B
games_1p_l          = $040C
games_1p_m          = $040D
games_1p_h          = $040E
games_2p_l          = $040F
games_2p_m          = $0410
games_2p_h          = $0411
secs_avg_l          = $0412
secs_avg_m          = $0413
secs_avg_h          = $0414
dblbuf_flg          = $0415
        
mid_x               = $0435

mid_y               = $0445

copyr_vid_cksum2    = $0455

hs_whichletter      = $0602
         
hs_timer            = $0605

hs_initials_8       = $0606
hs_initials_7       = $0609
hs_initials_6       = $060C
hs_initials_5       = $060F
hs_initials_4       = $0612
hs_initials_3       = $0615
hs_initials_2       = $0618
hs_initials_1       = $061B
         
hs_scores           = $0700
hs_score_8          = $0706
hs_score_7          = $0709
hs_score_6          = $070C
hs_score_5          = $070F
hs_score_4          = $0712
hs_score_3          = $0715
hs_score_2          = $0718
hs_score_1          = $071B
endofhiscores       = $071D

life_settings       = $071E
diff_settings       = $071F

col_ram             = $0800     ; Color RAM table, 8 bytes
col_ram1            = $0801
col_ram2            = $0802
col_ram3            = $0803
col_ram4            = $0804
col_ram5            = $0805
col_ram6            = $0806
col_ram7            = $0807

col_ram_upr         = $0808     ; Color RAM modifiers, such as sparkle bit, 8 bytes
col_ram_upr1        = $0809
col_ram_upr2        = $080A
col_ram_upr3        = $080B
col_ram_upr4        = $080C
col_ram_upr5        = $080D
col_ram_upr6        = $080E
col_ram_upr7        = $080F

cabsw:              = $0C00
optsw1              = $0D00
optsw2              = $0E00

vecram              = $2000
video_data          = $2F60
char_jsrtbl         = $31E4
ltr_jsrtbl          = $31FA

test_magic_tbl      = $3F16
diff_str_tbl        = $3F1E

vid_coins           = $4000
vg_go               = $4800
watchdog            = $5000
vg_reset            = $5800
earom_write         = $6000

eactl_mbst          = $6040
earom_rd            = $6050
mb_rd_l             = $6060
mb_rd_h             = $6070
mb_w_00             = $6080
mb_w_01             = $6081
mb_w_02             = $6082
mb_w_03             = $6083
mb_w_04             = $6084
mb_w_05             = $6085
mb_w_06             = $6086
mb_w_07             = $6087
mb_w_08             = $6088
mb_w_09             = $6089
mb_w_0a             = $608a
mb_w_0b             = $608b
mb_w_0c             = $608c
mb_w_0d             = $608d
mb_w_0e             = $608e
mb_w_0f             = $608f
mb_w_10             = $6090
mb_w_11             = $6091
mb_w_12             = $6092
mb_w_13             = $6093
mb_w_14             = $6094

pokey1              = $60C0
spinner_cabtyp      = $60C8
pokey1_rand         = $60CA

pokey2              = $60D0
zap_fire_starts     = $60D8
pokey2_rand         = $60DA
leds_flip           = $60E0

mb_w_15             = $6095
mb_w_16             = $6096

.org $9000
.if !OPTIMIZE
                        .byte   $02
                        .byte   $bb
                        .byte   $5a
                        .byte   $30
                        .byte   $50
                        .byte   $ee
                        .byte   $3d
                        .byte   $a8
                        .byte   $4d
.endif

InitLevel:              jsr     LoadLevelParams
                        jsr     InitEnemiesAndSpikes
                        jsr     ResetGameElements
                        jsr     InitSuperzapper
                        lda     #$fa
                        sta     $5b
                        lda     #$00
                        sta     $0106
                        sta     $5f
                        lda     #$00
                        sta     unknown_state
                        rts

InitializeGame:         jsr     InitPlayerPosition
                        jsr     LoadLevelParams
ResetGameElements:      jsr     ClearAllShots
                        jsr     ClearAllEnemies
                        jsr     InitEnemyLocations
                        jsr     ClearEnemyDeaths
                        jsr     ZeroSpinner
                        jsr     InitVector

                        lda     #$ff
                        sta     $0124
                        sta     pulsing
                        lda     #$00
                        sta     $0123
                        rts

State_ZoomOntoNew:      lda     #$10
                        sta     player_along
                        lda     #$00
                        sta     $29
                        sta     $2b
                        lda     $0121
                        sta     $2a
                        bpl     +
                        dec     $2b
+                       ldx     #$01
-                       lda     $2a
                        asl     a
                        ror     $2a
                        ror     $29
                        dex
                        bpl     -
                        lda     $29
                        clc
                        adc     $0122
                        sta     $0122
                        lda     $2a
                        adc     $68
                        sta     $68
                        lda     $2b
                        adc     $69
                        sta     $69
                        lda     $5f
                        clc
                        adc     #$18
                        sta     $5f
                        lda     $5b
                        adc     #$00
                        sta     $5b
                        cmp     #$fc
                        bcc     +
                        lda     #$01
                        sta     $0115
+                       lda     $5f
                        sec
                        sbc     $5d
                        lda     $5b
                        beq     +
                        sbc     #$ff
+                       bne     loc90bc
                        lda     $5d
                        sta     $5f
                        lda     #$ff
                        sta     $5b
                        lda     #$04
                        bit     game_mode
                        bmi     +
                        lda     #GS_LevelBegin
+                       sta     gamestate
                        ldx     curplayer
                        lda     #$00
                        sta     p1_startchoice,x
loc90bc:                lda     #$ff
                        sta     $0114
                        jmp     move_player

; Level Selection Code

PlayerLevelSelect:      lda     $0126                   ; See what level the player last completed and then run down
                        ldx     #NumberOfStartEntries   ;   the starting level table list by walking down it until 
loc90c9:                dex                             ;   we find one smaller or equal to the last completed
                        cmp     startlevtbl,x
                            
.if !ALT_START_TABLE
                        bcc loc90c9
.else
    .if !OPTIMIZE
                        nop                             ; Two 'nops' so that we take up the same amount of space
                        nop                             ;   if OPTIMIZE is off we emit nops so that code all lines up as original
    .endif
.endif
                        ldy     #$04
                        lda     diff_bits
                        and     #$04                    ; self-rating bit
                        beq     loc90ea                 ; branch if 1,3,5,7,9
                        lda     hs_score_1+2            ; top two digits of top score
                        cmp     #$30                    ; If highscore over 30,000, add level 11
                        bcc     loc90e0
                        iny
loc90e0:                cmp     #$50                    ; If highscore over 50,000, add level 13
                        bcc     loc90e5
                        iny
loc90e5:                cmp     #$70                    ; If highscore over 70,000, add level 15
                        bcc     loc90ea
                        iny
loc90ea:                lda     coinage_shadow
                        and     #$43                    ; coinage + 1 bit of bonus coins
                        cmp     #$40                    ; free play + 1/4,2/4,demo
                        bne     loc90f4
                        ldy     #$1b
loc90f4:                sty     $29
                        cpx     $29
                        bcs     loc90fc
                        ldx     $29
loc90fc:                stx     $0127
                        lda     game_mode
                        bpl     state_1c
                        lda     #$00
                        sta     $0126

state_1c:               ldx     $3f
                        stx     curplayer
                        beq     loc9111
                        jsr     SwapPlayerStates

loc9111:                lda     #$04
                        sta     $7c
                        lda     #$ff
                        sta     $5b
                        lda     #$00
                        sta     player_seg
                        sta     $51
                        sta     $7b
                        sta     hs_timer
                        ldx     game_mode
                        bpl     loc9144
                        lda     #$14
                        sta     hs_timer
                        lda     #$ff
                        sta     open_level
                        lda     #GS_LevelSelect
                        sta     gamestate
                        lda     #$08
                        sta     $01
                        lda     #$00
                        sta     curlevel
                        jsr     SetLevelColors
.if !ALT_START_TABLE
                        lda     #$10                    ; BCD value of seconds to wait at Level Selection
.else
                        lda     #$60                    ; BCD value of seconds to wait at Level Selection
.endif

loc9144:                sta     countdown_timer
                        jsr     ZeroSpinner

State_LevelSelect:      dec     hs_timer
                        bpl     loc9169
                        sed
                        lda     countdown_timer
                        sec
                        sbc     #$01
                        sta     countdown_timer
                        cld
                        bpl     loc915d
                        lda     #$10                    ; fire
                        sta     zap_fire_new
loc915d:                cmp     #$03
                        bne     loc9164
                        jsr     locccfe
loc9164:                lda     #$14
                        sta     hs_timer
loc9169:                jsr     locb0ab
                        lda     #$18
                        ldy     countdown_timer
                        cpy     #$08
                        bcs     loc9176
                        lda     #$78
loc9176:                and     zap_fire_new
                        beq     loc91ae
                        lda     #$00
                        sta     zap_fire_new
                        lda     player_seg

; New-level entry, used by attract mode(?) and new game start

                        tay
                        ldx     curplayer
                        sta     p1_startchoice,x
                        lda     startlevtbl,y
                        bit     game_mode
                        bmi     loc9197
                        ldy     #$01
                        sty     p1_lives
                        lda     pokey1_rand
                        and     #$07                            ; Attract mode levels are always under this limit
loc9197:                sta     p1_level,x
                        sta     curlevel
                        jsr     SetLevelColors
                        jsr     LoadLevelParams
                        jsr     InitEnemiesAndSpikes
                        jsr     InitSuperzapper
                        lda     #GS_LevelStartup
                        sta     gamestate
                        jsr     ZeroSpinner
loc91ae:                lda     zap_fire_new
                        and     #$07
                        sta     zap_fire_new
                        rts

; Loads the start bonus (level choice in A) into 29/2a/2b

ld_startbonus:          asl     a
                        tax
                        lda     #$00
                        sta     $29
                        lda     start_bonus,x
                        sta     $2a
                        lda     start_bonus+1,x
                        sta     $2b
                        rts

; Start bonuses, in BCD, low 00 not stored.

start_bonus:            
.if !ALT_START_TABLE
                        .word   $0000       ; Level  1 - Bonus:       0
                        .word   $0060       ; Level  3 - Bonus:   6,000
                        .word   $0160       ; Level  5 - Bonus:  16,000
                        .word   $0320       ; Level  7 - Bonus:  32,000
                        .word   $0540       ; Level  9 - Bonus:  54,000
                        .word   $0740       ; Level 11 - Bonus:  74,000
                        .word   $0940       ; Level 13 - Bonus:  94,000
                        .word   $1140       ; Level 15 - Bonus: 114,000
                        .word   $1340       ; Level 17 - Bonus: 134,000
                        .word   $1520       ; Level 20 - Bonus: 152,000
                        .word   $1700       ; Level 22 - Bonus: 170,000
                        .word   $1880       ; Level 24 - Bonus: 188,000
                        .word   $2080       ; Level 26 - Bonus: 208,000
                        .word   $2260       ; Level 28 - Bonus: 226,000
                        .word   $2480       ; Level 31 - Bonus: 248,000
                        .word   $2660       ; Level 33 - Bonus: 266,000
                        .word   $3000       ; Level 36 - Bonus: 300,000
                        .word   $3400       ; Level 40 - Bonus: 340,000
                        .word   $3820       ; Level 44 - Bonus: 382,000
                        .word   $4150       ; Level 47 - Bonus: 415,000
                        .word   $4390       ; Level 49 - Bonus: 439,000
                        .word   $4720       ; Level 52 - Bonus: 472,000
                        .word   $5310       ; Level 56 - Bonus: 531,000
                        .word   $5810       ; Level 60 - Bonus: 581,000
                        .word   $6240       ; Level 63 - Bonus: 624,000
                        .word   $6560       ; Level 65 - Bonus: 656,000
                        .word   $7660       ; Level 73 - Bonus: 766,000
                        .word   $8980       ; Level 81 - Bonus: 898,000
.else
                        .word   $0000       ; Level  1 - Bonus:       0
                        .word   $0060       ; Level  3 - Bonus:   6,000
                        .word   $0160       ; Level  5 - Bonus:  16,000
                        .word   $0320       ; Level  7 - Bonus:  32,000
                        .word   $0540       ; Level  9 - Bonus:  54,000
                        .word   $0740       ; Level 11 - Bonus:  74,000
                        .word   $0940       ; Level 13 - Bonus:  94,000
                        .word   $1140       ; Level 15 - Bonus: 114,000
                        .word   $1340       ; Level 17 - Bonus: 134,000
                        .word   $1520       ; Level 20 - Bonus: 152,000
                        .word   $1700       ; Level 22 - Bonus: 170,000
                        .word   $1880       ; Level 24 - Bonus: 188,000
                        .word   $2080       ; Level 26 - Bonus: 208,000
                        .word   $2260       ; Level 28 - Bonus: 226,000
                        .word   $2480       ; Level 31 - Bonus: 248,000
                        .word   $2660       ; Level 33 - Bonus: 266,000
                        .word   $3000       ; Level 36 - Bonus: 300,000
                        .word   $3400       ; Level 40 - Bonus: 340,000
                        .word   $3820       ; Level 44 - Bonus: 382,000
                        .word   $4150       ; Level 47 - Bonus: 415,000
                        .word   $4390       ; Level 49 - Bonus: 439,000
                        .word   $4720       ; Level 52 - Bonus: 472,000
                        .word   $5310       ; Level 56 - Bonus: 531,000
                        .word   $5810       ; Level 60 - Bonus: 581,000
                        .word   $6240       ; Level 63 - Bonus: 624,000
                        .word   $6560       ; Level 65 - Bonus: 656,000
                        .word   $6750       ; Level 67 - Bonus: 675,000
                        .word   $7000       ; Level 69 - Bonus: 700,000
                        .word   $7200       ; Level 71 - Bonus: 720,000
                        .word   $7660       ; Level 73 - Bonus: 766,000
                        .word   $7900       ; Level 75 - Bonus: 790,000
                        .word   $8200       ; Level 77 - Bonus: 820,000
                        .word   $8980       ; Level 80 - Bonus: 880,000
                        .word   $8980       ; Level 81 - Bonus: 898,000
                        .word   $8980       ; Level 82 - Bonus: 898,000
                        .word   $8980       ; Level 83 - Bonus: 898,000
                        .word   $8980       ; Level 84 - Bonus: 898,000
                        .word   $8980       ; Level 85 - Bonus: 898,000
                        .word   $8980       ; Level 86 - Bonus: 898,000
                        .word   $8980       ; Level 87 - Bonus: 898,000
                        .word   $8980       ; Level 88 - Bonus: 898,000
                        .word   $8980       ; Level 89 - Bonus: 898,000
                        .word   $8980       ; Level 90 - Bonus: 898,000
                        .word   $8980       ; Level 91 - Bonus: 898,000
                        .word   $8980       ; Level 92 - Bonus: 898,000
                        .word   $8980       ; Level 93 - Bonus: 898,000
                        .word   $8980       ; Level 94 - Bonus: 898,000
                        .word   $8980       ; Level 95 - Bonus: 898,000
                        .word   $8980       ; Level 96 - Bonus: 898,000
    .if ADD_LEVEL   
                        .word   $9950       ; Level 97 = Bonus: 995,000
                        .word   $9950       ; Level 107 = Bonus: 995,000
                        .word   $9950       ; Level 112 = Bonus: 995,000
                        .word   $9950       ; Level 107 = Bonus: 995,000
                        .word   $9950       ; Level 112 = Bonus: 995,000

    .endif

.endif

end_start_bonus:

; Start level numbers?  (ie, is this table maybe mapping from index
; of level-select entry chosen to actual level number?)

startlevtbl:            
.if !ALT_START_TABLE
                        .byte    1 - 1      ; Level  1 - Bonus:       0
                        .byte    3 - 1      ; Level  3 - Bonus:   6,000
                        .byte    5 - 1      ; Level  5 - Bonus:  16,000
                        .byte    7 - 1      ; Level  7 - Bonus:  32,000
                        .byte    9 - 1      ; Level  9 - Bonus:  54,000
                        .byte   11 - 1      ; Level 11 - Bonus:  74,000
                        .byte   13 - 1      ; Level 13 - Bonus:  94,000
                        .byte   15 - 1      ; Level 15 - Bonus: 114,000
                        .byte   17 - 1      ; Level 17 - Bonus: 134,000
                        .byte   20 - 1      ; Level 20 - Bonus: 152,000
                        .byte   22 - 1      ; Level 22 - Bonus: 170,000
                        .byte   24 - 1      ; Level 24 - Bonus: 188,000
                        .byte   26 - 1      ; Level 26 - Bonus: 208,000
                        .byte   28 - 1      ; Level 28 - Bonus: 226,000
                        .byte   31 - 1      ; Level 31 - Bonus: 248,000
                        .byte   33 - 1      ; Level 33 - Bonus: 266,000
                        .byte   36 - 1      ; Level 36 - Bonus: 300,000
                        .byte   40 - 1      ; Level 40 - Bonus: 340,000
                        .byte   44 - 1      ; Level 44 - Bonus: 382,000
                        .byte   47 - 1      ; Level 47 - Bonus: 415,000
                        .byte   49 - 1      ; Level 49 - Bonus: 439,000
                        .byte   52 - 1      ; Level 52 - Bonus: 472,000
                        .byte   56 - 1      ; Level 56 - Bonus: 531,000
                        .byte   60 - 1      ; Level 60 - Bonus: 581,000
                        .byte   63 - 1      ; Level 63 - Bonus: 624,000
                        .byte   65 - 1      ; Level 65 - Bonus: 656,000
                        .byte   73 - 1      ; Level 73 - Bonus: 766,000
                        .byte   81 - 1      ; Level 81 - Bonus: 898,000
.else
                        .byte    1 - 1      ; Level  1 - Bonus:       0
                        .byte    3 - 1      ; Level  3 - Bonus:   6,000
                        .byte    5 - 1      ; Level  5 - Bonus:  16,000
                        .byte    7 - 1      ; Level  7 - Bonus:  32,000
                        .byte    9 - 1      ; Level  9 - Bonus:  54,000
                        .byte   11 - 1      ; Level 11 - Bonus:  74,000
                        .byte   13 - 1      ; Level 13 - Bonus:  94,000
                        .byte   15 - 1      ; Level 15 - Bonus: 114,000
                        .byte   17 - 1      ; Level 17 - Bonus: 134,000
                        .byte   20 - 1      ; Level 20 - Bonus: 152,000
                        .byte   22 - 1      ; Level 22 - Bonus: 170,000
                        .byte   24 - 1      ; Level 24 - Bonus: 188,000
                        .byte   26 - 1      ; Level 26 - Bonus: 208,000
                        .byte   28 - 1      ; Level 28 - Bonus: 226,000
                        .byte   31 - 1      ; Level 31 - Bonus: 248,000
                        .byte   33 - 1      ; Level 33 - Bonus: 266,000
                        .byte   36 - 1      ; Level 36 - Bonus: 300,000
                        .byte   40 - 1      ; Level 40 - Bonus: 340,000
                        .byte   44 - 1      ; Level 44 - Bonus: 382,000
                        .byte   47 - 1      ; Level 47 - Bonus: 415,000
                        .byte   49 - 1      ; Level 49 - Bonus: 439,000
                        .byte   52 - 1      ; Level 52 - Bonus: 472,000
                        .byte   56 - 1      ; Level 56 - Bonus: 531,000
                        .byte   60 - 1      ; Level 60 - Bonus: 581,000
                        .byte   63 - 1      ; Level 63 - Bonus: 624,000
                        .byte   65 - 1      ; Level 65 - Bonus: 656,000
                        .byte   67 - 1      ; Level 67 - Bonus: 675,000
                        .byte   69 - 1      ; Level 69 - Bonus: 700,000
                        .byte   71 - 1      ; Level 71 - Bonus: 720,000
                        .byte   73 - 1      ; Level 73 - Bonus: 766,000
                        .byte   75 - 1      ; Level 75 - Bonus: 790,000
                        .byte   77 - 1      ; Level 77 - Bonus: 820,000
                        .byte   80 - 1      ; Level 80 - Bonus: 880,000
                        .byte   81 - 1      ; Level 81 - Bonus: 898,000
                        .byte   82 - 1      ; Level 82 - Bonus: 898,000
                        .byte   83 - 1      ; Level 83 - Bonus: 898,000
                        .byte   84 - 1      ; Level 84 - Bonus: 898,000
                        .byte   85 - 1      ; Level 85 - Bonus: 898,000
                        .byte   86 - 1      ; Level 86 - Bonus: 898,000
                        .byte   87 - 1      ; Level 87 - Bonus: 898,000
                        .byte   88 - 1      ; Level 88 - Bonus: 898,000
                        .byte   89 - 1      ; Level 89 - Bonus: 898,000
                        .byte   90 - 1      ; Level 90 - Bonus: 898,000
                        .byte   91 - 1      ; Level 91 - Bonus: 898,000
                        .byte   92 - 1      ; Level 92 - Bonus: 898,000
                        .byte   93 - 1      ; Level 93 - Bonus: 898,000
                        .byte   94 - 1      ; Level 94 - Bonus: 898,000
                        .byte   95 - 1      ; Level 95 - Bonus: 898,000
                        .byte   96 - 1      ; Level 96 - Bonus: 898,000
    .if ADD_LEVEL
                        .byte   97 - 1      ; Level 97
                        .byte   98 - 1      ; Level 98
                        .byte   99 - 1      ; Level 99

                        .byte  107 - 1
                        .byte  112 - 1
    .endif

.endif
end_startlevtbl:
                        .byte   $ff

; Make sure that the number of entries in the starting level table matches
;   the number of entries in the bonus table

NumberOfStartEntries    .equ    ((end_startlevtbl - startlevtbl))
.assert (NumberOfStartEntries) == ((end_start_bonus - start_bonus) / 2)


InitPlayerPosition:     lda     #$0e
                        sta     player_seg
                        lda     #END_OF_TUNNEL
                        sta     $51
                        lda     #$00
                        sta     $0106
                        lda     #$0f
                        sta     player_state
                        lda     #TOP_OF_TUNNEL
                        sta     player_along
                        rts

InitEnemiesAndSpikes:   lda     wave_enemies
                        sta     enemies_pending
                        lda     wave_spikeht
                        ldx     #$0f
-                       sta     lane_spike_height,x
                        dex
                        bpl     -
                        rts

; Initialize the pending_seg and pending_vid tables.

InitEnemyLocations:     lda     #$00
                        ldx     #$3f
-                       sta     pending_vid,x
                        dex
                        bpl     -

                        ldx     enemies_pending
                        dex
-                       lda     pokey1_rand
                        and     #$0f
                        sta     pending_seg,x
                        txa
                        asl     a
                        asl     a
                        asl     a
                        asl     a
                        ora     pending_seg,x
                        bne     +
                        lda     #$0f
+                       sta     pending_vid,x
                        dex
                        bpl     -
                        rts

; Clear all enemies - used by init code

ClearAllEnemies:        ldx     #$06
                        lda     #$00
-                       sta     enemy_along,x
                        dex
                        bpl     -

                        sta     NumEnemiesInTube
                        sta     NumEnemiesOnTop

                        sta     n_spikers           ; Zero out the enemy counts for each type
                        sta     n_flippers
                        sta     n_tankers
                        sta     n_pulsars
                        sta     n_fuseballs
                        rts

ClearAllShots:          lda     #$00
                        ldx     #MAX_TOTAL_SHOTS-1
-                       sta     PlayerShotPositions,x
                        dex
                        bpl     -
                        sta     PlayerShotCount
                        sta     EnemyShotCount
                        rts

; Fills 0x00 into 8.bytes at 030a, also in 0116.
; Another disassembly says this aborts enemy death sequences in progress.

ClearEnemyDeaths:       ldx     #MAX_ACTIVE_ENEMIES
                        lda     #$00
-                       sta     $030a,x
                        dex
                        bpl     -
                        sta     $0116
                        rts

ZeroSpinner:            lda     #$00
                        sta     $50
                        rts

; Swap players state - copies the state infomration back and forth between players

SwapPlayerStates:       ldx     #PlayerStateLen - 1
-                       lda     PlayerState,x
                        ldy     OtherPlayerState,x
                        sta     OtherPlayerState,x
                        tya
                        sta     PlayerState,x
                        dex
                        bpl     -
                        rts

LoadLevelParams:        lda     curlevel
                        cmp     #HIGHEST_LEVEL
                        bcc     +
                        lda     pokey2_rand             
                        and     #$1f
                        ora     #$40
+                       sta     $2b                     ; Effective Level Number
                        inc     $2b

; $2b now holds the effective level number
; Loop for X from $6f down through $03 (loop ends at $931d-$9326)

                        ldx     #ParameterTableLength
                        stx     $37

                        ; Table format:  SOMENUMBER FIRSTLEVEL LASTLEVEL

ParameterLoop:          ldx     $37
                        lda     ParametersTable,x
                        sta     $3c                     ; ($3b) points to the table of values
                        lda     ParametersTable-1,x
                        sta     $3b
                        lda     ParametersTable-2,x
                        sta     $2d
                        lda     ParametersTable-3,x     ; ($2c) points to the param that's going to get set
                        sta     $2c
                        
                        lda     #$01
                        sta     $38

                        ldy     #$00
loc92f6:                lda     ($2c),y
                        sta     $015e
                        beq     loc9319
                        lda     $2b                 ; Effective level number
                        iny
                        cmp     ($2c),y
                        iny
                        bcc     level_outside_range ; branch if lvl # is too low
                        cmp     ($2c),y
                        bne     loc930a             ; branch to $9313 if lvl # >= ($2c),y
                        clc
loc930a:                bcs     level_outside_range
                        iny
                        jsr     loc9677
                        jmp     loc9319

level_outside_range:    jsr     loc9683
                        clc
                        bcc     loc92f6

loc9319:                ldy     #$00
                        sta     ($3b),y
                        
                        lda     $37
                        sec
                        sbc     #$04
                        sta     $37
                        cmp     #$ff
                        bne     ParameterLoop

                        lda     diff_bits
                        and     #$03                ; difficulty
                        cmp     #$01                ; easy
                        bne     loc934d

                        ; Easy Difficulty - Does the following modifications to gameplay:
                        ;                 - Decreases the number of enemy shots iscreen
                        ;                 - Decreases flipper speed
                        ;                 - Decreases the amount of acceleration for flippers on top rails

                        dec     MaxEnemyShots
                        lda     spd_flipper_lsb
                        eor     #$ff
                        lsr     a
                        lsr     a
                        lsr     a
                        adc     spd_flipper_lsb
                        sta     spd_flipper_lsb
                        lda     curlevel
                        cmp     #$11
                        bcs     loc934a
                        dec     flip_top_accel
loc934a:                clv
                        bvc     SetEnemyParams
loc934d:                cmp     #$02                    ; hard
                        bne     SetEnemyParams

                        ; Hard Difficulty - Does the following modifications to gameplay:
                        ;                 - One additional enemy shot, but still limited to 3, which you'd have on higher levels anyway
                        ;                 - Flipper speed is increased 
                        ;                 - The number of enemies per wave is increased by 1/8th
                        ;                 - Pulsars always fire (as they eventually do on higher levels)
                        ;                 - No increase in active enemy count

HardDifficulty:         inc     MaxEnemyShots           ; Hard difficulty adds an enemy shot, but limited to 3
                        lda     MaxEnemyShots
                        cmp     #$03
                        bcc     loc9360
                        lda     #$03
                        sta     MaxEnemyShots

loc9360:                lda     spd_flipper_lsb
                        lsr     a
                        lsr     a
                        lsr     a
                        ora     #$e0
                        adc     spd_flipper_lsb
                        sta     spd_flipper_lsb

                        lda     wave_enemies
                        lsr     a
                        lsr     a
                        lsr     a
                        adc     wave_enemies
                        sta     wave_enemies
                        
                        lda     pulsar_fire             ; Pulsars shoot in hard difficulty (but they do on higher levels anyway)
                        ora     #$40
                        sta     pulsar_fire

SetEnemyParams:         lda     spd_spiker_lsb
                        jsr     crack_speed
                        sta     spd_spiker_lsb
                        sty     spd_spiker_msb
                        stx     hit_tol_spiker          ; hit tolerance for spikers
                        lda     enm_shotspd_lsb
                        jsr     crack_speed
                        sta     enm_shotspd_lsb
                        sty     enm_shotspd_msb
                        stx     $a7
                        lda     spd_flipper_lsb
                        jsr     crack_speed
                        sta     spd_flipper_lsb
                        sta     spd_tanker_lsb
                        sty     spd_tanker_msb
                        sty     spd_flipper_msb
                        stx     hit_tol_flipper         ; hit tolerance for flippers
                        stx     hit_tol_tanker          ; hit tolerance for tankers
                        stx     hit_tol_pulsar          ; hit tolerance for pulsars
                        lda     spd_flipper_lsb
                        asl     a
                        sta     spd_fuseball_lsb
                        lda     spd_flipper_msb
                        rol     a
                        sta     spd_fuseball_msb
                        lda     #$06
                        sta     hit_tol_fuseball        ; hit tolerance for fuseballs
                        lda     #$a0
                        sta     spd_pulsar_lsb
                        lda     #$fe
                        sta     spd_pulsar_msb
                        lda     #$01
                        sta     tanker_load+1
                        sta     tanker_load
                        rts

; Convert a speed value such as found in the $9607 tables to MSB and LSB
; values, and a shot hit tolerance.  Return the MSB value in A, the LSB
; value in Y, and the hit tolerance in X.

crack_speed:            ldy     #$ff
                        sty     $29
                        asl     a
                        rol     $29
                        asl     a
                        rol     $29
                        asl     a
                        rol     $29
                        ldy     $29
                        pha
                        tya
                        eor     #$ff
                        clc
                        adc     #$0d
                        lsr     a
                        tax
                        pla
                        rts

; Computation of various per-level parameters.  See the table at $9607 and
; the code at $92d6 for more.  Each.ds is commented with the address or
; symbol for the.byte it computes.

shot_holdoff_table:                                                 
                        .byte 8,1,14h,50h,0FDh
                        .byte 2,15h,40h,14h
                        .byte 2,41h,MAX_LEVEL,0Ah

MaxEnemyShots_table:                                                 
                        .byte 4,1,9,1,1,1,2,3,2,2,3,3                   ; From Level 1 to 9, valus as specified
                        .byte 2,10,64,2                                 ; From level 10 to 64, 2
                        .byte 2,65,LAST_GREEN,3                         ; From level 65 to 99, 3
.if ADD_LEVEL
                        .byte 2,97,112,3
.endif

spd_flipper_lsb_table:                                                 
                        .byte 8,1,8,212,251                             
                        .byte 4,9,16,175,172,172,172,168,164,160,160
                        .byte 8,17,25,175,253
                        .byte 8,26,32,157,253
                        .byte 8,33,39,148,253
                        .byte 8,40,48,146,255
                        .byte 8,49,64,136,255
                        .byte 12,65,LAST_GREEN,96,65
.if ADD_LEVEL
                        .byte 12,97,112,112,97
.endif

enm_shotspd_lsb_table:                                                 
                        .byte 10,1,MAX_LEVEL,192

spd_spiker_lsb_table:                                                 
                        .byte 10,1,20,0
                        .byte 10,21,32,208
                        .byte 10,33,48,216
                        .byte 10,49,LAST_GREEN,208
.if ADD_LEVEL
                        .byte 10,97,112,208
.endif

lethal_distance_table:                                              ;lethal_distance - Distance pulsar can be down tube but still lethal
                        .byte 2,1,32,160                            
                        .byte 2,33,64,160
                        .byte 2,65,MAX_LEVEL,192

pulse_beat_table:                                                 
                        .byte 2,1,48,4
                        .byte 2,49,64,6
                        .byte 2,65,LAST_GREEN,8
.if ADD_LEVEL
                        .byte 2,97,112,12
.endif

ParametersTable_43:                                                 ;tanker_load+2 
                        .byte 2,1,32,1
                        .byte 2,33,40,3
                        .byte 2,41,MAX_LEVEL,2

ParametersTable_47:                                                 ;tanker_load+3 
                        .byte 2,1,48,1
                        .byte 2,49,MAX_LEVEL,3

min_spikers_table:                                                 
                        .byte 4,1,4,0,0,0,1                         
                        .byte 2,5,16,2                              
                        .byte 2,17,19,0                             
                        .byte 2,20,32,1
                        .byte 2,35,39,1
                        .byte 2,44,MAX_LEVEL,1
                        .byte 0
max_spikers_table:                                                 
                        .byte 4,1,6,0,0,0,2,3,4
                        .byte 2,7,10,4
                        .byte 2,11,16,3
                        .byte 2,20,25,2
                        .byte 4,26,32,1,2,2,2,1,1,2
                        .byte 2,53,39,1
                        .byte 2,43,MAX_LEVEL,1
                        .byte 0

min_flippers_table:                                                 
                        .byte 2,1,4,1
                        .byte 2,5,MAX_LEVEL,0
                        .byte 0

max_flippers_table:                                                 
                        .byte 2,1,4,4
                        .byte 2,5,16,5
                        .byte 2,17,19,3
                        .byte 2,20,25,4
                        .byte 2,26,MAX_LEVEL,5
                        .byte 0

min_tankers_table:                                                 
                        .byte 4,1,4,0,0,1,0
                        .byte 2,5,16,1
                        .byte 2,17,32,1
                        .byte 2,33,39,1
                        .byte 2,40,MAX_LEVEL,1
                        .byte 0

max_tankers_table:                                                 
                        .byte 4,1,5,0,0,1,0,1
                        .byte 2,6,16,2
                        .byte 2,17,26,1
                        .byte 2,27,32,1
                        .byte 2,33,44,2
                        .byte 2,45,MAX_LEVEL,3
                        .byte 0

min_pulsars_table:                                                 
                        .byte 2,17,32,2
                        .byte 2,33,MAX_LEVEL,1
                        .byte 0

max_pulsars_table:                                            
                        .byte 4,17,32,5,3,2,2,2,2,2,2,2,2,2,2,2,3,4,2
                        .byte 2,33,MAX_LEVEL,3
                        .byte 0

min_fuseballs_table:                                                 
                        .byte 2,11,16,1                 ; From Level 11 to 16, 1 Fuseball min
                        .byte 2,22,25,1                 ; From Level 22 to 25, 1 Fuseball min 
                        .byte 2,27,MAX_LEVEL,1          ; From Level 27 to 98, 1 Fuseball min
                        .byte 0

max_fuseballs_table:                                                 
                        .byte 2,11,16,1                 ; From Level 11 to 16, 1 Fuseball max
                        .byte 2,22,25,1                 ; From Level 22 to 25, 1 Fuseball max
                        .byte 2,27,32,1                 ; From Level 27 to 32, 1 Fuseball max
                        .byte 2,33,39,4                 ; From Level 33 to 39, 4 Fuseballs max
                        .byte 2,40,MAX_LEVEL,3          ; From Level 40 to 99, 3 Fuseballs max
                        .byte 0

pulsar_fliprate_table:                                                 
                        .byte 4,17,18,40,20
                        .byte 12,19,32,20,40
                        .byte 8,33,39,20,255
                        .byte 12,40,MAX_LEVEL,20,10
                        .byte 0

fuse_move_flg_table:                                                 
                        .byte 12,17,32,0,64
                        .byte 12,33,48,64,192
                        .byte 2,49,MAX_LEVEL,192
                        .byte 0

fuse_move_prb_table:                                                 
                        .byte 2,1,16,220
                        .byte 2,17,39,192
                        .byte 8,40,64,192,1
                        .byte 2,65,LAST_GREEN,230     
.if ADD_LEVEL
                        .byte 2,97,112,245
.endif                                     

MaxActiveEnemies_table:                                                 
                        .byte 2,1,MAX_LEVEL,6                       ; 7 active enemies max on any level

wave_spikeht_table:                                                 
                        .byte 6,1,MAX_LEVEL,0,0,0,224,216,212,208,200,192,184,176,168,160,160,160,168
                        .byte 0A0h
                        .byte 9Ch
                        .byte 9Ah
                        .byte 98h

wave_enemies_table:                                                 
                        .byte 4,1,16,10,12,15,17,20,22,20,24,27,29,27,24,26,28,30,27
                        .byte 8,17,26,20,1      
                        .byte 2,27,39,27
                        .byte 8,40,48,29,1
                        .byte 8,49,64,31,1
                        .byte 8,65,80,35,1
                        .byte 8,81,MAX_LEVEL,43,1

flip_top_accel_table:                                                 
                        .byte 2,1,20,2
                        .byte 2,21,32,2
                        .byte 2,33,MAX_LEVEL,3

pulsar_fire_table:                                                 
                        .byte 2, 60, MAX_LEVEL, $40
                        .byte 0

flipper_move_table:                                                 
                        .byte 6,1,MAX_LEVEL,7,11,25,36,83,11,36,25,83,135,36,25,83,7,135,36


; See the code beginning $92d6.
        
ParametersTable:        .word    pulsar_fire_table
                        .word    pulsar_fire

                        .word    flip_top_accel_table
                        .word    flip_top_accel

                        .word    shot_holdoff_table
                        .word    shot_holdoff

                        .word    MaxEnemyShots_table
                        .word    MaxEnemyShots

                        .word    min_flippers_table
                        .word    min_flippers

                        .word    max_flippers_table
                        .word    max_flippers

                        .word    min_pulsars_table
                        .word    min_pulsars

                        .word    max_pulsars_table
                        .word    max_pulsars

                        .word    min_tankers_table
                        .word    min_tankers

                        .word    max_tankers_table
                        .word    max_tankers

                        .word    min_spikers_table
                        .word    min_spikers

                        .word    max_spikers_table
                        .word    max_spikers

                        .word    min_fuseballs_table
                        .word    min_fuseballs

                        .word    max_fuseballs_table
                        .word    max_fuseballs

                        .word    lethal_distance_table
                        .word    lethal_distance

                        .word    pulse_beat_table
                        .word    pulse_beat

                        .word    ParametersTable_43
                        .word    tanker_load+2

                        .word    ParametersTable_47
                        .word    tanker_load+3

                        .word    MaxActiveEnemies_table
                        .word    MaxActiveEnemies

                        .word    wave_enemies_table
                        .word    wave_enemies

                        .word    wave_spikeht_table
                        .word    wave_spikeht

                        .word    pulsar_fliprate_table
                        .word    pulsar_fliprate

                        .word    flipper_move_table
                        .word    flipper_move

                        .word    spd_spiker_lsb_table
                        .word    spd_spiker_lsb

                        .word    enm_shotspd_lsb_table
                        .word    enm_shotspd_lsb

                        .word    spd_flipper_lsb_table
                        .word    spd_flipper_lsb

                        .word    fuse_move_flg_table
                        .word    fuse_move_flg

                        .word    fuse_move_prb_table
                        .word    fuse_move_prb

ParameterTableLength    .equ * - ParametersTable - 1

loc9677:                ldx     $015e
                        lda     loc9690,x
                        pha
                        lda     loc9690-1,x
                        pha
                        rts

loc9683:                ldx     $015e
                        lda     loc969e,x
                        pha
                        lda     loc969e-1,x
                        pha
                        rts

; Jump table used by code at 9677.
; ltmin = first level-test.byte
; ltmax = second level-test.byte
; b[] =.bytes following level test.bytes
; thus, we have:                opcode ltmin ltmax b[0] b[1] b[2] etc...
; (loc) = contents of memory location loc
; lev = current level number
; lwb = (((lev-1)&15)+1 - level # within its block of 16 levels

                        .byte   00
loc9690:                .byte   00                  ; ( 0) not used - tested for at $92fb
                        .word   loc968f_02-1        ; ( 2) A = b[0]
                        .word   loc968f_04-1        ; ( 4) A = b[lev-ltmin]
                        .word   loc968f_06-1        ; ( 6) A = b[lwb-ltmin]
                        .word   loc968f_08-1        ; ( 8) A = b[0] + ((lev-ltmin) * b[1])
                        .word   loc968f_0a-1        ; (10) A = b[0] + ($0160)
                        .word   loc968f_0c-1        ; (12) A = b[(lev-ltmin)&1]

; Jump table used by code at 9683.

                        .byte   00
loc969e:                .byte   00                  ; not used - tested for at $92fb

                        .word   loc969d_02_0a-1     ; Y += 2
                        .word   loc969d_04_06-1     ; Y += ltmax - ltmin + 2
                        .word   loc969d_04_06-1     ; Y += ltmax - ltmin + 2
                        .word   loc969d_08_0c-1     ; Y += 3
                        .word   loc969d_02_0a-1     ; Y += 2
                        .word   loc969d_08_0c-1     ; Y += 3

loc968f_06:             lda     $2b
                        sec
                        sbc     #$01
                        and     #$0f
                        clc
                        adc     #$01
                        bpl     loc96b9

loc968f_04:             lda     $2b
loc96b9:                sty     $29
                        dey
                        dey
                        sec
                        sbc     ($2c),y
                        clc
                        adc     $29
                        tay                 

loc968f_02:             lda     ($2c),y
                        rts

loc969d_08_0c:          iny
loc969d_02_0a:          iny
                        iny
                        rts
loc969d_04_06:          lda     ($2c),y
                        dey
                        sec
                        sbc     ($2c),y
                        sta     $29
                        tya
                        sec
                        adc     $29
                        tay
                        iny
                        iny
                        rts
loc968f_0a:             lda     ($2c),y
                        clc
                        adc     spd_flipper_lsb
                        rts
loc968f_08:             jsr     loc96f4
                        tax
                        lda     ($2c),y
                        iny
                        cpx     #$00
                        beq     loc96f3
loc96ed:                clc
                        adc     ($2c),y
                        dex
                        bne     loc96ed
loc96f3:                rts

; Set A to current level number minus base level number

loc96f4:                lda     $2b
                        sty     $29
                        dey
                        dey
                        sec
                        sbc     ($2c),y
                        iny
                        iny
                        rts
loc968f_0c:             jsr     loc96f4
                        and     #$01
                        beq     loc9708
                        iny
loc9708:                lda     ($2c),y
                        rts

State_Playing:          jsr     move_player
                        jsr     CheckPlayerFire
                        jsr     check_zap
                        jsr     create_enemies
                        jsr     move_enemies
                        jsr     move_shots
                        jsr     enm_shoot
                        jsr     CheckAllPlayerShots
                        jsr     loca416
                        jmp     loca504

State_ZoomingDown:      lda     $0123
                        and     #$7f                            ; ~$80
                        sta     $0123
                        jsr     move_player
                        jsr     loc97f8
                        jsr     loca416
                        jsr     CheckPlayerFire
                        jsr     move_shots
                        lda     player_state
                        bpl     loc9748
                        jsr     loca504
loc9748:                rts

; handles player movement

move_player:            lda     player_state ; Check player's state (movement active or not)
                        bpl     stillalive   ; If player_state >= 0 (active), proceed to movement
                        rts                 ; If player_state < 0 (inactive, e.g. dying), exit (no movement)

stillalive:             ldx     #$00        ; Clear index X (set to 0, used as a default value)
                        lda     game_mode   ; Load current game mode (bit7 = attract mode flag)
                        bmi     notattract   ; If attract mode (demo, bit7 set), skip direct input handling
                        jsr     demoplay     ; Else (normal play), read and process player spinner input
                        clv                 ; Clear overflow flag (prepare for consistent branching)
                        bvc     applymove   ; Always branch (V=0) to apply movement (skip manual input section)

notattract:             lda     $50         ; Load spinner movement delta (low byte at $50, signed)
                        bpl     loc9768     ; If delta >= 0, skip negative clamping
                        cmp     #-31        ; Compare delta with -31 (limit for max left turn speed)
                        bcs     loc9765     ; If delta >= -31, within allowed range (no clamp needed)
                        lda     #-31        ; If delta < -31, clamp delta to -31 (limit turn speed leftward)
loc9765:                clv                 ; Clear overflow flag (for predictable branching)
                        bvc     loc976e     ; Always branch (continue after handling negative clamp)
loc9768:                cmp     #31         ; Compare delta with +31 (limit for max right turn speed)
                        bcc     loc976e     ; If delta <= 30, within allowed range (no clamp needed)
                        lda     #31         ; If delta > 30, clamp delta to +31 (limit turn speed rightward)
loc976e:                stx     $50         ; Reset spinner delta ($50) to 0 now that we've captured it

applymove:              sta     $2b         ; Save (clamped) delta into $2B (temp movement accumulator)
                        eor     #$ff        ; A = A XOR $FF (invert delta bits to prepare for subtraction)
                        sec                 ; Set carry (for two's complement addition of negative delta)
                        adc     $51         ; Add previous position ($51) to -delta (effectively $51 - delta)
                        sta     $2c         ; Store new player position into $2C (accumulated rotation value)
                        ldx     open_level  ; Load open_level flag (non-zero if level has open ends)
                        beq     loc979d     ; If level is closed (open_level = 0), skip open-end boundary checks
                        cmp     #$f0        ; Compare new position to $F0 (240 dec, beyond max for open level)
                        bcc     loc9786     ; If position < $F0, within allowed range (no upper clamp needed)
                        lda     #$ef        ; If position >= $F0, clamp to $EF (239, max allowed on open level)
                        sta     $2c         ; Store the clamped position back to $2C
loc9786:                eor     $2b         ; XOR new position with delta to check for crossing end boundaries
                        bpl     loc979d     ; If result >= 0, no wrap-around across ends; skip adjustment
                        lda     $2c         ; Otherwise, prepare to adjust for boundary crossing
                        eor     $51         ; XOR new position with old position to detect end-to-end wrap
                        bpl     loc979d     ; If result >= 0, no end-to-end crossing; skip further adjustment
                        lda     $51         ; Load old position again to determine crossing direction
                        bmi     loc9799     ; If old position was >= $80, crossed high->low end
                        lda     #$00        ; Else old position was < $80, crossed low->high end; set pos to 0
                        clv                 ; Clear overflow flag
                        bvc     loc979b     ; Always branch (continue to set final boundary position)
loc9799:                lda     #$ef        ; Crossing from high end to low end: set position to $EF (max)
loc979b:                sta     $2c         ; Store adjusted position after open-end wrap-around correction

loc979d:                lda     $2c         ; Load final position value
                        lsr     a           ; Shift right (divide position by 2)
                        lsr     a           ; Shift right (divide by 4)
                        lsr     a           ; Shift right (divide by 8)
                        lsr     a           ; Shift right (divide by 16) (now A = base segment index)
                        sta     $2a         ; Store base segment index (0-15) into $2A
                        clc                 ; Clear carry for next addition
                        adc     #$01        ; Add 1 to segment index (offset by one for internal use)
                        and     #$0f        ; Mask to 0x0F (wrap within 0-15 range)
                        sta     $2b         ; Store adjusted segment index into $2B (new player state)
                        lda     $2a         ; Load base segment index
                        cmp     player_seg  ; Compare with current player segment
                        beq     loc97b6     ; If segment hasn't changed, skip visual update
                        jsr     locccb5     ; If segment changed, update player orientation/graphics
loc97b6:                lda     $2a         ; Load new segment index
                        sta     player_seg  ; Update player's current segment
                        lda     $2b         ; Load new player state (segment offset)
                        sta     player_state ; Update player_state for next frame
                        lda     $2c         ; Load new position accumulator
                        sta     $51         ; Update saved spinner position ($51) for next movement
                        rts                 ; Return from move_player (movement processing complete)


; Find extant enemy which is highest up the tube.  Return -9 or 9 depending
; on which way we need to go to get to it, or -1 if there is no such enemy,
; or 0 if there is but we're already on the correct segment.

; Likely important for attact mode where the automated shooter always 
; moves to shoot the enemy furthest up the tube

demoplay:               lda     #$ff
                        sta     $29
                        sta     $2a
                        ldx     MaxActiveEnemies
loc97ce:                lda     enemy_along,x
                        beq     loc97db
                        cmp     $29
                        bcs     loc97db
                        sta     $29
                        stx     $2a
loc97db:                dex
                        bpl     loc97ce
                        ldx     $2a
                        bmi     loc97f7
                        lda     enemy_seg,x
                        ldy     player_seg
                        jsr     SubYFromAWithWrap
                        tay
                        beq     loc97f7
                        bmi     loc97f5
                        lda     #-9
                        clv
                        bvc     loc97f7
loc97f5:                lda     #$09
loc97f7:                rts
loc97f8:                lda     player_state
                        bpl     loc97fe
                        rts
loc97fe:                lda     $0106
                        bmi     loc9804
                        rts
loc9804:                lda     player_along
                        cmp     #TOP_OF_TUNNEL
                        bne     loc980e
                        jsr     locccee
loc980e:                lda     along_lsb
                        clc
                        adc     zoomspd_lsb
                        sta     along_lsb
                        lda     player_along
                        adc     zoomspd_msb
                        sta     player_along
                        bcs     loc9825
                        cmp     #END_OF_TUNNEL
loc9825:                bcc     loc9833
                        lda     #GS_ZoomOffEnd
                        sta     gamestate
                        jsr     locccf2
                        lda     #$ff
                        sta     player_along
loc9833:                lda     player_along
                        cmp     #$50
                        bcc     loc9842
                        lda     $0115
                        bne     loc9842
                        jsr     loca7bd
loc9842:                lda     $5c
                        clc
                        adc     zoomspd_lsb
                        sta     $5c
                        lda     $5f
                        adc     zoomspd_msb
                        bcc     +
                        inc     $5b
+                       cmp     $5f
                        beq     +
                        inc     $0114
+                       sta     $5f

; Accelerate based on current level value.  The computation here is
; [zoomspd_msb:zoomspd_lsb] += v, where v is
; (((((curlevel<<2)&$ff)<$30)?$30:((curlevel<<2)&$ff))+$20)&$ff, which
; simplifies to (((((curlevel&63)<12)?12:curlevel)<<2)+$20)&$ff.
; This means slow zooms starting at level 56 (where level<<2 hits $e0),
; because the carry out of the +$20 add is explicitly cleared ($9869).

                        lda     curlevel
                        asl     a
                        asl     a
                        cmp     #$30
                        bcc     loc9866 ; branch for 1-11 and 64-74
                        lda     #$30
loc9866:                clc
                        adc     #$20
                        clc
                        adc     zoomspd_lsb
                        sta     zoomspd_lsb

; Why not "bcc 1: ; inc zoomspd_msb ; 1:"?  I have no idea.

                        lda     zoomspd_msb
                        adc     #$00
                        sta     zoomspd_msb
                        lda     player_along
                        cmp     #$f0
                        bcs     loc98a1

; Check for player getting spiked
; I do not understand why scan all segments here, instead of just checking
; the value for player_seg, when $9886/$9889 ensure that only player_seg's
; value actually matters anyway.

                        ldx     #$0f
loc9881:                lda     lane_spike_height,x
                        beq     loc989e
                        cpx     player_seg
                        bne     loc989e
                        cmp     player_along
                        bcs     loc989e
                        jsr     sound_pulsar
                        jsr     pieces_death
                        lda     #$00
                        sta     $0115
                        jsr     ClearAllShots
loc989e:                dex
                        bpl     loc9881
loc98a1:                rts

create_enemies:         ldy     #$00
                        sty     $014f
                        lda     NumEnemiesInTube
                        clc
                        adc     NumEnemiesOnTop
                        cmp     MaxActiveEnemies
                        bcc     loc98b7
                        beq     loc98b7

                        ldy     #$ff
loc98b7:                lda     zap_running
                        beq     loc98be
                        ldy     #$ff
loc98be:                sty     $2f

                        ldx     #$3f
loc98c2:                lda     pending_vid,x
                        beq     next_slot
                        bit     $2f
                        bmi     loc98ee
                        sec
                        sbc     #$01
                        sta     pending_vid,x
                        bne     loc98d9
                        jsr     loc9923
                        clv
                        bvc     loc98ee
loc98d9:                cmp     #$3f
                        bne     loc98ee
                        ldy     pending_seg,x
                        lda     $014f
                        ora     $014f
                        and     locca38,y
                        beq     loc98ee
                        inc     pending_vid,x
loc98ee:                lda     pending_vid,x
                        cmp     #$40
                        bcc     loc9909
                        lda     timectr
                        and     #$01
                        bne     loc9906
                        lda     pending_seg,x
                        clc
                        adc     #$01
                        and     #$0f
                        sta     pending_seg,x
loc9906:                clv
                        bvc     next_slot
loc9909:                cmp     #$20
                        bcc     next_slot
                        ldy     pending_seg,x
                        lda     locca38,y
                        ora     $014f
                        sta     $014f
next_slot:              dex
                        bpl     loc98c2
                        lda     $014f
                        sta     $0150
                        rts

loc9923:                lda     #$f0
                        sta     $29
                        lda     pending_seg,x
                        sta     $2a
                        stx     $35
                        jsr     CreateNewEnemy
                        ldx     $35
                        lda     $29
                        beq     loc9945
                        jsr     loc994d
                        beq     loc9945
                        dec     enemies_pending
                        lda     #$00
                        sta     pending_vid,x
                        rts

loc9945:                lda     #$ff
                        sta     $2f
                        inc     pending_vid,x
                        rts

loc994d:                sty     $36
                        ldy     MaxActiveEnemies
loc9952:                lda     enemy_along,y
                        bne     loc999d
                        lda     $29                 ; along value
                        sta     enemy_along,y
                        lda     $2a                 ; segment number
                        cmp     #$0f
                        bne     loc996c
                        bit     open_level          ; If this is on segement 16 and we're not a closed level (hence no 16) pick a random one
                        bpl     loc996c
                        lda     pokey1_rand
                        and     #$0e
loc996c:                sta     enemy_seg,y
                        clc
                        adc     #$01
                        and     #$0f
                        sta     more_enemy_info,y
                        lda     #$00
                        sta     shot_delay,y
                        lda     $2c
                        sta     active_enemy_info,y
                        lda     $2d
                        sta     enm_move_pc,y
                        inc     NumEnemiesInTube
                        lda     $2b
                        sta     enemy_type_info,y
                        ldy     $36
                        and     #ENEMY_TYPE_MASK
                        stx     $36
                        tax
                        inc     n_enemy_by_type,x
                        ldx     $36
                        lda     #$10
                        rts
loc999d:                dey
                        bpl     loc9952
                        ldy     $36
                        lda     #$00
                        rts

; Pick an enemy type to create a new enemy as.
; First, compute the number available to be created for each type.

CreateNewEnemy:         lda     #$00
                        ldx     #$04
-                       sta     avl_enemy_by_type,x         ; Start with zero of each enemy type available
                        dex
                        bpl     -

                        ldx     #$04
-                       lda     max_enemy_by_type,x         ; Find available enemies by subtracting current from max
                        sec
                        sbc     n_enemy_by_type,x
                        bcc     +
                        sta     avl_enemy_by_type,x
+                       dex
                        bpl     -

; Now, count each tanker as two of the enemy type it's holding.
; Note that this can push the availability number through zero, in which
; case it wraps around to 255, but we use 'bpl' so it still works

                        ldy     MaxActiveEnemies
loc99c3:                lda     enemy_along,y
                        beq     loc99dc
                        lda     active_enemy_info,y
                        and     #$03
                        beq     loc99dc
                        tax
                        cpx     #$03                        ; 3 means fuseball, not tanker!
                        bne     loc99d6

                        ldx     #$05
loc99d6:                dec     avl_enemy_by_type-1,x               
                        dec     avl_enemy_by_type-1,x               
loc99dc:                dey
                        bpl     loc99c3

; Take this level's maximum enemy count, plus one, and subtract off the
; counts of each type of enemy.

                        ldx     #$04
                        lda     MaxActiveEnemies
                        clc
                        adc     #$01
loc99e7:                sec
                        sbc     n_enemy_by_type,x
                        dex
                        bpl     loc99e7

; Limit the number-available for each enemy type to the number we just
; computed, the total number of enemies available.  (In particular, this
; deals with the worst cases where availability has wrapped around.)

                        ldx     #$04
loc99f0:                cmp     avl_enemy_by_type,x
                        bcs     loc99f8
                        sta     avl_enemy_by_type,x
loc99f8:                dex
                        bpl     loc99f0

; Figure out how many enemy types have nonzero availability.

                        ldx     #$04
                        ldy     #$00
loc99ff:                lda     avl_enemy_by_type,x
                        beq     loc9a05
                        iny
loc9a05:                dex
                        bpl     loc99ff

; If no enemy types have nonzero availability, nothing to do.

                        tya
                        beq     loc9a82

; If only one type has nonzero availability, it's easy.

                        dey
                        bne     loc9a26

; Only one type possible.  Find the type and create the enemy.

                        ldx     #$04
loc9a10:                lda     avl_enemy_by_type,x
                        beq     loc9a20
                        lda     min_enemy_by_type,x
                        beq     loc9a20
                        jsr     loc9a87
                        beq     loc9a20
                        rts
loc9a20:                dex
                        bpl     loc9a10
                        clv
                        bvc     loc9a82

; Hard case:                multiple types possible.
; See if any of the minimum values are unsatisfied.

loc9a26:                sty     $61
                        ldx     #$04
loc9a2a:                lda     avl_enemy_by_type,x
                        beq     loc9a3d
                        lda     n_enemy_by_type,x
                        cmp     min_enemy_by_type,x
                        bcs     loc9a3d
                        jsr     loc9a87
                        beq     loc9a3d
                        rts

loc9a3d:                dex
                        bpl     loc9a2a

; No unsatisfied minima.  If we can do a spiker and we can do a tanker,
; have a look at the shortest spike, and if it's less than $cc high, create
; a spiker, else create a tanker.

                        lda     avl_spikers
                        beq     loc9a61
                        lda     avl_tankers
                        beq     loc9a61
                        ldy     $2a
                        lda     lane_spike_height,y
                        bne     loc9a53
                        lda     #$ff
loc9a53:                ldx     #$03        ; spiker
                        cmp     #$cc
                        bcs     loc9a5b
                        ldx     #$02        ; tanker
loc9a5b:                jsr     loc9a87
                        beq     loc9a61
                        rts

; Nothing yet.  Start at a random point and go through the list of enemies
; up to four times.  Each time through, for each type with nonzero minimum
; and availability, try to create one of it.

loc9a61:                lda     pokey2_rand
                        and     #$03
                        tax
                        inx
                        ldy     #$04
loc9a6a:                lda     min_enemy_by_type,x
                        beq     loc9a7a
                        lda     avl_enemy_by_type,x
                        beq     loc9a7a
                        jsr     loc9a87
                        beq     loc9a7a
                        rts
loc9a7a:                dex
                        bpl     loc9a7f
                        ldx     #$04
loc9a7f:                dey
                        bpl     loc9a6a
loc9a82:                lda     #$00
                        sta     $29
                        rts

; Try to create one enemy of the type found in x.  Return with Z set on
; failure, clear on success.

loc9a87:                txa                             ; x = enemy type
loc9a88:                asl     a
                        tay
                        lda     EnmCreateJumpTable+1,y  ; Dispatch by pushing address onto stack and then doing an RTS
                        pha
                        lda     EnmCreateJumpTable,y
                        pha
                        rts

; Jump table, used by code at 9a87, called from various places

EnmCreateJumpTable:     .word   make_flipper-1          ; flipper
                        .word   make_pulsar-1           ; pulsar
                        .word   make_tanker-1           ; tanker
                        .word   make_spiker-1           ; spiker
                        .word   make_fuseball-1         ; fuseball

make_flipper:           lda     EnemyCanFireTable       ; flipper
                        sta     $2c
                        lda     flipper_move
                        ldy     #$00                    ; flipper
                        beq     loc9af6

make_pulsar:            lda     CanPulsarFire           ; pulsar
                        ora     pulsar_fire
                        ldy     #$01                    ; pulsar
                        bne     loc9af1

make_fuseball:          ldy     #$04                    ; fuseball
                        bne     loc9aee

make_spiker:            ldy     #$03                    ; spiker
                        bne     loc9aee

make_tanker:            lda     pokey1_rand             ; tanker
                        and     #$03
                        tay

                        lda     #$04
                        sta     $2b
                        stx     $39
loc9ac7:                dec     $2b
                        bpl     loc9ad0
                        ldx     $39
                        lda     #$00
                        rts

loc9ad0:                dey
                        bpl     loc9ad5
                        ldy     #$03
loc9ad5:                ldx     tanker_load,y
                        cpx     #$03
                        bne     loc9ade
                        ldx     #$05
loc9ade:                lda     $013c,x
                        beq     loc9ac7
                        ldx     $39
                        lda     tanker_load,y
                        ora     #$40
                        ldy     #$02
                        bne     loc9af1
loc9aee:                lda     EnemyCanFireTable,y
loc9af1:                sta     $2c
                        lda     InitialPCodePC,y
loc9af6:                sty     $2b
                        sta     $2d
                        lda     $29
                        rts

; Values for $2d, per-enemy-type.  See $9af3.
; I think these are the initial movement p-code pc values.
; The flipper value is mostly ignored, using flipper_move instead.

InitialPCodePC:         .byte   $07             ; flipper
                        .byte   $72             ; pulsar
                        .byte   $07             ; tanker
                        .byte   $00             ; spiker
                        .byte   $61             ; fuseball

; Values for $2c, per-enemy-type.  See code at $9a87 and the fragments it
; branches to.  This ends up in the active_enemy_info vector for the enemy.

; When set to $40, I believe that indicates that this enemy can fire shots

EnemyCanFireTable:      
                        .byte   $40             ; flipper
CanPulsarFire:          .byte   $00             ; pulsar - ORed with pulsar_fire; see $9aac
                        .byte   $41             ; tanker - not actually used; see $9abb..$9aec
                        .byte   $40             ; spiker
                        .byte   $00             ; fuseball

loc9b07:                sty     $36
                        lda     $29
                        cmp     #$20
                        lda     $2b
                        bcs     loc9b18
                        tay
                        jsr     loc9aee
                        clv
                        bvc     loc9b1b
loc9b18:                jsr     loc9a88
loc9b1b:                ldy     $36
                        rts
move_enemies:           lda     player_state
                        bmi     loc9b56
                        ldx     MaxActiveEnemies
                        stx     $37
loc9b28:                ldx     $37
                        lda     enemy_along,x
                        beq     loc9b52
                        lda     #$01
                        sta     pcode_run
                        lda     enm_move_pc,x
                        sta     pcode_pc

;----------------------------------------------------------------------------
; P-Code Engine for enemy behaviors
;----------------------------------------------------------------------------
;
; The engine's pc is $010b, with the code itself at $a0f7.  The jump
; table at $9ba2 and the code it points to determines the actions of
; each p-opcode.  The p-machine is halted (ie, the interpreter loop
; here is exited) by setting $010a to zero.
;
;----------------------------------------------------------------------------

PCodeMainLoop:          lda     pcode_pc
                        tay
                        lda     PCodeProgram,y
                        jsr     ExecutePCodeOp
                        inc     pcode_pc
                        lda     pcode_run
                        bne     PCodeMainLoop

                        lda     pcode_pc
                        sta     enm_move_pc,x

loc9b52:                dec     $37
                        bpl     loc9b28
loc9b56:                lda     pulsing
                        clc
                        adc     pulse_beat
                        tay
                        eor     pulsing
                        sty     pulsing
                        bpl     loc9b7c
                        tya
                        bpl     loc9b6f
                        jsr     sound_pulsar
                        clv
                        bvc     loc9b7c
loc9b6f:                lda     n_pulsars               ; No pulsars to check for
                        beq     loc9b7c
                        lda     player_state
                        bmi     loc9b7c                 ; Player dying, don't pulse them
                        jsr     loccd02
loc9b7c:                lda     pulsing
                        bmi     loc9b88
                        cmp     #$0f
                        bcs     loc9b8c
                        clv
                        bvc     loc9b97
loc9b88:                cmp     #$c1
                        bcs     loc9b97
loc9b8c:                lda     pulse_beat
                        eor     #$ff
                        clc
                        adc     #$01
                        sta     pulse_beat
loc9b97:                rts

ExecutePCodeOp:         tay                                 ; A must contain the opcode which is a multiple of 2
                        lda     PCodeDispatch+1,y           ; Fetch the high byte
                        pha                                 ;   ...and push it on the stack as the high byte of the 'return' address
                        lda     PCodeDispatch,y             ; Fetch the low byte
                        pha                                 ;   ...and push it on the stack as the low byte of the 'return' address 
                        rts                                 ; Now 'return' to the address we just pushed

; See $9b3a for what this jump table is.

;--------------------------------------------------------------------------------------------
; PCode Operations
;--------------------------------------------------------------------------------------------

                        PCOP_Halt                   .equ    $00
                        PCOP_Store                  .equ    $02
                        PCOP_Skip2IfZero            .equ    $04
                        PCOP_Jump                   .equ    $06
                        PCOP_DecBranchIfElse        .equ    $08
                        PCOP_NOP                    .equ    $0a
                        PCOP_MoveTowardsTop         .equ    $0c
                        PCOP_SpikerStuff            .equ    $0e
                        PCOP_GetGameState           .equ    $10
                        PCOP_StartFlip              .equ    $12
                        PCOP_ContFInishFlip         .equ    $14
                        PCOP_ReverseLeftRightDir    .equ    $16
                        PCOP_CheckGrabPlayer        .equ    $18
                        PCOP_BranchOnZero           .equ    $1a
                        PCOP_CheckIfPastSpike       .equ    $1c
                        PCOP_FuseballMove           .equ    $1e
                        PCOP_CheckPlayerColl        .equ    $20
                        PCOP_PulsarMove             .equ    $22
                        PCOP_AimTowardsPlayerLR     .equ    $24
                        PCOP_CheckIfPulsing         .equ    $26

;--------------------------------------------------------------------------------------------
; PCode Dispatch Jump Table
;--------------------------------------------------------------------------------------------

PCodeDispatch:          .word   PC_Halt-1                   ; 00 = halt
                        .word   PC_Store-1                  ; 02 = next.byte -> enm_pc_storage,x
                        .word   PC_Skip2IfZero-1            ; 04 = if $010c holds zero, skip next two.bytes
                        .word   PC_Jump-1                   ; 06 = unconditional branch
                        .word   PC_DecBranchElseSkip-1      ; 08 = if (--enm_pc_storage,x) branch else skip
                        .word   PC_NOP-1                    ; 0a = nop
                        .word   PC_MoveTowardsTop-1         ; 0c = move per its type's speed setting, also handles reaching end-of-tube
                        .word   PC_SpikerStuff-1            ; 0e = grow spike, reverse, convert
                        .word   PC_GetGameState-1           ; 10 = $00<next.byte> contents -> enm_pc_storage,x
                        .word   PC_StartFlip-1              ; 12 = start flip
                        .word   PC_ContFinishFlip-1         ; 14 = continue/end flip
                        .word   PC_ReverseLeftRightDir-1    ; 16 = reverse direction (segmentwise)
                        .word   PC_CheckGrabPlayer-1        ; 18 = check and maybe grab player
                        .word   PC_BranchOnZero-1           ; 1a = if $010c == 0, branch
                        .word   PC_CheckIfPastSpike-1       ; 1c = enemy-above-spike? -> $010c
                        .word   PC_FuseballMove-1           ; 1e = fuseball movement?
                        .word   PC_CheckPlayerColl-1        ; 20 = check for enemy-touches-player death?
                        .word   PC_PulsarMove-1             ; 22 = do pulsar motion
                        .word   PC_AimTowardsPlayerLR-1     ; 24 = set enemy direction towards player
                        .word   PC_CheckIfPulsing-1         ; 26 = check for pulsing

PC_Halt:                lda     #$00                        ; Store zero in the pcode_run var to stop interpreter
                        sta     pcode_run                   ;   then fall through to the RTS used by the NOP

PC_NOP:                 rts                                 ; NOP, so just return 

PC_Store:               inc     pcode_pc                    ; Increment PC to skip past load byte
                        ldy     pcode_pc                    ; Load PC into Y register
                        lda     PCodeProgram,y              ; Load the byte into A from PCode storage
                        sta     enm_pc_storage,x            ; Store the byte in enemy storage
                        rts

PC_GetGameState:        inc     pcode_pc
                        ldy     pcode_pc
                        lda     PCodeProgram,y
                        tay
                        lda     gamestate,y
                        sta     enm_pc_storage,x
                        rts

PC_Skip2IfZero:         lda     $010c
                        bne     +
                        inc     pcode_pc
                        inc     pcode_pc
+                       rts

PC_BranchOnZero:        inc     pcode_pc
                        lda     $010c
                        bne     +
                        ldy     pcode_pc
                        lda     PCodeProgram,y
                        sta     pcode_pc
+                       rts

PC_DecBranchElseSkip:   dec     enm_pc_storage,x
                        bne     PC_Jump
                        inc     pcode_pc
                        clv
                        bvc     +

PC_Jump:                ldy     pcode_pc
                        lda     DoSpikerStuff,y
                        sta     pcode_pc
+                       rts

; Set $010c to 1 if the enemy is above the end of its segment's spike, 0 if not.

PC_CheckIfPastSpike:    ldy     enemy_seg,x
                        lda     lane_spike_height,y
                        bne     +
                        lda     #$ff
+                       cmp     enemy_along,x
                        bcs     loc9c35
                        lda     #$00
                        clv
                        bvc     loc9c37
loc9c35:                lda     #$01
loc9c37:                sta     $010c
                        rts

; Set $010c to $80 if we're pulsing now, or we will be four ticks in the future, or to $00 if not.

PC_CheckIfPulsing:      lda     pulse_beat
                        asl     a
                        asl     a
                        clc
                        adc     pulsing
                        and     pulsing
                        and     #$80
                        eor     #$80
                        sta     $010c
                        rts

PC_ReverseLeftRightDir: lda     enemy_type_info,x
                        eor     #$40                    ; Flip the "segment increasing or decreasing" bit to change direction
                        sta     enemy_type_info,x
                        rts

PC_MoveTowardsTop:      lda     enemy_type_info,x
                        and     #ENEMY_TYPE_MASK
                        tay
                        lda     active_enemy_info,x
                        bmi     MoveTowardsFarEnd

MoveTowardsTop:         lda     enemy_along_lsb,x
                        clc
                        adc     spd_flipper_lsb,y
                        sta     enemy_along_lsb,x
                        lda     enemy_along,x
                        adc     spd_flipper_msb,y
                        sta     enemy_along,x
                        cmp     player_along
                        beq     +
                        bcs     not_at_top_yet
+                       jsr     EnemyReachedTop
                        clv
                        bvc     no_tanker_split

not_at_top_yet:         cmp     #$20                    ; Depth of $20 is where tankers will split and open up
                        bcs     no_tanker_split
                        lda     active_enemy_info,x
                        and     #$03
                        beq     no_tanker_split         ; If bottom bits are zero, not a tanker
                        txa
                        pha
                        tay
                        jsr     loca06f
                        pla
                        tax
no_tanker_split:        clv
                        bvc     loc9cb5

MoveTowardsFarEnd:      lda     enemy_along_lsb,x
                        sec
                        sbc     spd_flipper_lsb,y
                        sta     enemy_along_lsb,x
                        lda     enemy_along,x
                        sbc     spd_flipper_msb,y
                        sta     enemy_along,x
                        cmp     #END_OF_TUNNEL
                        bcc     loc9cb5
                        lda     #END_OF_TUNNEL + 2
                        sta     enemy_along,x
loc9cb5:                rts

PC_PulsarMove:          ldy     #ENEMY_TYPE_PULSAR          ; 01
                        lda     active_enemy_info,x
                        bmi     loc9ccd
                        lda     enemy_along,x
                        cmp     lethal_distance
                        bcc     loc9cc7                     ; branch if closer than lethal_distance
                        ldy     #ENEMY_TYPE_FLIPPER         ; flipper
loc9cc7:                jsr     MoveTowardsTop              ; move per speed for type Y, includes EnemyReachedTop call
                        clv
                        bvc     loc9ce4
loc9ccd:                jsr     MoveTowardsFarEnd           ; move away per speed for type Y, leaves enemy_along value in A
                        ldy     enemies_pending
                        bne     loc9cd7
                        lda     #$ff
loc9cd7:                cmp     lethal_distance
                        bcc     loc9ce4                     ; branch if closer than lethal_distance
                        lda     active_enemy_info,x
                        eor     #$80
                        sta     active_enemy_info,x
loc9ce4:                lda     pulsing                     ; Check to see if pulsar kills player with pulse
                        bmi     notpulsed
                        lda     enemy_along,x
                        cmp     lethal_distance
                        bcs     notpulsed                   ; branch if farther away than lethal_distance
                        lda     player_seg
                        cmp     enemy_seg,x
                        bne     notpulsed
                        lda     player_state
                        cmp     more_enemy_info,x
                        bne     notpulsed
                        jsr     pieces_death
notpulsed:              rts
                        
                        .byte $16

; Reached the top of the tube.  Deal with it. X == active enemy number

EnemyReachedTop:        lda     player_along            ; Set enemy height to player height (why not just the same constant #$10, not sure?)
                        sta     enemy_along,x
                        
                        lda     enemy_type_info,x
                        and     #ENEMY_TYPE_MASK        ; Bottom 3 bits are enemy type
                        cmp     #ENEMY_TYPE_PULSAR      ; If it's not a pulsar, stick it to the top
                        bne     can_stick_to_top

                        lda     enemies_pending         ; But even pulsars stick when no pending enemies are left
                        beq     can_stick_to_top

JustBounceBack:         lda     active_enemy_info,x     ; Switch direction by flipping the $80 bit in the enemy's state info
                        eor     #$80
                        sta     active_enemy_info,x
                        rts

can_stick_to_top:       lda     enemy_type_info,x       ; If this is a flipper or pulsar and it's between segments, push it back down the 
                        bpl     StickToTopRail          ;    tube a notch so that it will have time to finish rotation before sticking to a segment
                        inc     enemy_along,x           ;    I guess also ensures a captured player is captured by a flipper done flipping
                        rts

StickToTopRail:         dec     NumEnemiesInTube        ; Move one enemy count from tube to top
                        lda     NumEnemiesOnTop
                        cmp     #$01                    ; If there's already exactly one enemy up top, don't bother aiming
                        beq     skip_aiming
                        jsr     PC_AimTowardsPlayerLR
                        clv
                        bvc     loc9d5e

skip_aiming:            ldy     #MAX_ACTIVE_ENEMIES - 1 ; 
loc9d3e:                lda     enemy_along,y
                        beq     loc9d51
                        sty     $38
                        cpx     $38
                        beq     loc9d51
                        lda     enemy_along,y
                        cmp     player_along
                        beq     loc9d54
loc9d51:                dey
                        bpl     loc9d3e

loc9d54:                lda     enemy_type_info,y       ; BUGBUG is this the code that changes pulsars to flippers because it wipes out
                        and     #$40                    ;        the low bits which would contain it's type information?  I think the
                        eor     #$40                    ;        author may have wanted or/eor, not and/eor.  That's typically how he
                        sta     enemy_type_info,x       ;        clears individual bits in similar cases (set the flip rather than mask).
                                                        ;        Could very well be code emitted by their macros as well.
loc9d5e:                lda     #$41
                        sta     pcode_pc
                        inc     NumEnemiesOnTop
                        rts

PC_AimTowardsPlayerLR:

                        lda     enemy_seg,x
                        tay
                        lda     player_seg
                        jsr     SubYFromAWithWrap       ; Figure out which direction player is (left or right)
                        asl     a
                        lda     enemy_type_info,x       ; Now point the enemy in the correct direction
                        bcs     loc9d7c
                        ora     #$40                    ; Enemy segment value will now be increasing
                        clv
                        bvc     loc9d7e
loc9d7c:                and     #~$40                   ; Enemy segment value will now be decreasing
loc9d7e:                sta     enemy_type_info,x
                        rts

; This code is used to continue and maybe end a flipper's flip, or other
; enemy movement from one segment to the next.

PC_ContFinishFlip:      ldy     more_enemy_info,x
                        lda     enemy_type_info,x
                        and     #$40
                        bne     loc9d90
                        iny
                        clv
                        bvc     loc9d91
loc9d90:                dey
loc9d91:                tya
                        and     #$0f
                        ora     #$80
                        sta     more_enemy_info,x
                        lda     enemy_type_info,x
                        and     #ENEMY_TYPE_MASK
                        cmp     #$04                        ; fuseball
                        bne     loc9dee
                        lda     more_enemy_info,x
                        and     #$07
                        bne     loc9deb
                        lda     more_enemy_info,x
                        and     #$08
                        beq     loc9dbb
                        lda     enemy_seg,x
                        clc
                        adc     #$01
                        and     #$0f
                        sta     enemy_seg,x
loc9dbb:                lda     enemy_type_info,x
                        and     #$7f
                        sta     enemy_type_info,x
                        lda     #$20
                        sta     more_enemy_info,x
                        lda     active_enemy_info,x
                        eor     #$80
                        sta     active_enemy_info,x
                        lda     enemies_pending
                        bne     loc9deb
                        lda     enemy_along,x
                        cmp     player_along
                        bne     loc9de3
                        jsr     loc9f81
                        clv
                        bvc     loc9deb
loc9de3:                lda     active_enemy_info,x
                        and     #$80
                        sta     active_enemy_info,x
loc9deb:                clv
                        bvc     loc9e26

; check if flip ended

loc9dee:                ldy     enemy_seg,x
                        lda     enemy_type_info,x
                        eor     #$40
                        jsr     get_angle
                        cmp     more_enemy_info,x
                        bne     loc9e26

; yes, stop flipping

                        lda     enemy_type_info,x
                        and     #$7f
                        sta     enemy_type_info,x
                        and     #$40
                        bne     loc9e1b
                        lda     enemy_seg,x
                        sta     more_enemy_info,x
                        sec
                        sbc     #$01
                        and     #$0f
                        sta     enemy_seg,x
                        clv
                        bvc     loc9e26
loc9e1b:                lda     enemy_seg,x
                        clc
                        adc     #$01
                        and     #$0f
                        sta     more_enemy_info,x
loc9e26:                lda     enemy_type_info,x
                        and     #$80
                        sta     $010c
                        rts

PC_CheckGrabPlayer:     lda     enemy_type_info,x
                        bmi     loc9e47
                        lda     enemy_seg,x
                        cmp     player_seg
                        bne     loc9e47
                        lda     more_enemy_info,x
                        cmp     player_state
                        bne     loc9e47
                        jsr     loca33a
loc9e47:                rts

PC_CheckPlayerColl:     lda     enemy_along,x
                        cmp     player_along
                        bne     loc9e5b
                        lda     enemy_seg,x
                        cmp     player_seg
                        bne     loc9e5b
                        jsr     loca343
loc9e5b:                rts

PC_StartFlip:           jsr     rev_if_edge
loc9e5f:                lda     enemy_type_info,x
                        ora     #$80
                        sta     enemy_type_info,x
                        and     #ENEMY_TYPE_MASK
                        cmp     #ENEMY_TYPE_FUSEBALL
                        bne     loc9e8c
                        lda     enemy_type_info,x
                        and     #$40
                        bne     loc9e79
                        lda     #$81
                        clv
                        bvc     loc9e86
loc9e79:                lda     enemy_seg,x
                        sec
                        sbc     #$01
                        and     #$0f
                        sta     enemy_seg,x
                        lda     #$87
loc9e86:                sta     more_enemy_info,x
                        clv
                        bvc     loc9eaa
loc9e8c:                lda     enemy_type_info,x
                        and     #$40
                        beq     loc9e9e
                        lda     enemy_seg,x
                        clc
                        adc     #$01
                        and     #$0f
                        sta     enemy_seg,x
loc9e9e:                lda     enemy_type_info,x
                        ldy     enemy_seg,x
                        jsr     get_angle
                        sta     more_enemy_info,x
loc9eaa:                rts

; Reverse motion direction if level open and we've run into an edge.

rev_if_edge:            lda     open_level
                        beq     loc9ed6
                        lda     enemy_type_info,x
                        and     #$40
                        beq     loc9ec9
                        lda     enemy_seg,x
                        cmp     #$0e
                        bcc     loc9ec6
                        lda     enemy_type_info,x
                        and     #$bf
                        sta     enemy_type_info,x
loc9ec6:                clv
                        bvc     loc9ed6
loc9ec9:                lda     enemy_seg,x
                        bne     loc9ed6
                        lda     enemy_type_info,x
                        ora     #$40
                        sta     enemy_type_info,x
loc9ed6:                rts

; Enter with motion value in A, segment number in Y; returns with angle
; value ($0-$f) in A, |ed with $80.

get_angle:              and     #$40
                        beq     loc9eeb
                        dey
                        tya
                        and     #$0f
                        tay
                        lda     tube_angle,y
                        clc
                        adc     #$08
                        and     #$0f
                        clv
                        bvc     loc9eee
loc9eeb:                lda     tube_angle,y
loc9eee:                ora     #$80
                        rts

PC_FuseballMove:        ldy     #$04
                        lda     active_enemy_info,x
                        bmi     loc9f43
                        lda     enemy_along_lsb,x
                        clc
                        adc     spd_fuseball_lsb
                        sta     enemy_along_lsb,x
                        lda     enemy_along,x
                        adc     spd_fuseball_msb
                        sta     enemy_along,x
                        cmp     player_along
                        bcs     loc9f19
                        lda     player_along
                        sta     enemy_along,x
                        clv
                        bvc     loc9f2a
loc9f19:                ldy     enemies_pending         ; if no pending, rush to top
                        beq     loc9f29
                        ldy     curlevel
                        cpy     #$11
                        bcs     loc9f26                 ; branch if level >= 17
                        cmp     #$20
loc9f26:                clv
                        bvc     loc9f2a
loc9f29:                rts

loc9f2a:                bcs     loc9f3d
                        lda     fuse_move_flg
                        bpl     loc9f37
                        jsr     loc9f81
                        clv
                        bvc     loc9f3a
loc9f37:                jsr     loc9f8a
loc9f3a:                clv
                        bvc     loc9f40
loc9f3d:                jsr     loc9f5f
loc9f40:                clv
                        bvc     loc9f5e
loc9f43:                jsr     MoveTowardsFarEnd       ; move away per speed for type Y
                        cmp     #$80
                        bcc     loc9f5b
                        bit     fuse_move_flg
                        bvc     loc9f55                 ; branch if level < 17
                        jsr     loc9f81
                        clv
                        bvc     loc9f58
loc9f55:                jsr     loc9f8a
loc9f58:                clv
                        bvc     loc9f5e
loc9f5b:                jsr     loc9f5f
loc9f5e:                rts
loc9f5f:                lda     enemy_along,x
                        and     #$20
                        beq     loc9f80
                        lda     pokey2_rand
                        cmp     fuse_move_prb
                        bcc     loc9f80
                        bit     fuse_move_flg
                        bvc     loc9f7d
                        txa
                        lsr     a
                        bcc     loc9f8a
                        jsr     loc9f81
                        clv
                        bvc     loc9f80
loc9f7d:                jsr     loc9f8a
loc9f80:                rts

loc9f81:                jsr     PC_AimTowardsPlayerLR       ; Aim then reverse == move away from player?
                        jsr     PC_ReverseLeftRightDir
                        jmp     loc9f99
loc9f8a:                lda     enemy_type_info,x
                        and     #$bf
                        bit     pokey1_rand
                        bvc     loc9f96
                        ora     #$40
loc9f96:                sta     enemy_type_info,x
loc9f99:                lda     open_level
                        beq     loc9fbc
                        lda     enemy_type_info,x
                        and     #$40
                        bne     loc9faf
                        lda     enemy_seg,x
                        cmp     #$0f
                        bcs     loc9fb4
                        clv
                        bvc     loc9fbc
loc9faf:                lda     enemy_seg,x
                        bne     loc9fbc
loc9fb4:                lda     enemy_type_info,x
                        eor     #$40
                        sta     enemy_type_info,x
loc9fbc:                lda     #$66
                        sta     pcode_pc
                        jmp     loc9e5f

PC_SpikerStuff:         lda     #$01
                        sta     $010c
                        ldy     enemy_seg,x
                        lda     lane_spike_height,y
                        bne     loc9fd6
                        lda     #$f1
                        sta     lane_spike_height,y
loc9fd6:                lda     enemy_along,x
                        cmp     lane_spike_height,y
                        bcs     loc9fe6
                        sta     lane_spike_height,y
                        lda     #$80
                        sta     $039a,y
loc9fe6:                lda     enemy_along,x
                        cmp     #$20
                        bcs     loc9ffd
                        lda     active_enemy_info,x
                        ora     #$80
                        sta     active_enemy_info,x
                        lda     #$20
                        sta     enemy_along,x
                        clv
                        bvc     loca027
loc9ffd:                cmp     #$f2
                        bcc     loca027
                        jsr     spiker_hop
                        lda     #$f0
                        sta     enemy_along,x

                        lda     enemies_pending
                        bne     loca027

; If no enemies pending, turn it into a flipper-holding tanker.  This is the code that causes that single last enemy to always
; come out as a tanker.

                        lda     active_enemy_info,x
                        and     #$fc
                        ora     #$01
                        sta     active_enemy_info,x
                        lda     enemy_type_info,x
                        and     #~ENEMY_TYPE_MASK
                        ora     #ENEMY_TYPE_TANKER
                        sta     enemy_type_info,x
                        lda     #$00
                        sta     $010c
loca027:                rts

spiker_hop:             lda     #$00
                        sta     $2d
                        lda     #$0f
                        sta     avl_spikers
                        lda     pokey2_rand
                        and     #$0f
                        tay
loca037:                cpy     #$0f
                        bne     loca040
                        lda     open_level
                        bne     loca04f
loca040:                lda     lane_spike_height,y
                        bne     loca047
                        lda     #$ff
loca047:                cmp     $2d
                        bcc     loca04f
                        sta     $2d
                        sty     $29
loca04f:                dey
                        bpl     loca054
                        ldy     #$0f
loca054:                dec     avl_spikers
                        bpl     loca037
                        lda     $29
                        sta     enemy_seg,x
                        clc
                        adc     #$01
                        and     #$0f
                        sta     more_enemy_info,x
                        lda     active_enemy_info,x
                        and     #$7f
                        sta     active_enemy_info,x
                        rts

; Enemy has reached the $20 point in the tube.  Handle it.

loca06f:                lda     enemy_along,y
                        sta     $29
                        cmp     player_along
                        bne     loca088
                        lda     enemy_type_info,y
                        and     #ENEMY_TYPE_MASK
                        cmp     #ENEMY_TYPE_FUSEBALL
                        beq     loca088
                        dec     NumEnemiesOnTop                 ; All enemies except fuseballs dec enemy topcount
                        clv
                        bvc     +
loca088:                dec     NumEnemiesInTube
+                       lda     #$00
                        sta     enemy_along,y
                        lda     enemy_type_info,y
                        and     #ENEMY_TYPE_MASK
                        stx     $35
                        tax
                        dec     n_enemy_by_type,x
                        ldx     $35
                        lda     active_enemy_info,y
                        and     #$03
                        beq     loca0f6
                        sec
                        sbc     #$01
                        cmp     #$02
                        bne     loca0ad
                        lda     #$04
loca0ad:                sta     $2b
                        lda     enemy_seg,y
                        sec
                        sbc     #$01
                        and     #$0f
                        cmp     #$0f
                        bcc     loca0c2
                        bit     open_level
                        bpl     loca0c2
                        lda     #$00
loca0c2:                sta     $2a
                        jsr     loc9b07
                        lda     $2d
                        sta     pcode_pc
                        dec     pcode_pc
                        lda     #$00
                        sta     pcode_run
                        jsr     loc994d
                        beq     loca0f6
                        lda     $2a
                        clc
                        adc     #$02
                        and     #$0f
                        cmp     #$0f
                        bne     loca0eb
                        bit     open_level
                        bpl     loca0eb
                        lda     #$0e
loca0eb:                sta     $2a
                        lda     $2b
                        ora     #$40
                        sta     $2b
                        jsr     loc994d
loca0f6:                rts

; See the comments on $9b3a for what this is.

;----------------------------------------------------------------------------------------------------------
; PCode Program
;----------------------------------------------------------------------------------------------------------
; The following is the "program" that is executed in PCode.  In order to make it readable, I've added a
; macro that yields the relative address of a label (relative to the beginning of the PCode Program) less
; one, which is how the table was coded by Atari.  I've opeted NOT to do the entire PCode program because
; I do not want to introduce a huge dependency on a macro, and the first few blocks can be returned to
; "normal" fairly easily.  The rest I left in binary as an exercise for the reader :-)
;----------------------------------------------------------------------------------------------------------

.MACRO  PCADDR, ?arg
    .byte ?arg-PCodeProgram-1
.ENDM

PCodeProgram:
    
SpikerEntry:            .byte   PCOP_MoveTowardsTop      ; 00:                move per speed
DoSpikerStuff:          .byte   PCOP_SpikerStuff         ; 01:                spike, reverse, convert to tanker
                        .byte   PCOP_BranchOnZero        ; 02:                branch conditional (if converted to tanker)
                         pcaddr JustMoveUp               ; 03:                   to 07
                        .byte   PCOP_Halt                ; 04:                done
                        .byte   PCOP_Jump                ; 05:                branch
                         pcaddr PCodeProgram             ; 06:                   to 00

; Entry point for "just move up".  Used for tankers, for flippers on some
; levels (shape 14 and shape 1, for example) and in some cases for the pieces when a tanker splits...

JustMoveUp:             .byte   PCOP_MoveTowardsTop      ; 07:                move per speed  <--+
                        .byte   PCOP_Halt                ; 08:                done               |
                        .byte   PCOP_Jump                ; 09:                branch             |
                         pcaddr JustMoveUp               ; 0a:                   to 07 ----------+

; Flipper entry point:                move 8 times, flip, repeat.  Don't move during flip.

FlipperEntry:           .byte   PCOP_Store               ; 0b:                store in enm_pc_storage,x...
                        .byte   $08                      ; 0c:                   ...08
MoveUp:                 .byte   PCOP_MoveTowardsTop      ; 0d:                move per speed
                        .byte   PCOP_Halt                ; 0e:                done
                        .byte   PCOP_DecBranchIfElse     ; 0f:                if --enm_pc_storage,x then branch
                         pcaddr MoveUp                   ; 10:                   to 0d
                        .byte   PCOP_StartFlip           ; 11:                start flip
HaltFlipper:            .byte   PCOP_Halt                ; 12:                done
                        .byte   PCOP_ContFInishFlip      ; 13:                continue/end flip
                        .byte   PCOP_Skip2IfZero         ; 14:                if $010c == 0, skip to 17
                        .byte   PCOP_Jump                ; 15:                branch
                         pcaddr HaltFlipper              ; 16:                   to 12
                        .byte   PCOP_Jump                ; 17:                branch
                         pcaddr FlipperEntry             ; 18:                   to 0b

; Flipper entry point:                flip constantly, moving for one tick between flips.

                        .byte   $0c ; 19:                move per speed
                        .byte   $00 ; 1a:                done
                        .byte   $12 ; 1b:                start flip
                        .byte   $00 ; 1c:                done
                        .byte   $14 ; 1d:                continue/end flip
                        .byte   $0c ; 1e:                move per speed
                        .byte   $04 ; 1f: if $010c == 0, skip to 22
                        .byte   $06 ; 20:                branch
                        .byte   $1b ; 21:   to 1c
                        .byte   $06 ; 22:                branch
                        .byte   $18 ; 23:   to 19

; Flipper entry point:                flips twice one way, three times the other, twice,
; three times, twice, three times, etc.  Move on every tick except the
; ones on which we start a flip.

                        .byte   $0c ; 24:                move per speed
                        .byte   $00 ; 25:                done
                        .byte   $02 ; 26:                store in enm_pc_storage,x...
                        .byte   $02 ; 27:   ...02
                        .byte   $12 ; 28:                start flip
                        .byte   $00 ; 29:                done
                        .byte   $14 ; 2a:                continue/end flip
                        .byte   $0c ; 2b:                move per speed
                        .byte   $04 ; 2c: if $010c == 0, skip to 2f
                        .byte   $06 ; 2d:                branch
                        .byte   $28 ; 2e:   to 29
                        .byte   $00 ; 2f:                done
                        .byte   $08 ; 30: if --enm_pc_storage,x then branch
                        .byte   $27 ; 31:   to 28
                        .byte   $16 ; 32:                reverse direction
                        .byte   $02 ; 33:                store in enm_pc_storage,x...
                        .byte   $03 ; 34:   ...03
                        .byte   $12 ; 35:                start flip
                        .byte   $00 ; 36:                done
                        .byte   $14 ; 37:                continue/end flip
                        .byte   $0c ; 38:                move per speed
                        .byte   $04 ; 39: if $010c == 0, skip to 3c
                        .byte   $06 ; 3a:                branch
                        .byte   $35 ; 3b:   to 36
                        .byte   $00 ; 3c:                done
                        .byte   $08 ; 3d: if --enm_pc_storage,x then branch
                        .byte   $34 ; 3e:   to 35
                        .byte   $16 ; 3f:                reverse direction
                        .byte   $06 ; 40:                branch
                        .byte   $23 ; 41:   to 24

; Action 0c jumps here upon reaching top-of-tube.

                        .byte   $02 ; 42:                store in enm_pc_storage,x...
                        .byte   $04 ; 43:   ...04
                        .byte   $18 ; 44:                check and maybe grab player
                        .byte   $00 ; 45:                done
                        .byte   $08 ; 46: if --enm_pc_storage,x then branch
                        .byte   $43 ; 47:   to 44
                        .byte   $12 ; 48:                start flip
                        .byte   $00 ; 49:                done
                        .byte   $10 ; 4a:                flip_top_accel -> enm_pc_storage,x
                        .byte   $b3 ; 4b:   (value for previous)
                        .byte   $14 ; 4c:                continue/end flip
                        .byte   $1a ; 4d: if $010c == 0, branch
                        .byte   $41 ; 4e:   to 42
                        .byte   $08 ; 4f: if --enm_pc_storage,x then branch
                        .byte   $4b ; 50:   to 4c
                        .byte   $06 ; 51:                branch
                        .byte   $48 ; 52:   to 49

; Flipper entry point: for levels where flippers ride spikes.
; Move every tick.

                        .byte   $00 ; 53:                done
                        .byte   $0c ; 54:                move per speed
                        .byte   $1c ; 55:                set $010c to enemy-above-spike-p
                        .byte   $1a ; 56: if $010c == 0, branch
                        .byte   $52 ; 57:   to 53
                        .byte   $12 ; 58:                start flip
                        .byte   $00 ; 59:                done
                        .byte   $0c ; 5a:                move per speed
                        .byte   $14 ; 5b:                continue/end flip
                        .byte   $1a ; 5c: if $010c == 0, branch
                        .byte   $52 ; 5d:   to 53
                        .byte   $00 ; 5e:                done
                        .byte   $06 ; 5f:                branch
                        .byte   $5a ; 60:   to 5b

; fuseball entry point.

                        .byte   $1e ; 61:                fuseball movement?
                        .byte   $20 ; 62:                check for enemy-touches-player death
                        .byte   $00 ; 63:                done
                        .byte   $06 ; 64:                branch
                        .byte   $60 ; 65:   to 61
                        .byte   $00 ; 66:                done

; PC_FuseballMove sets pc to here under some circumstances; see $9fbc.

                        .byte   $02 ; 67:                store in enm_pc_storage,x...
                        .byte   $03 ; 68:   ...03
                        .byte   $20 ; 69:                check for enemy-touches-player death
                        .byte   $00 ; 6a:                done
                        .byte   $08 ; 6b: if --enm_pc_storage,x then branch
                        .byte   $68 ; 6c:   to 69
                        .byte   $14 ; 6d:                continue/end flip
                        .byte   $1a ; 6e: if $010c == 0, branch
                        .byte   $60 ; 6f:   to 61
                        .byte   $06 ; 70:                branch
                        .byte   $65 ; 71:   to 66

; Pulsar entry point.

                        .byte   $10 ; 72:                pulsar_speed -> enm_pc_storage,x
                        .byte   $b2 ; 73:   (value for previous)
                        .byte   $22 ; 74: do pulsar motion
                        .byte   $00 ; 75:                done
                        .byte   $08 ; 76: if --enm_pc_storage,x then branch
                        .byte   $73 ; 77:   to 74
                        .byte   $26 ; 78:                check if pulsing
                        .byte   $1a ; 79: if not pulsing, branch
                        .byte   $7e ; 7a:   to 7f
                        .byte   $22 ; 7b: do pulsar motion
                        .byte   $00 ; 7c:                done
                        .byte   $06 ; 7d:                branch
                        .byte   $77 ; 7e:   to 78
                        .byte   $24 ; 7f:                enemy attract to player
                        .byte   $12 ; 80:                start flip
                        .byte   $00 ; 81:                done
                        .byte   $14 ; 82:                continue/end flip
                        .byte   $1a ; 83: if $010c == 0, branch
                        .byte   $71 ; 84:   to 72
                        .byte   $06 ; 85:                branch
                        .byte   $80 ; 86:   to 81

; Flipper entry point:                flip away from player, move four ticks, repeat.
; Move on every tick except those on which we start flips.

                        .byte   $24 ; 87:                enemy attract to player
                        .byte   $16 ; 88:                reverse enemy direction
                        .byte   $12 ; 89:                start flip
                        .byte   $00 ; 8a:                done
                        .byte   $0c ; 8b:                move per speed
                        .byte   $14 ; 8c:                continue/end flip
                        .byte   $04 ; 8d: if $010c == 0, skip to 90
                        .byte   $06 ; 8e:                branch
                        .byte   $89 ; 8f:   to 8a
                        .byte   $02 ; 90:                store in enm_pc_storage,x...
                        .byte   $04 ; 91:   ...04
                        .byte   $00 ; 92:                done
                        .byte   $0c ; 93:                move per speed
                        .byte   $08 ; 94: if --enm_pc_storage,x then branch
                        .byte   $91 ; 95:   to 92
                        .byte   $06 ; 96:                branch
                        .byte   $86 ; 97:   to 87

; Handle shots

move_shots:             ldx     #MAX_TOTAL_SHOTS-1
                        stx     $37
loca193:                ldx     $37
                        lda     PlayerShotPositions,x
                        beq     loca1df                 ; branch if this shot doesn't exist
                        cpx     #MAX_PLAYER_SHOTS       ; enemy or friendly?
                        bcs     loca1c0                 ; branch if enemy shot

; Friendly shot.  Move it down the tube.  $02f2, if set, appears to slow
; the shot down, presumably so it doesn't go off the back wall before it
; gets a chance to hit a spiker.

                        adc     #$09
                        ldy     $02f2,x
                        beq     loca1a8
                        sec
                        sbc     #$04
loca1a8:                sta     PlayerShotPositions,x

; Check to see if it has gone off the end of the tube

                        jsr     loca1fa
                        lda     PlayerShotPositions,x
                        cmp     #END_OF_TUNNEL
                        bcc     loca1bd

; Shot went off back end of tube; destroy it

                        dec     PlayerShotCount
                        lda     #$00
                        sta     PlayerShotPositions,x
loca1bd:                clv
                        bvc     loca1df

; Enemy shot

loca1c0:                lda     $02e6,x                 ; enm_shot_lsb-8
                        clc
                        adc     enm_shotspd_lsb
                        sta     $02e6,x                 ; enm_shot_lsb-8
                        lda     PlayerShotPositions,x
                        adc     enm_shotspd_msb

; Reached player's end of tube yet?

                        cmp     player_along
                        bcs     loca1dc

; Yes, at this end of tube

                        dec     EnemyShotCount
                        jsr     loca1e4 ; check to see if hit player
                        lda     #$00
loca1dc:                sta     PlayerShotPositions,x

; Next shot

loca1df:                dec     $37
                        bpl     loca193
                        rts

; Called to see if enemy shot hit player.  Enemy shot number is in X,
; offset by 8 (which is why we see PlayerShotSegments,x here instead of the
; EnemyShotSegments,x we'd expect to).

loca1e4:                lda     player_seg
                        cmp     EnemyShotSegments-MAX_PLAYER_SHOTS,x        ; BUGBUG I doubt this really wants to go backwards of EnemyShotSegments!
                        bne     loca1f9
                        lda     player_state
                        bmi     loca1f9
                        jsr     loca34b
                        lda     #$81
                        sta     player_state
loca1f9:                rts

; Called to see if player shot hit a spike.

loca1fa:                ldy     PlayerShotSegments,x
                        lda     lane_spike_height,y
                        beq     nospikehit                  ; Spike height 0 in this lane, so cannot hit
                        lda     PlayerShotPositions,x
                        cmp     lane_spike_height,y
                        bcc     loca22f
                        cmp     #END_OF_TUNNEL
                        bcc     loca210

                        lda     #$00                        ; If shot at end of tunnel, spike is now gone
loca210:                sta     lane_spike_height,y         ; Shot position becomes the new spike height
                        inc     $02f2,x
                        lda     #$c0
                        sta     $039a,y
                        jsr     locccf6
                        ldx     #$ff
                        
                        lda     #$00                        ; Add 1 to score
                        sta     $2a
                        sta     $2b
                        lda     #$01
                        sta     $29
                        jsr     inc_score
                        ldx     $37

loca22f:                lda     $02f2,x
                        cmp     #$02
                        bcc     nospikehit
                        lda     #$00
                        sta     PlayerShotPositions,x
                        dec     PlayerShotCount
nospikehit:             rts

; Check to see if player fires?

CheckPlayerFire:        lda     player_state            ; Check to see if player has died
                        bmi     loca2a5
                        lda     game_mode
                        bmi     FireShot                    ; Could this be attract mode decision to fire?
                        lda     $0106
                        sta     $29
                        ldx     #$0a                    ; BUGBUG Why 10?  There are only 4 max enemy shots, and the playershot table
loca24f:                lda     EnemyShotPositions,x    ;                 comes first so it's not an intentional overreach into that...
                        beq     loca268
                        lda     EnemyShotSegments,x
                        sec
                        sbc     player_seg
                        bpl     loca262
                        eor     #$ff
                        clc
                        adc     #$01
loca262:                cmp     #$02
                        bcs     loca268
                        inc     $29
loca268:                dex
                        bpl     loca24f
                        lda     $29
                        clv
                        bvc     loca274

FireShot:               lda     zap_fire_debounce
                        and     #$10                                ; fire
loca274:                beq     loca2a5
                        ldx     #MAX_PLAYER_SHOTS-1
loca278:                lda     PlayerShotPositions,x
                        bne     loca2a2
                        inc     PlayerShotCount
                        lda     player_along
                        sta     PlayerShotPositions,x
                        lda     player_seg
                        sta     PlayerShotSegments,x
                        lda     player_state
                        sta     $02c0,x
                        lda     #$00
                        sta     $02f2,x
                        jsr     locccea
                        lda     player_along
                        jsr     CheckPlayerShot
                        ldx     #$00
loca2a2:                dex
                        bpl     loca278
loca2a5:                rts

enm_shoot:              lda     player_state
                        bmi     CannotShoot                     ; High bit seems to mean death in progress, so no enemies shoot
                        ldx     #$06
CheckIfShouldShoot:     lda     enemy_along,x
                        beq     NoShot
                        cmp     #$30
                        bcc     NoShot
                        lda     active_enemy_info,x
                        and     #$40                            ; if $40 is not set (fuseball, some pulsars) then it doesn't shoot at all
                        beq     NoShot
                        dec     shot_delay,x
                        bpl     NoShot
                        inc     shot_delay,x                    ; Reset to zero, since it rolled under
                        lda     enemy_type_info,x
                        and     #$80
                        bne     NoShot                          ; Don't shoot while moving away from player
                        lda     pokey1_rand
                        ldy     EnemyShotCount                  ; Our chance of a new enemey shot is probabilistic, with the probability
                        cmp     enm_shot_prob,y                 ;   taken from this table depending on how many active shots there already are
                        bcc     NoShot

                        ldy     MaxEnemyShots
LookForAvailShot:       lda     EnemyShotPositions,y            ; Look for an unused enemy shot, indicated by a zero position
                        bne     shot_not_avail
                        lda     enemy_along,x                   ; Set shot position to wherever the enemy is
                        sta     EnemyShotPositions,y
                        lda     enemy_seg,x
                        sta     EnemyShotSegments,y                 ; Set the shot segment to the same as the enemy who fired it
                        lda     more_enemy_info,x
                        sta     $02c8,y
                        lda     shot_holdoff
                        sta     shot_delay,x                    ; Start the "shot_delay" countdown by setting to initial 'shot_holdoff' value
                        jsr     locccbd
                        inc     EnemyShotCount
                        ldy     #$00
shot_not_avail:         dey
                        bpl     LookForAvailShot
NoShot:                 dex
                        bpl     CheckIfShouldShoot
CannotShoot:            rts

; Chance of a new shot, indexed by number of existing shots.  See $a2cc.

enm_shot_prob:          .byte   $00
                        .byte   $e0
                        .byte   $f0
                        .byte   $fa
                        .byte   $ff

loca309:                stx     $37
                        lda     #$ff
                        sta     $02f2,x
                        tya
                        sec
                        sbc     #$04
                        tay
                        lda     enemy_seg,y
                        sta     $2d
                        lda     pokey2_rand
                        and     #$07
                        cmp     #$03
                        bcc     loca325
                        lda     #$00
loca325:                pha
                        clc
                        adc     #$02
                        jsr     loca3ca
                        jsr     loca06f
                        pla
                        clc
                        adc     #$05
                        tax
                        jsr     inc_score
                        ldx     $37
                        rts
loca33a:                lda     #$05
                        jsr     loca352
                        dec     player_state
                        rts
loca343:                lda     #$09
                        bne     loca34d

pieces_death:           lda     #$07
                        bne     loca34d

loca34b:                lda     #$ff
loca34d:                sta     $013b
                        lda     #$01
loca352:                sta     $2c
                        lda     player_along
                        sta     $29
                        lda     player_seg
                        sta     $2d
                        jsr     locccb0
                        jsr     loca3d6
                        lda     #$81
                        sta     player_state
                        lda     #$01
                        sta     $013c
                        rts

loca36f:                jsr     locccc1
                        lda     EnemyShotPositions,y
                        sta     $29
                        lda     EnemyShotSegments,y
                        sta     $2d
                        lda     #$00
                        jsr     loca3d4
                        lda     #$00
                        sta     EnemyShotPositions,y
                        dec     EnemyShotCount
                        lda     #$ff
                        sta     $02f2,x
                        rts

loca38e:                lda     #$ff
                        sta     $02f2,x
                        tya
                        sec
                        sbc     #$04
                        tay

;-----------------------------------------------------------------------------
; Kill Enemy
;-----------------------------------------------------------------------------
; Enemy number provided in Y register.  Adds score.
;-----------------------------------------------------------------------------

ZapEnemy:               lda     enemy_type_info,y
                        and     #$c0
                        cmp     #$c0
                        beq     loca3a7
                        lda     enemy_seg,y
                        clv
                        bvc     loca3af
loca3a7:                lda     enemy_seg,y
                        sec
                        sbc     #$01
                        and     #$0f
loca3af:                sta     $2d
                        lda     #$00
                        jsr     loca3ca
                        jsr     loca06f
                        lda     enemy_type_info,y
                        and     #ENEMY_TYPE_MASK
                        tay
                        ldx     loca3c5,y
                        jmp     inc_score

loca3c5:                .byte   $01                 ; Flipper   150
                        .byte   $02                 ; Pulsar    200
                        .byte   $03                 ; Tanker    100
                        .byte   $04                 ; Spiker     50
                        .byte   $01                 ; Fuseball  150 - Note that fuseballs are 150 for Superzap instead of 250/500/750 when shot

loca3ca:                pha
                        jsr     locccc1
                        lda     enemy_along,y
                        sta     $29
                        pla
loca3d4:                sta     $2c
loca3d6:                stx     $35
                        sty     $36
                        lda     #$00
                        sta     $2a
                        sta     $2b
                        ldx     #$07
loca3e2:                lda     $030a,x
                        beq     loca3fa
                        lda     $0312,x
                        cmp     $2a
                        bcc     loca3f2
                        sta     $2a
                        stx     $2b
loca3f2:                dex
                        bpl     loca3e2
                        dec     $0116
                        ldx     $2b
loca3fa:                lda     #$00
                        sta     $0312,x
                        lda     $2c
                        sta     $0302,x
                        lda     $29
                        sta     $030a,x
                        lda     $2d
                        sta     $02fa,x
                        inc     $0116
                        ldx     $35
                        ldy     $36
                        rts

loca416:                lda     $0116
                        beq     loca447
                        lda     #$00
                        sta     $0116
                        ldx     #$07
loca422:                lda     $030a,x
                        beq     loca444
                        lda     $0312,x
                        ldy     $0302,x
                        clc
                        adc     loca44e,y
                        sta     $0312,x
                        cmp     loca448,y
                        bcc     loca441
                        lda     #$00
                        sta     $030a,x
                        clv
                        bvc     loca444
loca441:                inc     $0116
loca444:                dex
                        bpl     loca422
loca447:                rts

loca448:                .byte   $10
                        .byte   $15
                        .byte   $20
                        .byte   $20
                        .byte   $20
                        .byte   $10
loca44e:                .byte   $03
                        .byte   $01
                        .byte   $03
                        .byte   $03
                        .byte   $03
                        .byte   $03

; Check player shots to see if they hit anything.

CheckAllPlayerShots:    ldx     #MAX_PLAYER_SHOTS-1
loca456:                lda     PlayerShotPositions,x
                        beq     loca45e
                        jsr     CheckPlayerShot
loca45e:                dex
                        bpl     loca456
                        rts

                        .byte   $ab

; Check to see if a player shot hit an enemy or enemy shot.  X is player
; shot number, A is player shot position.

CheckPlayerShot:        sta     $2e
                        ldy     #MAX_TOTAL_SHOTS-2      ; check enemies as well as their shots
loca467:                lda     EnemyShotPositions,y
                        beq     loca4eb
                        cmp     $2e
                        bcc     loca475
                        sbc     $2e
                        clv
                        bvc     loca47b
loca475:                lda     $2e
                        sec
                        sbc     EnemyShotPositions,y
loca47b:                cpy     #MAX_ENEMY_SHOTS        ; enemy, or enemy shot?
                        bcs     loca491
                        cmp     $a7
                        bcs     loca48e
                        lda     EnemyShotSegments,y
                        eor     PlayerShotSegments,x
                        bne     loca48e
                        jsr     loca36f
loca48e:                clv
                        bvc     loca4eb
loca491:                pha
                        sty     $38
                        lda     $027f,y                 ; enemy_type_info - 4
                        and     #ENEMY_TYPE_MASK
                        tay
                        pla
                        cmp     hit_tol_by_enm_type,y
                        bcs     loca4e9
                        cpy     #ENEMY_TYPE_FUSEBALL
                        bne     loca4c1
                        ldy     $38
                        lda     EnemyShotPositions,y    ; enemy_along - 4
                        cmp     player_along
                        beq     loca4be
                        lda     PlayerShotSegments,x
                        cmp     EnemyShotSegments,y     ; enemy_seg - 4
                        bne     loca4be
                        lda     $02c8,y                 ; more_enemy_info - 4
                        bpl     loca4be
                        jsr     loca309
loca4be:                clv
                        bvc     loca4e9
loca4c1:                ldy     $38
                        lda     $02c8,y                 ; more_enemy_info - 4
                        bpl     loca4d2
                        lda     EnemyShotSegments,y     ; enemy_seg - 4
                        cmp     $02c0,x                 ; what segment player was on when this shot was fired - but why not player_shot_seg, I don't know
                        beq     loca4e2
                        bne     loca4da
loca4d2:                lda     EnemyShotPositions,y    ; enemy_along - 4
                        cmp     player_along
                        beq     loca4e9
loca4da:                lda     EnemyShotSegments,y     ; enemy_seg - 4
                        cmp     PlayerShotSegments,x
                        bne     loca4e9
loca4e2:                stx     $37
                        jsr     loca38e
                        ldx     $37
loca4e9:                ldy     $38
loca4eb:                dey
                        bmi     loca4f1
                        jmp     loca467
loca4f1:                lda     $02f2,x
                        cmp     #$ff
                        bne     loca503
                        lda     #$00
                        sta     PlayerShotPositions,x
                        dec     PlayerShotCount
                        sta     $02f2,x
loca503:                rts
loca504:                lda     player_state
                        bpl     loca581
                        lda     PlayerShotCount
                        ora     EnemyShotCount
                        ora     $0116
                        bne     loca57e
                        ldx     MaxActiveEnemies
loca516:                lda     enemy_along,x
                        beq     loca529
                        clc
                        adc     #$0f
                        bcs     loca522
                        cmp     #$f0
loca522:                bcc     loca526
                        lda     #$00
loca526:                sta     enemy_along,x
loca529:                dex
                        bpl     loca516
                        ldx     curplayer
                        lda     p1_lives,x
                        cmp     #$01
                        bne     loca554
                        lda     #$00
                        sta     $010f
                        lda     #$01
                        sta     $0114
                        lda     $5f
                        sec
                        sbc     #$20
                        sta     $5f
                        lda     $5b
                        sbc     #$00
                        sta     $5b
                        cmp     #$fa
                        clc
                        bne     loca551
                        sec
loca551:                clv
                        bvc     loca561
loca554:                lda     player_along
                        clc
                        adc     #$0f
                        sta     player_along
                        bcs     loca561
                        cmp     #$f0
loca561:                bcc     loca57e
                        lda     #GS_Death
                        sta     gamestate
                        jsr     ClearAllShots
                        lda     NumEnemiesInTube
                        clc
                        adc     NumEnemiesOnTop
                        clc
                        adc     enemies_pending
                        cmp     #$3f
                        bcc     loca57b
                        lda     #$3f
loca57b:                sta     enemies_pending
loca57e:                clv
                        bvc     loca5ca

; Apparent anti-piracy provision.  If either checksum of the video RAM that
; holds the copyright message is wrong, and the P1 score is 17xxxx,
; increment one.byte of page zero, based on the low two digits of the score.
; See also $b1df and $b27d.

loca581:                lda     copyr_vid_cksum2
                        ora     copyr_vid_cksum1
                        beq     loca593
                        lda     #$17                ; Score must be at least 17XXXX
                        cmp     p1_score_h
                        bcs     loca593
                        ldx     p1_score_l          ; Take last 2 digits of score..,
                        inc     gamestate,x         ; ...Intentionally trash zero page!

; End apparent anti-piracy code

loca593:                lda     $0106
                        bne     loca5ca
                        lda     enemies_pending
                        ora     $0116
                        bne     loca5b5
                        ldy     MaxActiveEnemies
loca5a3:                lda     enemy_along,y
                        beq     loca5ac
                        cmp     #$11
                        bcs     loca5b5
loca5ac:                dey
                        bpl     loca5a3
                        jsr     loca5cb
                        jsr     ClearAllShots
loca5b5:                lda     zap_fire_debounce
                        and     #$60                    ; start1, start2
                        beq     loca5ca
                        bit     game_mode
                        bpl     loca5ca
                        lda     coinage_shadow
                        and     #$43
                        cmp     #$40
                        bne     loca5ca
                        jsr     loca5cb
loca5ca:                rts

; Level "over"; start zooming down tube.

loca5cb:                lda     #GS_ZoomingDown
                        sta     gamestate
                        lda     $0106
                        ora     #$80
                        sta     $0106
                        lda     #$00
                        sta     zoomspd_lsb
                        sta     along_lsb
                        sta     $5c
                        sta     $0123
                        lda     #$02
                        sta     zoomspd_msb

; Check to see if there are any spikes of nonzero height.

                        ldx     #$0f
loca5eb:                lda     lane_spike_height,x
                        beq     loca5f3
                        inc     $0123
loca5f3:                dex
                        bpl     loca5eb
                        lda     $0123
                        beq     NoSpikeWarning

; If there are any spikes, check level.

                        lda     curlevel
                        cmp     #$07                    ; On level 8 or below
                        bcs     NoSpikeWarning

; If level is low enough and there are spikes, display "AVOID SPIKES".

                        lda     #$1e                    ; time delay
                        sta     countdown_timer
                        lda     #GS_Delay               ; Start the short delay...
                        sta     gamestate
                        lda     #GS_ZoomingDown         ; ...after which we'll switch to zooming down the tube
                        sta     state_after_delay
                        lda     #$80
                        sta     $0123
NoSpikeWarning:         lda     #$ff
                        sta     zap_running
                        rts

State_HighScoreExplosion: 
                        lda     $010e
                        sta     $010d
                        ldx     #$0f
                        stx     $37
loca622:                ldx     $37
                        lda     enemy_type_info,x
                        bne     loca634
                        lda     $010e
                        beq     loca631
                        jsr     loca65b
loca631:                clv
                        bvc     loca63f
loca634:                jsr     loca6a9
                        jsr     loca721
                        lda     #$ff
                        sta     $010d
loca63f:                dec     $37
                        bpl     loca622
                        lda     timectr
                        and     #$01
                        bne     loca651
                        lda     $010e
                        beq     loca651
                        dec     $010e
loca651:                lda     $010d
                        bne     loca65a
                        lda     #GS_EnterInitials
                        sta     gamestate
loca65a:                rts

loca65b:                lda     timectr
                        and     #$00
                        bne     loca69a
                        lda     #$80
                        sta     $0263,x
                        sta     enemy_type_info,x
                        sta     $02a3,x
                        lda     pokey2_rand
                        sta     $02c3,x
                        jsr     plusminus_7
                        sta     $0323,x
                        lda     pokey1_rand
                        sta     $02e3,x
                        jsr     plusminus_7

; Why this rigamarole instead of just "lda #$00" or "and #$fe" before
; calling plusminus_7, I have no idea.

                        bmi     loca688
                        eor     #$ff
                        clc
                        adc     #$01
loca688:                sta     $0343,x
                        lda     pokey1_rand
                        sta     $0303,x
                        jsr     plusminus_7
                        sta     $0363,x
                        jsr     locccc1
loca69a:                rts

; Return with a random number in A, from 00-07 (if input A low bit is clear)
; or $f9-$00 (if input A low bit is set).

plusminus_7:            lsr     a
                        lda     pokey2_rand
                        and     #$07
                        bcc     loca6a8
                        eor     #$ff
                        clc
                        adc     #$01
loca6a8:                rts
loca6a9:                lda     $02e3,x
                        clc
                        adc     $0223,x
                        sta     $0223,x
                        lda     $0343,x
                        bmi     loca6c4
                        adc     enemy_type_info,x
                        cmp     #$f0
                        bcc     loca6c1
                        lda     #$00
loca6c1:                clv
                        bvc     loca6cd
loca6c4:                adc     enemy_type_info,x
                        cmp     #$10
                        bcs     loca6cd
                        lda     #$00
loca6cd:                tay
                        lda     $02c3,x
                        clc
                        adc     pending_seg,x
                        sta     pending_seg,x
                        lda     $0323,x
                        bmi     loca6e9
                        adc     $0263,x
                        cmp     #$f0
                        bcc     loca6e6
                        ldy     #$00
loca6e6:                clv
                        bvc     loca6f2
loca6e9:                adc     $0263,x
                        cmp     #$10
                        bcs     loca6f2
                        ldy     #$00
loca6f2:                sta     $0263,x
                        lda     $0303,x
                        clc
                        adc     pending_vid,x
                        sta     pending_vid,x
                        lda     $0363,x
                        bmi     loca710
                        adc     $02a3,x
                        cmp     #$f0
                        bcc     loca70d
                        ldy     #$00
loca70d:                clv
                        bvc     loca719
loca710:                adc     $02a3,x
                        cmp     #$10
                        bcs     loca719
                        ldy     #$00
loca719:                sta     $02a3,x
                        tya
                        sta     enemy_type_info,x
                        rts

loca721:                lda     #$fd
                        sta     $29
                        lda     $02c3,x
                        ldy     $0323,x
                        jsr     loca75d
                        sta     $02c3,x
                        tya
                        sta     $0323,x
                        lda     $02e3,x
                        ldy     $0343,x
                        jsr     loca75d
                        sta     $02e3,x
                        tya
                        sta     $0343,x
                        lda     $0303,x
                        ldy     $0363,x
                        jsr     loca75d
                        sta     $0303,x
                        tya
                        sta     $0363,x
                        lda     $29
                        bne     loca75c
                        sta     enemy_type_info,x
loca75c:                rts

loca75d:                sty     $2b
                        bit     $2b
                        bmi     loca772
                        sec
                        sbc     twenty_hex
                        sta     $2a
                        lda     $2b
                        sbc     #$00
                        bcc     loca77e
                        clv
                        bvc     loca784
loca772:                clc
                        adc     twenty_hex
                        sta     $2a
                        lda     $2b
                        adc     #$00
                        bcc     loca784
loca77e:                inc     $29
                        lda     #$00
                        sta     $2a
loca784:                tay
                        lda     $2a
                        rts

twenty_hex:             .byte   $20             ; Why not immediate mode? Why this byte?  Who knows.

loca789:                ldx     #$0f
loca78b:                lda     #$00
                        sta     enemy_type_info,x
                        dex
                        bpl     loca78b
                        lda     #$20
                        sta     $010e
                        sta     $010d
                        lda     #$04
                        sta     $01
                        lda     #$00
                        sta     $68
                        sta     $69
                        rts

; Subtract Y from A, returning (in A) the signed difference.  If the level
; is closed, do wraparound processing; if open, don't.

SubYFromAWithWrap:      sty     $2a
                        sec
                        sbc     $2a
                        sta     $2a
                        bit     open_level
                        bmi     open_level_nowrap
                        and     #$0f
                        bit     loca7bc
                        beq     open_level_nowrap
                        ora     #$f8
open_level_nowrap:      rts

loca7bc:                .byte   $08

loca7bd:                ldx     #$07
                        lda     #$00
loca7c1:                sta     $03fe,x
                        dex
                        bpl     loca7c1
                        lda     #$f0
                        sta     $0405
                        lda     #$ff
                        sta     $0115
                        rts
loca7d2:                lda     $0115
                        beq     loca830
                        lda     #$00
                        sta     $29

; There appears to be a loop beginning here
; for ($37=7;$37>=0;$37--) running from here through a827.  I'm not sure
; just what goes on inside it, yet, though.

                        ldx     #$07
                        stx     $37
loca7df:                ldx     $37
                        lda     $03fe,x
                        beq     loca7fe
                        sec
                        sbc     #$07
                        bcc     loca7ed
                        cmp     #$10
loca7ed:                bcs     loca7fb
                        ldy     $0115
                        bpl     loca7f9
                        lda     #$f0
                        clv
                        bvc     loca7fb
loca7f9:                lda     #$00
loca7fb:                clv
                        bvc     loca81e
loca7fe:                ldy     $0115
                        bpl     loca81e
                        txa
                        clc
                        adc     #$01
                        cmp     #$08
                        bcc     loca80d
                        lda     #$00
loca80d:                tay
                        lda     $03fe,y
                        beq     loca81e
                        cmp     #$d5
                        bcs     loca81c
                        lda     #$f0
                        clv
                        bvc     loca81e
loca81c:                lda     #$00
loca81e:                sta     $03fe,x
                        ora     $29
                        sta     $29
                        dec     $37
                        bpl     loca7df
                        lda     $29
                        bne     loca830
                        sta     $0115
loca830:                rts

InitSuperzapper:        lda     #$00
                        sta     zap_uses
                        sta     zap_running
                        rts

check_zap:              lda     game_mode
                        bpl     loca87c
                        lda     zap_running
                        bne     ZapIsRunning
                        lda     player_state
                        bmi     loca863             ; High bit means grabbed by pulsar or flipper
                        lda     zap_fire_new
                        and     #$08                ; zap
                        beq     loca863
                        lda     zap_uses
                        cmp     #MAX_ZAP_USES       ; 2
                        bcs     loca85d
                        inc     zap_uses
                        lda     #$01
                        sta     zap_running
loca85d:                lda     zap_fire_new
                        and     #$77                ; clear zap
                        sta     zap_fire_new
loca863:                clv
                        bvc     loca87c

ZapIsRunning:           inc     zap_running
                        ldx     zap_uses
                        lda     zap_running
                        cmp     zap_length,x
                        bcc     run_longer
                        
                        lda     #$00                ; Zap use done
                        sta     zap_running
run_longer:             jsr     DoSuperZap
loca87c:                lda     zap_fire_new
                        and     #$7f
                        sta     zap_fire_new
                        rts

zap_length:             .byte   $00                 ; Indexed by zapper use count

                        .byte   $13                 ; First zap runs a longer time
                        .byte   $05                 ; Second just long enough to kill one thing
                        .byte   $00
                        .byte   $00

DoSuperZap:             lda     zap_running
                        cmp     #$03
                        bcc     donezapwork
                        and     #$01                ; Maybe zap only kills every second frame?
                        bne     donezapwork

                        ldy     MaxActiveEnemies    
look_for_enemy_to_zap:  lda     enemy_along,y       
                        bne     found_enemy             
                        dey
                        bpl     look_for_enemy_to_zap

                        lda     #$00
                        sta     zap_running
donezapwork:            rts

found_enemy:            lda     active_enemy_info,y         ; Clear tanker payload bits so it doesn't split
                        and     #%11111100
                        sta     active_enemy_info,y
                        jmp     ZapEnemy
                        
                        .byte   $e1

; These are distances into the message table 

coinage_msgs:           .byte   ibMsgFreePlay               ; FREE PLAY
                        .byte   ibMsg1Coin2Crd              ; 1 COIN 2 PLAYS
                        .byte   ibMsg1Coin1Crd              ; 1 COIN 1 PLAY
                        .byte   ibMsg2Coin1Crd              ; 2 COINS 1 PLAY

loca8b4:                lda     #$01
                        sta     curscale
                        jsr     vapp_scale_A_0
                        ldy     #$05
                        jsr     vapp_setcolor
                        lda     game_mode
                        bmi     loca8ea
                        ldx     #ibMsgGameOver                      ; "GAME OVER"
                        lda     timectr
                        and     #$20
                        bne     loca8d8
                        ldx     #ibMsgInsCoin                       ; "INSERT COINS"
                        lda     credits
                        beq     loca8d8
                        bit     $a2
                        bmi     loca8d8
                        ldx     #ibMsgStart                         ; "PRESS START"
loca8d8:                jsr     vapp_msg
                        jsr     vapp_vcentre_1
                        lda     char_jsrtbl
                        sta     $2fa6
                        sta     $2fa8
loca8e7:                jsr     show_coin_stuff
loca8ea:                lda     #$01
                        ldy     #$00
                        jsr     show_player_stuff
                        bit     game_mode
                        bmi     loca8fe
                        lda     p2_score_l
                        ora     p2_score_m
                        ora     p2_score_h
                        clv
                        bvc     loca900
loca8fe:                lda     twoplayer
loca900:                beq     loca908
                        lda     #$01
                        tay
                        jsr     show_player_stuff
loca908:                lda     gamestate
                        cmp     #GS_Playing
                        beq     loca943
                        lda     #<endofhiscores
                        sta     $3b
                        lda     #>endofhiscores
                        sta     $3c
                        ldx     HiScoreOffset
                        jsr     loca9d7

; Checksum the code which displays the copyright message; see $aace
; See also $c8f5.

                        ldy     #$0a
                        lda     #$a7
loca920:                eor     locaace,y
                        dey
                        bpl     loca920
                        sta     copyr_disp_cksum1
                        ldx     hsinitidx
                        lda     #$02
                        sta     $38
loca930:                ldy     $38
                        lda     hs_initials_1,y
                        asl     a
                        tay
                        lda     ltr_jsrtbl,y
                        sta     $2f60,x
                        inx
                        inx
                        dec     $38
                        bpl     loca930
loca943:                lda     #>video_data
                        ldx     #<video_data
                        jsr     vapp_vjsr_AX
                        lda     $0123
                        bpl     loca954
                        ldx     #ibMsgAvoidSpk               ; "AVOID SPIKES"
                        jsr     vapp_msg
loca954:                lda     gamestate
                        cmp     #GS_ZoomOntoNew
                        bne     loca97c
                        lda     game_mode
                        bpl     loca97c
                        ldx     curplayer
                        lda     p1_startchoice,x
                        beq     loca972
                        ldx     #ibMsgBonusSpc               ; "BONUS "
                        jsr     vapp_msg
                        ldy     curplayer
                        ldx     p1_startchoice,y
                        jsr     vapp_startbonus
loca972:                ldx     #ibMsgRecharge               ; "SUPERZAPPER RECHARGE"
                        jsr     vapp_msg
                        ldx     #ibMsgLevelNS                ; "LEVEL"
                        jsr     vapp_msg
loca97c:                rts

; Indexed by player number; see $a9ce

p1scorepos:             .byte   p1_score_h
p2scorepos              .byte   p2_score_h

; On entry, A=1 and Y=player number

show_player_stuff:      ldx     gamestate
                        cpx     #GS_Playing
                        sty     $2b
                        cpy     curplayer
                        bne     loca98f
                        bit     game_mode
                        bpl     loca98f
                        lda     #$00
loca98f:                ora     #$70
                        ldx     ScaleOffset,y
                        sta     video_data,x
                        ldx     ShipsLeftOffset,y
                        lda     p1_lives,y
                        sta     $38
                        beq     loca9a7
                        cpy     curplayer
                        bne     loca9a7
                        dec     $38
loca9a7:                ldy     #$01
loca9a9:                lda     $3284
                        cpy     $38
                        bcc     loca9b5
                        beq     loca9b5
                        lda     $3286
loca9b5:                sta     video_data,x
                        inx
                        inx
                        iny
                        cpy     #$07
                        bcc     loca9a9
                        ldy     $2b
                        lda     gamestate
                        cmp     #$04
                        bne     loca9cb
                        cpy     curplayer
                        bne     loca9fb
loca9cb:                ldx     ScoresOffset,y
                        lda     p1scorepos,y
                        sta     $3b
                        lda     #$00
                        sta     $3c
loca9d7:                ldy     #$02
                        sty     $2a
                        sec
loca9dc:                php
                        ldy     #$00
                        lda     ($3b),y
                        lsr     a
                        lsr     a
                        lsr     a
                        lsr     a
                        plp
                        jsr     loca9fc
                        lda     $2a
                        bne     loca9ee
                        clc
loca9ee:                ldy     #$00
                        lda     ($3b),y
                        jsr     loca9fc
                        dec     $3b
                        dec     $2a
                        bpl     loca9dc
loca9fb:                rts
loca9fc:                and     #$0f
                        tay
                        beq     locaa02
                        clc
locaa02:                bcs     locaa05
                        iny
locaa05:                php
                        tya
                        asl     a
                        tay
                        lda     char_jsrtbl,y
                        sta     video_data,x
                        inx
                        inx
                        plp
                        rts

; Sets up the header for the text at the top of the screen.  Plugs in the
; level number, but none of the other variable pieces.

locaa13:                ldx     twoplayer
                        bit     game_mode
                        bmi     locaa23
                        lda     p2_score_l
                        ora     p2_score_m
                        ora     p2_score_h
                        beq     locaa23
                        ldx     #$01
locaa23:                lda     #<video_data
                        sta     vidptr_l
                        lda     #>video_data
                        sta     vidptr_h
                        lda     hdr_template_len,x
                        tay
                        sec
                        adc     vidptr_l
                        pha
locaa33:                lda     hdr_template,y
                        sta     (vidptr_l),y
                        dey
                        bne     locaa33
                        lda     hdr_template,y
                        sta     (vidptr_l),y
                        lda     game_mode
                        bpl     locaa54

; 2fa6 holds the vjsr for the tens digit of the level number

                        lda     #>video_data
                        sta     vidptr_h

                        ; BUGBUG the length can never be more than enough to wrap past FF of that memory page (from 2fa6 past 3000)

                        lda     #(levelnumoffset-hdr_template+<video_data)  
                        sta     vidptr_l
                        lda     curlevel
                        clc
                        adc     #$01
                        jsr     vapp_2dig_bin
locaa54:                pla
                        sta     vidptr_l
                        jmp     vapp_rts
locaa5a:                ldx     #ibMsgPlay                   ; "PLAY"
                        jsr     vapp_msg
                        jmp     locaa69
locaa62:                lda     #$30
                        ldx     #ibMsgGameOver               ; "GAME OVER"
                        jsr     vapp_msg_at_y
locaa69:                jsr     show_plyno
                        jmp     loca8e7
locaa6f:                jsr     loca8b4
                        lda     #$00
                        ldx     #ibMsgStart                 ; "PRESS START"
                        jmp     vapp_msg_at_y
locaa79:                lda     #$00
                        ldx     #ibMsg2CrdMin               ; "2 CREDIT MINIMUM"
                        jsr     vapp_msg_at_y
                        lda     timectr
                        and     #$1f
                        cmp     #$10
                        bcs     locaa8f
                        lda     #$e0
                        ldx     #ibMsgInsCoin               ; "INSERT COINS"
                        jsr     vapp_msg_at_y
locaa8f:                jmp     loca8b4

; show PLAYER and current player number

show_plyno:             ldx     #ibMsgPlayer                ; "PLAYER "
                        jsr     vapp_msg
locaa97:                lda     #$00
                        jsr     vapp_setscale
                        ldx     curplayer
locaa9e:                inx
                        stx     $61
                        lda     #$61
                        ldy     #$01
                        jmp     vapp_multdig_y_a

show_coin_stuff:        lda     coinage_shadow
                        and     #$03 ; coinage
                        tax
                        lda     coinage_msgs,x
                        tax
                        jsr     vapp_msg
                        dec     $016e
                        lda     optsw2_shadow
                        and     #$01                        ; 2-credit minimum
                        beq     locaacb
                        lda     timectr
                        and     #$20 ; flash
                        bne     locaacb
                        ldx     #ibMsg2CrdMin               ; "2 CREDIT MINIMUM"
                        jsr     vapp_msg
                        clv
                        bvc     locaace
locaacb:                jsr     locaeca
locaace:                ldx     #ibMsgAtari                 ; "(c) MCMLXXX ATARI"
                        jsr     vapp_msg
                        ldx     #ibMsgCredits               ; "CREDITS "
                        jsr     vapp_msg
                        lda     credits
                        cmp     #MAX_CREDITS                ; normally 40
                        bcc     locaae2
                        lda     #MAX_CREDITS                ; normally 40
                        sta     credits
locaae2:                jsr     vapp_2dig_bin
                        lda     uncredited
                        beq     locaaf2
                        lda     locaaf3+1
                        ldx     locaaf3
                        jsr     vapp_vjsr_AX
locaaf2:                rts

locaaf3:                .word   $0325c ; 1/2

; Converts number in accumulator (binary) to BCD, storing two-digit BCD
; in $29 (and leaving it in $2c) on return.  Discards the hundreds digit.

bin_to_bcd:             sed
                        sta     $29
                        lda     #$00
                        sta     $2c
                        ldy     #$07
locaafe:                asl     $29
                        lda     $2c
                        adc     $2c
                        sta     $2c
                        dey
                        bpl     locaafe
                        cld
                        sta     $29
                        rts

; $20 $80 = vcentre (why $20? who knows.)

vapp_vcentre_1:         lda     #$20
                        ldx     #$80
                        jmp     vapp_A_X_Y_0
vapp_msg:               lda     aMsgsColorAndYPos+1,x
vapp_msg_at_y:          stx     $35
                        sta     $2b
                        ldy     $35
                        lda     (strtbl),y
                        sta     $3b
                        iny
                        lda     (strtbl),y
                        sta     $3c

; If we're displaying the copyright message, save the location in video RAM
; where it's displayed, for the checksumming code at $b1df and $b27d.

                        cpx     #ibMsgAtari         ; "(c) MCMLXXX ATARI"
                        bne     locab32
                        lda     vidptr_l
                        sta     copyr_vid_loc
                        lda     vidptr_h
                        sta     copyr_vid_loc+1
locab32:                ldy     #$00
                        lda     ($3b),y
                        sta     $2a
                        jsr     vapp_vcentre_1
locab3b:                lda     #$00
                        sta     draw_z
                        lda     #$01
                        sta     curscale
                        jsr     vapp_scale_A_0
                        lda     $2a
                        ldx     $2b
                        jsr     vapp_ldraw_A_X
                        ldy     $35
                        lda     (strtbl),y
                        sta     $3b
                        iny
                        lda     (strtbl),y
                        sta     $3c
                        ldx     $35
                        lda     aMsgsColorAndYPos,x
                        pha
                        lsr     a
                        lsr     a
                        lsr     a
                        lsr     a
                        tay
                        jsr     vapp_setcolor
                        pla
                        and     #$0f
                        jsr     vapp_setscale
                        ldy     #$01
                        lda     #$00
                        sta     $2a
locab72:                lda     ($3b),y
                        sta     $2b
                        and     #$7f
                        iny
                        sty     $2c
                        tax
                        lda     char_jsrtbl,x
                        ldy     $2a
                        sta     (vidptr_l),y
                        iny
                        lda     char_jsrtbl+1,x
                        sta     (vidptr_l),y
                        iny
                        sty     $2a
                        ldy     $2c
                        bit     $2b
                        bpl     locab72
                        ldy     $2a
                        dey
                        jmp     inc_vi.word

; Append a message.  X holds message number, A holds delta-x from current
; position (delta-y is zero).

locab98:                stx     $35
                        sta     $2a
                        lda     #$00
                        sta     $2b
                        beq     locab3b

; Initialize the high-score list if either of the please-init bits is set.

maybe_init_hs:          jsr     check_settings
                        lda     hs_initflag
                        and     #$03
                        beq     locac07

; Initialize the low five scores on the high-score list, and if the
; please-init bits are set, the upper three as well.

init_hs:                jsr     check_settings
                        lda     #$08
                        sta     $0100
                        lda     hs_score_1
                        ora     hs_score_1+1
                        ora     hs_score_1+2
                        bne     locabc2
                        jsr     hs_needs_init
locabc2:                ldx     #$17
                        lda     hs_initflag
                        and     #$01
                        bne     locabcd
                        ldx     #$0e
locabcd:                lda     DefaultScoreInitials,x
                        sta     hs_initials_8,x
                        dex
                        bpl     locabcd
                        ldx     #$17
                        lda     hs_initflag
                        and     #$02
                        bne     locabe1
                        ldx     #$0e
locabe1:                lda     #$01
                        sta     hs_score_8,x
                        dex
                        bpl     locabe1
                        lda     hs_initflag
                        and     #$03
                        beq     locabff
                        lda     optsw2_shadow
                        and     #$f8
                        sta     life_settings
                        lda     diff_bits
                        and     #$03 ; difficulty
                        sta     diff_settings
locabff:                lda     hs_initflag
                        and     #$fc
                        sta     hs_initflag
locac07:                rts

; Default high score initials.

;.bytes are reversed compared to the order they're displayed in.

DefaultScoreInitials:   .byte    7,  4,  1      ; BEH
                        .byte   15,  9, 12      ; MJP
                        .byte   11,  3, 18      ; SDL   ??? (Sam Lee)
                        .byte   19,  5,  3      ; DFT   (David Frank Theurer, programmer)
                        .byte    7, 15, 12      ; MPH   ??? (Morgan Hoff)
                        .byte   17, 17, 17      ; RRR
                        .byte   18,  4,  3      ; DES   ??? (Doug Snyder - hardware, or Daver Sherman or Stubben or Storie)
                        .byte    3,  9,  4      ; EJD   ??? (Eric Durgrey - technician)

check_settings:         jsr     read_optsws
                        lda     optsw2_shadow
                        and     #$f8                ; initial lives & points per life
                        cmp     life_settings
                        bne     locac34
                        lda     diff_bits
                        and     #$03                ; difficulty
                        cmp     diff_settings
locac34:                beq     locac3e

hs_needs_init:          lda     hs_initflag
                        ora     #$03
                        sta     hs_initflag
locac3e:                rts

state_10:               lda     game_mode
                        and     #$bf                ; mask out second high bit
                        sta     game_mode
                        lda     coinage_shadow
                        and     #$43
                        cmp     #$40
                        bne     locac50
                        jsr     locca62
locac50:                jsr     locddfb
                        lda     #$00
                        sta     $0601
                        ldx     twoplayer
                        beq     locac5e
                        ldx     #$03
locac5e:                lda     p1_score_h,x
                        sta     $2c
                        lda     p1_score_m,x
                        sta     $2d
                        lda     p1_score_l,x
                        sta     $2e
                        txa
                        and     #$01
                        sta     $36
                        lda     #$00
                        sta     $2b
                        lda     #$1a
                        sta     $2a
                        sta     $29
                        lda     #$00
                        sta     hs_timer
                        ldy     #$fd
locac80:                lda     $0620,y
                        cmp     $2c
                        bne     locac9b
                        lda     $061f,y
                        cmp     $2d
                        bne     locac9b
                        cpy     #$52
                        bcc     locac9a
                        lda     $061e,y
                        cmp     $2e
                        clv
                        bvc     locac9b
locac9a:                sec
locac9b:                bcs     locacec
locac9d:                cpy     #$e8
                        bcc     locacbf
                        lda     $29
                        ldx     $051e,y
                        sta     $051e,y
                        stx     $29
                        lda     $2a
                        ldx     $051f,y
                        sta     $051f,y
                        stx     $2a
                        lda     $2b
                        ldx     $0520,y
                        sta     $0520,y
                        stx     $2b
locacbf:                lda     $2d
                        ldx     $061f,y
                        sta     $061f,y
                        stx     $2d
                        lda     $2c
                        ldx     $0620,y
                        sta     $0620,y
                        stx     $2c
                        cpy     #$52
                        bcc     locace1
                        lda     $2e
                        ldx     $061e,y
                        sta     $061e,y
                        stx     $2e
locace1:                cpy     #$55
                        bcc     locace6
                        dey
locace6:                dey
                        dey
                        bne     locac9d
                        ldy     #$02
locacec:                inc     hs_timer
                        cpy     #$55
                        bcc     locacf4
                        dey
locacf4:                dey
                        dey
                        bne     locac80
                        ldx     $36
                        lda     hs_timer
                        sta     $0600,x
                        dex
                        bmi     locad06
                        jmp     locac5e
locad06:                lda     $0601
                        cmp     $0600
                        bcc     locad15
                        cmp     #$63
                        bcs     locad15
                        inc     $0601
locad15:                lda     curplayer
                        eor     #$01
                        asl     a
                        asl     a
                        ora     curplayer
                        adc     #$05
                        sta     $0603
locad22:                ldy     #$14
                        lda     $0603
                        beq     locad6b
                        and     #$03
                        sta     curplayer
                        dec     curplayer
                        lsr     $0603
                        lsr     $0603
                        ldx     curplayer
                        lda     $0600,x
                        beq     locad68
                        cmp     #$09
                        bcs     locad68
                        asl     a
                        clc
                        adc     $0600,x
                        eor     #$ff
                        sec
                        sbc     #$e5
                        sta     hs_whichletter
                        jsr     locca48

; Entering high score?

                        lda     #$60                    ; 60 second window for entering high score
                        sta     hs_timer
                        lda     #$00
                        sta     zap_fire_new
                        sta     $50
                        lda     #$02
                        sta     $0604
                        jsr     loca789 
                        ldy     #GS_HighScoreExplosion
                        sty     gamestate
                        rts
locad68:                jmp     locad22
locad6b:                sty     gamestate
                        rts

; High score entry

State_EnterInitials:    lda     #$06
                        sta     $01
                        lda     timectr
                        and     #$1f
                        bne     locad82
                        dec     hs_timer
                        bne     locad82
                        ldy     #GS_Unknown14
                        sty     gamestate
                        rts
locad82:                ldx     hs_whichletter
                        lda     hs_initials_8,x
                        jsr     track_spinner

; enforce 0..$1a (0-26, A through.ds) range

                        tay
                        bpl     locad93
                        lda     #$1a
                        clv
                        bvc     locad99
locad93:                cmp     #$1b
                        bcc     locad99
                        lda     #$00
locad99:                ldx     hs_whichletter
                        sta     hs_initials_8,x
                        lda     zap_fire_new
                        and     #$18
                        tay
                        lda     zap_fire_new
                        and     #$67
                        sta     zap_fire_new
                        tya
                        beq     locadcd
                        dec     hs_whichletter
                        dec     $0604
                        bpl     locadc7
                        ldx     curplayer
                        lda     $0600,x
                        cmp     #$04
                        bcs     locadc1
                        jsr     locddf7
locadc1:                jsr     locad22
                        clv
                        bvc     locadcd
locadc7:                dex
                        lda     #$00
                        sta     hs_initials_8,x
locadcd:                rts

; Track the spinner, maybe?  Input value in A, return value in A is either
; unchanged, one higher, or one lower.  Adds $50 into $51 and clears $50.

track_spinner:          pha
                        lda     $50
                        asl     a
                        asl     a
                        asl     a
                        clc
                        adc     $51
                        sta     $51
                        pla
                        ldy     $50
                        bmi     locade3
                        adc     #$00
                        clv
                        bvc     locade5
locade3:                adc     #$ff
locade5:                ldy     #$00
                        sty     $50
                        rts
locadea:                jsr     loca8b4
                        lda     #$c0
                        ldx     #ibMsgPlayer            ; "PLAYER "
                        jsr     vapp_msg_at_y
                        dec     $016e
                        jsr     locaa97
                        ldx     #ibMsgInitials          ; "ENTER YOUR INITIALS"
                        jsr     vapp_msg
                        lda     #$a6
                        ldx     #ibMsgSpinKnob          ; "SPIN KNOB TO CHANGE"
                        jsr     vapp_msg_at_y
                        lda     #$9c
                        ldx     #ibMsgPressFire         ; "PRESS FIRE TO SELECT"
                        jsr     vapp_msg_at_y
                        ldx     #ibMsgAtari             ; "(c) MCMLXXX ATARI"
                        jsr     vapp_msg
                        lda     hs_whichletter
                        sec
                        sbc     $0604
                        jmp     locae4e
locae1c:                jsr     loca8b4
                        sei
                        lda     pokey1_rand
                        ldy     pokey1_rand
                        sty     $29
                        lsr     a
                        lsr     a
                        lsr     a
                        lsr     a
                        eor     $29
                        sta     $29
                        lda     pokey2_rand
                        ldy     pokey2_rand
                        cli
                        eor     $29
                        and     #$f0
                        eor     $29
                        sta     $29
                        tya
                        asl     a
                        asl     a
                        asl     a
                        asl     a
                        eor     $29
                        sta     $011f
                        jsr     locaf26
                        lda     #$ff
locae4e:                sta     $63
                        ldx     #ibMsgHiScores              ; "HIGH SCORES"
                        jsr     vapp_msg
                        lda     #$01
                        sta     $61
                        jsr     vapp_setscale
                        lda     #$28
                        sta     $2c
                        ldx     #$15
                        stx     $37
locae64:                jsr     vapp_vcentre_1
                        lda     #$00
                        sta     draw_z
                        lda     $2c
                        tax
                        sec
                        sbc     #$0a
                        sta     $2c
                        lda     #$d0
                        jsr     vapp_ldraw_A_X
                        ldy     #$07
                        lda     $63
                        cmp     $37
                        bne     locae82
                        ldy     #$00
locae82:                jsr     vapp_setcolor
                        lda     #$61
                        ldy     #$01
                        jsr     vapp_multdig_y_a
                        lda     #$a0
                        jsr     locb56a
                        lda     #$00
                        sta     draw_z
                        tax
                        lda     #$08
                        jsr     vapp_ldraw_A_X
                        inc     $61
                        lda     $37
                        jsr     locaef8
                        ldx     #$00
                        lda     #$08
                        jsr     vapp_ldraw_A_X
                        ldx     $37
                        lda     hs_score_8,x
                        sta     $56
                        lda     $0707,x
                        sta     $57
                        lda     $0708,x
                        sta     $58
                        lda     #$56
                        ldy     #$03
                        jsr     vapp_multdig_y_a
                        dec     $37
                        dec     $37
                        dec     $37
                        bpl     locae64
                        rts
locaeca:                lda     bonus_life_each
                        beq     locaee3
                        sta     $58
                        ldx     #ibMsgBonusEv               ; "BONUS EVERY "
                        jsr     vapp_msg
                        lda     #$00
                        sta     $56
                        sta     $57
                        lda     #$56
                        ldy     #$03
                        jsr     vapp_multdig_y_a
locaee3:                clc
                        ldy     #$10
                        lda     #$85
locaee8:                adc     xposMsgAtari,y              ; "(c) MCMLXXX ATARI" data
                        dey
                        bpl     locaee8
                        sta     copyr_cksum
                        rts
                        lda     hs_whichletter
                        sec
                        sbc     $0604
locaef8:                clc
                        adc     #$02
                        sta     $38
                        ldy     #$00
                        lda     #$02
                        sta     $39
locaf03:                ldx     $38
                        lda     hs_initials_8,x
                        cmp     #$1e
                        bcc     locaf0e
                        lda     #$1a
locaf0e:                asl     a
                        tax
                        lda     ltr_jsrtbl,x
                        sta     (vidptr_l),y
                        iny
                        lda     ltr_jsrtbl+1,x
                        sta     (vidptr_l),y
                        iny
                        dec     $38
                        dec     $39
                        bpl     locaf03
                        dey
                        jmp     inc_vi.word
locaf26:                lda     $0600
                        ora     $0601
                        beq     locaf6e
                        ldx     #ibMsgRanking                ; RANKING FROM 1 TO
                        jsr     vapp_msg
                        lda     #$63
                        jsr     locaf71
                        ldx     #$00
                        jsr     locaf3f
                        ldx     #$01
locaf3f:                lda     $0600,x
                        beq     locaf6e
                        pha
                        stx     $2e
                        ldy     #$03
                        jsr     vapp_setcolor
                        jsr     vapp_vcentre_1
                        lda     #$d0
                        ldy     $2e
                        ldx     locaf6f,y
                        jsr     vapp_ldraw_A_X
                        pla
                        jsr     locaf71
                        lda     #$a0
                        jsr     locb56a
                        lda     #$10 ; +16
                        ldx     #$04                ; "PLAYER "
                        jsr     locab98
                        ldx     $2e
                        jsr     locaa9e
locaf6e:                rts
locaf6f:                cpy     #$b0
locaf71:                cmp     #$63
                        bcc     vapp_2dig_bin
                        lda     #$63

; Displays a 2-digit number on the screen
                        
vapp_2dig_bin:          jsr     bin_to_bcd
                        lda     #$29
                        ldy     #$01
                        jmp     vapp_multdig_y_a
locaf81:                jsr     locca48
                        dec     $016e
                        ldy     #$03
                        jsr     vapp_setcolor
                        lda     #$01
                        sta     curscale
                        jsr     vapp_scale_A_0
                        ldx     #ibMsgAtari                  ; "(c) MCMLXXX ATARI"
                        lda     #$60                         ; Y coord (normally $92 for this msg)
                        jsr     vapp_msg_at_y
                        jsr     show_plyno
                        ldx     #$07
                        stx     $37
locafa1:                ldy     $37
                        ldx     selfrate_msgs,y
                        jsr     vapp_msg
                        dec     $37
                        bpl     locafa1
                        lda     player_seg
                        sec
                        sbc     $7b
                        bpl     locafbc
                        dec     $7b
                        dec     $7c
                        clv
                        bvc     locafe1
locafbc:                bne     locafcb
                        dec     $7c
                        dec     $7b
                        bpl     locafc8
                        inc     $7b
                        inc     $7c
locafc8:                clv
                        bvc     locafe1
locafcb:                lda     $7c
                        cmp     $0127
                        beq     locafd4
                        bcs     locafe1
locafd4:                sec
                        sbc     player_seg
                        bne     locafdb
                        clc
locafdb:                bcs     locafe1
                        inc     $7b
                        inc     $7c
locafe1:                lda     $7c
                        sta     $3a
                        ldx     #$04
                        stx     $37
locafe9:                ldy     #$05
                        jsr     vapp_setcolor
                        lda     #$00
                        sta     draw_z
                        jsr     vapp_vcentre_1
                        ldx     #$d8
                        ldy     $37
                        lda     locb096,y
                        clc
                        adc     #$f8
                        jsr     vapp_ldraw_A_X
                        ldx     $3a
                        ldy     startlevtbl,x
                        cpy     #MAX_LEVEL
                        bcs     locb042
                        iny
                        tya
                        jsr     vapp_2dig_bin
                        ldy     #$03
                        jsr     vapp_setcolor
                        jsr     vapp_vcentre_1
                        ldx     #$ba
                        ldy     $37
                        lda     locb096,y
                        clc
                        adc     #$ec
                        jsr     vapp_ldraw_A_X
                        ldx     $3a
                        jsr     vapp_startbonus
                        jsr     vapp_vcentre_1
                        ldx     #$cc
                        ldy     $37
                        lda     locb096,y
                        clc
                        adc     #$00
                        jsr     vapp_ldraw_A_X
                        ldx     $3a
                        lda     startlevtbl,x
                        jsr     locc4e1
locb042:                dec     $3a
                        dec     $37
                        bpl     locafe9
                        lda     #$00
                        sta     draw_z
                        jsr     vapp_vcentre_1
                        ldx     #ibMsgTime                      ; "TIME"
                        jsr     vapp_msg
                        lda     #$04
                        ldy     #$01
                        jsr     vapp_multdig_y_a
                        ldy     #$00
                        jsr     vapp_setcolor
                        jsr     vapp_vcentre_1
                        ldx     #$b8
                        jsr     locb0ab
                        sec
                        sbc     $7b
                        tay
                        lda     locb096,y
                        sec
                        sbc     #$16
                        jsr     vapp_ldraw_A_X
                        lda     #$e0
                        sta     draw_z
                        ldx     #$00
                        stx     $38
                        ldy     #$03
                        sty     $37
locb081:                ldy     $38
                        lda     levelselboxpts,y
                        tax
                        iny
                        lda     levelselboxpts,y
                        iny
                        sty     $38
                        jsr     vapp_ldraw_A_X
                        dec     $37
                        bpl     locb081
                        rts

; These appear to be X offsets of the tube pictures on the starting-level
; selection display.  See $aff9.

locb096:                .byte   $be
                        .byte   $e3
                        .byte   $09
                        .byte   $30
                        .byte   $58

                        ; Byte offsets into the message table

selfrate_msgs:          .byte   ibMsgRateSelf       ; "RATE YOURSELF"
                        .byte   ibMsgSpinKnob       ; "SPIN KNOB TO CHANGE"
                        .byte   ibMsgPressFire      ; "PRESS FIRE TO SELECT"
                        .byte   ibMsgNovice         ; "NOVICE"
                        .byte   ibMsgExpert         ; "EXPERT"
                        .byte   ibMsgLevel          ; "LEVEL"
                        .byte   ibMsgHole           ; "HOLE"
                        .byte   ibMsgBonus          ; "BONUS"

; Used at $b083 to draw the box around the selected level, on the starting
; level selection screen.

levelselboxpts:         .byte  00,  38          ; x=+0,y=+38
                        .byte  40,  00          ; x=+40,y=+0
                        .byte   0, -38          ; x=+0,y=-38
                        .byte -40,   0          ; x=-40,y=+0

locb0ab:                lda     player_seg
                        jsr     track_spinner
                        tay
                        bpl     locb0b9
                        lda     #$00
                        clv
                        bvc     locb0c1
locb0b9:                cmp     $0127
                        bcc     locb0c1
                        lda     $0127
locb0c1:                sta     player_seg
                        tay
                        rts

vapp_startbonus:        txa
                        jsr     ld_startbonus
                        lda     #$29
                        ldy     #$03
                        jmp     vapp_multdig_y_a

vapp_setcolor:          cpy     curcolor
                        beq     locb0dc
                        sty     curcolor
                        lda     #$08
                        jmp     vapp_sclstat_A_Y
locb0dc:                rts

vapp_setscale:          cmp     curscale
                        beq     locb0e6
                        sta     curscale
                        jmp     vapp_scale_A_0
locb0e6:                rts

state_1a:               lda     #GS_Delay
                        sta     gamestate
                        lda     #GS_GameStartup
                        sta     state_after_delay
                        lda     #$df
                        sta     countdown_timer
                        lda     #$12
                        sta     unknown_state
                        lda     #$19
                        sta     $014e
                        lda     #$18
                        sta     $014d
                        rts

locb102:                lda     #$34
                        ldx     #$aa
                        jsr     locb15a
                        lda     $014e
                        cmp     #$a0
                        bcs     locb115
                        adc     #$14
                        sta     $014e
locb115:                cmp     #$50
                        bcc     locb130
                        lda     $014d
                        clc
                        adc     #$08
                        sta     $014d
                        cmp     $014e
                        bcc     locb130
                        lda     #$a0
                        sta     $014d
                        lda     #$14
                        sta     $01
locb130:                rts
locb131:                lda     #$3f
                        ldx     #$4e
                        jsr     locb15a
                        lda     $014d
                        cmp     #$30
                        bcc     locb144
                        sbc     #$01
                        sta     $014d
locb144:                cmp     #$80
                        bcs     locb159
                        lda     $014e
                        sec
                        sbc     #$01
                        cmp     $014d
                        bcs     locb156
                        lda     $014d
locb156:                sta     $014e
locb159:                rts
locb15a:                sta     $57
                        stx     $56
                        lda     $014d
                        sta     $37
                        dec     $016e
locb166:                lda     $37
                        asl     a
                        asl     a
                        and     #$7f
                        tay
                        lda     $37
                        lsr     a
                        lsr     a
                        lsr     a
                        lsr     a
                        lsr     a
                        jsr     vapp_scale_A_Y
                        lda     $37
                        cmp     $014d
                        bne     locb183
                        lda     #$00
                        clv
                        bvc     locb18f
locb183:                lsr     a
                        lsr     a
                        lsr     a
                        nop
                        and     #$07
                        cmp     #$07
                        bne     locb18f
                        lda     #$03
locb18f:                tay
                        lda     #$68
                        jsr     vapp_sclstat_A_Y
                        lda     $57
                        ldx     $56
                        jsr     vapp_vjsr_AX
                        lda     $37
                        clc
                        adc     #$02
                        sta     $37
                        cmp     $014e
                        bcc     locb166
                        ldx     #ibMsgAtari                     ; ATARI
                        lda     #$d0
                        jsr     vapp_msg_at_y
                        
                        VECTOR_DATA_1 = $3FF2

                        lda     #>VECTOR_DATA_1
                        ldx     #<VECTOR_DATA_1
                        jmp     vapp_vjsr_AX
locb1b6:                jsr     locc1c3
                        lda     vecram
                        cmp     loccec6
                        bne     locb1c7
                        lda     $0133
                        bne     locb1c7
                        rts
locb1c7:                lda     $01
                        cmp     #$00
                        beq     locb209
                        lda     #$00
                        jsr     db_init_vi.word
                        jsr     locb332
                        bcs     locb1f5
                        jsr     locb20d
                        lda     $016e
                        beq     locb1f5

; Anti-piracy provision.  Checksum the video RAM that holds the copyright
; message.  See also $b27d and $a581.

                        ldy     #$27
                        lda     #$0e
                        sec
locb1e4:                sbc     (copyr_vid_loc),y
                        dey
                        bpl     locb1e4
                        tay
                        beq     locb1ee
                        eor     #$e5
locb1ee:                beq     locb1f2
                        eor     #$29
locb1f2:                sta     copyr_vid_cksum2
locb1f5:                lda     #$00
                        jsr     dblbuf_done
                        lda     loccec4
                        sta     vecram
                        lda     loccec5
                        sta     vecram+1
                        clv
                        bvc     locb20c
locb209:                jmp     locb230
locb20c:                rts

locb20d:                ldx     $01
                        lda     locb218+1,x
                        pha
                        lda     locb218,x
                        pha
                        rts

; Jump table, used just above, at $b20f

locb218:                .word   locb230-1               
                        .word   locd804-1
                        .word   locb8ba-1
                        .word   locadea-1
                        .word   locaf81-1
                        .word   locae1c-1
                        .word   locaa62-1
                        .word   locaa5a-1
                        .word   locaa6f-1
                        .word   locb102-1
                        .word   locb131-1
                        .word   locaa79-1

locb230:                lda     #$07
                        jsr     db_init_vi.word
                        jsr     draw_player
                        lda     #$07
                        jsr     dblbuf_done
                        lda     #$04
                        jsr     db_init_vi.word
                        jsr     draw_shots
                        lda     #$04
                        jsr     dblbuf_done
                        lda     #$03
                        jsr     db_init_vi.word
                        jsr     draw_enemies
                        lda     #$03
                        jsr     dblbuf_done
                        lda     #$06
                        jsr     db_init_vi.word
                        jsr     draw_explosions
                        lda     #$06
                        jsr     dblbuf_done
                        lda     #$05
                        jsr     db_init_vi.word
                        jsr     draw_pending
                        lda     #$05
                        jsr     dblbuf_done
                        lda     #$00
                        jsr     db_init_vi.word
                        jsr     loca8b4
                        lda     game_mode
                        bmi     locb28a

; Anti-piracy provision.  When not playing, checksum the video RAM that
; holds the copyright message.  See also $b1df and $a581.

                        lda     #$f2
                        clc
                        ldy     #$27
locb282:                adc     (copyr_vid_loc),y
                        dey
                        bpl     locb282
                        sta     copyr_vid_cksum1

; End anti-piracy provision.

locb28a:                lda     #$00
                        jsr     dblbuf_done
                        jsr     locb367
                        lda     #$01
                        jsr     db_init_vi.word
                        jsr     locc5c2
                        lda     #$01
                        jsr     dblbuf_done
                        lda     #$08
                        jsr     db_init_vi.word
                        jsr     locc54d
                        lda     #$08
                        jsr     dblbuf_done
                        lda     #$00
                        sta     $0114
                        lda     loccec2
                        sta     vecram
                        lda     loccec3
                        sta     vecram+1
                        rts

db_init_vi.word:        tax
                        asl     a
                        tay
                        lda     dblbuf_flg,x
                        bne     locb2cf
                        ldx     dblbuf_addr_B,y
                        lda     dblbuf_addr_B+1,y
                        clv
                        bvc     locb2d5
locb2cf:                ldx     dblbuf_addr_A,y
                        lda     dblbuf_addr_A+1,y
locb2d5:                stx     vidptr_l
                        sta     vidptr_h
                        lda     #$00
                        sta     $a9
                        rts
locb2de:                tax
                        asl     a
                        tay
                        lda     dblbuf_flg,x
                        bne     locb2ef
                        ldx     dblbuf_addr_A,y
                        lda     dblbuf_addr_A+1,y
                        clv
                        bvc     locb2f5
locb2ef:                ldx     dblbuf_addr_B,y
                        lda     dblbuf_addr_B+1,y
locb2f5:                stx     $3b
                        sta     $3c
                        lda     #$00
                        sta     $a9
                        rts

dblbuf_done:            pha
                        jsr     vapp_rts
                        pla
                        tax
                        asl     a
                        tay
                        lda     dblbuf_vjsr_loc,y
                        sta     $3b
                        lda     dblbuf_vjsr_loc+1,y
                        sta     $3c
                        lda     dblbuf_flg,x
                        eor     #$01
                        sta     dblbuf_flg,x
                        bne     locb323
                        lda     dblbuf_vjmp_C,y
                        ldx     dblbuf_vjmp_C+1,y
                        clv
                        bvc     locb329
locb323:                lda     dblbuf_vjmp_D,y
                        ldx     dblbuf_vjmp_D+1,y
locb329:                ldy     #$00
                        sta     ($3b),y
                        txa
                        iny
                        sta     ($3b),y
                        rts
locb332:                lda     loccec4
                        cmp     vecram
                        beq     locb33f
                        sta     vecram
                        sec
                        rts
locb33f:                lda     dblbuf_flg
                        bne     locb349
                        ldx     #$02
                        clv
                        bvc     locb34b
locb349:                ldx     #$08
locb34b:                lda     dblbuf_vjmp_C,x
                        ldy     #$00
                        sty     $016e
                        sta     (vidptr_l),y
                        iny
                        lda     dblbuf_vjmp_C+1,x
                        sta     (vidptr_l),y
                        lda     dblbuf_addr_A,x
                        sta     vidptr_l
                        lda     dblbuf_addr_A+1,x
                        sta     vidptr_h
                        clc
                        rts
locb367:                lda     $0114
                        beq     locb379
                        lda     #$02
                        jsr     db_init_vi.word
                        jsr     locc30d
                        lda     #$02
                        jsr     dblbuf_done
locb379:                lda     #$02
                        jsr     locb2de
                        lda     #$00
                        ldx     #$0f
locb382:                sta     $0425,x
                        dex
                        bpl     locb382
                        lda     $0106
                        bmi     locb3d6
                        ldx     MaxActiveEnemies
locb390:                lda     enemy_along,x
                        beq     locb3d3
                        ldy     #$00
                        lda     enemy_type_info,x
                        and     #$07
                        cmp     #$01
                        bne     locb3d3
                        iny
                        sty     $29
                        lda     enemy_type_info,x
                        and     #$80
                        bne     locb3c6
                        lda     pulsing
                        bmi     locb3bb
                        lda     enemy_along,x
                        cmp     lethal_distance
                        bcs     locb3bb
                        inc     $29
                        inc     $29
locb3bb:                lda     $29
                        ldy     more_enemy_info,x
                        ora     $0425,y
                        sta     $0425,y
locb3c6:                ldy     enemy_seg,x
                        lda     $29
                        ora     #$80
                        ora     $0425,y
                        sta     $0425,y
locb3d3:                dex
                        bpl     locb390
locb3d6:                lda     #$06
                        ldy     zap_running
                        beq     locb3e9
                        bmi     locb3e9
                        lda     timectr
                        and     #$07
                        cmp     #$07
                        bne     locb3e9
                        lda     #$01
locb3e9:                sta     $29
                        ldy     #$ff
                        ldx     #$ff
                        stx     $2c
                        lda     player_along
                        beq     locb401
                        lda     player_state
                        bmi     locb401
                        ldx     player_seg
                        ldy     player_state
locb401:                stx     $2a
                        sty     $2b
                        lda     $0124
                        bmi     locb412
                        and     #$0e
                        lsr     a
                        sta     $2c
                        dec     $0124
locb412:                ldx     #$0f
locb414:                ldy     #$06
                        lda     $0425,x
                        beq     locb427
                        and     #$02
                        beq     locb424
                        lda     timectr
                        and     #$01
                        tay
locb424:                clv
                        bvc     locb44b
locb427:                cpx     $2a
                        beq     locb42d
                        cpx     $2b
locb42d:                bne     locb434
                        ldy     #$01
                        clv
                        bvc     locb44b
locb434:                lda     $0124
                        bmi     locb449
                        txa
                        clc
                        adc     $2c
                        and     #$07
                        cmp     #$07
                        bne     locb445
                        lda     #$03
locb445:                tay
                        clv
                        bvc     locb44b
locb449:                ldy     $29
locb44b:                tya
                        ldy     locb476,x
                        sta     ($3b),y
                        dex
                        bpl     locb414
                        ldx     #$0f
                        bit     open_level
                        bpl     locb45c
                        dex
locb45c:                ldy     #$c0
                        lda     $0425,x
                        bpl     locb465
                        ldy     #$00
locb465:                sty     $58
                        ldy     locb487,x
                        lda     ($b0),y
                        and     #$1f
                        ora     $58
                        sta     ($b0),y
                        dex
                        bpl     locb45c
                        rts

; Function unknown.  Used at $b44c.

locb476:                .byte   $a8
                        .byte   $9c
                        .byte   $92
                        .byte   $86
                        .byte   $7c
                        .byte   $70
                        .byte   $66
                        .byte   $5a
                        .byte   $50
                        .byte   $44
                        .byte   $3a
                        .byte   $2e
                        .byte   $24
                        .byte   $18
                        .byte   $0e
                        .byte   $02
                        .byte   $b2
locb487:                .byte   $3b
                        .byte   $37
                        .byte   $33
                        .byte   $2f
                        .byte   $2b
                        .byte   $27
                        .byte   $23
                        .byte   $1f
                        .byte   $1b
                        .byte   $17
                        .byte   $13
                        .byte   $0f
                        .byte   $0b
                        .byte   $07
                        .byte   $03
                        .byte   $3f
                        .byte   $1d

draw_pending:           ldy     #red
                        sty     curcolor
                        lda     #$08
                        jsr     vapp_sclstat_A_Y
                        ldx     #$66
                        jsr     vapp_to_X_
                        lda     #$12
                        sta     $56
                        ldx     #$3f
                        stx     $37
                        ldy     #$00
locb4b0:                ldx     $37
                        lda     pending_vid,x
                        bne     locb4ba
                        jmp     locb549
locb4ba:                cmp     #$50
                        bcc     locb4c0
                        dec     $37

; Construct a vscale instruction based on the pending_vid,x value

locb4c0:                pha
                        and     #$3f
                        sta     (vidptr_l),y
                        pla
                        rol     a
                        rol     a
                        rol     a
                        and     #$03
                        clc
                        adc     #$01
                        ora     #$70
                        iny
                        sta     (vidptr_l),y
                        iny
                        lda     pending_seg,x
                        tax

; Construct a long draw (z=0) subtracting screen centre from $03?a,x values

                        lda     $038a,x
                        sec
                        sbc     $68
                        sta     $63
                        sta     (vidptr_l),y
                        iny
                        lda     $037a,x
                        sbc     $69
                        sta     $64
                        and     #$1f
                        sta     (vidptr_l),y
                        iny
                        lda     $036a,x
                        sta     $61
                        sta     (vidptr_l),y
                        iny
                        lda     $035a,x
                        sta     $62
                        and     #$1f
                        sta     (vidptr_l),y
                        iny

; Append a long draw, x=0 y=0 z=5

                        lda     #$00
                        sta     (vidptr_l),y
                        iny
                        sta     (vidptr_l),y
                        iny
                        sta     (vidptr_l),y
                        lda     #$a0
                        iny
                        sta     (vidptr_l),y
                        iny

; Append the negative of the long draw we constructed above.

                        lda     $63
                        eor     #$ff
                        clc
                        adc     #$01
                        sta     (vidptr_l),y
                        iny
                        lda     $64
                        eor     #$ff
                        adc     #$00
                        and     #$1f
                        sta     (vidptr_l),y
                        iny
                        lda     $61
                        eor     #$ff
                        clc
                        adc     #$01
                        sta     (vidptr_l),y
                        iny
                        lda     $62
                        eor     #$ff
                        adc     #$00
                        and     #$1f
                        sta     (vidptr_l),y
                        iny
                        cpy     #$f0
                        bcc     locb545
                        dey
                        jsr     inc_vi.word
                        ldy     #$00
locb545:                dec     $56
                        bmi     locb550
locb549:                dec     $37
                        bmi     locb550
                        jmp     locb4b0
locb550:                tya
                        beq     locb557
                        dey
                        jsr     inc_vi.word

; Anti-piracy code.  If the copyright string has been tampered with,
; and player 1's level is over 10, set $53 to #$7a.
; I'm not sure what this does, but it can't be good. :-)

locb557:                lda     copyr_cksum
                        beq     locb565
                        lda     p1_level
                        cmp     #$0a
                        bcc     locb565
                        lda     #$7a
                        sta     $53

; End anti-piracy code.

locb565:                lda     #$01
                        jmp     vapp_scale_A_0
locb56a:                pha
                        ldy     #$00
                        tya
                        sta     (vidptr_l),y
                        iny
                        sta     (vidptr_l),y
                        iny
                        sta     (vidptr_l),y
                        iny
                        pla
                        sta     (vidptr_l),y
                        lda     #$04
                        clc
                        adc     vidptr_l
                        sta     vidptr_l
                        bcc     locb585
                        inc     vidptr_h
locb585:                rts

draw_player:            lda     #$01
                        sta     curcolor
                        lda     player_along
                        beq     locb5ac
                        cmp     #$f0
                        bcs     locb5ac
                        sta     $57
                        sta     $2f
                        lda     player_state
                        cmp     #$81
                        beq     locb5ac
                        ldy     player_seg
                        lda     $51
                        lsr     a
                        and     #$07
                        clc
                        adc     #$01
                        jsr     draw_linegfx
locb5ac:                rts

draw_enemies:           lda     $0106
                        bmi     locb5d6
                        ldx     #$06
                        stx     $37
locb5b6:                ldx     $37
                        lda     enemy_along,x
                        beq     locb5d2
                        sta     $57
                        lda     enemy_type_info,x
                        and     #$18
                        lsr     a
                        lsr     a
                        lsr     a
                        sta     $55
                        lda     enemy_type_info,x
                        and     #$07
                        asl     a
                        jsr     DrawEnemyByType
locb5d2:                dec     $37
                        bpl     locb5b6
locb5d6:                rts

; Look up the address in the EnemyDrawVec, push it onto the stack, and do an RTS
; to "jump" to that address

DrawEnemyByType:        tay                                 ; Twice enemy type val (hence 0 = flipper, 2 = pulsar, etc)
                        lda     EnemyDrawVecs+1,y
                        pha
                        lda     EnemyDrawVecs,y
                        pha
                        rts

; Indexed by enemy type.  Draws enemy.

EnemyDrawVecs:          .word   DrawFlipper  - 1            ; flipper               
                        .word   DrawPulsar   - 1            ; pulsar
                        .word   DrawTanker   - 1            ; tanker
                        .word   DrawSpiker   - 1            ; spiker
                        .word   DrawFuseball - 1            ; fuseball

; Code to draw a flipper.

DrawFlipper:            lda     #$03
                        sta     curcolor
                        lda     enemy_type_info,x
                        bmi     locb602
                        ldy     enemy_seg,x
                        ldx     $55
                        lda     locb60b,x
                        jsr     draw_linegfx
                        clv
                        bvc     +
locb602:                jsr     locb634
                        ldy     #$00
                        jsr     locbdcb
+                       rts

; Graphic numbers for flippers.  Indexed by $18 bits of enemy_type_info value.

locb60b:                .byte   $00                         ; BUGBUG a table of zeros in ROM seems a tad odd to me.  I bet there used to
                        .byte   $00                         ;        be animation frames for flippers but they got removed at some point
                        .byte   $00
                        .byte   $00

; Code to draw a tanker.

DrawTanker:             lda     active_enemy_info,x
                        and     #$03
                        tay
                        lda     TankerIconTable,y
                        ldy     enemy_seg,x
                        jmp     graphic_at_mid

; Graphic number for tankers.  Indexed by contents value (active_enemy_info bits $03).

TankerIconTable:        .byte   $1a                         ; Table of graphics that show contents of tanker
                        .byte   $1a                         ; Split into two flipper center icon
                        .byte   $4a                         ; Split into two pulsars center icon
                        .byte   $4c                         ; Split into two fuseballs center icona

; Code to draw a spiker.

DrawSpiker:             ldy     enemy_seg,x
                        lda     timectr
                        and     #$03
                        asl     a
                        clc
                        adc     #$12
                        jmp     graphic_at_mid

; Not used; a table version of the value computed by $b629-$b62b.

.if !OPTIMIZE
                        .byte   $12
                        .byte   $14
                        .byte   $16
                        .byte   $18
.endif

locb634:                lda     $57
                        sta     $2f
                        ldy     enemy_seg,x
                        lda     tube_x,y
                        sta     $56
                        lda     tube_y,y
                        sta     $58
                        lda     more_enemy_info,x
                        and     #$0f
                        tay
                        lda     $56
                        eor     #$80
                        clc
                        adc     locb68b,y
                        bvc     locb65e
                        bpl     locb65c
                        lda     #$7f
                        clv
                        bvc     locb65e
locb65c:                lda     #$80
locb65e:                eor     #$80
                        sta     $2e
                        lda     $58
                        eor     #$80
                        clc
                        adc     locb687,y
                        bvc     locb675
                        bpl     locb673
                        lda     #$7f
                        clv
                        bvc     locb675
locb673:                lda     #$80
locb675:                eor     #$80
                        sta     $30
                        ldy     curtube
                        lda     lev_fscale,y
                        sta     fscale
                        lda     lev_fscale2,y
                        sta     fscale+1
                        rts

; Used - apparently overlappingly - at $b667 and $b68b.

; Maybe this is a sine wave, and the overlapping is sin-vs-cos?
locb687:                .byte   $00
                        .byte   $10
                        .byte   $1f
                        .byte   $28
locb68b:                .byte   $2c
                        .byte   $28
                        .byte   $1f
                        .byte   $10
                        .byte   $00
                        .byte   $f0
                        .byte   $e1
                        .byte   $d8
                        .byte   $d4
                        .byte   $d8
                        .byte   $e1
                        .byte   $f0
                        .byte   $00
                        .byte   $10
                        .byte   $1f
                        .byte   $28

; Code to draw a fuseball.

DrawFuseball:           lda     enemy_along,x
                        sta     $57
                        ldy     enemy_seg,x
                        lda     tube_x,y
                        sta     $56
                        lda     tube_y,y
                        sta     $58
                        lda     more_enemy_info,x
                        bpl     locb6d5
                        tya
                        clc
                        adc     #$01
                        and     #$0f
                        tay
                        lda     tube_x,y
                        sec
                        sbc     $56
                        jsr     locb6fa
                        clc
                        adc     $56
                        sta     $56
                        lda     tube_y,y
                        sec
                        sbc     $58
                        jsr     locb6fa
                        clc
                        adc     $58
                        sta     $58
locb6d5:                jsr     locc098
                        ldx     #$61
                        jsr     vapp_to_X_
                        lda     #$00
                        sta     $a9
                        jsr     locbd3e
                        sty     $a9
                        lda     timectr
                        and     #$03
                        asl     a
                        clc
                        adc     #$4e
                        tay
                        ldx     graphic_table+1,y
                        lda     graphic_table,y
                        ldy     $a9
                        jmp     vapp_A_X
locb6fa:                sta     $29
                        lda     more_enemy_info,x
                        and     #$07
                        sta     $2c
                        stx     $2b
                        ldx     #$02
                        lda     #$00
locb709:                lsr     $2c
                        bcc     locb710
                        clc
                        adc     $29
locb710:                asl     a
                        php
                        ror     a
                        plp
                        ror     a
                        dex
                        bpl     locb709
                        ldx     $2b
                        rts

; Code to draw a pulsar.

DrawPulsar:             lda     #$04
                        ldy     pulsing
                        bmi     locb724
                        lda     #$00
locb724:                sta     curcolor
                        lda     pulsing
                        clc
                        adc     #$40
                        lsr     a
                        lsr     a
                        lsr     a
                        lsr     a
                        cmp     #$05
                        bcc     locb736
                        lda     #$00
locb736:                tay
                        lda     PulsarAnimFrames,y
                        sta     $29
                        lda     enemy_type_info,x
                        bmi     locb74c
                        ldy     enemy_seg,x
                        lda     $29
                        jsr     draw_linegfx
                        clv
                        bvc     locb754
locb74c:                jsr     locb634
                        ldy     $29
                        jsr     locbdcb
locb754:                rts

PulsarAnimFrames:       .byte   $0d
                        .byte   $0c
                        .byte   $0b
                        .byte   $0a
                        .byte   $09
                        .byte   $09

draw_shots:             ldx     #MAX_TOTAL_SHOTS-1
                        stx     $37
locb75f:                ldx     $37
                        lda     PlayerShotPositions,x
                        beq     locb781
                        sta     $57
                        sta     $2f
                        cpx     #MAX_PLAYER_SHOTS                           ; player or enemy shot?
                        ldy     PlayerShotSegments,x
                        bcs     locb776
                        lda     #MAX_PLAYER_SHOTS                           ; BUGBUG was #$08 but is that the player shot limit for sure?
                        clv
                        bvc     locb77e
locb776:                lda     timectr
                        asl     a
                        and     #$06
                        clc
                        adc     #$20
locb77e:                jsr     graphic_at_mid
locb781:                dec     $37
                        bpl     locb75f
                        ldy     #$04
                        lda     PlayerShotCount
                        cmp     #MAX_PLAYER_SHOTS-2                         ; BUGBUG was #$06 but I'd expect #$07
                        bcc     locb796
                        ldy     #MAX_TOTAL_SHOTS-1
                        cmp     #MAX_PLAYER_SHOTS
                        bcc     locb796
                        ldy     #$0c
locb796:                sty     $0808
                        rts

draw_explosions:        ldy     #$00
                        sty     curcolor
                        ldx     #$07
                        stx     $37
locb7a2:                ldx     $37
                        lda     $030a,x
                        beq     locb7d2
                        sta     $57
                        lda     $02fa,x
                        sta     $29
                        ldy     $0302,x
                        cpy     #$01
                        bne     locb7bd
                        jsr     vapp_mid_graphic
                        clv
                        bvc     locb7d2
locb7bd:                lda     $0312,x
                        lsr     a
                        and     #$fe
                        cpy     #$02
                        bcc     locb7c9
                        lda     #$00
locb7c9:                clc
                        adc     locb7e5,y
                        ldy     $29
                        jsr     graphic_at_mid
locb7d2:                dec     $37
                        bpl     locb7a2
                        lda     $0720
                        beq     locb7e4
                        lda     curlevel
                        cmp     #$0d
                        bcc     locb7e4
                        sta     $01ff
locb7e4:                rts

; Table of graphics.  See $b7ca

locb7e5:                .byte   $00
                        .byte   $00
                        .byte   $5a
                        .byte   $58

                        lsr     $1c,x
vapp_mid_graphic:       ldy     $29
                        lda     mid_x,y
                        sta     $56
                        lda     mid_y,y
                        sta     $58
                        jsr     locc098
                        ldx     #$61
                        jsr     vapp_to_X_
                        ldx     $013b
                        dec     $013c
                        bne     locb811
                        inx
                        stx     $013b
                        lda     locb82a,x
                        sta     $013c
locb811:                ldy     locb83d,x
                        bmi     locb819
                        jsr     locb84e
locb819:                lda     $013b
                        asl     a
                        clc
                        adc     #$28                    ; hit-by-shot explosion
                        tay
                        ldx     graphic_table+1,y
                        lda     graphic_table,y
                        jmp     vapp_A_X_Y_0

locb82a:                .byte   $02
                        .byte   $02
                        .byte   $02
                        .byte   $02
                        .byte   $02
                        .byte   $04
                        .byte   $03
                        .byte   $02
                        .byte   $01
                        .byte   $20
                        .byte   $03
                        .byte   $03
                        .byte   $03
                        .byte   $03
                        .byte   $03
                        .byte   $03
                        .byte   $03
                        .byte   $3b
                        .byte   $b8
locb83d:                .byte   $00
                        .byte   $02
                        .byte   $02
                        .byte   $02
                        .byte   $02
                        .byte   $02
                        .byte   $02
                        .byte   $02
                        .byte   $04
                        .byte   $06
                        .byte   $ff
                        .byte   $ff
                        .byte   $ff
                        .byte   $ff
                        .byte   $ff
                        .byte   $ff
                        .byte   $ff

locb84e:                lda     locb857+1,y
                        pha
                        lda     locb857,y
                        pha
                        rts

locb857:                .word   locb85f-1           ; jump table for following functions
                        .word   locb875-1
                        .word   locb888-1
                        .word   locb896-1

locb85f:                lda     #$0c
                        sta     $080b
                        sta     $24
                        lda     #$04
                        sta     $080a
                        sta     $23
                        lda     #$00
                        sta     $22
                        sta     $0809
                        rts

locb875:                ldy     $22
                        ldx     #$02
-                       lda     $22,x
                        pha
                        sty     $22,x
                        tya
                        sta     $0809,x
                        pla
                        tay
                        dex
                        bpl     -
                        rts

locb888:                jsr     SetLevelColors
                        lda     #$7f
                        sta     $0139
                        lda     #$04
                        sta     $013a
                        rts

locb896:                lda     $0139
                        sta     $2ffc
                        lda     $013a
                        ora     #$70
                        sta     $2ffd
                        lda     #$c0
                        sta     $2fff
                        lda     $0139
                        sec
                        sbc     #$20
                        bpl     +
                        and     #$7f
                        dec     $013a
+                       sta     $0139
                        rts

; $3ff2 = code to move to extreme corners of the screen, not drawing.

locb8ba:                lda     #>VECTOR_DATA_1
                        ldx     #<VECTOR_DATA_1
                        jsr     vapp_vjsr_AX
                        lda     #$00
                        sta     $6a
                        sta     $6b
                        sta     $6c
                        sta     $6d
                        sta     player_along
                        sta     $68
                        sta     $69
                        lda     #$e0
                        sta     $5f
                        lda     #$ff
                        sta     $5b
                        jsr     locb967
                        sta     $77
                        stx     $76
                        ldx     #$0f
                        stx     $37
locb8e5:                ldx     $37
                        lda     enemy_type_info,x
                        beq     locb935
                        sta     $57
                        lda     $0263,x
                        sta     $56
                        lda     $02a3,x
                        sta     $58
                        jsr     locc098
                        lda     #$00
                        sta     draw_z
                        jsr     locb944
                        jsr     locc3ba
                        lda     #$a0
                        jsr     locb56a
                        jsr     locb944
                        ldx     #$61
                        jsr     locc772
                        jsr     locb955
                        jsr     vapp_scale_A_Y
                        lda     $37
                        and     #$07
                        cmp     #$07
                        bne     locb922
                        lda     #$00
locb922:                tay
                        sty     curcolor
                        lda     #$08
                        jsr     vapp_sclstat_A_Y
                        lda     #$00
                        jsr     vapp_sclstat_A_73
                        jsr     locb967
                        jsr     vapp_vjsr_AX
locb935:                dec     $37
                        bpl     locb8e5
                        jsr     locb944
                        lda     #$01
                        jsr     vapp_scale_A_0
                        jsr     vapp_rts
locb944:                ldx     vidptr_l
                        ldy     vidptr_h
                        lda     $76
                        sta     vidptr_l
                        stx     $76
                        lda     $77
                        sta     vidptr_h
                        sty     $77
                        rts
locb955:                lda     $57
                        lsr     a
                        lsr     a
                        lsr     a
                        lsr     a
                        ldy     #$00
locb95d:                iny
                        lsr     a
                        bne     locb95d
                        clc
                        adc     #$02
                        ldy     #$00
                        rts
locb967:                lda     dblbuf_flg
                        beq     locb975
                        lda     locce6e+1
                        ldx     locce6e
                        clv
                        bvc     locb97b
locb975:                lda     locce86+1
                        ldx     locce86
locb97b:                rts

; Two-dimensional arrays are indexed by level number first: x[L][P], etc.
;
; x,y:                coordinates of points
; angle:              angles of sectors (0-15 represents 0-360 degrees)
; remap:              order levels are encountered in
; scale:              (reciprocal of) scale of level
; y3d:                3-D Y offset - "camera height"
; y2d,y2db:           2-D Y offset - signed; y2db is high.byte, y2d low
; open:               $00 if level is closed, $ff if open
; fscale,fscale2:     (reciprocal of) flipper scale - fscale2<<7 | fscale

lev_x:                  .byte   $F0,$E7,$CF,$AA,$80,$56,$31,$19,$10,$19,$31,$56,$80,$AA,$CF,$E7 ; x[16][16]
                        .byte   $F0,$F0,$F0,$B8,$80,$48,$10,$10,$10,$10,$10,$48,$80,$B8,$F0,$F0
                        .byte   $F0,$F0,$B8,$B8,$80,$48,$48,$10,$10,$10,$48,$48,$80,$B8,$B8,$F0
                        .byte   $EC,$D5,$B1,$90,$70,$4F,$2B,$14,$14,$2B,$4F,$70,$90,$B1,$D5,$EC
                        .byte   $F0,$C0,$A0,$94,$6C,$60,$40,$10,$10,$40,$60,$6C,$94,$A0,$C0,$F0
                        .byte   $D9,$C2,$AC,$97,$80,$69,$52,$3C,$27,$10,$35,$5A,$80,$A6,$CA,$F0
                        .byte   $EA,$E0,$9C,$80,$64,$20,$16,$50,$16,$20,$64,$80,$9C,$E0,$EA,$B0
                        .byte   $10,$1E,$2C,$3A,$48,$56,$64,$70,$90,$9E,$AC,$BA,$C8,$D6,$E4,$F0
                        .byte   $10,$1E,$2D,$3C,$4B,$5A,$69,$78,$87,$96,$A5,$B4,$C3,$D2,$E1,$F0
                        .byte   $10,$10,$10,$10,$16,$29,$46,$69,$97,$BA,$D7,$EA,$F0,$F0,$F0,$F0
                        .byte   $10,$24,$30,$36,$3E,$49,$5A,$75,$94,$A4,$AC,$BA,$DA,$E2,$EA,$F0
                        .byte   $80,$70,$48,$20,$10,$20,$48,$70,$80,$90,$B8,$E0,$F0,$E0,$B8,$90
                        .byte   $DA,$A4,$87,$80,$79,$5C,$26,$10,$10,$20,$48,$80,$B8,$E0,$F0,$F0
                        .byte   $10,$10,$30,$30,$50,$50,$70,$70,$90,$90,$B0,$B0,$D0,$D0,$F0,$F0
                        .byte   $B0,$80,$50,$47,$18,$30,$18,$47,$50,$80,$B0,$B9,$E8,$D4,$E8,$B9
                        .byte   $10,$1E,$21,$28,$3C,$55,$66,$73,$8D,$9A,$AB,$C4,$D8,$DF,$E2,$F0
                        
lev_y:                  .byte   $80,$AA,$CF,$E7,$F0,$E7,$CF,$AA,$80,$56,$31,$19,$10,$19,$31,$56 ; y[16][16]
                        .byte   $80,$B8,$F0,$F0,$F0,$F0,$F0,$B8,$80,$48,$10,$10,$10,$10,$10,$48
                        .byte   $80,$B8,$B8,$F0,$F0,$F0,$B8,$B8,$80,$48,$48,$10,$10,$10,$48,$48
                        .byte   $94,$B0,$B8,$A7,$A7,$B8,$B0,$94,$6C,$50,$48,$59,$59,$48,$50,$6C
                        .byte   $96,$A3,$C5,$F0,$F0,$C5,$A3,$96,$6A,$5D,$3B,$10,$10,$3B,$5D,$6A
                        .byte   $3D,$6A,$97,$C4,$F0,$C4,$97,$6A,$3D,$10,$10,$10,$10,$10,$10,$10
                        .byte   $A0,$E0,$EA,$B0,$EA,$E0,$A0,$80,$60,$20,$16,$50,$16,$20,$60,$80
                        .byte   $F0,$D0,$B0,$90,$70,$50,$30,$10,$10,$30,$50,$70,$90,$B0,$D0,$F0
                        .byte   $40,$40,$40,$40,$40,$40,$40,$40,$40,$40,$40,$40,$40,$40,$40,$40
                        .byte   $F0,$CB,$A6,$80,$5C,$39,$20,$12,$12,$20,$39,$5C,$80,$A6,$CB,$F0
                        .byte   $C0,$A6,$8A,$6A,$4A,$2F,$14,$24,$20,$39,$59,$75,$72,$90,$B0,$D0
                        .byte   $80,$57,$48,$57,$80,$A9,$BA,$A9,$80,$57,$48,$57,$80,$A9,$BA,$A9
                        .byte   $E4,$E8,$B7,$80,$B7,$E8,$E4,$B2,$7A,$47,$20,$10,$20,$47,$7A,$B2
                        .byte   $90,$70,$70,$50,$50,$30,$30,$10,$10,$30,$30,$50,$50,$70,$70,$90
                        .byte   $E6,$D0,$E6,$B9,$AE,$80,$52,$47,$14,$30,$14,$47,$52,$80,$AE,$B9
                        .byte   $7E,$6A,$51,$3A,$2C,$2C,$38,$4E,$4E,$38,$2C,$2C,$3A,$51,$6A,$7E
                                    
lev_angle:              .byte   $05,$06,$07,$08,$09,$0A,$0B,$0C,$0D,$0E,$0F,$00,$01,$02,$03,$04 ; angle[16][16]
                        .byte   $04,$04,$08,$08,$08,$08,$0C,$0C,$0C,$0C,$00,$00,$00,$00,$04,$04
                        .byte   $04,$08,$04,$08,$08,$0C,$08,$0C,$0C,$00,$0C,$00,$00,$04,$00,$04
                        .byte   $06,$07,$09,$08,$07,$09,$0A,$0C,$0E,$0F,$01,$00,$0F,$01,$02,$04
                        .byte   $07,$06,$05,$08,$0B,$0A,$09,$0C,$0F,$0E,$0D,$00,$03,$02,$01,$04
                        .byte   $05,$05,$05,$05,$0B,$0B,$0B,$0B,$0B,$00,$00,$00,$00,$00,$00,$05
                        .byte   $04,$08,$0B,$05,$08,$0C,$0E,$09,$0C,$00,$03,$0D,$00,$04,$07,$02
                        .byte   $0D,$0D,$0D,$0D,$0D,$0D,$0D,$00,$03,$03,$03,$03,$03,$03,$03,$00
                        .byte   $00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00
                        .byte   $0C,$0C,$0C,$0D,$0E,$0F,$0F,$00,$01,$01,$02,$03,$04,$04,$04,$00
                        .byte   $0E,$0D,$0C,$0D,$0D,$0D,$01,$0F,$02,$03,$03,$00,$03,$03,$03,$00
                        .byte   $0B,$09,$07,$05,$03,$01,$0F,$0D,$0D,$0F,$01,$03,$05,$07,$09,$0B
                        .byte   $08,$0B,$0C,$04,$05,$08,$0B,$0C,$0D,$0E,$0F,$01,$02,$03,$04,$05
                        .byte   $0C,$00,$0C,$00,$0C,$00,$0C,$00,$04,$00,$04,$00,$04,$00,$04,$00
                        .byte   $0A,$06,$0C,$08,$0E,$0A,$00,$0C,$02,$0E,$04,$00,$06,$02,$08,$04
                        .byte   $0E,$0C,$0D,$0E,$00,$02,$02,$00,$0E,$0E,$00,$02,$03,$04,$02,$00

lev_remap:              .byte   $00,$01,$02,$03,$04,$05,$06,$07,$0D,$09,$08,$0C,$0E,$0F,$0A,$0B ; remap[16]

lev_scale:              .byte   $18,$1C,$18,$0F,$18,$18,$18,$18,$0A,$18,$10,$0F,$18,$0C,$14,$0A ; scale[16]

lev_y3d:                .byte   $50,$50,$50,$68,$50,$50,$68,$B0,$A0,$50,$90,$80,$20,$B0,$60,$A0 ; y3d[16]

lev_y2d:                .byte   $40,$20,$40,$80,$40,$40,$70,$60,$00,$20,$40,$00,$A0,$40,$40,$00 ; y2d[16]

lev_y2db:               .byte   $FF,$FF,$FF,$FF,$FF,$FF,$FF,$00,$01,$FF,$00,$00,$FE,$01,$FF,$01 ; Y2DB[16]

lev_open:               .byte   $00,$00,$00,$00,$00,$00,$00,$FF,$FF,$FF,$FF,$00,$00,$FF,$00,$FF ; open[16]

lev_fscale:             .byte   $00,$00,$60,$40,$00,$00,$48,$40,$50,$28,$50,$00,$00,$50,$00,$40 ; fscale[16]

lev_fscale2:            .byte   $04,$04,$03,$04,$04,$04,$03,$04,$05,$04,$04,$04,$04,$04,$04,$05 ; fscale2[16]

                        .byte   $3E 

graphic_at_mid:         sta     $55
                        lda     mid_x,y
                        sta     $56
                        lda     mid_y,y
                        sta     $58
locbd09:                jsr     locc098
                        ldx     #$61
                        jsr     vapp_to_X_
                        lda     #$00
                        sta     $a9
                        jsr     locbd3e
                        lda     $78
                        eor     #$07
                        asl     a
                        cmp     #$0a
                        bcs     locbd23
                        lda     #$0a
locbd23:                asl     a
                        asl     a
                        asl     a
                        asl     a
                        sta     (vidptr_l),y
                        iny
                        lda     #$60                ; BUGBUG is this the low byte of 2f60 (video_data)
                        sta     (vidptr_l),y
                        iny
                        sty     $a9
                        ldy     $55
                        ldx     graphic_table+1,y
                        lda     graphic_table,y
                        ldy     $a9
                        jmp     vapp_A_X
locbd3e:                lda     $57
                        cmp     #$10
                        bcc     locbd8c
                        sec
                        sbc     $5f
                        sta     mb_w_15
                        lda     #$00
                        sbc     $5b
                        sta     mb_w_16
                        lda     #$18
                        sta     mb_w_0c
                        lda     $a0
                        sta     mb_w_0e
                        sta     mb_w_14
locbd5e:                bit     eactl_mbst
                        bmi     locbd5e
                        lda     mb_rd_l
                        sta     $79
                        lda     mb_rd_h
                        sta     $7a
                        ldx     #$0f
                        stx     mb_w_0c
                        sec
                        sbc     #$01
                        bne     locbd79
                        lda     #$01
locbd79:                ldx     #$00
locbd7b:                inx
                        asl     $79
                        rol     a
                        bcc     locbd7b
                        lsr     a
                        eor     #$7f
                        clc
                        adc     #$01
                        tay
                        txa
                        clv
                        bvc     locbd90
locbd8c:                lda     #$01
                        ldy     #$00
locbd90:                sta     $78
                        pha
                        tya
                        ldy     $a9
                        sta     (vidptr_l),y
                        iny
                        pla
                        ora     #$70
                        sta     (vidptr_l),y
                        iny
                        rts

; Draw a rotatable/scalable graphic.
; Y = segment number
; A = graphic to draw
; $57 = position along tube

draw_linegfx:           sta     $36
                        lda     tube_x,y
                        sta     $56
                        lda     tube_y,y
                        sta     $58
                        lda     $57
                        sta     $2f
                        tya
                        clc
                        adc     #$01
                        and     #$0f
                        tax
                        lda     tube_x,x
                        sta     $2e
                        lda     tube_y,x
                        sta     $30
                        lda     #$00
                        sta     fscale
                        lda     #$04
                        sta     fscale+1
                        ldy     $36
locbdcb:                lda     $5b
                        bmi     locbdd6
                        lda     $57
                        cmp     $5f
                        bcs     locbdd6
                        rts
locbdd6:                lda     locbfb6,y
                        sta     rgr_pt_inx
                        lda     locbfc4,y
                        sta     $38
                        ldy     curcolor
                        lda     #$08
                        jsr     vapp_sclstat_A_Y
                        jsr     locc098
                        ldx     #$61
                        jsr     vapp_to_X_
                        lda     $2e
                        sta     $56
                        lda     $2f
                        sta     $57
                        lda     $30
                        sta     $58
                        jsr     locc098
                        ldy     fscale
                        lda     fscale+1
                        jsr     vapp_scale_A_Y
                        lda     $61
                        sec
                        sbc     $6a
                        sta     $79
                        lda     $62
                        sbc     $6b
                        sta     $9b
                        bmi     locbe1d
                        beq     locbe1a
                        lda     #$ff
                        sta     $79
locbe1a:                clv
                        bvc     locbe33
locbe1d:                cmp     #$ff
                        beq     locbe26
                        lda     #$ff
                        clv
                        bvc     locbe31
locbe26:                lda     $79
                        eor     #$ff
                        clc
                        adc     #$01
                        bcc     locbe31
                        lda     #$ff
locbe31:                sta     $79
locbe33:                lda     $63
                        sec
                        sbc     $6c
                        sta     $89
                        lda     $64
                        sbc     $6d
                        sta     $9d
                        bmi     locbe4b
                        beq     locbe48
                        lda     #$ff
                        sta     $89
locbe48:                clv
                        bvc     locbe5d
locbe4b:                cmp     #$ff
                        beq     locbe54
                        lda     #$ff
                        clv
                        bvc     locbe5b
locbe54:                lda     $89
                        eor     #$ff
                        clc
                        adc     #$01
locbe5b:                sta     $89
locbe5d:                lda     #$00
                        sta     $82
                        sta     $92
                        lda     $79
                        asl     a
                        rol     $82
                        sta     $7a
                        asl     a
                        sta     $7c
                        lda     $82
                        rol     a
                        sta     $84
                        lda     $7c
                        adc     $79
                        sta     $7d
                        lda     $84
                        adc     #$00
                        sta     $85
                        lda     $7a
                        adc     $79
                        sta     $7b
                        lda     $82
                        adc     #$00
                        sta     $83
                        sta     $86
                        lda     $7b
                        asl     a
                        sta     $7e
                        rol     $86
                        adc     $79
                        sta     $7f
                        lda     $86
                        adc     #$00
                        sta     $87
                        lda     $89
                        asl     a
                        rol     $92
                        sta     $8a
                        asl     a
                        sta     $8c
                        lda     $92
                        rol     a
                        sta     $94
                        lda     $8c
                        adc     $89
                        sta     $8d
                        lda     $94
                        adc     #$00
                        sta     $95
                        lda     $8a
                        adc     $89
                        sta     $8b
                        lda     $92
                        adc     #$00
                        sta     $93
                        sta     $96
                        lda     $8b
                        asl     a
                        sta     $8e
                        rol     $96
                        adc     $89
                        sta     $8f
                        lda     $96
                        adc     #$00
                        sta     $97
                        ldy     #$00
                        sty     $a9

; Top of loop for points in claw

locbedb:                ldy     $38
                        lda     ClawDrawData+1,y
                        cmp     #$01
                        bne     locbee6
                        lda     #$c0
locbee6:                sta     draw_z
                        lda     ClawDrawData,y
                        sta     $2d
                        iny
                        iny
                        sty     $38
                        tax
                        and     #$07
                        tay
                        txa
                        asl     a
                        sta     $2b
                        lsr     a
                        lsr     a
                        lsr     a
                        lsr     a
                        and     #$07
                        tax
                        lda     $2b
                        eor     $9b
                        bmi     locbf11
                        lda     $0078,y
                        sta     $61
                        lda     $0080,y
                        clv
                        bvc     locbf22
locbf11:                lda     $0078,y
                        eor     #$ff
                        clc
                        adc     #$01
                        sta     $61
                        lda     $0080,y
                        eor     #$ff
                        adc     #$00
locbf22:                sta     $62
                        lda     $2d
                        eor     $9d
                        bpl     locbf38
                        lda     $88,x
                        clc
                        adc     $61
                        sta     $61
                        lda     $90,x
                        adc     $62
                        clv
                        bvc     locbf43
locbf38:                lda     $61
                        sec
                        sbc     $88,x
                        sta     $61
                        lda     $62
                        sbc     $90,x
locbf43:                sta     $62
                        lda     $2b
                        eor     $9d
                        bmi     locbf56
                        lda     $0088,y
                        sta     $63
                        lda     $0090,y
                        clv
                        bvc     locbf67
locbf56:                lda     $0088,y
                        eor     #$ff
                        clc
                        adc     #$01
                        sta     $63
                        lda     $0090,y
                        eor     #$ff
                        adc     #$00
locbf67:                sta     $64
                        lda     $2d
                        eor     $9b
                        bpl     locbf7d
                        lda     $63
                        sec
                        sbc     $78,x
                        sta     $63
                        lda     $64
                        sbc     $80,x
                        clv
                        bvc     locbf88
locbf7d:                lda     $63
                        clc
                        adc     $78,x
                        sta     $63
                        lda     $64
                        adc     $80,x
locbf88:                sta     $64
                        ldy     $a9
                        lda     $63
                        sta     (vidptr_l),y
                        iny
                        lda     $64
                        and     #$1f
                        sta     (vidptr_l),y
                        iny
                        lda     $61
                        sta     (vidptr_l),y
                        iny
                        lda     $62
                        and     #$1f
                        ora     draw_z
                        sta     (vidptr_l),y
                        iny
                        sty     $a9
                        dec     rgr_pt_inx
                        beq     locbfaf
                        jmp     locbedb
locbfaf:                ldy     $a9
                        dey
                        jmp     inc_vi.word

; Rotatable graphics values.
; These are indexed by graphic number:
; 0 = flipper
; 1-8 = claw positions within segment
; 9-d = pulsars of varying jaggedness
; I don't know what this.byte is.


                        .byte   $08                     ; I don't want to 'OPTIMIZE' this in case it intentionally indexes backwards to this, just a hunch...

; Number of points.  Indexed by graphic number.

locbfb6:                .byte   $08
                        .byte   $08
                        .byte   $08
                        .byte   $08
                        .byte   $08
                        .byte   $08
                        .byte   $08
                        .byte   $08
                        .byte   $09
                        .byte   $06
                        .byte   $07
                        .byte   $07
                        .byte   $04
                        .byte   $02

; Starting offsets into points vector.  Indexed by graphic number.

locbfc4:                .byte   $00
                        .byte   $10
                        .byte   $20
                        .byte   $30
                        .byte   $40
                        .byte   $50
                        .byte   $60
                        .byte   $70
                        .byte   $80
                        .byte   $92
                        .byte   $9e
                        .byte   $ac
                        .byte   $ba
                        .byte   $c2

; Points vector.
; Each point occupies two.bytes.  The second is just a draw/nodraw flag,
; always 0 (nodraw) or 1 (draw).  The first holds the coordinates.  They
; are encoded thus:
; xx yyy xxx
; || ||| +++---> X coordinate
; || +++-------> Y coordinate
; |+-----------> 1 if X coord should be negated, 0 if not
; +------------> 1 if Y coord should be negated, 0 if not
; For example, the two.bytes at $bfd6 are $4a $01.  The $01 indicates that
; a line should be drawn; the $4a is 01 001 010, so we have X=-2 Y=1.
; flipper
; Why flippers use eight lines rather than six I don't know.  Maybe someone
; felt the crossing point in the middle should (FSVO "should") be on a
; point with integral coordinates.  Maybe it's a historical artifact from
; some previous flipper design (the format here means you can't have a
; delta of 8 or higher for a line, so if the upper points were pulled out
; to the ends, you couldn't do a six-line flipper).

ClawDrawData:           .word   $010c       ; 00 001 100
                        .word   $018c       ; 10 001 100    
                        .word   $014a       ; 01 001 010    
                        .word   $0109       ; 00 001 001     
                        .word   $01cb       ; 11 001 011
                        .word   $014b       ; 01 001 011
                        .word   $0189       ; 10 001 001
                        .word   $01ca       ; 11 001 010

; claw position 1

                        .word   $0190       ; 10 010 000
                        .word   $018a       ; 10 001 010
                        .word   $0123       ; 00 100 011
                        .word   $01db       ; 11 011 011
                        .word   $0141       ; 01 000 001
                        .word   $0110       ; 00 010 000
                        .word   $010a       ; 00 001 010
                        .word   $01cb       ; 11 001 011

; claw position 2

                        .word   $0191       ; 10 010 001
                        .word   $0117       ; 00 010 111
                        .word   $014b       ; 01 001 011
                        .word   $018a       ; 10 001 010
                        .word   $01ce       ; 11 001 110
                        .word   $0108       ; 00 001 000
                        .word   $010a       ; 00 001 010
                        .word   $01cb       ; 11 001 011

; claw position 3

                        .word   $0192       ; 10 010 010
                        .word   $0116       ; 00 010 110
                        .word   $014b       ; 01 001 011
                        .word   $018a       ; 10 001 010
                        .word   $01cd       ; 11 001 101
                        .word   $0149       ; 01 001 001
                        .word   $010a       ; 00 001 010
                        .word   $01cb       ; 11 001 011

; claw position 4

                        .word   $0193       ; 10 010 011
                        .word   $0115       ; 00 010 101
                        .word   $014b       ; 01 001 011
                        .word   $018a       ; 10 001 010
                        .word   $01cc       ; 11 001 100
                        .word   $014a       ; 01 001 010
                        .word   $010a       ; 00 001 010
                        .word   $01cb       ; 11 001 011

; claw position 5

                        .word   $0195       ; 10 010 101
                        .word   $0113       ; 00 010 011
                        .word   $014b       ; 01 001 011
                        .word   $018a       ; 10 001 010
                        .word   $01ca       ; 11 001 010
                        .word   $014c       ; 01 001 100
                        .word   $010a       ; 00 001 010
                        .word   $01cb       ; 11 001 011

; claw position 6

                        .word   $0196       ; 10 010 110
                        .word   $0112       ; 00 010 010
                        .word   $014b       ; 01 001 011
                        .word   $018a       ; 10 001 010
                        .word   $01c9       ; 11 001 001
                        .word   $014d       ; 01 001 101
                        .word   $010a       ; 00 001 010
                        .word   $01cb       ; 11 001 011

; claw position 7

                        .word   $0197       ; 10 010 111
                        .word   $0111       ; 00 010 001
                        .word   $014b       ; 01 001 011
                        .word   $018a       ; 10 001 010
                        .word   $0188       ; 10 001 000
                        .word   $014e       ; 01 001 110
                        .word   $010a       ; 00 001 010
                        .word   $01cb       ; 11 001 011

; claw position 8

                        .word   $000b       ; 00 001 011 no-draw
                        .word   $01a3       ; 10 100 011
                        .word   $010a       ; 00 001 010
                        .word   $0110       ; 00 010 000
                        .word   $014b       ; 01 001 011
                        .word   $018a       ; 10 001 010
                        .word   $0190       ; 10 010 000
                        .word   $0141       ; 01 000 001
                        .word   $015b       ; 01 011 011

; pulsar variant 1

                        .word   $019a       ; 10 011 010
                        .word   $0131       ; 00 110 001
                        .word   $01b1       ; 10 110 001
                        .word   $0131       ; 00 110 001
                        .word   $01b1       ; 10 110 001
                        .word   $011a       ; 00 011 010

; pulsar variant 2

                        .word   $0001       ; 00 000 001 no-draw
                        .word   $0191       ; 10 010 001
                        .word   $0121       ; 00 100 001
                        .word   $01a1       ; 10 100 001
                        .word   $0121       ; 00 100 001
                        .word   $01a1       ; 10 100 001
                        .word   $0111       ; 00 010 001

; pulsar variant 3

                        .word   $0001       ; 00 000 001 no-draw
                        .word   $0189       ; 10 001 001
                        .word   $0111       ; 00 010 001
                        .word   $0191       ; 10 010 001
                        .word   $0111       ; 00 010 001
                        .word   $0191       ; 10 010 001
                        .word   $0109       ; 00 001 001

; pulsar variant 4

                        .word   $0001       ; 00 000 001 no-draw
                        .word   $018a       ; 10 001 010
                        .word   $0112       ; 00 010 010
                        .word   $018a       ; 10 001 010

; pulsar variant 5

                        .word   $0001       ; 00 000 001 no-draw
                        .word   $0106       ; 00 000 110

locc098:                lda     $57
                        sec
                        sbc     $5f
                        sta     mb_w_15
                        lda     #$00
                        sbc     $5b
                        sta     mb_w_16
                        bpl     locc0b3
                        lda     #$00
                        sta     mb_w_16
                        lda     #$01
                        sta     mb_w_15
locc0b3:                lda     $58
                        cmp     y3d
                        bcc     locc0c0
                        sbc     y3d
                        ldx     #$00
                        clv
                        bvc     locc0c7
locc0c0:                lda     y3d
                        sec
                        sbc     $58
                        ldx     #$ff
locc0c7:                sta     mb_w_0e
                        sta     mb_w_14
                        stx     $33
                        lda     $56
                        cmp     $5e
                        bcc     locc0dc
                        sbc     $5e
                        ldx     #$00
                        clv
                        bvc     locc0e3
locc0dc:                lda     $5e
                        sec
                        sbc     $56
                        ldx     #$ff
locc0e3:                sta     $32
                        stx     $34
locc0e7:                bit     eactl_mbst
                        bmi     locc0e7
                        lda     mb_rd_l
                        sta     $63
                        lda     mb_rd_h
                        sta     $64
                        lda     $32
                        sta     mb_w_0e
                        sta     mb_w_14
                        lda     $33
                        bmi     locc11a
                        lda     $63
                        clc
                        adc     $68
                        sta     $63
                        lda     $64
                        adc     $69
                        bvc     locc115
                        lda     #$ff
                        sta     $63
                        lda     #$7f
locc115:                sta     $64
                        clv
                        bvc     locc12f
locc11a:                lda     $68
                        sec
                        sbc     $63
                        sta     $63
                        lda     $69
                        sbc     $64
                        bvc     locc12d
                        lda     #$00
                        sta     $63
                        lda     #$80
locc12d:                sta     $64
locc12f:                bit     eactl_mbst
                        bmi     locc12f
                        lda     mb_rd_l
                        sta     $61
                        lda     mb_rd_h
                        sta     $62
                        ldx     $34
                        bmi     locc158
                        lda     $61
                        clc
                        adc     $66
                        sta     $61
                        lda     $62
                        adc     $67
                        bvc     locc155
                        lda     #$ff
                        sta     $61
                        lda     #$7f
locc155:                sta     $62
                        rts
locc158:                lda     $66
                        sec
                        sbc     $61
                        sta     $61
                        lda     $67
                        sbc     $62
                        bvc     locc16b
                        lda     #$00
                        sta     $61
                        lda     #$80
locc16b:                sta     $62
                        rts

InitVector:             jsr     locaa13
                        lda     #$80
                        sta     $5e
                        lda     #$ff
                        sta     $0114
                        jsr     locc235
                        lda     $0133
                        bne     locc185
                        sta     vg_reset
locc185:                lda     #$00
                        sta     $0133
                        lda     loccec6
                        sta     vecram
                        lda     loccec7
                        sta     vecram+1

; Install the correct colors for curlevel.

SetLevelColors:         lda     curlevel                            ; Tke the current level
                        and     #$70                                ;   Round it to the start of this color level (eg 0, 16, 32, 48, etc)
                        cmp     #LAST_SHAPE_LEVEL                   ;   Compare it to (default 95) last level shape before the 3 extra 
                        bcc     +
                        lda     #LAST_SHAPE_LEVEL                   ; If higher, use LAST_SHAPE_LEVEL instead
+                       lsr     a                                   ; Divide by 2, since there are 8 colors per 16 levels
                        ora     #$07                                ; Set the bits 0000111, so we start at the end of the table and work backwards
                        tax                                         ; Use the result as the indexer into aLevelColors
                        ldy     #$07
loopLoadColors:         lda     aLevelColors,x                      ; Get the color from the color table
                        and     #$0f                                ; Mask away the top nibble leaving only the bottom
                        sta     $0019,y                             ; Store it in the $0019 array at the Yth index
                        sta     col_ram,y                           ; Store it in the $0800 array as the Yth index
                        lda     aLevelColors,x                      ; Get a fresh copy again
                        lsr     a                                   ; This time move the high nibble down to the low 
                        lsr     a
                        lsr     a
                        lsr     a
                        sta     $0021,y                             ; Store it in the $0021 table at the Yth index
                        sta     $0808,y                             ; Store it in the $0808 table at the Yth index
                        dex
                        dey
                        bpl     loopLoadColors                      ; Repeat until all 8 colors installed
                        rts

locc1c3:                lda     #$00
                        sta     $81
                        sta     $91
                        sta     $80
                        sta     $78
                        sta     $90
                        sta     $88
                        lda     #$00
                        sta     mb_w_00
                        sta     mb_w_01
                        sta     mb_w_04
                        sta     mb_w_05
                        sta     mb_w_06
                        sta     mb_w_07
                        sta     mb_w_09
                        sta     mb_w_03
                        sta     mb_w_0d
                        sta     mb_w_0e
                        sta     mb_w_0f
                        sta     mb_w_10
                        lda     #$0f
                        sta     mb_w_0c
                        rts

; Used at $c1a6 and $c1b1.
; Appear to be blocks of 8.bytes giving colours for the various
; blocks of 16 levels.

;                                      Player  Tanker  Flipper Pulsar          Spiker Field 
aLevelColors:           .byte   White, Yellow, Purple, Red,    Cyan+Sparkle,   Green, Blue,   Blue    ; Blue   Levels 1-16
                        .byte   White, Green,  Blue,   Purple, Yellow+Unk04,   Cyan,  Red,    Red     ; Red    Levels 17-32
                        .byte   White, Blue,   Cyan,   Green,  Purple+Sparkle, Red,   Yellow, Yellow  ; Yellow Levels 33-48
                        .byte   White, Blue,   Purple, Green,  Yellow+Sparkle, Red,   Cyan,   Cyan    ; Cyan   Levels 49-64
                        .byte   White, Yellow, Purple, Red,    Cyan+Sparkle,   Green, Black,  Blue    ; Black  Levels 65-80
                        .byte   White, Red,    Purple, Yellow, Cyan+Sparkle,   Blue,  Green,  Green   ; Green  Levels 81-96
.if ADD_LEVEL
                        .byte   White, Yellow, Blue,   Green,   White+Sparkle, Cyan,  Purple, Purple  ; Purple Levels 97-112
.endif
      
; During the level picker, this is the color table used to draw the preview of the levels
; The index returned here is placed in 'curcolor', but I'm not yet sure how that becomes an actual color yet..
                  
LevelSelectColors:      .byte   6           ; Blue
                        .byte   3           ; Red
                        .byte   1           ; Yellow
                        .byte   4           ; Cyan
                        .byte   0           ; Black
                        .byte   5           ; Green
.if !ADD_LEVEL                      
                        .byte   5           ; Green again
.else
                        .byte   2           ; Purple
.endif                                              
                        .byte   5           ; Should never be reached
                        
locc235:                ldx     curplayer
                        lda     p1_level,x
                        jsr     get_tube_no
                        pha
                        ldy     curtube
                        lda     lev_scale,y
                        eor     #$ff
                        clc
                        adc     #$01
                        sta     $5f
                        sta     $5d
                        lda     #$10
                        sec
                        sbc     $5f
                        sta     $a0
                        lda     #$ff
                        sta     $5b
                        lda     lev_y3d,y
                        sta     y3d
                        lda     lev_open,y
                        sta     open_level
                        lda     state_after_delay
                        cmp     #GS_DelayThenPlay
                        bne     locc275
                        lda     lev_y2d,y
                        sta     $68
                        lda     lev_y2db,y
                        sta     $69
                        clv
                        bvc     locc28d
locc275:                lda     lev_y2d,y
                        sec
                        sbc     $68
                        sta     $0121
                        lda     lev_y2db,y

                        .byte $ed, $69, $00           ; BUGBUG non-zero page, was 'sbc $0069'

                        ldx     #$03
locc286:                lsr     a
                        ror     $0121
                        dex
                        bpl     locc286
locc28d:                lda     #$00
                        sta     $66
                        sta     $67
                        lda     #$00
                        sta     $010f
                        sta     $0110
                        lda     #$2c
                        sta     $0113
                        pla
                        tay
                        ldx     #$0f
locc2a4:                lda     lev_x,y
                        sta     tube_x,x
                        lda     lev_y,y
                        sta     tube_y,x
                        lda     #$00
                        sta     $031a,x
                        sta     $033a,x
                        sta     $039a,x
                        lda     lev_angle,y
                        sta     tube_angle,x
                        dey
                        dex
                        bpl     locc2a4
                        ldy     #$00
                        ldx     #$0f
locc2c9:                lda     tube_x,y
                        sec
                        adc     tube_x,x
                        ror     a
                        sta     mid_x,x
                        lda     tube_y,y
                        sec
                        adc     tube_y,x
                        ror     a
                        sta     mid_y,x
                        dey
                        bpl     locc2e4
                        ldy     #$0f
locc2e4:                dex
                        bpl     locc2c9
                        rts

; Take the level number in A, do the random thing for level 99, and fetch
; the tube number for the level.  Also return a value with the tube number
; in the high four bits and $f in the low four bits; this is used as an
; index into the [][] tables.

; Returns A = High nibble has shape number, low bits all set

get_tube_no:            ldx     #$00
                        cmp     #HIGHEST_LEVEL
                        bcc     locc2f3
                        lda     pokey1_rand
                        and     #$5f

; This appears to be "high nibble into X, keep low nibble in A".  I can't
; see why not compute that directly with shifts and masks instead of a
; subtract loop.  Maybe the $10 here was an assembly-time constant rather
; than being a deeply-wired-in value?  That's not very plausible, though,
; as level numbers are shifted by four bits elsewhere.

locc2f3:                cmp     #$10
locc2f5:                bcc     locc2fb
                        inx
                        sec
                        sbc     #$10
locc2fb:                cmp     #$10
                        bcs     locc2f5
                        tay
                        lda     lev_remap,y
                        sta     curtube
                        asl     a
                        asl     a
                        asl     a
                        asl     a
                        ora     #$0f
                        rts
locc30d:                lda     $0110
                        bne     locc339
                        lda     #$f0
                        sta     $57
                        ldx     #$4f
                        jsr     locc473
                        sta     $0110
                        beq     locc323
                        sta     $010f
locc323:                lda     $010f
                        bne     locc339
                        lda     #$10
                        sta     $57
                        jsr     locc453
                        lda     $57
                        ldx     #$0f
                        jsr     locc473
                        sta     $010f
locc339:                lda     #$01
                        jsr     vapp_scale_A_0
                        ldy     #$06
                        sty     curcolor
                        ldx     $0110
                        beq     locc348
                        rts
locc348:                ldx     $0113
                        bne     locc34e
                        rts
locc34e:                ldx     #$0f
locc350:                lda     #$c0
                        jsr     locc3ee
                        dex
                        bpl     locc350
                        ldy     #$06
                        sty     curcolor
                        lda     #$08
                        jsr     vapp_sclstat_A_Y
                        ldy     #$4f
                        lda     $0110
                        jsr     locc36e
                        ldy     #$0f
                        lda     $010f
locc36e:                bne     locc3b9
                        sty     $37
                        lda     $032a,y
                        sta     $61
                        lda     $031a,y
                        sta     $62
                        lda     $034a,y
                        sta     $63
                        lda     $033a,y
                        sta     $64
                        ldx     #$61
                        jsr     locc772
                        lda     vidptr_l
                        sta     $b0
                        lda     vidptr_h
                        sta     $b1
                        ldx     #$0f
                        lda     open_level
                        beq     locc39b
                        dex
locc39b:                lda     #$c0
                        sta     draw_z
                        stx     $38
locc3a1:                dec     $37
                        lda     $37
                        and     #$0f
                        cmp     #$0f
                        bne     locc3b2
                        lda     $37
                        clc
                        adc     #$10
                        sta     $37
locc3b2:                jsr     locc423
                        dec     $38
                        bpl     locc3a1
locc3b9:                rts
locc3ba:                lda     $61
                        sec
                        sbc     $6a
                        sta     $6e
                        lda     $62
                        sbc     $6b
                        sta     $6f
                        lda     $63
                        sec
                        sbc     $6c
                        sta     $70
                        lda     $64
                        sbc     $6d
                        sta     $71
                        ldx     #$6e
                        jsr     locdf92
                        lda     $61
                        sta     $6a
                        lda     $62
                        sta     $6b
                        lda     $63
                        sta     $6c
                        lda     $64
                        sta     $6d
                        lda     #$c0
                        sta     draw_z
                        rts
locc3ee:                stx     $37
                        pha
                        ldy     curcolor
                        lda     #$08
                        jsr     vapp_sclstat_A_Y
                        jsr     locc43c
                        ldx     #$61
                        jsr     locc772
                        pla
                        sta     draw_z
                        pha
                        jsr     locc423
                        dec     $37
                        ldy     curcolor
                        lda     #$00
                        sta     draw_z
                        lda     #$08
                        jsr     vapp_sclstat_A_Y
                        jsr     locc423
                        pla
                        sta     draw_z
                        jsr     locc43c
                        jsr     locc3ba
                        ldx     $37
                        rts
locc423:                ldx     $37
                        lda     $032a,x
                        sta     $61
                        lda     $031a,x
                        sta     $62
                        lda     $034a,x
                        sta     $63
                        lda     $033a,x
                        sta     $64
                        jmp     locc3ba
locc43c:                ldx     $37
                        lda     $036a,x
                        sta     $61
                        lda     $035a,x
                        sta     $62
                        lda     $038a,x
                        sta     $63
                        lda     $037a,x
                        sta     $64
                        rts
locc453:                lda     $5b
                        bne     locc471
                        lda     $57
                        sec
                        sbc     $5f
                        bcc     locc460
                        cmp     #$0c
locc460:                bcs     locc471
                        lda     $5f
                        clc
                        adc     #$0f
                        bcs     locc46b
                        cmp     #$f0
locc46b:                bcc     locc46f
                        lda     #$f0
locc46f:                sta     $57
locc471:                rts

                        .byte   $db

locc473:                sta     $57
                        stx     $38
                        lda     #$00
                        sta     fscale
                        ldx     #$0f
                        stx     $37
locc47f:                ldx     $37
                        lda     tube_x,x
                        sta     $56
                        lda     tube_y,x
                        sta     $58
                        jsr     locc098
                        ldx     $38
                        ldy     $61
                        lda     $62
                        bmi     locc4a3
                        cmp     #$04
                        bcc     locc4a0
                        ldy     #$ff
                        lda     #$03
                        inc     fscale
locc4a0:                clv
                        bvc     locc4ad
locc4a3:                cmp     #$fc
                        bcs     locc4ad
                        ldy     #$01
                        lda     #$fc
                        inc     fscale
locc4ad:                sta     $031a,x
                        tya
                        sta     $032a,x
                        ldy     $63
                        lda     $64
                        bmi     locc4c7
                        cmp     #$04
                        bcc     locc4c4
                        ldy     #$ff
                        lda     #$03
                        inc     fscale
locc4c4:                clv
                        bvc     locc4d1
locc4c7:                cmp     #$fc
                        bcs     locc4d1
                        lda     #$fc
                        ldy     #$01
                        inc     fscale
locc4d1:                sta     $033a,x
                        tya
                        sta     $034a,x
                        dec     $38
                        dec     $37
                        bpl     locc47f
                        lda     fscale
                        rts
locc4e1:                jsr     get_tube_no
                        sta     $36
                        stx     $35
                        lda     #$00
                        sta     draw_z
                        lda     #$05
                        jsr     vapp_scale_A_0
                        lda     $35
                        and     #$07
                        tax
                        ldy     LevelSelectColors,x
                        sty     curcolor
                        lda     #$08
                        jsr     vapp_sclstat_A_Y
                        ldx     curtube
                        lda     $36
                        ldy     lev_open,x
                        bne     locc50d
                        sec
                        sbc     #$0f
locc50d:                tay
                        lda     lev_y,y
                        sta     $57
                        eor     #$80
                        tax
                        lda     lev_x,y
                        sta     $56
                        eor     #$80
                        jsr     vapp_ldraw_A_X
                        lda     #$c0
                        sta     draw_z
                        ldx     #$0f
                        stx     $38
locc528:                ldy     $36
                        lda     lev_x,y
                        tax
                        sec
                        sbc     $56
                        pha
                        stx     $56
                        lda     lev_y,y
                        tay
                        sec
                        sbc     $57
                        tax
                        sty     $57
                        pla
                        jsr     vapp_ldraw_A_X
                        dec     $36
                        dec     $38
                        bpl     locc528
                        lda     #$01
                        jmp     vapp_scale_A_0
locc54d:                lda     $0115
                        beq     locc5b1
                        lda     $5f
                        pha
                        lda     $5b
                        pha
                        lda     $a0
                        pha
                        lda     #$e8
                        sta     $5f
                        lda     #$ff
                        sta     $5b
                        lda     #$28
                        sta     $a0
                        ldx     #$07
                        stx     $37
locc56b:                ldx     $37
                        lda     $03fe,x
                        beq     locc5a4
                        sta     $57
                        lda     #$80
                        sta     $56
                        lda     #$80
                        sta     $58
                        lda     curlevel
                        cmp     #$05
                        bcs     locc587
                        lda     #$06
                        clv
                        bvc     locc590
locc587:                txa
                        and     #$07
                        cmp     #$07
                        bne     locc590
                        lda     #$04
locc590:                sta     curcolor
                        tay
                        lda     #$08
                        jsr     vapp_sclstat_A_Y
                        lda     $37
                        and     #$03
                        asl     a
                        adc     #$0a
                        sta     $55
                        jsr     locbd09
locc5a4:                dec     $37
                        bpl     locc56b
                        pla
                        sta     $a0
                        pla
                        sta     $5b
                        pla
                        sta     $5f
locc5b1:                lda     $011f
                        beq     locc5c1

; If P1 score is in the 150K-160K range, increment one byte of RAM in the
; $0200-$0299 range, depending on the low two digits of the score.  This is
; probably responsible for a few of the game crashes I've seen; there are
; some bytes that if struck by this _will_ cause trouble.

                        ldx     p1_score_h
                        cpx     #$15
                        bcc     locc5c1
                        ldx     p1_score_l
                        inc     player_seg,x
locc5c1:                rts
locc5c2:                lda     $0110
                        beq     locc5c8
                        rts
locc5c8:                lda     $5b
                        bne     locc5d3
                        lda     $5f
                        cmp     #$f0
                        bcc     locc5d3
                        rts
locc5d3:                lda     #$01
                        jsr     vapp_scale_A_0
                        lda     vidptr_l
                        pha
                        lda     vidptr_h
                        pha
                        lda     #$00
                        sta     $38
                        sta     $a9
                        ldx     #$0f
                        lda     open_level
                        beq     locc5ec
                        dex
locc5ec:                stx     $37
locc5ee:                ldx     #$03
                        ldy     $a9
locc5f2:                lda     locc669,x
                        sta     (vidptr_l),y
                        iny
                        dex
                        bpl     locc5f2
                        sty     $a9
                        lda     $0114
                        bne     locc64c
                        ldx     $38
                        lda     $039a,x
                        bmi     locc61a
                        ldx     #$0b
                        ldy     $a9
locc60d:                lda     ($aa),y
                        sta     (vidptr_l),y
                        iny
                        dex
                        bpl     locc60d
                        sty     $a9
                        clv
                        bvc     locc649
locc61a:                ldy     $a9
                        lda     ($aa),y
                        sta     (vidptr_l),y
                        sta     $6c
                        iny
                        lda     ($aa),y
                        sta     (vidptr_l),y
                        cmp     #$10
                        bcc     locc62d
                        ora     #$e0
locc62d:                sta     $6d
                        iny
                        lda     ($aa),y
                        sta     (vidptr_l),y
                        sta     $6a
                        iny
                        lda     ($aa),y
                        sta     (vidptr_l),y
                        cmp     #$10
                        bcc     locc641
                        ora     #$e0
locc641:                sta     $6b
                        iny
                        sty     $a9
                        jsr     locc6c7
locc649:                clv
                        bvc     locc652
locc64c:                jsr     locc66d
                        jsr     locc6c7
locc652:                ldx     $38
                        asl     $039a,x
                        inc     $38
                        dec     $37
                        bpl     locc5ee
                        pla
                        sta     $ab
                        pla
                        sta     $aa
                        ldy     $a9
                        dey
                        jmp     inc_vi.word

; These four.bytes are.byte-reversed video code.  The code is
; 6805  vstat z=0 c=5 sparkle=1
; 8040  vcentre
; This is used by the loop at c5f2.

locc669:                .byte   $80     
                        .byte   $40     
                        .byte   $68     
                        .byte   $05     

locc66d:                lda     $38
                        tax
                        clc
                        adc     #$01
                        and     #$0f
                        tay
                        lda     $036a,x
                        sec
                        adc     $036a,y
                        sta     $61
                        lda     $035a,x
                        adc     $035a,y
                        sta     $62
                        asl     a
                        ror     $62
                        ror     $61
                        lda     $038a,x
                        sec
                        adc     $038a,y
                        sta     $63
                        lda     $037a,x
                        adc     $037a,y
                        sta     $64
                        asl     a
                        ror     $64
                        ror     $63
                        ldy     $a9
                        lda     $63
                        sta     (vidptr_l),y
                        iny
                        sta     $6c
                        lda     $64
                        sta     $6d
                        and     #$1f
                        sta     (vidptr_l),y
                        iny
                        lda     $61
                        sta     (vidptr_l),y
                        iny
                        sta     $6a
                        lda     $62
                        sta     $6b
                        and     #$1f
                        sta     (vidptr_l),y
                        iny
                        sty     $a9
                        rts
locc6c7:                ldx     $38
                        lda     lane_spike_height,x
                        bne     locc6e4
                        ldy     $a9
                        ldx     #$03
locc6d2:                lda     #$00
                        sta     (vidptr_l),y
                        iny
                        lda     #$71
                        sta     (vidptr_l),y
                        iny
                        dex
                        bpl     locc6d2
                        sty     $a9
                        clv
                        bvc     locc73b
locc6e4:                sta     $57
                        jsr     locc453
                        lda     mid_x,x
                        sta     $56
                        lda     mid_y,x
                        sta     $58
                        jsr     locc098
                        jsr     locc73c
                        ldx     $38
                        lda     $039a,x
                        and     #$40
                        beq     locc721
                        jsr     locbd3e
                        lda     pokey1_rand
                        and     #$02
                        clc
                        adc     #$1c
                        tax
                        lda     graphic_table+1,x
                        iny
                        sta     (vidptr_l),y
                        dey
                        lda     graphic_table,x
                        sta     (vidptr_l),y
                        iny
                        iny
                        sty     $a9
                        clv
                        bvc     locc73b
locc721:                ldy     $a9
                        lda     #$00
                        sta     (vidptr_l),y
                        iny
                        lda     #$68
                        sta     (vidptr_l),y
                        iny
                        lda     $3db2
                        sta     (vidptr_l),y
                        iny
                        lda     $3db3
                        sta     (vidptr_l),y
                        iny
                        sty     $a9
locc73b:                rts
locc73c:                ldy     $a9
                        lda     $63
                        sec
                        sbc     $6c
                        sta     (vidptr_l),y
                        iny
                        lda     $64
                        sbc     $6d
                        and     #$1f
                        sta     (vidptr_l),y
                        iny
                        lda     $61
                        sec
                        sbc     $6a
                        sta     (vidptr_l),y
                        iny
                        lda     $62
                        sbc     $6b
                        and     #$1f
                        ora     #$a0
                        sta     (vidptr_l),y
                        iny
                        sty     $a9
                        rts

; On entry, X contains zero-page address of a four.byte block, holding
; AA BB CC DD.  Then, the following are appended to the video list:
;       vscale  b=1 l=0
;       vcentre
;       vldraw  x=X y=Y z=off
; where X=DDCC and Y=BBAA, in each case taken as 13-bit signed numbers
; (ie, the high three bits are dropped).

vapp_to_X_:             ldy     #$00
                        tya
                        sta     (vidptr_l),y
                        lda     #$71
                        iny
                        sta     (vidptr_l),y
                        iny
                        bne     locc774
locc772:                ldy     #$00
locc774:                lda     #$40
                        sta     (vidptr_l),y
                        lda     #$80
                        iny
                        sta     (vidptr_l),y
                        iny
                        lda     $02,x
                        sta     $6c
                        sta     (vidptr_l),y
                        iny
                        lda     timectr,x
                        sta     $6d
                        and     #$1f
                        sta     (vidptr_l),y
                        lda     gamestate,x
                        sta     $6a
                        iny
                        sta     (vidptr_l),y
                        lda     $01,x
                        sta     $6b
                        and     #$1f
                        iny
                        sta     (vidptr_l),y
                        jmp     inc_vi.word
locc7a0:                jsr     loccd95
                        lda     #GS_GameStartup
                        sta     gamestate

; This appears to be the game's main loop, from here through $c7bb.
; (This does not apply to reset-in-service-mode selftest, which has its
; own, different, main loop.)

GameMainLoop:           lda     $53
                        cmp     #$09
                        bcc     GameMainLoop

                        lda     #$00
                        sta     $53

                        jsr     locc7bd
                        jsr     locc891
                        jsr     locb1b6
                        clc
                        bcc     GameMainLoop

locc7bd:                lda     optsw1

; This and and cmp tests one of the bonus-coins bits and the coinage bits;
; the compare will show equal if "bonus coins" is set to one of
; "1 each 5", "1 each 3", or demo mode (frozen or not) and coinage is
; set to free play.  Why this is a useful thing to test I have no idea;
; perhaps it's documented as a magic combination?
; Another disassembly comments this as checking for demonstration
; freeze mode, which is inconsistent with the layout of the bits in
; $0d00 - for it to be that, it'd have to be and #$e0, cmp #$e0.

                        and     #$83
                        cmp     #$82
                        beq     locc7d9
                        jsr     loca7d2
                        ldx     gamestate
                        lda     zap_fire_new
                        ora     #$80
                        sta     zap_fire_new
                        lda     GameStateDispatchTable+1,x
                        pha
                        lda     GameStateDispatchTable,x
                        pha
locc7d9:                rts

; Jump table used by the code at c7d1
; Indexed by general game state.

GameStateDispatchTable: .word   State_GameStartup-1         ; 00 - Game Startup
                        .word   State_LevelStartup-1        ; 02 - Level Startup
                        .word   State_Playing-1             ; 04 - Playing
                        .word   State_Death-1               ; 06 - Death Start
                        .word   State_LevelBegin-1          ; 08 - Level Begin?
                        .word   State_Delay-1               ; 0A - Avoid Spikes, high-score, countdowns
                        .word   0000                        ; 0C - Unused, we assume
                        .word   State_ZoomOffEnd-1          ; 0E - Zoomed off end of level
                        .word   state_10-1                  ; 10 - ???
                        .word   State_EnterInitials-1       ; 12 - Entering Initials
                        .word   state_14-1                  ; 14 - ???
                        .word   State_LevelSelect-1         ; 16 - Level Selection Startup
                        .word   State_ZoomOntoNew-1         ; 18 - Zoomed into new level
                        .word   state_1a-1                  ; 1A - ???
                        .word   state_1c-1                  ; 1C - ???
                        .word   State_DelayThenPlay-1       ; 1E - A brief pause then flips to Playing state
                        .word   State_ZoomingDown-1         ; 20 - Zooing down tube
                        .word   State_ServiceDisplay-1      ; 22 - Service display
                        .word   State_HighScoreExplosion-1  ; 24 - High Score Explosion

;-----------------------------------------------------------------------------
; State_Delay
;-----------------------------------------------------------------------------
; This state is used as a delay with the player input (whether in-game or
; during high score input, etc) still active.  When the delay has expired,
; whatever value was in 'state_after_delay' becomes the new gamestate.
;-----------------------------------------------------------------------------

State_Delay:            lda     timectr
                        and     $016b
                        bne     locc818
                        lda     countdown_timer
                        beq     locc80d
                        dec     countdown_timer
locc80d:                bne     locc818
                        lda     state_after_delay           ; Replace game state with "next pending" state after delay
                        sta     gamestate
                        lda     #$00
                        sta     $016b
locc818:                jmp     move_player

; Check to see if either START button is pressed, to start a new game.
; Called only when credits > 0.

check_start:            lda     credits
                        ldy     #$00
                        cmp     #$02
                        lda     zap_fire_new
                        and     #$60                ; start1 & start2
                        sty     zap_fire_new
                        beq     locc871             ; branch if neither pressed
                        bcs     locc830
                        and     #$20                ; start1
                        clv
                        bvc     locc835
locc830:                iny
                        dec     credits
                        and     #$40                ; start2
locc835:                beq     locc83a
                        dec     credits
                        iny

; Y now holds 1 if 1p game, 2 if 2p

locc83a:                tya
                        sta     twoplayer
                        beq     locc86e         ; can this ever branch?  How could it be zero?
                        lda     game_mode
                        ora     #$c0            ; 11000000
                        sta     game_mode
                        lda     #$00
                        sta     coin_string
                        sta     $18
                        lda     #GS_GameStartup
                        sta     gamestate
                        dec     twoplayer
                        ldx     twoplayer
                        beq     locc857
                        ldx     #$03            ; 3 = games_2p_l - games_1p_l
locc857:                inc     games_1p_l,x
                        bne     locc85f
                        inc     games_1p_m,x
locc85f:                lda     $0100
                        sec
                        adc     twoplayer
                        cmp     #$63            ; 99 decimal
                        bcc     locc86b
                        lda     #$63            ; 99 again
locc86b:                sta     $0100
locc86e:                clv
                        bvc     locc890

; Branch here if neither start button pressed

locc871:                lda     $50
                        beq     locc890
                        bit     game_mode
                        bmi     locc890
                        lda     #$10
                        sta     unknown_state
                        lda     #$20
                        sta     countdown_timer
                        lda     #GS_Delay
                        sta     gamestate
                        lda     #GS_Unknown14           
                        sta     state_after_delay
                        lda     #$00
                        sta     $50
                        sta     $0123
locc890:                rts

locc891:                lda     cabsw
                        and     #$10                    ; service mode
                        bne     locc89f
                        lda     #GS_ServiceDisplay
                        sta     gamestate
                        clv
                        bvc     locc8e3
locc89f:                bit     game_mode
                        bvs     locc8e3
                        lda     optsw2_shadow
                        and     #$01                    ; 2-credit-minimum bit
                        beq     locc8d2
                        ldy     credits
                        bne     locc8b1
                        lda     #$80
                        sta     $a2
locc8b1:                bit     $a2
                        bpl     locc8d2
                        cpy     #$02
                        bcs     locc8ca
                        tya
                        beq     locc8c4
                        lda     #$16
                        sta     unknown_state
                        lda     #GS_Delay
                        sta     gamestate
locc8c4:                jmp     locc8d9
                        clv
                        bvc     locc8d2
locc8ca:                lda     #GS_Unknown14
                        sta     gamestate
                        lda     #$00
                        sta     $a2
locc8d2:                lda     credits
                        beq     locc8d9
                        jsr     check_start
locc8d9:                lda     coinage_shadow
                        and     #$03
                        bne     locc8e3
                        lda     #$02
                        sta     credits
locc8e3:                inc     timectr
                        lda     timectr
                        and     #$01
                        beq     locc8ee
                        jsr     locde1b
locc8ee:                lda     $0c
                        beq     SetDecimalIfPirated
                        jsr     locccfa

; Apparent anti-piracy test; if the copyright-displaying code has been
; twiddled, then gratuitously drop into decimal mode over level 19, thereby
; doing "interesting" things to arithmetic until we next have occasion to
; use decimal mode "legitimately".
; See also $a91c.

SetDecimalIfPirated:    lda     copyr_disp_cksum1
                        beq     locc901
                        lda     #$13
                        cmp     curlevel
                        bcs     locc901
                        sed

; End apparent anti-piracy test

locc901:                lda     zap_fire_new
                        and     #$80
                        beq     locc90b
                        lda     #$00
                        sta     zap_fire_new  ; Clear zap_fire_new
locc90b:                rts

State_GameStartup:      jsr     maybe_init_hs
                        jsr     InitVector
                        lda     game_mode
                        bpl     locc919
                        jsr     locca62
locc919:                lda     #$00
                        sta     p2_lives
                        ldx     twoplayer
                        stx     curplayer
locc921:                ldx     curplayer
                        lda     init_lives
                        .byte $9d, $48, $00 ; BUGBUG non-zero page, was 'sta     p1_lives,x'
                        lda     #$ff
                        .byte $9d, $46, 00  ; BUGBUG non-zero page, was 'sta     p1_level,x'
                        dec     curplayer
                        bpl     locc921
                        lda     #$00
                        sta     $3f
                        sta     $0115
                        lda     twoplayer
                        sta     curplayer
                        jmp     PlayerLevelSelect

State_LevelStartup:     lda     #$00
                        sta     unknown_state
                        lda     #GS_DelayThenPlay
                        sta     gamestate
                        sta     state_after_delay
                        lda     $3f
                        cmp     curplayer
                        beq     locc96c
                        sta     curplayer
                        lda     game_mode
                        bpl     locc96c
                        lda     #$0e
                        sta     unknown_state
                        lda     #GS_Delay
                        sta     gamestate
                        lda     #$50
                        ldy     flagbits
                        beq     locc967
                        lda     #$28
locc967:                sta     countdown_timer
                        jsr     SwapPlayerStates
locc96c:                jsr     locca48
                        ldx     curplayer
                        lda     p1_level,x
                        sta     curlevel
                        jsr     InitializeGame
                        jmp     loccd95

State_DelayThenPlay:    lda     #GS_Playing                          
                        sta     state_after_delay
                        lda     #$00
                        sta     unknown_state
                        lda     #GS_Delay
                        sta     gamestate
                        lda     #$14
                        sta     countdown_timer
                        rts

State_ZoomOffEnd:       ldx     curplayer
                        lda     p1_level,x
                        cmp     #HIGHEST_LEVEL                  ; 98 (level 99) is the max
                        bcs     level_already_maxed
                        inc     p1_level,x
                        inc     curlevel
level_already_maxed:    lda     #GS_ZoomOntoNew
                        sta     gamestate
                        lda     p1_startchoice,x                ; Award any start level bonus
                        beq     locc9ac
                        jsr     ld_startbonus
                        ldx     #$ff                            ; Indicates bonus value loaded into 29/2a/2b
                        jsr     inc_score
                        jsr     locccb9
locc9ac:                jmp     InitLevel

State_Death:            lda     #$00
                        sta     countdown_timer
                        ldx     curplayer                       ; Current player loses a life, decrement it
                        dec     p1_lives,x
                        lda     p1_lives                        ; Combine Player1 and Player2 lives
                        ora     p2_lives
                        bne     AnyPlayerLivesLeft          

                        jsr     State_LevelBegin
                        clv
                        bvc     locc9f0

AnyPlayerLivesLeft:     ldx     curplayer
                        lda     p1_lives,x
                        bne     PlayerOutOfLives
                        lda     #$0c
                        sta     unknown_state
                        lda     #$28
                        sta     countdown_timer
PlayerOutOfLives:       lda     twoplayer
                        beq     locc9db
                        lda     $3f
                        eor     #$01
                        sta     $3f
locc9db:                ldx     $3f
                        lda     p1_lives,x
                        beq     PlayerOutOfLives
                        lda     #$02
                        ldy     p1_level,x
                        iny
                        bne     locc9ea
                        lda     #GS_Unknown1C
locc9ea:                sta     state_after_delay
                        lda     #GS_Delay
                        sta     gamestate
locc9f0:                rts

State_LevelBegin:       lda     #$00
                        sta     $0126
                        ldx     twoplayer
-                       lda     p1_level,x
                        cmp     $0126
                        bcc     +
                        sta     $0126
+                       dex
                        bpl     -
                        ldy     $0126
                        beq     +
                        dec     $0126
+                       lda     #GS_Unknown14
                        bit     game_mode
                        bpl     +
                        lda     #GS_Unknown10
+                       sta     gamestate
                        rts

state_14:               lda     game_mode
                        and     #$3f                    ; ‭00111111‬
                        sta     game_mode
                        lda     #$00
                        sta     twoplayer
                        lda     #GS_Unknown1A           ; new gamestate
                        sta     state_after_delay
                        lda     #GS_Delay
                        sta     gamestate
                        lda     #$a0                    ; time delay
                        sta     countdown_timer
                        lda     #$01
                        sta     $016b
                        lda     #$0a
                        sta     unknown_state
                        rts

; Used at $98e6, $9910

locca38:                .byte   $80     
                        .byte   $40     
                        .byte   $20     
                        .byte   $10     
                        .byte   $08     
                        .byte   $04     
                        .byte   $02     
                        .byte   $01     
                        .byte   $01     
                        .byte   $02     
                        .byte   $04     
                        .byte   $08     
                        .byte   $10     
                        .byte   $20     
                        .byte   $40     
                        .byte   $80 
    
locca48:                ldy     #$10
                        lda     flagbits
                        beq     locca57
                        lda     curplayer
                        beq     locca57
                        lda     #$04
                        ldy     #$08
locca57:                eor     $a1
                        and     #$04
                        eor     $a1
                        sta     $a1
                        sty     $b4
                        rts

locca62:                lda     #$00
                        ldx     #$05
locca66:                sta     p1_score_l,x
                        dex
                        bpl     locca66
                        rts

;-----------------------------------------------------------------------------
; inc_score
;-----------------------------------------------------------------------------                      
; If x <= 8 then that is used as the enemy type and the player score is credited
; with a table-lookup value for that enemy type.
; 
; If x > 8 then the value at 2A/2B/2C is added to the current score.
;-----------------------------------------------------------------------------


inc_score:              sed
                        bit     game_mode
                        bpl     loccaef
                        ldy     curplayer
                        beq     locca77
                        ldy     #(p2_score_l - p1_score_l)          ; Distance between two scores
locca77:                cpx     #$08                                ; 
                        bcc     ScoreByType
                        lda     $29
                        clc
                        adc     p1_score_l,y
                        sta     p1_score_l,y
                        lda     $2a
                        adc     p1_score_m,y
                        sta     p1_score_m,y
                        lda     $2b
                        clv
                        bvc     loccaa6

ScoreByType:            lda     EnemyScoreValueLSB,x
                        clc
                        adc     p1_score_l,y
                        sta     p1_score_l,y
                        lda     EnemyScoreValueMSB,x
                        adc     p1_score_m,y
                        sta     p1_score_m,y
                        lda     #$00
loccaa6:                php
                        adc     p1_score_h,y
                        sta     p1_score_h,y
                        plp
                        beq     loccabb
                        ldx     bonus_life_each
                        beq     loccabb
                        cpx     $2b
                        beq     loccadc
                        bcc     loccadc
loccabb:                bcc     loccaef
                        ldx     bonus_life_each
                        beq     loccaee
                        cpx     #$03
                        bcc     loccad1
loccac6:                sec
                        sbc     bonus_life_each
                        beq     loccadc
                        bcs     loccac6
                        clv
                        bvc     loccaee
loccad1:                cpx     #$02
                        bne     loccadc
                        and     #$01
                        beq     loccadc
                        clv
                        bvc     loccaee
loccadc:                ldx     curplayer
                        lda     p1_lives,x
                        cmp     #$06
                        bcs     loccaee
                        inc     p1_lives,x
                        jsr     locccb9
                        lda     #$20
                        sta     $0124
loccaee:                sec
loccaef:                cld
                        rts

; EnemyScoreValue - What each enemy type is worth

EnemyScoreValueLSB:     .byte   $00         ; Flipper  150
                        .byte   $50         ; Pulsar   200
                        .byte   $00         ; Tanker   100
                        .byte   $00         ; Spiker    50
                        .byte   $50         ; Fuseball 250
                        .byte   $50         ; Fuseball 500
                        .byte   $00         ; Fuseball 750
                        .byte   $50

; High BCD nibbles of whatever's at caf1.

EnemyScoreValueMSB:     .byte   $00
                        .byte   $01         ; Flipper  150
                        .byte   $02         ; Pulsar   200
                        .byte   $01         ; Tanker   100
                        .byte   $00         ; Spiker    50
                        .byte   $02         ; Fuseball 250
                        .byte   $05         ; Fuseball 500
                        .byte   $07         ; Fuseball 750

; Not sure what this stuff is.
; It's blocks of 16.bytes, one of which is copied to $c0-$cf by the loop
; beginning around $ccc7.

; Block 0
loccb01:                .byte   $00, $00, $00, $00, $00, $00, $00, $00, $35, $38, $00, $00, $00, $00, $00, $00

; Block 1
                        .byte   $00, $00, $47, $4a, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00

; Block 2
                        .byte   $00, $00, $00, $00, $0d, $10, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00

; Block 3
                        .byte   $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $65, $68, $00, $00, $00, $00

; Block 4
                        .byte   $00, $00, $00, $00, $00, $00, $21, $32, $00, $00, $00, $00, $00, $00, $00, $00

; Block 5
                        .byte   $13, $1a, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00

; Block 6
                        .byte   $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $53, $56, $00, $00, $00, $00

; Block 7
                        .byte   $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $59, $5c, $00, $00, $00, $00

; Block 8
                        .byte   $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $3b, $3e

; Block 9
                        .byte   $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $41, $44, $00, $00

; Block a
                        .byte   $4d, $50, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00

; Block b
                        .byte   $5f, $62, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00

; Block c
                        .byte   $00, $00, $00, $00, $00, $00, $00, $00, $00, $00
loccbcb:                .byte   $6d
loccbcc:                .byte   $6d, $00
loccbce:                .byte   $00, $00, $00
                        .byte   $c0, $08, $04, $10, $00, $00, $a6, $20, $f8, $04, $00, $00, $40, $08, $04, $10
                        .byte   $00, $00, $a6, $20, $fe, $04, $00, $00, $10, $01, $07, $20, $00, $00, $a2, $01
                        .byte   $f8, $20, $00, $00, $08, $04, $20, $0a, $08, $04, $01, $09, $10, $0d, $04, $0c
                        .byte   $00, $00, $08, $04, $00, $0a, $68, $04, $00, $09, $68, $12, $ff, $09, $00, $00
                        .byte   $40, $01, $00, $01, $40, $01, $ff, $40, $30, $01, $ff, $30, $20, $01, $ff, $20
                        .byte   $18, $01, $ff, $18, $14, $01, $ff, $14, $12, $01, $ff, $12, $10, $01, $ff, $10
                        .byte   $00, $00, $a8, $93, $00, $02, $00, $00, $0f, $04, $00, $01, $00, $00, $a2, $04
                        .byte   $40, $01, $00, $00, $00, $03, $02, $09, $00, $00, $08, $03, $ff, $09, $00, $00
                        .byte   $80, $01, $e8, $05, $00, $00, $a1, $01, $01, $05, $00, $00, $01, $08, $02, $10
                        .byte   $00, $00, $86, $20, $00, $04, $00, $00, $18, $04, $00, $ff, $00, $00, $af, $04
                        .byte   $00, $ff, $00, $00, $c0, $02, $ff, $ff, $00, $00, $28, $02, $00, $f0, $00, $00
                        .byte   $10, $0b, $01, $40, $00, $00, $86, $40, $00, $0b, $00, $00, $20, $80, $00, $03
                        .byte   $00, $00, $a8, $40, $f8, $06, $00, $00, $b0, $02, $00, $ff, $00, $00, $c8, $01
                        .byte   $02, $ff, $c8, $01, $02, $ff, $00, $00, $c0, $01, $00, $01, $00, $00, $00

locccb0:                lda     #$5f        ; 13 1a 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                        jmp     locccc3

locccb5:                lda     #$0f        ; 0 0 0 0 0 0 0 0 35 38 0 0 0 0 0 0
                        bne     locccc3

locccb9:                lda     #$4f        ; 0 0 0 0 0 0 21 32 0 0 0 0 0 0 0 0
                        bne     locccc3

locccbd:                lda     #$8f        ; 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3b 3e
                        bne     locccc3

locccc1:                lda     #$1f        ; 0 0 47 4a 0 0 0 0 0 0 0 0 0 0 0 0
locccc3:                bit     $05
                        bpl     loccce9
locccc7:                stx     $31
                        sty     $32
loccccb:                tay
loccccc:                ldx     #$0f
locccce:                lda     loccb01,y
                        beq     loccce1
                        stx     $bf
                        sta     $c0,x
                        lda     #$01
                        sta     $e0,x
                        sta     $f0,x
                        lda     #$ff
                        sta     $bf
loccce1:                dey
                        dex
                        bpl     locccce
                        ldx     $31
                        ldy     $32
loccce9:                rts

locccea:                lda     #$2f        ; 0 0 0 0 0d 10 0 0 0 0 0 0 0 0 0 0
                        bne     locccc3

locccee:                lda     #$6f        ; 0 0 0 0 0 0 0 0 0 0 53 56 0 0 0 0
                        bne     locccc3

locccf2:                lda     #$7f        ; 0 0 0 0 0 0 0 0 0 0 59 5c 0 0 0 0
                        bne     locccc3

locccf6:                lda     #$9f        ; 0 0 0 0 0 0 0 0 0 0 0 0 41 44 0 0
                        bne     locccc3

locccfa:                lda     #$af        ; 4d 50 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                        bne     locccc7

locccfe:                lda     #$bf        ; 5f 62 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                        bne     locccc3

loccd02:                lda     #$3f        ; 0 0 0 0 0 0 0 0 0 0 65 68 0 0 0 0
                        bne     locccc3

sound_pulsar:           lda     #$cf        ; 0 0 0 0 0 0 0 0 0 0 6d 6d 0 0 0 0
                        bne     locccc3

loccd0a:                ldx     #$0f
loccd0c:                lda     $c0,x
                        beq     loccd8e
                        cpx     $bf
                        beq     loccd8e
                        dec     $e0,x
                        bne     loccd8e
                        dec     $f0,x
                        bne     loccd54
loccd1c:                inc     $c0,x
                        inc     $c0,x
                        lda     $c0,x
                        asl     a
                        tay
                        bcs     loccd36
                        lda     loccbcb,y
                        sta     $d0,x
                        lda     loccbce,y
                        sta     $f0,x
                        lda     loccbcc,y
                        clv
                        bvc     loccd43
loccd36:                lda     loccccb,y
                        sta     $d0,x
                        lda     locccce,y
                        sta     $f0,x
                        lda     loccccc,y
loccd43:                sta     $e0,x
                        bne     loccd51
                        sta     $c0,x
                        lda     $d0,x
                        beq     loccd51
                        sta     $c0,x
                        bne     loccd1c
loccd51:                clv
                        bvc     loccd7f
loccd54:                asl     a
                        tay
                        bcs     loccd63
                        lda     loccbcc,y
                        sta     $e0,x
                        lda     loccbcc+1,y
                        clv
                        bvc     loccd6b
loccd63:                lda     loccccc,y
                        sta     $e0,x
                        lda     loccccc+1,y
loccd6b:                ldy     $d0,x
                        clc
                        adc     $d0,x
                        sta     $d0,x
                        txa
                        lsr     a
                        bcc     loccd7f
                        tya
                        eor     $d0,x
                        and     #$f0
                        eor     $d0,x
                        sta     $d0,x
loccd7f:                lda     $d0,x
                        cpx     #$08
                        bcc     loccd8b
                        sta     spinner_cabtyp,x
                        clv
                        bvc     loccd8e
loccd8b:                sta     pokey1,x
loccd8e:                dex
                        bmi     loccd94
                        jmp     loccd0c
loccd94:                rts

; Commented in another disassembly as pokey initialization

loccd95:                lda     #$00
                        sta     $60cf
                        sta     $60df
                        sta     $0720
                        ldx     #$04
                        lda     pokey1_rand
                        ldy     pokey2_rand
loccda8:                cmp     pokey1_rand
                        bne     loccdb0
                        cpy     pokey2_rand
loccdb0:                beq     loccdb7
                        sta     $0720
                        ldx     #$00
loccdb7:                dex
                        bpl     loccda8
                        lda     #$07
                        sta     $60cf
                        sta     $60df
                        ldx     #$07
                        lda     #$00
loccdc6:                sta     pokey1,x
                        sta     pokey2,x
                        sta     $c0,x
                        sta     $d0,x
                        dex
                        bpl     loccdc6
                        lda     #$00
                        sta     spinner_cabtyp
                        lda     #$00
                        sta     zap_fire_starts
                        rts

;-------------------------------------------------------------------------------------------------------
;
; Pre-calced display code for the score, high score, level, number, ie: stuff at the top of the screen.
; First this pre-calced display code is copied into the $2f60/video_data buffer, then the variable pieces such
; as the score, number of remaining ships, etc, is copied into (on top of) this static copy.
;
; At the top of this table you will find the offsets for where each important piece of info, such
; as the player 1 score, player 2 score, etc, lives within the buffer.  This is how it knows where
; to go and overwrite the static data with the current game info.
;
; Note:  The total length cannot exceed $a0 bytes since much of the math does not account for
;        the possibility of an overflow past $3000, and who knows 
;
;-------------------------------------------------------------------------------------------------------

; Where to find the scaling factors for each player's score: see $a991

ScaleOffset:            .byte   p1scaleoff-hdr_template+1       ; Where to find the scale factor for P1 score
                        .byte   p2scaleoff-hdr_template+1       ; Where to find the scale factor for P1 score

; Where to find the code to draw each player's remaining ships: see $a997

ShipsLeftOffset:        .byte   p1shipoff-hdr_template          ; Index into video_data to draw P1 remaining ships
                        .byte   p2shipoff-hdr_template          ; Index into video_data to draw P2 remaining ships

; Where to find the code to draw the scores and high score: see $a9cb

ScoresOffset:           .byte   p1scoreoff-hdr_template         ; Index into video_data to draw P1 score
                        .byte   p2scoreoff-hdr_template         ; Index into video_data to draw P2 score

HiScoreOffset:          .byte   hiscoreoff-hdr_template         ; Index into video_data to draw High Score

; Offset of high-score initials code from video_data, used at $a929

hsinitidx:              .byte   hsinitoff-hdr_template          ; Index into video_data to draw HS Initials

hdr_template: 
                        .byte $00,$71           ; vscale  b=1 l=0                   [ Set scaling ]
                        .byte $c5,$68           ; vstat   z=12 c=5 sparkle=1        [ Color 5 ]
                        .byte $40,$80           ; vcenter                           [ Go to center of screen]
                        .byte $6c,$01,$40,$1e   ; vdraw   x=-449 y=+364 z=off       [ Move to -449,364 with pen off]

p1scaleoff:             .byte $00,$71           ; vscale  b=0 l=0                   [ Clear scaling ? ]

p1scoreoff:             
                      
                        .byte $b4,$a8           ; vjsr                              [ Draw _ ]
                        .byte $b4,$a8           ; vjsr                              [ Draw _ ]
                        .byte $b4,$a8           ; vjsr                              [ Draw _ ]
                        .byte $b4,$a8           ; vjsr                              [ Draw _ ]
                        .byte $b4,$a8           ; vjsr                              [ Draw _ ]
                        .byte $65,$a8           ; vjsr                              [ Draw 0 ]
                        
                        .byte $00,$00,$70,$1f   ; vldraw  x=-144 y=+0 z=off         [ Go left -144 ]
                        .byte $00,$71           ; vscale  b=1 l=0                   [ Set scaling to 1 ]
                        .byte $00,$58, $c1,$68  ; vsdraw  x=+0 y=-16 z=off          [ Move down 16 ]

p1shipoff:              .byte $3f,$a9           ; vjsr                              [ Space for Ship ]
                        .byte $3f,$a9           ; vjsr                              [ Space for Ship ]
                        .byte $3f,$a9           ; vjsr                              [ Space for Ship ]
                        .byte $3f,$a9           ; vjsr                              [ Space for Ship ]
                        .byte $3f,$a9           ; vjsr                              [ Space for Ship ]
                        .byte $3f,$a9           ; vjsr                              [ Space for Ship ]

                        .byte $30,$00,$D0,$1f   ; vldraw  x=-48 y=+48 z=off         [ Pen off, 48 left, 48 up ]
                        .byte $c5,$68           

hiscoreoff:             

                        .byte $b4,$a8           ; vjsr                              [ Draw _ ]
                        .byte $b4,$a8           ; vjsr                              [ Draw _ ]
                        .byte $b4,$a8           ; vjsr                              [ Draw _ ]
                        .byte $b4,$a8           ; vjsr                              [ Draw _ ]
                        .byte $b4,$a8           ; vjsr                              [ Draw _ ]
                        .byte $b4,$a8           ; vjsr                              [ Draw _ ]
                        
                        .byte $dc,$1f,$0,$00    ; vldraw  x=-36 y=+0 z=off          [ Pen off, 36 left, 0 down ]
                        .byte $c7,$68           ; vstat   z=12 c=7 sparkle=1        [ Color 7 ]
                        
levelnumoffset:         .byte $b4,$a8           ; vjsr                              [ Draw _ ] Level Number
                        .byte $b4,$a8           ; vjsr                              [ Draw _ ]
                        
                        .byte $c5,$68           ; vstat   z=12 c=5 sparkle=1        [ Color 5 ]
                        .byte $24,$00,$e8,$1f   ; vldraw  x=-24 y=+36 z=off         [ Pen off, 24 left, 36 up ]
                        
hsinitoff:              .byte $b4,$a8           ; vjsr                              [ Draw _ ] Hi Score Initials
                        .byte $b4,$a8           ; vjsr                              [ Draw _ ]
                        .byte $b4,$a8           ; vjsr                              [ Draw _ ]
                        
                        .byte $00,$71           ; vscale  b=1 l=0                   [ Set scaling]
                        .byte $e0,$1f,$28,$00   ; vldraw  x=+40 y=-32 z=off         [ Pen off, 40 right, 32 up]

p2scaleoff:             .byte $00,$71           ; vscale  b=1 l=0
                        
p2scoreoff:             

                        .byte $b4,$a8           ; vjsr                              [ Draw _ ]
                        .byte $b4,$a8           ; vjsr                              [ Draw _ ]
                        .byte $b4,$a8           ; vjsr                              [ Draw _ ]
                        .byte $b4,$a8           ; vjsr                              [ Draw _ ]
                        .byte $b4,$a8           ; vjsr                              [ Draw _ ]
                        .byte $65,$a8           ; vjsr                              [ Draw _ ]

                        .byte $00,$00,$70,$1f   ; vldraw  x=-144 y=+0 z=off
                        .byte $00,$71           ; vscale  b=1 l=0
                        .byte $00,$58           ; vjsr
                        .byte $c1,$68           ; vstat   z=12 c=1 sparkle=1        [ Color 1 - Same as Player ]
                        
p2shipoff:              .byte $3f,$a9           ; vjsr $127e                        [ Space for Ship ]
                        .byte $3f,$a9           ; vjsr $127e                        [ Space for Ship ]
                        .byte $3f,$a9           ; vjsr $127e                        [ Space for Ship ]
                        .byte $3f,$a9           ; vjsr $127e                        [ Space for Ship ]
                        .byte $3f,$a9           ; vjsr $127e                        [ Space for Ship ]
                        .byte $3f,$a9           ; vjsr $127e                        [ Space for Ship ]

hdr_template_end = *-1

hdr_template_len:       .byte $55                               ; Length when in demo mode?
                        .byte hdr_template_end-hdr_template     ; Length of header template canned data 
.assert (hdr_template_end-hdr_template < $a0)                   ; Otherwise would run off past $3000 when copied starting at $2f60

; Tables of addresses of areas to assemble various video sequences into.
; 
; These are the 6502-seen addresses; see the vjmp tables below, which hold
; the corresponding vector-generator addresses in the form of a vjmp.
; These tables are used at b2c1 and b2e1; which one is used is based on
; the values in the dblbuf_flg vector.

; Table A

dblbuf_addr_A:          .word       $2006
                        .word       $2202
                        .word       $240c
locce6e:                .word       $2692
                        .word       $2900
                        .word       $2a56
                        .word       $2cd8
                        .word       $2dbe
                        .word       $2e24

; Table B

dblbuf_addr_B:          .word       $2104
                        .word       $2306
                        .word       $254e
                        .word       $27c8
                        .word       $29aa
                        .word       $2b96
locce86:                .word       $2d4a
                        .word       $2df0
                        .word       $2ea6

; This table gives the location to stuff the vjmp from the tables below
; depending on which way the double-buffering flag is set.

dblbuf_vjsr_loc:        .word       $2004
                        .word       $2200
                        .word       $240a
                        .word       $2690
                        .word       $28fe
                        .word       $2a54
                        .word       $2cd6
                        .word       $2dbc
                        .word       $2e22

; These tables contain vjmp instructions corresponding to Tables A and B
; above.  The addresses here are the vector-generator-visible addresses
; that refer to the same video RAM as the 6502-visible addresses above.

; Table C

dblbuf_vjmp_C:          .word 0E003h
                        .byte    1
                        .byte 0E1h 
                        .byte    6
                        .byte 0E2h 
                        .byte  49h 
                        .byte 0E3h 
                        .byte  80h 
                        .byte 0E4h 
                        .byte  2Bh 
                        .byte 0E5h 
                        .byte  6Ch 
                        .byte 0E6h 
                        .byte 0DFh 
                        .byte 0E6h 
                        .byte  12h
                        .byte 0E7h 

; Table D

dblbuf_vjmp_D:          .byte  82h
                        .byte 0E0h 
                        .byte  83h 
                        .byte 0E1h 
                        .byte 0A7h 
                        .byte 0E2h 
                        .byte 0E4h 
                        .byte 0E3h 
                        .byte 0D5h 
                        .byte 0E4h 
                        .byte 0CBh 
                        .byte 0E5h 
                        .byte 0A5h 
                        .byte 0E6h 
                        .byte 0F8h 
                        .byte 0E6h 
                        .byte  53h 
                        .byte 0E7h 

; I don't know what these three are used for.  The first jumps to a
; routine that calls the double-buffered stuff (in an order different
; from that of the above tables); the second calls just one of them,
; and the third does nothing.  All three halt after doing their things.
; But see $b1bc.

loccEC2:                .byte 0DAh                          
loccEC3:                .byte 0EEh 
loccEC4:                .byte 0E4h                            
loccEC5:                .byte 0EEh 
locCEC6:                .byte 0E6h 
locCEC7:                .byte 0EEh                             

graphic_table: 
                        .byte  61h
                        .byte 0AAh 
                        .byte  7Ch 
                        .byte 0AAh 
                        .byte  91h 
                        .byte 0AAh 
                        .byte 0ADh 
                        .byte 0AAh 
                        .byte 0CAh 
                        .byte 0AAh 
                        .byte  14h
                        .byte 0ABh 
                        .byte  6Fh 
                        .byte 0ABh 
                        .byte 0C0h 
                        .byte 0ABh 
                        .byte  15h
                        .byte 0ACh 
                        .byte  66h 
                        .byte 0ACh 
                        .byte  7Dh 
                        .byte 0ACh 
                        .byte  94h 
                        .byte 0ACh 
                        .byte 0ABh 
                        .byte 0ACh 
                        .byte 0D8h 
                        .byte 0ACh 
                        .byte 0FAh 
                        .byte 0ACh 
                        .byte  0Dh
                        .byte 0ADh 
                        .byte  20h
                        .byte 0ADh 
                        .byte  39h 
                        .byte 0ADh 
                        .byte  51h 
                        .byte 0ADh 
                        .byte  6Ah 
                        .byte 0ADh 
                        .byte  8Ch 
                        .byte 0ADh 
                        .byte  8Ah 
                        .byte 0ADh 
                        .byte  88h 
                        .byte 0ADh 
                        .byte  86h 
                        .byte 0ADh 
                        .byte  84h 
                        .byte 0ADh 
                        .byte  82h 
                        .byte 0ADh 
                        .byte  86h 
                        .byte 0ADh 
                        .byte  8Ah 
                        .byte 0ADh 
                        .byte  8Ch 
                        .byte 0ADh 
                        .byte 0D7h 
                        .byte 0ADh 
                        .byte 0C2h 
                        .byte 0ADh 
                        .byte 0C5h 
                        .byte 0ADh 
                        .byte 0C8h 
                        .byte 0ADh 
                        .byte 0CBh 
                        .byte 0ADh 
                        .byte 0CEh 
                        .byte 0ADh 
                        .byte 0D1h 
                        .byte 0ADh 
                        .byte 0D4h 
                        .byte 0ADh 
                        .byte 0C2h 
                        .byte 0ACh 
                        .byte 0CBh 
                        .byte 0ACh 
                        .byte  35h 
                        .byte 0AEh 
                        .byte  59h 
                        .byte 0AEh 
                        .byte  7Eh 
                        .byte 0AEh 
                        .byte 0A2h 
                        .byte 0AEh 
                        .byte 0C5h 
                        .byte 0AEh 
                        .byte 0CBh 
                        .byte 0AEh 
                        .byte 0D2h 
                        .byte 0AEh 

loccf24:                ldx     #$02
loccf26:                .byte $ad, $08, $00                 ; BUGBUG Non-zero-page, was 'lda     zap_fire_shadow'
                        cpx     #$01
                        beq     loccf30
                        bcs     loccf31
                        lsr     a
loccf30:                lsr     a
loccf31:                lsr     a
                        lda     $0d,x
                        and     #$1f
                        bcs     loccf6f
                        beq     loccf4a
                        cmp     #$1b
                        bcs     loccf48
                        tay
                        lda     $07
                        and     #$07
                        cmp     #$07
                        tya
                        bcc     loccf4a
loccf48:                sbc     #$01
loccf4a:                sta     $0d,x
                        .byte $ad, $08, $00                 ; BUGBUG Non-zero-page, was 'lda     zap_fire_shadow'lda     zap_fire_shadow
                        and     #$08
                        bne     loccf57
                        lda     #$f0
                        sta     $0c
loccf57:                lda     $0c
                        beq     loccf63
                        dec     $0c
                        lda     #$00
                        sta     $0d,x
                        sta     $10,x
loccf63:                clc
                        lda     $10,x
                        beq     loccf8b
                        dec     $10,x
                        bne     loccf8b
                        sec
                        bcs     loccf8b
loccf6f:                cmp     #$1b
                        bcs     loccf7c
                        lda     $0d,x
                        adc     #$20
                        bcc     loccf4a
                        beq     loccf7c
                        clc
loccf7c:                lda     #$1f
                        bcs     loccf4a
                        sta     $0d,x
                        lda     $10,x
                        beq     loccf87
                        sec
loccf87:                lda     #$78
                        sta     $10,x
loccf8b:                bcc     loccfb7
                        lda     #$00
                        cpx     #$01
                        bcc     loccfa9
                        beq     loccfa1

; coin in right slot

                        lda     coinage_shadow
                        and     #$0c ; right slot multiplier
                        lsr     a
                        lsr     a
                        beq     loccfa9 ; branch if x1
                        adc     #$02
                        bne     loccfa9

; coin in left slot

loccfa1:                lda     coinage_shadow
                        and     #$10 ; left slot multiplier
                        beq     loccfa9
                        lda     #$01

; At this point, A holds the post-multiplier coin count, minus 1.

loccfa9:                sec
                        pha
                        adc     coin_string
                        sta     coin_string
                        pla
                        sec
                        adc     uncredited
                        sta     uncredited
                        inc     $13,x
loccfb7:                dex
                        bmi     loccfbd
                        jmp     loccf26
loccfbd:                lda     coinage_shadow
                        lsr     a                   ; extract bonus coins bits
                        lsr     a
                        lsr     a
                        lsr     a
                        lsr     a
                        tay
                        lda     coin_string
                        sec
                        sbc     loccfd9,y
                        bmi     loccfe1
                        sta     coin_string
                        inc     $18
                        cpy     #$03                ; setting for 2 bonus
                        bne     loccfe1
                        inc     $18
                        bne     loccfe1

; Bonus coins table (see code just above)

loccfd9:                .byte   $7f                 ; no bonus coins
                        .byte   $02                 ; 1 bonus for each 2
                        .byte   $04                 ; 1 bonus for each 4
                        .byte   $04                 ; 2 bonus for each 4
                        .byte   $05                 ; 1 bonus for each 5
                        .byte   $03                 ; 1 bonus for each 3
                        .byte   $7f                 ; no bonus coins
                        .byte   $7f                 ; no bonus coins

loccfe1:                lda     coinage_shadow
                        and     #$03                ; coins-to-credits bits (XOR $02)
                        tay
                        beq     locd002             ; branch if free play

; A now 1 for 1c/2c, 2 for 1c/1c, 3 for 2c/1c

                        lsr     a
                        adc     #$00
                        eor     #$ff

; A now fe for 1c/*, fd for 2c/1c

                        sec
                        adc     uncredited
                        bcs     loccffa
                        adc     $18
                        bmi     locd004
                        sta     $18
                        lda     #$00
loccffa:                cpy     #$02
                        bcs     locd000
                        inc     credits
locd000:                inc     credits
locd002:                sta     uncredited
locd004:                lda     $07
                        lsr     a
                        bcs     locd030
                        ldy     #$00
                        ldx     #$02
locd00d:                lda     $13,x
                        beq     locd01a
                        cmp     #$10
                        bcc     locd01a
                        adc     #$ef
                        iny
                        sta     $13,x
locd01a:                dex
                        bpl     locd00d
                        tya
                        bne     locd030
                        ldx     #$02
locd022:                lda     $13,x
                        beq     locd02d
                        clc
                        adc     #$ef
                        sta     $13,x
                        bmi     locd030
locd02d:                dex
                        bpl     locd022
locd030:                rts

;---------------------------------------------------------------------------------
; Message Tables
;---------------------------------------------------------------------------------

msgs_en:

MsgEnGameOver:          .WORD   xposMsgGameOver              ; "GAME OVER"
MsgEnPlayer:            .WORD   xposMsgPlayer                ; "PLAYER "
MsgEnPlayer2:           .WORD   xposMsgPlayer                ; "PLAYER "
MsgEnStart:             .WORD   xposMsgStart                 ; "PRESS START"
MsgEnPlay:              .WORD   xposMsgPlay                  ; "PLAY"
MsgEnInitials:          .WORD   xPosMsgInitials              ; "ENTER YOUR INITIALS"
MsgEnSpinKnob:          .WORD   xposMsgSpinKnob              ; "SPIN KNOB TO CHANGE"
MsgEnPressFire:         .WORD   xposMsgPressFire             ; "PRESS FIRE TO SELECT"
MsgEnHiScores:          .WORD   xposMsgHiScores              ; "HIGH SCORES"
MsgEnRanking:           .WORD   xposMsgRanking               ; "RANKING FROM 1 TO "
MsgEnRateSelf:          .WORD   xposMsgRateSelf              ; "RATE YOURSELF"
MsgEnNovice:            .WORD   xposMsgNovice                ; "NOVICE"
MsgEnExpert:            .WORD   xposMsgExpert                ; "EXPERT"
MsgEnBonus:             .WORD   xposMsgBonus                 ; "BONUS"
MsgEnTime:              .WORD   xposMsgTime                  ; "TIME"
MsgEnLevel:             .WORD   xposMsgLevel                 ; "LEVEL"
MsgEnHole:              .WORD   xposMsgHole                  ; "HOLE"
MsgEnInsCoin:           .WORD   xposMsgInsCoin               ; "INSERT COINS"
MsgEnFreePlay:          .WORD   xposMsgFreePlay              ; "FREE PLAY"
MsgEn1Coin2Crd:         .WORD   xposMsg1Coin2Crd             ; "1 COIN 2 PLAYS"
MsgEn1Coin1Crd:         .WORD   xposMsg1Coin1Crd             ; "1 COIN 1 PLAY"
MsgEn2Coin1Crd:         .WORD   xposMsg2Coin1Crd             ; "2 COINS 1 PLAY"
MsgEnAtari:             .WORD   xposMsgAtari                 ; "(c) MCMLXXX ATARI"
MsgEnCredits:           .WORD   xposMsgCredits               ; "CREDITS "
MsgEnBonusSpc:          .WORD   xposMsgBonusSpc              ; "BONUS "
MsgEn2CrdMin:           .WORD   xposMsg2CrdMin               ; "2 CREDIT MINIMUM"
MsgEnBonusEv:           .WORD   xposMsgBonusEv               ; "BONUS EVERY "
MsgEnAvoidSpk:          .WORD   xposMsgAvoidSpk              ; "AVOID SPIKES"
MsgEnLevelNS:           .WORD   xposMsgLevelNS               ; "LEVEL"
MsgEnRecharge:          .WORD   xposMsgRecharge              ; "SUPERZAPPER RECHARGE"

; Byte offset table - How far into the table each message is.  This distance
; will be the same for all languages so long as they are all kept in sync.

ibMsgGameOver  = MsgEnGameOver  - msgs_en
ibMsgPlayer    = MsgEnPlayer    - msgs_en
ibMsgPlayer2   = MsgEnPlayer2   - msgs_en
ibMsgStart     = MsgEnStart     - msgs_en
ibMsgPlay      = MsgEnPlay      - msgs_en
ibMsgInitials  = MsgEnInitials  - msgs_en
ibMsgSpinKnob  = MsgEnSpinKnob  - msgs_en
ibMsgPressFire = MsgEnPressFire - msgs_en
ibMsgHiScores  = MsgEnHiScores  - msgs_en
ibMsgRanking   = MsgEnRanking   - msgs_en
ibMsgRateSelf  = MsgEnRateSelf  - msgs_en
ibMsgNovice    = MsgEnNovice    - msgs_en
ibMsgExpert    = MsgEnExpert    - msgs_en
ibMsgBonus     = MsgEnBonus     - msgs_en
ibMsgTime      = MsgEnTime      - msgs_en
ibMsgLevel     = MsgEnLevel     - msgs_en
ibMsgHole      = MsgEnHole      - msgs_en
ibMsgInsCoin   = MsgEnInsCoin   - msgs_en
ibMsgFreePlay  = MsgEnFreePlay  - msgs_en
ibMsg1Coin2Crd = MsgEn1Coin2Crd - msgs_en
ibMsg1Coin1Crd = MsgEn1Coin1Crd - msgs_en
ibMsg2Coin1Crd = MsgEn2Coin1Crd - msgs_en
ibMsgAtari     = MsgEnAtari     - msgs_en
ibMsgCredits   = MsgEnCredits   - msgs_en
ibMsgBonusSpc  = MsgEnBonusSpc  - msgs_en
ibMsg2CrdMin   = MsgEn2CrdMin   - msgs_en
ibMsgBonusEv   = MsgEnBonusEv   - msgs_en
ibMsgAvoidSpk  = MsgEnAvoidSpk  - msgs_en
ibMsgLevelNS   = MsgEnLevelNS   - msgs_en
ibMsgRecharge  = MsgEnRecharge  - msgs_en

.if !REMOVE_LANGUAGES
msgs_fr:                .WORD   xposMsgGameOverFr            ; "FIN DE PARTIE"
                        .WORD   xposMsgPlayerFr              ; "JOUEUR "
                        .WORD   xposMsgPlayerFr              ; "JOUEUR "
                        .WORD   xposMsgStartFr               ; "APPUYEZ SUR START"
                        .WORD   xposMsgPlayFr                ; "JOUEZ"
                        .WORD   xPosMsgInitialsFr            ; "SVP ENTREZ VOS INITIALES"
                        .WORD   xposMsgSpinKnobFr            ; "TOURNEZ LE BOUTON POUR CHANGER"
                        .WORD   xposMsgPressFireFr           ; "POUSSEZ FEU QUAND CORRECTE"
                        .WORD   xposMsgHiScoresFr            ; "MEILLEURS SCORES"
                        .WORD   xposMsgRankingFr             ; "PLACEMENT DE 1 A "
                        .WORD   xposMsgRateSelfFr            ; "EVALUEZ-VOUS"
                        .WORD   xposMsgNovice                ; "NOVICE"
                        .WORD   xposMsgExpert                ; "EXPERT"
                        .WORD   xposMsgBonus                 ; "BONUS"
                        .WORD   xposMsgTimeFr                ; "DUREE"
                        .WORD   xposMsgLevelFr               ; "NIVEAU"
                        .WORD   xposMsgHoleFr                ; "TROU"
                        .WORD   xposMsgInsCoinFr             ; "INTRODUIRE LES PIECES"
                        .WORD   xposMsgFreePlay              ; "FREE PLAY"
                        .WORD   xposMsg1Coin2CrdFr           ; "1 PIECE 2 JOUEURS"
                        .WORD   xposMsg1Coin1CrdFr           ; "1 PIECE 1 JOUEUR"
                        .WORD   xposMsg2Coin1CrdFr           ; "2 PIECES 1 JOUEUR"
                        .WORD   xposMsgAtari                 ; "(c) MCMLXXX ATARI"
                        .WORD   xposMsgCredits               ; "CREDITS "
                        .WORD   xposMsgBonusSpc              ; "BONUS "
                        .WORD   xposMsg2CrdMinFr             ; "2 JEUX MINIMUM"
                        .WORD   xposMsgBonusEvFr             ; "BONUS CHAQUE "
                        .WORD   xposMsgAvoidSpkFr            ; "ATTENTION AUX LANCES"
                        .WORD   xposMsgLevelNSFr             ; "NIVEAU"
                        .WORD   xposMsgRecharge              ; "SUPERZAPPER RECHARGE"

msgs_de:                .WORD   xposMsgGameOverGer           ; "SPIELENDE"
                        .WORD   xposMsgPlayerGer             ; "SPIELER "
                        .WORD   xposMsgPlayerGer             ; "SPIELER "
                        .WORD   xposMsgStartGer              ; "START DRUECKEN"
                        .WORD   xposMsgPlayGer               ; "SPIEL"
                        .WORD   xPosMsgInitialsGer           ; "GEBEN SIE IHRE INITIALEN EIN"
                        .WORD   xposMsgSpinKnobGer           ; "KNOPF DREHEN ZUM WECHSELN"
                        .WORD   xposMsgPressFireGer          ; "FIRE DRUECKEN WENN RICHTIG"
                        .WORD   xposMsgHiScoresGer           ; "HOECHSTZAHLEN"
                        .WORD   xposMsgRankingGer            ; "RANGLISTE VON 1 ZUM "
                        .WORD   xposMsgRateSelfGer           ; "SELBST RECHNEN"
                        .WORD   xposMsgNoviceGer             ; "ANFAENGER"
                        .WORD   xposMsgExpertGer             ; "ERFAHREN"
                        .WORD   xposMsgBonus                 ; "BONUS"
                        .WORD   xposMsgTimeGer               ; "ZEIT"
                        .WORD   xposMsgLevelGer              ; "GRAD"
                        .WORD   xposMsgHoleGer               ; "LOCH"
                        .WORD   xposMsgInsCoinGer            ; "GELD EINWERFEN"
                        .WORD   xposMsgFreePlay              ; "FREE PLAY"
                        .WORD   xposMsg1Coin2CrdGer          ; "1 MUENZ 2 SPIELE"
                        .WORD   xposMsg1Coin1CrdGer          ; "1 MUENZE 1 SPIEL"
                        .WORD   xposMsg2Coin1CrdGer          ; "2 MUENZEN 1 SPIEL"
                        .WORD   xposMsgAtari                 ; "(c) MCMLXXX ATARI"
                        .WORD   xposMsgCreditsGer            ; "KREDITE "
                        .WORD   xposMsgBonusSpc              ; "BONUS "
                        .WORD   xposMsg2CrdMinGer            ; "2 SPIELE MINIMUM"
                        .WORD   xposMsgBonusEvGer            ; "BONUS JEDE "
                        .WORD   xposMsgAvoidSpkGer           ; "SPITZEN AUSWEICHEN"
                        .WORD   xposMsgLevelNSGer            ; "GRAD"
                        .WORD   xposMsgRechargeGer           ; "NEUER SUPERZAPPER"

msgs_es:                .WORD   xposMsgGameOverSpn           ; "JUEGO TERMINADO"
                        .WORD   xposMsgPlayerSpn             ; "JUGADOR "
                        .WORD   xposMsgPlayerSpn             ; "JUGADOR "
                        .WORD   xposMsgStartSpn              ; "PULSAR START"
                        .WORD   xposMsgPlaySpn               ; "JUEGUE"
                        .WORD   xPosMsgInitialsSpn           ; "ENTRE SUS INICIALES"
                        .WORD   xposMsgSpinKnobSpn           ; "GIRE LA PERILLA PARA CAMBIAR"
                        .WORD   xposMsgPressFireSpn          ; "OPRIMA FIRE PARA SELECCIONAR"
                        .WORD   xposMsgHiScoresSpn           ; "RECORDS"
                        .WORD   xposMsgRankingSpn            ; "RANKING DE 1 A "
                        .WORD   xposMsgRateSelfSpn           ; "CALIFIQUESE"
                        .WORD   xposMsgNoviceSpn             ; "NOVICIO"
                        .WORD   xposMsgExpertSpn             ; "EXPERTO"
                        .WORD   xposMsgBonus                 ; "BONUS"
                        .WORD   xposMsgTimeSpn               ; "TIEMPO"
                        .WORD   xposMsgLevelSpn              ; "NIVEL"
                        .WORD   xposMsgHoleSpn               ; "HOYO"
                        .WORD   xposMsgInsCoinSpn            ; "INSERTE FICHAS"
                        .WORD   xposMsgFreePlay              ; "FREE PLAY"
                        .WORD   xposMsg1Coin2CrdSpn          ; "1 MONEDA 2 JUEGOS"
                        .WORD   xposMsg1Coin1CrdSpn          ; "1 MONEDA 1 JUEGO"
                        .WORD   xposMsg2Coin1CrdSpn          ; "2 MONEDAS 1 JUEGO"
                        .WORD   xposMsgAtari                 ; "(c) MCMLXXX ATARI"
                        .WORD   xposMsgCreditsSpn            ; "CREDITOS "
                        .WORD   xposMsgBonusSpc              ; "BONUS "
                        .WORD   xposMsg2CrdMinSpn            ; "2 JUEGOS MINIMO"
                        .WORD   xposMsgBonusEvSpn            ; "BONUS CADA "
                        .WORD   xposMsgAvoidSpkSpn           ; "EVITE LAS PUNTAS"
                        .WORD   xposMsgLevelNSSpn            ; "NIVEL"
                        .WORD   xposMsgRechargeSpn           ; "NUEVO SUPERZAPPER"
.endif

 ;**** Message Tables
 ;
 ; Y-coordinates, colours, and sizes of messages.  X-coordinates vary with
 ; string length and thus with language; they are therefore stored with the
 ; (language-specific) string contents.
 ; Each message has two bytes here.  The first contains the message's
 ; colour in its high nibble, with its size (b value) in its low nibble.
 ; The second byte is the Y coordinate (signed).
 ; See the code at ab14 for more.

aMsgsColorAndYPos:      .BYTE   $51,$56                      ; GAME OVER
                        .BYTE   0,26                         ; PLAYER
                        .BYTE   1,$20                        ; PLAYER
                        .BYTE   $31,$56                      ; PRESS START
                        .BYTE   1,$38                        ; PLAY
                        .BYTE   $31,$B0                      ; ENTER YOUR INITIALS
                        .BYTE   $41,0                        ; SPIN KNOB TO CHANGE
                        .BYTE   $11,$F6                      ; PRESS FIRE TO SELECT
                        .BYTE   $30,$38                      ; HIGH SCORES
                        .BYTE   $31,$CE                      ; RANKING FROM 1 TO
                        .BYTE   $51,$A                       ; RATE YOURSELF
                        .BYTE   $31,$E2                      ; NOVICE
                        .BYTE   $31,$E2                      ; EXPERT
                        .BYTE   $51,$BA                      ; BONUS
                        .BYTE   $51,$98                      ; TIME
                        .BYTE   $51,$D8                      ; LEVEL
                        .BYTE   $51,$C9                      ; HOLE
                        .BYTE   $31,$56                      ; INSERT COINS
                        .BYTE   $51,$80                      ; FREE PLAY
                        .BYTE   $51,$80                      ; 1 COIN 2 PLAYS
                        .BYTE   $51,$80                      ; 1 COIN 1 PLAY
                        .BYTE   $51,$80                      ; 2 COINS 1 PLAY
                        .BYTE   $71,$92                      ; (c) MCMLXXX ATARI
                        .BYTE   $51,$80                      ; CREDITS
                        .BYTE   $31,$B0                      ; BONUS
                        .BYTE   $51,$89                      ; 2 CREDIT MINIMUM
                        .BYTE   $41,$89                      ; BONUS EVERY
                        .BYTE   0,0                          ; AVOID SPIKES
                        .BYTE   $71,$5A                      ; LEVEL
                        .BYTE   $71,$A0                      ; SUPERZAPPER RECHARGE

; Each string is preceded by the X-coordinate at which it should be drawn.
 ; Y-coordinates come from a table at $d122 (they don't vary with string
 ; length and hence don't have to be language-specific; X coordinates do,
 ; so they are attached to the language-specific strings).
 ; See the code at ab14 for more.

 ;                    "GAME OVER"
xposMsgGameOver:    .BYTE $E5                          
sMsgGameOver:       .BYTE 34,22,46,30,0,50,64,30,184

.if !REMOVE_LANGUAGES
 ;                    "FINE DE PARTIE"
xposMsgGameOverFr:  .BYTE $D9                          
sMsgGameOverFr:     .BYTE 32,38,48,0,28,30,0,52
                     .BYTE 22,56,60,38,158

 ;                    "SPIELENDE"
xposMsgGameOverGer: .BYTE $E5                          
sMsgGameOverGer:    .BYTE 58,52,38,30,44,30,48,28,158

 ;                    "JUEGO TERMINADO"
xposMsgGameOverSpn: .BYTE $D3                          
                    .BYTE 40,62,30,34,50,0,60,30,56
                     .BYTE 46,38,48,22,28,178
.endif

 ;                    "PLAYER "
xposMsgPlayer:      .BYTE $CD                                                                                
sMsgPlayer:         .BYTE 52,44,22,70,30,56,128

.if !REMOVE_LANGUAGES
 ;                    "JOUEUR"
xposMsgPlayerFr:    .BYTE $C6                          
                                                        
sMsgPlayerFr:       .BYTE 40,50,62,30,62,56,128

 ;                    "SPIELER"
xposMsgPlayerGer:   .BYTE $C6                          
                                                        
sMsgPlayerGer:      .BYTE 58,52,38,30,44,30,56,128

 ;                    "JUGADOR"
xposMsgPlayerSpn:   .BYTE $C6                          
                                                        
sMsgPlayerSpn:      .BYTE 40,62,34,22,28,50,56,128
.endif

 ;                    "PRESS START"
xposMsgStart:       .BYTE $DF                          
sMsgStart:          .BYTE 52,56,30,58,58,0,58,60,22
                     .BYTE 56,188

.if !REMOVE_LANGUAGES
 ;                    "APPUYEZ SUR START"
xposMsgStartFr:     .BYTE $CD                          
sMsgStartFr:        .BYTE 22,52,52,62,70,30,72,0,58
                     .BYTE 62,56,0,58,60,22,56,188

 ;                    "START DRUECKEN"
xposMsgStartGer:    .BYTE $D6                          
sMsgStartGer:       .BYTE 58,60,22,56,60,0,28,56,62
                     .BYTE 30,26,42,30,176

 ;                    "PULSAR START"
xposMsgStartSpn:    .BYTE $DC                          
sMsgStartSpn:       .BYTE 52,62,44,58,22,56,0,58,60
                     .BYTE 22,56,188
.endif

 ;                    "PLAY"
xposMsgPlay:        .BYTE $F4                          
sMsgPlay:           .BYTE 52,44,22,198

.if !REMOVE_LANGUAGES
 ;                    "JOUEZ"
xposMsgPlayFr:      .BYTE $F1 ; �                      
sMsgPlayFr:         .BYTE 40,50,62,30,200

 ;                    "SPIEL"
xposMsgPlayGer:     .BYTE $F1                          
sMsgPlayGer:        .BYTE 58,52,38,30,172

 ;                    "JUEGUE"
xposMsgPlaySpn:     .BYTE $EE                          
sMsgPlaySpn:        .BYTE 40,62,30,34,62,158
.endif

 ;                    "ENTER YOUR INITIALS"
xPosMsgInitials:    .BYTE $C7                          
sMsgInitials:       .BYTE 30,48,60,30,56,0,70,50,62
                     .BYTE 56,0,38,48,38,60,38,22,44
                     .BYTE 186
.if !REMOVE_LANGUAGES
 ;                    "SVP ENTREZ VOS INITIALES"
xPosMsgInitialsFr:  .BYTE $B8                          
sMsgInitialsFr:     .BYTE 58,64,52,0,30,48,60,56,30
                     .BYTE 72,0,64,50,58,0,38,48,38
                     .BYTE 60,38,22,44,30,186

 ;                    "GEBEN SIE IHRE INITIALEN EIN"
xPosMsgInitialsGer: .BYTE $AC                          
sMsgInitialsGer:    .BYTE 34,30,24,30,48,0,58,38,30
                     .BYTE 0,38,36,56,30,0,38,48,38
                     .BYTE 60,38,22,44,30,48,0,30,38
                     .BYTE 176

 ;                    "ENTRE SUS INICIALES"
xPosMsgInitialsSpn: .BYTE $C7                          
sMsgInitialsSpn:    .BYTE 30,48,60,56,30,0,58,62,58
                     .BYTE 0,38,48,38,26,38,22,44,30
                     .BYTE 186
.endif

 ;                    "SPIN KNOB TO CHANGE"
xposMsgSpinKnob:    .BYTE $C7                          
sMsgSpinKnob:       .BYTE 58,52,38,48,0,42,48,50,24
                     .BYTE 0,60,50,0,26,36,22,48,34
                     .BYTE 158

.if !REMOVE_LANGUAGES
 ;                    "TOURNEZ LE BOUTON POUR CHANGER"
xposMsgSpinKnobFr:  .BYTE $A6                          
sMsgSpinKnobFr:     .BYTE 60,50,62,56,48,30,72,0,44
                     .BYTE 30,0,24,50,62,60,50,48,0
                     .BYTE 52,50,62,56,0,26,36,22,48
                     .BYTE 34,30,184

 ;                    "KNOPF DREHEN ZUM WECHSELN"
xposMsgSpinKnobGer: .BYTE $B5                          
sMsgSpinKnobGer:    .BYTE 42,48,50,52,32,0,28,56,30
                     .BYTE 36,30,48,0,72,62,46,0,66
                     .BYTE 30,26,36,58,30,44,176

 ;                    "GIRE LA PERILLA PARA CAMBIAR"
xposMsgSpinKnobSpn: .BYTE $AC                          
sMsgSpinKnobSpn:    .BYTE 34,38,56,30,0,44,22,0,52
                     .BYTE 30,56,38,44,44,22,0,52,22
                     .BYTE 56,22,0,26,22,46,24,38,22
                     .BYTE 184
.endif

 ;                    "PRESS FIRE TO SELECT"
xposMsgPressFire:   .BYTE $C4                          
sMsgPressFire:      .BYTE 52,56,30,58,58,0,32,38,56
                     .BYTE 30,0,60,50,0,58,30,44,30
                     .BYTE 26,188

.if !REMOVE_LANGUAGES
 ;                    "POUSSEZ FEU QUAND CORRECTE"
xposMsgPressFireFr: .BYTE $B2                          
sMsgPressFireFr:    .BYTE 52,50,62,58,58,30,72,0,32
                     .BYTE 30,62,0,54,62,22,48,28,0
                     .BYTE 26,50,56,56,30,26,60,158

 ;                    "FIRE DRUECKEN WENN RICHTIG"
xposMsgPressFireGer: .BYTE $B2                          
sMsgPressFireGer:   .BYTE 32,38,56,30,0,28,56,62,30
                     .BYTE 26,42,30,48,0,66,30,48,48
                     .BYTE 0,56,38,26,36,60,38,162

 ;                    "OPRIMA FIRE PARA SELECCIONAR"
xposMsgPressFireSpn: .BYTE $AC                          
sMsgPressFireSpn:   .BYTE 50,52,56,38,46,22,0,32,38
                     .BYTE 56,30,0,52,22,56,22,0,58
                     .BYTE 30,44,30,26,26,38,50,48,22
                     .BYTE 184
.endif

 ;                    "HIGH SCORES"
xposMsgHiScores:    .BYTE $BC                          
sMsgHiScores:       .BYTE 36,38,34,36,0,58,26,50,56
                     .BYTE 30,186

.if !REMOVE_LANGUAGES
 ;                    "MEILLEURS SCORES"
xposMsgHiScoresFr:  .BYTE $9E                          
sMsgHiScoresFr:     .BYTE 46,30,38,44,44,30,62,56,58
                     .BYTE 0,58,26,50,56,30,186

 ;                    "HOECHSTZAHLEN"
xposMsgHiScoresGer: .BYTE $B0                          
sMsgHiScoresGer:    .BYTE 36,50,30,26,36,58,60,72,22
                     .BYTE 36,44,30,176

 ;                    "RECORDS"
xposMsgHiScoresSpn: .BYTE $D4                          
sMsgHiScoresSpn:    .BYTE $38,$1E,$1A,$32,$38,$1C,$BA
 ;
.endif

xposMsgRanking:     .BYTE $C2                          
sMsgRanking:        .BYTE 56,22,48,42,38,48,34,0,32
                     .BYTE 56,50,46,0,4,0,60,50,128

.if !REMOVE_LANGUAGES
 ;                    "PLACEMENT DE 1 A "
xposMsgRankingFr:   .BYTE $C2                          
sMsgRankingFr:      .BYTE 52,44,22,26,30,46,30,48,60
                     .BYTE 0,28,30,0,4,0,22,128

 ;                    "RANGLISTE VON 1 ZUM "
xposMsgRankingGer:  .BYTE $BC                          
sMsgRankingGer:     .BYTE 56,22,48,34,44,38,58,60,30
                     .BYTE 0,64,50,48,0,4,0,72,62
                     .BYTE 46,128

 ;                    "RANKING DE 1 A "
xposMsgRankingSpn:  .BYTE $C8 ; +                      
sMsgRankingSpn:     .BYTE 56,22,48,42,38,48,34,0,28
                     .BYTE 30,0,4,0,22,128
.endif

 ;                    "RATE YOURSELF"
xposMsgRateSelf:    .BYTE $D9                          
sMsgRateSelf:       .BYTE 56,22,60,30,0,70,50,62,56
                     .BYTE 58,30,44,160

.if !REMOVE_LANGUAGES
 ;                    "EVALUEZ-VOUS"
xposMsgRateSelfFr:  .BYTE $DC                          
sMsgRateSelfFr:     .BYTE 30,64,22,44,62,30,72,76,64
                     .BYTE 50,62,186

 ;                    "SELBST RECHNEN"
xposMsgRateSelfGer: .BYTE $D6                          
sMsgRateSelfGer:    .BYTE 58,30,44,24,58,60,0,56,30
                     .BYTE 26,36,48,30,176

 ;                    "CALIFIQUESE"
xposMsgRateSelfSpn: .BYTE $DF                          
sMsgRateSelfSpn:    .BYTE 26,22,44,38,32,38,54,62,30
                     .BYTE 58,158
.endif

 ;                    "NOVICE"
xposMsgNovice:      .BYTE $AA                          
sMsgNovice:         .BYTE 48,50,64,38,26,158

.if !REMOVE_LANGUAGES
 ;                    "NOVICIO"
xposMsgNoviceSpn:   .BYTE $AA                          
sMsgNoviceSpn:      .BYTE $30,$32,$40,$26,$1A,$26,$B2

 ;                    "ANFAENGER"
xposMsgNoviceGer:   .BYTE $AA                          
sMsgNoviceGer:      .BYTE 22,48,32,22,30,48,34,30,184
.endif

 ;                    "EXPERT"
xposMsgExpert:      .BYTE $4A                                                                                 
sMsgExpert:         .BYTE 30,68,52,30,56,188

.if !REMOVE_LANGUAGES

 ;                    "EXPERTO"
xposMsgExpertSpn:   .BYTE $45                          
sMsgExpertSpn:      .BYTE 30,68,52,30,56,60,178

 ;                    "ERFAHREN"
xposMsgExpertGer:   .BYTE $40                          
sMsgExpertGer:      .BYTE 30,56,32,22,36,56,30,176
.endif

 ;                    "BONUS"
xposMsgBonus:       .BYTE $8B                          
sMsgBonus:          .BYTE 24,50,48,62,186

 ;                    "TIME"
xposMsgTime:        .BYTE $E8                          
sMsgTime:           .BYTE 60,38,46,158

.if !REMOVE_LANGUAGES
 ;                    "DUREE"
xposMsgTimeFr:      .BYTE $E0                          
sMsgTimeFr:         .BYTE 28,62,56,30,158

 ;                    "ZEIT"
xposMsgTimeGer:     .BYTE $E8                          
sMsgTimeGer:        .BYTE 72,30,38,188

 ;                    "TIEMPO"
xposMsgTimeSpn:     .BYTE $E4                          
sMsgTimeSpn:        .BYTE 60,38,30,46,52,178
.endif

 ;                    "LEVEL"
xposMsgLevel:       .BYTE $8B                          
sMsgLevel:          .BYTE 44,30,64,30,172

.if !REMOVE_LANGUAGES
 ;                    "NIVEAU"
xposMsgLevelFr:     .BYTE $8B                          
sMsgLevelFr:        .BYTE 48,38,64,30,22,190

 ;                    "GRAD"
xposMsgLevelGer:    .BYTE $8B                          
sMsgLevelGer:       .BYTE 34,56,22,156

 ;                    "NIVEL"
xposMsgLevelSpn:    .BYTE $8B                          
sMsgLevelSpn:       .BYTE 48,38,64,30,172
.endif

 ;                    "HOLE"
xposMsgHole:        .BYTE $8B                          
sMsgHole:           .BYTE 36,50,44,158

.if !REMOVE_LANGUAGES
 ;                    "TROU"
xposMsgHoleFr:      .BYTE $8B                          
sMsgHoleFr:         .BYTE 60,56,50,190

 ;                    "HOYO"
xposMsgHoleSpn:     .BYTE $8B                          
sMsgHoleSpn:        .BYTE 36,50,70,178

 ;                    "LOCH"
xposMsgHoleGer:     .BYTE $8B                          
sMsgHoleGer:        .BYTE 44,50,26,164
.endif

 ;                    "INSERT COINS"
xposMsgInsCoin:     .BYTE $DC    
.if DAVEPL_MSG
sMsgInsCoin:        .BYTE 00,00,00,28,22,64,30,52,44,00,00,128
.else                      
sMsgInsCoin:        .BYTE 38,48,58,30,56,60,0,26,50
                    .BYTE 38,48,186
.endif
.if !REMOVE_LANGUAGES
 ;                    "INTRODUIRE LES PIECES"
xposMsgInsCoinFr:   .BYTE $C1                          
sMsgInsCoinFr:      .BYTE 38,48,60,56,50,28,62,38,56
                     .BYTE 30,0,44,30,58,0,52,38,30
                     .BYTE 26,30,186

 ;                    "GELD EINWERFEN"
xposMsgInsCoinGer:  .BYTE $D6                          
sMsgInsCoinGer:     .BYTE 34,30,44,28,0,30,38,48,66
                     .BYTE 30,56,32,30,176

 ;                    "INSERT FICHAS"
xposMsgInsCoinSpn:  .BYTE $D6                          
sMsgInsCoinSpn:     .BYTE 38,48,58,30,56,60,30,0,32
                     .BYTE 38,26,36,22,186
.endif

 ;                    "FREE PLAY"
xposMsgFreePlay:    .BYTE 0                            
sMsgFreePlay:       .BYTE 32,56,30,30,00,52,44,22,198

 ;                    "1 COIN 2 PLAYS"
xposMsg1Coin2Crd:   .BYTE $E                           
sMsg1Coin2Crd:      .BYTE 4,0,26,50,38,48,0,6,0
                     .BYTE 52,44,22,70,186

.if !REMOVE_LANGUAGES
 ;                    "1 PIECE 2 JOUEURS"
xposMsg1Coin2CrdFr: .BYTE $FA                          
sMsg1Coin2CrdFr:    .BYTE 4,0,52,38,30,26,30,0,6
                     .BYTE 0,40,50,62,30,62,56,186

 ;                    "1 MUENZ 2 SPIELE"
xposMsg1Coin2CrdGer: .BYTE 0                            
sMsg1Coin2CrdGer:   .BYTE 4,0,46,62,30,48,72,0,6
                     .BYTE 0,58,52,38,30,44,158

 ;                    "1 MONEDA 2 JUEGOS"
xposMsg1Coin2CrdSpn: .BYTE $FA                          
sMsg1Coin2CrdSpn:   .BYTE 4,0,46,50,48,30,28,22,0
                     .BYTE 6,0,40,62,30,34,50,186
.endif

 ;                    "1 COIN 1 PLAY"
xposMsg1Coin1Crd:   .BYTE $14                          
sMsg1Coin1Crd:      .BYTE 4,0,26,50,38,48,0,4,0
                     .BYTE 52,44,22,198

.if !REMOVE_LANGUAGES
 ;                    "1 PIECE 1 JOUEUR"
xposMsg1Coin1CrdFr: .BYTE 0                            
sMsg1Coin1CrdFr:    .BYTE 4,0,52,38,30,26,30,0,4
                     .BYTE 0,40,50,62,30,62,184

 ;                    "1 MUENZE 1 SPIEL"
xposMsg1Coin1CrdGer: .BYTE 0                            
sMsg1Coin1CrdGer:   .BYTE 4,0,46,62,30,48,72,30,0
                     .BYTE 4,0,58,52,38,30,172

 ;                    "1 MONEDA 1 JUEGO"
xposMsg1Coin1CrdSpn: .BYTE 0                            
sMsg1Coin1CrdSpn:   .BYTE 4,0,46,50,48,30,28,22,0
                     .BYTE 4,0,40,62,30,34,178
.endif

 ;                    "2 COINS 1 PLAY"
xposMsg2Coin1Crd:   .BYTE $E                           
sMsg2Coin1Crd:      .BYTE 6,0,26,50,38,48,58,0,4
                     .BYTE 0,52,44,22,198

.if !REMOVE_LANGUAGES
 ;                    "2 PIECES 1 JOUEUR"
xposMsg2Coin1CrdFr: .BYTE $FA                          
sMsg2Coin1CrdFr:    .BYTE 6,0,52,38,30,26,30,58,0
                     .BYTE 4,0,40,50,62,30,62,184

 ;                    "2 MUENZEN 1 SPIEL"
xposMsg2Coin1CrdGer: .BYTE $FA                          
sMsg2Coin1CrdGer:   .BYTE 6,0,46,62,30,48,72,30,48
                     .BYTE 0,4,0,58,52,38,30,172

 ;                    "2 MONEDA 1 JUEGO"
xposMsg2Coin1CrdSpn: .BYTE $FA                          
sMsg2Coin1CrdSpn:   .BYTE 6,0,46,50,48,30,28,22,58
                     .BYTE 0,4,0,40,62,30,34,178
.endif

 ;                    "(c) MCMLXXX ATARI"
xposMsgAtari:       .BYTE $D3                          
sMsgAtari:          .BYTE 80,0,46,26,46,44,68,68,68
                     .BYTE 0,22,60,22,56,166

 ;                    "CREDITS "
xposMsgCredits:     .BYTE $A0                          
sMsgCredits:        .BYTE 26,56,30,28,38,60,58,128

.if !REMOVE_LANGUAGES
 ;                    "KREDITE "
xposMsgCreditsGer:  .BYTE $A0                          
sMsgCreditsGer:     .BYTE 42,56,30,28,38,60,30,128

 ;                    "CREDITOS "
xposMsgCreditsSpn:  .BYTE $A0                          
sMsgCreditsSpn:     .BYTE 26,56,30,28,38,60,50,58,128
.endif

 ;                    "BONUS "
xposMsgBonusSpc:    .BYTE $DA                                                                                  
sMsgBonusSpc:       .BYTE 24,50,48,62,58,128

 ;                    "2 CREDIT MINIMUM"
xposMsg2CrdMin:     .BYTE $D0                          
sMsg2CrdMin:        .BYTE 6,0,26,56,30,28,38,60,0
                     .BYTE 46,38,48,38,46,62,174

.if !REMOVE_LANGUAGES
 ;                    "2 JEUX MINIMUM"
xposMsg2CrdMinFr:   .BYTE $D6                          
sMsg2CrdMinFr:      .BYTE 6,0,40,30,62,68,0,46,38
                     .BYTE 48,38,46,62,174

 ;                    "2 SPIELE MINIMUM"
xposMsg2CrdMinGer:  .BYTE $D0                          
sMsg2CrdMinGer:     .BYTE 6,0,58,52,38,30,44,30,0
                     .BYTE 46,38,48,38,46,62,174

 ;                    "2 JUEGOS MINIMO"
xposMsg2CrdMinSpn:  .BYTE $D3                          
sMsg2CrdMinSpn:     .BYTE 6,0,40,62,30,34,50,58,0
                     .BYTE 46,38,48,38,46,178
.endif

 ;                    "BONUS EVERY "
xposMsgBonusEv:     .BYTE $C8                          
sMsgBonusEv:        .BYTE 24,50,48,62,58,0,30,64,30
                     .BYTE 56,70,128

.if !REMOVE_LANGUAGES
 ;                    "BONUS CHAQUE"
xposMsgBonusEvFr:   .BYTE $CE                          
sMsgBonusEvFr:      .BYTE 24,50,48,62,58,0,26,36,22
                     .BYTE 54,62,30,128

 ;                    "BONUS JEDE "
xposMsgBonusEvGer:  .BYTE $CE                          
sMsgBonusEvGer:     .BYTE 24,50,48,62,58,0,40,30,28
                     .BYTE 30,128

 ;                    "BONUS CADA "
xposMsgBonusEvSpn:  .BYTE $C8                          
sMsgBonusEvSpn:     .BYTE 24,50,48,62,58,0,26,22,28
                     .BYTE 22,128
.endif

 ;                    "AVOID SPIKES"
xposMsgAvoidSpk:    .BYTE $B8                          
sMsgAvoidSpk:       .BYTE 22,64,50,38,28,0,58,52,38
                     .BYTE 42,30,186

.if !REMOVE_LANGUAGES
 ;                    "ATTENTION AUX LANCES"
xposMsgAvoidSpkFr:  .BYTE $88                          
sMsgAvoidSpkFr:     .BYTE 22,60,60,30,48,60,38,50,48
                     .BYTE 0,22,62,68,0,44,22,48,26
                     .BYTE 30,186

 ;                    "SPITZEN AUSWEICHEN"
xposMsgAvoidSpkGer: .BYTE $96                          
sMsgAvoidSpkGer:    .BYTE 58,52,38,60,72,30,48,0,22
                     .BYTE 62,58,66,30,38,26,36,30,176

 ;                    "EVITA LAS PUNTAS"
xposMsgAvoidSpkSpn: .BYTE $A0                          
sMsgAvoidSpkSpn:    .BYTE 30,64,38,60,30,0,44,22,58
                     .BYTE 0,52,62,48,60,22,186
.endif


 ;                    "LEVEL"
xposMsgLevelNS:     .BYTE $E0                          
sMsgLevelNS:        .BYTE 44,30,64,30,172

.if !REMOVE_LANGUAGES
 ;                    "NIVEAU"
xposMsgLevelNSFr:   .BYTE $DA                          
sMsgLevelNSFr:      .BYTE 48,38,64,30,22,190

 ;                    "GRAD"
xposMsgLevelNSGer:  .BYTE $E2                          
sMsgLevelNSGer:     .BYTE 34,56,22,156

 ;                    "NIVEL"
xposMsgLevelNSSpn:  .BYTE $E0                          
sMsgLevelNSSpn:     .BYTE 48,38,64,30,172
.endif

 ;                    "SUPERZAPPER RECHARGE"
xposMsgRecharge:    .BYTE $C4                          
sMsgRecharge:       .BYTE 58,62,52,30,56,72,22,52,52
                     .BYTE 30,56,0,56,30,26,36,22,56
                     .BYTE 34,158

.if !REMOVE_LANGUAGES
 ;                    "NEUER SUPERZAPPER"
xposMsgRechargeGer: .BYTE $CD                          
sMsgRechargeGer:    .BYTE 48,30,62,30,56,0,58,62,52
                     .BYTE 30,56,72,22,52,52,30,184

 ;                    "NEUVO SUPERZAPPER"
xposMsgRechargeSpn: .BYTE $CD                          
sMsgRechargeSpn:    .BYTE 48,62,30,64,50,0,58,62,52
                     .BYTE 30,56,72,22,52,52,30,184
.endif

language_base_tbl:      .word   msgs_en
.if !REMOVE_LANGUAGES
                        .word   msgs_fr
                        .word   msgs_de
                        .word   msgs_es
.else
                        .word   0
                        .word   0
                        .word   0
.endif        
; Updates optsw2_shadow, coinage_shadow, bonus_life_each, init_lives, and
; diff_bits, from the hardware.

read_optsws:            lda     optsw2
                        sta     optsw2_shadow
                        and     #$38 ; bonus life setting
                        lsr     a
                        lsr     a
                        lsr     a
                        tax
                        lda     bonus_pts_tbl,x
                        sta     bonus_life_each
                        lda     optsw1
                        eor     #$02 ; one of the coinage bits
                        sta     coinage_shadow
                        lda     optsw2_shadow
                        rol     a
                        rol     a
                        rol     a
                        and     #$03 ; lives
                        tax
                        lda     init_lives_tbl,x
                        sta     init_lives
                        lda     optsw2_shadow
                        and     #$06 ; language
                        tay
                        lda     language_base_tbl,y
                        sta     strtbl
                        lda     language_base_tbl+1,y
                        sta     strtbl+1
                        jsr     get_diff_bits
                        sta     diff_bits
                        rts

; Table mapping bonus life setting values to tens of thousands of points

bonus_pts_tbl:          .BYTE 2,1,3,4,5,6,7,0

; Table mapping initial lives setting values to initial lives

init_lives_tbl:         .BYTE 3,4,5,2

                        .byte $7c

nmi_irq_brk:            pha
                        txa
                        pha
                        tya
                        pha
                        cld
                        tsx
                        cpx     #$d0
                        bcc     locd713
                        lda     $53
                        bpl     locd717

locd713:                brk

                        jmp     reset
locd717:                sta     watchdog
                        sta     $60cb ; pokey 1 potgo
                        lda     spinner_cabtyp
                        eor     #$0f
                        tay
                        and     #$10 ; upright/cocktail bit
                        sta     flagbits
                        tya
                        sec
                        sbc     $52
                        and     #$0f
                        cmp     #$08
                        bcc     locd734
                        ora     #$f0
locd734:                clc
                        adc     $50
                        sta     $50
                        sty     $52
                        sta     $60db ; pokey 2 potgo
                        ldy     zap_fire_starts
                        lda     cabsw
                        sta     zap_fire_shadow
                        lda     zap_fire_tmp1
                        sty     zap_fire_tmp1
                        tay
                        and     zap_fire_tmp1
                        ora     zap_fire_debounce
                        sta     zap_fire_debounce
                        tya
                        ora     zap_fire_tmp1
                        and     zap_fire_debounce
                        sta     zap_fire_debounce
                        tay
                        eor     zap_fire_tmp2
                        and     zap_fire_debounce
                        ora     zap_fire_new
                        sta     zap_fire_new
                        sty     zap_fire_tmp2
                        lda     $b4
                        ldy     $13
                        bpl     locd76b
                        ora     #$04
locd76b:                ldy     $14
                        bpl     locd771
                        ora     #$02
locd771:                ldy     $15
                        bpl     locd777
                        ora     #$01
locd777:                sta     vid_coins
                        ldx     twoplayer
                        inx
                        ldy     game_mode
                        bne     locd791
                        ldx     #$00
                        ldy     $07
                        cpy     #$40
                        bcc     locd791
                        ldx     credits
                        cpx     #$02
                        bcc     locd791
                        ldx     #$03
locd791:                lda     locd7dd,x
                        eor     $a1
                        and     #$03
                        eor     $a1
                        sta     $a1
                        sta     leds_flip
                        jsr     loccf24
                        jsr     loccd0a
                        inc     $53
                        inc     $07
                        bne     locd7c9
                        inc     on_time_l
                        bne     locd7b8
                        inc     on_time_m
                        bne     locd7b8
                        inc     on_time_h
locd7b8:                bit     game_mode
                        bvc     locd7c9
                        inc     play_time_l
                        bne     locd7c9
                        inc     play_time_m
                        bne     locd7c9
                        inc     play_time_h
locd7c9:                bit     cabsw
                        bvc     locd7d7
                        inc     $0133
                        sta     vg_reset
                        sta     vg_go
locd7d7:                pla
                        tay
                        pla
                        tax
                        pla
                        rti

locd7dd:                .byte   $ff
                        .byte   $fd
                        .byte   $fe
                        .byte   $fc

; Non-selftest service display

State_ServiceDisplay:   lda     #$00
                        sta     game_mode
                        lda     #$02
                        sta     unknown_state
                        lda     earom_op
                        bne     locd803
                        lda     cabsw
                        and     #$10 ; service switch
                        beq     locd803
                        lda     #GS_GameStartup
                        sta     gamestate
                        lda     hs_initflag
                        and     #$03
                        beq     locd803
                        jsr     init_hs
locd803:                rts
locd804:                jsr     read_optsws
                        jsr     show_coin_stuff
                        jsr     vapp_test_i3
                        jsr     vapp_stats
                        lda     init_lives
                        sta     $37
                        jsr     vapp_vcentre_2
                        lda     #$e8 ; -24
                        ldx     #$c0 ; -64
                        jsr     vapp_ldraw_A_X
                        
                        ST_VECTOR_PLAYER = $326c            ; $326c = player nominal picture

locd81f:                lda     #>ST_VECTOR_PLAYER
                        ldx     #<ST_VECTOR_PLAYER
                        jsr     vapp_vjsr_AX
                        dec     $37
                        bne     locd81f
                        lda     diff_bits
                        and     #$03 ; difficulty
                        asl     a
                        tay
                        lda     diff_str_tbl+1,y
                        ldx     diff_str_tbl,y
                        jsr     vapp_vjsr_AX
                        lda     player_seg
                        jsr     track_spinner
                        sta     player_seg
                        and     #$06
                        pha
                        tay
                        lda     test_magic_tbl+1,y
                        ldx     test_magic_tbl,y
                        jsr     vapp_vjsr_AX
                        pla
                        lsr     a
                        tax
                        lda     zap_fire_debounce
                        and     test_magic_bits,x
                        cmp     test_magic_bits,x
                        bne     locd877
                        dex
                        dex
                        bpl     locd864

; 0, 1: fire&zap -> reset (enter selftest, since test switch is on)

                        jmp     reset
locd864:                bne     locd86c

; 2:    fire&start1 -> zero times

                        jsr     zero_times
                        clv
                        bvc     locd877

; 3:    fire&start2 -> zero scores

locd86c:                jsr     zero_scores
                        lda     hs_initflag
                        ora     #$03
                        sta     hs_initflag

; Common code, after magic button sequence handling done

locd877:                lda     earom_op
                        and     earom_clr
                        beq     locd886

; 346e = draw ERASING

                        ST_VECTOR_ERASING .equ $346E

                        lda     #>ST_VECTOR_ERASING
                        ldx     #<ST_VECTOR_ERASING

                        jsr     vapp_vjsr_AX
locd886:                jsr     vapp_vcentre_2
                        lda     coinage_shadow
                        and     #$1c ; coin-slot multiplier bits
                        lsr     a
                        lsr     a
                        tax
                        lda     locd8ba,x
                        ldy     #$ee ; -18
                        ldx     #$1b ; 27
                        jsr     vapp_ldraw_Y_X_2dig_A
                        lda     coinage_shadow
                        lsr     a ; extract bonus-coins bits
                        lsr     a
                        lsr     a
                        lsr     a
                        lsr     a
                        tax
                        lda     locd8c2,x
                        ldy     #$32 ; x offset 50
                        ldx     #$f8 ; y offset -8

; Append an ldraw per Y,X, then append A as a two-digit hex number.

vapp_ldraw_Y_X_2dig_A:  sta     $29
                        tya
                        jsr     vapp_ldraw_A_X
                        lda     #$29
                        ldy     #$01
                        jmp     vapp_multdig_y_a

; Magic button combinations for when test-mode switch is turned on live.
; 08 = zap, 10 = fire, 20 = start 1, 40 = start 2

test_magic_bits:        .byte   $18         ; fire&zap:    selftest
                        .byte   $18         ; fire&zap:    selftest
                        .byte   $30         ; fire&start1: zero times
                        .byte   $50         ; fire&start2: zero scores

; Coin-slot multiplier display values, two-digit BCD.  Indexed by the
; coin-slot multiplier bits in coinage_shadow.  These do not actually
; affect the multipliers used; they are used only for test mode display.

locd8ba:                .byte   $11
                        .byte   $14
                        .byte   $15
                        .byte   $16
                        .byte   $21
                        .byte   $24
                        .byte   $25
                        .byte   $26

; "BONUS ADDER" values - extra credits for multiple coins.  Indexed by
; the bonus-coin bits in coinage_shadow.  These do not actually affect
; bonus coins awarded; they are used only for test mode display.

locd8c2:                .byte   $00
                        .byte   $12
                        .byte   $14
                        .byte   $24
                        .byte   $15
                        .byte   $13
                        .byte   $00
                        .byte   $00

; Selftest of low RAM failed.

locd8ca:                tay
                        lda     #$00
locd8cd:                sty     $79
                        lsr     a
                        lsr     a
                        asl     a
                        tax
                        tya
                        and     #$0f
                        bne     locd8d9
                        inx
locd8d9:                txs
locd8da:                lda     #$a2
                        sta     $60c1
                        tsx
                        bne     locd8e9
                        lda     #$60
                        ldy     #$09
                        clv
                        bvc     locd8ed
locd8e9:                lda     #$c0
                        ldy     #$01
locd8ed:                sta     pokey1
                        lda     #$03
                        sta     leds_flip
                        ldx     #$00
locd8f7:                bit     cabsw
                        bmi     locd8f7
locd8fc:                bit     cabsw
                        bpl     locd8fc
                        sta     watchdog
                        dex
                        bne     locd8f7
                        dey
                        bne     locd8f7
                        stx     $60c1
                        lda     #$00
                        sta     leds_flip
                        ldy     #$09
locd914:                bit     cabsw
                        bmi     locd914
locd919:                bit     cabsw
                        bpl     locd919
                        sta     watchdog
                        dex
                        bne     locd914
                        dey
                        bne     locd914
                        tsx
                        dex
                        txs
                        bpl     locd8da
.if !REMOVE_SELFTEST                      
                        jmp     SelfTestROM
.else
                        jmp     reset
.endif

locd92f:                eor     (gamestate),y
locd931:                tay
                        lda     $01
                        cmp     #$20
                        bcc     locd93a
                        sbc     #$18
locd93a:                and     #$1f
                        jmp     locd8cd

reset:                  sei
                        sta     watchdog
                        sta     vg_reset

; clear all RAM: 0000-07ff (game RAM) and 2000-2fff (vector RAM)

                        ldx     #$ff
                        txs
                        cld
                        inx
                        txa
                        tay
locd94d:                sty     $00                             ; Being used as a zp pointer, not actually gamestate
                        stx     $01
                        ldy     #$00
locd953:                sta     ($00),y
                        iny
                        bne     locd953
                        inx
                        cpx     #$08
                        bne     locd95f
                        ldx     #$20
locd95f:                cpx     #$30
                        sta     watchdog
                        bcc     locd94d
                        sta     unknown_state
                        sta     leds_flip

; init pokeys

                        sta     $60cf
                        sta     $60df
                        ldx     #$07
                        stx     $60cf
                        stx     $60df
                        inx
locd97a:                sta     pokey1,x
                        sta     pokey2,x
                        dex
                        bpl     locd97a
                        lda     cabsw
                        and     #$10 ; selftest switch

.if !REMOVE_SELFTEST
                        beq     BeginSelfTest ; branch if selftest
.endif

; reset in non-selftest mode

locd98a:                sta     watchdog
                        dec     $0100
                        bne     locd98a
                        dec     $0101
                        bne     locd98a
                        lda     #$10
                        sta     $b4
                        jsr     locde11
                        jsr     init_hs
                        jsr     InitVector
                        cli
                        jmp     locc7a0

                        .byte   $a0

; reset in selftest mode
; Test low RAM: for each byte from $00 to $ff, store $11 in it, then store
; $00 in all other.bytes and verify it's there, then check the $11 is
; undisturbed.  Repeat this with $22, $44, and $88 as well.  (There are
; some faults this won't catch, such as a bit getting cloned from its
; corresponding bit in the other nibble, but it's not a bad check.)
; If any of the checks fail, branch to $d8ca.

BeginSelfTest:         
.if !REMOVE_SELFTEST
                        ldx     #$11
locd9ab:                txs
                        ldy     #$00
locd9ae:                tsx
                        stx     $0,y
                        ldx     #$01
locd9b3:                iny
                        lda     gamestate,y
                        beq     locd9bc
locd9b9:                jmp     locd8ca
locd9bc:                inx
                        bne     locd9b3
                        tsx
                        txa
                        sta     watchdog
                        iny
                        eor     gamestate,y
                        bne     locd9b9
                        sta     gamestate,y
                        iny
                        bne     locd9ae
                        tsx
                        txa
                        asl     a
                        tax
                        bcc     locd9ab

; Low RAM selftest passed.
; Test remaning RAM: $0100-$07ff and $2000-$2fff.  For each.byte, check
; that it's zero (which it should be, we cleared it above), then do a
; write-read-compare of $11, $22, $44, and $88 in it.  When done, store a
; $00 back in it.


TestMiddleRam:          ldy     #$00
                        ldx     #$01
locd9da:                sty     $00
                        stx     $01
                        ldy     #$00
locd9e0:                lda     ($00),y
                        beq     locd9e7
                        jmp     locd931
locd9e7:                lda     #$11
locd9e9:                sta     ($00),y
                        cmp     ($00),y
                        beq     locd9f2
                        jmp     locd92f
locd9f2:                asl     a
                        bcc     locd9e9
                        lda     #$00
                        sta     ($00),y
                        iny
                        bne     locd9e0
                        sta     watchdog
                        inx
                        cpx     #$08
                        bne     locda06
                        ldx     #$20
locda06:                cpx     #$30
                        bcc     locd9da

; Okay, all RAM passed selftest.
; Checksum ROM.  For each $0800 region, XOR all its.bytes together and
; XOR in its region number (0 for the first, 1 for the second, etc), then
; store the result in the.ds at $7d.  Ranges for each of the 12.bytes:
; $7d - $3000-$37ff
; $7e - $3800-$38ff
; $7f - $9000-$97ff
; $80 - $9800-$9fff
; $81 - $a000-$a7ff
; $82 - $a800-$afff
; $83 - $b000-$b7ff
; $84 - $b800-$bfff
; $85 - $c000-$c7ff
; $86 - $c800-$cfff
; $87 - $d000-$d7ff
; $88 - $d800-$dfff

SelfTestROM:            lda     #$00
                        tay
                        tax
                        sta     $3b
                        lda     #$30
                        sta     $3c
locda14:                lda     #$08
                        sta     $38
                        txa
locda19:                eor     ($3b),y
                        iny
                        bne     locda19
                        inc     $3c
                        sta     watchdog
                        dec     $38
                        bne     locda19
                        sta     $7d,x
                        inx
                        cpx     #$02
                        bne     locda32
                        lda     #$90
                        sta     $3c
locda32:                cpx     #$0c
                        bcc     locda14

; All checksums computed and stored in $7d-$88.

                        lda     $7d
                        beq     locda44
                        lda     #$40
                        ldx     #$a4
                        sta     $60c4
                        stx     $60c5
locda44:                ldx     #$05
                        lda     pokey1_rand
locda49:                cmp     pokey1_rand
                        bne     locda53
                        dex
                        bpl     locda49
                        sta     $7a
locda53:                ldx     #$05
                        lda     pokey2_rand
locda58:                cmp     pokey2_rand
                        bne     locda62
                        dex
                        bpl     locda58
                        sta     $7b

; I'm not sure what $de11 does, though I suspect it's an earom read.
; It appears to be loading stuff into the stats stored at $0406-$0411.

locda62:                jsr     locde11
                        ldy     #$02
                        lda     hs_initflag
                        beq     locda76
                        sta     $7c
                        jsr     locddf1
                        ldy     #$00
                        sty     hs_initflag
locda76:                sty     gamestate

; Load the colormap used by the selftest screens.

                        ldx     #$07
locda7a:                lda     locdaf9,x
                        sta     col_ram,x
                        dex
                        bpl     locda7a
                        lda     #$00
                        sta     leds_flip
                        lda     #$10
                        sta     vid_coins

; Top of selftest-mode main loop.
; Wait for the vector processor to be done.  Loop up to five times.

locda8d:                ldy     #$04

; Wait for 21 ($14+1) cycles of the 3KHz signal.

locda8f:                ldx     #$14
locda91:                bit     cabsw
                        bpl     locda91
locda96:                bit     cabsw
                        bmi     locda96
                        dex
                        bpl     locda91

; Have we run out of iterations?  If so, break out.

                        dey
                        bmi     locdaa9

; Poke the watchdog - don't want to get reset while waiting!

                        sta     watchdog

; Is the vector processor done?

                        bit     cabsw
                        bvc     locda8f ; tests vector processor halt bit

; Either the vector processor is done or we got tired of waiting.

locdaa9:                sta     vg_reset
                        lda     #<vecram
                        sta     vidptr_l
                        lda     #>vecram
                        sta     vidptr_h
                        sta     $60cb
                        lda     spinner_cabtyp
                        sta     $52
                        and     #%00001111
                        sta     $50
                        lda     cabsw
                        eor     #$ff
                        and     #$2f            ; keep diag step, slam, and coins
                        sta     zap_fire_new
                        and     #$28            ; keep diag step and slam
                        beq     locdad8
                        asl     zap_fire_tmp1
                        bcc     locdad5
                        inc     gamestate
                        inc     gamestate
locdad5:                clv
                        bvc     locdadc
locdad8:                lda     #$20
                        sta     zap_fire_tmp1
locdadc:                jsr     draw_selftest_scr
                        jsr     vapp_centre_halt
                        sta     vg_go
                        inc     timectr
                        lda     timectr
                        and     #$03
                        bne     locdaf0
                        jsr     locde1b
locdaf0:                lda     cabsw
                        and     #$10
                        beq     locda8d

; We depend on something else to break us out of this loop.  I suspect
; this "something" is the hardware watchdog.
; (could this be demo freeze mode as a quick guess)

Deadlock                bne     Deadlock

; Loaded to colour RAM; see $da7a

locdaf9:                .byte   $00
                        .byte   $04
                        .byte   $08
                        .byte   $0c
                        .byte   $03
                        .byte   $07
                        .byte   $0b
                        .byte   $0b

; Jump table, used just below at $db19.
; These are the various selftest screens.

locdb01:                
                        .word   selftest_0-1
                        .word   selftest_1-1
                        .word   selftest_2-1
                        .word   selftest_3-1
                        .word   selftest_4-1
                        .word   selftest_5-1
                        .word   selftest_6-1

draw_selftest_scr:      ldx     gamestate
                        cpx     #$0e
                        bcc     locdb19
                        ldx     #$02
                        stx     gamestate
locdb19:                lda     locdb01+1,x
                        pha
                        lda     locdb01,x
                        pha
                        rts

selftest_6:             
                      
                        lda     #$00
                        sta     leds_flip
                        sta     mb_w_00
                        sta     pokey1
                        sta     pokey2
                        sta     earom_write
                        sta     eactl_mbst
                        lda     eactl_mbst
                        lda     mb_rd_l
                        lda     mb_rd_h
                        lda     earom_rd
                        lda     #$08
                        sta     leds_flip
                        lda     #$01
                        ldx     #$1f
                        clc
locdb4c:                sta     mb_w_00,x
                        rol     a
                        dex
                        bpl     locdb4c

; $34a6 = draw box around screen

                        ST_VECTOR_BOX .equ $34a6                ; ROM location for vectors to draw a box around the screen

                        lda     #>ST_VECTOR_BOX
                        ldx     #<ST_VECTOR_BOX
                        jmp     vapp_vjsr_AX

; It's not clear to me this code is _ever_ executed...

selftest_0:             lda     earom_op
                        ora     $01c7
                        bne     locdb6e
                        jsr     locde11
                        lda     hs_initflag
                        sta     $7c
                        lda     #GS_LevelStartup
                        sta     gamestate
locdb6e:                rts

selftest_5:             lda     $50
                        lsr     a
                        tay
                        lda     #$68
                        jsr     vapp_sclstat_A_Y

; $334e = rectangular grid selftest; this display is mostly ROMed

                        ST_GRID_VECTOR .equ $334e               ; ROM location to draw a grid on the screen for self-test

                        ldx     #<ST_GRID_VECTOR
                        lda     #>ST_GRID_VECTOR
                        bne     locdb88

; $32b6 = coloured-lines selftest; this display is entirely ROMed

                        ST_VECTOR_4 .equ $32b6                  ; ROM location to draw colored lines on screen for self-test

selftest_4:             ldx     #<ST_VECTOR_4
                        lda     #>ST_VECTOR_4
                        bne     locdb88

; $330a = draw selftest screen 2; this display is entirely ROMed

                        ST_VECTOR_2 .equ $330a                  ; ROM location for self-test screeen 2

selftest_2:             lda     #>ST_VECTOR_2
                        ldx     #<ST_VECTOR_2
locdb88:                jsr     vapp_vjsr_AX
                        ldx     #$06
                        lda     #$00
locdb8f:                sta     $60c1,x
                        sta     $60d1,x
                        dex
                        dex
                        bpl     locdb8f
.endif
                        rts

selftest_3:             lda     timectr

                        and     #$3f
                        bne     locdba2
                        inc     $39
locdba2:                lda     $39
                        and     #$07
                        tax
                        ldy     locdbd5,x
                        lda     #$00
                        sta     $60c1,y
                        ldy     locdbd6,x
                        lda     locdfdc,x
                        sta     pokey1,y
                        lda     #$a8
                        sta     $60c1,y

; $3456 = draw full-screen crosshair

                        VECTOR_CROSSHAIRS .equ $3456

                        lda     #>VECTOR_CROSSHAIRS
                        ldx     #<VECTOR_CROSSHAIRS
                        jsr     vapp_vjsr_AX
                        lda     timectr
                        and     #$7f
                        tay
                        lda     #$01
                        jsr     vapp_scale_A_Y

; $34aa = draw box

                        VECTOR_BOX .equ $34AA

                        lda     #>VECTOR_BOX
                        ldx     #<VECTOR_BOX
                        jmp     vapp_vjsr_AX

; Selftest sound table of some sort - see $dba7

locdbd5:                .byte   $16
locdbd6:                .byte   $00
                        .byte   $10
                        .byte   $02
                        .byte   $12
                        .byte   $04
                        .byte   $14
                        .byte   $06
                        .byte   $16
                        .byte   $00
                        .byte   $ea

; Returns value with difficulty/rating bits in $07, something unknown
; ($20 bit of spinner/cabinet select.byte) in $08.
; Uses $37 as temporary storage.

get_diff_bits:          sta     $60db
                        lda     zap_fire_starts
                        and     #$07 ; difficulty/rating bits
                        sta     $37
                        sta     $60cb
                        lda     spinner_cabtyp
                        and     #$20 ; Unknown
                        lsr     a
                        lsr     a
                        ora     $37
                        rts

; Selftest screen 1 ($00 holds $02)

selftest_1:             lda     $2e
                        beq     locdc19
                        sta     mb_w_15
                        sta     mb_w_0d
                        lda     $2f
                        sta     mb_w_16
                        ldx     #$00
                        jsr     divide
                        cmp     #$01
                        bne     locdc15
                        tya
                        bne     locdc15
                        txa
                        bpl     locdc19
locdc15:                lda     #$ff
                        sta     $78
locdc19:                ldx     #$00
                        stx     draw_z
                        inc     $2e
                        bne     locdc27
                        inc     $2f
                        bpl     locdc27
                        stx     $2f
locdc27:                sta     $60db
                        lda     zap_fire_starts
                        and     #$78 ; zap, fire, start1, start2
                        sta     zap_fire_debounce
                        beq     locdc38
                        sta     pokey1
                        ldx     #$a4
locdc38:                stx     $60c1
                        ldx     #$00
                        lda     zap_fire_new
                        beq     locdc47
                        asl     a
                        sta     $60c2
                        ldx     #$a4
locdc47:                stx     $60c3
                        jsr     vapp_test_i3
                        ldy     zap_fire_debounce
                        lda     #$d0 ; -48
                        ldx     #$f0 ; -16
                        jsr     vapp_test_ibits
                        ldy     zap_fire_new
                        jsr     vapp_test_ibmove
                        lda     $52
                        and     #$10
                        beq     locdc7e

                        ST_VECTOR_C .equ $3482                  ; $3482 = draw cocktail-bit C
                        lda     #>ST_VECTOR_C
                        ldx     #<ST_VECTOR_C

                        jsr     vapp_vjsr_AX
                        ldy     #$10
                        lda     zap_fire_debounce
                        and     #$60 ; start1, start2
                        beq     locdc7e
                        eor     #$20
                        beq     locdc78
                        lda     #$04
                        ldy     #$08
locdc78:                sta     leds_flip
                        sty     vid_coins

                        ST_VECTOR_BOXLINE .equ $3492            ; $3492 = draw box around screen and line across the middle

locdc7e:                lda     #>ST_VECTOR_BOXLINE
                        ldx     #<ST_VECTOR_BOXLINE
                        jsr     vapp_vjsr_AX

; Show any nonzero checksums (stored in the 12.bytes from $7d to $88)

                        ldx     #$0b
locdc87:                lda     $7d,x
                        beq     locdca4
                        sta     $35
                        stx     $38
                        txa
                        jsr     vapp_digit
                        ldy     #$f4 ; -12
                        ldx     #$f4 ; -12
                        lda     $35
                        jsr     vapp_ldraw_Y_X_2dig_A
                        lda     #$0c ; 12, 12
                        tax
                        jsr     vapp_ldraw_A_X
                        ldx     $38
locdca4:                dex
                        bpl     locdc87
                        jsr     vapp_vcentre_2
                        lda     #$00
                        ldx     #$16
                        jsr     vapp_ldraw_A_X

; Show the 5 characters in $78-$7c

                        ldx     #$04
                        stx     $37
locdcb5:                ldx     $37
                        ldy     #$00
                        lda     $78,x
                        beq     locdcc0
                        ldy     locdce1,x
locdcc0:                lda     char_jsrtbl,y
                        ldx     char_jsrtbl+1,y
                        jsr     vapp_A_X_Y_0
                        dec     $37
                        bpl     locdcb5

; Draw the spinner line

                        ldx     #$ac
                        lda     #$30
                        jsr     vapp_ldraw_A_X
                        ldy     $50
                        lda     spinner_sine+4,y
                        ldx     spinner_sine,y
                        ldy     #$c0
                        jmp     vapp_ldraw_A_X_Y
locdce1:                rol     $3438
                        rol     $1e,x

; This appears to be doing a divide, but I'm not clear enough on how the
; mathbox works to be certain of the details.

divide:                 ldy     #$00
                        sty     draw_z
                        sty     secs_avg_h
                        sta     mb_w_0e
                        stx     mb_w_0f
                        sty     mb_w_10
                        ldx     #$10
                        stx     mb_w_0c
                        stx     mb_w_14
locdcfe:                dex
                        bmi     locdd0c
                        lda     eactl_mbst
                        bmi     locdcfe
                        lda     mb_rd_l
                        ldy     mb_rd_h
locdd0c:                rts

; Appends code to display the three lines of bits showing the configuration
; and input button values.

vapp_test_i3:           jsr     vapp_vcentre_2
                        lda     #$00
                        jsr     vapp_scale_A_0
                        lda     #$e8 ; -24
                        ldy     optsw1
                        jsr     locdd29
                        ldy     optsw2
                        jsr     vapp_test_ibmove
                        jsr     get_diff_bits
                        tay

vapp_test_ibmove:       lda     #$d0 ; -48
locdd29:                ldx     #$f8 ; -8

vapp_test_ibits:                sty     $35
                        jsr     vapp_ldraw_A_X
                        ldx     #$07
                        stx     $37
locdd34:                asl     $35
                        lda     #$00
                        rol     a
                        jsr     vapp_digit
                        dec     $37
                        bpl     locdd34
                        rts

; Display game statistics.

vapp_stats:             lda     games_2p_l
                        asl     a
                        sta     $29
                        lda     games_2p_m
                        rol     a
                        sta     $2a
                        lda     games_1p_l
                        clc
                        adc     $29
                        sta     mb_w_15
                        sta     $29
                        lda     games_1p_m
                        adc     $2a
                        sta     mb_w_16
                        ora     $29
                        bne     locdd69
                        lda     #$01
                        sta     mb_w_15
locdd69:                lda     play_time_l
                        sta     mb_w_0d
                        lda     play_time_m
                        ldx     play_time_h
                        jsr     divide
                        sta     secs_avg_l
                        sty     secs_avg_m


                        ST_VECTOR_LABELS    .equ $3dce          ; 3dce = draw the "SECONDS ON", "SECONDS PLAYED", etc, labels
                        lda     #>ST_VECTOR_LABELS
                        ldx     #<ST_VECTOR_LABELS

                        jsr     vapp_vjsr_AX
                        lda     #$06
                        sta     $3b
                        lda     #$04
                        sta     $3c
                        sta     $37
locdd8f:                ldy     #$00
                        sty     $31
                        sty     $32
                        sty     $33
                        sty     $34
                        lda     ($3b),y
                        sta     $56
                        inc     $3b
                        lda     ($3b),y
                        sta     $57
                        inc     $3b
                        lda     ($3b),y
                        sta     $58
                        inc     $3b

; From here to the cld at ddc8, code converts a 24-bit number stored in
; $56/$57/$58 into six-nibble BCD, stored in $31/$32/$33.  Only the low
; six digits are retained.

                        sed
                        ldy     #$17
                        sty     $38
locddb0:                rol     $56
                        rol     $57
                        rol     $58
                        ldy     #$03
                        ldx     #$00
locddba:                lda     $31,x
                        adc     $31,x
                        sta     $31,x
                        inx
                        dey
                        bpl     locddba
                        dec     $38
                        bpl     locddb0
                        cld
                        lda     #$31
                        ldy     #$04
                        jsr     vapp_multdig_y_a
                        lda     #$d0 ; -48
                        ldx     #$f8 ; -8
                        jsr     vapp_ldraw_A_X
                        dec     $37
                        bpl     locdd8f
                        rts
         
                        .byte $73

; Starting and ending offsets in EAROM of various pieces.

locdddd:                .byte   $00             ; Top three initials, start

locddde:                .byte   $09             ; Top three initials, end
                        .byte   $0a             ; Top three scores, start
                        .byte   $15             ; Top three scores, end
                        .byte   $16             ; Switched-on time, start
                        .byte   $22             ; Switched-on time, end

; Pointers to RAM versions of EAROM stuff

locdde3:                .word   hs_initials_3
                        .word   hs_score_3
                        .word   on_time_l

zero_times:             lda     #$04
                        bne     locddf3

zero_scores:            lda     #$03
                        bne     locddf3
locddf1:                lda     #$07
locddf3:                ldy     #$ff
                        bne     locddff
locddf7:                lda     #$03
                        bne     locddfd
locddfb:                lda     #$04
locddfd:                ldy     #$00
locddff:                sty     earom_clr       ; A now 3/4/7; Y now $00/$ff
                        pha
                        ora     $01c7
                        sta     $01c7
                        pla
                        ora     $01c8
                        sta     $01c8
                        rts
locde11:                lda     #$07
                        sta     $01c7
                        lda     #$00
                        sta     $01c8
locde1b:                lda     earom_op
                        bne     locde6b
                        lda     $01c7
                        beq     locde6b
                        ldx     #$00
                        stx     earom_blkoff
                        stx     earom_cksum
                        stx     $01ce

; This loop finds the highest bit in A and leaves it in $01ce - the bcc
; tests the C bit set by the asl; the dex doesn't touch C.  It also leaves
; the bit number of this bit in X (0 to 2, since A is $0-$7).

                        ldx     #$08
                        sec
locde33:                ror     $01ce
                        asl     a
                        dex
                        bcc     locde33
                        ldy     #$80
                        lda     $01ce
                        and     $01c8
                        bne     locde46
                        ldy     #$20
locde46:                sty     earom_op
                        lda     $01ce
                        eor     $01c7
                        sta     $01c7
                        txa
                        asl     a
                        tax
                        lda     locdddd,x
                        sta     earom_ptr
                        lda     locddde,x
                        sta     earom_blkend
                        lda     locdde3,x
                        sta     earom_memptr
                        lda     locdde3+1,x
                        sta     earom_memptr+1
locde6b:                ldy     #$00
                        sty     eactl_mbst
                        lda     earom_op
                        bne     locde76
                        rts
locde76:                ldy     earom_blkoff
                        ldx     earom_ptr
                        asl     a
                        bcc     locde8c

; EAROM op $80

                        sta     earom_write,x
                        lda     #$40
                        sta     earom_op
                        ldy     #$0e
                        clv
                        bvc     locdeff
locde8c:                bpl     locdeb3

; EAROM op $40

                        lda     #$80
                        sta     earom_op
                        lda     earom_clr
                        beq     locde9c
                        lda     #$00
                        sta     (earom_memptr),y
locde9c:                lda     (earom_memptr),y
                        cpx     earom_blkend
                        bcc     locdeab
                        lda     #$00
                        sta     earom_op
                        lda     earom_cksum
locdeab:                sta     earom_write,x
                        ldy     #$0c
                        clv
                        bvc     locdef2

; EAROM op $20

locdeb3:                lda     #$08
                        sta     eactl_mbst
                        sta     earom_write,x
                        lda     #$09
                        sta     eactl_mbst
                        nop
                        lda     #$08
                        sta     eactl_mbst
                        cpx     earom_blkend
                        lda     earom_rd
                        bcc     locdeee
                        eor     earom_cksum
                        beq     locdee6
                        lda     #$00
                        ldy     earom_blkoff
locded8:                sta     (earom_memptr),y
                        dey
                        bpl     locded8
                        lda     $01ce
                        ora     hs_initflag
                        sta     hs_initflag
locdee6:                lda     #$00
                        sta     earom_op
                        clv
                        bvc     locdef0
locdeee:                sta     (earom_memptr),y
locdef0:                ldy     #$00
locdef2:                clc
                        adc     earom_cksum
                        sta     earom_cksum
                        inc     earom_blkoff
                        inc     earom_ptr
locdeff:                sty     eactl_mbst
                        tya
                        bne     locdf08
                        jmp     locde1b
locdf08:                rts

vapp_rts:               lda     #$c0            ; vrts (first.byte)
                        bne     locdf12

vapp_centre_halt:       jsr     vapp_vcentre_2

                        lda     #$20            ; vhalt (first.byte)
locdf12:                ldy     #$00
                        sta     (vidptr_l),y
                        jmp     locdfac

; Appends the vjsr for the digit corresponding to the low four bits of A
; on entry.  If C is set, zeros become.dss; C is cleared if the digit
; is nonzero.

vapp_digit_lz:          bcc     vapp_digit
                        and     #$0f
                        beq     locdf24

vapp_digit:             and     #$0f
                        clc
                        adc     #$01
locdf24:                php
                        asl     a
                        ldy     #$00
                        tax
                        lda     char_jsrtbl,x
                        sta     (vidptr_l),y
                        lda     char_jsrtbl+1,x
                        iny
                        sta     (vidptr_l),y
                        jsr     inc_vi.word
                        plp
                        rts

; Appends a vjsr to the video list.  A holds high.byte of address to vjsr
; to; X holds low.byte.  Note that the $e0 bits of A are ignored.  (The
; low bit of X is discarded too, but the vjsr format compels this anyway;
; a vjsr to an odd address is not representible.)

vapp_vjsr_AX:           lsr     a
                        and     #$0f
                        ora     #$a0
                        ldy     #$01
                        sta     (vidptr_l),y
                        dey
                        txa
                        ror     a
                        sta     (vidptr_l),y
                        iny
                        bne     inc_vi.word

; Append a vscale or vstat.  The second.byte is A|$60, the first.byte is
; in $73, or Y, depending on which entry point.  I suppose this could
; generate a vrts or vjmp if entered with A having $80 set.

vapp_sclstat_A_73:      ldy     draw_z
vapp_sclstat_A_Y:       ora     #$60

                        tax
                        tya
                        jmp     vapp_A_X_Y_0

; $40 $80 = vcentre (why $40? who knows.)

vapp_vcentre_2:         lda     #$40
                        ldx     #$80

; Append first A, then X, to the video stream.

vapp_A_X_Y_0:           ldy     #$00
vapp_A_X:               sta     (vidptr_l),y
                        iny
                        txa
                        sta     (vidptr_l),y

; increment vidptr_l/vidptr_h by the offset accumulated in y

inc_vi.word:            tya
                        sec
                        adc     vidptr_l
                        sta     vidptr_l
                        bcc     locdf69
                        inc     vidptr_h
locdf69:                rts

; Append a vscale to the video stream, with l=0 and b coming from the
; value in A on entry (we assume it's in the range 0-7).

vapp_scale_A_0:         ldy     #$00

; ...fall through into...
; Append a vscale to the video stream, getting l from Y and b from A on
; entry (we assume they're in range).

vapp_scale_A_Y:         ora     #$70
                        tax
                        tya
                        jmp     vapp_A_X_Y_0

; Appends a long draw to the video list, just like vapp_ldraw_A_X below,
; except that the incoming Y value is stored in $73 first (and thus used
; as the Z value for the draw).

vapp_ldraw_A_X_Y:       sty     draw_z

; Appends a long draw to the video list.  The X coordinate of the draw
; comes from the A register on entry (sign-extended); the Y coordinate
; from the X register (again, sign-extended).  The Z value for the draw
; is the high three bits of $73.  The Y register and $6e-$71 are trashed.

vapp_ldraw_A_X:         ldy     #$00
                        asl     a
                        bcc     locdf7b
                        dey
locdf7b:                sty     $6f
                        asl     a
                        rol     $6f
                        sta     $6e
                        txa
                        asl     a
                        ldy     #$00
                        bcc     locdf89
                        dey
locdf89:                sty     $71
                        asl     a
                        rol     $71
                        sta     $70
                        ldx     #$6e
locdf92:                ldy     #$00
                        lda     $02,x
                        sta     (vidptr_l),y
                        lda     timectr,x
                        and     #$1f
                        iny
                        sta     (vidptr_l),y
                        lda     gamestate,x
                        iny
                        sta     (vidptr_l),y
                        lda     $01,x
                        eor     draw_z
                        and     #$1f
                        eor     draw_z
locdfac:                iny
                        sta     (vidptr_l),y
                        bne     inc_vi.word

; Appends a multidigit number.  A holds the zero page address of the low
; two digits of the number; Y holds the number of two-digit pairs to
; process.

vapp_multdig_y_a:       sec
                        php
                        dey
                        sty     $ae
                        clc
                        adc     $ae
                        plp
                        tax
locdfbb:                php
                        stx     $af
                        lda     gamestate,x
                        lsr     a
                        lsr     a
                        lsr     a
                        lsr     a
                        plp
                        jsr     vapp_digit_lz
                        lda     $ae
                        bne     locdfcd
                        clc
locdfcd:                ldx     $af
                        lda     gamestate,x
                        jsr     vapp_digit_lz
                        ldx     $af
                        dex
                        dec     $ae
                        bpl     locdfbb
                        rts

; Used at $dbb2

locdfdc:                .byte   $10
                        .byte   $10
                        .byte   $40
                        .byte   $40
                        .byte   $90
                        .byte   $90
                        .byte   $ff
                        .byte   $ff

; 20-element sine wave, for drawing the spinner line on selftest screen 1

spinner_sine:           .byte 0, 12, 22, 30, 32, 30, 22, 12, 0, -12, -22
                        .byte -30, -32, -30, -22, -12, 0, 12, 22, 30

locdff8:                .byte   $00
lastbyte:               .byte   $00

; Pad out the file so that the 6502 vectors wind up at the same spot.  Machine 
; won't even boot if you don't have this correct!

CPUVectors = $dffa

PADLENGTH               .equ    (CPUVectors-lastbyte-1)
.ECHO STR$(PADLENGTH) " bytes were added to pad the file"
.REPEAT PADLENGTH
.byte 00
.ENDREPEAT

vector_nmi:             .word   nmi_irq_brk
vector_reset:           .word   reset
vector_irq_brk:         .word   nmi_irq_brk

