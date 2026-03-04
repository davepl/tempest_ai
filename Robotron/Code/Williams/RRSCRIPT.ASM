*      OBJECT DESC

*      PTR TO PTR TO OBJECT 1           0
*      NUMBER OF IMAGES                 2
*      BYTES PER IMAGE                  3
*      MOVE LEFT ROUTINE                4
*      MOVE RIGHT ROUTINE               6
*      MOVE DOWN ROUTINE                8 
*      MOVE UP ROUTINE                  10
*      MOTION DESC LIST PTR, PTR        12
*
*

*      PROCESS DATA STRUCTURE

SCPTR  EQU    PD             SCRIPT POINTER
ODESC  EQU    SCPTR+2        OBJECT DESCRIPTOR
OBJ    EQU    ODESC+2        OBJECT BLOCK POINTER
IMAG   EQU    OBJ+2          CURRENT IMAGE NUMBER
LEV1   EQU    IMAG+1         LEVEL 1 SUBROUTINE SAVE
LEV2   EQU    LEV1+2
LOOP   EQU    LEV2+2         LOOP COUNT
USER   EQU    LOOP+1         ROUTINE STORE

*
*      SPAGE PROCESSOR - COMMANDS
*
*      VALID EUGSET CHARACTER GETS PRINTED AT CURSOR POSITION
*
*      SET CURSOR
*      CLEAR MESSAGE AREA
*      NEWLINE
*      START SCRIPT <PTR TO SCRIPT>
*      NUMBER GREATER THAN $5E IS SLEEP TIME
*      SLEEP IMMEDIATE COMMAND
*      PUT MESSAGE IN SCORE AREA X OFFSET, MESSAGE NUMBER 
*      END IT

       BRA    SPAGE    HISTRY ORG IS HERE

FAMPAG LDA    ATFLAG
       ORAA   #$80     TURN ON A BIT TO STOP SEIZURE BY COIN
       STA    ATFLAG
       BSR    SPGSUB
       LDA    #FAMMM
       JSR    WRD7V    PRINT THE PROMPTER
       LDX    #DUMPLR  DUMB PLAYER HELPER
       BRA    SPHELP   AND PROCESS THIS OTHER SCRIPT

SPGSUB JSR    GNCIDE
       JSR    INT20V   INITIALIZE THE STUFF
       JSR    SCRCLR
       JSR    COLST    START THE COLOR GUYS
       MAKP   ACPROC   START A CREDITS SCANNING PROCESS
       LDA    STATUS
       ANDA   #$F3
       STA    STATUS
       CLR    CURPLR
       LDA    #$CC
       STA    WALCOL
       JSR    TDISPV
       LDA    #TITLEM
       JMP    WRD7V    ROBOTRON TITLE

SPAGE  BSR    SPGSUB
       LDX    #HISTO
SPHELP LDU    CRPROC
       STX    PD,U
SPWAKE LDX    PD,U
       LDA    ,X+      GET A COMMAND
       STX    PD,U
       CMPA   #LASCOM  COMMAND?
       BHI    SPW1     NO
       LDX    #SCMTAB  TABLE OF PTRS
       ASLA
       JSR    [A,X]
       BRA    SPWAKE   AND GET NEXT GUY

SPW1   CMPA   #$5F     GREATER OR EQUAL TO 5F??
       BLO    SPW2
       LDX    #SPWAKE  SLEEP THE TIME
       JMP    SLEEP

SPW2   LDX    PD+2,U   CURSOR
       JSR    PR57V    PRINT THE CHAR
       STX    PD+2,U
       NAP    3,SPWAKE  AND TAKE A BREAK

LEFT   EQU    $14
TOP    EQU    $20
TOPLEF EQU    $1420

MESHIT EQU    $D8

CURSAB EQU    0
CLEARM EQU    1
NEWLIN EQU    2
SCRPT  EQU    3
SNOOZE EQU    4
MESS   EQU    5
DONE   EQU    6
COLOR  EQU    7
GRUNTS EQU    8
LASCOM EQU    9
DONE2  EQU    9

SCMTAB FDB    CURSAP
       FDB    CLRMP
       FDB    NEWLNP
       FDB    SCRPP
       FDB    SLEPP
       FDB    MESSP
       FDB    ENDP
       FDB    COLORP
       FDB    GRNTME
       FDB    DONE2P

DONE2P LDA    ATFLAG         SEE WHERE TO GO
       ANDA   #$7F
       STA    ATFLAG
       LBEQ   RUNIT          RUN THE BALLS...NORMAL EXIT
       JMP    HISTRY         OTHERWISE GO INTO HISTORICAL STUFF

GRNTME LDX    #GRPROC
       LDA    #STYPE
       JMP    MKPROC         YEA

GRPROC LDA    #14            GET THE GRUNT TABLE
       STA    PD,U           SAVE
GRLOOP NAP    $10,GRUNT1
GRUNT1 JSR    RAND           GET A RANDOM PATH
       ANDA   #$6
       LDX    #GSTRTS
       LDY    A,X            GET THE GUY
       JSR    OSTART         AND START HIM
       DEC    PD,U
       BNE    GRLOOP
       JMP    SUCIDE

GSTRTS FDB    GS1
       FDB    GS2
       FDB    GS3
       FDB    GS4

COLORP JSR    FETCH1
       STA    TEXCOL
       RTS

MESSP  LDA    #LEFT
       LDB    #MESHIT 
       TFR    D,X
       LDD    #$7406
       JSR    BLKCLR   CLEAR OUT OLD MESSAGE
       LDA    TXFLAV   SAVE THE FLAVOR
       PSHS   A
       JSR    FETCH1
       LDB    #MESHIT
       CLR    TXFLAV
       TFR    D,X
       JSR    FETCH1   GET MESSAGE NUMBER
       JSR    WRD5V    PRINT THE MESSAGE
       PULS   A
       STA    TXFLAV
       RTS

ENDP   LDA    STATUS
       ORAA   #$C
       STA    STATUS
       JSR    OINIT
       JSR    GNCIDE
       JMP    LOGORG   AND BACK TO LOGO PROGRAM

CURSAP JSR    FETCH2
       STY    PD+2,U   STORE NEW CURSOR
       CLR    TXFLAV
       RTS

CLRMP  JSR    FETCH1         GET THE ARG
       TFR    A,B
       LDA    #LEFT
       STD    PD+2,U         SET CURSOR
       LDX    #TOPLEF+$10
       CLR    TXFLAV
       JSR    FETCH1         GET THE SIZE TO CLEAR
       TFR    A,B            THIS IS THE Y RANGE
       SUBB   #$10           MAKE ROOM FOR ROBOTRON
       LDA    #$74  
       JMP    BLKCLR   CLEAR THE BLOCK

NEWLNP LDB    PD+3,U   GET THE CURSOR Y
       LDA    #LEFT
       ADDB   #11
       STD    PD+2,U
       CLR    TXFLAV
       RTS

SCRPP  JSR    FETCH2
       JMP    OSTART   START THE OBJECT

SLEPP  JSR    FETCH1   GET THE TIME
       LDX    #SPWAKE
       JMP    SLEEP

LEFFIR LDY    #-$280
       LDD    #$FF06
       BRA    RIFI1

RIGFIR LDY    #$280
       LDD    #$206
RIFI1  JSR    GETOB    GET AN OBJECT FOR THE BULLET
       STY    OXV,X    THIS IS THE BULLETS VELOCITY
       LDY    PD,U     GET OBJECT OF CALLER
       ADDB   OY16,Y
       STB    OY16,X
       ADDA   OX16,Y
       STA    OX16,X
       STD    OBJX,X   SETUP STUPID OLD OBJX
       LDD    LASPIC
       STD    OPICT,X
       STD    OLDPIC,X
       STX    OPTR     PUT ON LIST

       STX    PD,U     SAVE OBJECT POINTER
       LDA    PD+2,U   GET NAP TIME
       LDX    #FIRWAK
       JMP    SLEEP

FIRWAK LDX    PD,U     GET OBJECT
       JSR    KILLOF   CAN THE STUFF
       JMP    SUCIDE   AND DIE

*      OSTART - INITIATE NEW OBJECT PROCESSING
*      Y CONTAINS POINTER TO OBJECT ACTION DESCRIPTOR 
*      FIRST ACTION IS TO SET OBJECT DESCRIPTOR

OSTART PSHS   A,X
       LDA    #STYPE         JUST MAKE THE PROCESS
       LDX    #SCRIPT        START A SCRIPTOR PROCESS
       JSR    MSPROC         MAKE A SUPER PROCESS
       STY    PD,X           PASS IT THE SCRIPT
       PULS   A,X,PC         AND

SCRIPT JSR    GETOB          GET AN OBJECT
       STX    OBJ,U          SAVE 
       CLR    LOOP,U         CLEAR LOOP COUNTER
       BSR    SPROC          PROCESS THE SCRIPT ONCE TO GET OBJECT PTR
       BSR    SPROC          PROCESS ONCE MORE FOR INITIAL COORD
       LDU    CRPROC
       LDX    OBJ,U          GET OBJECT
       LDD    OPICT,X        GET NEW PICTURE
       STD    OLDPIC,X
       STX    OPTR           PUT ON LIST
SC2    BSR    SPROC
       BRA    SC2            AND JUST PROCESS THE SUCKER

SPROC  PULS   D              GET CALLER
       LDU    CRPROC
       STD    LEV1,U         SAVE CALLER
       JSR    FETCH1         FETCH A BYTE INTO A
       ASLA                  DOUBLE THE BYTE
       LDX    #SCTABL
       LDX    A,X
       STX    XTEMP          PUT ON STACK
       LDX    OBJ,U
       LDY    ODESC,U        PASS APPROPRIATE GUYS TO SAVE BYTES
       JSR    [XTEMP]        CALL THE GUY
       LDU    CRPROC
       JMP    [LEV1,U]       AND RETURN

*      MNEMONICS

SETOB  EQU    1        SET OBJECT DESC.  2 BYTE DESC FOLLOWS
SETIM  EQU    2        SET IMAGE NUMBER  CURRENT IMAGE TO USE  1 B
MLEFT  EQU    3        MOVE LEFT <1 BYTE> STEPS
MRIGHT EQU    4        MOVE RIGHT
MDOWN  EQU    5        MOVE DOWN
MUP    EQU    6        MOVE UP
SETPOS EQU    7        SET POSITION TO <1 BYTE X> <1 BYTE Y>
SETXV  EQU    8        SET X VELOCITY <2 BYTES>
SETYV  EQU    9        SET Y VELOCITY <2 BYTES>
CYCLE  EQU    10       CYCLE <N FRAMES EACH>, <M IMAGES OF OBJECT>
REST   EQU    11       SLEEP N FRAMES
DIE    EQU    12       KILL THE GUY
EXP    EQU    13       MAKE GUY EXPLODE, THEN DIE
JUMP   EQU    14       CONTINUE SCRIPT AT 2 BYTE ADDRESS
HIB    EQU    15       ERASE AND REMOVE FROM OBJECT LIST
REBORN EQU    16       BACK ONTO OBJECT LIST
FORK   EQU    17       START NEW SCRIPT
LFIRE  EQU    18       FIRE LEFT FROM OBJECT <1 BYTE FRAMES>
RFIRE  EQU    19       FIRE RIGHT FROM OBJECT <1 BYTE FRAMES>
LOOPER EQU    20       COUNT, LABEL
GHOST  EQU    21       CREATE A SCRIPT WITH MY OBJECT POINTER
SETRP  EQU    22
GDIE   EQU    23       GHOST DIE (KILL PROCESS, NOT OBJECT)
INCIM  EQU    24
MONO   EQU    25
RPROG  EQU    26
PDEAD  EQU    27

SCTABL FDB    DUMRTS   ZERO IS NOP
       FDB    SETOBP
       FDB    SETIMP
       FDB    LEFTP 
       FDB    RIGHTP
       FDB    DOWNP
       FDB    UPP
       FDB    SETPP
       FDB    SETXVP
       FDB    SETYVP
       FDB    CYCLEP
       FDB    RESTP
       FDB    DIEP
       FDB    EXPP
       FDB    JUMPP
       FDB    HIBP
       FDB    REBP
       FDB    FORKP
       FDB    LFIREP
       FDB    RFIREP
       FDB    LOOPP
       FDB    GHOSTP
       FDB    SRPP
       FDB    SUCIDE
       FDB    INCIMP
       FDB    MONOP
       FDB    RPROGP
       FDB    PDEADP

*      FOUR BYTES OF PARAMS
*
*      1) BOX COLOR
*      2) OBJECT COLOR
*      3) TIME TO DO THIS
*      4) SPECIAL CODE 0=NONE  1=OBJECT IN OWN COLOR                 

MONOP  PULS   D
       STD    LEV2,U
       JSR    HIBP           REMOVE THE GUY FROM THE LIST
       CLR    OFLAV,X        CANCEL THE FLAVOR ON THAT GUY
       JSR    FETCH2
       STY    USER+2,U       BOX COLOR, IMAGE COLOR
       JSR    FETCH2
       STY    USER+4,U       TIME,  CODE
       LDA    OX16,X
       LDB    OY16,X         GET OBJ COORD
       STD    OBJX,X         AND STOE
       STD    OBJID,X        COPY COORD TO SCREEN

MONOP2 NAP    3,MONOP3
MONOP3 LDX    OBJ,U          GET THE OBJECC
       BSR    OFFMEE

       LDD    OBJX,X         GET THE COORD
       ADDA   OXV,X          ADD THE VELOCITY FACTOR
       ADDB   OYV,X
       STD    OBJX,X         STORE BACK

       LDD    USER+2,U       GET COLORS
       LDY    OPICT,X        GET OBJECT IN Y
       JSR    HUMONV         TURN IT ON IN THOSE COLORS

       LDD    OBJX,X         GET TURN ON WE JUST DID
       STD    OBJID,X        AND SAVE

       LDA    USER+5,U
       BEQ    MONOP4         NOTHING SPECIAL

       BSR    ME16
       JSR    DMAON          TURN ON BRAIN IN THE SQUARE

MONOP4 DEC    USER+4,U
       BNE    MONOP2         THANK YOU SIR MAY I HAVE ANOTHER?

       BSR    ME16           UPDATE 
       BSR    OFFMEE         TURN OFF THE BOX 
       JSR    REBP           REBIRTH THE GUY

       JMP    [LEV2,U]

OFFMEE LDD    OBJID,X        USE OBJID TO TURN OFF
       LDY    OPICT,X
       JMP    PCTOFF         TURN OFF WHERE IT IS

ME16   LDD    OBJX,X
       STA    OX16,X
       STB    OY16,X
       CLR    OX16+1,X
       RTS

PDEADP STX    PD,U           SAVE OBJECT FOR 
       JSR    HIBP           OFF OF OBJ LIST
       JMP    PKPRCV

GHOSTP JSR    FETCH2         GET THE OPERAND
       LDX    #SC2           SCRIPT PROCESSOR STEADY STATE
       LDA    #STYPE
       JSR    MSPROC         MAKE A SUPER FOR GHOST
       STY    SCPTR,X        STORE SEPARATE SCRIPT POINTER
       LDD    OBJ,U          GET OBJECT POINTER
       STD    OBJ,X
       LDD    ODESC,U
       STD    ODESC,X
       CLR    LOOP,X
       RTS

LFIREP MAKP   LEFFIR
       BRA    RFIR11

RFIREP MAKP   RIGFIR
RFIR11 LDD    OBJ,U          GET OUR OBJECT
       STD    PD,X
       JSR    FETCH1         GET THE TIME
       STA    PD+2,X
       JMP    RESTP          NAP THE SECOND ARGUMENT TIME

LOOPP  JSR    FETCH1         GET ITERATIONS    
       JSR    FETCH2         GET LABEL
       LDB    LOOP,U         FIRST TIME??
       BNE    LOOP11         NO
       STA    LOOP,U         STORE OUR ITERATIONS
LOOP11 DEC    LOOP,U         1 LESS TIME THROUGH
       BEQ    LOOP22
       STY    SCPTR,U        NEXT INSTR COMES FROM LABEL
LOOP22 RTS                   AND RETURN

FORKP  JSR    FETCH2         GET THE OPERAND
       JMP    OSTART         AND START THE PROCESS UP

HIBP   JSR    KILLOF         TURN OFF OBJECT AND REMOVE FROM OBJECT LIST
       LDD    ,X
       STD    OFREE          GET IT BACK
       RTS

REBP   LDD    OPICT,X        MAKE THE PICTURES MATCH (EUGENE DAMAGE)
       STD    OLDPIC,X
       LDD    OPTR
       STD    ,X
       STX    OPTR           PUT THE OBJECT BACK ON THE LIST
       RTS

JUMPP  JSR    FETCH2         GET THE OPERAND
       STY    SCPTR,U        AND MAKE THIS NEXT SCRIPT ADDRESS
       RTS

EXPP   JSR    KILLOF         CAN THE GUY
       LDA    #ACTHIT+6     EXPLOSION IS KILLOF FOR NOW
       STA    CENTMP+1        Y CENTER OF EXPLOSION
       LDD    #$FF00         HORIZ LASER
       STD    LASDIR
       JSR    EXST           AND EXPLODE THE GUY
       JMP    SUCIDE

DIEP   JSR    KILLOF
       JMP    SUCIDE

RESTP  PULS   X              WAKEUP ADDRESS IS GUY WHO CALLED US
       JSR    FETCH1         GET THE TIME IN A
       JMP    SLEEP

CYCLEP PULS   D              GET RETURN
       STD    LEV2,U         SAVE
       JSR    FETCH2         GET THE OPERANDS
       STY    USER,U         SAVE FOR OUR USE    <TIME>, COUNTS
CYCLE3 LDY    ODESC,U        GET THE DESC
       JSR    INCIMP         USE THE ROUTINE TO KICK TO NEXT IMAGE
       LDA    USER,U         GET THE SLEEP TIME
       LDX    #CYCLE2
       JMP    SLEEP
CYCLE2 DEC    USER+1,U
       BNE    CYCLE3
       JMP    [LEV2,U]       RETURN

SETXVP JSR    FETCH2
       STY    OXV,X
       RTS

SETYVP JSR    FETCH2
       STY    OYV,X
       RTS

SETPP  JSR    FETCH2   GET THE COORDS
       TFR    Y,D      GET THEM IN D
       STB    OY16,X
       STA    OX16,X
       RTS             DONE

SRPP   JSR    FETCH2
       TFR    Y,D
       ADDA   OX16,X
       STA    OX16,X
       ADDB   OY16,X
       STB    OY16,X
       RTS

RPROGP LDA    OBJY,X   GET POSITION
       STA    USER,U   SAVE
       LDA    #$40
       STA    USER+2,U
RPROG1 JSR    RAND
       ANDA   #$7
       ADDA   USER,U
       STA    OBJY,X         CHANGE NEW IMAGE
       NAP    1,RPROG2
RPROG2 LDX    OBJ,U
       JSR    RAND
       ANDA   #$7
       NEGA
       ADDA   USER,U
       STA    OBJY,X   
       NAP    1,RPROG3
RPROG3 LDX    OBJ,U
       DEC    USER+2,U
       BNE    RPROG1
       LDA    USER,U
       STA    OBJY,X
       JMP    SUCIDE

LEFTP  JSR    FETCH1   GET NUMBER OF STEPS
       JMP    [4,Y]    AND DO THE LEFT ROUTINE FOR THIS OBJ

RIGHTP JSR    FETCH1
       JMP    [6,Y]

DOWNP  JSR    FETCH1
       JMP    [8,Y]

UPP    JSR    FETCH1
       JMP    [10,Y]

SETOBP JSR    FETCH2   GET THE OBJECT DESC
       STY    ODESC,U  SAVE THE OBJECT DESCRIPTOR
       CLRA            SET TO IMAGE 1
SETOB0 STA    IMAG,U   SAVE AS CURRENT IMAGE
SETOB1 BSR    GETIM    GET THE CURRENT IMAGE OF THE CURRENT OBJECT IN D
       LDX    OBJ,U
       STD    OPICT,X  STORE THE IMAGE
       RTS             DONE

INCIMP LDA    IMAG,U   GET CURRENT IMAGE
       INCA
       CMPA   2,Y      TOO LARGE??
       BLO    SETOB0
       CLRA
       BRA    SETOB0   SET THIS AS THE OBJECTS IMAGE

SETIMP JSR    FETCH1   GET THE IMAGE NUMBER
       BRA    SETOB1   SAVE AS THE CURRENT IMAGE

*      A GETS IMAGE NUMBER 0-NIMAG
*      D RETURNS THE IMAGE

GETIM  PSHS   Y,U
       LDU    CRPROC
       LDY    ODESC,U
       LDB    3,Y      BYTES PER IMAGE  
       MUL
       ADDD   [0,Y]      ADD BASE IMAGE
       PULS   Y,U,PC

FETCH2 PSHS   U,X
       BSR    SETUX
       LDY    ,X++
FETCH3 STX    SCPTR,U
       PULS   U,X,PC

SETUX  LDU    CRPROC
       LDX    SCPTR,U
       RTS

FETCH1 PSHS   U,X
       BSR    SETUX
       LDA    ,X+
       BRA    FETCH3

BRL    CLRB                  IMAGE ZERO
       STD    USER,U
       LDA    12,Y           GET STEP SIZE
       NEGA
       BRA    BRWALK

BRR    LDB    #3             START WITH IMAGE 3
       STD    USER,U
       LDA    12,Y
       CLRB
       BRA    BRWALK

BRD    LDB    #6
       STD    USER,U
       LDB    12,Y
       BRA    BRWAL1

BRU    LDB    #9
       STD    USER,U
       LDB    12,Y
       NEGB
BRWAL1 CLRA

BRWALK STD    USER+2,U       USER = NUMBER OF STEPS
*                            USER+1 = OFFSET OF FIRST IMAGE
*                            USER+2 = X OFFSET PER STEP
*                            USER+3 = Y OFFSET PER STEP

       PULS   D
       STD    LEV2,U
       LDD    #ANAEND-1
       STD    USER+4,U
BANA1  LDY    ODESC,U        GET DESCRIPTOR
       LDX    #BANA2
       LDA    13,Y           GET NAP TIME
       JMP    SLEEP

BANA2  LDX    OBJ,U          GET OBJECT
       LDD    USER+2,U       GET DELTAX,DELTAY
       BSR    DYDX
       LDY    USER+4,U       GET ANA POINTER
       LEAY   1,Y            INCREMENT
       CMPY   #ANAEND        TOO FAR??
       BLO    BANA3          NO
       LDY    #ANATAB
BANA3  LDA    ,Y             GET CURRENT GUY
       STY    USER+4,U       SAVE BACK
       ADDA   USER+1,U       ADD BASE FOR THIS DIRECTION
       JSR    SETOB0         PUT THAT IMAGE DOWN
       DEC    USER,U         1 LESS STEP
       BNE    BANA1
       JMP    [LEV2,U]

DYDX   PSHS   B
       CLRB
       ASRA
       RORB
       ADDD   OX16,X
       STD    OX16,X
       PULS   B
       ADDB   OY16,X
       STB    OY16,X
       RTS

ANATAB FCB    0,1,0,2
ANAEND EQU    *

ANAL   CLRB                  ZERO OFFSET
       BRA    ANAHUM

ANAR   LDB    #13            OFFSET FOR RIGHT
       BRA    ANAHUM

ANAD   LDB    #26
       BRA    ANAHUM

ANAU   LDB    #39

ANAHUM LDX    [12,Y]          GET POINTER TO VECTOR
       ABX                   OUR PART OF THE SCRIPT

       STX    USER,U         SAVE POINTER
       STA    USER+2,U       SAVE STEP COUNT

       PULS   D              SAVE CALLER
       STD    LEV2,U

ANA1   NAP    8,ANA2
ANA2   BSR    ANASUB
       DEC    USER+2,U       1 LESS STEP
       BNE    ANA1
       JMP    [LEV2,U]

ANASUB LDX    OBJ,U          GET OBJECT
       LDY    USER,U
       LDD    1,Y            GET X,Y  OFFSET
       BSR    DYDX
       LDA    ,Y
       LSRA
       LSRA                  FORM IMAGE NUMBER
       JSR    SETOB0         STORE AND SAVE IMAGE IN OBJECT

       LEAY   3,Y            PUSH TO NEXT
       LDB    ,Y
       CMPB   #$FF
       BNE    ANA3
       LEAY   -12,Y
ANA3   STY    USER,U         SAVE Y
       RTS


*      OBJECT DESCRIPTORS

MOMMY  FDB    MOMPIC
       FCB    12       12 IMAGES
       FCB    4        4 BYTES PER
       FDB    ANAL    
       FDB    ANAR
       FDB    ANAD
       FDB    ANAU
       FDB    HUMANA   ANIMATION SCRIPTOR

PROGIE FDB    PRGPIC
       FCB    1
       FCB    4

DADDY  FDB    DADPIC
       FCB    12
       FCB    4
       FDB    ANAL
       FDB    ANAR
       FDB    ANAD
       FDB    ANAU
       FDB    HUMANA

MIKEY  FDB    KIDPIC
       FCB    12
       FCB    4
       FDB    ANAL
       FDB    ANAR
       FDB    ANAD
       FDB    ANAU
       FDB    HUMANA

HULK   FDB    HLKPIC
       FCB    12
       FCB    4
       FDB    ANAL
       FDB    ANAR
       FDB    ANAD
       FDB    ANAU
       FDB    HLKANA

BRAIN  FDB    BRNPIC
       FCB    12
       FCB    4
       FDB    BRL
       FDB    BRR
       FDB    BRD
       FDB    BRU
       FCB    2        NUMBER OF PIXELS
       FCB    8

GRUNT  FDB    ROBPIC
       FCB    3
       FCB    4

POSTS  FDB    PSTPIC
       FCB    36        4 IMAGES EACH POST, 4 BYTES EACH
       FCB    4  

ENF    FDB    ENGPIC
       FCB    6
       FCB    4

YOU    FDB    MANPIC
       FCB    12
       FCB    4
       FDB    BRL
       FDB    BRR
       FDB    BRD
       FDB    BRU
       FCB    1
       FCB    2

SQUARE FDB    SQPIC
       FCB    9
       FCB    4

CIRC   FDB    CIRPIC
       FCB    8
       FCB    4

TANKG  FDB    TNGPIC
       FCB    5
       FCB    6

TANK   FDB    TNKPIC
       FCB    4
       FCB    4

POINTS FDB    P1KD
       FCB    5
       FCB    4

SKULLV FDB    SKLPIC
       FCB    1
       FCB    4

ACCHNG TST    ATFLAG         COIN SEEN, IS THIS THE FIRST??
       BNE    ACCHG1
       INC    ATFLAG         MAKE IT COIN MODE
       JMP    FAMPAG

ACCHG1 INC    ATFLAG         INDICATE COIN SEEN

ACPROC LDD    #$1005         AREA TO ERASE
       LDX    #$50EE         AREA TO ERASE
       JSR    BLKCLR         CLEAR OUT OLD REGION
       LDB    CREDIT
       STB    PD,U           SAVE CREDITS
       LDA    #ACRM
       BSR    CALL5          CALL WITHOUT HURTING COLOR,FLAVOR

ACPR4  LDA    #$20
       STA    PD+3,U
ACPR2  NAP    8,ACPR1
ACPR1  BSR    HCHEKK
       LDA    CREDIT         GET CREDITS
       CMPA   PD,U           COMPARE TO OLD NUMBER
       BNE    ACCHNG         NOT THE SAME, REFRESH
       DEC    PD+3,U         TIME FOR REPLAY MESSAGE??
       BNE    ACPR2          NOPE
       LDX    #REPLAY
       JSR    RCMOSB   GET THE REPLAY BYTE
       TSTB            SEE IF A REPLAY
       BEQ    ACPR4    NO MESSAGE DO NOTHING
       BSR    BCLEAR
       LDA    #EXTMES
       BSR    CALL5    PRINT THE REPLAY MESSAGE 
       LDA    #$A
       STA    PD+3,U
ACPR8  LDA    CREDIT
       CMPA   PD,U
       BEQ    ACPR5
ACPR7  BSR    BCLEAR   CLEAR THE MESSAGE
       BRA    ACCHNG   AND PRINT CREDITS
ACPR5  NAP    8,ACPR6
ACPR6  BSR    HCHEKK
       DEC    PD+3,U
       BNE    ACPR8
       BSR    BCLEAR
       BRA    ACPROC   GO BACK TO CREDITS

HCHEKK LDA    ATFLAG
       BMI    NOSTIK
       LDA    PIA2
       CMPA   #$41     BOTH STICKS UP?
       LBEQ   ENDP     YEP...GO TO HIGH SCORE TABLE
NOSTIK RTS

CALL5  PSHS   D
       LDA    TXFLAV
       LDB    TEXCOL         SAVE THE CURRENT COLOR
       PSHS   D
       LDD    2,S            GET PASSED A,B
       JSR    WRD5V          PUT EM OUT
       PULS   D              RESTORE COLOR
       STA    TXFLAV
       STB    TEXCOL
       PULS   D,PC

BCLEAR PSHS   D,X
       LDX    #$20EE
       LDD    #$5005
       JSR    BLKCLR         CLEAR OUT MESSAGE
       PULS   D,X,PC

       FCC    ' COPYRIGHT 1982 WILLIAMS ELECTRONICS INC.'
       FCC    '  ALL RIGHTS RESERVED.'
       FCC    '  "ROBOTRON: 2084"  '


SPACE  EQU    $3A
EXPT   EQU    $3B
COMMA  EQU    $3C
PERIOD EQU    $3D
COLON  EQU    $3F
HYPHEN EQU    $40
LPAREN EQU    $5B
RPAREN EQU    $5C

HISTO  FCB    CLEARM,TOP+$18,$80,COLOR,$DD
       FCC    'INSPIRED BY HIS NEVER ENDING'
       FCB    NEWLIN
       FCC    'QUEST FOR PROGRESS'
       FCB    COMMA,NEWLIN
       FCC    'IN '                                
       FCB    COLOR,$AA
       FCC    '2084'
       FCB    COLOR,$DD
       FCC    ' MAN PERFECTS THE '  
       FCB    COLOR,$AA
       FCC    'ROBOTRONS'
       FCB    COLON,SNOOZE,$20,NEWLIN,NEWLIN,COLOR,$FF
       FCC    'A ROBOT SPECIES SO ADVANCED THAT'
       FCB    NEWLIN
       FCC    'MAN IS INFERIOR TO HIS OWN CREATION'
       FCB    PERIOD,NEWLIN,NEWLIN,SNOOZE,$20,COLOR,$DD
       FCC    'GUIDED BY THEIR INFALLIBLE LOGIC'
       FCB    COMMA,NEWLIN
       FCC    'THE '
       FCB    COLOR,$AA
       FCC    'ROBOTRONS'
       FCB    COLOR,$DD
       FCC    ' CONCLUDE'                   
       FCB    COLON,NEWLIN,NEWLIN,SNOOZE,$30,COLOR,$AA
       FCC    'THE HUMAN RACE IS INEFFICIENT'
       FCB    COMMA,NEWLIN
       FCC    'AND THEREFORE MUST BE DESTROYED'
       FCB    PERIOD,$70,CLEARM,TOP+$16,$C0,COLOR,$DD
       FCB    SCRPT        
       FDB    PLAYRR
       FCC    'YOU ARE THE LAST HOPE OF MANKIND'
       FCB    PERIOD,NEWLIN,NEWLIN,$60,COLOR,$FF
       FCC    'DUE TO A GENETIC ENGINEERING ERROR'
       FCB    COMMA,NEWLIN                  
       FCC    'YOU POSSESS'                  
       FCC    ' SUPERHUMAN'
       FCC    ' POWERS'
       FCB    PERIOD,COLOR,$DD
       FCB    NEWLIN,NEWLIN
       FCC    'YOUR MISSION IS TO'                      
       FCB    COLOR,$AA
       FCC    ' STOP THE ROBOTRONS'
       FCB    COMMA,NEWLIN,COLOR,$DD
       FCC    'AND '
       FCB    COLOR,$AA
       FCC    'SAVE THE LAST HUMAN FAMILY'
       FCB    COLON,SCRPT
       FDB    FAMSCR
       FCB    SNOOZE,$20               CLEAR OUT
       FCB    MESS,$1C,MMOM,SNOOZE,$58           
       FCB    MESS,$18,MDAD,SNOOZE,$58
       FCB    MESS,$14,MKID          
       FCB    $70,MESS,$10,NULMES
       FCB    $70                               
       FCB    CLEARM,TOP+$38,$80,COLOR,$33,GRUNTS                
       FCC    'THE FORCE OF GROUND ROVING'
       FCB    NEWLIN
       FCC    'UNIT NETWORK TERMINATOR '
       FCB    NEWLIN,COLOR,$AA,MESS,$74,MGRUNT,LPAREN

       FCC    'GRUNT'
       FCB    RPAREN,SPACE            
       FCC    'ROBOTRONS'                     
       FCB    COLOR,$33
       FCC    ' SEEK TO'
       FCB    NEWLIN
       FCC    'DESTROY YOU'
       FCB    PERIOD,$FF     
       FCB    CLEARM,TOP+$38,$80,COLOR,$55,SCRPT
       FDB    SCHULK              START THE HULK 
       FCC    'THE' 
       FCB    COLOR,$AA
       FCB    MESS,LEFT,MHULK
       FCC    ' HULK ROBOTRONS '
       FCB    COLOR,$55
       FCC    'SEEK OUT'
       FCB    NEWLIN
       FCC    'AND ELIMINATE THE LAST HUMAN FAMILY'
       FCB    PERIOD,$F9    
       FCB    MESS,LEFT,NULMES
       FCB    $80    
       FCB    CLEARM,TOP+$50,$80,COLOR,$DD,SCRPT
       FDB    SCSC
       FCC    'THE'
       FCB    COLOR,$AA
       FCB    MESS,$2C,MSPCU
       FCC    ' SPHEREOIDS AND QUARKS  '
       FCB    COLOR,$DD,NEWLIN
       FCC    'ARE PROGRAMMED TO MANUFACTURE'
       FCB    NEWLIN,COLOR,$AA,MESS,$33,MENTK

       FCC    'ENFORCER AND TANK ROBOTRONS'
       FCB    PERIOD                             ENFORCER AND TANK MESS
       FCB    $A0,MESS,$10,NULMES
       FCB    CLEARM,TOP+$38,$80,COLOR,$FF
       FCC    'BEWARE OF THE INGENIOUS'
       FCB    NEWLIN,COLOR,$AA,MESS,$20,MBRAIN
       FCC    'BRAIN ROBOTRONS'
       FCB    COLOR,$FF
       FCC    ' THAT POSSESS'
       FCB    NEWLIN
       FCC    'THE POWER TO REPROGRAM'
       FCB    NEWLIN
       FCC    'HUMANS'      
       FCC    ' INTO SINISTER '
       FCB    COLOR,$AA              
       FCC    'PROGS'
       FCB    PERIOD,$73,MESS,$2C,MPROG,$80,SCRPT
       FDB    POSTER
       FCB    CLEARM,TOP+$44,$80
       FCB    SNOOZE,$20
       FCB    MESS,$10,NULMES          CLEAR OUT PROG BEFORE BRAIN EXPS
       FCB    SNOOZE,$40
       FCB    COLOR,$DD
       FCC    'AS YOU STRUGGLE TO SAVE'  
       FCB    NEWLIN
       FCC    'HUMANITY'              
       FCB    COMMA
       FCC    ' BE SURE TO AVOID'
       FCB    NEWLIN
       FCB    COLOR,$AA
       FCC    'ELECTRODES'
       FCB    COLOR,$DD
       FCC    ' IN YOUR PATH'
       FCB    PERIOD 
       FCB    $D0,$AD,DONE   

DUMPLR FCB    SCRPT
       FDB    HELPME         DO THE HELPME SCRIPT
       FCB    $FF,DONE2

ACTHIT EQU    $A0            HEIGHT OF ACTION

PLAYRR FCB    SETOB
       FDB    YOU
       FCB    SETPOS,$18,ACTHIT
       FCB    GHOST
       FDB    PSHOOT
       FCB    MRIGHT,$60
       FCB    SETIM,6        TURN AND FACE THE NICE PEOPLE 
       FCB    REST,$90
       FCB    REST,$FF
       FCB    REST,$FF
       FCB    MLEFT,$50
       FCB    MRIGHT,$68
       FCB    SETIM,3
       FCB    REST,$80       SHOOT GRUNT TIME
       FCB    REST,$80
       FCB    SETIM,6        BACK TO STRAIGHT AHEAD
       FCB    REST,$40
       FCB    MLEFT,$18      ALAS DO I SMELL HULK??
       FCB    MDOWN,$23      SHOW WE CAN MOVE DOWN AND FIRE LEFT
       FCB    SETIM,0        LOOK LEFT
       FCB    REST,$FF
       FCB    SETIM,6        LOOK STRAIGHT
       FCB    REST,$E0
       FCB    MUP,$3C
       FCB    SETIM,3        LOOK RIGHT AS WE SHOOT ENFORCER
       FCB    REST,$38
       FCB    MDOWN,$38      DOWN TO THE TANK
       FCB    SETIM,3        LOOK AT THE TANK
       FCB    REST,$20
       FCB    SETIM,6        LOOK AHEAD AGAIN
       FCB    REST,$E3
       FCB    MUP,$20
       FCB    SETIM,0        LOOK LEFT
       FCB    REST,$3C
       FCB    SETIM,6
       FCB    REST,$F1
       FCB    REST,$30
       FCB    MDOWN,$20      MOVE DOWN TO SHOOT PROG AND BRAIN
       FCB    SETIM,0
       FCB    REST,$20
       FCB    SETIM,6
       FCB    REST,$90
       FCB    REST,$90
       FCB    MRIGHT,$80
       FCB    EXP            EXPLODE THE PLAYER INTO AN ELECTRODE

PSHOOT FCB    REST,$FF
       FCB    REST,$60
PLOP1  FCB    RFIRE,$1D,5 
       FCB    LOOPER,10  
       FDB    PLOP1
PLOP2  FCB    LFIRE,$1B,5
       FCB    LOOPER,8
       FDB    PLOP2
PLOP3  FCB    RFIRE,$1D,5
       FCB    LOOPER,20
       FDB    PLOP3
PLOP4  FCB    LFIRE,$1B,5
       FCB    LOOPER,5
       FDB    PLOP4
PLOP5  FCB    RFIRE,$1D,5
       FCB    LOOPER,10
       FDB    PLOP5
PLOP6  FCB    REST,$FF
       FCB    REST,$FF
       FCB    REST,$50
PLOP7  FCB    RFIRE,$B,$10   SHOOT THE 14 GRUNTS
       FCB    LOOPER,14
       FDB    PLOP7
       FCB    REST,$E0       TIME TILL HULK SHOTS
PLOP8  FCB    LFIRE,$B,$B    HIT THE MUTHA
       FCB    LOOPER,$A     
       FDB    PLOP8
       FCB    REST,$FF,REST,$E3
       FCB    REST,$10
       FCB    RFIRE,$E,$80
       FCB    RFIRE,$18,$8
PLOP9  FCB    RFIRE,$20,$8
       FCB    LOOPER,3
       FDB    PLOP9
       FCB    REST,$FF
       FCB    REST,$44
       FCB    LFIRE,$09,$C0
       FCB    REST,$E0
       FCB    LFIRE,$8,$FF
       FCB    REST,$90
       FCB    RFIRE,$4,$10        SHOOT ELECTRODES
       FCB    RFIRE,$6,$10
       FCB    RFIRE,8,1
       FCB    GDIE           DEAD GHOST

FAMSCR FCB    SETOB 
       FDB    MOMMY
       FCB    SETPOS,$18,ACTHIT
       FCB    FORK
       FDB    DADDP
       FCB    FORK
       FDB    MIKEP
       FCB    MRIGHT,$28          WALK ON MOMMY
       FCB    HIB                 EASY, WHILE WE CHANGE IMAGE
       FCB    SETOB
       FDB    POINTS
       FCB    SETIM,0
       FCB    SETRP,0,4         
       FCB    REBORN              BACK ON
       FCB    REST,$30            DONE WITH MOMMY ETC.
       FCB    HIB
       FCB    SETOB 
       FDB    DADDY
       FCB    SETPOS,$18,ACTHIT
       FCB    REST,$40
       FCB    REBORN,MRIGHT,$30
       FCB    MUP,$10
       FCB    MRIGHT,$10
       FCB    MDOWN,$13
       FCB    MLEFT,$12
MOMDED FCB    HIB      
       FCB    SETOB
       FDB    SKULLV
       FCB    REBORN
       FCB    REST,$80,DIE

DADDP  FCB    SETOB
       FDB    DADDY
       FCB    SETPOS,$18,ACTHIT
       FCB    HIB                 NO IMAGE YET
       FCB    REST,$70
       FCB    REBORN
       FCB    MRIGHT,$1E 
       FCB    HIB
       FCB    SETOB
       FDB    POINTS
       FCB    SETIM,1
       FCB    SETRP,0,4         
       FCB    REBORN              BACK ON
       FCB    REST,$30,DIE        DONE WITH MOMMY ETC.

MIKEP  FCB    SETOB
       FDB    MIKEY
       FCB    SETPOS,$18,ACTHIT
       FCB    HIB
       FCB    REST,$D8          
       FCB    REBORN
       FCB    MRIGHT,$15
       FCB    HIB
       FCB    SETOB
       FDB    POINTS
       FCB    SETIM,2
       FCB    SETRP,0,4         
       FCB    REBORN              BACK ON
       FCB    REST,$30,HIB,SETOB
       FDB    MIKEY
       FCB    SETPOS,$18,ACTHIT,REST,$C0,REBORN
       FCB    $4,$4,$5,$8,$4,$10,$5,$18,$3,$1    VOODOO TO GET MIKEY DEAD
       FCB    JUMP
       FDB    MOMDED

GS1    FCB    SETOB
       FDB    GRUNT
       FCB    SETPOS,$86,ACTHIT-27 
GSS1   FCB    REST,$10
       FCB    SETRP,-2,4
       FCB    SETIM,1
       FCB    REST,$8
       FCB    SETRP,-3,4
       FCB    SETIM,0
       FCB    REST,$20
       FCB    SETRP,-2,4
       FCB    SETIM,2
       FCB    LOOPER,2
       FDB    GSS1
       FCB    JUMP
       FDB    GSSS1          CONTINUE IN LINE

GS2    FCB    SETOB
       FDB    GRUNT
       FCB    SETPOS,$88,ACTHIT-16
GSS2   FCB    REST,$9
       FCB    SETRP,-2,2
       FCB    SETIM,2
       FCB    REST,$12
       FCB    SETRP,-2,2
       FCB    SETIM,0
       FCB    REST,$12
       FCB    SETRP,-2,2
       FCB    SETIM,1
       FCB    REST,$B
       FCB    SETRP,-2,2
       FCB    SETIM,0
       FCB    LOOPER,2
       FDB    GSS2
       FCB    JUMP
       FDB    GSSS1          AND GO INTO THE MAINSTREAM

GS3    FCB    SETOB
       FDB    GRUNT
       FCB    SETPOS,$88,ACTHIT+28
GSS3   FCB    REST,$10
       FCB    SETRP,-2,-3
       FCB    SETIM,1
       FCB    REST,$6
       FCB    SETRP,-2,-3
       FCB    SETIM,0
       FCB    REST,$1A
       FCB    SETRP,-2,-3
       FCB    SETIM,2
       FCB    REST,$8
       FCB    SETRP,-2,-3
       FCB    LOOPER,2
       FDB    GSS3
       FCB    JUMP
       FDB    GSSS1

GS4    FCB    SETOB
       FDB    GRUNT
       FCB    SETPOS,$88,ACTHIT+17
       FCB    REST,$4
       FCB    SETRP,-2,-2
       FCB    SETIM,1
GSS4   FCB    REST,$9
       FCB    SETRP,-3,-2
       FCB    SETIM,0
       FCB    REST,$18
       FCB    SETRP,-2,-2
       FCB    SETIM,2
       FCB    REST,$15
       FCB    SETRP,-2,-2
       FCB    LOOPER,2
       FDB    GSS4
GSSS1  FCB    SETIM,0
       FCB    SETRP,-4,2
       FCB    REST,$4
       FCB    SETIM,2
       FCB    SETRP,-2,-2
       FCB    REST,$20
       FCB    SETIM,0
       FCB    SETRP,-4,2
       FCB    REST,$12
       FCB    EXP

SCHULK FCB    SETOB
       FDB    HULK
       FCB    SETPOS,$10,ACTHIT+$20
       FCB    GHOST
       FDB    HBOUNC 
       FCB    MRIGHT,$1C
       FCB    MUP,$10
       FCB    MRIGHT,$36
       FCB    DIE

HBOUNC FCB    REST,$7C
HBO1   FCB    SETRP,-2,0
       FCB    REST,$B
       FCB    LOOPER,$A
       FDB    HBO1
       FCB    GDIE           AND DIE

SCSC   FCB    SETOB
       FDB    CIRC
       FCB    SETPOS,$88,ACTHIT-25
       FCB    FORK
       FDB    SQP
       FCB    FORK
       FDB    TKP
       FCB    FORK
       FDB    ENP
       FCB    FORK
       FDB    DADD2             START DADDY TO GET REPROGRAMMED BY BRAIN
       FCB    SETXV,$FF,$40
       FCB    CYCLE,2,$60
       FCB    DIE

SQP    FCB    SETOB
       FDB    SQUARE
       FCB    SETPOS,$0A,ACTHIT+20
       FCB    SETXV,00,$C0
       FCB    CYCLE,2,$60
       FCB    DIE

TKP    FCB    SETOB
       FDB    TANKG
       FCB    SETPOS,$10,ACTHIT+20
       FCB    HIB
       FCB    REST,$10
       FCB    REBORN
TKP1   FCB    REST,$14
       FCB    SETRP,0,-1
       FCB    INCIM
       FCB    LOOPER,3
       FDB    TKP1
       FCB    HIB
       FCB    SETOB
       FDB    TANK
       FCB    REBORN
       FCB    SETXV,00,$80
       FCB    CYCLE,2,$70
       FCB    EXP            AND EXPLODE!

ENP    FCB    SETOB
       FDB    ENF
       FCB    SETPOS,$80,ACTHIT-25
       FCB    HIB
       FCB    REST,$10
       FCB    REBORN
ENF1   FCB    REST,$14
       FCB    INCIM
       FCB    LOOPER,5
       FDB    ENF1
       FCB    SETXV,$FF,$A0
       FCB    REST,$30
       FCB    EXP            AND GET SHOT

DADD2  FCB    SETOB
       FDB    MOMMY
       FCB    SETPOS,$0A,ACTHIT+20
       FCB    HIB
       FCB    REST,$C0
       FCB    REBORN
       FCB    MRIGHT,$20
       FCB    FORK
       FDB    BRAING
       FCB    MDOWN,$A
       FCB    MLEFT,$8 
       FCB    MRIGHT,$1A
       FCB    SETRP,3,$FE
       FCB    REST,2
       FCB    GHOST
       FDB    PSHAKE
       FCB    MONO,$AA,$BB,$28,$0
       FCB    FORK
       FDB    PSHADO
       FCB    SETXV,$2,0     MAKE THIS PROG WALK RIGHT
       FCB    MONO,$0,$AA,$2C,0
       FCB    DIE 

PSHADO FCB    SETOB
       FDB    MOMMY
       FCB    SETPOS,$31,ACTHIT+28
       FCB    SETIM,3
       FCB    HIB
       FCB    REST,$3
       FCB    FORK
       FDB    PSHAD1
       FCB    FORK
       FDB    PSHAD2 
       FCB    REST,$4
       FCB    REBORN
       FCB    SETXV,$2,0
       FCB    MONO,$EE,$0,$2C,0
       FDB    DIE    

PSHAD1 FCB    SETOB
       FDB    MOMMY
       FCB    SETPOS,$31,ACTHIT+28
       FCB    HIB
       FCB    REST,$4       
       FCB    SETIM,3
       FCB    REBORN      
       FCB    SETXV,$2,0
       FCB    MONO,$EE,$0,$2C,0
       FCB    DIE

PSHAD2 FCB    SETOB
       FDB    MOMMY
       FCB    SETPOS,$31,ACTHIT+28
       FCB    SETIM,3        LOOK RIGHT
       FCB    SETXV,$2,0
       FCB    MONO,$EE,$0,$2C,0
       FCB    DIE

PSHAKE FCB    RPROG                  

BRAING FCB    SETOB
       FDB    BRAIN
       FCB    SETPOS,$0A,ACTHIT
       FCB    MRIGHT,$10
       FCB    FORK
       FDB    CRUSER
       FCB    MDOWN,$0C
       FCB    MRIGHT,$10
       FCB    MONO,$BB,$BB,$38,1       BRAINER
       FCB    MRIGHT,$B 
       FCB    EXP

CRUSER FCB    SETOB
       FDB    CRUSM
       FCB    SETPOS,$1A,ACTHIT+4
       FCB    SETXV,$0,$60
       FCB    REST,$40
       FCB    DIE

CRUSM  FDB    CRUSA
       FCB    1,4

CRUSA  FDB    CRUSB

CRUSB  FCB    9,2
       FDB    CRUSC

CRUSC  FCB    $DD,$DD,$DD,$DD,$DD,$DD,$DD,$DA,$A0
       FCB    $DD,$DD,$DD,$DD,$DD,$DD,$DD,$DA,$A0

POSTER FCB    SETOB
       FDB    POSTS
       FCB    SETPOS,$86,ACTHIT+$21
       FCB    SETIM,12
       FCB    HIB
       FCB    REST,$48
       FCB    FORK
       FDB    POST1
       FCB    REST,$A
       FCB    FORK
       FDB    POST2
       FCB    REST,$A
       FCB    FORK
       FDB    POST3
       FCB    REST,$10

POST4  FCB    REBORN
       FCB    MONO,0,$DD,$B8,0
       FCB    PDEAD

POST1  FCB    SETOB
       FDB    POSTS
       FCB    SETPOS,$68,ACTHIT+$21
       FCB    MONO,0,$FF,$8D,0
       FCB    PDEAD

POST2  FCB    SETOB
       FDB    POSTS
       FCB    SETPOS,$72,ACTHIT+$21 
       FCB    SETIM,4
       FCB    MONO,0,$AA,$91,0
       FCB    PDEAD

POST3  FCB    SETOB
       FDB    POSTS
       FCB    SETPOS,$7C,ACTHIT+$21
       FCB    SETIM,8
       FCB    MONO,0,$CC,$92,0
       FCB    PDEAD

DUMHIT EQU    $68

HELPME FCB    SETOB 
       FDB    MOMMY
       FCB    SETPOS,$50,DUMHIT
       FCB    FORK
       FDB    DUMYOU
       FCB    FORK
       FDB    DAD1
       FCB    FORK
       FDB    MIK1
       FCB    FORK
       FDB    MOM2
       FCB    FORK
       FDB    MIK2
       FCB    MRIGHT,$9 
       FCB    HIB
       FCB    SETOB 
       FDB    POINTS
       FCB    SETIM,0
       FCB    SETRP,0,5
       FCB    REBORN
       FCB    REST,$50
       FCB    DIE

DUMYOU FCB    SETOB
       FDB    YOU
       FCB    SETPOS,$68,DUMHIT+2
       FCB    MLEFT,$80
       FCB    DIE

DAD1   FCB    SETOB
       FDB    DADDY
       FCB    SETPOS,$48,DUMHIT+1
       FCB    HIB
       FCB    REST,$10
       FCB    REBORN
       FCB    MRIGHT,$A 
       FCB    HIB
       FCB    SETOB        
       FDB    POINTS
       FCB    SETIM,1
       FCB    SETRP,0,4
       FCB    REBORN
       FCB    REST,$50
       FCB    DIE

MIK1   FCB    SETOB
       FDB    MIKEY
       FCB    SETPOS,$40,DUMHIT+3
       FCB    HIB
       FCB    REST,$20
       FCB    REBORN
       FCB    MRIGHT,$C 
       FCB    HIB
       FCB    SETOB
       FDB    POINTS
       FCB    SETIM,2
       FCB    SETRP,0,2
       FCB    REBORN
       FCB    REST,$50
       FCB    DIE

MOM2   FCB    SETOB
       FDB    MOMMY
       FCB    SETPOS,$38,DUMHIT
       FCB    HIB
       FCB    REST,$30
       FCB    REBORN
       FCB    MRIGHT,$D 
       FCB    HIB
       FCB    SETOB 
       FDB    POINTS
       FCB    SETIM,3
       FCB    SETRP,0,5
       FCB    REBORN
       FCB    REST,$50
       FCB    DIE

MIK2   FCB    SETOB
       FDB    MIKEY
       FCB    SETPOS,$30,DUMHIT+3
       FCB    HIB
       FCB    REST,$40
       FCB    REBORN
       FCB    MRIGHT,$F 
       FCB    HIB
       FCB    SETOB
       FDB    POINTS
       FCB    SETIM,4
       FCB    SETRP,0,2
       FCB    REBORN
       FCB    REST,$50
       FCB    DIE
