 OPT NOL
 LIB RRF
 OPT LIS
 STTL TRANSPORTER.........
OPOINT EQU    0
OW     EQU    OPOINT+2    OFFSETS IN MEMORY BLOCK 
OH     EQU    OW+1
OPT    EQU    OH+1      POINTER TO RAM DATA TO BE USED BY DMA
PICPTR EQU    OPT+2    OBJECT                             
BYTCNT EQU    PICPTR+2 NUMBER OF BYTES IN OBJECT
FRPTR  EQU    BYTCNT+1 POINTER TO CURRENT FRAME
GLTPTR EQU    FRPTR+2 
DATA   EQU    GLTPTR+2  DATA AREA 1 IS THIS DISTANCE INTO SECOND BLOCK
BKSIZE EQU    116+DATA    SIZE OF ALLOCATION BLOCKS 
 ORG HXTAB
FREEP  RMB    2
TRPTR  RMB    2
GTEMP  RMB    2
BYTES  RMB    1
LAST   RMB    2
TRRAM  EQU    *
       RMB    10*BKSIZE
TREND  EQU    *

       ORG    RTORG    PLACE TO RUN THE CODE FROM
 JMP TRNSTV

*****    T R A N S P O R T E R
*SOUNDS
TR1SND FCB $FF,$01,$01,$13,0  CLEAR THE SYSTEM

*      COLORS FOR PIXEL GLITTER

GLIT0  FCB    $99,$22,$55,$11,$99,$22,$55,$11,$99,$22  WH,RED,YEL,ORANG
       FCB    $55,$11,$99,$22,$55,$11,$99,$22,$55,$11 
GLIT1  FCB    $AA,$CC,$AA,$CC,$AA,$CC,$AA,$CC,$AA,$CC  CY,CY
       FCB    $AA,$CC,$AA,$CC,$AA,$CC,$AA,$CC,$AA,$CC 
GLIT2  FCB    $99,$77,$99,$77,$99,$77,$99,$77,$99,$77  WH,BL
       FCB    $99,$77,$99,$77,$99,$77,$99,$77,$99,$77
GLIT3  FCB    $11,$55,$11,$55,$11,$55,$11,$55,$11,$55  RED,YEL
       FCB    $11,$55,$11,$55,$11,$55,$11,$55,$11,$55
GLIT4  FCB    $FF,$EE,$DD,$CC,$BB,$AA,$FF,$EE,$DD,$CC
       FCB    $BB,$AA,$FF,$EE,$DD,$CC,$BB,$AA,$FF,$EE
GLIT5  FCB    $11,$66,$77,$BB,$AA,$11,$66,$77,$BB,$AA
       FCB    $11,$66,$77,$BB,$AA,$11,$66,$77,$BB,$AA          
GLIT6  FCB    $33,$55,$33,$55,$AA,$33,$55,$33,$55,$AA  PURP,YEL + SOME WH.
       FCB    $33,$55,$33,$55,$AA,$33,$55,$33,$55,$AA

GLITAB FDB    GLIT0
       FDB    GLIT1
       FDB    GLIT2
       FDB    GLIT3
       FDB    GLIT4
       FDB    GLIT5    
       FDB    GLIT6
       FDB    GLIT4

       FCC    ' COPYRIGHT 1982 WILLIAMS ELECTRONICS INC. '

TRINIT PSHS   X,Y 
       LDX    #TRRAM   POINT AT FIRST BLOCK
       STX    FREEP    FREE LIST
TRIN0  LEAY   BKSIZE,X FIND NEXT BLOCK
       STY    ,X       MAKE THIS ONE POINT AT NEXT ONE
       CMPY   #TREND   DID WE JUST POINT AT NON EXISTANT??
       BHS    TRIN1    YEP
       LEAX   ,Y       NOW POINT AT NEXT
       BRA    TRIN0
TRIN1  LDY    #0       END OF THE LINE
       STY    ,X
       STY    TRPTR    NO ONE ON LIST
       PULS   X,Y,PC

*      ONDRV  -  TRANSPORT GUY ON...
*      X = POINTER TO H,W,DATA_POINTER 
*      X RETURNS POINTER TO RAM IMAGE TO REFRESH

TRSTRT PSHS   Y,D,U    
       BSR    GETBLK   GET A DATA BLOCK
       BCS    ONBYE    NO BLOCKS...BYE
ONDRV0 STY    TRPTR    PUT THE MAIN MAN ON THE LIST
       STX    PICPTR,Y SAVE PICTURE POINTER
       LDD    ,X       GET W,H
       STD    OW,Y     STORE THEM
       MUL             FORM BYTE COUNT
       STB    BYTCNT,Y AND SAVE
       LEAU   DATA,Y   FIND WHERE DATA WILL START
       STU    OPT,Y
       BSR    CLRM     CLEAR MEMORY IN ALLOCATED BLOCK    
       LDU    #TRNTAB  APPEAR TABLE TO BE USED
       STU    FRPTR,Y  FIRST FRAME
       JSR    RAND     GET RANDOM GLITTER
       ANDA   #$7
       ASLA
       LDU    #GLITAB
       LDD    A,U
       STD    GLTPTR,Y
       LEAX   OW,Y     THIS LOOKS LIKE AN OBJECT...RETURN IT 
ONBYE  PULS   Y,U,D,PC PULL IT ALL AND RETURN

CLRM   PSHS   D,Y
       LDY    OPT,Y
CLRM1  CLR    ,Y+            CLEAR A BYTE
       DECB
       BNE    CLRM1
       PULS   PC,D,Y

GETBLK PSHS   X  
       LDY    FREEP
       BEQ    GETERR
       LDX    ,Y       GET NEXT
       STX    FREEP    MAKE LIST
       LDX    TRPTR    GET CURRENT LIST
       STX    ,Y       POINT AT IT
       CLC
       PULS   X,PC  
GETERR SEC
       PULS    X,PC

*      Y BLOCK FOR OBJECT TO UPDATE
*      X RETURNS PICPTR FOR OBJECT
*      U RETURNS TRANS TABLE POINTER

TMAKE  LDX    FRPTR,Y  GET FRAME POINTER 
       PSHS   X        AND SAVE IT SO WE MAY STILL USE U
       LEAX   1,X      KICK FRAM COUNTER
       STX    FRPTR,Y
       LDA    SEED
       ANDA   #$7
       LDX    GLTPTR,Y GET THE GLITTER POINTER
       LEAX   A,X      ADD THE RANDOM GUY
       STX    GTEMP    AND PUT IT IN RAM       
       LDX    PICPTR,Y LOAD THE ACTUAL PICTURE POINTER
       LDB    BYTCNT,Y LOAD THE BYTE COUNT IN THE OBJECT
       PULS   U,PC     PUT THE FRAME POINTER IN U AND RETURN

*      TRNSPT - X POINTS AT DATA FOR OBJECT
*               Y POINTS AT DATA AREA
*               B CONTAINS BYTES IN OBJECT
*               U EQUALS CURRENT TRANSPORT FRAME

TRNSPT PSHS   D,U,Y,X    
       LDX    2,X            POINT AT DATA TEMPLATE
       LDY    OPT,Y          POINT AT PHONEY OBJECT
       LDA    ,U             GET THE FRAME NUMBER
       BPL    XON            TRANS EM ON THIS FRAME
       ANDA   #$7F           TAKE OFF BIT AWAY.
       JSR    GETGRP         GET THE GROUP POINTER IN U.
POFF0  LDD    ,U++           GET THE NEXT POINT
       CMPA   BYTES          CHECK BYTE NUMBER
       BHS    TRNDUN         TOO LARGE TO BE IN THIS PICTURE
       COMB                  TAKE AWAY TO TURN OFF.
       ANDB   A,Y
       STB    A,Y            
       BRA    POFF0
XON    BITA   #$40         GLITTER FRAME??
       BNE    XONG         THEN GET OUT OF HERE!
       BSR    GETGRP
PON0   LDD    ,U++
       CMPA   BYTES 
       BHS    TRNDUN
       ANDB   A,X            GET WHAT THE BYTE LOOKS LIKE
       ORAB   A,Y            OR IT IN (IT WAS OFF)
       STB    A,Y
       BRA    PON0
TRNDUN PULS   D,PC,U,X,Y    

XONG   ANDA   #$3F           TAKE AWAY GLITTER BIT 
       BSR    GETGRP
PON0G  LDD    ,U++
       CMPA   BYTES 
       BHS    TRNDUN
       ANDB   A,X            GET WHAT THE BYTE LOOKS LIKE
       BEQ    PON0G          NOTHING THERE...DONE WITH THE BYTE
       LDB    [GTEMP]        GET THE CURRENT GLITTER BYTE
       INC    GTEMP+1        KICK GLITTER POINTER
****************************************************************************
*      BNE    PON1G          DIDN'T WRAP                  THESE 2 INSTRUCTIO
*      INC    GTEMP                           ARE SUPERFLOUS FOR MOST ADDRS
****************************************************************************
PON1G  ANDB   -1,U
       ORAB   A,Y            OR IT IN (IT WAS OFF)
       STB    A,Y
       BRA    PON0G

GETGRP LDU    #GRPLST        GROUP TABLE
       DECA                  MAKE 1-12, 0-11
       ASLA      
       LDU    A,U
       STB    BYTES          SAVE BYTE COUNT
       RTS

TRKIL  PSHS   X
       LDX    ,Y       GET THE NEXT GUY
       STX    [LAST]   MAKE THE LAST GUY POINT TO OUR NEXT
       LDX    FREEP    
       STX    ,Y       APPEND FREE LIST TO US
       STY    FREEP    AND MAKE US THE LIST
       LDY    LAST     MAKE POINT AT LAST FOR NEXT ITERATION
       PULS   X,PC

*********  TRANSPORTER UPDATE

UPDATE LDY    #TRPTR   WORK THROUGH LIST
       BRA    TRNUP0
TRNUP3 JSR    TMAKE    FOR ALL OF THE PARAMETERS
       CMPU   #TABBOT  TOO FAR ON ONE???
       BEQ    TRNUP1
       JSR    TRNSPT   DO THE POINT  
TRNUP0 STY    LAST
       LDY    ,Y       WALK TO NEXT
       BNE    TRNUP3   ITS GOOD
       RTS    DONE

TRNUP1 JSR    TRKIL  
       BRA    TRNUP0

TRNTAB FCB  $43
       FCB    $83
       FCB  $46
       FCB    $86
       FCB  $44
       FCB    $84
       FCB  $42
       FCB    $82
       FCB  $41
       FCB    $81
       FCB  $47
       FCB    $87
       FCB  $44
       FCB  $46
       FCB    $84
       FCB  $42
       FCB    $86
       FCB  $43
       FCB    $82
       FCB  $41
       FCB    $83
       FCB  $45
       FCB    $81
       FCB  $48
       FCB    $85
       FCB  $47
       FCB    $88
       FCB  $44
       FCB    $87
       FCB  $45
       FCB  $42
       FCB    $84
       FCB  $46
       FCB    $85
       FCB  $47
       FCB    $82
       FCB  $48
       FCB    $86
       FCB  $43
       FCB    $87
       FCB  $41
       FCB    $88
       FCB  $42 
       FCB    $83
       FCB  $44
       FCB    $81
       FCB  $45
       FCB  $47
       FCB    $82
       FCB  $46
       FCB    $84
COL2   FCB    $85
       FCB  $48
       FCB    $87
       FCB  $43
       FCB    $86
       FCB  $44
       FCB  $42
       FCB    $88
       FCB  $45
       FCB    $83
       FCB  $41
       FCB    $84
       FCB  $43
       FCB    $82
       FCB  $47
       FCB  $46
       FCB    $85
       FCB  $48
       FCB    $81
       FCB  $42
       FCB    $83
       FCB  $44
       FCB    $87
       FCB  $41
       FCB    $86
       FCB  $45
       FCB    $88
       FCB  $43
       FCB  $46
       FCB    $84
       FCB  $48
       FCB    $81
       FCB  $47
       FCB    $85
       FCB  $44
       FCB    $82
       FCB  $41
       FCB  $45
       FCB    $83
       FCB  $42
       FCB    $86
       FCB    $88
       FCB  $43
       FCB    $87
       FCB  $46 
       FCB    $84
       FCB  $48
       FCB    $85 
       FCB    7
       FCB    $81
       FCB    4
       FCB    5
       FCB    $82
       FCB    1
       FCB    $83
       FCB    2
       FCB    $86
       FCB    3
       FCB    $88
       FCB    6
       FCB    $87
       FCB    8
       FCB    $84
       FCB    7
       FCB    $85
       FCB    4
       FCB    $81
       FCB    5
       FCB    1
       FCB    $82
       FCB    $83
       FCB    2
       FCB    3
       FCB    $88
       FCB    8
       FCB    $84
       FCB    $87
       FCB    7
       FCB    4
TABBOT EQU    *
       FCB    0       FIX FLAW IN ALGORITHM

GRPLST FDB    GROUP1
       FDB    GROUP2
       FDB    GROUP3
       FDB    GROUP4
       FDB    GROUP5
       FDB    GROUP6
       FDB    GROUP7
       FDB    GROUP8

GRPTAB EQU    *
GROUP1 FCB    00,$F0,06,$0F,09,$0F,12,$F0,19,$F0,23,$0F,24,$0F,29,$F0
       FCB    34,$F0,37,$0F,43,$F0,46,$0F,48,$F0,50,$0F,60,$F0,62,$0F
       FCB    67,$0F,69,$F0,72,$F0,76,$0F,80,$0F,81,$F0,90,$F0,93,$0F  
       FCB    96,$0F,101,$F0,106,$0F,110,$0F,115,$F0
GROUP2 FCB    01,$0F,03,$F0,08,$F0,15,$0F,18,$F0,20,$0F,27,$0F,30,$F0
       FCB    32,$F0,34,$0F,45,$F0,47,$0F,51,$0F,53,$F0,57,$0F,61,$F0
       FCB    67,$F0,71,$0F,74,$0F,78,$F0,83,$F0,85,$0F,91,$0F,91,$F0  
       FCB    99,$0F,104,$F0,109,$F0,113,$0F,115,$0F
GROUP3 FCB    04,$0F,07,$F0,09,$F0,10,$0F,20,$F0,21,$0F,28,$0F,31,$F0
       FCB    33,$F0,35,$0F,41,$0F,46,$F0,52,$0F,54,$F0,58,$F0,61,$0F
       FCB    66,$F0,68,$0F,73,$0F,77,$F0,85,$F0,87,$0F,89,$0F,92,$F0  
       FCB    97,$F0,104,$0F,106,$F0,109,$0F,114,$0F
GROUP4 FCB    00,$0F,02,$F0,10,$F0,12,$0F,16,$F0,22,$0F,26,$F0,30,$0F 
       FCB    33,$0F,35,$F0,44,$0F,47,$F0,51,$F0,54,$0F,56,$F0,59,$0F
       FCB    66,$0F,70,$F0,73,$F0,77,$0F,82,$F0,84,$0F,90,$0F,93,$F0
       FCB    97,$0F,100,$F0,105,$0F,108,$F0,112,$F0
GROUP5 FCB    02,$0F,06,$F0,08,$0F,14,$F0,19,$0F,22,$F0,25,$F0,29,$0F
       FCB    36,$0F,39,$F0,41,$F0,43,$0F,48,$0F,49,$F0,60,$0F,62,$F0
       FCB    64,$0F,65,$F0,75,$F0,78,$0F,82,$0F,87,$F0,89,$F0,94,$0F
       FCB    98,$F0,101,$0F,103,$F0,107,$0F,111,$F0
GROUP6 FCB    01,$F0,07,$0F,11,$0F,13,$F0,17,$0F,21,$F0,26,$0F,28,$F0
       FCB    32,$0F,37,$F0,42,$F0,45,$0F,49,$0F,50,$F0,56,$0F,63,$F0
       FCB    68,$F0,70,$0F,75,$0F,79,$F0,84,$F0,86,$0F,88,$0F,95,$F0
       FCB    96,$F0,102,$F0,108,$0F,111,$0F,113,$F0
GROUP7 FCB    03,$0F,04,$F0,13,$0F,15,$F0,17,$F0,18,$0F,24,$F0,31,$0F
       FCB    38,$F0,39,$0F,40,$0F,44,$F0,52,$F0,55,$0F,59,$F0,63,$0F
       FCB    64,$F0,69,$0F,72,$0F,74,$F0,83,$0F,86,$F0,88,$F0,92,$0F
       FCB    98,$0F,100,$0F,107,$F0,112,$0F,114,$F0
GROUP8 FCB    05,$F0,05,$0F,11,$F0,14,$0F,16,$0F,23,$F0,25,$0F,27,$F0
       FCB    36,$F0,38,$0F,40,$F0,42,$0F,53,$0F,55,$F0,57,$F0,58,$0F
       FCB    65,$0F,71,$F0,76,$F0,79,$0F,80,$F0,81,$0F,94,$F0,95,$0F
       FCB    99,$F0,102,$0F,103,$0F,105,$F0,110,$F0
* 
*START TRANSPORTING
*
TRNSTV JSR TRINIT
 LDX RPTR
 BEQ TRN1X
 MAKP TRSPRC
 LDX RPTR
 LDD #0
 LDY #0
 PSHS D
 PSHS D
TRN00 LDD OPICT,X
 STD OLDPIC,X
 LEAY 1,Y
 CMPY #15
 BHI TRN01
 CMPD ,S SAME GUY?
 BEQ TRN0 YES
TRN01 LDY #0
 STD ,S SAVE NEW PICT
 PSHS X
 LDX OPICT,X
 JSR TRSTRT
 STX 4,S SAVE IMAGE
 PULS X SAVE NEW PICT
TRN0 LDD 2,S
 STD OPICT,X
 LDX ,X
 BNE TRN00
 LEAS 4,S
TRNLP JSR ROBUP OUTPUT ROBOTS
 JSR UPDATE  UPDATE IMAGES
 LDX TRPTR
 BEQ TRN1X
 NAP 1,TRNLP
TRN1X LDX RPTR
 BEQ TRN2X
TRN1XL LDD OLDPIC,X RESTORE IMAGES
 STD OPICT,X
 LDX ,X
 BNE TRN1XL
TRN2X JSR HXINIT RECOVER DATA BASE FOR HORIZ EXPLODES
 JMP SUCIDE
*
*TRANSPORTER SOUND PROCESS
*
TRSPRC LDD #TR1SND
 JSR SNDLD CLEAR OUT THE SYSTEM
 LDA #$48
 STA PD,U
TRSPRL LDB #$12
 JSR SDOUT
 DEC PD,U
 BEQ TRSPR1
 NAP 1,TRSPRL
TRSPR1 LDA #$24
 STA PD,U
TRSP1L LDB #$12
 JSR SDOUT
 DEC PD,U
 LBEQ SUCIDE
 NAP 2,TRSP1L
*
*UPDATE YOUR ROBOTS
*RAM-RAM
ROBUP LDX RPTR
 BEQ ROBUPX
ROBUPL LDY OPICT,X
 LDD ,Y
 EORA #4
 EORB #4
 SEI
 STD DMASIZ
 LDU 2,Y
 STU DMAORG
 LDD OBJX,X
 STD DMADES
 LDA #$6
 STA DMACTL
 CLI
 LDX ,X
 BNE ROBUPL
ROBUPX RTS
 END

