       OPT    NOL                                   
       LIB    RRFRED

       IFNC   &A,NOL
       OPT    LIS
       ENDIF

WSTART EQU    $183A

RAPRAM EQU    OLIST+(18*OSIZE)    LEAVE 18 OBJECTS FOR APPEARS

       ORG    0                   OFFSETS
O1PT   RMB    4                   TOP OBJECT POINTER
O2PT   RMB    4                   BOTTOM OBJECT POINTER
APADD  RMB    2                   ADRESS TO EXPAND
DATA1  RMB    14*11               DATA FOR FIRST OBJECT
DATA2  RMB    13*11               DATA FOR SECOND OBJECT
RAPSIZ EQU    *

       ORG    LOGORG 

       JMP    LOGGER
       FDB    COLC

LOGGER JSR    GNCIDE
       CLR    ATFLAG         NO COIN SEEN YET
       MAKP   CCHECK         MONITOR CREDITS
       LDX    #LOGG1         LOOK....UP IN THE SKY....
       CLRA                       ITS A BIRD......ITS A PLANE....NO,
       JSR    MSPROC                      ITS  S U P E R P R O C E S S !
       JMP    SUCIDE         AND DIE

CCHECK LDA    CREDIT
       STA    PD,U
CCHEK1 LDA    CREDIT
       CMPA   PD,U
       BEQ    CCHEK2                            
       INC    ATFLAG         INDICATE IN COIN JUST INSERTED MODE
       JMP    FAMPAG         GO TO FAMILY PAGE FOR COIN
CCHEK2 NAP    8,CCHEK1

RMAK   FDB    LR,LO,LB,LO,LT,LR,LO,LN,LC,0   ROBOTRON POINTERS

LOGG1  JSR    SCRCLR
       JSR    TABORG

       CLR    ATFLAG         NORMAL PATH INTO FAMILY PAGE...NO CREDS
       JMP    FAMPAG         AND DO THAT PAGE

RUNIT  CLR    ATFLAG
       JSR    GNCIDE
       JSR    OINIT
       JSR    SCRCLR
       MAKP   CCHECK         MONITOR CREDITS
       CLR    PCRAM+$C       ROBOTRON COLORS
       CLR    PCRAM+$E

       NAP    3,RUNNME       MAKE SURE THE PCRAM STUFF TAKES EFFECT

RUNNME LDX    #WSTART        PLACE FOR UPPER LEFT OF LOGO
       STX    PD+2,U

       LDX    #RAPRAM        RAM DATA AT
       LDY    #RMAK

RUNIT3 STY    PD,U           LETTER POINTER AT PD
       STX    PD+4,U         RAM POINTER AT PD+4   

       NAP    1,RUNIT2       SLEEPY TIME

RUNIT2 LDY    PD+4,U         GET DATA BLOCK
       LDD    #$B0E          11 X 14 FOR TOP
       STD    O1PT,Y
       LDB    #13            11 X 13 FOR BOTTOM
       STD    O2PT,Y
       LEAX   DATA1,Y        FIRST DATA POINTER
       STX    O1PT+2,Y       DATA POINTER
       LEAX   DATA2,Y
       STX    O2PT+2,Y
       LDX    PD+2,U         GET NEXT LETTER SCREEN POINTER
       STX    APADD,Y        AND SAVE FOR APPEAR

       LDY    PD,U           GET LETTER POINTER
       LDU    ,Y             GET NEXT LETTER
       BEQ    WDONE          DONE WRITING...TIME FOR BLASTOFF!

       CLRB                  NO FLAVOR PLEASE
       LDA    #$CE           COLORS
       PSHS   X              SAVE POINTER WE'RE USING
       JSR    LETOUT         PUT THE LETTER ON THE SCREEN
       LDU    CRPROC         RESTORE PROCESS BASE
       LEAX   $200,X         SPACER
       STX    PD+2,U         NEXT ONE

       PULS   X              PLACE WE WROTE IT

       LDY    PD+4,U         GET PLACE TO WRITE IT
       LEAY   DATA1,Y        POINT WHERE DATA GOES
       LDD    #$B1B          REAL FOR 11 BY 27
       JSR    RWMOVE         DO IT JACK!

       LDX    PD+4,U         GET RAM POINTER
       LEAX   RAPSIZ,X

       LDY    PD,U
       LEAY   2,Y
       BRA    RUNIT3

GETBAL PSHS   A
       LDA    FREEPL+1       SEE IF BALLS ALLOWED
       ANDA   #$0F           GET THE BYTE
       PULS   A,PC

*      TIME TO STAGE THE APPEARS

WDONE  BSR    GETBAL         SEE IF FANCY
       BEQ    NOBA11

       JSR    SCRCLR         CLEAR THE SCREEN

NOBA11 LDA    #$7
       STA    PCRAM+$C       RED

       LDA    #$3F
       STA    PCRAM+$E

       BSR    GETBAL
       LBEQ   LOGG2          NO APPEARS NECESSARY

       LDA    STATUS
       ANDA   #$FB
       STA    STATUS

       LDU    CRPROC

       LDY    #RAPRAM        POINT AT FIRST
       STY    PD,U
       LDA    #9             9 CHARS
       STA    PD+2,U
       BRA    WDONE1         NO EXTRA NAPPING!

WDONE0 NAP    8,WDONE1

WDONE1 LDY    PD,U           GET THE LETTER TO DO
       LDD    APADD,Y        GET THE ADDRESS OF TOP
       ADDB   #13
       STB    CENTMP+1
       SUBB   #13
       LEAX   O1PT,Y         GET THE OBJECT

       JSR    AMAP           DO THE APPEAR

       ADDB   #14            PULL TO THE TOP               
       STB    CENTMP+1

       LEAX   O2PT,Y         POINT AT SECOND OBJECT

       JSR    AMAP

       LEAY   RAPSIZ,Y       FORM NEXT
       STY    PD,U
       DEC    PD+2,U
       BNE    WDONE0
       NAP    $20,LOGG2      LET THEM FINISH, THEN DO OTHER STUFF

COLE   LDA    #$3F           COLOR F IS YELLOW ALSO
       STA    PCRAM+$F

       LDA    #$7
       STA    PCRAM+$D

       MAKP   LFV      MAKE THE LASER COLORER (COLOR A)

       LDY    #L2084   POINTER TO THE STUFF
       CLRB
       LDX    #$395C   SCREEN PTR
       LDA    #$FD     YELLOW PRIMARY,  RED TRIM
COLE2  LDU    [,Y++]
       BEQ    COLLLE   DONE!
       JSR    LETOUT   PUT IT OUT
       BRA    COLE2

COLLLE LDU    ,Y++
       BEQ    COLE1
       JSR    LETOUT
       BRA    COLLLE

COLE1  MAKP   COLC

       LDX    #CEPTR
       LDY    #PCRAM+$E
       LDA    #1

COLENT LDU    CRPROC
       STX    PD+2,U
       STY    PD+4,U
       STA    PD+6,U

COLE4  LDX    PD+2,U
       STX    PD,U

COLE3  LDX    PD,U
       LDA    ,X+
       BEQ    COLE4

       STA    [PD+4,U]
       STX    PD,U
       LDA    PD+6,U
       LDX    #COLE3
       JMP    SLEEP

COLC   LDA    LSEED
       BPL    COLC1

       LDA    #7       RED ME
       STA    PCRAM+$C
       JSR    RAND
       ANDA   #$7      3 BITS OF SLEEP
       INCA
       LDX    #COLC1
       JMP    SLEEP

COLC1  LDA    SEED     HOW RANDOM DO I FEEL?
       ANDA   #$3
       BEQ    COLC11   NOT VERY

       CLR    PCRAM+$C
       NAP    3,COLC11

COLC11 LDX    #COLTAB
       JSR    RAND
       ANDA   #$0F
       LDA    A,X
       STA    PCRAM+$C
       NAP    7,COLC2

COLC2  LDA    LSEED
       ANDA   #$3
       BEQ    COLC  

       CLR    PCRAM+$C
       NAP    4,COLC  

CEPTR  FCB    $3F,$3F,$3F,$37,$2F,$27,$1F,$17,$F,7,7,7
       FCB    $F,$17,$1F,$27,$2F,$37,$3F,$3F,$3F,0

COLTAB FCB    $FF,$C0,$C7,$1F,$7 
       FCB    $07,$C0,$C7
       FCB    $FF,$C0,$C7,$16,$7
       FCB    $FF,$C0,$C7

*OLTAB FCB $38,$FF,$3A,$3B,$3C
*FCB $3D,$3E,$3F,$37,$2F,$27,$FF,$FF
*FCB $FF,$47,$87,$87,$C7,$C7
*FCB $C6,$C5,$CC,$CB,$CA,$DA,$E8,$F8
*FCB $FA,$FB,$FD,$FF,$BF,$3F,$3E
*FCB $3C,0

HISTRY EQU    *

       IFC    &A,NOL

       OPT    LIS
       LIB    RRSCRIPT
       OPT    NOL

       ELSE
       LIB    RRSCRIPT

       ENDIF

LOGG2  JSR    OINIT          REDO THE OBJECT STRUCTURE

       LDA    STATUS         CAN EXPLOSIONS NOW
       ORAA   #4
       STA    STATUS

       MAKP   COLE           COLOR E PROCESS NEEDED,(ALSO PUT OUT 2084)
       JSR    CSET
       MAKP   MESSER         HANDLE OPERATOR MESSAGE THEN TAKE OVER

       JSR    GETBAL         SEE IF INHIBITED
       BNE    DOBALL

       MAKP   COL1           START THE WHITE PROCESS

       NAP    $FF,NOBA22
NOBA22 NAP    $FF,HISTRY

DOBALL JSR    WILSUB         SETUP FIRST WILL BALL
       LDA    #28            WE'LL DO 28 AND THEN DIE
       LDU    CRPROC
       STA    PD,U           COUNTER                          
RUN1   LDX    LASTX
       LDA    BALCOL
       JSR    BALOUT         OUTPUT A BALL AT INITIAL PLACE
       JSR    FORMX
       NAP    4,RUN2
RUN2   DEC    PD,U
       BNE    RUN1
       JMP    WILLME         NOW SPIN EM'                    

MESSER JSR    ATMCK          SEE IF OPERATOR MESSAGE IS INTACT
       BEQ    MESS01         YEP
       LDX    DOPMES         GET THE DEFAULT MESSAGE
       LDY    #MESOUT        52 BYTES WORTH
MESS02 LDD    ,X++ 
       STD    ,Y++
       CMPY   #MESOUT+52
       BLO    MESS02
       BRA    MESS03
MESS01 LDX    #OPMESS
       LDY    #MESOUT
MESS04 JSR    RCMOSD         GET 2 BYTES
       STD    ,Y++
       CMPY   #MESOUT+52
       BLO    MESS04
MESS03 LDA    MESOUT+50      GET X OF FIRST LINE
       LDB    #$86           HEIGHT
       LDY    #MESOUT        FIRST CHAR
       BSR    LINOUT         PUT OUT A LINE
       LDA    MESOUT+51
       LDB    #$96
       LDY    #MESOUT+25
       BSR    LINOUT         OUT LINE 2
       LDB    CREDIT 
       LDA    #CPM1          COPYRIGHT ETC
       JSR    WRD5V          PUT IT OUT
       JMP    SUCIDE         WE'RE DONE..BYE   

LINOUT PULS   X              GET RETURN ADDR
       LDU    CRPROC
       STX    PD,U           AND SAVE
       TFR    D,X            POINTER IN X PLEASE
       LDA    #25
       STA    PD+2,U         NUMBER OF LETTERS TO OUTPUT
       LDA    #$66           COLOR OF OPERATOR MESSAGE
       STA    TEXCOL         PUT IT AWAY
LINO1  LDA    ,Y+            GET A LETTER
       JSR    PR57V          PRINT IT
       STX    PD+3,U
       STY    PD+5,U
       NAP    2,LINO2
LINO2  LDX    PD+3,U
       LDY    PD+5,U
       DEC    PD+2,U
       BNE    LINO1
       JMP    [PD,U]

       FCC    ' ROBOTRON: 2084 '
       FCC    ' COPYRIGHT 1982 WILLIAMS ELECTRONICS INC. '
       FCC    ' ALL RIGHTS RESERVED '

WILSUB JSR    FINIT          FIRST TWO LOCATIONS FOR INITIAL MOVE
       LDY    #ARAM1
       LDA    #$10           MOVE OUTER COLOR TO ARAM1
       BSR    WMOVE          DO IT
       JMP    FINIT          INIT THE DATABASE AND RETURN

WILLME BSR    WILSUB         DO SETUP STUFF

       MAKP   COL1           MAKE THE COLOR PROC

       LDU    CRPROC
       LDX    #$2C0
       STX    PD+6,U         SAVE

       LDA    #6
       STA    PD+4,U

       LDX    LASTX
       LDA    BALCOL
       BRA    WILL55

WILL1  LDA    #6 
       STA    PD+4,U

WILL44 JSR    FORMX
       JSR    XUPD           UPDATE POINTER TO MAKE THIS STUPID BEAM RUN 

WILL55 LDY    ERPTR          GET HEIGHT OF GUY WE'RE GONNA ERASE
       LDB    VERTCT         IT MUST BE 10 LINES BEYOND US
       STB    XTEMP          KEEP IT HONEST
       CMPB   1,Y            FIND WHICH SIDE ITS ON
       BLO    WILL33         ITS ABOVE US!

       SUBB   1,Y            ITS BELOW US,  SUBTRACT US
       CMPB   #20
       BLS    WILL55         LOOP PLEASE
       LDB    XTEMP          GET THE COUNTER
       CMPB   #$EC           ARE WE REAL LOW??
       BHS    WILL55         THEN WE NEED TO LET IRQ IN TO CHANGE PCRAM
       BRA    WILL66

WILL33 LDB    1,Y            GET THE Y VALUE
       CMPB   #30            IS THE Y WE'RE TRYING TO PLOT HIGH??
       BLS    WILL55         THEN WAIT FOR BELOW US

       SUBB   VERTCT         ELSE FIND OUT HOW HIGH?
       CMPB   #30            LETS SAY WE NEED 30 LINES TO DO IT
       BLS    WILL55

WILL66 JSR    BALOFF         TURN OFF THE BALL WE'RE ABOUT TO DO

       JSR    BALOUT         PUT OUT THE BALL AT THAT POINT

       LDU    CRPROC
       DEC    PD+4,U
       BNE    WILL44

       LDX    PD+6,U         TIME LEFT
       LEAX   -1,X
       LBEQ   HISTRY         START OVER
       STX    PD+6,U

       NAP    1,WILL1

WMOVE  PSHS   Y,A            SAVE THE EVENTUAL TARGET
       LDX    LASTX          GET THE FIRST X FOR SETUP 
       PULS   A
       PSHS   X              SAVE THE X
       CLRB
       LDU    #BILL
       JSR    LETOUT         NOW ITS ALLEGEDLY ON SCREEN

       PULS   X,Y            GET BACK X (SCREEN)  Y (RAM)
       LDD    #$E1B          14 X 28 IN REAL       
       JMP    RWMOVE         DO THE XFER FROM PRIME REAL ESTATE AND RET

X1     EQU    $5 
X2     EQU    $85
Y1     EQU    $F 
Y2     EQU    $CF

FINIT  PSHS   D,X
       LDA    #X1
       LDB    #Y1
       STD    LASTX
       CLR    SIDE           SIDE 0
       CLR    SLIDER
       LDA    #$77
       STA    BALCOL
       LDX    #ERLIST
       STX    ERPTR
       LDD    #$13AF         ERASE SCREEN CENTER ON FIRST PASS
FINIT1 STD    ,X++
       CMPX   #EREND         WHOLE LIST??
       BLO    FINIT1
       PULS   D,PC,X

FORMX  PSHS   B  
       LDA    SIDE           SIDE 0,1,2 OR 3??
       ANDA   #$3            LOW IS ALL THATS IMPORTANT
       DECA
       BMI    SIDE0
       DECA      
       BMI    SIDE1
       DECA
       BMI    SIDE2
SIDE3  LDD    LASTX
       SUBB   #32            MOVE UP 1
       BCS    BADSID
       CMPB   #Y1            LOW Y??
       BHI    SIDEX          DONE IF HIGH
BADSID LDA    SLIDER         FIND SLIDE AMOUNT
       ADDA   #2             2 UNITS PER
       CMPA   #16
       BLO    SLID1
       CLR    SLIDER
       LDA    #X1+16
       LDB    #Y1
       BRA    SIDCH
SLID1  STA    SLIDER
       LDB    #Y1
       LDA    #X1
       ADDA   SLIDER         SHIFT OVER BY SLIDER
       BRA    SIDCH          NEW SIDE!
SIDE2  LDD    LASTX
       SUBA   #16
       BCS    SIDE25
       CMPA   #X1
       BHI    SIDEX
SIDE25 LDA    #X1
       LDB    SLIDER
       NEGB
       ASLB
       ADDB   #Y2            HIGH Y
       BRA    SIDCH
SIDE1  LDD    LASTX
       ADDB   #32
       CMPB   #Y2
       BLO    SIDEX
       LDB    #Y2
       LDA    #X2
       SUBA   SLIDER
       BRA    SIDCH
SIDE0  LDD    LASTX
       ADDA   #16
       CMPA   #X2
       BLO    SIDEX
       LDA    #X2
       LDB    SLIDER
       ASLB
       ADDB   #Y1
SIDCH  INC    SIDE           CHANGE TO A NEW SIDE
SIDEX  STD    LASTX          SAVE THE X
       TFR    D,X            RETURN IN X
       LDA    BALCOL         RETURN OLD COLOR
       SUBA   #$11           NEXT COLOR
       BCS    SIDE8                         
       BNE    SIDE7
SIDE8  LDA    #$77
SIDE7  STA    BALCOL
       PULS   B,PC

XUPD   PSHS   X
       LDX    ERPTR          GET CURRENT ERASE POSITION
       LEAX   2,X            POINT TO NEXT
       CMPX   #EREND         DONE??
       BLO    BALOF1         IN RANGE
       LDX    #ERLIST
BALOF1 STX    ERPTR          PLACE
       PULS   X,PC

BALOFF PSHS   X,D,CC
       LDX    [ERPTR]        GET ENTRY 32 AGO
       SEI
       STX    DEST           PLACE TO ERASE
       LDD    #$A1F          SIZE
       STD    HORIZ
       CLR    CONST
       LDD    #ARAM1         SOURCE IS WILLY BALL
       STD    ORIG                   
       LDA    #$1E
       STA    CONTRL
       PULS   CC,X,D,PC

BALOUT PSHS   CC,A
       STX    [ERPTR]        SAVE THIS BALLS POS. FOR ERASE MUCH LATER
       SEI
       STA    CONST          COLOR
       STX    DEST           PLACE TO GO
       LDD    #$A1F
       STD    HORIZ          SIZE ME
       LDD    #ARAM1         HALO
       STD    ORIG
       LDA    #$1E           SLOW SER TO BLK CONST   ZWS
       STA    CONTRL

*      LDD    #ARAM2         W
*      STA    ORIG

*      CLR    CONST          CLEAR THIS FRAME

*      LDA    #$1E           ZWS ETC
*      STA    CONTRL
       PULS   CC,A,PC

CSET   PSHS   X,Y,A
       LDX    #LCTAB   USE LOGO COLOR TABLE FOR 1-7   
       LDY    #PCRAM+1
COLLL  LDA    ,X+ 
       STA    ,Y+ 
       CMPY   #PCRAM+8 
       BLO    COLLL
       PULS   X,Y,A,PC

COL1   LDX    #PCRAM+1
       STX    PD+2,U
COLL   BSR    CSET          SET THE COLORS WE'LL USE
COLL5  LDX    PD+2,U   GET WHITE POINTER
       LEAX    1,X     KICK IT UP
       CMPX   #PCRAM+8
       BLO    COLL8
       LDX    #PCRAM+1
COLL8  STX    PD+2,U
       LDA    #$FF
       STA    ,X
       NAP    3,COLL

LCTAB  FCB    $7             RED
       FCB    $C0            BLUE
       FCB    $17            ORANGE-BROWN
       FCB    $30            GREEN 
       FCB    $C7            VIOLET
       FCB    $1F            ORANGE
       FCB    $3F            YELLOW

       LIB    RRLOGD

       END
