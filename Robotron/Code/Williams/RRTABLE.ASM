       OPT    NOL    
       LIB    RRFRED
       OPT    LIS

TCOL1  EQU    GETLRM
TCOL2  EQU    GETLRM+1

       ORG    TABORG         HIGH SCORE TABLE ORIGIN

*      PD - FONT SIZE                 
*      PD+1 NUMBER PER COLUMN
*      PD+2 NUMBER OF COLUMNS LEFT
*      PD+3 HEIGHT SPACING BETWEEN ROWS
*      PD+4 WIDTH SPACING BETWEEN COLUMNS
*      PD+5 NUMBER LEFT IN THIS COLUMN
*      PD+6 INDEX OF CURRENT ENTRY
*      PD+7 X ADDRESS DURING SLEEP
*      PD+9 CMOS PTR DURING SLEEP
*      PD+11 RETURN ADDRESS SAVE
*      PD+18 TOP LEVEL RETURN ADDRESS
*      PD+15 X RETURN
*      PD+16 Y RETURN
*      PD+17 NUMBER UNTIL SLEEP

TABLE  PULS   D              GET RETURN
       LDU    CRPROC
       STD    PD+18,U        SAVE "RETURN ADDRESS"
       MAKP   LOOPP          START THE COLORING PROCESS
       JSR    FRAMER         PUT OUT THE FRAME
       LDA    #$99
       STA    TCOL1  
       LDA    #$CC
       STA    TCOL2
       LDY    #TODAYS        POINT AT FIRST TODAYS SCORE
       LDU    CRPROC         POINT RIGHT
       LDA    #7             USE 7 FONT
       STA    PD,U
       LDD    #$502          5 PER COLUMN, 2 COLUMNS
       STD    PD+1,U
       LDD    #$934
       STD    PD+3,U         7 HEIGHT, 50 WIDTH
       LDA    #1             START WITH INDEX OF 1
       STA    PD+6,U         INDEX POINTER
       LDX    #$1A35         UPPER LEFT OF PLACE TO DO IT
       JSR    PRJNK          PRINT THE STUFF
       LDA    #$AA
       STA    TCOL1          NEW COLOR
       LDA    #$DD
       STA    TCOL2
       LDY    #GODSCR+34
       JSR    CLSET          SET THE COLOR
       LDX    #GODSCR+38     POINT AT LAST CHAR
TABLE1 JSR    RCMOSA         GET A CHAR
       LEAX   -4,X
       CMPA   #$3A           SPACE??
       BNE    TABLE2         NOPE  
       CMPX   #GODSCR        ARE WE POINTING AT FIRST??
       BHS    TABLE1
TABLE2 LEAX   2,X            POINT AT FIRST NON BLANK
       PSHS   X              SAVE ON STACK
       LDX    #$157A         PLACE FOR GOD
       LDA    #$31
       JSR    PR57V
       LDA    #$5C           RIGHT PAREN
       JSR    PR57V
       LEAX   $300,X         SPACE OVER
       LDY    #GODSCR
TABLE3 EXG    X,Y            SWAP CMOS, CURSOR
       JSR    RCMOSA         GET A LETTER
       EXG    X,Y            RESTORE CURSOR
       JSR    PR57V          PRINT IT
       CMPY   ,S             SEE IF WE'RE DOWN TO SPACES ONLY
       BLS    TABLE3
       LEAX   $200,X         SPACE OVER
       LDY    #GODSCR+40     POINT AT SCORE
       JSR    PRSCOR         PRINT OUT A SCORE (FONT IS 7 IN PD,U)

       STX    XTEMP          SAVE X
       LDX    #GA2
       JSR    RCMOSA
       CMPA   #3
       BEQ    NOINTS
       LDX    XTEMP

       LEAX   $500,X         MOVE FOR INITIALS
       LDA    #$5B           LEFT PAREN
       JSR    PR57V
       LDY    #GODINT
       LDB    #3
TABL33 EXG    X,Y
       JSR    RCMOSA
       EXG    X,Y
       JSR    PR57V
       DECB
       BNE    TABL33

       LDA    #$5C           RIGHT PAREN
       JSR    PR57V

NOINTS LDU    CRPROC
       LDA    #5       FONT
       STA    PD,U
       LDD    #$C03    9 ROWS BY FOUR COLUMNS
       STD    PD+1,U
       LDD    #$728    SPACING
       STD    PD+3,U
       LDA    #2       FIRST INDEX
       STA    PD+6,U
       LDX    #$1488   PLACE TO START
       LDY    #CMSCOR  PLACE OF SCORES
       BSR    PRJNK

       LDA    #SCRMES        PUT OUT HEADERS ETC.
       JSR    WRD7V          DO IT

       MAKP   DECAZ
       MAKP   COLA
       MAKP   COLC  
       MAKP   COLD

       LDU    CRPROC
       LDA    #200
       STA    PD,U           NUMBER OF FRAMES TO FREEZE
TAB888 NAP    3,TABLE6
TABLE6 DEC    PD,U
       BNE    TAB888

       LDA    #$FF
       STA    PD,U
TAB777 NAP    4,TAB999
TAB999 LDA    PIA3
       ANDA   #$3      ONLY 2 SWS HERE
       ORAA   PIA2     ANY PRESSED?
       BEQ    TABLE7   NONE PRESSED
       DEC    PD,U     1 LESS COUNT
       BNE    TAB777

TABLE7 JMP    [PD+18,U]      RETURN

PRJNK  PULS   D              GET RETURN
       LDU    CRPROC
       STD    PD+11,U        SAVE RETURN

       STX    PD+15,U         SAVE W,H RETURN STOPS
       STX    PD+7,U
       STY    PD+9,U

       LDA    PD+1,U         GET NUMMBER PER COLUMN
       STA    PD+5,U         MAKE THIS NUMBER LEFT IN THIS COMUMN

TOD44  LDA    #4             NUMBER OF ENTRIES PER SLEEP
       STA    PD+17,U

       LDX    PD+7,U         RESTORE POINTERS
       LDY    PD+9,U

TOD33  BSR    CLSET          SET THE COLOR FOR THIS LINE

       LDB    PD+6,U         GET INDEX
       LDA    #INDMES
       JSR    PWSUB          PRINT "WORD" AT PROPER HEIGHT

       PSHS   X              SAVE POINTER

       LDB    #3             3 CHARS
TOD1   EXG    X,Y            GET CMOS POINTER IN X
       JSR    RCMOSA         GET THE LETTER
       EXG    X,Y

       JSR    PCSUB          PRINT THE CHARACTER PROPERLY 
       DECB
       BNE    TOD1

       PULS   X              GET THE INITIAL POINTER BACK
       LDA    PD,U           GET HEIGHT OF FONT
       LDB    #3
       MUL
       LSRB                  1/2
       INCB                  ADD 2 PIXELS
       TFR    B,A
       LDB    PD,U           CHECK FOR 5
       CMPB   #5
       BNE    TABFIX
       INCA                  1 MORE IN CASE OF STUPID WWW
TABFIX CLRB
       LEAX   D,X            PLACE FOR SCORE

       JSR    PRSCOR         PRINT THE GUYS SCORE

       LDA    PD+6,U
       ADDA   #1
       DAA
       STA    PD+6,U

       TFR    X,D            MOVE CURSOR OVER
       LDA    PD+15,U        RESTORE X
       ADDB   PD+3,U         ADD COLUMN SPACING
       TFR    D,X            RETURN AS POINTER

       DEC    PD+5,U         1 LESS IN CURRENT COLUMN
       BEQ    TOD22          DONE WITH COLUMN

TOD66  DEC    PD+17,U        1 LESS TILL SLEEP
       BNE    TOD33

       STX    PD+7,U         SAVE POINTERS
       STY    PD+9,U

       NAP    1,TOD44

TOD22  LDD    PD+15,U        GET CORNER POINTER
       ADDA   PD+4,U         ADD WIDTH SPACING
       STA    PD+15,U
       TFR    D,X            AND SHOVE IN X

       LDA    PD+1,U         GET NUMBER PER COLUMN
       STA    PD+5,U         NUMBER TO DO
       DEC    PD+2,U         NUMBER OF COLUMNS
       BNE    TOD66

       JMP    [PD+11,U]      RETURN NOW

*      CLSET - Y POINTS AT SCORE-6...IF IT IS EQUAL TO 
*      EITHER PLAYERS SCORE, AND NOT THE NEXT CMOS SCORE,
*      THEN THE COLOR IS CHANGED TO TCOL2,  ELSE TCOL1

CLSET  PSHS   X,Y,D
       LEAY   6,Y            POINT AT SCORE
       LDX    #ZP1SCR        CHECK PLAYER 1'S SCORE
       BSR    CKME
       BEQ    CLS1           IT WAS GOOD
       LDB    PLRCNT         HOW MANY PLAYERS
       DECB
       BEQ    CLS2           NO GOOD
       LDX    #ZP2SCR
       BSR    CKME
       BNE    CLS2           NO GOOD
CLS1   BSR    CKNXT          SEE IF NEXT MATCHES
       BEQ    CLS2           IT MATCHES...HOLD OFF
       LDA    TCOL2          USE NORMAL COLOR
       BRA    CLS3
CLS2   LDA    TCOL1
CLS3   STA    TEXCOL
       PULS   X,Y,D,PC

*      CKME   BEQ  FOR EQUAL
*      Y POINTS AT CMOS SCORE
*      X POINTS AT RAM SCORE

CKME   PSHS   Y
       EXG    X,Y            CMOSER USE X
       JSR    RCMOSA
       ANDA   #$F
       LDB    #4
CKME2  CMPA   ,Y+            DO WE MATCH
       BNE    CKME1          NOPE
       JSR    RCMOSA         GET NEXT
       DECB
       BNE    CKME2
CKME1  PULS   Y,PC

CKNXT  LEAY   1,Y            PUSH PAST CHECKSUM     
       LEAX   SCRSIZ,Y       X POINTS AT NEXT SCORE
       LDB    #7             7 BYTES TO CHECK
CKNXT1 LDA    ,X+
       EORA   ,Y+            LOWER GUYS
       ANDA   #$F
       BNE    CKBYE          NOT EQUAL
       DECB
       BNE    CKNXT1
CKBYE  RTS

PRSCOR EXG    X,Y
       JSR    RCMOSA
       JSR    RCMOSB         GET 2 BYTES OF SCORE
       EXG    X,Y
       ANDA   #$0F           TAKE AWAY CHECK NIBBLE
       BNE    PR2SC          2 PARTER
       TSTB                  ALL ZERO??
       BNE    PR2SC
       LDA    #HIGH3
       BRA    PR3SC
PR2SC  PSHS   Y
       TFR    D,Y
       LDA    #HIGH1
       JSR    PWSUB          PRINT WORD IN PROPER FONT
       PULS   Y              RESTOE Y
       LDA    #AUDM2         USE AUDIT TEXT FOR LOW HALF
PR3SC  PSHS   A              SAVE MESSAGE
       EXG    X,Y
       JSR    RCMOSD         GET THE STUFF
       EXG    X,Y
       PSHS   Y              SAVE Y AGAIN
       TFR    D,Y            PRINTED STUFF IN Y
       LDA    2,S            RESTORE MESSAGE NUMBER
       BSR    PWSUB          PRINT IT
       PULS   Y              RESTOE Y
       PULS   A,PC           AND RETURN

PWSUB  PSHS   U,X,A
       LDX    #WRD7V
       LDA    PD,U           GET FONT
       CMPA   #7             BIG?? 
       BEQ    PWSUB1
       LDX    #WRD5V         SMALL
PWSUB1 TFR    X,U
       PULS   A,X
       JSR    ,U
       PULS   U,PC

PCSUB  PSHS   U,X,A
       LDX    #PR57V
       LDA    PD,U
       CMPA   #7
       BEQ    PWSUB1
       LDX    #PR35V
       BRA    PWSUB1
       FCC    ' ROBOTRON: 2084 '
       FCC    ' COPYRIGHT 1982 WILLIAMS ELECTRONICS INC. '
       FCC    ' ALL RIGHTS RESERVED '

*      HASSLER UPPER LEFT IS $0308
*      HASSLER LOWER RIGHT IS $94F4  (BOTH INCLUSIVE)

FRAMER PULS   D              GET RETURN ADDRESS
       LDU    CRPROC
       STD    PD+8,U         AND SAVE

       LDX    #PCRAM
       LDD    #0
FRAME7 STD    ,X++
       CMPX   #PCRAM+16
       BLO    FRAME7
*
       LDX    #$60D          TERMINAL POINT
       STX    PD+16,U
       LDA    #$11
       STA    PD+14,U
*
FRAME3 LDX    #$3E7D
       STX    PD+10,U
       LDY    #$597F
       STY    PD+12,U

FRAME1 LDA    #2
       STA    PD+15,U        SLEEP COUNT

       LDX    PD+10,U
       LDY    PD+12,U
       LDA    PD+14,U

FRAME2 JSR    MARQ

       CMPX   PD+16,U
       BEQ    FRBYE
       LEAX   -$102,X
       LEAY   $102,Y
       BSR    GETA           GET THE A TO USE

       DEC    PD+15,U
       BNE    FRAME2

       STA    PD+14,U
       STX    PD+10,U
       STY    PD+12,U
       NAP    1,FRAME1       NIGHTY NIGHT

GETA   PSHS   B
       LDB    PD+16,U
       CMPB   #6             WRITING??
       BNE    GETA1          NOPE...LEAVE THE ZERO
       SUBA   #$11
       BNE    GETA1
       LDA    #$88
GETA1  PULS   B,PC

FRBYE  LDX    PD+16,U        CHECK LIMIT
       CMPX   #$60D
       BEQ    FRCONT         NOT DONE YET        
       JMP    [PD+8,U]       RETURN
FRCONT LDX    #$E1D          THIS IS OTHER LIMIT
       STX    PD+16,U        NEW LIMIT
       CLR    PD+14,U        STORE ZEROS
       JMP    FRAME3

LOOPP  LDX    #COLTAB        GET FIRST OF COLOR TABLE
LOOP1  LDA    ,X+            GET THE NEXT BYTE
       BSR    OUTCOL         CYCLE IT IN
       STX    PD,U           SAVE X
       NAP    3,LOOP2
LOOP2  LDX    PD,U           RESTORE X
       CMPX   #TABEDD
       BLO    LOOP1
       BRA    LOOPP

OUTCOL LDY    #PCRAM+1        START AT 1
OUTC1  LDB    1,Y
       STB    ,Y+
       CMPY   #PCRAM+8
       BLO    OUTC1          NOT THERE YET
       STA    ,Y
       RTS

COLA   LDY    #PCRAM+$A      Y = PCRAM LOCATION
       LDX    #CATAB         X = PLACE TO START
COMMM  LDD    #CATAB         D = START OF TABLE

COLENT STD    PD+4,U
       STY    PD+2,U
       BRA    COLA4

COLA3  LDX    PD+4,U
COLA4  STX    PD,U
COLA5  LDX    PD,U
       LDA    ,X+      GET NEXT BYTE
       BEQ    COLA3
       STA    [PD+2,U]
       STX    PD,U
       NAP    4,COLA5

CATAB  FCB    $7,$7,$7,$7,$7,$7,$7,$7,$57,$A7,$FF,$FF,$A7,$57,0

CCTAB  FCB    $FF,$FF,$FF,$FF,$FF,$FF,$FF,$E4,$D2,$C0,$C0,$C0
       FCB    $D2,$E4,0

DECAZ  LDX    #CATAB+7       OUT OF PHASE PLEASE
       LDY    #PCRAM+9
       BRA    COMMM 

COLC   LDX    #CCTAB+7
       LDY    #PCRAM+$C
COLLL  LDD    #CCTAB
       BRA    COLENT

COLD   LDX    #CCTAB  
       LDY    #PCRAM+$D
       BRA    COLLL 

COLTAB FCB    $37,$2F,$27,$1F,$17,$47,$47,$87
       FCB    $87,$C7,$C7,$C6,$C5,$CC,$CB,$CA
       FCB    $C0,$D0,$98,$38,$33
TABEDD EQU    *

UPPER  EQU    PD
LOWER  EQU    PD+1
LEFT   EQU    PD+2
RIGHT  EQU    PD+3
HFLAV  EQU    PD+4
LFLAV  EQU    PD+5

*      MARQ - DRAW A LINKY MARQUIS.  PASS UPPER LEFT IN X.
*      PASS LOWER RIGHT IN Y.  
*      RANGES ARE INCLUSIVE.  LOWER-UPPER MUST BE EVEN (ODD INCLUSIVE)
*      A CONTAINS THE HIGH AND LOW FLAVOR COLORS (<OUTER><INNER>)

MARQ   PSHS   X,Y,D  
       LDU    CRPROC
       ANDA   #$F0
       STA    HFLAV,U  SAVE FLAVOR COLORS
       LDA    ,S
       ANDA   #$0F
       STA    LFLAV,U
       TFR    X,D
       STA    LEFT,U
       STB    UPPER,U
       TFR    Y,D
       STA    RIGHT,U
       STB    LOWER,U
       SUBB   UPPER,U        LOWER-UPPER ODD???
       RORB                  CHECK
       BCC    MARQ1          NOPE
       DEC    LOWER,U        NOT THAT LOW...OK?

MARQ1  LDA    UPPER,U        DO TOP LINE
       BSR    HHIGH          HIGH PART OF TOP HORIZONTAL LINE
       INCA
       BSR    HLOW           LOW PART
*
       LDA    LOWER,U
       BSR    HHIGH          1 LESS FOR GEN PURP
       DECA
       BSR    HLOW
*
       LDA    LEFT,U         LEFT BYTE
       BSR    VHIGH          HIGH FLAVOR STUFF IN A VERT LINE
       BSR    VLOW           LOW FLAVOR STUFF IN A VERT LINE
*
       LDA    RIGHT,U
       BSR    VHIGH
       DECA
       BSR    VLOW
       PULS   D,X,Y,PC

*      VHIGH - PLOT VERTICAL LINE FROM [UPPER TO LOWER]
*      CONSISTING OF EVERY OTHER BYTE GETTING THE HIGH NIBBLE COLOR
*      A HAS THE X TO USE

VHIGH  PSHS   D,X
       BSR    SETUPV   SETUP X,B
       INCB            1 MORE TO BE INCLUSIVE
       LDA    HFLAV,U  GET HIGH FLAVOR
VHIGH1 STA    ,X++     STORE THE BYTE
       DECB
       BNE    VHIGH1
       PULS   D,X,PC

VLOW   PSHS   D,X
       BSR    SETUPV   SETUP X,B
       LEAX   1,X      MOVE X UP 1
       LDA    LFLAV,U  USE LOW FLAVOR
       BRA    VHIGH1

SETUPV LDB    UPPER,U  GET UPPER Y
       TFR    D,X      THIS IS PONTER
       LDB    LOWER,U
       SUBB   UPPER,U  THIS IS GUARANTEED EVEN
       LSRB            1/2 PLEASE (EVERY 2 IS PLOTTED)
       RTS

*      HHIGH - THIS ROUTINE PLOTS HIGH COLOR IN THE HIGH HALF OF
*      CONSECUTIVE HORIZONTAL PIXELS GOING FROM 
*      [LEFT TO RIGHT] AT THE HEIGHT GIVEN IN REGISTER A.

HHIGH  PSHS   X,D
       BSR    SETUPH   SETUP COUNT IN B, X POINTS AT FIRST BYTE
       INCB            ADD 1 TO MAKE IT INCLUSIVE
       LDA    HFLAV,U  GET THE DATA BYTE
HHIGH1 STA    ,X       STORE 1
       LEAX   $100,X   PUSH TO NEXT
       DECB            1 LESS
       BNE    HHIGH1
       PULS   X,D,PC

*      HLOW - THIS ROUTINE PLOTS LOW COLOR IN THE LOW HALF OF
*      CONSECUTIVE HORIZONTAL PIXELS GOING FROM
*      [LEFT TO RIGHT) AT THE HEIGHT GIVEN IN REGISTER A.

HLOW   PSHS   X,D
       BSR    SETUPH
       LDA    LFLAV,U  USE LOW FLAVOR
       BRA    HHIGH1   AND DO NORMAL PROCESSING ABOVE 

SETUPH TFR    A,B      MOVE Y OVER
       LDA    LEFT,U   GET LEFT X
       TFR    D,X      MAKE POINTER
       LDB    RIGHT,U
       SUBB   LEFT,U   FIND DIFFERENCE (EXCLUSIVE
       RTS             AND RETURN

       END
