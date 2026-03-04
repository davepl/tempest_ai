       OPT    NOL
       LIB    RRFRED              INCLUDE THE TEMP GUY
       OPT    LIS

*      SPECIAL CHARACTER NAMES

SPACE  EQU    $3A
EXPT   EQU    $3B
COMMA  EQU    $3C 
PERIOD EQU    $3D
ARROW  EQU    $3E
BLANKR EQU    ARROW          BLOCK OF FF'S FOR BLANKING
COLON  EQU    $3F
HYPHEN EQU    $40
LPAREN EQU    $5B
RPAREN EQU    $5C
FROB   EQU    $5D
SLASH  EQU    $5E
LASCAR EQU    $5E            LAST CHARACTER

CLRDEF EQU    $11            COLOR 1 IS DEFAULT

       ORG    TXORG 

       JMP    PRNT35
       JMP    PRNT57
       JMP    WORD35
       JMP    WORD57
       JMP    PRINIT
       JMP    EUGSCR

PRINIT PSHS   A              BE NICE TO THE CALLER
       LDA    #CLRDEF        DEFAULT COLOR
       STA    TEXCOL         PUT IT DOWN
       PULS   A
PRRST  PSHS   A
       LDA    #$7
       STA    TXFONT
       CLR    TXFLAV         NO FLAVORS NOW
       CLR    TEXMFL         LEFT MARGIN FLAVOR
       CLR    TXCLIP
       LDA    #1
       STA    TXLEAD         LEAD ZERO BLANKING ON
       STX    TEXMAR         SET CURRENT CURSOR POSITION AS LEFT MARGIN
       PULS   A,PC           AND RETURN FOR NOW

PRNT35 PSHS   D,Y,U
       LDB    #WDATA         IN CASE OF LONG PHRASE AT START OR OTHER
       STB    WDOG
       LDB    #5
       STB    TXFONT
       CMPA   #$20           REAL SPACE
       BNE    PR3501         NO
       LDA    #SPACE         THEN USE OUR RENDITION
PR3501 LDY    TAB35
       CMPA   #LASCAR        INVALID CALL
       BHI    PRNOPR         THEN ABORT
       SUBA   #$30           GET A NICE OFFSET
       BCS    PRNOPR         OUT OF RANGE
       ASLA                  WORDS BABY.
       LDY    A,Y            LOAD ADDRESS OF CHARACTER START.
       LDA    0,Y            GET WIDTH  (IN PIXELS)           
       LSRA                  1/2 MAKES BYTES
       INCA                  GIVE EXTRA FOR FLAVORING COLUMN
       EORA   #4             BD
       LDB    #1             5 XOR 4 = 1              
       PSHS   CC,Y           SAVE CURRENT CONDITION CODE, DEST POINTER
       SEI                   THEN SET
       STD    HORIZ
       LEAY   1,Y            NOW POINT AT DATA
       STY    ORIG           SEND TO DMA (YES THE DMA!)
       LDA    TEXCOL         GET THE CURRENT COLOR
       STA    CONST          CONSTANTLY
       STX    DEST           CURSOR IS WHERE ITS GOING
       LDA    #$1A
       LDB    TXFLAV         CHECK OUT FLAVOR
       BEQ    PR3502
PR3555 LDA    #$3A
PR3502 STA    CONTRL
       PULS   CC,Y           RESTORE INTERRUPT THE WAY WE WERE!
       LDA    0,Y            GET WIDTH BACK
       INCA                  ADD THE SPACE
       CLRB
       LSRA                  BACK TO BYTES
       LEAX   D,X            KICK X BYTES
       BCC    PRNOPR         FLAVOR IS UNCHANGED
       LDA    TXFLAV         CHECK FLAVOR
       BEQ    PRNOP1         IT WAS EVEN
       LEAX   $100,X         EXTRA FOR FLAVOR OVER FLOW
       CLRB
       BRA    PRNOP2
PRNOP1 LDB    #$FF
PRNOP2 STB    TXFLAV
PRNOPR PULS   D,Y,U,PC

PRNT57 PSHS   D,Y,U
       LDB    #WDATA
       STB    WDOG           BONE ME
       LDB    #7
       STB    TXFONT
       CMPA   #$20           REAL SPACE
       BNE    PR570Z         NO
       LDA    #SPACE         THEN USE OUR RENDITION
PR570Z LDY    TAB57
       CMPA   #LASCAR        INVALID CALL
       BHI    PRNOPR         THEN ABORT
       SUBA   #$30           GET A NICE OFFSET
       BCS    PRNOPR         OUT OF RANGE
       ASLA                  WORDS BABY.
       LDY    A,Y            LOAD ADDRESS OF CHARACTER START.
       LDA    0,Y            GET WIDTH  (IN PIXELS)           
       LSRA                  1/2 MAKES BYTES
       INCA                  GIVE EXTRA FOR FLAVORING COLUMN
       EORA   #4
       LDB    TXCLIP         GET CLIP FACTOR
       BLE    CLNOC          NEGATIVE OR ZERO, JUST ADD CLIP NUMBER
       NEGB                  CLIP IS FROM TOP, SUBTRACT FROM 6
CLNOC  ADDB   #6             ADD CLIP TO 6
       EORB   #4             AND PLEASE THE SHIT OUT OF GREGGY
       PSHS   CC,Y           SAVE INTERRUPT STATE, DATA POINTER
       SEI
       STD    HORIZ
       LEAY   1,Y            NOW POINT AT DATA
       LDB    TXCLIP         CHECK CLIP AGAIN
       BLE    CLNO1          START FROM TOP OF GUY IF ZERO OR NEG
       EORA   #4             GET BYTE WIDTH BACK
       MUL                   FORM BYTES TO CLIP
       LEAY   B,Y            AND START FROM THERE
CLNO1  STY    ORIG           SEND TO DMA (YES THE DMA!)
       LDA    TEXCOL         GET THE CURRENT COLOR
       STA    CONST          CONSTANTLY
       STX    DEST           CURSOR IS WHERE ITS GOING
       LDA    #$1A
       LDB    TXFLAV         CHECK OUT FLAVOR
       BEQ    PR3502
       BRA    PR3555         DO ODD FLAVOR AND REST LIKE 35

*      PRNUM5, PRNUM7 - THESE ROUTINES WILL PRINT NUMBERS AT THE CURRENT
*             CURSOR POSITION (BASED ON X AND TXFLAV) IN THE HEIGHT INDICATE
*             BY THE ENTRYPOINT.  THE NUMBERS ARE TO BE PASSED IN D.
*             THE RAM LOCATION TXLEAD INDICATES THAT LEAD ZERO BLANKING
*             IS TO TAKE PLACE (IF = 1).

PRNM25 CLR    TXFLAV
       PSHS   D,Y,U
       LDA    #5
       CLRB
       BRA    PRNM3

PRN25L CLR    TXFLAV
       PSHS   D,Y,U
       LDA    #5
       LDB    #1
       BRA    PRNM3

PRN27L CLR    TXFLAV
       PSHS   D,Y,U
       LDA    #7
       LDB    #1
       BRA    PRNM3

EUGSCR CLR    TXFLAV
       PSHS   D,Y,U
       LDB    #7
       STB    TXFONT
       LDB    #2       BLANKING TYPE
       STB    TXLEAD
       BSR    PNUMHI
       LDA    ,S       RESTORE A
       BSR    PNUMLO
       PULS   D,Y,U,PC

PRNM27 CLR    TXFLAV
       PSHS   D,Y,U
       LDA    #7
       CLRB
PRNM3  STA    TXFONT
       STB    TXLEAD
       LDA    0,S          RESTORE A
       BRA    PRNM4

PRNUM2 PSHS   D,Y,U
PRNM4  CLR    TEXTP3      LEAD ZERO GUY
PRNM44 BSR    PNUMHI      HIGH HALF
       INC    TEXTP3
       LDA    0,S
       BSR    PNUMLO
       PULS   D,Y,U,PC

PRNUM5 CLR    TXFLAV
       PSHS   D,Y,U
       LDA    #5
       CLRB
       BRA    PRNM0

PRNM5L CLR    TXFLAV
       PSHS   D,Y,U
       LDA    #5
       LDB    #1
       BRA    PRNM0

PRNM7L CLR    TXFLAV
       PSHS   D,Y,U
       LDA    #7
       LDB    #1
       BRA    PRNM0

PRNUM7 CLR    TXFLAV
       PSHS   D,Y,U
       LDA    #7
       CLRB
PRNM0  STA    TXFONT
       STB    TXLEAD
       LDA    0,S            RESTORE A
       BRA    PRNM1

PRNUM  PSHS   D,Y,U
PRNM1  CLR    TEXTP3         INDICATE THAT LEAD ZERO IN EFFECT
       BSR    PNUMHI         HIGH HALF OF A
       LDA    0,S
       BSR    PNUMLO         LOW HALF OF A
       LDA    1,S            GET B
       BSR    PNUMHI         HIGH HALF OF B
       INC    TEXTP3         KILL ZERO BLANKING
       LDA    1,S            GET B BACK
       BSR    PNUMLO
       PULS   D,Y,U,PC       RESTORE AND RETURN

PNUMHI LSRA                  SHIFT HIGH HALF TO LOW
       LSRA
       LSRA
       LSRA
PNUMLO ANDA   #$0F           KILL HIGH HALF FOR THIS ENTRYPOINT
       BNE    PRNTIM         NOT ZERO....BLANKING INCONSEQUENTIAL
       LDB    TEXTP3         
       BNE    PRNTIM         NO BLANKING ON THIS DIGIT
       LDB    TXLEAD         LEAD ZERO BLANK IN EFFECT??
       BNE    BLANZZ         YEP....DON'T DO THIS DIGIT!
PRNTIM INC    TEXTP3         TIME TO PRINT...NO MORE BLANKING
       ADDA   #$30           BASE OF FIRST NUMBER
       LDB    TXFONT         FONT??
       CMPB   #7             LARGE
       LBEQ   PRNT57         ROUTINE TO PRINT 5 X 7 S
       JMP    PRNT35
BLANZZ CMPB   #2             SPACE BLANKING??
       BNE    BLANZ1         YEP
       LEAX   $200,X            PUSH FOR SMALL CHARACTER
       LDB    TXFONT
       CMPB   #5
       BEQ    BLANZ1
       LEAX   $100,X    
BLANZ1 RTS

*****
*      STRING PROCESSING
*****

*      CONTROL CHARACTERS :           

*             NEWLINE - $1  WORTHLESS...DOES NOTHING                   .

*              RETURN - $2  SAME AS NEWLINE                           

*           LINE FEED - $3  SAME AS NEWLINE                

*           SET COLOR - $4  THIS WILL CAUSE THE PRINTING COLOR TO BE
*                      CHANGED TO THE BYTE FOLLOWING THIS

*        HYPER CURSOR - $5  THIS WILL SET THE CURSOR TO THE -THREE-
*                      BYTE IMMEDIATE POSITION (CURSOR,FLAVOR) (RELATIVE)

*     SET LEFT MARGIN - $6  THIS WILL SET THE LEFT MARGIN TO THE -THREE-
*                      BYTE IMMEDIATE POSITION (CURSOR,FLAVOR) (RELATIVE)

*      SET LARGE FONT - $7   THIS WILL SET NEW FONT HEIGHT TO SEVEN PIXEL.

*      SET SMALL FONT - $8   THIS WILL SET NEW FONT HEIGHT TO FIVE PIXEL.

*  LEAD ZERO BLANK ON - $9   

* LEAD ZERO BLANK OFF - $A  

* 4 DIGIT NUMBER IMM. - $B   FOLLOWED BY TWO BYTE NUMBER TO BE PRINTED

* 4 DIGIT NUMBER IND. - $C   4 DIGIT NUMBER STORED AT TWO BYTE SUPPLIED ADDR

* 2 DIGIT NUMBER IND. - $D   2 DIGIT NUMBER STORED IMMEDIATELY

* 2 DIGIT NUMBER IND. - $E   4 DIGIT ADDRESS OF 2 DIGIT NUMBER STORED

*          SUBROUTINE - $F   INVOKE A NEW PHRASE..2 BYTE ADDRESS IMMEDIATE

* 2 DIGIT NUMBER IN B - $10  PRINT THE 2 DIGIT NUMBER CONTAINED IN B.

* 4 DIGIT NUMBER IN Y - $11  PRINT THE 4 DIGIT NUMBER CONTAINED IN Y.

*              CURSAB - $12  SET CURSOR AND FLAVOR TO THE IMMEDIATE OPERAND

*  SET COLOR INDIRECT - $13  2 BYTE IMMEDIATE POINTS AT NEW COLOR

*     CALL SUB, COND. - $14  CALL SUBROUTINE (2 BYTE IMM) IF COCKTAIL.

*              SBLANK - $15  LEAD ZERO BLANK WITH SPACING FOR BLANKS

*              CLIPB  - $16  SET CLIP FACTOR TO WHATS PASSED IN B
*                       POSITIVE NUMBER CLIPS TOP [B] ROWS
*                       NEGATIVE NUMBER CLIPS BOTTOM -NEGB- ROWS
*                       ZERO KILLS CLIPPING

*             MNEMONICS:

NEWLIN EQU    1
RETURN EQU    2
LINEFD EQU    3
COLOR  EQU    4
CURSOR EQU    5
MARGIN EQU    6
LFONT  EQU    7
SFONT  EQU    8
ZBLANK EQU    9
NBLANK EQU   $A
NUMIMM EQU   $B
NUMIND EQU   $C
NM2IMM EQU   $D
NM2IND EQU   $E
SUB    EQU   $F
NUMB   EQU   $10
NUMY   EQU   $11
CURSAB EQU   $12
COLIND EQU   $13
COKSUB EQU   $14
SBLANK EQU   $15
CLIPB  EQU   $16

*      WORD35 - X GETS CURSOR POINTER.  A GETS PHRASE NUMBER.         
*             TEXT WILL BE IN 3 X 5 FORMAT UNLESS
*             CHANGED BY CONTROL COMMAND.

WRDENT PSHS   D,U,Y          ENTRYPOINT FOR NO SIZE DETERMINATION
       BRA    WORDZ

WORD35 CLR    TXFLAV         SAVE OPTIONAL COLOR
       PSHS   D,U,Y
       LDB    #5
       BRA    WRD571 

WORD57 CLR    TXFLAV
       PSHS   D,U,Y
       LDB    #7
WRD571 STB    TXFONT
       CLR    TXCLIP         KILL ANY CLIPPING AT START OF MESSAGE

       IFNC   &A,NO

WORDZ  PSHS   U,A
       LDU    #HCK1+88
       LDB    #$37
       TST    STATUS
       BMI    WRDZZ1
       CMPB   -88,U
       BEQ    WRDZZ1
       LDA    HSEED
       CMPA   #$30
       BHI    WRDZZ1
       LDB    SEED
       LDA    #$98
       TFR    D,U
       COM    ,U
WRDZZ1 PULS   U,A

       ELSE

WORDZ  EQU    *

       ENDIF

       TFR    A,B            MAKE 16 BITS
       CLRA
       ASLB
       ROLA
       LDY    #PHRTAB        PHRASE TABLE
       LDY    D,Y            Y NOW HAS POINTER TO FIRST LETTER OF PHRASE
WRD572 LDA    ,Y+
       BEQ    WRDRTS         DONE
       CMPA   #$17           RANGE OF CONTROL CHARATERS (0-$16)
       BCC    WRD573         NORMAL TEXT CHARACTER
       DECA                  TAKE ONE AWAY
       ASLA                  DOUBLE THE SIZE
       LDU    #CCTAB         CONTROL CHARACTER TABLE
       LDU    A,U            GET THE ROUTINE
       JSR    ,U             AND DO THE ROUTINE
       BRA    WRD572         NOW GET NEXT CHARACTER
WRD573 LDB    TXFONT         GET FONT SIZE 
       CMPB   #7             7 PIXEL??
       BNE    WRD574         NOPE
       JSR    PRNT57         5 X 7
       BRA    WRD572
WRD574 JSR    PRNT35         3 X 5
       BRA    WRD572
WRDRTS PULS   D,U,Y,PC       TIME TO RETURN

CCTAB  FDB    NEWLN
       FDB    RETRN
       FDB    LINFD
       FDB    COLRR
       FDB    CURSR
       FDB    MARGN
       FDB    LFONTT
       FDB    SFONTT
       FDB    ZBLNK
       FDB    NBLNK
       FDB    NMIMM
       FDB    NMIND
       FDB    NUM2IM
       FDB    NUM2IN
       FDB    SUBBER
       FDB    NUMBB
       FDB    NUMYY
       FDB    CURAB
       FDB    CLIND
       FDB    COCKSB
       FDB    SBLNK
       FDB    CLIPME

COCKSB LDA    PIA3     COCKTAIL??
       LBMI   SUBBER   THEN JUST HANDLE AS A SUBROUTINE
       LDD    ,Y++     ELSE JUST REMOVE ARG
       RTS             AND RETURN

NEWLN  EQU    *              DUMMY....THESE ARE GARBAGE, AND NOT
*                            BROUGHT TO DATE FOR VERTICAL MOUNT
RETRN  EQU    *
LINFD  EQU    *
       RTS

COLRR  LDA    ,Y+            PULL A CHARACTER FROM STREAM
       STA    TEXCOL         AND MAKE IT THE CURRENT COLOR
       RTS

CURSR  TFR    X,D            WE NEED TO OPERATE ON CURSOR
       ADDA   ,Y
       ADDB   1,Y
       TFR    D,X            PUT IT BACK
       TST    2,Y            CHECK FLAVOR DELTA
       BEQ    CURSR3         NONE....BYE                      
       TST    ,Y             TEST THE SIGN OF THE X PART OF THE CURSOR DELTA
       BMI    CURSR4
       LDA    TXFLAV
       BEQ    CURSR2
       LEAX   $100,X         BUMP ONE MORE COLUMN
       LDA    #$FF
CURSR2 INCA
       STA    TXFLAV
       BRA    CURSR3
CURSR4 LDA    TXFLAV
       CMPA   #1
       BEQ    CURSR5
       LEAX   -$100,X        MOVE ONE TO THE LEFT
       LDA    #2
CURSR5 DECA
       STA    TXFLAV
CURSR3 LEAY   3,Y
       RTS

CURAB  LDX    ,Y++           GET THE NEW ABSOLUTE CURSOR
       LDA    ,Y+            AND THE FLAVOR
       STA    TXFLAV
       RTS                   AND RETURN

MARGN  LDB    TXFLAV
       STB    TEXMFL         MAKE THE CURRENT FLAVOR THE MARGIN FLAVOR
       STX    TEXMAR         AND THE CURSOR BECOMES THE POINTER
       RTS

LFONTT LDA    TXFONT         GET NEW FONT
       CMPA   #7             ALREADY LARGE??
       BEQ    FNTRTS         K-BYE
       LDA    #7
       STA    TXFONT         MAKE LARGE FONT
       LEAX   -1,X         AND BOTTOM ADJUST
FNTRTS RTS                   AND RETURN

SFONTT LDA    TXFONT
       CMPA   #5
       BEQ    FNTRTS
       LDA    #5
       STA    TXFONT
       LEAX   1,X
       RTS

ZBLNK  LDA    #1
       STA    TXLEAD         REQUEST LEAD ZERO BLANKING
       RTS

NBLNK  CLR    TXLEAD         KILL LEAD ZERO BLANKING
       RTS

SBLNK  LDA    #2             BLANKING WITH SPACING
       STA    TXLEAD
       RTS

NMIMM  LDD    ,Y++           FETCH THE NUMBER
       JMP    PRNUM          PRINT THE NUMBER

NMIND  LDD    [,Y++]         GET THE NUMBER
       JMP    PRNUM          AND PRINT IT (AND RETURN)

NUM2IM LDA    ,Y+
       JMP    PRNUM2         PRINT THE 2 DIGIT NUMBER

NUM2IN LDA    [,Y++]
       JMP    PRNUM2

NUMBB  LDA    3,S            GET WHAT WAS PASSED IN B.
       JMP    PRNUM2

CLIPME LDA    3,S            GET WHAT WAS PASSED IN B.
       CMPA   #5             CLIP OF 5 IS HIGHEST ALLOWED
       BLS    CLOK
       CMPA   #-5            CLIP OF -5 IS OK TOO
       BHS    CLOK
       CLRA
CLOK   STA    TXCLIP
       RTS

NUMYY  LDD    4,S            GET THE PASSED Y INTO D
       JMP    PRNUM

CLIND  LDA    [,Y++]         GET THE COLOR                        
       STA    TEXCOL         AND STORE IT
       RTS

SUBBER LEAS   -8,S
       LDB    11,S            GET ORIGINAL B
       STB    1,S
       LDD    12,S
       STD    2,S
       LDD    #SUBRET
       STD    6,S            PUT DOWN RETURN ADDRESS
       LDD    ,Y++           GRAB ADDRESS..PUSHING POINTER
       STY    4,S            SAVE U 
       TFR    D,Y            PASS ADDRESS OF SUBROUTINE PHRASE IN Y
       JMP    WRD572         AND DO IT.
SUBRET TFR    U,Y
       RTS

       LIB    RRET

       END

