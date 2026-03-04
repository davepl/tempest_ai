*      ROBOTRON PART OF LOGO                             

*      IF SIGN BIT IS ON, THE BYTE IS A KEY BYTE,
*      ELSE IT IS A DATA BYTE.

*      KEY BYTE:

*      1 R L C   N N N N

*      R = REPEAT LAST COLUMN   N N N N   TIMES
*      L = END OF LETTER
*      C = END OF COLUMN

*      ELSE THE BYTE IS OF THE FORM:
*
*      0 B C D   D D D D 
*
*      BC  00 = ZERO  10 = COLOR 1  01 = COLOR 2   11 = WHO KNOWS?
*      D = NUMBER OF ROWS TO GET THIS

REPEAT EQU    $C0
ENDLET EQU    $A0
ENDCOL EQU    $90

ON     EQU    $40      INDICATE COLOR IS ON
ON2    EQU    $20      COLOR 2 ON

*      LETTER R
*      LETTER R
*      LETTER R

LR     FCB    1        COLUMN 1
       FCB    25+ON2
       FCB    ENDCOL

       FCB    2+ON2
       FCB    23+ON    COLUMN 2
       FCB    2+ON2
       FCB    ENDCOL

       FCB    1+ON2    COL 3
       FCB    25+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    1+ON2    COL 4
       FCB    3+ON
       FCB    7+ON2
       FCB    15+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    1+ON2
       FCB    3+ON     COLUMN 5
       FCB    1+ON2
       FCB    5    
       FCB    2+ON2
       FCB    13+ON
       FCB    2+ON2
       FCB    ENDCOL

       FCB    1+ON2
       FCB    3+ON     COLUMN 6
       FCB    1+ON2
       FCB    6    
       FCB    1+ON2
       FCB    2+ON
       FCB    12+ON2
       FCB    ENDCOL

       FCB    1+ON2
       FCB    3+ON     COLUMN 7
       FCB    1+ON2
       FCB    6
       FCB    1+ON2
       FCB    2+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    REPEAT+6 COLUMN 8,9,10,11,12,13

       FCB    1+ON2
       FCB    3+ON     COLUMN 14
       FCB    2+ON2
       FCB    5
       FCB    1+ON2
       FCB    2+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    1+ON2
       FCB    4+ON    COLUMN 15
       FCB    7+ON2
       FCB    2+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    1+ON2    COL 16
       FCB    13+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    3+ON2
       FCB    11+ON    COLUMN 17
       FCB    2+ON2
       FCB    ENDCOL

       FCB    2        COLUMN 18
       FCB    10+ON2
       FCB    3+ON
       FCB    11+ON2
       FCB    ENDCOL

       FCB    10       COLUMN 19
       FCB    2+ON2
       FCB    13+ON
       FCB    2+ON2
       FCB    ENDCOL

       FCB    11       COLUMN     20
       FCB    2+ON2
       FCB    12+ON
       FCB    2+ON2
       FCB    ENDCOL

       FCB    12       21
       FCB    14+ON2
       FCB    ENDCOL

       FCB    ENDLET   END OF LETTER

*      LETTER O
*      LETTER O
*      LETTER O

LO     FCB    1        COLUMN 1
       FCB    25+ON2
       FCB    ENDCOL

       FCB    2+ON2
       FCB    23+ON    COLUMN 2
       FCB    2+ON2
       FCB    ENDCOL

       FCB    1+ON2
       FCB    25+ON
       FCB    1+ON2    COL 3
       FCB    ENDCOL

       FCB    1+ON2    4
       FCB    3+ON
       FCB    19+ON2
       FCB    3+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    1+ON2
       FCB    2+ON
       FCB    2+ON2
       FCB    17
       FCB    2+ON2
       FCB    2+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    1+ON2
       FCB    2+ON     COLUMN 6
       FCB    1+ON2
       FCB    19
       FCB    1+ON2
       FCB    2+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    REPEAT+8    COLUMN 7,8,9,10,11,12,13,14

       FCB    1+ON2
       FCB    2+ON         COLUMN 15
       FCB    4+ON2
       FCB    16
       FCB    1+ON2
       FCB    2+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    1+ON2
       FCB    5+ON           COLUMN 16
       FCB    2+ON2
       FCB    15
       FCB    1+ON2
       FCB    2+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    1+ON2
       FCB    6+ON           COLUMN 17
       FCB    2+ON2
       FCB    13
       FCB    2+ON2
       FCB    2+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    1+ON2          18
       FCB    7+ON
       FCB    15+ON2
       FCB    3+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    1+ON2
       FCB    25+ON
       FCB    1+ON2
       FCB    ENDCOL         19

       FCB    2+ON2
       FCB    23+ON          20
       FCB    2+ON2
       FCB    ENDCOL

       FCB    1
       FCB    25+ON2
       FCB    ENDCOL         21

       FCB    ENDLET         END OF LETTER

*      LETTER B
*      LETTER B
*      LETTER B

LB     FCB    1              COLUMN 1
       FCB    25+ON2
       FCB    ENDCOL

       FCB    2+ON2
       FCB    23+ON          2
       FCB    1+ON2
       FCB    ENDCOL

       FCB    1+ON2
       FCB    24+ON          3
       FCB    2+ON2
       FCB    ENDCOL

       FCB    1+ON2          4
       FCB    2+ON
       FCB    7+ON2
       FCB    16+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    1+ON2
       FCB    2+ON           5
       FCB    1+ON2
       FCB    5
       FCB    2+ON2
       FCB    15+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    1+ON2
       FCB    2+ON           6
       FCB    1+ON2
       FCB    6
       FCB    1+ON2
       FCB    15+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    1+ON2          7
       FCB    2+ON
       FCB    1+ON2
       FCB    6
       FCB    1+ON2
       FCB    3+ON
       FCB    9+ON2
       FCB    3+ON 
       FCB    1+ON2
       FCB    ENDCOL

       FCB    1+ON2
       FCB    2+ON           8
       FCB    1+ON2
       FCB    6
       FCB    1+ON2
       FCB    2+ON
       FCB    2+ON2
       FCB    7
       FCB    2+ON2
       FCB    2+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    1+ON2
       FCB    2+ON           9
       FCB    1+ON2
       FCB    6
       FCB    1+ON2
       FCB    2+ON
       FCB    1+ON2
       FCB    9
       FCB    1+ON2
       FCB    2+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    REPEAT+5     10,11,12,13,14

       FCB    1+ON2
       FCB    2+ON         15
       FCB    2+ON2
       FCB    5
       FCB    1+ON2
       FCB    2+ON
       FCB    1+ON2
       FCB    9
       FCB    1+ON2
       FCB    2+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    1+ON2
       FCB    3+ON
       FCB    7+ON2
       FCB    2+ON
       FCB    1+ON2
       FCB    9
       FCB    1+ON2
       FCB    2+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    1+ON2        17
       FCB    12+ON
       FCB    1+ON2
       FCB    9
       FCB    1+ON2
       FCB    2+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    2+ON2
       FCB    11+ON          18
       FCB    2+ON2
       FCB    7
       FCB    2+ON2
       FCB    2+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    1              19
       FCB    10+ON2
       FCB    3+ON
       FCB    9+ON2
       FCB    3+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    10             20
       FCB    1+ON2
       FCB    14+ON
       FCB    2+ON2
       FCB    ENDCOL

       FCB    10             21
       FCB    2+ON2
       FCB    11+ON
       FCB    3+ON2
       FCB    ENDCOL

       FCB    11             22
       FCB    13+ON2
       FCB    ENDCOL

       FCB    ENDLET

*      LETTER T
*      LETTER T
*      LETTER T

LT     FCB    1              1
       FCB    3+ON2
       FCB    ENDCOL

       FCB    2+ON2
       FCB    1+ON           2
       FCB    2+ON2
       FCB    ENDCOL

       FCB    1+ON2          3            
       FCB    3+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    REPEAT+4       4,5,6,7,8

       FCB    1+ON2
       FCB    3+ON           9
       FCB    2+ON2
       FCB    ENDCOL

       FCB    1+ON2
       FCB    4+ON          10
       FCB    21+ON2
       FCB    ENDCOL

       FCB    1+ON2          11
       FCB    24+ON
       FCB    2+ON2
       FCB    ENDCOL

       FCB    1+ON2
       FCB    25+ON          12
       FCB    1+ON2
       FCB    ENDCOL

       FCB    REPEAT+1       13

       FCB    1+ON2
       FCB    4+ON           14
       FCB    7+ON2
       FCB    13+ON
       FCB    2+ON2
       FCB    ENDCOL

       FCB    1+ON2
       FCB    3+ON           15
       FCB    2+ON2
       FCB    5
       FCB    15+ON2
       FCB    ENDCOL

       FCB    1+ON2
       FCB    3+ON           16
       FCB    1+ON2
       FCB    ENDCOL

       FCB    REPEAT+4       17,18,19   

       FCB    2+ON2          20
       FCB    1+ON
       FCB    2+ON2
       FCB    ENDCOL

       FCB    1              21
       FCB    3+ON2
       FCB    ENDCOL
       FCB    ENDLET

*      LETTER N
*      LETTER N
*      LETTER N

LN     FCB    1        1
       FCB    25+ON2
       FCB    ENDCOL

       FCB    2+ON2
       FCB    23+ON    2
       FCB    2+ON2
       FCB    ENDCOL

       FCB    1+ON2    3
       FCB    25+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    1+ON2
       FCB    2+ON     4
       FCB    8+ON2
       FCB    15+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    1+ON2
       FCB    2+ON     5
       FCB    1+ON2
       FCB    6
       FCB    1+ON2
       FCB    15+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    1+ON2
       FCB    2+ON     6
       FCB    1+ON2
       FCB    6
       FCB    2+ON2
       FCB    12+ON
       FCB    2+ON2
       FCB    ENDCOL

       FCB    1+ON2
       FCB    2+ON     7
       FCB    1+ON2
       FCB    7
       FCB    14+ON2
       FCB    ENDCOL

       FCB    1+ON2
       FCB    2+ON     8
       FCB    1+ON2
       FCB    ENDCOL

       FCB    REPEAT+8  9,10,11,12,13,14,15,16

       FCB    1+ON2
       FCB    2+ON    17
       FCB    23+ON2
       FCB    ENDCOL

       FCB    1+ON2
       FCB    24+ON     18
       FCB    2+ON2
       FCB    ENDCOL

       FCB    2+ON2     19    
       FCB    23+ON
       FCB    2+ON2
       FCB    ENDCOL

       FCB    1        20
       FCB    25+ON2
       FCB    ENDCOL

       FCB    ENDLET

*      COLON
*      COLON
*      COLON

LC     FCB    3        1
       FCB    4+ON2
       FCB    14
       FCB    4+ON2
       FCB    ENDCOL

       FCB    2        2
       FCB    2+ON2
       FCB    2+ON
       FCB    2+ON2
       FCB    12
       FCB    2+ON2
       FCB    2+ON
       FCB    2+ON2
       FCB    ENDCOL

       FCB    2        3
       FCB    1+ON2
       FCB    4+ON
       FCB    1+ON2
       FCB    12
       FCB    1+ON2
       FCB    4+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    REPEAT+5  4,5,6

       FCB    2        7
       FCB    2+ON2
       FCB    2+ON
       FCB    2+ON2
       FCB    12
       FCB    2+ON2
       FCB    2+ON
       FCB    2+ON2
       FCB    ENDCOL

       FCB    3        8
       FCB    4+ON2
       FCB    14
       FCB    4+ON2
       FCB    ENDCOL

       FCB    ENDLET

******
*
*      2084 DEFINITION
*
******

L8     FCB    1        1
       FCB    1+ON2
       FCB    12+ON
       FCB    3+ON2
       FCB    12+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    2        2
       FCB    1+ON2
       FCB    12+ON
       FCB    1+ON2
       FCB    12+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    3        3
       FCB    1+ON2
       FCB    23+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    4        4
       FCB    1+ON2
       FCB    21+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    5        5
       FCB    1+ON2
       FCB    3+ON
       FCB    6+ON2
       FCB    1+ON
       FCB    6+ON2 
       FCB    3+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    6        6
       FCB    1+ON2
       FCB    3+ON
       FCB    1+ON2
       FCB    4
       FCB    1+ON2
       FCB    4
       FCB    1+ON2
       FCB    3+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    7        7
       FCB    1+ON2
       FCB    3+ON
       FCB    1+ON2
       FCB    3
       FCB    1+ON2
       FCB    3
       FCB    1+ON2
       FCB    3+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    8        8
       FCB    1+ON2
       FCB    3+ON
       FCB    3+ON2
       FCB    1+ON
       FCB    3+ON2
       FCB    3+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    9        9
       FCB    1+ON2
       FCB    11+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    10       10
       FCB    1+ON2
       FCB    9+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    11       11
       FCB    1+ON2
       FCB    3+ON
       FCB    1+ON2
       FCB    3+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    12       12
       FCB    8+ON2
       FCB    ENDCOL

       FCB    ENDLET

L4     FCB    13       1
       FCB    1+ON2
       FCB    5+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    12       2
       FCB    1+ON2
       FCB    6+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    11       3
       FCB    1+ON2
       FCB    7+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    10       4
       FCB    1+ON2
       FCB    8+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    9        5
       FCB    1+ON2
       FCB    5+ON
       FCB    1+ON2
       FCB    3+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    8        6
       FCB    1+ON2
       FCB    5+ON
       FCB    2+ON2
       FCB    3+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    8        7
       FCB    1+ON2
       FCB    4+ON
       FCB    1+ON2
       FCB    1
       FCB    1+ON2
       FCB    3+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    8        8
       FCB    1+ON2
       FCB    3+ON
       FCB    1+ON2
       FCB    2
       FCB    1+ON2
       FCB    3+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    8        9
       FCB    1+ON2
       FCB    3+ON
       FCB    4+ON2
       FCB    3+ON
       FCB    5+ON2
       FCB    ENDCOL

       FCB    8        10
       FCB    1+ON2
       FCB    14+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    REPEAT+3  11,12,13

       FCB    8        14
       FCB    8+ON2
       FCB    3+ON
       FCB    5+ON2
       FCB    ENDCOL

       FCB    15       15
       FCB    1+ON2
       FCB    3+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    REPEAT+3       16,17,18

       FCB    14       19
       FCB    1+ON2 
       FCB    4+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    REPEAT+4       20,21,22,23

       FCB    15             24
       FCB    1+ON2
       FCB    3+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    16             25
       FCB    1+ON2
       FCB    2+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    17       26
       FCB    1+ON2
       FCB    1+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    18       27
       FCB    2+ON2
       FCB    ENDCOL

       FCB    ENDLET

L2084  FDB    L2,L0,LZER              
       FDB    L8,L4
LZER   FDB    0

BILL   FCB    10       1          BALL IS COLOR 1 (HIGH)
       FCB    7+ON                W    IS COLOR 2 (LOW)
       FCB    ENDCOL

       FCB    8        2
       FCB    11+ON
       FCB    ENDCOL

       FCB    6        3
       FCB    15+ON
       FCB    ENDCOL

       FCB    5        4
       FCB    17+ON
       FCB    ENDCOL

       FCB    4        5
       FCB    4+ON
       FCB    2+ON2
       FCB    13+ON
       FCB    ENDCOL

       FCB    3        6
       FCB    7+ON
       FCB    2+ON2
       FCB    12+ON
       FCB    ENDCOL

       FCB    2        7
       FCB    9+ON
       FCB    4+ON2
       FCB    10+ON
       FCB    ENDCOL

       FCB    2        8
       FCB    10+ON
       FCB    5+ON2
       FCB    8+ON
       FCB    ENDCOL

       FCB    1        9
       FCB    12+ON
       FCB    7+ON2
       FCB    6+ON
       FCB    ENDCOL

       FCB    1        10
       FCB    13+ON
       FCB    8+ON2
       FCB    4+ON
       FCB    ENDCOL

       FCB    13+ON    11
       FCB    7+ON2
       FCB    7+ON
       FCB    ENDCOL

       FCB    8+ON     12
       FCB    2+ON2
       FCB    2+ON
       FCB    6+ON2
       FCB    9+ON
       FCB    ENDCOL

       FCB    9+ON     13
       FCB    7+ON2
       FCB    11+ON
       FCB    ENDCOL

       FCB    10+ON
       FCB    4+ON2    14
       FCB    13+ON
       FCB    ENDCOL

       FCB    9+ON     13
       FCB    7+ON2
       FCB    11+ON
       FCB    ENDCOL

       FCB    8+ON     12
       FCB    2+ON2
       FCB    2+ON
       FCB    6+ON2
       FCB    9+ON
       FCB    ENDCOL

       FCB    13+ON    11
       FCB    7+ON2
       FCB    7+ON
       FCB    ENDCOL

       FCB    1        10
       FCB    13+ON
       FCB    8+ON2
       FCB    4+ON
       FCB    ENDCOL

       FCB    1        9
       FCB    12+ON
       FCB    7+ON2
       FCB    6+ON
       FCB    ENDCOL

       FCB    2        8
       FCB    10+ON
       FCB    5+ON2
       FCB    8+ON
       FCB    ENDCOL

       FCB    2        7
       FCB    9+ON
       FCB    4+ON2
       FCB    10+ON
       FCB    ENDCOL

       FCB    3        6
       FCB    7+ON
       FCB    2+ON2
       FCB    12+ON
       FCB    ENDCOL

       FCB    4        5
       FCB    4+ON
       FCB    2+ON2
       FCB    13+ON
       FCB    ENDCOL

       FCB    5        4
       FCB    17+ON
       FCB    ENDCOL

       FCB    6        3
       FCB    15+ON
       FCB    ENDCOL

       FCB    8        2
       FCB    11+ON
       FCB    ENDCOL

       FCB    10       1          BALL IS COLOR 1 (HIGH)
       FCB    7+ON                W    IS COLOR 2 (LOW)
       FCB    ENDCOL

       FCB    ENDLET

*      LETOUT - OUTPUT A LETTER STORED IN LOGO FORMAT
*
*      X = UPPER LEFT TO OUTPUT, RETURNS UPPER LEFT COLUMN FOLLOWING LETTER
*      U = POINTER TO LETTER DATA
*      Y = ROUTINE TO UPDATE COLOR
*      A = COLOR 1, COLOR 2  INITIALLY
*      B = INITIAL FLAVOR,  RETURNS FLAVOR JUST FOLLOWING LETTER

LETOUT PSHS   A,U,Y
       ANDA   #$F0
       PSHS   A
       LSRA
       LSRA
       LSRA
       LSRA
       ORAA   0,S+           COLOR 1 FORMED
       STA    LOGCOL
       LDA    ,S
       ANDA   #$0F
       PSHS   A
       ASLA
       ASLA
       ASLA
       ASLA
       ORAA   ,S+
       STA    LOGCL2         COLOR 2 
CAGAIN BSR    DOCOL          DO A COLUMN
       BCC    CAGAIN         NOT END OF LETTER...DO ANOTHER
       PULS   A,U,Y,PC       AND RETURN

DOCOL  LDA    ,U             CHECK FIRST BYTE OF THE COLUMN
       COMA
       BITA   #$C0           REPEATER??
       BEQ    DOCOL0         YEP...NO COLUMN SAVE FOR THIS START
       STU    LOGCST         SAVE COLUMN START
DOCOL0 STX    LOGX           SAVE THE X FOR COLUMN END
DOCOL2 LDA    ,U+            GET A DATA BYTE
       BPL    DOCOL1         ORDINARY BOREDOM
       BITA   #$20           END OF LETTER??
       BEQ    DOCOL3         NOPE
       SEC
DUMRTS RTS                   INDICATE END OF LETTER
DOCOL3 BITA   #$10           END OF COLUMN??
       BEQ    DOCOL4         NOPE

       BSR    CEND           DO TERMINAL TYPE OF STUFF
       CLC
       RTS                   AND RETURN

DOCOL4 BITA   #$40           MUST BE REPEAT TIME
       BEQ    DOCOL2         INVALID...CONTINUE
       ANDA   #$F            GET COUNT
       STA    LOGCNT         SAVE THE COUNT

       STU    LOGSAV         SAVE THE POINTER
DOCOL5 LDU    LOGCST         GET START OF COLUMN POINTER
       BSR    DOCOL0         CURSE AGAIN        
       DEC    LOGCNT         1 LESS
       BNE    DOCOL5         AND DO IT AGAIN

       LDU    LOGSAV         GET OUR POINTER BACK PLEASE
       CLC                   CLEAR OUT THE CARRY GUY
       RTS

DOCOL1 BITA   #$60           COLOR??
       BNE    DCOL6          YEP

       LEAX   A,X            ADD COUNT TO X IS ALL WE NEED TO DO
       BRA    DOCOL2         AND GET NEXT BYTE

DCOL6  PSHS   CC,A           SAVE
       ANDA   #$1F           REMOVE GARBAGE BITS 
       SEI
       STX    DEST           PLACE
       LEAX   A,X            FORM NEW POINTER
       EORA   #4             THROW GREG A BONE
       STA    HORIZ+1        HEIGHT
       LDA    #5
       STA    HORIZ          1 WIDE
       STD    ORIG           CONSTANT GARBAGE
       LDA    1,S            GET DIRECTIVE BYTE BACK
       BITA   #$40
       BEQ    COL2U          USE COLOR 2
       LDA    LOGCOL         GET COLOR1
       BRA    COL3U
COL2U  LDA    LOGCL2         USE COLOR 2
COL3U  STA    CONST
       LDA    #$12           CONSTANT STUFF ETC, BUT....
       TSTB                  WHICH FLAVOR??
       BEQ    DCOL7          EVEN FLAVOR
       ORAA   #$80           ODD FLAVOR...SUPRESS HIGH
       BRA    DCOL8
DCOL7  ORAA   #$40           EVEN FLAVOR...SUPRESS LOW
DCOL8  STA    CONTRL         DONE!
       PULS   CC,A
       BRA    DOCOL2         GET THE NEXT BYTE

CEND   LDX    LOGX           RESTORE START X
       TSTB                  EVEN FLAVOR??
       BEQ    CEND1          YEP
       LDB    #$FF           MAKE NON-ZERO FLAVOR FF IN CASE
       LEAX   $100,X         MOVE OVER 1
CEND1  COMB                  CHANGE FLAVOR
       RTS                   AND RETURN
