 STTL HORIZ EXPLODES
 
*
*INIT DMASIZ EXPLOSION DATA STRUCTURE
*
HXINV  PSHS   X,D    
       LDX    #EX      FIRST
       STX    EXFREE
EXIN0  LEAX   EXSIZE,X FIND NEXT
       STX    -EXSIZE,X  STORE
       CMPX   #EXEND-EXSIZE       BLO    EXIN0
EXIN1  LDD    #0
       STD    ,X
       STD    EXPTR
       STD    APPTR
       PULS   X,D,PC
*
*GET A DATA BLOCK IN U  FOR EXPLOSIONS
*
GETBLK PSHS   X
       LDU    EXFREE
       BEQ    NOBLK          NONE
       LDX    ,U             GET NEXT
       STX    EXFREE
       LDX    EXPTR          GET LIST
       STX    ,U             APPEND US
       STU    EXPTR
       CLC
       PULS   X,PC           RETURN IN U
NOBLK  SEC
       PULS   X,PC    
*
*GET AN APPEAR BLOCK
*
GETAP  PSHS   X
       LDU    EXFREE
       BEQ    NOBLK
       LDX    ,U
       STX    EXFREE
       LDX    APPTR
       STX    ,U
       STU    APPTR
       CLC
       PULS   X,PC
*
*START A DMASIZ APPEAR X=OBJECT
*
HAPSTV PSHS   X,U,D,Y    
       JSR    GETAP         GET A BLOCK TO USE
       BCS    APBYE
       LDD    OBJX,X
       LDX    OPICT,X
       STD    UL,U          STORE UPPER LEFT (FIRST FRAME)
       LDB    CENTMP
       STB    XCENT,U
       SUBB   UL,U          SUBTRACT X UPPER LEFT
       BCS    NWCEAP        NO GOOD
       CMPB   0,X           COMPARE 
       BLO    APNTOK
NWCEAP LDB    ,X            GET WIDTH
       STB    XOF,U         SAVE OFFSET
       ADDB   UL,U          ADD TOP         
       STB    XCENT,U       THIS IS 
       BRA    CCONTA
APNTOK ASLB                 DOUBLE CENTER FOR BYTE ADJUSTMENT
       STB    XOF,U
CCONTA LDD    #$4646         SUPRESS LOW GUYS, SLOW, ZERO SUPRESS, SER-BLK
       STD    DMACNT,U
       LDD    ,X            FETCH H,W
       STD    WH,U          SAVE
       LDA    #1            ONLY ERASE 1 ROW FIRST TIME THROUGH
       STA    XHITE,U       HITE FOR FIRST ERASE
       EORA   #$4
       EORB   #$4
       STD    DMAWH,U        FORMED
       LDX    2,X            GET DATA POINTER
       STX    PICPTR,U       SAVE OBJECT DESC
       LDD    #$1000         1ST FRAME SIZE
       STD    XSIZER,U

DMOVE  LEAX   DATA-1,U   LET X POINT AT DATA STORAGE AREA
       STX    TEMP1    SAVE POINTER
       LDY    PICPTR,U U POINTS AT DATA
       LDB    WH+1,U   ITERATE OVER THE HEIGHT
       STB    TEMP2+1
HSET2  LDX    TEMP1    GET SAVED POINTER
       LEAX   1,X      MOVE TO NEXT LOCATION
       STX    TEMP1    AND SAVE
       LDA    WH,U     GET WIDTH
       STA    TEMP2    SAVE AS COUNTER
HSET1  LDA    ,Y+      GET A BYTE                6
       STA    ,X       STORE                     5
       ABX             ADD HEIGHT TO GET TO NEXT 4
       ASLA                                      2
       ASLA                                      2
       ASLA                                      2
       ASLA                                      2
       STA    ,X                                 5
       ABX                                       4
       DEC    TEMP2                              6
       BNE    HSET1                              3    
       DEC    TEMP2+1  1 LESS ROW    
       BNE    HSET2
APBYE  CLC
       PULS   X,D,U,PC,Y
*
*START A DMASIZ EXPLOSION
*X=OBJ
HEXSTV PSHS   X,U,D,Y   
       JSR    DMAOFF
       JSR    GETBLK        GET A BLOCK TO USE
       BCS    APBYE         DONE...NO BLOCKS
       LDD    OBJX,X
       LDX    OPICT,X
       STD    UL,U          STORE UPPER LEFT (FIRST FRAME)
       LDB    CENTMP
       STB    XCENT,U
       SUBB   UL,U          SUBTRACT X UPPER LEFT
       BCS    NWCENT        NO GOOD
       CMPB   0,X           COMPARE 
       BLO    CENTOK
NWCENT LDB    ,X            GET WIDTH
       STB    XOF,U         SAVE OFFSET
       ADDB   UL,U          ADD TOP         
       STB    XCENT,U       THIS IS 
       BRA    CCONT 
CENTOK ASLB                 DOUBLE CENTER FOR BYTE ADJUSTMENT
       STB    XOF,U
CCONT  LDD    #$4646         SUPRESS UPPER, SLOW, SUPRESS, SER TO BLOCK
       STD    DMACNT,U       THIS IS DMA CONTROL BYTE
       LDD    ,X            FETCH H,W
       STD    WH,U          SAVE
       ASLA                  DOUBLE WIDTH FOR HEIGHT TO USE
       STA    XHITE,U       HITE FOR FIRST ERASE
       LDA    #1            HEIGHT IS 1 FOR DMA 
       EORA   #$4
       EORB   #$4
       STD    DMAWH,U        FORMED
       LDX    2,X            GET DATA POINTER
       STX    PICPTR,U       SAVE OBJECT DESC
       LDD    #$0            1ST FRAME MIN
       STD    XSIZER,U
       LDA    #$10           NUMBER OF FRAMES LEFT
       STA    FRAMES,U
       BRA    DMOVE

*      THIS CODE DMA'S THE DMASIZONTAL LINES
*      A JUMP INTO HERE IS MADE BASED ON NUMBER OF
*      VERTICAL LINES IN OBJECT

BSLOOP STA    ,Y           Y FOR CURRENT TRANSFER
       STX    DMAORG           DATA POINTER
       STU    DMACTL       DO THE LINE
       ABX                   NEXT DMAORG
       ADDA   XSIZE          ADD SPACING
BSSIZE EQU    *-BSLOOP

       STA    ,Y
       STX    DMAORG
       STU    DMACTL
       ABX
       ADDA   XSIZE

       STA    ,Y
       STX    DMAORG
       STU    DMACTL
       ABX
       ADDA   XSIZE

       STA    ,Y
       STX    DMAORG
       STU    DMACTL
       ABX
       ADDA   XSIZE

       STA    ,Y
       STX    DMAORG
       STU    DMACTL
       ABX
       ADDA   XSIZE

       STA    ,Y
       STX    DMAORG
       STU    DMACTL
       ABX
       ADDA   XSIZE

       STA    ,Y
       STX    DMAORG
       STU    DMACTL
       ABX
       ADDA   XSIZE

       STA    ,Y
       STX    DMAORG
       STU    DMACTL
       ABX
       ADDA   XSIZE

       STA    ,Y
       STX    DMAORG
       STU    DMACTL
       ABX
       ADDA   XSIZE

       STA    ,Y
       STX    DMAORG
       STU    DMACTL
       ABX
       ADDA   XSIZE

       STA    ,Y
       STX    DMAORG
       STU    DMACTL
       ABX
       ADDA   XSIZE

       STA    ,Y
       STX    DMAORG
       STU    DMACTL
       ABX
       ADDA   XSIZE

       STA    ,Y
       STX    DMAORG
       STU    DMACTL
       ABX
       ADDA   XSIZE

       STA    ,Y
       STX    DMAORG
       STU    DMACTL
       ABX
       ADDA   XSIZE

       STA    ,Y
       STX    DMAORG
       STU    DMACTL
       ABX
       ADDA   XSIZE

       STA    ,Y
       STX    DMAORG
       STU    DMACTL
       CLI
       PULS   Y,PC

ERLOOP STA    ,U         PUT HEIGHT TO ERASE DOWN
       STB    ,X         ERASE PLEASE
       ADDA   XSIZE
ERSIZ  EQU    *-ERLOOP

       STA    ,U
       STB    ,X
       ADDA   XSIZE

       STA    ,U
       STB    ,X
       ADDA   XSIZE

       STA    ,U
       STB    ,X
       ADDA   XSIZE

       STA    ,U
       STB    ,X
       ADDA   XSIZE

       STA    ,U
       STB    ,X
       ADDA   XSIZE

       STA    ,U
       STB    ,X
       ADDA   XSIZE

       STA    ,U
       STB    ,X
       ADDA   XSIZE

       STA    ,U
       STB    ,X
       ADDA   XSIZE

       STA    ,U
       STB    ,X
       ADDA   XSIZE

       STA    ,U
       STB    ,X
       ADDA   XSIZE

       STA    ,U
       STB    ,X
       ADDA   XSIZE

       STA    ,U
       STB    ,X
       ADDA   XSIZE

       STA    ,U
       STB    ,X
       ADDA   XSIZE

       STA    ,U
       STB    ,X
       ADDA   XSIZE

       STA    ,U
       STB    ,X
       CLI
       RTS
*
*ERASE AN EXPLOSION
*Y=DATA BLOCK
*
ERASE  LDB    XHITE,Y        ACTUALLY PLOTTED
       SUBB   #16            MAX OF 16
       NEGB
       LDA    #ERSIZ
       MUL
       LDX    #ERLOOP        START OF LOOP
       ABX
       PSHS   X              PLACE TO JUMP

       LDA    XSIZER,Y
       STA    XSIZE

       LDD    UL,Y         GET UPPER LEFT
       SEI
       STB    DMADES+1         HIGH HALF OF SCREEN STUFF

       LDB    DMACNT,Y       SUPRESS HIGH, DMACONANT, SLOW, BLOCK DMADES
       ORAB   #$10           ERASE
       ANDB   #$F7           NO SUPRESS

       LDU    #0
       STU    DMACON          CONSTANT, AND HIGH HALF OF DMAORG
       STB    DMAORG+1         LOW HALF OF ORIG

       LDU    DMAWH,Y
       STU    DMASIZ

       LDU    #DMADES
       LDX    #DMACTL

       RTS
*
*KILL AN EXPLOSION,APPEAR
*Y=DATA BLOCK
*
KILEXP LDU    #EXPTR         LETS WALK LIST TO FIND HIM
KILEX2 CMPY   ,U             ARE WE SET
       BEQ    KILEX1         YEP
       LDU    ,U             KEEP WALKING
       BNE    KILEX2         STILL STUFF THERE
       SEI
       BRA    *              ERROR TRAP
KILEX1 LDD    ,Y             GET GUY WE POINT AT
       STD    ,U             POINT AROUND US
       LDD    EXFREE
       STD    ,Y             APPEND FREE LIST
       STY    EXFREE
       LEAY   ,U             RETURN GUY BEFORE US TO WALK FROM
       RTS
*
*WRITE APPEARS
*Y=BLOCK
AWRITE LDD XSIZER,Y
 SUBD #$80
 CMPA XSIZER,Y ANY CHANGE?
 BNE AWRIT0 YIP.
 STB XSIZER+1,Y
 RTS
AWRIT0 JSR ERASE ERASE THE COCK-SUCKER
 LDA STATUS SCROLL?
 BNE AWRIT1
 LDB PX+1 SCROLL PLAYER APPEAR
 STB UL+1,Y
 LDB XOF,Y
 LSRB
 ADDB PX
 STB XCENT,Y
AWRIT1 LDA    WH,Y           GET WIDTH
       ASLA                  DOUBLE FOR PIXEL WIDTH
       DECA                  NO NEED TO DO END COLUMN
       STA    WIDE

       LDD    XSIZER,Y       GET SIZE  
       SUBD   #$080          SMALLEN IT
       TSTA     
       BHI   APGO
       LDU    #APPTR
       JMP    KILEX2
*
*WRITE EXPLOSION
*Y=DATA BLOCK
*
WRITE  DEC    FRAMES,Y       ONE LESS FRAME
       BEQ    KILEXP         DONE

       LDA    WH,Y   GET WIDTH 
       ASLA          DOUBLE FOR PIXEL WIDTH
       DECA            NO NEED TO DO ZEROS ON LEFT COLUMN
       STA    WIDE

       LDD    XSIZER,Y
       ADDD   #$100 
APGO   STD    XSIZER,Y
       STA    XSIZE

       LSRA            1/2 AND SAVE
       LDB    XOF,Y    GET OFFSET
       BNE    APGO1    OBSCURE BUG CHECK (NO 1/2 UNIT FROM TOP)
       CLRA
APGO1  STA    TEMP2

       LEAX   DATA,Y       POINT AT THE DATA

       LDA    XSIZE
       LDB    XOF,Y          GET OFFSET
       MUL                   DISTANCE UP FROM CENTER


       STD    TEMP1          SAVE
       LDB    XCENT,Y        GET CENTER
       CLRA
       SUBD   TEMP1          LETS BE GENERAL PURPOSE
       ADDB   TEMP2          ADD 1/2 SIZE
       ADCA   #0             AND PROPAGATE
       BNE    CHK2
       CMPB   #XMIN
       BHI    CHK3     WERE OK
CHK2   DEC    WIDE
       ADDB   XSIZE
       ADCA   #0
       BNE    CHK2
CHK1   CMPB   #XMIN
       BLS    CHK2
       STB    UL,Y         STORE IN PLACE
       LDD    WH,Y     FIND OFFSET
       SUBA   WIDE
       MUL
       ABX
       BRA    CHK3A
CHK3   STB    UL,Y
CHK3A  LDA    WIDE
       DECA
       LDB    XSIZE          FIND FILL DEFLECTION
       MUL
       ADDB   UL,Y           ADD FIRST POINT
       ADCA   #0             FIND FULL DEFLECTION
       BEQ    CHK4           AOK
CHK5   DEC    WIDE           1 LESS
       SUBB   XSIZE          1 LESS DOWNWARD
       SBCA   #0
       BNE    CHK5
CHK4   CMPB   #XMAX
       BHS    CHK5

*      FIND "LOOP" ENTRY

       LDA    WIDE           GET HEIGHT
       LBEQ   KILEXP
       STA    XHITE,Y        SAVE FOR ERASE
       SUBA   #16            16 = 0 OFFSET, 0 = 16 OFFSET
       NEGA
       LDB    #BSSIZE        SIZE OF ITERATION
       MUL
       ADDD   #BSLOOP        BEGINNING
       PSHS   D,Y

       LDU    DMACNT,Y       SUPRESS HIGH NIB, SLOW, SUPRESS, SER TO BLOCK
       LDD    DMAWH,Y        GET WH

       SEI                   NO INTS BABY
       STD    DMASIZ
       LDA    UL+1,Y         GET Y OF DMADESINATION 
       STA    DMADES+1

       LDB    WH+1,Y      GET INITIAL Y HEIGHT (AMOUNT TO ADD TO SOURCE PTR)
       LDA    UL,Y        GET X WIDTH AT START   

       LDY    #DMADES     PLACE TO STOE STUFF

       RTS                   JUMP INTO IT
*EXPLOSION APPEAR UPDATE
*
HXUPDV LDY EXPTR
 BEQ APUPD
HXUPDL JSR ERASE
 JSR WRITE
 LDY ,Y
 BNE HXUPDL
APUPD LDY APPTR
 BEQ APUPDX
APUPDL JSR AWRITE
 LDY ,Y
 BNE APUPDL
APUPDX RTS

REPEAT EQU    $C0
ENDLET EQU    $A0
ENDCOL EQU    $90

ON     EQU    $40      INDICATE COLOR IS ON
ON2    EQU    $20      COLOR 2 ON

L2V    FCB    23       1
       FCB    1+ON2    COLOR 2
       FCB    ENDCOL

       FCB    13       2
       FCB    1+ON2
       FCB    8
       FCB    2+ON2
       FCB    ENDCOL

       FCB    12       3
       FCB    2+ON2
       FCB    7
       FCB    1+ON2
       FCB    1+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    11       4
       FCB    1+ON2
       FCB    1+ON
       FCB    1+ON2
       FCB    6
       FCB    1+ON2
       FCB    2+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    10       5
       FCB    1+ON2
       FCB    2+ON
       FCB    1+ON2
       FCB    5
       FCB    1+ON2
       FCB    3+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    9        6
       FCB    ON2+1
       FCB    3+ON
       FCB    1+ON2
       FCB    4
       FCB    1+ON2
       FCB    4+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    8        7
       FCB    1+ON2
       FCB    4+ON
       FCB    1+ON2
       FCB    3
       FCB    1+ON2
       FCB    5+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    8        8
       FCB    1+ON2
       FCB    4+ON
       FCB    1+ON2
       FCB    2
       FCB    1+ON2
       FCB    6+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    8        9
       FCB    1+ON2
       FCB    4+ON
       FCB    1+ON2
       FCB    1
       FCB    1+ON2
       FCB    3+ON
       FCB    1+ON2
       FCB    3+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    8        10
       FCB    1+ON2
       FCB    3+ON
       FCB    1+ON2
       FCB    1
       FCB    1+ON2
       FCB    3+ON
       FCB    2+ON2
       FCB    3+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    8        11
       FCB    1+ON2
       FCB    3+ON
       FCB    1+ON2
       FCB    1
       FCB    1+ON2
       FCB    2+ON
       FCB    1+ON2
       FCB    1
       FCB    1+ON2
       FCB    3+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    REPEAT+4  12,13,14,15

       FCB    8        16
       FCB    1+ON2
       FCB    3+ON
       FCB    3+ON2
       FCB    2+ON
       FCB    1+ON2
       FCB    1
       FCB    1+ON2
       FCB    3+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    8        17
       FCB    1+ON2
       FCB    8+ON
       FCB    1+ON2
       FCB    1
       FCB    1+ON2
       FCB    3+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    REPEAT+2 18,19

       FCB    8        20
       FCB    10+ON2
       FCB    1
       FCB    1+ON2
       FCB    3+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    18       21
       FCB    1+ON2
       FCB    4+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    REPEAT+3  22,23,24

       FCB    ENDLET              END OF THE 2 (ROUGHLY)

L0V    FCB    11       1
       FCB    8+ON2
       FCB    4+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    10       2
       FCB    1+ON2
       FCB    10+ON
       FCB    3+ON2
       FCB    ENDCOL

       FCB    9        3
       FCB    1+ON2
       FCB    11+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    8        4
       FCB    1+ON2
       FCB    13+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    7        5
       FCB    1+ON2
       FCB    4+ON
       FCB    7+ON2
       FCB    4+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    6        6
       FCB    1+ON2
       FCB    4+ON
       FCB    1+ON2
       FCB    7
       FCB    1+ON2
       FCB    4+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    5        7
       FCB    1+ON2
       FCB    4+ON
       FCB    1+ON2
       FCB    9
       FCB    1+ON2
       FCB    4+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    4        8
       FCB    1+ON2
       FCB    4+ON
       FCB    13+ON2
       FCB    4+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    3        9
       FCB    1+ON2
       FCB    23+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    2        10
       FCB    1+ON2
       FCB    25+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    1        11
       FCB    1+ON2
       FCB    27+ON
       FCB    1+ON2
       FCB    ENDCOL

       FCB    31+ON2   12 
       FCB    ENDCOL

       FCB    ENDLET
