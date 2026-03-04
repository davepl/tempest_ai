 OPT NOL
 LIB RRF
 OPT LIS
 STTL DIAGONAL EXPLOSIONS
*
*      EXPLODING OBJECT DATA STRUCTURE
*
       ORG    DXTAB          PLACE FOR DATA

EX     EQU    *
NEXTZ  RMB    2  LIST POINTER
PICPTZ RMB    2  POINTER TO DATA   
XCENTZ RMB    1  SCREEN X CENTER (LEFT X OF CENTER LINE)
YCENTZ RMB    1  SCREEN Y CENTER
YOFZ   RMB    1  OFFSET IN Y TO OBJECT CENTER  (PIXELS) (0-HEIGHT)
XSIZEZ RMB    1  X SIZE USED LAST (THIS) FRAME
YSIZEZ RMB    2  CURRENT Y POINT SPACING                          
FRAMEZ RMB    1  FRAMES TO RUN IN EFFECT
ULZ    RMB    2  UPPER LEFT
WHZ    RMB    2  WIDTH,HEIGHT
DMAWHZ RMB    2  DMA HEIGHT WIDTH
YHITEZ RMB    1
SLOPEZ RMB    1  SLOPE OF DIAGONAL LINE WE ARE EXPLODING (SIGN BIT)
DATAZ  RMB    16*2    LIST OF ROW DATA POINTERS (FOR SPEED MAN)
ESIZEZ EQU    *

NEXT   EQU    NEXTZ-EX
PICPTR EQU    PICPTZ-EX
XCENT  EQU    XCENTZ-EX
YCENT  EQU    YCENTZ-EX
YOF    EQU    YOFZ-EX
XSIZER EQU    XSIZEZ-EX
YSIZER EQU    YSIZEZ-EX
FRAMES EQU    FRAMEZ-EX
UL     EQU    ULZ-EX
WH     EQU    WHZ-EX
DMAWH  EQU    DMAWHZ-EX
YHITE  EQU    YHITEZ-EX
SLOPE  EQU    SLOPEZ-EX
DATA   EQU    DATAZ-EX
EXSIZE EQU    ESIZEZ-EX

       RMB    ((10-1)*EXSIZE)    NUMBER OF BLOCKS

EXEND  EQU    *

       ORG    DXRAM      (BASE PAGE)

YYCNT   RMB    1        WHEN CHOPPING UP X, SAVE TO UPDATE ULY
YOFF   RMB    1        ACTUAL PIXEL OFFSET TO CENTER (USED)
XSIZE  RMB    1        DISTANCE BETWEEN X POINTS THIS FRAME
YSIZE  RMB    1        DISTANCE BETWEEN Y POINTS THIS FRAME
EXPTR  RMB    2
APPTR  RMB    2        APPEAR POINTER
EXFREE RMB    2
HITE   RMB    1
TEMP1  RMB    2
TEMP2  RMB    1
*
*VECTORS
*
       ORG    RDXORG
 JMP EXINV
 JMP EXSTZ
 JMP APSTZ
 JMP DXUPDV
*
*INIT DATA STRUCT
*

EXINV  PSHS   X,Y    
       LDX    #EX      FIRST
       STX    EXFREE
EXIN0  LEAY   EXSIZE,X FIND NEXT
       STY    ,X       STORE
       CMPY   #EXEND   DONE??        
       BHS    EXIN1
       LEAX   ,Y       MAKE US NEXT
       BRA    EXIN0
EXIN1  LDY    #0
       STY    ,X
       STY    EXPTR
       STY    APPTR
       PULS   X,Y,PC

GETBLK PSHS   Y
       LDU    EXFREE
       BEQ    NOBLK          NONE
       LDY    ,U             GET NEXT
       STY    EXFREE
       LDY    EXPTR          GET LIST
       STY    ,U             APPEND US
       STU    EXPTR
       CLC
       PULS   Y,PC           RETURN IN U
NOBLK  SEC
       PULS   Y,PC    

GETAP  PSHS   Y
       LDU    EXFREE
       BEQ    NOBLK
       LDY    ,U
       STY    EXFREE
       LDY    APPTR
       STY    ,U
       STU    APPTR
       CLC
       PULS   Y,PC
*
*START AN APPEAR
*X=OBJ,A=SLOPE
APSTZ  PSHS   X,Y,D,U
       JSR    GETAP    GET AN APPEAR BLOCK
       BCS    APBY     NONE
       STA    SLOPE,U  SAVE SLOPE
       LDD    OBJX,X
       LDX    OPICT,X
       STD    UL,U     START AS UPPER LEFT
       STA    XCENT,U  SAVE X CENTER
       LDB    CENTMP+1
       STB    YCENT,U
       SUBB   UL+1,U   SUBTRACT Y UPPER LEFT
       BCS    NWCEAP   NO GOOD
       CMPB   1,X      COMPARE 
       BLO    APNTOK
NWCEAP LDB    ,X       GET WIDTH
       LSRB            1/2 Y
       STB    YOF,U    SAVE OFFSET
       ADDB   UL+1,U   ADD TOP         
       STB    YCENT,U  THIS IS 
       BRA    CCOAP 
APNTOK STB    YOF,U
CCOAP  LDD    ,X       FETCH H,W
       STD    WH,U     SAVE
       LDB    #1       HEIGHT IS 1 FOR DMA 
       STB    YHITE,U  SMALL HEIGHT FOR FIRST (CRAZY) ERASE
       EORA   #$4
       EORB   #$4
       STD    DMAWH,U  FORMED
       LDX    2,X      GET DATA POINTER
       STX    PICPTR,U SAVE OBJECT DESC
       LDD    #$1000   START LARGE FOR APPEAR
       STD    YSIZER,U
       CLR    XSIZER,U ERASE NORMAL OBJECT FIRST FRAME
       BSR    DATALD   LOAD THE ROW DATA PTRS
APBY   PULS   X,D,U,PC,Y   AND RETURN TO CALLER

DATALD LDX    PICPTR,U GET POINTER TO DATA
       LDB    WH,U     WIDTH IN B                   
       LDA    WH+1,U   HEIGHT IN A
       LEAU   DATA,U   POINT AT THE DATA
DATLD1 STX    ,U++     STORE POINTER TO CURRENT ROW
       ABX             ADDING WIDTH FORMS POINTER TO NEXT ROW
       DECA            1 LESS ROW (OFF HEIGHT)
       BNE    DATLD1   DONE?
       RTS             YEP


*
*START AN EXPLOSION
*X=OBJ,A=SLOPE
EXSTZ  PSHS   X,U,D,Y
       JSR    GETBLK   GET A BLOCK TO USE
       BCS    EXBYE    DONE...NO BLOCKS
       STA    SLOPE,U
       LDD    OBJX,X
       LDX    OPICT,X
       STD    UL,U     STORE UPPER LEFT (FIRST FRAME)
       STA    XCENT,U  SAVE LEFT X
       LDB    CENTMP+1
       STB    YCENT,U
       SUBB   UL+1,U   SUBTRACT Y UPPER LEFT
       BCS    NWCENT   NO GOOD
       CMPB   1,X      COMPARE 
       BLO    CENTOK
NWCENT LDB    ,X       GET WIDTH
       LSRB            1/2 Y
       STB    YOF,U    SAVE OFFSET
       ADDB   UL+1,U   ADD TOP         
       STB    YCENT,U  THIS IS 
       BRA    CCONT 
CENTOK STB    YOF,U
CCONT  LDD    ,X       FETCH H,W
       STD    WH,U     SAVE
       STB    YHITE,U  HITE FOR FIRST ERASE
       LDB    #1       HEIGHT IS 1 FOR DMA 
       EORA   #$4
       EORB   #$4
       STD    DMAWH,U  FORMED
       LDX    2,X      GET DATA POINTER
       STX    PICPTR,U SAVE OBJECT DESC
       LDD    #$100    1 UNIT IS MIN
       STD    YSIZER,U
       CLR    XSIZER,U
       LDA    #$10    NUMBER OF BYTES TO ERASE, NUMBER OF FRAMES LEFT
       STA    FRAMES,U
       BSR    DATALD   LOAD THE ROW POINTERS
EXBYE  PULS   X,D,U,PC,Y   AND RETURN TO CALLER

*      THIS CODE DMA'S THE DMASIZONTAL LINES
*      A JUMP INTO HERE IS MADE BASED ON NUMBER OF
*      DMASIZONTAL LINES IN OBJECT

BSLOOP LDX    ,Y++           GET THE ROW POINTER
       STX    DMAORG           THIS IS ORIGIN
       STD    DMADES           D IS DEST
       STU    DMACTL         THIS IS THE CONTROL REGISTER
       ADDD   XSIZE          KICK POINTER TO NEXT ONE
BSSIZE EQU    *-BSLOOP

       LDX    ,Y++           GET THE ROW POINTER
       STX    DMAORG           THIS IS ORIGIN
       STD    DMADES           D IS DEST
       STU    DMACTL         THIS IS THE CONTROL REGISTER
       ADDD   XSIZE          KICK POINTER TO NEXT ONE

       LDX    ,Y++           GET THE ROW POINTER
       STX    DMAORG           THIS IS ORIGIN
       STD    DMADES           D IS DEST
       STU    DMACTL         THIS IS THE CONTROL REGISTER
       ADDD   XSIZE          KICK POINTER TO NEXT ONE

       LDX    ,Y++           GET THE ROW POINTER
       STX    DMAORG           THIS IS ORIGIN
       STD    DMADES           D IS DEST
       STU    DMACTL         THIS IS THE CONTROL REGISTER
       ADDD   XSIZE          KICK POINTER TO NEXT ONE

       LDX    ,Y++           GET THE ROW POINTER
       STX    DMAORG           THIS IS ORIGIN
       STD    DMADES           D IS DEST
       STU    DMACTL         THIS IS THE CONTROL REGISTER
       ADDD   XSIZE          KICK POINTER TO NEXT ONE

       LDX    ,Y++           GET THE ROW POINTER
       STX    DMAORG           THIS IS ORIGIN
       STD    DMADES           D IS DEST
       STU    DMACTL         THIS IS THE CONTROL REGISTER
       ADDD   XSIZE          KICK POINTER TO NEXT ONE

       LDX    ,Y++           GET THE ROW POINTER
       STX    DMAORG           THIS IS ORIGIN
       STD    DMADES           D IS DEST
       STU    DMACTL         THIS IS THE CONTROL REGISTER
       ADDD   XSIZE          KICK POINTER TO NEXT ONE

       LDX    ,Y++           GET THE ROW POINTER
       STX    DMAORG           THIS IS ORIGIN
       STD    DMADES           D IS DEST
       STU    DMACTL         THIS IS THE CONTROL REGISTER
       ADDD   XSIZE          KICK POINTER TO NEXT ONE

       LDX    ,Y++           GET THE ROW POINTER
       STX    DMAORG           THIS IS ORIGIN
       STD    DMADES           D IS DEST
       STU    DMACTL         THIS IS THE CONTROL REGISTER
       ADDD   XSIZE          KICK POINTER TO NEXT ONE

       LDX    ,Y++           GET THE ROW POINTER
       STX    DMAORG           THIS IS ORIGIN
       STD    DMADES           D IS DEST
       STU    DMACTL         THIS IS THE CONTROL REGISTER
       ADDD   XSIZE          KICK POINTER TO NEXT ONE

       LDX    ,Y++           GET THE ROW POINTER
       STX    DMAORG           THIS IS ORIGIN
       STD    DMADES           D IS DEST
       STU    DMACTL         THIS IS THE CONTROL REGISTER
       ADDD   XSIZE          KICK POINTER TO NEXT ONE

       LDX    ,Y++           GET THE ROW POINTER
       STX    DMAORG           THIS IS ORIGIN
       STD    DMADES           D IS DEST
       STU    DMACTL         THIS IS THE CONTROL REGISTER
       ADDD   XSIZE          KICK POINTER TO NEXT ONE

       LDX    ,Y++           GET THE ROW POINTER
       STX    DMAORG           THIS IS ORIGIN
       STD    DMADES           D IS DEST
       STU    DMACTL         THIS IS THE CONTROL REGISTER
       ADDD   XSIZE          KICK POINTER TO NEXT ONE

       LDX    ,Y++           GET THE ROW POINTER
       STX    DMAORG           THIS IS ORIGIN
       STD    DMADES           D IS DEST
       STU    DMACTL         THIS IS THE CONTROL REGISTER
       ADDD   XSIZE          KICK POINTER TO NEXT ONE

       LDX    ,Y++           GET THE ROW POINTER
       STX    DMAORG           THIS IS ORIGIN
       STD    DMADES           D IS DEST
       STU    DMACTL         THIS IS THE CONTROL REGISTER
       ADDD   XSIZE          KICK POINTER TO NEXT ONE

       LDX    ,Y++           GET THE ROW POINTER
       STX    DMAORG           THIS IS ORIGIN
       STD    DMADES           D IS DEST
       STU    DMACTL         THIS IS THE CONTROL REGISTER

       CLI
       PULS   Y,PC           RESTORE Y, RETURN

ERLOOP STD    DMADES           DEST TO ERASE
       STU    DMACTL-1         BYTE TO DO IT
       ADDD   XSIZE          ADD OFFSETS
ERSIZ  EQU    *-ERLOOP

       STD    DMADES
       STU    DMACTL-1
       ADDD   XSIZE

       STD    DMADES
       STU    DMACTL-1
       ADDD   XSIZE

       STD    DMADES
       STU    DMACTL-1
       ADDD   XSIZE

       STD    DMADES
       STU    DMACTL-1
       ADDD   XSIZE

       STD    DMADES
       STU    DMACTL-1
       ADDD   XSIZE

       STD    DMADES
       STU    DMACTL-1
       ADDD   XSIZE

       STD    DMADES
       STU    DMACTL-1
       ADDD   XSIZE

       STD    DMADES
       STU    DMACTL-1
       ADDD   XSIZE

       STD    DMADES
       STU    DMACTL-1
       ADDD   XSIZE

       STD    DMADES
       STU    DMACTL-1
       ADDD   XSIZE

       STD    DMADES
       STU    DMACTL-1
       ADDD   XSIZE

       STD    DMADES
       STU    DMACTL-1
       ADDD   XSIZE

       STD    DMADES
       STU    DMACTL-1
       ADDD   XSIZE

       STD    DMADES
       STU    DMACTL-1
       ADDD   XSIZE

       STD    DMADES
       STU    DMACTL-1
       ADDD   XSIZE

       CLI
       RTS                                        

ERASE  LDB    YHITE,Y        ACTUALLY PLOTTED
       SUBB   #16            MAX OF 16
       NEGB
       ASLB
       ASLB
       ASLB
       LDX    #ERLOOP
       ABX

       LDD    XSIZER,Y
       STD    XSIZE

       LDA    RWCTMP         GET RWBYTE
       LDB    #$12           ERASE

       TFR    D,U
       LDD    #0

       SEI

       STD    DMACON          CONSTANT, AND HIGH HALF OF DMAORG

       LDD    DMAWH,Y
       STD    DMASIZ

       LDD    UL,Y           DMADESINATION STARTS WITH UPPER LEFT

       JMP    ,X      

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
       RTS                   DONE!
*
*WRITE YOUR APPEAR!!!
*Y=EXPLOSION
AWRITE LDD YSIZER,Y
 SUBD #$100
 CMPA YSIZER,Y ANY SIZE CHANGE?
 BNE AWRIT0 YES
 STB YSIZER+1,Y NO DICE
 RTS
AWRIT0 JSR ERASE
 LDA STATUS
 BNE AWRIT1
 LDD PX SCROLL THE COCK SUCKER
 STA XCENT,Y
 ADDB YOF,Y
 STB YCENT,Y
AWRIT1 LDA    WH+1,Y   GET HEIGHT
       STA    HITE     PRELIMINARY ATTEMPT AT HEIGHT

       LDD    YSIZER,Y GET CURRENT SIZE
       SUBD   #$0100    TAKE AWAY SOME SIZE
       CMPA   #1
       BHI    APGO     DONE!
       LDU    #APPTR   FIND HE WHO POINTS AT HIM....NOW KILL IT
       BRA    KILEX2   NOW ITS JUST LIKE AN EXPLODE KILL 

WRITE  DEC    FRAMES,Y       ONE LESS FRAME
       BEQ    KILEXP         DONE

       LDA    WH+1,Y   GET HEIGHT
       STA    HITE

       LDD    YSIZER,Y
       ADDD   #$100
APGO   STA    YSIZE
       STD    YSIZER,Y

       LSRA            1/2 FOR X SIZE
       STA    TEMP2    ALSO SAVE TO ADD 1/2 SIZE UNIT FOR CENTER LINE
       LDB    SLOPE,Y  CHECK SLOPE
       BPL    APGG1    POSITIVE
       NEGA            NEGATIVE
APGG1  STA    XSIZER,Y AND SAVE
       STA    XSIZE    SCRATCH SOME SPEED

       LDB    YOF,Y    CHECK FOR ZERO OFFSET
       BNE    APGO1
       STB    TEMP2    IF DOWN FROM TOP....FIX TOP ONE (OBSCURE BUG)

APGO1  LDA    YSIZE
       LDB    YOF,Y          GET OFFSET
       STB    YOFF           SAVE FOR X CALCULATION
       MUL                   DISTANCE UP FROM CENTER

       STD    TEMP1          SAVE
       LDB    YCENT,Y        GET CENTER
       CLRA
       SUBD   TEMP1          LETS BE GENERAL PURPOSE
       ADDB   TEMP2          ADD 1/2 SIZE
       ADCA   #0             AND PROPAGATE
       BNE    CHK2           NO GOOD..TIME TO SCREW AROUND
*
*      DON'T WORRY EUGENE...I GOT THIS FIX THIS TIME
*
       CMPB   #YMIN
       BHI    CHK3           IN RANGE

CHK2   DEC    HITE           1 LESS
       DEC    YOFF
       ADDB   YSIZE          BRING BACK FROM NEGATIVE NOWHERE
       ADCA   #0
       BNE    CHK2           AND LOOP A WHILE

       CMPB   #YMIN          POSITIVE-VILLE...BIG ENOUGH 
       BLS    CHK2           NOPE

CHK3   STB    UL+1,Y         STORE IN PLACE
       LDA    YOFF           ACTUAL Y OFFSET USED
       LDB    XSIZE          WHICH SLOPE??
       BMI    XNEG           BUMMER

       MUL                   FIND DEFLECTION
       STD    TEMP1          WHERE HAVE I SEED CODE LIKE THIS
       LDB    XCENT,Y        GET X "CENTER"
       CLRA                  X CENTER HAS BEEN COMPUTED

       SUBD   TEMP1          FIND WHERE LEFTMOST X OCCURS
       TSTA                  NEGATIVE??
       BNE    CKK2A          YEP...SHIFT IT IN

       CMPB   #XMIN          IN RANGE??
       BHI    CKK3AA         RIGHT ON

CKK2A  CLR    YYCNT           COUNT TO UDPATE 1 STARTING POINT

CKK2B  DEC    HITE           CHOP SOME MORE AWAY
       INC    YYCNT           INCICATE 1 MORE
       ADDB   XSIZE          ADD THE X UNITS
       ADCA   #0
       BNE    CKK2B          AND ITERATE

       CMPB   #XMIN
       BLS    CKK2B          THANK YOU SIR MAY I HAVE ANOTHER

CKK3C  STB    UL,Y           THIS IS THE UPPER LEFT X

       LDB    YYCNT           GET THE YER
       LDA    YSIZE          GET Y SIZE
       MUL                   NUMBER OF Y UNITS FROM X CLIP

       ADDB   UL+1,Y
       STB    UL+1,Y

       BRA    CKK3B

XNEG   NEGB                  MAKE SIZE POSITIVE (NEG STATE BY CONTROL HERE)
       MUL                   NUMBER OF BYTES POS DISPLACEMENT
       ADDB   WH,Y           ADD IN THE WIDTH, (GUAR NO CARRY IF SIZE<32I
       ADDB   XCENT,Y        ADD CENTER TO FIND LAST PIXEL PLOTTED
       ADCA   #0             ALL RIGHT!

       BNE    XN2            NO GOOD...BIG CARRY 

       CMPB   #XMAX          IN RANGE??
       BLS    CKK3A          SAME IS OK SINCE THIS IS 1 BEYOND OB

XN2    CLR    YYCNT           WE NEED TO KNOW HOW MANY CLIPPED

XN22   DEC    HITE           1 MORE Y UNIT
       INC    YYCNT
       ADDB   XSIZE          ADD THE SIZE (REMEMBER, ITS NEGATIVE)
       ADCA   #$FF           ADD EXTENDED SIGN
       BNE    XN22           STILL TOO LARGE

       CMPB   #XMAX          DID WE MAKE IT??
       BHI    XN22           NOPE
       SUBB   WH,Y           REMOVE THE WIDTH TO GET BACK TO LEFT
       BRA    CKK3C          ADJUST Y FOR THE DAMAGES
*                            (CKK3C IS UP, EUGENE)

CKK3A  SUBB   WH,Y           GET BACK TO LEFT OF OB
CKK3AA STB    UL,Y           SAVE GOOD X

CKK3B  LEAX   DATA,Y         POINT AT FIRST ITEM 
       LDB    WH+1,Y         GET HEIGHT
       SUBB   HITE           FIND NUMBER WE WILL SKIP    
       ASLB                  POINTERS ARE 2 BYTES
       ABX                   WE NOW POINT AT FIRST ONE WE WILL USE

*
*      TIME FOR THE BUTT CLIP
*

       LDA    HITE           GET HEIGHT
       DECA
       LDB    YSIZE          FIND FILL DEFLECTION
       MUL
       ADDB   UL+1,Y         ADD FIRST POINT
       ADCA   #0             FIND FULL DEFLECTION
       BEQ    CHK4           AOK
CHK5   DEC    HITE           1 LESS
       BEQ    KLJMP          EXIT IMMEDIATELY ON TOTAL CLIP
       SUBB   YSIZE          1 LESS DOWNWARD
       SBCA   #0
       BNE    CHK5
CHK4   CMPB   #YMAX
       BHS    CHK5

       LDA    HITE           GET HITE BACK
       DECA                  FIRSTIES DON'T COUNT
       LDB    XSIZE          GET X SIZE
       BMI    XNT            NEGATIVE SLOPE

       MUL                   DEFLECTION OF LAST GUY
       ADDB   WH,Y           ADD WIDTH
       ADDB   UL,Y           ADD STARTING X
       ADCA   #0             PROP IT
       BEQ    NCK4           NOT HUGE
NCK5   DEC    HITE           TAKE 1 AWAY
       BEQ    KLJMP          TOTAL CLIP...BYE
       SUBB   XSIZE          TRY AND BRING DOWN TO SIZE
       SBCA   #0
       BNE    NCK5
NCK4   CMPB   #XMAX
       BHI    NCK5           EQUAL IS OK
       BRA    NCK6

XNT    NEGB                  MAKE 
       MUL
       STD    TEMP1          SAVE OFFSET

       LDB    UL,Y           GET UPPER LEFT
       CLRA

       SUBD   TEMP1          REMOVE OFFSET
       TSTA

       BEQ    NNCK4          NOT WAY TOO SMALL

NNCK5  DEC    HITE           1 LESS
       BEQ    KLJMP
       SUBB   XSIZE          ADDING NEG MOVES US POSITVE
       SBCA   #$FF           SUB SIGN EXT
       BNE    NNCK5          KEEP GOING

NNCK4  CMPB   #XMIN
       BLS    NNCK5          KEEP GOING

*      FIND "LOOP" ENTRY

NCK6   LDA    HITE           GET HEIGHT
       BNE    NOKILE
KLJMP  JMP    KILEXP         NO HEIGHT

NOKILE STA    YHITE,Y        SAVE FOR ERASE
       SUBA   #16            16 = 0 OFFSET, 0 = 16 OFFSET
       NEGA
       LDB    #BSSIZE        SIZE OF ITERATION
       MUL
       ADDD   #BSLOOP        BEGINNING
       PSHS   D,Y            SETUP STACK AS FOLLOWS:    

*      PTR     ->   D = JUMP ADDRESS
*                   Y = PUSHED TO SAVE FOR RTS
*                   <RETURN ADDRESS>

*                   PULS   Y,PC     AT END OF LOOP CAUSES PROPER RETURN

       LDU    #$0A0A         CONTROL REGISTER
       LDD    DMAWH,Y        GET WH

       SEI                   NO INTS BABY
       STD    DMASIZ
       LDD    UL,Y           GET DMADESINATION IN D 

       LEAY   ,X             MAKE Y POINT AT PROPER DATA GUY

       RTS                   RETURN, MY ASS!
*
*UPDATE YOUR DIAGONALS
*
DXUPDV LDY EXPTR
 BEQ APUPD
DXUPDL JSR ERASE
 JSR WRITE
 LDY ,Y
 BNE DXUPDL
APUPD LDY APPTR
 BEQ APUPDX
APUPDL JSR AWRITE
 LDY ,Y
 BNE APUPDL
APUPDX RTS
 END

