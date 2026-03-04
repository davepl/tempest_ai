FDRET  LDU    #RAMERJ      RAM ERROR RETURN
       BRA    FDROM0

FDROM  LDU    #DIG11Z
FDROM0 LDS    #HSTK
FDROM5 LDY    #FDROM6
       LDA    #1
       JMP    DELA1
FDROM6 LDA    PIA0 
       BITA   #$2
       BNE    FDROM5
FDROM1 LDY    #FDROM7
       LDA    #1
       JMP    DELA1
FDROM7 LDA    PIA0                                         
       BITA   #$2
       BEQ    FDROM1
FDROM2 LDY    #FDROM8
       LDA    #1  
       JMP    DELA1
FDROM8 LDA    PIA0
       BITA   #2
       BNE    FDROM2
       JMP    ,U    

AVWAIT PULS   D
       LDU    CRPROC
       STD    PD+6,U
AVWAIZ LDA    PIA0          SEE IF ADVANCE PRESSED
       BITA   #$2            WELL??
       BNE    AVWAT2 
       NAP    1,AVWAIZ       WELL, WAIT FOR IT.
AVWAT2 CLR    PCRAM
       JSR    SCRCLR
       BRA    AVWAT5
AVWAT4 LDA    PIA0
       BITA   #$2            WAIT FOR IT TO BE RELEASED
       BEQ    AVWAT3
AVWAT5 NAP    2,AVWAT4
AVWAT3 JMP    [PD+6,U]

AVCHK  PSHS   A
       LDA    PIA0
       BITA   #$2
       BEQ    AVCHK1
       SEC
       PULS   A,PC
AVCHK1 CLC
       PULS   A,PC

*
*    DISPLAY BOOKKEEPPING
*

BOOK   LDA   #$FF   
       STA    STATUS
       JSR    INITCM   COLOR MATRIX PROPER.
       BSR    AVCHK
       BCC    BOOK3
       BSR    AVWAIT   WAIT FOR HIM TO LET GO.
BOOK3  LDA    #BOOKM
       JSR    WRD7V    PRINT THE BOOKKEEPING MESSAGE
       LDU    #SLOT1
       LDA    #AUDMES  FIRST OF THE AUDIT MESSAGES 
BOOK0  PSHS   A        SAVE MESSAGE NUMBER
       JSR    WRD7V    PRINT IT.
       EXG    U,X
       JSR    RCMOSB   GET THE TOP BYTE
       PSHS   B
       JSR    RCMOSD
       TFR    D,Y
       PULS   B
       EXG    U,X
       BSR    BOOKSB   PRINT B,Y WITH APPROPRIATE BLANKING

       PULS   A        RESTORE A
       INCA
       CMPU   #BKEND
       BLO    BOOK0    THANK YOU SIR, MAY I HAVE ANOTHER??
*
*      COMPUTE AVERAGE PLAY TIME, AVERAGE LIVES PER GAME
*              

       LDA    #AVTIM
       JSR    WRD7V

       PSHS   X
       LDX    #BKTIME
       LDY    #BKPLAY  PLAYS
       JSR    BCDIV    DO THE DIVIDE
       PULS   X

       LDB    QUO      GET TOP OF QUOTIENT
       LDY    QUO+1

       BSR    BOOKSB   PRINT OUT TH 6 WITH THE RITE KIND OF BLANKING

       LDA    QUO+3    GET 100THS
       ANDA   #$F      1/10
       STA    DVTEMP   SAVE

       LDA    QUO+3
       BITA   #$10     1 IN THE 10'S
       BEQ    BOOK45

       ANDA   #$EF     TAKE THE BIT AWAY
       LSRA            1/2
       ADDA   #5       RESTORE THE WHATEVER
       DAA
       BRA    BOOK46

BOOK45 LSRA            1/2
BOOK46 ADDA   DVTEMP   ADD THE 1/10
       DAA
       TFR    A,B

       LDA    #AVTEND
       JSR    WRD7V    PUT OUT : END
                    
       LDA    #AVPLA
       JSR    WRD7V

       PSHS   X
       LDX    #BKMEN  
       LDY    #BKPLAY  PLAYS
       BSR    BCDIV    DO THE DIVIDE
       PULS   X

       LDB    QUO      GET TOP OF QUOTIENT
       LDY    QUO+1

       BSR    BOOKSB   PRINT OUT TH 6 WITH THE RITE KIND OF BLANKING

       LDB    QUO+3    GET 100THS
       LDA    #AVPLEN
       JSR    WRD7V

       JSR    AVWAIT
       JMP    ADJV    NOW GO DO ADJUSTMENT MODE

BOOKSB TSTB
       BEQ    BOOK1    
       LDA    #AUDM1
       JSR    WRD7V     FULL SIX DIGITS USED
       LDA    #AUDM2
       JSR    WRD7V
       BRA    BOOK2
BOOK1  LDA    #AUDM3
       JSR    WRD7V
BOOK2  RTS

CBCDA  PSHS   B
       JSR    RCMOSA   GET THE BYTE
       TFR    A,B
       ANDA   #$F
       CMPA   #$9
       BLS    CMCDB
       LDA    #9
CMCDB  ANDB   #$F0
       CMPB   #$90
       BLS    CMCDC
       LDB    #$90
CMCDC  PSHS   B
       ORA    ,S+
       PULS   B,PC

*      DIVIDE - PASS X POINTS AT DIVIDEND
*               PASS Y POINTS AT DIVISOR
*               QUOTIENT 4 BYTES AT QUO IN BCD WITH 100THS
*               QUO: QQ QQ QQ . DD

BCDIV  PSHS   X,Y,D,U
       LDD    #0
       STD    QUO
       STD    QUO+2    ZERO QUOTIENT IN CASE OF ZERO DIVIDE

       BSR    CBCDA    GET 2 BYTES OF DIVIDEND
       STA    DIV
       BSR    CBCDA
       STA    DIV+1
       BSR    CBCDA 
       STA    DIV+2
       LDD    #0
       STA    DIV+3
       STD    DIV+4
       STA    DIVSOR+3
       STD    DIVSOR+4

       TFR    Y,X      POINT AT DIVISOR
       JSR    CBCDA
       STA    DIVSOR
       JSR    CBCDA
       STA    DIVSOR+1
       JSR    CBCDA
       STA    DIVSOR+2

       BNE    BCDIV1   NOT ZERO....WE MAY CONTIUE

       LDD    DIVSOR   CHECK FOR ALL ZERO
       BEQ    DIVEX    EXIT...ZERO DIVISOR

BCDIV1 LDU    #QUO+2   POINT AT UNITS QUOTIENT VALUE

BCDIV4 TST    DIVSOR   WE NEED TO SHIFT DIVISOR DOWN
       BNE    BCDIV2   ITS IN THE RANGE WE LIKE

       LEAU   -1,U     MOVE TO PROPER SIGNIF

       LDB    #5       BYTES TO SHIFT
       LDX    #DIVSOR
BCDIV3 LDA    1,X
       STA    0,X+
       DECB
       BNE    BCDIV3
       CLR    ,X       SHIFT IN ZEROS

       BRA    BCDIV4

BCDIV2 LDX    #DIV

BCDIV5 BSR    SUBME    DO A SUBTRACT
       BCC    BCDIV5   DONE WITH THIS DIGIT

       LDY    #DIVSOR+5      LAST DIGIT
       LDB    #5             SHIFT 3 DOWN

BCDIV6 LDA    -1,Y           GET PREVIOUS BYTE
       STA    ,Y
       LEAY   -1,Y           PUSH BACK
       DECB
       BNE    BCDIV6
       CLR    ,Y             SHIFTED

       LEAU   1,U
       CMPU   #QUO+3         LAST QUO BYTE
       BLS    BCDIV5
DIVEX  PULS   X,Y,D,U,PC

SUBME  LDY    #DIVSOR        POINT AT DIVISOR
       LDA    #0             MAKE SURE WE FIT               
SUBM1  LDB    A,X            GET DIVIDEND BYTE (MUST BE LARGER)
       SUBB   ,Y+            WELL??
       BHI    SUBGO          IT IS
       BLO    SUBBYE         IT ISN'T..DONE

       INCA
       CMPA   #6
       BLO    SUBM1

SUBGO  LDB    #6
       LDY    #DIVTMP
SUBGO1 LDA    #$99     GET A BYTE
       SUBA   6,Y
       STA    ,Y+
       DECB
       BNE    SUBGO1

       LDB    #6
       CLC
       LDA    #1
SUBGO2 ADCA   ,-Y
       DAA
       STA    ,Y       AND STORE BACK
       LDA    #0
       DECB
       BNE    SUBGO2

       LDB    #5             INITIAL OFFSET WHILE WALKING
       LDY    #DIVTMP+6
       CLC
SUBGO3 LDA    ,-Y            GET THE SUBTREHEND
       ADCA   B,X            "SUBTRACT" FROM DIVISOR
       DAA
       STA    B,X            STORE BACK TO DIVISOR
       DECB
       BPL    SUBGO3
       LDA    ,U             GET CURRENT QUOTIENT BYTE
       ADDA   #1
       DAA
       STA    ,U             KICK IT
       CLC
       RTS

SUBBYE SEC
       RTS

*
*      SOUND TEST
*

SNDSRT JSR    SCRCLR         CLEAR THE SCREEN
SNDSR1 LDD    #$FE01        SET UP FOR SOUND LINE 1
       STD    PD,U           
       RTS

SNDCYC PULS   D
       STD    PD+6,U
       LDA    #$3F
       STA    SOUND
       NAP    1,SNDC55
SNDC55 LDA    #$2C
       STA    SOUND
       NAP    1,SNDCY6
SNDCY6 LDA    #$3F
       STA    SOUND
       NAP    1,SNDCY1
SNDCY1 LDD    PD,U
       ANDA   #$3F
       STA    SOUND
       LDA    #SNDTEX
       JSR    WRD7V
       LDA    #$40
       STA    PD+4,U
SNDCY4 NAP    1,SNDCY5
SNDCY5 JSR    AVCHK      ADVANCE???
       BCS    SNDCY3     THEN GO
       DEC    PD+4,U
       BNE    SNDCY4
SNDCY3 LDA    PD+2,U         SEE WHO SET THIS UP??
       BNE    SNDC90         AUTOCY DID....ALWAYS ADVANCE
       LDA    PIA0           CHECK MANUAL/AUTO
       RORA
       BCC    SNDC91         MANUAL...DON'T MOVE ON.
SNDC90 LDD    PD,U
       SEC
       ROLA                  MOVE TO NEXT LINE
       INCB
       CMPB   #7             UP TO LINE 7??
       BLO    SNDCY2
       BSR    SNDSR1
SNDCY2 STD    PD,U
SNDC91 JMP    [PD+6,U]

*
*      CMOS RAM TEST
*

CMTEST LDX    #CMOS
       LDY    #RAMALS        USE SAMRAM
CMTST1 LDA    ,X+            DON'T DO DOUBLES BECAUSE OF HARDWARE BRAIN DAM.
       STA    ,Y+ 
       CMPX   #CMOS+$400     DONE??
       BNE    CMTST1
       LDB    #6             ENOUGH ITERATIONS TO ASSURE
       ORCC   #$3F           STOP INTERRUPTS
CMTST2 LDU    HSEED
       LDY    SEED
       LDX    #CMOS
CMTST3 JSR    RAND
       STA    ,X+
       LDA    #WDATA
       STA    WDOG
       CMPX   #CMOS+$400     DONE??
       BNE    CMTST3
       STY    SEED          RESTORE SEED
       STU    HSEED
       LDX    #CMOS
CMTST4 JSR    RAND
       EORA   ,X+
       ANDA   #$F
       BNE    CMEROR
       LDA    #WDATA
       STA    WDOG
       CMPX   #CMOS+$400
       BNE    CMTST4
       DECB                ANOTHER PASS...DONE??
       BNE    CMTST2
       BSR    RAMBAK   
       CLC                   CLEAR IT FOR REAL
       RTS                   AND RETURN

RAMBAK LDU    #RAMALS
       LDY    #CMOS
RAMBK0 LDA    ,U+            
       STA    ,Y+ 
       CMPY   #CMOS+$400
       BNE    RAMBK0
       RTS

CMEROR BSR    RAMBAK
       SEC
       RTS

*      RAM TEST....Y = PLACE TO RETURN IF NO ERROR
*      X = SEED TO START WITH
*      A = ITERATIONS TO MAKE

RAMTST ORCC   #$3F     NO INTERRUPTS DURING TEST
       CLR    RWCNTL   SET TO RAM READ
       TFR    A,DP     COUNT AT DP.
       TFR    X,D      START WITH A PASSED SEED.
RAM2   TFR    D,U      SAVE THE SEED
RAM0   LDX    #0       MEMORY POINTER
RAM3   COMB            DONT ASK
       BITB   #9
       BNE    RAM4
       COMB
       RORA
       RORB
       BRA    RAM6
RAM4   COMB
       BITB   #9
       BNE    RAM5
       RORA
       RORB
       BRA    RAM6
RAM5   LSRA
       RORB
RAM6   STD    ,X++
       EXG    X,D      SINCE IRQ RUNS OUT OF RAM, STROKE ROVER
       TSTB
       BNE    RAM6B    EVERY 256!
       LDB    #WDATA
       STB    WDOG     HAVE A BONE..EVER BEEN BONED.....
       TFR    DP,B     CHECK IS NORMAL DIAGS
       CMPB   #$FF     FF MEANS FRONT DOOR 
       BNE    RAM6C
       LDB    PIA0     CHECK FOR ADV.
       BITB   #2
       BEQ    RAM6C    NOT PRESSED.
       JMP    ,Y       TIME TO RETURN
RAM6C  CLRB            RETURN B TO ZERO
RAM6B  EXG    X,D      TRADE BACK
       CMPX   #RAMTOP  DONE??
       BNE    RAM3     NOPE..CONTINUE
*
       TFR    U,D      RESTORE SEED.
       LDX    #0
RAM7   COMB
       BITB   #9
       BNE    RAM8
       COMB
       RORA
       RORB
       BRA    RAM10
RAM8   COMB
       BITB   #9
       BNE    RAM9
       RORA
       RORB
       BRA    RAM10
RAM9   LSRA
       RORB
RAM10  CMPD   ,X++
       BNE    RERROR   RAM ERROR!
RAM25  EXG    X,D      CHECK FOR END OF PAGE
       TSTB
       BNE    RAM17
       LDB    #WDATA
       STB    WDOG
       TFR    DP,B     SEE IF NORMAL RUN
       CMPB   #$FF     FRONT DOOR??
       BNE    RAM17C   NOPE
       LDB    PIA0     CHECK ADV.
       BITB   #2
       BEQ    RAM17C
       JMP    ,Y       JUST RETURN (NO ERRORS)
RAM17C CLRB
RAM17  EXG    X,D      TRADE BACK
       CPX    #RAMTOP  DONE??
       BNE    RAM7
       TFR    D,U      SHOVE NEW SEED OVER
       TFR    DP,A
       CMPA   #$FF     FF INDICATES FRONT DOOR   
       BNE    RAM99
       TFR    U,D
       JMP    RAM0     DO ANOTHER ITERATION IN THIS TEST.
RAM99  DECA   TAKE ONE AWAY            
       TFR    A,DP     RETURN
       CMPA   #$80     ZERO OUT IN AUTO CYCLE???
       BEQ    RAM99Z   YEP....
       TSTA            SET FOR BELOW BRANCH
       TFR    U,D      MAKE SEED AND SAVED SEED LOOK ALIKE
       LBNE   RAM0     DO ANOTHER ITERATION
RAM99Z LDB    #1       SET BACK TO ROM
       STB    RWCNTL 
       JMP    ,Y       RETURN
RERROR LEAX   -2,X     BACK TO ERROR POINT
       EORA   ,X       FIND DIFFERENCE.
       EORB   1,X
       TSTA            NO DIFFERENCE?
       BNE    RERR2    DIFF.
       TSTB
       BNE    RERR2
       LEAX   2,X      UNDO THIS BULLSHIT...MAYBE ALPHA BABY.
       BRA    RAM25    AND CONTINUE
RERR2  LDU    #$30     FIND BANK NUMBER
       EXG    X,D
       CLRB
       EXG    X,D
RERR0  CMPX   #0
       BEQ    RERR1    DONE.
       LEAX   -$100,X
       LEAU   $10,U
       CMPU   #$30
       BLS    RERR0
       LDU    #$10
       BRA    RERR0
RERR1  LEAU   1,U
       ASRA
       BCS    RERR5    THIS IS THE BIT
       ASRB
       BCS    RERR5
       BRA    RERR1

RERR5  TFR    U,D      BRING THE BANK,BIT INTO B
       LDA    #1       INDICATE RAM ERROR
       STA    RWCNTL   BACK TO THAT JAZZ
       LDS    #RERR6   RETURN HERE
       BRA    PULSE    PULSE THE LEDS
RERR6  LDA    #RAM>>8
       TFR    A,DP
       TFR    CC,A
       COMA
       BITA   #$C0     INITIAL TEST???
       BEQ    RER098   YEP
       LDA    #FDRAMM
       BRA    RER099
RER098 LDA    #RERRM
RER099 LDS    #HSTK    SET UP THE STACK
       JSR    SCRCLR
       JSR    WRD7V    PRINT THE MESSAGE.
       TFR    CC,A     SEE WHICH TEST
       BITA   #$40     FIRQ SAYS FRONT DOOR
       BNE    REROR7   NOPE
       JMP    FDRET    DO THE FRONT DOOR RETURN
REROR7 LDY    #INIT    OTHERWISE START...TRY AND RUN THE GAME.
       BRA    DELAY

DELAY  LDA    #32
DELA1  LDX    #$5800
DELA2  LEAX   -1,X
       LDB    #WDATA
       STB    WDOG
       CMPX   #0
       BNE    DELA2
       DECA
       BNE    DELA1
       JMP    ,Y       RETURN

*
*      PULSE......PUT SOME CRAP IN LED'S 
*      LOW HALF OF A IS ERROR CODE...
*      BOTH HALVES OF B ARE THE RELEVANT DATA.
*

PULSE  TFR    D,U      SAVE A,B
       LDA    #2
       TFR    A,DP
PULSE0 TFR    U,D
       LDY    #PULSE1
       BRA    PSSUB
PULSE1 LDA    #2
       LDY    #PULSE2
       BRA    DELA1
PULSE2 LDY    #PULSE3
       BRA    BLKLED
PULSE3 LDA    #1
       LDY    #PULSE4
       BRA    DELA1
PULSE4 TFR    U,D      GET DATA BACK
       TFR    B,A      SHIFT B DOWN
       LSRA
       LSRA
       LSRA
       LSRA            PROPER HALF
       LDY    #PULSE5
       BRA    PSSUB    PULSE IT.
PULSE5 LDA    #2
       LDY    #PULSE6
       BRA    DELA1
PULSE6 LDY    #PULSE7
       BRA    BLKLED
PULSE7 LDA    #1
       LDY    #PULSE8
       BRA    DELA1 
PULSE8 TFR    U,D
       TFR    B,A
       LDY    #PULSE9
       BRA    PSSUB
PULSE9 LDA    #2
       LDY    #PULS10
       BRA    DELA1
PULS10 LDY    #PULS11
       BRA    BLKLED
PULS11 LDA    #5 
       LDY    #PULS12
       JMP    DELA1
PULS12 TFR    DP,A       SEE IF FIRST PASS??
       DECA
       TFR    A,DP
       BNE    PULSE0
       LDY    #PULS95
       BRA    BLKLED   BLANK EM.
PULS95 TFR    U,D      RESTORE D
       JMP    ,S       AND RETURN

BLKLED LDA    #$3C
       STA    PIA0+1
       INCA
       STA    PIA1+1
       LDA    #$C0
       STA    PIA1
       JMP    ,Y       AND RETURN

*      PSSUB - PUT LOW HALF OF A TO LEDS

PSSUB  TFR    A,B      SAVE A COPY.
       RORA            BIT 0 - CARRY
       RORA            BIT 0 - BIT 7
       RORA            BIT 0 - BIT 6
       ANDA   #$C0
       STA    PIA1     THOSE BITS OUT.
       LDA    #$34     ASSUME ZERO
       BITB   #$4      SEE IF 1
       BEQ    PSSUB1   NOPE
       LDA    #$3C
PSSUB1 STA    PIA1+1   THATS ALL FOR THAT BIT
       LDA    #$34
       BITB   #$8
       BEQ    PSSUB2
       LDA    #$3C
PSSUB2 STA    PIA0+1
       JMP    ,Y       AND RETURN

ROMTST ORCC   #$3F     NO INTERRUPTS WHILE THIS RUNS
       LDX    #ROMTAB  INDEX INTO TABLE
ROM0   CMPX   #ROMEND  DONE??
       BEQ    ROMDUN   YEP   
       LDA    1,X      LOOK TO SEE IF PART STUFFED??
       BEQ    ROMBOT   NOPE...MOVE TO NEXT
       LDA    ,X       GET BASE
       CLRB
       TFR    D,U      USE U TO POINT THROUGH
       LDA    #WDATA   FOR ROVER
ROM1   ADDB   ,U+      ADD A BYTE
       STA    WDOG
       EXG    D,U      SEE IF DUN
       CMPA   2,X      ARE WE TO NEXT PART
       EXG    D,U      CHANGE BACK
       BNE    ROM1     NOPE
       CMPB   1,X      EXPECTED SUM???
       BNE    ROMERR   NOPE....LETS SEND OUT PROBLEM WARNING.
ROMBOT LEAX   2,X      MOVE TO NEXT PART
       BRA    ROM0
ROMERR LDA    ,X       GET THE ADDRESS OF THE BAD PART
       LSRA
       LSRA
       LSRA
       LSRA
       CMPA   #$D      ONE OF TOP 3 PARTS??
       BLO    ROMER1   NOPE
       SUBA   #4       GROUP THEM WITH THE LOWER PARTS
ROMER1 ADDA   #1       MAKE ERRORS FROM 1-12
       DAA             AND MAKE BCK
       TFR    A,B
       LDA    #2       ROM ERROR
       LDS    #RERR3   RETURN
       JMP    PULSE    AND SEND OUT THE CODE
RERR3  LDA    #RAM>>8
       TFR    A,DP
       LDA    #WDATA
       STA    WDOG
       LDS    #HSTK
       JSR    SCRCLR
       TFR    CC,A
       COMA
       BITA   #$C0     INITIAL TEST??
       BEQ    RERR94
       LDA    #FDROMM
       BRA    RERR95
RERR94 LDA    #ROMERM
RERR95 JSR    WRD7V    PUT THE MESSAGE ON THE SCREEN
       TFR    CC,B     SEE WHERE WE CAM FROM
       BITB   #$40     FIRQ SAYS FRONT DOOR
       BNE    ROMER3
       JMP    FDROM
ROMER3 LDY    #INIT    NOW TRY AND RUN THE GAME
       JMP    DELAY    BUT NOT UNTIL A SHORT DELAY.
ROMDUN JMP    ,Y       ALL ROMS OK...RETURN.

*      ROMTAB  - THIS TABLE INDICATES THE BASE OF THE PART, AND THE
*      RESULTING SUM.  IF SUM=0 THEN PART IS NOT STUFFED.

ROMTAB FCB  $00,0      0000
       FCB  $10,0      1000
       FCB  $20,0      2000
       FCB  $30,0      3000
       FCB  $40,0      4000
       FCB  $50,0      5000
       FCB  $60,0      6000
       FCB  $70,0      7000
       FCB  $80,0      8000
       FCB  $90,0      9000
       FCB  $A0,0      A000
       FCB  $B0,0      B000
       FCB  $C0,0      C000
       FCB  $D0,0      D000
       FCB  $E0,0      E000
       FCB  $F0,0      F000
ROMEND FCB  0
       FCB  0          CHECK BYTE (FUDGER) FOR THIS PART ($F000)

       FCC    '(C)1982 WILLIAMS ELEC.INC'

       ORG    $FFF0
       FDB    PWRUPV,PWRUPV,PWRUPV,PWRUPV
       FDB    PWRUPV
       FDB    PWRUPV,PWRUPV,PWRUPV

       END

