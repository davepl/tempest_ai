       OPT    NOL       
       LIB    RRFRED
       OPT    LIS

       ORG    RAMALS         USE SAM RAM FOR STORAGE
XSTOR  RMB    2
YSTOR  RMB    2
ASTOR  RMB    1
USTOR  RMB    2

FROBX  EQU    $C 
ADJX   EQU    $11
ADVALX EQU    $59

 ORG TSBORG
 JMP ADJUST
 JMP CKCMOS
 JMP OPTSTZ
 JMP GETLET
 JMP CMOSMV
       FDB    OPD1
*
* CMOS MOVE: DESTINATION=Y, SOURCE=X, COUNT=B
*
CMOSMV PSHS A NEED A
CMMV1 LDA ,X+ GET SOURCE
 EXG X,Y SWAP
 JSR WCMOSA RITE IT
 EXG X,Y SWAP BACK
 DECB DONE?
 BNE CMMV1 NO,LOOP
 PULS A,PC RETURN CLEAN
*
* CLEAR CMOS: CRASHES A,B,X
*
CLRALL LDX #CMOS POINT TO START
CLRA1  CLR ,X+
 CMPX #CMOS+$400
 BNE CLRA1  NO, LOOP
 RTS BYE
*
* INIT CMOS: SAVES STATE
*
CMINIZ PSHS X,Y,D SAVE STATE
 LDX #DEFALT POINT TO ROM
 LDY #CMOS POINT TO DESTINATION
 LDAB #DEFSIZ LENGTH
 BSR CMOSMV MOVE THEM ALL
 PULS X,Y,D,PC ALL DONE

OPSET PSHS X,Y,D
 LDX #OPD1
 LDY #OPMESS
 LDB #52      50 BYTES WORTH
 BSR CMOSMV   DO THE XFER
 JSR OPCHK FIND THE CHECK BYTE
 LDX #OMESUM  PLACE TO STORE IT
 JSR   WCMOSA WRITE IT
 PULS X,Y,D,PC
*
*
* DEFAULT HERE FOR NOW
*
DEFALT FCB $25 REPLAY @10,000
 FCB $3 NSHIP
 FCB $03 CSELCT
 FCB $01 SLOT1M
 FCB $04 SLOT2M
 FCB $01 SLOT3M
 FCB $01 CUNITC
 FCB $00 CUNITB
 FCB $00 MINUNT
 FCB $01 "GAME WORKS = 0" ADJUSTMENT
 FCB $03 GA1   MASTER DIFFICULTY
 FCB $03 GA2  NUMBER OF LETTERS FOR GOD
 FCB 0   FACTORY SETTINGS
 FCB 0 CLEAR AUDIT TOTALS
 FCB 0 CLEAR HSTD 
 FCB 0 AUTOCYCLE
 FCB 0 ENTER OPERATOR MESSAGE
 FCB 0 ENTER HSTD
DEFSIZ EQU *-DEFALT
OPD1 FCC '     HELLO WORLD!'             
 FCC '        ' SPACES AFTER PRESENTED BY
OPD2 FCC 'WILLIAMS ELECTRONICS INC'
 FCB $3D PERIOD
 FCB $2F,$28
*
* COINAGE SELECT TABLE
*
CSELCT FCB $01,$04,$01,$01,$0,$0 FACTORY USE FOR CUSTOM START
 FCB $01,$04,$01,$02,$04,$00 50C..3/$1.00
 FCB $06,$00,$01,$01,$00,$00 NEWNEW GERMAN
 FCB $01,$04,$01,$01,$00,$00 25C
 FCB $01,$16,$06,$02,$00,$00 NEW FRENCH
 FCB $01,$04,$01,$02,$00,$00 50C
 FCB $01,$00,$04,$01,$00,$00
 FCB $01,$00,$02,$01,$00,$00
 FCB $01,$00,$02,$02,$00,$00
 FCB $01,$04,$01,$01,$00,$00 FREE PLAY
*
ADJH   EQU    30

ADJTBL FCB    0,$50    REPLAY LEVEL 
       FDB    REPTTT
       FCB    0,ADJH
*
       FCB    1,$20    SHIPS FOR 1 CREDIT
       FDB    CRED1T
       FCB    0,ADJH+9 
*
       FCB    0,9      CSELECT
       FDB    CONSLT
       FCB    0,ADJH+20
*
       FCB    0,$99    LEFT MULT
       FDB    NOWORD
       FCB    1,ADJH+29
*
       FCB    0,$99    CENTER
       FDB    NOWORD
       FCB    1,ADJH+38
*
       FCB    0,$99     RIGHT
       FDB    NOWORD
       FCB    1,ADJH+47
*
       FCB    1,$99     COIN UNITS FOR CREDIT
       FDB    NOWORD
       FCB    1,ADJH+56
*
       FCB    0,$99     COIN UNITS FOR BONUS
       FDB    NOWORD
       FCB    1,ADJH+65
*
       FCB    0,$99     MINIMUM COINAGE NEEDED
       FDB    NOWORD
       FCB    1,ADJH+74 
*
       FCB    0,1      FREE PLAY
       FDB    YESNO
       FCB    0,ADJH+85 
*
       FCB    0,$10    MASTER DIFFICULTY
       FDB    MASTT
       FCB    0,ADJH+94
*
       FCB    3,$20     LETTERS IN GODS NAME
       FDB    GODLT
       FCB    0,ADJH+103
*
       FCB    0,1
       FDB    YESNO1
       FCB    0,ADJH+114
*
       FCB    0,1
       FDB    YESNO1
       FCB    0,ADJH+123
*
       FCB    0,1
       FDB    YESNO1
       FCB    0,ADJH+132
*
       FCB    0,1
       FDB    YESNO1
       FCB    0,ADJH+141
*
       FCB    0,1
       FDB    YESNO1
       FCB    0,ADJH+150
*
       FCB    0,1
       FDB    YESNO1
       FCB    0,ADJH+159

REPTTT FCB    0,NOREPM
       FCB    $20,REPMM1
       FCB    $25,REPMM2
       FCB    $30,REPMM3
       FCB    $50,REPMM4
       FCB    $FF

CRED1T FCB    0,NOMESS
       FCB    2,HIVOL
       FCB    3,RECOM
       FCB    4,POORP
       FCB    5,NOMESS
       FCB    $FF

NOWORD FCB    0,NOMESS
       FCB    $FF

CONSLT FCB    0,CUSTM
       FCB    1,COINM1
       FCB    2,COINM1+1
       FCB    3,COINM1+2
       FCB    4,COINM1+3
       FCB    5,COINM1+4
       FCB    6,COINM1+5
       FCB    7,COINM1+6
       FCB    8,COINM1+7
       FCB    9,COINMF
       FCB    $FF

YESNO  FCB    0,NOM 
       FCB    1,YESM
       FCB    $FF

YESNO1 FCB    0,NOM
       FCB    1,YES1M
       FCB    $FF

MASTT  FCB    0,EXLIB
       FCB    3,LIB  
       FCB    5,RECOM
       FCB    6,CONS 
       FCB    8,EXCON
       FCB    $FF

GODLT  FCB    0,RECOM 
       FCB    $4,NOMESS
       FCB    $FF

*
***    DISPLAY FULL TEXT FOR A LINE.
*
*      Y = CMOS LOCATION
*      U = POINTER TO TABLE ENTRY FOR THIS LOCATION.
*      A = TEXT MESSAGE FOR LOCATION

DISADJ PSHS   A,Y            SAVE TEXT
       LDA    #ADJX          ADJUSTMENT X
       TST    4,U            PROTECTED??
       BEQ    DSADJ1
       LDA    #ADJX+5        THEN INDENT.
DSADJ1 LDB    5,U            GET Y FOR THIS GUY
       TFR    D,X            AND MAKE IT CURSOR
       LDA    ,S             RESTORE A
       JSR    WRD5FV         PRINT THE NAME OF THE ADJUSTMENT
       BSR    PRVAL          PRINT THE VALUE APPROPRIATELY  (AND MESSAGE)
       PULS   A,Y,PC         AND RETURN

PRVAL  PSHS   Y,D,X
       EXG    X,Y
       JSR    RCMOSB         GET THE VALUE
       CMPB   #$FF
       BNE    PRVALZ         STUPID FIX
       LDB    #$EE
PRVALZ EXG    X,Y            CHANGE BACK
       EXG    X,D      
       LDA    #ADVALX
       LDB    5,U            HEIGHT
       EXG    X,D
       PSHS   D
       LDD    #$3C05  
       JSR    BLKCLR
       PULS   D
       BSR    GETMES         GET THE ACCOMANYING MESSAGE
       JSR    WRD5FV         AND PRINT
       PULS   Y,D,PC,X       AND RETURN

GETMES PSHS   Y
       LDY    2,U            GET TO AREA TO PARSE
GETMS1 CMPB   ,Y++           DO WE RATE OVER THIS ONE
       BHS    GETMS1
       LDA    -3,Y           GRAB THE APPROPRIATE MESSAGE POINTER
       PULS   Y,PC           AND RETURN

PAGE1  JSR    SCRCLR
       LDA    #$66
       STA    TEXCOL         GREEN FOR WALLY!
       LDU    #ADJTBL
       LDY    #CMOS
       LDA    #ADJMES
       LDX    #ENDADJ        PLACE TO FINISH

       PSHS   X              ARCHAIC PUSH
PRNPG1 BSR    DISADJ         PRINT OUT THE CURRENT LINE
       LEAU   6,U            PUSH TO NEXT ENTRY
       LEAY   2,Y
       INCA
       CMPY   0,S            SEE IF DONE
       BNE    PRNPG1
       LDA    #CONTMS        TITLES AND STUFF
       JSR    WRD5V          PUT THEM IN
       PULS   X,PC

ADJUST JSR    CKADJ          SEE IF OUT OF WHACK TO BEGIN WITH
       BEQ    ADJJJJ         LOOKS OK TO ME
       JSR    CMINIZ         INITIALIZE THE SETTINGS
ADJJJJ LDX    #SPECFN        ZERO OUT THE SPECIAL FUNCTION BYTES
ADJS11 CLR    ,X+
       CMPX   #ENDADJ
       BLO    ADJS11
       JSR    MAKCHK         PUT DOWN THE APPROPRIATE CHECKSUM
       BSR    PAGE1          PUT THE FIRST PAGE UP
       LDU    #ADJTBL        START OF TABLE
       LDY    #CMOS          START OF MEMORY
       LDA    #ADJMES
       JSR    SSTATE         STORE THE STATE
       JSR    FROBON         PUT FROB IN PLACE.
       MAKP   ADVSCN         PROCESS TO SCAN ADVANCE AND TERMINATE IF HIT.
*      
*      BEGIN STEADY STATE MODE.
*
ADJST0 JSR    UPDOWN         CHECK FOR UP/DOWN
       BSR    INCDEC         CHECK FOR THRUST/FIRE
       NAP    1,ADJST0

INCDEC PULS   D
       LDU    CRPROC
       STD    PD+6,U
       LDA    #$20
       STA    PD,U
       LDA    PIA2  CHECK FOR FIRE UP, FIRE DOWN
       BMI    DECME    FIRE DOWN...DECREMENT
       BITA   #$40
       BNE    INCME    FIRE UP...
IDRET  JMP    [PD+6,U]

INCME  BSR    INC1          ONE INCREMENT
INCME0 NAP    1,INCME1
INCME1 LDA    PIA2
       BITA   #$40
       BEQ    IDRET          LET GO
       DEC    PD,U           DELAY
       BNE    INCME0
       LDA    #5
       STA    PD,U
       BRA    INCME

DECME  BSR    DEC1
DECME0 NAP    1,DECME1
DECME1 LDA    PIA2
       BPL    IDRET    HE LET GO
       DEC    PD,U
       BNE    DECME0
       LDA    #5
       STA    PD,U
       BRA    DECME

INC1   JSR    LSTATE         GET THE STATE LOADED
       TFR    Y,X            COPY POINTER
       JSR    RCMOSA         GET THE CURRENT VALUE
       CMPY   #REPLAY        IS THIS THE REPLAY LEVEL??
       BEQ    INC22
       ADDA   #1
       DAA                   FORM NEW VALUE
       BCS    INC11          NO WRAP AROUND
       CMPA   1,U            COMPARE TO HIGHEST ALLOWED.
       BHI    INC11          TOO HIGH....GET OUT OF HERE
DEC11  TFR    Y,X            RESTORE POINTER
       JSR    WCMOSA         PUT THE NUMBER BACK IN MEMORY
       JSR    MAKCHK         REFORM THE CHECK BYTE
       JSR    PRVAL          PRINT OUT THE NEW VALUE AND INFORMATION.
       JSR    SPCASE         CHECK FOR SPECIAL CASES.
INC11  RTS                   AND RETURN

DEC1   JSR    LSTATE
       TFR    Y,X
       JSR    RCMOSA
       TSTA
       BEQ    INC11          ALL THE WAY DOWN
       CMPY   #REPLAY        IS THIS THE REPLAY ADJUSTMENT??
       BEQ    DEC22
       ADDA   #$99
       DAA
       CMPA   0,U            SEE IF TOO LOW
       BLO    INC11          TOO LOW...BYE
       BRA    DEC11          DO THE REST    

INC22  BSR    FINDME
       LDA    1,X
       BRA    DEC11          AND FINISH

DEC22  BSR    FINDME         MAKE X POINT TO CLOSEST APPROX
       LDA    -1,X           FETCH THE VALUE BEFORE
       BRA    DEC11

FINDME LDX    #RVTAB
FINDM1 CMPA   ,X             ARE WE NEAR??
       BLS    FMBYE          YEP
       LEAX   1,X
       CMPX   #RVEND
       BLO    FINDM1
FMBYE  RTS

*      REPLAY ADJUSTMENT TABLE

       FCB    0
RVTAB  FCB    0
       FCB    $20
       FCB    $25
       FCB    $30
RVEND  FCB    $50
       FCB    $50

PROTCK PSHS   A,X
       LDA    4,U            CHECK PROTECTION CODE
       BEQ    PROTOK
       LDX    #COINSL        COIN PROTECTED
       DECA                  1??
       BEQ    PROTC1
       LDX    #GA1
PROTC1 JSR    RCMOSA
       TSTA
       BEQ    PROTOK
       SEC
       PULS   A,X,PC
PROTOK CLC    
       PULS   A,X,PC

FROBON PSHS   X,D
       LDB    5,U      Y COORD OF FROB 
       LDA    #FROBX
       TFR    D,X
       LDA    #FROBM
       JSR    WRD5FV
       PULS   X,D,PC

FROBOF PSHS   X,D
       LDB    5,U
       LDA    #FROBX
       TFR    D,X
       LDD    #$0305   WIDTH,HEIGHT
       JSR    BLKCLR
       PULS   X,D,PC

UPDOWN PULS   D
       LDU    CRPROC
       STD    PD+6,U         SAVE RETURN ADDRESS
       LDA    #$30           TYPOMATIC FIRST DELAY
       STA    PD,U           INDICATE VIRGIN SCAN
       LDA    PIA2           CHECK FOR DOWN
       RORA                  MOVE UP??
       BCS    UPPER
       RORA
       BCS    DOWNER
UDRET  JMP    [PD+6,U]       NO ACTION...RETURN

DOWNER BSR    DOWN1          GO DOWN ONE (WHATEVER THAT MEANS)
DOWN2  NAP    1,DOWN3
DOWN3  LDA    PIA2           CHECK IF STILL PRESSED
       BITA   #$2            WELL??
       BEQ    UDRET          DONE WITH THIS PASS...NO LONGER PRESSED
       DEC    PD,U           TIME ME
       BNE    DOWN2          SLEEP ANOTHER FRAME BABE 
       LDA    #8             FAST RATE
       STA    PD,U
       BRA    DOWNER         NOW DO ANOTHER STEP.

UPPER  BSR    UP1
UPPER1 NAP    1,UPPER2
UPPER2 LDA    PIA2
       RORA
       BCC    UDRET          LET GO
       DEC    PD,U
       BNE    UPPER1
       LDA    #8
       STA    PD,U
       BRA    UPPER

DOWN1  JSR    LSTATE   GET US OUR STATE
       BSR    FROBOF   CAN THE FROB
DOWN11 CMPY   #ENDADJ-2   ARE WE ON LAST ENTRY??
       BEQ    DOWNRT         THEN RETURN
       LEAY   2,Y            MOVE TO NEXT
       INCA
       LEAU   6,U
       JSR    PROTCK         CHECK PROTECTION
       BCS    DOWN11
       JSR    SSTATE
DOWNRT JSR    FROBON         NOW TURN THE FROB ON.
       RTS

UP1    JSR    LSTATE
       BSR    FROBOF   CAN THE FROB
UP11   CMPY   #CMOS    FIRST ENTRY
       BEQ    DOWNRT   YEP...BYE
       LEAY   -2,Y
       DECA
       LEAU   -6,U
       JSR    PROTCK
       BCS    UP11     PROTECTED...PUSH TO NEXT ONE.
       JSR    SSTATE
       JMP    FROBON   TURN THE FROB ON AND RETURN.

ADVSCN NAP    1,ADVSC1                
ADVSC1 LDA    PIA0     SCAN FOR ADVANCE
       BITA   #2
       BEQ    ADVSCN   JUST KEEP ON SCANNING
       JSR    GNCIDE   KILL THE TEST PROCESS
       JSR    SCRCLR
ADVSC2 LDA    PIA0
       BITA   #2
       BNE    ADVSC2   WAIT FOR RELEASE
ADVSC3 LDA    SPECFN+3 CLEAR AUDITS??
       ANDA   #$F
       BEQ    ADVSC4   
       CLR    SPECFN+3
       JSR    MAKCHK
       JSR    SCRCLR
       JSR    AUDCK4   CLEAR THE AUDIT TOTALS AND PRINT THE MESSAGE.
       NAP    $40,ADVSC4
ADVSC4 LDA    SPECFN+5      CHECK AUDIT MESSAGE
       ANDA   #$F
       BEQ    ADVSC6        NOT THERE
       CLR    SPECFN+5
       JSR    MAKCHK
       JSR    HIGHRS        GO AND DO IT
       NAP    $40,ADVSC6
ADVSC6 LDA    SPECFN+9      OPERATOR MESSAGE
       ANDA   #$F           LOOK AT RELEVANCE
       BEQ    ADVSC5
       CLR    SPECFN+9
       JSR    MAKCHK
       LDA    #$3A           FILL WITH SPACES
       LDX    #OPMESS
       LDB    #50            50 CHARACTERS
ADV6ZZ JSR    WCMOSA
       DECB
       BNE    ADV6ZZ
       JSR    SCRCLR         CLEAR THE SCREEN
       LDA    #OMESM
       JSR    WRD7V          PRINT A PROMPTING MESSAGE
       LDY    #OPMESS        POINT AT CMOS AREA
       LDX    #$2530         SCREEN ECHO
       LDD    #$1980         25 CHARS, SPECIALS ALLOWED
       JSR    GETLET         GET THE MESSAGE
       LDB    #$30           DISPLAY HEIGHT
       LDX    #OPL1          PLACE TO STORE RESULT
       LDY    #OPMESS        CMOS
       JSR    SLIDE
       LDA    #OMESM
       JSR    WRD7V
       JSR    REDISP         DISPLAY CENTERED TOP LINE
       LDY    #OPMESS+50
       LDX    #$2540
       LDD    #$1980
       JSR    GETLET
       LDB    #$40
       LDX    #OPL2
       LDY    #OPMESS+50
       BSR    SLIDE
       JSR    OPCHK          FORM THE CHECK BYTE
       LDX    #OMESUM        POINT
       JSR    WCMOSA         AND STORE IT.
ADVSC5 LDA    SPECFN+11      ENTER GOD'S NAME??
       ANDA   #$F
       BEQ    ADVSC8
       CLR    SPECFN+11
       JSR    MAKCHK
       JSR    GODSET        GO AND DO IT
ADVSC8 LDA    SPECFN+7      AUTO CYCLE
       ANDA   #$F
       BEQ    NOAUTZ
       CLR    SPECFN+7
       JSR    MAKCHK
       BSR    FSCHK
       JMP    AUTOCY
NOAUTZ BSR    FSCHK
****************************************************
*
*      INHIBIT SWITCHES (ADVANCE IN PARTIC)
*
*      LDD    #0
*      STD    SWPROC
*      STD    SWPROC+4
*
*
*****************************************************
       JMP    INIT     NOW GO INTO GAME OVER.

FSCHK  LDA    SPECFN+1
       ANDA   #$F      FACTORY SETTINGS??
       BEQ    ADVS33
       INC    ADJSUM
       INC    ADJSUM   BASH THE CHECK BYTE...THIS WILL RESTORE.
       CLR    SPECFN+1 AND UNDO IT
ADVS33 RTS

*      PD = ECHO PTR
*      PD+2 = CMOS OF 25 CHARS
*      PD+4 = CMOS OF STORED OFFSET
*      PD+6 = RETURN

SLIDE  LDU    CRPROC
       LDA    #$25     START X
       STD    PD,U     SAVE ECHO POINTER
       PULS   D
       STD    PD+6,U
       STX    PD+4,U
       STY    PD+2,U
       LDA    #$25
       JSR    WCMOSA 
       LDA    #OPCNTM
       JSR    WRD7V
SLIDE0 NAP    4,SLIDE1
SLIDE1 LDA    PIA0
       BITA   #2       DEBOUNCE ADV.
       BNE    SLIDE0
SLIDE5 LDA    PIA3     CHECK FOR THRUST OF FIRE RIGHT
       BITA   #2       FIRE??
       BEQ    SLIDE3   NOPE
       LDX    PD+4,U   GET MEMORY POINTER
       JSR    RCMOSA   FETCH THE BYTE
       LDX    PD+4,U   RESTORE POINTER
       INCA            ADD ONE
       CMPA   #$3A     NO FARTHER RIGHT THAN 38
       BLS    SLIDE4
       LDA    #$3A
       BRA    SLIDE4
SLIDE3 RORA            FIRE LEFT??
       BCC    SLIDE6   NOTHING PRESSED...LOOP
       LDX    PD+4,U
       JSR    RCMOSA
       LDX    PD+4,U
       DECA
       CMPA   #$13
       BHS    SLIDE4
       LDA    #$13
SLIDE4 JSR    WCMOSA
       STA    PD,U           ALSO MAKE X OF ECHO POINTER
       BSR    REDISP
       NAP    $10,SLIDE5
SLIDE6 LDA    PIA0           CHECK FOR ADVANCE
       BITA   #2
       BNE    SLIDE7         TIME TO LEAVE
       NAP    4,SLIDE5
SLIDE7 JSR    SCRCLR
SLIDE8 NAP    4,SLIDE9
SLIDE9 LDA    PIA0
       BITA   #2
       BNE    SLIDE8
       JMP    [PD+6,U]

REDISP CLR    TXFLAV
       LDX    PD,U           UPPER LEFT TO ECHO
       LEAX   -$100,X
       LDA    #90
       LDB    #9
       JSR    BLKCLR         CLEAR THE BLOCK
       LDX    PD,U           CURSOR
       CLR    TXFLAV
       LDY    PD+2,U         DATA POINTER
       LDB    #25
RDSP1  EXG    X,Y
       JSR    RCMOSA
       EXG    X,Y
       JSR    PR57V
       DECB
       BNE    RDSP1
       RTS                   DONE!

SSTATE STY    YSTOR
       STU    USTOR
       STA    ASTOR
       RTS

LSTATE LDY    YSTOR
       LDU    USTOR
       LDA    ASTOR
       RTS

SPCASE CMPY   #COINSL  COIN SELECT
       BEQ    INCOIN
       RTS
INCOIN TFR    Y,X
       JSR    RCMOSA   GET THE INDEX
       ASLA
       PSHS   A
       ASLA
       ADDA   ,S+      6X
       LDX    #CSELCT
       LEAX   A,X      SLIDE DOWN
INCON1 LEAU   6,U
       LEAY   2,Y
       LDA    ,X+
       PSHS   X
       TFR    Y,X
       JSR    WCMOSA
       PULS   X
       JSR    PRVAL
       CMPY   #MINUNT
       BLO    INCON1

MAKCHK PSHS   X,A
       BSR    FCHK     FIND THE NEW CHECKSUM.
       LDX    #ADJSUM
       JSR    WCMOSA
       PULS   X,A,PC

FCHK   PSHS   B,X,Y
       LDX    #CMOS
       LDY    #ENDADJ
       BSR    FCHKZ
       PULS   B,X,Y,PC

OPCHK  LDX    #OPMESS
       LDY    #OPMESS+104
*
FCHKZ  STY    XTEMP
       CLRA
FCHK1  LDB    ,X+
       ANDB   #$F
       PSHS   B
       ADDA   ,S+
       CMPX   XTEMP   
       BNE    FCHK1
       ADDA   #$37     FUDGE FACTOR
       RTS            

CKADJ  BSR    FCHK
       PSHS   A
       LDX    #ADJSUM
       JSR    RCMOSA
       CMPA   ,S+
       RTS

CKCMOS BSR    OPTST    CHECK OPERATOR MESSAGE
       BSR    CKADJ    CHECK ADJ 
       BEQ    CKSRT0
       LDA    #WDATA
       STA    WDOG
       JSR    CMINIZ
       LDA    #WDATA
       STA    WDOG
       BSR    MAKCHK
       LDA    #WDATA
       STA    WDOG
       JSR    SCRCLR    CLEAR THE SCREEN
       LDA    #WDATA
       STA    WDOG
       BSR    AUDCHK    CHECK FOR FAULTY AUDITS
       JSR    HSCHKV    CHECK FOR FAULTY HIGH SCORES
       JSR    PUPHSC    NOW VALIDATE
       BSR    CKADJ
       BEQ    CKSNOR
       LDA    #CMIMES   TELL THE OPERATOR WHAT IS GOING ON.
CMLOP0 JSR    WRD7V
CMLOP  LDA    #WDATA
       STA    WDOG
       LDA    PIA0      CHECK
       BITA   #2
       BEQ    CMLOP     NOT PRESSED CONTINUE TO HANG
CKSMRT RTS     
CKSRT0 JMP    PUPHSC   FIX UP THE HIGH SCORE AREA AND RETURN
CKSNOR LDA    #RESMES
       BRA    CMLOP0

AUDCHK LDX    #SLOT1
       LDAB   #4       FOUR BAD BYTES IN AUDIT AREA AFTER RESET IS ENOUGH.
AUDCK2 LDA    ,X+      GET A BYTE
       ANDA   #$F
       CMPA   #9       HEX???
       BLS    AUDCK1   NOPE
       DECB            ONE MORE
       BEQ    AUDCK4
AUDCK1 CMPX   #BKEND   DONE??
       BNE    AUDCK2 
       RTS
AUDCK4 LDA    #BOOKRS
       JSR    WRD7V    PRINT IT
       LDX    #SLOT1
AUDCK5 CLR    ,X+      AND CLEAR EM.
       CMPX   #BKEND   
       BNE    AUDCK5
OPTRTS RTS

OPTST  BSR    OPTSTZ   CHECK THE BYTE
       BEQ    OPTRTS   ITS OK
       JMP    OPSET    NO GOOD DEFAULT IT.

*
*      COMPARE FOR OPERATOR MESSAGE CHECK BYTE
*      BEQ FOR OK
*

OPTSTZ JSR    OPCHK    FORM CHECK BYTE
       PSHS   A        SAVE
       LDX    #OMESUM  GET THE SUM
       JSR    RCMOSA
       CMPA   ,S+
       RTS
*
***    GETLET    GET A STRING OF LETTERS FROM THE USER
*
*      A = NUMBER OF LETTERS TO FETCH
*      B = $80 FOR ALL, 00 FOR ALPHA ONLY.
*      X = UPPER LEFT OF REGION TO ECHO IN
*      Y = ADDRESS TO STORE IN (RAM OR CMOS)
*
*      PD = NUMBER OF LETTERS LEFT TO GET
*      PD+1 = 0 FOR SPECIALS ALLOWED, $80 FOR ALPHA ONLY
*      PD+2 = ECHO ADDR
*      PD+4 = ADDRESS TO STORE LETTER
*      PD+6 = RETURN
GETLET LDU    CRPROC
       STD    PD,U     SAVE COUNT
       STX    PD+2,U
       STY    PD+4,U
       PULS   D        GET RETURN ADDRESS
       STD    PD+6,U
GETLZZ NAP    4,GETLZ1
GETLZ1 BSR    GETFIR
       BNE    GETLZZ   IT IS PRESSED!
       MAKP   TIMPRC
       LDD    PD+6,U
       STD    PD+6,X   PASS RETURN ADDRESS
       LDD    PD,U
       STD    PD,X
       STU    PD+2,X   PASS US TO KILL
       STX    GETLRM   SAVE HIS ID
       LDA    #$99     SET COLOR TO WHITE
       STA    TEXCOL   
       CLR    TXFLAV   MAKE ZERO FLAVOR
GETLLL CLR    GETLST  MAKE LAST NOT AVAILABLE.
       CLR    GETLST+1
       CLR    GETLST+2
GETLT0 BSR    G0SUB    CLEAR THE REGION
GETLT1 NAP    2,GETLT2

G0SUB  JSR    CLRREG
       JSR    PRFROB
       LDA    #$3A
       JMP    LSTY

GETFIR LDA    PIA2
       BITA   #$40     FIRE UP?   
GETFR  RTS             RETURN PROPER CC

GETLT2 LDA    PIA2     SCAN FOR DOWN
       RORA            CHECK FOR MOVE UP
       LBCS   LUP
       RORA
       LBCS   LDOWN

GETRET BSR    GETFIR   CHECK FOR FIRE UP
       BEQ    GETLT1   NOT PRESSED..LOOP

       LDA    PD+1,U   GET ALPHA 'BIT'
       ANDA   #$80
       ADDA   #$20     TIME FOR CHASE = 1/4

G1LET  STA    PD+1,U
       JSR    LLDY     GET THE CURRENT LETTER
       CMPA   #$5E     IS IT A RUB
       BEQ    GETRUB
       LDA    TXFLAV
       STA    GETLST+2
       JSR    NOFROB   TAKE THE FROB AWAY
       JSR    LLDY     GET CURRENT LETTER
       LDX    PD+2,U   GET POINTER
       STX    GETLST   SAVE CURRENT ECHO POINTER
       JSR    PR57V    PUT THE LETTER OUT
       STX    PD+2,U   THIS MAKES NEW POINTER
       JSR    PUSHY    PUSH THE Y POINTER TO NEXT
       DEC    PD,U     1 LESS
       BEQ    G2LET
       JSR    G0SUB    MAKE IT LOOK LIKE WE'RE GETTING NEXT

GETLT3 NAP    2,GETLT4
GETLT4 BSR    GETFIR   STILL DOWN??
       BEQ    GETLT5   NOPE

       LDA    PD,U     GET LETTERS LEFT
       DECA            1???
       BEQ    GETLT3   NO TYPOMATIC OF LAST LETTER
       DEC    PD+1,U   ONE LESS COUNT
       BEQ    GETLT5
       LDA    PD+1,U   CHECK FOR 80
       CMPA   #$80
       BNE    GETLT3
GETLT5 JSR    GETFIR
       BEQ    GETLT1   NOT PRESSED...BACK TO MAIN LOOP
       LDA    PD+1,U   GET COUNTER
       ANDA   #$80
       ADDA   #4
       BRA    G1LET 
G2LET  LDX    GETLRM   KILL THE TIMER PROCESS
       JSR    KILL
       JMP    [PD+6,U]

GETRUB BSR    NOFROB    MAKE LIKE WE'RE NOT HERE.
       BSR    CLRREG
       LDA    #$3A
       BSR    LSTY     STORE A SPACE AT THE CURRENT SPOT
       LDX    GETLST
       STX    PD+2,U
       LDB    GETLST+2
       STB    TXFLAV
       INC    PD,U     ONE MORE CHARACTER TO GET.
       LDY    PD+4,U   GET THE Y
       LEAY   -1,Y     MOVE BACK ONE ASSUMING RAM
       CMPY   #$C000   CMOS??
       BLO    GETRB1   NOPE
       LEAY   -1,Y
GETRB1 STY    PD+4,U
GETRB6 NAP    1,GETRB5
GETRB5 JSR    GETFIR
       BNE    GETRB6
       JMP    GETLLL   NOW GO AND GET THE LAST CHARACTER AGAIN.

CLRREG PSHS   D
       LDX    PD+2,U   GET POINTER TO REGION
       LDD    #$407    WIDTH,HEIGHT
       JSR    BLKCLR   CLEAR THE BLOCK
       PULS   D,PC

PRFROB LDA    #$99     
PRFRB1 LDX    PD+2,U   GET POINTER
       STA    8,X 
       STA    $108,X
       STA    $208,X
       RTS

NOFROB CLRA
       BRA    PRFRB1

LLDY   LDY    PD+4,U   GET Y POINTER
       CMPY   #$C000
       BHS    LDCMOS   CMOS
       LDA    ,Y       ELSE LOAD THE NUMBER
       RTS
LDCMOS PSHS   X
       LDX    PD+4,U   LOAD INTO X
       JSR    RCMOSA   LOAD IT
       PULS   X,PC     AND RETURN

LSTY   LDY    PD+4,U   GET POINTER
       CMPY   #$C000   CMOS??
       BHS    LSCMOS
       STA    ,Y
       RTS
LSCMOS PSHS   X
       LDX    PD+4,U
       JSR    WCMOSA
       PULS   X,PC

PUSHY  LDY    PD+4,U
       LEAY   1,Y
       CMPY   #$C000
       BLO    PUSHY1   NOT CMOS..DONE
       LEAY   1,Y
PUSHY1 STY    PD+4,U
LDRET  RTS

DELAY1 LDX    #$2000
DELAY2 LEAX   -1,X
       BNE    DELAY2
       RTS

LDOWN  LDA    #10      1 DOWN ITERATION
LDOWN0 BSR    LDN1     1 SHOT
LDOWN1 BSR    DELAY1   SHORT DELAY
       LDB    PIA2     SCAN
       BITB   #$2
       LBEQ   GETRET   GONE
       DECA
       BNE    LDOWN1
       NAP    1,LDOWN3
LDOWN3 LDA    #1
       BRA    LDOWN0

LUP    LDA    #10
LUP0   BSR    LUPP1
LUP1   BSR    DELAY1  
       LDB    PIA2
       RORB
       LBCC   GETRET
       DECA
       BNE    LUP1
       NAP    1,LUP3
LUP3   LDA    #1
       BRA    LUP0

LUPP1  PSHS   A        SAVE A PLEASE
       JSR    LLDY     GET THE CURRENT VALUE
       INCA            PUSH TO NEXT
       TST    PD+1,U   SEE IF ALPHA ONLY
       BPL    LUPP2    YES
       TST    GETLST   RUB ALLOWED??
       BEQ    LUP1ZZ   NOPE.
       CMPA   #$5E
       BLS    LUPP1Z   IN RANGE.
LUP1ZZ CMPA   #$5D     HIGHER THAN SLASH              
       BLS    LUPP1Z
       LDA    #$30     THEN MAKE 0
LUPP1Z CMPA   #$3E     UNUSED PLACE??
       BNE    LUPP3
       INCA
       BRA    LUPP3    PRINT TIME
LUPP2  CMPA   #$5A     HIGHER THAN Z??
       BLS    LUPP2Z
       TST    GETLST   RUB ALLOWED??
       BEQ    LUP2ZZ   NO...MOVE TO SPACE
       CMPA   #$5E     PAST BACK FRBO
       BHI    LUP2ZZ   YEP
       LDA    #$5E     MAKE US THE BACK FROB
       BRA    LUPP2Z
LUP2ZZ LDA    #$3A     THEN MAKE SPACE
LUPP2Z CMPA   #$3B     1 AFTER SPACE (ALPHA)
       BNE    LUPP3
       LDA    #$41     PUSH TO A.
LUPP3  JSR    LSTY     STORE THE NEW ONE.
       JSR    CLRREG   CUT CLEAR OUT THE REGION
       LDB    TXFLAV   SAVE THE FLAVOR
       LDX    PD+2,U   GET CURSOR
       JSR    PR57V
       STB    TXFLAV   RESTORE FLAVOR
LUPBYE PULS   A,PC     AND RETURN

LDN1   PSHS   A
       JSR    LLDY     GET THE CURRENT VALUE
       DECA
       TST    PD+1,U   ALPHA ONLY??
       BPL    LDN2
       CMPA   #$30     COMPARE TO ZERO
       BHS    LDN1Z  
       TST    GETLST
       BEQ    LDN1ZZ   NO RUB...SET TO SLASH
       LDA    #$5E     RUB
       BRA    LDN1Z
LDN1ZZ LDA    #$5D     SET TO SLASH       
LDN1Z  CMPA   #$3E     UNUSED ONE??
       BNE    LDN3ZZ
       DECA
       BRA    LDN3ZZ
LDN2   CMPA   #$39     BEFORE SPACE
       BNE    LDN2Z    THEN EXIT
       TST    GETLST   RUB??
       BEQ    LDN2ZZ   NO...LOAD Z
       LDA    #$5E
       BRA    LDN2Z
LDN2ZZ LDA    #$5A     THEN LOAD Z
LDN2Z  CMPA   #$40     BEFORE A     
       BNE    LDN3     NOPE
       LDA    #$3A     MAKE SPACE
LDN3   CMPA   #$5D     BELOW RUB
       BNE    LDN3ZZ
       LDA    #$5A     THEN LOAD Z
LDN3ZZ BRA    LUPP3  

TIMPRC TST    PD+1,U      DIAGNOSTIC CALL??
       BMI    TIMADV
TIMPR0 NAP    $FF,TIMPR1  4 SECONDS OF SLEEP
TIMPR1 NAP    $FF,TIMPR6
TIMPR6 NAP    $82,TIMPR7
TIMPR7 DEC    PD,U
       BNE    TIMPR0      8 SECONDS PER LETTER
TIMPR2 LDX    PD+2,U
       LEAU   ,X          PUT IN U PLEASE
       JSR    LLDY        LOAD THE CURRENT CHARACTER
       CMPA   #$5E        RUB???
       BNE    TIMAZZ
       LDA    #$3A
       JSR    LSTY        AND STORE IT
TIMAZZ LDU    CRPROC
       JSR    KILL
       JMP    [PD+6,U]
TIMADV LDA    PIA0
       BITA   #2
       BNE    TIMPR2       CAN IT ON ADVANCE
       NAP    1,TIMADV

       END
