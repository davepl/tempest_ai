       OPT    NOL
       LIB    RRFRED
       OPT    LIS

       ORG    TSCORG   PLACE TO RUN FROM
       JMP    CKHS     CHECK HIGH SCORES (POWER UP)
       JMP    ENDGAM   END OF GAME HIGH SCORE PROCESSING
       JMP    GODENT   ENTER GOD
       JMP    RESHSC   RESET HIGH SCORES
       JMP    HSCHK    CHECK HIGH SCORES AFTER FAC SET RESTORE
       JMP    HSBUT    HIGH SCORE RESET BUTTON
*
HSBUT  LDA    #$18     3 SECONDS REQUIRED
       STA    PD,U
       LDA    #$3F
       STA    SOUND

HSBUT1 NAP    8,HSBUT2 CHECK BUTTON
HSBUT2 LDA    PIA0
       BITA   #8       STILL PRESSED??
       BEQ    HSBUT3   NOPE....BYE
       DEC    PD,U
       BNE    HSBUT1   NOT ENOUGH TIME YET
       LDY    #GODINT  JUST PUT IN DEFAULT NAME
       LDX    #DEFHSR
       LDB    #23      NOT THE SCORE, JUST THE NAMES
       JSR    CMSMVV
       JSR    MKGODC   MAKE THE NEW CHECK BYTE
       CLR    SOUND    INDICATE MISSION ACCOMPLISHED
HSBUT3 JMP    SUCIDE

*
HSCHK  LDY    #CMSCOR  START AT BEGINNING OF TABLE
       LDB    #8       8 BAD ENTRIES IS ADEQUATE FOR FULL BASH
HSCHK1 JSR    FSCCK    FORM THE CHECK BYTE
       EORA   6,Y      COMPARE TO CHECK BYTE THERE
       ANDA   #$0F
       BEQ    HSCHK2
       DECB
       BEQ    RESHSC    8 BAD ONES..RESET
HSCHK2 LDA    #WDATA
       STA    WDOG
       LEAY   SCRSIZ,Y
       CMPY   #TODAYS  BEYOND REGION??
       BLO    HSCHK1
       RTS
*
***    RESHSC          RESET HIGH SCORES
*
RESHSC LDA    #WDATA
       STA    WDOG
       LDX    #DEFHSR  DEFAULT SCORES
       LDY    #GODINT  GODS INITIALS FOLLOWED BY GODS SCORE FOLLOWED BY REST
       LDB    #CDEFS1  SIZE OF DEFAULT TABLE
       JSR    CMSMVV   TRANSFER THE BLOCK
       LDX    #DEFSC2
       LDY    #GODINT+(2*CDEFS1)
       LDB    #CDEFS2
       JSR    CMSMVV
       JSR    MKGODC   
       LDY    #CMSCOR  WALK THROUGH
RESHS1 JSR    MKSCCK
       LDA    #WDATA
       STA    WDOG
       LEAY   SCRSIZ,Y
       CMPY   #TODAYS
       BLO    RESHS1
       LDA    #HSRESM     SAY THAT THEY WERE RESET
       JMP    WRD7V       AND RETURN

DEFHSR FCC    'BIL'
DEFGOD FCC    'WILLY'  
       FCB    $3A
       FCC    'ELKTRIX'
       FCB    $3A,$3A,$3A,$3A,$3A,$3A,$3A
       FCB    $00,$15,$17,$82
       FCC    'VID'
       FCB    $00,$12,$21,$45
       FCC    'KID'
       FCB    $00,$12,$21,$35
TODTAB FCC    'DRJ'
       FCB    $00,$05,$21,$27
       FCC    'LED'
       FCB    $00,$05,$02,$18
       FCC    'EPJ'
       FCB    $00,$04,$12,$55
       FCC    'JER'
       FCB    $00,$04,$12,$50
       FCC    'KID'
       FCB    $00,$03,$19,$20
       FCC    'MLG'
       FCB    $00,$03,$19,$19
       FCC    'SSR'
       FCB    $00,$02,$66,$45
       FCC    'UNA'
       FCB    $00,$02,$66,$35
       FCC    'JRS'
       FCB    $00,$02,$52,$50 
       FCC    'CJM'        
       FCB    $00,$02,$41,$10                 
ENDTOD FCC    'KJF'
       FCB    $00,$02,$30,$20
       FCC    'MRS'
       FCB    $00,$02,$20,$35
       FCC    'PGD'
       FCB    $00,$02,$10,$90
       FCC     'NJM'
       FCB     $00,$02,$09,$65
       FCC     'NHD'
       FCB     $00,$02,$09,$60
CDEFS1 EQU    *-DEFHSR
DEFSC2 FCC     'DON'
       FCB     $00,$01,$82,$80
       FCC     'VIV'
       FCB     $00,$01,$82,$80
       FCC    'GWW'
       FCB    $00,$01,$81,$05
       FCC    'CRB'
       FCB    $00,$01,$80,$55
       FCC    'MDR'
       FCB    $00,$01,$75,$65
       FCC    'BAC'
       FCB    $00,$01,$72,$56
       FCC    'W'
       FCB    $3A
       FCC    'R'
       FCB     $00,$01,$70,$70
       FCC     'MPT'
       FCB     $00,$01,$60,$60
       FCC     'SUE'
       FCB     $00,$01,$55,$20
       FCC     'MOM'
       FCB     $00,$01,$44,$80
       FCC     'DAD'
       FCB     $00,$01,$44,$79
       FCC     'SFD'
       FCB     $00,$01,$44,$78
       FCC     'AKD'
       FCB     $00,$01,$44,$77
       FCC     'CWK'
       FCB     $00,$01,$33,$30
       FCC     'TMH'
       FCB     $00,$01,$32,$70
       FCC     'EJS'
       FCB     $00,$01,$31,$20
       FCC     'RAY'
       FCB     $00,$01,$30,$65
       FCC     'GAY'
       FCB     $00,$01,$29,$65
       FCC     'RKM'
       FCB     $00,$01,$28,$55
       FCC     'CNS'
       FCB     $00,$01,$27,$55
*
CDEFS2 EQU    *-DEFSC2          LENGTH OF TABLE
*
NULSCR FCB    $3A,$3A,$3A
       FCB    $00,$01,$00,$00
NULSIZ EQU    *-NULSCR

HSTUNE FCB    $C0,1,1,$1B,1,1,2,0      PLEBIAN HIGH     
GODSND EQU    *
ULTUNE FCB    $C0             UPPER LEFT OR GOD              
       FCB    1,$FF,$3C
       FCB    1,$FF,$00      CONTINUE AIR-RAID
       FCB    1,$20,00
       FCB    1,$C0,$36      FALLING PLANE
       FCB    1,$60,$3D      FALLING BOMB
       FCB    2,$A,$11       2 THUNDER CRACKS
       FCB    2,$40,$3E,0    AND THE AFTERSHOCK

CLRSCR PSHS   X,Y,B          SAVE BASHED
       LDX    #NULSCR        POINT AT NULL SCORE
       LDB    #NULSIZ        HALF AN ENTRYS WORTH OF BYTES
       JSR    CMSMVV
       PULS   X,Y,B,PC

MKGODC PSHS   A    
       BSR    FGODC          FOR GODS CHECK BYTE  IN LOW HALF OF A
       STA    GODSCR+40      STORE AFTER THE CHARACTERS
       PULS   A,PC

*
**     FORM GOD'S CHECK BYTE
*

FGODC  PSHS   X 
       LDX    #GODINT        LETS START AT THE VERY BEGINNING....
       CLRA   
FGODC0 ADDA   ,X             ADD THE CURRENT NIBBLE
FGODC1 LEAX   1,X
       CMPX   #GODSCR+40     CHECK BYTE???
       BEQ    FGODC1
       CMPX   #GODSCR+48     DONE???
       BNE    FGODC0
       PULS   X,PC           RETURN IN LOW HALF OF A

MKSCCK PSHS   A  
       BSR    FSCCK          FORM THE CHECK BYTE
       STA    6,Y            STORE IT
       PULS   A,PC

FSCCK  PSHS   Y,B
       LDB    #SCRSIZ
       CLRA
MSCCK0 CMPB   #SCRSIZ-6      ABOUT TO DO CHECK BYTE??
       BEQ    MSCCK1
       ADDA   ,Y             ADD THE CURRENT BYTE
MSCCK1 LEAY   1,Y
       DECB
       BNE    MSCCK0
       PULS   Y,B,PC

CKHS   LDA    #50            DONT GO THROUGH MORE THAN 50
       PSHS   A
       LDY    #CMSCOR        WALK THROUGH SCORES
CKHS7  BSR    FSCCK          FORM THE SCORE CHECK BYTE
       EORA   6,Y
       ANDA   #$F
       BEQ    CKHS5
CKHS4  JSR    RMENTR         REMOVE THE ENTRY
       CLR    CREDST
       CLR    CREDST+1
       DEC    ,S             MAKE SURE WE DON'T REMOVE TOO MANY
       BEQ    CKHS67
       BRA    CKHS7          DON'T PUSH POINTER YET.
CKHS5  LDA    #3
       LDB    #4
       BSR    CKENT
       BCS    CKHS4
CKHS6  LEAY   SCRSIZ,Y       MOVE TO NEXT
       CMPY   #TODAYS
       BLO    CKHS7
CKHS67 PULS   A              TAKE ENTRY OFF OF STACK
       LDX    #TODTAB        TODAY'S TABLE (NORMAL BYTES)
       LDY    #TODAYS
       LDB    #ENDTOD-TODTAB
       JSR    CMSMVV
       BSR    FGODC          NOW CHECK OUT GOD.         
       EORA   GODSCR+40      CHECK IT OUT
       ANDA   #$F
       BEQ    CKHS1          OK..CONTINUE
       BSR    REMGOD
CKHS1  LDY    #GODINT        BEGINNING OF TEXT ONLY PART-CHECK ALPHA NUM
       LDA    #23            20 CHARS OF TEXT
       LDB    #4             4 BYTES OF NUMBERS PLEASE
       BSR    CKENT          MAKE SURE ALPHAS AND NUMERICS
       BCC    CKHS3
       BSR    REMGOD         REMOVE GOD
CKHS3  RTS

REMGOD LDX    #GODSCR        GOD BAD FIRST BLANK OUT
       LDA    #$3A 
CKHS2Z JSR    WCMOSA
       CMPX   #GODSCR+40     DONE??
       BLO    CKHS2Z         DO ANOTHER CHARACTER
       LDX    #CMSCOR        #2           
       LDY    #GODSCR
       LDA    #6             6 NIBBLES
       JSR    BLKMOV         MOVED
       LDY    #GODINT        X STILL POINTS AT FIRST GUYS INITIALS
       JSR    BLKMOV         MOV EM TOO
       LDX    #CMSCOR+6      #2 SCORE
       LDY    #GODSCR+40     #2 SCORE GOES HERE
       LDA    #8
       JSR    BLKMOV
       JSR    MKGODC         MAKE THE CHECK BYTE
       LDY    #CMSCOR        NOW REMEOVE #2
       JMP    RMENTR         REMOVE #2 AND RETURN

CKENT  PSHS   D,X
       LDB    #WDATA
       STB    WDOG
       TFR    Y,X            CMOSABLE REGISTER
CKENT1 JSR    RCMOSB         READ A BYTE
       CMPB   #$41           LOWER THAN A??
       BHS    CKENT4         NOPE...HIGH ENOUGH CHECK TOO HIGH
       CMPB   #$3A           TOO LOW..CHECK SPACE
       BNE    CKENT5         NOPE...NOT VALID
CKENT4 CMPB   #$5A
       BHI    CKENT5         HIGHER THAN Z
       DECA                  ONE LESS ON THIS FRONT
       BNE    CKENT1
       LDA    1,S            GET THE SECOND PART COUNTER BACK
       JSR    RCMOSB         GET THE CHECK BYTE
       ANDB   #$F
       CMPB   #9
       BHI    CKENT5         NO HEX ALLOWED
       DECA
CKENT2 JSR    RCMOSB
       PSHS   B
       ANDB   #$F            LOOK AT LOW HALF
       CMPB   #9
       PULS   B
       BHI    CKENT5         NOT NUMERIC...ERROR!
       ANDB   #$F0
       CMPB   #$99
       BHI    CKENT5
       DECA
       BNE    CKENT2
CKENT3 CLC
CKENT8 LDA    #WDATA
       STA    WDOG
       PULS   X,D,PC
CKENT5 SEC                            
       BRA    CKENT8   

*
***    RMENTR - REMOVE SCORE ENTRY POINTED TO BY Y.
*

RMENTR PSHS   X,Y,D
       LEAX    SCRSIZ,Y       X POINTS PAST Y
RMENT0 CMPX   #TODAYS        ARE WE BEYOND IN X.
       BHS    RMENT1         YEP...DONE.
       LDA    #SCRSIZ
       BSR    BLKMOV         MOVE THE BLOCK X TO Y
       LEAY   SCRSIZ,Y
       LEAX   SCRSIZ,X
       LDA    #WDATA
       STA    WDOG
       BRA    RMENT0
RMENT1 JSR    CLRSCR         CLEAR THE BLOCK NOW POINTED AT BY Y (BOTTOM)
       JSR    MKSCCK         AND FORM THE CHECK BYTE
       PULS   X,Y,D,PC

*      BLKMOV  A BYTES FROM [X] TO [Y]

BLKMOV PSHS   X,Y,D
BLKMV1 LDB    ,X+
       STB    ,Y+
       DECA
       BNE    BLKMV1
       PULS   X,Y,D,PC

ENDGAM JSR    GNCIDE         KILL COLOR PROCS ETC.
       JSR    SCRCLR
       JSR    CRINIT         PRETTY COLORS PLEASE
       JSR    P1SW           POINT AT PLAYER 1
       LDX    #ZP1SCR        PLAYER 1'S SCORE
       LDB    #1             PLAYER '1'
       BSR    EGSUB          DO IT UP FOR HIM
       LDB    PLRCNT         HOW MANY???
       DECB
       BEQ    ENDGMR         RETURN.
       JSR    SCRCLR
       LDA    PIA3           COCKTAIL??
       BPL    ENDGM1         NOPE
       JSR    P2SW           THEN FACE THE NATION
ENDGM1 LDX    #ZP2SCR
       LDB    #2
       BSR    EGSUB
       JSR    SCRCLR
       JSR    P1SW           AND SWITCH TO PLAYER 1
ENDGMR JMP    GOV            JUMP TO GAME OVER (AMODE) PROCESSING

EGSUB  PULS   Y              GET RETURN ADDRESS
       STY    EGRAM
       STX    PLRX           SAVE PARAMS
       STB    CURPLR
       JSR    GODCHK         SEE IF GOD
       BCC    EGSUB1
       JSR    GODESB         HAVE HIM ENTER IT.
       LDY    #CMSCOR        BUBBLE TO VACATE THIS POSITION
       LDX    #TODAYS-SCRSIZ FROM BOTTOM 
       JSR    BUBDN          BUBBLE EM DOWN
       LDX    #GODSCR+40     POINT AT SOON TO BE EX-GOD
       LDY    #CMSCOR+6      THIS IS WHERE IT GOES
       LDA    #8             8 NIBBLES
       JSR    BLKMOV         MOVE IT DOWN
       LDX    #GODINT        AND HIS INITIALS
       LDY    #CMSCOR
       LDA    #6
       JSR    BLKMOV         AND MOVE IT
       LDY    #CMSCOR        POINT AT #2 
       JSR    MKSCCK         AND FORM HIS CHECK BYTE
       LDX    PLRX           POINT AT SCORE
       LDY    #GODSCR+40
       LDB    #4
       JSR    CMSMVV
       JSR    MKGODC
       LDX    #NULSCR    BLANK INITIALS
       LDY    #GODINT  
       LDB    #3
       JSR    CMSMVV
       JSR    SCRCLR
       LDA    #NOWMES        TELL HIM TO ENTER HIS INITIALS NOW.
       STA    EGRAM2         INDICATE GOD JUST ENTERED
       LDX    #GA2           CHECK NUMBER OF INITIALS
       JSR    RCMOSB
       CMPB   #3
       BEQ    GET333         THEN USE INITIALS AS INITIALS
       BRA    GETHM2         JUMP INTO THE ROUTINE TO DO THAT.
*                     CODE TO GET GOD HERE.
EGSUB1 JSR    TODCHK         CHECK TODAYS HIGH
       BCS    GETHIM
       JSR    ALLCHK         CHECK ALL TIME
       BCS    GETHIM
       JMP    [EGRAM]        RETURN...HOW BORING.
*
GETHIM CLR    EGRAM2
       LDD    #HSTUNE        HIGH SCORE TUNE
       CMPY   #TODAYS        IS IT POINTING AT THE HIGHEST
       BNE    GETHM1         NOPE
       LDD    #ULTUNE
GETHM1 JSR    SNDLDS
       LDA    #CONG
GETHM2 LDB    CURPLR         GET PLAYER NUMBER
       JSR    SCRCLR
       JSR    WRD7V          PUT THE MESSAGE OUT
       LDD    #$3A3A         SPACES
       STD    ALTBL
       STA    ALTBL+2
       LDD    #$300          3 CHARS PLEASE, NO FUNNY ONES
       LDX    #$4680         PLACE TO DO IT
       LDY    #ALTBL         PLACE FOR HS.
       JSR    GETLT          
GET333 JSR    TODCHK         SEE WHERE TO PUT IT ETC.
       BCC    GETHM3         NOT THIS TIME JACK
       LDX    #TODEND-SCRSIZ END OF TODAYS
       JSR    SCTRNS
GETHM3 JSR    ALLCHK         CHECK FOR ALL TIME
       BCC    GETHM5         NOPE
       TST    EGRAM2         WAS THIS GOD??
       BEQ    GETHM4         NOPE
       LDX    #ALTBL
       LDY    #GODINT        JUST STORE THE INITIALS ON THIS SIDE
       LDB    #3
       JSR    CMSMVV         DONE.
       JSR    MKGODC         FORM GOD'S NEW CHECK BYTE
       LDA    #5             INTERESTED IF THERE ARE 5 OTHERS!
       BSR    SETBZZ         WELL????
       BCC    GETHM5         NOPE.
       TFR    X,Y            MOVE LOWEST POINTED TO INTO Y
       JSR    RMENTR         REMOVE IT.
       BRA    GTTHM8         AND TELL HIM IT WAS REMOVED
GETHM4 JSR    SETBOT         FIND PLACE TO BUBBLE DOWN TO (5 MAX RULE)
       PSHS   CC             SAVE ANSWER
       PSHS   X    
       CMPY   ,S++           
       BHI    GETHM8         NOT BETTER THAN HIS TOP 5
       BSR    SCTRNS         DONE.
GETHM8 PULS   CC             GET CARRY BACK
       BCC    GETHM5
GTTHM8 JSR    SCRCLR 
       LDA    #ONLY5M
       JSR    WRD7V
       NAP    $60,GETHM5
GETHM5 JMP    [EGRAM]

SETBZZ PSHS   Y,D
       BRA    SETBT0

SETBOT PSHS   Y,D
       LDX    #GODINT        SEE IF MATCH OF GOD INITIALS
       BSR    SETSUB
       LDA    #4             ASSUME MATCH
       BCS    SETBT0         YEP...ONLY 4 MORE NEEDED
       INCA
SETBT0 STA    XTEMP
       LDX    #CMSCOR        POINT AT FIRST ENTRY
SETBT1 BSR    SETSUB         SEE IF THIS ONE MATCHES
       BCC    SETBT3         NOPE...
       DEC    XTEMP          FOUND
       BEQ    SETBT5         AND TIME TO EXIT
SETBT3 LEAX   SCRSIZ,X       PUSH TO NEXT
       CMPX   #TODAYS
       BLO    SETBT1         NOT DONE...DO ANOTHER
       LDX    #TODAYS-SCRSIZ BUBBLE FROM BOTTOM
       CLC
       PULS   Y,D,PC
SETBT5 SEC
       PULS   Y,D,PC         AND RETURN IT

SETSUB PSHS   X
       LDY    #ALTBL         ALT TABLE
       LDB    #3
SETSB2 JSR    RCMOSA   
       CMPA   ,Y+            COMPARE TO ALT LETTER
       BNE    SETSB1         NO GOOD...MOVE TO NEXT
       DECB                  ONE LESS TO DO
       BNE    SETSB2
       SEC
       PULS   X,PC
SETSB1 CLC                   NO MATCH
       PULS   X,PC

SCTRNS PSHS   Y
       JSR    BUBDN          BUBBLE EM TO MAKE ROOM
       LDX    #ALTBL
       LDB    #3
       JSR    CMSMVV         TRANSFER THE BLOCK
       LDX    PLRX
       LDB    #4
       JSR    CMSMVV         TRANSFER THE SCORE
       PULS   Y
       JMP    MKSCCK         FORM THE CHECK BYTE

BUBDN  PSHS   X,Y
       TFR    X,Y            BOTTOM IS DESTINATION
BUBDN1 CMPY   2,S            IS OUR DESTINATION WHAT WE WANT TO FILL??
       BEQ    BUBDUN         THEN WE'RE DONE
       LEAX   -SCRSIZ,Y      X IS SOURCE
       LDA    #SCRSIZ        THAT MANY BYTES
       JSR    BLKMOV         MOVE THE BLOCK
       LEAY   -SCRSIZ,Y
       BRA    BUBDN1
BUBDUN PULS   X,Y,PC         BYE

GODENT PULS   D              RETURN
       STD    EGRAM
       LDB    #1
       BSR    GODESB         HAVE THE GUY ENTER IT
       JSR    MKGODC         MAKE THE CHECK BYTE AND RETURN
       JMP    [EGRAM]

GODESB PULS   Y              GET RETURN ADDRESS
       STY    EGRAM2         SAVE IT
       CLRA
       TFR    D,Y            PASS PLAYER IN Y             
       LDD    #GODSND
       JSR    SNDLDS         MAKE THE SOUND (TUNE)
       LDX    #GA2           GET GOD LETTERS
       JSR    RCMOSB
       JSR    SCRCLR
       CMPB   #3             ONLY 3???
       BNE    GMANY          NO
       LDB    CURPLR         GET PLAYER
       LDA    #CONG
       JSR    WRD7V          PRINT THE NORMAL MESSAGE
       LDA    #3
       LDX    #$4680
       BRA    GTHREE
GMANY  LDA    #GODMES        
       JSR    WRD7V          PRINT OUT CONGRATS
       TFR    B,A            MAX LETTERS IN A NOW
       JSR    NEWBCH         CALL THE RIGHT ONE
       LDX    #$2D80         PLACE TO DO IT
GTHREE LDY    #ALTBL+20      PLACE TO STORE THE CHARS
       LDB    #$3A           PUT SPACES IN
GODES1 STB    ,-Y
       CMPY   #ALTBL    
       BHI    GODES1
       CLRB                  ALPHA ONLY
       JSR    GETLT          ENTER THE STUFF
       LDX    #ALTBL
       LDY    #GODSCR
       LDB    #20
       JSR    CMSMVV
       JMP    [EGRAM2]       RETURN

GODCHK PSHS   X,Y
       LDY    #GODSCR+40    POINT AT THE SCORE PART
       LDX    PLRX           POINT AT HIS SCORE
       BSR    CMPSCR        COMPARE SCORES
       PULS   X,Y,PC         RETURN THE CARRY BIT

TODCHK PSHS   X              RETURN ENTRY POINTER IF OK
       LDY    #TODAYS+6 
       LDX    PLRX
TODCK1 BSR    CMPSCR         COMPARE
       BCS    TODBYE
       LEAY   SCRSIZ,Y       PUSH TO NEXT
       CMPY   #TODEND        BEYOND
       BLO    TODCK1
       CLC
       PULS   X,PC
TODBYE LEAY   -6,Y           PUSH BACK TO BEGINNING OF ENTRY TO REPLACE
       PULS   X,PC           AND RETURN WITH CARRY SET

ALLCHK PSHS   X
       LDY    #GODSCR+40     POINT AT SCORE PART OF GOD ENTRY
       LDX    PLRX
ALCK1  BSR    CMPSCR
       BCS    TODBYE
       LEAY   SCRSIZ,Y
       CMPY   #TODAYS-SCRSIZ BEYOND VISIBLES??????
       BLO    ALCK1          NOPE..CONTINUE
       CLC
       PULS   X,PC           NOT GOOD ENOUGH

CMPSCR PSHS   Y,X,D
       EXG    X,Y            MAKE X POINT AT CMOS
       LDB    #4             4 BYTES THROUGH
CMPSC1 JSR    RCMOSA         READ A BYTE
       CMPB   #4             FIRST TIME THROUGH??
       BNE    CMPSC2
       ANDA   #$0F           THEN TAKE AWAY CHECK BYTE
CMPSC2 CMPA   ,Y+            COMPARE TO PLAYERS SCORE
       BHI    CMPSC4         CMOS SCORE HIGHER
       BLO    CMPSC5         PLAYER'S SCORE HIGHER
       DECB                  SAME...ARE WE DONE??
       BNE    CMPSC1         NOPE...COMPARE ANOTHER BYTE
CMPSC4 CLC
       PULS   X,Y,D,PC
CMPSC5 SEC                   PLAYER IS HIGHER
       PULS   X,Y,D,PC

SNDLDS CLR    SNDPRI         MAKE SURE WE GET OUT
       JMP    SNDLD

       END
