       OPT    NOL
       LIB    RRF
       OPT    LIS
       STTL LINKY MARQUEE EFFECT
*
*RAM AREA
*
 ORG RMRAM
UPPER RMB 1
LOWER RMB 1
LEFT RMB 1
RIGHT RMB 1
HFLAV RMB 1
LFLAV RMB 1
*
*VECTORS
*
       ORG    RMORG 
 JMP ENDWAV

ENDWAV PULS   D              GET RETURN
       LDU    CRPROC
       STD    PD,U           SAVE PROCESS GUY
       MAKP   WCPROC         START THE COLOR PROCESS

       LDA    #$EF

FR33   LDX    #$3B80         UPPER LEFT AT STAT
       LDY    #$5A82         LOWER RIGHT AT START

FR22   STX    PD+2,U
       STY    PD+4,U 
       STA    PD+6,U

       NAP    1,FR11   THE SLEEPER

FR11   LDX    PD+2,U
       LDY    PD+4,U
       LDA    PD+6,U

       LDB    #2       INTERATIONS UNTIL SLEEP
       STB    PD+7,U   COUNTER

FR0    JSR    MARQ
       TSTA
       BEQ    FR1D     IF ZERO...NO CHANGE
       CMPA   #$12
       BNE    FR1C
       LDA    #$EF
       BRA    FR1D
FR1C   CMPA   #$F1
       BNE    FR1B
       LDA    #$DE
       BRA    FR1D
FR1B   CMPA   #$23           STUPID O1
       BNE    FR1A
       LDA    #$F1
       BRA    FR1D
FR1A   SUBA   #$22
FR1D   CMPX   #$616          DID WE JUST DO CENTER
       BEQ    FR2            YEP
       LEAX   -$102,X
       LEAY   $102,Y
       DEC    PD+7,U
       BNE    FR0
       BRA    FR22           SLEEP TIME

*      LDB    TIMER          SEE IF WE ARE WANTED??
*      BNE    FR22           TAKE A NAP
*      BRA    FR0            GET ANOTHER FRAME IN

FR2    TSTA                  DONE??
       BEQ    SUCTIM
       CLRA
       BRA    FR33           AND JUST CLEAR 'EM OUT

SUCTIM JMP    [PD,U]         RETURN!

TABTAB FDB    DECTAB,$1F00   PRIMRIES TO SECONDARIES
       FDB    DECT1,$3F00    PRI - SEC - WHITE
       FDB    DECT2,$3F00
       FDB    BRTTAB,$F00
       FDB    DCATAB,$F00
       FDB    BPRTAB,$F00
       FDB    GYRTAB,$F00
       FDB    BFTAB,$1F00
       FDB    CLLTAB,$1F00
       FDB    LOGTAB,$F00
       FDB    BFTAB2,$1F00
       FDB    DECTB4,$1F00

BRTTAB FCB 1,2,3,4,5,6,7,$F,$17,$1F,$2D,$34
       FCB $3A,$7A,$BA,$FA,$F8,$F0,$E0,$D0,$C0,$C0
       FCB    0,0

DCATAB FCB $C0,$C0,$D0,$E0,$F0,$F8,$FA,$BA,$7A,$3A
 FCB $34,$2D,$1F,$17,$F,7,6,5,4,3,2,1,0,0

BPRTAB FCB $C0,$C1,$C2,$C3,$C4,$C5,$C6,$C7,$87,$87,$47,$47
 FCB $07,$07,$47,$47,$87,$87,$C7,$C7,$C6,$C5,$C4,$C3
 FCB $C2,$C1,0,0

GYRTAB FCB $38,$38,$31,$3A,$3B,$3C,$2D,$2E,$2F,$27
 FCB $1F,$17,$17,$0F,$07,$07,$0F,$17,$17,$1F,$27,$2F
 FCB $2E,$2D,$2C,$3B,$3A,$39,0,0

BFTAB2 FCB $38,$39,$3A,$3B,$3C
 FCB $3D,$3E,$3F,$37,$2F,$27,$17,$0F
 FCB $07,$06,$05,$04,$03,$02
 FCB $01,$01,$01,$49,$CA,$DA,$E8,$F8
 FCB $F9,$FA,$FB,$FD,$FF,$BF,$3F,$3E
 FCB $C0,$C0,$C0,$07,$07,$38,$38,$38
 FCB $07,$C0,$38,$FF,$FF
 FCB 0,0

BFTAB FCB $38,$39,$3A,$3B,$3C
 FCB $3D,$3E,$3F,$37,$2F,$27,$1F,$17
 FCB $47,$47,$87,$87,$C7,$C7
 FCB $C6,$C5,$CC,$CB,$CA,$DA,$E8,$F8
 FCB $F9,$FA,$FB,$FD,$FF,$BF,$3F,$3E
 FCB $C0,$C0,$C0,$07,$07,$38,$38,$38
 FCB $07,$C0,$38,$FF,$FF
 FCB 0,0

CLLTAB FCB $38,$39,$3A,$3B,$3C
 FCB $3D,$3E,$3F,$37,$2F,$27,$1F,$17
 FCB $47,$47,$87,$87,$C7,$C7
 FCB $C6,$C5,$CC,$CB,$CA,$DA,$E8,$F8
 FCB $F9,$FA,$FB,$FD,$FF,$BF,$3F,$3E
 FCB $3C
 FCB 0,0

LOGTAB FCB $37,$2F,$27,$1F,$17,$47,$47,$87
 FCB $87,$C7,$C7,$C6,$C5,$CC,$CB,$CA
 FCB $C0,$D0,$98,$38,$33
 FCB 0,0

DECT2  FCB    $07,$0F,$17,$1F,$27,$2F,$37,$3F
       FCB    $3F,$7F,$7F,$BF,$BF,$FF,$FF,$FF,$BF,$BF
       FCB    $7F,$7F,$3F,$3F
       FCB    $3E,$3D,$3C,$3B,$3A,$39,$38,$38
       FCB    $30,$28,$20,$08,$08,$49         
       FCB    $52,$A5,$FB,$FC,$FD,$FE,$FF,$FF
       FCB    $FE,$FD,$FC,$FB,$FA,$F9,$F8
       FCB    $F0,$E8
       FCB    $E0,$D8,$D0,$C8,$C0,$80,$40,$01
       FCB    $01,$01,$01,$02,$03                 
       FCB    $04,$05,$06,$4F,$EF,$F7,$FF,$FF
       FCB    $F7,$EF,$E7,$DF,$D7,$CF,$C7
       FCB    $87,$87,$47
       FCB    $47,$7,0,0

DECT1  FCB    $07,$0F,$17,$1F,$27,$2F,$37,$3F
       FCB    $3F,$7F,$7F,$BF,$BF,$FF,$FF,$FF,$BF,$BF
       FCB    $7F,$7F,$3F,$3F
       FCB    $3E,$3D,$3C,$3B,$3A,$39,$38,$38
       FCB    $78,$78,$B8,$B8,$F8,$F8         
       FCB    $F9,$FA,$FB,$FC,$FD,$FE,$FF,$FF
       FCB    $FE,$FD,$FC,$FB,$FA,$F9,$F8
       FCB    $F0,$E8
       FCB    $E0,$D8,$D0,$C8,$C0,$C1,$C2,$C3
       FCB    $C4,$C5,$C6,$C7,$C7                 
       FCB    $CF,$D7,$DF,$E7,$EF,$F7,$FF,$FF
       FCB    $F7,$EF,$E7,$DF,$D7,$CF,$C7
       FCB    $87,$87,$47
       FCB    $47,$7,0,0

DECTB4 FCB    $07,$0F,$17,$1F,$27,$2F,$37,$3F
       FCB    $3E,$3D,$3C,$3B,$3A,$39,$38,$38
       FCB    $78,$78,$B8,$B8,$F8,$F8,$F0,$E8 
       FCB    $E0,$D8,$D0,$C8,$C0,$80,$41,$01
       FCB    $01,$02,$03,$04,$05,$06,$07,$07
       FCB    $07,$7,0,0

DECTAB FCB    $07,$0F,$17,$1F,$27,$2F,$37,$3F
       FCB    $3E,$3D,$3C,$3B,$3A,$39,$38,$38
       FCB    $78,$78,$B8,$B8,$F8,$F8,$F0,$E8 
       FCB    $E0,$D8,$D0,$C8,$C0,$C1,$C2,$C3
       FCB    $C4,$C5,$C6,$C7,$C7,$87,$87,$47
       FCB    $47,$7,0,0

WCPROC LDY    #TABTAB
PCCALL JSR    RAND           GET 
       ANDA   #$F            LOOK AT LOW
       CMPA   #12            0-11 PLEASE
       BHS    PCCALL
       ASLA
       ASLA                  4 PER
       TFR    A,B
       LDX    B,Y            GET THE POINTER
       ADDB   #2             POINT AT MASK
       JSR    RAND
       ANDA   B,Y            MASK IT
       STX    PD,U           SAVE START
       LEAX   A,X
       STX    PD+2,U         SAVE IT

PCSUB  LDX    PD+2,U         GET THE POINTER INTO OUR THING
       LEAX   1,X
       TST    ,X             END OF LIST
       BNE    PCCS1
       LDX    PD,U           START OVER
PCCS1  STX    PD+2,U         MOVE TO NEXT

RCS1   LDY    #PCRAM+1       FOR COLOR MATRIX
RCS3   LDA    ,X+            GET A BYTE
       BNE    RCS2
       LDX    PD,U           START TABLE OVER
       BRA    RCS3
RCS2   STA    ,Y+            STORE IT
       CMPY   #PCRAM+16
       BLO    RCS3           AND LOOP

       LDB    #$00           MOVING BLACK
       LDA    PD+4,U
       DECA
       CMPA   #$5 
       BLO    PCS7
       LDA    #4
PCS7   STA    PD+4,U         OFFSETTER
       LDX    #PCRAM+1
PCS4   STB    A,X
       LEAX   5,X
       CMPX   #PCRAM+16
       BLO    PCS4
       NAP    1,PCSUB

*      MARQ - DRAW A LINKY MARQUIS.  PASS UPPER LEFT IN X.
*      PASS LOWER RIGHT IN Y.  
*      RANGES ARE INCLUSIVE.  LOWER-UPPER MUST BE EVEN (ODD INCLUSIVE)
*      A CONTAINS THE HIGH AND LOW FLAVOR COLORS (<OUTER><INNER>)

MARQ   PSHS   X,Y,D,U
       ANDA   #$F0
       STA    HFLAV    SAVE FLAVOR COLORS
       LDA    ,S
       ANDA   #$0F
       STA    LFLAV
       TFR    X,D
       STA    LEFT
       STB    UPPER
       TFR    Y,D
       STA    RIGHT
       STB    LOWER
       SUBB   UPPER          LOWER-UPPER ODD???
       RORB                  CHECK
       BCC    MARQ1          NOPE
       DEC    LOWER          NOT THAT LOW...OK?

MARQ1  JSR    GHIGH          GET HIGH DUPED  BYTE IN B 
       LDA    UPPER          GET THE UPPER 
       JSR    HHIGH          HIGH PART OF TOP DMASIZONTAL LINE
       INCA
       JSR    GLOW
       BSR    HLOW           LOW PART
*
       LDA    LOWER
       JSR    GHIGH
       BSR    HHIGH          1 LESS FOR GEN PURP
       DECA
       JSR    GLOW
       BSR    HLOW 
*
       LDA    LEFT           LEFT BYTE
       JSR    GNORM
       BSR    VHIGH          HIGH FLAVOR STUFF IN A VERT LINE
*
       LDA    RIGHT
       JSR    GFLIP
       BSR    VHIGH

MAR11  PULS   U,D,X,Y,PC

*      VHIGH - PLOT VERTICAL LINE FROM [UPPER TO LOWER]
*      CONSISTING OF EVERY OTHER BYTE GETTING THE HIGH NIBBLE COLOR
*      A HAS THE X TO USE

VHIGH  PSHS   D,X,CC
       BSR    SETUPV   SETUP X,B
       SEI
       STA    DMACON    STORE CONSTANT
       LDA    #5       1 WIDE, B HIGH
       EORB   #4
       STD    DMASIZ
       LDD    #0
       STD    DMAORG     FROM NOWHERE
       STX    DMADES     DESTINATION PEGGED
       LDA    #$12     WRITE THE DMACONANT LINES
       STA    DMACTL
       PULS   CC,D,X,PC

SETUPV PSHS   B
       LDB    UPPER    GET UPPER Y
       INCB            START 1 AFTER
       TFR    D,X      THIS IS PONTER
       LDB    LOWER
       SUBB   UPPER    THIS IS GUARANTEED EVEN
       DECB            EXCLUSIVE PLEASE
       PULS   A,PC

*      HHIGH - THIS ROUTINE PLOTS HIGH COLOR IN THE HIGH HALF OF
*      CONSECUTIVE DMASIZONTAL PIXELS GOING FROM 
*      [LEFT TO RIGHT] AT THE HEIGHT GIVEN IN REGISTER A.

HHIGH  PSHS   X,D,CC
       BSR    SETUPH   SETUP COUNT IN B, X POINTS AT FIRST BYTE
       INCB            ADD 1 TO MAKE IT INCLUSIVE
HHIGH1 SEI
       STA    DMACON    STORE THE CONSTANT
       EORB   #4
       STB    DMASIZ
       LDA    #5
       STA    DMASIZ+1  1 HIGH
       STD    DMAORG     GOOD ORIGINATION
       STX    DMADES
       LDA    #$12
       STA    DMACTL
       PULS   X,D,PC,CC

HLOW   PSHS   X,D,CC
       BSR    SETUPH   SETUP
       DECB            EXCLUSIVE
       LEAX   $100,X   MOVE OVER 1
       BRA    HHIGH1   AND GO

SETUPH PSHS   B        SAVE COLOR
       TFR    A,B      MOVE Y OVER
       LDA    LEFT     GET LEFT X
       TFR    D,X      MAKE POINTER
       LDB    RIGHT
       SUBB   LEFT     FIND DIFFERENCE (EXCLUSIVE
       PULS   A,PC     AND RETURN

GHIGH  LDB    HFLAV    GET HIGH FLAVOR
       LSRB
       LSRB
       LSRB
       LSRB
       ORAB   HFLAV    HIGH FLAVOR IN BOTH
       RTS

GLOW   LDB    LFLAV
       ASLB
       ASLB
       ASLB
       ASLB
       ORAB   LFLAV
       RTS

GNORM  LDB    HFLAV
       ORAB   LFLAV
       RTS

GFLIP  LDB    HFLAV
       LSRB
       LSRB
       LSRB
       LSRB
       PSHS   B
       LDB    LFLAV
       ASLB
       ASLB
       ASLB
       ASLB
       ORAB   ,S+
       RTS

       END
