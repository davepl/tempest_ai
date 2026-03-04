       OPT    NOL
       LIB    RRFRED
       OPT    LIS

       ORG    SETORG   SET US HERE
       FDB    TAB35Z
       FDB    TAB57Z

TAB35Z FDB    C0
       FDB    C1
       FDB    C2
       FDB    C3
       FDB    C4
       FDB    C5
       FDB    C6
       FDB    C7
       FDB    C8
       FDB    C9
       FDB    CSP
       FDB    CEX
       FDB    CCO
       FDB    CPE
       FDB    CAR
       FDB    ACL
       FDB    AHY
       FDB    CA
       FDB    CB
       FDB    CCC
       FDB    CD
       FDB    CE
       FDB    CF
       FDB    CG
       FDB    CH
       FDB    CI
       FDB    CJ
       FDB    CK
       FDB    CL
       FDB    CM
       FDB    CN
       FDB    CO
       FDB    CP
       FDB    CQ
       FDB    CR
       FDB    CS
       FDB    CT
       FDB    CU
       FDB    CV
       FDB    CW
       FDB    CX
       FDB    CY
       FDB    CZ
       FDB    CLP
       FDB    CRP
       FDB    CFRB
       FDB    CSLSH

C0     FCB    3
       FCB    $FF,$F0
       FCB    $F0,$F0
       FCB    $F0,$F0
       FCB    $F0,$F0
       FCB    $FF,$F0

C1     FCB    3
       FCB    $0F,$00 
       FCB    $0F,$00
       FCB    $0F,$00
       FCB    $0F,$00
       FCB    $0F,$00

C2     FCB    3
       FCB    $FF,$F0
       FCB    $00,$F0
       FCB    $FF,$F0
       FCB    $F0,$00 
       FCB    $FF,$F0

C3     FCB    3
       FCB    $FF,$F0
       FCB    $00,$F0
       FCB    $0F,$F0
       FCB    $00,$F0
       FCB    $FF,$F0

C4     FCB    3
       FCB    $F0,$F0
       FCB    $F0,$F0
       FCB    $FF,$F0
       FCB    $00,$F0
       FCB    $00,$F0

C5     FCB    3
       FCB    $FF,$F0
       FCB    $F0,$00
       FCB    $FF,$F0
       FCB    $00,$F0
       FCB    $FF,$F0

C6     FCB    3
       FCB    $FF,$F0
       FCB    $F0,$00
       FCB    $FF,$F0
       FCB    $F0,$F0
       FCB    $FF,$F0

C7     FCB    3
       FCB    $FF,$F0
       FCB    $00,$F0
       FCB    $00,$F0
       FCB    $00,$F0
       FCB    $00,$F0

C8     FCB    3
       FCB    $FF,$F0
       FCB    $F0,$F0
       FCB    $FF,$F0
       FCB    $F0,$F0
       FCB    $FF,$F0

C9     FCB    3
       FCB    $FF,$F0
       FCB    $F0,$F0
       FCB    $FF,$F0
       FCB    $00,$F0
       FCB    $00,$F0

CSP    FCB    1
       FCB    $00
       FCB    $00
       FCB    $00
       FCB    $00
       FCB    $00

CEX    FCB    1
       FCB    $F0        
       FCB    $F0  
       FCB    $F0
       FCB    $00
       FCB    $F0

CCO    FCB    3
       FCB    $00,$00
       FCB    $00,$00
       FCB    $00,$00
       FCB    $FF,$00
       FCB    $0F,$00

CPE    FCB    1
       FCB    $00
       FCB    $00
       FCB    $00
       FCB    $00
       FCB    $F0        

CAR    FCB    5
       FCB    $00,$F0,$00
       FCB    $0F,$FF,$00
       FCB    $FF,$FF,$F0
       FCB    $00,$F0,$00
       FCB    $00,$F0,$00

ACL    FCB    1  
       FCB    $00
       FCB    $F0
       FCB    $00
       FCB    $F0
       FCB    $00

AHY    FCB    3
       FCB    $00,$00
       FCB    $00,$00
       FCB    $FF,$F0
       FCB    $00,$00
       FCB    $00,$00

CA     FCB    3
       FCB    $FF,$F0
       FCB    $F0,$F0
       FCB    $FF,$F0
       FCB    $F0,$F0
       FCB    $F0,$F0

CB     FCB    3
       FCB    $FF,$F0
       FCB    $F0,$F0
       FCB    $FF,$00
       FCB    $F0,$F0
       FCB    $FF,$F0

CCC    FCB    3
       FCB    $FF,$F0
       FCB    $F0,$00
       FCB    $F0,$00
       FCB    $F0,$00
       FCB    $FF,$F0

CD     FCB    3
       FCB    $FF,$00
       FCB    $F0,$F0
       FCB    $F0,$F0
       FCB    $F0,$F0
       FCB    $FF,$00

CE     FCB    3
       FCB    $FF,$F0
       FCB    $F0,$00
       FCB    $FF,$00
       FCB    $F0,$00
       FCB    $FF,$F0

CF     FCB    3
       FCB    $FF,$F0
       FCB    $F0,$00
       FCB    $FF,$00
       FCB    $F0,$00
       FCB    $F0,$00

CG     FCB    3
       FCB    $FF,$F0
       FCB    $F0,$00
       FCB    $F0,$F0
       FCB    $F0,$F0
       FCB    $FF,$F0

CH     FCB    3
       FCB    $F0,$F0
       FCB    $F0,$F0
       FCB    $FF,$F0
       FCB    $F0,$F0
       FCB    $F0,$F0

CI     FCB    3
       FCB    $FF,$F0
       FCB    $0F,$00
       FCB    $0F,$00
       FCB    $0F,$00
       FCB    $FF,$F0

CJ     FCB    3
       FCB    $00,$F0
       FCB    $00,$F0
       FCB    $00,$F0
       FCB    $F0,$F0
       FCB    $FF,$F0

CK     FCB    3
       FCB    $F0,$F0
       FCB    $F0,$F0
       FCB    $FF,$00
       FCB    $F0,$F0
       FCB    $F0,$F0

CL     FCB    3
       FCB    $F0,$00
       FCB    $F0,$00
       FCB    $F0,$00
       FCB    $F0,$00
       FCB    $FF,$F0

CM     FCB    5
       FCB    $FF,$FF,$F0
       FCB    $F0,$F0,$F0
       FCB    $F0,$F0,$F0
       FCB    $F0,$00,$F0
       FCB    $F0,$00,$F0

CN     FCB    3
       FCB    $FF,$F0
       FCB    $F0,$F0
       FCB    $F0,$F0
       FCB    $F0,$F0
       FCB    $F0,$F0

CO     FCB    3
       FCB    $FF,$F0
       FCB    $F0,$F0
       FCB    $F0,$F0
       FCB    $F0,$F0
       FCB    $FF,$F0

CP     FCB    3
       FCB    $FF,$F0
       FCB    $F0,$F0
       FCB    $FF,$F0
       FCB    $F0,$00
       FCB    $F0,$00

CQ     FCB    3
       FCB    $FF,$F0
       FCB    $F0,$F0
       FCB    $FF,$F0
       FCB    $00,$F0
       FCB    $00,$F0

CR     FCB    3
       FCB    $FF,$F0
       FCB    $F0,$F0
       FCB    $FF,$00
       FCB    $F0,$F0
       FCB    $F0,$F0

CS     FCB    3
       FCB    $FF,$F0
       FCB    $F0,$00
       FCB    $FF,$F0
       FCB    $00,$F0
       FCB    $FF,$F0

CT     FCB    3
       FCB    $FF,$F0
       FCB    $0F,$00
       FCB    $0F,$00
       FCB    $0F,$00
       FCB    $0F,$00

CU     FCB    $3
       FCB    $F0,$F0
       FCB    $F0,$F0
       FCB    $F0,$F0
       FCB    $F0,$F0
       FCB    $FF,$F0

CV     FCB    3
       FCB    $F0,$F0
       FCB    $F0,$F0
       FCB    $F0,$F0
       FCB    $0F,$00
       FCB    $0F,$00

CW     FCB    5
       FCB    $F0,$00,$F0
       FCB    $F0,$00,$F0
       FCB    $F0,$F0,$F0
       FCB    $F0,$F0,$F0
       FCB    $FF,$FF,$F0

CX     FCB    3
       FCB    $F0,$F0
       FCB    $F0,$F0
       FCB    $0F,$00
       FCB    $F0,$F0
       FCB    $F0,$F0

CY     FCB    3
       FCB    $F0,$F0
       FCB    $F0,$F0
       FCB    $FF,$F0
       FCB    $0F,$00 
       FCB    $0F,$00

CZ     FCB    3
       FCB    $FF,$F0
       FCB    $00,$F0
       FCB    $0F,$00
       FCB    $F0,$00
       FCB    $FF,$F0

CLP    FCB    3
       FCB    $0F,$00
       FCB    $F0,$00
       FCB    $F0,$00
       FCB    $F0,$00
       FCB    $0F,$00

CRP    FCB    3
       FCB    $0F,$00
       FCB    $00,$F0
       FCB    $00,$F0
       FCB    $00,$F0
       FCB    $0F,$00

CFRB   FCB    7
       FCB    $00,$0F,$00,$00
       FCB    $00,$00,$F0,$00
       FCB    $FF,$00,$FF,$00
       FCB    $00,$00,$F0,$00
       FCB    $00,$0F,$00,$00

CSLSH  FCB    3
       FCB    $00,$F0
       FCB    $00,$F0
       FCB    $0F,$00
       FCB    $F0,$00
       FCB    $F0,$00

TAB57Z FDB    D0
       FDB    D1
       FDB    D2
       FDB    D3
       FDB    D4
       FDB    D5
       FDB    D6
       FDB    D7
       FDB    D8
       FDB    D9
       FDB    DSP
       FDB    DEX
       FDB    DCO
       FDB    DPE
       FDB    DAR
       FDB    DCL
       FDB    DHY
       FDB    DA
       FDB    DB
       FDB    DCC
       FDB    DD
       FDB    DE
       FDB    DF
       FDB    DG
       FDB    DH
       FDB    DI
       FDB    DJ
       FDB    DK
       FDB    DL
       FDB    DM
       FDB    DN
       FDB    DO
       FDB    DP
       FDB    DQ
       FDB    DR
       FDB    DS
       FDB    DT
       FDB    DU
       FDB    DV
       FDB    DW
       FDB    DX
       FDB    DY
       FDB    DZ
       FDB    DLP
       FDB    DRP
       FDB    DFRB
       FDB    DSLSH

D0     FCB    5
       FCB    $99,$99,$90
       FCB    $90,$00,$90
       FCB    $90,$00,$90
       FCB    $90,$00,$90
       FCB    $90,$00,$90
       FCB    $99,$99,$90

D1     FCB    5
       FCB    $00,$99,$00
       FCB    $00,$09,$00
       FCB    $00,$09,$00
       FCB    $00,$09,$00
       FCB    $00,$09,$00
       FCB    $09,$99,$90

D2     FCB    5
       FCB    $09,$99,$90
       FCB    $00,$00,$90
       FCB    $99,$99,$90
       FCB    $90,$00,$00
       FCB    $90,$00,$00
       FCB    $99,$99,$00

D3     FCB    5
       FCB    $09,$99,$90
       FCB    $00,$00,$90
       FCB    $09,$99,$90
       FCB    $00,$00,$90
       FCB    $00,$00,$90
       FCB    $99,$99,$90

D4     FCB    5
       FCB    $90,$00,$00
       FCB    $90,$09,$00
       FCB    $90,$09,$00
       FCB    $99,$99,$90
       FCB    $00,$09,$00
       FCB    $00,$09,$00

D5     FCB    5
       FCB    $99,$99,$00
       FCB    $90,$00,$00
       FCB    $99,$99,$90
       FCB    $00,$00,$90
       FCB    $00,$00,$90
       FCB    $99,$99,$90

D6     FCB    5
       FCB    $99,$99,$00
       FCB    $90,$00,$00
       FCB    $99,$99,$90
       FCB    $90,$00,$90
       FCB    $90,$00,$90
       FCB    $99,$99,$90

D7     FCB    5
       FCB    $99,$99,$90
       FCB    $00,$00,$90
       FCB    $00,$09,$00
       FCB    $00,$90,$00
       FCB    $09,$00,$00
       FCB    $90,$00,$00

D8     FCB    5
       FCB    $99,$99,$90
       FCB    $90,$00,$90
       FCB    $99,$99,$90
       FCB    $90,$00,$90
       FCB    $90,$00,$90
       FCB    $99,$99,$90

D9     FCB    5
       FCB    $99,$99,$90
       FCB    $90,$00,$90
       FCB    $99,$99,$90
       FCB    $00,$00,$90
       FCB    $00,$00,$90
       FCB    $09,$99,$90

DSP    FCB    3                   SPACE
       FCB    $00,$00
       FCB    $00,$00
       FCB    $00,$00
       FCB    $00,$00
       FCB    $00,$00
       FCB    $00,$00

DEX    FCB    2
       FCB    $FF,$00
       FCB    $FF,$00
       FCB    $FF,$00
       FCB    $FF,$00
       FCB    $00,$00
       FCB    $FF,$00
       FCB    $FF,$00

DCO    FCB    2                   COMMA
       FCB    $00,$00
       FCB    $00,$00
       FCB    $00,$00
       FCB    $FF,$00
       FCB    $FF,$00
       FCB    $F0,$00

DPE    FCB    2
       FCB    $00,$00
       FCB    $00,$00
       FCB    $00,$00
       FCB    $00,$00
       FCB    $FF,$00
       FCB    $FF,$00

DAR    FCB    9                       BLANKR  USED TO BLOCK ERASE
       FCB    $FF,$FF,$FF,$FF,$F0
       FCB    $FF,$FF,$FF,$FF,$F0
       FCB    $FF,$FF,$FF,$FF,$F0
       FCB    $FF,$FF,$FF,$FF,$F0
       FCB    $FF,$FF,$FF,$FF,$F0
       FCB    $FF,$FF,$FF,$FF,$F0

DCL    FCB    2                   COLON
       FCB    $00,$00
       FCB    $FF,$00
       FCB    $FF,$00
       FCB    $00,$00
       FCB    $FF,$00
       FCB    $FF,$00

DHY    FCB    3
       FCB    $00,$00
       FCB    $00,$00
       FCB    $00,$00
       FCB    $22,$20
       FCB    $00,$00
       FCB    $00,$00

DA     FCB    5
       FCB    $66,$66,$60
       FCB    $60,$06,$60
       FCB    $66,$66,$60
       FCB    $60,$06,$60
       FCB    $60,$06,$60
       FCB    $60,$06,$60

DB     FCB    5
       FCB    $66,$66,$00
       FCB    $60,$06,$00
       FCB    $66,$66,$60
       FCB    $66,$00,$60
       FCB    $66,$00,$60
       FCB    $66,$66,$60

DCC    FCB    5
       FCB    $66,$66,$60
       FCB    $66,$00,$00
       FCB    $66,$00,$00
       FCB    $66,$00,$00
       FCB    $66,$00,$00
       FCB    $66,$66,$60

DD     FCB    5
       FCB    $66,$66,$00
       FCB    $66,$00,$60
       FCB    $66,$00,$60
       FCB    $66,$00,$60
       FCB    $66,$00,$60
       FCB    $66,$66,$00

DE     FCB    5
       FCB    $66,$66,$60
       FCB    $60,$00,$00
       FCB    $66,$66,$60
       FCB    $66,$00,$00
       FCB    $66,$00,$00
       FCB    $66,$66,$60

DF     FCB    5
       FCB    $66,$66,$60
       FCB    $60,$00,$00
       FCB    $66,$66,$60
       FCB    $66,$00,$00
       FCB    $66,$00,$00
       FCB    $66,$00,$00

DG     FCB    5
       FCB    $66,$66,$60
       FCB    $66,$06,$60
       FCB    $66,$00,$00
       FCB    $66,$06,$60
       FCB    $66,$00,$60
       FCB    $66,$66,$60

DH     FCB    5
       FCB    $66,$00,$60
       FCB    $66,$00,$60
       FCB    $66,$66,$60
       FCB    $66,$00,$60
       FCB    $66,$00,$60
       FCB    $66,$00,$60

DI     FCB    4
       FCB    $06,$60,$00
       FCB    $06,$60,$00
       FCB    $06,$60,$00
       FCB    $06,$60,$00
       FCB    $06,$60,$00
       FCB    $06,$60,$00

DJ     FCB    5
       FCB    $00,$00,$60
       FCB    $00,$00,$60
       FCB    $00,$00,$60
       FCB    $00,$00,$60
       FCB    $66,$00,$60
       FCB    $66,$66,$60

DK     FCB    5
       FCB    $66,$00,$60
       FCB    $66,$06,$00
       FCB    $66,$60,$00
       FCB    $66,$60,$00
       FCB    $66,$06,$00
       FCB    $66,$00,$60

DL     FCB    5
       FCB    $60,$00,$00
       FCB    $60,$00,$00
       FCB    $60,$00,$00
       FCB    $60,$00,$00
       FCB    $66,$66,$60
       FCB    $66,$66,$60

DM     FCB    5
       FCB    $66,$66,$60
       FCB    $60,$60,$60
       FCB    $60,$60,$60
       FCB    $60,$00,$60
       FCB    $60,$00,$60
       FCB    $60,$00,$60

DN     FCB    5
       FCB    $66,$66,$60
       FCB    $60,$00,$60
       FCB    $66,$00,$60
       FCB    $66,$00,$60
       FCB    $66,$00,$60
       FCB    $66,$00,$60

DO     FCB    5
       FCB    $66,$66,$60
       FCB    $60,$06,$60
       FCB    $60,$00,$60
       FCB    $60,$00,$60
       FCB    $60,$00,$60
       FCB    $66,$66,$60

DP     FCB    5
       FCB    $66,$66,$60
       FCB    $60,$00,$60
       FCB    $66,$66,$60
       FCB    $66,$00,$00
       FCB    $66,$00,$00
       FCB    $66,$00,$00

DQ     FCB    5
       FCB    $66,$66,$60
       FCB    $66,$00,$60
       FCB    $66,$00,$60
       FCB    $66,$00,$60
       FCB    $66,$06,$00
       FCB    $66,$60,$60

DR     FCB    5
       FCB    $66,$66,$00
       FCB    $60,$06,$00
       FCB    $66,$66,$60
       FCB    $66,$00,$60
       FCB    $66,$00,$60
       FCB    $66,$00,$60

DS     FCB    5
       FCB    $66,$66,$60
       FCB    $60,$00,$00
       FCB    $66,$66,$60
       FCB    $00,$00,$60
       FCB    $66,$00,$60
       FCB    $66,$66,$60

DT     FCB    5
       FCB    $66,$66,$60
       FCB    $00,$60,$00
       FCB    $00,$66,$00
       FCB    $00,$66,$00
       FCB    $00,$66,$00
       FCB    $00,$66,$00

DU     FCB    5
       FCB    $60,$06,$60
       FCB    $60,$06,$60
       FCB    $60,$06,$60
       FCB    $60,$06,$60
       FCB    $60,$06,$60
       FCB    $66,$66,$60

DV     FCB    5
       FCB    $66,$00,$60
       FCB    $66,$00,$60
       FCB    $66,$00,$60
       FCB    $66,$00,$60
       FCB    $06,$06,$00
       FCB    $00,$60,$00

DW     FCB    5
       FCB    $60,$00,$60
       FCB    $60,$00,$60
       FCB    $60,$00,$60
       FCB    $60,$60,$60
       FCB    $60,$60,$60
       FCB    $66,$66,$60

DX     FCB    5
       FCB    $60,$00,$60
       FCB    $06,$06,$00
       FCB    $00,$60,$00
       FCB    $00,$60,$00
       FCB    $06,$06,$00
       FCB    $60,$00,$60

DY     FCB    5
       FCB    $66,$00,$60
       FCB    $66,$00,$60
       FCB    $66,$66,$60
       FCB    $00,$60,$00
       FCB    $00,$60,$00
       FCB    $00,$60,$00

DZ     FCB    5
       FCB    $66,$66,$60
       FCB    $00,$06,$00
       FCB    $00,$60,$00
       FCB    $06,$00,$00
       FCB    $66,$66,$60
       FCB    $66,$66,$60

DLP    FCB    3
       FCB    $00,$F0
       FCB    $0F,$00
       FCB    $F0,$00
       FCB    $F0,$00
       FCB    $0F,$00
       FCB    $00,$F0

DRP    FCB    3
       FCB    $F0,$00
       FCB    $0F,$00
       FCB    $00,$F0
       FCB    $00,$F0
       FCB    $0F,$00
       FCB    $F0,$00

DFRB   FCB    4                   SLASH
       FCB    $00,$0F,$00
       FCB    $00,$F0,$00
       FCB    $00,$F0,$00
       FCB    $0F,$00,$00
       FCB    $0F,$00,$00
       FCB    $F0,$00,$00

DSLSH  FCB    5
       FCB    $00,$F0,$00
       FCB    $0F,$F0,$00
       FCB    $FF,$FF,$F0
       FCB    $0F,$F0,$00
       FCB    $00,$F0,$00
       FCB    $00,$00,$00

       END
