
**************************************************************************
*                                                                        *
*          This is the patch that was written by Christian Gingras       *
*          to fix the famous enforcer diagonal explosion bug.            *
*          It is used to patch the $4000 rom in the final robotron       *
*          version (called version 6 in software.....R5 is what is       *
*          on the listing).                                              *
*                                                                        *
**************************************************************************

 OPT NOL
 LIB RRF
 OPT LIS

 STTL DIAGONAL EXPLOSIONS

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

KLJMP    EQU     $4A59

         ORG     $4990
         JMP     UN

         ORG     $49B7
         JMP     DEUX

         ORG     $49E2
         JMP     TROIS

         ORG     $4A97

UN       DEC     HITE
         BEQ     KLJMP
         DEC     YOFF
         JMP     $4994

DEUX     DEC     HITE
         BEQ     KLJMP
         INC     YYCNT
         JMP     $49BB

TROIS    DEC     HITE
         BEQ     KLJMP
         INC     YYCNT
         JMP     $49E6

         FCC     '05/1987 CHR.G.'

         ORG     $4AD7
*         FCB     $FD             THIS SETS JAP ZAPPER CORRECT
         FCB     $4C             THIS SETS CHECKSUM FOR THE PART CORRECT.

 END

