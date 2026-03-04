*
*      ROBOTRON 3-10-82 RELEASE 1
*

       ORG    $26B7
       FCB    $65
       ORG    $5E77
       FCB    $68
       ORG    $DE54
       FCB    $1B,$1B,$1B,$1B,$1B,$1B,$1B,$1B
*
*      CHECKSUMS
*
*
*      0-$4D
*      1-$19
*      2-$D9
*      3-$1B
*      4-$7E
*      5-$DD
*      6-$F1
*      7-$3E
*      8-$6D
*
*      D-$4D
*      E-$48

*      SUPER CHECKERS - 98F3 = $37
*                       BE72 = $4A


**************************************************************************
*                                                                        *
*          Here is a detailed explaination of the Robotron 2084          *
*          "JAP ZAPPERS".  This was a primitive system setup to          *
*          try and make it more difficult for game copiers to            *
*          Modify the game.  There were scrambled sections of the        *
*          code that would try and monitor "random" checksums            *
*          of regions of the code.  If they detected that the code       *
*          had been modified, they would cause random program            *
*          events to occur at a low frequency (the random and            *
*          low frequency combination would make it harder to track       *
*          down the culprits).  We always assumed that developers        *
*          with good tools would find these quickly, but Christian       *
*          Gingras, removed this small nuisance while trying to          *
*          fix the other bugs he had found in the program (with          *
*          the aid of his Radio Shack Color Computer).                   *
*                                                                        *
*          There are 2 regions for which checksums are kept.  The        *
*          checksums are not the sum of all bytes, but the sum           *
*          of every 8 and 16 bytes respectively.  The first checksum     *
*          is the sum of every 16th byte from $0007 to $8927 inclusive   *
*          and is stored at $98F3 (HCK1).  This sum should come out to   *
*          $37.  The second checksum is the sum of every 8th byte        *
*          from $D015 to $EAAD and is stored at $BE72 (HCK2).  This sum  *
*          should come out to $4A.                                       *
*                                                                        *
*          There are 6 places in the software that are concerned         *
*          with the checksums.                                           *
*                                                                        *
*          In RRTEXT, at label WORDZ (the call for message handling)     *
*          there is a check of HCK1 for its $37 value, and handling      *
*          to zap if incorrect.                                          *
*                                                                        *
*          In RRS22, the code to set the HCK1 value (by performing the   *
*          required addition) is found within the PINIT routine.         *
*          PINIT is used to initialize the process structure at the      *
*          start of each turn.                                           *
*                                                                        *
*          In RRS22 there is a check of HCK1 for its $37 value in the    *
*          ASCAN routine.  ASCAN is called at the start of each wave     *
*          by each object to make sure the space it wants is un-occupied *
*                                                                        *
*          In RRG23 the code to setup the HCK2 value (by performing      *
*          the required addition) is found within the game start         *
*          routine (START1 and START2).                                  *
*                                                                        *
*          In RRX7 there is a check of HCK2 for its $4A value in         *
*          the explosion starter (EXSTV).                                *
*                                                                        *
*          In RRB10 there is a routine that computes the HCK2 value      *
*          (on the fly in case the copiers goose the values into         *
*          ram) and zaps if it does not come up with the $4A value.      *
*          This is found in the brain starter (BRNSTV) and is run        *
*          in any waves that contain BRAIN robots.                       *
*                                                                        *
*          The checksums are computed with the equivilents of the        *
*          following code:                                               *
*                                                                        *
*                            HCK1 ($98F3) = $37                          *
*                                                                        *
*                                  clra                                  *
*                                  ldx     #$0007                        *
*                          loop:   adda    0,x                           *
*                                  leax    $10,x                         *
*                                  cmpx    #$8935                        *
*                                  blo     loop                          *
*                                  sta     hck1                          *
*                                                                        *
*                                                                        *
*                            HCK2 ($BE72) = $4A                          *
*                                                                        *
*                                  clra                                  *
*                                  ldx     #$d015                        *
*                          loop:   adda    0,x                           *
*                                  leax    $8,x                          *
*                                  cmpx    #$EAB1                        *
*                                  blo     loop                          *
*                                  sta     hck2                          *
*                                                                        *
*                                                                        *
**************************************************************************

       END

