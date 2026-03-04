*
*      Additional patch bytes for version 6.   It is assumed that
*      the binary of all of these sources, Plus these patches,
*      plus setting the checksum block around ffb0 comprises
*      the correct binary image.
*

       ORG    $26B1
       FCB    $55,$55,$55,$55,$55,$55,$55,$55
       ORG    $DF34
       FCB    $A,$A,$A,$A,$A,$A,$A       
       ORG    $E3B0
       FCB    $26,$26,$26,$26,$26,$26,$26,$26,$26
       ORG    $6D2E
       FCB    $21
       ORG    $E398
       FCB    $A6
       END

