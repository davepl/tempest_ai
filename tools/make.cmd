@echo off
goto normal

.\HXA65W.EXE original.asm 
echo Checking original ROM for match...
fc /b TEMPEST.OBJ ..\..\roms\0x9000.rom >NUL
if ERRORLEVEL 1 echo "ERROR: File does not match original ROM" & goto :EOF
echo GOOD: Original file MATCH.

:normal

.\HXA65W.EXE -m Tempest.asm 
fc /b TEMPEST.OBJ ..\..\roms\0x9000.rom >NUL
if ERRORLEVEL 1 echo Testing: File does NOT match original ROM
if NOT ERRORLEVEL 1 echo Testing: File DOES match original ROM

; del /q "\\daveplhome\d\mamesrc\roms\tempest\*.*" 2>NUL
.\splitrom.exe tempest.obj
xcopy  /y 136* z:\OneDrive\Tempest\roms\NewFormat\Tempest


