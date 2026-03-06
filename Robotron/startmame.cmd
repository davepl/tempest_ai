@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "LUA_SCRIPT=%SCRIPT_DIR%Scripts\main.lua"
set "ROM_DIR=%SCRIPT_DIR%roms"
set "COUNT=%~1"
if "%COUNT%"=="" set "COUNT=1"

echo Launching %COUNT% Robotron MAME instance(s)...
for /l %%x in (1,1,%COUNT%) do (
    start /b mame robotron -rompath "%ROM_DIR%" -skip_gameinfo -autoboot_script "%LUA_SCRIPT%" -nothrottle -sound none -window >nul
)
