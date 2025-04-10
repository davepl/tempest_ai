
To execute MAME, which will run LUA, which will spin up the Python:

1) Make sure you have ~/mame/roms/tempest1/*.* in place from the roms folder.

2) mame tempest1 -autoboot_script ~/source/repos/tempest/Scripts/main.lua -skip_gameinfo

Adjust the path on this command line, and in the main.lua code that launches python, to reflect your actual paths.

Example command line to run mame.  Assumes you're in the mame folder and that roms is a subdir of current:

mame tempest1 -skip_gameinfo -nothrottle -sound none -autoboot_script ~/source/repos/tempest/Scripts/main.lua

start /b mame tempest1 -skip_gameinfo -autoboot_script c:\users\dave\source\repos\tempest_ai\Scripts\main.lua -nothrottle -sound none -frameskip 9-window >nul

Headless on MacOS:
SDL_VIDEODRIVER=dummy mame tempest1 -video none -seconds_to_run 1000000000 -sound none &


Training Notes
--------------

Apr-09-25: 
Removed movement award, letting proximity do the work.
At 5M frames, spinner alternates between inaction and basic aiming on level 1 and tracking on level 2
Added engineered params for distance to enemy, etc, and it learned to move around (but not always shoot) in 1M frames
Started grabbing inferred values, thinking it more accurately captures the current frame's action

Now at 5M frames it was hunting effectively

ToDo:

- Consider adding fractional lsb info to current positions/depths
- 