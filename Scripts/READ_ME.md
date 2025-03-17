
To execute MAME, which will run LUA, which will spin up the Python:

1) Make sure you have ~/mame/roms/tempest1/*.* in place from the roms folder.

2) mame tempest1 -autoboot_script ~/source/repos/tempest/Scripts/main.lua -skip_gameinfo

Adjust the path on this command line, and in the main.lua code that launches python, to reflect your actual paths.

Example command line to run mame.  Assumes you're in the mame folder and that roms is a subdir of current:

mame tempest1 -skip_gameinfo -nothrottle -sound none -autoboot_script ~/source/repos/tempest/Scripts/main.lua
