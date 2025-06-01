This is just a dump of my source code folder for the AI, not (yet) by any stretch carefully organized!

The code is all in the Scripts folder.  The LUA files have .lua extensions and the Python files have .py extensions.
Everything should work in theory on Mac, Windows, or Linux.
Code supports CPU, MacGPU, and CUDA

First, make sure you have a working copy of current MAME installed.  You'll need the TEMPEST1 roms in your TEMPEST1 folder under your ROMS folder.  Make sure you can manually start mame and run Tempest before proceeding.

In the main.lua file you will need to update the name of the server to be whatever machine you plan to run the python server on.  localhost will work if the clients and server are the same machine.

local SOCKET_ADDRESS          = "socket.localhost:9999"

To run the server, simply run:

python Scripts/main.py

Then run a MAME client:

On Mac/Linux, this is likely as follows, but update the full path to your main.lua file

mame tempest1 -skip_gameinfo -nothrottle -sound none -autoboot_script ~/source/repos/tempest/Scripts/main.lua

On Windows

start /b mame tempest1 -skip_gameinfo -autoboot_script c:\users\dave\source\repos\tempest_ai\Scripts\main.lua -nothrottle -sound none -frameskip 9-window >nul


