This is a system that uses LUA script within Mame and a Python script to operate as a Reinforcement Learning system for Tempest.

The system is comprised of N clients of MAME, each running a custom startup script called 'main.lua' which registers a frame callback.  Each time a frame is generated in the Tempest game, the frame callback inspects the game memory to extract on the order of 175 parameters about the game state.  Only things visible to a human player are considered.  Those parameters are flattened to a floating point array and passed over a standard socket to the python application.  The python application returns the action the player should take in this instance.  Those recommendations are applied to the game's control input, and operation proceeds to the next frame.

On the python side, a socket server receives requests from the LUA clients across the socket interface each time a packet of state+reward+etc data is received.  That data is placed in a training queue and added to a replay buffer of on the order of 2M frames.  

A background thread operates in the Python app which pulls training requests from a queue and submits them to the GPU for processing.  It randomly samples its batch from the 2M frames in the replay buffer.  The replay buffer is constantly refreshed at one end by new incoming frames as old ones are removed to make room.

Each time a frame request comes in that needs a control input decision, the RL system decides whether that should be done by the expert or the DQN AI.  A random number is chosen and if it is below the current expert ratio, the expert is used.  It uses information passed in by the frame data to select a target and generate fire/zap/spinner controls.  If the DQN is selected, the epsilon rate is checked against a new random number.  If it is under the epsilon rate, a random control input for the discrete states will be used, and noise will be applied to whatever the model returns.  Otherwise, the model is asked to make the control decision.

Frames are placed into the replay buffer as they are received from the N clients. Episodes are tracked per client and end when the player dies.  

In the LUA code, there are two reward calculations made. One is subjective and based on things like moving to a good target lane.  The other is objective and is based on points only, with large level rewards filtered out.  The rewards are multiplied by a scaling factor, with subjective being kept lower than objective.  This allows the objective points to dominate the decisions while being "nudged" towards good play with the smaller subjective bonus.

Because some ratio of the frames are randomly influenced, the AI will be able to "stumble" into new situations that work out well wrt reward and survival.  This allows the student AI to exceed the teacher of the expert system by training on frames that randomly led to better outcomes, backpropagating the episode into the model weights, and thereby becoming "smarter" than even the expert over time.

The system uses a hybrid action space with discrete fire/zap actions (4 combinations) and continuous spinner control. Training uses n-step returns when enabled, and includes target network updates for stability.


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

May 18-25:
With smaller model and enemy avoidance rewards, makes it to yellow on a couple of players by 25M frames (Start 15)

