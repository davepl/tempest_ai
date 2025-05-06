#!/bin/bash

# 1. Terminate any running mame processes
killall -9 mame

# 2. Update tempest_ai repo
cd ~/source/repos/tempest_ai || exit 1
git pull

# 3. Run N copies of mame in background (default 1)
N=${1:-1}
cd ~/mame || exit 1

for ((i=1; i<=N; i++)); do
  mame tempest1 -sound none -video none -skip_gameinfo -autoboot_script ~/source/repos/tempest_ai/Scripts/main.lua -nothrottle > /dev/null 2>&1 &
done