# Tempest AI — Teaching a Neural Network to Play a 1981 Arcade Classic

Tempest AI is a reinforcement learning system that learns to play Atari's **Tempest** (1981) by watching the game run inside the MAME arcade emulator. A Lua script reads the game's memory every frame, a Python application trains a neural network on the GPU, and the network's decisions are fed back to the game controls — all in real time, at thousands of frames per second.

This README explains the architecture for programmers who may not be deep-learning specialists. No prior RL knowledge is assumed.

---

## Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [High-Level Architecture](#high-level-architecture)
3. [The MAME Side (Lua)](#the-mame-side-lua)
4. [The Python Side](#the-python-side)
5. [The Socket Bridge](#the-socket-bridge)
6. [The Neural Network](#the-neural-network)
7. [Training Loop](#training-loop)
8. [The Expert System](#the-expert-system)
9. [Exploration and Learning](#exploration-and-learning)
10. [Reward Design](#reward-design)
11. [Live Dashboard](#live-dashboard)
12. [Running the System](#running-the-system)
13. [Project Structure](#project-structure)
14. [Key Configuration](#key-configuration)
15. [Keyboard Controls](#keyboard-controls)

---

## What This Project Does

The goal is to train an AI agent that plays Tempest **better than its own teacher**. It does this through a combination of:

- **An expert system** — a hand-coded rule engine that knows basic Tempest strategy (aim at the nearest enemy, avoid pulsars, dodge fuseballs).
- **A deep neural network** — a Rainbow DQN variant that starts by imitating the expert, then gradually takes over and discovers strategies the expert never knew.
- **Reinforcement learning** — the network learns from its own experience: what actions led to high scores and long survival, and what led to death.

---

## High-Level Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│  MAME Emulator (one or more instances)                                 │
│                                                                        │
│  ┌──────────┐    reads    ┌──────────┐   serializes   ┌────────────┐   │
│  │ Tempest  │───memory───▶│  Lua     │───195 floats──▶│  TCP       │   │
│  │ ROM      │             │  Scripts │   + rewards    │  Socket    │   │
│  │          │◀──controls──│          │◀──3 bytes──────│            │   │
│  └──────────┘   (fire,    └──────────┘   (fire,zap,   └─────┬──────┘   │
│                  zap,                     spinner)           │         │
│                  spinner)                                    │         │
└──────────────────────────────────────────────────────────────┼─────────┘
                                                               │ TCP
┌──────────────────────────────────────────────────────────────┼────────┐
│  Python Application                                          │        │
│                                                              ▼        │
│  ┌────────────┐  frames  ┌──────────────┐  batches  ┌──────────────┐  │
│  │  Socket    │─────────▶│   Replay     │──────────▶│  Training    │  │
│  │  Server    │          │   Buffer     │           │  Thread      │  │
│  │            │◀─action──│   (15M)      │           │  (GPU)       │  │
│  └────────────┘          └──────────────┘           └──────┬───────┘  │
│        │                                                   │          │
│        │ inference    ┌──────────────┐    weight sync      │          │
│        └─────────────▶│  Rainbow     │◀────────────────────┘          │
│                       │  DQN Model   │                                │
│                       └──────────────┘                                │
│                                                                       │
│  ┌────────────────┐                                                   │
│  │  Web Dashboard │  http://localhost:8765                            │
│  └────────────────┘                                                   │
└───────────────────────────────────────────────────────────────────────┘
```

Multiple MAME instances can connect simultaneously. Each one is a separate "client" that plays its own game, generating training data in parallel.  The can be on multiple different client machines if desired.  

---

## The MAME Side (Lua)

MAME has a built-in Lua scripting engine. When you launch MAME with `-autoboot_script`, it runs a Lua script that can:

- **Read and write the game's memory** (the 6502 CPU's address space)
- **Override input controls** (fire button, zap button, spinner dial)
- **Register a callback** that runs every frame

### What Happens Each Frame

The Lua code (`Scripts/main.lua` + modules) performs these steps every game frame:

1. **Read game state from memory** — The script reads ~80 memory addresses to extract
   everything a human player could see: player position, enemy positions and types,
   enemy depths, shot positions, spike heights, level geometry, score, lives, and more.

2. **Compute the expert action** — A rule-based expert system (`Scripts/logic.lua`)
   analyzes the game state and decides what it *would* do: which segment to aim for,
   whether to fire or use the superzapper, which direction to spin.

3. **Calculate rewards** — Two reward signals are computed:
   - **Objective reward**: based on score changes (points from killing enemies)
   - **Subjective reward**: based on positioning quality (are you aimed at a threat?
     are you avoiding danger?)

4. **Serialize everything into a binary packet** — The 195 game-state values are
   normalized to `[-1, +1]` floats. Along with rewards, expert recommendations,
   and metadata, they're packed into a ~780-byte binary message.

5. **Send the packet over a TCP socket** — The Lua script connects to the Python
   server at startup and streams frames continuously.

6. **Receive a 3-byte action reply** — The Python side responds with:
   - `fire` (0 or 1)
   - `zap` (0 or 1)
   - `spinner` (signed byte: -32 to +31, controlling rotation speed and direction)

7. **Apply the action to the game controls** — The fire and zap buttons are set
   via MAME's I/O port API, and the spinner value is written directly to the
   memory address that the game reads for dial input.

### Lua Module Breakdown

| File           | Purpose                                                                          |
|----------------|----------------------------------------------------------------------------------|
| `main.lua`     | Entry point: frame callback, socket I/O, binary serialization                    |
| `state.lua`    | Game state classes: reads ~80 memory addresses into structured objects           |
| `logic.lua`    | Expert system: rule-based target selection, threat avoidance, reward calculation |
| `display.lua`  | Optional on-screen debug overlay (game state, enemy tables)                      |

### State Vector (195 Features)

The 195 normalized float values sent each frame include:

| Category             | Count | Examples                                                                     |
|----------------------|-------|------------------------------------------------------------------------------|
| Game state           |     5 | gamestate, game mode, countdown, lives, level                                |
| Player               |    23 | position, alive, depth, zap uses, 8 shot positions, 8 shot segments          |
| Level geometry       |    35 | level number, open/closed, shape, 16 spike heights, 16 tube angles           |
| Enemy global         |    23 | counts by type, spawn slots, speeds, pulsar state                            |
| Enemy per-slot (×7)  |    42 | type, direction, between-segments, moving away, can shoot, split behavior    |
| Enemy spatial (×7)   |    49 | segments, depths, top-of-tube flags, shot positions, pulsar lanes, top-rail  |
| Danger proximity     |     3 | nearest threat depth in player's lane, left, and right                       |
| Enemy velocity (×7)  |    14 | per-slot segment delta and depth delta from previous frame                   |

---

## The Python Side

The Python application (`Scripts/main.py`) is the brain of the system. It:

1. **Creates the neural network** and loads any previously saved weights
2. **Starts a TCP socket server** to accept connections from MAME instances
3. **Runs a background training thread** that continuously improves the network
4. **Serves a live web dashboard** for monitoring training progress
5. **Handles keyboard commands** for real-time tuning

### Threading Model

| Thread               | Role                                                                        |
|----------------------|-----------------------------------------------------------------------------|
| Main thread          | Startup, periodic autosave, shutdown coordination                           |
| Socket server        | Accepts MAME connections, spawns per-client handler threads                 |
| Per-client threads   | Receive frames, request inference, send actions, store transitions          |
| Training thread      | Samples from replay buffer, runs gradient updates on GPU                    |
| Inference batcher    | Collects inference requests across clients, runs batched GPU forward passes |
| Async replay buffer  | Queues `step()` calls so client threads don't block on buffer writes        |
| Stats reporter       | Prints formatted metrics to the terminal every 30 seconds                   |
| Dashboard server     | HTTP server for the live web UI                                             |

---

## The Socket Bridge

The communication between Lua and Python uses a simple custom binary protocol over TCP:

### Lua → Python (per frame)
```
[2 bytes: payload length (big-endian uint16)]
[payload]:
    [2 bytes: num_values (uint16)]
    [8 bytes: subjective reward (float64)]
    [8 bytes: objective reward (float64)]
    [1 byte: gamestate]
    [1 byte: game_mode]
    [1 byte: done flag]
    [2 bytes: frame counter (uint16)]
    [4 bytes: score (uint32)]
    [1 byte: save signal]
    [1 byte: commanded fire]
    [1 byte: commanded zap]
    [2 bytes: commanded spinner (int16)]
    [2 bytes: expert target segment (int16)]
    [1 byte: player segment]
    [1 byte: is_open_level]
    [1 byte: expert fire]
    [1 byte: expert zap]
    [1 byte: level number]
    [num_values × 4 bytes: state vector (float32 each)]
```

### Python → Lua (per frame)
```
[1 byte: fire (int8)]
[1 byte: zap (int8)]
[1 byte: spinner (int8, range -32..+31)]
```

The protocol is designed for minimal latency. At full speed with throttling disabled, a single MAME instance can push 2,000–2,800 frames per second.

---

## The Neural Network

The model is a **Rainbow DQN** variant — a combination of several improvements to the original Deep Q-Network algorithm. Here's what each component does in plain terms:

### Architecture Summary

```
Input (195 floats)
    │
    ├──▶ Lane-Cross-Attention Encoder
    │      16 lane tokens (spike, angle, player_here, sin/cos position)
    │      × 7 enemy tokens (type, direction, seg, depth, velocity, ...)
    │      → Cross-attention: each lane "looks at" nearby enemies
    │      → 128-dim summary vector
    │
    └──▶ Full state vector (195 floats)
            │
            ├─ concatenate with attention summary ──▶ [323 floats]
            │
            ▼
    Trunk: 2 × (Linear 323→384 + LayerNorm + ReLU)
            │
            ├──▶ Value stream:  Linear 384→192→51   (how good is this state?)
            └──▶ Advantage stream: Linear 384→192→44×51 (how much better is each action?)
                    │
                    ▼
            Combine via dueling formula → Q-value distribution per action
            44 actions × 51 atoms each
```

### Key Techniques

| Technique                    | What it does                                                                                       | Why it helps                                                                                        |
|------------------------------|----------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| **C51 (Categorical DQN)**    | Predicts a *probability distribution* over 51 values (-100 to +100) instead of a single Q-value    | More stable learning; the network can express uncertainty about outcomes                            |
| **Dueling Architecture**     | Splits into "how good is this state?" and "how much better is this action vs. average?"            | Faster learning — can learn state value without evaluating every action                             |
| **Factored Actions**         | 44 joint actions = 4 fire/zap combos × 11 spinner speeds                                           | More efficient than a flat 44-action head; fire/zap and spinner have separate jointly-trained heads |
| **Enemy Attention**          | Multi-head self-attention over 7 enemy slots, cross-attention to 16 lane tokens                    | Lets the network focus on the most relevant enemies and understand spatial relationships            |
| **Prioritized Replay (PER)** | Samples training examples proportional to how "surprising" they were (high TD error)               | Focuses learning on the hardest, most informative experiences                                       |
| **N-step Returns**           | Uses 12-step lookahead for computing target values instead of just 1 step                          | Better credit assignment — connects actions to rewards 12 frames later                              |

### Model Size

- ~1.2M trainable parameters
- Small enough to run inference in ~5ms on a modern GPU
- Can handle batched inference across 20+ simultaneous MAME clients

---

## Training Loop

Training happens continuously in a background thread while games are being played.

### One Training Step

1. **Sample a batch** of 256 transitions from the replay buffer (weighted by priority)
2. **Compute target distributions** using the C51 distributional Bellman equation:
   - Use the *online* network to pick the best next action (Double DQN)
   - Use the *target* network to evaluate that action's value distribution
   - Project the Bellman-shifted distribution onto the fixed atom support
3. **Compute the loss** — cross-entropy between predicted and target distributions
4. **Add behavioral cloning loss** — for transitions that came from the expert,
   add a small loss that encourages the network to match the expert's action
5. **Backpropagate** with gradient clipping (max norm 5.0)
6. **Update priorities** in the replay buffer based on TD error
7. **Periodically hard-sync** the target network (every 2,500 steps)

### Replay Buffer

The replay buffer holds up to **15 million** transitions. Each transition stores:

- State (195 floats)
- Action taken (joint index 0–43)
- Reward received
- Next state
- Whether the episode ended
- N-step horizon length
- Whether this was an expert or DQN action
- Priority (for PER sampling)

At ~2,000 FPS, the buffer fills in about 2 hours.  Old transitions are overwritten as new ones arrive, keeping the data fresh.

---

## The Expert System

The expert system (`Scripts/logic.lua`) is a hand-coded Tempest player. It's not great — maybe a 6/10 player — but it provides a crucial learning scaffold:

### Expert Strategy

1. **Target selection** — Hunt enemies in priority order: fuseballs (most dangerous) > flippers > tankers > spikers
2. **Pulsar avoidance** — When pulsars are active (pulsing), move away from their lanes
3. **Fuseball evasion** — If a fuseball is within 2 segments and shallow, flee 3 segments away
4. **Top-rail flipper handling** — Special logic for flippers that have reached the tube rim
5. **Tube zoom navigation** — During the between-level zoom, steer to lanes with the shortest spikes
6. **Fire control** — Fire with 95% probability each frame (conserve occasionally)
7. **Superzapper** — Use the zapper based on Lua-computed signals

### How the Expert Bootstraps Learning

Early in training (~50% expert ratio), half of all frames use the expert's action. This teaches the network basic competence quickly. The expert ratio decays to 2% over 5 million frames. By then, the DQN has surpassed the expert and mostly drives on its own.

A **behavioral cloning (BC) loss** provides additional guidance: for expert-generated transitions, the network receives a small extra loss encouraging it to match the expert's chosen action. This weight also decays over time.

---

## Exploration and Learning

How does the AI get *better* than the expert? Through structured exploration:

### Epsilon-Greedy Exploration

With probability ε (epsilon), the agent takes a **random action** instead of its best guess.
- ε starts at 100% (fully random) and decays to 1% over 500,000 frames
- Random fire/zap exploration   uses a reduced superzapper probability (5%) to avoid wasting the limited superzapper on random frames
- Optional **epsilon pulses** periodically boost ε back up to 25% late in training to escape local optima

### The Key Insight

When a random or semi-random action happens to produce a great outcome (enemy killed, level cleared, longer survival), that entire sequence gets stored in the replay buffer with a *high reward*. The network trains on it, learns the pattern, and starts choosing that action *intentionally* in similar situations. This is how the student surpasses the teacher — it stumbles into strategies the expert never considered, and reinforces the ones that work.

---

## Reward Design

The reward signal has two components, combined additively:

### Objective Reward (score-based)
- Derived from game score changes
- Scaled by `obj_reward_scale` (0.01)
- Large level-completion bonuses are filtered out to prevent reward spikes
- Death incurs a penalty

### Subjective Reward (positioning-based)
- Rewards for being in a lane free of nearby threats
- Penalties for being in a lane with shallow enemies
- Scaled by `subj_reward_scale` (0.01) — kept lower than objective so points dominate

### Clipping
All rewards are clipped to [-10, +10] to prevent extreme values from destabilizing training.

---

## Live Dashboard

The Python app serves a real-time web dashboard at `http://localhost:8765` with:

- **Gauges** — Current loss, reward, epsilon, expert ratio
- **Mini charts** — Rolling history with log-compressed time scale for: DQN reward (100K/1M/5M windows), loss, gradient norm, agreement, episode length, and more
- **Metric cards** — FPS, client/web connections, inference time, replay buffer size, Q-range, learning rate
- **Big zoomable charts** — Click any mini chart to expand it

The dashboard uses a single self-contained HTML page served from `Scripts/metrics_dashboard.py` with no external dependencies — all CSS, JavaScript, and chart rendering is embedded inline.

---

## Getting Started (From Scratch)

### 1. Clone the Repository

```bash
git clone https://github.com/davepl/tempest_ai.git
cd tempest_ai
```

### 2. Install Python Dependencies

Requires **Python 3.10+**. A virtual environment is recommended:

```bash
python3 -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

This installs PyTorch, NumPy, and supporting packages. For GPU-accelerated training (strongly recommended), install the CUDA version of PyTorch — see [pytorch.org](https://pytorch.org/get-started/locally/) for the correct command for your system.

### 3. Install MAME

MAME must be installed and accessible from your terminal.

- **Linux:** `sudo apt install mame` or build from [mamedev.org](https://www.mamedev.org/)
- **macOS:** `brew install mame`
- **Windows:** Download from [mamedev.org](https://www.mamedev.org/release.html) and add to your PATH

Verify it works:

```bash
mame -help
```

### 4. Place the Tempest ROMs

You need the **tempest1** ROM set. Place the ROM files in MAME's ROM directory:

```bash
# Find MAME's ROM path (usually ~/mame/roms or similar)
mame -listxml | head -5     # check MAME is working

# Copy your ROM files into place
cp -r roms/tempest1 ~/.mame/roms/    # Linux (path may vary)
```

On Windows, the default is typically `C:\mame\roms\tempest1\`. On macOS with Homebrew, check `$(brew --prefix)/Cellar/mame/*/share/mame/roms/`.

Verify MAME can find the ROMs:

```bash
mame tempest1 -verifyroms
```

### 5. Start the Python Server

```bash
python3 Scripts/main.py
```

You should see output confirming:
- Socket server listening on port **9999**
- Dashboard available at **http://localhost:8765**
- Model loaded (or initialized fresh on first run)

### 6. Launch One or More MAME Clients

In a separate terminal:

```bash
mame tempest1 -skip_gameinfo -nothrottle -sound none \
    -autoboot_script /path/to/tempest_ai/Scripts/main.lua
```

Replace `/path/to/tempest_ai` with the actual path to your clone. MAME will connect to the Python server and begin playing automatically.

To collect data faster, launch **multiple instances** in parallel:

```bash
# Linux — launch 4 headless clients
for i in $(seq 1 4); do
  SDL_VIDEODRIVER=dummy mame tempest1 -video none -sound none -nothrottle \
      -autoboot_script /path/to/tempest_ai/Scripts/main.lua &
done
```

### 7. Monitor Training

Open the live dashboard in your browser:

```
http://localhost:8765
```

The dashboard shows real-time gauges, charts, and metrics. Training begins automatically once enough frames have been collected in the replay buffer.

---

## Running the System

### Prerequisites

- **MAME** — installed and in your PATH (or specify full path)
- **Tempest ROMs** — place the `tempest1` ROM set in MAME's ROM directory
- **Python 3.10+** with PyTorch and NumPy
- **NVIDIA GPU** recommended (CUDA) for training speed; CPU works but is much slower

### Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 1: Start the Python Application

```bash
cd /path/to/tempest_ai
python3 Scripts/main.py
```

This starts the socket server (port 9999), the web dashboard (port 8765), and waits for MAME connections.

### Step 2: Launch MAME in a separate console/window

```bash
mame tempest1 -skip_gameinfo -nothrottle -sound none \
    -autoboot_script /path/to/tempest_ai/Scripts/main.lua
```

Key MAME flags:
| Flag               | Purpose                                |
|--------------------|----------------------------------------|
| `-nothrottle`      | Run as fast as possible (no 60fps cap) |
| `-sound none`      | Disable audio for speed                |
| `-skip_gameinfo`   | Skip the ROM info screen               |
| `-autoboot_script` | Run the Lua script at startup          |

You can launch multiple MAME instances pointing to the same Python server for parallel data collection.  They can be on different machines as long as they target the same server.

### Headless Operation (Linux)

```bash
SDL_VIDEODRIVER=dummy mame tempest1 -video none -sound none -nothrottle -autoboot_script /path/to/tempest_ai/Scripts/main.lua &
```

---

## Project Structure

```
tempest_ai/
├── Scripts/
│   ├── main.lua              # MAME entry point: frame callback, socket I/O, serialization
│   ├── state.lua             # Game state extraction from 6502 memory
│   ├── logic.lua             # Expert system + reward calculation
│   ├── display.lua           # Optional on-screen debug overlay
│   │
│   ├── main.py               # Python entry point: startup, threads, shutdown
│   ├── config.py             # All configuration: hyperparameters, server, metrics
│   ├── aimodel.py            # Neural network architecture + Rainbow agent
│   ├── training.py           # C51 training step (GPU)
│   ├── socket_server.py      # TCP server bridging Lua ↔ Python
│   ├── replay_buffer.py      # Prioritized experience replay buffer
│   ├── nstep_buffer.py       # N-step return accumulator
│   ├── metrics_dashboard.py  # Live web dashboard (self-contained HTML/CSS/JS)
│   └── metrics_display.py    # Terminal metrics formatting
│
├── models/
│   └── tempest_model_latest.pt   # Saved model weights + optimizer state
│
├── roms/                     # Tempest ROM files
│   └── tempest1/             # MAME ROM set
│
├── Code/
│   └── Atari/                # Original Tempest 6502 assembly source (reference)
│
├── tests/                    # Unit tests (pytest)
│   ├── test_state_extraction.py
│   ├── test_avoidance_logic.py
│   └── ...
│
├── requirements.txt
├── startmame.sh              # Convenience script to launch MAME
└── README.md                 # This file
```

---

## Key Configuration

All tunable parameters live in `Scripts/config.py`. Notable ones:

| Parameter                     | Value       | Description                            |
|-------------------------------|-------------|----------------------------------------|
| `state_size`                  | 195         | Number of input features per frame     |
| `num_joint_actions`           | 44          | 4 fire/zap × 11 spinner =total actions |
| `trunk_hidden`                | 384         | Hidden layer width                     |
| `num_atoms`                   | 51          | C51 distribution support size          |
| `v_min / v_max`               | -100 / 100  | Q-value distribution range             |
| `batch_size`                  | 256         | Training batch size                    |
| `lr`                          | 5e-5        | Learning rate                          |
| `n_step`                      | 12          | N-step return horizon                  |
| `gamma`                       | 0.99        | Discount factor                        |
| `memory_size`                 | 15,000,000  | Replay buffer capacity                 |
| `epsilon_decay_frames`        | 500,000     | Frames to decay ε from 1.0 to 0.01     |
| `expert_ratio_decay_frames`   | 5,000,000   | Frames to decay expert usage 50% → 2%  |
| `target_update_period`        | 2,500       | Steps between target network syncs     |

---

## Keyboard Controls

While the Python app is running in an interactive terminal:

| Key     | Action                                     |
|---------|--------------------------------------------|
| `q`     | Quit (saves model first)                   |
| `s`     | Force save model                           |
| `t`     | Toggle training on/off                     |
| `P`     | Toggle epsilon pulse                       |
| `p`     | Toggle epsilon override                    |
| `o`     | Toggle expert override                     |
| `e`     | Toggle expert mode                         |
| `v`     | Toggle verbose logging                     |
| `c`     | Clear screen, reprint header               |
| `h`     | Reprint metrics header                     |
| `Space` | Print one metrics row                      |
| `7/8/9` | Decrease / reset / increase expert ratio   |
| `4/5/6` | Decrease / reset / increase epsilon        |
| `L/l`   | Double / halve learning rate               |
| `a`     | Analyze attention patterns                 |
| `b`     | Print replay buffer stats                  |
| `f`     | Flush replay buffer                        |

---

## License

This project is for educational and research purposes. Tempest is a trademark of Atari. ROM files are not included; you must supply your own legally obtained copies.
=======
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
