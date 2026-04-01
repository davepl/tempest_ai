#!/usr/bin/env python3
"""Robotron AI v3 — Main entry point.

Boots the PPO agent, socket server, and training coordinator.
Run from the Robotron/Scripts directory:
    python -m v3.main
"""

import os
import sys
import time
import signal
import threading
import torch

from .config import CONFIG, GAME_SETTINGS, MODEL_DIR, CHECKPOINT_PATH
from .agent import PPOAgent
from .socket_server import SocketServer

# ── Banner ──────────────────────────────────────────────────────────────────

BANNER = """
╔═══════════════════════════════════════════════════════════════════╗
║  ROBOTRON AI v3 — Set Transformer + PPO                         ║
║  Neurosymbolic RL with Potential Field Expert Guidance           ║
╚═══════════════════════════════════════════════════════════════════╝
"""


def main():
    print(BANNER)

    # Device info
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_mem / 1024**3:.1f} GB)")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("  Device: Apple Metal (MPS)")
    else:
        print("  Device: CPU")

    print(f"  Model dir: {MODEL_DIR}")
    print(f"  Server: {CONFIG.server.host}:{CONFIG.server.port}")
    print()

    # Initialize agent
    agent = PPOAgent()

    # Try to load existing checkpoint
    if CHECKPOINT_PATH.exists():
        if agent.load():
            print(f"Resumed from {agent.total_frames:,} frames")
        else:
            print("Starting fresh (checkpoint load failed)")
    else:
        print("Starting fresh (no checkpoint found)")

    # Load game settings
    GAME_SETTINGS.load()

    print(f"\n  Expert ratio: {agent.get_expert_ratio():.1%}")
    print(f"  Epsilon: {agent.get_epsilon():.3f}")
    print(f"  BC weight: {agent._get_bc_weight():.3f}")
    print(f"  LR: {CONFIG.train.lr:.1e}")
    print()

    # Socket server
    server = SocketServer(agent)

    # Graceful shutdown
    shutdown_event = threading.Event()

    def signal_handler(sig, frame):
        print("\nShutting down...")
        shutdown_event.set()
        server.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Status reporter thread
    def status_reporter():
        while not shutdown_event.is_set():
            shutdown_event.wait(30.0)
            if shutdown_event.is_set():
                break
            m = server.metrics
            expert_r = agent.get_expert_ratio()
            eps = agent.get_epsilon()
            bc_w = agent._get_bc_weight()
            lr = agent.optimizer.param_groups[0]["lr"]

            print(
                f"[v3] Frames: {m.total_frames:>12,} | "
                f"FPS: {m.fps:>7.1f} | "
                f"AvgRwd: {m.avg_reward:>8.1f} | "
                f"EpLen: {m.avg_ep_len:>6.1f} | "
                f"Peak: {m.peak_game_score:>8,} | "
                f"Expert: {expert_r:>5.1%} | "
                f"Eps: {eps:>5.3f} | "
                f"BC: {bc_w:>5.3f} | "
                f"LR: {lr:.1e} | "
                f"Loss: {agent.last_loss:>8.6f}"
            )

    reporter = threading.Thread(target=status_reporter, daemon=True)
    reporter.start()

    # Auto-save thread
    def auto_saver():
        while not shutdown_event.is_set():
            shutdown_event.wait(300.0)  # save every 5 minutes
            if shutdown_event.is_set():
                break
            agent.save()
            print("[v3] Auto-saved checkpoint")

    saver = threading.Thread(target=auto_saver, daemon=True)
    saver.start()

    # Run server (blocking)
    try:
        server.start()
    except KeyboardInterrupt:
        pass
    finally:
        print("Saving final checkpoint...")
        agent.save()
        print("Done.")


if __name__ == "__main__":
    main()
