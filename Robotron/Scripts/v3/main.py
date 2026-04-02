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
import select as _select
import torch

from .config import CONFIG, GAME_SETTINGS, MODEL_DIR, CHECKPOINT_PATH
from .agent import PPOAgent
from .socket_server import SocketServer
from .metrics_display import display_metrics_header, display_metrics_row
from .metrics_dashboard import MetricsDashboard


# ── Keyboard handler (non-blocking stdin) ───────────────────────────────────

_termios = _tty = _fcntl = None
try:
    import termios as _termios, tty as _tty, fcntl as _fcntl
except ImportError:
    pass


def _env_enabled(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "off", "no"}


class _KeyboardHandler:
    def __init__(self):
        self.fd = None
        self.old_settings = None
        if sys.stdin.isatty() and _termios:
            try:
                self.fd = sys.stdin.fileno()
                self.old_settings = _termios.tcgetattr(self.fd)
            except Exception:
                self.fd = None

    def start(self):
        if self.fd is not None and _tty and _fcntl:
            try:
                # Cbreak keeps single-character input without disabling normal
                # output newline handling for the rest of the terminal.
                if hasattr(_tty, "setcbreak"):
                    _tty.setcbreak(self.fd)
                else:
                    _tty.setraw(self.fd)
                flags = _fcntl.fcntl(self.fd, _fcntl.F_GETFL)
                _fcntl.fcntl(self.fd, _fcntl.F_SETFL, flags | os.O_NONBLOCK)
            except Exception:
                pass

    def check_key(self) -> str | None:
        if self.fd is None:
            return None
        try:
            if _select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                return sys.stdin.read(1)
        except Exception:
            pass
        return None

    def restore(self):
        if self.fd is not None and self.old_settings and _termios:
            try:
                _termios.tcsetattr(self.fd, _termios.TCSADRAIN, self.old_settings)
            except Exception:
                pass

    def _safe_print(self, *args, **kwargs):
        self.restore()
        try:
            print(*args, **kwargs, flush=True)
        except Exception:
            pass
        self.start()

# ── Banner ──────────────────────────────────────────────────────────────────

BANNER = """
╔═══════════════════════════════════════════════════════════════════╗
║  ROBOTRON AI v3 — Set Transformer + PPO                         ║
║  Neurosymbolic RL with Potential Field Expert Guidance           ║
╚═══════════════════════════════════════════════════════════════════╝
"""


def main():
    if any(arg.startswith("-Xgil") for arg in sys.argv[1:]):
        print("Note: `-Xgil=0` must appear before `-m v3.main` to affect the interpreter.")
    print(BANNER)

    # Device info
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
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
    dashboard = None

    # Graceful shutdown
    shutdown_event = threading.Event()

    def signal_handler(sig, frame):
        print("\nShutting down...")
        shutdown_event.set()
        server.stop()
        if dashboard is not None:
            dashboard.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Dashboard (web UI on port 8796)
    dash_host = os.environ.get("ROBOTRON_DASHBOARD_HOST", "0.0.0.0")
    dash_port = int(os.environ.get("ROBOTRON_DASHBOARD_PORT", "8796"))
    if os.environ.get("ROBOTRON_DASHBOARD_BROWSER") is not None:
        open_browser = _env_enabled("ROBOTRON_DASHBOARD_BROWSER", False)
    elif os.environ.get("ROBOTRON_NO_BROWSER") is not None:
        open_browser = not _env_enabled("ROBOTRON_NO_BROWSER", False)
    else:
        # Browser auto-launch creates a web client immediately, which turns on
        # preview capture and can materially reduce training throughput.
        open_browser = False
    dashboard = MetricsDashboard(
        metrics_obj=server.metrics,
        agent_obj=agent,
        host=dash_host,
        port=dash_port,
        open_browser=open_browser,
    )
    dashboard.start()

    # Tabular status reporter
    def status_reporter():
        display_metrics_header()
        while not shutdown_event.is_set():
            shutdown_event.wait(10.0)
            if shutdown_event.is_set():
                break
            display_metrics_row(server.metrics, agent)

    reporter = threading.Thread(target=status_reporter, daemon=True)
    reporter.start()

    # Keyboard handler for interactive hotkeys
    kb = _KeyboardHandler()
    kb.start()

    def keyboard_listener():
        while not shutdown_event.is_set():
            try:
                key = kb.check_key()
                if not key:
                    time.sleep(0.1)
                    continue
                if key == "q":
                    kb._safe_print("\nQuit requested...")
                    shutdown_event.set()
                    server.stop()
                    dashboard.stop()
                    break
                elif key == "h":
                    kb.restore()
                    display_metrics_header()
                    kb.start()
                elif key == " ":
                    kb.restore()
                    display_metrics_row(server.metrics, agent)
                    kb.start()
                elif key == "s":
                    kb._safe_print("Saving checkpoint...")
                    agent.save()
                    kb._safe_print("Saved.")
                elif key.lower() == "c":
                    kb.restore()
                    print("\033[2J\033[H", end="", flush=True)
                    display_metrics_header()
                    kb.start()
                # ── Expert ratio controls (numpad row 7/8/9) ────────
                elif key == "7":
                    msg = agent.decrease_expert_ratio()
                    kb._safe_print(msg)
                elif key == "8":
                    msg = agent.restore_natural_expert_ratio()
                    kb._safe_print(msg)
                elif key == "9":
                    msg = agent.increase_expert_ratio()
                    kb._safe_print(msg)
                # ── Epsilon controls (numpad row 4/5/6) ─────────────
                elif key == "4":
                    msg = agent.decrease_epsilon()
                    kb._safe_print(msg)
                elif key == "5":
                    msg = agent.restore_natural_epsilon()
                    kb._safe_print(msg)
                elif key == "6":
                    msg = agent.increase_epsilon()
                    kb._safe_print(msg)
                # ── Toggle controls ─────────────────────────────────
                elif key == "o":
                    msg = agent.toggle_override()
                    kb._safe_print(msg)
                elif key == "e":
                    msg = agent.toggle_expert_mode()
                    kb._safe_print(msg)
                elif key == "t":
                    msg = agent.toggle_training()
                    kb._safe_print(msg)
                elif key == "v":
                    msg = agent.toggle_verbose()
                    kb._safe_print(msg)
                # ── Learning rate ───────────────────────────────────
                elif key == "L":
                    for pg in agent.optimizer.param_groups:
                        pg["lr"] = min(1e-2, pg["lr"] * 2.0)
                    lr = agent.optimizer.param_groups[0]["lr"]
                    kb._safe_print(f"LR increased to {lr:.2e}")
                elif key == "l":
                    for pg in agent.optimizer.param_groups:
                        pg["lr"] = max(1e-7, pg["lr"] / 2.0)
                    lr = agent.optimizer.param_groups[0]["lr"]
                    kb._safe_print(f"LR decreased to {lr:.2e}")
                elif key == "?":
                    kb._safe_print(
                        "\n  Hotkeys:\n"
                        "    h=header  SPACE=row  s=save  c=clear  q=quit\n"
                        "    7/8/9 = expert ratio  down/natural/up\n"
                        "    4/5/6 = epsilon       down/natural/up\n"
                        "    o=override(expert→0)  e=expert(→100%)\n"
                        "    t=training on/off  v=verbose  L/l=LR up/down\n"
                        "    ?=this help"
                    )
                time.sleep(0.05)
            except Exception:
                time.sleep(0.1)

    kb_thread = threading.Thread(target=keyboard_listener, daemon=True)
    kb_thread.start()

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
        kb.restore()
        print("Saving final checkpoint...")
        agent.save()
        print("Done.")


if __name__ == "__main__":
    main()
