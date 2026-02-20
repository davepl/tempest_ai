#!/usr/bin/env python3
# ==================================================================================================================
# ||  TEMPEST AI v2 • APPLICATION ENTRY POINT                                                                    ||
# ||  Boots socket server, spawns keyboard/stats threads, coordinates shutdown.                                   ||
# ==================================================================================================================
"""Tempest AI main entry point — Rainbow-Attention engine."""

import os, sys, time, threading, traceback
import socket
import torch

from aimodel import RainbowAgent, KeyboardHandler, print_with_terminal_restore
from config import RL_CONFIG, MODEL_DIR, LATEST_MODEL_PATH, IS_INTERACTIVE, metrics, SERVER_CONFIG
from metrics_dashboard import MetricsDashboard
from metrics_display import display_metrics_header, display_metrics_row
from socket_server import SocketServer


def _env_enabled(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "off", "no"}


def _has_desktop_session() -> bool:
    # Respect explicit SSH sessions as non-local by default.
    if any(os.getenv(k) for k in ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY")):
        return False

    # Display/session markers used by Linux/*nix desktop sessions.
    if os.getenv("DISPLAY") or os.getenv("WAYLAND_DISPLAY") or os.getenv("MIR_SOCKET"):
        return True
    if os.getenv("XDG_CURRENT_DESKTOP") or os.getenv("DESKTOP_SESSION"):
        return True
    if (os.getenv("XDG_SESSION_TYPE") or "").strip().lower() in {"x11", "wayland", "mir"}:
        return True

    # Windows service session is typically headless.
    if os.name == "nt":
        return (os.getenv("SESSIONNAME") or "").strip().lower() != "services"

    # macOS local terminal sessions usually have a desktop but no DISPLAY.
    if sys.platform == "darwin":
        return True

    # Safe fallback for unknown/non-desktop environments.
    return False


def _resolve_dashboard_host() -> str:
    explicit = os.getenv("TEMPEST_DASHBOARD_HOST", "").strip()
    if explicit:
        return explicit
    return "127.0.0.1" if _has_desktop_session() else "0.0.0.0"


def _best_lan_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            if ip and not ip.startswith("127."):
                return ip
    except Exception:
        pass
    try:
        ip = socket.gethostbyname(socket.gethostname())
        if ip and not ip.startswith("127."):
            return ip
    except Exception:
        pass
    return "127.0.0.1"


def _resolve_dashboard_url_host(bind_host: str) -> str:
    explicit = os.getenv("TEMPEST_DASHBOARD_PUBLIC_HOST", "").strip()
    if explicit:
        return explicit
    if bind_host in {"0.0.0.0", "::", "[::]"}:
        return _best_lan_ip()
    return bind_host


# ── Buffer stats ────────────────────────────────────────────────────────────
def print_buffer_stats(agent, kb):
    try:
        if not hasattr(agent, "memory") or agent.memory is None:
            print("\nNo replay buffer")
            return
        stats = agent.memory.get_partition_stats()
        print("\n" + "=" * 70)
        print("REPLAY BUFFER STATISTICS".center(70))
        print("=" * 70)
        total = stats.get("total_size", 0)
        cap = stats.get("total_capacity", max(1, total))
        print(f"  Total:   {total:>12,} / {cap:>12,}")
        print(f"  Agent:   {stats.get('dqn', 0):>12,}   ({stats.get('frac_dqn', 0)*100:>5.1f}%)")
        print(f"  Expert:  {stats.get('expert', 0):>12,}   ({stats.get('frac_expert', 0)*100:>5.1f}%)")
        print("=" * 70 + "\n")
        if kb and IS_INTERACTIVE:
            kb.set_raw_mode()
    except Exception as e:
        print(f"\nBuffer stats error: {e}")
        if kb and IS_INTERACTIVE:
            kb.set_raw_mode()


# ── Stats reporter thread ──────────────────────────────────────────────────
def stats_reporter(agent, kb):
    print("Starting stats reporter thread...")
    display_metrics_header()
    last = time.time()
    seen = False

    while True:
        try:
            now = time.time()
            if now - last >= 30.0:
                display_metrics_row(agent, kb)
                last = now
            srv = metrics.global_server
            if srv is None:
                time.sleep(0.1)
                continue
            if getattr(srv, "running", False):
                seen = True
            elif seen:
                print("Server stopped, exiting stats reporter")
                break
            time.sleep(0.1)
        except Exception as e:
            print(f"Stats reporter error: {e}")
            traceback.print_exc()
            break


# ── Keyboard handler thread ────────────────────────────────────────────────
def keyboard_handler(agent, kb):
    print("Starting keyboard handler thread...")
    while True:
        try:
            key = kb.check_key()
            if not key:
                time.sleep(0.1)
                continue

            if key == "q":
                print("Quit requested...")
                try:
                    if metrics.global_server:
                        metrics.global_server.running = False
                        metrics.global_server.stop()
                except Exception:
                    pass
                try:
                    agent.stop()
                except Exception:
                    pass
                break
            elif key == "s":
                print("Saving model...")
                agent.save(LATEST_MODEL_PATH, is_forced_save=True)
            elif key == "o":
                metrics.toggle_override(kb)
                display_metrics_row(agent, kb)
            elif key == "e":
                metrics.toggle_expert_mode(kb)
                display_metrics_row(agent, kb)
            elif key == "P":
                metrics.toggle_epsilon_pulse(kb)
                from config import RL_CONFIG
                status = "ON" if RL_CONFIG.plateau_pulse_enabled else "OFF"
                print_with_terminal_restore(kb, f"\nPlateau pulse: {status}")
                display_metrics_row(agent, kb)
            elif key == "p":
                metrics.toggle_epsilon_override(kb)
                display_metrics_row(agent, kb)
            elif key.lower() == "v":
                metrics.toggle_verbose_mode(kb)
                display_metrics_row(agent, kb)
            elif key.lower() == "t":
                metrics.toggle_training_mode(kb)
                agent.training_enabled = metrics.training_enabled
                display_metrics_row(agent, kb)
            elif key.lower() == "c":
                from metrics_display import clear_screen
                clear_screen()
                display_metrics_header()
            elif key.lower() == "h":
                display_metrics_header()
            elif key == " ":
                display_metrics_row(agent, kb)
            elif key == "7":
                metrics.decrease_expert_ratio(kb)
                display_metrics_row(agent, kb)
            elif key == "8":
                metrics.restore_natural_expert_ratio(kb)
                display_metrics_row(agent, kb)
            elif key == "9":
                metrics.increase_expert_ratio(kb)
                display_metrics_row(agent, kb)
            elif key == "4":
                metrics.decrease_epsilon(kb)
                display_metrics_row(agent, kb)
            elif key == "5":
                metrics.restore_natural_epsilon(kb)
                display_metrics_row(agent, kb)
            elif key == "6":
                metrics.increase_epsilon(kb)
                display_metrics_row(agent, kb)
            elif key == "a":
                print_with_terminal_restore(kb, "\nAnalyzing attention patterns...")
                report = agent.diagnose_attention()
                print_with_terminal_restore(kb, report)
            elif key == "r":
                print_with_terminal_restore(kb, "\nResetting attention weights (keeping trunk + heads)...")
                agent.reset_attention_weights()
                display_metrics_row(agent, kb)
            elif key == "b":
                print_buffer_stats(agent, kb)
            elif key == "f":
                print_with_terminal_restore(kb, "\nFlushing replay buffer...")
                agent.flush_replay_buffer()
                print_with_terminal_restore(kb, "Replay buffer flushed.")
                display_metrics_row(agent, kb)
            elif key == "L":
                RL_CONFIG.lr = min(1e-2, RL_CONFIG.lr * 2.0)
                print_with_terminal_restore(kb, f"LR increased to {RL_CONFIG.lr:.2e}")
                display_metrics_row(agent, kb)
            elif key == "l":
                RL_CONFIG.lr = max(1e-6, RL_CONFIG.lr / 2.0)
                print_with_terminal_restore(kb, f"LR decreased to {RL_CONFIG.lr:.2e}")
                display_metrics_row(agent, kb)

            time.sleep(0.1)
        except BlockingIOError:
            time.sleep(0.1)
            continue
        except Exception as e:
            try:
                print(f"Keyboard error: {e}")
            except BlockingIOError:
                pass
            break


# ── Network info ────────────────────────────────────────────────────────────
def print_network_info(agent):
    print("\n" + "=" * 90)
    print("TEMPEST AI v2 — Rainbow-Attention Engine".center(90))
    print("=" * 90)

    net = agent.online_net
    tp = sum(p.numel() for p in net.parameters())
    tr = sum(p.numel() for p in net.parameters() if p.requires_grad)

    print(f"\n📐 Architecture:")
    print(f"   State size:       {agent.state_size}")
    print(f"   Actions:          {RL_CONFIG.num_firezap_actions} fire/zap × {RL_CONFIG.num_spinner_actions} spinner = {RL_CONFIG.num_joint_actions}")
    print(f"   Trunk:            {RL_CONFIG.trunk_layers} layers × {RL_CONFIG.trunk_hidden} hidden")
    print(f"   Enemy attention:  {'ON' if RL_CONFIG.use_enemy_attention else 'OFF'} ({RL_CONFIG.attn_heads} heads, dim={RL_CONFIG.attn_dim})")
    print(f"   Distributional:   {'C51 ({} atoms, [{}, {}])'.format(RL_CONFIG.num_atoms, RL_CONFIG.v_min, RL_CONFIG.v_max) if RL_CONFIG.use_distributional else 'OFF'}")
    print(f"   Dueling:          {'ON' if RL_CONFIG.use_dueling else 'OFF'}")
    print(f"   Parameters:       {tp:,} total, {tr:,} trainable")

    print(f"\n⚙️  Training:")
    print(f"   LR:               {RL_CONFIG.lr:.2e} → {RL_CONFIG.lr_min:.2e} (cosine)")
    print(f"   Batch size:       {RL_CONFIG.batch_size}")
    print(f"   γ = {RL_CONFIG.gamma},  n-step = {RL_CONFIG.n_step}")
    print(f"   PER α={RL_CONFIG.priority_alpha}, β={RL_CONFIG.priority_beta_start}→1.0")
    print(f"   Target update:    every {RL_CONFIG.target_update_period} steps (hard)")
    print(f"   Grad clip:        {RL_CONFIG.grad_clip_norm}")

    print(f"\n🎓 Exploration:")
    print(f"   ε:    {RL_CONFIG.epsilon_start} → {RL_CONFIG.epsilon_end} over {RL_CONFIG.epsilon_decay_frames:,} frames")
    print(f"   Expert: {RL_CONFIG.expert_ratio_start*100:.0f}% → {RL_CONFIG.expert_ratio_end*100:.0f}% over {RL_CONFIG.expert_ratio_decay_frames:,} frames")
    print(f"   BC weight: {RL_CONFIG.expert_bc_weight} → {RL_CONFIG.expert_bc_min_weight}")

    print(f"\n⌨️  Keys: [q]uit [s]ave [c]lear [h]eader [space]row [o]override [e]xpert [p]epsilon [t]rain [v]erbose [a]ttention")
    print(f"   [7/8/9] expert−/reset/+   [4/5/6] epsilon−/reset/+   [b] buffer stats   [f] flush buffer")
    print("\n" + "=" * 90 + "\n")


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    agent = RainbowAgent(state_size=RL_CONFIG.state_size)
    dev = getattr(agent.device, "type", "unknown")
    print(f"🧮 Device: {dev.upper()}")

    print_network_info(agent)

    dashboard = None

    if os.path.exists(LATEST_MODEL_PATH):
        loaded = agent.load(LATEST_MODEL_PATH)
        if loaded:
            print(f"✓ Loaded model from: {LATEST_MODEL_PATH}\n")
        else:
            print("⚠ Model load failed/incompatible, starting fresh\n")
    else:
        print("⚠ No model found, starting fresh\n")

    dashboard_enabled = _env_enabled("TEMPEST_DASHBOARD", True)
    dashboard_host = _resolve_dashboard_host()
    desktop_session = _has_desktop_session()
    if os.getenv("TEMPEST_DASHBOARD_BROWSER") is None:
        dashboard_open_browser = desktop_session
    else:
        dashboard_open_browser = _env_enabled("TEMPEST_DASHBOARD_BROWSER", desktop_session)
    try:
        dashboard_port = int(os.getenv("TEMPEST_DASHBOARD_PORT", "8765"))
    except Exception:
        dashboard_port = 8765
    if dashboard_enabled:
        try:
            dashboard = MetricsDashboard(
                metrics_obj=metrics,
                agent_obj=agent,
                host=dashboard_host,
                port=dashboard_port,
                open_browser=dashboard_open_browser,
            )
            dashboard.start()
            dashboard_url_host = _resolve_dashboard_url_host(dashboard.host)
            dashboard_url = f"http://{dashboard_url_host}:{dashboard.port}"
            print(f"📊 Metrics dashboard: {dashboard_url}")
            if dashboard.host != dashboard_url_host:
                print(f"   Bound on {dashboard.host}:{dashboard.port}")
        except Exception as e:
            dashboard = None
            print(f"⚠ Dashboard startup failed: {e}")

    server = SocketServer(SERVER_CONFIG.host, SERVER_CONFIG.port, agent, metrics)
    metrics.global_server = server
    metrics.client_count = 0

    srv_thread = threading.Thread(target=server.start, daemon=True)
    srv_thread.start()

    kb = None
    if IS_INTERACTIVE:
        kb = KeyboardHandler()
        kb.setup_terminal()
        threading.Thread(target=keyboard_handler, args=(agent, kb), daemon=True).start()

    threading.Thread(target=stats_reporter, args=(agent, kb), daemon=True).start()

    last_save = time.time()
    try:
        while srv_thread.is_alive() and not server.shutdown_event.is_set():
            if time.time() - last_save >= 300:
                # Quiet periodic autosave to keep metrics rows clean.
                agent.save(LATEST_MODEL_PATH, show_status=False)
                last_save = time.time()
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt, shutting down...")
    finally:
        agent.save(LATEST_MODEL_PATH)
        print("Final model saved")
        if IS_INTERACTIVE and kb:
            kb.restore_terminal()
        try:
            server.stop()
        except Exception:
            pass
        try:
            agent.stop()
        except Exception:
            pass
        try:
            srv_thread.join(timeout=2.0)
        except Exception:
            pass
        try:
            if dashboard:
                dashboard.stop()
        except Exception:
            pass
        print("Shutdown complete")


if __name__ == "__main__":
    main()
