#!/usr/bin/env python3
# ==================================================================================================================
# ||                                                                                                              ||
# ||                                   TEMPEST AI • APPLICATION ENTRY POINT                                       ||
# ||                                                                                                              ||
# ||  FILE: Scripts/main.py                                                                                       ||
# ||  ROLE: Boots the socket server, spawns keyboard and stats threads, and coordinates graceful shutdown.         ||
# ||                                                                                                              ||
# ||  NEED TO KNOW:                                                                                               ||
# ||   - Creates model dir; instantiates DiscreteDQNAgent; loads latest model if present.                         ||
# ||   - Starts SocketServer (Lua <-> Python bridge) and metrics display loop.                                    ||
# ||   - Keyboard controls: save (s), quit (q), toggles (o,e,p,t,v), LR adjust (l/L), bucket stats (b).          ||
# ||                                                                                                              ||
# ||  CONSUMES: RL_CONFIG, MODEL_DIR, LATEST_MODEL_PATH, SERVER_CONFIG, metrics                                   ||
# ||  PRODUCES: running server, periodic metrics rows, on-exit model save                                         ||
# ||                                                                                                              ||
# ==================================================================================================================
"""
Tempest AI Main Entry Point
Coordinates the socket server, metrics display, and keyboard handling.
"""

import os
import time
import math
import threading
from datetime import datetime
import traceback
import torch

from aimodel import DiscreteDQNAgent, KeyboardHandler
from config import RL_CONFIG, MODEL_DIR, LATEST_MODEL_PATH, IS_INTERACTIVE, metrics, SERVER_CONFIG

from metrics_display import display_metrics_header, display_metrics_row
from socket_server import SocketServer


def print_bucket_stats(agent, kb_handler):
    """Print replay buffer statistics (Stratified: Agent vs Expert)."""
    try:
        if not hasattr(agent, 'memory') or agent.memory is None:
            print("\nNo replay buffer available")
            return

        stats = agent.memory.get_partition_stats()
        
        print("\n" + "=" * 90)
        print(" " * 30 + "REPLAY BUFFER STATISTICS")
        print("=" * 90)

        print(f"\n{'OVERALL STATISTICS':<40}")
        print("-" * 90)
        total_size = stats.get('total_size', 0)
        total_capacity = stats.get('total_capacity', max(1, total_size))
        
        print(f"  Total Size:          {total_size:>12,} / {total_capacity:>12,}")
        print(f"  Agent Experiences:   {stats.get('dqn', 0):>12,}   ({stats.get('frac_dqn', 0.0)*100:>5.1f}%)")
        print(f"  Expert Experiences:  {stats.get('expert', 0):>12,}   ({stats.get('frac_expert', 0.0)*100:>5.1f}%)")

        print("\n" + "=" * 90 + "\n")

        if kb_handler and IS_INTERACTIVE:
            kb_handler.set_raw_mode()

    except Exception as exc:
        print(f"\nError printing buffer stats: {exc}")
        traceback.print_exc()
        if kb_handler and IS_INTERACTIVE:
            kb_handler.set_raw_mode()

def stats_reporter(agent, kb_handler):
    """Thread function to report stats periodically"""
    print("Starting stats reporter thread...")

    # NOTE: The model is loaded once in main() before the server starts.
    # Re-loading here can race with live frame processing and reset restored counters.
    last_report = time.time()
    report_interval = 30.0  # Print every 30 seconds
    seen_server_running = False
    
    # Display the header once at the beginning
    display_metrics_header()
    
    while True:
        try:
            current_time = time.time()
            if current_time - last_report >= report_interval:
                display_metrics_row(agent, kb_handler)
                last_report = current_time
            
            # Exit only after server has run at least once and then stops.
            # This avoids startup races where the reporter checks before server.start() sets running=True.
            server_ref = metrics.global_server
            if server_ref is None:
                time.sleep(0.1)
                continue
            if getattr(server_ref, "running", False):
                seen_server_running = True
            elif seen_server_running:
                print("Server stopped running, exiting stats reporter")
                break
                
            time.sleep(0.1)
        except Exception as e:
            print(f"Error in stats reporter: {e}")
            traceback.print_exc()
            break

def keyboard_input_handler(agent, keyboard_handler):
    """Thread function to handle keyboard input"""
    print("Starting keyboard input handler thread...")
    
    while True:
        try:
            # Check for keyboard input
            key = keyboard_handler.check_key()
            
            if key:
                # Handle different keys
                if key == 'q':
                    print("Quit command received, shutting down...")
                    try:
                        if metrics.global_server:
                            metrics.global_server.running = False
                            metrics.global_server.stop()
                    except Exception:
                        pass
                    try:
                        if agent:
                            agent.stop()
                    except Exception:
                        pass
                    break
                elif key == 's':
                    print("Save command received, saving model...")
                    agent.save(LATEST_MODEL_PATH, is_forced_save=True)
                elif key == 'o':
                    metrics.toggle_override(keyboard_handler)
                    display_metrics_row(agent, keyboard_handler)
                elif key == 'e':
                    metrics.toggle_expert_mode(keyboard_handler)
                    display_metrics_row(agent, keyboard_handler)
                elif key.lower() == 'p':
                    metrics.toggle_epsilon_override(keyboard_handler)
                    display_metrics_row(agent, keyboard_handler)
                elif key.lower() == 'v':
                    metrics.toggle_verbose_mode(keyboard_handler)
                    display_metrics_row(agent, keyboard_handler)
                elif key.lower() == 't':
                    metrics.toggle_training_mode(keyboard_handler)
                    # Propagate to agent
                    try:
                        agent.training_enabled = metrics.training_enabled
                    except Exception:
                        pass
                    display_metrics_row(agent, keyboard_handler)
                elif key.lower() == 'c':
                    from metrics_display import clear_screen
                    clear_screen()
                    display_metrics_header()
                elif key.lower() == 'h':
                    # Do hard target update before displaying header
                    # agent.update_target_network() # Not exposed in DiscreteDQNAgent yet, maybe add?
                    display_metrics_header()
                elif key == ' ':  # Handle space key
                    # Print only one row (no header)
                    display_metrics_row(agent, keyboard_handler)
                elif key == '7':
                    metrics.decrease_expert_ratio(keyboard_handler)
                    display_metrics_row(agent, keyboard_handler)
                elif key == '8':
                    metrics.restore_natural_expert_ratio(keyboard_handler)
                    display_metrics_row(agent, keyboard_handler)
                elif key == '9':
                    metrics.increase_expert_ratio(keyboard_handler)
                    display_metrics_row(agent, keyboard_handler)
                elif key == '4':
                    metrics.decrease_epsilon(keyboard_handler)
                    display_metrics_row(agent, keyboard_handler)
                elif key == '5':
                    metrics.restore_natural_epsilon(keyboard_handler)
                    display_metrics_row(agent, keyboard_handler)
                elif key == '6':
                    metrics.increase_epsilon(keyboard_handler)
                    display_metrics_row(agent, keyboard_handler)
                elif key == 'b':
                    # Print replay buffer statistics
                    print_bucket_stats(agent, keyboard_handler)
            
            time.sleep(0.1)
        except Exception as e:
            print(f"Error in keyboard input handler: {e}")
            break

def print_network_config(agent):
    """Display network architecture and key hyperparameters at startup"""
    print("\n" + "="*100)
    print("TEMPEST AI - NETWORK CONFIGURATION".center(100))
    print("="*100)
    
    # Network Architecture
    print("\n📐 NETWORK ARCHITECTURE:")
    print(f"   State Size:        {agent.state_size}")
    print(f"   Action Space:      Joint Discrete (4 FireZap x 64 Spinner = 256)")
    
    # Get layer sizes from the network
    print(f"\n   Shared Trunk:      {len(agent.qnetwork_local.shared_layers)} layers")
    for i, layer in enumerate(agent.qnetwork_local.shared_layers):
        if isinstance(layer, torch.nn.Linear):
            print(f"      Layer {i+1}:        {layer.in_features} → {layer.out_features}")
    
    # Head architecture
    if getattr(agent.qnetwork_local, 'use_dueling', False):
        print("\n   Q Head:            Dueling (Value + Advantage) → 256")
    else:
        try:
            print(
                f"\n   Q Head:            "
                f"{agent.qnetwork_local.q_fc.in_features} → {agent.qnetwork_local.q_fc.out_features} → 256"
            )
        except Exception:
            print("\n   Q Head:            256 outputs")
    
    # Count total parameters
    total_params = sum(p.numel() for p in agent.qnetwork_local.parameters())
    trainable_params = sum(p.numel() for p in agent.qnetwork_local.parameters() if p.requires_grad)
    print(f"\n   Total Parameters:  {total_params:,}")
    print(f"   Trainable:         {trainable_params:,}")
    
    # Training Hyperparameters
    print("\n⚙️  TRAINING HYPERPARAMETERS:")
    print(f"   Learning Rate:     {agent.learning_rate:.6f}")
    print(f"   Batch Size:        {agent.batch_size:,}")
    print(f"   Gamma (γ):         {agent.gamma}")
    print(f"   Epsilon (ε):       {agent.epsilon} (exploration)")
    print(f"   Memory Size:       {agent.memory.capacity:,} transitions")
    
    # Loss Configuration
    print("\n⚖️  LOSS CONFIGURATION:")
    print(f"   Loss Function:     SmoothL1Loss (Huber)")
    print(f"   Total Loss:        Joint Q-Loss (+ optional expert imitation)")
    
    # Expert Configuration
    print("\n🎓 EXPERT GUIDANCE:")
    print(f"   Expert Ratio:      {RL_CONFIG.expert_ratio_start*100:.0f}%")
    
    # Optimization
    print("\n🚀 OPTIMIZATION:")
    print(f"   Gradient Clip:     {float(getattr(RL_CONFIG, 'grad_clip_norm', 10.0) or 10.0):.1f} (max norm)")
    print(f"   Training Workers:  1 (serialized optimizer)")
    
    # Keyboard Controls
    print("\n⌨️  KEYBOARD CONTROLS:")
    print(f"   [q] Quit           [s] Save Model       [c] Clear Screen")
    print(f"   [o] Override       [e] Expert Mode      [p] Force Epsilon")
    print(f"   [t] Training       [v] Verbose          [space] Print Row")
    print(f"   [7] Dec Expert     [8] Reset Expert     [9] Inc Expert")
    print(f"   [4] Dec Epsilon    [5] Reset Epsilon    [6] Inc Epsilon")
    print(f"   [b] Buffer Stats   - Display replay buffer statistics")
    
    print("\n" + "="*100 + "\n")

def main():
    """Main function to run the Tempest AI application"""

    # Create model directory if it doesn't exist
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    # Initialize the Agent
    agent = DiscreteDQNAgent(
        state_size       = RL_CONFIG.state_size,
        learning_rate    = RL_CONFIG.lr,
        gamma            = RL_CONFIG.gamma,
        epsilon          = RL_CONFIG.epsilon,
        memory_size      = RL_CONFIG.memory_size,
        batch_size       = RL_CONFIG.batch_size
    )
    # Surface which accelerator PyTorch selected so slowdowns are obvious
    device_label = getattr(getattr(agent, "device", None), "type", "unknown")
    print(f"🧮 Device:          {device_label.upper()}")

    # Display network configuration and hyperparameters
    print_network_config(agent)

    # Load the model if it exists
    if os.path.exists(LATEST_MODEL_PATH):
        agent.load(LATEST_MODEL_PATH)
        print(f"✓ Loaded model from: {LATEST_MODEL_PATH}\n")
    else:
        print(f"⚠ No existing model found, starting fresh\n")

    # Initialize the socket server
    server = SocketServer(SERVER_CONFIG.host, SERVER_CONFIG.port, agent, metrics)
    
    # Set the global server reference in metrics
    metrics.global_server = server
    
    # Initialize client_count in metrics
    metrics.client_count = 0
    
    # Start the server in a separate thread
    server_thread = threading.Thread(target=server.start)
    server_thread.daemon = True
    server_thread.start()
    
    # Set up keyboard handler for interactive mode
    keyboard_handler = None
    if IS_INTERACTIVE:
        keyboard_handler = KeyboardHandler()
        keyboard_handler.setup_terminal()
        keyboard_thread = threading.Thread(target=keyboard_input_handler, args=(agent, keyboard_handler))
        keyboard_thread.daemon = True
        keyboard_thread.start()
    
    # Start the stats reporter in a separate thread
    stats_thread = threading.Thread(target=stats_reporter, args=(agent, keyboard_handler))
    stats_thread.daemon = True
    stats_thread.start()
    
    # Track last save time
    last_save_time = time.time()
    save_interval = 300  # 5 minutes in seconds
    
    try:
        # Keep the main thread alive while server thread is active.
        # Do not gate this on server.running directly (it is set in the server thread and can race at startup).
        while server_thread.is_alive() and not server.shutdown_event.is_set():
            current_time = time.time()
            # Save model every 5 minutes
            if current_time - last_save_time >= save_interval:
                agent.save(LATEST_MODEL_PATH)
                last_save_time = current_time
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received, saving and shutting down...")
    
    finally:
        # Save the model before exiting
        agent.save(LATEST_MODEL_PATH)
        print("Final model state saved")
        
        # Restore terminal settings
        if IS_INTERACTIVE and keyboard_handler:
            keyboard_handler.restore_terminal()
        
        # Stop server and agent gracefully
        try:
            if server:
                server.stop()
        except Exception:
            pass
        try:
            if agent:
                agent.stop()
        except Exception:
            pass
        
        # Join server thread to avoid abrupt abort on exit
        try:
            server_thread.join(timeout=2.0)
        except Exception:
            pass
        
        print("Application shutdown complete")

if __name__ == "__main__":
    main()
