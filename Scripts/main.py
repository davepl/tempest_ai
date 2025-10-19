#!/usr/bin/env python3
# ==================================================================================================================
# ||                                                                                                              ||
# ||                                   TEMPEST AI • APPLICATION ENTRY POINT                                       ||
# ||                                                                                                              ||
# ||  FILE: Scripts/main.py                                                                                       ||
# ||  ROLE: Boots the socket server, spawns keyboard and stats threads, and coordinates graceful shutdown.         ||
# ||                                                                                                              ||
# ||  NEED TO KNOW:                                                                                               ||
# ||   - Creates model dir; instantiates HybridDQNAgent; loads latest model if present.                           ||
# ||   - Starts SocketServer (Lua <-> Python bridge) and metrics display loop.                                    ||
# ||   - Keyboard controls: save (s), quit (q), toggles (o,e,p,t,v), LR adjust (l/L), header (c).                ||
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
import threading
from datetime import datetime
import traceback
import torch

from aimodel import HybridDQNAgent, KeyboardHandler
from config import RL_CONFIG, MODEL_DIR, LATEST_MODEL_PATH, IS_INTERACTIVE, metrics, SERVER_CONFIG

from metrics_display import display_metrics_header, display_metrics_row
from socket_server import SocketServer

def stats_reporter(agent, kb_handler):
    """Thread function to report stats periodically"""
    print("Starting stats reporter thread...")
    
    # Load the model if it exists
    if os.path.exists(LATEST_MODEL_PATH):
        agent.load(LATEST_MODEL_PATH)
    last_report = time.time()
    report_interval = 30.0  # Print every 30 seconds
    
    # Display the header once at the beginning
    display_metrics_header()
    
    while True:
        try:
            current_time = time.time()
            if current_time - last_report >= report_interval:
                display_metrics_row(agent, kb_handler)
                last_report = current_time
            
            # Check if server is still running
            if metrics.global_server is None or not metrics.global_server.running:
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
                            agent.stop(join=True, timeout=2.0)
                    except Exception:
                        pass
                    break
                elif key == 's':
                    print("Save command received, saving model...")
                    agent.save(LATEST_MODEL_PATH)
                    print(f"Model saved to {LATEST_MODEL_PATH}")
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
                        agent.set_training_enabled(metrics.training_enabled)
                    except Exception:
                        pass
                    display_metrics_row(agent, keyboard_handler)
                elif key.lower() == 'c':
                    from metrics_display import clear_screen
                    clear_screen()
                    display_metrics_header()
                elif key.lower() == 'h':
                    # Do hard target update before displaying header
                    agent.update_target_network()
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
                elif key == 'L':
                    agent.adjust_learning_rate(0.00005, keyboard_handler)
                    display_metrics_row(agent, keyboard_handler)
                elif key == 'l':
                    agent.adjust_learning_rate(-0.00005, keyboard_handler)
                    display_metrics_row(agent, keyboard_handler)
            
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
    print(f"   Discrete Actions:  {agent.discrete_actions} (FIRE/ZAP combinations)")
    print(f"   Continuous Output: 1 (Spinner: -0.9 to +0.9)")
    
    # Get layer sizes from the network
    print(f"\n   Shared Trunk:      {len(agent.qnetwork_local.shared_layers)} layers")
    for i, layer in enumerate(agent.qnetwork_local.shared_layers):
        if isinstance(layer, torch.nn.Linear):
            print(f"      Layer {i+1}:        {layer.in_features} → {layer.out_features}")
    
    # Head architectures
    shared_out = agent.qnetwork_local.shared_layers[-1].out_features if agent.qnetwork_local.shared_layers else 0
    discrete_hidden = agent.qnetwork_local.discrete_fc.out_features
    continuous_hidden = agent.qnetwork_local.continuous_fc1.out_features
    
    print(f"\n   Discrete Head:     {shared_out} → {discrete_hidden} → {agent.discrete_actions}")
    print(f"   Continuous Head:   {shared_out} → {continuous_hidden} → {agent.qnetwork_local.continuous_fc2.out_features} → 1")
    
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
    print(f"   Epsilon (ε):       {agent.epsilon} → {agent.epsilon_min} (exploration)")
    print(f"   Memory Size:       {agent.memory.capacity:,} transitions")
    print(f"   Target Update:     Every {RL_CONFIG.target_update_freq} steps")
    
    # Loss Configuration
    print("\n⚖️  LOSS CONFIGURATION:")
    print(f"   Discrete Loss:     TD (Huber) + BC (Cross-Entropy)")
    print(f"   Continuous Loss:   MSE with advantage weighting")
    print(f"   Loss Weights:      Discrete={RL_CONFIG.discrete_loss_weight:.1f}, Continuous={RL_CONFIG.continuous_loss_weight:.1f}")
    bc_weight = getattr(RL_CONFIG, 'discrete_bc_weight', 1.0)
    print(f"   BC Weight:         {bc_weight:.1f} (behavioral cloning)")
    max_q = getattr(RL_CONFIG, 'max_q_value', None)
    print(f"   Max Q-Value Clip:  {max_q:.1f}" if max_q else "   Max Q-Value Clip:  None")
    td_clip = getattr(RL_CONFIG, 'td_target_clip', None)
    print(f"   TD Target Clip:    {td_clip:.1f}" if td_clip else "   TD Target Clip:    None")
    
    # Expert Configuration
    print("\n🎓 EXPERT GUIDANCE:")
    print(f"   Expert Ratio:      {RL_CONFIG.expert_ratio_start*100:.0f}%")
    print(f"   Superzap Gate:     {'Enabled' if RL_CONFIG.enable_superzap_gate else 'Disabled'}")
    
    # Optimization
    print("\n🚀 OPTIMIZATION:")
    print(f"   Gradient Clip:     10.0 (max norm)")
    print(f"   N-Step Returns:    {RL_CONFIG.n_step}-step")
    print(f"   Training Workers:  {RL_CONFIG.training_workers}")
    
    print("\n" + "="*100 + "\n")

def main():
    """Main function to run the Tempest AI application"""

    # Create model directory if it doesn't exist
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    # Initialize the Agent
    # Use HybridDQNAgent (4 discrete fire/zap + 1 continuous spinner)
    
    agent = HybridDQNAgent(
        state_size       = RL_CONFIG.state_size,
        discrete_actions = 4,
        learning_rate    = RL_CONFIG.lr,
        gamma            = RL_CONFIG.gamma,
        epsilon          = RL_CONFIG.epsilon,
        epsilon_min      = RL_CONFIG.epsilon_min,
        memory_size      = RL_CONFIG.memory_size,
        batch_size       = RL_CONFIG.batch_size
    )

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
        # Keep the main thread alive
        while server.running:
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
                agent.stop(join=True, timeout=2.0)
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