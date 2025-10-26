#!/usr/bin/env python3
"""
Simple test to measure target network update performance.
"""

import time
import torch
import torch.nn as nn

def create_test_network():
    """Create a simple network similar to the DQN architecture"""
    layers = []
    layers.append(nn.Linear(171, 512))
    layers.append(nn.ReLU())
    layers.append(nn.Linear(512, 512))
    layers.append(nn.ReLU())
    layers.append(nn.Linear(512, 256))
    layers.append(nn.ReLU())
    layers.append(nn.Linear(256, 256))
    layers.append(nn.ReLU())
    layers.append(nn.Linear(256, 128))
    layers.append(nn.ReLU())
    layers.append(nn.Linear(128, 64))
    layers.append(nn.ReLU())
    layers.append(nn.Linear(64, 4))  # discrete actions
    
    return nn.ModuleList(layers)

def test_soft_update_performance():
    print("Testing soft target update performance...")
    
    # Create local and target networks
    local_net = create_test_network()
    target_net = create_test_network()
    
    # Copy initial weights
    for tgt_p, src_p in zip(target_net.parameters(), local_net.parameters()):
        tgt_p.data.copy_(src_p.data)
    
    # Test soft updates
    tau = 0.005
    num_updates = 1000
    
    start_time = time.time()
    for _ in range(num_updates):
        with torch.no_grad():
            for tgt_p, src_p in zip(target_net.parameters(), local_net.parameters()):
                tgt_p.data.mul_(1.0 - tau).add_(src_p.data, alpha=tau)
    soft_time = time.time() - start_time
    
    print(".2f")
    print(".4f")
    
    # Test hard updates
    start_time = time.time()
    for _ in range(num_updates):
        for tgt_p, src_p in zip(target_net.parameters(), local_net.parameters()):
            tgt_p.data.copy_(src_p.data)
    hard_time = time.time() - start_time
    
    print(".2f")
    print(".4f")
    
    speedup = soft_time / hard_time if hard_time > 0 else float('inf')
    print(".1f")
    
    return soft_time, hard_time, speedup

if __name__ == "__main__":
    test_soft_update_performance()
