#!/usr/bin/env python3
"""Quick verification that the updated architecture builds and runs."""
import sys
sys.path.insert(0, "Robotron/Scripts")
import torch
from config import RL_CONFIG
from aimodel import RainbowNet

model = RainbowNet(RL_CONFIG.state_size)
print("Model created successfully")
print(f"use_directional_lanes: {model.use_directional_lanes}")
print(f"compact_dense_size: {model.compact_dense_size}")
print(f"core_feature_count: {model.core_feature_count}")
print(f"extra_context_features: {model.extra_context_features}")
print(f"stack_depth: {model.stack_depth}")

trunk_first = model.trunk[0]
print(f"trunk[0] input_features: {trunk_first.in_features}")
print(f"trunk[0] out_features: {trunk_first.out_features}")

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {total:,}")
print(f"Trainable params: {trainable:,}")

B = 4
state = torch.randn(B, RL_CONFIG.state_size)
with torch.no_grad():
    log_p = model(state, log=True)
    print(f"Output shape: {log_p.shape}")
    probs = model(state, log=False)
    q = (probs * model.support.unsqueeze(0).unsqueeze(0)).sum(dim=2)
    print(f"Q-values shape: {q.shape}")
    print(f"Q-values range: [{q.min().item():.1f}, {q.max().item():.1f}]")
print("Forward pass OK")
