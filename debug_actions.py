#!/usr/bin/env python3
"""
Debug script to analyze DQN action selection after 1M frames.
Tests if Q-values are reasonable or if there's a bug in action selection.
"""

import sys
import torch
import numpy as np

sys.path.insert(0, '/home/dave/source/repos/tempest_ai/Scripts')

from config import SPINNER_MAPPING, RL_CONFIG
from aimodel import HybridDQNAgent

# Load the trained model
print("Loading model from models/tempest_model_latest.pt...")
agent = HybridDQNAgent(
    state_size=RL_CONFIG.state_size,
    discrete_actions=RL_CONFIG.discrete_action_size,
    spinner_actions=RL_CONFIG.spinner_action_size
)

try:
    checkpoint = torch.load('models/tempest_model_latest.pt', map_location='cpu')
    
    # Try different checkpoint formats
    if 'local_state_dict' in checkpoint:
        agent.qnetwork_local.load_state_dict(checkpoint['local_state_dict'])
        agent.qnetwork_inference.load_state_dict(checkpoint['local_state_dict'])
    elif 'qnetwork_local' in checkpoint:
        agent.qnetwork_local.load_state_dict(checkpoint['qnetwork_local'])
        agent.qnetwork_inference.load_state_dict(checkpoint['qnetwork_inference'])
    elif 'model_state_dict' in checkpoint:
        agent.qnetwork_local.load_state_dict(checkpoint['model_state_dict'])
        agent.qnetwork_inference.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Maybe the checkpoint IS the state dict
        agent.qnetwork_local.load_state_dict(checkpoint)
        agent.qnetwork_inference.load_state_dict(checkpoint)
    
    print(f"✓ Model loaded successfully")
    if isinstance(checkpoint, dict):
        print(f"  Frame count: {checkpoint.get('frame_count', 'unknown')}")
        print(f"  Epsilon: {checkpoint.get('epsilon', 'unknown')}")
        print(f"  Expert ratio: {checkpoint.get('expert_ratio', 'unknown')}")
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Create test states
print("\nTesting action selection on random states...")
print("=" * 80)

agent.qnetwork_inference.eval()

for test_idx in range(10):
    # Generate random state
    state = np.random.randn(RL_CONFIG.state_size).astype(np.float32)
    
    # Get Q-values
    state_tensor = torch.from_numpy(state).float().unsqueeze(0).to('cpu')
    agent.qnetwork_inference.to('cpu')  # Move model to CPU for debugging
    with torch.no_grad():
        firezap_q, spinner_q = agent.qnetwork_inference(state_tensor)
    
    # Get actions (greedy, no epsilon)
    firezap_action = firezap_q.argmax(dim=1).item()
    spinner_action = spinner_q.argmax(dim=1).item()
    spinner_value = SPINNER_MAPPING[spinner_action]
    
    # Check for issues
    firezap_q_np = firezap_q.cpu().numpy()[0]
    spinner_q_np = spinner_q.cpu().numpy()[0]
    
    firezap_has_nan = np.isnan(firezap_q_np).any()
    spinner_has_nan = np.isnan(spinner_q_np).any()
    firezap_has_inf = np.isinf(firezap_q_np).any()
    spinner_has_inf = np.isinf(spinner_q_np).any()
    
    status = "✓"
    issues = []
    if firezap_has_nan or spinner_has_nan:
        status = "✗"
        issues.append("NaN")
    if firezap_has_inf or spinner_has_inf:
        status = "✗"
        issues.append("Inf")
    if abs(spinner_q_np).max() > 100:
        status = "⚠"
        issues.append(f"Large Q-values (max={abs(spinner_q_np).max():.1f})")
    
    issues_str = ", ".join(issues) if issues else "OK"
    
    print(f"{status} Test {test_idx+1}:")
    print(f"   Fire/Zap: action={firezap_action}, Q-values={firezap_q_np}")
    print(f"   Spinner:  action={spinner_action} (value={spinner_value:6.3f}), Q-values={spinner_q_np}")
    print(f"   Status: {issues_str}")
    print()

# Test specific edge cases
print("\n" + "=" * 80)
print("Testing edge cases...")
print("=" * 80)

# Test with all zeros
state_zeros = np.zeros(RL_CONFIG.state_size, dtype=np.float32)
state_tensor = torch.from_numpy(state_zeros).float().unsqueeze(0)
with torch.no_grad():
    firezap_q, spinner_q = agent.qnetwork_inference(state_tensor)
    
firezap_action = firezap_q.argmax(dim=1).item()
spinner_action = spinner_q.argmax(dim=1).item()
spinner_value = SPINNER_MAPPING[spinner_action]

print(f"\nAll-zeros state:")
print(f"  Fire/Zap: action={firezap_action}, Q-range=[{firezap_q.min():.3f}, {firezap_q.max():.3f}]")
print(f"  Spinner:  action={spinner_action} (value={spinner_value:6.3f}), Q-range=[{spinner_q.min():.3f}, {spinner_q.max():.3f}]")

# Test with all ones
state_ones = np.ones(RL_CONFIG.state_size, dtype=np.float32)
state_tensor = torch.from_numpy(state_ones).float().unsqueeze(0)
with torch.no_grad():
    firezap_q, spinner_q = agent.qnetwork_inference(state_tensor)
    
firezap_action = firezap_q.argmax(dim=1).item()
spinner_action = spinner_q.argmax(dim=1).item()
spinner_value = SPINNER_MAPPING[spinner_action]

print(f"\nAll-ones state:")
print(f"  Fire/Zap: action={firezap_action}, Q-range=[{firezap_q.min():.3f}, {firezap_q.max():.3f}]")
print(f"  Spinner:  action={spinner_action} (value={spinner_value:6.3f}), Q-range=[{spinner_q.min():.3f}, {spinner_q.max():.3f}]")

print("\n" + "=" * 80)
print("Analysis complete!")
print("=" * 80)
