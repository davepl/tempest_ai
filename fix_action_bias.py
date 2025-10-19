#!/usr/bin/env python3
"""Fix action bias by reinitializing the discrete Q-head with balanced initialization"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Scripts'))

import torch
import torch.nn as nn
from aimodel import HybridDQNAgent
from config import RL_CONFIG

def fix_action_bias():
    """Reinitialize Q-head to remove action bias"""
    print("=" * 80)
    print("FIXING ACTION BIAS")
    print("=" * 80)
    
    # Load model
    model_path = 'models/tempest_model_latest.pt'
    backup_path = 'models/tempest_model_latest.pt.before_bias_fix'
    
    try:
        # Backup original model
        print(f"\n1. Creating backup: {backup_path}")
        import shutil
        shutil.copy2(model_path, backup_path)
        print("   ✓ Backup created")
        
        # Load checkpoint
        print(f"\n2. Loading model: {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"   ✓ Model loaded (frame {checkpoint.get('frame_count', 0)})")
        
        # Create agent to get network structure
        agent = HybridDQNAgent(
            state_size=RL_CONFIG.state_size,
            discrete_actions=4,
            learning_rate=RL_CONFIG.lr,
        )
        
        # Load state dicts
        agent.qnetwork_local.load_state_dict(checkpoint['local_state_dict'])
        agent.qnetwork_target.load_state_dict(checkpoint['target_state_dict'])
        
        print("\n3. Analyzing current Q-head bias...")
        # Get the discrete Q-branch layers
        discrete_fc = agent.qnetwork_local.discrete_fc
        discrete_out = agent.qnetwork_local.discrete_out
        
        # Show current weights
        with torch.no_grad():
            fc_weight_mean = discrete_fc.weight.mean().item()
            fc_weight_std = discrete_fc.weight.std().item()
            fc_bias_mean = discrete_fc.bias.mean().item()
            fc_bias_std = discrete_fc.bias.std().item()
            
            out_weight_mean = discrete_out.weight.mean().item()
            out_weight_std = discrete_out.weight.std().item()
            out_bias_mean = discrete_out.bias.mean().item()
            out_bias_std = discrete_out.bias.std().item()
            
        print(f"   discrete_fc weights: mean={fc_weight_mean:8.4f}, std={fc_weight_std:8.4f}")
        print(f"   discrete_fc bias:    mean={fc_bias_mean:8.4f}, std={fc_bias_std:8.4f}")
        print(f"   discrete_out weights: mean={out_weight_mean:8.4f}, std={out_weight_std:8.4f}")
        print(f"   discrete_out bias:    mean={out_bias_mean:8.4f}, std={out_bias_std:8.4f}")
        
        print("\n4. Reinitializing ENTIRE discrete Q-branch with balanced initialization...")
        # Reinitialize both discrete layers with smaller weights to reduce bias
        with torch.no_grad():
            # Xavier uniform initialization (good for preventing bias)
            nn.init.xavier_uniform_(discrete_fc.weight, gain=0.5)
            nn.init.constant_(discrete_fc.bias, 0.0)  # Zero bias
            
            nn.init.xavier_uniform_(discrete_out.weight, gain=0.01)  # Very small gain for output
            nn.init.constant_(discrete_out.bias, 0.0)  # Zero bias to start neutral
            
            fc_weight_mean_new = discrete_fc.weight.mean().item()
            fc_weight_std_new = discrete_fc.weight.std().item()
            out_weight_mean_new = discrete_out.weight.mean().item()
            out_weight_std_new = discrete_out.weight.std().item()
            
        print(f"   discrete_fc weights: mean={fc_weight_mean_new:8.4f}, std={fc_weight_std_new:8.4f}")
        print(f"   discrete_out weights: mean={out_weight_mean_new:8.4f}, std={out_weight_std_new:8.4f}")
        
        # Update both local and target networks
        agent.qnetwork_target.load_state_dict(agent.qnetwork_local.state_dict())
        
        print("\n5. Saving fixed model...")
        # Update checkpoint with fixed networks
        checkpoint['local_state_dict'] = agent.qnetwork_local.state_dict()
        checkpoint['target_state_dict'] = agent.qnetwork_target.state_dict()
        
        # Save fixed model
        torch.save(checkpoint, model_path)
        print(f"   ✓ Fixed model saved to {model_path}")
        
        # Verify fix
        print("\n6. Verifying fix...")
        import numpy as np
        
        # Test on random states
        num_samples = 1000
        random_states = np.random.randn(num_samples, RL_CONFIG.state_size).astype(np.float32)
        
        agent.qnetwork_local.eval()
        with torch.no_grad():
            states_tensor = torch.FloatTensor(random_states).to(agent.device)
            discrete_q, _ = agent.qnetwork_local(states_tensor)
            discrete_actions = discrete_q.argmax(dim=1).cpu()
        
        action_counts = np.bincount(discrete_actions.numpy(), minlength=4)
        action_percentages = 100.0 * action_counts / num_samples
        
        print(f"\n   New Action Distribution (n={num_samples} random states):")
        for i in range(4):
            fire = "F" if (i & 0b10) else "-"
            zap = "Z" if (i & 0b01) else "-"
            print(f"     Action {i} ({fire}{zap}): {action_counts[i]:4d} ({action_percentages[i]:5.1f}%)")
        
        max_pct = action_percentages.max()
        min_pct = action_percentages.min()
        bias_ratio = max_pct / min_pct if min_pct > 0 else float('inf')
        
        print(f"\n   Bias ratio: {bias_ratio:.2f}x (was 32.69x)")
        
        if bias_ratio < 3.0:
            print("   ✓ Action bias significantly reduced!")
        else:
            print(f"   ⚠️  Bias still high ({bias_ratio:.1f}x), may need further adjustment")
        
        print("\n" + "=" * 80)
        print("FIX COMPLETE")
        print("=" * 80)
        print(f"\nOriginal model backed up to: {backup_path}")
        print("You can now restart training with reduced action bias.")
        print("\nNOTE: Agreement may still start low while the buffer fills with new data,")
        print("but it should improve quickly as the model learns from balanced actions.")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = fix_action_bias()
    sys.exit(0 if success else 1)
