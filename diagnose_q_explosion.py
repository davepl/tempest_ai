"""Diagnose Q-value explosion in discrete head."""
import torch
import sys
sys.path.insert(0, './Scripts')

from config import RL_CONFIG
from aimodel import Agent

def main():
    print("=== Q-VALUE EXPLOSION DIAGNOSIS ===\n")
    
    # Load model
    agent = Agent(
        state_size=RL_CONFIG.state_size,
        action_size=RL_CONFIG.action_size,
        discrete_action_size=RL_CONFIG.discrete_action_size,
        config=RL_CONFIG
    )
    
    try:
        checkpoint = torch.load('models/tempest_model_latest.pt', map_location=agent.device)
        agent.qnetwork_local.load_state_dict(checkpoint['model_state_dict'])
        agent.qnetwork_target.load_state_dict(checkpoint['target_model_state_dict'])
        print(f"✓ Loaded model from frame {checkpoint.get('frame_count', 'unknown')}\n")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return
    
    # Create dummy batch
    batch_size = 32
    state_size = RL_CONFIG.state_size
    dummy_states = torch.randn(batch_size, state_size).to(agent.device)
    
    print("=== LOCAL NETWORK Q-VALUES ===")
    with torch.no_grad():
        discrete_q, continuous = agent.qnetwork_local(dummy_states)
        
        print(f"Discrete Q-values shape: {discrete_q.shape}")
        print(f"Min Q-value: {discrete_q.min().item():.4f}")
        print(f"Max Q-value: {discrete_q.max().item():.4f}")
        print(f"Mean Q-value: {discrete_q.mean().item():.4f}")
        print(f"Std Q-value: {discrete_q.std().item():.4f}")
        
        print("\nQ-values by action (mean ± std):")
        for i in range(discrete_q.shape[1]):
            action_q = discrete_q[:, i]
            print(f"  Action {i}: {action_q.mean().item():8.4f} ± {action_q.std().item():.4f}")
        
        print("\nQ-value distribution:")
        for i in range(discrete_q.shape[1]):
            action_q = discrete_q[:, i]
            print(f"  Action {i}: min={action_q.min().item():8.4f}, "
                  f"max={action_q.max().item():8.4f}, "
                  f"median={action_q.median().item():8.4f}")
    
    print("\n=== TARGET NETWORK Q-VALUES ===")
    with torch.no_grad():
        target_q, _ = agent.qnetwork_target(dummy_states)
        
        print(f"Min Q-value: {target_q.min().item():.4f}")
        print(f"Max Q-value: {target_q.max().item():.4f}")
        print(f"Mean Q-value: {target_q.mean().item():.4f}")
        print(f"Std Q-value: {target_q.std().item():.4f}")
        
        print("\nQ-values by action (mean ± std):")
        for i in range(target_q.shape[1]):
            action_q = target_q[:, i]
            print(f"  Action {i}: {action_q.mean().item():8.4f} ± {action_q.std().item():.4f}")
    
    print("\n=== NETWORK WEIGHT STATISTICS ===")
    print("\nDiscrete head weights:")
    for name, param in agent.qnetwork_local.named_parameters():
        if 'discrete' in name:
            print(f"  {name:30s}: mean={param.data.mean().item():8.6f}, "
                  f"std={param.data.std().item():.6f}, "
                  f"min={param.data.min().item():8.4f}, "
                  f"max={param.data.max().item():8.4f}")
    
    print("\n=== GRADIENT STATISTICS (if available) ===")
    # Forward pass with gradients
    agent.qnetwork_local.train()
    discrete_q, continuous = agent.qnetwork_local(dummy_states)
    
    # Simulate loss
    fake_targets = torch.zeros_like(discrete_q[:, 0:1])
    discrete_q_selected = discrete_q.gather(1, torch.zeros(batch_size, 1, dtype=torch.long).to(agent.device))
    loss = torch.nn.functional.huber_loss(discrete_q_selected, fake_targets)
    loss.backward()
    
    print("\nDiscrete head gradients:")
    for name, param in agent.qnetwork_local.named_parameters():
        if 'discrete' in name and param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_mean = param.grad.mean().item()
            grad_std = param.grad.std().item()
            print(f"  {name:30s}: norm={grad_norm:8.6f}, "
                  f"mean={grad_mean:8.6f}, "
                  f"std={grad_std:.6f}")
    
    print("\n=== DIAGNOSIS ===")
    with torch.no_grad():
        discrete_q, _ = agent.qnetwork_local(dummy_states)
        max_q = discrete_q.max().item()
        min_q = discrete_q.min().item()
        mean_q = discrete_q.mean().item()
        std_q = discrete_q.std().item()
        
        print(f"\n1. Q-value range: [{min_q:.2f}, {max_q:.2f}]")
        if abs(max_q) > 1000 or abs(min_q) > 1000:
            print("   ⚠️  WARNING: Q-values are exploding (>1000)!")
        elif abs(mean_q) > 100:
            print("   ⚠️  WARNING: Q-values are very large (mean >100)!")
        else:
            print("   ✓ Q-values are in reasonable range")
        
        print(f"\n2. Q-value spread: std={std_q:.2f}")
        if std_q > 500:
            print("   ⚠️  WARNING: Very high variance in Q-values!")
        else:
            print("   ✓ Q-value variance is reasonable")
        
        print(f"\n3. TD target clip: {RL_CONFIG.td_target_clip}")
        if RL_CONFIG.td_target_clip is None:
            print("   ⚠️  WARNING: No TD target clipping enabled!")
        elif abs(max_q) > RL_CONFIG.td_target_clip:
            print(f"   ⚠️  WARNING: Q-values exceed clip threshold!")
        else:
            print("   ✓ Q-values within clip threshold")

if __name__ == '__main__':
    main()
