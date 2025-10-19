"""Check what discrete actions are actually stored in replay buffer."""
import sys
sys.path.insert(0, './Scripts')

import torch
import numpy as np
from config import RL_CONFIG
from aimodel import Agent

def main():
    print("=== REPLAY BUFFER ACTION ANALYSIS ===\n")
    
    # Create agent
    agent = Agent(
        state_size=RL_CONFIG.state_size,
        action_size=RL_CONFIG.action_size,
        discrete_action_size=RL_CONFIG.discrete_action_size,
        config=RL_CONFIG
    )
    
    # Load model if exists
    try:
        checkpoint = torch.load('models/tempest_model_latest.pt', map_location=agent.device)
        agent.qnetwork_local.load_state_dict(checkpoint['model_state_dict'])
        agent.qnetwork_target.load_state_dict(checkpoint['target_model_state_dict'])
        
        # Try to load memory
        if 'memory' in checkpoint:
            import pickle
            agent.memory.load_from_checkpoint(checkpoint['memory'])
            print(f"✓ Loaded model and memory from frame {checkpoint.get('frame_count', 'unknown')}")
        else:
            print(f"✓ Loaded model from frame {checkpoint.get('frame_count', 'unknown')} (no memory)")
    except Exception as e:
        print(f"⚠ Could not load model: {e}")
        print("Creating fresh agent to analyze structure\n")
    
    print(f"\nReplay buffer size: {len(agent.memory)}")
    
    if len(agent.memory) < 100:
        print("⚠ Buffer too small for analysis")
        return
    
    # Sample a batch
    batch_size = min(2048, len(agent.memory))
    print(f"Sampling {batch_size} transitions...")
    
    batch = agent.memory.sample(batch_size)
    if batch is None:
        print("✗ Failed to sample batch")
        return
    
    states, discrete_actions, continuous_actions, rewards, next_states, dones, actors, horizons = batch
    
    # Convert discrete actions to numpy for analysis
    discrete_np = discrete_actions.cpu().numpy().flatten()
    
    print("\n=== DISCRETE ACTION DISTRIBUTION ===")
    print(f"Shape: {discrete_actions.shape}")
    print(f"Unique actions: {np.unique(discrete_np)}")
    
    print("\nAction frequency:")
    for action_id in range(4):
        count = (discrete_np == action_id).sum()
        pct = 100.0 * count / len(discrete_np)
        print(f"  Action {action_id}: {count:6d} ({pct:5.2f}%)")
    
    # Analyze by actor type
    print("\n=== ACTION DISTRIBUTION BY ACTOR ===")
    actors_np = np.array(actors)
    
    dqn_mask = actors_np == 'dqn'
    expert_mask = actors_np == 'expert'
    
    n_dqn = dqn_mask.sum()
    n_expert = expert_mask.sum()
    
    print(f"\nDQN actions: {n_dqn} ({100.0 * n_dqn / len(actors_np):.1f}%)")
    if n_dqn > 0:
        dqn_actions = discrete_np[dqn_mask]
        for action_id in range(4):
            count = (dqn_actions == action_id).sum()
            pct = 100.0 * count / len(dqn_actions)
            print(f"  Action {action_id}: {count:6d} ({pct:5.2f}%)")
    
    print(f"\nExpert actions: {n_expert} ({100.0 * n_expert / len(actors_np):.1f}%)")
    if n_expert > 0:
        expert_actions = discrete_np[expert_mask]
        for action_id in range(4):
            count = (expert_actions == action_id).sum()
            pct = 100.0 * count / len(expert_actions)
            print(f"  Action {action_id}: {count:6d} ({pct:5.2f}%)")
    
    # Check rewards
    print("\n=== REWARD DISTRIBUTION ===")
    rewards_np = rewards.cpu().numpy().flatten()
    print(f"Min reward: {rewards_np.min():.2f}")
    print(f"Max reward: {rewards_np.max():.2f}")
    print(f"Mean reward: {rewards_np.mean():.4f}")
    print(f"Median reward: {np.median(rewards_np):.2f}")
    
    print("\nReward by action:")
    for action_id in range(4):
        action_mask = discrete_np == action_id
        if action_mask.sum() > 0:
            action_rewards = rewards_np[action_mask]
            print(f"  Action {action_id}: mean={action_rewards.mean():8.4f}, "
                  f"median={np.median(action_rewards):8.4f}, "
                  f"std={action_rewards.std():.4f}")
    
    print("\n=== DIAGNOSIS ===")
    
    # Check if action 2 (FIRE) is present
    fire_count = (discrete_np == 2).sum()
    fire_pct = 100.0 * fire_count / len(discrete_np)
    
    print(f"\n1. Action 2 (FIRE) frequency: {fire_count} ({fire_pct:.2f}%)")
    if fire_pct < 50:
        print("   ⚠️  WARNING: FIRE action is underrepresented!")
        print("   Expected ~95% from experts, but seeing much less.")
    else:
        print("   ✓ FIRE action is well represented")
    
    # Check expert ratio
    expert_pct = 100.0 * n_expert / len(actors_np)
    print(f"\n2. Expert data in buffer: {expert_pct:.1f}%")
    if expert_pct < 20:
        print("   ⚠️  WARNING: Very low expert data!")
    elif expert_pct > 80:
        print("   ⚠️  WARNING: Very high expert data - DQN may not be exploring!")
    else:
        print("   ✓ Expert ratio seems reasonable")
    
    # Check if expert actions match expected pattern
    if n_expert > 0:
        expert_fire = (discrete_np[expert_mask] == 2).sum()
        expert_fire_pct = 100.0 * expert_fire / n_expert
        print(f"\n3. Expert FIRE action: {expert_fire_pct:.1f}%")
        if expert_fire_pct < 80:
            print("   ⚠️  WARNING: Experts not using FIRE as expected!")
        else:
            print("   ✓ Experts using FIRE as expected")

if __name__ == '__main__':
    main()
