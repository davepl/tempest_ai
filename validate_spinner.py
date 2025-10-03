#!/usr/bin/env python3
"""
Validation: Ensure SPINNER_MAPPING values work correctly with game encoding.
"""

# Test the complete pipeline
SPINNER_MAPPING = {
    0: -29/32,  # -0.90625
    1: -19/32,  # -0.59375
    2: -10/32,  # -0.3125
    3: -3/32,   # -0.09375
    4: 0.0,
    5: 3/32,    # 0.09375
    6: 10/32,   # 0.3125
    7: 19/32,   # 0.59375
    8: 29/32,   # 0.90625
}

def encode_to_game(spinner_value):
    """Game encoding: int(round(value * 32))"""
    return int(round(spinner_value * 32))

print("Validation: DQN action → game → player movement")
print("=" * 70)

for action in range(9):
    # Step 1: DQN selects discrete action
    spinner_value = SPINNER_MAPPING[action]
    
    # Step 2: Send to game
    game_int = encode_to_game(spinner_value)
    
    # Step 3: Game applies movement (game_int is the movement command)
    # The game uses this value directly for spinner movement
    
    print(f"Action {action}: DQN outputs {spinner_value:8.5f} → "
          f"game receives {game_int:3d} → "
          f"player moves by {game_int}/32 = {game_int/32:8.5f}")
    
    # Verify perfect round-trip
    assert abs(spinner_value - game_int/32) < 1e-10, f"Round-trip error for action {action}!"

print("=" * 70)
print("✓ ALL ACTIONS VALIDATED - Perfect encoding/decoding!")
