#!/usr/bin/env python3
"""Test script to verify fire/zap encoding/decoding consistency."""

import sys
import os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Scripts'))

from aimodel import fire_zap_to_discrete, discrete_to_fire_zap

def test_encoding_consistency():
    """Test that encoding and decoding are consistent."""
    print("Testing fire/zap encoding/decoding consistency...")

    test_cases = [
        (False, False),  # 0: no fire, no zap
        (True, False),   # 1: fire, no zap
        (False, True),   # 2: no fire, zap
        (True, True),    # 3: fire and zap
    ]

    all_passed = True

    for fire, zap in test_cases:
        # Encode
        encoded = fire_zap_to_discrete(fire, zap)
        # Decode
        decoded_fire, decoded_zap = discrete_to_fire_zap(encoded)

        print(f"Original: fire={fire}, zap={zap} -> Encoded: {encoded} -> Decoded: fire={decoded_fire}, zap={decoded_zap}")

        if (fire, zap) != (decoded_fire, decoded_zap):
            print(f"ERROR: Mismatch! Expected {(fire, zap)}, got {(decoded_fire, decoded_zap)}")
            all_passed = False

    if all_passed:
        print("✓ All encoding/decoding tests passed!")
    else:
        print("✗ Some tests failed!")

    return all_passed

if __name__ == "__main__":
    test_encoding_consistency()