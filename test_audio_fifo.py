#!/usr/bin/env python3
"""Test script to verify audio FIFO can be read."""

import os
import sys
import struct
import time

FIFO_PATH = "/tmp/robotron_audio_client0.wav"

def test_fifo():
    print(f"Testing FIFO: {FIFO_PATH}")
    
    if not os.path.exists(FIFO_PATH):
        print(f"ERROR: FIFO does not exist")
        return False
    
    if not os.stat(FIFO_PATH).st_mode & 0o010000:  # S_ISFIFO
        print(f"ERROR: {FIFO_PATH} is not a FIFO")
        return False
    
    print(f"Opening FIFO for reading...")
    start = time.time()
    
    try:
        with open(FIFO_PATH, 'rb', buffering=0) as f:
            elapsed = time.time() - start
            print(f"✓ FIFO opened in {elapsed:.2f}s")
            
            print("Reading WAV header (44 bytes)...")
            header = f.read(44)
            print(f"✓ Got {len(header)} bytes")
            
            if len(header) < 44:
                print(f"ERROR: Incomplete header")
                return False
            
            # Parse header
            if header[0:4] != b'RIFF':
                print(f"ERROR: Not a RIFF file: {header[0:4]}")
                return False
            
            if header[8:12] != b'WAVE':
                print(f"ERROR: Not a WAVE file: {header[8:12]}")
                return False
            
            channels = struct.unpack('<H', header[22:24])[0]
            sample_rate = struct.unpack('<I', header[24:28])[0]
            bits_per_sample = struct.unpack('<H', header[34:36])[0]
            
            print(f"✓ Format: {channels} channels, {sample_rate} Hz, {bits_per_sample} bits")
            
            print("Reading first 10 PCM chunks...")
            for i in range(10):
                chunk = f.read(4096)
                if not chunk:
                    print(f"ERROR: EOF at chunk {i}")
                    return False
                print(f"  Chunk {i+1}: {len(chunk)} bytes")
            
            print(f"✓ Audio FIFO is working correctly!")
            return True
            
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_fifo()
    sys.exit(0 if success else 1)
