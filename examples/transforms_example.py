"""
Example demonstrating Logicle and Hyperlog transforms from FlowUtils

This example shows how to use the pure Python implementation of
logicle and hyperlog transforms for flow cytometry data, including
integration with FlowCytometryTools for reading FCS files.

Usage:
    python transforms_example.py [path/to/fcs/file]

If no FCS file is provided, uses simulated flow cytometry data.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from flowutils import transforms


def load_fcs_with_flowcytometrytools(fcs_path):
    """
    Load FCS data using FlowCytometryTools
    """
    try:
        from FlowCytometryTools import FCMeasurement
        
        print(f"Loading FCS file: {fcs_path}")
        sample = FCMeasurement(ID='Sample', datafile=fcs_path)
        data = sample.data.values
        channels = list(sample.data.columns)
        
        print(f"✓ Loaded {len(data):,} events with {len(channels)} channels")
        return data, channels
        
    except ImportError:
        print("FlowCytometryTools not available. Install with:")
        print("  pip install FlowCytometryTools")
        return None, None
        
    except Exception as e:
        print(f"Error loading FCS file: {e}")
        return None, None


def main():
    # Check if FCS file provided as argument
    if len(sys.argv) > 1:
        fcs_path = sys.argv[1]
        data, channels = load_fcs_with_flowcytometrytools(fcs_path)
        if data is not None:
            # Use first two channels for demonstration
            data_raw = data[:1000, :2]  # Limit to 1000 events for visualization
            print(f"Using channels: {channels[0]}, {channels[1]}")
        else:
            data_raw = None
    else:
        data_raw = None
    
    # Create example data if no FCS file loaded
    if data_raw is None:
        print("Using simulated flow cytometry data")
        # Create example data with a range including negative, zero, and positive values
        # This simulates typical flow cytometry data
        data_raw = np.array([
            -1000, -500, -100, -50, -10, -5, -1, 
            0, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000
        ])
        
        # Reshape for consistency with FCS data format
        data_raw = data_raw.reshape(-1, 1)

    print("FlowUtils Pure Python Transforms Example")
    print("=" * 50)
    print(f"Data shape: {data_raw.shape}")
    print(f"Data range: {data_raw.min():.1f} to {data_raw.max():.1f}")

    # Transform parameters (typical values for flow cytometry)
    T = 262144  # Top of scale
    M = 4.5     # Number of decades
    W = 0.5     # Width of linear region (decades)
    A = 0       # Additional negative decades

    print(f"\nTransform parameters:")
    print(f"T (top of scale): {T}")
    print(f"M (decades): {M}")
    print(f"W (linear width): {W}")
    print(f"A (negative decades): {A}")

    # Apply transforms
    print("\n" + "="*30 + " TRANSFORMS " + "="*30)

    # Use first channel for single-channel data
    if data_raw.shape[1] == 1:
        channel_data = data_raw[:, 0]
    else:
        channel_data = data_raw[:, 0]  # Use first channel
        
    # Logicle Transform
    print("\n1. Logicle Transform:")
    logicle_result = transforms.logicle(channel_data, channel_indices=None, t=T, m=M, w=W, a=A)
    print(f"   Original range: [{channel_data.min():.1f}, {channel_data.max():.1f}]")
    print(f"   Logicle range:  [{logicle_result.min():.6f}, {logicle_result.max():.6f}]")

    # Hyperlog Transform  
    print("\n2. Hyperlog Transform:")
    hyperlog_result = transforms.hyperlog(channel_data, channel_indices=None, t=T, m=M, w=W, a=A)
    print(f"   Original range: [{channel_data.min():.1f}, {channel_data.max():.1f}]")
    print(f"   Hyperlog range: [{hyperlog_result.min():.6f}, {hyperlog_result.max():.6f}]")

    # Test inverse transforms
    print("\n" + "="*30 + " INVERSE TRANSFORMS " + "="*30)

    # Logicle inverse
    logicle_inverse = transforms.logicle_inverse(logicle_result, channel_indices=None, t=T, m=M, w=W, a=A)
    logicle_error = np.max(np.abs(channel_data - logicle_inverse))
    print(f"\n3. Logicle Inverse (Round-trip test):")
    print(f"   Max error: {logicle_error:.2e}")
    print(f"   Accuracy: {'PASS' if logicle_error < 1e-6 else 'FAIL'}")

    # Hyperlog inverse
    hyperlog_inverse = transforms.hyperlog_inverse(hyperlog_result, channel_indices=None, t=T, m=M, w=W, a=A)  
    hyperlog_error = np.max(np.abs(channel_data - hyperlog_inverse))
    print(f"\n4. Hyperlog Inverse (Round-trip test):")
    print(f"   Max error: {hyperlog_error:.2e}")
    print(f"   Accuracy: {'PASS' if hyperlog_error < 1e-6 else 'FAIL'}")

    print("\n" + "="*77)
    print("✓ Transform examples completed successfully!")
    print("These transforms are optimized for flow cytometry data visualization.")


if __name__ == "__main__":
    main()
