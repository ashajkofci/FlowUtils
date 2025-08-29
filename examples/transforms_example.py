"""
Example demonstrating Logicle and Hyperlog transforms from FlowUtils

This example shows how to use the pure Python implementation of
logicle and hyperlog transforms for flow cytometry data.
"""

import numpy as np
import matplotlib.pyplot as plt
from flowutils import transforms

# Create example data with a range including negative, zero, and positive values
# This simulates typical flow cytometry data
data_raw = np.array([
    -1000, -500, -100, -50, -10, -5, -1, 
    0, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000
])

print("FlowUtils Pure Python Transforms Example")
print("=" * 50)
print(f"Original data range: {data_raw.min():.1f} to {data_raw.max():.1f}")

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

# Apply Logicle Transform
print("\n1. LOGICLE TRANSFORM")
print("-" * 30)
logicle_data = transforms._logicle(data_raw, t=T, m=M, w=W, a=A)
print(f"Logicle output range: {logicle_data.min():.6f} to {logicle_data.max():.6f}")

# Test round-trip accuracy
logicle_inverse_data = transforms._logicle_inverse(logicle_data, t=T, m=M, w=W, a=A)
logicle_error = np.max(np.abs(data_raw - logicle_inverse_data))
print(f"Round-trip error: {logicle_error:.2e} (should be very small)")

# Apply Hyperlog Transform
print("\n2. HYPERLOG TRANSFORM")
print("-" * 30)
hyperlog_data = transforms._hyperlog(data_raw, t=T, m=M, w=W, a=A)
print(f"Hyperlog output range: {hyperlog_data.min():.6f} to {hyperlog_data.max():.6f}")

# Test round-trip accuracy
hyperlog_inverse_data = transforms._hyperlog_inverse(hyperlog_data, t=T, m=M, w=W, a=A)
hyperlog_error = np.max(np.abs(data_raw - hyperlog_inverse_data))
print(f"Round-trip error: {hyperlog_error:.2e} (should be very small)")

# Multi-dimensional data example
print("\n3. MULTI-DIMENSIONAL DATA EXAMPLE")
print("-" * 40)

# Simulate 3-channel flow cytometry data (1000 events, 3 channels)
np.random.seed(42)  # For reproducible results
multi_data = np.column_stack([
    np.random.normal(1000, 500, 1000),    # Channel 0: Forward scatter
    np.random.normal(500, 200, 1000),     # Channel 1: Side scatter  
    np.random.exponential(100, 1000) - 50 # Channel 2: Fluorescence (can go negative)
])

print(f"Multi-channel data shape: {multi_data.shape}")
print(f"Channel ranges:")
for i in range(3):
    print(f"  Channel {i}: {multi_data[:, i].min():.1f} to {multi_data[:, i].max():.1f}")

# Apply logicle transform to fluorescence channels only (channel 2)
logicle_multi = transforms.logicle(multi_data, channel_indices=[2], t=T, m=M, w=W, a=A)
print(f"\nAfter logicle transform on channel 2:")
print(f"  Channel 2 range: {logicle_multi[:, 2].min():.6f} to {logicle_multi[:, 2].max():.6f}")
print(f"  Channels 0,1 unchanged: {np.array_equal(multi_data[:, [0,1]], logicle_multi[:, [0,1]])}")

# Apply hyperlog transform to fluorescence channels only (channel 2)  
hyperlog_multi = transforms.hyperlog(multi_data, channel_indices=[2], t=T, m=M, w=W, a=A)
print(f"\nAfter hyperlog transform on channel 2:")
print(f"  Channel 2 range: {hyperlog_multi[:, 2].min():.6f} to {hyperlog_multi[:, 2].max():.6f}")
print(f"  Channels 0,1 unchanged: {np.array_equal(multi_data[:, [0,1]], hyperlog_multi[:, [0,1]])}")

# Create visualization (optional - requires matplotlib)
try:
    import matplotlib.pyplot as plt
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Original vs transformed data comparison
    ax1.scatter(range(len(data_raw)), data_raw, alpha=0.7, label='Original')
    ax1.set_ylabel('Value')
    ax1.set_xlabel('Data Point Index')
    ax1.set_title('Original Data')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Logicle transformed
    ax2.scatter(range(len(logicle_data)), logicle_data, alpha=0.7, color='orange', label='Logicle')
    ax2.set_ylabel('Transformed Value')
    ax2.set_xlabel('Data Point Index')
    ax2.set_title('Logicle Transform')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Hyperlog transformed
    ax3.scatter(range(len(hyperlog_data)), hyperlog_data, alpha=0.7, color='green', label='Hyperlog')
    ax3.set_ylabel('Transformed Value') 
    ax3.set_xlabel('Data Point Index')
    ax3.set_title('Hyperlog Transform')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Transform comparison
    ax4.plot(data_raw, logicle_data, 'o-', label='Logicle', alpha=0.7)
    ax4.plot(data_raw, hyperlog_data, 's-', label='Hyperlog', alpha=0.7)
    ax4.set_xlabel('Original Value')
    ax4.set_ylabel('Transformed Value')
    ax4.set_title('Transform Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('flowutils_transforms_example.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n4. VISUALIZATION")
    print("-" * 20)
    print("Plots saved as 'flowutils_transforms_example.png'")
    
except ImportError:
    print(f"\n4. VISUALIZATION")
    print("-" * 20)
    print("Matplotlib not available - skipping plots")

print(f"\n5. PERFORMANCE NOTE")
print("-" * 25)
print("This pure Python implementation prioritizes:")
print("- Compatibility with numpy 1.22+")
print("- No C dependencies") 
print("- Mathematical accuracy")
print("- Code clarity and maintainability")
print("\nFor high-performance applications with large datasets,")
print("consider vectorization or compilation with numba.")

print(f"\nExample completed successfully!")