"""
Simple example of FlowUtils Logicle and Hyperlog transforms

This basic example demonstrates the core functionality without
external dependencies beyond numpy.
"""

import numpy as np
from flowutils import transforms

def main():
    print("FlowUtils - Pure Python Transforms")
    print("=" * 40)
    
    # Test data with negative, zero, and positive values
    test_data = np.array([-100, -10, -1, 0, 1, 10, 100, 1000])
    print(f"Original data: {test_data}")
    
    # Standard parameters
    t, m, w, a = 1000, 4.0, 1.0, 0
    
    print(f"\nTransform parameters: T={t}, M={m}, W={w}, A={a}")
    
    # Apply Logicle transform
    logicle_result = transforms._logicle(test_data, t=t, m=m, w=w, a=a)
    print(f"\nLogicle transform:")
    print(f"Input:  {test_data}")
    print(f"Output: {logicle_result}")
    
    # Verify round-trip
    logicle_inverse = transforms._logicle_inverse(logicle_result, t=t, m=m, w=w, a=a)
    logicle_error = np.max(np.abs(test_data - logicle_inverse))
    print(f"Round-trip error: {logicle_error:.2e}")
    
    # Apply Hyperlog transform  
    hyperlog_result = transforms._hyperlog(test_data, t=t, m=m, w=w, a=a)
    print(f"\nHyperlog transform:")
    print(f"Input:  {test_data}")
    print(f"Output: {hyperlog_result}")
    
    # Verify round-trip
    hyperlog_inverse = transforms._hyperlog_inverse(hyperlog_result, t=t, m=m, w=w, a=a)
    hyperlog_error = np.max(np.abs(test_data - hyperlog_inverse))
    print(f"Round-trip error: {hyperlog_error:.2e}")
    
    # Multi-channel example
    print(f"\nMulti-channel example:")
    multi_data = np.random.rand(5, 3) * 1000  # 5 events, 3 channels
    print(f"Original shape: {multi_data.shape}")
    
    # Transform only channel 0
    transformed = transforms.logicle(multi_data, [0], t=1000, m=4.0, w=1.0, a=0)
    print(f"Channel 0 transformed, channels 1-2 unchanged")
    print(f"Channel 1 unchanged: {np.array_equal(multi_data[:, 1], transformed[:, 1])}")
    print(f"Channel 2 unchanged: {np.array_equal(multi_data[:, 2], transformed[:, 2])}")
    
    print(f"\nExample completed successfully!")

if __name__ == "__main__":
    main()