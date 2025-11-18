#!/usr/bin/env python3
"""
Example script to create NumPy .npy files for testing the gr-ray-trace code.

This script demonstrates how to create sample data files that can be loaded
by the C++ application using the cnpy library.
"""

import numpy as np

def create_example_data():
    """Create example .npy files with different shapes and data types."""
    
    # Example 1: 1D array of doubles
    data_1d = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    np.save('example_1d.npy', data_1d)
    print(f"Created example_1d.npy with shape {data_1d.shape}")
    
    # Example 2: 2D array of doubles
    data_2d = np.array([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0],
                        [10.0, 11.0, 12.0]], dtype=np.float64)
    np.save('example_2d.npy', data_2d)
    print(f"Created example_2d.npy with shape {data_2d.shape}")
    
    # Example 3: 3D array representing a small grid
    nx, ny, nz = 10, 10, 10
    data_3d = np.random.rand(nx, ny, nz).astype(np.float64)
    np.save('example_3d.npy', data_3d)
    print(f"Created example_3d.npy with shape {data_3d.shape}")
    
    # Example 4: Integer array
    data_int = np.arange(100, dtype=np.int32)
    np.save('example_int.npy', data_int)
    print(f"Created example_int.npy with shape {data_int.shape}")
    
    print("\nAll example .npy files created successfully!")
    print("\nTo use these files with the C++ code, uncomment the relevant")
    print("sections in main.cpp and rebuild the application.")

if __name__ == "__main__":
    create_example_data()
