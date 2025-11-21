#!/usr/bin/env python3
"""
Example script to create NumPy .npy files for testing the gr-ray-trace code.

This script demonstrates how to create sample data files that can be loaded
by the C++ application using the cnpy library.
"""

import numpy as np
import matplotlib.pyplot as plt

def give_hamr_array_shapes():
    base_path = r'/home/siddhant/scratch/rayTracingTestData/'
    arrays_names = [r'bsq.npz', 
                    r'bu.npz', 
                    r'pgas.npz',
                    r'phi.npz',
                    r'rho.npz',
                    r'r.npz',
                    r'theta.npz',
                    r'Tgas.npz',
                    r'ug.npz',
                    r'uu.npz']
    for name in arrays_names:
        arr = np.load(base_path + name)
        # print keys in data
        for key in arr.keys():
            print(f"  {key}: shape = {arr[key].shape}, dtype = {arr[key].dtype}")
            # Change dtype to float64 and save as .npy
            new_arr = arr[key].astype(np.float64)
            # if key contains 'r', 'theta', or 'phi', save as 1D array
            if key == 'r':
                new_arr = new_arr[:,0,0]
                print('r array is ', new_arr)
            elif key == 'theta':
                new_arr = new_arr[0,:,0]
                print('theta array is ', new_arr)
            elif key == 'phi':
                new_arr = new_arr[0,0,:]
                print('phi array is ', new_arr)
            
            np.save((base_path + f"{key}.npy"), new_arr)
    return 

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

def load_photon_output_data():
    output_dir = '/home/siddhant/scratch/GR-ray-tracer/output/'
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    x, y, z, weights = [], [], [], []
    nphotons = 0
    for step in range(1, 1000, 1):
        file_path = f"{output_dir}photons_step_{step:d}.npy"
        data = np.load(file_path)
        print(f"Loaded {file_path} with shape {data.shape} and dtype {data.dtype}")
        x.append(data[:, 1])
        y.append(data[:, 2])
        z.append(data[:, 3])
        weights.append(data[:, -1])
        if step == 1: nphotons = data.shape[0]
    x = np.concatenate(x)
    y = np.concatenate(y)
    z = np.concatenate(z)
    weights = np.concatenate(weights)

    # Add sphere for black hole
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    r = 2.0  # Radius of the black hole
    xs = r * np.outer(np.cos(u), np.sin(v))
    ys = r * np.outer(np.sin(u), np.sin(v))
    zs = r * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(xs, ys, zs, color='r', alpha=0.9)
    for i in range(10):
        i_rand = np.random.randint(0, nphotons)
        ax.plot(x[i_rand::nphotons], y[i_rand::nphotons], z[i_rand::nphotons], alpha=0.3)
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim([-8, 8])
    ax.set_ylim([-8, 8])
    ax.set_zlim([-8, 8])
    ax2 = fig.add_subplot(111)
    ax2.hist(weights, bins=50, log=True)
    ax2.set_xlabel("Photon Weights")
    ax2.set_ylabel("Count")
    plt.savefig("photon_trajectories.png")

def make_photon_image():
    output_dir = '/home/siddhant/scratch/GR-ray-tracer/output/'
    image_width = 200
    image_height = 200
    
    for step in [0, 1000]:
        file_path = f"{output_dir}photons_step_{step:d}.npy"
        data = np.load(file_path)
        print(f"Loaded {file_path} with shape {data.shape} and dtype {data.dtype}")
        if step == 0:
            coords_x = data[:, 1]
            coords_y = data[:, 2]
            coords_z = data[:, 3]
            x_cen = np.mean(coords_x)
            y_cen = np.mean(coords_y)
            z_cen = np.mean(coords_z)
            theta = np.arccos(z_cen / np.sqrt(x_cen**2 + y_cen**2 + z_cen**2))
            phi = np.arctan2(y_cen, x_cen)
            
            # project coords to theta hat and phi hat plane
            theta_hat_x = np.cos(theta) * np.cos(phi)
            theta_hat_y = np.cos(theta) * np.sin(phi)
            theta_hat_z = -np.sin(theta)
            phi_hat_x = -np.sin(phi)
            phi_hat_y = np.cos(phi)
            phi_hat_z = 0.0
            theta_hat_x = 1
            theta_hat_y = 0
            theta_hat_z = 0
            phi_hat_x = 0
            phi_hat_y = 1
            phi_hat_z = 0
            
            image_coords_x = coords_x * theta_hat_x + coords_y * theta_hat_y + coords_z * theta_hat_z
            image_coords_y = coords_x * phi_hat_x + coords_y * phi_hat_y + coords_z * phi_hat_z
        if step == 1000:
            weights = data[:, -1]
    
    # Now bin the image coords to create an image
    image = np.zeros((image_height, image_width))
    image_coords_x_array = np.linspace(image_coords_x.min(), image_coords_x.max(), image_width)
    image_coords_y_array = np.linspace(image_coords_y.min(), image_coords_y.max(), image_height)
    
    hist, xedges, yedges = np.histogram2d(image_coords_x, image_coords_y, 
                                          bins=[image_coords_x_array, image_coords_y_array], 
                                          weights=weights, density=False)

    from matplotlib.colors import LogNorm
    plt.pcolormesh(hist, cmap='inferno', norm=LogNorm(vmin=3e2, vmax=hist.max()))
    plt.colorbar(label='Photon Weights')
    plt.xlabel('Image X Pixel')
    plt.ylabel('Image Y Pixel')
    plt.title('Accumulated Photon Weights on Image Plane')
    plt.savefig("photon_image.png", dpi=300)


if __name__ == "__main__":
    #create_example_data()
    #give_hamr_array_shapes()
    #load_photon_output_data()
    make_photon_image()
