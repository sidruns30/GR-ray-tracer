#!/usr/bin/env python3
"""
Example script to create NumPy .npy files for testing the gr-ray-trace code.

This script demonstrates how to create sample data files that can be loaded
by the C++ application using the cnpy library.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

def give_hamr_array_shapes():
    base_path = r'/Users/siddhantsolanki/Desktop/code/GR-ray-tracer/build/'
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
            # Change dtype to float32 and save as .npy
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

def create_example_data(output_dir='.', nr=32, ntheta=16, nphi=32,
                         r_min=1.5, r_max=1000.0):
    """Create the r/theta/phi/rho/Tgas/vel/mag .npy files gr-ray-trace expects
    (see src/input/load_python_arrays.hpp: load_numpy_field_bundle) so the
    binary can actually be run without a real HAMR dataset.

    Grid conventions enforced by the loader:
      - r must be uniformly spaced in log-space (np.logspace satisfies this).
      - theta, phi must be uniformly spaced linearly (np.linspace satisfies this).
      - rho, Tgas have shape (nr, ntheta, nphi).
      - vel, mag have shape (nr, ntheta, nphi, 4) -- placeholder 4-vectors
        (u^mu, b^mu); nothing in the geodesic integrator consumes them yet,
        so their values only need to be finite and correctly shaped.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    r = np.logspace(np.log10(r_min), np.log10(r_max), nr)
    theta = np.linspace(1e-3, np.pi - 1e-3, ntheta)
    phi = np.linspace(0.0, 2.0 * np.pi, nphi, endpoint=False)

    rr, tt, _ = np.meshgrid(r, theta, phi, indexing='ij')

    # Rough thin-disk-like profile, purely illustrative -- not physically
    # consumed by the vacuum geodesic integrator today.
    r0, scale_height = 10.0, 0.3
    rho = 1.0 * (rr / r0) ** -1.5 * np.exp(-((tt - np.pi / 2.0) / scale_height) ** 2) + 1e-6
    Tgas = 1.0e9 * (rr / r0) ** -1.0 + 1.0

    vel = np.zeros((nr, ntheta, nphi, 4), dtype=np.float64)
    vel[..., 0] = 1.0  # u^t = 1, spatial components 0 (placeholder)
    mag = np.zeros((nr, ntheta, nphi, 4), dtype=np.float64)

    arrays = {
        'r.npy': r,
        'theta.npy': theta,
        'phi.npy': phi,
        'rho.npy': rho,
        'Tgas.npy': Tgas,
        'vel.npy': vel,
        'mag.npy': mag,
    }
    for name, arr in arrays.items():
        path = out / name
        np.save(path, arr.astype(np.float64))
        print(f"Created {path} with shape {arr.shape}")

    print(f"\nAll grid .npy files written to {out.resolve()}")
    print("Run the binary from that directory (or pass --grid-r/--density/... "
          "pointing at these files) to use them.")

def plot_photon_trajectories():
    output_dir = '/home/siddhant/scratch/GR-ray-tracer/output/'
    
    # ----------------------------
    # Load Photon Data
    # ----------------------------
    xs, ys, zs, ws = [], [], [], []
    for step in range(0, 700, 2):
        f = f"{output_dir}photon_output_{step}_rank0.npz"
        data = np.load(f)
        xs.append(data["x1"])
        ys.append(data["x2"])
        zs.append(data["x3"])
        ws.append(data["I"])

    x = np.concatenate(xs)
    y = np.concatenate(ys)
    z = np.concatenate(zs)
    w = np.concatenate(ws)

    # ----------------------------
    # Configure Figure
    # ----------------------------
    plt.style.use("dark_background")

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)
    
    # Make background fully black
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")

    # ----------------------------
    # Black Hole Sphere
    # ----------------------------
    r_bh = 2.0
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    xsph = r_bh * np.outer(np.cos(u), np.sin(v))
    ysph = r_bh * np.outer(np.sin(u), np.sin(v))
    zsph = r_bh * np.outer(np.ones_like(u), np.cos(v))

    ax.plot_surface(
        xsph, ysph, zsph,
        rstride=2, cstride=2,
        color="gray",
        edgecolor="none",
        alpha=0.35,   # matte black-hole shading
        antialiased=True,
    )

    # ----------------------------
    # Photon Trajectories
    # ----------------------------
    # Pick a subset for clarity
    stride = 20
    for i in range(stride):
        ax.plot(
            x[i::stride], y[i::stride], z[i::stride],
            lw=1.0,
            alpha=0.8,
            color=plt.cm.jet((i / stride)**0.5, )
        )

    # ----------------------------
    # Labels & Bounds
    # ----------------------------
    lim = 8
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])

    ax.set_xlabel("X", labelpad=10, fontsize=12, color="white")
    ax.set_ylabel("Y", labelpad=10, fontsize=12, color="white")
    ax.set_zlabel("Z", labelpad=10, fontsize=12, color="white")

    # ----------------------------
    # Camera Angle
    # ----------------------------
    elev = 22
    for azim in (0, 120, 240):
        ax.view_init(elev=elev, azim=azim)
        fig.savefig(f"photon_trajectories_phi{azim}.png", dpi=300, bbox_inches="tight")
    plt.close()


def make_photon_image():
    output_dir = '/home/siddhant/scratch/GR-ray-tracer/output/'
    image_width = 200
    image_height = 200
    weights_list = []
    for step in range(0, 9001, 1000):
        file_path = f"{output_dir}photon_output_{step:d}_rank0.npz"
        data = np.load(file_path)
        if step == 0:
            coords_x = data["x1"]
            coords_y = data["x2"]
            coords_z = data["x3"]
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

            image_coords_x = coords_x * theta_hat_x + coords_y * theta_hat_y + coords_z * theta_hat_z
            image_coords_y = coords_x * phi_hat_x + coords_y * phi_hat_y + coords_z * phi_hat_z
        else:
            weights = data["I"]
            weights_list.append(weights)

    weights = np.sum(np.array(weights_list), axis=0)
    # Now bin the image coords to create an image
    image = np.zeros((image_height, image_width))
    image_coords_x_array = np.linspace(image_coords_x.min(), image_coords_x.max(), image_width)
    image_coords_y_array = np.linspace(image_coords_y.min(), image_coords_y.max(), image_height)
    
    hist, xedges, yedges = np.histogram2d(image_coords_x, image_coords_y, 
                                          bins=[image_coords_x_array, image_coords_y_array], 
                                          weights=weights, density=False)

    from matplotlib.colors import LogNorm, Normalize
    plt.pcolormesh(hist, cmap='inferno', norm=Normalize(vmin=3e2, vmax=hist.max()))
    plt.colorbar(label='Photon Weights')
    plt.xlabel('Image X Pixel')
    plt.ylabel('Image Y Pixel')
    plt.title('Accumulated Photon Weights on Image Plane')
    plt.savefig("photon_image.png", dpi=300)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate example grid .npy files for gr-ray-trace.")
    parser.add_argument("--output-dir", default=".", help="Directory to write r/theta/phi/rho/Tgas/vel/mag.npy to")
    parser.add_argument("--nr", type=int, default=32)
    parser.add_argument("--ntheta", type=int, default=16)
    parser.add_argument("--nphi", type=int, default=32)
    parser.add_argument("--r-min", type=float, default=1.5)
    parser.add_argument("--r-max", type=float, default=1000.0)
    args = parser.parse_args()

    create_example_data(output_dir=args.output_dir, nr=args.nr, ntheta=args.ntheta, nphi=args.nphi,
                         r_min=args.r_min, r_max=args.r_max)
    #give_hamr_array_shapes()
    #plot_photon_trajectories()
    #make_photon_image()
