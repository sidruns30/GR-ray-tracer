#!/usr/bin/env python3
"""Generate a small grid/MHD input bundle for local runs and tests."""

import argparse
from pathlib import Path

import numpy as np


def create_example_data(output_dir=".", nr=32, ntheta=16, nphi=32,
                        r_min=3.0, r_max=1000.0):
    """Create the coordinate and field arrays consumed by the C++ loader."""
    if min(nr, ntheta, nphi) < 2:
        raise ValueError("nr, ntheta, and nphi must each be at least 2")
    if r_min <= 0.0 or r_max <= r_min:
        raise ValueError("require 0 < r_min < r_max")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    r = np.logspace(np.log10(r_min), np.log10(r_max), nr)
    theta = np.linspace(1e-3, np.pi - 1e-3, ntheta)
    phi = np.linspace(0.0, 2.0 * np.pi, nphi, endpoint=False)
    radius, polar_angle, _ = np.meshgrid(r, theta, phi, indexing="ij")

    # Dimensionless placeholder primitives. With the conversion factors in
    # config/fluid_emission.toml, temperature=1 represents 1e10 K.
    reference_radius = 10.0
    scale_height = 0.3
    density = (radius / reference_radius) ** -1.5
    density *= np.exp(-((polar_angle - np.pi / 2.0) / scale_height) ** 2)
    density += 1e-6
    temperature = (radius / reference_radius) ** -1.0 + 1.0e-6

    velocity = np.zeros((nr, ntheta, nphi, 4), dtype=np.float64)
    velocity[..., 0] = 1.0
    magnetic = np.zeros_like(velocity)

    arrays = {
        "r.npy": r,
        "theta.npy": theta,
        "phi.npy": phi,
        "rho.npy": density,
        "Tgas.npy": temperature,
        "vel.npy": velocity,
        "mag.npy": magnetic,
    }
    for name, array in arrays.items():
        path = output / name
        np.save(path, array.astype(np.float64, copy=False))
        print(f"Created {path} with shape {array.shape}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--nr", type=int, default=32)
    parser.add_argument("--ntheta", type=int, default=16)
    parser.add_argument("--nphi", type=int, default=32)
    parser.add_argument("--r-min", type=float, default=3.0)
    parser.add_argument("--r-max", type=float, default=1000.0)
    args = parser.parse_args()
    create_example_data(args.output_dir, args.nr, args.ntheta, args.nphi,
                        args.r_min, args.r_max)


if __name__ == "__main__":
    main()
