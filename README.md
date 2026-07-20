# GR-ray-tracer

GR ray tracing with MPI, OpenMP, and Kokkos.

## Build

### macOS
```bash
brew install libomp kokkos
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build -j
ctest --test-dir build
```

### HPC
Load your MPI/OpenMP/Kokkos modules, then:
```bash
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build -j
```

## Inputs

The executable accepts NumPy `.npy` files for the grid coordinates and fields
(see [Command-line flags](#command-line-flags)). If omitted, it falls back to
`./r.npy`, `./theta.npy`, `./phi.npy`, `./rho.npy`, `./Tgas.npy`, `./vel.npy`,
and `./mag.npy` in the current working directory. If a file isn't found there,
generate placeholder grid data with `python3 src/create_example_data.py
--output-dir <dir>`.

## Command-line flags

| Flag | Default | Description |
| --- | --- | --- |
| `--config <path>` | none | TOML file overriding black hole mass/spin, camera setup, photon count, termination radii, integration tolerances, and output cadence -- see [Simulation config](#simulation-config) |
| `--grid-r <path>` | `./r.npy` | 1D `r` coordinate array (must be log-uniformly spaced) |
| `--grid-theta <path>` | `./theta.npy` | 1D `theta` coordinate array (must be linearly uniform) |
| `--grid-phi <path>` | `./phi.npy` | 1D `phi` coordinate array (must be linearly uniform) |
| `--density <path>` | `./rho.npy` | 3D density field, shape `(nr, ntheta, nphi)` |
| `--temperature <path>` | `./Tgas.npy` | 3D temperature field, shape `(nr, ntheta, nphi)` |
| `--velocity <path>` | `./vel.npy` | 4D fluid velocity field, shape `(nr, ntheta, nphi, 4)` |
| `--magnetic <path>` | `./mag.npy` | 4D magnetic field, shape `(nr, ntheta, nphi, 4)` |
| `--output-dir <path>` | `./output/` | Directory photon/observation output is written to |
| `--scatter` | off | Enable photon scattering |
| `--scatter-optical-depth <float>` | `0.0` | Optical depth used by the scattering model |
| `--scatter-albedo <float>` | `1.0` | Scattering albedo (1.0 = fully scattering, 0.0 = fully absorbing) |
| `--integrator <rk4\|rk45>` | `rk45` | Geodesic integrator: fixed-step RK4 or adaptive RK45 |

## Simulation config

Black hole mass/spin, camera setup, photon count, run length, termination
radii, integration tolerances, and output cadence are read from
[`src/utils.hpp`](src/utils.hpp)'s built-in defaults unless overridden with a
TOML file:

```bash
./build/src/gr-ray-trace --config config/example.toml
```

[`config/example.toml`](config/example.toml) documents every available key
(and shows the compiled-in default for each, so it's a safe copy/paste
starting point). Keys omitted from your file keep their default -- you only
need to list what you're changing, e.g.:

```toml
[black_hole]
mass = 2.0
spin = 0.5

[photons]
count = 200
```

Photons terminate once their Kerr-Schild radius leaves `[termination.r_min,
termination.r_max]`, which default to the event horizon and `1.5 *
camera.distance` respectively unless set explicitly.

## Runtime

```bash
./build/src/gr-ray-trace \
  --config config/example.toml \
  --grid-r r.npy \
  --grid-theta theta.npy \
  --grid-phi phi.npy \
  --density rho.npy \
  --temperature Tgas.npy \
  --velocity vel.npy \
  --magnetic mag.npy \
  --output-dir ./output \
  --scatter \
  --scatter-optical-depth 0.5 \
  --scatter-albedo 0.9 \
  --integrator rk45
```

## Tests

The build includes:
- NumPy loader regression
- Geodesic observable checks
- Analytic RK4/RK45 stepping checks
