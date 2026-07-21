# GR-ray-tracer

General-relativistic ray tracing with MPI and Kokkos. The same source supports
CPU and GPU execution; the active backend is selected by the installed Kokkos
package used at configure time.

## CPU build

### macOS (Homebrew)

```bash
brew install cmake open-mpi kokkos libomp
cmake -S . -B build-cpu -DBUILD_TESTING=ON
cmake --build build-cpu -j
ctest --test-dir build-cpu --output-on-failure
```

### Linux/HPC

Load an MPI implementation and a CPU Kokkos build (Serial, Threads, or
OpenMP), then point CMake at that Kokkos installation if it is not already on
the search path:

```bash
cmake -S . -B build-cpu \
  -DCMAKE_PREFIX_PATH=/path/to/kokkos-cpu-install \
  -DBUILD_TESTING=ON
cmake --build build-cpu -j
ctest --test-dir build-cpu --output-on-failure
```

## GPU build

Kokkos must be compiled for the target accelerator before configuring this
project. For NVIDIA GPUs, a typical CUDA build is:

```bash
cmake -S /path/to/kokkos -B /path/to/kokkos-build-cuda \
  -DCMAKE_INSTALL_PREFIX=/path/to/kokkos-cuda-install \
  -DCMAKE_CXX_COMPILER=/path/to/kokkos/bin/nvcc_wrapper \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ENABLE_CUDA_LAMBDA=ON \
  -DKokkos_ARCH_HOPPER90=ON
cmake --build /path/to/kokkos-build-cuda -j
cmake --install /path/to/kokkos-build-cuda

cmake -S . -B build-gpu \
  -DCMAKE_CXX_COMPILER=/path/to/kokkos-cuda-install/bin/nvcc_wrapper \
  -DCMAKE_PREFIX_PATH=/path/to/kokkos-cuda-install \
  -DBUILD_TESTING=ON
cmake --build build-gpu -j
ctest --test-dir build-gpu --output-on-failure
```

Replace `Kokkos_ARCH_HOPPER90` with the Kokkos architecture option for the
actual GPU. On AMD systems, use a Kokkos installation built with
`Kokkos_ENABLE_HIP=ON` and configure this project with the same HIP compiler
and installation prefix. The MPI implementation should be GPU-aware when
running across multiple nodes.

## Inputs

The executable accepts one directory containing the NumPy grid coordinates and
fields (see [Command-line flags](#command-line-flags)). It looks for `r.npy`,
`theta.npy`, `phi.npy`, `rho.npy`, `Tgas.npy`, `vel.npy`, and `mag.npy` in that
directory and reports all missing files before initialization. The default
directory is the current working directory. Scalar fields have shape
`(nr, ntheta, nphi)`. Velocity and magnetic fields have shape
`(nr, ntheta, nphi, 4)` and contain contravariant spherical Kerr-Schild
components ordered `(t, r, theta, phi)`. If a file isn't found there,
generate placeholder grid data with `python3 src/create_example_data.py
--output-dir <dir>`, or pass `--vacuum` to skip grid/MHD data entirely and
trace camera-created vacuum geodesics. `--vacuum` also forces scattering off, since
scattering has no medium to scatter off of without a density grid.

Scattering samples the nearest rank-local fluid cell and receives its CGS
density, temperature, coordinate four-velocity, and fluid-frame magnetic
four-vector. It is skipped when no local fluid cell covers the photon or when
the run is in vacuum. The current grey model scatters elastically and
isotropically in the fluid tetrad; its random draws come from a Kokkos random
pool on both CPU and GPU backends.

## Command-line flags

| Flag | Default | Description |
| --- | --- | --- |
| `--config <path>` | none | TOML file overriding black hole mass/spin, camera setup, photon count, termination radii, integration tolerances, and output cadence -- see [Simulation config](#simulation-config) |
| `--vacuum` | off | Skip loading grid/MHD data entirely; forces `--scatter` off |
| `--numpy-dir <path>` | `.` | Directory containing `r.npy`, `theta.npy`, `phi.npy`, `rho.npy`, `Tgas.npy`, `vel.npy`, and `mag.npy` |
| `--output-dir <path>` | `./output` | Directory photon/observation output is written to |
| `--output-variables <csv>` | see [Output](#output) | Comma-separated arrays to include; overrides `output.variables` from TOML |
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
and includes documented starting values. Keys omitted from your file keep
their default -- you only
need to list what you're changing, e.g.:

```toml
[black_hole]
mass = 2.0
spin = 0.5

[photons]
count = 200
```

### Fluid-frame superphotons and code units

Set both `camera.use_pinhole` and `camera.use_image` to `false` to generate
superphotons throughout the loaded grid. In this mode `photons.count` is
ignored: every rank creates `photons.superphotons_per_cell` packets in each of
its local cells. Packet IDs are globally unique unsigned integers, starting at
zero, and runs are rejected before the ID space would exceed one billion.

Every packet has one frequency in Hz. The built-in blackbody generator samples
the Planck energy distribution using the cell's physical temperature. The
power-law generator samples `dE/dnu proportional to nu^(-slope)` between
`nu_min_hz` and `nu_max_hz`. In either case, `energy_per_cell_erg` is divided
equally among that cell's packets and stored in `I`; consequently, summing `I`
over the packets recovers the represented spectral energy. `frequency` is the
frequency in the emitting fluid frame. The geodesic momentum is normalized so
future frequency-dependent interactions can recover the frequency measured by
a local fluid observer from the packet frequency and `-p_mu u^mu` redshift
factor without putting enormous Hz-scale values into the affine integrator.

NumPy values are converted before emission using the `[units]` multipliers:

```toml
[units]
length_cm_per_code = 1.0
time_s_per_code = 1.0
density_g_cm3_per_code = 1.0e-10
temperature_k_per_code = 1.0e10  # Tgas=1 means 1e10 K
four_velocity_cm_s_per_code = 2.99792458e10
magnetic_gauss_per_code = 1.0e4

[photons]
generator = "blackbody"
superphotons_per_cell = 16
energy_per_cell_erg = 1.0e30
```

The initializer converts the coordinate four-velocity to Cartesian
Kerr-Schild form, constructs an orthonormal tetrad whose time leg is the fluid
velocity, and emits isotropically in that rest frame. Magnetic components
provided to an emissivity are also projected into the fluid tetrad. To add a
different emissivity, implement the intentionally small `GeneratePhotonsCustom`
extension point in `src/radiative_transfer/photon_generation.hpp`; it receives
a `FluidCellCGS` and returns a sampled frequency and represented packet energy.
`config/fluid_emission.toml` is a complete non-camera starting configuration.

### MPI domain decomposition

Specify the number of partitions in `(r, theta, phi)` order. Exactly one entry
may be `-1`; it is inferred from the number of MPI ranks. Phi varies fastest,
so this setting partitions only phi and automatically uses all ranks:

```toml
[parallel]
decomposition = [1, 1, -1]
```

For 16 ranks, `[2, 2, -1]` resolves to `[2, 2, 4]`. The resolved product must
equal the rank count, and no axis may have more partitions than grid cells.

Photons terminate once their Kerr-Schild radius leaves `[termination.r_min,
termination.r_max]`, which default to the event horizon and `1.5 *
camera.distance` respectively unless set explicitly.

## Runtime

```bash
./build-cpu/src/gr-ray-trace \
  --config config/example.toml \
  --numpy-dir /path/to/numpy-fields \
  --output-dir ./output \
  --scatter \
  --scatter-optical-depth 0.5 \
  --scatter-albedo 0.9 \
  --integrator rk45
```

Use MPI to run more than one rank, for example:

```bash
mpirun -n 4 ./build-cpu/src/gr-ray-trace --config config/example.toml --vacuum
```

## Output

Every checkpoint produces exactly one file per MPI rank:

```text
output_step_<step>_rank<rank>.npz
```

Each archive contains named NumPy arrays. Select arrays in the TOML file:

```toml
[output]
interval = 20
stride = 4
variables = ["id", "frequency", "x1", "x2", "x3", "k0", "I", "terminate", "image_I"]
```

`output.stride = N` writes rank-local photon indices `0, N, 2N, ...` while
observation products continue to use all photons. The default stride is one.

The command line can override the TOML selection:

```bash
./build-cpu/src/gr-ray-trace --vacuum \
  --output-variables x0,x1,x2,x3,k0,k1,k2,k3,I,image_I
```

Available photon arrays are `id`, `frequency` (Hz), `x0`-`x3`, `k0`-`k3`, `I`, `Q`, `U`, `V`,
`dlambda`, `terminate`, `theta_disp`, and `phi_disp`. Observation arrays are
`image_I`, `image_Q`, `image_U`, `image_V`, `lightcurve_I`,
`spectrum_frequency_hz`, and `spectrum_I`.
Use `all` to write every available array. The default selection preserves the
previous photon and observation content while consolidating it into one
archive.

Analyze completed output or watch checkpoints as they appear:

```bash
python3 src/analyze_output.py post --output-dir ./output
python3 src/analyze_output.py watch --output-dir ./output --mpi-size 4
```

Trajectory and conservation analysis requires `x1`, `x2`, `x3`, and
`k0`-`k3`. Observation plots are generated for whichever observation arrays
are present.

## Tests

The build includes NumPy/domain-decomposition regression tests, fluid-frame
photon generation and scattering checks, output-stride checks, geodesic
observable checks, and RK4/RK45 convergence tests. Run the full suite or only
the convergence test with:

```bash
ctest --test-dir build-cpu --output-on-failure
ctest --test-dir build-cpu -L convergence --output-on-failure
```

## Scaling tests

The scaling runner performs repeated strong- and weak-scaling runs, uses the
median wall time, and writes throughput, speedup, and parallel efficiency to a
CSV file. On a cluster using Open MPI:

```bash
python3 benchmarks/run_scaling.py \
  --executable ./build-cpu/src/gr-ray-trace \
  --ranks 1,2,4,8,16,32 \
  --mode both \
  --photons 1000000 \
  --steps 200 \
  --repeats 3 \
  --output scaling_cpu.csv
```

For Slurm, request an allocation first and let `srun` launch each sweep:

```bash
python3 benchmarks/run_scaling.py \
  --executable ./build-gpu/src/gr-ray-trace \
  --ranks 1,2,4,8 \
  --launcher "srun --cpu-bind=cores --gpus-per-task=1 --gpu-bind=closest" \
  --tasks-flag=-n \
  --mode both \
  --photons 1000000 \
  --steps 200 \
  --output scaling_gpu.csv
```

In strong mode, `--photons` is the fixed global packet count. In weak mode,
it is the packet count per rank. Use a large enough count and step count that
startup and the minimal checkpoint do not dominate the measurement.
