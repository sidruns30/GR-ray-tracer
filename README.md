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

Set `input.numpy_dir` in the TOML file to the directory containing the NumPy
grid coordinates and fields. The program looks for `r.npy`,
`theta.npy`, `phi.npy`, `rho.npy`, `Tgas.npy`, `vel.npy`, and `mag.npy` in that
directory and reports all missing files before initialization. The default
directory is the current working directory. The three coordinate files and
the scalar fluid fields all have shape `(nr, ntheta, nphi)`. Coordinate values
may be nonuniform and may depend on all three logical indices. Velocity and
magnetic fields have shape
`(nr, ntheta, nphi, 4)` and contain contravariant spherical Kerr-Schild
components ordered `(t, r, theta, phi)`. If a file is not found there,
generate an example Fishbone-Moncrief torus with
`python3 src/create_example_data.py --output-dir <dir>`. Both supported modes
need this grid: image mode uses it as the emitting disk and disk mode emits
packets from its cells. `input.vacuum=true` is therefore rejected.

`src/create_example_data.py` has one public function,
`create_fishbone_moncrief_torus`, which builds an analytic, hydrostatic
constant-angular-momentum torus (Fishbone & Moncrief 1976) on a
user-specified `(nr, ntheta, nphi)` grid. Its `--mass` and `--spin` must match
`[black_hole]` in the simulation TOML that will consume the data, since the
torus solution is computed in that spacetime. `--r-in` and `--r-pressure-max`
set the torus inner edge and density-maximum radius; not every combination
(together with spin) produces a torus with a closed outer edge -- the script
prints a warning if the disk instead fills the grid out to `--r-max`. Cells
outside the torus get a small floor density/temperature (`--rho-floor`) and a
zero-angular-momentum-observer velocity, which stays a valid four-velocity
at any radius outside the horizon (including inside the ergosphere).

Scattering samples the nearest rank-local fluid cell and receives its CGS
density, temperature, coordinate four-velocity, and fluid-frame magnetic
four-vector. It is skipped when no local fluid cell covers the photon or when
the run is in vacuum. The current grey model scatters elastically and
isotropically in the fluid tetrad; its random draws come from a Kokkos random
pool on both CPU and GPU backends.

Radiative transfer applies two further, independent effects at that same
sampled fluid cell, once per integration step: optical-depth absorption,
which only attenuates Stokes I/Q/U/V (multiplicatively, by `exp(-dtau)`), and
emissivity, which only adds to Stokes I. Both are user-editable extension
points -- `ComputeOpticalDepthIncrement` and `ComputeEmissivityIncrement` in
[`src/radiative_transfer/radiative_transfer.hpp`](src/radiative_transfer/radiative_transfer.hpp)
-- with default bodies implementing a simple grey, density-weighted opacity
and a density*temperature-weighted emissivity, each scaled by a TOML-set
coefficient. Replace either body to depend on `frequency_hz` or
`fluid.magnetic_gauss` for a physical model (e.g. synchrotron), and see that
file's header comment for the full contract. Enable with:

```toml
[radiative_transfer]
absorption_enabled = true
emission_enabled = true
absorption_coefficient = 1.0
emission_coefficient = 1.0
```

## Program input

The executable accepts exactly one argument: the simulation TOML file. Runtime
flags are intentionally not supported, so every simulation setting is recorded
in one reproducible configuration:

```bash
./build-cpu/src/gr-ray-trace config/example.toml
```

The following TOML keys replace the former runtime flags:

| TOML key | Default | Description |
| --- | --- | --- |
| `simulation.mode` | `"image"` | Photon workflow: `"image"` or `"disk"` |
| `input.vacuum` | `false` | Reserved; both current modes require `false` |
| `input.numpy_dir` | `"."` | Directory containing the seven required NumPy arrays |
| `integration.integrator` | `"rk45"` | Geodesic integrator: `"rk4"` or `"rk45"` |
| `scattering.enabled` | `false` | Enable photon scattering when fluid data is available |
| `scattering.optical_depth` | `0.0` | Non-negative optical depth used by the scattering model |
| `scattering.albedo` | `1.0` | Scattering albedo in `[0, 1]` |
| `scattering.seed` | `12345` | Random seed for reproducible scattering draws |
| `radiative_transfer.absorption_enabled` | `false` | Enable optical-depth attenuation of Stokes I/Q/U/V |
| `radiative_transfer.emission_enabled` | `false` | Enable emissivity additions to Stokes I |
| `radiative_transfer.absorption_coefficient` | `0.0` | Non-negative scale factor for the default grey opacity |
| `radiative_transfer.emission_coefficient` | `0.0` | Non-negative scale factor for the default grey emissivity |
| `output.directory` | `"./output"` | Directory receiving photon and observation archives |
| `output.variables` | see [Output](#output) | Arrays included in each output archive |

## Simulation config

Black hole mass/spin, camera setup, photon count, run length, termination
radii, integration tolerances, scattering, input paths, and output behavior
are configured by the required TOML file. Keys omitted from it retain the
built-in defaults in [`src/utils.hpp`](src/utils.hpp):

```bash
./build/src/gr-ray-trace config/example.toml
```

[`config/example.toml`](config/example.toml) documents every available key
and includes documented starting values. You only need to list values that
differ from their defaults, for example:

```toml
[black_hole]
mass = 2.0
spin = 0.5

[photons]
count = 200
```

### Image and disk modes

The mode is explicit and mutually exclusive:

```toml
[simulation]
mode = "image"  # or "disk"
```

In `image` mode, rays start across the finite image plane and are traced
inward. A ray that reaches the first positive-density fluid cell is assigned
an emitted spectrum in that fluid frame, then retraces toward the camera while
radiative interactions are applied. Rays swallowed by the black hole, lost
through the outer boundary, or missing the finite camera plane are rejected.
Only packets that cross the camera plane contribute to the final image and
spectrum.

In `disk` mode there is no camera. Packets are generated throughout the loaded
grid and transported forward. This mode writes packet checkpoints only; it
does not create image, spectrum, or light-curve products.

The removed pinhole camera and the former `camera.use_pinhole` /
`camera.use_image` flags are not supported. Image geometry is configured with
`camera.distance`, `camera.theta`, `camera.phi`, `camera.width`, and
`camera.height`.

### Fluid-frame superphotons and code units

Set `simulation.mode = "disk"` to generate superphotons throughout the loaded
grid. In this mode `photons.count` is
ignored: every rank creates `photons.superphotons_per_cell` packets in each of
its local cells. Packet IDs are globally unique unsigned integers, starting at
zero. Counts and IDs use 64 bits, and runs are rejected above 100 billion
photons globally. Each rank retains a 32-bit local kernel index, so distribute
large runs across enough MPI ranks that no rank receives more than roughly
2.1 billion photons; GPU memory will normally impose a much lower local limit.

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

After every integration step, active photons are assigned to the closest MPI
grid domain from their Kerr-Schild `(r, theta, phi)` position. Complete photon
records are grouped by destination and transferred with one `MPI_Alltoallv`,
so a receiving GPU continues with the same ID, momentum, frequency, Stokes
state, adaptive step, and image phase. This fixes the former behavior where a
photon stayed on its launch rank and silently stopped seeing fluid after it
crossed a domain boundary. Classification, counting, packing, and local
compaction run in Kokkos on the active CPU/GPU backend. Only outgoing bundles
are staged through host memory, and a step with no crossings skips full-state
movement. This portable path does not require CUDA-aware MPI.

Photons terminate once their Kerr-Schild radius leaves `[termination.r_min,
termination.r_max]`, which default to the event horizon and `1.5 *
camera.distance` respectively unless set explicitly.

## Runtime

```bash
./build-cpu/src/gr-ray-trace config/example.toml
```

Use MPI to run more than one rank, for example:

```bash
mpirun -n 4 ./build-cpu/src/gr-ray-trace config/example.toml
```

## Output

Every checkpoint produces exactly one file per MPI rank:

```text
output_step_<step>_rank<rank>.npz
```

Each archive contains named NumPy arrays. Select arrays in the TOML file:

```toml
[output]
directory = "./output"
interval = 20
stride = 4
variables = ["id", "frequency", "x1", "x2", "x3", "k0", "I", "terminate", "phase"]
```

`output.stride = N` writes rank-local photon indices `0, N, 2N, ...`. The
default stride is one. Because ownership changes, downstream tools must join
rank files by `id`; keep `id` selected whenever trajectories are needed.

For very large distributed runs, increase `output.stride` and select only the
arrays needed for analysis. Each rank writes a classic non-ZIP64 NPZ archive,
which must remain below 4 GB. The current writer also creates full host mirrors
of selected arrays before applying the stride, so provision node RAM for those
mirrors; for example, `id`, `frequency`, and `I` require approximately 24 bytes
of host memory per rank-local photon.

Available checkpoint arrays are `id`, `frequency` (Hz),
`emission_frame_energy`, `x0`-`x3`, `k0`-`k3`,
`I`, `Q`, `U`, `V`, `dlambda`, `terminate`, `phase`, `theta_disp`, and
`phi_disp`. Use `all` to write every available array. Phase values are
`0=disk`, `1=image backward`, `2=image forward`, `3=arrived`, and
`4=rejected`.

At the end of image mode, rank zero also writes `image_products.npz`. It
contains globally reduced `image_I`, `image_Q`, `image_U`, `image_V`,
`spectrum_frequency_hz`, `spectrum_I`, and `arrived_count`. Its image and
spectrum axes are controlled by `output.image_nx`, `output.image_ny`,
`output.spectrum_bins`, `output.spectrum_min_hz`, and
`output.spectrum_max_hz`. Disk mode never writes this file.

Analyze completed output or watch checkpoints as they appear:

```bash
python3 src/analyze_output.py post --output-dir ./output
python3 src/analyze_output.py watch --output-dir ./output --mpi-size 4
```

Trajectory and conservation analysis requires `x1`, `x2`, `x3`, and
`k0`-`k3`. The analyzer plots final camera products when
`image_products.npz` is present and otherwise treats the run as disk mode.

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
CSV file. It creates a temporary zero-density grid sized for the largest rank
count, so the benchmark needs NumPy but no production input snapshot. On a
cluster using Open MPI:

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
