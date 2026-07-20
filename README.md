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

The executable accepts NumPy `.npy` files for:
- `--grid-r`
- `--grid-theta`
- `--grid-phi`
- `--density`
- `--temperature`
- `--velocity`
- `--magnetic`

If omitted, it falls back to `./r.npy`, `./theta.npy`, `./phi.npy`, `./rho.npy`, `./Tgas.npy`, `./vel.npy`, and `./mag.npy`.

## Runtime

```bash
./build/src/gr-ray-trace \
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
