# gr-ray-trace
A general relativistic ray-tracing code on GPUs that makes use of the Kokkos Framework.

## Overview
This project provides a C++ implementation that uses the following libraries:
- **MPI**
- **OpenMP**
- **Kokkos**
- **cnpy**

## Dependencies
- CMake (>= 3.16)
- C++ compiler with C++17 support
- MPI implementation (OpenMPI, MPICH, etc.)
- OpenMP
- Kokkos (with desired backend: OpenMP, CUDA, HIP, etc.)
- cnpy library
- zlib (required by cnpy)

## Building the Project

### Install Dependencies

#### Kokkos
```bash
git clone https://github.com/kokkos/kokkos.git
cd kokkos
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/path/to/install \
         -DKokkos_ENABLE_OPENMP=ON \
         -DKokkos_ENABLE_CUDA=ON  # Optional: for NVIDIA GPUs
make install
```

#### cnpy
```bash
git clone https://github.com/rogersce/cnpy.git
cd cnpy
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/path/to/install
make install
```
#### mpi (find your MPI installation)



Option 2 — load the module and use find_package inside CMakeLists (modern CMake):
```cmake
# CMakeLists.txt (after project(...))
find_package(MPI REQUIRED COMPONENTS C CXX)
if(MPI_FOUND)
    message(STATUS "MPI include: ${MPI_INCLUDE_PATH}")
    message(STATUS "MPI libs: ${MPI_LIBRARIES}")
    # Prefer imported targets when available
    target_link_libraries(my_target PRIVATE MPI::MPI_C MPI::MPI_CXX)
endif()
```

### Build gr-ray-trace
```bash
mkdir build && cd build
cmake .. -DKokkos_DIR=/path/to/kokkos/lib/cmake/Kokkos \
         -DCNPY_INCLUDE_DIR=/path/to/cnpy/include \
         -DCNPY_LIBRARY=/path/to/cnpy/lib/libcnpy.a
make
```

## Running the Code

### Single process
```bash
./gr-ray-trace
```

### With MPI (multiple processes)
```bash
mpirun -np 4 ./gr-ray-trace
```

### With specific OpenMP thread count
```bash
export OMP_NUM_THREADS=8
mpirun -np 2 ./gr-ray-trace
```

## Code Structure

### Main Function
The `main` function initializes:
1. MPI environment
2. Kokkos execution space
3. OpenMP threading
4. Example parallel computations

### Loading NPY Files
```cpp
std::vector<double> data;
std::vector<size_t> shape;
load_npy_file<double>("example_data.npy", data, shape);
```

### Initializing Kokkos Arrays
```cpp
// 1D array
auto array_1d = initialize_kokkos_array_1d(data, "my_array");

// 2D array
auto array_2d = initialize_kokkos_array_2d(data, dim0, dim1, "my_2d_array");
```

## Creating Test Data
A Python script is provided to create example `.npy` files for testing:
```bash
python3 create_example_data.py
```

This will create several example `.npy` files with different shapes and data types that you can use to test the data loading functionality.

## License
See LICENSE file for details.
