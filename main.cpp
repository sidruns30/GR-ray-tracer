#include <iostream>
#include <vector>
#include <string>
#include <mpi.h>
#include <omp.h>
#include <Kokkos_Core.hpp>
#include "cnpy.h"

// Function to load npy files
template<typename T>
void load_npy_file(const std::string& filename, std::vector<T>& data, std::vector<size_t>& shape) {
    // Load the npy file
    cnpy::NpyArray arr = cnpy::npy_load(filename);
    
    // Get shape information
    shape = arr.shape;
    
    // Get the total number of elements
    size_t num_elements = 1;
    for (size_t dim : shape) {
        num_elements *= dim;
    }
    
    // Resize the data vector
    data.resize(num_elements);
    
    // Copy data from the npy array
    T* loaded_data = arr.data<T>();
    for (size_t i = 0; i < num_elements; ++i) {
        data[i] = loaded_data[i];
    }
    
    std::cout << "Loaded npy file: " << filename << std::endl;
    std::cout << "Shape: [";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];
        if (i < shape.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

// Function to initialize Kokkos arrays
template<typename T>
Kokkos::View<T*> initialize_kokkos_array_1d(const std::vector<T>& host_data, const std::string& label) {
    // Create a Kokkos view on the device
    Kokkos::View<T*> device_array(label, host_data.size());
    
    // Create a mirror view on the host
    auto host_array = Kokkos::create_mirror_view(device_array);
    
    // Copy data from std::vector to host mirror
    for (size_t i = 0; i < host_data.size(); ++i) {
        host_array(i) = host_data[i];
    }
    
    // Deep copy from host to device
    Kokkos::deep_copy(device_array, host_array);
    
    std::cout << "Initialized Kokkos array '" << label << "' with " 
              << host_data.size() << " elements" << std::endl;
    
    return device_array;
}

// Overload for 2D arrays
template<typename T>
Kokkos::View<T**> initialize_kokkos_array_2d(const std::vector<T>& host_data, 
                                              size_t dim0, size_t dim1, 
                                              const std::string& label) {
    // Create a Kokkos view on the device
    Kokkos::View<T**> device_array(label, dim0, dim1);
    
    // Create a mirror view on the host
    auto host_array = Kokkos::create_mirror_view(device_array);
    
    // Copy data from std::vector to host mirror
    for (size_t i = 0; i < dim0; ++i) {
        for (size_t j = 0; j < dim1; ++j) {
            host_array(i, j) = host_data[i * dim1 + j];
        }
    }
    
    // Deep copy from host to device
    Kokkos::deep_copy(device_array, host_array);
    
    std::cout << "Initialized Kokkos 2D array '" << label << "' with dimensions [" 
              << dim0 << ", " << dim1 << "]" << std::endl;
    
    return device_array;
}

int main(int argc, char* argv[]) {
    // Initialize MPI
    int mpi_rank, mpi_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    
    if (mpi_rank == 0) {
        std::cout << "=== GR Ray-Trace Initialization ===" << std::endl;
        std::cout << "MPI processes: " << mpi_size << std::endl;
    }
    
    // Initialize Kokkos
    Kokkos::initialize(argc, argv);
    {
        if (mpi_rank == 0) {
            std::cout << "Kokkos execution space: " 
                      << typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;
            
            #ifdef KOKKOS_ENABLE_OPENMP
            std::cout << "Kokkos OpenMP backend enabled" << std::endl;
            #endif
            #ifdef KOKKOS_ENABLE_CUDA
            std::cout << "Kokkos CUDA backend enabled" << std::endl;
            #endif
            #ifdef KOKKOS_ENABLE_HIP
            std::cout << "Kokkos HIP backend enabled" << std::endl;
            #endif
        }
        
        // Set number of OpenMP threads
        #ifdef _OPENMP
        int num_threads = omp_get_max_threads();
        if (mpi_rank == 0) {
            std::cout << "OpenMP threads available: " << num_threads << std::endl;
        }
        #endif
        
        // Example: Load an npy file (if it exists)
        // Uncomment and modify the following lines to load actual data
        /*
        std::vector<double> data;
        std::vector<size_t> shape;
        load_npy_file<double>("example_data.npy", data, shape);
        
        // Initialize Kokkos array from loaded data
        auto kokkos_array = initialize_kokkos_array_1d(data, "example_array");
        */
        
        // Example parallel region with OpenMP
        #pragma omp parallel
        {
            #ifdef _OPENMP
            int tid = omp_get_thread_num();
            #pragma omp critical
            {
                std::cout << "Rank " << mpi_rank << ", Thread " << tid 
                          << " is running" << std::endl;
            }
            #endif
        }
        
        // Example Kokkos parallel_for
        const int N = 10;
        Kokkos::View<double*> test_array("test", N);
        
        Kokkos::parallel_for("initialize_test", N, KOKKOS_LAMBDA(const int i) {
            test_array(i) = i * 2.0;
        });
        Kokkos::fence();
        
        // Copy back to host to verify
        auto host_test = Kokkos::create_mirror_view(test_array);
        Kokkos::deep_copy(host_test, test_array);
        
        if (mpi_rank == 0) {
            std::cout << "\nTest Kokkos array values: ";
            for (int i = 0; i < N; ++i) {
                std::cout << host_test(i) << " ";
            }
            std::cout << std::endl;
        }
        
        // MPI Barrier to synchronize all processes
        MPI_Barrier(MPI_COMM_WORLD);
        
        if (mpi_rank == 0) {
            std::cout << "\n=== Initialization Complete ===" << std::endl;
        }
    }
    
    // Finalize Kokkos
    Kokkos::finalize();
    
    // Finalize MPI
    MPI_Finalize();
    
    return 0;
}
