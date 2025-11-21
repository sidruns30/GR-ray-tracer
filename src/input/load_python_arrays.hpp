/*
    Load the HAMR data stored in numpy arrays and store them in Kokkos views
*/
#pragma once
#include "../utils.hpp"
#include "../utils.cpp"



// Function to load npy files into Kokkos views

// 1D array loader
template<typename T>
void load_1d_numpy_array(   const std::string& filename, 
                            Kokkos::View<T*>& kokkos_array
                            ) {
    // Load the npy file using cnpy
    cnpy::NpyArray arr = cnpy::npy_load(filename);
    
    // Check dimensions
    if (arr.shape.size() != 1) 
    {
        throw std::runtime_error("Expected 1D array in " + filename);
    }
    
    size_t size = arr.shape[0];
    
    // Create a Kokkos view on the device
    kokkos_array = Kokkos::View<T*>("loaded_array", size);
    
    // Create a mirror view on the host
    auto host_array = Kokkos::create_mirror_view(kokkos_array);
    
    // Copy data from cnpy to host mirror
    T* data_ptr = arr.data<T>();
    for (size_t i = 0; i < size; ++i) {
        host_array(i) = data_ptr[i];
    }
    
    // Deep copy from host to device
    Kokkos::deep_copy(kokkos_array, host_array);

    if (verbose ) {
        std::cout << "Loaded 1D numpy array from " << filename << " with " 
                  << size << " elements into Kokkos view." << std::endl;
    }
}

//  2D array loader
template<typename T>
void load_2d_numpy_array(   const std::string& filename, 
                            Kokkos::View<T**>& kokkos_array
                            ) {
    // Load the npy file using cnpy
    cnpy::NpyArray arr = cnpy::npy_load(filename);

    // Check dimensions
    if (arr.shape.size() != 2) {
        throw std::runtime_error("Expected 2D array in " + filename);
    }

    size_t dim0 = arr.shape[0];
    size_t dim1 = arr.shape[1];

    // Create a Kokkos view on the device
    kokkos_array = Kokkos::View<T**>("loaded_array", dim0, dim1);

    // Create a mirror view on the host
    auto host_array = Kokkos::create_mirror_view(kokkos_array);

    // Copy data from cnpy to host mirror
    T* data_ptr = arr.data<T>();
    for (size_t i = 0; i < dim0; ++i) {
        for (size_t j = 0; j < dim1; ++j) {
            host_array(i, j) = data_ptr[i * dim1 + j];
        }
    }

    // Deep copy from host to device
    Kokkos::deep_copy(kokkos_array, host_array);

    if (verbose ) {
        std::cout << "Loaded 2D numpy array from " << filename << " with dimensions [" 
                  << dim0 << ", " << dim1 << "] into Kokkos view." << std::endl;
    }
}

// 3D array loader
template<typename T>
void load_3d_numpy_array(   const std::string& filename, 
                            Kokkos::View<T***>& kokkos_array
                            ) {
    // Load the npy file using cnpy
    cnpy::NpyArray arr = cnpy::npy_load(filename);
    
    // Check dimensions
    if (arr.shape.size() != 3) {
        throw std::runtime_error("Expected 3D array in " + filename);
    }

    size_t dim0 = arr.shape[0];
    size_t dim1 = arr.shape[1];
    size_t dim2 = arr.shape[2];

    // Create a Kokkos view on the device
    kokkos_array = Kokkos::View<T***>("loaded_array", dim0, dim1, dim2);

    // Create a mirror view on the host
    auto host_array = Kokkos::create_mirror_view(kokkos_array);

    // Copy data from cnpy to host mirror
    T* data_ptr = arr.data<T>();
    for (size_t i = 0; i < dim0; ++i) {
        for (size_t j = 0; j < dim1; ++j) {
            for (size_t k = 0; k < dim2; ++k) {
                host_array(i, j, k) = data_ptr[i * dim1 * dim2 + j * dim2 + k];
            }
        }
    }

    // Deep copy from host to device
    Kokkos::deep_copy(kokkos_array, host_array);

    if (verbose ) {
        std::cout << "Loaded 3D numpy array from " << filename << " with dimensions [" 
                  << dim0 << ", " << dim1 << ", " << dim2 << "] into Kokkos view." << std::endl;
    }
}

// 4D array loader
template<typename T>
void load_4d_numpy_array(   const std::string& filename, 
                            Kokkos::View<T****>& kokkos_array
                            ) {
    // Load the npy file using cnpy
    cnpy::NpyArray arr = cnpy::npy_load(filename);

    // Check dimensions
    if (arr.shape.size() != 4) {
        throw std::runtime_error("Expected 4D array in " + filename);
    }

    size_t dim0 = arr.shape[0];
    size_t dim1 = arr.shape[1];
    size_t dim2 = arr.shape[2];
    size_t dim3 = arr.shape[3];

    // Create a Kokkos view on the device
    kokkos_array = Kokkos::View<T****>("loaded_array", dim0, dim1, dim2, dim3);

    // Create a mirror view on the host
    auto host_array = Kokkos::create_mirror_view(kokkos_array);

    // Copy data from cnpy to host mirror
    T* data_ptr = arr.data<T>();
    for (size_t i = 0; i < dim0; ++i) {
        for (size_t j = 0; j < dim1; ++j) {
            for (size_t k = 0; k < dim2; ++k) {
                for (size_t l = 0; l < dim3; ++l) {
                    host_array(i, j, k, l) = data_ptr[i * dim1 * dim2 * dim3 + j * dim2 * dim3 + k * dim3 + l];
                }
            }
        }
    }

    // Deep copy from host to device
    Kokkos::deep_copy(kokkos_array, host_array);

    if (verbose ) {
        std::cout << "Loaded 4D numpy array from " << filename << " with dimensions [" 
                  << dim0 << ", " << dim1 << ", " << dim2 << ", " << dim3 << "] into Kokkos view." << std::endl;
    }
}

template<typename T>
void load_coordinate_array(const std::string coord_type, 
                           const std::string& filename, 
                           Kokkos::View<T*>& kokkos_array) {
    
    cnpy::NpyArray arr = cnpy::npy_load(filename);
    // Check dimensions
    if (arr.shape.size() != 1) 
    {
        throw std::runtime_error("Expected 1D array in " + filename);
    }
    
    size_t size = arr.shape[0];
    // Create a Kokkos view on the device
    kokkos_array = Kokkos::View<T*>("loaded_array", size);
    
    // Create a mirror view on the host
    auto host_array = Kokkos::create_mirror_view(kokkos_array);
    
    // Copy data from cnpy to host mirror
    T* data_ptr = arr.data<T>();
    for (size_t i = 0; i < size; ++i) {
        host_array(i) = data_ptr[i];
    }
    
    // Deep copy from host to device
    Kokkos::deep_copy(kokkos_array, host_array);

    if (verbose ) {
        std::cout << "Loaded coordinate array for " << coord_type << " from " << filename << std::endl;
        std::cout << "Array size: " << size << std::endl;
    }
    if (coord_type == "r") {
        r_min = host_array(0);
        r_max = host_array(size - 1);
        nr = size;
        dlog_r = log(host_array(1)/host_array(0));
        // Check uniform spacing in log-space
        for (size_t i = 1; i < size - 1; ++i) {
            double dlogr_i = log(host_array(i+1)/host_array(i));
            if (fabs(dlogr_i - dlog_r) > 1e-2) {
                throw std::runtime_error("r coordinate array is not uniformly spaced in log-space.");
            }
            if (verbose ) {
                std::cout << host_array(i) << " ";
            }
        }
        std::cout << std::endl;
    }
    else if (coord_type == "theta") {
        theta_min = host_array(0);
        theta_max = host_array(size - 1);
        ntheta = size;
        // Check uniform spacing
        double dtheta = host_array(1) - host_array(0);
        for (size_t i = 1; i < size - 1; ++i) {
            double dtheta_i = host_array(i+1) - host_array(i);
            if (fabs(dtheta_i - dtheta) > 1e-2) {
                throw std::runtime_error("theta coordinate array is not uniformly spaced.");
            }
            if (verbose ) {
                std::cout << host_array(i) << " ";
            }
        }
        std::cout << std::endl;
    }
    else if (coord_type == "phi") {
        phi_min = host_array(0);
        phi_max = host_array(size - 1);
        nphi = size;
        // Check uniform spacing
        double dphi = host_array(1) - host_array(0);
        for (size_t i = 1; i < size - 1; ++i) {
            double dphi_i = host_array(i+1) - host_array(i);
            if (fabs(dphi_i - dphi) > 1e-2) {
                throw std::runtime_error("phi coordinate array is not uniformly spaced.");
            }
            if (verbose ) {
                std::cout << host_array(i) << " ";
            }
        }
        std::cout << std::endl;
    }
    return;
}

// Load all the arrays needed for HAMR simulation
// (This function can be expanded as needed to load specific arrays)
void load_hamr_numpy_arrays(const std::string& base_path, 
                            Kokkos::View<double*>& r,
                            Kokkos::View<double*>& theta,
                            Kokkos::View<double*>& phi, 
                            Kokkos::View<double***>& rho,
                            Kokkos::View<double***>& bsqr,
                            Kokkos::View<double***>& pgas,
                            Kokkos::View<double***>& Tgas,
                            Kokkos::View<double***>& ug,
                            Kokkos::View<double****>& bu,
                            Kokkos::View<double****>& uu) {
    load_coordinate_array<double>("r", base_path + "r.npy", r);
    load_coordinate_array<double>("theta", base_path + "theta.npy", theta);
    load_coordinate_array<double>("phi", base_path + "phi.npy", phi);
    load_3d_numpy_array<double>(base_path + "rho.npy", rho);
    load_3d_numpy_array<double>(base_path + "bsq.npy", bsqr);
    load_3d_numpy_array<double>(base_path + "pgas.npy", pgas);
    load_3d_numpy_array<double>(base_path + "Tgas.npy", Tgas);
    load_3d_numpy_array<double>(base_path + "ug.npy", ug);
    load_4d_numpy_array<double>(base_path + "bu.npy", bu);
    load_4d_numpy_array<double>(base_path + "uu.npy", uu);
    if (verbose) {
        std::cout << "All HAMR numpy arrays loaded from base path: " << base_path << std::endl;
    }
    return;
}
