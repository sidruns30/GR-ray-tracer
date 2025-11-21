#include "utils.hpp"
#include "utils.cpp"
#include "input/load_python_arrays.hpp"
#include "radiative_transfer/initialize_photons.hpp"
#include "geodesic_integrate/cartesian_kerr_schild.hpp"


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
        // Load the HAMR data from numpy files into Kokkos views
        Kokkos::View<double*> r, theta, phi;
        Kokkos::View<double***>  rho, bsqr, pgas, Tgas, ug;
        Kokkos::View<double****> bu, uu;
        std::string base_path = "/home/siddhant/scratch/rayTracingTestData/";
        load_hamr_numpy_arrays(base_path, r, theta, phi, rho, bsqr, pgas, Tgas, ug, bu, uu);

        // Initialize camera and pixels
        if (verbose && mpi_rank == 0) {
            std::cout << "Grid dimensions: nr=" << nr << ", ntheta=" << ntheta << ", nphi=" << nphi << std::endl;
            std::cout << "r range: [" << r_min << ", " << r_max << "], theta range: [" 
                      << theta_min << ", " << theta_max << "], phi range: [" 
                      << phi_min << ", " << phi_max << "]" << std::endl;
        }
        // Initialize a fraction of photons per mpi process for back tracing from camera
        const int photons_per_process = nphotons / mpi_size;
        Kokkos::View<double**> photons("camera_pixels", photons_per_process, 9);

        if (use_pinhole_camera) {
            InitializePhotonsPinhole(
                photons_per_process,
                camera_distance,
                camera_theta,
                camera_phi,
                mpi_rank,
                mpi_size,
                photons
                );
            }

        else if (use_image_camera) {
            InitializePhotonsImageCamera(
                photons_per_process,
                camera_distance,
                camera_theta,
                camera_phi,
                mpi_rank,
                mpi_size,
                photons
                );
        }

        std::cout << "Starting geodesic integration on rank " << mpi_rank << " with " 
                  << photons_per_process << " photons." << std::endl;

        // Integrate geodesics
        integrate_geodesics(
            photons,
            r, theta, phi, rho, bsqr, pgas, Tgas, bu, uu,
            dtau, mpi_rank
            );
    }
    Kokkos::fence();
    // Finalize Kokkos
    Kokkos::finalize();
    
    // Finalize MPI
    MPI_Finalize();
    std::cout << "MPI finalize called from rank " << mpi_rank << std::endl;
    return 0;
}

