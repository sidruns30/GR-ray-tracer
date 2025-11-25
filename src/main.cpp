#include "utils.hpp"
#include "utils.cpp"
#include "input/load_python_arrays.hpp"
#include "radiative_transfer/initialize_photons.hpp"
#include "geodesic_integrate/integrate_cart_ks.hpp"
#include "output/display.hpp"


int main(int argc, char* argv[]) {
    // Initialize MPI
    int mpi_rank, mpi_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    
    if (mpi_rank == 0) {
        INIT("=== GR Ray-Trace Initialization ===");
        INFO("MPI processes: " + std::to_string(mpi_size));
    }
    // Initialize Kokkos
    Kokkos::initialize(argc, argv);
    {
        Timers timers(max_steps, output_interval);

        timers.AddTimer("Load HAMR Data");
        timers.BeginTimer("Load HAMR Data");
        // Load the HAMR data from numpy files into Kokkos views
        Kokkos::View<real*> r, theta, phi;
        Kokkos::View<real***>  rho, bsqr, pgas, Tgas, ug;
        Kokkos::View<real****> bu, uu;
        std::string base_path = "/home/siddhant/scratch/rayTracingTestData/";
        load_hamr_numpy_arrays(base_path, r, theta, phi, rho, bsqr, pgas, Tgas, ug, bu, uu);
        timers.EndTimer("Load HAMR Data");

        // Initialize camera and pixels
        if (verbose && mpi_rank == 0) {
            INFO("Grid dimensions: nr=" + std::to_string(nr) + ", ntheta=" + std::to_string(ntheta) + ", nphi=" + 
            std::to_string(nphi), Colors::cyan);
            INFO("r coordinate from " + std::to_string(r_min) + " to " + std::to_string(r_max), Colors::hotpink);
            INFO("theta coordinate from " + std::to_string(theta_min) + " to " + std::to_string(theta_max), Colors::hotpink);
            INFO("phi coordinate from " + std::to_string(phi_min) + " to " + std::to_string(phi_max), Colors::hotpink);
        }
        // Initialize a fraction of photons per mpi process for back tracing from camera
        const int photons_per_process = nphotons / mpi_size;
        Photons photons(photons_per_process, "photons_");

        if (use_pinhole_camera) {
            timers.AddTimer("Initialize Pinhole Camera");
            timers.BeginTimer("Initialize Pinhole Camera");
            initialize_photons_pinhole(
                photons_per_process,
                camera_distance,
                camera_theta,
                camera_phi,
                mpi_rank,
                mpi_size,
                photons
                );
            timers.EndTimer("Initialize Pinhole Camera");
            }

        else if (use_image_camera) {
            timers.AddTimer("Initialize Image Camera");
            timers.BeginTimer("Initialize Image Camera");
            initialize_photons_image_camera(
                photons_per_process,
                camera_distance,
                camera_theta,
                camera_phi,
                mpi_rank,
                mpi_size,
                photons
                );
            timers.EndTimer("Initialize Image Camera");
        }

        else 
        {
            timers.AddTimer("Initialize User-Defined Photons");
            timers.BeginTimer("Initialize User-Defined Photons");
            initialize_photons_user(
                photons_per_process,
                photons
                );
            timers.EndTimer("Initialize User-Defined Photons");
        }

        INFO("=== Starting Geodesic Integration ===", Colors::cyan);

        // Integrate geodesics
        integrate_geodesics(
            photons,
            r, theta, phi, rho, Tgas, 
            mpi_rank, timers
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

