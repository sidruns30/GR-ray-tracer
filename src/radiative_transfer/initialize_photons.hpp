/*
    Initialize photons for ray tracing simulation

    Populate the Kokkos view "photons" with initial positions and wavevectors
    based on a pinhole camera model.
    The campre
*/

# pragma once
#include "../utils.hpp"
#include "../utils.cpp"
#include <Kokkos_Random.hpp>



void InitializePhotonsPinhole(
    const int photons_per_process,
    const double camera_distance,
    const double camera_theta,
    const double camera_phi,
    const int mpi_rank,
    const int mpi_size,
    Kokkos::View<double**> photons)
{
    // camera location
    double cam_x = camera_distance * sin(camera_theta) * cos(camera_phi);
    double cam_y = camera_distance * sin(camera_theta) * sin(camera_phi);
    double cam_z = camera_distance * cos(camera_theta);

    // phi sector for this MPI rank
    double delta_phi = 2.0 * M_PI / mpi_size;
    double phi_min = mpi_rank * delta_phi;
    double phi_max = (mpi_rank + 1) * delta_phi;

    Kokkos::Random_XorShift64_Pool<> rand_pool(12345 + mpi_rank);

    Kokkos::parallel_for(
        "InitPhotons",
        Kokkos::RangePolicy<>(0, photons_per_process),
        KOKKOS_LAMBDA(int i) {

            auto rand_gen = rand_pool.get_state();

            double r   = rand_gen.drand(0.0, 50.0); // Sample radius in image plane
            double phi = rand_gen.drand(phi_min, phi_max);

            double kx = -cam_x + r * cos(phi);
            double ky = -cam_y + r * sin(phi);
            double kz = -cam_z;
            double k_norm = sqrt(kx * kx + ky * ky + kz * kz);

            rand_pool.free_state(rand_gen);

            photons(i, IT)  = 0.0;
            photons(i, IX)  = cam_x;
            photons(i, IY)  = cam_y;
            photons(i, IZ)  = cam_z;

            photons(i, IKT)  = 1.0;
            photons(i, IKX) = kx / k_norm;
            photons(i, IKY) = ky / k_norm;
            photons(i, IKZ) = kz / k_norm;

            photons(i, IW)  = 1.0;
        }
    );
    if (verbose && mpi_rank == 0) {
        // Print the information for the first photon
        auto photons_host = Kokkos::create_mirror_view(photons);
        Kokkos::deep_copy(photons_host, photons);
        std::cout << "First photon initialized at (t, x, y, z): ("
                  << photons_host(0, IT) << ", "
                  << photons_host(0, IX) << ", "
                  << photons_host(0, IY) << ", "
                  << photons_host(0, IZ) << ")" << std::endl;
        std::cout << "with wavevector (k_t, k_x, k_y, k_z): ("
                  << photons_host(0, IKT) << ", "
                  << photons_host(0, IKX) << ", "
                  << photons_host(0, IKY) << ", "
                  << photons_host(0, IKZ) << ")" << std::endl;

        std::cout << "Initialized " << photons_per_process << " photons per MPI process using pinhole camera." << std::endl;
    }
    return;
}

void InitializePhotonsImageCamera(
    const int photons_per_process,
    const double camera_distance,
    const double camera_theta,
    const double camera_phi,
    const int mpi_rank,
    const int mpi_size,
    Kokkos::View<double**> photons)
{
    // Construct a plane perpendicular to the camera direction at distance camera_distance
    // Sample photons uniformly across the plane area
    double plane_x_center = camera_distance * sin(plane_theta) * cos(plane_phi);
    double plane_y_center = camera_distance * sin(plane_theta) * sin(plane_phi);
    double plane_z_center = camera_distance * cos(plane_theta);

    double scale = sqrt(plane_area / (image_width * image_height));
    double plane_dim1 = scale * static_cast<double>(image_width);
    double plane_dim2 = scale * static_cast<double>(image_height);

    // Define basis vectors for the image plane
    double theta_hat[3] = {
        cos(plane_theta) * cos(plane_phi),
        cos(plane_theta) * sin(plane_phi),
        -sin(plane_theta)
        };
    double phi_hat[3] = {
        -sin(plane_phi),
        cos(plane_phi),
        0.0
        };

    Kokkos::Random_XorShift64_Pool<> rand_pool(12345 + mpi_rank);

    Kokkos::parallel_for(
        "InitPhotonsImageCamera",
        Kokkos::RangePolicy<>(0, photons_per_process),
        KOKKOS_LAMBDA(int i) {

            auto rand_gen = rand_pool.get_state();

            double theta_disp   = rand_gen.drand(-plane_dim1/2, plane_dim1/2); // Sample radius in image plane
            double phi_disp = rand_gen.drand(-plane_dim2/2, plane_dim2/2);

            rand_pool.free_state(rand_gen);

            double photon_x = plane_x_center + theta_disp * theta_hat[0] + phi_disp * phi_hat[0];
            double photon_y = plane_y_center + theta_disp * theta_hat[1] + phi_disp * phi_hat[1];
            double photon_z = plane_z_center + theta_disp * theta_hat[2] + phi_disp * phi_hat[2];

            // Compute the direction of the photon
            double kx = -plane_x_center;
            double ky = -plane_y_center;
            double kz = -plane_z_center;

            // Normalize the direction
            double k_norm = sqrt(kx * kx + ky * ky + kz * kz);
            kx /= k_norm;
            ky /= k_norm;
            kz /= k_norm;

            // Set the photon properties
            photons(i, IT)  = 0.0;
            photons(i, IX)  = photon_x;
            photons(i, IY)  = photon_y;
            photons(i, IZ)  = photon_z;

            photons(i, IKT)  = 1.0;
            photons(i, IKX) = kx;
            photons(i, IKY) = ky;
            photons(i, IKZ) = kz;

            photons(i, IW)  = 1.0;
        }
    );

    if (verbose && mpi_rank == 0) {
        // Print the information for the first photon
        auto photons_host = Kokkos::create_mirror_view(photons);
        Kokkos::deep_copy(photons_host, photons);
        std::cout << "First photon initialized at (t, x, y, z): ("
                  << photons_host(0, IT) << ", "
                  << photons_host(0, IX) << ", "
                  << photons_host(0, IY) << ", "
                  << photons_host(0, IZ) << ")" << std::endl;
        std::cout << "with wavevector (k_t, k_x, k_y, k_z): ("
                  << photons_host(0, IKT) << ", "
                  << photons_host(0, IKX) << ", "
                  << photons_host(0, IKY) << ", "
                  << photons_host(0, IKZ) << ")" << std::endl;

        std::cout << "Initialized " << photons_per_process << " photons per MPI process using pinhole camera." << std::endl;
    }

    return;
}