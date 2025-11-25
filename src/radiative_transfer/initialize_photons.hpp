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
#include "../metrics/cartesian_kerr_schild.hpp"



void initialize_photons_pinhole(
    const int photons_per_process,
    const real camera_distance,
    const real camera_theta,
    const real camera_phi,
    const int mpi_rank,
    const int mpi_size,
    Photons &photons)
{
    // camera location
    real cam_x = camera_distance * sin(camera_theta) * cos(camera_phi);
    real cam_y = camera_distance * sin(camera_theta) * sin(camera_phi);
    real cam_z = camera_distance * cos(camera_theta);
    Kokkos::Random_XorShift64_Pool<> rand_pool(12345 + mpi_rank);
    Kokkos::parallel_for(
        "InitPhotons",
        Kokkos::RangePolicy<>(0, photons_per_process),
        KOKKOS_LAMBDA(int i) {
            auto rand_gen = rand_pool.get_state();
            real r   = rand_gen.drand(0.0, 50.0);
            real phi = rand_gen.drand(0, 2.0 * PI);
            real kx = -cam_x + r * cos(phi);
            real ky = -cam_y + r * sin(phi);
            real kz = -cam_z;
            real k_norm = sqrt(kx * kx + ky * ky + kz * kz);
            rand_pool.free_state(rand_gen);
            photons.x0(i)  = 0.0;
            photons.x1(i)  = cam_x;
            photons.x2(i)  = cam_y;
            photons.x3(i)  = cam_z;
            photons.k0(i)  = 1.0;
            photons.k1(i)  = kx / k_norm;
            photons.k2(i)  = ky / k_norm;
            photons.k3(i)  = kz / k_norm;
            photons.I(i)  = 0.0;
            photons.dlambda(i) = dlambda * camera_distance;
            photons.terminate(i)  = false;
        }
    );
    // Siddhant: Hopefully host views are not the memory bottleneck here
    photons.create_mirror_views();
    photons.copy_to_host();
    compute_cartesian_kerr_schild_norms(photons.x0, photons.x1, photons.x2, photons.x3,
                                    photons.k0, photons.k1, photons.k2, photons.k3,
                                    photons.I);
    return;
}

void initialize_photons_image_camera(
    const int photons_per_process,
    const real camera_distance,
    const real camera_theta,
    const real camera_phi,
    const int mpi_rank,
    const int mpi_size,
    Photons &photons)
{
    // Construct a plane perpendicular to the camera direction at distance camera_distance
    // Sample photons uniformly across the plane area
    real plane_x_center = camera_distance * sin(plane_theta) * cos(plane_phi);
    real plane_y_center = camera_distance * sin(plane_theta) * sin(plane_phi);
    real plane_z_center = camera_distance * cos(plane_theta);
    // Define basis vectors for the image plane
    // Siddhant: I think these are legit but lmk if someone has a better idea for bases
    real theta_hat[3] = {
        cos(plane_theta) * cos(plane_phi),
        cos(plane_theta) * sin(plane_phi),
        -sin(plane_theta)
        };
    real phi_hat[3] = {
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
            real theta_disp   = rand_gen.drand(-plane_dim1/2, plane_dim1/2); // Sample radius in image plane
            real phi_disp = rand_gen.drand(-plane_dim2/2, plane_dim2/2);
            rand_pool.free_state(rand_gen);
            real photon_x = plane_x_center + theta_disp * theta_hat[0] + phi_disp * phi_hat[0];
            real photon_y = plane_y_center + theta_disp * theta_hat[1] + phi_disp * phi_hat[1];
            real photon_z = plane_z_center + theta_disp * theta_hat[2] + phi_disp * phi_hat[2];
            real kx = -plane_x_center;
            real ky = -plane_y_center;
            real kz = -plane_z_center;
            real k_norm = sqrt(kx * kx + ky * ky + kz * kz);
            kx /= k_norm;
            ky /= k_norm;
            kz /= k_norm;
            // Set the photon properties
            photons.x0(i)  = 0.0;
            photons.x1(i)  = photon_x;
            photons.x2(i)  = photon_y;
            photons.x3(i)  = photon_z;
            photons.k0(i)  = 1.0;
            photons.k1(i)  = kx;
            photons.k2(i)  = ky;
            photons.k3(i)  = kz;
            photons.I(i)  = 0.0;
            photons.dlambda(i) = dlambda * camera_distance;
            photons.terminate(i)  = false;
        }
    );
    photons.create_mirror_views();
    photons.copy_to_host();
    compute_cartesian_kerr_schild_norms(photons.x0, photons.x1, photons.x2, photons.x3,
                                    photons.k0, photons.k1, photons.k2, photons.k3,
                                    photons.I);
    return;
}

// Custom photon initialization
void initialize_photons_user(
    const int photons_per_process,
    Photons &photons)
{

    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    // User-defined initialization logic goes here
    Kokkos::Random_XorShift64_Pool<> rand_pool(12345 + mpi_rank);
    Kokkos::parallel_for(
        "InitPhotonsUser",
        Kokkos::RangePolicy<>(0, photons_per_process),
        KOKKOS_LAMBDA(int i) {
            auto rand_gen = rand_pool.get_state();
            // Sample photon properties
            photons.x0(i)  = 0.0;
            photons.x1(i)  = rand_gen.drand(1.0, 3.0);
            photons.x2(i)  = rand_gen.drand(1.0, 3.0);
            photons.x3(i)  = rand_gen.drand(1.0, 3.0);
            auto x         = photons.x1(i);
            auto y         = photons.x2(i);
            auto z         = photons.x3(i);
            auto RSQ       = SQR(x) + SQR(y) + SQR(z);
            auto r         = sqrt(0.5 * (RSQ - SQR(a_BH) + sqrt(SQR(RSQ - SQR(a_BH)) + 4.0 * SQR(a_BH * z))));
            photons.k0(i)  = 0.0;
            photons.k1(i)  = 0.0;
            photons.k2(i)  = 0.0;
            photons.k3(i)  = 0.0;
            photons.I(i)  = 0.0;
            photons.dlambda(i) = 1e-2;
            photons.terminate(i)  = false;
        }
    );
    photons.create_mirror_views();
    photons.copy_to_host();
    compute_cartesian_kerr_schild_norms(photons.x0, photons.x1, photons.x2, photons.x3,
                                    photons.k0, photons.k1, photons.k2, photons.k3,
                                    photons.I);
    return;
}
