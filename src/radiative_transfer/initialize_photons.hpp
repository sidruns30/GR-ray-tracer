/*
    Initialize photons for ray tracing simulation

    Populate the Kokkos view "photons" with initial positions and wavevectors
    based on a pinhole camera model.
    The campre
*/

#pragma once

#include <Kokkos_Random.hpp>

#include "../utils.hpp"
#include "../metrics/cartesian_kerr_schild.hpp"
#include "../metrics/kerr_schild_core.hpp"



inline void initialize_photons_pinhole(
    const int photons_per_process,
    const real camera_distance,
    const real camera_theta,
    const real camera_phi,
    const int mpi_rank,
    const int mpi_size,
    Photons &photons)
{
    (void)mpi_size;
    // camera location
    real cam_x = camera_distance * sin(camera_theta) * cos(camera_phi);
    real cam_y = camera_distance * sin(camera_theta) * sin(camera_phi);
    real cam_z = camera_distance * cos(camera_theta);
    // Local captures of runtime-configurable globals: device code can't read
    // `extern real` globals directly (see kerr_schild_core.hpp), but the
    // KOKKOS_LAMBDA below captures locals in this scope by value, which works.
    const real a_BH_ = a_BH;
    const real M_BH_ = M_BH;
    const real dlambda_init = dlambda;
    Kokkos::Random_XorShift64_Pool<> rand_pool(12345 + mpi_rank);
    Kokkos::parallel_for(
        "InitPhotons",
        Kokkos::RangePolicy<>(0, photons_per_process),
        KOKKOS_LAMBDA(int i) {
            auto rand_gen = rand_pool.get_state();
            real r   = rand_gen.drand(0.0, pinhole_aperture_radius);
            real phi = rand_gen.drand(0, 2.0 * PI);
            real screen_u = r * cos(phi);
            real screen_v = r * sin(phi);
            real kx = -cam_x + screen_u;
            real ky = -cam_y + screen_v;
            real kz = -cam_z;
            real k_norm = sqrt(kx * kx + ky * ky + kz * kz);
            rand_pool.free_state(rand_gen);
            photons.x0(i)  = 0.0;
            photons.x1(i)  = cam_x;
            photons.x2(i)  = cam_y;
            photons.x3(i)  = cam_z;
            // Screen-space impact parameters this photon's direction was sampled
            // at, in the pinhole aperture plane -- fixed for the photon's lifetime,
            // used by observation.hpp to bin it onto a stable camera-screen image.
            photons.theta_disp(i) = screen_u;
            photons.phi_disp(i) = screen_v;
            // Build a genuinely null contravariant wavevector at (0,cam_x,cam_y,cam_z)
            // with this spatial direction, then lower it to the covariant p_mu that
            // compute_rhs actually propagates (see kerr_schild_core.hpp).
            const real X[4] = {0.0, cam_x, cam_y, cam_z};
            const real K_spatial[3] = {kx / k_norm, ky / k_norm, kz / k_norm};
            real p_cov[4];
            kerr_schild::null_covariant_momentum_from_spatial_direction(X, K_spatial, a_BH_, M_BH_, p_cov);
            photons.k0(i)  = p_cov[0];
            photons.k1(i)  = p_cov[1];
            photons.k2(i)  = p_cov[2];
            photons.k3(i)  = p_cov[3];
            photons.I(i)  = 1.0;
            photons.Q(i)  = 0.0;
            photons.U(i)  = 0.0;
            photons.V(i)  = 0.0;
            photons.dlambda(i) = dlambda_init * camera_distance;
            photons.terminate(i)  = false;
        }
    );
    // Siddhant: Hopefully host views are not the memory bottleneck here
    photons.create_mirror_views();
    photons.copy_to_host();
    return;
}

inline void initialize_photons_image_camera(
    const int photons_per_process,
    const real camera_distance,
    const real camera_theta,
    const real camera_phi,
    const int mpi_rank,
    const int mpi_size,
    Photons &photons)
{
    (void)camera_theta;
    (void)camera_phi;
    (void)mpi_size;
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
    // Local captures of runtime-configurable globals -- see the comment in
    // initialize_photons_pinhole above.
    const real a_BH_ = a_BH;
    const real M_BH_ = M_BH;
    const real dlambda_init = dlambda;
    const real plane_dim1_ = plane_dim1;
    const real plane_dim2_ = plane_dim2;
    Kokkos::Random_XorShift64_Pool<> rand_pool(12345 + mpi_rank);
    Kokkos::parallel_for(
        "InitPhotonsImageCamera",
        Kokkos::RangePolicy<>(0, photons_per_process),
        KOKKOS_LAMBDA(int i) {
            auto rand_gen = rand_pool.get_state();
            real screen_u = rand_gen.drand(-plane_dim1_/2, plane_dim1_/2); // Sample displacement in image plane
            real screen_v = rand_gen.drand(-plane_dim2_/2, plane_dim2_/2);
            rand_pool.free_state(rand_gen);
            real photon_x = plane_x_center + screen_u * theta_hat[0] + screen_v * phi_hat[0];
            real photon_y = plane_y_center + screen_u * theta_hat[1] + screen_v * phi_hat[1];
            real photon_z = plane_z_center + screen_u * theta_hat[2] + screen_v * phi_hat[2];
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
            // Build a genuinely null contravariant wavevector at (0,photon_x,photon_y,photon_z)
            // with this spatial direction, then lower it to the covariant p_mu that
            // compute_rhs actually propagates (see kerr_schild_core.hpp).
            const real X[4] = {0.0, photon_x, photon_y, photon_z};
            const real K_spatial[3] = {kx, ky, kz};
            real p_cov[4];
            kerr_schild::null_covariant_momentum_from_spatial_direction(X, K_spatial, a_BH_, M_BH_, p_cov);
            photons.k0(i)  = p_cov[0];
            photons.k1(i)  = p_cov[1];
            photons.k2(i)  = p_cov[2];
            photons.k3(i)  = p_cov[3];
            photons.I(i)  = 1.0;
            photons.Q(i)  = 0.0;
            photons.U(i)  = 0.0;
            photons.V(i)  = 0.0;
            photons.dlambda(i) = dlambda_init * camera_distance;
            photons.terminate(i)  = false;
            // Screen-space displacement this photon's pixel was sampled at in the
            // image plane -- fixed for the photon's lifetime, used by
            // observation.hpp to bin it onto a stable camera-screen image.
            photons.theta_disp(i) = screen_u;
            photons.phi_disp(i) = screen_v;
        }
    );
    photons.create_mirror_views();
    photons.copy_to_host();
    return;
}

// Custom photon initialization
inline void initialize_photons_user(
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
            photons.k0(i)  = 0.0;
            photons.k1(i)  = 0.0;
            photons.k2(i)  = 0.0;
            photons.k3(i)  = 0.0;
            photons.I(i)  = 1.0;
            photons.Q(i)  = 0.0;
            photons.U(i)  = 0.0;
            photons.V(i)  = 0.0;
            photons.dlambda(i) = 1e-2;
            photons.terminate(i)  = false;
        }
    );
    photons.create_mirror_views();
    photons.copy_to_host();
    return;
}
