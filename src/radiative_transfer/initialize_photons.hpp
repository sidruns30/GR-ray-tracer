// Photon initialization for the supported camera models.

#pragma once

#include <Kokkos_Random.hpp>

#include "../utils.hpp"
#include "../metrics/cartesian_kerr_schild.hpp"
#include "../metrics/kerr_schild_core.hpp"
#include "fluid_frame.hpp"
inline void initialize_photons_pinhole(
    const int photons_per_process,
    const real camera_distance,
    const real camera_theta,
    const real camera_phi,
    const int mpi_rank,
    const int mpi_size,
    const std::uint32_t global_id_offset,
    const PhotonGenerationConfig& generation,
    Photons &photons)
{
    (void)mpi_size;
    const real cam_x = camera_distance * sin(camera_theta) * cos(camera_phi);
    const real cam_y = camera_distance * sin(camera_theta) * sin(camera_phi);
    const real cam_z = camera_distance * cos(camera_theta);
    // Local captures of runtime-configurable globals: device code can't read
    // `extern real` globals directly (see kerr_schild_core.hpp), but the
    // KOKKOS_LAMBDA below captures locals in this scope by value, which works.
    const real a_BH_ = a_BH;
    const real M_BH_ = M_BH;
    const real dlambda_init = dlambda;
    const real pinhole_aperture_radius_ = pinhole_aperture_radius;
    const real frequency_hz = generation.camera_frequency_hz;
    const real packet_energy_erg = generation.camera_packet_energy_erg;
    Kokkos::Random_XorShift64_Pool<> rand_pool(12345 + mpi_rank);
    Kokkos::parallel_for(
        "InitPhotons",
        Kokkos::RangePolicy<>(0, photons_per_process),
        KOKKOS_LAMBDA(int i) {
            photons.id(i) = global_id_offset + static_cast<std::uint32_t>(i);
            photons.frequency(i) = frequency_hz;
            auto rand_gen = rand_pool.get_state();
            const real radius = rand_gen.drand(0.0, pinhole_aperture_radius_);
            const real azimuth = rand_gen.drand(0, 2.0 * PI);
            const real screen_u = radius * cos(azimuth);
            const real screen_v = radius * sin(azimuth);
            const real kx = -cam_x + screen_u;
            const real ky = -cam_y + screen_v;
            const real kz = -cam_z;
            const real k_norm = sqrt(kx * kx + ky * ky + kz * kz);
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
            photons.I(i)  = packet_energy_erg;
            photons.Q(i)  = 0.0;
            photons.U(i)  = 0.0;
            photons.V(i)  = 0.0;
            photons.dlambda(i) = dlambda_init * camera_distance;
            photons.terminate(i)  = false;
        }
    );
}

inline void initialize_photons_image_camera(
    const int photons_per_process,
    const real camera_distance,
    const real camera_theta,
    const real camera_phi,
    const int mpi_rank,
    const int mpi_size,
    const std::uint32_t global_id_offset,
    const PhotonGenerationConfig& generation,
    Photons &photons)
{
    (void)camera_theta;
    (void)camera_phi;
    (void)mpi_size;
    // Construct a plane perpendicular to the camera direction at distance camera_distance
    // Sample photons uniformly across the plane area
    const real plane_x_center = camera_distance * sin(plane_theta) * cos(plane_phi);
    const real plane_y_center = camera_distance * sin(plane_theta) * sin(plane_phi);
    const real plane_z_center = camera_distance * cos(plane_theta);
    // Orthonormal tangent basis at the camera's spherical position.
    const real theta_hat[3] = {
        cos(plane_theta) * cos(plane_phi),
        cos(plane_theta) * sin(plane_phi),
        -sin(plane_theta)
        };
    const real phi_hat[3] = {
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
    const real frequency_hz = generation.camera_frequency_hz;
    const real packet_energy_erg = generation.camera_packet_energy_erg;
    Kokkos::Random_XorShift64_Pool<> rand_pool(12345 + mpi_rank);
    Kokkos::parallel_for(
        "InitPhotonsImageCamera",
        Kokkos::RangePolicy<>(0, photons_per_process),
        KOKKOS_LAMBDA(int i) {
            photons.id(i) = global_id_offset + static_cast<std::uint32_t>(i);
            photons.frequency(i) = frequency_hz;
            auto rand_gen = rand_pool.get_state();
            const real screen_u = rand_gen.drand(-plane_dim1_/2, plane_dim1_/2);
            const real screen_v = rand_gen.drand(-plane_dim2_/2, plane_dim2_/2);
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
            photons.I(i)  = packet_energy_erg;
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
}

// Custom photon initialization
inline void initialize_photons_user(
    const int photons_per_process,
    const int mpi_rank,
    const std::uint32_t global_id_offset,
    const Kokkos::View<real*>& r,
    const Kokkos::View<real*>& theta,
    const Kokkos::View<real*>& phi,
    const Kokkos::View<real***>& density,
    const Kokkos::View<real***>& temperature,
    const Kokkos::View<real****>& velocity,
    const Kokkos::View<real****>& magnetic,
    const PhotonGenerationConfig& generation,
    const UnitConversions& units,
    Photons &photons)
{
    const std::size_t ntheta_local = theta.extent(0);
    const std::size_t nphi_local = phi.extent(0);
    const int packets_per_cell = generation.superphotons_per_cell;
    const PhotonGeneratorType generator_type = generation.generator;
    const real energy_per_cell = generation.energy_per_cell_erg;
    const real power_law_slope = generation.power_law_slope;
    const real nu_min = generation.nu_min_hz;
    const real nu_max = generation.nu_max_hz;
    const real custom_frequency = generation.custom_frequency_hz;
    const real density_scale = units.density_g_cm3_per_code;
    const real temperature_scale = units.temperature_k_per_code;
    const real velocity_scale = units.four_velocity_cm_s_per_code;
    const real magnetic_scale = units.magnetic_gauss_per_code;
    const real length_scale = units.length_cm_per_code;
    const real time_scale = units.time_s_per_code;
    const real spin = a_BH;
    const real mass = M_BH;
    const real step_size = dlambda;

    Kokkos::Random_XorShift64_Pool<> rand_pool(12345 + mpi_rank);
    Kokkos::parallel_for(
        "GenerateFluidFrameSuperphotons",
        Kokkos::RangePolicy<>(0, photons_per_process),
        KOKKOS_LAMBDA(int i) {
            auto rand_gen = rand_pool.get_state();
            const std::size_t cell = static_cast<std::size_t>(i / packets_per_cell);
            const std::size_t cell_i = cell / (ntheta_local * nphi_local);
            const std::size_t remainder = cell % (ntheta_local * nphi_local);
            const std::size_t cell_j = remainder / nphi_local;
            const std::size_t cell_k = remainder % nphi_local;

            const real radius = r(cell_i);
            const real polar = theta(cell_j);
            const real azimuth = phi(cell_k);
            real position[4];
            spherical_kerr_schild_to_cartesian(radius, polar, azimuth, spin, position);

            real spherical_velocity[4];
            real spherical_magnetic[4];
            FluidCellCGS fluid;
            for (int component = 0; component < 3; ++component) {
                fluid.position_cm[component] = position[component + 1] * length_scale;
            }
            fluid.coordinate_time_s = position[0] * time_scale;
            fluid.density_g_cm3 = density(cell_i, cell_j, cell_k) * density_scale;
            fluid.temperature_k = temperature(cell_i, cell_j, cell_k) * temperature_scale;
            for (int component = 0; component < 4; ++component) {
                spherical_velocity[component] = velocity(cell_i, cell_j, cell_k, component);
                spherical_magnetic[component] = magnetic(cell_i, cell_j, cell_k, component);
            }
            transform_spherical_four_vector_to_cartesian(
                radius, polar, azimuth, spin, spherical_velocity,
                fluid.coordinate_four_velocity_cm_s);
            for (int component = 0; component < 4; ++component) {
                fluid.coordinate_four_velocity_cm_s[component] *= velocity_scale;
            }
            real cartesian_magnetic[4];
            transform_spherical_four_vector_to_cartesian(
                radius, polar, azimuth, spin, spherical_magnetic, cartesian_magnetic);
            for (int component = 0; component < 4; ++component) {
                cartesian_magnetic[component] *= magnetic_scale;
            }

            real tetrad[4][4];
            const bool valid_tetrad = build_fluid_tetrad(
                position, fluid.coordinate_four_velocity_cm_s, spin, mass, tetrad);
            if (valid_tetrad) {
                real metric[4][4];
                kerr_schild::compute_metric(position, spin, mass, metric);
                fluid.magnetic_gauss[0] = -metric_inner_product(
                    metric, tetrad[0], cartesian_magnetic);
                for (int axis = 1; axis < 4; ++axis) {
                    fluid.magnetic_gauss[axis] = metric_inner_product(
                        metric, tetrad[axis], cartesian_magnetic);
                }
            }
            real sampled_frequency = 0.0;
            real packet_energy = 0.0;
            bool generated = false;
            if (valid_tetrad && generator_type == PhotonGeneratorType::Blackbody) {
                generated = GenerateBlackbody(
                    fluid, rand_gen, packets_per_cell, energy_per_cell,
                    sampled_frequency, packet_energy);
            } else if (valid_tetrad && generator_type == PhotonGeneratorType::PowerLaw) {
                generated = GeneratePowerLaw(
                    fluid, rand_gen, packets_per_cell, power_law_slope,
                    nu_min, nu_max, energy_per_cell, sampled_frequency, packet_energy);
            } else if (valid_tetrad) {
                generated = GeneratePhotonsCustom(
                    fluid, rand_gen, packets_per_cell, custom_frequency,
                    energy_per_cell, sampled_frequency, packet_energy);
            }

            const real mu = 2.0 * rand_gen.drand() - 1.0;
            const real direction_phi = 2.0 * PI * rand_gen.drand();
            const real transverse = Kokkos::sqrt(Kokkos::fmax(real(0.0), 1.0 - mu * mu));
            const real direction_x = transverse * Kokkos::cos(direction_phi);
            const real direction_y = transverse * Kokkos::sin(direction_phi);
            const real direction_z = mu;
            real p_cov[4] = {0.0, 0.0, 0.0, 0.0};
            if (generated) {
                fluid_frame_direction_to_covariant_momentum(
                    position, tetrad, direction_x, direction_y, direction_z, spin, mass, p_cov);
            }
            rand_pool.free_state(rand_gen);

            photons.id(i) = global_id_offset + static_cast<std::uint32_t>(i);
            photons.frequency(i) = sampled_frequency;
            photons.x0(i) = position[0];
            photons.x1(i) = position[1];
            photons.x2(i) = position[2];
            photons.x3(i) = position[3];
            photons.k0(i) = p_cov[0];
            photons.k1(i) = p_cov[1];
            photons.k2(i) = p_cov[2];
            photons.k3(i) = p_cov[3];
            photons.I(i) = generated ? packet_energy : 0.0;
            photons.Q(i)  = 0.0;
            photons.U(i)  = 0.0;
            photons.V(i)  = 0.0;
            photons.dlambda(i) = step_size;
            photons.terminate(i) = !generated;
            photons.theta_disp(i) = 0.0;
            photons.phi_disp(i) = 0.0;
        }
    );
}
