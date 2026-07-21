#pragma once

#include <cstdint>

#include <Kokkos_Random.hpp>

#include "../utils.hpp"
#include "fluid_frame.hpp"

struct ScatteringModel {
    bool enabled = false;
    real optical_depth = 0.0;
    real albedo = 1.0;
    std::uint64_t seed = 0;
};

using ScatteringRandomPool =
    Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>;

// Views and conversion factors needed to sample a rank-local fluid cell.
// `available` is set on the host only when every required field exists, so
// vacuum integrations never enter the scattering path on the device.
struct ScatteringFluidGrid {
    Kokkos::View<real*> r;
    Kokkos::View<real*> theta;
    Kokkos::View<real*> phi;
    Kokkos::View<real***> density;
    Kokkos::View<real***> temperature;
    Kokkos::View<real****> velocity;
    Kokkos::View<real****> magnetic;
    real radial_log_spacing = 0.0;
    real polar_spacing = 0.0;
    real azimuth_spacing = 0.0;
    real length_scale = 1.0;
    real time_scale = 1.0;
    real density_scale = 1.0;
    real temperature_scale = 1.0;
    real velocity_scale = 1.0;
    real magnetic_scale = 1.0;
    real spin = 0.0;
    real mass = 1.0;
    bool available = false;
};

template <typename ViewType>
KOKKOS_INLINE_FUNCTION
int nearest_linear_coordinate(const ViewType& coordinates, real value, real spacing) {
    if (coordinates.extent(0) == 0 || !(spacing > 0.0)) return -1;
    int nearest = 0;
    real best_distance = Kokkos::abs(value - coordinates(0));
    for (std::size_t i = 1; i < coordinates.extent(0); ++i) {
        const real distance = Kokkos::abs(value - coordinates(i));
        if (distance < best_distance) {
            nearest = static_cast<int>(i);
            best_distance = distance;
        }
    }
    return best_distance <= 0.500001 * spacing ? nearest : -1;
}

template <typename ViewType>
KOKKOS_INLINE_FUNCTION
int nearest_radial_coordinate(const ViewType& coordinates, real radius,
                              real logarithmic_spacing) {
    if (coordinates.extent(0) == 0 || !(radius > 0.0) ||
        !(logarithmic_spacing > 0.0)) return -1;
    int nearest = 0;
    real best_distance = Kokkos::abs(Kokkos::log(radius / coordinates(0)));
    for (std::size_t i = 1; i < coordinates.extent(0); ++i) {
        const real distance = Kokkos::abs(Kokkos::log(radius / coordinates(i)));
        if (distance < best_distance) {
            nearest = static_cast<int>(i);
            best_distance = distance;
        }
    }
    return best_distance <= 0.500001 * logarithmic_spacing ? nearest : -1;
}

template <typename ViewType>
KOKKOS_INLINE_FUNCTION
int nearest_azimuth_coordinate(const ViewType& coordinates, real azimuth,
                               real spacing) {
    if (coordinates.extent(0) == 0 || !(spacing > 0.0)) return -1;
    int nearest = 0;
    real best_distance = 2.0 * PI;
    for (std::size_t i = 0; i < coordinates.extent(0); ++i) {
        real distance = Kokkos::abs(azimuth - coordinates(i));
        distance = Kokkos::fmod(distance, 2.0 * PI);
        distance = Kokkos::fmin(distance, 2.0 * PI - distance);
        if (distance < best_distance) {
            nearest = static_cast<int>(i);
            best_distance = distance;
        }
    }
    return best_distance <= 0.500001 * spacing ? nearest : -1;
}

// Samples the nearest local cell and constructs its orthonormal fluid tetrad
// at the photon's current position. Returning false means this rank has no
// applicable medium at that position, so scattering must not be called.
KOKKOS_INLINE_FUNCTION
bool sample_scattering_fluid(const real state[8], const ScatteringFluidGrid& grid,
                             FluidCellCGS& fluid, real tetrad[4][4]) {
    if (!grid.available) return false;

    const real radius = kerr_schild::compute_r(
        state[IX], state[IY], state[IZ], grid.spin);
    if (!(radius > 0.0) || !Kokkos::isfinite(radius)) return false;
    const real cosine_theta = Kokkos::fmax(
        real(-1.0), Kokkos::fmin(real(1.0), state[IZ] / radius));
    const real polar = Kokkos::acos(cosine_theta);
    real azimuth = Kokkos::atan2(
        radius * state[IY] - grid.spin * state[IX],
        radius * state[IX] + grid.spin * state[IY]);
    if (azimuth < 0.0) azimuth += 2.0 * PI;

    const int i = nearest_radial_coordinate(grid.r, radius, grid.radial_log_spacing);
    const int j = nearest_linear_coordinate(grid.theta, polar, grid.polar_spacing);
    const int k = nearest_azimuth_coordinate(grid.phi, azimuth, grid.azimuth_spacing);
    if (i < 0 || j < 0 || k < 0) return false;

    fluid.position_cm[0] = state[IX] * grid.length_scale;
    fluid.position_cm[1] = state[IY] * grid.length_scale;
    fluid.position_cm[2] = state[IZ] * grid.length_scale;
    fluid.coordinate_time_s = state[IT] * grid.time_scale;
    fluid.density_g_cm3 = grid.density(i, j, k) * grid.density_scale;
    fluid.temperature_k = grid.temperature(i, j, k) * grid.temperature_scale;
    if (!(fluid.density_g_cm3 > 0.0) || !Kokkos::isfinite(fluid.density_g_cm3) ||
        !Kokkos::isfinite(fluid.temperature_k)) return false;

    real spherical_velocity[4];
    real spherical_magnetic[4];
    for (int component = 0; component < 4; ++component) {
        spherical_velocity[component] = grid.velocity(i, j, k, component);
        spherical_magnetic[component] = grid.magnetic(i, j, k, component);
    }
    transform_spherical_four_vector_to_cartesian(
        grid.r(i), grid.theta(j), grid.phi(k), grid.spin,
        spherical_velocity, fluid.coordinate_four_velocity_cm_s);
    for (int component = 0; component < 4; ++component) {
        fluid.coordinate_four_velocity_cm_s[component] *= grid.velocity_scale;
    }

    const real position[4] = {state[IT], state[IX], state[IY], state[IZ]};
    if (!build_fluid_tetrad(
            position, fluid.coordinate_four_velocity_cm_s,
            grid.spin, grid.mass, tetrad)) return false;

    real cartesian_magnetic[4];
    transform_spherical_four_vector_to_cartesian(
        grid.r(i), grid.theta(j), grid.phi(k), grid.spin,
        spherical_magnetic, cartesian_magnetic);
    for (int component = 0; component < 4; ++component) {
        cartesian_magnetic[component] *= grid.magnetic_scale;
    }
    real metric[4][4];
    kerr_schild::compute_metric(position, grid.spin, grid.mass, metric);
    fluid.magnetic_gauss[0] = -metric_inner_product(
        metric, tetrad[0], cartesian_magnetic);
    for (int axis = 1; axis < 4; ++axis) {
        fluid.magnetic_gauss[axis] = metric_inner_product(
            metric, tetrad[axis], cartesian_magnetic);
    }
    return true;
}

template <typename RandomGenerator>
KOKKOS_INLINE_FUNCTION
bool maybe_scatter_photon(real state[8], real stokes[4], real& frequency_hz,
                          const ScatteringModel& model,
                          const FluidCellCGS& fluid,
                          const real tetrad[4][4],
                          real spin, real mass,
                          RandomGenerator& random) {
    // The fluid state is deliberately part of this interface even though the
    // current grey model only requires the tetrad. Frequency-dependent models
    // can directly use density, temperature, velocity, magnetic field, and nu.
    (void)fluid;
    (void)frequency_hz;
    if (!model.enabled || !(model.optical_depth > 0.0)) return false;

    const real scatter_probability = 1.0 - Kokkos::exp(-model.optical_depth);
    if (random.drand() >= scatter_probability) return false;

    // Preserve the photon's energy measured by the local fluid observer, so
    // this is elastic scattering in that frame rather than a momentum reset.
    real local_energy = 0.0;
    for (int mu = 0; mu < 4; ++mu) local_energy -= state[IKT + mu] * tetrad[0][mu];
    if (!(local_energy > 0.0) || !Kokkos::isfinite(local_energy)) return false;

    const real mu = 2.0 * random.drand() - 1.0;
    const real direction_phi = 2.0 * PI * random.drand();
    const real transverse = Kokkos::sqrt(
        Kokkos::fmax(real(0.0), 1.0 - mu * mu));
    const real position[4] = {state[IT], state[IX], state[IY], state[IZ]};
    real scattered_momentum[4];
    fluid_frame_direction_to_covariant_momentum(
        position, tetrad,
        transverse * Kokkos::cos(direction_phi),
        transverse * Kokkos::sin(direction_phi), mu,
        spin, mass, scattered_momentum);
    for (int component = 0; component < 4; ++component) {
        state[IKT + component] = local_energy * scattered_momentum[component];
    }

    stokes[0] *= model.albedo;
    stokes[1] *= 0.5 * model.albedo;
    stokes[2] *= 0.5 * model.albedo;
    stokes[3] *= 0.25 * model.albedo;
    return true;
}
