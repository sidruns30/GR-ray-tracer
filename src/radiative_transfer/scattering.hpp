#pragma once

#include <cstdint>

#include <Kokkos_Random.hpp>

#include "../utils.hpp"
#include "fluid_frame.hpp"

struct ScatteringModel {
    bool enabled = false;
    real albedo = 1.0;
    std::uint64_t seed = 0;
};

using ScatteringRandomPool =
    Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>;

// Views and conversion factors needed to sample a rank-local fluid cell.
// `available` is set on the host only when every required field exists, so
// vacuum integrations never enter the scattering path on the device.
struct ScatteringFluidGrid {
    Kokkos::View<real***> r;
    Kokkos::View<real***> theta;
    Kokkos::View<real***> phi;
    Kokkos::View<real***> density;
    Kokkos::View<real***> temperature;
    Kokkos::View<real****> velocity;
    Kokkos::View<real****> magnetic;
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

KOKKOS_INLINE_FUNCTION
real periodic_angle_distance(real lhs, real rhs) {
    real distance = Kokkos::fmod(Kokkos::abs(lhs - rhs), 2.0 * PI);
    return Kokkos::fmin(distance, 2.0 * PI - distance);
}

template <typename ViewType>
KOKKOS_INLINE_FUNCTION
int nearest_radial_index(const ViewType& coordinates, real radius, int j, int k) {
    if (coordinates.extent(0) == 0 || !(radius > 0.0)) return -1;
    int nearest = 0;
    const real first = coordinates(0, j, k);
    if (!(first > 0.0)) return -1;
    real best_distance = Kokkos::abs(Kokkos::log(radius / first));
    for (std::size_t i = 1; i < coordinates.extent(0); ++i) {
        const real coordinate = coordinates(i, j, k);
        if (!(coordinate > 0.0)) continue;
        const real distance = Kokkos::abs(Kokkos::log(radius / coordinate));
        if (distance < best_distance) {
            nearest = static_cast<int>(i);
            best_distance = distance;
        }
    }
    return nearest;
}

template <typename ViewType>
KOKKOS_INLINE_FUNCTION
int nearest_polar_index(const ViewType& coordinates, real polar, int i, int k) {
    if (coordinates.extent(1) == 0) return -1;
    int nearest = 0;
    real best_distance = Kokkos::abs(polar - coordinates(i, 0, k));
    for (std::size_t j = 1; j < coordinates.extent(1); ++j) {
        const real distance = Kokkos::abs(polar - coordinates(i, j, k));
        if (distance < best_distance) {
            nearest = static_cast<int>(j);
            best_distance = distance;
        }
    }
    return nearest;
}

template <typename ViewType>
KOKKOS_INLINE_FUNCTION
int nearest_azimuth_index(const ViewType& coordinates, real azimuth, int i, int j) {
    if (coordinates.extent(2) == 0) return -1;
    int nearest = 0;
    real best_distance = periodic_angle_distance(azimuth, coordinates(i, j, 0));
    for (std::size_t k = 1; k < coordinates.extent(2); ++k) {
        const real distance = periodic_angle_distance(azimuth, coordinates(i, j, k));
        if (distance < best_distance) {
            nearest = static_cast<int>(k);
            best_distance = distance;
        }
    }
    return nearest;
}

template <typename ViewType>
KOKKOS_INLINE_FUNCTION
bool within_radial_cell(const ViewType& coordinates, real radius, int i, int j, int k) {
    const real center = coordinates(i, j, k);
    if (!(center > 0.0) || !(radius > 0.0)) return false;
    const real center_distance = Kokkos::abs(Kokkos::log(radius / center));
    if (coordinates.extent(0) == 1) return center_distance <= 1.0e-10;
    real neighbor_span = 0.0;
    if (i > 0 && coordinates(i - 1, j, k) > 0.0) {
        neighbor_span = Kokkos::fmax(neighbor_span,
            Kokkos::abs(Kokkos::log(center / coordinates(i - 1, j, k))));
    }
    if (static_cast<std::size_t>(i + 1) < coordinates.extent(0) &&
        coordinates(i + 1, j, k) > 0.0) {
        neighbor_span = Kokkos::fmax(neighbor_span,
            Kokkos::abs(Kokkos::log(center / coordinates(i + 1, j, k))));
    }
    return center_distance <= 0.500001 * neighbor_span;
}

template <typename ViewType>
KOKKOS_INLINE_FUNCTION
bool within_polar_cell(const ViewType& coordinates, real polar, int i, int j, int k) {
    const real center = coordinates(i, j, k);
    const real center_distance = Kokkos::abs(polar - center);
    if (coordinates.extent(1) == 1) return center_distance <= 1.0e-10;
    real neighbor_span = 0.0;
    if (j > 0) neighbor_span = Kokkos::fmax(
        neighbor_span, Kokkos::abs(center - coordinates(i, j - 1, k)));
    if (static_cast<std::size_t>(j + 1) < coordinates.extent(1)) {
        neighbor_span = Kokkos::fmax(
            neighbor_span, Kokkos::abs(center - coordinates(i, j + 1, k)));
    }
    return center_distance <= 0.500001 * neighbor_span;
}

template <typename ViewType>
KOKKOS_INLINE_FUNCTION
bool within_azimuth_cell(const ViewType& coordinates, real azimuth, int i, int j, int k) {
    const real center = coordinates(i, j, k);
    const real center_distance = periodic_angle_distance(azimuth, center);
    if (coordinates.extent(2) == 1) return center_distance <= 1.0e-10;
    real neighbor_span = 0.0;
    if (k > 0) neighbor_span = Kokkos::fmax(
        neighbor_span, periodic_angle_distance(center, coordinates(i, j, k - 1)));
    if (static_cast<std::size_t>(k + 1) < coordinates.extent(2)) {
        neighbor_span = Kokkos::fmax(
            neighbor_span, periodic_angle_distance(center, coordinates(i, j, k + 1)));
    }
    return center_distance <= 0.500001 * neighbor_span;
}

template <typename RadialView, typename PolarView, typename AzimuthView>
KOKKOS_INLINE_FUNCTION
bool locate_structured_grid_cell(
    const RadialView& radial_coordinates,
    const PolarView& polar_coordinates,
    const AzimuthView& azimuth_coordinates,
    real radius, real polar, real azimuth,
    int& i, int& j, int& k) {
    if (radial_coordinates.extent(0) == 0 ||
        radial_coordinates.extent(1) == 0 ||
        radial_coordinates.extent(2) == 0) return false;

    i = static_cast<int>(radial_coordinates.extent(0) / 2);
    j = static_cast<int>(radial_coordinates.extent(1) / 2);
    k = static_cast<int>(radial_coordinates.extent(2) / 2);
    // Coordinate fields may be nonuniform and depend on every logical index.
    // A few coordinate-descent passes find the nearest structured-grid cell
    // without an exhaustive scan over all nr*ntheta*nphi cells.
    for (int iteration = 0; iteration < 3; ++iteration) {
        i = nearest_radial_index(radial_coordinates, radius, j, k);
        if (i < 0) return false;
        j = nearest_polar_index(polar_coordinates, polar, i, k);
        if (j < 0) return false;
        k = nearest_azimuth_index(azimuth_coordinates, azimuth, i, j);
        if (k < 0) return false;
    }
    return within_radial_cell(radial_coordinates, radius, i, j, k) &&
        within_polar_cell(polar_coordinates, polar, i, j, k) &&
        within_azimuth_cell(azimuth_coordinates, azimuth, i, j, k);
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

    int i = -1;
    int j = -1;
    int k = -1;
    if (!locate_structured_grid_cell(
            grid.r, grid.theta, grid.phi,
            radius, polar, azimuth, i, j, k)) return false;

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
        grid.r(i, j, k), grid.theta(i, j, k), grid.phi(i, j, k), grid.spin,
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
        grid.r(i, j, k), grid.theta(i, j, k), grid.phi(i, j, k), grid.spin,
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

// Free-electron mass, used to turn a mass density into the number density
// Thomson scattering depends on. Assumes fully ionized hydrogen (mu_e = 1),
// so n_e = rho / m_p.
inline constexpr real proton_mass_cgs = 1.67262192369e-24; // g
inline constexpr real sigma_thomson_cgs = 6.6524587158e-25; // cm^2

template <typename RandomGenerator>
KOKKOS_INLINE_FUNCTION
bool maybe_scatter_photon(real state[8], real stokes[4], real& frequency_hz,
                          const ScatteringModel& model,
                          const real photondlambda,
                          const real length_scale,
                          const FluidCellCGS& fluid,
                          const real tetrad[4][4],
                          real spin, real mass,
                          RandomGenerator& random) {
    // The fluid state is deliberately part of this interface even though the
    // current grey model only requires the tetrad. Frequency-dependent models
    // can directly use density, temperature, velocity, magnetic field, and nu.
    (void)frequency_hz;
    if (!model.enabled) return false;

    // Thomson-scattering optical depth accumulated over this step. dx^mu/dlambda
    // is the photon's contravariant momentum (see compute_hamiltonian_rhs), so
    // photondlambda carries the same code-length units as state[IX..IZ] and
    // needs the same length_scale used for fluid.position_cm to become a
    // physical path length in cm.
    const real path_length_cm = photondlambda * length_scale;
    const real electron_number_density_cm3 = fluid.density_g_cm3 / proton_mass_cgs;
    const real optical_depth_step = sigma_thomson_cgs * electron_number_density_cm3 * path_length_cm;

    const real scatter_probability = 1.0 - Kokkos::exp(-optical_depth_step);
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
