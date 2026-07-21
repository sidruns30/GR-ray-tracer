#pragma once

#include "../metrics/kerr_schild_core.hpp"
#include "../utils.hpp"

KOKKOS_INLINE_FUNCTION
real metric_inner_product(const real g[4][4], const real a[4], const real b[4]) {
    real result = 0.0;
    for (int mu = 0; mu < 4; ++mu) {
        for (int nu = 0; nu < 4; ++nu) result += g[mu][nu] * a[mu] * b[nu];
    }
    return result;
}

KOKKOS_INLINE_FUNCTION
void spherical_kerr_schild_to_cartesian(real r, real theta, real phi, real spin,
                                        real position[4]) {
    const real sin_theta = Kokkos::sin(theta);
    const real cos_theta = Kokkos::cos(theta);
    const real sin_phi = Kokkos::sin(phi);
    const real cos_phi = Kokkos::cos(phi);
    position[0] = 0.0;
    position[1] = (r * cos_phi - spin * sin_phi) * sin_theta;
    position[2] = (r * sin_phi + spin * cos_phi) * sin_theta;
    position[3] = r * cos_theta;
}

KOKKOS_INLINE_FUNCTION
void transform_spherical_four_vector_to_cartesian(real r, real theta, real phi, real spin,
                                                   const real spherical[4], real cartesian[4]) {
    const real sin_theta = Kokkos::sin(theta);
    const real cos_theta = Kokkos::cos(theta);
    const real sin_phi = Kokkos::sin(phi);
    const real cos_phi = Kokkos::cos(phi);
    const real radial_x = cos_phi * sin_theta;
    const real radial_y = sin_phi * sin_theta;
    const real radial_z = cos_theta;
    const real theta_x = (r * cos_phi - spin * sin_phi) * cos_theta;
    const real theta_y = (r * sin_phi + spin * cos_phi) * cos_theta;
    const real theta_z = -r * sin_theta;
    const real phi_x = (-r * sin_phi - spin * cos_phi) * sin_theta;
    const real phi_y = (r * cos_phi - spin * sin_phi) * sin_theta;

    cartesian[0] = spherical[0];
    cartesian[1] = radial_x * spherical[1] + theta_x * spherical[2] + phi_x * spherical[3];
    cartesian[2] = radial_y * spherical[1] + theta_y * spherical[2] + phi_y * spherical[3];
    cartesian[3] = radial_z * spherical[1] + theta_z * spherical[2];
}

// Builds an orthonormal tetrad with e[0] equal to the fluid four-velocity.
KOKKOS_INLINE_FUNCTION
bool build_fluid_tetrad(const real position[4], const real velocity[4],
                        real spin, real mass, real e[4][4]) {
    real g[4][4];
    kerr_schild::compute_metric(position, spin, mass, g);

    const real velocity_norm = metric_inner_product(g, velocity, velocity);
    if (!(velocity_norm < 0.0) || !Kokkos::isfinite(velocity_norm)) return false;
    const real inverse_norm = 1.0 / Kokkos::sqrt(-velocity_norm);
    const real time_sign = velocity[0] >= 0.0 ? 1.0 : -1.0;
    for (int mu = 0; mu < 4; ++mu) e[0][mu] = time_sign * velocity[mu] * inverse_norm;

    for (int axis = 0; axis < 3; ++axis) {
        real candidate[4] = {0.0, 0.0, 0.0, 0.0};
        candidate[axis + 1] = 1.0;

        const real time_projection = metric_inner_product(g, e[0], candidate);
        for (int mu = 0; mu < 4; ++mu) candidate[mu] += time_projection * e[0][mu];
        for (int previous = 0; previous < axis; ++previous) {
            const real projection = metric_inner_product(g, e[previous + 1], candidate);
            for (int mu = 0; mu < 4; ++mu) candidate[mu] -= projection * e[previous + 1][mu];
        }

        const real norm = metric_inner_product(g, candidate, candidate);
        if (!(norm > 1.0e-20) || !Kokkos::isfinite(norm)) return false;
        const real inverse_spatial_norm = 1.0 / Kokkos::sqrt(norm);
        for (int mu = 0; mu < 4; ++mu) e[axis + 1][mu] = candidate[mu] * inverse_spatial_norm;
    }
    return true;
}

KOKKOS_INLINE_FUNCTION
void fluid_frame_direction_to_covariant_momentum(const real position[4],
                                                 const real tetrad[4][4],
                                                 real direction_x,
                                                 real direction_y,
                                                 real direction_z,
                                                 real spin,
                                                 real mass,
                                                 real p_cov[4]) {
    real wavevector[4];
    for (int mu = 0; mu < 4; ++mu) {
        wavevector[mu] = tetrad[0][mu] +
            direction_x * tetrad[1][mu] +
            direction_y * tetrad[2][mu] +
            direction_z * tetrad[3][mu];
    }
    real g[4][4];
    kerr_schild::compute_metric(position, spin, mass, g);
    for (int mu = 0; mu < 4; ++mu) {
        p_cov[mu] = 0.0;
        for (int nu = 0; nu < 4; ++nu) p_cov[mu] += g[mu][nu] * wavevector[nu];
    }
}
