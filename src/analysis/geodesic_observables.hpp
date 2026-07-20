#pragma once

#include "../utils.hpp"
#include "../metrics/cartesian_kerr_schild.hpp"
#include "../metrics/kerr_schild_core.hpp"

struct GeodesicObservables {
    real norm = 0.0;
    real energy = 0.0;
    real angular_momentum_z = 0.0;
    real carter = 0.0;
};

KOKKOS_INLINE_FUNCTION
GeodesicObservables compute_geodesic_observables(const real state[8], real a_BH, real M_BH) {
    GeodesicObservables obs;
    const real X[4] = {state[IT], state[IX], state[IY], state[IZ]};
    const real P[4] = {state[IKT], state[IKX], state[IKY], state[IKZ]};
    const real x = X[1];
    const real y = X[2];
    const real z = X[3];
    const real r = kerr_schild::compute_r(x, y, z, a_BH);
    const real sin_theta = sqrt(std::max(real(0.0), 1.0 - SQR(z / r)));
    const real cos_theta = z / r;
    const real sin_theta_safe = std::max(real(1e-15), sin_theta);

    obs.norm = cartesian_kerr_schild_norm(X, P, a_BH, M_BH);
    obs.energy = -P[0];
    obs.angular_momentum_z = x * P[2] - y * P[1];

    const real p_theta = ((x * P[1] + y * P[2]) * cos_theta / sin_theta_safe) - r * sin_theta * P[3];
    obs.carter = p_theta * p_theta
               + SQR(cos_theta) * (SQR(a_BH) * SQR(obs.energy) + SQR(obs.angular_momentum_z) / SQR(sin_theta_safe));
    return obs;
}
