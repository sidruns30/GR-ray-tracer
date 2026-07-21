#include <cmath>
#include <cstdlib>

#include "../analysis/geodesic_observables.hpp"

int main() {
    real state_a[8] = {0.0, 10.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0};
    real state_b[8] = {0.0, 12.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0};

    const auto obs_a = compute_geodesic_observables(state_a, a_BH, M_BH);
    const auto obs_b = compute_geodesic_observables(state_b, a_BH, M_BH);

    if (std::abs(obs_a.energy - 1.0) >= 1e-14) return EXIT_FAILURE;
    if (std::abs(obs_b.energy - 1.0) >= 1e-14) return EXIT_FAILURE;
    if (std::abs(obs_a.angular_momentum_z) >= 1e-14) return EXIT_FAILURE;
    if (std::abs(obs_b.angular_momentum_z) >= 1e-14) return EXIT_FAILURE;
    if (std::abs(obs_a.carter) >= 1e-12) return EXIT_FAILURE;
    if (std::abs(obs_b.carter) >= 1e-12) return EXIT_FAILURE;
    if (std::abs(obs_a.energy - obs_b.energy) >= 1e-14) return EXIT_FAILURE;
    if (std::abs(obs_a.angular_momentum_z - obs_b.angular_momentum_z) >= 1e-14) return EXIT_FAILURE;
    if (std::abs(obs_a.carter - obs_b.carter) >= 1e-12) return EXIT_FAILURE;

    // Nonzero-y case: y=0 states above can't distinguish a correct L_z = x*p_y - y*p_x
    // from a broken formula (e.g. a sign flip, or a formula that only ever used x).
    // Values are hand-computed against the formula in geodesic_observables.hpp.
    real state_c[8] = {0.0, 3.0, 4.0, 0.0, -2.0, 0.5, -0.3, 0.0};
    const auto obs_c = compute_geodesic_observables(state_c, a_BH, M_BH);
    const real expected_energy_c = 2.0;
    const real expected_Lz_c = 3.0 * (-0.3) - 4.0 * 0.5; // x*p_y - y*p_x = -2.9
    if (std::abs(obs_c.energy - expected_energy_c) >= 1e-14) return EXIT_FAILURE;
    if (std::abs(obs_c.angular_momentum_z - expected_Lz_c) >= 1e-14) return EXIT_FAILURE;

    // Regression guard for the covariant-momentum initialization fix: a photon
    // wavevector built via null_covariant_momentum_from_spatial_direction at a
    // point with nonzero y (so the Kerr-Schild off-diagonal terms are exercised)
    // must be null under the true metric, i.e. obs.norm ~ 0.
    const real X[4] = {0.0, 5.0, -7.0, 2.0};
    const real K_spatial[3] = {0.4, -0.5, 0.768}; // arbitrary unit-ish spatial direction
    real p_cov[4];
    if (!kerr_schild::null_covariant_momentum_from_spatial_direction(X, K_spatial, a_BH, M_BH, p_cov)) {
        return EXIT_FAILURE;
    }
    real state_null[8] = {X[0], X[1], X[2], X[3], p_cov[0], p_cov[1], p_cov[2], p_cov[3]};
    const auto obs_null = compute_geodesic_observables(state_null, a_BH, M_BH);
    if (std::abs(obs_null.norm) >= 1e-10) return EXIT_FAILURE;

    // The optimized Hamiltonian contraction must match the explicit metric
    // derivative used by the original implementation.
    real rhs[8];
    kerr_schild::compute_hamiltonian_rhs(X, p_cov, a_BH, M_BH, rhs);
    real ginv[4][4], dginv[4][4][4];
    kerr_schild::compute_inverse_metric(X, a_BH, M_BH, ginv);
    kerr_schild::compute_inverse_metric_deriv(X, a_BH, M_BH, dginv);
    for (int mu = 0; mu < 4; ++mu) {
        real expected_dx = 0.0;
        for (int nu = 0; nu < 4; ++nu) expected_dx += ginv[mu][nu] * p_cov[nu];
        if (std::abs(rhs[mu] - expected_dx) >= 1e-12) return EXIT_FAILURE;

        real derivative_contraction = 0.0;
        for (int alpha = 0; alpha < 4; ++alpha) {
            for (int beta = 0; beta < 4; ++beta) {
                derivative_contraction += dginv[mu][alpha][beta] * p_cov[alpha] * p_cov[beta];
            }
        }
        if (std::abs(rhs[4 + mu] + 0.5 * derivative_contraction) >= 1e-12) return EXIT_FAILURE;
    }

    return 0;
}
