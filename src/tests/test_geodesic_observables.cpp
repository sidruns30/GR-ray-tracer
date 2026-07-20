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

    return 0;
}
