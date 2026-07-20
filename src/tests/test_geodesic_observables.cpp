#include <cmath>
#include <cstdlib>

#include "../analysis/geodesic_observables.hpp"

int main() {
    real state_a[8] = {0.0, 10.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0};
    real state_b[8] = {0.0, 12.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0};

    const auto obs_a = compute_geodesic_observables(state_a);
    const auto obs_b = compute_geodesic_observables(state_b);

    if (std::abs(obs_a.energy - 1.0) >= 1e-14) return EXIT_FAILURE;
    if (std::abs(obs_b.energy - 1.0) >= 1e-14) return EXIT_FAILURE;
    if (std::abs(obs_a.angular_momentum_z) >= 1e-14) return EXIT_FAILURE;
    if (std::abs(obs_b.angular_momentum_z) >= 1e-14) return EXIT_FAILURE;
    if (std::abs(obs_a.carter) >= 1e-12) return EXIT_FAILURE;
    if (std::abs(obs_b.carter) >= 1e-12) return EXIT_FAILURE;
    if (std::abs(obs_a.energy - obs_b.energy) >= 1e-14) return EXIT_FAILURE;
    if (std::abs(obs_a.angular_momentum_z - obs_b.angular_momentum_z) >= 1e-14) return EXIT_FAILURE;
    if (std::abs(obs_a.carter - obs_b.carter) >= 1e-12) return EXIT_FAILURE;

    return 0;
}
