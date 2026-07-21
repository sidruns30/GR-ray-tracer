#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include "../geodesic_integrate/integrate_cart_ks.hpp"
#include "../geodesic_integrate/rk_integrators.hpp"

namespace {

std::array<real, 8> integrate_geodesic_rk4(
    const std::array<real, 8>& initial, int steps, real interval) {
    Photons storage(1);
    Geodesic_cartesian_kerr_schild system(
        storage, 0.0, 1.0e6, a_BH, M_BH,
        atol_default, rtol_default, min_scale, max_scale, safety);
    std::array<real, 8> state = initial;
    const real dt = interval / static_cast<real>(steps);
    for (int step = 0; step < steps; ++step) system.rk4_step(state.data(), dt);
    return state;
}

real max_error(const std::array<real, 8>& state,
               const std::array<real, 8>& reference) {
    real error = 0.0;
    for (int i = 0; i < 8; ++i) {
        error = std::max(error, std::abs(state[i] - reference[i]));
    }
    return error;
}

real adaptive_oscillator_error(real tolerance, bool& passed) {
    auto rhs = [](const real state[2], real derivative[2]) {
        derivative[0] = state[1];
        derivative[1] = -state[0];
    };
    real state[2] = {1.0, 0.0};
    real time = 0.0;
    real proposed_dt = 0.2;
    constexpr real final_time = 2.0;
    int attempts = 0;
    while (time < final_time && attempts++ < 100000) {
        const real attempted_dt = std::min(proposed_dt, final_time - time);
        real next_dt = attempted_dt;
        real k1[2];
        rhs(state, k1);
        bool accepted = false;
        rk_detail::rk45_step<2>(
            state, &next_dt, &accepted, rhs, k1,
            tolerance, tolerance, min_scale, max_scale, safety);
        proposed_dt = next_dt;
        if (accepted) time += attempted_dt;
    }
    passed = passed && attempts < 100000;
    return std::max(std::abs(state[0] - std::cos(final_time)),
                    std::abs(state[1] + std::sin(final_time)));
}

} // namespace

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    bool passed = true;
    {
        const real position[4] = {0.0, 6.0, 2.0, 1.0};
        const real direction[3] = {-0.4, 0.8, 0.2};
        real momentum[4];
        passed = passed && kerr_schild::null_covariant_momentum_from_spatial_direction(
            position, direction, a_BH, M_BH, momentum);
        const std::array<real, 8> initial = {
            position[0], position[1], position[2], position[3],
            momentum[0], momentum[1], momentum[2], momentum[3]};

        constexpr real interval = 4.0;
        const auto reference = integrate_geodesic_rk4(initial, 8192, interval);
        const real error_8 = max_error(integrate_geodesic_rk4(initial, 8, interval), reference);
        const real error_16 = max_error(integrate_geodesic_rk4(initial, 16, interval), reference);
        const real error_32 = max_error(integrate_geodesic_rk4(initial, 32, interval), reference);
        // Halving the step should approach fourth-order convergence. A lower
        // bound leaves room for reference error and backend floating-point differences.
        passed = passed && error_8 / error_16 > 10.0;
        passed = passed && error_16 / error_32 > 10.0;

        const real loose_error = adaptive_oscillator_error(1.0e-5, passed);
        const real medium_error = adaptive_oscillator_error(1.0e-7, passed);
        const real tight_error = adaptive_oscillator_error(1.0e-9, passed);
        passed = passed && medium_error < loose_error;
        passed = passed && tight_error < medium_error;
        if (!passed) {
            std::cerr << "RK4 errors (8,16,32 steps): "
                      << error_8 << ", " << error_16 << ", " << error_32
                      << "; RK45 errors (1e-5,1e-7,1e-9): "
                      << loose_error << ", " << medium_error << ", " << tight_error
                      << '\n';
        }
    }
    Kokkos::finalize();
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
