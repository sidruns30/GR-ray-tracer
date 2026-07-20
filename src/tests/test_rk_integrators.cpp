#include <cassert>
#include <cmath>

#include "../geodesic_integrate/rk_integrators.hpp"

int main() {
    real state_rk4[2] = {1.0, -2.0};
    auto rhs = [](const real in[2], real out[2]) {
        out[0] = in[1];
        out[1] = 0.0;
    };

    rk_detail::rk4_step<2>(state_rk4, 0.5, rhs);
    assert(std::abs(state_rk4[0] - 0.0) < 1e-14);
    assert(std::abs(state_rk4[1] + 2.0) < 1e-14);

    real state_rk45[2] = {1.0, -2.0};
    real dt = 0.5;
    bool accepted = false;
    rk_detail::rk45_step<2>(state_rk45, dt, accepted, rhs);
    assert(accepted);
    assert(std::abs(state_rk45[0] - 0.0) < 1e-14);
    assert(std::abs(state_rk45[1] + 2.0) < 1e-14);
    return 0;
}
