#pragma once

#include "../utils.hpp"

namespace rk_detail {

template <std::size_t N, typename RHS>
inline void rk4_step(real (&state)[N], real dt, const RHS& rhs) {
    real k1[N], k2[N], k3[N], k4[N], tmp[N];
    rhs(state, k1);
    for (std::size_t i = 0; i < N; ++i) tmp[i] = state[i] + 0.5 * dt * k1[i];
    rhs(tmp, k2);
    for (std::size_t i = 0; i < N; ++i) tmp[i] = state[i] + 0.5 * dt * k2[i];
    rhs(tmp, k3);
    for (std::size_t i = 0; i < N; ++i) tmp[i] = state[i] + dt * k3[i];
    rhs(tmp, k4);
    for (std::size_t i = 0; i < N; ++i) {
        state[i] += (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
}

template <std::size_t N, typename RHS>
inline void rk45_step(real (&state)[N], real& dt, bool& accepted, const RHS& rhs,
                      real tolerance = tol, real min_step_scale = min_scale,
                      real max_step_scale = max_scale, real safety_factor = safety) {
    const real a21 = 1.0 / 4.0;
    const real a31 = 3.0 / 32.0, a32 = 9.0 / 32.0;
    const real a41 = 1932.0 / 2197.0, a42 = -7200.0 / 2197.0, a43 = 7296.0 / 2197.0;
    const real a51 = 439.0 / 216.0, a52 = -8.0, a53 = 3680.0 / 513.0, a54 = -845.0 / 4104.0;
    const real a61 = -8.0 / 27.0, a62 = 2.0, a63 = -3544.0 / 2565.0, a64 = 1859.0 / 4104.0, a65 = -11.0 / 40.0;

    const real b4_1 = 25.0 / 216.0;
    const real b4_3 = 1408.0 / 2565.0;
    const real b4_4 = 2197.0 / 4104.0;
    const real b4_5 = -1.0 / 5.0;
    const real b5_1 = 16.0 / 135.0;
    const real b5_3 = 6656.0 / 12825.0;
    const real b5_4 = 28561.0 / 56430.0;
    const real b5_5 = -9.0 / 50.0;
    const real b5_6 = 2.0 / 55.0;

    real k1[N], k2[N], k3[N], k4[N], k5[N], k6[N], tmp[N];
    rhs(state, k1);
    for (std::size_t i = 0; i < N; ++i) tmp[i] = state[i] + dt * a21 * k1[i];
    rhs(tmp, k2);
    for (std::size_t i = 0; i < N; ++i) tmp[i] = state[i] + dt * (a31 * k1[i] + a32 * k2[i]);
    rhs(tmp, k3);
    for (std::size_t i = 0; i < N; ++i) tmp[i] = state[i] + dt * (a41 * k1[i] + a42 * k2[i] + a43 * k3[i]);
    rhs(tmp, k4);
    for (std::size_t i = 0; i < N; ++i) tmp[i] = state[i] + dt * (a51 * k1[i] + a52 * k2[i] + a53 * k3[i] + a54 * k4[i]);
    rhs(tmp, k5);
    for (std::size_t i = 0; i < N; ++i) tmp[i] = state[i] + dt * (a61 * k1[i] + a62 * k2[i] + a63 * k3[i] + a64 * k4[i] + a65 * k5[i]);
    rhs(tmp, k6);

    real x4[N], x5[N], err = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        x4[i] = state[i] + dt * (b4_1 * k1[i] + b4_3 * k3[i] + b4_4 * k4[i] + b4_5 * k5[i]);
        x5[i] = state[i] + dt * (b5_1 * k1[i] + b5_3 * k3[i] + b5_4 * k4[i] + b5_5 * k5[i] + b5_6 * k6[i]);
        err = std::max(err, std::abs(x5[i] - x4[i]));
    }

    if (err == 0.0) {
        accepted = true;
        dt *= max_step_scale;
        for (std::size_t i = 0; i < N; ++i) state[i] = x5[i];
        return;
    }

    real scale = safety_factor * std::pow(tolerance / err, 0.25);
    scale = std::clamp(scale, min_step_scale, max_step_scale);
    if (err < tolerance) {
        accepted = true;
        dt *= scale;
        for (std::size_t i = 0; i < N; ++i) state[i] = x5[i];
    } else {
        accepted = false;
        dt *= scale;
    }
}

} // namespace rk_detail

