#pragma once

#include "../utils.hpp"

namespace rk_detail {

// Pointer outputs are used consistently across host and device compilers.
template <std::size_t N, typename RHS>
KOKKOS_FUNCTION
inline void rk4_step(real* state, real dt, const RHS& rhs) {
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

// k1 = rhs(state) is identical across rejected retries of the same step (only dt
// changes, state doesn't), so the caller computes it once and passes it in here
// instead of it being recomputed on every retry.
//
// atol/rtol_/etc. must be passed explicitly rather than defaulted from the
// globals of the same name: atol_default/rtol_default are runtime-configurable
// `extern real` globals with no device-side symbol, so reading them by name
// (even as a default-argument initializer) fails to compile for a GPU backend
// with "identifier undefined in device code" -- see kerr_schild_core.hpp.
template <std::size_t N, typename RHS>
KOKKOS_FUNCTION
inline void rk45_step(real* state, real* dt, bool* accepted, const RHS& rhs, const real* k1,
                      real atol, real rtol_, real min_step_scale, real max_step_scale, real safety_factor) {
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

    const real dt_val = *dt;
    real k2[N], k3[N], k4[N], k5[N], k6[N], tmp[N];
    for (std::size_t i = 0; i < N; ++i) tmp[i] = state[i] + dt_val * a21 * k1[i];
    rhs(tmp, k2);
    for (std::size_t i = 0; i < N; ++i) tmp[i] = state[i] + dt_val * (a31 * k1[i] + a32 * k2[i]);
    rhs(tmp, k3);
    for (std::size_t i = 0; i < N; ++i) tmp[i] = state[i] + dt_val * (a41 * k1[i] + a42 * k2[i] + a43 * k3[i]);
    rhs(tmp, k4);
    for (std::size_t i = 0; i < N; ++i) tmp[i] = state[i] + dt_val * (a51 * k1[i] + a52 * k2[i] + a53 * k3[i] + a54 * k4[i]);
    rhs(tmp, k5);
    for (std::size_t i = 0; i < N; ++i) tmp[i] = state[i] + dt_val * (a61 * k1[i] + a62 * k2[i] + a63 * k3[i] + a64 * k4[i] + a65 * k5[i]);
    rhs(tmp, k6);

    real x4[N], x5[N], err = 0.0;
    bool finite = true;
    for (std::size_t i = 0; i < N; ++i) {
        x4[i] = state[i] + dt_val * (b4_1 * k1[i] + b4_3 * k3[i] + b4_4 * k4[i] + b4_5 * k5[i]);
        x5[i] = state[i] + dt_val * (b5_1 * k1[i] + b5_3 * k3[i] + b5_4 * k4[i] + b5_5 * k5[i] + b5_6 * k6[i]);
        // Mixed relative/absolute error norm -- position components (~1e4) and
        // momentum components (~1) no longer share one absolute tolerance.
        const real scale_i = atol + rtol_ * Kokkos::fmax(Kokkos::abs(x5[i]), Kokkos::abs(x4[i]));
        const real ratio = Kokkos::abs(x5[i] - x4[i]) / scale_i;
        // std::max silently keeps the old `err` when compared against NaN (any
        // comparison with NaN is false), so a step landing on a coordinate/
        // curvature singularity (e.g. crossing r=0) could otherwise slip through
        // as "small error" and get accepted, permanently corrupting the state.
        if (!Kokkos::isfinite(x5[i]) || !Kokkos::isfinite(ratio)) finite = false;
        err = Kokkos::fmax(err, ratio);
    }

    if (!finite) {
        *accepted = false;
        *dt = dt_val * min_step_scale;
        return;
    }

    if (err == 0.0) {
        *accepted = true;
        *dt = dt_val * max_step_scale;
        for (std::size_t i = 0; i < N; ++i) state[i] = x5[i];
        return;
    }

    real scale = safety_factor * Kokkos::pow(real(1.0) / err, real(0.25));
    scale = Kokkos::fmin(max_step_scale, Kokkos::fmax(min_step_scale, scale));
    if (err <= 1.0) {
        *accepted = true;
        *dt = dt_val * scale;
        for (std::size_t i = 0; i < N; ++i) state[i] = x5[i];
    } else {
        *accepted = false;
        *dt = dt_val * scale;
    }
}

} // namespace rk_detail
