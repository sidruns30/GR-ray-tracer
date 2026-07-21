/*
    Integrate the geodesic equations in Cartesian Kerr-Schild coordinates

    Inputs: Kokkos::View<real**> photons("camera_pixels", photons_per_process, 9);
            Each row corresponds to a photon with:
            (t, x, y, z, k_t, k_x, k_y, k_z, intensity)

            Kokkos::View<real***> r, theta, phi, rho, bsqr, pgas, Tgas, ug;
            Kokkos::View<real****> bu, uu;
            Compute emmisivity to the intensity and forward integrate
            Use RK4 step for integration

*/
#pragma once
#include <cstdio>
#ifdef KOKKOS_ENABLE_CUDA
#include <cuda_runtime.h>
#endif
#include "../utils.hpp"
#include "../input/load_python_arrays.hpp"
#include "../output/write_output.hpp"
#include "../output/display.hpp"
#include "../metrics/cartesian_kerr_schild.hpp"
#include "../metrics/kerr_schild_core.hpp"
#include "../radiative_transfer/observation.hpp"
#include "../radiative_transfer/scattering.hpp"
#include "rk_integrators.hpp"

struct Geodesic_cartesian_kerr_schild {
    Kokkos::View<real*> photon_x0;
    Kokkos::View<real*> photon_x1;
    Kokkos::View<real*> photon_x2;
    Kokkos::View<real*> photon_x3;
    Kokkos::View<real*> photon_k0;
    Kokkos::View<real*> photon_k1;
    Kokkos::View<real*> photon_k2;
    Kokkos::View<real*> photon_k3;
    Kokkos::View<real*> photon_I;
    Kokkos::View<real*> photon_Q;
    Kokkos::View<real*> photon_U;
    Kokkos::View<real*> photon_V;
    Kokkos::View<real*> photon_dlambda;
    Kokkos::View<bool*> photon_terminate;
    // Diagnostic: why a photon was terminated this step (0=not terminated this
    // step, 1=horizon-absorbed, 2=left [r_term_min, r_term_max], 3=RK45 rejected
    // every attempt) -- see the histogram printed in integrate_geodesics below.
    Kokkos::View<int*> photon_termination_reason;

    Kokkos::View<real*> r, theta, phi;
    Kokkos::View<real***> rho, Tgas;
    real r_min, r_max, theta_min, theta_max, phi_min, phi_max, dlog_r;
    
    size_t nr, ntheta, nphi;
    const real r_term_min;
    const real r_term_max;
    // Captured by value from the (runtime-configurable) globals at functor
    // construction, since device code can't read `extern real` globals directly
    // -- see the comment in kerr_schild_core.hpp.
    const real a_BH;
    const real M_BH;
    // Same capture-by-value reasoning as a_BH/M_BH above, for rk45_step's
    // tolerance/scale parameters (rk_integrators.hpp no longer defaults them
    // from atol_default/rtol_default directly).
    const real atol;
    const real rtol;
    const real min_step_scale;
    const real max_step_scale;
    const real safety_factor;
    const ScatteringModel scattering_model;
    const std::size_t step_index;
    const IntegratorType integrator;

    Geodesic_cartesian_kerr_schild(
      Photons &photon_,
      const Kokkos::View<int*>& termination_reason_,
      const Kokkos::View<real*>& r_,
      const Kokkos::View<real*>& theta_,
      const Kokkos::View<real*>& phi_,
      const Kokkos::View<real***>& rho_,
      const Kokkos::View<real***>& Tgas_,
      real r_min_, real r_max_, real theta_min_, real theta_max_,
      real phi_min_, real phi_max_, real dlog_r_,
      size_t nr_, size_t ntheta_, size_t nphi_, const real r_term_min_, const real r_term_max_,
      real a_BH_, real M_BH_,
      real atol_, real rtol_, real min_step_scale_, real max_step_scale_, real safety_factor_,
      const ScatteringModel& scattering_model_ = ScatteringModel{},
      std::size_t step_index_ = 0,
      IntegratorType integrator_ = IntegratorType::RK45)
      : photon_x0(photon_.x0), photon_x1(photon_.x1), photon_x2(photon_.x2), photon_x3(photon_.x3),
       photon_k0(photon_.k0), photon_k1(photon_.k1), photon_k2(photon_.k2), photon_k3(photon_.k3),
       photon_I(photon_.I), photon_Q(photon_.Q), photon_U(photon_.U), photon_V(photon_.V),
       photon_dlambda(photon_.dlambda), photon_terminate(photon_.terminate),
       photon_termination_reason(termination_reason_),
       r(r_), theta(theta_), phi(phi_), rho(rho_), Tgas(Tgas_),
       r_min(r_min_), r_max(r_max_), theta_min(theta_min_), theta_max(theta_max_),
       phi_min(phi_min_), phi_max(phi_max_), dlog_r(dlog_r_),
       nr(nr_), ntheta(ntheta_), nphi(nphi_), r_term_min(r_term_min_), r_term_max(r_term_max_),
       a_BH(a_BH_), M_BH(M_BH_),
       atol(atol_), rtol(rtol_), min_step_scale(min_step_scale_), max_step_scale(max_step_scale_),
       safety_factor(safety_factor_),
       scattering_model(scattering_model_), step_index(step_index_), integrator(integrator_) {};

    KOKKOS_FUNCTION
    void compute_rhs(const real state[8], real dstates[8]) const {
        real x[4];
        x[0] = state[IT];
        x[1] = state[IX];
        x[2] = state[IY];
        x[3] = state[IZ];
        real p_cov[4];
        p_cov[0] = state[IKT];
        p_cov[1] = state[IKX];
        p_cov[2] = state[IKY];
        p_cov[3] = state[IKZ];
        real ginv[4][4];
        real dginv[4][4][4];
        kerr_schild::compute_inverse_metric(x, a_BH, M_BH, ginv);
        kerr_schild::compute_inverse_metric_deriv(x, a_BH, M_BH, dginv);
        real p_contra[4];
        for (int mu=0; mu<4; ++mu) {
          real sum = 0.0;
          for (int nu=0; nu<4; ++nu) sum += ginv[mu][nu] * p_cov[nu];
          p_contra[mu] = sum;
        }
        for (int mu=0; mu<4; ++mu) dstates[mu] = p_contra[mu];
        for (int mu=0; mu<4; ++mu) {
          real sum = 0.0;
          for (int alpha=0; alpha<4; ++alpha) {
            for (int beta=0; beta<4; ++beta) {
              sum += dginv[mu][alpha][beta] * p_cov[alpha] * p_cov[beta];
            }
          }
          dstates[4 + mu] = -0.5 * sum;
        }
        return;
    }

    // rk4_step/rk45_step below duplicate rk_detail::rk4_step/rk45_step's
    // Butcher-tableau logic inline (calling compute_rhs directly) rather than
    // delegating to that generic template. On an H200/Hopper90 build, calling
    // rk_detail::rk4_step<8>/rk45_step<8> -- with a self-capturing KOKKOS_LAMBDA
    // as the RHS, and separately after replacing that with passing the functor
    // itself, and separately again after converting its reference output
    // params to pointers -- was a complete no-op every time: state/dt/accepted
    // and even freshly-added debug output pointers came back at their
    // untouched initial values, as if the template function's body never ran.
    // A *direct*, non-template compute_rhs(...) call from operator() worked
    // correctly and matched CPU throughout. rk_detail:: itself is left as-is
    // for its host-only unit test (test_rk_integrators.cpp).
    KOKKOS_FUNCTION
    void rk4_step(real* state, real dt_local) const {
        real k1[8], k2[8], k3[8], k4[8], tmp[8];
        compute_rhs(state, k1);
        for (int i = 0; i < 8; ++i) tmp[i] = state[i] + 0.5 * dt_local * k1[i];
        compute_rhs(tmp, k2);
        for (int i = 0; i < 8; ++i) tmp[i] = state[i] + 0.5 * dt_local * k2[i];
        compute_rhs(tmp, k3);
        for (int i = 0; i < 8; ++i) tmp[i] = state[i] + dt_local * k3[i];
        compute_rhs(tmp, k4);
        for (int i = 0; i < 8; ++i) {
            state[i] += (dt_local / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
    }

    // Returns whether the step was accepted (drives the retry loop in
    // operator() below) rather than writing it through a bool* -- every other
    // proven-working function in this file returns real by value or writes
    // only real* outputs; this is the first (and, until isolated, suspect)
    // case of a bool output mixed with real* outputs in the same call.
    KOKKOS_FUNCTION
    int rk45_step(
        real* state,
        real* dlambda,
        const real* k1,
        real* error_out,
        int* finite_out
    ) const
    {
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

        const real dt_val = *dlambda;
        real k2[8], k3[8], k4[8], k5[8], k6[8], tmp[8];
        for (int i = 0; i < 8; ++i) tmp[i] = state[i] + dt_val * a21 * k1[i];
        compute_rhs(tmp, k2);
        for (int i = 0; i < 8; ++i) tmp[i] = state[i] + dt_val * (a31 * k1[i] + a32 * k2[i]);
        compute_rhs(tmp, k3);
        for (int i = 0; i < 8; ++i) tmp[i] = state[i] + dt_val * (a41 * k1[i] + a42 * k2[i] + a43 * k3[i]);
        compute_rhs(tmp, k4);
        for (int i = 0; i < 8; ++i) tmp[i] = state[i] + dt_val * (a51 * k1[i] + a52 * k2[i] + a53 * k3[i] + a54 * k4[i]);
        compute_rhs(tmp, k5);
        for (int i = 0; i < 8; ++i) tmp[i] = state[i] + dt_val * (a61 * k1[i] + a62 * k2[i] + a63 * k3[i] + a64 * k4[i] + a65 * k5[i]);
        compute_rhs(tmp, k6);

        real x4[8], x5[8], err = 0.0;
        bool finite = true;
        for (int i = 0; i < 8; ++i) {
            x4[i] = state[i] + dt_val * (b4_1 * k1[i] + b4_3 * k3[i] + b4_4 * k4[i] + b4_5 * k5[i]);
            x5[i] = state[i] + dt_val * (b5_1 * k1[i] + b5_3 * k3[i] + b5_4 * k4[i] + b5_5 * k5[i] + b5_6 * k6[i]);
            const real scale_i = atol + rtol * Kokkos::fmax(Kokkos::abs(x5[i]), Kokkos::abs(x4[i]));
            const real ratio = Kokkos::abs(x5[i] - x4[i]) / scale_i;
            if (!Kokkos::isfinite(x5[i]) || !Kokkos::isfinite(ratio)) finite = false;
            err = Kokkos::fmax(err, ratio);
        }

        if (error_out) *error_out = err;
        if (finite_out) *finite_out = finite ? 1 : 0;

        if (!finite) {
            *dlambda = dt_val * min_step_scale;
            return 0;
        }

        if (err == 0.0) {
            *dlambda = dt_val * max_step_scale;
            for (int i = 0; i < 8; ++i) state[i] = x5[i];
            return 1;
        }

        real scale = safety_factor * Kokkos::pow(real(1.0) / err, real(0.25));
        scale = Kokkos::fmin(max_step_scale, Kokkos::fmax(min_step_scale, scale));
        if (err <= 1.0) {
            *dlambda = dt_val * scale;
            for (int i = 0; i < 8; ++i) state[i] = x5[i];
            return 1;
        } else {
            *dlambda = dt_val * scale;
            return 0;
        }
    }


    KOKKOS_FUNCTION
    void operator()(const int idx) const {
        // TEMPORARY: tracing why idx==0's state was observed bit-identical
        // across many steps on an H200 run -- remove once root-caused.
        if (idx == 0) {
            printf("[GPUDBG] step=%llu enter idx=0 terminate=%d\n",
                   static_cast<unsigned long long>(step_index), static_cast<int>(photon_terminate(idx)));
        }
        if (photon_terminate(idx)) {
            return;
        }
        // Local state
        real state[8] = {
            photon_x0(idx),
            photon_x1(idx),
            photon_x2(idx),
            photon_x3(idx),
            photon_k0(idx),
            photon_k1(idx),
            photon_k2(idx),
            photon_k3(idx)
        };
        if (idx == 0) {
            printf("[GPUDBG] step=%llu idx=0 pre-state t=%.9f x=%.9f y=%.9f z=%.9f kt=%.9f kx=%.9f ky=%.9f kz=%.9f dlambda=%.9e integrator=%d\n",
                   static_cast<unsigned long long>(step_index), state[0], state[1], state[2], state[3],
                   state[4], state[5], state[6], state[7],
                   photon_dlambda(idx), static_cast<int>(integrator));
            real dbg_rhs[8];
            compute_rhs(state, dbg_rhs);
            real dbg_ginv[4][4];
            const real dbg_X[4] = {state[IT], state[IX], state[IY], state[IZ]};
            kerr_schild::compute_inverse_metric(dbg_X, a_BH, M_BH, dbg_ginv);
            printf("[GPUDBG] step=%llu idx=0 a_BH=%.9f M_BH=%.9f rhs=(%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e) ginv00=%.9e ginv11=%.9e\n",
                   static_cast<unsigned long long>(step_index), a_BH, M_BH,
                   dbg_rhs[0], dbg_rhs[1], dbg_rhs[2], dbg_rhs[3],
                   dbg_rhs[4], dbg_rhs[5], dbg_rhs[6], dbg_rhs[7],
                   dbg_ginv[0][0], dbg_ginv[1][1]);
        }
        // Compute distance traveled for adaptive step sizing if needed
        real photon_distance = kerr_schild::compute_r(state[IX], state[IY], state[IZ], a_BH);
        if (photon_distance <= r_term_min) {
            // Absorbed by the horizon: no radiation reaches the camera along this
            // ray, so it contributes nothing to image_I/Q/U/V -- this is what
            // makes the black-hole shadow visible in the observation image.
            photon_I(idx) = 0.0;
            photon_Q(idx) = 0.0;
            photon_U(idx) = 0.0;
            photon_V(idx) = 0.0;
            photon_terminate(idx) = true;
            photon_termination_reason(idx) = 1;
            return;
        }
        if (photon_distance > r_term_max) {
            photon_terminate(idx) = true;
            photon_termination_reason(idx) = 2;
            return;
        }
        
        // Print the enum constants as well as the captured value.  The original
        // trace printed integrator=1 but did not print anything from inside the
        // RK45 branch, which means either a stale binary was running or 1 was not
        // actually the underlying value of IntegratorType::RK45.
        if (idx == 0) {
            printf("[GPUDBG] step=%llu dispatch integrator=%d RK4=%d RK45=%d\n",
                   static_cast<unsigned long long>(step_index),
                   static_cast<int>(integrator),
                   static_cast<int>(IntegratorType::RK4),
                   static_cast<int>(IntegratorType::RK45));
        }

        if (integrator == IntegratorType::RK4) {
            rk4_step(state, photon_dlambda(idx));
        }
        else if (integrator == IntegratorType::RK45) {
            real k1[8];
            compute_rhs(state, k1);

            // Never take the address of a Kokkos::View element and pass it into a
            // nested device function.  Keep the adaptive step in thread-local
            // storage, then commit it explicitly to the View after the attempt
            // loop.  This also makes aliasing and writeback unambiguous.
            real dlambda_local = photon_dlambda(idx);
            int step_accepted = 0;
            constexpr int max_rk45_attempts = 12;

            for (int attempt = 0; attempt < max_rk45_attempts; ++attempt) {
                real attempt_error = real(0.0);
                int attempt_finite = 0;
                step_accepted = rk45_step(
                    state, &dlambda_local, k1, &attempt_error, &attempt_finite);
                if (idx == 0) {
                    printf("[GPUDBG] step=%llu idx=0 rk45 attempt=%d accepted=%d finite=%d err=%.9e dlambda=%.9e state0=%.9e state1=%.9e\n",
                           static_cast<unsigned long long>(step_index), attempt,
                           step_accepted, attempt_finite, attempt_error,
                           dlambda_local, state[0], state[1]);
                }
                if (step_accepted) break;
            }

            photon_dlambda(idx) = dlambda_local;
            if (!step_accepted) {
                photon_terminate(idx) = true;
                photon_termination_reason(idx) = 3;
                return;
            }
        }
        else {
            // Do not silently write an unchanged state when an invalid or stale
            // enum value reaches the device.
            if (idx == 0) {
                printf("[GPUDBG] ERROR: unknown integrator value %d; no step executed\n",
                       static_cast<int>(integrator));
            }
            photon_terminate(idx) = true;
            photon_termination_reason(idx) = 3;
            return;
        }

        if (idx == 0) {
            printf("[GPUDBG] step=%llu idx=0 post-integrate state t=%.9f x=%.9f y=%.9f z=%.9f kt=%.9f kx=%.9f ky=%.9f kz=%.9f dlambda=%.9e\n",
                   static_cast<unsigned long long>(step_index), state[0], state[1], state[2], state[3],
                   state[4], state[5], state[6], state[7], photon_dlambda(idx));
        }

        real stokes[4] = {
            photon_I(idx),
            photon_Q(idx),
            photon_U(idx),
            photon_V(idx)
        };
        maybe_scatter_photon(state, stokes, scattering_model, static_cast<std::size_t>(idx), step_index);

        photon_x0(idx) = state[IT];
        photon_x1(idx) = state[IX];
        photon_x2(idx) = state[IY];
        photon_x3(idx) = state[IZ];
        photon_k0(idx) = state[IKT];
        photon_k1(idx) = state[IKX];
        photon_k2(idx) = state[IKY];
        photon_k3(idx) = state[IKZ];
        photon_I(idx) = stokes[0];
        photon_Q(idx) = stokes[1];
        photon_U(idx) = stokes[2];
        photon_V(idx) = stokes[3];
        return;
    }
};

// Perform geodesic integration until terminate_percent of photons have terminated
inline void integrate_geodesics(
    Photons &photons,
    const Kokkos::View<real*>& r,
    const Kokkos::View<real*>& theta,
    const Kokkos::View<real*>& phi,
    const Kokkos::View<real***>& rho,
    const Kokkos::View<real***>& Tgas,
    const int rank,
    Timers &timers,
    const ScatteringModel& scattering_model = ScatteringModel{},
    IntegratorType integrator = IntegratorType::RK45)
{
    float termination_fraction = 0.0f;
    const size_t num_photons = photons.x0.extent(0);
    // Diagnostic accompanying the termination-fraction log line below -- see the
    // photon_termination_reason comment in the functor above.
    Kokkos::View<int*> termination_reason("termination_reason", num_photons);
    timers.AddTimer({"Geodesic Integration", "MPI Counts Send/Recv", "Active Photons Calc",  "Output"});
    for (auto current_step = 0; current_step < max_steps; current_step++)
    {
        timers.BeginTimer("Geodesic Integration");
        Geodesic_cartesian_kerr_schild step_functor(photons, termination_reason, r, theta, phi, rho, Tgas,
            r_min, r_max, theta_min, theta_max, phi_min, phi_max, dlog_r, nr, ntheta, nphi,
            termination_r_min, termination_r_max, a_BH, M_BH,
            atol_default, rtol_default, min_scale, max_scale, safety,
            scattering_model, static_cast<std::size_t>(current_step), integrator);
        Kokkos::parallel_for(
        "RK4 Cartesian Kerr-Schild Step",
        Kokkos::RangePolicy<>(0, num_photons),
        step_functor
        );
        // TEMPORARY: unconditional fence + explicit CUDA error check every step,
        // to catch a silent CUDA runtime error (e.g. stack overflow / illegal
        // access) that Release-mode Kokkos::fence() alone won't surface -- see
        // the idx=0 trace in operator() vanishing mid-kernel with no output on
        // an H200/Hopper90 run. Remove once root-caused (this defeats the
        // batched fencing below and will be slow).
#ifdef KOKKOS_ENABLE_CUDA
        {
            Kokkos::fence();
            cudaError_t cuda_err = cudaGetLastError();
            if (cuda_err != cudaSuccess) {
                printf("[GPUDBG] CUDA ERROR after step %d parallel_for: %s (%s)\n",
                       current_step, cudaGetErrorName(cuda_err), cudaGetErrorString(cuda_err));
            }
        }
#endif
        timers.EndTimer("Geodesic Integration");
        // Siddhant: Can be converted to a pretty print function later
        if (current_step % output_interval == 0 || current_step == max_steps - 1) {
            Kokkos::fence();
            timers.BeginTimer("MPI Counts Send/Recv");
            // Get the number of terminated photons per process
            int local_terminated = 0;
            Kokkos::parallel_reduce(
                "Count Terminated Photons",
                Kokkos::RangePolicy<>(0, num_photons),
                KOKKOS_LAMBDA(const int i, int& local_count) {
                    if (photons.terminate(i)) {
                        local_count += 1;
                    }
                }, local_terminated
            );
            // Reduce across all MPI processes
            int global_terminated = 0;
            MPI_Allreduce(&local_terminated, &global_terminated, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            termination_fraction = static_cast<float>(global_terminated) / static_cast<float>(nphotons);
            // Histogram of *why* terminated photons were terminated (see
            // photon_termination_reason above) -- summed across MPI ranks like
            // global_terminated, so this stays meaningful with rank>1.
            int local_reason_counts[3] = {0, 0, 0};
            for (int reason = 1; reason <= 3; ++reason) {
                int count = 0;
                Kokkos::parallel_reduce(
                    "Count Termination Reasons",
                    Kokkos::RangePolicy<>(0, num_photons),
                    KOKKOS_LAMBDA(const int i, int& local_count) {
                        if (termination_reason(i) == reason) local_count += 1;
                    }, count
                );
                local_reason_counts[reason - 1] = count;
            }
            int global_reason_counts[3] = {0, 0, 0};
            MPI_Allreduce(local_reason_counts, global_reason_counts, 3, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            if (rank == 0) {
                INFO("After step " + std::to_string(current_step) + ", termination fraction: " + std::to_string(termination_fraction),
            Colors::hotpink);
                INFO("  Termination reasons -- horizon: " + std::to_string(global_reason_counts[0]) +
                     ", escaped range: " + std::to_string(global_reason_counts[1]) +
                     ", RK45 rejected all attempts: " + std::to_string(global_reason_counts[2]),
                     Colors::hotpink);
                timers.PrintTimers(current_step);
            }
            timers.EndTimer("MPI Counts Send/Recv");


            timers.BeginTimer("Active Photons Calc");
            // Copy to host and print photon positions
            photons.copy_to_host();
            // Find the indices of 5 photons that are not terminated
            std::vector<size_t> active_photon_indices;
            for (size_t i = 0; i < num_photons; i++) {
                if (!photons.terminate_host(i)) {
                    active_photon_indices.push_back(i);
                }
            }
            // Pick 5 random active photons to display
            std::shuffle(active_photon_indices.begin(), active_photon_indices.end(), std::default_random_engine(42));
            for (size_t i = 0; i < std::min(size_t(5), active_photon_indices.size()); i++) {
                INFO("Active Photon " + std::to_string(active_photon_indices[i]) + ": (t, x, y, z) = (" +
                     std::to_string(photons.x0_host(active_photon_indices[i])) + ", " +
                     std::to_string(photons.x1_host(active_photon_indices[i])) + ", " +
                     std::to_string(photons.x2_host(active_photon_indices[i])) + ", " +
                     std::to_string(photons.x3_host(active_photon_indices[i])) + "), Intensity = " +
                     std::to_string(photons.I_host(active_photon_indices[i])) + ")");
            }
            timers.EndTimer("Active Photons Calc");

            timers.BeginTimer("Output");
            write_photon_output(photons, current_step, rank);
            write_observation_products(photons, current_step, rank);
            timers.EndTimer("Output");
        }
        if (termination_fraction >= termination_percent) {
            break;
        }
    }
    INFO("Rank " + std::to_string(rank) + ": Geodesic integration completed.");
    return;
}