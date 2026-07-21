// Geodesic integration in Cartesian Kerr-Schild coordinates.
#pragma once
#include "../utils.hpp"
#include "../output/write_output.hpp"
#include "../output/display.hpp"
#include "../metrics/kerr_schild_core.hpp"
#include "../radiative_transfer/scattering.hpp"
#include "../input/load_python_arrays.hpp"

struct Geodesic_cartesian_kerr_schild {
    Kokkos::View<real*> photon_x0;
    Kokkos::View<real*> photon_x1;
    Kokkos::View<real*> photon_x2;
    Kokkos::View<real*> photon_x3;
    Kokkos::View<real*> photon_k0;
    Kokkos::View<real*> photon_k1;
    Kokkos::View<real*> photon_k2;
    Kokkos::View<real*> photon_k3;
    Kokkos::View<real*> photon_frequency;
    Kokkos::View<real*> photon_I;
    Kokkos::View<real*> photon_Q;
    Kokkos::View<real*> photon_U;
    Kokkos::View<real*> photon_V;
    Kokkos::View<real*> photon_dlambda;
    Kokkos::View<bool*> photon_terminate;

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
    const ScatteringFluidGrid scattering_fluid;
    ScatteringRandomPool scattering_random_pool;
    const IntegratorType integrator;

    Geodesic_cartesian_kerr_schild(
      Photons &photon_,
      const real r_term_min_, const real r_term_max_,
      real a_BH_, real M_BH_,
      real atol_, real rtol_, real min_step_scale_, real max_step_scale_, real safety_factor_,
      const ScatteringModel& scattering_model_ = ScatteringModel{},
      const ScatteringFluidGrid& scattering_fluid_ = ScatteringFluidGrid{},
      const ScatteringRandomPool& scattering_random_pool_ = ScatteringRandomPool(1),
      IntegratorType integrator_ = IntegratorType::RK45)
      : photon_x0(photon_.x0), photon_x1(photon_.x1), photon_x2(photon_.x2), photon_x3(photon_.x3),
       photon_k0(photon_.k0), photon_k1(photon_.k1), photon_k2(photon_.k2), photon_k3(photon_.k3),
       photon_frequency(photon_.frequency),
       photon_I(photon_.I), photon_Q(photon_.Q), photon_U(photon_.U), photon_V(photon_.V),
       photon_dlambda(photon_.dlambda), photon_terminate(photon_.terminate),
       r_term_min(r_term_min_), r_term_max(r_term_max_),
       a_BH(a_BH_), M_BH(M_BH_),
       atol(atol_), rtol(rtol_), min_step_scale(min_step_scale_), max_step_scale(max_step_scale_),
       safety_factor(safety_factor_),
       scattering_model(scattering_model_), scattering_fluid(scattering_fluid_),
       scattering_random_pool(scattering_random_pool_), integrator(integrator_) {};

    KOKKOS_FUNCTION
    void compute_rhs(const real state[8], real dstates[8]) const {
        kerr_schild::compute_hamiltonian_rhs(state, state + 4, a_BH, M_BH, dstates);
    }

    // rk4_step/rk45_step below duplicate rk_detail::rk4_step/rk45_step's
    // Butcher-tableau logic inline (calling compute_rhs directly) rather than
    // delegating to that generic template, mainly so both share this functor's
    // members directly. rk_detail:: is kept in sync and still used by its
    // host-only unit test (test_rk_integrators.cpp).
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

    // The adaptive step remains thread-local until the retry loop finishes;
    // device backends must not pass a live View element through a nested pointer.
    KOKKOS_FUNCTION
    int rk45_step(
        real* state,
        real* dlambda,
        const real* k1
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
        // Compute distance traveled for adaptive step sizing if needed
        real photon_distance = kerr_schild::compute_r(state[IX], state[IY], state[IZ], a_BH);
        if (photon_distance <= r_term_min) {
            // Horizon-crossing rays contribute no radiation to the camera.
            photon_I(idx) = 0.0;
            photon_Q(idx) = 0.0;
            photon_U(idx) = 0.0;
            photon_V(idx) = 0.0;
            photon_terminate(idx) = true;
            return;
        }
        if (photon_distance > r_term_max) {
            photon_terminate(idx) = true;
            return;
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
                step_accepted = rk45_step(state, &dlambda_local, k1);
                if (step_accepted) break;
            }

            photon_dlambda(idx) = dlambda_local;
            if (!step_accepted) {
                photon_terminate(idx) = true;
                return;
            }
        }
        else {
            // Do not silently write an unchanged state when an invalid or stale
            // enum value reaches the device -- should never happen, but fail
            // loudly rather than quietly leaving the photon frozen.
            photon_terminate(idx) = true;
            return;
        }

        real stokes[4] = {
            photon_I(idx),
            photon_Q(idx),
            photon_U(idx),
            photon_V(idx)
        };
        if (scattering_model.enabled) {
            FluidCellCGS fluid;
            real tetrad[4][4];
            if (sample_scattering_fluid(state, scattering_fluid, fluid, tetrad)) {
                auto random = scattering_random_pool.get_state();
                maybe_scatter_photon(
                    state, stokes, photon_frequency(idx), scattering_model,
                    fluid, tetrad, a_BH, M_BH, random);
                scattering_random_pool.free_state(random);
            }
        }

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
    const NumpyFieldViews& fields,
    const UnitConversions& units,
    const int rank,
    Timers &timers,
    const OutputSelection& output_selection,
    const ScatteringModel& scattering_model = ScatteringModel{},
    IntegratorType integrator = IntegratorType::RK45)
{
    const bool fluid_available =
        fields.r.extent(0) > 0 && fields.theta.extent(0) > 0 && fields.phi.extent(0) > 0 &&
        fields.density.extent(0) > 0 && fields.temperature.extent(0) > 0 &&
        fields.velocity.extent(3) == 4 && fields.magnetic.extent(3) == 4;
    ScatteringModel active_scattering = scattering_model;
    active_scattering.enabled = scattering_model.enabled && fluid_available;
    const ScatteringFluidGrid scattering_fluid{
        fields.r, fields.theta, fields.phi,
        fields.density, fields.temperature, fields.velocity, fields.magnetic,
        fields.coordinate_spacing[0], fields.coordinate_spacing[1], fields.coordinate_spacing[2],
        units.length_cm_per_code, units.time_s_per_code,
        units.density_g_cm3_per_code, units.temperature_k_per_code,
        units.four_velocity_cm_s_per_code, units.magnetic_gauss_per_code,
        a_BH, M_BH, active_scattering.enabled};
    ScatteringRandomPool scattering_random_pool(
        active_scattering.seed + static_cast<std::uint64_t>(rank));
    float termination_fraction = 0.0f;
    const size_t num_photons = photons.x0.extent(0);
    timers.AddTimer({"Geodesic Integration", "Termination Count", "Output"});
    for (int current_step = 0; current_step < max_steps; ++current_step)
    {
        timers.BeginTimer("Geodesic Integration");
        Geodesic_cartesian_kerr_schild step_functor(photons, termination_r_min, termination_r_max, a_BH, M_BH,
            atol_default, rtol_default, min_scale, max_scale, safety,
            active_scattering, scattering_fluid, scattering_random_pool, integrator);
        Kokkos::parallel_for(
        "Integrate Cartesian Kerr-Schild Geodesics",
        Kokkos::RangePolicy<>(0, num_photons),
        step_functor
        );
        timers.EndTimer("Geodesic Integration");
        if (current_step % output_interval == 0 || current_step == max_steps - 1) {
            Kokkos::fence();
            timers.BeginTimer("Termination Count");
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
            int global_terminated = 0;
            MPI_Allreduce(&local_terminated, &global_terminated, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            termination_fraction = static_cast<float>(global_terminated) / static_cast<float>(nphotons);
            timers.EndTimer("Termination Count");
            if (rank == 0) {
                INFO("After step " + std::to_string(current_step) + ", termination fraction: " + std::to_string(termination_fraction),
            Colors::hotpink);
            }

            timers.BeginTimer("Output");
            write_output_step(photons, static_cast<std::size_t>(current_step), rank, output_selection);
            timers.EndTimer("Output");
            if (rank == 0) {
                timers.PrintTimers(current_step);
            }
        }
        if (termination_fraction >= termination_percent) {
            break;
        }
    }
    if (rank == 0) {
        INFO("Geodesic integration completed.");
    }
    return;
}
