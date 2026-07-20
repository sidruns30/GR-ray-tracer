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

    Kokkos::View<real*> r, theta, phi;
    Kokkos::View<real***> rho, Tgas;
    real r_min, r_max, theta_min, theta_max, phi_min, phi_max, dlog_r;
    
    size_t nr, ntheta, nphi;
    const real r_term_min;
    const real r_term_max;
    const ScatteringModel scattering_model;
    const std::size_t step_index;
    const IntegratorType integrator;

    Geodesic_cartesian_kerr_schild(
      Photons &photon_,
      const Kokkos::View<real*>& r_,
      const Kokkos::View<real*>& theta_,
      const Kokkos::View<real*>& phi_,
      const Kokkos::View<real***>& rho_,
      const Kokkos::View<real***>& Tgas_,
      real r_min_, real r_max_, real theta_min_, real theta_max_,
      real phi_min_, real phi_max_, real dlog_r_, 
      size_t nr_, size_t ntheta_, size_t nphi_, const real r_term_min_, const real r_term_max_,
      const ScatteringModel& scattering_model_ = ScatteringModel{},
      std::size_t step_index_ = 0,
      IntegratorType integrator_ = IntegratorType::RK45)
      : photon_x0(photon_.x0), photon_x1(photon_.x1), photon_x2(photon_.x2), photon_x3(photon_.x3),
       photon_k0(photon_.k0), photon_k1(photon_.k1), photon_k2(photon_.k2), photon_k3(photon_.k3),
       photon_I(photon_.I), photon_Q(photon_.Q), photon_U(photon_.U), photon_V(photon_.V),
       photon_dlambda(photon_.dlambda), photon_terminate(photon_.terminate),
       r(r_), theta(theta_), phi(phi_), rho(rho_), Tgas(Tgas_),
       r_min(r_min_), r_max(r_max_), theta_min(theta_min_), theta_max(theta_max_),
       phi_min(phi_min_), phi_max(phi_max_), dlog_r(dlog_r_),
       nr(nr_), ntheta(ntheta_), nphi(nphi_), r_term_min(r_term_min_), r_term_max(r_term_max_),
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
        kerr_schild::compute_inverse_metric(x, ginv);
        kerr_schild::compute_inverse_metric_deriv(x, dginv);
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

    KOKKOS_FUNCTION
    void rk4_step(real (&state)[8], real dt_local) const {
        const auto* self = this;
        auto rhs = KOKKOS_LAMBDA(const real in[8], real out[8]) {
            self->compute_rhs(in, out);
        };
        rk_detail::rk4_step<8>(state, dt_local, rhs);
    }

    // step_accepted drives the retry loop in operator() below.
    KOKKOS_FUNCTION
    void rk45_step(
        real (&state)[8],
        real& dlambda,
        bool& step_accepted,
        const real (&k1)[8]
    ) const
    {
        const auto* self = this;
        auto rhs = KOKKOS_LAMBDA(const real in[8], real out[8]) {
            self->compute_rhs(in, out);
        };
        rk_detail::rk45_step<8>(state, dlambda, step_accepted, rhs, k1);
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
        real photon_distance = kerr_schild::compute_r(state[IX], state[IY], state[IZ]);
        if (photon_distance <= r_term_min) {
            // Absorbed by the horizon: no radiation reaches the camera along this
            // ray, so it contributes nothing to image_I/Q/U/V -- this is what
            // makes the black-hole shadow visible in the observation image.
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
            bool step_accepted = false;
            int max_attempts = 20;
            int attempts = 0;
            // state (hence k1 = compute_rhs(state)) is unchanged across rejected
            // retries -- only dlambda shrinks -- so k1 is computed once and reused.
            real k1[8];
            compute_rhs(state, k1);
            while (!step_accepted && attempts < max_attempts) {
                rk45_step(state, photon_dlambda(idx), step_accepted, k1);
                attempts++;
            }
            if (!step_accepted) {
                photon_terminate(idx) = true;
                return;
            }
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
    timers.AddTimer({"Geodesic Integration", "MPI Counts Send/Recv", "Active Photons Calc",  "Output"});
    for (auto current_step = 0; current_step < max_steps; current_step++)
    {
        timers.BeginTimer("Geodesic Integration");
        Geodesic_cartesian_kerr_schild step_functor(photons, r, theta, phi, rho, Tgas,
            r_min, r_max, theta_min, theta_max, phi_min, phi_max, dlog_r, nr, ntheta, nphi,
            termination_r_min, termination_r_max, scattering_model,
            static_cast<std::size_t>(current_step), integrator);
        Kokkos::parallel_for(
        "RK4 Cartesian Kerr-Schild Step",
        Kokkos::RangePolicy<>(0, num_photons),
        step_functor
        );
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
            if (rank == 0) {
                INFO("After step " + std::to_string(current_step) + ", termination fraction: " + std::to_string(termination_fraction), 
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