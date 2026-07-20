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
    const real r_horizon;
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
      size_t nr_, size_t ntheta_, size_t nphi_, const real r_horizon_,
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
       nr(nr_), ntheta(ntheta_), nphi(nphi_), r_horizon(r_horizon_),
       scattering_model(scattering_model_), step_index(step_index_), integrator(integrator_) {};

    // Emissivity calculation can be added here as needed
    KOKKOS_FUNCTION
    void add_emissivity(real state[8], real& intensity) const {
        // Find the nearest cell in the grid and add emissivity to intensity
        real x = state[IX];
        real y = state[IY];
        real z = state[IZ];
        real RSQ = SQR(x) + SQR(y) + SQR(z);
        real r = sqrt(0.5 * (RSQ - SQR(a_BH) + sqrt(SQR(RSQ - SQR(a_BH)) + 4.0 * SQR(a_BH * z))));
        real theta = acos(z / r);
        real phi = (atan2(y, x) - atan2(a_BH, r));
        if (phi < 0) phi += 2.0*PI;

        size_t ir =  static_cast<size_t>((log(r) - log(r_min)) / dlog_r);
        size_t itheta = static_cast<size_t>((theta - theta_min) / ((theta_max - theta_min) / ntheta));
        size_t iphi = static_cast<size_t>((phi - phi_min) / ((phi_max - phi_min) / nphi));

        // Bounds checking with proper range check
        if (r > r_min && r < r_max &&
            theta > theta_min && theta < theta_max &&
            phi > phi_min && phi < phi_max &&
            ir < nr && itheta < ntheta && iphi < nphi
        ) {
            intensity += rho(ir, itheta, iphi) * (Tgas(ir, itheta, iphi));
        }
        else
        {
            intensity += 0.0;
        }
    }
    
    KOKKOS_FUNCTION
    void compute_inverse_metric(const real X[4], real ginv[4][4]) const {
        for (int a=0;a<4;++a) for (int b=0;b<4;++b) ginv[a][b] = 0.0;
        real x = X[1];
        real y = X[2];
        real z = X[3];
        real RSQ = SQR(x) + SQR(y) + SQR(z);
        real r = sqrt(0.5 * (RSQ - SQR(a_BH) + sqrt(SQR(RSQ - SQR(a_BH)) + 4.0 * SQR(a_BH * z))));
        ginv[0][0] = (-2*M_BH*CUBE(r) - SQR(a_BH * z) - QUAD(r))/(SQR(a_BH * z)+ QUAD(r));
        ginv[0][1] = 2*M_BH*CUBE(r)*(a_BH * y + r * x)/((SQR(a_BH) + SQR(r))*(SQR(a_BH * z) + QUAD(r)));
        ginv[0][2] = -2*M_BH*CUBE(r)*(a_BH * x - r * y)/((SQR(a_BH) + SQR(r))*(SQR(a_BH * z) + QUAD(r)));
        ginv[0][3] = 2*M_BH*SQR(r)*z/(SQR(a_BH * z) + QUAD(r));
        ginv[1][0] = 2*M_BH*CUBE(r)*(a_BH * y + r * x)/((SQR(a_BH) + SQR(r))*(SQR(a_BH * z) + QUAD(r)));
        ginv[1][1] = (-2*M_BH*CUBE(r)*SQR(a_BH * y + r * x) + SQR(SQR(a_BH) + SQR(r))*(SQR(a_BH * z) + QUAD(r)))/ (SQR(SQR(a_BH) + SQR(r))*(SQR(a_BH * z) + QUAD(r)));
        ginv[1][2] = 2*M_BH*CUBE(r)*(a_BH * x - r * y)*(a_BH * y + r * x)/(SQR(SQR(a_BH) + SQR(r))*(SQR(a_BH * z) + QUAD(r)));
        ginv[1][3] = -2*M_BH*SQR(r)*z*(a_BH * y + r * x)/((SQR(a_BH) + SQR(r))*(SQR(a_BH * z) + QUAD(r)));
        ginv[2][0] = -2*M_BH*CUBE(r)*(a_BH * x - r * y)/((SQR(a_BH) + SQR(r))*(SQR(a_BH * z) + QUAD(r)));
        ginv[2][1] = 2*M_BH*CUBE(r)*(a_BH * x - r * y)*(a_BH * y + r * x)/(SQR(SQR(a_BH) + SQR(r))*(SQR(a_BH * z) + QUAD(r)));
        ginv[2][2] = (-2*M_BH*CUBE(r)*SQR(a_BH * x - r * y) + SQR(SQR(a_BH) + SQR(r))*(SQR(a_BH * z) + QUAD(r)))/(SQR(SQR(a_BH) + SQR(r))*(SQR(a_BH * z) + QUAD(r)));
        ginv[2][3] = 2*M_BH*SQR(r)*z*(a_BH * x - r * y)/((SQR(a_BH) + SQR(r))*(SQR(a_BH * z) + QUAD(r)));
        ginv[3][0] = 2*M_BH*SQR(r)*z/(SQR(a_BH * z) + QUAD(r));
        ginv[3][1] = -2*M_BH*SQR(r)*z*(a_BH * y + r * x)/((SQR(a_BH) + SQR(r))*(SQR(a_BH * z) + QUAD(r)));
        ginv[3][2] = 2*M_BH*SQR(r)*z*(a_BH * x - r * y)/((SQR(a_BH) + SQR(r))*(SQR(a_BH * z) + QUAD(r)));
        ginv[3][3] = (-2*M_BH*r*SQR(z) + SQR(a_BH * z) + QUAD(r))/(SQR(a_BH * z) + QUAD(r));
        return;
    }

    KOKKOS_FUNCTION
    void compute_inverse_metric_deriv(const real X[4],
                                      real dginv[4][4][4]) const
    {
        real ginv[4][4];
        // Second order finite differencing for metric
        for (int mu=0; mu<4; ++mu) {
            real X_ph[4] = {X[0], X[1], X[2], X[3]};
            real X_mh[4] = {X[0], X[1], X[2], X[3]};
            X_ph[mu] += metric_derivative_h;
            X_mh[mu] -= metric_derivative_h;
            compute_inverse_metric(X_ph, ginv);
            real ginv_ph[4][4];
            for (int a=0;a<4;++a) for (int b=0;b<4;++b) ginv_ph[a][b] = ginv[a][b];
            compute_inverse_metric(X_mh, ginv);
            real ginv_mh[4][4];
            for (int a=0;a<4;++a) for (int b=0;b<4;++b) ginv_mh[a][b] = ginv[a][b];
            for (int a=0;a<4;++a) {
                for (int b=0;b<4;++b) {
                    dginv[mu][a][b] = (ginv_ph[a][b] - ginv_mh[a][b]) / (2.0 * metric_derivative_h);
                }
            }
        }
        return;
    }


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
        compute_inverse_metric(x, ginv);
        compute_inverse_metric_deriv(x, dginv);
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

    // Siddhant: This computes the error as well but I choose not to use it for now
    // In the future, we can output the error as well
    KOKKOS_FUNCTION
    void rk45_step(
        real (&state)[8],
        real& dlambda,
        bool& step_accepted
    ) const
    {
        const auto* self = this;
        auto rhs = KOKKOS_LAMBDA(const real in[8], real out[8]) {
            self->compute_rhs(in, out);
        };
        rk_detail::rk45_step<8>(state, dlambda, step_accepted, rhs);
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
        real x = state[IX];
        real y = state[IY];
        real z = state[IZ];
        real RSQ = SQR(x) + SQR(y) + SQR(z);
        real photon_distance = sqrt(0.5 * (RSQ - SQR(a_BH) + sqrt(SQR(RSQ - SQR(a_BH)) + 4.0 * SQR(a_BH * z))));
        if (photon_distance <= r_horizon || photon_distance > 1.5 * camera_distance) {
            photon_terminate(idx) = true;
            return;}
        
        if (integrator == IntegratorType::RK4) {
            rk4_step(state, photon_dlambda(idx));
        }
        else if (integrator == IntegratorType::RK45) {
            bool step_accepted = false;
            int max_attempts = 20;
            int attempts = 0;
            while (!step_accepted && attempts < max_attempts) {
                rk45_step(state, photon_dlambda(idx), step_accepted);
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
        // Add emissivity to intensity
        //real intensity = photon_I(idx);
        //add_emissivity(state, intensity);
        //photon_I(idx) = intensity;
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
    timers.AddTimer({"Geodesic Integration", "Norm Calculation", "MPI Counts Send/Recv", "Active Photons Calc",  "Output"});
    for (auto current_step = 0; current_step < max_steps; current_step++)
    {
        timers.BeginTimer("Geodesic Integration");
        Geodesic_cartesian_kerr_schild step_functor(photons, r, theta, phi, rho, Tgas,
            r_min, r_max, theta_min, theta_max, phi_min, phi_max, dlog_r, nr, ntheta, nphi,
            R_HORIZON, scattering_model, static_cast<std::size_t>(current_step), integrator);
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