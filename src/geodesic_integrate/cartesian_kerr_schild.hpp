/*
    Integrate the geodesic equations in Cartesian Kerr-Schild coordinates

    Inputs: Kokkos::View<double**> photons("camera_pixels", photons_per_process, 9);
            Each row corresponds to a photon with:
            (t, x, y, z, k_t, k_x, k_y, k_z, intensity)

            Kokkos::View<double***> r, theta, phi, rho, bsqr, pgas, Tgas, ug;
            Kokkos::View<double****> bu, uu;
            Compute emmisivity to the intensity and forward integrate
            Use RK4 step for integration

*/
#pragma once
#include "../utils.hpp"
#include "../utils.cpp"
#include "../input/load_python_arrays.hpp"
#include "../output/write_output.hpp"

struct GeodesicRK4_carteisian_kerr_schild {
    Kokkos::View<double**> photons;
    Kokkos::View<double*> r, theta, phi;
    Kokkos::View<double***> rho, bsqr, pgas, Tgas;
    Kokkos::View<double****> bu, uu;
    double dt;
    double r_min, r_max, theta_min, theta_max, phi_min, phi_max, dlog_r;
    size_t nr, ntheta, nphi;

    GeodesicRK4_carteisian_kerr_schild(
      Kokkos::View<double**> photons_,
      const Kokkos::View<double*>& r_,
      const Kokkos::View<double*>& theta_,
      const Kokkos::View<double*>& phi_,
      const Kokkos::View<double***>& rho_,
      const Kokkos::View<double***>& bsqr_,
      const Kokkos::View<double***>& pgas_,
      const Kokkos::View<double***>& Tgas_,
      const Kokkos::View<double****>& bu_,
      const Kokkos::View<double****>& uu_,
      double dt_, 
      double r_min_, double r_max_, double theta_min_, double theta_max_,
      double phi_min_, double phi_max_, double dlog_r_, 
      size_t nr_, size_t ntheta_, size_t nphi_)
      : photons(photons_), r(r_), theta(theta_), phi(phi_), rho(rho_), bsqr(bsqr_),
        pgas(pgas_), Tgas(Tgas_), bu(bu_), uu(uu_), dt(dt_),
        r_min(r_min_), r_max(r_max_), theta_min(theta_min_), theta_max(theta_max_),
        phi_min(phi_min_), phi_max(phi_max_), dlog_r(dlog_r_),
        nr(nr_), ntheta(ntheta_), nphi(nphi_) {};


    // Emissivity calculation can be added here as needed
    KOKKOS_INLINE_FUNCTION
    void add_emissivity(double state[8], double& intensity) const {
        // Find the nearest cell in the grid and add emissivity to intensity
        double x = state[IX];
        double y = state[IY];
        double z = state[IZ];
        double RSQ = SQR(x) + SQR(y) + SQR(z);
        double r = sqrt(0.5 * (RSQ - SQR(a_BH) + sqrt(SQR(RSQ - SQR(a_BH)) + 4.0 * SQR(a_BH * z))));
        double theta = acos(z / r);
        double phi = atan2(y, x) - atan2(a_BH, r);
        phi = fmod(phi, 2.0*PI);
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
            intensity += rho(ir, itheta, iphi) * QUAD(Tgas(ir, itheta, iphi));
        }
        else
        {
            intensity += 0.0;
        }
    }
    
    KOKKOS_INLINE_FUNCTION
    void compute_inverse_metric(const double X[4], double ginv[4][4]) const {
        for (int a=0;a<4;++a) for (int b=0;b<4;++b) ginv[a][b] = 0.0;

        double x = X[1];
        double y = X[2];
        double z = X[3];
        double RSQ = SQR(x) + SQR(y) + SQR(z);
        double r = sqrt(0.5 * (RSQ - SQR(a_BH) + sqrt(SQR(RSQ - SQR(a_BH)) + 4.0 * SQR(a_BH * z))));

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

    KOKKOS_INLINE_FUNCTION
    void compute_inverse_metric_deriv(const double X[4],
                                      double dginv[4][4][4]) const
    {
        double ginv[4][4];
        double ginv_current[4][4];
        compute_inverse_metric(X, ginv_current);
        // Second order finite differencing for metric
        for (int mu=0; mu<4; ++mu) {
            double X_ph[4] = {X[0], X[1], X[2], X[3]};
            double X_mh[4] = {X[0], X[1], X[2], X[3]};
            X_ph[mu] += metric_derivative_h;
            X_mh[mu] -= metric_derivative_h;
            compute_inverse_metric(X_ph, ginv);
            double ginv_ph[4][4];
            for (int a=0;a<4;++a) for (int b=0;b<4;++b) ginv_ph[a][b] = ginv[a][b];
            compute_inverse_metric(X_mh, ginv);
            double ginv_mh[4][4];
            for (int a=0;a<4;++a) for (int b=0;b<4;++b) ginv_mh[a][b] = ginv[a][b];
            for (int a=0;a<4;++a) {
                for (int b=0;b<4;++b) {
                    dginv[mu][a][b] = (ginv_ph[a][b] - ginv_mh[a][b]) / (2.0 * metric_derivative_h);
                }
            }
        }
        return;
    }


    KOKKOS_INLINE_FUNCTION
    void compute_rhs(const double state[8], double dstates[8]) const {
        double x[4];
        x[0] = state[IT];
        x[1] = state[IX];
        x[2] = state[IY];
        x[3] = state[IZ];

        double p_cov[4];
        p_cov[0] = state[IKT];
        p_cov[1] = state[IKX];
        p_cov[2] = state[IKY];
        p_cov[3] = state[IKZ];

        double ginv[4][4];
        double dginv[4][4][4];
        compute_inverse_metric(x, ginv);
        compute_inverse_metric_deriv(x, dginv);

        double p_contra[4];
        for (int mu=0; mu<4; ++mu) {
          double sum = 0.0;
          for (int nu=0; nu<4; ++nu) sum += ginv[mu][nu] * p_cov[nu];
          p_contra[mu] = sum;
        }

        for (int mu=0; mu<4; ++mu) dstates[mu] = p_contra[mu];

        // dp_mu/dlambda = -1/2 * (∂_mu g^{alpha beta}) p_alpha p_beta
        for (int mu=0; mu<4; ++mu) {
          double sum = 0.0;
          for (int alpha=0; alpha<4; ++alpha) {
            for (int beta=0; beta<4; ++beta) {
              sum += dginv[mu][alpha][beta] * p_cov[alpha] * p_cov[beta];
            }
          }
          dstates[4 + mu] = -0.5 * sum;
        }
        return;
    }

    KOKKOS_INLINE_FUNCTION
    void rk4_step(double state[8], double dt_local) const {
      double k1[8], k2[8], k3[8], k4[8];
      double tmp[8];

      compute_rhs(state, k1);

      for (int i=0;i<8;++i) tmp[i] = state[i] + 0.5 * dt_local * k1[i];
      compute_rhs(tmp, k2);

      for (int i=0;i<8;++i) tmp[i] = state[i] + 0.5 * dt_local * k2[i];
      compute_rhs(tmp, k3);

      for (int i=0;i<8;++i) tmp[i] = state[i] + dt_local * k3[i];
      compute_rhs(tmp, k4);

      for (int i=0;i<8;++i)
        state[i] += (dt_local / 6.0) * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]);
        return;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int idx, int &count) const {
    // Increment counter if photon is outside bounds
    double photon_distance = sqrt(SQR(photons(idx, IX)) + SQR(photons(idx, IY)) + SQR(photons(idx, IZ)));
    if (photon_distance > 1.1 * camera_distance || photon_distance < 0.9 * M_BH) {
        count += 1;
    }
    else{
        // Local state
        double state[8];
        for (int i = 0; i < 8; ++i) state[i] = photons(idx, i);
        rk4_step(state, dt);
        for (int i = 0; i < 8; ++i) photons(idx, i) = state[i];
        // Add emissivity to intensity
        double intensity = photons(idx, IW);
        add_emissivity(state, intensity);
        photons(idx, IW) = intensity;
    }
  }
};

float take_rk4_step(
    Kokkos::View<double**> photons,
    const Kokkos::View<double*>& r,
    const Kokkos::View<double*>& theta,
    const Kokkos::View<double*>& phi,
    const Kokkos::View<double***>& rho,
    const Kokkos::View<double***>& bsqr,
    const Kokkos::View<double***>& pgas,
    const Kokkos::View<double***>& Tgas,
    const Kokkos::View<double****>& bu,
    const Kokkos::View<double****>& uu,
    const double dtau)
{
  // Photons: N x 8
  const size_t photons_per_process = photons.extent(0);
  if (photons_per_process == 0) return 0.0f;
  // Launch parallel_for
  GeodesicRK4_carteisian_kerr_schild functor(photons, r, theta, phi, rho, bsqr, pgas, Tgas, bu, uu, dtau,
    r_min, r_max, theta_min, theta_max, phi_min, phi_max, dlog_r, nr, ntheta, nphi);
  int out_of_bounds = 0;
    Kokkos::parallel_reduce(
    "integrate_geodesics_rk4",
    Kokkos::RangePolicy<>(0, (int)photons_per_process),
    functor,
    out_of_bounds
    );
  Kokkos::fence();
  return (float)out_of_bounds / (float)photons_per_process;
}

// Perform geodesic integration until termination condition met
void integrate_geodesics(
    Kokkos::View<double**> photons,
    const Kokkos::View<double*>& r,
    const Kokkos::View<double*>& theta,
    const Kokkos::View<double*>& phi,
    const Kokkos::View<double***>& rho,
    const Kokkos::View<double***>& bsqr,
    const Kokkos::View<double***>& pgas,
    const Kokkos::View<double***>& Tgas,
    const Kokkos::View<double****>& bu,
    const Kokkos::View<double****>& uu,
    const double dtau, 
    const int rank)
{
    if (verbose && rank == 0) {
        std::cout << "Starting geodesic integration..." << std::endl;
    }
    float termination_fraction = 0.0f;
    int step = 0;
    do {
        termination_fraction = take_rk4_step(photons, r, theta, phi, rho, bsqr, pgas, Tgas, bu, uu, dtau);
        if (verbose) {
            if (step % 100 == 0 && rank == 0) {
                std::cout << "After step " << step << ", termination fraction: " << termination_fraction << std::endl;
            }
            if (step % output_interval == 0) {
                Kokkos::fence();
                auto all_photons = gather_photons(photons);
                if (rank == 0) {
                    write_photon_output(all_photons, step);
                }
            }
            step++;
        }
    } while (termination_fraction < termination_percent);
    std::cout << "Rank " << rank << ": Geodesic integration completed in " << step << " steps." << std::endl;
    return;
}