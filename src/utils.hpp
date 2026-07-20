#pragma once
#include <algorithm>
#include <cmath>
#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include <Kokkos_Core.hpp>

#include "cnpy.h"

using real = double;
inline constexpr real PI = 3.14159265358979323846;
// Physics/simulation parameters below are runtime-configurable (see
// src/input/toml_config.hpp) -- these are just their built-in defaults,
// used verbatim when no --config TOML file is given.
extern real M_BH; // Black hole mass in geometric units
extern real a_BH; // Black hole spin parameter; must satisfy |a_BH| <= M_BH (|a_BH| == M_BH is extremal)
extern real R_HORIZON; // Event horizon radius = M_BH + sqrt(M_BH^2 - a_BH^2), recomputed on config load
inline constexpr size_t IT = 0;
inline constexpr size_t IX = 1;
inline constexpr size_t IY = 2;
inline constexpr size_t IZ = 3;
inline constexpr size_t IKT = 4;
inline constexpr size_t IKX = 5;
inline constexpr size_t IKY = 6;
inline constexpr size_t IKZ = 7;
KOKKOS_INLINE_FUNCTION
constexpr real SQR(const real x) { return x * x; }
KOKKOS_INLINE_FUNCTION
constexpr real CUBE(const real x) { return x * x * x; }
KOKKOS_INLINE_FUNCTION
constexpr real QUAD(const real x) { return x * x * x * x; }
inline const MPI_Datatype MPI_real = (std::is_same<real, double>::value) ? MPI_DOUBLE : MPI_FLOAT;

struct Photons {
    Kokkos::View<real*> x0, x1, x2, x3;
    Kokkos::View<real*> k0, k1, k2, k3;
    Kokkos::View<real*> I, Q, U, V, dlambda;
    Kokkos::View<bool*> terminate;
    // Fixed camera-screen coordinates the photon was launched from (set once at
    // init time by initialize_photons.hpp) -- used by observation.hpp to bin
    // image_I/Q/U/V onto a stable camera-screen pixel grid, as opposed to the
    // photon's current (time-varying) x1/x2 position.
    Kokkos::View<real*> theta_disp, phi_disp;
    Photons(int size, const std::string& label = "photons")
        : x0("x0", size),
          x1("x1", size),
          x2("x2", size),
          x3("x3", size),
          k0("k0", size),
          k1("k1", size),
          k2("k2", size),
          k3("k3", size),
          I("I", size),
          Q("Q", size),
          U("U", size),
          V("V", size),
          dlambda("dlambda", size),
          terminate("terminate", size),
          theta_disp("theta_disp", size),
          phi_disp("phi_disp", size)
    {
        (void)label;
    }
    // Mirror Views
    Kokkos::View<real*, Kokkos::HostSpace> x0_host, x1_host, x2_host, x3_host;
    Kokkos::View<real*, Kokkos::HostSpace> k0_host, k1_host, k2_host, k3_host;
    Kokkos::View<real*, Kokkos::HostSpace> I_host, Q_host, U_host, V_host, dlambda_host;
    Kokkos::View<bool*, Kokkos::HostSpace> terminate_host;
    Kokkos::View<real*, Kokkos::HostSpace> theta_disp_host, phi_disp_host;

    void create_mirror_views() {
        x0_host = Kokkos::create_mirror_view(x0);
        x1_host = Kokkos::create_mirror_view(x1);
        x2_host = Kokkos::create_mirror_view(x2);
        x3_host = Kokkos::create_mirror_view(x3);
        k0_host = Kokkos::create_mirror_view(k0);
        k1_host = Kokkos::create_mirror_view(k1);
        k2_host = Kokkos::create_mirror_view(k2);
        k3_host = Kokkos::create_mirror_view(k3);
        I_host  = Kokkos::create_mirror_view(I);
        Q_host  = Kokkos::create_mirror_view(Q);
        U_host  = Kokkos::create_mirror_view(U);
        V_host  = Kokkos::create_mirror_view(V);
        dlambda_host  = Kokkos::create_mirror_view(dlambda);
        terminate_host = Kokkos::create_mirror_view(terminate);
        theta_disp_host = Kokkos::create_mirror_view(theta_disp);
        phi_disp_host = Kokkos::create_mirror_view(phi_disp);
    }

    void copy_to_host() {
        // If user forgot to allocate mirrors
        if (!x0_host.data()) {
            create_mirror_views();
        }
        Kokkos::deep_copy(x0_host, x0);
        Kokkos::deep_copy(x1_host, x1);
        Kokkos::deep_copy(x2_host, x2);
        Kokkos::deep_copy(x3_host, x3);
        Kokkos::deep_copy(k0_host, k0);
        Kokkos::deep_copy(k1_host, k1);
        Kokkos::deep_copy(k2_host, k2);
        Kokkos::deep_copy(k3_host, k3);
        Kokkos::deep_copy(I_host, I);
        Kokkos::deep_copy(Q_host, Q);
        Kokkos::deep_copy(U_host, U);
        Kokkos::deep_copy(V_host, V);
        Kokkos::deep_copy(dlambda_host, dlambda);
        Kokkos::deep_copy(terminate_host, terminate);
        Kokkos::deep_copy(theta_disp_host, theta_disp);
        Kokkos::deep_copy(phi_disp_host, phi_disp);
    }
};

// Debugging
inline constexpr bool verbose = true;

// Parameters for the grid data -- assume that is uniform in log-spherical coords
extern size_t nr;
extern size_t ntheta;
extern size_t nphi;
extern real r_min;
extern real r_max;
extern real theta_min;
extern real theta_max;
extern real phi_min;
extern real phi_max;
extern real dlog_r;

// Parameters for a pinhole camera setup
extern bool use_pinhole_camera;
extern real camera_theta;      // Polar angle of camera position
extern real camera_phi;        // Azimuthal angle of camera position
extern real target_rmin;       // Minimum radius for photon targeting
extern real target_rmax;       // Maximum radius for photon targeting
// Radius of the disk in the camera's local screen plane that initial photon
// directions are sampled over (see initialize_photons_pinhole). Shared with
// observation.hpp so image bins use the same fixed screen-space extent.
inline constexpr real pinhole_aperture_radius = 50.0;

// Parameters for image camera setup
extern bool use_image_camera;
extern real plane_dim1;        // Dimension 1 of image plane in (rg/c)
extern real plane_dim2;        // Dimension 2 of image plane in (rg/c)
extern real plane_theta;       // Polar angle of image plane
extern real plane_phi;         // Azimuthal angle of image plane

// Parameters for both
extern real camera_distance;   // Distance of camera from origin in rg/c

// Number of photons to trace
extern int nphotons;
extern int max_steps;

// Parameters for geodesic integration
extern real termination_percent;

// Photons terminate once their Kerr-Schild radius leaves [termination_r_min, termination_r_max].
// Default to R_HORIZON / 1.5*camera_distance respectively unless overridden (see toml_config.hpp).
extern real termination_r_min;
extern real termination_r_max;

enum class IntegratorType : int { RK4 = 0, RK45 = 1 };

// RK45 parameters
extern real dlambda; // Siddhant: this is just an initial guess, will be adapted
// Mixed relative/absolute error norm: err_i = |x5_i - x4_i| / (atol + rtol*|x_i|).
// atol dominates for near-zero components (e.g. momentum crossing zero), rtol for large ones (e.g. position ~1e4).
extern real atol_default;
extern real rtol_default;
// Siddhant: for adaptive step sizing, better not change. Not exposed via TOML config.
inline constexpr real safety = 0.9;
inline constexpr real min_scale = 0.01;
inline constexpr real max_scale = 10.0;

// Parameters for output
extern size_t output_interval;
inline std::string output_directory = "./output/";