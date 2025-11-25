#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <mpi.h>
#include <omp.h>
#include <Kokkos_Core.hpp>
#include "cnpy.h"
#include <algorithm>
#include <random>

using real = double;
// Global variables: don't change these
inline constexpr real PI = 3.14159265358979323846;
inline constexpr real M_BH = 1.0; // Black hole mass in geometric units
inline constexpr real a_BH = 1.0;    // Black hole spin parameter (0 <= a < 1)
inline const real R_HORIZON = M_BH + sqrt(M_BH * M_BH - a_BH * a_BH); // Event horizon radius
inline constexpr size_t IT = 0;
inline constexpr size_t IX = 1;
inline constexpr size_t IY = 2;
inline constexpr size_t IZ = 3;
inline constexpr size_t IKT = 4;
inline constexpr size_t IKX = 5;
inline constexpr size_t IKY = 6;
inline constexpr size_t IKZ = 7;
inline constexpr size_t IW = 8;
KOKKOS_INLINE_FUNCTION
constexpr real SQR(const real x) { return x * x; }
KOKKOS_INLINE_FUNCTION
constexpr real CUBE(const real x) { return x * x * x; }
KOKKOS_INLINE_FUNCTION
constexpr real QUAD(const real x) { return x * x * x * x; }
MPI_Datatype MPI_real = (std::is_same<real, double>::value) ? MPI_DOUBLE : MPI_FLOAT;

struct Photons {
    Kokkos::View<real*> x0, x1, x2, x3;
    Kokkos::View<real*> k0, k1, k2, k3;
    Kokkos::View<real*> I, dlambda;
    Kokkos::View<bool*> terminate;
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
          dlambda("dlambda", size),
          terminate("terminate", size)
    {}
    // Mirror Views
    Kokkos::View<real*, Kokkos::HostSpace> x0_host, x1_host, x2_host, x3_host;
    Kokkos::View<real*, Kokkos::HostSpace> k0_host, k1_host, k2_host, k3_host;
    Kokkos::View<real*, Kokkos::HostSpace> I_host, dlambda_host;
    Kokkos::View<bool*, Kokkos::HostSpace> terminate_host;

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
        dlambda_host  = Kokkos::create_mirror_view(dlambda);
        terminate_host = Kokkos::create_mirror_view(terminate);
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
        Kokkos::deep_copy(dlambda_host, dlambda);
        Kokkos::deep_copy(terminate_host, terminate);
    }
};

// Debugging
extern const bool verbose = true;

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
extern const bool use_pinhole_camera = false;
extern const real camera_theta = M_PI / 4.0;  // Polar angle of camera position
extern const real camera_phi = M_PI / 4.0;    // Azimuthal angle of camera position
extern const real target_rmin = 0.0;          // Minimum radius for photon targeting
extern const real target_rmax = 2.0;         // Maximum radius for photon targeting

// Parameters for image camera setup
extern const bool use_image_camera = true;
extern const real plane_dim1 = 20.0;        // Dimension 1 of image plane in (rg/c)
extern const real plane_dim2 = 20.0;        // Dimension 2 of image plane in (rg/c)
extern const real plane_theta = M_PI / 2.0;  // Polar angle of image plane
extern const real plane_phi = M_PI / 4.0;    // Azimuthal angle of image plane

// Parameters for both
extern const real camera_distance = 10000.0;   // Distance of camera from origin in rg/c

// Number of photons to trace
extern const int nphotons = 20;
extern const int max_steps = 10000;

// Parameters for geodesic integration
extern const real metric_derivative_h = 1e-6;
extern const real termination_percent = 0.9;

enum class IntegratorType : int { RK4 = 0, RK45 = 1 };
inline constexpr IntegratorType integrator_type = IntegratorType::RK45;
inline constexpr bool USE_RK45 = (integrator_type == IntegratorType::RK45);
inline constexpr bool USE_RK4  = (integrator_type == IntegratorType::RK4);

// RK45 parameters
extern const real dlambda = 0.1; // Siddhant: this is just an initial guess, will be adapted
extern const real tol = 1e-8;
// Siddhant: for adaptive step sizing, better not change. 
extern const real safety = 0.9;
extern const real min_scale = 0.01;
extern const real max_scale = 10.0;

// Parameters for output
extern const size_t output_interval = 1;
extern const std::string output_directory = "/home/siddhant/scratch/GR-ray-tracer/output/";