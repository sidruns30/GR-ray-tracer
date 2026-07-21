#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <mpi.h>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "cnpy.h"
#include "output/output_variables.hpp"
#include "types.hpp"

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

#include "radiative_transfer/photon_generation.hpp"

struct Photons {
    Kokkos::View<std::uint32_t*> id;
    Kokkos::View<real*> frequency;
    Kokkos::View<real*> x0, x1, x2, x3;
    Kokkos::View<real*> k0, k1, k2, k3;
    Kokkos::View<real*> I, Q, U, V, dlambda;
    Kokkos::View<bool*> terminate;
    // Fixed camera-screen coordinates used to place each photon in observation images.
    Kokkos::View<real*> theta_disp, phi_disp;
    Photons(int size, const std::string& label = "photons")
        : id("id", size),
          frequency("frequency", size),
          x0("x0", size),
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
    // Host mirrors are allocated lazily because production runs may contain
    // millions of photons and do not otherwise need a duplicate host copy.
    Kokkos::View<std::uint32_t*, Kokkos::HostSpace> id_host;
    Kokkos::View<real*, Kokkos::HostSpace> frequency_host;
    Kokkos::View<real*, Kokkos::HostSpace> x0_host, x1_host, x2_host, x3_host;
    Kokkos::View<real*, Kokkos::HostSpace> k0_host, k1_host, k2_host, k3_host;
    Kokkos::View<real*, Kokkos::HostSpace> I_host, Q_host, U_host, V_host, dlambda_host;
    Kokkos::View<bool*, Kokkos::HostSpace> terminate_host;
    Kokkos::View<real*, Kokkos::HostSpace> theta_disp_host, phi_disp_host;

    template <typename DeviceView, typename HostView>
    static void copy_field_if(bool needed, const DeviceView& device, HostView& host) {
        if (!needed) return;
        if (!host.data()) host = Kokkos::create_mirror_view(device);
        Kokkos::deep_copy(host, device);
    }

    void copy_to_host(const OutputSelection& selection) {
        const bool observations = selection.needs_observation_products();
        copy_field_if(selection.contains("id"), id, id_host);
        copy_field_if(selection.contains("frequency") || observations, frequency, frequency_host);
        copy_field_if(selection.contains("x0") || observations, x0, x0_host);
        copy_field_if(selection.contains("x1"), x1, x1_host);
        copy_field_if(selection.contains("x2"), x2, x2_host);
        copy_field_if(selection.contains("x3"), x3, x3_host);
        copy_field_if(selection.contains("k0") || observations, k0, k0_host);
        copy_field_if(selection.contains("k1"), k1, k1_host);
        copy_field_if(selection.contains("k2"), k2, k2_host);
        copy_field_if(selection.contains("k3"), k3, k3_host);
        copy_field_if(selection.contains("I") || observations, I, I_host);
        copy_field_if(selection.contains("Q") || observations, Q, Q_host);
        copy_field_if(selection.contains("U") || observations, U, U_host);
        copy_field_if(selection.contains("V") || observations, V, V_host);
        copy_field_if(selection.contains("dlambda"), dlambda, dlambda_host);
        copy_field_if(selection.contains("terminate"), terminate, terminate_host);
        copy_field_if(selection.contains("theta_disp") || observations, theta_disp, theta_disp_host);
        copy_field_if(selection.contains("phi_disp") || observations, phi_disp, phi_disp_host);
    }
};

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
extern real dlambda; // Initial step size; RK45 adapts it independently for each photon.
// Mixed relative/absolute error norm: err_i = |x5_i - x4_i| / (atol + rtol*|x_i|).
// atol dominates for near-zero components (e.g. momentum crossing zero), rtol for large ones (e.g. position ~1e4).
extern real atol_default;
extern real rtol_default;
// Conservative step-size controller bounds shared by CPU and GPU backends.
inline constexpr real safety = 0.9;
inline constexpr real min_scale = 0.01;
inline constexpr real max_scale = 10.0;

// Parameters for output
extern size_t output_interval;
inline std::string output_directory = "./output";
