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
// used when the required TOML file omits a key.
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
    Kokkos::View<std::uint64_t*> id;
    Kokkos::View<real*> frequency;
    // Energy -p.u in the emitting fluid frame. Image mode uses its ratio to
    // the camera-frame energy to redshift the emitted frequency on arrival.
    Kokkos::View<real*> emission_frame_energy;
    Kokkos::View<real*> x0, x1, x2, x3;
    Kokkos::View<real*> k0, k1, k2, k3;
    Kokkos::View<real*> I, Q, U, V, dlambda;
    Kokkos::View<bool*> terminate;
    // Disk packets use Disk. Image packets move through Backward, Forward,
    // and Arrived; Rejected identifies rays that never return to the camera.
    Kokkos::View<std::uint8_t*> phase;
    // Fixed camera-screen coordinates used to place each photon in observation images.
    Kokkos::View<real*> theta_disp, phi_disp;
    Photons(int size, const std::string& label = "photons")
        : id("id", size),
          frequency("frequency", size),
          emission_frame_energy("emission_frame_energy", size),
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
          phase("phase", size),
          theta_disp("theta_disp", size),
          phi_disp("phi_disp", size)
    {
        (void)label;
    }
    // Host mirrors are allocated lazily because production runs may contain
    // millions of photons and do not otherwise need a duplicate host copy.
    Kokkos::View<std::uint64_t*, Kokkos::HostSpace> id_host;
    Kokkos::View<real*, Kokkos::HostSpace> frequency_host;
    Kokkos::View<real*, Kokkos::HostSpace> emission_frame_energy_host;
    Kokkos::View<real*, Kokkos::HostSpace> x0_host, x1_host, x2_host, x3_host;
    Kokkos::View<real*, Kokkos::HostSpace> k0_host, k1_host, k2_host, k3_host;
    Kokkos::View<real*, Kokkos::HostSpace> I_host, Q_host, U_host, V_host, dlambda_host;
    Kokkos::View<bool*, Kokkos::HostSpace> terminate_host;
    Kokkos::View<std::uint8_t*, Kokkos::HostSpace> phase_host;
    Kokkos::View<real*, Kokkos::HostSpace> theta_disp_host, phi_disp_host;

    template <typename DeviceView, typename HostView>
    static void copy_field_if(bool needed, const DeviceView& device, HostView& host) {
        if (!needed) return;
        if (!host.data()) host = Kokkos::create_mirror_view(device);
        Kokkos::deep_copy(host, device);
    }

    void copy_to_host(const OutputSelection& selection) {
        copy_field_if(selection.contains("id"), id, id_host);
        copy_field_if(selection.contains("frequency"), frequency, frequency_host);
        copy_field_if(selection.contains("emission_frame_energy"),
                      emission_frame_energy, emission_frame_energy_host);
        copy_field_if(selection.contains("x0"), x0, x0_host);
        copy_field_if(selection.contains("x1"), x1, x1_host);
        copy_field_if(selection.contains("x2"), x2, x2_host);
        copy_field_if(selection.contains("x3"), x3, x3_host);
        copy_field_if(selection.contains("k0"), k0, k0_host);
        copy_field_if(selection.contains("k1"), k1, k1_host);
        copy_field_if(selection.contains("k2"), k2, k2_host);
        copy_field_if(selection.contains("k3"), k3, k3_host);
        copy_field_if(selection.contains("I"), I, I_host);
        copy_field_if(selection.contains("Q"), Q, Q_host);
        copy_field_if(selection.contains("U"), U, U_host);
        copy_field_if(selection.contains("V"), V, V_host);
        copy_field_if(selection.contains("dlambda"), dlambda, dlambda_host);
        copy_field_if(selection.contains("terminate"), terminate, terminate_host);
        copy_field_if(selection.contains("phase"), phase, phase_host);
        copy_field_if(selection.contains("theta_disp"), theta_disp, theta_disp_host);
        copy_field_if(selection.contains("phi_disp"), phi_disp, phi_disp_host);
    }


    void copy_observation_to_host() {
        copy_field_if(true, frequency, frequency_host);
        copy_field_if(true, I, I_host);
        copy_field_if(true, Q, Q_host);
        copy_field_if(true, U, U_host);
        copy_field_if(true, V, V_host);
        copy_field_if(true, theta_disp, theta_disp_host);
        copy_field_if(true, phi_disp, phi_disp_host);
        copy_field_if(true, phase, phase_host);
    }
};

enum class PhotonPhase : std::uint8_t {
    Disk = 0,
    ImageBackward = 1,
    ImageForward = 2,
    ImageArrived = 3,
    Rejected = 4
};

// Parameters for the grid data -- assume that is uniform in log-spherical coords
extern size_t nr;
extern size_t ntheta;
extern size_t nphi;

// Image-plane camera parameters. Disk mode does not use a camera.
extern real camera_theta;
extern real camera_phi;
extern real plane_dim1;
extern real plane_dim2;
extern real camera_distance;   // Distance of camera from origin in rg/c

// Global photon count and its supported limit. IDs use 64 bits so production
// runs can distribute up to 100 billion uniquely numbered packets across GPUs.
inline constexpr std::uint64_t max_global_photons = 100000000000ULL;
extern std::uint64_t nphotons;
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
extern size_t output_stride;
inline std::string output_directory = "./output";
inline std::string simulation_name = "output";
