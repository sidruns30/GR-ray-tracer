#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <mpi.h>
#include <omp.h>
#include <Kokkos_Core.hpp>
#include "cnpy.h"

// Global variables: don't change these
inline constexpr double PI = 3.14159265358979323846;
inline constexpr double M_BH = 1.0; // Black hole mass in geometric units
inline constexpr double a_BH = 0.9;    // Black hole spin parameter (0 <= a < 1)
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
constexpr double SQR(const double x) { return x * x; }
KOKKOS_INLINE_FUNCTION
constexpr double CUBE(const double x) { return x * x * x; }
KOKKOS_INLINE_FUNCTION
constexpr double QUAD(const double x) { return x * x * x * x; }


// User defined variables
extern const bool verbose = true;

// Parameters for the grid data -- assume that is uniform in log-spherical coords
extern size_t nr;
extern size_t ntheta;
extern size_t nphi;
extern double r_min;
extern double r_max;
extern double theta_min;
extern double theta_max;
extern double phi_min;
extern double phi_max;
extern double dlog_r;

// Parameters for a pinhole camera setup
extern const bool use_pinhole_camera = false;
extern const int nphotons_pinhole = 1000;
extern const double camera_theta = M_PI / 4.0;  // Polar angle of camera position
extern const double camera_phi = M_PI / 4.0;    // Azimuthal angle of camera position
extern const double target_rmin = 0.0;          // Minimum radius for photon targeting
extern const double target_rmax = 2.0;         // Maximum radius for photon targeting

// Parameters for image camera setup
extern const bool use_image_camera = !use_pinhole_camera;
extern const int image_width = 3000;   // Image width in pixels
extern const int image_height = 3000;  // Image height in pixels
extern const double plane_theta = 0. * M_PI / 2.0;  // Polar angle of image plane
extern const double plane_phi = M_PI / 4.0;    // Azimuthal angle of image plane
extern const double plane_area = 400.0;        // Area of image plane in (rg/c)^2

extern const double camera_distance = 50.0;   // Distance of camera from origin in rg/c
extern const int nphotons = use_image_camera ? (image_width * image_height) : nphotons_pinhole;

// Parameters for geodesic integration
extern const double metric_derivative_h = 1e-6;
extern const float termination_percent = 0.9;
extern const double dtau = 0.1;

// Parameters for output
extern const size_t output_interval = 1000;
extern const std::string output_directory = "/home/siddhant/scratch/GR-ray-tracer/output/";