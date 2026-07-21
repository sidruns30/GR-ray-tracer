#include "utils.hpp"

// Grid limits
size_t nr = 0;
size_t ntheta = 0;
size_t nphi = 0;

// Black hole parameters. Defaults below are used verbatim unless overridden
// by the runtime TOML file (see src/input/toml_config.hpp).
real M_BH = 1.0;
real a_BH = 1.0;
real R_HORIZON = M_BH + Kokkos::sqrt(M_BH * M_BH - a_BH * a_BH);

// Image-plane camera setup
real camera_theta = PI / 4.0;
real camera_phi = PI / 4.0;
real plane_dim1 = 20.0;
real plane_dim2 = 20.0;

// Shared camera parameter
real camera_distance = 10000.0;

// Photon count / run length
std::uint64_t nphotons = 20;
int max_steps = 10000;

// Geodesic integration control
real termination_percent = 0.9;
real termination_r_min = R_HORIZON;
real termination_r_max = 1.5 * camera_distance;
real dlambda = 0.1;
real atol_default = 1e-8;
real rtol_default = 1e-8;

// Output cadence
size_t output_interval = 1;
size_t output_stride = 1;
