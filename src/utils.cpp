#include "utils.hpp"

// Grid limits
size_t nr = 0;
size_t ntheta = 0;
size_t nphi = 0;
real r_min = 0.0;
real r_max = 0.0;
real theta_min = 0.0;
real theta_max = 0.0;
real phi_min = 0.0;
real phi_max = 0.0;
real dlog_r = 0.0;

// Black hole parameters. Defaults below are used verbatim unless overridden
// by a --config TOML file (see src/input/toml_config.hpp).
real M_BH = 1.0;
real a_BH = 1.0;
real R_HORIZON = M_BH + sqrt(M_BH * M_BH - a_BH * a_BH);

// Pinhole camera setup
bool use_pinhole_camera = false;
real camera_theta = PI / 4.0;
real camera_phi = PI / 4.0;
real target_rmin = 0.0;
real target_rmax = 2.0;

// Image camera setup
bool use_image_camera = true;
real plane_dim1 = 20.0;
real plane_dim2 = 20.0;
real plane_theta = PI / 2.0;
real plane_phi = PI / 4.0;

// Shared camera parameter
real camera_distance = 10000.0;

// Photon count / run length
int nphotons = 20;
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
