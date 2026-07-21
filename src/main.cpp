#include "utils.hpp"
#include "simulation_options.hpp"
#include "input/toml_config.hpp"
#include "radiative_transfer/initialize_photons.hpp"
#include "geodesic_integrate/integrate_cart_ks.hpp"
#include "output/display.hpp"


int main(int argc, char* argv[]) {
    SimulationOptions options;
    try {
        options = parse_simulation_options(argc, argv);
    } catch (const std::exception& e) {
        ERROR(std::string("Failed to parse command-line arguments: ") + e.what());
        return 1;
    }
    output_directory = options.output_dir;
    // write_output.hpp/observation.hpp build file paths by concatenating
    // output_directory directly with filenames (no path separator inserted),
    // so a --output-dir without a trailing slash would otherwise spill files
    // into the current directory with a mashed-together name instead of the
    // intended subdirectory.
    if (!output_directory.empty() && output_directory.back() != '/') {
        output_directory += '/';
    }

    // Initialize MPI
    int mpi_rank, mpi_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    if (mpi_rank == 0) {
        INIT("=== GR Ray-Trace Initialization ===");
        INFO("MPI processes: " + std::to_string(mpi_size));
    }

    if (!options.config_path.empty()) {
        try {
            toml_config::load_and_apply_toml_config(options.config_path);
        } catch (const std::exception& e) {
            if (mpi_rank == 0) {
                ERROR(std::string("Failed to load config file: ") + e.what());
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (options.vacuum && options.enable_scattering) {
        if (mpi_rank == 0) {
            WARN("--vacuum was passed together with --scatter; disabling scattering since it has no "
                 "medium to scatter off of without grid/MHD data.");
        }
        options.enable_scattering = false;
    }

    // Initialize Kokkos
    Kokkos::initialize(argc, argv);
    {
        Timers timers(max_steps, output_interval);
        NumpyFieldViews fields; // left default-constructed (empty views) in --vacuum mode

        if (!options.vacuum) {
            timers.AddTimer("Load HAMR Data");
            timers.BeginTimer("Load HAMR Data");
            NumpyFieldPaths paths = options.fields;
            if (paths.r.empty()) paths.r = "./r.npy";
            if (paths.theta.empty()) paths.theta = "./theta.npy";
            if (paths.phi.empty()) paths.phi = "./phi.npy";
            if (paths.density.empty()) paths.density = "./rho.npy";
            if (paths.temperature.empty()) paths.temperature = "./Tgas.npy";
            if (paths.velocity.empty()) paths.velocity = "./vel.npy";
            if (paths.magnetic.empty()) paths.magnetic = "./mag.npy";

            try {
                fields = load_numpy_field_bundle(paths);
            } catch (const std::exception& e) {
                if (mpi_rank == 0) {
                    ERROR(std::string("Failed to load input grid data: ") + e.what());
                    ERROR("Expected grid files: " + paths.r + ", " + paths.theta + ", " + paths.phi + ", " +
                          paths.density + ", " + paths.temperature + ", " + paths.velocity + ", " + paths.magnetic);
                    ERROR("Generate example grid data with: python3 src/create_example_data.py --output-dir <dir>, "
                          "or point at real data with --grid-r/--grid-theta/--grid-phi/--density/--temperature/"
                          "--velocity/--magnetic, or pass --vacuum to skip grid data and run pure vacuum geodesics.");
                }
                Kokkos::finalize();
                MPI_Finalize();
                return 1;
            }
            timers.EndTimer("Load HAMR Data");

            if (verbose && mpi_rank == 0) {
                INFO("Grid dimensions: nr=" + std::to_string(nr) + ", ntheta=" + std::to_string(ntheta) + ", nphi=" +
                std::to_string(nphi), Colors::cyan);
                INFO("r coordinate from " + std::to_string(r_min) + " to " + std::to_string(r_max), Colors::hotpink);
                INFO("theta coordinate from " + std::to_string(theta_min) + " to " + std::to_string(theta_max), Colors::hotpink);
                INFO("phi coordinate from " + std::to_string(phi_min) + " to " + std::to_string(phi_max), Colors::hotpink);
            }
        } else if (mpi_rank == 0) {
            INFO("Running in --vacuum mode: no grid/MHD data loaded, scattering disabled.", Colors::cyan);
        }

        // Initialize a fraction of photons per mpi process for back tracing from camera
        const int photons_per_process = nphotons / mpi_size;
        Photons photons(photons_per_process, "photons_");

        if (use_pinhole_camera) {
            timers.AddTimer("Initialize Pinhole Camera");
            timers.BeginTimer("Initialize Pinhole Camera");
            initialize_photons_pinhole(
                photons_per_process,
                camera_distance,
                camera_theta,
                camera_phi,
                mpi_rank,
                mpi_size,
                photons
                );
            timers.EndTimer("Initialize Pinhole Camera");
            }

        else if (use_image_camera) {
            timers.AddTimer("Initialize Image Camera");
            timers.BeginTimer("Initialize Image Camera");
            initialize_photons_image_camera(
                photons_per_process,
                camera_distance,
                camera_theta,
                camera_phi,
                mpi_rank,
                mpi_size,
                photons
                );
            timers.EndTimer("Initialize Image Camera");
        }

        else 
        {
            timers.AddTimer("Initialize User-Defined Photons");
            timers.BeginTimer("Initialize User-Defined Photons");
            initialize_photons_user(
                photons_per_process,
                photons
                );
            timers.EndTimer("Initialize User-Defined Photons");
        }

        INFO("=== Starting Geodesic Integration ===", Colors::cyan);

        // Integrate geodesics
        integrate_geodesics(
            photons,
            fields.r, fields.theta, fields.phi, fields.density, fields.temperature,
            mpi_rank, timers,
            ScatteringModel{
                options.enable_scattering,
                options.scattering_optical_depth,
                options.scattering_albedo,
                12345u
            },
            options.integrator
            );
    }
    Kokkos::fence();
    // Finalize Kokkos
    Kokkos::finalize();
    
    // Finalize MPI
    MPI_Finalize();
    std::cout << "MPI finalize called from rank " << mpi_rank << std::endl;
    return 0;
}
