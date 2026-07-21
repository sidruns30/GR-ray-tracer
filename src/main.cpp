#include "utils.hpp"
#include "simulation_options.hpp"
#include "input/toml_config.hpp"
#include "radiative_transfer/initialize_photons.hpp"
#include "geodesic_integrate/integrate_cart_ks.hpp"
#include "output/display.hpp"

#include <limits>


int main(int argc, char* argv[]) {
    SimulationOptions options;
    try {
        options = parse_simulation_options(argc, argv);
    } catch (const std::exception& e) {
        ERROR(std::string("Failed to parse command-line arguments: ") + e.what());
        return 1;
    }
    output_directory = options.output_dir;
    std::string output_variables(default_output_variables);
    PhotonGenerationConfig photon_generation;
    UnitConversions units;
    DomainDecompositionSpec decomposition_spec;

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
            toml_config::load_and_apply_toml_config(
                options.config_path, output_variables, photon_generation, units, decomposition_spec);
            if (mpi_rank == 0) {
                INFO("Loaded simulation config from " + options.config_path);
            }
        } catch (const std::exception& e) {
            if (mpi_rank == 0) {
                ERROR(std::string("Failed to load config file: ") + e.what());
            }
            MPI_Finalize();
            return 1;
        }
    }

    try {
        photon_generation.validate();
        units.validate();
        decomposition_spec.resolve_partitions(mpi_size);
    } catch (const std::exception& e) {
        if (mpi_rank == 0) ERROR(std::string("Invalid runtime configuration: ") + e.what());
        MPI_Finalize();
        return 1;
    }

    if (options.output_variables) {
        output_variables = *options.output_variables;
    }
    OutputSelection output_selection;
    try {
        output_selection = OutputSelection::parse(output_variables);
    } catch (const std::exception& e) {
        if (mpi_rank == 0) {
            ERROR(std::string("Invalid output variable selection: ") + e.what());
        }
        MPI_Finalize();
        return 1;
    }

    if (options.vacuum && options.enable_scattering) {
        if (mpi_rank == 0) {
            WARN("--vacuum was passed together with --scatter; disabling scattering since it has no "
                 "medium to scatter off of without grid/MHD data.");
        }
        options.enable_scattering = false;
    }

    NumpyFieldPaths numpy_paths;
    if (!options.vacuum) {
        try {
            numpy_paths = discover_numpy_field_paths(options.numpy_dir);
        } catch (const std::exception& e) {
            if (mpi_rank == 0) {
                ERROR(std::string("Invalid NumPy input: ") + e.what());
                ERROR("Generate example data with: python3 src/create_example_data.py --output-dir <dir>, "
                      "then select it with --numpy-dir <dir>.");
            }
            MPI_Finalize();
            return 1;
        }
    }

    // Initialize Kokkos
    Kokkos::initialize(argc, argv);
    {
        Timers timers(max_steps, output_interval);
        NumpyFieldViews fields; // left default-constructed (empty views) in --vacuum mode

        if (!options.vacuum) {
            timers.AddTimer("Load HAMR Data");
            timers.BeginTimer("Load HAMR Data");
            try {
                fields = load_numpy_field_bundle(
                    numpy_paths, decomposition_spec, mpi_rank, mpi_size);
            } catch (const std::exception& e) {
                if (mpi_rank == 0) {
                    ERROR(std::string("Failed to load input grid data: ") + e.what());
                }
                Kokkos::finalize();
                MPI_Finalize();
                return 1;
            }
            timers.EndTimer("Load HAMR Data");

            if (mpi_rank == 0) {
                const auto partitions = fields.decomposition.partitions;
                INFO("Global grid dimensions: nr=" + std::to_string(fields.global_extents[0]) +
                     ", ntheta=" + std::to_string(fields.global_extents[1]) +
                     ", nphi=" + std::to_string(fields.global_extents[2]), Colors::cyan);
                INFO("MPI domain decomposition [r, theta, phi] = [" +
                     std::to_string(partitions[0]) + ", " + std::to_string(partitions[1]) + ", " +
                     std::to_string(partitions[2]) + "]", Colors::cyan);
                INFO("r coordinate from " + std::to_string(r_min) + " to " + std::to_string(r_max), Colors::hotpink);
                INFO("theta coordinate from " + std::to_string(theta_min) + " to " + std::to_string(theta_max), Colors::hotpink);
                INFO("phi coordinate from " + std::to_string(phi_min) + " to " + std::to_string(phi_max), Colors::hotpink);
            }
        } else if (mpi_rank == 0) {
            INFO("Running in --vacuum mode: no grid/MHD data loaded, scattering disabled.", Colors::cyan);
        }

        const bool camera_mode = use_pinhole_camera || use_image_camera;
        if (!camera_mode && options.vacuum) {
            if (mpi_rank == 0) ERROR("Fluid-frame photon generation requires grid data; remove --vacuum.");
            Kokkos::finalize();
            MPI_Finalize();
            return 1;
        }

        std::uint64_t local_photon_count = 0;
        if (camera_mode) {
            local_photon_count = static_cast<std::uint64_t>(
                nphotons / mpi_size + (mpi_rank < nphotons % mpi_size ? 1 : 0));
        } else {
            local_photon_count = static_cast<std::uint64_t>(fields.r.extent(0)) *
                fields.theta.extent(0) * fields.phi.extent(0) *
                static_cast<std::uint64_t>(photon_generation.superphotons_per_cell);
        }

        std::uint64_t global_photon_count = 0;
        MPI_Allreduce(&local_photon_count, &global_photon_count, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
        if (global_photon_count > 1000000000ULL ||
            local_photon_count > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
            if (mpi_rank == 0) ERROR("Generated photon count exceeds the supported ID or local-index range.");
            Kokkos::finalize();
            MPI_Finalize();
            return 1;
        }
        nphotons = static_cast<int>(global_photon_count);
        if (mpi_rank == 0) {
            INFO("Global superphoton count: " + std::to_string(global_photon_count), Colors::cyan);
        }
        std::uint64_t global_id_offset_64 = 0;
        MPI_Exscan(&local_photon_count, &global_id_offset_64, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
        if (mpi_rank == 0) global_id_offset_64 = 0;
        const auto global_id_offset = static_cast<std::uint32_t>(global_id_offset_64);
        const int photons_per_process = static_cast<int>(local_photon_count);
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
                global_id_offset,
                photon_generation,
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
                global_id_offset,
                photon_generation,
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
                mpi_rank,
                global_id_offset,
                fields.r, fields.theta, fields.phi,
                fields.density, fields.temperature, fields.velocity, fields.magnetic,
                photon_generation,
                units,
                photons
                );
            timers.EndTimer("Initialize User-Defined Photons");
        }

        if (mpi_rank == 0) {
            INFO("=== Starting Geodesic Integration ===", Colors::cyan);
        }

        // Integrate geodesics
        integrate_geodesics(
            photons,
            fields,
            units,
            mpi_rank, timers,
            output_selection,
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
    return 0;
}
