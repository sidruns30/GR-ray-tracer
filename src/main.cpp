#include "utils.hpp"
#include "simulation_options.hpp"
#include "input/load_python_arrays.hpp"
#include "input/toml_config.hpp"
#include "radiative_transfer/initialize_photons.hpp"
#include "geodesic_integrate/integrate_cart_ks.hpp"
#include "output/display.hpp"

#include <array>
#include <limits>


int main(int argc, char* argv[]) {
    std::string config_path;
    try {
        config_path = parse_config_path(argc, argv);
    } catch (const std::exception& e) {
        ERROR(e.what());
        return 1;
    }

    SimulationOptions options;
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

    try {
        toml_config::load_and_apply_toml_config(
            config_path, options, output_variables, photon_generation, units, decomposition_spec);
        output_directory = options.output_dir;
        if (mpi_rank == 0) {
            INFO("Loaded simulation config from " + config_path);
        }
    } catch (const std::exception& e) {
        if (mpi_rank == 0) {
            ERROR(std::string("Failed to load config file: ") + e.what());
        }
        MPI_Finalize();
        return 1;
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
            WARN("input.vacuum=true disables scattering because no grid/MHD medium is loaded.");
        }
        options.enable_scattering = false;
    }
    if (options.vacuum && (options.enable_absorption || options.enable_emission)) {
        if (mpi_rank == 0) {
            WARN("input.vacuum=true disables radiative transfer because no grid/MHD medium is loaded.");
        }
        options.enable_absorption = false;
        options.enable_emission = false;
    }
    if (options.vacuum) {
        if (mpi_rank == 0) {
            ERROR("simulation.mode image and disk both require the NumPy fluid/disk grid; "
                  "set input.vacuum=false.");
        }
        MPI_Finalize();
        return 1;
    }

    NumpyFieldPaths numpy_paths;
    if (!options.vacuum) {
        try {
            numpy_paths = discover_numpy_field_paths(options.numpy_dir);
        } catch (const std::exception& e) {
            if (mpi_rank == 0) {
                ERROR(std::string("Invalid NumPy input: ") + e.what());
                ERROR("Generate example data with: python3 src/create_example_data.py --output-dir <dir>, "
                      "then set input.numpy_dir in the TOML file.");
            }
            MPI_Finalize();
            return 1;
        }
    }

    // Initialize Kokkos
    Kokkos::initialize();
    {
        Timers timers(output_interval);
        NumpyFieldViews fields; // Empty views are intentional when input.vacuum is true.

        if (!options.vacuum) {
            timers.AddTimer("Load Numpy Data");
            timers.BeginTimer("Load Numpy Data");
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
            timers.EndTimer("Load Numpy Data");

            std::array<real, 3> global_coordinate_min{};
            std::array<real, 3> global_coordinate_max{};
            MPI_Allreduce(
                fields.coordinate_min.data(), global_coordinate_min.data(), 3,
                MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
            MPI_Allreduce(
                fields.coordinate_max.data(), global_coordinate_max.data(), 3,
                MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            if (mpi_rank == 0) {
                const auto partitions = fields.decomposition.partitions;
                INFO("Global grid dimensions: nr=" + std::to_string(fields.global_extents[0]) +
                     ", ntheta=" + std::to_string(fields.global_extents[1]) +
                     ", nphi=" + std::to_string(fields.global_extents[2]), Colors::cyan);
                INFO("MPI domain decomposition [r, theta, phi] = [" +
                     std::to_string(partitions[0]) + ", " + std::to_string(partitions[1]) + ", " +
                     std::to_string(partitions[2]) + "]", Colors::cyan);
                INFO("r coordinate from " + std::to_string(global_coordinate_min[0]) + " to " +
                     std::to_string(global_coordinate_max[0]), Colors::hotpink);
                INFO("theta coordinate from " + std::to_string(global_coordinate_min[1]) + " to " +
                     std::to_string(global_coordinate_max[1]), Colors::hotpink);
                INFO("phi coordinate from " + std::to_string(global_coordinate_min[2]) + " to " +
                     std::to_string(global_coordinate_max[2]), Colors::hotpink);
            }
        } else if (mpi_rank == 0) {
            INFO("Running in vacuum mode: no grid/MHD data loaded, scattering disabled.", Colors::cyan);
        }

        std::uint64_t local_photon_count = 0;
        if (options.mode == SimulationMode::Image) {
            const auto rank_count = static_cast<std::uint64_t>(mpi_size);
            const auto rank = static_cast<std::uint64_t>(mpi_rank);
            local_photon_count = nphotons / rank_count +
                (rank < nphotons % rank_count ? 1ULL : 0ULL);
        } else {
            local_photon_count = static_cast<std::uint64_t>(fields.r.extent(0)) *
                fields.r.extent(1) * fields.r.extent(2) *
                static_cast<std::uint64_t>(photon_generation.superphotons_per_cell);
        }

        std::uint64_t global_photon_count = 0;
        MPI_Allreduce(&local_photon_count, &global_photon_count, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
        if (global_photon_count > max_global_photons) {
            if (mpi_rank == 0) {
                ERROR("Global photon count exceeds the supported limit of 100 billion.");
            }
            Kokkos::finalize();
            MPI_Finalize();
            return 1;
        }
        if (local_photon_count > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
            if (mpi_rank == 0) {
                ERROR("A rank received more photons than its 32-bit local kernel index supports; "
                      "use more MPI ranks or a finer domain decomposition.");
            }
            Kokkos::finalize();
            MPI_Finalize();
            return 1;
        }
        nphotons = global_photon_count;
        if (mpi_rank == 0) {
            INFO("Global superphoton count: " + std::to_string(global_photon_count), Colors::cyan);
        }
        std::uint64_t global_id_offset = 0;
        MPI_Exscan(&local_photon_count, &global_id_offset, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
        if (mpi_rank == 0) global_id_offset = 0;
        const int photons_per_process = static_cast<int>(local_photon_count);
        Photons photons(photons_per_process, "photons_");

        if (options.mode == SimulationMode::Image) {
            timers.AddTimer("Initialize Image Camera");
            timers.BeginTimer("Initialize Image Camera");
            initialize_photons_image(
                photons_per_process,
                camera_distance,
                camera_theta,
                camera_phi,
                mpi_rank,
                mpi_size,
                global_id_offset,
                photons
                );
            timers.EndTimer("Initialize Image Camera");
        }

        else
        {
            timers.AddTimer("Initialize Disk Photons");
            timers.BeginTimer("Initialize Disk Photons");
            initialize_photons_disk(
                photons_per_process,
                mpi_rank,
                global_id_offset,
                fields.r, fields.theta, fields.phi,
                fields.density, fields.temperature, fields.velocity, fields.magnetic,
                photon_generation,
                units,
                photons
                );
            timers.EndTimer("Initialize Disk Photons");
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
            options,
            photon_generation,
            ScatteringModel{
                options.enable_scattering,
                options.scattering_optical_depth,
                options.scattering_albedo,
                options.scattering_seed
            },
            RadiativeTransferModel{
                options.enable_absorption,
                options.enable_emission,
                options.absorption_coefficient,
                options.emission_coefficient
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
