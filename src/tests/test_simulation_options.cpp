#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include "../input/load_python_arrays.hpp"
#include "../input/toml_config.hpp"
#include "../simulation_options.hpp"

namespace {

bool throws_with_text(const auto& operation, const std::string& expected) {
    try {
        operation();
    } catch (const std::exception& error) {
        return std::string(error.what()).find(expected) != std::string::npos;
    }
    return false;
}

} // namespace

int main() {
    namespace fs = std::filesystem;
    bool passed = true;

    char program[] = "gr-ray-trace";
    char config_argument[] = "simulation.toml";
    char* arguments[] = {program, config_argument};
    passed = passed && parse_config_path(2, arguments) == "simulation.toml";

    char* missing_arguments[] = {program};
    passed = passed && throws_with_text(
        [&] { (void)parse_config_path(1, missing_arguments); },
        "Usage: gr-ray-trace <config.toml>");

    char removed_option[] = "--config";
    char removed_value[] = "simulation.toml";
    char* removed_arguments[] = {program, removed_option, removed_value};
    passed = passed && throws_with_text(
        [&] { (void)parse_config_path(3, removed_arguments); },
        "Usage: gr-ray-trace <config.toml>");

    const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
    const fs::path root = fs::temp_directory_path() /
        ("grraytracer_numpy_dir_" + std::to_string(unique));
    const fs::path grid = root / "grid";
    const fs::path config = root / "runtime.toml";
    fs::create_directories(root);

    {
        std::ofstream stream(config);
        stream << "[simulation]\n"
               << "mode = \"disk\"\n"
               << "name = \"my-run\"\n"
               << "[input]\n"
               << "vacuum = true\n"
               << "numpy_dir = \"/data/snapshot\"\n"
               << "[photons]\n"
               << "count = 100000000000\n"
               << "[integration]\n"
               << "integrator = \"rk4\"\n"
               << "[scattering]\n"
               << "enabled = true\n"
               << "optical_depth = 0.75\n"
               << "albedo = 0.25\n"
               << "seed = 987654321\n"
               << "[output]\n"
               << "directory = \"/tmp/grraytracer-output\"\n"
               << "variables = [\"id\", \"frequency\"]\n";
    }
    SimulationOptions options;
    std::string output_variables(default_output_variables);
    PhotonGenerationConfig photon_generation;
    UnitConversions units;
    DomainDecompositionSpec decomposition;
    toml_config::load_and_apply_toml_config(
        config.string(), options, output_variables, photon_generation, units, decomposition);
    passed = passed && options.vacuum;
    passed = passed && options.mode == SimulationMode::Disk;
    passed = passed && simulation_name == "my-run";
    passed = passed && options.numpy_dir == "/data/snapshot";
    passed = passed && options.output_dir == "/tmp/grraytracer-output";
    passed = passed && options.enable_scattering;
    passed = passed && options.scattering_optical_depth == 0.75;
    passed = passed && options.scattering_albedo == 0.25;
    passed = passed && options.scattering_seed == 987654321u;
    passed = passed && options.integrator == IntegratorType::RK4;
    passed = passed && output_variables == "[\"id\", \"frequency\"]";
    passed = passed && nphotons == max_global_photons;

    const fs::path oversized_config = root / "oversized.toml";
    {
        std::ofstream stream(oversized_config);
        stream << "[photons]\ncount = 100000000001\n";
    }
    SimulationOptions oversized_options;
    std::string oversized_output_variables(default_output_variables);
    PhotonGenerationConfig oversized_generation;
    UnitConversions oversized_units;
    DomainDecompositionSpec oversized_decomposition;
    passed = passed && throws_with_text(
        [&] {
            toml_config::load_and_apply_toml_config(
                oversized_config.string(), oversized_options,
                oversized_output_variables, oversized_generation,
                oversized_units, oversized_decomposition);
        },
        "photons.count must be in [1, 100000000000]");

    const toml_config::RawConfig invalid_bool{{"input.vacuum", "yes"}};
    passed = passed && throws_with_text(
        [&] { (void)toml_config::get_bool(invalid_bool, "input.vacuum", false); },
        "must be true or false");
    passed = passed && throws_with_text(
        [] { (void)parse_simulation_mode("pinhole"); },
        "must be 'image' or 'disk'");

    const fs::path absent = root / "absent";
    passed = passed && throws_with_text(
        [&] { (void)discover_numpy_field_paths(absent.string()); },
        "does not exist");

    fs::create_directories(grid);
    std::ofstream(grid / "r.npy").put('\0');
    passed = passed && throws_with_text(
        [&] { (void)discover_numpy_field_paths(grid.string()); },
        "theta.npy");

    for (const char* name : {
             "theta.npy", "phi.npy", "rho.npy", "Tgas.npy", "vel.npy", "mag.npy"}) {
        std::ofstream(grid / name).put('\0');
    }
    const NumpyFieldPaths paths = discover_numpy_field_paths(grid.string());
    passed = passed && fs::path(paths.r).filename() == "r.npy";
    passed = passed && fs::path(paths.temperature).filename() == "Tgas.npy";
    passed = passed && fs::path(paths.magnetic).filename() == "mag.npy";

    std::error_code cleanup_error;
    fs::remove_all(root, cleanup_error);
    passed = passed && !cleanup_error;
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
