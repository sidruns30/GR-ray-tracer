#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

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
    char numpy_option[] = "--numpy-dir=/data/snapshot";
    char* arguments[] = {program, numpy_option};
    const SimulationOptions options = parse_simulation_options(2, arguments);
    passed = passed && options.numpy_dir == "/data/snapshot";

    char removed_option[] = "--grid-r";
    char removed_value[] = "r.npy";
    char* removed_arguments[] = {program, removed_option, removed_value};
    passed = passed && throws_with_text(
        [&] { (void)parse_simulation_options(3, removed_arguments); },
        "Unknown command-line argument");

    const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
    const fs::path root = fs::temp_directory_path() /
        ("grraytracer_numpy_dir_" + std::to_string(unique));
    const fs::path absent = root.string() + "_absent";
    passed = passed && throws_with_text(
        [&] { (void)discover_numpy_field_paths(absent.string()); },
        "does not exist");

    fs::create_directories(root);
    std::ofstream(root / "r.npy").put('\0');
    passed = passed && throws_with_text(
        [&] { (void)discover_numpy_field_paths(root.string()); },
        "theta.npy");

    for (const char* name : {
             "theta.npy", "phi.npy", "rho.npy", "Tgas.npy", "vel.npy", "mag.npy"}) {
        std::ofstream(root / name).put('\0');
    }
    const NumpyFieldPaths paths = discover_numpy_field_paths(root.string());
    passed = passed && fs::path(paths.r).filename() == "r.npy";
    passed = passed && fs::path(paths.temperature).filename() == "Tgas.npy";
    passed = passed && fs::path(paths.magnetic).filename() == "mag.npy";

    std::error_code cleanup_error;
    fs::remove_all(root, cleanup_error);
    passed = passed && !cleanup_error;
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
