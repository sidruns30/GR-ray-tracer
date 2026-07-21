#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "radiative_transfer/scattering.hpp"
#include "radiative_transfer/radiative_transfer.hpp"

enum class SimulationMode : std::uint8_t { Image = 0, Disk = 1 };

struct SimulationOptions {
    SimulationMode mode = SimulationMode::Image;
    std::string numpy_dir = ".";
    std::string output_dir = output_directory;
    bool vacuum = false;
    bool enable_scattering = false;
    real scattering_optical_depth = 0.0;
    real scattering_albedo = 1.0;
    std::uint64_t scattering_seed = 12345u;
    bool enable_absorption = false;
    bool enable_emission = false;
    real absorption_coefficient = 0.0;
    real emission_coefficient = 0.0;
    IntegratorType integrator = IntegratorType::RK45;
    std::size_t image_nx = 256;
    std::size_t image_ny = 256;
    std::size_t spectrum_bins = 128;
    real spectrum_min_hz = 1.0e8;
    real spectrum_max_hz = 1.0e22;
};

inline SimulationMode parse_simulation_mode(const std::string& value) {
    if (value == "image") return SimulationMode::Image;
    if (value == "disk") return SimulationMode::Disk;
    throw std::runtime_error("simulation.mode must be 'image' or 'disk'");
}

inline std::string parse_config_path(int argc, char* argv[]) {
    const std::string path = argc == 2 ? argv[1] : "";
    if (path.empty() || path.front() == '-') {
        throw std::runtime_error("Usage: gr-ray-trace <config.toml>");
    }
    return path;
}
