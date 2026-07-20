#pragma once

#include <stdexcept>
#include <string>

#include "input/load_python_arrays.hpp"
#include "radiative_transfer/scattering.hpp"

struct SimulationOptions {
    NumpyFieldPaths fields;
    std::string output_dir = output_directory;
    bool enable_scattering = false;
    real scattering_optical_depth = 0.0;
    real scattering_albedo = 1.0;
    IntegratorType integrator = IntegratorType::RK45;
};

inline SimulationOptions parse_simulation_options(int argc, char* argv[]) {
    SimulationOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto next_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("Missing value for ") + name);
            }
            return argv[++i];
        };

        if (arg == "--grid-r") options.fields.r = next_value("--grid-r");
        else if (arg == "--grid-theta") options.fields.theta = next_value("--grid-theta");
        else if (arg == "--grid-phi") options.fields.phi = next_value("--grid-phi");
        else if (arg == "--density") options.fields.density = next_value("--density");
        else if (arg == "--temperature") options.fields.temperature = next_value("--temperature");
        else if (arg == "--velocity") options.fields.velocity = next_value("--velocity");
        else if (arg == "--magnetic") options.fields.magnetic = next_value("--magnetic");
        else if (arg == "--output-dir") options.output_dir = next_value("--output-dir");
        else if (arg == "--scatter-optical-depth") options.scattering_optical_depth = std::stod(next_value("--scatter-optical-depth"));
        else if (arg == "--scatter-albedo") options.scattering_albedo = std::stod(next_value("--scatter-albedo"));
        else if (arg == "--scatter") options.enable_scattering = true;
        else if (arg == "--integrator") {
            const auto mode = next_value("--integrator");
            if (mode == "rk4") {
                options.integrator = IntegratorType::RK4;
            } else if (mode == "rk45") {
                options.integrator = IntegratorType::RK45;
            } else {
                throw std::runtime_error("Unknown integrator: " + mode);
            }
        }
    }
    return options;
}

