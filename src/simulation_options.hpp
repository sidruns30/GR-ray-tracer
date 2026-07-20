#pragma once

#include <stdexcept>
#include <string>

#include "input/load_python_arrays.hpp"
#include "radiative_transfer/scattering.hpp"

struct SimulationOptions {
    NumpyFieldPaths fields;
    std::string output_dir = output_directory;
    std::string config_path; // optional TOML file overriding utils.hpp defaults, see input/toml_config.hpp
    bool enable_scattering = false;
    real scattering_optical_depth = 0.0;
    real scattering_albedo = 1.0;
    IntegratorType integrator = IntegratorType::RK45;
};

inline SimulationOptions parse_simulation_options(int argc, char* argv[]) {
    SimulationOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        // Accept both "--flag value" and "--flag=value".
        std::string flag = arg;
        std::string inline_value;
        bool has_inline_value = false;
        const auto eq = arg.find('=');
        if (arg.rfind("--", 0) == 0 && eq != std::string::npos) {
            flag = arg.substr(0, eq);
            inline_value = arg.substr(eq + 1);
            has_inline_value = true;
        }

        const auto next_value = [&](const char* name) -> std::string {
            if (has_inline_value) return inline_value;
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("Missing value for ") + name);
            }
            return argv[++i];
        };

        if (flag == "--grid-r") options.fields.r = next_value("--grid-r");
        else if (flag == "--grid-theta") options.fields.theta = next_value("--grid-theta");
        else if (flag == "--grid-phi") options.fields.phi = next_value("--grid-phi");
        else if (flag == "--density") options.fields.density = next_value("--density");
        else if (flag == "--temperature") options.fields.temperature = next_value("--temperature");
        else if (flag == "--velocity") options.fields.velocity = next_value("--velocity");
        else if (flag == "--magnetic") options.fields.magnetic = next_value("--magnetic");
        else if (flag == "--output-dir") options.output_dir = next_value("--output-dir");
        else if (flag == "--config") options.config_path = next_value("--config");
        else if (flag == "--scatter-optical-depth") options.scattering_optical_depth = std::stod(next_value("--scatter-optical-depth"));
        else if (flag == "--scatter-albedo") options.scattering_albedo = std::stod(next_value("--scatter-albedo"));
        else if (flag == "--scatter") options.enable_scattering = true;
        else if (flag == "--integrator") {
            const auto mode = next_value("--integrator");
            if (mode == "rk4") {
                options.integrator = IntegratorType::RK4;
            } else if (mode == "rk45") {
                options.integrator = IntegratorType::RK45;
            } else {
                throw std::runtime_error("Unknown integrator: " + mode);
            }
        }
        else {
            throw std::runtime_error("Unknown command-line argument: " + arg +
                                      " (see README.md for supported flags)");
        }
    }
    return options;
}

