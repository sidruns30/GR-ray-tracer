/*
    Minimal TOML-subset config loader for the runtime-configurable simulation
    parameters declared in utils.hpp (black hole mass/spin, camera setup,
    photon count, termination radii, integration tolerances, output cadence).

    Only the flat subset of TOML actually needed here is supported:
      - blank lines and full-line/trailing '#' comments
      - '[section]' headers, kept purely for readability; keys are looked up
        by their dotted "section.key" name (e.g. "black_hole.mass")
      - 'key = value' with a scalar value, plus the single-line string array
        used by output.variables
    This deliberately avoids pulling in a full TOML dependency for a handful
    of scalar settings.
*/
#pragma once

#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "../utils.hpp"
#include "../output/display.hpp"
#include "domain_decomposition.hpp"

namespace toml_config {

inline std::string strip(const std::string& s) {
    const auto begin = s.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) return "";
    const auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(begin, end - begin + 1);
}

// Parses "key = value" lines (ignoring [section] headers and comments) into a
// flat map keyed by "section.key" (or just "key" outside any section).
inline std::map<std::string, std::string> parse_flat_toml(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Failed to open config file: " + path);
    }

    std::map<std::string, std::string> values;
    std::string section;
    std::string line;
    while (std::getline(in, line)) {
        // Strip a trailing comment that isn't inside a quoted string.
        bool in_quotes = false;
        for (std::size_t i = 0; i < line.size(); ++i) {
            if (line[i] == '"') in_quotes = !in_quotes;
            else if (line[i] == '#' && !in_quotes) { line = line.substr(0, i); break; }
        }
        const std::string trimmed = strip(line);
        if (trimmed.empty()) continue;

        if (trimmed.front() == '[' && trimmed.back() == ']') {
            section = strip(trimmed.substr(1, trimmed.size() - 2));
            continue;
        }

        const auto eq = trimmed.find('=');
        if (eq == std::string::npos) {
            throw std::runtime_error("Malformed config line (expected 'key = value'): " + line);
        }
        const std::string key = strip(trimmed.substr(0, eq));
        std::string value = strip(trimmed.substr(eq + 1));
        if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
            value = value.substr(1, value.size() - 2);
        }
        const std::string full_key = section.empty() ? key : section + "." + key;
        values[full_key] = value;
    }
    return values;
}

using RawConfig = std::map<std::string, std::string>;

inline real get_real(const RawConfig& cfg, const std::string& key, real fallback) {
    const auto it = cfg.find(key);
    if (it == cfg.end()) return fallback;
    return std::stod(it->second);
}

inline int get_int(const RawConfig& cfg, const std::string& key, int fallback) {
    const auto it = cfg.find(key);
    if (it == cfg.end()) return fallback;
    return std::stoi(it->second);
}

inline size_t get_size_t(const RawConfig& cfg, const std::string& key, size_t fallback) {
    const auto it = cfg.find(key);
    if (it == cfg.end()) return fallback;
    if (!it->second.empty() && it->second.front() == '-') {
        throw std::runtime_error(key + " must not be negative");
    }
    return static_cast<size_t>(std::stoull(it->second));
}

inline bool get_bool(const RawConfig& cfg, const std::string& key, bool fallback) {
    const auto it = cfg.find(key);
    if (it == cfg.end()) return fallback;
    return it->second == "true";
}

inline std::string get_string(const RawConfig& cfg, const std::string& key, std::string fallback) {
    const auto it = cfg.find(key);
    return it == cfg.end() ? std::move(fallback) : it->second;
}

// Loads `path` and overwrites the runtime-configurable globals in utils.hpp.
// Keys absent from the file keep whatever value the global already had (i.e.
// the compiled-in defaults from utils.cpp), so a config file only needs to
// specify the settings it wants to change.
inline void load_and_apply_toml_config(const std::string& path,
                                       std::string& output_variables,
                                       PhotonGenerationConfig& photon_generation,
                                       UnitConversions& units,
                                       DomainDecompositionSpec& decomposition) {
    const RawConfig cfg = parse_flat_toml(path);

    M_BH = get_real(cfg, "black_hole.mass", M_BH);
    a_BH = get_real(cfg, "black_hole.spin", a_BH);
    if (M_BH <= 0.0) {
        throw std::runtime_error("black_hole.mass must be positive (got " + std::to_string(M_BH) + ")");
    }
    if (Kokkos::abs(a_BH) > M_BH) {
        throw std::runtime_error("black_hole.spin must satisfy |spin| <= mass, i.e. dimensionless spin "
                                  "a/M in [-1, 1] (got spin=" + std::to_string(a_BH) +
                                  ", mass=" + std::to_string(M_BH) + "); |spin| > mass is a naked singularity.");
    }
    R_HORIZON = M_BH + Kokkos::sqrt(
        Kokkos::fmax(real(0.0), M_BH * M_BH - a_BH * a_BH));

    use_pinhole_camera = get_bool(cfg, "camera.use_pinhole", use_pinhole_camera);
    camera_theta = get_real(cfg, "camera.theta", camera_theta);
    camera_phi = get_real(cfg, "camera.phi", camera_phi);
    target_rmin = get_real(cfg, "camera.target_r_min", target_rmin);
    target_rmax = get_real(cfg, "camera.target_r_max", target_rmax);

    use_image_camera = get_bool(cfg, "camera.use_image", use_image_camera);
    plane_dim1 = get_real(cfg, "camera.plane_dim1", plane_dim1);
    plane_dim2 = get_real(cfg, "camera.plane_dim2", plane_dim2);
    plane_theta = get_real(cfg, "camera.plane_theta", plane_theta);
    plane_phi = get_real(cfg, "camera.plane_phi", plane_phi);

    camera_distance = get_real(cfg, "camera.distance", camera_distance);

    nphotons = get_int(cfg, "photons.count", nphotons);
    photon_generation.generator = parse_photon_generator_type(
        get_string(cfg, "photons.generator", "blackbody"));
    photon_generation.superphotons_per_cell = get_int(
        cfg, "photons.superphotons_per_cell", photon_generation.superphotons_per_cell);
    photon_generation.energy_per_cell_erg = get_real(
        cfg, "photons.energy_per_cell_erg", photon_generation.energy_per_cell_erg);
    photon_generation.power_law_slope = get_real(
        cfg, "photons.power_law_slope", photon_generation.power_law_slope);
    photon_generation.nu_min_hz = get_real(cfg, "photons.nu_min_hz", photon_generation.nu_min_hz);
    photon_generation.nu_max_hz = get_real(cfg, "photons.nu_max_hz", photon_generation.nu_max_hz);
    photon_generation.camera_frequency_hz = get_real(
        cfg, "photons.camera_frequency_hz", photon_generation.camera_frequency_hz);
    photon_generation.camera_packet_energy_erg = get_real(
        cfg, "photons.camera_packet_energy_erg", photon_generation.camera_packet_energy_erg);
    photon_generation.custom_frequency_hz = get_real(
        cfg, "photons.custom_frequency_hz", photon_generation.custom_frequency_hz);

    units.length_cm_per_code = get_real(cfg, "units.length_cm_per_code", units.length_cm_per_code);
    units.time_s_per_code = get_real(cfg, "units.time_s_per_code", units.time_s_per_code);
    units.density_g_cm3_per_code = get_real(
        cfg, "units.density_g_cm3_per_code", units.density_g_cm3_per_code);
    units.temperature_k_per_code = get_real(
        cfg, "units.temperature_k_per_code", units.temperature_k_per_code);
    units.four_velocity_cm_s_per_code = get_real(
        cfg, "units.four_velocity_cm_s_per_code", units.four_velocity_cm_s_per_code);
    units.magnetic_gauss_per_code = get_real(
        cfg, "units.magnetic_gauss_per_code", units.magnetic_gauss_per_code);

    decomposition = DomainDecompositionSpec::parse(
        get_string(cfg, "parallel.decomposition", "[1, 1, -1]"));
    max_steps = get_int(cfg, "integration.max_steps", max_steps);
    termination_percent = get_real(cfg, "integration.termination_percent", termination_percent);
    dlambda = get_real(cfg, "integration.initial_step", dlambda);
    atol_default = get_real(cfg, "integration.atol", atol_default);
    rtol_default = get_real(cfg, "integration.rtol", rtol_default);

    // Derived defaults (R_HORIZON, 1.5*camera_distance) are computed from the
    // possibly-just-overridden values above, then only replaced if the config
    // file explicitly sets termination.r_min / termination.r_max.
    termination_r_min = get_real(cfg, "termination.r_min", R_HORIZON);
    termination_r_max = get_real(cfg, "termination.r_max", 1.5 * camera_distance);

    output_interval = get_size_t(cfg, "output.interval", output_interval);
    output_stride = get_size_t(cfg, "output.stride", output_stride);
    output_variables = get_string(cfg, "output.variables", std::move(output_variables));

    if (nphotons <= 0 || nphotons > 1000000000) {
        throw std::runtime_error("photons.count must be in [1, 1000000000]");
    }
    if (max_steps <= 0) throw std::runtime_error("integration.max_steps must be positive");
    if (output_interval == 0) throw std::runtime_error("output.interval must be positive");
    if (output_stride == 0) throw std::runtime_error("output.stride must be positive");
    if (termination_percent <= 0.0 || termination_percent > 1.0) {
        throw std::runtime_error("integration.termination_percent must be in (0, 1]");
    }
    photon_generation.validate();
    units.validate();

}

} // namespace toml_config
