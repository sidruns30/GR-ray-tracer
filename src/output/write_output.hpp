#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <system_error>
#include <vector>

#include "../radiative_transfer/observation.hpp"
#include "../utils.hpp"

inline void add_photon_array(cnpy::NpzWriter& archive,
                             const Photons& photons,
                             const std::string& name,
                             std::size_t photon_count) {
    const std::vector<std::size_t> shape{photon_count};
    if (name == "id") archive.add(name, photons.id_host.data(), shape);
    else if (name == "frequency") archive.add(name, photons.frequency_host.data(), shape);
    else if (name == "x0") archive.add(name, photons.x0_host.data(), shape);
    else if (name == "x1") archive.add(name, photons.x1_host.data(), shape);
    else if (name == "x2") archive.add(name, photons.x2_host.data(), shape);
    else if (name == "x3") archive.add(name, photons.x3_host.data(), shape);
    else if (name == "k0") archive.add(name, photons.k0_host.data(), shape);
    else if (name == "k1") archive.add(name, photons.k1_host.data(), shape);
    else if (name == "k2") archive.add(name, photons.k2_host.data(), shape);
    else if (name == "k3") archive.add(name, photons.k3_host.data(), shape);
    else if (name == "I") archive.add(name, photons.I_host.data(), shape);
    else if (name == "Q") archive.add(name, photons.Q_host.data(), shape);
    else if (name == "U") archive.add(name, photons.U_host.data(), shape);
    else if (name == "V") archive.add(name, photons.V_host.data(), shape);
    else if (name == "dlambda") archive.add(name, photons.dlambda_host.data(), shape);
    else if (name == "theta_disp") archive.add(name, photons.theta_disp_host.data(), shape);
    else if (name == "phi_disp") archive.add(name, photons.phi_disp_host.data(), shape);
    else if (name == "terminate") {
        std::vector<std::uint8_t> terminated(photon_count);
        for (std::size_t i = 0; i < photon_count; ++i) {
            terminated[i] = photons.terminate_host(i) ? 1u : 0u;
        }
        archive.add(name, terminated.data(), shape);
    }
}

inline void add_observation_array(cnpy::NpzWriter& archive,
                                  const ObservationProducts& products,
                                  const std::string& name) {
    if (name == "image_I") {
        archive.add(name, products.image_I.data(), {products.image_ny, products.image_nx});
    } else if (name == "image_Q") {
        archive.add(name, products.image_Q.data(), {products.image_ny, products.image_nx});
    } else if (name == "image_U") {
        archive.add(name, products.image_U.data(), {products.image_ny, products.image_nx});
    } else if (name == "image_V") {
        archive.add(name, products.image_V.data(), {products.image_ny, products.image_nx});
    } else if (name == "lightcurve_I") {
        archive.add(name, products.lightcurve_I.data(), {products.lightcurve_bins});
    } else if (name == "spectrum_frequency_hz") {
        archive.add(name, products.spectrum_frequency_hz.data(), {products.spectrum_bins});
    } else if (name == "spectrum_I") {
        archive.add(name, products.spectrum_I.data(), {products.spectrum_bins});
    }
}

// Each checkpoint is published atomically so live readers never observe a
// partially written archive.
inline void write_output_step(Photons& photons,
                              std::size_t step,
                              int rank,
                              const OutputSelection& selection) {
    namespace fs = std::filesystem;
    const fs::path directory(output_directory);
    fs::create_directories(directory);

    photons.copy_to_host(selection);
    std::optional<ObservationProducts> observations;
    if (selection.needs_observation_products()) {
        observations = build_observation_products(photons);
    }

    const std::string filename =
        "output_step_" + std::to_string(step) + "_rank" + std::to_string(rank) + ".npz";
    const fs::path final_path = directory / filename;
    const fs::path temporary_path = final_path.string() + ".tmp";

    try {
        cnpy::NpzWriter archive(temporary_path.string());
        const std::size_t photon_count = photons.x0.extent(0);
        for (const std::string& name : selection.names()) {
            if (name.rfind("image_", 0) == 0 || name == "lightcurve_I" ||
                name == "spectrum_frequency_hz" || name == "spectrum_I") {
                add_observation_array(archive, *observations, name);
            } else {
                add_photon_array(archive, photons, name, photon_count);
            }
        }
        archive.close();
    } catch (...) {
        fs::remove(temporary_path);
        throw;
    }

    std::error_code error;
    fs::rename(temporary_path, final_path, error);
    if (error) {
        fs::remove(temporary_path);
        throw std::runtime_error("Failed to publish output archive " + final_path.string() + ": " + error.message());
    }
}
