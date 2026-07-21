#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <system_error>
#include <vector>

#include "../radiative_transfer/observation.hpp"
#include "../utils.hpp"

template <typename HostView>
auto make_strided_host_values(const HostView& source,
                              std::size_t source_count,
                              std::size_t stride) {
    using value_type = typename HostView::non_const_value_type;
    const std::size_t output_count = source_count == 0 ? 0 : 1 + (source_count - 1) / stride;
    std::vector<value_type> values(output_count);
    for (std::size_t output_index = 0; output_index < output_count; ++output_index) {
        values[output_index] = source(output_index * stride);
    }
    return values;
}

inline void add_photon_array(cnpy::NpzWriter& archive,
                             const Photons& photons,
                             const std::string& name,
                             std::size_t photon_count,
                             std::size_t stride) {
    const std::size_t output_count = photon_count == 0 ? 0 : 1 + (photon_count - 1) / stride;
    const std::vector<std::size_t> shape{output_count};
    const auto add_strided = [&](const auto& source) {
        const auto values = make_strided_host_values(source, photon_count, stride);
        archive.add(name, values.data(), shape);
    };
    if (name == "id") add_strided(photons.id_host);
    else if (name == "frequency") add_strided(photons.frequency_host);
    else if (name == "x0") add_strided(photons.x0_host);
    else if (name == "x1") add_strided(photons.x1_host);
    else if (name == "x2") add_strided(photons.x2_host);
    else if (name == "x3") add_strided(photons.x3_host);
    else if (name == "k0") add_strided(photons.k0_host);
    else if (name == "k1") add_strided(photons.k1_host);
    else if (name == "k2") add_strided(photons.k2_host);
    else if (name == "k3") add_strided(photons.k3_host);
    else if (name == "I") add_strided(photons.I_host);
    else if (name == "Q") add_strided(photons.Q_host);
    else if (name == "U") add_strided(photons.U_host);
    else if (name == "V") add_strided(photons.V_host);
    else if (name == "dlambda") add_strided(photons.dlambda_host);
    else if (name == "theta_disp") add_strided(photons.theta_disp_host);
    else if (name == "phi_disp") add_strided(photons.phi_disp_host);
    else if (name == "terminate") {
        std::vector<std::uint8_t> terminated(output_count);
        for (std::size_t output_index = 0; output_index < output_count; ++output_index) {
            terminated[output_index] =
                photons.terminate_host(output_index * stride) ? 1u : 0u;
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
                add_photon_array(archive, photons, name, photon_count, output_stride);
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
