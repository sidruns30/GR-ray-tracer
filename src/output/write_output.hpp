// Atomic checkpoint and final image-product output.
#pragma once

#include <cstdint>
#include <filesystem>
#include <system_error>
#include <vector>

#include "../radiative_transfer/observation.hpp"
#include "../simulation_options.hpp"
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
    else if (name == "emission_frame_energy") add_strided(photons.emission_frame_energy_host);
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
    else if (name == "phase") add_strided(photons.phase_host);
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

inline void publish_archive(const std::filesystem::path& final_path,
                            const auto& write_contents) {
    namespace fs = std::filesystem;
    fs::create_directories(final_path.parent_path());
    const fs::path temporary_path = final_path.string() + ".tmp";
    try {
        cnpy::NpzWriter archive(temporary_path.string());
        write_contents(archive);
        archive.close();
    } catch (...) {
        fs::remove(temporary_path);
        throw;
    }
    std::error_code error;
    fs::rename(temporary_path, final_path, error);
    if (error) {
        fs::remove(temporary_path);
        throw std::runtime_error(
            "Failed to publish output archive " + final_path.string() + ": " + error.message());
    }
}

// One atomic archive per output step and rank. Since ownership can change,
// consumers must join rank files by the globally unique photon ID.
inline void write_output_step(Photons& photons,
                              std::size_t step,
                              int rank,
                              const OutputSelection& selection) {
    photons.copy_to_host(selection);
    const std::filesystem::path path = std::filesystem::path(output_directory) /
        (simulation_name + "_step_" + std::to_string(step) + "_rank" + std::to_string(rank) + ".npz");
    publish_archive(path, [&](cnpy::NpzWriter& archive) {
        const std::size_t photon_count = photons.x0.extent(0);
        for (const std::string& name : selection.names()) {
            add_photon_array(archive, photons, name, photon_count, output_stride);
        }
    });
}

// Image products use fixed global axes. Arrays are summed across ranks and
// written once by rank zero, so no post-processing guesswork is required.
inline void write_image_products(Photons& photons, int rank,
                                 const SimulationOptions& options) {
    photons.copy_observation_to_host();
    ObservationProducts local = build_observation_products(
        photons, options.image_nx, options.image_ny, options.spectrum_bins,
        options.spectrum_min_hz, options.spectrum_max_hz, plane_dim1, plane_dim2);
    ObservationProducts global = local;
    const int image_count = static_cast<int>(local.image_I.size());
    const int spectrum_count = static_cast<int>(local.spectrum_I.size());
    MPI_Reduce(local.image_I.data(), global.image_I.data(), image_count,
               MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(local.image_Q.data(), global.image_Q.data(), image_count,
               MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(local.image_U.data(), global.image_U.data(), image_count,
               MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(local.image_V.data(), global.image_V.data(), image_count,
               MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(local.spectrum_I.data(), global.spectrum_I.data(), spectrum_count,
               MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local.arrived_count, &global.arrived_count, 1,
               MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank != 0) return;

    const std::filesystem::path path =
        std::filesystem::path(output_directory) / (simulation_name + "_image_products.npz");
    publish_archive(path, [&](cnpy::NpzWriter& archive) {
        archive.add("image_I", global.image_I.data(), {global.image_ny, global.image_nx});
        archive.add("image_Q", global.image_Q.data(), {global.image_ny, global.image_nx});
        archive.add("image_U", global.image_U.data(), {global.image_ny, global.image_nx});
        archive.add("image_V", global.image_V.data(), {global.image_ny, global.image_nx});
        archive.add("spectrum_frequency_hz", global.spectrum_frequency_hz.data(),
                    {global.spectrum_bins});
        archive.add("spectrum_I", global.spectrum_I.data(), {global.spectrum_bins});
        archive.add("arrived_count", &global.arrived_count, {std::size_t(1)});
    });
}
