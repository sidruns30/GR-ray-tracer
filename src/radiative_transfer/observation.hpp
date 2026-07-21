// Camera products constructed from image-mode packets that reached the plane.
#pragma once

#include <cstdint>
#include <vector>

#include "../utils.hpp"

struct ObservationProducts {
    std::size_t image_nx = 0;
    std::size_t image_ny = 0;
    std::size_t spectrum_bins = 0;
    std::uint64_t arrived_count = 0;
    std::vector<real> image_I;
    std::vector<real> image_Q;
    std::vector<real> image_U;
    std::vector<real> image_V;
    std::vector<real> spectrum_frequency_hz;
    std::vector<real> spectrum_I;
};

inline std::size_t observation_bin(real value, real minimum,
                                   real maximum, std::size_t bins) {
    if (bins <= 1 || !(maximum > minimum)) return 0;
    const real fraction = (value - minimum) / (maximum - minimum);
    const real scaled = Kokkos::fmax(
        real(0.0), Kokkos::fmin(real(1.0), fraction));
    return static_cast<std::size_t>(scaled * static_cast<real>(bins - 1));
}

inline ObservationProducts build_observation_products(
    const Photons& photons,
    std::size_t image_nx,
    std::size_t image_ny,
    std::size_t spectrum_bins,
    real spectrum_min_hz,
    real spectrum_max_hz,
    real image_width,
    real image_height) {
    ObservationProducts products;
    products.image_nx = image_nx;
    products.image_ny = image_ny;
    products.spectrum_bins = spectrum_bins;
    products.image_I.assign(image_nx * image_ny, 0.0);
    products.image_Q.assign(image_nx * image_ny, 0.0);
    products.image_U.assign(image_nx * image_ny, 0.0);
    products.image_V.assign(image_nx * image_ny, 0.0);
    products.spectrum_frequency_hz.resize(spectrum_bins);
    products.spectrum_I.assign(spectrum_bins, 0.0);

    const real log_min = Kokkos::log10(spectrum_min_hz);
    const real log_max = Kokkos::log10(spectrum_max_hz);
    for (std::size_t bin = 0; bin < spectrum_bins; ++bin) {
        const real fraction = spectrum_bins > 1
            ? static_cast<real>(bin) / static_cast<real>(spectrum_bins - 1)
            : 0.0;
        products.spectrum_frequency_hz[bin] =
            Kokkos::pow(real(10.0), log_min + fraction * (log_max - log_min));
    }

    for (std::size_t i = 0; i < photons.phase_host.extent(0); ++i) {
        if (static_cast<PhotonPhase>(photons.phase_host(i)) != PhotonPhase::ImageArrived) {
            continue;
        }
        ++products.arrived_count;
        const std::size_t pixel_x = observation_bin(
            photons.theta_disp_host(i), -0.5 * image_width, 0.5 * image_width, image_nx);
        const std::size_t pixel_y = observation_bin(
            photons.phi_disp_host(i), -0.5 * image_height, 0.5 * image_height, image_ny);
        const std::size_t pixel = pixel_y * image_nx + pixel_x;
        products.image_I[pixel] += photons.I_host(i);
        products.image_Q[pixel] += photons.Q_host(i);
        products.image_U[pixel] += photons.U_host(i);
        products.image_V[pixel] += photons.V_host(i);

        const real frequency = photons.frequency_host(i);
        if (frequency >= spectrum_min_hz && frequency <= spectrum_max_hz) {
            const std::size_t bin = observation_bin(
                Kokkos::log10(frequency), log_min, log_max, spectrum_bins);
            products.spectrum_I[bin] += photons.I_host(i);
        }
    }
    return products;
}
