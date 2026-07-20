#pragma once

#include <algorithm>
#include <filesystem>
#include <vector>

#include "../utils.hpp"

struct ObservationProducts {
    std::size_t image_nx = 0;
    std::size_t image_ny = 0;
    std::size_t lightcurve_bins = 0;
    std::size_t spectrum_bins = 0;
    std::vector<real> image_I;
    std::vector<real> image_Q;
    std::vector<real> image_U;
    std::vector<real> image_V;
    std::vector<real> lightcurve_I;
    std::vector<real> spectrum_I;
};

inline constexpr std::size_t default_image_bins = 256;
inline constexpr std::size_t default_lightcurve_bins = 128;
inline constexpr std::size_t default_spectrum_bins = 128;

inline ObservationProducts make_observation_products(std::size_t nx, std::size_t ny,
                                                     std::size_t nt, std::size_t nnu) {
    ObservationProducts products;
    products.image_nx = nx;
    products.image_ny = ny;
    products.lightcurve_bins = nt;
    products.spectrum_bins = nnu;
    products.image_I.assign(nx * ny, 0.0);
    products.image_Q.assign(nx * ny, 0.0);
    products.image_U.assign(nx * ny, 0.0);
    products.image_V.assign(nx * ny, 0.0);
    products.lightcurve_I.assign(nt, 0.0);
    products.spectrum_I.assign(nnu, 0.0);
    return products;
}

inline void accumulate_observation(ObservationProducts& products,
                                   const Photons& photons,
                                   std::size_t photon_index,
                                   std::size_t pixel_x,
                                   std::size_t pixel_y,
                                   std::size_t time_bin,
                                   std::size_t spectrum_bin) {
    if (pixel_x < products.image_nx && pixel_y < products.image_ny) {
        const std::size_t idx = pixel_y * products.image_nx + pixel_x;
        products.image_I[idx] += photons.I_host(photon_index);
        products.image_Q[idx] += photons.Q_host(photon_index);
        products.image_U[idx] += photons.U_host(photon_index);
        products.image_V[idx] += photons.V_host(photon_index);
    }
    if (time_bin < products.lightcurve_bins) {
        products.lightcurve_I[time_bin] += photons.I_host(photon_index);
    }
    if (spectrum_bin < products.spectrum_bins) {
        products.spectrum_I[spectrum_bin] += photons.I_host(photon_index);
    }
}

inline ObservationProducts build_observation_products(Photons& photons,
                                                      std::size_t image_bins = default_image_bins,
                                                      std::size_t lightcurve_bins = default_lightcurve_bins,
                                                      std::size_t spectrum_bins = default_spectrum_bins) {
    ObservationProducts products = make_observation_products(image_bins, image_bins, lightcurve_bins, spectrum_bins);
    photons.copy_to_host();
    const std::size_t n = photons.x0_host.extent(0);
    if (n == 0) {
        return products;
    }

    real x_min = photons.x1_host(0), x_max = photons.x1_host(0);
    real y_min = photons.x2_host(0), y_max = photons.x2_host(0);
    real t_min = photons.x0_host(0), t_max = photons.x0_host(0);
    real nu_min = photons.k0_host(0), nu_max = photons.k0_host(0);
    for (std::size_t i = 1; i < n; ++i) {
        x_min = std::min(x_min, photons.x1_host(i));
        x_max = std::max(x_max, photons.x1_host(i));
        y_min = std::min(y_min, photons.x2_host(i));
        y_max = std::max(y_max, photons.x2_host(i));
        t_min = std::min(t_min, photons.x0_host(i));
        t_max = std::max(t_max, photons.x0_host(i));
        nu_min = std::min(nu_min, photons.k0_host(i));
        nu_max = std::max(nu_max, photons.k0_host(i));
    }

    const real x_span = std::max(real(1e-15), x_max - x_min);
    const real y_span = std::max(real(1e-15), y_max - y_min);
    const real t_span = std::max(real(1e-15), t_max - t_min);
    const real nu_span = std::max(real(1e-15), nu_max - nu_min);

    for (std::size_t i = 0; i < n; ++i) {
        const auto bin_from_value = [](real value, real min_value, real span, std::size_t bins) {
            const real scaled = (value - min_value) / span;
            const auto idx = static_cast<std::size_t>(std::clamp<real>(scaled * static_cast<real>(bins - 1), 0.0, static_cast<real>(bins - 1)));
            return idx;
        };
        const std::size_t pixel_x = bin_from_value(photons.x1_host(i), x_min, x_span, image_bins);
        const std::size_t pixel_y = bin_from_value(photons.x2_host(i), y_min, y_span, image_bins);
        const std::size_t time_bin = bin_from_value(photons.x0_host(i), t_min, t_span, lightcurve_bins);
        const std::size_t spectrum_bin = bin_from_value(photons.k0_host(i), nu_min, nu_span, spectrum_bins);
        accumulate_observation(products, photons, i, pixel_x, pixel_y, time_bin, spectrum_bin);
    }

    return products;
}

inline void write_observation_products(Photons& photons, std::size_t step, int rank) {
    std::filesystem::create_directories(output_directory);
    const auto products = build_observation_products(photons);
    const std::string prefix = output_directory + "observation_step_" + std::to_string(step) + "_rank" + std::to_string(rank);
    cnpy::npy_save(prefix + "_image_I.npy", products.image_I.data(), {products.image_nx, products.image_ny});
    cnpy::npy_save(prefix + "_image_Q.npy", products.image_Q.data(), {products.image_nx, products.image_ny});
    cnpy::npy_save(prefix + "_image_U.npy", products.image_U.data(), {products.image_nx, products.image_ny});
    cnpy::npy_save(prefix + "_image_V.npy", products.image_V.data(), {products.image_nx, products.image_ny});
    cnpy::npy_save(prefix + "_lightcurve_I.npy", products.lightcurve_I.data(), {products.lightcurve_bins});
    cnpy::npy_save(prefix + "_spectrum_I.npy", products.spectrum_I.data(), {products.spectrum_bins});
}
