#pragma once

#include <algorithm>
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
    std::vector<real> spectrum_frequency_hz;
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
    products.spectrum_frequency_hz.assign(nnu, 0.0);
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

inline std::size_t bin_from_value(real value, real min_value, real span, std::size_t bins) {
    const real scaled = (value - min_value) / span;
    return static_cast<std::size_t>(std::clamp<real>(scaled * static_cast<real>(bins - 1), 0.0, static_cast<real>(bins - 1)));
}

// The caller must copy the required photon fields to host before invoking this
// function. Keeping that transfer at the checkpoint boundary avoids duplicate
// device-to-host copies when raw photon fields and observations are both saved.
inline ObservationProducts build_observation_products(const Photons& photons,
                                                      std::size_t image_bins = default_image_bins,
                                                      std::size_t lightcurve_bins = default_lightcurve_bins,
                                                      std::size_t spectrum_bins = default_spectrum_bins) {
    ObservationProducts products = make_observation_products(image_bins, image_bins, lightcurve_bins, spectrum_bins);
    const std::size_t n = photons.x0_host.extent(0);
    if (n == 0) {
        return products;
    }

    // Fixed screen-space bounds make pixels comparable across steps and ranks.
    real screen_half1, screen_half2;
    if (use_image_camera) {
        screen_half1 = plane_dim1 / 2.0;
        screen_half2 = plane_dim2 / 2.0;
    } else if (use_pinhole_camera) {
        screen_half1 = pinhole_aperture_radius;
        screen_half2 = pinhole_aperture_radius;
    } else {
        screen_half1 = 1.0;
        screen_half2 = 1.0;
    }
    const real screen_span1 = std::max(real(1e-15), 2.0 * screen_half1);
    const real screen_span2 = std::max(real(1e-15), 2.0 * screen_half2);

    // These bins use per-rank checkpoint bounds; their indices are not global axes.
    real t_min = photons.x0_host(0), t_max = photons.x0_host(0);
    real nu_min = photons.frequency_host(0), nu_max = photons.frequency_host(0);
    for (std::size_t i = 1; i < n; ++i) {
        t_min = std::min(t_min, photons.x0_host(i));
        t_max = std::max(t_max, photons.x0_host(i));
        nu_min = std::min(nu_min, photons.frequency_host(i));
        nu_max = std::max(nu_max, photons.frequency_host(i));
    }
    const real t_span = std::max(real(1e-15), t_max - t_min);
    const real log_nu_min = std::log10(std::max(real(1e-300), nu_min));
    const real log_nu_max = std::log10(std::max(real(1e-300), nu_max));
    const real log_nu_span = std::max(real(1e-15), log_nu_max - log_nu_min);

    for (std::size_t bin = 0; bin < spectrum_bins; ++bin) {
        const real fraction = spectrum_bins > 1
            ? static_cast<real>(bin) / static_cast<real>(spectrum_bins - 1)
            : 0.0;
        products.spectrum_frequency_hz[bin] =
            std::pow(real(10.0), log_nu_min + fraction * log_nu_span);
    }

    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t pixel_x = bin_from_value(photons.theta_disp_host(i), -screen_half1, screen_span1, image_bins);
        const std::size_t pixel_y = bin_from_value(photons.phi_disp_host(i), -screen_half2, screen_span2, image_bins);
        const std::size_t time_bin = bin_from_value(photons.x0_host(i), t_min, t_span, lightcurve_bins);
        const std::size_t spectrum_bin = bin_from_value(
            std::log10(std::max(real(1e-300), photons.frequency_host(i))),
            log_nu_min, log_nu_span, spectrum_bins);
        accumulate_observation(products, photons, i, pixel_x, pixel_y, time_bin, spectrum_bin);
    }

    return products;
}
