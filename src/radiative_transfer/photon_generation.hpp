#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <Kokkos_Core.hpp>

#include "../types.hpp"

enum class PhotonGeneratorType : std::uint8_t {
    Blackbody = 0,
    PowerLaw = 1,
    Custom = 2
};

struct UnitConversions {
    real length_cm_per_code = 1.0;
    real time_s_per_code = 1.0;
    real density_g_cm3_per_code = 1.0;
    real temperature_k_per_code = 1.0;
    real four_velocity_cm_s_per_code = 1.0;
    real magnetic_gauss_per_code = 1.0;

    void validate() const {
        const real values[6] = {
            length_cm_per_code, time_s_per_code, density_g_cm3_per_code,
            temperature_k_per_code, four_velocity_cm_s_per_code,
            magnetic_gauss_per_code
        };
        for (const real value : values) {
            if (!(value > 0.0) || !Kokkos::isfinite(value)) {
                throw std::runtime_error("all [units] conversion factors must be finite and positive");
            }
        }
    }
};

struct PhotonGenerationConfig {
    PhotonGeneratorType generator = PhotonGeneratorType::Blackbody;
    int superphotons_per_cell = 1;
    real energy_per_cell_erg = 1.0;
    real power_law_slope = 2.0;
    real nu_min_hz = 1.0e9;
    real nu_max_hz = 1.0e20;
    real custom_frequency_hz = 2.30e11;

    void validate() const {
        if (superphotons_per_cell <= 0) {
            throw std::runtime_error("photons.superphotons_per_cell must be positive");
        }
        if (!(energy_per_cell_erg >= 0.0) || !Kokkos::isfinite(energy_per_cell_erg)) {
            throw std::runtime_error("photons.energy_per_cell_erg must be finite and non-negative");
        }
        if (generator == PhotonGeneratorType::PowerLaw) {
            if (!Kokkos::isfinite(power_law_slope)) {
                throw std::runtime_error("photons.power_law_slope must be finite");
            }
            if (!(nu_min_hz > 0.0) || !(nu_max_hz > nu_min_hz) ||
                !Kokkos::isfinite(nu_min_hz) || !Kokkos::isfinite(nu_max_hz)) {
                throw std::runtime_error("power-law generation requires 0 < photons.nu_min_hz < photons.nu_max_hz");
            }
        }
        if (generator == PhotonGeneratorType::Custom &&
            (!(custom_frequency_hz > 0.0) || !Kokkos::isfinite(custom_frequency_hz))) {
            throw std::runtime_error("photons.custom_frequency_hz must be finite and positive");
        }
    }
};

inline PhotonGeneratorType parse_photon_generator_type(const std::string& name) {
    if (name == "blackbody") return PhotonGeneratorType::Blackbody;
    if (name == "power-law" || name == "power_law" || name == "powerlaw") {
        return PhotonGeneratorType::PowerLaw;
    }
    if (name == "custom") return PhotonGeneratorType::Custom;
    throw std::runtime_error(
        "photons.generator must be 'blackbody', 'power-law', or 'custom' (got '" + name + "')");
}

struct FluidCellCGS {
    real position_cm[3] = {0.0, 0.0, 0.0};
    real coordinate_time_s = 0.0;
    real density_g_cm3 = 0.0;
    real temperature_k = 0.0;
    // Orthonormal components in the fluid tetrad; the time component is
    // included because GRMHD data commonly stores magnetic four-vectors.
    real magnetic_gauss[4] = {0.0, 0.0, 0.0, 0.0};
    // Cartesian Kerr-Schild coordinate components, converted to cm/s.
    real coordinate_four_velocity_cm_s[4] = {1.0, 0.0, 0.0, 0.0};
};

template <typename RandomGenerator>
KOKKOS_INLINE_FUNCTION
bool GenerateBlackbody(const FluidCellCGS& cell, RandomGenerator& random,
                       int packets_per_cell, real energy_per_cell_erg,
                       real& frequency_hz, real& packet_energy_erg) {
    if (!(cell.temperature_k > 0.0)) return false;

    // x=h*nu/(k*T). The Planck energy distribution is sampled exactly as
    // a mixture of Gamma(4, rate=n) distributions weighted by 1/n^4.
    constexpr real inverse_zeta4 = 0.9239384029215902;
    const real choose = random.drand();
    real cumulative = 0.0;
    int mixture_index = 1;
    for (; mixture_index < 64; ++mixture_index) {
        const real n = static_cast<real>(mixture_index);
        cumulative += inverse_zeta4 / (n * n * n * n);
        if (choose <= cumulative) break;
    }
    const real product = Kokkos::fmax(real(1.0e-300),
        random.drand() * random.drand() * random.drand() * random.drand());
    const real x = -Kokkos::log(product) / static_cast<real>(mixture_index);
    constexpr real boltzmann_over_planck = 2.0836619123327576e10;
    frequency_hz = x * boltzmann_over_planck * cell.temperature_k;
    packet_energy_erg = energy_per_cell_erg / static_cast<real>(packets_per_cell);
    return Kokkos::isfinite(frequency_hz) && frequency_hz > 0.0;
}

template <typename RandomGenerator>
KOKKOS_INLINE_FUNCTION
bool GeneratePowerLaw(const FluidCellCGS&, RandomGenerator& random,
                      int packets_per_cell, real slope,
                      real nu_min_hz, real nu_max_hz, real energy_per_cell_erg,
                      real& frequency_hz, real& packet_energy_erg) {
    const real draw = random.drand();
    if (Kokkos::abs(slope - 1.0) < 1.0e-12) {
        frequency_hz = nu_min_hz * Kokkos::pow(nu_max_hz / nu_min_hz, draw);
    } else {
        const real exponent = 1.0 - slope;
        const real lower = Kokkos::pow(nu_min_hz, exponent);
        const real upper = Kokkos::pow(nu_max_hz, exponent);
        frequency_hz = Kokkos::pow(lower + draw * (upper - lower), 1.0 / exponent);
    }
    packet_energy_erg = energy_per_cell_erg / static_cast<real>(packets_per_cell);
    return Kokkos::isfinite(frequency_hz) && frequency_hz > 0.0;
}

// Extension point for application-specific emissivity. Replace this body to
// use any FluidCellCGS fields needed by the desired physical model.
template <typename RandomGenerator>
KOKKOS_INLINE_FUNCTION
bool GeneratePhotonsCustom(const FluidCellCGS&, RandomGenerator&,
                           int packets_per_cell, real configured_frequency_hz,
                           real energy_per_cell_erg, real& sampled_frequency_hz,
                           real& packet_energy_erg) {
    sampled_frequency_hz = configured_frequency_hz;
    packet_energy_erg = energy_per_cell_erg / static_cast<real>(packets_per_cell);
    return true;
}
