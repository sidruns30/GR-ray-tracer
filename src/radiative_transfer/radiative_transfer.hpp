#pragma once

#include "../utils.hpp"
#include "photon_generation.hpp"

// ============================================================================
// Radiative transfer: optical depth (absorption) and emissivity.
//
// This models the two terms of the radiative transfer equation as
// independent, user-editable effects applied once per geodesic integration
// step, at the same rank-local fluid cell scattering already samples (see
// sample_scattering_fluid in scattering.hpp):
//
//   1. Optical depth ONLY attenuates the existing Stokes I, Q, U, V,
//      multiplicatively, by exp(-dtau). It never adds intensity.
//   2. Emissivity ONLY adds newly emitted intensity to Stokes I. It is not
//      itself attenuated by this step's own optical depth (i.e. a photon
//      does not partially self-absorb the radiation it picks up in the same
//      step -- accumulate over enough steps and this converges to the usual
//      coupled solution for typical step sizes).
//
// Both extension points below receive the FluidCellCGS sampled at the
// photon's current position (density, temperature, magnetic field, and
// four-velocity in CGS units -- see FluidCellCGS in photon_generation.hpp),
// the photon's current frequency in Hz, and `path_length_step`, the distance
// the integrator is about to advance by (in the same affine-parameter units
// as `dlambda`; multiply by a physical length scale inside your function if
// you need proper CGS path length).
//
// -- ComputeOpticalDepthIncrement --
// Return the optical depth `dtau >= 0` accumulated while crossing this step.
// A physical absorption coefficient alpha_nu [1/length] combined with a path
// length ds gives dtau = alpha_nu * ds. The default body below is a simple
// grey (frequency-independent), density-proportional opacity, scaled by
// `model.absorption_coefficient` (set in the simulation TOML's
// [radiative_transfer] section) -- replace the body to use `frequency_hz`,
// `fluid.temperature_k`, or `fluid.magnetic_gauss` for a physical opacity
// (e.g. Kramers free-free: alpha_nu ~ rho^2 * T^-1.5 * nu^-2).
//
// -- ComputeEmissivityIncrement --
// Return the intensity added while crossing this step, in the same units as
// Stokes I. A physical emission coefficient j_nu combined with a path length
// ds gives dI = j_nu * ds. The default body below is a simple grey thermal
// emission proportional to density * temperature, scaled by
// `model.emission_coefficient` -- replace the body with, for example,
// thermal synchrotron emissivity using `fluid.magnetic_gauss` and
// `fluid.temperature_k`.
//
// To disable one effect entirely, set `absorption_enabled` /
// `emission_enabled` to false in [radiative_transfer] rather than editing
// these bodies; they are only called when their respective flag is set.
// ============================================================================

struct RadiativeTransferModel {
    bool absorption_enabled = false;
    bool emission_enabled = false;
    real absorption_coefficient = 0.0;
    real emission_coefficient = 0.0;
};

KOKKOS_INLINE_FUNCTION
real ComputeOpticalDepthIncrement(const FluidCellCGS& fluid, real frequency_hz,
                                  real path_length_step, real absorption_coefficient) {
    (void)frequency_hz;
    return Kokkos::fmax(real(0.0),
        absorption_coefficient * fluid.density_g_cm3 * Kokkos::abs(path_length_step));
}

KOKKOS_INLINE_FUNCTION
real ComputeEmissivityIncrement(const FluidCellCGS& fluid, real frequency_hz,
                                real path_length_step, real emission_coefficient) {
    (void)frequency_hz;
    return Kokkos::fmax(real(0.0), emission_coefficient * fluid.density_g_cm3 *
        fluid.temperature_k * Kokkos::abs(path_length_step));
}

// Attenuates stokes by exp(-dtau) (if absorption is enabled), then adds
// newly emitted intensity to Stokes I (if emission is enabled). Order
// matches the physical picture of a step of absorbing, emitting plasma.
KOKKOS_INLINE_FUNCTION
void apply_radiative_transfer_step(real stokes[4], const FluidCellCGS& fluid,
                                   real frequency_hz, real path_length_step,
                                   const RadiativeTransferModel& model) {
    if (model.absorption_enabled) {
        const real dtau = ComputeOpticalDepthIncrement(
            fluid, frequency_hz, path_length_step, model.absorption_coefficient);
        const real attenuation = Kokkos::exp(-dtau);
        for (int component = 0; component < 4; ++component) {
            stokes[component] *= attenuation;
        }
    }
    if (model.emission_enabled) {
        stokes[0] += ComputeEmissivityIncrement(
            fluid, frequency_hz, path_length_step, model.emission_coefficient);
    }
}
