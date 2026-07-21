#include <cstdlib>

#include "../radiative_transfer/radiative_transfer.hpp"

int main() {
    bool passed = true;

    FluidCellCGS fluid;
    fluid.density_g_cm3 = 1.0;
    fluid.temperature_k = 2.0;

    // Absorption alone must attenuate all four Stokes components identically
    // and must not add any intensity.
    {
        real stokes[4] = {4.0, 2.0, 1.0, 0.5};
        RadiativeTransferModel model;
        model.absorption_enabled = true;
        model.absorption_coefficient = 1.0;
        apply_radiative_transfer_step(stokes, fluid, 1.0e14, 1.0, model);
        const real expected_attenuation = Kokkos::exp(real(-1.0));
        passed = passed && Kokkos::abs(stokes[0] - 4.0 * expected_attenuation) < 1.0e-12;
        passed = passed && Kokkos::abs(stokes[1] - 2.0 * expected_attenuation) < 1.0e-12;
        passed = passed && Kokkos::abs(stokes[2] - 1.0 * expected_attenuation) < 1.0e-12;
        passed = passed && Kokkos::abs(stokes[3] - 0.5 * expected_attenuation) < 1.0e-12;
    }

    // Emission alone must only add to Stokes I, leaving Q, U, V untouched.
    {
        real stokes[4] = {4.0, 2.0, 1.0, 0.5};
        RadiativeTransferModel model;
        model.emission_enabled = true;
        model.emission_coefficient = 1.0;
        apply_radiative_transfer_step(stokes, fluid, 1.0e14, 1.0, model);
        const real expected_emission =
            model.emission_coefficient * fluid.density_g_cm3 * fluid.temperature_k * 1.0;
        passed = passed && Kokkos::abs(stokes[0] - (4.0 + expected_emission)) < 1.0e-12;
        passed = passed && stokes[1] == 2.0;
        passed = passed && stokes[2] == 1.0;
        passed = passed && stokes[3] == 0.5;
    }

    // Disabling both must leave the Stokes vector bit-for-bit unchanged.
    {
        real stokes[4] = {4.0, 2.0, 1.0, 0.5};
        RadiativeTransferModel model;
        apply_radiative_transfer_step(stokes, fluid, 1.0e14, 1.0, model);
        passed = passed && stokes[0] == 4.0 && stokes[1] == 2.0 &&
                  stokes[2] == 1.0 && stokes[3] == 0.5;
    }

    // A negative path length (as used by backward-traced image rays, where
    // dlambda is flipped negative) must attenuate/emit using its magnitude,
    // not vanish or flip sign.
    {
        real stokes[4] = {4.0, 0.0, 0.0, 0.0};
        RadiativeTransferModel model;
        model.absorption_enabled = true;
        model.absorption_coefficient = 1.0;
        apply_radiative_transfer_step(stokes, fluid, 1.0e14, -1.0, model);
        const real expected_attenuation = Kokkos::exp(real(-1.0));
        passed = passed && Kokkos::abs(stokes[0] - 4.0 * expected_attenuation) < 1.0e-12;
    }

    // Zero density (vacuum-like cell) must leave intensity unchanged by
    // absorption and add no emission.
    {
        FluidCellCGS vacuum_fluid;
        vacuum_fluid.density_g_cm3 = 0.0;
        vacuum_fluid.temperature_k = 5.0;
        real stokes[4] = {4.0, 2.0, 1.0, 0.5};
        RadiativeTransferModel model;
        model.absorption_enabled = true;
        model.emission_enabled = true;
        model.absorption_coefficient = 10.0;
        model.emission_coefficient = 10.0;
        apply_radiative_transfer_step(stokes, vacuum_fluid, 1.0e14, 1.0, model);
        passed = passed && Kokkos::abs(stokes[0] - 4.0) < 1.0e-12;
        passed = passed && stokes[1] == 2.0 && stokes[2] == 1.0 && stokes[3] == 0.5;
    }

    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
