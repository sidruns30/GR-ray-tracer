#include <cmath>
#include <cstdlib>

#include "../analysis/geodesic_observables.hpp"
#include "../radiative_transfer/scattering.hpp"

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    bool passed = true;
    {
        Kokkos::View<real*> r("r", 1);
        Kokkos::View<real*> theta("theta", 1);
        Kokkos::View<real*> phi("phi", 1);
        Kokkos::View<real***> density("density", 1, 1, 1);
        Kokkos::View<real***> temperature("temperature", 1, 1, 1);
        Kokkos::View<real****> velocity("velocity", 1, 1, 1, 4);
        Kokkos::View<real****> magnetic("magnetic", 1, 1, 1, 4);
        Kokkos::deep_copy(r, 10.0);
        Kokkos::deep_copy(theta, PI / 2.0);
        Kokkos::deep_copy(phi, 0.0);
        Kokkos::deep_copy(density, 1.0);
        Kokkos::deep_copy(temperature, 2.0);
        Kokkos::deep_copy(velocity, 0.0);
        Kokkos::deep_copy(magnetic, 0.0);
        auto velocity_host = Kokkos::create_mirror_view(velocity);
        velocity_host(0, 0, 0, 0) = 1.0;
        Kokkos::deep_copy(velocity, velocity_host);

        ScatteringFluidGrid grid{
            r, theta, phi, density, temperature, velocity, magnetic,
            0.1, 0.1, 0.1,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            a_BH, M_BH, true};
        ScatteringModel model{true, 1.0e6, 0.5, 42};
        ScatteringRandomPool random_pool(model.seed);
        const real spin = a_BH;
        const real mass = M_BH;

        real position[4];
        spherical_kerr_schild_to_cartesian(10.0, PI / 2.0, 0.0, a_BH, position);
        const real direction[3] = {-1.0, 0.0, 0.0};
        real momentum[4];
        passed = passed && kerr_schild::null_covariant_momentum_from_spatial_direction(
            position, direction, a_BH, M_BH, momentum);

        Kokkos::View<real*> state("state", 8);
        Kokkos::View<real*> stokes("stokes", 4);
        Kokkos::View<real*> frequency("frequency", 1);
        Kokkos::View<int*> flags("flags", 2);
        auto state_host = Kokkos::create_mirror_view(state);
        for (int component = 0; component < 4; ++component) {
            state_host(component) = position[component];
            state_host(IKT + component) = momentum[component];
        }
        Kokkos::deep_copy(state, state_host);
        auto stokes_host = Kokkos::create_mirror_view(stokes);
        stokes_host(0) = 2.0;
        stokes_host(1) = 1.0;
        stokes_host(2) = 1.0;
        stokes_host(3) = 1.0;
        Kokkos::deep_copy(stokes, stokes_host);
        Kokkos::deep_copy(frequency, 2.3e11);

        Kokkos::parallel_for("TestFluidScattering", 1, KOKKOS_LAMBDA(int) {
            real local_state[8];
            real local_stokes[4];
            for (int component = 0; component < 8; ++component) {
                local_state[component] = state(component);
            }
            for (int component = 0; component < 4; ++component) {
                local_stokes[component] = stokes(component);
            }
            FluidCellCGS fluid;
            real tetrad[4][4];
            const bool sampled = sample_scattering_fluid(local_state, grid, fluid, tetrad);
            bool scattered = false;
            if (sampled) {
                auto random = random_pool.get_state();
                real local_frequency = frequency(0);
                scattered = maybe_scatter_photon(
                    local_state, local_stokes, local_frequency, model,
                    fluid, tetrad, spin, mass, random);
                frequency(0) = local_frequency;
                random_pool.free_state(random);
            }
            for (int component = 0; component < 8; ++component) {
                state(component) = local_state[component];
            }
            for (int component = 0; component < 4; ++component) {
                stokes(component) = local_stokes[component];
            }
            flags(0) = sampled ? 1 : 0;
            flags(1) = scattered ? 1 : 0;
        });

        state_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, state);
        stokes_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, stokes);
        const auto frequency_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, frequency);
        const auto flags_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, flags);
        real final_state[8];
        for (int component = 0; component < 8; ++component) {
            final_state[component] = state_host(component);
        }
        passed = passed && flags_host(0) == 1 && flags_host(1) == 1;
        passed = passed && Kokkos::abs(stokes_host(0) - 1.0) < 1.0e-14;
        passed = passed && frequency_host(0) == 2.3e11;
        passed = passed && Kokkos::abs(
            compute_geodesic_observables(final_state, a_BH, M_BH).norm) < 1.0e-10;

        // A grid explicitly marked unavailable represents vacuum/missing fluid
        // data and must prevent the scattering function from being reached.
        ScatteringFluidGrid vacuum_grid;
        FluidCellCGS unused_fluid;
        real unused_tetrad[4][4];
        passed = passed && !sample_scattering_fluid(
            final_state, vacuum_grid, unused_fluid, unused_tetrad);
    }
    Kokkos::finalize();
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
