#include <cmath>
#include <cstdint>
#include <cstdlib>

#include "../analysis/geodesic_observables.hpp"
#include "../radiative_transfer/initialize_photons.hpp"

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    bool passed = true;
    {
        constexpr int packets_per_cell = 64;
        Kokkos::View<real***> r("r", 1, 1, 1);
        Kokkos::View<real***> theta("theta", 1, 1, 1);
        Kokkos::View<real***> phi("phi", 1, 1, 1);
        Kokkos::View<real***> density("density", 1, 1, 1);
        Kokkos::View<real***> temperature("temperature", 1, 1, 1);
        Kokkos::View<real****> velocity("velocity", 1, 1, 1, 4);
        Kokkos::View<real****> magnetic("magnetic", 1, 1, 1, 4);

        Kokkos::deep_copy(r, 10.0);
        Kokkos::deep_copy(theta, PI / 2.0);
        Kokkos::deep_copy(phi, 0.25);
        Kokkos::deep_copy(density, 2.0);
        Kokkos::deep_copy(temperature, 1.0);
        Kokkos::deep_copy(velocity, 0.0);
        Kokkos::deep_copy(magnetic, 0.0);
        auto velocity_host = Kokkos::create_mirror_view(velocity);
        velocity_host(0, 0, 0, 0) = 1.0;
        Kokkos::deep_copy(velocity, velocity_host);

        UnitConversions units;
        units.temperature_k_per_code = 1.0e10;
        PhotonGenerationConfig generation;
        generation.generator = PhotonGeneratorType::Blackbody;
        // density = 2.0 (code units) * normalization = 32.0 -> int(64.0) ==
        // packets_per_cell superphotons from the single cell.
        generation.superphoton_count_normalization = 32.0;
        generation.energy_per_cell_erg = 8.0;

        const auto cell_offsets = build_superphoton_cell_offsets(
            density, generation.superphoton_count_normalization);
        passed = passed && cell_offsets.total_photons == static_cast<std::uint64_t>(packets_per_cell);

        Photons photons(packets_per_cell);
        constexpr std::uint64_t large_id_offset = 50000000000ULL;
        initialize_photons_disk(
            packets_per_cell, 0, large_id_offset, r, theta, phi, density, temperature,
            velocity, magnetic, generation, units, cell_offsets.offsets, photons);
        Kokkos::fence();

        auto ids = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, photons.id);
        auto frequencies = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, photons.frequency);
        auto energies = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, photons.I);
        auto terminated = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, photons.terminate);
        auto phases = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, photons.phase);
        auto x0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, photons.x0);
        auto x1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, photons.x1);
        auto x2 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, photons.x2);
        auto x3 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, photons.x3);
        auto k0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, photons.k0);
        auto k1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, photons.k1);
        auto k2 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, photons.k2);
        auto k3 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, photons.k3);

        real total_energy = 0.0;
        for (int i = 0; i < packets_per_cell; ++i) {
            passed = passed && ids(i) == large_id_offset + static_cast<std::uint64_t>(i);
            passed = passed && frequencies(i) > 0.0 && Kokkos::isfinite(frequencies(i));
            passed = passed && !terminated(i);
            passed = passed &&
                phases(i) == static_cast<std::uint8_t>(PhotonPhase::Disk);
            total_energy += energies(i);
            const real state[8] = {
                x0(i), x1(i), x2(i), x3(i), k0(i), k1(i), k2(i), k3(i)};
            passed = passed &&
                Kokkos::abs(compute_geodesic_observables(state, a_BH, M_BH).norm) < 1.0e-10;
        }
        passed = passed &&
            Kokkos::abs(total_energy - generation.energy_per_cell_erg) < 1.0e-12;

        // Power-law generation samples the configured energy spectrum and
        // divides the cell's total represented energy equally among packets.
        generation.generator = PhotonGeneratorType::PowerLaw;
        generation.nu_min_hz = 1.0e9;
        generation.nu_max_hz = 1.0e12;
        generation.power_law_slope = 2.5;
        initialize_photons_disk(
            packets_per_cell, 0, 1000, r, theta, phi, density, temperature,
            velocity, magnetic, generation, units, cell_offsets.offsets, photons);
        Kokkos::fence();
        frequencies = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, photons.frequency);
        for (int i = 0; i < packets_per_cell; ++i) {
            passed = passed && frequencies(i) >= generation.nu_min_hz;
            passed = passed && frequencies(i) <= generation.nu_max_hz;
        }
    }
    Kokkos::finalize();
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
