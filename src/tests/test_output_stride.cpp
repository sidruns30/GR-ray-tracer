#include <cstdlib>
#include <cstdint>

#include "../output/write_output.hpp"

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    bool passed = false;
    {
        constexpr std::uint64_t first_id = 50000000000ULL;
        Kokkos::View<std::uint64_t*, Kokkos::HostSpace> ids("ids", 10);
        for (std::uint64_t index = 0; index < 10; ++index) ids(index) = first_id + index;

        const auto every_third = make_strided_host_values(ids, ids.extent(0), 3);
        passed = every_third.size() == 4 &&
            every_third[0] == first_id && every_third[1] == first_id + 3 &&
            every_third[2] == first_id + 6 && every_third[3] == first_id + 9;

        Photons photons(2);
        auto frequency = Kokkos::create_mirror_view(photons.frequency);
        auto intensity = Kokkos::create_mirror_view(photons.I);
        auto phase = Kokkos::create_mirror_view(photons.phase);
        auto screen_u = Kokkos::create_mirror_view(photons.theta_disp);
        auto screen_v = Kokkos::create_mirror_view(photons.phi_disp);
        frequency(0) = frequency(1) = 1.0e12;
        intensity(0) = 2.0;
        intensity(1) = 100.0;
        phase(0) = static_cast<std::uint8_t>(PhotonPhase::ImageArrived);
        phase(1) = static_cast<std::uint8_t>(PhotonPhase::Rejected);
        screen_u(0) = screen_u(1) = 0.0;
        screen_v(0) = screen_v(1) = 0.0;
        Kokkos::deep_copy(photons.frequency, frequency);
        Kokkos::deep_copy(photons.I, intensity);
        Kokkos::deep_copy(photons.Q, 0.0);
        Kokkos::deep_copy(photons.U, 0.0);
        Kokkos::deep_copy(photons.V, 0.0);
        Kokkos::deep_copy(photons.phase, phase);
        Kokkos::deep_copy(photons.theta_disp, screen_u);
        Kokkos::deep_copy(photons.phi_disp, screen_v);
        photons.copy_observation_to_host();
        const ObservationProducts products = build_observation_products(
            photons, 4, 4, 8, 1.0e9, 1.0e15, 20.0, 20.0);
        real image_sum = 0.0;
        for (const real value : products.image_I) image_sum += value;
        passed = passed && products.arrived_count == 1 && image_sum == 2.0;
    }
    Kokkos::finalize();
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
