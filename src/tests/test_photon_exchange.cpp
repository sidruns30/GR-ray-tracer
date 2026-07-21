#include <cstdlib>

#include "../mpi/photon_exchange.hpp"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 2) {
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    Kokkos::initialize(argc, argv);
    bool passed = true;
    {
        NumpyFieldViews fields;
        fields.coordinate_min = {9.0, 0.0, rank == 0 ? 0.0 : PI};
        fields.coordinate_max = {11.0, PI, rank == 0 ? PI : 2.0 * PI};
        PhotonDomainMap domains(fields, size);

        Photons photons(1);
        const real azimuth = rank == 0 ? 1.5 * PI : 0.5 * PI;
        auto records = std::vector<PhotonWireRecord>{PhotonWireRecord{
            static_cast<std::uint64_t>(100 + rank), 2.3e11, 1.0,
            0.0, 10.0 * Kokkos::cos(azimuth), 10.0 * Kokkos::sin(azimuth), 0.0,
            -1.0, 1.0, 0.0, 0.0,
            static_cast<real>(rank + 1), 0.0, 0.0, 0.0,
            0.1, static_cast<real>(rank), -static_cast<real>(rank),
            0, static_cast<std::uint8_t>(PhotonPhase::Disk)}};
        photons = host_records_to_photons(records);

        exchange_photon_ownership(photons, domains, rank, size, a_BH);
        const auto received = photons_to_host_records(photons);
        passed = received.size() == 1 &&
            received[0].id == static_cast<std::uint64_t>(100 + (1 - rank)) &&
            received[0].I == static_cast<real>((1 - rank) + 1) &&
            received[0].phase == static_cast<std::uint8_t>(PhotonPhase::Disk);

        // A second exchange has no boundary crossings and exercises the fast
        // path that leaves device-resident photon state untouched.
        exchange_photon_ownership(photons, domains, rank, size, a_BH);
        const auto unchanged = photons_to_host_records(photons);
        passed = passed && unchanged.size() == 1 &&
            unchanged[0].id == static_cast<std::uint64_t>(100 + (1 - rank));
    }
    int local = passed ? 1 : 0;
    int global = 0;
    MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    Kokkos::finalize();
    MPI_Finalize();
    return global == 1 ? EXIT_SUCCESS : EXIT_FAILURE;
}
