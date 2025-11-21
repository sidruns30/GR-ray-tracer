#pragma once
#include "../utils.hpp"
#include "../utils.cpp"

using PhotonView = Kokkos::View<double**>;

PhotonView gather_photons(PhotonView local_photons) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t local_N = local_photons.extent(0);
    size_t n_cols = local_photons.extent(1);

    std::vector<int> counts(size);
    int local_count = static_cast<int>(local_N * n_cols);
    MPI_Gather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

   

    std::vector<int> displs(size, 0);
    int total_count = 0;
    if (rank == 0) {
        for (int i = 0; i < size; ++i) {
            displs[i] = total_count;
            total_count += counts[i];
        }
    }

    std::vector<double> local_flat(local_count);
    auto local_host = Kokkos::create_mirror_view(local_photons);
    Kokkos::deep_copy(local_host, local_photons);

    for (size_t i = 0; i < local_N; ++i)
        for (size_t j = 0; j < n_cols; ++j)
            local_flat[i*n_cols + j] = local_host(i,j);

    // Step 3: Gather all flattened data on rank 0
    std::vector<double> all_flat;
    if (rank == 0) all_flat.resize(total_count);

    MPI_Gatherv(local_flat.data(), local_count, MPI_DOUBLE,
                all_flat.data(), counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Step 4: Copy to Kokkos view on rank 0
    PhotonView gathered_photons;
    if (rank == 0) {
        size_t total_rows = total_count / n_cols;
        gathered_photons = PhotonView("gathered_photons", total_rows, n_cols);

        auto host_view = Kokkos::create_mirror_view(gathered_photons);
        for (size_t i = 0; i < total_rows; ++i)
            for (size_t j = 0; j < n_cols; ++j)
                host_view(i,j) = all_flat[i*n_cols + j];

        Kokkos::deep_copy(gathered_photons, host_view);
    }

    return gathered_photons; // valid only on rank 0
}

// Write photon data to output file -> need to add MPI
void write_photon_output(const PhotonView& photons, const size_t step) {
    const std::string filename = output_directory + "photons_step_" + std::to_string(step) + ".npy";
    // Create host mirror
    auto host_photons = Kokkos::create_mirror_view(photons);
    Kokkos::deep_copy(host_photons, photons);
    size_t N = photons.extent(0);
    size_t n_cols = photons.extent(1);
    // Write to numpy file using cnpy
    std::vector<size_t> shape = {N, n_cols};
    std::vector<double> data(N * n_cols);
    for (size_t i = 0; i < N; ++i)
    {
        for (size_t j = 0; j < n_cols; ++j)
        {
            data[i * n_cols + j] = host_photons(i, j);
        }
    }
    cnpy::npy_save(filename, &data[0], shape, "w");
    if (verbose) {
        std::cout << "Wrote photon data to " << filename << " with shape ["
                  << N << ", " << n_cols << "]." << std::endl;
    }
}