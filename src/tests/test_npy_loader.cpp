#include <cstdlib>
#include <filesystem>
#include <vector>

#include "../input/load_python_arrays.hpp"
#include "../radiative_transfer/scattering.hpp"

int main() {
    Kokkos::initialize();
    bool passed = true;
    {
        const auto dir = std::filesystem::temp_directory_path() / "grraytracer_npy_loader";
        std::filesystem::create_directories(dir);

        std::vector<double> r(3 * 2 * 4);
        std::vector<double> theta(3 * 2 * 4);
        std::vector<double> phi(3 * 2 * 4);
        std::vector<double> density(3 * 2 * 4);
        std::vector<double> temperature(3 * 2 * 4);
        std::vector<double> velocity(3 * 2 * 4 * 4);
        std::vector<double> magnetic(3 * 2 * 4 * 4);
        for (std::size_t i = 0; i < 3; ++i) {
            for (std::size_t j = 0; j < 2; ++j) {
                for (std::size_t k = 0; k < 4; ++k) {
                    const std::size_t index = (i * 2 + j) * 4 + k;
                    // Deliberately nonuniform, fully three-dimensional coordinates.
                    r[index] = 1.0 + static_cast<double>(i * i) + 0.01 * j + 0.001 * k;
                    theta[index] = 0.1 + 0.2 * static_cast<double>(j * j) + 0.005 * i;
                    phi[index] = 0.05 + 0.3 * static_cast<double>(k * k) + 0.002 * j;
                    density[index] = static_cast<double>(index + 1);
                    temperature[index] = 100.0 + static_cast<double>(index);
                }
            }
        }
        for (std::size_t i = 0; i < velocity.size(); ++i) {
            velocity[i] = 0.01 * static_cast<double>(i + 1);
            magnetic[i] = 0.02 * static_cast<double>(i + 1);
        }

        cnpy::npy_save((dir / "r.npy").string(), r.data(), {3, 2, 4});
        cnpy::npy_save((dir / "theta.npy").string(), theta.data(), {3, 2, 4});
        cnpy::npy_save((dir / "phi.npy").string(), phi.data(), {3, 2, 4});
        cnpy::npy_save((dir / "rho.npy").string(), density.data(), {3, 2, 4});
        cnpy::npy_save((dir / "Tgas.npy").string(), temperature.data(), {3, 2, 4});
        cnpy::npy_save((dir / "vel.npy").string(), velocity.data(), {3, 2, 4, 4});
        cnpy::npy_save((dir / "mag.npy").string(), magnetic.data(), {3, 2, 4, 4});

        NumpyFieldPaths paths;
        paths.r = (dir / "r.npy").string();
        paths.theta = (dir / "theta.npy").string();
        paths.phi = (dir / "phi.npy").string();
        paths.density = (dir / "rho.npy").string();
        paths.temperature = (dir / "Tgas.npy").string();
        paths.velocity = (dir / "vel.npy").string();
        paths.magnetic = (dir / "mag.npy").string();

        auto views = load_numpy_field_bundle(paths);
        passed = passed && views.r.extent(0) == 3 &&
            views.r.extent(1) == 2 && views.r.extent(2) == 4;
        passed = passed && views.theta.extent(0) == 3 &&
            views.theta.extent(1) == 2 && views.theta.extent(2) == 4;
        passed = passed && views.phi.extent(0) == 3 &&
            views.phi.extent(1) == 2 && views.phi.extent(2) == 4;
        passed = passed && views.density.extent(0) == 3 &&
            views.density.extent(1) == 2 && views.density.extent(2) == 4;
        passed = passed && views.velocity.extent(3) == 4;
        passed = passed && views.magnetic.extent(3) == 4;

        auto r_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, views.r);
        auto rho_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, views.density);
        passed = passed && r_host(2, 1, 3) == r[2 * 2 * 4 + 1 * 4 + 3];
        passed = passed && rho_host(2, 1, 3) == density[2 * 2 * 4 + 1 * 4 + 3];

        // The device lookup must recover a cell from nonuniform 3D coordinate
        // fields without relying on constant axis spacing.
        Kokkos::View<int*> located_cell("located_cell", 4);
        const real query_r = r[2 * 2 * 4 + 1 * 4 + 3];
        const real query_theta = theta[2 * 2 * 4 + 1 * 4 + 3];
        const real query_phi = phi[2 * 2 * 4 + 1 * 4 + 3];
        const auto radial_grid = views.r;
        const auto polar_grid = views.theta;
        const auto azimuth_grid = views.phi;
        Kokkos::parallel_for("LocateNonuniformCell", 1, KOKKOS_LAMBDA(int) {
            int i = -1;
            int j = -1;
            int k = -1;
            located_cell(0) = locate_structured_grid_cell(
                radial_grid, polar_grid, azimuth_grid,
                query_r, query_theta, query_phi, i, j, k) ? 1 : 0;
            located_cell(1) = i;
            located_cell(2) = j;
            located_cell(3) = k;
        });
        const auto located_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, located_cell);
        passed = passed && located_host(0) == 1 &&
            located_host(1) == 2 && located_host(2) == 1 && located_host(3) == 3;

        // A -1 infers the phi partition count from the number of MPI tasks.
        // The resulting local fields must contain the matching global slice.
        const DomainDecompositionSpec phi_decomposition{{1, 1, -1}};
        auto rank0 = load_numpy_field_bundle(paths, phi_decomposition, 0, 2);
        auto rank1 = load_numpy_field_bundle(paths, phi_decomposition, 1, 2);
        passed = passed && rank0.decomposition.partitions[2] == 2;
        passed = passed && rank0.phi.extent(2) == 2 && rank1.phi.extent(2) == 2;
        auto rank0_rho = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, rank0.density);
        auto rank1_rho = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, rank1.density);
        passed = passed && rank0_rho(2, 1, 0) == density[2 * 2 * 4 + 1 * 4];
        passed = passed && rank1_rho(2, 1, 0) == density[2 * 2 * 4 + 1 * 4 + 2];
    }
    Kokkos::finalize();
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
