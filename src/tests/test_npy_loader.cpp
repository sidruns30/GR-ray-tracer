#include <cstdlib>
#include <filesystem>
#include <vector>

#include "../input/load_python_arrays.hpp"

int main() {
    Kokkos::initialize();
    bool passed = true;
    {
        const auto dir = std::filesystem::temp_directory_path() / "grraytracer_npy_loader";
        std::filesystem::create_directories(dir);

        const std::vector<double> r = {1.0, 2.0, 4.0};
        const std::vector<double> theta = {0.1, 0.2};
        const std::vector<double> phi = {0.0, 1.0, 2.0, 3.0};
        std::vector<double> density(3 * 2 * 4);
        std::vector<double> temperature(3 * 2 * 4);
        std::vector<double> velocity(3 * 2 * 4 * 4);
        std::vector<double> magnetic(3 * 2 * 4 * 4);
        for (std::size_t i = 0; i < density.size(); ++i) {
            density[i] = static_cast<double>(i + 1);
            temperature[i] = 100.0 + static_cast<double>(i);
        }
        for (std::size_t i = 0; i < velocity.size(); ++i) {
            velocity[i] = 0.01 * static_cast<double>(i + 1);
            magnetic[i] = 0.02 * static_cast<double>(i + 1);
        }

        cnpy::npy_save((dir / "r.npy").string(), r.data(), {r.size()});
        cnpy::npy_save((dir / "theta.npy").string(), theta.data(), {theta.size()});
        cnpy::npy_save((dir / "phi.npy").string(), phi.data(), {phi.size()});
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
        passed = passed && views.r.extent(0) == r.size();
        passed = passed && views.theta.extent(0) == theta.size();
        passed = passed && views.phi.extent(0) == phi.size();
        passed = passed && views.density.extent(0) == 3 &&
            views.density.extent(1) == 2 && views.density.extent(2) == 4;
        passed = passed && views.velocity.extent(3) == 4;
        passed = passed && views.magnetic.extent(3) == 4;

        auto r_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, views.r);
        auto rho_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, views.density);
        passed = passed && r_host(0) == 1.0;
        passed = passed && rho_host(2, 1, 3) == density[2 * 2 * 4 + 1 * 4 + 3];

        // A -1 infers the phi partition count from the number of MPI tasks.
        // The resulting local fields must contain the matching global slice.
        const DomainDecompositionSpec phi_decomposition{{1, 1, -1}};
        auto rank0 = load_numpy_field_bundle(paths, phi_decomposition, 0, 2);
        auto rank1 = load_numpy_field_bundle(paths, phi_decomposition, 1, 2);
        passed = passed && rank0.decomposition.partitions[2] == 2;
        passed = passed && rank0.phi.extent(0) == 2 && rank1.phi.extent(0) == 2;
        auto rank0_rho = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, rank0.density);
        auto rank1_rho = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, rank1.density);
        passed = passed && rank0_rho(2, 1, 0) == density[2 * 2 * 4 + 1 * 4];
        passed = passed && rank1_rho(2, 1, 0) == density[2 * 2 * 4 + 1 * 4 + 2];
    }
    Kokkos::finalize();
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
