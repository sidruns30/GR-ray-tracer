#include <cassert>
#include <filesystem>
#include <vector>

#include "../input/load_python_arrays.hpp"

int main() {
    Kokkos::initialize();
    {
        const auto dir = std::filesystem::temp_directory_path() / "grraytracer_npy_loader";
        std::filesystem::create_directories(dir);

        const std::vector<double> r = {1.0, 2.0, 4.0};
        const std::vector<double> theta = {0.1, 0.2};
        const std::vector<double> phi = {0.0, 1.0, 2.0, 3.0};
        std::vector<double> density(3 * 2 * 4);
        std::vector<double> temperature(3 * 2 * 4);
        std::vector<double> velocity(3 * 2 * 4 * 3);
        std::vector<double> magnetic(3 * 2 * 4 * 3);
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
        cnpy::npy_save((dir / "vel.npy").string(), velocity.data(), {3, 2, 4, 3});
        cnpy::npy_save((dir / "mag.npy").string(), magnetic.data(), {3, 2, 4, 3});

        NumpyFieldPaths paths;
        paths.r = (dir / "r.npy").string();
        paths.theta = (dir / "theta.npy").string();
        paths.phi = (dir / "phi.npy").string();
        paths.density = (dir / "rho.npy").string();
        paths.temperature = (dir / "Tgas.npy").string();
        paths.velocity = (dir / "vel.npy").string();
        paths.magnetic = (dir / "mag.npy").string();

        auto views = load_numpy_field_bundle(paths);
        assert(views.r.extent(0) == r.size());
        assert(views.theta.extent(0) == theta.size());
        assert(views.phi.extent(0) == phi.size());
        assert(views.density.extent(0) == 3 && views.density.extent(1) == 2 && views.density.extent(2) == 4);
        assert(views.velocity.extent(3) == 3);
        assert(views.magnetic.extent(3) == 3);

        auto r_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, views.r);
        auto rho_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, views.density);
        assert(r_host(0) == 1.0);
        assert(rho_host(2, 1, 3) == density[2 * 2 * 4 + 1 * 4 + 3]);
    }
    Kokkos::finalize();
    return 0;
}
