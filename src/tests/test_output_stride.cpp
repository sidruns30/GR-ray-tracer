#include <cstdlib>

#include "../output/write_output.hpp"

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    bool passed = false;
    {
        Kokkos::View<std::uint32_t*, Kokkos::HostSpace> ids("ids", 10);
        for (std::uint32_t index = 0; index < 10; ++index) ids(index) = 100 + index;

        const auto every_third = make_strided_host_values(ids, ids.extent(0), 3);
        passed = every_third.size() == 4 &&
            every_third[0] == 100 && every_third[1] == 103 &&
            every_third[2] == 106 && every_third[3] == 109;
    }
    Kokkos::finalize();
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
