#pragma once
#include <filesystem>
#include <sstream>

#include "../utils.hpp"

// Write into photon output file from all MPI processes in order
inline void write_photon_output(const Photons& photons, const size_t step, const int rank)
{
    (void)rank;
    std::filesystem::create_directories(output_directory);
    const size_t Nphotons = photons.x0_host.extent(0);
    const std::string prefix = output_directory + "photon_output_" + std::to_string(step) + "_rank" + std::to_string(rank);
    cnpy::npy_save(prefix + "_x0.npy", photons.x0_host.data(), {Nphotons});
    cnpy::npy_save(prefix + "_x1.npy", photons.x1_host.data(), {Nphotons});
    cnpy::npy_save(prefix + "_x2.npy", photons.x2_host.data(), {Nphotons});
    cnpy::npy_save(prefix + "_x3.npy", photons.x3_host.data(), {Nphotons});
    cnpy::npy_save(prefix + "_k0.npy", photons.k0_host.data(), {Nphotons});
    cnpy::npy_save(prefix + "_k1.npy", photons.k1_host.data(), {Nphotons});
    cnpy::npy_save(prefix + "_k2.npy", photons.k2_host.data(), {Nphotons});
    cnpy::npy_save(prefix + "_k3.npy", photons.k3_host.data(), {Nphotons});
    cnpy::npy_save(prefix + "_I.npy", photons.I_host.data(), {Nphotons});
    cnpy::npy_save(prefix + "_Q.npy", photons.Q_host.data(), {Nphotons});
    cnpy::npy_save(prefix + "_U.npy", photons.U_host.data(), {Nphotons});
    cnpy::npy_save(prefix + "_V.npy", photons.V_host.data(), {Nphotons});
    return;
}
