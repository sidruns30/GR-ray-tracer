#pragma once
#include "../utils.hpp"
#include "../utils.cpp"

// Write into photon output file from all MPI processes in order
void write_photon_output(const Photons& photons, const size_t step, const int rank)
{
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    size_t Nphotons = photons.x0_host.extent(0);
    std::ostringstream fname;
    fname << output_directory + "photon_output_" << step << "_rank" << rank << ".npz";
    cnpy::npz_save(fname.str().c_str(), "x0", photons.x0_host.data(), {Nphotons}, "w");
    cnpy::npz_save(fname.str().c_str(), "x1", photons.x1_host.data(), {Nphotons}, "a");
    cnpy::npz_save(fname.str().c_str(), "x2", photons.x2_host.data(), {Nphotons}, "a");
    cnpy::npz_save(fname.str().c_str(), "x3", photons.x3_host.data(), {Nphotons}, "a");
    cnpy::npz_save(fname.str().c_str(), "k0", photons.k0_host.data(), {Nphotons}, "a");
    cnpy::npz_save(fname.str().c_str(), "k1", photons.k1_host.data(), {Nphotons}, "a");
    cnpy::npz_save(fname.str().c_str(), "k2", photons.k2_host.data(), {Nphotons}, "a");
    cnpy::npz_save(fname.str().c_str(), "k3", photons.k3_host.data(), {Nphotons}, "a");
    cnpy::npz_save(fname.str().c_str(), "I", photons.I_host.data(), {Nphotons}, "a");
    return;
}
