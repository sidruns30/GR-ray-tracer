/*
    Code to do manipulations in Cartesian Kerr-Schild coordinates
    such as raising / lowering indices and computing norms
*/
#pragma once
#include "../utils.hpp"
#include "kerr_schild_core.hpp"

// Compute the norm of the contravariant vector V at position X in Cartesian Kerr-Schild coordinates
KOKKOS_INLINE_FUNCTION
real cartesian_kerr_schild_norm(const real X[4], const real V[4]) {
    real g_con[4][4];
    kerr_schild::compute_inverse_metric(X, g_con);
    real norm = 0.0;
    for (int mu = 0; mu < 4; ++mu) {
        for (int nu = 0; nu < 4; ++nu) {
            norm += g_con[mu][nu] * V[mu] * V[nu];
        }
    }
    return norm;
}

// Kernel to compute the norm of a 4-vector in Cartesian Kerr-Schild coordinates
inline void compute_cartesian_kerr_schild_norms(
    const Kokkos::View<real*>& x0,
    const Kokkos::View<real*>& x1,
    const Kokkos::View<real*>& x2,
    const Kokkos::View<real*>& x3,
    const Kokkos::View<real*>& k0,
    const Kokkos::View<real*>& k1,
    const Kokkos::View<real*>& k2,
    const Kokkos::View<real*>& k3,
    Kokkos::View<real*>& norms
) {
    const size_t N = x1.extent(0);
    Kokkos::parallel_for(
        "Compute Cartesian Kerr-Schild Norms",
        Kokkos::RangePolicy<>(0, N),
        KOKKOS_LAMBDA(const int i) {
            real X[4] = {
                x0(i),
                x1(i),
                x2(i),
                x3(i)
            };
            real V[4] = {
                k0(i),
                k1(i),
                k2(i),
                k3(i)
            };
            norms(i) = cartesian_kerr_schild_norm(X, V);
        }
    );
    return;
}
