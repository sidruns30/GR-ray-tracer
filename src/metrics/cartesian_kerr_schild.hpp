/*
    Code to do manipulations in Cartesian Kerr-Schild coordinates
    such as raising / lowering indices and computing norms
*/
#pragma once
#include "../utils.hpp"
#include "../utils.cpp"

// Siddhant: Input is a *covariant* 4-vector V_mu at position X^mu
KOKKOS_INLINE_FUNCTION
real cartesian_kerr_schild_norm(const real X[4], const real V[4]) {
    // Compute the Cartesian Kerr-Schild metric at position X
    real g_con[4][4];
    real x = X[1];
    real y = X[2];
    real z = X[3];
    real RSQ = SQR(x) + SQR(y) + SQR(z);
    real r = sqrt(0.5 * (RSQ - SQR(a_BH) + sqrt(SQR(RSQ - SQR(a_BH)) + 4.0 * SQR(a_BH * z))));
    for (int a=0;a<4;++a) for (int b=0;b<4;++b) g_con[a][b] = 0.0;
    g_con[0][0] = (-2*M_BH*CUBE(r) - SQR(a_BH * z) - QUAD(r))/(SQR(a_BH * z)+ QUAD(r));
    g_con[0][1] = 2*M_BH*CUBE(r)*(a_BH * y + r * x)/((SQR(a_BH) + SQR(r))*(SQR(a_BH * z) + QUAD(r)));
    g_con[0][2] = -2*M_BH*CUBE(r)*(a_BH * x - r * y)/((SQR(a_BH) + SQR(r))*(SQR(a_BH * z) + QUAD(r)));
    g_con[0][3] = 2*M_BH*SQR(r)*z/(SQR(a_BH * z) + QUAD(r));
    g_con[1][0] = 2*M_BH*CUBE(r)*(a_BH * y + r * x)/((SQR(a_BH) + SQR(r))*(SQR(a_BH * z) + QUAD(r)));
    g_con[1][1] = (-2*M_BH*CUBE(r)*SQR(a_BH * y + r * x) + SQR(SQR(a_BH) + SQR(r))*(SQR(a_BH * z) + QUAD(r)))/ (SQR(SQR(a_BH) + SQR(r))*(SQR(a_BH * z) + QUAD(r)));
    g_con[1][2] = 2*M_BH*CUBE(r)*(a_BH * x - r * y)*(a_BH * y + r * x)/(SQR(SQR(a_BH) + SQR(r))*(SQR(a_BH * z) + QUAD(r)));
    g_con[1][3] = -2*M_BH*SQR(r)*z*(a_BH * y + r * x)/((SQR(a_BH) + SQR(r))*(SQR(a_BH * z) + QUAD(r)));
    g_con[2][0] = -2*M_BH*CUBE(r)*(a_BH * x - r * y)/((SQR(a_BH) + SQR(r))*(SQR(a_BH * z) + QUAD(r)));
    g_con[2][1] = 2*M_BH*CUBE(r)*(a_BH * x - r * y)*(a_BH * y + r * x)/(SQR(SQR(a_BH) + SQR(r))*(SQR(a_BH * z) + QUAD(r)));
    g_con[2][2] = (-2*M_BH*CUBE(r)*SQR(a_BH * x - r * y) + SQR(SQR(a_BH) + SQR(r))*(SQR(a_BH * z) + QUAD(r)))/(SQR(SQR(a_BH) + SQR(r))*(SQR(a_BH * z) + QUAD(r)));
    g_con[2][3] = 2*M_BH*SQR(r)*z*(a_BH * x - r * y)/((SQR(a_BH) + SQR(r))*(SQR(a_BH * z) + QUAD(r)));
    g_con[3][0] = 2*M_BH*SQR(r)*z/(SQR(a_BH * z) + QUAD(r));
    g_con[3][1] = -2*M_BH*SQR(r)*z*(a_BH * y + r * x)/((SQR(a_BH) + SQR(r))*(SQR(a_BH * z) + QUAD(r)));
    g_con[3][2] = 2*M_BH*SQR(r)*z*(a_BH * x - r * y)/((SQR(a_BH) + SQR(r))*(SQR(a_BH * z) + QUAD(r)));
    g_con[3][3] = (-2*M_BH*r*SQR(z) + SQR(a_BH * z) + QUAD(r))/(SQR(a_BH * z) + QUAD(r));
    real norm = 0.0;
    for (int mu = 0; mu < 4; ++mu) {
        for (int nu = 0; nu < 4; ++nu) {
            norm += g_con[mu][nu] * V[mu] * V[nu];
        }
    }
    return norm;
}

// Kernel to compute the norm of a 4-vector in Cartesian Kerr-Schild coordinates
void compute_cartesian_kerr_schild_norms(
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
