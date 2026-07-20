/*
    Shared analytic Kerr-Schild metric core for Cartesian Kerr-Schild coordinates.

    Kerr-Schild decomposition (eta signature -,+,+,+):
        g_{mu nu} = eta_{mu nu} + 2 H l_mu l_nu
        g^{mu nu} = eta^{mu nu} - 2 H l^mu l^nu
    where l_mu = (1, (r x + a y)/(r^2+a^2), (r y - a x)/(r^2+a^2), z/r) is null
    (l_mu l^mu = 0), H = M r^3 / (r^4 + a^2 z^2), and r(x,y,z) solves the
    oblate-spheroidal relation x^2+y^2+z^2 = r^2 + a^2 z^2/r^2.

    Because l is null under eta, g^{mu nu} is the exact matrix inverse of
    g_{mu nu} (Sherman-Morrison identity), so both are built from the same
    H, l here -- single source of truth for what was previously three
    hand-duplicated copies of g^{mu nu} plus a finite-differenced derivative.
*/
#pragma once
#include "../utils.hpp"

namespace kerr_schild {

// a_BH/M_BH are threaded through explicitly rather than read as globals: they
// are runtime-configurable (see toml_config.hpp) plain `extern real` storage,
// which has no device-side symbol, so referencing them by name from code that
// gets compiled for a GPU backend fails with "identifier undefined in device
// code" -- callers must pass in a value already captured on the host/functor.

KOKKOS_INLINE_FUNCTION
real compute_r(real x, real y, real z, real a_BH) {
    const real rsq = SQR(x) + SQR(y) + SQR(z);
    const real A = rsq - SQR(a_BH);
    const real S = sqrt(SQR(A) + 4.0 * SQR(a_BH * z));
    return sqrt(0.5 * (A + S));
}

// r and dr/dx^k for k=0,1,2 (x,y,z)
KOKKOS_INLINE_FUNCTION
void compute_r_and_deriv(real x, real y, real z, real a_BH, real& r, real dr[3]) {
    const real rsq = SQR(x) + SQR(y) + SQR(z);
    const real A = rsq - SQR(a_BH);
    const real S = sqrt(SQR(A) + 4.0 * SQR(a_BH * z));
    r = sqrt(0.5 * (A + S));
    const real one_plus_A_over_S = 1.0 + A / S;
    dr[0] = x * one_plus_A_over_S / (2.0 * r);
    dr[1] = y * one_plus_A_over_S / (2.0 * r);
    dr[2] = (z * one_plus_A_over_S + 2.0 * SQR(a_BH) * z / S) / (2.0 * r);
}

// l_mu (covariant, l[0]=1) and, if requested, dl[k][mu] = d(l_mu)/dx^k for k=0,1,2 (x,y,z)
KOKKOS_INLINE_FUNCTION
void compute_l(real x, real y, real z, real r, real a_BH, real l[4]) {
    const real B = SQR(r) + SQR(a_BH);
    l[0] = 1.0;
    l[1] = (r * x + a_BH * y) / B;
    l[2] = (r * y - a_BH * x) / B;
    l[3] = z / r;
}

KOKKOS_INLINE_FUNCTION
void compute_l_and_deriv(real x, real y, real z, real r, const real dr[3], real a_BH, real l[4], real dl[3][4]) {
    const real B = SQR(r) + SQR(a_BH);
    l[0] = 1.0;
    l[1] = (r * x + a_BH * y) / B;
    l[2] = (r * y - a_BH * x) / B;
    l[3] = z / r;

    for (int k = 0; k < 3; ++k) {
        const real dx_dk = (k == 0) ? 1.0 : 0.0;
        const real dy_dk = (k == 1) ? 1.0 : 0.0;
        const real dz_dk = (k == 2) ? 1.0 : 0.0;
        const real dB_dk = 2.0 * r * dr[k];

        dl[k][0] = 0.0;
        dl[k][1] = ((dr[k] * x + r * dx_dk) + a_BH * dy_dk - l[1] * dB_dk) / B;
        dl[k][2] = ((dr[k] * y + r * dy_dk) - a_BH * dx_dk - l[2] * dB_dk) / B;
        dl[k][3] = (dz_dk - l[3] * dr[k]) / r;
    }
}

KOKKOS_INLINE_FUNCTION
real compute_H(real r, real z, real a_BH, real M_BH) {
    const real D = QUAD(r) + SQR(a_BH * z);
    return M_BH * CUBE(r) / D;
}

KOKKOS_INLINE_FUNCTION
real compute_H_and_deriv(real r, real z, const real dr[3], real a_BH, real M_BH, real dH[3]) {
    const real D = QUAD(r) + SQR(a_BH * z);
    const real H = M_BH * CUBE(r) / D;
    for (int k = 0; k < 3; ++k) {
        const real dz_dk = (k == 2) ? 1.0 : 0.0;
        const real dD_dk = 4.0 * CUBE(r) * dr[k] + 2.0 * SQR(a_BH) * z * dz_dk;
        dH[k] = M_BH * (3.0 * SQR(r) * dr[k] * D - CUBE(r) * dD_dk) / SQR(D);
    }
    return H;
}

// Covariant metric g_{mu nu} at position X = (t, x, y, z)
KOKKOS_INLINE_FUNCTION
void compute_metric(const real X[4], real a_BH, real M_BH, real g[4][4]) {
    const real r = compute_r(X[1], X[2], X[3], a_BH);
    real ldown[4];
    compute_l(X[1], X[2], X[3], r, a_BH, ldown);
    const real H = compute_H(r, X[3], a_BH, M_BH);
    const real eta[4] = {-1.0, 1.0, 1.0, 1.0}; // diagonal entries of eta_{mu nu}
    for (int mu = 0; mu < 4; ++mu) {
        for (int nu = 0; nu < 4; ++nu) {
            g[mu][nu] = 2.0 * H * ldown[mu] * ldown[nu] + ((mu == nu) ? eta[mu] : 0.0);
        }
    }
}

// Contravariant metric g^{mu nu} at position X = (t, x, y, z)
KOKKOS_INLINE_FUNCTION
void compute_inverse_metric(const real X[4], real a_BH, real M_BH, real ginv[4][4]) {
    const real r = compute_r(X[1], X[2], X[3], a_BH);
    real ldown[4];
    compute_l(X[1], X[2], X[3], r, a_BH, ldown);
    const real lup[4] = {-ldown[0], ldown[1], ldown[2], ldown[3]};
    const real H = compute_H(r, X[3], a_BH, M_BH);
    const real eta_inv[4] = {-1.0, 1.0, 1.0, 1.0}; // diagonal entries of eta^{mu nu}
    for (int mu = 0; mu < 4; ++mu) {
        for (int nu = 0; nu < 4; ++nu) {
            ginv[mu][nu] = -2.0 * H * lup[mu] * lup[nu] + ((mu == nu) ? eta_inv[mu] : 0.0);
        }
    }
}

// Analytic derivative of the contravariant metric: dginv[alpha][mu][nu] = d(g^{mu nu}) / dX^alpha
// The metric is stationary, so dginv[0][*][*] = 0 identically.
KOKKOS_INLINE_FUNCTION
void compute_inverse_metric_deriv(const real X[4], real a_BH, real M_BH, real dginv[4][4][4]) {
    real r, dr[3];
    compute_r_and_deriv(X[1], X[2], X[3], a_BH, r, dr);
    real ldown[4], dldown[3][4];
    compute_l_and_deriv(X[1], X[2], X[3], r, dr, a_BH, ldown, dldown);
    real dH[3];
    const real H = compute_H_and_deriv(r, X[3], dr, a_BH, M_BH, dH);

    const real lup[4] = {-ldown[0], ldown[1], ldown[2], ldown[3]};

    for (int mu = 0; mu < 4; ++mu) {
        for (int nu = 0; nu < 4; ++nu) {
            dginv[0][mu][nu] = 0.0;
        }
    }
    for (int k = 0; k < 3; ++k) {
        const real dlup_k[4] = {-dldown[k][0], dldown[k][1], dldown[k][2], dldown[k][3]};
        for (int mu = 0; mu < 4; ++mu) {
            for (int nu = 0; nu < 4; ++nu) {
                dginv[k + 1][mu][nu] =
                    -2.0 * (dH[k] * lup[mu] * lup[nu] + H * (dlup_k[mu] * lup[nu] + lup[mu] * dlup_k[nu]));
            }
        }
    }
}

// Given a spatial contravariant direction K^i at position X, solves for K^0 such
// that K^mu = (K^0, K^i) is null (g_{mu nu} K^mu K^nu = 0) -- picking the
// future-directed (K^0 > 0) root -- then lowers the index to return the
// covariant photon momentum p_mu = g_{mu nu} K^nu. Returns false if no
// future-directed real root exists (should not happen for physically sensible
// spatial directions outside the horizon).
KOKKOS_INLINE_FUNCTION
bool null_covariant_momentum_from_spatial_direction(const real X[4], const real K_spatial[3], real a_BH, real M_BH, real p_cov[4]) {
    real g[4][4];
    compute_metric(X, a_BH, M_BH, g);
    const real a = g[0][0];
    const real b = 2.0 * (g[0][1] * K_spatial[0] + g[0][2] * K_spatial[1] + g[0][3] * K_spatial[2]);
    real c = 0.0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            c += g[i + 1][j + 1] * K_spatial[i] * K_spatial[j];
        }
    }
    const real disc = SQR(b) - 4.0 * a * c;
    if (disc < 0.0) return false;
    const real sq = sqrt(disc);
    const real K0_1 = (-b + sq) / (2.0 * a);
    const real K0_2 = (-b - sq) / (2.0 * a);
    const real K0 = (K0_1 > 0.0) ? K0_1 : K0_2;
    if (!(K0 > 0.0)) return false;

    const real K_contra[4] = {K0, K_spatial[0], K_spatial[1], K_spatial[2]};
    for (int mu = 0; mu < 4; ++mu) {
        real sum = 0.0;
        for (int nu = 0; nu < 4; ++nu) sum += g[mu][nu] * K_contra[nu];
        p_cov[mu] = sum;
    }
    return true;
}

} // namespace kerr_schild
