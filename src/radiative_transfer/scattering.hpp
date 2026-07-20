#pragma once

#include <cstdint>

#include "../utils.hpp"

struct ScatteringModel {
    bool enabled = false;
    real optical_depth = 0.0;
    real albedo = 1.0;
    std::uint64_t seed = 0;
};

KOKKOS_INLINE_FUNCTION
std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

KOKKOS_INLINE_FUNCTION
real uniform01(std::uint64_t seed) {
    const std::uint64_t bits = splitmix64(seed);
    return static_cast<real>((bits >> 11) * (1.0 / 9007199254740992.0));
}

KOKKOS_INLINE_FUNCTION
void isotropic_scatter_direction(real kx, real ky, real kz, std::uint64_t seed,
                                        real& out_kx, real& out_ky, real& out_kz) {
    const real mu = 2.0 * uniform01(seed) - 1.0;
    const real phi = 2.0 * PI * uniform01(seed + 1);
    const real sin_theta = sqrt(std::max(real(0.0), 1.0 - mu * mu));
    const real nx = sin_theta * cos(phi);
    const real ny = sin_theta * sin(phi);
    const real nz = mu;

    const real old_norm = sqrt(kx * kx + ky * ky + kz * kz);
    if (old_norm == 0.0) {
        out_kx = nx;
        out_ky = ny;
        out_kz = nz;
        return;
    }

    const real ux = kx / old_norm;
    const real uy = ky / old_norm;
    const real uz = kz / old_norm;

    const real ax = (std::abs(ux) < 0.9) ? 1.0 : 0.0;
    const real ay = (std::abs(ux) < 0.9) ? 0.0 : 1.0;
    const real az = 0.0;
    real ex = uy * az - uz * ay;
    real ey = uz * ax - ux * az;
    real ez = ux * ay - uy * ax;
    const real e_norm = sqrt(ex * ex + ey * ey + ez * ez);
    ex /= e_norm;
    ey /= e_norm;
    ez /= e_norm;

    const real fx = uy * ez - uz * ey;
    const real fy = uz * ex - ux * ez;
    const real fz = ux * ey - uy * ex;

    out_kx = nx * ex + ny * fx + nz * ux;
    out_ky = nx * ey + ny * fy + nz * uy;
    out_kz = nx * ez + ny * fz + nz * uz;
}

KOKKOS_INLINE_FUNCTION
bool maybe_scatter_photon(real state[8], real stokes[4], const ScatteringModel& model,
                                 std::size_t photon_index, std::size_t step_index) {
    if (!model.enabled || model.optical_depth <= 0.0) {
        return false;
    }

    const real scatter_prob = 1.0 - std::exp(-model.optical_depth);
    const real draw = uniform01(model.seed + 1315423911ULL * photon_index + 2654435761ULL * step_index);
    if (draw >= scatter_prob) {
        return false;
    }

    real kx = state[IKX];
    real ky = state[IKY];
    real kz = state[IKZ];
    real skx = 0.0, sky = 0.0, skz = 0.0;
    isotropic_scatter_direction(kx, ky, kz, model.seed ^ (photon_index + 1), skx, sky, skz);
    state[IKX] = skx;
    state[IKY] = sky;
    state[IKZ] = skz;
    state[IKT] = std::abs(state[IKT]);

    stokes[0] *= model.albedo;
    stokes[1] *= 0.5 * model.albedo;
    stokes[2] *= 0.5 * model.albedo;
    stokes[3] *= 0.25 * model.albedo;
    return true;
}
