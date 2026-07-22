#!/usr/bin/env python3
"""Generate a Fishbone-Moncrief torus grid/MHD input bundle for local runs
and tests.

Fishbone & Moncrief (1976) describes an analytic, hydrostatic-equilibrium
torus of constant specific angular momentum orbiting a Kerr black hole. It is
the standard initial condition used by GRMHD codes (HARM, iharm3d, BHAC,
KORAL, ...) and is a physically reasonable stand-in for an accretion disk
when generating example data for this ray tracer.

All of the torus algebra below is carried out in "geometric" units where the
black hole mass is 1 (G = c = M = 1), using the dimensionless spin
`a_hat = spin / mass`. This is safe for any `mass` because the vacuum Kerr
solution is scale invariant under (r, a, M) -> (r / M, a / M, 1): every
length divides out to M=1, is evaluated with the standard textbook formulas,
and the two velocity components that do not divide out cleanly (u^t, u^phi)
are converted back explicitly (see the comments below the metric block).
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams.update({
    "font.family":      "serif",
    "font.size":         11,
    "axes.labelsize":    13,
    "axes.titlesize":    13,
    "legend.fontsize":   10,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "xtick.top":         True,
    "ytick.right":       True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "axes.linewidth":    1.0,
    "lines.linewidth":   1.8,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "mathtext.fontset":  "cm",
    # set ytick size minor and major
    "ytick.major.size": 10,
    "ytick.minor.size": 3,
    "xtick.major.size": 10,
    "xtick.minor.size": 3,
})


def _keplerian_specific_angular_momentum(r_hat, spin_hat):
    """Specific angular momentum of a prograde equatorial circular geodesic.

    Bardeen, Press & Teukolsky (1972), eq. 2.12-2.13, evaluated at mass=1:
    l(r) = L/E = (r^2 - 2*a*sqrt(r) + a^2) / (r^1.5 - 2*sqrt(r) + a).
    The Fishbone-Moncrief torus fixes the (constant) torus specific angular
    momentum to this value evaluated at the pressure-maximum radius.
    """
    return (r_hat**2 - 2.0 * spin_hat * np.sqrt(r_hat) + spin_hat**2) / (
        r_hat**1.5 - 2.0 * np.sqrt(r_hat) + spin_hat
    )


def _boyer_lindquist_metric(r_hat, theta, spin_hat):
    """g_tt, g_tphi, g_phiphi for the Kerr metric at mass=1 (standard BL form)."""
    sin2 = np.sin(theta) ** 2
    sigma = r_hat**2 + spin_hat**2 * np.cos(theta) ** 2
    delta = r_hat**2 - 2.0 * r_hat + spin_hat**2
    a_term = (r_hat**2 + spin_hat**2) ** 2 - delta * spin_hat**2 * sin2

    g_tt = -(1.0 - 2.0 * r_hat / sigma)
    g_tphi = -2.0 * r_hat * spin_hat * sin2 / sigma
    g_phiphi = a_term * sin2 / sigma
    return g_tt, g_tphi, g_phiphi


def _rigidly_rotating_four_velocity(r_hat, theta, spin_hat, angular_momentum):
    """u^t, u^phi, and (covariant) u_t (mass=1 units), for u^r = u^theta = 0
    and specific angular momentum -u_phi/u_t fixed to `angular_momentum`.

    Solving u_phi = -l*u_t together with u_phi = g_tphi*u^t + g_phiphi*u^phi
    and u_t = g_tt*u^t + g_tphi*u^phi gives Omega = u^phi/u^t; the timelike
    normalization g_munu u^mu u^nu = -1 then fixes u^t (contravariant, > 0).
    The conserved specific energy -u_t (covariant) is what the Fishbone-
    Moncrief enthalpy below actually needs -- it is not the same quantity as
    u^t once the metric is non-diagonal. Returns NaN where no physical (real,
    future-directed) solution exists, which is used below to identify the
    torus boundary and to fall back to an atmosphere.
    """
    g_tt, g_tphi, g_phiphi = _boyer_lindquist_metric(r_hat, theta, spin_hat)
    omega = -(g_tphi + angular_momentum * g_tt) / (g_phiphi + angular_momentum * g_tphi)
    normalization = -(g_tt + 2.0 * omega * g_tphi + omega**2 * g_phiphi)
    with np.errstate(invalid="ignore", divide="ignore"):
        u_upper_t = 1.0 / np.sqrt(normalization)
    u_lower_t = u_upper_t * (g_tt + omega * g_tphi)
    return u_upper_t, omega * u_upper_t, u_lower_t


def create_fishbone_moncrief_torus(
    output_dir=".",
    nr=64,
    ntheta=64,
    nphi=32,
    r_min=None,
    r_max=50.0,
    mass=1.0,
    spin=0.9375,
    r_in=6.0,
    r_pressure_max=10.0,
    gamma=4.0 / 3.0,
    rho_floor=1.0e-6,
    plot=False
):
    """Create the (nr, ntheta, nphi) coordinate and fluid arrays for a
    Fishbone-Moncrief torus, in the format the C++ loader expects.

    Parameters
    ----------
    output_dir : directory that receives r.npy, theta.npy, phi.npy, rho.npy,
        Tgas.npy, vel.npy, and mag.npy.
    nr, ntheta, nphi : grid resolution. Must each be >= 2.
    r_min, r_max : radial extent of the grid, in the same code units as
        `mass`/`spin` (i.e. matching [black_hole] in the simulation TOML).
        `r_min` defaults to just outside the horizon.
    mass, spin : black hole mass and spin, matching [black_hole].mass and
        [black_hole].spin in the simulation TOML that will consume this data.
    r_in : torus inner edge radius.
    r_pressure_max : radius of the torus pressure/density maximum; together
        with `spin` this sets the (constant) specific angular momentum of
        the torus via the Keplerian value at that radius. Not every
        (r_in, r_pressure_max, spin) combination produces a torus with a
        closed outer edge -- if `r_pressure_max` is too large relative to
        `r_in` for the given spin, the equipotential surface never closes
        and the "torus" fills the whole grid out to `r_max`. A warning is
        printed if that appears to have happened; shrink r_pressure_max
        or grow r_in and re-run if a compact torus is required.
    gamma : adiabatic index of the ideal-gas equation of state used to turn
        the torus enthalpy into density and temperature.
    rho_floor : dimensionless density (and temperature) assigned outside the
        torus, representing a tenuous, non-rotating atmosphere.
    """
    if min(nr, ntheta, nphi) < 2:
        raise ValueError("nr, ntheta, and nphi must each be at least 2")
    if mass <= 0.0:
        raise ValueError("mass must be positive")
    if abs(spin) > mass:
        raise ValueError("require |spin| <= mass")
    if gamma <= 1.0:
        raise ValueError("gamma must be greater than 1")
    if not (r_pressure_max > r_in > 0.0):
        raise ValueError("require 0 < r_in < r_pressure_max")

    horizon = mass + np.sqrt(mass**2 - spin**2)
    if r_min is None:
        r_min = 1.2 * horizon
    if r_min <= horizon or r_max <= r_min:
        raise ValueError(
            f"require horizon ({horizon:.6g}) < r_min < r_max")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    r = np.logspace(np.log10(r_min), np.log10(r_max), nr)
    theta = np.linspace(1e-3, np.pi - 1e-3, ntheta)
    phi = np.linspace(0.0, 2.0 * np.pi, nphi, endpoint=False)
    radius, polar_angle, azimuth = np.meshgrid(r, theta, phi, indexing="ij")

    # All torus algebra below is in mass=1 units; see the module docstring.
    spin_hat = spin / mass
    r_hat = radius / mass
    r_in_hat = r_in / mass
    r_pressure_max_hat = r_pressure_max / mass

    angular_momentum = _keplerian_specific_angular_momentum(r_pressure_max_hat, spin_hat)
    u_upper_t, u_upper_phi, u_lower_t = _rigidly_rotating_four_velocity(
        r_hat, polar_angle, spin_hat, angular_momentum)
    _, _, u_lower_t_in = _rigidly_rotating_four_velocity(
        np.array(r_in_hat), np.array(np.pi / 2.0), spin_hat, angular_momentum)

    # Fishbone-Moncrief 1976: for constant specific angular momentum, the log
    # specific enthalpy satisfies ln(h) = ln(u_t,in) - ln(u_t(r, theta)) using
    # the covariant (conserved-energy) u_t, i.e. h * u_t is constant through
    # the torus (u_t < 0, so this ratio of two negative numbers is positive).
    # Cells where h < 1 or r < r_in fall outside the torus and become
    # atmosphere below.
    with np.errstate(invalid="ignore", divide="ignore"):
        specific_enthalpy = u_lower_t_in / u_lower_t
    in_torus = (
        (r_hat >= r_in_hat) & np.isfinite(specific_enthalpy) & (specific_enthalpy >= 1.0))
    if in_torus[-1].any():
        print(
            "Warning: the torus equipotential does not close before r_max "
            f"({r_max:g}); the disk fills the grid out to its outer boundary. "
            "Shrink r_pressure_max or grow r_in for a compact torus.")

    # Ideal-gas polytrope with the polytropic constant fixed to 1; the
    # resulting density is renormalized below so its peak value is 1.
    density = np.where(
        in_torus,
        np.power(
            np.clip((gamma - 1.0) / gamma * (specific_enthalpy - 1.0), 0.0, None),
            1.0 / (gamma - 1.0)),
        0.0)
    peak_density = density.max()
    if peak_density > 0.0:
        density /= peak_density
    # temperature = (gamma - 1) * internal_energy / density = density^(gamma-1)
    # for this polytrope (polytropic constant 1), which is why it only needs
    # `density` and not a separately tracked internal energy.
    temperature = np.where(in_torus, np.power(np.maximum(density, 1e-300), gamma - 1.0), 0.0)
    density = np.where(in_torus, density, rho_floor)
    temperature = np.where(in_torus, temperature, rho_floor)

    # Outside the torus, fall back to a zero-angular-momentum observer
    # (ZAMO, u_phi = 0): unlike a static observer, this stays a valid unit
    # timelike four-velocity everywhere outside the horizon, including
    # inside the ergosphere where frame dragging forbids g_tt < 0 observers.
    g_tt_zamo, g_tphi_zamo, g_phiphi_zamo = _boyer_lindquist_metric(
        r_hat, polar_angle, spin_hat)
    omega_zamo = -g_tphi_zamo / g_phiphi_zamo
    lapse_sq = g_tphi_zamo**2 / g_phiphi_zamo - g_tt_zamo
    zamo_u_t = 1.0 / np.sqrt(np.clip(lapse_sq, 1e-300, None))
    u_t_final = np.where(in_torus, u_upper_t, zamo_u_t)
    u_phi_final_hat = np.where(in_torus, u_upper_phi, omega_zamo * zamo_u_t)

    # u^t is dimensionless and unaffected by the mass=1 rescaling (t and r
    # both scale by `mass`, so d t_hat/d tau_hat = dt/dtau). u^phi has units
    # of 1/length, so converting back from mass=1 units divides by `mass`.
    # u^r = u^theta = 0 throughout: the torus is purely toroidal, and this
    # coordinate basis is shared with Boyer-Lindquist wherever the radial and
    # polar four-velocity components vanish (see fluid_frame.hpp).
    velocity = np.zeros((nr, ntheta, nphi, 4), dtype=np.float64)
    velocity[..., 0] = u_t_final
    velocity[..., 3] = u_phi_final_hat / mass
    magnetic = np.zeros_like(velocity)

    arrays = {
        "r.npy": radius,
        "theta.npy": polar_angle,
        "phi.npy": azimuth,
        "rho.npy": density,
        "Tgas.npy": temperature,
        "vel.npy": velocity,
        "mag.npy": magnetic,
    }
    for name, array in arrays.items():
        path = output / name
        np.save(path, array.astype(np.float64, copy=False))
        print(f"Created {path} with shape {array.shape}")
    if plot:
        x_2d = radius[:, :, 0] * np.sin(polar_angle[:, :, 0]) * np.cos(azimuth[:, :, 0])
        z_2d = radius[:, :, 0] * np.cos(polar_angle[:, :, 0])
        density_2d = density[:, :, 0]
        temperature_2d = temperature[:, :, 0]
        u_phi_2d = u_upper_phi[:, :, 0] / mass
        # Plot the torus density, velocity and temperature in r-theta plane at phi=0
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        titles = [r"$\rho$", r"$u^\phi$", r"$T$"]
        xlabels = [r"$x$", r"$x$", r"$x$"]
        ylabels = [r"$z$", r"$z$", r"$z$"]
        cmaps = ["viridis", "plasma", "inferno"]
        norms = [mpl.colors.LogNorm(vmin=density_2d.min(), vmax=density_2d.max()),
                 mpl.colors.Normalize(vmin=u_phi_2d.min(), vmax=u_phi_2d.max()),
                 mpl.colors.LogNorm(vmin=temperature_2d.min(), vmax=temperature_2d.max())]
        data = [density_2d, u_phi_2d, temperature_2d]
        for ax, title, xlabel, ylabel, cmap, norm, d in zip(axes, titles, xlabels, ylabels, cmaps, norms, data):
            im = ax.pcolormesh(x_2d, z_2d, d, shading="auto", cmap=cmap, norm=norm)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            fig.colorbar(im, ax=ax)
            ax.set_facecolor("#B3B2B2")
            if ax != axes[0]:
                ax.set_ylabel("")
                ax.set_yticklabels([])
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.1)
        plt.show()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--nr", type=int, default=64)
    parser.add_argument("--ntheta", type=int, default=64)
    parser.add_argument("--nphi", type=int, default=32)
    parser.add_argument("--r-min", type=float, default=None,
                        help="defaults to just outside the horizon")
    parser.add_argument("--r-max", type=float, default=50.0)
    parser.add_argument("--mass", type=float, default=1.0,
                        help="must match [black_hole].mass in the simulation TOML")
    parser.add_argument("--spin", type=float, default=0.9375,
                        help="must match [black_hole].spin in the simulation TOML")
    parser.add_argument("--r-in", type=float, default=6.0,
                        help="torus inner edge radius")
    parser.add_argument("--r-pressure-max", type=float, default=10.0,
                        help="radius of the torus density/pressure maximum")
    parser.add_argument("--gamma", type=float, default=4.0 / 3.0,
                        help="adiabatic index")
    parser.add_argument("--rho-floor", type=float, default=1.0e-6,
                        help="dimensionless atmosphere density/temperature")
    parser.add_argument("--plot", type=bool, default=True,
                        help="plot the generated torus")
    args = parser.parse_args()
    create_fishbone_moncrief_torus(
        args.output_dir, args.nr, args.ntheta, args.nphi,
        args.r_min, args.r_max, args.mass, args.spin,
        args.r_in, args.r_pressure_max, args.gamma, args.rho_floor, args.plot)


if __name__ == "__main__":
    main()
