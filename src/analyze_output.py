#!/usr/bin/env python3
"""
Analyze GR-ray-tracer photon output: post-hoc plots, plus a live terminal
monitor (photon stats + an in-terminal x-y density histogram) that can run
alongside the C++ binary while it is still writing output.

File layout produced by src/output/write_output.hpp and
src/radiative_transfer/observation.hpp, under --output-dir (default matches
utils.hpp's `output_directory = "./output/"`):

    photon_output_<step>_rank<rank>_<field>.npy
        field in x0,x1,x2,x3 (t,x,y,z), k0,k1,k2,k3 (p_t,p_x,p_y,p_z covariant
        momentum), I,Q,U,V (Stokes parameters)
    observation_step_<step>_rank<rank>_<field>.npy
        field in image_I,image_Q,image_U,image_V, lightcurve_I, spectrum_I

Photon identity is (rank, local index) and is stable across steps -- the
Photons Kokkos::View is fixed-size and terminated photons simply stop being
updated rather than removed -- so per-photon quantities are tracked over time
by concatenating ranks in sorted order at each step.

Caveat carried over from the C++ side: build_observation_products() (see
src/radiative_transfer/observation.hpp) bins each rank's photons into image /
lightcurve / spectrum axes scaled to *that rank's own* local min/max of
x,y,t,k0. Different ranks therefore generally have different physical axis
ranges for the same pixel/bin index, so per-rank observation products are NOT
directly summable pixel-by-pixel -- this script plots them per-rank rather
than pretending otherwise.

Usage:
    python3 analyze_output.py post   --output-dir ./output/
    python3 analyze_output.py watch  --output-dir ./output/
"""
import argparse
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

# Kerr-Schild parameters -- must match src/utils.hpp (M_BH, a_BH). Override
# with --mass/--spin if utils.hpp has since changed.
DEFAULT_M_BH = 1.0
DEFAULT_A_BH = 1.0

PHOTON_FIELDS = ["x0", "x1", "x2", "x3", "k0", "k1", "k2", "k3", "I", "Q", "U", "V"]
PHOTON_RE = re.compile(r"photon_output_(\d+)_rank(\d+)_(\w+)\.npy$")
OBS_FIELDS = ["image_I", "image_Q", "image_U", "image_V", "lightcurve_I", "spectrum_I"]
OBS_RE = re.compile(r"observation_step_(\d+)_rank(\d+)_(\w+)\.npy$")


# ---------------------------------------------------------------------------
# Discovery / loading
# ---------------------------------------------------------------------------
def discover_photon_steps(output_dir: Path):
    """{step: {rank: set(fields present)}} for photon_output_* files in output_dir."""
    steps = {}
    for f in output_dir.glob("photon_output_*.npy"):
        m = PHOTON_RE.match(f.name)
        if not m:
            continue
        step, rank, field = int(m.group(1)), int(m.group(2)), m.group(3)
        if field not in PHOTON_FIELDS:
            continue
        steps.setdefault(step, {}).setdefault(rank, set()).add(field)
    return steps


def discover_observation_steps(output_dir: Path):
    """{step: {rank: set(fields present)}} for observation_step_* files in output_dir."""
    steps = {}
    for f in output_dir.glob("observation_step_*.npy"):
        m = OBS_RE.match(f.name)
        if not m:
            continue
        step, rank, field = int(m.group(1)), int(m.group(2)), m.group(3)
        if field not in OBS_FIELDS:
            continue
        steps.setdefault(step, {}).setdefault(rank, set()).add(field)
    return steps


def load_photon_step(output_dir: Path, step: int, ranks):
    """Concatenate all photon fields for `step` across sorted `ranks`."""
    data = {f: [] for f in PHOTON_FIELDS}
    for rank in sorted(ranks):
        for f in PHOTON_FIELDS:
            data[f].append(np.load(output_dir / f"photon_output_{step}_rank{rank}_{f}.npy"))
    return {f: np.concatenate(v) for f, v in data.items()}


def load_observation_step(output_dir: Path, step: int, ranks):
    """{rank: {field: array}} for observation products at `step` (kept per-rank, see module docstring)."""
    out = {}
    for rank in sorted(ranks):
        out[rank] = {
            f: np.load(output_dir / f"observation_step_{step}_rank{rank}_{f}.npy") for f in OBS_FIELDS
        }
    return out


# ---------------------------------------------------------------------------
# Physics: vectorized port of kerr_schild_core.hpp / geodesic_observables.hpp
# ---------------------------------------------------------------------------
def compute_r(x, y, z, a):
    rsq = x**2 + y**2 + z**2
    A = rsq - a**2
    S = np.sqrt(A**2 + 4.0 * a**2 * z**2)
    return np.sqrt(0.5 * (A + S))


def compute_observables(state, M, a):
    """norm, energy, Lz, carter -- see src/analysis/geodesic_observables.hpp."""
    x, y, z = state["x1"], state["x2"], state["x3"]
    pt, px, py, pz = state["k0"], state["k1"], state["k2"], state["k3"]

    r = compute_r(x, y, z, a)
    sin_theta = np.sqrt(np.clip(1.0 - (z / r) ** 2, 0.0, None))
    cos_theta = z / r
    sin_theta_safe = np.maximum(1e-15, sin_theta)

    B = r**2 + a**2
    H = M * r**3 / (r**4 + a**2 * z**2)
    l1, l2, l3 = (r * x + a * y) / B, (r * y - a * x) / B, z / r
    lup_dot_p = -pt + l1 * px + l2 * py + l3 * pz  # lup = (-1, l1, l2, l3)

    norm = -(pt**2) + px**2 + py**2 + pz**2 - 2.0 * H * lup_dot_p**2
    energy = -pt
    Lz = x * py - y * px
    p_theta = (x * px + y * py) * cos_theta / sin_theta_safe - r * sin_theta * pz
    carter = p_theta**2 + cos_theta**2 * (a**2 * energy**2 + Lz**2 / sin_theta_safe**2)
    return {"r": r, "norm": norm, "energy": energy, "Lz": Lz, "carter": carter}


def r_horizon(M, a):
    return M + np.sqrt(max(M**2 - a**2, 0.0))


# ---------------------------------------------------------------------------
# Terminal stats line + likely-terminated heuristic
# ---------------------------------------------------------------------------
def likely_terminated_count(state, prev_state):
    """Photons frozen (bitwise-identical x,y,z,p) since the previous saved step
    never get their state touched again once terminate=true (see operator() in
    integrate_cart_ks.hpp), so exact equality across a step is a reliable proxy
    for termination -- `terminate` itself isn't written to disk."""
    if prev_state is None:
        return None
    same = np.ones_like(state["x1"], dtype=bool)
    for f in ("x1", "x2", "x3", "k0", "k1", "k2", "k3"):
        same &= state[f] == prev_state[f]
    return int(np.sum(same))


def format_stats_line(step, state, obs, prev_state):
    n = state["x1"].size
    terminated = likely_terminated_count(state, prev_state)
    term_str = f"{terminated:5d}" if terminated is not None else "  n/a"
    return (
        f"step {step:>6d} | N={n:5d} | est.terminated={term_str} | "
        f"<r>={np.mean(obs['r']):8.3f} r_min={np.min(obs['r']):8.3f} | "
        f"<|H|>={np.mean(np.abs(obs['norm'])):9.2e} | "
        f"<E>={np.mean(obs['energy']):8.4f}+-{np.std(obs['energy']):.1e} | "
        f"<Lz>={np.mean(obs['Lz']):8.4f}+-{np.std(obs['Lz']):.1e} | "
        f"<I>={np.mean(state['I']):8.4f}"
    )


# ---------------------------------------------------------------------------
# Dependency-free terminal 2D density histogram (x-y plane)
# ---------------------------------------------------------------------------
DENSITY_RAMP = " .:-=+*#%@"


def render_ascii_histogram(x, y, width, height, xlim, ylim, horizon_radius=None):
    xlo, xhi = xlim
    ylo, yhi = ylim
    if xhi <= xlo:
        xhi = xlo + 1.0
    if yhi <= ylo:
        yhi = ylo + 1.0

    H, _, _ = np.histogram2d(x, y, bins=[width, height], range=[[xlo, xhi], [ylo, yhi]])
    H = np.log1p(H).T[::-1]  # rows = y bins, high y first so it prints on top
    vmax = H.max()
    levels = np.zeros_like(H, dtype=int) if vmax <= 0 else (H / vmax * (len(DENSITY_RAMP) - 1)).astype(int)

    grid = [[DENSITY_RAMP[v] for v in row] for row in levels]

    if horizon_radius is not None and horizon_radius > 0:
        dx = (xhi - xlo) / width
        dy = (yhi - ylo) / height
        cell_diag = 0.5 * np.hypot(dx, dy)
        for row in range(height):
            cy = yhi - (row + 0.5) * dy  # matches the [::-1] flip above
            for col in range(width):
                cx = xlo + (col + 0.5) * dx
                if abs(np.hypot(cx, cy) - horizon_radius) < cell_diag:
                    grid[row][col] = "o"

    lines = ["".join(row) for row in grid]
    header = f"x in [{xlo:.2f}, {xhi:.2f}]"
    footer = f"y in [{ylo:.2f}, {yhi:.2f}]  ('o' = event horizon, r={horizon_radius:.3f})" \
        if horizon_radius is not None else f"y in [{ylo:.2f}, {yhi:.2f}]"
    return [header] + lines + [footer]


# ---------------------------------------------------------------------------
# `post` subcommand: full post-hoc analysis + plots
# ---------------------------------------------------------------------------
def cmd_post(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(args.output_dir)
    plots_dir = Path(args.plots_dir) if args.plots_dir else output_dir / "analysis_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    photon_steps = discover_photon_steps(output_dir)
    if not photon_steps:
        sys.exit(f"No photon_output_*.npy files found under {output_dir}")

    steps_sorted = sorted(photon_steps)
    rh = r_horizon(args.mass, args.spin)

    print(f"Found {len(steps_sorted)} step(s) in {output_dir}: "
          f"{steps_sorted[0]}..{steps_sorted[-1]}")
    print(f"Event horizon radius (M={args.mass}, a={args.spin}): {rh:.4f}\n")

    # Load every step up front (needed for trajectories + conservation plots).
    all_state, all_obs = {}, {}
    prev_state = None
    for step in steps_sorted:
        ranks = photon_steps[step]
        state = load_photon_step(output_dir, step, ranks)
        obs = compute_observables(state, args.mass, args.spin)
        all_state[step] = state
        all_obs[step] = obs
        print(format_stats_line(step, state, obs, prev_state))
        prev_state = state

    # --- Conserved-quantity drift vs step ---
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    for ax, key, label in zip(axes, ("norm", "energy", "Lz"), ("null norm H", "energy E=-p_t", "L_z")):
        means = np.array([np.mean(all_obs[s][key]) for s in steps_sorted])
        stds = np.array([np.std(all_obs[s][key]) for s in steps_sorted])
        ax.plot(steps_sorted, means, color="C0")
        ax.fill_between(steps_sorted, means - stds, means + stds, color="C0", alpha=0.3)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("step")
    fig.suptitle("Conserved-quantity drift across photons (mean +/- std)")
    fig.tight_layout()
    fig.savefig(plots_dir / "conserved_quantities.png", dpi=150)
    plt.close(fig)

    # --- Trajectories (subset of photons, consistent global index across steps) ---
    n_photons = all_state[steps_sorted[0]]["x1"].size
    rng = np.random.default_rng(42)
    subset = rng.choice(n_photons, size=min(args.max_trajectories, n_photons), replace=False)

    def trajectory_plot(coord_a, coord_b, label_a, label_b, filename):
        fig, ax = plt.subplots(figsize=(7, 7))
        traj_a = np.array([all_state[s][coord_a][subset] for s in steps_sorted])
        traj_b = np.array([all_state[s][coord_b][subset] for s in steps_sorted])
        ax.plot(traj_a, traj_b, color="steelblue", alpha=0.5, linewidth=0.8)
        ax.scatter(traj_a[-1], traj_b[-1], color="crimson", s=8, zorder=3, label="final position")
        theta_circ = np.linspace(0, 2 * np.pi, 200)
        ax.plot(rh * np.cos(theta_circ), rh * np.sin(theta_circ), color="black", linewidth=1.5,
                label=f"horizon r={rh:.2f}")
        ax.set_xlabel(label_a)
        ax.set_ylabel(label_b)
        ax.set_aspect("equal")
        ax.legend()
        ax.set_title(f"{len(subset)} photon trajectories")
        fig.tight_layout()
        fig.savefig(plots_dir / filename, dpi=150)
        plt.close(fig)

    trajectory_plot("x1", "x2", "x", "y", "trajectories_xy.png")
    trajectory_plot("x1", "x3", "x", "z", "trajectories_xz.png")

    # --- Final radius histogram ---
    fig, ax = plt.subplots(figsize=(7, 5))
    final_r = all_obs[steps_sorted[-1]]["r"]
    ax.hist(final_r, bins=50, color="slateblue")
    ax.axvline(rh, color="black", linestyle="--", label=f"horizon r={rh:.2f}")
    ax.set_xlabel("r")
    ax.set_ylabel("photon count")
    ax.set_title(f"Radius distribution at step {steps_sorted[-1]}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "radius_histogram.png", dpi=150)
    plt.close(fig)

    # --- Observation products (image / lightcurve / spectrum), last available step, per rank ---
    obs_steps = discover_observation_steps(output_dir)
    if obs_steps:
        last_obs_step = max(obs_steps)
        ranks = sorted(obs_steps[last_obs_step])
        products = load_observation_step(output_dir, last_obs_step, ranks)

        fig, axes = plt.subplots(len(ranks), 4, figsize=(14, 3.2 * len(ranks)), squeeze=False)
        for row, rank in enumerate(ranks):
            for col, field in enumerate(("image_I", "image_Q", "image_U", "image_V")):
                ax = axes[row][col]
                im = ax.imshow(products[rank][field].T, origin="lower", cmap="inferno")
                ax.set_title(f"rank {rank}: {field}")
                fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(f"Observation images at step {last_obs_step} "
                     "(per-rank local pixel grids -- not directly comparable/summable, see module docstring)")
        fig.tight_layout()
        fig.savefig(plots_dir / "observation_images.png", dpi=150)
        plt.close(fig)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        for rank in ranks:
            ax1.plot(products[rank]["lightcurve_I"], label=f"rank {rank}")
            ax2.plot(products[rank]["spectrum_I"], label=f"rank {rank}")
        ax1.set_title("Lightcurve I (per-rank local time bins)")
        ax1.set_xlabel("time bin")
        ax2.set_title("Spectrum I (per-rank local frequency bins)")
        ax2.set_xlabel("frequency bin")
        ax1.legend()
        ax2.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / "lightcurve_spectrum.png", dpi=150)
        plt.close(fig)
    else:
        print("No observation_step_*.npy files found -- skipping image/lightcurve/spectrum plots.")

    print(f"\nPlots written to {plots_dir}")


# ---------------------------------------------------------------------------
# `watch` subcommand: live terminal monitor while the simulation runs
# ---------------------------------------------------------------------------
def clear_previous_render(n_lines):
    if n_lines:
        sys.stdout.write(f"\x1b[{n_lines}A\x1b[J")


def cmd_watch(args):
    output_dir = Path(args.output_dir)
    term_w, term_h = shutil.get_terminal_size(fallback=(100, 40))
    width = args.width or max(40, min(140, term_w - 4))
    height = args.height or max(15, min(45, term_h - 10))
    rh = r_horizon(args.mass, args.spin)

    xlim = tuple(args.xlim) if args.xlim else None
    ylim = tuple(args.ylim) if args.ylim else None
    bbox = None  # running [xlo, xhi, ylo, yhi] used when xlim/ylim not pinned by the user

    print(f"Watching {output_dir} for new photon_output_* steps (Ctrl+C to stop)...")
    seen_steps = set()
    known_ranks = set()
    prev_state = None
    last_render_lines = 0

    try:
        while True:
            steps = discover_photon_steps(output_dir)
            for step in sorted(s for s in steps if s not in seen_steps):
                ranks_for_step = steps[step]
                known_ranks |= set(ranks_for_step)
                expected_ranks = range(args.mpi_size) if args.mpi_size else range(max(known_ranks) + 1)
                ready = all(
                    r in ranks_for_step and set(PHOTON_FIELDS) <= ranks_for_step[r] for r in expected_ranks
                )
                if not ready:
                    continue  # rank(s) haven't finished writing this step yet

                try:
                    state = load_photon_step(output_dir, step, expected_ranks)
                except (OSError, ValueError):
                    continue  # file(s) mid-write; retry next poll

                seen_steps.add(step)
                obs = compute_observables(state, args.mass, args.spin)

                if xlim is None or ylim is None:
                    x, y = state["x1"], state["x2"]
                    frame_box = [x.min(), x.max(), y.min(), y.max()]
                    bbox = frame_box if bbox is None else [
                        min(bbox[0], frame_box[0]), max(bbox[1], frame_box[1]),
                        min(bbox[2], frame_box[2]), max(bbox[3], frame_box[3]),
                    ]
                frame_xlim = xlim or (bbox[0], bbox[1])
                frame_ylim = ylim or (bbox[2], bbox[3])

                clear_previous_render(last_render_lines)
                stats_line = format_stats_line(step, state, obs, prev_state)
                hist_lines = render_ascii_histogram(
                    state["x1"], state["x2"], width, height, frame_xlim, frame_ylim, horizon_radius=rh
                )
                print(stats_line)
                print("\n".join(hist_lines))
                last_render_lines = 1 + len(hist_lines)
                prev_state = state

            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        print("\nStopped watching.")


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    common = dict(default="./output/", help="Directory the simulation writes to (matches utils.hpp output_directory)")

    p_post = sub.add_parser("post", help="Post-hoc analysis: terminal stats table + saved plots")
    p_post.add_argument("--output-dir", **common)
    p_post.add_argument("--plots-dir", default=None, help="Where to save plots (default: <output-dir>/analysis_plots)")
    p_post.add_argument("--mass", type=float, default=DEFAULT_M_BH)
    p_post.add_argument("--spin", type=float, default=DEFAULT_A_BH)
    p_post.add_argument("--max-trajectories", type=int, default=40)
    p_post.set_defaults(func=cmd_post)

    p_watch = sub.add_parser("watch", help="Live terminal monitor while the simulation runs")
    p_watch.add_argument("--output-dir", **common)
    p_watch.add_argument("--mass", type=float, default=DEFAULT_M_BH)
    p_watch.add_argument("--spin", type=float, default=DEFAULT_A_BH)
    p_watch.add_argument("--mpi-size", type=int, default=None,
                          help="Number of MPI ranks the run uses; default: auto-detect from files seen so far")
    p_watch.add_argument("--poll-interval", type=float, default=1.0, help="Seconds between directory polls")
    p_watch.add_argument("--width", type=int, default=None, help="Histogram width in characters (default: fit terminal)")
    p_watch.add_argument("--height", type=int, default=None, help="Histogram height in characters (default: fit terminal)")
    p_watch.add_argument("--xlim", type=float, nargs=2, default=None, metavar=("XMIN", "XMAX"))
    p_watch.add_argument("--ylim", type=float, nargs=2, default=None, metavar=("YMIN", "YMAX"))
    p_watch.set_defaults(func=cmd_watch)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
