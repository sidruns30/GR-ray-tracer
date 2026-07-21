#!/usr/bin/env python3
"""
Analyze GR-ray-tracer photon output: post-hoc plots, plus a live terminal
monitor (photon stats + an in-terminal x-y density histogram) that can run
alongside the C++ binary while it is still writing output.

Each rank writes one `output_step_<step>_rank<rank>.npz` archive per checkpoint.
The archive contains the fields selected by `output.variables` in the TOML
configuration or `--output-variables` on the simulation command line.

When `id` is present, concatenated rank output is sorted by globally unique
packet ID so trajectories remain aligned across steps. Without `id`, the
fallback identity is (rank, local index), which is stable because terminated
packets are not removed.

image_I/Q/U/V (see build_observation_products() in
src/radiative_transfer/observation.hpp) bin each photon by its FIXED
camera-screen launch coordinate (theta_disp/phi_disp, set once at init time in
src/radiative_transfer/initialize_photons.hpp) over a range set by the active
camera's screen geometry (image-plane camera: +-plane_dim1/2 x +-plane_dim2/2;
pinhole camera: +-pinhole_aperture_radius square) -- NOT the photon's current,
time-varying x/y position, and NOT a per-call min/max. So pixel (i, j) means
the same camera-screen location in every output checkpoint, each dump is
one "photograph" of where photons-per-screen-pixel currently are, and (since
the range depends only on config, not on the data) image products ARE
directly comparable/summable across MPI ranks and across steps.

Caveat that still applies: lightcurve_I (binned by coordinate time x0) and
spectrum_I (log-binned by frequency in Hz) are each rescaled to *that call's
own* local min/max, per rank, per dump -- so those two products
are NOT directly comparable across ranks or across steps, and this script
plots them per-rank rather than pretending otherwise.

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

# Camera screen geometry -- must match src/utils.hpp / config TOML (camera.*).
# Used only to label/extent the image_I/Q/U/V axes; wrong values mislabel the
# image but don't affect which pixel a photon lands in (that's fixed in the
# C++ side, see observation.hpp). Override with --plane-dim1/--plane-dim2 for
# the image-plane camera, or --pinhole-aperture-radius for the pinhole camera.
DEFAULT_PLANE_DIM1 = 20.0
DEFAULT_PLANE_DIM2 = 20.0
DEFAULT_PINHOLE_APERTURE_RADIUS = 50.0

PHOTON_FIELDS = [
    "id", "frequency", "x0", "x1", "x2", "x3", "k0", "k1", "k2", "k3",
    "I", "Q", "U", "V", "dlambda", "terminate", "theta_disp", "phi_disp",
]
OBS_FIELDS = [
    "image_I", "image_Q", "image_U", "image_V", "lightcurve_I",
    "spectrum_frequency_hz", "spectrum_I",
]
OUTPUT_RE = re.compile(r"output_step_(\d+)_rank(\d+)\.npz$")


# ---------------------------------------------------------------------------
# Discovery / loading
# ---------------------------------------------------------------------------
def discover_output_steps(output_dir: Path):
    """Return {step: set(ranks)} for complete, atomically published archives."""
    steps = {}
    for path in output_dir.glob("output_step_*_rank*.npz"):
        m = OUTPUT_RE.match(path.name)
        if not m:
            continue
        step, rank = int(m.group(1)), int(m.group(2))
        steps.setdefault(step, set()).add(rank)
    return steps


def archive_path(output_dir: Path, step: int, rank: int):
    return output_dir / f"output_step_{step}_rank{rank}.npz"


def load_photon_step(output_dir: Path, step: int, ranks):
    """Concatenate photon arrays available on every requested rank."""
    rank_data = []
    for rank in sorted(ranks):
        with np.load(archive_path(output_dir, step, rank)) as archive:
            rank_data.append({name: archive[name] for name in archive.files if name in PHOTON_FIELDS})
    common_fields = set.intersection(*(set(data) for data in rank_data)) if rank_data else set()
    combined = {
        field: np.concatenate([data[field] for data in rank_data])
        for field in PHOTON_FIELDS if field in common_fields
    }
    if "id" in combined:
        order = np.argsort(combined["id"], kind="stable")
        combined = {field: values[order] for field, values in combined.items()}
    return combined


def load_observation_step(output_dir: Path, step: int, ranks):
    """Load observation arrays without combining rank-local histogram axes."""
    out = {}
    for rank in sorted(ranks):
        with np.load(archive_path(output_dir, step, rank)) as archive:
            out[rank] = {name: archive[name] for name in archive.files if name in OBS_FIELDS}
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
    """Use `terminate` when selected, otherwise infer frozen photon states."""
    if "terminate" in state:
        return int(np.count_nonzero(state["terminate"]))
    required = {"x1", "x2", "x3", "k0", "k1", "k2", "k3"}
    if prev_state is None or not required <= state.keys() or not required <= prev_state.keys():
        return None
    same = np.ones_like(state["x1"], dtype=bool)
    for f in ("x1", "x2", "x3", "k0", "k1", "k2", "k3"):
        same &= state[f] == prev_state[f]
    return int(np.sum(same))


def format_stats_line(step, state, obs, prev_state):
    n = state["x1"].size
    terminated = likely_terminated_count(state, prev_state)
    term_str = f"{terminated:5d}" if terminated is not None else "  n/a"
    intensity = f" | <I>={np.mean(state['I']):8.4f}" if "I" in state else ""
    return (
        f"step {step:>6d} | N={n:5d} | est.terminated={term_str} | "
        f"<r>={np.mean(obs['r']):8.3f} r_min={np.min(obs['r']):8.3f} | "
        f"<|H|>={np.mean(np.abs(obs['norm'])):9.2e} | "
        f"<E>={np.mean(obs['energy']):8.4f}+-{np.std(obs['energy']):.1e} | "
        f"<Lz>={np.mean(obs['Lz']):8.4f}+-{np.std(obs['Lz']):.1e}"
        f"{intensity}"
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

    photon_steps = discover_output_steps(output_dir)
    if not photon_steps:
        sys.exit(f"No output_step_*_rank*.npz files found under {output_dir}")

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
        required = {"x1", "x2", "x3", "k0", "k1", "k2", "k3"}
        if not required <= state.keys():
            missing = ", ".join(sorted(required - state.keys()))
            sys.exit("Post-hoc trajectory/conservation analysis requires output variables "
                     f"x1,x2,x3,k0,k1,k2,k3; missing: {missing}")
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

    # --- Observation images: one camera-screen picture per dump ---
    # image_I/Q/U/V now bin by each photon's fixed screen coordinate over a
    # range set purely by camera config (see module docstring), so unlike
    # lightcurve/spectrum they're safe to sum across ranks into one image.
    obs_steps = photon_steps
    first_obs_step = min(obs_steps)
    first_obs_ranks = sorted(obs_steps[first_obs_step])
    first_products = load_observation_step(output_dir, first_obs_step, first_obs_ranks)
    common_obs_fields = set.intersection(*(set(first_products[rank]) for rank in first_obs_ranks))
    image_fields = [field for field in ("image_I", "image_Q", "image_U", "image_V")
                    if field in common_obs_fields]
    if image_fields:
        half1, half2 = (args.plane_dim1 / 2.0, args.plane_dim2 / 2.0) if not args.pinhole \
            else (args.pinhole_aperture_radius, args.pinhole_aperture_radius)
        extent = [-half1, half1, -half2, half2]

        images_dir = plots_dir / "observation_images"
        images_dir.mkdir(parents=True, exist_ok=True)
        for step in sorted(obs_steps):
            ranks = sorted(obs_steps[step])
            products = load_observation_step(output_dir, step, ranks)

            fig, axes = plt.subplots(1, len(image_fields), figsize=(3.5 * len(image_fields), 3.5), squeeze=False)
            for col, field in enumerate(image_fields):
                combined = np.zeros_like(products[ranks[0]][field])
                for rank in ranks:
                    combined += products[rank][field]
                im = axes[0, col].imshow(combined, origin="lower", cmap="inferno", extent=extent)
                axes[0, col].set_title(field)
                axes[0, col].set_xlabel("screen u")
                axes[0, col].set_ylabel("screen v")
                fig.colorbar(im, ax=axes[0, col], fraction=0.046)
            fig.suptitle(f"Observation image at step {step} (summed over {len(ranks)} rank(s))")
            fig.tight_layout()
            fig.savefig(images_dir / f"observation_image_step_{step:06d}.png", dpi=150)
            plt.close(fig)
        print(f"Wrote {len(obs_steps)} per-dump observation image(s) to {images_dir}")

    else:
        print("No image arrays were selected -- skipping observation image plots.")

    series_fields = [field for field in ("lightcurve_I", "spectrum_I") if field in common_obs_fields]
    if series_fields:
        last_obs_step = max(obs_steps)
        ranks = sorted(obs_steps[last_obs_step])
        products = load_observation_step(output_dir, last_obs_step, ranks)
        fig, axes = plt.subplots(1, len(series_fields), figsize=(6 * len(series_fields), 4), squeeze=False)
        for col, field in enumerate(series_fields):
            for rank in ranks:
                x_values = np.arange(products[rank][field].size)
                if field == "spectrum_I" and "spectrum_frequency_hz" in products[rank]:
                    x_values = products[rank]["spectrum_frequency_hz"]
                    axes[0, col].set_xscale("log")
                    axes[0, col].set_xlabel("frequency (Hz)")
                axes[0, col].plot(x_values, products[rank][field], label=f"rank {rank}")
            axes[0, col].set_title(f"{field} (per-rank local bins)")
            if field != "spectrum_I" or "spectrum_frequency_hz" not in products[ranks[0]]:
                axes[0, col].set_xlabel("bin")
            axes[0, col].legend()
        fig.tight_layout()
        fig.savefig(plots_dir / "lightcurve_spectrum.png", dpi=150)
        plt.close(fig)

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

    print(f"Watching {output_dir} for new output archives (Ctrl+C to stop)...")
    seen_steps = set()
    known_ranks = set()
    prev_state = None
    last_render_lines = 0

    try:
        while True:
            steps = discover_output_steps(output_dir)
            for step in sorted(s for s in steps if s not in seen_steps):
                ranks_for_step = steps[step]
                known_ranks |= set(ranks_for_step)
                expected_ranks = range(args.mpi_size) if args.mpi_size else range(max(known_ranks) + 1)
                ready = all(r in ranks_for_step for r in expected_ranks)
                if not ready:
                    continue  # rank(s) haven't finished writing this step yet

                try:
                    state = load_photon_step(output_dir, step, expected_ranks)
                except (OSError, ValueError):
                    continue

                required = {"x1", "x2", "x3", "k0", "k1", "k2", "k3"}
                if not required <= state.keys():
                    missing = ", ".join(sorted(required - state.keys()))
                    sys.exit("Live monitoring requires output variables x1,x2,x3,k0,k1,k2,k3; "
                             f"missing: {missing}")

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
    p_post.add_argument("--pinhole", action="store_true",
                         help="Set if the run used the pinhole camera (camera.use_pinhole=true) "
                              "instead of the default image-plane camera, for correct image screen extent")
    p_post.add_argument("--plane-dim1", type=float, default=DEFAULT_PLANE_DIM1,
                         help="Image-plane camera width (config camera.plane_dim1); ignored with --pinhole")
    p_post.add_argument("--plane-dim2", type=float, default=DEFAULT_PLANE_DIM2,
                         help="Image-plane camera height (config camera.plane_dim2); ignored with --pinhole")
    p_post.add_argument("--pinhole-aperture-radius", type=float, default=DEFAULT_PINHOLE_APERTURE_RADIUS,
                         help="Pinhole camera screen radius (src/utils.hpp pinhole_aperture_radius); only used with --pinhole")
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
