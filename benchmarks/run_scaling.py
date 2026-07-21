#!/usr/bin/env python3
"""Run reproducible strong- and weak-scaling sweeps and write a CSV summary."""

from __future__ import annotations

import argparse
import csv
import shlex
import statistics
import subprocess
import tempfile
import time
from pathlib import Path


def parse_ranks(value: str) -> list[int]:
    ranks = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not ranks or any(rank <= 0 for rank in ranks):
        raise argparse.ArgumentTypeError("ranks must be a comma-separated list of positive integers")
    return ranks


def write_config(path: Path, photons: int, steps: int) -> None:
    path.write_text(
        f"""[black_hole]
mass = 1.0
spin = 0.9

[camera]
use_pinhole = false
use_image = true
distance = 100.0
plane_dim1 = 20.0
plane_dim2 = 20.0
plane_theta = 1.5707963267948966
plane_phi = 0.0

[photons]
count = {photons}
camera_frequency_hz = 2.30e11
camera_packet_energy_erg = 1.0

[integration]
max_steps = {steps}
termination_percent = 1.0
initial_step = 1e-4
atol = 1e-8
rtol = 1e-8

[termination]
r_min = 1.5
r_max = 1000000.0

[output]
interval = {steps + 1}
variables = ["id"]
""",
        encoding="utf-8",
    )


def run_once(args: argparse.Namespace, ranks: int, photons: int, work_dir: Path) -> float:
    config = work_dir / f"config_{ranks}_{photons}.toml"
    output = work_dir / f"output_{ranks}_{photons}"
    write_config(config, photons, args.steps)
    command = [
        *shlex.split(args.launcher),
        args.tasks_flag,
        str(ranks),
        str(args.executable),
        "--config",
        str(config),
        "--vacuum",
        "--output-dir",
        str(output),
        "--integrator",
        args.integrator,
    ]
    started = time.perf_counter()
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        rendered = " ".join(shlex.quote(part) for part in command)
        raise RuntimeError(
            f"scaling run failed ({rendered})\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--ranks", type=parse_ranks, default=parse_ranks("1,2,4"))
    parser.add_argument("--mode", choices=("strong", "weak", "both"), default="both")
    parser.add_argument("--photons", type=int, default=100_000,
                        help="total photons for strong scaling; photons per rank for weak scaling")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--launcher", default="mpirun",
                        help="launcher command, including any site-specific options")
    parser.add_argument("--tasks-flag", default="-n")
    parser.add_argument("--integrator", choices=("rk4", "rk45"), default="rk4")
    parser.add_argument("--output", type=Path, default=Path("scaling_results.csv"))
    args = parser.parse_args()

    if not args.executable.is_file():
        parser.error(f"executable does not exist: {args.executable}")
    if args.photons <= 0 or args.steps <= 0 or args.repeats <= 0:
        parser.error("photons, steps, and repeats must be positive")

    modes = ("strong", "weak") if args.mode == "both" else (args.mode,)
    rows: list[dict[str, int | float | str]] = []
    with tempfile.TemporaryDirectory(prefix="grraytracer-scaling-") as temporary:
        work_dir = Path(temporary)
        for mode in modes:
            mode_rows = []
            for ranks in args.ranks:
                total_photons = args.photons if mode == "strong" else args.photons * ranks
                samples = [
                    run_once(args, ranks, total_photons, work_dir)
                    for _ in range(args.repeats)
                ]
                elapsed = statistics.median(samples)
                row = {
                    "mode": mode,
                    "ranks": ranks,
                    "total_photons": total_photons,
                    "steps": args.steps,
                    "median_seconds": elapsed,
                    "photon_steps_per_second": total_photons * args.steps / elapsed,
                }
                mode_rows.append(row)
            baseline = mode_rows[0]
            for row in mode_rows:
                speedup = baseline["median_seconds"] / row["median_seconds"]
                if mode == "strong":
                    efficiency = speedup / (row["ranks"] / baseline["ranks"])
                else:
                    efficiency = speedup
                row["speedup"] = speedup
                row["parallel_efficiency"] = efficiency
                rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
