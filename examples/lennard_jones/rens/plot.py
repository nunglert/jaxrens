"""Plot the outputs of the LJ-8 NPT example run.

Usage:
    cd experiments/examples/lj8_npt
    python plot.py                       # defaults: ./output, prefix=lj8_npt
    python plot.py --dir ./output --prefix lj8_npt --out ./figures
    python plot.py --t-min 0.01 --t-max 3.0 --n-t 200
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from jaxrens.postprocess import Monitor
from jaxrens.postprocess.plotting import (
    plot_acceptance_rates,
    plot_energy_trace,
    plot_free_energy,
    plot_heat_capacity,
    plot_log_evidence_trace,
    plot_partition_function,
    plot_step_sizes,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", default="./output", type=Path,
                   help="Run output directory (default: ./output)")
    p.add_argument("--prefix", default="lj8_npt",
                   help="Output file prefix (default: lj8_npt)")
    p.add_argument("--out", default="./figures", type=Path,
                   help="Where to save figures (default: ./figures)")
    p.add_argument("--t-min", default=0.01, type=float,
                   help="Minimum temperature for T-dependent plots")
    p.add_argument("--t-max", default=3.0, type=float,
                   help="Maximum temperature for T-dependent plots")
    p.add_argument("--n-t", default=200, type=int,
                   help="Number of temperature grid points")
    p.add_argument("--show", action="store_true",
                   help="Display figures interactively in addition to saving")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    mon = Monitor.from_directory(args.dir, prefix=args.prefix, label=args.prefix)
    print(f"Loaded: {mon}")
    print(f"  n_dead = {mon.n_dead}")
    print(f"  log_Z  = {mon.log_evidence:.4f}")
    print(f"  NPT    = {mon.is_npt}")

    T = np.linspace(args.t_min, args.t_max, args.n_t)

    # Trace plots (iteration-domain)
    fig_trace, (ax_e, ax_lz) = plt.subplots(1, 2, figsize=(11, 4))
    plot_energy_trace(mon, ax=ax_e)
    plot_log_evidence_trace(mon, ax=ax_lz)
    fig_trace.tight_layout()
    trace_path = args.out / f"{args.prefix}_traces.png"
    fig_trace.savefig(trace_path, dpi=150)
    print(f"Saved {trace_path}")

    # Thermodynamics plots (temperature-domain)
    fig_thermo, axs = plt.subplots(1, 3, figsize=(15, 4))
    plot_partition_function(mon, T, ax=axs[0])
    plot_heat_capacity(mon, T, ax=axs[1])
    plot_free_energy(mon, T, ax=axs[2])
    fig_thermo.tight_layout()
    thermo_path = args.out / f"{args.prefix}_thermodynamics.png"
    fig_thermo.savefig(thermo_path, dpi=150)
    print(f"Saved {thermo_path}")

    # Adaptation plots — only drawn when the trace file was produced
    if mon.adaptation_trace is not None:
        fig_adapt, (ax_ss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
        plot_step_sizes(mon, ax=ax_ss)
        ax_ss.set_title("Step sizes")
        plot_acceptance_rates(mon, ax=ax_acc)
        ax_acc.set_title("Acceptance rates")
        fig_adapt.tight_layout()
        adapt_path = args.out / f"{args.prefix}_adaptation.png"
        fig_adapt.savefig(adapt_path, dpi=150)
        print(f"Saved {adapt_path}")
    else:
        print("No adaptation trace found; skipping adaptation figure.")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
