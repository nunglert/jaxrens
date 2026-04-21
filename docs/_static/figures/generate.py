"""Generate the static PNG figures used by the core-concepts subpages.

Run from the repo root:
    python docs/_static/figures/generate.py

Each figure is generated from analytical / synthetic data so the script has
no dependency on live NS runs.  Committed PNGs live alongside this script.
Regenerate whenever the conceptual explanation changes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FIGDIR = Path(__file__).resolve().parent
plt.rcParams.update(
    {
        "figure.figsize": (5.0, 3.2),
        "savefig.dpi": 140,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 10,
    }
)


def fig_ns_prior_mass() -> None:
    """Prior-mass contraction and E_max trace, NS-loop page."""
    n_live = 128
    n_iter = 1600
    i = np.arange(n_iter)
    log_X = -i / n_live

    emin, emax0 = -6.0, 1.5
    emax = emax0 + (emin - emax0) * (1 - np.exp(-i / 250))
    emax += 0.08 * np.sin(i * 0.015) * np.exp(-i / 400)

    fig, ax1 = plt.subplots()
    color1 = "#1f77b4"
    ax1.plot(i, log_X, color=color1, lw=1.8, label=r"$\log X_i$")
    ax1.set_xlabel("NS iteration $i$")
    ax1.set_ylabel(r"$\log X_i$  (prior mass)", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "#d62728"
    ax2.plot(i, emax, color=color2, lw=1.5, label=r"$E_\mathrm{max}(i)$")
    ax2.set_ylabel(r"$E_\mathrm{max}(i)$  (arb.)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.grid(False)

    ax1.set_title("Prior-mass contraction & Emax descent (synthetic)")
    fig.tight_layout()
    fig.savefig(FIGDIR / "ns_prior_mass.png")
    plt.close(fig)


def fig_mwg_acceptance() -> None:
    """Per-move acceptance trace, MWG page (synthetic trace)."""
    rng = np.random.default_rng(0)
    n_iter = 600
    i = np.arange(n_iter)

    def noisy_line(target_start: float, target_end: float, tau: float) -> np.ndarray:
        baseline = target_end + (target_start - target_end) * np.exp(-i / tau)
        return np.clip(baseline + 0.06 * rng.standard_normal(n_iter), 0.0, 1.0)

    series = {
        "galilean": noisy_line(0.9, 0.45, 200),
        "volume":   noisy_line(0.8, 0.40, 150),
        "shear":    noisy_line(0.7, 0.35, 180),
        "stretch":  noisy_line(0.7, 0.35, 180),
    }

    fig, ax = plt.subplots()
    for name, y in series.items():
        ax.plot(i, y, lw=1.4, label=name)
    ax.axhspan(0.3, 0.5, color="tab:green", alpha=0.08, lw=0)
    ax.set_xlabel("NS iteration")
    ax.set_ylabel("acceptance rate")
    ax.set_ylim(0, 1)
    ax.set_title("Per-move acceptance under adaptation (synthetic)")
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGDIR / "mwg_acceptance.png")
    plt.close(fig)


def fig_ensemble_tilt() -> None:
    """H(V) for NVT, NPT, muVT (schematic), backends/ensembles page."""
    V = np.linspace(6.0, 60.0, 400)
    U = 3.0 / V ** 2 - 2.0 / V + 0.5
    U -= U.min()

    P_npt = 0.05
    mu = 0.8
    N_of_V = 1.0 + 0.01 * V

    H_nvt = U
    H_npt = U + P_npt * V
    H_muvt = U + P_npt * V - mu * N_of_V

    fig, ax = plt.subplots()
    ax.plot(V, H_nvt, lw=1.8, label=r"$H_\mathrm{NVT} = U$")
    ax.plot(V, H_npt, lw=1.8, label=r"$H_\mathrm{NPT} = U + PV$")
    ax.plot(V, H_muvt, lw=1.8, label=r"$H_{\mu VT} = U + PV - \mu N$")
    ax.set_xlabel(r"cell volume $V$")
    ax.set_ylabel(r"effective energy $H$")
    ax.set_title("Ensemble correction tilts the integrand (schematic)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGDIR / "ensemble_tilt.png")
    plt.close(fig)


def fig_rens_acceptance() -> None:
    """Synthetic RENS pairwise acceptance vs log pressure ratio."""
    log_ratio = np.linspace(0.0, 3.5, 64)
    alpha = np.exp(-0.65 * log_ratio)
    alpha += 0.02 * np.random.default_rng(1).standard_normal(log_ratio.size)
    alpha = np.clip(alpha, 0.0, 1.0)

    fig, ax = plt.subplots()
    ax.plot(log_ratio, alpha, "o-", lw=1.4, ms=4)
    ax.set_xlabel(r"$\log(P_j / P_i)$  (adjacent-pair pressure ratio)")
    ax.set_ylabel(r"RENS swap acceptance $\alpha_{ij}$")
    ax.set_ylim(0, 1)
    ax.set_title("Pressure-RENS acceptance falls off with pressure spacing")
    fig.tight_layout()
    fig.savefig(FIGDIR / "rens_acceptance.png")
    plt.close(fig)


def fig_pytree_shapes() -> None:
    """Shape table for SingleRun / VmapRuns / PmapVmapRuns."""
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.grid(False)

    header = ["", "positions", "energy", "log_evidence"]
    rows = [
        ("SingleRun",       "(K, A, 3)",       "(K,)",      "()"),
        ("VmapRuns",        "(R, K, A, 3)",    "(R, K)",    "(R,)"),
        ("PmapVmapRuns",    "(G, P, K, A, 3)", "(G, P, K)", "(G, P)"),
    ]

    col_x = [0.4, 2.2, 5.4, 7.6]
    header_y = 4.1
    row_ys = [3.1, 2.2, 1.3]
    cell_colors = {
        "SingleRun":     "#e7f1fb",
        "VmapRuns":      "#fff3e0",
        "PmapVmapRuns":  "#e9f5e9",
    }

    for x, text in zip(col_x, header):
        ax.text(x, header_y, text, fontsize=10, fontweight="bold", family="monospace")

    for row, y in zip(rows, row_ys):
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (col_x[0] - 0.2, y - 0.35),
                col_x[-1] + 2.0 - col_x[0] + 0.2,
                0.7,
                boxstyle="round,pad=0.05",
                linewidth=0.8,
                facecolor=cell_colors[row[0]],
                edgecolor="#666",
            )
        )
        for x, cell in zip(col_x, row):
            ax.text(x, y, cell, fontsize=10, family="monospace", va="center")

    ax.set_title(
        "Batch-axis shapes for the three BatchDescriptor backends",
        fontsize=11, pad=6,
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / "pytree_shapes.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_ns_prior_mass()
    fig_mwg_acceptance()
    fig_ensemble_tilt()
    fig_rens_acceptance()
    fig_pytree_shapes()
    print(f"wrote figures to {FIGDIR}")
