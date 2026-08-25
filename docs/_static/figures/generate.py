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

    def noisy_line(
        target_start: float, target_end: float, tau: float
    ) -> np.ndarray:
        baseline = target_end + (target_start - target_end) * np.exp(-i / tau)
        return np.clip(baseline + 0.06 * rng.standard_normal(n_iter), 0.0, 1.0)

    series = {
        "galilean": noisy_line(0.9, 0.45, 200),
        "volume": noisy_line(0.8, 0.40, 150),
        "shear": noisy_line(0.7, 0.35, 180),
        "stretch": noisy_line(0.7, 0.35, 180),
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
    U = 3.0 / V**2 - 2.0 / V + 0.5
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
        ("SingleRun", "(K, A, 3)", "(K,)", "()"),
        ("VmapRuns", "(R, K, A, 3)", "(R, K)", "(R,)"),
        ("PmapVmapRuns", "(G, P, K, A, 3)", "(G, P, K)", "(G, P)"),
    ]

    col_x = [0.4, 2.2, 5.4, 7.6]
    header_y = 4.1
    row_ys = [3.1, 2.2, 1.3]
    cell_colors = {
        "SingleRun": "#e7f1fb",
        "VmapRuns": "#fff3e0",
        "PmapVmapRuns": "#e9f5e9",
    }

    for x, text in zip(col_x, header):
        ax.text(
            x,
            header_y,
            text,
            fontsize=10,
            fontweight="bold",
            family="monospace",
        )

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
        fontsize=11,
        pad=6,
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / "pytree_shapes.png")
    plt.close(fig)


def _count_loc(path: Path) -> int:
    """Non-blank, non-comment LoC in a Python file."""
    n = 0
    with path.open() as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                n += 1
    return n


def _walk_pkg(src_root: Path) -> list[tuple[str, str, int, int, int]]:
    """Walk ``src_root`` and collect (id, parent, loc, n_classes, n_funcs) rows.

    Skips `__pycache__` and files ≤5 LoC. `id` is the dotted module path
    (e.g. ``jaxrens.sampling.moves.galilean``), `parent` is the dotted
    parent path (``jaxrens.sampling.moves``). The root node's parent is
    the empty string.
    """
    import ast

    skip_dirs = {"__pycache__"}
    rows: list[tuple[str, str, int, int, int]] = []
    pkg_name = src_root.name

    rows.append((pkg_name, "", 0, 0, 0))

    def _analyze(py_file: Path) -> tuple[int, int, int]:
        loc = _count_loc(py_file)
        try:
            tree = ast.parse(py_file.read_text())
        except SyntaxError:
            return loc, 0, 0
        n_classes = sum(isinstance(n, ast.ClassDef) for n in ast.walk(tree))
        n_funcs = sum(
            isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            for n in ast.walk(tree)
        )
        return loc, n_classes, n_funcs

    subpkgs: dict[str, list[tuple[str, int, int, int]]] = {}
    top_modules: list[tuple[str, int, int, int]] = []

    for entry in sorted(src_root.iterdir()):
        if entry.name in skip_dirs or entry.name.startswith("."):
            continue
        if entry.is_dir():
            name = entry.name
            for py in sorted(entry.rglob("*.py")):
                if any(p in skip_dirs for p in py.parts):
                    continue
                loc, nc, nf = _analyze(py)
                if loc <= 5:
                    continue
                rel = py.relative_to(src_root).with_suffix("")
                mod_id = f"{pkg_name}." + ".".join(rel.parts)
                parent_id = f"{pkg_name}." + ".".join(rel.parts[:-1])
                subpkgs.setdefault(name, []).append((mod_id, loc, nc, nf))
                if parent_id != f"{pkg_name}.{name}":
                    rows.append((parent_id, f"{pkg_name}.{name}", 0, 0, 0))
        elif entry.suffix == ".py":
            loc, nc, nf = _analyze(entry)
            if loc <= 5:
                continue
            top_modules.append((f"{pkg_name}.{entry.stem}", loc, nc, nf))

    parent_seen = {"", pkg_name}
    for subpkg_name, modules in subpkgs.items():
        sub_id = f"{pkg_name}.{subpkg_name}"
        rows.append((sub_id, pkg_name, 0, 0, 0))
        parent_seen.add(sub_id)
        for mod_id, loc, nc, nf in modules:
            parent_id = ".".join(mod_id.split(".")[:-1])
            if parent_id not in parent_seen:
                grandparent = ".".join(parent_id.split(".")[:-1]) or pkg_name
                rows.append((parent_id, grandparent, 0, 0, 0))
                parent_seen.add(parent_id)
            rows.append((mod_id, parent_id, loc, nc, nf))

    for mod_id, loc, nc, nf in top_modules:
        rows.append((mod_id, pkg_name, loc, nc, nf))

    seen = set()
    dedup = []
    for row in rows:
        if row[0] in seen:
            continue
        seen.add(row[0])
        dedup.append(row)

    # Bottom-up aggregation: every parent's (LoC, classes, functions) becomes
    # the sum over its entire subtree.  Needed so hover tooltips on subpackage
    # tiles show the right totals and so plotly's ``branchvalues="total"``
    # mode renders without value mismatches.
    children_map: dict[str, list[str]] = {}
    row_by_id: dict[str, tuple[str, str, int, int, int]] = {}
    for row in dedup:
        row_by_id[row[0]] = row
        children_map.setdefault(row[1], []).append(row[0])

    totals: dict[str, tuple[int, int, int]] = {}

    def _total(rid: str) -> tuple[int, int, int]:
        if rid in totals:
            return totals[rid]
        _id, _parent, loc, nc, nf = row_by_id[rid]
        sub_loc, sub_nc, sub_nf = loc, nc, nf
        for child in children_map.get(rid, []):
            cl, cc, cf = _total(child)
            sub_loc += cl
            sub_nc += cc
            sub_nf += cf
        totals[rid] = (sub_loc, sub_nc, sub_nf)
        return totals[rid]

    for rid in row_by_id:
        _total(rid)

    return [(rid, parent, *totals[rid]) for rid, parent, _, _, _ in dedup]


def _subpkg_of(rid: str, pkg_name: str) -> str:
    """Top-level subpackage name for tile ``rid`` (e.g. ``sampling``).

    Returns ``"_root"`` for the package root itself, so it can be coloured
    with a neutral tone distinct from any subpackage.
    """
    if rid == pkg_name or not rid.startswith(pkg_name + "."):
        return "_root"
    return rid.split(".", 2)[1]


def _fig_pkg_treemap_html(rows: list[tuple[str, str, int, int, int]]) -> None:
    """Interactive plotly treemap (HTML with CDN plotly.js)."""
    import plotly.colors as pc
    import plotly.graph_objects as go

    pkg_name = next((r[0] for r in rows if r[1] == ""), "jaxrens")
    ids = [r[0] for r in rows]
    parents = [r[1] for r in rows]
    labels = [r[0].rsplit(".", 1)[-1] if "." in r[0] else r[0] for r in rows]
    values = [r[2] for r in rows]
    customdata = [[r[3], r[4], r[0]] for r in rows]

    subpkgs = sorted({_subpkg_of(r[0], pkg_name) for r in rows} - {"_root"})
    palette = pc.qualitative.Set3
    subpkg_color = {
        name: palette[i % len(palette)] for i, name in enumerate(subpkgs)
    }
    subpkg_color["_root"] = "#f0f0f0"
    colors = [subpkg_color[_subpkg_of(r[0], pkg_name)] for r in rows]

    fig = go.Figure(
        go.Treemap(
            ids=ids,
            parents=parents,
            labels=labels,
            values=values,
            # Parent values are real subtree totals (see ``_walk_pkg``'s
            # bottom-up aggregation), so ``total`` mode is the correct
            # choice: plotly verifies parent.value == sum(children.value)
            # and the hover tooltip shows the true cumulated LoC.
            branchvalues="total",
            customdata=customdata,
            hovertemplate=(
                "<b>%{customdata[2]}</b><br>"
                "Lines of code: %{value}<br>"
                "Classes: %{customdata[0]}<br>"
                "Functions: %{customdata[1]}"
                "<extra></extra>"
            ),
            textinfo="label",
            root_color="#eeeeee",
            marker=dict(
                colors=colors,
                line=dict(color="white", width=1.2),
            ),
            pathbar=dict(visible=True, thickness=24),
        )
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        title=dict(
            text=(
                "jaxrens package structure — click a tile to drill down, "
                "hover for details, export SVG from the plotly toolbar"
            ),
            font=dict(size=12),
            x=0.5,
            y=0.99,
        ),
        height=600,
    )
    out_path = FIGDIR / "pkg_treemap.html"
    fig.write_html(
        out_path,
        include_plotlyjs="cdn",
        full_html=True,
        div_id="jaxrens-pkg-treemap",
    )
    print(f"  wrote {out_path}")


def _plotly_color_to_rgba(c: str) -> tuple[float, float, float, float]:
    """Convert a plotly palette string (``rgb(r,g,b)`` or ``#rrggbb``) to RGBA.

    Matplotlib rejects ``rgb(141,211,199)`` — it only accepts percentage rgb or
    hex.  Translate plotly's qualitative palettes for the SVG path.
    """
    import re

    s = c.strip()
    m = re.match(r"rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", s)
    if m:
        r, g, b = (int(m.group(i)) for i in (1, 2, 3))
        return (r / 255.0, g / 255.0, b / 255.0, 1.0)
    return s  # hex / named colour — matplotlib handles it directly.


def _fig_pkg_treemap_svg(rows: list[tuple[str, str, int, int, int]]) -> None:
    """Static SVG fallback via matplotlib + squarify (two-level flatten)."""
    import matplotlib.patches as mpatches
    import plotly.colors as pc
    import squarify

    pkg_name = next((r[0] for r in rows if r[1] == ""), "jaxrens")
    subpackages: dict[str, dict[str, int]] = {}
    top_files: dict[str, int] = {}
    for rid, parent, loc, _, _ in rows:
        if rid == pkg_name or loc == 0:
            continue
        if parent == pkg_name:
            top_files[rid.rsplit(".", 1)[-1]] = loc
        else:
            sub = (
                parent.split(".")[1]
                if parent.startswith(pkg_name + ".")
                else parent
            )
            leaf = rid.split(".", 2)[-1]
            subpackages.setdefault(sub, {})[leaf] = loc

    groups = dict(subpackages)
    if top_files:
        groups["(root)"] = top_files

    # Same Set3 qualitative palette as the HTML treemap so the two artefacts
    # look visually consistent (subpackage identity is conveyed by hue).
    group_order = sorted(
        groups, key=lambda g: sum(groups[g].values()), reverse=True
    )
    palette = pc.qualitative.Set3
    group_colors = {
        name: _plotly_color_to_rgba(palette[i % len(palette)])
        for i, name in enumerate(group_order)
    }

    W, H = 100.0, 60.0
    totals = [sum(groups[g].values()) for g in group_order]
    outer = squarify.normalize_sizes(totals, W, H)
    outer_rects = squarify.squarify(outer, 0, 0, W, H)

    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.grid(False)

    for g_name, rect in zip(group_order, outer_rects):
        base = group_colors[g_name]
        ax.add_patch(
            mpatches.Rectangle(
                (rect["x"], rect["y"]),
                rect["dx"],
                rect["dy"],
                facecolor=base,
                edgecolor="white",
                linewidth=2.0,
                alpha=0.30,
            )
        )
        mods = groups[g_name]
        sizes = list(mods.values())
        inner_sizes = squarify.normalize_sizes(sizes, rect["dx"], rect["dy"])
        inner_rects = squarify.squarify(
            inner_sizes,
            rect["x"],
            rect["y"],
            rect["dx"],
            rect["dy"],
        )
        for (m_name, _), ir in zip(mods.items(), inner_rects):
            ax.add_patch(
                mpatches.Rectangle(
                    (ir["x"], ir["y"]),
                    ir["dx"],
                    ir["dy"],
                    facecolor=base,
                    edgecolor="white",
                    linewidth=0.8,
                    alpha=0.80,
                )
            )
            if ir["dx"] > 5.0 and ir["dy"] > 2.2:
                ax.text(
                    ir["x"] + ir["dx"] / 2,
                    ir["y"] + ir["dy"] / 2,
                    m_name,
                    ha="center",
                    va="center",
                    fontsize=min(9, ir["dx"] / 6.0, ir["dy"] / 1.3),
                )
        if rect["dx"] > 10 and rect["dy"] > 4:
            ax.text(
                rect["x"] + 0.6,
                rect["y"] + rect["dy"] - 0.8,
                g_name,
                ha="left",
                va="top",
                fontsize=11,
                fontweight="bold",
                bbox=dict(
                    facecolor="white", alpha=0.85, pad=1.8, edgecolor="none"
                ),
            )

    total = sum(totals)
    ax.set_title(
        f"jaxrens package structure — subpackages outside, modules inside, "
        f"areas ∝ lines of code (total {total})",
        fontsize=10,
        pad=6,
    )
    fig.tight_layout()
    out_path = FIGDIR / "pkg_treemap.svg"
    fig.savefig(out_path, format="svg")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Tutorial energy surfaces
#
# These show the reader *what is being sampled* before any NS output is
# discussed.  Both are computed straight from the backend, so they cannot
# drift from the model the tutorial configs actually run; the walker overlay
# in the Gaussian figure is read from a completed tutorial run and is skipped
# if that run's output is not present.
# ---------------------------------------------------------------------------

_TUT = FIGDIR.parents[2] / "examples" / "tutorials"
_TUTFIG = FIGDIR / "tutorials"


def _tutorial_config(name: str) -> dict:
    """Load a tutorial's YAML config.

    The surface figures are built from the same file the tutorial tells the
    reader to run, so tuning a parameter moves the plot with it instead of
    leaving the two quietly disagreeing.
    """
    import yaml

    with open(_TUT / name / "config.yaml") as fh:
        return yaml.safe_load(fh)


def fig_gauss2d_surface() -> None:
    """Gaussian-mixture landscape, with the walker population contracting."""
    import jax.numpy as jnp

    from jaxrens.backends.toy import create_gaussian_mixture

    cfg = _tutorial_config("00_gaussian_2d")
    centers = cfg["backend"]["centers"]
    backend = create_gaussian_mixture(
        centers=centers, sigma=cfg["backend"]["sigma"]
    )

    lim = 3.2
    n = 320
    gx = np.linspace(-lim, lim, n)
    X, Y = np.meshgrid(gx, gx)
    pts = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], axis=-1)
    cell = jnp.eye(3) * 8.0
    types = jnp.zeros(1, dtype=int)
    E = np.array(
        [
            float(backend(jnp.asarray(p)[None, :], types, cell).energy)
            for p in pts
        ]
    ).reshape(X.shape)

    snaps = sorted(
        (_TUT / "00_gaussian_2d" / "output").glob("*.traj.snap.*.extxyz"),
        key=lambda q: int(q.name.split(".snap.")[1].split(".")[0]),
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    for ax in axes:
        im = ax.pcolormesh(X, Y, E, cmap="viridis", shading="auto")
        ax.contour(X, Y, E, levels=14, colors="w", linewidths=0.4, alpha=0.6)
        ax.set_xlabel("x")
        ax.set_aspect("equal")
        # Pin the limits: the scatter overlay would otherwise stretch the
        # right-hand panel and the two would no longer be comparable.
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
    axes[0].set_ylabel("y")
    fig.colorbar(im, ax=axes, label="energy  $E(x, y)$", shrink=0.9)

    axes[0].scatter(
        [c[0] for c in centers],
        [c[1] for c in centers],
        marker="x",
        c="r",
        s=45,
        lw=1.4,
        label="Gaussian centres",
    )
    axes[0].legend(loc="lower left", fontsize=8, framealpha=0.85)
    axes[0].set_title(
        "the landscape: four basins, one of them merged (deeper)", fontsize=10
    )

    if snaps:
        from ase.io import read

        picks = [snaps[0], snaps[len(snaps) // 2], snaps[-1]]
        colors = ["#ffffff", "#ffd166", "#ef476f"]
        for q, colour in zip(picks, colors, strict=False):
            it = int(q.name.split(".snap.")[1].split(".")[0])
            pos = np.array([a.get_positions()[0] for a in read(q, index=":")])
            axes[1].scatter(
                pos[:, 0],
                pos[:, 1],
                s=7,
                c=colour,
                edgecolors="k",
                linewidths=0.2,
                label=f"iteration {it}",
            )
        axes[1].legend(loc="lower left", fontsize=8, framealpha=0.85)
        axes[1].set_title(
            "live population contracting onto the deepest basin", fontsize=10
        )
    else:
        axes[1].set_title("(run the tutorial to overlay walkers)", fontsize=10)

    _TUTFIG.mkdir(parents=True, exist_ok=True)
    out = _TUTFIG / "gauss2d_surface.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}")


def fig_rens_toy_surface() -> None:
    """Toy-model enthalpy surface H(a, d), irreducible wedge, per pressure."""
    import jax.numpy as jnp

    from jaxrens.backends.toy import create_rens_toy

    # Parameters come from the tutorial's own config, not from literals here:
    # the figure has to show the surface the tutorial actually samples, and a
    # second copy of these numbers would drift the first time either is tuned.
    cfg = _tutorial_config("01_rens_toy")
    backend = create_rens_toy(
        **{
            k: v
            for k, v in cfg["backend"].items()
            if k not in ("type", "periodic")
        }
    )
    pressures = list(cfg["ensemble"]["pressure"])

    # d on the horizontal axis, a on the vertical: the irreducible region
    # d <= a/2 is then the *upper-left* wedge, which is how the paper orients
    # it.  (With a horizontal it lands lower-right and reads inverted.)
    n = 260
    a_grid = np.linspace(0.5, 4.0, n)
    d_grid = np.linspace(0.0, 2.0, n)
    D, A = np.meshgrid(d_grid, a_grid)

    types = jnp.zeros(2, dtype=int)
    U = np.empty(A.shape)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            a, d = float(A[i, j]), float(D[i, j])
            pos = jnp.array([[0.0, 0.0, 0.0], [float(d), 0.0, 0.0]])
            cell = jnp.diag(jnp.array([float(a), 1.0, 1.0]))
            U[i, j] = float(backend(pos, types, cell).energy)

    # Irreducible wedge: d and a - d describe the same configuration under
    # periodicity, so everything with d > a/2 is a mirror image.  Masking it
    # is what makes the surface readable -- and matches Figure 3a of the paper.
    wedge = D <= A / 2.0

    fig, axes = plt.subplots(
        1, 3, figsize=(12, 4.4), sharey=True, constrained_layout=True
    )
    for ax, P in zip(axes, pressures, strict=False):
        H = np.where(wedge, U + P * A, np.nan)
        im = ax.pcolormesh(D, A, H, cmap="viridis", shading="auto")
        ax.contour(D, A, H, levels=16, colors="w", linewidths=0.4, alpha=0.6)
        ax.plot(a_grid / 2.0, a_grid, color="k", lw=1.0)
        ax.set_xlabel("separation  $d$")
        ax.set_xlim(d_grid[0], d_grid[-1])
        ax.set_ylim(a_grid[0], a_grid[-1])
        ax.set_title(f"$P = {P}$", fontsize=10)
    axes[0].set_ylabel("box length  $a$")
    fig.colorbar(im, ax=axes, label="enthalpy  $H = U + P a$", shrink=0.9)
    fig.suptitle(
        "irreducible wedge of the toy-model enthalpy surface "
        "($d \\leq a/2$)",
        fontsize=11,
    )

    _TUTFIG.mkdir(parents=True, exist_ok=True)
    out = _TUTFIG / "rens_toy_surface.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}")


def fig_pkg_treemap() -> None:
    """Produce interactive HTML + static SVG treemaps of ``src/jaxrens/``."""
    src_root = Path(__file__).resolve().parents[3] / "src" / "jaxrens"
    if not src_root.is_dir():
        print(f"skipping treemap: {src_root} not found")
        return
    rows = _walk_pkg(src_root)
    _fig_pkg_treemap_html(rows)
    _fig_pkg_treemap_svg(rows)
    old_png = FIGDIR / "pkg_treemap.png"
    if old_png.exists():
        old_png.unlink()
        print(f"  removed {old_png}")


if __name__ == "__main__":
    fig_ns_prior_mass()
    fig_mwg_acceptance()
    fig_ensemble_tilt()
    fig_rens_acceptance()
    fig_pytree_shapes()
    fig_pkg_treemap()
    fig_gauss2d_surface()
    fig_rens_toy_surface()
    print(f"wrote figures to {FIGDIR}")
