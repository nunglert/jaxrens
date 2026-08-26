"""Generate the interactive/static treemaps of the ``jaxrens`` package layout.

Run from the repo root:
    python docs/_static/figures/generate_treemap.py

Unlike the other figure groups this one is derived from the *live* package
layout (module count, LoC, classes, functions) rather than synthetic data, so
it is also regenerated on every docs build (see ``docs/conf.py``) instead of
staying purely a committed artefact.
"""

from __future__ import annotations

from pathlib import Path

from _common import FIGDIR, count_loc


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
        loc = count_loc(py_file)
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
    import matplotlib.pyplot as plt
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


def main() -> None:
    fig_pkg_treemap()


if __name__ == "__main__":
    main()
    print(f"wrote treemap figures to {FIGDIR}")
