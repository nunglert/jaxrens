"""Generate the tutorial energy-surface figures used by the docs.

Run from the repo root:
    python docs/_static/figures/generate_tutorials.py

These show the reader *what is being sampled* before any NS output is
discussed.  All are computed straight from the backend, so they cannot drift
from the model the tutorial configs actually run; the walker/trajectory
overlays in ``fig_gauss2d_walkers`` and ``fig_gauss2d_trajectory`` are read
from a completed tutorial run and are skipped if that run's output is not
present.
"""

from __future__ import annotations

import numpy as np

from _common import FIGDIR, GRID_BATCH

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


def _gauss2d_landscape() -> tuple[np.ndarray, np.ndarray, np.ndarray, list, float]:
    """Energy grid + Gaussian centres for the 00_gaussian_2d backend.

    Shared by the landscape-only figure and the walkers/trajectory figure so
    both draw an identical background instead of two independently
    recomputed (and potentially drifting) copies.
    """
    import jax
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
    pts = jnp.stack(
        [jnp.asarray(X.ravel()), jnp.asarray(Y.ravel()), jnp.zeros(X.size)],
        axis=-1,
    )
    cell = jnp.eye(3) * 8.0
    types = jnp.zeros(1, dtype=int)

    # One call per grid point is dominated by dispatch overhead, not the
    # (tiny) energy evaluation itself -- batching points into chunks trades
    # that overhead for one traced call per chunk instead of per point.
    # jax.lax.map(..., batch_size=...) vmaps within each chunk and scans
    # across chunks, so peak memory is one chunk's intermediates rather than
    # all n*n at once.
    def _energy(p: jnp.ndarray) -> jnp.ndarray:
        return backend(p[None, :], types, cell).energy

    E = np.array(
        jax.lax.map(_energy, pts, batch_size=GRID_BATCH)
    ).reshape(X.shape)
    return X, Y, E, centers, lim


def fig_gauss2d_surface() -> None:
    """Gaussian-mixture landscape alone -- what NS is asked to search.

    Deliberately has no walker/trajectory overlay: this figure introduces the
    *problem*, before any NS output exists to show.  See
    ``fig_gauss2d_walkers`` for the population/trajectory figure.
    """
    import matplotlib.pyplot as plt

    X, Y, E, centers, lim = _gauss2d_landscape()

    fig, ax = plt.subplots(figsize=(5.6, 4.6), constrained_layout=True)
    im = ax.pcolormesh(X, Y, E, cmap="viridis", shading="auto")
    ax.contour(X, Y, E, levels=14, colors="w", linewidths=0.4, alpha=0.6)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    fig.colorbar(im, ax=ax, label="energy  $E(x, y)$", shrink=0.9)

    ax.scatter(
        [c[0] for c in centers],
        [c[1] for c in centers],
        marker="x",
        c="r",
        s=45,
        lw=1.4,
        label="Gaussian centres",
    )
    ax.legend(loc="lower left", fontsize=8, framealpha=0.85)
    ax.set_title(
        "the landscape: four basins, one of them merged (deeper)", fontsize=10
    )

    _TUTFIG.mkdir(parents=True, exist_ok=True)
    out = _TUTFIG / "gauss2d_surface.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}")


def fig_gauss2d_walkers() -> None:
    """Walker-population contraction as small multiples, one panel per snapshot.

    Each panel is the landscape with the live population overlaid at one
    ``output.snapshot_interval`` dump, so the contraction reads as a strip of
    stills instead of one overlay crowded with every snapshot at once.
    Falls back to a single placeholder panel when the tutorial hasn't
    actually been run, since the source data only exists then.
    """
    import matplotlib.pyplot as plt
    from ase.io import read

    X, Y, E, _centers, lim = _gauss2d_landscape()
    out_dir = _TUT / "00_gaussian_2d" / "output"
    snaps = sorted(
        out_dir.glob("*.traj.snap.*.extxyz"),
        key=lambda q: int(q.name.split(".snap.")[1].split(".")[0]),
    )

    # Evenly spaced across the whole run (not just first/mid/last), so the
    # strip reads as a sequence rather than three snapshots.
    n_panels = min(6, len(snaps)) if snaps else 1
    picks = (
        [snaps[i] for i in np.linspace(0, len(snaps) - 1, n_panels).round().astype(int)]
        if snaps
        else []
    )

    ncols = min(6, n_panels)
    nrows = -(-n_panels // ncols)  # ceil division
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.4 * ncols, 3.1 * nrows),
        constrained_layout=True,
        squeeze=False,
    )
    axes_flat = axes.ravel()

    for ax in axes_flat:
        ax.pcolormesh(X, Y, E, cmap="viridis", shading="auto")
        ax.contour(X, Y, E, levels=10, colors="w", linewidths=0.3, alpha=0.5)
        ax.set_aspect("equal")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xticks([])
        ax.set_yticks([])

    if picks:
        for ax, q in zip(axes_flat, picks, strict=False):
            it = int(q.name.split(".snap.")[1].split(".")[0])
            pos = np.array([a.get_positions()[0] for a in read(q, index=":")])
            ax.scatter(
                pos[:, 0],
                pos[:, 1],
                s=6,
                color="#ffd166",
                edgecolors="k",
                linewidths=0.15,
            )
            ax.set_title(f"iter {it}", fontsize=9)
    else:
        axes_flat[0].set_title(
            "(run the tutorial to overlay walkers)", fontsize=9
        )
    for ax in axes_flat[n_panels:]:
        ax.axis("off")

    fig.suptitle(
        "walker population contracting onto the deepest basin", fontsize=11
    )

    _TUTFIG.mkdir(parents=True, exist_ok=True)
    out = _TUTFIG / "gauss2d_walkers.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}")


def fig_gauss2d_trajectory() -> None:
    """Full NS trajectory: every culled point, coloured by iteration.

    Reads the run's main ``*.traj.extxyz`` (excluding the periodic
    ``*.traj.snap.*`` dumps handled by ``fig_gauss2d_walkers``) -- one dead
    point per iteration, since ``traj_interval: 1`` -- so this is the
    complete history of what NS explored and discarded, not a handful of
    snapshots.  Falls back to a placeholder title when the tutorial hasn't
    actually been run, since the source data only exists then.
    """
    import matplotlib.pyplot as plt
    from ase.io import read

    X, Y, E, _centers, lim = _gauss2d_landscape()
    out_dir = _TUT / "00_gaussian_2d" / "output"
    # The dead-point trajectory shares the "*.traj*" stem with the snapshot
    # files, so exclude those explicitly rather than trying to glob past them.
    traj_files = [
        q for q in out_dir.glob("*.traj.extxyz") if ".snap." not in q.name
    ]

    fig, ax = plt.subplots(figsize=(5.6, 4.6), constrained_layout=True)
    ax.pcolormesh(X, Y, E, cmap="viridis", shading="auto")
    ax.contour(X, Y, E, levels=14, colors="w", linewidths=0.4, alpha=0.6)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    if traj_files:
        frames = read(traj_files[0], index=":")
        pos = np.array([a.get_positions()[0] for a in frames])
        iters = np.array(
            [a.info.get("iter", i) for i, a in enumerate(frames)]
        )
        sc = ax.scatter(
            pos[:, 0], pos[:, 1], s=5, c=iters, cmap="plasma", linewidths=0
        )
        fig.colorbar(sc, ax=ax, label="NS iteration", shrink=0.9)
        ax.set_title("NS trajectory: every culled point, in order", fontsize=10)
    else:
        ax.set_title(
            "(run the tutorial to overlay the trajectory)", fontsize=10
        )

    _TUTFIG.mkdir(parents=True, exist_ok=True)
    out = _TUTFIG / "gauss2d_trajectory.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}")


def fig_rens_toy_surface() -> None:
    """Toy-model enthalpy surface H(a, d), irreducible wedge, per pressure."""
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt

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

    # d on the horizontal axis, a on the vertical: the valid region d <= a is
    # then the *upper-left* wedge, which is how the paper orients it.  (With
    # a horizontal it lands lower-right and reads inverted.)
    n = 260
    amax  = 3.0
    a_grid = np.linspace(0.0, amax, n)
    d_grid = np.linspace(0.0, amax, n)
    D, A = np.meshgrid(d_grid, a_grid)

    types = jnp.zeros(2, dtype=int)

    # n*n = 67600 single-pair evaluations -- batch them in chunks instead of
    # one vmap over the whole grid, which OOMs (each point's periodic-image
    # sum is small, but n*n of them live at once under a plain vmap).
    # jax.lax.map(..., batch_size=...) vmaps within each chunk and scans
    # across chunks, so peak memory is one chunk's worth of intermediates.
    def _energy(ad: jnp.ndarray) -> jnp.ndarray:
        a, d = ad[0], ad[1]
        pos = jnp.array([[0.0, 0.0, 0.0], [d, 0.0, 0.0]])
        cell = jnp.diag(jnp.array([a, 1.0, 1.0]))
        return backend(pos, types, cell).energy

    ad_grid = jnp.stack(
        [jnp.asarray(A.ravel()), jnp.asarray(D.ravel())], axis=-1
    )
    U = np.array(
        jax.lax.map(_energy, ad_grid, batch_size=GRID_BATCH)
    ).reshape(A.shape)

    # Mask off d > a: a separation larger than the box length isn't a valid
    # configuration, so masking it is what makes the surface readable.
    wedge = D <= A

    # The corner near a -> 0 diverges (the periodic images pile up on top of
    # each other there), which would otherwise swamp the colour scale and
    # flatten every other feature to a single shade.  Capping at vmax makes
    # the interesting structure visible; extend="max" on the colorbar marks
    # that values above it are clipped rather than genuinely uniform.
    vmax = 10.0
    levels = np.linspace(0.0, vmax, 10)

    fig, axes = plt.subplots(
        1, 3, figsize=(12, 4.4), sharey=True, constrained_layout=True
    )
    for ax, P in zip(axes, pressures, strict=False):
        # Mask to the wedge *before* baselining: the mirror region (d > a)
        # and the a -> 0 corner take on huge, physically meaningless values
        # that would otherwise drag the baseline far below anything in the
        # valid region.
        H = np.where(wedge, np.array(U + P * A), np.nan)
        H -= np.nanmin(H)
        im = ax.pcolormesh(
            D, A, H, cmap="viridis", shading="auto", vmin=0.0, vmax=vmax
        )
        ax.contour(D, A, H, levels=levels, colors="w", linewidths=0.4, alpha=0.6)
        ax.plot(a_grid, a_grid, color="k", lw=1.0)
        ax.set_xlabel("separation  $d$")
        ax.set_xlim(d_grid[0], d_grid[-1])
        ax.set_ylim(a_grid[0], a_grid[-1])
        ax.set_title(f"$P = {P}$", fontsize=10)
    axes[0].set_ylabel("box length  $a$")
    fig.colorbar(
        im,
        ax=axes,
        label="enthalpy  $H = U + P a$",
        shrink=0.9,
        extend="max",
    )
    fig.suptitle(
        "irreducible wedge of the toy-model enthalpy surface ($d \\leq a$)",
        fontsize=11,
    )

    _TUTFIG.mkdir(parents=True, exist_ok=True)
    out = _TUTFIG / "rens_toy_surface.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> None:
    fig_gauss2d_surface()
    fig_gauss2d_walkers()
    fig_gauss2d_trajectory()
    fig_rens_toy_surface()


if __name__ == "__main__":
    main()
    print(f"wrote tutorial figures to {FIGDIR}")
