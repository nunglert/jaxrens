"""``jaxrens analyze <checkpoint>`` — a thermodynamic observable vs temperature.

Unlike ``jaxrens plot`` (one artefact file -> one plot), a thermodynamic
observable needs both a run's checkpoint (``n_live``, live energies, log
evidence) and its ``.energies`` log (dead-point energies) — exactly what
``Monitor.from_directory`` assembles from a directory + prefix.  So this
dispatches on a checkpoint file, recovers ``<prefix>`` from its name, and
loads the sibling files from the same directory::

    <prefix>.checkpoint.h5 / <prefix>.final.checkpoint.h5, plus --observable
        heat_capacity        Cv(T)          (default)
        partition_function   log Z(T)
        free_energy          F(T)

The primary output is data, not a picture — ``--format csv`` (default, fixed-
width aligned columns) or ``--format json`` (nests naturally for an
observable that is not one scalar per T) — ready for a notebook or another
run's comparison without a detour through a PNG.  Pass ``--plot`` to
additionally render one, via the same ``plot_*`` functions ``jaxrens plot``
uses elsewhere.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_CHECKPOINT_SUFFIXES = (".final.checkpoint.h5", ".checkpoint.h5")

# name -> (Monitor method name, human title, data column header)
_OBSERVABLES = {
    "heat_capacity": ("heat_capacity", "heat capacity", "Cv"),
    "partition_function": ("partition_function", "log partition function", "log_Z"),
    "free_energy": ("free_energy", "free energy", "F"),
}

_FORMATS = ("csv", "json")

# Fixed-width, right-aligned columns: a CSV meant to be opened in an editor
# and actually read, not just parsed. 8 significant figures is already past
# what Monte Carlo noise in these observables supports, so this is not
# losing precision that matters -- it's dropping the noise of a bare
# float64 repr (17 digits) that swamps the column and buys nothing.
_CSV_COLUMN_WIDTH = 16
_CSV_SIG_FIGS = 8


def _prefix_from_checkpoint(path: Path) -> str:
    """Recover ``output.out_file_prefix`` from a checkpoint filename.

    ``<prefix>.final.checkpoint.h5`` and ``<prefix>.checkpoint.h5`` are the
    two forms ``Monitor.from_directory`` looks for; try the longer suffix
    first since it is a superset of the shorter one.
    """
    name = path.name
    for suffix in _CHECKPOINT_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    raise ValueError(
        f"Unrecognised checkpoint filename {path.name!r}.  Expected one of: "
        + ", ".join(f"<prefix>{s}" for s in _CHECKPOINT_SUFFIXES)
    )


def _default_output(
    input_path: Path, prefix: str, observable: str, suffix: str
) -> Path:
    return input_path.parent / f"{prefix}.{observable}.{suffix}"


def _write_csv(path: Path, T: np.ndarray, column: str, values: np.ndarray) -> None:
    """Fixed-width, right-aligned columns -- readable in a plain editor.

    Only defined for one scalar value per temperature; a non-scalar
    observable (e.g. per-species, or a distribution) does not have a
    sensible fixed-column CSV shape, so use ``--format json`` for those.
    """
    if values.ndim != 1:
        raise ValueError(
            f"{column!r} has shape {values.shape}, not one scalar per "
            f"temperature -- CSV only supports scalar observables. Use "
            f"--format json, which nests arbitrary shapes directly."
        )
    w, p = _CSV_COLUMN_WIDTH, _CSV_SIG_FIGS
    with path.open("w", newline="") as fh:
        fh.write(f"{'T':>{w}},{column:>{w}}\n")
        for t, v in zip(T.tolist(), values.tolist(), strict=True):
            fh.write(f"{t:>{w}.{p}g},{v:>{w}.{p}g}\n")


def _write_json(
    path: Path,
    T: np.ndarray,
    column: str,
    values: np.ndarray,
    *,
    observable: str,
    prefix: str,
    k_b: float,
) -> None:
    """Self-describing and shape-agnostic: ``values.tolist()`` nests however
    many dimensions the observable actually has, so a future non-scalar
    observable (per-species Cv, a distribution per T, ...) needs no format
    change here -- only CSV's fixed-column shape would.
    """
    payload = {
        "observable": observable,
        "column": column,
        "prefix": prefix,
        "k_b": k_b,
        "T": T.tolist(),
        column: values.tolist(),
    }
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")


def analyze_file(
    input_path: Path,
    *,
    observable: str = "heat_capacity",
    t_min: float,
    t_max: float,
    n_t: int = 200,
    k_b: float = 1.0,
    fmt: str = "csv",
    output_path: Path | None = None,
    plot: bool = False,
    plot_path: Path | None = None,
) -> tuple[Path, Path | None]:
    """Load a run's checkpoint and write one thermodynamic observable vs T.

    Args:
        input_path: Path to ``<prefix>.checkpoint.h5`` or
            ``<prefix>.final.checkpoint.h5``.  The sibling ``<prefix>.energies``
            file in the same directory supplies the dead-point energies (see
            ``Monitor.from_directory``).
        observable: One of ``"heat_capacity"``, ``"partition_function"``,
            ``"free_energy"``.
        t_min, t_max: Temperature sweep bounds, in whatever units ``k_b`` is
            calibrated for (reduced units by default, ``k_b=1.0``).  No
            default range is guessed — the right scale depends on the
            backend's energy units, so silently picking one would be more
            likely to mislead than to help.
        n_t: Number of temperature points.
        k_b: Boltzmann constant in the run's energy units per unit of T,
            e.g. ``8.617e-5`` (eV/K) if energies are in eV and T should read
            in Kelvin.  Default ``1.0`` (reduced units).
        fmt: ``"csv"`` (default, fixed-width aligned columns; scalar
            observables only) or ``"json"`` (nests any shape, self-describing).
        output_path: Destination data file.  Defaults to a sibling
            ``<prefix>.<observable>.{csv,json}``.
        plot: Also render a PNG of the same data.
        plot_path: Destination PNG when ``plot`` is set.  Defaults to a
            sibling ``<prefix>.<observable>.png``.

    Returns:
        ``(data_path, png_path)`` — ``png_path`` is ``None`` unless ``plot``
        is set.
    """
    from jaxrens.postprocess.monitor import Monitor

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if observable not in _OBSERVABLES:
        raise ValueError(
            f"Unknown observable {observable!r}.  Choose one of: "
            + ", ".join(_OBSERVABLES)
        )
    if fmt not in _FORMATS:
        raise ValueError(
            f"Unknown format {fmt!r}.  Choose one of: " + ", ".join(_FORMATS)
        )
    prefix = _prefix_from_checkpoint(input_path)
    method_name, title, column = _OBSERVABLES[observable]

    monitor = Monitor.from_directory(input_path.parent, prefix=prefix)
    T = np.linspace(t_min, t_max, n_t)
    values = np.asarray(getattr(monitor, method_name)(T, k_B=k_b))

    if output_path is None:
        output_path = _default_output(input_path, prefix, observable, fmt)
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        _write_csv(output_path, T, column, values)
    else:
        _write_json(
            output_path, T, column, values,
            observable=observable, prefix=prefix, k_b=k_b,
        )

    if not plot:
        return output_path, None

    import matplotlib.pyplot as plt

    from jaxrens.postprocess.plotting import (
        plot_free_energy,
        plot_heat_capacity,
        plot_partition_function,
    )

    plot_fn = {
        "heat_capacity": plot_heat_capacity,
        "partition_function": plot_partition_function,
        "free_energy": plot_free_energy,
    }[observable]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_fn(monitor, T, ax=ax, k_B=k_b)
    ax.set_title(f"{title} · {prefix}", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if plot_path is None:
        plot_path = _default_output(input_path, prefix, observable, "png")
    else:
        plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)

    return output_path, plot_path
