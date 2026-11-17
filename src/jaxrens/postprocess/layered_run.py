"""Structural observables of a layered run on every NS point.

The functions of :mod:`jaxrens.postprocess.observables` act on one
configuration.  Here they are evaluated on every point of a run, the dead
points (from the trajectory) followed by the live walkers (from the
checkpoint), giving per-point arrays for :class:`Monitor` estimators::

    coll, obs = load_layered_observables("output")
    mon = coll[0]
    mon.expectation(obs[0]["gallery_width"], T)
    mon.class_probability(obs[0]["n_empty"] == 1, T)

A trajectory usually keeps one dead point every ``output.traj_interval``
iterations.  Frame ``info['iter']`` and ``info['cull_rank']`` give the
dead point of each frame, ``iter * n_cull + cull_rank``; with
``fill="hold"`` each frame's values stand for the dead points up to the
next frame, with ``fill="none"`` a frame is required for every dead point.

- LayeredParams / layered_params: host, intercalant and in-plane repeats,
  from the layered moves of the run config
- dead_point_rows / fill_dead_points: frames to dead points
- configuration_observables: every observable of one configuration
- layered_observables: every observable on every point of one replica
- load_layered_observables: all replicas of an output directory, cached
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from jaxrens.postprocess.layers import layer_geometry
from jaxrens.postprocess.observables import (
    cell_observables,
    layer_observables,
    li_order_observables,
    registry_observables,
)

logger = logging.getLogger(__name__)

OBSERVABLES_VERSION = 1
"""Bump when an observable changes, so cached values are recomputed."""

_LAYERED_MOVES = ("gallery_hop", "layer_slide", "stacking_shift")


@dataclass(frozen=True)
class LayeredParams:
    """What the layered observables need to know about the system.

    Attributes:
        host_species: Elements forming the sheets.
        intercalant_species: Elements in the galleries.
        in_plane_repeats: Host primitive cells along a and b.
    """

    host_species: tuple[str, ...]
    intercalant_species: tuple[str, ...]
    in_plane_repeats: tuple[int, int] = (3, 3)


def layered_params(config: dict) -> LayeredParams:
    """Read the layered parameters from a run config.

    ``host_species`` and ``in_plane_repeats`` come from the layered moves
    (``gallery_hop``, ``layer_slide``, ``stacking_shift``), which share
    them.  The intercalant is the ``species`` of ``gallery_hop``, or every
    non-host element of ``init.start_species`` without one.

    Args:
        config: Parsed run config, e.g. ``<prefix>.config.snapshot.yaml``.

    Raises:
        ValueError: If the config has no layered move, or no intercalant
            can be determined.
    """
    moves = [
        m for m in config.get("moves") or [] if m.get("type") in _LAYERED_MOVES
    ]
    if not moves:
        raise ValueError(
            "the run config has no layered move (gallery_hop, layer_slide, "
            "stacking_shift); pass LayeredParams explicitly."
        )
    first = moves[0]
    host = _as_tuple(first.get("host_species") or ("C",))
    repeats = tuple(int(m) for m in first.get("in_plane_repeats") or (3, 3))
    hops = [m for m in moves if m.get("type") == "gallery_hop"]
    if hops:
        intercalant = tuple(
            dict.fromkeys(s for m in hops for s in _as_tuple(m["species"]))
        )
    else:
        intercalant = tuple(s for s in _start_species(config) if s not in host)
    if not intercalant:
        raise ValueError(
            "cannot tell the intercalant from the run config (no "
            "gallery_hop move and no non-host species in init); pass "
            "LayeredParams explicitly."
        )
    return LayeredParams(host, intercalant, repeats)


def _as_tuple(species) -> tuple[str, ...]:
    return (species,) if isinstance(species, str) else tuple(species)


def _start_species(config: dict) -> list[str]:
    """Element symbols of ``init.start_species`` (``"Z count ..."`` pairs
    or symbols), or [] if absent."""
    from ase.data import chemical_symbols

    spec = (config.get("init") or {}).get("start_species")
    if not spec:
        return []
    items = str(spec).replace(",", " ").split()
    symbols = []
    for item in items[::2]:
        symbols.append(chemical_symbols[int(item)] if item.isdigit() else item)
    return symbols


def dead_point_rows(iters, cull_ranks, n_cull: int) -> np.ndarray:
    """Dead point of every trajectory frame, ``iter * n_cull + cull_rank``.

    Args:
        iters: ``info['iter']`` of every frame.
        cull_ranks: ``info['cull_rank']`` of every frame, or None when the
            run culled one walker per iteration.
        n_cull: Walkers culled per iteration.
    """
    iters = np.asarray(iters, dtype=int)
    ranks = (
        np.zeros_like(iters)
        if cull_ranks is None
        else np.asarray(cull_ranks, dtype=int)
    )
    return iters * n_cull + ranks


def fill_dead_points(values, rows, n_dead: int, fill: str = "hold"):
    """Values of the trajectory frames on every dead point.

    Args:
        values: Frame values, ``(n_frames, ...)``.
        rows: Dead point of every frame (:func:`dead_point_rows`).
        n_dead: Number of dead points of the run.
        fill: ``"hold"``: each frame's value stands for the dead points up
            to the next frame (points before the first frame take its
            value).  ``"none"``: every dead point needs its own frame.

    Returns:
        ``(n_dead, ...)`` values.

    Raises:
        ValueError: If a frame points outside the run, two frames point to
            the same dead point, or ``fill="none"`` and a dead point has no
            frame.
    """
    values = np.asarray(values)
    rows = np.asarray(rows, dtype=int)
    if fill not in ("hold", "none"):
        raise ValueError(f"fill must be 'hold' or 'none', not {fill!r}.")
    if len(rows) == 0:
        raise ValueError("the trajectory has no frames.")
    if rows.min() < 0 or rows.max() >= n_dead:
        raise ValueError(
            f"trajectory frames point to dead points [{rows.min()}, "
            f"{rows.max()}] but the run has {n_dead}: trajectory and "
            "checkpoint do not belong to the same run."
        )
    order = np.argsort(rows, kind="stable")
    rows, values = rows[order], values[order]
    if (np.diff(rows) == 0).any():
        raise ValueError("two trajectory frames point to the same dead point.")
    if fill == "none":
        if len(rows) != n_dead:
            raise ValueError(
                f"the trajectory has {len(rows)} of the {n_dead} dead points "
                "(traj_interval > 1?); use fill='hold' to hold each frame "
                "over the points up to the next one."
            )
        return values
    pos = np.searchsorted(rows, np.arange(n_dead), side="right") - 1
    return values[np.clip(pos, 0, None)]


def configuration_observables(
    positions,
    cell,
    symbols: Sequence[str],
    params: LayeredParams,
    *,
    n_sheets: int | None = None,
    sheet_size: int | None = None,
) -> dict[str, Any]:
    """Every structural observable of one configuration.

    Merges :func:`cell_observables`, :func:`layer_observables`,
    :func:`registry_observables` and :func:`li_order_observables`.
    Raises like :func:`jaxrens.postprocess.layers.layer_geometry` if the
    sheets are not the expected ones.
    """
    symbols = np.asarray(symbols)
    host = np.isin(symbols, params.host_species)
    intercalant = np.isin(symbols, params.intercalant_species)
    repeats = params.in_plane_repeats
    geo = layer_geometry(
        positions, cell, host, n_sheets=n_sheets, sheet_size=sheet_size
    )
    return {
        **cell_observables(positions, cell, in_plane_repeats=repeats),
        **layer_observables(geo, intercalant),
        **registry_observables(
            positions, cell, geo, intercalant, in_plane_repeats=repeats
        ),
        **li_order_observables(
            positions, cell, geo, intercalant, in_plane_repeats=repeats
        ),
    }


def _evaluate(configs, params, layout, where):
    """Observables of a list of ``(positions, cell, symbols)``, stacked
    per name; ``where(k)`` names configuration ``k`` in error messages."""
    rows = []
    for k, (pos, cell, sym) in enumerate(configs):
        try:
            rows.append(
                configuration_observables(pos, cell, sym, params, **layout)
            )
        except ValueError as err:
            raise ValueError(f"{where(k)}: {err}") from err
    return {
        name: np.stack([np.asarray(r[name], dtype=float) for r in rows])
        for name in rows[0]
    }


def _live_configs(monitor) -> list | None:
    """Live walkers as ``(positions, cell, symbols)``, or None if the
    Monitor does not carry their configurations."""
    pos, types, cells = (
        monitor.live_positions,
        monitor.live_types,
        monitor.live_cells,
    )
    if pos is None or types is None or cells is None or not monitor.symbol_map:
        return None
    return [
        (pos[k], cells[k], [monitor.symbol_map[int(t)] for t in types[k]])
        for k in range(len(pos))
    ]


def layered_observables(
    monitor, frames, params: LayeredParams, *, fill: str = "hold"
) -> dict[str, np.ndarray]:
    """Every structural observable on every point of one replica.

    Args:
        monitor: The replica's :class:`Monitor`.
        frames: Its dead-point trajectory, ``ase.Atoms`` carrying
            ``info['iter']`` (and ``info['cull_rank']`` when n_cull > 1).
        params: System parameters (:func:`layered_params`).
        fill: How to cover the dead points without a frame, see
            :func:`fill_dead_points`.

    Returns:
        ``name -> (n_points, ...)``: the dead points, then the live
        walkers.  If the Monitor carries no live configurations, a warning
        is logged and the arrays cover the dead points only; the Monitor
        then averages over the dead points alone.
    """
    if not frames:
        raise ValueError("the trajectory has no frames.")
    first = frames[0]
    geo = layer_geometry(
        first.positions,
        first.cell[:],
        np.isin(first.get_chemical_symbols(), params.host_species),
    )
    layout = {"n_sheets": geo.n_sheets, "sheet_size": geo.sheet_size}

    dead = _evaluate(
        [(a.positions, a.cell[:], a.get_chemical_symbols()) for a in frames],
        params,
        layout,
        lambda k: f"trajectory frame iter={frames[k].info.get('iter')}",
    )
    has_rank = any("cull_rank" in a.info for a in frames)
    rows = dead_point_rows(
        [a.info["iter"] for a in frames],
        [a.info.get("cull_rank", 0) for a in frames] if has_rank else None,
        monitor.n_cull,
    )
    dead = {
        name: fill_dead_points(v, rows, monitor.n_dead, fill)
        for name, v in dead.items()
    }

    if monitor.n_points == monitor.n_dead:
        return dead
    live_configs = _live_configs(monitor)
    if live_configs is None:
        logger.warning(
            "%s: no live-walker configurations in the checkpoint; the "
            "layered observables cover the dead points only.",
            monitor.label,
        )
        return dead
    live = _evaluate(
        live_configs, params, layout, lambda k: f"live walker {k}"
    )
    return {name: np.concatenate([dead[name], live[name]]) for name in dead}


def load_layered_observables(
    path,
    *,
    prefix: str | None = None,
    config=None,
    params: LayeredParams | None = None,
    fill: str = "hold",
    cache: bool = True,
):
    """Monitors and layered observables of every replica of a run.

    Args:
        path: NS output directory.
        prefix: ``output.out_file_prefix``; from the config if None.
        config: Run config file; ``<prefix>.config.snapshot.yaml`` in
            ``path`` (the only one there when ``prefix`` is None), else the
            ``config.yaml`` that :class:`MonitorCollection` discovers.
        params: System parameters; read from the config if None.
        fill: See :func:`fill_dead_points`.
        cache: Reuse ``<traj>.obs.npz`` next to each trajectory when it
            matches the run, and write it otherwise.

    Returns:
        ``(collection, observables)``: the :class:`MonitorCollection` and,
        per replica in the same order, the dict of
        :func:`layered_observables`.
    """
    from jaxrens.postprocess.collection import MonitorCollection, _load_yaml

    path = Path(path)
    cfg_path = _find_config(path, prefix, config)
    cfg = _load_yaml(cfg_path) if cfg_path is not None else {}
    if prefix is None:
        prefix = (cfg.get("output") or {}).get("out_file_prefix") or "ns"
    if params is None:
        params = layered_params(cfg)
    coll = MonitorCollection.from_multi_run_directory(
        path, prefix=prefix, config=cfg_path
    )
    observables = []
    for i, mon in enumerate(coll):
        traj = _traj_file(path, prefix, i, len(coll))
        observables.append(_cached(mon, traj, params, fill, cache))
    return coll, observables


def _find_config(path: Path, prefix, config) -> Path | None:
    if config is not None:
        return Path(config)
    if prefix is not None:
        snap = path / f"{prefix}.config.snapshot.yaml"
        return snap if snap.is_file() else None
    snaps = sorted(path.glob("*.config.snapshot.yaml"))
    if len(snaps) == 1:
        return snaps[0]
    return None


def _traj_file(path: Path, prefix: str, i: int, n_runs: int) -> Path:
    single = path / f"{prefix}.traj.extxyz"
    if n_runs == 1 and single.exists():
        return single
    return path / f"{prefix}.run{i:02d}.traj.extxyz"


def _fingerprint(mon, traj: Path, params: LayeredParams, fill: str) -> str:
    stat = traj.stat()
    return json.dumps(
        {
            "version": OBSERVABLES_VERSION,
            "params": [
                list(params.host_species),
                list(params.intercalant_species),
                list(params.in_plane_repeats),
            ],
            "fill": fill,
            "traj": [stat.st_size, stat.st_mtime_ns],
            "n_dead": mon.n_dead,
            "live_energies": np.asarray(mon.live_energies).round(10).tolist(),
        },
        sort_keys=True,
    )


def _cached(mon, traj: Path, params, fill, cache) -> dict[str, np.ndarray]:
    from ase.io import read

    if not traj.exists():
        raise FileNotFoundError(f"trajectory not found: {traj}")
    cache_path = traj.with_name(traj.name.removesuffix(".extxyz") + ".obs.npz")
    key = _fingerprint(mon, traj, params, fill)
    if cache and cache_path.exists():
        with np.load(cache_path) as data:
            if str(data["__key__"]) == key:
                return {k: data[k] for k in data.files if k != "__key__"}
    frames = read(str(traj), index=":")
    obs = layered_observables(mon, frames, params, fill=fill)
    if cache:
        np.savez(cache_path, __key__=np.asarray(key), **obs)
    return obs
