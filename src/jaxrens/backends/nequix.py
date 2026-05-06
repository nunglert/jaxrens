"""Nequix energy backend.

Wraps a pre-trained Nequix model (equinox module, loaded via
``nequix.calculator.from_pretrained`` or ``nequix.model.load_model``)
behind the EnergyBackend interface. JIT-compatible neighbor finding via
the shared supercell helper (same as the MACE backend).

nequix is an optional dependency. All imports are guarded.

Energy-only path
----------------
``nequix.model.Nequix.__call__`` wraps the energy core in
``eqx.filter_grad`` (for forces) and additionally in a strain-perturbation
grad (for stress). Nested sampling needs only the scalar energy, so the
backend calls ``model.node_energies`` directly, bypassing both backward
passes. This is typically ~3× cheaper than calling the full
``model(graph)`` forward per evaluation.

Non-periodic systems
--------------------
For non-periodic systems (cell == 0), pass ``supercell_trafo=(1, 1, 1)``.
With larger supercell transforms and a zero cell, the supercell images
collapse onto the origin and real edges are double-counted.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import jax.numpy as jnp

from jaxrens.backends._graph_neighbors import (
    _make_image_offsets,
    _supercell_edges,
)

logger = logging.getLogger(__name__)

_NEQUIX_AVAILABLE = False
_NEQUIX_IMPORT_ERROR = ""
try:
    import equinox as eqx
    import jraph
    from nequix.model import Nequix, load_model

    _NEQUIX_AVAILABLE = True
except ImportError as exc:
    _NEQUIX_IMPORT_ERROR = str(exc)


def _require_nequix() -> None:
    if not _NEQUIX_AVAILABLE:
        raise ImportError(
            f"nequix is required for the Nequix backend but not installed: "
            f"{_NEQUIX_IMPORT_ERROR}"
        )


def is_available() -> bool:
    return _NEQUIX_AVAILABLE


# ---------------------------------------------------------------------------
# Energy-only compute path
# ---------------------------------------------------------------------------


def _compute_energy_only(
    model: Any,
    positions: jnp.ndarray,
    species: jnp.ndarray,
    senders: jnp.ndarray,
    receivers: jnp.ndarray,
    cart_shifts: jnp.ndarray,
    n_atoms: int,
) -> jnp.ndarray:
    """Compute total potential energy via ``model.node_energies``.

    Mirrors ``Nequix.__call__``'s energy path at
    ``reference_codes/nequix/nequix/model.py:423-492`` but skips the
    ``eqx.filter_grad`` wrappers used there for forces and strain stress.

    A single ghost node (index ``n_atoms``) absorbs the overflow edges
    produced by ``_supercell_edges``. The ghost is padded with
    ``species=0`` and ``position=0`` so that all indexing remains in-range;
    its node energy contribution is discarded by slicing before the sum.

    Cartesian shifts from ``_supercell_edges`` are added directly to the
    displacement — no fractional-shift + cell multiply needed.
    """
    padded_pos = jnp.zeros((n_atoms + 1, 3), dtype=positions.dtype)
    padded_pos = padded_pos.at[:n_atoms].set(positions)

    padded_species = jnp.zeros(n_atoms + 1, dtype=species.dtype)
    padded_species = padded_species.at[:n_atoms].set(species)

    displacements = padded_pos[senders] - padded_pos[receivers] + cart_shifts

    node_energies = model.node_energies(
        displacements, padded_species, senders, receivers,
    )  # (n_atoms + 1, 1)

    return jnp.sum(node_energies[:n_atoms, 0])


# ---------------------------------------------------------------------------
# NequixBackend
# ---------------------------------------------------------------------------


class NequixBackend:
    """Nequix energy backend for jaxrens.

    Satisfies the ``EnergyBackend`` protocol via duck typing: scalar energy
    from ``(positions, species, cell, max_neighbors)`` plus a neighbor-count
    report for the outer-loop bucket dispatch.

    The equinox ``Nequix`` module lives on the instance and is closed over
    by ``__call__``. All JAX arrays passed in must be concrete / traced
    (no eager numpy), so the backend is safe under ``jax.jit`` and
    ``jax.vmap``.
    """

    def __init__(
        self,
        model: Any,
        config: dict,
        supercell_trafo: tuple[int, int, int] = (2, 2, 2),
    ):
        self.r_cutoff = float(config["cutoff"])
        # nequix stores atomic_numbers in sorted order; atomic_numbers_to_indices
        # maps Z -> 0-indexed slot. We expose the sorted list for downstream.
        atomic_numbers = sorted(config["atomic_numbers"])
        self.atomic_numbers = list(atomic_numbers)
        self.num_species = len(atomic_numbers)
        self._config = config
        self._model = model
        self.sc_a, self.sc_b, self.sc_c = supercell_trafo
        self._image_offsets = jnp.array(
            _make_image_offsets(self.sc_a, self.sc_b, self.sc_c),
            dtype=jnp.int32,
        )

    def __call__(
        self,
        positions: jnp.ndarray,
        species: jnp.ndarray,
        cell: jnp.ndarray,
        max_neighbors: int = 50,
        ensemble_params: dict[str, Any] | None = None,
    ) -> tuple[jnp.ndarray, int, bool]:
        n_atoms = positions.shape[0]
        max_edges = n_atoms * max_neighbors

        senders, receivers, cart_shifts, n_actual, overflow = _supercell_edges(
            positions, cell, self.r_cutoff, max_edges, self._image_offsets,
        )

        energy = _compute_energy_only(
            self._model, positions, species,
            senders, receivers, cart_shifts, n_atoms,
        )

        is_real = senders < n_atoms
        real_senders = jnp.where(is_real, senders, 0)
        counts = jnp.zeros(n_atoms, dtype=jnp.int32).at[real_senders].add(
            is_real.astype(jnp.int32),
        )
        neighbor_count = jnp.max(counts)

        return energy, neighbor_count, overflow


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _resolve_model_path(model_name_or_path: str) -> str:
    """Accept either a bundled-model name or a path to a .nqx file.

    Bundled names are defined in ``NequixCalculator.URLS`` (e.g.
    ``nequix-mp-1``, ``nequix-oam-1``) and are auto-downloaded to
    ``~/.cache/nequix/models/``. Paths are used as-is.
    """
    _require_nequix()
    from nequix.calculator import NequixCalculator

    p = Path(model_name_or_path).expanduser()
    if p.exists():
        return str(p)
    if model_name_or_path in NequixCalculator.URLS:
        from nequix.calculator import from_pretrained

        # from_pretrained downloads (if needed) and returns (model, config),
        # but we want just the path. Call urlretrieve directly.
        import urllib.request

        base = Path("~/.cache/nequix/models/").expanduser()
        base.mkdir(parents=True, exist_ok=True)
        cache_path = base / f"{model_name_or_path}.nqx"
        if not cache_path.exists():
            logger.info(
                "Downloading nequix model %s to %s", model_name_or_path, cache_path,
            )
            urllib.request.urlretrieve(
                NequixCalculator.URLS[model_name_or_path], cache_path,
            )
        return str(cache_path)
    raise FileNotFoundError(
        f"Nequix model {model_name_or_path!r} is neither an existing path "
        f"nor a known bundled name ({sorted(NequixCalculator.URLS.keys())})"
    )


def create_nequix(
    model_name_or_path: str | None = None,
    supercell_trafo: tuple[int, int, int] = (2, 2, 2),
    **kwargs: Any,
) -> NequixBackend:
    """Create a Nequix energy backend from a bundled name or a .nqx path.

    Args:
        model_name_or_path: Either a bundled model name (e.g. ``nequix-mp-1``,
            auto-downloaded on first use) or a path to a local .nqx file.
        supercell_trafo: ``(sc_a, sc_b, sc_c)`` supercell expansion for
            neighbor finding. Must satisfy ``min(cell_diag * sc) >= 2 * r_cutoff``
            to capture all true neighbors. Use ``(1, 1, 1)`` for non-periodic
            (zero-cell) systems to avoid double-counting.

    Returns:
        NequixBackend instance.
    """
    _require_nequix()
    if model_name_or_path is None:
        raise ValueError("model_name_or_path is required for the Nequix backend")

    path = _resolve_model_path(model_name_or_path)
    model, config = load_model(path, kernel=False)

    logger.info(
        "Nequix backend created: r_cutoff=%.2f, %d species, supercell=%s",
        float(config["cutoff"]),
        len(config["atomic_numbers"]),
        supercell_trafo,
    )

    return NequixBackend(model=model, config=config, supercell_trafo=supercell_trafo)
