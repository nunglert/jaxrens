"""Nequix energy backend.

Wraps a pre-trained Nequix model (an equinox.Module loaded via
``nequix.model.load_model``) behind the ``EnergyBackend`` interface.
JIT-compatible neighbor finding via the shared supercell helper
(``backends._graph_neighbors``) — same primitive the MACE backend uses.

nequix is an optional dependency. All imports are guarded.

Energy-only path
----------------
``Nequix.__call__`` wraps the energy core in ``eqx.filter_grad`` (for
forces) and additionally in a strain-perturbation grad (for stress).
Nested sampling needs only the scalar energy, so the backend calls
``model.node_energies`` directly, bypassing both backward passes.
``value_and_grad(backend)`` from the galilean kernel still differentiates
correctly through the energy-only path because nequix's `node_energies`
is a pure forward function.

Graph convention bridging
-------------------------
``_supercell_edges`` follows the MACE/jraph convention where
``senders[k]`` is the *central* atom and ``receivers[k]`` is the
*neighbor* atom (in its periodic image, located at
``positions[receivers[k]] + shifts[k]``).  Nequix's ``node_energies``
expects the opposite: messages flow FROM the neighbor TO the central
atom (``messages[senders]`` is read at the source, ``scatter_sum(dst=
receivers)`` aggregates at the destination).  This wrapper swaps the
two when calling into nequix; the displacement vector is computed as
``positions[receivers] - positions[senders] + shifts`` (from the centre
to its image-shifted neighbor) which is exactly nequix's
``positions[senders'] - positions[receivers'] + offsets`` after the swap.

Non-periodic systems
--------------------
For non-periodic systems (``cell == 0``), use ``supercell_trafo=(1, 1, 1)``
and pass a real cell vector.  The wrapper applies the same ``safe_cell``
trick as the NeuralIL backend (substitute a large dummy cell when the
trace is zero) so the supercell expansion has a well-defined geometry.
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
from jaxrens.backends.base import BackendResult
from jaxrens.utils.cell import wrap_positions

logger = logging.getLogger(__name__)

_NEQUIX_AVAILABLE = False
_NEQUIX_IMPORT_ERROR = ""
try:
    import equinox as eqx  # noqa: F401  (imported for availability check)
    from nequix.model import load_model

    _NEQUIX_AVAILABLE = True
except ImportError as exc:
    _NEQUIX_IMPORT_ERROR = str(exc)


def _require_nequix() -> None:
    if not _NEQUIX_AVAILABLE:
        raise ImportError(
            "nequix is required for the Nequix backend but is not installed.\n"
            "Install it with:  pip install '.[nequix]'\n"
            f"Original import error: {_NEQUIX_IMPORT_ERROR}"
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
    shifts: jnp.ndarray,
    n_atoms: int,
) -> jnp.ndarray:
    """Compute total potential energy via ``model.node_energies``.

    A single ghost node (index ``n_atoms``) absorbs the overflow edges
    produced by ``_supercell_edges``: senders/receivers point to it for
    out-of-bounds entries, and its node energy contribution is dropped
    by slicing before the sum.
    """
    padded_pos = jnp.zeros((n_atoms + 1, 3), dtype=positions.dtype)
    padded_pos = padded_pos.at[:n_atoms].set(positions)

    padded_species = jnp.zeros(n_atoms + 1, dtype=species.dtype)
    padded_species = padded_species.at[:n_atoms].set(species)

    # Centre → neighbor-at-image vector.  See module docstring for the
    # convention reconciliation between _supercell_edges and nequix.
    displacements = padded_pos[receivers] - padded_pos[senders] + shifts

    # Swap senders/receivers for nequix's message-direction convention:
    # nequix aggregates messages at `receivers`, and we want them
    # aggregated at the centre atom (our `senders`).
    node_energies = model.node_energies(
        displacements,
        padded_species,
        senders=receivers,
        receivers=senders,
    )  # (n_atoms + 1, 1)

    return jnp.sum(node_energies[:n_atoms, 0])


# ---------------------------------------------------------------------------
# NequixBackend
# ---------------------------------------------------------------------------


class NequixBackend:
    """Nequix energy backend for jaxrens.

    Satisfies the ``EnergyBackend`` protocol via duck typing: scalar
    energy from ``(positions, species, cell, max_neighbors)`` plus a
    per-atom max-neighbor-count report for the outer-loop bucket
    dispatch.
    """

    def __init__(
        self,
        model: Any,
        config: dict,
        supercell_trafo: tuple[int, int, int] = (1, 1, 1),
    ):
        self.r_cutoff = float(config["cutoff"])
        # nequix stores `atomic_numbers` as a sorted list at training
        # time; this is the order the species-index dimension uses.
        atomic_numbers = sorted(config["atomic_numbers"])
        self._atomic_numbers = tuple(int(z) for z in atomic_numbers)
        self.num_species = len(atomic_numbers)
        self._config = config
        self._model = model
        self.supercell_trafo = tuple(supercell_trafo)
        self.sc_a, self.sc_b, self.sc_c = self.supercell_trafo
        self._image_offsets = jnp.array(
            _make_image_offsets(self.sc_a, self.sc_b, self.sc_c),
            dtype=jnp.int32,
        )

    @property
    def atomic_numbers(self) -> tuple[int, ...]:
        """Atomic numbers (Z) of the species the model was trained on,
        in the order matching the model's species-index dimension.  Used
        by the resolver to map user-supplied Z numbers in
        ``start_species`` to the model's contiguous 0-based type indices.
        """
        return self._atomic_numbers

    def __call__(
        self,
        positions: jnp.ndarray,
        species: jnp.ndarray,
        cell: jnp.ndarray,
        max_neighbors: int = 50,
        ensemble_params: dict[str, Any] | None = None,
    ) -> BackendResult:
        n_atoms = positions.shape[0]
        max_edges = n_atoms * max_neighbors

        positions = wrap_positions(positions, cell)

        # Substitute a large dummy cell when the user passed cell == 0
        # (non-periodic systems).  Same trick as the NeuralIL backend —
        # ensures _supercell_edges has a well-defined image geometry.
        safe_cell = jnp.where(
            jnp.trace(cell) == 0,
            1000.0 * jnp.eye(3),
            cell,
        )

        (
            senders,
            receivers,
            shifts,
            _,
            overflow,
            true_max_per_atom,
        ) = _supercell_edges(
            positions,
            safe_cell,
            self.r_cutoff,
            max_edges,
            self._image_offsets,
        )

        energy = _compute_energy_only(
            self._model,
            positions,
            species,
            senders,
            receivers,
            shifts,
            n_atoms,
        )

        return BackendResult(
            energy=energy,
            max_neighbor_count=true_max_per_atom,
            overflow=overflow,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _resolve_model_path(model_name_or_path: str) -> str:
    """Accept either a bundled-model name or a path to a ``.nqx`` file.

    Bundled names are defined in ``NequixCalculator.URLS`` (e.g.
    ``nequix-mp-1``, ``nequix-oam-1``); these are auto-downloaded to
    ``~/.cache/nequix/models/`` on first use.  Existing paths are used
    as-is.
    """
    _require_nequix()
    from nequix.calculator import NequixCalculator

    p = Path(model_name_or_path).expanduser()
    if p.exists():
        return str(p)
    if model_name_or_path in NequixCalculator.URLS:
        import urllib.request

        cache_dir = Path("~/.cache/nequix/models/").expanduser()
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{model_name_or_path}.nqx"
        if not cache_path.exists():
            logger.info(
                "Downloading nequix model %s to %s",
                model_name_or_path,
                cache_path,
            )
            urllib.request.urlretrieve(
                NequixCalculator.URLS[model_name_or_path],
                cache_path,
            )
        return str(cache_path)
    raise FileNotFoundError(
        f"Nequix model {model_name_or_path!r} is neither an existing "
        f"path nor a known bundled name "
        f"({sorted(NequixCalculator.URLS.keys())})"
    )


def create_nequix(
    checkpoint_path: str | None = None,
    supercell_trafo: tuple[int, int, int] = (1, 1, 1),
    **kwargs: Any,
) -> NequixBackend:
    """Create a Nequix energy backend from a bundled name or a ``.nqx`` path.

    Args:
        checkpoint_path: Either a bundled model name (e.g.
            ``"nequix-mp-1"``, auto-downloaded on first use) or a path
            to a local ``.nqx`` file.
        supercell_trafo: ``(sc_a, sc_b, sc_c)`` supercell expansion for
            neighbor finding.  Must satisfy
            ``min(cell_diag * sc) >= 2 * r_cutoff`` to capture all true
            neighbors.  Default ``(1, 1, 1)`` is correct only when the
            unit cell is already at least ``2 * r_cutoff`` along every
            axis; bump for tighter cells (e.g. ``(2, 2, 2)`` mirrors
            MACE's default).

    Returns:
        ``NequixBackend`` instance.
    """
    _require_nequix()
    if checkpoint_path is None:
        raise ValueError("checkpoint_path is required for the Nequix backend")

    path = _resolve_model_path(checkpoint_path)
    model, config = load_model(path, kernel=False)

    logger.info(
        "Nequix backend created: r_cutoff=%.2f, %d species, supercell=%s",
        float(config["cutoff"]),
        len(config["atomic_numbers"]),
        supercell_trafo,
    )

    return NequixBackend(
        model=model,
        config=config,
        supercell_trafo=supercell_trafo,
    )
