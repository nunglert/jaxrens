"""MACE-JAX energy backend.

Wraps a pre-trained MACE model (loaded via mace-jax) behind the EnergyBackend
interface. JIT-compatible neighbor finding via supercell expansion +
static-shape edge extraction (jnp.nonzero with size parameter).

mace-jax is an optional dependency. All imports are guarded.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from jaxrens.backends._graph_neighbors import (
    _compute_true_max_neighbors,
    _make_image_offsets,
    _max_neighbor_count_from_mask,
    _neighbor_mask,
    _supercell_edges,
)
from jaxrens.backends.base import BackendResult
from jaxrens.utils.cell import wrap_positions

# Re-export the edge-finding helpers under the `mace` namespace for
# backwards-compat with existing tests (`tests/test_mace.py` imports
# `_make_image_offsets` / `_supercell_edges` from here).
__all__ = [
    "_compute_true_max_neighbors",
    "_make_image_offsets",
    "_max_neighbor_count_from_mask",
    "_neighbor_mask",
    "_supercell_edges",
    "MACEBackend",
    "create_mace",
    "create_mace_from_pickle",
    "is_available",
]

logger = logging.getLogger(__name__)

_MACE_JAX_AVAILABLE = False
_MACE_IMPORT_ERROR = ""
try:
    from flax import nnx, serialization
    from mace_jax.tools.bundle import load_model_bundle

    _MACE_JAX_AVAILABLE = True
except ImportError as exc:
    _MACE_IMPORT_ERROR = str(exc)


def _require_mace():
    if not _MACE_JAX_AVAILABLE:
        raise ImportError(
            "mace-jax is required for the MACE backend but is not installed.\n"
            "Install it with:  pip install '.[mace]'\n"
            f"Original import error: {_MACE_IMPORT_ERROR}"
        )


def is_available() -> bool:
    return _MACE_JAX_AVAILABLE


# ---------------------------------------------------------------------------
# Build the data dict that mace-jax expects
# ---------------------------------------------------------------------------


def _build_mace_data(
    positions: jnp.ndarray,
    species: jnp.ndarray,
    cell: jnp.ndarray,
    senders: jnp.ndarray,
    receivers: jnp.ndarray,
    shifts: jnp.ndarray,
    num_species: int,
    n_atoms: int,
) -> dict[str, jnp.ndarray]:
    """Assemble the flat dict that mace-jax's prepare_graph reads.

    Follows jraph padding convention: N real atoms + 1 ghost node,
    2 graphs (real + ghost). Ghost edges point sender=receiver=N.

    Args:
        positions: (N, 3) real atom positions.
        species: (N,) integer species indices (0-based, matching model's z_table).
        cell: (3, 3) unit cell.
        senders: (max_edges,) sender node indices.
        receivers: (max_edges,) receiver node indices.
        shifts: (max_edges, 3) Cartesian shift vectors.
        num_species: Number of species in the model.
        n_atoms: Number of real atoms N.

    Returns:
        Dict with keys: positions, node_attrs, node_attrs_index,
        edge_index, shifts, unit_shifts, batch, ptr, cell, head.
    """
    n_nodes = n_atoms + 1  # +1 ghost node
    max_edges = senders.shape[0]

    # Positions: pad with one ghost node at origin
    padded_pos = jnp.zeros((n_nodes, 3), dtype=positions.dtype)
    padded_pos = padded_pos.at[:n_atoms].set(positions)

    # One-hot species: (n_nodes, num_species), ghost is all-zero
    node_attrs = jnp.zeros((n_nodes, num_species), dtype=positions.dtype)
    node_attrs = node_attrs.at[jnp.arange(n_atoms), species].set(1.0)

    # Species index (argmax of one-hot)
    node_attrs_index = jnp.zeros(n_nodes, dtype=jnp.int32)
    node_attrs_index = node_attrs_index.at[:n_atoms].set(species)

    # Edge index: (2, max_edges) — [senders; receivers]
    edge_index = jnp.stack([senders, receivers], axis=0).astype(jnp.int32)

    # Batch assignment: all real atoms -> graph 0, ghost -> graph 1
    batch = jnp.zeros(n_nodes, dtype=jnp.int32)
    batch = batch.at[n_atoms].set(1)

    # Pointer: cumulative node counts [0, n_atoms, n_nodes]
    ptr = jnp.array([0, n_atoms, n_nodes], dtype=jnp.int32)

    # Cell: (2, 3, 3) — real cell + zeros for ghost graph
    padded_cell = jnp.zeros((2, 3, 3), dtype=positions.dtype)
    padded_cell = padded_cell.at[0].set(cell)

    # Unit shifts: we don't compute these explicitly, set to zeros
    # (MACE uses Cartesian shifts directly via get_edge_vectors_and_lengths)
    unit_shifts = jnp.zeros_like(shifts)

    # Head index for multi-head models
    head = jnp.zeros(2, dtype=jnp.int32)

    return {
        "positions": padded_pos,
        "node_attrs": node_attrs,
        "node_attrs_index": node_attrs_index,
        "edge_index": edge_index,
        "shifts": shifts,
        "unit_shifts": unit_shifts,
        "batch": batch,
        "ptr": ptr,
        "cell": padded_cell,
        "head": head,
    }


# ---------------------------------------------------------------------------
# MACEBackend
# ---------------------------------------------------------------------------


class MACEBackend:
    """MACE-JAX energy backend for jaxrens.

    Model weights live on the instance. JIT-compatible supercell
    neighbor finding. Satisfies EnergyBackend protocol via duck typing.

    The graphdef (model structure) is captured in a closure for JIT.
    The state (model parameters) is stored as arrays.
    """

    def __init__(
        self,
        graphdef,
        state,
        r_cutoff: float,
        num_species: int,
        atomic_numbers: list[int],
        supercell_trafo: tuple[int, int, int] = (2, 2, 2),
    ):
        self.r_cutoff = r_cutoff
        self.num_species = num_species
        self.atomic_numbers = list(atomic_numbers)
        self.sc_a, self.sc_b, self.sc_c = supercell_trafo
        self.image_offsets = jnp.array(
            _make_image_offsets(self.sc_a, self.sc_b, self.sc_c),
            dtype=jnp.int32,
        )

        # Store graphdef and state for functional model calls
        self._graphdef = graphdef
        self._state = state

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

        # 1. Find edges via supercell expansion.  ``true_max_per_atom`` is the
        # actual max neighbor count per atom computed from the pre-truncation
        # mask; the outer-loop overflow retry relies on this being independent
        # of ``max_edges`` so that escalation doesn't saturate at the bucket.
        (
            senders,
            receivers,
            shifts,
            n_actual,
            overflow,
            true_max_per_atom,
        ) = _supercell_edges(
            positions,
            cell,
            self.r_cutoff,
            max_edges,
            self.image_offsets,
        )

        # 2. Build data dict for mace-jax
        data = _build_mace_data(
            positions,
            species,
            cell,
            senders,
            receivers,
            shifts,
            self.num_species,
            n_atoms,
        )

        # 3. Run MACE model (energy-only path)
        model = nnx.merge(self._graphdef, self._state)
        out = model._energy_fn(data)
        energy = out["energy"][0]  # scalar energy for the real graph

        return BackendResult(
            energy=energy,
            max_neighbor_count=true_max_per_atom,
            overflow=overflow,
        )

    def max_neighbors_for(
        self,
        positions: jnp.ndarray,
        cell: jnp.ndarray,
    ) -> jnp.ndarray:
        """Per-atom max neighbor count for ``(positions, cell)``, geometry only.

        Used at init time so the NS loop can start with a correctly-sized
        neighbor bucket and accurate per-walker ``max_neighbor_count``,
        without probing the GNN forward pass.  vmap-friendly.
        """
        # Wrap to match the energy path so the bucket sizing counts the same
        # neighbors the forward pass will see.
        positions = wrap_positions(positions, cell)
        return _compute_true_max_neighbors(
            positions,
            cell,
            self.r_cutoff,
            self.image_offsets,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_mace(
    model_path: str | None = None,
    supercell_trafo: tuple[int, int, int] = (2, 2, 2),
    **kwargs: Any,
) -> MACEBackend:
    """Create a MACE-JAX energy backend from a model bundle.

    Loads a pre-trained MACE model from a bundle directory
    (config.json + params.msgpack) or a .ckpt checkpoint file.

    The backend always runs in float32.  mace-jax's bundle loader toggles
    ``jax_enable_x64`` based on the ``dtype`` passed to ``load_model_bundle``;
    we pin it to float32 so the rest of jaxrens (cells / positions / energies
    are all float32 by construction) doesn't get silently promoted.

    Args:
        model_path: Path to model bundle directory or checkpoint file.
        supercell_trafo: (sc_a, sc_b, sc_c) supercell expansion for
            neighbor finding. Must satisfy min(cell_diag * sc) >= 2 * r_cutoff.

    Returns:
        MACEBackend instance.
    """
    _require_mace()

    if model_path is None:
        raise ValueError("model_path is required for the MACE backend.")

    bundle = load_model_bundle(model_path, dtype="float32")

    # Build the NNX module and split into graphdef + state
    from mace_jax.tools import model_builder

    config = bundle.config
    config, _, _ = model_builder._normalize_atomic_config(config)
    module = model_builder._build_jax_model(config, rngs=nnx.Rngs(0))
    graphdef, state = nnx.split(module)

    # Load saved params into the state
    from mace_jax.tools.bundle import _replace_state_with_specials

    _replace_state_with_specials(state, bundle.params)

    r_cutoff = float(config["r_max"])
    atomic_numbers = [int(z) for z in config["atomic_numbers"]]
    num_species = len(atomic_numbers)

    logger.info(
        "MACE backend created: r_cutoff=%.2f, %d species, supercell=%s",
        r_cutoff,
        num_species,
        supercell_trafo,
    )

    return MACEBackend(
        graphdef=graphdef,
        state=state,
        r_cutoff=r_cutoff,
        num_species=num_species,
        atomic_numbers=atomic_numbers,
        supercell_trafo=supercell_trafo,
    )


def create_mace_from_pickle(
    pickle_path: str,
    supercell_trafo: tuple[int, int, int] = (2, 2, 2),
) -> MACEBackend:
    """Create a MACE backend from a pickle file with graphdef + state.

    This is the simplest loading path — avoids config serialization issues.
    Use save_mace_test_fixture.py to generate the pickle.

    Args:
        pickle_path: Path to model_bundle.pkl.
        supercell_trafo: Supercell expansion for neighbor finding.

    Returns:
        MACEBackend instance.
    """
    _require_mace()

    import pickle

    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    return MACEBackend(
        graphdef=data["graphdef"],
        state=data["state"],
        r_cutoff=float(data["r_cutoff"]),
        num_species=int(data["num_species"]),
        atomic_numbers=list(data["atomic_numbers"]),
        supercell_trafo=supercell_trafo,
    )
