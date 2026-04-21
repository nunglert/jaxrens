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
            f"mace-jax is required for the MACE backend but not installed: "
            f"{_MACE_IMPORT_ERROR}"
        )


def is_available() -> bool:
    return _MACE_JAX_AVAILABLE


# ---------------------------------------------------------------------------
# JIT-compatible supercell edge finding
# ---------------------------------------------------------------------------


def _make_image_offsets(sc_a: int, sc_b: int, sc_c: int) -> np.ndarray:
    """Generate integer image offset vectors for a centered supercell.

    Offsets range from -sc//2 to +sc//2 in each direction, ensuring
    neighbors are found in all periodic directions.

    Returns:
        (sc_dim, 3) integer array of image offsets.
    """
    a = np.arange(-(sc_a // 2), sc_a // 2 + 1)
    b = np.arange(-(sc_b // 2), sc_b // 2 + 1)
    c = np.arange(-(sc_c // 2), sc_c // 2 + 1)
    grid = np.stack(np.meshgrid(a, b, c, indexing="ij"), axis=-1)
    return grid.reshape(-1, 3)


def _supercell_edges(
    positions: jnp.ndarray,
    cell: jnp.ndarray,
    r_cutoff: float,
    max_edges: int,
    image_offsets: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Find edges within cutoff using supercell expansion. JIT-compatible.

    Args:
        positions: (N, 3) atom positions in the unit cell.
        cell: (3, 3) unit cell matrix (rows are lattice vectors).
        r_cutoff: Cutoff radius.
        max_edges: Static size for edge buffer.
        image_offsets: (sc_dim, 3) integer image offsets (static).

    Returns:
        senders: (max_edges,) sender indices in [0, N).
        receivers: (max_edges,) receiver indices in [0, N).
        shifts: (max_edges, 3) Cartesian shift vectors.
        n_actual: Scalar, actual number of edges found.
        overflow: Bool, True if n_actual > max_edges.
    """
    n_atoms = positions.shape[0]
    sc_dim = image_offsets.shape[0]

    # Cartesian shift for each image: (sc_dim, 3)
    cart_shifts = image_offsets @ cell

    # Supercell positions: (sc_dim * N, 3)
    # For each image s and atom j: pos_j + cart_shifts[s]
    super_positions = (
        positions[None, :, :] + cart_shifts[:, None, :]
    ).reshape(-1, 3)

    # Displacements: receiver_pos - sender_pos
    # delta[i, k] = super_positions[k] - positions[i] for sender i, supercell atom k
    # shape: (N, sc_dim * N, 3)
    delta = super_positions[None, :, :] - positions[:, None, :]

    # Distances: (N, sc_dim * N)
    distances = jnp.linalg.norm(delta, axis=-1)

    # Edge mask: within cutoff and not self-interaction
    mask = (distances > 1e-10) & (distances < r_cutoff)

    # Static-shape edge extraction
    flat_mask = mask.ravel()
    n_total = n_atoms * sc_dim * n_atoms
    fill_val = n_total  # out-of-bounds sentinel
    flat_indices = jnp.nonzero(flat_mask, size=max_edges, fill_value=fill_val)[0]

    n_actual = jnp.sum(flat_mask)
    overflow = n_actual > max_edges

    # Decode flat indices -> (sender_in_unit_cell, supercell_atom_index)
    # flat index k maps to: sender = k // (sc_dim * N), j_sc = k % (sc_dim * N)
    sc_n = sc_dim * n_atoms
    senders = flat_indices // sc_n
    j_sc = flat_indices % sc_n

    # j_sc -> (image_index, receiver_in_unit_cell)
    receivers = j_sc % n_atoms
    image_idx = j_sc // n_atoms

    # Cartesian shifts for each edge
    shifts = cart_shifts[image_idx]

    # Ghost handling: sentinel indices get sender=receiver=N (ghost node), shift=0
    is_ghost = flat_indices >= n_total
    senders = jnp.where(is_ghost, n_atoms, senders)
    receivers = jnp.where(is_ghost, n_atoms, receivers)
    shifts = jnp.where(is_ghost[:, None], 0.0, shifts)

    return senders, receivers, shifts, n_actual, overflow


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
    ) -> tuple[jnp.ndarray, int, bool]:
        n_atoms = positions.shape[0]
        max_edges = n_atoms * max_neighbors

        # 1. Find edges via supercell expansion
        senders, receivers, shifts, n_actual, overflow = _supercell_edges(
            positions, cell, self.r_cutoff, max_edges, self.image_offsets,
        )

        # 2. Build data dict for mace-jax
        data = _build_mace_data(
            positions, species, cell,
            senders, receivers, shifts,
            self.num_species, n_atoms,
        )

        # 3. Run MACE model (energy-only path)
        model = nnx.merge(self._graphdef, self._state)
        out = model._energy_fn(data)
        energy = out["energy"][0]  # scalar energy for the real graph

        # 4. Compute max neighbor count per atom
        counts = jnp.zeros(n_atoms, dtype=jnp.int32)
        # Only count real edges (not ghost)
        real_senders = jnp.where(senders < n_atoms, senders, 0)
        is_real = senders < n_atoms
        counts = counts.at[real_senders].add(is_real.astype(jnp.int32))
        neighbor_count = jnp.max(counts)

        return energy, neighbor_count, overflow


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
        r_cutoff, num_species, supercell_trafo,
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
