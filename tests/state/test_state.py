"""Tests for the state dataclasses ``WalkerState`` and ``NSState`` as JAX
pytrees, for ``WalkerState.from_record``, and for the frozen ``NSConfig``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxrens.state.config import NSConfig
from jaxrens.state.mc_state import make_mc_state_class
from jaxrens.state.ns import NSState
from jaxrens.state.walker import WalkerState


class TestWalkerStatePytree:
    """Check that ``WalkerState`` works as a JAX pytree: it round-trips
    through flatten/unflatten, keeps its tree structure, passes through
    ``jit`` and ``vmap``, and keeps the static ``n_atoms`` out of the leaves.
    """

    def test_tree_flatten_unflatten(self, dummy_walker):
        leaves, treedef = jax.tree_util.tree_flatten(dummy_walker)
        reconstructed = treedef.unflatten(leaves)
        assert jnp.array_equal(reconstructed.positions, dummy_walker.positions)
        assert jnp.array_equal(reconstructed.types, dummy_walker.types)
        assert jnp.array_equal(reconstructed.energy, dummy_walker.energy)
        assert reconstructed.n_atoms == dummy_walker.n_atoms

    def test_tree_structure_stability(self, dummy_walker):
        walker2 = dummy_walker.set(energy=jnp.array(-2.0))
        struct1 = jax.tree_util.tree_structure(dummy_walker)
        struct2 = jax.tree_util.tree_structure(walker2)
        assert struct1 == struct2

    def test_jit_passthrough(self, dummy_walker):
        @jax.jit
        def get_energy(w: WalkerState) -> jnp.ndarray:
            return w.energy

        result = get_energy(dummy_walker)
        assert jnp.allclose(result, dummy_walker.energy)

    def test_jit_functional_update(self, dummy_walker):
        @jax.jit
        def shift_energy(w: WalkerState) -> WalkerState:
            return w.set(energy=w.energy + 1.0)

        result = shift_energy(dummy_walker)
        assert jnp.allclose(result.energy, dummy_walker.energy + 1.0)

    def test_vmap_over_walkers(self, dummy_walker):
        batch_positions = jnp.stack([dummy_walker.positions] * 4)
        batch_types = jnp.stack([dummy_walker.types] * 4)
        batch_energies = jnp.array([-1.0, -2.0, -3.0, -4.0])
        batch_cells = jnp.stack([dummy_walker.cell] * 4)

        batch = WalkerState(
            positions=batch_positions,
            types=batch_types,
            energy=batch_energies,
            cell=batch_cells,
            n_atoms=3,
        )

        @jax.vmap
        def get_energy(w: WalkerState) -> jnp.ndarray:
            return w.energy

        result = get_energy(batch)
        assert jnp.array_equal(result, batch_energies)

    def test_set_preserves_static(self, dummy_walker):
        updated = dummy_walker.set(energy=jnp.array(-99.0))
        assert updated.n_atoms == dummy_walker.n_atoms

    def test_static_field_not_in_leaves(self, dummy_walker):
        leaves = jax.tree_util.tree_leaves(dummy_walker)
        # The leaves are positions, types, energy and cell; n_atoms is static.
        assert len(leaves) == 4

    def test_nonperiodic_walker(self, dummy_walker_nonperiodic):
        """A walker with ``cell=None`` flattens and unflattens like any
        other."""
        leaves, treedef = jax.tree_util.tree_flatten(dummy_walker_nonperiodic)
        reconstructed = treedef.unflatten(leaves)
        assert reconstructed.cell is None
        assert jnp.array_equal(
            reconstructed.positions, dummy_walker_nonperiodic.positions
        )


class TestNSStatePytree:
    """Check that ``NSState`` round-trips through flatten/unflatten, passes
    through ``jit`` and supports functional updates with ``set``."""

    def _make_ns_state(self):
        MCState = make_mc_state_class()
        n_walkers = 10
        n_moves = 1
        population = MCState(
            positions=jnp.zeros((n_walkers, 3, 3)),
            types=jnp.zeros((n_walkers, 3), dtype=jnp.int32),
            energy=jnp.zeros(n_walkers),
            cell=jnp.tile(5.0 * jnp.eye(3), (n_walkers, 1, 1)),
            step_size=jnp.zeros(n_walkers),
            step_sizes=jnp.full((n_walkers, n_moves), 0.1),
            n_accepted=jnp.zeros((n_walkers, n_moves), dtype=jnp.int32),
            n_proposed=jnp.zeros((n_walkers, n_moves), dtype=jnp.int32),
            max_neighbor_count=jnp.zeros(n_walkers, dtype=jnp.int32),
            overflow=jnp.zeros(n_walkers, dtype=jnp.bool_),
            ensemble_params={},
        )
        return NSState(
            population=population,
            log_evidence=jnp.array(-jnp.inf),
            iteration=jnp.array(0, dtype=jnp.int32),
            emax=jnp.array(jnp.inf, dtype=jnp.float32),
            rng_key=jax.random.key(0),
            n_walkers=10,
            n_atoms=3,
        )

    def test_tree_flatten_unflatten(self):
        state = self._make_ns_state()
        leaves, treedef = jax.tree_util.tree_flatten(state)
        reconstructed = treedef.unflatten(leaves)
        assert jnp.array_equal(
            reconstructed.population.positions, state.population.positions
        )
        assert reconstructed.n_walkers == state.n_walkers
        assert reconstructed.n_atoms == state.n_atoms

    def test_jit_passthrough(self):
        state = self._make_ns_state()

        @jax.jit
        def get_iteration(s: NSState) -> jnp.ndarray:
            return s.iteration

        assert jnp.array_equal(get_iteration(state), jnp.array(0))

    def test_set_functional_update(self):
        state = self._make_ns_state()
        updated = state.set(iteration=jnp.array(42, dtype=jnp.int32))
        assert jnp.array_equal(updated.iteration, jnp.array(42))
        assert jnp.array_equal(state.iteration, jnp.array(0))  # unchanged


class TestWalkerStateFromRecord:
    """Check that ``WalkerState.from_record`` rebuilds a walker from the loose
    dict used on disk and at the callback boundary."""

    @staticmethod
    def _record(**extra):
        return {
            "positions": np.arange(9, dtype=np.float64).reshape(3, 3),
            "types": np.array([0, 1, 1]),
            "energy": -1.5,
            **extra,
        }

    def test_numpy_record_becomes_jax_arrays(self):
        walker = WalkerState.from_record(self._record())
        assert isinstance(walker.positions, jax.Array)
        assert isinstance(walker.types, jax.Array)
        assert isinstance(walker.energy, jax.Array)
        assert jnp.array_equal(walker.positions, jnp.arange(9.0).reshape(3, 3))
        assert jnp.array_equal(walker.types, jnp.array([0, 1, 1]))
        assert walker.energy == -1.5

    @pytest.mark.parametrize("key", ["cell", "box"])
    def test_cell_is_read_from_cell_or_box(self, key):
        # The extxyz and HDF5 writers store the cell under ``box``.
        cell = 5.0 * np.eye(3)
        walker = WalkerState.from_record(self._record(**{key: cell}))
        assert jnp.array_equal(walker.cell, jnp.asarray(cell))

    def test_cell_key_wins_over_box(self):
        record = self._record(cell=5.0 * np.eye(3), box=9.0 * np.eye(3))
        walker = WalkerState.from_record(record)
        assert float(walker.cell[0, 0]) == 5.0

    def test_missing_cell_gives_none(self):
        assert WalkerState.from_record(self._record()).cell is None

    def test_n_atoms_is_inferred_from_positions(self):
        assert WalkerState.from_record(self._record()).n_atoms == 3


def test_ns_config_frozen():
    config = NSConfig()
    with pytest.raises(AttributeError):
        config.n_live = 100  # type: ignore
