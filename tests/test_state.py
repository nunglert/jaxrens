"""Test state dataclasses: pytree registration, JIT compatibility, vmap, serialization.

Part of PR 2: state dataclasses.
Verifies that WalkerState and NSState work correctly as JAX pytrees.
"""

import jax
import jax.numpy as jnp
import pytest

from jaxrens.state.walker import WalkerState
from jaxrens.state.ns import NSState
from jaxrens.state.mc_state import make_mc_state_class


class TestWalkerStatePytree:
    """WalkerState must be a valid JAX pytree."""

    def test_tree_flatten_unflatten(self, dummy_walker):
        """Flatten and unflatten should roundtrip."""
        leaves, treedef = jax.tree_util.tree_flatten(dummy_walker)
        reconstructed = treedef.unflatten(leaves)
        assert jnp.array_equal(reconstructed.positions, dummy_walker.positions)
        assert jnp.array_equal(reconstructed.types, dummy_walker.types)
        assert jnp.array_equal(reconstructed.energy, dummy_walker.energy)
        assert reconstructed.n_atoms == dummy_walker.n_atoms

    def test_tree_structure_stability(self, dummy_walker):
        """Tree structure should not change between identical walkers."""
        walker2 = dummy_walker.set(energy=jnp.array(-2.0))
        struct1 = jax.tree_util.tree_structure(dummy_walker)
        struct2 = jax.tree_util.tree_structure(walker2)
        assert struct1 == struct2

    def test_jit_passthrough(self, dummy_walker):
        """WalkerState should pass through JIT without error."""

        @jax.jit
        def get_energy(w: WalkerState) -> jnp.ndarray:
            return w.energy

        result = get_energy(dummy_walker)
        assert jnp.allclose(result, dummy_walker.energy)

    def test_jit_functional_update(self, dummy_walker):
        """set() should work inside JIT."""

        @jax.jit
        def shift_energy(w: WalkerState) -> WalkerState:
            return w.set(energy=w.energy + 1.0)

        result = shift_energy(dummy_walker)
        assert jnp.allclose(result.energy, dummy_walker.energy + 1.0)

    def test_vmap_over_walkers(self, dummy_walker):
        """vmap should work over a batch of walkers."""
        # Create batch of 4 walkers
        batch_positions = jnp.stack([dummy_walker.positions] * 4)
        batch_types = jnp.stack([dummy_walker.types] * 4)
        batch_energies = jnp.array([-1.0, -2.0, -3.0, -4.0])
        batch_boxes = jnp.stack([dummy_walker.box] * 4)

        batch = WalkerState(
            positions=batch_positions,
            types=batch_types,
            energy=batch_energies,
            box=batch_boxes,
            n_atoms=3,
        )

        @jax.vmap
        def get_energy(w: WalkerState) -> jnp.ndarray:
            return w.energy

        result = get_energy(batch)
        assert jnp.array_equal(result, batch_energies)

    def test_set_preserves_static(self, dummy_walker):
        """set() should preserve static fields."""
        updated = dummy_walker.set(energy=jnp.array(-99.0))
        assert updated.n_atoms == dummy_walker.n_atoms

    def test_static_field_not_in_leaves(self, dummy_walker):
        """Static fields should not appear in pytree leaves."""
        leaves = jax.tree_util.tree_leaves(dummy_walker)
        # n_atoms is static, should not be in leaves
        # Leaves: positions, types, energy, box = 4 leaves
        assert len(leaves) == 4

    def test_nonperiodic_walker(self, dummy_walker_nonperiodic):
        """Walker with box=None should work as pytree."""
        leaves, treedef = jax.tree_util.tree_flatten(dummy_walker_nonperiodic)
        reconstructed = treedef.unflatten(leaves)
        assert reconstructed.box is None
        assert jnp.array_equal(reconstructed.positions, dummy_walker_nonperiodic.positions)


class TestNSStatePytree:
    """NSState must be a valid JAX pytree."""

    def _make_ns_state(self):
        MCState = make_mc_state_class()
        n_walkers = 10
        n_moves = 1
        population = MCState(
            positions=jnp.zeros((n_walkers, 3, 3)),
            types=jnp.zeros((n_walkers, 3), dtype=jnp.int32),
            energy=jnp.zeros(n_walkers),
            box=jnp.tile(5.0 * jnp.eye(3), (n_walkers, 1, 1)),
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
            dead_energies=jnp.full(1000, jnp.inf),
            dead_positions=jnp.zeros((1000, 3, 3)),
            dead_volumes=jnp.zeros(1000),
            log_evidence=jnp.array(-jnp.inf),
            iteration=jnp.array(0, dtype=jnp.int32),
            n_dead=jnp.array(0, dtype=jnp.int32),
            rng_key=jax.random.key(0),
            n_walkers=10,
            n_atoms=3,
            max_dead=1000,
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
        assert jnp.array_equal(state.iteration, jnp.array(0))  # original unchanged


class TestConfigs:
    """Test configuration dataclasses."""

    def test_ns_config_frozen(self):
        config = NSConfig()
        with pytest.raises(AttributeError):
            config.n_live = 100  # type: ignore

    def test_ns_config_defaults(self):
        from jaxrens.state.config import NSConfig
        config = NSConfig()
        assert config.n_live == 500
        assert config.max_iterations == 50_000
        assert config.n_mcmc_steps == 20

    def test_backend_config_max_neighbors(self):
        from jaxrens.state.config import BackendConfig
        config = BackendConfig(max_neighbors_list=[10, 20, 30, 40, 50])
        assert config.max_neighbors_list == [10, 20, 30, 40, 50]
        assert config.max_neighbors_offset == 5

    def test_output_config_defaults(self):
        from jaxrens.state.config import OutputConfig
        config = OutputConfig()
        assert config.format == "extxyz"
        assert config.checkpoint_interval == 100


# Import NSConfig at module level for TestConfigs
from jaxrens.state.config import NSConfig
