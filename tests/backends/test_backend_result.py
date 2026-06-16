"""Unit tests for the BackendResult pytree and eval_energy_and_forces.

Covers the new structured backend return contract (energy + control +
reserved diagnostics slots) and the native-or-autodiff force dispatch.
"""

import jax
import jax.numpy as jnp
import pytest

from jaxrens.backends.base import BackendResult, eval_energy_and_forces


# ---------------------------------------------------------------------------
# Fake backends
# ---------------------------------------------------------------------------


class EnergyOnlyBackend:
    """Quadratic bowl, energy-only (exercises the autodiff fallback)."""

    r_cutoff = 4.5

    def __call__(self, positions, species, cell, max_neighbors, ensemble_params=None):
        # control fields take the sentinel defaults (no neighbor list)
        return BackendResult(energy=(positions**2).sum())


class NativeForceBackend:
    """Same energy, but advertises a (deliberately non-conservative) force."""

    r_cutoff = 4.5

    def __call__(self, positions, species, cell, max_neighbors, ensemble_params=None):
        return BackendResult(
            energy=(positions**2).sum(),
            max_neighbor_count=jnp.asarray(3, jnp.int32),
            overflow=jnp.asarray(False),
        )

    def energy_and_forces(self, positions, species, cell, max_neighbors, ensemble_params=None):
        res = self(positions, species, cell, max_neighbors, ensemble_params=ensemble_params)
        # intentionally NOT -dE/dx, to prove the native path is used verbatim
        return res._replace(forces=jnp.full_like(positions, 7.0))


@pytest.fixture
def pos():
    return jnp.array([[0.0, 0.0, 0.0], [1.0, 0.5, -0.5], [2.0, 1.0, 0.0]])


@pytest.fixture
def species():
    return jnp.array([0, 0, 1], dtype=jnp.int32)


@pytest.fixture
def cell():
    return 5.0 * jnp.eye(3)


# ---------------------------------------------------------------------------
# BackendResult as a pytree
# ---------------------------------------------------------------------------


class TestBackendResultPytree:
    def test_energy_only_leaves(self):
        r = BackendResult(energy=jnp.array(1.5))
        leaves = jax.tree_util.tree_leaves(r)
        # energy + the two control sentinels; the None fields contribute no leaves
        assert len(leaves) == 3

    def test_none_fields_excluded(self):
        r = BackendResult(energy=jnp.array(1.0))
        assert r.forces is None
        assert r.energy_members is None and r.forces_members is None

    def test_flatten_unflatten_roundtrip(self):
        r = BackendResult(energy=jnp.array(2.0), forces=jnp.ones((3, 3)))
        leaves, treedef = jax.tree_util.tree_flatten(r)
        r2 = jax.tree_util.tree_unflatten(treedef, leaves)
        assert r2.energy == r.energy
        assert jnp.array_equal(r2.forces, r.forces)

    def test_sentinel_and_filled_share_treedef(self):
        sentinel = BackendResult(energy=jnp.array(1.0))
        filled = BackendResult(
            energy=jnp.array(1.0),
            max_neighbor_count=jnp.asarray(7),
            overflow=jnp.asarray(True),
        )
        assert jax.tree_util.tree_structure(sentinel) == jax.tree_util.tree_structure(filled)

    def test_replace_forwards_other_fields(self):
        # the wrapper pattern: modify energy, pass everything else through
        base = BackendResult(
            energy=jnp.array(1.0),
            max_neighbor_count=jnp.asarray(5),
            forces=jnp.ones((2, 3)),
        )
        wrapped = base._replace(energy=base.energy + 10.0)
        assert wrapped.energy == 11.0
        assert wrapped.max_neighbor_count == 5
        assert jnp.array_equal(wrapped.forces, base.forces)

    def test_legacy_tuple(self):
        r = BackendResult(
            energy=jnp.array(3.0),
            max_neighbor_count=jnp.asarray(4),
            overflow=jnp.asarray(False),
        )
        e, count, overflow = r.legacy()
        assert (float(e), int(count), bool(overflow)) == (3.0, 4, False)


# ---------------------------------------------------------------------------
# treedef stability (the lax.scan / lax.cond safety property)
# ---------------------------------------------------------------------------


class TestTreedefStability:
    def test_backend_treedef_stable_across_calls(self, pos, species, cell):
        b = NativeForceBackend()
        t1 = jax.tree_util.tree_structure(b(pos, species, cell, 10))
        t2 = jax.tree_util.tree_structure(b(pos * 2.0, species, cell, 10))
        assert t1 == t2

    def test_threads_through_scan(self, pos, species, cell):
        b = EnergyOnlyBackend()

        def body(carry, _):
            res = b(carry, species, cell, 10)
            return carry, res.energy

        _, energies = jax.lax.scan(body, pos, xs=None, length=3)
        assert energies.shape == (3,)


# ---------------------------------------------------------------------------
# eval_energy_and_forces
# ---------------------------------------------------------------------------


class TestEvalEnergyAndForces:
    def test_autodiff_fallback_is_negative_grad(self, pos, species, cell):
        b = EnergyOnlyBackend()
        res = eval_energy_and_forces(b, pos, species, cell, 10)
        assert res.forces is not None
        assert jnp.allclose(res.forces, -2.0 * pos)  # E = sum(x^2) -> F = -2x
        assert float(res.energy) == float((pos**2).sum())

    def test_fallback_preserves_control_fields(self, pos, species, cell):
        # control fields from __call__ must survive the value_and_grad(has_aux) path
        b = NativeForceBackend()
        # temporarily hide the native method to force the fallback
        delattr_native = b.energy_and_forces
        object.__setattr__(b, "energy_and_forces", None)
        try:
            res = eval_energy_and_forces(b, pos, species, cell, 10)
        finally:
            object.__setattr__(b, "energy_and_forces", delattr_native)
        assert int(res.max_neighbor_count) == 3
        assert jnp.allclose(res.forces, -2.0 * pos)

    def test_native_path_used_verbatim(self, pos, species, cell):
        # native energy_and_forces returns forces=7 (not -dE/dx) — must be used as-is
        b = NativeForceBackend()
        res = eval_energy_and_forces(b, pos, species, cell, 10)
        assert jnp.allclose(res.forces, 7.0)
        assert int(res.max_neighbor_count) == 3

    def test_jit_and_vmap(self, species, cell):
        b = EnergyOnlyBackend()
        batch = jnp.arange(2 * 3 * 3, dtype=jnp.float32).reshape(2, 3, 3)
        fn = jax.jit(jax.vmap(lambda p: eval_energy_and_forces(b, p, species, cell, 10).forces))
        forces = fn(batch)
        assert forces.shape == (2, 3, 3)
        assert jnp.allclose(forces, -2.0 * batch)
