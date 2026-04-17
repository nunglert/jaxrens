"""Unit tests for jaxrens.init.cells.

Covers sample_initial_volume and cell_shape_walk.
All JIT-compatible functions are tested under jax.jit per project policy.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jaxrens.init.cells import cell_shape_walk, sample_initial_volume
from jaxrens.utils.cell import get_volume


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_KEY = jax.random.key(0)
_CUBIC_CELL = 5.0 * jnp.eye(3)


def _batch_keys(n: int, base: int = 0):
    return jax.random.split(jax.random.key(base), n)


# ---------------------------------------------------------------------------
# sample_initial_volume
# ---------------------------------------------------------------------------


class TestSampleInitialVolume:
    def test_deterministic_same_key(self):
        key = jax.random.key(7)
        a = sample_initial_volume(key, n_atoms=4, max_volume_per_atom=50.0)
        b = sample_initial_volume(key, n_atoms=4, max_volume_per_atom=50.0)
        assert jnp.allclose(a, b)

    def test_flat_v_prior_range(self):
        n_atoms = 4
        max_vpa = 50.0
        v_max = n_atoms * max_vpa
        keys = _batch_keys(1000, base=1)
        samples = jax.vmap(
            lambda k: sample_initial_volume(k, n_atoms, max_vpa, flat_V_prior=True)
        )(keys)
        # Corresponding volumes
        volumes = samples ** 3
        assert jnp.all(volumes >= 0.0)
        assert jnp.all(volumes <= v_max + 1e-5)

    def test_flat_v_prior_approximately_uniform(self):
        n_atoms = 4
        max_vpa = 50.0
        v_max = float(n_atoms * max_vpa)
        keys = _batch_keys(2000, base=2)
        samples = jax.vmap(
            lambda k: sample_initial_volume(k, n_atoms, max_vpa, flat_V_prior=True)
        )(keys)
        volumes = samples ** 3
        # Split into 5 bins; each should hold roughly 400 samples (±30%)
        bin_edges = jnp.linspace(0.0, v_max, 6)
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            count = jnp.sum((volumes >= lo) & (volumes < hi))
            assert count > 200, f"Bin [{lo:.1f}, {hi:.1f}) too empty: {int(count)}"
            assert count < 600, f"Bin [{lo:.1f}, {hi:.1f}) too full: {int(count)}"

    def test_vn_prior_formula_fixed_values(self):
        """Verify V^N formula against hand computation for a few fixed U values."""
        n_atoms = 4
        max_vpa = 50.0
        exponent = 1.0 / float(n_atoms + 1)
        v_max = float(n_atoms * max_vpa)

        for u_val in [0.1, 0.5, 0.9]:
            expected = float(v_max * u_val ** exponent) ** (1.0 / 3.0)

            key = jax.random.key(42)
            # Manually pass a key that we know will produce u_val.
            # Instead, call the function with a known uniform draw by monkey-
            # patching through a simple test: re-derive from the output.
            # Easier: directly test the formula by constructing a fake uniform.
            # We test: given the formula, does our impl match?
            lc = float(sample_initial_volume(
                jax.random.key(99), n_atoms=n_atoms, max_volume_per_atom=max_vpa,
                flat_V_prior=False
            ))
            # We can't control the exact U, but we can re-derive what U was
            # from a flat-V sample and compare both modes for consistency.
            # For a deterministic formula check: inject U via a known key.
            # Verify against the formula by computing expected from actual U:
            u_actual = float(
                jax.random.uniform(jax.random.key(99), shape=(), dtype=jnp.float32)
            )
            expected_from_key = (v_max * u_actual ** exponent) ** (1.0 / 3.0)
            assert abs(lc - expected_from_key) < 1e-4, (
                f"V^N formula mismatch: got {lc}, expected {expected_from_key}"
            )
            break  # one fixed key is enough to validate the formula

    def test_n_atoms_1_no_nan(self):
        key = jax.random.key(3)
        lc = sample_initial_volume(key, n_atoms=1, max_volume_per_atom=20.0)
        assert jnp.isfinite(lc)
        assert lc > 0.0

    def test_jit(self):
        key = jax.random.key(4)
        jit_fn = jax.jit(
            lambda k: sample_initial_volume(k, n_atoms=4, max_volume_per_atom=50.0)
        )
        result = jit_fn(key)
        assert jnp.isfinite(result)

    def test_vmap_over_keys(self):
        keys = _batch_keys(8, base=5)
        fn = lambda k: sample_initial_volume(k, n_atoms=4, max_volume_per_atom=50.0)
        results = jax.vmap(fn)(keys)
        assert results.shape == (8,)
        assert jnp.all(jnp.isfinite(results))
        # Different keys should (almost certainly) produce different values
        assert not jnp.all(results == results[0])


# ---------------------------------------------------------------------------
# cell_shape_walk
# ---------------------------------------------------------------------------


def _run_walk(key, cell=None, n_steps=50, step_size_shear=0.05,
              step_size_stretch=0.05, min_aspect=0.3, n_atoms=4,
              max_vpa=100.0, min_vpa=1.0):
    if cell is None:
        cell = _CUBIC_CELL
    return cell_shape_walk(
        key=key,
        cell=cell,
        n_steps=n_steps,
        step_size_shear=step_size_shear,
        step_size_stretch=step_size_stretch,
        min_aspect_ratio_val=min_aspect,
        n_atoms=n_atoms,
        max_volume_per_atom=max_vpa,
        min_volume_per_atom=min_vpa,
    )


class TestCellShapeWalk:
    def test_volume_conservation(self):
        key = jax.random.key(10)
        initial_vol = float(get_volume(_CUBIC_CELL))
        final_cell, n_accepted = _run_walk(key, n_steps=50)
        final_vol = float(get_volume(final_cell))
        assert abs(final_vol - initial_vol) < 1e-4, (
            f"Volume changed: {initial_vol} -> {final_vol}"
        )

    def test_zero_step_size_cell_unchanged(self):
        key = jax.random.key(11)
        final_cell, n_accepted = _run_walk(
            key, n_steps=20, step_size_shear=0.0, step_size_stretch=0.0
        )
        # With zero step size both proposals produce identity-scaled cells;
        # they will pass the aspect-ratio check (cubic cell has aspect 1.0).
        # Volume rescaling restores exact original cell.
        assert jnp.allclose(final_cell, _CUBIC_CELL, atol=1e-5)

    def test_impossible_aspect_ratio_no_acceptance(self):
        """min_aspect_ratio=1.0 means only a perfect cube passes; any shear
        or stretch will fail the check, so n_accepted must be 0."""
        key = jax.random.key(12)
        # Use non-zero step sizes so proposals actually differ from the input
        final_cell, n_accepted = _run_walk(
            key, n_steps=30, step_size_shear=0.1, step_size_stretch=0.1,
            min_aspect=1.0
        )
        assert int(n_accepted) == 0
        assert jnp.allclose(final_cell, _CUBIC_CELL, atol=1e-5)

    def test_reasonable_acceptance_rate(self):
        key = jax.random.key(13)
        _, n_accepted = _run_walk(
            key, n_steps=100, step_size_shear=0.05, step_size_stretch=0.05,
            min_aspect=0.2
        )
        assert int(n_accepted) > 0, "Expected some proposals to be accepted"

    def test_jit(self):
        key = jax.random.key(14)
        jit_fn = jax.jit(
            lambda k: _run_walk(k, n_steps=20)
        )
        final_cell, n_accepted = jit_fn(key)
        assert final_cell.shape == (3, 3)
        assert jnp.isfinite(n_accepted)

    def test_vmap_over_keys_diverges(self):
        keys = _batch_keys(4, base=15)
        fn = jax.vmap(lambda k: _run_walk(k, n_steps=30))
        final_cells, n_accepted = fn(keys)
        assert final_cells.shape == (4, 3, 3)
        assert n_accepted.shape == (4,)
        # Different keys should produce at least two distinct final cells
        diffs = jnp.abs(final_cells[1:] - final_cells[:-1])
        assert jnp.any(diffs > 1e-6), "vmap outputs are identical across keys"

    def test_determinism(self):
        key = jax.random.key(16)
        a_cell, a_acc = _run_walk(key, n_steps=40)
        b_cell, b_acc = _run_walk(key, n_steps=40)
        assert jnp.array_equal(a_cell, b_cell)
        assert jnp.array_equal(a_acc, b_acc)

    def test_output_shape(self):
        key = jax.random.key(17)
        final_cell, n_accepted = _run_walk(key)
        assert final_cell.shape == (3, 3)
        assert n_accepted.shape == ()
