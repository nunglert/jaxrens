"""Unit tests for jaxrens.init.rejection.

Covers rejection_sample_positions and the internal pair-distance check.
All JIT-compatible functions are tested under jax.jit per project policy.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jaxrens.init.rejection import _check_min_distance, rejection_sample_positions


# ---------------------------------------------------------------------------
# Harmonic backend fixture
# ---------------------------------------------------------------------------

class _HarmonicBackend:
    """Minimal harmonic backend for tests; k=1.0."""

    r_cutoff = 0.0

    def __call__(self, positions, species, cell, max_neighbors, ensemble_params=None):
        energy = 0.5 * jnp.sum(positions ** 2)
        return energy, 0, False


_BACKEND = _HarmonicBackend()
_CELL = 6.0 * jnp.eye(3, dtype=jnp.float32)
_TYPES = jnp.zeros((4,), dtype=jnp.int32)


# ---------------------------------------------------------------------------
# _check_min_distance
# ---------------------------------------------------------------------------

class TestCheckMinDistance:
    def test_well_separated_atoms(self):
        positions = jnp.array([
            [0.5, 0.5, 0.5],
            [3.0, 3.0, 3.0],
        ], dtype=jnp.float32)
        too_close = _check_min_distance(positions, _CELL, min_distance=1.0)
        assert not bool(too_close)

    def test_overlapping_atoms(self):
        positions = jnp.array([
            [0.5, 0.5, 0.5],
            [0.6, 0.5, 0.5],
        ], dtype=jnp.float32)
        too_close = _check_min_distance(positions, _CELL, min_distance=1.0)
        assert bool(too_close)

    def test_jit(self):
        positions = jnp.array([[1.0, 1.0, 1.0], [4.0, 4.0, 4.0]], dtype=jnp.float32)
        jit_fn = jax.jit(_check_min_distance, static_argnames=("min_distance",))
        result = jit_fn(positions, _CELL, min_distance=1.0)
        assert result.shape == ()


# ---------------------------------------------------------------------------
# rejection_sample_positions: success case
# ---------------------------------------------------------------------------

class TestRejectionSampleSuccess:
    def test_uniform_mode_succeeds(self):
        key = jax.random.key(42)
        positions, energy = rejection_sample_positions(
            key,
            cell=_CELL,
            types=_TYPES,
            n_atoms=4,
            energy_fn=_BACKEND,
            start_energy_ceiling=1e6,
            min_distance=0.5,
            max_tries=50,
            mode="uniform",
            grid_distance=1.5,
        )
        assert positions.shape == (4, 3)
        assert jnp.isfinite(energy)
        assert float(energy) <= 1e6

    def test_grid_mode_succeeds_first_try(self):
        key = jax.random.key(7)
        positions, energy = rejection_sample_positions(
            key,
            cell=_CELL,
            types=_TYPES,
            n_atoms=4,
            energy_fn=_BACKEND,
            start_energy_ceiling=1e6,
            min_distance=1.5,
            max_tries=5,
            mode="grid",
            grid_distance=1.5,
        )
        assert positions.shape == (4, 3)
        assert jnp.isfinite(energy)

    def test_result_respects_energy_ceiling(self):
        key = jax.random.key(10)
        ceiling = 1e3
        positions, energy = rejection_sample_positions(
            key,
            cell=_CELL,
            types=_TYPES,
            n_atoms=4,
            energy_fn=_BACKEND,
            start_energy_ceiling=ceiling,
            min_distance=0.5,
            max_tries=100,
            mode="uniform",
            grid_distance=1.5,
        )
        assert float(energy) <= ceiling


# ---------------------------------------------------------------------------
# rejection_sample_positions: failure cases
# ---------------------------------------------------------------------------

class TestRejectionSampleFailure:
    def test_energy_ceiling_too_low_raises(self):
        key = jax.random.key(0)
        with pytest.raises(RuntimeError) as exc_info:
            rejection_sample_positions(
                key,
                cell=_CELL,
                types=_TYPES,
                n_atoms=4,
                energy_fn=_BACKEND,
                start_energy_ceiling=1e-10,
                min_distance=0.5,
                max_tries=20,
                mode="uniform",
                grid_distance=1.5,
            )
        msg = str(exc_info.value)
        assert "n_energy_over_ceiling" in msg
        assert "n_nan_energy" in msg
        assert "n_atoms_too_close" in msg
        # All 20 attempts should be accounted for
        import re
        numbers = [int(x) for x in re.findall(r"\d+", msg) if int(x) <= 20]
        total = sum(
            int(x) for x in re.findall(r"n_energy_over_ceiling': (\d+)", msg)
        ) + sum(
            int(x) for x in re.findall(r"n_nan_energy': (\d+)", msg)
        ) + sum(
            int(x) for x in re.findall(r"n_atoms_too_close': (\d+)", msg)
        )
        assert total == 20

    def test_min_distance_too_large_raises(self):
        key = jax.random.key(1)
        # min_distance = 4.0 is larger than half the cell edge (3.0), so all
        # atom pairs will appear too close under minimum-image convention.
        with pytest.raises(RuntimeError) as exc_info:
            rejection_sample_positions(
                key,
                cell=_CELL,
                types=_TYPES,
                n_atoms=4,
                energy_fn=_BACKEND,
                start_energy_ceiling=1e6,
                min_distance=4.0,
                max_tries=20,
                mode="uniform",
                grid_distance=1.5,
            )
        msg = str(exc_info.value)
        assert "n_atoms_too_close" in msg
        n_close = int(__import__("re").search(r"n_atoms_too_close': (\d+)", msg).group(1))
        assert n_close > 0

    def test_error_message_sums_to_max_tries(self):
        key = jax.random.key(2)
        max_tries = 15
        with pytest.raises(RuntimeError) as exc_info:
            rejection_sample_positions(
                key,
                cell=_CELL,
                types=_TYPES,
                n_atoms=4,
                energy_fn=_BACKEND,
                start_energy_ceiling=1e-10,
                min_distance=0.5,
                max_tries=max_tries,
                mode="uniform",
                grid_distance=1.5,
            )
        import re
        msg = str(exc_info.value)
        e_count = int(re.search(r"n_energy_over_ceiling': (\d+)", msg).group(1))
        nan_count = int(re.search(r"n_nan_energy': (\d+)", msg).group(1))
        close_count = int(re.search(r"n_atoms_too_close': (\d+)", msg).group(1))
        assert e_count + nan_count + close_count == max_tries


