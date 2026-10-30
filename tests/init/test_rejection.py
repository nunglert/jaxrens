"""Tests for rejection_sample_positions and _check_min_distance.

Both live in jaxrens.init.rejection.  The sampler is run against a minimal
harmonic backend, so the energy of a configuration is known in closed form.
"""

from __future__ import annotations

import re

import jax
import jax.numpy as jnp
import pytest

from jaxrens.backends.base import BackendResult
from jaxrens.init.rejection import (
    _check_min_distance,
    rejection_sample_positions,
)


class _HarmonicBackend:
    """Minimal harmonic backend for tests; k=1.0."""

    r_cutoff = 0.0

    def __call__(
        self, positions, species, cell, max_neighbors, ensemble_params=None
    ):
        energy = 0.5 * jnp.sum(positions**2)
        return BackendResult(energy=energy)


_BACKEND = _HarmonicBackend()
_CELL = 6.0 * jnp.eye(3, dtype=jnp.float32)
_TYPES = jnp.zeros((4,), dtype=jnp.int32)


class TestCheckMinDistance:
    """Check the minimum-image pair-distance test in _check_min_distance.

    The tests cover a well separated pair, an overlapping pair, and the
    function under jax.jit.
    """

    def test_well_separated_atoms(self):
        positions = jnp.array(
            [
                [0.5, 0.5, 0.5],
                [3.0, 3.0, 3.0],
            ],
            dtype=jnp.float32,
        )
        too_close = _check_min_distance(positions, _CELL, min_distance=1.0)
        assert not bool(too_close)

    def test_overlapping_atoms(self):
        positions = jnp.array(
            [
                [0.5, 0.5, 0.5],
                [0.6, 0.5, 0.5],
            ],
            dtype=jnp.float32,
        )
        too_close = _check_min_distance(positions, _CELL, min_distance=1.0)
        assert bool(too_close)

    def test_jit(self):
        positions = jnp.array(
            [[1.0, 1.0, 1.0], [4.0, 4.0, 4.0]], dtype=jnp.float32
        )
        jit_fn = jax.jit(
            _check_min_distance, static_argnames=("min_distance",)
        )
        result = jit_fn(positions, _CELL, min_distance=1.0)
        assert result.shape == ()


class TestRejectionSampleSuccess:
    """Check that rejection_sample_positions returns a valid configuration.

    The tests cover the uniform and grid modes and the energy ceiling.
    """

    @pytest.mark.parametrize(
        "mode, seed, min_distance, max_tries",
        [("uniform", 42, 0.5, 50), ("grid", 7, 1.5, 5)],
    )
    def test_mode_succeeds(self, mode, seed, min_distance, max_tries):
        positions, energy = rejection_sample_positions(
            jax.random.key(seed),
            cell=_CELL,
            types=_TYPES,
            n_atoms=4,
            energy_fn=_BACKEND,
            start_energy_ceiling=1e6,
            min_distance=min_distance,
            max_tries=max_tries,
            mode=mode,
            grid_distance=1.5,
        )
        assert positions.shape == (4, 3)
        assert jnp.isfinite(energy)
        assert float(energy) <= 1e6

    def test_result_respects_energy_ceiling(self):
        key = jax.random.key(10)
        ceiling = 40
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

    @pytest.mark.parametrize("mode", ["uniform", "grid"])
    def test_result_respects_min_distance(self, mode):
        min_distance = 1.5
        n_atoms = 4
        positions, _ = rejection_sample_positions(
            jax.random.key(5),
            cell=_CELL,
            types=_TYPES,
            n_atoms=n_atoms,
            energy_fn=_BACKEND,
            start_energy_ceiling=1e6,
            min_distance=min_distance,
            max_tries=100,
            mode=mode,
            grid_distance=1.5,
        )
        # Minimum image for the cubic cell: wrap each displacement into
        # [-L/2, L/2] along every axis.
        edge = float(_CELL[0, 0])
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                diff = positions[i] - positions[j]
                diff = diff - edge * jnp.round(diff / edge)
                assert float(jnp.linalg.norm(diff)) >= min_distance - 1e-4


class TestRejectionSampleFailure:
    """Check the RuntimeError raised when every attempt is rejected.

    The tests cover an unreachable energy ceiling and a minimum distance
    that is too large, and check that the per-reason counters in the message
    add up to max_tries.
    """

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

    def test_min_distance_too_large_raises(self):
        key = jax.random.key(1)
        # min_distance = 4.0 exceeds half the cell edge (3.0), so under the
        # minimum-image convention most draws have a pair that is too close.
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
        n_close = int(re.search(r"n_atoms_too_close': (\d+)", msg).group(1))
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
        msg = str(exc_info.value)
        e_count = int(
            re.search(r"n_energy_over_ceiling': (\d+)", msg).group(1)
        )
        nan_count = int(re.search(r"n_nan_energy': (\d+)", msg).group(1))
        close_count = int(
            re.search(r"n_atoms_too_close': (\d+)", msg).group(1)
        )
        assert e_count + nan_count + close_count == max_tries
