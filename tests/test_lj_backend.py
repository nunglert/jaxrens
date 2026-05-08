"""Unit tests for ``LJBackend`` — single-species and per-species paths.

Focus areas:
- Backward-compatible scalar API yields the legacy energy.
- Per-species (Lorentz-Berthelot) reduces to scalar when all species are equal.
- Two-species table produces species-dependent energies under composition swaps.
- JIT-compatibility of both paths.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jaxrens.backends.lj import LJBackend, create_lj


def _two_atom_positions(r: float = 1.5) -> jnp.ndarray:
    return jnp.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0]])


def _no_cell() -> jnp.ndarray:
    return jnp.zeros((3, 3))


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_scalar_default(self):
        be = create_lj()
        assert be._per_species is False
        assert be.epsilon == 1.0
        assert be.sigma == 1.0
        assert be.n_species == 1

    def test_per_species_table(self):
        be = create_lj(epsilon=[1.0, 0.8], sigma=[1.0, 1.1])
        assert be._per_species is True
        assert be.n_species == 2
        assert be._eps_table.shape == (2,)
        assert be._sig_table.shape == (2,)

    def test_rank_mismatch_raises(self):
        with pytest.raises(ValueError, match="same rank"):
            create_lj(epsilon=[1.0, 1.0], sigma=1.0)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="matching length"):
            create_lj(epsilon=[1.0, 1.0], sigma=[1.0, 1.0, 1.0])

    def test_empty_table_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            create_lj(epsilon=[], sigma=[])

    def test_high_rank_raises(self):
        with pytest.raises(ValueError, match="scalar or 1-D"):
            create_lj(epsilon=[[1.0, 1.0]], sigma=[[1.0, 1.0]])


# ---------------------------------------------------------------------------
# Energy parity: per-species reduces to scalar when species table is uniform
# ---------------------------------------------------------------------------


class TestPerSpeciesReducesToScalar:
    """A length-1 (or all-equal) table must match the scalar path exactly."""

    def test_length_one_matches_scalar(self):
        pos = _two_atom_positions(r=1.5)
        types = jnp.array([0, 0], dtype=jnp.int32)
        cell = _no_cell()

        scalar_be = create_lj(epsilon=1.0, sigma=1.0)
        table_be = create_lj(epsilon=[1.0], sigma=[1.0])

        e_scalar, _, _ = scalar_be(pos, types, cell)
        e_table, _, _ = table_be(pos, types, cell)
        assert jnp.allclose(e_scalar, e_table, atol=1e-6, rtol=1e-6)

    def test_two_equal_entries_match_scalar(self):
        pos = _two_atom_positions(r=1.5)
        types = jnp.array([1, 0], dtype=jnp.int32)  # mixed types but same params
        cell = _no_cell()

        scalar_be = create_lj(epsilon=0.7, sigma=0.9)
        table_be = create_lj(epsilon=[0.7, 0.7], sigma=[0.9, 0.9])

        e_scalar, _, _ = scalar_be(pos, types, cell)
        e_table, _, _ = table_be(pos, types, cell)
        assert jnp.allclose(e_scalar, e_table, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# Composition dependence: cross-species pair differs from same-species pair
# ---------------------------------------------------------------------------


class TestCompositionDependence:
    """Two-species LJ with distinct ε/σ must produce composition-dependent E."""

    def test_homo_vs_hetero_pairs_differ(self):
        # ε_AA=1.0, σ_AA=1.0; ε_BB=0.5, σ_BB=1.2 → ε_AB=√0.5, σ_AB=1.1
        # Pair energies at r=1.5 should all differ.
        pos = _two_atom_positions(r=1.5)
        cell = _no_cell()
        be = create_lj(epsilon=[1.0, 0.5], sigma=[1.0, 1.2])

        e_AA, _, _ = be(pos, jnp.array([0, 0], dtype=jnp.int32), cell)
        e_BB, _, _ = be(pos, jnp.array([1, 1], dtype=jnp.int32), cell)
        e_AB, _, _ = be(pos, jnp.array([0, 1], dtype=jnp.int32), cell)

        assert not jnp.allclose(e_AA, e_BB), "AA and BB pair energies must differ"
        assert not jnp.allclose(e_AA, e_AB), "AA and AB pair energies must differ"
        assert not jnp.allclose(e_BB, e_AB), "BB and AB pair energies must differ"

    def test_lorentz_berthelot_mixing_value(self):
        """Cross-species pair energy follows ε_ij=√(ε_i·ε_j), σ_ij=(σ_i+σ_j)/2."""
        eps_a, eps_b = 1.0, 0.5
        sig_a, sig_b = 1.0, 1.2
        r = 1.4

        pos = _two_atom_positions(r=r)
        cell = _no_cell()
        be = create_lj(epsilon=[eps_a, eps_b], sigma=[sig_a, sig_b])
        e_AB, _, _ = be(pos, jnp.array([0, 1], dtype=jnp.int32), cell)

        eps_ij = float(jnp.sqrt(eps_a * eps_b))
        sig_ij = 0.5 * (sig_a + sig_b)
        expected = 4.0 * eps_ij * ((sig_ij / r) ** 12 - (sig_ij / r) ** 6)
        assert jnp.allclose(e_AB, expected, atol=1e-5, rtol=1e-5), (
            f"got {float(e_AB):.6f}, expected {expected:.6f}"
        )


# ---------------------------------------------------------------------------
# JIT compatibility
# ---------------------------------------------------------------------------


class TestJIT:
    def test_scalar_path_jits(self):
        be = create_lj(epsilon=1.0, sigma=1.0, cutoff=2.5)
        pos = _two_atom_positions(r=1.5)
        types = jnp.array([0, 0], dtype=jnp.int32)
        cell = _no_cell()

        @jax.jit
        def _energy(p, t, c):
            return be(p, t, c)[0]

        e = _energy(pos, types, cell)
        assert jnp.isfinite(e)

    def test_per_species_path_jits(self):
        be = create_lj(epsilon=[1.0, 0.5], sigma=[1.0, 1.2], cutoff=2.5)
        pos = _two_atom_positions(r=1.5)
        types = jnp.array([0, 1], dtype=jnp.int32)
        cell = _no_cell()

        @jax.jit
        def _energy(p, t, c):
            return be(p, t, c)[0]

        e = _energy(pos, types, cell)
        assert jnp.isfinite(e)


# ---------------------------------------------------------------------------
# Existing scalar behaviour preserved (regression)
# ---------------------------------------------------------------------------


class TestScalarRegression:
    def test_two_atom_energy_matches_closed_form(self):
        eps, sig, r = 1.0, 1.0, 1.2
        be = create_lj(epsilon=eps, sigma=sig)
        pos = _two_atom_positions(r=r)
        cell = _no_cell()
        e, _, _ = be(pos, jnp.array([0, 0], dtype=jnp.int32), cell)
        expected = 4.0 * eps * ((sig / r) ** 12 - (sig / r) ** 6)
        assert jnp.allclose(e, expected, atol=1e-5, rtol=1e-5)
