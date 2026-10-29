"""Tests for ``LJBackend`` and ``create_lj``.

They cover the single-species and per-species (Lorentz-Berthelot) paths,
the triclinic minimum-image convention checked against a brute-force
reference, the ``supercell_trafo`` image enumeration, and the startup warning
that ``resolve`` emits when the cell prior allows cells smaller than twice the
cutoff.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxrens.backends.lj import create_lj
from jaxrens.cli.resolve import resolve
from jaxrens.cli.schema.root import RootSpec


def _two_atom_positions(r: float = 1.5) -> jnp.ndarray:
    return jnp.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0]])


def _no_cell() -> jnp.ndarray:
    return jnp.zeros((3, 3))


class TestConstruction:
    """Check that ``create_lj`` stores scalar or per-species parameters and
    rejects inconsistent shapes."""

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


class TestPerSpeciesReducesToScalar:
    """Check that a length-1 or all-equal species table gives exactly the
    scalar energy."""

    def test_length_one_matches_scalar(self):
        pos = _two_atom_positions(r=1.5)
        types = jnp.array([0, 0], dtype=jnp.int32)
        cell = _no_cell()

        scalar_be = create_lj(epsilon=1.0, sigma=1.0)
        table_be = create_lj(epsilon=[1.0], sigma=[1.0])

        e_scalar = scalar_be(pos, types, cell).energy
        e_table = table_be(pos, types, cell).energy
        assert jnp.allclose(e_scalar, e_table, atol=1e-6, rtol=1e-6)

    def test_two_equal_entries_match_scalar(self):
        pos = _two_atom_positions(r=1.5)
        # mixed types but same params
        types = jnp.array([1, 0], dtype=jnp.int32)
        cell = _no_cell()

        scalar_be = create_lj(epsilon=0.7, sigma=0.9)
        table_be = create_lj(epsilon=[0.7, 0.7], sigma=[0.9, 0.9])

        e_scalar = scalar_be(pos, types, cell).energy
        e_table = table_be(pos, types, cell).energy
        assert jnp.allclose(e_scalar, e_table, atol=1e-6, rtol=1e-6)


class TestCompositionDependence:
    """Check that a two-species LJ with distinct ε and σ gives an energy that
    depends on which species form the pair, following the Lorentz-Berthelot
    mixing rules."""

    def test_homo_vs_hetero_pairs_differ(self):
        # ε_AA=1.0, σ_AA=1.0; ε_BB=0.5, σ_BB=1.2, so ε_AB=√0.5, σ_AB=1.1.
        # The pair energies at r=1.5 should all differ.
        pos = _two_atom_positions(r=1.5)
        cell = _no_cell()
        be = create_lj(epsilon=[1.0, 0.5], sigma=[1.0, 1.2])

        e_AA = be(pos, jnp.array([0, 0], dtype=jnp.int32), cell).energy
        e_BB = be(pos, jnp.array([1, 1], dtype=jnp.int32), cell).energy
        e_AB = be(pos, jnp.array([0, 1], dtype=jnp.int32), cell).energy

        assert not jnp.allclose(e_AA, e_BB)
        assert not jnp.allclose(e_AA, e_AB)
        assert not jnp.allclose(e_BB, e_AB)

    def test_lorentz_berthelot_mixing_value(self):
        """Cross-species pair energy follows ε_ij=√(ε_i·ε_j) and
        σ_ij=(σ_i+σ_j)/2."""
        eps_a, eps_b = 1.0, 0.5
        sig_a, sig_b = 1.0, 1.2
        r = 1.4

        pos = _two_atom_positions(r=r)
        cell = _no_cell()
        be = create_lj(epsilon=[eps_a, eps_b], sigma=[sig_a, sig_b])
        e_AB = be(pos, jnp.array([0, 1], dtype=jnp.int32), cell).energy

        eps_ij = float(jnp.sqrt(eps_a * eps_b))
        sig_ij = 0.5 * (sig_a + sig_b)
        expected = 4.0 * eps_ij * ((sig_ij / r) ** 12 - (sig_ij / r) ** 6)
        assert jnp.allclose(e_AB, expected, atol=1e-5, rtol=1e-5)


class TestJIT:
    """Check that the scalar and the per-species paths compile under
    ``jax.jit``."""

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


def test_two_atom_energy_matches_closed_form():
    eps, sig, r = 1.0, 1.0, 1.2
    be = create_lj(epsilon=eps, sigma=sig)
    pos = _two_atom_positions(r=r)
    cell = _no_cell()
    e = be(pos, jnp.array([0, 0], dtype=jnp.int32), cell).energy
    expected = 4.0 * eps * ((sig / r) ** 12 - (sig / r) ** 6)
    assert jnp.allclose(e, expected, atol=1e-5, rtol=1e-5)


def _brute_force_pair_energy(
    positions, cell, epsilon, sigma, cutoff, supercell
):
    """Reference Python loop computing LJ pair energy with explicit periodic
    images. Used to validate the JAX implementation against a slow but
    obviously-correct baseline."""
    pos = np.asarray(positions)
    cell = np.asarray(cell)
    n = pos.shape[0]
    sc_a, sc_b, sc_c = supercell
    offsets = []
    for a in range(-(sc_a // 2), sc_a // 2 + 1):
        for b in range(-(sc_b // 2), sc_b // 2 + 1):
            for c in range(-(sc_c // 2), sc_c // 2 + 1):
                offsets.append((a, b, c))
    offsets = np.asarray(offsets, dtype=float)

    inv_cell = np.linalg.inv(cell)
    energy = 0.0
    for i in range(n):
        for j in range(n):
            dr = pos[j] - pos[i]
            # Triclinic MIC.
            dr_frac = dr @ inv_cell
            dr_mic = dr - np.round(dr_frac) @ cell
            for k, off in enumerate(offsets):
                if i == j and np.all(off == 0):
                    continue
                dr_kij = dr_mic + off @ cell
                r2 = float((dr_kij**2).sum())
                if cutoff is not None and r2 >= cutoff**2:
                    continue
                sig_r6 = (sigma**2 / r2) ** 3
                pair = 4.0 * epsilon * (sig_r6**2 - sig_r6)
                energy += 0.5 * pair
    return energy


class TestTriclinicMIC:
    """Check that sheared cells use the true nearest image, by comparing with
    a brute-force reference, that a cubic cell gives the diagonal result, and
    that the energy is invariant under large lattice translations."""

    def test_sheared_cell_matches_brute_force(self):
        # 8-atom random configuration in a sheared cell.
        key = jax.random.PRNGKey(0)
        cell = jnp.array(
            [
                [5.0, 0.5, 0.0],
                [0.6, 5.0, 0.3],
                [0.2, 0.4, 5.0],
            ]
        )
        positions = jax.random.uniform(key, (8, 3), minval=-1.0, maxval=6.0)

        be = create_lj(epsilon=1.0, sigma=1.0, cutoff=2.5)
        e = be(positions, jnp.zeros(8, dtype=jnp.int32), cell).energy
        ref = _brute_force_pair_energy(
            positions,
            cell,
            1.0,
            1.0,
            2.5,
            (1, 1, 1),
        )
        assert jnp.allclose(e, ref, atol=1e-4, rtol=1e-4)

    def test_diagonal_cell_unchanged(self):
        # In a cubic cell the triclinic MIC reduces to the diagonal one.
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.2, 0.0, 0.0],
                [0.0, 1.5, 0.0],
            ]
        )
        cell = jnp.diag(jnp.array([10.0, 10.0, 10.0]))
        be = create_lj(epsilon=1.0, sigma=1.0, cutoff=2.5)
        e = be(positions, jnp.zeros(3, dtype=jnp.int32), cell).energy
        ref = _brute_force_pair_energy(
            positions,
            cell,
            1.0,
            1.0,
            2.5,
            (1, 1, 1),
        )
        assert jnp.allclose(e, ref, atol=1e-5, rtol=1e-5)

    def test_invariant_under_large_lattice_translations(self):
        # NS walkers are not wrapped, so |x| >> L is routine. The MIC
        # matmuls must keep full precision there: under the GPU default
        # (TF32-like) matmul precision the error was ~1e-2.
        key_pos, key_shift = jax.random.split(jax.random.PRNGKey(1))
        cell = jnp.eye(3) * 6.0034575  # not exactly representable in TF32
        positions = jax.random.uniform(key_pos, (8, 3), minval=0.0, maxval=6.0)
        shifts = jax.random.randint(key_shift, (8, 3), -8, 9)
        shifted = positions + shifts.astype(jnp.float32) @ cell

        be = create_lj(epsilon=1.0, sigma=1.0, cutoff=2.5)
        types = jnp.zeros(8, dtype=jnp.int32)
        e = jax.jit(lambda p: be(p, types, cell).energy)
        assert jnp.allclose(e(shifted), e(positions), atol=1e-3)


class TestSupercellTrafo:
    """Check that ``supercell_trafo`` adds periodic images: extra images
    change nothing in a spacious cell, a tight cell needs them, the backend
    jits, and an entry below 1 is rejected."""

    def test_extra_images_beyond_cutoff_change_nothing(self):
        # Spacious cubic cell: (1,1,1) and (2,2,2) must agree because every
        # extra image is beyond the cutoff.
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.4, 0.0, 0.0],
                [0.0, 1.4, 0.0],
            ]
        )
        cell = jnp.diag(jnp.array([10.0, 10.0, 10.0]))
        be_a = create_lj(
            epsilon=1.0, sigma=1.0, cutoff=2.5, supercell_trafo=(1, 1, 1)
        )
        be_b = create_lj(
            epsilon=1.0, sigma=1.0, cutoff=2.5, supercell_trafo=(2, 2, 2)
        )
        e_a = be_a(positions, jnp.zeros(3, dtype=jnp.int32), cell).energy
        e_b = be_b(positions, jnp.zeros(3, dtype=jnp.int32), cell).energy
        assert jnp.allclose(e_a, e_b, atol=1e-5, rtol=1e-5)

    def test_tight_cell_picks_up_images(self):
        # The cell side 3.55 is below 2 * cutoff = 5, so (2, 2, 2) images
        # are needed for the correct energy.
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.2, 0.0, 0.0],
            ]
        )
        cell = jnp.diag(jnp.array([3.55, 3.55, 3.55]))
        be_mic = create_lj(
            epsilon=1.0, sigma=1.0, cutoff=2.5, supercell_trafo=(1, 1, 1)
        )
        be_sc = create_lj(
            epsilon=1.0, sigma=1.0, cutoff=2.5, supercell_trafo=(2, 2, 2)
        )
        e_mic = be_mic(positions, jnp.zeros(2, dtype=jnp.int32), cell).energy
        e_sc = be_sc(positions, jnp.zeros(2, dtype=jnp.int32), cell).energy
        ref_mic = _brute_force_pair_energy(
            positions,
            cell,
            1.0,
            1.0,
            2.5,
            (1, 1, 1),
        )
        ref_sc = _brute_force_pair_energy(
            positions,
            cell,
            1.0,
            1.0,
            2.5,
            (2, 2, 2),
        )
        assert jnp.allclose(e_mic, ref_mic, atol=1e-4, rtol=1e-4)
        assert jnp.allclose(e_sc, ref_sc, atol=1e-4, rtol=1e-4)
        # The images add real neighbours, so the energies must differ.
        assert not jnp.allclose(e_mic, e_sc, atol=1e-3, rtol=1e-3)

    def test_jit_compatibility(self):
        be = create_lj(
            epsilon=1.0, sigma=1.0, cutoff=2.5, supercell_trafo=(2, 2, 2)
        )

        @jax.jit
        def energy_fn(pos, sp, c):
            e = be(pos, sp, c).energy
            return e

        pos = jnp.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
        e = energy_fn(
            pos,
            jnp.zeros(2, dtype=jnp.int32),
            jnp.diag(jnp.array([4.0, 4.0, 4.0])),
        )
        assert jnp.isfinite(e)

    def test_invalid_supercell_trafo_raises(self):
        with pytest.raises(ValueError, match=">= 1"):
            create_lj(supercell_trafo=(0, 1, 1))


class TestStartupCutoffWarning:
    """Check that ``resolve`` warns when the cell prior of a periodic LJ run
    allows cells smaller than twice the cutoff, and stays quiet for a
    spacious prior, without a cutoff, or for a non-periodic cluster."""

    def _minimal_config(self):
        return {
            "run": {"n_live": 8, "max_iterations": 5, "seed": 42},
            "backend": {
                "type": "lj",
                "epsilon": 1.0,
                "sigma": 1.0,
                "cutoff": 2.5,
                "periodic": True,
            },
            "ensemble": {
                "type": "npt",
                "pressure": 0.1,
                "pressure_units": "eva3",
            },
            "moves": [
                {
                    "type": "gmc",
                    "n_reflect": 4,
                    "step_size": 0.1,
                    "weight": 1.0,
                },
            ],
            "init": {
                "start_species": "18 8",
                "random_initialise_pos": True,
                "pos_randomization_mode": "grid",
                "grid_distance": 1.0,
                "random_initialise_cell": True,
                "initial_walk": {
                    "n_walks": 1,
                    "walklength": 1,
                    "adjust_interval": 1,
                    "emax_offset_per_atom": 1.0,
                },
            },
            "output": {
                "format": "extxyz",
                "working_dir": "./output",
                "out_file_prefix": "lj_unsafe_test",
            },
        }

    def test_warns_on_tight_cell_prior(self, caplog):
        cfg = self._minimal_config()
        # Tight cell prior: smallest cube side ≈ (1.5*8)^(1/3) ≈ 2.29 σ,
        # less than 2*cutoff=5.0. Must warn.
        cfg["cell"] = {
            "max_volume_per_atom": 10.0,
            "min_volume_per_atom": 1.5,
            "min_aspect_ratio": 0.6,
            "flat_V_prior": False,
        }
        root = RootSpec.model_validate(cfg)
        with caplog.at_level("WARNING", logger="jaxrens.cli.resolve"):
            resolve(root)
        msgs = [r.getMessage() for r in caplog.records]
        assert any("LJ cutoff vs cell-prior bounds" in m for m in msgs), msgs

    def test_no_warning_on_spacious_cell_prior(self, caplog):
        cfg = self._minimal_config()
        # Generous cell prior: smallest cube side ≈ (50*8)^(1/3) ≈ 7.37 σ,
        # times min_aspect 0.9 = 6.63 σ, comfortably above 2*cutoff=5.0.
        cfg["cell"] = {
            "max_volume_per_atom": 200.0,
            "min_volume_per_atom": 50.0,
            "min_aspect_ratio": 0.9,
            "flat_V_prior": False,
        }
        root = RootSpec.model_validate(cfg)
        with caplog.at_level("WARNING", logger="jaxrens.cli.resolve"):
            resolve(root)
        msgs = [r.getMessage() for r in caplog.records]
        assert not any(
            "LJ cutoff vs cell-prior bounds" in m for m in msgs
        ), msgs

    def test_no_warning_without_cutoff(self, caplog):
        cfg = self._minimal_config()
        cfg["backend"]["cutoff"] = None
        cfg["cell"] = {
            "max_volume_per_atom": 10.0,
            "min_volume_per_atom": 1.5,
            "min_aspect_ratio": 0.6,
            "flat_V_prior": False,
        }
        root = RootSpec.model_validate(cfg)
        with caplog.at_level("WARNING", logger="jaxrens.cli.resolve"):
            resolve(root)
        msgs = [r.getMessage() for r in caplog.records]
        assert not any(
            "LJ cutoff vs cell-prior bounds" in m for m in msgs
        ), msgs

    def test_no_warning_when_non_periodic(self, caplog):
        """A cluster in vacuum has no periodic images, so the interaction
        between cutoff, cell and supercell_trafo that this warning is about
        does not exist, even with the same tight cell prior that triggers it
        when periodic.
        """
        cfg = self._minimal_config()
        cfg["backend"]["periodic"] = False
        # Same tight cell prior as in test_warns_on_tight_cell_prior, which
        # would warn if this were periodic.
        cfg["cell"] = {
            "max_volume_per_atom": 10.0,
            "min_volume_per_atom": 1.5,
            "min_aspect_ratio": 0.6,
            "flat_V_prior": False,
        }
        root = RootSpec.model_validate(cfg)
        with caplog.at_level("WARNING", logger="jaxrens.cli.resolve"):
            resolve(root)
        msgs = [r.getMessage() for r in caplog.records]
        assert not any(
            "LJ cutoff vs cell-prior bounds" in m for m in msgs
        ), msgs
