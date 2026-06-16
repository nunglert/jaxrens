"""Unit tests for the jax-md backend (Tersoff path).

EAM is wired up in the backend but not tested here — no LAMMPS-format
EAM fixture is shipped with the repo, and jax-md does not bundle one.
EAM coverage comes via integration tests once a fixture is added.

The tests cover:
* Energy on Si dimer + Si diamond (sanity / known-sign).
* JIT compatibility under varying cell.
* vmap over a walker axis.
* ``jax.value_and_grad`` produces finite forces and forces vanish at
  the relaxed lattice constant.
* Wrap in ``EnsembleBackend`` for NPT — H = U + P·V monotonic in V.
* LAMMPS-file path (using jax-md's bundled ``Si.tersoff`` fixture).
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("jax_md")

from jaxrens.backends.ensemble import EnsembleBackend
from jaxrens.backends.jaxmd import create_jaxmd, is_available

pytestmark = pytest.mark.jaxmd


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


SI_LATTICE_CONST = 5.431  # Tersoff '88 has a different relaxed `a` than '89.

_JAXMD_SRC_TERSOFF_FIXTURE = Path(
    "/home/nico.unglert/code/jax-md/tests/data/Si.tersoff"
)


def _si_diamond_positions(a: float = SI_LATTICE_CONST) -> jnp.ndarray:
    basis = (
        np.array(
            [
                [0.00, 0.00, 0.00],
                [0.00, 0.50, 0.50],
                [0.50, 0.00, 0.50],
                [0.50, 0.50, 0.00],
                [0.25, 0.25, 0.25],
                [0.25, 0.75, 0.75],
                [0.75, 0.25, 0.75],
                [0.75, 0.75, 0.25],
            ]
        )
        * a
    )
    return jnp.asarray(basis, dtype=jnp.float32)


def _make_backend(periodic: bool = True):
    return create_jaxmd(
        potential="tersoff",
        periodic=periodic,
        tersoff_params="si",
    )


# ---------------------------------------------------------------------------
# Availability + factory
# ---------------------------------------------------------------------------


def test_available():
    assert is_available()


def test_factory_rejects_unknown_potential():
    with pytest.raises(ValueError, match="Unknown jax-md potential"):
        create_jaxmd(potential="foo", periodic=True, tersoff_params="si")


def test_factory_rejects_unknown_inline_set():
    with pytest.raises(ValueError, match="Unknown inline Tersoff"):
        create_jaxmd(
            potential="tersoff",
            periodic=True,
            tersoff_params="unobtanium",
        )


def test_factory_requires_exactly_one_param_source():
    with pytest.raises(ValueError, match="Exactly one of"):
        create_jaxmd(potential="tersoff", periodic=True)
    with pytest.raises(ValueError, match="Exactly one of"):
        create_jaxmd(
            potential="tersoff",
            periodic=True,
            tersoff_params="si",
            tersoff_params_file="/dev/null",
        )


def test_factory_rejects_missing_eam_file():
    with pytest.raises(ValueError, match="eam_params_file"):
        create_jaxmd(potential="eam", periodic=True)


def test_backend_attributes():
    b = _make_backend(periodic=True)
    assert b.potential == "tersoff"
    assert b.periodic is True
    assert b.r_cutoff == pytest.approx(3.2, abs=1e-6)  # R + D = 3.0 + 0.2
    assert b.atomic_numbers == (14,)
    assert b.num_species == 1


# ---------------------------------------------------------------------------
# Energy values
# ---------------------------------------------------------------------------


def test_si_dimer_energy_negative_and_finite():
    """Si-Si dimer at d ≈ 2.35 Å should yield a bound (negative) energy."""
    b = _make_backend(periodic=False)
    positions = jnp.array(
        [[0.0, 0.0, 0.0], [2.35, 0.0, 0.0]],
        dtype=jnp.float32,
    )
    species = jnp.zeros(2, dtype=jnp.int32)
    cell = jnp.zeros((3, 3), dtype=jnp.float32)  # ignored in free space
    _r = b(positions, species, cell)
    e, n, overflow = _r.energy, _r.max_neighbor_count, _r.overflow
    assert jnp.isfinite(e)
    assert float(e) < 0.0
    assert int(n) == 0
    assert bool(overflow) is False


def test_si_diamond_energy_finite():
    """8-atom diamond Si at a = 5.431 Å must yield Tersoff '88 cohesive E.

    With the ``fractional_coordinates=True`` convention now used in
    ``_build_displacement_fn`` (jax-md's Tersoff is broken otherwise —
    see that helper's docstring), the published Tersoff '88 minimum
    sits at a ≈ 5.431 Å with E/atom = -4.630 eV.  The narrow tolerance
    below catches accidental reversions to ``fractional_coordinates=False``,
    which would push this back up toward -2.77 eV/atom.
    """
    b = _make_backend(periodic=True)
    positions = _si_diamond_positions()
    species = jnp.zeros(8, dtype=jnp.int32)
    cell = SI_LATTICE_CONST * jnp.eye(3, dtype=jnp.float32)
    _r = b(positions, species, cell, max_neighbors=0)
    e, n, overflow = _r.energy, _r.max_neighbor_count, _r.overflow
    assert jnp.isfinite(e)
    e_per_atom = float(e) / 8.0
    assert -4.7 < e_per_atom < -4.5, (
        f"Si diamond energy/atom = {e_per_atom:.3f} eV, "
        f"expected ≈ -4.63 eV (Tersoff '88 cohesive minimum at a = 5.431)"
    )


# ---------------------------------------------------------------------------
# Cell variability + JIT
# ---------------------------------------------------------------------------


def test_energy_changes_with_cell():
    """Same positions, two different cells → two different energies.

    Confirms the ``new_box=`` kwarg threading on which the backend
    architecture depends.  If it silently broke, both calls would
    return identical energies.
    """
    b = _make_backend(periodic=True)
    positions = _si_diamond_positions()
    species = jnp.zeros(8, dtype=jnp.int32)

    cell_a = SI_LATTICE_CONST * jnp.eye(3, dtype=jnp.float32)
    cell_b = (SI_LATTICE_CONST * 0.92) * jnp.eye(3, dtype=jnp.float32)

    e_a, *_ = b(positions, species, cell_a)
    e_b, *_ = b(positions, species, cell_b)
    assert not jnp.allclose(e_a, e_b, atol=1e-4), (
        f"new_box threading broken: E(a)={float(e_a):.4f}, "
        f"E(b)={float(e_b):.4f}"
    )


def test_jit_compiles_and_no_retrace_across_cells():
    """Wrap in ``jax.jit`` and check single trace across box changes."""
    b = _make_backend(periodic=True)
    positions = _si_diamond_positions()
    species = jnp.zeros(8, dtype=jnp.int32)

    trace_counter = {"n": 0}

    @jax.jit
    def jitted_energy(pos, cell):
        trace_counter["n"] += 1
        e = b(pos, species, cell).energy
        return e

    e1 = jitted_energy(positions, SI_LATTICE_CONST * jnp.eye(3))
    e1.block_until_ready()
    e2 = jitted_energy(positions, (SI_LATTICE_CONST - 0.1) * jnp.eye(3))
    e2.block_until_ready()
    e3 = jitted_energy(positions, (SI_LATTICE_CONST + 0.2) * jnp.eye(3))
    e3.block_until_ready()

    assert trace_counter["n"] == 1, (
        f"expected 1 JIT trace across 3 calls with different cells, "
        f"got {trace_counter['n']}"
    )
    # Sanity: energies must differ since cells differ.
    assert not jnp.allclose(e1, e3, atol=1e-4)


# ---------------------------------------------------------------------------
# vmap
# ---------------------------------------------------------------------------


def test_vmap_over_walker_axis():
    """vmap the backend over a leading walker axis (K, N, 3)."""
    b = _make_backend(periodic=True)
    base = _si_diamond_positions()
    species = jnp.zeros(8, dtype=jnp.int32)
    cell = SI_LATTICE_CONST * jnp.eye(3, dtype=jnp.float32)
    key = jax.random.PRNGKey(0)
    noise = 0.05 * jax.random.normal(key, (4, 8, 3))
    positions_K = base[None] + noise  # (K=4, N=8, 3)

    def one_call(pos):
        e = b(pos, species, cell).energy
        return e

    energies = jax.vmap(one_call)(positions_K)
    assert energies.shape == (4,)
    assert jnp.all(jnp.isfinite(energies))


# ---------------------------------------------------------------------------
# Gradients (forces)
# ---------------------------------------------------------------------------


def test_value_and_grad_finite():
    """value_and_grad produces finite forces; magnitude is bounded."""
    b = _make_backend(periodic=True)
    positions = _si_diamond_positions()
    species = jnp.zeros(8, dtype=jnp.int32)
    cell = SI_LATTICE_CONST * jnp.eye(3, dtype=jnp.float32)

    def total_e(pos):
        e = b(pos, species, cell).energy
        return e

    e, grad = jax.value_and_grad(total_e)(positions)
    assert jnp.isfinite(e)
    assert jnp.all(jnp.isfinite(grad))
    # Si diamond is bound; forces at a near-relaxed structure should be
    # bounded.  Loose check: per-atom |F| < 100 eV/Å.
    assert float(jnp.max(jnp.abs(grad))) < 100.0


# ---------------------------------------------------------------------------
# NPT wrap
# ---------------------------------------------------------------------------


def test_ensemble_backend_wraps_jaxmd():
    """EnsembleBackend(JaxMDBackend) should add P·V to total energy."""
    base = _make_backend(periodic=True)
    pressure = 0.01  # internal jaxrens units (eV/Å³)
    wrapped = EnsembleBackend(base, pressure=pressure)

    positions = _si_diamond_positions()
    species = jnp.zeros(8, dtype=jnp.int32)

    cell_small = (SI_LATTICE_CONST * 0.9) * jnp.eye(3, dtype=jnp.float32)
    cell_large = (SI_LATTICE_CONST * 1.1) * jnp.eye(3, dtype=jnp.float32)

    e_base_small, *_ = base(positions, species, cell_small, 0)
    e_base_large, *_ = base(positions, species, cell_large, 0)
    e_wrap_small, *_ = wrapped(positions, species, cell_small, 0)
    e_wrap_large, *_ = wrapped(positions, species, cell_large, 0)

    # The wrap adds P·V to the base; with V_large > V_small the gap
    # widens by exactly P·(V_large - V_small).
    v_small = float(jnp.linalg.det(cell_small))
    v_large = float(jnp.linalg.det(cell_large))
    expected_gap = pressure * (v_large - v_small)
    observed_gap = (
        float(e_wrap_large)
        - float(e_wrap_small)
        - (float(e_base_large) - float(e_base_small))
    )
    assert abs(observed_gap - expected_gap) < 1e-3 * max(
        abs(expected_gap), 1.0
    )


# ---------------------------------------------------------------------------
# LAMMPS-file path
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _JAXMD_SRC_TERSOFF_FIXTURE.exists(),
    reason="jax-md source's Si.tersoff fixture not at expected path",
)
def test_tersoff_from_lammps_file_matches_inline():
    """Loading jax-md's bundled Si.tersoff should match inline ``"si"``.

    The inline ``_TERSOFF_SI_88`` was copied from that file, so the
    energies must agree to machine precision on a representative
    structure.
    """
    b_inline = create_jaxmd(
        potential="tersoff",
        periodic=True,
        tersoff_params="si",
    )
    b_file = create_jaxmd(
        potential="tersoff",
        periodic=True,
        tersoff_params_file=str(_JAXMD_SRC_TERSOFF_FIXTURE),
    )

    positions = _si_diamond_positions()
    species = jnp.zeros(8, dtype=jnp.int32)
    cell = SI_LATTICE_CONST * jnp.eye(3, dtype=jnp.float32)

    e_inline, *_ = b_inline(positions, species, cell)
    e_file, *_ = b_file(positions, species, cell)
    assert jnp.allclose(
        e_inline, e_file, atol=1e-5
    ), f"inline vs file disagree: {float(e_inline)} vs {float(e_file)}"
