"""Tests for the backend-agnostic SoftCoreBackend wrapper.

The wrapper adds a fixed (parameter-free) repulsive Morse term to any
EnergyBackend.  Tests confirm:

1. The pure-function ``_softcore_energy`` matches the analytic Morse
   formula on hand-built two-atom configurations.
2. Parity with the NeuralIL ``FixedRepulsiveMorse`` ground truth for a
   small cluster (when neuralil is installed).
3. Far-apart atoms contribute exactly zero (cutoff respected).
4. The wrapper preserves the base backend's energy at large distances.
5. ``jax.jit`` and ``jax.vmap`` are both compatible.
6. Composition with ``EnsembleBackend`` (NPT stack) returns
   ``U + E_core + P * V``.
7. ``__getattr__`` forwards unknown attributes to the base backend.
8. Close-contact pairs incur a strong, finite penalty.
9. The schema mutex on NeuralILBackendSpec rejects the
   ``softcore: true`` + ``softcore_repulsion: {...}`` combo.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxrens.backends.softcore import (
    DEFAULT_SOFTCORE_KWARGS,
    SoftCoreBackend,
    _smooth_cutoff,
    _softcore_energy,
)
from jaxrens.backends.toy import HarmonicBackend


A0 = DEFAULT_SOFTCORE_KWARGS["a0"]
B0 = DEFAULT_SOFTCORE_KWARGS["b0"]
D0 = DEFAULT_SOFTCORE_KWARGS["d0"]
R_CUT = DEFAULT_SOFTCORE_KWARGS["r_core_cut"]
R_SWITCH = DEFAULT_SOFTCORE_KWARGS["r_core_switch"]


def _two_atoms_at(r: float) -> jnp.ndarray:
    return jnp.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0]], dtype=jnp.float32)


def _cell(L: float = 10.0) -> jnp.ndarray:
    return L * jnp.eye(3, dtype=jnp.float32)


def _species(n: int = 2) -> jnp.ndarray:
    return jnp.zeros(n, dtype=jnp.int32)


# ---------------------------------------------------------------------------
# Pure-function tests
# ---------------------------------------------------------------------------


class TestSoftCoreEnergy:
    @pytest.mark.parametrize("r", [0.3, 0.5, 0.7, 0.9, 1.1])
    def test_matches_analytic_formula(self, r):
        """E_pair(r) = phi(r) * smooth_cutoff(r) for a two-atom config.

        With the (N, N) double-sum and the 0.5 prefactor the total energy
        equals exactly ``phi(r) * smooth_cutoff(r)`` for one pair.
        """
        positions = _two_atoms_at(r)
        E = float(
            _softcore_energy(
                positions, jnp.zeros((3, 3)), A0, B0, D0, R_SWITCH, R_CUT,
            )
        )
        phi = D0 * np.exp(-2.0 * A0 * (r - B0))
        # Below r_switch the cutoff is exactly 1.
        if r < R_SWITCH:
            expected = phi
        else:
            cutoff = float(_smooth_cutoff(jnp.asarray(r), R_SWITCH, R_CUT))
            expected = phi * cutoff
        assert E == pytest.approx(expected, rel=1e-5, abs=1e-6)

    def test_zero_past_cutoff(self):
        """E = 0 (bit-exact) for atoms more than r_core_cut apart."""
        for r in (1.3, 1.5, 2.0, 5.0):
            positions = _two_atoms_at(r)
            E = float(
                _softcore_energy(
                    positions, jnp.zeros((3, 3)), A0, B0, D0, R_SWITCH, R_CUT,
                )
            )
            assert E == 0.0, f"r={r}: expected 0, got {E}"

    def test_diagonal_is_masked(self):
        """A configuration with a single atom yields zero (no self-pair)."""
        positions = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32)
        E = float(
            _softcore_energy(
                positions, jnp.zeros((3, 3)), A0, B0, D0, R_SWITCH, R_CUT,
            )
        )
        assert E == 0.0

    def test_mic_catches_across_boundary_pair(self):
        """Minimum-image distance counts a pair straddling a cell boundary.

        Two atoms at x=0.08 and x=3.92 in a 4 Å cell are 0.16 Å apart by
        MIC but 3.84 Å in raw coordinates.  The periodic path must penalise
        them; the non-periodic (zero-cell) path must see ~0.
        """
        L = 4.0
        cell = L * jnp.eye(3, dtype=jnp.float32)
        pos = jnp.array(
            [[0.08, 0.0, 0.0], [3.92, 0.0, 0.0]], dtype=jnp.float32,
        )
        e_mic = float(
            _softcore_energy(pos, cell, A0, B0, D0, R_SWITCH, R_CUT)
        )
        e_raw = float(
            _softcore_energy(
                pos, jnp.zeros((3, 3)), A0, B0, D0, R_SWITCH, R_CUT,
            )
        )
        # MIC distance 0.16 Å < r_switch → full Morse term.
        expected = D0 * np.exp(-2.0 * A0 * (0.16 - B0))
        assert e_mic == pytest.approx(expected, rel=1e-3)
        # Raw distance 3.84 Å > r_cut → contributes ~nothing (float32 noise
        # in the smooth cutoff far past r_cut, vs ~293 eV from MIC).
        assert e_raw < 1e-6
        assert e_mic > 1e6 * e_raw

    def test_mic_path_jit_compatible(self):
        """The periodic (MIC) branch compiles under jax.jit."""
        cell = 4.0 * jnp.eye(3, dtype=jnp.float32)
        pos = jnp.array(
            [[0.08, 0.0, 0.0], [3.92, 0.0, 0.0]], dtype=jnp.float32,
        )
        f = jax.jit(
            lambda p, c: _softcore_energy(p, c, A0, B0, D0, R_SWITCH, R_CUT)
        )
        assert float(f(pos, cell)) == pytest.approx(
            float(_softcore_energy(pos, cell, A0, B0, D0, R_SWITCH, R_CUT)),
            rel=1e-5,
        )

    def test_smooth_cutoff_endpoints(self):
        """smooth_cutoff is exactly 1 below r_switch, 0 above r_cut."""
        # Below r_switch: exactly 1
        for r in (0.1, 0.5, 0.74):
            assert float(_smooth_cutoff(jnp.asarray(r), R_SWITCH, R_CUT)) \
                == pytest.approx(1.0, abs=1e-6)
        # Above r_cut: exactly 0
        for r in (1.26, 1.5, 5.0):
            assert float(_smooth_cutoff(jnp.asarray(r), R_SWITCH, R_CUT)) \
                == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Wrapper tests
# ---------------------------------------------------------------------------


class TestSoftCoreBackendWrapper:
    def test_base_preserved_past_cutoff(self):
        """Atoms past r_core_cut → wrapped energy equals base alone."""
        base = HarmonicBackend(k=2.0)
        wrapped = SoftCoreBackend(base)
        positions = _two_atoms_at(2.0)  # > R_CUT
        species = _species(2)
        cell = _cell()

        E_base, _, _ = base(positions, species, cell, 0).legacy()
        E_wrapped, _, _ = wrapped(positions, species, cell, 0).legacy()
        assert float(E_wrapped) == pytest.approx(float(E_base), rel=1e-6)

    def test_close_contact_penalty(self):
        """Atoms at r=0.3 incur a large finite penalty over the base."""
        base = HarmonicBackend(k=0.0)  # zero-out the base
        wrapped = SoftCoreBackend(base)
        positions = _two_atoms_at(0.3)
        species = _species(2)
        cell = _cell()

        E_wrapped, _, _ = wrapped(positions, species, cell, 0).legacy()
        # phi(0.3) = exp(-2 * (0.3 - 3.0)) = exp(5.4) ≈ 221
        expected = D0 * np.exp(-2.0 * A0 * (0.3 - B0))
        assert float(E_wrapped) == pytest.approx(expected, rel=1e-4)

    def test_wrapper_threads_cell_for_mic(self):
        """SoftCoreBackend.__call__ passes cell → catches a boundary pair."""
        base = HarmonicBackend(k=0.0)  # isolate the soft-core term
        wrapped = SoftCoreBackend(base)
        cell = 4.0 * jnp.eye(3, dtype=jnp.float32)
        species = _species(2)
        # 0.16 Å apart by MIC across the x-boundary; 3.84 Å raw.
        pos = jnp.array(
            [[0.08, 0.0, 0.0], [3.92, 0.0, 0.0]], dtype=jnp.float32,
        )
        E, _, _ = wrapped(pos, species, cell, 0).legacy()
        expected = D0 * np.exp(-2.0 * A0 * (0.16 - B0))
        assert float(E) == pytest.approx(expected, rel=1e-3)

    def test_jit_compatible(self):
        """The wrapped __call__ runs under jax.jit and matches eager."""
        base = HarmonicBackend(k=1.0)
        wrapped = SoftCoreBackend(base)
        positions = _two_atoms_at(0.5)
        species = _species(2)
        cell = _cell()

        def go(pos):
            E, _, _ = wrapped(pos, species, cell, 0).legacy()
            return E

        E_eager = float(go(positions))
        E_jit = float(jax.jit(go)(positions))
        assert E_jit == pytest.approx(E_eager, rel=1e-5)

    def test_vmap_compatible(self):
        """vmap over a batch of (K, N, 3) walker positions works."""
        base = HarmonicBackend(k=1.0)
        wrapped = SoftCoreBackend(base)
        species = _species(2)
        cell = _cell()

        batch = jnp.stack(
            [_two_atoms_at(r) for r in (0.5, 1.0, 2.0)], axis=0,
        )

        def one(pos):
            E, _, _ = wrapped(pos, species, cell, 0).legacy()
            return E

        Es = jax.vmap(one)(batch)
        assert Es.shape == (3,)
        # The r=2.0 case should equal the base alone
        E_base_far, _, _ = base(batch[2], species, cell, 0).legacy()
        assert float(Es[2]) == pytest.approx(float(E_base_far), rel=1e-5)

    def test_attr_forwarding(self):
        """Unknown attribute lookups fall through to the base."""
        base = HarmonicBackend(k=3.0)
        wrapped = SoftCoreBackend(base)
        # ``r_cutoff`` is mirrored explicitly.
        assert wrapped.r_cutoff == base.r_cutoff
        # ``k`` is base-only; __getattr__ should find it.
        assert wrapped.k == 3.0

    def test_compose_with_ensemble_backend(self):
        """EnsembleBackend(SoftCoreBackend(base)) returns U + E_core + P*V."""
        from jaxrens.backends.ensemble import EnsembleBackend
        from jaxrens.utils.cell import get_volume

        base = HarmonicBackend(k=1.0)
        stacked = EnsembleBackend(
            SoftCoreBackend(base), pressure=0.5,
        )
        positions = _two_atoms_at(0.5)
        species = _species(2)
        cell = _cell()

        E, _, _ = stacked(positions, species, cell, 0).legacy()

        # Reconstruct expected.
        E_U = 0.5 * 1.0 * float(jnp.sum(positions**2))
        E_core = float(
            _softcore_energy(positions, cell, A0, B0, D0, R_SWITCH, R_CUT)
        )
        V = float(get_volume(cell))
        expected = E_U + E_core + 0.5 * V
        assert float(E) == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# Parity with the NeuralIL FixedRepulsiveMorse (when neuralil is installed)
# ---------------------------------------------------------------------------


try:
    from neuralil.softcore.model import FixedRepulsiveMorse  # noqa: F401

    _FIXED_MORSE_AVAILABLE = True
except ImportError:
    _FIXED_MORSE_AVAILABLE = False


fixed_morse_required = pytest.mark.skipif(
    not _FIXED_MORSE_AVAILABLE,
    reason="neuralil.softcore.model not importable",
)


@fixed_morse_required
class TestParityWithFixedRepulsiveMorse:
    def test_two_atom_parity(self):
        """At a single intra-cell distance the wrapper matches FixedRepulsiveMorse.

        FixedRepulsiveMorse returns per-atom contributions; total = sum.
        For one pair at distance r both atoms see 0.5 * phi(r) so the sum
        equals phi(r) — the same value the wrapper's pure function
        returns.
        """
        from neuralil.softcore.model import FixedRepulsiveMorse

        r = 0.5
        positions = _two_atoms_at(r)
        radii = jnp.array(
            [[0.0, r], [r, 0.0]], dtype=jnp.float32,
        )
        types = jnp.array([0, 1], dtype=jnp.int32)
        all_types = types

        softcore = FixedRepulsiveMorse(A0, B0, D0, R_CUT, R_SWITCH)
        morse_atomic = softcore.apply(
            {}, radii, types, all_types,
            method=softcore.calc_atomic_energies,
        )
        ref_total = float(jnp.sum(morse_atomic))
        # FixedRepulsiveMorse works from raw radii (no PBC), so compare
        # against the non-periodic (zero-cell) path.
        wrapper_total = float(
            _softcore_energy(
                positions, jnp.zeros((3, 3)), A0, B0, D0, R_SWITCH, R_CUT,
            )
        )
        assert wrapper_total == pytest.approx(ref_total, rel=1e-5)


# ---------------------------------------------------------------------------
# Schema mutex test
# ---------------------------------------------------------------------------


class TestSchemaMutex:
    def test_neuralil_spec_rejects_both_softcore_flags(self):
        """NeuralILBackendSpec rejects softcore=True + softcore_repulsion={}."""
        from pydantic import ValidationError

        from jaxrens.cli.schema.backend import NeuralILBackendSpec

        with pytest.raises(ValidationError, match="not both"):
            NeuralILBackendSpec(
                checkpoint_path="/tmp/nonexistent.pkl",
                softcore=True,
                softcore_repulsion={
                    "a0": 1.0, "b0": 3.0, "d0": 1.0,
                    "r_core_cut": 1.25, "r_core_switch": 0.75,
                },
            )

    def test_softcore_spec_rejects_inverted_cutoffs(self):
        """SoftCoreSpec rejects r_core_switch >= r_core_cut."""
        from pydantic import ValidationError

        from jaxrens.cli.schema.backend import SoftCoreSpec

        with pytest.raises(ValidationError, match="strictly"):
            SoftCoreSpec(r_core_switch=1.5, r_core_cut=1.0)

    def test_backend_config_carries_softcore_dict(self):
        """A spec with softcore_repulsion ends up in BackendConfig as a dict."""
        from jaxrens.cli.schema.backend import LJBackendSpec

        spec = LJBackendSpec(
            epsilon=1.0,
            sigma=1.0,
            softcore_repulsion={
                "a0": 2.0, "b0": 3.0, "d0": 1.0,
                "r_core_cut": 1.25, "r_core_switch": 0.75,
            },
        )
        cfg = spec.to_backend_config()
        assert cfg.softcore_repulsion is not None
        assert cfg.softcore_repulsion["a0"] == 2.0
        assert cfg.softcore_repulsion["r_core_cut"] == 1.25

    def test_backend_config_softcore_none_by_default(self):
        from jaxrens.cli.schema.backend import LJBackendSpec

        spec = LJBackendSpec(epsilon=1.0, sigma=1.0)
        cfg = spec.to_backend_config()
        assert cfg.softcore_repulsion is None
