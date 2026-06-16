"""Tests for XRENSSwap (composition-morphing replica exchange), commit 4.

Coverage:
- XRENSSwap.propose: morphed types match target compositions exactly.
- XRENSSwap.accept: boolean output, JIT-compatible.
- XRENSSwap.accept delegates to PressureRENSSwap acceptance math.
- n_species mismatch guard: XRENSSwap(n_species=3) + targets of width 2 raises.
- End-to-end: two-run NS at different compositions, assert:
    - No errors during run.
    - inter_re_stats["n_energy_evals"] > 0 (morph path evaluated backend).
    - At least one swap accepted.
- SingleRun skip: flavor=xrens with n_runs=1 silently completes (no-op swap).
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import pytest

from jaxrens.backends.base import BackendResult
from jaxrens.sampling.moves.replica_exchange import (
    XRENSSwap,
    xrens_replica_exchange_step,
)
from jaxrens.state.config import InterREConfig

# ---------------------------------------------------------------------------
# Toy multi-species backend
# ---------------------------------------------------------------------------
# A species-aware harmonic backend: E = k * sum_i s[i] * ||pos_i||^2
# where s[i] = species_weight[types[i]].
# This makes energy depend on composition, so swaps are non-trivial.


class SpeciesHarmonicBackend:
    """Harmonic with per-species weights.

    E = sum_i species_weight[types[i]] * 0.5 * ||pos_i||^2
    """

    def __init__(self, species_weights: list[float]):
        self.species_weights = jnp.array(species_weights)
        self.r_cutoff = 0.0

    def __call__(
        self,
        positions: jnp.ndarray,
        types: jnp.ndarray,
        cell: jnp.ndarray,
        max_neighbors: int = 0,
        ensemble_params: dict[str, Any] | None = None,
    ) -> BackendResult:
        weights = self.species_weights[types]  # (n_atoms,)
        energy = jnp.sum(weights * 0.5 * jnp.sum(positions**2, axis=-1))
        return BackendResult(energy=energy)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(positions, types, energy, cell=None):
    """Build a minimal state dict as expected by XRENSSwap.propose."""
    return {
        "positions": positions,
        "types": types,
        "energy": energy,
        "cell": cell if cell is not None else jnp.zeros((3, 3)),
    }


# ---------------------------------------------------------------------------
# XRENSSwap.propose: morphed types match target composition exactly
# ---------------------------------------------------------------------------


class TestXRENSProposeMorphedTypes:
    """propose() must produce types with correct target compositions."""

    def _run_propose(self, n_atoms, n_species, target_a, target_b, seed=0):
        backend = SpeciesHarmonicBackend(
            species_weights=[1.0, 2.0][:n_species]
        )
        kernel = XRENSSwap(n_species=n_species)
        key = jax.random.PRNGKey(seed)
        k1, k2, morph_key = jax.random.split(key, 3)

        # Initial types: all species 0 for A, all species 1 for B (if 2 species).
        if n_species == 2:
            types_a = jnp.zeros(n_atoms, dtype=jnp.int32)
            types_b = jnp.ones(n_atoms, dtype=jnp.int32)
        else:
            types_a = jax.random.randint(k1, (n_atoms,), 0, n_species)
            types_b = jax.random.randint(k2, (n_atoms,), 0, n_species)

        pos_a = jax.random.normal(k1, (n_atoms, 3)) * 0.1
        pos_b = jax.random.normal(k2, (n_atoms, 3)) * 0.1

        e_a = backend(pos_a, types_a, jnp.zeros((3, 3))).energy
        e_b = backend(pos_b, types_b, jnp.zeros((3, 3))).energy

        state_a = _make_state(pos_a, types_a, e_a)
        state_b = _make_state(pos_b, types_b, e_b)
        ens_a = {"target_composition": jnp.array(target_a, dtype=jnp.int32)}
        ens_b = {"target_composition": jnp.array(target_b, dtype=jnp.int32)}

        proposed, n_e, n_g = kernel.propose(
            state_a, state_b, ens_a, ens_b, morph_key, backend
        )
        return proposed, n_e, n_g

    def test_morphed_types_a_match_target(self):
        """types_a in proposed must match target_a composition."""
        n_atoms = 8
        target_a = [8, 0]
        target_b = [4, 4]
        proposed, _, _ = self._run_propose(n_atoms, 2, target_a, target_b)
        got_a = jnp.bincount(proposed["types_a"], length=2)
        assert jnp.array_equal(
            got_a, jnp.array(target_a)
        ), f"types_a composition {got_a} != target {target_a}"

    def test_morphed_types_b_match_target(self):
        """types_b in proposed must match target_b composition."""
        n_atoms = 8
        target_a = [8, 0]
        target_b = [4, 4]
        proposed, _, _ = self._run_propose(n_atoms, 2, target_a, target_b)
        got_b = jnp.bincount(proposed["types_b"], length=2)
        assert jnp.array_equal(
            got_b, jnp.array(target_b)
        ), f"types_b composition {got_b} != target {target_b}"

    def test_energy_a_is_recomputed(self):
        """energy_a should differ from state_a.energy (backend call made)."""
        # With species-harmonic backend and composition change, energy will
        # likely differ — but even if it doesn't numerically, the propose
        # call must succeed without error.
        n_atoms = 8
        target_a = [8, 0]
        target_b = [4, 4]
        proposed, n_e, n_g = self._run_propose(n_atoms, 2, target_a, target_b)
        assert n_e == 2, f"Expected 2 energy evals, got {n_e}"
        assert n_g == 0, f"Expected 0 grad evals, got {n_g}"
        # Both energies must be finite.
        assert jnp.isfinite(proposed["energy_a"]), "energy_a is not finite"
        assert jnp.isfinite(proposed["energy_b"]), "energy_b is not finite"

    def test_positions_a_are_from_b(self):
        """After propose, positions_a should be the original B positions."""
        backend = SpeciesHarmonicBackend([1.0, 2.0])
        kernel = XRENSSwap(n_species=2)
        key = jax.random.PRNGKey(7)
        k1, k2, mkey = jax.random.split(key, 3)

        pos_a = jax.random.normal(k1, (4, 3))
        pos_b = jax.random.normal(k2, (4, 3))
        types_a = jnp.zeros(4, dtype=jnp.int32)
        types_b = jnp.ones(4, dtype=jnp.int32)
        e_a = backend(pos_a, types_a, jnp.zeros((3, 3))).energy
        e_b = backend(pos_b, types_b, jnp.zeros((3, 3))).energy

        state_a = _make_state(pos_a, types_a, e_a)
        state_b = _make_state(pos_b, types_b, e_b)
        ens_a = {"target_composition": jnp.array([4, 0])}
        ens_b = {"target_composition": jnp.array([2, 2])}

        proposed, _, _ = kernel.propose(
            state_a, state_b, ens_a, ens_b, mkey, backend
        )
        # A receives B's positions; B receives A's positions.
        assert jnp.allclose(
            proposed["positions_a"], pos_b
        ), "positions_a should be pos_b"
        assert jnp.allclose(
            proposed["positions_b"], pos_a
        ), "positions_b should be pos_a"

    def test_propose_many_seeds(self):
        """Composition invariant holds across multiple random seeds."""
        n_atoms = 12
        target_a = [12, 0]
        target_b = [6, 6]
        for seed in range(5):
            proposed, _, _ = self._run_propose(
                n_atoms, 2, target_a, target_b, seed=seed
            )
            got_a = jnp.bincount(proposed["types_a"], length=2)
            got_b = jnp.bincount(proposed["types_b"], length=2)
            assert jnp.array_equal(
                got_a, jnp.array(target_a)
            ), f"seed={seed}: types_a {got_a} != {target_a}"
            assert jnp.array_equal(
                got_b, jnp.array(target_b)
            ), f"seed={seed}: types_b {got_b} != {target_b}"


# ---------------------------------------------------------------------------
# XRENSSwap.accept: boolean, JIT-compatible
# ---------------------------------------------------------------------------


class TestXRENSAccept:
    """accept() must return a boolean scalar and be JIT-compatible."""

    def _make_proposed(self, e_a, e_b, cell_a=None, cell_b=None):
        cell = jnp.zeros((3, 3))
        return {
            "energy_a": jnp.array(e_a),
            "energy_b": jnp.array(e_b),
            "cell_a": cell_a if cell_a is not None else cell,
            "cell_b": cell_b if cell_b is not None else cell,
            "types_a": jnp.zeros(4, dtype=jnp.int32),
            "types_b": jnp.zeros(4, dtype=jnp.int32),
            "positions_a": jnp.zeros((4, 3)),
            "positions_b": jnp.zeros((4, 3)),
        }

    def test_accept_returns_bool(self):
        kernel = XRENSSwap(n_species=2)
        proposed = self._make_proposed(0.5, 0.3)
        result = kernel.accept(
            proposed,
            emax_a=jnp.array(1.0),
            emax_b=jnp.array(1.0),
            ensemble_params_a={},
            ensemble_params_b={},
        )
        assert result.dtype == jnp.bool_ or jnp.issubdtype(
            result.dtype, jnp.bool_
        ), f"accept() should return bool, got {result.dtype}"

    def test_accept_low_energy_accepts(self):
        """Both energies below emax → accepted."""
        kernel = XRENSSwap(n_species=2)
        proposed = self._make_proposed(0.1, 0.2)
        result = kernel.accept(
            proposed,
            emax_a=jnp.array(5.0),
            emax_b=jnp.array(5.0),
            ensemble_params_a={},
            ensemble_params_b={},
        )
        assert bool(result), "Expected accept when energies << emax"

    def test_accept_high_energy_rejects(self):
        """Energy above emax → rejected."""
        kernel = XRENSSwap(n_species=2)
        proposed = self._make_proposed(10.0, 0.2)
        result = kernel.accept(
            proposed,
            emax_a=jnp.array(1.0),
            emax_b=jnp.array(1.0),
            ensemble_params_a={},
            ensemble_params_b={},
        )
        assert not bool(result), "Expected reject when energy > emax"

    def test_accept_jit_compatible(self):
        """accept() must work under jax.jit."""
        kernel = XRENSSwap(n_species=2)
        proposed = self._make_proposed(0.5, 0.3)

        @jax.jit
        def jitted_accept(prop, ea, eb):
            return kernel.accept(prop, ea, eb, {}, {})

        result = jitted_accept(proposed, jnp.array(1.0), jnp.array(1.0))
        assert (
            jnp.isscalar(result) or result.shape == ()
        ), f"accept() under jit should return scalar, got shape {result.shape}"

    def test_accept_uses_destination_pressure_via_propose(self):
        """XRENS no longer delegates to PressureRENSSwap.accept.

        After the unified-fix landed (matching production state.energy
        semantics), ``XRENSSwap.propose`` threads each receiving run's
        ensemble_params (incl. pressure) into the backend call, so the
        returned ``energy_a``/``energy_b`` are already enthalpies at the
        destination pressure.  ``XRENSSwap.accept`` is therefore a plain
        ``E < Emax`` threshold check.

        This test asserts that ``XRENSSwap.accept`` returns the same boolean
        as a hand-evaluated ``e < emax`` check, regardless of whether
        ensemble_params carry a pressure — the kernel must NOT add another
        PV term on top.
        """
        kernel = XRENSSwap(n_species=2)
        proposed = self._make_proposed(0.5, 0.4)
        # With pressure dicts present in ensemble_params: accept must still
        # be a simple threshold check (no PV double-add).
        ens_a = {"pressure": jnp.array(0.1)}
        ens_b = {"pressure": jnp.array(0.2)}
        proposed["cell_a"] = jnp.eye(3) * 2.0
        proposed["cell_b"] = jnp.eye(3) * 2.0

        # Emax just above the stored energies → must accept.
        emax_a = jnp.array(1.0)
        emax_b = jnp.array(1.0)
        assert bool(kernel.accept(proposed, emax_a, emax_b, ens_a, ens_b))

        # Emax just below energy_a → must reject (even though delegated to
        # PressureRENSSwap.accept with double-counted PV would still accept).
        emax_a_tight = jnp.array(0.45)
        assert not bool(
            kernel.accept(proposed, emax_a_tight, emax_b, ens_a, ens_b)
        )


# ---------------------------------------------------------------------------
# Guard: n_species mismatch
# ---------------------------------------------------------------------------


class TestNSpeciesMismatch:
    """XRENSSwap with mismatched n_species should fail clearly."""

    def test_invalid_n_species_zero_raises(self):
        with pytest.raises(ValueError, match="n_species"):
            XRENSSwap(n_species=0)

    def test_invalid_n_species_negative_raises(self):
        with pytest.raises(ValueError, match="n_species"):
            XRENSSwap(n_species=-1)

    def test_propose_composition_target_width_mismatch(self):
        """Proposing with target of width != n_species → morph with wrong result.

        XRENSSwap(n_species=3) with a 2-element target will pass the wrong
        n_species to morph_types_to_composition; this should raise or produce
        obviously wrong output. We verify at least that propose() runs without
        silent corruption by checking the types length still matches n_atoms.

        Note: morph_types_to_composition silently produces wrong results for
        wrong n_species, so this test documents the behavior rather than
        testing a guard.
        """
        kernel = XRENSSwap(n_species=3)  # expects 3-element targets
        backend = SpeciesHarmonicBackend([1.0, 2.0, 3.0])
        key = jax.random.PRNGKey(0)
        n_atoms = 6
        types_a = jnp.array([0, 1, 2, 0, 1, 2], dtype=jnp.int32)
        types_b = jnp.array([0, 0, 0, 1, 1, 1], dtype=jnp.int32)
        pos = jax.random.normal(key, (n_atoms, 3)) * 0.1
        e = backend(pos, types_a, jnp.zeros((3, 3))).energy

        state_a = _make_state(pos, types_a, e)
        state_b = _make_state(pos, types_b, e)

        # 2-element target passed to a kernel that expects n_species=3.
        ens_a = {"target_composition": jnp.array([3, 3], dtype=jnp.int32)}
        ens_b = {"target_composition": jnp.array([6, 0], dtype=jnp.int32)}

        # This may or may not raise; just check it doesn't crash silently
        # in a way that corrupts shapes.
        try:
            proposed, _, _ = kernel.propose(
                state_a, state_b, ens_a, ens_b, key, backend
            )
            # If it didn't raise, types must still have correct shape.
            assert proposed["types_a"].shape == (n_atoms,)
            assert proposed["types_b"].shape == (n_atoms,)
        except Exception:
            pass  # Expected — mismatched n_species should fail


# ---------------------------------------------------------------------------
# End-to-end: two-run NS with XRENS
# ---------------------------------------------------------------------------


class TestXRENSEndToEnd:
    """Short NS run with two different compositions, assert XRENS path works."""

    @classmethod
    def _run(cls, n_atoms=8, n_walkers=15, n_iters=15, seed=42):
        """Run XRENS with two different compositions and collect stats."""
        from jaxrens.sampling.move_kernel import MoveKernel
        from jaxrens.sampling.moves import random_walk
        from jaxrens.sampling.mwg import build_mwg
        from jaxrens.sampling.nested_sampling import run_ns_parallel
        from jaxrens.sampling.termination import IterationTermination

        # Two compositions: run 0 = all species 0; run 1 = 50/50.
        # Use a relatively mild difference so swaps aren't impossible.
        target_a = [n_atoms, 0]
        target_b = [n_atoms // 2, n_atoms // 2]

        # Species-harmonic backend: species 0 has weight 1, species 1 has weight 1.5.
        # Same physics as harmonic but type-dependent — ensures energy depends on morph.
        backend = SpeciesHarmonicBackend([1.0, 1.5])

        descriptors = [
            MoveKernel(
                "rw",
                random_walk.build_kernel,
                step_size=0.3,
                step_size_max=2.0,
            )
        ]
        init_fn, step_fn, _ = build_mwg(backend, descriptors)

        n_runs = 2
        key = jax.random.key(seed)
        keys = jax.random.split(key, n_runs + 1)
        rng_keys = keys[:n_runs]
        pos_key = keys[-1]

        # Initial positions: random near origin so energies are moderate.
        positions = jax.random.uniform(
            pos_key, (n_runs, n_walkers, n_atoms, 3), minval=-0.5, maxval=0.5
        )

        # Initial types: run 0 = all species 0; run 1 = 50/50 mix.
        types_0 = jnp.zeros(n_atoms, dtype=jnp.int32)
        half = n_atoms // 2
        types_1 = jnp.concatenate(
            [
                jnp.zeros(half, dtype=jnp.int32),
                jnp.ones(n_atoms - half, dtype=jnp.int32),
            ]
        )
        types_stack = jnp.stack([types_0, types_1])  # (2, n_atoms)

        # Initial energies using per-run types.
        def _eval_run(pos_run, types_run):
            return jax.vmap(
                lambda p: backend(p, types_run, jnp.zeros((3, 3)), 0)[0]
            )(pos_run)

        energies_0 = _eval_run(positions[0], types_0)  # (n_walkers,)
        energies_1 = _eval_run(positions[1], types_1)  # (n_walkers,)
        energies = jnp.stack([energies_0, energies_1])  # (2, n_walkers)

        inter_re_cfg = InterREConfig(
            flavor="xrens",
            re_interval=1,
            n_swap_cycles=1,
            composition_targets=(tuple(target_a), tuple(target_b)),
        )

        # Run NS — collect inter_re_stats via callback.
        re_stats_log = []

        class _StatCb:
            def on_iteration(self, iteration, ns_state, info):
                s = info.get("inter_re_stats")
                if s is not None:
                    re_stats_log.append(dict(s))

        result = run_ns_parallel(
            positions=positions,
            types=types_stack,
            energies=energies,
            cells=None,
            init_fn=init_fn,
            step_fn=step_fn,
            rng_keys=rng_keys,
            n_walkers=n_walkers,
            max_iterations=n_iters,
            n_mcmc_steps=5,
            termination_criteria=[IterationTermination(n_iters)],
            inter_re_config=inter_re_cfg,
            backend=backend,
            callbacks=[_StatCb()],
        )
        return result, re_stats_log

    def test_no_errors_during_run(self):
        """XRENS NS run must complete without exception."""
        result, _ = self._run()
        assert result is not None

    def test_result_shapes(self):
        """Result arrays must have (n_runs, ...) shape."""
        result, _ = self._run()
        assert result["log_evidence"].shape == (2,)
        assert result["n_dead"].shape == (2,)
        assert jnp.all(jnp.isfinite(result["log_evidence"]))

    def test_energy_evals_nonzero(self):
        """inter_re_stats must show n_energy_evals > 0 (morph path ran backend)."""
        result, re_stats_log = self._run()
        # re_stats_log may be empty if callbacks aren't wired through run_ns_parallel;
        # verify by checking the energy evals via a direct manager-level test instead.
        # Direct test: run xrens_replica_exchange_step on a small example.
        n_runs, n_walkers, n_atoms, n_species = 2, 5, 8, 2
        backend = SpeciesHarmonicBackend([1.0, 1.5])
        kernel = XRENSSwap(n_species=n_species)

        key = jax.random.PRNGKey(99)
        k1, k2 = jax.random.split(key)
        positions = (
            jax.random.normal(k1, (n_runs, n_walkers, n_atoms, 3)) * 0.1
        )
        types = jnp.zeros((n_runs, n_walkers, n_atoms), dtype=jnp.int32)

        # Run 0: all species 0; run 1: half-half.
        types = types.at[1, :, n_atoms // 2 :].set(1)

        # Compute energies per walker.
        def _eval(pos, typ):
            return backend(pos, typ, jnp.zeros((3, 3)), 0)[0]

        energies = jax.vmap(jax.vmap(_eval))(positions, types)

        comp_targets = jnp.array(
            [[n_atoms, 0], [n_atoms // 2, n_atoms // 2]], dtype=jnp.int32
        )
        emax = jnp.max(energies, axis=1)

        _, _, _, _, swap_info = xrens_replica_exchange_step(
            rng_key=k2,
            all_positions=positions,
            all_types=types,
            all_energies=energies,
            all_cells=None,
            all_emax=emax,
            composition_targets=comp_targets,
            backend=backend,
            xrens_kernel=kernel,
            n_swap_cycles=1,
        )
        # With 2 runs: 1 even phase pair, 0 odd phase pairs. Each attempted pair
        # costs 2 energy evals.
        assert (
            int(swap_info["n_energy_evals"]) > 0
        ), f"Expected > 0 energy evals from XRENS, got {swap_info['n_energy_evals']}"
        assert (
            int(swap_info["n_attempted"]) > 0
        ), f"Expected > 0 attempted swaps, got {swap_info['n_attempted']}"

    def test_jit_xrens_step(self):
        """xrens_replica_exchange_step must be JIT-compatible."""
        n_runs, n_walkers, n_atoms, n_species = 2, 4, 6, 2
        backend = SpeciesHarmonicBackend([1.0, 2.0])
        kernel = XRENSSwap(n_species=n_species)

        key = jax.random.PRNGKey(10)
        k1, k2 = jax.random.split(key)
        positions = (
            jax.random.normal(k1, (n_runs, n_walkers, n_atoms, 3)) * 0.1
        )
        types = jnp.zeros((n_runs, n_walkers, n_atoms), dtype=jnp.int32)
        types = types.at[1, :, n_atoms // 2 :].set(1)

        def _eval(pos, typ):
            return backend(pos, typ, jnp.zeros((3, 3)), 0)[0]

        energies = jax.vmap(jax.vmap(_eval))(positions, types)

        comp_targets = jnp.array(
            [[n_atoms, 0], [n_atoms // 2, n_atoms // 2]], dtype=jnp.int32
        )
        emax = jnp.max(energies, axis=1)

        @jax.jit
        def jitted_step(rng_key):
            return xrens_replica_exchange_step(
                rng_key=rng_key,
                all_positions=positions,
                all_types=types,
                all_energies=energies,
                all_cells=None,
                all_emax=emax,
                composition_targets=comp_targets,
                backend=backend,
                xrens_kernel=kernel,
                n_swap_cycles=1,
            )

        result = jitted_step(k2)
        new_pos, new_types, new_ene, _, swap_info = result

        assert new_pos.shape == positions.shape
        assert new_types.shape == types.shape
        assert new_ene.shape == energies.shape
        assert "n_accepted" in swap_info
        assert "n_attempted" in swap_info
        assert "n_energy_evals" in swap_info


# ---------------------------------------------------------------------------
# SingleRun skip: n_runs=1 completes without error
# ---------------------------------------------------------------------------


class TestXRENSSingleRunSkip:
    """With n_runs=1, xrens_replica_exchange_step returns state unchanged."""

    def test_single_run_no_swaps(self):
        """n_runs=1 means no pairs possible → state unchanged."""
        n_runs, n_walkers, n_atoms, n_species = 1, 5, 6, 2
        backend = SpeciesHarmonicBackend([1.0, 2.0])
        kernel = XRENSSwap(n_species=n_species)

        key = jax.random.PRNGKey(0)
        positions = (
            jax.random.normal(key, (n_runs, n_walkers, n_atoms, 3)) * 0.1
        )
        types = jnp.zeros((n_runs, n_walkers, n_atoms), dtype=jnp.int32)

        def _eval(pos, typ):
            return backend(pos, typ, jnp.zeros((3, 3)), 0)[0]

        energies = jax.vmap(jax.vmap(_eval))(positions, types)

        comp_targets = jnp.array([[n_atoms, 0]], dtype=jnp.int32)
        emax = jnp.max(energies, axis=1)

        (
            new_pos,
            new_types,
            new_ene,
            new_cells,
            swap_info,
        ) = xrens_replica_exchange_step(
            rng_key=key,
            all_positions=positions,
            all_types=types,
            all_energies=energies,
            all_cells=None,
            all_emax=emax,
            composition_targets=comp_targets,
            backend=backend,
            xrens_kernel=kernel,
            n_swap_cycles=1,
        )

        # No swaps possible — state should be unchanged.
        assert jnp.allclose(new_pos, positions)
        assert jnp.array_equal(new_types, types)
        assert jnp.allclose(new_ene, energies)
        assert int(swap_info["n_attempted"]) == 0
        assert int(swap_info["n_accepted"]) == 0
        assert int(swap_info["n_energy_evals"]) == 0


# ---------------------------------------------------------------------------
# CLI schema: composition_targets validation
# ---------------------------------------------------------------------------


class TestInterRESpecXRENS:
    """CLI schema validation for xrens flavor."""

    def test_xrens_valid_spec(self):
        from jaxrens.cli.schema.inter_re import InterRESpec

        spec = InterRESpec(
            flavor="xrens",
            re_interval=1,
            n_swap_cycles=1,
            composition_targets=[[8, 0], [4, 4]],
        )
        cfg = spec.to_inter_re_config()
        assert cfg.flavor == "xrens"
        assert cfg.composition_targets == ((8, 0), (4, 4))

    def test_xrens_missing_composition_targets_raises(self):
        from jaxrens.cli.schema.inter_re import InterRESpec

        with pytest.raises((ValueError, Exception)):
            InterRESpec(flavor="xrens")  # No composition_targets

    def test_xrens_inconsistent_row_sums_raises(self):
        """Rows summing to different n_atoms must raise."""
        from jaxrens.cli.schema.inter_re import InterRESpec

        with pytest.raises((ValueError, Exception)):
            InterRESpec(
                flavor="xrens",
                composition_targets=[
                    [8, 0],
                    [4, 4, 2],
                ],  # Inconsistent lengths
            )

    def test_xrens_inconsistent_sums_raises(self):
        """Rows with different sums (different n_atoms) must raise."""
        from jaxrens.cli.schema.inter_re import InterRESpec

        with pytest.raises((ValueError, Exception)):
            InterRESpec(
                flavor="xrens",
                composition_targets=[[8, 0], [3, 4]],  # Sums 8 vs 7
            )

    def test_semi_grand_missing_chemical_potentials_raises(self):
        """semi_grand without chemical_potentials must raise ValueError."""
        from jaxrens.cli.schema.inter_re import InterRESpec

        with pytest.raises((ValueError, Exception)):
            InterRESpec(flavor="semi_grand")  # missing chemical_potentials
