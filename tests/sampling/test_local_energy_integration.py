"""Integration tests for local/incremental energy updates through build_mwg.

Exercises the full wiring: MoveKernel.affects -> build_mwg's capability
check and cache plumbing -> species_swap/alchemical_morph's local path
-> NeuralILBackend.atomic_energies_for -> agreement with a from-scratch
full recomputation. Requires the real NeuralIL backend (skipped otherwise).
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxrens.backends.neuralil import _NEURALIL_IMPORT_ERROR, is_available

neuralil_required = pytest.mark.skipif(
    not is_available(),
    reason=f"NeuralIL not installed: {_NEURALIL_IMPORT_ERROR}",
)

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "neuralil_tiny"


@pytest.fixture
def toy_backend():
    from jaxrens.backends.toy import create_harmonic

    return create_harmonic(k=1.0)


def _local_extra_state_fields(n_terms=0):
    return {
        "atomic_energies": (
            jnp.ndarray,
            lambda positions, types: jnp.zeros(positions.shape[0]),
        ),
        "raw_energy": (jnp.ndarray, lambda positions, types: jnp.asarray(0.0)),
        "term_energies": (
            jnp.ndarray,
            lambda positions, types, n=n_terms: jnp.zeros(n),
        ),
    }


def _swap_kernel_descriptor(max_affected, n_species, species=None):
    from jaxrens.sampling.move_kernel import MoveKernel
    from jaxrens.sampling.moves import swap

    return MoveKernel(
        name="species_swap",
        build_kernel=swap.build_kernel,
        kernel_kwargs={
            "n_species": n_species,
            "species": species,
            "max_affected": max_affected,
        },
        mutates=frozenset({"types"}),
        affects="local",
        extra_state_fields=_local_extra_state_fields(),
    )


def _morph_kernel_descriptor(max_affected, n_species):
    from jaxrens.sampling.move_kernel import MoveKernel
    from jaxrens.sampling.moves import alchemical

    return MoveKernel(
        name="alchemical_morph",
        build_kernel=alchemical.build_morph_kernel,
        kernel_kwargs={
            "n_species": n_species,
            "max_affected": max_affected,
        },
        mutates=frozenset({"types"}),
        affects="local",
        extra_state_fields=_local_extra_state_fields(),
    )


def _single_atom_kernel_descriptor(max_affected):
    from jaxrens.sampling.move_kernel import MoveKernel
    from jaxrens.sampling.moves import single_atom

    return MoveKernel(
        name="single_atom",
        build_kernel=single_atom.build_kernel,
        kernel_kwargs={"max_affected": max_affected},
        mutates=frozenset({"positions"}),
        affects="local",
        extra_state_fields=_local_extra_state_fields(),
    )


def _single_atom_sweep_kernel_descriptor(n_atoms, max_affected):
    from jaxrens.sampling.move_kernel import MoveKernel
    from jaxrens.sampling.moves import single_atom

    return MoveKernel(
        name="single_atom_sweep",
        build_kernel=single_atom.build_sweep_kernel,
        kernel_kwargs={
            "n_atoms": n_atoms,
            "max_affected": max_affected,
        },
        mutates=frozenset({"positions"}),
        affects="local",
        extra_state_fields=_local_extra_state_fields(),
    )


def _volume_kernel_descriptor(n_atoms, step_size=2.0):
    from jaxrens.sampling.move_kernel import MoveKernel
    from jaxrens.sampling.moves import volume

    return MoveKernel(
        name="volume",
        build_kernel=volume.build_kernel,
        kernel_kwargs={
            "n_atoms": n_atoms,
            "max_vol_per_atom": 1000.0,
            "min_vol_per_atom": 0.5,
            "min_aspect": 0.3,
        },
        weight=1.0,
        step_size=step_size,
        reject_reasons=frozenset({"energy", "cell", "prior"}),
        mutates=frozenset({"positions", "cell"}),
        affects="all",
    )


class TestPatchTermEnergies:
    """``patch_term_energies`` (the generalized per-``EnergyTerm`` cache
    patcher) in isolation — no NeuralIL required, since it only ever reads
    ``backend.terms``, never the model. Covers the two branches every local
    move kernel relies on: a term with its own ``local_energy_delta``
    (``SoftCoreTerm``) and a term without one, which must fall back to a
    full two-evaluation recompute for itself only (``EnsembleTerm``) — see
    ``backends/hamiltonian.py::EnergyTerm.local_energy_delta`` and
    ``sampling/local_energy.py::patch_term_energies``.
    """

    def test_no_terms_is_a_noop(self):
        from jaxrens.sampling.local_energy import patch_term_energies

        backend = object()  # no .terms attribute at all
        term_energies = jnp.zeros(0)
        touched = jnp.array([0])
        positions = jnp.zeros((2, 3))
        types = jnp.zeros(2, dtype=jnp.int32)
        cell = jnp.eye(3) * 10.0

        result = patch_term_energies(
            backend,
            term_energies,
            touched,
            positions,
            types,
            positions,
            types,
            cell,
            None,
        )
        assert result.shape == (0,)

    def test_ensemble_term_fallback_matches_full_recompute(self):
        """EnsembleTerm has no local_energy_delta override, so the
        fallback (energy(new) - energy(old), added to the cache) must
        reproduce exactly what re-evaluating EnsembleTerm.energy from
        scratch would give."""
        from jaxrens.backends.ensemble import EnsembleTerm
        from jaxrens.backends.hamiltonian import Hamiltonian
        from jaxrens.sampling.local_energy import patch_term_energies

        class _ZeroModel:
            r_cutoff = 1.0

        term = EnsembleTerm(
            pressure=0.5, chemical_potentials=jnp.array([0.1, -0.2])
        )
        backend = Hamiltonian(_ZeroModel(), [term])

        old_types = jnp.array([0, 0, 1])
        new_types = jnp.array([0, 1, 1])  # species counts change
        positions = jnp.zeros((3, 3))
        cell = jnp.eye(3) * 5.0

        term_energies_old = jnp.array(
            [term.energy(positions, old_types, cell)]
        )
        patched = patch_term_energies(
            backend,
            term_energies_old,
            jnp.array([1]),
            positions,
            old_types,
            positions,
            new_types,
            cell,
            None,
        )
        expected = term.energy(positions, new_types, cell)
        assert jnp.allclose(patched[0], expected)

    def test_softcore_and_ensemble_terms_combine_correctly(self):
        """The case the original design would have silently dropped:
        SoftCoreTerm *and* EnsembleTerm active together. Each term's
        contribution must be patched independently and sum to the exact
        total a from-scratch Hamiltonian evaluation would give."""
        from jaxrens.backends.ensemble import EnsembleTerm
        from jaxrens.backends.hamiltonian import Hamiltonian
        from jaxrens.backends.softcore import SoftCoreTerm
        from jaxrens.sampling.local_energy import patch_term_energies

        class _ZeroModel:
            r_cutoff = 1.0

            def __call__(self, positions, species, cell, max_neighbors, **kw):
                from jaxrens.backends.base import BackendResult

                return BackendResult(energy=jnp.asarray(0.0))

        softcore = SoftCoreTerm()
        ensemble = EnsembleTerm(pressure=0.3)
        backend = Hamiltonian(_ZeroModel(), [softcore, ensemble])

        rng = np.random.default_rng(0)
        positions = jnp.asarray(
            rng.uniform(0.0, 10.0, size=(10, 3)), dtype=jnp.float32
        )
        types = jnp.zeros(10, dtype=jnp.int32)
        cell = jnp.eye(3, dtype=jnp.float32) * 10.0

        touched = jnp.array([2])
        new_positions = positions.at[2].add(jnp.array([0.3, -0.1, 0.0]))

        term_energies_old = jnp.array(
            [t.energy(positions, types, cell) for t in backend.terms]
        )
        patched = patch_term_energies(
            backend,
            term_energies_old,
            touched,
            positions,
            types,
            new_positions,
            types,
            cell,
            None,
        )
        expected_total = sum(
            float(t.energy(new_positions, types, cell)) for t in backend.terms
        )
        scale = max(abs(expected_total), 1.0)
        assert abs(float(jnp.sum(patched)) - expected_total) < 1e-2 * scale


class TestBuildMwgGracefulDowngrade:
    """build_mwg never rejects a move-set: a move that declares
    affects='local' but whose backend doesn't support the subset-energy
    query it needs is silently downgraded to 'all' (full recompute for
    that move only), with a warning logged via the standard `logging`
    module. Every local move now recomputes its affected set fresh from
    the current state on every proposal, so combining it with ANY other
    move (position- or cell-mutating included) is safe by construction —
    the only remaining reason for a downgrade is genuine backend
    incapability, which is what this test checks.
    """

    def test_incapable_backend_downgrades_with_warning(
        self, toy_backend, caplog
    ):
        from jaxrens.sampling.mwg import build_mwg

        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.5, 0.0, 0.0],
                [0.0, 1.5, 0.0],
                [1.5, 1.5, 0.0],
            ]
        )
        cell = 10.0 * np.eye(3)
        desc = _swap_kernel_descriptor(max_affected=6, n_species=2)

        with caplog.at_level("WARNING"):
            init_fn, step_fn, _ = build_mwg(toy_backend, [desc])
        assert any(
            "local energy capability" in r.message for r in caplog.records
        )
        # The move-set still works — just without the speedup.
        state = init_fn(
            jnp.asarray(positions),
            jnp.array([0, 1, 0, 1]),
            0.0,
            jnp.asarray(cell),
        )
        assert not hasattr(state, "atomic_energies")
        new_state, info = step_fn(
            jax.random.key(0), state, likelihood_constraint=1e9
        )
        assert new_state.positions.shape == positions.shape


@neuralil_required
class TestLocalUpdateThroughBuildMwg:
    @pytest.fixture
    def backend(self):
        from jaxrens.backends.neuralil import create_neuralil

        return create_neuralil(
            pickle_file=str(FIXTURE_DIR / "model.pkl"),
            supercell_trafo=(1, 1, 1),
        )

    @pytest.fixture
    def config(self):
        ref = np.load(FIXTURE_DIR / "reference.npz", allow_pickle=True)
        return (
            jnp.array(ref["positions"]),
            jnp.array(ref["types"], dtype=jnp.int32),
            jnp.array(ref["cell"]),
            int(ref["max_neighbors"]),
        )

    def test_swap_via_build_mwg_matches_full_recompute_over_many_steps(
        self, backend, config
    ):
        """The strongest regression guard: drive many local-path swap
        steps through the real build_mwg/init_fn/step_fn wiring, and after
        every accepted step confirm state.energy matches an independent
        from-scratch recomputation. Also exercises seed_local_energy_cache
        (the mandatory post-init seeding step)."""
        from jaxrens.sampling.local_energy import seed_local_energy_cache
        from jaxrens.sampling.mwg import build_mwg
        from jaxrens.sampling.neighbor_list import (
            initial_image_bucket_for_cell,
        )

        positions, species, cell, mn = config
        n_atoms = positions.shape[0]
        n_species = int(species.max()) + 1
        if n_species < 2:
            pytest.skip("fixture has only one species; swap needs >= 2")

        image_bucket = initial_image_bucket_for_cell(
            np.asarray(cell), backend.r_cutoff
        )
        desc = _swap_kernel_descriptor(
            max_affected=n_atoms, n_species=n_species
        )

        init_fn, step_fn, _ = build_mwg(backend, [desc])

        energy0 = backend(positions, species, cell, mn).energy
        state = init_fn(
            positions,
            species,
            energy0,
            cell,
            max_neighbors=mn,
            image_bucket=image_bucket,
        )
        state = seed_local_energy_cache(state, backend)

        key = jax.random.key(0)
        for i in range(30):
            key, step_key = jax.random.split(key)
            state, info = step_fn(step_key, state, likelihood_constraint=1e9)

            full = backend(state.positions, state.types, state.cell, mn).energy
            assert abs(float(state.energy) - float(full)) < 1e-3 * max(
                abs(float(full)), 1.0
            ), f"step {i}: incremental energy diverged from full recompute"

    def test_morph_via_build_mwg_matches_full_recompute_over_many_steps(
        self, backend, config
    ):
        """Same regression guard as the swap test, for alchemical_morph."""
        from jaxrens.sampling.local_energy import seed_local_energy_cache
        from jaxrens.sampling.mwg import build_mwg
        from jaxrens.sampling.neighbor_list import (
            initial_image_bucket_for_cell,
        )

        positions, species, cell, mn = config
        n_atoms = positions.shape[0]
        n_species = int(species.max()) + 1

        image_bucket = initial_image_bucket_for_cell(
            np.asarray(cell), backend.r_cutoff
        )
        desc = _morph_kernel_descriptor(
            max_affected=n_atoms, n_species=n_species
        )

        init_fn, step_fn, _ = build_mwg(backend, [desc])

        energy0 = backend(positions, species, cell, mn).energy
        state = init_fn(
            positions,
            species,
            energy0,
            cell,
            max_neighbors=mn,
            image_bucket=image_bucket,
        )
        state = seed_local_energy_cache(state, backend)

        key = jax.random.key(1)
        for i in range(30):
            key, step_key = jax.random.split(key)
            state, info = step_fn(step_key, state, likelihood_constraint=1e9)

            full = backend(state.positions, state.types, state.cell, mn).energy
            assert abs(float(state.energy) - float(full)) < 1e-3 * max(
                abs(float(full)), 1.0
            ), f"step {i}: incremental energy diverged from full recompute"

    def test_single_atom_via_build_mwg_matches_full_recompute_over_many_steps(
        self, backend, config
    ):
        """Same regression guard as swap/morph, for the DYNAMIC case
        (single_atom): positions move, so this exercises
        dynamic_affected_mask's fresh-per-proposal neighbor search rather
        than the static table."""
        from jaxrens.sampling.local_energy import seed_local_energy_cache
        from jaxrens.sampling.mwg import build_mwg
        from jaxrens.sampling.neighbor_list import (
            initial_image_bucket_for_cell,
        )

        positions, species, cell, mn = config
        n_atoms = positions.shape[0]

        image_bucket = initial_image_bucket_for_cell(
            np.asarray(cell), backend.r_cutoff
        )
        desc = _single_atom_kernel_descriptor(max_affected=n_atoms)

        init_fn, step_fn, _ = build_mwg(backend, [desc])

        energy0 = backend(positions, species, cell, mn).energy
        state = init_fn(
            positions,
            species,
            energy0,
            cell,
            max_neighbors=mn,
            step_size=0.15,
            image_bucket=image_bucket,
        )
        state = seed_local_energy_cache(state, backend)

        key = jax.random.key(2)
        for i in range(30):
            key, step_key = jax.random.split(key)
            state, info = step_fn(step_key, state, likelihood_constraint=1e9)

            full = backend(state.positions, state.types, state.cell, mn).energy
            assert abs(float(state.energy) - float(full)) < 1e-3 * max(
                abs(float(full)), 1.0
            ), f"step {i}: incremental energy diverged from full recompute"

    def test_single_atom_sweep_via_build_mwg_matches_full_recompute(
        self, backend, config
    ):
        """Same regression guard, for single_atom_sweep (the local update
        threaded through an existing lax.scan carry)."""
        from jaxrens.sampling.local_energy import seed_local_energy_cache
        from jaxrens.sampling.mwg import build_mwg
        from jaxrens.sampling.neighbor_list import (
            initial_image_bucket_for_cell,
        )

        positions, species, cell, mn = config
        n_atoms = positions.shape[0]

        image_bucket = initial_image_bucket_for_cell(
            np.asarray(cell), backend.r_cutoff
        )
        desc = _single_atom_sweep_kernel_descriptor(
            n_atoms, max_affected=n_atoms
        )

        init_fn, step_fn, _ = build_mwg(backend, [desc])

        energy0 = backend(positions, species, cell, mn).energy
        state = init_fn(
            positions,
            species,
            energy0,
            cell,
            max_neighbors=mn,
            step_size=0.1,
            image_bucket=image_bucket,
        )
        state = seed_local_energy_cache(state, backend)

        key = jax.random.key(3)
        for i in range(5):
            key, step_key = jax.random.split(key)
            state, info = step_fn(step_key, state, likelihood_constraint=1e9)

            full = backend(state.positions, state.types, state.cell, mn).energy
            assert abs(float(state.energy) - float(full)) < 1e-3 * max(
                abs(float(full)), 1.0
            ), f"sweep {i}: incremental energy diverged from full recompute"

    def test_mixing_dynamic_local_move_with_position_only_global_move(
        self, backend, config
    ):
        """Answers a real question: if the move-set ALSO includes a
        NON-optimized move that stays affects="all", does mixing it with a
        local move stay correct? This exercises mwg.py::_wrap's cache-
        invalidation path (recompute the cache from scratch after every
        accepted affects="all" move) — not exercised by any other test
        here, since they all use a single, purely local move-set.

        Uses single_atom (affects="local") + random_walk (affects="all",
        mutates only "positions").
        """
        from jaxrens.sampling.local_energy import seed_local_energy_cache
        from jaxrens.sampling.move_kernel import MoveKernel
        from jaxrens.sampling.moves import random_walk
        from jaxrens.sampling.mwg import build_mwg
        from jaxrens.sampling.neighbor_list import (
            initial_image_bucket_for_cell,
        )

        positions, species, cell, mn = config
        n_atoms = positions.shape[0]

        image_bucket = initial_image_bucket_for_cell(
            np.asarray(cell), backend.r_cutoff
        )
        single_atom_desc = _single_atom_kernel_descriptor(max_affected=n_atoms)

        random_walk_desc = MoveKernel(
            name="random_walk",
            build_kernel=random_walk.build_kernel,
            weight=1.0,
            step_size=0.05,
            mutates=frozenset({"positions"}),
            affects="all",  # deliberately NOT opted into local_update
        )

        init_fn, step_fn, _ = build_mwg(
            backend, [single_atom_desc, random_walk_desc]
        )

        energy0 = backend(positions, species, cell, mn).energy
        state = init_fn(
            positions,
            species,
            energy0,
            cell,
            max_neighbors=mn,
            step_sizes=jnp.array([0.15, 0.05]),
            image_bucket=image_bucket,
        )
        state = seed_local_energy_cache(state, backend)

        key = jax.random.key(4)
        for i in range(20):
            key, step_key = jax.random.split(key)
            state, info = step_fn(step_key, state, likelihood_constraint=1e9)

            full = backend(state.positions, state.types, state.cell, mn).energy
            assert abs(float(state.energy) - float(full)) < 1e-3 * max(
                abs(float(full)), 1.0
            ), (
                f"step {i} (move_idx={int(info.move_idx)}): incremental "
                "energy diverged from full recompute after mixing a local "
                "move with a non-optimized global move"
            )

    def test_swap_and_single_atom_together_stay_consistent(
        self, backend, config
    ):
        """Two DIFFERENT local moves in the same set: species_swap
        relabels species without moving anything, single_atom moves one
        atom's position — before Stage 3's unification, combining these
        would have silently invalidated swap's precomputed static neighbor
        table the moment single_atom moved a position. Both now recompute
        their affected set fresh from the current state on every proposal,
        so this combination must stay correct with no downgrade."""
        from jaxrens.sampling.local_energy import seed_local_energy_cache
        from jaxrens.sampling.mwg import build_mwg
        from jaxrens.sampling.neighbor_list import (
            initial_image_bucket_for_cell,
        )

        positions, species, cell, mn = config
        n_atoms = positions.shape[0]
        n_species = int(species.max()) + 1
        if n_species < 2:
            pytest.skip("fixture has only one species; swap needs >= 2")

        image_bucket = initial_image_bucket_for_cell(
            np.asarray(cell), backend.r_cutoff
        )
        swap_desc = _swap_kernel_descriptor(
            max_affected=n_atoms, n_species=n_species
        )
        single_atom_desc = _single_atom_kernel_descriptor(max_affected=n_atoms)

        init_fn, step_fn, _ = build_mwg(backend, [swap_desc, single_atom_desc])

        energy0 = backend(positions, species, cell, mn).energy
        state = init_fn(
            positions,
            species,
            energy0,
            cell,
            max_neighbors=mn,
            step_sizes=jnp.array([0.0, 0.15]),
            image_bucket=image_bucket,
        )
        state = seed_local_energy_cache(state, backend)

        key = jax.random.key(5)
        for i in range(30):
            key, step_key = jax.random.split(key)
            state, info = step_fn(step_key, state, likelihood_constraint=1e9)

            full = backend(state.positions, state.types, state.cell, mn).energy
            assert abs(float(state.energy) - float(full)) < 1e-3 * max(
                abs(float(full)), 1.0
            ), (
                f"step {i} (move_idx={int(info.move_idx)}): incremental "
                "energy diverged after combining two local moves"
            )

    def test_local_move_mixed_with_volume_move_no_downgrade_stays_consistent(
        self, backend, config, caplog
    ):
        """The scenario Stage 3 exists for: a local move (single_atom)
        combined with a cell-mutating move (volume) in the same set. Under
        the old static-neighbor-table/precomputed-image-count design this
        combination either had to be rejected or silently downgraded; now
        both moves recompute their own needed geometry fresh on every
        proposal, so this must run with NO downgrade warning and stay
        numerically correct even as the volume move actually changes the
        cell."""
        from jaxrens.sampling.local_energy import seed_local_energy_cache
        from jaxrens.sampling.mwg import build_mwg
        from jaxrens.sampling.neighbor_list import (
            initial_image_bucket_for_cell,
        )

        positions, species, cell, mn = config
        n_atoms = positions.shape[0]

        image_bucket = initial_image_bucket_for_cell(
            np.asarray(cell), backend.r_cutoff
        )
        single_atom_desc = _single_atom_kernel_descriptor(max_affected=n_atoms)
        volume_desc = _volume_kernel_descriptor(n_atoms, step_size=2.0)

        with caplog.at_level("WARNING"):
            init_fn, step_fn, _ = build_mwg(
                backend, [single_atom_desc, volume_desc]
            )

        energy0 = backend(positions, species, cell, mn).energy
        state = init_fn(
            positions,
            species,
            energy0,
            cell,
            max_neighbors=mn,
            step_sizes=jnp.array([0.15, 2.0]),
            image_bucket=image_bucket,
        )
        state = seed_local_energy_cache(state, backend)

        key = jax.random.key(6)
        volume_accepted_at_least_once = False
        for i in range(30):
            key, step_key = jax.random.split(key)
            state, info = step_fn(step_key, state, likelihood_constraint=1e9)
            if int(info.move_idx) == 1 and bool(info.accepted):
                volume_accepted_at_least_once = True

            full = backend(state.positions, state.types, state.cell, mn).energy
            assert abs(float(state.energy) - float(full)) < 1e-3 * max(
                abs(float(full)), 1.0
            ), (
                f"step {i} (move_idx={int(info.move_idx)}): incremental "
                "energy diverged from full recompute after mixing a local "
                "move with a cell-mutating move"
            )

        assert not any(
            "falling back to full energy recomputation" in r.message
            for r in caplog.records
        ), "local move must not be downgraded when combined with a volume move"
        assert (
            volume_accepted_at_least_once
        ), "test is only meaningful if the cell actually changed at least once"
