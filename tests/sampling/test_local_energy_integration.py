"""Integration tests for local/incremental energy updates through build_mwg.

Exercises the full wiring: MoveKernel.affects -> build_mwg's capability
check and cache plumbing -> single_atom_swap/alchemical_morph's local path
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


def _local_extra_state_fields():
    return {
        "atomic_energies": (
            jnp.ndarray,
            lambda positions, types: jnp.zeros(positions.shape[0]),
        ),
        "raw_energy": (jnp.ndarray, lambda positions, types: jnp.asarray(0.0)),
    }


def _swap_kernel_descriptor(max_affected):
    from jaxrens.sampling.move_kernel import MoveKernel
    from jaxrens.sampling.moves import single_atom

    return MoveKernel(
        name="single_atom_swap",
        build_kernel=single_atom.build_swap_kernel,
        kernel_kwargs={"max_affected": max_affected},
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
        desc = _swap_kernel_descriptor(max_affected=6)

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
        desc = _swap_kernel_descriptor(max_affected=n_atoms)

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
        """Two DIFFERENT local moves in the same set: single_atom_swap
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
        swap_desc = _swap_kernel_descriptor(max_affected=n_atoms)
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
