"""Unit tests for sampling/neighbor_list.py — pure geometry, no backend."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from jaxrens.sampling.neighbor_list import (
    affected_indices_from_mask,
    compute_periodic_image_offsets,
    dynamic_affected_mask,
    local_affected_mask,
)


def _brute_force_neighbors(positions, cell, r_cutoff, image_range=3):
    """Reference implementation: O(N^2 * n_images), explicit multi-image
    search over a generously wide, hardcoded range — independent of
    local_affected_mask's own (geometry-derived) image-count logic, so
    it's a genuine cross-check rather than testing the function against
    itself. Correct for any r_cutoff as long as image_range is generous
    enough for the test's own cell/cutoff combination.
    """
    n = positions.shape[0]
    offsets = [
        (i, j, k)
        for i in range(-image_range, image_range + 1)
        for j in range(-image_range, image_range + 1)
        for k in range(-image_range, image_range + 1)
    ]
    neighbors = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                # local_affected_mask always includes a touched atom's own
                # index separately (not as a "self-neighbor" via a
                # periodic image, even when r_cutoff exceeds the cell
                # width) — the "affected set" never needs "i is its own
                # neighbor" for that reason.
                continue
            for off in offsets:
                shift = np.array(off) @ cell
                d = positions[i] - (positions[j] + shift)
                dist = np.linalg.norm(d)
                if dist < r_cutoff:
                    neighbors[i].add(j)
    return [sorted(s) for s in neighbors]


class TestLocalAffectedMask:
    """Species-only affected-set search (single_atom_swap, alchemical_morph)
    — positions never move, so unlike TestDynamicAffectedMask there is no
    before/after distinction; a single neighbor search against the current
    positions is enough. Recomputed fresh on every call from whatever
    positions/cell are passed in (see neighbor_list.py module docstring),
    so these tests exercise it directly against brute-force geometry rather
    than through any cached structure."""

    def _true_affected(self, touched, positions, cell, r_cutoff):
        neighbors = _brute_force_neighbors(positions, cell, r_cutoff)
        affected = set(touched)
        for idx in touched:
            affected |= set(neighbors[idx])
        return affected

    def test_matches_brute_force_small_lattice(self):
        rng = np.random.default_rng(0)
        n_atoms = 40
        cell = 12.0 * np.eye(3)
        positions = rng.uniform(0, 12.0, size=(n_atoms, 3))
        r_cutoff = 3.0
        touched_idx = 7

        image_offsets = jnp.asarray(
            compute_periodic_image_offsets(cell, r_cutoff)
        )
        mask = local_affected_mask(
            jnp.array([touched_idx]),
            jnp.asarray(positions),
            jnp.asarray(cell),
            r_cutoff,
            image_offsets,
        )
        got = set(int(i) for i in np.nonzero(np.asarray(mask))[0])
        expected = self._true_affected(
            [touched_idx], positions, cell, r_cutoff
        )
        assert got == expected

    def test_includes_touched_atom_itself(self):
        positions = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]])
        cell = 20.0 * np.eye(3)
        r_cutoff = 1.0  # no neighbors within cutoff

        image_offsets = jnp.asarray(
            compute_periodic_image_offsets(cell, r_cutoff)
        )
        mask = local_affected_mask(
            jnp.array([0]),
            jnp.asarray(positions),
            jnp.asarray(cell),
            r_cutoff,
            image_offsets,
        )
        assert bool(mask[0]) and not bool(mask[1])

    def test_two_touched_atoms_union(self):
        """single_atom_swap touches two atoms — the affected set is the
        UNION of both neighborhoods, not just one."""
        positions = np.array(
            [
                [0.0, 0.0, 0.0],  # touched atom a
                [1.0, 0.0, 0.0],  # neighbor of a only
                [8.0, 0.0, 0.0],  # touched atom b
                [7.5, 0.0, 0.0],  # neighbor of b only
                [15.0, 0.0, 0.0],  # neighbor of neither
            ]
        )
        cell = 20.0 * np.eye(3)
        r_cutoff = 1.5

        image_offsets = jnp.asarray(
            compute_periodic_image_offsets(cell, r_cutoff)
        )
        mask = local_affected_mask(
            jnp.array([0, 2]),
            jnp.asarray(positions),
            jnp.asarray(cell),
            r_cutoff,
            image_offsets,
        )
        got = set(int(i) for i in np.nonzero(np.asarray(mask))[0])
        assert got == {0, 1, 2, 3}

    def test_duplicate_touched_index_is_harmless(self):
        """alchemical_morph touches only one atom, padded as [idx, idx]."""
        positions = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [10.0, 0.0, 0.0]]
        )
        cell = 20.0 * np.eye(3)
        r_cutoff = 2.0

        image_offsets = jnp.asarray(
            compute_periodic_image_offsets(cell, r_cutoff)
        )
        mask = local_affected_mask(
            jnp.array([0, 0]),
            jnp.asarray(positions),
            jnp.asarray(cell),
            r_cutoff,
            image_offsets,
        )
        assert list(np.asarray(mask)) == [True, True, False]

    def test_cutoff_larger_than_half_cell_width(self):
        """The user's real-world case: a small cell where r_cutoff exceeds
        half the box width, so a neighbor can require crossing MORE than
        one periodic boundary (a single nearest-image fold is not enough —
        this must correctly search multiple images per axis)."""
        rng = np.random.default_rng(1)
        n_atoms = 12
        cell = 4.0 * np.eye(3)  # small cell
        positions = rng.uniform(0, 4.0, size=(n_atoms, 3))
        r_cutoff = (
            6.5  # > half the cell width (2.0), also > the cell width itself
        )
        touched_idx = 3

        image_offsets = jnp.asarray(
            compute_periodic_image_offsets(cell, r_cutoff)
        )
        mask = local_affected_mask(
            jnp.array([touched_idx]),
            jnp.asarray(positions),
            jnp.asarray(cell),
            r_cutoff,
            image_offsets,
        )
        got = set(int(i) for i in np.nonzero(np.asarray(mask))[0])
        expected = self._true_affected(
            [touched_idx], positions, cell, r_cutoff
        ) | {touched_idx}
        # Brute-force reference above uses a generous fixed image_range;
        # cross-check with it directly for this cutoff/cell ratio.
        brute_expected = set(
            _brute_force_neighbors(positions, cell, r_cutoff, image_range=4)[
                touched_idx
            ]
        ) | {touched_idx}
        assert got == brute_expected == expected

    def test_cutoff_much_larger_than_cell_requires_many_images(self):
        """An extreme ratio (cutoff several times the cell width) still
        must be exact, not just 'better than before' — correctness must
        not degrade as the cutoff/cell-size ratio grows."""
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.5, 0.5]])
        cell = 2.0 * np.eye(3)
        r_cutoff = 9.0  # 4.5x the cell width

        image_offsets = jnp.asarray(
            compute_periodic_image_offsets(cell, r_cutoff)
        )
        mask = local_affected_mask(
            jnp.array([0]),
            jnp.asarray(positions),
            jnp.asarray(cell),
            r_cutoff,
            image_offsets,
        )
        got = set(int(i) for i in np.nonzero(np.asarray(mask))[0])
        expected = set(
            _brute_force_neighbors(positions, cell, r_cutoff, image_range=8)[0]
        ) | {0}
        assert got == expected

    def test_periodic_wraparound_small_cell(self):
        positions = np.array([[0.0, 0.0, 0.0], [3.9, 0.0, 0.0]])
        cell = 4.0 * np.eye(3)
        r_cutoff = 1.0  # atom 1 is 0.1 away via the periodic image (4.0 - 3.9)

        image_offsets = jnp.asarray(
            compute_periodic_image_offsets(cell, r_cutoff)
        )
        mask = local_affected_mask(
            jnp.array([0]),
            jnp.asarray(positions),
            jnp.asarray(cell),
            r_cutoff,
            image_offsets,
        )
        assert bool(mask[1]), "periodic-image neighbor must be affected"

    def test_jit_compatible(self):
        import jax

        cell = 10.0 * np.eye(3)
        positions = jnp.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [5.0, 5.0, 5.0]]
        )
        r_cutoff = 2.0
        image_offsets = jnp.asarray(
            compute_periodic_image_offsets(np.asarray(cell), r_cutoff)
        )

        f = jax.jit(
            lambda touched, pos, c: local_affected_mask(
                touched, pos, c, r_cutoff, image_offsets
            )
        )
        mask = f(jnp.array([0]), positions, cell)
        assert bool(mask[0]) and bool(mask[1]) and not bool(mask[2])


class TestAffectedIndicesFromMask:
    def test_affected_indices_overflow_flag(self):
        n_atoms = 4
        mask = jnp.array([True, True, True, False])
        indices, overflow = affected_indices_from_mask(mask, max_affected=2)
        assert bool(overflow)
        indices2, overflow2 = affected_indices_from_mask(mask, max_affected=4)
        assert not bool(overflow2)
        assert sorted(int(x) for x in indices2 if x < n_atoms) == [0, 1, 2]


class TestDynamicAffectedMask:
    def _true_affected(
        self, atom_idx, positions_before, positions_after, cell, r_cutoff
    ):
        """Brute-force reference: touched atom + true neighbors under
        either its old or new position (multi-image, generous range)."""
        before = _brute_force_neighbors(
            positions_before, cell, r_cutoff, image_range=4
        )
        after = _brute_force_neighbors(
            positions_after, cell, r_cutoff, image_range=4
        )
        return {atom_idx} | set(before[atom_idx]) | set(after[atom_idx])

    def test_matches_brute_force_no_boundary_crossing(self):
        rng = np.random.default_rng(2)
        n_atoms = 20
        cell = 10.0 * np.eye(3)
        positions = rng.uniform(0, 10.0, size=(n_atoms, 3))
        r_cutoff = 2.5
        atom_idx = 3
        displacement = np.array(
            [0.1, -0.05, 0.02]
        )  # small, no crossing expected
        new_positions = positions.copy()
        new_positions[atom_idx] += displacement

        image_offsets = jnp.asarray(
            compute_periodic_image_offsets(cell, r_cutoff)
        )
        mask = dynamic_affected_mask(
            jnp.asarray(atom_idx),
            jnp.asarray(positions),
            jnp.asarray(new_positions),
            jnp.asarray(cell),
            r_cutoff,
            image_offsets,
        )
        got = set(int(i) for i in np.nonzero(np.asarray(mask))[0])
        expected = self._true_affected(
            atom_idx, positions, new_positions, cell, r_cutoff
        )
        assert got == expected

    def test_catches_atom_entering_cutoff_shell(self):
        """The key correctness property: a neighbor that was OUTSIDE the
        cutoff before the move but ENTERS it after must be in the affected
        set (checking only the old position would miss it; checking only
        the new position would miss an atom that LEFT — this test covers
        the 'entering' half, the next test covers 'leaving')."""
        cell = 20.0 * np.eye(3)
        positions = np.array(
            [
                [0.0, 0.0, 0.0],  # atom 0: the one that moves
                [3.4, 0.0, 0.0],  # atom 1: just outside cutoff=3.0 initially
            ]
        )
        r_cutoff = 3.0
        new_positions = positions.copy()
        new_positions[0] = [
            1.0,
            0.0,
            0.0,
        ]  # now 2.4 from atom 1 -> inside cutoff

        image_offsets = jnp.asarray(
            compute_periodic_image_offsets(cell, r_cutoff)
        )
        mask = dynamic_affected_mask(
            jnp.asarray(0),
            jnp.asarray(positions),
            jnp.asarray(new_positions),
            jnp.asarray(cell),
            r_cutoff,
            image_offsets,
        )
        assert bool(
            mask[1]
        ), "atom entering the cutoff shell after the move must be affected"

    def test_catches_atom_leaving_cutoff_shell(self):
        cell = 20.0 * np.eye(3)
        positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        r_cutoff = 3.0  # atom 1 starts inside cutoff
        new_positions = positions.copy()
        new_positions[0] = [
            5.0,
            0.0,
            0.0,
        ]  # now 3.0 away from atom 1 -> outside

        image_offsets = jnp.asarray(
            compute_periodic_image_offsets(cell, r_cutoff)
        )
        mask = dynamic_affected_mask(
            jnp.asarray(0),
            jnp.asarray(positions),
            jnp.asarray(new_positions),
            jnp.asarray(cell),
            r_cutoff,
            image_offsets,
        )
        assert bool(
            mask[1]
        ), "atom leaving the cutoff shell after the move must be affected"

    def test_periodic_wraparound_small_cell(self):
        """Combines both correctness lessons: a moving atom in a cell
        smaller than the cutoff, where the affected neighbor is only
        reachable through a periodic image."""
        cell = 4.0 * np.eye(3)
        positions = np.array([[0.0, 0.0, 0.0], [3.9, 0.0, 0.0]])
        r_cutoff = 1.0  # atom 1 is 0.1 away via the periodic image (4.0 - 3.9)
        new_positions = positions.copy()
        new_positions[0] = [
            0.5,
            0.0,
            0.0,
        ]  # moves further from the image (0.6 away)

        image_offsets = jnp.asarray(
            compute_periodic_image_offsets(cell, r_cutoff)
        )
        mask = dynamic_affected_mask(
            jnp.asarray(0),
            jnp.asarray(positions),
            jnp.asarray(new_positions),
            jnp.asarray(cell),
            r_cutoff,
            image_offsets,
        )
        assert bool(
            mask[1]
        ), "periodic-image neighbor under the OLD position must be affected"

    def test_jit_compatible(self):
        import jax

        cell = 10.0 * np.eye(3)
        positions = jnp.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [5.0, 5.0, 5.0]]
        )
        new_positions = positions.at[0].set(jnp.array([0.5, 0.0, 0.0]))
        r_cutoff = 2.0
        image_offsets = jnp.asarray(
            compute_periodic_image_offsets(np.asarray(cell), r_cutoff)
        )

        f = jax.jit(
            lambda a, b, c: dynamic_affected_mask(
                jnp.asarray(0), a, b, c, r_cutoff, image_offsets
            )
        )
        mask = f(positions, new_positions, cell)
        assert bool(mask[0]) and bool(mask[1]) and not bool(mask[2])
