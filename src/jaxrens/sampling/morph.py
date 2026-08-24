"""Pure-JAX, JIT-compatible atom-type morphing primitive.

Provides `morph_types_to_composition`: deterministic label relabeling
that reshuffles an existing types array to match a target composition.
JIT/vmap/pmap safe; no Python-level indexing on traced values.

"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Int, Key

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _counts_from_types(
    types: Int[Array, "N"], n_species: int
) -> Int[Array, "n_species"]:
    """Count atoms per species.

    Args:
        types: shape (n_atoms,), int dtype in [0, n_species-1].
        n_species: static int; number of distinct species labels.

    Returns:
        shape (n_species,), int32 — count of atoms for each species index.
    """
    return jnp.bincount(types, length=n_species)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def morph_types_to_composition(
    rng_key: Key[Array, ""],
    types: Int[Array, "N"],
    target_composition: Int[Array, "n_species"],
    n_species: int,
) -> Int[Array, "N"]:
    """Deterministically relabel atom types to match `target_composition`.

    Uses random selection to pick which existing atoms get relabeled so the
    resulting composition matches the target. Deterministic given rng_key;
    pure JAX, JIT-compatible, no Python-side indexing on traced values.

    Algorithm (scan over species, no dynamic indexing):
      1. Compute delta = target_composition - current_composition per species.
      2. For each donor species d (delta[d] < 0): draw ``|delta[d]|`` random atom
         indices among ``{i | types[i] == d}`` using uniform scores + top_k.
         Collect all chosen donor indices into a fixed-size buffer (length
         n_atoms) in scan order.
      3. Fill a parallel receiver-label buffer: for each receiver species r
         (delta[r] > 0), write label r into delta[r] positions.
      4. Apply the reassignments: for positions 0..total_moves-1 in the
         picked-index buffer, set types[picked[j]] = receiver_label[j].

    Invariants:
      - sum(target_composition) == n_atoms  (caller's responsibility)
      - output.shape == types.shape
      - jnp.bincount(output, length=n_species) == target_composition

    Note on runtime assertions:
      The condition sum(target_composition) == n_atoms is NOT checked at trace
      time (it would require a Python-level comparison on a traced value).
      Violations produce silently wrong results. Callers must ensure this
      invariant before calling.

    Args:
        rng_key: JAX PRNG key.
        types: shape (n_atoms,), int dtype — current per-atom species labels
            in [0, n_species-1].
        target_composition: shape (n_species,), int dtype — desired count per
            species; must satisfy sum == n_atoms.
        n_species: static Python int — number of distinct species labels.
            Must match len(target_composition). Determines scan lengths; must
            be known at trace time.

    Returns:
        new_types: shape (n_atoms,), same dtype as `types` — relabeled atom
            types matching `target_composition`.
    """
    n_atoms = types.shape[0]
    k = n_species

    # Current counts and per-species deltas
    n_current = _counts_from_types(types, k)
    delta = (target_composition - n_current).astype(jnp.int32)  # (k,)

    # One PRNG key per donor species (keys[0] unused; keys[1..k] per species)
    keys = jax.random.split(rng_key, k + 1)
    keys_species = keys[1:]  # shape (k,); index d → keys_species[d]

    # -----------------------------------------------------------------------
    # Phase 1: collect donor atom indices into a fixed-size buffer
    # -----------------------------------------------------------------------
    # picked[0..total_moves-1] holds atom indices to be relabeled.
    # Entries beyond total_moves are -1 (sentinel, never read).
    picked = -jnp.ones((n_atoms,), dtype=jnp.int32)
    ptr0 = jnp.array(0, dtype=jnp.int32)

    def pick_from_donor_species(carry, d):
        """Collect donor indices for species d into the picked buffer."""
        picked_buf, ptr = carry

        # How many atoms to take from this species (0 for non-donor species)
        k_need = jnp.where(delta[d] < 0, -delta[d], 0).astype(jnp.int32)
        is_donor = k_need > 0

        # Random scores for all atoms; atoms of wrong species get +inf so
        # they never appear in the top-k. Uniform scores in [0, 1) otherwise.
        scores = jax.random.uniform(keys_species[d], (n_atoms,))
        scores = jnp.where(types == d, scores, jnp.inf)

        # top_k on negated scores selects the k_need atoms with the smallest
        # (most random) uniform scores — i.e., a uniform sample without
        # replacement. We request n_atoms candidates and later mask to k_need.
        _, idxs = lax.top_k(-scores, n_atoms)  # shape (n_atoms,)

        # Write first k_need indices into picked[ptr:ptr+k_need] via scan.
        # The scan index j identifies the j-th top candidate; we write only
        # while j < k_need and the species is a donor.
        def write_one(carry2, j):
            buf2, ptr2 = carry2
            do = is_donor & (j < k_need)
            idx = idxs[j]
            buf2 = lax.cond(
                do,
                lambda b: b.at[ptr2 + j].set(idx),
                lambda b: b,
                buf2,
            )
            return (buf2, ptr2), None

        (picked_buf, ptr), _ = lax.scan(
            write_one,
            (picked_buf, ptr),
            jnp.arange(n_atoms, dtype=jnp.int32),
        )
        ptr = ptr + k_need
        return (picked_buf, ptr), None

    (picked, total_moves), _ = lax.scan(
        pick_from_donor_species,
        (picked, ptr0),
        jnp.arange(k, dtype=jnp.int32),
    )

    # -----------------------------------------------------------------------
    # Phase 2: build receiver-label buffer in the same index order
    # -----------------------------------------------------------------------
    # receiver_for_pos[j] = the species label to assign to picked[j].
    # Entries beyond total_moves are -1 (sentinel, never read).
    receiver_for_pos = -jnp.ones((n_atoms,), dtype=jnp.int32)
    ptr0 = jnp.array(0, dtype=jnp.int32)

    def fill_receiver_species(carry, r):
        """Fill receiver labels for species r into the receiver buffer."""
        recv_buf, ptr = carry
        need = jnp.where(delta[r] > 0, delta[r], 0).astype(jnp.int32)
        is_recv = need > 0

        def fill_one(carry2, j):
            arr, ptr2 = carry2
            do = is_recv & (j < need)
            arr = lax.cond(
                do,
                lambda a: a.at[ptr2 + j].set(r),
                lambda a: a,
                arr,
            )
            return (arr, ptr2), None

        (recv_buf, ptr), _ = lax.scan(
            fill_one,
            (recv_buf, ptr),
            jnp.arange(n_atoms, dtype=jnp.int32),
        )
        ptr = ptr + need
        return (recv_buf, ptr), None

    (receiver_for_pos, _), _ = lax.scan(
        fill_receiver_species,
        (receiver_for_pos, ptr0),
        jnp.arange(k, dtype=jnp.int32),
    )

    # -----------------------------------------------------------------------
    # Phase 3: apply reassignments
    # -----------------------------------------------------------------------
    # For each position i in [0, total_moves): set types[picked[i]] = receiver_for_pos[i].
    # We scan over all n_atoms positions and gate writes on i < total_moves.
    new_types = types

    def apply_one(new_t, i):
        do = i < total_moves
        idx = picked[i]
        sp = receiver_for_pos[i]
        new_t = lax.cond(
            do,
            lambda t: t.at[idx].set(sp),
            lambda t: t,
            new_t,
        )
        return new_t, None

    new_types, _ = lax.scan(
        apply_one,
        new_types,
        jnp.arange(n_atoms, dtype=jnp.int32),
    )

    return new_types
