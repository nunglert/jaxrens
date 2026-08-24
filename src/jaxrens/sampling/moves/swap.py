"""Species-swap move kernel for multi-component nested sampling.

Exchanges the identities of two atoms of *different* species.  Positions
and the cell are untouched, so the move explores chemical ordering at
fixed composition and fixed geometry — the productive degree of freedom
for alloys, where a displacement move has to tunnel through a barrier to
achieve what one swap does directly.

Why the pair is drawn by construction
-------------------------------------
The obvious implementation — draw two atom indices uniformly, then reject
the draw if they happen to share a species — applies the species test
after the energy call, so every same-species draw costs a full backend
evaluation and returns a guaranteed rejection.  The wasted fraction is
``sum_s x_s**2`` — 50% for an equimolar binary, and ~97% for a single
solute in 63 host atoms.  It also caps the achievable acceptance rate at
``1 - sum_s x_s**2``, which for an equimolar binary is exactly the 0.5
default ``target_acceptance``.

This kernel instead draws a pair that is unlike *by construction*:

1. Count the atoms of each species (``C`` reductions over ``n_atoms``).
2. Draw an unordered species pair ``(s, t)``, ``s != t``, with weight
   ``n_s * n_t`` — i.e. proportional to how many unlike pairs that species
   combination actually contributes.
3. Draw ``k_s`` uniform in ``[0, n_s)`` and take the ``k_s``-th atom of
   species ``s`` via a running count plus a binary search; likewise for
   ``t``.

Every proposal is a real swap, so the acceptance rate reflects the
physics rather than the composition.

Uniformity and detailed balance
-------------------------------
Step 2 followed by step 3 gives each unlike pair ``{i, j}`` probability::

    (n_s * n_t / Z) * (1 / n_s) * (1 / n_t) = 1 / Z,
    Z = sum_{s<t} n_s * n_t = total number of unlike pairs

— exactly uniform over unlike pairs, independent of species abundance.
A swap permutes ``types`` but leaves the *multiset* of types fixed, so
``Z`` and the set of unlike pairs are invariant, and the reverse proposal
has the same probability as the forward one.  The proposal is therefore
symmetric and the ``E < Emax`` gate alone is a valid NS acceptance test.

(For contrast: selecting the first atom uniformly and the second
uniformly among the unlike atoms — the obvious alternative — is still
symmetric, but it is *not* uniform over pairs once three or more species
are present, because the pair probability then picks up a
``1/(N - n_s) + 1/(N - n_t)`` factor.)

Cost
----
``C`` reductions plus two prefix sums over ``n_atoms``, all O(n_atoms)
with log depth on GPU.  No sort, no permutation, no Gumbel top-k — the
selection is negligible beside the single backend call, and it keeps the
traced graph small, which matters because this kernel is vmapped over
walkers and scanned over MWG sub-steps.

Species scoping
---------------
``species=(code, ...)`` restricts the draw to pairs drawn from those type
codes, mirroring the scoping in ``galilean.build_kernel``.  Registering
one scoped swap per species pair gives each pair its own acceptance
statistics and its own ``weight`` in the MWG mix — useful when one
substitution is cheap and another is nearly always rejected.

Note that this kernel does **not** read ``state.step_size``: a swap has no
continuous magnitude to tune.  Its acceptance rate is invariant under
step-size bisection, so it should be excluded from step-size adaptation
rather than bisected fruitlessly.

Single-walker function, designed for pmap(vmap(vmap(...))) wrapping.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, Int

from jaxrens.sampling.base import MoveInfo
from jaxrens.unvalidated import unvalidated


def _nth_atom_of_species(
    types: Int[Array, " n_atoms"],
    code: Int[Array, ""],
    k: Int[Array, ""],
) -> Int[Array, ""]:
    """Index of the ``k``-th (0-based) atom whose type equals ``code``.

    ``running[i]`` is the number of matching atoms in ``types[: i + 1]``, so
    it is non-decreasing and steps by exactly one at each match.  The first
    position where it reaches ``k + 1`` is therefore the ``k``-th match, and
    a binary search finds it in log depth without materializing an index
    list (which would need a data-dependent shape).

    The clamp only bites when ``k >= count(code)``, which the caller
    prevents for every pair it can actually select; it exists so the
    degenerate no-pair case stays in bounds instead of reading past the end.
    """
    running = jnp.cumsum((types == code).astype(jnp.int32))
    return jnp.minimum(
        jnp.searchsorted(running, k + 1, side="left"), types.shape[0] - 1
    )


@unvalidated(
    concern=("no production NS run has used this move."),
    since="0.2.2",
    clears_when=(
        "Production runs delivering correct physics for a binary system."
    ),
)
def build_kernel(
    backend: Any,
    n_species: int,
    species: Sequence[int] | None = None,
):
    """Build a species-swap kernel.

    Args:
        backend: EnergyBackend instance.
        n_species: Total number of distinct type codes in the system.
        species: Type codes this move may exchange.  ``None`` (the default)
            allows every species.  At least two distinct codes are required
            — a one-species scope could never produce a swap.

    Returns:
        step function: (rng_key, state, Emax) -> (new_state, MoveInfo)

    Raises:
        ValueError: If ``n_species < 2``, or if ``species`` names fewer than
            two distinct codes or a code outside ``[0, n_species)``.
    """
    if n_species < 2:
        raise ValueError(
            f"swap.build_kernel requires n_species >= 2, got {n_species}. "
            "A swap move on a single-species system has nothing to exchange."
        )

    if species is None:
        codes = tuple(range(n_species))
    else:
        codes = tuple(dict.fromkeys(int(c) for c in species))
        out_of_range = [c for c in codes if not 0 <= c < n_species]
        if out_of_range:
            raise ValueError(
                f"swap species codes {out_of_range} outside [0, {n_species}). "
                "Codes are the contiguous type indices stored in "
                "WalkerState.types, not atomic numbers."
            )
        if len(codes) < 2:
            raise ValueError(
                f"swap species={tuple(species)} resolves to {len(codes)} "
                "distinct type code(s); at least two are needed for a swap. "
                "A single-species scope would be a no-op that never accepts."
            )

    # Unordered species pairs, as static Python tuples.  Held as index pairs
    # into ``codes`` (not as codes directly) so the count lookup and the code
    # lookup share one gather each.
    pair_lo, pair_hi = zip(
        *[
            (a, b)
            for a in range(len(codes))
            for b in range(len(codes))
            if a < b
        ]
    )
    n_pairs = len(pair_lo)

    def step(rng_key, state, likelihood_constraint):
        key_pair, key_lo, key_hi = jax.random.split(rng_key, 3)
        types = state.types

        codes_arr = jnp.asarray(codes, dtype=types.dtype)
        lo_arr = jnp.asarray(pair_lo, dtype=jnp.int32)
        hi_arr = jnp.asarray(pair_hi, dtype=jnp.int32)

        # (C,) occupancy of each in-scope species, recomputed every call so
        # the kernel stays correct alongside composition-changing moves
        # (alchemical_morph) rather than trusting a build-time table.
        counts = jnp.stack(
            [jnp.sum(types == c).astype(jnp.int32) for c in codes]
        )

        # Weight each species pair by the number of unlike atom pairs it
        # contributes; this is what makes the atom pair uniform overall.
        weights = (counts[lo_arr] * counts[hi_arr]).astype(jnp.float32)
        total = jnp.sum(weights)
        has_pair = total > 0

        # Inverse-CDF draw.  ``side="right"`` skips zero-weight pairs: pair p
        # is hit iff cumulative[p - 1] <= u < cumulative[p], an empty interval
        # when its weight is zero.  ``total`` is replaced by 1.0 when the
        # walker holds no unlike pair at all, so the draw stays finite; the
        # resulting proposal is discarded via ``has_pair`` below.
        cumulative = jnp.cumsum(weights)
        u = jax.random.uniform(key_pair) * jnp.where(has_pair, total, 1.0)
        pair_idx = jnp.minimum(
            jnp.searchsorted(cumulative, u, side="right"), n_pairs - 1
        )

        slot_lo, slot_hi = lo_arr[pair_idx], hi_arr[pair_idx]
        code_lo, code_hi = codes_arr[slot_lo], codes_arr[slot_hi]

        # maximum(..., 1) only guards the no-pair case; any selectable pair
        # has both counts >= 1 by construction.
        k_lo = jax.random.randint(
            key_lo, (), 0, jnp.maximum(counts[slot_lo], 1)
        )
        k_hi = jax.random.randint(
            key_hi, (), 0, jnp.maximum(counts[slot_hi], 1)
        )

        idx_lo = _nth_atom_of_species(types, code_lo, k_lo)
        idx_hi = _nth_atom_of_species(types, code_hi, k_hi)

        # code_lo != code_hi, so idx_lo != idx_hi and the two writes commute.
        new_types = types.at[idx_lo].set(code_hi).at[idx_hi].set(code_lo)

        result = backend(
            state.positions,
            new_types,
            state.cell,
            state.max_neighbors,
            ensemble_params=state.ensemble_params,
        )

        # ``energy < constraint`` (not ``>= constraint`` negated) so a NaN
        # energy rejects: NaN fails every comparison, and the accept-shaped
        # test is the one where that failure lands on "reject".
        accepted = (result.energy < likelihood_constraint) & has_pair

        new_state = state.set(
            types=jnp.where(accepted, new_types, types),
            energy=jnp.where(accepted, result.energy, state.energy),
            # Neighbour-list diagnostics describe the configuration that was
            # actually evaluated, so they survive rejection — except in the
            # no-pair case, where the evaluated types are meaningless.
            max_neighbor_count=jnp.maximum(
                state.max_neighbor_count,
                jnp.where(has_pair, result.max_neighbor_count, 0),
            ),
            overflow=state.overflow | (result.overflow & has_pair),
        )

        info = MoveInfo(
            accepted=accepted,
            log_likelihood=-new_state.energy,
            n_evaluations=1,
            reject_reason=jnp.where(accepted, 0, 1).astype(jnp.int32),
        )

        return new_state, info

    return step
