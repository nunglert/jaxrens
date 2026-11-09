"""Replica exchange moves across parallel NS runs.

Unlike standard moves that operate on a single walker, replica exchange
swaps walkers between different parallel runs (across the P dimension).
This improves mixing when runs have different energy constraints.

Structure:
    One driver, :func:`replica_exchange_step`, runs the swap schedule
    (even/odd neighbour pairs, one random walker per run, per-pair
    statistics) for every flavor.  The flavor-specific part is a
    :class:`SwapKernel`, whose only job is :meth:`SwapKernel.propose`: given
    one walker from each run of a pair, build the walker that would *enter*
    each run, with its energy expressed in that run's ensemble.  Acceptance
    is then the same for every flavor: each incoming walker must lie below
    the receiving run's Emax.

Kernels:
    ``EnsembleSwap`` covers pressure RE and semi-grand (μ) RE.  The walkers
    trade runs unchanged and their stored energies are re-based to the
    receiving run's pressure and chemical potentials (zero backend calls).
    ``XRENSSwap`` morphs the incoming walker's types to the receiving run's
    target composition and re-evaluates its energy (two backend calls).

Use :func:`make_swap_kernel` to build the kernel for an ``InterREConfig``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, Key

from jaxrens.backends.ensemble import rebase_ensemble_energy
from jaxrens.sampling.morph import morph_types_to_composition
from jaxrens.unvalidated import unvalidated

# ---------------------------------------------------------------------------
# Swap state and kernel contract
# ---------------------------------------------------------------------------


class SwapWalker(NamedTuple):
    """The part of a walker that moves between runs in a swap.

    A JAX pytree.  The same type describes a single walker (as seen by
    :meth:`SwapKernel.propose`) and the whole ``(R, K, ...)`` population
    (as seen by :func:`replica_exchange_step`).  ``cell`` is ``None`` for
    non-periodic systems.

    Attributes:
        positions: Cartesian coordinates, shape ``(*B, N, 3)``.
        types: Integer species labels, shape ``(*B, N)``.
        energy: Stored energy in the frame of the run the walker is in
            (``U + P·V − μ·N``, the ``EnsembleTerm`` convention), shape
            ``(*B,)``.
        cell: Unit cell, shape ``(*B, 3, 3)``, or ``None``.
    """

    positions: Float[Array, "*B N 3"]
    types: Int[Array, "*B N"]
    energy: Float[Array, "*B"]
    cell: Float[Array, "*B 3 3"] | None = None


class SwapKernel(ABC):
    """One flavor of inter-RE swap.

    Class attributes:
        required_params: Per-run ensemble-param keys the kernel cannot work
            without.  The manager raises at construction if one is missing.
        optional_params: Per-run keys the kernel uses when present.
        n_energy_evals: Backend energy evaluations per attempted swap.
    """

    required_params: tuple[str, ...] = ()
    optional_params: tuple[str, ...] = ()
    n_energy_evals: int = 0

    @property
    def param_keys(self) -> tuple[str, ...]:
        """All per-run ensemble-param keys the kernel reads."""
        return self.required_params + self.optional_params

    @abstractmethod
    def propose(
        self,
        walker_i: SwapWalker,
        walker_j: SwapWalker,
        params_i: dict[str, Any],
        params_j: dict[str, Any],
        rng_key: Key[Array, ""],
        backend: Any,
    ) -> tuple[SwapWalker, SwapWalker]:
        """Build the walkers that would enter runs i and j.

        Args:
            walker_i: Walker picked from run i, energy in run i's frame.
            walker_j: Walker picked from run j, energy in run j's frame.
            params_i: Run i's ensemble params (keys from :attr:`param_keys`).
            params_j: Run j's ensemble params.
            rng_key: PRNG key for kernels that need randomness.
            backend: Energy backend, for kernels that re-evaluate energies.

        Returns:
            ``(into_i, into_j)``: ``into_i`` is built from ``walker_j`` and
            carries its energy in run i's frame; ``into_j`` symmetrically.
        """


class EnsembleSwap(SwapKernel):
    """Pressure / semi-grand RE: walkers trade runs, energies are re-based.

    The two walkers move unchanged (positions, types, cell); each run keeps
    its own pressure and chemical potentials.  The stored energy of a walker
    entering run r becomes ``U + P_r·V − μ_r·N``
    (:func:`~jaxrens.backends.ensemble.rebase_ensemble_energy`), so every
    stored energy in a run stays in that run's frame and the run's Emax
    contour stays well-defined.  Missing params drop their term: without
    pressures and chemical potentials this is plain energy RE.

    Zero backend calls.

    Args:
        required_params: Per-run keys to insist on, e.g.
            ``("chemical_potentials",)`` for semi-grand RE.
    """

    optional_params = ("pressure", "chemical_potentials")

    def __init__(self, required_params: tuple[str, ...] = ()) -> None:
        self.required_params = tuple(required_params)
        self.optional_params = tuple(
            k for k in EnsembleSwap.optional_params if k not in required_params
        )

    def propose(
        self,
        walker_i: SwapWalker,
        walker_j: SwapWalker,
        params_i: dict[str, Any],
        params_j: dict[str, Any],
        rng_key: Key[Array, ""],
        backend: Any,
    ) -> tuple[SwapWalker, SwapWalker]:
        into_i = _rebased(walker_j, params_j, params_i)
        into_j = _rebased(walker_i, params_i, params_j)
        return into_i, into_j


def _rebased(walker: SwapWalker, params_from, params_to) -> SwapWalker:
    """``walker`` with its stored energy moved from one run's frame to another's."""
    energy = rebase_ensemble_energy(
        walker.energy, walker.cell, walker.types, params_from, params_to
    )
    return walker._replace(energy=energy)


class XRENSSwap(SwapKernel):
    """Composition-swap RE: morph types to the receiving run, re-evaluate.

    The walker entering run r gets its types morphed to run r's
    ``target_composition`` and its energy re-evaluated by the backend under
    run r's remaining ensemble params, so the result is directly in run r's
    frame.

    Two backend energy evaluations per attempted swap.

    Args:
        n_species: Number of species labels; must match the width of every
            ``target_composition``.
    """

    required_params = ("target_composition",)
    optional_params = ("pressure", "chemical_potentials")
    n_energy_evals = 2

    @unvalidated(
        concern=("no production NS run has used this move."),
        since="0.2.2",
        clears_when=(
            "Production runs delivering correct physics for a binary system."
        ),
    )
    def __init__(self, n_species: int) -> None:
        if not isinstance(n_species, int) or n_species < 1:
            raise ValueError(
                f"n_species must be a positive Python int, got {n_species!r}"
            )
        self.n_species = n_species

    def _enter(
        self,
        walker: SwapWalker,
        params: dict[str, Any],
        rng_key: Key[Array, ""],
        backend: Any,
    ) -> SwapWalker:
        target = jnp.asarray(params["target_composition"], dtype=jnp.int32)
        if target.shape[-1] != self.n_species:
            raise ValueError(
                f"XRENSSwap(n_species={self.n_species}) got a "
                f"target_composition of width {target.shape[-1]}."
            )
        types = morph_types_to_composition(
            rng_key, walker.types, target, self.n_species
        )
        backend_params = {
            k: v for k, v in params.items() if k != "target_composition"
        } or None
        energy = backend(
            walker.positions,
            types,
            walker.cell,
            0,  # max_neighbors
            ensemble_params=backend_params,
        ).energy
        return walker._replace(types=types, energy=energy)

    def propose(
        self,
        walker_i: SwapWalker,
        walker_j: SwapWalker,
        params_i: dict[str, Any],
        params_j: dict[str, Any],
        rng_key: Key[Array, ""],
        backend: Any,
    ) -> tuple[SwapWalker, SwapWalker]:
        key_i, key_j = jax.random.split(rng_key)
        into_i = self._enter(walker_j, params_i, key_i, backend)
        into_j = self._enter(walker_i, params_j, key_j, backend)
        return into_i, into_j


def make_swap_kernel(config: Any) -> SwapKernel:
    """Build the swap kernel for an ``InterREConfig``.

    Args:
        config: :class:`~jaxrens.state.config.InterREConfig`.

    Returns:
        ``EnsembleSwap`` for ``"pressure"`` and ``"semi_grand"``,
        ``XRENSSwap`` for ``"xrens"``.

    Raises:
        ValueError: If the flavor's per-run inputs are missing.
        NotImplementedError: For an unknown flavor.
    """
    if config.flavor == "pressure":
        return EnsembleSwap()
    if config.flavor == "semi_grand":
        if config.chemical_potentials is None:
            raise ValueError(
                "inter_re flavor 'semi_grand' requires chemical_potentials in "
                "InterREConfig. Each run must have a per-species μ vector."
            )
        return EnsembleSwap(required_params=("chemical_potentials",))
    if config.flavor == "xrens":
        if config.composition_targets is None:
            raise ValueError(
                "inter_re flavor 'xrens' requires composition_targets in "
                "InterREConfig. Each run must have a target composition."
            )
        return XRENSSwap(n_species=len(config.composition_targets[0]))
    raise NotImplementedError(
        f"inter_re flavor {config.flavor!r} is not yet implemented. "
        f"Supported flavors: 'pressure', 'xrens', 'semi_grand'."
    )


# ---------------------------------------------------------------------------
# Pair schedule
# ---------------------------------------------------------------------------


def get_swap_pairs(n_runs: int, phase: int) -> Int[Array, "n_pairs 2"]:
    """Get non-overlapping swap pairs for replica exchange.

    Phase 0 (even): [(0,1), (2,3), (4,5), ...]
    Phase 1 (odd):  [(1,2), (3,4), (5,6), ...]

    Args:
        n_runs: Number of parallel runs.
        phase: 0 for even pairs, 1 for odd pairs.

    Returns:
        Array of shape (n_pairs, 2) with run indices.
    """
    starts = jnp.arange(phase, n_runs - 1, 2)
    pairs = jnp.stack([starts, starts + 1], axis=-1)
    return pairs


def _pad_pairs(
    pairs: Int[Array, "k 2"], n_valid: int, max_len: int
) -> tuple[Int[Array, "max_len 2"], Bool[Array, "max_len"]]:
    """Pad a (k, 2) pair array to (max_len, 2) and return it with a validity mask."""
    if pairs.shape[0] < max_len:
        padding = jnp.zeros((max_len - pairs.shape[0], 2), dtype=pairs.dtype)
        pairs = jnp.concatenate([pairs, padding], axis=0)
    mask = jnp.arange(max_len) < n_valid
    return pairs, mask


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def replica_exchange_step(
    rng_key: Key[Array, ""],
    walkers: SwapWalker,
    emax: Float[Array, "R"],
    run_params: dict[str, Array],
    kernel: SwapKernel,
    backend: Any = None,
    n_swap_cycles: int = 1,
) -> tuple[SwapWalker, dict[str, Array]]:
    """Run inter-replica swaps across R parallel runs.

    Each cycle attempts the even pairs (0,1), (2,3), ... and then the odd
    pairs (1,2), (3,4), ....  For each pair one walker is drawn uniformly
    from each run, ``kernel.propose`` builds the walkers that would enter
    the two runs, and the swap is accepted iff each incoming walker's
    energy is below the receiving run's Emax.

    Args:
        rng_key: JAX PRNG key.
        walkers: Population with leaves shaped ``(R, K, ...)``; energies in
            each run's own frame.
        emax: ``(R,)`` NS contour of each run.
        run_params: Per-run ensemble params, each leaf shaped ``(R, ...)``
            (e.g. ``{"pressure": (R,), "chemical_potentials": (R, S)}``).
        kernel: Swap flavor.
        backend: Energy backend, passed to ``kernel.propose``.
        n_swap_cycles: Number of even+odd cycles.

    Returns:
        ``(new_walkers, swap_info)``; ``swap_info`` has ``n_accepted``,
        ``n_attempted``, ``n_energy_evals``, and per-pair
        ``n_accepted_per_pair`` / ``n_attempted_per_pair`` of shape
        ``(R - 1,)`` indexed by ``min(i, j)``.
    """
    n_runs, n_walkers = walkers.positions.shape[:2]
    n_pairs = max(n_runs - 1, 0)

    even_pairs = get_swap_pairs(n_runs, 0)
    odd_pairs = get_swap_pairs(n_runs, 1)
    n_even = even_pairs.shape[0]
    n_odd = odd_pairs.shape[0]
    max_pairs = max(n_even, n_odd)

    if max_pairs == 0:
        zeros = jnp.zeros(n_pairs, dtype=jnp.int32)
        return walkers, _swap_info(zeros, zeros, kernel)

    even_pairs, even_mask = _pad_pairs(even_pairs, n_even, max_pairs)
    odd_pairs, odd_mask = _pad_pairs(odd_pairs, n_odd, max_pairs)

    def _attempt_one_swap(carry, swap_input):
        state, acc_per_pair = carry
        pair, valid, swap_key = swap_input
        run_i, run_j = pair[0], pair[1]

        k1, k2 = jax.random.split(swap_key)
        wi = jax.random.randint(k1, (), 0, n_walkers)
        wj = jax.random.randint(k2, (), 0, n_walkers)

        walker_i = jax.tree.map(lambda a: a[run_i, wi], state)
        walker_j = jax.tree.map(lambda a: a[run_j, wj], state)
        params_i = {k: v[run_i] for k, v in run_params.items()}
        params_j = {k: v[run_j] for k, v in run_params.items()}

        into_i, into_j = kernel.propose(
            walker_i,
            walker_j,
            params_i,
            params_j,
            jax.random.fold_in(swap_key, 2),
            backend,
        )
        do_swap = (
            valid
            & (into_i.energy < emax[run_i])
            & (into_j.energy < emax[run_j])
        )

        def _write(arr, new_i, new_j):
            arr = arr.at[run_i, wi].set(
                jnp.where(do_swap, new_i, arr[run_i, wi])
            )
            return arr.at[run_j, wj].set(
                jnp.where(do_swap, new_j, arr[run_j, wj])
            )

        state = jax.tree.map(_write, state, into_i, into_j)
        # Pairs are always (k, k+1), so pair_id = min(i, j); even and odd
        # phases write disjoint ids and padding entries have valid=False.
        pair_id = jnp.minimum(run_i, run_j)
        acc_per_pair = acc_per_pair.at[pair_id].add(do_swap.astype(jnp.int32))
        return (state, acc_per_pair), None

    def _do_one_phase(state, acc_per_pair, att_per_pair, pairs, mask, key):
        pair_keys = jax.random.split(key, max_pairs)
        (state, acc_per_pair), _ = jax.lax.scan(
            _attempt_one_swap, (state, acc_per_pair), (pairs, mask, pair_keys)
        )
        pair_ids = jnp.minimum(pairs[:, 0], pairs[:, 1])
        att_per_pair = att_per_pair.at[pair_ids].add(mask.astype(jnp.int32))
        return state, acc_per_pair, att_per_pair

    def _do_one_cycle(carry, cycle_key):
        state, acc_per_pair, att_per_pair = carry
        even_key, odd_key = jax.random.split(cycle_key)
        state, acc_per_pair, att_per_pair = _do_one_phase(
            state, acc_per_pair, att_per_pair, even_pairs, even_mask, even_key
        )
        state, acc_per_pair, att_per_pair = _do_one_phase(
            state, acc_per_pair, att_per_pair, odd_pairs, odd_mask, odd_key
        )
        return (state, acc_per_pair, att_per_pair), None

    cycle_keys = jax.random.split(rng_key, n_swap_cycles)
    zeros = jnp.zeros(n_pairs, dtype=jnp.int32)
    (walkers, acc_per_pair, att_per_pair), _ = jax.lax.scan(
        _do_one_cycle, (walkers, zeros, zeros), cycle_keys
    )
    return walkers, _swap_info(acc_per_pair, att_per_pair, kernel)


def _swap_info(acc_per_pair, att_per_pair, kernel: SwapKernel) -> dict:
    n_attempted = jnp.sum(att_per_pair)
    return {
        "n_accepted": jnp.sum(acc_per_pair),
        "n_attempted": n_attempted,
        "n_energy_evals": n_attempted * kernel.n_energy_evals,
        "n_accepted_per_pair": acc_per_pair,
        "n_attempted_per_pair": att_per_pair,
    }
