"""Fixed-Emax MCMC burn-in to decorrelate live walkers before NS proper.

The burn-in runs the existing MWG step_fn at a constant Emax computed once
from the live-walker population. No new MCMC kernel; no modification to ns_step.

Public API:
    initial_walk(key, ns_state, step_fn, ...)  ->  NSState

Parallelism axes:
    walker_batch_size: chunk the walker vmap via lax.map(batch_size=N).
    run_batch_size: chunk the run vmap via lax.map(batch_size=M) when batched=True.
    Both default to None (full vmap — fastest when memory allows).
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp

from jaxrens.sampling.adaptation.stepsize_handler import adjust_step_size
from jaxrens.state.ns import NSState


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _one_walk(
    key: jax.Array,
    ns_state: NSState,
    step_fn: Callable,
    walklength: int,
    emax: jnp.ndarray,
    walker_batch_size: int | None,
) -> tuple[NSState, Any]:
    """Advance all walkers in a single-run NSState by walklength steps.

    Args:
        key: PRNG key. Consumed to produce (n_walkers, walklength) sub-keys.
        ns_state: Single-run NSState. Population shape (n_walkers, ...).
        step_fn: MWG step fn. Signature: (rng_key, mcstate, emax) -> (mcstate, info).
        walklength: Number of MCMC steps per walker.
        emax: Scalar energy ceiling.
        walker_batch_size: If None, vmap over all walkers. If int, chunk via
            lax.map(batch_size=walker_batch_size). Must divide n_walkers.

    Returns:
        (new_ns_state, last_accepted) where last_accepted is (n_walkers, walklength).
    """
    population = ns_state.population
    n_walkers = ns_state.n_walkers

    walker_keys = jax.random.split(key, n_walkers)
    # chain_keys: (n_walkers, walklength)
    chain_keys = jax.vmap(lambda k: jax.random.split(k, walklength))(walker_keys)

    def walker_fn(walker_state: Any, wkeys: jax.Array) -> tuple[Any, Any]:
        """Scan step_fn over walklength keys for one walker."""
        def scan_body(state, k):
            new_state, info = step_fn(k, state, emax)
            return new_state, info.accepted
        final, accepted_arr = jax.lax.scan(scan_body, walker_state, wkeys)
        return final, accepted_arr

    if walker_batch_size is None:
        new_population, accepted = jax.vmap(walker_fn)(population, chain_keys)
    else:
        # lax.map processes (population[i], chain_keys[i]) pairs in chunks.
        new_population, accepted = jax.lax.map(
            lambda x: walker_fn(x[0], x[1]),
            (population, chain_keys),
            batch_size=walker_batch_size,
        )

    return ns_state.set(population=new_population), accepted


def _apply_adaptation(
    ns_state: NSState,
    emax: jnp.ndarray,
    key: jax.Array,
    per_move_fns: list[Callable],
    adaptation_policies: tuple,
    current_step_sizes: jnp.ndarray,
    adjust_n_samples: int,
    adjust_max_rounds: int,
    jit_adjust_fns: list[Callable],
    batched: bool,
) -> tuple[NSState, jnp.ndarray]:
    """Update step sizes for all move types.

    When batched=False: operates on a single run.
    When batched=True: vmaps over the leading run axis for each move.

    Args:
        ns_state: Current NSState (single-run or batched).
        emax: Scalar (single-run) or (n_runs,) (batched) energy ceiling.
        key: PRNG key.
        per_move_fns: Per-move step functions.
        adaptation_policies: Tuple of ResolvedAdaptationPolicy.
        current_step_sizes: (n_move_types,) for single-run, (n_runs, n_move_types) batched.
        adjust_n_samples: Trial walker count per adaptation round.
        adjust_max_rounds: Max bisection rounds.
        jit_adjust_fns: Pre-JIT'd adjust functions, one per move.
        batched: Whether ns_state has a leading run axis.

    Returns:
        (new_ns_state, new_step_sizes)
    """
    population = ns_state.population

    if not batched:
        n_walkers = ns_state.n_walkers
        for move_idx, policy in enumerate(adaptation_policies):
            key, key_adj = jax.random.split(key)
            new_ss, _ = jit_adjust_fns[move_idx](
                population,
                per_move_fns[move_idx],
                current_step_sizes[move_idx],
                emax,
                key_adj,
                adjust_n_samples,
                policy.min_rate,
                policy.max_rate,
                policy.adjust_factor,
                policy.step_size_max,
                adjust_max_rounds,
            )
            current_step_sizes = current_step_sizes.at[move_idx].set(new_ss)

        new_ss_pop = jnp.broadcast_to(
            current_step_sizes,
            (n_walkers, current_step_sizes.shape[0]),
        )
        new_population = population.set(step_sizes=new_ss_pop)
        return ns_state.set(population=new_population), current_step_sizes

    else:
        # batched: current_step_sizes is (n_runs, n_move_types).
        # emax is (n_runs,). population fields are (n_runs, n_walkers, ...).
        n_runs = emax.shape[0]
        n_walkers = population.step_sizes.shape[1]

        for move_idx, policy in enumerate(adaptation_policies):
            key, key_adj = jax.random.split(key)
            run_keys = jax.random.split(key_adj, n_runs)

            def adjust_one_run(run_pop, run_ss, run_emax, run_key):
                new_ss, _ = jit_adjust_fns[move_idx](
                    run_pop,
                    per_move_fns[move_idx],
                    run_ss,
                    run_emax,
                    run_key,
                    adjust_n_samples,
                    policy.min_rate,
                    policy.max_rate,
                    policy.adjust_factor,
                    policy.step_size_max,
                    adjust_max_rounds,
                )
                return new_ss

            per_run_ss = jax.vmap(adjust_one_run)(
                population,
                current_step_sizes[:, move_idx],
                emax,
                run_keys,
            )
            current_step_sizes = current_step_sizes.at[:, move_idx].set(per_run_ss)

        # Broadcast: (n_runs, n_move_types) -> (n_runs, n_walkers, n_move_types)
        new_ss_pop = jnp.broadcast_to(
            current_step_sizes[:, None, :],
            (n_runs, n_walkers, current_step_sizes.shape[1]),
        )
        new_population = population.set(step_sizes=new_ss_pop)
        return ns_state.set(population=new_population), current_step_sizes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def initial_walk(
    key: jax.Array,
    ns_state: NSState,
    step_fn: Callable,
    *,
    n_walks: int,
    walklength: int,
    adjust_interval: int,
    emax_offset_per_atom: float,
    n_atoms: int,
    batched: bool = False,
    walker_batch_size: int | None = None,
    run_batch_size: int | None = None,
    per_move_fns: list[Callable] | None = None,
    adaptation_policies: tuple | None = None,
    adjust_n_samples: int = 50,
    adjust_max_rounds: int = 15,
) -> NSState:
    """Run fixed-Emax MCMC to decorrelate walkers from their initialization.

    Emax = max(live_energies) + emax_offset_per_atom * n_atoms, held constant
    across all n_walks * walklength steps.

    Parallelism:
        walker_batch_size=None: vmap over all walkers simultaneously (fastest).
        walker_batch_size=N: chunk via lax.map(batch_size=N). Must divide n_walkers.
        run_batch_size=None (batched=True): vmap over all runs simultaneously.
        run_batch_size=M (batched=True): chunk via lax.map(batch_size=M). Must divide n_runs.

    Args:
        key: JAX PRNG key.
        ns_state: NSState after init_ns. Single-run when batched=False
            (population shape (n_walkers, ...)); batched-run when batched=True
            (population shape (n_runs, n_walkers, ...)).
        step_fn: MWG step function from build_mwg().
            Signature: (rng_key, mcstate, emax) -> (mcstate, MoveInfo).
        n_walks: Number of outer walk iterations. n_walks=0 is a no-op.
        walklength: MCMC steps per walk per walker.
        adjust_interval: Run step-size adaptation every this many walks (>0).
        emax_offset_per_atom: Added to max(live_energies)/max across runs
            to set Emax. Units: energy per atom.
        n_atoms: Number of atoms per walker (Python int).
        batched: Whether ns_state has a leading run axis (n_runs, ...).
        walker_batch_size: Chunk size for walker vmap. None = full vmap.
            Must divide n_walkers evenly.
        run_batch_size: Chunk size for run vmap. None = full vmap over runs.
            Must divide n_runs evenly. Ignored when batched=False.
        per_move_fns: Per-move step functions (third return of build_mwg).
            Required for step-size adaptation; None disables it.
        adaptation_policies: Tuple of ResolvedAdaptationPolicy, one per move.
            Required when per_move_fns is not None.
        adjust_n_samples: Number of trial walkers per adaptation round.
        adjust_max_rounds: Max bisection rounds per adaptation call.

    Returns:
        New NSState with live walkers advanced. Same pytree shape as input.
        Dead-point arrays, log_evidence, iteration, n_dead are unchanged.

    Raises:
        ValueError: If walker_batch_size does not evenly divide n_walkers, or
            if run_batch_size does not evenly divide n_runs.
    """
    if n_walks == 0:
        return ns_state

    # --- Validate chunking parameters ---
    if batched:
        n_runs = ns_state.population.energy.shape[0]
        n_walkers = ns_state.population.energy.shape[1]
    else:
        n_walkers = ns_state.n_walkers

    if walker_batch_size is not None and n_walkers % walker_batch_size != 0:
        raise ValueError(
            f"walker_batch_size={walker_batch_size} does not evenly divide "
            f"n_walkers={n_walkers}. "
            f"Choose a divisor of {n_walkers}."
        )

    if batched and run_batch_size is not None and n_runs % run_batch_size != 0:
        raise ValueError(
            f"run_batch_size={run_batch_size} does not evenly divide "
            f"n_runs={n_runs}. "
            f"Choose a divisor of {n_runs}."
        )

    # --- Compute fixed Emax ---
    population = ns_state.population
    emax_offset = float(emax_offset_per_atom) * n_atoms

    if not batched:
        emax = jnp.max(population.energy) + emax_offset
    else:
        # population.energy shape: (n_runs, n_walkers)
        emax = jnp.max(population.energy, axis=1) + emax_offset  # (n_runs,)

    # --- Setup adaptation ---
    use_adaptation = (
        per_move_fns is not None
        and adaptation_policies is not None
        and adjust_interval > 0
    )

    if use_adaptation:
        n_moves = len(per_move_fns)
        jit_adjust_fns = [
            jax.jit(adjust_step_size, static_argnums=(1, 5, 6, 7, 8, 9, 10))
            for _ in range(n_moves)
        ]
        if not batched:
            current_step_sizes = population.step_sizes[0]  # (n_move_types,)
        else:
            current_step_sizes = population.step_sizes[:, 0, :]  # (n_runs, n_move_types)
    else:
        jit_adjust_fns = None
        current_step_sizes = None

    # --- Build one-walk function for chosen parallelism ---
    if not batched:
        # Single-run: _one_walk handles walker chunking internally.
        jit_one_walk = jax.jit(
            lambda k, s: _one_walk(k, s, step_fn, walklength, emax, walker_batch_size)
        )

    else:
        # Batched: vmap or chunked-vmap over the run axis.
        # emax is (n_runs,); must be vmapped in_axes=0 alongside ns_state.
        def _batched_one_walk(key, ns_state_batched, emax_batched):
            """Apply _one_walk vmapped over runs."""
            def one_run(run_key, run_state, run_emax):
                return _one_walk(
                    run_key, run_state, step_fn, walklength, run_emax, walker_batch_size
                )

            if run_batch_size is None:
                run_keys = jax.random.split(key, n_runs)
                return jax.vmap(one_run)(run_keys, ns_state_batched, emax_batched)
            else:
                run_keys = jax.random.split(key, n_runs)
                return jax.lax.map(
                    lambda x: one_run(x[0], x[1], x[2]),
                    (run_keys, ns_state_batched, emax_batched),
                    batch_size=run_batch_size,
                )

        jit_one_walk = jax.jit(
            lambda k, s, e: _batched_one_walk(k, s, e)
        )

    # --- Outer walk loop ---
    for walk_i in range(n_walks):
        if use_adaptation and walk_i > 0 and walk_i % adjust_interval == 0:
            key, key_adapt = jax.random.split(key)
            ns_state, current_step_sizes = _apply_adaptation(
                ns_state=ns_state,
                emax=emax,
                key=key_adapt,
                per_move_fns=per_move_fns,
                adaptation_policies=adaptation_policies,
                current_step_sizes=current_step_sizes,
                adjust_n_samples=adjust_n_samples,
                adjust_max_rounds=adjust_max_rounds,
                jit_adjust_fns=jit_adjust_fns,
                batched=batched,
            )

        key, sub = jax.random.split(key)

        if not batched:
            ns_state, _ = jit_one_walk(sub, ns_state)
        else:
            ns_state, _ = jit_one_walk(sub, ns_state, emax)

    return ns_state
