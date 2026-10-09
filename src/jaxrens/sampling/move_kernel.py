"""MoveKernel: declarative description of an MC move type.

Used by the MWG factory (build_mwg) to assemble move kernels and
dispatch weights without the user touching build_kernel directly.
"""

from __future__ import annotations

import dataclasses
from dataclasses import field
from typing import Any, Callable


@dataclasses.dataclass(frozen=True)
class MoveKernel:
    """Describes one move type for the MWG sampler.

    Attributes:
        name: Human-readable label (e.g. "random_walk", "volume").
        build_kernel: Reference to the module's build_kernel function.
            Signature: build_kernel(energy_fn, params, **kernel_kwargs)
            -> step_fn(rng_key, state, likelihood_constraint) -> (state, MoveInfo)
        kernel_kwargs: Extra keyword arguments forwarded to build_kernel
            (e.g. n_reflect for Galilean, n_atoms for volume).
        weight: Relative probability of selecting this move type.
            Weights are normalized to probabilities by the MWG factory.
        step_size: Per-move step size. Injected into MCState.step_size
            before calling the move's step_fn.
        extra_state_fields: Move-specific fields to add to MCState.
            Keys are field names, values are (type, initializer_fn) tuples.
            The initializer is called as initializer(positions, types) and
            must return the initial value for that field.
            Example for Galilean:
                {"direction": (jnp.ndarray, lambda pos, types: jnp.zeros_like(pos))}
            The MWG factory unions extra_state_fields from all descriptors
            to build the MCState class dynamically.
        reject_reasons: Set of reject-reason bucket names this move can emit.
            Valid values: "energy", "cell", "prior". "accepted" (bucket 0) is
            always relevant and is excluded from this set. Used by the monitor
            to suppress uninformative zero-columns in the reject breakdown.
            Default is frozenset({"energy"}) — energy-only rejection.
        mutates: Set of state *aspects* this move writes — a subset of
            {"positions", "cell", "types"} (see jaxrens.constraints). A
            configuration constraint gates this move only when the move
            mutates an aspect the constraint depends on; the pairing is
            computed once, statically, in ``build_mwg``. The default is the
            full aspect set, so a move that forgets to declare its mutations
            is conservatively gated by every constraint (never silently
            skipped) rather than bypassing one.
        affects: ``"all"`` (default) or ``"local"``. A structural fact
            about the move itself, independent of any backend —
            ``build_mwg`` combines this with the backend's declared
            locality capability (see ``jaxrens.backends.locality``) to
            decide whether a per-atom energy cache can be patched
            incrementally instead of the move recomputing the whole
            system's energy from scratch. Every move not opted into this
            keeps the default ``"all"`` — no behaviour change unless a
            move kernel explicitly declares ``"local"``.

            ``"local"`` (``single_atom_swap``, ``alchemical_morph``,
            ``single_atom``, ``single_atom_sweep``): the move recomputes
            its affected-atom set fresh, from the CURRENT
            ``state.positions``/``state.cell``, on every proposal (see
            ``sampling/neighbor_list.py`` module docstring) — there is no
            cached structure computed once from a reference geometry that
            another move could invalidate, so ``"local"`` is safe in ANY
            combination of moves, cell-mutating ones (``volume``, ``shear``,
            ``stretch``) included. ``build_mwg`` never rejects a move-set;
            the only thing that can still prevent a ``"local"`` move from
            running its incremental path is the backend itself not
            supporting the subset-energy query it needs, in which case it
            is silently downgraded to ``"all"`` for that move only, with a
            message logged via the standard ``logging`` module — that move
            simply gets no speedup and runs its ordinary full-recompute
            path, exactly as if it had never declared ``"local"``. See
            ``mwg.py::_resolve_local_affects``/``_downgrade_to_full``.
    """

    name: str
    build_kernel: Callable
    kernel_kwargs: dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    step_size: float = 0.1
    step_size_max: float = 10.0
    min_rate: float = 0.25
    max_rate: float = 0.65
    extra_state_fields: dict[str, tuple[type, Callable]] = field(
        default_factory=dict
    )
    reject_reasons: frozenset[str] = field(
        default_factory=lambda: frozenset({"energy"})
    )
    mutates: frozenset[str] = field(
        default_factory=lambda: frozenset({"positions", "cell", "types"})
    )
    affects: str = "all"
