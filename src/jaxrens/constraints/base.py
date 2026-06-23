"""Composable configuration constraints for nested sampling.

A *constraint* is a hard predicate on a walker configuration — a function
``(positions, types, cell) -> bool`` that is jit/vmap/pmap safe and returns a
scalar boolean (``True`` == the configuration is allowed). Constraints
restrict the support of the prior the chain explores; a proposal that would
move a walker into a forbidden region is rejected exactly like one that
violates the likelihood threshold.

Each constraint type is described by a :class:`ConstraintDescriptor`, which
carries:

* ``depends_on`` — the set of state *aspects* the predicate reads
  (``"positions"``, ``"cell"``, ``"types"``). A move is gated by a constraint
  only when the move *mutates* an aspect the constraint *depends on*; this
  pairing is computed once, statically, when the sampler is assembled (see
  :func:`make_move_gate` and ``build_mwg``). A move that cannot affect a
  constraint's inputs provably preserves it, so the check is skipped entirely
  — not merely short-circuited at runtime.
* ``reject_reason`` — the ``MoveInfo.reject_reason`` bucket to report when
  this constraint is the cause of a rejection (default ``4`` = configuration
  constraint; cell-geometry uses ``2`` to stay backward-compatible with the
  existing reject breakdown).

Constraints whose enforcement is *claimed* by a move kernel (e.g. cell
geometry, which the cell-move kernels check internally so they can gate
their own neighbor-bucket bookkeeping) are simply not registered as gate
descriptors — that keeps them evaluated exactly once rather than twice.
"""

from __future__ import annotations

import dataclasses
from dataclasses import field
from typing import Any, Callable, Protocol

import jax.numpy as jnp

# Valid state aspects a constraint may depend on / a move may mutate.
ASPECTS: frozenset[str] = frozenset({"positions", "cell", "types"})


class Constraint(Protocol):
    """A hard predicate on a single walker configuration.

    Operates on one walker; batched via vmap/pmap by the sampler wrapper.
    Returns a scalar boolean array — ``True`` when the configuration is
    allowed.
    """

    def __call__(
        self,
        positions: jnp.ndarray,
        types: jnp.ndarray,
        cell: jnp.ndarray,
    ) -> jnp.ndarray:
        ...


@dataclasses.dataclass(frozen=True)
class ConstraintDescriptor:
    """Declarative description of one constraint type for the sampler.

    Attributes:
        name: Human-readable label (e.g. ``"minimum_distance"``).
        depends_on: State aspects the predicate reads. Must be a subset of
            :data:`ASPECTS`. Used to pair the constraint with the moves that
            can violate it.
        build: Factory called as ``build(**build_kwargs)`` to produce the
            :class:`Constraint` predicate. Deferred so the (possibly array-
            valued) parameters are captured once at sampler-assembly time.
        build_kwargs: Keyword arguments forwarded to ``build``.
        reject_reason: ``MoveInfo.reject_reason`` bucket reported when this
            constraint causes a rejection. Default ``4``.
    """

    name: str
    depends_on: frozenset[str]
    build: Callable[..., Constraint]
    build_kwargs: dict[str, Any] = field(default_factory=dict)
    reject_reason: int = 4

    def __post_init__(self) -> None:
        unknown = self.depends_on - ASPECTS
        if unknown:
            raise ValueError(
                f"Constraint {self.name!r} declares unknown depends_on "
                f"aspects {sorted(unknown)}; valid aspects are {sorted(ASPECTS)}."
            )


def make_move_gate(
    descriptors: tuple[ConstraintDescriptor, ...],
    mutates: frozenset[str],
) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple] | None:
    """Build the constraint gate for a single move type.

    Selects the descriptors whose ``depends_on`` intersects this move's
    ``mutates`` set — i.e. the constraints the move could actually violate —
    instantiates their predicates once, and returns a combined gate.

    Args:
        descriptors: All registered gate constraints.
        mutates: State aspects this move type writes.

    Returns:
        ``None`` when no registered constraint depends on an aspect this move
        mutates (the caller then uses its no-op fast path), otherwise a
        function ``(positions, types, cell) -> (ok, reject_reason)`` where
        ``ok`` is a scalar bool (all constraints satisfied) and
        ``reject_reason`` is the int32 bucket of the *first* violated
        constraint (0 when ``ok``).
    """
    relevant = [d for d in descriptors if d.depends_on & mutates]
    if not relevant:
        return None

    # Instantiate each predicate once, pairing it with its reject bucket.
    preds = [
        (int(d.reject_reason), d.build(**d.build_kwargs)) for d in relevant
    ]

    def gate(positions, types, cell):
        ok = jnp.asarray(True)
        reason = jnp.int32(0)
        for bucket, pred in preds:
            this_ok = jnp.asarray(pred(positions, types, cell), dtype=bool)
            # Record the bucket of the first failure only (ok still True).
            reason = jnp.where(ok & ~this_ok, jnp.int32(bucket), reason)
            ok = ok & this_ok
        return ok, reason

    return gate
