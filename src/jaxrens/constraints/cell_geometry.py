"""Cell-geometry configuration constraint.

Rejects simulation cells outside the allowed volume-per-atom band or below a
minimum aspect ratio — the hard geometric guard that the cell-move kernels
(volume / shear / stretch) apply to keep walkers in physically reasonable
boxes (``MoveInfo.reject_reason == 2``).

This expresses that guard in the constraints vocabulary (a ``depends_on={"cell"}``
predicate) so its definition lives in one place and the cell kernels *consult*
it rather than each owning the geometry logic.

**Kernel-claimed, not gate-registered.** Every cell-mutating move already
checks this predicate internally — it must, because the result also gates the
move's neighbor-bucket bookkeeping (a sub-min-volume cell can inflate the
neighbor count ~10x, which should not leak into the outer-loop bucket) and its
reject-reason ordering (energy > cell > prior). Registering it with the
central MWG gate as well would evaluate it a second time on exactly those
moves for no benefit, since the kernel has already enforced it. So the
resolver does **not** add this as a gate descriptor; the
:func:`cell_geometry_descriptor` factory exists for completeness / future
moves that don't self-check. The stochastic volume V^N prior
(``reject_reason == 3``) is intentionally *not* part of this — it stays in the
volume kernel, being a proposal-dependent Metropolis factor, not a hard
predicate on the configuration.
"""

from __future__ import annotations

from jaxrens.constraints.base import ConstraintDescriptor
from jaxrens.utils.cell import check_cell_shape


def build_cell_geometry(
    n_atoms: int,
    max_vol_per_atom: float = 100.0,
    min_vol_per_atom: float = 1.0,
    min_aspect: float = 0.5,
) -> object:
    """Build a cell-geometry predicate.

    Args:
        n_atoms: Number of atoms (volume-per-atom needs it; static).
        max_vol_per_atom: Upper bound on volume per atom.
        min_vol_per_atom: Lower bound on volume per atom.
        min_aspect: Minimum cell aspect ratio.

    Returns:
        ``is_valid(positions, types, cell) -> bool`` scalar predicate. Only
        ``cell`` is read; ``positions``/``types`` are accepted to satisfy the
        :class:`~jaxrens.constraints.base.Constraint` signature.
    """

    def is_valid(positions, types, cell):
        return check_cell_shape(
            cell, n_atoms, max_vol_per_atom, min_vol_per_atom, min_aspect
        )

    return is_valid


def cell_geometry_descriptor(
    n_atoms: int,
    max_vol_per_atom: float = 100.0,
    min_vol_per_atom: float = 1.0,
    min_aspect: float = 0.5,
) -> ConstraintDescriptor:
    """Wrap the cell-geometry guard as a :class:`ConstraintDescriptor`.

    Carries ``reject_reason=2`` to keep the existing reject breakdown. Provided
    for future gate use; see the module docstring on why the cell kernels claim
    this constraint rather than letting the gate enforce it.
    """
    return ConstraintDescriptor(
        name="cell_geometry",
        depends_on=frozenset({"cell"}),
        build=build_cell_geometry,
        build_kwargs={
            "n_atoms": n_atoms,
            "max_vol_per_atom": max_vol_per_atom,
            "min_vol_per_atom": min_vol_per_atom,
            "min_aspect": min_aspect,
        },
        reject_reason=2,
    )
