"""Shared accept/reject logic for cell-deforming moves.

``volume``, ``shear`` and ``stretch`` all propose a new cell + scaled
positions, evaluate the backend, and then apply the same acceptance and
state-update contract.  That contract lives here so the three kernels stay in
sync.
"""

from __future__ import annotations

import jax.numpy as jnp

from jaxrens.base import MoveInfo


def finalize_cell_move(
    state,
    new_positions,
    new_cell,
    new_energy,
    count,
    overflow,
    cell_valid,
    energy_ok,
    prior_ok=None,
):
    """Accept/reject + state update shared by volume/shear/stretch.

    Gates the proposal on energy and cell shape (and, for the volume move, an
    extra volume-prior factor ``prior_ok``).  Builds ``reject_reason`` with
    priority energy > cell > prior so the most actionable reason is reported
    when several apply.

    The neighbor-bucket signals (``max_neighbor_count``, ``overflow``) are
    gated on ``cell_valid``: hard cell-shape rejections describe configurations
    the chain will never live at, so letting their counts leak into state would
    permanently inflate the neighbor bucket for proposals rejected on the spot.

    Args:
        state: Current walker state.
        new_positions: Proposed positions.
        new_cell: Proposed cell.
        new_energy: Backend energy of the proposal.
        count: Backend ``max_neighbor_count`` for the proposal.
        overflow: Backend overflow flag for the proposal.
        cell_valid: Whether the proposed cell passes the geometry guard.
        energy_ok: Whether ``new_energy < likelihood_constraint``.
        prior_ok: Optional volume-prior acceptance flag (volume move only). When
            ``None`` the move has no prior factor and no reason-3 bucket.

    Returns:
        ``(new_state, MoveInfo)``.
    """
    if prior_ok is None:
        accepted = energy_ok & cell_valid
        # Reject priority: energy > cell
        reject_reason = jnp.where(
            accepted,
            jnp.int32(0),
            jnp.where(~energy_ok, jnp.int32(1), jnp.int32(2)),
        )
    else:
        accepted = energy_ok & cell_valid & prior_ok
        # Reject priority: energy > cell > prior
        reject_reason = jnp.where(
            accepted,
            jnp.int32(0),
            jnp.where(
                ~energy_ok,
                jnp.int32(1),
                jnp.where(~cell_valid, jnp.int32(2), jnp.int32(3)),
            ),
        )

    new_state = state.set(
        positions=jnp.where(accepted, new_positions, state.positions),
        energy=jnp.where(accepted, new_energy, state.energy),
        cell=jnp.where(accepted, new_cell, state.cell),
        max_neighbor_count=jnp.maximum(
            state.max_neighbor_count,
            jnp.where(cell_valid, count, 0),
        ),
        overflow=state.overflow | (overflow & cell_valid),
    )

    info = MoveInfo(
        accepted=accepted,
        log_likelihood=-new_state.energy,
        n_evaluations=1,
        reject_reason=reject_reason,
    )

    return new_state, info
