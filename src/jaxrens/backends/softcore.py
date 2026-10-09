"""SoftCoreTerm — a fixed repulsive Morse term added to any energy model.

MLIPs (MACE, NeuralIL, Nequix, …) generally have no defined behaviour for
close-contact configurations far outside their training distribution, where
the learned potential can become attractive or NaN. During nested sampling
at high pressure or with aggressive cell moves walkers can drift into such
configurations and irreversibly collapse atoms onto each other.

This term adds a parameter-free repulsive Morse contribution::

    phi(r) = d0 * exp(-2 * a0 * (r - b0))

multiplied by a smooth cutoff function that is exactly 1 below ``r_switch``
and exactly 0 above ``r_cut`` (default 1.25 Å). The contribution is summed
over all atom pairs using minimum-image distances when a periodic cell is
present (so close contacts across a boundary are counted), and raw
Cartesian distances for non-periodic systems. MIC is single-image — no
supercell expansion — which is exact as long as ``r_cut`` stays below half
the shortest cell vector.

Usage::

    model = MACEBackend(...)
    H = Hamiltonian(model, [SoftCoreTerm(), EnsembleTerm(pressure=P)])

The term contributes its own energy and forces (autodiff of the soft-core
energy alone), so the model keeps its native force path.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from jaxrens.backends.geometry import pairwise_distances
from jaxrens.backends.hamiltonian import EnergyTerm, Hamiltonian

#: Default soft-core Morse parameters (``a0``, ``b0``, ``d0``, ``r_core_cut``,
#: ``r_core_switch``) used when a backend enables soft-core repulsion without
#: overriding them.
DEFAULT_SOFTCORE_KWARGS: dict[str, float] = {
    "a0": 1.0,
    "b0": 3.0,
    "d0": 1.0,
    "r_core_cut": 1.25,
    "r_core_switch": 0.75,
}


def _aux_f(t: jnp.ndarray) -> jnp.ndarray:
    return jnp.where(t > 0.0, jnp.exp(-1.0 / jnp.where(t > 0.0, t, 1.0)), 0.0)


def _aux_g(t: jnp.ndarray) -> jnp.ndarray:
    f_t = _aux_f(t)
    return f_t / (f_t + _aux_f(1.0 - t))


def _smooth_cutoff(
    r: jnp.ndarray,
    r_switch: float,
    r_cut: float,
) -> jnp.ndarray:
    """Smooth-bump partition-of-unity cutoff.

    Equals 1 for ``r <= r_switch``, 0 for ``r >= r_cut``, C^infty in between.
    Ported verbatim from ``neuralil.model.smooth_cutoff`` so the soft-core
    wrapper has no NeuralIL dependency.
    """
    r2 = r * r
    rs2 = r_switch * r_switch
    rc2 = r_cut * r_cut
    return 1.0 - _aux_g((r2 - rs2) / (rc2 - rs2))


def _softcore_energy(
    positions: jnp.ndarray,
    cell: jnp.ndarray,
    a0: float,
    b0: float,
    d0: float,
    r_core_switch: float,
    r_core_cut: float,
) -> jnp.ndarray:
    """Total soft-core Morse repulsion for one configuration.

    Sums ``phi(r_ij) * smooth_cutoff(r_ij)`` over all distinct pairs ``i < j``.
    Distances use the minimum-image convention when ``cell`` is a real
    (non-degenerate) lattice, so close contacts across a periodic boundary
    are counted; for a non-periodic system (``cell`` all zeros / singular)
    raw Cartesian distances are used. MIC is single-image ("111", O(N^2),
    no supercell expansion) — exact as long as ``r_core_cut`` is below half
    the shortest cell vector, which holds for the short soft-core range at
    any non-pathological cell.

    The diagonal is masked by pushing self-distances past the cutoff.

    Args:
        positions: ``(N, 3)`` atomic positions.
        cell: ``(3, 3)`` lattice matrix (rows are lattice vectors); zeros
            for a non-periodic system.
        a0, b0, d0: Fixed Morse parameters (steepness, equilibrium offset,
            prefactor).
        r_core_switch: Radius at which the cutoff starts ramping from 1.
        r_core_cut: Radius beyond which the contribution is exactly 0.

    Returns:
        Scalar total soft-core repulsion energy.
    """
    r = pairwise_distances(positions, cell)

    # Self-pairs and any spurious near-zeros: push past the cutoff so they
    # contribute exactly zero through the smooth_cutoff factor.
    diag_mask = r < 1e-10
    r_safe = jnp.where(diag_mask, 2.0 * r_core_cut, r)

    phi = d0 * jnp.exp(-2.0 * a0 * (r_safe - b0))
    cutoffs = _smooth_cutoff(r_safe, r_core_switch, r_core_cut)
    contributions = phi * cutoffs

    # 0.5 to undo the i<->j double-count of the full (N, N) sum.
    return 0.5 * jnp.sum(contributions)


def _rect_distances(
    positions_rows: jnp.ndarray,
    positions_cols: jnp.ndarray,
    frac_rows: jnp.ndarray,
    frac_cols: jnp.ndarray,
    safe_cell: jnp.ndarray,
    periodic: jnp.ndarray,
) -> jnp.ndarray:
    """``(rows, cols)`` MIC distance matrix from PRE-COMPUTED fractional
    coordinates, same formula as ``geometry.pairwise_distances``.

    Taking ``frac_rows``/``frac_cols`` as arguments (instead of computing
    ``positions @ inv(cell)`` again here) matters, not just for speed: XLA's
    matmul is not shape-invariant at the ULP level, so an independent
    ``(K, 3) @ (3, 3)`` matmul for the touched rows can round differently
    from the ``(N, 3) @ (3, 3)`` matmul used to get the full system's
    fractional coordinates — a difference amplified by the soft-core
    cutoff's steep derivative near its switch radius into an energy error
    far above floating-point noise. Reusing one shared ``frac`` array (see
    :func:`_softcore_touched_energy`) makes the touched-rows computation
    bit-identical, on the overlapping pairs, to what the full ``(N, N)``
    computation would produce.
    """
    raw_delta = positions_rows[:, None, :] - positions_cols[None, :, :]
    df = frac_rows[:, None, :] - frac_cols[None, :, :]
    df = df - jnp.round(df)
    mic_delta = df @ safe_cell

    delta = jnp.where(periodic, mic_delta, raw_delta)
    d2 = jnp.sum(delta**2, axis=-1)
    nonzero = d2 > 0.0
    return jnp.where(nonzero, jnp.sqrt(jnp.where(nonzero, d2, 1.0)), 0.0)


def _softcore_pair_terms(
    r: jnp.ndarray,
    a0: float,
    b0: float,
    d0: float,
    r_core_switch: float,
    r_core_cut: float,
    mask_self: jnp.ndarray,
) -> jnp.ndarray:
    """``phi(r) * smooth_cutoff(r)`` elementwise, with ``mask_self`` entries
    (self-pairs, or any other pair to exclude) pushed past the cutoff first.

    Shared by the full ``(N, N)`` sum and the touched-rows ``(K, N)`` /
    touched-submatrix ``(K, K)`` sums used by :func:`_softcore_touched_energy`
    — same formula, different distance matrix shape.
    """
    r_safe = jnp.where(mask_self, 2.0 * r_core_cut, r)
    phi = d0 * jnp.exp(-2.0 * a0 * (r_safe - b0))
    cutoffs = _smooth_cutoff(r_safe, r_core_switch, r_core_cut)
    return phi * cutoffs


def _softcore_touched_energy(
    positions: jnp.ndarray,
    cell: jnp.ndarray,
    touched_idx: jnp.ndarray,
    a0: float,
    b0: float,
    d0: float,
    r_core_switch: float,
    r_core_cut: float,
) -> jnp.ndarray:
    """Sum of all soft-core pairs that involve at least one atom in
    ``touched_idx`` — the only pairs that can change when just those atoms
    move — computed in O(K*N) instead of the full O(N^2) sum.

    Let ``T`` = ``touched_idx``, ``U`` = every other atom.  The row-wise sum
    ``S_row = sum_{i in T, j != i} phi(r_ij)`` counts each ``T-U`` pair once
    (only ``T``'s row scans it) but each ``T-T`` pair twice (both rows scan
    each other). The touched-submatrix sum ``S_TT`` isolates exactly that
    double-counted part, so ``S_row - 0.5*S_TT`` is the true (no double
    count, no missing pair) contribution of every pair touching ``T`` —
    algebraically identical to what ``_softcore_energy`` would attribute to
    those pairs out of its full ``0.5 * sum`` over all pairs.

    Same single minimum-image convention as ``_softcore_energy`` (no
    multi-image enumeration): exact as long as ``r_core_cut`` stays below
    half the shortest cell vector, same contract as the full-sum version.
    """
    touched_positions = positions[touched_idx]

    # Fractional coordinates computed ONCE for the whole system, then
    # sliced for the touched rows — see ``_rect_distances`` for why this
    # (rather than a second, independently-shaped matmul) is required for
    # exactness, not just reuse.
    periodic = jnp.abs(jnp.linalg.det(cell)) > 1e-10
    safe_cell = jnp.where(periodic, cell, jnp.eye(3, dtype=cell.dtype))
    frac_all = positions @ jnp.linalg.inv(safe_cell)
    frac_touched = frac_all[touched_idx]

    # (K, N): distance from each touched atom to every atom (itself included).
    r_row = _rect_distances(
        touched_positions, positions, frac_touched, frac_all, safe_cell, periodic
    )
    self_row = touched_idx[:, None] == jnp.arange(positions.shape[0])[None, :]
    s_row = jnp.sum(
        _softcore_pair_terms(
            r_row, a0, b0, d0, r_core_switch, r_core_cut, self_row
        )
    )

    # (K, K): distances among touched atoms only, to recover the double-
    # counted T-T part of s_row — same ``frac_touched`` on both sides, so
    # it agrees exactly with the diagonal block of ``r_row``.
    r_tt = _rect_distances(
        touched_positions,
        touched_positions,
        frac_touched,
        frac_touched,
        safe_cell,
        periodic,
    )
    self_tt = r_tt < 1e-10
    s_tt = jnp.sum(
        _softcore_pair_terms(
            r_tt, a0, b0, d0, r_core_switch, r_core_cut, self_tt
        )
    )

    return s_row - 0.5 * s_tt


class SoftCoreTerm(EnergyTerm):
    """Fixed repulsive Morse soft core as an additive Hamiltonian term.

    Depends only on geometry (positions, cell), not on species or trainable
    parameters, so it composes with any model (LJ, MACE, Nequix, NeuralIL,
    jaxmd, toy).  Forces come from autodiff of this term alone
    (:meth:`EnergyTerm.energy_and_forces`).
    """

    def __init__(
        self,
        a0: float = 1.0,
        b0: float = 3.0,
        d0: float = 1.0,
        r_core_cut: float = 1.25,
        r_core_switch: float = 0.75,
    ):
        self.a0 = float(a0)
        self.b0 = float(b0)
        self.d0 = float(d0)
        self.r_core_cut = float(r_core_cut)
        self.r_core_switch = float(r_core_switch)

    def energy(
        self,
        positions: jnp.ndarray,
        species: jnp.ndarray,
        cell: jnp.ndarray,
        ensemble_params: dict[str, Any] | None = None,
    ) -> jnp.ndarray:
        """Total soft-core repulsion ``E_core`` for one configuration."""
        return _softcore_energy(
            positions,
            cell,
            self.a0,
            self.b0,
            self.d0,
            self.r_core_switch,
            self.r_core_cut,
        )

    def local_energy_delta(
        self,
        touched_idx: jnp.ndarray,
        old_positions: jnp.ndarray,
        old_types: jnp.ndarray,
        new_positions: jnp.ndarray,
        new_types: jnp.ndarray,
        cell: jnp.ndarray,
        ensemble_params: dict[str, Any] | None = None,
    ) -> jnp.ndarray:
        """Exact ``E_core(new) - E_core(old)`` in O(K*N), not O(N^2).

        Independent of ``types``: the soft-core term depends only on
        geometry, so a species-only change (swap/morph) makes this exactly
        zero without special-casing — ``_softcore_touched_energy`` on
        identical ``old_positions``/``new_positions`` at ``touched_idx``
        returns the same value for both terms.
        """
        touched_idx = jnp.asarray(touched_idx)
        old_contribution = _softcore_touched_energy(
            old_positions,
            cell,
            touched_idx,
            self.a0,
            self.b0,
            self.d0,
            self.r_core_switch,
            self.r_core_cut,
        )
        new_contribution = _softcore_touched_energy(
            new_positions,
            cell,
            touched_idx,
            self.a0,
            self.b0,
            self.d0,
            self.r_core_switch,
            self.r_core_cut,
        )
        return new_contribution - old_contribution


def SoftCoreBackend(base: Any, **softcore_kwargs: float) -> Hamiltonian:
    """Convenience: ``base`` plus a :class:`SoftCoreTerm`.

    Returns a flat :class:`Hamiltonian` (``base`` may itself be one; its
    terms are kept and the soft-core term appended).
    """
    return Hamiltonian(base, [SoftCoreTerm(**softcore_kwargs)])
