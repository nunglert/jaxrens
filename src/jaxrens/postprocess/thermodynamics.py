"""Thermodynamic post-processing of nested sampling results.

NumPy-based for unconditional float64 precision.  Sampling is pinned to
float32 by the root ``__init__`` (see the comment there); post-processing
runs in a separate process, so the precision regimes don't interfere.

Energy convention: ``E = -log(L)``, so higher energy = lower likelihood.

Batch-shape support
-------------------
All public functions accept ``dead_energies`` with arbitrary leading batch
dimensions ``(*batch, max_dead)``.  Output shape mirrors the input batch
prefix (output of the trailing axis-reduction).  Examples:

* ``(max_dead,)``          → scalar output
* ``(n_runs, max_dead)``   → ``(n_runs,)``
* ``(G, P, max_dead)``     → ``(G, P)``

Temperature axis
----------------
``beta`` is a scalar or a 1-D array of ``n_T`` inverse temperatures.  A 1-D
``beta`` adds a trailing ``n_T`` axis to the output (``(*batch, n_T)``), and
:func:`thermal_weights` returns ``(*batch, n_T, n_pts)``, so a whole
temperature grid costs one pass over the points.

Points are always ordered dead first (``.energies`` row order), then the
live walkers; :func:`log_point_weights` gives their log prior-mass weights.

JAX-array inputs are auto-converted to NumPy via ``np.asarray``; outputs are
always NumPy float64 (or 0-d / array thereof).  JIT-compilation of these
functions is **not** supported; they run on the host in fp64.  Cv etc.
on a (n_dead,) array of a few thousand entries is far below the cost of
a typical sampling iteration, so JITability bought nothing in practice
while masking precision loss.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from jaxrens.prior_mass import log_dead_weights, log_live_weight

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _as_f64(x) -> np.ndarray:
    """Coerce array-like (including JAX arrays) to ``np.ndarray[float64]``."""
    return np.asarray(x, dtype=np.float64)


def calc_log_weights(
    n_dead,
    n_live,
    n_cull: int = 1,
) -> np.ndarray:
    """Compute log prior-mass weights for dead points.

    pymatnest convention (``ns_analyse.calc_log_a``); see
    :mod:`jaxrens.prior_mass` for the formulae.  Dead points are indexed
    over the per-cull sequence, ``n_cull`` per iteration in descending
    energy — exactly the row order of a ``.energies`` file.

    Args:
        n_dead: Number of dead points collected (scalar).
        n_live: Number of live walkers (scalar).
        n_cull: Number culled per iteration (default 1).

    Returns:
        ``np.ndarray[float64]`` of shape ``(n_dead,)``.
    """
    return log_dead_weights(int(np.asarray(n_dead)), int(n_live), int(n_cull))


def calc_log_weights_live(
    n_dead,
    n_live,
    n_cull: int = 1,
) -> np.ndarray:
    """Compute log weight for each remaining live walker.

    Each of the ``n_live`` live walkers carries ``X_D / n_live`` of the
    remaining prior mass after ``n_dead`` dead points (see
    :func:`jaxrens.prior_mass.log_live_weight`).

    Args:
        n_dead: Number of dead points collected (scalar).
        n_live: Number of live walkers (scalar).
        n_cull: Number culled per iteration.

    Returns:
        0-d ``np.ndarray[float64]`` (the scalar log weight per live walker).
    """
    return log_live_weight(int(np.asarray(n_dead)), int(n_live), int(n_cull))


def log_point_weights(
    n_dead,
    n_live,
    n_cull: int = 1,
    *,
    n_live_points: int | None = None,
) -> np.ndarray:
    """Log prior-mass weights of every point, dead points first.

    The dead points carry :func:`calc_log_weights`; each live walker
    carries :func:`calc_log_weights_live`.  This is the point order every
    function in this module uses: dead points in ``.energies`` row order,
    then the live walkers.

    Args:
        n_dead: Number of dead points.
        n_live: Number of live walkers of the run (``K``), which sets the
            prior-mass ladder.
        n_cull: Number culled per iteration.
        n_live_points: Number of live points to append.  Defaults to
            ``n_live``; differs only when the live set passed in is not the
            full population.

    Returns:
        ``np.ndarray[float64]`` of shape ``(n_dead + n_live_points,)``.
    """
    if n_live_points is None:
        n_live_points = int(n_live)
    log_w_dead = calc_log_weights(n_dead, n_live, n_cull)
    log_w_live = calc_log_weights_live(n_dead, n_live, n_cull)
    return np.concatenate(
        [log_w_dead, np.full(int(n_live_points), float(log_w_live))]
    )


def _merge_points(dead_E: np.ndarray, live_E: np.ndarray) -> np.ndarray:
    """Dead then live values along the last axis, live broadcast to the
    dead batch prefix."""
    live_b = np.broadcast_to(live_E, dead_E.shape[:-1] + live_E.shape[-1:])
    return np.concatenate([dead_E, live_b], axis=-1)


def _along_temperature(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Insert the temperature axis before the point axis when ``beta`` is
    1-D, so ``x`` broadcasts against ``(*batch, n_T, n_pts)``."""
    return x[..., None, :] if beta.ndim == 1 else x


def _log_boltzmann_terms(
    beta: np.ndarray, all_E: np.ndarray, log_w: np.ndarray
) -> np.ndarray:
    """Unnormalised ``log(a_i) - beta * E_i``.

    Shape ``(*batch, n_pts)`` for a scalar ``beta`` and
    ``(*batch, n_T, n_pts)`` for a 1-D ``beta`` of length ``n_T``.
    """
    if beta.ndim == 0:
        return log_w - beta * all_E
    if beta.ndim == 1:
        return log_w - beta[:, None] * all_E[..., None, :]
    raise ValueError(f"beta must be a scalar or 1-D, got shape {beta.shape}")


def thermal_weights(
    beta,
    dead_energies,
    live_energies,
    n_live,
    n_cull: int = 1,
    return_log_Z: bool = False,
):
    """Normalised thermal weights ``w_i ∝ a_i exp(-beta E_i)`` of every point.

    Points are the dead points followed by the live walkers (see
    :func:`log_point_weights`).  A thermal average of any per-point
    quantity ``O`` is then ``Σ_i w_i O_i``, so one weight matrix serves
    every observable of a run.

    Args:
        beta: Inverse temperature, scalar or 1-D of length ``n_T``.
        dead_energies: Dead point energies, shape ``(*batch, n_dead)``.
        live_energies: Live walker energies, shape ``(*batch, n_live)`` or
            ``(n_live,)``.
        n_live: Number of live walkers.
        n_cull: Number culled per iteration.
        return_log_Z: Also return the log partition function, the
            normalisation of the weights.

    Returns:
        Weights summing to one along the last axis, shape
        ``(*batch, n_dead + n_live)`` for a scalar ``beta`` and
        ``(*batch, n_T, n_dead + n_live)`` for a 1-D one.  With
        ``return_log_Z``, the tuple ``(w, log_Z)``, ``log_Z`` of shape
        ``w.shape[:-1]``.
    """
    beta = _as_f64(beta)
    dead_E = _as_f64(dead_energies)
    all_E = _merge_points(dead_E, _as_f64(live_energies))
    n_dead = dead_E.shape[-1]
    n_pts = all_E.shape[-1]
    log_w = log_point_weights(
        n_dead, n_live, n_cull, n_live_points=n_pts - n_dead
    )
    log_t = _log_boltzmann_terms(beta, all_E, log_w)
    log_Z = logsumexp(log_t, axis=-1, keepdims=True)
    w = np.exp(log_t - log_Z)
    return (w, log_Z[..., 0]) if return_log_Z else w


def log_evidence(
    dead_energies,
    live_energies,
    n_live,
    n_cull: int = 1,
) -> np.ndarray:
    """Total log evidence ``log Z`` from dead and live points.

    Supports arbitrary leading batch dimensions on ``dead_energies`` and
    ``live_energies``.  Output batch shape mirrors the input (minus the
    trailing ``max_dead`` / ``n_live`` axes).

    Args:
        dead_energies: Dead point energies, shape ``(*batch, n_dead)``.
        live_energies: Live walker energies, shape ``(*batch, n_live)`` or
            ``(n_live,)`` (broadcast over batch).
        n_live: Number of live walkers (scalar).
        n_cull: Number culled per iteration.

    Returns:
        ``log Z`` with shape ``batch`` — scalar for 1-D input.
    """
    return partition_function(
        1.0, dead_energies, live_energies, n_live, n_cull
    )


def partition_function(
    beta,
    dead_energies,
    live_energies,
    n_live,
    n_cull: int = 1,
    dead_volumes=None,
    live_volumes=None,
) -> np.ndarray:
    """Log partition function ``log Z(beta)`` at inverse temperature ``beta``.

    ``Z(beta) = Σ_i w_i exp(-beta · E_i)`` over the merged dead+live ensemble.

    Args:
        beta: Inverse temperature ``1/(k_B T)``, scalar or 1-D.
        dead_energies: Dead point energies, shape ``(*batch, n_dead)``.
        live_energies: Live walker energies, shape ``(*batch, n_live)`` or
            ``(n_live,)``.
        n_live: Number of live walkers.
        n_cull: Number culled per iteration.
        dead_volumes: Optional ``P V`` terms added to ``dead_energies``,
            for energies that do not include them yet.  Shape must
            broadcast with ``dead_energies``.  Energies streamed by an NPT
            run are already ``H = U + P V``: pass nothing then.
        live_volumes: Optional ``P V`` terms added to ``live_energies``,
            same rules as ``dead_volumes``.

    Returns:
        ``log Z(beta)`` with shape ``batch`` (scalar for 1-D input), plus a
        trailing ``n_T`` axis for a 1-D ``beta``.
    """
    beta = _as_f64(beta)
    dead_E = _as_f64(dead_energies)
    live_E = _as_f64(live_energies)
    if dead_volumes is not None:
        dead_E = dead_E + _as_f64(dead_volumes)
    if live_volumes is not None:
        live_E = live_E + _as_f64(live_volumes)

    all_E = _merge_points(dead_E, live_E)
    n_dead = dead_E.shape[-1]
    log_w = log_point_weights(
        n_dead, n_live, n_cull, n_live_points=all_E.shape[-1] - n_dead
    )
    return logsumexp(_log_boltzmann_terms(beta, all_E, log_w), axis=-1)


def heat_capacity(
    beta,
    dead_energies,
    live_energies,
    n_live,
    n_cull: int = 1,
) -> np.ndarray:
    """Heat capacity ``C_v(beta) = beta² · Var(E)`` at inverse temperature beta.

    Uses the **deviation form** ``Σ w_i (E_i - <E>)²`` instead of the algebraic
    identity ``<E²> - <E>²``: the latter exhibits catastrophic cancellation at
    low T (distribution collapses onto a single walker, ``<E²>`` and ``<E>²``
    both approach ``E_min²``, and their difference is dominated by float noise).
    Even in fp64 the deviation form is the safer default; in fp32 the algebraic
    form gives visible negative-Cv spikes.

    Args:
        beta: Inverse temperature, scalar or 1-D.
        dead_energies: Dead point energies, shape ``(*batch, n_dead)``.
        live_energies: Live walker energies, shape ``(*batch, n_live)`` or
            ``(n_live,)``.
        n_live: Number of live walkers.
        n_cull: Number culled per iteration.

    Returns:
        ``C_v`` with shape ``batch`` (scalar for 1-D input), plus a trailing
        ``n_T`` axis for a 1-D ``beta``.
    """
    beta = _as_f64(beta)
    w = thermal_weights(beta, dead_energies, live_energies, n_live, n_cull)
    all_E = _along_temperature(
        _merge_points(_as_f64(dead_energies), _as_f64(live_energies)), beta
    )
    mean_E = np.sum(w * all_E, axis=-1, keepdims=True)
    var_E = np.sum(w * (all_E - mean_E) ** 2, axis=-1)
    return beta**2 * var_E


def expectation(
    observable_values,
    beta,
    dead_energies,
    live_energies,
    n_live,
    n_cull: int = 1,
) -> np.ndarray:
    """Thermal expectation ``<O>(beta)`` for an observable on the dead+live set.

    Args:
        observable_values: Observable values, shape
            ``(*batch, n_dead + n_live)`` or ``(n_dead + n_live,)``
            (broadcast over batch).
        beta: Inverse temperature, scalar or 1-D.
        dead_energies: Dead point energies, shape ``(*batch, n_dead)``.
        live_energies: Live walker energies, shape ``(*batch, n_live)`` or
            ``(n_live,)``.
        n_live: Number of live walkers.
        n_cull: Number culled per iteration.

    Returns:
        ``<O>`` with shape ``batch`` (scalar for 1-D input), plus a trailing
        ``n_T`` axis for a 1-D ``beta``.
    """
    beta = _as_f64(beta)
    w = thermal_weights(beta, dead_energies, live_energies, n_live, n_cull)
    # Points shape (*batch, n_pts): the weights without their n_T axis.
    pts_shape = w.shape[:-2] + w.shape[-1:] if beta.ndim == 1 else w.shape
    obs = np.broadcast_to(_as_f64(observable_values), pts_shape)
    return np.sum(w * _along_temperature(obs, beta), axis=-1)


def free_energy(beta, log_Z_beta) -> np.ndarray:
    """Helmholtz free energy ``F = -log Z(beta) / beta``.

    Args:
        beta: Inverse temperature (scalar or broadcastable).
        log_Z_beta: Log partition function at ``beta``, any shape.

    Returns:
        ``F`` with the same shape as ``log_Z_beta``.
    """
    beta = _as_f64(beta)
    log_Z_beta = _as_f64(log_Z_beta)
    return -log_Z_beta / beta


# ---------------------------------------------------------------------------
# Estimators on a weight matrix
# ---------------------------------------------------------------------------
#
# ``w`` is any output of :func:`thermal_weights`: shape ``(*lead, n_pts)``,
# where ``lead`` holds batch and temperature axes.  Observables are values
# per point, shape ``(n_pts,)`` for a scalar or ``(n_pts, k)`` for a vector
# observable (one column per component, e.g. per gallery or per RDF bin).
# Results have shape ``lead`` for a scalar observable and ``(*lead, k)`` for
# a vector one.  Computing ``w`` once and reusing it for every observable
# is what makes many observables over a temperature grid cheap.


def _check_points(w: np.ndarray, x: np.ndarray, name: str) -> None:
    if x.ndim not in (1, 2) or x.shape[0] != w.shape[-1]:
        raise ValueError(
            f"{name} must have shape (n_pts,) or (n_pts, k) with "
            f"n_pts={w.shape[-1]} (the last axis of w), got {x.shape}"
        )


def weighted_mean(w, obs) -> np.ndarray:
    """Thermal average ``<O> = Σ_i w_i O_i``.

    Args:
        w: Normalised weights, shape ``(*lead, n_pts)``.
        obs: Per-point values, shape ``(n_pts,)`` or ``(n_pts, k)``.

    Returns:
        Shape ``lead``, or ``(*lead, k)`` for a vector observable.
    """
    w = _as_f64(w)
    obs = _as_f64(obs)
    _check_points(w, obs, "obs")
    return w @ obs


def weighted_covariance(w, a, b) -> np.ndarray:
    """Thermal covariance ``Cov(A, B) = Σ_i w_i (A_i - <A>) (B_i - <B>)``.

    Uses the deviation form, like :func:`heat_capacity`: the algebraic
    form ``<AB> - <A><B>`` loses all precision at low temperature, where
    the weights collapse onto a few points and both terms approach the
    same number.

    ``a`` and ``b`` broadcast against each other per point, so a scalar
    ``a`` of shape ``(n_pts,)`` with a vector ``b`` of shape ``(n_pts, k)``
    gives the ``k`` covariances ``Cov(A, B_j)``.  Two vector observables
    pair up component by component (no full ``k x k`` matrix).

    Args:
        w: Normalised weights, shape ``(*lead, n_pts)``.
        a: Per-point values, shape ``(n_pts,)`` or ``(n_pts, k)``.
        b: Per-point values, shape ``(n_pts,)`` or ``(n_pts, k)``.

    Returns:
        Shape ``lead``, or ``(*lead, k)`` if either input is a vector.
    """
    w = _as_f64(w)
    a = _as_f64(a)
    b = _as_f64(b)
    _check_points(w, a, "a")
    _check_points(w, b, "b")
    mean_a = w @ a
    mean_b = w @ b
    if a.ndim == 1 and b.ndim == 1:
        # The deviations have the size of w itself: vectorise.
        dev = (a - mean_a[..., None]) * (b - mean_b[..., None])
        return np.sum(w * dev, axis=-1)
    # Vector observables: one row of w at a time, so the deviations stay
    # (n_pts, k) instead of (*lead, n_pts, k).
    a2 = a if a.ndim == 2 else a[:, None]
    b2 = b if b.ndim == 2 else b[:, None]
    w_rows = w.reshape(-1, w.shape[-1])
    ma = np.reshape(mean_a, (w_rows.shape[0], -1))
    mb = np.reshape(mean_b, (w_rows.shape[0], -1))
    out = np.stack(
        [
            w_rows[r] @ ((a2 - ma[r]) * (b2 - mb[r]))
            for r in range(w_rows.shape[0])
        ]
    )
    return out.reshape(w.shape[:-1] + out.shape[-1:])


def weighted_variance(w, obs) -> np.ndarray:
    """Thermal variance ``Var(O) = Cov(O, O)``, in the deviation form.

    Args:
        w: Normalised weights, shape ``(*lead, n_pts)``.
        obs: Per-point values, shape ``(n_pts,)`` or ``(n_pts, k)``.

    Returns:
        Shape ``lead``, or ``(*lead, k)`` for a vector observable.
    """
    return weighted_covariance(w, obs, obs)


def class_probability(w, mask) -> np.ndarray:
    """Thermal probability ``<1_C>`` of the points in a class ``C``.

    Args:
        w: Normalised weights, shape ``(*lead, n_pts)``.
        mask: Class membership per point, boolean, shape ``(n_pts,)`` or
            ``(n_pts, k)`` for ``k`` classes at once.

    Returns:
        Shape ``lead``, or ``(*lead, k)``.
    """
    w = _as_f64(w)
    mask = np.asarray(mask, dtype=bool)
    _check_points(w, mask, "mask")
    return w @ mask.astype(np.float64)


def conditional_mean(w, obs, mask=None) -> np.ndarray:
    """Thermal average restricted to a class, ``<O 1_C> / <1_C>``.

    Points where ``obs`` is NaN are left out as well, so an observable that
    is undefined on some points (e.g. the width of the empty gallery in a
    configuration with no empty gallery) averages over the points where it
    is defined.

    Args:
        w: Normalised weights, shape ``(*lead, n_pts)``.
        obs: Per-point values, shape ``(n_pts,)`` or ``(n_pts, k)``; NaN
            marks an undefined value.
        mask: Class membership per point, boolean, broadcasting against
            ``obs``.  ``None`` keeps every point (only NaNs are dropped).

    Returns:
        Shape ``lead``, or ``(*lead, k)``.  NaN where the class (after
        dropping NaNs) has zero weight.
    """
    w = _as_f64(w)
    obs = _as_f64(obs)
    _check_points(w, obs, "obs")
    keep = np.isfinite(obs)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        _check_points(w, mask, "mask")
        # Pair a scalar with a vector per point: (n_pts,) -> (n_pts, 1).
        if mask.ndim == 2 and obs.ndim == 1:
            obs, keep = obs[:, None], keep[:, None]
        elif mask.ndim == 1 and obs.ndim == 2:
            mask = mask[:, None]
        keep = keep & mask
    num = w @ np.where(keep, obs, 0.0)
    den = w @ keep.astype(np.float64)
    return np.divide(
        num, den, out=np.full(np.shape(num), np.nan), where=den > 0.0
    )
