"""NS callback implementations for monitoring, checkpointing, and I/O.

These are plugged into the NS outer loop via the callbacks parameter.
Callbacks receive NSState objects (not dicts).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from jaxrens.state.ns import NSState
from jaxrens.utils.cell import get_volume

logger = logging.getLogger(__name__)


def _ns_state_to_checkpoint_dict(ns_state: NSState) -> dict:
    """Convert NSState to a dict suitable for save_checkpoint.

    Live-walker state + scalars only.  Dead-point history is *not* in the
    checkpoint — the canonical record lives in ``<prefix>.energies`` (text
    file appended each iteration by ``EnergyLogger``) and
    ``<prefix>.traj.<format>`` (per-iter dead-point trajectory written by
    ``TrajectoryCallback`` / ``BatchedTrajectoryCallback``).  Postprocessing
    (``postprocess.Monitor.from_directory``) reads from those streamed
    artifacts; the HDF5 checkpoint is the restart anchor for live state.

    Handles single-run scalar shapes (``iteration`` / ``n_dead`` as Python
    ints) and batched multi-run shapes (``(G, P)`` or ``(n_runs,)``) by
    keeping batched scalars as arrays; ``save_checkpoint`` is batched-safe
    since WORKLOG 2026-04-18 Task A.
    """
    pop = ns_state.population
    ep = ns_state.population.ensemble_params if hasattr(ns_state.population, "ensemble_params") else {}
    is_npt = isinstance(ep, dict) and "pressure" in ep

    it_arr = jnp.asarray(ns_state.iteration)
    iteration = int(it_arr) if it_arr.ndim == 0 else it_arr
    # n_dead ≡ iteration (both increment per ns_step) — kept as a separate
    # field for backward compat with consumers that read it.
    n_dead = iteration

    result = {
        "positions": pop.positions,
        "types": pop.types,
        "energies": pop.energy,
        "cells": pop.cell,
        "log_evidence": ns_state.log_evidence,
        "iteration": iteration,
        "n_dead": n_dead,
        "n_walkers": ns_state.n_walkers,
    }
    if is_npt:
        # Vectorized volume: vmap over all leading axes by flattening then reshape.
        cell = pop.cell
        flat = cell.reshape(-1, 3, 3)
        vols = jax.vmap(get_volume)(flat)
        result["live_volumes"] = vols.reshape(cell.shape[:-2])
    else:
        result["live_volumes"] = None
    return result


def _format_reject_breakdown(
    counts: Any,
    reasons_used: "frozenset[str] | None" = None,
) -> str:
    """Format reject-reason counts: [n_accepted, n_energy, n_cell, n_prior].

    Args:
        counts: Array-like of length 4: [n_accepted, n_energy, n_cell, n_prior].
        reasons_used: When provided, only include columns for reasons in this
            set.  Valid values: ``"energy"``, ``"cell"``, ``"prior"``.
            When ``None`` (backward compat), fall back to current behaviour:
            print all three columns if there were any rejects.

    Returns:
        Empty string when there were no rejects (all accepted) or no moves.
        Otherwise a compact ``"   reject: E=x% C=y% P=z%"`` suffix with
        columns filtered to ``reasons_used``.  If there were rejects but none
        fell in the declared ``reasons_used`` set (indicates a code bug),
        returns ``"   reject: ???=<n>"`` as a visible signal.
    """
    # Bucket index map: name -> column index in counts
    _BUCKET_IDX = {"energy": 1, "cell": 2, "prior": 3}
    _BUCKET_LABEL = {"energy": "E", "cell": "C", "prior": "P"}
    _ALL_REASONS = ("energy", "cell", "prior")

    c = np.asarray(counts).astype(np.int64)
    if c.sum() == 0:
        return ""
    n_total = int(c.sum())
    n_acc = int(c[0])
    if n_acc == n_total:
        return ""
    n_rej = n_total - n_acc
    if n_rej == 0:
        return ""

    if reasons_used is None:
        # Backward-compat: print all three columns
        pct_e = 100 * int(c[1]) / n_rej
        pct_c = 100 * int(c[2]) / n_rej
        pct_p = 100 * int(c[3]) / n_rej
        return f"   reject: E={pct_e:>3.0f}% C={pct_c:>3.0f}% P={pct_p:>3.0f}%"

    # Filter to declared reasons; build ordered parts list
    parts = []
    for reason in _ALL_REASONS:
        if reason in reasons_used:
            idx = _BUCKET_IDX[reason]
            label = _BUCKET_LABEL[reason]
            pct = 100 * int(c[idx]) / n_rej
            parts.append(f"{label}={pct:>3.0f}%")

    if not parts:
        # reasons_used is non-empty but none match our known reasons —
        # or reasons_used is an empty frozenset; shouldn't happen normally.
        return f"   reject: ???={n_rej}"

    # Check if the declared reasons account for all rejects; if not, something
    # emitted an undeclared reason — flag it without hiding the data.
    n_accounted = sum(int(c[_BUCKET_IDX[r]]) for r in reasons_used if r in _BUCKET_IDX)
    if n_accounted < n_rej:
        parts.append(f"???={n_rej - n_accounted}")

    return "   reject: " + " ".join(parts)


def _is_batched(ns_state: Any, info: "dict | None" = None) -> bool:
    """Return True when ns_state holds multiple parallel runs.

    Prefers ``info["_batcher"].is_batched`` when the *info* dict is provided
    and the key is present (commit 4+).  Falls back to the legacy ndim-sniff
    on ``ns_state.log_evidence`` for backward compatibility with callers that
    do not pass an info dict.

    Args:
        ns_state: ``NSState`` or result dict.
        info: Optional info dict from the NS outer loop.  When present and
            contains ``"_batcher"``, the batcher's ``is_batched`` property
            is authoritative.

    Returns:
        ``True`` for multi-run (batched) NS runs.
    """
    if info is not None:
        batcher = info.get("_batcher")
        if batcher is not None:
            return batcher.is_batched

    # Fallback: recover the batcher from log_evidence's shape prefix.
    from jaxrens.sampling.batch_descriptor import from_shape_prefix

    if isinstance(ns_state, NSState):
        log_ev = jnp.asarray(ns_state.log_evidence)
    else:
        log_ev = jnp.asarray(ns_state.get("log_evidence", jnp.array(0.0)))
    return from_shape_prefix(log_ev.shape).is_batched


class ProgressCallback:
    """Prints a multi-line per-move adaptation summary at each info interval.

    Canonical single location for the periodic human-readable NS summary.
    The duplicate ``logger.info`` call in ``nested_sampling.py`` around line 543
    has been removed; this callback is the only summary log source.

    Single-run format::

        iter=500  Emax=-102.3  log_Z=99.0  dt=0.8s
          galilean    ss=5.000     acc=0.62
          volume      ss=0.011     acc=0.10

    Batched format (n_runs > 1)::

        iter=500  Emax=[-102.3..-101.8]  log_Z=[99.0..99.3]  dt=0.8s
          galilean    ss=5.000±0.08     acc=0.62±0.03
          volume      ss=0.011±0.001    acc=0.10±0.02

    When ``info`` does not carry ``step_sizes_per_move`` (no full_auto), the
    per-move block is omitted and only the header line is printed.
    """

    def __init__(self, info_interval: int = 100):
        self.info_interval = info_interval
        self._start_time = time.time()
        self._last_print_time = self._start_time

    def on_iteration(self, iteration: int, ns_state: Any, info: dict) -> None:
        if iteration % self.info_interval != 0 and iteration != 0:
            return

        dt = time.time() - self._last_print_time
        self._last_print_time = time.time()

        batched = _is_batched(ns_state, info)

        # ---- header line ------------------------------------------------
        # Evaluation counter suffix (backward-compat: absent for callers that
        # don't populate cumulative_n_evaluations_per_move)
        cum_evals_arr = info.get("cumulative_n_evaluations_per_move")
        cum_grad_arr = info.get("cumulative_n_grad_evaluations_per_move")
        if cum_evals_arr is not None and cum_grad_arr is not None:
            cum_e = float(np.asarray(cum_evals_arr).sum())
            cum_g = float(np.asarray(cum_grad_arr).sum())
            eval_suffix = f"  nE={cum_e:.2e}  nG={cum_g:.2e}"
        else:
            eval_suffix = ""

        if batched:
            emax_arr = jnp.asarray(info.get("emax", 0))
            log_z_arr = jnp.asarray(
                ns_state.log_evidence if isinstance(ns_state, NSState)
                else info.get("log_evidence", float("-inf"))
            )
            header = (
                f"iter={iteration}"
                f"  Emax=[{float(jnp.min(emax_arr)):.4g}..{float(jnp.max(emax_arr)):.4g}]"
                f"  log_Z=[{float(jnp.min(log_z_arr)):.4f}..{float(jnp.max(log_z_arr)):.4f}]"
                f"  dt={dt:.1f}s"
                f"{eval_suffix}"
            )
        else:
            log_z = float(
                ns_state.log_evidence if isinstance(ns_state, NSState)
                else info.get("log_evidence", float("-inf"))
            )
            header = (
                f"iter={iteration}"
                f"  Emax={float(info.get('emax', 0)):.6g}"
                f"  log_Z={log_z:.4f}"
                f"  dt={dt:.1f}s"
                f"{eval_suffix}"
            )

        # ---- per-move rows -----------------------------------------------
        ss_per_move = info.get("step_sizes_per_move")    # (n_moves,) or (n_runs, n_moves)
        move_names = info.get("move_names")
        # Tuple of frozensets: which reject-reason buckets each move can emit.
        # None when not provided (e.g. legacy callers or run_ns_parallel).
        move_reject_reasons = info.get("move_reject_reasons")  # tuple[frozenset] | None

        # Chain-level acceptance: always present from ns_step (preferred).
        # Fall back to trial-phase acceptance_rates_per_move for backward compat.
        n_accepted_per_move = info.get("n_accepted_per_move")   # (n_moves,) int32
        n_proposed_per_move = info.get("n_proposed_per_move")   # (n_moves,) int32
        # reject_reason_counts_per_move[:, 0] = accepted,
        #   [:, 1] = energy reject, [:, 2] = cell reject, [:, 3] = prior reject.
        rr_counts_per_move = info.get("reject_reason_counts_per_move")  # (n_moves, 4) or None

        lines = [header]
        if ss_per_move is not None:
            ss = jnp.asarray(ss_per_move)
            n_moves = ss.shape[-1] if ss.ndim >= 1 else 1
            if move_names is None:
                move_names = [f"move_{i}" for i in range(n_moves)]

            # Build per-move acceptance rates from chain-level counts when available.
            if n_accepted_per_move is not None and n_proposed_per_move is not None:
                n_acc = jnp.asarray(n_accepted_per_move)
                n_prop = jnp.asarray(n_proposed_per_move)
                acc_per_move = n_acc / jnp.maximum(n_prop, 1)
                rc = jnp.asarray(rr_counts_per_move) if rr_counts_per_move is not None else None
            else:
                # Backward compat: trial-phase rates from adjust_step_size.
                trial_acc = info.get("acceptance_rates_per_move")
                if trial_acc is None:
                    acc_per_move = None
                    rc = None
                else:
                    acc_per_move = trial_acc
                    reject_counts = info.get("reject_counts_per_move")
                    rc = jnp.asarray(reject_counts) if reject_counts is not None else None

            if acc_per_move is not None:
                acc = jnp.asarray(acc_per_move)

                # Flatten any leading batch dims (ndim >= 2) into one run axis so
                # the batched-stats branch sees shape (n_runs_flat, n_moves).
                # This handles both VmapRuns (ndim==2) and PmapVmapRuns (ndim==3,
                # shape (G,P,n_moves)).
                if batched and ss.ndim >= 2:
                    # Flatten leading shape-prefix dims to a single (n_runs,) axis.
                    # Prefer the batcher when present; legacy callers without
                    # info["_batcher"] fall back to a plain reshape.
                    batcher = info.get("_batcher") if info is not None else None
                    if batcher is not None:
                        ss_flat = batcher.flatten(ss)
                        acc_flat = batcher.flatten(acc)
                        rc_flat = batcher.flatten(jnp.asarray(rc)) if rc is not None else None
                    else:
                        n_moves_here = ss.shape[-1]
                        ss_flat = ss.reshape(-1, n_moves_here)
                        acc_flat = acc.reshape(-1, n_moves_here)
                        rc_flat = (
                            jnp.asarray(rc).reshape(-1, n_moves_here, 4)
                            if rc is not None else None
                        )
                    ss_mean = jnp.mean(ss_flat, axis=0)
                    ss_std = jnp.std(ss_flat, axis=0)
                    acc_mean = jnp.mean(acc_flat, axis=0)
                    acc_std = jnp.std(acc_flat, axis=0)
                    rc_sum = jnp.sum(rc_flat, axis=0) if rc_flat is not None else None
                    for k, name in enumerate(move_names):
                        row = (
                            f"  {name:<16}  ss={float(ss_mean[k]):>9.3e}±{float(ss_std[k]):>8.2e}"
                            f"  acc={float(acc_mean[k]):>4.2f}±{float(acc_std[k]):>4.2f}"
                        )
                        if rc_sum is not None:
                            reasons_k = move_reject_reasons[k] if move_reject_reasons is not None else None
                            row += _format_reject_breakdown(rc_sum[k], reasons_used=reasons_k)
                        lines.append(row)
                else:
                    # single-run: ss shape (n_moves,)
                    if ss.ndim == 2:
                        ss = ss[0]
                        acc = acc[0] if acc.ndim == 2 else acc
                    for k, name in enumerate(move_names):
                        row = (
                            f"  {name:<16}  ss={float(ss[k]):>9.3e}"
                            f"  acc={float(acc[k]):>4.2f}"
                        )
                        if rc is not None:
                            reasons_k = move_reject_reasons[k] if move_reject_reasons is not None else None
                            if rc.ndim == 3:
                                row += _format_reject_breakdown(rc[0, k], reasons_used=reasons_k)
                            else:
                                row += _format_reject_breakdown(rc[k], reasons_used=reasons_k)
                        lines.append(row)

        # ---- inter-RE stats row (when present) ----
        re_stats = info.get("inter_re_stats")
        if re_stats is not None:
            n_att = re_stats.get("n_swap_pairs_attempted", 0)
            n_acc = re_stats.get("n_swap_pairs_accepted", 0)
            acc = n_acc / max(n_att, 1)
            re_evals = re_stats.get("n_energy_evals", 0)
            lines.append(
                f"  {'inter_re':<16}  n_pairs={n_att:>3d}  acc={acc:.2f}  evals={re_evals}"
            )

        logger.info("\n".join(lines))

    def on_finish(self, ns_state: Any) -> None:
        elapsed = time.time() - self._start_time
        if isinstance(ns_state, NSState):
            iteration_arr = jnp.asarray(ns_state.iteration)
            log_z_arr = jnp.asarray(ns_state.log_evidence)
        else:
            log_z_arr = jnp.asarray(ns_state.get("log_evidence", float("-inf")))
            iteration_arr = jnp.asarray(ns_state.get("iteration", 0))

        # Iterations stay in lock-step across replicas (outer Python loop
        # advances all runs by 1 per tick), so a single int suffices.
        iter_n = int(jnp.max(iteration_arr))
        if log_z_arr.ndim > 0:
            logger.info(
                "NS finished: %d iterations, log_Z=[%.4f..%.4f], elapsed=%.1fs",
                iter_n,
                float(jnp.min(log_z_arr)),
                float(jnp.max(log_z_arr)),
                elapsed,
            )
        else:
            logger.info(
                "NS finished: %d iterations, log_Z=%.4f, elapsed=%.1fs",
                iter_n,
                float(log_z_arr),
                elapsed,
            )


class AdaptationCallback:
    """Writes per-move adaptation data to an HDF5 trace file.

    Wraps an :class:`~jaxrens.io.adaptation_log.AdaptationLogger` and calls
    ``write_entry`` on every iteration where ``info["step_sizes_per_move"]``
    is present.  Iterations between adaptation adjustments are skipped (the
    info key is absent).

    Args:
        logger_obj: A ready-to-use ``AdaptationLogger`` instance.
    """

    def __init__(self, logger_obj: Any) -> None:
        self._adaptation_logger = logger_obj

    def on_start(self, ns_state: Any, start_info: dict | None = None) -> None:
        """Write the mandatory iter-0 baseline row.

        This pins the initial step sizes for post-hoc reconstruction:
        active ss at iter k = last row whose ``iter <= k``.  Without
        this row, a non-``full_auto`` run would produce an empty file
        (no adjustment events ever fire), and a ``full_auto`` run
        would mis-attribute the pre-first-adjust period to the first
        adjustment event's step size.

        Adjustment-event-only fields (``n_rounds``, ``cap_hits``,
        ``floor_hits``, ``converged``, ``bracket_detected``,
        ``reject_reason_counts``) are written as zeros / ``True`` to
        signal "no bisection ran for this row".  Evaluation counters
        are zero too.

        Acceptance rates default to 0 here: real chain rates are
        absent at iter 0 (no NS step has run yet), and the trial-phase
        rates from bisection don't exist either.  Downstream readers
        should ignore acc rates on the iter-0 row.
        """
        if start_info is None:
            return
        ss = start_info.get("step_sizes_per_move")
        if ss is None:
            return
        ss_arr = jnp.asarray(ss)
        batcher = start_info.get("_batcher")
        if batcher is not None:
            ss_flat = batcher.flatten(ss_arr)
        else:
            ss_flat = ss_arr[None, :] if ss_arr.ndim == 1 else ss_arr

        n_runs = self._adaptation_logger.n_runs
        n_moves = self._adaptation_logger.n_moves
        ss_np = np.asarray(ss_flat, dtype=np.float32)
        acc_np = np.zeros((n_runs, n_moves), dtype=np.float32)

        zero_int = np.zeros((n_runs, n_moves), dtype=np.int32)
        zero_int64 = np.zeros((n_runs, n_moves), dtype=np.int64)
        baseline_adj_stats = {
            "n_rounds": zero_int,
            # converged=True signals "no bisection needed / not run".
            "converged": np.ones((n_runs, n_moves), dtype=bool),
            "cap_hits": zero_int,
            "floor_hits": zero_int,
            "bracket_detected": np.zeros((n_runs, n_moves), dtype=bool),
            "reject_reason_counts": np.zeros((n_runs, n_moves, 4), dtype=np.int32),
        }
        self._adaptation_logger.write_entry(
            iteration=0,
            step_sizes=ss_np,
            acceptance_rates=acc_np,
            adjustment_stats=baseline_adj_stats,
            n_evaluations=zero_int64,
            n_grad_evaluations=zero_int64,
        )

    def on_iteration(self, iteration: int, ns_state: Any, info: dict) -> None:
        ss = info.get("step_sizes_per_move")
        acc = info.get("acceptance_rates_per_move")
        if ss is None or acc is None:
            return

        ss_np = jnp.asarray(ss)
        acc_np = jnp.asarray(acc)

        # Ensure (n_runs, n_moves) shape via the batcher when available;
        # legacy callers (no info["_batcher"]) fall back to ndim-based reshape.
        batcher = info.get("_batcher")
        if batcher is not None:
            ss_np = batcher.flatten(ss_np)
            acc_np = batcher.flatten(acc_np)
        else:
            if ss_np.ndim == 1:
                ss_np = ss_np[None, :]
                acc_np = acc_np[None, :]
            elif ss_np.ndim >= 3:
                n_moves_here = ss_np.shape[-1]
                ss_np = ss_np.reshape(-1, n_moves_here)
                acc_np = acc_np.reshape(-1, n_moves_here)

        # Collect per-adjust-call diagnostic stats when present (v2 schema).
        # All values arrive shaped ``(*shape_prefix, n_moves[, 4])`` straight
        # from ``AdaptationManager.apply`` — for PmapVmapRuns(G, P) that's
        # ``(G, P, n_moves[, 4])``.  Coerce to the flat ``(n_runs, ...)``
        # layout the HDF5 schema expects using the same ``batcher.flatten``
        # the ss/acc path uses above.  Without this the dataset is created
        # at the correct shape on the iter-0 baseline row (which is already
        # flat) but real adjust events would fail with a broadcast error.
        def _flatten_for_hdf5(arr):
            arr = np.asarray(arr)
            if batcher is not None:
                return np.asarray(batcher.flatten(arr))
            # No batcher available: fall back to ndim-based reshape.
            if arr.ndim == 1:
                return arr[None, :]
            if arr.ndim >= 3:
                # Last 1 or 2 axes are payload (n_moves[, 4]); rest is prefix.
                # Heuristic: reject_reason_counts has trailing 4, others trail at n_moves.
                tail = arr.shape[-2:] if arr.shape[-1] == 4 else arr.shape[-1:]
                return arr.reshape((-1,) + tail)
            return arr

        adjustment_stats: "dict[str, np.ndarray] | None" = None
        _adj_keys = (
            "adjustment_n_rounds",
            "adjustment_converged",
            "adjustment_cap_hits",
            "adjustment_floor_hits",
            "adjustment_bracket_detected",
            "reject_counts_per_move",
        )
        _adj_rename = {
            "adjustment_n_rounds": "n_rounds",
            "adjustment_converged": "converged",
            "adjustment_cap_hits": "cap_hits",
            "adjustment_floor_hits": "floor_hits",
            "adjustment_bracket_detected": "bracket_detected",
            "reject_counts_per_move": "reject_reason_counts",
        }
        for info_key in _adj_keys:
            val = info.get(info_key)
            if val is not None:
                if adjustment_stats is None:
                    adjustment_stats = {}
                adjustment_stats[_adj_rename[info_key]] = _flatten_for_hdf5(val)

        # Collect per-iter evaluation counts for v3 schema (shape (n_runs, n_moves))
        n_evals_raw = info.get("n_evaluations_per_move")
        n_grad_evals_raw = info.get("n_grad_evaluations_per_move")
        n_evals_np: "np.ndarray | None" = None
        n_grad_evals_np: "np.ndarray | None" = None
        if n_evals_raw is not None:
            n_evals_np = np.asarray(
                _flatten_for_hdf5(n_evals_raw), dtype=np.int64,
            )
        if n_grad_evals_raw is not None:
            n_grad_evals_np = np.asarray(
                _flatten_for_hdf5(n_grad_evals_raw), dtype=np.int64,
            )

        self._adaptation_logger.write_entry(
            iteration=iteration,
            step_sizes=np.asarray(ss_np),
            acceptance_rates=np.asarray(acc_np),
            adjustment_stats=adjustment_stats,
            n_evaluations=n_evals_np,
            n_grad_evaluations=n_grad_evals_np,
        )

    def on_finish(self, ns_state: Any) -> None:
        self._adaptation_logger.close()


class AccRatesCallback:
    """Writes per-iteration chain-phase per-move acceptance counts to HDF5.

    Independent of ``full_auto``: the chain counters ``n_accepted_per_move``
    and ``n_proposed_per_move`` are populated by ``ns_step`` on every
    iteration, so this callback fires regardless of whether bisection
    adaptation is active.  ``AdaptationCallback`` complements this by
    capturing the *sparse* step-size event series; this callback captures
    the *dense* per-iter chain-rate series.

    Stores raw counts (not derived rates) so downstream code can
    re-aggregate over arbitrary windows.

    Args:
        logger_obj: A ready-to-use ``AccRatesLogger`` instance.
        interval: Fire every ``interval`` iterations.  Default ``1``
            (every iter); set higher to reduce I/O on long runs.
    """

    def __init__(self, logger_obj: Any, interval: int = 1) -> None:
        self._logger = logger_obj
        self._interval = max(1, int(interval))

    def on_iteration(self, iteration: int, ns_state: Any, info: dict) -> None:
        if iteration % self._interval != 0:
            return
        n_acc = info.get("n_accepted_per_move")
        n_prop = info.get("n_proposed_per_move")
        if n_acc is None or n_prop is None:
            return

        n_acc_arr = jnp.asarray(n_acc)
        n_prop_arr = jnp.asarray(n_prop)

        batcher = info.get("_batcher")
        if batcher is not None:
            n_acc_arr = batcher.flatten(n_acc_arr)
            n_prop_arr = batcher.flatten(n_prop_arr)
        else:
            if n_acc_arr.ndim == 1:
                n_acc_arr = n_acc_arr[None, :]
                n_prop_arr = n_prop_arr[None, :]
            elif n_acc_arr.ndim >= 3:
                n_moves_here = n_acc_arr.shape[-1]
                n_acc_arr = n_acc_arr.reshape(-1, n_moves_here)
                n_prop_arr = n_prop_arr.reshape(-1, n_moves_here)

        self._logger.write_entry(
            iteration=iteration,
            n_accepted=np.asarray(n_acc_arr, dtype=np.int64),
            n_proposed=np.asarray(n_prop_arr, dtype=np.int64),
        )

    def on_finish(self, ns_state: Any) -> None:
        self._logger.close()


class RECallback:
    """Writes per-fire inter-RE swap counts to HDF5.

    Reads ``info["inter_re_stats"]`` (populated by ``run_loop`` when
    the manager fires) and forwards the per-pair counts to an
    :class:`~jaxrens.io.re_stats_log.RELogger`.  Skips iterations on
    which the manager did not fire (info key absent or
    ``n_attempted_per_pair`` is empty / sums to zero) — the resulting
    file's iteration index is therefore the authoritative record of
    when the manager actually did work.

    Args:
        logger_obj: A ready-to-use ``RELogger`` instance.
    """

    def __init__(self, logger_obj: Any) -> None:
        self._logger = logger_obj

    def on_iteration(self, iteration: int, ns_state: Any, info: dict) -> None:
        re_stats = info.get("inter_re_stats")
        if re_stats is None:
            return
        n_att = re_stats.get("n_attempted_per_pair")
        if n_att is None:
            return
        n_att_arr = np.asarray(n_att, dtype=np.int32)
        if n_att_arr.size == 0 or int(n_att_arr.sum()) == 0:
            # Manager didn't fire this iteration (or fired with no
            # valid pairs).  Skip — keeps the file iteration-indexed
            # by actual fires only.
            return
        n_acc_arr = np.asarray(
            re_stats["n_accepted_per_pair"], dtype=np.int32
        )
        self._logger.write_entry(
            iteration=iteration,
            n_accepted_per_pair=n_acc_arr,
            n_attempted_per_pair=n_att_arr,
        )

    def on_finish(self, ns_state: Any) -> None:
        self._logger.close()


class EnergyCheckCallback:
    """Warns if energy is not decreasing as expected.

    ndim-agnostic: reduces batched ``info["emax"]`` to a scalar via ``max``
    before comparing to the previous iteration.
    """

    def __init__(self):
        self._prev_emax = float("inf")

    def on_iteration(self, iteration: int, ns_state: Any, info: dict) -> None:
        emax_raw = info.get("emax", 0)
        emax_arr = jnp.asarray(emax_raw)
        emax = float(emax_arr if emax_arr.ndim == 0 else jnp.max(emax_arr))
        if emax > self._prev_emax and iteration > 0:
            logger.warning(
                "iter=%d: Emax increased (%.6f > %.6f)", iteration, emax, self._prev_emax
            )
        self._prev_emax = emax

    def on_finish(self, ns_state: Any) -> None:
        pass


class CheckpointCallback:
    """Saves checkpoints at configured intervals."""

    def __init__(
        self,
        working_dir: Path | str,
        interval: int = 100,
        prefix: str = "ns",
        symbol_map: dict[int, str] | None = None,
    ):
        self.working_dir = Path(working_dir)
        self.interval = interval
        self.prefix = prefix
        self.symbol_map = symbol_map

    def on_start(self, ns_state: Any, start_info: dict | None = None) -> None:
        from jaxrens.io.checkpoint import save_checkpoint

        path = self.working_dir / f"{self.prefix}.initial.checkpoint.h5"
        state_dict = (
            _ns_state_to_checkpoint_dict(ns_state)
            if isinstance(ns_state, NSState) else ns_state
        )
        save_checkpoint(path, state_dict, self.symbol_map)

    def on_iteration(self, iteration: int, ns_state: Any, info: dict) -> None:
        if iteration > 0 and iteration % self.interval == 0:
            from jaxrens.io.checkpoint import save_checkpoint

            path = self.working_dir / f"{self.prefix}.checkpoint.h5"
            state_dict = (
                _ns_state_to_checkpoint_dict(ns_state)
                if isinstance(ns_state, NSState) else ns_state
            )
            save_checkpoint(path, state_dict, self.symbol_map)

    def on_finish(self, ns_state: Any) -> None:
        from jaxrens.io.checkpoint import save_checkpoint

        path = self.working_dir / f"{self.prefix}.final.checkpoint.h5"
        state_dict = (
            _ns_state_to_checkpoint_dict(ns_state)
            if isinstance(ns_state, NSState) else ns_state
        )
        save_checkpoint(path, state_dict, self.symbol_map)


class MemProfileCallback:
    """One-shot device-memory snapshot on the first post-compile iteration.

    Intended as a debug aid when the NS loop OOMs or when you want to see
    which HLO op owns the peak allocation (typically the vmap'd backward
    pass through the energy backend inside the galilean/HMC kernels).

    Enabled only when the ``JAXRENS_MEMPROF`` env var names the output
    filename; otherwise the callback is not constructed and the run pays
    nothing.  Inspect the result with ``pprof -top -cum <file.prof.gz>``.

    Fires twice at most: once from ``on_start`` with the ``.baseline``
    suffix (captures post-init state before any NS compile) and once from
    the first ``on_iteration`` after the first step completes (captures
    the steady-state live set).  The baseline snapshot is the only one
    that survives when iter 0 itself OOMs — which is the common failure
    mode we're trying to diagnose.
    """

    def __init__(self, out_path: Path | str):
        self.out_path = str(out_path)
        self.iter_done = False

    def _baseline_path(self) -> str:
        p = Path(self.out_path)
        return str(p.with_name(p.stem + ".baseline" + p.suffix))

    def on_start(self, ns_state: Any, start_info: dict | None = None) -> None:
        import jax

        jax.profiler.save_device_memory_profile(self._baseline_path())

    def on_iteration(self, iteration: int, ns_state: Any, info: dict) -> None:
        if self.iter_done:
            return
        import jax

        jax.profiler.save_device_memory_profile(self.out_path)
        self.iter_done = True


class TrajectoryCallback:
    """Writes dead points and snapshots at configured intervals."""

    def __init__(
        self,
        writer: Any,
        energy_logger: Any | None = None,
        traj_interval: int = 1,
        snapshot_interval: int = 100,
    ):
        self.writer = writer
        self.energy_logger = energy_logger
        self.traj_interval = traj_interval
        self.snapshot_interval = snapshot_interval

    def on_iteration(self, iteration: int, ns_state: Any, info: dict) -> None:
        if iteration % self.traj_interval == 0:
            # Latest culled walker: read directly from info (set by ns_step).
            # No need to look it up in any history buffer.
            worst_idx = int(info.get("worst_idx", 0))
            if isinstance(ns_state, NSState):
                dead_walker = {
                    "positions": info["dead_position"],
                    "types": ns_state.population.types[0],
                    "energy": float(info.get("emax", 0)),
                }
                cell = ns_state.population.cell[worst_idx]
                if jnp.any(cell != 0):
                    dead_walker["box"] = cell
            else:
                # Dict-fallback path for non-NSState callers.  Latest dead point
                # comes from info if available, else falls back to the legacy
                # state["dead_positions"][n_dead-1] read.
                if "dead_position" in info:
                    pos_latest = info["dead_position"]
                else:
                    n_dead_local = int(ns_state.get("n_dead", 0))
                    pos_latest = ns_state["dead_positions"][n_dead_local - 1]
                dead_walker = {
                    "positions": pos_latest,
                    "types": ns_state["types"][0],
                    "energy": float(info.get("emax", 0)),
                }
                if ns_state.get("cells") is not None:
                    dead_walker["box"] = ns_state["cells"][worst_idx]
            self.writer.write_dead_point(iteration, dead_walker, float(info["emax"]))

        if self.energy_logger is not None:
            # Write the culled-walker energy and volume so postprocess can
            # reconstruct the dead-point trace from this single file.
            dead_v = info.get("dead_volume", 0.0)
            self.energy_logger.write_entry(
                iteration,
                float(info.get("emax", 0)),
                volume=float(np.asarray(dead_v)),
            )

        if self.snapshot_interval and iteration > 0 and iteration % self.snapshot_interval == 0:
            if isinstance(ns_state, NSState):
                snapshot_dict = _ns_state_to_checkpoint_dict(ns_state)
                self.writer.write_walker_snapshot(iteration, snapshot_dict)
            else:
                self.writer.write_walker_snapshot(iteration, ns_state)

    def on_finish(self, ns_state: Any) -> None:
        self.writer.close()
        if self.energy_logger is not None:
            self.energy_logger.close()


class BatchedTrajectoryCallback:
    """Trajectory + energy logging for multi-run NS.

    Holds one writer and one (optional) energy logger per replica.  On each
    iteration, iterates flat replica indices and writes the per-replica dead
    walker + energy entry.  Snapshots follow the same pattern.

    For ``PmapVmapRuns`` the state arrays have leading axes ``(G, P, ...)``;
    this callback flattens them to ``(G*P, ...)`` for per-replica slicing.
    """

    def __init__(
        self,
        writers: list,
        energy_loggers: list | None = None,
        traj_interval: int = 1,
        snapshot_interval: int = 100,
    ):
        self.writers = list(writers)
        self.energy_loggers = list(energy_loggers) if energy_loggers is not None else None
        if self.energy_loggers is not None and len(self.energy_loggers) != len(self.writers):
            raise ValueError(
                f"BatchedTrajectoryCallback: len(writers)={len(self.writers)} vs "
                f"len(energy_loggers)={len(self.energy_loggers)}."
            )
        self.traj_interval = traj_interval
        self.snapshot_interval = snapshot_interval

    def on_iteration(self, iteration: int, ns_state: Any, info: dict) -> None:
        # Use the batcher (when present in info) to flatten ``(*shape_prefix, …)``
        # arrays to ``(n_runs, …)`` for per-replica indexing.  Fall back to the
        # legacy ndim-sniff when callers don't pass info["_batcher"].
        batcher = info.get("_batcher") if info is not None else None
        if batcher is not None:
            n_runs = batcher.n_runs
            flatten = batcher.flatten
        else:
            le = jnp.asarray(ns_state.log_evidence)
            n_batch_axes = max(1, int(le.ndim))
            n_runs = int(np.prod(le.shape)) if le.ndim >= 1 else 1

            def flatten(x):
                arr = x if isinstance(x, np.ndarray) else jnp.asarray(x)
                if n_batch_axes == 1:
                    return arr
                return arr.reshape((-1,) + arr.shape[n_batch_axes:])

        if n_runs != len(self.writers):
            raise RuntimeError(
                f"BatchedTrajectoryCallback: ns_state has {n_runs} replicas "
                f"but {len(self.writers)} writers were registered."
            )

        pop = ns_state.population
        types = flatten(pop.types)
        cells = flatten(pop.cell)

        # info["emax"] is (*shape_prefix,) when batched; flatten via the
        # batcher (or by ravel for the legacy fallback).
        emax_arr = jnp.asarray(info.get("emax", 0.0))
        if batcher is not None:
            emax_flat = batcher.flatten(emax_arr)
        else:
            emax_flat = (
                emax_arr.reshape(-1) if emax_arr.ndim >= 1 else emax_arr[None]
            )
        # Latest culled position per replica — sourced from info, shape
        # ``(*shape_prefix, n_atoms, 3)``; flatten leading dims for indexing.
        latest_pos = np.asarray(info["dead_position"])
        latest_pos_flat = np.asarray(flatten(latest_pos))
        # For SingleRun (n_batch_axes=0 effectively but we set min 1), the
        # batched callback shouldn't be used; assume we always have a batch.

        if iteration % self.traj_interval == 0:
            for r in range(n_runs):
                dead_walker = {
                    "positions": latest_pos_flat[r],
                    "types": types[r, 0] if types.ndim >= 2 else types[0],
                    "energy": float(emax_flat[r]),
                }
                cell_r = cells[r, 0] if cells.ndim >= 3 else cells[0]
                if jnp.any(cell_r != 0):
                    dead_walker["box"] = cell_r
                self.writers[r].write_dead_point(
                    iteration, dead_walker, float(emax_flat[r]),
                )

        if self.energy_loggers is not None:
            # Per-replica volume of the culled walker, sourced from info.
            dv_arr = np.asarray(info.get("dead_volume", 0.0))
            dv_flat = (
                dv_arr.reshape(-1) if dv_arr.ndim >= 1
                else np.broadcast_to(dv_arr, (n_runs,))
            )
            for r in range(n_runs):
                self.energy_loggers[r].write_entry(
                    iteration,
                    float(emax_flat[r]),
                    volume=float(dv_flat[r]),
                )

        if (
            self.snapshot_interval
            and iteration > 0
            and iteration % self.snapshot_interval == 0
        ):
            positions_flat = flatten(pop.positions)
            energies_flat = flatten(pop.energy)
            for r in range(n_runs):
                # Live-walker snapshot only.  Dead-point history lives on
                # disk via the per-iteration writers above.
                snap = {
                    "positions": positions_flat[r],
                    "types": types[r] if types.ndim >= 2 else types,
                    "energies": energies_flat[r],
                    "cells": cells[r] if cells.ndim >= 3 else cells,
                    "iteration": iteration,
                    "n_dead": iteration,
                }
                self.writers[r].write_walker_snapshot(iteration, snap)

    def on_finish(self, ns_state: Any) -> None:
        for w in self.writers:
            w.close()
        if self.energy_loggers is not None:
            for e in self.energy_loggers:
                e.close()
