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
    """Convert NSState to a dict suitable for save_checkpoint."""
    pop = ns_state.population
    ep = ns_state.population.ensemble_params if hasattr(ns_state.population, "ensemble_params") else {}
    is_npt = isinstance(ep, dict) and "pressure" in ep
    result = {
        "positions": pop.positions,
        "types": pop.types,
        "energies": pop.energy,
        "cells": pop.cell,
        "dead_energies": ns_state.dead_energies,
        "dead_positions": ns_state.dead_positions,
        "dead_volumes": ns_state.dead_volumes if is_npt else None,
        "log_evidence": ns_state.log_evidence,
        "iteration": int(ns_state.iteration),
        "n_dead": int(ns_state.n_dead),
        "n_walkers": ns_state.n_walkers,
    }
    if is_npt:
        result["live_volumes"] = jax.vmap(get_volume)(pop.cell)
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


def _is_batched(ns_state: Any) -> bool:
    """Return True when ns_state holds multiple parallel runs.

    Detection: ``ns_state.log_evidence`` shape is () for single-run and
    (n_runs,) for batched.  We check ndim > 0 on the JAX array.
    """
    if isinstance(ns_state, NSState):
        return jnp.asarray(ns_state.log_evidence).ndim > 0
    # dict form (run_ns_parallel result)
    ev = ns_state.get("log_evidence", jnp.array(0.0))
    return jnp.asarray(ev).ndim > 0


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

        batched = _is_batched(ns_state)

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

                if batched and ss.ndim == 2:
                    # ss shape (n_runs, n_moves), acc shape (n_runs, n_moves)
                    ss_mean = jnp.mean(ss, axis=0)
                    ss_std = jnp.std(ss, axis=0)
                    acc_mean = jnp.mean(acc, axis=0)
                    acc_std = jnp.std(acc, axis=0)
                    # Reject counts: sum over runs, shape (n_moves, 4)
                    rc_sum = jnp.sum(rc, axis=0) if rc is not None else None
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

        logger.info("\n".join(lines))

    def on_finish(self, ns_state: Any) -> None:
        elapsed = time.time() - self._start_time
        if isinstance(ns_state, NSState):
            logger.info(
                "NS finished: %d iterations, log_Z=%.4f, elapsed=%.1fs",
                int(ns_state.iteration),
                float(ns_state.log_evidence),
                elapsed,
            )
        else:
            logger.info(
                "NS finished: %d iterations, log_Z=%.4f, elapsed=%.1fs",
                ns_state["iteration"],
                float(ns_state["log_evidence"]),
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

    def on_iteration(self, iteration: int, ns_state: Any, info: dict) -> None:
        ss = info.get("step_sizes_per_move")
        acc = info.get("acceptance_rates_per_move")
        if ss is None or acc is None:
            return

        ss_np = jnp.asarray(ss)
        acc_np = jnp.asarray(acc)

        # Ensure (n_runs, n_moves) shape — logger handles 1D -> (1, n_moves)
        if ss_np.ndim == 1:
            ss_np = ss_np[None, :]
            acc_np = acc_np[None, :]

        # Collect per-adjust-call diagnostic stats when present (v2 schema)
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
                adjustment_stats[_adj_rename[info_key]] = np.asarray(val)

        # Collect per-iter evaluation counts for v3 schema (shape (n_runs, n_moves))
        n_evals_raw = info.get("n_evaluations_per_move")
        n_grad_evals_raw = info.get("n_grad_evaluations_per_move")
        n_evals_np: "np.ndarray | None" = None
        n_grad_evals_np: "np.ndarray | None" = None
        if n_evals_raw is not None:
            arr = np.asarray(n_evals_raw, dtype=np.int64)
            # Ensure (n_runs, n_moves) shape
            n_evals_np = arr[None, :] if arr.ndim == 1 else arr
        if n_grad_evals_raw is not None:
            arr = np.asarray(n_grad_evals_raw, dtype=np.int64)
            n_grad_evals_np = arr[None, :] if arr.ndim == 1 else arr

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


class EnergyCheckCallback:
    """Warns if energy is not decreasing as expected."""

    def __init__(self):
        self._prev_emax = float("inf")

    def on_iteration(self, iteration: int, ns_state: Any, info: dict) -> None:
        emax = float(info.get("emax", 0))
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

    def on_iteration(self, iteration: int, ns_state: Any, info: dict) -> None:
        if iteration > 0 and iteration % self.interval == 0:
            from jaxrens.io.checkpoint import save_checkpoint

            path = self.working_dir / f"{self.prefix}.checkpoint.h5"
            state_dict = _ns_state_to_checkpoint_dict(ns_state) if isinstance(ns_state, NSState) else ns_state
            save_checkpoint(path, state_dict, self.symbol_map)

    def on_finish(self, ns_state: Any) -> None:
        from jaxrens.io.checkpoint import save_checkpoint

        path = self.working_dir / f"{self.prefix}.final.checkpoint.h5"
        state_dict = _ns_state_to_checkpoint_dict(ns_state) if isinstance(ns_state, NSState) else ns_state
        save_checkpoint(path, state_dict, self.symbol_map)


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
            if isinstance(ns_state, NSState):
                dead_walker = {
                    "positions": ns_state.dead_positions[ns_state.n_dead - 1],
                    "types": ns_state.population.types[0],
                    "energy": float(info.get("emax", 0)),
                }
                # cell is always present on MCState (zeros for non-periodic)
                worst_idx = int(info.get("worst_idx", 0))
                cell = ns_state.population.cell[worst_idx]
                if jnp.any(cell != 0):
                    dead_walker["box"] = cell
            else:
                worst_idx = int(info.get("worst_idx", 0))
                dead_walker = {
                    "positions": ns_state["dead_positions"][ns_state["n_dead"] - 1],
                    "types": ns_state["types"][0],
                    "energy": float(info.get("emax", 0)),
                }
                if ns_state.get("cells") is not None:
                    dead_walker["box"] = ns_state["cells"][worst_idx]
            self.writer.write_dead_point(iteration, dead_walker, float(info["emax"]))

        if self.energy_logger is not None:
            self.energy_logger.write_entry(
                iteration, float(info.get("emax", 0))
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
