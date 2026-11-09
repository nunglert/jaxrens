"""Tests for the progress and energy-check helpers in cli.monitor.

Covers _format_reject_breakdown (the reject-reason column), the column
layout of ProgressCallback lines for single and batched runs, its header
for a PmapVmapRuns run, and the per-replica Emax check of
EnergyCheckCallback.
"""

from __future__ import annotations

import logging

import jax.numpy as jnp
import numpy as np

from jaxrens.cli.monitor import (
    EnergyCheckCallback,
    ProgressCallback,
    _format_reject_breakdown,
)
from jaxrens.sampling.batch_descriptor import PmapVmapRuns


class TestFormatRejectBreakdown:
    """Check the reject-reason column of cli.monitor's progress lines.

    Percentages use the fixed-width format ``{pct:>3.0f}%`` (4 chars with
    ``%``):
    - 100% -> ``"100%"`` (no leading space)
    - 60%  -> ``" 60%"`` (one leading space)
    - 7%   -> ``"  7%"`` (two leading spaces)
    """

    def test_backward_compat_none_reasons_used(self):
        """reasons_used=None shows all three columns."""
        result = _format_reject_breakdown([10, 3, 2, 0], reasons_used=None)
        assert "E= 60%" in result
        assert "C= 40%" in result
        assert "P=  0%" in result

    def test_all_accepted_returns_empty(self):
        """No rejects gives an empty string for any reasons_used."""
        assert (
            _format_reject_breakdown(
                [10, 0, 0, 0], reasons_used=frozenset({"energy"})
            )
            == ""
        )
        assert _format_reject_breakdown([10, 0, 0, 0], reasons_used=None) == ""
        assert (
            _format_reject_breakdown(
                [10, 0, 0, 0],
                reasons_used=frozenset({"energy", "cell", "prior"}),
            )
            == ""
        )

    def test_all_three_reasons(self):
        """counts [5, 3, 1, 1]: 3, 1 and 1 of 5 rejects."""
        result = _format_reject_breakdown(
            [5, 3, 1, 1], reasons_used=frozenset({"energy", "cell", "prior"})
        )
        assert "E= 60%" in result
        assert "C= 20%" in result
        assert "P= 20%" in result

    def test_unknown_reject_flagged(self):
        """Rejects outside the declared reasons add a ??? flag."""
        # 5 rejects in cell bucket (2) but reasons_used only says energy (1)
        result = _format_reject_breakdown(
            [10, 0, 5, 0], reasons_used=frozenset({"energy"})
        )
        assert "???" in result

    def test_pct_token_width_is_four_chars(self):
        """Two-digit percentages take one leading space."""
        # 53 energy and 47 cell rejects, nothing accepted.
        result = _format_reject_breakdown(
            [0, 53, 47, 0], reasons_used=frozenset({"energy", "cell"})
        )
        assert "E= 53%" in result
        assert "C= 47%" in result

    def test_single_digit_pct_padded(self):
        """Single-digit percent is padded to two leading spaces: ``P=  7%``."""
        # 1 reject in prior out of 14 total
        result = _format_reject_breakdown(
            [1, 7, 6, 1], reasons_used=frozenset({"energy", "cell", "prior"})
        )
        assert "P=  7%" in result


def _capture_progress_lines(info: dict, batched: bool = False) -> list[str]:
    """Run ProgressCallback.on_iteration once and return the lines of its
    log message.

    ns_state is a plain dict: _is_batched reads it with .get().
    """
    if batched:
        ns_state = {"log_evidence": jnp.array([1.0, 1.1, 1.05])}
    else:
        ns_state = {"log_evidence": jnp.array(1.0)}

    cb = ProgressCallback(info_interval=1)

    records = []

    class _RecordHandler(logging.Handler):
        def emit(self, record):
            records.append(self.format(record))

    handler = _RecordHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    target_logger = logging.getLogger("jaxrens.cli.monitor")
    target_logger.addHandler(handler)
    old_level = target_logger.level
    target_logger.setLevel(logging.DEBUG)
    try:
        cb.on_iteration(100, ns_state, info)
    finally:
        target_logger.removeHandler(handler)
        target_logger.setLevel(old_level)

    assert records
    # The callback emits one multi-line message.
    return records[0].split("\n")


def _acc_col_offset(row: str) -> int:
    """Return the character index of 'acc=' within a per-move row string."""
    idx = row.find("acc=")
    assert idx != -1, row
    return idx


class TestProgressCallbackAlignment:
    """Check that acc= starts at the same column on every per-move row."""

    def _make_single_run_info(
        self, n_moves: int = 3, with_rejects: bool = True
    ):
        move_names = ["galilean", "volume", "shear_strain"]
        ss = jnp.array([5.0, 0.011, 0.5])
        acc = jnp.array([0.62, 0.10, 0.50])

        if with_rejects:
            rc = jnp.array(
                [
                    [620, 380, 0, 0],  # galilean: 38% energy reject
                    [100, 90, 810, 0],  # volume: energy + cell rejects
                    [500, 500, 0, 0],  # shear: energy rejects
                ]
            )
            move_reject_reasons = (
                frozenset({"energy"}),
                frozenset({"energy", "cell"}),
                frozenset({"energy"}),
            )
        else:
            rc = None
            move_reject_reasons = None

        return {
            "emax": -102.3,
            "step_sizes_per_move": ss,
            "move_names": move_names,
            "n_accepted_per_move": jnp.array([620, 100, 500]),
            "n_proposed_per_move": jnp.array([1000, 1000, 1000]),
            "reject_reason_counts_per_move": rc,
            "move_reject_reasons": move_reject_reasons,
        }

    def _make_batched_info(self, n_runs: int = 3, n_moves: int = 3):
        move_names = ["galilean", "volume", "shear_strain"]
        ss = jnp.ones((n_runs, n_moves)) * jnp.array([5.0, 0.011, 0.5])
        acc = jnp.ones((n_runs, n_moves)) * jnp.array([0.62, 0.10, 0.50])
        rc = jnp.ones((n_runs, n_moves, 4), dtype=jnp.int32) * jnp.array(
            [500, 300, 200, 0]
        )

        return {
            "emax": jnp.array([-102.3, -102.1, -102.5]),
            "step_sizes_per_move": ss,
            "move_names": move_names,
            "n_accepted_per_move": acc * 1000,
            "n_proposed_per_move": jnp.ones((n_runs, n_moves), dtype=jnp.int32)
            * 1000,
            "reject_reason_counts_per_move": rc,
            "move_reject_reasons": (
                frozenset({"energy"}),
                frozenset({"energy", "cell"}),
                frozenset({"energy"}),
            ),
        }

    def test_single_run_with_rejects_acc_aligned(self):
        info = self._make_single_run_info(with_rejects=True)
        lines = _capture_progress_lines(info, batched=False)
        move_rows = [l for l in lines if "acc=" in l]
        assert len(move_rows) == 3
        offsets = [_acc_col_offset(r) for r in move_rows]
        assert len(set(offsets)) == 1, "\n".join(move_rows)

    def test_single_run_reject_and_no_reject_same_column(self):
        """A row with a reject suffix and one without have acc= at the same
        column. galilean accepts everything; volume has rejects."""
        info = {
            "emax": -50.0,
            "step_sizes_per_move": jnp.array([5.0, 0.011]),
            "move_names": ["galilean", "volume"],
            "n_accepted_per_move": jnp.array([1000, 200]),
            "n_proposed_per_move": jnp.array([1000, 1000]),
            "reject_reason_counts_per_move": jnp.array(
                [
                    [
                        1000,
                        0,
                        0,
                        0,
                    ],  # galilean: all accepted -> no reject suffix
                    [200, 500, 300, 0],  # volume: energy+cell rejects
                ]
            ),
            "move_reject_reasons": (
                frozenset({"energy"}),
                frozenset({"energy", "cell"}),
            ),
        }
        lines = _capture_progress_lines(info, batched=False)
        move_rows = [l for l in lines if "acc=" in l]
        assert len(move_rows) == 2
        offsets = [_acc_col_offset(r) for r in move_rows]
        assert len(set(offsets)) == 1, "\n".join(move_rows)
        assert "reject:" not in move_rows[0]
        assert "reject:" in move_rows[1]

    def test_batched_acc_aligned(self):
        info = self._make_batched_info()
        lines = _capture_progress_lines(info, batched=True)
        move_rows = [l for l in lines if "acc=" in l]
        assert len(move_rows) == 3
        offsets = [_acc_col_offset(r) for r in move_rows]
        assert len(set(offsets)) == 1, "\n".join(move_rows)


class TestProgressCallbackPmapVmap:
    """Check ProgressCallback with a (G, P)-shaped log_evidence."""

    def test_pmap_vmap_header_shows_ranges(self):
        n_gpu, n_per_gpu = 1, 2
        descriptor = PmapVmapRuns(n_gpu=n_gpu, n_per_gpu=n_per_gpu)

        ns_state = {"log_evidence": jnp.array([[5.1, 6.2]])}

        # ns_state is a plain dict, so log_Z is read from info.
        info = {
            "emax": jnp.array([[-10.0, -9.5]]),
            "log_evidence": ns_state["log_evidence"],
            "_batcher": descriptor,
        }

        cb = ProgressCallback(info_interval=1)

        records = []

        class _RecordHandler(logging.Handler):
            def emit(self, record):
                records.append(self.format(record))

        handler = _RecordHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        target_logger = logging.getLogger("jaxrens.cli.monitor")
        target_logger.addHandler(handler)
        old_level = target_logger.level
        target_logger.setLevel(logging.DEBUG)
        try:
            cb.on_iteration(0, ns_state, info)
        finally:
            target_logger.removeHandler(handler)
            target_logger.setLevel(old_level)

        assert records
        header = records[0].split("\n")[0]
        # Batched runs show Emax and log_Z as [min..max] ranges.
        assert "Emax=[-10..-9.5]" in header
        assert "log_Z=[5.1000..6.2000]" in header


class TestEnergyCheckCallbackPerReplica:
    """Check that EnergyCheckCallback tests Emax monotonicity per replica;
    a global max would hide one replica going up."""

    @staticmethod
    def _capture_warnings(cb_action):
        records: list[str] = []

        class _Rec(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        h = _Rec()
        target = logging.getLogger("jaxrens.cli.monitor")
        target.addHandler(h)
        old = target.level
        target.setLevel(logging.WARNING)
        try:
            cb_action()
        finally:
            target.removeHandler(h)
            target.setLevel(old)
        return records

    def test_scalar_singlerun_descending_no_warn(self):
        cb = EnergyCheckCallback()
        msgs = self._capture_warnings(
            lambda: (
                cb.on_iteration(0, None, {"emax": np.array(10.0)}),
                cb.on_iteration(1, None, {"emax": np.array(8.0)}),
                cb.on_iteration(2, None, {"emax": np.array(5.0)}),
            )
        )
        assert msgs == []

    def test_scalar_singlerun_ascending_warns(self):
        cb = EnergyCheckCallback()
        msgs = self._capture_warnings(
            lambda: (
                cb.on_iteration(0, None, {"emax": np.array(10.0)}),
                cb.on_iteration(1, None, {"emax": np.array(11.0)}),
            )
        )
        assert len(msgs) == 1
        assert "non-monotonic" in msgs[0]

    def test_one_replica_ascends_others_descend_warns(self):
        cb = EnergyCheckCallback()
        # 16 replicas going down, except replica 3, which goes up.
        prev = np.linspace(50.0, 35.0, 16)
        curr = np.linspace(48.0, 34.0, 16).copy()
        curr[3] = prev[3] + 2.5
        msgs = self._capture_warnings(
            lambda: (
                cb.on_iteration(0, None, {"emax": prev}),
                cb.on_iteration(1, None, {"emax": curr}),
            )
        )
        assert len(msgs) == 1
        assert "non-monotonic on 1 replica" in msgs[0]
        assert "replica[3]" in msgs[0]

    def test_pmap_vmap_shape_warns(self):
        cb = EnergyCheckCallback()
        prev = np.array([[20.0, 19.0], [18.0, 17.0]])
        curr = np.array([[19.0, 19.5], [17.5, 16.5]])  # (0, 1) goes up
        msgs = self._capture_warnings(
            lambda: (
                cb.on_iteration(0, None, {"emax": prev}),
                cb.on_iteration(1, None, {"emax": curr}),
            )
        )
        assert len(msgs) == 1
        assert "replica[0,1]" in msgs[0]

    def test_iteration_zero_never_warns(self):
        cb = EnergyCheckCallback()
        msgs = self._capture_warnings(
            lambda: cb.on_iteration(
                0, None, {"emax": np.array([[100.0, 200.0]])}
            )
        )
        assert msgs == []
