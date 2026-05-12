"""Tests for ``RootSpec.interval_units`` scaling in the resolver.

Covers ``_scale_interval`` and ``_apply_interval_units`` directly, plus the
end-to-end flow through ``resolve`` to verify the eight iteration-counted
fields are correctly rewritten on the path into the runtime dataclasses
(``OutputConfig``, ``NSConfig``, ``InterREConfig``, ``IterationTermination``,
and the full-auto ``adjust_interval`` consumed by ``run_ns``).
"""

from __future__ import annotations

import pytest

from jaxrens.cli.resolve import (
    _apply_interval_units,
    _scale_interval,
    resolve,
)
from jaxrens.cli.schema import RootSpec
from jaxrens.cli.schema.termination import IterationTerminationSpec


# ---------------------------------------------------------------------------
# Fixture: a minimal RootSpec dict that exercises all 8 interval fields.
# ---------------------------------------------------------------------------


def _full_interval_dict(*, n_live: int = 10) -> dict:
    """Minimal RootSpec dict with all 8 interval-counted fields set."""
    return {
        "run": {
            "n_live": n_live,
            "max_iterations": 50,
            "n_mcmc_steps": 5,
            "seed": 0,
        },
        "moves": [{"move_type": "random_walk", "step_size": 0.3}],
        "backend": {"backend_type": "harmonic"},
        "output": {
            "format": "none",
            "working_dir": ".",
            "info_interval": 7,
            "traj_interval": 3,
            "snapshot_interval": 11,
            "checkpoint_interval": 13,
        },
        "termination": [
            {"type": "iteration", "max_iterations": 4},
        ],
        "adaptation": {"full_auto": True, "adjust_interval": 2},
        "inter_re": {"flavor": "pressure", "re_interval": 5},
    }


# ---------------------------------------------------------------------------
# _scale_interval — unit-level behaviour
# ---------------------------------------------------------------------------


class TestScaleInterval:
    def test_none_passes_through(self):
        assert _scale_interval(None, factor=10) is None

    def test_int_factor_one_identity(self):
        assert _scale_interval(7, factor=1) == 7

    def test_int_scaled(self):
        assert _scale_interval(7, factor=10) == 70

    def test_float_rounded(self):
        assert _scale_interval(0.2, factor=500) == 100
        assert _scale_interval(100.7, factor=1) == 101

    def test_clamps_to_one(self):
        # Tiny fractional values must not collapse to zero.
        assert _scale_interval(0.0001, factor=10) == 1
        assert _scale_interval(0, factor=10) == 1


# ---------------------------------------------------------------------------
# _apply_interval_units — RootSpec rewriting
# ---------------------------------------------------------------------------


class TestApplyIntervalUnits:
    def test_default_absolute_passes_through(self):
        root = RootSpec.model_validate(_full_interval_dict(n_live=10))
        out = _apply_interval_units(root)

        assert out.interval_units == "absolute"
        assert out.output.info_interval == 7
        assert out.output.traj_interval == 3
        assert out.output.snapshot_interval == 11
        assert out.output.checkpoint_interval == 13
        assert out.run.max_iterations == 50
        assert out.adaptation.adjust_interval == 2
        assert out.inter_re.re_interval == 5
        assert isinstance(out.termination[0], IterationTerminationSpec)
        assert out.termination[0].max_iterations == 4

    def test_per_walker_scales_all_eight_fields(self):
        d = _full_interval_dict(n_live=10)
        d["interval_units"] = "per_walker"
        root = RootSpec.model_validate(d)
        out = _apply_interval_units(root)

        assert out.output.info_interval == 70
        assert out.output.traj_interval == 30
        assert out.output.snapshot_interval == 110
        assert out.output.checkpoint_interval == 130
        assert out.run.max_iterations == 500
        assert out.adaptation.adjust_interval == 20
        assert out.inter_re.re_interval == 50
        assert out.termination[0].max_iterations == 40

    def test_per_walker_accepts_floats(self):
        d = _full_interval_dict(n_live=500)
        d["interval_units"] = "per_walker"
        d["output"]["info_interval"] = 0.2  # 5 log lines per sweep
        root = RootSpec.model_validate(d)
        out = _apply_interval_units(root)
        assert out.output.info_interval == 100

    def test_per_walker_max_iterations_none_preserved(self):
        d = _full_interval_dict(n_live=10)
        d["interval_units"] = "per_walker"
        d["run"].pop("max_iterations")  # leave it at the default ``None``
        # No termination block either, so the optional ``run.max_iterations``
        # is the only iteration cap and must round-trip as None.
        d.pop("termination", None)
        root = RootSpec.model_validate(d)
        out = _apply_interval_units(root)
        assert out.run.max_iterations is None

    def test_absolute_mode_floats_still_cast_to_int(self):
        # Even in absolute mode, the resolver normalises to int via round().
        d = _full_interval_dict(n_live=10)
        d["output"]["info_interval"] = 100.7
        root = RootSpec.model_validate(d)
        out = _apply_interval_units(root)
        assert isinstance(out.output.info_interval, int)
        assert out.output.info_interval == 101

    def test_no_termination_block_handled(self):
        d = _full_interval_dict(n_live=10)
        d.pop("termination")
        d["interval_units"] = "per_walker"
        root = RootSpec.model_validate(d)
        out = _apply_interval_units(root)
        assert out.termination is None

    def test_no_inter_re_block_handled(self):
        d = _full_interval_dict(n_live=10)
        d.pop("inter_re")
        d["interval_units"] = "per_walker"
        root = RootSpec.model_validate(d)
        out = _apply_interval_units(root)
        assert out.inter_re is None


# ---------------------------------------------------------------------------
# End-to-end via resolve — values reach the runtime dataclasses
# ---------------------------------------------------------------------------


class TestResolveIntervals:
    def test_per_walker_flows_through_resolve(self):
        d = _full_interval_dict(n_live=10)
        d["interval_units"] = "per_walker"
        # The fixture's inter_re=pressure block is there to exercise the
        # `_apply_interval_units` scaling of `inter_re.re_interval` (see
        # tests above).  At resolve-time, however, inter_re=pressure
        # demands a list-valued ensemble.pressure — drop the block here
        # so the resolver takes the SingleRun path; the interval scaling
        # we want to verify happens upstream of that branch.
        d.pop("inter_re", None)
        root = RootSpec.model_validate(d)
        resolved = resolve(root)

        # Output dataclass receives absolute-iter ints.
        assert resolved.output.info_interval == 70
        assert resolved.output.snapshot_interval == 110
        assert resolved.output.checkpoint_interval == 130

        # NSConfig.max_iterations scaled too.
        assert resolved.ns.max_iterations == 500

        # The IterationTermination criterion appears in resolved.termination
        # (it is appended to whatever the spec produced, plus any explicit
        # iteration spec).  Confirm there is one whose limit matches the
        # scaled value.
        from jaxrens.sampling.termination import IterationTermination
        iter_terms = [
            t for t in resolved.termination if isinstance(t, IterationTermination)
        ]
        assert iter_terms, "expected at least one IterationTermination"
        # The explicit termination[0] from the fixture scales 4 → 40; the
        # auto-appended IterationTermination from ns.max_iterations scales
        # 50 → 500.  Both should be present.
        max_iters = sorted(int(t.max_iterations) for t in iter_terms)
        assert max_iters == [40, 500]

        # The full-auto adjust interval flows through ``adaptation_cfg``.
        assert resolved.adaptation_cfg.adjust_interval == 20

    def test_absolute_default_unchanged(self):
        d = _full_interval_dict(n_live=10)
        d.pop("inter_re", None)
        root = RootSpec.model_validate(d)
        resolved = resolve(root)
        assert resolved.output.info_interval == 7
        assert resolved.output.snapshot_interval == 11
        assert resolved.ns.max_iterations == 50
        assert resolved.adaptation_cfg.adjust_interval == 2
