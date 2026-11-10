"""Tests for ``RootSpec.interval_units`` scaling in the resolver.

``_scale_interval`` and ``_apply_interval_units`` are tested directly, and
``resolve`` is checked to pass the scaled values on to the runtime configs.
In ``per_walker`` mode every interval is multiplied by n_live / n_cull.
"""

from __future__ import annotations

import pytest

from jaxrens.cli.resolve import _apply_interval_units, _scale_interval, resolve
from jaxrens.cli.schema import RootSpec
from jaxrens.cli.schema.termination import IterationTerminationSpec
from jaxrens.sampling.termination import IterationTermination


def _full_interval_dict(*, n_live: int = 10) -> dict:
    """Minimal RootSpec dict with distinct values for the interval fields."""
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


class TestScaleInterval:
    """Check rounding, clamping and None handling of one scaled value."""

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


class TestApplyIntervalUnits:
    """Check which RootSpec fields are scaled, and by what factor.

    The factor is n_live / n_cull in ``per_walker`` mode and 1 in
    ``absolute`` mode, where values are still rounded to int.
    """

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

    def test_per_walker_scales_by_n_live(self):
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

    def test_per_walker_divides_by_n_cull(self):
        """A sweep is n_live culls = n_live / n_cull iterations.

        n_live=100 with n_cull=10 is 10 iterations per sweep, the same as
        n_live=10 with n_cull=1, so the fields scale exactly as above.
        """
        d = _full_interval_dict(n_live=100)
        d["run"]["n_cull"] = 10
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

    def test_per_walker_n_cull_keeps_dead_points_per_interval(self):
        """Changing n_cull keeps the dead points (culls) per interval fixed."""
        culls = {}
        for n_cull in (1, 4, 20):
            d = _full_interval_dict(n_live=400)
            d["run"]["n_cull"] = n_cull
            d["interval_units"] = "per_walker"
            d["output"]["info_interval"] = 0.5
            out = _apply_interval_units(RootSpec.model_validate(d))
            culls[n_cull] = out.output.info_interval * n_cull
        assert culls == {1: 200, 4: 200, 20: 200}

    def test_per_walker_n_cull_rounds_and_clamps(self):
        d = _full_interval_dict(n_live=30)
        d["run"]["n_cull"] = 7  # 30 / 7 = 4.29 iterations per sweep
        d["interval_units"] = "per_walker"
        d["output"]["info_interval"] = 0.01
        out = _apply_interval_units(RootSpec.model_validate(d))
        assert out.output.traj_interval == 13  # round(3 * 4.29)
        assert out.output.info_interval == 1  # clamped, never 0

    def test_absolute_mode_ignores_n_cull(self):
        d = _full_interval_dict(n_live=100)
        d["run"]["n_cull"] = 10
        out = _apply_interval_units(RootSpec.model_validate(d))
        assert out.output.info_interval == 7
        assert out.run.max_iterations == 50

    def test_per_walker_scales_temperature_lag_interval_and_interval(self):
        """With 500 walkers, 0.5 and 0.2 sweeps are 250 and 100 iterations."""
        d = _full_interval_dict(n_live=500)
        d["interval_units"] = "per_walker"
        d["output"]["temperature_lag_interval"] = 0.5
        d["output"]["temperature_interval"] = 0.2
        root = RootSpec.model_validate(d)
        out = _apply_interval_units(root)
        assert out.output.temperature_lag_interval == 250
        assert out.output.temperature_interval == 100

    @pytest.mark.parametrize(
        "name",
        [
            "acc_rates_interval",
            "max_neighbors_interval",
            "collision_check_interval",
            "flush_interval",
        ],
    )
    def test_per_walker_scales_log_intervals(self, name):
        d = _full_interval_dict(n_live=10)
        d["interval_units"] = "per_walker"
        d["output"][name] = 0.5
        out = _apply_interval_units(RootSpec.model_validate(d))
        assert getattr(out.output, name) == 5

    def test_per_walker_leaves_other_termination_criteria(self):
        d = _full_interval_dict(n_live=10)
        d["interval_units"] = "per_walker"
        d["termination"] = [
            {"type": "iteration", "max_iterations": 4},
            {"type": "prior_mass", "threshold": 1e-5},
        ]
        out = _apply_interval_units(RootSpec.model_validate(d))
        assert out.termination[0].max_iterations == 40
        assert out.termination[1].threshold == 1e-5

    def test_temperature_lag_interval_none_passes_through(self):
        """A null ``temperature_lag_interval`` (callback off) stays None."""
        d = _full_interval_dict(n_live=10)
        d["interval_units"] = "per_walker"
        d["output"]["temperature_lag_interval"] = None
        root = RootSpec.model_validate(d)
        out = _apply_interval_units(root)
        assert out.output.temperature_lag_interval is None

    def test_per_walker_max_iterations_none_preserved(self):
        d = _full_interval_dict(n_live=10)
        d["interval_units"] = "per_walker"
        d["run"].pop("max_iterations")
        d.pop("termination", None)
        root = RootSpec.model_validate(d)
        out = _apply_interval_units(root)
        assert out.run.max_iterations is None

    def test_absolute_mode_floats_still_cast_to_int(self):
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


class TestResolveIntervals:
    """Check that ``resolve`` hands the scaled values to the runtime configs.

    The fixture's ``inter_re`` block is dropped here because a pressure
    replica exchange needs a list-valued ``ensemble.pressure`` at resolve
    time.
    """

    def test_per_walker_flows_through_resolve(self):
        d = _full_interval_dict(n_live=10)
        d["interval_units"] = "per_walker"
        d.pop("inter_re", None)
        root = RootSpec.model_validate(d)
        resolved = resolve(root)

        assert resolved.output.info_interval == 70
        assert resolved.output.snapshot_interval == 110
        assert resolved.output.checkpoint_interval == 130

        assert resolved.ns.max_iterations == 500

        iter_terms = [
            t
            for t in resolved.termination
            if isinstance(t, IterationTermination)
        ]
        # The explicit iteration criterion scales 4 -> 40; the one resolve
        # appends for run.max_iterations scales 50 -> 500.
        max_iters = sorted(int(t.max_iterations) for t in iter_terms)
        assert max_iters == [40, 500]

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
