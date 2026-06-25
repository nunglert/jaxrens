"""Semi-grand μPT ensemble: schema spec, resolver wiring, and run path.

Covers the generalisation of the ensemble layer beyond pressure-only NPT:
chemical potentials are now configurable via ``ensemble: {type: semi_grand}``
and threaded through the generic ``ensemble_params`` dict (no more
pressure-special-casing on the state).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxrens.cli.resolve import resolve
from jaxrens.cli.run import run_from_config
from jaxrens.cli.schema import RootSpec
from jaxrens.cli.schema.ensemble import SemiGrandEnsembleSpec
from jaxrens.sampling.batch_descriptor import PmapVmapRuns, SingleRun
from jaxrens.state.config import (
    BackendConfig,
    MoveConfig,
    NSConfig,
    OutputConfig,
)

_GPA_TO_EVA3 = 0.006241509


def _minimal_dict() -> dict:
    return {
        "run": {
            "n_live": 20,
            "max_iterations": 30,
            "n_mcmc_steps": 5,
            "seed": 0,
        },
        "moves": [{"move_type": "random_walk", "step_size": 0.3}],
        "backend": {"backend_type": "harmonic"},
        "output": {"format": "none", "working_dir": ".", "info_interval": 999},
    }


# ---------------------------------------------------------------------------
# Schema: SemiGrandEnsembleSpec
# ---------------------------------------------------------------------------


class TestSemiGrandSpec:
    def test_single_mu_vector(self):
        spec = SemiGrandEnsembleSpec(chemical_potentials=[0.0, -0.3])
        assert spec.cohort_size() == 1
        params = spec.to_ensemble_params(cohort_index=0)
        assert params["chemical_potentials"] == [0.0, -0.3]
        assert params["pressure"] == pytest.approx(0.0)

    def test_mu_plus_pressure_eva3(self):
        spec = SemiGrandEnsembleSpec(chemical_potentials=[0.5], pressure=0.1)
        params = spec.to_ensemble_params()
        assert params["pressure"] == pytest.approx(0.1)
        assert params["chemical_potentials"] == [0.5]

    def test_pressure_gpa_converted(self):
        spec = SemiGrandEnsembleSpec(
            chemical_potentials=[0.5], pressure=1.0, pressure_units="gpa"
        )
        assert spec.to_ensemble_params()["pressure"] == pytest.approx(
            _GPA_TO_EVA3
        )

    def test_mu_cohort_indexing(self):
        spec = SemiGrandEnsembleSpec(
            chemical_potentials=[[0.0, -0.3], [0.1, -0.2], [0.2, -0.1]]
        )
        assert spec.cohort_size() == 3
        assert spec.to_ensemble_params(cohort_index=1)[
            "chemical_potentials"
        ] == [
            0.1,
            -0.2,
        ]

    def test_pressure_broadcasts_across_mu_cohort(self):
        spec = SemiGrandEnsembleSpec(
            chemical_potentials=[[0.0], [0.1]], pressure=0.2
        )
        assert spec.cohort_size() == 2
        for r in range(2):
            assert spec.to_ensemble_params(cohort_index=r)[
                "pressure"
            ] == pytest.approx(0.2)

    def test_ragged_mu_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            SemiGrandEnsembleSpec(chemical_potentials=[[0.0, -0.3], [0.1]])

    def test_empty_mu_rejected(self):
        with pytest.raises(ValueError, match="n_species"):
            SemiGrandEnsembleSpec(chemical_potentials=[])

    def test_mu_pressure_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="disagree in length"):
            SemiGrandEnsembleSpec(
                chemical_potentials=[[0.0], [0.1], [0.2]], pressure=[0.1, 0.2]
            )


# ---------------------------------------------------------------------------
# Resolver wiring
# ---------------------------------------------------------------------------


class TestSemiGrandResolver:
    def test_single_run_threads_chemical_potentials(self):
        d = _minimal_dict()
        d["ensemble"] = {"type": "semi_grand", "chemical_potentials": [0.5]}
        resolved = resolve(RootSpec.model_validate(d))
        assert isinstance(resolved.batcher, SingleRun)
        ep = resolved.ensemble_params_per_run[0]
        np.testing.assert_allclose(
            np.asarray(ep["chemical_potentials"]), [0.5], atol=1e-6
        )

    def test_mu_list_resolves_to_multi_replica(self):
        d = _minimal_dict()
        d["ensemble"] = {
            "type": "semi_grand",
            "chemical_potentials": [[0.5], [1.0], [1.5]],
        }
        resolved = resolve(RootSpec.model_validate(d))
        assert isinstance(resolved.batcher, PmapVmapRuns)
        mus = [
            float(np.asarray(p["chemical_potentials"])[0])
            for p in resolved.ensemble_params_per_run
        ]
        assert mus == pytest.approx([0.5, 1.0, 1.5])

    def test_conflict_with_inter_re_semi_grand_raises(self):
        d = _minimal_dict()
        d["ensemble"] = {
            "type": "semi_grand",
            "chemical_potentials": [[0.5], [1.0]],
        }
        d["inter_re"] = {
            "flavor": "semi_grand",
            "chemical_potentials": [[0.5], [1.0]],
            "re_interval": 5,
        }
        with pytest.raises(
            ValueError, match="Conflicting chemical potentials"
        ):
            resolve(RootSpec.model_validate(d))


# ---------------------------------------------------------------------------
# Run path
# ---------------------------------------------------------------------------


class TestSemiGrandRun:
    def test_run_from_config_with_chemical_potentials(self, tmp_path):
        """A μ-only ``ensemble_params`` wraps EnsembleBackend and completes."""
        ns_config = NSConfig(
            n_live=12, max_iterations=10, n_mcmc_steps=2, seed=0
        )
        move_config = MoveConfig(move_type="random_walk", step_size=0.3)
        backend_config = BackendConfig(backend_type="harmonic")
        output_config = OutputConfig(
            format="none",
            working_dir=tmp_path,
            info_interval=999,
            temperature_lag_interval=None,
        )
        key = jax.random.key(5)
        positions = jax.random.uniform(
            key, (12, 1, 3), minval=-3.0, maxval=3.0
        )
        types = jnp.zeros((1,), dtype=jnp.int32)
        cells = jnp.broadcast_to(jnp.eye(3) * 6.0, (12, 3, 3))
        result = run_from_config(
            ns_config,
            move_config,
            backend_config,
            output_config,
            initial_positions=positions,
            initial_types=types,
            initial_cells=cells,
            ensemble_params={"chemical_potentials": [0.5]},
        )
        assert jnp.isfinite(result["log_evidence"])
