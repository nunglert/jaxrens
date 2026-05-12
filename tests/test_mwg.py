"""Canonical JIT-plumbing test for build_mwg + ns_step.

Replaces three redundant JIT smoke tests that lived in test_schema.py:
  - TestJitEndToEnd::test_spec_descriptor_mwg_ns_step_jit
  - TestJitEndToEndBackendSpec::test_resolved_energy_backend_in_ns_step_jit
  - TestJitEndToEndBackendSpec::test_double_well_backend_in_ns_step_jit
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxrens.backends.toy import create_harmonic
from jaxrens.cli.resolve import resolve
from jaxrens.cli.schema import RootSpec
from jaxrens.sampling.mwg import build_mwg
from jaxrens.sampling.nested_sampling import init_ns, ns_step


def _minimal_dict() -> dict:
    return {
        "run": {"n_live": 8, "max_iterations": 5, "n_mcmc_steps": 3, "seed": 1},
        "moves": [{"type": "random_walk", "step_size": 0.3}],
        "backend": {"backend_type": "harmonic"},
        "output": {"format": "none", "working_dir": ".", "info_interval": 999},
    }


class TestMwgJitPlumbing:
    """Single canonical test: spec -> descriptor -> build_mwg -> ns_step under jit."""

    def test_spec_to_ns_step_jit(self):
        """Confirm the spec->descriptor->mwg->ns_step pipeline traces cleanly under JIT."""
        root = RootSpec.model_validate(_minimal_dict())
        resolved = resolve(root)

        backend = create_harmonic()
        init_fn, step_fn, _ = build_mwg(backend, list(resolved.move_descriptors))

        key = jax.random.key(42)
        key, key_pos = jax.random.split(key)
        positions = jax.random.uniform(key_pos, (8, 1, 3), minval=-2.0, maxval=2.0)
        types = jnp.zeros((1,), dtype=jnp.int32)

        energies = jax.vmap(
            lambda pos: backend(pos, types, jnp.zeros((3, 3)), 0)[0]
        )(positions)

        ns_state = init_ns(init_fn, positions, types, energies, cells=None, rng_key=key)

        jit_ns_step = jax.jit(ns_step, static_argnames=("step_fn", "n_mcmc_steps"))
        new_state, info = jit_ns_step(ns_state, step_fn, n_mcmc_steps=3)

        assert new_state.iteration > 0 or new_state.n_dead > 0
        assert jnp.isfinite(new_state.log_evidence) or new_state.n_dead == 0


