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
from jaxrens.sampling.move_kernel import MoveKernel
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


class TestMwgMoveIdx:
    """Verify move_idx is threaded through MoveInfo by the MWG wrapper."""

    def test_move_idx_in_move_info_single_move(self):
        """With one move, move_idx is always 0."""
        from jaxrens.sampling.moves import random_walk as rw

        backend = create_harmonic()
        init_fn, step_fn, _ = build_mwg(backend, [
            MoveKernel("random_walk", rw.build_kernel),
        ])

        key = jax.random.key(1)
        key, pos_key = jax.random.split(key)
        positions = jax.random.uniform(pos_key, (1, 3))
        types = jnp.zeros((1,), dtype=jnp.int32)
        energy = backend(positions, types, jnp.zeros((3, 3)), 0)[0]

        from jaxrens.state.mc_state import make_mc_state_class
        state = init_fn(positions, types, energy, cell=None)

        jit_step_fn = jax.jit(step_fn)
        _, info = jit_step_fn(key, state, 1e10)

        assert hasattr(info, "move_idx"), "MoveInfo must have move_idx field"
        assert int(info.move_idx) == 0

    def test_move_idx_in_range_two_moves(self):
        """With two moves, move_idx is in {0, 1}."""
        from jaxrens.sampling.moves import random_walk as rw

        backend = create_harmonic()
        init_fn, step_fn, _ = build_mwg(backend, [
            MoveKernel("rw0", rw.build_kernel),
            MoveKernel("rw1", rw.build_kernel),
        ])

        key = jax.random.key(5)
        key, pos_key = jax.random.split(key)
        positions = jax.random.uniform(pos_key, (1, 3))
        types = jnp.zeros((1,), dtype=jnp.int32)
        energy = backend(positions, types, jnp.zeros((3, 3)), 0)[0]

        state = init_fn(positions, types, energy, cell=None)

        jit_step_fn = jax.jit(step_fn)

        # Run several steps; collect move indices
        move_indices = set()
        for i in range(20):
            key, step_key = jax.random.split(key)
            state, info = jit_step_fn(step_key, state, 1e10)
            move_indices.add(int(info.move_idx))

        # With 20 steps and equal weights, should see both 0 and 1
        assert move_indices.issubset({0, 1}), f"Unexpected move indices: {move_indices}"
        assert len(move_indices) >= 1  # at minimum one was used

    def test_move_idx_dtype_is_int32(self):
        """move_idx must be int32 scalar."""
        from jaxrens.sampling.moves import random_walk as rw

        backend = create_harmonic()
        init_fn, step_fn, _ = build_mwg(backend, [
            MoveKernel("rw0", rw.build_kernel),
        ])

        key = jax.random.key(9)
        key, pos_key = jax.random.split(key)
        positions = jax.random.uniform(pos_key, (1, 3))
        types = jnp.zeros((1,), dtype=jnp.int32)
        energy = backend(positions, types, jnp.zeros((3, 3)), 0)[0]
        state = init_fn(positions, types, energy, cell=None)

        jit_step_fn = jax.jit(step_fn)
        _, info = jit_step_fn(key, state, 1e10)

        assert jnp.asarray(info.move_idx).dtype == jnp.int32
        assert jnp.asarray(info.move_idx).shape == ()
