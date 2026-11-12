"""Test that build_mwg and ns_step trace and run under jax.jit."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jaxrens.backends.toy import create_harmonic
from jaxrens.cli.resolve import resolve
from jaxrens.cli.schema import RootSpec
from jaxrens.sampling.move_kernel import MoveKernel
from jaxrens.sampling.moves import random_walk
from jaxrens.sampling.mwg import build_mwg
from jaxrens.sampling.nested_sampling import init_ns, ns_step


def _minimal_dict() -> dict:
    return {
        "run": {
            "n_live": 8,
            "max_iterations": 5,
            "n_mcmc_steps": 3,
            "seed": 1,
        },
        "moves": [{"type": "random_walk", "step_size": 0.3}],
        "backend": {"backend_type": "harmonic"},
        "output": {"format": "none", "working_dir": ".", "info_interval": 999},
    }


def test_spec_to_ns_step_jit():
    """Run spec, descriptor, build_mwg and ns_step end to end under jit."""
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

    ns_state = init_ns(
        init_fn, positions, types, energies, cells=None, rng_key=key
    )

    jit_ns_step = jax.jit(ns_step, static_argnames=("step_fn", "n_mcmc_steps"))
    new_state, info = jit_ns_step(ns_state, step_fn, n_mcmc_steps=3)

    assert new_state.iteration == 1
    assert jnp.isfinite(new_state.log_evidence)


def test_moves_are_chosen_with_probability_proportional_to_weight():
    """Weights 3 and 1 select the first move in about 75% of the calls."""
    backend = create_harmonic()
    init_fn, step_fn, _ = build_mwg(
        backend,
        [
            MoveKernel("heavy", random_walk.build_kernel, weight=3.0),
            MoveKernel("light", random_walk.build_kernel, weight=1.0),
        ],
    )
    positions = jnp.array([[0.5, 0.0, 0.0]])
    types = jnp.zeros((1,), dtype=jnp.int32)
    energy = backend(positions, types, jnp.zeros((3, 3)), 0)[0]
    state = init_fn(positions, types, energy)

    n_calls = 4000
    rng_keys = jax.random.split(jax.random.key(0), n_calls)
    move_keys = jax.random.split(jax.random.key(1), n_calls)
    _, info = jax.jit(jax.vmap(step_fn, in_axes=(0, None, None, 0)))(
        rng_keys, state, 100.0, move_keys
    )

    assert float(jnp.mean(info.move_idx == 0)) == pytest.approx(0.75, abs=0.05)


def test_conflicting_extra_field_types_raise():
    """Two moves that declare one extra field with different types fail."""

    def zeros(positions, types):
        return jnp.zeros_like(positions)

    moves = [
        MoveKernel(
            "a",
            random_walk.build_kernel,
            extra_state_fields={"aux": (jnp.ndarray, zeros)},
        ),
        MoveKernel(
            "b",
            random_walk.build_kernel,
            extra_state_fields={"aux": (float, zeros)},
        ),
    ]

    with pytest.raises(ValueError, match="Conflicting types for extra field"):
        build_mwg(create_harmonic(), moves)
