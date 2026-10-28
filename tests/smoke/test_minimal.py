"""Sub-second smoke checks: package imports and a minimal NS run through
each dispatcher (``run_ns``, ``run_ns_parallel``, ``run_ns_multi_gpu``).

Only the harmonic backend is used, so nothing here compiles a real model.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxrens.backends.toy import create_harmonic
from jaxrens.sampling.move_kernel import MoveKernel
from jaxrens.sampling.moves import random_walk
from jaxrens.sampling.mwg import build_mwg
from jaxrens.sampling.nested_sampling import (
    run_ns,
    run_ns_multi_gpu,
    run_ns_parallel,
)

_N_WALKERS: int = 4
_N_ATOMS: int = 1
# run_ns adds a PriorMassTermination next to IterationTermination, and with
# 4 walkers it usually fires before _MAX_ITERATIONS. The tests only check
# that the loop ran, so n_dead may land anywhere in [1, _MAX_ITERATIONS].
_MAX_ITERATIONS: int = 50
_N_MCMC_STEPS: int = 10
_N_RUNS_PARALLEL: int = 2
_N_GPU: int = 1
_N_PER_GPU: int = 2


def test_package_imports() -> None:
    """Public entry points import."""
    import jaxrens  # noqa: F401
    from jaxrens.cli.cli import main  # noqa: F401
    from jaxrens.sampling.nested_sampling import (  # noqa: F401
        run_ns,
        run_ns_multi_gpu,
        run_ns_parallel,
    )


def test_minimal_ns_completes() -> None:
    """Smallest legal single run: no cells, callbacks, burn-in, RENS or
    adaptation."""
    backend = create_harmonic(k=1.0)
    descriptors = [
        MoveKernel(
            name="random_walk",
            build_kernel=random_walk.build_kernel,
            step_size=0.1,
            weight=1.0,
            kernel_kwargs={},
            extra_state_fields={},
        ),
    ]
    init_fn, step_fn, _ = build_mwg(backend, descriptors)

    positions = jnp.zeros((_N_WALKERS, _N_ATOMS, 3))
    types = jnp.zeros((_N_ATOMS,), dtype=jnp.int32)
    energies = jax.vmap(lambda p: backend(p, types, jnp.zeros((3, 3)), 0)[0])(
        positions
    )

    result = run_ns(
        positions=positions,
        types=types,
        energies=energies,
        cells=None,
        init_fn=init_fn,
        step_fn=step_fn,
        rng_key=jax.random.key(0),
        max_iterations=_MAX_ITERATIONS,
        n_mcmc_steps=_N_MCMC_STEPS,
    )
    assert jnp.isfinite(result["log_evidence"])
    n_dead = int(result["n_dead"])
    assert 1 <= n_dead <= _MAX_ITERATIONS


def test_minimal_ns_parallel_completes() -> None:
    """Same run through ``run_ns_parallel``, with a leading ``n_runs`` axis."""
    backend = create_harmonic(k=1.0)
    descriptors = [
        MoveKernel(
            name="random_walk",
            build_kernel=random_walk.build_kernel,
            step_size=0.1,
            weight=1.0,
            kernel_kwargs={},
            extra_state_fields={},
        ),
    ]
    init_fn, step_fn, _ = build_mwg(backend, descriptors)

    positions = jnp.zeros((_N_RUNS_PARALLEL, _N_WALKERS, _N_ATOMS, 3))
    types = jnp.zeros((_N_ATOMS,), dtype=jnp.int32)
    energies = jax.vmap(
        jax.vmap(lambda p: backend(p, types, jnp.zeros((3, 3)), 0)[0])
    )(positions)
    rng_keys = jax.random.split(jax.random.key(0), _N_RUNS_PARALLEL)

    result = run_ns_parallel(
        positions=positions,
        types=types,
        energies=energies,
        cells=None,
        init_fn=init_fn,
        step_fn=step_fn,
        rng_keys=rng_keys,
        max_iterations=_MAX_ITERATIONS,
        n_mcmc_steps=_N_MCMC_STEPS,
    )
    log_z = jnp.asarray(result["log_evidence"])
    assert log_z.shape == (_N_RUNS_PARALLEL,)
    assert jnp.all(jnp.isfinite(log_z))
    n_dead = jnp.asarray(result["n_dead"])
    assert n_dead.shape == (_N_RUNS_PARALLEL,)
    assert jnp.all((n_dead >= 1) & (n_dead <= _MAX_ITERATIONS))


def test_minimal_ns_multi_gpu_completes() -> None:
    """Same run through ``run_ns_multi_gpu`` with ``n_gpu=1, n_per_gpu=2``.

    ``pmap`` over a single device does nothing, so this also runs on CPU.
    """
    backend = create_harmonic(k=1.0)
    descriptors = [
        MoveKernel(
            name="random_walk",
            build_kernel=random_walk.build_kernel,
            step_size=0.1,
            weight=1.0,
            kernel_kwargs={},
            extra_state_fields={},
        ),
    ]
    init_fn, step_fn, _ = build_mwg(backend, descriptors)

    n_total = _N_GPU * _N_PER_GPU
    positions = jnp.zeros((n_total, _N_WALKERS, _N_ATOMS, 3))
    types = jnp.zeros((_N_ATOMS,), dtype=jnp.int32)
    energies = jax.vmap(
        jax.vmap(lambda p: backend(p, types, jnp.zeros((3, 3)), 0)[0])
    )(positions)
    rng_keys = jax.random.split(jax.random.key(0), n_total)

    result = run_ns_multi_gpu(
        positions=positions,
        types=types,
        energies=energies,
        cells=None,
        init_fn=init_fn,
        step_fn=step_fn,
        rng_keys=rng_keys,
        n_gpu=_N_GPU,
        n_per_gpu=_N_PER_GPU,
        max_iterations=_MAX_ITERATIONS,
        n_mcmc_steps=_N_MCMC_STEPS,
    )
    log_z = jnp.asarray(result["log_evidence"])
    assert log_z.shape == (_N_GPU, _N_PER_GPU)
    assert jnp.all(jnp.isfinite(log_z))
    n_dead = jnp.asarray(result["n_dead"])
    assert n_dead.shape == (_N_GPU, _N_PER_GPU)
    assert jnp.all((n_dead >= 1) & (n_dead <= _MAX_ITERATIONS))
    positions_out = jnp.asarray(result["positions"])
    assert positions_out.shape == (
        _N_GPU,
        _N_PER_GPU,
        _N_WALKERS,
        _N_ATOMS,
        3,
    )
