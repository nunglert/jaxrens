"""Sub-second smoke checks: package install, CLI wiring, minimal NS skeleton.

The point of these tests is to fail loudly when the build is fundamentally
broken — they're the first signal in CI.  They deliberately avoid:

* Real physics backends (uses ``HarmonicBackend`` only — no JIT compile of
  LJ / MACE / NeuralIL).
* Adaptation, burn-in, replica exchange, callbacks, disk I/O — all of
  which have their own integration / unit tests.
* Cell-aware paths.

If anything here fails, the full suite (``tests/`` and
``tests/integration/``) won't be trustworthy until it's resolved.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Shared parameters
# ---------------------------------------------------------------------------
# Tweak in one place if the smoke suite ever needs to be tuned for runtime
# vs. coverage.  The current values comfortably finish in under a second
# per test on CPU after the first JIT compile, while still being big enough
# to exercise the loop body more than a single iteration.
_N_WALKERS: int = 4
_N_ATOMS: int = 1
_MAX_ITERATIONS: int = 50
_N_MCMC_STEPS: int = 10
_N_RUNS_PARALLEL: int = 2
_N_GPU: int = 1
_N_PER_GPU: int = 2

# n_dead assertion bound.  ``run_ns`` adds a default ``PriorMassTermination``
# (driven by ``convergence_threshold``) alongside the user's
# ``IterationTermination(max_iterations)``.  With ``n_walkers=4`` the prior
# mass shrinks fast and PriorMassTermination typically fires before iter 50,
# so n_dead lands somewhere in [1, _MAX_ITERATIONS].  The smoke test only
# cares that the loop *ran*, not which criterion ended it.


def test_package_imports() -> None:
    """If this fails, the wheel/install is broken."""
    import jaxrens  # noqa: F401
    from jaxrens.cli.cli import main  # noqa: F401
    from jaxrens.sampling.nested_sampling import (  # noqa: F401
        run_ns,
        run_ns_multi_gpu,
        run_ns_parallel,
    )


def test_minimal_ns_completes() -> None:
    """Tiniest legal single-run NS: harmonic backend, no cells, no callbacks,
    no burn-in, no RENS, no adaptation.

    Catches: ``run_ns`` signature changes, ``init_ns`` / ``ns_step`` /
    ``_run_loop`` regressions, result-dict shape changes, RNG plumbing
    breakage.  Does *not* exercise: any real backend, any I/O, any of the
    batched dispatchers, anything cell-aware.
    """
    from jaxrens.backends.toy import create_harmonic
    from jaxrens.sampling.move_kernel import MoveKernel
    from jaxrens.sampling.moves import random_walk
    from jaxrens.sampling.mwg import build_mwg
    from jaxrens.sampling.nested_sampling import run_ns

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
    energies = jax.vmap(
        lambda p: backend(p, types, jnp.zeros((3, 3)), 0)[0]
    )(positions)

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
    assert 1 <= n_dead <= _MAX_ITERATIONS, (
        f"n_dead={n_dead} outside [1, {_MAX_ITERATIONS}]"
    )


def test_minimal_ns_parallel_completes() -> None:
    """Same as :func:`test_minimal_ns_completes` but routed through
    :func:`run_ns_parallel` — exercises the ``(n_runs, ...)`` batch
    shape, the vmap-wrapped ``ns_step``, and the per-run PRNG splitting.

    Catches: vmap'd init / step regressions, batch-shape mismatches,
    per-run rng plumbing, ``log_evidence.shape == (n_runs,)`` invariants
    in the result dict.
    """
    from jaxrens.backends.toy import create_harmonic
    from jaxrens.sampling.move_kernel import MoveKernel
    from jaxrens.sampling.moves import random_walk
    from jaxrens.sampling.mwg import build_mwg
    from jaxrens.sampling.nested_sampling import run_ns_parallel

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
    # (n_runs, n_walkers) energies — double vmap over leading axes.
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
    assert log_z.shape == (_N_RUNS_PARALLEL,), (
        f"expected ({_N_RUNS_PARALLEL},), got {log_z.shape}"
    )
    assert jnp.all(jnp.isfinite(log_z)), f"non-finite log_evidence: {log_z}"
    n_dead = jnp.asarray(result["n_dead"])
    assert n_dead.shape == (_N_RUNS_PARALLEL,)
    assert jnp.all((n_dead >= 1) & (n_dead <= _MAX_ITERATIONS)), (
        f"n_dead={n_dead} outside [1, {_MAX_ITERATIONS}] per replica"
    )


def test_minimal_ns_multi_gpu_completes() -> None:
    """Same as :func:`test_minimal_ns_parallel_completes` but routed through
    :func:`run_ns_multi_gpu` with ``n_gpu=1, n_per_gpu=2``.

    pmap over a single device is an effective no-op; vmap over the
    ``n_per_gpu`` axis does the actual work.  This means the test runs on
    *any* machine with ≥ 1 JAX device (CPU is fine — no ``@pytest.mark.gpu``
    needed) while still exercising the full multi-GPU code path.

    Catches: ``PmapVmapRuns`` descriptor wiring, ``init_ns_multi_gpu``
    flatten / reshape regressions, ``(G, P, ...)`` result-dict shape
    invariants, the pmap'd ``ns_step`` compile path.
    """
    from jaxrens.backends.toy import create_harmonic
    from jaxrens.sampling.move_kernel import MoveKernel
    from jaxrens.sampling.moves import random_walk
    from jaxrens.sampling.mwg import build_mwg
    from jaxrens.sampling.nested_sampling import run_ns_multi_gpu

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
    assert log_z.shape == (_N_GPU, _N_PER_GPU), (
        f"expected (n_gpu, n_per_gpu)=({_N_GPU}, {_N_PER_GPU}), got {log_z.shape}"
    )
    assert jnp.all(jnp.isfinite(log_z)), f"non-finite log_evidence: {log_z}"
    n_dead = jnp.asarray(result["n_dead"])
    assert n_dead.shape == (_N_GPU, _N_PER_GPU)
    assert jnp.all((n_dead >= 1) & (n_dead <= _MAX_ITERATIONS)), (
        f"n_dead={n_dead} outside [1, {_MAX_ITERATIONS}] per replica"
    )
    # Live walkers retained their (G, P, n_walkers, n_atoms, 3) layout.
    positions_out = jnp.asarray(result["positions"])
    assert positions_out.shape == (_N_GPU, _N_PER_GPU, _N_WALKERS, _N_ATOMS, 3)
