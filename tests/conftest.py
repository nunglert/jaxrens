"""Shared test fixtures for jaxrens.

Provides dummy data, known-answer toy problems, and helper functions
used across test modules.

Also handles **pytest-xdist GPU pinning**: when running under ``pytest -n
N`` on a multi-GPU host, each worker is pinned to its own GPU via
``CUDA_VISIBLE_DEVICES`` *before* JAX is imported.  JAX preallocates
~75% VRAM per process, so two workers on the same GPU OOM at startup —
sharing a GPU between processes is not supported by design.  Multi-GPU
integration tests (``-m multi_gpu``) need every device visible and must
be run in a separate, non-xdist invocation.
"""

from __future__ import annotations

import os
import subprocess


def _pin_xdist_worker_to_gpu() -> None:
    """Pin each pytest-xdist worker to a distinct GPU.

    Reads ``PYTEST_XDIST_WORKER`` (set by xdist to ``gw0``, ``gw1``, ...
    inside each worker subprocess), counts visible GPUs via
    ``nvidia-smi -L``, and sets ``CUDA_VISIBLE_DEVICES = str(worker_id)``
    so each worker sees exactly one device.  Must run before any
    ``import jax`` because JAX reads ``CUDA_VISIBLE_DEVICES`` once at
    backend init and preallocates ~75% of that GPU's VRAM — two workers
    on the same GPU OOM at startup, and JAX cannot share a device
    between processes.

    No-op when:

    * the process is not an xdist worker (controller / non-xdist run);
    * ``CUDA_VISIBLE_DEVICES`` is already pinned to a single device by
      the caller — respect their choice;
    * ``nvidia-smi`` is unavailable or reports zero GPUs (CPU run).

    Raises ``RuntimeError`` when ``-n`` exceeds the visible GPU count;
    silent oversubscription would either OOM or thrash, neither useful.
    """
    worker = os.environ.get("PYTEST_XDIST_WORKER", "")
    if not worker.startswith("gw"):
        return  # not running under pytest-xdist

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cvd and "," not in cvd:
        return  # caller already pinned to a single device, respect that

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "-L"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return  # no GPU / no nvidia-smi → leave JAX on CPU

    n_gpus = sum(1 for line in out.splitlines() if line.startswith("GPU "))
    if n_gpus == 0:
        return

    worker_id = int(worker[2:])
    if worker_id >= n_gpus:
        raise RuntimeError(
            f"pytest-xdist worker {worker} maps to GPU {worker_id}, but "
            f"only {n_gpus} GPU(s) are visible.  JAX does not share GPUs "
            f"between processes; rerun with -n <= {n_gpus}."
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_id)


_pin_xdist_worker_to_gpu()


# JAX (and anything that pulls it transitively, like jaxrens.state.*) is
# imported lazily inside fixtures so the xdist controller never touches
# CUDA — the controller doesn't run tests, only orchestrates workers, so
# initialising JAX there would just preallocate VRAM on GPU 0 and
# collide with worker ``gw0``.
import pytest


# ---------------------------------------------------------------------------
# Dummy data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng_key():
    """Fresh JAX PRNG key."""
    import jax

    return jax.random.key(42)


@pytest.fixture
def dummy_positions_3():
    """3-atom positions for minimal tests."""
    import jax.numpy as jnp

    return jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


@pytest.fixture
def dummy_types_3():
    """3-atom type codes."""
    import jax.numpy as jnp

    return jnp.array([0, 0, 1])


@pytest.fixture
def dummy_cell():
    """Cubic cell with side length 5.0."""
    import jax.numpy as jnp

    return 5.0 * jnp.eye(3)


@pytest.fixture
def dummy_walker(dummy_positions_3, dummy_types_3, dummy_cell):
    """Single WalkerState with 3 atoms in a cell."""
    import jax.numpy as jnp

    from jaxrens.state.walker import WalkerState

    return WalkerState(
        positions=dummy_positions_3,
        types=dummy_types_3,
        energy=jnp.array(-1.5),
        cell=dummy_cell,
        n_atoms=3,
    )


@pytest.fixture
def dummy_walker_nonperiodic(dummy_positions_3, dummy_types_3):
    """Single WalkerState without periodic boundaries."""
    import jax.numpy as jnp

    from jaxrens.state.walker import WalkerState

    return WalkerState(
        positions=dummy_positions_3,
        types=dummy_types_3,
        energy=jnp.array(-1.5),
        cell=None,
        n_atoms=3,
    )


# ---------------------------------------------------------------------------
# Toy energy functions (known-answer problems)
# ---------------------------------------------------------------------------


@pytest.fixture
def harmonic_energy_fn():
    """Harmonic potential: E = 0.5 * sum(positions^2).

    Known log-evidence for 3D harmonic with N atoms in a box of side L:
    log Z = N * 3/2 * log(2*pi) - 3*N*log(L)  (approximate, large L)
    """
    import jax.numpy as jnp

    def energy_fn(params, positions, types, cell=None, **kwargs):
        return 0.5 * jnp.sum(positions**2)

    return energy_fn


@pytest.fixture
def lj_pair_energy_fn():
    """Simple Lennard-Jones pair potential for testing.

    E = sum_{i<j} 4 * epsilon * [(sigma/r_ij)^12 - (sigma/r_ij)^6]
    with epsilon=1, sigma=1.
    """
    import jax.numpy as jnp

    def energy_fn(params, positions, types, cell=None, **kwargs):
        n_atoms = positions.shape[0]
        energy = jnp.array(0.0)
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                dr = positions[i] - positions[j]
                r2 = jnp.sum(dr**2)
                r6 = r2**3
                r12 = r6**2
                energy = energy + 4.0 * (1.0 / r12 - 1.0 / r6)
        return energy

    return energy_fn


# ---------------------------------------------------------------------------
# Multi-GPU helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def n_devices():
    """Number of available JAX devices."""
    import jax

    return jax.local_device_count()


@pytest.fixture
def is_multi_gpu(n_devices):
    """Whether multiple GPUs are available."""
    return n_devices > 1
