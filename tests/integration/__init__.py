"""Integration test helpers."""

from __future__ import annotations

import pytest


def multi_gpu_n_devices(allowed: tuple[int, ...] = (2, 4)) -> int:
    """Return the number of local JAX devices, asserting it is in ``allowed``.

    Multi-GPU integration tests are sized so they work on either 2 or 4
    devices: ``n_per_gpu`` is fixed at 2 and the pressure list is sliced
    to ``n_gpu * 2``.  A device count outside ``allowed`` fails the test
    rather than skipping — silent skips on hardware misconfigs are a
    blind spot.
    """
    import jax

    n = len(jax.local_devices())
    if n not in allowed:
        pytest.fail(
            f"multi-GPU test needs one of {allowed} local devices, got {n}"
        )
    return n
