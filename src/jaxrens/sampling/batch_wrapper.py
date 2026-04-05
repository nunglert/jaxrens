"""Batch wrapper: three-level parallelism for multi-GPU execution.

Creates the pmap(vmap(vmap(...))) wrapper that turns single-walker
move kernels into batched, multi-GPU operations.

The wrapper handles the (G, P, K, ...) data shape convention:
  G = n_gpu_parallel (pmap axis)
  P = n_runs_per_gpu (outer vmap)
  K = n_walkers (inner vmap)
"""

from __future__ import annotations

from typing import Any, Callable

import jax


def create_batch_wrapper(
    platform: str = "gpu",
    n_gpu_parallel: int = 1,
) -> Callable:
    """Create the appropriate batch wrapper for the platform.

    Args:
        platform: "gpu" (single GPU), "multi-gpu", or "cpu".
        n_gpu_parallel: Number of GPUs (only used for multi-gpu).

    Returns:
        A function that wraps a single-walker kernel into a batched version.
        The wrapper takes a function with signature:
            f(key, state, constraint) -> (state, info)
        and returns a function that operates on batched inputs.
    """
    if platform == "multi-gpu" and n_gpu_parallel > 1:
        # pmap over GPUs, vmap over runs, vmap over walkers
        def wrapper(func: Callable) -> Callable:
            return jax.pmap(jax.vmap(jax.vmap(func, in_axes=(0, 0, None)), in_axes=(0, 0, None)))
        return wrapper
    else:
        # Single device: jit(vmap(vmap(vmap(...)))) for (G=1, P, K)
        # or jit(vmap(vmap(...))) for (P, K) when no GPU dimension needed
        def wrapper(func: Callable) -> Callable:
            return jax.jit(jax.vmap(jax.vmap(func, in_axes=(0, 0, None)), in_axes=(0, 0, None)))
        return wrapper
