"""Bucketed kernel compilation and runtime dispatch.

Implements the CompiledKernelSet pattern for handling dynamic max_neighbors
in NeuralIL-style backends. Pre-compiles kernels for a list of max_neighbors
values, then dispatches to the best-fit kernel at runtime.

This is jaxnest's proven strategy, formalized as a clean abstraction:
- Pre-compile: one kernel per max_neighbors bucket
- Dispatch: select smallest bucket >= current max + offset
- Retry: on violation, escalate to next bucket

For backends without neighbor concerns (toy, LJ), use SingleKernelSet
which wraps a single kernel with no dispatch overhead.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


def find_closest_higher_number(
    input_num: int,
    number_list: list[int],
    extra_offset: int = 0,
) -> int:
    """Find the smallest value in number_list >= input_num, then shift by extra_offset.

    Args:
        input_num: Target number to exceed.
        number_list: Sorted list of available bucket values.
        extra_offset: Additional index offset into the list.

    Returns:
        Selected bucket value.

    Raises:
        ValueError: If no bucket is large enough.

    Examples:
        >>> find_closest_higher_number(12, [4, 6, 8, 10, 15, 20])
        15
        >>> find_closest_higher_number(12, [4, 6, 8, 10, 15, 20], extra_offset=1)
        20
    """
    sorted_list = sorted(number_list)
    for i, val in enumerate(sorted_list):
        if val >= input_num:
            target_idx = min(i + extra_offset, len(sorted_list) - 1)
            return sorted_list[target_idx]
    raise ValueError(
        f"No bucket large enough for {input_num}. "
        f"Available: {sorted_list}. Consider adding larger buckets."
    )


class CompiledKernelSet:
    """Pre-compiled kernels for different max_neighbors values.

    This is the abstraction over jaxnest's NeuralILBackendHandler pattern.
    For backends that don't need multi-kernel dispatch, use SingleKernelSet.

    Usage:
        kernel_set = CompiledKernelSet(
            kernel_factory=lambda max_neighbors: compile_kernel(max_neighbors),
            max_neighbors_list=[30, 35, 40, 45, 50],
            max_neighbors_offset=5,
        )

        # In the outer NS loop:
        kernel = kernel_set.select(current_max_neighbors)
        result = kernel(walkers, ...)

        # Check for violations and retry if needed:
        if has_violation(result):
            kernel = kernel_set.select(current_max_neighbors, extra_offset=1)
            result = kernel(walkers, ...)
    """

    def __init__(
        self,
        kernel_factory: Callable[[int], Any],
        max_neighbors_list: list[int],
        max_neighbors_offset: int = 5,
    ):
        """
        Args:
            kernel_factory: Callable that takes max_neighbors and returns
                a compiled kernel (e.g., a pmap'd/vmap'd function).
            max_neighbors_list: List of max_neighbors values to pre-compile for.
            max_neighbors_offset: Safety margin added to current neighbor count
                before bucket selection.
        """
        self.max_neighbors_list = sorted(max_neighbors_list)
        self.max_neighbors_offset = max_neighbors_offset
        self._kernels: dict[int, Any] = {}

        for n in self.max_neighbors_list:
            logger.info("Pre-compiling kernel for max_neighbors=%d", n)
            self._kernels[n] = kernel_factory(n)

        self.current_max_neighbors: int = self.max_neighbors_list[0]

    def select(
        self,
        current_neighbor_count: int,
        extra_offset: int = 0,
    ) -> Any:
        """Select the best-fit pre-compiled kernel.

        Args:
            current_neighbor_count: Maximum neighbor count in current walkers.
            extra_offset: Additional bucket offset (used during retry escalation).

        Returns:
            Pre-compiled kernel for the selected bucket.
        """
        target = current_neighbor_count + self.max_neighbors_offset
        bucket = find_closest_higher_number(
            target, self.max_neighbors_list, extra_offset
        )
        self.current_max_neighbors = bucket
        return self._kernels[bucket]

    def adjust_and_run(
        self,
        walkers: Any,
        get_neighbor_count: Callable[[Any], int],
        run_fn: Callable[[Any, Any], tuple[Any, Any]],
        check_violation: Callable[[Any], bool],
        max_retries: int = 3,
    ) -> tuple[Any, Any]:
        """Select kernel, run, retry with escalation on violation.

        Args:
            walkers: Current walker state.
            get_neighbor_count: Callable to extract max neighbor count from walkers.
            run_fn: Callable(kernel, walkers) -> (result_walkers, info).
            check_violation: Callable(result_walkers) -> bool.
            max_retries: Maximum number of escalation retries.

        Returns:
            (result_walkers, info) from successful run.

        Raises:
            ValueError: If all retries exhausted.
        """
        current_count = get_neighbor_count(walkers)

        for attempt in range(max_retries + 1):
            kernel = self.select(current_count, extra_offset=attempt)
            result_walkers, info = run_fn(kernel, walkers)

            if not check_violation(result_walkers):
                return result_walkers, info

            logger.warning(
                "Neighbor violation detected (attempt %d/%d, bucket=%d). Escalating.",
                attempt + 1,
                max_retries + 1,
                self.current_max_neighbors,
            )

        raise ValueError(
            f"Neighbor violations persisted after {max_retries + 1} attempts. "
            f"Max bucket: {self.max_neighbors_list[-1]}. "
            f"Consider adding larger buckets to max_neighbors_list."
        )


class SingleKernelSet:
    """Trivial wrapper for backends without neighbor dispatch.

    Provides the same interface as CompiledKernelSet but with a single kernel.
    Used for toy, LJ, and other backends where max_neighbors is not relevant.
    """

    def __init__(self, kernel: Any):
        self._kernel = kernel
        self.current_max_neighbors: int = 0

    def select(self, current_neighbor_count: int = 0, extra_offset: int = 0) -> Any:
        return self._kernel
