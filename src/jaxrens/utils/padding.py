"""Padding helpers for `jax.lax.map(batch_size=N)` call sites.

`jax.lax.map` requires the input's leading axis to be exactly divisible by
`batch_size`.  `pad_to_multiple` lifts that constraint by repeating the last
entry until the leading axis is a multiple of the chunk; callers slice the
padding off the output before downstream aggregation.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def pad_to_multiple(tree: Any, axis_len: int, chunk: int) -> tuple[Any, int]:
    """Pad a pytree's leading axis up to the next multiple of ``chunk``.

    Each leaf's last entry is repeated ``n_pad`` times along axis 0.  Returns
    ``(tree, 0)`` unchanged when already divisible — no allocation, no copy.
    Works on plain arrays (treated as singleton pytrees) and on nested
    pytrees (e.g. MCState).

    The repeated leaf is a valid copy of the existing last entry, which is
    important when the padded slots will be run through code that cannot
    tolerate dummy zero-state (e.g. an energy backend on coincident atoms).
    """
    n_pad = (-axis_len) % chunk
    if n_pad == 0:
        return tree, 0
    padded = jax.tree.map(
        lambda x: jnp.concatenate(
            [x, jnp.broadcast_to(x[-1:], (n_pad, *x.shape[1:]))], axis=0
        ),
        tree,
    )
    return padded, n_pad
