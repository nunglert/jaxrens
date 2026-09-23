"""Shared ``jax.shard_map`` mesh construction and ``pmap``-compatibility shim.

Part of the pmap -> shard_map migration (see the design plan). Centralizes
the ``Mesh`` construction that used to be duplicated ad hoc in
``nested_sampling.py`` and ``batch_descriptor.py``, and provides
:func:`pmap_like`, an adapter that lets existing per-device function bodies
(written against ``jax.pmap``'s calling convention) run unmodified under
``jax.shard_map``.

Why the adapter is needed
--------------------------
``jax.pmap(fn, axis_name=...)`` strips the mapped axis from every input leaf
before calling ``fn`` and re-adds it (by stacking each device's output) on
the way out. ``jax.shard_map`` instead keeps a size-1 mapped axis in each
device's local view of a fully-partitioned array, and requires
``out_specs``-consistent shapes explicitly. :func:`pmap_like` squeezes that
size-1 axis off every input leaf and re-adds it to every output leaf, so
callers can write ``fn`` exactly as they would for ``jax.pmap`` and swap the
transform without touching the body. This has been verified bit-for-bit
against ``jax.pmap`` for the collective ops jaxrens relies on (``all_gather``,
``axis_index``, ``psum``, including inside ``lax.scan``/``lax.while_loop``
bodies).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from beartype.typing import Callable
from jax.sharding import Mesh, PartitionSpec


def build_mesh(axis_name: str, n_devices: int) -> Mesh:
    """Return a 1-D :class:`jax.sharding.Mesh` named *axis_name* over the
    first *n_devices* local devices.

    Single place that replaces the ``Mesh(jax.local_devices()[:n], (name,))``
    construction previously repeated in ``nested_sampling.py`` and
    ``batch_descriptor.py``.
    """
    return Mesh(jax.local_devices()[:n_devices], (axis_name,))


def pmap_like(
    fn: Callable, axis_name: str, mesh: Mesh, *, check_vma: bool = False
) -> Callable:
    """Wrap *fn* so ``jax.shard_map`` reproduces ``jax.pmap``'s calling
    convention for it.

    *fn* should be written as if for ``jax.pmap(fn, axis_name=axis_name)``:
    it receives each input leaf with the mapped axis already removed, and
    returns output leaves without that axis (pmap re-adds it by stacking).

    The returned callable is a plain ``jax.shard_map``-wrapped function with
    ``in_specs = out_specs = PartitionSpec(axis_name)`` applied uniformly
    across every leaf of the (possibly pytree-valued) arguments/outputs —
    matching ``jax.pmap``'s default ``in_axes=0``/``out_axes=0`` behavior.
    Unlike ``jax.pmap``, the result composes with ``jax.jit``.

    ``check_vma`` controls whether ``shard_map``'s "varying manual axis"
    type checker (new relative to ``pmap``, see
    https://docs.jax.dev/en/latest/notebooks/shard_map.html#scan-vma) is
    enabled. It rejects some ``lax.while_loop``/``lax.scan`` carries that
    mix a Python-literal initial value with a per-device-varying value
    produced inside the loop body — a pattern that was always legal under
    ``pmap``, which never tracked this distinction. Defaults to ``False``
    (pmap's old "trust the caller" model) so ``pmap_like`` is a faithful,
    unmodified port by default; callers whose wrapped function has been
    updated to type-check under it (e.g. via ``jax.lax.pcast`` on the
    affected carry entries — see
    ``jaxrens.sampling.adaptation.stepsize_handler.adjust_step_size``) can
    pass ``check_vma=True`` to get the checker's safety net back.
    """

    def _squeeze(x):
        return jnp.squeeze(x, axis=0)

    def _unsqueeze(x):
        return x[None, ...]

    def wrapped(*args):
        args = jax.tree.map(_squeeze, args)
        out = fn(*args)
        return jax.tree.map(_unsqueeze, out)

    spec = PartitionSpec(axis_name)
    return jax.shard_map(
        wrapped, mesh=mesh, in_specs=spec, out_specs=spec, check_vma=check_vma
    )
