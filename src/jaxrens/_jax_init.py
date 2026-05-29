"""Idempotent JAX float32 pin + CPU-fallback warning.

Imported by every jaxrens module that uses JAX, BEFORE any other JAX use.
Python caches modules in ``sys.modules`` so the second-and-onwards imports
are a single dict lookup.

The pin must run before any third-party backend (notably mace-jax) gets a
chance to flip ``jax_enable_x64`` at construction time — jaxrens represents
positions, cells and energies in float32 throughout and ``lax.cond`` branch
dtype invariants (e.g. in ``init/cells.py::cell_shape_walk``) depend on it.
"""

import os
import warnings

import jax

jax.config.update("jax_enable_x64", False)


def _warn_if_cpu_only() -> None:
    """Emit a one-shot ``RuntimeWarning`` when JAX is not using a GPU.

    jaxrens is only CI-tested on GPU; CPU runs are untested and may hit
    numerical, performance, or backend-compatibility issues (notably the
    NeuralIL bucketed-kernel dispatch and the multi-GPU pmap paths).  This
    warning is informational — it does NOT block CPU runs (some users
    legitimately want to smoke-test on a laptop).  Suppress via the standard
    ``warnings.filterwarnings("ignore", ...)`` mechanism or by setting
    ``JAXRENS_SUPPRESS_CPU_WARNING=1``.
    """
    if os.environ.get("JAXRENS_SUPPRESS_CPU_WARNING"):
        return
    # ``JAX_PLATFORMS=cpu`` is the explicit pin; ``jax.devices()`` is the
    # ground truth for "what JAX actually picked".  Check both — the env
    # var is informational (lets us mention it in the message), but the
    # device list is authoritative.
    platforms_env = os.environ.get("JAX_PLATFORMS", "")
    try:
        devices = jax.devices()
    except RuntimeError:
        # No backend at all — pointless to warn, will fail at first jit anyway.
        return
    if any(d.platform == "gpu" for d in devices):
        return
    detail = (
        f" (JAX_PLATFORMS={platforms_env!r})"
        if platforms_env == "cpu"
        else ""
    )
    warnings.warn(
        f"jaxrens is running on CPU{detail}.  The project is only CI-tested "
        f"on GPU; CPU runs are untested and may produce unexpected numerical "
        f"or performance behaviour.  Unset JAX_PLATFORMS or install a "
        f"GPU-enabled JAX build to use GPU.  Suppress this warning by setting "
        f"JAXRENS_SUPPRESS_CPU_WARNING=1.",
        RuntimeWarning,
        stacklevel=2,
    )


_warn_if_cpu_only()
