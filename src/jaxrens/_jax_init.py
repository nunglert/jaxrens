"""Idempotent JAX precision config + CPU-fallback warning.

Imported by every jaxrens module that uses JAX, BEFORE any other JAX use.
Python caches modules in ``sys.modules`` so the second-and-onwards imports
are a single dict lookup.

The precision config must run before any third-party backend (notably
mace-jax) gets a chance to flip ``jax_enable_x64`` at construction time —
jaxrens represents positions, cells and energies in float32 throughout and
``lax.cond`` branch dtype invariants (e.g. in
``init/cells.py::cell_shape_walk``) assume it.  We default to float32 but
honour an explicit ``JAX_ENABLE_X64`` opt-in (with a warning) for users who
deliberately want double precision.

Two independent JAX knobs make up the precision policy:

* ``jax_enable_x64`` -- whether the *default* float dtype is 64-bit.  jaxrens
  pins this ``False`` so the sampling pipeline stays float32.
* ``jax_explicit_x64_dtypes`` -- what happens when code *explicitly* asks for
  float64 while ``jax_enable_x64`` is off.  We pin it to ``"warn"`` (truncate
  to float32 + ``UserWarning``) so a stray float64 in the sampling code is a
  loud tripwire.  Post-processing that genuinely needs double precision (e.g.
  the Steinhardt order parameters, whose spherical-harmonic sums lose all
  signal in float32) opts in for the scope of its computation via the
  ``allow_explicit_x64()`` context manager below -- defaults stay float32, so
  the sampling pipeline is untouched.
"""

import contextlib
import os
import tempfile
import warnings

import jax


def _configure_x64() -> None:
    """Default JAX to float32; honour an explicit x64 opt-in with a warning.

    jaxrens represents positions, cells and energies in float32 throughout
    and several ``lax.cond`` branch dtype invariants assume it.  We set
    ``jax_enable_x64`` *explicitly* (not just leave it at JAX's default) so a
    third-party backend can't silently flip it on at construction time.

    A user who deliberately wants double precision can set the standard
    ``JAX_ENABLE_X64`` env var; we honour it but warn that jaxrens is untested
    in 64-bit mode.
    """
    raw = os.environ.get("JAX_ENABLE_X64", "")
    enable = raw.strip().lower() in ("1", "true", "yes")
    jax.config.update("jax_enable_x64", enable)
    # Pin the explicit-float64 policy to a loud tripwire for the sampling code;
    # post-processing opts into 64-bit locally via ``allow_explicit_x64()``.
    # (No-op once x64 is on outright -- explicit float64 is honoured then.)
    if not enable:
        jax.config.update("jax_explicit_x64_dtypes", "warn")
    if enable:
        warnings.warn(
            f"jaxrens: JAX_ENABLE_X64={raw!r} enables 64-bit (double) "
            f"precision.  jaxrens is developed and CI-tested in float32 only; "
            f"64-bit is untested and several lax.cond dtype invariants assume "
            f"float32, so runs may fail or behave unexpectedly.  Unset "
            f"JAX_ENABLE_X64 to use the supported float32 default.",
            RuntimeWarning,
            stacklevel=2,
        )


@contextlib.contextmanager
def allow_explicit_x64():
    """Honour explicit float64 / complex128 dtypes for the enclosed block.

    jaxrens keeps ``jax_enable_x64`` off and the explicit-x64 policy at
    ``"warn"`` so the sampling pipeline stays float32 and any stray float64
    there is flagged.  Post-processing that legitimately needs double precision
    wraps its computation in this context manager: explicitly-requested
    float64/complex128 arrays are honoured inside, while *default* float dtypes
    remain float32 (so float32 code paths are unaffected), and the previous
    policy is restored on exit -- leaving no global precision state behind.

    Idempotent and re-entrant (nesting restores the outer policy).  No-op when
    ``jax_enable_x64`` is already on, since float64 is honoured outright then.
    """
    prev = jax.config.jax_explicit_x64_dtypes
    jax.config.update("jax_explicit_x64_dtypes", "allow")
    try:
        yield
    finally:
        jax.config.update("jax_explicit_x64_dtypes", prev)


def _check_tmpdir_writable() -> None:
    """Fail fast with a clear message when ``$TMPDIR`` is not writable.

    JAX writes compilation artifacts there; if it's read-only JAX raises a
    confusing ``JaxRuntimeError`` deep inside the first jit.  Bail here
    instead, with an actionable message.
    """
    tmpdir = os.environ.get("TMPDIR", "/tmp")
    try:
        with tempfile.NamedTemporaryFile(dir=tmpdir, delete=True):
            pass
    except OSError as e:
        raise RuntimeError(
            f"jaxrens: {tmpdir!r} is not writable ({e}). "
            f"JAX writes compilation artifacts there and will fail with a "
            f"confusing JaxRuntimeError later. "
            f"Set $TMPDIR to a writable directory before importing jaxrens."
        ) from e


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
        f" (JAX_PLATFORMS={platforms_env!r})" if platforms_env == "cpu" else ""
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


_configure_x64()  # must precede any JAX op — always runs

# The TMPDIR + CPU checks are about *executing* JAX work.  Commands that
# import JAX-bound modules but run no JAX op (notably ``jaxrens dump-schema``,
# which only serialises pydantic models) set ``JAXRENS_SKIP_RUNTIME_CHECKS=1``
# before importing, so the checks don't false-alarm.  Everything that actually
# uses JAX — every run, and every test that imports a jaxrens JAX module and
# then calls into it — still triggers them.
if not os.environ.get("JAXRENS_SKIP_RUNTIME_CHECKS"):
    _check_tmpdir_writable()
    _warn_if_cpu_only()
