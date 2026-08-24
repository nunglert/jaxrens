"""Mark code paths that have never been exercised in a production simulation.

This is about *validation*, not test coverage.  A function marked here may be
fully unit-tested and still carry the marker: what is missing is a real nested
sampling run whose output someone looked at and believed.  The two axes are
orthogonal — use ``# pragma: no cover`` for the coverage gap, and this module
for "I am not yet confident this does what I intended".

Usage::

    from jaxrens.unvalidated import unvalidated

    @unvalidated(
        concern="the accept criterion is a hard delta_H < 1.0 cut, not a "
                "Metropolis test",
        since="0.2.2",
        clears_when="an NS run whose HMC accept rate tracks the random-walk move",
    )
    def build_kernel(...):
        ...

The decorator registers at *import* (so the unvalidated surface is enumerable
without calling anything, see :data:`REGISTRY`) and warns on the *first call*
(so a user only hears about the paths they actually touch).  Every marker that
fires is recorded in :func:`triggered`, which is what the HDF5 trajectory
writer stamps into the output file — a transient stderr warning is invisible in
a batch job, but a caveat attached to the data survives.

Policy is set by the ``JAXRENS_UNVALIDATED`` env var:

``warn`` (default)
    Emit an :class:`UnvalidatedFeatureWarning` once per feature, and log it to
    the ``jaxrens`` logger (so it lands in the run's log file).
``ignore``
    Stay silent, but still record the marker in :func:`triggered` so the output
    file is stamped regardless.
``error``
    Raise.  Useful for a production campaign where you want a hard guarantee
    that nothing unvalidated was touched.

Deliberately stdlib-only: this module is imported from JAX-free paths (notably
:mod:`jaxrens.io.trajectory`) and must not pull JAX in.
"""

from __future__ import annotations

import functools
import logging
import os
import textwrap
import warnings
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

__all__ = [
    "REGISTRY",
    "Unvalidated",
    "UnvalidatedFeatureWarning",
    "format_summary",
    "reset",
    "triggered",
    "unvalidated",
    "warn_unvalidated",
]

logger = logging.getLogger("jaxrens.unvalidated")

_ENV_VAR = "JAXRENS_UNVALIDATED"
_POLICIES = ("warn", "ignore", "error")

F = TypeVar("F", bound=Callable[..., Any])


class UnvalidatedFeatureWarning(UserWarning):
    """A code path that has never been exercised in a production simulation.

    Unit tests may well cover it; what is missing is a real run whose output
    someone checked.  Its own category (rather than a bare ``UserWarning``) so
    it can be filtered, escalated, or silenced on its own.
    """


@dataclass(frozen=True)
class Unvalidated:
    """Provenance record for one unvalidated code path."""

    feature: str
    """Dotted name, e.g. ``jaxrens.sampling.moves.hmc.build_kernel``."""

    concern: str
    """What specifically is not trusted.  The field that earns its keep."""

    since: str
    """Version the marker was added, so a stale one is visible."""

    clears_when: str = ""
    """What run would justify removing the marker.  Empty if unspecified."""


REGISTRY: dict[str, Unvalidated] = {}
"""Every marker declared at import time, keyed by feature name."""

_TRIGGERED: dict[str, Unvalidated] = {}
_WARNED_BAD_POLICY = False


def _policy() -> str:
    """Read ``JAXRENS_UNVALIDATED``, falling back to ``warn`` on nonsense.

    Read per call (not cached at import) so a test or a driver script can flip
    the policy after jaxrens is already imported.
    """
    global _WARNED_BAD_POLICY
    raw = os.environ.get(_ENV_VAR, "").strip().lower()
    if not raw:
        return "warn"
    if raw in _POLICIES:
        return raw
    if not _WARNED_BAD_POLICY:
        _WARNED_BAD_POLICY = True
        warnings.warn(
            f"jaxrens: ignoring invalid {_ENV_VAR}={raw!r} "
            f"(expected one of {', '.join(_POLICIES)}); using 'warn'.",
            RuntimeWarning,
            stacklevel=3,
        )
    return "warn"


def _message(record: Unvalidated) -> str:
    parts = [
        f"{record.feature} has not been validated in a production run "
        f"(unvalidated since {record.since}).",
        f"Concern: {record.concern}",
    ]
    if record.clears_when:
        parts.append(f"Cleared by: {record.clears_when}")
    parts.append(
        f"Set {_ENV_VAR}=ignore to silence this, or =error to make it fatal."
    )
    return "  ".join(parts)


def warn_unvalidated(
    feature: str,
    *,
    concern: str,
    since: str,
    clears_when: str = "",
    stacklevel: int = 2,
) -> None:
    """Flag *feature* as unvalidated, once per process.

    The call form, for when it is a *branch* rather than a whole function that
    you distrust — usually the honest granularity, since it is rarely an entire
    function you are unsure about::

        if n_species > 2:
            warn_unvalidated(
                "swap: >2 species",
                concern="only binary systems have been run",
                since="0.2.2",
            )

    Registers *feature* on first use, so call-form markers show up in
    :data:`REGISTRY` too (from the first call rather than from import).

    Raises:
        UnvalidatedFeatureWarning: If ``JAXRENS_UNVALIDATED=error``.  Raised on
            every call, not just the first — a marker that was caught and
            swallowed must not go quiet afterwards.
    """
    record = REGISTRY.setdefault(
        feature, Unvalidated(feature, concern, since, clears_when)
    )
    policy = _policy()

    if policy == "error":
        raise UnvalidatedFeatureWarning(_message(record))

    if feature in _TRIGGERED:
        return
    _TRIGGERED[feature] = record

    if policy == "ignore":
        return

    message = _message(record)
    warnings.warn(
        message, UnvalidatedFeatureWarning, stacklevel=stacklevel + 1
    )
    # Also to the logger: the warning goes to stderr, which is nowhere to be
    # found in a batch job, while the logger writes into the run's log file.
    logger.warning("%s", message)


def unvalidated(
    *, concern: str, since: str, clears_when: str = ""
) -> Callable[[F], F]:
    """Decorator marking a function as never run in a production simulation.

    Prefer it on *builders* and other setup-time entry points.  On a function
    that ends up inside ``jax.jit`` the warning fires at trace time only — once
    per compile, at an unpredictable point, and it can be swallowed under
    nested transforms.  Marking ``build_kernel`` instead of the traced ``step``
    it returns gives a deterministic, attributable warning.

    Args:
        concern: What specifically is not trusted about this path.
        since: Version the marker was added.
        clears_when: What run would justify removing it.

    Returns:
        A decorator that leaves the signature and return value untouched, adds
        a Sphinx ``.. warning::`` admonition to the docstring, and exposes the
        :class:`Unvalidated` record as ``__jaxrens_unvalidated__``.
    """

    def deco(fn: F) -> F:
        name = f"{fn.__module__}.{fn.__qualname__}"
        record = Unvalidated(name, concern, since, clears_when)
        REGISTRY[name] = record

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warn_unvalidated(
                name,
                concern=concern,
                since=since,
                clears_when=clears_when,
                # +1 for this wrapper: point at the caller, not at us.
                stacklevel=3,
            )
            return fn(*args, **kwargs)

        # Enumerable off the wrapper as well as via REGISTRY, so a caller
        # holding a function object can ask about it directly.
        wrapper.__jaxrens_unvalidated__ = record  # type: ignore[attr-defined]
        wrapper.__doc__ = _with_admonition(fn.__doc__, record)
        return wrapper  # type: ignore[return-value]

    return deco


def triggered() -> tuple[Unvalidated, ...]:
    """Markers that have actually fired in this process, in trigger order.

    Distinct from :data:`REGISTRY`, which is everything *declared*.  This is
    the set that applies to the run in progress, which is what belongs in the
    output file's metadata.
    """
    return tuple(_TRIGGERED.values())


def format_summary(records: tuple[Unvalidated, ...] | None = None) -> str:
    """Render *records* (default: :func:`triggered`) as one line per feature."""
    if records is None:
        records = triggered()
    return "\n".join(
        f"{r.feature} (since {r.since}): {r.concern}" for r in records
    )


def reset() -> None:
    """Forget which markers have fired, so they warn again.

    For tests: the one-shot suppression is process-global, so without this the
    second test to exercise a marked path sees no warning.
    """
    _TRIGGERED.clear()


def _base_indent(doc: str) -> str:
    """Indentation shared by the docstring's body lines (after the summary)."""
    body = [ln for ln in doc.splitlines()[1:] if ln.strip()]
    if not body:
        return ""
    return min((ln[: len(ln) - len(ln.lstrip())] for ln in body), key=len)


def _with_admonition(doc: str | None, record: Unvalidated) -> str:
    """Append a Sphinx ``.. warning::`` block describing *record* to *doc*."""
    lines = [
        ".. warning::",
        "",
        f"   **Unvalidated** (since {record.since}) — this code path has never",
        "   been exercised in a production simulation.  It may be unit-tested;",
        "   what is missing is a real run whose output was checked.",
        "",
        f"   *Concern:* {record.concern}",
    ]
    if record.clears_when:
        lines += ["", f"   *Cleared by:* {record.clears_when}"]
    block = "\n".join(lines) + "\n"

    if not doc or not doc.strip():
        return block
    return doc.rstrip() + "\n\n" + textwrap.indent(block, _base_indent(doc))
