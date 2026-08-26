"""Pydantic schema for the [adaptation] section of a jaxrens YAML config.

``AdaptationSpec`` holds the step-size adaptation policy with a
``defaults + per_move`` overlay.  ``resolve_for(key)`` returns the effective
policy for a named move: per-move fields that are None fall through to
defaults, and defaults fields that are None fall through to the hardcoded
library values that ``MoveKernel`` and ``build_adapt_step`` use today.
"""

from __future__ import annotations

import warnings

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# Hardcoded fallbacks — match MoveKernel defaults and run_ns defaults
# ---------------------------------------------------------------------------

_FALLBACK_MIN_RATE: float = 0.25
_FALLBACK_MAX_RATE: float = 0.65
_FALLBACK_ADJUST_FACTOR: float = 1.5
_FALLBACK_STEP_SIZE_MAX: float = 10.0


# ---------------------------------------------------------------------------
# AdaptationPolicy
# ---------------------------------------------------------------------------


class AdaptationPolicy(BaseModel):
    """Per-move adaptation knobs.  All fields optional for overlay semantics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    min_rate: float | None = Field(
        default=None,
        description=(
            "Acceptance rate below which the step size is shrunk.  "
            "``null`` inherits from ``defaults``, then from the library "
            "fallback (0.25)."
        ),
    )
    max_rate: float | None = Field(
        default=None,
        description=(
            "Acceptance rate above which the step size is grown.  "
            "``null`` inherits from ``defaults``, then from the library "
            "fallback (0.65)."
        ),
    )
    adjust_factor: float | None = Field(
        default=None,
        description=(
            "Multiplicative factor applied to the step size on each "
            "adjustment (grow by it, shrink by its inverse).  ``null`` "
            "inherits from ``defaults``, then from the library fallback "
            "(1.5)."
        ),
    )
    step_size_max: float | None = Field(
        default=None,
        description=(
            "Ceiling on the adapted step size, in the move's own units.  "
            "``null`` inherits from ``defaults``, then from the library "
            "fallback (10.0)."
        ),
    )

    def filled(
        self,
        fallback: "AdaptationPolicy | None" = None,
    ) -> "ResolvedAdaptationPolicy":
        """Return a fully populated policy with no None fields.

        Resolution order: self → fallback → hardcoded library defaults.
        """
        base = fallback or AdaptationPolicy()
        return ResolvedAdaptationPolicy(
            min_rate=self.min_rate
            if self.min_rate is not None
            else (
                base.min_rate
                if base.min_rate is not None
                else _FALLBACK_MIN_RATE
            ),
            max_rate=self.max_rate
            if self.max_rate is not None
            else (
                base.max_rate
                if base.max_rate is not None
                else _FALLBACK_MAX_RATE
            ),
            adjust_factor=self.adjust_factor
            if self.adjust_factor is not None
            else (
                base.adjust_factor
                if base.adjust_factor is not None
                else _FALLBACK_ADJUST_FACTOR
            ),
            step_size_max=self.step_size_max
            if self.step_size_max is not None
            else (
                base.step_size_max
                if base.step_size_max is not None
                else _FALLBACK_STEP_SIZE_MAX
            ),
        )


class ResolvedAdaptationPolicy(BaseModel):
    """Fully resolved adaptation policy — no None fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    min_rate: float = Field(
        description="Acceptance rate below which the step size shrinks."
    )
    max_rate: float = Field(
        description="Acceptance rate above which the step size grows."
    )
    adjust_factor: float = Field(
        description="Multiplicative step-size adjustment factor."
    )
    step_size_max: float = Field(description="Ceiling on the step size.")


# ---------------------------------------------------------------------------
# AdaptationSpec
# ---------------------------------------------------------------------------


class AdaptationSpec(BaseModel):
    """Top-level adaptation config with defaults and per-move overrides.

    ``adjust_interval`` is the iteration cadence at which step-size adaptation
    fires.  ``0`` disables adaptation (the manager's ``is_active`` guard).
    The legacy YAML key ``full_auto_steps`` is accepted with a
    ``DeprecationWarning`` and remapped onto ``adjust_interval`` at parse time.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    full_auto: bool = Field(
        default=True,
        description=(
            "Use the bisection step-size search instead of the cheap "
            "grow/shrink rule.  Each adaptation event runs up to "
            "``adjust_max_rounds`` trial batches of ``adjust_n_samples`` "
            "walkers to land the acceptance rate inside "
            "``[min_rate, max_rate]``.  More accurate, markedly more "
            "expensive per event."
        ),
    )
    adjust_interval: int | float = Field(
        default=100,
        description=(
            "Iteration cadence at which step-size adaptation fires.  "
            "Set ``0`` to disable adaptation entirely, freezing every move "
            "at its configured ``step_size``.  Honours "
            "``interval_units`` — in ``per_walker`` mode use a value "
            "<= 1 so adaptation runs at least once per walker-sweep."
        ),
    )
    adjust_n_samples: int = Field(
        default=50,
        description=(
            "Walkers sampled per trial round when ``full_auto`` is set.  "
            "Larger values give a less noisy acceptance estimate per "
            "round at linear cost."
        ),
    )
    adjust_max_rounds: int = Field(
        default=15,
        description=(
            "Maximum bisection rounds per adaptation event when "
            "``full_auto`` is set.  Bounds the worst-case cost of a "
            "single event."
        ),
    )
    trial_batch_size: int | None = Field(
        default=8,
        description=(
            "Chunk size for the trial-batch vmap during ``full_auto`` "
            "adaptation, for memory control.  ``null`` vmaps over all "
            "trial walkers at once (fastest, but largest memory footprint)."
        ),
    )
    defaults: AdaptationPolicy = Field(
        default_factory=AdaptationPolicy,
        description=(
            "Baseline adaptation policy applied to every move that has no "
            "``per_move`` override."
        ),
    )
    per_move: dict[str, AdaptationPolicy] = Field(
        default_factory=dict,
        description=(
            "Per-move policy overrides, keyed by move name (the move's "
            "``name:`` if set, otherwise its ``type:``).  Fields left "
            "unset fall through to ``defaults``, then to the library "
            "fallbacks."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_full_auto_steps(cls, v: object) -> object:
        """Map legacy ``full_auto_steps`` → ``adjust_interval`` with a warning."""
        if (
            isinstance(v, dict)
            and "full_auto_steps" in v
            and "adjust_interval" not in v
        ):
            warnings.warn(
                "adaptation.full_auto_steps is deprecated; rename to "
                "adaptation.adjust_interval.",
                DeprecationWarning,
                stacklevel=2,
            )
            v = dict(v)
            v["adjust_interval"] = v.pop("full_auto_steps")
        return v

    def resolve_for(self, move_name_or_type: str) -> ResolvedAdaptationPolicy:
        """Return the effective policy for a named move.

        Fields from ``per_move[move_name_or_type]`` override ``defaults``;
        remaining None fields fall through to hardcoded library fallbacks.
        """
        override = self.per_move.get(move_name_or_type)
        if override is not None:
            return override.filled(fallback=self.defaults)
        return self.defaults.filled()
