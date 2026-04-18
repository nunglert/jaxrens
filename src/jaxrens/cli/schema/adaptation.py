"""Pydantic schema for the [adaptation] section of a jaxrens YAML config.

``AdaptationConfig`` holds the step-size adaptation policy with a
``defaults + per_move`` overlay.  ``resolve_for(key)`` returns the effective
policy for a named move: per-move fields that are None fall through to
defaults, and defaults fields that are None fall through to the hardcoded
library values that ``MoveKernel`` and ``adjust_step_size`` use today.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


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

    min_rate: float | None = None
    max_rate: float | None = None
    adjust_factor: float | None = None
    step_size_max: float | None = None

    def filled(
        self,
        fallback: "AdaptationPolicy | None" = None,
    ) -> "ResolvedAdaptationPolicy":
        """Return a fully populated policy with no None fields.

        Resolution order: self → fallback → hardcoded library defaults.
        """
        base = fallback or AdaptationPolicy()
        return ResolvedAdaptationPolicy(
            min_rate=self.min_rate if self.min_rate is not None
            else (base.min_rate if base.min_rate is not None else _FALLBACK_MIN_RATE),
            max_rate=self.max_rate if self.max_rate is not None
            else (base.max_rate if base.max_rate is not None else _FALLBACK_MAX_RATE),
            adjust_factor=self.adjust_factor if self.adjust_factor is not None
            else (base.adjust_factor if base.adjust_factor is not None else _FALLBACK_ADJUST_FACTOR),
            step_size_max=self.step_size_max if self.step_size_max is not None
            else (base.step_size_max if base.step_size_max is not None else _FALLBACK_STEP_SIZE_MAX),
        )


class ResolvedAdaptationPolicy(BaseModel):
    """Fully resolved adaptation policy — no None fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    min_rate: float
    max_rate: float
    adjust_factor: float
    step_size_max: float


# ---------------------------------------------------------------------------
# AdaptationConfig
# ---------------------------------------------------------------------------

class AdaptationConfig(BaseModel):
    """Top-level adaptation config with defaults and per-move overrides."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    full_auto: bool = False
    full_auto_steps: int = 0
    adjust_n_samples: int = 50
    adjust_max_rounds: int = 15
    defaults: AdaptationPolicy = Field(default_factory=AdaptationPolicy)
    per_move: dict[str, AdaptationPolicy] = Field(default_factory=dict)

    def resolve_for(self, move_name_or_type: str) -> ResolvedAdaptationPolicy:
        """Return the effective policy for a named move.

        Fields from ``per_move[move_name_or_type]`` override ``defaults``;
        remaining None fields fall through to hardcoded library fallbacks.
        """
        override = self.per_move.get(move_name_or_type)
        if override is not None:
            return override.filled(fallback=self.defaults)
        return self.defaults.filled()
