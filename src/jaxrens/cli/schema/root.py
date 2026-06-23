"""Root pydantic schema combining all sections."""

from __future__ import annotations

import warnings
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from jaxrens.cli.schema.adaptation import AdaptationSpec
from jaxrens.cli.schema.backend import BackendSpec
from jaxrens.cli.schema.cell import CellSpec
from jaxrens.cli.schema.constraints import ConstraintSpec
from jaxrens.cli.schema.ensemble import (
    EnsembleSpec,
    NPTEnsembleSpec,
    NVTEnsembleSpec,
)
from jaxrens.cli.schema.init import InitSpec
from jaxrens.cli.schema.inter_re import InterRESpec
from jaxrens.cli.schema.moves import MoveSpec
from jaxrens.cli.schema.output import OutputSpec
from jaxrens.cli.schema.run import RunSpec
from jaxrens.cli.schema.termination import TerminationSpec


def _coerce_move_dict(d: object) -> object:
    """Map legacy ``move_type`` key and the ``type: galilean`` alias.

    - ``move_type:`` → ``type:`` (legacy key name).
    - ``type: galilean`` → ``type: gmc`` (canonical name; ``GMCMoveSpec`` is
      the only Galilean-MC class).
    """
    if isinstance(d, dict) and "type" not in d and "move_type" in d:
        d = dict(d)
        d["type"] = d.pop("move_type")
    if isinstance(d, dict) and d.get("type") == "galilean":
        d = dict(d)
        d["type"] = "gmc"
    return d


def _coerce_backend_dict(d: object) -> object:
    """Map legacy ``backend_type`` key to the discriminator ``type`` field."""
    if isinstance(d, dict) and "type" not in d and "backend_type" in d:
        d = dict(d)
        d["type"] = d.pop("backend_type")
    return d


# Per-walker interval sanity thresholds (walker-sweeps). In ``per_walker`` mode
# an interval is a number of sweeps, so values far outside the usual range are
# almost always a mistake — typically a raw iteration count left over from
# ``absolute`` mode. Adaptation slower than once per sweep means step sizes
# barely adapt; replica-exchange / trajectory output many times per sweep is
# pure overhead. These are advisory warnings only — both ends remain legal.
_ADAPT_INTERVAL_MAX_SWEEPS = 1.0
_OUTPUT_INTERVAL_MIN_SWEEPS = 0.05


class RootSpec(BaseModel):
    """Top-level YAML config schema."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    interval_units: Literal["absolute", "per_walker"] = Field(
        default="absolute",
        description=(
            "How the resolver interprets every iteration-counted field. "
            "``absolute`` (default): values are raw NS iteration counts. "
            "``per_walker``: values are walker-sweeps, where one sweep equals "
            "``run.n_live`` iterations; the resolver multiplies each affected "
            "field by ``n_live`` before building the runtime dataclasses, so "
            "an interval expressed in sweeps stays comparable across configs "
            "with different ``n_live``. Affected fields: "
            "``output.{info,traj,snapshot,checkpoint,flush}_interval``, "
            "``output.{temperature_lag,temperature,acc_rates,max_neighbors,"
            "collision_check}_interval``, ``run.max_iterations``, "
            "``termination[iteration].max_iterations``, "
            "``inter_re.re_interval``, and ``adaptation.adjust_interval``. "
            "Scaled values are rounded to the nearest int and clamped to "
            ">= 1, so a fractional sweep like ``0.001`` never collapses to 0."
        ),
    )
    run: RunSpec
    moves: list[MoveSpec]
    backend: BackendSpec
    output: OutputSpec
    termination: list[TerminationSpec] | None = None
    adaptation: AdaptationSpec = Field(default_factory=AdaptationSpec)
    # ensemble defaults to NVT; NPTEnsembleSpec is synthesized from run.pressure
    # when no explicit ensemble: key is provided (backward compatibility).
    ensemble: EnsembleSpec = Field(default_factory=NVTEnsembleSpec)
    init: InitSpec = Field(
        default_factory=lambda: InitSpec(start_species="1 1")
    )
    cell: CellSpec = Field(default_factory=CellSpec)
    # inter_re is optional; None → no replica-exchange swaps (zero overhead).
    inter_re: InterRESpec | None = None
    # Configuration constraints (e.g. minimum inter-atomic distance). Empty by
    # default → no constraint gating, zero overhead. See jaxrens.constraints.
    constraints: list[ConstraintSpec] = Field(default_factory=list)

    @field_validator("moves", mode="before")
    @classmethod
    def _normalize_moves(cls, v: object) -> list[object]:
        """Accept a single mapping and wrap it in a list.

        Also rewrites legacy ``move_type`` key to the discriminator ``type``.
        """
        if isinstance(v, dict):
            v = [v]
        if isinstance(v, list):
            return [_coerce_move_dict(item) for item in v]
        return v  # type: ignore[return-value]

    @field_validator("backend", mode="before")
    @classmethod
    def _normalize_backend(cls, v: object) -> object:
        """Rewrite legacy ``backend_type`` key to the discriminator ``type``."""
        return _coerce_backend_dict(v)

    @field_validator("termination", mode="before")
    @classmethod
    def _normalize_termination(cls, v: object) -> object:
        """Accept a single termination dict and wrap it in a list."""
        if isinstance(v, dict):
            return [v]
        return v

    @model_validator(mode="after")
    def _resolve_legacy_pressure(self) -> "RootSpec":
        """Synthesize NPTEnsembleSpec from ``run.pressure`` when no ``ensemble:`` key.

        Rules:
        - If ``run.pressure`` is set AND no explicit ``ensemble:`` was provided
          (i.e. ensemble is still the NVT default), synthesize NPTEnsembleSpec.
        - If both ``run.pressure`` is set AND ``ensemble:`` was explicitly provided,
          raise a ValidationError — the user must pick one.
        - If neither is set, NVT is used (default behavior, no change).

        Note: ``run.pressure`` is deprecated in favour of ``ensemble:``.
        Use ``ensemble: {type: npt, pressure: <value>}`` in new configs.
        """
        has_legacy_pressure = self.run.pressure is not None
        # NVTEnsembleSpec is the default_factory default — it's what we see when
        # the user did NOT provide an explicit ensemble key.
        ensemble_is_default = isinstance(self.ensemble, NVTEnsembleSpec)

        if has_legacy_pressure and not ensemble_is_default:
            raise ValueError(
                "Conflicting ensemble specification: both run.pressure and ensemble: "
                "are set. Remove run.pressure and use ensemble: exclusively."
            )

        if has_legacy_pressure and ensemble_is_default:
            # Synthesize NPT from legacy field; pressure is already in eV/Å³ from
            # the old convention (NSConfig.pressure stores eV/Å³).
            object.__setattr__(
                self,
                "ensemble",
                NPTEnsembleSpec(
                    pressure=self.run.pressure,  # type: ignore[arg-type]
                    pressure_units="eva3",
                ),
            )

        return self

    @model_validator(mode="after")
    def _warn_unusual_per_walker_intervals(self) -> "RootSpec":
        """Warn about ``per_walker`` intervals that are almost always mistakes.

        Only meaningful when ``interval_units == "per_walker"`` (values are
        walker-sweeps). Catches two common footguns — usually a raw iteration
        count accidentally left in a ``per_walker`` config:

        - ``adaptation.adjust_interval > 1`` — step-size adaptation runs less
          than once per sweep, so step sizes effectively never adapt;
        - ``inter_re.re_interval`` / ``output.traj_interval < 0.05`` — replica
          exchange / trajectory writing fires >~20x per sweep, which is huge
          overhead (and output volume) for no benefit.

        Advisory only: the values remain valid, this just surfaces the likely
        unintended behaviour at config-load time.
        """
        if self.interval_units != "per_walker":
            return self

        adjust_interval = self.adaptation.adjust_interval
        if adjust_interval > _ADAPT_INTERVAL_MAX_SWEEPS:
            warnings.warn(
                f"adaptation.adjust_interval={adjust_interval} with "
                "interval_units=per_walker means step-size adaptation runs only "
                f"every {adjust_interval} walker-sweeps — almost never. Use a "
                "value <= 1 to adapt at least once per sweep, or set "
                "interval_units: absolute if you meant raw iterations.",
                UserWarning,
                stacklevel=2,
            )

        too_frequent = [
            (
                "inter_re.re_interval",
                self.inter_re.re_interval if self.inter_re else None,
            ),
            ("output.traj_interval", self.output.traj_interval),
        ]
        for label, value in too_frequent:
            if value is not None and 0 < value < _OUTPUT_INTERVAL_MIN_SWEEPS:
                warnings.warn(
                    f"{label}={value} with interval_units=per_walker fires "
                    f"~{1 / value:.0f}x per walker-sweep, which is rarely "
                    "intended and very expensive. Values below "
                    f"{_OUTPUT_INTERVAL_MIN_SWEEPS} are unusual; did you mean a "
                    "larger value, or interval_units: absolute?",
                    UserWarning,
                    stacklevel=2,
                )

        return self
