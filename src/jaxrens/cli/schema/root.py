"""Root pydantic schema combining all sections."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from jaxrens.cli.schema.adaptation import AdaptationConfig
from jaxrens.cli.schema.backend import BackendSpec
from jaxrens.cli.schema.cell import CellConfig
from jaxrens.cli.schema.ensemble import EnsembleSpec, NVTEnsembleSpec, NPTEnsembleSpec
from jaxrens.cli.schema.init import InitConfig
from jaxrens.cli.schema.inter_re import InterREConfigSpec
from jaxrens.cli.schema.moves import MoveSpec
from jaxrens.cli.schema.output import OutputSchema
from jaxrens.cli.schema.run import RunSchema
from jaxrens.cli.schema.termination import TerminationSpec


def _coerce_move_dict(d: object) -> object:
    """Map legacy ``move_type`` key to the discriminator ``type`` field."""
    if isinstance(d, dict) and "type" not in d and "move_type" in d:
        d = dict(d)
        d["type"] = d.pop("move_type")
    return d


def _coerce_backend_dict(d: object) -> object:
    """Map legacy ``backend_type`` key to the discriminator ``type`` field."""
    if isinstance(d, dict) and "type" not in d and "backend_type" in d:
        d = dict(d)
        d["type"] = d.pop("backend_type")
    return d


class RootConfig(BaseModel):
    """Top-level YAML config schema."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run: RunSchema
    moves: list[MoveSpec]
    backend: BackendSpec
    output: OutputSchema
    termination: list[TerminationSpec] | None = None
    adaptation: AdaptationConfig = Field(default_factory=AdaptationConfig)
    # ensemble defaults to NVT; NPTEnsembleSpec is synthesized from run.pressure
    # when no explicit ensemble: key is provided (backward compatibility).
    ensemble: EnsembleSpec = Field(default_factory=NVTEnsembleSpec)
    init: InitConfig = Field(
        default_factory=lambda: InitConfig(start_species="1 1")
    )
    cell: CellConfig = Field(default_factory=CellConfig)
    # inter_re is optional; None → no replica-exchange swaps (zero overhead).
    inter_re: InterREConfigSpec | None = None

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
    def _resolve_legacy_pressure(self) -> "RootConfig":
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
