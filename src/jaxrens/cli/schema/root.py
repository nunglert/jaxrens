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
from jaxrens.cli.schema.ensemble import EnsembleSpec, NVTEnsembleSpec
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
    run: RunSpec = Field(
        description=(
            "Sampler sizing and reproducibility: walker count, MCMC steps "
            "per replacement, seed."
        ),
    )
    moves: list[MoveSpec] = Field(
        description=(
            "Ordered list of MCMC move kernels composed by the "
            "Metropolis-within-Gibbs scheduler.  Each entry's ``type:`` "
            "picks the kernel and its ``weight`` sets the dispatch "
            "probability.  A single mapping is accepted and wrapped in a "
            "list."
        ),
    )
    backend: BackendSpec = Field(
        description=(
            "The energy model.  ``type:`` selects one of the built-in "
            "potentials or machine-learned backends."
        ),
    )
    output: OutputSpec = Field(
        description=(
            "Where results are written, at what cadence, and which "
            "optional diagnostic logs are enabled."
        ),
    )
    termination: list[TerminationSpec] | None = Field(
        default=None,
        description=(
            "Stopping criteria; the run ends when **any** of them fires.  "
            "A single mapping is wrapped in a list.  ``null`` falls back "
            "to ``run.max_iterations`` and "
            "``run.convergence_threshold``."
        ),
    )
    adaptation: AdaptationSpec = Field(
        default_factory=AdaptationSpec,
        description=(
            "Step-size adaptation policy, with per-move overrides.  "
            "Adaptation is on by default; set ``adaptation.adjust_interval: "
            "0`` to freeze every move at its configured ``step_size``."
        ),
    )
    ensemble: EnsembleSpec = Field(
        default_factory=NVTEnsembleSpec,
        description=(
            "Thermodynamic ensemble.  Defaults to NVT; ``type: npt`` adds "
            "``P*V`` and ``type: semi_grand`` adds ``-mu*N``.  A "
            "list-valued driving parameter here is what fans a run out "
            "across replicas."
        ),
    )
    init: InitSpec = Field(
        default_factory=lambda: InitSpec(start_species="1 1"),
        description=(
            "Where the atoms come from and how the initial population is "
            "randomised.  Exactly one source-of-atoms key must be set."
        ),
    )
    cell: CellSpec = Field(
        default_factory=CellSpec,
        description=(
            "Cell-geometry bounds.  The single source of truth for the "
            "volume, shear, and stretch kernels — the move specs "
            "themselves carry no copies of these."
        ),
    )
    inter_re: InterRESpec | None = Field(
        default=None,
        description=(
            "Inter-replica exchange (RENS).  Omit to disable swaps "
            "entirely."
        ),
    )
    constraints: list[ConstraintSpec] = Field(
        default_factory=list,
        description=(
            "Hard configuration constraints applied throughout sampling.  "
            "A constraint rejects any proposal moving a walker into a "
            "forbidden region, exactly like the likelihood threshold, and "
            "is enforced only on the moves that can violate it.  Omitting "
            "the key means no constraints and zero overhead."
        ),
    )

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

    @model_validator(mode="after")
    def _warn_n_cull_postprocessing_unvalidated(self) -> "RootSpec":
        """Flag ``run.n_cull > 1`` as unvalidated on the postprocessing path.

        The sampler itself culls ``n_cull`` walkers per iteration correctly
        for any value.  But ``Monitor.from_directory`` — the loader every
        ``jaxrens analyze``/``jaxrens plot`` observable goes through — and
        ``postprocess.collection`` both reconstruct a run's ``Monitor`` with
        ``n_cull`` hardcoded to ``1``, not read back from the config or the
        checkpoint.  For ``n_cull == 1`` (the default) that hardcoding is
        correct by construction; for ``n_cull > 1`` it silently mismatches
        the run that actually produced the data, biasing every downstream
        prior-mass weight, log Z, and thermodynamic observable — no
        production run has exercised that combination, hence
        :func:`~jaxrens.unvalidated.warn_unvalidated` rather than a plain
        warning: it is tracked in the same registry as everything else this
        codebase doesn't yet trust, controllable via ``JAXRENS_UNVALIDATED``,
        and (once a run actually starts) stamped into the output file's
        metadata rather than only flashing past on stderr.
        """
        if self.run.n_cull > 1:
            from jaxrens.unvalidated import warn_unvalidated

            warn_unvalidated(
                "run.n_cull > 1",
                concern=(
                    f"n_cull={self.run.n_cull}: the sampler culls "
                    f"{self.run.n_cull} walkers per iteration correctly, "
                    "but Monitor.from_directory (used by `jaxrens analyze` "
                    "and `jaxrens plot`) hardcodes n_cull=1 when "
                    "reconstructing prior-mass weights from disk, so "
                    "downstream log Z / heat-capacity / free-energy "
                    "estimates from the CLI will be wrong for this run "
                    "unless you call the lower-level "
                    "postprocess.thermodynamics functions yourself with "
                    "the correct n_cull."
                ),
                since="0.3.1",
                clears_when=(
                    "Monitor.from_directory reads n_cull from the "
                    "checkpoint/config instead of hardcoding 1, and a "
                    "production run with n_cull > 1 has had its "
                    "postprocessed observables checked against the raw "
                    "dead-point ladder."
                ),
                stacklevel=2,
            )
        return self
