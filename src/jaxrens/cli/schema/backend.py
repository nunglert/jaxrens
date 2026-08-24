"""Pydantic schema for the [backend] section of a jaxrens YAML config.

Each concrete spec class carries exactly the fields its constructor accepts.
``to_backend_config()`` and ``build_backend()`` are the seam between the CLI
config layer and the library core — they replace the flat backend schema
dispatch that used to live in ``cli/resolve.py``.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from jaxrens.backends.base import EnergyBackend
from jaxrens.state.config import BackendConfig

# ---------------------------------------------------------------------------
# Soft-core wrapper spec
# ---------------------------------------------------------------------------


class SoftCoreSpec(BaseModel):
    """Fixed repulsive Morse soft-core wrapper.

    When present on a backend spec, the resolver and runtime wrap the
    built backend with ``SoftCoreBackend`` (see
    ``jaxrens.backends.softcore``).  Adds a parameter-free, isotropic
    repulsive Morse term to the underlying potential to suppress
    close-contact pathologies during nested sampling.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    a0: float = 1.0
    b0: float = 3.0
    d0: float = 1.0
    r_core_cut: float = 1.25
    r_core_switch: float = 0.75

    @model_validator(mode="after")
    def _check_cutoff_order(self) -> "SoftCoreSpec":
        if self.r_core_switch >= self.r_core_cut:
            raise ValueError(
                f"r_core_switch ({self.r_core_switch}) must be strictly "
                f"less than r_core_cut ({self.r_core_cut})."
            )
        return self


# ---------------------------------------------------------------------------
# Base spec
# ---------------------------------------------------------------------------


class BaseBackendSpec(BaseModel):
    """Fields shared by every backend type."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    periodic: bool = False

    # Optional soft-core repulsion wrapper.  When set, the resolver and
    # runtime wrap the built backend with ``SoftCoreBackend`` (adds a
    # parameter-free repulsive Morse term to the bare potential).  See
    # ``jaxrens.backends.softcore`` for the wrapper and ``SoftCoreSpec``
    # above for the parameters.  Backend-agnostic: works with MACE,
    # Nequix, LJ, NeuralIL, etc.  For NeuralIL, prefer the (slightly
    # cheaper) per-backend ``softcore: true`` flag on ``NeuralILBackendSpec``
    # — the two are mutually exclusive (would otherwise double-count).
    softcore_repulsion: Optional[SoftCoreSpec] = None

    # Overflow-retry ladder.  The outer NS loop picks the smallest entry
    # >= (observed max neighbor count + max_neighbors_offset) as the new
    # bucket size; kernels are JIT-recompiled once per distinct bucket,
    # so keeping this list short bounds the number of recompilations.
    # Ignored by backends that don't do neighbor finding (LJ, toy).
    max_neighbors_list: list[int] = Field(
        default_factory=lambda: [30, 35, 40, 45, 50]
    )
    max_neighbors_offset: int = Field(default=5, ge=0)

    # Hysteresis-gated bucket shrinking (opt-in).  ``shrink_dwell = 0``
    # (default) preserves the pre-existing escalate-only behaviour.  When
    # set > 0, after ``shrink_dwell`` consecutive iterations of
    # ``observed + offset <= next_smaller_entry`` the outer loop steps
    # the bucket one ladder entry down.  Going back to a previously-
    # visited bucket reuses the JAX compilation cache, so the compile
    # budget stays bounded by ``len(max_neighbors_list)``.
    max_neighbors_shrink_dwell: int = Field(default=0, ge=0)

    @field_validator("max_neighbors_list")
    @classmethod
    def _ladder_is_sorted_and_positive(cls, v: list[int]) -> list[int]:
        if len(v) == 0:
            raise ValueError(
                "backend.max_neighbors_list is empty. It is the neighbour-"
                "list capacity ladder the run climbs on overflow, so it needs "
                "at least one entry, e.g. max_neighbors_list: [64, 96, 128]."
            )
        if any(x <= 0 for x in v):
            raise ValueError(
                f"max_neighbors_list entries must be positive, got {v}."
            )
        if any(a >= b for a, b in zip(v, v[1:], strict=False)):
            raise ValueError(
                f"max_neighbors_list must be strictly ascending, got {v}."
            )
        return v

    @property
    def backend_type(self) -> str:
        """Backward-compatible alias for the ``type`` discriminator field."""
        return self.type  # type: ignore[attr-defined]

    def to_backend_config(self) -> BackendConfig:
        """Produce the library ``BackendConfig`` dataclass.

        Subclasses override ``_backend_config_extras`` to inject their
        specific fields (checkpoint_path, cutoff, etc.).  Shared fields
        (periodic, max_neighbors_list/offset) are handled here.
        ``n_atoms`` is derived from the initial walker positions at resolve
        time, not stored on the backend spec.
        """
        softcore_repulsion = (
            self.softcore_repulsion.model_dump()
            if self.softcore_repulsion is not None
            else None
        )
        return BackendConfig(
            backend_type=self.type,  # type: ignore[attr-defined]
            periodic=self.periodic,
            max_neighbors_list=list(self.max_neighbors_list),
            max_neighbors_offset=self.max_neighbors_offset,
            max_neighbors_shrink_dwell=self.max_neighbors_shrink_dwell,
            softcore_repulsion=softcore_repulsion,
            **self._backend_config_extras(),
        )

    def _backend_config_extras(self) -> dict:
        """Subclass hook: fields beyond the BaseBackendSpec common set."""
        return {"checkpoint_path": None, "cutoff": None}

    def build_backend(self) -> EnergyBackend:
        """Construct and return the ``EnergyBackend`` instance.

        Subclasses override to pass their specific constructor kwargs.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Concrete specs — toy backends
# ---------------------------------------------------------------------------


class HarmonicBackendSpec(BaseBackendSpec):
    type: Literal["harmonic"] = "harmonic"
    k: float = 1.0

    def build_backend(self) -> EnergyBackend:
        from jaxrens.backends.toy import create_harmonic

        return create_harmonic(k=self.k)


class DoubleWellBackendSpec(BaseBackendSpec):
    type: Literal["double_well"] = "double_well"
    a: float = 1.0
    b: float = 1.0

    def build_backend(self) -> EnergyBackend:
        from jaxrens.backends.toy import create_double_well

        return create_double_well(a=self.a, b=self.b)


class GaussianMixtureBackendSpec(BaseBackendSpec):
    type: Literal["gaussian_mixture"] = "gaussian_mixture"
    centers: Optional[list[list[float]]] = None
    sigma: float = 0.5

    def build_backend(self) -> EnergyBackend:
        from jaxrens.backends.toy import create_gaussian_mixture

        return create_gaussian_mixture(centers=self.centers, sigma=self.sigma)


# ---------------------------------------------------------------------------
# Concrete specs — production backends
# ---------------------------------------------------------------------------


class LJBackendSpec(BaseBackendSpec):
    type: Literal["lj"] = "lj"
    epsilon: float = 1.0
    sigma: float = 1.0
    cutoff: Optional[float] = None
    # Periodic-image expansion for the all-pairs sum. Must satisfy
    # min(perp_distance · sc) >= 2 · cutoff to capture every neighbor;
    # the resolver emits a startup warning if the cell prior permits
    # cells that would violate this bound.
    supercell_trafo: tuple[int, int, int] = (1, 1, 1)

    def _backend_config_extras(self) -> dict:
        return {"checkpoint_path": None, "cutoff": self.cutoff}

    def build_backend(self) -> EnergyBackend:
        from jaxrens.backends.lj import create_lj

        return create_lj(
            epsilon=self.epsilon,
            sigma=self.sigma,
            cutoff=self.cutoff,
            supercell_trafo=self.supercell_trafo,
        )


class NeuralILBackendSpec(BaseBackendSpec):
    type: Literal["neuralil"] = "neuralil"
    checkpoint_path: str
    # Periodic-image expansion for descriptor generation. Must satisfy
    # ``min(cell_axis_length * sc) >= 2 * r_cut`` (the cutoff is read
    # from the pickle's ``r_cut`` attribute). Defaults to (1,1,1) for
    # backwards compatibility with the no-supercell-needed integration
    # test; bump to (2,2,2) or (3,3,3) for tight unit cells.
    supercell_trafo: tuple[int, int, int] = (1, 1, 1)
    # Soft-core override. ``None`` means "use whatever the pickle says"
    # (read from ``constructor_kwargs['softcore']``); explicit ``True``/
    # ``False`` overrides the pickle. ``softcore_kwargs`` overrides the
    # pickle-stored kwargs (merged on top of package defaults).
    #
    # Mutually exclusive with the inherited ``softcore_repulsion`` wrapper
    # field on ``BaseBackendSpec``: enabling both would add the Morse
    # repulsion twice. Pick one — the per-backend flag here uses the
    # ``SoftCoreNeuralIL`` subclass which shares the descriptor's
    # neighbor discovery (slightly cheaper); the wrapper field is
    # backend-agnostic.
    softcore: Optional[bool] = None
    softcore_kwargs: Optional[dict[str, float]] = None

    @model_validator(mode="after")
    def _check_softcore_mutex(self) -> "NeuralILBackendSpec":
        if self.softcore is True and self.softcore_repulsion is not None:
            raise ValueError(
                "Set either `softcore: true` (NeuralIL-internal path) OR "
                "`softcore_repulsion: {...}` (generic wrapper), not both — "
                "enabling both would add the soft-core Morse term twice."
            )
        return self

    def _backend_config_extras(self) -> dict:
        return {"checkpoint_path": self.checkpoint_path, "cutoff": None}

    def build_backend(self) -> EnergyBackend:
        from jaxrens.backends.neuralil import create_neuralil

        return create_neuralil(
            pickle_file=self.checkpoint_path,
            supercell_trafo=self.supercell_trafo,
            softcore=self.softcore,
            softcore_kwargs=self.softcore_kwargs,
        )


class MACEBackendSpec(BaseBackendSpec):
    type: Literal["mace"] = "mace"
    checkpoint_path: str
    supercell_trafo: tuple[int, int, int] = (2, 2, 2)

    def _backend_config_extras(self) -> dict:
        return {"checkpoint_path": self.checkpoint_path, "cutoff": None}

    def build_backend(self) -> EnergyBackend:
        from jaxrens.backends.mace import create_mace

        return create_mace(
            model_path=self.checkpoint_path,
            supercell_trafo=self.supercell_trafo,
        )


class NequixBackendSpec(BaseBackendSpec):
    type: Literal["nequix"] = "nequix"
    # Either a path to a local ``.nqx`` checkpoint or a bundled-model name
    # (e.g. ``"nequix-mp-1"``, auto-downloaded by the nequix package).
    checkpoint_path: str
    supercell_trafo: tuple[int, int, int] = (1, 1, 1)

    def _backend_config_extras(self) -> dict:
        return {"checkpoint_path": self.checkpoint_path, "cutoff": None}

    def build_backend(self) -> EnergyBackend:
        from jaxrens.backends.nequix import create_nequix

        return create_nequix(
            checkpoint_path=self.checkpoint_path,
            supercell_trafo=self.supercell_trafo,
        )


class JaxMDBackendSpec(BaseBackendSpec):
    """jax-md classical analytic potentials (Tersoff, EAM).

    All-pairs computation — no neighbor list, so ``max_neighbors_list``
    and ``max_neighbors_offset`` from the base spec are ignored.  See
    ``backends/jaxmd.py`` for the architectural rationale.
    """

    type: Literal["jaxmd"] = "jaxmd"
    potential: Literal["tersoff", "eam"]
    tersoff_params: Optional[str] = None
    tersoff_params_file: Optional[str] = None
    eam_params_file: Optional[str] = None

    @field_validator("potential")
    @classmethod
    def _check_potential(cls, v: str) -> str:
        # Pydantic Literal already enforces this, but a clearer error
        # message helps when the discriminator value is right but the
        # ``potential`` field is mis-typed.
        if v not in ("tersoff", "eam"):
            raise ValueError(
                f"`potential` must be 'tersoff' or 'eam', got {v!r}."
            )
        return v

    def _check_params_consistency(self) -> None:
        if self.potential == "tersoff":
            n = sum(
                x is not None
                for x in (self.tersoff_params, self.tersoff_params_file)
            )
            if n != 1:
                raise ValueError(
                    "potential='tersoff' requires exactly one of "
                    "`tersoff_params` (inline name) or "
                    "`tersoff_params_file` (LAMMPS-format path)."
                )
            if self.eam_params_file is not None:
                raise ValueError(
                    "`eam_params_file` must be unset when potential='tersoff'."
                )
        elif self.potential == "eam":
            if self.eam_params_file is None:
                raise ValueError("potential='eam' requires `eam_params_file`.")
            if (
                self.tersoff_params is not None
                or self.tersoff_params_file is not None
            ):
                raise ValueError(
                    "Tersoff fields must be unset when potential='eam'."
                )

    def model_post_init(self, __context: Any) -> None:
        self._check_params_consistency()

    def _backend_config_extras(self) -> dict:
        # Library BackendConfig only carries checkpoint_path / cutoff.
        # jax-md cutoff is derived at build time from the params, so
        # ``cutoff=None`` is right here; the runtime instance owns its
        # own r_cutoff.
        cp = (
            self.tersoff_params
            or self.tersoff_params_file
            or self.eam_params_file
        )
        return {"checkpoint_path": cp, "cutoff": None}

    def build_backend(self) -> EnergyBackend:
        from jaxrens.backends.jaxmd import create_jaxmd

        return create_jaxmd(
            potential=self.potential,
            periodic=self.periodic,
            tersoff_params=self.tersoff_params,
            tersoff_params_file=self.tersoff_params_file,
            eam_params_file=self.eam_params_file,
        )


# ---------------------------------------------------------------------------
# Discriminated union
# ---------------------------------------------------------------------------

BackendSpec = Annotated[
    Union[
        HarmonicBackendSpec,
        DoubleWellBackendSpec,
        GaussianMixtureBackendSpec,
        LJBackendSpec,
        NeuralILBackendSpec,
        MACEBackendSpec,
        NequixBackendSpec,
        JaxMDBackendSpec,
    ],
    Field(discriminator="type"),
]
