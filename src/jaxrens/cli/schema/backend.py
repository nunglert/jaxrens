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

    a0: float = Field(
        default=1.0,
        description="Morse well depth prefactor (energy units).",
    )
    b0: float = Field(
        default=3.0,
        description="Morse decay constant (inverse length units).",
    )
    d0: float = Field(
        default=1.0,
        description="Morse equilibrium distance (Angstrom).",
    )
    r_core_cut: float = Field(
        default=1.25,
        description=(
            "Distance above which the soft-core term is exactly zero "
            "(Angstrom).  Set it *below* the shortest physical bond in "
            "your system — too large and the wall dominates the real "
            "potential, jamming walkers at the cutoff instead of letting "
            "them find the true minimum."
        ),
    )
    r_core_switch: float = Field(
        default=0.75,
        description=(
            "Distance below which the soft-core term is at full strength "
            "(Angstrom); between here and ``r_core_cut`` it is smoothly "
            "switched off.  Must be strictly less than ``r_core_cut``."
        ),
    )

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

    periodic: bool = Field(
        default=False,
        description=(
            "Treat the cell as periodic, applying the minimum-image "
            "convention and periodic-image expansion.  Must be ``true`` "
            "for any condensed-phase run and for every cell move."
        ),
    )

    # Optional soft-core repulsion wrapper.  When set, the resolver and
    # runtime wrap the built backend with ``SoftCoreBackend`` (adds a
    # parameter-free repulsive Morse term to the bare potential).  See
    # ``jaxrens.backends.softcore`` for the wrapper and ``SoftCoreSpec``
    # above for the parameters.  Backend-agnostic: works with MACE,
    # Nequix, LJ, NeuralIL, etc.  For NeuralIL, prefer the (slightly
    # cheaper) per-backend ``softcore: true`` flag on ``NeuralILBackendSpec``
    # — the two are mutually exclusive (would otherwise double-count).
    softcore_repulsion: Optional[SoftCoreSpec] = Field(
        default=None,
        description=(
            "Wrap the backend in a parameter-free repulsive Morse "
            "soft-core term to suppress close-contact pathologies at high "
            "``E_max``.  Backend-agnostic — works with MACE, Nequix, LJ, "
            "NeuralIL.  For NeuralIL prefer the slightly cheaper "
            "per-backend ``softcore: true`` flag; setting both is "
            "rejected, since it would double-count the repulsion."
        ),
    )

    # Overflow-retry ladder.  The outer NS loop picks the smallest entry
    # >= (observed max neighbor count + max_neighbors_offset) as the new
    # bucket size; kernels are JIT-recompiled once per distinct bucket,
    # so keeping this list short bounds the number of recompilations.
    # Ignored by backends that don't do neighbor finding (LJ, toy).
    max_neighbors_list: list[int] = Field(
        default_factory=lambda: [30, 35, 40, 45, 50],
        description=(
            "Ascending ladder of neighbour-list capacities.  On overflow "
            "the outer loop picks the smallest entry >= (observed maximum "
            "+ ``max_neighbors_offset``) and recompiles once per distinct "
            "bucket, so a short ladder bounds the number of "
            "recompilations.  Ignored by backends that do no neighbour "
            "finding (LJ, toy)."
        ),
    )
    max_neighbors_offset: int = Field(
        default=5,
        ge=0,
        description=(
            "Headroom added to the observed maximum neighbour count "
            "before choosing a ladder entry, so a marginal increase does "
            "not immediately force another resize."
        ),
    )

    # Hysteresis-gated bucket shrinking (opt-in).  ``shrink_dwell = 0``
    # (default) preserves the pre-existing escalate-only behaviour.  When
    # set > 0, after ``shrink_dwell`` consecutive iterations of
    # ``observed + offset <= next_smaller_entry`` the outer loop steps
    # the bucket one ladder entry down.  Going back to a previously-
    # visited bucket reuses the JAX compilation cache, so the compile
    # budget stays bounded by ``len(max_neighbors_list)``.
    max_neighbors_shrink_dwell: int = Field(
        default=0,
        ge=0,
        description=(
            "Consecutive iterations that must fit in the next smaller "
            "bucket before the capacity is stepped down.  ``0`` (default) "
            "keeps the escalate-only behaviour.  Shrinking back to an "
            "already-visited bucket reuses the JAX compilation cache, so "
            "the compile budget stays bounded by ``max_neighbors_list``."
        ),
    )

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
    """Toy isotropic harmonic well about the origin.  For tests.

    Has an analytic partition function, which makes it the reference for
    checking that the evidence estimator is unbiased.

    ::

        backend:
          type: harmonic
          k: 1.0
    """

    type: Literal["harmonic"] = Field(
        default="harmonic",
        description="Discriminator selecting this backend.",
    )
    k: float = Field(default=1.0, description="Spring constant.")

    def build_backend(self) -> EnergyBackend:
        from jaxrens.backends.toy import create_harmonic

        return create_harmonic(k=self.k)


class DoubleWellBackendSpec(BaseBackendSpec):
    """Toy double-well potential with a tunable barrier.  For tests.

    Two basins separated by a barrier, so it exercises the sampler's ability
    to cross one — the toy analogue of a first-order transition.

    ::

        backend:
          type: double_well
          a: 1.0
          b: 1.0
    """

    type: Literal["double_well"] = Field(
        default="double_well",
        description="Discriminator selecting this backend.",
    )
    a: float = Field(default=1.0, description="Quartic coefficient.")
    b: float = Field(default=1.0, description="Quadratic coefficient.")

    def build_backend(self) -> EnergyBackend:
        from jaxrens.backends.toy import create_double_well

        return create_double_well(a=self.a, b=self.b)


class GaussianMixtureBackendSpec(BaseBackendSpec):
    """Toy multi-modal Gaussian mixture.  For testing mode-hopping.

    ``centers`` defaults to a built-in set; give your own to control how many
    modes there are and how far apart they sit.

    ::

        backend:
          type: gaussian_mixture
          sigma: 0.5
          centers:
            - [0.0, 0.0, 0.0]
            - [2.0, 0.0, 0.0]
    """

    type: Literal["gaussian_mixture"] = Field(
        default="gaussian_mixture",
        description="Discriminator selecting this backend.",
    )
    centers: Optional[list[list[float]]] = Field(
        default=None,
        description=(
            "Mixture-component centres, one coordinate list each.  "
            "``null`` uses the built-in default set."
        ),
    )
    sigma: float = Field(
        default=0.5, description="Width shared by every component."
    )

    def build_backend(self) -> EnergyBackend:
        from jaxrens.backends.toy import create_gaussian_mixture

        return create_gaussian_mixture(centers=self.centers, sigma=self.sigma)


class RENSToyBackendSpec(BaseBackendSpec):
    """Two particles in a periodic 1-D box — the RENS-paper toy model.

    Multi-modal in ``(a, d)`` despite its simplicity, which makes it the
    cheapest way to see replica exchange actually do something: independent
    runs at different pressures settle into different basins, and the swaps
    are what rescue them.

    Drive it with the ``lattice_1d`` and ``distance_1d`` moves — the general
    3-D cell moves scale isotropically and would break the ``diag(a, 1, 1)``
    cell this backend relies on.  See
    :class:`~jaxrens.backends.toy.RENSToyBackend` for the embedding.

    ::

        backend:
          type: rens_toy
          periodic: true
          mu: 1.0
          sigma: 0.2
          r_cut: 3.0
    """

    type: Literal["rens_toy"] = Field(
        default="rens_toy",
        description="Discriminator selecting this backend.",
    )
    eps_rep: float = Field(
        default=10.0,
        description=(
            "Height of the repulsive core, ``E_rep`` (eq 14).  The core is a "
            "Gaussian and therefore *bounded*: set it too low and the two "
            "particles simply interpenetrate, since the attraction from "
            "their periodic self-images outweighs a finite overlap cost."
        ),
    )
    h_rep: float = Field(
        default=8.0,
        description=(
            "Decay constant of the repulsive core; larger values make the "
            "repulsion narrower."
        ),
    )
    eps_attr: float = Field(
        default=1.0,
        description="Depth of the attractive well, ``E_attr`` (eq 15).",
    )
    mu: float = Field(
        default=1.0,
        description=(
            "Separation at which the attractive well sits.  Sets the "
            "preferred particle spacing, and with it the box lengths that "
            "are commensurate with it."
        ),
    )
    sigma: float = Field(
        default=0.2, description="Width of the attractive well."
    )
    r_cut: float = Field(
        default=3.0,
        description=(
            "Interaction cutoff.  Pairs and periodic images beyond it "
            "contribute nothing."
        ),
    )
    n_images: int = Field(
        default=8,
        ge=1,
        description=(
            "Periodic images summed on each side.  Static (it fixes the "
            "compiled shape), so it only has to satisfy "
            "``n_images * a_min >= r_cut`` for the smallest box the cell "
            "prior permits; the cutoff mask does the physics."
        ),
    )

    def build_backend(self) -> EnergyBackend:
        from jaxrens.backends.toy import create_rens_toy

        return create_rens_toy(
            eps_rep=self.eps_rep,
            h_rep=self.h_rep,
            eps_attr=self.eps_attr,
            mu=self.mu,
            sigma=self.sigma,
            r_cut=self.r_cut,
            n_images=self.n_images,
        )


# ---------------------------------------------------------------------------
# Concrete specs — production backends
# ---------------------------------------------------------------------------


class LJBackendSpec(BaseBackendSpec):
    """Lennard-Jones, computed all-pairs with optional periodic images.

    Set ``periodic: true`` and a ``cutoff`` for condensed-phase work.  Mind
    the interaction between ``cutoff``, the ``cell:`` volume bounds and
    ``supercell_trafo``: the resolver warns at startup when the smallest cell
    the prior permits is too thin to honour the cutoff.

    ::

        backend:
          type: lj
          epsilon: 1.0
          sigma: 1.0
          cutoff: 2.5
          periodic: true
    """

    type: Literal["lj"] = Field(
        default="lj", description="Discriminator selecting this backend."
    )
    epsilon: float = Field(
        default=1.0, description="Well depth, in the run's energy units."
    )
    sigma: float = Field(
        default=1.0, description="Zero-crossing distance of the potential."
    )
    cutoff: Optional[float] = Field(
        default=None,
        description=(
            "Interaction cutoff in the same length units as ``sigma``.  "
            "``null`` means no cutoff (full all-pairs sum)."
        ),
    )
    # Periodic-image expansion for the all-pairs sum. Must satisfy
    # min(perp_distance · sc) >= 2 · cutoff to capture every neighbor;
    # the resolver emits a startup warning if the cell prior permits
    # cells that would violate this bound.
    supercell_trafo: tuple[int, int, int] = Field(
        default=(1, 1, 1),
        description=(
            "Periodic-image expansion for the all-pairs sum.  Must satisfy "
            "``min(perpendicular_distance * sc) >= 2 * cutoff`` to capture "
            "every neighbour; the resolver warns at startup when the cell "
            "prior permits cells that would violate it."
        ),
    )

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
    """NeuralIL machine-learned potential loaded from a pickle.

    ``supercell_trafo`` must be large enough that the cell spans twice the
    model's ``r_cut`` (read from the pickle); raise it for tight unit cells.

    ::

        backend:
          type: neuralil
          checkpoint_path: models/neuralil_si.pkl
          periodic: true
          supercell_trafo: [2, 2, 2]
          softcore: true
    """

    type: Literal["neuralil"] = Field(
        default="neuralil",
        description="Discriminator selecting this backend.",
    )
    checkpoint_path: str = Field(
        description="Path to the trained NeuralIL model pickle.",
    )
    # Periodic-image expansion for descriptor generation. Must satisfy
    # ``min(cell_axis_length * sc) >= 2 * r_cut`` (the cutoff is read
    # from the pickle's ``r_cut`` attribute). Defaults to (1,1,1) for
    # backwards compatibility with the no-supercell-needed integration
    # test; bump to (2,2,2) or (3,3,3) for tight unit cells.
    supercell_trafo: tuple[int, int, int] = Field(
        default=(1, 1, 1),
        description=(
            "Periodic-image expansion for descriptor generation.  Must "
            "satisfy ``min(cell_axis_length * sc) >= 2 * r_cut``, with "
            "``r_cut`` read from the pickle.  Bump to ``[2, 2, 2]`` or "
            "``[3, 3, 3]`` for tight unit cells."
        ),
    )
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
    softcore: Optional[bool] = Field(
        default=None,
        description=(
            "Enable NeuralIL's internal soft-core repulsion, which shares "
            "the descriptor's neighbour discovery and so is slightly "
            "cheaper than the generic ``softcore_repulsion`` wrapper.  "
            "``null`` (default) uses whatever the pickle was trained with; "
            "``true`` / ``false`` overrides it.  Mutually exclusive with "
            "``softcore_repulsion``."
        ),
    )
    softcore_kwargs: Optional[dict[str, float]] = Field(
        default=None,
        description=(
            "Override the pickle-stored soft-core parameters, merged on "
            "top of the package defaults."
        ),
    )

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
    """MACE-JAX message-passing potential.

    Foundation models without a proper short-range repulsive term let walkers
    collapse at high ``E_max``; pair this with ``softcore_repulsion`` (or a
    ``minimum_distance`` constraint) when that happens.

    ::

        backend:
          type: mace
          checkpoint_path: models/mace_si.model
          periodic: true
          supercell_trafo: [2, 2, 2]
          softcore_repulsion:
            r_core_cut: 1.25
            r_core_switch: 0.75
    """

    type: Literal["mace"] = Field(
        default="mace", description="Discriminator selecting this backend."
    )
    checkpoint_path: str = Field(
        description=(
            "Path to the MACE model file.  See the MACE models page for "
            "how to convert a torch checkpoint into a JAX-loadable bundle."
        ),
    )
    supercell_trafo: tuple[int, int, int] = Field(
        default=(2, 2, 2),
        description=(
            "Periodic-image expansion for neighbour finding.  Must be "
            "large enough that the cell spans twice the model's receptive "
            "field; the default suits typical small unit cells."
        ),
    )

    def _backend_config_extras(self) -> dict:
        return {"checkpoint_path": self.checkpoint_path, "cutoff": None}

    def build_backend(self) -> EnergyBackend:
        from jaxrens.backends.mace import create_mace

        return create_mace(
            model_path=self.checkpoint_path,
            supercell_trafo=self.supercell_trafo,
        )


class NequixBackendSpec(BaseBackendSpec):
    """Nequix message-passing potential.

    ``checkpoint_path`` takes either a local ``.nqx`` file or a bundled model
    name, which the ``nequix`` package downloads on first use.

    ::

        backend:
          type: nequix
          checkpoint_path: nequix-mp-1
          periodic: true
    """

    type: Literal["nequix"] = Field(
        default="nequix",
        description="Discriminator selecting this backend.",
    )
    checkpoint_path: str = Field(
        description=(
            "Either a path to a local ``.nqx`` checkpoint, or a bundled "
            "model name such as ``nequix-mp-1``, which the ``nequix`` "
            "package downloads on first use."
        ),
    )
    supercell_trafo: tuple[int, int, int] = Field(
        default=(1, 1, 1),
        description=(
            "Periodic-image expansion for neighbour finding; raise it for "
            "cells smaller than twice the model cutoff."
        ),
    )

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

    Tersoff needs exactly one of ``tersoff_params`` (a built-in set) or
    ``tersoff_params_file`` (a LAMMPS-format file); EAM needs
    ``eam_params_file`` and no Tersoff key.  Mixing them is rejected at
    parse time.

    ::

        backend:
          type: jaxmd
          potential: tersoff
          tersoff_params: Si
          periodic: true

    ::

        backend:
          type: jaxmd
          potential: eam
          eam_params_file: potentials/Cu_u3.eam
          periodic: true
    """

    type: Literal["jaxmd"] = Field(
        default="jaxmd",
        description="Discriminator selecting this backend.",
    )
    potential: Literal["tersoff", "eam"] = Field(
        description=(
            "Which analytic potential to build.  ``tersoff`` needs exactly "
            "one of ``tersoff_params`` / ``tersoff_params_file``; ``eam`` "
            "needs ``eam_params_file``."
        ),
    )
    tersoff_params: Optional[str] = Field(
        default=None,
        description=(
            "Name of a built-in Tersoff parameter set.  Mutually "
            "exclusive with ``tersoff_params_file``."
        ),
    )
    tersoff_params_file: Optional[str] = Field(
        default=None,
        description=(
            "Path to a LAMMPS-format Tersoff parameter file.  Mutually "
            "exclusive with ``tersoff_params``."
        ),
    )
    eam_params_file: Optional[str] = Field(
        default=None,
        description=(
            "Path to an EAM parameter file.  Required when "
            "``potential: eam``, and must be unset otherwise."
        ),
    )

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
        RENSToyBackendSpec,
        LJBackendSpec,
        NeuralILBackendSpec,
        MACEBackendSpec,
        NequixBackendSpec,
        JaxMDBackendSpec,
    ],
    Field(discriminator="type"),
]
