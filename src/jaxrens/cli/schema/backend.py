"""Pydantic schema for the [backend] section of a jaxrens YAML config.

Each concrete spec class carries exactly the fields its constructor accepts.
``to_backend_config()`` and ``build_backend()`` are the seam between the CLI
config layer and the library core — they replace the flat backend schema
dispatch that used to live in ``cli/resolve.py``.
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from jaxrens.backends.base import EnergyBackend
from jaxrens.state.config import BackendConfig


# ---------------------------------------------------------------------------
# Base spec
# ---------------------------------------------------------------------------

class BaseBackendSpec(BaseModel):
    """Fields shared by every backend type."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    periodic: bool = False

    # Overflow-retry ladder.  The outer NS loop picks the smallest entry
    # >= (observed max neighbor count + max_neighbors_offset) as the new
    # bucket size; kernels are JIT-recompiled once per distinct bucket,
    # so keeping this list short bounds the number of recompilations.
    # Ignored by backends that don't do neighbor finding (LJ, toy).
    max_neighbors_list: list[int] = Field(
        default_factory=lambda: [30, 35, 40, 45, 50]
    )
    max_neighbors_offset: int = Field(default=5, ge=0)

    @field_validator("max_neighbors_list")
    @classmethod
    def _ladder_is_sorted_and_positive(cls, v: list[int]) -> list[int]:
        if len(v) == 0:
            raise ValueError("max_neighbors_list must be non-empty.")
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
        return BackendConfig(
            backend_type=self.type,  # type: ignore[attr-defined]
            periodic=self.periodic,
            max_neighbors_list=list(self.max_neighbors_list),
            max_neighbors_offset=self.max_neighbors_offset,
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

    def _backend_config_extras(self) -> dict:
        return {"checkpoint_path": None, "cutoff": self.cutoff}

    def build_backend(self) -> EnergyBackend:
        from jaxrens.backends.lj import create_lj
        return create_lj(epsilon=self.epsilon, sigma=self.sigma, cutoff=self.cutoff)


class NeuralILBackendSpec(BaseBackendSpec):
    type: Literal["neuralil"] = "neuralil"
    checkpoint_path: str

    def _backend_config_extras(self) -> dict:
        return {"checkpoint_path": self.checkpoint_path, "cutoff": None}

    def build_backend(self) -> EnergyBackend:
        from jaxrens.backends.neuralil import create_neuralil
        return create_neuralil(pickle_file=self.checkpoint_path)


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
    ],
    Field(discriminator="type"),
]
