"""Pydantic schema for the [backend] section of a jaxrens YAML config.

Each concrete spec class carries exactly the fields its constructor accepts.
``to_backend_config()`` and ``build_backend()`` are the seam between the CLI
config layer and the library core — they replace the flat backend schema
dispatch that used to live in ``cli/resolve.py``.
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from jaxrens.backends.base import EnergyBackend
from jaxrens.state.config import BackendConfig


# ---------------------------------------------------------------------------
# Base spec
# ---------------------------------------------------------------------------

class BaseBackendSpec(BaseModel):
    """Fields shared by every backend type."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    periodic: bool = False

    @property
    def backend_type(self) -> str:
        """Backward-compatible alias for the ``type`` discriminator field."""
        return self.type  # type: ignore[attr-defined]

    def to_backend_config(self) -> BackendConfig:
        """Produce the library ``BackendConfig`` dataclass.

        Subclasses that carry NeuralIL-specific fields override this method.
        All other backends use ``BackendConfig`` defaults for those fields.
        ``n_atoms`` is derived from the initial walker positions at resolve
        time, not stored on the backend spec.
        """
        return BackendConfig(
            backend_type=self.type,  # type: ignore[attr-defined]
            checkpoint_path=None,
            periodic=self.periodic,
            cutoff=None,
        )

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

    def to_backend_config(self) -> BackendConfig:
        return BackendConfig(
            backend_type="lj",
            checkpoint_path=None,
            periodic=self.periodic,
            cutoff=self.cutoff,
        )

    def build_backend(self) -> EnergyBackend:
        from jaxrens.backends.lj import create_lj
        return create_lj(epsilon=self.epsilon, sigma=self.sigma, cutoff=self.cutoff)


class NeuralILBackendSpec(BaseBackendSpec):
    type: Literal["neuralil"] = "neuralil"
    checkpoint_path: str
    max_neighbors_list: list[int] = Field(default_factory=lambda: [30, 35, 40, 45, 50])
    max_neighbors_offset: int = 5

    def to_backend_config(self) -> BackendConfig:
        return BackendConfig(
            backend_type="neuralil",
            checkpoint_path=self.checkpoint_path,
            periodic=self.periodic,
            cutoff=None,
            max_neighbors_list=list(self.max_neighbors_list),
            max_neighbors_offset=self.max_neighbors_offset,
        )

    def build_backend(self) -> EnergyBackend:
        from jaxrens.backends.neuralil import create_neuralil
        return create_neuralil(pickle_file=self.checkpoint_path)


class MACEBackendSpec(BaseBackendSpec):
    type: Literal["mace"] = "mace"
    checkpoint_path: str
    dtype: str = "float64"
    supercell_trafo: tuple[int, int, int] = (2, 2, 2)

    def to_backend_config(self) -> BackendConfig:
        return BackendConfig(
            backend_type="mace",
            checkpoint_path=self.checkpoint_path,
            periodic=self.periodic,
            cutoff=None,
        )

    def build_backend(self) -> EnergyBackend:
        from jaxrens.backends.mace import create_mace
        return create_mace(
            model_path=self.checkpoint_path,
            dtype=self.dtype,
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
