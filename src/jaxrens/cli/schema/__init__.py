"""Pydantic v2 schemas for jaxrens YAML configuration."""

import jaxrens._jax_init  # noqa: F401 -- pins jax_enable_x64=False before any JAX op
from jaxrens.cli.schema.adaptation import (
    AdaptationPolicy,
    AdaptationSpec,
    ResolvedAdaptationPolicy,
)
from jaxrens.cli.schema.backend import (
    BackendSpec,
    BaseBackendSpec,
    DoubleWellBackendSpec,
    GaussianMixtureBackendSpec,
    HarmonicBackendSpec,
    LJBackendSpec,
    MACEBackendSpec,
    NeuralILBackendSpec,
)
from jaxrens.cli.schema.cell import CellSpec
from jaxrens.cli.schema.ensemble import (
    BaseEnsembleSpec,
    EnsembleSpec,
    NPTEnsembleSpec,
    NVTEnsembleSpec,
    SemiGrandEnsembleSpec,
)
from jaxrens.cli.schema.init import InitialWalkSpec, InitSpec
from jaxrens.cli.schema.moves import (
    AlchemicalMorphMoveSpec,
    BaseMoveSpec,
    GalleryHopMoveSpec,
    GMCMoveSpec,
    HMCMoveSpec,
    LayeredMoveSpec,
    LayerSlideMoveSpec,
    MoveSpec,
    MoveType,
    RandomWalkMoveSpec,
    ShearMoveSpec,
    SingleAtomMoveSpec,
    SingleAtomSweepMoveSpec,
    StackingShiftMoveSpec,
    StretchMoveSpec,
    VolumeMoveSpec,
)
from jaxrens.cli.schema.output import OutputSpec
from jaxrens.cli.schema.root import RootSpec
from jaxrens.cli.schema.run import RunSpec
from jaxrens.cli.schema.termination import (
    BaseTerminationSpec,
    EnergyTerminationSpec,
    IterationTerminationSpec,
    PriorMassTerminationSpec,
    TemperatureTerminationSpec,
    TerminationSpec,
)

__all__ = [
    "AdaptationSpec",
    "AdaptationPolicy",
    "AlchemicalMorphMoveSpec",
    "BackendSpec",
    "BaseBackendSpec",
    "BaseEnsembleSpec",
    "BaseMoveSpec",
    "BaseTerminationSpec",
    "CellSpec",
    "DoubleWellBackendSpec",
    "EnergyTerminationSpec",
    "EnsembleSpec",
    "GMCMoveSpec",
    "GalleryHopMoveSpec",
    "GaussianMixtureBackendSpec",
    "HMCMoveSpec",
    "HarmonicBackendSpec",
    "InitSpec",
    "InitialWalkSpec",
    "IterationTerminationSpec",
    "LJBackendSpec",
    "LayerSlideMoveSpec",
    "LayeredMoveSpec",
    "MACEBackendSpec",
    "MoveSpec",
    "MoveType",
    "NPTEnsembleSpec",
    "NVTEnsembleSpec",
    "SemiGrandEnsembleSpec",
    "NeuralILBackendSpec",
    "OutputSpec",
    "PriorMassTerminationSpec",
    "RandomWalkMoveSpec",
    "ResolvedAdaptationPolicy",
    "RootSpec",
    "RunSpec",
    "ShearMoveSpec",
    "SingleAtomMoveSpec",
    "SingleAtomSweepMoveSpec",
    "StackingShiftMoveSpec",
    "StretchMoveSpec",
    "TemperatureTerminationSpec",
    "TerminationSpec",
    "VolumeMoveSpec",
]
