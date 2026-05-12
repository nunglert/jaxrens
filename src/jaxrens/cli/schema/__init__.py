"""Pydantic v2 schemas for jaxrens YAML configuration."""

from jaxrens.cli.schema.adaptation import (
    AdaptationSpec,
    AdaptationPolicy,
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
)
from jaxrens.cli.schema.init import InitSpec, InitialWalkSpec
from jaxrens.cli.schema.moves import (
    AlchemicalMorphMoveSpec,
    AlchemicalShiftMoveSpec,
    BaseMoveSpec,
    GMCMoveSpec,
    HMCMoveSpec,
    MoveSpec,
    MoveType,
    RandomWalkMoveSpec,
    ShearMoveSpec,
    SingleAtomMoveSpec,
    SingleAtomSweepMoveSpec,
    SingleAtomSwapMoveSpec,
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
    "AlchemicalShiftMoveSpec",
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
    "GaussianMixtureBackendSpec",
    "HMCMoveSpec",
    "HarmonicBackendSpec",
    "InitSpec",
    "InitialWalkSpec",
    "IterationTerminationSpec",
    "LJBackendSpec",
    "MACEBackendSpec",
    "MoveSpec",
    "MoveType",
    "NPTEnsembleSpec",
    "NVTEnsembleSpec",
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
    "SingleAtomSwapMoveSpec",
    "StretchMoveSpec",
    "TemperatureTerminationSpec",
    "TerminationSpec",
    "VolumeMoveSpec",
]
