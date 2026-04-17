"""Pydantic v2 schemas for jaxrens YAML configuration."""

from jaxrens.cli.schema.adaptation import (
    AdaptationConfig,
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
from jaxrens.cli.schema.cell import CellConfig
from jaxrens.cli.schema.ensemble import (
    BaseEnsembleSpec,
    EnsembleSpec,
    NPTEnsembleSpec,
    NVTEnsembleSpec,
)
from jaxrens.cli.schema.init import InitConfig, InitialWalkConfig
from jaxrens.cli.schema.moves import (
    AlchemicalMorphMoveSpec,
    AlchemicalShiftMoveSpec,
    BaseMoveSpec,
    GalileanMoveSpec,
    GmcMoveSpec,
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
from jaxrens.cli.schema.output import OutputSchema
from jaxrens.cli.schema.root import RootConfig
from jaxrens.cli.schema.run import RunSchema
from jaxrens.cli.schema.termination import (
    BaseTerminationSpec,
    EnergyTerminationSpec,
    IterationTerminationSpec,
    PriorMassTerminationSpec,
    TemperatureTerminationSpec,
    TerminationSpec,
)

__all__ = [
    "AdaptationConfig",
    "AdaptationPolicy",
    "AlchemicalMorphMoveSpec",
    "AlchemicalShiftMoveSpec",
    "BackendSpec",
    "BaseBackendSpec",
    "BaseEnsembleSpec",
    "BaseMoveSpec",
    "BaseTerminationSpec",
    "CellConfig",
    "DoubleWellBackendSpec",
    "EnergyTerminationSpec",
    "EnsembleSpec",
    "GalileanMoveSpec",
    "GaussianMixtureBackendSpec",
    "GmcMoveSpec",
    "HMCMoveSpec",
    "HarmonicBackendSpec",
    "InitConfig",
    "InitialWalkConfig",
    "IterationTerminationSpec",
    "LJBackendSpec",
    "MACEBackendSpec",
    "MoveSpec",
    "MoveType",
    "NPTEnsembleSpec",
    "NVTEnsembleSpec",
    "NeuralILBackendSpec",
    "OutputSchema",
    "PriorMassTerminationSpec",
    "RandomWalkMoveSpec",
    "ResolvedAdaptationPolicy",
    "RootConfig",
    "RunSchema",
    "ShearMoveSpec",
    "SingleAtomMoveSpec",
    "SingleAtomSweepMoveSpec",
    "SingleAtomSwapMoveSpec",
    "StretchMoveSpec",
    "TemperatureTerminationSpec",
    "TerminationSpec",
    "VolumeMoveSpec",
]
