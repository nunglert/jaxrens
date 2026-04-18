"""Pydantic schema for the [moves] section of a jaxrens YAML config.

Each concrete spec class carries exactly the fields that its kernel builder
accepts.  ``to_move_config()`` and ``to_descriptor()`` are the seam between
the CLI config layer and the library core — they replace the
``_MOVE_REGISTRY`` / ``_build_kernel_kwargs`` / ``_extra_state_fields``
side-channel that used to live in ``cli/run.py``.
"""

from __future__ import annotations

from typing import Annotated, Any, Callable, Literal, Union

import jax.numpy as jnp
from pydantic import BaseModel, ConfigDict, Field

from jaxrens.sampling.move_kernel import MoveKernel
from jaxrens.sampling.moves import alchemical, galilean, hmc, random_walk, single_atom, volume, shear, stretch
from jaxrens.state.config import MoveConfig

# ---------------------------------------------------------------------------
# MoveType literal — kept for backward compatibility with callers that do
# ``from jaxrens.cli.schema.moves import MoveType``.
# ---------------------------------------------------------------------------

MoveType = Literal[
    "random_walk",
    "galilean",
    "gmc",
    "hmc",
    "single_atom",
    "single_atom_sweep",
    "single_atom_swap",
    "volume",
    "shear",
    "stretch",
    "alchemical_morph",
    "alchemical_shift",
]


# ---------------------------------------------------------------------------
# Base spec
# ---------------------------------------------------------------------------

class BaseMoveSpec(BaseModel):
    """Fields shared by every move type."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    step_size: float = 0.1
    weight: float = 1.0
    adaptation_warmup: int = 100
    target_acceptance: float = 0.5
    name: str | None = None

    @property
    def move_type(self) -> str:
        """Backward-compatible alias for the ``type`` discriminator field."""
        return self.type  # type: ignore[attr-defined]

    def _effective_name(self) -> str:
        return self.name if self.name is not None else self.type  # type: ignore[attr-defined]

    def to_move_config(self) -> MoveConfig:
        """Produce the library ``MoveConfig`` dataclass."""
        return MoveConfig(
            move_type=self.type,  # type: ignore[attr-defined]
            step_size=self.step_size,
            n_steps=self._n_steps(),
            weight=self.weight,
            adaptation_warmup=self.adaptation_warmup,
            target_acceptance=self.target_acceptance,
        )

    def _n_steps(self) -> int:
        """Override in subclasses that carry a steps-like field."""
        return 10

    def _build_kernel(self) -> Callable:
        raise NotImplementedError

    def _kernel_kwargs(self) -> dict[str, Any]:
        return {}

    def _extra_state_fields(self) -> dict[str, tuple[type, Callable]]:
        return {}

    def to_descriptor(self) -> MoveKernel:
        """Produce the ``MoveKernel`` for ``build_mwg``."""
        return MoveKernel(
            name=self._effective_name(),
            build_kernel=self._build_kernel(),
            kernel_kwargs=self._kernel_kwargs(),
            weight=self.weight,
            step_size=self.step_size,
            extra_state_fields=self._extra_state_fields(),
        )


# ---------------------------------------------------------------------------
# Concrete specs
# ---------------------------------------------------------------------------

class RandomWalkMoveSpec(BaseMoveSpec):
    type: Literal["random_walk"] = "random_walk"

    def _build_kernel(self) -> Callable:
        return random_walk.build_kernel


class GalileanMoveSpec(BaseMoveSpec):
    type: Literal["galilean"] = "galilean"
    n_reflect: int = 5

    def _n_steps(self) -> int:
        return self.n_reflect

    def _build_kernel(self) -> Callable:
        return galilean.build_kernel

    def _kernel_kwargs(self) -> dict[str, Any]:
        return {"n_reflect": self.n_reflect}

    def _extra_state_fields(self) -> dict[str, tuple[type, Callable]]:
        return {
            "direction": (
                jnp.ndarray,
                lambda positions, types: jnp.zeros_like(positions),
            ),
        }


class GmcMoveSpec(BaseMoveSpec):
    """Alias for Galilean MC — same kernel, different registry key."""

    type: Literal["gmc"] = "gmc"
    n_reflect: int = 5

    def _n_steps(self) -> int:
        return self.n_reflect

    def _build_kernel(self) -> Callable:
        return galilean.build_kernel

    def _kernel_kwargs(self) -> dict[str, Any]:
        return {"n_reflect": self.n_reflect}

    def _extra_state_fields(self) -> dict[str, tuple[type, Callable]]:
        return {
            "direction": (
                jnp.ndarray,
                lambda positions, types: jnp.zeros_like(positions),
            ),
        }


class HMCMoveSpec(BaseMoveSpec):
    type: Literal["hmc"] = "hmc"
    n_leapfrog: int = 10

    def _n_steps(self) -> int:
        return self.n_leapfrog

    def _build_kernel(self) -> Callable:
        return hmc.build_kernel

    def _kernel_kwargs(self) -> dict[str, Any]:
        return {"n_leapfrog": self.n_leapfrog}


class SingleAtomMoveSpec(BaseMoveSpec):
    type: Literal["single_atom"] = "single_atom"

    def _build_kernel(self) -> Callable:
        return single_atom.build_kernel


class SingleAtomSweepMoveSpec(BaseMoveSpec):
    type: Literal["single_atom_sweep"] = "single_atom_sweep"
    n_atoms: int

    def _build_kernel(self) -> Callable:
        return single_atom.build_sweep_kernel

    def _kernel_kwargs(self) -> dict[str, Any]:
        return {"n_atoms": self.n_atoms}


class SingleAtomSwapMoveSpec(BaseMoveSpec):
    type: Literal["single_atom_swap"] = "single_atom_swap"

    def _build_kernel(self) -> Callable:
        return single_atom.build_swap_kernel


class VolumeMoveSpec(BaseMoveSpec):
    type: Literal["volume"] = "volume"
    n_atoms: int
    max_vol_per_atom: float = 100.0
    min_vol_per_atom: float = 1.0
    min_aspect: float = 0.5
    flat_v_prior: bool = False

    def _build_kernel(self) -> Callable:
        return volume.build_kernel

    def _kernel_kwargs(self) -> dict[str, Any]:
        return {
            "n_atoms": self.n_atoms,
            "max_vol_per_atom": self.max_vol_per_atom,
            "min_vol_per_atom": self.min_vol_per_atom,
            "min_aspect": self.min_aspect,
            "flat_v_prior": self.flat_v_prior,
        }


class ShearMoveSpec(BaseMoveSpec):
    type: Literal["shear"] = "shear"
    n_atoms: int
    max_vol_per_atom: float = 100.0
    min_vol_per_atom: float = 1.0
    min_aspect: float = 0.5

    def _build_kernel(self) -> Callable:
        return shear.build_kernel

    def _kernel_kwargs(self) -> dict[str, Any]:
        return {
            "n_atoms": self.n_atoms,
            "max_vol_per_atom": self.max_vol_per_atom,
            "min_vol_per_atom": self.min_vol_per_atom,
            "min_aspect": self.min_aspect,
        }


class StretchMoveSpec(BaseMoveSpec):
    type: Literal["stretch"] = "stretch"
    n_atoms: int
    max_vol_per_atom: float = 100.0
    min_vol_per_atom: float = 1.0
    min_aspect: float = 0.5

    def _build_kernel(self) -> Callable:
        return stretch.build_kernel

    def _kernel_kwargs(self) -> dict[str, Any]:
        return {
            "n_atoms": self.n_atoms,
            "max_vol_per_atom": self.max_vol_per_atom,
            "min_vol_per_atom": self.min_vol_per_atom,
            "min_aspect": self.min_aspect,
        }


class AlchemicalMorphMoveSpec(BaseMoveSpec):
    type: Literal["alchemical_morph"] = "alchemical_morph"
    n_species: int

    def _build_kernel(self) -> Callable:
        return alchemical.build_morph_kernel

    def _kernel_kwargs(self) -> dict[str, Any]:
        return {"n_species": self.n_species}


class AlchemicalShiftMoveSpec(BaseMoveSpec):
    type: Literal["alchemical_shift"] = "alchemical_shift"

    def _build_kernel(self) -> Callable:
        return alchemical.build_shift_kernel


# ---------------------------------------------------------------------------
# Discriminated union
# ---------------------------------------------------------------------------

MoveSpec = Annotated[
    Union[
        RandomWalkMoveSpec,
        GalileanMoveSpec,
        GmcMoveSpec,
        HMCMoveSpec,
        SingleAtomMoveSpec,
        SingleAtomSweepMoveSpec,
        SingleAtomSwapMoveSpec,
        VolumeMoveSpec,
        ShearMoveSpec,
        StretchMoveSpec,
        AlchemicalMorphMoveSpec,
        AlchemicalShiftMoveSpec,
    ],
    Field(discriminator="type"),
]
