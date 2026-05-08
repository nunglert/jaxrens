"""Pydantic schema for the [moves] section of a jaxrens YAML config.

Each concrete spec class carries exactly the fields that its kernel builder
accepts.  ``to_move_config()`` and ``to_descriptor()`` are the seam between
the CLI config layer and the library core — they replace the
``_MOVE_REGISTRY`` / ``_build_kernel_kwargs`` / ``_extra_state_fields``
side-channel that used to live in ``cli/run.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Callable, Literal, Union

import jax.numpy as jnp
from pydantic import BaseModel, ConfigDict, Field

from jaxrens.sampling.move_kernel import MoveKernel
from jaxrens.sampling.moves import alchemical, galilean, hmc, random_walk, single_atom, volume, shear, stretch
from jaxrens.state.config import MoveConfig

if TYPE_CHECKING:
    from jaxrens.cli.schema.cell import CellSpec

# ---------------------------------------------------------------------------
# MoveType literal — kept for backward compatibility with callers that do
# ``from jaxrens.cli.schema.moves import MoveType``.
# ---------------------------------------------------------------------------

MoveType = Literal[
    "random_walk",
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

    def _kernel_kwargs(
        self,
        n_atoms: int | None = None,
        cell_cfg: "CellSpec | None" = None,
    ) -> dict[str, Any]:
        """Return kernel keyword arguments.

        Simple move specs ignore ``n_atoms`` and ``cell_cfg``.  Cell-move
        specs (volume, shear, stretch) and sweep specs use them to populate
        ``n_atoms`` / cell-geometry bounds from the resolver-provided values
        rather than duplicating those fields on the spec.

        Args:
            n_atoms: Number of atoms, derived from the resolved initial
                positions.  ``None`` is accepted by specs that don't need it.
            cell_cfg: ``CellSpec`` carrying cell-geometry constraints.
                ``None`` is accepted by specs that don't need it.
        """
        return {}

    def _extra_state_fields(self) -> dict[str, tuple[type, Callable]]:
        return {}

    def _reject_reasons(self) -> frozenset[str]:
        """Return the set of reject-reason buckets this move can emit.

        Subclasses override this when their kernel can emit cell or prior
        rejection reasons (buckets 2 and 3 respectively).  The default is
        energy-only rejection (bucket 1), which covers all atom-displacement
        moves that never call check_cell_shape or sample from a prior.
        """
        return frozenset({"energy"})

    def to_descriptor(
        self,
        *,
        n_atoms: int | None = None,
        cell_cfg: "CellSpec | None" = None,
    ) -> MoveKernel:
        """Produce the ``MoveKernel`` for ``build_mwg``.

        Args:
            n_atoms: Number of atoms, derived from the resolved initial
                positions at resolver time.  Simple moves (random_walk,
                galilean, …) ignore this.  Cell-move specs (volume, shear,
                stretch) and single_atom_sweep use it to populate
                ``kernel_kwargs["n_atoms"]``.
            cell_cfg: ``CellSpec`` carrying cell-geometry constraints.
                Cell-move specs use it to populate ``max_volume_per_atom``,
                ``min_volume_per_atom``, ``min_aspect_ratio``, and
                ``flat_V_prior`` in ``kernel_kwargs``.  Simple moves ignore it.
        """
        return MoveKernel(
            name=self._effective_name(),
            build_kernel=self._build_kernel(),
            kernel_kwargs=self._kernel_kwargs(n_atoms=n_atoms, cell_cfg=cell_cfg),
            weight=self.weight,
            step_size=self.step_size,
            extra_state_fields=self._extra_state_fields(),
            reject_reasons=self._reject_reasons(),
        )


# ---------------------------------------------------------------------------
# Concrete specs
# ---------------------------------------------------------------------------

class RandomWalkMoveSpec(BaseMoveSpec):
    type: Literal["random_walk"] = "random_walk"

    def _build_kernel(self) -> Callable:
        return random_walk.build_kernel


class GMCMoveSpec(BaseMoveSpec):
    """Galilean Monte Carlo move.

    The legacy YAML key ``type: galilean`` is accepted via a pre-validator
    coercion in ``root.py::_coerce_move_dict`` and rewritten to ``type: gmc``
    at parse time.
    """

    type: Literal["gmc"] = "gmc"
    n_reflect: int = 5

    def _n_steps(self) -> int:
        return self.n_reflect

    def _build_kernel(self) -> Callable:
        return galilean.build_kernel

    def _kernel_kwargs(self, n_atoms: int | None = None, cell_cfg: "CellSpec | None" = None) -> dict[str, Any]:
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

    def _kernel_kwargs(self, n_atoms: int | None = None, cell_cfg: "CellSpec | None" = None) -> dict[str, Any]:
        return {"n_leapfrog": self.n_leapfrog}


class SingleAtomMoveSpec(BaseMoveSpec):
    type: Literal["single_atom"] = "single_atom"

    def _build_kernel(self) -> Callable:
        return single_atom.build_kernel


class SingleAtomSweepMoveSpec(BaseMoveSpec):
    type: Literal["single_atom_sweep"] = "single_atom_sweep"

    def _build_kernel(self) -> Callable:
        return single_atom.build_sweep_kernel

    def _kernel_kwargs(self, n_atoms: int | None = None, cell_cfg: "CellSpec | None" = None) -> dict[str, Any]:
        if n_atoms is None:
            raise ValueError(
                "SingleAtomSweepMoveSpec.to_descriptor() requires n_atoms to "
                "be provided by the resolver (derived from init positions). "
                "Call to_descriptor(n_atoms=...) with the atom count."
            )
        return {"n_atoms": n_atoms}


class SingleAtomSwapMoveSpec(BaseMoveSpec):
    type: Literal["single_atom_swap"] = "single_atom_swap"

    def _build_kernel(self) -> Callable:
        return single_atom.build_swap_kernel


class VolumeMoveSpec(BaseMoveSpec):
    type: Literal["volume"] = "volume"

    def _reject_reasons(self) -> frozenset[str]:
        return frozenset({"energy", "cell", "prior"})

    def _build_kernel(self) -> Callable:
        return volume.build_kernel

    def _kernel_kwargs(self, n_atoms: int | None = None, cell_cfg: "CellSpec | None" = None) -> dict[str, Any]:
        if n_atoms is None:
            raise ValueError(
                "VolumeMoveSpec.to_descriptor() requires n_atoms to be "
                "provided by the resolver (derived from init positions)."
            )
        if cell_cfg is None:
            raise ValueError(
                "VolumeMoveSpec.to_descriptor() requires cell_cfg to be "
                "provided by the resolver (from the [cell] config section)."
            )
        return {
            "n_atoms": n_atoms,
            "max_vol_per_atom": cell_cfg.max_volume_per_atom,
            "min_vol_per_atom": cell_cfg.min_volume_per_atom,
            "min_aspect": cell_cfg.min_aspect_ratio,
            "flat_v_prior": cell_cfg.flat_V_prior,
        }


class ShearMoveSpec(BaseMoveSpec):
    type: Literal["shear"] = "shear"

    def _reject_reasons(self) -> frozenset[str]:
        return frozenset({"energy", "cell"})

    def _build_kernel(self) -> Callable:
        return shear.build_kernel

    def _kernel_kwargs(self, n_atoms: int | None = None, cell_cfg: "CellSpec | None" = None) -> dict[str, Any]:
        if n_atoms is None:
            raise ValueError(
                "ShearMoveSpec.to_descriptor() requires n_atoms to be "
                "provided by the resolver (derived from init positions)."
            )
        if cell_cfg is None:
            raise ValueError(
                "ShearMoveSpec.to_descriptor() requires cell_cfg to be "
                "provided by the resolver (from the [cell] config section)."
            )
        return {
            "n_atoms": n_atoms,
            "max_vol_per_atom": cell_cfg.max_volume_per_atom,
            "min_vol_per_atom": cell_cfg.min_volume_per_atom,
            "min_aspect": cell_cfg.min_aspect_ratio,
        }


class StretchMoveSpec(BaseMoveSpec):
    type: Literal["stretch"] = "stretch"

    def _reject_reasons(self) -> frozenset[str]:
        return frozenset({"energy", "cell"})

    def _build_kernel(self) -> Callable:
        return stretch.build_kernel

    def _kernel_kwargs(self, n_atoms: int | None = None, cell_cfg: "CellSpec | None" = None) -> dict[str, Any]:
        if n_atoms is None:
            raise ValueError(
                "StretchMoveSpec.to_descriptor() requires n_atoms to be "
                "provided by the resolver (derived from init positions)."
            )
        if cell_cfg is None:
            raise ValueError(
                "StretchMoveSpec.to_descriptor() requires cell_cfg to be "
                "provided by the resolver (from the [cell] config section)."
            )
        return {
            "n_atoms": n_atoms,
            "max_vol_per_atom": cell_cfg.max_volume_per_atom,
            "min_vol_per_atom": cell_cfg.min_volume_per_atom,
            "min_aspect": cell_cfg.min_aspect_ratio,
        }


class AlchemicalMorphMoveSpec(BaseMoveSpec):
    type: Literal["alchemical_morph"] = "alchemical_morph"
    n_species: int
    # NOTE: n_species could in principle be derived from len(symbol_map) in
    # init_resolved, but that would require threading symbol_map through the
    # resolver to to_descriptor().  Since it is single-valued and small, keeping
    # it on the spec is a pragmatic trade-off; the inconsistency is flagged here.

    def _build_kernel(self) -> Callable:
        return alchemical.build_morph_kernel

    def _kernel_kwargs(self, n_atoms: int | None = None, cell_cfg: "CellSpec | None" = None) -> dict[str, Any]:
        return {"n_species": self.n_species}


class AlchemicalShiftMoveSpec(BaseMoveSpec):
    type: Literal["alchemical_shift"] = "alchemical_shift"

    def _build_kernel(self) -> Callable:
        return alchemical.build_shift_kernel

    # Inherits the no-op _kernel_kwargs from BaseMoveSpec.


# ---------------------------------------------------------------------------
# Discriminated union
# ---------------------------------------------------------------------------

MoveSpec = Annotated[
    Union[
        RandomWalkMoveSpec,
        GMCMoveSpec,
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
