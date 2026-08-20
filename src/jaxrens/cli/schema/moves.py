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
from pydantic import BaseModel, ConfigDict, Field, field_validator

from jaxrens.sampling.move_kernel import MoveKernel
from jaxrens.sampling.moves import (
    alchemical,
    galilean,
    hmc,
    random_walk,
    shear,
    single_atom,
    stretch,
    volume,
)
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
        symbol_map: dict[int, str] | None = None,
    ) -> dict[str, Any]:
        """Return kernel keyword arguments.

        Simple move specs ignore ``n_atoms``, ``cell_cfg`` and ``symbol_map``.
        Cell-move specs (volume, shear, stretch) and sweep specs use the first
        two to populate ``n_atoms`` / cell-geometry bounds from the
        resolver-provided values rather than duplicating those fields on the
        spec; species-scoped specs use the third to map element symbols to
        type codes.

        Args:
            n_atoms: Number of atoms, derived from the resolved initial
                positions.  ``None`` is accepted by specs that don't need it.
            cell_cfg: ``CellSpec`` carrying cell-geometry constraints.
                ``None`` is accepted by specs that don't need it.
            symbol_map: ``{type_code: element_symbol}`` for the resolved
                system.  ``None`` is accepted by specs that don't need it.
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

    def _mutates(self) -> frozenset[str]:
        """Return the state aspects this move writes (see jaxrens.constraints).

        Drives which configuration constraints gate this move.  The default
        is ``{"positions"}`` — atom-displacement moves.  Cell moves override
        to add ``"cell"`` (they co-transform atoms, so they keep
        ``"positions"`` too), and species-changing moves override to
        ``{"types"}``.
        """
        return frozenset({"positions"})

    def to_descriptor(
        self,
        *,
        n_atoms: int | None = None,
        cell_cfg: "CellSpec | None" = None,
        symbol_map: dict[int, str] | None = None,
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
            symbol_map: ``{type_code: element_symbol}`` for the resolved
                system, as carried by ``ResolvedInit``.  Only species-scoped
                specs (``gmc`` with ``species``) need it; all others ignore it.
        """
        return MoveKernel(
            name=self._effective_name(),
            build_kernel=self._build_kernel(),
            kernel_kwargs=self._kernel_kwargs(
                n_atoms=n_atoms, cell_cfg=cell_cfg, symbol_map=symbol_map
            ),
            weight=self.weight,
            step_size=self.step_size,
            extra_state_fields=self._extra_state_fields(),
            reject_reasons=self._reject_reasons(),
            mutates=self._mutates(),
        )


# ---------------------------------------------------------------------------
# Concrete specs
# ---------------------------------------------------------------------------


class RandomWalkMoveSpec(BaseMoveSpec):
    type: Literal["random_walk"] = "random_walk"

    def _build_kernel(self) -> Callable:
        return random_walk.build_kernel


class GMCMoveSpec(BaseMoveSpec):
    """Galilean Monte Carlo move, optionally scoped to one element sublattice.

    The legacy YAML key ``type: galilean`` is accepted via a pre-validator
    coercion in ``root.py::_coerce_move_dict`` and rewritten to ``type: gmc``
    at parse time.

    ``species`` restricts the move to the atoms of the named element(s) and
    holds the rest fixed.  Declaring one scoped move per element gives each
    sublattice an *independently adapted* step size, because the MWG sampler
    stores step sizes per move and the adaptation manager bisects each move
    separately.  That matters for systems where one sublattice melts well
    before the other: in a single joint move the step size is capped by
    whichever sublattice is stiffer.

    ::

        moves:
          - {type: gmc, species: Ge, step_size: 0.3, weight: 3}
          - {type: gmc, species: Si, step_size: 0.05, weight: 1}

    Two caveats worth knowing:

    * Step sizes are **not comparable across scopes**.  The direction is a
      unit vector over the moving subspace, so per-atom displacement scales
      as ``step_size / sqrt(3 * n_moving)`` — a minority sublattice takes
      larger per-atom steps at equal nominal step size.  Harmless (each is
      adapted on its own acceptance), but don't read the two numbers as
      being on the same scale.
    * Each scoped move costs a full energy+force call per reflection, so
      two scoped moves are 2x the evaluations of one joint move for the same
      ``n_reflect``.  Use ``weight`` to spend the budget where it pays.
    """

    type: Literal["gmc"] = "gmc"
    n_reflect: int = 5
    species: tuple[str, ...] | None = None

    @field_validator("species", mode="before")
    @classmethod
    def _wrap_bare_symbol(cls, v: Any) -> Any:
        """Accept ``species: Ge`` as shorthand for ``species: [Ge]``."""
        return (v,) if isinstance(v, str) else v

    def _n_steps(self) -> int:
        return self.n_reflect

    def _build_kernel(self) -> Callable:
        return galilean.build_kernel

    def _effective_name(self) -> str:
        """Auto-name scoped moves so they stay distinguishable downstream.

        Move names key the monitor's per-move columns, the adaptation
        diagnostics, and ``adaptation.resolve_for`` overrides — two moves both
        called ``"gmc"`` would collide in all three.
        """
        if self.name is not None:
            return self.name
        if self.species:
            return "gmc_" + "_".join(self.species)
        return self.type

    def _direction_field(self) -> str:
        """MCState field name holding this move's persistent direction.

        Unscoped moves keep the historical ``"direction"`` name (they all act
        on the same full subspace, so sharing is benign, and restarts and
        hand-built ``MoveKernel``s keep working).  Scoped moves each get their
        own field — ``build_mwg`` unions ``extra_state_fields`` by name, so a
        shared field would let the Ge move zero out the Si move's persistent
        direction on every call.
        """
        if not self.species:
            return "direction"
        return f"direction_{self._effective_name()}"

    def _species_codes(
        self, symbol_map: dict[int, str] | None
    ) -> tuple[int, ...] | None:
        """Resolve element symbols to the contiguous type codes used by types."""
        if not self.species:
            return None
        if symbol_map is None:
            raise ValueError(
                "GMCMoveSpec with species=... requires symbol_map to be "
                "provided by the resolver (it maps element symbols to the "
                "type codes stored in WalkerState.types). Build descriptors "
                "via ResolvedConfig.move_descriptors rather than "
                "setup_mwg()/MoveConfig, which carry no species information."
            )
        code_of = {sym: code for code, sym in symbol_map.items()}
        unknown = [s for s in self.species if s not in code_of]
        if unknown:
            raise ValueError(
                f"gmc species {unknown} not present in the system "
                f"(symbols: {sorted(code_of)}). A species-scoped move whose "
                f"element is absent would silently become a no-op that always "
                f"accepts."
            )
        return tuple(code_of[s] for s in self.species)

    def _kernel_kwargs(
        self,
        n_atoms: int | None = None,
        cell_cfg: "CellSpec | None" = None,
        symbol_map: dict[int, str] | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"n_reflect": self.n_reflect}
        # Only emit the scoping kwargs when the move is actually scoped: the
        # kernel defaults are exactly ``species=None`` /
        # ``direction_field="direction"``, so an unscoped spec keeps producing
        # the historical single-key kernel_kwargs and stays comparable with
        # descriptors built before species scoping existed.
        codes = self._species_codes(symbol_map)
        if codes is not None:
            kwargs["species"] = codes
            kwargs["direction_field"] = self._direction_field()
        return kwargs

    def _extra_state_fields(self) -> dict[str, tuple[type, Callable]]:
        return {
            self._direction_field(): (
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

    def _kernel_kwargs(
        self,
        n_atoms: int | None = None,
        cell_cfg: "CellSpec | None" = None,
        symbol_map: dict[int, str] | None = None,
    ) -> dict[str, Any]:
        return {"n_leapfrog": self.n_leapfrog}


class SingleAtomMoveSpec(BaseMoveSpec):
    type: Literal["single_atom"] = "single_atom"

    def _build_kernel(self) -> Callable:
        return single_atom.build_kernel


class SingleAtomSweepMoveSpec(BaseMoveSpec):
    type: Literal["single_atom_sweep"] = "single_atom_sweep"

    def _build_kernel(self) -> Callable:
        return single_atom.build_sweep_kernel

    def _kernel_kwargs(
        self,
        n_atoms: int | None = None,
        cell_cfg: "CellSpec | None" = None,
        symbol_map: dict[int, str] | None = None,
    ) -> dict[str, Any]:
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

    def _mutates(self) -> frozenset[str]:
        return frozenset({"types"})


class VolumeMoveSpec(BaseMoveSpec):
    type: Literal["volume"] = "volume"

    def _reject_reasons(self) -> frozenset[str]:
        return frozenset({"energy", "cell", "prior"})

    def _mutates(self) -> frozenset[str]:
        return frozenset({"positions", "cell"})

    def _build_kernel(self) -> Callable:
        return volume.build_kernel

    def _kernel_kwargs(
        self,
        n_atoms: int | None = None,
        cell_cfg: "CellSpec | None" = None,
        symbol_map: dict[int, str] | None = None,
    ) -> dict[str, Any]:
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

    def _mutates(self) -> frozenset[str]:
        return frozenset({"positions", "cell"})

    def _build_kernel(self) -> Callable:
        return shear.build_kernel

    def _kernel_kwargs(
        self,
        n_atoms: int | None = None,
        cell_cfg: "CellSpec | None" = None,
        symbol_map: dict[int, str] | None = None,
    ) -> dict[str, Any]:
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

    def _mutates(self) -> frozenset[str]:
        return frozenset({"positions", "cell"})

    def _build_kernel(self) -> Callable:
        return stretch.build_kernel

    def _kernel_kwargs(
        self,
        n_atoms: int | None = None,
        cell_cfg: "CellSpec | None" = None,
        symbol_map: dict[int, str] | None = None,
    ) -> dict[str, Any]:
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

    def _kernel_kwargs(
        self,
        n_atoms: int | None = None,
        cell_cfg: "CellSpec | None" = None,
        symbol_map: dict[int, str] | None = None,
    ) -> dict[str, Any]:
        return {"n_species": self.n_species}

    def _mutates(self) -> frozenset[str]:
        return frozenset({"types"})


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
